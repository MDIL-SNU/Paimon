import asyncio
from datetime import datetime
from typing import Any

from pydantic import ValidationError
from llama_index.core.workflow import (
    step,
    Workflow,
    Context,
    WorkflowRuntimeError,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer, BaseMemory
from llama_index.core.tools import FunctionTool, AsyncBaseTool
from llama_index.core.agent.workflow.base_agent import DEFAULT_MAX_ITERATIONS

import paimon.world as world
import paimon.models as mdl
import paimon.agent.registry as agent_registry
import paimon.workflow.events as evt
from paimon import cfg as paimon_config
from paimon.interface.cli import CommandLineInterface
from paimon.knowledge.library import get_knowledge
from paimon.world.environment import Environment
from paimon.world.slurm import get_slurm_tracker
from paimon.world.remote_python import pycall_summarize_structure
from paimon.world.expert_tools import (
    retrieve_expert_knowledge_tool,
    retrieve_umlip_knowledge_tool,
    extract_paper_methodology_tool,
)
from paimon.world.web_search_tools import web_search_tool
from paimon.util.log import debug
from paimon.util.tool_factory import create_model_tool
from paimon.token_sum import TokenSum
from paimon.llm import get_llm, run_agent_pipeline
import paimon.audit as audit
from paimon.workflow._plan_validation import verify_and_process_plan
from paimon.workflow.prompt_helpers import (
    generate_initial_user_prompt,
    generate_agent_prompt,
    generate_subtask_report,
)
from paimon.workflow.subtask_agent import get_single_agent_workflow_with_context


async def _poll_slurm_job(
    ctx: Context,
    agent_state: mdl.SubtaskAgentState,
    poll_interval_sec: float = 1,
) -> str:
    """Poll SLURM job status until completion or timeout."""
    env_id = agent_state.env_id
    job_id = agent_state.slurm_job_id
    wait_min = agent_state.max_wait_minutes
    assert env_id and job_id and wait_min, f"{env_id}, {job_id}, {wait_min}"

    slurm = await get_slurm_tracker()
    start_time = datetime.now()
    return_str = ""
    while True:
        job = slurm.job_states[env_id][job_id]
        state = job["state"]

        if state not in ["RUNNING", "PENDING"]:
            return_str = (
                f"Job {job_id} is no longer running or pending (state={state})."
            )
            agent_state.slurm_job_id = None
            await ctx.store.set("agent_state", agent_state)
            break

        elapsed_min = (datetime.now() - start_time).total_seconds() / 60.0
        if elapsed_min >= wait_min:
            return_str = "The waiting period has ended."
            break

        await asyncio.sleep(poll_interval_sec)
    return return_str


# Folder name convention for upload user-provided files to working directory
EXTERNAL_FILES = "external_files"

plan_outline_tool = create_model_tool(mdl.PlanOutline)
plan_outline_tool.metadata.name = "outline_plan"

subtask_tool = create_model_tool(mdl.Subtask)
subtask_tool.metadata.name = "create_subtask"

abort_task_tool_default = create_model_tool(mdl.AbortTask)
abort_task_tool_default.metadata.name = "abort_task"

# complete task tool when there is no user rquested value
complete_task_tool_default = create_model_tool(
    mdl.CompleteTask.get_model_with_output_values([])
)
complete_task_tool_default.metadata.name = "complete_task"


class PaimonDynamicPlanWorkflow(Workflow):
    """Plan and assign agent for each subtasks until plan accomplished.

    Following operations are NOT this workflow's reponsibility.
    - Environment must be created beforehand
    - If start event has files, they must be already exist in {env}/external_files.
    - File upload is not its responsibility
    - Save intermediate states

    ---
    IMPORATNT
    TODO: subtask fail => call ask agent to re-start task instead of discarding
    this case fails
    """

    def __init__(
        self,
        llm: LLM | str = "base_reasoning",
        agent_names: list[str] | None = None,
        with_critic: bool = paimon_config.critic_config.use,
        max_agent_iterations: int = paimon_config.max_agent_iterations,
        with_web_search: bool = paimon_config.web_search_config.planner,
        with_web_search_executors: bool = paimon_config.web_search_config.executor,
        num_retry_on_subtask_fail: int = 0,
        forbid_request_user_input_tool: bool = False,
        request_user_input_tool_budget: int = 10**18,
        stay_alive_after_completion: bool = False,
        timeout: float | None = None,
        verbose: bool = False,
        tool_overrides: dict | None = None,
        max_action: int = 50,
        **kwargs,
    ) -> None:
        """Initialize workflow. The parameters of init should be runtime agnostic.
        Runtime states such as 'task' or 'plan', are managed by context.

        Parameters
        ----------
        llm
            the root, supervisor & planning LLM
        agent_names
            list of agent names visible to supervisor llm. None uses all registered
        with_critic
            attach critic gate to executor agents
        max_agent_iterations
            max tool-call iterations per executor agent
        with_web_search
            enable web search tool for the planner
        with_web_search_executors
            enable web search tool for executor agents
        num_retry_on_subtask_fail
            number of retries when a subtask fails
        forbid_request_user_input_tool
            disallow planner from requesting user input
        request_user_input_tool_budget
            max number of user input requests. None means unlimited
        stay_alive_after_completion
            keep the workflow alive after the plan completes
        timeout
            workflow-level timeout in seconds
        verbose
            enable verbose logging
        tool_overrides
            override default tool set for executor agents
        max_action
            maximum number of planner actions before stopping
        """
        super().__init__(timeout=timeout, verbose=verbose, **kwargs)
        if isinstance(llm, str):
            llm = get_llm(llm)
        self.llm = llm
        self.system_prompt = get_knowledge(
            "planner/system_prompt_v2", with_debug_preamble=paimon_config.debug_preamble
        )
        self.forbid_request_user_input_tool = forbid_request_user_input_tool
        self.request_user_input_tool_budget = request_user_input_tool_budget
        self.stay_alive_after_completion = stay_alive_after_completion

        self.agent_names = agent_names or agent_registry.list_agents()  # all agent
        self.max_agent_iterations = max_agent_iterations
        self.num_retry_on_subtask_fail = num_retry_on_subtask_fail
        self.with_web_search = with_web_search
        self.with_web_search_executors = with_web_search_executors

        self.with_critic = with_critic
        debug(f"[init] critic: {self.with_critic}")

        agent_desc_str_list = []
        for agent_name in self.agent_names:
            dct = agent_registry.describe_agent(agent_name)
            agent_desc_str_list.append(f"- {agent_name}: {dct['description']}")
        self.agent_str = "\n\n".join(agent_desc_str_list)

        self.max_action: int = max_action
        self.tool_overrides = tool_overrides
        debug("workflow __init__")

    async def _decide_tool_call(
        self, ctx: Context[mdl.PlanState], tools: list
    ) -> tuple[AsyncBaseTool, str, dict[str, Any], str]:
        """Call planner agent with tools routine."""

        state: mdl.PlanState = await ctx.store.get_state()
        tools_by_name = {tool.metadata.name: tool for tool in tools}
        llm_metadata = {
            "role": "planner",
            "env_id": state.env_id,
        }

        response, tool_calls, tok_used = await run_agent_pipeline(
            llm=self.llm,
            tools=tools,
            chat_history=state.chat_history,
            allow_parallel_tool_calls=False,
            tool_required=True,
            agent_name="planner",
            metadata=llm_metadata,
        )
        assert state.token_sum
        state.token_sum += tok_used
        state.chat_history.append(response.message)
        state.action_count += 1

        assert len(tool_calls) == 1
        tool_call = tool_calls[0]
        tool_name = tool_call.tool_name
        tool = tools_by_name[tool_name]
        tool_input = tool_call.tool_kwargs
        if tool.requires_context and tool.ctx_param_name:
            tool_input[tool.ctx_param_name] = ctx

        await ctx.store.set_state(state)
        ctx.write_event_to_stream(
            evt.ToolCall(
                tool_name=tool_name,
                tool_kwargs=tool_call.tool_kwargs,
                tool_id=tool_call.tool_id,
            )
        )
        return tool, tool_name, tool_input, tool_call.tool_id

    async def _append_tool_call_result(
        self,
        ctx: Context[mdl.PlanState],
        tool_name: str,
        tool_msg: str,  # tool output
        tool_id: str,
        is_error: bool = False,
    ) -> None:
        """Update chat_history with tool_msg and stream tool result event"""
        state: mdl.PlanState = await ctx.store.get_state()
        state.chat_history.append(
            ChatMessage(
                role="tool",
                content=tool_msg,
                additional_kwargs={"tool_call_id": tool_id},
            )
        )
        await ctx.store.set_state(state)
        ctx.write_event_to_stream(
            evt.SimpleToolCallResultEvent(
                tool_name=tool_name,
                tool_output=tool_msg,
                tool_id=tool_id,
                is_error=is_error,
            )
        )

    @step
    async def initialize_task(
        self, ctx: Context[mdl.PlanState], ev: evt.PaimonStartEvent
    ) -> evt.StartPlanningWithRetrieval | evt.StopEvent:
        # assume fresh start
        # TODO: logic for restart from an existing env
        state: mdl.PlanState = await ctx.store.get_state()

        state.task = ev.user_msg  # Initial user message
        state.task_name = ev.session_name
        state.token_sum = TokenSum([])
        state.user_requested_values = ev.user_requested_values
        state.remaining_request_user_input = self.request_user_input_tool_budget

        if not world.is_id_already_present(ev.env_id):
            raise ValueError(f"provided id does not exist: {ev.env_id}")

        if ev.expert_knowledge and not get_knowledge(name=ev.expert_knowledge):
            raise ValueError(
                f"Knowledge ({ev.expert_knowledge}) is assigned but not found"
            )

        # make sure env_id is in register
        world.new_environment(id=ev.env_id)
        state.env_id = ev.env_id
        env = world.get_env(ev.env_id)

        # File upload is NOT this workflow's job.
        # We just make sure claimed files does exist
        for filename in ev.files or []:
            if not env.file_exists(filename, sub_wd=EXTERNAL_FILES):
                raise ValueError(
                    f"{filename} does not exist in the external_file folder"
                )
        # We don't need to save files as variable since they can be listed by env

        llm_dump = self.llm.model_dump(mode="json")
        llm_dump.pop("api_key", None)
        state.dumps["planner_llm"] = llm_dump

        await ctx.store.set_state(state)
        ret_ev = evt.StartPlanningWithRetrieval(
            task=state.task,
            files=ev.files,
            expert_knowledge=ev.expert_knowledge,
            user_requested_values=state.user_requested_values,
        )

        ctx.write_event_to_stream(ret_ev)
        return ret_ev

    @step
    async def receive_human_response(
        self,
        ctx: Context[mdl.PlanState],
        ev: evt.HumanResponsedWithStepEvent,
    ) -> evt.StartPlanningWithRetrievalLoop | evt.ResumeCheckAndStart:
        state: mdl.PlanState = await ctx.store.get_state()
        chat_history = state.chat_history

        env = world.get_env(state.env_id)
        user_msg = f"<user_message>{ev.response}</user_message>"

        if ev.files:
            user_msg += "\n<system>The user uploaded files in this message.</system>"
            user_msg += "\n<user_uploaded_files>"
            missing_files = []
            for filename in ev.files:
                if not env.file_exists(filename, EXTERNAL_FILES):
                    debug(f"{filename} does not exist in the external_file folder!")
                    missing_files.append(filename)
                else:
                    user_msg += f"\n- {filename}"
            user_msg += "\n</user_uploaded_files>"

            if missing_files:
                user_msg += (
                    f"\n<system>Following files are not found: {missing_files}."
                )
                user_msg += " Ask user to retry if needed</system>"

            user_msg += "\n<system>Below is the current snapshot of the external_files directory after this upload.</system>"
            external_files_ls = env.list_working_directory("external_files")
            user_msg += f"\n<external_files>{external_files_ls}\n</external_files>"

        chat_history.append(ChatMessage(role="user", content=user_msg))
        await ctx.store.set_state(state)
        if ev.step == "plan_with_retrieval_loop":
            return evt.StartPlanningWithRetrievalLoop()
        elif ev.step == "check_and_start":
            return evt.ResumeCheckAndStart()
        else:
            raise RuntimeError(f"Unknow step: {ev.step}")

    @step
    async def plan_with_retrieval(
        self, ctx: Context[mdl.PlanState], ev: evt.StartPlanningWithRetrieval
    ) -> evt.StartPlanningWithRetrievalLoop:
        """Planning step that uses retrieval-based expert knowledge.

        Prepare plan_with_retrieval loop. (build initial context)
        """

        debug("[plan with retrieval] Enter plan_with_retrieval stage")
        state: mdl.PlanState = await ctx.store.get_state()
        env = world.get_env(state.env_id)

        plan_sys_txt = self.system_prompt.replace(
            "{{agent_shorts}}",
            agent_registry.short_agent_descriptions(self.agent_names),
        ).replace("{{umlip}}", get_knowledge("forcefield/umlip"))

        plan_user_txt = generate_initial_user_prompt(
            env,
            user_query=ev.task,
            user_requested_values=state.user_requested_values,
            expert_knowledge=ev.expert_knowledge,
        )

        chat_history: list[ChatMessage] = [
            ChatMessage(role="system", content=plan_sys_txt),
            ChatMessage(role="user", content=plan_user_txt),
        ]
        state.chat_history = chat_history
        await ctx.store.set_state(state)
        return evt.StartPlanningWithRetrievalLoop()

    @step
    async def plan_with_retrieval_loop(
        self, ctx: Context[mdl.PlanState], ev: evt.StartPlanningWithRetrievalLoop
    ) -> evt.InputRequiredWithStepEvent | evt.StartTasks:
        state: mdl.PlanState = await ctx.store.get_state()

        tools = [
            retrieve_expert_knowledge_tool,
            retrieve_umlip_knowledge_tool,
            extract_paper_methodology_tool,
            plan_outline_tool,
            self._make_request_user_input_tool(),
        ]
        if self.with_web_search:
            tools.append(web_search_tool)

        # Escaped if PlanOutline is created successfully.
        while True:
            await ctx.store.set_state(state)
            tool, tool_name, tool_input, tool_id = await self._decide_tool_call(
                ctx, tools
            )
            state: mdl.PlanState = await ctx.store.get_state()

            ev_to_return = None
            tool_msg = None
            is_error = False
            break_flag = False

            if tool_name == "request_user_input":
                debug("[plan with retrieval] request_user_input tool called")
                try:
                    message = tool_input.get("message")
                    tool_msg = (await tool.acall(**tool_input)).content
                except ValueError as e:
                    is_error = True
                    tool_msg = str(e)
                assert isinstance(message, str)
                if not is_error:
                    ev_to_return = evt.InputRequiredWithStepEvent(
                        prefix=message, step="plan_with_retrieval_loop"
                    )

            elif tool_name in (
                "retrieve_expert_knowledge",
                "retrieve_umlip_knowledge",
                "extract_paper_methodology",
                "web_search",
            ):
                tool_output = await tool.acall(**tool_input)
                debug(f"[plan_with_retrieval] {tool_name}")
                tool_msg = tool_output.content

            elif tool_name == "outline_plan":
                # Try to create plan outline (gated by retrieval count)
                try:
                    plan = mdl.PlanOutline(**tool_input)
                    verify_and_process_plan(
                        plan, allowed_agent_names=self.agent_names
                    )
                    # Plan outline accepted - add long agent descriptions
                    used_agents = list(
                        set([subtask_stub.agent for subtask_stub in plan.outline])
                    )
                    outline_tool_output_template = get_knowledge(
                        "planner/outline_tool_output"
                    )
                    tool_msg = outline_tool_output_template.format(
                        agent_longs=agent_registry.long_agent_descriptions(
                            used_agents
                        )
                    )
                    state.current_plan = plan
                    break_flag = True

                except ValidationError as e:
                    debug(f"[plan_with_retrieval] plan stage error: {e}")
                    tool_msg = "Tool validation failed. Check the format."
                    is_error = True
                except ValueError as e:
                    debug(f"[plan_with_retrieval] plan outline failed: {e}")
                    tool_msg = str(e)
                    is_error = True

            else:
                assert False, "logic flaw"
            await ctx.store.set_state(state)
            await self._append_tool_call_result(
                ctx, tool_name, tool_msg, tool_id, is_error=is_error
            )

            if ev_to_return:
                await ctx.store.set_state(state)
                # this event is automatically written to the stream when return
                if not isinstance(ev_to_return, evt.InputRequiredEvent):
                    ctx.write_event_to_stream(ev_to_return)
                return ev_to_return
            if break_flag:
                break
        # /while True

        assert state.current_plan and state.env_id and state.task_name, (
            "[plan] logic flaw"
        )
        await ctx.store.set_state(state)
        ret_ev = evt.StartTasks(
            plan=state.current_plan, env_id=state.env_id, task_name=state.task_name
        )
        ctx.write_event_to_stream(ret_ev)
        return ret_ev

    def _make_request_user_input_tool(self) -> FunctionTool:
        # The tool is special. It returns 1) event to return 2) plain tool msg
        # If event to return is None, it means tool call is failed
        async def request_user_input(message: str) -> str:
            """Ask the user a clarifying message and wait for response."""

            if self.forbid_request_user_input_tool:
                raise ValueError("""\
request_user_input is disabled by system configuration. Do not attempt to call this tool again; proceed without user interaction.
 """)  # noqa: E501
            else:
                tool_msg = "Your query is sent to the user."
            return tool_msg

        return FunctionTool.from_defaults(
            name="request_user_input",
            description="""\
Use this tool to request explicit user input when the task cannot be safely or correctly progressed without human judgment, clarification, validation, or decision-making.
This tool should be used when assumptions must be confirmed, interpretations require alignment, alternative directions need to be discussed, or responsibility for the next step must be transferred to the user.
Do not use this tool for routine updates or informational messages.
""",  # noqa: E501
            async_fn=request_user_input,
        )

    def _make_discard_subtasks_tool(
        self,
        ev: evt.ResumeCheckAndStart
        | evt.StartTasks
        | evt.SubtaskSuccess
        | evt.SubtaskFail,
    ) -> FunctionTool:
        """Create tool for discarding subtasks from a given point."""

        async def discard_subtasks_from(
            ctx: Context[mdl.PlanState], subtask_name: str
        ) -> str:
            debug(
                f"[check_and_start] discard_subtasks_from tool called {subtask_name}"
            )
            state: mdl.PlanState = await ctx.store.get_state()
            completed_subtasks = state.completed_subtasks
            env = world.get_env(state.env_id)

            if isinstance(ev, evt.SubtaskFail) and ev.subtask.name == subtask_name:
                # Failed subtask will be discarded anyway. So this call is pointless
                return f"Discarded subtasks: {subtask_name}"

            if subtask_name not in completed_subtasks:
                raise ValueError(
                    f"No such subtask ({subtask_name}) in completed list"
                )
            discard_from_subtask_name = subtask_name

            flag = False
            discarded_subtask_names = []
            discard_wds = []
            for name in list(completed_subtasks):
                if name == discard_from_subtask_name:
                    flag = True
                if flag:
                    discarded_subtask_names.append(name)
                    discard_wds.append(completed_subtasks.pop(name).sub_wd)

            discarded_subtask_names_str = "\n".join(discarded_subtask_names)
            for discard_wd in discard_wds:
                debug(f"{discard_wd} discarded and moved")
                await env.discard_sub_wd(discard_wd, prefix="discarded")

            state.completed_subtasks = completed_subtasks
            await ctx.store.set_state(state)

            return f"Discarded subtasks: {discarded_subtask_names_str}"

        return FunctionTool.from_defaults(
            name="discard_subtasks_after",
            description=(
                "Discard completed subtasks to restart the plan from a "
                "specific step. You can use this tool to discard multiple "
                "subtasks that have failed, or even those marked as "
                "successful, in order to restart the simulation from that "
                "subtask."
            ),
            async_fn=discard_subtasks_from,
        )

    def _make_consult_agent_tool(
        self,
        ev: evt.ResumeCheckAndStart
        | evt.StartTasks
        | evt.SubtaskSuccess
        | evt.SubtaskFail,
    ) -> FunctionTool:
        """Create tool for asking questions to agents of completed subtasks."""

        async def consult_agent(
            ctx: Context[mdl.PlanState],
            subtask_name_of_the_agent: str,
            analysis_request: str,
        ) -> str:
            state: mdl.PlanState = await ctx.store.get_state()
            completed_subtasks = state.completed_subtasks
            target = subtask_name_of_the_agent
            debug(f"[check_and_start] consult_agent:\n {target}//{analysis_request}")

            if isinstance(ev, evt.StartTasks):
                raise ValueError(
                    "If no subtasks have been created yet, this tool must not be "
                    "used. If you are attempting to use it immediately after "
                    "producing the outline plan, you must first create a subtask "
                    "by calling the create_subtask tool."
                )

            target = subtask_name_of_the_agent
            if isinstance(ev, evt.SubtaskFail) and target == ev.subtask.name:
                subtask = ev.subtask
            elif target in completed_subtasks:
                subtask = completed_subtasks[target]
            else:
                raise ValueError("No such subtask name exists")

            assert subtask.agent_ctx is not None
            assert subtask.memory is not None
            assert subtask.agent_wf is not None

            agent = subtask.agent_wf.agents[subtask.agent]  # type: ignore

            tools_wo_ends = []
            for tool in agent.tools:
                if tool.metadata.name in ("complete_task", "abort_task"):
                    continue
                tools_wo_ends.append(tool)
            _tools_orig = agent.tools
            agent.tools = tools_wo_ends

            agent.tool_required = False
            msg = f"""\
<planner>
{analysis_request}
</planner>
<system>
You are providing analysis about a completed execution.

Your role is to explain the state, outcome, or failure of the task based on existing artifacts and execution results.
At this point, you do not have access to complete_task or abort_task tools, as you are not performing the task.

Answer the planner’s request using natural language.

Do NOT continue execution or attempt to complete the task again.
Do NOT assume that you will perform further actions.

If the planner message requests or implies performing additional work, continuing execution, or modifying the task, explicitly state that you cannot perform further actions and that your role is limited to analysis and explanation.

When describing possible next steps, describe them in terms of what the task requires, not what you will or can do.

Use task-centered language such as:
- "The task failed because..."
- "The execution requires..."
- "A new task would need..."

Avoid agent-centered language such as:
- "I will try again"
- "I can continue if..."
- "You should let me..."

The planner is responsible for deciding the next action.
Your role is analysis and explanation only.
</system>
"""  # noqa: E501
            with audit.scope(env_id=state.env_id, sub_wd=subtask.sub_wd):
                answer = await subtask.agent_wf.run(
                    user_msg=msg,
                    ctx=subtask.agent_ctx,
                    memory=subtask.memory,
                )
            debug(f"[check_and_start] consult_agent answer:\n {answer}")
            agent.tool_required = True
            agent.tools = _tools_orig
            return answer.response.content

        return FunctionTool.from_defaults(
            name="consult_agent",
            description="""\
Consult the executor agent for analysis or explanation about a completed or failed subtask.
The executor provides analysis based on the existing execution and artifacts.
This tool is for understanding what happened during execution.
If further execution is required, create a new subtask instead of continuing the previous one.

Restrictions:
1) Do NOT instruct extra tasks using this tool.
2) This tool only be used for subtasks that are already completed, or for the immediately preceding subtask that was just executed.

Usage ordering:
This tool must be called before any other tools when you are consulting the agent about a subtask that has just failed.
Calling other tool (e.g. creat_subtask, request_user_input) makes the failed agent unreachable.
""",  # noqa: E501
            async_fn=consult_agent,
        )

    def _subtask_event_to_tool_msg(
        self,
        ev: evt.SubtaskSuccess | evt.SubtaskFail,
        completed_subtasks: dict[str, mdl.SubtaskWithDir],
    ) -> str:
        """Append subtask result to chat history."""
        tool_msg = generate_subtask_report(ev)

        if len(completed_subtasks) > 0:
            status_msg = "<completed_subtasks>\n"
            for i, name in enumerate(completed_subtasks, 1):
                subtask = completed_subtasks[name]
                status_msg += f'{i}. "{name}" by "{subtask.agent}"\n'
            status_msg += "</completed_subtasks>"
            tool_msg += "\n" + status_msg

        return tool_msg

    async def _handle_model_tool(
        self,
        tool_name: str,
        tool: Any,
        tool_input: dict,
        tool_id: str,
        completed_subtasks: dict[str, mdl.SubtaskWithDir],
    ) -> tuple[evt.StartSubtask | evt.TaskComplete | evt.TaskFail | None, str]:
        """Handle model-based tools (complete/subtask/abort).

        Returns (event, should_continue). If event is not None, break loop.
        If should_continue is True, continue to next iteration.
        """

        def _validate_subtask_tool_call(output_model) -> tuple[str, bool]:
            if output_model.name in completed_subtasks:
                tool_msg = (
                    f"{output_model.name} is already in the completed subtask list. "
                    "Use the discard tool first if you want to retry."
                )
                return tool_msg, True
            else:
                for dep in output_model.dependencies:
                    if dep in completed_subtasks:
                        continue
                    tool_msg = (
                        f"The dependency ({dep}) was not found among the "
                        "completed subtasks."
                    )
                    return tool_msg, True
            return "", False

        tool_msg = ""
        ev_to_return = None

        try:
            output_model = tool.fn(**tool_input)
        except ValidationError:
            tool_msg = "Validation error raised."

        # Tool name is constant, while tool impl could be variable
        if tool_name == complete_task_tool_default.metadata.name and not tool_msg:
            ev_to_return = evt.TaskComplete(
                complete_task=output_model, tool_id=tool_id
            )

        elif tool_name == abort_task_tool_default.metadata.name and not tool_msg:
            ev_to_return = evt.TaskFail(excuse=output_model.excuse, tool_id=tool_id)

        elif tool_name == subtask_tool.metadata.name and not tool_msg:
            tool_msg, should_continue = _validate_subtask_tool_call(output_model)
            if not should_continue:  # validation success
                subtask_to_start = mdl.SubtaskWithDir.from_subtask(
                    subtask=output_model,
                    task_number=len(completed_subtasks) + 1,
                    tool_id=tool_id,
                )
                ev_to_return = evt.StartSubtask(subtask=subtask_to_start)
        assert ev_to_return or tool_msg, "[handle model tool] logic flaw"
        return ev_to_return, tool_msg

    async def _handle_function_tool(
        self,
        tool_name: str,
        tool: AsyncBaseTool,
        tool_input: dict,
    ) -> tuple[str, bool]:
        """Handle function tools (consult_agent/discard/outline)."""
        is_error = False
        try:
            tool_output = await tool.acall(**tool_input)
            tool_msg = tool_output.content
            if tool_name == plan_outline_tool.metadata.name:
                tool_msg = "Revised plan received."
        except ValidationError as e:
            tool_msg = "Tool validation failed. Check the format."
            is_error = True
        except ValueError as e:
            tool_msg = str(e)
            is_error = True

        return tool_msg, is_error

    @step
    async def check_and_start(
        self,
        ctx: Context[mdl.PlanState],
        ev: evt.ResumeCheckAndStart
        | evt.StartTasks
        | evt.SubtaskSuccess
        | evt.SubtaskFail,
    ) -> (
        evt.StartSubtask
        | evt.InputRequiredWithStepEvent
        | evt.TaskComplete
        | evt.TaskFail
        | evt.StopEvent
    ):
        """
        Check current plan state and start next subtask or complete/fail task.
        This step does not have side effect "before" while True loop
        As while True loop necessitates only chat_history and token_sum, it is
        restartable
        """
        state = await ctx.store.get_state()

        completed_subtasks = state.completed_subtasks
        requested_output_values = state.user_requested_values
        env = world.get_env(state.env_id)

        # Create tools
        complete_task_tool = create_model_tool(
            mdl.CompleteTask.get_model_with_output_values(requested_output_values)
        )
        complete_task_tool.metadata.description = (
            complete_task_tool_default.metadata.description
        )

        abort_task_tool = create_model_tool(mdl.AbortTask)
        abort_task_tool.metadata.name = "abort_task"

        if self.stay_alive_after_completion:
            complete_task_tool.metadata.description += """\
\nThe 'report' must ends with a natural suggestion for possible next simulations.
"""
            abort_task_tool.metadata.description += """\
\nThe 'execuse' must ends by explaining the issue and proposing specific options for how to proceed.
 """

        discard_subtasks_from_tool = self._make_discard_subtasks_tool(ev)
        consult_agent_tool = self._make_consult_agent_tool(ev)
        request_user_input_tool = self._make_request_user_input_tool()

        tools = [
            subtask_tool,
            consult_agent_tool,
            request_user_input_tool,
            discard_subtasks_from_tool,
            plan_outline_tool,
            complete_task_tool,
            abort_task_tool,
        ]
        if self.with_web_search:
            tools.append(web_search_tool)
        model_tool_names = [
            complete_task_tool.metadata.name,
            subtask_tool.metadata.name,
            abort_task_tool.metadata.name,
        ]

        ev_to_return: (
            evt.StartSubtask
            | evt.InputRequiredWithStepEvent
            | evt.TaskComplete
            | evt.TaskFail
            | None
        ) = None
        while True:
            state: mdl.PlanState = await ctx.store.get_state()
            if state.action_count >= self.max_action:
                return evt.StopEvent(f"Max action count: {self.max_action} reached")

            # it read & write plan state
            tool, tool_name, tool_input, tool_id = await self._decide_tool_call(
                ctx, tools
            )
            is_error = False
            state: mdl.PlanState = await ctx.store.get_state()

            if tool_name == "request_user_input":
                debug("[check and start] request_user_input tool called")
                if state.remaining_request_user_input == 0:
                    is_error = True
                    tool_msg = """\
request_user_input call rejected: the interaction budget has been fully exhausted. 
You must proceed without further user input and make the best decision on the available context.
Further calls to this tool will fail.
"""  # noqa: E501
                else:
                    try:
                        message = tool_input.get("message")  # The agent's maessage to user
                        tool_msg = (await tool.acall(**tool_input)).content  # output of this tool
                    except ValueError as e:
                        is_error = True
                        tool_msg = str(e)
                    assert isinstance(message, str)
                    if not is_error:
                        ev_to_return = evt.InputRequiredWithStepEvent(
                            prefix=message, step="check_and_start"
                        )
                        state.remaining_request_user_input -= 1
                        if state.remaining_request_user_input < 5:
                            tool_msg += f" Remaining request_user_input budget: {state.remaining_request_user_input}"

            elif tool_name in model_tool_names:
                ev_to_return, tool_msg = await self._handle_model_tool(
                    tool_name,
                    tool,
                    tool_input,
                    tool_id,
                    completed_subtasks,
                )
                # If success, it should have a event to return
                is_error = not ev_to_return

            else:
                tool_msg, is_error = await self._handle_function_tool(
                    tool_name, tool, tool_input
                )
                ev_to_return = None

            if tool_msg:
                assert isinstance(ev_to_return, evt.InputRequiredWithStepEvent) or (
                    ev_to_return is None
                ), "[check and start] logic flaw"
                await self._append_tool_call_result(
                    ctx, tool_name, tool_msg, tool_id, is_error=is_error
                )

            await ctx.store.set_state(state)
            if ev_to_return:
                break
        #  /while True:

        assert ev_to_return is not None, "[check and start] logic flaw"

        state: mdl.PlanState = await ctx.store.get_state()
        if isinstance(ev, evt.SubtaskFail):
            debug(f"{ev.subtask.sub_wd} FAILED and moved")
            await env.discard_sub_wd(ev.subtask.sub_wd, prefix="failed")

        _subtasks = list(completed_subtasks.values())
        if isinstance(ev_to_return, evt.StartSubtask):
            _subtasks += [ev_to_return.subtask]

        assert state.current_plan is not None
        updated_plan = mdl.Plan(subtasks=_subtasks)  # type: ignore
        state.current_plan = updated_plan
        await ctx.store.set_state(state)

        # this event is automatically written to the stream when return
        if not isinstance(ev_to_return, evt.InputRequiredEvent):
            ctx.write_event_to_stream(ev_to_return)
        return ev_to_return

    async def _may_long_wait_subtask_run(
        self,
        agent_wf: Workflow,
        agent_ctx: Context,
        subtask: mdl.SubtaskWithDir,
        env: Environment,
        prompt: str,
        memory: BaseMemory,
    ) -> tuple[bool, str]:
        chat_hist = None
        while True:
            if not chat_hist:
                wf_output = await agent_wf.run(
                    user_msg=prompt,
                    ctx=agent_ctx,
                    memory=memory,
                    max_iterations=self.max_agent_iterations,
                )
            else:
                wf_output = await agent_wf.run(
                    chat_history=chat_hist,
                    ctx=agent_ctx,
                    memory=memory,
                    max_iterations=self.max_agent_iterations,
                )

            agent_state: mdl.SubtaskAgentState = await agent_ctx.store.get(
                "agent_state"
            )
            task_status = agent_state.task_status

            if task_status in (mdl.TaskStatus.SUCCESS, mdl.TaskStatus.FAIL):
                return task_status is mdl.TaskStatus.SUCCESS, wf_output

            elif task_status is mdl.TaskStatus.WAIT:
                break_reason = await _poll_slurm_job(agent_ctx, agent_state)
                prompt = (
                    f"You're receiving a polling result. Reason: {break_reason}. "
                    "Check the updated files in your working directory "
                    "and make the next decision."
                )
                # TODO: make it tool output not user
                memory.put(ChatMessage(role="user", content=prompt))
                chat_hist = memory.get()

            else:
                assert False
        #  /while True:

    @step
    async def start_subtask(
        self, ctx: Context[mdl.PlanState], ev: evt.StartSubtask
    ) -> evt.SubtaskSuccess | evt.SubtaskRetryCheck:
        state: mdl.PlanState = await ctx.store.get_state()

        task = state.task
        plan = state.current_plan
        assert isinstance(plan, mdl.Plan) and task

        completed_subtasks = state.completed_subtasks
        subtask = ev.subtask
        assert subtask.name not in completed_subtasks

        env_id = state.env_id
        env = world.get_env(env_id)

        prompt = generate_agent_prompt(
            plan=plan,
            global_task=task,
            target_subtask=subtask,
            environment=env,
        )

        try:
            (
                agent_wf,
                agent_ctx,
                modified_prompt,
            ) = await get_single_agent_workflow_with_context(
                subtask.agent,
                env_id=env.id,
                subtask=subtask,
                with_critic=self.with_critic,
                with_web_search=self.with_web_search_executors,
                original_prompt=prompt,
            )
            prompt = modified_prompt  # Use modified prompt
        except Exception as e:
            error_msg = f"Failed to initialize agent {subtask.agent}: {str(e)}"
            debug(error_msg)
            # Fail immediately with HOOK_FAILURE
            ret_ev = evt.SubtaskRetryCheck(
                subtask=subtask, reason=mdl.SubtaskFailReason.HOOK_FAILURE
            )
            ctx.write_event_to_stream(ret_ev)
            return ret_ev

        if self.tool_overrides:
            agent = agent_wf.agents[subtask.agent]
            if agent.tools:
                tools_tmp = []
                for tool in agent.tools:
                    tool_name = tool.metadata.get_name()
                    tools_tmp.append(
                        self.tool_overrides[tool_name](tool, agent.name)
                        if tool_name in self.tool_overrides
                        else tool
                    )
                agent.tools = tools_tmp

        agent_memory = ChatMemoryBuffer.from_defaults(token_limit=200000)
        error_message_to_planner, fail_reason = None, None

        start = datetime.now()
        with audit.scope(env_id=env_id, sub_wd=subtask.sub_wd):
            try:
                success, wf_output = await self._may_long_wait_subtask_run(
                    agent_wf=agent_wf,
                    agent_ctx=agent_ctx,
                    subtask=subtask,
                    env=env,
                    prompt=prompt,
                    memory=agent_memory,
                )
            except WorkflowRuntimeError as e:
                num_iterations = await agent_ctx.store.get("num_iterations", default=0)
                max_iterations = await agent_ctx.store.get(
                    "max_iterations", default=DEFAULT_MAX_ITERATIONS
                )
                if num_iterations >= max_iterations:
                    success = False
                    fail_reason = mdl.SubtaskFailReason.MAX_ITERATION
                    error_message_to_planner = (
                        f"[system] The agent has reached the maximum allowed "
                        f"iterations ({max_iterations}). It is recommended to "
                        "retry with a different instruction."
                    )
                else:
                    raise e
            audit.flush()
        end = datetime.now()
        debug(f"[start_subtask] {subtask.agent} runtime: {end - start}")

        agent_state: mdl.SubtaskAgentState = await agent_ctx.store.get("agent_state")
        message_to_planner = (
            error_message_to_planner or agent_state.message_to_planner
        )

        subtask.agent_ctx = agent_ctx
        subtask.agent_wf = agent_wf
        subtask.memory = agent_memory
        subtask.reported_message_to_planner = message_to_planner

        if not success:
            reason = fail_reason or agent_state.subtask_fail_reason
            assert isinstance(reason, mdl.SubtaskFailReason)

            debug(f"Subtask {subtask.name} fail: {reason}")
            ret_ev = evt.SubtaskRetryCheck(subtask=subtask, reason=reason)
            ctx.write_event_to_stream(ret_ev)
            return ret_ev

        debug(f"Subtask {subtask.name} success")
        subtask.reported_file_usage_summary = agent_state.file_usage_summary
        subtask.reported_output_values = agent_state.output_values
        output_files_summary = {}
        for of in subtask.output_files:
            fname = of.enumerate()[0] if of.replicates > 1 else of.filename
            # TODO: remove hard-code
            if fname.endswith((".extxyz", ".xyz", ".lammps-data")):
                output_files_summary[fname] = await pycall_summarize_structure(
                    env, fname, sub_wd=subtask.sub_wd
                )
        if output_files_summary:
            subtask.reported_output_files_summary = output_files_summary

        # update completed subtask
        ret_ev = evt.SubtaskSuccess(subtask=subtask)
        debug(f"Subtask success: {subtask.name}")

        state.completed_subtasks[subtask.name] = subtask
        await ctx.store.set_state(state)

        tool_msg = self._subtask_event_to_tool_msg(ret_ev, state.completed_subtasks)
        await self._append_tool_call_result(
            ctx,
            tool_name="create_subtask",
            tool_msg=tool_msg,
            tool_id=ret_ev.subtask.tool_id,
        )
        ctx.write_event_to_stream(ret_ev)

        return ret_ev

    @step
    async def check_retry(
        self, ctx: Context[mdl.PlanState], ev: evt.SubtaskRetryCheck
    ) -> evt.StartSubtask | evt.SubtaskFail:
        state: mdl.PlanState = await ctx.store.get_state()
        env = world.get_env(state.env_id)

        subtask = ev.subtask
        msg = subtask.reported_message_to_planner or "No record"

        if subtask.retry_cnt == self.num_retry_on_subtask_fail:
            ret_ev = evt.SubtaskFail(subtask=subtask, reason=ev.reason, message=msg)
            tool_msg = self._subtask_event_to_tool_msg(
                ret_ev, state.completed_subtasks
            )
            await self._append_tool_call_result(
                ctx,
                tool_name="create_subtask",
                tool_msg=tool_msg,
                tool_id=ret_ev.subtask.tool_id,
            )
            ctx.write_event_to_stream(ret_ev)
            return ret_ev

        debug(f"[check_retry] retry {subtask.name}, reason: {ev.reason}")
        debug(f"[check_retry] count: {subtask.retry_cnt}")

        # check failure type and filter => let's just retry all of them if num > 0
        # record any information for later investigation
        fname = str(ev.reason) or "UNKNOWN"
        fname = f"__{fname}__"
        env.write_file(content=msg, remote_path=fname, sub_wd=subtask.sub_wd)

        # discard the sub_wd
        to_dir = await env.discard_sub_wd(subtask.sub_wd, "retry")
        debug(f"[check_retry] {subtask.name} is discarded to {to_dir}")

        # remove any old field in the subtask
        subtask.agent_wf = None
        subtask.agent_ctx = None
        subtask.memory = None
        subtask.reported_output_values = None
        subtask.reported_output_files_summary = None
        subtask.reported_file_usage_summary = None
        subtask.reported_message_to_planner = None

        # increase retry_cnt += 1
        subtask.retry_cnt += 1

        # return StartSubtask event with the subtask
        return evt.StartSubtask(subtask=subtask)

    @step
    async def task_complete(
        self, ctx: Context[mdl.PlanState], ev: evt.TaskComplete
    ) -> evt.StopEvent | evt.InputRequiredWithStepEvent:
        state = await ctx.store.get_state()
        state.chat_history.append(
            ChatMessage(
                role="tool",
                content="Your report is forwarded to the user",
                additional_kwargs={"tool_call_id": ev.tool_id},
            )
        )

        if not self.stay_alive_after_completion:
            env = world.get_env(state.env_id)
            env.write_json(ev.complete_task.model_dump(), "complete.json")
            env.write_file(ev.complete_task.report, "TASK_SUCCESS")
            ev_to_return = evt.StopEvent(f"__TASK_COMPLETE__\n__{state.env_id}")
        else:
            state.remaining_request_user_input = self.request_user_input_tool_budget
            ev_to_return = evt.InputRequiredWithStepEvent(
                prefix=ev.complete_task.report, step="check_and_start"
            )
        await ctx.store.set_state(state)
        return ev_to_return

    @step
    async def task_fail(
        self, ctx: Context[mdl.PlanState], ev: evt.TaskFail
    ) -> evt.StopEvent | evt.InputRequiredWithStepEvent:
        state = await ctx.store.get_state()
        state.chat_history.append(
            ChatMessage(
                role="tool",
                content="Your excuse is forwarded to the user",
                additional_kwargs={"tool_call_id": ev.tool_id},
            )
        )

        if not self.stay_alive_after_completion:
            env = world.get_env(state.env_id)
            env.write_file(ev.excuse, "TASK_FAIL")
            ev_to_return = evt.StopEvent(f"__TASK_FAIL__\n__{state.env_id}")
        else:
            state.remaining_request_user_input = self.request_user_input_tool_budget
            ev_to_return = evt.InputRequiredWithStepEvent(
                prefix=ev.excuse, step="check_and_start"
            )
        await ctx.store.set_state(state)
        return ev_to_return


async def main() -> None:
    from paimon.models import Value

    user_msg = "What is the lattice param of Si alpha quartz computed via SevenNet-0? Use only 2 subtasks. Please use consult_agent and request_user_input at lesst once during the job."

    # Answer: total 1064 atoms, 7 LiPF6, 56 DEC, 1.058 molal is the best fit (950 - 1150 atom size window)
    cli = CommandLineInterface()

    wf = PaimonDynamicPlanWorkflow(
        llm="fast_reasoning",
        with_critic=False,
        expert_knowledge="expert/empty",
        link_wd_prefix="/mnt/odin_paimon",
        use_retrieval_planning=True,
        verbose=True,
    )

    start_event = evt.PaimonStartEvent(
        user_msg=user_msg,
        user_requested_values=[Value(name="lattice_param", unit="nm")],
    )
    ret = await cli.attach_workflow(wf.run(start_event=start_event))


if __name__ == "__main__":
    asyncio.run(main())
