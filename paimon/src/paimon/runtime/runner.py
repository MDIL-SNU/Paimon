import os.path as osp
import asyncio
import json
from asyncio import StreamReader, StreamWriter

from llama_index.llms.openai import OpenAIResponses
from llama_index.core.base.llms.types import ChatMessage

import paimon.world as world
from paimon.agent.paimon_agent import PaimonAgent
from paimon.runtime.interface.web import WebInterface
from paimon.models import Plan, Subtask, TaskStatus
from paimon.token_sum import TokenSum
from paimon.util.artifacts import (
    save_global_run_artifacts,
    restore_global_run_artifacts,
)
from paimon.llm.base import get_llm
from paimon.util.log import info, debug
from paimon.workflow.dynamic_plan import _poll_slurm_job


def _build_preference_prompt_block(key: str | None) -> str:
    """Return XML prompt block for the given user preference key, or empty string."""
    if not key:
        return ""
    from paimon.runtime.user_preference import get_preferences

    pref = get_preferences()[key]
    return f"<user_preference>\n{pref.prompt}\n</user_preference>"


async def runner_multi_agent_workflow(env_id: str, config: dict, socket_path: str):
    from paimon.workflow.dynamic_plan import PaimonDynamicPlanWorkflow
    from paimon.workflow.events import PaimonStartEvent
    from paimon.runtime.interface.web import MultiAgentWebInterface
    from paimon.runtime.user_preference import get_preferences

    env = world.get_env(env_id)
    # Note: is_new = config["is_new"] can be used for session resurrection

    # Initialize workflow with config options
    llm_model = config.get("llm", "base_reasoning")
    extra = config.get("extra", {})

    # Preference overrides are defaults; explicit extra values win
    user_preference_key = config.get("user_preference")
    pref_overrides = (
        get_preferences()[user_preference_key].overrides
        if user_preference_key
        else {}
    )
    merged_extra = {**pref_overrides, **extra}

    wf = PaimonDynamicPlanWorkflow(
        llm=llm_model,
        verbose=merged_extra.pop("verbose", True),
        **merged_extra,
    )
    interface = MultiAgentWebInterface(
        socket_path=socket_path,
        env=env,
        config=config,
    )

    info(f"[runner] {env_id}: Start multi-agent loop")
    try:
        # Unlike subsequent user inputs are event-driven,
        # the first input is explicitely requested here as start event requires this.
        save_global_run_artifacts(
            env=env,
            metadata={"action_count": 0, **config},
            last_event={"name": "InputRequiredEvent"},
        )
        user_msg, user_files = await interface.get_input()

        # Prepend user preference prompt to the first user message
        user_msg = (
            _build_preference_prompt_block(config.get("user_preference"))
            + "\n"
            + user_msg
        )

        save_global_run_artifacts(
            env=env,
            last_event={
                "name": "user_msg",
                "content": user_msg,
                "files": user_files,
            },
        )
        start_event = PaimonStartEvent(
            session_name=config.get("task_name", "Unknown"),
            env_id=env.id,
            user_msg=user_msg,
            files=user_files,
            expert_knowledge=None,
        )
        wf_handler = wf.run(start_event=start_event)
        await interface.attach_workflow(wf_handler)
    finally:
        save_global_run_artifacts(
            env=env,
            last_event={"name": "termination"},
        )
        debug("[runner] Escape multi-agent loop. Closing.")


async def runner_single_agent_workflow(env_id: str, config: dict, socket_path: str):
    from llama_index.core.memory import ChatMemoryBuffer

    from paimon.workflow.subtask_agent import (
        get_standalone_agent_workflow_with_context,
    )

    env = world.get_env(env_id)
    llm_model = config.get("llm", "gpt-5-mini")
    llm_model = f"openai/{llm_model}"  # TODO: currently hard-coded for openai
    reasoning = config.get("reasoning", "low").lower()
    extra = config.get("extra", {})

    # TODO: OpenAI only
    llm = get_llm(
        llm_model, reasoning_options={"effort": reasoning, "summary": "detailed"}
    )

    agent_name = config.get("agent")
    assert agent_name, "agent name is not found"
    # Dummy plan for parsing from web server
    plan: Plan = Plan(
        subtasks=[
            Subtask(
                name="working directory",
                primary_task_type="Analysis",
                agent=agent_name,
                instruction="",
            )
        ]
    )

    agent_wf, agent_ctx = await get_standalone_agent_workflow_with_context(
        agent_name=agent_name,
        env_id=env_id,
        llm=llm,
        **extra,
    )

    agent: PaimonAgent = agent_wf.agents[agent_name]  # type: ignore
    state = await agent_ctx.store.get("agent_state")
    is_new = config["is_new"]  # must exist

    agent_memory = ChatMemoryBuffer.from_defaults(token_limit=400000)
    if is_new:
        chat_history: list[ChatMessage] = []
        # Create an 01_working_directory, for web server to check
        _ = env.get_sub_wd_path(sub_wd=state.sub_wd)
    else:
        prev_dict = restore_global_run_artifacts(env)
        chat_history = prev_dict["chat"]
        metadata = prev_dict["metadata"]  # TODO: this is .globals.json
        tokens = prev_dict["tokens"]
        agent.token_used = tokens
        _ = prev_dict["last_event"]  # TODO
        if "openai" in llm_model:
            prv_resp_id = metadata["previous_response_id"]
            llm._previous_response_id = prv_resp_id
        # Restore agent venv from previous session
        saved_venv = metadata.get("current_venv")
        if saved_venv:
            state.current_venv = saved_venv
            await agent_ctx.store.set("agent_state", state)

    reader: StreamReader
    writer: StreamWriter
    try:
        # 1 Turn = 1 loop = 1 user msg
        while True:
            # Save "waiting for input" state
            state = await agent_ctx.store.get("agent_state")
            save_global_run_artifacts(
                env=env,
                metadata={
                    "action_count": len(chat_history) // 2,
                    "previous_response_id": getattr(llm, "_previous_response_id"),
                    "current_venv": state.current_venv,
                    **config,
                },
                chat=chat_history,
                tokens=agent.token_used,
                json_artifacts={".plan.json": plan},
                last_event={"name": "InputRequiredEvent"},
            )

            info(f"[runner] {env_id}: Open socket")
            reader, writer = await asyncio.open_unix_connection(socket_path)

            # Read user message from socket
            data = await reader.read()

            # Immediately save event on response, for sync
            # Real response event with rich information is written later
            save_global_run_artifacts(
                env=env,
                last_event={"name": "HumanResponseEvent"},
            )

            request = json.loads(data.decode("utf-8"))

            debug(f"[runner] {env_id}: Received: {request}")
            assert "type" in request, "invalid payload"

            if request["type"] == "terminate":
                debug(f"[runner] {env_id}: Termination signal received. Break loop.")
                break

            assert request["type"] == "user_msg"
            if "user_msg" not in request or not isinstance(request["user_msg"], str):
                raise ValueError("invalid payload")

            # Prepend user preference prompt on first message only
            msg = request["user_msg"]
            if is_new and not chat_history:
                msg = (
                    _build_preference_prompt_block(config.get("user_preference"))
                    + "\n"
                    + msg
                )
            mid_chat_files = request.get("files", [])
            lines = ""
            user_provided_fnames = []
            for file_path in mid_chat_files:
                info(f"[runner] {env_id}: Uploading mid-chat file: {file_path}")
                await env.put(file_path, sub_wd="01_working_directory")
                fname = osp.basename(file_path)
                user_provided_fnames.append(fname)
                lines += f"\n- {fname}"
            if lines:
                msg += f"""\
\n<user provided files>
The user uploaded the following files in this message:{lines}
</user provided files>"""

            wd_str = env.list_working_directory(state.sub_wd)
            wd_str = f"\n<wd_snapshot>\n{wd_str}</wd_snapshot>"
            msg += wd_str

            user_msg = ChatMessage(content=msg, role="user")
            if user_provided_fnames:
                user_msg.additional_kwargs["files"] = user_provided_fnames

            chat_history.append(user_msg)
            save_global_run_artifacts(
                env=env,
                last_event={
                    "name": "user_msg",
                    "content": msg,
                    "files": user_provided_fnames,
                },
            )

            # Create WebInterface with all dependencies
            interface = WebInterface(
                writer=writer,
                agent=agent,
                env=env,
                config=config,
            )

            # Run agent workflow via interface
            debug(f"[runner] {env_id}: Start agent workflow")
            wf_handler = agent_wf.run(
                chat_history=chat_history,
                ctx=agent_ctx,
                memory=agent_memory,
                max_iterations=50,
            )
            agent_output = await interface.attach_workflow(wf_handler)

            # Handle SLURM WAIT status (standalone mode)
            state = await agent_ctx.store.get("agent_state")
            debug(
                f"[runner] {env_id}: "
                f"task_status={state.task_status}, "
                f"slurm_job_id={state.slurm_job_id}"
            )
            while state.task_status is TaskStatus.WAIT:
                break_reason = await _poll_slurm_job(agent_ctx, state)
                # Reset to INIT so a stale WAIT doesn't re-enter this loop
                # if the agent exits without calling a status-changing tool.
                state.task_status = TaskStatus.INIT
                await agent_ctx.store.set("agent_state", state)

                resume_msg = (
                    f"You're receiving a polling result. Reason: {break_reason}. "
                    "Check the updated files in your working directory "
                    "and make the next decision."
                )
                agent_memory.put(ChatMessage(role="user", content=resume_msg))
                resume_hist = agent_memory.get()

                wf_handler = agent_wf.run(
                    chat_history=resume_hist,
                    ctx=agent_ctx,
                    memory=agent_memory,
                    max_iterations=50,
                )
                await interface.attach_workflow(wf_handler)
                state = await agent_ctx.store.get("agent_state")
                debug(
                    f"[runner] {env_id}: after resume "
                    f"task_status={state.task_status}, "
                    f"slurm_job_id={state.slurm_job_id}"
                )

            chat_history = agent_memory.get()  # it contains tool calls

            # Signal done
            await interface.send_event({"type": "done"})

            debug(f"[runner] {env_id}: Response ended. Close socket.")
            writer.close()
            await writer.wait_closed()
    finally:
        save_global_run_artifacts(
            env=env,
            last_event={"name": "termination"},
        )
        debug("[runner] Escape loop. Closing.")


async def runner_chat(env_id, config, socket_path):
    chat_history: list[ChatMessage] = []
    llm_model = config.get("llm", "gpt-5-nano")
    llm = OpenAIResponses(model=llm_model, track_previous_responses=True)
    env = world.get_env(env_id)

    reader: StreamReader
    writer: StreamWriter

    toks = TokenSum([])
    try:
        while True:
            save_global_run_artifacts(
                env=env,
                metadata={"action_count": len(chat_history) // 2, **config},
                chat=chat_history,
                tokens=toks,
                last_event={"name": "InputRequiredEvent"},
            )

            info(f"[runner] {env_id}: Open socket")
            # establish socket connection
            # It knocks the server. the callback function will be called.
            # In this case, the callback function will **request** for a user msg.
            reader, writer = await asyncio.open_unix_connection(socket_path)

            data = await reader.read()
            request = json.loads(data.decode("utf-8"))

            debug(f"[runner] {env_id}: Received: {request}")
            assert "type" in request, "invalid payload"

            if request["type"] == "terminate":
                debug(f"[runner] {env_id}: Termination signal received. Break loop.")
                break
            else:
                assert request["type"] == "user_msg"

            if "user_msg" not in request or not isinstance(request["user_msg"], str):
                # TODO
                raise ValueError("invalid payload")

            msg = ChatMessage(content=request["user_msg"], role="user")
            chat_history.append(msg)

            debug(f"[runner] {env_id}: Start call")
            response_gen = await llm.astream_chat([msg])

            async for response in response_gen:
                if response.delta:  # response.delta is str | None
                    writer.write(response.delta.encode("utf-8"))
                    await writer.drain()

            chat_history.append(response.message)

            # TODO: Streaming response has different type for response.raw
            # new_toks = get_usage(llm, response, "gpt-5-nano runner")
            # toks = toks + new_toks

            save_global_run_artifacts(
                env=env,
                metadata={"action_count": len(chat_history) // 2, **config},
                chat=chat_history,
                tokens=toks,
                last_event={"name": "Response"},
            )

            debug(f"[runner] {env_id}: Response ended. Close socket.")
            writer.close()
            await writer.wait_closed()
    finally:
        debug("[runner] Escape loop. Closing.")
