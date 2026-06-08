# Critics and its logic attached to gated tools
from typing import Literal
from collections import defaultdict
import asyncio
import traceback

from pydantic import ValidationError
from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool

from paimon import cfg
from paimon.llm import get_llm
from paimon.models import (
    Verdict,
    CriticalOpinion,
    SubtaskAgentState,
    CriticCommitteeState,
    TaskStatus,
    SubtaskFailReason,
)
from paimon.agent import ToolSystemError
from paimon.world import get_env
from paimon.util.chat import chat_hist_to_str, dump_chat
from paimon.util.tool_factory import create_model_tool
from paimon.util.log import debug, debug_var, debug_assert
from paimon.llm import run_agent_pipeline
from paimon.knowledge.library import get_knowledge

# read only
critical_opinion_tool = create_model_tool(CriticalOpinion)


def llm_critic_default() -> LLM:
    return get_llm(cfg.critic_config.critic_llm)


def _get_critic_system_prompt(
    mode: Literal["Completion Review", "Pre-Submission Gate"],
    agent_name: str | None = None,
) -> str:
    path_map = {
        "Completion Review": "agents/critic/completion_review",
        "Pre-Submission Gate": "agents/critic/pre_submission_gate",
    }

    prompts = [get_knowledge("agents/critic/common"), get_knowledge(path_map[mode])]

    if agent_name:
        c_name = agent_name.lower().replace(" ", "_")
        prompts.append(get_knowledge(f"agents/{c_name}/critic", default=""))

    return "\n".join(p for p in prompts if p)


async def spokesman_msg(
    opinions: list[CriticalOpinion],
    metadata: dict[str, str] | None = None,
) -> str:
    # TODO: make it configurable
    llm = get_llm("fast_reasoning")
    ops = "\n".join([op.model_dump_json() for op in opinions])
    inputs = [
        ChatMessage(
            role="system", content=get_knowledge("agents/critic/message_summarizer")
        ),
        ChatMessage(role="user", content=ops),
    ]

    resp, _, _ = await run_agent_pipeline(
        llm=llm,
        chat_history=inputs,
        agent_name="spokesman",
        metadata=metadata,
    )

    return str(resp.message.content)


async def chat_critic(
    index: int,
    llm: LLM,
    sys_prompt: str,
    critic_chat_hist: list[ChatMessage],
    agent_traj: list[ChatMessage] | str,
    metadata: dict[str, str] | None = None,
) -> tuple[int, CriticalOpinion]:
    # The logic relies on mutability of critic_chat_hist
    # So the instance must be used with "append". No replacement or reassign
    debug("[chat_critic] enter")
    if isinstance(agent_traj, str):
        memory_str = agent_traj
    else:
        memory_str = chat_hist_to_str(agent_traj)

    if len(critic_chat_hist) == 0:
        # Fresh start
        critic_chat_hist.append(
            ChatMessage(
                role="system",
                content=sys_prompt,
            ),
        )
    else:
        critic_chat_hist.append(
            ChatMessage(
                role="user",
                content="""\
[Message to critic]
Based on your opinion, the agent have done more steps. Update your decision.
""",
            )
        )
    critic_chat_hist.append(ChatMessage(role="user", content=memory_str))

    num_try = 0
    while num_try < 3:
        resp, tool_calls, _ = await run_agent_pipeline(
            llm=llm,
            tools=[critical_opinion_tool],
            chat_history=critic_chat_hist,
            allow_parallel_tool_calls=False,
            tool_required=True,
            agent_name=f"critic_{index}",
            metadata=metadata,
        )
        critic_chat_hist.append(resp.message)

        debug("[chat_critic] resp received")
        assert len(tool_calls) == 1, "multiple tool call"
        tool_call = tool_calls[0]
        tool_id = tool_call.tool_id

        cop = None
        if tool_call.tool_name != "CriticalOpinion":
            tool_msg = (
                "You must use the CriticalOpinion tool. Other tools are not allowed."
            )
        else:
            try:
                cop = critical_opinion_tool.fn(**tool_call.tool_kwargs)
                tool_msg = (
                    "Your opinion is received and will be reported to the agent."
                )
            except ValidationError as e:
                tool_msg = str(e)

        critic_chat_hist.append(
            ChatMessage(
                role="tool",
                content=tool_msg,
                additional_kwargs={"tool_call_id": tool_id},
            )
        )
        if cop:
            break
        num_try += 1

    if not cop:
        raise ValueError(
            "Critic failed to generate valid opinion more than 3 times!"
        )
    return index, cop


async def criticize_agent(
    agent_context: Context,
    agent_state: SubtaskAgentState,
    mode: Literal["Completion Review", "Pre-Submission Gate"],
) -> tuple[Verdict, str]:
    """Criticize agent's report and its trajectory with number of different
    critics. Returns next subtask status and return message to the agent.
    If next subtask status is 'Success', return string is empty

    If agent_state has memory about previous critic opinions, the critics
    that raise concern or reject is resurrected and continues.

    Parameters
    ----------
    agent_context
        agent context
    agent_state
        agent_state from the ctx (TODO: redundant)
    mode
        critic mode. One of "Completion Review" or "Pre-Submission Gate"

    Returns
    -------
    TaskStatus
        The next status of this subtask
    Message to the agent
        Return message to the agent based on critic result
    """

    if mode == "Completion Review":
        assert agent_state.task_status == TaskStatus.SUCCESS, (
            "[criticize_agent]task status not match"
        )

    ccs = await agent_context.store.get("critic_committee_state", None)

    agent_sys = ChatMessage(role="system", content=agent_state.system_prompt)
    agent_mem = (await agent_context.store.get("memory")).get_all()
    scratchpad = await agent_context.store.get("scratchpad")
    traj = [agent_sys] + agent_mem + scratchpad
    # TODO: ccs config is mutable, but this logic does not allow it.
    if ccs is None:
        debug("[critic] ccs init")
        ccs = CriticCommitteeState(last_agent_traj_index=len(traj))
        ccs.critic_llms = [llm_critic_default() for _ in range(ccs.num_critics)]
        ccs.critic_memories = [[] for _ in range(ccs.num_critics)]
        ccs.last_verdicts = [None for _ in range(ccs.num_critics)]
        traj_str = chat_hist_to_str(traj)
    else:
        debug("[critic] ccs again")
        # Display "from" last_agent_traj_index
        traj_str = chat_hist_to_str(traj, ccs.last_agent_traj_index)
        # If hit this block once more, the index should be updated
        ccs.last_agent_traj_index = len(traj)

    assert (
        isinstance(ccs.critic_memories, list)
        and isinstance(ccs.last_verdicts, list)
        and isinstance(ccs.critic_llms, list)
    ), "[criticize_agent] typing assert"

    updated_verdicts = ccs.last_verdicts.copy()

    metadata = {
        "env_id": agent_state.env_id,
        "sub_wd": agent_state.sub_wd,
        "mode": mode,
    }

    tasks = []
    debug("[critic] starting tasks")
    for i, (critic_llm, chat_hist, last_verdict) in enumerate(
        zip(ccs.critic_llms, ccs.critic_memories, ccs.last_verdicts)
    ):
        meta_crit = metadata.copy()
        meta_crit.update({"role": "critic", "critic_index": str(i)})
        if last_verdict is None or last_verdict in [Verdict.REJECT, Verdict.CONCERN]:
            tasks.append(
                asyncio.create_task(
                    chat_critic(
                        index=i,
                        llm=critic_llm,  # type: ignore
                        sys_prompt=_get_critic_system_prompt(
                            mode, agent_state.agent_name
                        ),
                        critic_chat_hist=chat_hist,
                        agent_traj=traj_str,
                        metadata=meta_crit,
                    )
                )
            )

    opinion_list: list[tuple[int, CriticalOpinion]] = await asyncio.gather(*tasks)
    debug("[critic] await done")
    op_dct = defaultdict(list)
    for i, op in opinion_list:
        op_dct[op.verdict].append(op)
        updated_verdicts[i] = op.verdict

    ccs.current_turn = ccs.current_turn + 1
    ccs.last_verdicts = updated_verdicts
    await agent_context.store.set("critic_committee_state", ccs)

    meta_spok = metadata.copy()
    meta_spok.update({"role": "spokesman"})
    ret_decision, ret_msg = None, None

    if Verdict.MALICIOUS in op_dct:
        debug("[critic] malicious")
        decisions = op_dct[Verdict.MALICIOUS]
        ret_msg = await spokesman_msg(decisions, meta_spok)
        ret_decision = Verdict.MALICIOUS

    elif Verdict.REJECT in op_dct:
        debug("[critic] reject")
        decisions = op_dct[Verdict.REJECT] + op_dct[Verdict.CONCERN]
        ret_msg = await spokesman_msg(decisions, meta_spok)
        ret_decision = Verdict.REJECT

    elif (
        len(op_dct[Verdict.CONCERN]) / ccs.num_critics
    ) > ccs.need_actions_concern_ratio:
        debug("[critic] concern")
        decisions = op_dct[Verdict.CONCERN]
        ret_msg = await spokesman_msg(op_dct[Verdict.CONCERN], meta_spok)
        ret_decision = Verdict.CONCERN

    else:
        debug("[critic] pass")
        ret_decision, ret_msg = Verdict.PASS, ""

    # dump critic hist and token usage
    env = get_env(agent_state.env_id)
    to_dump = {}
    for idx, critic_memory in enumerate(ccs.critic_memories):
        to_dump[f"critic_{idx}"] = dump_chat(critic_memory, drop_early=1)
    env.write_json(to_dump, ".critic_memory.json", agent_state.sub_wd)

    return ret_decision, ret_msg


def _attach_to_submit_tool(submit_tool: FunctionTool):
    submit_fn = submit_tool.async_fn

    async def submit_w_criteria(*args, **kwargs):
        assert "ctx" in kwargs, "[submit_w_criteria] ctx not in kwargs"

        ctx: Context = kwargs["ctx"]
        agent_state: SubtaskAgentState = await ctx.store.get("agent_state")

        ccs = await ctx.store.get("critic_committee_state", None)
        if ccs and ccs.submit_ticket:
            # Short-circuit for already-passed case
            return await submit_fn(*args, **kwargs)

        try:
            verdict, critic_msg = await criticize_agent(
                ctx, agent_state, mode="Pre-Submission Gate"
            )
        except ToolSystemError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            bt = traceback.format_exc()
            debug(f"[COMPLETE_TASK][CRITIC][BUG] TRACEBACK:\n {bt}")
            raise ToolSystemError(f"[COMPLETE_TASK][CRITIC][BUG] {e}") from e

        # state is changed with "criticize_agent" call, and must exist now
        ccs = await ctx.store.get("critic_committee_state")
        last_chance = ccs.current_turn == ccs.maximum_turn
        if last_chance and verdict != Verdict.PASS:
            agent_state.task_status = TaskStatus.FAIL
            agent_state.subtask_fail_reason = SubtaskFailReason.CRITIC_MAX_ITERATION
            agent_state.message_to_planner = critic_msg
            return_message = critic_msg

        elif verdict == Verdict.MALICIOUS:  # to planner
            agent_state.task_status = TaskStatus.FAIL
            agent_state.subtask_fail_reason = SubtaskFailReason.CRITIC_MALICIOUS

            agent_state.message_to_planner = critic_msg
            return_message = critic_msg
            await ctx.store.set("agent_state", agent_state)

        elif verdict in [Verdict.CONCERN, Verdict.REJECT]:  # to agent
            agent_state.task_status = TaskStatus.HOLD
            return_message = f"""\
Job submission is blocked. Reflect below explanations through additional actions and retry with the job submission tool.\n\n{critic_msg}
"""
            await ctx.store.set("agent_state", agent_state)

        else:  # verdict == Verdict.PASS:  # to planner
            async with ctx.store.edit_state() as state_now:
                state_now.critic_committee_state.submit_ticket = True

            return_message = (
                "Job submission is permitted. Starting job submission ..."
            )
            await ctx.store.set("agent_state", agent_state)
            submit_msg = await submit_fn(*args, **kwargs)
            return_message = return_message + "\n" + submit_msg

        return return_message

    submit_tool._async_fn = submit_w_criteria
    return submit_tool


def _attach_to_complete_tool(complete_task_tool: FunctionTool):
    assert complete_task_tool.metadata.get_name() == "complete_task", (
        "[attach_critic_committee]: not complete_task tool"
    )

    complete_task_fn = complete_task_tool.async_fn

    async def complete_task_w_criteria(*args, **kwargs):
        assert "ctx" in kwargs, "[complete_task_w_criteria] ctx not in kwargs"

        orig_msg = await complete_task_fn(*args, **kwargs)
        # TaskStatus should be one of SUCCESS or HOLD (system check not pass)

        ctx: Context = kwargs["ctx"]
        agent_state: SubtaskAgentState = await ctx.store.get("agent_state")

        if agent_state.task_status == TaskStatus.HOLD:
            # complete tool's system validations failed (file existence, etc)
            # nothing to do.
            return orig_msg

        elif agent_state.task_status == TaskStatus.SUCCESS:
            # If agent passed minimal checks of "complete_task" tool, then
            # apply critic => overwrite task_status
            try:
                verdict, critic_msg = await criticize_agent(
                    ctx, agent_state, mode="Completion Review"
                )
            except ToolSystemError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as e:
                bt = traceback.format_exc()
                debug(f"[COMPLETE_TASK][CRITIC][BUG] TRACEBACK:\n {bt}")
                raise ToolSystemError(f"[COMPLETE_TASK][CRITIC][BUG] {e}") from e

            ccs: CriticCommitteeState = await ctx.store.get(
                "critic_committee_state", None
            )
            assert ccs, "critic committee state is not found"
            last_chance = ccs.current_turn == ccs.maximum_turn

            return_message = None
            if last_chance and verdict != Verdict.PASS:  # to planner
                msg = f"Maximum attempt reached.\n{critic_msg}"
                agent_state.task_status = TaskStatus.FAIL
                agent_state.subtask_fail_reason = (
                    SubtaskFailReason.CRITIC_MAX_ITERATION
                )
                agent_state.message_to_planner = msg
                return_message = msg

            elif verdict == Verdict.MALICIOUS:  # to planner
                msg = f"This agent is failed.\n{critic_msg}"

                agent_state.task_status = TaskStatus.FAIL
                agent_state.subtask_fail_reason = SubtaskFailReason.CRITIC_MALICIOUS
                agent_state.message_to_planner = msg
                return_message = msg

            elif verdict in [Verdict.CONCERN, Verdict.REJECT]:  # to agent
                agent_state.task_status = TaskStatus.HOLD

                return_message = f"""\
Task completion is blocked. Reflect below explanations through additional actions and retry with the complete_task tool.\n\n{critic_msg}"""
            else:  # verdict == Verdict.PASS:  # to planner
                agent_state.task_status = TaskStatus.SUCCESS
                return_message = orig_msg

            await ctx.store.set("agent_state", agent_state)
            return return_message
        else:
            raise ToolSystemError(
                f"submit_and_wait tool returned unexpected task_status: {agent_state.task_status}"
            )

    complete_task_tool._async_fn = complete_task_w_criteria
    return complete_task_tool


def attach_critic_committee(tool: FunctionTool) -> FunctionTool:
    """
    Attach critic committee to the tool based on its name
    If it is not applicable, raises an error

    Parameters
    ----------
    tool
        Target tool to attach committee. Should be one of "complete_task"
        or "submit*" tools

    Returns
    -------
    tool
        modified tool
    """
    tool_name = tool.metadata.get_name()
    setattr(tool, "_original_tool_async_fn", tool.async_fn)

    if tool_name == "complete_task":
        return _attach_to_complete_tool(tool)
    elif tool_name in ["submit_and_wait", "submit_job"]:
        return _attach_to_submit_tool(tool)
    else:
        raise ValueError(f"Unsupported tool: {tool_name}")


if __name__ == "__main__":
    print(dir(Verdict))
