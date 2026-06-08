import traceback
import json
from typing import Sequence, Optional, Any

from llama_index.core.agent.workflow.workflow_events import (
    AgentInput,
    AgentOutput,
    ToolCallResult,
)
from llama_index.core.base.llms.types import TextBlock
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import BaseMemory
from llama_index.core.tools import AsyncBaseTool
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow.function_agent import FunctionAgent
from llama_index.core.tools import ToolOutput

from paimon.models import SubtaskAgentState, TaskStatus
from paimon.world import get_env
from paimon.util.log import debug, debug_var
from paimon.util.chat import dump_chat
from paimon.token_sum import TokenSum
from paimon.llm import run_agent_pipeline
import paimon.audit as audit


class ToolSystemError(Exception):
    """
    Custom exception raised inside the tool to escape from llamaindex agent
    """

    SENTINEL = "<<<TOOL_SYSTEM_ERROR>>>"

    def __init__(self, message: str, tool_meta: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.tool_meta = tool_meta or {}

    def __str__(self) -> str:
        payload = {
            "message": self.message,
            "tool_meta": self.tool_meta,
        }
        # Keep it single-line to simplify parsing
        return f"{self.SENTINEL}{json.dumps(payload, separators=(',', ':'))}"

    @classmethod
    def from_string(cls, s: str) -> Optional["ToolSystemError"]:
        """from the serialized string, or return None if it doesn't match."""
        if not s or not s.startswith(cls.SENTINEL):
            return None
        json_part = s[len(cls.SENTINEL) :]
        try:
            data = json.loads(json_part)
        except json.JSONDecodeError:
            return None
        # be defensive about missing keys
        message = data.get("message", "Unknown ToolSystemError")
        tool_meta = data.get("tool_meta") or {}
        return cls(message=message, tool_meta=tool_meta)


def parse_tool_system_error(tool_output: ToolOutput) -> ToolSystemError | None:
    """
    Returns dict with keys: code, message, meta
    or None if not a tagged error.
    """
    s = tool_output.raw_output or tool_output.content or ""
    if not s.startswith(ToolSystemError.SENTINEL):
        return None
    ts = None
    if ts := ToolSystemError.from_string(s):
        ts.tool_meta["tool_name"] = tool_output.tool_name
    return ts


class PaimonAgent(FunctionAgent):
    """
    Some additional features

    Attributes
    ----------
    list_wd_for_tool_call: bool
        automatically append 'ls' of working directory for every tool call. Saved to
        memory.
    """

    critic_gate_tool_names: list[str] | None = None

    token_used: TokenSum | None = None

    scratchpad_key: str = "scratchpad"

    tool_required: bool = True
    list_wd_for_tool_call: bool = False

    async def _dump(self, ctx, memory) -> None:
        state: SubtaskAgentState = await ctx.store.get("agent_state")
        env = get_env(state.env_id)
        scratchpad = await ctx.store.get(self.scratchpad_key, default=[])
        to_dump = dump_chat(
            chat_history=scratchpad,
            memory=memory,
            system_prompt=self.system_prompt,
            drop_early=1,
        )
        to_dump["current_venv"] = state.current_venv
        env.write_json(to_dump, ".agent_memory.json", state.sub_wd)

        if self.token_used:
            env.write_json(
                self.token_used.to_dict(), ".agent_tokens.json", state.sub_wd
            )

        audit.flush()

    async def take_step(
        self,
        ctx: Context,
        llm_input: list[ChatMessage],
        tools: Sequence[AsyncBaseTool],
        memory: BaseMemory,  # not used
    ) -> AgentOutput:
        """Take a single step with the function calling agent."""
        if not self.llm.metadata.is_function_calling_model:
            raise ValueError("LLM must be a FunctionCallingLLM")

        scratchpad: list[ChatMessage] = await ctx.store.get(
            self.scratchpad_key, default=[]
        )

        current_llm_input = [*llm_input, *scratchpad]

        ctx.write_event_to_stream(
            AgentInput(input=current_llm_input, current_agent_name=self.name)
        )

        state: SubtaskAgentState = await ctx.store.get("agent_state")
        llm_metadata = {
            "role": "executor",
            "env_id": state.env_id,
            "agent_name": state.agent_name,
            "sub_wd": state.sub_wd,
        }

        last_chat_response, tool_calls, toks = await run_agent_pipeline(
            llm=self.llm,
            tools=tools,
            chat_history=current_llm_input,
            allow_parallel_tool_calls=False,
            tool_required=self.tool_required,
            agent_name=self.name,
            metadata=llm_metadata,
        )

        self.token_used = self.token_used + toks if self.token_used else toks

        # only add to scratchpad if we didn't select the handoff tool
        scratchpad.append(last_chat_response.message)
        await ctx.store.set(self.scratchpad_key, scratchpad)
        await self._dump(ctx, memory)

        raw = (
            last_chat_response.raw.model_dump()
            if isinstance(last_chat_response.raw, BaseModel)
            else last_chat_response.raw
        )
        return AgentOutput(
            response=last_chat_response.message,
            tool_calls=tool_calls or [],
            raw=raw,
            current_agent_name=self.name,
        )

    async def handle_tool_call_results(
        self, ctx: Context, results: list[ToolCallResult], memory: BaseMemory
    ) -> None:
        """Handle tool call results for function calling agent."""
        scratchpad: list[ChatMessage] = await ctx.store.get(
            self.scratchpad_key, default=[]
        )

        # Paimon agent must have this
        state: SubtaskAgentState = await ctx.store.get("agent_state")
        env = get_env(state.env_id)

        for tool_call_result in results:
            tool_output: ToolOutput = tool_call_result.tool_output
            if system_err := parse_tool_system_error(tool_output):
                raise system_err

            if tool_output.is_error:
                debug(f"Tool Call ERROR. {tool_output.model_dump_json()}")
                if tool_output.exception:
                    exe = "".join(
                        traceback.format_exception(
                            type(tool_output.exception),
                            tool_output.exception,
                            tool_output.exception.__traceback__,
                        )
                    )
                    debug(f"Tool Call Exception. {exe}\n")
                else:
                    debug(f"Tool output exeption is empty")

            if (
                self.list_wd_for_tool_call
                and isinstance(tool_call_result.tool_output.blocks[-1], TextBlock)
                and not tool_call_result.return_direct
            ):
                list_str = env.list_working_directory(state.sub_wd)
                tool_call_result.tool_output.blocks[
                    -1
                ].text += f"\n<wd_snapshot>\n{list_str}</wd_snapshot>"

            scratchpad.append(
                ChatMessage(
                    role="tool",
                    blocks=tool_call_result.tool_output.blocks,
                    additional_kwargs={"tool_call_id": tool_call_result.tool_id},
                )
            )
            task_status = state.task_status
            debug(
                f"[paimon_agent] tool={tool_call_result.tool_name} "
                f"task_status={task_status}"
            )

            if tool_call_result.tool_name == "submit_and_wait":
                if task_status is TaskStatus.WAIT:
                    tool_call_result.return_direct = True
                elif task_status == TaskStatus.HOLD:
                    tool_call_result.return_direct = False
                elif task_status == TaskStatus.FAIL:
                    tool_call_result.return_direct = True
                else:
                    debug(
                        "[paimon_agent] !! submit_and_wait tool returned "
                        f"unexpected task_status: {task_status} overwrite to hold"
                        "Check tool call error"
                    )
                    state.task_status = TaskStatus.HOLD
                    tool_call_result.return_direct = False
                    await ctx.store.set("agent_state", state)


            elif tool_call_result.tool_name == "complete_task":
                if tool_call_result.tool_output.is_error:
                    debug("Complete tool call error")
                    tool_call_result.return_direct = False

                if task_status in (TaskStatus.SUCCESS, TaskStatus.FAIL):
                    tool_call_result.return_direct = True

                elif task_status == TaskStatus.HOLD:
                    tool_call_result.return_direct = False

                else:
                    raise ValueError(
                        "[paimon_agent] complete_task tool returned "
                        f"unexpected task_status: {task_status}"
                    )

            """
            MEMO: I can not understant purpose of this assistant chat message addition.
            It becomes right this, if agent ends the task with return_direct=True,
            
            assistant (tool call) => tool output => assistant (content is same as the tool output)

            As a result, assistant memory ends with the tool output that it did not say.
            More over, the last two chat message content is identical.

            It becomes awkward if we start conversation with the agent.

            In conclusion, I comment out this section which is originated from llama-index

            elif (
                tool_call_result.return_direct
                and tool_call_result.tool_name != "handoff"
            ):
                scratchpad.append(
                    ChatMessage(
                        role="assistant",
                        content=str(tool_call_result.tool_output.content),
                        additional_kwargs={"tool_call_id": tool_call_result.tool_id},
                    )
                )
                break
            """

        await ctx.store.set(self.scratchpad_key, scratchpad)
        await self._dump(ctx, memory)

    async def finalize(
        self, ctx: Context, output: AgentOutput, memory: BaseMemory
    ) -> AgentOutput:
        """Finalize the function calling agent.

        Adds all in-progress messages to memory.

        Room for customization:
        Executed when 'no-tool calls' or 'return_direct=True' call
        As it puts all the scratch pad into memory, hand-off becomes very long
        """
        await self._dump(ctx, memory)

        scratchpad: list[ChatMessage] = await ctx.store.get(
            self.scratchpad_key, default=[]
        )
        await memory.aput_messages(scratchpad)

        # reset scratchpad
        await ctx.store.set(self.scratchpad_key, [])

        return output
