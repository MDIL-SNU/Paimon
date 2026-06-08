import json

from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.base.llms.types import (
    ChatMessage,
    ThinkingBlock,
    ToolCallBlock,
)

from paimon.models import NormalizedToolCall


def dump_chat(
    chat_history: list[ChatMessage],
    memory: BaseMemory | None = None,
    system_prompt: str | None = None,
    drop_early: int = 0,
) -> dict:
    """
    Concatenate chat message in [system_prompt] + [memory] + [chat_history] order
    Returns dumpable dict
    """

    msgs = memory.get_all() if memory else []
    sysp = (
        [ChatMessage(role="system", content=system_prompt)] if system_prompt else []
    )
    lst = (sysp + msgs + chat_history)[drop_early:]
    tmp = ChatMemoryBuffer.from_defaults(lst)
    return tmp.to_dict()


def normalize_tool_call_block(block: ToolCallBlock) -> NormalizedToolCall:
    """Convert a plain ToolCallBlock into a NormalizedToolCall.

    Handles both deserialized blocks (string tool_kwargs) and already-parsed ones.
    If the block is already a NormalizedToolCall, returns it as-is.
    """
    if isinstance(block, NormalizedToolCall):
        return block
    kwargs = block.tool_kwargs
    if isinstance(kwargs, str):
        kwargs = json.loads(kwargs)
    return NormalizedToolCall(
        **block.model_dump(),
        normalized_tool_kwargs=kwargs,
    )


def chat_hist_to_str(chat_hist: list[ChatMessage], start_idx: int = 0) -> str:
    ret = []
    agent_step = 1
    for i, chat in enumerate(chat_hist):
        role = chat.role.value
        if role in ["system", "user", "developer"]:
            ret.append(f"=== {role.upper()} | PROMPT ===\n{chat.content}")
        elif role == "assistant":
            # reasoning => tool => content order
            for block in chat.blocks:
                if isinstance(block, ThinkingBlock) and block.content:
                    ret.append(
                        f"=== STEP {agent_step} | {role.upper()} | REASONING ==="
                        f"\n{block.content}"
                    )
                elif isinstance(block, ToolCallBlock):
                    block = normalize_tool_call_block(block)
                    ret.append(
                        f"=== STEP {agent_step} | {role.upper()}: "
                        f"TOOL CALL '{block.tool_name}' | ARGUMENTS ==="
                        f"\n{block.normalized_tool_kwargs}"
                    )
                elif chat.content:
                    ret.append(f"=== {role.upper()} | MESSAGE ===\n{chat.content}")
        elif role == "tool":
            # assert last_tool_call_name is not None
            ret.append(
                f"=== STEP {agent_step} | {role.upper()} | OUTPUT ==="
                f"\n{chat.content}"
            )
            agent_step += 1

        if i < start_idx:
            # run logic, but ignore outputs if it is below start_idx
            ret = []
    return "\n".join(ret)
