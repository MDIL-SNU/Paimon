from typing import Sequence

from llama_index.core.base.llms.types import ToolCallBlock
from llama_index.core.llms import ChatMessage, ChatResponse, LLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.tools import AsyncBaseTool

from paimon.token_sum import TokenSum
from paimon.llm import get_usage, achat_with_tools, achat, chat_with_tools, chat
from paimon.util.chat import normalize_tool_call_block
import paimon.audit as audit


def _stamp_and_push(toks: TokenSum, role: str | None) -> None:
    """Set role on each entry and push to the audit scope."""
    for entry in toks.items:
        entry.role = role
        audit.push(entry)


async def run_agent_pipeline(
    llm: LLM,
    chat_history: list[ChatMessage],
    *,
    tools: Sequence[AsyncBaseTool] | None = None,
    allow_parallel_tool_calls: bool = False,
    tool_required: bool = False,
    agent_name: str = "Unknown",
    metadata: dict | None = None,
    **kwargs,
) -> tuple[ChatResponse, list[ToolSelection], TokenSum]:
    """
    Run a standard LLM -> tool extraction -> usage pipeline.

    Parameters
    ----------
    llm : BaseLLM
        LLM instance implementing achat_with_tools and get_tool_calls_from_response.
    tools : list
        Tool definitions passed to achat_with_tools.
    chat_history : list
        Messages to feed into achat_with_tools.
    allow_parallel_tool_calls : bool, False
    tool_required : bool, False
    agent_name : str | "None"
        Name for usage logging.
    metadata: dict | None
        metadata paimon uses internally
    kwargs :
        kwargs to 'achat_with_tools'

    Returns
    -------
    response
        Raw LLM response object.
    tool_calls : list[ToolSelection]
        Extracted tool call descriptors.
    usage : TokenSum
        Token usage info.
    """

    if tools:
        last_chat_response = await achat_with_tools(
            llm=llm,
            tools=tools,
            chat_history=chat_history,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            tool_required=tool_required,
            metadata=metadata,
            **kwargs,
        )
        tool_calls: list[ToolSelection] = llm.get_tool_calls_from_response(  # type: ignore
            last_chat_response, error_on_no_tool_call=tool_required
        )

    else:
        last_chat_response = await achat(
            llm=llm,
            chat_history=chat_history,
            metadata=metadata,
            **kwargs,
        )
        tool_calls = []

    tmp = []
    idx = 0
    for block in last_chat_response.message.blocks:
        if isinstance(block, ToolCallBlock):
            # Assume the index is preserved. I'm not sure but there's no way
            target_tc = tool_calls[idx]
            idx += 1
            block = normalize_tool_call_block(block)
            block.normalized_tool_kwargs = target_tc.tool_kwargs
        tmp.append(block)
    last_chat_response.message.blocks = tmp

    toks = get_usage(llm, last_chat_response, agent_name)
    role = (metadata or {}).get("role")
    _stamp_and_push(toks, role)

    return last_chat_response, tool_calls, toks
