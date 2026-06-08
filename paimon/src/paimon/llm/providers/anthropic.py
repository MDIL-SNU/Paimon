from llama_index.core.base.llms.types import (
    MessageRole,
    ChatResponse,
    ChatMessage,
    CachePoint,
    CacheControl,
)
from llama_index.llms.anthropic import Anthropic
from anthropic.types import Usage, ToolUseBlock

from paimon import cfg
from paimon.token_sum import TokenUsageEntry
from paimon.util.log import debug


USE_CACHE_DEFAULT = True


def init_llm(model, thinking: bool = True, budget_tokens: int = 12000, **kwargs):
    # TODO: redirect accidently given reasoning_effort to thinking?
    thinking_dict = None
    if thinking:
        thinking_dict = {"type": "enabled", "budget_tokens": budget_tokens}

    kwargs.pop("metadata", None)
    return Anthropic(
        model,
        max_tokens=64000,
        timeout=cfg.timeout,
        thinking_dict=thinking_dict,
        # cache_idx=-1,  # TODO: It seems not work as well as I want
        **kwargs,
    )


def _apply_simple_caching(
    chat_history: list[ChatMessage], metadata: dict[str, str] | None = None
) -> list[ChatMessage]:
    """
    Put cache control at the last of message and rely on Anthropic to do the right

    TODO: it is more efficient to not requrest cache if it is obvious that the chat
    will end here.
    """
    ttl = "5m"
    if metadata and metadata.get("role", "None") == "planner":
        ttl = "1h"
    cp_block = CachePoint(cache_control=CacheControl(type="ephemeral", ttl=ttl))

    # remove all previous CachePoints, except system prompt
    for chat in chat_history:
        blocks_tmp = []
        if chat.role == MessageRole.SYSTEM:
            continue
        for block in chat.blocks:
            if not isinstance(block, CachePoint):
                blocks_tmp.append(block)
        chat.blocks = blocks_tmp

    # Append at the last
    chat_history[-1].blocks.append(cp_block.model_copy())
    return chat_history


def llm_preprocess(
    llm: Anthropic,
    chat_history: list[ChatMessage],
    metadata: dict[str, str] | None = None,  # paimon uses internally
    **kwargs,
):
    if llm.thinking_dict and llm.thinking_dict.get("type", None) == "enabled":
        kwargs["temperature"] = 1.0  # reasoning model must use this
        if kwargs.get("tool_required"):
            kwargs["tool_required"] = False  # it can not be used with reasoning

    if USE_CACHE_DEFAULT:
        _apply_simple_caching(chat_history, metadata)

    # metadata is not applicable to Anthropic API (only possible for openai response)
    kwargs["chat_history"] = chat_history
    return kwargs


def get_usage_from_response(
    llm: Anthropic, response: ChatResponse, name: str
) -> TokenUsageEntry:
    assert response.raw, "[Anthropic:get_usage_from_response] raw is not found"
    usage: Usage = response.raw["usage"]
    llm_model = response.raw["model"]

    tool_calls = []
    for cnt in response.raw["content"]:
        if isinstance(cnt, ToolUseBlock):
            tool_calls.append(cnt.name)
    tool_calls = tool_calls or "no tool call"
    input_cached = usage.cache_read_input_tokens or 0
    
    # TODO: this is expensive than just input tokens and pricing depends on "ttl"
    input_cache_creation = usage.cache_creation_input_tokens or 0

    return TokenUsageEntry(
        name=name,
        llm_model=llm_model,
        tool_call=tool_calls,
        input_tokens=usage.input_tokens + input_cached + input_cache_creation,
        cached_tokens=input_cached,  # todo: IDK what it is
        reasoning_tokens=0,  # not a.v.a
        output_tokens=usage.output_tokens,
        total_tokens=(usage.input_tokens + usage.output_tokens),
    )
