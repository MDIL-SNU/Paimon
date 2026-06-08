"""Paimon package base module. Execute runtime constant code"""

from llama_index.core.base.llms.types import ChatResponse
from llama_index.llms.openai import OpenAIResponses
from openai.types.responses import Response

from paimon.token_sum import TokenUsageEntry
from paimon import cfg


def init_llm(
    model,
    reasoning_options: dict[str, str] | None = None,
    verbosity: str = "low",
    metadata: dict[str, str] | None = None,
    reasoning_effort: str = "medium",
    **kwargs,
):
    reasoning_options = reasoning_options or {
        "effort": reasoning_effort,
        "summary": "detailed",
    }
    return OpenAIResponses(
        model=model,
        temperature=cfg.temperature,
        store=cfg.open_ai_store,
        track_previous_responses=False,
        reasoning_options=reasoning_options,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        additional_kwargs={"text": {"verbosity": verbosity}, "metadata": metadata},
        **kwargs,
    )


def llm_preprocess(llm: OpenAIResponses, **kwargs):
    return kwargs


def get_usage_from_response(
    llm: OpenAIResponses, response: ChatResponse, name: str
) -> TokenUsageEntry:
    assert response.raw and isinstance(response.raw, Response), (
        "response is not from openai_response api"
    )
    response_openai = response.raw

    usage = response_openai.usage
    assert usage, "usage is not avaialble"
    llm_model = response_openai.model

    tool_call = [
        item.name
        for item in getattr(response.raw, "output", [])
        if item.type == "function_call"
    ]
    tool_call = tool_call or "no tool call"

    input_tokens = usage.input_tokens
    cached_tokens = usage.input_tokens_details.cached_tokens
    output_tokens = usage.output_tokens
    reasoning_tokens = usage.output_tokens_details.reasoning_tokens
    total_tokens = usage.total_tokens

    return TokenUsageEntry(
        name=name,
        llm_model=llm_model,
        tool_call=tool_call,
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
    )
