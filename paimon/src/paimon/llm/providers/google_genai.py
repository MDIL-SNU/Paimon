"""
Google genai (gemini)
Their llms are stateless and had to return all "signature" to retrieve previous
thought

Cache is implicit, but there exists explicit options provided via API
"""

from llama_index.core.base.llms.types import ChatResponse
from llama_index.llms.google_genai import GoogleGenAI
import google.genai.types as types

from paimon.token_sum import TokenUsageEntry
from paimon.util.log import debug


def init_llm(model: str, thinking_budget: int = -1, **kwargs):
    """
    For 2.5 models,
    thinking budget = 0 to disable
                    = -1 to auto
    thinking level (similar to openai effort) is supported from version 3

    [Important] As we put generation_config directly, other kwargs are ignored in
    the __init__ of llamaindex, e.g., temperature
    """

    return GoogleGenAI(
        model=model,
        generation_config=types.GenerateContentConfig(
            temperature=1.0,
            max_output_tokens=64000,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True, thinking_budget=thinking_budget
            ),
        ),
        **kwargs,
    )


def llm_preprocess(
    llm: GoogleGenAI,
    metadata: dict[str, str] | None = None,  # paimon uses internally
    **kwargs,
):
    kwargs["labels"] = metadata
    return kwargs


def get_usage_from_response(
    llm: GoogleGenAI, response: ChatResponse, name: str
) -> TokenUsageEntry:
    assert response.raw, "[GoogleGenAI:get_usage_from_response] raw is not found"
    llm_model = llm.metadata.model_name

    usage = types.GenerateContentResponseUsageMetadata(
        **response.raw["usage_metadata"]
    )

    tool_calls = []
    for part in response.raw["content"]["parts"]:
        if (
            "function_call" in part
            and part["function_call"]
            and "name" in part["function_call"]
        ):
            tool_calls.append(part["function_call"]["name"])
    tool_calls = tool_calls or "no tool call"

    cached_input_token = usage.cached_content_token_count or 0
    reasoning_token = usage.thoughts_token_count or 0
    input_token = usage.prompt_token_count or 0
    candidates_token = usage.candidates_token_count or 0
    output_token = candidates_token + reasoning_token

    return TokenUsageEntry(
        name=name,
        llm_model=llm_model,
        tool_call=tool_calls,
        input_tokens=input_token,
        cached_tokens=cached_input_token,
        reasoning_tokens=reasoning_token,
        output_tokens=output_token,
        total_tokens=usage.total_token_count or 0,
    )
