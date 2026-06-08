from typing import Literal, Callable

from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import ChatResponse

from paimon import cfg
from paimon.token_sum import TokenSum


def get_method(
    llm: LLM, method: Literal["llm_preprocess", "get_usage_from_response"]
) -> Callable:
    class_name = llm.class_name()  # llama-index fixed classifier
    if class_name == "openai_responses_llm":
        from .providers.openai_response import (
            llm_preprocess,
            get_usage_from_response,
        )
    elif class_name == "Anthropic_LLM":
        from .providers.anthropic import llm_preprocess, get_usage_from_response
    elif class_name == "GenAI":
        from .providers.google_genai import llm_preprocess, get_usage_from_response
    else:
        raise NotImplementedError(class_name)

    if method == "llm_preprocess":
        return llm_preprocess
    elif method == "get_usage_from_response":
        return get_usage_from_response
    else:
        raise ValueError(f"Wrong method: {method}")


def get_llm(llm_class: str, metadata: dict[str, str] | None = None, **kwargs) -> LLM:
    """Get LLM model from string

    Parameters
    ----------
    llm_class
        should be one of "fast", "fast_reasoning", "base" and "base_reasoning"
    metadata:
        metadata that paimon uses internally.
    kwargs
        Additional LLM specific keyword arguements

    Returns
    -------
    LLM
    """
    may_llm_class = llm_class
    if not llm_class.endswith("_llm"):
        may_llm_class = llm_class + "_llm"

    # If one of "fast_llm", "base_llm", etc, read from config
    # if not, it is full specification of the llm (e.g. openai/gpt-5)
    llm_keyword = getattr(cfg, may_llm_class, llm_class)

    class_kwargs_attr = may_llm_class.replace("_llm", "_kwargs")
    class_kwargs = getattr(cfg, class_kwargs_attr, {})
    kwargs = {**class_kwargs, **kwargs}

    tmp = llm_keyword.split("/")
    if len(tmp) == 1:
        api = "openai"  # default
        model = llm_keyword
    else:
        api, model = tmp[0], "/".join(tmp[1:])

    if api == "openai":
        from .providers.openai_response import init_llm
    elif api == "anthropic":
        from .providers.anthropic import init_llm
    elif api == "google_genai":
        from .providers.google_genai import init_llm
    else:
        raise NotImplementedError(f"{llm_class} is not implemented")

    kwargs["metadata"] = metadata

    return init_llm(model, **kwargs)


def get_usage(llm: LLM, response: ChatResponse, name: str) -> TokenSum:
    """
    Create a TokenUsageEntry object from a chat response object.

    Args:
        llm: llm
        response: The chat response object (expects response.raw to have usage)
        name: Name of the agent making the request

    Returns:
        TokenUsageEntry object
    """
    return TokenSum(
        items=[
            get_method(llm, "get_usage_from_response")(
                llm=llm, response=response, name=name
            )
        ]
    )
