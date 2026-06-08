"""
Wrapper functions of llama-index to add custom logics
"""
from typing import Sequence

from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM

from .base import get_method
from paimon.util.log import debug, debug_var


async def achat_with_tools(
    llm: LLM,
    tools: list | Sequence,
    user_msg: str | ChatMessage | None = None,
    chat_history: list[ChatMessage] | None = None,
    tool_required: bool = False,
    allow_parallel_tool_calls: bool = False,
    metadata: dict | None = None,  # only for paimon (openai response)
    **kwargs,
):
    kwargs = get_method(llm, "llm_preprocess")(
        llm=llm,
        tools=tools,
        user_msg=user_msg,
        chat_history=chat_history,
        tool_required=tool_required,
        allow_parallel_tool_calls=allow_parallel_tool_calls,
        metadata=metadata,
        **kwargs,
    )
    return await llm.achat_with_tools(**kwargs)  # type: ignore


async def achat(
    llm: LLM,
    chat_history: list[ChatMessage] | None = None,
    metadata: dict | None = None,  # only for paimon (openai response)
    **kwargs,
):
    kwargs = get_method(llm, "llm_preprocess")(
        llm=llm,
        chat_history=chat_history,
        metadata=metadata,
        **kwargs,
    )
    # to make consistency with achat_with_tools, keep chat_history arg but redirect
    messages = kwargs.pop("chat_history")
    return await llm.achat(messages=messages, **kwargs)  # type: ignore


def chat_with_tools(
    llm: LLM,
    tools: list | Sequence,
    user_msg: str | ChatMessage | None = None,
    chat_history: list[ChatMessage] | None = None,
    tool_required: bool = False,
    allow_parallel_tool_calls: bool = False,
    metadata: dict | None = None,  # only for paimon (openai response)
    **kwargs,
):
    kwargs = get_method(llm, "llm_preprocess")(
        llm=llm,
        tools=tools,
        user_msg=user_msg,
        chat_history=chat_history,
        tool_required=tool_required,
        allow_parallel_tool_calls=allow_parallel_tool_calls,
        metadata=metadata,
        **kwargs,
    )
    return llm.chat_with_tools(**kwargs)  # type: ignore


def chat(
    llm: LLM,
    chat_history: list[ChatMessage] | None = None,
    metadata: dict | None = None,  # only for paimon (openai response)
    **kwargs,
):
    kwargs = get_method(llm, "llm_preprocess")(
        llm=llm,
        chat_history=chat_history,
        metadata=metadata,
        **kwargs,
    )
    messages = kwargs.pop("chat_history")
    return llm.chat(messages=messages, **kwargs)  # type: ignore


