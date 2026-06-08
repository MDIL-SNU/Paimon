from pydantic import BaseModel

from llama_index.core.llms import ChatMessage

from paimon.util.tool_factory import create_model_tool
from .base import get_llm
from .pipeline import run_agent_pipeline


async def is_true(query: str) -> bool:
    """Ask LLM for whether it is true or false. Uses 'fast' LLM for prediction.

    Example:
        ans = paimon.llm.is_true("Is sky blue?")

    Parameters
    ----------
    query
        True or false question for LLM

    Returns
    -------
    bool
    """

    class TrueOrFalseResponse(BaseModel):
        is_true: bool

    tool = create_model_tool(TrueOrFalseResponse)
    llm = get_llm("fast")

    user_msg = ChatMessage(content=query, role="user")
    _, tool_calls, _ = await run_agent_pipeline(
        llm=llm,
        chat_history=[user_msg],
        tools=[tool],
        tool_required=True,
        agent_name="is_true",
        metadata={"role": "is_true"},
    )
    tool_call = tool_calls[0]
    output_model = TrueOrFalseResponse(**tool_call.tool_kwargs)
    return output_model.is_true
