from typing import Literal

from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage

from paimon.knowledge.library import get_knowledge
import paimon.llm
from paimon.llm import run_agent_pipeline


async def query_python_library(
    library: Literal["freud"],
    query: str,
) -> str:
    # TODO: MDAnalysis
    sup = ["freud"]
    if library not in sup:
        return f"Unsupported library '{library}'. Supported libraries: {sup}."

    prompt = """You are a Python library documentation assistant specializing in {library}.

Role: You are an expert assistant that helps users understand and use the {library} Python library by providing accurate function signatures, class definitions, and documentation.

Documentation: 
{docs}

Instructions:
1. When answering queries about functions or classes, always copy-paste the exact function signature including parameter types and return types
2. When answering queries about member functions, answer both class definition (with __init__) and the function.
3. Summarize docstrings, including only the essential information required for correct usage.
4. If multiple related functions exist, mention them with their signatures
5. Preserve all type annotations exactly as they appear
6. If the query is ambiguous, provide the most commonly used functions/methods that match
7. Always format code signatures using proper Python syntax highlighting
8. Use a concise, professional tone. Do not include pleasantries or offers of further help.
"""  # noqa: E501

    prompt = prompt.format(library=library, docs=get_knowledge(f"code/{library}"))
    llm = paimon.llm.get_llm(llm_class="fast")

    inp = [
        ChatMessage(role="system", content=prompt),
        ChatMessage(role="user", content=query),
    ]
    resp, _, _ = await run_agent_pipeline(
        llm=llm,
        chat_history=inp,
        agent_name="retrieval",
        metadata={"role": "query_python_library"},
    )
    return str(resp.message.content)


query_python_library_tool = FunctionTool.from_defaults(
    name="query_python_library",
    description="""This tool queries the documentation for Python libraries (e.g., freud) and provides exact function signatures, parameter types, and return types. Specify the library name and your question about functions, classes, or usage patterns.""",  # noqa: E501
    async_fn=query_python_library,
)


