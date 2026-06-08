"""LLM-based document summarizer for RAG embedding keys."""

from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage

from paimon.llm import get_llm, run_agent_pipeline
from paimon.util.tool_factory import create_model_tool


class DocumentSummary(BaseModel):
    """Structured summary for embedding."""

    title: str = Field(description="Document title or main topic")
    summary: str = Field(description="2-3 sentence summary of key content")

    model_config = {"extra": "forbid"}


SUMMARIZER_PROMPT = """\
You are a document summarizer for a RAG (Retrieval-Augmented Generation) system.
Create a concise, searchable summary that will be used as an embedding key.

Document:
{content}

Create a summary that:
1. Captures the main topic and purpose of this document
2. Includes key technical terms and concepts
3. Is 2-3 sentences maximum
4. Would help match user queries about this topic

Use the DocumentSummary tool to provide your response.
"""

_summary_tool = create_model_tool(DocumentSummary)


async def summarize_for_embedding(
    content: str,
    title: str = "",
    llm_class: str = "fast",
    max_content_chars: int = 8000,
) -> str:
    """Generate embedding key from document content using LLM.

    Args:
        content: Full document content
        title: Optional title hint
        llm_class: LLM class to use (default: "fast")
        max_content_chars: Max chars to send to LLM

    Returns:
        Formatted embedding key: "Title: {title}\n\nSummary: {summary}"
    """
    truncated = content[:max_content_chars]
    prompt = SUMMARIZER_PROMPT.format(content=truncated)
    chat_history = [ChatMessage(role="user", content=prompt)]

    _, tool_calls, _ = await run_agent_pipeline(
        llm=get_llm(llm_class, metadata={"role": "doc_summarizer"}),
        tools=[_summary_tool],
        chat_history=chat_history,
        allow_parallel_tool_calls=False,
        tool_required=True,
        agent_name="doc_summarizer",
    )

    if not tool_calls:
        # Fallback: use first paragraph as summary
        first_para = content.split("\n\n")[0][:500]
        return f"Title: {title or 'Untitled'}\n\nSummary: {first_para}"

    result = DocumentSummary(**tool_calls[0].tool_kwargs)
    final_title = title or result.title
    return f"Title: {final_title}\n\nSummary: {result.summary}"
