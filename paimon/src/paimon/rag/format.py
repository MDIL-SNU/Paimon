"""Generic RAG result formatting utilities.

Provides formatters for different types of RAG retrieval results.
"""

from __future__ import annotations
from typing import Any

from llama_index.core.llms import ChatMessage

from paimon.llm import get_llm, run_agent_pipeline
from paimon import cfg


def add_truncation_marker(
    text: str,
    max_chars: int,
    marker: str = "\n\n[TRUNCATED]",
) -> str:
    """Add truncation marker if text exceeds max length.

    Args:
        text: Text to truncate
        max_chars: Maximum characters
        marker: Truncation marker

    Returns:
        Truncated text with marker if needed
    """
    if len(text) > max_chars:
        return text[: max_chars - len(marker)] + marker
    return text


def format_code_search_results(
    results: list[dict[str, Any]],
    include_context: bool = True,
    include_source: bool = True,
    max_results: int | None = None,
    max_chars: int = 15000,
) -> str:
    """Format code search results for agent consumption.

    Args:
        results: List of code search results with similarity scores
        include_context: Whether to include context_before/after
        include_source: Whether to include source URLs
        max_results: Maximum number of results to format (None = all)
        max_chars: Maximum total characters

    Returns:
        Formatted text string
    """
    if not results:
        return "No code examples found for the query."

    if max_results:
        results = results[:max_results]

    output = []
    output.append(f"Found {len(results)} relevant code examples:\n")

    for i, result in enumerate(results, 1):
        similarity = result.get("similarity_score", 0.0)
        output.append(f"\n{'=' * 60}")
        output.append(f"\nExample {i} (relevance: {similarity:.2f})")

        if include_source and result.get("source_url"):
            output.append(f"\nSource: {result['source_url']}")

        lang = result.get("language", "")
        if lang:
            output.append(f"\nLanguage: {lang}")

        if include_context and result.get("context_before"):
            output.append(f"\nContext: {result['context_before'][:200]}...")

        code = result.get("code", "")
        output.append(f"\nCode:\n```{lang}")
        output.append(code)
        output.append("```")

        if include_context and result.get("context_after"):
            output.append(f"\nNote: {result['context_after'][:200]}...")

    full_text = "\n".join(output)
    return add_truncation_marker(full_text, max_chars)


# LAMMPS-specific formatters

ANNOTATOR_PROMPT = """
You are a reasoning-offloading RAG agent for atomic simulation scripting.

You will be given:
- An instruction describing the user's actual simulation intent.
- A query used to retrieve documentation.
- A retrieved document describing one or more commands and their options.

Your role is NOT to summarize the document or generate a runnable script.

Your role is to:
- Interpret the retrieved document strictly in the context of the given instruction.
- Identify which command(s) from this document are relevant to the instruction.
- Determine which options SHOULD be used, and which options exist in the document but SHOULD NOT be used for this instruction.

Important constraints:
- The instruction has higher priority than the document.
- If a functionality appears to have already been handled elsewhere in the instruction, do NOT recommend performing it again using options from this document.
- Do NOT introduce commands or options that are not present in the retrieved document.
- Do NOT omit the command itself, even if you judge it to be unnecessary in isolation.

When applicable, explicitly point out:
- Options that might appear useful from the document alone, but would cause redundancy, conflict, or unintended behavior when combined with the instruction.
- The reasoning behind enabling or disabling each relevant option, based only on the instruction and the document.

Assume that an upstream agent will combine multiple such documents.
Your output should help the upstream agent avoid redundant reasoning and prevent physically or logically incorrect simulation setups.

Focus on decision-making and reasoning, not completeness or verbosity.

INPUTS:
<instruction>
{task_description}
</instruction>
<retrieval_query>
{original_query}
</retrieval_query>
<retrieved document>
{full_doc}
</retrieved document>
"""  # noqa: E501


def format_basic(result, max_chars: int = 10000) -> str:
    """Format LAMMPS retrieval result with basic information.

    Args:
        result: RetrievalResult from LAMMPS RAG
        max_chars: Maximum characters

    Returns:
        Formatted text string
    """
    output = []
    output.append(f"Command: {result.command_name}")
    output.append(f"\nSyntax:\n{result.syntax}")
    output.append(f"\nExamples:\n{result.examples}")
    output.append(f"\nDescription:\n{result.quoted_description}")

    full_text = "".join(output)
    return add_truncation_marker(full_text, max_chars)


def format_all_sections(result, max_chars: int = 10000) -> str:
    """Format LAMMPS retrieval result with all sections.

    Args:
        result: RetrievalResult from LAMMPS RAG
        max_chars: Maximum characters

    Returns:
        Formatted text string
    """
    output = [f"Command: {result.command_name}\n"]

    for section_name, section_content in result.doc_raw.all_sections.items():
        if section_name == "Description":
            output.append(f"\n{section_name}:\n{result.quoted_description}")
        else:
            output.append(f"\n{section_name}:\n{section_content}")

    full_text = "".join(output)
    return add_truncation_marker(full_text, max_chars)


async def format_annotated(
    original_query: str,
    result,
    task_description: str,
    max_chars: int = 50000,
) -> str:
    """Format LAMMPS retrieval result with LLM annotation.

    Args:
        original_query: Original retrieval query
        result: RetrievalResult from LAMMPS RAG
        task_description: Task description for context
        max_chars: Maximum characters

    Returns:
        Tuple of (formatted text, token usage)
    """
    full_doc = format_all_sections(result, max_chars=50000)

    annotator_prompt = ANNOTATOR_PROMPT.format(
        task_description=task_description,
        original_query=original_query,
        full_doc=full_doc,
    )

    if cfg.rag_config.annotator:
        llm = get_llm(cfg.rag_config.annotator_LLM)
        messages = [ChatMessage(role="user", content=annotator_prompt)]
        contextual_note, _, _ = await run_agent_pipeline(
            llm=llm,
            chat_history=messages,
            agent_name="annotator",
            metadata={"role": "annotator"},
        )

        # Combine contextual note with full documentation
        output = f"""\
<annotator recommendations>
{contextual_note.message.content}
</annotator recommendations>

{full_doc}
        """

        return add_truncation_marker(output, max_chars)
    else:
        return full_doc


def format_code_search_compact(
    results: list[dict[str, Any]],
    max_results: int = 5,
    max_code_lines: int = 10,
) -> str:
    """Compact format for code search results (minimal verbosity).

    Args:
        results: List of code search results
        max_results: Maximum number of results
        max_code_lines: Maximum lines of code per result

    Returns:
        Compact formatted text
    """
    if not results:
        return "No code examples found."

    results = results[:max_results]
    output = []

    for i, result in enumerate(results, 1):
        score = result.get("similarity_score", 0.0)
        code = result.get("code", "")
        lines = code.split("\n")

        output.append(f"\n[{i}] Score: {score:.2f}")

        # Show first N lines
        for line in lines[:max_code_lines]:
            output.append(f"  {line}")

        if len(lines) > max_code_lines:
            output.append(f"  ... ({len(lines) - max_code_lines} more lines)")

    return "\n".join(output)
