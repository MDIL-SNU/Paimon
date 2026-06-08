"""Data models for documentation RAG systems."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


@dataclass
class DocEntry:
    """Generic document entry for RAG.

    Used for expert knowledge files and other simple text documents.
    Simpler than LAMMPSDoc - just stores the full content with an embedding key.
    """

    doc_id: str
    title: str
    content: str
    embedding_key: str
    filepath: Path
    doc_type: str = "generic"


@dataclass
class LAMMPSDoc:
    """LAMMPS documentation page.

    Attributes:
        filepath: Path to the RST file
        command_name: Command name (e.g., "compute_msd")
        command_type: Command type (e.g., "compute", "fix", "pair")
        syntax: Syntax section content
        examples: Examples section content
        description_paragraphs: List of description paragraphs
        full_content: Raw RST content
        all_sections: Dictionary mapping section names to their content (includes ALL sections)
    """
    filepath: Path
    command_name: str
    command_type: str
    syntax: str
    examples: str
    description_paragraphs: list[str]
    full_content: str
    all_sections: dict[str, str]


class RetrievalResult(BaseModel):
    """Final retrieval result with structured output."""

    command_name: str = Field(description="Name of the selected command")
    syntax: str = Field(description="Syntax section")
    examples: str = Field(description="Examples section")
    quoted_description: str = Field(
        description="Selected important paragraphs from description"
    )
    doc_raw: LAMMPSDoc

    model_config = {"extra": "forbid"}


class SubQuery(BaseModel):
    """A primitive search query decomposed from a complex task."""

    role: Literal["dynamics", "constraints", "building_block", "record"] = Field(
        description="Type of query: dynamics, constraints, building_block, or record"
    )
    description: str = Field(
        description="One-sentence explanation of what documentation is needed"
    )
    query: str = Field(description="Actual query string to send to RAG retriever")

    model_config = {"extra": "forbid"}


class SubQueryList(BaseModel):
    """List of subqueries for query decomposition."""

    subqueries: list[SubQuery] = Field(
        description="List of primitive search queries decomposed from the task"
    )

    model_config = {"extra": "forbid"}


class CommandSelection(BaseModel):
    """LLM selection of command and paragraph numbers."""

    command_index: int | None = Field(
        description="Index of selected command (1-based), or None if no match"
    )
    paragraph_numbers: list[int] = Field(
        description="List of selected paragraph numbers (1-based)"
    )

    model_config = {"extra": "forbid"}


class ComplexRetrievalResult(BaseModel):
    """Result from complex query decomposition and retrieval."""

    task_description: str
    subqueries: list[SubQuery]
    results: list[RetrievalResult | None]
    formatted_output: str

    model_config = {"extra": "forbid"}
