"""Documentation RAG systems.

Retrieval-augmented generation systems for documentation:
- LAMMPS command documentation (RST files)
- Expert knowledge (plain text files)
"""

from .models import (
    DocEntry,
    LAMMPSDoc,
    RetrievalResult,
    SubQuery,
    SubQueryList,
    CommandSelection,
    ComplexRetrievalResult,
)
from .parser import LAMMPSDocParser
from .retrievers import LAMMPSRetrieverBuilder, DocRetrieverBuilder
from .rag import LAMMPSRAGSystem, ExpertRAGSystem
from ..format import format_basic, format_all_sections, format_annotated

__version__ = "2.0.0"

__all__ = [
    # Generic
    "DocEntry",
    "DocRetrieverBuilder",
    "ExpertRAGSystem",
    # LAMMPS-specific
    "LAMMPSDoc",
    "RetrievalResult",
    "SubQuery",
    "SubQueryList",
    "CommandSelection",
    "ComplexRetrievalResult",
    "LAMMPSDocParser",
    "LAMMPSRetrieverBuilder",
    "LAMMPSRAGSystem",
    # Formatters
    "format_basic",
    "format_all_sections",
    "format_annotated",
]
