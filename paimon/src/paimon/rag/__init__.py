"""RAG (Retrieval-Augmented Generation) systems.

All RAG systems use a single shared ChromaDB database with separate collections:
- LAMMPS: 'lammps_docs' collection for LAMMPS documentation
- Expert: 'expert_knowledge' collection for expert knowledge files
- Code Search: 'code_<pkg>' collections for package-specific code examples

This package contains:
- docs: Documentation RAG (LAMMPS RST docs, expert knowledge)
- code_search: Web-based code extraction and search
- Unified CLI for building and querying indexes
"""

# Re-export documentation RAG
from .docs import (
    # Generic
    DocEntry,
    DocRetrieverBuilder,
    ExpertRAGSystem,
    # LAMMPS-specific
    LAMMPSDoc,
    RetrievalResult,
    SubQuery,
    SubQueryList,
    CommandSelection,
    ComplexRetrievalResult,
    LAMMPSDocParser,
    LAMMPSRetrieverBuilder,
    LAMMPSRAGSystem,
    format_basic,
    format_all_sections,
    format_annotated,
)

# Indexer utilities
from .indexer import (
    # LAMMPS
    build_lammps_index,
    query_lammps_docs,
    get_lammps_doc_by_name,
    # Expert knowledge
    build_expert_index,
    query_expert_knowledge,
)
from .code_search.local_indexer import (
    build_package_index,
    query_package_code,
)

__version__ = "2.0.0"

__all__ = [
    # Generic
    "DocEntry",
    "DocRetrieverBuilder",
    "ExpertRAGSystem",
    # LAMMPS RAG
    "LAMMPSDoc",
    "RetrievalResult",
    "SubQuery",
    "SubQueryList",
    "CommandSelection",
    "ComplexRetrievalResult",
    "LAMMPSDocParser",
    "LAMMPSRetrieverBuilder",
    "LAMMPSRAGSystem",
    "format_basic",
    "format_all_sections",
    "format_annotated",
    # LAMMPS Indexer
    "build_lammps_index",
    "query_lammps_docs",
    "get_lammps_doc_by_name",
    # Expert Knowledge Indexer
    "build_expert_index",
    "query_expert_knowledge",
    # Code Search Indexer
    "build_package_index",
    "query_package_code",
]
