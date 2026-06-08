"""Documentation indexer with CLI interface.

Provides utilities for building and querying documentation indexes:
- LAMMPS: RST documentation with first-paragraph embedding
- Expert: Plain text files with LLM-generated summary embedding

All RAG systems use a single shared ChromaDB database with separate collections.
"""

from pathlib import Path

from paimon.rag.docs import LAMMPSRAGSystem, ExpertRAGSystem


async def build_lammps_index(
    doc_dir: str | Path | None = None,
    chroma_db_path: str | None = None,
    embed_model: str | None = None,
    force_rebuild: bool = False,
    embedding_strategy: str = "first_paragraph",
) -> dict[str, any]:
    """Build LAMMPS documentation index from RST files.

    Args:
        doc_dir: Directory containing LAMMPS RST files (uses config if None)
        chroma_db_path: Shared ChromaDB path (uses config if None)
        embed_model: Embedding model name (uses config if None)
        force_rebuild: Force rebuild even if index exists
        embedding_strategy: Strategy for embedding (currently only "first_paragraph")

    Returns:
        Build statistics dictionary
    """
    doc_dir_path = Path(doc_dir) if doc_dir else None

    rag_system = LAMMPSRAGSystem(
        doc_dir=doc_dir_path,
        chroma_path=chroma_db_path,
        embed_model=embed_model,
        force_rebuild=force_rebuild,
    )

    # Build index
    rag_system.build_index()

    # Get statistics
    rst_files = list(rag_system.doc_dir.glob("*.rst"))
    collection = rag_system.retriever_builder.chroma_collection

    if collection:
        total_docs = collection.count()
    else:
        total_docs = 0

    # Check parsing errors
    from paimon.rag.docs.parser import LAMMPSDocParser

    failed_files = [
        {"file": str(err.filepath.name), "error": err.error}
        for err in LAMMPSDocParser.parsing_errors
    ]

    stats = {
        "embedding_strategy": embedding_strategy,
        "total_files": len(rst_files),
        "processed_files": total_docs,
        "failed_files": failed_files,
        "chroma_db_path": rag_system.chroma_path,
        "doc_dir": str(rag_system.doc_dir),
    }

    return stats


async def query_lammps_docs(
    query: str,
    top_k: int = 4,
    doc_dir: str | Path | None = None,
    chroma_db_path: str | None = None,
    use_hybrid: bool | None = None,
) -> list[dict[str, any]]:
    """Query LAMMPS documentation index.

    Args:
        query: Search query
        top_k: Number of results to return
        doc_dir: Directory containing LAMMPS RST files (uses config if None)
        chroma_db_path: Shared ChromaDB path (uses config if None)
        use_hybrid: Use hybrid search (uses config if None)

    Returns:
        List of retrieval results with metadata
    """
    doc_dir_path = Path(doc_dir) if doc_dir else None

    rag_system = LAMMPSRAGSystem(
        doc_dir=doc_dir_path,
        chroma_path=chroma_db_path,
        use_hybrid=use_hybrid,
    )

    # Build index (loads existing if available)
    rag_system.build_index()

    # Retrieve results
    result, candidates = await rag_system.retrieve(query, top_k_stage1=top_k)

    # Format results for CLI output
    results = []
    if result:
        results.append(
            {
                "command_name": result.command_name,
                "syntax": result.syntax,
                "examples": result.examples,
                "description_paragraphs": result.doc_raw.description_paragraphs,
                "quoted_description": result.quoted_description,
                "all_sections": result.doc_raw.all_sections,
            }
        )

    # Add candidates as additional context
    for candidate in candidates[1:]:  # Skip first (already in result)
        results.append(
            {
                "command_name": candidate.command_name,
                "syntax": candidate.syntax,
                "examples": candidate.examples,
                "description_paragraphs": candidate.description_paragraphs,
                "quoted_description": "\n\n".join(candidate.description_paragraphs),
                "all_sections": candidate.all_sections,
            }
        )

    return results[:top_k]


def get_lammps_doc_by_name(
    command_name: str,
    doc_dir: str | Path | None = None,
    chroma_db_path: str | None = None,
) -> dict[str, any] | None:
    """Retrieve LAMMPS documentation by exact command name.

    Args:
        command_name: Exact command name (e.g., "fix_npt", "compute_msd")
        doc_dir: Directory containing LAMMPS RST files (uses config if None)
        chroma_db_path: Shared ChromaDB path (uses config if None)

    Returns:
        Document dictionary if found, None otherwise
    """
    doc_dir_path = Path(doc_dir) if doc_dir else None

    rag_system = LAMMPSRAGSystem(
        doc_dir=doc_dir_path,
        chroma_path=chroma_db_path,
    )

    # Build index (loads existing if available)
    rag_system.build_index()

    # Get document by name
    result = rag_system.get_document_by_name(command_name)

    if result:
        return {
            "command_name": result.command_name,
            "syntax": result.syntax,
            "examples": result.examples,
            "description_paragraphs": result.doc_raw.description_paragraphs,
            "quoted_description": result.quoted_description,
            "all_sections": result.doc_raw.all_sections,
        }

    return None


async def build_expert_index(
    knowledge_dir: str | Path | None = None,
    chroma_db_path: str | None = None,
    embed_model: str | None = None,
    force_rebuild: bool = False,
    use_summary: bool = True,
) -> dict:
    """Build expert knowledge index from text files.

    Args:
        knowledge_dir: Directory containing .txt files (default: knowledge/expert/)
        chroma_db_path: Shared ChromaDB path (uses config if None)
        embed_model: Embedding model name (uses config if None)
        force_rebuild: Force rebuild even if index exists
        use_summary: Use LLM summary for embedding (else first paragraph)

    Returns:
        Build statistics dictionary
    """
    knowledge_dir_path = Path(knowledge_dir) if knowledge_dir else None

    rag_system = ExpertRAGSystem(
        knowledge_dir=knowledge_dir_path,
        embed_model=embed_model,
        chroma_path=chroma_db_path,
        use_summary=use_summary,
        force_rebuild=force_rebuild,
    )

    stats = await rag_system.build_index()
    stats["embedding_strategy"] = "summarize" if use_summary else "first_paragraph"
    stats["chroma_db_path"] = rag_system.chroma_path

    return stats


async def query_expert_knowledge(
    query: str,
    top_k: int = 3,
    knowledge_dir: str | Path | None = None,
    chroma_db_path: str | None = None,
) -> list[dict]:
    """Query expert knowledge index.

    Args:
        query: Search query
        top_k: Number of results to return
        knowledge_dir: Directory containing .txt files
        chroma_db_path: Shared ChromaDB path

    Returns:
        List of matching documents with content
    """
    knowledge_dir_path = Path(knowledge_dir) if knowledge_dir else None

    rag_system = ExpertRAGSystem(
        knowledge_dir=knowledge_dir_path,
        chroma_path=chroma_db_path,
        use_summary=False,  # Don't regenerate summaries for query
        force_rebuild=False,
    )

    # Load existing index
    await rag_system.build_index()

    # Retrieve results
    docs = await rag_system.retrieve_all(query, top_k=top_k)

    return [
        {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "content": doc.content,
            "filepath": str(doc.filepath),
        }
        for doc in docs
    ]
