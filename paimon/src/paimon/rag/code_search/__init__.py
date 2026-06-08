"""Code search RAG module for URL-based HTML documentation extraction.

This module provides:
- Web crawling and code extraction from documentation URLs
- ChromaDB-based vector storage for extracted code blocks
- Semantic search over cached code examples
"""

from paimon.rag.code_search.chroma_store import (
    CodeSearchStore,
    get_store,
    reset_store,
)
from paimon.rag.code_search.crawler import (
    create_crawler,
    close_crawler,
    extract_code_single_page,
    extract_code_smart_crawl,
    validate_and_normalize_url,
)
from paimon.rag.code_search.extractors import (
    detect_content_type_and_source,
    extract_code_blocks,
)

__all__ = [
    # Store
    "CodeSearchStore",
    "get_store",
    "reset_store",
    # Crawler
    "create_crawler",
    "close_crawler",
    "extract_code_single_page",
    "extract_code_smart_crawl",
    "validate_and_normalize_url",
    # Extractors
    "detect_content_type_and_source",
    "extract_code_blocks",
]
