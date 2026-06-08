"""Local HTML indexer for building package-specific vector stores.

Pre-builds vector stores from local HTML documentation files instead of
web crawling. Designed for offline use and faster indexing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from paimon import cfg
from paimon.rag.code_search.chroma_store import CodeSearchStore
from paimon.rag.code_search.extractors import (
    extract_readthedocs_code_blocks,
    extract_smart_context_before,
    extract_smart_context_after,
)


@dataclass
class LocalIndexConfig:
    """Configuration for local HTML indexing."""

    pkg_name: str
    html_root: Path
    chroma_db_path: str | None = None
    collection_name: str | None = None
    embed_model: str = "text-embedding-3-small"
    min_code_length: int = 3
    max_context_chars: int = 1000


class LocalPackageIndexer:
    """Indexes local HTML documentation into package-specific vector stores."""

    def __init__(self, config: LocalIndexConfig):
        """Initialize indexer with configuration.

        Args:
            config: Configuration for local indexing
        """
        self.config = config

        # Use shared ChromaDB path with package-specific collection
        chroma_path = config.chroma_db_path or cfg.rag_config.chroma_db_path
        collection_name = config.collection_name or f"code_{config.pkg_name}"

        self.store = CodeSearchStore(
            chroma_db_path=chroma_path,
            collection_name=collection_name,
            embed_model=config.embed_model,
        )

    async def index_package(self, force_rebuild: bool = False) -> dict[str, any]:
        """Build vector store from local HTML files.

        Args:
            force_rebuild: If True, clear existing data before indexing

        Returns:
            Indexing statistics
        """
        print(f"Indexing {self.config.pkg_name} from {self.config.html_root}")

        # Clear if rebuilding
        if force_rebuild:
            print("  Clearing existing data...")
            await self.store.clear()

        # Find all HTML files
        html_files = list(self.config.html_root.rglob("*.html"))
        print(f"  Found {len(html_files)} HTML files")

        # Filter out non-documentation files
        html_files = [
            f for f in html_files
            if not any(
                part in f.parts
                for part in ["_static", "_sources", "_modules", "_downloads"]
            )
        ]
        print(f"  Filtered to {len(html_files)} documentation files")

        stats = {
            "pkg_name": self.config.pkg_name,
            "total_files": len(html_files),
            "processed_files": 0,
            "total_blocks": 0,
            "failed_files": [],
        }

        # Process each file
        for i, html_file in enumerate(html_files, 1):
            try:
                rel_path = html_file.relative_to(self.config.html_root)
                print(f"  [{i}/{len(html_files)}] Processing {rel_path}")

                blocks = await self._extract_from_file(html_file, rel_path)

                if blocks:
                    # Use relative path as "URL" for local files
                    pseudo_url = f"local://{self.config.pkg_name}/{rel_path}"
                    await self.store.save(pseudo_url, blocks, "local_html")

                    stats["processed_files"] += 1
                    stats["total_blocks"] += len(blocks)
                    print(f"    Saved {len(blocks)} code blocks")
                else:
                    print(f"    No code blocks found")

            except Exception as e:
                print(f"    ERROR: {str(e)}")
                stats["failed_files"].append({
                    "file": str(rel_path),
                    "error": str(e),
                })

        print(f"\nIndexing complete:")
        print(f"  Files processed: {stats['processed_files']}/{stats['total_files']}")
        print(f"  Total code blocks: {stats['total_blocks']}")
        print(f"  Failed files: {len(stats['failed_files'])}")

        return stats

    async def _extract_from_file(
        self,
        html_file: Path,
        rel_path: Path,
    ) -> list[dict[str, any]]:
        """Extract code blocks from a single HTML file.

        Args:
            html_file: Path to HTML file
            rel_path: Relative path from HTML root

        Returns:
            List of code blocks with metadata
        """
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Extract code blocks
        blocks = extract_readthedocs_code_blocks(
            html_content,
            min_length=self.config.min_code_length,
        )

        # Add context and format for storage
        formatted_blocks = []
        for i, block in enumerate(blocks):
            formatted_block = {
                "code": block.get("code", ""),
                "summary": "",  # No LLM summary for batch indexing
                "context_before": block.get("context_before", "")[:self.config.max_context_chars],
                "context_after": block.get("context_after", "")[:self.config.max_context_chars],
                "type": block.get("type", "code"),
                "language": block.get("language", "python"),
                "index": i,
            }
            formatted_blocks.append(formatted_block)

        return formatted_blocks


async def build_package_index(
    pkg_name: str,
    html_root: str | Path,
    chroma_db_path: str | None = None,
    force_rebuild: bool = False,
) -> dict[str, any]:
    """Build vector store for a package from local HTML files.

    Args:
        pkg_name: Package name (e.g., "ase", "numpy")
        html_root: Root directory containing HTML documentation
        chroma_db_path: Optional custom ChromaDB path
        force_rebuild: If True, clear existing data before indexing

    Returns:
        Indexing statistics
    """
    config = LocalIndexConfig(
        pkg_name=pkg_name,
        html_root=Path(html_root),
        chroma_db_path=chroma_db_path,
    )

    indexer = LocalPackageIndexer(config)
    return await indexer.index_package(force_rebuild=force_rebuild)


async def query_package_code(
    pkg_name: str,
    query: str,
    top_k: int = 5,
    chroma_db_path: str | None = None,
) -> list[dict[str, any]]:
    """Query pre-built package vector store for code examples.

    Args:
        pkg_name: Package name (e.g., "ase", "numpy")
        query: Search query
        top_k: Number of results to return
        chroma_db_path: Optional custom ChromaDB path (default: shared DB)

    Returns:
        List of matching code blocks with metadata
    """
    chroma_path = chroma_db_path or cfg.rag_config.chroma_db_path
    collection_name = f"code_{pkg_name}"

    store = CodeSearchStore(
        chroma_db_path=chroma_path,
        collection_name=collection_name,
    )
    results = await store.search(query=query, top_k=top_k)

    return results
