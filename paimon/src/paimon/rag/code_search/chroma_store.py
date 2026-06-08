"""ChromaDB storage for extracted code blocks using llama-index.

Uses llama-index's ChromaVectorStore for consistency with LAMMPS RAG.
Embedding handled by llama-index's VectorStoreIndex.

Mapping from CASCADE's Supabase extracted_code table:
    url           -> metadata["url"]
    code_text     -> TextNode.text
    summary       -> metadata["summary"]
    context_before-> metadata["context_before"]
    context_after -> metadata["context_after"]
    code_type     -> metadata["code_type"]
    language      -> metadata["language"]
    index         -> metadata["index"]
    extraction_method -> metadata["extraction_method"]
    embedding     -> VectorStoreIndex handles automatically
"""

from __future__ import annotations

import hashlib
from typing import Any

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from paimon import cfg


class CodeSearchStore:
    """llama-index ChromaVectorStore-based storage for extracted code blocks."""

    def __init__(
        self,
        chroma_db_path: str | None = None,
        collection_name: str | None = None,
        embed_model: str | None = None,
    ):
        """Initialize the ChromaDB store with llama-index.

        Args:
            chroma_db_path: Path to shared ChromaDB directory (default from config)
            collection_name: Collection name (default: "code_search")
            embed_model: Embedding model name (default from config)
        """
        self.chroma_db_path = chroma_db_path or cfg.rag_config.chroma_db_path
        self.collection_name = collection_name or "code_search"
        self.embed_model_name = embed_model or cfg.rag_config.embed_model

        # Initialize embedding model (llama-index)
        self.embed_model = OpenAIEmbedding(model=self.embed_model_name)

        # Initialize ChromaDB client
        self.db = chromadb.PersistentClient(path=self.chroma_db_path)

        # Check if collection exists
        collections = [c.name for c in self.db.list_collections()]
        if self.collection_name in collections:
            self.chroma_collection = self.db.get_collection(self.collection_name)
        else:
            self.chroma_collection = self.db.create_collection(
                self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        # Create vector store wrapper
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.chroma_collection
        )

        # Load or create index
        if self.chroma_collection.count() > 0:
            # Load existing index
            self.vector_index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model,
            )
        else:
            # Create empty index (will be populated on first save)
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self.vector_index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                embed_model=self.embed_model,
            )

    def _generate_id(self, url: str, index: int) -> str:
        """Generate a unique ID for a code block."""
        content = f"{url}:{index}"
        return hashlib.md5(content.encode()).hexdigest()

    async def check_exists(self, url: str) -> bool:
        """Check if code has already been extracted from a given URL.

        Args:
            url: URL to check

        Returns:
            True if code exists for this URL, False otherwise
        """
        try:
            results = self.chroma_collection.get(
                where={"url": url},
                limit=1,
            )
            return len(results["ids"]) > 0
        except Exception:
            return False

    async def save(
        self,
        url: str,
        code_blocks: list[dict[str, Any]],
        extraction_method: str = "single_page",
    ) -> bool:
        """Save extracted code blocks to ChromaDB via llama-index.

        Args:
            url: Source URL of the code
            code_blocks: List of extracted code blocks
            extraction_method: Method used for extraction

        Returns:
            Boolean indicating success
        """
        if not code_blocks:
            return True

        try:
            nodes: list[TextNode] = []

            for block in code_blocks:
                idx = block.get("index", 0)
                block_id = self._generate_id(url, idx)

                # Extract components
                code_text = block.get("code", "")
                summary = block.get("summary", "")
                context_before = block.get("context_before", "")[:1000]
                context_after = block.get("context_after", "")[:1000]

                # Combine for embedding (matching CASCADE's approach)
                # This improves semantic search by including explanatory context
                embedding_text = (
                    f"Code: {code_text}\n"
                    f"Summary: {summary}\n"
                    f"Context: {context_before} {context_after}"
                )

                # Metadata for retrieval and filtering
                metadata = {
                    "url": url,
                    "summary": summary,
                    "context_before": context_before,
                    "context_after": context_after,
                    "code_type": block.get("type", ""),
                    "language": block.get("language", ""),
                    "index": idx,
                    "extraction_method": extraction_method,
                    "code_only": code_text,  # Store raw code separately
                }

                # Create TextNode with combined embedding text
                node = TextNode(
                    text=embedding_text,
                    id_=block_id,
                    metadata=metadata,
                    # All fields already in embedding text, but keep for retrieval
                    excluded_embed_metadata_keys=[
                        "context_before",
                        "context_after",
                        "summary",
                        "code_only",
                    ],
                    excluded_llm_metadata_keys=[
                        "context_before",
                        "context_after",
                    ],
                )
                nodes.append(node)

            # Insert nodes into index
            # llama-index VectorStoreIndex automatically handles embeddings
            for node in nodes:
                self.vector_index.insert_nodes([node])

            return True

        except Exception as e:
            print(f"Error saving extracted code to ChromaDB: {e}")
            return False

    async def get(self, url: str) -> list[dict[str, Any]] | None:
        """Retrieve extracted code from ChromaDB for a given URL.

        Args:
            url: URL to retrieve code for

        Returns:
            List of code blocks, or None if not found
        """
        try:
            results = self.chroma_collection.get(
                where={"url": url},
                include=["documents", "metadatas"],
            )

            if not results["ids"]:
                return None

            code_blocks: list[dict[str, Any]] = []
            for i, doc_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]

                code_block = {
                    "code": metadata.get("code_only", ""),  # Return raw code
                    "summary": metadata.get("summary", ""),
                    "context_before": metadata.get("context_before", ""),
                    "context_after": metadata.get("context_after", ""),
                    "type": metadata.get("code_type", ""),
                    "language": metadata.get("language", ""),
                    "index": metadata.get("index", 0),
                    "source_url": metadata.get("url", ""),
                }
                code_blocks.append(code_block)

            # Sort by index
            code_blocks.sort(key=lambda x: x.get("index", 0))
            return code_blocks

        except Exception as e:
            print(f"Error retrieving extracted code from ChromaDB: {e}")
            return None

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for relevant code blocks using semantic similarity via llama-index.

        Args:
            query: Search query
            top_k: Number of matches to return
            filter_metadata: Optional metadata filters (e.g., {"language": "python"})

        Returns:
            List of relevant code blocks with similarity scores
        """
        try:
            # Use llama-index retriever for semantic search
            retriever = self.vector_index.as_retriever(
                similarity_top_k=top_k,
                # filters parameter would go here if llama-index supports it
            )

            # Retrieve nodes
            nodes = retriever.retrieve(query)

            code_blocks: list[dict[str, Any]] = []

            for node_with_score in nodes:
                node = node_with_score.node
                score = node_with_score.score  # Similarity score (higher is better)
                metadata = node.metadata

                # Apply metadata filters manually if specified
                if filter_metadata:
                    match = all(
                        metadata.get(k) == v for k, v in filter_metadata.items()
                    )
                    if not match:
                        continue

                code_block = {
                    "code": metadata.get("code_only", ""),  # Return raw code
                    "summary": metadata.get("summary", ""),
                    "context_before": metadata.get("context_before", ""),
                    "context_after": metadata.get("context_after", ""),
                    "type": metadata.get("code_type", ""),
                    "language": metadata.get("language", ""),
                    "index": metadata.get("index", 0),
                    "source_url": metadata.get("url", ""),
                    "similarity_score": score if score else 0.0,
                }
                code_blocks.append(code_block)

            return code_blocks

        except Exception as e:
            print(f"Error searching code blocks: {e}")
            return []

    async def clear(self, url: str | None = None) -> int:
        """Clear extracted code from the store.

        Args:
            url: If provided, only clear entries for this URL.
                 If None, clear all entries.

        Returns:
            Number of entries cleared
        """
        try:
            if url:
                # Get IDs for this URL
                results = self.chroma_collection.get(
                    where={"url": url},
                )
                if results["ids"]:
                    # Delete from vector index
                    for doc_id in results["ids"]:
                        self.vector_index.delete_ref_doc(doc_id, delete_from_docstore=True)
                    return len(results["ids"])
                return 0
            else:
                # Clear all - delete and recreate collection
                count = self.chroma_collection.count()
                self.db.delete_collection(self.collection_name)

                # Recreate collection
                self.chroma_collection = self.db.create_collection(
                    self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )

                # Recreate vector store and index
                self.vector_store = ChromaVectorStore(
                    chroma_collection=self.chroma_collection
                )
                storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store
                )
                self.vector_index = VectorStoreIndex(
                    nodes=[],
                    storage_context=storage_context,
                    embed_model=self.embed_model,
                )

                return count

        except Exception as e:
            print(f"Error clearing extracted code: {e}")
            return 0

    def count(self) -> int:
        """Return the total number of stored code blocks."""
        return self.chroma_collection.count()

    def list_urls(self) -> list[str]:
        """List all unique URLs in the store."""
        try:
            # Get all metadata
            results = self.chroma_collection.get(include=["metadatas"])
            urls = set()
            for metadata in results["metadatas"]:
                if metadata.get("url"):
                    urls.add(metadata["url"])
            return sorted(urls)
        except Exception:
            return []


# Module-level singleton instance
_store: CodeSearchStore | None = None


def get_store() -> CodeSearchStore:
    """Get the singleton CodeSearchStore instance.

    Uses configuration from paimon.cfg.rag_config.

    Returns:
        The CodeSearchStore instance
    """
    global _store
    if _store is None:
        _store = CodeSearchStore()
    return _store


def reset_store() -> None:
    """Reset the singleton store instance."""
    global _store
    _store = None


__all__ = [
    "CodeSearchStore",
    "get_store",
    "reset_store",
]
