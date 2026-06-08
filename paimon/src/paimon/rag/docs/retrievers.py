"""Retriever implementations for documentation RAG systems."""

import json
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers.fusion_retriever import (
    QueryFusionRetriever,
    FUSION_MODES,
)
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever

from .models import DocEntry, LAMMPSDoc
from .parser import LAMMPSDocParser
from paimon.util.log import debug


class LAMMPSRetrieverBuilder:
    """Builder for LAMMPS document retrievers.
    Stores all parsed data in ChromaDB metadata
    """

    def __init__(
        self,
        doc_dir: Path,
        embed_model: OpenAIEmbedding,
        chroma_path: str = "./chroma_db",
        collection_name: str = "lammps_docs",
    ):
        """Initialize retriever builder.

        Args:
            doc_dir: Directory containing LAMMPS RST files
            embed_model: Embedding model
            chroma_path: Path to ChromaDB storage
            collection_name: ChromaDB collection name
        """
        self.doc_dir = doc_dir
        self.embed_model = embed_model
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.vector_index = None
        self.db = None
        self.chroma_collection = None

    def build_vector_index(
        self, rst_files: list[Path] | None = None, force_rebuild: bool = False
    ):
        """Build vector index with ChromaDB.
        Stores all parsed data in metadata, eliminating need for document list.

        Args:
            rst_files: List of RST files to index (None = check if exists)
            force_rebuild: Force rebuild even if collection exists
        """
        # Create ChromaDB client
        self.db = chromadb.PersistentClient(path=self.chroma_path)

        # Check if collection exists
        collections = [c.name for c in self.db.list_collections()]
        collection_exists = self.collection_name in collections

        if collection_exists and not force_rebuild:
            debug(f"Loading existing collection: {self.collection_name}")
            self.chroma_collection = self.db.get_collection(self.collection_name)

            # Load existing index
            vector_store = ChromaVectorStore(
                chroma_collection=self.chroma_collection
            )
            self.vector_index = VectorStoreIndex.from_vector_store(
                vector_store, embed_model=self.embed_model
            )
            debug("Vector index loaded from ChromaDB")
            return

        # Build new index
        if collection_exists:
            debug(f"Deleting existing collection: {self.collection_name}")
            self.db.delete_collection(self.collection_name)

        debug(f"Creating new collection: {self.collection_name}")
        self.chroma_collection = self.db.create_collection(self.collection_name)

        # Parse documents and create nodes
        debug("Parsing documents and building index")
        nodes = []

        assert rst_files, "No rst files provided"
        for i, filepath in enumerate(rst_files):
            try:
                # Parse document
                doc = LAMMPSDocParser.parse_rst(filepath)

                # Create indexing text (first paragraph for embedding)
                first_para = (
                    doc.description_paragraphs[0]
                    if doc.description_paragraphs
                    else ""
                )
                index_text = f"Command: {doc.command_name}\n\n{first_para}"

                # Store ALL parsed data in metadata (JSON for lists and dicts)
                metadata = {
                    "filename": str(filepath.name),
                    "filepath": str(filepath),
                    "command_name": doc.command_name,
                    "command_type": doc.command_type,
                    "syntax": doc.syntax,
                    "examples": doc.examples,
                    "description_paragraphs": json.dumps(doc.description_paragraphs),
                    "all_sections": json.dumps(
                        doc.all_sections
                    ),  # NEW: Store all sections
                    "full_content": doc.full_content,  # Store original RST content
                }

                node = TextNode(
                    text=index_text,
                    id_=str(i),
                    metadata=metadata,
                    excluded_embed_metadata_keys=[
                        "syntax",
                        "examples",
                        "description_paragraphs",
                        "all_sections",
                        "full_content",
                        "filepath",
                    ],
                    excluded_llm_metadata_keys=[
                        "syntax",
                        "examples",
                        "description_paragraphs",
                        "all_sections",
                        "full_content",
                        "filepath",
                    ],
                )
                nodes.append(node)

                if (i + 1) % 100 == 0:
                    debug(f"Parsed {i + 1} documents")

            except Exception as e:
                debug(f"Failed to parse {filepath}: {e}")

        debug(f"Parsed {len(nodes)} documents")

        # Dump parsing errors if any
        if LAMMPSDocParser.parsing_errors:
            error_dump_path = Path(self.chroma_path) / "parsing_errors.json"
            LAMMPSDocParser.dump_errors(error_dump_path)
            debug(
                f"{len(LAMMPSDocParser.parsing_errors)} parsing errors "
                f"(see {error_dump_path})"
            )

        # Build index
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.vector_index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
        )

        debug("Vector index built and saved to ChromaDB")

    def get_documents_from_nodes(
        self, nodes: list, deduplicate: bool = True
    ) -> list[LAMMPSDoc]:
        """Reconstruct LAMMPSDoc objects from node metadata.

        Args:
            nodes: Retrieved nodes from ChromaDB

        Returns:
            List of LAMMPSDoc objects
        """
        documents = []
        command_name_set = set()

        for node in nodes:
            metadata = node.metadata
            if deduplicate and metadata["command_name"] in command_name_set:
                continue
            command_name_set.add(metadata["command_name"])

            # Reconstruct LAMMPSDoc from metadata (no parsing needed!)
            doc = LAMMPSDoc(
                filepath=Path(metadata["filepath"]),
                command_name=metadata["command_name"],
                command_type=metadata["command_type"],
                syntax=metadata["syntax"],
                examples=metadata["examples"],
                description_paragraphs=json.loads(
                    metadata["description_paragraphs"]
                ),
                full_content=metadata.get(
                    "full_content", ""
                ),  # Restore original RST content
                all_sections=json.loads(
                    metadata.get("all_sections", "{}")
                ),  # Restore all sections
            )
            documents.append(doc)

        return documents

    def get_vector_retriever(self, top_k: int = 4):
        """Get vector retriever.

        Args:
            top_k: Number of documents to retrieve

        Returns:
            Vector retriever
        """
        if self.vector_index is None:
            raise ValueError(
                "Vector index not built. Call build_vector_index() first."
            )

        return self.vector_index.as_retriever(similarity_top_k=top_k)

    def get_bm25_retriever(self, top_k: int = 10):
        """Get BM25 retriever.

        Note: BM25 requires nodes which are created from ChromaDB.

        Args:
            top_k: Number of documents to retrieve

        Returns:
            BM25 retriever
        """
        if self.chroma_collection is None:
            raise ValueError(
                "Collection not loaded. Call build_vector_index() first."
            )

        # Retrieve all documents from ChromaDB for BM25
        results = self.chroma_collection.get(include=["metadatas", "documents"])

        nodes = []
        for i, (doc_text, metadata) in enumerate(
            zip(results["documents"], results["metadatas"])
        ):
            node = TextNode(text=doc_text, id_=str(i), metadata=metadata)
            nodes.append(node)

        return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)

    def get_hybrid_retriever(self, top_k: int = 4, top_k_candid: int = 40):
        """Get hybrid retriever (vector + BM25). Use RRF.

        Args:
            top_k: Number of documents to retrieve
            top_k_candid: vector & BM25 retriever top_k

        Returns:
            Hybrid fusion retriever
        """

        # RRF candidates num = 40
        vector_retriever = self.get_vector_retriever(top_k_candid)
        bm25_retriever = self.get_bm25_retriever(top_k_candid)

        return QueryFusionRetriever(
            # top_k in this stage m
            [vector_retriever, bm25_retriever],
            similarity_top_k=top_k,
            num_queries=1,  # disable query generation
            mode=FUSION_MODES.RECIPROCAL_RANK,
            use_async=True,
            verbose=False,
        )


class DocRetrieverBuilder:
    """Builder for generic document retrievers.

    Simpler than LAMMPSRetrieverBuilder - works with DocEntry objects
    that have pre-computed embedding keys.
    """

    def __init__(
        self,
        embed_model: OpenAIEmbedding,
        chroma_path: str = "./chroma_db",
        collection_name: str = "docs",
    ):
        self.embed_model = embed_model
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.vector_index = None
        self.db = None
        self.chroma_collection = None

    def build_vector_index(
        self,
        docs: list[DocEntry] | None = None,
        force_rebuild: bool = False,
    ) -> None:
        """Build vector index from DocEntry objects.

        Args:
            docs: List of DocEntry objects with pre-computed embedding_key
            force_rebuild: Force rebuild even if collection exists
        """
        self.db = chromadb.PersistentClient(path=self.chroma_path)

        collections = [c.name for c in self.db.list_collections()]
        collection_exists = self.collection_name in collections

        if collection_exists and not force_rebuild:
            debug(f"Loading existing collection: {self.collection_name}")
            self.chroma_collection = self.db.get_collection(self.collection_name)
            vector_store = ChromaVectorStore(
                chroma_collection=self.chroma_collection
            )
            self.vector_index = VectorStoreIndex.from_vector_store(
                vector_store, embed_model=self.embed_model
            )
            debug("Vector index loaded from ChromaDB")
            return

        if collection_exists:
            debug(f"Deleting existing collection: {self.collection_name}")
            self.db.delete_collection(self.collection_name)

        debug(f"Creating new collection: {self.collection_name}")
        self.chroma_collection = self.db.create_collection(self.collection_name)

        if not docs:
            debug("No documents provided, creating empty index")
            vector_store = ChromaVectorStore(
                chroma_collection=self.chroma_collection
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.vector_index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                embed_model=self.embed_model,
            )
            return

        debug(f"Building index from {len(docs)} documents")
        nodes = []

        for i, doc in enumerate(docs):
            metadata = {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "content": doc.content,
                "filepath": str(doc.filepath),
                "doc_type": doc.doc_type,
            }

            node = TextNode(
                text=doc.embedding_key,
                id_=str(i),
                metadata=metadata,
                excluded_embed_metadata_keys=["content", "filepath"],
                excluded_llm_metadata_keys=["content", "filepath"],
            )
            nodes.append(node)

        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.vector_index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
        )

        debug(f"Vector index built with {len(nodes)} documents")

    def get_documents_from_nodes(
        self,
        nodes: list,
        deduplicate: bool = True,
    ) -> list[DocEntry]:
        """Reconstruct DocEntry objects from node metadata."""
        documents = []
        doc_id_set = set()

        for node in nodes:
            metadata = node.metadata
            if deduplicate and metadata["doc_id"] in doc_id_set:
                continue
            doc_id_set.add(metadata["doc_id"])

            doc = DocEntry(
                doc_id=metadata["doc_id"],
                title=metadata["title"],
                content=metadata["content"],
                embedding_key=node.text,
                filepath=Path(metadata["filepath"]),
                doc_type=metadata.get("doc_type", "generic"),
            )
            documents.append(doc)

        return documents

    def get_vector_retriever(self, top_k: int = 4):
        """Get vector retriever."""
        if self.vector_index is None:
            raise ValueError(
                "Vector index not built. Call build_vector_index() first."
            )
        return self.vector_index.as_retriever(similarity_top_k=top_k)
