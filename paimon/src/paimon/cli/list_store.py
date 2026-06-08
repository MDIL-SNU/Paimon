"""List contents of RAG ChromaDB collections."""

from __future__ import annotations

import argparse

import chromadb


def add_subparser(
    subparsers: argparse._SubParsersAction,
) -> None:
    p = subparsers.add_parser(
        "list",
        help="List RAG store contents",
    )
    p.add_argument(
        "collection",
        nargs="?",
        default=None,
        help="Collection name (omit to list all)",
    )
    p.add_argument(
        "--meta",
        action="store_true",
        help="Show metadata keys per item",
    )


def run(args: argparse.Namespace) -> int:
    db = chromadb.PersistentClient(
        path=args.chroma_db_path
    )

    if args.collection is None:
        return _list_collections(db)

    return _list_items(db, args.collection, args.meta)


def _list_collections(
    db: chromadb.ClientAPI,
) -> int:
    collections = db.list_collections()
    for col in collections:
        name = col if isinstance(col, str) else col.name
        count = db.get_collection(name).count()
        print(f"{name}\t{count}")
    return 0


def _list_items(
    db: chromadb.ClientAPI,
    collection_name: str,
    show_meta: bool,
) -> int:
    col = db.get_collection(collection_name)
    result = col.get(include=["metadatas"])

    for doc_id, meta in zip(
        result["ids"], result["metadatas"]
    ):
        if show_meta:
            keys = (
                ",".join(sorted(meta.keys()))
                if meta
                else ""
            )
            print(f"{doc_id}\t{keys}")
        else:
            label = (
                (meta or {}).get("command_name")
                or (meta or {}).get("title")
                or (meta or {}).get("url")
                or (meta or {}).get("doc_id")
                or ""
            )
            print(f"{doc_id}\t{label}")
    return 0
