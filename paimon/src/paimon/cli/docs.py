"""CLI for documentation RAG (LAMMPS and expert knowledge)."""

import argparse
import asyncio
import json
import sys

from paimon.rag.indexer import (
    build_lammps_index,
    query_lammps_docs,
    build_expert_index,
    query_expert_knowledge,
)


def _print_build_result(stats: dict, output: str | None, source: str) -> None:
    print("\n" + "=" * 70)
    print(f"BUILD COMPLETE ({source})")
    print("=" * 70)
    print(f"Embedding strategy: {stats.get('embedding_strategy', 'N/A')}")

    if "total_files" in stats:
        processed = stats.get("processed_files", stats.get("indexed_docs", 0))
        print(f"Files processed: {processed}/{stats['total_files']}")

    if "indexed_docs" in stats:
        print(f"Documents indexed: {stats['indexed_docs']}")

    if "failed_files" in stats and stats["failed_files"]:
        print(f"Failed files: {len(stats['failed_files'])}")
        for err in stats["failed_files"][:5]:
            print(f"  - {err}")
        if len(stats["failed_files"]) > 5:
            print(f"  ... and {len(stats['failed_files']) - 5} more")

    if output:
        with open(output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to: {output}")


def _print_lammps_query_result(results: list[dict], output: str | None) -> None:
    print("\n" + "=" * 70)
    print(f"FOUND {len(results)} RESULTS")
    print("=" * 70)

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}")
        print("-" * 70)
        print(f"Command: {result['command_name']}")
        print("\nSyntax:")
        syntax = result.get("syntax", "")[:200]
        print(f"  {syntax}...")
        print("\nDescription (first paragraph):")
        desc = result.get("description_paragraphs", [""])[0]
        print(f"  {desc[:300]}...")

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output}")


def _print_expert_query_result(results: list[dict], output: str | None) -> None:
    print("\n" + "=" * 70)
    print(f"FOUND {len(results)} RESULTS")
    print("=" * 70)

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}")
        print("-" * 70)
        print(f"Title: {result['title']}")
        print(f"File: {result['doc_id']}")
        print("\nContent (preview):")
        content = result.get("content", "")[:500]
        print(f"  {content}...")

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output}")


async def _run_build_lammps(args: argparse.Namespace) -> int:
    print("Building LAMMPS documentation index")
    print(f"Doc directory: {args.doc_dir or 'using config default'}")

    try:
        stats = await build_lammps_index(
            doc_dir=args.doc_dir,
            chroma_db_path=args.chroma_db_path,
            embed_model=args.embed_model,
            force_rebuild=args.force_rebuild,
            embedding_strategy=args.embedding_strategy,
        )
        _print_build_result(stats, args.output, "lammps")
        return 0
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


async def _run_build_expert(args: argparse.Namespace) -> int:
    print("Building expert knowledge index")
    print(f"Knowledge dir: {args.knowledge_dir or 'using default'}")

    use_summary = args.embedding_strategy == "summarize"

    try:
        stats = await build_expert_index(
            knowledge_dir=args.knowledge_dir,
            chroma_db_path=args.chroma_db_path,
            embed_model=args.embed_model,
            force_rebuild=args.force_rebuild,
            use_summary=use_summary,
        )
        _print_build_result(stats, args.output, "expert")
        return 0
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


async def _run_query_lammps(args: argparse.Namespace) -> int:
    print(f"Query (lammps): {args.query}")

    try:
        results = await query_lammps_docs(
            query=args.query,
            top_k=args.top_k,
            doc_dir=args.doc_dir,
            chroma_db_path=args.chroma_db_path,
            use_hybrid=args.use_hybrid,
        )
        _print_lammps_query_result(results, args.output)
        return 0
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


async def _run_query_expert(args: argparse.Namespace) -> int:
    print(f"Query (expert): {args.query}")

    try:
        results = await query_expert_knowledge(
            query=args.query,
            top_k=args.top_k,
            knowledge_dir=args.knowledge_dir,
            chroma_db_path=args.chroma_db_path,
        )
        _print_expert_query_result(results, args.output)
        return 0
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add docs subparser to parent parser."""
    parser = subparsers.add_parser(
        "docs", help="Documentation RAG (LAMMPS and expert knowledge)"
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # build
    build_p = sub.add_parser("build", help="Build documentation index")
    build_p.add_argument(
        "source",
        choices=["lammps", "expert"],
        help="Documentation source type",
    )
    build_p.add_argument("--doc-dir", help="Directory containing LAMMPS RST files")
    build_p.add_argument("--knowledge-dir", help="Directory containing .txt files")
    build_p.add_argument("--embed-model", help="OpenAI embedding model")
    build_p.add_argument(
        "--embedding-strategy",
        choices=["first_paragraph", "summarize"],
        default="summarize",
        help="Embedding strategy (default: summarize)",
    )
    build_p.add_argument(
        "--force-rebuild", action="store_true", help="Clear existing data"
    )
    build_p.add_argument("--output", "-o", help="Save build stats to JSON file")

    # query
    query_p = sub.add_parser("query", help="Query documentation index")
    query_p.add_argument(
        "source",
        choices=["lammps", "expert"],
        help="Documentation source type",
    )
    query_p.add_argument("query", help="Search query")
    query_p.add_argument(
        "--top-k", type=int, default=4, help="Number of results (default: 4)"
    )
    query_p.add_argument("--doc-dir", help="Directory containing LAMMPS RST files")
    query_p.add_argument("--knowledge-dir", help="Directory containing .txt files")
    query_p.add_argument(
        "--use-hybrid",
        action="store_true",
        help="Use hybrid search (LAMMPS only)",
    )
    query_p.add_argument("--output", "-o", help="Save results to JSON file")


def run(args: argparse.Namespace) -> int:
    """Run docs command."""
    if not args.command:
        print("Usage: paimon docs {build,query} ...", file=sys.stderr)
        return 1

    if args.command == "build":
        if args.source == "lammps":
            return asyncio.run(_run_build_lammps(args))
        elif args.source == "expert":
            return asyncio.run(_run_build_expert(args))
    elif args.command == "query":
        if args.source == "lammps":
            return asyncio.run(_run_query_lammps(args))
        elif args.source == "expert":
            return asyncio.run(_run_query_expert(args))

    return 1
