"""CLI for Code Search RAG."""

import argparse
import asyncio
import json
import sys

from paimon.rag.code_search.local_indexer import (
    build_package_index,
    query_package_code,
)


def _print_build_result(stats: dict, output: str | None) -> None:
    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    print(f"Package: {stats['pkg_name']}")
    print(f"Files processed: {stats['processed_files']}/{stats['total_files']}")
    print(f"Total code blocks: {stats['total_blocks']}")
    print(f"Failed files: {len(stats['failed_files'])}")

    if stats["failed_files"]:
        print("\nFailed files:")
        for fail in stats["failed_files"]:
            print(f"  - {fail['file']}: {fail['error']}")

    if output:
        with open(output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to: {output}")


def _print_query_result(results: list[dict], output: str | None) -> None:
    print("\n" + "=" * 70)
    print(f"FOUND {len(results)} RESULTS")
    print("=" * 70)

    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (similarity: {result['similarity_score']:.3f})")
        print("-" * 70)
        print(f"Source: {result.get('source_url', 'N/A')}")
        print(f"Language: {result.get('language', 'N/A')}")
        print("\nCode:")
        code = result.get("code", "")
        lines = code.split("\n")
        for line in lines[:10]:
            print(f"  {line}")
        if len(lines) > 10:
            print(f"  ... ({len(lines) - 10} more lines)")

        if result.get("context_before"):
            print(f"\nContext: {result['context_before'][:100]}...")

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output}")


async def _run_build(args: argparse.Namespace) -> int:
    print(f"Building Code Search index for: {args.pkg_name}")
    print(f"HTML root: {args.html_root}")
    if args.chroma_db_path:
        print(f"ChromaDB path: {args.chroma_db_path}")

    try:
        stats = await build_package_index(
            pkg_name=args.pkg_name,
            html_root=args.html_root,
            chroma_db_path=args.chroma_db_path,
            force_rebuild=args.force_rebuild,
        )
        _print_build_result(stats, args.output)
        return 0
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


async def _run_query(args: argparse.Namespace) -> int:
    print(f"Querying package: {args.pkg_name}")
    print(f"Query: {args.query}")

    try:
        results = await query_package_code(
            pkg_name=args.pkg_name,
            query=args.query,
            top_k=args.top_k,
            chroma_db_path=args.chroma_db_path,
        )
        _print_query_result(results, args.output)
        return 0
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add code-search subparser to parent parser."""
    parser = subparsers.add_parser(
        "code-search", help="Code Search RAG (HTML documentation)"
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # build
    build_p = sub.add_parser("build", help="Build package index from HTML files")
    build_p.add_argument("pkg_name", help="Package name (e.g., ase, numpy)")
    build_p.add_argument(
        "html_root", help="Root directory containing HTML documentation"
    )
    build_p.add_argument(
        "--force-rebuild", action="store_true", help="Clear existing data"
    )
    build_p.add_argument("--output", "-o", help="Save build stats to JSON file")

    # query
    query_p = sub.add_parser("query", help="Query pre-built package index")
    query_p.add_argument("pkg_name", help="Package name (e.g., ase, numpy)")
    query_p.add_argument("query", help="Search query")
    query_p.add_argument(
        "--top-k", type=int, default=5, help="Number of results (default: 5)"
    )
    query_p.add_argument("--output", "-o", help="Save results to JSON file")


def run(args: argparse.Namespace) -> int:
    """Run code-search command."""
    if not args.command:
        print("Usage: paimon code-search {build,query} ...", file=sys.stderr)
        return 1

    if args.command == "build":
        return asyncio.run(_run_build(args))
    elif args.command == "query":
        return asyncio.run(_run_query(args))

    return 1
