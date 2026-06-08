"""Paimon CLI.

Usage:
    paimon code-search {build,query} ...
    paimon docs {build,query} {lammps,expert} ...
    paimon extract <paper> <task> ...
    paimon list [collection] [--meta]
"""

import argparse
import os
import sys

from paimon import cfg
from paimon.cli import code_search, docs, extract, list_store


def main() -> int:
    config_path = os.getenv(
        "PAIMON_YAML", os.path.expanduser("~/.config/paimon.yaml")
    )

    description = f"""Paimon CLI

Configuration: {config_path}
ChromaDB: {cfg.rag_config.chroma_db_path}
"""

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chroma-db-path",
        default=cfg.rag_config.chroma_db_path,
        help=f"Shared ChromaDB path (default: {cfg.rag_config.chroma_db_path})",
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")

    code_search.add_subparser(subparsers)
    docs.add_subparser(subparsers)
    extract.add_subparser(subparsers)
    list_store.add_subparser(subparsers)

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        return 1

    if args.subcommand == "code-search":
        return code_search.run(args)
    elif args.subcommand == "docs":
        return docs.run(args)
    elif args.subcommand == "extract":
        return extract.run(args)
    elif args.subcommand == "list":
        return list_store.run(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
