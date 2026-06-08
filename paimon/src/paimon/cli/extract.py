"""CLI for paper methodology extraction."""

import argparse
import sys
from pathlib import Path

from paimon import cfg
from paimon.extraction import extract_methodology


def _print_result(result, output: str | None, turns: bool) -> None:
    if turns:
        print("\n" + "=" * 70)
        print("TURN 1: Extraction")
        print("=" * 70)
        print(result.turn1_response)

        print("\n" + "=" * 70)
        print("TURN 2: Additions")
        print("=" * 70)
        print(result.turn2_response)

        print("\n" + "=" * 70)
        print("TURN 3: Protocol")
        print("=" * 70)
        print(result.protocol)
    else:
        print(result.protocol)

    if output:
        with open(output, "w") as f:
            f.write(result.protocol)
        print(f"\nProtocol saved to: {output}", file=sys.stderr)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add extract subparser to parent parser."""
    parser = subparsers.add_parser(
        "extract",
        help="Extract methodology from research papers",
        description=f"""Extract methodology from research papers

Uses OpenAI's responses API for multi-turn extraction:
  1. Extract all methodological details
  2. Review for missed procedural information
  3. Synthesize into a self-contained simulation protocol

Default model: {cfg.base_reasoning_llm}
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paper", help="Path to the main paper PDF")
    parser.add_argument("task", help="What to extract (e.g., 'MD simulation of X')")
    parser.add_argument(
        "--supporting-info", "-s", help="Path to supporting information PDF"
    )
    parser.add_argument(
        "--model",
        "-m",
        default=cfg.base_reasoning_llm,
        help=f"LLM model (default: {cfg.base_reasoning_llm})",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="medium",
        help="OpenAI reasoning effort (default: medium)",
    )
    parser.add_argument(
        "--verbosity",
        choices=["low", "medium", "high"],
        default="medium",
        help="OpenAI text verbosity (default: medium)",
    )
    parser.add_argument("--output", "-o", help="Save protocol to file")
    parser.add_argument(
        "--show-turns",
        action="store_true",
        help="Show all three turns, not just final protocol",
    )


def run(args: argparse.Namespace) -> int:
    """Run extract command."""
    paper_path = Path(args.paper)
    if not paper_path.exists():
        print(f"ERROR: Paper not found: {paper_path}", file=sys.stderr)
        return 1

    if args.supporting_info:
        si_path = Path(args.supporting_info)
        if not si_path.exists():
            print(f"ERROR: Supporting info not found: {si_path}", file=sys.stderr)
            return 1

    print(f"Paper: {args.paper}", file=sys.stderr)
    print(f"Task: {args.task}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    if args.supporting_info:
        print(f"Supporting info: {args.supporting_info}", file=sys.stderr)
    print("", file=sys.stderr)

    try:
        result = extract_methodology(
            paper_path=args.paper,
            task=args.task,
            supporting_info_path=args.supporting_info,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            verbosity=args.verbosity,
        )
        _print_result(result, args.output, args.show_turns)
        return 0
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
