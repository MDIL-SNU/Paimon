"""Fetch packmol userguide and example .inp files, save to knowledge directory.

Run once to populate knowledge/agents/packmol_agent/ before using the packmol agent.

Usage:
    python -m paimon.developers.fetch_packmol_knowledge
"""

import urllib.request
from pathlib import Path

from paimon.rag.web_crawl import crawl

KNOWLEDGE_ROOT = Path(__file__).resolve().parent.parent / "knowledge"

USERGUIDE_URL = "https://m3g.github.io/packmol/userguide.shtml"
USERGUIDE_KEY = "agents/packmol_agent/userguide"

EXAMPLES = {
    "agents/packmol_agent/examples/mixture": "https://m3g.github.io/packmol/examples/mixture-comment.inp",
    "agents/packmol_agent/examples/interface": "https://m3g.github.io/packmol/examples/interface-comment.inp",
    "agents/packmol_agent/examples/bilayer": "https://m3g.github.io/packmol/examples/bilayer-comment.inp",
    "agents/packmol_agent/examples/spherical_vesicle": "https://m3g.github.io/packmol/examples/spherical-comment.inp",
    "agents/packmol_agent/examples/solvated_protein": "https://m3g.github.io/packmol/examples/solvprotein-comment.inp",
}


def _save(key: str, text: str) -> Path:
    path = KNOWLEDGE_ROOT / (key.replace("/", "/") + ".txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def main() -> None:
    # userguide (HTML -> markdown via crawl)
    print(f"Fetching userguide from {USERGUIDE_URL} ...")
    text = crawl(USERGUIDE_URL)
    path = _save(USERGUIDE_KEY, text)
    print(f"  Saved {len(text)} chars -> {path}")

    # example .inp files (plain text, direct download)
    for key, url in EXAMPLES.items():
        print(f"Fetching {key} from {url} ...")
        text = urllib.request.urlopen(url).read().decode("utf-8")
        path = _save(key, text)
        print(f"  Saved {len(text)} chars -> {path}")

    print("Done.")


if __name__ == "__main__":
    main()
