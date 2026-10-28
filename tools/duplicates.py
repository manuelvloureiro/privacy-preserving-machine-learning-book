"""Find paragraphs that repeat another paragraph, within a book and across books (PLAN step 8).

    uv run python tools/duplicates.py report           # every pair past the threshold, most similar first
    uv run python tools/duplicates.py check            # fail on a pair that tools/duplicates.toml does not allow

Two paragraphs repeat each other when most of the shorter one's runs of `shingle` words also occur
in the longer one (containment, at least `threshold`). Each paragraph is its prose only, as
tools/prose.py reads it: no code, tables, captions or display math, and each inline formula one
word. A result belongs in one home and is cited elsewhere (AGENTS.md), so a repeat is either cut
or listed under `allow` with the reason it stays.
"""

import argparse
import re
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import prose

ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path(__file__).with_name("duplicates.toml")
BOOKS = ["book-full", "book-practitioner", "book-course", "book-opendp"]


@dataclass
class Paragraph:
    page: str  # path from the repository root
    section: str
    text: str
    shingles: set[str]


def paragraphs(page: str, raw: str, shingle: int, min_words: int) -> list[Paragraph]:
    found: list[Paragraph] = []
    for section, block in prose.prose_blocks(raw):
        text, _, _ = prose.plain(block)
        words = re.findall(r"[a-z]+", re.sub(r"⟦\d+⟧", " x ", text.lower()))
        if len(words) >= min_words:
            shingles = {" ".join(words[i : i + shingle]) for i in range(len(words) - shingle + 1)}
            found.append(Paragraph(page, section, text, shingles))
    return found


def every_page() -> list[Path]:
    pages = [ROOT / book / p for book in BOOKS for p in prose.pages_of(book)]
    return pages + sorted(ROOT.glob("notes/*/index.md"))


def repeats(config: dict[str, Any], paras: list[Paragraph]) -> list[tuple[float, Paragraph, Paragraph]]:
    holders: defaultdict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(paras):
        for s in p.shingles:
            holders[s].append(i)
    shared: defaultdict[tuple[int, int], int] = defaultdict(int)
    for owners in holders.values():
        if len(owners) <= config["max_holders"]:  # a stock phrase, not a repeat
            for pair in combinations(owners, 2):
                shared[pair] += 1
    found: list[tuple[float, Paragraph, Paragraph]] = []
    for (a, b), n in shared.items():
        score = n / min(len(paras[a].shingles), len(paras[b].shingles))
        if score >= config["threshold"]:
            found.append((score, paras[a], paras[b]))
    return sorted(found, key=lambda f: -f[0])


def allowed(config: dict[str, Any], a: Paragraph, b: Paragraph) -> bool:
    pages = {a.page, b.page}
    return any(set(entry["pages"]) == pages for entry in cast(list[dict[str, Any]], config.get("allow", [])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["report", "check"])
    args = parser.parse_args()
    config = tomllib.loads(CONFIG.read_text())
    paras = [
        p
        for page in every_page()
        for p in paragraphs(str(page.relative_to(ROOT)), page.read_text(), config["shingle"], config["min_words"])
    ]
    found = repeats(config, paras)
    failing = [f for f in found if not allowed(config, f[1], f[2])]
    shown = found if args.command == "report" else failing
    for score, a, b in shown:
        mark = "" if args.command == "check" or (score, a, b) in failing else "  (allowed)"
        print(f"{score:.2f}  {a.page} § {a.section}\n      {b.page} § {b.section}{mark}\n      {a.text[:110]}…\n")
    print(
        f"{len(failing)} repeated paragraphs not allowed, {len(found) - len(failing)} allowed (tools/duplicates.toml)"
    )
    if args.command == "check" and failing:
        sys.exit(1)


if __name__ == "__main__":
    main()
