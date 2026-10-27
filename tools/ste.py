"""Report where the books depart from ASD-STE100 Simplified Technical English (PROSE.md, rule 25).

    uv run python tools/ste.py report --book book-practitioner             # every page, most issues first
    uv run python tools/ste.py report --book book-practitioner --page part04-privacy-accounting/01-renyi-dp.md --worst 10
    uv run python tools/ste.py report --book book-practitioner --rules      # issue counts per check

The checks are stelint's (https://github.com/mikkovihonen/stelint), run with uvx at the version
pinned in tools/prose.toml. They see each page's prose only: tools/prose.py leaves out tables,
code, captions and display math, and each inline formula becomes the word "X". The words of
terms.toml and of `[ste] technical_words` are allowed as technical names, which is the first
exception of rule 25, and the checks in `[ste] ignore` are dropped. STE allows exceptions, so this
is a report that points at sentences to reconsider, never a gate.
"""

import argparse
import fcntl
import json
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import prose
import terms

OUT = prose.ROOT / "_build" / "ste"
ISSUE = re.compile(r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+) STE100\.(?P<rule>\w+): (?:\[[^\]]*\] )?(?P<message>.*)$")


@dataclass
class Block:
    page: str
    section: str
    text: str  # the paragraph as stelint sees it
    sentences: int
    issues: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])  # (check, message)


def config() -> dict[str, Any]:
    return cast(dict[str, Any], tomllib.loads(prose.CONFIG.read_text())["ste"])


def blocks_of(book: str, page: str) -> list[Block]:
    """The page's paragraphs in prose only, as tools/prose.py reads them."""
    out: list[Block] = []
    for section, source in prose.prose_blocks((prose.ROOT / book / page).read_text()):
        text = re.sub(r"⟦\d+⟧", "X", prose.plain(source)[0])
        if text:
            out.append(Block(page, section, text, len(prose.split_sentences(text))))
    return out


def glossary(cfg: dict[str, Any], out: Path) -> Path:
    """A stelint override that allows the books' technical names, and its glossaries.yaml."""
    acronyms, index = terms.load_terms()
    phrases = [a.expansion for a in acronyms.values()] + list(index) + [a for v in index.values() for a in v]
    words = {w.lower() for p in phrases for w in re.findall(r"[A-Za-z][A-Za-z-]+", p)}
    words |= set(cast(list[str], cfg.get("technical_words", [])))
    nouns = cast(list[str], cfg.get("technical_nouns", []))
    entries = [
        {
            "namespace": "words",
            "name": "NON_APPROVED_WORDS",
            "rules": ["PROSE.md rule 25: technical names"],
            "type": "mapping",
            "data": dict.fromkeys(sorted(words), "__REMOVE__"),
        },
        {
            "namespace": "words",
            "name": "TECHNICAL_VERBS_NOT_AS_NOUNS",
            "rules": ["PROSE.md rule 25: technical names"],
            "type": "mapping",
            "data": dict.fromkeys(nouns, "__REMOVE__"),
        },
    ]
    (out / "books.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (out / "glossaries.yaml").write_text("glossaries:\n  - path: books.jsonl\n    cardinality: 100\n")
    return out / "glossaries.yaml"


def attach(output: str, lines: dict[int, Block], cfg: dict[str, Any]) -> int:
    """Attach each issue of stelint's output to the paragraph on its line; return how many were kept."""
    ignore = set(cast(list[str], cfg.get("ignore", [])))
    limit = int(cfg.get("sentence_words", 25))
    seen: set[tuple[int, int, str]] = set()
    for line in output.splitlines():
        m = ISSUE.match(line)
        if not m or m["rule"] in ignore:
            continue
        if m["rule"] == "SentenceLength":
            # stelint reports a sentence once per limit it knows; rule 25 allows 25 words in a description.
            n = re.search(r"has (\d+) words", m["message"])
            if n and int(n.group(1)) <= limit:
                continue
            message = f"{n.group(1) if n else 'too many'} words; a descriptive sentence has at most {limit}"
        else:
            message = m["message"]
        key = (int(m["line"]), int(m["col"]), m["rule"])
        block = lines.get(key[0])
        if block is None or key in seen:
            continue
        seen.add(key)
        block.issues.append((m["rule"], message))
    return len(seen)


class stelint_slot:
    """Hold one of a few machine-wide slots while stelint runs.

    Each stelint run loads spaCy, about 500 MB. Many agents check pages at once, and on a shared
    machine an unbounded number of runs exhausts memory, so every run takes one of `slots` file
    locks in _build/ste/ and waits while all of them are held.
    """

    def __init__(self, slots: int) -> None:
        self.slots = max(1, slots)
        self.handle: Any = None

    def __enter__(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        while True:
            for i in range(self.slots):
                handle = (OUT / f"slot-{i}.lock").open("w")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                self.handle = handle
                return
            time.sleep(2)

    def __exit__(self, *exc: object) -> None:
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


def lint(book: str, pages: list[str], cfg: dict[str, Any]) -> list[Block]:
    """Run stelint once over every page's prose and attach each issue to its paragraph."""
    blocks = [b for p in pages for b in blocks_of(book, p)]
    lines: dict[int, Block] = {}
    text: list[str] = []
    for b in blocks:
        lines[len(text) + 1] = b
        text += [b.text, ""]
    # Each run works in its own folder, so that runs started at the same time do not share files.
    OUT.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=book.replace("/", "-") + "-", dir=OUT))
    source = out / "prose.md"
    source.write_text("\n".join(text))
    runner = cast(list[str], cfg["runner"])
    config_file = glossary(cfg, out)
    # stelint 0.1.5 applies `--glossaries` only after its checks have imported the base lists, so it
    # runs in the glossary's folder, where it also finds glossaries.yaml at import time.
    with stelint_slot(int(cfg.get("max_parallel", 2))):
        run = subprocess.run(
            [*runner, "--glossaries", config_file.name, source.name],
            capture_output=True,
            text=True,
            check=False,
            cwd=out,
        )
    found = attach(run.stdout, lines, cfg)
    if "Traceback" in run.stderr or (run.returncode not in (0, 1) and not found):
        sys.exit(f"stelint failed:\n{run.stderr[-2000:]}")
    return blocks


def show(blocks: list[Block], worst: int, by_rule: bool) -> None:
    pages: dict[str, list[Block]] = {}
    for b in blocks:
        pages.setdefault(b.page, []).append(b)

    def rate(bs: list[Block]) -> float:
        return 100 * sum(len(b.issues) for b in bs) / max(1, sum(b.sentences for b in bs))

    for page, bs in sorted(pages.items(), key=lambda kv: -rate(kv[1])):
        counts = Counter(rule for b in bs for rule, _ in b.issues)
        print(f"\n## {page}  {sum(counts.values())} issues, {rate(bs):.0f} per 100 sentences")
        if counts:
            print("   " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
        for b in sorted(bs, key=lambda b: -len(b.issues))[:worst]:
            if not b.issues:
                break
            print(f"\n   [{len(b.issues)}] {b.section}: {b.text[:110]}...")
            for rule, message in b.issues:
                print(f"       • {rule}: {message[:150]}")
    total = Counter(rule for b in blocks for rule, _ in b.issues)
    print(f"\n{sum(total.values())} issues in {sum(b.sentences for b in blocks)} sentences, {rate(blocks):.0f} per 100")
    if by_rule:
        for rule, n in total.most_common():
            print(f"  {n:6}  {rule}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["report"])
    parser.add_argument("--book", default="book-practitioner", help="a book folder, or notes/<name> for a study note")
    parser.add_argument("--page", help="one page, relative to the book's folder")
    parser.add_argument("--worst", type=int, default=3, help="paragraphs to show per page")
    parser.add_argument("--rules", action="store_true", help="also count the issues of each check")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    book = cast(str, args.book)
    pages = [cast(str, args.page)] if args.page else prose.pages_of(book)
    blocks = lint(book, pages, config())
    if args.json:
        print(json.dumps([asdict(b) for b in blocks], indent=1))
    else:
        show(blocks, cast(int, args.worst), cast(bool, args.rules))


if __name__ == "__main__":
    main()
