"""Check that every page writes the repository's notation as NOTATION.md fixes it.

    uv run python tools/notation.py check                     # exit 1 on any finding, in every book
    uv run python tools/notation.py check --book book-course  # one book, or notes/
    uv run python tools/notation.py check book-course/part02-dp-basics/*.md   # some pages
    uv run python tools/notation.py report                    # how often each rule fires, per book

The rules are notation.toml at the repository's root: forbidden spellings, each looked for in the
math, in the prose, in the prose outside headings, link text and captions, or in the whole page
(for a phrase that names a symbol, such as "sampling rate $\\gamma$"), and pairs of meanings
of one symbol that must not meet on a page. A page is any .md or .qmd file of a book's folder or of
a study note, so a book's README and the exercise pages are checked too. Code blocks, inline code
and HTML comments are never checked: code follows Python's names, not the book's.
"""

import argparse
import re
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ["book-full", "book-exercises", "book-practitioner", "book-course", "notes"]
GENERATED = {"glossary.md", "term-index.md"}
ASYMPTOTIC = re.compile(r"(?:\\tilde\s*)?(?:O|\\Theta|\\Omega|o|\\omega)\s*\($")


@dataclass
class Rule:
    find: re.Pattern[str]
    use: str
    scope: str  # math, text, body or page
    exceptions: list[re.Pattern[str]] = field(default_factory=list[re.Pattern[str]])
    max_per_page: int = 0
    outside_asymptotic: bool = False


@dataclass
class Conflict:
    symbol: str
    first: re.Pattern[str]
    second: re.Pattern[str]
    fix: str


@dataclass
class Finding:
    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path.relative_to(ROOT)}:{self.line}: {self.message}"


def load_rules(path: Path = ROOT / "notation.toml") -> tuple[list[Rule], list[Conflict]]:
    data = tomllib.loads(path.read_text())
    rules = [
        Rule(
            find=re.compile(r["find"]),
            use=r["use"],
            scope=r["in"],
            exceptions=[re.compile(e) for e in r.get("except", [])],
            max_per_page=r.get("max_per_page", 0),
            outside_asymptotic=r.get("outside_asymptotic", False),
        )
        for r in cast(list[dict[str, Any]], data.get("rule", []))
    ]
    for rule in rules:
        if rule.scope not in ("math", "text", "body", "page"):
            raise ValueError(f"notation.toml: unknown scope {rule.scope!r} for {rule.find.pattern!r}")
    conflicts = [
        Conflict(c["symbol"], re.compile(c["first"]), re.compile(c["second"]), c["fix"])
        for c in cast(list[dict[str, Any]], data.get("conflict", []))
    ]
    return rules, conflicts


# --- Splitting a page into math and prose ---------------------------------------------------


def blank(text: str, start: int, end: int) -> str:
    """Replace text[start:end] by spaces, keeping its newlines so offsets and lines stay put."""
    return text[:start] + re.sub(r"[^\n]", " ", text[start:end]) + text[end:]


@dataclass
class Split:
    math: list[tuple[int, str]]  # (offset in the page, the math's source without its dollars)
    prose: str  # the page with code, math, link targets and attributes blanked out
    code_free: str  # the page with only code, comments and front matter blanked out


def split_page(text: str) -> Split:
    """Find the page's math and its prose. Offsets refer to the original text."""
    for pattern in (
        r"<!--.*?-->",
        r"^(```|~~~).*?^\1[^\n]*$",  # fenced code
        r"^---\n.*?\n---$",  # YAML front matter
        r"`[^`\n]+`",  # inline code, which includes {python} expressions
    ):
        for m in list(re.finditer(pattern, text, flags=re.DOTALL | re.MULTILINE)):
            if pattern.startswith("^---") and m.start() != 0:
                continue
            text = blank(text, m.start(), m.end())
    code_free = text
    math: list[tuple[int, str]] = []
    for pattern in (r"(?<!\\)\$\$(.+?)\$\$", r"(?<![\\$])\$(?!\s)((?:[^$\n]|\n(?!\s*\n))+?)(?<![\s\\])\$(?!\d)"):
        for m in list(re.finditer(pattern, text, flags=re.DOTALL)):
            math.append((m.start(1), m.group(1)))
            text = blank(text, m.start(), m.end())
    for pattern in (r"\]\([^)\s]*\)", r"\{[#.][^}]*\}", r"<[^>\n]+>", r"https?://\S+", r"(?<![\w])@[\w-]+"):
        for m in list(re.finditer(pattern, text)):
            text = blank(text, m.start(), m.end())
    math.sort()
    return Split(math, text, code_free)


def plain_string_spans(text: str) -> list[tuple[int, int]]:
    """Where Markdown makes plain text: headings, link text, captions and environment names."""
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"^(?:#+ |: |!\[).*$", text, flags=re.MULTILINE):
        spans.append((m.start(), m.end()))
    for m in re.finditer(r"\[[^\]\n]*\]\(", text):
        spans.append((m.start(), m.end()))
    for m in re.finditer(r'name="[^"\n]*"|title="[^"\n]*"|fig-cap: .*$|tbl-cap: .*$', text, flags=re.MULTILINE):
        spans.append((m.start(), m.end()))
    return spans


def inside_asymptotic(math: str, at: int) -> bool:
    """Whether position `at` lies inside the parentheses of O(...), Theta(...) or Omega(...)."""
    depth_opened_by: list[bool] = []
    for i, ch in enumerate(math[:at]):
        if ch in "([{":
            depth_opened_by.append(ch == "(" and bool(ASYMPTOTIC.search(math[: i + 1])))
        elif ch in ")]}" and depth_opened_by:
            depth_opened_by.pop()
    return any(depth_opened_by)


# --- Checking -------------------------------------------------------------------------------


def masked(text: str, rule: Rule) -> str:
    for exception in rule.exceptions:
        for m in list(exception.finditer(text)):
            text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
    return text


def check_page(path: Path, rules: list[Rule], conflicts: list[Conflict]) -> list[Finding]:
    original = path.read_text()
    split = split_page(original)
    spans = plain_string_spans(original)

    def line_of(offset: int) -> int:
        return original.count("\n", 0, offset) + 1

    findings: list[Finding] = []
    for rule in rules:
        hits: list[tuple[int, str]] = []
        if rule.scope == "math":
            for offset, math in split.math:
                for m in rule.find.finditer(masked(math, rule)):
                    if rule.outside_asymptotic and inside_asymptotic(math, m.start()):
                        continue
                    hits.append((offset + m.start(), m.group()))
        elif rule.scope == "page":
            for m in rule.find.finditer(masked(split.code_free, rule)):
                hits.append((m.start(), m.group()))
        else:
            for m in rule.find.finditer(masked(split.prose, rule)):
                if rule.scope == "body" and any(a <= m.start() < b for a, b in spans):
                    continue
                hits.append((m.start(), m.group()))
        if len(hits) > rule.max_per_page:
            for offset, found in hits[rule.max_per_page :]:
                findings.append(Finding(path, line_of(offset), f"{found.strip()!r}: write {rule.use}"))
    for c in conflicts:
        first, second = c.first.search(split.code_free), c.second.search(split.code_free)
        if first and second:
            findings.append(
                Finding(
                    path,
                    1,
                    f"{c.symbol} has two meanings on this page ({first.group()!r} and {second.group()!r}): {c.fix}",
                )
            )
    return sorted(findings, key=lambda f: f.line)


def pages(book: str) -> list[Path]:
    folder = ROOT / book
    return sorted(
        p
        for p in folder.rglob("*")
        if p.suffix in (".md", ".qmd") and "_freeze" not in p.parts and p.name not in GENERATED
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "report"])
    parser.add_argument("--book", choices=BOOKS, help="check one book (or notes); the default is all of them")
    parser.add_argument("paths", nargs="*", type=Path, help="check only these pages")
    args = parser.parse_args()
    rules, conflicts = load_rules()
    books = [args.book] if args.book else BOOKS
    if args.command == "check":
        chosen = [p.resolve() for p in args.paths] or [p for b in books for p in pages(b)]
        findings = [f for p in chosen for f in check_page(p, rules, conflicts)]
        for f in findings:
            print(f)
        print(f"{len(findings)} notation findings in {', '.join(books)} (see NOTATION.md)", file=sys.stderr)
        sys.exit(1 if findings else 0)
    counts: dict[str, Counter[str]] = {}
    for b in books:
        for p in pages(b):
            for f in check_page(p, rules, conflicts):
                key = f.message.split(": write ")[-1] if ": write " in f.message else f.message.split(":")[-1]
                counts.setdefault(key.strip(), Counter())[b] += 1
    print(f"{'rule':<70}" + "".join(f"{b.removeprefix('book-'):>13}" for b in books))
    for key, by_book in sorted(counts.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"{key[:69]:<70}" + "".join(f"{by_book[b]:>13}" for b in books))


if __name__ == "__main__":
    main()
