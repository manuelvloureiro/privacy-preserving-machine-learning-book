"""Check that every page has the shape AGENTS.md gives its kind, so the skeleton is a rule a tool keeps.

    uv run python tools/structure.py check                       # every book and note; exit 1 on a finding
    uv run python tools/structure.py check --book book-full      # one book
    uv run python tools/structure.py check --book book-full --page part03-dp-foundations/03-laplace-gaussian.md
    uv run python tools/structure.py check --strict              # also the pages listed as legacy
    uv run python tools/structure.py report --book book-full     # words per page, environments, status

tools/structure.toml maps each page of each book to a skeleton by glob and says what the skeleton
requires: the headings that close the page and their order, headings that may not appear, that every
`##` carries an id with the chapter's label prefix and every label uses it, numbered environments and
the proof after each result, captioned tables, the word budget, and the exact last line. Pages under a
book's `legacy` list are skipped by `check` until they are converted, so the gate can be switched on
before every page passes it. Code blocks and HTML comments are never read.
"""

import argparse
import fnmatch
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "tools" / "structure.toml"
RESULT_KINDS = ("def", "thm", "lem", "prp", "cor", "exm")
HEADING = re.compile(r"^(#{1,6}) (.*?)\s*$")
HEADING_ID = re.compile(r"\s*\{#([\w-]+)[^}]*\}\s*$")
LABEL = re.compile(r"\{#((?:sec|eq|fig|tbl|def|thm|lem|prp|cor|exm)-[\w-]+)")
ENV_OPEN = re.compile(r"^:{3,}\s*\{#((def|thm|lem|prp|cor|exm)-[\w-]+)")
PROOF_OPEN = re.compile(r"^:{3,}\s*\{\.proof")
CITED = re.compile(r"<!--\s*proof:\s*\S[^>]*-->")  # <!-- proof: cited --> or <!-- proof: in @sec-x -->
CAPTION = re.compile(r"^: .*\{#(tbl-[\w-]+)\}\s*$")
BOLD_RESULT = re.compile(r"^\*\*(Theorem|Lemma|Definition|Proposition|Corollary)\b")


@dataclass
class Skeleton:
    name: str
    single_title: bool = True  # notes use # for their sections
    opening: bool = False
    tail: list[str] = field(default_factory=list[str])
    forbidden_h2: list[str] = field(default_factory=list[str])
    forbidden_h3: list[str] = field(default_factory=list[str])
    section_ids: str = ""  # "sec": every ## carries {#sec-<prefix>-...}
    labels_use_prefix: bool = False
    environments: list[str] = field(default_factory=list[str])
    min_environments_per_topic: int = 0
    proof_after: list[str] = field(default_factory=list[str])
    forbid_bold_results: bool = False
    tables_captioned: bool = False
    summary_table: str = ""
    max_words: int = 0
    max_section_words: int = 0
    warn_below_words: int = 0
    last_line_any: list[str] = field(default_factory=list[str])
    required_h3_per_section: list[str] = field(default_factory=list[str])
    optional_h3: list[str] = field(default_factory=list[str])


@dataclass
class BookRules:
    pages: list[tuple[str, str]]  # (glob, skeleton name), first match wins
    legacy: set[str]


@dataclass
class Config:
    books: dict[str, BookRules]
    skeletons: dict[str, Skeleton]


def load_config(path: Path | None = None) -> Config:
    data = tomllib.loads((path or CONFIG).read_text())
    skeletons = {
        name: Skeleton(name=name, **cast(dict[str, Any], spec))
        for name, spec in cast(dict[str, Any], data.get("skeletons", {})).items()
    }
    books: dict[str, BookRules] = {}
    for book, spec in cast(dict[str, Any], data.get("books", {})).items():
        spec = cast(dict[str, Any], spec)
        pages = [(str(p["glob"]), str(p["skeleton"])) for p in cast(list[dict[str, Any]], spec.get("pages", []))]
        for _, name in pages:
            if name not in skeletons:
                raise ValueError(f"structure.toml: {book} uses an unknown skeleton {name!r}")
        books[book] = BookRules(pages, set(cast(list[str], spec.get("legacy", []))))
    return Config(books, skeletons)


# --- Reading a page -------------------------------------------------------------------------


@dataclass
class Heading:
    level: int
    text: str
    id: str | None
    line: int


@dataclass
class Env:
    kind: str  # def, thm, ..., or "proof"
    id: str | None
    line: int


@dataclass
class Table:
    line: int
    caption_id: str | None


@dataclass
class Page:
    path: Path
    lines: list[str]  # code and comments blanked, so line numbers hold
    headings: list[Heading]
    envs: list[Env]
    tables: list[Table]
    labels: list[tuple[str, int]]
    cited: set[int]  # lines carrying <!-- proof: cited -->
    bold_results: list[int]
    last_line: str

    def words(self, start: int = 0, end: int | None = None) -> int:
        return sum(len(line.split()) for line in self.lines[start:end])

    def sections(self) -> list[tuple[Heading, int, int]]:
        """Each ## heading with the line range it spans (to the next ## or the end)."""
        h2 = [h for h in self.headings if h.level == 2]
        out: list[tuple[Heading, int, int]] = []
        for i, h in enumerate(h2):
            end = h2[i + 1].line - 1 if i + 1 < len(h2) else len(self.lines)
            out.append((h, h.line, end))
        return out


def blank_blocks(text: str) -> tuple[str, set[int]]:
    """Blank fenced code, HTML comments, display math and front matter, keeping newlines; remember the cited markers."""
    cited = {text.count("\n", 0, m.start()) + 1 for m in CITED.finditer(text)}
    for pattern in (r"^(```|~~~).*?^\1[^\n]*$", r"<!--.*?-->", r"(?<!\\)\$\$.*?\$\$"):
        for m in list(re.finditer(pattern, text, flags=re.DOTALL | re.MULTILINE)):
            text = text[: m.start()] + re.sub(r"[^\n]", " ", text[m.start() : m.end()]) + text[m.end() :]
    if text.startswith("---\n"):
        m = re.match(r"^---\n.*?\n---[^\n]*", text, flags=re.DOTALL)
        if m:
            text = re.sub(r"[^\n]", " ", m.group()) + text[m.end() :]
    return text, cited


def parse(path: Path) -> Page:
    text, cited = blank_blocks(path.read_text())
    lines = text.split("\n")
    headings: list[Heading] = []
    envs: list[Env] = []
    tables: list[Table] = []
    labels: list[tuple[str, int]] = []
    bold: list[int] = []
    in_table = False
    for n, line in enumerate(lines, start=1):
        if m := HEADING.match(line):
            raw = m.group(2)
            ident = HEADING_ID.search(raw)
            headings.append(
                Heading(len(m.group(1)), HEADING_ID.sub("", raw).strip(), ident.group(1) if ident else None, n)
            )
        if m := ENV_OPEN.match(line):
            envs.append(Env(m.group(2), m.group(1), n))
        elif PROOF_OPEN.match(line):
            envs.append(Env("proof", None, n))
        if line.startswith("|"):
            if not in_table:
                tables.append(Table(n, None))
                in_table = True
        elif in_table and line.strip():
            in_table = False
            if (m := CAPTION.match(line)) and tables:
                tables[-1].caption_id = m.group(1)
        # a blank line may separate the table from its caption; look one line ahead
        elif in_table and not line.strip() and n < len(lines) and (m := CAPTION.match(lines[n])) and tables:
            tables[-1].caption_id = m.group(1)
            in_table = False
        for m in LABEL.finditer(line):
            labels.append((m.group(1), n))
        if BOLD_RESULT.match(line.strip()):
            bold.append(n)
    last = next((line for line in reversed(lines) if line.strip()), "")
    return Page(path, lines, headings, envs, tables, labels, cited, bold, last.strip())


def normalize(heading: str) -> str:
    """The comparable form of a heading: text before a colon or dash, lower-cased."""
    text = re.split(r"[:—–]| - ", heading, maxsplit=1)[0]
    return re.sub(r"\s+", " ", text).strip().lower()


def matches(heading: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(normalize(heading), p.lower()) for p in patterns)


def prefix_of(page: Page, skel: Skeleton) -> str | None:
    """The chapter's label prefix: the segment after sec- in the first ## id."""
    for h in page.headings:
        if h.level == 2 and h.id and h.id.startswith("sec-"):
            parts = h.id.split("-")
            if len(parts) >= 2:
                return parts[1]
    return None


# --- Checks ---------------------------------------------------------------------------------


@dataclass
class Finding:
    path: Path
    line: int
    message: str
    warning: bool = False

    def __str__(self) -> str:
        kind = "warning" if self.warning else "error"
        return f"{self.path.relative_to(ROOT)}:{self.line}: {kind}: {self.message}"


def check_headings(page: Page, skel: Skeleton, prefix: str | None) -> list[Finding]:
    out: list[Finding] = []
    titles = [h for h in page.headings if h.level == 1]
    if skel.single_title and len(titles) != 1:
        out.append(Finding(page.path, titles[1].line if len(titles) > 1 else 1, "a page has exactly one # title"))
    h2 = [h for h in page.headings if h.level == 2]
    if skel.opening and titles and h2:
        body = page.words(titles[0].line, h2[0].line - 1)
        if body == 0:
            out.append(Finding(page.path, h2[0].line, "the page opens with prose before its first ## section"))
    seen: set[str] = set()
    for h in page.headings:
        if h.level == 2:
            if matches(h.text, skel.forbidden_h2):
                out.append(Finding(page.path, h.line, f'forbidden heading "## {h.text}" ({skel.name})'))
            full = re.sub(r"\s+", " ", h.text).strip().lower()
            if skel.single_title and full in seen:
                out.append(Finding(page.path, h.line, f'duplicate heading "## {h.text}"'))
            seen.add(full)
            in_tail = normalize(h.text) in {t.lower() for t in skel.tail}
            if skel.section_ids and not in_tail and (not h.id or not h.id.startswith(f"{skel.section_ids}-")):
                out.append(
                    Finding(
                        page.path,
                        h.line,
                        f'"## {h.text}" needs an id {{#{skel.section_ids}-{prefix or "<prefix>"}-...}}',
                    )
                )
            elif skel.section_ids and prefix and h.id and prefix_in(h.id) != prefix:
                out.append(Finding(page.path, h.line, f"id {h.id} does not use the chapter's prefix {prefix!r}"))
        elif h.level == 3 and matches(h.text, skel.forbidden_h3):
            out.append(Finding(page.path, h.line, f'forbidden heading "### {h.text}" ({skel.name})'))
    if skel.tail:
        last = [normalize(h.text) for h in h2[-len(skel.tail) :]]
        if last != [t.lower() for t in skel.tail]:
            line = h2[-1].line if h2 else 1
            out.append(
                Finding(page.path, line, "the page ends with the sections " + " > ".join(f"## {t}" for t in skel.tail))
            )
    if skel.required_h3_per_section:
        tail = {t.lower() for t in skel.tail}
        for h, start, end in page.sections():
            if normalize(h.text) in tail:
                continue
            found = [normalize(x.text) for x in page.headings if x.level == 3 and start < x.line <= end]
            if not found:  # a plain prose section, such as a closing table of recipes
                continue
            for want in skel.required_h3_per_section:
                if want.lower() not in found:
                    out.append(Finding(page.path, h.line, f'"## {h.text}" lacks "### {want}"'))
            allowed = {w.lower() for w in skel.required_h3_per_section} | {w.lower() for w in skel.optional_h3}
            for x in page.headings:
                if x.level == 3 and start < x.line <= end and normalize(x.text) not in allowed:
                    out.append(
                        Finding(page.path, x.line, f'"### {x.text}" is not one of this appendix\'s sub-headings')
                    )
    return out


def prefix_in(label: str) -> str:
    parts = label.split("-")
    return parts[1] if len(parts) >= 2 else ""


def check_labels(page: Page, skel: Skeleton, prefix: str | None) -> list[Finding]:
    out: list[Finding] = []
    if skel.labels_use_prefix and prefix:
        for label, line in page.labels:
            if prefix_in(label) != prefix:
                out.append(Finding(page.path, line, f"label {label} does not use the chapter's prefix {prefix!r}"))
    if skel.forbid_bold_results:
        for line in page.bold_results:
            out.append(
                Finding(page.path, line, "a result is a numbered environment (::: {#thm-...}), not a bold lead-in")
            )
    return out


def check_environments(page: Page, skel: Skeleton) -> list[Finding]:
    out: list[Finding] = []
    if skel.min_environments_per_topic:
        tail = {t.lower() for t in skel.tail}
        for h, start, end in page.sections():
            if normalize(h.text) in tail:
                continue
            n = sum(1 for e in page.envs if e.kind in skel.environments and start < e.line <= end)
            if n < skel.min_environments_per_topic:
                out.append(
                    Finding(
                        page.path,
                        h.line,
                        f'"## {h.text}" has no numbered environment (definition, theorem, example, ...)',
                    )
                )
    if skel.proof_after:
        h2_lines = sorted(h.line for h in page.headings if h.level == 2)
        for i, e in enumerate(page.envs):
            if e.kind not in skel.proof_after:
                continue
            next_env = page.envs[i + 1].line if i + 1 < len(page.envs) else len(page.lines) + 1
            next_h2 = next((n for n in h2_lines if n > e.line), len(page.lines) + 1)
            limit = min(next_env, next_h2)
            proved = i + 1 < len(page.envs) and page.envs[i + 1].kind == "proof" and page.envs[i + 1].line < next_h2
            cited = any(e.line < c < limit for c in page.cited)
            if not proved and not cited:
                out.append(
                    Finding(
                        page.path,
                        e.line,
                        f"{e.id} has no proof after it (a .proof div, or <!-- proof: cited --> where the text says where the proof is)",
                    )
                )
    return out


def check_tables(page: Page, skel: Skeleton, prefix: str | None) -> list[Finding]:
    out: list[Finding] = []
    if skel.tables_captioned:
        for t in page.tables:
            if not t.caption_id:
                out.append(Finding(page.path, t.line, "a table needs a caption line `: Caption. {#tbl-...}`"))
    if skel.summary_table and prefix:
        want = skel.summary_table.format(prefix=prefix)
        for h, start, end in page.sections():
            if normalize(h.text) == "summary":
                ids = [t.caption_id for t in page.tables if start < t.line <= end]
                if ids != [want]:
                    out.append(Finding(page.path, h.line, f"## Summary holds exactly one table, captioned {{#{want}}}"))
    return out


def check_budget(page: Page, skel: Skeleton) -> list[Finding]:
    out: list[Finding] = []
    total = page.words()
    if skel.max_words and total > skel.max_words:
        out.append(Finding(page.path, 1, f"{total} words, over the budget of {skel.max_words}"))
    if skel.warn_below_words and total < skel.warn_below_words:
        out.append(Finding(page.path, 1, f"{total} words, under the usual {skel.warn_below_words}", warning=True))
    if skel.max_section_words:
        for h, start, end in page.sections():
            n = page.words(start, end)
            if n > skel.max_section_words:
                out.append(Finding(page.path, h.line, f'"## {h.text}" has {n} words, over {skel.max_section_words}'))
    return out


def check_tail(page: Page, skel: Skeleton, rel: str) -> list[Finding]:
    out: list[Finding] = []
    if skel.last_line_any and not any(re.match(p, page.last_line) for p in skel.last_line_any):
        out.append(
            Finding(
                page.path, len(page.lines), "the last line is one of the pointers AGENTS.md gives this kind of page"
            )
        )
    return out


def check_page(path: Path, rel: str, skel: Skeleton) -> list[Finding]:
    page = parse(path)
    prefix = prefix_of(page, skel)
    findings = (
        check_headings(page, skel, prefix)
        + check_labels(page, skel, prefix)
        + check_environments(page, skel)
        + check_tables(page, skel, prefix)
        + check_budget(page, skel)
        + check_tail(page, skel, rel)
    )
    return sorted(findings, key=lambda f: f.line)


# --- Pages and command line -------------------------------------------------------------------


def pages_of(book: str, config: Config) -> list[tuple[str, Skeleton | None]]:
    """Every page of a book or note with its skeleton, by the first glob that matches it."""
    folder = ROOT / book
    rules = config.books.get(book)
    if rules is None:
        return []
    out: list[tuple[str, Skeleton | None]] = []
    for p in sorted(folder.rglob("*.md")):
        rel = p.relative_to(folder).as_posix()
        if p.name in ("README.md", "glossary.md", "term-index.md") or "_freeze" in p.parts:
            continue
        name = next((s for g, s in rules.pages if fnmatch.fnmatchcase(rel, g)), None)
        out.append((rel, config.skeletons[name] if name else None))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "report"])
    parser.add_argument("--book", help="one book or note folder, as in structure.toml; the default is all")
    parser.add_argument("--page", help="one page, relative to the book's folder")
    parser.add_argument("--strict", action="store_true", help="also check the pages listed as legacy")
    args = parser.parse_args()
    config = load_config()
    books = [cast(str, args.book)] if args.book else list(config.books)
    rows: list[tuple[str, str, Skeleton | None, list[Finding], bool]] = []
    for book in books:
        legacy = config.books[book].legacy
        for rel, skel in pages_of(book, config):
            if args.page and rel != args.page:
                continue
            if skel is None:
                rows.append(
                    (
                        book,
                        rel,
                        None,
                        [Finding(ROOT / book / rel, 1, "no skeleton in structure.toml matches this page")],
                        False,
                    )
                )
                continue
            is_legacy = rel in legacy and not args.strict and not args.page
            findings = [] if is_legacy else check_page(ROOT / book / rel, rel, skel)
            rows.append((book, rel, skel, findings, is_legacy))
    if args.command == "check":
        errors = 0
        for _, _, _, findings, _ in rows:
            for f in findings:
                print(f)
                errors += not f.warning
        skipped = sum(1 for r in rows if r[4])
        print(f"{errors} structure errors in {len(rows)} pages ({skipped} legacy pages skipped)", file=sys.stderr)
        sys.exit(1 if errors else 0)
    print(f"{'page':<70}{'skeleton':<20}{'words':>7}{'envs':>6}  status")
    for book, rel, skel, findings, is_legacy in sorted(rows, key=lambda r: -parse(ROOT / r[0] / r[1]).words()):
        page = parse(ROOT / book / rel)
        envs = sum(1 for e in page.envs if e.kind != "proof")
        status = (
            "legacy"
            if is_legacy
            else (
                "ok" if not any(not f.warning for f in findings) else f"{sum(not f.warning for f in findings)} errors"
            )
        )
        print(f"{(book + '/' + rel)[:69]:<70}{(skel.name if skel else '-'):<20}{page.words():>7}{envs:>6}  {status}")


if __name__ == "__main__":
    main()
