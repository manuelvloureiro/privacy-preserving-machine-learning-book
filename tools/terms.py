"""Find and check acronyms, then build the book's glossary and index of terms.

    uv run python tools/terms.py acronyms        # report every acronym and where it's used
    uv run python tools/terms.py candidates      # suggest index terms not yet in terms.toml
    uv run python tools/terms.py check           # exit 1 on any acronym or index problem
    uv run python tools/terms.py build           # write glossary.md, term-index.md, _terms.json
    uv run python tools/terms.py build --book _build/pdf --profile pdf   # into a staged copy

The book's files are read from book/ unless --book names another copy of it; build writes its three
files into that copy. The make targets build from staged copies in _build/, so generated files never
land among the sources. --profile picks the Quarto profile (_quarto-<profile>.yml) whose appendices
count as part of the book; the default is the website profile, which includes the solutions.

The curated list is book/terms.toml:

    [acronyms]
    MGF = "moment generating function"                      # must be expanded on every page
    ML = { expansion = "machine learning", expand = false } # well known: no expansion needed

    [index]
    "Chernoff bound" = []                         # matched case-insensitively, plurals included
    "privacy loss" = ["privacy-loss"]             # other spellings to match

A "page" is one chapter file, which is one web page. A reader can land on any page, so each
page has to expand an acronym the first time it uses it, as "moment generating function (MGF)".
The PDF index uses real page numbers (book/filters/term-index.lua reads _terms.json); the web
index links to the section where each term appears.
"""

import argparse
import json
import re
import sys
import tomllib
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass
class Layout:
    """Where the book is read from and where the generated files go."""

    book: Path
    profile: str = "website"  # the Quarto profile whose appendices count as part of the book

    @property
    def terms(self) -> Path:
        return self.book / "terms.toml"

    @property
    def glossary(self) -> Path:
        return self.book / "glossary.md"

    @property
    def term_index(self) -> Path:
        return self.book / "term-index.md"

    @property
    def terms_json(self) -> Path:
        return self.book / "_terms.json"


layout = Layout(Path(__file__).resolve().parent.parent / "book")
GENERATED = {"glossary.md", "term-index.md"}

# Uppercase tokens that are not acronyms.
NOT_ACRONYMS = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "OK", "TODO"}
ACRONYM = re.compile(r"(?<![\w$\\])([A-Z][A-Z0-9]*[A-Z][A-Z0-9]*)(s?)(?![\w])")


# --- Reading the book ---------------------------------------------------------------------


@dataclass
class Section:
    heading: str
    anchor: str
    text: str  # prose only: math, code and markup removed


@dataclass
class Page:
    source: str  # path relative to book/, e.g. part01-prerequisites/02-concentration.md
    title: str
    anchor: str = ""  # id of the page's level-1 heading, which starts the page in the combined PDF
    sections: list[Section] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.sections)


def book_files() -> list[str]:
    """Chapters and appendices in reading order, from _quarto.yml and the profile's file."""
    config = cast(dict[str, Any], yaml.safe_load((layout.book / "_quarto.yml").read_text()))
    profile_file = layout.book / f"_quarto-{layout.profile}.yml"
    if profile_file.exists():  # a profile's book keys extend the base ones, as Quarto merges them
        extra = cast(dict[str, Any], yaml.safe_load(profile_file.read_text()) or {})
        config["book"] = {**cast(dict[str, Any], config["book"]), **cast(dict[str, Any], extra.get("book", {}))}
    files: list[str] = []

    def collect(entries: list[Any]) -> None:
        for entry in entries:
            if isinstance(entry, str):
                files.append(entry)
            elif isinstance(entry, dict):
                part = cast(dict[str, Any], entry)
                if str(part.get("part", "")).endswith((".md", ".qmd")):  # a part page has its own text
                    files.append(str(part["part"]))
                collect(cast(list[Any], part.get("chapters", [])))

    book = cast(dict[str, Any], config["book"])
    collect(cast(list[Any], book.get("chapters", [])))
    collect(cast(list[Any], book.get("appendices", [])))
    files = [f for f in files if Path(f).name not in GENERATED]  # written by build, read as output
    missing = [f for f in files if not (layout.book / f).exists()]
    if missing:
        print(
            f"note: skipping {len(missing)} files listed in _quarto.yml but not written yet: " + ", ".join(missing),
            file=sys.stderr,
        )
    return [f for f in files if f not in missing]


def strip_markup(text: str) -> str:
    """Leave only prose: drop code, math, HTML comments, attributes and link targets."""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"^(```|~~~).*?^\1", " ", text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)\$[^$\n]+?(?<!\\)\$", " ", text)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"\{[#.][^}]*\}", " ", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s*:\s", "", text, flags=re.MULTILINE)  # table captions read as prose
    return re.sub(r"[*_]{1,3}(?=\w)|(?<=\w)[*_]{1,3}", "", text)


def pandoc_identifier(heading: str) -> str:
    """Pandoc's auto_identifiers rule, which Quarto uses for headings without an explicit id."""
    text = re.sub(r"\$([^$]*)\$", r"\1", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*`]", "", text)
    text = "".join(c for c in text if c.isalnum() or c in "_-. " or unicodedata.category(c) == "Zs")
    text = re.sub(r"\s+", "-", text.strip()).lower()
    text = re.sub(r"^[^a-z]+", "", text)
    return text or "section"


def read_page(source: str) -> Page:
    raw = (layout.book / source).read_text()
    raw = re.sub(r"\A---\n.*?\n---\n", "", raw, flags=re.DOTALL)  # front matter
    raw = re.sub(r"^(```|~~~).*?^\1", "", raw, flags=re.DOTALL | re.MULTILINE)  # so '#' in code isn't a heading
    page = Page(source, title=Path(source).stem)
    seen: Counter[str] = Counter()
    heading, anchor, lines = "", "", list[str]()

    def flush() -> None:
        if lines or heading:
            page.sections.append(Section(heading, anchor, strip_markup("\n".join(lines))))

    for line in raw.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*?)\s*(\{[^}]*\})?\s*$", line)
        if not m:
            lines.append(line)
            continue
        flush()
        heading, attrs, lines = m.group(2), m.group(3) or "", []
        explicit = re.search(r"#([\w:-]+)", attrs)
        if explicit:
            anchor = explicit.group(1)
        else:
            base = pandoc_identifier(heading)
            anchor = base if not seen[base] else f"{base}-{seen[base]}"
            seen[base] += 1
        if len(m.group(1)) == 1 and not page.anchor:
            page.title = re.sub(r"^\d+(\.\d+)*\.?\s*", "", strip_markup(heading)).strip()
            page.anchor = anchor
    flush()
    return page


def read_book() -> list[Page]:
    return [read_page(f) for f in book_files()]


# --- The curated list ---------------------------------------------------------------------


@dataclass(frozen=True)
class Acronym:
    name: str
    expansion: str
    expand: bool  # must each page expand it at first use?


def load_names() -> set[str]:
    """Uppercase names that aren't acronyms to explain ([names] in terms.toml)."""
    if not layout.terms.exists():
        return set()
    groups = cast(dict[str, list[str]], tomllib.loads(layout.terms.read_text()).get("names", {}))
    return {name for group in groups.values() for name in group}


def load_terms() -> tuple[dict[str, Acronym], dict[str, list[str]]]:
    if not layout.terms.exists():
        return {}, {}
    data = tomllib.loads(layout.terms.read_text())
    acronyms: dict[str, Acronym] = {}
    for name, value in cast(dict[str, Any], data.get("acronyms", {})).items():
        if isinstance(value, str):
            acronyms[name] = Acronym(name, value, expand=True)
        else:
            entry = cast(dict[str, Any], value)
            acronyms[name] = Acronym(name, str(entry["expansion"]), expand=bool(entry.get("expand", True)))
    index = {
        term: [str(a) for a in cast(list[Any], aliases)]
        for term, aliases in cast(dict[str, Any], data.get("index", {})).items()
    }
    return acronyms, index


# --- Acronyms -----------------------------------------------------------------------------


@dataclass
class Use:
    page: str
    position: int  # offset in the page's prose
    expanded_as: str | None  # the words before "(ACR)" when this use is an expansion
    in_heading: bool = False  # headings count as uses, but needn't carry the expansion


def find_acronym_uses(pages: list[Page]) -> dict[str, list[Use]]:
    """Every acronym use in headings and prose, noting the ones written as an expansion.

    "moment generating function (MGF)" records the words before the parenthesis;
    "PPT (probabilistic polynomial time)" records the words inside it. Either counts when the
    initials spell the acronym or the words are the expansion in terms.toml, so that
    "pseudorandom generator (PRG)" counts although "pseudorandom" gives a single P.
    """
    ignored = NOT_ACRONYMS | load_names()
    known = {name: normalise(a.expansion) for name, a in load_terms()[0].items()}
    uses: dict[str, list[Use]] = defaultdict(list)
    for page in pages:
        offset = 0
        for section in page.sections:
            for text, in_heading in ((strip_markup(section.heading), True), (section.text, False)):
                for m in ACRONYM.finditer(text):
                    name = m.group(1)
                    if name in ignored:
                        continue
                    expected = known.get(name)
                    expanded_as = None
                    if text[m.start() - 1 : m.start()] == "(" and text[m.end() : m.end() + 1] == ")":
                        before = " ".join(re.findall(r"[\w'’–-]+", text[: m.start() - 1])[-8:])
                        if expected and normalise(before).endswith(expected):
                            expanded_as = before
                        else:  # "(ACR)" alone is a mention, not an expansion
                            expanded_as = initials_match(name, before)
                    elif after := re.match(r"\s*\(([^()]{3,80})\)", text[m.end() :]):
                        inside = after.group(1).strip()
                        if (expected and normalise(inside) == expected) or initials_match(name, inside) == inside:
                            expanded_as = inside
                    uses[name].append(Use(page.source, offset + m.start(), expanded_as, in_heading))
                offset += len(text) + 1
    return dict(uses)


def initials_match(name: str, words: str) -> str | None:
    """The shortest tail of `words` whose initials spell the acronym, if any."""
    tokens = [t.strip(",;:") for t in words.split() if t.strip(",;:")]
    letters = [c for c in name if c.isalpha()]
    for start in range(len(tokens) - 1, -1, -1):
        tail = tokens[start:]
        caps = "".join(part[0] for t in tail for part in re.split(r"[–—-]", t) if part).upper()
        significant = "".join(
            t[0] for t in tail if t.lower() not in {"of", "and", "the", "for", "a", "in", "to"}
        ).upper()
        if "".join(letters) in (caps, significant):
            return " ".join(tail)
    return None


def normalise(words: str) -> str:
    """Lowercase, one spelling for dashes, plurals dropped, so "Kullback-Leibler" matches "Kullback–Leibler"."""
    words = re.sub(r"[–—-]", "-", words.lower())
    return " ".join(re.sub(r"(?<=[a-z])s$", "", w) for w in words.split())


@dataclass
class Problem:
    level: str  # "error" or "warning"
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.where}: {self.message}"


def check_acronyms(pages: list[Page], acronyms: dict[str, Acronym]) -> list[Problem]:
    problems: list[Problem] = []
    uses = find_acronym_uses(pages)
    for name, found in sorted(uses.items()):
        known = acronyms.get(name)
        if known is None:
            guess = next((m for u in found if u.expanded_as and (m := initials_match(name, u.expanded_as))), None)
            hint = f' (the text expands it as "{guess}")' if guess else ""
            problems.append(Problem("error", name, f"not in terms.toml{hint}"))
            continue
        if len(found) == 1 and known.expand and found[0].expanded_as is None:
            problems.append(
                Problem(
                    "error",
                    f"{found[0].page}: {name}",
                    f'used once in the book and not expanded; write "{known.expansion}" instead',
                )
            )
            continue
        by_page: dict[str, list[Use]] = defaultdict(list)
        for use in found:
            by_page[use.page].append(use)
        for page, page_uses in by_page.items():
            expansions = [u for u in page_uses if u.expanded_as is not None]
            for u in expansions:
                if u.expanded_as and not normalise(u.expanded_as).endswith(normalise(known.expansion)):
                    problems.append(
                        Problem(
                            "error",
                            f"{page}: {name}",
                            f'expanded as "...{u.expanded_as[-40:]}", expected "{known.expansion}"',
                        )
                    )
            if not known.expand:
                continue
            if not expansions:
                problems.append(
                    Problem(
                        "error",
                        f"{page}: {name}",
                        f'used {len(page_uses)}× but never expanded; first use should read "{known.expansion} ({name})"',
                    )
                )
            elif expansions[0] is not next(u for u in page_uses if not u.in_heading or u.expanded_as):
                problems.append(Problem("error", f"{page}: {name}", "used before it is expanded; expand the first use"))
            if len(expansions) > 1:
                problems.append(
                    Problem("warning", f"{page}: {name}", f"expanded {len(expansions)}× on one page; once is enough")
                )
    for name in sorted(set(acronyms) - set(uses)):
        problems.append(Problem("warning", name, "in terms.toml but never used"))
    return problems


def report_acronyms(pages: list[Page], acronyms: dict[str, Acronym]) -> None:
    uses = find_acronym_uses(pages)
    width = max((len(n) for n in uses), default=4)
    for name, found in sorted(uses.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        known = acronyms.get(name)
        meaning = known.expansion if known else "?? not in terms.toml"
        counts = Counter(Path(u.page).stem for u in found)
        where = ", ".join(f"{p} {n}×" for p, n in counts.items())
        print(f"{name:<{width}}  {len(found):>3}×  {meaning}\n{'':<{width}}         {where}")
    print()
    for problem in check_acronyms(pages, acronyms):
        print(problem)


# --- Index terms --------------------------------------------------------------------------


def term_pattern(term: str, aliases: list[str]) -> re.Pattern[str]:
    variants = sorted({term, *aliases}, key=len, reverse=True)
    alternation = "|".join(re.escape(v).replace(r"\ ", r"[\s-]+") for v in variants)
    return re.compile(rf"(?<![\w-])(?:{alternation})(?:s|es)?(?![\w])", re.IGNORECASE)


@dataclass(frozen=True)
class Location:
    page: Page
    section: Section


def locate_terms(pages: list[Page], index: dict[str, list[str]]) -> dict[str, list[Location]]:
    """For each term, one location per page: the section whose heading names the term,
    otherwise the first section that mentions it."""
    skip = (
        set(cast(list[str], tomllib.loads(layout.terms.read_text()).get("index_skip_pages", [])))
        if layout.terms.exists()
        else set[str]()
    )
    found: dict[str, list[Location]] = {}
    for term, aliases in index.items():
        pattern = term_pattern(term, aliases)
        found[term] = []
        for page in pages:
            if page.source in skip:
                continue
            in_heading = [s for s in page.sections if pattern.search(strip_markup(s.heading))]
            in_text = [s for s in page.sections if pattern.search(s.text)]
            if in_heading or in_text:
                found[term].append(Location(page, (in_heading or in_text)[0]))
    return found


def check_index(pages: list[Page], index: dict[str, list[str]]) -> list[Problem]:
    return [
        Problem("warning", term, "index term not found in the book")
        for term, locations in locate_terms(pages, index).items()
        if not locations
    ]


def candidates(pages: list[Page], acronyms: dict[str, Acronym], index: dict[str, list[str]]) -> None:
    """Bold phrases are the book's defined terms; list those not yet indexed."""
    known = {t.lower() for t in index} | {a.lower() for al in index.values() for a in al}
    counts: Counter[str] = Counter()
    first: dict[str, str] = {}
    for page in pages:
        raw = (layout.book / page.source).read_text()
        for m in re.finditer(r"\*\*([^*\n]{3,60}?)\*\*", raw):
            phrase = re.sub(r"[.:—–-]+$", "", strip_markup(m.group(1))).strip()
            if not phrase or "$" in m.group(1) or phrase.lower() in known or phrase in acronyms:
                continue
            if re.match(r"^(Fact|Step|Example|Exercise|Proof|Note|Use|Why|Part)\b", phrase):
                continue
            counts[phrase] += 1
            first.setdefault(phrase, page.source)
    for phrase, n in counts.most_common():
        print(f"{n:>3}×  {phrase:<50}  {first[phrase]}")


# --- Generated pages ----------------------------------------------------------------------

HEADER = "<!-- Generated by tools/terms.py from terms.toml. Edit terms.toml, then run `make terms`. -->\n\n"


def section_link(loc: Location) -> str:
    target = loc.page.source + (f"#{loc.section.anchor}" if loc.section.anchor else "")
    return f"[{loc.page.title}]({target})"


def write_glossary(pages: list[Page], acronyms: dict[str, Acronym]) -> None:
    uses = find_acronym_uses(pages)
    titles = {p.source: p for p in pages}
    lines = [HEADER, "# Glossary {#sec-glossary}\n\n", "Acronyms used in the book and the pages that use them.\n\n"]
    for name in sorted(acronyms, key=str.lower):
        if name not in uses:
            continue
        pages_used = list(dict.fromkeys(u.page for u in uses[name]))
        links = ", ".join(f"[{titles[p].title}]({p})" for p in pages_used)
        lines.append(
            f"**{name}**\n:   {acronyms[name].expansion[0].upper()}{acronyms[name].expansion[1:]}. Used in {links}.\n\n"
        )
    layout.glossary.write_text("".join(lines).rstrip() + "\n")


def write_index(pages: list[Page], index: dict[str, list[str]]) -> None:
    located = locate_terms(pages, index)
    lines = [HEADER, "# Index {#sec-index}\n\n", '::: {.content-visible when-format="html"}\n\n']
    letter = ""
    for term in sorted(located, key=str.lower):
        if not located[term]:
            continue
        if term[0].upper() != letter:
            letter = term[0].upper()
            lines.append(f"\n## {letter} {{.unnumbered .unlisted}}\n\n")
        links = ", ".join(section_link(loc) for loc in located[term])
        lines.append(f"- **{term}**: {links}\n")
    lines.append('\n:::\n\n::: {.content-visible when-format="pdf"}\n\n```{=latex}\n\\printindex\n```\n\n:::\n')
    layout.term_index.write_text("".join(lines))
    # For filters/term-index.lua, which marks PDF index entries so they carry real page numbers.
    # The PDF is one document, so skipped pages are named by the id of their first heading.
    skip_pages = set(cast(list[str], tomllib.loads(layout.terms.read_text()).get("index_skip_pages", [])))
    skip = sorted({p.anchor for p in pages if p.source in skip_pages} | {"sec-glossary", "sec-index"})
    variants = {term: sorted({v.lower() for v in [term, *aliases]}) for term, aliases in index.items()}
    layout.terms_json.write_text(
        json.dumps({"skip": skip, "terms": variants}, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    )


# --- Command line -------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["acronyms", "candidates", "check", "build"])
    parser.add_argument("--book", type=Path, default=layout.book, help="the copy of the book to read and write")
    parser.add_argument("--profile", default=layout.profile, help="the Quarto profile whose appendices count")
    args = parser.parse_args()
    command: str = args.command
    layout.book, layout.profile = cast(Path, args.book).resolve(), cast(str, args.profile)

    pages = read_book()
    acronyms, index = load_terms()
    if command == "acronyms":
        report_acronyms(pages, acronyms)
    elif command == "candidates":
        candidates(pages, acronyms, index)
    elif command == "check":
        problems = check_acronyms(pages, acronyms) + check_index(pages, index)
        for problem in problems:
            print(problem)
        errors = sum(p.level == "error" for p in problems)
        print(f"{errors} errors, {len(problems) - errors} warnings")
        sys.exit(1 if errors else 0)
    else:
        write_glossary(pages, acronyms)
        write_index(pages, index)
        print(f"wrote {layout.glossary}, {layout.term_index.name}, {layout.terms_json.name}")


if __name__ == "__main__":
    main()
