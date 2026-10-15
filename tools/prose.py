"""Measure how hard the books' prose is to read, and point at the sentences that make it so.

    uv run python tools/prose.py report --book book-practitioner             # every page, worst first
    uv run python tools/prose.py report --book book-practitioner --page part01-prerequisites/02-concentration.md
    uv run python tools/prose.py report --book book-full --worst 20         # the 20 worst paragraphs
    uv run python tools/prose.py check --book book-practitioner             # exit 1 if a page fails
    uv run python tools/prose.py text "A sentence to test, with $x$ in it." # analyse a snippet

The checks are the measurable half of PROSE.md: long and overloaded sentences, chained clauses,
formulas and numbers piled into one sentence, symbols used before they are explained in words,
too many new concepts in one paragraph, long openings before the main clause, nominalizations,
and the reading grade of each section. Thresholds are per book, in tools/prose.toml. The checks
cannot tell whether an explanation is right or whether a paragraph follows from the one before;
PROSE.md's editorial review does that. They find where to look.

Formulas count as one word each, so a sentence is judged on its prose. Tables, figures, code,
display captions and reference lists are skipped.
"""

import argparse
import json
import re
import sys
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import terms

ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path(__file__).resolve().parent / "prose.toml"
SKIP_SECTIONS = ("further reading", "references")

# --- Reading a page into paragraphs and sentences ----------------------------------------


@dataclass
class Sentence:
    text: str  # prose, with each formula replaced by a placeholder
    math: list[str]  # the formulas, in order
    ids: list[int]  # their placeholder numbers in the text, which count across the paragraph
    words: int
    flags: list[str] = field(default_factory=list[str])


@dataclass
class Paragraph:
    section: str
    source: str  # the paragraph as written, for the report
    sentences: list[Sentence]
    bold: list[str]
    words: int
    flags: list[str] = field(default_factory=list[str])


MATH = re.compile(r"(?<![\\$])\$(?!\$)([^$\n]+?)(?<!\\)\$")
DISPLAY = re.compile(r"\$\$.*?\$\$(\s*\{#[^}]*\})?", re.DOTALL)
TOKEN = re.compile(r"⟦\d+⟧|[A-Za-z][A-Za-z'’-]*|\d[\d,.]*\d|\d")
ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "vs.", "cf.", "Fig.", "Eq.", "approx.", "resp.")


def prose_blocks(raw: str) -> list[tuple[str, str]]:
    """(section heading, paragraph) pairs, without code, tables, figures, captions or display labels."""
    raw = re.sub(r"\A---\n.*?\n---\n", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)
    raw = re.sub(r"^(```|~~~).*?^\1\s*$", "", raw, flags=re.DOTALL | re.MULTILINE)
    raw = DISPLAY.sub(" ⟦display⟧ ", raw)
    blocks: list[tuple[str, str]] = []
    section, lines = "", list[str]()

    def flush() -> None:
        text = " ".join(lines).strip()
        if text and not section.lower().startswith(SKIP_SECTIONS):
            blocks.append((section, text))
        lines.clear()

    for line in raw.splitlines():
        stripped = line.strip()
        heading = re.match(r"^(#{1,6})\s+(.*?)\s*(\{[^}]*\})?\s*$", stripped)
        if heading:
            flush()
            section = heading.group(2)
            continue
        if not stripped or stripped.startswith(("|", "![", ": ", ":::")) or re.match(r"^\{#[^}]*\}$", stripped):
            flush()
            continue
        if re.match(r"^([-*]|\d+\.)\s+", stripped):  # each list item is its own paragraph
            flush()
            stripped = re.sub(r"^([-*]|\d+\.)\s+", "", stripped)
        lines.append(stripped)
    flush()
    return blocks


def plain(text: str) -> tuple[str, list[str], list[str]]:
    """The paragraph's prose with formulas as placeholders, its formulas, and its bold terms."""
    bold = [re.sub(r"[*_`]", "", b) for b in re.findall(r"\*\*(.+?)\*\*", text)]
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\{[#.][^}]*\}", "", text)
    text = re.sub(
        r"@(fig|tbl|eq|sec)-[\w-]+", lambda m: {"fig": "Figure", "tbl": "Table"}.get(m.group(1), "Section"), text
    )
    text = re.sub(r"`[^`]*`", "code", text)
    math: list[str] = []

    def keep(m: re.Match[str]) -> str:
        math.append(m.group(1))
        return f"⟦{len(math) - 1}⟧"

    text = MATH.sub(keep, text)
    text = text.replace("⟦display⟧", f"⟦{len(math)}⟧")
    if "⟦" in text and len(math) < len(re.findall(r"⟦\d+⟧", text)):
        math.append("display")
    text = re.sub(r"[*_]{1,3}(?=\S)|(?<=\S)[*_]{1,3}", "", text)
    return re.sub(r"\s+", " ", text).strip(), math, bold


def split_sentences(text: str) -> list[str]:
    for i, abbr in enumerate(ABBREVIATIONS):
        text = text.replace(abbr, f"§{i}§")
    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9⟦\"(\[])", text)
    out: list[str] = []
    for part in parts:
        for i, abbr in enumerate(ABBREVIATIONS):
            part = part.replace(f"§{i}§", abbr)
        if part.strip():
            out.append(part.strip())
    return out


# --- What the checks look for -------------------------------------------------------------

GREEK = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "varepsilon",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "vartheta",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "sigma",
    "tau",
    "phi",
    "varphi",
    "chi",
    "psi",
    "omega",
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Pi",
    "Sigma",
    "Phi",
    "Psi",
    "Omega",
]
NOT_SYMBOLS = {"e", "d"}  # Euler's number and the differential
DEFINITION_CUE = re.compile(
    r"\b(where|write|written|writes|call|called|calls|denote|denotes|denoted|let|named|stands for|"
    r"is the|are the|be the|is a|is an|means|measures|counts)\b",
    re.IGNORECASE,
)
# A formula is named in words when a determiner comes shortly before it: "some value $a$",
# "the noise multiplier $\\sigma$", "each record $x$".
INTRODUCED_BY_NOUN = re.compile(
    r"\b(the|a|an|some|any|its|their|each|every|this|that|these|those)(\s+[A-Za-z'-]+){0,3}\s*$", re.IGNORECASE
)
OPENERS = re.compile(
    r"^(If|When|Because|Although|Though|Since|While|Unless|Once|Given|As|After|Before|For|In|With|Under|By|To)\b"
)
NOMINALIZATION = re.compile(r"\b[a-z]{3,}(?:tion|sion|ment|ness|ity|ance|ence|ancy|ency)s?\b")
TECHNICAL_NOUNS = {
    "probability",
    "distribution",
    "function",
    "information",
    "section",
    "question",
    "position",
    "inequality",
    "divergence",
    "variance",
    "sensitivity",
    "composition",
    "population",
    "mention",
    "sentence",
    "evidence",
    "difference",
    "privacy",
    "entity",
    "identity",
    "utility",
    "quantity",
    "majority",
    "minority",
    "community",
    "equation",
    "condition",
    "definition",
    "relation",
    "notation",
    "intuition",
    "deviation",
    "operation",
    "iteration",
    "experiment",
    "element",
    "statement",
    "argument",
    "moment",
    "comment",
    "document",
    "segment",
    "fragment",
    "instance",
    "distance",
    "sequence",
    "reference",
    "audience",
    "confidence",
    "property",
    "convexity",
    "smoothness",
    "regression",
    "classification",
    "computation",
    "optimization",
    "approximation",
    "estimation",
    "authority",
    "university",
    "capacity",
    "velocity",
    "density",
    "entropy",
    "duplication",
    "reconstruction",
    "membership",
    "inference",
    "attention",
    "direction",
    "fraction",
    "collection",
    "election",
    "solution",
}
WORD_SYLLABLES = re.compile(r"[aeiouy]+")


def symbols_in(formula: str) -> list[str]:
    f = re.sub(r"\\(text|mathrm|operatorname|mathbf|textbf)\s*\{[^}]*\}", " ", formula)
    f = re.sub(r"\\mathbb\s*\{?[A-Z]\}?", " ", f)
    f = re.sub(
        r"\\(Pr|ln|log|exp|min|max|sum|prod|sqrt|frac|left|right|cdot|le|ge|leq|geq|in|to|infty|approx)\b", " ", f
    )
    found: list[str] = []
    for m in re.finditer(r"\\([A-Za-z]+)|(?<![A-Za-z\\])([A-Za-z])(?![A-Za-z])", f):
        name = m.group(1)
        if name is not None:
            if name in GREEK:
                found.append("\\" + name)
        elif m.group(2) not in NOT_SYMBOLS:
            found.append(m.group(2))
    return found


def numbers_in(sentence: Sentence) -> int:
    prose = re.sub(r"⟦\d+⟧", " ", sentence.text)
    count = len(re.findall(r"(?<![\w.])\d[\d,.]*\d?%?(?![\w])", prose))
    for formula in sentence.math:
        if re.search(r"\d", re.sub(r"_\{[^}]*\}|_\w|\^\{?\d\}?", "", formula)):
            count += 1
    return count


def syllables(word: str) -> int:
    if word.startswith("⟦"):
        return 2
    return max(1, len(WORD_SYLLABLES.findall(word.lower())) - (1 if word.lower().endswith("e") else 0))


def grade(sentences: list[Sentence]) -> float:
    """Flesch-Kincaid grade level of a list of sentences, formulas counted as two-syllable words."""
    words = [w for s in sentences for w in TOKEN.findall(s.text)]
    if not words or not sentences:
        return 0.0
    return 0.39 * len(words) / len(sentences) + 11.8 * sum(syllables(w) for w in words) / len(words) - 15.59


# --- Running the checks on a page ---------------------------------------------------------


@dataclass
class Report:
    page: str
    paragraphs: list[Paragraph]
    sentences: int
    flagged: int
    mean_words: float
    grades: dict[str, float]
    counts: Counter[str]

    @property
    def flagged_share(self) -> float:
        return self.flagged / self.sentences if self.sentences else 0.0


def analyse(page: str, raw: str, limits: dict[str, Any], allowed: set[str], acronyms: set[str]) -> Report:
    seen_symbols: set[str] = set(allowed)
    seen_acronyms: set[str] = set()
    paragraphs: list[Paragraph] = []
    for section, source in prose_blocks(raw):
        text, math, bold = plain(source)
        sentences: list[Sentence] = []
        new_concepts = len(bold)
        for chunk in split_sentences(text):
            ids = [int(i) for i in re.findall(r"⟦(\d+)⟧", chunk)]
            s = Sentence(chunk, [math[i] if i < len(math) else "display" for i in ids], ids, len(TOKEN.findall(chunk)))
            check_sentence(s, limits, seen_symbols)
            new_concepts += sum(1 for f in s.flags if f.startswith("new symbol"))
            for acronym in re.findall(r"\b[A-Z][A-Z0-9]*[A-Z]\b", re.sub(r"⟦\d+⟧", " ", chunk)):
                if acronym in acronyms and acronym not in seen_acronyms:
                    seen_acronyms.add(acronym)
                    new_concepts += 1
            sentences.append(s)
        words = sum(s.words for s in sentences)
        p = Paragraph(section, source, sentences, bold, words)
        if words > limits["max_paragraph_words"]:
            p.flags.append(f"paragraph of {words} words (limit {limits['max_paragraph_words']})")
        if new_concepts > limits["max_new_concepts_per_paragraph"]:
            p.flags.append(
                f"{new_concepts} new concepts in one paragraph (limit {limits['max_new_concepts_per_paragraph']}): "
                + ", ".join(bold + [f.split(" ")[2] for s in sentences for f in s.flags if f.startswith("new symbol")])
            )
        paragraphs.append(p)
    sentences = [s for p in paragraphs for s in p.sentences]
    by_section: dict[str, list[Sentence]] = {}
    for p in paragraphs:
        by_section.setdefault(p.section, []).extend(p.sentences)
    grades = {sec: round(grade(ss), 1) for sec, ss in by_section.items() if len(ss) >= 3}
    counts: Counter[str] = Counter(f.split(" (")[0].split(":")[0] for s in sentences for f in s.flags)
    counts.update(f.split(" (")[0].split(":")[0] for p in paragraphs for f in p.flags)
    for g in grades.values():
        if g > limits["grade_max"]:
            counts[f"section grade above {limits['grade_max']}"] += 1
    flagged = sum(1 for s in sentences if any(not f.startswith("note") for f in s.flags))
    mean = sum(s.words for s in sentences) / len(sentences) if sentences else 0.0
    return Report(page, paragraphs, len(sentences), flagged, round(mean, 1), grades, counts)


def check_sentence(s: Sentence, limits: dict[str, Any], seen_symbols: set[str]) -> None:
    prose = re.sub(r"⟦\d+⟧", "X", s.text)
    if s.words > limits["max_sentence_words"]:
        s.flags.append(f"long sentence: {s.words} words (limit {limits['max_sentence_words']})")
    punct = len(re.findall(r"[,;:()]|—| – ", prose))
    if punct > limits["max_sentence_punctuation"]:
        s.flags.append(f"overloaded: {punct} commas, colons, semicolons, parentheses or dashes")
    chained = len(re.findall(r"[;:](?=\s)", prose))
    if chained > limits["max_chained_clauses"]:
        s.flags.append(f"chained clauses: {chained} colons or semicolons")
    if len(s.math) > limits["max_math_per_sentence"]:
        s.flags.append(f"formula pile-up: {len(s.math)} formulas (limit {limits['max_math_per_sentence']})")
    numbers = numbers_in(s)
    if numbers > limits["max_numbers_per_sentence"]:
        s.flags.append(f"number pile-up: {numbers} numbers (limit {limits['max_numbers_per_sentence']})")
    if s.text.startswith("⟦"):
        s.flags.append("starts with a formula")
    head = prose.split(":")[0] if ":" in prose else ""
    if head and head.count(",") >= 2 and len(TOKEN.findall(head)) <= 20:
        s.flags.append("front-loaded list before a colon")
    opener = OPENERS.match(s.text)
    if opener:
        before = s.text.split(",")[0]
        n = len(TOKEN.findall(before))
        if "," in s.text and n > limits["max_opening_clause_words"]:
            s.flags.append(f"long opening: {n} words before the main clause")
    noms = [w for w in NOMINALIZATION.findall(prose.lower()) if w.rstrip("s") not in TECHNICAL_NOUNS]
    if len(noms) > limits["max_nominalizations_per_sentence"]:
        s.flags.append(f"nominalizations: {', '.join(noms)}")
    new = 0
    for i, formula in enumerate(s.math):
        for sym in symbols_in(formula):
            if sym in seen_symbols:
                continue
            seen_symbols.add(sym)
            new += 1
            if not introduced(s.text, s.ids[i], formula):
                s.flags.append(f"new symbol {sym} not explained in words where it first appears")
    if new > limits["max_new_symbols_per_sentence"]:
        s.flags.append(f"{new} new symbols in one sentence (limit {limits['max_new_symbols_per_sentence']})")


def introduced(text: str, index: int, formula: str) -> bool:
    """Is the formula at this position explained in words right where it appears?

    It is when a definition word sits next to it ("where $n$ is", "write $X$ for", "$\\delta$, the
    slack") or, for a single symbol, when a determiner names it ("some value $a$", "the noise
    multiplier $\\sigma$"). Naming a compound formula ("the squared deviation $(X - \\mu)^2$") does not
    explain the symbols inside it.
    """
    m = re.search(rf"⟦{index}⟧", text)
    if not m:
        return True
    before = re.sub(r"⟦\d+⟧", "X", text[: m.start()])
    after = re.sub(r"⟦\d+⟧", "X", text[m.end() :])
    near_before = " ".join(before.split()[-6:])
    near_after = " ".join(after.split()[:6])
    if re.search(
        r"\b(where|write|writes|call|called|calls|denote|denoted|denotes|let|named|written)\b",
        near_before,
        re.IGNORECASE,
    ):
        return True
    if re.match(
        r"^\s*(,\s*(the|a|an|which|where)\b|\(|(is|are) (the|a|an)\b|denotes|stands for|for the\b)", near_after
    ):
        return True
    single = re.fullmatch(r"\s*\\?[A-Za-z]+(\s*[_^]\s*(\{[^}]*\}|\w))*\s*", formula) is not None
    return single and INTRODUCED_BY_NOUN.search(before.rstrip()) is not None


# --- Command line -------------------------------------------------------------------------


def load(book: str) -> tuple[dict[str, Any], set[str]]:
    config = tomllib.loads(CONFIG.read_text())
    profile = cast(dict[str, str], config["books"]).get(book, "full")
    limits = cast(dict[str, Any], config["profiles"])[profile]
    allowed = set(cast(dict[str, list[str]], config.get("allowed_symbols", {})).get(book, []))
    return limits, allowed


def pages_of(book: str) -> list[str]:
    terms.layout.book = ROOT / book
    return [f for f in terms.book_files() if Path(f).name not in ("index.md", "study_plan.md")]


def show(report: Report, worst: int, limits: dict[str, Any]) -> None:
    status = "FAIL" if failing(report, limits) else "ok"
    print(
        f"\n## {report.page}  [{status}]  {report.flagged}/{report.sentences} sentences flagged "
        f"({report.flagged_share:.0%}), mean {report.mean_words} words"
    )
    hard = {s: g for s, g in report.grades.items() if g > limits["grade_max"]}
    if hard:
        print("   reading grade above limit: " + "; ".join(f"{s or '(top)'} {g}" for s, g in hard.items()))
    if report.counts:
        print("   " + ", ".join(f"{k} {v}" for k, v in report.counts.most_common()))
    ranked = sorted(report.paragraphs, key=lambda p: -(len(p.flags) + sum(len(s.flags) for s in p.sentences)))
    for p in ranked[:worst]:
        n = len(p.flags) + sum(len(s.flags) for s in p.sentences)
        if not n:
            break
        print(f"\n   [{n}] {p.section}: {p.source[:110]}...")
        for f in p.flags:
            print(f"       ¶ {f}")
        for s in p.sentences:
            for f in s.flags:
                print(f"       • {f}\n         in: {s.text[:140]}")


def failing(report: Report, limits: dict[str, Any]) -> bool:
    return (
        report.flagged_share > limits["page_flagged_sentence_share"]
        or report.mean_words > limits["mean_sentence_words"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["report", "check", "text"])
    parser.add_argument("text", nargs="?", help="for `text`: the prose to analyse")
    parser.add_argument("--book", default="book-practitioner", help="book-full, book-exercises or book-practitioner")
    parser.add_argument("--page", help="one page, relative to the book's folder")
    parser.add_argument("--worst", type=int, default=5, help="paragraphs to show per page")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    book: str = args.book
    limits, allowed = load(book)
    acronyms = set(terms.load_terms()[0])
    if args.command == "text":
        reports = [analyse("(text)", cast(str, args.text) or sys.stdin.read(), limits, allowed, acronyms)]
    else:
        pages = [cast(str, args.page)] if args.page else pages_of(book)
        reports = [analyse(p, (ROOT / book / p).read_text(), limits, allowed, acronyms) for p in pages]
    if args.json:
        print(json.dumps([asdict(r) | {"failing": failing(r, limits)} for r in reports], indent=1, default=str))
    elif args.command == "check":
        bad = [r for r in reports if failing(r, limits)]
        for r in sorted(reports, key=lambda r: -r.flagged_share):
            print(
                f"{'FAIL' if r in bad else 'ok  '}  {r.flagged_share:4.0%} flagged  mean {r.mean_words:4} words  {r.page}"
            )
        print(f"{len(bad)} of {len(reports)} pages fail")
        sys.exit(1 if bad else 0)
    else:
        for r in sorted(reports, key=lambda r: -r.flagged_share):
            show(r, args.worst, limits)
        total = sum(r.sentences for r in reports)
        print(f"\n{sum(r.flagged for r in reports)}/{total} sentences flagged across {len(reports)} pages")


if __name__ == "__main__":
    main()
