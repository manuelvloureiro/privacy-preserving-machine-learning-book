"""The computed content of the study note notes/ppml-2026-review/: a systematic review of what was published on
privacy-preserving machine learning in 2026.

    uv run python code/notes/ppml_2026_review.py [--out DIR]

The review's records come from one bibliographic database, OpenAlex, searched on 2 November 2026 with the thirteen
search strings in ppml_2026_review_search.py, which also screens the records and writes the frozen table this
script reads, code/notes/ppml_2026_review_records.csv. This script holds the screening and classification rules
(so that the test can check them on examples), reads the frozen table, computes every number the note quotes, draws
its figures and writes the Markdown tables it includes:

- the flow of records from the searches to the included set (_flow.md), with the count of each search string;
- the precision of the automatic screening, estimated on a hand-screened random sample with a Clopper-Pearson
  interval (_screening.md), from ppml_2026_review_screening.csv;
- the themes of the included records, how often each occurs and with which others (_themes.md, themes.svg,
  cooccurrence.svg), where each record belongs (_sources.md, _venues.md), the application domains and model
  types (_domains.md), and the records per month (months.svg).

Output: the SVG figures and Markdown snippets, written to _build/figures/notes/ppml-2026-review/ (or --out), and
every number the note quotes, printed.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

matplotlib.rcParams["svg.hashsalt"] = "ppml_2026_review"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

NOTE = "ppml-2026-review"
HERE = Path(__file__).resolve().parent
FIGURES = HERE.parents[1] / "_build" / "figures" / "notes" / NOTE
RECORDS = HERE / "ppml_2026_review_records.csv"
SEARCH = HERE / "ppml_2026_review_search.json"
SCREENING = HERE / "ppml_2026_review_screening.csv"
SEARCH_DATE = "2026-11-02"

# --- Screening rules ----------------------------------------------------------------------------------------------
#
# A record is included when it is a paper (not a dataset, a thesis, software or an editorial note), in English, not
# retracted, its title or abstract names a machine-learning object (a learning cue), and it is about privacy: its
# title or abstract names a privacy technique or attack (a technique cue), or its title has a privacy word. Federated
# learning and the phrase "privacy-preserving" are settings, not techniques: a paper that only has them is included
# only when privacy is in its title. The cues are regular expressions on the lower-cased text.

TECHNIQUE_CUES: dict[str, str] = {
    "dp": r"differential(?:ly)?[- ]priva|\bdp-?sgd\b|\bldp\b|\brdp\b|r[ée]nyi (?:differential|privacy)|\bzcdp\b|\bf-dp\b",
    "attack": (
        r"membership inference|gradient (?:inversion|leakage)|model inversion|reconstruction attack|attribute inference|"
        r"(?:training[- ])?data extraction|memoriz|inference attack|property inference|privacy (?:leakage|attack)|"
        r"re-?identification|deanonymi[sz]|linkage attack"
    ),
    "audit": r"privacy audit|auditing|empirical privacy|privacy (?:estimation|measurement)|canar(?:y|ies)",
    "crypto": (
        r"homomorphic|multi-?party computation|secure (?:computation|inference|two-party|multi-?party|aggregation)|secret shar|garbled|"
        r"trusted execution|\btees?\b|confidential computing|encrypted (?:inference|training|computation|data)|"
        r"functional encryption|\bfhe\b|\bmpc\b|\bsmpc\b|zero-knowledge|oblivious"
    ),
    "unlearning": r"unlearn|right to be forgotten|data deletion",
    "synthetic": r"synthetic (?:data|text|tabular|images?|records)|data synthesis|synthetic data generation",
    "anonymization": r"anonymi[sz]|k-anonymity|pseudonymi[sz]",
}
SETTING_CUES: dict[str, str] = {
    "federated": r"federated|\bsplit learning\b",
    "general": r"privacy[- ]preserving|privacy[- ]protect|privacy[- ]enhancing|private (?:training|inference|learning|fine-?tuning|prediction)",
}
TITLE_PRIVACY = (
    r"privacy|private|privately|anonym|confidential|leak|unlearn|secure|encrypt|membership inference|inversion|memoriz"
)
LEARNING_CUE = (
    r"machine learning|deep learning|neural|language model|\bllms?\b|federated|training|classif|regression|transformer|"
    r"\bgnns?\b|recommend|synthetic data|generative|diffusion|fine[- ]?tun|embedding|\bmodels?\b|learning|predict"
)
PAPER_TYPES = ("article", "preprint", "conference-paper", "review", "book-chapter")

# The themes of the synthesis, each a regular expression on the same text; a record can carry several.
THEMES: dict[str, tuple[str, str]] = {
    "training": (
        "Differentially private learning",
        (
            r"(?:differential(?:ly)?[- ]priva|\bdp\b).{0,80}(?:training|fine-?tun|stochastic gradient|optimi[sz]|gradient|"
            r"accountant|accounting|composition|amplification|clipping|learning rate|hyperparameter|learn|model|neural|network|"
            r"classif|regression|estimator|algorithm|embedding|\bllm|language model)|"
            r"\bdp-?sgd\b|\bdp-?ftrl\b|private(?:ly)? (?:training|fine-?tuning|optimi[sz]ation|learning|learn|stochastic|sgd)|"
            r"(?:training|fine-?tun).{0,60}(?:differential(?:ly)?[- ]priva|\bdp\b)"
        ),
    ),
    "federated": ("Federated learning", r"federated|split learning|secure aggregation"),
    "attacks": (
        "Privacy attacks and defences",
        (
            r"membership inference|gradient (?:inversion|leakage)|model inversion|reconstruction attack|attribute inference|"
            r"data extraction|memoriz|inference attack|property inference|re-?identification|deanonymi[sz]|linkage attack"
        ),
    ),
    "audit": (
        "Auditing and measuring privacy",
        r"privacy audit|auditing|empirical privacy|privacy (?:estimation|measurement)|canar(?:y|ies)",
    ),
    "crypto": (
        "Cryptographic and hardware protection",
        (
            r"homomorphic|multi-?party computation|secure (?:computation|inference|two-party|multi-?party|aggregation)|secret shar|garbled|"
            r"trusted execution|\btees?\b|confidential computing|encrypted (?:inference|training|computation|data)|functional encryption|"
            r"\bfhe\b|\bmpc\b|\bsmpc\b|zero-knowledge|oblivious"
        ),
    ),
    "synthetic": (
        "Synthetic data and data release",
        r"synthetic (?:data|text|tabular|images?|records)|data synthesis|data release|query release|data publishing",
    ),
    "unlearning": ("Machine unlearning", r"unlearn|right to be forgotten|data deletion"),
    "local": (
        "Local and shuffle differential privacy",
        r"local(?:ly)? differential|\bldp\b|shuffle model|shuffling|randomi[sz]ed response",
    ),
    "mechanisms": (
        "Mechanisms, definitions and theory",
        (
            r"(?:laplace|gaussian|exponential|staircase|discrete) mechanism|lower bound|minimax|sample complexity|mean estimation|"
            r"\bhistogram|composition theorem|privacy loss|privacy profile|f-dp|r[ée]nyi|zero-?concentrated|\bpufferfish|"
            r"metric privacy|geo-?indistinguishab|privacy amplification|hockey-?stick|trade-?off function|utility bound|optimal mechanism"
        ),
    ),
    "survey": (
        "Surveys and systematizations",
        r"\bsurvey\b|systematic (?:literature )?review|systematization|\bsok\b|taxonomy|roadmap|tutorial|\boverview\b|position paper|scoping review|meta-analysis",
    ),
}
TITLE_ONLY_THEMES = ("survey",)  # a survey is one by its title; "survey data" in an abstract is not

# Application domains and model types, from the same text; a record can carry several or none.
DOMAINS: dict[str, tuple[str, str]] = {
    "llm": (
        "Language models",
        r"language model|\bllms?\b|large language|\bgpt|transformer|in-context|prompt|retrieval-augmented|\brag\b|text generation|\bnlp\b|chatbot|foundation model",
    ),
    "vision": (
        "Images and video",
        r"\bimages?\b|\bvision\b|\bcnn|imagenet|cifar|\bface\b|facial|\bvideo|medical imaging|segmentation|object detection",
    ),
    "graph": (
        "Graphs",
        r"graph neural|\bgnns?\b|knowledge graph|graph (?:data|learning|embedding|structure)|social network",
    ),
    "tabular": (
        "Tables and statistics",
        r"tabular|\btables?\b|census|microdata|relational|statistical (?:release|queries|analysis)|\bsurvey data\b",
    ),
    "health": (
        "Health and medicine",
        r"medical|clinical|\bhealth|patient|\behrs?\b|electronic health|genom|hospital|disease|biomedical|radiolog|cancer|diagnos",
    ),
    "finance": ("Finance", r"financ|\bbank|credit (?:card|scor|risk)|fraud|insurance|\bloans?\b|stock market"),
    "iot": (
        "Internet of things, edge and vehicles",
        r"\biot\b|internet of things|edge (?:devices?|computing|intelligence|nodes?)|vehicular|\bvehicles?\b|smart grid|sensor networks?|wireless|\b[56]g\b|\buavs?\b|\bdrones?\b|industrial|smart (?:city|cities|home)",
    ),
    "recsys": ("Recommendation", r"recommend"),
    "speech": ("Speech and audio", r"\bspeech\b|\baudio\b|speaker|voice"),
    "location": (
        "Locations, trajectories and time series",
        r"time[- ]series|trajector|mobility|spatio-?temporal|\bgps\b|location (?:data|privacy|traces?|information)",
    ),
}

_TECHNIQUE = {k: re.compile(v) for k, v in TECHNIQUE_CUES.items()}
_SETTING = {k: re.compile(v) for k, v in SETTING_CUES.items()}
_TITLE_PRIVACY = re.compile(TITLE_PRIVACY)
_LEARNING = re.compile(LEARNING_CUE)
_THEMES = {k: re.compile(v[1]) for k, v in THEMES.items()}
_DOMAINS = {k: re.compile(v[1]) for k, v in DOMAINS.items()}


def normalize_title(title: str) -> str:
    """A title reduced to its letters and digits, to find the same paper under two records."""
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def privacy_cues(text: str) -> list[str]:
    """The technique and setting cues the lower-cased text matches, techniques first."""
    return [k for k, p in _TECHNIQUE.items() if p.search(text)] + [k for k, p in _SETTING.items() if p.search(text)]


def has_learning_cue(text: str) -> bool:
    return _LEARNING.search(text) is not None


def screen(title: str, abstract: str, kind: str, language: str, retracted: bool) -> str:
    """Why a record is excluded, or "included"."""
    if retracted:
        return "retracted"
    if kind not in PAPER_TYPES:
        return "not a paper"
    if language != "en":
        return "not in English"
    text = f"{title} {abstract}".lower()
    cues = privacy_cues(text)
    if not cues:
        return "no privacy technique or setting"
    if not has_learning_cue(text):
        return "no machine learning"
    if not any(c in TECHNIQUE_CUES for c in cues) and not _TITLE_PRIVACY.search(title.lower()):
        return "privacy only as a setting"
    return "included"


def themes_of(title: str, abstract: str) -> list[str]:
    """The themes a record carries, in the order of THEMES; "other" when none matches."""
    text = f"{title} {abstract}".lower()
    found = [k for k, p in _THEMES.items() if p.search(title.lower() if k in TITLE_ONLY_THEMES else text)]
    return found or ["other"]


def domains_of(title: str, abstract: str) -> list[str]:
    text = f"{title} {abstract}".lower()
    return [k for k, p in _DOMAINS.items() if p.search(text)]


# --- The frozen table ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    """One included record of the review, as ppml_2026_review_records.csv stores it."""

    id: str
    databases: tuple[str, ...]  # "openalex", "arxiv", or both
    title: str
    date: str
    kind: str
    source: str
    source_kind: str
    cited_by: int  # -1 when unknown: arXiv has no citation counts
    queries: tuple[str, ...]
    themes: tuple[str, ...]
    domains: tuple[str, ...]
    first_author: str
    authors: int
    has_abstract: bool
    topic: str


def read_records(path: Path = RECORDS) -> list[Record]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        Record(
            id=r["id"],
            databases=tuple(r["databases"].split(";")),
            title=r["title"],
            date=r["date"],
            kind=r["kind"],
            source=r["source"],
            source_kind=r["source_kind"],
            cited_by=int(r["cited_by"]),
            queries=tuple(r["queries"].split(";")),
            themes=tuple(r["themes"].split(";")),
            domains=tuple(d for d in r["domains"].split(";") if d),
            first_author=r["first_author"],
            authors=int(r["authors"]),
            has_abstract=r["has_abstract"] == "1",
            topic=r["topic"],
        )
        for r in rows
    ]


# --- Screening precision ------------------------------------------------------------------------------------------


def clopper_pearson(successes: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    """The exact two-sided confidence interval for a proportion, from the Beta quantiles."""
    if trials == 0:
        return (0.0, 1.0)
    alpha = 1 - level
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return (lower, upper)


def read_screening(path: Path = SCREENING) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# --- The reading rule of the synthesis ----------------------------------------------------------------------------
#
# For each theme the synthesis reads, by abstract: every record of the theme from one of the field's core venues (a random
# twelve when there are more), the five most cited records of the theme in OpenAlex, and eight records of the theme drawn
# at random from arXiv. The selection is seeded, so it is the same on every run.

CORE_VENUES = (
    "Proceedings on Privacy Enhancing Technologies",
    "USENIX",
    "Computer and Communications Security",
    "Symposium on Security and Privacy",
    "Network and Distributed System Security",
    "International Conference on Learning Representations",
    "International Conference on Machine Learning",
    "Neural Information Processing Systems",
    "Artificial Intelligence and Statistics",
    "AAAI Conference on Artificial Intelligence",
    "Transactions on Machine Learning Research",
    "IEEE Transactions on Information Forensics and Security",
    "IEEE Transactions on Dependable and Secure Computing",
    "Journal of Privacy and Confidentiality",
    "Transactions on Privacy and Security",
)
CORE_READ, CITED_READ, ARXIV_READ = 12, 5, 8


def is_core_venue(source: str) -> bool:
    return any(c.lower() in source.lower() for c in CORE_VENUES)


def reading_list(
    records: list[Record], theme: str, seed: int = 1
) -> tuple[list[Record], list[Record], list[Record], int]:
    """The records read for a theme: core-venue records, most cited, random arXiv; and how many core-venue records there are."""
    rows = [r for r in records if theme in r.themes and r.has_abstract]
    rng = random.Random(seed)
    core = [r for r in rows if is_core_venue(r.source)]
    core_pick = core if len(core) <= CORE_READ else rng.sample(core, CORE_READ)
    cited = sorted([r for r in rows if r.cited_by >= 0 and r not in core_pick], key=lambda r: (-r.cited_by, r.date))[
        :CITED_READ
    ]
    arxiv = [r for r in rows if r.source_kind == "arXiv" and r not in core_pick and r not in cited]
    arxiv_pick = rng.sample(arxiv, min(ARXIV_READ, len(arxiv)))
    return core_pick, cited, arxiv_pick, len(core)


CORE_VENUE_NAMES = {
    "Proceedings on Privacy Enhancing Technologies": "Proceedings on Privacy Enhancing Technologies",
    "USENIX": "USENIX Security Symposium",
    "Computer and Communications Security": "ACM Conference on Computer and Communications Security",
    "Symposium on Security and Privacy": "IEEE Symposium on Security and Privacy",
    "Network and Distributed System Security": "Network and Distributed System Security Symposium",
    "International Conference on Learning Representations": "International Conference on Learning Representations",
    "International Conference on Machine Learning": "International Conference on Machine Learning",
    "Neural Information Processing Systems": "Conference on Neural Information Processing Systems",
    "Artificial Intelligence and Statistics": "Conference on Artificial Intelligence and Statistics",
    "AAAI Conference on Artificial Intelligence": "AAAI Conference on Artificial Intelligence",
    "Transactions on Machine Learning Research": "Transactions on Machine Learning Research",
    "IEEE Transactions on Information Forensics and Security": "IEEE Transactions on Information Forensics and Security",
    "IEEE Transactions on Dependable and Secure Computing": "IEEE Transactions on Dependable and Secure Computing",
    "Journal of Privacy and Confidentiality": "Journal of Privacy and Confidentiality",
    "Transactions on Privacy and Security": "ACM Transactions on Privacy and Security",
}


def core_venues_table(records: list[Record]) -> str:
    lines = ["| Core venue | Included records |", "|:---|---:|"]
    for pattern in CORE_VENUES:
        count = sum(1 for r in records if pattern.lower() in r.source.lower())
        lines.append(f"| {CORE_VENUE_NAMES[pattern]} | {count:,} |")
    lines.append("")
    lines.append(
        ": The fifteen venues the review treats as the core of the field, and the included records at each. Two of the three large "
        "machine-learning conferences have none and the third has five, because OpenAlex indexes little of their 2026 proceedings. {#tbl-core}"
    )
    return "\n".join(lines) + "\n"


def reading_table(records: list[Record]) -> str:
    lines = [
        "| Theme | Records with an abstract | At a core venue | Read: core venue | Read: most cited | Read: random arXiv |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for key, (name, _) in THEMES.items():
        core, cited, arxiv, n_core = reading_list(records, key)
        rows = sum(1 for r in records if key in r.themes and r.has_abstract)
        lines.append(f"| {name} | {rows:,} | {n_core} | {len(core)} | {len(cited)} | {len(arxiv)} |")
    lines.append("")
    lines.append(
        ": What was read for each theme: every record from a core venue of the field (a random twelve when there are more), "
        "the five most cited in OpenAlex, and eight drawn at random from arXiv. {#tbl-reading}"
    )
    return "\n".join(lines) + "\n"


# --- Counting ----------------------------------------------------------------------------------------------------


def theme_counts(records: list[Record]) -> Counter[str]:
    return Counter(t for r in records for t in r.themes)


def domain_counts(records: list[Record]) -> Counter[str]:
    return Counter(d for r in records for d in r.domains)


def cooccurrence(records: list[Record], keys: list[str]) -> np.ndarray:
    """How many records carry each pair of themes; the diagonal holds each theme's count."""
    index = {k: i for i, k in enumerate(keys)}
    m = np.zeros((len(keys), len(keys)), dtype=np.int64)
    for r in records:
        present = [index[t] for t in r.themes if t in index]
        for i in present:
            for j in present:
                m[i, j] += 1
    return m


def monthly_counts(records: list[Record]) -> dict[str, int]:
    return dict(sorted(Counter(r.date[:7] for r in records).items()))


def share(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def median(values: list[int]) -> float:
    """The median of the known values; a count of -1 is unknown and left out."""
    known = [v for v in values if v >= 0]
    return float(np.median(known)) if known else 0.0


# --- Figures -----------------------------------------------------------------------------------------------------


def figure_themes(records: list[Record], out: Path) -> None:
    counts = theme_counts(records)
    keys: list[str] = list(THEMES) + ["other"]
    names: list[str] = [THEMES[k][0] if k in THEMES else "No theme matched" for k in keys]
    values: list[int] = [counts.get(k, 0) for k in keys]
    order: list[int] = sorted(range(len(keys)), key=lambda i: values[i])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(range(len(order)), [values[i] for i in order], color="#4c72b0")
    ax.set_yticks(range(len(order)), [names[i] for i in order])
    for row, i in enumerate(order):
        ax.text(values[i] + max(values) * 0.01, row, f"{values[i]:,}", va="center", fontsize=9)
    ax.set_xlabel("Included records that carry the theme (a record can carry several)")
    ax.set_xlim(0, max(values) * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "themes.svg")
    plt.close(fig)


def figure_cooccurrence(records: list[Record], out: Path) -> None:
    keys = [k for k in THEMES if k != "survey"]
    m = cooccurrence(records, keys)
    diag = np.diag(m).astype(float)
    shares = m / np.where(diag[:, None] == 0, 1, diag[:, None])  # row i: share of theme i's records that also carry j
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    labels = [THEMES[k][0] for k in keys]
    ax.set_xticks(range(len(keys)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(keys)), labels, fontsize=8)
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i != j:
                ax.text(
                    j,
                    i,
                    f"{shares[i, j]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if shares[i, j] < 0.6 else "white",
                )
            else:
                ax.text(j, i, f"{int(diag[i]):,}", ha="center", va="center", fontsize=7, color="white")
    ax.set_xlabel("also carries this theme")
    ax.set_ylabel("of the records with this theme (count on the diagonal)")
    fig.colorbar(im, ax=ax, fraction=0.04, label="share")
    fig.tight_layout()
    fig.savefig(out / "cooccurrence.svg")
    plt.close(fig)


def figure_months(records: list[Record], out: Path) -> None:
    months = monthly_counts(records)
    kinds = ["journal", "conference", "arXiv", "other repository", "book", "other"]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bottom = np.zeros(len(months))
    for kind, color in zip(kinds, colors):
        values = np.array(
            [sum(1 for r in records if r.date[:7] == m and r.source_kind == kind) for m in months], dtype=float
        )
        ax.bar(list(months), values, bottom=bottom, label=kind, color=color)
        bottom += values
    ax.set_ylabel("Included records")
    ax.set_xlabel("Publication month, as OpenAlex records it")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out / "months.svg")
    plt.close(fig)


# --- Tables ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Search:
    """The counts the search script wrote: each search string's records in each database and each stage of the screening."""

    date: str
    queries: dict[
        str, tuple[str, str, int, int]
    ]  # key -> (OpenAlex string, arXiv string, OpenAlex records, arXiv records)
    returned: int
    distinct_ids: int
    distinct_titles: int
    exclusions: dict[str, int]
    included: int


def read_search(path: Path = SEARCH) -> Search:
    raw = cast(dict[str, Any], json.loads(path.read_text()))
    queries = cast(dict[str, dict[str, Any]], raw["queries"])
    return Search(
        date=str(raw["search_date"]),
        queries={
            k: (str(q["string"]), str(q["arxiv_string"]), int(q["openalex"]["count"]), int(q["arxiv"]["count"]))
            for k, q in queries.items()
        },
        returned=int(raw["returned"]),
        distinct_ids=int(raw["distinct_ids"]),
        distinct_titles=int(raw["distinct_titles"]),
        exclusions={str(k): int(v) for k, v in cast(dict[str, Any], raw["exclusions"]).items()},
        included=int(raw["included"]),
    )


def flow_table(search: Search) -> str:
    """The search strings and the records each returned in each database."""
    lines = [
        "| String | OpenAlex search, over title and abstract | OpenAlex records | arXiv records |",
        "|:---|:---|---:|---:|",
    ]
    for key, (string, _, openalex, arxiv) in search.queries.items():
        lines.append(f"| {key} | `{string}` | {openalex:,} | {arxiv:,} |")
    lines.append("")
    lines.append(
        f": The thirteen search strings, as run on OpenAlex (publication year 2026) on {SEARCH_DATE}, and the records each returned "
        "there and on arXiv (first submitted in 2026). The arXiv form of each string, with its field prefixes, is in the search script. "
        "Zero OpenAlex records means the string was not run there, because the quota ran out. {#tbl-queries}"
    )
    return "\n".join(lines) + "\n"


def exclusion_table(search: Search, records: list[Record]) -> str:
    """The records at each stage, from the searches to the included set."""
    lines = ["| Stage | Records |", "|:---|---:|"]
    lines.append(f"| Returned by the searches, with repeats | {search.returned:,} |")
    lines.append(f"| Distinct records (one per identifier in its database) | {search.distinct_ids:,} |")
    lines.append(f"| Distinct papers (one per title) | {search.distinct_titles:,} |")
    for reason, count in search.exclusions.items():
        lines.append(f"| Excluded: {reason} | {count:,} |")
    lines.append(f"| Included | {len(records):,} |")
    lines.append("")
    lines.append(": The flow of records from the searches to the included set. {#tbl-flow}")
    return "\n".join(lines) + "\n"


def screening_table(screening: list[dict[str, str]]) -> str:
    lines = [
        "| Sample | Records read | Decision agrees with the rule | Share | 95% interval |",
        "|:---|---:|---:|---:|:---|",
    ]
    for sample, label in (
        ("included", "Included by the rule"),
        ("excluded", "Excluded by the rule for want of a learning cue"),
    ):
        rows = [r for r in screening if r["sample"] == sample]
        agree = sum(1 for r in rows if r["decision"] == ("include" if sample == "included" else "exclude"))
        lo, hi = clopper_pearson(agree, len(rows))
        lines.append(f"| {label} | {len(rows)} | {agree} | {share(agree, len(rows)):.0%} | {lo:.0%} to {hi:.0%} |")
    lines.append("")
    lines.append(
        ": The hand screening of two random samples of records, and how often the reader agreed with the rule. The interval is the exact Clopper–Pearson interval. {#tbl-screening}"
    )
    return "\n".join(lines) + "\n"


def themes_table(records: list[Record]) -> str:
    counts = theme_counts(records)
    n = len(records)
    lines = ["| Theme | What the rule looks for | Records | Share |", "|:---|:---|---:|---:|"]
    for key, (name, _) in THEMES.items():
        lines.append(f"| {name} | {THEME_GLOSS[key]} | {counts.get(key, 0):,} | {share(counts.get(key, 0), n):.0%} |")
    lines.append(f"| No theme matched | | {counts.get('other', 0):,} | {share(counts.get('other', 0), n):.0%} |")
    lines.append("")
    lines.append(
        ": The themes of the included records. A record can carry several themes, so the shares add up to more than one. {#tbl-themes}"
    )
    return "\n".join(lines) + "\n"


THEME_GLOSS = {
    "training": "DP-SGD and other private training or fine-tuning, and the accounting of its budget",
    "federated": "federated or split learning, and secure aggregation",
    "attacks": "membership, attribute and property inference, reconstruction, gradient inversion, extraction and memorization",
    "audit": "auditing or estimating a guarantee from a trained model",
    "crypto": "homomorphic encryption, secure computation, secret sharing, trusted hardware and zero-knowledge proofs",
    "synthetic": "synthetic data, data release and query release",
    "unlearning": "unlearning, deletion and the right to be forgotten",
    "local": "local differential privacy, randomized response and the shuffle model",
    "mechanisms": "mechanisms, divergences, composition, lower bounds and estimation",
    "survey": "a survey, review, systematization or taxonomy, by its title",
}


def sources_table(records: list[Record]) -> str:
    counts = Counter(r.source_kind for r in records)
    n = len(records)
    lines = [
        "| Where the record was published | Records | Share | With a citation count | Cited at least once |",
        "|:---|---:|---:|---:|---:|",
    ]
    for kind in ("journal", "conference", "arXiv", "other repository", "book", "other"):
        rows = [r for r in records if r.source_kind == kind]
        known = [r.cited_by for r in rows if r.cited_by >= 0]
        lines.append(
            f"| {SOURCE_GLOSS[kind]} | {counts.get(kind, 0):,} | {share(counts.get(kind, 0), n):.0%} | {len(known):,} | "
            f"{share(sum(1 for c in known if c > 0), len(known)):.0%} |"
        )
    lines.append("")
    lines.append(
        ": The included records by the kind of venue listed as their primary location. OpenAlex gives a citation count; "
        "a record found only on arXiv has none, so the last column is the share of the records with a count. {#tbl-sources}"
    )
    return "\n".join(lines) + "\n"


SOURCE_GLOSS = {
    "journal": "a journal",
    "conference": "conference proceedings",
    "arXiv": "arXiv",
    "other repository": "another repository (Zenodo, SSRN, Research Square, ...)",
    "book": "a book or book series",
    "other": "no venue listed, or another kind",
}


def venues_table(records: list[Record], top: int = 20) -> str:
    counts = Counter(r.source for r in records if r.source and r.source_kind in ("journal", "conference"))
    lines = ["| Venue | Records | Cited at least once |", "|:---|---:|---:|"]
    for name, count in counts.most_common(top):
        cited = sum(1 for r in records if r.source == name and r.cited_by > 0)
        lines.append(f"| {name} | {count:,} | {share(cited, count):.0%} |")
    lines.append("")
    lines.append(f": The {top} journals and proceedings with the most included records. {{#tbl-venues}}")
    return "\n".join(lines) + "\n"


def domains_table(records: list[Record]) -> str:
    n = len(records)
    lines = [
        "| Domain or model type | Records | Share | With differential privacy | With federated learning | With attacks |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for key, (name, _) in DOMAINS.items():
        rows = [r for r in records if key in r.domains]
        lines.append(
            f"| {name} | {len(rows):,} | {share(len(rows), n):.0%} | "
            f"{share(sum(1 for r in rows if 'dp' in r.queries or 'training' in r.themes or 'local' in r.themes or 'mechanisms' in r.themes), len(rows)):.0%} | "
            f"{share(sum(1 for r in rows if 'federated' in r.themes), len(rows)):.0%} | "
            f"{share(sum(1 for r in rows if 'attacks' in r.themes), len(rows)):.0%} |"
        )
    none = sum(1 for r in records if not r.domains)
    lines.append(f"| No domain matched | {none:,} | {share(none, n):.0%} | | | |")
    lines.append("")
    lines.append(
        ": The application domains and model types of the included records, and the share of each that works with differential "
        "privacy (the record was returned by the differential-privacy string or carries one of its themes), with federated learning "
        "or with attacks. {#tbl-domains}"
    )
    return "\n".join(lines) + "\n"


# --- Main --------------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    records = read_records()
    search = read_search()
    assert search.included == len(records)
    screening = read_screening()
    n = len(records)
    print(f"included records: {n:,}")
    print(
        f"with an abstract: {sum(1 for r in records if r.has_abstract):,} ({share(sum(1 for r in records if r.has_abstract), n):.0%})"
    )
    counts = theme_counts(records)
    for key in list(THEMES) + ["other"]:
        print(f"theme {key}: {counts.get(key, 0):,} ({share(counts.get(key, 0), n):.1%})")
    print(f"records with one theme: {sum(1 for r in records if len(r.themes) == 1 and r.themes != ('other',)):,}")
    dcounts = domain_counts(records)
    for key in DOMAINS:
        print(f"domain {key}: {dcounts.get(key, 0):,} ({share(dcounts.get(key, 0), n):.1%})")
    print(f"no domain: {sum(1 for r in records if not r.domains):,}")
    scounts = Counter(r.source_kind for r in records)
    for kind, count in scounts.most_common():
        print(f"source {kind}: {count:,} ({share(count, n):.1%})")
    months = monthly_counts(records)
    for m, c in months.items():
        print(f"month {m}: {c:,}")
    cited = [r.cited_by for r in records if r.cited_by >= 0]
    print(
        f"citations known for {len(cited):,} records: median {median(cited):.0f}, uncited {share(sum(1 for c in cited if c == 0), len(cited)):.0%}, max {max(cited):,}"
    )
    print(
        f"at least 10 citations: {sum(1 for c in cited if c >= 10):,}; at least 50: {sum(1 for c in cited if c >= 50):,}"
    )
    dbs = Counter(";".join(r.databases) for r in records)
    for db, count in dbs.most_common():
        print(f"database {db}: {count:,} ({share(count, n):.1%})")
    for sample in ("included", "excluded"):
        rows = [r for r in screening if r["sample"] == sample]
        agree = sum(1 for r in rows if r["decision"] == ("include" if sample == "included" else "exclude"))
        lo, hi = clopper_pearson(agree, len(rows))
        print(f"screening {sample}: {agree}/{len(rows)} agree, {lo:.3f} to {hi:.3f}")

    # The pairs of themes and the domain splits the note's synthesis quotes.
    def with_theme(rows: list[Record], theme: str) -> int:
        return sum(1 for r in rows if theme in r.themes)

    fl = [r for r in records if "federated" in r.themes]
    print(
        f"federated records that also carry: training {share(with_theme(fl, 'training'), len(fl)):.0%}, crypto {share(with_theme(fl, 'crypto'), len(fl)):.0%}, "
        f"attacks {share(with_theme(fl, 'attacks'), len(fl)):.0%}, unlearning {share(with_theme(fl, 'unlearning'), len(fl)):.0%}, "
        f"none of the other themes {share(sum(1 for r in fl if r.themes == ('federated',)), len(fl)):.0%}"
    )
    for theme in ("training", "attacks", "crypto", "unlearning", "synthetic", "audit", "local", "mechanisms"):
        rows = [r for r in records if theme in r.themes]
        print(
            f"{theme}: {len(rows):,} records, {share(with_theme(rows, 'federated'), len(rows)):.0%} federated, "
            f"{share(sum(1 for r in rows if 'llm' in r.domains), len(rows)):.0%} language models, "
            f"{share(sum(1 for r in rows if r.source_kind == 'arXiv'), len(rows)):.0%} on arXiv, "
            f"{share(sum(1 for r in rows if 'health' in r.domains), len(rows)):.0%} health"
        )
    blockchain = sum(1 for r in records if "blockchain" in r.title.lower())
    print(
        f"blockchain in the title: {blockchain:,} ({share(blockchain, n):.1%}); of the federated records {share(sum(1 for r in fl if 'blockchain' in r.title.lower()), len(fl)):.1%}"
    )
    jan1 = sum(1 for r in records if r.date == "2026-01-01")
    print(f"dated exactly 2026-01-01: {jan1:,} ({share(jan1, n):.1%})")
    surveys = [r for r in records if "survey" in r.themes]
    print(
        f"surveys: {len(surveys):,}, federated {share(with_theme(surveys, 'federated'), len(surveys)):.0%}, health {share(sum(1 for r in surveys if 'health' in r.domains), len(surveys)):.0%}"
    )
    zenodo = sum(1 for r in records if r.source.startswith("Zenodo"))
    print(
        f"Zenodo: {zenodo:,} ({share(zenodo, n):.1%}); other repositories in all {sum(1 for r in records if r.source_kind == 'other repository'):,}"
    )
    one_theme = Counter(r.themes[0] for r in records if len(r.themes) == 1)
    print(f"records with exactly one theme, by theme: {dict(one_theme.most_common())}")
    arxiv_only = [r for r in records if r.databases == ("arxiv",)]
    print(
        f"found on arXiv only: {len(arxiv_only):,}; of them matched by the dp or fl string {sum(1 for r in arxiv_only if {'dp', 'fl'} & set(r.queries)):,}"
    )
    print(
        f"records whose only cue is a setting (federated or privacy-preserving) with privacy in the title: n/a; records from the fl string only: {sum(1 for r in records if r.queries == ('fl',)):,}"
    )

    core_total = sum(1 for r in records if is_core_venue(r.source))
    print(f"records at a core venue: {core_total:,} ({share(core_total, n):.1%})")
    for venue in (
        "Proceedings on Privacy Enhancing Technologies",
        "Network and Distributed System Security",
        "Symposium on Security and Privacy",
        "USENIX",
        "Computer and Communications Security",
        "AAAI Conference on Artificial Intelligence",
        "Information Forensics and Security",
        "Dependable and Secure Computing",
        "Scientific Reports",
        "Drug Delivery",
    ):
        print(f"venue {venue}: {sum(1 for r in records if venue.lower() in r.source.lower()):,}")
    for key in THEMES:
        core, cited, arxiv, n_core = reading_list(records, key)
        print(f"reading {key}: core {n_core} (read {len(core)}), cited {len(cited)}, arxiv {len(arxiv)}")

    figure_themes(records, out)
    figure_cooccurrence(records, out)
    figure_months(records, out)
    (out / "_queries.md").write_text(flow_table(search))
    (out / "_flow.md").write_text(exclusion_table(search, records))
    (out / "_screening.md").write_text(screening_table(screening))
    (out / "_themes.md").write_text(themes_table(records))
    (out / "_sources.md").write_text(sources_table(records))
    (out / "_venues.md").write_text(venues_table(records))
    (out / "_domains.md").write_text(domains_table(records))
    (out / "_reading.md").write_text(reading_table(records))
    (out / "_core_venues.md").write_text(core_venues_table(records))
    print(f"wrote figures and tables to {out}")


if __name__ == "__main__":
    main()
