"""The search and screening of the systematic review in notes/ppml-2026-review/: fetch every 2026 record that one of
the review's search strings returns, screen them with the rules of ppml_2026_review.py, and write the frozen table the
note's script reads.

    uv run python code/notes/ppml_2026_review_search.py            # reuse the cached download, rescreen
    uv run python code/notes/ppml_2026_review_search.py --fetch    # download again (the results change daily)

Two databases are searched. OpenAlex, an index of journals, proceedings and repositories, is searched with the two
broad strings (differential privacy; federated learning with a privacy word): its free quota of requests ran out on
the search day after those two, and its arXiv records then cover the narrower strings only where a paper also matches
one of the two. arXiv is searched directly with all thirteen strings, by the date of first submission. The download is
cached in data/openalex-ppml-2026/<query>/<window>.jsonl and data/openalex-ppml-2026/arxiv/<query>.jsonl (git-ignored,
like the datasets). The frozen outputs are committed next to this script: ppml_2026_review_records.csv, one row per
included paper, and ppml_2026_review_search.json, the count of each search string and of each exclusion. The searches
ran on 2 November 2026; both databases add records every day, so a later run returns more.

OpenAlex pages a filter to 10,000 results, so each of its strings is run over four windows of the publication date. The
federated-learning string's last window (October to December) was not downloaded before the quota ran out and is
missing; ppml_2026_review_search.json records its count.
"""

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, cast

from ppml_2026_review import SEARCH_DATE, domains_of, normalize_title, privacy_cues, screen, themes_of

HERE = Path(__file__).resolve().parent
CACHE = HERE.parents[1] / "data" / "openalex-ppml-2026"
BASE = "https://api.openalex.org/works"
MAILTO = "mvm.loureiro@gmail.com"  # OpenAlex's polite pool asks for a contact address
FIELDS = ",".join(
    [
        "id",
        "doi",
        "title",
        "publication_date",
        "publication_year",
        "type",
        "language",
        "is_retracted",
        "cited_by_count",
        "primary_location",
        "authorships",
        "primary_topic",
        "abstract_inverted_index",
        "indexed_in",
        "open_access",
        "referenced_works_count",
        "locations_count",
    ]
)
WINDOWS = [
    ("2026-01-01", "2026-03-31"),
    ("2026-04-01", "2026-06-30"),
    ("2026-07-01", "2026-09-30"),
    ("2026-10-01", "2026-12-31"),
]

# The search strings, each run over the title and abstract of every work with publication year 2026. OpenAlex stems
# words, so "privacy" also matches "private" where it says so in its documentation only for the same stem.
QUERIES: dict[str, str] = {
    "dp": '"differential privacy" OR "differentially private"',
    "fl": '"federated learning" AND (privacy OR attack OR leakage OR "secure aggregation")',
    "mia": '"membership inference"',
    "ppml": '"privacy-preserving machine learning" OR "privacy preserving machine learning" OR "private machine learning" OR "privacy-preserving deep learning" OR "privacy-preserving inference" OR "private inference"',
    "he": '"homomorphic encryption" AND (learning OR inference OR "neural network" OR model)',
    "mpc": '("multi-party computation" OR "multiparty computation" OR "secure computation") AND (learning OR inference OR "neural network")',
    "tee": '("trusted execution environment" OR "confidential computing") AND (learning OR inference OR "neural network")',
    "unlearn": '"machine unlearning"',
    "memor": '(memorization OR "data extraction" OR "training data") AND ("language model" OR LLM) AND privacy',
    "inversion": '"gradient inversion" OR "gradient leakage" OR "model inversion" OR "reconstruction attack" OR "attribute inference"',
    "synth": '"synthetic data" AND (privacy OR "differentially private")',
    "audit": '"privacy audit" OR "privacy auditing" OR "auditing differential privacy"',
    "split": '"split learning" AND privacy',
}


OPENALEX_QUERIES = ("dp", "fl")  # the strings run on OpenAlex before its quota ran out

# The same thirteen strings in the arXiv API's syntax: ti: is the title, abs: the abstract, all: every field.
ARXIV_QUERIES: dict[str, str] = {
    "dp": '(ti:"differential privacy" OR abs:"differential privacy" OR ti:"differentially private" OR abs:"differentially private")',
    "fl": '(ti:"federated learning" OR abs:"federated learning") AND (all:privacy OR all:attack OR all:leakage OR abs:"secure aggregation")',
    "mia": '(ti:"membership inference" OR abs:"membership inference")',
    "ppml": '(abs:"privacy-preserving machine learning" OR abs:"privacy preserving machine learning" OR abs:"private machine learning" OR abs:"privacy-preserving deep learning" OR abs:"privacy-preserving inference" OR abs:"private inference" OR ti:"private inference" OR ti:"privacy-preserving machine learning")',
    "he": '(ti:"homomorphic encryption" OR abs:"homomorphic encryption") AND (all:learning OR all:inference OR abs:"neural network" OR all:model)',
    "mpc": '(abs:"multi-party computation" OR abs:"multiparty computation" OR abs:"secure computation" OR ti:"multi-party computation") AND (all:learning OR all:inference OR abs:"neural network")',
    "tee": '(abs:"trusted execution environment" OR abs:"confidential computing" OR ti:"trusted execution") AND (all:learning OR all:inference OR abs:"neural network")',
    "unlearn": '(ti:"machine unlearning" OR abs:"machine unlearning")',
    "memor": '(all:memorization OR abs:"data extraction" OR abs:"training data") AND (abs:"language model" OR all:LLM) AND all:privacy',
    "inversion": '(abs:"gradient inversion" OR abs:"gradient leakage" OR abs:"model inversion" OR abs:"reconstruction attack" OR abs:"attribute inference" OR ti:"gradient inversion" OR ti:"model inversion")',
    "synth": '(abs:"synthetic data" OR ti:"synthetic data") AND (all:privacy OR abs:"differentially private")',
    "audit": '(abs:"privacy audit" OR abs:"privacy auditing" OR abs:"auditing differential privacy" OR ti:"privacy auditing" OR ti:"privacy audit")',
    "split": '(ti:"split learning" OR abs:"split learning") AND all:privacy',
}
ARXIV = "https://export.arxiv.org/api/query"
ARXIV_WINDOW = "submittedDate:[202601010000 TO 202611022359]"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def get(url: str, tries: int = 10) -> dict[str, Any]:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:  # a 429 or a dropped connection: wait longer each time
            print("  retry", i, str(e)[:70], flush=True)
            time.sleep(4 + 4 * i)
    raise RuntimeError(url)


def fetch(key: str, query: str) -> None:
    """Download every record the query returns, one JSON line each, one file per date window with a marker that holds
    OpenAlex's count for the window; a window with a marker is not downloaded again, so a stopped run resumes."""
    folder = CACHE / key
    folder.mkdir(parents=True, exist_ok=True)
    for lo, hi in WINDOWS:
        if (folder / f"{lo}.done").exists():
            continue
        flt = f"publication_year:2026,from_publication_date:{lo},to_publication_date:{hi},title_and_abstract.search:{query}"
        page, n, total = 1, 0, 0
        with (folder / f"{lo}.jsonl").open("w") as f:
            while True:
                params = {"filter": flt, "per-page": 200, "page": page, "select": FIELDS, "mailto": MAILTO}
                d = get(BASE + "?" + urllib.parse.urlencode(params))
                if page == 1:
                    total = int(d["meta"]["count"])
                    assert total <= 10000, (key, lo, total)
                results = cast(list[dict[str, Any]], d["results"])
                for w in results:
                    f.write(json.dumps(w) + "\n")
                    n += 1
                print(key, lo, n, "/", total, flush=True)
                if len(results) < 200:
                    break
                page += 1
                time.sleep(1.5)
        (folder / f"{lo}.done").write_text(json.dumps({"count": total, "fetched": n}))


def get_text(url: str, tries: int = 10) -> str:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read().decode("utf-8")
        except Exception as e:
            print("  retry", i, str(e)[:70], flush=True)
            time.sleep(5 + 5 * i)
    raise RuntimeError(url)


def arxiv_entry(entry: ET.Element) -> dict[str, Any]:
    """An arXiv Atom entry in the shape of an OpenAlex work, so that one screening reads both."""

    def text(tag: str) -> str:
        node = entry.find(ATOM + tag)
        return " ".join((node.text or "").split()) if node is not None else ""

    arxiv_id = re.sub(r"v\d+$", "", text("id").rsplit("/abs/", 1)[-1])  # the version is dropped
    authors = [a.find(ATOM + "name") for a in entry.findall(ATOM + "author")]
    primary = entry.find(ARXIV_NS + "primary_category")
    return {
        "id": f"arxiv:{arxiv_id}",
        "title": text("title"),
        "publication_date": text("published")[:10],
        "publication_year": 2026,
        "type": "preprint",
        "language": "en",
        "is_retracted": False,
        "cited_by_count": -1,  # arXiv has no citation counts
        "primary_location": {"source": {"display_name": "arXiv (direct search)", "type": "repository"}},
        "authorships": [{"author": {"display_name": a.text or ""}} for a in authors if a is not None],
        "primary_topic": {"display_name": primary.get("term", "") if primary is not None else ""},
        "abstract": text("summary"),
    }


def fetch_arxiv(key: str, query: str) -> None:
    """Download every arXiv record the query returns among the papers first submitted in 2026 up to the search date."""
    folder = CACHE / "arxiv"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / f"{key}.done").exists():
        return
    start, n, total = 0, 0, 0
    with (folder / f"{key}.jsonl").open("w") as f:
        while True:
            params = {"search_query": f"{query} AND {ARXIV_WINDOW}", "start": start, "max_results": 500}
            root = ET.fromstring(get_text(ARXIV + "?" + urllib.parse.urlencode(params)))
            total_node = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
            total = int(total_node.text or 0) if total_node is not None else 0
            entries = root.findall(ATOM + "entry")
            for entry in entries:
                f.write(json.dumps(arxiv_entry(entry)) + "\n")
                n += 1
            print("arxiv", key, n, "/", total, flush=True)
            start += 500
            if not entries or start >= total:
                break
            time.sleep(3)  # arXiv asks for a pause between calls
    (folder / f"{key}.done").write_text(json.dumps({"count": total, "fetched": n}))


def abstract_of(work: dict[str, Any]) -> str:
    """OpenAlex stores an abstract as word -> positions; put the words back in order."""
    if "abstract" in work:
        return str(work["abstract"])
    index = cast(dict[str, list[int]] | None, work.get("abstract_inverted_index"))
    if not index:
        return ""
    positions: list[tuple[int, str]] = [(p, word) for word, ps in index.items() for p in ps]
    positions.sort()
    return " ".join(word for _, word in positions)


def text_field(d: dict[str, Any] | None, key: str) -> str:
    """A string field of an OpenAlex object, or "" when the object or the field is missing."""
    value = d.get(key) if d else None
    return str(value) if value else ""


def source_of(work: dict[str, Any]) -> tuple[str, str]:
    """The display name of the primary source and its kind: journal, conference, arXiv, other repository, book, other."""
    location = cast(dict[str, Any] | None, work.get("primary_location"))
    source = cast(dict[str, Any] | None, location.get("source")) if location else None
    name = text_field(source, "display_name")
    kind = text_field(source, "type")
    if name.startswith("arXiv"):
        return name, "arXiv"
    if kind == "repository":
        return name, "other repository"
    if kind == "journal":
        return name, "journal"
    if kind == "conference" or (work.get("type") == "conference-paper" and name):
        return name, "conference"
    if kind in ("book series", "ebook platform") or work.get("type") in ("book-chapter", "book"):
        return name, "book"
    return name, "other"


def load_raw() -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    """Every search's records by database, with repeats across windows removed, and each database's count. The keys
    are "openalex:<string>" and "arxiv:<string>"."""
    raw: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for key in OPENALEX_QUERIES:
        seen: set[str] = set()
        name = f"openalex:{key}"
        raw[name] = []
        counts[name] = {"count": 0, "fetched": 0}
        for lo, _ in WINDOWS:
            if not (CACHE / key / f"{lo}.done").exists():
                continue  # the federated-learning string's last window was never downloaded
            with (CACHE / key / f"{lo}.jsonl").open() as f:
                for line in f:
                    w = cast(dict[str, Any], json.loads(line))
                    if w["id"] not in seen:
                        seen.add(w["id"])
                        raw[name].append(w)
            marker = cast(dict[str, int], json.loads((CACHE / key / f"{lo}.done").read_text()))
            counts[name]["count"] += int(marker["count"])
        counts[name]["fetched"] = len(raw[name])
    for key in ARXIV_QUERIES:
        name = f"arxiv:{key}"
        with (CACHE / "arxiv" / f"{key}.jsonl").open() as f:
            raw[name] = [cast(dict[str, Any], json.loads(line)) for line in f]
        marker = cast(dict[str, int], json.loads((CACHE / "arxiv" / f"{key}.done").read_text()))
        counts[name] = {"count": int(marker["count"]), "fetched": len(raw[name])}
    return raw, counts


def screen_all(raw: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], Counter[str], int, int, int]:
    """Union the queries' records, drop repeats and screen; return the included rows and the stage counts."""
    returned = sum(len(v) for v in raw.values())
    by_id: dict[str, dict[str, Any]] = {}
    queries_of: dict[str, set[str]] = {}
    for key, works in raw.items():
        for w in works:
            by_id.setdefault(w["id"], w)
            queries_of.setdefault(w["id"], set()).add(key)
    # The same paper can appear as a preprint and as the published version: keep one per title, preferring the
    # published one and the larger citation count.
    rank = {"article": 0, "conference-paper": 0, "review": 0, "book-chapter": 1, "preprint": 2}

    def order(w: dict[str, Any]) -> tuple[int, int, int]:
        return (rank.get(w["type"], 3), 1 if w["id"].startswith("arxiv:") else 0, -int(w["cited_by_count"]))

    by_title: dict[str, dict[str, Any]] = {}
    for w in by_id.values():
        t = normalize_title(w.get("title") or "")
        if not t:
            t = w["id"]
        current = by_title.get(t)
        if current is None or order(w) < order(current):
            if current is not None:
                queries_of[w["id"]] |= queries_of[current["id"]]
            by_title[t] = w
        else:
            queries_of[current["id"]] |= queries_of[w["id"]]
    exclusions: Counter[str] = Counter()
    included: list[dict[str, Any]] = []
    for w in by_title.values():
        title = w.get("title") or ""
        abstract = abstract_of(w)
        verdict = screen(title, abstract, w["type"], w.get("language") or "", bool(w.get("is_retracted")))
        if verdict != "included":
            exclusions[verdict] += 1
            continue
        name, kind = source_of(w)
        authors = cast(list[dict[str, Any]], w.get("authorships") or [])
        first = text_field(cast(dict[str, Any] | None, authors[0].get("author")), "display_name") if authors else ""
        included.append(
            {
                "id": w["id"].rsplit("/", 1)[-1],
                "databases": ";".join(sorted({q.split(":")[0] for q in queries_of[w["id"]]})),
                "title": " ".join(title.split()),
                "date": w["publication_date"],
                "kind": w["type"],
                "source": name,
                "source_kind": kind,
                "cited_by": w["cited_by_count"],
                "queries": ";".join(sorted({q.split(":")[1] for q in queries_of[w["id"]]})),
                "themes": ";".join(themes_of(title, abstract)),
                "domains": ";".join(domains_of(title, abstract)),
                "cues": ";".join(privacy_cues(f"{title} {abstract}".lower())),
                "first_author": first,
                "authors": len(authors),
                "has_abstract": "1" if abstract else "0",
                "topic": text_field(cast(dict[str, Any] | None, w.get("primary_topic")), "display_name"),
            }
        )
    included.sort(key=lambda r: (r["date"], r["id"]))
    return included, exclusions, returned, len(by_id), len(by_title)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true", help="download again instead of reading the cache")
    args = parser.parse_args()
    for key in OPENALEX_QUERIES:  # downloaded only on --fetch: the cached windows are read as they are
        if args.fetch:
            for lo, _ in WINDOWS:
                (CACHE / key / f"{lo}.done").unlink(missing_ok=True)
            fetch(key, QUERIES[key])
    for key, query in ARXIV_QUERIES.items():
        if args.fetch:
            (CACHE / "arxiv" / f"{key}.done").unlink(missing_ok=True)
        fetch_arxiv(key, query)
    raw, counts = load_raw()
    included, exclusions, returned, distinct_ids, distinct_titles = screen_all(raw)
    fields = list(included[0].keys())
    with (HERE / "ppml_2026_review_records.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(included)
    summary = {
        "search_date": SEARCH_DATE,
        "databases": ["OpenAlex", "arXiv"],
        "queries": {
            k: {
                "string": q,
                "arxiv_string": ARXIV_QUERIES[k],
                "openalex": counts.get(f"openalex:{k}", {"count": 0, "fetched": 0}),
                "arxiv": counts[f"arxiv:{k}"],
            }
            for k, q in QUERIES.items()
        },
        "returned": returned,
        "distinct_ids": distinct_ids,
        "distinct_titles": distinct_titles,
        "exclusions": dict(exclusions.most_common()),
        "included": len(included),
    }
    (HERE / "ppml_2026_review_search.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
