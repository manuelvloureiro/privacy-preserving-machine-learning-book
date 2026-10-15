"""Collect the numbered labels of a rendered book, so another book can cross-reference them.

    uv run python tools/xrefs.py --site _book/book-full --book book-full \
        --skip-defined-in _build/exercises-website --out _build/exercises-website/_xrefs.json

Quarto resolves @eq-name, @sec-name, @fig-name and @tbl-name only within one book, so the
exercise book cannot reference the full edition's equations directly. This reads the full
edition's HTML and writes, for each label, the text Quarto gives a reference to it ("Equation 20.3")
and its page and anchor. filters/xrefs.lua turns the exercise book's references into links with it.
Labels the referring book defines itself (--skip-defined-in) are left out, so Quarto resolves them.
"""

import argparse
import json
import re
from pathlib import Path

NBSP = " "
EQUATION = re.compile(r'<span id="(eq-[\w-]+)">\s*<span class="math display">.*?\\tag\{([\d.]+)\}', re.DOTALL)
SECTION = re.compile(r'<section id="(sec-[\w-]+)"[^>]*\bdata-number="([\d.]+)"')
FLOAT = re.compile(r'<div id="((?:fig|tbl)-[\w-]+)"[^>]*class="quarto-float')
FLOAT_NUMBER = re.compile(r"(Figure|Table)&nbsp;([\d.]+)")
DEFINED = re.compile(r"\{#((?:eq|sec|fig|tbl)-[\w-]+)")


def labels_in_page(html: str) -> dict[str, str]:
    """Each numbered label on one rendered page, with the text a reference to it shows."""
    found = {m.group(1): f"Equation{NBSP}{m.group(2)}" for m in EQUATION.finditer(html)}
    found |= {m.group(1): f"Section{NBSP}{m.group(2)}" for m in SECTION.finditer(html)}
    for m in FLOAT.finditer(html):
        number = FLOAT_NUMBER.search(html, m.end())
        if number:
            found[m.group(1)] = f"{number.group(1)}{NBSP}{number.group(2)}"
    return found


def collect(site: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for page in sorted(site.rglob("*.html")):
        href = page.relative_to(site).as_posix()
        for label, text in labels_in_page(page.read_text()).items():
            labels[label] = {"text": text, "href": f"{href}#{label}"}
    return labels


def defined_in(book: Path) -> set[str]:
    return {m.group(1) for f in book.rglob("*.md") for m in DEFINED.finditer(f.read_text())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", type=Path, required=True, help="the rendered book to read labels from")
    parser.add_argument("--book", required=True, help="the referenced book's folder name, e.g. book-full")
    parser.add_argument("--skip-defined-in", type=Path, help="the referring book: labels it defines are skipped")
    parser.add_argument("--out", type=Path, required=True, help="where to write the JSON")
    args = parser.parse_args()
    site: Path = args.site
    skip: Path | None = args.skip_defined_in
    labels = collect(site)
    if not labels:
        raise SystemExit(f"no labels found in {site}: render it first")
    for label in defined_in(skip) if skip else set[str]():
        labels.pop(label, None)
    out: Path = args.out
    out.write_text(json.dumps({"book": args.book, "labels": labels}, ensure_ascii=False, indent=1, sort_keys=True))
    print(f"wrote {len(labels)} labels to {out}")


if __name__ == "__main__":
    main()
