"""The course and the full edition write the constants they share in the same form (PLAN step 8).

The course links to no book, so nothing ties its formulas to the full edition's but this test: each
constant below must appear, written exactly so, on each page named for it, and no page of either
book may write a variant of it.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ("book-course", "book-full")

# name: (the formula as both books write it, the pages that state it, a variant no page may write)
CONSTANTS = {
    "classic Gaussian scale": (
        r"\sqrt{2\ln(1.25/\delta)}",
        ["book-course/part02-dp-basics/03-approximate-dp.md", "book-full/part03-dp-foundations/03-laplace-gaussian.md"],
        r"\\sqrt\{\\ln\(1\.25/\\delta\)\}|\\ln\(1\.25\\delta\)|1\.25\\delta",  # a dropped 2 or a lost fraction
    ),
    "advanced composition, tanh form": (
        r"\varepsilon\sqrt{2k\ln(1/\delta')} + k\varepsilon\tanh(\varepsilon/2)",
        [
            "book-course/part02-dp-basics/04-advanced-composition.md",
            "book-full/part03-dp-foundations/04-composition.md",
        ],
        r"k\\varepsilon\\tanh\\varepsilon",
    ),
    "advanced composition, Dwork-Rothblum-Vadhan form": (
        r"k\varepsilon(e^\varepsilon - 1)",
        [
            "book-course/part02-dp-basics/04-advanced-composition.md",
            "book-full/part03-dp-foundations/04-composition.md",
        ],
        r"k\\varepsilon\(e\^\{?\\varepsilon\}?\+1\)",
    ),
}


@pytest.mark.parametrize("name", CONSTANTS)
def test_each_book_states_the_constant_in_the_same_form(name: str) -> None:
    formula, pages, _ = CONSTANTS[name]
    missing = [p for p in pages if formula not in (ROOT / p).read_text()]
    assert not missing, f"{name}: {formula} is not written so on {missing}"


@pytest.mark.parametrize("name", CONSTANTS)
def test_no_page_writes_a_variant(name: str) -> None:
    _, _, variant = CONSTANTS[name]
    found = [
        f"{page.relative_to(ROOT)}:{n}"
        for book in BOOKS
        for page in sorted((ROOT / book).rglob("*.md"))
        for n, line in enumerate(page.read_text().splitlines(), 1)
        if re.search(variant, line)
    ]
    assert not found, f"{name}: a variant on {found}"


def homes() -> list[tuple[str, str]]:
    """The (label, page) rows of AGENTS.md's table of one home per result."""
    text = (ROOT / "AGENTS.md").read_text().split("### One home per result", 1)[1]
    return re.findall(r"^\| [^|]+ \| `([\w-]+)` \| `([^`]+)` \|$", text, flags=re.MULTILINE)


def test_agents_lists_the_homes() -> None:
    assert len(homes()) >= 20


@pytest.mark.parametrize(("label", "page"), homes())
def test_each_result_is_defined_once_on_its_home(label: str, page: str) -> None:
    book = page.split("/")[0]
    defined = [
        str(p.relative_to(ROOT)) for p in (ROOT / book).rglob("*.md") if re.search(rf"\{{#{label}[ }}]", p.read_text())
    ]
    assert defined == [page], f"{label} is defined on {defined}, not only on {page}"
