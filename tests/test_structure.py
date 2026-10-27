"""tools/structure.py keeps every page in the shape AGENTS.md gives its kind, and lets the right shape through."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import structure

CONFIG = structure.load_config()
CHAPTER = CONFIG.skeletons["full-chapter"]

GOOD_CHAPTER = """# The Laplace mechanism

The question this chapter answers, the one idea, and the running example: a count of 1,234 patients.

## Sensitivity {#sec-lg-sensitivity}

The noise must hide what one record can change, so the first step is to measure that change.

::: {#def-lg-sensitivity name="Sensitivity"}
The sensitivity is the largest change over neighbours, $\\Delta_1 f = \\max_{D \\sim D'} \\|f(D) - f(D')\\|_1$.
:::

::: {#thm-lg-laplace name="Laplace mechanism"}
Adding $\\mathrm{Lap}(\\Delta_1 f / \\varepsilon)$ noise is $\\varepsilon$-DP.
:::

::: {.proof}
1. The ratio of densities is at most $e^{\\varepsilon}$ by the triangle inequality.
:::

::: {#exm-lg-count}
For the count, $\\Delta_1 f = 1$, so the scale is $1/\\varepsilon$.
:::

## Code {#sec-lg-code}

The script checks every number above. Run `uv run python code/part03-dp-foundations/laplace_gaussian.py`.

| Quantity | Value |
| --- | --- |
| scale | 1.0000 |

: What the script prints. {#tbl-lg-code}

## Summary {#sec-lg-summary}

| Result | Statement | Use |
| --- | --- | --- |
| @thm-lg-laplace | scale $\\Delta_1 f/\\varepsilon$ | every count |

: The results of this chapter. {#tbl-lg-summary}

## References {#sec-lg-references}

- Dwork et al., *Calibrating noise to sensitivity*, 2006.
"""


def errors(
    tmp_path: Path, text: str, skeleton: structure.Skeleton, rel: str = "part03-dp-foundations/03-laplace-gaussian.md"
) -> list[str]:
    page = tmp_path / "page.md"
    page.write_text(text)
    return [f.message for f in structure.check_page(page, rel, skeleton) if not f.warning]


def test_a_chapter_in_the_right_shape_passes(tmp_path: Path) -> None:
    assert errors(tmp_path, GOOD_CHAPTER, CHAPTER) == []


def test_the_old_scaffolding_is_forbidden(tmp_path: Path) -> None:
    text = GOOD_CHAPTER.replace(
        "## Sensitivity {#sec-lg-sensitivity}",
        "## Toolkit\n\nFacts.\n\n## Sensitivity {#sec-lg-sensitivity}\n\n### Statement",
    )
    found = errors(tmp_path, text, CHAPTER)
    assert any('"## Toolkit"' in m for m in found)
    assert any('"### Statement"' in m for m in found)


def test_labels_use_the_chapter_prefix(tmp_path: Path) -> None:
    found = errors(tmp_path, GOOD_CHAPTER.replace("{#tbl-lg-code}", "{#tbl-foo-code}"), CHAPTER)
    assert any("tbl-foo-code" in m and "prefix" in m for m in found)


def test_a_theorem_needs_a_proof_or_a_citation_marker(tmp_path: Path) -> None:
    without = GOOD_CHAPTER.replace(
        "::: {.proof}\n1. The ratio of densities is at most $e^{\\varepsilon}$ by the triangle inequality.\n:::\n", ""
    )
    assert any("thm-lg-laplace has no proof" in m for m in errors(tmp_path, without, CHAPTER))
    cited = without.replace(
        "is $\\varepsilon$-DP.\n:::",
        "is $\\varepsilon$-DP.\n:::\n\n<!-- proof: cited -->\nThe proof is in Dwork and Roth (2014), Theorem 3.6.",
    )
    assert not any("has no proof" in m for m in errors(tmp_path, cited, CHAPTER))


def test_the_summary_holds_one_captioned_table_and_every_table_is_captioned(tmp_path: Path) -> None:
    found = errors(tmp_path, GOOD_CHAPTER.replace("\n: What the script prints. {#tbl-lg-code}\n", "\n"), CHAPTER)
    assert any("caption" in m for m in found)
    found = errors(tmp_path, GOOD_CHAPTER.replace("{#tbl-lg-summary}", "{#tbl-lg-results}"), CHAPTER)
    assert any("Summary" in m for m in found)


def test_a_table_float_div_captions_the_tables_inside_it(tmp_path: Path) -> None:
    table = "| Quantity | Value |\n| --- | --- |\n| scale | 1.0000 |\n"
    plain = table + "\n: What the script prints. {#tbl-lg-code}\n"
    floated = (
        '::: {#tbl-lg-code}\n\n::: {.content-visible when-format="html"}\n\n' + table + "\n:::\n\n"
        "What the script prints.\n\n:::\n"
    )
    assert plain in GOOD_CHAPTER
    assert not any("caption" in m for m in errors(tmp_path, GOOD_CHAPTER.replace(plain, floated), CHAPTER))
    unlabelled = floated.replace("{#tbl-lg-code}", "{.column-page}")
    assert any("caption" in m for m in errors(tmp_path, GOOD_CHAPTER.replace(plain, unlabelled), CHAPTER))


def test_the_budget_and_the_tail_order_are_checked(tmp_path: Path) -> None:
    long = GOOD_CHAPTER.replace("The question this chapter answers", "word " * 8001)
    assert any("over the budget" in m for m in errors(tmp_path, long, CHAPTER))
    swapped = GOOD_CHAPTER.replace("## Summary {#sec-lg-summary}", "## Zummary {#sec-lg-summary}")
    assert any("ends with the sections" in m for m in errors(tmp_path, swapped, CHAPTER))


def test_display_math_starting_with_a_bar_is_not_a_table(tmp_path: Path) -> None:
    text = GOOD_CHAPTER.replace(
        "For the count,", "The error obeys\n$$\n|M(D) - f(D)| \\le b \\ln(1/\\beta).\n$$\nFor the count,"
    )
    assert errors(tmp_path, text, CHAPTER) == []


def test_every_page_of_every_book_has_a_skeleton() -> None:
    for book in CONFIG.books:
        for rel, skeleton in structure.pages_of(book, CONFIG):
            assert skeleton is not None, f"{book}/{rel} matches no skeleton in tools/structure.toml"


def test_legacy_pages_exist() -> None:
    for book, rules in CONFIG.books.items():
        for rel in rules.legacy:
            assert (ROOT / book / rel).exists(), (
                f"legacy page {book}/{rel} no longer exists; drop it from structure.toml"
            )
