"""tools/notation.py finds the notation NOTATION.md forbids, and leaves alone what it allows."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import notation

RULES, CONFLICTS = notation.load_rules()


def findings(tmp_path: Path, text: str) -> list[str]:
    page = tmp_path / "page.md"
    page.write_text(text)
    return [f.message for f in notation.check_page(page, RULES, CONFLICTS)]


def test_math_and_prose_are_split_apart() -> None:
    split = notation.split_page("A $\\epsilon$ and `\\epsilon` and\n\n```python\nx = 1\n```\n$$\nD_\\lambda\n$$\n")
    assert [m.strip() for _, m in split.math] == ["\\epsilon", "D_\\lambda"]
    assert "epsilon" not in split.prose


def test_forbidden_math_is_found_and_code_is_not(tmp_path: Path) -> None:
    assert findings(tmp_path, "The noise is $\\epsilon$.\n")
    assert not findings(tmp_path, "The noise is `$\\epsilon$`.\n")
    assert findings(tmp_path, "Rényi: $D_\\lambda(P\\|Q)$.\n")
    assert not findings(tmp_path, "Rényi: $D_\\alpha(P\\|Q)$, and $D_{\\mathrm{KL}}(P\\|Q)$.\n")


def test_unicode_epsilon_is_allowed_only_where_markdown_makes_plain_text(tmp_path: Path) -> None:
    assert not findings(tmp_path, "# What ε means\n\nSee [the ε page](a.md).\n\n: δ for each ε {#tbl-x}\n")
    assert findings(tmp_path, "The run spends ε = 1.\n")


def test_a_bare_log_is_allowed_only_inside_big_o(tmp_path: Path) -> None:
    assert not findings(tmp_path, "The error is $O(\\sqrt{n\\log(1/\\delta)})$ and $\\ln 2$.\n")
    assert findings(tmp_path, "The entropy is $\\log 2$.\n")


def test_a_phrase_that_names_a_symbol_is_checked_across_math_and_prose(tmp_path: Path) -> None:
    assert findings(tmp_path, "With sampling rate $\\gamma$ the cost falls.\n")
    assert not findings(tmp_path, "With sampling rate $q$ the cost falls.\n")


def test_max_per_page_allows_one_also_called_mention(tmp_path: Path) -> None:
    assert not findings(tmp_path, "The learning rate, also called the step size, is fixed.\n")
    assert findings(tmp_path, "The step size is fixed. A larger step size diverges.\n")


def test_spelling_exceptions_cover_cited_titles(tmp_path: Path) -> None:
    assert not findings(tmp_path, "Dwork et al., *Analyze Gauss*.\n")
    assert findings(tmp_path, "We analyze the mechanism.\n")


def test_two_meanings_of_one_symbol_on_a_page_are_a_conflict(tmp_path: Path) -> None:
    text = "Add $\\mathrm{Lap}(b)$ noise.\n\nThe hidden bit $b$ picks the world.\n"
    assert any("two meanings" in m for m in findings(tmp_path, text))
    assert not any("two meanings" in m for m in findings(tmp_path, "Add $\\mathrm{Lap}(b)$ noise.\n"))


def test_a_and_b_as_the_errors_of_a_test_are_found(tmp_path: Path) -> None:
    assert findings(tmp_path, "A test has type I error $a$ and type II error $b$.\n")
    assert findings(tmp_path, "Its errors are $a_\\phi = \\mathbb E_P[\\phi]$ and the curve $T(P, Q)(a)$.\n")
    assert findings(tmp_path, "At a false positive rate $a$ the curve is $G_\\mu(a)$.\n")
    assert not findings(tmp_path, "A test has type I error $\\alpha$, and $\\beta_\\phi = 1 - \\mathbb E_Q[\\phi]$.\n")
    assert not findings(tmp_path, "Two coins give $\\mathrm{kl}(a\\|b)$, with $a = \\Pr_P[A]$ and $b = \\Pr_Q[A]$.\n")


def test_the_errors_of_a_test_give_way_to_another_meaning_of_alpha_or_beta(tmp_path: Path) -> None:
    renyi = "The Rényi order is $\\alpha$, so $D_\\alpha(P\\|Q) \\le \\varepsilon(\\alpha)$.\n\n"
    assert any("two meanings" in m for m in findings(tmp_path, renyi + "A test has type I error $\\alpha$.\n"))
    assert any("two meanings" in m for m in findings(tmp_path, renyi + "The curve is $G_\\mu(\\alpha)$.\n"))
    assert not any("two meanings" in m for m in findings(tmp_path, renyi + "The curve is $G_\\mu(\\mathrm{FPR})$.\n"))
    failure = "With probability at least $1 - \\beta$ the error is small.\n\n"
    assert any("two meanings" in m for m in findings(tmp_path, failure + "Its type II error $\\beta$ is large.\n"))
    assert not any("two meanings" in m for m in findings(tmp_path, failure + "Its type II error $\\mathrm{FNR}$.\n"))
    assert not findings(tmp_path, "A test has type I error $\\alpha$ and type II error $\\beta$.\n")


def test_the_laplace_scale_keeps_b_and_is_never_renamed_s(tmp_path: Path) -> None:
    text = "Add $\\mathrm{Lap}(b)$ noise.\n\nThe hidden bit $b$ picks the world.\n"
    messages = [m for m in findings(tmp_path, text) if "two meanings" in m]
    assert messages and "name it s" not in messages[0]
