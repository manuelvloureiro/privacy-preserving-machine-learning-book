"""code/notes/ppml_2026_review.py: its screening and theme rules, its interval, and a run that reproduces the note's numbers."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "notes" / "ppml_2026_review.py"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ppml_2026_review", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_screening_rules() -> None:
    m = load()
    # A paper on DP-SGD is included; the same abstract as a dataset or in another language is not.
    assert m.screen("Private fine-tuning", "We train with DP-SGD.", "preprint", "en", False) == "included"
    assert m.screen("Private fine-tuning", "We train with DP-SGD.", "dataset", "en", False) == "not a paper"
    assert m.screen("Private fine-tuning", "We train with DP-SGD.", "article", "zh", False) == "not in English"
    assert m.screen("Private fine-tuning", "We train with DP-SGD.", "article", "en", True) == "retracted"
    # Privacy as a motivation, with no technique named, is excluded; a technique with no learning object too.
    assert (
        m.screen("A smart grid", "Users worry about privacy, so we train a model.", "article", "en", False)
        == "no privacy technique or setting"
    )
    assert (
        m.screen("Private counts", "A differentially private histogram of ages.", "article", "en", False)
        == "no machine learning"
    )
    # Federated learning that calls itself privacy-preserving names no technique: included only with privacy in the title.
    assert (
        m.screen(
            "FL for anomaly detection", "A privacy-preserving federated learning framework.", "article", "en", False
        )
        == "privacy only as a setting"
    )
    assert (
        m.screen(
            "Privacy-preserving federated anomaly detection", "A federated learning framework.", "article", "en", False
        )
        == "included"
    )
    assert (
        m.screen("FL for anomaly detection", "Federated learning with secure aggregation.", "article", "en", False)
        == "included"
    )


def test_theme_and_domain_rules() -> None:
    m = load()
    themes = m.themes_of(
        "Membership inference against fine-tuned language models", "We attack LLMs fine-tuned with DP-SGD."
    )
    assert "attacks" in themes and "training" in themes and "survey" not in themes
    assert set(m.themes_of("A survey of federated unlearning", "...")) == {"federated", "unlearning", "survey"}
    assert m.themes_of("Private data release", "We publish survey data with Laplace noise.") == [
        "synthetic"
    ]  # "survey" counts by title only
    assert m.themes_of("Something else entirely", "with privacy-preserving training") == ["other"]
    assert m.domains_of("Private GNNs for patient records", "graph neural networks on electronic health records") == [
        "graph",
        "health",
    ]
    assert m.themes_of("Privately Learning Decision Lists", "a differentially private Winnow") == ["training"]
    assert m.normalize_title("DP-SGD: A Study!") == m.normalize_title("dp sgd a study")


def test_clopper_pearson() -> None:
    m = load()
    lo, hi = m.clopper_pearson(95, 100)
    assert lo == pytest.approx(0.8872, abs=1e-3) and hi == pytest.approx(0.9836, abs=1e-3)
    assert m.clopper_pearson(0, 10)[0] == 0.0 and m.clopper_pearson(10, 10)[1] == 1.0
    assert m.clopper_pearson(0, 0) == (0.0, 1.0)
    assert m.median([3, -1, 1, 2]) == 2.0  # -1 means unknown


def test_frozen_table_matches_the_rules() -> None:
    """Every row of the frozen table carries the themes and domains the rules give its title (the abstract is not stored,
    so a theme the abstract added may be present too, never one the title gives that the row lacks)."""
    m = load()
    records = m.read_records()
    assert len(records) > 1000
    assert len({r.id for r in records}) == len(records)
    assert all(r.date.startswith("2026") for r in records)
    assert all(r.kind in m.PAPER_TYPES for r in records)
    assert all(set(r.databases) <= {"openalex", "arxiv"} for r in records)
    assert all(r.cited_by >= 0 for r in records if "openalex" in r.databases)
    for r in records:
        for theme in m.themes_of(r.title, ""):
            assert theme == "other" or theme in r.themes, (r.id, theme)


def test_run_reproduces_the_note(tmp_path: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "included records:" in out
    for name in (
        "themes.svg",
        "cooccurrence.svg",
        "months.svg",
        "_flow.md",
        "_focus.md",
        "_reading.md",
        "_core_venues.md",
        "_queries.md",
        "_screening.md",
        "_themes.md",
        "_sources.md",
        "_venues.md",
        "_domains.md",
    ):
        assert (tmp_path / name).exists(), name
