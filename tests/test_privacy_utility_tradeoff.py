"""code/notes/privacy_utility_tradeoff.py: its privacy translations and frontier, and a short run of the experiment."""

import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "notes" / "privacy_utility_tradeoff.py"

pytestmark = pytest.mark.libs  # the script imports torch, Opacus and diffprivlib


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("privacy_utility_tradeoff", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_belief_ceiling_and_its_inverse() -> None:
    m = load()
    assert m.belief_ceiling(1.0) == pytest.approx(math.e / (1 + math.e))
    assert m.belief_ceiling(0.0) == pytest.approx(0.5)
    assert m.epsilon_for_belief(0.75) == pytest.approx(math.log(3))  # the note's policy: a 75% ceiling
    for eps in (0.1, 1.0, 3.0):
        assert m.epsilon_for_belief(m.belief_ceiling(eps)) == pytest.approx(eps)


def test_membership_bound() -> None:
    m = load()
    assert m.tpr_bound(1.0, 0.01) == pytest.approx(math.e * 0.01 + 1e-5)
    assert m.tpr_bound(10.0, 0.01) == 1.0


def test_frontier_and_reach() -> None:
    m = load()
    # (privacy loss, utility): the second point is dominated by the first, the fourth by the third.
    assert m.non_dominated([(0.1, 5.0), (0.3, 4.0), (1.0, 9.0), (3.0, 9.0)]) == [0, 2]
    assert m.reached(np.array([0.9, 0.1, 0.8, 0.2]), np.array([1, 1, 0, 0]), 2) == 1


def test_features_lie_within_the_public_norm_bound() -> None:
    m = load()
    sys.path.insert(0, str(SCRIPT.parents[1]))
    import practitioner_datasets

    X: np.ndarray = m.features(practitioner_datasets.adult())
    assert X.min() >= 0.0 and X.max() <= 1.0
    norms: np.ndarray = np.sqrt((X**2).sum(axis=1))
    assert norms.max() <= m.DATA_NORM


def test_short_run_writes_the_note_inputs(tmp_path: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--seeds", "1", "--size-seeds", "1", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # The campaign facts the note quotes as text.
    assert "rows: train 32,561  campaign 16,281  features 47" in out
    assert "offers = eligible customers in the campaign: 3,846  random offers reach 908.5" in out
    assert "policy: belief ceiling 75% from even odds -> epsilon <= 1.0986" in out
    for name in ("technical.svg", "business.svg", "sizes.svg", "_frontier.md", "_menu.md", "_sizes.md", "_decision.md"):
        assert (tmp_path / name).stat().st_size > 0
