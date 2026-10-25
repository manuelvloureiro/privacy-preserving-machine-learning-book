"""code/notes/python_privacy_libraries.py: its public bounds, scales and accountants, and a short run of the three jobs."""

import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "notes" / "python_privacy_libraries.py"

pytestmark = pytest.mark.libs  # the script imports OpenDP, diffprivlib, Opacus and dp-accounting

BOUNDS = (6.981, 28.11)  # the published Min and Max of the mean radius column
N = 569


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("python_privacy_libraries", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_bounds_come_from_the_description() -> None:
    m = load()
    data = m.load()
    assert data.bounds == BOUNDS
    assert len(data.column) == N and len(data.y_train) == 455 and len(data.y_test) == 114
    assert data.x_train.shape[1] == 30 and data.x_train.min() >= 0.0 and data.x_train.max() <= 1.0
    assert data.outside == 14  # rounded summary values; clipping moves these onto the bounds
    from sklearn.datasets import load_breast_cancer

    bounds = m.public_bounds(load_breast_cancer().DESCR)
    assert len(bounds) == 30
    assert bounds["mean radius"] == BOUNDS and "radius error" in bounds and "worst radius" in bounds


def test_textbook_scales_and_widths() -> None:
    m = load()
    count_scale, mean_scale = m.laplace_scales(N, BOUNDS, 0.5)
    assert count_scale == 2.0
    assert mean_scale == pytest.approx((28.11 - 6.981) / (N * 0.5))
    assert m.laplace_half_width(2.0) == pytest.approx(2 * math.log(20))
    assert m.delta_for(455) == 1e-3
    assert m.sampling_rate(455, 32) == pytest.approx(1 / 15)


def test_the_three_implementations_use_the_same_scales() -> None:
    m = load()
    data = m.load()
    textbook = m.laplace_scales(N, BOUNDS, 0.5)
    by_hand = m.by_hand(data.column, data.bounds, 1.0, np.random.default_rng(0))
    opendp = m.with_opendp(data.column, data.bounds, 1.0)
    diffprivlib = m.with_diffprivlib(data.column, data.bounds, 1.0, 0)
    for release in (by_hand, opendp, diffprivlib):
        assert release.count_scale == pytest.approx(textbook[0], rel=1e-6)
        assert release.mean_scale == pytest.approx(textbook[1], rel=1e-6)
        assert abs(release.mean - data.column.mean()) < 1.0 and abs(release.count - N) < 30
    assert opendp.mean_scale >= textbook[1]  # OpenDP rounds its scale up for floating-point arithmetic


def test_opendp_public_length_trap() -> None:
    m = load()
    data = m.load()
    # One contribution with the length public: no neighbouring dataset exists, so the noise all but vanishes.
    assert m.opendp_public_length(data.column, data.bounds, 1.0, 1) < 1e-6
    # Two contributions (one replaced record): the sum's sensitivity is the width of the bounds.
    assert m.opendp_public_length(data.column, data.bounds, 1.0, 2) == pytest.approx(28.11 - 6.981, rel=1e-6)


def test_budget_refusals() -> None:
    m = load()
    data = m.load()
    assert m.opendp_refusal(data.column, data.bounds, 1.0).startswith("ValueError")
    assert m.diffprivlib_refusal(data.column, data.bounds, 1.0).startswith("BudgetError")
    assert m.diffprivlib_logistic_regression().startswith("TypeError")


def test_accountants_are_ordered() -> None:
    m = load()
    eps = m.accountants(1.0, 1 / 15, 300, 1e-3)
    assert eps["PLD lower estimate"] <= eps["dp-accounting PLD"] <= eps["Opacus RDP"]
    assert eps["dp-accounting PLD"] <= eps["dp-accounting RDP"]
    assert abs(eps["Opacus PRV"] - eps["dp-accounting PLD"]) / eps["dp-accounting PLD"] < 0.01
    assert eps["Opacus GDP"] < eps["PLD lower estimate"]  # the GDP approximation claims more than the run has
    assert eps["dp-accounting PLD"] == pytest.approx(5.5604, abs=1e-3)
    assert eps["Opacus RDP"] == pytest.approx(6.4300, abs=1e-3)


def test_short_run_writes_the_note_inputs(tmp_path: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--seeds", "1", "--draws", "20", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for expected in (
        "versions: opendp 0.16.0  diffprivlib 0.6.6  opacus 1.6.0  dp-accounting 0.6.0  scikit-learn 1.8.0",
        "records 569  train 455  test 114  features 30",
        "majority class share of the test set: 0.632",
        "mean radius: public bounds [6.981, 28.11]  true mean 14.1273",
        "raw values outside the published bounds, over all features: 14",
        "textbook scales: count 2.0000  mean 0.074267  mean 95% half-width 0.2225",
        "OpenDP 95% half-width of the count: 5.9915",
        "OpenDP: ValueError: Privacy allowance has been exhausted",
        "diffprivlib: BudgetError: Privacy spend of (0.5,0) not permissible",
        "delta 0.001 (1/n^1.1 = 1.19e-03)  sampling rate 0.0667  steps 300",
        "epsilon 1: noise multiplier 3.145",
        "diffprivlib LogisticRegression with scikit-learn 1.8.0: TypeError",
        "Opacus RDP           6.4300",
        "Opacus PRV           5.5710",
        "Opacus GDP           5.2834",
        "dp-accounting RDP    6.4556",
        "dp-accounting PLD    5.5604",
        "PLD lower estimate   5.5454",
    ):
        assert expected in out, expected
    for name in ("training.svg", "accounting.svg", "_release.md", "_training.md", "_accounting.md"):
        assert (tmp_path / name).stat().st_size > 0
