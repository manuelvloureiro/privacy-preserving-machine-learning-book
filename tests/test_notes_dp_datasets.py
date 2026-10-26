"""code/notes/dp_datasets.py: its noise-floor arithmetic, its marginal counts, and a run that reproduces the note's numbers."""

import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "notes" / "dp_datasets.py"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dp_datasets", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_noise_floor() -> None:
    m = load()
    assert m.count_noise_sd(1.0) == pytest.approx(math.sqrt(2))
    assert m.relative_error(10.0, 1.0) == pytest.approx(0.1414, abs=1e-4)  # German Credit's 1%: 10 applicants
    # The size at which a 1% count has 10% relative error at ε = 1, quoted by the note.
    assert m.records_for_relative_error(0.01, 1.0, 0.1) == pytest.approx(1414.2, abs=0.1)
    n = m.records_for_relative_error(0.01, 0.1, 0.1)
    assert m.relative_error(0.01 * n, 0.1) == pytest.approx(0.1)


def test_marginal_counts() -> None:
    m = load()
    sizes = {"a": 2, "b": 3, "c": 4}
    assert m.joint_cells(sizes) == 24
    assert m.marginal_cells(sizes, 1) == (3, 9)
    assert m.marginal_cells(sizes, 2) == (3, 2 * 3 + 2 * 4 + 3 * 4)
    assert m.marginal_cells(sizes, 3) == (1, 24)


def test_histogram_error_splits_into_bias_and_noise() -> None:
    m = load()
    true = np.array([100.0, 200.0])
    bounded = np.array([90.0, 180.0])
    # Squared bias averages (100 + 400) / 2 = 250; the noise variance at k = 5, ε = 1 is 2 · 25 = 50.
    assert m.histogram_rmse(true, bounded, 5, 1.0) == pytest.approx(math.sqrt(300.0))
    assert m.best_bound([1, 2, 3], [5.0, 1.0, 2.0]) == 2


@pytest.mark.libs  # reads the datasets cached in data/ by `make datasets`
def test_run_reproduces_the_note(tmp_path: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for expected in (
        "n for a 1% count with 10% relative error: eps=0.1 14,142  eps=1 1,414",
        "online retail: lines 541,909  with a customer id 406,829  customers 4,372",
        "lines per customer: median=42.0  p90=212.0  p99=671.9  mean=93.1  max=7,983.0",
        "months 13  lines per month mean 31,295  min 17,661  max 65,598",
        "adult: records 48,842  attributes 15  joint cells 76,204,800,000,000",
        "2-way marginals 105  cells 11,926  records per cell 430.0  noise sd at eps=1 split evenly 148.5",
        "3-way marginals 455  cells 515,212  records per cell 43.1  noise sd at eps=1 split evenly 643.5",
    ):
        assert expected in out, expected
    for name in ("noise_floor.svg", "contributions.svg", "_sizes.md", "_contributions.md", "_adult_domain.md"):
        assert (tmp_path / name).stat().st_size > 0
