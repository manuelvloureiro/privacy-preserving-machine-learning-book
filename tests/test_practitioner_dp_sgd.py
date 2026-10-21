"""code/part05-dp-ml/practitioner_dp_sgd.py runs its default (seed 0) and prints what 13-train-dp-sgd.md quotes.

The run takes about 40 s; the page quotes this run, so the test runs it in full. It also checks the JSON record
that practitioner_release.py reads.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml" / "practitioner_dp_sgd.py"
RELEASE = SCRIPT.with_name("practitioner_release.py")

EXPECTED = [
    # Step 1
    "unit: one person (one row); n=30,000  validation=2,000  test=16,842",
    "delta = 10^floor(log10(n^-1.1)) = 1e-05",
    # Step 2
    "features=91  missing values kept as category '?'",
    "weights=6,018",
    # Step 3
    "learning-rate sweep on the public stand-in rows (fit 1,500, score 500):",
    "  lr=0.05: 0.814  lr=0.1: 0.816  lr=0.2: 0.826  lr=0.5: 0.792  lr=1: 0.812  lr=2: 0.768",
    "non-private learning rate 0.2; DP-SGD learning rate 2 (x10)",
    "non-private: test accuracy 0.847",
    "clip only (C=1, no noise): test accuracy 0.845",
    "always '<=50K': test accuracy 0.759",
    # Step 4
    "validator errors: 0",
    # Step 5
    "at initialisation: median norm 1.444  90th percentile 1.538",
    "fraction clipped at C=1: 1.000",
    "batch B=1,000 (physical 500)  q=1/ceil(n/B)=0.0333  epochs=10  steps=300",
    "noise std on the averaged gradient sigma*C/B at sigma=1: 0.0010",
    # Steps 6 and 7
    "sigma=0.973",
    "steps=300  q=0.0333",
    "epsilon (Opacus PRV) = 3.99",
    "test accuracy at epsilon=4: 0.846",
    # Step 8
    "epsilon PRV (Opacus) = 3.99",
    "epsilon PLD (dp_accounting) = 3.98",
    "epsilon RDP (dp_accounting) = 4.50",
    # Step 10
    "clip only: test accuracy 0.845",
    "epsilon=0.25 (sigma=8.125): test accuracy 0.793",
    "epsilon=1 (sigma=2.383): test accuracy 0.843",
    "epsilon=2 (sigma=1.431): test accuracy 0.845",
    "epsilon=4 (sigma=0.973): test accuracy 0.846",
    "epsilon=8 (sigma=0.722): test accuracy 0.846",
    "non-private Female: 0.924 [0.917, 0.931]  (5,186/5,613)",
    "non-private Male: 0.808 [0.800, 0.815]  (9,071/11,229)",
    "DP Female: 0.922 [0.915, 0.929]  (5,177/5,613)",
    "DP Male: 0.807 [0.800, 0.815]  (9,065/11,229)",
    "cost of privacy, Female: +0.002",
    "cost of privacy, Male: +0.001",
]


@pytest.mark.libs
def test_practitioner_dp_sgd_prints_what_the_page_quotes(tmp_path: Path) -> None:
    record = tmp_path / "run.json"
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--record", str(record)], capture_output=True, text=True, check=True
    ).stdout
    lines = out.splitlines()
    for expected in EXPECTED:
        assert expected in lines, f"missing line: {expected!r}"
    assert f"record written to {record}" in lines

    run = json.loads(record.read_text())
    assert run["n"] == 30_000 and run["delta"] == 1e-5 and run["steps"] == 300 and run["epochs"] == 10
    assert run["accountant"] == "prv" and run["sampler"] == "poisson" and run["tuning"]["counted_in_epsilon"] is True
    assert round(run["sigma"], 3) == 0.973 and round(run["sample_rate"], 4) == 0.0333
    assert run["max_grad_norm"] == 1.0
    assert round(run["epsilon_prv"], 2) == 3.99
    assert round(run["epsilon_pld"], 2) == 3.98
    assert round(run["epsilon_rdp"], 2) == 4.50
    assert run["library_versions"]["opacus"] == "1.6.0"
    assert run["library_versions"]["dp-accounting"] == "0.6.0"

    # practitioner_release.py's built-in record is a copy of this run's: they must not drift apart.
    spec = importlib.util.spec_from_file_location("practitioner_release", RELEASE)
    assert spec is not None and spec.loader is not None
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    assert run == release.ADULT_RUN_RECORD, "update ADULT_RUN_RECORD in practitioner_release.py from this run"
