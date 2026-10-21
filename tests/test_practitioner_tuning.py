"""code/part05-dp-ml/practitioner_tuning.py runs its default (seed 0) and prints what step 9 of 13-train-dp-sgd.md quotes.

The run takes about 35 s; the page quotes this run, so the test runs it in full.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml" / "practitioner_tuning.py"

EXPECTED = [
    "n=30,000  delta=1e-05  q=0.0333  steps=300",
    "learning rates: 0.05, 0.1, 0.2, 0.5, 1, 2",
    # The cost of each strategy
    "one run, no tuning: sigma=1.029",
    "grid of 6, all runs counted: sigma=1.816",
    "random search, Poisson mean 4 runs: sigma=1.256",
    "naive grid of 6 at sigma=1.029: reports epsilon=4.00",
    "naive grid, true epsilon of all 6 runs = 9.84",
    # The random search
    "runs drawn: 2",
    "run 1: lr=0.05  validation accuracy 0.822",
    "run 2: lr=0.05  validation accuracy 0.817",
    "chosen lr=0.05: test accuracy 0.828",
    "total epsilon of the search (RDP) = 4.00",
    # The non-private sweep
    "non-private lr=0.05: validation accuracy 0.839",
    "non-private lr=0.1: validation accuracy 0.837",
    "non-private lr=0.2: validation accuracy 0.838",
    "non-private lr=0.5: validation accuracy 0.832",
    "non-private lr=1: validation accuracy 0.826",
    "non-private lr=2: validation accuracy 0.833",
    "best non-private lr on the private rows (not used): 0.05",
]


@pytest.mark.libs
def test_practitioner_tuning_prints_what_the_page_quotes() -> None:
    out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    for expected in EXPECTED:
        assert expected in lines, f"missing line: {expected!r}"
