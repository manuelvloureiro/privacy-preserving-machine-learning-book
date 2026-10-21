"""code/part04-privacy-accounting/practitioner_calibrate.py runs and prints every number that
book-practitioner/part04-privacy-accounting/10-calibrate-dp-sgd.md quotes.

The script's default run is the one the page quotes (about 45-60 s on a 12-core CPU). Accounting is deterministic, so
the lines are compared exactly.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part04-privacy-accounting" / "practitioner_calibrate.py"

EXPECTED = [
    "Adult rows: 48,842; Part 5's training split: n=30,000",
    "delta_for(30,000) = 1e-05",
    "delta_for(60,000) = 1e-06",
    "delta_for(569) = 1e-04",
    "Poisson: B=1000, E=10: q=0.033333, T=300",
    "Opacus DPDataLoader.from_data_loader: q=1/30=0.033333, 30 steps an epoch, 300 in 10 epochs",
    "chosen run uses the loader's q and steps; get_noise_multiplier at that q counts int(E/q)=300: equal True",
    "dataset             n  delta     B   E  eps      T  sigma PRV  sigma PLD  sigma/B x1000",
    "Adult           30000  1e-05   100  10    1   3000       0.98       0.99           9.81",
    "Adult           30000  1e-05   100  10    4   3000       0.62       0.62           6.16",
    "Adult           30000  1e-05   100  10    8   3000       0.51       0.51           5.09",
    "Adult           30000  1e-05   100  20    1   6000       1.21       1.20          12.06",
    "Adult           30000  1e-05   100  20    4   6000       0.66       0.67           6.64",
    "Adult           30000  1e-05   100  20    8   6000       0.55       0.55           5.47",
    "Adult           30000  1e-05   300  10    1   1000       1.43       1.42           4.75",
    "Adult           30000  1e-05   300  10    4   1000       0.74       0.74           2.45",
    "Adult           30000  1e-05   300  10    8   1000       0.59       0.60           1.96",
    "Adult           30000  1e-05   300  20    1   2000       1.87       1.85           6.22",
    "Adult           30000  1e-05   300  20    4   2000       0.82       0.82           2.75",
    "Adult           30000  1e-05   300  20    8   2000       0.64       0.64           2.15",
    "Adult           30000  1e-05  1000  10    1    300       2.38       2.37           2.38",
    "Adult           30000  1e-05  1000  10    4    300       0.97       0.98           0.97",
    "Adult           30000  1e-05  1000  10    8    300       0.72       0.73           0.72",
    "Adult           30000  1e-05  1000  20    1    600       3.24       3.19           3.24",
    "Adult           30000  1e-05  1000  20    4    600       1.16       1.16           1.16",
    "Adult           30000  1e-05  1000  20    8    600       0.82       0.83           0.82",
    "MNIST           60000  1e-06   256  20    1   4687       1.46       1.46           5.68",
    "MNIST           60000  1e-06   256  20    4   4687       0.73       0.73           2.84",
    "MNIST           60000  1e-06   256  20    8   4687       0.59       0.60           2.30",
    "breast cancer     569  1e-04    64  20    1    177       4.96       4.91          77.51",
    "breast cancer     569  1e-04    64  20    4    177       1.66       1.67          25.94",
    "breast cancer     569  1e-04    64  20    8    177       1.08       1.08          16.86",
    "largest gap between PRV and PLD sigma: 1.6% (Adult, B=300, E=10, eps=8)",
    "chosen cell: Adult, B=1000, E=10, eps=4: sigma=0.972900 (Opacus PRV, as Part 5)",
    "Opacus RDP         eps=4.50",
    "Opacus PRV         eps=3.99",
    "dp_accounting RDP  eps=4.50",
    "dp_accounting PLD  eps=3.98",
    "Opacus GDP         eps=3.34",
    "PLD <= target: True; RDP >= PLD: True; |PRV - PLD|/PLD < 3%: True",
    "crosscheck: pass",
    "PLD bracket (grid 1e-4): lower 3.9656 (optimistic), upper 3.9806 (pessimistic); width 0.0150",
    "Poisson (PLD)                              eps=3.98",
    "fixed size, add/remove, sigma/2 (PLD)      eps=23.03",
    "fixed size, replace-one (RDP)              eps=52.60",
    "shuffle, zero-out, no amplification (PLD)  eps=18.51",
    "PLDAccountant supports sampling without replacement: add/remove False, replace-one False",
    "steps per epoch: 30; planned: 10 epochs = 300 steps",
    "epoch  1: eps=1.70",
    "epoch  2: eps=2.09",
    "epoch  4: eps=2.69",
    "epoch  6: eps=3.18",
    "epsilon never decreases over 10 epochs: True",
    "stopped early at epoch 6: eps=3.18; report the planned maximum: eps=3.99",
    "after resuming at epoch 6: eps at epoch 10=3.99; uninterrupted run: eps=3.99; equal: True",
    "steps recorded: 300; planned (the loader's 10 x 30): 300; equal: True",
    "Unit: one person (a row of Adult); neighbouring datasets differ by adding or removing one person",
    "Data: n = 30,000; Poisson sampling with q = 0.033333 (B = 1000); T = 300 steps (10 epochs)",
    "Mechanism: DP-SGD, clipping norm C = 1.0, noise multiplier sigma = 0.973",
    "Accountant: PLD, dp-accounting 0.6.0; calibrated with Opacus 1.6.0 (PRV)",
    "Guarantee: (epsilon = 3.99, delta = 1e-05); epsilon = 6.30 at delta = 1/n^2 = 1.1e-09",
    "Not covered: hyperparameter tuning runs (see Part 5), and any model released before this one",
]


@pytest.mark.libs
def test_practitioner_calibrate_prints_the_page_numbers() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT)], check=False, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    printed = [line for line in result.stdout.splitlines() if line.strip() and not line.startswith("==")]
    assert printed == EXPECTED
