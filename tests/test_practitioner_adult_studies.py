"""code/part05-dp-ml/practitioner_adult_studies.py prints what the practitioner DP-SGD pages 02 and 06-09 quote.

The whole script trains about 260 models (25 minutes on four processes), too long for the test suite. So the test
runs --check, which prints every accounting number the pages quote and five of the training runs, one from each
kind of trainer: Opacus from scratch, microbatches, an adapter on the pretrained model, households and groups.
Each run repeats exactly, so matching these five checks that the others still give the pages' numbers.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml" / "practitioner_adult_studies.py"

EXPECTED = [
    # 02-dp-sgd.md
    "epsilon at sigma=0.973, Poisson: 3.98; fixed-size batches, sigma/2: 23.03 (PLD)",
    "budget 4.0 1: test 0.8437",
    "microbatch 4: test 0.8206",
    # 06-public-data.md
    "pretrained on the public rows, no fine-tuning: 0.8383",
    "trainable weights, full: 6,018",
    "trainable weights, last layer: 130",
    "trainable weights, adapter: 750",
    "epsilon=1 (sigma=2.383), full: noise length 0.1848  signal length 0.1137  ratio 1.6",
    "epsilon=1 (sigma=2.383), last layer: noise length 0.0272  signal length 0.0726  ratio 0.4",
    "public adapter 4 1.0 0.3 0: choose 0.8427  score 0.8465",
    # 07-user-level.md
    "households 3,000: median 4 rows, largest 359",
    "delta for 3,000 households: 1e-04",
    "row-level epsilon at sigma=0.973 (the full edition's RDP accountant): 4.50",
    "group privacy, k=2: epsilon 12.43",
    "group privacy, k=10: epsilon inf",
    "household of k=1 rows: exact 4.59  screen at rate kq 4.50  (-2%)",
    "household of k=2 rows: exact 10.02  screen at rate kq 9.25  (-8%)",
    "household of k=5 rows: exact 29.96  screen at rate kq 25.37  (-15%)",
    "household of k=10 rows: exact 74.82  screen at rate kq 57.57  (-23%)",
    "household of k=20 rows: exact 202.89  screen at rate kq 138.67  (-32%)",
    "epsilon=4 with 100 rows per batch (q=0.003333): sigma 0.581",
    "epsilon=4 with 100 households per batch (q=0.03333): sigma 1.03",
    "cap k=1: 3,000 rows kept, sigma for household-level epsilon=4 1.03",
    "cap k=2: 5,394 rows kept, sigma for household-level epsilon=4 1.13",
    "cap k=50: 25,088 rows kept, sigma for household-level epsilon=4 4.29",
    "households households 4.0 0.2: test 0.8432  sigma 1.0300  rows 30000.0000",
    # 08-disparate-impact.md
    "White: 25,670 training rows, 14,352 test rows",
    "Asian-Pac-Islander: 926 training rows, 532 test rows",
    "Amer-Indian-Eskimo: 283 training rows, 167 test rows",
    "Other: 245 training rows, 141 test rows",
    "Black: 2,876 training rows, 1,650 test rows",
    "oversampling x4 at epsilon=1: sigma 2.383 -> 8.828",
    "exact 95% interval at accuracy 0.85, Asian-Pac-Islander (532 test rows): [0.816, 0.879]",
    "exact 95% interval at accuracy 0.85, Other (141 test rows): [0.781, 0.905]",
    "race 1.0 0: White 0.8358  Asian-Pac-Islander 0.8158  Amer-Indian-Eskimo 0.9401  Other 0.8936  Black 0.9024",
    # 09-choosing-epsilon.md
    (
        "epsilon=4 (sigma=0.973): pair at FPR 1e-3: 0.05461  curve at FPR 0.001: [0.01221, 0.01234]  "
        "curve at FPR 0.01: [0.06447, 0.06515]"
    ),
    (
        "epsilon=8 (sigma=0.722): pair at FPR 1e-3: 0.9997  curve at FPR 0.001: [0.04414, 0.04452]  "
        "curve at FPR 0.01: [0.155, 0.1563]"
    ),
    "right flags at prior 1%, FPR 1e-2, epsilon=8: 0.1363",
]


@pytest.mark.libs
def test_practitioner_adult_studies_prints_what_the_pages_quote() -> None:
    out = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    for expected in EXPECTED:
        assert expected in lines, f"missing line: {expected!r}"
