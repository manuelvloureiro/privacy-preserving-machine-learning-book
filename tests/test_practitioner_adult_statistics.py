"""code/part03-dp-foundations/practitioner_adult_statistics.py runs and prints what practitioner Part 3's Adult
examples quote.

The diffprivlib lines are seeded (random_state=0) and checked exactly, as are OpenDP's scales and accuracies and the
true values. OpenDP's noise cannot be seeded, so its released values are checked against the true values: the median
within one year of age, and each of the 32 marginal counts within four times its 95% accuracy (Laplace noise of scale
1 exceeds that with probability about 1e-6 per cell).
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations" / "practitioner_adult_statistics.py"

DETERMINISTIC = [
    "rows: 48,842",
    "1/n^1.1 = 7.0e-06  delta: 1e-06",
    "private mean age: 38.64  private count over 50: 9,813",
    "spent: (epsilon=1.0, delta=0.0)",
    (
        "refused: Privacy spend of (0.1,0) not permissible; will exceed remaining privacy budget. "
        "Use BudgetAccountant.remaining() to check remaining budget."
    ),
    "remaining: epsilon=0.00 delta=1e-06",
    "true mean age: 38.64  true count over 50: 9,808",
    "group of 10 at (0.5, 1e-5): (5.0, 2.272e-03)",
    "simple bound k e^((k-1) eps) delta: (5.0, 9.002e-03)",
    "delta multiplier, group of 10 at eps=1: 12818.3",
    "Laplace count: epsilon 0.5 per record, 5.0 for 10",
    "Gaussian count, group of 10: epsilon 6.43 at delta 1e-5",
    "Gaussian count, group of 10: epsilon 4.57 at delta 2.272e-03",
    "true median age: 37.0  diffprivlib median: 37.72",
    "opendp median: 0.5-Quantile, ExponentialMin, scale=2",
    "marginal cells: 32  Integer Laplace, scale=1  95% accuracy=3.38",
    "smallest true cell: Preschool >50K = 1",
    (
        "PrivacyLeakWarning: Bounds have not been specified and will be calculated on the data provided. This will "
        "result in additional privacy leakage. To ensure differential privacy and no additional privacy leakage, specify "
        "bounds for each dimension."
    ),
    "with bounds (17, 90): 37.72",
    "true share over 50K: 0.2393",
    "local estimate: 0.2359  sd 0.0039",
    "central estimate: 0.2393  sd 0.000026",
    "local sd / central sd: 149",
    "true mean hours: 40.422",
    "bounds (1, 99): 40.421  bounds (20, 60): 40.361",
    "bias of (20, 60): -0.061",
    "people outside (20, 60): 8.7%",
    "noise scale (1, 99): 0.0020  (20, 60): 0.0008",
]


@pytest.mark.libs
def test_adult_statistics() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True, timeout=120)
    lines = result.stdout.splitlines()
    for expected in DETERMINISTIC:
        assert expected in lines, expected

    released = lines[lines.index("released (one run):") + 1 :]
    median = re.fullmatch(r"  opendp median age: (\d+)", released[0])
    assert median is not None
    assert abs(int(median.group(1)) - 37) <= 1

    cells = [re.fullmatch(r"  (\S+) (\S+): (-?\d+)  true (\d+)  accuracy ([\d.]+)", line) for line in released[1:]]
    assert len(cells) == 32
    for cell in cells:
        assert cell is not None
        noisy, true, accuracy = int(cell.group(3)), int(cell.group(4)), float(cell.group(5))
        assert abs(noisy - true) <= 4 * accuracy, cell.group(0)
