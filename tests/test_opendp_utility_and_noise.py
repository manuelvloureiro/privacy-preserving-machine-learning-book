"""code/part02-tabular-data/opendp_utility_and_noise.py runs and prints what the OpenDP book's chapter "Assessing
utility and choosing the noise" quotes.

The deterministic lines (conversions, scales, accuracies, thresholds, budgets, the epsilon of rho) are checked
exactly. OpenDP's noise cannot be seeded, so the released counts are checked against the true values within three
times their 95% accuracy, the 36 grouped counts by how many land within their bar (at least 30 of 36, which a
correct run misses less than once in ten thousand), and the means of the clamping experiment within four standard
errors of their expectations (a clamped Gaussian of scale 30 has mean 11.97 and standard deviation about 17.5, a
signed one mean 0 and standard deviation 30, each averaged over 1,000 draws).
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part02-tabular-data" / "opendp_utility_and_noise.py"

DETERMINISTIC = [
    "opendp version: 0.16.0",
    "true count: 3,811,932",
    "gaussian scale for accuracy 2 at alpha 0.05: 1.0204",
    "accuracy of that scale: 2.0000",
    "laplace count: Integer Laplace scale=180 95% accuracy=539.731",
    "budget per query: epsilon=0.2000",
    "error bar relative to the true count: 0.014%",
    "continuous laplace accuracy at scale 180: 539.2318",
    "discrete laplace scale for accuracy 100: 33.2152",
    "epsilon that scale needs at sensitivity 36: 1.0838  (5.42 times the share)",
    "laplace count at epsilon 2: Integer Laplace scale=90 95% accuracy=270.115",
    "privacy loss: (ZeroConcentratedDivergence, 0.1)",
    "gaussian count: Integer Gaussian scale=180 95% accuracy=354",
    "budget per query: rho=0.0200  sqrt(2 rho)=0.2000",
    "continuous gaussian accuracy at scale 180: 352.7935",
    "gaussian count at rho 0.4: Integer Gaussian scale=90 95% accuracy=177",
    "L1 sensitivity of the grouped count: 36  L2 when the 36 records fall in 36 groups: 6",
    "privacy loss with a threshold: (Approximate(ZeroConcentratedDivergence), (0.1, 1e-07))",
    "grouped count, 36 contributions: Integer Gaussian scale=180 95% accuracy=354 threshold=1133",
    "grouped count, 1 per year and quarter: Integer Gaussian scale=30 95% accuracy=60 threshold=184",
    "grouped count, 1 per year and quarter, pure DP: Integer Laplace scale=180 95% accuracy=539.731 threshold=3712",
    "true groups: 36  smallest 82,504  largest 131,656",
    "rho=0.1 at delta=1e-07: epsilon=2.3484",
    "closed form rho + 2 sqrt(rho ln(1/delta)): 2.6391",
    "rho=0.02 (one query) at delta=1e-07: epsilon=0.9951",
    "the whole context converted: epsilon=2.3484 delta=1e-07",
    "dp.len() without signed: warns=True clamped=True",
    "expected clamped mean at scale 30: scale/sqrt(2 pi)=11.9683",
    "standard error of the signed mean: scale/sqrt(1000)=0.9487",
    "clamping experiment (one run): 1000 noisy counts of nobody at gaussian scale 30",
]

NUMBER = r"(-?[\d,]+(?:\.\d+)?)"


def number(pattern: str, out: str) -> list[float]:
    m = re.search(pattern, out, re.MULTILINE)
    assert m, f"no line matches {pattern!r}"
    return [float(g.replace(",", "")) for g in m.groups()]


@pytest.fixture(scope="module")
def out() -> str:
    return subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout


@pytest.mark.libs
def test_deterministic_lines(out: str) -> None:
    lines = out.splitlines()
    for expected in DETERMINISTIC:
        assert expected in lines, f"missing line: {expected}"


@pytest.mark.libs
def test_clamping_experiment_means(out: str) -> None:
    clamped, signed = number(rf"^  unsigned mean {NUMBER}  signed mean {NUMBER}  \(true count 0\)$", out)
    assert abs(clamped - 11.9683) <= 4 * 17.5 / 1000**0.5  # the clamped noise has a positive mean
    assert abs(signed) <= 4 * 30 / 1000**0.5  # the signed noise has mean zero


@pytest.mark.libs
def test_released_values_near_truth(out: str) -> None:
    (laplace,) = number(rf"^  laplace count {NUMBER}  \(true 3,811,932, within 539\.7 with probability 0\.95\)$", out)
    assert abs(laplace - 3_811_932) <= 3 * 539.731
    (gaussian,) = number(rf"^  gaussian count {NUMBER}  \(true 3,811,932, within 354\.0 with probability 0\.95\)$", out)
    assert abs(gaussian - 3_811_932) <= 3 * 354
    quarters = re.findall(r"^  (\d{4}) Q(\d) count ([\d,]+)  \(true ([\d,]+), within 60\)$", out, re.MULTILINE)
    assert [(y, q) for y, q, _, _ in quarters] == [("2005", "1"), ("2005", "2"), ("2005", "3"), ("2005", "4")]
    for _, _, released, true in quarters:
        assert abs(int(released.replace(",", "")) - int(true.replace(",", ""))) <= 3 * 60
    released, total, inside = number(
        rf"^  groups released {NUMBER} of {NUMBER}; within 60 of the truth: {NUMBER}$", out
    )
    assert (released, total) == (36, 36)  # every year and quarter has far more than the threshold of 184 rows
    assert inside >= 30
