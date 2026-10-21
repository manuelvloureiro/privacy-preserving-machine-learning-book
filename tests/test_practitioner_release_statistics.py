"""code/part03-dp-foundations/practitioner_release_statistics.py runs and prints what practitioner Part 3's
release-statistics workflow quotes.

The deterministic lines (counts, truncation trade-offs, budget split, noise scales, accuracies, the threshold, true
values) are checked exactly. OpenDP's noise cannot be seeded, so the released values are checked against the true
values: within three times their 95% accuracy for the scalars, four times for the five countries (ten values, so
that a correct run fails less than once in ten thousand), and the number of released products within the band of
products whose true count lies within three accuracies of the threshold.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations" / "practitioner_release_statistics.py"
)

DETERMINISTIC = [
    "customers: 4,372  purchase lines: 397,884",
    "delta: 1e-05",
    "lines per customer: p50=41 p90=208 p99=662 max=7847",
    "invoice lines: 541,909",
    "lines without a customer id (dropped): 135,080",
    "lines with a customer id: 406,829 by 4,372 customers",
    "1/n^1.1 = 9.89e-05",
    "epsilon: 1",
    "rho for epsilon=1 at delta/2: 0.0284",
    "returns and cancellations, Quantity <= 0 (dropped): 8,905",
    "lines with UnitPrice <= 0 (dropped): 40",
    "purchase lines: 397,884 by 4,338 customers",
    "k=10: lines kept 10.2%  customers truncated 84.8%",
    "k=25: lines kept 22.4%  customers truncated 64.6%",
    "k=50: lines kept 36.9%  customers truncated 44.0%",
    "k=100: lines kept 55.1%  customers truncated 24.9%",
    "k=200: lines kept 72.7%  customers truncated 10.4%",
    "k=500: lines kept 88.2%  customers truncated 1.8%",
    "line value bounds: [0, 200] pounds  lines clipped: 1.02%",
    "customers with lines in more than one country: 8",
    "products per customer: p50=35 p90=143 p99=354 max=1787",
    "budget scalars: rho=0.0071 delta=1.25e-06",
    "budget countries: rho=0.0071 delta=1.25e-06",
    "budget products: rho=0.0142 delta=2.50e-06",
    "whole release as (epsilon, delta): (1.0000, 1e-05)",
    "scalars len: Frame Length, Integer Gaussian, scale=1,453.38 95% accuracy=2,850.0",
    "scalars revenue: Sum, Float Gaussian, scale=290,804.33 95% accuracy=569,966.0",
    "scalars median: 0.5-Quantile, GumbelMin, scale=1,453.38 95% accuracy=none",
    "per country len: Frame Length, Integer Gaussian, scale=1,185.25 95% accuracy=2,324.0",
    "per country revenue: Sum, Float Gaussian, scale=237,698.48 95% accuracy=465,880.5",
    "m=1: pairs kept 1.6%  customers truncated 97.9%  scale=5.93  threshold=29  products above it: 1",
    "m=5: pairs kept 7.9%  customers truncated 92.4%  scale=13.27  threshold=66  products above it: 9",
    "m=10: pairs kept 15.1%  customers truncated 83.6%  scale=18.77  threshold=96  products above it: 19",
    "m=20: pairs kept 27.5%  customers truncated 68.3%  scale=26.54  threshold=138  products above it: 30",
    "m=50: pairs kept 52.7%  customers truncated 38.6%  scale=41.96  threshold=225  products above it: 50",
    "per product customers: Frame Length, Integer Gaussian, scale=41.96 95% accuracy=83.0 threshold=225",
    "true lines after truncation: 219,291 (all purchase lines: 397,884)",
    "true revenue after truncation and clipping: 4,318,947 (all lines, unclipped: 8,911,408)",
    "truncation and clipping keep 48.5% of the revenue",
    "true median line value after truncation: 15.00 (all lines: 11.80)",
    "true United Kingdom: lines 196,948  revenue 3,719,773",
    "true Germany: lines 5,379  revenue 131,902",
    "true France: lines 4,793  revenue 107,446",
    "true Belgium: lines 1,549  revenue 31,722",
    "true Spain: lines 1,387  revenue 31,927",
    "keys with no purchase lines: 11 of 38",
    "lines mapped to Other: 1,750 from 11 source values",
    "products with any purchase: 3,665",
    "products with true count >= threshold: 50",
    "products with true count >= threshold-3x accuracy: 3512",
    "products with true count >= threshold+3x accuracy: 2",
    (
        "This release satisfies (epsilon = 1, delta = 1e-05)-differential privacy for adding or removing one customer and all their invoice lines, with at most 100 lines per customer (sampled at random), at most 50 random products per customer in the product counts, and line values clipped to [0, 200] pounds, under the central model, using OpenDP 0.16.0 with Gaussian noise (accounted as 0.0284-zCDP plus 5e-06 for thresholds, converted to epsilon and delta). The market list was fixed in advance; the product list is not public."
    ),
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
def test_released_values_near_truth(out: str) -> None:
    lines_hat, revenue_hat, median_hat = number(rf"^  lines {NUMBER}  revenue {NUMBER}  median {NUMBER}$", out)
    assert abs(lines_hat - 219_291) <= 3 * 2_850.0
    assert abs(revenue_hat - 4_318_947) <= 3 * 569_966.0
    assert abs(median_hat - 15.0) <= 3.0  # no stated accuracy; runs give about 14 against 15
    for country in ("United Kingdom", "Germany", "France", "Belgium", "Spain"):
        true_lines, true_revenue = number(rf"^true {country}: lines {NUMBER}  revenue {NUMBER}$", out)
        hat_lines, hat_revenue = number(rf"^  {country}: lines {NUMBER}  revenue {NUMBER}$", out)
        assert abs(hat_lines - true_lines) <= 4 * 2_324.0, country
        assert abs(hat_revenue - true_revenue) <= 4 * 465_880.5, country
    (released,) = number(rf"^  products released: {NUMBER}$", out)
    assert 2 <= released <= 3512
    for hat, true in re.findall(r"^  product \S+: customers (\d+)  true (\d+)$", out, re.MULTILINE):
        assert int(hat) >= 225 and int(true) >= 225 - 3 * 83.0
