"""code/part03-framework/opendp_combinators.py runs and prints what the OpenDP book's Combinators chapter quotes.

The deterministic lines (privacy maps, stability maps, accuracies, privacy profiles, conversions, error messages,
the LFS scales and the true values) are checked exactly. OpenDP's noise cannot be seeded, so the released values
are checked against the true values within three times their 95% accuracy: the toy sums and counts against the
true clamped sum and count, and the LFS sums against the true clipped sum of HWUSUAL.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part03-framework" / "opendp_combinators.py"

DETERMINISTIC = [
    "opendp version: 0.16.0",
    "toy data: [9, 7, 5, 2, 3, 0, 0, 0, 1, 8, 7, 10, 5, 6, 10, 8, 6, 5, 6, 10]",
    "true count: 20  true clamped sum: 108",
    "clamp >> sum: map(1)=10",
    "clamp >> sum >> laplace: map(1)=2.0 check(1, 2.0)=True",
    "with >>: map(1)=2.0",
    "post-processed: map(1)=2.0",
    "domain mismatch: Intermediate domains don't match. See https://github.com/opendp/opendp/discussions/297",
    "    output_domain: AtomDomain(T=f64)",
    "    input_domain:  AtomDomain(T=i32)",
    "metric mismatch: Intermediate metrics don't match. See https://github.com/opendp/opendp/discussions/297",
    "    output_metric: AbsoluteDistance(i32)",
    "    input_metric:  AbsoluteDistance(f64)",
    "sum: scale=5 95% accuracy=15.4537",
    "count: scale=1 95% accuracy=3.3756",
    "composition: map(1)=3.0  parts: 2.0 + 1.0",
    "composition output measure: MaxDivergence",
    "zCDP composition: map(1)=2.50  parts: 2.00 + 0.50",
    "mixed measures: All output measures must be the same",
    "adaptive: map(1)=3.0",
    "second query over its share: insufficient budget for query: 2.0 > 1.0",
    "third query: out of queries",
    "LFS context over 1: accountant map(36)=1.0 d_mids=[1.0]",
    "LFS HWUSUAL sum over 1: scale=2,881.04 95% accuracy=8,630.8",
    "LFS context over 3: accountant map(36)=1.0 d_mids=[0.3333, 0.3333, 0.3333]",
    "LFS HWUSUAL sum over 3: scale=8,643.11 95% accuracy=25,892.4",
    "true HWUSUAL sum: 56,315,594",
    "odometer loss: before=0.0 after sum=2.0 after count=3.0 after another count=4.0",
    "odometer loss at d_in=2 after three queries: 8.0",
    "filter: map(1)=2.0",
    "filter third query: filter is now exhausted: pending privacy loss (3.0) would exceed privacy budget (2.0)",
    "filter loss after refusal: filter is exhausted: no more queries can be answered",
    "context without a split: queryable=OdometerQueryable spent=1.0 remaining=1.0",
    "transformation >> compositor: d_in=10 map(1)=3.0",
    "laplace on the sum: map(10)=2.0",
    "nested: outer map(1)=4.0 inner map(1)=3.0",
    "pure DP to zCDP: epsilon=1.0 rho=0.50",
    "gaussian: rho=0.1000",
    "profile at rho=0.1: epsilon(delta=1e-07)=2.3484 delta(epsilon=1)=0.008933",
    "simple conversion: 2.6391  improved conversion: 2.3484",
    "fix_delta: map(1.0)=(2.3484, 1e-07)  measure=Approximate(MaxDivergence)",
    "approximate: map(1.0)=(1.0, 0.0)  measure=Approximate(MaxDivergence)",
    "fixed to profile: epsilon(delta=1e-07)=1.0 epsilon(delta=0)=1.0",
    "laplace count + gaussian count: map(1)=(6.6709, 1e-07)",
    "sample mean: map(2)=2.0000",
    "amplified, population 100: map(2)=0.4940  formula=0.4940",
    "amplified, population 1000: map(2)=0.0619  formula=0.0619",
    "amplifying an unsized domain: elements of the vector domain have unknown size",
]

NUMBER = r"(-?[\d,]+(?:\.\d+)?)"
TRUE_SUM = 108
TRUE_COUNT = 20
SUM_ACCURACY = 15.4537
COUNT_ACCURACY = 3.3756
TRUE_HOURS = 56_315_594
HOURS_ACCURACY = {1: 8_630.8, 3: 25_892.4}


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
    total, count = number(rf"^  composition: sum {NUMBER}  count {NUMBER}  \(true 108 and 20\)$", out)
    assert abs(total - TRUE_SUM) <= 3 * SUM_ACCURACY
    assert abs(count - TRUE_COUNT) <= 3 * COUNT_ACCURACY
    total, count = number(rf"^  adaptive: sum {NUMBER}  count {NUMBER}$", out)
    assert abs(total - TRUE_SUM) <= 3 * SUM_ACCURACY
    assert abs(count - TRUE_COUNT) <= 3 * COUNT_ACCURACY
    total, count_1, count_2 = number(rf"^  odometer: sum {NUMBER}  counts {NUMBER} {NUMBER}$", out)
    assert abs(total - TRUE_SUM) <= 3 * SUM_ACCURACY
    assert abs(count_1 - TRUE_COUNT) <= 3 * COUNT_ACCURACY
    assert abs(count_2 - TRUE_COUNT) <= 3 * COUNT_ACCURACY
    count_1, count_2 = number(rf"^  filter: counts {NUMBER} {NUMBER}$", out)
    assert abs(count_1 - TRUE_COUNT) <= 3 * COUNT_ACCURACY
    assert abs(count_2 - TRUE_COUNT) <= 3 * COUNT_ACCURACY
    (count,) = number(rf"^  context without a split: count {NUMBER}$", out)
    assert abs(count - TRUE_COUNT) <= 3 * COUNT_ACCURACY
    sum_on_sum, inner_sum, inner_count, outer_count = number(
        rf"^  nested: sum on the sum {NUMBER}  inner sum {NUMBER}  inner count {NUMBER}  outer count {NUMBER}$", out
    )
    assert abs(sum_on_sum - TRUE_SUM) <= 3 * SUM_ACCURACY
    assert abs(inner_sum - TRUE_SUM) <= 3 * SUM_ACCURACY
    assert abs(inner_count - TRUE_COUNT) <= 3 * COUNT_ACCURACY
    assert abs(outer_count - TRUE_COUNT) <= 3 * COUNT_ACCURACY
    for queries, accuracy in HOURS_ACCURACY.items():
        (hours,) = number(
            rf"^  LFS HWUSUAL sum over {queries}: {NUMBER}  \(true 56,315,594, within [\d,]+ with probability 0\.95\)$",
            out,
        )
        assert abs(hours - TRUE_HOURS) <= 3 * accuracy, queries
