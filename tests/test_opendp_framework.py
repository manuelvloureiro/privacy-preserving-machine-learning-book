"""code/part03-framework/opendp_framework.py runs and prints what the OpenDP book's framework chapter quotes.

The deterministic lines (the five elements of the Laplace measurement, its maps and checks, the domains and metrics
of the chain, the stability of the sum, the chain's privacy loss, the scales of both APIs, the refusal) are checked
exactly. OpenDP's noise cannot be seeded, so the released values are checked against the true values within three
times their 95% accuracy (the mean's bar is the one at the true size; its released scale is a little different).
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part03-framework" / "opendp_framework.py"

DETERMINISTIC = [
    "opendp version: 0.16.0",
    "laplace scale 5: type=Measurement",
    "laplace scale 5: input domain=AtomDomain(T=f64)",
    "laplace scale 5: input metric=AbsoluteDistance(f64)",
    "laplace scale 5: output measure=MaxDivergence",
    "laplace scale 5: function=Function",
    "laplace scale 5: map(1.0)=0.2 map(10.0)=2.0",
    "laplace scale 5: check(1.0, 0.2)=True check(1.0, 0.19)=False",
    "laplace scale 5: 95% accuracy=14.9787",
    "hours: n=1,496,192 min=0 max=80 true sum=56,315,594",
    "clamp: input domain=VectorDomain(AtomDomain(T=i32))",
    "clamp: output domain=VectorDomain(AtomDomain(bounds=[0, 80], T=i32))  map(36)=36",
    "sum: output domain=AtomDomain(T=i32)  output metric=AbsoluteDistance(i32)",
    "sum: map(1)=80 map(36)=2880",
    "chain scale 5: input domain=VectorDomain(AtomDomain(T=i32))",
    "chain scale 5: input metric=SymmetricDistance()  output measure=MaxDivergence",
    "chain scale 5: map(36)=576.0",
    "chain for epsilon 1 at d_in 36: scale=2880.0 map(36)=1.0",
    "chain for epsilon 1 at d_in 36: 95% accuracy=8,628.2 (0.01532% of the sum)",
    "true count: 100  true mean: 50.93",
    "unit: d_in=1 input metric=SymmetricDistance()  (Context API: (SymmetricDistance(), 1))",
    "loss: d_out=1.0 output measure=MaxDivergence  (Context API: (MaxDivergence, 1.0))",
    "input domain: VectorDomain(AtomDomain(nan=true, T=f64))",
    "compositor: type=Measurement map(1)=1.0",
    "queryable: type=Queryable",
    "context accountant: type=Measurement d_mids=[0.3333, 0.3333, 0.3333]",
    "count: sensitivity=1  output metric=AbsoluteDistance(i32)",
    "count at scale 3.0: map(1)=0.33333333333333337 check(1, 1/3)=False",
    "count by search: scale=3.0000000000000004 check(1, 1/3)=True",
    "count scale: framework=3.0000 context=3.0000 equal=True",
    "count: 95% accuracy=9.4457",
    "mean: output domain=AtomDomain(T=f64)  output metric=AbsoluteDistance(f64)",
    "mean at size 100: sensitivity=1.0000",
    "mean scale at size 100: framework=3.0000 context=3.0000 equal=True",
    "mean at size 100: 95% accuracy=8.9872",
    "mean scale at the released count: equal=True",
    "fourth query: FailedFunction: out of queries",
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
    (noisy_zero,) = number(rf"^  laplace scale 5 on 0\.0: {NUMBER}  \(within 15\.0 with probability 0\.95\)$", out)
    assert abs(noisy_zero) <= 3 * 14.9787
    (hours_sum,) = number(rf"^  hours sum {NUMBER}  \(true 56,315,594, within 8,628 with probability 0\.95\)$", out)
    assert abs(hours_sum - 56_315_594) <= 3 * 8_628.2
    (count,) = number(rf"^  count {NUMBER}  \(true 100, within 9\.4 with probability 0\.95\)$", out)
    assert abs(count - 100) <= 3 * 9.4457
    scale, mean = number(rf"^  mean scale {NUMBER}  mean {NUMBER}  \(true 50\.93\)$", out)
    assert 2.0 <= scale <= 4.5  # 300 / dp_count, with dp_count within its bar
    assert abs(mean - 50.93) <= 3 * 8.9872 + 5  # the noise, plus the resize's padding or dropping of a few rows
    (third,) = number(rf"^  third query, a second count {NUMBER}$", out)
    assert abs(third - 100) <= 3 * 9.4457
