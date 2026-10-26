"""code/part01-getting-started/opendp_first_release.py runs and prints what the OpenDP book's first chapter quotes.

The deterministic lines (privacy maps, the unit and the budget, scales, accuracies, the refusal) are checked exactly.
OpenDP's noise cannot be seeded, so the released count and mean are checked against the true values within three
times their 95% accuracy (the mean's bar is the one at the true size; its released scale is a little different).
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part01-getting-started" / "opendp_first_release.py"

DETERMINISTIC = [
    "opendp version: 0.16.0",
    "laplace scale 1: map(1.0)=1.0 map(2.0)=2.0",
    "laplace scale 1: check(1.0, 1.0)=True check(1.0, 0.5)=False",
    "laplace scale 2: map(1.0)=0.5",
    "true count: 100  true mean: 50.93",
    "privacy unit: (SymmetricDistance(), 1)",
    "privacy loss: (MaxDivergence, 1.0)",
    "bounds: (0.0, 100.0)  imputed value: 50",
    "budget per query: [0.3333, 0.3333, 0.3333]",
    "count: scale=3.0000 95% accuracy=9.4457",
    "mean at size 100: scale=3.0000 95% accuracy=8.9872",
    "budget spent after two queries: [0.3333, 0.3333]",
    "fourth query: Privacy allowance has been exhausted",
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
    (count,) = number(rf"^  count {NUMBER}  \(true 100, within 9\.4 with probability 0\.95\)$", out)
    assert abs(count - 100) <= 3 * 9.4457
    scale, mean = number(rf"^  mean scale {NUMBER}  mean {NUMBER}  \(true 50\.93\)$", out)
    assert 2.0 <= scale <= 4.5  # 300 / dp_count, with dp_count within its bar
    assert abs(mean - 50.93) <= 3 * 8.9872 + 5  # the noise, plus the resize's padding or dropping of a few rows
    (third,) = number(rf"^  third query, a second count {NUMBER}$", out)
    assert abs(third - 100) <= 3 * 9.4457
