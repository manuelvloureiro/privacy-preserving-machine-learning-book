"""code/part03-framework/opendp_transformations.py runs and prints what the OpenDP book's Transformations chapter quotes.

The deterministic lines (the true values, every stability map and `check`, the scales and accuracies, the scores,
the refusals) are checked exactly. OpenDP's noise cannot be seeded, so the released sum and mean are checked against
the true values within three times their 95% accuracy; the released median must be the true one, because the gap
between the two best scores is over five hundred times the noise scale (the page states and justifies this); and
the float sum truncated to 2^20 rows by a random sample must be the truncated share of the true sum, within 2%.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part03-framework" / "opendp_transformations.py"

DETERMINISTIC = [
    "opendp version: 0.16.0",
    "hours vector: 1,496,192 values",
    "true sum: 56,315,594  true mean: 37.6393  true median: 37",
    "int sum: value 56,315,594  map(1)=98  map(36)=3528",
    "sized int sum: map(36)=1764  map(35)=1666",
    "float sum, size limit 2^20=1,048,576: map(36)=3528.0000009127",
    "float sum, size limit 4,000,000: value 56,315,594  map(36)=3528.0000038179",
    "sum release: scale=7056  map(36)=0.5000  95% accuracy=21,138.4",
    "mean without a size: dataset size must be known. Either specify size in the input domain or use make_resize",
    "mean chain: value 37.6393  map(36)=0.0023580",
    "mean release: scale=0.0047160  map(36)=0.5000  95% accuracy=0.0141",
    "variance chain: value 137.6290  map(36)=0.2311",
    "median scores: lowest 15,190 at 37 hours, next 89,573 at 36  map(36)=36",
    "median release: scale=144  map(36)=0.5000  score gap=74,383  gap/scale=517",
    "int sum chain: check(36, 3528)=True  check(36, 3527)=False",
    "mismatched chain: DomainMismatch: Intermediate domains don't match. See https://github.com/opendp/opendp/discussions/297",
    "    output_domain: VectorDomain(AtomDomain(bounds=[0, 98], T=i32))",
    "    input_domain:  VectorDomain(AtomDomain(bounds=[0, 100], T=i32))",
]

TRUE_SUM = 56_315_594
TRUE_MEAN = 37.6393
TRUNCATED_SHARE = 2**20 / 1_496_192  # the random sample keeps this share of the rows

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
    (total,) = number(rf"^  sum {NUMBER}  \(true 56,315,594, within 21,138 with probability 0\.95\)$", out)
    assert abs(total - TRUE_SUM) <= 3 * 21_138.4
    (truncated,) = number(rf"^  float sum truncated to 2\^20 rows {NUMBER}  \(true 56,315,594; about 70% of it\)$", out)
    assert abs(truncated - TRUNCATED_SHARE * TRUE_SUM) <= 0.02 * TRUE_SUM
    (mean,) = number(rf"^  mean {NUMBER}  \(true 37\.6393, within 0\.0141 with probability 0\.95\)$", out)
    assert abs(mean - TRUE_MEAN) <= 3 * 0.0141
    (median,) = number(rf"^  median {NUMBER}  \(true 37\)$", out)
    assert median == 37
