"""code/part02-tabular-data/opendp_essential_statistics.py runs and prints what the OpenDP book's chapter
"Essential statistics" quotes.

The deterministic lines (the unit and the budget, the summarize table, scales, accuracies, refusals, the warning,
the true values) are checked exactly. OpenDP's noise cannot be seeded, so the released count, sums and means are
checked against the true values within three times their 95% accuracy. The joint mean is a ratio of two noisy
numbers, so its bound is the sum's bar plus the count's bar times the mean, over the count (about 0.25), rounded up
to 0.3. The median and the quartiles are checked for equality with the true ones: on this data the nearest other
candidate scores at least 74,383 above the true quartile against a noise scale of at most 3,240, so the exponential
mechanism picks another candidate with probability below 1e-15 (the script prints the gaps).
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part02-tabular-data" / "opendp_essential_statistics.py"

DETERMINISTIC = [
    "opendp version: 0.16.0",
    "privacy unit: (SymmetricDistance(), 36)",
    "privacy loss: (MaxDivergence, 1.0)",
    "budget per query: [0.2, 0.2, 0.2, 0.2, 0.2]",
    "margin: max_length=5,400,000",
    "count summary (alpha=0.05):",
    "shape: (1, 5)",
    "│ column ┆ aggregate    ┆ distribution    ┆ scale ┆ accuracy   │",
    "│ len    ┆ Frame Length ┆ Integer Laplace ┆ 180.0 ┆ 539.731115 │",
    "count: scale=180.0 95% accuracy=539.73 99% accuracy=829.43",
    "sensitivity 36 over epsilon 0.2: scale 180.0",
    (
        "without signed=True: FutureWarning: The default value is currently signed=False, and counts are clamped "
        "at zero, consistent with past releases, but the default will be changing to signed=True."
    ),
    "HWUSUAL len alone: scale=180.0",
    "HWUSUAL count alone: scale=180.0",
    "HWUSUAL null_count alone: scale=180.0",
    "HWUSUAL n_unique alone: scale=180.0",
    "five counts in one query: scale=900.0 each, 95% accuracy=2696.7",
    "sum HWUSUAL: Sum, Integer Laplace, scale=14,400.00 95% accuracy=43,139.0",
    "sensitivity 36 x 80 = 2,880 over epsilon 0.2: scale 14,400",
    "without a margin: must specify 'max_length' in a margin with by=[]",
    "float sum, max_length 5,400,000: HWUSUAL: Sum, Float Laplace, scale=14,405.18 95% accuracy=43,154.1",
    "float sum, max_length 1,000,000,000: HWUSUAL: Sum, Float Laplace, scale=192,035.68 95% accuracy=575,287.5",
    "int sum, max_length 1,000,000,000: HWUSUAL: Sum, Integer Laplace, scale=14,400.00 95% accuracy=43,139.0",
    "sum without fill_null: scale=14,400.0 (accepted; a null becomes the midpoint 40)",
    (
        "clip, sum and laplace by hand, without fill_null: input data (HWUSUAL) might contain nulls. "
        "Preprocess your data with `.fill_null`."
    ),
    "joint HWUSUAL: Sum, Integer Laplace, scale=28,800.00 95% accuracy=86,277.6",
    "joint len: Frame Length, Integer Laplace, scale=360.00 95% accuracy=1,079.0",
    "dp.mean on the add/remove context: HWUSUAL: Sum, Integer Laplace, scale=28,800.00 95% accuracy=86,277.6",
    "dp.mean on the add/remove context: HWUSUAL: Length, Integer Laplace, scale=360.00 95% accuracy=1,079.0",
    "bounded mean HWUSUAL: Sum, Integer Laplace, scale=7,200.00 95% accuracy=21,569.8",
    "bounded mean HWUSUAL: Length, Integer Laplace, scale=0.00 95% accuracy=none",
    (
        "dp.mean after a filter inside the bounded context: HWUSUAL: Sum, Integer Laplace, scale=28,800.00 "
        "95% accuracy=86,277.6"
    ),
    (
        "dp.mean after a filter inside the bounded context: HWUSUAL: Length, Integer Laplace, scale=360.00 "
        "95% accuracy=1,079.0"
    ),
    "median HWUSUAL: 0.5-Quantile, ExponentialMin, scale=360.00 95% accuracy=none",
    "score sensitivity 36: scale 2 x 36 / 0.2 = 360",
    "median without a margin: Must know max_length",
    "quartiles 0.25-Quantile: 0.25-Quantile, ExponentialMin, scale=3,240.00 95% accuracy=none",
    "quartiles 0.5-Quantile: 0.5-Quantile, ExponentialMin, scale=1,080.00 95% accuracy=none",
    "quartiles 0.75-Quantile: 0.75-Quantile, ExponentialMin, scale=3,240.00 95% accuracy=none",
    "0.25-quantile alone: scale=1,080.0",
    "true responses: 3,811,932",
    "true responses with HWUSUAL not 99: 1,496,192 (nulls among them: 0; in the whole column: 20,819)",
    "true sum of hours after the filter, clipped to [0, 80]: 56,315,594 (values above 80: 0)",
    "true sum of the whole column, clipped: 240,742,034 with nulls as 40, 239,909,274 with nulls dropped",
    "true mean: 37.64  (bounded mean 95% accuracy: 0.0144)",
    "true quartiles: 35, 37, 40",
    "score gap from the true quartile to the nearest other candidate: 0.25: 219,506  0.5: 74,383  0.75: 116,811",
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
    (count,) = number(rf"^  responses {NUMBER}  \(true 3,811,932, within 539\.7 with probability 0\.95\)$", out)
    assert abs(count - 3_811_932) <= 3 * 539.731115
    (total,) = number(rf"^  sum of hours {NUMBER}  \(true 56,315,594, within 43,139\.0\)$", out)
    assert abs(total - 56_315_594) <= 3 * 43_139.04473
    joint_sum, joint_len, joint_mean = number(
        rf"^  joint: sum {NUMBER}  count {NUMBER}  mean {NUMBER}  \(true 37\.64\)$", out
    )
    assert abs(joint_sum - 56_315_594) <= 3 * 86_277.589474
    assert abs(joint_len - 1_496_192) <= 3 * 1_078.963271
    assert abs(joint_mean - 56_315_594 / 1_496_192) <= 0.3  # (3 x 86,278 + 37.64 x 3 x 1,079) / 1,496,192 = 0.25
    (mean,) = number(rf"^  bounded mean {NUMBER}  \(true 37\.6393, within 0\.0144\)$", out)
    assert abs(mean - 56_315_594 / 1_496_192) <= 3 * 21_569.772352 / 1_496_192
    median, q1, q2, q3 = number(
        rf"^  median {NUMBER}  quartiles {NUMBER}, {NUMBER}, {NUMBER}  \(true 37; 35, 37, 40\)$", out
    )
    assert (median, q1, q2, q3) == (37, 35, 37, 40)  # the score gaps make any other candidate negligible
    (whole,) = number(rf"^  sum of the whole column {NUMBER}  \(true with nulls as 40: 240,742,034\)$", out)
    assert abs(whole - 240_742_034) <= 3 * 43_139.04473  # and 832,760 from the sum that drops the nulls
