"""code/part01-getting-started/opendp_data_context.py runs and prints what the OpenDP book's second chapter quotes.

The deterministic lines (the sizes of the LFS sample, the unit and the budget and its splits, the mock query's
schema and refusals, the count's scale and accuracy, the margin, the true count) are checked exactly. OpenDP's noise
cannot be seeded, so the released count is checked against the true count within three times its 95% accuracy.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part01-getting-started" / "opendp_data_context.py"

DETERMINISTIC = [
    "opendp version: 0.16.0",
    (
        "schema: PIDENT Int64, YEAR Int64, QUARTER Int64, SEX Int64, AGE Float64, HWUSUAL Float64, HWACTUAL Float64, "
        "ILOSTAT Int64"
    ),
    "rows: 3,811,932  people: 1,682,306",
    "years: 2005 to 2013  quarters: 36",
    "responses per person: max 15  distinct quarters per person: max 4",
    "privacy unit: (SymmetricDistance(), 36)",
    "privacy loss: (MaxDivergence, 1.0)",
    "budget split evenly over 5: [0.2, 0.2, 0.2, 0.2, 0.2]",
    "budget split by weights [2, 1, 1, 1]: [0.4, 0.2, 0.2, 0.2]",
    "query type: LazyFrameQuery",
    (
        "query schema: Schema({'PIDENT': Int64, 'YEAR': Int64, 'QUARTER': Int64, 'SEX': Int64, 'AGE': Float64, "
        "'HWUSUAL': Float64, 'HWACTUAL': Float64, 'ILOSTAT': Int64})"
    ),
    "collect before release: You must call `.release()` before executing a query.",
    (
        'plain Polars mean: MakeMeasurement("Expr is not recognized at this time: col("AGE").mean(). '
        'If you would like to see this supported, please file an issue.")'
    ),
    "│ len    ┆ Frame Length ┆ Integer Laplace ┆ 180.0 ┆ 539.731115 │",
    "count at alpha=0.01: accuracy=829.4299",
    (
        "dp.len() without signed: FutureWarning: The default value is currently signed=False, and counts are clamped "
        "at zero, consistent with past releases, but the default will be changing to signed=True."
    ),
    "margin: Margin(by=[], max_length=5400000, max_groups=None, invariant=None)",
    "sum without a margin: MakeTransformation(\"must specify 'max_length' in a margin with by=[]\")",
    "sum with the margin: scale=14,405.2  (36 x 80 over epsilon / 5 = 14,400)",
    "true count: 3,811,932",
    "  budget spent: [0.2]",
    '  second collect: FailedFunction("OnceFrame has been exhausted")',
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
def test_released_count_near_truth(out: str) -> None:
    (count,) = number(rf"^  count {NUMBER}  \(true 3,811,932, within 539\.7 with probability 0\.95\)$", out)
    assert abs(count - 3_811_932) <= 3 * 539.73
