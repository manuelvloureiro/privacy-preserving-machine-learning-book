"""code/part02-tabular-data/opendp_preparing_microdata.py runs and prints what the OpenDP book's chapter on preparing
microdata quotes.

The deterministic lines (the unit and the budget, the schema, noise scales, accuracies, the threshold, the error
messages, the true values and the dev-time error bar) are checked exactly. OpenDP's noise cannot be seeded, so the
released values are checked against the true values: the mean of hours lost within three times its 95% error bar
(the bar that the sum's and the length's accuracies imply, each taken at three times), and the three released tables
within four times the accuracy of a count (thirteen values in all, so that a correct run fails less than once in a
hundred thousand).
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part02-tabular-data" / "opendp_preparing_microdata.py"

DETERMINISTIC = [
    "opendp version: 0.16.0",
    "rows: 3,811,932  people: 1,682,306",
    "privacy unit: (SymmetricDistance(), 36)",
    "privacy loss: (Approximate(MaxDivergence), (1.0, 1e-07))",
    "budget per query: epsilon=0.25 delta=2.5e-08",
    "schema: PIDENT=Int64  YEAR=Int64  QUARTER=Int64  SEX=Int64  AGE=Float64  HWUSUAL=Float64  HWACTUAL=Float64  ILOSTAT=Int64",
    "binned counts: len Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9",
    "without keys: len Frame Length, Integer Laplace, scale=144.00 threshold=2,973",
    (
        "true bins: null=20,819  [-inf, 0)=0  [0, 20)=89,495  [20, 40)=908,061  [40, 60)=406,947  [60, 80)=78,469  "
        "[80, 98)=13,220  [98, inf)=2,294,921"
    ),
    "filtered sum: HWUSUAL Sum, Integer Laplace, scale=11,520.00 95% accuracy=34,511.3",
    "sample and aggregate: HWUSUAL Sum, Integer Laplace, scale=17,280.00",
    "sample and aggregate: HWUSUAL Length, Integer Laplace, scale=576.00",
    "one group per person: HWUSUAL Sum, Integer Laplace, scale=480.00",
    "one group per person: HWUSUAL Length, Integer Laplace, scale=16.00",
    "float sum: HWUSUAL Sum, Float Laplace, scale=14,405.18 95% accuracy=43,154.1",
    "integer sum: HWUSUAL Sum, Integer Laplace, scale=14,400.00 95% accuracy=43,139.0",
    "prepared sum: HWUSUAL Sum, Integer Laplace, scale=14,400.00",
    "cast without fill_null: input data (HWUSUAL) might contain nulls. Preprocess your data with `.fill_null`.",
    (
        "clip before fill_null: input domain must consist of bounded data. Either specify bounds in the input domain or "
        "use make_clamp."
    ),
    "fill_null before fill_nan: input data (HWUSUAL) might contain nans. Preprocess your data with `.fill_nan`.",
    "float prepared sum: HWUSUAL Sum, Float Laplace, scale=14,405.18",
    "filled from a column: HWUSUAL Sum, Integer Laplace, scale=14,400.00",
    "filled from a nullable column: fill expression must not be nullable",
    "drop_nulls in with_columns: drop_nulls is not allowed in a row-by-row context",
    (
        'replace_strict without default: summarize scale=2.0; release: InvalidOperation(ErrString("incomplete mapping '
        'specified for `replace_strict`\\n\\nHint: Pass a `default` value to set unmapped values."))'
    ),
    "replace_strict with default: released keys one other two",
    "over 64: len Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9",
    "true over 64: true=1,190,845  false=2,620,832  null=255",
    "is_in: IsIn { nulls_equal: false } is not currently supported",
    "employed: len Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9",
    (
        'abs: Expr is not recognized at this time: [(col("HWACTUAL")) - (col("HWUSUAL"))].abs(). If you would like to '
        "see this supported, please file an issue."
    ),
    (
        "to_date without format: format must be specified; otherwise Polars will attempt to infer the format from the "
        "data, resulting in an unstable transformation"
    ),
    (
        "nanosecond datetime: Nanoseconds are not currently supported due to potential panics when parsing inputs. "
        "Please open an issue on the OpenDP repository if you would find this functionality useful. Otherwise, consider "
        "parsing into micro- or millisecond datetimes instead."
    ),
    (
        'concat_str: Expr is not recognized at this time: ConcatHorizontal { delimiter: "-", ignore_nulls: false }. If '
        "you would like to see this supported, please file an issue."
    ),
    "per month: len Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9",
    "first dates: 2005 Q1 -> 2005-03-01  2005 Q2 -> 2005-06-01",
    "true per month: 3=953,695  6=958,396  9=933,257  12=966,584",
    "hours lost: LOST Sum, Integer Laplace, scale=11,520.00 95% accuracy=34,511.3",
    "hours lost: LOST Length, Integer Laplace, scale=288.00 95% accuracy=863.3",
    "hours lost, bounds (-80, 80): LOST Sum, Integer Laplace, scale=23,040.00 95% accuracy=69,022.2",
    "true prepared rows: 1,490,113  true sum of hours lost: -9,289,648  true mean: -6.2342",
    "rows the clip changes: 42,830 (2.874%)  mean without the clip: -6.5319",
    "mean error bar at 95%: 0.0268",
    "budget spent: 4 of 4 queries",
]

COUNT_ACCURACY = 431.884579  # the 95% accuracy of a count at scale 144, printed rounded as 431.9
SUM_ACCURACY = 34511.33578
LENGTH_ACCURACY = 863.270461
TRUE_ROWS = 1_490_113
TRUE_MEAN = -6.2342
NUMBER = r"(-?[\d,]+(?:\.\d+)?)"


def number(pattern: str, out: str) -> list[float]:
    m = re.search(pattern, out, re.MULTILINE)
    assert m, f"no line matches {pattern!r}"
    return [float(g.replace(",", "")) for g in m.groups()]


def table(prefix: str, out: str) -> dict[str, int]:
    """The `key=value` pairs of the line that starts with the prefix, two spaces between pairs."""
    m = re.search(rf"^{re.escape(prefix)}(.*)$", out, re.MULTILINE)
    assert m, f"no line starts with {prefix!r}"
    pairs = [pair.rsplit("=", 1) for pair in m.group(1).strip().split("  ")]
    return {key: int(value.replace(",", "")) for key, value in pairs}


@pytest.fixture(scope="module")
def out() -> str:
    return subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout


@pytest.mark.libs
def test_deterministic_lines(out: str) -> None:
    lines = out.splitlines()
    for expected in DETERMINISTIC:
        assert expected in lines, f"missing line: {expected}"


@pytest.mark.libs
def test_released_tables_near_truth(out: str) -> None:
    true_bins = table("true bins: ", out)
    bins = table("  bins: ", out)
    assert set(bins) == set(true_bins) - {"null"}  # the public keys; the null bin is not among them
    for key, value in bins.items():
        assert abs(value - true_bins[key]) <= 4 * COUNT_ACCURACY, key
    true_over = table("true over 64: ", out)
    over = table("  over 64: ", out)
    assert set(over) == {"true", "false"}
    for key, value in over.items():
        assert abs(value - true_over[key]) <= 4 * COUNT_ACCURACY, key
    true_months = table("true per month: ", out)
    months = table("  per month: ", out)
    assert list(months) == ["3", "6", "9", "12"]
    for key, value in months.items():
        assert abs(value - true_months[key]) <= 4 * COUNT_ACCURACY, key


@pytest.mark.libs
def test_released_mean_near_truth(out: str) -> None:
    (mean_hat,) = number(
        rf"^  hours lost: mean {NUMBER}  \(true -6\.2342, within 0\.0268 with probability 0\.95\)$", out
    )
    # The released mean is a noisy sum over a noisy length; with both within three times their 95% accuracy, the
    # mean is within this bar of the truth.
    bar = (3 * SUM_ACCURACY + abs(TRUE_MEAN) * 3 * LENGTH_ACCURACY) / (TRUE_ROWS - 3 * LENGTH_ACCURACY)
    assert abs(mean_hat - TRUE_MEAN) <= bar
