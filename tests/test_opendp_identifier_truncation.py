"""code/part02-tabular-data/opendp_identifier_truncation.py runs and prints what the OpenDP book's chapter on
identifier truncation and bounds quotes.

The deterministic lines (the unit, the facts about the identifiers, scales, accuracies, thresholds, query plans,
error messages, the true values) are checked exactly. OpenDP's noise cannot be seeded, so the released means are
checked against the true values within three times their error bar, which the script derives from the 95% bars
of the sum and the length, and the 36 quarters within four times their 95% accuracy (36 values per table, so
that a correct run fails less than once in ten thousand). The quarters' hours are compared with the first row of
each identifier and quarter while the query keeps a random one; that difference is a few thousand hours against
a bar of 19,173, so the four-fold bar covers it.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part02-tabular-data" / "opendp_identifier_truncation.py"

DETERMINISTIC = [
    "opendp version: 0.16.0",
    'privacy unit: (SymmetricIdDistance { identifier: col("PIDENT") }, 1)',
    "identifiers: 1,682,306  rows: 3,811,932",
    "rows per identifier: p50=2 p90=4 p99=7 max=15",
    "identifiers with more than 10 rows: 195",
    "quarters per identifier: max 4  rows per identifier and quarter: max 8",
    "budget per query: epsilon=0.25 delta=2.5e-09",
    (
        'mean without truncation: A step in your query is not recognized at this time: "DF ["PIDENT", "YEAR", '
        '"QUARTER", "SEX", ...]; PROJECT */8 COLUMNS". ""If you would like to see this supported, please file an issue.'
    ),
    "mean, 10 rows per identifier: HWUSUAL Sum, Integer Laplace, scale=6,400.0 95% accuracy=19,173.2",
    "mean, 10 rows per identifier: HWUSUAL Length, Integer Laplace, scale=80.0 95% accuracy=240.2",
    'truncate_per_group(10) plan: FILTER [(0.int_range([len().cast(Int64)]).shuffle().over([col("PIDENT")])) < (10)]',
    "keep=first: scales 6,400.0, 80.0",
    "keep=last: scales 6,400.0, 80.0",
    "keep=SortBy(ILOSTAT): scales 6,400.0, 80.0",
    "rows with HWUSUAL not 99: 1,496,192 by 809,710 identifiers",
    "ILOSTAT values among them: [1]",
    "identifiers with more than 10 such rows: 18  rows dropped: 20",
    "mean error bar at the true size: 0.0189",
    "unit of 36 contributions, no identifier: scales 23,040.0, 288.0",
    (
        "per quarter without truncate_num_groups: num_groups (None), total contributions (None), and per_group "
        "(Some(1)) are not sufficiently well-defined. This is likely due to a missing truncation earlier in the data "
        "pipeline."
    ),
    "per quarter, 10 quarters per identifier: len Frame Length, Integer Laplace, scale=80.0 95% accuracy=240.2 threshold=1,714",
    "per quarter, 10 quarters per identifier: HWUSUAL Sum, Integer Laplace, scale=6,400.0 95% accuracy=19,173.2",
    (
        'truncate_num_groups(10, keep=first) plan: FILTER [(col("QUARTER").as_struct([col("YEAR")]).rank()'
        '.over([col("PIDENT")])) < (10)]'
    ),
    "identifier-quarters after the filter: 1,241,896  identifiers with 4 quarters: 13,948",
    "truncate_num_groups(4, keep=first): keeps at most 3 quarters, 1,227,948 identifier-quarters; scale 32.0 threshold 657",
    "truncate_num_groups(5, keep=first): keeps at most 4 quarters, 1,241,896 identifier-quarters; scale 40.0 threshold 830",
    (
        "a filter between the two truncations: Expr is not recognized at this time: dyn int: 0.int_range([len()])"
        '.shuffle().over([col("PIDENT"), col("QUARTER"), col("YEAR")]). If you would like to see this supported, '
        "please file an issue."
    ),
    "mean of one row per identifier: HWUSUAL Sum, Integer Laplace, scale=640.0 95% accuracy=1,917.8",
    "mean of one row per identifier: HWUSUAL Length, Integer Laplace, scale=8.0 95% accuracy=24.5",
    "true mean of one row per identifier: 37.0970  (before the cast to int: 37.2359; all rows: 37.6393)",
    "per-identifier mean error bar at the true size: 0.0035",
    "keep=sample: summarize scales 80.0, 6,400.0; collect: multiple fields with name 'QUARTER' found",
    "shares spent after the failed collect: 4 of 4",
    "two identifiers, one per quarter, 5 quarters: len Frame Length, Integer Laplace, scale=80.0 95% accuracy=240.2 threshold=1,714",
    "two identifiers, one per quarter, 5 quarters: HWUSUAL Sum, Integer Laplace, scale=6,400.0 95% accuracy=19,173.2",
    "one identifier as a bound, 5 quarters: scales 40.0, 3,200.0",
    "num_groups=10 promised in the unit: len Frame Length, Integer Laplace, scale=80.0 95% accuracy=240.2 threshold=1,714",
    "num_groups=3 promised in the unit: len Frame Length, Integer Laplace, scale=24.0 95% accuracy=72.4 threshold=486",
    "num_groups=3 promised in the unit: HWUSUAL Sum, Integer Laplace, scale=1,920.0 95% accuracy=5,752.3",
    "identifiers that break the promise of 3 quarters: 13,948",
    "true mean, 10 rows per identifier: 37.6393  (1,496,172 rows)",
    "true per quarter (first row per identifier and quarter): 36 quarters, rows from 27,228 to 41,868",
    "true 2005Q1: rows 29,561  hours 1,110,610",
    "true 2005Q2: rows 29,406  hours 1,106,170",
    "true 2005Q3: rows 28,364  hours 1,070,826",
    "true 2005Q4: rows 28,681  hours 1,080,009",
    "true 2013Q4: rows 36,964  hours 1,374,675",
]

NUMBER = r"(-?[\d,]+(?:\.\d+)?)"
LEN_ACCURACY = 240.2
SUM_ACCURACY = 19_173.2
PROMISE_LEN_ACCURACY = 72.4


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
def test_released_means_near_truth(out: str) -> None:
    (mean,) = number(rf"^  mean, 10 rows per identifier: {NUMBER}  \(true 37\.6393, within 0\.0189", out)
    assert abs(mean - 37.6393) <= 3 * 0.0189 + 0.0002  # the bar, plus the rounding of the printed values
    (person,) = number(rf"^  mean of one row per identifier: {NUMBER}  \(true 37\.0970, within 0\.0035", out)
    assert abs(person - 37.0970) <= 3 * 0.0035 + 0.0002


@pytest.mark.libs
def test_released_quarters_near_truth(out: str) -> None:
    truth = re.findall(rf"^true (\d{{4}}Q\d): rows {NUMBER}  hours {NUMBER}$", out, re.MULTILINE)
    assert len(truth) == 36
    released = {
        q: (rows, hours)
        for q, rows, hours in re.findall(rf"^  (\d{{4}}Q\d): rows {NUMBER}  hours {NUMBER}$", out, re.MULTILINE)
    }
    assert len(released) == 36
    for q, rows, hours in truth:
        rows_hat, hours_hat = (float(v.replace(",", "")) for v in released[q])
        assert abs(rows_hat - float(rows.replace(",", ""))) <= 4 * LEN_ACCURACY, q
        assert abs(hours_hat - float(hours.replace(",", ""))) <= 4 * SUM_ACCURACY, q
    (released_quarters,) = number(r"^  per quarter: (\d+) quarters released;", out)
    assert released_quarters == 36
    # The other two grouped releases: the script counts the quarters within four times the 95% accuracy itself.
    assert "  two bounds, 5 quarters: 36 quarters released; rows within four times their 95% accuracy: 36 of 36" in out
    assert "  num_groups=3 promised: 36 quarters released; rows within four times their 95% accuracy: 36 of 36" in out
    assert PROMISE_LEN_ACCURACY < LEN_ACCURACY  # a promise of three quarters buys a smaller bar than ten
