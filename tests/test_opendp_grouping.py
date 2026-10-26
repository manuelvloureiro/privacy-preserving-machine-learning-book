"""code/part02-tabular-data/opendp_grouping.py runs and prints what the OpenDP book's chapter "Grouping" quotes.

The deterministic lines (true counts, scales, accuracies, thresholds, tail probabilities, error messages) are
checked exactly. OpenDP's noise cannot be seeded, so the released values are checked against the true values: the
stable-keys release has exactly the 14 groups whose true count exceeds the threshold (the nearest groups on either
side are thousands of rows away, far beyond the accuracy), every released group exceeds the threshold and has a true
count above it, each named count is within three times its 95% accuracy, the largest error in a table of up to 54
counts is within four times the accuracy (so that a correct run fails less than once in a thousand), and each mean is
within three times the accuracy the page states for it.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part02-tabular-data" / "opendp_grouping.py"

ACCURACY = 431.884579  # the 95% accuracy of every count at scale 144
THRESHOLD = 2773

DETERMINISTIC = [
    "rows: 3,811,932  people: 1,682,306",
    "groups by AGE and ILOSTAT: 16",
    "smallest groups: AGE=null ILOSTAT=1 rows=255; AGE=75 ILOSTAT=1 rows=847; AGE=65 ILOSTAT=2 rows=13,157",
    "largest group: AGE=7 ILOSTAT=9 rows=688,977",
    (
        'without delta: The key-set of {col("AGE"), col("ILOSTAT")} is private and cannot be released without a '
        "filter or join. Please pass a filtering threshold into make_private_lazyframe or conduct a join against a "
        "public key-set."
    ),
    "stable keys: privacy loss (Approximate(MaxDivergence), (0.25, 1e-07))",
    "stable keys len: Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9 threshold=2773",
    "one person's 36 rows clear 2773 with probability 2.77e-09; over 36 groups 9.98e-08 <= delta 1e-07",
    "at threshold 2772: over 36 groups 1.01e-07 > delta",
    "groups with true count above the threshold: 14 of 16",
    "threshold with delta=1e-05: scale=144.00 threshold=2110",
    "threshold with epsilon=1: scale=36.00 threshold=720",
    "explicit keys: 54 keys, 36 with rows in the data, 18 without",
    "explicit keys len: Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9",
    "right join len: Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9",
    "inner join: only left or right joins can be used to release key-sets",
    "invariant keys len: Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9",
    "true quarterly counts: 36 quarters, smallest 82,504 (2006 Q3), largest 131,656 (2011 Q4)",
    "error bar as a share of the smallest quarter: 0.52%, of the largest: 0.33%",
    "rows with HWUSUAL < 99: 1,496,192",
    "invariant lengths HWUSUAL: Sum, Integer Laplace, scale=1,764.00 95% accuracy=5,285.0",
    "invariant lengths HWUSUAL: Length, Integer Laplace, scale=0.00 95% accuracy=none",
    "true lengths: SEX=1 779,714  SEX=2 716,478",
    "true mean hours: SEX=1 40.778  SEX=2 34.224",
    "mean 95% accuracy: SEX=1 0.0068  SEX=2 0.0074",
    (
        'filter inside the query: The key-set of {col("SEX")} is private and cannot be released without a filter or '
        "join. Please pass a filtering threshold into make_private_lazyframe or conduct a join against a public "
        "key-set."
    ),
    "max_groups=16: threshold=2656",
    "sum per SEX without max_length: must specify 'max_length' in a margin with by=[col(\"SEX\")]",
    "ungrouped count len: Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9",
    (
        "error bar as a share of the true count: all rows 0.011%, largest group 0.063%, "
        "smallest group above the threshold 3.3%"
    ),
    "smallest true count above the threshold: 13,157  largest below it: 847",
    "test bound for a table: 4 x accuracy = 1,727.5",
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
def test_stable_keys_release(out: str) -> None:
    groups, smallest_released, smallest_true = number(
        rf"^  stable groups released: {NUMBER}  smallest released count: {NUMBER}  "
        rf"smallest true count among them: {NUMBER}$",
        out,
    )
    assert groups == 14  # 13,157 rows above the threshold and 847 below: no group is within reach of 2773
    assert smallest_released > THRESHOLD  # the library keeps a group only when its noisy count exceeds it
    assert smallest_true >= THRESHOLD - 3 * ACCURACY  # a released group's true count is not far below the threshold
    for age, status, true in ((47, 1, 652_801), (65, 2, 13_157)):
        (released,) = number(rf"^  stable AGE={age} ILOSTAT={status}: {NUMBER}  \(true {true:,}\)$", out)
        assert abs(released - true) <= 3 * ACCURACY, (age, status)
    (largest,) = number(rf"^  stable largest \|released - true\|: {NUMBER}$", out)
    assert largest <= 4 * ACCURACY


@pytest.mark.libs
def test_explicit_and_invariant_keys_release(out: str) -> None:
    (empty,) = number(rf"^  explicit YEAR=2005 ILOSTAT=4: {NUMBER}  \(true 0\)$", out)
    assert abs(empty) <= 3 * ACCURACY  # a key without rows is noise around zero
    (released,) = number(rf"^  explicit YEAR=2013 ILOSTAT=1: {NUMBER}  \(true 188,349\)$", out)
    assert abs(released - 188_349) <= 3 * ACCURACY
    keys, largest = number(rf"^  explicit keys released: {NUMBER}  largest \|released - true\|: {NUMBER}$", out)
    assert keys == 54 and largest <= 4 * ACCURACY
    (quarter,) = number(rf"^  quarter 2005 Q1: {NUMBER}  \(true 87,372\)$", out)
    assert abs(quarter - 87_372) <= 3 * ACCURACY
    quarters, inside, total, largest = number(
        rf"^  quarters released: {NUMBER}  within their 95% accuracy: {NUMBER} of {NUMBER}  "
        rf"largest \|released - true\|: {NUMBER}$",
        out,
    )
    assert quarters == total == 36 and inside >= 28 and largest <= 4 * ACCURACY


@pytest.mark.libs
def test_invariant_lengths_release(out: str) -> None:
    men, women = number(rf"^  mean hours: SEX=1 {NUMBER} \(true 40\.778\)  SEX=2 {NUMBER} \(true 34\.224\)$", out)
    assert abs(men - 40.778) <= 3 * 0.0068 + 0.001  # the sum's accuracy over the public length, plus rounding
    assert abs(women - 34.224) <= 3 * 0.0074 + 0.001
