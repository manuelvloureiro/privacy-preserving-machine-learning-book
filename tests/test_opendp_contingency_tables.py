"""code/part05-synthetic-data/opendp_contingency_tables.py runs and prints what the OpenDP book's chapter
"Contingency tables" quotes.

The deterministic lines (the true counts, the budgets, the two refusals, the one-way scale and threshold, the keys,
cuts, thresholds and schema, the standard deviations and accuracies) are checked exactly. OpenDP's noise cannot be
seeded, so the released projections are checked against the true values within three times their 95% accuracy:
the largest error over the 16 cells of the AGE x HWUSUAL projection against the per-cell bar, the uniform
projection of SEX before its marginal is measured (three equal values whose sum is the model's total), the SEX
cells after the second query, and the ILOSTAT and SEX keys once the released marginals are refitted with 10,000 steps of
mirror descent. The release itself fits with mbi's default of 1,000 steps, and a key with no rows is the slowest
cell to fit, because the model can empty it only in the limit; ten times the steps leave a tenth of its rows or fewer.
On this data the default fit leaves about a thousand rows in the empty null key of ILOSTAT and several hundred in the
null key of SEX, taken from the real keys. So, as the page states, each released null is positive, below a tenth of
a percent of the rows and above what the refit leaves; each real key is held within three times its accuracy plus
the rows left in null; and the refit brings every key of ILOSTAT and SEX, null included, within three times its
accuracy.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part05-synthetic-data" / "opendp_contingency_tables.py"

ILOSTAT_ACCURACY = 477.06  # a one-way projection read off a two-way marginal at scale 121.70: 2 x 121.70 x 1.96
TWO_WAY_ACCURACY = 238.53  # one cell of a measured two-way marginal
SEX_ACCURACY = 223.13  # the second query's marginal at scale 113.84
ROWS = 3_811_932
TRUE_ILOSTAT = {"1": 1_517_011, "2": 155_661, "3": 1_450_283, "9": 688_977, "null": 0}
TRUE_SEX = {"1": 1_828_992, "2": 1_982_940, "null": 0}

DETERMINISTIC = [
    "opendp version: 0.16.0  mbi version: 1.1.0",
    "distinct values: SEX=2  AGE=7  HWUSUAL=83  ILOSTAT=4",
    "cells in the full table of the four raw columns: 4,648",
    "true ILOSTAT counts: 1=1,517,011  2=155,661  3=1,450,283  9=688,977",
    "ILOSTAT codes in the codebook but not in the data: [4, 5]",
    "rows with AGE null: 255  with HWUSUAL null: 20,819",
    "privacy unit: (SymmetricDistance(), 36)",
    "privacy loss: (Approximate(ZeroConcentratedDivergence), (0.2, 1e-07))",
    "delta with keys for every column: delta (1e-07) must be zero because keys and cuts span all columns",
    "no delta with ILOSTAT unkeyed: delta (0.0) must be nonzero because keys and cuts don't span all columns",
    "oneway split: 0.125 of rho=0.1 -> rho=0.0125 one-way, rho=0.04375 per count",
    "oneway scale: 227.68  oneway threshold: 1364",
    "scale from the formula 36 / sqrt(2 x 0.0125): 227.68",
    "scale of each count of the workload, 36 / sqrt(2 x 0.04375): 121.70",
    (
        "keys: SEX=[1, 2, null]  AGE=[(-inf, 20.0], (20.0, 40.0], (40.0, 60.0], (60.0, inf]]  "
        "HWUSUAL=[(-inf, 1.0], (1.0, 20.0], (20.0, 40.0], (40.0, inf]]  ILOSTAT=[1, 2, 3, 9, null]"
    ),
    "cuts: AGE=[20.0, 40.0, 60.0]  HWUSUAL=[1.0, 20.0, 40.0]",
    "thresholds: {'ILOSTAT': 1364}",
    "schema: {'SEX': Int64, 'AGE': String, 'HWUSUAL': String, 'ILOSTAT': Int64}",
    "std ILOSTAT: 243.40  95% accuracy: 477.06",
    "std AGE x HWUSUAL: 121.70  95% accuracy: 238.53",
    "std HWUSUAL: 181.42  95% accuracy: 355.58",
    "std SEX: attrs (('SEX',)) are not covered by the query set",
    "std SEX after the second query: 113.84  95% accuracy: 223.13",
    "marginals in the model: [('ILOSTAT',), ('AGE', 'HWUSUAL'), ('HWUSUAL', 'ILOSTAT'), ('SEX',)]",
    "privacy loss spent: (0.15, 1e-07)",
    "  AGE (-inf, 20.0]: (-inf, 1.0]=2,733  (1.0, 20.0]=13,651  (20.0, 40.0]=95,686  (40.0, inf]=1,051,617",
    "  AGE (20.0, 40.0]: (-inf, 1.0]=1,441  (1.0, 20.0]=31,415  (20.0, 40.0]=370,616  (40.0, inf]=258,601",
    "  AGE (40.0, 60.0]: (-inf, 1.0]=1,285  (1.0, 20.0]=46,743  (20.0, 40.0]=437,820  (40.0, inf]=309,224",
    "  AGE (60.0, inf]: (-inf, 1.0]=1,502  (1.0, 20.0]=28,084  (20.0, 40.0]=122,405  (40.0, inf]=1,039,109",
    "test bounds, 3 x accuracy: 1,431.2 for ILOSTAT, 715.6 per two-way cell, 669.4 for SEX",
]

NUMBER = r"(-?[\d,]+(?:\.\d+)?)"
SIGNED = r"([+-][\d,]+)"


def number(pattern: str, out: str) -> list[float]:
    m = re.search(pattern, out, re.MULTILINE)
    assert m, f"no line matches {pattern!r}"
    return [float(g.replace(",", "")) for g in m.groups()]


def ilostat(block: str, indent: str) -> dict[str, float]:
    """The ILOSTAT counts of one block, by key, with the printed deviation checked against the true count."""
    counts: dict[str, float] = {}
    for key, true in TRUE_ILOSTAT.items():
        released, off = number(rf"^{indent}ILOSTAT {key}: {NUMBER}  \(true {true:,}, off by {SIGNED}\)$", block)
        assert off == released - true, key
        assert released >= 0, key
        counts[key] = released
    return counts


@pytest.fixture(scope="module")
def out() -> str:
    return subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def blocks(out: str) -> tuple[str, str, str]:
    """The released ILOSTAT counts of the first table, of the second, and of the dev-time refit."""
    first, rest = out.split("  ILOSTAT after the second query:")
    second, refit = rest.split("  the same marginals refitted at 10,000 steps (dev time, post-processing):")
    return first, second, refit


def sex(line: str) -> dict[str, float]:
    """The three SEX counts of one line, by key."""
    return {key: number(rf"{key}: {NUMBER}  \(true {true:,}\)", line)[0] for key, true in TRUE_SEX.items()}


def check_fit(counts: dict[str, float], truth: dict[str, int], accuracy: float, refit_null: float) -> None:
    """A released column of the default fit: null positive, small and drained by the refit; real keys short by null."""
    assert 0 < counts["null"] < ROWS / 1000  # under a tenth of a percent of the rows
    assert counts["null"] > refit_null  # ten times the steps leave fewer rows in the empty key
    for key, true in truth.items():
        assert abs(counts[key] - true) <= 3 * accuracy + counts["null"], key


@pytest.fixture(scope="module")
def refit_counts(blocks: tuple[str, str, str]) -> tuple[dict[str, float], dict[str, float]]:
    """The refitted counts of ILOSTAT and of SEX."""
    refit = blocks[2]
    line = next(x for x in refit.splitlines() if x.startswith("    SEX: "))
    return ilostat(refit, "    "), sex(line)


@pytest.mark.libs
def test_deterministic_lines(out: str) -> None:
    lines = out.splitlines()
    for expected in DETERMINISTIC:
        assert expected in lines, f"missing line: {expected}"


@pytest.mark.libs
def test_first_table_near_truth(
    out: str, blocks: tuple[str, str, str], refit_counts: tuple[dict[str, float], dict[str, float]]
) -> None:
    first, second, _ = blocks
    refit_ilostat, _ = refit_counts
    for counts in (ilostat(first, "  "), ilostat(second, "    ")):
        check_fit(
            counts, {k: v for k, v in TRUE_ILOSTAT.items() if k != "null"}, ILOSTAT_ACCURACY, refit_ilostat["null"]
        )
    (largest,) = number(rf"^  AGE x HWUSUAL largest \|released - true\|: {NUMBER}  \(accuracy 238\.53\)$", out)
    assert largest <= 3 * TWO_WAY_ACCURACY
    # A null AGE or HWUSUAL lands in the last bin of its column; placed anywhere else, this cell would be 20,819 off.
    last_bin, off = number(
        rf"^  last bin of both, the nulls' cell: {NUMBER}  \(true 1,039,109, off by {SIGNED}\)$", out
    )
    assert off == last_bin - 1_039_109
    assert abs(off) <= 3 * TWO_WAY_ACCURACY


@pytest.mark.libs
def test_refit_near_truth(
    blocks: tuple[str, str, str], refit_counts: tuple[dict[str, float], dict[str, float]]
) -> None:
    # Refitting the released marginals with 10,000 steps brings every key within its bar: the gap was the fit.
    refit_ilostat, refit_sex = refit_counts
    for key, true in TRUE_ILOSTAT.items():
        assert abs(refit_ilostat[key] - true) <= 3 * ILOSTAT_ACCURACY, key
    for key, true in TRUE_SEX.items():
        assert abs(refit_sex[key] - true) <= 3 * SEX_ACCURACY, key
    (largest,) = number(rf"^    AGE x HWUSUAL largest \|refit - true\|: {NUMBER}  \(accuracy 238\.53\)$", blocks[2])
    assert largest <= 3 * TWO_WAY_ACCURACY


@pytest.mark.libs
def test_sex_before_and_after(out: str, refit_counts: tuple[dict[str, float], dict[str, float]]) -> None:
    men, women, other = number(rf"^  SEX before the second query: {NUMBER} {NUMBER} {NUMBER}  \(uniform\)$", out)
    assert men == women == other  # a column no marginal covers is spread evenly over its keys
    assert abs(men + women + other - ROWS) <= 3 * ILOSTAT_ACCURACY  # the model's total, read off the marginals
    after = sex(out.split("  SEX after:")[1].splitlines()[0])
    check_fit(after, {k: v for k, v in TRUE_SEX.items() if k != "null"}, SEX_ACCURACY, refit_counts[1]["null"])
