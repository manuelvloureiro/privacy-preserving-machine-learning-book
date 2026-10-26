"""code/part04-extending/opendp_plugins.py runs and prints what the OpenDP book's chapter on plugins quotes.

The deterministic lines (maps, checks, error messages, the Theil-Sen scores and candidate ranges, the grouping
scores and the candidates within the selection bound) are checked exactly. OpenDP's noise cannot be seeded, so the
released values are checked against the bounds the page states: the two Laplace releases within three times their
95% accuracy, the Theil-Sen medians within the printed candidate ranges (each holds with probability 1 - 1e-4),
and the released grouping columns among the candidates whose score is within 910 of the best (same probability).
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part04-extending" / "opendp_plugins.py"

DETERMINISTIC = [
    "opendp version: 0.16.0",
    'without the flag: Attempted to use function that requires "contrib", "honest-but-curious", but not enabled',
    "constant: input=AtomDomain(T=i32) metric=AbsoluteDistance(i32) measure=MaxDivergence",
    "sum >> constant: map(1)=0.0 release=denied",
    "python laplace scale 2: map(1.0)=0.5 built-in map(1.0)=0.5",
    "halved map, same noise: map(1.0)=0.25 check(1.0, 0.25)=True",
    "laplace scale 2: 95% accuracy=5.9915",
    "repeat 2: [1, 2] -> [1, 2, 1, 2] map(1)=2 map(3)=6",
    "cast >> repeat 2 >> clamp (1, 2) >> sum: map(1)=4 on ['0', '1', '2', '3'] = 12",
    "... >> laplace scale 1: map(1)=4.0",
    "domain mismatch: DomainMismatch(\"Intermediate domains don't match",
    "anything_constant: map(36)=0.0 release=denied",
    "budget spent by the plugin query: [0.5]",
    "clamp >> sum >> int_constant: release=denied",
    "register twice: 'anything_constant' is already registered in the Context API. Please choose a different name.",
    "register without make_: constructor.__name__ must start with 'make_', found select_cols",
    "theil-sen: x cuts -1.5 and 1.5, 100 candidates from -10.0 to 10.0 step 0.2020",
    "pairwise predict: map(1)=1 medians: map(1)=2.0 theil-sen: map(1)=2.0 map(2)=4.0",
    "toy data: n=100 least-squares slope=2.0573 intercept=1.0376",
    "pairs: 50 medians at the cuts -2.2424 and 4.2857",
    "selection bound: 26.2 score units at noise scale 2.0",
    "cut -1.5: best candidate -2.3232 score 2; within the bound: -2.9293 to -1.5152 (8 candidates)",
    "cut 1.5: best candidate 4.3434 score 8; within the bound: 3.3333 to 6.3636 (16 candidates)",
    "LFS: 3,811,932 rows; candidate columns SEX, ILOSTAT, YEAR, QUARTER, AGE, PIDENT; 63 candidate sets",
    "distinct values: SEX 2 ILOSTAT 4 YEAR 9 QUARTER 4 AGE 7 PIDENT 1,682,306",
    "score: groups of at least 100 rows",
    "rank 1: SEX, ILOSTAT, YEAR, QUARTER, AGE score 996 of 1,105 groups",
    "rank 2: ILOSTAT, YEAR, QUARTER, AGE score 504 of 570 groups",
    "rank 3: SEX, YEAR, QUARTER, AGE score 432 of 470 groups",
    "sets with PIDENT: 32, largest score 0",
    "select scale 72: map(36)=1.0 monotonic would be 0.5",
    "context query: param scale=72.0",
    "within 910 of the best: 10 candidates",
    "  candidate: SEX, ILOSTAT, YEAR, QUARTER, AGE score 996",
    "  candidate: SEX, YEAR, AGE score 108",
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
def test_released_values_within_their_bounds(out: str) -> None:
    (laplace,) = number(rf"^  python laplace of 10\.0: {NUMBER}  \(within 6\.0 with probability 0\.95\)$", out)
    assert abs(laplace - 10.0) <= 3 * 5.9915
    (repeated,) = number(
        rf"^  repeated sum of \['0', '1', '2', '3'\] with laplace scale 1: {NUMBER}  \(true 12\)$", out
    )
    assert abs(repeated - 12) <= 3 * 2.9957  # the 95% accuracy of Laplace noise at scale 1
    slope, intercept, low_cut, high_cut = number(
        rf"^  theil-sen: slope {NUMBER} intercept {NUMBER}  \(medians at the cuts {NUMBER} and {NUMBER}\)$", out
    )
    assert abs(low_cut - (slope * -1.5 + intercept)) < 1e-3 and abs(high_cut - (slope * 1.5 + intercept)) < 1e-3
    for cut, median in ((-1.5, low_cut), (1.5, high_cut)):
        low, high = number(rf"^cut {cut}: .*; within the bound: {NUMBER} to {NUMBER} \(\d+ candidates\)$", out)
        assert low - 1e-4 <= median <= high + 1e-4, f"median at {cut} outside its candidate range"
    candidates = re.findall(r"^  candidate: (.+) score \d+$", out, re.MULTILINE)
    assert len(candidates) == 10
    m = re.search(r"^  grouping columns: (.+)$", out, re.MULTILINE)
    assert m and m.group(1) in candidates, "released grouping columns are not within the bound of the best"
