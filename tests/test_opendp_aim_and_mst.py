"""code/part05-synthetic-data/opendp_aim_and_mst.py runs and prints what the OpenDP book's chapter on AIM and MST quotes.

The deterministic lines (the facts about the data, the unit and the budget, the one-way scale and threshold, the budget
arithmetic of each algorithm, the conversion of rho to epsilon, the statement, the sizes and types of the synthetic
frames) are checked exactly. OpenDP's noise cannot be seeded, so what a release contains is checked structurally: AIM
measures marginals of two or three of the four columns at the scale its rounds allow or finer, MST measures three pairs
that form a spanning tree, Sequential measures the two guaranteed marginals. The synthetic data's marginals are checked
against the real data's with the bound the page states: the total variation distance of a marginal with k cells stays
under 3 k std / (2 n), three standard deviations in every cell at once, where std is the table's own estimate. The two
releases that follow the documentation and leave HWUSUAL's nulls in place must break that bound on HWUSUAL, which is
what the page's null section shows; the release that fills the nulls must meet it on every marginal it measured.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part05-synthetic-data" / "opendp_aim_and_mst.py"

COLUMNS = ("SEX", "AGE", "HWUSUAL", "ILOSTAT")
REAL_ROWS = 3_811_932

DETERMINISTIC = [
    "opendp version: 0.16.0",
    "rows: 3,811,932  AGE nulls: 255  HWUSUAL nulls: 20,819",
    "HWUSUAL = 99 (not applicable): 2,294,921",
    "privacy unit: (SymmetricDistance(), 36)",
    "privacy loss: (Approximate(ZeroConcentratedDivergence), (0.3, 3e-07))",
    "query budget: rho=0.1 delta=1e-07",
    "aim: oneway scale=227.6840 threshold=1364",
    "aim: rho per one-way marginal=0.0125  scale 36/sqrt(2 rho)=227.6840",
    "aim: rounds=10  rho per round=0.0050  measure=0.0045  select=0.0005  scale of a round's marginal=379.4733",
    "aim: rounds by default=64  max_size=80 MB",
    "aim keys ILOSTAT: [1, 2, 3, 9, None]  thresholds: {'ILOSTAT': 1364}",
    "mst: oneway scale=227.6840 threshold=1364",
    "mst: pairs=3  rho per pair=0.0150  scale of a pair=207.8461",
    "mst pairs form a spanning tree: True",
    "sequential: rho per guaranteed marginal=0.0125  scale=227.6840",
    "sequential: rho per AIM round=0.0025  scale of a round's marginal=536.6563",
    "sequential guaranteed marginals present: True",
    "synthetic dtypes: {'SEX': Int64, 'AGE': Float64, 'HWUSUAL': Float64, 'ILOSTAT': Int64}",
    "synthesize rows=100,000 method=sample: shape (100000, 4)",
    "budget spent: (0.3, 3e-07)",
    "rho=0.3 at delta=1e-07: epsilon=4.2709",
    "whole context converted: epsilon=4.2709 delta=4e-07",
    (
        "This release of three contingency tables and the synthetic rows drawn from them satisfies "
        "(rho = 0.3, delta = 3e-07)-zero-concentrated differential privacy, which is "
        "(epsilon = 4.27, delta = 4e-07)-differential privacy, for adding or removing all of one person's survey "
        "responses, at most 36 of them, under the central model, using OpenDP 0.16.0 with Gaussian noise. "
        "The keys of SEX and the cuts of AGE and HWUSUAL were fixed in advance; the keys of ILOSTAT were released "
        "with a threshold of 1364. Generating the rows is post-processing and spent nothing."
    ),
    "identifier unit, 4 rows per person: oneway scale=25.2982 threshold=142",
    "identifier unit: scale ratio 9.00 = 36/4",
]

NUMBER = r"(-?[\d,]+(?:\.\d+)?)"


def numbers(pattern: str, out: str) -> list[float]:
    m = re.search(pattern, out, re.MULTILINE)
    assert m, f"no line matches {pattern!r}"
    return [float(g.replace(",", "")) for g in m.groups()]


def cliques(name: str, out: str) -> list[tuple[tuple[str, ...], float]]:
    """The measured multi-way marginals of a release, with their noise standard deviations, in release order."""
    found: list[tuple[tuple[str, ...], float]] = []
    for clique, std in re.findall(rf"^{name} measured: \(([^)]*)\) std=([\d.]+)$", out, re.MULTILINE):
        columns = tuple(c.strip().strip("'") for c in clique.split(","))
        found.append((columns, float(std)))
    assert found, f"no measured marginals printed for {name}"
    return found


def distances(name: str, kind: str, out: str) -> dict[str, tuple[float, float]]:
    """The marginal label mapped to (total variation distance, bound) for one release and kind of line."""
    pattern = rf"^{kind} (.+?) \({name}\): tv=([\d.]+) largest cell error=[\d,]+ rows  std=[\d.]+ bound=([\d.]+)$"
    found = {label: (float(tv), float(bound)) for label, tv, bound in re.findall(pattern, out, re.MULTILINE)}
    assert found, f"no {kind} lines printed for {name}"
    return found


@pytest.fixture(scope="module")
def out() -> str:
    return subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout


@pytest.mark.libs
def test_deterministic_lines(out: str) -> None:
    lines = out.splitlines()
    for expected in DETERMINISTIC:
        assert expected in lines, f"missing line: {expected}"


@pytest.mark.libs
def test_aim_measures_marginals_of_the_four_columns(out: str) -> None:
    for columns, std in cliques("aim", out):
        assert 2 <= len(columns) <= 3 and set(columns) <= set(COLUMNS), columns
        assert std <= 379.48  # a round's scale, or finer after the budget per round is raised
    (seconds,) = numbers(rf"^aim release \(one run\): {NUMBER} s$", out)
    assert seconds > 0


@pytest.mark.libs
def test_mst_measures_a_spanning_tree_of_pairs(out: str) -> None:
    pairs = cliques("mst", out)
    assert len(pairs) == 3
    assert all(len(columns) == 2 and abs(std - 207.85) < 0.01 for columns, std in pairs), pairs
    assert set().union(*(set(columns) for columns, _ in pairs)) == set(COLUMNS)


@pytest.mark.libs
def test_sequential_measures_the_guaranteed_marginals_first(out: str) -> None:
    measured = cliques("sequential", out)
    assert [columns for columns, _ in measured[:2]] == [("SEX", "ILOSTAT"), ("AGE", "ILOSTAT")]
    assert all(abs(std - 227.68) < 0.01 for _, std in measured[:2])
    assert all(std <= 536.66 for _, std in measured[2:])


@pytest.mark.libs
def test_model_totals(out: str) -> None:
    # The nulls of HWUSUAL are dropped by the one-way step, so the two documentation-style releases estimate a total
    # thousands of rows short; the release that fills them estimates it within a few hundred.
    for name in ("aim", "mst"):
        (total,) = numbers(rf"^{name} model total \(one run\): {NUMBER}  real rows 3,811,932$", out)
        assert total < REAL_ROWS - 2_000, (name, total)
    (total,) = numbers(rf"^sequential model total \(one run\): {NUMBER}  real rows 3,811,932$", out)
    assert abs(total - REAL_ROWS) <= 2_000, total


@pytest.mark.libs
def test_synthetic_frame(out: str) -> None:
    rows, cols, total = numbers(
        rf"^synthesize \(aim, one run\): shape \({NUMBER}, {NUMBER}\)  model total {NUMBER}  ", out
    )
    # synthesize() truncates the model's total to a whole number of rows (mbi's int(self.total)), and the total is
    # printed with one decimal, so the printed total exceeds the row count by at most about one.
    assert cols == 4 and -0.1 <= total - rows <= 1.1
    assert REAL_ROWS - 20_000 <= rows <= REAL_ROWS + 2_000
    age_min, age_max, hw_min, hw_max = numbers(
        rf"^synthetic AGE from {NUMBER} to {NUMBER}  HWUSUAL from {NUMBER} to {NUMBER}$", out
    )
    # Values are drawn uniformly inside each bin, and the open-ended bins reach one unit past their edge.
    assert 19.0 <= age_min <= 19.01 and 60.99 <= age_max <= 61.0
    assert 0.0 <= hw_min <= 0.01 and 40.99 <= hw_max <= 41.0
    m = re.search(r"^synthetic SEX values: \[(.*)\]  ILOSTAT values: \[(.*)\]$", out, re.MULTILINE)
    assert m
    sex = {v.strip() for v in m.group(1).split(",")}
    ilostat = {v.strip() for v in m.group(2).split(",")}
    assert sex <= {"1", "2", "None"} and {"1", "2"} <= sex
    assert ilostat <= {"1", "2", "3", "9", "None"} and {"1", "2", "3", "9"} <= ilostat


@pytest.mark.libs
def test_marginals_against_the_real_data(out: str) -> None:
    for name in ("aim", "mst"):
        oneway = distances(name, "one-way", out)
        assert set(oneway) == set(COLUMNS)
        for column in ("SEX", "AGE"):
            tv, bound = oneway[column]
            assert tv < bound, (name, column, tv, bound)
        tv, bound = oneway["HWUSUAL"]
        assert tv > bound, (name, tv, bound)  # the 20,819 nulls, dropped by one step and kept by the other
    oneway = distances("sequential", "one-way", out)
    assert set(oneway) == set(COLUMNS)
    for column, (tv, bound) in oneway.items():
        assert tv < bound, (column, tv, bound)
    measured = distances("sequential", "measured", out)
    assert {"(SEX, ILOSTAT)", "(AGE, ILOSTAT)"} <= set(measured)
    for label, (tv, bound) in measured.items():
        assert tv < bound, (label, tv, bound)
    # The fitted HWUSUAL counts of the documentation-style release: the last bin, which should also hold the nulls,
    # comes out thousands of rows short.
    (difference,) = numbers(rf"^  \(40\.0, inf\]: fitted [\d,]+  real 2,658,551  difference {NUMBER}$", out)
    assert difference < -5_000
