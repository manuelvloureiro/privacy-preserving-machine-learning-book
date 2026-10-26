"""code/part04-extending/opendp_statistical_modelling.py runs and prints what the OpenDP book's chapter
"Statistical modelling" quotes.

The deterministic lines (privacy maps, spaces, bounds, cut points, grid steps, the exact least-squares fits and
median lines, the exact decompositions, the feature-flag errors) are checked exactly. OpenDP's private medians and
the PCA's noise cannot be seeded, so the released fits are checked within the bounds the page states: the toy slope
within 0.3 of least squares and its intercept within 0.5 (about three grid steps at a cut point); the survey slope
within 0.1 of the median line and its intercept within 4 hours (two grid steps at the lower cut point); the PCA at
row norm 1 within 1 of each exact singular value and a cosine above 0.9 with each exact component; at row norm 40
only the first singular value (within 30) and the first cosine (above 0.9); the two-component fit like the first.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part04-extending" / "opendp_statistical_modelling.py"

DETERMINISTIC = [
    "opendp version: 0.16.0",
    "regression space: NPArray2Domain(num_columns=2, nan=True, T=f64) SymmetricDistance() MaxDivergence",
    "regression map: scale=1 runs=1 -> epsilon=2.0",
    "regression map: scale=2 runs=1 -> epsilon=1.0",
    "regression map: scale=2 runs=3 -> epsilon=3.0",
    "toy: n=100 cut points=[-1.5, 1.5] candidate step=0.2020",
    "toy least squares: slope=2.0275 intercept=0.9725",
    "toy median line, no noise: slope=2.0379 intercept=1.0786 readings=50",
    "survey: working people in 2013 Q4 with hours recorded: 36,914 (one row each)",
    "survey: of them with 0 actual hours that week: 4,109",
    "survey: bounds=(0.0, 98.0) cut points=[24.5, 73.5] candidate step=0.9899",
    "survey least squares: slope=0.8582 intercept=-0.1935",
    "survey median line, no noise: slope=1.0000 intercept=0.0000 medians=[24.50, 73.50]",
    "survey readings: 16,463 pairs; within one candidate step of the medians: 5,606 and 5,252",
    "survey private fit: scale=2 epsilon=1.0",
    "pca table: 10,000 rows x 4 columns, row length mean=8.53 max=36.56",
    "pca table: rows longer than 1: 99.9%  longer than 40: 0.0%",
    "pca exact, raw rows: singular values=[901.9, 278.6, 222.2, 156.9]",
    "pca exact, rows clamped to 1: singular values=[80.48, 40.67, 34.29, 26.29]",
    "pca exact, rows clamped to 1: first component=[-0.614, -0.534, -0.526, -0.247]",
    "pca model: PCA(epsilon=1.0, n_components=4, n_features=4, n_samples=10000, row_norm=1.0)",
    "pca space: NPArray2Domain(size=10000, num_columns=4, nan=True, T=f64) SymmetricDistance() MaxDivergence",
    "pca map: map(2)=2.0 map(1)=2.0 check(2, 1.0)=False check(2, 2.0)=True",
    "make_private_pca unit_epsilon=1: map(2)=1.0 map(1)=1.0",
    "pca n_components=2: map(2)=2.0",
    "pca composition of epsilon 1 and 0.5: map(2)=3.0",
    "bounded domain: NPArray2Domain(origin=[0. 0. 0. 0.], norm=1.0, p=2, size=10000, num_columns=4, nan=True, T=f64)",
    "pca on the bounded domain: map(1)=0.50 map(2)=1.00 map(4)=2.00",
    'regression without honest-but-curious: Attempted to use function that requires "contrib", "honest-but-curious", but not enabled',
    'pca without honest-but-curious: Attempted to use function that requires "contrib", "honest-but-curious", but not enabled',
    'pca without idealized-numerics: Attempted to use function that requires "contrib", "idealized-numerics", but not enabled',
    "regression without idealized-numerics: constructed",
    "pca with n_samples one short of the table: fitted without complaint",
]

NUMBER = r"(-?[\d,]+(?:\.\d+)?)"
LIST = r"\[([^\]]*)\]"

EXACT_CLAMPED = [80.48, 40.67, 34.29, 26.29]
EXACT_RAW_FIRST = 901.9


def numbers(pattern: str, out: str) -> list[float]:
    m = re.search(pattern, out, re.MULTILINE)
    assert m, f"no line matches {pattern!r}"
    return [float(g.replace(",", "")) for g in m.groups()]


def listed(pattern: str, out: str) -> list[float]:
    """The numbers of the first bracketed list captured by the pattern."""
    m = re.search(pattern, out, re.MULTILINE)
    assert m, f"no line matches {pattern!r}"
    return [float(v) for v in m.group(1).split(",")]


@pytest.fixture(scope="module")
def out() -> str:
    return subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout


@pytest.mark.libs
def test_deterministic_lines(out: str) -> None:
    lines = out.splitlines()
    for expected in DETERMINISTIC:
        assert expected in lines, f"missing line: {expected}"


@pytest.mark.libs
def test_runs_clean(out: str) -> None:
    result = subprocess.run([sys.executable, "-W", "error", str(SCRIPT)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.libs
def test_released_fits_near_their_targets(out: str) -> None:
    slope, intercept = numbers(rf"^  toy private fit: slope={NUMBER} intercept={NUMBER}$", out)
    assert abs(slope - 2.0275) <= 0.3  # about three grid steps (0.2 each, cut points 3 apart) at a cut point
    assert abs(intercept - 0.9725) <= 0.5
    slope, intercept = numbers(rf"^  survey private fit: slope={NUMBER} intercept={NUMBER}$", out)
    assert abs(slope - 1.0) <= 0.1  # the median line; two grid steps of 0.99 over cut points 49 apart
    assert abs(intercept - 0.0) <= 4.0  # two grid steps at the lower cut point move it by about 3 hours
    (predicted,) = numbers(rf"^  survey private fit at 35 usual hours: {NUMBER} actual$", out)
    assert abs(predicted - (slope * 35 + intercept)) <= 0.01


@pytest.mark.libs
def test_released_pca_near_exact(out: str) -> None:
    values = listed(rf"^  pca row_norm=1: singular values={LIST}$", out)
    assert all(abs(v - e) <= 1.0 for v, e in zip(values, EXACT_CLAMPED, strict=True))
    cos = listed(rf"^  pca row_norm=1: \|cos\| with exact components={LIST}$", out)
    assert len(cos) == 4 and min(cos) >= 0.9
    mean = listed(rf"^  pca row_norm=1: mean={LIST}  explained variance ratio=", out)
    assert max(abs(m) for m in mean) <= 0.05  # the true mean is zero; the private one spends a third of epsilon = 2
    loose = listed(rf"^  pca row_norm=40: singular values={LIST} \(exact", out)
    assert abs(loose[0] - EXACT_RAW_FIRST) <= 30.0  # Laplace noise on an eigenvalue near 813,000 at scale 4,800
    loose_cos = listed(rf"^  pca row_norm=40: \|cos\| with exact components={LIST}$", out)
    assert loose_cos[0] >= 0.9  # the first direction survives a loose norm; the others are not checked
    two = listed(rf"^  pca n_components=2: returned the model: True  singular values={LIST}$", out)
    assert all(abs(v - e) <= 1.0 for v, e in zip(two, EXACT_CLAMPED[:2], strict=True))
    two_cos = listed(rf"^  pca n_components=2: \|cos\| with exact components={LIST}$", out)
    assert len(two_cos) == 2 and min(two_cos) >= 0.9
