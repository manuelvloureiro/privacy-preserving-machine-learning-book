"""code/part01-prerequisites/concentration.py runs, writes its figures, and reproduces the numbers in book-full/part01-prerequisites/02-concentration.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part01-prerequisites"


def test_concentration_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "concentration.py", tmp_path)
    for expected in (
        "empirical=9.62e-03",
        "Chernoff=1.35e-01",
        "bound=7.60  empirical failure=0.0486",
        "exact=1.35e-03  Chebyshev=1.11e-01  Chernoff=1.11e-02  half=5.55e-03  Mills=1.48e-03",
        "t=10] empirical=1.42e-02  bound=2.87e-01  normal approx=1.27e-02",
        "t=20] empirical=3.40e-05  bound=6.74e-03  normal approx=3.87e-06",
        "empirical=1.12e-03  Azuma=1.83e-02",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "markov_die.svg",
        "chernoff_lambda.svg",
        "bernoulli_tails.svg",
        "gaussian_tails.svg",
        "laplace_max.svg",
        "laplace_sum.svg",
        "azuma_adaptive.svg",
    }
