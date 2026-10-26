"""code/part01-prerequisites/solutions_concentration.py runs and reproduces the numbers in book-exercises/part01-prerequisites/02-concentration.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part01-prerequisites"


def test_solutions_concentration_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_concentration.py", tmp_path)
    for expected in (
        "[Ex4] ln(200)/(2e-4)=26491.6  n=26492",
        "gap bound=15.20",
        "exact=3.45e-03  Chernoff=2.10e-02  simplified=3.57e-02",
        "Azuma=1.83e-02  normal approx=1.01e-03  exact for always 0.7=9.28e-04",
        "rule=after one head: empirical=9.13e-04",
        "rule=after two heads: empirical=1.10e-03",
        "rule=when behind: empirical=1.18e-03",
        "rule=always 0.7: empirical=9.37e-04",
        (
            "p_hat=0.0006  Hoeffding=0.0179  Bernstein plug-in=0.00185  Bernstein=0.00256  "
            "Chernoff=0.00224  Chernoff exact exponent=0.00189  Clopper-Pearson=0.00155"
        ),
    ):
        assert expected in out
    assert not any(tmp_path.iterdir())  # the solutions draw no figures
