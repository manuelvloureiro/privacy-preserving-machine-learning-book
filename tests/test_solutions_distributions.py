"""code/part01-prerequisites/solutions_distributions.py runs and reproduces the numbers in book-exercises/part01-prerequisites/01-distributions.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part01-prerequisites"


def test_solutions_distributions_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_distributions.py", tmp_path)
    for expected in (
        "n=20    p=0.3   E[S(S-1)] 34.2000 (n(n-1)p² = 34.2000)   var 4.2000 (np(1-p) = 4.2000)",
        "n=1000  p=0.05  E[S(S-1)] 2497.5000 (n(n-1)p² = 2497.5000)   var 47.5000 (np(1-p) = 47.5000)",
        "mean 1.198   var 1.200   (q mu = 1.200)",
        "n=10     TV 0.171835   n * TV 1.7184",
        "n=100    TV 0.012598   n * TV 1.2598",
        "n=1000   TV 0.001228   n * TV 1.2276",
        "k=2     simulated 0.557   bound 1.177   ratio 0.47",
        "k=10    simulated 1.547   bound 2.146   ratio 0.72",
        "k=100   simulated 2.505   bound 3.035   ratio 0.83",
        "k=1000  simulated 3.242   bound 3.717   ratio 0.87",
        "k=2 exact 1/√π = 0.564",
        "d=10    mean 0.0021   std 0.3178   1/√d 0.3162   √d * std 1.005   max |cos| 0.908",
        (
            "d=1000  mean -0.0003   std 0.0315   1/√d 0.0316   √d * std 0.997   max |cos| 0.121 "
            "(within 7.0° of a right angle)"
        ),
    ):
        assert expected in out
    assert list(tmp_path.iterdir()) == []
