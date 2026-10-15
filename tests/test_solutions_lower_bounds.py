"""code/part03-dp-foundations/solutions_lower_bounds.py runs and reproduces the numbers in book-exercises/part03-dp-foundations/09-lower-bounds.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_lower_bounds_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_lower_bounds.py", tmp_path)
    for expected in (
        # Exercise 1
        "N=1e6: add/remove lower 6.3821  replace lower 6.3821  Laplace add/remove 16.79  Laplace replace 33.57",
        # Exercise 2
        "m=2^20: forced k >= 13",
        "m=2^1000: forced k >= 13",
        "limit m -> inf: k = 12.26  ln(1/delta)/eps = 13.82",
        # Exercise 3
        "c=0.25: E[f sum(x-p)]=0.1667  2c/3=0.1667  2E[pf]=0.1667  E|f-mean|=0.4500  bound=-0.2333",
        "c=0.5: E[f sum(x-p)]=0.3333  2c/3=0.3333  2E[pf]=0.3333  E|f-mean|=0.3000  bound=0.0667",
        "c=0.75: E[f sum(x-p)]=0.5000  2c/3=0.5000  2E[pf]=0.5000  E|f-mean|=0.1500  bound=0.3667",
        "c=1: E[f sum(x-p)]=0.6667  2c/3=0.6667  2E[pf]=0.6667  E|f-mean|=0.0000  bound=0.6667",
        "E|mean|: n=1 1.0000  n=5 0.6000  n=100 0.5050  limit 0.5",
        # Exercise 4
        "best alpha0=0.1167  padded constant=0.0272  (2/3-4beta)^2/8=0.0272",
        "d=1e4, alpha=0.1: n >= 15.8",
        "d=1e4, alpha=0.01: n >= 158.4",
        # Exercise 5
        "n=100: d = 5678  d/n^2 = 0.5678",
        "n=200: d = 22274  d/n^2 = 0.5568",
        # Exercise 6
        "member mean 26.15 (predicted 26.67)  non-member sd 17.11 (predicted 17.14)",
        "advantage 0.563 (predicted 0.563)",
        # Exercise 7
        "r=0.9, delta=1e-6: eps >= 2.1972",
        "r=0.99: eps >= 4.5951",
        "eps=1 allows at most r = 0.7311",
    ):
        assert expected in out
    assert list(tmp_path.iterdir()) == []
