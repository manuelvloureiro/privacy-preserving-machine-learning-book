"""code/part03-dp-foundations/solutions_sensitivity.py runs and reproduces the numbers in book-exercises/part03-dp-foundations/06-sensitivity.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_sensitivity_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_sensitivity.py", tmp_path)
    for expected in (
        # Exercise 1
        "min age=0.00  max age=80.92  LS=0.09990  GS=Λ/n=0.09990  lower bound Λ/(2n)=0.04995",
        # Exercise 2
        "gamma=2: formula=1.0000  numeric=1.0000",
        "gamma=3: formula=1.5874  numeric=1.5874",
        "gamma=4: formula=2.2795  numeric=2.2795",
        "ages: S*=0.1632  NRS scale 6S*/ε=0.979  sharp scale 2S*/ε=0.326",
        # Exercise 3
        "n=1001 r=901: x_r=55.21  LS=0.0114  S*=0.2666 at k=7  noise scale 6S*/ε=1.599",
        "median: S*=0.1632 at k=7",
        # Exercise 4
        "distance needed for pass probability 0.99: 17.73 -> 18  pass probability at 18: 0.9924",
        "A^(17)=0.7892  distance at b=18  at b=1: 21",
        # Exercise 5
        "approximate B*=265.9  exact best divisor=250  rmse=0.0396",
        "B=200: rmse=0.0421  B=250: rmse=0.0396  B=400: rmse=0.0465",
        # Exercise 6
        "C*=5.692  bias=0.0034  noise sd=0.0080  rmse (no sampling term)=0.0087",
        "n=100: C*=3.616",
        "n=10000: C*=7.834",
        # Exercise 7
        "x: A=0,100,100,100,100,100  S*=84.648 at k=1",
        "x': A=100,100,100,100,100,100  S*=100.000 at k=0",
        "ratio S*(x')/S*(x)=1.1814  e^β=1.1814",
    ):
        assert expected in out, expected
    assert list(tmp_path.iterdir()) == []
