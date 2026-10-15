"""code/part03-dp-foundations/solutions_definition.py runs and reproduces the numbers in book-exercises/part03-dp-foundations/01-definition.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_definition_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_definition.py", tmp_path)
    for expected in (
        # Exercise 1
        "k=2 eps=1.0986: truth prob=0.7500  other prob=0.2500  smallest eps over all pairs=1.0986",
        "k=4 eps=1.0000: truth prob=0.4754  other prob=0.1749  smallest eps over all pairs=1.0000",
        "k=10 eps=1.0000: truth prob=0.2320  other prob=0.0853  smallest eps over all pairs=1.0000",
        # Exercise 2
        "eps=1 delta=1e-6 -> replace eps=2 delta=3.7183e-06",
        "add/remove delta(0)=0.1000  replace delta(0)=0.1000  conversion bound=0.2000",
        # Exercise 3
        "eps0=ln3 eps=0: formula=0.5000  hockey-stick=0.5000",
        "eps0=ln3 eps=0.5: formula=0.3378  hockey-stick=0.3378",
        "eps0=ln3 eps=1: formula=0.0704  hockey-stick=0.0704",
        # Exercise 4
        "k=1: largest loss=1.0986  k ln3=1.0986  posterior after k yes=0.7500",
        "k=2: largest loss=2.1972  k ln3=2.1972  posterior after k yes=0.9000",
        "k=3: largest loss=3.2958  k ln3=3.2958  posterior after k yes=0.9643",
        # Exercise 5
        "n=1e6 risk 0.01: delta=1.0050e-08",
        # Exercise 6
        "bound at eps=0: delta=1e-5 -> 1.0000e-05",
        "release w.p. 0.1, n=3: TV=0.1000  bound=0.1000",
        # Exercise 7
        "eps=0.1: Laplace=0.048771  bound=0.049958  ratio=0.9762  1-bound=9.5004e-01  1-Laplace=9.5123e-01",
        "eps=1: Laplace=0.393469  bound=0.462117  ratio=0.8514  1-bound=5.3788e-01  1-Laplace=6.0653e-01",
        "eps=5: Laplace=0.917915  bound=0.986614  ratio=0.9304  1-bound=1.3386e-02  1-Laplace=8.2085e-02",
        "eps=10: Laplace=0.993262  bound=0.999909  ratio=0.9934  1-bound=9.0796e-05  1-Laplace=6.7379e-03",
        "smallest ratio=0.8284 at eps=1.76",
    ):
        assert expected in out, expected
    assert list(tmp_path.iterdir()) == []
