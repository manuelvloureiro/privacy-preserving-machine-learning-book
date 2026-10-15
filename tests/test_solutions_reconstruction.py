"""code/part02-attacks/solutions_reconstruction.py runs and reproduces the numbers in book-exercises/part02-attacks/09-reconstruction.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-attacks"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_reconstruction_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_reconstruction.py", tmp_path)
    for expected in (
        # Exercise 1
        "d=[1, 1, 1, 0, 0, 0, 0, 0, 0, 1]",
        "width 2: consistent candidates=1  max distance=0  fewest separating subsets over one-bit flips=256",
        "width 4: consistent candidates=1  max distance=0  fewest separating subsets over one-bit flips=128",
        "worst-case curator with E=2: consistent candidates=912  max distance=8",
        # Exercise 2
        "theta=0.2500: escape=0.1875  threshold=0.8333  tolerance E <= 0.1667  queries m >= 33296",
        "theta=0.1111: escape=0.2634  threshold=0.5556  tolerance E <= 0.0278  queries m >= 22617",
        # Exercise 3
        "a=[1.0, 1.0, 1.0]: enumeration=21.0000  formula=21.0000",
        "a=[1.0, 2.0]: enumeration=41.0000  formula=41.0000",
        "a=[0.5, 0.5, 0.5, 0.5]: enumeration=2.5000  formula=2.5000",
        # Exercise 4
        "diagonal factor n/(n+1) at n=500: 0.9980",
        # Exercise 5
        "sigma=2 sqrt(n): m/n=106.1",
        "sigma=5 sqrt(n): m/n=657.9",
        "n=100, m=10611, sigma=2 sqrt(n): simulated=0.9100  predicted=0.9011",
        # Exercise 6
        "n=1000 beta=0.01 width 10: values=102  log2=6.6724  m >= 137.8",
        # Exercise 7
        "Hoeffding bound=59.03  observed max error=12.00  kept=445  recovered=0.9420  ceiling=0.95",
        # Exercise 8
        "sums: total=240  female=121  male=119  female + male=240",
        "withhold married mean: 45 blocks, 7 patterns, true pattern share 0.51",
    ):
        assert expected in out
    for group in ("total", "female", "male", "black", "white"):
        assert f"withhold {group} mean: 1 blocks, 1 patterns, true pattern share 1.00" in out
    assert "max entry error=" in out
    assert list(tmp_path.iterdir()) == []
