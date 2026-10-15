"""code/solutions_attribute_inference.py runs and reproduces the numbers in solutions/03-attribute-inference.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "book" / "part02-attacks" / "code"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_attribute_inference_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_attribute_inference.py", tmp_path)
    for expected in (
        # Exercise 1
        "Pr[smoker | x, y]: x=0,y=0:0.3077  x=0,y=1:0.7273  x=1,y=0:0.3913  x=1,y=1:0.7778",
        "V(S)=0.5000  V(S|X)=0.6000  V(S|X,Y)=0.7000  leakage from x=0.2630 bits  from (x, y)=0.4854 bits",
        # Exercise 2
        "Pr[smoker | young]=0.2222  Pr[smoker | old]=0.3913",
        "x=1,y=1: scores 0.210/0.210 -> s=0 (tie)",
        "success=0.7060  imputation V(S|X,Y)=0.7480",
        # Exercise 3
        "FPR=0.1: TPR=0.10785  (ratio TPR/FPR=1.0785)",
        "FPR=0.001: TPR=0.00108",
        # Exercise 4
        "success=0.7420  imputation=0.7480  loss=0.0060",
        # Exercise 5
        "both: Phi(sqrt(a^2+b^2))=Phi(1.1180)=0.8682  (simulated 0.8686)",
        "a=0.5: Phi(a)=0.6915  b=1: Phi(b)=0.8413",
        # Exercise 6
        "s=1,x=0,y=0 (mass 0.072): 0.2222 vs 0.7000 -> wrong",
        "s=1,x=1,y=0 (mass 0.054): 0.3913 vs 0.4261 -> wrong",
        "members=0.8740  non-members=0.7480  advantage=0.1260",
    ):
        assert expected in out
    assert list(tmp_path.iterdir()) == []
