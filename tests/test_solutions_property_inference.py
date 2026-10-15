"""code/part02-attacks/solutions_property_inference.py runs and reproduces the numbers in book-exercises/part02-attacks/04-property-inference.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-attacks"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_property_inference_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_property_inference.py", tmp_path)
    for expected in (
        # Exercise 1
        "Bin(4, 1/2) pmf x 16: 1 4 6 4 1",
        "TV(X, X+1)=0.3750  max pmf=0.3750",
        "Bin(99, 1/2): TV(X, X+1)=0.0796  max pmf=0.0796  half of it=0.0398  membership game=0.0398",
        "X uniform on {0, 2}: TV(X, X+1)=1.0000  max pmf=0.5000",
        # Exercise 2
        "Phi^-1(0.95)=1.6449  normal approximation n=270.6  exact smallest n=269  exact accuracy there=0.9501  "
        "at n=268: 0.9495",
        # Exercise 3
        "n=200, |t1 - t0|=0.1: model accuracy (Gaussian prediction)=0.6763  records matching it=21  n (kappa/2)^2=20.9",
        "n=1000, |t1 - t0|=0.1: model accuracy (Gaussian prediction)=0.8467  records matching it=105  "
        "n (kappa/2)^2=104.6",
        # Exercise 4
        "h=4: h! 2^h = 384",
        "h=8: h! 2^h = 10321920",
        "h=100: h! 2^h = 1.1831e+188",
        "same function: True  same second-moment features: True  raw parameters differ: True",
        # Exercise 5
        "k=1: Delta=1.827  predicted accuracy=0.819  simulated=0.817",
        "k=4: Delta=3.653  predicted accuracy=0.966  simulated=0.970",
        "one model on 800 records: Delta=3.653",
        # Exercise 6
        "sigma=0: property advantage=1.0000  membership advantage=0.0040",
        "sigma=10: property advantage=1.0000  membership advantage=0.0039",
        "Phi^-1(0.55)=0.1257  sigma for property advantage 0.1: 3979  (property 0.1000, membership 0.000050)",
    ):
        assert expected in out
    assert list(tmp_path.iterdir()) == []
