"""code/solutions_adaptive_composition.py runs and reproduces the numbers in solutions/08-adaptive-composition.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "book" / "part04-privacy-accounting" / "code"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_adaptive_composition_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_adaptive_composition.py", tmp_path)
    for expected in (
        # Exercise 1
        "(1,1): P=0.5625  Q=0.0625  L=+2.1972  spent=1.6946  N_T=1.6531",
        "(1,0): P=0.1875  Q=0.1875  L=+0.0000  spent=1.6946  N_T=0.1837",
        "(0,stop): P=0.2500  Q=0.7500  L=-1.0986  spent=0.8473  N_T=0.1429",
        "E_P[N_T]=1.0000  joint D_2=1.6740  budget B=1.6946  one step=0.8473",
        # Exercise 2
        "ln(1/delta)=13.8155  rho*=0.017469  check=1.0000",
        "eps0=0.01: k_max=floor(2 rho*/eps0^2)=349  Whitehouse statistic at k=349: 0.9994  at k=350: 1.0009",
        "optimal order 1 + sqrt(ln(1/delta)/rho*)=29.12",
        # Exercise 3
        "delta=0.05: infinite horizon=0.05  horizon s=1000: 0.05  worst fixed time: 0.00719  ratio delta/worst=7.0",
        "delta=1e-05: infinite horizon=1e-05  horizon s=1000: 1e-05  worst fixed time: 7.99e-07  ratio delta/worst=12.5",
        # Exercise 4
        "worst case eps=5.299 at alpha*=5.8  B=2.900",
        "half-norm record: spend at alpha*=0.7250  eps at alpha*=3.124  eps optimized=2.524 at alpha=10.6",
        "zCDP slope of the half-norm record rho=0.125",
        "per-step spend=0.007250  steps until the filter excludes it: 400",
        # Exercise 5
        "fixed time over ORDERS: 7.786 at alpha=4.4  continuous: 7.786",
        "Ville, |orders|=154: 9.137  continuous with ln(|orders|/delta): 9.136",
        "Ville, 10 orders: 8.605",
        "Ville at the single order 7.8 (optimal for rho=0.25): 9.493",
    ):
        assert expected in out, expected
    assert list(tmp_path.iterdir()) == []
