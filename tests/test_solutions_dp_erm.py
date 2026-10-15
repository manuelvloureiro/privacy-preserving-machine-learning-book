"""code/part05-dp-ml/solutions_dp_erm.py runs and reproduces the numbers in book-exercises/part05-dp-ml/01-dp-erm.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_dp_erm_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_dp_erm.py", tmp_path)
    for expected in (
        # Exercise 1
        "eps=0.25: classic sigma=19.3792  exact sigma=13.2855  ratio=1.459",
        "eps=0.5: classic sigma=9.6896  exact sigma=7.0318  ratio=1.378",
        "eps=1.0: classic sigma=4.8448  exact sigma=3.7306  ratio=1.299",
        # Exercise 2
        "R=2: L=3.0  sensitivity at lambda=0.1, n=1000: 0.060  Gaussian sigma (eps=1, delta=1e-5)=0.2238",
        # Exercise 3
        "eps=0.1, lambda=0.01: smallest n with eps'>0 is 488",
        "n=487: eps'=0.05000  extra lambda=0.0103",
        "n=488: eps'=0.00008  extra lambda=0.0000",
        # Exercise 4
        "eps=1, delta=1e-5: sigma^2=13.918  crossover d=13",
        "d=12: pure d(d+1)=156  Gaussian d sigma^2=167.0",
        "d=13: pure d(d+1)=182  Gaussian d sigma^2=180.9",
        "d=100: pure d(d+1)=10100  Gaussian d sigma^2=1391.8",
        "eps=0.1: eps^2 sigma^2=9.455  crossover d=9",
        # Exercise 5
        "T=10: exact sigma=0.00236  basic=0.00726 (3.08x)  advanced=0.01194 (5.06x)",
        "T=100: exact sigma=0.00746  basic=0.07240 (9.70x)  advanced=0.04032 (5.40x)",
        "T=1000: exact sigma=0.02359  basic=0.72383 (30.68x)  advanced=0.13534 (5.74x)",
        # Exercise 6
        "largest |ln det ratio|/ln(1+c/(n lam)) = 0.9759",
        "extreme instance: ratio = 1.0000",
        "n=5, lambda=0.1: ln(1+a)=0.4055  2ln(1+a)=0.8109  eps' with the sharper bound=0.5945",
        # Exercise 7
        "n=10000, L=1, 10% slack: g=1e-05",
    ):
        assert expected in out, expected
