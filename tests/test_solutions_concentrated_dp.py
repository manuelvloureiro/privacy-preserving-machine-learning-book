"""code/part04-privacy-accounting/solutions_concentrated_dp.py runs and reproduces the numbers in book-exercises/part04-privacy-accounting/03-concentrated-dp.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part04-privacy-accounting"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_concentrated_dp_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_concentrated_dp.py", tmp_path)
    for expected in (
        # Exercise 1
        "smallest rho=0.3679  KL=t+e^-t-1=0.3679  eps^2/2=0.5000  randomized response=0.4621",
        "overcharge of eps^2/2 for Laplace=1.36  overcharge of the sharp eps tanh(eps/2)=1.26",
        "alpha=1.001: D_alpha=0.3682  D_alpha/alpha=0.3678",
        "alpha=2: D_alpha=0.6191  D_alpha/alpha=0.3096",
        "alpha=10: D_alpha=0.9287  D_alpha/alpha=0.0929",
        "alpha=100: D_alpha=0.9930  D_alpha/alpha=0.0099",
        # Exercise 2
        "ln(1/delta)=11.5129  rho=0.02082  per step=2.0820e-04  sigma(k=100)=49.01",
        "rho=0.02082  simple eps=1.0000  improved eps=0.8118  exact Gaussian eps=0.7416",
        # Exercise 3
        "R=Bern(0.4): D_2(P||Q)=1.1787  D_3(P||R)=0.1934  D_4(R||Q)=0.9294  rhs=(4/3)D_3+D_4=1.1872",
        "R=Bern(0.5): D_2(P||Q)=1.1787  D_3(P||R)=0.3662  D_4(R||Q)=0.6904  rhs=(4/3)D_3+D_4=1.1787",
        "log-odds of 0.2, 0.4, 0.5, 0.8: -1.3863  -0.4055  0.0000  1.3863",
        # Exercise 4
        "per stability step rho=0.03125  per Gaussian rho=0.020  total rho=1.6250  delta_a=2.0000e-07",
        "eps at total delta 1e-5 (improved)=9.463",
        "Gaussians alone eps=7.084  + basic 5 = 12.084",
        # Exercise 5
        "closed form=1.7181e-04  binomial sum=1.7181e-04",
        "threshold 1 + 2 sigma^2 ln(1/q)=10.21  lower bound at 10: -0.1169  at 11: 0.4343",
        # Exercise 6
        "omega=4: rho=0.9079  (omega-1)^2 rho=8.171  alpha*=4.561  branch=second  eps=7.469",
        "omega=8: rho=1.1171  (omega-1)^2 rho=54.736  alpha*=4.210  branch=first  eps=8.289",
    ):
        assert expected in out, expected
    assert list(tmp_path.iterdir()) == []
