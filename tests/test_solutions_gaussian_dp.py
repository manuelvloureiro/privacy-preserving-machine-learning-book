"""code/part04-privacy-accounting/solutions_gaussian_dp.py runs and reproduces the numbers in book-exercises/part04-privacy-accounting/04-gaussian-dp.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part04-privacy-accounting"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_gaussian_dp_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_gaussian_dp.py", tmp_path)
    for expected in (
        # Exercise 1
        "[Laplace eps=0.5] smallest mu=0.5618  closed form 2 Phi^-1(1 - e^(-eps/2)/2)=0.5618",
        "attained at a=0.3895 (diagonal point e^(-eps/2)/2=0.3894)",
        "attained at a=0.3035 (diagonal point e^(-eps/2)/2=0.3033)",
        "GDP eps at delta=1e-5=2.2706",
        "[Laplace eps=1] smallest mu=1.0301  closed form 2 Phi^-1(1 - e^(-eps/2)/2)=1.0301",
        "GDP eps at delta=1e-5=4.5301",
        # Exercise 2
        "[mu=0.5] max |1 - h(h(a)) - G_2mu(a)| on a grid=2.2e-16",
        "[mu=0.5 single point] eps at delta=1e-5=1.9931",
        "[group k=2] GDP mu=1: eps at delta=1e-5=4.3772  from the point: (3.9862, 8.34e-05)  GDP delta(3.9862)=4.98e-05  ratio=1.7",
        "[group k=4] GDP mu=2: eps at delta=1e-5=9.9973  from the point: (7.9724, 4.57e-03)  GDP delta(7.9724)=5.20e-04  ratio=8.8",
        # Exercise 3
        "[eps0=1] tanh(1/2)=0.4621  max |RRxRR - max(f_2,0, f_0,tanh)| on a grid=2.2e-16",
        "[delta(0)] from corners=0.4621  exact=0.4621",
        "[delta(0.5)] from corners=0.4152  exact=0.4152",
        "[delta(1)] from corners=0.3378  exact=0.3378",
        "[delta(1.5)] from corners=0.2103  exact=0.2103",
        "[delta(2)] from corners=0.0000  exact=0.0000",
        "[corners] pi^2=0.5344  (1-pi)^2=0.0723  corners=[(0.0, 1.0), (0.0723, 0.4656), (0.2689, 0.2689), (0.4656, 0.0723), (1.0, 0.0)]",
        # Exercise 4
        "[steps ratio eps 4 vs 2] 3.40",
        "[k=100 delta(1)] exact=0.1257  CLT=0.1271  difference=0.0014",
        "[eps=1 delta=1e-5] mu*=0.2681  delta at mu*=1.00e-05  full-batch sigma=20: k=28",
        "[DP-SGD q=0.01 sigma=1 eps=1] mu*=0.2681  max k=418  eps at that k=0.9998",
        "[DP-SGD q=0.01 sigma=1 eps=2] mu*=0.5016  max k=1463  eps at that k=1.9992",
        "[DP-SGD q=0.01 sigma=1 eps=4] mu*=0.9249  max k=4978  eps at that k=3.9996",
        # Exercise 5
        "[k=100 eps0=0.1000] kl=0.004996  eps0 tanh(eps0/2)=0.004996  mu=1.0004  2 sqrt(k) sinh(eps0/2)=1.0004  gamma=0.0562",
        "[k=10 eps0=0.3162] kl=0.049587  eps0 tanh(eps0/2)=0.049587  mu=1.0042  2 sqrt(k) sinh(eps0/2)=1.0042  gamma=0.1837",
        "[k=400 eps0=0.0500] kl=0.001250  eps0 tanh(eps0/2)=0.001250  mu=1.0001  2 sqrt(k) sinh(eps0/2)=1.0001  gamma=0.0280",
        "gamma=0.0562  0.56(1+t^2)cosh(eps0/2)/sqrt(k)=0.0562",
        # Exercise 6
        "[q=1 sigma=1 k=100] CLT mu=13.1083  exact mu=10.0000  ratio=1.3108",
        "[q=1 sigma=2 k=100] CLT mu=5.3294  exact mu=5.0000  ratio=1.0659",
        "[q=1 sigma=5 k=100] CLT mu=2.0202  exact mu=2.0000  ratio=1.0101",
    ):
        assert expected in out, expected
    assert list(tmp_path.iterdir()) == []
