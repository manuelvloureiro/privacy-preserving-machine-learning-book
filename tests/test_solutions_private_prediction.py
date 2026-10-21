"""code/part05-dp-ml/solutions_private_prediction.py runs and reproduces the numbers in
book-exercises/part05-dp-ml/05-private-prediction.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_private_prediction_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_private_prediction.py", tmp_path)
    for expected in (
        "gap=10, b=5: Pr[runner-up wins]=0.1353  simulated 0.1356",
        "gap=10, Gaussian with the same sd (sigma=7.07): 0.1587",
        "gap=30, b=5: Laplace 4.96e-03  Gaussian 1.35e-03",
        "sigma=51.8, Pr=0.9: gap=93.8  fraction of 250 teachers agreeing=0.688",
        "sigma=51.8, Pr=0.99: gap=170.3  fraction of 250 teachers agreeing=0.841",
        "eps=1: GNMax noise is smaller from Q=5 answers",
        "eps=4: GNMax noise is smaller from Q=6 answers",
        "eps0=0.1, alpha=8: slope=0.2500  q=0.001: slope*q=2.50e-04  bound=2.50e-04",
        "eps0=0.1, alpha=32: slope=0.7890  q=0.001: slope*q=7.89e-04  bound=7.80e-04",
        "check slope=0.0222  answers slope=0.2869  total eps=3.59",
        "plain GNMax on all 1000 queries, sigma=40: eps=5.38",
        "sigma1 at which the check costs as much as the answers: 41.7",
        "gap=150: cost 0.00123  at gap-2: 0.00135  ratio 1.096",
        "gap=200: cost 8.37e-05  at gap-2: 9.46e-05  ratio 1.130",
        "gap=250: cost 2.65e-06  at gap-2: 3.08e-06  ratio 1.165",
        "Q=1000, sigma=7.39: rate 0.04 eps=1.00  no subsampling eps=45.8",
        "no subsampling, eps=1, Q=1000: sigma=180.9",
        "expected number of queries that use one record: 40",
        # Exercise 8
        "votes 0-0: gap 0 dist 0  votes 5-4: gap 1 dist 0  votes 6-4: gap 2 dist 0  votes 7-4: gap 3 dist 1  "
        "votes 55-45: gap 10 dist 4  votes 135-115: gap 20 dist 9",
        "eps0=1, pool=1000: t=74.8  smallest gap that reaches it: 151",
        "eps0=1, pool=10000: t=84.0  smallest gap that reaches it: 171",
        "4 ln(10)=9.2",
        # Exercise 9
        "eps0=0.5, c=20: eps=9.93  t=168.1  smallest gap 339",
        "eps0=1, c=20: eps=20.00  t=84.0  smallest gap 171",
        "1000 labels with c=20: unstable share at most 0.0196",
    ):
        assert expected in out, expected
