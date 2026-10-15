"""code/part05-dp-ml/solutions_dp_sgd.py runs and reproduces the numbers in book-exercises/part05-dp-ml/02-dp-sgd.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_dp_sgd_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_dp_sgd.py", tmp_path)
    for expected in (
        "eps=8: sigma=0.917  noise length=0.265  break-even steps=104",
        "eps=4: sigma=1.36  noise length=0.392  break-even steps=228",
        "eps=1: sigma=4.13  noise length=1.191  break-even steps=2100",
        "n=10000, q=0.01: relative sd of |S|=0.0995  log10 P(|S|=0)=-43.6  P(|S|<50)~2.5e-07",
        "n=1000, q=0.01: relative sd of |S|=0.3146  log10 P(|S|=0)=-4.36  P(|S|<5)~0.056",
        "Adam, C=0.01 against C=10: largest relative difference of the updates = 7.3e-05",
        "768->768: crossover T=543",
        "768->3072: crossover T=1086",
        "4096->4096: crossover T=2896",
        "eps=4, delta=1e-5: rho=0.2977",
        "q=0.01, T=10000: quadratic sigma=1.296  with e^(1/sigma^2)-1: 1.463  accountant: 1.36",
        "q=0.001, T=100000: quadratic sigma=0.410  with e^(1/sigma^2)-1: 0.718  accountant: 0.704",
        "q=0.01, sigma=1, T=100: eps=1.21",
        "q=0.01, sigma=1, T=1000: eps=2.10",
        "q=0.01, sigma=1, T=10000: eps=6.71",
        "q=0.01, sigma=1, T=100000: eps=27.23",
        "q=0.01, sigma=1: steps until eps=1: 5",
        "q=0.01, sigma=1: steps until eps=2: 882",
        "q=0.01, sigma=1: steps until eps=4: 3859",
    ):
        assert expected in out, expected
