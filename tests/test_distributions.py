"""code/distributions.py runs, writes its figures, and reproduces the numbers in 01-distributions.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "book" / "part01-prerequisites" / "code"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_distributions_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "distributions.py", tmp_path)
    for expected in (
        "Binomial(20, 0.3)  mean  6.000 vs  6.001   var  4.200 vs  4.198",
        "Poisson(4)         mean  4.000 vs  3.998   var  4.000 vs  4.001",
        "Laplace(0, b=2)    mean  0.000 vs -0.000   var  8.000 vs  7.993",
        "max CDF gap = 0.0006",
        "d=1      mean 0.802   std 0.611",
        "d=10000  mean 1.000   std 0.007",
        "n=1    0.0625   mean 1.998   var 2.001",
        "n=5    0.1216   mean 2.000   var 3.596",
        "n=100  0.1347   mean 2.001   var 3.984",
        "Exp    0.1353",
        "n=1   skewness 2.000   max CDF gap 0.1587",
        "n=2   skewness 1.414   max CDF gap 0.0945",
        "n=5   skewness 0.894   max CDF gap 0.0596",
        "n=30  skewness 0.365   max CDF gap 0.0243",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "poisson_limit.svg",
        "exponential_limit.svg",
        "laplace_vs_normal.svg",
        "difference_of_exponentials.svg",
        "gaussian_norms.svg",
        "shifted_log_ratio.svg",
        "clt_exponential.svg",
    }
