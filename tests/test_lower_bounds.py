"""code/part03-dp-foundations/lower_bounds.py runs, writes its figures, and reproduces the numbers in book-full/part03-dp-foundations/09-lower-bounds.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_lower_bounds_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "lower_bounds.py", tmp_path)
    for expected in (
        # Toolkit
        "advantage bound=0.4621  bit accuracy bound=0.7311",
        "group delta: k=5: 8.579e-05  k=20: 2.824e+02",
        # Packing
        "N=1e+03: mode k >= 6.86  noisy-max gap 11.09  histogram lower 2.93  Laplace exact 9.88  union 9.90",
        "N=1e+06: mode k >= 13.76  noisy-max gap 18.44  histogram lower 6.38  Laplace exact 16.79  union 16.81",
        "N=1e+09: mode k >= 20.67  noisy-max gap 25.65  histogram lower 9.84  Laplace exact 23.69  union 23.72",
        "N=1e+03 simulated Laplace 95% error 9.84",
        "N=1e+03 noisy-max failure at gap 11.09: simulated 0.0228",
        "d=10: pure k >= 6.88  approx (delta=1e-06) forces k >= 7",
        "d=100: pure k >= 69.26  approx (delta=1e-06) forces k >= 13",
        "d=1000: pure k >= 693.10  approx (delta=1e-06) forces k >= 13",
        "ln(1/delta)/eps = 13.82",
        # Sample complexity
        "histogram N=1e6, alpha=0.1: private n >= 64  sampling n = 875",
        "histogram N=1e6, alpha=0.01: private n >= 638  sampling n = 87522",
        "d=100: pure lower 683  Laplace upper 15202  fingerprint lower 1.6  Gaussian upper 2941",
        "d=1000: pure lower 6921  Laplace upper 198070  fingerprint lower 5.0  Gaussian upper 10836",
        "d=10000: pure lower 69304  Laplace upper 2441215  fingerprint lower 15.8  Gaussian upper 38570",
        "d=100000: pure lower 693137  Laplace upper 29017315  fingerprint lower 50.1  Gaussian upper 134299",
        "Gaussian upper beats pure lower from d = 2731",
        "Gaussian mu(eps=1, delta=1e-6) = 0.2367",
        # Fingerprinting lemma
        "mean: E[f sum(x-p)]=0.6667  2E[pf]=0.6667  E|f-mean|=0.0000  bound 2/3-2E|f-mean|=0.6667",
        "half mean: E[f sum(x-p)]=0.3333  2E[pf]=0.3333  E|f-mean|=0.3000  bound 2/3-2E|f-mean|=0.0667",
        "zero: E[f sum(x-p)]=0.0000  2E[pf]=0.0000  E|f-mean|=0.6000  bound 2/3-2E|f-mean|=-0.5333",
        # Tracing
        "sigma=0: predicted gap 26.67, sd 16.97; simulated member mean 26.70  non-member mean 0.02  sd 16.92",
        "sigma=0.00: advantage 0.578 (predicted 0.568, worst-case TV 1.000)  TPR at 1% FPR 0.242",
        "sigma=1.00: advantage 0.264 (predicted 0.259, worst-case TV 0.629)  TPR at 1% FPR 0.052",
        "sigma=6.67: advantage 0.049 (predicted 0.044, worst-case TV 0.107)  TPR at 1% FPR 0.014",
        "Gaussian sigma for (1, 1e-5)-DP: 6.67; DP advantage bound 0.462; TPR bound at 1% FPR 0.0272",
        # Dinur-Nissim
        "sigma=0.1 sqrt(n): recovered 1.0000  eps >= inf  Gaussian eps <= 294.17",
        "sigma=0.25 sqrt(n): recovered 0.9607  eps >= 3.20  Gaussian eps <= 69.24",
        "sigma=0.5 sqrt(n): recovered 0.8058  eps >= 1.42  Gaussian eps <= 26.36",
        "sigma=1 sqrt(n): recovered 0.6710  eps >= 0.71  Gaussian eps <= 11.00",
        "sigma=2 sqrt(n): recovered 0.5810  eps >= 0.33  Gaussian eps <= 4.89",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "lower_packing_budget.svg",
        "lower_histogram.svg",
        "lower_sample_complexity.svg",
        "lower_fingerprinting.svg",
        "lower_tracing_scores.svg",
        "lower_tracing_advantage.svg",
        "lower_dinur_nissim.svg",
    }
