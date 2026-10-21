"""code/part03-dp-foundations/laplace_gaussian.py runs, writes its figures, and reproduces the numbers in book-full/part03-dp-foundations/03-laplace-gaussian.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_laplace_gaussian_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "laplace_gaussian.py", tmp_path)
    for expected in (
        # Sensitivity
        "count: add/remove: l1=1.0000 l2=1.0000  replace: l1=0.0000 l2=0.0000",
        "sum, records in [0, 3]: add/remove: l1=3.0000 l2=3.0000  replace: l1=3.0000 l2=3.0000",
        "histogram, 4 bins: add/remove: l1=1.0000 l2=1.0000  replace: l1=2.0000 l2=1.4142",
        "threshold counts, d = 3: add/remove: l1=3.0000 l2=1.7321  replace: l1=3.0000 l2=1.7321",
        "mean of 3 records in [0, 3], replace: l1=1.0000",
        # Laplace mechanism
        "privacy loss ranges over [-1.0000, 1.0000]",
        "Pr[M(D) <= 0]=0.5002  Pr[M(D') <= 0]=0.1838  ratio=2.7221  e^eps=2.7183",
        "worst ratio e^(1/0.9)=3.0377",
        "b=10  E|err|=10 (simulated 9.978)  95% width b ln 20=29.96 (simulated 29.85)  relative error 0.0243",
        "all bins within b ln(20/0.05)=5.99 (simulated failure rate 0.0464)",
        "sensitivity 0.1000  b=0.1000  95% width=0.2996",
        # Classic calibration
        "eps=1 delta=1e-05: sigma=4.8448  t=4.7416  exact delta=4.114e-08  P[L>eps]=1.060e-06  Mills=1.104e-06",
        "proof needs t >= 0.5262, which holds for every eps <= 1 when delta <= 0.7448",
        # Exact curve
        "24 of 30 (eps, mu) pairs with delta > 1e-12: max relative error 3.1e-06",
        "mu=1: delta(0)=0.3829 (TV)  delta(1)=0.1269 (integrated 0.1269, Monte Carlo 0.1270)  P[L>1]=0.3085  "
        "e Q[L>1]=0.1816  delta(2)=0.0209  delta(4)=4.71e-05",
        "sigma=2 (mu=0.5): P[L>1]=0.0401  delta(1)=0.0068",
        "mu=1: eps at delta=1e-5 is 4.3772",
        "eps=0.05: classic=96.8961  analytic=57.7707  ratio=1.677  true delta at classic=1.29e-09",
        "eps=0.1: classic=48.4481  analytic=30.7496  ratio=1.576  true delta at classic=2.64e-09",
        "eps=0.25: classic=19.3792  analytic=13.2855  ratio=1.459  true delta at classic=7.10e-09",
        "eps=0.5: classic=9.6896  analytic=7.0318  ratio=1.378  true delta at classic=1.61e-08",
        "eps=1: classic=4.8448  analytic=3.7306  ratio=1.299  true delta at classic=4.11e-08",
        "eps=2: classic=2.4224  analytic=1.9938  ratio=1.215  true delta at classic=1.34e-07",
        "eps=4: classic=1.2112  analytic=1.0812  ratio=1.120  true delta at classic=6.84e-07",
        "eps=8: classic=0.6056  analytic=0.6002  ratio=1.009  true delta at classic=7.97e-06",
        "bisection check: delta at analytic sigma=1.000e-05",
        "classic sigma first violates delta=1e-05 at eps=8.42",
        "count, eps=0.1: analytic sigma=30.75  95% width=60.27",
        # Laplace against Gaussian
        "Laplace rms=1.4142  Gaussian rms=3.7306",
        "threshold counts: Laplace better while d < 6.96",
        "d=1: Laplace rms=1.41 (simulated 1.42)  Gaussian rms=3.73 (simulated 3.73)  ratio=0.38",
        "d=10: Laplace rms=14.14 (simulated 14.09)  Gaussian rms=11.80 (simulated 11.78)  ratio=1.20",
        "d=100: Laplace rms=141.42 (simulated 141.07)  Gaussian rms=37.31 (simulated 37.28)  ratio=3.79",
        "d=1000: Laplace rms=1414.21 (simulated 1412.90)  Gaussian rms=117.97 (simulated 118.11)  ratio=11.99",
        # Privacy loss distribution of the Laplace mechanism
        "eps=1: atom at +eps=0.5000 (simulated 0.5007)  atom at -eps=0.1839 (simulated 0.1841)  continuous part=0.3161  "
        "mean loss=0.3679 (simulated 0.3682)",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "lapgauss_privacy_loss.svg",
        "lapgauss_loss_distributions.svg",
        "lapgauss_delta_curves.svg",
        "lapgauss_calibration.svg",
        "lapgauss_high_dim.svg",
    }
