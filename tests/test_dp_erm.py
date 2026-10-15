"""code/part05-dp-ml/dp_erm.py runs, writes its figures, and reproduces the numbers in book-full/part05-dp-ml/01-dp-erm.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_dp_erm_matches_chapter(tmp_path: Path) -> None:
    out = run(CODE / "dp_erm.py", tmp_path)
    for expected in (
        # Toolkit
        "eps=1, delta=1e-5: mu=0.2681  exact sigma per unit sensitivity=3.7306  classic=4.8448",
        # Output perturbation
        "1-D example, n=20, lambda=0.5: w*(D)=0.2128  w*(D')=0.1296  gap=0.0832  bound=0.2000  sigma=0.7461",
        "n=10000: sensitivity 2/(lam n)=0.02  Gaussian sigma=0.07461",
        "n=10000: |w*|=2.886",
        "grad tol 1e-06: sensitivity of an approximate minimizer at n=10000 = 0.0202",
        "grad tol 0.0001: sensitivity of an approximate minimizer at n=10000 = 0.04",
        "grad tol 0.001: sensitivity of an approximate minimizer at n=10000 = 0.22",
        "Hessian at w*, n=10000: eigenvalues in [0.0263, 0.0330]  tr(H)=0.3129  tr(H^-1)=320.7",
        # Objective perturbation
        "n=100: eps'=0.5537  extra lambda=0",
        "n=10000: eps'=0.9950  extra lambda=0",
        "1-D check, n=5, lambda=0.1, 500 datasets: largest |ln p/p'|=0.3779 (eps=1)  noise part 0.0992 "
        "(budget eps'=0.1891)  Jacobian part 0.3395 (budget 0.8109)",
        # Noisy gradient descent
        "n=1000: steps T=100  per-step sigma on the mean gradient=0.07461  estimate=6.725e-02",
        "n=3000: steps T=156  per-step sigma on the mean gradient=0.03106  estimate=1.080e-02",
        "n=10000: steps T=218  per-step sigma on the mean gradient=0.01102  estimate=1.301e-03",
        "n=30000: steps T=274  per-step sigma on the mean gradient=0.004117  estimate=1.779e-04",
        "n=100000: steps T=335  per-step sigma on the mean gradient=0.001366  estimate=1.930e-05",
        "noiseless GD, n=10000, 100 steps: excess risk=2.679e-11",
        # Comparison
        "n=1000: output-Gauss=8.170e-02  output-pure=7.060e-02  objective=1.009e-02  noisy-GD=6.558e-02",
        "n=3000: output-Gauss=1.211e-02  output-pure=7.206e-03  objective=6.303e-04  noisy-GD=8.888e-03",
        "n=10000: output-Gauss=9.274e-04  output-pure=8.321e-04  objective=8.156e-05  noisy-GD=1.384e-03",
        "n=30000: output-Gauss=9.973e-05  output-pure=8.943e-05  objective=7.135e-06  noisy-GD=1.538e-04",
        "n=100000: output-Gauss=8.543e-06  output-pure=6.379e-06  objective=8.798e-07  noisy-GD=1.621e-05",
        "n=1000: test accuracy nonprivate=0.747  output-Gauss=0.681  output-pure=0.686  objective=0.736  "
        "noisy-GD=0.695",
        "n=10000: test accuracy nonprivate=0.753  output-Gauss=0.751  output-pure=0.752  objective=0.753  "
        "noisy-GD=0.751",
        "log-log slope of excess risk against n, output-Gauss: -2.01",
        "log-log slope of excess risk against n, output-pure: -2.00",
        "log-log slope of excess risk against n, objective: -2.01",
        "log-log slope of excess risk against n, noisy-GD: -1.79",
        "output-Gauss: predicted 8.711e-04  measured 9.274e-04",
        "output-pure: predicted 6.885e-04  measured 8.321e-04",
        "objective: predicted 7.127e-05  measured 8.156e-05",
        "T=20: 1.027e-03  T=50: 2.875e-04  T=100: 7.137e-04  T=200: 1.076e-03  T=500: 2.939e-03  T=1000: 6.018e-03",
    ):
        assert expected in out, expected
    for name in ("erm_output_1d.svg", "erm_objective_density.svg", "erm_comparison.svg", "erm_noisy_gd_steps.svg"):
        assert (tmp_path / name).exists()
