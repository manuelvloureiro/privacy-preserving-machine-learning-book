"""code/appendix-derivations/apx_optimization.py runs, writes its figures, and reproduces the numbers in book-full/appendix-derivations/07-optimization-estimation.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "appendix-derivations"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_appendix_optimization_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "apx_optimization.py", tmp_path)
    for expected in (
        # Convexity and the linear rate of gradient descent.
        "over 2000 pairs = 0.0372",
        "β=0.26 κ=26.0 1-1/κ=0.9615 Hessian eigenvalues at w*: 0.0404..0.0550",
        "gap 0.7131, distance 0.8445; typical gap ratio (steps 20..40) 0.712; (1 - λ_min(w*)/β)² = 0.713",
        "[GD t=25] gap=1.22e-05  PL bound=5.30e-02  distance bound=1.04e-01",
        "[GD t=50] gap=2.59e-09  PL bound=1.99e-02",
        # Co-coercivity.
        "f'(0)=-0.5000 f'(2)=-0.1192 lhs=0.7616 rhs=0.5800",
        "smallest lhs/rhs=1.0006",
        "smallest ratio over 2000 pairs=5.3704",
        "[Gradient step η=4] largest distance ratio over the pairs=0.9993",
        "[Gradient step η=8] largest distance ratio over the pairs=0.9988",
        "[Gradient step η=10] largest distance ratio over the pairs=1.4985",
        # The lower bound.
        "[Lower bound T=10] lower=0.1158  subgradient method=0.1285  upper=0.3162  min f(w_t)=0.0000",
        "[Lower bound T=100] lower=0.0452  subgradient method=0.0519  upper=0.1000  min f(w_t)=0.0000",
        "[Lower bound T=1000] lower=0.0153  subgradient method=0.0178  upper=0.0316  min f(w_t)=0.0000",
        "[Lower bound T=10000] lower=0.0050  subgradient method=0.0058  upper=0.0100  min f(w_t)=0.0000",
        "method / lower bound = 1.17",
        "γ=0.6340 α=0.3660 f*=-0.1830",
        # The 1-NN limit.
        "Bayes error=0.250  limit E[2p(1-p)]=0.338",
        "[1-NN n=100] exact median nearest-neighbour distance=0.758",
        "[1-NN n=4000] exact median nearest-neighbour distance=0.489",
        "[1-NN n=1000000] exact median nearest-neighbour distance=0.261",
        "[1-NN n=1000000000] exact median nearest-neighbour distance=0.121",
        "[1-NN n=100, 4000 test points] error=0.398  plug-in=0.392",
        "[1-NN n=4000, 4000 test points] error=0.356  plug-in=0.358  median distance=0.488",
        # M-estimators and influence.
        "max |σ(1-σ)(1-2σ)|=0.09623",
        "[Sandwich t=0.5 λ=0] θ*=(1.704, 0.000, -0.000)  Σ₁ diag=(12.59, 6.94, 6.04)",
        "[Sandwich t=0.5 λ=0.001] θ*=(1.683, 0.000, 0.000)  Σ₁ diag=(11.83, 6.76, 5.89)",
        "[Well specified t=0] θ*=(2.000, 1.000, -0.000)  max |G - H| below 1e-12: True",
        "G diag=(0.076, 0.191, 0.166)  H diag=(0.078, 0.166, 0.166)",
        "[Refits n=50, 2000 models] n·Var=(22.06, 9.62, 7.94)",
        "within ±1.96 sd: 0.902",
        "[Refits n=200, 2000 models] n·Var=(13.61, 7.48, 6.16)",
        "within ±1.96 sd: 0.935",
        "[Refits n=1000, 2000 models] n·Var=(11.72, 6.95, 5.92)",
        "within ±1.96 sd: 0.950",
        "[Influence, typical: 188 points] mean measured=0.2287  mean predicted=0.2291  median measured/predicted=1.003",
        "[Influence, outliers: 12 points] mean measured=3.9221  mean predicted=8.2102  median measured/predicted=0.671",
        # The Gibbs posterior and the centroid.
        "[Gibbs n=10] identity error below 1e-12: True  error of the τ approximation=0.054",
        "max |direct - formula| log-LR below 1e-10: True",
        "advantage: LR=0.2804  calibrated=0.2800  raw loss=0.0422",
        "sd of log-LR=0.757  sd of dropped term ‖z‖²/(2(n-1))=0.0504",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "apx_opt_convexity.svg",
        "apx_opt_gd_linear.svg",
        "apx_opt_cocoercive.svg",
        "apx_opt_lower_bound.svg",
        "apx_opt_nn.svg",
        "apx_opt_mest_hist.svg",
        "apx_opt_mest_variance.svg",
        "apx_opt_gibbs.svg",
    }
