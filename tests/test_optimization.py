"""code/part01-prerequisites/optimization.py runs, writes its figures, and reproduces the numbers in book-full/part01-prerequisites/06-optimization.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part01-prerequisites"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_optimization_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "optimization.py", tmp_path)
    for expected in (
        "[logistic point w=(0,0)] loss=0.6931  |grad|=0.5000  curvature=0.2500",
        "[logistic point w=(3,4)] loss=0.0067  |grad|=0.0067  curvature=0.0066",
        "lambda_min=0.101  beta=0.969  kappa=9.58  GD steps to 1e-6: observed=122  predicted=126",
        "|w*|=3.152  F*=0.5198  data curvature=0.0233",
        "sigma1^2=0.1722",
        "[minibatch B=1] with repl: 1.75e-01 vs 1.72e-01   without: 1.75e-01 vs 1.72e-01",
        "[minibatch B=10] with repl: 1.74e-02 vs 1.72e-02   without: 1.68e-02 vs 1.71e-02",
        "[minibatch B=100] with repl: 1.73e-03 vs 1.72e-03   without: 1.57e-03 vs 1.55e-03",
        "[minibatch B=500] with repl: 3.47e-04 vs 3.44e-04   without: 1.74e-04 vs 1.72e-04",
        "[convex SGD T=100] eta=0.5000  gap=4.44e-02  bound=5.00e-01",
        "[convex SGD T=1000] eta=0.1581  gap=6.46e-03  bound=1.58e-01",
        "[convex SGD T=10000] eta=0.0500  gap=7.71e-04  bound=5.00e-02",
        "|w*|=0.839  G=1.5  uniform gap=9.09e-05 bound=1.15e-02  weighted gap=7.85e-05 bound=4.50e-03  last=9.61e-05",
        "[noise sigma=0 T=10000] eta*=0.0500  bound=0.050  gap at eta*=4.99e-04  best grid eta=0.178 gap=1.85e-04",
        "[noise sigma=1 T=10000] eta*=0.0354  bound=0.071  gap at eta*=1.58e-03  best grid eta=0.056 gap=1.38e-03",
        "[noise sigma=3 T=10000] eta*=0.0158  bound=0.158  gap at eta*=9.58e-03  best grid eta=0.056 gap=5.10e-03",
        "[sigmoid loss] beta=0.0962  |grad F(0)|^2=4.06e-03",
        "[non-convex T=100] eta=1.289  mean |grad|^2=1.64e-03  bound=1.55e-02",
        "[non-convex T=1000] eta=0.408  mean |grad|^2=5.22e-04  bound=4.90e-03",
        "[non-convex T=10000] eta=0.129  mean |grad|^2=1.87e-04  bound=1.55e-03",
        "[ERM n=100 lam=0.01] max shift=2.52e-01  bound=2.00e+00  ratio=0.126",
        "[ERM n=100 lam=0.1] max shift=5.98e-02  bound=2.00e-01  ratio=0.299",
        "[ERM n=100 lam=1] max shift=8.55e-03  bound=2.00e-02  ratio=0.427",
        "[ERM n=1000 lam=0.01] max shift=2.13e-02  bound=2.00e-01  ratio=0.106",
        "[ERM n=1000 lam=0.1] max shift=6.66e-03  bound=2.00e-02  ratio=0.333",
        "[ERM n=1000 lam=1] max shift=8.54e-04  bound=2.00e-03  ratio=0.427",
        "[ERM n=10000 lam=0.01] max shift=2.06e-03  bound=2.00e-02  ratio=0.103",
        "[ERM n=10000 lam=0.1] max shift=6.35e-04  bound=2.00e-03  ratio=0.317",
        "[ERM n=10000 lam=1] max shift=7.80e-05  bound=2.00e-04  ratio=0.390",
        "[double well n=11] argmin before=1.0427  after=-1.0427  shift=2.0854",
        "[double well n=1001] argmin before=1.0005  after=-1.0005  shift=2.0010",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "optimization_lipschitz_smooth.svg",
        "optimization_gd_sgd_paths.svg",
        "optimization_minibatch_variance.svg",
        "optimization_sgd_convex.svg",
        "optimization_noise_step_size.svg",
        "optimization_nonconvex_sgd.svg",
        "optimization_erm_sensitivity.svg",
        "optimization_nonconvex_jump.svg",
    }
