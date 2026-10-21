"""code/part01-prerequisites/stability.py runs, writes its figures, and reproduces the numbers in book-full/part01-prerequisites/07-stability-generalization.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part01-prerequisites"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_stability_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "stability.py", tmp_path)
    for expected in (
        # renaming table
        "z1=0 z1'=1 prob=0.21  train=0.00  test=1.00  swapped=0.25",
        "z1=1 z1'=0 prob=0.21  train=0.25  test=0.25  swapped=1.00",
        "E train=0.0750  E test=0.2850  E swapped=0.2850  gap=0.2100  beta=0.75",
        # replace-one changes
        "[Replace-one n=100 lam=0.01] param bound=2.0e+00 change=4.06e-01  loss bound=2.0e+00 change=3.40e-01  ratio=0.17",
        "[Replace-one n=100 lam=0.1] param bound=2.0e-01 change=8.24e-02  loss bound=2.0e-01 change=4.74e-02  ratio=0.24",
        "[Replace-one n=100 lam=1] param bound=2.0e-02 change=9.76e-03  loss bound=2.0e-02 change=5.05e-03  ratio=0.25",
        "[Replace-one n=1000 lam=0.01] param bound=2.0e-01 change=3.53e-02  loss bound=2.0e-01 change=3.19e-02  ratio=0.16",
        "[Replace-one n=1000 lam=0.1] param bound=2.0e-02 change=8.09e-03  loss bound=2.0e-02 change=4.99e-03  ratio=0.25",
        "[Replace-one n=10000 lam=0.01] param bound=2.0e-02 change=3.36e-03  loss bound=2.0e-02 change=2.85e-03  ratio=0.14",
        "[Replace-one n=10000 lam=0.1] param bound=2.0e-03 change=8.04e-04  loss bound=2.0e-03 change=4.77e-04  ratio=0.24",
        "[Replace-one n=10000 lam=1] param bound=2.0e-04 change=9.76e-05  loss bound=2.0e-04 change=5.00e-05  ratio=0.25",
        # expected gap
        "[Gap n=50 lam=0.01] train=0.4603  test=0.5781  gap=1.19e-01  bound=4.0e+00  ratio=0.030",
        "[Gap n=200 lam=0.01] train=0.5146  test=0.5452  gap=2.80e-02  bound=1.0e+00  ratio=0.028",
        "[Gap n=1000 lam=0.01] train=0.5311  test=0.5381  gap=5.96e-03  bound=2.0e-01  ratio=0.030",
        "[Gap n=50 lam=0.1] train=0.6073  test=0.6401  gap=3.19e-02  bound=4.0e-01  ratio=0.080",
        "[Gap n=200 lam=0.1] train=0.6288  test=0.6373  gap=8.30e-03  bound=1.0e-01  ratio=0.083",
        "[Gap n=1000 lam=0.1] train=0.6346  test=0.6365  gap=1.78e-03  bound=2.0e-02  ratio=0.089",
        "[Gap n=50 lam=1] train=0.6806  test=0.6854  gap=4.59e-03  bound=4.0e-02  ratio=0.115",
        "[Gap n=200 lam=1] train=0.6845  test=0.6856  gap=1.10e-03  bound=1.0e-02  ratio=0.110",
        "[Gap n=1000 lam=1] train=0.6852  test=0.6855  gap=2.53e-04  bound=2.0e-03  ratio=0.127",
        # high-probability width
        "[High-prob n=10000 lam=0.1 delta=0.05] M=10.000  width=0.614",
        "[High-prob n=10000 lam=1 delta=0.05] M=1.313  width=0.065",
        # self-influence
        "gap=0.1482±0.0073  mean influence=0.1422±0.0025  against trend=0.284  with trend=0.097  share against=0.24",
        "gap=0.0420±0.0017  mean influence=0.0416±0.0003  against trend=0.055  with trend=0.037  share against=0.25",
        # SGD
        "epoch=1] train=0.559  test=0.596  gap=0.036  HRS bound=0.5  divergence=0.130  bound=0.5",
        "epoch=2] train=0.515  test=0.567  gap=0.052  HRS bound=1.0  divergence=0.189  bound=1.0",
        "epoch=5] train=0.478  test=0.557  gap=0.079  HRS bound=2.5  divergence=0.277  bound=2.5",
        "epoch=10] train=0.470  test=0.567  gap=0.097  HRS bound=5.0  divergence=0.363  bound=5.0",
        "epoch=20] train=0.467  test=0.582  gap=0.115  HRS bound=10.0  divergence=0.355  bound=10.0",
        "epoch=40] train=0.467  test=0.588  gap=0.121  HRS bound=20.0  divergence=0.395  bound=20.0",
        # 1-NN against logistic regression
        "[Bayes error=0.250  1-NN limit=0.338]",
        "[0-1 error n=50] 1-NN train=0.000 test=0.404  logistic train=0.170 test=0.301",
        "[0-1 error n=200] 1-NN train=0.000 test=0.380  logistic train=0.232 test=0.274",
        "[0-1 error n=1000] 1-NN train=0.000 test=0.367  logistic train=0.244 test=0.266",
        "[0-1 error n=4000] 1-NN train=0.000 test=0.352  logistic train=0.249 test=0.259",
        # choosing what to measure after looking
        "[Holdout reuse n=1000 d=100] kept=33.3  largest feature error=0.0436  fixed-query bound=0.0515  "
        "reused=0.6080  split=0.5032  true=0.5",
        "[Holdout reuse n=1000 d=1000] kept=325.9  largest feature error=0.0538  fixed-query bound=0.0616  "
        "reused=0.8053  split=0.5018  true=0.5",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "stability_neighbours.svg",
        "stability_erm_bound.svg",
        "stability_gap.svg",
        "stability_high_probability.svg",
        "stability_sgd_divergence.svg",
        "stability_sgd_epochs.svg",
        "stability_nearest_neighbour.svg",
        "stability_self_influence.svg",
        "stability_holdout_reuse.svg",
    }
