"""code/appendix-derivations/apx_matrices.py runs, writes its figures, and reproduces the numbers in
book-full/appendix-derivations/08-matrices.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "appendix-derivations"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_appendix_matrices_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "apx_matrices.py", tmp_path)
    for expected in (
        # The inverse-Wishart mean (tbl-apx-mat-wishart and the checks after it)
        "[Wishart n=50 m=60] predicted m/(m-n-1)=6.667  Gaussian=6.661  0/1 rows=6.456",
        "[Wishart n=50 m=100] predicted m/(m-n-1)=2.041  Gaussian=2.041  0/1 rows=2.000",
        "[Wishart n=50 m=200] predicted m/(m-n-1)=1.342  Gaussian=1.342  0/1 rows=1.329",
        "[Wishart n=50 m=400] predicted m/(m-n-1)=1.146  Gaussian=1.146  0/1 rows=1.140",
        "residual=4.931982  1/[inv]_11=4.931982",
        "[inverse chi2 k=11] Monte Carlo=0.1112  1/(k-2)=0.1111",
        "off/diag Gaussian=0.0013  0/1 rows=0.1000",
        # The smallest singular value
        "[smin n=500 m=2000] s_min=11.44  s_min/sqrt(m)=0.2558  half(1-sqrt(n/m))=0.2500  floor=1/14=0.0714",
        "[good queries n=100 m=400] weakest=0.3075  random min=0.5550  ones=1.0000  lemma=0.1875",
        "[net] floor at n=1: 0.0748 >= 1/14 = 0.0714",
        "[queries needed n=100 eta=0.01] m >= 115627 = 1156 n",
        "wrong bits <= 4*14^2 E^2 = 784 E^2",
        # The prefix-sum matrix (tbl-apx-mat-prefix)
        "eigenvector residual=1.4e-14  svd gap=2.7e-15",
        "[duality T=16] trace norm=26.0552  random U max=4.9448  U_A V_A^T=26.0552",
        "a decreasing=True  b increasing=True  float bounds k<10^4: True  a-1/pi=7.96e-06  1/pi-b=9.95e-11 at k=9999",
        "[prefix T=100] floor ln(2T+1)/pi=1.688  |A|_*/T=2.178  sqrt RMSE=2.371  ratio=1.089  sum r^2=2.531",
        "ceiling=2.647  lnT/pi=1.466",
        "[prefix T=10000] floor ln(2T+1)/pi=3.152  |A|_*/T=3.634  sqrt RMSE=3.836  ratio=1.056  sum r^2=3.998",
        "ceiling=4.115  lnT/pi=2.932",
        "[prefix T=1000000] floor ln(2T+1)/pi=4.618  |A|_*/T=5.100  sqrt RMSE=5.302  ratio=1.040  sum r^2=5.464",
        "ceiling=5.581  lnT/pi=4.398",
        "[prefix T=100] proved ratio <= 1.568",
        # The optimal factorization (tbl-apx-mat-t2 and the checks)
        "[T=2] identity RMSE=1.2247  sqrt RMSE=1.1859  optimum rho=0.3820 sum v=2.6180 RMSE=1.1441  bound=1.1180",
        "[Frobenius T=100] iterations=155  RMSE=2.2355  sqrt RMSE=2.3707",
        "[Frobenius T=256] iterations=193  RMSE=2.5244  sqrt RMSE=2.6688",
        "[Frobenius T=64] RMSE=2.0999  sqrt(tr Lambda/T)=2.0999  KKT residual=5.5e-13  diag gap=8.4e-11",
        "above diagonal=0.0e+00  LR-A=6.7e-16  min f(Y)-f(X)=3.4857",
        # Streaming
        "[LQ tree T=64, 127 rows] LR-A=0.0e+00  LL^T gap=0.0e+00  sens before=2.6458  after=2.6458  above diagonal=0.0e+00",
        "[adaptive T=8 taus=1,5] padded mean=0.4996 var=0.9985  plain mean=0.2681 var=0.5207",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "mat_wishart_mean.svg",
        "mat_wishart_residual.svg",
        "mat_smin.svg",
        "mat_directions.svg",
        "mat_prefix_bounds.svg",
        "mat_wallis.svg",
        "mat_eigenvectors.svg",
        "mat_frobenius.svg",
        "mat_adaptive.svg",
    }
