"""code/appendix-derivations/apx_core_conversions.py runs, writes its figures, and reproduces the numbers in book-full/appendix-derivations/04-conversions.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "appendix-derivations"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_appendix_core_conversions_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "apx_core_conversions.py", tmp_path)
    for expected in (
        # What ε means for any test
        "line1 FNR>=0.3495  line2 FNR>=0.8641  bound=0.8641  with delta=1e-5: 0.864076",
        "FPR=0.01: FNR>=0.9728  FPR=0.25: FNR>=0.3204",
        "eps=0.5: 0.3775  eps=1: 0.2689  eps=2: 0.1192  check FPR+e*FNR at eps=1: 1.0000",
        "tests=256  smallest slack=0.000000",
        # What ε means for belief
        "prior odds=0.010101  posterior odds=0.027457  posterior=0.0267",
        "[posterior prior=0.5 eps=1] max=0.7311",
        "[posterior eps=0.5] prior 0.01 -> 0.0164  prior 0.5 -> 0.6225",
        "[posterior eps=2] prior 0.01 -> 0.0695  prior 0.5 -> 0.8808",
        "[posterior eps=4] prior 0.01 -> 0.3555  prior 0.5 -> 0.9820",
        "D: 0.007311  D': 0.266252  sum: 0.273563  posterior: 0.0267",
        "[RR posterior simulation, 2e6 draws] 0.0267  exact 0.0267",
        "[Gaussian mu=1 at eps=4.3772] Pr[L > eps]=5.284e-05  ratio to delta=5.28",
        "bound=1.582e-05  exact Pr[L > eps+1]=5.381e-07",
        # RDP
        "ln(1/delta)=11.5129  alpha*=5.7985  Mironov eps=5.2985",
        "improved eps=4.7284 at alpha=5.430  exact eps=4.3772  saving=0.5701  Mironov excess=0.9213",
        "Pr[L > eps]=7.992e-07  exact delta=1.300e-07  at improved eps: exact delta=2.098e-06  paid/got=76.9",
        "[RDP mu=1 alpha=2] eps(alpha)=1.0000  Mironov=12.5129",
        "[RDP mu=1 alpha=5.8] eps(alpha)=2.9000  Mironov=5.2985",
        "[RDP mu=1 alpha=32] eps(alpha)=16.0000  Mironov=16.3714",
        "P[L > eps]=0.0100  largest P(S) - e^eps Q(S)=0.00989  largest loss=4.605",
        "1.4965  formula 1.5000",
        # zCDP
        "simple=2.5243  improved=2.1657  exact=1.9931  simple/exact=1.267  gap recovered=0.68",
        "simple=3.2349  improved=2.8136  exact=2.5944",
        "alpha*=5.7985  simple=5.2985  improved=4.7284  exact=4.3772",
        "simple=20.1743  improved=19.0473  exact=17.8566  simple/exact=1.130  gap recovered=0.49",
        "[zCDP mix] rho=0.3250  simple eps=4.1937",
        "rho=0.02082  sigma=4.901",
        # GDP
        "a=0.01: 0.9076  a=0.05: 0.7405  a=0.1: 0.6109  threshold at 0.05: 1.6449",
        "t=1.5000  a*=0.066807  power=0.308538  delta=0.126937  formula=0.126937  search=0.126937",
        "mu=0.5: 0.0068  mu=1: 0.1269  mu=2: 0.5099",
        "mu=0.5: 1.9931  mu=1: 4.3772  mu=2: 9.9973",
        "4 x 0.5 -> 1.0000",
        "FPR=0.0497  FNR=0.7413  exact 0.7405",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "conv_testing_region.svg",
        "conv_posterior.svg",
        "conv_rdp_order.svg",
        "conv_gdp.svg",
    }
