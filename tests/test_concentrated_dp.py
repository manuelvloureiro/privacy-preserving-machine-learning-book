"""code/part04-privacy-accounting/concentrated_dp.py runs, writes its figures, and reproduces the numbers in book-full/part04-privacy-accounting/03-concentrated-dp.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part04-privacy-accounting"

# Checks of the public functions, run in a subprocess with the code directory on the path.
PUBLIC_CHECKS = """
import math
import concentrated_dp as cdp
assert cdp.gaussian_zcdp_rho(1.0) == 0.5
assert cdp.gaussian_zcdp_rho(2.0, sensitivity=2.0) == 0.5
assert cdp.pure_to_zcdp(1.0) == 0.5
for eps in (0.5, 1.0, 3.0):
    # every eps-DP curve lies under the sharp line; randomized response is the worst case
    rho = cdp.pure_to_zcdp_sharp(eps)
    assert rho <= cdp.pure_to_zcdp(eps)
    assert all(cdp.rr_rdp(a, eps) <= a * rho * (1 + 1e-12) for a in (1.001, 1.5, 2.0, 10.0, 100.0))
assert math.isclose(cdp.zcdp_to_dp_simple(0.5, 1e-5), 0.5 + 2 * math.sqrt(0.5 * math.log(1e5)))
for rho in (0.01, 0.5, 5.0):
    improved = cdp.zcdp_to_dp_improved(rho, 1e-5)
    assert cdp.gaussian_eps(math.sqrt(2 * rho), 1e-5) < improved < cdp.zcdp_to_dp_simple(rho, 1e-5)
assert math.isclose(cdp.subsampled_gaussian_rdp_int(2, 0.01, 1.0), math.log1p(1e-4 * (math.e - 1)), rel_tol=1e-12)
assert math.isclose(cdp.subsampled_gaussian_rdp_int(7, 1 - 1e-15, 2.0), 7 / 8, rel_tol=1e-9)
assert cdp.smallest_zcdp_rho([2.0, 4.0], [1.0, 3.0]) == 0.75
print("ok")
"""


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_public_functions() -> None:
    """The improved conversion sits between the exact Gaussian and the simple formula; q = 1 recovers the Gaussian."""
    result = subprocess.run([sys.executable, "-c", PUBLIC_CHECKS], cwd=CODE, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "ok"


def test_concentrated_dp_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "concentrated_dp.py", tmp_path)
    for expected in (
        # Section 29.3
        "sigma=1: rho=0.5  eps(2)=1  eps(10)=5",
        "sigma=2, sensitivity=1: rho=0.125;  10 steps at sigma=1: rho=5",
        "mean=0.4990  var=1.0013  ln E[e^L]=1.0000",
        "tail P[L > rho + 1]: exact=1.587e-01  zCDP bound=6.065e-01  simulated=1.586e-01",
        "tail P[L > rho + 2]: exact=2.275e-02  zCDP bound=1.353e-01  simulated=2.261e-02",
        "tail P[L > rho + 4]: exact=3.167e-05  zCDP bound=3.355e-04  simulated=2.700e-05",
        "exact/bound=0.094  Mills factor sqrt(rho/pi)/lambda=0.100",
        # Section 29.4
        "eps=0.5: eps^2/2=0.1250  KL=eps tanh(eps/2)=0.1225  smallest rho for RR=0.1225  sharp rho=0.1225",
        "eps=1: eps^2/2=0.5000  KL=eps tanh(eps/2)=0.4621  smallest rho for RR=0.4621  sharp rho=0.4621",
        "eps=5: the sharp line 4.9331 alpha reaches eps=5 at alpha=1.0136",
        "ratios to exact: line at alpha=2 1.360  line at alpha=50 25.2  cap at alpha=50 1.0064",
        "RR at eps=1, keep prob=0.7311",
        "alpha=1.5: exact=0.6273  Hoeffding (KL mean)=0.7121  alpha*eps^2/2=0.7500  min(eps, alpha*eps^2/2)=0.7500",
        "alpha=2: exact=0.7353  Hoeffding (KL mean)=0.9621  alpha*eps^2/2=1.0000  min(eps, alpha*eps^2/2)=1.0000",
        "alpha=5: exact=0.9217  Hoeffding (KL mean)=2.4621  alpha*eps^2/2=2.5000  min(eps, alpha*eps^2/2)=1.0000",
        "alpha=10: exact=0.9652  Hoeffding (KL mean)=4.9621  alpha*eps^2/2=5.0000  min(eps, alpha*eps^2/2)=1.0000",
        "alpha=50: exact=0.9936  Hoeffding (KL mean)=24.9621  alpha*eps^2/2=25.0000  min(eps, alpha*eps^2/2)=1.0000",
        "bounded range eps=0.5: max D_alpha/(alpha eps^2/8) over binary pairs = 0.9965",
        "bounded range eps=1: max D_alpha/(alpha eps^2/8) over binary pairs = 0.9864",
        "bounded range eps=4: max D_alpha/(alpha eps^2/8) over binary pairs = 0.8348",
        # Section 29.5
        "ln(1/delta)=11.5129",
        "rho=0.5: best alpha (simple)=5.799",
        "rho=0.01: mu=0.1414  simple=0.689  improved fine=0.546  improved ORDERS=0.546  exact Gaussian=0.497",
        "rho=0.1: mu=0.4472  simple=2.246  improved fine=1.914  improved ORDERS=1.914  exact Gaussian=1.760",
        "rho=0.5: mu=1.0000  simple=5.299  improved fine=4.728  improved ORDERS=4.729  exact Gaussian=4.377",
        "rho=1: mu=1.4142  simple=7.786  improved fine=7.077  improved ORDERS=7.077  exact Gaussian=6.573",
        "rho=5: mu=3.1623  simple=20.174  improved fine=19.047  improved ORDERS=19.054  exact Gaussian=17.857",
        "rho=50: mu=10.0000  simple=97.985  improved fine=96.035  improved ORDERS=96.116  exact Gaussian=91.817",
        "rho=50: best order of the improved conversion on the fine grid=1.472",
        "rho=0.5 ratios to exact: simple=1.210  improved=1.080",
        "rho for eps=1 (simple): 0.02082  sigma=4.901",
        # Section 29.6
        "group of 1: rho=0.5  eps at 1e-5 (improved)=4.728  exact Gaussian with shift 1=4.377",
        "group of 2: rho=2  eps at 1e-5 (improved)=10.725  exact Gaussian with shift 2=9.997",
        "group of 4: rho=8  eps at 1e-5 (improved)=25.919  exact Gaussian with shift 4=24.382",
        "group of 8: rho=32  eps at 1e-5 (improved)=68.616  exact Gaussian with shift 8=65.319",
        "group of 2: zCDP slope 2  RDP route: order 32, eps 96, slope 3",
        "group of 4: zCDP slope 8  RDP route: order 16, eps 288, slope 18",
        "group of 8: zCDP slope 32  RDP route: order 8, eps 864, slope 108",
        "alpha=3, k=2: lhs=6  rhs=6",
        "pure eps=1 for group of 8: 8",
        # Section 29.7
        "rho=6.25  delta_approx=1.0000e-06  eps at total delta 1e-5=22.100",
        "Gaussians alone at delta 9e-6 eps=19.119, plus basic composition of the stability steps eps=5: total 24.119",
        # Section 29.8
        "alpha=2: eps=1.7181e-04  eps/alpha=8.5907e-05  unsubsampled alpha/2=1  lower bound=-8.2103",
        "alpha=4: eps=3.6315e-04  eps/alpha=9.0789e-05  unsubsampled alpha/2=2  lower bound=-4.1402",
        "alpha=8: eps=8.9364e-04  eps/alpha=1.1171e-04  unsubsampled alpha/2=4  lower bound=-1.2631",
        "alpha=10: eps=3.8270e-02  eps/alpha=3.8270e-03  unsubsampled alpha/2=5  lower bound=-0.1169",
        "alpha=12: eps=9.7801e-01  eps/alpha=8.1501e-02  unsubsampled alpha/2=6  lower bound=0.9762",
        "alpha=16: eps=3.0879e+00  eps/alpha=1.9299e-01  unsubsampled alpha/2=8  lower bound=3.0878",
        "alpha=64: eps=2.7322e+01  eps/alpha=4.2690e-01  unsubsampled alpha/2=32  lower bound=27.3217",
        "alpha=512: eps=2.5139e+02  eps/alpha=4.9099e-01  unsubsampled alpha/2=256  lower bound=251.3858",
        "lower bound turns positive at alpha = 1 + 2 sigma^2 ln(1/q) = 10.21",
        "smallest zCDP rho on integer ORDERS=0.4910  (unsubsampled rho=0.5)",
        "q^2 (e^(1/sigma^2) - 1)/2=8.5914e-05",
        "k=10000 via zCDP (rho=1/(2 sigma^2) per step): rho=5000  eps simple=5479.9  eps improved=5475.8",
        "ratio zCDP slope / tCDP slope (omega=8): 4476",
        "k=10000 via RDP on integer ORDERS: Mironov eps=7.469  improved eps=6.719",
        "best integer order: Mironov 4  improved 4",
        "tCDP omega=2: rho per step=8.5907e-05  k=10000: rho=0.8591  eps=13.231",
        "tCDP omega=3: rho per step=8.8213e-05  k=10000: rho=0.8821  eps=8.403",
        "tCDP omega=4: rho per step=9.0789e-05  k=10000: rho=0.9079  eps=7.469",
        "tCDP omega=5: rho per step=9.3733e-05  k=10000: rho=0.9373  eps=7.507",
        "tCDP omega=6: rho per step=9.7250e-05  k=10000: rho=0.9725  eps=7.665",
        "tCDP omega=8: rho per step=1.1171e-04  k=10000: rho=1.1171  eps=8.289",
        "tCDP omega=10: rho per step=3.8270e-03  k=10000: rho=38.2704  eps=80.252",
        "tCDP omega=16: rho per step=1.9299e-01  k=10000: rho=1929.9067  eps=2228.027",
        "omega=4: simple formula ignoring omega=7.374  needs alpha=4.561",
        "rho=0.1  theorem: rho'=1.300e-04  omega=11.51  condition 3rho(2+log2(1/rho))=1.597 <= ln(1/s)=4.605",
        "Poisson curve on integer orders <= omega: rho'=1.128e-05",
        "k=10000: theorem eps=9.037  RDP improved on integer ORDERS=2.046",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "concentrated_dp_privacy_loss.svg",
        "concentrated_dp_rr_line.svg",
        "concentrated_dp_conversion.svg",
        "concentrated_dp_subsampled.svg",
    }
