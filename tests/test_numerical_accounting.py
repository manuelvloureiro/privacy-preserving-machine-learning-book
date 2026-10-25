"""code/part04-privacy-accounting/numerical_accounting.py runs, writes its figures, and reproduces the numbers in book-full/part04-privacy-accounting/06-numerical-accounting.md."""

import subprocess
import sys
from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part04-privacy-accounting"

# The public functions, called as chapter 37 calls them: a Gaussian δ, and the sandwich for a short DP-SGD run.
PUBLIC_API_CHECK = """
import math
from numerical_accounting import (compose_pld, discretize_subsampled_gaussian_pld, gaussian_pld_delta,
                                  pld_delta, pld_epsilon, pld_epsilon_subsampled_gaussian)
assert math.isclose(gaussian_pld_delta(1.0, 1.0), 0.126937, abs_tol=1e-6)
up = pld_epsilon_subsampled_gaussian(0.01, 1.0, 10, 1e-5, h=1e-4, pessimistic=True)
down = pld_epsilon_subsampled_gaussian(0.01, 1.0, 10, 1e-5, h=1e-4, pessimistic=False)
assert down <= 0.3800 <= up and math.isclose(up - down, 10 * 1e-4, abs_tol=1e-6), (up, down)
pld = compose_pld(discretize_subsampled_gaussian_pld(0.01, 1.0, 1e-4, True), 10)
assert math.isclose(pld_delta(pld, pld_epsilon(pld, 1e-5)), 1e-5, rel_tol=1e-9)
"""

# Regression: a sharply peaked PLD whose FFT round-off noise used to stretch the composed grid down to
# ℓ ≈ -2200, where e^{-ℓ} overflowed in pld_epsilon and the add direction came out NaN (q = 0.001, σ = 0.8,
# k = 10,000; the same happens at h = 1e-5, but that run needs minutes and gigabytes).
OVERFLOW_CHECK = """
import math
import numpy as np
from numerical_accounting import LOSS_FLOOR, PLD, compose_pld, discretize_subsampled_gaussian_pld, pld_delta, pld_epsilon
from numerical_accounting import pld_epsilon_subsampled_gaussian
for pessimistic in (True, False):
    for direction in ("remove", "add"):
        pld = compose_pld(discretize_subsampled_gaussian_pld(0.001, 0.8, 1e-3, pessimistic, direction), 10_000)
        assert pld.offset * pld.h >= LOSS_FLOOR - pld.h, (pessimistic, direction, pld.offset * pld.h)
        assert not math.isnan(pld_epsilon(pld, 1e-5)), (pessimistic, direction)
up = pld_epsilon_subsampled_gaussian(0.001, 0.8, 10_000, 1e-5, h=1e-3, pessimistic=True)
down = pld_epsilon_subsampled_gaussian(0.001, 0.8, 10_000, 1e-5, h=1e-3, pessimistic=False)
assert down <= 0.78 <= up <= down + 10 * 1.0, (up, down)  # a tight accountant gives 0.78; the width is at most kh
# A PLD on a grid reaching far below -709 (e^{-ℓ} would overflow): ε must still be the exact root.
wide = PLD(-100_000, 1e-2, np.full(100_100, 1 / 100_100), 0.0, True)  # uniform on [-1000, 1]
eps = pld_epsilon(wide, 1e-4)
assert 0 < eps < 1 and math.isclose(pld_delta(wide, eps), 1e-4, rel_tol=1e-9), eps
"""


def test_public_functions() -> None:
    subprocess.run([sys.executable, "-c", PUBLIC_API_CHECK], cwd=CODE, check=True)


def test_pld_epsilon_no_overflow() -> None:
    subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", OVERFLOW_CHECK], cwd=CODE, check=True)


def test_numerical_accounting_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "numerical_accounting.py", tmp_path)
    for expected in (
        # Section 32.3
        "Monte Carlo mean=0.4990  variance=1.0013  E[e^-L]=1.0015",
        "eps=0: closed form delta=0.382925  quadrature=0.382925  Monte Carlo=0.382644",
        "eps=0.5: closed form delta=0.238422  quadrature=0.238422  Monte Carlo=0.238748",
        "eps=1: closed form delta=0.126937  quadrature=0.126937  Monte Carlo=0.126967",
        "eps=2: closed form delta=0.020924  quadrature=0.020924  Monte Carlo=0.020922",
        "epsilon at delta=1e-5: 4.3772",
        "sum (P - e^eps Q)_+ = 0.2351  P[L>eps] - e^eps Q[L>eps] = 0.2351",
        # Sections 32.4 and 32.6: Gaussian steps
        "k=10: exact epsilon=17.8566  grid h=1e-3: pessimistic=17.8616  optimistic=17.8516",
        "pessimistic PLD mean=5.0050  variance=10.0000",
        "k=100: exact epsilon=91.8173  grid h=1e-3: pessimistic=91.8673  optimistic=91.7673",
        "pessimistic PLD mean=50.0500  variance=100.0000",
        # Section 32.5: randomized response
        "one step: losses -1.0986, 1.0986 with probabilities 0.2500, 0.7500",
        "rounded up: losses ['-1', '1.5'] masses ['0.2500', '0.7500']  E[e^-L]=0.8469",
        "rounded down: losses ['-1.5', '1'] masses ['0.2500', '0.7500']  E[e^-L]=1.3963",
        "j=0 (+eps0 count): prob=0.0156  exact loss=-3.2958  pessimistic loss=-3  optimistic loss=-4.5",
        "j=1 (+eps0 count): prob=0.1406  exact loss=-1.0986  pessimistic loss=-0.5  optimistic loss=-2",
        "j=2 (+eps0 count): prob=0.4219  exact loss=1.0986  pessimistic loss=2  optimistic loss=0.5",
        "j=3 (+eps0 count): prob=0.4219  exact loss=3.2958  pessimistic loss=4.5  optimistic loss=3",
        "eps=0: exact delta=0.6875  pessimistic=0.7820  optimistic=0.5669",
        "eps=1: exact delta=0.4190  pessimistic=0.6758  optimistic=0.3648",
        "eps=2: exact delta=0.3064  pessimistic=0.3872  optimistic=0.2667",
        # Section 32.6: FFT
        "squaring delta(1)=0.81891018  one-shot k-th power delta(1)=0.81891018  exact=0.81851782",
        "bins: one step 15884, after k=10 50257 (ratio 3.16, sqrt(10)=3.16), one-shot window 158831",
        "width 10 around the mean (no padding): delta(1)=0.83478562",
        "width 20 around the mean (no padding): delta(1)=0.81890091",
        "width 40 around the mean (no padding): delta(1)=0.81891018",
        "mass below 0=0.0569  above 10=0.0569  share of the mass above 10 that is above 11=0.51",
        # Section 32.7: one subsampled step
        "remove: loss range [-0.0101, 3.2858]  bins at h=1e-5: 329590  E[e^-L]=0.999995",
        "add: loss range [-2.8926, 0.0101]  bins at h=1e-5: 290268",
        "eps=0: exact delta=3.829249e-03  pessimistic=3.830792e-03",
        "eps=0.5: exact delta=2.214519e-07  pessimistic=2.214633e-07",
        "eps=1: exact delta=2.732009e-09  pessimistic=2.732117e-09",
        "eps=0: exact delta=3.829249e-03  pessimistic=3.832688e-03",
        "eps=0.5: exact delta=0.000000e+00  pessimistic=0.000000e+00",
        "one step, exact epsilon at delta=1e-5: remove 0.1995  add 0.0091",
        "alpha=2: binomial=0.00017181  integration=0.00017181",
        "alpha=10: binomial=0.34443377  integration=0.34443377",
        # Section 32.8: DP-SGD
        "k=1: PLD upper=0.1995  lower=0.1994  width=0.0000",
        "RDP=0.9555 (alpha=9.9)  CLT=0.0368",
        "k=10: PLD upper=0.3800  lower=0.3799  width=0.0001",
        "RDP=1.0353 (alpha=9.4)  CLT=0.1304",
        "k=100: PLD upper=0.7185  lower=0.7175  width=0.0010",
        "RDP=1.2141 (alpha=8.8)  CLT=0.4575",
        "k=1000: PLD upper=1.8332  lower=1.8232  width=0.0100",
        "RDP=2.1014 (alpha=7.8)  CLT=1.6177",
        "k=10000: PLD upper=6.2377  lower=6.1377  width=0.1000",
        "RDP=6.7127 (alpha=4.1)  CLT=6.0071 (mu=1.3108)",
        "RDP minus PLD upper=0.7560  PLD lower minus CLT=0.1626  width=1.0e-05",
        "RDP minus PLD upper=0.2682  PLD lower minus CLT=0.2055  width=1.0e-02",
        "RDP minus PLD upper=0.4750  PLD lower minus CLT=0.1306  width=1.0e-01",
        "width=1.0e-04",
        "width=1.0e-03",
        "add 0.0091/0.0090",
        "add 0.0655/0.0654",
        "add 0.3387/0.3377",
        "add 1.4217/1.4117",
        "add 5.6615/5.5615",
        "bins of the composed pessimistic remove PLD at h=1e-5: {1: 329590, 10: 337272, 100: 425852, 1000: 767721, 10000: 2103251}",
        "h=0.001: upper=2.3279  lower=1.3279  width=1.0000  k*h=1.0000",
        "h=0.0001: upper=1.8782  lower=1.7782  width=0.1000  k*h=0.1000",
        "h=1e-05: upper=1.8332  lower=1.8232  width=0.0100  k*h=0.0100",
        "for h=1e-3, 1e-4, 1e-5: [7687, 76894, 767721]  ratios 10.00, 9.98",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "numerical_accounting_pld_weight.svg",
        "numerical_accounting_rounding.svg",
        "numerical_accounting_subsampled_loss.svg",
        "numerical_accounting_histograms.svg",
        "numerical_accounting_epsilon_vs_k.svg",
    }
