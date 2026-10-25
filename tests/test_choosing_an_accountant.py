"""code/part04-privacy-accounting/choosing_an_accountant.py runs, writes its figures, and reproduces the numbers in book-full/part04-privacy-accounting/09-choosing-an-accountant.md."""

import subprocess
import sys
from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part04-privacy-accounting"

# The public functions, called as a reader would: the table for short runs, and σ for a target ε.
PUBLIC_API_CHECK = """
import math
from choosing_an_accountant import accountant_table, sigma_for_target, epsilon
rows = accountant_table(0.01, 1.0, 1e-5, [1, 10], h=1e-4)
assert [r.steps for r in rows] == [1, 10]
r = rows[1]
assert r.rdp_mironov >= r.rdp_improved >= r.rdp_optimal >= r.pld_upper >= r.pld_lower > r.clt, r
assert math.isclose(r.rdp_improved, 1.0353, abs_tol=1e-4) and math.isclose(r.clt, 0.1304, abs_tol=1e-4)
s = sigma_for_target(1.0, 0.01, 10_000, 1e-5, "rdp-improved")
assert math.isclose(s, 4.126, abs_tol=1e-3), s
assert epsilon("rdp-improved", 0.01, s, 10_000, 1e-5) <= 1.0
s_clt = sigma_for_target(1.0, 0.01, 10_000, 1e-5, "clt")
assert math.isclose(epsilon("clt", 0.01, s_clt, 10_000, 1e-5), 1.0, abs_tol=1e-6)
from choosing_an_accountant import max_direction
assert max_direction([1.0, math.inf]) == math.inf
try:
    max_direction([1.0, math.nan])
    raise AssertionError("NaN not caught")
except ValueError:
    pass
"""


def test_public_functions() -> None:
    subprocess.run([sys.executable, "-c", PUBLIC_API_CHECK], cwd=CODE, check=True)


def test_choosing_an_accountant_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "choosing_an_accountant.py", tmp_path)
    for expected in (
        # Section 35.3
        "k=1: basic=0.20  advanced=1.13  RDP Mironov=1.3175  RDP improved=0.9555  RDP optimal=0.4899  CLT=0.0368  PLD upper=0.1995  PLD lower=0.1994",
        "k=10: basic=3.64  advanced=7.88  RDP Mironov=1.4145  RDP improved=1.0353  RDP optimal=0.7951  CLT=0.1304  PLD upper=0.3800  PLD lower=0.3799",
        "k=100: basic=58.01  advanced=78.10  RDP Mironov=1.6118  RDP improved=1.2141  RDP optimal=1.1538  CLT=0.4575  PLD upper=0.7185  PLD lower=0.7175",
        "k=1000: basic=839.39  advanced=1258.64  RDP Mironov=2.5380  RDP improved=2.1014  RDP optimal=2.1009  CLT=1.6177  PLD upper=1.8332  PLD lower=1.8232",
        "k=10000: basic=11296.93  advanced=24379.55  RDP Mironov=7.4287  RDP improved=6.7127  RDP optimal=6.7127  CLT=6.0071  PLD upper=6.2377  PLD lower=6.1377",
        "k=1 ratio to PLD upper: basic=1.00  advanced=5.66  rdp-mironov=6.61  rdp-improved=4.79  rdp-optimal=2.46  clt=0.18  pld-lower=1.00",
        "k=10 ratio to PLD upper: basic=9.58  advanced=20.74  rdp-mironov=3.72  rdp-improved=2.72  rdp-optimal=2.09  clt=0.34  pld-lower=1.00",
        "k=100 ratio to PLD upper: basic=80.73  advanced=108.69  rdp-mironov=2.24  rdp-improved=1.69  rdp-optimal=1.61  clt=0.64  pld-lower=1.00",
        "k=1000 ratio to PLD upper: basic=457.88  advanced=686.57  rdp-mironov=1.38  rdp-improved=1.15  rdp-optimal=1.15  clt=0.88  pld-lower=0.99",
        "k=10000 ratio to PLD upper: basic=1811.07  advanced=3908.41  rdp-mironov=1.19  rdp-improved=1.08  rdp-optimal=1.08  clt=0.96  pld-lower=0.98",
        "k=1 slack of RDP improved over PLD upper: conversion=0.4656  curve=0.2905  total=0.7561  (379.1%)",
        "k=10 slack of RDP improved over PLD upper: conversion=0.2402  curve=0.4151  total=0.6554  (172.5%)",
        "k=100 slack of RDP improved over PLD upper: conversion=0.0603  curve=0.4353  total=0.4956  (69.0%)",
        "k=1000 slack of RDP improved over PLD upper: conversion=0.0005  curve=0.2677  total=0.2681  (14.6%)",
        "k=10000 slack of RDP improved over PLD upper: conversion=0.0000  curve=0.4750  total=0.4750  (7.6%)",
        "PLD upper >= PLD lower at every k in [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]: True",
        "observed (not implied by theory): RDP optimal >= PLD upper at every k: True; smallest curve slack 0.2677, largest kh 0.1000",
        "CLT below PLD lower at every k: True",
        # Section 35.4
        "target eps=1: sigma advanced=24.682  rdp-mironov=4.974  rdp-improved=4.126  clt=3.796  pld-upper=3.988  pld-lower=3.654",
        "noise relative to PLD upper: advanced=6.19  RDP Mironov=1.247  RDP improved=1.034  CLT=0.952  PLD lower=0.916",
        "check: PLD upper eps at sigma=3.9884 is 1.0000; RDP improved eps there is 1.0389",
        "at the CLT's sigma=3.7963: PLD lower eps=0.9550  PLD upper eps=1.0550",
        "target eps=3: sigma advanced=10.505  rdp-mironov=1.854  rdp-improved=1.662  clt=1.549  pld-upper=1.584  pld-lower=1.547",
        "noise relative to PLD upper: advanced=6.63  RDP Mironov=1.170  RDP improved=1.049  CLT=0.978  PLD lower=0.976",
        "check: PLD upper eps at sigma=1.5839 is 3.0000; RDP improved eps there is 3.2046",
        "at the CLT's sigma=1.5489: PLD lower eps=2.9938  PLD upper eps=3.0938",
        "target eps=8: sigma advanced=5.711  rdp-mironov=0.962  rdp-improved=0.917  clt=0.868  pld-upper=0.885  pld-lower=0.880",
        "noise relative to PLD upper: advanced=6.45  RDP Mironov=1.086  RDP improved=1.036  CLT=0.980  PLD lower=0.994",
        "check: PLD upper eps at sigma=0.8851 is 7.9999; RDP improved eps there is 8.6330",
        "at the CLT's sigma=0.8676: PLD lower eps=8.2594  PLD upper eps=8.3594",
        # Section 35.5
        "Gaussian, no sampling, sigma=10, k=100: exact (GDP)=4.3772  RDP improved=4.7285 (8.0% above)  RDP Mironov=5.2985 (21.0% above)",
        "factors over the Poisson PLD upper bound: fixed-size 6.96  no amplification 14.72",
        "randomized response eps0=0.1, k=100: exact (PLD)=4.3068  RDP improved=4.6154  zCDP improved=4.7263  advanced composition=5.8502  basic=10.0000",
        "subsampled Gaussian k=10000: PLD [6.1377, 6.2377]  RDP improved=6.7127 (7.6% above the upper bound)  CLT=6.0071",
        # Section 35.6
        "RDP=47.42  PLD upper=43.42  vs Poisson RDP=6.71  PLD upper=6.24",
        "mu=sqrt(100)/sigma=10  eps=91.8173",
        "replace-one neighbours (shift 2C per epoch): mu=2*sqrt(100)/sigma=20  eps=284.39",
        "10 full training runs composed (tuning on private data): RDP improved=27.23",
        "100 full training runs composed (tuning on private data): RDP improved=145.17",
        # Section 35.7
        "expected batch size q*n=500  epochs k*q=100  1/n=2.0e-05  n^-1.1=6.8e-06  1/n^2=4.0e-10",
        "delta=2.0e-05: PLD upper=6.0068  PLD lower=5.9068  width=0.1000  RDP improved=6.4884",
        "delta=1.0e-05: PLD upper=6.2377  PLD lower=6.1377  width=0.1000  RDP improved=6.7127",
        "delta=6.8e-06: PLD upper=6.3641  PLD lower=6.2641  width=0.1000  RDP improved=6.8353",
        "delta=1.0e-06: PLD upper=6.9574  PLD lower=6.8574  width=0.1000  RDP improved=7.4140",
        "delta=4.0e-10: PLD upper=9.0502  PLD lower=8.9378  width=0.1124  RDP improved=9.4540",
        "alpha=4.1  eps(alpha)=3.7336  (converts to 6.7127)",
        # Section 35.8
        "q=1, sigma=1, k=10: exact=17.8566  RDP accountant=19.0536  CLT=25.5853 (mu=4.1452 vs exact mu=3.1623)",
        "PLD accountant on the plain Gaussian, mu=1, k=10, h=1e-3: [17.8516, 17.8616]  closed form=17.8566",
        "delta of the composed pessimistic PLD at its own eps: 1.000e-05",
        "PLD upper increasing in k: True",
    ):
        assert expected in out, expected
    for name in ("choosing_an_accountant_eps_vs_k.svg", "choosing_an_accountant_noise.svg"):
        assert (tmp_path / name).stat().st_size > 0
