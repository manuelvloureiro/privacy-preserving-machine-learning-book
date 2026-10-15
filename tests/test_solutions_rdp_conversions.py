"""code/solutions_rdp_conversions.py runs and reproduces the numbers in solutions/02-rdp-conversions.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "book" / "part04-privacy-accounting" / "code"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_rdp_conversions_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_rdp_conversions.py", tmp_path)
    for expected in (
        # Exercise 1
        "alpha=2 eps=1: c* formula=9.196986e-02  grid=9.196986e-02  z*=1.6931",
        "alpha=2: ratio=0.250000  alpha*ratio=0.5000",
        "alpha=10: ratio=0.038742  alpha*ratio=0.3874",
        "alpha=100: ratio=0.003697  alpha*ratio=0.3697",
        "alpha=1000: ratio=0.000368  alpha*ratio=0.3681",
        "1/e=0.3679",
        "alpha=6 eps=4: c* formula=1.380552e-10  grid=1.380552e-10  z*=4.1823",
        # Exercise 2
        "rho=0.5 delta=1e-05: Mironov alpha*=5.799 eps*=5.299 (direct 5.299)  improved alpha*=5.432 eps*=4.728",
        "rho=0.5 delta=1e-10: Mironov alpha*=7.786 eps*=7.286 (direct 7.286)  improved alpha*=7.483 eps*=6.839",
        "rho=0.005 delta=1e-05: Mironov alpha*=48.985 eps*=0.485 (direct 0.485)  improved alpha*=40.525 eps*=0.375",
        "rho=0.005 delta=1e-10: Mironov alpha*=68.861 eps*=0.684 (direct 0.684)  improved alpha*=62.467 eps*=0.604",
        "gap=8.9e-16",
        # Exercise 3
        "alpha=2 a=0.001: exact=4.2431e-02  crude=5.2137e-02  refined=4.1452e-02  crude/exact=1.2287  refined/exact=0.9769",
        "alpha=2 a=1e-06: exact=1.3118e-03  crude=1.6487e-03  refined=1.3108e-03  crude/exact=1.2568  refined/exact=0.9992",
        "alpha=2: (1 - e^(-(alpha-1)eps(alpha)))^(1/alpha)=0.7951",
        "alpha=8 a=0.001: exact=7.8529e-02  crude=7.8529e-02  refined=7.8529e-02  crude/exact=1.0000  refined/exact=1.0000",
        "alpha=8 a=1e-06: exact=1.8622e-04  crude=1.8622e-04  refined=1.8622e-04  crude/exact=1.0000  refined/exact=1.0000",
        "alpha=8: (1 - e^(-(alpha-1)eps(alpha)))^(1/alpha)=1.0000",
        # Exercise 4
        "alpha=2: Laplace RDP formula=0.009644  numeric=0.009644  Gaussian-like alpha eps0^2/2=0.010000",
        "alpha=20: Laplace RDP formula=0.065854  numeric=0.065854  Gaussian-like alpha eps0^2/2=0.100000",
        "basic=10.000  advanced=5.850  RDP improved=4.533 (alpha 5.8)  RDP Mironov=5.071 (alpha 6.3)  "
        "via min(eps0, alpha eps0^2/2)=4.729  gain of the exact curve=0.196",
        # Exercise 5
        "sigma=1 k=10000: rho=5000  continuous alpha*=1.048 eps=5475.7911  grid alpha=1.1 eps=5611.7783  excess=2.48%",
        "sigma=10 k=1: rho=0.005  continuous alpha*=40.525 eps=0.3753  grid alpha=41 eps=0.3753  excess=0.01%",
        "sigma=100 k=1: rho=5e-05  continuous alpha*=338.312 eps=0.0308  grid alpha=256 eps=0.0323  excess=4.76%",
        "sigma=300 k=1: rho=5.556e-06  continuous alpha*=919.694 eps=0.0091  grid alpha=512 eps=0.0112  excess=22.86%",
        # Exercise 6
        "alpha=1.5: improved=21.8663  optimal=19.5415  c*=6.873e-06  c*/delta=0.687",
        "alpha=2: improved=11.1266  optimal=10.6680  c*=3.679e-06  c*/delta=0.368",
        "alpha=4: improved=5.0879  optimal=5.0870  c*=2.479e-08  c*/delta=0.002",
        "alpha=8: improved=5.2141  optimal=5.2141  c*=6.914e-18  c*/delta=0.000",
    ):
        assert expected in out, expected
