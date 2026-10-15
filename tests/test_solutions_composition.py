"""code/solutions_composition.py runs and reproduces the numbers in solutions/04-composition.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "book" / "part03-dp-foundations" / "code"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_composition_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_composition.py", tmp_path)
    for expected in (
        # Exercise 1
        "eps=0.01: tanh form=0.000050  eps^2/2=0.000050  DRV=0.000101  ratio tanh/DRV=0.4975",
        "eps=0.1: tanh form=0.004996  eps^2/2=0.005000  DRV=0.010517  ratio tanh/DRV=0.4750",
        "eps=0.5: tanh form=0.122459  eps^2/2=0.125000  DRV=0.324361  ratio tanh/DRV=0.3775",
        "eps=1: tanh form=0.462117  eps^2/2=0.500000  DRV=1.718282  ratio tanh/DRV=0.2689",
        "eps=2: tanh form=1.523188  eps^2/2=2.000000  DRV=12.778112  ratio tanh/DRV=0.1192",
        # Exercise 2
        "threshold=28.8  first k=29  advanced at k=28: 2.8336  at k=29: 2.8891",
        "tanh form: threshold=25.5  first k=26",
        "eps=0.5: threshold=186.6  first k=187",
        # Exercise 3
        "eps'=0 ln 3: exact=1/2  enumeration=0.5000  formula=0.5000",
        "eps'=1 ln 3: exact=3/8  enumeration=0.3750  formula=0.3750",
        "eps'=2 ln 3: exact=0  enumeration=0.0000  formula=0.0000",
        # Exercise 4
        "1-(1-delta)^k=4.999988e-06  k*delta=5.000000e-06  optimal=4.4469  advanced=5.9926  advanced tanh=5.4404",
        "target 4e-6 < 1-(1-delta)^k: optimal eps=inf",
        # Exercise 5
        "smallest eps=ln 2.5=0.9163  t=0.7143  K(.|0)=[0.7, 0.3, 0.0]  K(.|1)=[0.0, 0.3, 0.7]  rebuilt P=[0.5, 0.3, 0.2]",
        "at eps=0.8: K(.|0)=[0.7448, 0.3, -0.0448]",
        # Exercise 6
        "mu_k=sqrt(k)/sigma=1.5811  exact eps=7.511",
        "true mean loss k/(2 sigma^2)=1.25  per-step eps0 at delta0=5e-09: 0.2453  k*eps0*tanh(eps0/2)=29.93  "
        "eps0*sqrt(2k ln(1/delta'))=38.32",
        # Exercise 7
        "(true, true): P=1/2  Q=1/12  e^L=6",
        "(true, flipped): P=1/4  Q=1/6  e^L=3/2",
        "(flipped, true): P=1/6  Q=1/4  e^L=2/3",
        "(flipped, flipped): P=1/12  Q=1/2  e^L=1/6",
        "delta(0)=1/2",
        "delta(ln 3/2)=3/8",
        "delta(ln 2)=1/3",
    ):
        assert expected in out, expected
    assert list(tmp_path.iterdir()) == []
