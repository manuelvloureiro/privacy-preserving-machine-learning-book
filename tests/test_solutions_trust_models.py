"""code/solutions_trust_models.py runs and reproduces the numbers in solutions/08-trust-models.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "book" / "part03-dp-foundations" / "code"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_trust_models_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_trust_models.py", tmp_path)
    for expected in (
        # Exercise 1
        "s=0.5: t=0.7500  eps0=1.0986",
        "s=0.25: t=0.6250  eps0=0.5108",
        "s=0.1: t=0.5500  eps0=0.2007",
        # Exercise 2
        "grid minimum at p=0.5000: n·Var=3.6827",
        "optimized (p=1/2): 3.6827  symmetric: 3.9177",
        "p=0.3: q=0.1362  n·Var=4.3842",
        "p=0.4: q=0.1970  n·Var=3.8361",
        "p=0.6: q=0.3556  n·Var=3.8361",
        "p=0.7: q=0.4619  n·Var=4.3842",
        # Exercise 3
        "local=920674  central=1415  factor e/(e−1)^2=0.9207",
        # Exercise 4
        "largest eps0=5.809  closed form there=0.5000  condition allows eps0 <= 8.093",
        "e^eps0=333.3  tanh=0.9940  ln(4/δ)=19.81  sqrt term=0.6500  1/n term=0.0027  u=0.6487  "
        "ln(1+u)=0.5000  u/ln(1+u)=1.30",
        # Exercise 5
        "m=0: honest=10000  eps=0.5009  condition allows eps0 <= 3.763",
        "m=5000: honest=5000  eps=0.6535  condition allows eps0 <= 3.070",
        "m=9000: honest=1000  eps=inf  condition allows eps0 <= 1.460",
        # Exercise 6
        "mean=-0.0077  var=8.025 (exact 8)  P[|X|>=4]=0.1361 (exact 0.1353)",
        # Exercise 7
        "30% dropped: eps=1.212",
        "per-client sigma: planned=0.597  over-provisioned for 30%=0.714",
        "variance if nobody drops: factor 1.429 of the needed",
        "ε reaches 2 when the honest fraction falls to 0.279",
    ):
        assert expected in out
    assert list(tmp_path.iterdir()) == []
