"""code/solutions_laplace_gaussian.py runs and reproduces the numbers in solutions/03-laplace-gaussian.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "book" / "part03-dp-foundations" / "code"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_laplace_gaussian_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_laplace_gaussian.py", tmp_path)
    for expected in (
        # Exercise 1
        "first-order rms=0.00295  simulated=0.00296  replace-neighbour Laplace rms=0.00141",
        # Exercise 2
        "Pr ratio on y <= 0 is e^(1/0.9)=3.0377 > e=2.7183",
        "b=1, Delta_1=1: delta(0.5)=0.2212  delta(0)=0.3935",
        # Exercise 3
        "density identity gap=0.0e+00  slope formula=-0.1816  finite difference=-0.1816",
        "mu=1: delta(0)=0.3829 = 2 Phi(1/2) - 1 = 0.3829",
        # Exercise 4
        "step 1: mu in [0.001000, 0.316228]  step 2: mu in [0.017783, 0.316228]  step 3: mu in [0.074989, 0.316228]  "
        "step 10: mu in [0.122983, 0.124373]  step 20: mu in [0.124106, 0.124107]  step 60: mu in [0.124106, 0.124106]",
        "eps=0.5 delta=1e-6: analytic sigma=8.0576  classic sigma=10.5976  ratio=1.315",
        # Exercise 5
        "k=16 sigma=4: mu=1.0000  exact eps at delta=1e-5: 4.3772  each alone: eps=1.0864 at delta/k  "
        "basic composition: eps=17.3818",
        # Exercise 6
        "Laplace l2 rms=0.02828  Gaussian l2 rms=0.00746  ratio=3.79",
    ):
        assert expected in out
    assert list(tmp_path.iterdir()) == []
