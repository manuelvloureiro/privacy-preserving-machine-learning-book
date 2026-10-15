"""code/part01-prerequisites/solutions_optimization.py runs and reproduces the numbers in book-exercises/part01-prerequisites/06-optimization.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part01-prerequisites"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_optimization_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_optimization.py", tmp_path)
    for expected in (
        "[ex1] strong convexity=0.1  smoothness=0.35  Lipschitz=1.5  gradient at w=-Rx: 1.4933",
        "[ex2] |w*|^2=10.849  beta=0.25",
        "[ex2 T=1] gap=1.21e-01  bound=1.36e+00",
        "[ex2 T=3] gap=6.02e-02  bound=4.52e-01",
        "[ex2 T=10] gap=9.24e-03  bound=1.36e-01",
        "[ex2 T=30] gap=1.67e-04  bound=4.52e-02",
        "[ex3 T=100] exact=0.0050  simulated=0.00499 +- 2.2e-05",
        "[ex3 T=1000] exact=0.0005  simulated=0.00050 +- 2.2e-06",
        "[ex4] 1/(6 sqrt 3)=0.0962  grid max=0.0962 at |s|=1.3170  ln(2+sqrt 3)=1.3170",
        "[ex5] covariance=-7/3",
        "[ex5 B=1] enumerated=14/3  formula=14/3",
        "[ex5 B=2] enumerated=7/6  formula=7/6",
        "[ex5 B=3] enumerated=0  formula=0",
        "[ex6] critical batch=80.6",
        "[ex6 B=1] T=10000  eta=0.129  bound=1.55e-03",
        "[ex6 B=10] T=1000  eta=1.289  bound=1.55e-03",
        "[ex6 B=80] T=125  eta=10.316  bound=1.55e-03",
        "[ex6 B=200] T=50  eta=10.392  bound=2.24e-03",
        "[ex6 B=1000] T=10  eta=10.392  bound=9.69e-03",
        "[ex6 B=10000] T=1  eta=10.392  bound=9.62e-02",
        "[ex7] predicted ratio=0.150",
        "[ex7 worst] shift=2.38e-02  ratio=0.119  Gamma=1.133  mu=0.0404  sharp bound=2.81e-02",
        "[ex7 all] shift / sharp bound between 0.74 and 0.88  Gamma between 0.103 and 1.133  mu between 0.0402 and 0.0404",
        "[ex8 n=101 lam=0.1] move=0.0990  formula=0.0990  bound=0.1980",
        "[ex8 n=1001 lam=0.1] move=0.0100  formula=0.0100  bound=0.0200",
        "[ex8 n=11 lam=0.1] move=0.9091  formula=0.9091  bound=1.8182",
        "[ex8 n=11 lam=0.05] move=1.0000  formula=1.0000  bound=3.6364",
    ):
        assert expected in out
    assert not any(tmp_path.iterdir())
