"""code/part03-dp-foundations/solutions_mean_estimation.py runs and reproduces the numbers in book-exercises/part03-dp-foundations/11-mean-estimation.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_mean_estimation_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_mean_estimation.py", tmp_path)
    for expected in (
        # Exercise 1
        "alpha=0.05: root=337.2  n needed=338  rmse at n=0.0499  simple bound=383  non-private=100",
        "alpha=0.01: root=3137.5  n needed=3138  rmse at n=0.0100  simple bound=3914  non-private=2500",
        # Exercise 2
        "m=217223  eps_1=0.0611  failure bound at eps_1=0.050",
        "fine rmse with eps_2=0.9389: 0.0523  (with eps_2=0.75: 0.0609)",
        # Exercise 3
        "n=1000: tau_b=18.80  excess formula=0.001826  exact bias=-0.001826  bound 1/tau=0.0532",
        "n=100000: tau_b=188.03  excess formula=0.000021  exact bias=-0.000021  bound 1/tau=0.0053",
        # Exercise 4
        "d=10: n=821  n/sqrt(d)=260",
        "d=100: n=1699  n/sqrt(d)=170",
        "d=1000: n=4556  n/sqrt(d)=144",
        "4/mu(eps_t, delta_t)=126",
        # Exercise 5
        "d=100: alpha=1: unpadded n >= 2.7  padded n >= 3.2",
        "d=1000: alpha=1: unpadded n >= 11.1  padded n >= 32.3",
        "d=10000: alpha=1: unpadded n >= 37.6  padded n >= 323.2",
    ):
        assert expected in out, expected
    assert list(tmp_path.iterdir()) == []
