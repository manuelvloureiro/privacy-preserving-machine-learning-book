"""code/part03-dp-foundations/solutions_properties.py runs and reproduces the numbers in book-exercises/part03-dp-foundations/02-properties.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_properties_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_properties.py", tmp_path)
    for expected in (
        # Exercise 1
        "P(majority 1 | bit 1)=0.84375 (54/64=0.84375)  P(majority 1 | bit 0)=0.15625 (10/64=0.15625)  ratio=5.4000",
        "exact eps=1.6864  ln 5.4=1.6864  composition bound=3.2958",
        # Exercise 2
        "eps=0.5 delta=1e-05 k=10: sum=2.272e-03  simple=9.002e-03",
        "eps=1, k=10: simple/sum=6.3215",
        "eps=0.1, k=10: simple/sum=1.5054",
        "eps=0.01, k=10: simple/sum=1.0456",
        "alarm mechanism, size 10: a_0=2.272e-03  1 - e^-eps=0.3935  largest neighbouring delta(eps)=1.000e-05",
        # Exercise 3
        "user level: eps=1.0000  delta=7.761e-07",
        "10 runs: eps=10.0000  delta=7.761e-06",
        "compose first, then group: eps=10.0000  delta=3.447e-03  ratio to group first=444",
        # Exercise 4
        "add/remove: eps=1.0000  replace across bins: eps=2.0000  replace within a slot: eps=1.0000",
        "b=2: add/remove eps=0.5000  replace across bins eps=1.0000",
        # Exercise 5
        "E_(e^0)=1.0000  E_(e^1)=1.0000  E_(e^5)=1.0000  E_(e^20)=1.0000",
        "fresh noise: pure eps of the pair=1.0000",
        # Exercise 6
        "eps0=0.1 k=1000 delta=1e-06: mean=4.996  spread term=16.623  Hoeffding eps=21.62  exact eps=19.345  basic=100",
        "exact P_D[L > 21.62]=5.319e-08",
        "truth probability at eps0=0.1: 0.5250  2t - 1=0.04996  tanh(eps0/2)=0.04996",
        # Exercise 7
        "bit 1: 0.750000  0.187500  0.046875  0.015625",
        "bit 0: 0.250000  0.187500  0.140625  0.421875",
        "exact eps=3.2958  3 ln 3=3.2958",
    ):
        assert expected in out, expected
    assert not any(tmp_path.iterdir())
