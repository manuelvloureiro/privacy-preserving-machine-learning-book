"""code/part05-deployments/local_dp.py runs, writes its figure, and reproduces the numbers in book-course/part05-deployments/01-local-dp.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-deployments"


def test_local_dp_matches_chapter(tmp_path: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(CODE / "local_dp.py"), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for expected in (
        "eps=1 alpha=0.01 beta=0.05: central n=300",
        "eps=1 alpha=0.01 beta=0.05: local n=86370",
        "ratio local/central=288",
        "40 yes of 50 given no=1.21e-19  given yes=1.40e-01",
        "likelihood ratio=4^30=1.153e+18  composition eps=69.3",
        "f=0.5 h=2: eps_inf=4.3944",
        "p=0.5 q=0.75: q*=0.6875  p*=0.5625  eps_1=1.0743",
        "26^8=208,827,064,576  26^2*256=173,056",
        "ratio=0.765",
        "Pr[A(0) != A(m)]=0.6068",
    ):
        assert expected in out
    assert (tmp_path / "ldp-gap.svg").exists()
