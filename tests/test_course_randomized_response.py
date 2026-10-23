"""code/part02-dp-basics/randomized_response.py runs, writes its figure, and reproduces the numbers in book-course/part02-dp-basics/01-randomized-response.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-dp-basics"


def test_randomized_response_matches_chapter(tmp_path: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(CODE / "randomized_response.py"), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for expected in (
        "eps=1: truth prob=0.7311  2 gamma=0.4621",
        "n=10000 beta=0.05: coth(eps/2)=2.1640  sqrt(ln(2/beta)/(2n))=0.01358  bound=0.0294",
        "sd=0.0096",
        "eps=0.5 n=10000: simulated 95th percentile=0.0385  bound/simulated=1.44",
        "eps=2 n=10000: simulated 95th percentile=0.0081  bound/simulated=2.20",
    ):
        assert expected in out
    assert (tmp_path / "rr-error.svg").exists()
