"""code/part04-learning-statistics/private_erm.py runs, writes its figure, and reproduces the numbers in book-course/part04-learning-statistics/02-private-erm.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part04-learning-statistics"


def test_private_erm_matches_chapter(tmp_path: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(CODE / "private_erm.py"), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # The noise of private SGD quoted in the chapter, σ = 299.9 nL, and the bound per record.
    assert "n=10000 eps=1 delta=1e-06: sigma/(nL)=299.9" in out
    assert "d=100: sqrt(d) sigma/(nL)=2999  bound per record=0.30" in out
    # Regularization shrinks the effect of the moved point by orders of magnitude.
    gaps = [float(line.split("max curve gap=")[1]) for line in out.splitlines() if "max curve gap=" in line]
    assert len(gaps) == 2 and gaps[1] < gaps[0] / 10
    assert (tmp_path / "erm-regularization.svg").exists()
