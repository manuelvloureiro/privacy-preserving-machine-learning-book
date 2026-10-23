"""code/part02-dp-basics/approximate_dp.py runs, writes its figure, and reproduces the numbers in book-course/part02-dp-basics/03-approximate-dp.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-dp-basics"


def test_approximate_dp_matches_chapter(tmp_path: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(CODE / "approximate_dp.py"), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for expected in (
        "atom at +eps=0.5000  atom at -eps=0.1839",
        "eps=1 delta=1e-05: sigma/Delta=4.9016",
        "same sigma, eps=0.5: exact delta=6.082e-04",
        "same sigma, eps=1: exact delta=3.019e-08",
        "same sigma, eps=2: exact delta=3.008e-24",
        "sigma=Delta: Pr[L > 1]=0.3085  exact delta(1)=0.1269",
        "classic, delta<=0.74: t>=1.0239  u>=0.5356  factor=0.983",
        "classic, delta>0.74: t>0.6680  u>-0.0805  tail<=0.533",
        "classic, largest exact delta / delta over a grid of eps<=1: 0.233",
        "smallest sigma/Delta for (1, 1e-5): 3.7306",
        "d=100 n=10000 eps=1 delta=1e-6: laplace rms=0.1414  gaussian rms=0.0535  ratio=2.64",
        "simulated: laplace rms=0.1413  gaussian rms=0.0535",
    ):
        assert expected in out
    assert (tmp_path / "adp-loss.svg").exists()
