"""code/part03-query-release/multiplicative_weights.py runs, writes its figure, and reproduces the numbers in book-course/part03-query-release/01-multiplicative-weights.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-query-release"


def test_multiplicative_weights_matches_chapter(tmp_path: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(CODE / "multiplicative_weights.py"), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for expected in (
        "T=354892",
        "|Q|=1e+06: basic=3.362e+08  advanced=3.535e+06  pmw=4.292e+07",
        "|Q|=1e+09: basic=4.744e+11  advanced=1.577e+08  pmw=5.271e+07",
        "|Q|=128 alpha=0.05: uniform error=0.184  updates=702  bound=8872",
        "|Q|=128 alpha=0.02: uniform error=0.184  updates=2016  bound=55452",
    ):
        assert expected in out
    assert (tmp_path / "pmw-convergence.svg").exists()
