"""code/part03-dp-foundations/solutions_query_release.py runs and reproduces the numbers in book-exercises/part03-dp-foundations/06-query-release.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_query_release_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_query_release.py", tmp_path)
    for expected in (
        # Exercise 1
        "1024 experts, 200 runs: largest mistakes=8  mean=4.29  bound=10",
        # Exercise 2
        "1/log2(4/3)=2.4094",
        "1024 experts, best wrong 5 times, 200 runs: largest mistakes=15  bound=36.14",
        # Exercise 3
        "N=256 alpha=0.05 gamma=0.025: bound=8872",
        "N=256 alpha=0.05 gamma=0.0125: bound=11830",
        "N=256 alpha=0.05 gamma=0.0375: bound=11830",
        # Exercise 4
        "N=2^20 |Q|=1e6 alpha=0.05: search n=5.14e+08  best T=48610  L=42.78  closed form=5.12e+08",
        "approximate DP (chapter): 1.56e+07",
        # Exercise 5
        "Hoeffding per query=1.52e-04  union bound over 6560 queries=0.997",
        "m=1897: largest error of a sample, 200 samples: mean=0.0261  max=0.0451  fraction <= 0.05: 1.00",
        "m=200: mean=0.0834  fraction <= 0.05: 0.01",
        # Exercise 6
        "N=2^20 alpha=0.05: private MW needs fewer records from |Q| = 1.1e+09",
        "N=2^10 alpha=0.05: from |Q| = 5.1e+08",
    ):
        assert expected in out, expected
