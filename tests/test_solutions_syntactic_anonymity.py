"""code/part02-attacks/solutions_syntactic_anonymity.py runs and reproduces the numbers in book-exercises/part02-attacks/08-syntactic-anonymity.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-attacks"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_syntactic_anonymity_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_syntactic_anonymity.py", tmp_path)
    for expected in (
        # Exercise 1
        "lambda=17.40  log2 n=19.93  bits allowed=15.81  below log2 n by 4.12",
        "k=2: lambda=4.61  lambda/k=2.30  (k + 2.33 sqrt(k))/k=2.65",
        "k=10: lambda=17.40  lambda/k=1.74  (k + 2.33 sqrt(k))/k=1.74",
        "k=100: lambda=123.61  lambda/k=1.24  (k + 2.33 sqrt(k))/k=1.23",
        # Exercise 2
        "largest top share q=0.733",
        "needs c > 7q/(6(1 - q))=3.21; posterior bound c/(1+c) at that c=0.762",
        "q < 12/19=0.632",
        # Exercise 3
        "class {3k,4k,5k}: EMD=0.375",
        "class {6k,8k,11k}: EMD=0.167",
        # Exercise 4
        "m=8 l=3: pinned=0.4762  simulated=0.4777",
        "m=8 l=2: pinned=0.8571  simulated=0.8565",
        "m=20 l=3: pinned=0.7953  simulated=0.7966",
        # Exercise 5
        "k=5: sizes 5..12  classes within [k, 2k-1]=0.977  linkage=0.1525  bounds 1/(2k-1)=0.1111  1/k=0.2000",
        "k=10: sizes 10..21  classes within [k, 2k-1]=0.985  linkage=0.0738",
    ):
        assert expected in out, expected
