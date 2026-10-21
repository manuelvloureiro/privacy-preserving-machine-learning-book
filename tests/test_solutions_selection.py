"""code/part03-dp-foundations/solutions_selection.py runs and reproduces the numbers in book-exercises/part03-dp-foundations/05-selection.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_selection_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_selection.py", tmp_path)
    for expected in (
        # Exercise 1
        "k=2: loss=1.0000  closed form=1.0000",
        "k=3: loss=1.3106  closed form=1.3106",
        "k=10: loss=1.7761  closed form=1.7761",
        "k=1000: loss=1.9977  closed form=1.9977",
        # Exercise 2
        "true median=25  OPT=-9  argmax=25  sensitivity (brute force)=1",
        "gap bound=15.22  exact Pr[gap > bound]=4.54e-05  Pr[output = 25]=0.7310  E|output - 25|=0.2690",
        "candidates within the bound: 24..25",
        "r=24: below=490  above=501  u=-11  gap=2",
        "r=25: below=500  above=491  u=-9  gap=0",
        "r=26: below=510  above=481  u=-29  gap=20",
        "records equal to 25: 10",
        # Exercise 3
        "location beta ln sum e^(u/beta)=5.8890  mean=7.0434  (location + beta gamma = 7.0434)  simulated mean=7.0430",
        "max u + beta ln k = 7.2189",
        # Exercise 4
        "a=2 ln(2/beta)=7.38  c=4 ln(2k/beta)=42.39  improvement=0.413",
        "equal split alpha=84.77 (Dwork-Roth formula 84.77)  terms: threshold 14.76, queries 84.77",
        "best split: alpha=49.76  eps1=0.148  eps2=0.852  check=49.76",
        # Exercise 5
        "Pr[bot top | D]: exact=0.1967  simulated=0.1968",
        "Pr[bot top | D']: exact=0  simulated=0.0000",
        # Exercise 6
        "d/b=0.5: EM=0.6225  Laplace=0.6209",
        "d/b=1: EM=0.7311  Laplace=0.7241",
        "d/b=2: EM=0.8808  Laplace=0.8647",
        "d/b=4: EM=0.9820  Laplace=0.9725",
        "d/b=1 simulated: Gumbel=0.7308  Laplace=0.7239",
        "50 models, eps=1, scale 2: EM Pr[best]=0.3935 E[gap]=1.54  Laplace Pr[best]=0.4241 E[gap]=1.40",
        # Exercise 7
        "generic e^(eps u/2): Z(D)=4.0000  Z(D')=4.0000  largest |ln density ratio|=0.5000  rms error=2.8284",
        "without the 2: Z(D)=2.0000  Z(D')=2.0000  largest |ln density ratio|=1.0000  rms error=1.4142",
        # Exercise 8
        "|C|=1001 alpha=0.05 eps=1: sampling=33913  privacy=793",
        "|C|=1000000 alpha=0.05 eps=1: sampling=56015  privacy=1345",
        "|C|=1001 alpha=0.05 eps=0.01: sampling=33913  privacy=79236",
        "privacy dominates below eps=0.0234 (alpha=0.05, |C|=1001)",
        # Exercise 9
        "advanced beats basic from c=29",
        "c=28: basic=3120  advanced=3155",
        "c=29: basic=3240  advanced=3218",
        "c=1000: basic=140035  advanced=23691",
    ):
        assert expected in out, expected
    assert list(tmp_path.iterdir()) == []
