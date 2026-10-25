"""code/part02-attacks/solutions_memorization.py runs and reproduces the numbers in book-exercises/part02-attacks/05-memorization-extraction.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-attacks"


def test_solutions_memorization_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_memorization.py", tmp_path)
    for expected in (
        # Exercise 1
        "E[exposure]=1.1428  Pr[exposure >= 2]=0.2  Pr[exposure >= 1]=0.5",
        "|R| = 4: E[exposure]=0.8538",
        # Exercise 2
        "bound ln(100)/p=104.19 -> 105  exact=102  Pr[success] with 102 samples=0.9901  with 101=0.9896",
        # Exercise 3
        "order 6, k=1: exposure=16.61  log2|R|=16.61",
        "order 8, k=1: exposure=16.61  log2|R|=16.61",
        "k=1: p(true)=0.2245 >= 1/V=0.0256",
        # Exercise 4
        "per step: [2, 3, 6, 2, 2]  k*=6  after the split: k*=3  if ties went the true digit's way: k*=5",
        # Exercise 5
        "bound on FPR=0.125  TPR=0.25  FPR=0.15  mean exposure planted=1.82  held out=1.64",
        "all 99980 unplanted pins: fraction with exposure >= 3 = 0.1250",
        # Exercise 6
        "k=1: mean p(line)=1.15e-10  expected copies in 20,000=0.000  predicted fraction generated=0.00  observed=0.00",
        "k=4: mean p(line)=4.81e-07  expected copies in 20,000=0.010  predicted fraction generated=0.01  observed=0.02",
        "k=8: mean p(line)=1.23e-05  expected copies in 20,000=0.246  predicted fraction generated=0.22  observed=0.25",
        "k=16: mean p(line)=1.34e-04  expected copies in 20,000=2.680  predicted fraction generated=0.91  observed=0.96",
        "k=32: mean p(line)=7.52e-04  expected copies in 20,000=15.049  predicted fraction generated=1.00  observed=1.00",
    ):
        assert expected in out, expected
    assert list(tmp_path.iterdir()) == []
