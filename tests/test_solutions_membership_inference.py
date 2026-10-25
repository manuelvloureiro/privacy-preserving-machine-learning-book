"""code/part02-attacks/solutions_membership_inference.py runs and reproduces the numbers in book-exercises/part02-attacks/02-membership-inference.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-attacks"


def test_solutions_membership_inference_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_membership_inference.py", tmp_path)
    for expected in (
        # Exercise 1
        "train error=0.1495  test error=0.2663",
        "pi=0.5: attack accuracy=0.5584  always 'non-member'=0.5000",
        "pi=0.1: attack accuracy=0.3247  always 'non-member'=0.9000",
        # Exercise 2
        "z=3.0902 s=0.5: s*z - s^2/2=1.4201  from densities=1.4201",
        "z=3.0902 s=4: s*z - s^2/2=4.3608  from densities=4.3608",
        "z=1 s=4: s*z - s^2/2=-4.0000  from densities=-4.0000",
        "log-threshold c=1.497  z-cut typical=3.244  z-cut atypical=2.374  FPR typical=5.90e-04  "
        "FPR atypical=0.0088  pooled FPR=0.0010  TPR=0.0503",
        # Exercise 3
        "q=0.05: AUC=0.5250  A: advantage=0.0500 TPR=0.05095  B: mu=0.0887 AUC check=0.5250 advantage=0.0354 TPR=0.00134",
        "q=0.01: AUC=0.5050  A: advantage=0.0100 TPR=0.01099  B: mu=0.0177 AUC check=0.5050 advantage=0.0071 TPR=0.00106",
        # Exercise 4
        "alpha=0.001: zero false positives needs n >= 2994.2, so 2995",
        "alpha=0.0001: zero false positives needs n >= 29955.8, so 29956",
        "alpha=1e-05: zero false positives needs n >= 299571.7, so 299572",
        "non-member pairs in the experiment=64000  expected false positives at 1e-3=64",
        # Exercise 5
        "d=1000 n=100: separation=3.1623  formula TPR=0.5287  simulated TPR=0.5268  FPR=0.0010",
        "d=100 n=100: separation=1.0000  formula TPR=0.0183  simulated TPR=0.0181  FPR=0.0009",
        "d=250 n=25: separation=3.1623  formula TPR=0.5287  simulated TPR=0.5350  FPR=0.0013",
        # Exercise 6
        "pool after removal=950  inlier member pairs=60800 and 60800",
        "inlier TPR at FPR 0.001: with outliers=0.0027  without=0.0037",
        "inlier mean separation: with outliers=0.502  without=0.621",
        # Exercise 7
        "class -1: a=-0.179 b=0.238  class +1: a=-0.128 b=0.158",
        "class -1: attack says 'member' with probability > 1/2 when phi > 0.751",
        "class +1: attack says 'member' with probability > 1/2 when phi > 0.813",
        "loss attack TPR at FPR 1%=0.0103",
    ):
        assert expected in out
    assert list(tmp_path.iterdir()) == []
