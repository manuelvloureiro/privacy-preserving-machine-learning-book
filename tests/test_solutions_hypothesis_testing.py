"""code/part01-prerequisites/solutions_hypothesis_testing.py runs and reproduces the numbers in book-exercises/part01-prerequisites/04-hypothesis-testing.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part01-prerequisites"


def test_solutions_hypothesis_testing_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_hypothesis_testing.py", tmp_path)
    for expected in (
        # Exercise 1
        "[mu=1] T(0.05)=0.7405  T(T(0.05))=0.0500  T(0.7405)=0.0500",
        "mu in {0.5, 1, 2, 3}: below 1e-9: True",
        # Exercise 2
        "T(P,Q) corners: (0.0, 1.0)  (0.5, 0.2)  (1.0, 0.0)",
        "T(Q,P) corners: (0.0, 1.0)  (0.2, 0.5)  (1.0, 0.0)",
        # Exercise 3
        "[kappa=1] integral pieces: 0.3161  0.0774  0.0774  0.3161  sum=0.7869",
        "[kappa=1] TV formula=0.3935  half sum=0.3935  numeric=0.3935",
        # Exercise 4
        "[effect 0.25] bound=208.3  n=209  power(n-1)=0.8996  power(n)=0.9011",
        "[effect 0.125] bound=833.1  n=834",
        "bound ratio=4.0000",
        # Exercise 5
        "Phi(6/sqrt2)=0.999989  AUC=0.5250  simulated AUC=0.5259",
        "threshold=3.0902  bulk part=0.00095  far part=0.04991  TPR@1e-3=0.0509",
        "equal-AUC shift mu=0.0887  TPR@1e-3=0.0013  ratio=38",
        # Exercise 6
        "[s=0.2] T(0)=0.8  T(0.3)=0.5",
        "[s=0.2] TV numeric=0.2000",
        "[s=0.2 LR test alpha=0.3] simulated size=0.3004  power=0.5001",
        # Exercise 7
        "[spam] min error over 16 deterministic tests=0.30  region={2, 3}",
        "[Gaussian mu=1] TV=0.3829  (1-TV)/2=0.3085  grid min=0.3085",
        # Exercise 8
        "[n=10] beta=6.458e-02  rate=0.2740  Mills lower=0.2487  upper=0.4088",
        "[n=100] beta=3.268e-17  rate=0.3796  Mills lower=0.3795  upper=0.4468",
        "[n=1000] beta=9.520e-198  rate=0.4537  Mills lower=0.4537  upper=0.4807",
        "[n=10000] Mills lower=0.4842  upper=0.4937",
        "[n=100000] Mills lower=0.4949  upper=0.4980",
        "[n=1000000] Mills lower=0.4984  upper=0.4994",
        # Exercise 9
        "[one-sided 95%] bound=2994.2  n=2995  0.999^2994=0.05001  0.999^2995=0.04996",
        "[CP 90% two-sided upper, k=0] n=2994: 1.0001e-03  n=2995: 9.9974e-04",
        "[two-sided 95%] bound=3687.0  n=3688",
    ):
        assert expected in out, expected
    assert not any(tmp_path.iterdir())
