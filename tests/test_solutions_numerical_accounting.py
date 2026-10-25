"""code/part04-privacy-accounting/solutions_numerical_accounting.py runs and reproduces the numbers in book-exercises/part04-privacy-accounting/06-numerical-accounting.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part04-privacy-accounting"


def test_solutions_numerical_accounting_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_numerical_accounting.py", tmp_path)
    for expected in (
        # Exercise 1
        "swapped PLD delta(1)=0.127282  original delta(1)=0.127028  exact=0.126937",
        "remove E[e^-L]=0.999995  deficit moved to +inf=5.0e-06",
        "eps=0: from swapped remove PLD=3.830792e-03  direct=3.832688e-03  exact=3.829249e-03",
        "eps=0.005: from swapped remove PLD=9.660230e-04  direct=9.652622e-04  exact=9.631443e-04",
        # Exercise 2
        "eps=1: P[L>eps]=0.3085  delta=0.1269  lower bound=0.0624 at eps'=1.504",
        "eps=3: P[L>eps]=0.0062  delta=0.0015  lower bound=0.0007 at eps'=3.281",
        # Exercise 3
        "h=0.5000: E[e^-L] up=0.8469 down=1.3963  eps at delta=0.3: up=3.2583 down=1.7583",
        "h=0.1000: E[e^-L] up=0.9292 down=1.0270  eps at delta=0.3: up=2.0583 down=1.7583",
        "h=0.5493: E[e^-L] up=1.0000 down=1.0000  eps at delta=0.3: up=2.0541 down=2.0541",
        "h=0.001 k=100: width=0.1000  k*h=0.1000",
        "h=0.001 k=1000: width=1.0000  k*h=1.0000",
        "h=0.0001 k=100: width=0.0100  k*h=0.0100",
        "h=0.0001 k=1000: width=0.1000  k*h=0.1000",
        # Exercise 4
        "width 10: |delta error|=1.59e-02  mass outside=1.14e-01  Chebyshev=0.4000",
        "width 20: |delta error|=9.27e-06  mass outside=1.57e-03  Chebyshev=0.1000",
        "width 30: |delta error|=1.81e-08  mass outside=2.10e-06  Chebyshev=0.0444",
        "width 40: |delta error|=2.86e-12  mass outside=2.54e-10  Chebyshev=0.0250",
        # Exercise 5
        "pessimistic: remove 1.8749  add 1.4836  max 1.8749",
        "optimistic: remove 1.8649  add 1.4736  max 1.8649",
        "the sigma=10 Gaussian release alone: epsilon at delta=1e-5 = 0.3407",
    ):
        assert expected in out, expected
