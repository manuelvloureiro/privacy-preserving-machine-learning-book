"""code/part02-attacks/solutions_linkage.py runs and reproduces the numbers in book-exercises/part02-attacks/07-linkage-reidentification.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-attacks"


def test_solutions_linkage_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_linkage.py", tmp_path)
    for expected in (
        # Exercise 1
        "class sizes [2, 2, 4]: success=0.375  1/min size=0.500",
        "class sizes [1, 1, 1, 2, 3]: success=0.625",
        "class sizes [3, 3, 3]: success=0.333  1/min size=0.333",
        # Exercise 2
        "f=0.5: bits=33.4  beyond log2 n=0.53",
        "f=0.99: bits=39.5  beyond log2 n=6.64",
        "n=1e+01: (1 - 1/n)^(n-1)=0.3874",
        "n=1e+02: (1 - 1/n)^(n-1)=0.3697",
        "n=1e+04: (1 - 1/n)^(n-1)=0.3679",
        "n=8e+09: (1 - 1/n)^(n-1)=0.3679",
        # Exercise 3
        "n=30 M=730: expected pairs=0.5959  union bound=0.596  exact=0.4534  lower 1-e^-pairs=0.4489",
        "birthday only (M=365): union bound=1.192  exact=0.7063",
        # Exercise 4
        "n=10000 M=10000: E[U]/n=0.3679  sd(U/n)=0.00482  simulated mean=0.3678 sd=0.00483  1.96 sd=0.0095",
        "Var U=2325.6  sum of indicator variances=2325.5",
        "n=10000 M=2500: E[U]/n=0.0183  sd(U/n)=0.00118  simulated mean=0.0183 sd=0.00117",
        "Var U=139.5  sum of indicator variances=179.7  covariance part=-40.2",
        "McDiarmid 95% half-width=0.0272",
        # Exercise 5
        "m=4: expected false matches=0.0048  (m=3: 0.48)",
        "posterior at prior 1/2=0.9952  at prior 0.01=0.6778",
        # Exercise 6
        "lambda for kappa=1/2 at pi=0.01: 1.0101",
        "lambda=0.1 pi=0.01: exact binomial=0.90992  Poisson=0.90992  floor 1/(1+lambda)=0.90909",
        "lambda=1 pi=0.01: exact binomial=0.50251  Poisson=0.50251  floor 1/(1+lambda)=0.50000",
        # Exercise 7
        "e^-phi=0.2231",
        "simulated, n=1000: exponential=0.2257  normal=0.0024",
        "normal approximation exp(-phi sqrt(2 ln n))=0.0038",
        # Exercise 8
        "licences=2000000 (20.9 bits)  seconds at 1e7/s=0.2  passwords 62^10=8.39e+17 (59.5 bits)  years=2661",
        "trips: cells=4.1e+10  n/M=0.00422  unique fraction under uniform cells=0.9958",
    ):
        assert expected in out, expected
    assert list(tmp_path.iterdir()) == []
