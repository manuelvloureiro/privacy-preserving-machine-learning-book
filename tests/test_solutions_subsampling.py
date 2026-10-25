"""code/part04-privacy-accounting/solutions_subsampling.py runs and reproduces the numbers in book-exercises/part04-privacy-accounting/05-subsampling.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part04-privacy-accounting"


def test_solutions_subsampling_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_subsampling.py", tmp_path)
    for expected in (
        # Exercise 1
        "eps=1: after q1=0.1 0.1586; then q2=0.1 0.01704; directly with q1*q2 0.01704",
        # Exercise 2
        "terms j=0..3: 0.97029900  0.02940300  0.00080733  0.00002009",
        "A_3=1.00052942  eps(3)=ln(A_3)/2=2.6464e-04  from the script 2.6464e-04",
        "privacy cost of j=2: 0.00051033  of j=3: 0.00001909  share of j=2: 0.96",
        "without the e-factors the terms sum to 1.00000000",
        # Exercise 3
        "eps=0.1: add/remove=0.00105  exact replace-one=0.00200  group bound=0.00221  exact/add-remove=1.91  "
        "group/add-remove=2.10  Laplace event ln-ratio=0.00200",
        "eps=1: add/remove=0.01704  exact replace-one=0.02338  group bound=0.06193  exact/add-remove=1.37  "
        "group/add-remove=3.64  Laplace event ln-ratio=0.02338",
        "eps=5: add/remove=0.90589  exact replace-one=0.91587  group bound=5.39931  exact/add-remove=1.01  "
        "group/add-remove=5.96  Laplace event ln-ratio=0.91587",
        # Exercise 4
        "bracket: eps at sigma=0.5 is 47.42, at sigma=10 is 0.377",
        "target eps=1: sigma=4.126  (check: eps=1.000 at alpha=18)",
        "target eps=2: sigma=2.278  (check: eps=2.000 at alpha=10.1)",
        "target eps=8: sigma=0.917  (check: eps=7.999 at alpha=3.7)",
        # Exercise 5
        "sigma=1: Poisson eps=6.71  fixed-size eps=47.42",
        "sigma=2: Poisson eps=2.35  fixed-size eps=6.71",
        # Exercise 6
        "eps=1: eps'=0.01704  q*delta(eps)=1.26937e-03  E(mixture||mu0)=1.26937e-03  E(mu0||mixture)=0.00000e+00",
        "eps=2: eps'=0.06193  q*delta(eps)=2.09236e-04  E(mixture||mu0)=2.09236e-04  E(mu0||mixture)=0.00000e+00",
        "eps=3: eps'=0.17467  q*delta(eps)=1.53719e-05  E(mixture||mu0)=1.53719e-05  E(mu0||mixture)=0.00000e+00",
    ):
        assert expected in out, expected
