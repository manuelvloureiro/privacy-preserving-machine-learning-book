"""code/part03-dp-foundations/solutions_trust_models.py runs and reproduces the numbers in book-exercises/part03-dp-foundations/09-trust-models.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations"


def test_solutions_trust_models_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_trust_models.py", tmp_path)
    for expected in (
        # Exercise 1
        "s=0.5: t=0.7500  eps0=1.0986",
        "s=0.25: t=0.6250  eps0=0.5108",
        "s=0.1: t=0.5500  eps0=0.2007",
        # Exercise 2
        "grid minimum at p=0.5000: n·Var=3.6827",
        "optimized (p=1/2): 3.6827  symmetric: 3.9177",
        "p=0.3: q=0.1362  n·Var=4.3842",
        "p=0.4: q=0.1970  n·Var=3.8361",
        "p=0.6: q=0.3556  n·Var=3.8361",
        "p=0.7: q=0.4619  n·Var=4.3842",
        # Exercise 3
        "local=920674  central=1415  factor e/(e−1)^2=0.9207",
        # Exercise 4
        "largest eps0=5.809  closed form there=0.5000  condition allows eps0 <= 8.093",
        "e^eps0=333.3  tanh=0.9940  ln(4/δ)=19.81  sqrt term=0.6500  1/n term=0.0027  u=0.6487  "
        "ln(1+u)=0.5000  u/ln(1+u)=1.30",
        # Exercise 5
        "m=0: honest=10000  eps=0.5009  condition allows eps0 <= 3.763",
        "m=5000: honest=5000  eps=0.6535  condition allows eps0 <= 3.070",
        "m=9000: honest=1000  eps=inf  condition allows eps0 <= 1.460",
        # Exercise 6
        "mean=-0.0077  var=8.025 (exact 8)  P[|X|>=4]=0.1361 (exact 0.1353)",
        # Exercise 7
        "30% dropped: eps=1.212",
        "per-client sigma: planned=0.597  over-provisioned for 30%=0.714",
        "variance if nobody drops: factor 1.429 of the needed",
        "ε reaches 2 when the honest fraction falls to 0.279",
        # Exercise 8
        "7 ones in 10 days: log ratio=4.3944  posterior=0.9878",
        "5 ones in 10 days: posterior=0.5000",
        "majority wrong below 0.1 after 7 days (0.07056)",
        "majority wrong below 0.01 after 19 days (0.008903)",
        "majority wrong below 0.001 after 33 days (0.000951)",
        "memoized: wrong w.p. 0.25 forever",
        # Exercise 9
        "homepages: p*=0.59375  q*=0.65625  eps_1=0.5343  eps_inf=2.0433",
        "process names: p*=0.56250  q*=0.68750  eps_1=1.0743  eps_inf=4.3944",
        # Exercise 10
        "ε0=1: e^ε0+1=3.718  w=3: 3.7701  w=4: 3.6917  w=5: 3.8218  OUE: 3.6827  k-RR at k=1000: 338.9  bits for w: 2.00",
        "ε0=2: e^ε0+1=8.389  w=7: 0.7319  w=8: 0.7246  w=9: 0.7252  OUE: 0.7241  k-RR at k=1000: 24.6  bits for w: 3.00",
        # Exercise 11
        "± 15 min, 365 days: report ever changes w.p. 0.0126",
        "± 60 min, 365 days: report ever changes w.p. 0.0503",
        "kept bits differ w.p. 0.6068",
        "γ=0.1: one round 0.776  per day the kept bit differs 2.197",
        "γ=0.2: one round 0.569  per day the kept bit differs 1.386",
        "γ=0.3: one round 0.374  per day the kept bit differs 0.847",
    ):
        assert expected in out, expected
    assert list(tmp_path.iterdir()) == []
