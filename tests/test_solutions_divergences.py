"""code/part01-prerequisites/solutions_divergences.py runs and reproduces the numbers in book-exercises/part01-prerequisites/03-divergences.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part01-prerequisites"


def test_solutions_divergences_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_divergences.py", tmp_path)
    for expected in (
        # Exercise 1
        "KL(P_X‖Q_X)=0.0823  KL(P_Y|even‖Q_Y|even)=0.3023  KL(P_Y|odd‖Q_Y|odd)=0.0000",
        "expected conditional KL=0.2116  sum=0.2939  direct KL=0.2939",
        # Exercise 2
        "KL(N(0,1)‖N(0,4)): formula=0.3181  quadrature=0.3181",
        "KL(N(0,4)‖N(0,1)): formula=0.8069",
        "D_2(N(0,1)‖N(0,4)): formula=0.4133  quadrature=0.4133",
        "D_2(N(1,1)‖N(0,4)): formula=0.5562  quadrature=0.5562",
        "finite only for alpha < 1.3333; D_1.2=1.5976 (quadrature 1.5976)  D_2=inf",
        # Exercise 3
        "H^2=0.1317  Bhattacharyya coefficient=0.9342  -2 ln(1 - H^2/2)=0.1362  D_0.5 directly=0.1362",
        "D_0.5 at mu=1 is 0.2500; at mu=1/sqrt(2) is 0.1250",
        # Exercise 4
        "gamma=1: m(6)=0.2500  TV(P,M)=0.2500  E_gamma=0.3333",
        "gamma=1.5: m(6)=0.3333  TV(P,M)=0.1667  E_gamma=0.2500",
        "gamma=2: m(6)=0.4000  TV(P,M)=0.1000  E_gamma=0.1667",
        "gamma=2.5: m(6)=0.4545  TV(P,M)=0.0455  E_gamma=0.0833",
        "gamma=3: m(6)=0.5000  TV(P,M)=0.0000  E_gamma=0.0000",
        # Exercise 5
        "gamma=1: raw E_gamma=0.3333  parity E_gamma=0.2000",
        "gamma=1.2: raw E_gamma=0.3000  parity E_gamma=0.1000",
        "gamma=2: raw E_gamma=0.1667  parity E_gamma=0.0000",
        # Exercise 6
        "delta=0.2: ratio=1.0285  1 + 2delta^2/3=1.0267",
        "delta=0.1: ratio=1.0068  1 + 2delta^2/3=1.0067",
        "delta=0.01: ratio=1.0001  1 + 2delta^2/3=1.0001",
        # Exercise 7
        "kl(0.6, 0.5)=0.02014  Pinsker: n >= 64  Bretagnolle-Huber: n >= 51",
        "exact TV at n=64: 0.5805  exact TV first reaches 0.8 at n=162",
        # Exercise 8
        "n=1: 0.1000  n=2: 0.1100  n=3: 0.1480  n=4: 0.1627  n=16: 0.3143  n=64: 0.5805  n=256: 0.8930",
    ):
        assert expected in out
    assert list(tmp_path.iterdir()) == []
