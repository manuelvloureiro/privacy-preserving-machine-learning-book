"""code/part03-dp-foundations/solutions_discrete_mechanisms.py runs and reproduces the numbers in book-exercises/part03-dp-foundations/08-discrete-mechanisms.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations"


def test_solutions_discrete_mechanisms_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_discrete_mechanisms.py", tmp_path)
    for expected in (
        # Exercise 1
        "gamma=1/2: expected draws=1.6487  simulated=1.6597",
        "gamma=1: expected draws=3.0862  simulated=3.0854",
        "gamma=5/2: expected draws=3.9414  simulated=3.9233",
        "just below 1: 2.7183  at 1: 3.0862",
        # Exercise 2
        "r=e^-1: Pr[G1-G2=0]=0.4621  (1-r)/(1+r)=0.4621",
        # Exercise 3
        "scale=0.5: 2r/(1-r)^2=0.3620  sinh form=0.3620  2 scale^2=0.5000  gap=0.1380",
        "scale=1: 2r/(1-r)^2=1.8413  sinh form=1.8413  2 scale^2=2.0000  gap=0.1587",
        "scale=10: 2r/(1-r)^2=199.8334  sinh form=199.8334  2 scale^2=200.0000  gap=0.1666",
        "scale=100: 2r/(1-r)^2=19999.8333  sinh form=19999.8333  2 scale^2=20000.0000  gap=0.1667",
        # Exercise 4
        "Pr[|Z|>=3]=0.0728  continuous Pr[|X|>=3]=0.0498  Pr[|Z|>=4]=0.0268",
        "95% radius: eps=1: 3  eps=0.1: 30  continuous eps=0.1: 29.96",
        # Exercise 5
        "sigma^2=1/4: closed form=0.5185  by summation=0.5185",
        "sigma^2=1: closed form=0.5418  by summation=0.5418",
        "sigma^2=4: closed form=0.6629  by summation=0.6629",
        "sigma^2=100: closed form=0.7532  by summation=0.7532",
        "sigma^2=10000: closed form=0.7601  by summation=0.7601",
        "limit sqrt(pi/2) e^-1/2=0.7602",
        # Exercise 6
        (
            "sigma^2=0.25 each: Pr[sum=0]=0.6414  N_Z(0,0.5) at 0=0.5641  var of sum=0.4300  "
            "var of N_Z(0,0.5)=0.4990  TV=8.01e-02"
        ),
        (
            "sigma^2=1 each: Pr[sum=0]=0.2821  N_Z(0,2) at 0=0.2821  var of sum=2.0000  var of N_Z(0,2)=2.0000  "
            "TV=5.17e-05"
        ),
        "sigma^2=4 each: Pr[sum=0]=0.1410",
        # Exercise 7
        "p=12: P(S)=0.4791  reverse=0.2033",
    ):
        assert expected in out
    assert list(tmp_path.iterdir()) == []
