"""code/part04-privacy-accounting/solutions_choosing_an_accountant.py runs and reproduces the numbers in book-exercises/part04-privacy-accounting/09-choosing-an-accountant.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part04-privacy-accounting"


def test_solutions_choosing_an_accountant_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_choosing_an_accountant.py", tmp_path)
    for expected in (
        # Exercise 1
        "k=1: RDP improved=0.9555 (alpha=9.9)  RDP optimal=0.4899  PLD upper=0.1995  conversion slack=0.4656  curve slack=0.2905",
        "k=100: RDP improved=1.2141 (alpha=8.8)  RDP optimal=1.1538  PLD upper=0.7185  conversion slack=0.0603  curve slack=0.4353",
        "k=10000: RDP improved=6.7127 (alpha=4.1)  RDP optimal=6.7127  PLD upper=6.2377  conversion slack=0.0000  curve slack=0.4750",
        "k=1: ln(1/delta)/(alpha-1) at the best order = 1.2936",
        # Exercise 2
        "RDP improved: k=2195 (eps=3.0000, at k+1 3.0006)",
        "PLD upper: k=2643 (eps=2.9995, at k+1 3.0001)",
        "CLT: k=3009  PLD lower at that k=3.1823",
        "epochs: RDP 21.95  PLD 26.43  extra training from PLD 20%",
        # Exercise 3
        "delta=1e-05: PLD upper=6.2377  RDP improved=6.7127",
        "delta=1e-06: PLD upper=6.9574  RDP improved=7.4140",
        "delta=1e-08: PLD upper=8.2356  RDP improved=8.6650",
        # Exercise 4
        "RDP improved=4.9544  PLD upper=4.5981  lower=4.4981",
        "phases alone (PLD upper): sigma=1 4.2268  sigma=2 1.5024  sum 5.7292",
        # Exercise 5
        "CLT=1.6177  PLD lower=1.8232  PLD upper=1.8332  RDP improved=2.1014  RDP Mironov=2.5380  advanced=1258.64",
    ):
        assert expected in out, expected
