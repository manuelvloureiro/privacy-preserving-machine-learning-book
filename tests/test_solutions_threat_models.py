"""code/part02-attacks/solutions_threat_models.py runs and reproduces the numbers in book-exercises/part02-attacks/01-threat-models.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-attacks"


def test_solutions_threat_models_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_threat_models.py", tmp_path)
    for expected in (
        # Exercise 1
        "prior 0.5: bound=0.6915  best=0.6915",
        "prior 0.1: bound=0.9383  best=0.9013",
        # Exercise 2
        "m=10: TV by sum=0.2461  central weight=0.2461  Stirling=0.2523",
        "m=100: TV by sum=0.0796  central weight=0.0796  Stirling=0.0798",
        # Exercise 3
        "normal approximation m=250.3  exact smallest m=250  TV there=0.0500  normal=0.0500",
        # Exercise 4
        "m=1: fraction recovered=1.0000",
        "m=2: fraction recovered=0.7500",
        "m=3: fraction recovered=0.7500",
        "m=10: fraction recovered=0.6230",
        # Exercise 5
        "TPR=0.001999  FPR=0.001000  advantage=0.000999  precision at prior 1/2=0.6666",
        "simulated: TPR=0.001945  FPR=0.001004",
        # Exercise 6
        "mean in=0.5000  sd in=0.7106  sd out=0.7071",
        "normal approximation=0.2763  simulated=0.2771",
        # Exercise 7
        "TV of mixtures=0.5000  rest known=1.0000",
    ):
        assert expected in out
    assert list(tmp_path.iterdir()) == []
