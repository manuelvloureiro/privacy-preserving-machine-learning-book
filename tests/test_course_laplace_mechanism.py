"""code/part02-dp-basics/laplace_mechanism.py runs, writes its figure, and reproduces the numbers in book-course/part02-dp-basics/02-laplace-mechanism.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-dp-basics"


def test_laplace_mechanism_matches_chapter(tmp_path: Path) -> None:
    out = run(CODE / "laplace_mechanism.py", tmp_path)
    for expected in (
        "eps=1 n=10000 beta=0.05: laplace bound=3.00e-04",
        "randomized response bound=0.0294  ratio=98",
        "k=100: counting bound=760.1  histogram bound=15.20",
        "histogram, simulated 95th percentile of the largest error=15.17",
    ):
        assert expected in out
    assert (tmp_path / "lap-mechanism.svg").exists()
