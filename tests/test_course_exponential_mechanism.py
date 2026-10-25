"""code/part02-dp-basics/exponential_mechanism.py runs, writes its figure, and reproduces the numbers in book-course/part02-dp-basics/05-exponential-mechanism.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-dp-basics"


def test_exponential_mechanism_matches_chapter(tmp_path: Path) -> None:
    out = run(CODE / "exponential_mechanism.py", tmp_path)
    for expected in (
        "n=1000 eps=0.5 m=500: gamma=267.40  OPT=266.88",
        "guarantee at 95%=228.56  loss allowed=38.84",
        "mechanism: mean revenue=263.86  5th percentile=256.52",
        "loss at 5th percentile=10.88  guarantee/actual loss=3.6",
        "ln(2|C|/beta)=17.552",
        "sampling term=14041  privacy term=1404",
    ):
        assert expected in out
    assert (tmp_path / "em-pricing.svg").exists()
