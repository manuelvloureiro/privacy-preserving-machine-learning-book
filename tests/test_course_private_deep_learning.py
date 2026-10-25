"""code/part04-learning-statistics/private_deep_learning.py runs, writes its figure, and reproduces the numbers in book-course/part04-learning-statistics/03-private-deep-learning.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part04-learning-statistics"


def test_private_deep_learning_matches_chapter(tmp_path: Path) -> None:
    out = run(CODE / "private_deep_learning.py", tmp_path)
    for expected in (
        "epochs=100 T=10000: composition eps=22.08  accountant eps=1.26",
        "epochs=400 T=40000: composition eps=61.41  accountant eps=2.58",
        "eps=1 delta=1e-05 beta=0.01: threshold=23.64  teachers=170",
    ):
        assert expected in out
    assert (tmp_path / "dpdl-accounting.svg").exists()
