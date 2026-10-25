"""code/part03-query-release/packing_lower_bounds.py runs, writes its figure, and reproduces the numbers in book-course/part03-query-release/04-packing-lower-bounds.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-query-release"


def test_packing_lower_bounds_matches_chapter(tmp_path: Path) -> None:
    out = run(CODE / "packing_lower_bounds.py", tmp_path)
    for expected in (
        "k=1000 eps=1: n >= 6.21",
        "d=1000 alpha=0.1 eps=1: lower=3458  upper=76010  ratio=22.0",
        "k=1000000 alpha=0.01 eps=1: lower=607  upper=2902  ratio=4.8",
        "k=1e+12: upper/lower=4.37",
        "Laplace k=10000 n=1981: max error <= alpha in 217 of 400 runs (54%)",
        "packing k=5 n=20 alpha=0.1: t/n=0.25  disjoint=True",
    ):
        assert expected in out
    assert (tmp_path / "pack-histograms.svg").exists()
