"""code/part04-learning-statistics/mean_estimation.py runs, writes its figures, and reproduces the numbers in book-course/part04-learning-statistics/04-mean-estimation.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part04-learning-statistics"


def test_mean_estimation_matches_chapter(tmp_path: Path) -> None:
    out = run(CODE / "mean_estimation.py", tmp_path)
    for expected in (
        "n=4000 eps=1 beta=0.002: tau=5.51  sampling=0.0588",
        "R=1e6: naive bound=3107",
        "R=1e6: T=18  needs n>=3580",
        "R=1e6: shrinking bound=0.098",
        "R=1e6: m=181360  needs n>293  histogram bound=0.162",
        "R=1e+06: nonprivate=0.0250  naive=1050.8814  histogram=0.0461  shrinking=0.0295",
        "best tau=sqrt(eps n/(2 ln(1/beta)))=25.8  bound=0.150",
    ):
        assert expected in out
    for name in ("mean-gaussian.svg", "mean-heavy.svg"):
        assert (tmp_path / name).exists()
