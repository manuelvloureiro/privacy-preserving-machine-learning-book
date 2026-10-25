"""code/part05-dp-ml/solutions_user_level.py runs and reproduces the numbers in book-exercises/part05-dp-ml/07-user-level.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml"


def test_solutions_user_level_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_user_level.py", tmp_path)
    for expected in (
        "eps=1, delta=1e-6, k=2: eps_k=2  delta_k=3.718e-06",
        "eps=1, delta=1e-6, k=5: eps_k=5  delta_k=8.579e-05",
        "eps=1, delta=1e-6, k=10: eps_k=10  delta_k=1.282e-02",
        "record delta 10^-5: record eps 6.71  k=2: user delta 10^-2.1  k=5: user delta 10^6.7  k=10: user delta 10^21.2",
        "record delta 10^-10: record eps 9.78  k=2: user delta 10^-5.8  k=5: user delta 10^7.0  k=10: user delta 10^28.2",
        (
            "record delta 10^-20: record eps 14.31  k=2: user delta 10^-13.8  k=5: user delta 10^4.9  "
            "k=10: user delta 10^35.9"
        ),
        (
            "record delta 10^-50: record eps 24.94  k=2: user delta 10^-39.2  k=5: user delta 10^-6.7  "
            "k=10: user delta 10^47.5"
        ),
        (
            "record delta 10^-100: record eps 41.02  k=2: user delta 10^-82.2  k=5: user delta 10^-28.7  "
            "k=10: user delta 10^60.3"
        ),
        "k=2, q=0.01, sigma=1: eps(2) exact 6.992273e-04  enumeration 6.992273e-04  rate kq 6.870766e-04  ratio 1.0177",
        "k=5, q=0.01, sigma=1: eps(2) exact 4.626224e-03  enumeration 4.626224e-03  rate kq 4.286504e-03  ratio 1.0793",
        (
            "k=10, q=0.01, sigma=1: eps(2) exact 7.990343e+00  enumeration 7.990343e+00  rate kq 1.703686e-02  "
            "ratio 469.0032"
        ),
        "user-level DP-SGD, q=0.1, T=1000, eps=4: sigma=3.78",
        "k=2: per-record clipping, summed: sigma=7.56  check at alpha=3: 1.094286e-03 vs 1.094286e-03",
        "k=5: per-record clipping, summed: sigma=18.9  check at alpha=3: 1.094286e-03 vs 1.094286e-03",
        "user with 1 records, |g|=0.01, C=0.5: average-then-clip 0.01  sum-then-clip 0.01",
        "user with 10 records, |g|=0.01, C=0.5: average-then-clip 0.01  sum-then-clip 0.1",
        "user with 348 records, |g|=0.01, C=0.5: average-then-clip 0.01  sum-then-clip 0.5",
        "Gaussian noise for one unit of sensitivity at (1, 1e-5): 3.731",
        "cap k=1: kept=1000  bias=-9000  noise sd=3.7  rmse=9000.0",
        "cap k=10: kept=5063  bias=-4937  noise sd=37.3  rmse=4937.1",
        "cap k=50: kept=8437  bias=-1563  noise sd=186.5  rmse=1574.1",
        "cap k=100: kept=9214  bias=-786  noise sd=373.1  rmse=870.0",
        "cap k=200: kept=9756  bias=-244  noise sd=746.1  rmse=785.0",
        "cap k=348: kept=10000  bias=0  noise sd=1298.3  rmse=1298.3",
        "best cap k=158: kept=9625  rmse=698.6",
    ):
        assert expected in out, expected
