"""code/part05-dp-ml/user_level.py runs, writes its figures, and reproduces the numbers in book-full/part05-dp-ml/07-user-level.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml"


def test_user_level_matches_chapter(tmp_path: Path) -> None:
    out = run(CODE / "user_level.py", tmp_path)
    for expected in (
        # The population
        (
            "users=1000  records=10000  mean=10  median=4  largest=348  users with 1 record=193  "
            "users with >= 10 records=244  users with >= 50 records=30"
        ),
        (
            "largest 10 users hold 17.9% of the records; users with >= 50 records hold 30.6%; "
            "the smallest half of the users hold 10.3%"
        ),
        "label imbalance |mean y| in the 10 largest users: 0.21 (whole set 0.06)",
        # Accountant checks
        "k=1 against the subsampled Gaussian, alpha=4.1: 3.733633e-04 vs 3.733633e-04",
        "k=2, alpha=2: integral 6.992273e-04  pair-moment formula 6.992273e-04  reverse direction 6.119238e-04",
        "k=5, alpha=2: integral 4.626224e-03  pair-moment formula 4.626224e-03  reverse direction 3.412123e-03",
        "sampled together, k=2, alpha=3: binomial sum 8.219438e-02  same as sigma/k: 8.219438e-02",
        # A user with k records
        "k=1: user eps=6.72 (alpha=4)  group privacy 6.71 (record eps 6.71)  one record at rate kq: 6.71",
        "k=2: user eps=15.54 (alpha=2.5)  group privacy 18.61 (record eps 9.30)  one record at rate kq: 15.32",
        "k=5: user eps=52.36 (alpha=1.65)  group privacy 118.12 (record eps 23.62)  one record at rate kq: 50.11",
        "k=10: user eps=146.20 (alpha=1.3)  group privacy inf (record eps inf)  one record at rate kq: 134.50",
        "k=50: user eps=2161.43 (alpha=1.05)  group privacy inf (record eps inf)  one record at rate kq: 1673.08",
        # Noise
        (
            "eps=1: record-level sigma=1.52  (a) all records, k=348: sigma=449  (b) user-level: sigma=12.9  "
            "(c) cap 2: sigma=14.4"
        ),
        (
            "eps=2: record-level sigma=1.03  (a) all records, k=348: sigma=237  (b) user-level: sigma=6.89  "
            "(c) cap 2: sigma=7.62"
        ),
        (
            "eps=4: record-level sigma=0.778  (a) all records, k=348: sigma=128  (b) user-level: sigma=3.78  "
            "(c) cap 2: sigma=4.18"
        ),
        (
            "eps=8: record-level sigma=0.616  (a) all records, k=348: sigma=70.3  (b) user-level: sigma=2.18  "
            "(c) cap 2: sigma=2.4"
        ),
        # Contribution bounding
        "cap k=1: records kept=1000 (10.0%)  q=0.1000  kq=0.100  sigma=3.78  test accuracy 0.8917",
        "cap k=2: records kept=1807 (18.1%)  q=0.0553  kq=0.111  sigma=4.18  test accuracy 0.8920",
        "cap k=5: records kept=3506 (35.1%)  q=0.0285  kq=0.143  sigma=5.34  test accuracy 0.8790",
        "cap k=10: records kept=5063 (50.6%)  q=0.0198  kq=0.198  sigma=7.34  test accuracy 0.8276",
        "cap k=20: records kept=6631 (66.3%)  q=0.0151  kq=0.302  sigma=11.2  test accuracy 0.7562",
        "cap k=50: records kept=8437 (84.4%)  q=0.0119  kq=0.593  sigma=21.8  test accuracy 0.6401",
        # Strategies
        "test points nearest to a heavy user's record: 68.3%",
        "eps=1: (a) example-level, all records 0.5318  (b) user-level 0.8065  (c) cap 2 0.7084",
        "eps=2: (a) example-level, all records 0.5582  (b) user-level 0.9028  (c) cap 2 0.8374",
        "eps=4: (a) example-level, all records 0.5713  (b) user-level 0.9347  (c) cap 2 0.8810",
        "(b) user-level: near heavy users 0.9361  elsewhere 0.9319",
        "eps=8: (a) example-level, all records 0.5478  (b) user-level 0.9442  (c) cap 2 0.9333",
        "non-private, 1000 steps: test accuracy 0.9612  near heavy users 0.9607  elsewhere 0.9622",
        "user-level DP-SGD on one random record per user, eps=4: test accuracy 0.9070",
        # Heavy users
        "share of the expected update under example-level training 30.6%, under user-level training at most 3.0%",
    ):
        assert expected in out, expected
    for name in ("ulvl_population.svg", "ulvl_user_epsilon.svg", "ulvl_strategies.svg"):
        assert (tmp_path / name).exists()
