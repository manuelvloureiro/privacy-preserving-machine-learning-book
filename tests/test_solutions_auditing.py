"""code/part05-dp-ml/solutions_auditing.py runs and reproduces the numbers in book-exercises/part05-dp-ml/10-auditing.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml"


def test_solutions_auditing_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_auditing.py", tmp_path)
    for expected in (
        "FPR=0.05, TPR=0.95: direct=2.94  mirror=2.94  bound=2.94",
        "FPR=0.001, TPR=0.05: direct=3.91  mirror=0.05  bound=3.91",
        "FPR=0.5, TPR=0.999: direct=0.69  mirror=6.21  bound=6.21",
        "n=100: exact=3.28  approx ln(n/ln 40)=3.30",
        "n=1000: exact=5.60  approx ln(n/ln 40)=5.60",
        "n=10000: exact=7.90  approx ln(n/ln 40)=7.91",
        "n=100000: exact=10.21  approx ln(n/ln 40)=10.21",
        "eps=6.71: exact trials=3038  approx e^eps ln 40=3035",
        "mu=1.3108: eps=6.01  touching FPR=8.1e-08  TPR=4.3e-05  OUT trials for 10 false positives=1.2e+08",
        "mu=0.5000: eps=1.99  touching FPR=1.1e-05  TPR=9.3e-05  OUT trials for 10 false positives=8.8e+05",
        "mu=3.0000: eps=16.68  touching FPR=8.4e-13  TPR=2.5e-05  OUT trials for 10 false positives=1.2e+13",
        "sigma=0.5: mu_LR/mu_sum=3.661  step factor=13.40",
        "sigma=1: mu_LR/mu_sum=1.311  step factor=1.72",
        "sigma=2: mu_LR/mu_sum=1.066  step factor=1.14",
        "sigma=4: mu_LR/mu_sum=1.016  step factor=1.03",
        "pure DP, eps=6.71: canaries needed=2467",
        "delta=1e-06: canaries needed=2483",
        "delta=1e-05: canaries needed=2648",
        "m=r=v=2000, delta=1e-5: eps_lo=6.45",
        "m=r=v=2500, delta=1e-5: eps_lo=6.66",
        "m=r=v=5000, delta=1e-5: eps_lo=7.28",
        "p-value of 75/100 at eps=ln 3, delta=0: 0.553",
        "randomized response, eps=ln 3: mean correct out of 100 = 75.01",
        "p-value of 85/100 at eps=ln 3, delta=0: 0.0111",
    ):
        assert expected in out, expected
