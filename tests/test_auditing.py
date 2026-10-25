"""code/part05-dp-ml/auditing.py runs, writes its figures, and reproduces the numbers in book-full/part05-dp-ml/10-auditing.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml"


def test_auditing_matches_chapter(tmp_path: Path) -> None:
    out = run(CODE / "auditing.py", tmp_path)
    for expected in (
        # The claim and the toolkit
        "q=0.01, sigma=1, T=10000, delta=1e-5: eps=6.71",
        "k=10, n=10000: 97.5% one-sided bounds [0.00048, 0.00184]",
        "k=0, n=3000: 95% upper bound 9.9808e-04",
        # From an attack to a bound
        "point estimate fpr=0.001, tpr=0.5: eps=6.21",
        "fp=1/1000, tp=500/1000: fpr_hi=0.00556 tpr_lo=0.4685 eps_lo=4.43",
        "fp=10/10000, tp=5000/10000: fpr_hi=0.00184 tpr_lo=0.4902 eps_lo=5.59",
        "perfect attack, n=100 per hypothesis: eps_lo=3.28",
        "perfect attack, n=1000 per hypothesis: eps_lo=5.60",
        "perfect attack, n=10000 per hypothesis: eps_lo=7.90",
        "perfect attack, n=100000 per hypothesis: eps_lo=10.21",
        "perfect attack, n=1000000 per hypothesis: eps_lo=12.51",
        "certify eps=1.00: 12",
        "certify eps=2.00: 30",
        "certify eps=4.00: 204",
        "certify eps=6.71: 3038",
        "certify eps=8.00: 10999",
        # Canaries and where eps lives
        "CLT mu=1.3108  eps at 1e-5=6.01  touching point FPR=8.1e-08 TPR=4.3e-05",
        "sum of projections: mu about q*sqrt(T)/sigma=1.00",
        "sum of projections, with the extra variance of K: 0.9951  eps at 1e-5=4.35",
        "GDP mu=1.3108 at FPR=0.01: TPR=0.1549  ln(TPR/FPR)=2.74",
        "GDP mu=1.3108 at FPR=0.001: TPR=0.0376  ln(TPR/FPR)=3.63",
        "GDP mu=1.3108 at FPR=0.0001: TPR=0.0080  ln(TPR/FPR)=4.38",
        # Tight audits
        "5 false positives: FPR=1.0e-04 TPR=0.0126 ln((TPR-delta)/FPR)=4.84  local mu=1.481",
        "10 false positives: FPR=2.0e-04 TPR=0.0203 ln((TPR-delta)/FPR)=4.62  local mu=1.492",
        "50 false positives: FPR=1.0e-03 TPR=0.0428 ln((TPR-delta)/FPR)=3.76  local mu=1.371",
        "500 false positives: FPR=1.0e-02 TPR=0.1530 ln((TPR-delta)/FPR)=2.73  local mu=1.303",
        "5000 false positives: FPR=1.0e-01 TPR=0.5074 ln((TPR-delta)/FPR)=1.62  local mu=1.300",
        "n=1000 per hypothesis: eval fp=433/500 tp=496/500  eps_lo=1.64  mu_lo=0.795 (fp=433, tp=496)  GDP eps=3.36  ceiling=4.91",
        "n=3000 per hypothesis: eval fp=1/1500 tp=49/1500  eps_lo=1.88  mu_lo=1.064 (fp=1010, tp=1432)  GDP eps=4.70  ceiling=6.01",
        "n=10000 per hypothesis: eval fp=6/5000 tp=254/5000  eps_lo=2.84  mu_lo=1.147 (fp=333, tp=1977)  GDP eps=5.13  ceiling=7.21",
        "n=50000 per hypothesis: eval fp=3/25000 tp=368/25000  eps_lo=3.63  mu_lo=1.185 (fp=18, tp=828)  GDP eps=5.33  ceiling=8.82",
        "sum audit, n=10000 per hypothesis: eps_lo=1.97",
        "sum audit, n=100000 per hypothesis: eps_lo=3.26",
        "sum audit, n=1000000 per hypothesis: eps_lo=2.27",
        # One run
        "m=r=100, v=75, delta=0: eps_lo=0.702",
        "m=r=100, v=75, delta=1e-4: eps_lo=0.699",
        "m=1000, r=100, v=75, delta=1e-4: eps_lo=0.673",
        "all 100 guesses right: eps_lo=3.49 (delta=0)",
        "all 1000 guesses right: eps_lo=5.81 (delta=0)  5.78 (delta=1e-5)",
        "all 10000 guesses right: eps_lo=8.11 (delta=0)  7.83 (delta=1e-5)",
        (
            "m=1000: r=20: 20 right, eps_lo=1.21  r=50: 50 right, eps_lo=2.22  r=100: 92 right, eps_lo=1.58  "
            "r=200: 185 right, eps_lo=1.88  r=500: 425 right, eps_lo=1.43  r=1000: 745 right, eps_lo=0.90"
        ),
        "m=5000: r=20: 20 right, eps_lo=0.00",
        "r=500: 479 right, eps_lo=2.53",
        # Black box
        (
            "DP-SGD sigma=1: included=482  test acc=0.9645  mean loss in=5.565 out=6.013  r=20: 13  r=50: 28  "
            "r=100: 59  r=200: 119  r=500: 293  r=1000: 556  best r=500: 293 right, eps_lo=0.13"
        ),
        (
            "clipping only, sigma=0: included=493  test acc=0.9638  mean loss in=5.511 out=6.155  r=20: 13  r=50: 36  "
            "r=100: 67  r=200: 125  r=500: 307  r=1000: 589  best r=500: 307 right, eps_lo=0.24"
        ),
        (
            "non-private: included=481  test acc=0.9155  mean loss in=0.023 out=9.041  r=20: 16  r=50: 43  "
            "r=100: 91  r=200: 188  r=500: 482  r=1000: 967  best r=1000: 967 right, eps_lo=2.96"
        ),
        # Bugs
        "per-step ceiling on mu with 5000 trials per hypothesis: 6.359",
        "noise=1C  per-step mu_lo=0.886 (claimed 1)  accountant at sigma=1/mu_lo: eps=5.38",
        "noise=0.5C  per-step mu_lo=1.745 (claimed 1)  accountant at sigma=1/mu_lo: eps=29.27",
        "noise=0.01C  per-step mu_lo=6.359 (claimed 1)  accountant at sigma=1/mu_lo: eps=4740.35",
        "miscounted n (accountant uses q=0.005): claimed eps=3.06",
        "composed at the claimed q=0.005: eps=2.47  (not refuted)",
        "end-to-end LR audit (n=50000): eps_lo=3.63  refuted=True",
        "end-to-end sum audit (n=10000): fp=1 tp=5000/5000  eps_lo=7.21  claimed 6.71",
    ):
        assert expected in out, expected
    for name in ("audit_region.svg", "audit_trials.svg", "audit_one_run.svg", "audit_bugs.svg"):
        assert (tmp_path / name).exists()
