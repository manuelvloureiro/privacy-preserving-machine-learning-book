"""code/part01-prerequisites/hypothesis_testing.py runs, writes its figures, and reproduces the numbers in book-full/part01-prerequisites/04-hypothesis-testing.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part01-prerequisites"


def test_hypothesis_testing_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "hypothesis_testing.py", tmp_path)
    for expected in (
        # Toolkit and simple-vs-simple
        "likelihood ratios q/p: 0:0.250  1:0.667  2:1.500  3:4.000",
        "[reject {3}     ] alpha=0.10  beta=0.60  power=0.40",
        "[reject {2,3}   ] alpha=0.30  beta=0.30  power=0.70",
        "[reject {1,2,3} ] alpha=0.60  beta=0.10  power=0.90",
        "[reject {0}     ] alpha=0.40  beta=0.90  power=0.10",
        "[reject {0,3}   ] alpha=0.50  beta=0.50  power=0.50",
        # Neyman-Pearson
        "[NP alpha=0.25] eta=1.50  gamma=0.75  alpha=0.250  power=0.625  beta=0.375",
        "qualifying=11727  best power=0.587",
        "[TV ham/spam] 0.40",
        # Inverse normal
        "Phi^-1(0.95)=1.6449  Phi^-1(0.999)=3.0902  Phi^-1(0.975)=1.9600  Phi^-1(0.99)=2.3263  Phi^-1(0.9)=1.2816",
        # Gaussian trade-off
        "[mu=1 alpha=0.05] threshold=1.6449  beta=0.7405  power=0.2595",
        "[mu=1 alpha=0.001] threshold=3.0902  beta=0.9817  power=0.0183",
        "[mu=2 alpha=0.05] threshold=1.6449  beta=0.3612  power=0.6388",
        "[mu=2 alpha=0.001] threshold=3.0902  beta=0.8622  power=0.1378",
        "[simulated mu=2 alpha=0.05] alpha=0.0503  beta=0.3617",
        "T(0.05)=0.7405  T(T(0.05))=0.0500",
        "effect 0.25] n=209",
        # Laplace trade-off
        "[breakpoints] e^-kappa/2=0.1839",
        "[Laplace kappa=1 alpha=0.05] beta=0.8641  power=0.1359",
        "[Laplace kappa=1 alpha=0.25] beta=0.3679  power=0.6321",
        "[Laplace kappa=1 alpha=0.75] beta=0.0920",
        "[simulated Laplace alpha=0.25] alpha=0.2498  beta=0.3685",
        "[TV Laplace kappa=1] 0.3935  matching Gaussian mu=1.0301",
        "[same TV, alpha=0.001] Laplace power=2.718e-03  Gaussian power=1.969e-02",
        "[same TV, alpha=1e-06] Laplace power=2.718e-06  Gaussian power=9.829e-05",
        # Best advantage
        "TV formula=0.3829  grid max gap=0.3829 at alpha=0.3085",
        "TV formula=0.3935  grid max gap=0.3935 at alpha=0.3033",
        # AUC
        "AUC=0.7602  TPR@1e-3=0.0183  TPR@0.1=0.3891  empirical AUC=0.7605",
        "[B: N(0,1) vs N(1.5811,4)] AUC=0.7602  TPR@1e-3=0.2253  TPR@0.1=0.5595  empirical AUC=0.7601",
        "AUC=0.5250  TPR@1e-3=0.0509  equal-AUC Gaussian mu=0.0887 TPR@1e-3=0.0013",
        # Bayesian view
        "alpha=0.05] LR of a positive=12.8  posterior=0.114",
        "alpha=0.001] LR of a positive=137.8  posterior=0.582",
        "threshold=3.2976  alpha=4.88e-04  power=0.0972",
        "Bayes error=0.3085",
        # Finite trials
        "[CP k=10 n=10000] [0.00048, 0.00184]",
        "[CP k=100 n=100000] [0.00081, 0.00122]",
        "[Wald k=10 n=10000] [0.00038, 0.00162]",
        "half-width=0.0136",
        "upper=9.981e-04",
        "n needed for 1e-3: 2995",
        "reps=2000] 0.9720",
        "FP=10  FPR CI=[0.00048, 0.00184]  TP=1350  TPR=0.1350  TPR CI=[0.1284, 0.1419]",
        # Exercises
        "n=10: 0.2740  n=100: 0.3796  n=1000: 0.4537",
        "T(0)=0.8  TV=0.2",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "hypothesis_testing_errors.svg",
        "hypothesis_testing_discrete_tradeoff.svg",
        "hypothesis_testing_gaussian_tradeoff.svg",
        "hypothesis_testing_laplace_tradeoff.svg",
        "hypothesis_testing_tv_advantage.svg",
        "hypothesis_testing_auc_low_fpr.svg",
        "hypothesis_testing_posterior.svg",
        "hypothesis_testing_rate_ci.svg",
    }
