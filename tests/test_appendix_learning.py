"""code/appendix-derivations/apx_learning.py runs, writes its figures, and reproduces the numbers in
book-full/appendix-derivations/13-private-learning.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "appendix-derivations"


def test_appendix_learning_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "apx_learning.py", tmp_path)
    for expected in (
        "mu(1, 1e-5) = 0.2681",
        (
            "[ERM strong n=10000] T=718515  sigma=0.6325  bound at T=2.2268e-03  closed form=2.2268e-03  "
            "uniform-average bound=8.0638e-03  lower=1.8991e-09"
        ),
        "[ERM measured n=300] T=647  measured=2.2640e-01  bound=2.4742e+00  ratio=10.9",
        "[ERM measured n=1000] T=7186  measured=1.3489e-02  bound=2.2268e-01  ratio=16.5",
        "[ERM measured n=3000] T=64667  measured=1.0754e-03  bound=2.4742e-02  ratio=23.0",
        "d=10: K=1.8402",
        "[ERM gap d=10 n=10000] lower=1.8991e-09  upper=2.2268e-03  ratio=1.17e+06",
        "[ERM gap d=1000 n=10000] lower=5.2144e-07  upper=2.2268e-01  ratio=4.27e+05",
        "[ERM gap d=10000 n=10000] lower=5.1991e-06  upper=2.2268e+00  ratio=4.28e+05",
        "ratio=8.59e+03",
        "ratio=8.6e+03",
        "lower=2.7835e-04  approx=8.7070e-04  measured=9.6417e-04  upper=7.2372e-03",
        "lower=3.2872e-07  approx=7.1266e-05  measured=7.1178e-05  upper=2.2221e-04",
        "output=7.5  objective=3.1",
        "[ERM objective per-draw inequalities] hold=True",
        (
            "q~=4.0050e-03  mu2=93.981  mu1=94.981  eps1=0.05936  eps2=0.05874  p-bound=4.5014e-03  term A=0.999477  "
            "term B=9.1545e-03  bound=1.2278e-03  rigorous=1.2278e-03  exact=2.2362e-04  condition=True"
        ),
        "p-bound/q~ = 1.1240  exp(2 sqrt(ln 1/q~)/sigma - 1/sigma^2) = 1.1240",
        "condition holds from gap 20; increasing on (0, q~] at all 57 such gaps: True; exact two-label below bound: True",
        "[PATE condition holds for every q~ below the limit] True",
        "[PATE smallest gap where the bound beats alpha/sigma^2] 116  ratio at 150=5.49",
        "terms=0.99503 0.0064 0.0008  sum=1.00223  average ratio off i*=0.72",
        "largest q~ = 3.6321e-01",
        "exact worst D=1.5188e-04  q~=2.7303e-03  bound=8.7544e-04  data-independent=5.0000e-03",
        "convex and increasing in q on every tested (alpha, eps0): True",
        (
            "gamma=0.02692  best lam_hat=100.95  one run=0.3842  tail=0.4904  gamma term=0.0358  mean term=0.5756  "
            "bound=1.4860  exact=0.3912  limit lam_hat->inf=1.4599"
        ),
        "[TNB RR order 2] grid bound=3.0561  limit=3.0299",
        "[TNB RR order 20] grid bound=1.1224  limit=1.0962",
        "[TNB chain] every step holds on random cases: True",
        "gamma=0.10000  closed form=2.7791  numeric=2.7791  lam*=5.7985  lam_hat=4.7985",
        "A2 aligned=1.232011  eps(2)=0.208648  A2 at 90 deg=1.021664  A2 at 60 deg=1.066665",
        "quadrature=1.516754  formula=1.516754",
        "[User random configurations below aligned] True",
        "M1*M2=0: 0.948825  (1,1): 0.112710  (1,2)+(2,1): 0.064501  (2,2): 0.025084  (5,5): 7.032e-03",
        "group=majority] descent=-0.01900  clipping=0.00000  curvature=0.00459  noise=1.156e-05  total=-0.01440",
        "group=minority] descent=-0.40000  clipping=0.35000  curvature=0.00459  noise=1.156e-05  total=-0.04540",
        "simulated=0.01707  formula=0.01709",
        "floor=5.9702e-03  expected change at the floor=0.0e+00",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "learn_erm_rates.svg",
        "learn_erm_reduction.svg",
        "learn_erm_sandwich.svg",
        "learn_pate_terms.svg",
        "learn_pate_q.svg",
        "learn_tuning_terms.svg",
        "learn_tuning_square.svg",
        "learn_user_angle.svg",
        "learn_user_pairs.svg",
        "learn_fair_terms.svg",
    }
