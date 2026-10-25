"""code/appendix-derivations/apx_probability.py runs, writes its figures, and reproduces the numbers in book-full/appendix-derivations/05-probability.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "appendix-derivations"


def test_appendix_probability_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "apx_probability.py", tmp_path)
    for expected in (
        "[Poisson 2 + 3] largest |convolution - Poisson(5)| over k < 40 = 3.3e-16",
        "[Exp - Exp] largest |overlap integral - Lap(1) density| = 7.2e-05",
        "largest gap to Gamma(5, 1) = 1.1e-03",
        "[Normal 1 + 4] largest |convolution - N(0, 5) density| = 1.2e-12",
        "var u=0.9983  var v=1.0010  corr=-0.0005  Pr[both > 1]=0.0251  product=0.0252",
        "(ε/4)(1 + ε|z|)e^(-ε|z|)| = 7.5e-09",
        "is a most likely count in 480 of 480 pairs",
        "[Stirling n=10] ln n! = 15.1044  crude [14.0259, 16.3768]  sharp remainder=8.331e-03  1/(12n)=8.333e-03",
        "[Stirling n=1] ln n! = 0.0000  crude [0.0000, 0.3863]  sharp remainder=8.106e-02",
        "[Wallis j=5] C(2j,j) 4^-j sqrt(πj) = 0.97535",
        "[Wallis j=50] C(2j,j) 4^-j sqrt(πj) = 0.99750",
        "[Reconstruction m=10] exact=0.62305  closed=0.62305  approx=0.62616",
        "[Reconstruction m=11] exact=0.62305  closed=0.62305",
        "[Reconstruction m=100] exact=0.53979  closed=0.53979  approx=0.53989",
        "[Reconstruction m=1000] exact=0.51261  closed=0.51261  approx=0.51262",
        "[Shift TV Bin(10,0.5)] TV=0.2461  mode height=0.2461",
        "[Shift TV Bin(100,0.5)] TV=0.0796  mode height=0.0796",
        "largest npq |ln ratio| over npq >= 4, n <= 20000 = 0.187",
        "[Peak p=0.5] max_k p_k sqrt(2πnpq)  n=10: 0.9754  n=100: 0.9975  n=1000: 0.9998",
        "[Peak p=0.05] max_k p_k sqrt(2πnpq)  n=10: 1.0344  n=100: 0.9834  n=1000: 0.9983",
        "|E h(S) - E h(Z)| = 0.0365  bound = 0.1220  largest step = 0.00122  per-swap bound = 0.00407",
        "E|X - 1|^3 = 2.4146  E|G|^3 = 1.5958",
        "[CDF gap n=30] L=4.408e-01  ε*=3.046  Lindeberg bound=1.6201  Berry–Esseen 0.48 L=0.2116",
        "[CDF gap n=100000000] L=2.415e-04  ε*=0.466  Lindeberg bound=0.2478",
        "[CDF gap n=30] exact largest gap = 0.0243",
        "[Norm d=1] Pr[‖Z‖ >= √d + t]  t=1: 4.55e-02  t=2: 2.70e-03  t=3: 6.33e-05",
        "[Norm d=1000] Pr[‖Z‖ >= √d + t]  t=1: 7.76e-02  t=2: 2.44e-03  t=3: 1.37e-05",
        "t=1: 6.07e-01  t=2: 1.35e-01  t=3: 1.11e-02",
        "t in [0, 6] = 0.498",
        "[Laplace diff t=8] formula=0.0872  integral=0.0872  simulated=0.0867",
        "[Threshold P=1000] t = 74.8",
        "[Threshold P=10000] t = 84.0",
        "[Laplace diff equal b=1, t=3] tail=0.0622  near-equal b1=1.001=0.0623",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "apx_prob_convolution.svg",
        "apx_prob_rotation.svg",
        "apx_prob_shift_tv.svg",
        "apx_prob_stirling.svg",
        "apx_prob_peak.svg",
        "apx_prob_lindeberg.svg",
        "apx_prob_smoothing.svg",
        "apx_prob_norm_tails.svg",
        "apx_prob_norm_density.svg",
        "apx_prob_laplace_diff.svg",
    }
