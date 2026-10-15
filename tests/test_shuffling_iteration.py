"""code/shuffling_iteration.py runs, writes its figures, and reproduces the numbers in 07-shuffling-iteration.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "book" / "part04-privacy-accounting" / "code"

# The public functions, called as chapter 35 would call them, plus closed-form checks.
PUBLIC_API_CHECK = """
import math
from shuffling_iteration import (clones_epsilon, hidden_state_limit_mu, noisy_gd_last_iterate_mu,
                                 noisy_gd_one_pass_mu, shuffled_rr_delta, shuffled_rr_epsilon,
                                 shuffled_rr_pair_epsilon, shuffled_rr_pmfs_general)
# the fast pair (others hold 0) agrees with the direct convolution
p, q = shuffled_rr_pmfs_general(1.0, 40, 0)
g = math.exp(0.3)
direct = max(float((p - g * q).clip(min=0).sum()), float((q - g * p).clip(min=0).sum()))
assert abs(direct - shuffled_rr_delta(0.3, 1.0, 40)) < 1e-12
# the general pair with m = 0 gives the same epsilon as the fast pair; m and n - 1 - m are mirror images
assert abs(shuffled_rr_pair_epsilon(2.0, 200, 0, 1e-6) - shuffled_rr_epsilon(2.0, 200, 1e-6)) < 1e-9
assert abs(shuffled_rr_pair_epsilon(2.0, 200, 3, 1e-6) - shuffled_rr_pair_epsilon(2.0, 200, 196, 1e-6)) < 1e-9
# exact shuffled randomized response is below the clone bound; the bound's condition can fail
assert shuffled_rr_epsilon(2.0, 10_000, 1e-6) < clones_epsilon(2.0, 10_000, 1e-6)
assert math.isinf(clones_epsilon(4.0, 10_000, 1e-6))
# no contraction: last iterate = composition; strict contraction: the limit; one pass with c = 1
assert abs(noisy_gd_last_iterate_mu(1.0, 25, 1.0, 1.0, 1) - 5.0) < 1e-12
assert abs(noisy_gd_last_iterate_mu(0.5, 200, 1.0, 1.0, 1) - hidden_state_limit_mu(0.5, 1.0, 1.0)) < 1e-12
assert abs(noisy_gd_one_pass_mu(1.0, 16, 2.0, 1.0, 5) - 1 / 8) < 1e-12
"""


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_shuffling_iteration_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "shuffling_iteration.py", tmp_path)
    for expected in (
        # Section 33.3: shuffled randomized response
        "others' count b x 16 = 9  6  1",
        "P(count) x 64 = 27  27  9  1   Q(count) x 64 = 9  33  19  3",
        "ratios P/Q = 3.0000  0.8182  0.4737  0.3333",
        "eps=0.5: shuffled delta=0.1900  one local report delta=0.3378",
        "eps=1: shuffled delta=0.0396  one local report delta=0.0704",
        "0.4226  0.4203  0.1413  0.0158   exact: 0.4219  0.4219  0.1406  0.0156   largest gap: 0.0015",
        "eps0=2: truth probability theta=0.8808",
        "eps0=4: truth probability theta=0.9820  expected 1-reports among 999 others holding 0: 18.0",
        "expected clones (n - 1) e^-eps0=18.3",
        "theta(1-theta)/(2 theta-1)^2=0.0190",
        "n=1000 eps0=1: m=0: 0.126614  largest over all m: 0.126614 at m=0",
        "n=1000 eps0=2: m=0: 0.323279  largest over all m: 0.323324 at m=2  excess 4.5e-05  m=499: 0.2858",
        "n=1000 eps0=4: m=0: 1.691006  largest over all m: 1.691006 at m=0",
        "n=10000 eps0=2: m=0: 0.087016  largest over m <= 30: 0.087020 at m=3  excess 4.1e-06",
        "n=100000 eps0=4: m=0: 0.084714  largest over m <= 30: 0.084716 at m=8  excess 2.2e-06",
        # Section 33.4: clones
        "clone part=[0.1667, 0.1667]  LO(0)=[0.875, 0.125]",
        "n=1000: eps0 must be <= 1.460",
        "n=10000: eps0 must be <= 3.763",
        "n=100000: eps0 must be <= 6.066",
        "n=1000 eps0=1: exact RR=0.1266  CLT approx=0.1212  clones numeric=0.1824  clones closed form=0.5662",
        "numeric/exact=1.44  closed/exact=4.47  closed/numeric=3.10",
        "numeric/exact=2.36",
        "Erlingsson at eps0=4, smallest over the figure's n=316..100000: 7.49e+06",
        "n=1000 eps0=2: exact RR=0.3233  CLT approx=0.2890  clones numeric=0.5455  clones closed form=n/a",
        "n=1000 eps0=4: exact RR=1.6910  CLT approx=0.9667  clones numeric=3.9900  clones closed form=n/a",
        "n=10000 eps0=1: exact RR=0.0357  CLT approx=0.0352  clones numeric=0.0530  clones closed form=0.2140",
        "sqrt(n)*exact=3.566",
        "n=10000 eps0=2: exact RR=0.0870  CLT approx=0.0844  clones numeric=0.1550  clones closed form=0.5009",
        "sqrt(n)*exact=8.702",
        "n=10000 eps0=4: exact RR=0.3146  CLT approx=0.2816  clones numeric=0.6009  clones closed form=n/a",
        "sqrt(n)*exact=31.464",
        "n=100000 eps0=1: exact RR=0.0101  CLT approx=0.0101  clones numeric=0.0153  clones closed form=0.0726  "
        "Erlingsson=0.4285",
        "closed/exact=7.15  closed/numeric=4.75",
        "n=100000 eps0=2: exact RR=0.0247  CLT approx=0.0244  clones numeric=0.0450  clones closed form=0.1862",
        "closed/exact=7.55  closed/numeric=4.13",
        "n=100000 eps0=4: exact RR=0.0847  CLT approx=0.0822  clones numeric=0.1698  clones closed form=0.5346",
        "numeric/exact=2.00  closed/exact=6.31  closed/numeric=3.15",
        "eps0=1: 1.46  eps0=2: 1.76  eps0=4: 1.96",
        # Section 33.5: iteration
        "T=4: even schedule cost=0.2500  exact D_2(N(0,T)||N(1,T))=0.2500",
        "T=10: even schedule cost=0.1000  exact D_2(N(0,T)||N(1,T))=0.1000",
        "uneven schedule a=(0.8, 0.2): cost=0.6800",
        "t=1: RDP eps(2)=0.000100  (eps, delta)=(0.0481, 1e-5) at alpha=512",
        "t=5000: RDP eps(2)=0.000200  (eps, delta)=(0.0707, 1e-5) at alpha=256",
        "t=9901: RDP eps(2)=0.010000  (eps, delta)=(0.4849, 1e-5) at alpha=49",
        "t=10000: RDP eps(2)=1.000000  (eps, delta)=(5.2985, 1e-5) at alpha=5.8",
        "one Gaussian mechanism, mu = 1, exact: eps=4.3772",
        "t=1 at alpha=512: RDP term=0.0256  conversion term=0.0225",
        "t=50: bound mu=0.1400  reset mu=0.1400  c=1 mu=0.1000  c=0.95 mu=0.0240",
        # Section 33.6: bounded domain
        "T=2000: composition=20.0000  plateau bound=4.0000",
        "T=10: 0.0961  T=100: 0.3693  T=400: 0.3786  T=1000: 0.3786  T=2000: 0.3786",
        "with 500 cells: 0.3779  bound/walk at T=2000: 10.57",
        # Section 33.7: the quadratic
        "c=0.99: T=1: 1.0000  T=10: 3.1610  T=100: 9.6101  T=1000: 14.1061",
        "c=0.9: T=1: 1.0000  T=10: 3.0291  T=100: 4.3588  T=1000: 4.3589",
        "c=0.99: limit mu=14.1067",
        "composition mu=10.0000 eps=91.8173; last iterate c=0.99 mu=1.4107 eps=6.5534",
        "exact limit RDP=199.0000  Chourasia et al. limit=400.0000  ratio=0.4975",
        "mu=4.3381  exact=4.3365",
        "mu=2.8391  composition of its 10 steps=3.1623",
    ):
        assert expected in out, expected
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "shuffling_iteration_bounds.svg",
        "shuffling_iteration_counts.svg",
        "shuffling_iteration_hidden_state.svg",
        "shuffling_iteration_positions.svg",
        "shuffling_iteration_shift.svg",
    ]


def test_public_api() -> None:
    subprocess.run([sys.executable, "-c", PUBLIC_API_CHECK], cwd=CODE, check=True)
