"""code/part03-dp-foundations/trust_models.py runs, writes its figures, and reproduces the numbers in book-full/part03-dp-foundations/09-trust-models.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations"


def test_trust_models_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "trust_models.py", tmp_path)
    for expected in (
        # Central against local
        "keep p=0.7500  Var factor e^ε/(e^ε−1)^2=0.7500  sd=0.00866  (simulated mean 0.3000, sd 0.00861)",
        "count sd: local=86.6  central Laplace=1.287  ratio=67.3",
        # The √n gap
        "n=1000: local sd=3.03e-02  central sd=1.41e-03  ratio=21  lower bound (β=0.1)=1.04e-02",
        "n=100000: local sd=3.03e-03  central sd=1.41e-05  ratio=215  lower bound (β=0.1)=1.04e-03",
        "n=10000000: local sd=3.03e-04  central sd=1.41e-07  ratio=2146  lower bound (β=0.1)=1.04e-04",
        "one report, Bern(0.5) vs Bern(0.6), ε=1: KL=0.00429  bound (e^ε−1)^2 TV^2=0.02952",
        # Frequency oracles
        "unary encodings: symmetric eps=1.0000 optimized eps=1.0000",
        "k=2: k-RR=0.921  SUE=3.918  OUE=3.683",
        "k=8: k-RR=2.953  SUE=3.918  OUE=3.683",
        "k=32: k-RR=11.082  SUE=3.918  OUE=3.683",
        "k=128: k-RR=43.596  SUE=3.918  OUE=3.683",
        "k-RR wins for k < 3e^ε + 2 = 10.15",
        # Clones
        (
            "R(x)=[0.1667, 0.1667, 0.5, 0.1667]  clones=[0.1111, 0.1111, 0.0556, 0.0556]  "
            "leftover=[0.0556, 0.0556, 0.4444, 0.1111]  leftover mass=0.6667"
        ),
        # Shuffling
        "ε0 must be ≤ 3.763",
        "ε0=0.5: closed form=0.0939  numeric (Thm 3.2)=0.0204  shuffled binary RR (exact, lower bound)=0.0162",
        "ε0=1: closed form=0.2140  numeric (Thm 3.2)=0.0530  shuffled binary RR (exact, lower bound)=0.0357",
        "ε0=2: closed form=0.5009  numeric (Thm 3.2)=0.1550  shuffled binary RR (exact, lower bound)=0.0870",
        "ε0=3: closed form=0.8241  numeric (Thm 3.2)=0.3207  shuffled binary RR (exact, lower bound)=0.1692",
        "n=1000: largest ε0 for central ε=1: 1.460",
        "n=10000: largest ε0 for central ε=1: 3.504",
        "n=100000: largest ε0 for central ε=1: 5.701",
        "n=1000000: largest ε0 for central ε=1: 7.992",
        "8-ary RR at ε0=7.992 keeps the truth w.p. 0.998",
        # Distributed
        "total σ=5.975  per-client σ=0.597",
        "honest fraction 0.8: ε=1.128",
        "honest fraction 0.5: ε=1.455",
        "honest fraction 0.1: ε=3.525",
        "sum of 100 Skellam shares with total variance 25: mean=0.004  var=25.15",
        "secure sum of [[3, -2], [5, 7], [0, 1]] mod 2^20: [8, 6]",
        # Matched ε
        (
            "n=1000: central=6.40e-05 (sim 6.35e-05)  distributed=2.86e-04 (sim 2.94e-04)  "
            "shuffle=9.35e-03 (sim 9.51e-03)  local=2.71e-02 (sim 2.57e-02)"
        ),
        (
            "n=10000: central=6.40e-07 (sim 6.98e-07)  distributed=2.86e-06 (sim 2.90e-06)  "
            "shuffle=4.88e-05 (sim 4.99e-05)  local=2.71e-03 (sim 2.79e-03)"
        ),
        (
            "n=100000: central=6.40e-09 (sim 6.19e-09)  distributed=2.86e-08 (sim 2.77e-08)  "
            "shuffle=4.76e-07 (sim 4.58e-07)  local=2.71e-04 (sim 2.78e-04)"
        ),
        # Local hashing
        "ε=1: e^ε + 1=3.72  best w=4  local hashing=3.6917  optimized unary=3.6827  w=2: 4.6827",
        "ε=4: e^ε + 1=55.60  best w=56  local hashing=0.0760  optimized unary=0.0760  w=2: 1.0760",
        # Heavy hitters
        "w=56  users per level≈166666  sd of one estimate=6.75e-04  candidates per level=[26, 676, 2054, 1638, 1378, 1378]",
        "found 50 strings: 50 real words, 0 false",
        "smallest frequency found=0.0033  largest missed=0.0035  last-level threshold=3.80 sd=0.0026",
        (
            "testing all 308915776 strings instead: threshold 6.18 sd=0.0042; "
            "at the prefix threshold it would expect 22418 false alarms"
        ),
        # Local mean
        "true mean=82.74  1-bit sd=1.420 (worst case 1.558; simulated mean 82.61, sd 1.415)  Laplace sd=2.036",
        "worst-case 1-bit / Laplace sd=0.765  as ε→0: 0.707  equal at ε=2.324",
        # Repeated collection
        "fresh RR at ln 3, 75 ones in 100 days: log ratio=54.93  ratio=7.18e+23",
        "days=1: majority of fresh reports wrong w.p. 0.25",
        "days=5: majority of fresh reports wrong w.p. 0.104",
        "days=25: majority of fresh reports wrong w.p. 0.00337",
        "days=101: majority of fresh reports wrong w.p. 3.26e-08",
        "RAPPOR homepages (h=2, f=0.75, p=0.5, q=0.75): eps_inf=2.043  eps_1=0.5343",
        "α-point rounding, ε=1: memoized bits agree w.p. 0.3932",
        "± 5 min over 30 days: report ever changes α-point=0.0038 (theory 0.0039)  fresh=1.0000",
        "± 15 min over 30 days: report ever changes α-point=0.0119 (theory 0.0118)  fresh=1.0000",
        "± 30 min over 30 days: report ever changes α-point=0.0236 (theory 0.0237)  fresh=0.9999",
        "one day, n=1000000: true mean=82.68  α-point estimate=82.17",
        "output perturbation γ=0.2 on ε=1: one round 0.569  per round where the memoized bit differs ln(0.8/0.2)=1.386",
        # Shuffled bits
        "random bit w.p. 6.76e-04  expected random senders=676  count sd=18.4  local at ε0=1: 960",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "trust_models.svg",
        "trust_rr_gap.svg",
        "trust_frequency_oracles.svg",
        "trust_clones.svg",
        "trust_shuffle_bound.svg",
        "trust_matched_error.svg",
        "trust_collusion.svg",
        "trust_local_hashing.svg",
        "trust_heavy_hitters.svg",
        "trust_repeated.svg",
    }
