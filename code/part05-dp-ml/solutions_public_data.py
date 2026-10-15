"""Check the worked solutions to the exercises of the public data and private fine-tuning chapter.

    uv run python code/part05-dp-ml/solutions_public_data.py

Prints every number quoted in book-exercises/part05-dp-ml/06-public-data.md (seed 0). Makes no figures;
--out is accepted and ignored.
"""

import os

# The models are tiny; multi-threaded BLAS only adds overhead.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import math
from pathlib import Path

import numpy as np
from dp_erm import noisy_gd_sigma
from dp_sgd import Params, accuracy, forward, make_task, sigmoid
from numpy.typing import NDArray
from public_data import (
    FineTuneConfig,
    agreement,
    lora_clipped_sum,
    lora_init,
    pretrained,
    private_sigma,
    private_teacher,
    public_teacher,
    split_test,
    train_last_layer,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]

# --- Exercise 1: counting the adapter's weights ----------------------------------------------------------


def adapter_count(d_in: int, d_out: int, rank: int) -> int:
    return rank * (d_in + d_out)


# --- Exercise 2: the noise cost of a finite run ----------------------------------------------------------


def finite_fraction(h: float, eta: float, steps: int) -> float:
    """Share of the stationary variance that T noisy steps reach along curvature h: 1 - (1 - ηh)^{2T}."""
    return 1 - (1 - eta * h) ** (2 * steps)


def power_law_deff(h1: float, power: float, p: int, eta: float, steps: int) -> tuple[float, float]:
    """(d_eff, trace) for the spectrum h_j = h1 j^{-power}, j = 1..p."""
    h = h1 * np.arange(1, p + 1, dtype=float) ** (-power)
    return float(np.sum(1 - (1 - eta * h) ** (2 * steps))), float(h.sum())


# --- Exercise 3: a private linear probe without clipping -----------------------------------------------


def noisy_gd_probe(
    rng: np.random.Generator,
    feats: Array,
    y: Array,
    w0: Array,
    steps: int,
    eps: float,
    delta: float,
    lr: float,
    clip: float,
) -> Array:
    """Full-batch noisy gradient descent on the logistic loss of a linear model on fixed features. With
    clip = inf the per-record gradient length is bounded by max ‖φ(x)‖ ≤ √k, the Lipschitz constant;
    otherwise each record's gradient is clipped to `clip`. Replace-one neighbours, Gaussian DP composition."""
    n = y.size
    lipschitz = math.sqrt(feats.shape[1]) if math.isinf(clip) else clip
    sigma = noisy_gd_sigma(n, steps, eps, delta, lipschitz)
    w = w0.copy()
    for _ in range(steps):
        d = -y * sigmoid(-y * (feats @ w))
        norms = np.abs(d) * np.linalg.norm(feats, axis=1)
        scale = np.ones_like(norms) if math.isinf(clip) else np.minimum(1.0, clip / np.maximum(norms, 1e-300))
        g = feats.T @ (d * scale) / n
        w = w - lr * (g + sigma * rng.standard_normal(w.shape))
    return w


# --- Exercise 4: the adapter's first step -------------------------------------------------------------


def first_step_update(
    rng: np.random.Generator, base: Params, x: Array, y: Array, sigma: float, clip: float, scale: float
) -> tuple[Array, Array]:
    """ΔW1 after one noisy adapter step from U = 0, with step η/B = scale, and V."""
    lp = lora_init(rng, base, 4)
    s = lora_clipped_sum(base, lp, x, y, clip)
    gu = s.gu + sigma * clip * rng.standard_normal(s.gu.shape)
    gv = s.gv + sigma * clip * rng.standard_normal(s.gv.shape)
    u, v = lp.u - scale * gu, lp.v - scale * gv
    return u @ v.T - lp.u @ lp.v.T, lp.v


# --- Exercise 5: how related must the public data be -----------------------------------------------------

MISMATCH: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.15, 0.3), (0.5, 1.0), (1.0, math.pi / 2))

# --- Exercise 6: a record in both datasets ------------------------------------------------------------


def tpr_bound(eps: float, delta: float, fpr: float) -> float:
    return min(1.0, math.exp(eps) * fpr + delta)


def print_table(rng: np.random.Generator) -> None:
    print("== Exercise 1")
    for d_in, d_out, r in ((768, 768, 1), (768, 768, 8), (768, 768, 64), (51, 64, 4)):
        k = adapter_count(d_in, d_out, r)
        full = d_in * d_out
        print(
            f"{d_in}x{d_out}, rank {r}: adapter {k:,}  full {full:,}  share {k / full:.4f}  "
            f"noise length ratio sqrt(full/adapter) {math.sqrt(full / k):.2f}"
        )

    print("== Exercise 2")
    for h in (1e-5, 1e-4, 1e-3, 1e-1):
        print(f"h={h:g}, eta=0.1, T=10000: share of stationary variance reached {finite_fraction(h, 0.1, 10_000):.4f}")
    for power in (1.0, 1.5, 2.0):
        deff, tr = power_law_deff(0.7, power, 3329, 0.1, 10_000)
        print(
            f"h_j=0.7 j^-{power:g}, p=3329: trace {tr:.3f}  d_eff {deff:.0f}  bound 2 eta T trace {2 * 0.1 * 10_000 * tr:.0f}"
        )

    print("== Exercise 3")
    task = make_task()
    report = split_test(task.test)[1]
    base = pretrained()
    feats = forward(base, task.train.x).a1b
    print(
        f"Lipschitz bound sqrt(65)={math.sqrt(65):.3f}  largest feature norm {float(np.max(np.linalg.norm(feats, axis=1))):.3f}"
    )
    for clip, lr in ((math.inf, 1.0), (1.0, 1.0)):
        for steps in (100, 400):
            sigma = noisy_gd_sigma(task.train.y.size, steps, 1.0, 1e-5, math.sqrt(65) if math.isinf(clip) else clip)
            w = noisy_gd_probe(rng, feats, task.train.y, base.w2, steps, 1.0, 1e-5, lr, clip)
            acc = accuracy(Params(base.w1, w), report)
            print(
                f"clip={clip:g}, T={steps}: per-step noise sd {sigma:.4f}  noise length {sigma * math.sqrt(65):.4f}  report accuracy {acc:.4f}"
            )

    print("== Exercise 4")
    for scale in (0.1 / 100, 0.3 / 100, 1.0):
        du, v = first_step_update(rng, base, task.train.x[:100], task.train.y[:100], 4.13, 0.5, scale)
        proj = du @ v @ np.linalg.solve(v.T @ v, v.T)
        sv: Array = np.linalg.svd(du, compute_uv=False)
        big: NDArray[np.bool_] = sv > 1e-10 * sv[0]
        print(
            f"eta/B={scale:g}: rank of the update {int(big.sum())}  "
            f"relative size of the part outside span(V) {float(np.linalg.norm(du - proj) / np.linalg.norm(du)):.1e}"
        )

    print("== Exercise 5")
    for shift, rot in MISMATCH:
        p0 = pretrained(shift, rot)
        agree = agreement(public_teacher(shift, rot), private_teacher(), report.x)
        tuned = train_last_layer(rng, p0, task.train, FineTuneConfig(private_sigma(1.0), 0.3))
        print(
            f"shift={shift:g}, rotation={rot:.2f}: teacher agreement {agree:.4f}  pretrained {accuracy(p0, report):.4f}  "
            f"private last layer eps=1 {accuracy(tuned, report):.4f}"
        )

    print("== Exercise 6")
    for fpr in (0.001, 0.01):
        print(f"eps=1, delta=1e-5, FPR={fpr:g}: TPR <= {tpr_bound(1.0, 1e-5, fpr):.5f}")
    print(f"two copies in the private data: eps={2 * 1.0:g}  delta={2 * math.exp(1.0) * 1e-5:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    args = parser.parse_args()
    print_table(np.random.default_rng(args.seed))


if __name__ == "__main__":
    main()
