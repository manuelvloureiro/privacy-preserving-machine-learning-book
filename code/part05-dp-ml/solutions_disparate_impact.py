"""Check the worked solutions to the exercises of the disparate impact chapter.

    uv run python code/part05-dp-ml/solutions_disparate_impact.py

Prints every number quoted in book-exercises/part05-dp-ml/08-disparate-impact.md (seed 0). Makes no
figures; --out is accepted and ignored.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import math
from pathlib import Path

import numpy as np
from disparate_impact import clopper_pearson, effective_share, noise_floor
from dp_sgd import (
    DELTA_RUN,
    Q_RUN,
    STEPS_RUN,
    clipped_grad_sum,
    dpsgd_epsilon,
    init_params,
    make_task,
    noise_multiplier_for,
)
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]


# --- Exercise 1: upweighting a group does not survive clipping ------------------------------------------


def upweighted_share(share: float, upweight: float, norm_minor: float, norm_major: float, clip: float) -> float:
    """Minority's share of the expected clipped update when its losses are multiplied by `upweight`.

    Every record of a group has the same gradient norm; upweighting multiplies the minority's norm."""
    w_minor = min(1.0, clip / (upweight * norm_minor)) * upweight
    w_major = min(1.0, clip / norm_major)
    return share * w_minor * norm_minor / (share * w_minor * norm_minor + (1 - share) * w_major * norm_major)


# --- Exercise 2: the noise floor, exactly ---------------------------------------------------------------


def ar1_variance(a: float, lr: float, tau: float) -> float:
    """Stationary variance of x ← (1 - a)x - lr·τ·ξ: lr²τ²/(1 - (1 - a)²)."""
    return lr**2 * tau**2 / (1 - (1 - a) ** 2)


# --- Exercise 4: how many test records to see a gap ----------------------------------------------------


def records_for_gap(p_major: float, p_minor: float, n_major: int, z: float = 1.96) -> int:
    """Smallest minority test size m at which the normal 95% interval for the gap excludes zero."""
    gap = p_major - p_minor
    var_major = p_major * (1 - p_major) / n_major
    m = 1
    while z * math.sqrt(var_major + p_minor * (1 - p_minor) / m) >= gap:
        m += 1
    return m


# --- Exercise 6: global scaling is a large clipping norm with a small step ------------------------------


def global_scaling_check(rng: np.random.Generator, clip: float, z: float, lr: float) -> float:
    """Largest difference between one DPSGD-Global step (scale by C/Z, noise σC) and one DP-SGD step with
    clipping norm Z, noise σZ and step size lr·C/Z, on the same batch and the same standard normal noise."""
    task = make_task()
    p = init_params(rng)
    x, y = task.train.x[:200], task.train.y[:200]
    full = clipped_grad_sum(p, x, y, math.inf)
    assert float(full.norms.max()) <= z
    xi1 = rng.standard_normal(full.g1.shape)
    xi2 = rng.standard_normal(full.g2.shape)
    sigma = 1.0
    b = 100.0
    # Global scaling: every gradient times C/Z (all norms are below Z), noise σC.
    g1 = (clip / z) * full.g1 + sigma * clip * xi1
    g2 = (clip / z) * full.g2 + sigma * clip * xi2
    step_a = np.concatenate([(lr * g1 / b).ravel(), lr * g2 / b])
    # DP-SGD with clipping norm Z (clips nothing), noise σZ, step lr·C/Z.
    cs = clipped_grad_sum(p, x, y, z)
    h1 = cs.g1 + sigma * z * xi1
    h2 = cs.g2 + sigma * z * xi2
    step_b = np.concatenate([(lr * clip / z * h1 / b).ravel(), lr * clip / z * h2 / b])
    return float(np.max(np.abs(step_a - step_b)))


def print_table(rng: np.random.Generator) -> None:
    print("== Exercise 1")
    for upweight in (1.0, 4.0, 20.0):
        print(
            f"upweight {upweight:g}: minority share of the update {upweighted_share(0.05, upweight, 4.0, 0.2, 0.5):.4f}"
        )
    print(f"no clipping: minority share {0.05 * 4 / (0.05 * 4 + 0.95 * 0.2):.4f}")
    print(f"effective share with weights 1 and 0.125: {effective_share(0.05, 1.0, 0.125):.4f}")

    print("== Exercise 2")
    tau = 0.0068
    for share in (0.95, 0.05):
        a = 0.5 * share
        v = ar1_variance(a, 0.5, tau)
        print(
            f"share {share:g}: a={a:.4f}  variance per coordinate {v:.3e}  excess loss (51 coords) {25.5 * v:.3e}  "
            f"formula {noise_floor(share, 0.5, tau, 51, 1.0):.3e}  small-a approximation {51 * 0.5 * tau**2 / (4 * share):.3e}"
        )

    print("== Exercise 3")
    for share in (0.2, 0.05, 0.01):
        print(f"share {share:g}: break-even steps relative to the whole population {1 / share**2:.0f}")

    print("== Exercise 4")
    lo, hi = clopper_pearson(70, 100)
    print(f"70 of 100: 95% CI [{lo:.3f}, {hi:.3f}]")
    for p_minor in (0.90, 0.85):
        print(f"majority 0.95 on 9500, minority {p_minor:g}: records needed {records_for_gap(0.95, p_minor, 9500)}")

    print("== Exercise 5")
    for w in (2.0, 4.0, 8.0):
        eps = dpsgd_epsilon(w * Q_RUN, 1.36, STEPS_RUN, DELTA_RUN)
        sigma = noise_multiplier_for(w * Q_RUN, STEPS_RUN, 4.0, DELTA_RUN)
        print(f"oversample x{w:g}: eps of minority records at sigma=1.36 {eps:.2f}  sigma for eps=4 {sigma:.3g}")

    print("== Exercise 6")
    print(f"largest difference between the two updates {global_scaling_check(rng, 0.5, 50.0, 0.5):.1e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    args = parser.parse_args()
    print_table(np.random.default_rng(args.seed))


if __name__ == "__main__":
    main()
