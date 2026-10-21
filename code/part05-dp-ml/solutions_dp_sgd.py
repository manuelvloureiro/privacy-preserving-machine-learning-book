"""Check the worked solutions to the exercises of the DP-SGD chapter.

    uv run python code/part05-dp-ml/solutions_dp_sgd.py

Prints every number quoted in book-exercises/part05-dp-ml/02-dp-sgd.md (seed 0). Makes no figures;
--out is accepted and ignored.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from dp_sgd import CLIP_RUN, N_TRAIN, Q_RUN, bisect, dpsgd_epsilon, microbatch_noise, noise_multiplier_for
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]
P_WEIGHTS = 3329
SIGNAL = 0.026  # length of the averaged clipped gradient in the running example, from the chapter

# --- Exercise 1: when does signal beat noise ------------------------------------------------------


def break_even_steps(sigma: float, clip: float, batch: float, p: int, signal: float) -> float:
    noise = sigma * clip / batch * math.sqrt(p)
    return (noise / signal) ** 2


# --- Exercise 2: dividing by the realized batch size ------------------------------------------------


def batch_size_stats(n: int, q: float) -> tuple[float, float, float]:
    """(relative standard deviation of |S|, log10 P(|S| = 0), P(|S| < B/2) by a normal approximation)."""
    mean = n * q
    sd = math.sqrt(n * q * (1 - q))
    p_empty_log10 = n * math.log10(1 - q)
    z = (mean / 2 - mean) / sd
    return sd / mean, p_empty_log10, 0.5 * math.erfc(-z / math.sqrt(2))


# --- Exercise 3: Adam is invariant to the clipping norm -----------------------------------------------


def adam_step(grads: list[Array], lr: float, eps: float = 1e-8, b1: float = 0.9, b2: float = 0.999) -> Array:
    """The parameter change after feeding `grads` to Adam, from zero state."""
    m = np.zeros_like(grads[0])
    v = np.zeros_like(grads[0])
    total = np.zeros_like(grads[0])
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        total = total - lr * m_hat / (np.sqrt(v_hat) + eps)
    return total


def adam_invariance(rng: np.random.Generator, clip_a: float, clip_b: float) -> float:
    """Largest relative difference between Adam's updates on normalized-plus-noise gradients scaled by two C."""
    base = [rng.standard_normal(100) for _ in range(20)]
    ua = adam_step([clip_a * g for g in base], 1e-3)
    ub = adam_step([clip_b * g for g in base], 1e-3)
    return float(np.max(np.abs(ua - ub) / np.abs(ua)))


# --- Exercise 4: where ghost clipping stops paying ----------------------------------------------------


def ghost_crossover(p_in: int, p_out: int) -> float:
    """Sequence length T at which 2T² = p_in p_out."""
    return math.sqrt(p_in * p_out / 2)


# --- Exercise 5: the small-q approximation for the noise multiplier --------------------------------


def rho_for(eps: float, delta: float) -> float:
    """ρ with ρ + 2√(ρ ln(1/δ)) = ε."""
    log_inv = math.log(1 / delta)
    s = (-2 * math.sqrt(log_inv) + math.sqrt(4 * log_inv + 4 * eps)) / 2
    return s * s


def sigma_quadratic(eps: float, delta: float, q: float, steps: int) -> float:
    """σ from T q² α/(2σ²) = ρα: σ² = T q²/(2ρ)."""
    return math.sqrt(steps * q * q / (2 * rho_for(eps, delta)))


def sigma_chi2(eps: float, delta: float, q: float, steps: int) -> float:
    """σ from T q² (e^{1/σ²} - 1) α/2 = ρα."""
    return 1 / math.sqrt(math.log1p(2 * rho_for(eps, delta) / (steps * q * q)))


# --- Exercise 6: how ε grows with the number of steps ---------------------------------------------


def steps_for_eps(eps: float, q: float, sigma: float, delta: float) -> int:
    return round(bisect(lambda t: dpsgd_epsilon(q, sigma, round(t), delta) >= eps, 1, 10**6, 40))


def print_table(rng: np.random.Generator) -> None:
    print("== Exercise 1")
    batch = Q_RUN * N_TRAIN
    for eps in (8.0, 4.0, 1.0):
        sigma = noise_multiplier_for(Q_RUN, 10_000, eps, 1e-5)
        k = break_even_steps(sigma, CLIP_RUN, batch, P_WEIGHTS, SIGNAL)
        print(
            f"eps={eps:g}: sigma={sigma:.3g}  noise length={sigma * CLIP_RUN / batch * math.sqrt(P_WEIGHTS):.3f}  "
            f"break-even steps={k:.0f}"
        )
    print("== Exercise 2")
    rel, p0, half = batch_size_stats(N_TRAIN, Q_RUN)
    print(f"n=10000, q=0.01: relative sd of |S|={rel:.4f}  log10 P(|S|=0)={p0:.1f}  P(|S|<50)~{half:.1e}")
    rel, p0, half = batch_size_stats(1000, 0.01)
    print(f"n=1000, q=0.01: relative sd of |S|={rel:.4f}  log10 P(|S|=0)={p0:.2f}  P(|S|<5)~{half:.3f}")
    print("== Exercise 3")
    print(
        f"Adam, C=0.01 against C=10: largest relative difference of the updates = {adam_invariance(rng, 0.01, 10.0):.1e}"
    )
    print("== Exercise 4")
    for p_in, p_out in ((768, 768), (768, 3072), (4096, 4096)):
        print(f"{p_in}->{p_out}: crossover T={ghost_crossover(p_in, p_out):.0f}")
    print("== Exercise 5")
    rho = rho_for(4.0, 1e-5)
    print(f"eps=4, delta=1e-5: rho={rho:.4f}")
    print(
        f"q=0.01, T=10000: quadratic sigma={sigma_quadratic(4.0, 1e-5, 0.01, 10_000):.3f}  "
        f"with e^(1/sigma^2)-1: {sigma_chi2(4.0, 1e-5, 0.01, 10_000):.3f}  accountant: "
        f"{noise_multiplier_for(0.01, 10_000, 4.0, 1e-5):.3g}"
    )
    print(
        f"q=0.001, T=100000: quadratic sigma={sigma_quadratic(4.0, 1e-5, 0.001, 100_000):.3f}  "
        f"with e^(1/sigma^2)-1: {sigma_chi2(4.0, 1e-5, 0.001, 100_000):.3f}  accountant: "
        f"{noise_multiplier_for(0.001, 100_000, 4.0, 1e-5):.3g}"
    )
    print("== Exercise 6")
    for steps in (100, 1_000, 10_000, 100_000):
        print(f"q=0.01, sigma=1, T={steps}: eps={dpsgd_epsilon(0.01, 1.0, steps, 1e-5):.2f}")
    for eps in (1.0, 2.0, 4.0):
        print(f"q=0.01, sigma=1: steps until eps={eps:g}: {steps_for_eps(eps, 0.01, 1.0, 1e-5)}")

    print("== Exercise 7")
    batch = Q_RUN * N_TRAIN
    for size in (2, 4):
        ratio = microbatch_noise(1.0, CLIP_RUN, batch, size) / microbatch_noise(1.0, CLIP_RUN, batch, 1)
        print(
            f"microbatch of {size}: noise ratio {ratio:.0f}  per-example DP-SGD with sigma={ratio:.0f}: "
            f"eps={dpsgd_epsilon(Q_RUN, ratio, 10_000, 1e-5):.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    args = parser.parse_args()
    print_table(np.random.default_rng(args.seed))


if __name__ == "__main__":
    main()
