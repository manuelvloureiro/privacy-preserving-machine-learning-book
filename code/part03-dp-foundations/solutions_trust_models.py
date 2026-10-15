"""Check the worked solutions to the exercises of the trust-models chapter.

    uv run python code/part03-dp-foundations/solutions_trust_models.py

Prints every number quoted in book-exercises/part03-dp-foundations/08-trust-models.md (seed 0 for the one simulation).
Makes no figures; --out is accepted and ignored.

Conventions as in trust_models.py: replace neighbours, ε0 the local level, (ε, δ) the central one.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from trust_models import (
    bisect,
    central_count_sd,
    collusion_eps,
    fmt_closed_form,
    fmt_max_eps0,
    gaussian_sigma,
    oracle_variance,
    oue_probs,
    rr_variance,
    sue_probs,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: forced response ----------------------------------------------------------------


def forced_response_keep(s: float) -> float:
    """Truthful w.p. s, else a fair coin: the report equals the bit w.p. s + (1 − s)/2."""
    return s + (1 - s) / 2


def forced_response_eps(s: float) -> float:
    """ε0 = ln(t/(1 − t)) = ln((1 + s)/(1 − s))."""
    t = forced_response_keep(s)
    return math.log(t / (1 - t))


# --- Exercise 2: the best unary encoding --------------------------------------------------------


def unary_q(p: float, eps0: float) -> float:
    """Solve p(1 − q)/(q(1 − p)) = e^ε0 for q: q = p / (p + (1 − p) e^ε0)."""
    return p / (p + (1 - p) * math.exp(eps0))


def unary_rare_variance(p: float, eps0: float) -> float:
    """n · Var of a rare value's frequency for the unary encoding with keep probability p."""
    q = unary_q(p, eps0)
    return q * (1 - q) / (p - q) ** 2


def best_unary_p(eps0: float, grid: int = 100_000) -> tuple[float, float]:
    """Minimize the rare-value variance over p on a grid; returns (p, variance)."""
    ps = [0.5 * (i + 1) / grid + 0.25 for i in range(grid)]  # p in (0.25, 0.75]
    best = min(ps, key=lambda p: unary_rare_variance(p, eps0))
    return best, unary_rare_variance(best, eps0)


# --- Exercise 3: sample sizes -------------------------------------------------------------------


def local_people_needed(sd: float, eps: float) -> int:
    """Smallest n with randomized-response sd e^{ε/2}/((e^ε − 1)√n) ≤ sd."""
    return math.ceil(rr_variance(1, eps) / sd**2)


def central_people_needed(sd: float, eps: float) -> int:
    """Smallest n with central Laplace fraction sd √2/(nε) ≤ sd."""
    return math.ceil(central_count_sd(1, eps) / sd)


# --- Exercise 6: Laplace as a sum of Gamma differences ------------------------------------------


def gamma_split_laplace(rng: np.random.Generator, n: int, b: float, size: int) -> np.ndarray:
    """Σ_i (G_i − G'_i) with G, G' ~ Gamma(shape 1/n, scale b): should be Lap(b)."""
    g = rng.gamma(1 / n, b, size=(n, size)) - rng.gamma(1 / n, b, size=(n, size))
    return g.sum(axis=0)


def laplace_tail(t: float, b: float) -> float:
    """Pr[|Lap(b)| ≥ t] = e^{−t/b}."""
    return math.exp(-t / b)


# --- Exercise 7: dropouts -----------------------------------------------------------------------


def overprovisioned_client_sigma(
    eps: float, delta: float, sensitivity: float, clients: int, drop_fraction: float
) -> float:
    """Per-client σ so that the clients who remain still carry the full central variance."""
    sigma = gaussian_sigma(eps, delta, sensitivity)
    return sigma / math.sqrt(clients * (1 - drop_fraction))


# --- The page's numbers -------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("[Exercise 1: forced response]")
    for s in (0.5, 0.25, 0.1):
        print(f"  s={s:g}: t={forced_response_keep(s):.4f}  eps0={forced_response_eps(s):.4f}")

    print("[Exercise 2: best unary encoding at ε0 = 1]")
    p, var = best_unary_p(1.0)
    print(f"  grid minimum at p={p:.4f}: n·Var={var:.4f}")
    print(
        f"  optimized (p=1/2): {oracle_variance(*oue_probs(1.0), 0.0, 1):.4f}  "
        f"symmetric: {oracle_variance(*sue_probs(1.0), 0.0, 1):.4f}"
    )
    for p in (0.3, 0.4, 0.6, 0.7):
        print(f"  p={p:g}: q={unary_q(p, 1.0):.4f}  n·Var={unary_rare_variance(p, 1.0):.4f}")

    print("[Exercise 3: people needed for sd 0.001 at ε = 1]")
    print(
        f"  local={local_people_needed(0.001, 1.0)}  central={central_people_needed(0.001, 1.0)}  "
        f"factor e/(e−1)^2={rr_variance(1, 1.0):.4f}"
    )

    print("[Exercise 4: n = 1e6, δ = 1e-8, target ε = 0.5]")
    n, delta = 1_000_000, 1e-8
    eps0 = fmt_max_eps0(0.5, n, delta)
    print(
        f"  largest eps0={eps0:.3f}  closed form there={fmt_closed_form(eps0, n, delta):.4f}  "
        f"condition allows eps0 <= {math.log(n / (16 * math.log(2 / delta))):.3f}"
    )
    e0 = math.exp(eps0)
    root, linear = 8 * math.sqrt(e0 * math.log(4 / delta) / n), 8 * e0 / n
    u = math.tanh(eps0 / 2) * (root + linear)
    print(
        f"  e^eps0={e0:.1f}  tanh={math.tanh(eps0 / 2):.4f}  ln(4/δ)={math.log(4 / delta):.2f}  "
        f"sqrt term={root:.4f}  1/n term={linear:.4f}  u={u:.4f}  ln(1+u)={math.log1p(u):.4f}  u/ln(1+u)={u / math.log1p(u):.2f}"
    )

    print("[Exercise 5: Sybils, ε0 = 2, δ = 1e-6]")
    for m in (0, 5_000, 9_000):
        honest = 10_000 - m
        print(
            f"  m={m}: honest={honest}  eps={fmt_closed_form(2.0, honest, 1e-6):.4f}  "
            f"condition allows eps0 <= {math.log(honest / (16 * math.log(2 / 1e-6))):.3f}"
        )

    print("[Exercise 6: Gamma split of Lap(2), n = 100]")
    x = gamma_split_laplace(rng, 100, 2.0, 200_000)
    print(
        f"  mean={x.mean():.4f}  var={x.var():.3f} (exact 8)  "
        f"P[|X|>=4]={float((np.abs(x) >= 4).mean()):.4f} (exact {laplace_tail(4, 2):.4f})"
    )

    print("[Exercise 7: dropouts, 100 clients, (1, 1e-6), Δ2 = √2]")
    sens = math.sqrt(2)
    print(f"  30% dropped: eps={collusion_eps(1.0, 1e-6, 0.7):.3f}")
    print(
        f"  per-client sigma: planned={gaussian_sigma(1.0, 1e-6, sens) / 10:.3f}  "
        f"over-provisioned for 30%={overprovisioned_client_sigma(1.0, 1e-6, sens, 100, 0.3):.3f}"
    )
    print(f"  variance if nobody drops: factor {1 / 0.7:.3f} of the needed")
    check = bisect(lambda f: collusion_eps(1.0, 1e-6, f) - 2.0, 0.05, 1.0)
    print(f"  ε reaches 2 when the honest fraction falls to {check:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="accepted for a uniform interface; no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
