"""Check the numbers in the study note notes/concentrated-privacy/index.md.

    uv run python code/notes/concentrated_privacy.py

Prints every number the note quotes: the Laplace mechanism's mean privacy loss against the two
bounds the note derives, advanced composition against basic composition, the zCDP and mCDP
conversions, and the exact privacy profile of the Gaussian mechanism. The note draws no figures.
"""

import math

import numpy as np
from scipy.stats import norm

# --- One round's mean privacy loss -----------------------------------------------------------


def laplace_mean_loss(eps: float) -> float:
    """Exact mean privacy loss (the KL divergence) of the Laplace mechanism with Δ/b = ε: ε - 1 + e^{-ε}."""
    return eps - 1.0 + math.exp(-eps)


def laplace_mean_loss_by_integration(eps: float) -> float:
    """The same mean, by integrating L(y) = |y - ε| - |y| against the Laplace(0, 1) density centred at ε."""

    y = np.linspace(-50.0, 50.0, 2_000_001)
    integrand = 0.5 * np.exp(-np.abs(y - eps)) * (np.abs(y) - np.abs(y - eps))
    return float(np.trapezoid(integrand, y))


def laplace_loss_split(eps: float) -> tuple[float, float, float]:
    """For the Laplace mechanism with Δ/b = ε, written with b = 1 and true answers ε and 0: the probability
    that the loss is +ε (output at or above ε), the probability that it is -ε (output at or below 0), and the
    contribution of the outputs in between to the mean loss, ∫_0^ε (2y - ε) e^{-(ε - y)}/2 dy."""
    between = 0.5 * math.exp(-eps) * ((eps - 2) * math.exp(eps) + eps + 2)
    return 0.5, 0.5 * math.exp(-eps), between


def chord_mean_bound(eps: float) -> float:
    """The note's bound on the mean loss of an ε-DP round: ε tanh(ε/2)."""
    return eps * math.tanh(eps / 2.0)


def loose_mean_bound(eps: float) -> float:
    """Dwork and Rothblum's looser form of the same bound: ε(e^ε - 1)/2."""
    return eps * math.expm1(eps) / 2.0


# --- Composition -------------------------------------------------------------------------------


def advanced_epsilon(eps: float, k: int, eta: float) -> float:
    """Advanced composition of k ε-DP rounds: ε√(2k ln(1/η)) + kε(e^ε - 1)/2."""
    return eps * math.sqrt(2.0 * k * math.log(1.0 / eta)) + k * loose_mean_bound(eps)


def zcdp_epsilon(rho: float, eta: float) -> float:
    """The conversion of ρ-zCDP to (ε, η)-DP: ρ + 2√(ρ ln(1/η))."""
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / eta))


def mcdp_epsilon(mu: float, tau: float, eta: float) -> float:
    """The conversion of a (μ, τ) mean and subgaussian bound to (ε, η)-DP: μ + τ√(2 ln(1/η))."""
    return mu + tau * math.sqrt(2.0 * math.log(1.0 / eta))


# --- The Gaussian mechanism's exact privacy profile ----------------------------------------


def gaussian_delta(eps: float, gamma: float) -> float:
    """Smallest δ for a given ε when the means differ by γ standard deviations: Φ(γ/2 - ε/γ) - e^ε Φ(-γ/2 - ε/γ)."""
    return float(norm.cdf(gamma / 2.0 - eps / gamma) - math.exp(eps) * norm.cdf(-gamma / 2.0 - eps / gamma))


def gaussian_epsilon(delta: float, gamma: float) -> float:
    """The ε at which gaussian_delta equals δ, by bisection (gaussian_delta decreases in ε)."""
    lo, hi = 0.0, 100.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if gaussian_delta(mid, gamma) > delta:
            lo = mid
        else:
            hi = mid
    return lo


def best_advanced_split(gamma: float, k: int, delta_total: float, steps: int = 99) -> tuple[float, float, float, float]:
    """Advanced composition of k Gaussian rounds, each (ε_0, δ_0)-DP, over a grid of splits
    kδ_0 + η = δ_total. Returns (total ε, ε_0, δ_0, η) for the split with the smallest total."""
    best: tuple[float, float, float, float] | None = None
    for i in range(1, steps + 1):
        delta0 = (i / (steps + 1)) * delta_total / k
        eta = delta_total - k * delta0
        eps0 = gaussian_epsilon(delta0, gamma)
        total = advanced_epsilon(eps0, k, eta)
        if best is None or total < best[0]:
            best = (total, eps0, delta0, eta)
    assert best is not None
    return best


def main() -> None:
    print("== One round's mean loss, Laplace mechanism with eps=1")
    print(
        f"exact={laplace_mean_loss(1.0):.3f}  integrated={laplace_mean_loss_by_integration(1.0):.3f}  "
        f"chord={chord_mean_bound(1.0):.3f}  loose={loose_mean_bound(1.0):.3f}"
    )

    top, bottom, between = laplace_loss_split(1.0)
    print(f"P(loss=+1)={top:.1f}  P(loss=-1)={bottom:.3f}  between={between:.3f}  mean={top - bottom + between:.3f}")

    print("== 100 rounds of 0.1-DP, eta=1e-5")
    eps, k, eta = 0.1, 100, 1e-5
    print(f"ln(1/eta)={math.log(1 / eta):.3f}")
    print(f"basic={k * eps:.2f}")
    sqrt_term = eps * math.sqrt(2 * k * math.log(1 / eta))
    print(
        f"advanced: sqrt term={sqrt_term:.2f}  mean term={k * loose_mean_bound(eps):.2f}  total={advanced_epsilon(eps, k, eta):.2f}"
    )
    rho = k * eps**2 / 2
    print(f"zCDP: rho total={rho:.2f}  eps={zcdp_epsilon(rho, eta):.2f}")

    print("== 100 Gaussian rounds, sensitivity 1, sigma=5, delta=1e-5")
    gamma, delta = 1 / 5, 1e-5
    rho_round = gamma**2 / 2
    rho_total = k * rho_round
    print(f"rho per round={rho_round:.2f}  rho total={rho_total:.1f}")
    print(
        f"zCDP eps={zcdp_epsilon(rho_total, delta):.2f}  sqrt part={2 * math.sqrt(rho_total * math.log(1 / delta)):.3f}"
    )
    print(f"mCDP eps={mcdp_epsilon(k * rho_round, math.sqrt(k) * gamma, delta):.2f}")
    print(f"exact eps={gaussian_epsilon(delta, math.sqrt(k) * gamma):.2f}")
    eps0 = gaussian_epsilon(delta / k, gamma)
    print(f"basic: eps0={eps0:.3f}  total={k * eps0:.1f}")
    total, eps0, delta0, eta = best_advanced_split(gamma, k, delta)
    print(
        f"advanced: eps0={eps0:.3f}  delta0={delta0:.1e}  eta={eta:.1e}  mean term={k * loose_mean_bound(eps0):.1f}  total={total:.1f}"
    )


if __name__ == "__main__":
    main()
