"""Check the derivations of the appendix on random matrices and factorizations, and draw its figures.

    uv run python code/appendix-derivations/apx_matrices.py
    uv run python code/appendix-derivations/apx_matrices.py --seed 1 --out /tmp/figures

Prints the numbers quoted in book-full/appendix-derivations/08-matrices.md (they match exactly with
the default seed), then saves the appendix's figures.
"""

import argparse
import math
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

# Small matrices only: one BLAS thread is faster than many, and keeps runs on a busy machine predictable.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "apx_matrices"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


# --- The mean of an inverse Wishart matrix ---------------------------------------------------------


def gaussian_matrix(rng: np.random.Generator, m: int, n: int) -> Array:
    return rng.standard_normal((m, n))


def subset_matrix(rng: np.random.Generator, m: int, n: int) -> Array:
    """m random subsets of n people: independent fair 0/1 entries."""
    return (rng.random((m, n)) < 0.5).astype(np.float64)


def subset_inverse_scale(n: int) -> float:
    """Diagonal entry of Σ⁻¹ for Σ = E[qqᵀ] = (I + 11ᵀ)/4: 4(1 - 1/(n + 1)) = 4n/(n + 1)."""
    return 4 * n / (n + 1)


def scaled_inverse_diagonal(rng: np.random.Generator, m: int, n: int, reps: int, subsets: bool) -> float:
    """m times the average diagonal entry of (ZᵀZ)⁻¹ over reps draws. For 0/1 rows it is divided by the
    diagonal of Σ⁻¹, so that both kinds of rows are predicted to give m/(m - n - 1)."""
    total = 0.0
    for _ in range(reps):
        z = subset_matrix(rng, m, n) if subsets else gaussian_matrix(rng, m, n)
        total += float(np.mean(np.diag(np.linalg.inv(z.T @ z))))
    scale = subset_inverse_scale(n) if subsets else 1.0
    return m * total / reps / scale


def wishart_prediction(m: int, n: int) -> float:
    """m · E[(ZᵀZ)⁻¹]_ii = m/(m - n - 1) for Gaussian rows, m ≥ n + 2."""
    return m / (m - n - 1)


def residual_and_inverse(z: Array) -> tuple[float, float]:
    """‖(I - P)z₁‖², with P the projection on the other columns, and 1/[(ZᵀZ)⁻¹]₁₁: equal by step 2."""
    z1, rest = z[:, 0], z[:, 1:]
    coef, *_ = np.linalg.lstsq(rest, z1, rcond=None)
    resid = z1 - rest @ coef
    return float(resid @ resid), 1.0 / float(np.linalg.inv(z.T @ z)[0, 0])


def first_residuals(rng: np.random.Generator, m: int, n: int, reps: int) -> Array:
    """1/[(ZᵀZ)⁻¹]₁₁ over reps Gaussian matrices: a sample of χ² with m - n + 1 degrees of freedom."""
    out = np.empty(reps)
    for i in range(reps):
        z = gaussian_matrix(rng, m, n)
        out[i] = 1.0 / float(np.linalg.inv(z.T @ z)[0, 0])
    return out


def chi2_pdf(x: Array, k: int) -> Array:
    """Density of χ² with k degrees of freedom, x^{k/2-1} e^{-x/2} / (2^{k/2} Γ(k/2))."""
    log_norm = (k / 2) * math.log(2) + math.lgamma(k / 2)
    return np.exp((k / 2 - 1) * np.log(x) - x / 2 - log_norm)


def inverse_chi2_mean(rng: np.random.Generator, k: int, reps: int) -> float:
    """Monte Carlo E[1/X] for X ~ χ²_k, to compare with 1/(k - 2)."""
    x = (rng.standard_normal((reps, k)) ** 2).sum(axis=1)
    return float(np.mean(1.0 / x))


def inverse_offdiag_ratio(rng: np.random.Generator, m: int, n: int, reps: int, subsets: bool) -> float:
    """Average |off-diagonal| over average diagonal of the mean of (ZᵀZ)⁻¹: 0 by the symmetry step."""
    acc = np.zeros((n, n))
    for _ in range(reps):
        z = subset_matrix(rng, m, n) if subsets else gaussian_matrix(rng, m, n)
        acc += np.linalg.inv(z.T @ z)
    acc /= reps
    off = acc[~np.eye(n, dtype=bool)]
    return float(np.mean(np.abs(off)) / np.mean(np.diag(acc)))


# --- The smallest singular value of a random subset matrix -------------------------------------------


def smin(a: Array) -> float:
    return float(np.linalg.svd(a, compute_uv=False)[-1])


def smin_ratio(rng: np.random.Generator, m: int, n: int, reps: int) -> float:
    """Average s_min(A)/√m over reps random subset matrices."""
    return sum(smin(subset_matrix(rng, m, n)) for _ in range(reps)) / reps / math.sqrt(m)


def smin_heuristic(m: int, n: int) -> float:
    """The cited asymptotic s_min/√m ≈ ½(1 - √(n/m))."""
    return 0.5 * (1 - math.sqrt(n / m))


GOOD_LEVEL = 1 / 16  # a query is good for x when ⟨q, x⟩² > ‖x‖²/16
GOOD_PROB = 3 / 16  # the chapter's lemma: Pr[good] ≥ 3/16 for every fixed x
COUNT_SHARE = 1 / 8  # Hoeffding: more than m/8 good queries, except with probability e^{-m/128}
NET_FLOOR = 1 / 14  # proved: s_min(A) ≥ √m/14


def net_floor(n: int) -> float:
    """The bound of the net step: ‖Ax‖/√m ≥ (1/√128)(1 - 1/(80√n)) - 1/80 for every unit x (grid step 1/(40n))."""
    return math.sqrt(COUNT_SHARE * GOOD_LEVEL) * (1 - 1 / (80 * math.sqrt(n))) - 1 / 80


def queries_needed(n: int, eta: float) -> float:
    """m ≥ 128(n ln(80n + 1) + ln(1/η)): the union bound over (80n + 1)^n grid points, each failing w.p. e^{-m/128}."""
    return 128 * (n * math.log(80 * n + 1) + math.log(1 / eta))


def good_share(a: Array, x: Array) -> float:
    """Fraction of the queries (rows of a) with ⟨q, x⟩² > ‖x‖²/16."""
    return float(np.mean((a @ x) ** 2 > GOOD_LEVEL * float(x @ x)))


def random_unit_vectors(rng: np.random.Generator, n: int, count: int) -> Array:
    v = rng.standard_normal((count, n))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def weakest_direction(a: Array) -> Array:
    """Right singular vector of the smallest singular value."""
    return np.linalg.svd(a)[2][-1]


# --- The prefix-sum matrix -------------------------------------------------------------------------


def prefix_matrix(t: int) -> Array:
    return np.tril(np.ones((t, t)))


def prefix_inverse_gram(t: int) -> Array:
    """(A⁻¹)ᵀA⁻¹ with A⁻¹ = I - N: 2 on the diagonal except a 1 in the last entry, -1 beside it."""
    a_inv = np.eye(t) - np.eye(t, k=-1)
    return a_inv.T @ a_inv


def prefix_angles(t: int) -> Array:
    """θ_k = (2k - 1)π/(2T + 1), k = 1..T."""
    k = np.arange(1, t + 1, dtype=np.float64)
    return (2 * k - 1) * math.pi / (2 * t + 1)


def prefix_singular_values(t: int) -> Array:
    """σ_k(A) = 1/(2 sin(θ_k/2)), largest first."""
    return 1 / (2 * np.sin(prefix_angles(t) / 2))


def prefix_eigvec_residual(t: int) -> float:
    """Largest |Mv - λv| over k, for v_j = sin(jθ_k) normalized and λ = 4 sin²(θ_k/2)."""
    mat = prefix_inverse_gram(t)
    j = np.arange(1, t + 1, dtype=np.float64)
    worst = 0.0
    for theta in prefix_angles(t).tolist():
        v: Array = np.sin(j * theta)
        v = v / np.linalg.norm(v)
        lam = 4 * math.sin(theta / 2) ** 2
        worst = max(worst, float(np.max(np.abs(mat @ v - lam * v))))
    return worst


def singular_value_error(t: int) -> float:
    """Largest relative gap between the closed form and a numerical SVD."""
    num = np.linalg.svd(prefix_matrix(t), compute_uv=False)
    return float(np.max(np.abs(num / prefix_singular_values(t) - 1)))


def trace_norm(t: int) -> float:
    return float(prefix_singular_values(t).sum())


def trace_duality_check(rng: np.random.Generator, t: int, trials: int) -> tuple[float, float]:
    """(largest tr(UᵀA) over random U with ‖U‖_op = 1, tr(UᵀA) for U = U_A V_Aᵀ), against ‖A‖_*."""
    a = prefix_matrix(t)
    best = -math.inf
    for _ in range(trials):
        u = rng.standard_normal((t, t))
        u /= np.linalg.svd(u, compute_uv=False)[0]
        best = max(best, float(np.trace(u.T @ a)))
    ua, _, vat = np.linalg.svd(a)
    return best, float(np.trace((ua @ vat).T @ a))


def sqrt_coeffs(t: int) -> Array:
    """r_k = C(2k, k)/4^k for k < T, by r_k = r_{k-1}(2k - 1)/(2k)."""
    k = np.arange(1, t, dtype=np.float64)
    return np.concatenate(([1.0], np.cumprod((2 * k - 1) / (2 * k))))


def wallis_sequences(kmax: int) -> tuple[Array, Array]:
    """a_k = (k + ½)r_k² (decreasing to 1/π) and b_k = (k + ¼)r_k² (increasing to 1/π), k = 0..kmax-1."""
    r2 = sqrt_coeffs(kmax) ** 2
    k = np.arange(kmax, dtype=np.float64)
    return (k + 0.5) * r2, (k + 0.25) * r2


def wallis_ratios_exact(kmax: int) -> tuple[bool, bool]:
    """Step 4 with exact rational arithmetic: a_{k+1}/a_k = (2k+3)(2k+1)/(2k+2)² < 1 and
    b_{k+1}/b_k = (4k+5)(2k+1)²/((4k+1)(2k+2)²) > 1 for 0 ≤ k < kmax."""
    a_ok = all((2 * k + 3) * (2 * k + 1) < (2 * k + 2) ** 2 for k in range(kmax))
    b_ok = all((4 * k + 5) * (2 * k + 1) ** 2 > (4 * k + 1) * (2 * k + 2) ** 2 for k in range(kmax))
    return a_ok, b_ok


def wallis_bounds_hold(kmax: int) -> bool:
    """1/(π(k + ½)) ≤ r_k² ≤ 1/(π(k + ¼)) in floating point for k < kmax (the upper gap is about
    1/(32πk²) relative, so kmax = 10⁴ keeps it far above rounding error)."""
    a_k, b_k = wallis_sequences(kmax)
    return bool(np.all(a_k >= 1 / math.pi) and np.all(b_k <= 1 / math.pi))


def sqrt_sens2(t: int) -> float:
    """sens(R)² = Σ_{k<T} r_k² for the square root, which is also its largest error e_T."""
    return float(np.sum(sqrt_coeffs(t) ** 2))


def sqrt_rmse(t: int) -> float:
    """Root-mean-square error per unit of privacy: sens(R)·‖L‖_F/√T, with ‖L‖_F² = Σ_k (T - k) r_k²."""
    r2 = sqrt_coeffs(t) ** 2
    frob2 = float(np.sum((t - np.arange(t)) * r2))
    return math.sqrt(float(r2.sum()) * frob2 / t)


def lower_bound_floor(t: int) -> float:
    """‖A‖_*/T ≥ ln(2T + 1)/π, from sin x ≤ x and Σ 1/(2k - 1) ≥ ½ ln(2T + 1)."""
    return math.log(2 * t + 1) / math.pi


def sqrt_ceiling(t: int) -> float:
    """RMSE of the square root ≤ Σ r_k² ≤ 1 + (0.8 + ln((4T - 3)/5))/π."""
    return 1 + (0.8 + math.log((4 * t - 3) / 5)) / math.pi


# --- The optimal Frobenius factorization -----------------------------------------------------------


def sqrtm_psd(m: Array) -> Array:
    w, v = np.linalg.eigh(m)
    return (v * np.sqrt(np.maximum(w, 0.0))) @ v.T


def x_from_v(g: Array, v: Array) -> Array:
    """X(v) = V^{-1/2}(V^{1/2} G V^{1/2})^{1/2} V^{-1/2}, with G = AᵀA."""
    sv = np.sqrt(v)
    return sqrtm_psd(sv[:, None] * g * sv[None, :]) / (sv[:, None] * sv[None, :])


@dataclass(frozen=True)
class FixedPoint:
    v: Array
    iterations: int
    gaps: list[float]  # max_i |X(v)_ii - 1| before each update


@cache
def frobenius_fixed_point(t: int, tol: float = 1e-10, max_iter: int = 2000) -> FixedPoint:
    """Iterate v ← diag((V^{1/2}AᵀAV^{1/2})^{1/2}) from v = 1, stopping when max |Δv/v| < tol (the chapter's rule)."""
    a = prefix_matrix(t)
    g = a.T @ a
    v = np.ones(t)
    gaps: list[float] = []
    it = 0
    for it in range(1, max_iter + 1):
        sv = np.sqrt(v)
        v_new = np.diag(sqrtm_psd(sv[:, None] * g * sv[None, :])).copy()
        gap = float(np.max(np.abs(v_new - v) / v))  # equals max |X(v)_ii - 1|
        gaps.append(gap)
        v = v_new
        if gap < tol:
            break
    return FixedPoint(v, it, gaps)


def lower_triangular_root(x: Array) -> Array:
    """Lower-triangular R with RᵀR = X: Cholesky of the reversed matrix, reversed back."""
    j = np.eye(x.shape[0])[::-1]
    return j @ np.linalg.cholesky(j @ x @ j).T @ j


@dataclass(frozen=True)
class OptimumCheck:
    rmse: float
    trace_lambda_rmse: float  # √(Σ v_i / T)
    kkt_residual: float  # max |XΛX - AᵀA|
    diag_gap: float  # max |X_ii - 1|
    triangular: float  # largest entry of R above the diagonal
    factor_gap: float  # max |LR - A|
    worst_perturbed: float  # smallest f(Y) - f(X) over random feasible Y (≥ 0)


def frobenius_objective(g: Array, x: Array) -> float:
    return float(np.trace(g @ np.linalg.inv(x)))


def check_optimum(rng: np.random.Generator, t: int, trials: int) -> OptimumCheck:
    fp = frobenius_fixed_point(t)
    a = prefix_matrix(t)
    g = a.T @ a
    x = x_from_v(g, fp.v)
    r = lower_triangular_root(x)
    ell = a @ np.linalg.inv(r)
    rmse = float(np.linalg.norm(ell) / math.sqrt(t))  # sens(R) = 1
    base = frobenius_objective(g, x)
    worst = math.inf
    for _ in range(trials):
        h = rng.standard_normal((t, t))
        y = x + 0.05 * (h + h.T) / math.sqrt(t)
        d = np.sqrt(np.diag(y))
        y = y / (d[:, None] * d[None, :])  # rescale to diagonal exactly 1, so Y is feasible
        if np.min(np.linalg.eigvalsh(y)) > 0:
            worst = min(worst, frobenius_objective(g, y) - base)
    return OptimumCheck(
        rmse=rmse,
        trace_lambda_rmse=math.sqrt(float(fp.v.sum()) / t),
        kkt_residual=float(np.max(np.abs(x @ np.diag(fp.v) @ x - g))),
        diag_gap=float(np.max(np.abs(np.diag(x) - 1))),
        triangular=float(np.max(np.abs(np.triu(r, 1)))),
        factor_gap=float(np.max(np.abs(ell @ r - a))),
        worst_perturbed=worst,
    )


def sqrt_column_norms(t: int) -> Array:
    """Column lengths of the square root's encoder, divided by the longest: column j holds r_0..r_{T-j}."""
    cum = np.sqrt(np.cumsum(sqrt_coeffs(t) ** 2))
    return cum[::-1] / cum[-1]


# --- Streaming: lower-triangular factors and adaptive participation -----------------------------------


def lq_lower(ell: Array, r: Array) -> tuple[Array, Array]:
    """From any A = LR (L of size T×m, m ≥ T), the lower-triangular L' = L Qᵀ and R' = QR with L = L'Q."""
    q, upper = np.linalg.qr(ell.T)  # Lᵀ = Q₁U, so L = UᵀQ₁ᵀ with Uᵀ lower-triangular
    return upper.T, q.T @ r


@dataclass(frozen=True)
class LqCheck:
    rows_before: int
    factor_gap: float  # max |L'R' - A|
    cov_gap: float  # max |L'L'ᵀ - LLᵀ|
    sens_before: float
    sens_after: float
    above_diag: float  # largest entry above the diagonal in L' and R'


def column_sens(r: Array) -> float:
    return float(np.sqrt((r**2).sum(axis=0)).max())


def tree_factorization(t: int) -> tuple[Array, Array]:
    """Binary tree for T a power of two: R has one row per node (1 on the steps it covers), row t of L picks
    the nodes of the binary decomposition of t."""
    nodes: list[tuple[int, int]] = []
    level = 0
    while (1 << level) <= t:
        nodes += [(s, s + (1 << level)) for s in range(0, t, 1 << level)]
        level += 1
    index = {node: i for i, node in enumerate(nodes)}
    r = np.zeros((len(nodes), t))
    for i, (lo, hi) in enumerate(nodes):
        r[i, lo:hi] = 1.0
    ell = np.zeros((t, len(nodes)))
    for step in range(1, t + 1):
        start = 0
        for lv in reversed(range(level)):
            if step >> lv & 1:
                ell[step - 1, index[(start, start + (1 << lv))]] = 1.0
                start += 1 << lv
    return ell, r


def check_lq(t: int) -> LqCheck:
    """Gram-Schmidt reduction of the tree's (2T - 1)-row factorization to T×T lower-triangular factors."""
    a = prefix_matrix(t)
    ell, r = tree_factorization(t)
    l2, r2 = lq_lower(ell, r)
    return LqCheck(
        rows_before=r.shape[0],
        factor_gap=float(np.max(np.abs(l2 @ r2 - a))),
        cov_gap=float(np.max(np.abs(l2 @ l2.T - ell @ ell.T))),
        sens_before=column_sens(r),
        sens_after=column_sens(r2),
        above_diag=float(max(np.max(np.abs(np.triu(l2, 1))), np.max(np.abs(np.triu(r2, 1))))),
    )


@dataclass(frozen=True)
class AdaptiveLoss:
    plain: Array  # privacy loss ln(p/q) under P, without padding
    padded: Array  # with the padding row that makes Σ μ_j² = μ² on every path
    mu: float


def adaptive_privacy_loss(rng: np.random.Generator, t: int, taus: tuple[int, int], paths: int) -> AdaptiveLoss:
    """Square-root encoder, one coordinate, C = 1, noise s = sens(R) so μ = 1. The record adds g₁ = 1 at step
    τ₁ and g₂ = tanh(Y_{τ₂-1}) at step τ₂, chosen from the released outputs. Simulated under P (record absent);
    the loss of row j is μ_j²/2 - Δ_j z_j/ς², with Δ_j = R_{jτ₁}g₁ + R_{jτ₂}g₂ and z_j the noise."""
    r = sqrt_coeffs(t)
    rmat = np.zeros((t, t))
    for i in range(t):
        rmat[i, : i + 1] = r[: i + 1][::-1]
    c1, c2 = rmat[:, taus[0] - 1], rmat[:, taus[1] - 1]
    sens = float(np.linalg.norm(c1 + c2))  # entries are positive, so aligned contributions are the worst case
    noise = sens  # s = sens(R), so μ = 1
    z = rng.standard_normal((paths, t)) * noise
    g2 = np.tanh(z[:, taus[1] - 2] / noise)  # world P: the outputs are the noise (other records cancel)
    delta = c1[None, :] + g2[:, None] * c2[None, :]
    mu_j2 = delta**2 / noise**2
    plain = (mu_j2 / 2 - delta * z / noise**2).sum(axis=1)
    pad2 = np.maximum(1.0 - mu_j2.sum(axis=1), 0.0)
    padded = plain + pad2 / 2 - np.sqrt(pad2) * rng.standard_normal(paths)
    return AdaptiveLoss(plain, padded, 1.0)


# --- The numbers quoted in the text --------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    n = 50
    for m in (60, 100, 200, 400):
        gauss = scaled_inverse_diagonal(rng, m, n, 4000, subsets=False)
        subs = scaled_inverse_diagonal(rng, m, n, 4000, subsets=True)
        print(
            f"[Wishart n={n} m={m}] predicted m/(m-n-1)={wishart_prediction(m, n):.3f}  "
            f"Gaussian={gauss:.3f}  0/1 rows={subs:.3f}"
        )
    z = gaussian_matrix(rng, 60, n)
    res, inv = residual_and_inverse(z)
    print(f"[Wishart regression n={n} m=60] residual={res:.6f}  1/[inv]_11={inv:.6f}")
    k = 11
    print(f"[inverse chi2 k={k}] Monte Carlo={inverse_chi2_mean(rng, k, 1_000_000):.4f}  1/(k-2)={1 / (k - 2):.4f}")
    print(
        f"[Wishart symmetry n=10 m=30] off/diag Gaussian={inverse_offdiag_ratio(rng, 30, 10, 20_000, False):.4f}  "
        f"0/1 rows={inverse_offdiag_ratio(rng, 30, 10, 20_000, True):.4f}"
    )

    for n, m in ((100, 400), (500, 2000)):
        a = subset_matrix(rng, m, n)
        s = smin(a)
        x = weakest_direction(a)
        xs = random_unit_vectors(rng, n, 2000)
        shares = [good_share(a, v) for v in xs]
        ones = np.ones(n) / math.sqrt(n)
        print(
            f"[smin n={n} m={m}] s_min={s:.2f}  s_min/sqrt(m)={s / math.sqrt(m):.4f}  "
            f"half(1-sqrt(n/m))={smin_heuristic(m, n):.4f}  floor=1/14={NET_FLOOR:.4f}"
        )
        print(
            f"[good queries n={n} m={m}] weakest={good_share(a, x):.4f}  "
            f"random min={min(shares):.4f}  ones={good_share(a, ones):.4f}  lemma={GOOD_PROB:.4f}"
        )
    print(f"[net] floor at n=1: {net_floor(1):.4f} >= 1/14 = {NET_FLOOR:.4f}")
    for n in (100, 1000):
        need = queries_needed(n, 0.01)
        print(f"[queries needed n={n} eta=0.01] m >= {need:.0f} = {need / n:.0f} n")
    print(f"[least squares] wrong bits <= 4*14^2 E^2 = {4 * 14**2} E^2")

    t = 256
    print(f"[prefix T={t}] eigenvector residual={prefix_eigvec_residual(t):.1e}  svd gap={singular_value_error(t):.1e}")
    best, eq = trace_duality_check(rng, 16, 2000)
    print(f"[duality T=16] trace norm={trace_norm(16):.4f}  random U max={best:.4f}  U_A V_A^T={eq:.4f}")
    a_ok, b_ok = wallis_ratios_exact(1_000_000)
    a_k, b_k = wallis_sequences(10_000)
    print(
        f"[Wallis] exact ratios k<10^6: a decreasing={a_ok}  b increasing={b_ok}  "
        f"float bounds k<10^4: {wallis_bounds_hold(10_000)}  "
        f"a-1/pi={a_k[-1] - 1 / math.pi:.2e}  1/pi-b={1 / math.pi - b_k[-1]:.2e} at k={len(a_k) - 1}"
    )
    for t in (100, 10_000, 1_000_000):
        bound = trace_norm(t) / t
        rm = sqrt_rmse(t)
        print(
            f"[prefix T={t}] floor ln(2T+1)/pi={lower_bound_floor(t):.3f}  |A|_*/T={bound:.3f}  "
            f"sqrt RMSE={rm:.3f}  ratio={rm / bound:.3f}  sum r^2={sqrt_sens2(t):.3f}  "
            f"ceiling={sqrt_ceiling(t):.3f}  lnT/pi={math.log(t) / math.pi:.3f}"
        )
    t = 100
    print(f"[prefix T={t}] proved ratio <= {sqrt_ceiling(t) / lower_bound_floor(t):.3f}")

    fp2 = frobenius_fixed_point(2)
    x2 = x_from_v(prefix_matrix(2).T @ prefix_matrix(2), fp2.v)
    print(
        f"[T=2] identity RMSE={math.sqrt(1.5):.4f}  sqrt RMSE={sqrt_rmse(2):.4f}  "
        f"optimum rho={x2[0, 1]:.4f} sum v={float(fp2.v.sum()):.4f} RMSE={math.sqrt(float(fp2.v.sum()) / 2):.4f}  "
        f"bound={trace_norm(2) / 2:.4f}"
    )
    for t in (100, 256):
        fp = frobenius_fixed_point(t)
        print(
            f"[Frobenius T={t}] iterations={fp.iterations}  RMSE={math.sqrt(float(fp.v.sum()) / t):.4f}  sqrt RMSE={sqrt_rmse(t):.4f}"
        )
    c = check_optimum(rng, 64, 200)
    print(
        f"[Frobenius T=64] RMSE={c.rmse:.4f}  sqrt(tr Lambda/T)={c.trace_lambda_rmse:.4f}  "
        f"KKT residual={c.kkt_residual:.1e}  diag gap={c.diag_gap:.1e}  above diagonal={c.triangular:.1e}  "
        f"LR-A={c.factor_gap:.1e}  min f(Y)-f(X)={c.worst_perturbed:.4f}"
    )
    print(f"[Frobenius T=64] sqrt RMSE={sqrt_rmse(64):.4f}  bound |A|_*/T={trace_norm(64) / 64:.4f}")

    lq = check_lq(64)
    print(
        f"[LQ tree T=64, {lq.rows_before} rows] LR-A={lq.factor_gap:.1e}  LL^T gap={lq.cov_gap:.1e}  sens before={lq.sens_before:.4f}  "
        f"after={lq.sens_after:.4f}  above diagonal={lq.above_diag:.1e}"
    )
    loss = adaptive_privacy_loss(rng, 8, (1, 5), 1_000_000)
    print(
        f"[adaptive T=8 taus=1,5] padded mean={np.mean(loss.padded):.4f} var={np.var(loss.padded):.4f}  "
        f"plain mean={np.mean(loss.plain):.4f} var={np.var(loss.plain):.4f}  target mean={loss.mu**2 / 2:.4f} var={loss.mu**2:.4f}"
    )


# --- Figures -------------------------------------------------------------------------------------------


def plot_wishart_mean(rng: np.random.Generator, n: int = 50) -> Figure:
    ms = [56, 60, 65, 70, 80, 90, 100, 125, 150, 200, 250, 300, 400, 500]
    ratios = [m / n for m in ms]
    gauss = [scaled_inverse_diagonal(rng, m, n, 200, subsets=False) for m in ms]
    subs = [scaled_inverse_diagonal(rng, m, n, 200, subsets=True) for m in ms]
    fine = np.linspace(55.5, 500, 400)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(fine / n, fine / (fine - n - 1), "k", label=r"$m/(m-n-1)$, inverse-Wishart mean")
    ax.axhline(1, color="gray", linestyle="--", label="1, inverting the average Gram matrix")
    ax.plot(ratios, gauss, "o", label="Gaussian rows, simulated")
    ax.plot(ratios, subs, "s", markerfacecolor="none", label=r"0/1 rows, simulated (÷ diagonal of $\Sigma^{-1}$)")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="queries per unknown, m/n",
        ylabel=r"$m \times$ average diagonal of $(Z^\top Z)^{-1}$",
        title=f"The inverse of a random Gram matrix, n = {n}",
    )
    ax.set_xticks([1.2, 1.5, 2, 3, 5, 10], ["1.2", "1.5", "2", "3", "5", "10"])
    ax.legend()
    return fig


def plot_wishart_residual(rng: np.random.Generator, m: int = 60, n: int = 50) -> Figure:
    k = m - n + 1
    sample = first_residuals(rng, m, n, 20_000)
    xs = np.linspace(0.05, 35, 400)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(sample, bins=70, range=(0, 35), density=True, alpha=0.5, label=r"$1/[(Z^\top Z)^{-1}]_{11}$, 20,000 draws")
    ax.plot(xs, chi2_pdf(xs, k), "k", label=rf"$\chi^2$ density with $m-n+1 = {k}$ degrees of freedom")
    ax.axvline(k, color="gray", linestyle=":", label=f"its mean, {k}")
    ax.set(
        xlabel=r"squared length of the residual $(I-P)z_1$", ylabel="density", title=f"m = {m} rows, n = {n} columns"
    )
    ax.legend(loc="upper right")
    return fig


def plot_smin(rng: np.random.Generator) -> Figure:
    ratios = [1.25, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    fine = np.geomspace(1.05, 64, 300)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(fine, 0.5 * (1 - np.sqrt(1 / fine)), "k", label=r"$\frac{1}{2}(1 - \sqrt{n/m})$, cited asymptotic")
    for n, marker in ((50, "o"), (100, "s"), (200, "^")):
        ys = [smin_ratio(rng, int(r * n), n, 3) for r in ratios]
        ax.plot(ratios, ys, marker, markerfacecolor="none", label=f"simulated, n = {n}")
    ax.axhline(
        NET_FLOOR, color="C3", linestyle="--", label=r"proved floor $1/14$ (needs $m \geq 128\,n\ln(80n+1) + \ldots$)"
    )
    ax.set(
        xscale="log",
        ylim=(0, 0.5),
        xlabel="queries per person, m/n",
        ylabel=r"$s_{\min}(A)/\sqrt{m}$",
        title="Smallest singular value of a random subset matrix",
    )
    ax.set_xticks([1.25, 2, 4, 8, 16, 32, 64], ["1.25", "2", "4", "8", "16", "32", "64"])
    ax.legend(loc="lower right", fontsize=8.5)
    return fig


def plot_directions(rng: np.random.Generator, n: int = 100, m: int = 400) -> Figure:
    a = subset_matrix(rng, m, n)
    xs = random_unit_vectors(rng, n, 4000)
    vals = np.sum((xs @ a.T) ** 2, axis=1) / m
    weakest = smin(a) ** 2 / m
    ones = float(np.sum((a @ (np.ones(n) / math.sqrt(n))) ** 2)) / m
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = [float(b) for b in np.geomspace(3e-3, 60, 90)]
    ax.hist(vals, bins=bins, alpha=0.6, label="4,000 random unit directions x")
    ax.axvline(NET_FLOOR**2, color="C3", linestyle="--", label=r"proved for every x: $1/196$")
    ax.axvline(COUNT_SHARE * GOOD_LEVEL, color="C1", linestyle=":", label=r"each grid point, after Hoeffding: $1/128$")
    ax.axvline(weakest, color="k", label=rf"weakest direction, $s_{{\min}}^2/m = {weakest:.3f}$")
    ax.axvline(ones, color="C2", label=rf"all-ones direction: {ones:.1f}")
    ax.set(
        xscale="log",
        xlabel=r"$\|Ax\|^2/m$",
        ylabel="count",
        title=f"How much A shrinks a unit vector, n = {n}, m = {m}",
    )
    ax.set_ylim(0, 1.7 * ax.get_ylim()[1])
    ax.legend(fontsize=8.5, loc="upper left")
    return fig


def plot_prefix_bounds() -> Figure:
    ts = np.unique(np.geomspace(10, 1_000_000, 40).astype(int))
    bound = [trace_norm(int(t)) / t for t in ts]
    rm = [sqrt_rmse(int(t)) for t in ts]
    floor = [lower_bound_floor(int(t)) for t in ts]
    ceil = [sqrt_ceiling(int(t)) for t in ts]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.fill_between(ts, floor, ceil, color="0.9", label="proved sandwich")
    ax.plot(ts, ceil, color="0.6", linewidth=1)
    ax.plot(ts, floor, color="0.6", linewidth=1)
    ax.plot(ts, rm, "C1", linewidth=2, label="square root: RMSE")
    ax.plot(ts, bound, "C0", linewidth=2, label=r"lower bound $\|A\|_*/T$")
    ax.plot(ts, np.log(ts) / math.pi, "k--", label=r"$(\ln T)/\pi$")
    ax.set(
        xscale="log",
        xlabel="number of steps T",
        ylabel="error per unit of privacy",
        title="The optimal factorization lies between the blue and orange curves",
    )
    ax.legend(loc="upper left")
    return fig


def plot_wallis() -> Figure:
    a_k, b_k = wallis_sequences(100_000)
    k = np.arange(len(a_k))
    sel = np.unique(np.geomspace(1, len(a_k) - 1, 200).astype(int))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(k[sel], a_k[sel], "C0", label=r"$a_k = (k + \frac{1}{2})\,r_k^2$, decreasing")
    ax.plot(k[sel], b_k[sel], "C1", label=r"$b_k = (k + \frac{1}{4})\,r_k^2$, increasing")
    ax.axhline(1 / math.pi, color="k", linestyle="--", label=r"$1/\pi$")
    ax.set(
        xscale="log",
        ylim=(0.27, 0.40),
        xlabel="k",
        ylabel="value",
        title=r"Squeezing $r_k^2$ between $\frac{1}{\pi(k+1/2)}$ and $\frac{1}{\pi(k+1/4)}$",
    )
    ax.legend()
    return fig


def plot_eigenvectors(t: int = 16) -> Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    fine = np.linspace(0, t + 1, 400)
    j = np.arange(0, t + 2)
    for idx, theta in enumerate(prefix_angles(t)[:3]):
        color = f"C{idx}"
        ax.plot(fine, np.sin(fine * theta), color=color, alpha=0.4)
        ax.plot(j[1 : t + 1], np.sin(j[1 : t + 1] * theta), "o", color=color, label=f"k = {idx + 1}")
        ax.plot([0, t + 1], [0.0, math.sin((t + 1) * theta)], "x", color=color, markersize=9)
    ax.axvline(t + 0.5, color="gray", linestyle=":", label=r"mirror at $T + \frac{1}{2}$")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set(
        xlabel="position j",
        ylabel=r"$\sin(j\omega_k)$",
        title=rf"Eigenvectors of $(A^{{-1}})^\top A^{{-1}}$ for T = {t}: zero at j = 0, flat at j = T + ½",
    )
    ax.legend(loc="lower left", fontsize=9)
    return fig


def plot_frobenius() -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for t in (16, 64, 256):
        fp = frobenius_fixed_point(t)
        left.plot(range(1, len(fp.gaps) + 1), fp.gaps, label=f"T = {t} ({fp.iterations} iterations)")
    left.axhline(1e-10, color="gray", linestyle=":", label="stopping tolerance")
    left.set(yscale="log", xlabel="iteration", ylabel=r"$\max_i |X_{ii} - 1|$", title="The fixed-point iteration")
    left.legend()
    t = 64
    steps = np.arange(1, t + 1)
    right.plot(steps, sqrt_column_norms(t), "C1", label="square root, longest column scaled to 1")
    right.plot(steps, np.ones(t), "C0", label="optimum: every column has length 1")
    right.set(
        ylim=(0, 1.1),
        xlabel="column j of the encoder R",
        ylabel="column length",
        title=f"Where the privacy budget goes, T = {t}",
    )
    right.legend(loc="lower left")
    return fig


def plot_adaptive(rng: np.random.Generator) -> Figure:
    loss = adaptive_privacy_loss(rng, 8, (1, 5), 200_000)
    xs = np.linspace(-4, 5, 400)
    mu = loss.mu
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(loss.padded, bins=90, range=(-4, 5), density=True, alpha=0.5, label="with the padding row")
    ax.hist(loss.plain, bins=90, range=(-4, 5), density=True, histtype="step", color="C1", label="as released")
    ax.plot(
        xs,
        np.exp(-((xs - mu**2 / 2) ** 2) / (2 * mu**2)) / math.sqrt(2 * math.pi * mu**2),
        "k",
        label=r"normal density $N(\mu^2/2, \mu^2)$, $\mu = 1$",
    )
    ax.set(
        xlabel="privacy loss ln(p/q) under P",
        ylabel="density",
        title="Two participations, the second chosen adaptively",
    )
    ax.legend()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="directory for the figures")
    args = parser.parse_args()
    seed: int = args.seed
    out: Path = args.out

    print_table(np.random.default_rng(seed))

    rng = np.random.default_rng(seed + 1)  # separate stream, so the table stays reproducible
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "mat_wishart_mean.svg": plot_wishart_mean(rng),
        "mat_wishart_residual.svg": plot_wishart_residual(rng),
        "mat_smin.svg": plot_smin(rng),
        "mat_directions.svg": plot_directions(rng),
        "mat_prefix_bounds.svg": plot_prefix_bounds(),
        "mat_wallis.svg": plot_wallis(),
        "mat_eigenvectors.svg": plot_eigenvectors(),
        "mat_frobenius.svg": plot_frobenius(),
        "mat_adaptive.svg": plot_adaptive(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
