"""Check the numbers in the DP-FTRL and matrix factorization chapter and draw its figures.

    uv run python code/part05-dp-ml/matrix_mechanisms.py
    uv run python code/part05-dp-ml/matrix_mechanisms.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 04-matrix-mechanisms.md (they match exactly with the default seed),
then saves the chapter's figures.

Gradient descent without momentum releases, in effect, the prefix sums S_t = x_1 + ... + x_t of the
clipped gradient sums x_t: the model after t steps is w_0 - η S_t / B. Releasing all T prefix sums is
the linear workload A x, with A the lower-triangular all-ones T×T matrix. A matrix mechanism factors
A = B C, adds Gaussian noise to C x and releases B(C x + z) = A x + B z. Its privacy is set by the
sensitivity of C x (the largest column norm of C for one participation), its error by the rows of B.

The script compares four factorizations: the identity (C = I, B = A: independent noise on every
step, DP-SGD without amplification), noise on the prefix sums (C = A, B = I), the binary tree of
Dwork et al. (2010) and Chan, Shi & Song (2010), and the square root C = B = A^{1/2}, whose entries
are the coefficients of (1 - x)^{-1/2}; plus the optimal factorization for the Frobenius error,
computed by the fixed point of Denisov et al. (2022). It then trains the part's running-example
network with each of them, in a fixed data order, against DP-SGD with Poisson sampling.
"""

import os

# The models are tiny; multi-threaded BLAS only adds overhead.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "matrix_mechanisms"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from dp_erm import gaussian_mu
from dp_sgd import (
    CLIP_RUN,
    N_TRAIN,
    DpsgdConfig,
    Params,
    Task,
    accuracy,
    clipped_grad_sum,
    init_params,
    make_task,
    noise_multiplier_for,
    train_dpsgd,
)
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

DELTA_RUN = 1e-5
BATCH_RUN = 100  # fixed batch size: n/B = 100 steps per epoch, so a record's participations are 100 steps apart
EPOCHS_RUN: tuple[int, ...] = (1, 5, 10)
EPS_TRAIN: tuple[float, ...] = (2.0, 8.0)
LRS_RUN: tuple[float, ...] = (0.5, 2.0)
SEEDS_RUN = 3
MECHANISMS: tuple[str, ...] = ("identity", "tree", "sqrt", "banded")
LABELS: dict[str, str] = {
    "dpsgd": "DP-SGD, Poisson sampling",
    "identity": "fixed order, independent noise",
    "tree": "fixed order, tree",
    "sqrt": "fixed order, square root",
    "banded": "fixed order, banded square root",
}
FACTOR_LABELS: dict[str, str] = {
    "identity": "independent noise (identity)",
    "prefix": "noise on each prefix sum",
    "tree": "binary tree",
    "sqrt": "square root",
    "optimal": "optimal factorization",
}
COLORS: dict[str, str] = {
    "dpsgd": "black",
    "identity": "tab:red",
    "tree": "tab:orange",
    "sqrt": "tab:blue",
    "banded": "tab:green",
    "prefix": "tab:gray",
    "optimal": "tab:purple",
}


# --- The workload: prefix sums ---------------------------------------------------------------------


def prefix_matrix(t: int) -> Array:
    """A: the lower-triangular all-ones T×T matrix, so (A x)_t = x_1 + ... + x_t."""
    return np.tril(np.ones((t, t)))


def toeplitz_lower(c: Array, t: int) -> Array:
    """The lower-triangular Toeplitz matrix with first column c (zero-padded or cut to length T)."""
    m = np.zeros((t, t))
    for k in range(min(c.size, t)):
        m += np.diag(np.full(t - k, float(c[k])), -k)
    return m


def sqrt_coeffs(t: int) -> Array:
    """r_k = C(2k, k)/4^k, the coefficients of (1 - x)^{-1/2}: r_0 = 1, r_k = r_{k-1}(2k - 1)/(2k)."""
    r = np.ones(t)
    for k in range(1, t):
        r[k] = r[k - 1] * (2 * k - 1) / (2 * k)
    return r


def inverse_series(c: Array, t: int) -> Array:
    """The first T coefficients h of 1/c(x), so that toeplitz(h) = toeplitz(c)^{-1}."""
    h = np.zeros(t)
    h[0] = 1 / c[0]
    for k in range(1, t):
        m = min(k, c.size - 1)
        h[k] = -float(np.dot(c[1 : m + 1], h[k - 1 :: -1][:m])) / c[0]
    return h


# --- Tree aggregation ---------------------------------------------------------------------------


def tree_levels(t: int) -> int:
    """Number of levels of the binary tree over T leaves: ⌈log2 T⌉ + 1."""
    return math.ceil(math.log2(t)) + 1 if t > 1 else 1


def tree_nodes(t: int) -> list[tuple[int, int]]:
    """Nodes (level, start) of the binary tree over T leaves; a node at level l covers [start, start + 2^l)."""
    return [(lev, s) for lev in range(tree_levels(t)) for s in range(0, t, 2**lev)]


def dyadic_decomposition(t: int) -> list[tuple[int, int]]:
    """The nodes whose union is [0, t): one per set bit of t, from the top level down."""
    out: list[tuple[int, int]] = []
    start = 0
    for lev in range(t.bit_length() - 1, -1, -1):
        if t & (1 << lev):
            out.append((lev, start))
            start += 1 << lev
    return out


def tree_factorization(t: int) -> tuple[Array, Array]:
    """(B, C): C has one row per node (1 on the leaves it covers), B picks the dyadic nodes of each prefix."""
    nodes = tree_nodes(t)
    index = {node: i for i, node in enumerate(nodes)}
    c = np.zeros((len(nodes), t))
    for i, (lev, s) in enumerate(nodes):
        c[i, s : min(s + 2**lev, t)] = 1.0
    b = np.zeros((t, len(nodes)))
    for row in range(t):
        for node in dyadic_decomposition(row + 1):
            b[row, index[node]] = 1.0
    return b, c


# --- The optimal factorization for the Frobenius error -------------------------------------------


def sqrtm_psd(m: Array) -> Array:
    w, v = np.linalg.eigh(m)
    return (v * np.sqrt(np.maximum(w, 0.0))) @ v.T


@dataclass(frozen=True)
class Optimal:
    b: Array
    c: Array
    iterations: int


def optimal_factorization(a: Array, tol: float = 1e-10, max_iter: int = 1000) -> Optimal:
    """Minimize ‖B‖_F² subject to A = BC and column norms of C at most 1, by the fixed point
    v = diag((V^{1/2} AᵀA V^{1/2})^{1/2}), X = V^{-1/2}(V^{1/2} AᵀA V^{1/2})^{1/2} V^{-1/2} = CᵀC
    (Denisov et al. 2022, Theorem 3.2). C is the lower-triangular square root of X, B = A C^{-1}."""
    g = a.T @ a
    v = np.ones(a.shape[1])
    it = 0
    for it in range(1, max_iter + 1):
        sv = np.sqrt(v)
        v_new = np.diag(sqrtm_psd(sv[:, None] * g * sv[None, :])).copy()
        done = float(np.max(np.abs(v_new - v) / v)) < tol
        v = v_new
        if done:
            break
    sv = np.sqrt(v)
    x = sqrtm_psd(sv[:, None] * g * sv[None, :]) / (sv[:, None] * sv[None, :])
    # Lower-triangular C with CᵀC = X: Cholesky of the reversed matrix, reversed back.
    j = np.eye(x.shape[0])[::-1]
    lower = np.linalg.cholesky(j @ x @ j)
    c = j @ lower.T @ j
    return Optimal(a @ np.linalg.inv(c), c, it)


@cache
def optimal_for(t: int) -> Optimal:
    return optimal_factorization(prefix_matrix(t))


# --- Privacy and error of a factorization --------------------------------------------------------


def max_column_norm(c: Array) -> float:
    return float(np.sqrt((c**2).sum(axis=0)).max())


def row_norms(b: Array) -> Array:
    return np.sqrt((b**2).sum(axis=1))


def dense_factorization(name: str, t: int) -> tuple[Array, Array]:
    """(B, C) as dense matrices, for small T and for checking the structured formulas."""
    a = prefix_matrix(t)
    if name == "identity":
        return a, np.eye(t)
    if name == "prefix":
        return np.eye(t), a
    if name == "tree":
        return tree_factorization(t)
    if name == "sqrt":
        s = toeplitz_lower(sqrt_coeffs(t), t)
        return s, s
    if name == "optimal":
        o = optimal_for(t)
        return o.b, o.c
    raise ValueError(name)


@dataclass(frozen=True)
class Profile:
    """One participation per record: sensitivity of Cx and the standard deviation of each prefix sum's
    error, for noise z ~ N(0, I) (multiply both by the noise scale)."""

    sens: float
    rows: Array

    def scaled(self) -> Array:
        """Error of each prefix sum at μ = 1: noise standard deviation sens, times the row norms of B."""
        return self.sens * self.rows

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.scaled() ** 2)))

    @property
    def max_error(self) -> float:
        return float(np.max(self.scaled()))


def profile(name: str, t: int) -> Profile:
    """Structured formulas, valid for any T (no T×T matrices except for the optimal factorization)."""
    steps = np.arange(1, t + 1, dtype=np.float64)
    if name == "identity":
        return Profile(1.0, np.sqrt(steps))
    if name == "prefix":
        return Profile(math.sqrt(t), np.ones(t))
    if name == "tree":
        pops = np.array([k.bit_count() for k in range(1, t + 1)], dtype=np.float64)
        return Profile(math.sqrt(tree_levels(t)), np.sqrt(pops))
    if name == "sqrt":
        cum = np.cumsum(sqrt_coeffs(t) ** 2)
        return Profile(math.sqrt(float(cum[-1])), np.sqrt(cum))
    if name == "optimal":
        o = optimal_for(t)
        return Profile(max_column_norm(o.c), row_norms(o.b))
    raise ValueError(name)


def check_profiles(t: int) -> float:
    """Largest relative difference between the structured formulas and the dense matrices."""
    worst = 0.0
    for name in ("identity", "prefix", "tree", "sqrt"):
        b, c = dense_factorization(name, t)
        assert np.allclose(b @ c, prefix_matrix(t))
        p = profile(name, t)
        worst = max(worst, abs(max_column_norm(c) / p.sens - 1), float(np.max(np.abs(row_norms(b) / p.rows - 1))))
    return worst


def prefix_singular_values(t: int) -> Array:
    """Singular values of A: 1/(2 sin((2k - 1)π/(4T + 2))), k = 1..T."""
    k = np.arange(1, t + 1, dtype=np.float64)
    return 1 / (2 * np.sin((2 * k - 1) * math.pi / (4 * t + 2)))


def trace_norm_bound(t: int) -> float:
    """Lower bound on the RMSE of any factorization: ‖B‖_F maxcol(C) ≥ ‖A‖_*/√T, so RMSE ≥ ‖A‖_*/T."""
    return float(prefix_singular_values(t).sum()) / t


def psi_bound(t: int) -> float:
    """The closed form printed in Theorem 1 of Fichtenberger, Henzinger & Upadhyay (arXiv v6): 1 + ln((4T - 3)/5)/π."""
    return 1 + math.log((4 * t - 3) / 5) / math.pi


def sqrt_sum_bounds(t: int) -> tuple[float, float]:
    """Bounds on Σ_{k<T} r_k² from 1/(π(k + 1/2)) ≤ r_k² ≤ 1/(π(k + 1/4)) and comparing sums with integrals:
    1 + ln((2T + 1)/3)/π ≤ Σ ≤ 1 + (4/5 + ln((4T - 3)/5))/π."""
    return 1 + math.log((2 * t + 1) / 3) / math.pi, 1 + (0.8 + math.log((4 * t - 3) / 5)) / math.pi


def sqrt_sum_asymptotic(t: int) -> float:
    """Σ_{k<T} r_k² ≈ (ln T + γ + 4 ln 2)/π for large T."""
    return (math.log(t) + 0.5772156649015329 + 4 * math.log(2)) / math.pi


def wallis_ratio_range(t: int) -> tuple[float, float]:
    """(max_k r_k √(π(k + 1/4)), min_k r_k √(π(k + 1/2))) over 1 ≤ k < T: at most 1 and at least 1."""
    k = np.arange(1, t, dtype=np.float64)
    r = sqrt_coeffs(t)[1:]
    return float(np.max(r * np.sqrt(math.pi * (k + 0.25)))), float(np.min(r * np.sqrt(math.pi * (k + 0.5))))


# --- Several participations ---------------------------------------------------------------------


def participations(t: int, sep: int, epochs: int, offset: int) -> IntArray:
    """Steps (0-indexed) at which a record in slot `offset` participates: offset, offset + sep, ..."""
    return np.arange(offset, t, sep, dtype=np.int64)[:epochs]


def toeplitz_sensitivity(c: Array, t: int, sep: int, epochs: int) -> float:
    """max over offsets of ‖C u‖ for C = toeplitz(c) and u the participation indicator. Valid as the
    sensitivity because C has non-negative entries, so CᵀC ≥ 0 and aligned contributions are the worst."""
    assert np.all(c >= 0)
    best = 0.0
    for off in range(min(sep, t)):
        u = np.zeros(t)
        u[participations(t, sep, epochs, off)] = 1.0
        cu = np.convolve(c[:t], u)[:t]
        best = max(best, float(np.linalg.norm(cu)))
    return best


def tree_sensitivity(t: int, sep: int, epochs: int) -> float:
    """max over offsets of √(Σ_nodes (participations in the node)²)."""
    best = 0.0
    for off in range(min(sep, t)):
        steps = participations(t, sep, epochs, off)
        total = 0
        for lev in range(tree_levels(t)):
            counts = np.bincount(steps >> lev)
            total += int(np.sum(counts**2))
        best = max(best, math.sqrt(total))
    return best


def band_coeffs(t: int, band: int) -> Array:
    return sqrt_coeffs(t)[:band].copy()


def multi_sensitivity(name: str, t: int, sep: int, epochs: int) -> float:
    if name == "identity":
        return math.sqrt(min(epochs, math.ceil(t / sep)))
    if name == "tree":
        return tree_sensitivity(t, sep, epochs)
    if name == "sqrt":
        return toeplitz_sensitivity(sqrt_coeffs(t), t, sep, epochs)
    if name == "banded":
        return toeplitz_sensitivity(band_coeffs(t, sep), t, sep, epochs)
    raise ValueError(name)


def decoder_rows(name: str, t: int, sep: int) -> Array:
    """Row norms of B for the mechanisms used in training (the error of each prefix sum per unit noise)."""
    if name == "banded":
        h = inverse_series(band_coeffs(t, sep), t)
        return np.sqrt(np.cumsum(np.cumsum(h) ** 2))
    return profile(name, t).rows


def step_noise_coeffs(name: str, t: int, sep: int) -> Array:
    """h with ζ = toeplitz(h) z the noise added to each step's gradient: C^{-1} for square factorizations."""
    if name == "identity":
        return np.ones(1)
    if name == "sqrt":
        return inverse_series(sqrt_coeffs(t), t)
    if name == "banded":
        return inverse_series(band_coeffs(t, sep), t)
    raise ValueError(name)


# --- Noise for training ---------------------------------------------------------------------------


def fft_convolve(h: Array, z: Array) -> Array:
    """Causal convolution along the time axis: out_t = Σ_{s ≤ t} h_{t-s} z_s, for every column of z."""
    t = z.shape[0]
    if h.size == 1:
        return h[0] * z
    size = 1 << (2 * t - 1).bit_length()
    out = np.fft.irfft(np.fft.rfft(h, size)[:, None] * np.fft.rfft(z, size, axis=0), size, axis=0)
    return np.asarray(out[:t], dtype=np.float64)


def tree_prefix_noise(rng: np.random.Generator, t: int, dim: int, scale: float) -> Array:
    """N_t = the sum of the noise of the dyadic nodes covering [0, t), for t = 1..T."""
    out = np.zeros((t, dim))
    ends = np.arange(1, t + 1, dtype=np.int64)
    for lev in range(tree_levels(t)):
        node_noise = scale * rng.standard_normal((t // 2**lev + 1, dim))
        has = (ends >> lev) & 1 == 1
        # The level-l node in the decomposition of [0, e) starts at (e >> (l+1)) << (l+1), node index e >> (l+1) * 2.
        idx = (ends[has] >> (lev + 1)) * 2
        out[has] += node_noise[idx]
    return out


def step_noise(rng: np.random.Generator, name: str, t: int, dim: int, scale: float, sep: int) -> Array:
    """The noise added to each step's clipped gradient sum, (T, dim): differences of the prefix-sum noise B z."""
    if name == "tree":
        prefix = tree_prefix_noise(rng, t, dim, scale)
        return np.diff(prefix, axis=0, prepend=np.zeros((1, dim)))
    z = scale * rng.standard_normal((t, dim))
    return fft_convolve(step_noise_coeffs(name, t, sep), z)


# --- Training in a fixed order ----------------------------------------------------------------------


@dataclass(frozen=True)
class FixedOrderConfig:
    mechanism: str  # identity, tree, sqrt or banded
    noise: float  # standard deviation of z in units of the clipping norm (0 for no noise)
    batch: int = BATCH_RUN
    epochs: int = 1
    lr: float = 1.0
    clip: float = CLIP_RUN


def train_fixed_order(rng: np.random.Generator, task: Task, cfg: FixedOrderConfig) -> Params:
    """Gradient descent over a fixed data order (one public random permutation, the same every epoch), with
    the noise of the chosen factorization: w_{t+1} = w_t - η (x_t + ζ_t)/B, so w_t = w_0 - η Ŝ_{t-1}/B."""
    n = task.train.y.size
    per_epoch = n // cfg.batch
    t_total = per_epoch * cfg.epochs
    p = init_params(rng)
    order = rng.permutation(n)
    dim = p.size
    zeta = (
        step_noise(rng, cfg.mechanism, t_total, dim, cfg.noise * cfg.clip, per_epoch)
        if cfg.noise > 0
        else np.zeros((t_total, dim))
    )
    n1 = p.w1.size
    for t in range(t_total):
        k = t % per_epoch
        idx = order[k * cfg.batch : (k + 1) * cfg.batch]
        cs = clipped_grad_sum(p, task.train.x[idx], task.train.y[idx], cfg.clip)
        g1 = cs.g1 + zeta[t, :n1].reshape(cs.g1.shape)
        g2 = cs.g2 + zeta[t, n1:]
        p = Params(p.w1 - cfg.lr * g1 / cfg.batch, p.w2 - cfg.lr * g2 / cfg.batch)
    return p


@cache
def dpsgd_sigma(steps: int, eps: float) -> float:
    return noise_multiplier_for(BATCH_RUN / N_TRAIN, steps, eps, DELTA_RUN)


def noise_scale(name: str, epochs: int, eps: float) -> float:
    """Standard deviation of z, in units of C, for (ε, δ): sensitivity / μ(ε, δ), exact Gaussian accounting."""
    per_epoch = N_TRAIN // BATCH_RUN
    sens = multi_sensitivity(name, per_epoch * epochs, per_epoch, epochs)
    return sens / gaussian_mu(eps, DELTA_RUN)


def run_method(rng: np.random.Generator, task: Task, name: str, epochs: int, eps: float, lr: float) -> float:
    steps = epochs * (N_TRAIN // BATCH_RUN)
    if name == "dpsgd":
        cfg = DpsgdConfig(q=BATCH_RUN / N_TRAIN, sigma=dpsgd_sigma(steps, eps), clip=CLIP_RUN, lr=lr, steps=steps)
        return accuracy(train_dpsgd(rng, task.train, cfg)[0], task.test)
    cfg = FixedOrderConfig(name, noise_scale(name, epochs, eps), epochs=epochs, lr=lr)
    return accuracy(train_fixed_order(rng, task, cfg), task.test)


def final_prefix_noise(name: str, epochs: int, eps: float) -> float:
    """Standard deviation, in units of C, of the noise in the last prefix sum, which the final model carries."""
    per_epoch = N_TRAIN // BATCH_RUN
    t = per_epoch * epochs
    if name == "dpsgd":
        return dpsgd_sigma(t, eps) * math.sqrt(t)
    return noise_scale(name, epochs, eps) * float(decoder_rows(name, t, per_epoch)[-1])


# --- The chapter's numbers ----------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Structured formulas against dense matrices")
    print(f"T=256: largest relative difference = {check_profiles(256):.1e}")
    s = prefix_singular_values(256)
    print(
        f"T=256: singular values closed form vs SVD, largest difference = "
        f"{float(np.max(np.abs(np.sort(s) - np.sort(np.linalg.svd(prefix_matrix(256), compute_uv=False))))):.1e}"
    )
    r = sqrt_coeffs(6)
    print("square-root coefficients r_0..r_5: " + ", ".join(f"{x:.4f}" for x in r))
    s4 = toeplitz_lower(sqrt_coeffs(4), 4)
    print(f"T=4: max |C C - A| = {float(np.max(np.abs(s4 @ s4 - prefix_matrix(4)))):.1e}")
    h = inverse_series(sqrt_coeffs(6), 6)
    print("inverse coefficients h_0..h_5: " + ", ".join(f"{x:.4f}" for x in h))
    big = inverse_series(sqrt_coeffs(100_000), 100_000)
    print(f"sum of h_k^2 over 100000 terms = {float(np.sum(big**2)):.4f}  (4/pi = {4 / math.pi:.4f})")

    print("== Single participation, mu = 1 (noise sd = sensitivity)")
    for t in (100, 256, 10_000):
        names = ("identity", "prefix", "tree", "sqrt") + (("optimal",) if t <= 256 else ())
        for name in names:
            p = profile(name, t)
            print(f"T={t} {name}: sensitivity={p.sens:.3f}  rmse={p.rmse:.2f}  max error={p.max_error:.2f}")
        print(f"T={t} lower bound on rmse (trace norm) = {trace_norm_bound(t):.2f}")
        cum = float(np.sum(sqrt_coeffs(t) ** 2))
        lo, hi = sqrt_sum_bounds(t)
        print(
            f"T={t} square root: sum r_k^2 = {cum:.3f}  bounds [{lo:.3f}, {hi:.3f}]  asymptotic "
            f"{sqrt_sum_asymptotic(t):.3f}  printed Psi(T) = {psi_bound(t):.3f}"
        )
    for t in (100, 256):
        print(f"T={t} optimal factorization: {optimal_for(t).iterations} fixed-point iterations")
    hi_ratio, lo_ratio = wallis_ratio_range(100_000)
    print(f"k < 100000: max r_k sqrt(pi(k+1/4)) = {hi_ratio:.6f}  min r_k sqrt(pi(k+1/2)) = {lo_ratio:.6f}")
    print(f"tree levels: T=100: {tree_levels(100)}  T=10000: {tree_levels(10_000)}")

    print("== Several participations, separation b=100")
    for epochs in (1, 2, 5, 10, 20):
        t = 100 * epochs
        parts: list[str] = []
        for name in MECHANISMS:
            sens = multi_sensitivity(name, t, 100, epochs)
            last = float(decoder_rows(name, t, 100)[-1])
            parts.append(f"{name} sens={sens:.3f} last={sens * last:.2f}")
        print(f"E={epochs} T={t}: " + "  ".join(parts))
    c1 = math.sqrt(float(np.sum(sqrt_coeffs(100) ** 2)))
    print(f"banded check: sqrt(20)*||c||={math.sqrt(20) * c1:.3f}")

    print("== Noise for training (units of C), delta=1e-5, B=100, n=10000")
    for eps in EPS_TRAIN:
        print(f"eps={eps:g}: mu={gaussian_mu(eps, DELTA_RUN):.3f}")
        for epochs in EPOCHS_RUN:
            steps = epochs * 100
            parts = [f"dpsgd sigma={dpsgd_sigma(steps, eps):.3g}"]
            parts += [f"{name} s={noise_scale(name, epochs, eps):.3f}" for name in MECHANISMS]
            print(f"  E={epochs} T={steps}: " + "  ".join(parts))
            parts = [f"{name}={final_prefix_noise(name, epochs, eps):.2f}" for name in ("dpsgd",) + MECHANISMS]
            print(f"  E={epochs} noise in the final prefix sum: " + "  ".join(parts))
    hs = step_noise_coeffs("sqrt", 2000, 100)
    print(f"per-step noise sd of the square root, relative to s: {math.sqrt(float(np.sum(hs**2))):.3f}")

    print("== Training at equal epsilon (best step size of 0.5, 2; mean +- sd over 3 runs)")
    task = make_task()
    for eps in EPS_TRAIN:
        for epochs in EPOCHS_RUN:
            parts = []
            for name in ("dpsgd",) + MECHANISMS:
                by_lr = {lr: [run_method(rng, task, name, epochs, eps, lr) for _ in range(SEEDS_RUN)] for lr in LRS_RUN}
                best = max(LRS_RUN, key=lambda lr: float(np.mean(by_lr[lr])))
                accs = by_lr[best]
                parts.append(f"{name} {np.mean(accs):.4f}+-{np.std(accs):.4f} (lr={best:g})")
            print(f"eps={eps:g} E={epochs}: " + "  ".join(parts))
    for epochs in EPOCHS_RUN:
        by_lr = {
            lr: [
                accuracy(
                    train_fixed_order(rng, task, FixedOrderConfig("identity", 0.0, epochs=epochs, lr=lr)), task.test
                )
                for _ in range(SEEDS_RUN)
            ]
            for lr in LRS_RUN
        }
        best = max(LRS_RUN, key=lambda lr: float(np.mean(by_lr[lr])))
        accs = by_lr[best]
        print(f"no noise, clipped, fixed order, E={epochs}: {np.mean(accs):.4f}+-{np.std(accs):.4f} (lr={best:g})")


# --- Figures --------------------------------------------------------------------------------


def plot_noise_paths(rng: np.random.Generator) -> Figure:
    """One coordinate of the step noise and of its running sum, independent against square root, μ = 1."""
    t = 1000
    sens_sqrt = profile("sqrt", t).sens
    z = rng.standard_normal(t)
    indep = z
    corr = fft_convolve(step_noise_coeffs("sqrt", t, t), sens_sqrt * z[:, None])[:, 0]
    tree = np.diff(tree_prefix_noise(rng, t, 1, math.sqrt(tree_levels(t)))[:, 0], prepend=0.0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    shown = 60
    axes[0].plot(
        np.arange(1, shown + 1),
        indep[:shown],
        color=COLORS["identity"],
        marker="o",
        ms=3,
        label="independent (identity)",
    )
    axes[0].plot(
        np.arange(1, shown + 1), corr[:shown], color=COLORS["sqrt"], marker="s", ms=3, label="correlated (square root)"
    )
    axes[0].set_xlabel("step $t$")
    axes[0].set_ylabel("noise on the step's gradient")
    axes[0].set_title("Noise added to each step")
    axes[0].legend(fontsize=8)
    steps = np.arange(1, t + 1)
    axes[1].plot(steps, np.cumsum(indep), color=COLORS["identity"], label="independent (identity)")
    axes[1].plot(steps, np.cumsum(tree), color=COLORS["tree"], label="tree")
    axes[1].plot(steps, np.cumsum(corr), color=COLORS["sqrt"], label="square root")
    for name in ("identity", "sqrt"):
        sd = profile(name, t).scaled()
        axes[1].fill_between(steps, -2 * sd, 2 * sd, color=COLORS[name], alpha=0.12, lw=0)
    axes[1].set_xlabel("step $t$")
    axes[1].set_ylabel("noise in the prefix sum $S_t$")
    axes[1].set_title("Accumulated noise (bands: two standard deviations)")
    axes[1].legend(fontsize=8)
    return fig


def plot_errors() -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    t = 256
    steps = np.arange(1, t + 1)
    for name in ("identity", "prefix", "tree", "sqrt", "optimal"):
        axes[0].plot(
            steps,
            profile(name, t).scaled(),
            color=COLORS[name],
            label=FACTOR_LABELS[name],
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step $t$")
    axes[0].set_ylabel("standard deviation of the error of $S_t$")
    axes[0].set_title(r"$T = 256$, one participation, $\mu = 1$")
    axes[0].legend(fontsize=8)
    ts = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 10_000]
    for name in ("identity", "prefix", "tree", "sqrt"):
        axes[1].loglog(
            ts,
            [profile(name, x).rmse for x in ts],
            color=COLORS[name],
            marker="o",
            ms=3,
            label=FACTOR_LABELS[name],
        )
    small = [x for x in ts if x <= 256]
    axes[1].loglog(
        small,
        [profile("optimal", x).rmse for x in small],
        color=COLORS["optimal"],
        marker="s",
        ms=4,
        label=FACTOR_LABELS["optimal"],
    )
    axes[1].loglog(ts, [trace_norm_bound(x) for x in ts], color="black", ls=":", label="lower bound (trace norm)")
    axes[1].set_xlabel("number of steps $T$")
    axes[1].set_ylabel("root-mean-square error over $t$")
    axes[1].set_title("Error against the horizon")
    axes[1].legend(fontsize=8)
    return fig


def plot_matrices() -> Figure:
    t = 16
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
    for j, name in enumerate(("tree", "sqrt", "optimal")):
        b, c = dense_factorization(name, t)
        for i, (m, what) in enumerate(((c, "C (encoder)"), (b, "B (decoder)"))):
            ax = axes[i, j]
            lim = float(np.max(np.abs(m)))
            im = ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
            ax.set_title(f"{FACTOR_LABELS[name]}: {what}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"{m.shape[1]} columns", fontsize=8)
            ax.set_ylabel(f"{m.shape[0]} rows", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return fig


def plot_accuracy(rng: np.random.Generator) -> Figure:
    """Test accuracy against ε at 5 epochs, one run per point, step size 1."""
    task = make_task()
    eps_list = (1.0, 2.0, 4.0, 8.0, 16.0)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for name, marker in (("dpsgd", "o"), ("identity", "v"), ("tree", "^"), ("sqrt", "s"), ("banded", "D")):
        accs = [run_method(rng, task, name, 5, e, 1.0) for e in eps_list]
        ax.semilogx(eps_list, accs, marker=marker, color=COLORS[name], label=LABELS[name])
    ax.set_xticks(eps_list, [f"{e:g}" for e in eps_list])
    ax.set_xlabel(r"$\varepsilon$ at $\delta = 10^{-5}$")
    ax.set_ylabel("test accuracy")
    ax.set_title("5 epochs, batch 100, learning rate 1")
    ax.legend(fontsize=8)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="directory for the figures")
    args = parser.parse_args()
    seed: int = args.seed
    out: Path = args.out

    print_table(np.random.default_rng(seed))

    out.mkdir(parents=True, exist_ok=True)
    fig_rng = np.random.default_rng(seed + 1)
    figures = {
        "mfac_noise_paths.svg": plot_noise_paths(fig_rng),
        "mfac_errors.svg": plot_errors(),
        "mfac_matrices.svg": plot_matrices(),
        "mfac_accuracy.svg": plot_accuracy(fig_rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
