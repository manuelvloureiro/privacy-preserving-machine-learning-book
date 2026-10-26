"""Check the numbers in the user-level differential privacy chapter and draw its figures.

    uv run python code/part05-dp-ml/user_level.py
    uv run python code/part05-dp-ml/user_level.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 07-user-level.md (they match exactly with the default seed), then saves
the chapter's figures.

The running example of the part (dp_sgd.make_task: 10,000 training records, a 50 → 64 → 1 tanh network)
with its records assigned to 1,000 users. User sizes are heterogeneous: each user gets one record, and
the other 9,000 are spread with probabilities proportional to log-normal weights (log standard deviation
1.5), so the largest user holds 348 records and the median user 4. A user's records are neighbours in
input space (each user is a nearest-neighbour cluster around a random seed record), so they are
correlated, as one person's messages or photos are. The population is fixed (USER_SEED); --seed changes
only the training randomness.

Three ways of training with a user-level guarantee are compared at the same user-level (ε, δ):
  (a) example-level DP-SGD on all records, accounted for the largest user;
  (b) user-level DP-SGD: sample users, average each user's gradient, clip it, add noise;
  (c) contribution bounding: keep at most k records per user, then example-level DP-SGD accounted
      for a user with k records.
The accountant for (a) and (c) is the Rényi divergence between N(0, σ²) and the mixture
Σ_m Binom(k, q)(m) N(m, σ²): a user with k records has a binomial number of them in each Poisson batch.
"""

import os

# The models are tiny; multi-threaded BLAS only adds overhead.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure

configure("user_level")

import matplotlib.pyplot as plt
import numpy as np
from dp_sgd import (
    ORDERS,
    Dataset,
    DpsgdConfig,
    Params,
    accuracy,
    backward,
    bisect,
    epsilon_from_rdp,
    init_params,
    logsumexp,
    make_task,
    noise_multiplier_for,
    rdp_curve,
    subsampled_gaussian_rdp,
    train_dpsgd,
)
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

# The user population.
N_USERS = 1_000
LOG_SPREAD = 1.5  # standard deviation of the log-normal user weights
USER_SEED = 42  # the population is a fixed object, like the task
HEAVY = 50  # a user with at least this many records is "heavy" in the heterogeneity section

# Training settings.
Q_RUN, STEPS_RUN, CLIP_RUN, LR_RUN = 0.01, 10_000, 0.5, 0.5  # the DP-SGD chapter's run: ε = 6.71 per record
BATCH, STEPS_CMP = 100, 1_000  # the comparison: every strategy takes 1,000 steps of expected batch 100 units
DELTA = 1e-5
SIGMA_DEFAULT = 1.0  # the part's running example: ε = 6.71 per record
K_GRID: tuple[int, ...] = (1, 2, 5, 10, 50)
EPS_GRID: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
CAP_GRID: tuple[int, ...] = (1, 2, 5, 10, 20, 50)
CAP_RUN = 2  # the cap used in the comparison across budgets

# Orders for the user-level accountant. A sub-grid of the part's ORDERS: the conversion is valid at any
# set of orders, and a coarser one only costs a little tightness.
USER_ORDERS: list[float] = (
    [1 + x / 20 for x in range(1, 20)]
    + [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    + [12.0, 14.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0, 128.0, 256.0]
)


# --- The user population -------------------------------------------------------------------------


def user_counts(rng: np.random.Generator, n: int, users: int, spread: float) -> IntArray:
    """Records per user, largest first: one each, plus n - users spread by log-normal weights."""
    weights = np.exp(spread * rng.standard_normal(users))
    counts = 1 + rng.multinomial(n - users, weights / weights.sum())
    return np.sort(counts)[::-1].astype(np.int64)


def assign_users(rng: np.random.Generator, x: Array, counts: IntArray) -> IntArray:
    """User of each record: each user, largest first, takes a random free record and its nearest free neighbours."""
    n = x.shape[0]
    free_mask = np.ones(n, dtype=bool)
    owner = np.empty(n, dtype=np.int64)
    sq = (x**2).sum(axis=1)
    for u, cu in enumerate(counts.tolist()):
        free = np.flatnonzero(free_mask)
        seed = int(free[rng.integers(free.size)])
        dist = sq[free] - 2 * x[free] @ x[seed]  # squared distance up to the constant ‖x_seed‖²
        take = free if cu >= free.size else free[np.argpartition(dist, cu - 1)[:cu]]
        owner[take] = u
        free_mask[take] = False
    return owner


@dataclass(frozen=True)
class Users:
    data: Dataset  # records sorted by user
    starts: IntArray  # first record of each user
    counts: IntArray  # records per user, largest user first

    def records(self, u: int) -> slice:
        return slice(int(self.starts[u]), int(self.starts[u] + self.counts[u]))


def make_users(data: Dataset, users: int = N_USERS, seed: int = USER_SEED) -> Users:
    rng = np.random.default_rng(seed)
    counts = user_counts(rng, data.y.size, users, LOG_SPREAD)
    owner = assign_users(rng, data.x, counts)
    order = np.argsort(owner, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    return Users(Dataset(data.x[order], data.y[order]), starts, counts)


def cap_users(rng: np.random.Generator, users: Users, k: int) -> Users:
    """Contribution bounding: keep a random k of each user's records (all of them if fewer)."""
    keep: list[IntArray] = []
    for u in range(users.counts.size):
        start, count = int(users.starts[u]), int(users.counts[u])
        idx = np.arange(start, start + count, dtype=np.int64)
        keep.append(idx if idx.size <= k else np.sort(rng.choice(idx, k, replace=False)))
    counts = np.minimum(users.counts, k)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    sel = np.concatenate(keep)
    return Users(Dataset(users.data.x[sel], users.data.y[sel]), starts, counts)


# --- Accounting: a user with k records under example-level Poisson sampling ---------------------


def log_binom_pmf(k: int, q: float) -> Array:
    """ln P(M = m) for M ~ Binom(k, q), m = 0..k: how many of a user's k records are in the batch."""
    return np.array(
        [
            math.lgamma(k + 1)
            - math.lgamma(m + 1)
            - math.lgamma(k - m + 1)
            + m * math.log(q)
            + (k - m) * math.log1p(-q)
            for m in range(k + 1)
        ]
    )


def _log_mixture_ratio(z: Array, log_w: Array, shifts: Array, sigma: float) -> Array:
    """ln Σ_m w_m N(z; s_m, σ²)/N(z; 0, σ²) = ln Σ_m w_m exp((2 s_m z - s_m²)/(2σ²))."""
    e = log_w[None, :] + (2 * shifts[None, :] * z[:, None] - shifts[None, :] ** 2) / (2 * sigma**2)
    top = e.max(axis=1)
    return top + np.log(np.exp(e - top[:, None]).sum(axis=1))


def _log_normal_pdf(z: Array, sigma: float) -> Array:
    return -(z**2) / (2 * sigma**2) - math.log(sigma * math.sqrt(2 * math.pi))


def mixture_rdp_both(alpha: float, log_w: Array, shifts: Array, sigma: float) -> tuple[float, float]:
    """(D_α(P_mix‖P_0), D_α(P_0‖P_mix)) by the trapezoid rule on a grid of step σ/10.

    The first integrand has its mass near 0 and, for large α, near α·max shift; the grid covers both."""
    h = sigma / 10
    top = float(shifts.max())
    z = np.arange(-12 * sigma, alpha * top + 12 * sigma + h, h)
    a = logsumexp(_log_normal_pdf(z, sigma) + alpha * _log_mixture_ratio(z, log_w, shifts, sigma)) + math.log(h)
    z2 = np.arange(-12 * sigma - top, 12 * sigma + top + h, h)
    b = logsumexp(_log_normal_pdf(z2, sigma) - (alpha - 1) * _log_mixture_ratio(z2, log_w, shifts, sigma))
    b += math.log(h)
    return a / (alpha - 1), b / (alpha - 1)


def user_rdp(alpha: float, q: float, sigma: float, k: int) -> float:
    """RDP of one example-level DP-SGD step for a user with k records (aligned gradients, the tight case)."""
    shifts = np.arange(k + 1, dtype=np.float64)
    return max(mixture_rdp_both(alpha, log_binom_pmf(k, q), shifts, sigma))


def unit_rdp_integer(alpha: int, q: float, sigma: float, k: int) -> float:
    """Exact RDP at integer α when the user's k records are sampled together (user-level sampling, summed
    per-record clipped gradients): the binomial sum with exp(k²(j² - j)/(2σ²))."""
    terms = np.array(
        [
            math.lgamma(alpha + 1)
            - math.lgamma(j + 1)
            - math.lgamma(alpha - j + 1)
            + (alpha - j) * math.log1p(-q)
            + j * math.log(q)
            + k * k * (j * j - j) / (2 * sigma**2)
            for j in range(alpha + 1)
        ]
    )
    return logsumexp(terms) / (alpha - 1)


def pair_moment_alpha2(q: float, sigma: float, k: int) -> float:
    """ε(2) from the moment formula A_2 = E[exp(M_1 M_2/σ²)], M_1, M_2 iid Binom(k, q)."""
    p = np.exp(log_binom_pmf(k, q))
    m = np.arange(k + 1, dtype=np.float64)
    return math.log(float(p @ np.exp(np.outer(m, m) / sigma**2) @ p))


def eps_from_curve(orders: Sequence[float], rdp: Sequence[float], delta: float) -> tuple[float, float]:
    """(ε, best order): the improved conversion of the DP-SGD chapter, at the given orders."""
    return eps_from_curve_log(orders, rdp, math.log(delta))


def eps_from_curve_log(orders: Sequence[float], rdp: Sequence[float], log_delta: float) -> tuple[float, float]:
    """The same conversion with ln δ as input, for the tiny δ that group privacy needs per record."""
    return min(
        (max(0.0, e + math.log((a - 1) / a) - (log_delta + math.log(a)) / (a - 1)), a)
        for a, e in zip(orders, rdp, strict=True)
    )


@cache
def user_epsilon(q: float, sigma: float, steps: int, k: int, delta: float) -> tuple[float, float]:
    """User-level (ε, best order) of example-level DP-SGD for a user with k records."""
    rdp = [steps * user_rdp(a, q, sigma, k) for a in USER_ORDERS]
    return eps_from_curve(USER_ORDERS, rdp, delta)


@cache
def sigma_for_user(q: float, steps: int, k: int, eps: float, delta: float) -> float:
    """Smallest σ (three significant digits) for which example-level DP-SGD is (ε, δ) for a k-record user.
    Bisection on ln σ, since σ ranges over three orders of magnitude as k grows."""
    log_sigma = bisect(
        lambda ls: user_epsilon(q, math.exp(ls), steps, k, delta)[0] <= eps, math.log(0.3), math.log(10.0 * k + 20), 18
    )
    sigma = math.exp(log_sigma)
    digits = 2 - math.floor(math.log10(sigma))
    return math.ceil(sigma * 10**digits) / 10**digits


def record_epsilon(q: float, sigma: float, steps: int, delta: float) -> float:
    return epsilon_from_rdp([steps * e for e in rdp_curve(q, sigma)], delta)[0]


def record_epsilon_log(q: float, sigma: float, steps: int, log_delta: float) -> float:
    return eps_from_curve_log(ORDERS, [steps * e for e in rdp_curve(q, sigma)], log_delta)[0]


def generic_group_epsilon(q: float, sigma: float, steps: int, k: int, delta: float) -> tuple[float, float]:
    """Group privacy from a record-level guarantee: choose δ_rec so that δ_rec (e^{kε}-1)/(e^ε-1) = δ.
    Returns (k ε_rec, ε_rec)."""

    def log_user_delta(log_d: float) -> float:
        e = record_epsilon_log(q, sigma, steps, log_d)
        if e <= 0:
            return log_d + math.log(k)
        # ln((e^{kε} - 1)/(e^ε - 1)), computed without overflow
        return log_d + k * e + math.log1p(-math.exp(-k * e)) - e - math.log1p(-math.exp(-e))

    # ε_rec falls as δ_rec grows, but the user's δ need not be monotone in δ_rec: scan a grid of ln δ_rec and
    # keep the largest feasible one (it gives the smallest ε). No feasible point means no (ε, δ) at all.
    grid = np.linspace(math.log(delta) - 2000, math.log(delta), 4001)
    feasible = [float(ld) for ld in grid if log_user_delta(float(ld)) <= math.log(delta)]
    if not feasible:
        return math.inf, math.inf
    e = record_epsilon_log(q, sigma, steps, max(feasible))
    return k * e, e


# --- Training ----------------------------------------------------------------------------------


def run_example_level(rng: np.random.Generator, users: Users, sigma: float, test: Dataset) -> float:
    """Example-level DP-SGD on all of `users`' records: Poisson rate BATCH/n, STEPS_CMP steps, C = 0.5, η = 0.5."""
    q = BATCH / users.data.y.size
    params, _ = train_dpsgd(rng, users.data, DpsgdConfig(q=q, sigma=sigma, clip=CLIP_RUN, lr=LR_RUN, steps=STEPS_CMP))
    return accuracy(params, test)


def sigma_example_level(users: Users, eps: float) -> float:
    """σ that example-level DP-SGD on `users` needs for user-level (ε, δ), accounted for its largest user."""
    return sigma_for_user(BATCH / users.data.y.size, STEPS_CMP, int(users.counts.max()), eps, DELTA)


def sigma_user_level(users: Users, eps: float) -> float:
    """σ that user-level DP-SGD needs: the ordinary accountant with users as the unit."""
    return noise_multiplier_for(BATCH / users.counts.size, STEPS_CMP, eps, DELTA)


def user_grads(p: Params, users: Users, sel: IntArray) -> tuple[Array, Array]:
    """Each selected user's average gradient, flattened: (|sel|, w1 size) and (|sel|, w2 size)."""
    chosen: list[int] = [int(u) for u in sel]
    idx = np.concatenate([np.arange(int(users.starts[u]), int(users.starts[u] + users.counts[u])) for u in chosen])
    bw = backward(p, users.data.x[idx], users.data.y[idx])
    g1 = np.empty((sel.size, p.w1.size))
    g2 = np.empty((sel.size, p.w2.size))
    pos = 0
    for row, u in enumerate(chosen):
        c = int(users.counts[u])
        g1[row] = (bw.d1[pos : pos + c].T @ bw.fwd.a0[pos : pos + c]).ravel() / c
        g2[row] = bw.d2[pos : pos + c] @ bw.fwd.a1b[pos : pos + c] / c
        pos += c
    return g1, g2


def train_user_dpsgd(
    rng: np.random.Generator,
    users: Users,
    sigma: float,
    q: float = BATCH / N_USERS,
    clip: float = CLIP_RUN,
    lr: float = LR_RUN,
    steps: int = STEPS_CMP,
) -> Params:
    """User-level DP-SGD: Poisson-sample users, average each user's gradient over its records, clip the
    average to C, add N(0, σ²C²) to the sum, divide by the expected number of users qU."""
    n_users = users.counts.size
    p = init_params(rng, users.data.x.shape[1])
    b_expected = q * n_users
    for _ in range(steps):
        sel = np.flatnonzero(rng.random(n_users) < q).astype(np.int64)
        s1 = np.zeros(p.w1.size)
        s2 = np.zeros(p.w2.size)
        if sel.size:
            g1, g2 = user_grads(p, users, sel)
            norms = np.sqrt((g1**2).sum(axis=1) + (g2**2).sum(axis=1))
            scale = np.minimum(1.0, clip / np.maximum(norms, 1e-300))
            s1 = scale @ g1
            s2 = scale @ g2
        s1 = s1 + sigma * clip * rng.standard_normal(s1.shape)
        s2 = s2 + sigma * clip * rng.standard_normal(s2.shape)
        p = Params(p.w1 - lr * s1.reshape(p.w1.shape) / b_expected, p.w2 - lr * s2 / b_expected)
    return p


def heavy_test_mask(users: Users, test: Dataset) -> NDArray[np.bool_]:
    """Test points whose nearest training record belongs to a heavy user (at least HEAVY records)."""
    owner = np.repeat(np.arange(users.counts.size), users.counts)
    heavy_rec = users.counts[owner] >= HEAVY
    x = users.data.x
    sq = (x**2).sum(axis=1)
    nearest = np.argmin(sq[None, :] - 2 * test.x @ x.T, axis=1)
    return heavy_rec[nearest]


# --- The chapter's numbers ------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    task = make_task()
    users = make_users(task.train)
    c = users.counts
    n = int(c.sum())
    print("== The user population")
    print(
        f"users={c.size}  records={n}  mean={c.mean():.0f}  median={np.median(c):.0f}  largest={c[0]}  "
        f"users with 1 record={int((c == 1).sum())}  users with >= 10 records={int((c >= 10).sum())}  "
        f"users with >= {HEAVY} records={int((c >= HEAVY).sum())}"
    )
    print(
        f"largest 10 users hold {c[:10].sum() / n:.1%} of the records; users with >= {HEAVY} records hold "
        f"{c[c >= HEAVY].sum() / n:.1%}; the smallest half of the users hold {np.sort(c)[: c.size // 2].sum() / n:.1%}"
    )
    labels = [abs(float(users.data.y[users.records(u)].mean())) for u in range(10)]
    print(
        f"label imbalance |mean y| in the 10 largest users: {np.mean(labels):.2f} (whole set "
        f"{abs(float(task.train.y.mean())):.2f})"
    )

    print("== Accountant checks")
    print(
        f"k=1 against the subsampled Gaussian, alpha=4.1: {user_rdp(4.1, Q_RUN, 1.0, 1):.6e} vs "
        f"{subsampled_gaussian_rdp(4.1, Q_RUN, 1.0):.6e}"
    )
    for k in (2, 5):
        lw = log_binom_pmf(k, Q_RUN)
        a, b = mixture_rdp_both(2.0, lw, np.arange(k + 1, dtype=np.float64), 1.0)
        print(
            f"k={k}, alpha=2: integral {a:.6e}  pair-moment formula {pair_moment_alpha2(Q_RUN, 1.0, k):.6e}  "
            f"reverse direction {b:.6e}"
        )
    print(
        f"sampled together, k=2, alpha=3: binomial sum {unit_rdp_integer(3, Q_RUN, 1.0, 2):.6e}  "
        f"same as sigma/k: {subsampled_gaussian_rdp(3.0, Q_RUN, 0.5):.6e}"
    )

    print("== A user with k records under the running example (q=0.01, sigma=1, T=10000, delta=1e-5)")
    for k in K_GRID:
        exact, order = user_epsilon(Q_RUN, SIGMA_DEFAULT, STEPS_RUN, k, DELTA)
        generic, rec = generic_group_epsilon(Q_RUN, SIGMA_DEFAULT, STEPS_RUN, k, DELTA)
        rate = record_epsilon(min(1.0, k * Q_RUN), SIGMA_DEFAULT, STEPS_RUN, DELTA)
        print(
            f"k={k}: user eps={exact:.2f} (alpha={order:g})  group privacy {generic:.2f} (record eps {rec:.2f})  "
            f"one record at rate kq: {rate:.2f}"
        )

    print("== Noise for a user-level budget (1,000 steps, expected batch 100 units)")
    k_max = int(c[0])
    capped = {k: cap_users(rng, users, k) for k in CAP_GRID}
    for eps in EPS_GRID:
        print(
            f"eps={eps:g}: record-level sigma={noise_multiplier_for(BATCH / n, STEPS_CMP, eps, DELTA):.3g}  "
            f"(a) all records, k={k_max}: sigma={sigma_example_level(users, eps):.3g}  "
            f"(b) user-level: sigma={sigma_user_level(users, eps):.3g}  "
            f"(c) cap {CAP_RUN}: sigma={sigma_example_level(capped[CAP_RUN], eps):.3g}"
        )

    print("== Contribution bounding at user-level eps=4")
    for k in CAP_GRID:
        cu = capped[k]
        m = int(cu.counts.sum())
        sigma = sigma_example_level(cu, 4.0)
        acc = run_example_level(rng, cu, sigma, task.test)
        print(
            f"cap k={k}: records kept={m} ({m / n:.1%})  q={BATCH / m:.4f}  kq={k * BATCH / m:.3f}  sigma={sigma:.3g}  "
            f"test accuracy {acc:.4f}"
        )

    print("== Three strategies at the same user-level budget")
    heavy = heavy_test_mask(users, task.test)
    print(f"test points nearest to a heavy user's record: {heavy.mean():.1%}")
    for eps in EPS_GRID:
        acc_a = run_example_level(rng, users, sigma_example_level(users, eps), task.test)
        p_b = train_user_dpsgd(rng, users, sigma_user_level(users, eps))
        cu = capped[CAP_RUN]
        acc_c = run_example_level(rng, cu, sigma_example_level(cu, eps), task.test)
        print(
            f"eps={eps:g}: (a) example-level, all records {acc_a:.4f}  (b) user-level {accuracy(p_b, task.test):.4f}  "
            f"(c) cap {CAP_RUN} {acc_c:.4f}"
        )
        if eps == 4.0:
            heavy_acc = accuracy(p_b, Dataset(task.test.x[heavy], task.test.y[heavy]))
            light_acc = accuracy(p_b, Dataset(task.test.x[~heavy], task.test.y[~heavy]))
            print(f"  (b) user-level: near heavy users {heavy_acc:.4f}  elsewhere {light_acc:.4f}")
    p_np, _ = train_dpsgd(rng, users.data, DpsgdConfig(sigma=0.0, clip=math.inf, steps=STEPS_CMP))
    heavy_acc = accuracy(p_np, Dataset(task.test.x[heavy], task.test.y[heavy]))
    light_acc = accuracy(p_np, Dataset(task.test.x[~heavy], task.test.y[~heavy]))
    print(
        f"non-private, 1000 steps: test accuracy {accuracy(p_np, task.test):.4f}  near heavy users {heavy_acc:.4f}  "
        f"elsewhere {light_acc:.4f}"
    )
    one = cap_users(rng, users, 1)
    p_one = train_user_dpsgd(rng, one, sigma_user_level(one, 4.0))
    print(f"user-level DP-SGD on one random record per user, eps=4: test accuracy {accuracy(p_one, task.test):.4f}")

    print("== Influence of heavy users")
    share_records = c[c >= HEAVY].sum() / n
    share_users = (c >= HEAVY).sum() / c.size
    print(
        f"users with >= {HEAVY} records: share of the expected update under example-level training "
        f"{share_records:.1%}, under user-level training at most {share_users:.1%}"
    )


# --- Figures ----------------------------------------------------------------------------------------


def plot_population() -> Figure:
    users = make_users(make_task().train)
    c = users.counts
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.unique(np.round(np.logspace(0, math.log10(float(c.max()) + 1), 25)))
    axes[0].hist(c, bins=bins, color="tab:blue", edgecolor="white")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("records per user")
    axes[0].set_ylabel("number of users")
    axes[0].set_title(f"{c.size:,} users, {int(c.sum()):,} records")
    share = np.cumsum(c) / c.sum()
    axes[1].plot(np.arange(1, c.size + 1) / c.size * 100, share * 100, color="black")
    axes[1].plot([0, 100], [0, 100], color="tab:gray", ls=":", label="equal sizes")
    axes[1].set_xlabel("largest users, % of all users")
    axes[1].set_ylabel("% of the records they hold")
    axes[1].legend()
    return fig


def plot_user_epsilon() -> Figure:
    ks = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50)
    exact = [user_epsilon(Q_RUN, SIGMA_DEFAULT, STEPS_RUN, k, DELTA)[0] for k in ks]
    generic = [generic_group_epsilon(Q_RUN, SIGMA_DEFAULT, STEPS_RUN, k, DELTA)[0] for k in ks]
    rate = [record_epsilon(min(1.0, k * Q_RUN), SIGMA_DEFAULT, STEPS_RUN, DELTA) for k in ks]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.loglog(ks, generic, marker="s", color="tab:red", ls="--", label="group privacy from the record-level guarantee")
    ax.loglog(ks, exact, marker="o", color="black", label="user accountant (binomial mixture)")
    ax.loglog(ks, rate, marker="^", color="tab:blue", ls=":", label="one record sampled at rate $kq$")
    ax.loglog(ks, [exact[0] * k for k in ks], color="tab:gray", ls="-.", label=r"linear: $k \times 6.71$")
    ax.set_xlabel("records of the user, $k$")
    ax.set_ylabel(r"user-level $\varepsilon$ at $\delta = 10^{-5}$")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_title(r"Example-level DP-SGD, $q = 0.01$, $\sigma = 1$, 10,000 steps")
    ax.legend(fontsize=8)
    return fig


def plot_strategies(rng: np.random.Generator) -> Figure:
    task = make_task()
    users = make_users(task.train)
    capped = cap_users(rng, users, CAP_RUN)
    k_max = int(users.counts[0])
    eps_list = (1.0, 2.0, 4.0, 8.0, 16.0)
    acc_a: list[float] = []
    acc_b: list[float] = []
    acc_c: list[float] = []
    for eps in eps_list:
        acc_a.append(run_example_level(rng, users, sigma_example_level(users, eps), task.test))
        acc_b.append(accuracy(train_user_dpsgd(rng, users, sigma_user_level(users, eps)), task.test))
        acc_c.append(run_example_level(rng, capped, sigma_example_level(capped, eps), task.test))
    p_np, _ = train_dpsgd(rng, users.data, DpsgdConfig(sigma=0.0, clip=math.inf, steps=STEPS_CMP))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.semilogx(eps_list, acc_b, marker="o", color="black", label="(b) user-level DP-SGD, per-user clipping")
    ax.semilogx(eps_list, acc_c, marker="s", color="tab:blue", label=f"(c) at most {CAP_RUN} records per user")
    ax.semilogx(eps_list, acc_a, marker="^", color="tab:red", label=f"(a) all records, accounted for k = {k_max}")
    ax.axhline(accuracy(p_np, task.test), color="tab:gray", ls="--", label="non-private")
    ax.set_xticks(eps_list, [f"{e:g}" for e in eps_list])
    ax.set_xlabel(r"user-level $\varepsilon$ at $\delta = 10^{-5}$")
    ax.set_ylabel("test accuracy")
    ax.set_title("1,000 steps, expected batch of 100 users or records")
    ax.legend(fontsize=8, loc="center right")
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
        "ulvl_population.svg": plot_population(),
        "ulvl_user_epsilon.svg": plot_user_epsilon(),
        "ulvl_strategies.svg": plot_strategies(fig_rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
