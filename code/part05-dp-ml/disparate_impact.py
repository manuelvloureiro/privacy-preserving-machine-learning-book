"""Check the numbers in the disparate impact chapter and draw its figures.

    uv run python code/part05-dp-ml/disparate_impact.py
    uv run python code/part05-dp-ml/disparate_impact.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 08-disparate-impact.md (they match exactly with the default seed), then
saves the chapter's figures.

The running example of the DP-SGD chapter with a minority group. The majority records are the first
n - n_min records of the fixed training set of dp_sgd.make_task(): x ~ N(0, I) in R^50, labels from
the random teacher network. The minority records come from a shifted, flatter region: x = 8u + z for
a fixed unit vector u, with z standard normal in a fixed 15-dimensional subspace orthogonal to u, and
their labels follow a different rule, the sign of one coordinate of z. So the minority has larger
inputs (norm about 8.9 against 7.1) and needs something the majority does not teach. The student
network, DP-SGD, the accountant and the hyperparameters are those of dp_sgd.py:
q = 0.01, C = 0.5, step size 0.5, 10,000 steps, δ = 1e-5.

Also: the clipping example in two dimensions, a quadratic model of the noise floor of a small group,
and Clopper–Pearson intervals for per-group accuracy.
"""

import os

# The models are tiny; multi-threaded BLAS only adds overhead.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "disparate_impact"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from dp_sgd import (
    CLIP_RUN,
    D_IN,
    DELTA_RUN,
    LR_RUN,
    N_TEST,
    N_TRAIN,
    Q_RUN,
    STEPS_RUN,
    Dataset,
    DpsgdConfig,
    Params,
    accuracy,
    bisect,
    clipped_grad_sum,
    dpsgd_epsilon,
    init_params,
    make_task,
    noise_multiplier_for,
)
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]

SHIFT = 8.0  # distance of the minority's inputs from the origin
SHARE_RUN = 0.05  # the minority's share of the training set
MINORITY_DIM = 15  # the minority's inputs vary in a 15-dimensional subspace
MINORITY_SEED = 43  # the minority's direction, rule and records are fixed objects, like make_task()
EPS_GRID: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
P_WEIGHTS = 3329


# --- The data: the running example with a minority group --------------------------------------------


@dataclass(frozen=True)
class GroupTask:
    train: Dataset
    group: NDArray[np.int64]  # 0 for the majority, 1 for the minority, per training record
    test_major: Dataset
    test_minor: Dataset

    @property
    def n_minor(self) -> int:
        return int(self.group.sum())


def minority_basis(d: int = D_IN, k: int = MINORITY_DIM) -> tuple[Array, Array]:
    """The fixed unit vector u (where the minority lives) and an orthonormal (d, k) basis of the subspace
    its inputs vary in, orthogonal to u. The minority's rule is the sign of the first basis coordinate."""
    rng = np.random.default_rng(MINORITY_SEED)
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    raw = rng.standard_normal((d, k))
    raw -= np.outer(u, u @ raw)
    basis, _ = np.linalg.qr(raw)
    return u, basis


def make_minority(rng: np.random.Generator, n: int, shift: float = SHIFT) -> Dataset:
    """x = shift·u + Σ_j c_j b_j with c ~ N(0, I_k), label sign(c_1)."""
    u, basis = minority_basis()
    c = rng.standard_normal((n, basis.shape[1]))
    return Dataset(shift * u + c @ basis.T, np.where(c[:, 0] >= 0, 1.0, -1.0))


def make_group_task(share: float = SHARE_RUN, n: int = N_TRAIN) -> GroupTask:
    """n training records, a fraction `share` of them from the minority; 20,000 test records per group."""
    task = make_task()
    rng = np.random.default_rng(MINORITY_SEED + 1)
    test_minor = make_minority(rng, N_TEST)
    pool = make_minority(rng, n)  # the same minority records, in the same order, for every share
    n_minor = round(share * n)
    x = np.vstack([task.train.x[: n - n_minor], pool.x[:n_minor]])
    y = np.concatenate([task.train.y[: n - n_minor], pool.y[:n_minor]])
    group = np.concatenate([np.zeros(n - n_minor, dtype=np.int64), np.ones(n_minor, dtype=np.int64)])
    return GroupTask(Dataset(x, y), group, task.test, test_minor)


def group_sub(data: Dataset, mask: NDArray[np.bool_]) -> Dataset:
    return Dataset(data.x[mask], data.y[mask])


# --- Training with per-group sampling rates and per-group logs ----------------------------------------


@dataclass
class GroupLog:
    steps: list[int] = field(default_factory=list[int])
    median_norm: list[tuple[float, float]] = field(default_factory=list[tuple[float, float]])
    clipped_frac: list[tuple[float, float]] = field(default_factory=list[tuple[float, float]])
    weight: list[tuple[float, float]] = field(default_factory=list[tuple[float, float]])  # mean min(1, C/‖g‖)
    test_acc: list[tuple[float, float]] = field(default_factory=list[tuple[float, float]])


def clip_stats(p: Params, data: Dataset, clip: float) -> tuple[float, float, float]:
    """(median per-example gradient norm, fraction with norm above C, mean clipping weight min(1, C/‖g‖))."""
    norms = clipped_grad_sum(p, data.x, data.y, clip).norms
    return (
        float(np.median(norms)),
        float(np.mean(norms > clip)),
        float(np.mean(np.minimum(1.0, clip / np.maximum(norms, 1e-300)))),
    )


def train_groups(
    rng: np.random.Generator,
    gt: GroupTask,
    cfg: DpsgdConfig,
    oversample: float = 1.0,
    log_every: int = 0,
) -> tuple[Params, GroupLog]:
    """DP-SGD as in dp_sgd.train_dpsgd, except that minority records are sampled at rate oversample·q.

    The update still divides by the public B = qn. With oversample = 1 it is train_dpsgd, draw for draw.
    """
    data = gt.train
    n = data.y.size
    p = init_params(rng, data.x.shape[1])
    rates = np.where(gt.group == 1, cfg.q * oversample, cfg.q)
    b_expected = cfg.q * n
    noise_sd = cfg.sigma * cfg.clip if cfg.sigma > 0 else 0.0
    probe_major = Dataset(data.x[:1000], data.y[:1000])
    minor = group_sub(data, gt.group == 1)
    probe_minor = Dataset(minor.x[:500], minor.y[:500])
    log = GroupLog()
    for t in range(cfg.steps + 1):
        if log_every and t % log_every == 0:
            sa = clip_stats(p, probe_major, cfg.clip if math.isfinite(cfg.clip) else CLIP_RUN)
            sb = clip_stats(p, probe_minor, cfg.clip if math.isfinite(cfg.clip) else CLIP_RUN)
            log.steps.append(t)
            log.median_norm.append((sa[0], sb[0]))
            log.clipped_frac.append((sa[1], sb[1]))
            log.weight.append((sa[2], sb[2]))
            log.test_acc.append((accuracy(p, gt.test_major), accuracy(p, gt.test_minor)))
        if t == cfg.steps:
            break
        idx = np.flatnonzero(rng.random(n) < rates)
        cs = clipped_grad_sum(p, data.x[idx], data.y[idx], cfg.clip)
        g1, g2 = cs.g1, cs.g2
        if noise_sd > 0:
            g1 = g1 + noise_sd * rng.standard_normal(g1.shape)
            g2 = g2 + noise_sd * rng.standard_normal(g2.shape)
        p = Params(p.w1 - cfg.lr * g1 / b_expected, p.w2 - cfg.lr * g2 / b_expected)
    return p, log


def group_accuracies(rng: np.random.Generator, gt: GroupTask, cfg: DpsgdConfig, oversample: float = 1.0) -> Array:
    """(majority, minority) test accuracy of one training run."""
    p, _ = train_groups(rng, gt, cfg, oversample)
    return np.array([accuracy(p, gt.test_major), accuracy(p, gt.test_minor)])


def repeat(rng: np.random.Generator, gt: GroupTask, cfg: DpsgdConfig, runs: int, oversample: float = 1.0) -> Array:
    """(runs, 2) array of per-group test accuracies."""
    return np.array([group_accuracies(rng, gt, cfg, oversample) for _ in range(runs)])


NONPRIVATE = DpsgdConfig(sigma=0.0, clip=math.inf)
CLIP_ONLY = DpsgdConfig(sigma=0.0)


def private_config(eps: float, clip: float = CLIP_RUN, lr: float = LR_RUN) -> DpsgdConfig:
    return DpsgdConfig(sigma=noise_multiplier_for(Q_RUN, STEPS_RUN, eps, DELTA_RUN), clip=clip, lr=lr)


# --- Section: clipping reweights groups ----------------------------------------------------------


def clipped_mean(grads: Array, counts: Array, clip: float) -> Array:
    """(1/n) Σ_i min(1, C/‖g_i‖) g_i for groups of identical gradients: grads (k, d), counts (k,)."""
    norms = np.linalg.norm(grads, axis=1)
    scale = np.minimum(1.0, clip / norms) if math.isfinite(clip) else np.ones_like(norms)
    return (counts[:, None] * scale[:, None] * grads).sum(axis=0) / counts.sum()


def angle_deg(a: Array, b: Array) -> float:
    cos = float(a @ b) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


TOY_GRADS = np.array([[0.2, 0.0], [0.0, 4.0]])  # a well-fitted majority record, a badly fitted minority record
TOY_COUNTS = np.array([95.0, 5.0])


def effective_share(share: float, w_major: float, w_minor: float) -> float:
    """The minority's share of the expected update when each group is scaled by its mean clipping weight."""
    return share * w_minor / (share * w_minor + (1 - share) * w_major)


# --- Section: the noise floor of a small group -------------------------------------------------------


def noise_floor(share: float, lr: float, tau: float, dim: int, curvature: float) -> float:
    """Stationary excess loss of a group whose loss (λ/2)‖w_g - w*‖² lives on its own dim-dimensional
    weights, under w ← w - lr(share·λ(w - w*) + τξ): dim·lr·τ²/(2·share·(2 - lr·share·λ))."""
    a = lr * share * curvature
    return dim * lr * tau**2 / (2 * share * (2 - a))


def simulate_noise_floor(
    rng: np.random.Generator, share: float, lr: float, tau: float, dim: int, curvature: float, steps: int
) -> float:
    """Average excess loss over the second half of a simulated run started at the optimum."""
    w = np.zeros(dim)
    total = 0.0
    for t in range(steps):
        w = w - lr * (share * curvature * w + tau * rng.standard_normal(dim))
        if t >= steps // 2:
            total += 0.5 * curvature * float(w @ w)
    return total / (steps - steps // 2)


def break_even_steps(noise_len: float, signal_len: float) -> float:
    """Steps after which a constant drift of length signal_len beats a random walk of step noise_len."""
    return (noise_len / signal_len) ** 2


# --- Section: measuring with small groups ------------------------------------------------------------


def binom_cdf(k: int, n: int, p: float) -> float:
    """Pr[Bin(n, p) ≤ k]."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    j = np.arange(k + 1)
    log_pmf = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in j]) + (
        j * math.log(p) + (n - j) * math.log1p(-p)
    )
    top = float(log_pmf.max())
    return min(1.0, math.exp(top) * float(np.exp(log_pmf - top).sum()))


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """The exact two-sided interval for a binomial proportion from k successes in n trials."""
    a = 1 - level
    lo = 0.0 if k == 0 else bisect(lambda p: 1 - binom_cdf(k - 1, n, p) >= a / 2, 0.0, 1.0, 60)
    hi = 1.0 if k == n else bisect(lambda p: binom_cdf(k, n, p) <= a / 2, 0.0, 1.0, 60)
    return lo, hi


def correct_on_subset(rng: np.random.Generator, p: Params, data: Dataset, m: int) -> int:
    """Number of correct predictions on m test records drawn without replacement."""
    idx = rng.choice(data.y.size, size=m, replace=False)
    return round(accuracy(p, Dataset(data.x[idx], data.y[idx])) * m)


# --- Section: mitigations ------------------------------------------------------------------------


def oversampled_sigma(eps: float, oversample: float) -> float:
    """Noise multiplier at which the oversampled records (rate oversample·q) meet (ε, δ)."""
    return noise_multiplier_for(Q_RUN * oversample, STEPS_RUN, eps, DELTA_RUN)


# --- The chapter's numbers ---------------------------------------------------------------------


def fmt(accs: Array) -> str:
    m = accs.mean(axis=0)
    s = accs.std(axis=0)
    return f"majority {m[0]:.4f} +- {s[0]:.4f}  minority {m[1]:.4f} +- {s[1]:.4f}  gap {m[0] - m[1]:.4f}"


def print_table(rng: np.random.Generator) -> None:
    gt = make_group_task()
    u, basis = minority_basis()
    print("== The task")
    x = gt.train.x
    print(
        f"n={gt.train.y.size}  minority records={gt.n_minor}  share={SHARE_RUN:g}  "
        f"mean input norm: majority {np.linalg.norm(x[gt.group == 0], axis=1).mean():.2f}  "
        f"minority {np.linalg.norm(x[gt.group == 1], axis=1).mean():.2f}"
    )
    print(f"positive labels: majority {np.mean(gt.test_major.y > 0):.3f}  minority {np.mean(gt.test_minor.y > 0):.3f}")
    print(
        f"minority: centre {SHIFT:g} u, varies in {basis.shape[1]} dimensions, "
        f"largest |u.b_j| = {float(np.max(np.abs(u @ basis))):.1e}"
    )

    print("== Per-group accuracy against epsilon (share 0.05, 3 runs each)")
    base = repeat(rng, gt, NONPRIVATE, 3)
    print(f"non-private: {fmt(base)}")
    clip_only = repeat(rng, gt, CLIP_ONLY, 3)
    print(f"clipped, no noise: {fmt(clip_only)}")
    priv: dict[float, Array] = {}
    for eps in EPS_GRID:
        cfg = private_config(eps)
        priv[eps] = repeat(rng, gt, cfg, 3)
        print(f"eps={eps:g} (sigma={cfg.sigma:.3g}): {fmt(priv[eps])}")
    b = base.mean(axis=0)
    for name, accs in [("clipped, no noise", clip_only)] + [(f"eps={e:g}", priv[e]) for e in EPS_GRID]:
        cost = b - accs.mean(axis=0)
        print(
            f"privacy cost {name}: majority {cost[0]:.4f}  minority {cost[1]:.4f}  ratio {cost[1] / cost[0]:.1f}"
            if cost[0] > 0
            else f"privacy cost {name}: majority {cost[0]:.4f}  minority {cost[1]:.4f}"
        )

    print("== Clipping in two dimensions (95 records with gradient (0.2, 0), 5 with (0, 4), C=0.5)")
    full = clipped_mean(TOY_GRADS, TOY_COUNTS, math.inf)
    clipped = clipped_mean(TOY_GRADS, TOY_COUNTS, CLIP_RUN)
    scaled = clipped_mean(TOY_GRADS / 8.0, TOY_COUNTS, CLIP_RUN)
    print(f"unclipped mean ({full[0]:.4f}, {full[1]:.4f}) length {np.linalg.norm(full):.4f}")
    print(f"clipped mean ({clipped[0]:.4f}, {clipped[1]:.4f}) length {np.linalg.norm(clipped):.4f}")
    print(f"angle between them {angle_deg(full, clipped):.1f} degrees")
    print(
        f"minority share of the update: unclipped {full[1] / full.sum():.3f}  clipped {clipped[1] / clipped.sum():.3f}"
    )
    print(f"scaled by C/Z with Z=4: ({scaled[0]:.4f}, {scaled[1]:.4f})  angle {angle_deg(full, scaled):.1f} degrees")

    print("== Clipping during training (share 0.05, C=0.5): majority / minority")
    for name, cfg in (("clipped, no noise", CLIP_ONLY), ("eps=4", private_config(4.0))):
        _, log = train_groups(rng, gt, cfg, log_every=2500)
        for t, med, frac, w, acc in zip(
            log.steps, log.median_norm, log.clipped_frac, log.weight, log.test_acc, strict=True
        ):
            print(
                f"{name} step {t}: median norm {med[0]:.1e} / {med[1]:.1e}  clipped {frac[0]:.2f} / {frac[1]:.2f}  "
                f"mean weight {w[0]:.3f} / {w[1]:.3f}  effective share {effective_share(SHARE_RUN, w[0], w[1]):.4f}  "
                f"test acc {acc[0]:.4f} / {acc[1]:.4f}"
            )

    print("== Clipping norm at eps=4 with lr*C = 0.25 (2 runs each; C=0.5 is the row eps=4 above)")
    for clip in (0.05, 5.0):
        lr = LR_RUN * CLIP_RUN / clip
        print(f"C={clip:g} eps=4: {fmt(repeat(rng, gt, private_config(4.0, clip, lr), 2))}")

    print("== The noise floor of a small group (quadratic model)")
    tau = private_config(4.0).sigma * CLIP_RUN / (Q_RUN * N_TRAIN)
    print(f"tau = sigma*C/B at eps=4: {tau:.4f}")
    for share in (0.95, 0.2, 0.05, 0.01):
        pred = noise_floor(share, LR_RUN, tau, 51, 1.0)
        sim = simulate_noise_floor(rng, share, LR_RUN, tau, 51, 1.0, 40_000)
        print(
            f"share {share:g}: predicted excess loss {pred:.2e}  simulated {sim:.2e}  relaxation steps {1 / (LR_RUN * share):.0f}"
        )
    noise_len = tau * math.sqrt(P_WEIGHTS)
    for share in (1.0, 0.2, 0.05, 0.01):
        print(f"break-even, signal share {share:g} of 0.026: {break_even_steps(noise_len, share * 0.026):.0f} steps")

    print("== Group size (non-private and eps=4, 2 runs each; share 0.05 is the table above)")
    for share in (0.01, 0.2):
        g = make_group_task(share)
        a_np = repeat(rng, g, NONPRIVATE, 2)
        a_dp = repeat(rng, g, private_config(4.0), 2)
        cost = a_np.mean(axis=0) - a_dp.mean(axis=0)
        print(f"share {share:g} ({g.n_minor} records) non-private: {fmt(a_np)}")
        print(f"share {share:g} eps=4: {fmt(a_dp)}  privacy cost majority {cost[0]:.4f}  minority {cost[1]:.4f}")

    print("== Measuring on small test sets")
    p, _ = train_groups(rng, gt, private_config(4.0))
    full_acc = accuracy(p, gt.test_minor)
    print(f"model at eps=4: minority accuracy on 20000 test records {full_acc:.4f}")
    for m in (100, 500, 2000):
        k = correct_on_subset(rng, p, gt.test_minor, m)
        lo, hi = clopper_pearson(k, m)
        print(f"{m} minority test records: {k} correct  estimate {k / m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    k = correct_on_subset(rng, p, gt.test_major, 1900)
    lo, hi = clopper_pearson(k, 1900)
    print(f"1900 majority test records: {k} correct  estimate {k / 1900:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    print("== Mitigation: oversampling the minority four times (eps=4 for every record, 3 runs)")
    sigma = oversampled_sigma(4.0, 4.0)
    eps_major = dpsgd_epsilon(Q_RUN, sigma, STEPS_RUN, DELTA_RUN)
    accs = repeat(rng, gt, DpsgdConfig(sigma=sigma), 3, oversample=4.0)
    print(f"oversample x4: sigma={sigma:.3g}  eps of majority records={eps_major:.2f}  {fmt(accs)}")
    sigma1 = private_config(4.0).sigma
    eps_minor = dpsgd_epsilon(4 * Q_RUN, sigma1, STEPS_RUN, DELTA_RUN)
    print(f"oversample x4 at sigma={sigma1:.3g}: eps of minority records={eps_minor:.2f}")


# --- Figures --------------------------------------------------------------------------------


def plot_privacy_accuracy(rng: np.random.Generator) -> Figure:
    gt = make_group_task()
    eps_list = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    accs = np.array([group_accuracies(rng, gt, private_config(e)) for e in eps_list])
    base = group_accuracies(rng, gt, NONPRIVATE)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for j, (color, label) in enumerate((("black", "majority (95%)"), ("tab:red", "minority (5%)"))):
        ax.semilogx(eps_list, accs[:, j], marker="o", color=color, label=f"{label}, DP-SGD")
        ax.axhline(base[j], color=color, ls="--", lw=1, label=f"{label}, non-private")
    ax.set_xticks(eps_list, [f"{e:g}" for e in eps_list])
    ax.set_xlabel(r"$\varepsilon$ at $\delta = 10^{-5}$")
    ax.set_ylabel("test accuracy of the group")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Who pays for privacy: per-group accuracy")
    return fig


def plot_clipping_toy() -> Figure:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    full = clipped_mean(TOY_GRADS, TOY_COUNTS, math.inf)
    clipped = clipped_mean(TOY_GRADS, TOY_COUNTS, CLIP_RUN)
    scaled = clipped_mean(TOY_GRADS / 8.0, TOY_COUNTS, CLIP_RUN)
    arrows = (
        (
            (0.19, 0.0),
            "tab:gray",
            1.2,
            "-|>",
            "each group's pull: majority 0.95 × 0.2,\nminority 0.05 × 4, or 0.05 × 0.5 clipped",
        ),
        ((0.0, 0.2), "tab:gray", 1.2, "-|>", None),
        ((0.0, 0.025), "tab:gray", 1.2, "-|>", None),
        (tuple(full), "black", 3.0, "->", "average gradient, no clipping"),
        (tuple(clipped), "tab:red", 2.0, "->", "average clipped gradient, C = 0.5"),
        (tuple(scaled), "tab:blue", 2.0, "->", "every gradient scaled by 1/8"),
    )
    for (x, y), color, lw, style, label in arrows:
        ax.annotate(
            "", xy=(float(x), float(y)), xytext=(0, 0), arrowprops={"arrowstyle": style, "color": color, "lw": lw}
        )
        if label is not None:
            ax.plot([], [], color=color, lw=lw, label=label)
    ax.set_xlim(-0.02, 0.3)
    ax.set_ylim(-0.02, 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("direction that helps the majority")
    ax.set_ylabel("direction that helps the minority")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.03, 1.0))
    return fig


def plot_clipping_training(rng: np.random.Generator) -> Figure:
    gt = make_group_task()
    _, log_np = train_groups(rng, gt, NONPRIVATE, log_every=250)
    _, log_dp = train_groups(rng, gt, private_config(4.0), log_every=250)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for j, (color, label) in enumerate((("black", "majority"), ("tab:red", "minority"))):
        axes[0].semilogy(log_np.steps, [m[j] for m in log_np.median_norm], color=color, label=f"{label}, non-private")
        axes[0].semilogy(
            log_dp.steps, [m[j] for m in log_dp.median_norm], color=color, ls="--", label=rf"{label}, $\varepsilon = 4$"
        )
        axes[1].plot(log_dp.steps, [w[j] for w in log_dp.weight], color=color, label=label)
    axes[0].axhline(CLIP_RUN, color="tab:gray", ls=":", label="clipping norm $C = 0.5$")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("median per-example gradient norm")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel(r"step ($\varepsilon = 4$)")
    axes[1].set_ylabel(r"mean clipping weight $\min(1, C/\|g_i\|)$")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    return fig


def plot_group_size(rng: np.random.Generator) -> Figure:
    shares = (0.01, 0.03, 0.1, 0.3)
    rows: list[tuple[Array, Array]] = []
    for s in shares:
        g = make_group_task(s)
        rows.append((group_accuracies(rng, g, NONPRIVATE), group_accuracies(rng, g, private_config(4.0))))
    tau = private_config(4.0).sigma * CLIP_RUN / (Q_RUN * N_TRAIN)
    grid = np.geomspace(0.005, 1.0, 60)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].semilogx(
        shares, [r[0][1] for r in rows], marker="o", color="tab:red", ls="--", label="minority, non-private"
    )
    axes[0].semilogx(shares, [r[1][1] for r in rows], marker="o", color="tab:red", label=r"minority, $\varepsilon = 4$")
    axes[0].semilogx(shares, [r[1][0] for r in rows], marker="s", color="black", label=r"majority, $\varepsilon = 4$")
    axes[0].set_xticks(shares, [f"{s:g}" for s in shares])
    axes[0].set_xlabel("minority's share of the training set")
    axes[0].set_ylabel("test accuracy of the group")
    axes[0].legend(fontsize=8)
    axes[1].loglog(grid, [noise_floor(float(s), LR_RUN, tau, 51, 1.0) for s in grid], color="black", label="formula")
    sims = [simulate_noise_floor(rng, s, LR_RUN, tau, 51, 1.0, 20_000) for s in (0.01, 0.05, 0.2, 0.95)]
    axes[1].loglog((0.01, 0.05, 0.2, 0.95), sims, "o", color="tab:red", label="simulation")
    axes[1].set_xlabel("group's share $s_g$")
    axes[1].set_ylabel("stationary excess loss of the group")
    axes[1].set_title(r"Quadratic model, $\varepsilon = 4$ noise")
    axes[1].legend(fontsize=8)
    return fig


def plot_intervals() -> Figure:
    sizes = np.array([25, 50, 100, 200, 500, 1000, 2000, 5000])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for acc, color in ((0.95, "black"), (0.7, "tab:red")):
        widths = []
        for m in sizes:
            lo, hi = clopper_pearson(round(acc * int(m)), int(m))
            widths.append(hi - lo)
        ax.loglog(sizes, widths, marker="o", color=color, label=f"observed accuracy {acc:g}")
    ticks = [0.02, 0.05, 0.1, 0.2, 0.4]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_yticks([], minor=True)
    ax.set_xlabel("test records in the group")
    ax.set_ylabel("width of the 95% Clopper–Pearson interval")
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
        "fair_privacy_accuracy.svg": plot_privacy_accuracy(fig_rng),
        "fair_clipping_toy.svg": plot_clipping_toy(),
        "fair_clipping_training.svg": plot_clipping_training(fig_rng),
        "fair_group_size.svg": plot_group_size(fig_rng),
        "fair_intervals.svg": plot_intervals(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
