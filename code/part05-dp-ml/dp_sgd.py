"""Check the numbers in the DP-SGD chapter and draw its figures.

    uv run python code/part05-dp-ml/dp_sgd.py
    uv run python code/part05-dp-ml/dp_sgd.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 02-dp-sgd.md (they match exactly with the default seed), then saves the
chapter's figures.

The running example of the part's neural-network chapters: inputs x ~ N(0, I) in R^50, labels
y = sign(teacher(x)) from a fixed random teacher network (50 → 8 tanh units → 1), no label noise. The
student is a two-layer network (50 → 64 tanh units → 1, with biases: 3,329 weights) trained with the
logistic loss on n = 10,000 records. DP-SGD runs with Poisson sampling rate q = 0.01 (expected batch
B = 100), clipping norm C = 0.5, noise multiplier σ = 1 and 10,000 steps, the running example of the
privacy accounting part, which gives ε = 6.71 at δ = 1e-5.

Per-example gradient norms are computed without forming per-example gradients ("ghost clipping"): for
a dense layer, the gradient of record i is the outer product δ_i a_iᵀ, whose norm is ‖δ_i‖ ‖a_i‖.

Public functions used by later chapters: Dataset, Params, make_teacher, make_dataset, init_params,
forward, accuracy, logistic_losses, clipped_grad_sum, microbatch_noise, DpsgdConfig, train_dpsgd, ORDERS,
subsampled_gaussian_rdp, rdp_curve, epsilon_from_rdp, dpsgd_epsilon, noise_multiplier_for.
"""

import os

# The models are tiny; multi-threaded BLAS only adds overhead.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "dp_sgd"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

# The running example.
D_IN, HIDDEN, TEACHER_HIDDEN = 50, 64, 8
N_TRAIN, N_TEST = 10_000, 20_000
Q_RUN, SIGMA_RUN, CLIP_RUN, LR_RUN, STEPS_RUN = 0.01, 1.0, 0.5, 0.5, 10_000
DELTA_RUN = 1e-5
EPS_GRID: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)


# --- Accounting: RDP of the Poisson-subsampled Gaussian ----------------------------------------------
# The same computation as the subsampling chapter of the accounting part, repeated here so that this
# part's scripts stand on their own.

ORDERS: list[float] = [1 + x / 10 for x in range(1, 100)] + [float(a) for a in range(12, 64)] + [128.0, 256.0, 512.0]


def logsumexp(values: Array) -> float:
    top = float(np.max(values))
    return top + math.log(float(np.sum(np.exp(values - top))))


def bisect(f: Callable[[float], bool], lo: float, hi: float, steps: int = 100) -> float:
    """Smallest x in [lo, hi] with f(x) true, for f false-then-true (returns the upper end of the bracket)."""
    for _ in range(steps):
        mid = (lo + hi) / 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _log_a_binomial(alpha: int, q: float, sigma: float) -> float:
    """ln A_α = ln Σ_j C(α, j)(1-q)^{α-j} q^j exp((j² - j)/(2σ²)), integer α."""
    terms = np.array(
        [
            math.lgamma(alpha + 1)
            - math.lgamma(j + 1)
            - math.lgamma(alpha - j + 1)
            + (alpha - j) * math.log1p(-q)
            + j * math.log(q)
            + (j * j - j) / (2 * sigma**2)
            for j in range(alpha + 1)
        ]
    )
    return logsumexp(terms)


def _log_mixture_ratio(z: Array, q: float, sigma: float) -> Array:
    return np.logaddexp(math.log1p(-q), math.log(q) + (2 * z - 1) / (2 * sigma**2))


def _log_normal_pdf(z: Array, sigma: float) -> Array:
    return -(z**2) / (2 * sigma**2) - math.log(sigma * math.sqrt(2 * math.pi))


def _grid(lo: float, hi: float, sigma: float) -> Array:
    return np.linspace(lo, hi, int((hi - lo) / (sigma / 100)) + 1)


def _log_a_numeric(alpha: float, q: float, sigma: float) -> float:
    z = _grid(-12 * sigma, alpha + 12 * sigma, sigma)
    return logsumexp(_log_normal_pdf(z, sigma) + alpha * _log_mixture_ratio(z, q, sigma)) + math.log(z[1] - z[0])


def _log_b_numeric(alpha: float, q: float, sigma: float) -> float:
    z = _grid(-12 * sigma - 2, 12 * sigma + 2, sigma)
    return logsumexp(_log_normal_pdf(z, sigma) - (alpha - 1) * _log_mixture_ratio(z, q, sigma)) + math.log(z[1] - z[0])


def subsampled_gaussian_rdp(alpha: float, q: float, sigma: float) -> float:
    """RDP ε(α) of the Poisson-subsampled Gaussian with sensitivity 1 (add/remove): the larger direction."""
    if q >= 1:
        return alpha / (2 * sigma**2)
    log_a = _log_a_binomial(int(alpha), q, sigma) if float(alpha).is_integer() else _log_a_numeric(alpha, q, sigma)
    return max(log_a, _log_b_numeric(alpha, q, sigma)) / (alpha - 1)


@cache
def _rdp_curve(q: float, sigma: float) -> tuple[float, ...]:
    return tuple(subsampled_gaussian_rdp(a, q, sigma) for a in ORDERS)


def rdp_curve(q: float, sigma: float) -> list[float]:
    return list(_rdp_curve(q, sigma))


def epsilon_from_rdp(rdp: Sequence[float], delta: float) -> tuple[float, float]:
    """(ε, best order) by the improved conversion ε(α) + ln((α-1)/α) - (ln δ + ln α)/(α - 1) (Balle et al. 2020)."""
    return min(
        (max(0.0, e + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)), a)
        for a, e in zip(ORDERS, rdp, strict=True)
    )


def dpsgd_epsilon(q: float, sigma: float, steps: int, delta: float) -> float:
    """ε at δ of `steps` DP-SGD steps with Poisson rate q and noise multiplier σ, by the RDP accountant."""
    return epsilon_from_rdp([steps * e for e in rdp_curve(q, sigma)], delta)[0]


def noise_multiplier_for(q: float, steps: int, eps: float, delta: float) -> float:
    """The smallest σ (to three significant digits) for which DP-SGD meets (ε, δ)."""
    sigma = bisect(lambda s: dpsgd_epsilon(q, s, steps, delta) <= eps, 0.3, 200.0, 40)
    digits = 2 - math.floor(math.log10(sigma))
    return math.ceil(sigma * 10**digits) / 10**digits


# --- The model: a two-layer tanh network with a logistic loss ---------------------------------------


@dataclass(frozen=True)
class Dataset:
    x: Array  # (n, d)
    y: Array  # (n,), labels ±1


@dataclass(frozen=True)
class Params:
    w1: Array  # (hidden, d + 1), last column is the bias
    w2: Array  # (hidden + 1,), last entry is the bias

    @property
    def size(self) -> int:
        return self.w1.size + self.w2.size


def make_teacher(rng: np.random.Generator, d: int = D_IN, hidden: int = TEACHER_HIDDEN) -> Params:
    return Params(rng.standard_normal((hidden, d + 1)), rng.standard_normal(hidden + 1))


def init_params(rng: np.random.Generator, d: int = D_IN, hidden: int = HIDDEN) -> Params:
    return Params(
        rng.standard_normal((hidden, d + 1)) / math.sqrt(d + 1), rng.standard_normal(hidden + 1) / math.sqrt(hidden + 1)
    )


@dataclass(frozen=True)
class Forward:
    a0: Array  # inputs with a 1 appended, (b, d + 1)
    a1: Array  # hidden activations, (b, hidden)
    a1b: Array  # hidden activations with a 1 appended, (b, hidden + 1)
    f: Array  # outputs (logits), (b,)


def forward(p: Params, x: Array) -> Forward:
    ones = np.ones((x.shape[0], 1))
    a0 = np.hstack([x, ones])
    a1 = np.tanh(a0 @ p.w1.T)
    a1b = np.hstack([a1, ones])
    return Forward(a0, a1, a1b, a1b @ p.w2)


def make_dataset(rng: np.random.Generator, n: int, teacher: Params) -> Dataset:
    x = rng.standard_normal((n, teacher.w1.shape[1] - 1))
    return Dataset(x, np.where(forward(teacher, x).f >= 0, 1.0, -1.0))


def sigmoid(s: Array) -> Array:
    return 0.5 * (1.0 + np.tanh(0.5 * s))


def logistic_losses(p: Params, data: Dataset) -> Array:
    return np.logaddexp(0.0, -data.y * forward(p, data.x).f)


def accuracy(p: Params, data: Dataset) -> float:
    correct: NDArray[np.bool_] = np.where(forward(p, data.x).f >= 0, 1.0, -1.0) == data.y
    return float(correct.mean())


# --- Per-example gradients: explicit and "ghost" --------------------------------------------------


@dataclass(frozen=True)
class Backward:
    fwd: Forward
    d1: Array  # gradient of the loss with respect to the hidden pre-activations, (b, hidden)
    d2: Array  # gradient with respect to the output, (b,)


def backward(p: Params, x: Array, y: Array) -> Backward:
    fwd = forward(p, x)
    d2 = -y * sigmoid(-y * fwd.f)  # d/df of ln(1 + e^{-yf})
    d1 = d2[:, None] * p.w2[None, :-1] * (1 - fwd.a1**2)
    return Backward(fwd, d1, d2)


def ghost_norms(bw: Backward) -> Array:
    """‖g_i‖ from ‖δ_i‖ ‖a_i‖ per layer: ‖d1_i‖² ‖a0_i‖² + d2_i² ‖a1b_i‖², never forming g_i."""
    sq = (bw.d1**2).sum(axis=1) * (bw.fwd.a0**2).sum(axis=1) + bw.d2**2 * (bw.fwd.a1b**2).sum(axis=1)
    return np.sqrt(sq)


def explicit_grads(bw: Backward) -> Array:
    """Per-example gradients formed explicitly, flattened: (b, hidden·(d+1) + hidden + 1)."""
    g1 = bw.d1[:, :, None] * bw.fwd.a0[:, None, :]
    g2 = bw.d2[:, None] * bw.fwd.a1b
    return np.hstack([g1.reshape(g1.shape[0], -1), g2])


@dataclass(frozen=True)
class ClippedSum:
    g1: Array
    g2: Array
    norms: Array  # per-example gradient norms before clipping
    clipped: int  # how many were clipped


def clipped_grad_sum(p: Params, x: Array, y: Array, clip: float) -> ClippedSum:
    """Σ_i g_i min(1, C/‖g_i‖) by reweighting the back-propagated errors: (d1 ∘ c)ᵀ a0 and a1bᵀ (d2 ∘ c)."""
    bw = backward(p, x, y)
    norms = ghost_norms(bw)
    scale = np.minimum(1.0, clip / np.maximum(norms, 1e-300)) if math.isfinite(clip) else np.ones_like(norms)
    return ClippedSum(
        (bw.d1 * scale[:, None]).T @ bw.fwd.a0, bw.fwd.a1b.T @ (bw.d2 * scale), norms, int(np.sum(scale < 1))
    )


def microbatch_clipped_sum(p: Params, x: Array, y: Array, group: IntArray, clip: float) -> tuple[Array, Array]:
    """Σ_j clip_C(mean of the gradients in microbatch j), with microbatch j holding the sampled records
    whose group id is j. Group ids are fixed in advance (here the record index mod B/m, standing in for a hash of
    a record identifier, since no record is ever added or removed in the experiments), so a batch holds about
    B/m microbatches of about m records, and one record changes one microbatch's clipped mean, by at most 2C."""
    bw = backward(p, x, y)
    grads = explicit_grads(bw)
    ids, inverse, counts = np.unique(group, return_inverse=True, return_counts=True)
    member = (inverse[None, :] == np.arange(ids.size)[:, None]).astype(np.float64)  # (groups, records)
    means = (member @ grads) / counts[:, None]
    norms: Array = np.sqrt((means**2).sum(axis=1))
    total: Array = (means * np.minimum(1.0, clip / np.maximum(norms, 1e-300))[:, None]).sum(axis=0)
    h, d1 = p.w1.shape
    return total[: h * d1].reshape(h, d1), total[h * d1 :]


def microbatch_noise(sigma: float, clip: float, batch: float, size: int) -> float:
    """Noise standard deviation per coordinate on the averaged update with microbatches of `size` records:
    the sum of clipped microbatch means has sensitivity 2C (C when size is 1), and is divided by the
    expected number of microbatches, batch/size."""
    sensitivity = clip if size == 1 else 2 * clip
    return sigma * sensitivity / (batch / size)


# --- DP-SGD --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DpsgdConfig:
    q: float = Q_RUN  # Poisson sampling rate
    sigma: float = SIGMA_RUN  # noise multiplier (0 for non-private)
    clip: float = CLIP_RUN  # clipping norm C (inf for no clipping)
    lr: float = LR_RUN  # step size on the averaged gradient
    steps: int = STEPS_RUN
    microbatch: int = 1  # records per microbatch; 1 is per-example clipping


@dataclass
class TrainLog:
    steps: list[int] = field(default_factory=list[int])
    test_acc: list[float] = field(default_factory=list[float])
    clipped_frac: list[float] = field(default_factory=list[float])
    median_norm: list[float] = field(default_factory=list[float])
    signal_norm: list[float] = field(default_factory=list[float])  # ‖clipped sum‖ / B


def train_dpsgd(
    rng: np.random.Generator,
    data: Dataset,
    cfg: DpsgdConfig,
    init: Params | None = None,
    test: Dataset | None = None,
    log_every: int = 0,
) -> tuple[Params, TrainLog]:
    """DP-SGD: Poisson batches, per-example clipping, Gaussian noise σC on the sum, divide by B = qn."""
    n = data.y.size
    p = init if init is not None else init_params(rng, data.x.shape[1])
    b_expected = cfg.q * n
    noise_sd = cfg.sigma * cfg.clip if cfg.sigma > 0 else 0.0
    log = TrainLog()
    if cfg.microbatch > 1:
        # Microbatches: groups fixed in advance (the index stands in for a hash of a record identifier), so adding
        # a record changes one group only.
        # The clipped group means have sensitivity 2C, so the noise is doubled to keep the same ε, and the
        # sum is divided by the expected number of groups.
        noise_sd *= 2
        b_expected /= cfg.microbatch
    groups: IntArray = np.arange(n, dtype=np.int64) % max(1, round(cfg.q * n / cfg.microbatch))
    for t in range(cfg.steps):
        idx = np.flatnonzero(rng.random(n) < cfg.q)
        if cfg.microbatch > 1:
            g1, g2 = microbatch_clipped_sum(p, data.x[idx], data.y[idx], groups[idx], cfg.clip)
            p = Params(
                p.w1 - cfg.lr * (g1 + noise_sd * rng.standard_normal(g1.shape)) / b_expected,
                p.w2 - cfg.lr * (g2 + noise_sd * rng.standard_normal(g2.shape)) / b_expected,
            )
            continue
        cs = clipped_grad_sum(p, data.x[idx], data.y[idx], cfg.clip)
        g1, g2 = cs.g1, cs.g2
        if log_every and t % log_every == 0:
            log.steps.append(t)
            log.test_acc.append(accuracy(p, test) if test is not None else float("nan"))
            log.clipped_frac.append(cs.clipped / max(idx.size, 1))
            log.median_norm.append(float(np.median(cs.norms)) if idx.size else float("nan"))
            log.signal_norm.append(math.sqrt(float((g1**2).sum() + (g2**2).sum())) / b_expected)
        if noise_sd > 0:
            g1 = g1 + noise_sd * rng.standard_normal(g1.shape)
            g2 = g2 + noise_sd * rng.standard_normal(g2.shape)
        p = Params(p.w1 - cfg.lr * g1 / b_expected, p.w2 - cfg.lr * g2 / b_expected)
    return p, log


@dataclass(frozen=True)
class Task:
    train: Dataset
    test: Dataset


TASK_SEED = 0  # the running example's data are a fixed object: the same in every chapter, table and figure


def make_task(n: int = N_TRAIN, seed: int = TASK_SEED) -> Task:
    """The running example: a fixed random teacher, n training records and 20,000 test records."""
    rng = np.random.default_rng(seed)
    teacher = make_teacher(rng)
    return Task(make_dataset(rng, n, teacher), make_dataset(rng, N_TEST, teacher))


def run(rng: np.random.Generator, task: Task, cfg: DpsgdConfig) -> float:
    params, _ = train_dpsgd(rng, task.train, cfg)
    return accuracy(params, task.test)


# --- Section: batch size at a fixed budget ---------------------------------------------------------


def batch_size_row(q: float, epochs: int, n: int, eps: float, delta: float) -> tuple[int, float, float]:
    """(steps, σ needed, noise standard deviation on the averaged gradient in units of C) at fixed epochs."""
    steps = max(1, round(epochs / q))
    if q >= 1:
        # Full batches: T Gaussian steps compose like one with μ = √T/σ.
        sigma = full_batch_sigma(steps, eps, delta)
    else:
        sigma = noise_multiplier_for(q, steps, eps, delta)
    return steps, sigma, sigma / (q * n)


def gaussian_delta(eps: float, mu: float) -> float:
    def phi(x: float) -> float:
        return 0.5 * math.erfc(-x / math.sqrt(2))

    return phi(mu / 2 - eps / mu) - math.exp(eps) * phi(-mu / 2 - eps / mu)


def full_batch_sigma(steps: int, eps: float, delta: float) -> float:
    mu = bisect(lambda m: gaussian_delta(eps, m) > delta, 1e-6, 50.0)
    return math.sqrt(steps) / mu


# --- Section: ghost clipping cost ------------------------------------------------------------------


def layer_costs(batch: int, seq: int, p_in: int, r_out: int) -> tuple[int, int]:
    """Floats stored to clip one dense layer: explicit per-example gradients (B·p·r) against the ghost
    norm's Gram matrices (B·T² for each of the inputs and the errors, plus nothing when T = 1)."""
    explicit = batch * p_in * r_out
    ghost = batch * (p_in + r_out) if seq == 1 else 2 * batch * seq * seq
    return explicit, ghost


def ghost_norm_sequence(a: Array, d: Array) -> float:
    """‖Σ_t d_t a_tᵀ‖_F for one example of a sequence model, from the two T×T Gram matrices."""
    return math.sqrt(float(np.sum((a @ a.T) * (d @ d.T))))


# --- The chapter's numbers ----------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Accounting for the running example")
    eps, alpha = epsilon_from_rdp([STEPS_RUN * e for e in rdp_curve(Q_RUN, SIGMA_RUN)], DELTA_RUN)
    print(f"q=0.01, sigma=1, T=10000, delta=1e-5: eps={eps:.2f} at alpha={alpha:g}")
    task = make_task()
    p0 = init_params(rng)
    print(f"weights p={p0.size}  expected batch B={Q_RUN * N_TRAIN:.0f}  epochs={Q_RUN * STEPS_RUN:.0f}")
    b = Q_RUN * N_TRAIN
    sd = SIGMA_RUN * CLIP_RUN / b
    print(
        f"noise on the averaged gradient: sd per coordinate={sd:.4f}  expected length about "
        f"sd*sqrt(p)={sd * math.sqrt(p0.size):.3f}  (clipped mean has length at most C={CLIP_RUN})"
    )
    for e in EPS_GRID:
        print(f"eps={e:g}: sigma={noise_multiplier_for(Q_RUN, STEPS_RUN, e, DELTA_RUN):.3g}")

    print("== Ghost norms")
    bw = backward(p0, task.train.x[:256], task.train.y[:256])
    explicit = np.linalg.norm(explicit_grads(bw), axis=1)
    ghost = ghost_norms(bw)
    print(
        f"256 examples: largest relative difference ghost vs explicit = {float(np.max(np.abs(ghost / explicit - 1))):.1e}"
    )
    a = rng.standard_normal((16, 12))
    d = rng.standard_normal((16, 7))
    g = d.T @ a
    print(
        f"sequence of 16 tokens: Gram-matrix norm={ghost_norm_sequence(a, d):.6f}  explicit={float(np.linalg.norm(g)):.6f}"
    )
    for batch, seq, p_in, r_out in (
        (100, 1, 51, 64),
        (32, 1, 768, 768),
        (32, 512, 768, 768),
        (32, 512, 768, 3072),
        (8, 4096, 4096, 4096),
    ):
        ex, gh = layer_costs(batch, seq, p_in, r_out)
        print(
            f"B={batch}, T={seq}, p={p_in}, r={r_out}: explicit {ex:,} floats  ghost {gh:,} floats  ratio {ex / gh:.1f}"
        )

    print("== Training the running example")
    nonpriv = DpsgdConfig(sigma=0.0, clip=math.inf)
    for steps in (500, 1000, 2000, 5000, 10000):
        acc = run(rng, task, DpsgdConfig(sigma=0.0, clip=math.inf, steps=steps))
        print(f"non-private SGD, {steps} steps: test accuracy {acc:.4f}")
    del nonpriv
    params, log = train_dpsgd(rng, task.train, DpsgdConfig(), test=task.test, log_every=1000)
    print(f"DP-SGD sigma=1 (eps=6.71): test accuracy {accuracy(params, task.test):.4f}")
    for t, acc, frac, med, sig in zip(
        log.steps, log.test_acc, log.clipped_frac, log.median_norm, log.signal_norm, strict=True
    ):
        print(
            f"  step {t}: test acc {acc:.4f}  clipped {frac:.2f}  median norm {med:.3f}  "
            f"length of clipped mean {sig:.3f}"
        )

    print("== Privacy against accuracy (3 seeds)")
    for e in EPS_GRID:
        sigma = noise_multiplier_for(Q_RUN, STEPS_RUN, e, DELTA_RUN)
        accs = [run(rng, task, DpsgdConfig(sigma=sigma)) for _ in range(3)]
        print(f"eps={e:g} (sigma={sigma:.3g}): test accuracy {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    accs = [run(rng, task, DpsgdConfig(sigma=0.0, clip=math.inf, steps=1000)) for _ in range(3)]
    print(f"non-private, 1000 steps: test accuracy {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    accs = [run(rng, task, DpsgdConfig(sigma=0.0)) for _ in range(3)]
    print(f"clipped, no noise, 10000 steps: test accuracy {np.mean(accs):.4f} +- {np.std(accs):.4f}")

    print("== Clipping norm (sigma=1, eps=6.71)")
    for clip in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        fixed = run(rng, task, DpsgdConfig(clip=clip))
        scaled = run(rng, task, DpsgdConfig(clip=clip, lr=LR_RUN * CLIP_RUN / clip))
        print(f"C={clip:g}: fixed lr {fixed:.4f}  lr*C fixed {scaled:.4f}")

    print("== Batch size at eps=4, delta=1e-5, 100 epochs")
    for q in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        steps, sigma, noise = batch_size_row(q, 100, N_TRAIN, 4.0, DELTA_RUN)
        print(
            f"q={q:g}: B={q * N_TRAIN:.0f}  steps={steps}  sigma={sigma:.3g}  sigma/B={noise:.3g}  "
            f"sigma/sqrt(q)={sigma / math.sqrt(q):.3g}  (e^(1/sigma^2)-1)*sigma^2={math.expm1(1 / sigma**2) * sigma**2:.3g}"
        )
    for q, lrs in ((0.003, (0.25, 0.5, 1.0)), (0.01, (0.5, 1.0, 2.0)), (0.03, (1.0, 2.0, 4.0)), (0.1, (2.0, 4.0, 8.0))):
        steps, sigma, _ = batch_size_row(q, 100, N_TRAIN, 4.0, DELTA_RUN)
        accs = [run(rng, task, DpsgdConfig(q=q, sigma=sigma, lr=lr, steps=steps)) for lr in lrs]
        print(f"q={q:g}: " + "  ".join(f"lr={lr:g}: {a:.4f}" for lr, a in zip(lrs, accs, strict=True)))

    print("== Microbatches (sigma=1, eps=6.71, B=100)")
    b = Q_RUN * N_TRAIN
    for size in (1, 4, 16):
        sd = microbatch_noise(SIGMA_RUN, CLIP_RUN, b, size)
        print(
            f"microbatch of {size}: noise sd per coordinate={sd:.4f} ({sd / microbatch_noise(SIGMA_RUN, CLIP_RUN, b, 1):.0f}x)"
        )
    for size in (4, 16):
        print(f"microbatch of {size}: test accuracy {run(rng, task, DpsgdConfig(microbatch=size)):.4f}")


# --- Figures --------------------------------------------------------------------------------


def plot_privacy_accuracy(rng: np.random.Generator) -> Figure:
    task = make_task()
    eps_list = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    accs = [run(rng, task, DpsgdConfig(sigma=noise_multiplier_for(Q_RUN, STEPS_RUN, e, DELTA_RUN))) for e in eps_list]
    base = run(rng, task, DpsgdConfig(sigma=0.0, clip=math.inf, steps=1000))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.semilogx(eps_list, accs, marker="o", color="black", label="DP-SGD, 10,000 steps, $q = 0.01$, $C = 0.5$")
    ax.axhline(base, color="tab:gray", ls="--", label="non-private SGD")
    ax.set_xlabel(r"$\varepsilon$ at $\delta = 10^{-5}$")
    ax.set_ylabel("test accuracy")
    ax.set_xticks(eps_list, [f"{e:g}" for e in eps_list])
    ax.legend()
    ax.set_title("The running example: privacy against accuracy")
    return fig


def plot_training_curves(rng: np.random.Generator) -> Figure:
    task = make_task()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for sigma, color, label in (
        (0.0, "tab:gray", "clipped, no noise"),
        (1.0, "black", r"$\sigma = 1$ ($\varepsilon = 6.71$)"),
        (4.0, "tab:red", r"$\sigma = 4$"),
    ):
        _, log = train_dpsgd(rng, task.train, DpsgdConfig(sigma=sigma), test=task.test, log_every=250)
        axes[0].plot(log.steps, log.test_acc, color=color, label=label)
        axes[1].plot(log.steps, log.signal_norm, color=color, label=label)
    sd = SIGMA_RUN * CLIP_RUN / (Q_RUN * N_TRAIN) * math.sqrt(init_params(rng).size)
    axes[1].axhline(sd, color="black", ls=":", label=r"noise length, $\sigma = 1$")
    axes[1].axhline(4 * sd, color="tab:red", ls=":", label=r"noise length, $\sigma = 4$")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("test accuracy")
    axes[0].legend()
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("length of the averaged clipped gradient")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    return fig


def plot_clipping(rng: np.random.Generator) -> Figure:
    task = make_task()
    clips = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    fixed = [run(rng, task, DpsgdConfig(clip=c)) for c in clips]
    scaled = [run(rng, task, DpsgdConfig(clip=c, lr=LR_RUN * CLIP_RUN / c)) for c in clips]
    _, log = train_dpsgd(rng, task.train, DpsgdConfig(sigma=0.0, clip=math.inf), test=task.test, log_every=500)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].semilogx(clips, fixed, marker="o", color="tab:blue", label=r"fixed step $\eta = 0.5$")
    axes[0].semilogx(clips, scaled, marker="s", color="black", label=r"fixed $\eta C = 0.25$")
    axes[0].set_xlabel("clipping norm $C$")
    axes[0].set_ylabel("test accuracy")
    axes[0].legend()
    axes[0].set_title(r"$\sigma = 1$, $\varepsilon = 6.71$")
    axes[1].plot(log.steps, log.median_norm, color="black")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("step (non-private run)")
    axes[1].set_ylabel("median per-example gradient norm")
    return fig


def plot_batch_size() -> Figure:
    qs = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    rows = [batch_size_row(q, 100, N_TRAIN, 4.0, DELTA_RUN) for q in qs]
    bs = [q * N_TRAIN for q in qs]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].loglog(bs, [r[1] for r in rows], marker="o", color="black", label=r"noise multiplier $\sigma$")
    ref = np.array(bs)
    axes[0].loglog(ref, rows[0][1] * np.sqrt(ref / ref[0]), color="tab:gray", ls=":", label=r"$\propto\sqrt{B}$")
    axes[0].set_xlabel("expected batch size $B = qn$")
    axes[0].legend()
    axes[0].set_title(r"$\varepsilon = 4$, 100 epochs, $n = 10{,}000$")
    axes[1].loglog(
        bs, [r[2] for r in rows], marker="o", color="black", label=r"noise on the averaged gradient, $\sigma/B$"
    )
    axes[1].loglog(ref, rows[0][2] * np.sqrt(ref[0] / ref), color="tab:gray", ls=":", label=r"$\propto 1/\sqrt{B}$")
    axes[1].set_xlabel("expected batch size $B = qn$")
    axes[1].set_ylabel("in units of $C$")
    axes[1].legend()
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
        "dpsgd_privacy_accuracy.svg": plot_privacy_accuracy(fig_rng),
        "dpsgd_training.svg": plot_training_curves(fig_rng),
        "dpsgd_clipping.svg": plot_clipping(fig_rng),
        "dpsgd_batch_size.svg": plot_batch_size(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
