"""Check the numbers in the public data and private fine-tuning chapter and draw its figures.

    uv run python code/part05-dp-ml/public_data.py
    uv run python code/part05-dp-ml/public_data.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 06-public-data.md (they match exactly with the default seed), then saves
the chapter's figures.

The private task is the part's running example (dp_sgd.make_task): 50-dimensional standard normal
inputs labelled by a fixed random teacher network, n = 10,000 training records, a 64-unit tanh student
with 3,329 weights. The PUBLIC task is a related but different one: 50,000 inputs from the same
distribution, labelled by a second teacher whose hidden units are the private teacher's with a small
perturbation (each first-layer weight moves by 0.15 standard deviations) and whose output weights are
partly replaced by fresh ones (rotated by 0.3 radians towards a random vector). Pretraining on it
helps on the private task without solving it: the pretrained student labels 91% of the private test
inputs correctly, against 96% for a student trained non-privately on the private data.

The pretrained student is then fine-tuned with DP-SGD on the private records in four ways: all 3,329
weights; the last layer only (65 weights, a convex logistic regression on frozen features); a rank-r
update W1 + U Vᵀ to the first layer plus the last layer (low-rank adaptation, 115 r + 65 weights); and,
for comparison, training from scratch. The step size of each is chosen on a validation half of the
test set at ε = 1; accuracy is reported on the other half.

Public functions: private_teacher, public_task, pretrained, split_test, LoraParams, lora_merge,
lora_clipped_sum, train_last_layer, train_lora, finetune, trainable_count, noise_length,
ggn_eigenvalues, effective_dimension, noise_cost, public_basis, captured_fraction, break_even_steps.
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
matplotlib.rcParams["svg.hashsalt"] = "public_data"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from dp_sgd import (
    TASK_SEED,
    Backward,
    Dataset,
    DpsgdConfig,
    Params,
    accuracy,
    backward,
    clipped_grad_sum,
    explicit_grads,
    forward,
    init_params,
    make_dataset,
    make_task,
    make_teacher,
    noise_multiplier_for,
    sigmoid,
    train_dpsgd,
)
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]

# The public task: a fixed object, like the private one.
PUBLIC_SEED = 1
N_PUBLIC = 50_000
TEACHER_SHIFT = 0.15  # standard deviation of the perturbation of the teacher's hidden units
TEACHER_ROTATION = 0.3  # radians by which the teacher's output weights turn towards a random vector
PRETRAIN = DpsgdConfig(q=100 / N_PUBLIC, sigma=0.0, clip=math.inf, lr=0.5, steps=10_000)

# Private fine-tuning: the DP-SGD configuration of the part, with the step size chosen per method.
Q, CLIP, STEPS, DELTA = 0.01, 0.5, 10_000, 1e-5
EPS_TABLE: tuple[float, ...] = (0.5, 1.0, 2.0, 8.0)
EPS_FIGURE: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
RANK = 4
METHODS: tuple[str, ...] = ("scratch", "full", "last", "lora")
LABELS: dict[str, str] = {
    "scratch": "from scratch, all 3,329 weights",
    "full": "pretrained, all 3,329 weights",
    "last": "pretrained, last layer (65 weights)",
    "lora": f"pretrained, rank-{RANK} adapter + last layer (525 weights)",
}
LR_GRID: dict[str, tuple[float, ...]] = {
    "scratch": (0.25, 0.5, 1.0),
    "full": (0.03, 0.1, 0.3),
    "last": (0.1, 0.3, 1.0),
    "lora": (0.03, 0.1, 0.3),
}
SEEDS_TABLE = 2
K_SUBSPACE: tuple[int, ...] = (1, 4, 16, 64, 256)
K_MAX = 256  # public directions kept (from 500 public records)


# --- The two tasks -----------------------------------------------------------------------------------


def private_teacher() -> Params:
    """The teacher of the running example: make_task draws it first from the task's seed."""
    return make_teacher(np.random.default_rng(TASK_SEED))


def related_teacher(rng: np.random.Generator, base: Params, shift: float, rotation: float) -> Params:
    """A teacher that shares most of `base`'s structure: perturbed hidden units, partly new output weights."""
    w1 = base.w1 + shift * rng.standard_normal(base.w1.shape)
    w2 = math.cos(rotation) * base.w2 + math.sin(rotation) * rng.standard_normal(base.w2.shape)
    return Params(w1, w2)


@cache
def _public(shift: float, rotation: float, n: int) -> tuple[Params, Dataset]:
    rng = np.random.default_rng(PUBLIC_SEED)
    teacher = related_teacher(rng, private_teacher(), shift, rotation)
    return teacher, make_dataset(rng, n, teacher)


def public_task(shift: float = TEACHER_SHIFT, rotation: float = TEACHER_ROTATION, n: int = N_PUBLIC) -> Dataset:
    return _public(shift, rotation, n)[1]


def public_teacher(shift: float = TEACHER_SHIFT, rotation: float = TEACHER_ROTATION) -> Params:
    return _public(shift, rotation, N_PUBLIC)[0]


@cache
def pretrained(shift: float = TEACHER_SHIFT, rotation: float = TEACHER_ROTATION, steps: int = PRETRAIN.steps) -> Params:
    """The student trained non-privately on the public task: a fixed object, shared by every experiment."""
    rng = np.random.default_rng(PUBLIC_SEED + 1)
    cfg = DpsgdConfig(q=PRETRAIN.q, sigma=0.0, clip=math.inf, lr=PRETRAIN.lr, steps=steps)
    params, _ = train_dpsgd(rng, public_task(shift, rotation), cfg)
    return params


def split_test(test: Dataset) -> tuple[Dataset, Dataset]:
    """(validation, report): the first and second halves of the test set."""
    half = test.y.size // 2
    return Dataset(test.x[:half], test.y[:half]), Dataset(test.x[half:], test.y[half:])


def agreement(a: Params, b: Params, x: Array) -> float:
    same: NDArray[np.bool_] = (forward(a, x).f >= 0) == (forward(b, x).f >= 0)
    return float(same.mean())


# --- Section: why dimension costs ----------------------------------------------------------------------


def trainable_count(method: str, rank: int = RANK, d_in: int = 51, hidden: int = 64) -> int:
    """Weights that fine-tuning changes (d_in and hidden include the bias)."""
    last = hidden + 1
    if method == "last":
        return last
    if method == "lora":
        return rank * (d_in + hidden) + last
    return hidden * d_in + last


def noise_length(sigma: float, clip: float, batch: float, k: int) -> float:
    """Approximate length σC√k/B of the noise on the averaged gradient in k trainable coordinates."""
    return sigma * clip * math.sqrt(k) / batch


def noise_cost(h: Array, eta: float, s: float, steps: int) -> Array:
    """Expected loss increase from T noisy steps along directions of curvature h, noise sd s per coordinate:
    h v_T / 2 with v_T = η²s² (1 - (1-ηh)^{2T}) / (1 - (1-ηh)²)."""
    shrink = (1 - eta * h) ** 2
    var = eta**2 * s**2 * np.where(h > 0, (1 - shrink**steps) / np.maximum(1 - shrink, 1e-300), 2.0 * steps)
    return h * var / 2


def effective_dimension(h: Array, eta: float, steps: int) -> float:
    """Σ_j (1 - (1 - ηh_j)^{2T}): how many directions the noise has had time to fill."""
    return float(np.sum(1 - (1 - eta * h) ** (2 * steps)))


def output_gradients(p: Params, x: Array) -> Array:
    """Per-example gradients of the logit f with respect to all weights, (b, p)."""
    fwd = forward(p, x)
    d1 = p.w2[None, :-1] * (1 - fwd.a1**2)
    return explicit_grads(Backward(fwd, d1, np.ones(x.shape[0])))


def ggn_eigenvalues(p: Params, x: Array, last_only: bool = False) -> Array:
    """Eigenvalues, largest first, of the Gauss-Newton curvature (1/m) Σ s(f)(1 - s(f)) ∇f ∇fᵀ of the
    logistic loss at p, estimated from the m inputs x (the Hessian without the term that vanishes when
    the model fits). From the m x m Gram matrix, which has the same nonzero eigenvalues."""
    f = forward(p, x).f
    s = sigmoid(f) * (1 - sigmoid(f))
    j = output_gradients(p, x)
    if last_only:
        j = j[:, p.w1.size :]
    js = j * np.sqrt(s)[:, None]
    m = x.shape[0]
    small = js.T @ js if js.shape[1] < m else js @ js.T
    return np.maximum(np.linalg.eigvalsh(small / m)[::-1], 0.0)


# --- Section: features: the last layer as private logistic regression -------------------------------------


@dataclass(frozen=True)
class FineTuneConfig:
    sigma: float
    lr: float
    q: float = Q
    clip: float = CLIP
    steps: int = STEPS


def train_last_layer(rng: np.random.Generator, base: Params, data: Dataset, cfg: FineTuneConfig) -> Params:
    """DP-SGD on the output layer alone: logistic regression on the frozen features a1b (65 numbers)."""
    feats = forward(base, data.x).a1b
    n = data.y.size
    batch = cfg.q * n
    w2 = base.w2.copy()
    for _ in range(cfg.steps):
        idx = np.flatnonzero(rng.random(n) < cfg.q)
        h, y = feats[idx], data.y[idx]
        d2 = -y * sigmoid(-y * (h @ w2))
        norms = np.abs(d2) * np.linalg.norm(h, axis=1)
        scale = np.minimum(1.0, cfg.clip / np.maximum(norms, 1e-300))
        g = h.T @ (d2 * scale) + cfg.sigma * cfg.clip * rng.standard_normal(w2.shape)
        w2 = w2 - cfg.lr * g / batch
    return Params(base.w1, w2)


def fit_last_layer(base: Params, data: Dataset, lam: float = 1e-4, iters: int = 30) -> Params:
    """Non-private logistic regression on the frozen features, by Newton's method: the probe's ceiling."""
    feats = forward(base, data.x).a1b
    n = data.y.size

    def objective(w: Array) -> float:
        return float(np.mean(np.logaddexp(0.0, -data.y * (feats @ w)))) + lam / 2 * float(w @ w)

    w = base.w2.copy()
    for _ in range(iters):
        m = data.y * (feats @ w)
        g = feats.T @ (-data.y * sigmoid(-m)) / n + lam * w
        s = sigmoid(m) * (1 - sigmoid(m))
        hess = (feats * s[:, None]).T @ feats / n + lam * np.eye(w.size)
        step = np.linalg.solve(hess, g)
        t = 1.0
        while objective(w - t * step) > objective(w) and t > 1e-8:  # damped Newton: never go uphill
            t /= 2
        w = w - t * step
    return Params(base.w1, w)


# --- Section: low-rank adapters --------------------------------------------------------------------------


@dataclass(frozen=True)
class LoraParams:
    u: Array  # (hidden, r), starts at zero
    v: Array  # (d + 1, r), random
    w2: Array  # (hidden + 1,)


def lora_init(rng: np.random.Generator, base: Params, rank: int) -> LoraParams:
    hidden, d_in = base.w1.shape
    return LoraParams(np.zeros((hidden, rank)), rng.standard_normal((d_in, rank)) / math.sqrt(d_in), base.w2.copy())


def lora_merge(base: Params, lp: LoraParams) -> Params:
    return Params(base.w1 + lp.u @ lp.v.T, lp.w2)


@dataclass(frozen=True)
class LoraSum:
    gu: Array
    gv: Array
    g2: Array
    norms: Array


def lora_parts(base: Params, lp: LoraParams, x: Array, y: Array) -> tuple[Backward, Array, Array]:
    """Backward pass through W1 + U Vᵀ, and the two projected vectors Vᵀa_i and Uᵀδ_i of each record."""
    bw = backward(lora_merge(base, lp), x, y)
    return bw, bw.fwd.a0 @ lp.v, bw.d1 @ lp.u


def lora_clipped_sum(base: Params, lp: LoraParams, x: Array, y: Array, clip: float) -> LoraSum:
    """Σ_i clip(g_i) for the adapter's weights, by ghost norms: record i's gradient for U is the outer
    product δ_i (Vᵀa_i)ᵀ, for V it is a_i (Uᵀδ_i)ᵀ, for w2 it is d2_i a1b_i."""
    bw, av, du = lora_parts(base, lp, x, y)
    sq = (
        (bw.d1**2).sum(axis=1) * (av**2).sum(axis=1)
        + (bw.fwd.a0**2).sum(axis=1) * (du**2).sum(axis=1)
        + bw.d2**2 * (bw.fwd.a1b**2).sum(axis=1)
    )
    norms = np.sqrt(sq)
    c = np.minimum(1.0, clip / np.maximum(norms, 1e-300)) if math.isfinite(clip) else np.ones_like(norms)
    return LoraSum((bw.d1 * c[:, None]).T @ av, bw.fwd.a0.T @ (du * c[:, None]), bw.fwd.a1b.T @ (bw.d2 * c), norms)


def lora_explicit_grads(base: Params, lp: LoraParams, x: Array, y: Array) -> Array:
    """Per-example adapter gradients formed explicitly, flattened (for checking the ghost norms)."""
    bw, av, du = lora_parts(base, lp, x, y)
    gu = bw.d1[:, :, None] * av[:, None, :]
    gv = bw.fwd.a0[:, :, None] * du[:, None, :]
    g2 = bw.d2[:, None] * bw.fwd.a1b
    b = x.shape[0]
    return np.hstack([gu.reshape(b, -1), gv.reshape(b, -1), g2])


def lora_loss(base: Params, lp: LoraParams, x: Array, y: Array) -> float:
    return float(np.sum(np.logaddexp(0.0, -y * forward(lora_merge(base, lp), x).f)))


def train_lora(rng: np.random.Generator, base: Params, data: Dataset, cfg: FineTuneConfig, rank: int) -> Params:
    """DP-SGD on (U, V, w2) with W1 = W1_pretrained + U Vᵀ: Poisson batches, clipping, noise σC on each sum."""
    lp = lora_init(rng, base, rank)
    n = data.y.size
    batch = cfg.q * n
    sd = cfg.sigma * cfg.clip
    for _ in range(cfg.steps):
        idx = np.flatnonzero(rng.random(n) < cfg.q)
        s = lora_clipped_sum(base, lp, data.x[idx], data.y[idx], cfg.clip)
        gu = s.gu + sd * rng.standard_normal(s.gu.shape)
        gv = s.gv + sd * rng.standard_normal(s.gv.shape)
        g2 = s.g2 + sd * rng.standard_normal(s.g2.shape)
        lp = LoraParams(lp.u - cfg.lr * gu / batch, lp.v - cfg.lr * gv / batch, lp.w2 - cfg.lr * g2 / batch)
    return lora_merge(base, lp)


def finetune(rng: np.random.Generator, method: str, data: Dataset, cfg: FineTuneConfig, rank: int = RANK) -> Params:
    """Train privately by one of METHODS, from the pretrained student (or from scratch)."""
    dp = DpsgdConfig(q=cfg.q, sigma=cfg.sigma, clip=cfg.clip, lr=cfg.lr, steps=cfg.steps)
    if method == "scratch":
        return train_dpsgd(rng, data, dp)[0]
    if method == "full":
        return train_dpsgd(rng, data, dp, init=pretrained())[0]
    if method == "last":
        return train_last_layer(rng, pretrained(), data, cfg)
    if method == "lora":
        return train_lora(rng, pretrained(), data, cfg, rank)
    raise ValueError(method)


def clipped_mean_length(method: str, p: Params, data: Dataset, clip: float = CLIP, rank: int = RANK) -> float:
    """Length of the average clipped gradient over all of `data`, in the method's trainable weights, at the
    start of training (for the adapter: U = 0, V at its random start, drawn from a fixed seed)."""
    n = data.y.size
    if method == "lora":
        lp = lora_init(np.random.default_rng(0), p, rank)
        s = lora_clipped_sum(p, lp, data.x, data.y, clip)
        return math.sqrt(float((s.gu**2).sum() + (s.gv**2).sum() + (s.g2**2).sum())) / n
    if method == "last":
        h = forward(p, data.x).a1b
        d2 = -data.y * sigmoid(-data.y * forward(p, data.x).f)
        c = np.minimum(1.0, clip / np.maximum(np.abs(d2) * np.linalg.norm(h, axis=1), 1e-300))
        return float(np.linalg.norm(h.T @ (d2 * c))) / n
    cs = clipped_grad_sum(p, data.x, data.y, clip)
    return math.sqrt(float((cs.g1**2).sum() + (cs.g2**2).sum())) / n


# --- Section: gradients near a public subspace -----------------------------------------------------------


def clipped_example_grads(p: Params, x: Array, y: Array, clip: float = CLIP) -> Array:
    g = explicit_grads(backward(p, x, y))
    norms = np.linalg.norm(g, axis=1)
    return g * np.minimum(1.0, clip / np.maximum(norms, 1e-300))[:, None]


def public_basis(p: Params, x: Array, y: Array, k_max: int) -> Array:
    """Orthonormal columns spanning the top-k_max principal directions of the public clipped per-example
    gradients, (p, k_max); computed from the m x m Gram matrix."""
    g = clipped_example_grads(p, x, y)
    w, u = np.linalg.eigh(g @ g.T)
    order = np.argsort(w)[::-1][:k_max]
    return (g.T @ u[:, order]) / np.sqrt(w[order])


def captured_fraction(vec: Array, basis: Array, k: int) -> float:
    """‖P_k vec‖² / ‖vec‖² for the span of the first k columns of an orthonormal basis."""
    return float(np.sum((basis[:, :k].T @ vec) ** 2) / np.sum(vec**2))


def break_even_steps(signal: float, captured: float, k: int, noise_sd: float) -> float:
    """Steps after which the kept signal, of length signal·√captured and constant direction, outgrows the
    random walk of k coordinates of noise with standard deviation noise_sd: (noise_sd √k / kept signal)²."""
    return (noise_sd * math.sqrt(k) / (signal * math.sqrt(captured))) ** 2


# --- The chapter's numbers ----------------------------------------------------------------------------


def private_sigma(eps: float) -> float:
    return noise_multiplier_for(Q, STEPS, eps, DELTA)


def choose_lr(
    rng: np.random.Generator, method: str, train: Dataset, val: Dataset, eps: float
) -> tuple[float, list[float]]:
    sigma = private_sigma(eps)
    accs = [accuracy(finetune(rng, method, train, FineTuneConfig(sigma, lr)), val) for lr in LR_GRID[method]]
    return LR_GRID[method][int(np.argmax(accs))], accs


def print_table(rng: np.random.Generator) -> dict[str, float]:
    """Print every number the chapter quotes; return the step sizes chosen for each method."""
    task = make_task()
    val, report = split_test(task.test)
    base = pretrained()
    teacher = private_teacher()

    print("== The public task and the pretrained model")
    pub = public_task()
    print(
        f"public records={pub.y.size}  public teacher agrees with private labels on {agreement(public_teacher(), teacher, report.x):.4f}"
    )
    print(
        f"pretrained: public training accuracy {accuracy(base, pub):.4f}  private report accuracy {accuracy(base, report):.4f}"
    )
    nonpriv = train_dpsgd(
        np.random.default_rng(TASK_SEED + 1), task.train, DpsgdConfig(sigma=0.0, clip=math.inf, steps=1000)
    )[0]
    print(f"non-private from scratch, 1000 steps: report accuracy {accuracy(nonpriv, report):.4f}")
    ft = train_dpsgd(
        np.random.default_rng(TASK_SEED + 1), task.train, DpsgdConfig(sigma=0.0, clip=math.inf, steps=1000), init=base
    )[0]
    print(f"non-private fine-tuning of all weights, 1000 steps: report accuracy {accuracy(ft, report):.4f}")
    probe = fit_last_layer(base, task.train)
    print(f"non-private last layer (exact logistic regression): report accuracy {accuracy(probe, report):.4f}")

    print("== Trainable weights and noise length (q=0.01, C=0.5, B=100)")
    for m in ("full", "lora", "last"):
        print(f"{m}: k={trainable_count(m)}")
    for r in (1, 2, 4, 8, 16):
        print(f"lora rank {r}: k={trainable_count('lora', r)}")
    feats = forward(base, task.train.x).a1b
    print(
        f"feature norm: largest {float(np.max(np.linalg.norm(feats, axis=1))):.3f}  bound sqrt(65)={math.sqrt(65):.3f}"
    )
    for eps in EPS_TABLE:
        sigma = private_sigma(eps)
        lengths = "  ".join(
            f"k={trainable_count(m)}: {noise_length(sigma, CLIP, Q * task.train.y.size, trainable_count(m)):.3f}"
            for m in ("full", "lora", "last")
        )
        print(f"eps={eps:g} sigma={sigma:.3g}: noise length {lengths}")
    sigma1 = private_sigma(1.0)
    for m in METHODS:
        p = init_params(np.random.default_rng(0)) if m == "scratch" else base
        sig = clipped_mean_length(m, p, task.train)
        ratio = noise_length(sigma1, CLIP, Q * task.train.y.size, trainable_count(m)) / sig
        print(
            f"signal at start, {m}: length of the clipped mean {sig:.4f}  eps=1 noise/signal {ratio:.1f}  "
            f"break-even steps {ratio**2:.0f}"
        )

    print("== Curvature at the pretrained model (Gauss-Newton, 2,000 private records)")
    h = ggn_eigenvalues(base, task.train.x[:2000])
    h_last = ggn_eigenvalues(base, task.train.x[:2000], last_only=True)
    print(f"all weights: largest eigenvalue {h[0]:.3f}  trace {h.sum():.3f}  nonzero {int(np.sum(h > 1e-12))}")
    print(f"last layer: largest eigenvalue {h_last[0]:.3f}  smallest {h_last[-1]:.2e}  trace {h_last.sum():.3f}")
    for eta in (0.1, 0.5):
        print(
            f"eta={eta:g}, T=10000: d_eff all weights {effective_dimension(h, eta, STEPS):.0f}  "
            f"last layer {effective_dimension(h_last, eta, STEPS):.1f}  eigenvalues above 1/(2 eta T): {int(np.sum(h > 1 / (2 * eta * STEPS)))}"
        )
    s1 = private_sigma(1.0) * CLIP / (Q * task.train.y.size)
    for eta in (0.1, 0.5):
        full_cost = float(noise_cost(h, eta, s1, STEPS).sum())
        last_cost = float(noise_cost(h_last, eta, s1, STEPS).sum())
        print(f"eps=1, eta={eta:g}: predicted noise cost last layer {last_cost:.5f}")
        print(
            f"eps=1 (s={s1:.4f}), eta={eta:g}: predicted noise cost all weights {full_cost:.4f}  stationary bound p*eta*s^2/4={base.size * eta * s1**2 / 4:.4f}"
        )

    print("== Private gradients and the public subspace (500 public records)")
    basis = public_basis(base, pub.x[:500], pub.y[:500], K_MAX)
    gq = clipped_example_grads(base, task.train.x, task.train.y)
    mean = gq.mean(axis=0)
    sig_sq = float(np.sum(mean**2))
    print(f"private clipped mean: length {math.sqrt(sig_sq):.4f}")
    rand = np.linalg.qr(rng.standard_normal((mean.size, 256)))[0]
    for k in K_SUBSPACE:
        print(
            f"k={k}: mean captured {captured_fraction(mean, basis, k):.3f}  random {captured_fraction(mean, rand, k):.3f}  "
            f"(k/p={k / mean.size:.3f})"
        )
    for eps in (1.0, 8.0):
        sd = private_sigma(eps) * CLIP / (Q * task.train.y.size)
        cells = [
            f"k={k}: {break_even_steps(math.sqrt(sig_sq), captured_fraction(mean, basis, k), k, sd):.0f}"
            for k in K_SUBSPACE
        ]
        print(
            f"eps={eps:g}: break-even steps all weights {break_even_steps(math.sqrt(sig_sq), 1.0, mean.size, sd):.0f}  "
            + "  ".join(cells)
        )

    print("== Ghost norms for the adapter")
    lp = lora_init(rng, base, RANK)
    lp = LoraParams(0.1 * rng.standard_normal(lp.u.shape), lp.v, lp.w2)
    explicit = np.linalg.norm(lora_explicit_grads(base, lp, task.train.x[:256], task.train.y[:256]), axis=1)
    ghost = lora_clipped_sum(base, lp, task.train.x[:256], task.train.y[:256], math.inf).norms
    print(
        f"256 examples: largest relative difference ghost vs explicit = {float(np.max(np.abs(ghost / explicit - 1))):.1e}"
    )

    print("== Step size chosen on the validation half at eps=1")
    lrs: dict[str, float] = {}
    for m in METHODS:
        lrs[m], accs = choose_lr(rng, m, task.train, val, 1.0)
        print(
            f"{m}: "
            + "  ".join(f"lr={lr:g}: {a:.4f}" for lr, a in zip(LR_GRID[m], accs, strict=True))
            + f"  -> lr={lrs[m]:g}"
        )

    print(f"== Private fine-tuning, report half, {SEEDS_TABLE} runs each")
    for eps in EPS_TABLE:
        sigma = private_sigma(eps)
        cells: list[str] = []
        for m in METHODS:
            accs = [
                accuracy(finetune(rng, m, task.train, FineTuneConfig(sigma, lrs[m])), report)
                for _ in range(SEEDS_TABLE)
            ]
            cells.append(f"{m} {np.mean(accs):.4f} +- {np.std(accs):.4f}")
        print(f"eps={eps:g}: " + "  ".join(cells))

    print(f"== Adapter rank at eps=1 (lr={lrs['lora']:g}), {SEEDS_TABLE} runs each")
    for r in (1, 4, 16):
        cfg = FineTuneConfig(private_sigma(1.0), lrs["lora"])
        accs = [accuracy(finetune(rng, "lora", task.train, cfg, rank=r), report) for _ in range(SEEDS_TABLE)]
        print(f"rank {r}: k={trainable_count('lora', r)}  report accuracy {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    return lrs


# --- Figures --------------------------------------------------------------------------------


def plot_accuracy(rng: np.random.Generator, lrs: dict[str, float]) -> Figure:
    task = make_task()
    report = split_test(task.test)[1]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    styles = {"scratch": ("tab:gray", "o"), "full": ("black", "s"), "last": ("tab:blue", "^"), "lora": ("tab:red", "D")}
    for m in METHODS:
        accs = [
            accuracy(finetune(rng, m, task.train, FineTuneConfig(private_sigma(e), lrs[m])), report) for e in EPS_FIGURE
        ]
        color, marker = styles[m]
        ax.semilogx(EPS_FIGURE, accs, marker=marker, color=color, label=LABELS[m])
    ax.axhline(accuracy(pretrained(), report), color="tab:blue", ls=":", label="pretrained, no fine-tuning")
    ax.set_xticks(EPS_FIGURE, [f"{e:g}" for e in EPS_FIGURE])
    ax.minorticks_off()
    ax.set_xlabel(r"$\varepsilon$ at $\delta = 10^{-5}$")
    ax.set_ylabel("test accuracy (report half)")
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_noise_signal() -> Figure:
    task = make_task()
    base = pretrained()
    batch = Q * task.train.y.size
    sigma = private_sigma(1.0)
    ranks = (1, 2, 4, 8, 16)
    ks = [trainable_count("last")] + [trainable_count("lora", r) for r in ranks] + [trainable_count("full")]
    signal = (
        [clipped_mean_length("last", base, task.train)]
        + [clipped_mean_length("lora", base, task.train, rank=r) for r in ranks]
        + [clipped_mean_length("full", base, task.train)]
    )
    grid = np.geomspace(30, 5000, 50)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    axes[0].loglog(
        grid,
        [noise_length(sigma, CLIP, batch, int(k)) for k in grid],
        color="black",
        label="noise length σC·√k / B",
    )
    axes[0].loglog(ks, signal, marker="o", color="tab:red", ls="", label="clipped mean gradient at the start")
    names = ["last layer"] + [f"r = {r}" for r in ranks] + ["all weights"]
    for k, s, name in zip(ks, signal, names, strict=True):
        axes[0].annotate(name, (k, s), textcoords="offset points", xytext=(4, -11), fontsize=7)
    axes[0].set_xlabel("trainable weights $k$")
    axes[0].set_ylabel("length")
    axes[0].set_title(r"$\varepsilon = 1$ ($\sigma = 4.13$), $B = 100$, $C = 0.5$")
    axes[0].legend(fontsize=8, loc="center right")
    ratio = [noise_length(sigma, CLIP, batch, k) / s for k, s in zip(ks, signal, strict=True)]
    axes[1].loglog(ks, ratio, marker="o", color="black")
    for k, r_, name in zip(ks, ratio, names, strict=True):
        axes[1].annotate(name, (k, r_), textcoords="offset points", xytext=(4, 4), fontsize=7)
    axes[1].set_xlabel("trainable weights $k$")
    axes[1].set_ylabel("noise length / signal length")
    axes[1].set_xlim(40, 8000)
    return fig


def plot_subspace(rng: np.random.Generator) -> Figure:
    task = make_task()
    base = pretrained()
    pub = public_task()
    h = ggn_eigenvalues(base, task.train.x[:2000])
    basis = public_basis(base, pub.x[:500], pub.y[:500], K_MAX)
    mean = clipped_example_grads(base, task.train.x, task.train.y).mean(axis=0)
    rand = np.linalg.qr(rng.standard_normal((mean.size, K_MAX)))[0]
    ks = np.arange(1, K_MAX + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    nz = h[h > 1e-12]
    axes[0].loglog(np.arange(1, nz.size + 1), nz, color="black")
    for eta, ls in ((0.1, "--"), (0.5, ":")):
        axes[0].axhline(1 / (2 * eta * STEPS), color="tab:red", ls=ls, label=rf"$1/(2\eta T)$, $\eta = {eta:g}$")
    axes[0].set_xlabel("index $j$")
    axes[0].set_ylabel("curvature $h_j$")
    axes[0].set_title("Gauss-Newton curvature at the pretrained model")
    axes[0].legend(fontsize=8)
    axes[1].semilogx(
        ks,
        [captured_fraction(mean, basis, int(k)) for k in ks],
        color="black",
        label="top m public gradient directions",
    )
    axes[1].semilogx(
        ks, [captured_fraction(mean, rand, int(k)) for k in ks], color="tab:gray", ls="--", label="m random directions"
    )
    axes[1].set_xlabel("subspace dimension $m$")
    axes[1].set_ylabel("share of the private mean gradient's energy")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8, loc="upper left")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="directory for the figures")
    args = parser.parse_args()
    seed: int = args.seed
    out: Path = args.out

    lrs = print_table(np.random.default_rng(seed))

    out.mkdir(parents=True, exist_ok=True)
    fig_rng = np.random.default_rng(seed + 1)
    figures = {
        "pub_accuracy.svg": plot_accuracy(fig_rng, lrs),
        "pub_noise_signal.svg": plot_noise_signal(),
        "pub_subspace.svg": plot_subspace(fig_rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
