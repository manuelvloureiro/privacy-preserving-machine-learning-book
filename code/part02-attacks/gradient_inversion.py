"""Recover training inputs and labels from the gradient of a small neural network.

    uv run python code/part02-attacks/gradient_inversion.py
    uv run python code/part02-attacks/gradient_inversion.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 06-gradient-inversion.md (they match exactly with the default seed),
then saves the chapter's figures.

The model is a one-hidden-layer network written in NumPy: x ∈ R^d, a = W1 x + b1, h = σ(a),
z = W2 h + b2, p = softmax(z), loss = -ln p_y. The activation σ is ReLU or the sigmoid. The data
are 8×8 synthetic images of four shapes (d = 64, K = 4 classes). A client computes the gradient
of the average loss on its batch; the attacker knows the parameters and sees that gradient.
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure

configure("gradient_inversion")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]
Activation = Literal["relu", "sigmoid"]

SIDE = 8  # images are SIDE × SIDE
D = SIDE * SIDE  # input dimension
K = 4  # classes: horizontal bar, vertical bar, diagonal, square outline
CLASS_NAMES = ("horizontal", "vertical", "diagonal", "square")


# --- Data: 8×8 images of four shapes ---------------------------------------------------------


def shape_template(label: int, offset: int) -> Array:
    """A clean 8×8 image of shape `label`, shifted by `offset` pixels (0 to 3)."""
    img = np.zeros((SIDE, SIDE))
    if label == 0:  # horizontal bar, two pixels thick
        img[2 + offset : 4 + offset, 1:7] = 1.0
    elif label == 1:  # vertical bar
        img[1:7, 2 + offset : 4 + offset] = 1.0
    elif label == 2:  # diagonal, either direction
        for i in range(SIDE):
            img[i, i if offset % 2 == 0 else SIDE - 1 - i] = 1.0
    else:  # square outline of side 4 + offset
        s = 4 + offset % 3
        lo = (SIDE - s) // 2
        img[lo, lo : lo + s] = img[lo + s - 1, lo : lo + s] = 1.0
        img[lo : lo + s, lo] = img[lo : lo + s, lo + s - 1] = 1.0
    return img


TEMPLATES = np.array([[shape_template(c, o).ravel() for o in range(4)] for c in range(K)])  # (K, 4, D)


def make_images(rng: np.random.Generator, n: int, noise: float = 0.1) -> tuple[Array, NDArray[np.int64]]:
    """n flattened images in [0, 1] with random shapes, shifts and pixel noise, and their labels."""
    labels = rng.integers(0, K, size=n)
    offsets = rng.integers(0, 4, size=n)
    x = np.clip(TEMPLATES[labels, offsets] + noise * rng.standard_normal((n, D)), 0.0, 1.0)
    return x, labels.astype(np.int64)


# --- The model and its gradient (Toolkit, Facts 1-3) ---------------------------------------


@dataclass(frozen=True)
class Params:
    w1: Array  # (m, d)
    b1: Array  # (m,)
    w2: Array  # (K, m)
    b2: Array  # (K,)


@dataclass(frozen=True)
class Grads:
    w1: Array
    b1: Array
    w2: Array
    b2: Array

    def flat(self) -> Array:
        return np.concatenate([self.w1.ravel(), self.b1, self.w2.ravel(), self.b2])


def init_params(rng: np.random.Generator, m: int, d: int = D, k: int = K, bias_scale: float = 0.1) -> Params:
    """Random initialization: W ~ N(0, 1/fan_in), biases ~ N(0, bias_scale²)."""
    return Params(
        w1=rng.standard_normal((m, d)) / math.sqrt(d),
        b1=bias_scale * rng.standard_normal(m),
        w2=rng.standard_normal((k, m)) / math.sqrt(m),
        b2=bias_scale * rng.standard_normal(k),
    )


def sigmoid(a: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-a))


def activate(a: Array, act: Activation) -> tuple[Array, Array, Array]:
    """σ(a), σ'(a), σ''(a) elementwise (ReLU: σ'' = 0 away from 0)."""
    if act == "relu":
        h = np.maximum(a, 0.0)
        d1 = (a > 0).astype(np.float64)
        return h, d1, np.zeros_like(a)
    s = sigmoid(a)
    d1 = s * (1 - s)
    return s, d1, d1 * (1 - 2 * s)


def softmax(z: Array) -> Array:
    """Row-wise softmax of a (B, K) array."""
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class Forward:
    a: Array  # (B, m) pre-activations
    h: Array  # (B, m) hidden activations
    s1: Array  # (B, m) σ'(a)
    s2: Array  # (B, m) σ''(a)
    p: Array  # (B, K) predicted probabilities
    d2: Array  # (B, K) p - e_y: the loss gradient with respect to the logits (Fact 2)
    back: Array  # (B, m) W2ᵀ d2: what flows back into the hidden layer
    d1: Array  # (B, m) back ⊙ σ'(a): the loss gradient with respect to a (Fact 1)


def forward(params: Params, x: Array, y: NDArray[np.int64], act: Activation) -> Forward:
    """Forward pass and the backward signals for a batch x of shape (B, d)."""
    a = x @ params.w1.T + params.b1
    h, s1, s2 = activate(a, act)
    p = softmax(h @ params.w2.T + params.b2)
    d2 = p.copy()
    d2[np.arange(len(y)), y] -= 1.0
    back = d2 @ params.w2
    return Forward(a=a, h=h, s1=s1, s2=s2, p=p, d2=d2, back=back, d1=back * s1)


def gradients(params: Params, x: Array, y: NDArray[np.int64], act: Activation) -> Grads:
    """Gradient of the average cross-entropy over the batch: a sum of outer products (Fact 3)."""
    f = forward(params, x, y, act)
    b = len(y)
    return Grads(w1=f.d1.T @ x / b, b1=f.d1.mean(axis=0), w2=f.d2.T @ f.h / b, b2=f.d2.mean(axis=0))


def mean_loss(params: Params, x: Array, y: NDArray[np.int64], act: Activation) -> float:
    f = forward(params, x, y, act)
    return float(-np.mean(np.log(f.p[np.arange(len(y)), y])))


# --- Section 14.3: exact recovery from a fully connected layer ------------------------------------


def unit_ratios(g: Grads, tol: float = 1e-12) -> tuple[Array, NDArray[np.bool_]]:
    """Row j of ∇W1 divided by entry j of ∇b1, for every hidden unit; mask of units with ∇b1_j ≠ 0."""
    usable = np.abs(g.b1) > tol
    safe = np.where(usable, g.b1, 1.0)
    return g.w1 / safe[:, None], usable


def exact_recovery(g: Grads) -> Array:
    """x̂ from the unit with the largest |∇b1_j| (any unit with ∇b1_j ≠ 0 gives the same x̂)."""
    j = int(np.argmax(np.abs(g.b1)))
    return g.w1[j] / g.b1[j]


def singular_values(g: Grads) -> Array:
    return np.linalg.svd(g.w1, compute_uv=False)


# --- Section 14.4: label recovery from the last layer ---------------------------------------------


def label_from_bias(g: Grads) -> int:
    """The only negative entry of ∇b2 = p - e_y is the label (single example)."""
    return int(np.argmin(g.b2))


def label_from_weights(g: Grads) -> int:
    """With h ≥ 0, row y of ∇W2 = (p_y - 1) hᵀ is the only row with a negative sum."""
    return int(np.argmin(g.w2.sum(axis=1)))


def label_counts_estimate(g: Grads, batch: int, mean_p: Array) -> Array:
    """n_k = Σ_i p_ik - B ∇b2_k, with Σ_i p_ik replaced by B times a guess `mean_p` of the average prediction."""
    return batch * (mean_p - g.b2)


def mean_prediction(params: Params, x: Array, act: Activation) -> Array:
    """The model's average predicted probabilities on inputs x (the attacker's own data)."""
    return forward(params, x, np.zeros(len(x), dtype=np.int64), act).p.mean(axis=0)


# --- Section 14.5: Deep Leakage from Gradients ----------------------------------------------------


def matching_loss_and_grad(
    params: Params, x_hat: Array, y: NDArray[np.int64], target: Grads, act: Activation
) -> tuple[float, Array]:
    """R(x̂) = ‖∇θ ℓ(θ; x̂, y) - g‖² summed over all four parameter blocks, and ∂R/∂x̂.

    The derivative is backpropagation through the gradient computation itself ("double
    backprop"), written out by hand for this one-hidden-layer model.
    """
    b = len(y)
    f = forward(params, x_hat, y, act)
    g = Grads(w1=f.d1.T @ x_hat / b, b1=f.d1.mean(axis=0), w2=f.d2.T @ f.h / b, b2=f.d2.mean(axis=0))
    rw1, rb1, rw2, rb2 = g.w1 - target.w1, g.b1 - target.b1, g.w2 - target.w2, g.b2 - target.b2
    loss = float(np.sum(rw1**2) + np.sum(rb1**2) + np.sum(rw2**2) + np.sum(rb2**2))
    c = 2.0 / b
    # Gradients of R with respect to the intermediate quantities of every example i.
    g_d2 = c * (f.h @ rw2.T + rb2)  # through ∇W2 = mean d2 hᵀ and ∇b2 = mean d2
    g_h = c * (f.d2 @ rw2)  # through ∇W2
    g_d1 = c * (x_hat @ rw1.T + rb1)  # through ∇W1 = mean d1 xᵀ and ∇b1 = mean d1
    g_x = c * (f.d1 @ rw1)  # through ∇W1 directly
    # d1 = (W2ᵀ d2) ⊙ σ'(a)
    g_d2 = g_d2 + (g_d1 * f.s1) @ params.w2.T
    g_a = g_d1 * f.back * f.s2
    # d2 = softmax(z) - e_y, so ∂R/∂z = p ⊙ (∂R/∂p - <p, ∂R/∂p>)
    g_z = f.p * (g_d2 - np.sum(f.p * g_d2, axis=1, keepdims=True))
    # z = W2 h + b2 and h = σ(a)
    g_h = g_h + g_z @ params.w2
    g_a = g_a + g_h * f.s1
    # a = W1 x + b1
    g_x = g_x + g_a @ params.w1
    return loss, g_x


@dataclass(frozen=True)
class DlgResult:
    x_hat: Array  # (B, d) final reconstruction
    losses: list[float]  # matching loss at every iterate, including the start and the end
    path: list[Array]  # every iterate, including the start and the end


def dlg(
    params: Params,
    target: Grads,
    y: NDArray[np.int64],
    rng: np.random.Generator,
    act: Activation = "sigmoid",
    steps: int = 600,
    lr: float = 0.05,
) -> DlgResult:
    """Deep Leakage from Gradients: start from noise and minimize R(x̂) with Adam (labels known)."""
    x = rng.uniform(0.0, 1.0, size=(len(y), params.w1.shape[1]))
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    beta1, beta2, eps = 0.9, 0.999, 1e-12
    losses: list[float] = []
    path: list[Array] = [x.copy()]
    for t in range(1, steps + 1):
        loss, grad = matching_loss_and_grad(params, x, y, target, act)
        losses.append(loss)
        m1 = beta1 * m1 + (1 - beta1) * grad
        m2 = beta2 * m2 + (1 - beta2) * grad**2
        step = lr * (1 - t / (steps + 1))  # linear decay to 0
        x = x - step * (m1 / (1 - beta1**t)) / (np.sqrt(m2 / (1 - beta2**t)) + eps)
        path.append(x.copy())
    losses.append(matching_loss_and_grad(params, x, y, target, act)[0])
    return DlgResult(x_hat=x, losses=losses, path=path)


def finite_difference_gap(
    params: Params, x_hat: Array, y: NDArray[np.int64], target: Grads, act: Activation, h: float = 1e-6
) -> float:
    """Largest gap between ∂R/∂x̂ from matching_loss_and_grad and central finite differences."""
    _, grad = matching_loss_and_grad(params, x_hat, y, target, act)
    numeric = np.zeros_like(x_hat)
    for idx in np.ndindex(*x_hat.shape):
        up = x_hat.copy()
        down = x_hat.copy()
        up[idx] += h
        down[idx] -= h
        r_up = matching_loss_and_grad(params, up, y, target, act)[0]
        r_down = matching_loss_and_grad(params, down, y, target, act)[0]
        numeric[idx] = (r_up - r_down) / (2 * h)
    return float(np.max(np.abs(numeric - grad)))


def relative_error(x_hat: Array, x: Array) -> float:
    return float(np.linalg.norm(x_hat - x) / np.linalg.norm(x))


def closest(x_hat: Array, target: Array) -> tuple[int, float]:
    """Index of the reconstruction closest to `target`, and its relative error."""
    dist = np.linalg.norm(x_hat - target[None, :], axis=1)
    j = int(np.argmin(dist))
    return j, float(dist[j] / np.linalg.norm(target))


def matched_errors(x_hat: Array, x: Array) -> Array:
    """For each true example, the relative error of the closest reconstruction."""
    dist = np.linalg.norm(x[:, None, :] - x_hat[None, :, :], axis=2)
    return dist.min(axis=1) / np.linalg.norm(x, axis=1)


# --- Section 14.6: batches -------------------------------------------------------------------------


def isolation_probability(batch: int, width: int) -> float:
    """P(a given example is the only one in the batch to switch on at least one of `width` units),
    if each unit switches on for each example independently with probability 1/2."""
    return 1.0 - (1.0 - 2.0**-batch) ** width


def recovered_by_ratios(g: Grads, x: Array, tol: float = 1e-6) -> NDArray[np.bool_]:
    """For each example in the batch: does some unit ratio ∇W1_j / ∇b1_j equal it (relative error < tol)?"""
    ratios, usable = unit_ratios(g)
    cand = ratios[usable]
    if len(cand) == 0:
        return np.zeros(len(x), dtype=np.bool_)
    return matched_errors(cand, x) < tol


def isolation_rate(
    rng: np.random.Generator, batch: int, width: int, trials: int, inputs: Literal["gaussian", "images"]
) -> float:
    """Fraction of batch examples recovered exactly by the unit ratios of a random ReLU network."""
    hits = 0
    for _ in range(trials):
        params = init_params(rng, width, bias_scale=0.0 if inputs == "gaussian" else 0.1)
        if inputs == "gaussian":
            x = rng.standard_normal((batch, D))
            y = rng.integers(0, K, size=batch).astype(np.int64)
        else:
            x, y = make_images(rng, batch)
        hits += int(recovered_by_ratios(gradients(params, x, y, "relu"), x).sum())
    return hits / (batch * trials)


def batch_label_counts_exact(
    rng: np.random.Generator, batch: int, width: int, trials: int, guess: Literal["uniform", "auxiliary"]
) -> float:
    """Fraction of random batches whose label counts are all recovered exactly from ∇b2.

    The attacker guesses the average prediction as uniform (1/K) or as the model's average
    prediction on 200 images of its own.
    """
    exact = 0
    for _ in range(trials):
        params = init_params(rng, width)
        x, y = make_images(rng, batch)
        if guess == "uniform":
            mean_p = np.full(K, 1.0 / K)
        else:
            mean_p = mean_prediction(params, make_images(rng, 200)[0], "relu")
        estimate = np.rint(label_counts_estimate(gradients(params, x, y, "relu"), batch, mean_p))
        exact += int(np.array_equal(estimate, np.bincount(y, minlength=K)))
    return exact / trials


def span_residuals(g: Grads, x: Array) -> Array:
    """Distance of each row of ∇W1 from the span of the batch inputs, relative to the row norm."""
    q, _ = np.linalg.qr(x.T)  # orthonormal basis of span{x_i}
    rows = g.w1
    resid = rows - (rows @ q) @ q.T
    norms = np.linalg.norm(rows, axis=1)
    keep = norms > 0
    return np.linalg.norm(resid[keep], axis=1) / norms[keep]


@dataclass(frozen=True)
class DlgBatchStats:
    median_error: float  # median over trials of the median matched relative error
    recovered: float  # fraction of examples recovered to relative error < 0.05
    median_loss: float  # median final matching loss


def dlg_batch_stats(rng: np.random.Generator, batch: int, width: int, trials: int, steps: int) -> DlgBatchStats:
    errors: list[float] = []
    hits = 0
    losses: list[float] = []
    for _ in range(trials):
        params = init_params(rng, width)
        x, y = make_images(rng, batch)
        result = dlg(params, gradients(params, x, y, "sigmoid"), y, rng, steps=steps)
        errs = matched_errors(result.x_hat, x)
        errors.append(float(np.median(errs)))
        hits += int(np.sum(errs < 0.05))
        losses.append(result.losses[-1])
    return DlgBatchStats(float(np.median(errors)), hits / (batch * trials), float(np.median(losses)))


# --- A worked example small enough to do by hand ----------------------------------------------

TINY = Params(
    w1=np.array([[0.4, -0.2, 0.1], [0.3, 0.5, -0.4]]),
    b1=np.array([0.1, -0.1]),
    w2=np.array([[1.0, -1.0], [0.5, 0.5]]),
    b2=np.array([0.0, 0.0]),
)
TINY_X = np.array([[0.5, 1.0, 0.2]])
TINY_Y = np.array([0], dtype=np.int64)


# --- The chapter's numbers ------------------------------------------------------------------

WIDTH = 32  # hidden units of the attacked network
DLG_STEPS = 600
DLG_TRIALS = 6


def fmt(v: Array, digits: int = 4) -> str:
    return "(" + ", ".join(f"{float(t):.{digits}f}" for t in v) + ")"


def print_table(rng: np.random.Generator) -> None:
    print("== Worked example: d = 3, m = 2, K = 2, ReLU ==")
    f = forward(TINY, TINY_X, TINY_Y, "relu")
    g = gradients(TINY, TINY_X, TINY_Y, "relu")
    print(f"a={fmt(f.a[0])}  h={fmt(f.h[0])}  z={fmt(f.h[0] @ TINY.w2.T + TINY.b2)}  p={fmt(f.p[0])}")
    print(f"d2=p-e_y={fmt(f.d2[0])}  W2^T d2={fmt(f.back[0])}  d1={fmt(f.d1[0])}")
    print(f"grad W1 row 1={fmt(g.w1[0])}  row 2={fmt(g.w1[1])}  grad b1={fmt(g.b1)}")
    ratios, _ = unit_ratios(g)
    print(f"ratios: unit 1 {fmt(ratios[0])}  unit 2 {fmt(ratios[1])}")
    print(f"grad b2={fmt(g.b2)}  label from bias={label_from_bias(g)}  grad W2 row sums={fmt(g.w2.sum(axis=1))}")

    print("\n== Section 14.3: exact recovery, one 8x8 image, ReLU, m = 32 ==")
    params = init_params(rng, WIDTH)
    x, y = make_images(rng, 1)
    g = gradients(params, x, y, "relu")
    ratios, usable = unit_ratios(g)
    errs = matched_errors(x, ratios[usable])  # error of every usable unit's ratio
    sv = singular_values(g)
    print(f"units with nonzero grad b1: {int(usable.sum())} of {WIDTH}")
    print(f"largest relative error over those units: {float(errs.max()):.1e}")
    print(f"singular values of grad W1: s1={sv[0]:.4f}  s2/s1={sv[1] / sv[0]:.1e}")
    print(f"gradient entries: {g.flat().size}  input pixels: {D}")

    print("\n== Section 14.4: labels ==")
    hits_bias = hits_weights = 0
    n_single = 1000
    for _ in range(n_single):
        p = init_params(rng, WIDTH)
        xi, yi = make_images(rng, 1)
        gi = gradients(p, xi, yi, "relu")
        hits_bias += int(label_from_bias(gi) == yi[0])
        hits_weights += int(label_from_weights(gi) == yi[0])
    print(
        f"single example, {n_single} trials: bias rule {hits_bias / n_single:.3f}  "
        f"weight rule {hits_weights / n_single:.3f}"
    )
    xb, yb = make_images(rng, 16)
    gb = gradients(params, xb, yb, "relu")
    uniform = label_counts_estimate(gb, 16, np.full(K, 1.0 / K))
    aux = label_counts_estimate(gb, 16, mean_prediction(params, make_images(rng, 200)[0], "relu"))
    true_p = label_counts_estimate(gb, 16, forward(params, xb, yb, "relu").p.mean(axis=0))
    print(
        f"batch of 16: true counts {np.bincount(yb, minlength=K).tolist()}  uniform guess {fmt(uniform, 2)}  "
        f"auxiliary guess {fmt(aux, 2)}  true p {fmt(true_p, 2)}"
    )
    for b in (4, 16, 64):
        print(
            f"batch {b}: all counts exact, uniform guess {batch_label_counts_exact(rng, b, WIDTH, 200, 'uniform'):.3f}  "
            f"auxiliary guess {batch_label_counts_exact(rng, b, WIDTH, 200, 'auxiliary'):.3f}"
        )

    print("\n== Section 14.5: DLG, sigmoid, m = 32 ==")
    params = init_params(rng, WIDTH)
    x, y = make_images(rng, 1)
    target = gradients(params, x, y, "sigmoid")
    res = dlg(params, target, y, rng, steps=DLG_STEPS)
    for t in (0, 10, 50, 100, 200, 400, DLG_STEPS):
        print(f"iteration {t}: relative error {relative_error(res.path[t], x):.1e}  matching loss {res.losses[t]:.1e}")
    print(f"unknowns per example {D}; gradient equations {target.flat().size}")
    for b in (1, 2, 4, 8):
        s = dlg_batch_stats(rng, b, WIDTH, DLG_TRIALS, DLG_STEPS)
        print(
            f"B={b}: median relative error {s.median_error:.3f}  recovered (<0.05) {s.recovered:.2f}  "
            f"median matching loss {s.median_loss:.1e}"
        )

    print("\n== Section 14.6: batches, ReLU ==")
    params = init_params(rng, WIDTH)
    x, y = make_images(rng, 4)
    g = gradients(params, x, y, "relu")
    sv = singular_values(g)
    print("B=4: singular values / s1: " + " ".join(f"{v:.1e}" for v in sv[:6] / sv[0]))
    print(f"B=4: largest distance of a row of grad W1 from span(x): {float(span_residuals(g, x).max()):.1e}")
    rec = recovered_by_ratios(g, x)
    print(f"B=4: examples recovered exactly by unit ratios: {int(rec.sum())} of 4")
    for m in (64, 256, 1024):
        for b in (4, 8, 12, 16):
            print(
                f"m={m} B={b}: formula {isolation_probability(b, m):.3f}  "
                f"gaussian inputs {isolation_rate(rng, b, m, 20, 'gaussian'):.3f}  "
                f"images {isolation_rate(rng, b, m, 20, 'images'):.3f}"
            )
    m_half = math.ceil(math.log(2) / -math.log1p(-(2.0**-10)))
    print(f"width for isolation probability 1/2 at B=10: {m_half}")

    print("\n== Check of the DLG gradient, sigmoid, m = 32 ==")
    params = init_params(rng, WIDTH)
    xc, yc = make_images(rng, 2)
    target = gradients(params, xc, yc, "sigmoid")
    gap = finite_difference_gap(params, rng.uniform(0.0, 1.0, size=xc.shape), yc, target, "sigmoid")
    print(f"double backprop against finite differences, batch of 2: largest gap {gap:.1e}")


# --- Figures --------------------------------------------------------------------------------


def show_image(ax: Axes, v: Array, title: str) -> None:
    ax.imshow(v.reshape(SIDE, SIDE), cmap="gray_r", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_rank_one(rng: np.random.Generator) -> Figure:
    """The input, the gradient of the first layer (every row a multiple of it), and one row ratio."""
    params = init_params(rng, WIDTH)
    x, y = make_images(rng, 1)
    g = gradients(params, x, y, "relu")
    fig = plt.figure(figsize=(9.0, 3.2))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 2.6, 0.25, 1.0])
    show_image(fig.add_subplot(grid[0]), x[0], "client's input $x$")
    ax = fig.add_subplot(grid[1])
    lim = float(np.abs(g.w1).max())
    ax.imshow(g.w1, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    ax.set_title(r"$\nabla_{W_1}\ell$: row $j$ is $\gamma_j\,x^\top$", fontsize=9)
    ax.set_xlabel("pixel (64)")
    ax.set_ylabel("hidden unit $j$ (32)")
    ax_b = fig.add_subplot(grid[2])
    ax_b.imshow(g.b1[:, None], cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    ax_b.set_title(r"$\nabla_{b_1}\ell$", fontsize=9)
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    j = int(np.argmax(np.abs(g.b1)))
    show_image(fig.add_subplot(grid[3]), g.w1[j] / g.b1[j], rf"row {j} / $\nabla_{{b_1}}\ell$ entry {j}")
    return fig


def plot_labels(rng: np.random.Generator) -> Figure:
    """Last-layer bias gradient for one example (one negative entry) and label counts for a batch."""
    params = init_params(rng, WIDTH)
    x, y = make_images(rng, 1)
    g = gradients(params, x, y, "relu")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax = axes[0]
    colors = ["C3" if k == y[0] else "C0" for k in range(K)]
    ax.bar(range(K), g.b2, color=colors)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(range(K), CLASS_NAMES)
    ax.set_ylabel(r"$\nabla_{b_2}\ell = p - e_y$")
    ax.set_title(f"one example (label: {CLASS_NAMES[int(y[0])]}, in red)", fontsize=10)
    batch = 16
    xb, yb = make_images(rng, batch)
    gb = gradients(params, xb, yb, "relu")
    aux = label_counts_estimate(gb, batch, mean_prediction(params, make_images(rng, 200)[0], "relu"))
    uniform = label_counts_estimate(gb, batch, np.full(K, 1.0 / K))
    ax = axes[1]
    pos = np.arange(K)
    ax.bar(pos - 0.27, np.bincount(yb, minlength=K), width=0.27, label="true count", color="C7")
    ax.bar(pos, aux, width=0.27, label="estimate, average $p$ from own data", color="C2")
    ax.bar(pos + 0.27, uniform, width=0.27, label="estimate, $p = 1/K$", color="C1")
    ax.set_xticks(pos, CLASS_NAMES)
    ax.set_ylabel("examples in the batch")
    ax.set_title(f"a batch of {batch}: $B(\\bar p - \\nabla_{{b_2}}\\ell)$", fontsize=10)
    ax.set_ylim(0, 1.5 * max(float(np.bincount(yb, minlength=K).max()), float(uniform.max()), float(aux.max())))
    ax.legend(fontsize=8, loc="upper left")
    return fig


def plot_dlg_progress(rng: np.random.Generator) -> Figure:
    """DLG reconstruction of one image over the iterations, with the loss and error curves."""
    params = init_params(rng, WIDTH)
    x, y = make_images(rng, 1)
    res = dlg(params, gradients(params, x, y, "sigmoid"), y, rng, steps=DLG_STEPS)
    keep = (0, 10, 25, 50, 100, DLG_STEPS)
    fig = plt.figure(figsize=(9.0, 5.0))
    grid = fig.add_gridspec(2, len(keep) + 1, height_ratios=[1.0, 1.6])
    for i, t in enumerate(keep):
        show_image(fig.add_subplot(grid[0, i]), np.clip(res.path[t][0], 0.0, 1.0), f"iteration {t}")
    show_image(fig.add_subplot(grid[0, len(keep)]), x[0], "true input")
    ax = fig.add_subplot(grid[1, :])
    its = np.arange(len(res.losses))
    ax.semilogy(its, res.losses, label=r"matching loss $\|\nabla_\theta\ell(\hat x) - g\|^2$")
    ax.semilogy(its, [relative_error(v, x) for v in res.path], label=r"relative error $\|\hat x - x\| / \|x\|$")
    ax.set_xlabel("iteration")
    ax.set_ylim(1e-33, 1e2)
    ax.legend(fontsize=9)
    return fig


def plot_dlg_batch(rng: np.random.Generator, trials: int = 3) -> Figure:
    """DLG error against batch size, and a batch of four reconstructed side by side with the truth."""
    batches = [1, 2, 3, 4, 6, 8]
    fig = plt.figure(figsize=(9.0, 4.0))
    grid = fig.add_gridspec(2, 5, width_ratios=[3.0, 1.0, 1.0, 1.0, 1.0])
    ax = fig.add_subplot(grid[:, 0])
    medians: list[float] = []
    for b in batches:
        errs: list[float] = []
        for _ in range(trials):
            params = init_params(rng, WIDTH)
            x, y = make_images(rng, b)
            res = dlg(params, gradients(params, x, y, "sigmoid"), y, rng, steps=DLG_STEPS)
            errs.extend(matched_errors(res.x_hat, x).tolist())
        ax.scatter(
            np.full(len(errs), b) + rng.uniform(-0.15, 0.15, len(errs)),
            np.maximum(errs, 1e-17),
            s=10,
            alpha=0.5,
            color="C0",
        )
        medians.append(float(np.median(errs)))
    ax.semilogy(batches, np.maximum(medians, 1e-17), "o-", color="C3", label="median over examples")
    ax.axhline(0.05, color="gray", ls=":", lw=1, label="relative error 0.05")
    ax.set_xlabel("batch size $B$")
    ax.set_ylabel("relative error of the closest reconstruction")
    ax.legend(fontsize=8, loc="lower right")
    params = init_params(rng, WIDTH)
    x, y = make_images(rng, 4)
    res = dlg(params, gradients(params, x, y, "sigmoid"), y, rng, steps=DLG_STEPS)
    for i in range(4):
        j, err = closest(res.x_hat, x[i])
        show_image(fig.add_subplot(grid[0, 1 + i]), x[i], "true" if i == 0 else "")
        show_image(fig.add_subplot(grid[1, 1 + i]), np.clip(res.x_hat[j], 0.0, 1.0), f"error {err:.2f}")
    return fig


def plot_isolation(rng: np.random.Generator, trials: int = 10) -> Figure:
    """Fraction of a batch recovered exactly by unit ratios, against the independence formula."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bs = np.arange(1, 21)
    for m, color in ((64, "C0"), (256, "C1"), (1024, "C2")):
        ax.plot(bs, [isolation_probability(int(b), m) for b in bs], color=color, label=f"$m = {m}$: formula")
        pts = bs[1::2]
        ax.plot(
            pts,
            [isolation_rate(rng, int(b), m, trials, "gaussian") for b in pts],
            "o",
            color=color,
            ms=5,
            label="Gaussian inputs" if m == 64 else None,
        )
        ax.plot(
            pts,
            [isolation_rate(rng, int(b), m, trials, "images") for b in pts],
            "s",
            color=color,
            mfc="none",
            ms=6,
            label="images" if m == 64 else None,
        )
    ax.set_xlabel("batch size $B$")
    ax.set_ylabel("fraction of the batch recovered exactly")
    ax.set_xticks(range(2, 21, 2))
    ax.legend(fontsize=8, loc="lower left")
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
        "gradinv_rank_one.svg": plot_rank_one(rng),
        "gradinv_labels.svg": plot_labels(rng),
        "gradinv_dlg_progress.svg": plot_dlg_progress(rng),
        "gradinv_dlg_batch.svg": plot_dlg_batch(rng),
        "gradinv_isolation.svg": plot_isolation(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
