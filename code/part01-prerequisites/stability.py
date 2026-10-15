"""Check the stability bounds in the stability chapter and draw its figures.

    uv run python code/part01-prerequisites/stability.py
    uv run python code/part01-prerequisites/stability.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 07-stability-generalization.md (they match exactly with the
default seed), then saves the chapter's figures.

The running example is L2-regularized logistic regression on unit-norm features, so the
logistic loss is 1-Lipschitz in the weights (L = 1) and 1/4-smooth.
"""

import argparse
import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "stability"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]

D = 10  # feature dimension of the running example
SIGNAL = 5.0  # norm of the true weight vector: P(y = 1 | x) = sigmoid(SIGNAL * x_1)


# --- The renaming trick on a one-example data set ----------------------------------------------


@dataclass(frozen=True)
class RenamingRow:
    z1: int
    z1_fresh: int
    prob: float
    train: float  # ℓ(A(S), z_1)
    test: float  # ℓ(A(S), z_1')
    swapped: float  # ℓ(A(S^(1)), z_1)


def shrunk_mean(z: int) -> float:
    """The toy algorithm A(S) = z_1 / 2 for a one-example data set S = (z_1)."""
    return z / 2


def squared_loss(w: float, z: int) -> float:
    return (w - z) ** 2


def renaming_rows(p: float) -> list[RenamingRow]:
    """Every outcome (z_1, z_1') of two independent Bernoulli(p) draws, with the three losses."""
    rows: list[RenamingRow] = []
    for z1 in (0, 1):
        for z1_fresh in (0, 1):
            prob = (p if z1 else 1 - p) * (p if z1_fresh else 1 - p)
            rows.append(
                RenamingRow(
                    z1,
                    z1_fresh,
                    prob,
                    train=squared_loss(shrunk_mean(z1), z1),
                    test=squared_loss(shrunk_mean(z1), z1_fresh),
                    swapped=squared_loss(shrunk_mean(z1_fresh), z1),
                )
            )
    return rows


def renaming_expectations(p: float) -> tuple[float, float, float]:
    """E[train loss], E[test loss], E[ℓ(A(S^(1)), z_1)] for the toy algorithm."""
    rows = renaming_rows(p)
    return (
        sum(r.prob * r.train for r in rows),
        sum(r.prob * r.test for r in rows),
        sum(r.prob * r.swapped for r in rows),
    )


def shrunk_mean_stability() -> float:
    """sup over z_1, z_1', z of |ℓ(A(z_1), z) - ℓ(A(z_1'), z)| for the toy algorithm."""
    return max(
        abs(squared_loss(shrunk_mean(a), z) - squared_loss(shrunk_mean(b), z))
        for a in (0, 1)
        for b in (0, 1)
        for z in (0, 1)
    )


# --- Data and regularized logistic regression ---------------------------------------------------


def sample_data(rng: np.random.Generator, n: int, d: int = D, signal: float = SIGNAL) -> tuple[Array, Array]:
    """n points with x uniform on the unit sphere and y = ±1, P(y = 1 | x) = sigmoid(signal * x_1)."""
    x = rng.standard_normal((n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    p = 1 / (1 + np.exp(-signal * x[:, 0]))
    y = np.where(rng.random(n) < p, 1.0, -1.0)
    return x, y


def sigmoid(t: Array) -> Array:
    return 0.5 * (1 + np.tanh(0.5 * t))  # overflow-free


def logistic_losses(w: Array, x: Array, y: Array) -> Array:
    """Per-example logistic loss ln(1 + exp(-y <w, x>))."""
    return np.logaddexp(0.0, -y * (x @ w))


def mean_logistic_loss(w: Array, x: Array, y: Array) -> float:
    return float(np.mean(logistic_losses(w, x, y)))


def zero_one_error(w: Array, x: Array, y: Array) -> float:
    return float(np.mean(y * (x @ w) <= 0))


def regularized_objective(w: Array, x: Array, y: Array, lam: float) -> float:
    return mean_logistic_loss(w, x, y) + 0.5 * lam * float(w @ w)


def fit_logistic(x: Array, y: Array, lam: float, w0: Array | None = None, tol: float = 1e-12) -> Array:
    """argmin_w (1/n) Σ ln(1 + exp(-y_i <w, x_i>)) + (λ/2)||w||², by damped Newton.

    The objective is λ-strongly convex, so the minimizer is unique and Newton converges.
    """
    n, d = x.shape
    w: Array = np.zeros(d) if w0 is None else w0.copy()
    eye: Array = np.eye(d)
    for _ in range(100):
        m: Array = y * (x @ w)
        s = sigmoid(-m)  # = -d loss / d margin
        grad: Array = -(x.T @ (y * s)) / n + lam * w
        if float(np.max(np.abs(grad))) < tol:
            break
        hess: Array = (x.T * (s * (1 - s))) @ x / n + lam * eye
        step: Array = np.linalg.solve(hess, grad)
        decrease, t = float(grad @ step), 1.0
        if decrease > 1e-12:  # backtrack far from the optimum; near it, rounding would stall the test
            f0 = regularized_objective(w, x, y, lam)
            while regularized_objective(w - t * step, x, y, lam) > f0 - 0.25 * t * decrease and t > 1e-10:
                t /= 2
        w = w - t * step
    return w


def erm_stability_bound(lipschitz: float, lam: float, n: int) -> float:
    """Uniform stability 2L²/(λn) of λ-regularized ERM with an L-Lipschitz convex loss."""
    return 2 * lipschitz**2 / (lam * n)


# --- Replace-one sensitivity of regularized logistic regression ---------------------------------


@dataclass(frozen=True)
class ReplaceOne:
    weight_change: float  # max_i ||w(S) - w(S^(i))||
    loss_change: float  # max_i max_z |ℓ(w(S), z) - ℓ(w(S^(i)), z)| over the probe points


def replace_one_change(rng: np.random.Generator, n: int, lam: float, k: int = 10, n_probe: int = 2000) -> ReplaceOne:
    """Fit on S, then on S^(i) for k random i, where z_i is replaced by its label-flipped copy.

    The loss change is maximized over fresh probe points and, for each i, over the two points
    ±(w - w')/||w - w'|| with both labels, which is where a 1-Lipschitz loss can change most.
    """
    x, y = sample_data(rng, n)
    xp, yp = sample_data(rng, n_probe)
    w = fit_logistic(x, y, lam)
    weight_change, loss_change = 0.0, 0.0
    for i in rng.choice(n, size=k, replace=False):
        y2 = y.copy()
        y2[i] = -y2[i]
        w2 = fit_logistic(x, y2, lam, w0=w)
        dw = w2 - w
        norm = float(np.linalg.norm(dw))
        u = dw / norm
        xs = np.vstack([xp, u, -u, u, -u])
        ys = np.concatenate([yp, [1.0, 1.0, -1.0, -1.0]])
        change = float(np.max(np.abs(logistic_losses(w, xs, ys) - logistic_losses(w2, xs, ys))))
        weight_change, loss_change = max(weight_change, norm), max(loss_change, change)
    return ReplaceOne(weight_change, loss_change)


# --- Population risk by quadrature ---------------------------------------------------------------


@cache
def sphere_quadrature(d: int = D, m: int = 80) -> tuple[Array, Array, Array]:
    """Nodes (a, b) and weights for E f(x_1, x_2) with x uniform on the unit sphere in R^d.

    (x_1, x_2) has density proportional to (1 - a² - b²)^((d - 4)/2) on the unit disk, so a
    midpoint grid on [-1, 1]² with those weights integrates any smooth f(x_1, x_2).
    """
    g = (np.arange(m) + 0.5) / m * 2 - 1
    a, b = (v.ravel() for v in np.meshgrid(g, g, indexing="ij"))
    weight = np.clip(1 - a * a - b * b, 0.0, None) ** ((d - 4) / 2)
    keep = weight > 0
    return a[keep], b[keep], weight[keep] / weight[keep].sum()


def population_risk(w: Array) -> Array:
    """R(w) = E ln(1 + exp(-y <w, x>)) for each row of w, without sampling error.

    By rotation symmetry, <w, x> has the law of w_1 x_1 + ||w_2:d|| x_2, and y only depends on x_1,
    so R is a two-dimensional integral against the true label probabilities.
    """
    w2 = np.atleast_2d(w)
    a, b, weight = sphere_quadrature(w2.shape[1])
    m = w2[:, :1] * a + np.linalg.norm(w2[:, 1:], axis=1, keepdims=True) * b
    p = 1 / (1 + np.exp(-SIGNAL * a))
    return (p * np.logaddexp(0.0, -m) + (1 - p) * np.logaddexp(0.0, m)) @ weight


# --- Generalization gap of regularized logistic regression --------------------------------------


@dataclass(frozen=True)
class Gap:
    train: float  # E[R̂_S(w_S)]
    test: float  # E[R(w_S)]
    gap: float  # E[R(w_S) - R̂_S(w_S)], estimated with a control variate (see erm_gap)


def erm_gap(rng: np.random.Generator, n: int, lam: float, trials: int) -> Gap:
    """Average train loss, population loss and gap of λ-regularized ERM over independent data sets.

    For a fixed reference model w_ref, R(w_ref) - R̂_S(w_ref) has mean exactly zero, so
    subtracting it leaves the expected gap unchanged while cancelling most of the noise
    that the draw of S puts into R̂_S. w_ref is the fit on a large independent sample.
    """
    w_ref = fit_logistic(*sample_data(rng, 20_000), lam)
    r_ref = float(population_risk(w_ref)[0])
    weights = np.empty((trials, D))
    train = np.empty(trials)
    train_ref = np.empty(trials)
    for t in range(trials):
        x, y = sample_data(rng, n)
        weights[t] = fit_logistic(x, y, lam, w0=w_ref)
        train[t], train_ref[t] = mean_logistic_loss(weights[t], x, y), mean_logistic_loss(w_ref, x, y)
    test = population_risk(weights)
    gap = (test - train) - (r_ref - train_ref)
    return Gap(float(np.mean(train)), float(np.mean(test)), float(np.mean(gap)))


# --- Self-influence: the gap is the average effect of replacing an example on its own loss -----


@dataclass(frozen=True)
class SelfInfluence:
    gap: Array  # per data set: R(w_S) - R̂_S(w_S)
    mean_influence: Array  # per data set: (1/n) Σ_i ℓ(w(S^(i)), z_i) - ℓ(w(S), z_i)
    influence: Array  # every per-example value, pooled over data sets
    agreement: Array  # y_i x_i1 for the same examples: negative means the label is the less likely one


def self_influence(rng: np.random.Generator, n: int, lam: float, trials: int) -> SelfInfluence:
    """For each data set, replace each z_i in turn by a fresh draw z_i' and refit."""
    gaps = np.empty(trials)
    means = np.empty(trials)
    influence: list[Array] = []
    agreement: list[Array] = []
    for t in range(trials):
        x, y = sample_data(rng, n)
        xf, yf = sample_data(rng, n)  # z_i' for every i
        w = fit_logistic(x, y, lam)
        own = logistic_losses(w, x, y)
        gaps[t] = float(population_risk(w)[0]) - float(np.mean(own))
        values = np.empty(n)
        for i in range(n):
            x2, y2 = x.copy(), y.copy()
            x2[i], y2[i] = xf[i], yf[i]
            w2 = fit_logistic(x2, y2, lam, w0=w)
            values[i] = float(logistic_losses(w2, x[i : i + 1], y[i : i + 1])[0]) - float(own[i])
        means[t] = float(np.mean(values))
        influence.append(values)
        agreement.append(y * x[:, 0])
    return SelfInfluence(gaps, means, np.concatenate(influence), np.concatenate(agreement))


def mean_and_se(v: Array) -> tuple[float, float]:
    return float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v)))


# --- The high-probability bound -----------------------------------------------------------------


def bousquet_elisseeff_bound(beta: float, n: int, loss_max: float, delta: float) -> float:
    """Width ε such that R ≤ R̂ + ε with probability ≥ 1 - δ, for replace-one stability β."""
    return beta + (2 * n * beta + loss_max) * math.sqrt(math.log(1 / delta) / (2 * n))


# --- SGD: coupled runs and the Hardt–Recht–Singer bound -----------------------------------------


def hrs_bound(lipschitz: float, n: int, step_sizes_sum: float) -> float:
    """Uniform stability (2L²/n) Σ η_t of SGD on a convex, smooth, L-Lipschitz loss."""
    return 2 * lipschitz**2 * step_sizes_sum / n


@dataclass(frozen=True)
class SgdRun:
    divergence: Array  # (trials, steps + 1): ||w_t - w_t'|| for the coupled runs on S and S^(1)
    train: Array  # (epochs + 1,): average train loss after each epoch, on S
    test: Array  # (epochs + 1,): average population loss after each epoch


def sgd_coupled(rng: np.random.Generator, n: int, eta: float, epochs: int, trials: int) -> SgdRun:
    """Unregularized SGD on the logistic loss, one example per step drawn uniformly at random.

    Each trial runs twice with the same index sequence: on S and on S^(1), where z_1 is
    replaced by a fresh draw. All trials run in parallel as a batch.
    """
    steps = epochs * n
    xs = np.empty((trials, n, D))
    ys = np.empty((trials, n))
    for t in range(trials):
        xs[t], ys[t] = sample_data(rng, n)
    xf, yf = sample_data(rng, trials)
    xs2, ys2 = xs.copy(), ys.copy()
    xs2[:, 0], ys2[:, 0] = xf, yf
    w = np.zeros((trials, D))
    w2 = np.zeros((trials, D))
    rows = np.arange(trials)
    divergence = np.zeros((trials, steps + 1))
    train = np.zeros(epochs + 1)
    test = np.zeros(epochs + 1)

    def record(epoch: int) -> None:
        margins: Array = ys * np.einsum("tnd,td->tn", xs, w)
        train[epoch] = float(np.mean(np.logaddexp(0.0, -margins)))
        test[epoch] = float(np.mean(population_risk(w)))

    record(0)
    for step in range(1, steps + 1):
        idx = rng.integers(n, size=trials)
        for ww, xx, yy in ((w, xs, ys), (w2, xs2, ys2)):
            xi, yi = xx[rows, idx], yy[rows, idx]
            s = sigmoid(-yi * np.einsum("td,td->t", xi, ww))
            ww += eta * (yi * s)[:, None] * xi
        divergence[:, step] = np.linalg.norm(w - w2, axis=1)
        if step % n == 0:
            record(step // n)
    return SgdRun(divergence, train, test)


# --- An unstable algorithm: 1-nearest neighbour -------------------------------------------------


def one_nn_errors(rng: np.random.Generator, n: int, trials: int, n_test: int = 2000) -> tuple[float, float]:
    """Average train and test 0-1 error of the 1-nearest-neighbour rule."""
    xt, yt = sample_data(rng, n_test)
    train, test = 0.0, 0.0
    for _ in range(trials):
        x, y = sample_data(rng, n)
        nearest = np.argmax(xt @ x.T, axis=1)  # unit vectors: largest inner product = nearest
        own = np.argmax(x @ x.T, axis=1)  # each training point is its own nearest neighbour
        test += float(np.count_nonzero(y[nearest] != yt)) / n_test
        train += float(np.count_nonzero(y[own] != y)) / n
    return train / trials, test / trials


def logistic_errors(
    rng: np.random.Generator, n: int, lam: float, trials: int, n_test: int = 2000
) -> tuple[float, float]:
    """Average train and test 0-1 error of λ-regularized logistic regression."""
    xt, yt = sample_data(rng, n_test)
    train, test = 0.0, 0.0
    for _ in range(trials):
        x, y = sample_data(rng, n)
        w = fit_logistic(x, y, lam)
        train += zero_one_error(w, x, y)
        test += zero_one_error(w, xt, yt)
    return train / trials, test / trials


def bayes_and_cover_hart() -> tuple[float, float]:
    """Bayes error E[min(p, 1 - p)] and the large-n 1-NN error E[2p(1 - p)], p = P(y = 1 | x)."""
    a, _, weight = sphere_quadrature()
    p = 1 / (1 + np.exp(-SIGNAL * a))
    return float(np.minimum(p, 1 - p) @ weight), float((2 * p * (1 - p)) @ weight)


# --- The chapter's table --------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    p = 0.3
    for r in renaming_rows(p):
        print(
            f"[Renaming p={p}] z1={r.z1} z1'={r.z1_fresh} prob={r.prob:.2f}  "
            f"train={r.train:.2f}  test={r.test:.2f}  swapped={r.swapped:.2f}"
        )
    train, test, swapped = renaming_expectations(p)
    print(
        f"[Renaming p={p}] E train={train:.4f}  E test={test:.4f}  E swapped={swapped:.4f}  "
        f"gap={test - train:.4f}  beta={shrunk_mean_stability():.2f}"
    )

    for n in (100, 1000, 10_000):
        for lam in (0.01, 0.1, 1.0):
            r = replace_one_change(rng, n, lam)
            b = erm_stability_bound(1.0, lam, n)
            print(
                f"[Replace-one n={n} lam={lam:g}] param bound={2 / (lam * n):.1e} change={r.weight_change:.2e}  "
                f"loss bound={b:.1e} change={r.loss_change:.2e}  ratio={r.loss_change / b:.2f}"
            )

    for lam in (0.01, 0.1, 1.0):
        for n in (50, 200, 1000):
            g = erm_gap(rng, n, lam, trials=100)
            print(
                f"[Gap n={n} lam={lam:g}] train={g.train:.4f}  test={g.test:.4f}  gap={g.gap:.2e}  "
                f"bound={erm_stability_bound(1.0, lam, n):.1e}  ratio={g.gap / erm_stability_bound(1.0, lam, n):.3f}"
            )

    for lam, n in ((0.1, 10_000), (1.0, 10_000)):
        loss_max = math.log1p(math.exp(1 / lam))  # ||w|| <= L/λ and ||x|| <= 1
        width = bousquet_elisseeff_bound(erm_stability_bound(1.0, lam, n), n, loss_max, 0.05)
        print(f"[High-prob n={n} lam={lam:g} delta=0.05] M={loss_max:.3f}  width={width:.3f}")

    for lam in (0.01, 0.1):
        s = self_influence(rng, 40, lam, trials=100)
        g, g_se = mean_and_se(s.gap)
        m, m_se = mean_and_se(s.mean_influence)
        against = s.influence[s.agreement < 0]
        along = s.influence[s.agreement >= 0]
        print(
            f"[Self-influence n=40 lam={lam:g}] gap={g:.4f}±{g_se:.4f}  mean influence={m:.4f}±{m_se:.4f}  "
            f"against trend={float(np.mean(against)):.3f}  with trend={float(np.mean(along)):.3f}  "
            f"share against={len(against) / len(s.influence):.2f}"
        )

    n, eta, epochs = 100, 0.25, 40
    run = sgd_coupled(rng, n, eta, epochs, trials=100)
    mean_div = np.mean(run.divergence, axis=0)
    for e in (1, 2, 5, 10, 20, 40):
        print(
            f"[SGD n={n} eta={eta} epoch={e}] train={run.train[e]:.3f}  test={run.test[e]:.3f}  "
            f"gap={run.test[e] - run.train[e]:.3f}  HRS bound={hrs_bound(1.0, n, eta * e * n):.1f}  "
            f"divergence={float(mean_div[e * n]):.3f}  bound={2 * eta * e * n / n:.1f}"
        )

    bayes, cover_hart = bayes_and_cover_hart()
    print(f"[Bayes error={bayes:.3f}  1-NN limit={cover_hart:.3f}]")
    for n in (50, 200, 1000, 4000):
        nn_train, nn_test = one_nn_errors(rng, n, trials=10)
        lr_train, lr_test = logistic_errors(rng, n, 0.01, trials=10)
        print(
            f"[0-1 error n={n}] 1-NN train={nn_train:.3f} test={nn_test:.3f}  "
            f"logistic train={lr_train:.3f} test={lr_test:.3f}"
        )


# --- Figures --------------------------------------------------------------------------------


def plot_neighbours(rng: np.random.Generator, n: int = 30) -> Figure:
    """1-D logistic regression on S and on S^(i), where one example's label is flipped."""
    x = rng.uniform(-1, 1, size=(n, 1))
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-SIGNAL * x[:, 0])), 1.0, -1.0)
    i = int(np.argmax(np.where(y > 0, x[:, 0], -np.inf)))  # the most typical positive example
    y2 = y.copy()
    y2[i] = -y2[i]
    grid = np.linspace(-1, 1, 201)[:, None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, lam in zip(axes, (0.01, 1.0)):
        w, w2 = fit_logistic(x, y, lam), fit_logistic(x, y2, lam)
        ax.plot(grid[:, 0], sigmoid(grid @ w), label=f"trained on S (w = {float(w[0]):.2f})")
        ax.plot(grid[:, 0], sigmoid(grid @ w2), "--", label=f"trained on S$^{{(i)}}$ (w = {float(w2[0]):.2f})")
        others = np.arange(n) != i
        ax.plot(x[others, 0], (y[others] + 1) / 2, "k.", alpha=0.5)
        ax.plot([x[i, 0]], [1.0], "o", color="C0", markersize=9, label="$z_i$ (label +1)")
        ax.plot([x[i, 0]], [0.0], "s", color="C1", markersize=9, label="$z_i'$ (label flipped)")
        bound = 2 / (lam * n)
        ax.set(xlabel="x", title=f"λ = {lam:g}: |w - w'| = {float(abs(w2[0] - w[0])):.3f}, bound 2L/(λn) = {bound:.3g}")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("predicted P(y = +1 | x)")
    return fig


def plot_erm_bound(rng: np.random.Generator) -> Figure:
    ns = [30, 100, 300, 1000, 3000, 10_000]
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, lam in enumerate((0.01, 0.1, 1.0)):
        changes = [replace_one_change(rng, n, lam).loss_change for n in ns]
        ax.plot(ns, changes, "o-", color=f"C{c}", label=f"measured, λ = {lam:g}")
        ax.plot(
            ns,
            [erm_stability_bound(1.0, lam, n) for n in ns],
            "--",
            color=f"C{c}",
            label=f"$2L^2/(\\lambda n)$, λ = {lam:g}",
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="training set size n",
        ylabel="largest loss change",
        title="Replace-one loss change, regularized logistic regression",
    )
    ax.legend(fontsize=8, ncol=2)
    return fig


def plot_gap(rng: np.random.Generator) -> Figure:
    ns = [25, 50, 100, 200, 400, 1000]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for c, lam in enumerate((0.01, 0.1, 1.0)):
        gaps = [erm_gap(rng, n, lam, trials=40).gap for n in ns]
        left.plot(ns, gaps, "o-", color=f"C{c}", label=f"expected gap, λ = {lam:g}")
        left.plot(ns, [erm_stability_bound(1.0, lam, n) for n in ns], "--", color=f"C{c}", label=f"bound, λ = {lam:g}")
    left.set(
        xscale="log",
        yscale="log",
        ylim=(1e-5, 20),
        xlabel="training set size n",
        ylabel="E[test loss - train loss]",
        title="The gap falls like 1/n, below the stability bound",
    )
    left.legend(fontsize=7, ncol=2, loc="lower left")
    lams = [10 ** (k / 2) for k in range(-6, 3)]
    fits = [erm_gap(rng, 50, lam, trials=40) for lam in lams]
    right.plot(lams, [g.train for g in fits], "o-", label="train loss")
    right.plot(lams, [g.test for g in fits], "o-", label="test (population) loss")
    right.set(
        xscale="log", xlabel="regularization λ", ylabel="logistic loss", title="n = 50: the gap closes as λ grows"
    )
    right.legend()
    return fig


def plot_high_probability(n_max: int = 10**6, loss_max: float = 1.0, delta: float = 0.05) -> Figure:
    ns = [int(10 ** (k / 4)) for k in range(4, 4 * int(math.log10(n_max)) + 1)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, power in ((r"$\beta = 1/n$", 1.0), (r"$\beta = 1/n^{3/4}$", 0.75), (r"$\beta = 1/\sqrt{n}$", 0.5)):
        ax.plot(ns, [bousquet_elisseeff_bound(float(n) ** -power, n, loss_max, delta) for n in ns], label=label)
    ax.plot(
        ns,
        [loss_max * math.sqrt(math.log(1 / delta) / (2 * n)) for n in ns],
        "k--",
        label=r"one fixed model: $M\sqrt{\ln(1/\delta)/(2n)}$",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="training set size n",
        ylabel=f"width at δ = {delta}",
        title="High-probability width of the McDiarmid bound (M = 1)",
    )
    ax.legend(fontsize=8)
    return fig


def plot_sgd_divergence(rng: np.random.Generator, n: int = 100, eta: float = 0.25, epochs: int = 5) -> Figure:
    run = sgd_coupled(rng, n, eta, epochs, trials=200)
    steps = np.arange(epochs * n + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    for t in range(3):
        ax.plot(steps, run.divergence[t], color="gray", linewidth=0.8, label="single runs" if t == 0 else None)
    ax.plot(steps, np.mean(run.divergence, axis=0), "k", linewidth=2, label="average over 200 runs")
    ax.plot(steps, [2 * eta * float(s) / n for s in steps], "C3--", label=r"bound $(2L/n)\sum_t \eta_t$")
    for e in range(1, epochs + 1):
        ax.axvline(e * n, color="gray", linestyle=":", linewidth=0.6)
    ax.set(
        xlabel="SGD step t",
        ylabel=r"$\|w_t - w_t'\|$",
        ylim=(0, 2 * eta * epochs * 1.05),
        title=f"Coupled SGD on S and S$^{{(1)}}$, n = {n}, η = {eta}",
    )
    ax.legend(loc="upper left")
    return fig


def plot_sgd_epochs(rng: np.random.Generator, n: int = 100, eta: float = 0.25, epochs: int = 40) -> Figure:
    run = sgd_coupled(rng, n, eta, epochs, trials=50)
    es = np.arange(epochs + 1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(es, run.train, label="train loss")
    left.plot(es, run.test, label="test (population) loss")
    best = int(np.argmin(run.test))
    left.axvline(best, color="gray", linestyle=":", label=f"lowest test loss (epoch {best})")
    left.set(xlabel="epochs", ylabel="logistic loss", title="Training longer overfits")
    left.legend()
    right.plot(es[1:], (run.test - run.train)[1:], "o-", markersize=3, label="measured gap")
    right.plot(es[1:], [hrs_bound(1.0, n, eta * float(e) * n) for e in es[1:]], "--", label=r"bound $2L^2\eta T/n$")
    right.set(
        yscale="log",
        xlabel="epochs",
        ylabel="test loss - train loss",
        title=f"Gap and stability bound, n = {n}, η = {eta}",
    )
    right.legend()
    return fig


def plot_nearest_neighbour(rng: np.random.Generator) -> Figure:
    ns = [25, 50, 100, 200, 500, 1000, 2000, 4000]
    nn = [one_nn_errors(rng, n, trials=5) for n in ns]
    lr = [logistic_errors(rng, n, 0.01, trials=5) for n in ns]
    bayes, cover_hart = bayes_and_cover_hart()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, [e[1] for e in nn], "o-", color="C0", label="1-NN test")
    ax.plot(ns, [e[0] for e in nn], "o--", color="C0", label="1-NN train")
    ax.plot(ns, [e[1] for e in lr], "s-", color="C1", label="logistic (λ = 0.01) test")
    ax.plot(ns, [e[0] for e in lr], "s--", color="C1", label="logistic (λ = 0.01) train")
    ax.axhline(cover_hart, color="C0", linestyle=":", label=f"1-NN limit E[2p(1-p)] = {cover_hart:.3f}")
    ax.axhline(bayes, color="k", linestyle=":", label=f"Bayes error = {bayes:.3f}")
    ax.set(
        xscale="log",
        ylim=(-0.02, 0.65),
        xlabel="training set size n",
        ylabel="0-1 error",
        title="A memorizer's gap does not close",
    )
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    return fig


def plot_self_influence(rng: np.random.Generator, n: int = 40) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, lam in zip(axes, (0.01, 0.1)):
        s = self_influence(rng, n, lam, trials=30)
        against = s.agreement < 0
        ax.plot(s.agreement[~against], s.influence[~against], ".", alpha=0.4, label="label agrees with the trend")
        ax.plot(s.agreement[against], s.influence[against], ".", alpha=0.4, label="label against the trend")
        ax.axhline(
            float(np.mean(s.influence)), color="k", linestyle="--", label=f"average = {float(np.mean(s.influence)):.3f}"
        )
        ax.set(xlabel=r"$y_i\, x_{i1}$ (how typical the label is)", title=f"λ = {lam:g}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"self-influence $\ell(w(S^{(i)}), z_i) - \ell(w(S), z_i)$")
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
        "stability_neighbours.svg": plot_neighbours(rng),
        "stability_erm_bound.svg": plot_erm_bound(rng),
        "stability_gap.svg": plot_gap(rng),
        "stability_high_probability.svg": plot_high_probability(),
        "stability_sgd_divergence.svg": plot_sgd_divergence(rng),
        "stability_sgd_epochs.svg": plot_sgd_epochs(rng),
        "stability_nearest_neighbour.svg": plot_nearest_neighbour(rng),
        "stability_self_influence.svg": plot_self_influence(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
