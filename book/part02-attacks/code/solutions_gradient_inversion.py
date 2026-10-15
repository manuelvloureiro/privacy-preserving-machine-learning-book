"""Check the worked solutions to the exercises of the gradient inversion chapter.

    uv run python book/part02-attacks/code/solutions_gradient_inversion.py

Prints every number quoted in solutions/06-gradient-inversion.md (seed 0). Makes no figures;
--out is accepted and ignored.

Model and data as in gradient_inversion.py: a one-hidden-layer network on 8×8 images of four shapes.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from gradient_inversion import (
    WIDTH,
    Array,
    Grads,
    K,
    Params,
    gradients,
    init_params,
    label_counts_estimate,
    make_images,
    matched_errors,
    mean_prediction,
    relative_error,
    softmax,
)
from numpy.typing import NDArray

FIGURES = Path(__file__).resolve().parent.parent / "figures"


# --- Exercise 1: logistic regression with a bias ----------------------------------------------


def logistic_gradient(w: Array, b: float, x: Array, y: int) -> tuple[Array, float]:
    """Gradient of ln(1 + exp(-y(w·x + b))) in (w, b), labels y ∈ {-1, +1}."""
    margin = y * (float(w @ x) + b)
    coef = -y / (1.0 + math.exp(margin))  # -y σ(-margin)
    return coef * x, coef


# --- Exercise 2: no bias in the first layer ---------------------------------------------------


def gradients_no_bias(params: Params, x: Array, y: NDArray[np.int64]) -> Grads:
    """Gradients of a ReLU network whose first layer has no bias (b1 = 0, its gradient is not sent)."""
    zero = Params(w1=params.w1, b1=np.zeros_like(params.b1), w2=params.w2, b2=params.b2)
    g = gradients(zero, x, y, "relu")
    return Grads(w1=g.w1, b1=np.zeros_like(g.b1), w2=g.w2, b2=g.b2)


def top_right_singular_vector(m: Array) -> Array:
    _, _, vt = np.linalg.svd(m)
    return vt[0]


def scale_mismatch(params: Params, u: Array, c: float, y: NDArray[np.int64], target: Grads) -> float:
    g = gradients_no_bias(params, c * u[None, :], y)
    return float(np.sum((g.flat() - target.flat()) ** 2))


def recover_scale(params: Params, u: Array, y: NDArray[np.int64], target: Grads) -> float:
    """Grid search over c ∈ [-30, 30], then golden-section refinement around the best grid point."""
    grid = np.linspace(-30.0, 30.0, 6001)
    vals = [scale_mismatch(params, u, float(c), y, target) for c in grid]
    i = int(np.argmin(vals))
    lo, hi = float(grid[max(i - 1, 0)]), float(grid[min(i + 1, len(grid) - 1)])
    phi = (math.sqrt(5) - 1) / 2
    for _ in range(200):
        a = hi - phi * (hi - lo)
        b = lo + phi * (hi - lo)
        if scale_mismatch(params, u, a, y, target) < scale_mismatch(params, u, b, y, target):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2


# --- Exercise 3: labels with a tanh hidden layer ---------------------------------------------


def tanh_last_layer_grad(params: Params, x: Array, y: int) -> Array:
    """∇W2 for one example with h = tanh(W1 x + b1): (p - e_y) hᵀ."""
    h: Array = np.tanh(params.w1 @ x + params.b1)
    p: Array = softmax((params.w2 @ h + params.b2)[None, :])[0]
    p[y] -= 1.0
    return p[:, None] * h[None, :]


def odd_one_out(row_sums: Array) -> int:
    """The row whose sign differs from all the others (K ≥ 3)."""
    signs = np.sign(row_sums)
    majority = 1.0 if np.sum(signs > 0) > np.sum(signs < 0) else -1.0
    return int(np.argmax(signs != majority))


# --- Exercise 5: isolation with correlated inputs --------------------------------------------


def isolation_per_unit_uniform(batch: int) -> float:
    """∫_0^1 q (1 - q)^{B-1} dq = 1/(B(B+1)): a unit fires on a uniform fraction q of the population."""
    return 1.0 / (batch * (batch + 1))


def isolation_per_unit_measured(rng: np.random.Generator, batch: int, width: int, trials: int) -> float:
    """Fraction of (unit, example) pairs where the unit fires on that example only, on images."""
    hits = 0
    for _ in range(trials):
        params = init_params(rng, width)
        x, _ = make_images(rng, batch)
        active: NDArray[np.bool_] = (x @ params.w1.T + params.b1) > 0  # (B, m)
        per_unit: NDArray[np.int64] = active.sum(axis=0)
        hits += int(np.count_nonzero(per_unit == 1))
    return hits / (trials * width * batch)


def width_for_half(per_unit: float) -> int:
    return math.ceil(math.log(2) / -math.log1p(-per_unit))


# --- Exercise 6: several local steps ----------------------------------------------------------


def local_update(params: Params, x: Array, y: NDArray[np.int64], steps: int, lr: float) -> Grads:
    """θ_0 - θ_T after T steps of gradient descent on the client's own data (the update it sends)."""
    p = params
    for _ in range(steps):
        g = gradients(p, x, y, "relu")
        p = Params(w1=p.w1 - lr * g.w1, b1=p.b1 - lr * g.b1, w2=p.w2 - lr * g.w2, b2=p.b2 - lr * g.b2)
    return Grads(w1=params.w1 - p.w1, b1=params.b1 - p.b1, w2=params.w2 - p.w2, b2=params.b2 - p.b2)


# --- The page's numbers -----------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Exercise 1 ==")
    gw, gb = logistic_gradient(np.array([1.0, -1.0]), 0.0, np.array([2.0, 1.0]), 1)
    print(f"grad w=({gw[0]:.4f}, {gw[1]:.4f})  grad b={gb:.4f}  ratio=({gw[0] / gb:.4f}, {gw[1] / gb:.4f})")

    print("\n== Exercise 2 ==")
    params = init_params(rng, WIDTH)
    x, y = make_images(rng, 1)
    target = gradients_no_bias(params, x, y)
    u = top_right_singular_vector(target.w1)
    x0: Array = x[0]
    cos = abs(float(u @ x0)) / float(np.sqrt(np.sum(x0**2)))
    print(f"|cos(top singular vector, x)| = {cos:.6f}")
    c = recover_scale(params, u, y, target)
    print(
        f"recovered scale c={c:.4f}  ||x||={float(np.sqrt(np.sum(x0**2))):.4f}  "
        f"relative error {relative_error(c * u, x0):.1e}"
    )
    print(f"mismatch at -c: {scale_mismatch(params, u, -c, y, target):.2e}")

    print("\n== Exercise 3 ==")
    trials = 1000
    argmin_hits = odd_hits = 0
    for _ in range(trials):
        p = init_params(rng, WIDTH)
        xi, yi = make_images(rng, 1)
        sums = tanh_last_layer_grad(p, xi[0], int(yi[0])).sum(axis=1)
        argmin_hits += int(int(np.argmin(sums)) == yi[0])
        odd_hits += int(odd_one_out(sums) == yi[0])
    print(f"tanh, {trials} trials: argmin rule {argmin_hits / trials:.3f}  odd-one-out rule {odd_hits / trials:.3f}")

    print("\n== Exercise 4 ==")
    batch, n_batches = 16, 200
    err_uniform: list[float] = []
    err_aux: list[float] = []
    for _ in range(n_batches):
        p = init_params(rng, WIDTH)
        xb, yb = make_images(rng, batch)
        g = gradients(p, xb, yb, "relu")
        counts = np.bincount(yb, minlength=K)
        uni = label_counts_estimate(g, batch, np.full(K, 1.0 / K))
        aux = label_counts_estimate(g, batch, mean_prediction(p, make_images(rng, 200)[0], "relu"))
        err_uniform.append(float(np.max(np.abs(uni - counts))))
        err_aux.append(float(np.max(np.abs(aux - counts))))
    print(
        f"B=16, {n_batches} batches: largest count error, median: uniform {np.median(err_uniform):.3f}  "
        f"auxiliary {np.median(err_aux):.3f}"
    )
    print(
        f"share of batches with every error below 1/2: uniform {np.mean(np.array(err_uniform) < 0.5):.3f}  "
        f"auxiliary {np.mean(np.array(err_aux) < 0.5):.3f}"
    )

    print("\n== Exercise 5 ==")
    for b in (4, 10, 16):
        print(
            f"B={b}: independent 2^-B={2.0**-b:.2e}  uniform fraction 1/(B(B+1))={isolation_per_unit_uniform(b):.2e}  "
            f"measured on images {isolation_per_unit_measured(rng, b, 1024, 20):.2e}"
        )
    print(
        f"width for probability 1/2 at B=10: independent {width_for_half(2.0**-10)}  "
        f"uniform fraction {width_for_half(isolation_per_unit_uniform(10))}"
    )

    print("\n== Exercise 6 ==")
    params = init_params(rng, WIDTH)
    x, y = make_images(rng, 1)
    upd = local_update(params, x, y, steps=5, lr=0.5)
    j = int(np.argmax(np.abs(upd.b1)))
    print(
        f"5 local steps, lr 0.5: relative error of update row ratio {relative_error(upd.w1[j] / upd.b1[j], x[0]):.1e}"
    )
    sv = np.linalg.svd(upd.w1, compute_uv=False)
    print(f"singular values of the update: s2/s1={sv[1] / sv[0]:.1e}")
    for steps in (1, 5):
        hits = 0
        for _ in range(200):
            p = init_params(rng, WIDTH)
            xb, yb = make_images(rng, 2)
            upd = local_update(p, xb, yb, steps=steps, lr=0.5)
            keep = np.abs(upd.b1) > 1e-12
            hits += int(np.sum(matched_errors(upd.w1[keep] / upd.b1[keep][:, None], xb) < 1e-6))
        print(f"batch of 2, {steps} local step(s), 200 batches: examples recovered exactly {hits / 400:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="accepted for a uniform interface; no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
