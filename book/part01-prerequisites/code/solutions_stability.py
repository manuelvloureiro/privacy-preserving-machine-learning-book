"""Compute the numbers quoted in the solutions to the stability exercises.

    uv run python book/part01-prerequisites/code/solutions_stability.py
    uv run python book/part01-prerequisites/code/solutions_stability.py --seed 1

Prints every number in solutions/07-stability-generalization.md (it matches exactly with the
default seed). The exercises ask for no figures, so --out is accepted but unused.
"""

import argparse
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from stability import (
    Array,
    D,
    fit_logistic,
    logistic_losses,
    mean_and_se,
    renaming_expectations,
    sample_data,
    shrunk_mean_stability,
    sigmoid,
    sphere_quadrature,
)

FIGURES = Path(__file__).resolve().parent.parent / "figures"


# --- Exercise 2: the renaming table for every p -----------------------------------------------


def renaming_gap(p: float) -> float:
    """E[test] - E[train] for the toy algorithm A(S) = z_1 / 2, by enumerating the four outcomes."""
    train, test, _ = renaming_expectations(p)
    return test - train


def renaming_gap_formula(p: float) -> float:
    """The closed form p(1 - p) derived in the solution."""
    return p * (1 - p)


def renaming_gap_argmax(grid: int = 1000) -> tuple[float, float]:
    """The p in {0, 1/grid, ..., 1} with the largest enumerated gap, and that gap."""
    best = max(range(grid + 1), key=lambda k: renaming_gap(k / grid))
    return best / grid, renaming_gap(best / grid)


# --- Exercises 3 and 4: the sample mean on coin flips, enumerated exactly ---------------------


def mean_square_loss(w: float, z: float) -> float:
    return (w - z) ** 2


def sample_mean_stability(n: int) -> float:
    """sup |ℓ(A(S), z) - ℓ(A(S^(i)), z)| for the sample mean of n points in [0, 1] with squared loss.

    The loss difference is |w - w'| |w + w' - 2z|, linear in each coordinate of S, z_i, z_i' and z
    separately, so the supremum over the box is attained at a corner: every z_j in {0, 1}.
    By symmetry only the number of other ones matters.
    """
    best = 0.0
    for ones in range(n):  # number of ones among the n - 1 unchanged points
        for zi, zi_new, z in product((0, 1), repeat=3):
            w, w2 = (ones + zi) / n, (ones + zi_new) / n
            best = max(best, abs(mean_square_loss(w, z) - mean_square_loss(w2, z)))
    return best


@dataclass(frozen=True)
class CoinMean:
    train: float  # E[(1/n) Σ (z̄ - z_i)²]
    test: float  # E[(z̄ - z)²], z a fresh coin
    loo: float  # E[(1/n) Σ (z̄_{-i} - z_i)²], the leave-one-out estimate
    test_smaller: float  # E[(z̄' - z)²] for a training set of size n - 1


def coin_mean(n: int, p: float = 0.5) -> CoinMean:
    """Exact expectations for the sample mean of n Bernoulli(p) coins, summing over the count of ones."""
    train = test = loo = test_smaller = 0.0
    for k in range(n + 1):
        prob = math.comb(n, k) * p**k * (1 - p) ** (n - k)
        w = k / n
        train += prob * (k * mean_square_loss(w, 1) + (n - k) * mean_square_loss(w, 0)) / n
        test += prob * (p * mean_square_loss(w, 1) + (1 - p) * mean_square_loss(w, 0))
        # leaving out a one gives mean (k - 1)/(n - 1), evaluated at 1; leaving out a zero gives k/(n - 1) at 0
        loo_sum = 0.0
        if k > 0:
            loo_sum += k * mean_square_loss((k - 1) / (n - 1), 1)
        if k < n:
            loo_sum += (n - k) * mean_square_loss(k / (n - 1), 0)
        loo += prob * loo_sum / n
    for k in range(n):
        prob = math.comb(n - 1, k) * p**k * (1 - p) ** (n - 1 - k)
        w = k / (n - 1)
        test_smaller += prob * (p * mean_square_loss(w, 1) + (1 - p) * mean_square_loss(w, 0))
    return CoinMean(train, test, loo, test_smaller)


# --- Exercise 5: SGD on a strongly convex loss ------------------------------------------------


def contraction(eta: float, rho: float, mu: float) -> float:
    """The factor 1 - η ρ μ / (ρ + μ) of a gradient step on a μ-strongly convex, ρ-smooth function."""
    return 1 - eta * rho * mu / (rho + mu)


def strongly_convex_divergence_bound(lipschitz: float, n: int, eta: float, rho: float, mu: float, steps: int) -> float:
    """Unrolled bound (2ηL/n) Σ_{k<T} q^k on E||w_T - w_T'||."""
    q = contraction(eta, rho, mu)
    return 2 * eta * lipschitz / n * (1 - q**steps) / (1 - q)


def strongly_convex_limit(lipschitz: float, n: int, rho: float, mu: float) -> float:
    """The bound for every T: (2L/n)(ρ + μ)/(ρ μ)."""
    return 2 * lipschitz / n * (rho + mu) / (rho * mu)


def sgd_regularized_divergence(
    rng: np.random.Generator, n: int, eta: float, lam: float, epochs: list[int], trials: int
) -> list[float]:
    """Mean ||w_T - w_T'|| after each listed epoch, for coupled SGD runs on S and S^(1).

    Each step uses the per-example loss ln(1 + exp(-y<w, x>)) + (λ/2)||w||², which is λ-strongly
    convex and (1/4 + λ)-smooth. Both runs share the index sequence; z_1 is replaced by a fresh draw.
    """
    xs = np.empty((trials, n, D))
    ys = np.empty((trials, n))
    for t in range(trials):
        xs[t], ys[t] = sample_data(rng, n)
    xs2, ys2 = xs.copy(), ys.copy()
    xs2[:, 0], ys2[:, 0] = sample_data(rng, trials)
    w = np.zeros((trials, D))
    w2 = np.zeros((trials, D))
    rows = np.arange(trials)
    out: list[float] = []
    for step in range(1, max(epochs) * n + 1):
        idx = rng.integers(n, size=trials)
        for ww, xx, yy in ((w, xs, ys), (w2, xs2, ys2)):
            xi, yi = xx[rows, idx], yy[rows, idx]
            s = sigmoid(-yi * np.einsum("td,td->t", xi, ww))
            ww += eta * ((yi * s)[:, None] * xi - lam * ww)
        if step % n == 0 and step // n in epochs:
            out.append(float(np.mean(np.linalg.norm(w - w2, axis=1))))
    return out


# --- Exercise 6: k-nearest neighbours ---------------------------------------------------------


def knn_predict(x_train: Array, y_train: Array, x_query: Array, k: int) -> Array:
    """Majority vote of the k nearest training labels (k odd, so no ties). Unit vectors: nearest = largest <., .>."""
    sims = x_query @ x_train.T
    nearest = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    return np.sign(np.sum(y_train[nearest], axis=1))


def knn_errors(
    rng: np.random.Generator, n: int, ks: tuple[int, ...], trials: int, n_test: int = 2000
) -> dict[int, tuple[float, float]]:
    """Average train and test 0-1 error of k-NN for each k, on the same training and test sets.

    A training point counts as its own neighbour, as it does when the trained rule is applied to it.
    """
    xt, yt = sample_data(rng, n_test)
    sums = {k: [0.0, 0.0] for k in ks}
    for _ in range(trials):
        x, y = sample_data(rng, n)
        for k in ks:
            train_wrong: Array = (knn_predict(x, y, x, k) != y).astype(np.float64)
            test_wrong: Array = (knn_predict(x, y, xt, k) != yt).astype(np.float64)
            sums[k][0] += float(np.mean(train_wrong))
            sums[k][1] += float(np.mean(test_wrong))
    return {k: (v[0] / trials, v[1] / trials) for k, v in sums.items()}


def knn_worst_case_flip(k: int) -> tuple[float, float]:
    """The 0-1 loss at a test point before and after one replacement, in the construction of the solution.

    The k training points nearest to the test point x = e_1 carry (k + 1)/2 labels +1 and (k - 1)/2
    labels -1; the test label is +1. Replacing one +1 neighbour by the same point labelled -1 flips the vote.
    """
    angles = np.linspace(0.01, 0.2, k)  # k points near e_1, all closer than the far points below
    near: Array = np.column_stack([np.cos(angles), np.sin(angles), np.zeros((k, D - 2))])
    far: Array = np.tile(-np.eye(D)[0], (20, 1))
    x_train = np.vstack([near, far])
    y_train = np.concatenate([np.ones((k + 1) // 2), -np.ones((k - 1) // 2), np.ones(20)])
    query = np.eye(D)[:1]
    before = float(knn_predict(x_train, y_train, query, k)[0] != 1.0)
    y_replaced = y_train.copy()
    y_replaced[0] = -1.0  # z_i' = (x_i, -y_i)
    after = float(knn_predict(x_train, y_replaced, query, k)[0] != 1.0)
    return before, after


# --- Exercise 7: the decaying step size -------------------------------------------------------


def harmonic_step_sum(c: float, steps: int) -> float:
    return c * sum(1 / t for t in range(1, steps + 1))


def feldman_vondrak_2018_rate(beta: float, n: int, delta: float) -> float:
    """√((β + 1/n) ln(1/δ)), the order of the 2018 bound, constants dropped."""
    return math.sqrt((beta + 1 / n) * math.log(1 / delta))


def bkz_rate(beta: float, n: int, delta: float) -> float:
    """β ln n ln(1/δ) + √(ln(1/δ)/n), the order of the 2019/2020 bounds, constants dropped."""
    return beta * math.log(n) * math.log(1 / delta) + math.sqrt(math.log(1 / delta) / n)


def be_width(beta: float, n: int, loss_max: float, delta: float) -> float:
    """Width of the Bousquet–Elisseeff bound, @eq-stability-be."""
    return beta + (2 * n * beta + loss_max) * math.sqrt(math.log(1 / delta) / (2 * n))


# --- Exercise 8: self-influence with less label noise -----------------------------------------


def population_risk_signal(w: Array, signal: float) -> float:
    """R(w) for the running example with P(y = 1 | x) = sigmoid(signal x_1), by quadrature."""
    a, b, weight = sphere_quadrature(len(w))
    m = w[0] * a + float(np.linalg.norm(w[1:])) * b
    p = 1 / (1 + np.exp(-signal * a))
    return float((p * np.logaddexp(0.0, -m) + (1 - p) * np.logaddexp(0.0, m)) @ weight)


def against_trend_probability(signal: float) -> float:
    """P(y x_1 < 0) = E[sigmoid(-signal |x_1|)], which is also the Bayes error."""
    a, _, weight = sphere_quadrature()
    return float((1 / (1 + np.exp(signal * np.abs(a)))) @ weight)


@dataclass(frozen=True)
class SelfInfluenceSummary:
    gap: tuple[float, float]  # mean and standard error over data sets
    influence: tuple[float, float]
    against: float  # mean self-influence of examples with y_i x_i1 < 0
    along: float
    share_against: float
    train: float
    test: float


def self_influence_signal(
    rng: np.random.Generator, n: int, lam: float, trials: int, signal: float
) -> SelfInfluenceSummary:
    """The self-influence experiment of the chapter, with the label probability sigmoid(signal x_1)."""
    gaps, means, trains, tests = np.empty(trials), np.empty(trials), np.empty(trials), np.empty(trials)
    influence: list[Array] = []
    agreement: list[Array] = []
    for t in range(trials):
        x, y = sample_data(rng, n, signal=signal)
        xf, yf = sample_data(rng, n, signal=signal)
        w = fit_logistic(x, y, lam)
        own = logistic_losses(w, x, y)
        trains[t], tests[t] = float(np.mean(own)), population_risk_signal(w, signal)
        gaps[t] = tests[t] - trains[t]
        values = np.empty(n)
        for i in range(n):
            x2, y2 = x.copy(), y.copy()
            x2[i], y2[i] = xf[i], yf[i]
            w2 = fit_logistic(x2, y2, lam, w0=w)
            values[i] = float(logistic_losses(w2, x[i : i + 1], y[i : i + 1])[0]) - float(own[i])
        means[t] = float(np.mean(values))
        influence.append(values)
        agreement.append(y * x[:, 0])
    infl, agr = np.concatenate(influence), np.concatenate(agreement)
    return SelfInfluenceSummary(
        mean_and_se(gaps),
        mean_and_se(means),
        float(np.mean(infl[agr < 0])),
        float(np.mean(infl[agr >= 0])),
        float(np.mean(agr < 0)),
        float(np.mean(trains)),
        float(np.mean(tests)),
    )


# --- The solutions' table ---------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        train, test, swapped = renaming_expectations(p)
        print(
            f"[Ex2 p={p}] E train={train:.4f}  E test={test:.4f}  E renamed={swapped:.4f}  "
            f"gap={renaming_gap(p):.4f}  p(1-p)={renaming_gap_formula(p):.4f}"
        )
    p_best, g_best = renaming_gap_argmax()
    print(f"[Ex2] argmax p={p_best:.3f}  max gap={g_best:.4f}  beta={shrunk_mean_stability():.2f}")

    n = 10
    c = coin_mean(n)
    print(
        f"[Ex3 n={n} fair coins] train={c.train:.4f}  test={c.test:.4f}  gap={c.test - c.train:.4f}  "
        f"2sigma^2/n={2 * 0.25 / n:.4f}  bound 2/n={2 / n:.4f}  exact sup={sample_mean_stability(n):.4f}"
    )
    c2 = coin_mean(2)
    print(f"[Ex3 n=2 fair coins] train={c2.train:.4f}  test={c2.test:.4f}  gap={c2.test - c2.train:.4f}")
    print(
        f"[Ex4 n={n} fair coins] LOO={c.loo:.5f}  test(n-1)={c.test_smaller:.5f}  "
        f"test(n)={c.test:.5f}  train={c.train:.5f}"
    )

    n, eta, lam, lip = 100, 0.25, 0.1, 2.0
    rho, mu = 0.25 + lam, lam
    epochs = [1, 5, 20, 40, 100]
    divergence = sgd_regularized_divergence(rng, n, eta, lam, epochs, trials=100)
    print(
        f"[Ex5 n={n} eta={eta} lam={lam} L={lip} rho={rho} mu={mu}] q={contraction(eta, rho, mu):.5f}  "
        f"limit={strongly_convex_limit(lip, n, rho, mu):.4f}"
    )
    for e, div in zip(epochs, divergence):
        print(
            f"[Ex5 epoch={e}] measured={div:.4f}  strongly convex bound="
            f"{strongly_convex_divergence_bound(lip, n, eta, rho, mu, e * n):.4f}  convex bound={2 * eta * lip * e:.1f}"
        )

    for n in (200, 1000):
        errs = knn_errors(rng, n, (1, 15), trials=10)
        (tr1, te1), (tr15, te15) = errs[1], errs[15]
        print(
            f"[Ex6 n={n}] 1-NN train={tr1:.3f} test={te1:.3f} gap={te1 - tr1:.3f}  "
            f"15-NN train={tr15:.3f} test={te15:.3f} gap={te15 - tr15:.3f}"
        )
    before, after = knn_worst_case_flip(15)
    print(f"[Ex6 worst case k=15] loss before={before:.0f}  after={after:.0f}")

    for nn in (10**4,):
        hs = harmonic_step_sum(1.0, nn)
        print(
            f"[Ex7 c=1 T=n={nn}] sum eta={hs:.4f}  1+ln T={1 + math.log(nn):.4f}  "
            f"HRS={2 * hs / nn:.6f}  bound={2 * (1 + math.log(nn)) / nn:.6f}"
        )
    delta = 0.05
    for nn in (10**3, 10**4, 10**6):
        beta = 2 / math.sqrt(nn)
        print(
            f"[Ex7 n={nn} beta=2/sqrt(n)={beta:.4f}] BE width={be_width(beta, nn, 1.0, delta):.3f}  "
            f"FV2018 rate={feldman_vondrak_2018_rate(beta, nn, delta):.3f}  "
            f"BKZ rate={bkz_rate(beta, nn, delta):.3f}"
        )
    print(f"[Ex7] limit of the stability term 2*sqrt(2 ln(1/delta))={2 * math.sqrt(2 * math.log(1 / delta)):.3f}")

    for signal in (5.0, 20.0):
        print(f"[Ex8 signal={signal:g}] P(against trend)=Bayes error={against_trend_probability(signal):.3f}")
    s = self_influence_signal(rng, 40, 0.01, trials=100, signal=20.0)
    print(
        f"[Ex8 signal=20 n=40 lam=0.01] gap={s.gap[0]:.4f}±{s.gap[1]:.4f}  "
        f"mean influence={s.influence[0]:.4f}±{s.influence[1]:.4f}  against trend={s.against:.3f}  "
        f"with trend={s.along:.3f}  share against={s.share_against:.3f}  train={s.train:.4f}  test={s.test:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="unused: these solutions draw no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
