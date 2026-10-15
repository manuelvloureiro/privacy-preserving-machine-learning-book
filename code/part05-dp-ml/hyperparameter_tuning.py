"""Check the numbers in the hyperparameter-tuning chapter and draw its figures.

    uv run python code/part05-dp-ml/hyperparameter_tuning.py
    uv run python code/part05-dp-ml/hyperparameter_tuning.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 03-hyperparameter-tuning.md (they match exactly with the default seed), then
saves the chapter's figures.

The chapter asks what it costs to run a private training algorithm many times and release the best run.
It computes, exactly, the privacy of "best of K runs" for a two-outcome randomized response and a
three-outcome mechanism, for a fixed K and for random K (geometric, logarithmic, truncated negative
binomial, Poisson); evaluates the Rényi bounds of Papernot & Steinke (2022) on DP-SGD's accountant; and
tunes the step size of the part's running example (dp_sgd.py) with five strategies at the same total ε.

The tuning experiments use 1,000 DP-SGD steps per run (10 epochs) instead of the running example's 10,000,
so that the script trains a few hundred models in about two minutes. Expected accuracies of a tuning
procedure are estimated by resampling from a pool of independent runs per candidate.
"""

import os

# The models are tiny; multi-threaded BLAS only adds overhead.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "hyperparameter_tuning"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from dp_sgd import (
    ORDERS,
    Dataset,
    DpsgdConfig,
    Task,
    accuracy,
    bisect,
    epsilon_from_rdp,
    init_params,
    make_task,
    rdp_curve,
    train_dpsgd,
)
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]
Pgf = Callable[[float], float]

DELTA = 1e-5
# The accounting examples use the running example of DP-SGD: q = 0.01, σ = 1, 10,000 steps (ε = 6.71).
Q_RUN, SIGMA_RUN, STEPS_RUN = 0.01, 1.0, 10_000
# The tuning experiments: 1,000 steps per run, a grid of eight step sizes, a private validation set.
STEPS_TUNE = 1_000
LR_GRID: tuple[float, ...] = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LR_GUESS = 0.5  # the step size of the DP-SGD chapter, reused without tuning
N_VAL = 5_000  # the first 5,000 of the 20,000 held-out records are the private validation set
SIGMA_VAL = 20.0  # Gaussian noise on each validation count (correct predictions out of 5,000)
PROXY_SEED = 1  # a proxy task: same construction, a different random teacher
EPS_TABLE = 2.0
EPS_FIGURE: tuple[float, ...] = (1.0, 2.0, 4.0)


# --- Toolkit: the best of K draws ------------------------------------------------------------------
# Outcomes are listed from best to worst. "Best of K" returns the best of K independent draws.


def fixed_pgf(k: int) -> Pgf:
    return lambda x: x**k


def poisson_pgf(mu: float) -> Pgf:
    return lambda x: math.exp(mu * (x - 1))


def geometric_pgf(gamma: float) -> Pgf:
    """K on {1, 2, ...} with P[K = k] = γ(1-γ)^(k-1): stop after each run with probability γ."""
    return lambda x: gamma * x / (1 - (1 - gamma) * x)


def tnb_pgf(eta: float, gamma: float) -> Pgf:
    """The truncated negative binomial D_{η,γ} of Papernot & Steinke; η = 0 is the logarithmic distribution."""
    if eta == 0:
        return lambda x: math.log(1 - (1 - gamma) * x) / math.log(gamma)
    return lambda x: ((1 - (1 - gamma) * x) ** (-eta) - 1) / (gamma ** (-eta) - 1)


def tnb_mean(eta: float, gamma: float) -> float:
    if eta == 0:
        return (1 / gamma - 1) / math.log(1 / gamma)
    return eta * (1 - gamma) / (gamma * (1 - gamma**eta))


def tnb_gamma(eta: float, mean: float) -> float:
    """The γ for which D_{η,γ} has the given mean (the mean falls as γ grows)."""
    return bisect(lambda g: tnb_mean(eta, g) <= mean, 1e-15, 1 - 1e-12, 200)


def tnb_pmf(eta: float, gamma: float, kmax: int) -> list[float]:
    """P[K = k] for k = 1..kmax."""
    out: list[float] = []
    for k in range(1, kmax + 1):
        if eta == 0:
            out.append((1 - gamma) ** k / (k * math.log(1 / gamma)))
        else:
            prod = math.exp(sum(math.log((ell + eta) / (ell + 1)) for ell in range(k)))
            out.append((1 - gamma) ** k / (gamma ** (-eta) - 1) * prod)
    return out


def best_of_pmf(pmf: Sequence[float], pgf: Pgf) -> list[float]:
    """Distribution of the best of K draws: P[best = y_i] = f(P[draw no better than y_i]) - f(P[draw worse])."""
    out: list[float] = []
    for i in range(len(pmf)):
        out.append(pgf(sum(pmf[i:])) - pgf(sum(pmf[i + 1 :])))
    return out


def die_best_of(k_values: Sequence[int]) -> float:
    """P[the best of K rolls of a die is a 6], K uniform on k_values."""
    pmf = [1 / 6] * 6
    return float(np.mean([best_of_pmf(pmf, fixed_pgf(k))[0] for k in k_values]))


# --- Exact privacy of discrete mechanisms --------------------------------------------------------


def max_loss(p: Sequence[float], q: Sequence[float]) -> float:
    """The pure-DP ε of a pair: the largest |ln p_i/q_i| over outcomes."""
    return max(abs(math.log(a / b)) for a, b in zip(p, q, strict=True))


def renyi(p: Sequence[float], q: Sequence[float], alpha: float) -> float:
    """D_α(p ‖ q) for discrete distributions."""
    return math.log(sum(a**alpha * b ** (1 - alpha) for a, b in zip(p, q, strict=True))) / (alpha - 1)


def rr_pair(eps: float) -> tuple[list[float], list[float]]:
    """Randomized response on two outcomes, best first: (1, e^ε)/(1 + e^ε) on D and (e^ε, 1)/(1 + e^ε) on D'."""
    z = 1 + math.exp(eps)
    return [1 / z, math.exp(eps) / z], [math.exp(eps) / z, 1 / z]


def three_outcome_pair(eps: float, a: float, b: float) -> tuple[list[float], list[float]]:
    """An ε-DP pair on (top, middle, bottom) built to make the middle outcome of a geometric best-of costly:
    on D the top is rare and the middle likely; on D' the top is e^ε more likely and the middle e^ε less."""
    p = [a, b, 1 - a - b]
    pp = [a * math.exp(eps), b * math.exp(-eps), 1 - a * math.exp(eps) - b * math.exp(-eps)]
    return p, pp


def best_of_loss(p: Sequence[float], pp: Sequence[float], pgf: Pgf) -> float:
    return max_loss(best_of_pmf(p, pgf), best_of_pmf(pp, pgf))


def loss_atoms(p: Sequence[float], pp: Sequence[float]) -> list[tuple[float, float]]:
    """The privacy loss ln(p_i/pp_i) of each outcome with its probability under p."""
    return [(math.log(a / b), a) for a, b in zip(p, pp, strict=True)]


def fixed_k_renyi_lower(eps: float, k: int, alpha: float) -> float:
    """Papernot & Steinke, Proposition 17: D_α of the best of k randomized responses is at least this."""
    return k * eps - k * math.log1p(math.exp(-eps)) / (alpha - 1)


# --- Liu & Talwar: random stopping -------------------------------------------------------------------


def lt_output_prob(p: float, p_better: float, gamma: float) -> float:
    """P[output y] under geometric stopping: γp / ((p_> (1-γ) + γ)(p_≥ (1-γ) + γ))."""
    p_at_least = p_better + p
    return gamma * p / ((p_better * (1 - gamma) + gamma) * (p_at_least * (1 - gamma) + gamma))


def lt_miss_prob(p: float, gamma: float) -> float:
    """P[the output is below the top p-quantile] under geometric stopping: Σ_j γ(1-γ)^(j-1)(1-p)^j."""
    return gamma * (1 - p) / (1 - (1 - gamma) * (1 - p))


# --- Papernot & Steinke: Rényi bounds for a random number of runs -------------------------------------


def monotone_in_order(curve: Sequence[float]) -> list[float]:
    """RDP at order α also holds at every smaller order: replace ε(α) by min over α' ≥ α."""
    out = list(curve)
    for i in range(len(out) - 2, -1, -1):
        out[i] = min(out[i], out[i + 1])
    return out


_ORD = np.array(ORDERS)


def delta_from_rdp(curve: Sequence[float], eps_hat: float) -> float:
    """(α, ε(α))-RDP implies (ε̂, δ̂)-DP with δ̂ = e^{(α-1)(ε(α) - ε̂)} (1 - 1/α)^(α-1) / α; best order."""
    return float(_delta_matrix(curve, np.array([eps_hat]))[0])


def _delta_matrix(curve: Sequence[float], eps_hats: Array) -> Array:
    """δ̂ for each ε̂ in eps_hats, minimized over the orders of the curve, capped at 1."""
    a = _ORD[None, :]
    x = (a - 1) * (np.array(curve)[None, :] - eps_hats[:, None]) + (a - 1) * np.log1p(-1 / a) - np.log(a)
    return np.exp(np.minimum(x.min(axis=1), 0.0))


def poisson_rdp(curve: Sequence[float], mu: float) -> list[float]:
    """Theorem 6: ε'(λ) = ε(λ) + μ δ̂ + ln μ/(λ-1), with (ε̂, δ̂)-DP at e^ε̂ = 1 + 1/(λ-1)."""
    deltas = _delta_matrix(curve, np.log1p(1 / (_ORD - 1)))
    out = np.array(curve) + mu * deltas + math.log(mu) / (_ORD - 1)
    return monotone_in_order([float(v) for v in out])


def tnb_rdp(curve: Sequence[float], eta: float, mean: float) -> list[float]:
    """Theorem 2: ε'(λ) = ε(λ) + (1+η)(1 - 1/λ̂)ε(λ̂) + (1+η)ln(1/γ)/λ̂ + ln E[K]/(λ-1), best λ̂."""
    gamma = tnb_gamma(eta, mean)
    extra = min(
        [
            (1 + eta) * (1 - 1 / ah) * eh + (1 + eta) * math.log(1 / gamma) / ah
            for ah, eh in zip(ORDERS, curve, strict=True)
        ]
        + [(1 + eta) * math.log(1 / gamma)]
    )
    return monotone_in_order(
        [e + extra + math.log(tnb_mean(eta, gamma)) / (a - 1) for a, e in zip(ORDERS, curve, strict=True)]
    )


def fixed_rdp(curve: Sequence[float], k: int) -> list[float]:
    """Composition: k runs cost k times the curve."""
    return [k * e for e in curve]


def tnb_rdp_exact_curve(eps_of: Callable[[float], float], eta: float, gamma: float, lam: float) -> float:
    """Theorem 2 at one order λ for a base whose RDP curve is known in closed form, best λ̂ on a fine grid."""
    lam_hats = [1 + x / 20 for x in range(1, 2000)]
    extra = min(
        [(1 + eta) * (1 - 1 / ah) * eps_of(ah) + (1 + eta) * math.log(1 / gamma) / ah for ah in lam_hats]
        + [(1 + eta) * math.log(1 / gamma)]
    )
    return eps_of(lam) + extra + math.log(tnb_mean(eta, gamma)) / (lam - 1)


Strategy = tuple[str, float]  # ("fixed", k), ("poisson", μ), ("tnb0", mean), ("tnb1", mean), ("tnb0.5", mean)


def tuned_rdp(curve: Sequence[float], strategy: Strategy) -> list[float]:
    kind, param = strategy
    if kind == "fixed":
        return fixed_rdp(curve, int(param))
    if kind == "poisson":
        return poisson_rdp(curve, param)
    return tnb_rdp(curve, float(kind[3:]), param)


def dpsgd_run_rdp(q: float, sigma: float, steps: int, sigma_val: float = 0.0) -> list[float]:
    """One training run's RDP curve. With a noisy validation count (noise σ_val) on a disjoint validation set,
    one record is in the training set or in the validation set, so the run's curve is the larger of the two."""
    train = [steps * e for e in rdp_curve(q, sigma)]
    if sigma_val <= 0:
        return train
    return [max(t, a / (2 * sigma_val**2)) for a, t in zip(ORDERS, train, strict=True)]


def tuned_epsilon(q: float, sigma: float, steps: int, strategy: Strategy, sigma_val: float = 0.0) -> float:
    return epsilon_from_rdp(tuned_rdp(dpsgd_run_rdp(q, sigma, steps, sigma_val), strategy), DELTA)[0]


def sigma_for_tuning(q: float, steps: int, eps: float, strategy: Strategy, sigma_val: float = 0.0) -> float:
    """The smallest noise multiplier with three significant digits for which the whole tuning procedure meets
    ε, by bisection over the three-digit values from 0.300 to 99.9."""
    grid = (
        [x / 1000 for x in range(300, 1000)] + [x / 100 for x in range(100, 1000)] + [x / 10 for x in range(100, 1000)]
    )
    lo, hi = 0, len(grid) - 1  # invariant: grid[hi] meets ε
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if tuned_epsilon(q, grid[mid], steps, strategy, sigma_val) <= eps:
            hi = mid
        else:
            lo = mid
    return grid[hi]


# --- Selection by a noisy validation score -------------------------------------------------------


def exp_mech_probs(scores: Sequence[float], eps: float, sens: float) -> list[float]:
    w = np.array(scores) * eps / (2 * sens)
    w = np.exp(w - w.max())
    return [float(x) for x in w / w.sum()]


def exp_mech_gap_bound(k: int, eps: float, sens: float) -> float:
    """Expected gap of the exponential mechanism: at most 2Δ(ln k + 1)/ε."""
    return 2 * sens * (math.log(k) + 1) / eps


# --- The tuning experiment -------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    train: Dataset
    val: Dataset
    test: Dataset


def make_split(task: Task) -> Split:
    """Train on the running example's 10,000 records; the first 5,000 held-out records are a private
    validation set and the other 15,000 the test set."""
    t = task.test
    return Split(task.train, Dataset(t.x[:N_VAL], t.y[:N_VAL]), Dataset(t.x[N_VAL:], t.y[N_VAL:]))


@dataclass(frozen=True)
class Pool:
    """Independent runs for each candidate step size: validation counts and test accuracies, (candidates, runs)."""

    val_count: Array
    test_acc: Array


def train_pool(rng: np.random.Generator, split: Split, sigma: float, runs: int) -> Pool:
    val = np.zeros((len(LR_GRID), runs))
    test = np.zeros((len(LR_GRID), runs))
    for i, lr in enumerate(LR_GRID):
        for r in range(runs):
            params, _ = train_dpsgd(rng, split.train, DpsgdConfig(sigma=sigma, lr=lr, steps=STEPS_TUNE))
            val[i, r] = round(accuracy(params, split.val) * N_VAL)
            test[i, r] = accuracy(params, split.test)
    return Pool(val, test)


@dataclass(frozen=True)
class Outcome:
    mean: float  # mean test accuracy of the released model
    sd: float  # its standard deviation over repetitions of the whole procedure
    best_lr_frac: float  # how often the released model's step size is the pool's best on average
    mean_runs: float  # average number of training runs on the private data


def simulate_selection(
    rng: np.random.Generator, pool: Pool, draw_k: Callable[[np.random.Generator], int] | None, reps: int, chance: float
) -> Outcome:
    """Repeat a tuning procedure `reps` times by resampling runs from the pool. draw_k None: the full grid,
    one run per candidate; otherwise K runs, each of a uniformly random candidate (random search). The run with
    the largest noisy validation count is released; K = 0 releases an untrained model (accuracy `chance`)."""
    n_cand: int = pool.val_count.shape[0]
    n_runs: int = pool.val_count.shape[1]
    best = int(np.argmax(pool.test_acc.mean(axis=1)))
    accs: list[float] = []
    hits = 0
    total_runs = 0
    for _ in range(reps):
        k = n_cand if draw_k is None else draw_k(rng)
        cands: NDArray[np.int64] = np.arange(n_cand) if draw_k is None else rng.integers(0, n_cand, size=k)
        total_runs += k
        if k == 0:
            accs.append(chance)
            continue
        runs = rng.integers(0, n_runs, size=k)
        noisy = pool.val_count[cands, runs] + SIGMA_VAL * rng.standard_normal(k)
        j = int(np.argmax(noisy))
        accs.append(float(pool.test_acc[cands[j], runs[j]]))
        hits += int(cands[j] == best)
    return Outcome(float(np.mean(accs)), float(np.std(accs)), hits / reps, total_runs / reps)


def simulate_proxy(rng: np.random.Generator, proxy: Pool, real: Pool, reps: int) -> Outcome:
    """Tune on public proxy data (full grid, best proxy validation count), then train once on the private data."""
    n_cand, n_runs = proxy.val_count.shape
    best = int(np.argmax(real.test_acc.mean(axis=1)))
    accs: list[float] = []
    hits = 0
    for _ in range(reps):
        runs = rng.integers(0, n_runs, size=n_cand)
        j = int(np.argmax(proxy.val_count[np.arange(n_cand), runs]))
        accs.append(float(real.test_acc[j, rng.integers(0, real.test_acc.shape[1])]))
        hits += int(j == best)
    return Outcome(float(np.mean(accs)), float(np.std(accs)), hits / reps, 1.0)


def poisson_draw(mu: float) -> Callable[[np.random.Generator], int]:
    return lambda rng: int(rng.poisson(mu))


def tnb_draw(eta: float, mean: float) -> Callable[[np.random.Generator], int]:
    gamma = tnb_gamma(eta, mean)
    pmf = np.array(tnb_pmf(eta, gamma, 2000))
    cdf = np.cumsum(pmf / pmf.sum())
    return lambda rng: int(np.searchsorted(cdf, rng.random())) + 1


STRATEGIES: tuple[tuple[str, Strategy], ...] = (
    ("grid of 8, composition", ("fixed", 8.0)),
    ("random search, Poisson, mean 8", ("poisson", 8.0)),
    ("random search, logarithmic, mean 8", ("tnb0", 8.0)),
)


@dataclass(frozen=True)
class TuningResult:
    eps: float
    sigma_single: float
    rows: list[tuple[str, float, float, Outcome]]  # (name, σ, total ε, outcome)
    pools: dict[str, Pool]  # the runs behind each row, by row name


def tuning_experiment(rng: np.random.Generator, eps: float, runs: int, reps: int) -> TuningResult:
    split = make_split(make_task())
    proxy_split = make_split(make_task(seed=PROXY_SEED))
    chance = accuracy(init_params(rng), split.test)
    sigma1 = sigma_for_tuning(Q_RUN, STEPS_TUNE, eps, ("fixed", 1.0), SIGMA_VAL)
    pool1 = train_pool(rng, split, sigma1, runs)
    proxy = train_pool(rng, proxy_split, sigma1, runs)
    eps1 = tuned_epsilon(Q_RUN, sigma1, STEPS_TUNE, ("fixed", 1.0), SIGMA_VAL)
    rows: list[tuple[str, float, float, Outcome]] = []
    pools: dict[str, Pool] = {"single": pool1, "proxy": proxy}
    guess = LR_GRID.index(LR_GUESS)
    g = pool1.test_acc[guess]
    best = int(np.argmax(pool1.test_acc.mean(axis=1)))
    rows.append(
        ("no tuning, step size 0.5", sigma1, eps1, Outcome(float(g.mean()), float(g.std()), float(guess == best), 1))
    )
    eps_naive = tuned_epsilon(Q_RUN, sigma1, STEPS_TUNE, ("fixed", 8.0), SIGMA_VAL)
    rows.append(
        ("grid of 8, single-run ε reported", sigma1, eps_naive, simulate_selection(rng, pool1, None, reps, chance))
    )
    rows.append(("proxy data, then one run", sigma1, eps1, simulate_proxy(rng, proxy, pool1, reps)))
    for name, strategy in STRATEGIES:
        sigma = sigma_for_tuning(Q_RUN, STEPS_TUNE, eps, strategy, SIGMA_VAL)
        pool = train_pool(rng, split, sigma, runs)
        pools[name] = pool
        kind, param = strategy
        draw = (
            None if kind == "fixed" else poisson_draw(param) if kind == "poisson" else tnb_draw(float(kind[3:]), param)
        )
        total = tuned_epsilon(Q_RUN, sigma, STEPS_TUNE, strategy, SIGMA_VAL)
        rows.append((name, sigma, total, simulate_selection(rng, pool, draw, reps, chance)))
    return TuningResult(eps, sigma1, rows, pools)


# --- The chapter's numbers ----------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> TuningResult:
    print("== Toolkit: the best of K draws")
    print(f"die, best of 2 rolls: P[6]={die_best_of([2]):.4f}  K uniform on {{1,2}}: P[6]={die_best_of([1, 2]):.4f}")

    print("== Best of k randomized responses")
    eps = 1.0
    p, pp = rr_pair(eps)
    print(f"eps=1: P_D=({p[0]:.4f}, {p[1]:.4f})  P_D'=({pp[0]:.4f}, {pp[1]:.4f})")
    for k in (1, 2, 3, 10):
        a, aa = best_of_pmf(p, fixed_pgf(k)), best_of_pmf(pp, fixed_pgf(k))
        atoms = loss_atoms(a, aa)
        print(
            f"k={k}: exact eps={max_loss(a, aa):.4f}  P_D[worse]={a[1]:.4f}  "
            + "  ".join(f"loss {lo:+.4f} w.p. {pr:.4f}" for lo, pr in atoms)
            + f"  D_2={renyi(a, aa, 2):.4f}  k*D_2(one run)={k * renyi(p, pp, 2):.4f}  "
            f"lower bound={fixed_k_renyi_lower(eps, k, 2):.4f}"
        )
    for name, pgf in (
        ("Poisson mean 10", poisson_pgf(10)),
        ("geometric mean 10", geometric_pgf(0.1)),
        ("logarithmic mean 10", tnb_pgf(0, tnb_gamma(0, 10))),
    ):
        a, aa = best_of_pmf(p, pgf), best_of_pmf(pp, pgf)
        atoms = loss_atoms(a, aa)
        print(
            f"{name}: exact eps={max_loss(a, aa):.4f}  "
            + "  ".join(f"loss {lo:+.4f} w.p. {pr:.4f}" for lo, pr in atoms)
        )

    print("== Composition on the running example (q=0.01, sigma=1, T=10000, delta=1e-5)")
    base = dpsgd_run_rdp(Q_RUN, SIGMA_RUN, STEPS_RUN)
    eps_run = epsilon_from_rdp(base, DELTA)[0]
    for k in (1, 3, 10, 30, 100):
        e = epsilon_from_rdp(fixed_rdp(base, k), DELTA)[0]
        s = sigma_for_tuning(Q_RUN, STEPS_RUN, eps_run, ("fixed", float(k)))
        print(f"k={k}: eps={e:.2f}  ratio to one run={e / eps_run:.2f}  sigma to keep eps={eps_run:.2f}: {s:.3g}")

    print("== Random stopping (Liu & Talwar)")
    for gamma in (0.5, 0.1, 0.01, 0.001):
        a, aa = best_of_pmf(p, geometric_pgf(gamma)), best_of_pmf(pp, geometric_pgf(gamma))
        print(f"randomized response eps=1, geometric gamma={gamma:g}: exact eps={max_loss(a, aa):.4f}")
    for gamma, a3, b3 in ((0.1, 0.01, 0.001), (0.01, 0.01, 0.0001), (1e-4, 0.01, 1e-6), (1e-6, 1e-3, 1e-9)):
        t, tt = three_outcome_pair(1.0, a3, b3)
        geo = best_of_loss(t, tt, geometric_pgf(gamma))
        lg = best_of_loss(t, tt, tnb_pgf(0, gamma))
        mid = lt_output_prob(t[1], t[0], gamma) / lt_output_prob(tt[1], tt[0], gamma)
        print(
            f"three outcomes a={a3:g} b={b3:g} gamma={gamma:g}: base eps={max_loss(t, tt):.4f}  "
            f"geometric exact eps={geo:.4f} (middle by formula {math.log(mid):.4f})  logarithmic exact eps={lg:.4f}"
        )
    for gamma, pq in ((1 / 8, 1 / 8), (1 / 80, 1 / 8), (1 / 80, 1 / 80)):
        print(
            f"utility: gamma={gamma:.4g} (mean {1 / gamma:.0f} runs), top fraction p={pq:.4g}: "
            f"P[miss]={lt_miss_prob(pq, gamma):.4f}  bound gamma/p={gamma / pq:.4f}"
        )

    print("== Random number of runs (Papernot & Steinke)")
    for eta, mean in ((0.0, 10.0), (1.0, 10.0), (0.5, 10.0)):
        gamma = tnb_gamma(eta, mean)
        pmf = tnb_pmf(eta, gamma, 1)
        print(f"eta={eta:g}, mean {mean:g}: gamma={gamma:.4g}  P[K=1]={pmf[0]:.4f}")
    print(f"Poisson mean 10: P[K=0]={math.exp(-10):.2e}  P[K<=1]={11 * math.exp(-10):.2e}")
    # Check the bounds against the exact Rényi divergence on randomized response with eps = 0.5.
    e_rr = 0.5
    p5, pp5 = rr_pair(e_rr)

    def rr_rdp(lam: float) -> float:
        return max(renyi(p5, pp5, lam), renyi(pp5, p5, lam))

    for lam in (2.0, 5.0, 20.0):
        gamma = tnb_gamma(0, 10)
        a, aa = best_of_pmf(p5, tnb_pgf(0, gamma)), best_of_pmf(pp5, tnb_pgf(0, gamma))
        exact = max(renyi(a, aa, lam), renyi(aa, a, lam))
        bound = tnb_rdp_exact_curve(rr_rdp, 0.0, gamma, lam)
        comp = 10 * rr_rdp(lam)
        print(
            f"randomized response eps=0.5, logarithmic mean 10, order {lam:g}: exact D={exact:.4f}  "
            f"Theorem 2 bound={bound:.4f}  composition of 10={comp:.4f}"
        )
    lam = 1 + 1 / math.expm1(e_rr)
    a, aa = best_of_pmf(p5, poisson_pgf(10)), best_of_pmf(pp5, poisson_pgf(10))
    exact = max(renyi(a, aa, lam), renyi(aa, a, lam))
    print(
        f"randomized response eps=0.5, Poisson mean 10, order {lam:.4f}: exact D={exact:.4f}  "
        f"Theorem 6 bound={rr_rdp(lam) + math.log(10) / (lam - 1):.4f}  composition of 10={10 * rr_rdp(lam):.4f}"
    )
    print("running example, total eps against the expected number of runs:")
    for k in (3, 10, 30, 100, 1000):
        row = [epsilon_from_rdp(tuned_rdp(base, s), DELTA)[0] for s in _strategies_for(k)]
        print(
            f"  mean {k}: composition {row[0]:.2f}  Poisson {row[1]:.2f}  geometric {row[2]:.2f}  "
            f"eta=0.5 {row[3]:.2f}  logarithmic {row[4]:.2f}"
        )

    print(f"== Tuning the running example at eps={EPS_TABLE:g} (1000 steps, 3 runs per candidate)")
    res = tuning_experiment(rng, EPS_TABLE, runs=3, reps=4000)
    for name, sigma, total, out in res.rows:
        one = epsilon_from_rdp(dpsgd_run_rdp(Q_RUN, sigma, STEPS_TUNE, SIGMA_VAL), DELTA)[0]
        print(
            f"{name}: sigma={sigma:.3g}  one run eps={one:.2f}  total eps={total:.2f}  "
            f"test accuracy {out.mean:.4f} +- {out.sd:.4f}  best step size {out.best_lr_frac:.2f}  runs {out.mean_runs:.2f}"
        )
    g8 = tnb_gamma(0, 8)
    print(
        f"logarithmic mean 8: gamma={g8:.4f}  P[K=1]={tnb_pmf(0, g8, 1)[0]:.4f}  "
        f"Poisson mean 8: P[a given step size is never tried]={math.exp(-1):.4f}  P[K=0]={math.exp(-8):.2e}"
    )

    print("== Selection by a noisy validation score")
    sigma_c = next(row[1] for row in res.rows if row[0] == "grid of 8, composition")
    pool = res.pools["grid of 8, composition"]
    counts = [float(c) for c in pool.val_count[:, 0]]
    print(
        "validation counts (of 5000): "
        + "  ".join(f"lr={lr:g}: {c:.0f}" for lr, c in zip(LR_GRID, counts, strict=True))
    )
    top = max(counts)
    for e_sel in (0.01, 0.1, 1.0):
        pr = exp_mech_probs(counts, e_sel, 1.0)
        gap = sum(w * (top - c) for w, c in zip(pr, counts, strict=True))
        print(
            f"exponential mechanism eps={e_sel:g}: P[best]={max(pr):.4f}  expected gap={gap:.2f} correct "
            f"({gap / N_VAL * 100:.3f}%)  bound={exp_mech_gap_bound(len(counts), e_sel, 1.0):.1f}"
        )
    val_rdp = [a / (2 * SIGMA_VAL**2) for a in ORDERS]
    print(f"noisy validation count, sigma_val={SIGMA_VAL:g}, alone: eps={epsilon_from_rdp(val_rdp, DELTA)[0]:.3f}")
    print(f"8 noisy counts composed: eps={epsilon_from_rdp([8 * v for v in val_rdp], DELTA)[0]:.3f}")
    tr = dpsgd_run_rdp(Q_RUN, sigma_c, STEPS_TUNE)
    print(
        f"sigma={sigma_c:.3g}, 1000 steps: one run eps={epsilon_from_rdp(tr, DELTA)[0]:.3f}  "
        f"8 runs eps={epsilon_from_rdp(fixed_rdp(tr, 8), DELTA)[0]:.3f}  "
        f"with the noisy counts (max per run) eps={tuned_epsilon(Q_RUN, sigma_c, STEPS_TUNE, ('fixed', 8.0), SIGMA_VAL):.3f}"
    )
    return res


def _strategies_for(k: int) -> list[Strategy]:
    m = float(k)
    return [("fixed", m), ("poisson", m), ("tnb1", m), ("tnb0.5", m), ("tnb0", m)]


# --- Figures --------------------------------------------------------------------------------


def plot_leak() -> Figure:
    eps = 1.0
    p, pp = rr_pair(eps)
    means = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    ax = axes[0]
    series: list[tuple[str, Callable[[float], Pgf], str, str]] = [
        ("fixed number of runs", lambda m: fixed_pgf(int(m)), "black", "o"),
        ("Poisson", lambda m: poisson_pgf(m), "tab:blue", "s"),
        ("geometric", lambda m: geometric_pgf(1 / m), "tab:orange", "^"),
        ("logarithmic", lambda m: tnb_pgf(0, tnb_gamma(0, m)), "tab:green", "D"),
    ]
    for label, make, color, marker in series:
        xs = [m for m in means if not (label != "fixed number of runs" and m == 1)]
        ys = [best_of_loss(p, pp, make(float(m))) for m in xs]
        ax.plot(xs, ys, color=color, marker=marker, ms=4, label=label)
    ax.axhline(3 * eps, color="tab:orange", ls=":", lw=1)
    ax.axhline(2 * eps, color="tab:green", ls=":", lw=1)
    ax.text(50, 3 * eps * 1.06, r"bound $3\varepsilon$ (geometric)", ha="right", fontsize=8, color="tab:orange")
    ax.text(50, 2 * eps * 1.06, r"bound $2\varepsilon$ (logarithmic)", ha="right", fontsize=8, color="tab:green")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("expected number of runs")
    ax.set_ylabel(r"exact $\varepsilon$ of the best-of release")
    ax.set_title(r"best of $K$ randomized responses, $\varepsilon = 1$")
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    width = 0.3
    for j, (label, pgf, color) in enumerate(
        (
            ("one run", fixed_pgf(1), "tab:gray"),
            ("best of 10", fixed_pgf(10), "black"),
            ("best of Poisson(10)", poisson_pgf(10), "tab:blue"),
        )
    ):
        a, aa = best_of_pmf(p, pgf), best_of_pmf(pp, pgf)
        atoms = loss_atoms(a, aa)
        for i, (lo, pr) in enumerate(atoms):
            ax.vlines(lo + (j - 1) * width / 3, 0, pr, color=color, lw=3, label=label if i == 0 else None)
            if abs(lo) > 0.5:
                ax.text(lo + (j - 1) * width / 3, pr + 0.02, f"{lo:.2f}", ha="center", fontsize=7, color=color)
    ax.set_xlabel("privacy loss ln(P_D[y] / P_D'[y]) of the released outcome y")
    ax.set_ylabel(r"probability under $D$")
    ax.set_ylim(0, 1.05)
    ax.set_title("where the loss comes from")
    ax.legend(fontsize=8, loc="upper center")
    return fig


def plot_overhead() -> Figure:
    base = dpsgd_run_rdp(Q_RUN, SIGMA_RUN, STEPS_RUN)
    eps_run = epsilon_from_rdp(base, DELTA)[0]
    means = [2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
    labels = (
        ("composition (fixed number)", "black", "o"),
        ("Poisson", "tab:blue", "s"),
        ("geometric", "tab:orange", "^"),
        (r"negative binomial, $\eta = 0.5$", "tab:purple", "v"),
        (r"logarithmic ($\eta = 0$)", "tab:green", "D"),
    )
    curves = [
        [epsilon_from_rdp(tuned_rdp(base, s), DELTA)[0] for m in means for s in [_strategies_for(m)[i]]]
        for i in range(5)
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for (label, color, marker), ys in zip(labels, curves, strict=True):
        ax.loglog(means, ys, color=color, marker=marker, ms=4, label=label)
    ax.axhline(eps_run, color="tab:gray", ls="--", label=f"one run, $\\varepsilon = {eps_run:.2f}$")
    ax.set_xlabel("expected number of training runs")
    ax.set_ylabel(r"$\varepsilon$ of the tuning procedure, $\delta = 10^{-5}$")
    ax.set_title(r"DP-SGD runs with $q = 0.01$, $\sigma = 1$, 10,000 steps each")
    ax.legend(fontsize=8)
    return fig


def plot_k_distributions() -> Figure:
    mean = 8.0
    ks = np.arange(0, 31)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    pois = [math.exp(-mean + k * math.log(mean) - math.lgamma(k + 1)) for k in ks]
    geo_g = 1 / mean
    geo = [0.0] + [geo_g * (1 - geo_g) ** (k - 1) for k in ks[1:]]
    lg = [0.0] + tnb_pmf(0, tnb_gamma(0, mean), int(ks[-1]))
    ax.plot(ks, pois, color="tab:blue", marker="s", ms=3, label="Poisson")
    ax.plot(ks, geo, color="tab:orange", marker="^", ms=3, label="geometric")
    ax.plot(ks, lg, color="tab:green", marker="D", ms=3, label="logarithmic")
    ax.vlines(8, 0, 1, color="black", lw=2, label="fixed, $k = 8$")
    ax.set_ylim(0, 0.35)
    ax.set_xlabel("number of runs $K$")
    ax.set_ylabel("probability")
    ax.set_title("Four distributions of the number of runs, each with mean 8")
    ax.legend(fontsize=8)
    return fig


def plot_accuracy(rng: np.random.Generator, at_table: TuningResult) -> Figure:
    """Accuracy against total ε. The point at the table's ε reuses the table's runs, to save time; the other
    budgets train their own (one run per candidate)."""
    results = [at_table if e == at_table.eps else tuning_experiment(rng, e, runs=1, reps=2000) for e in EPS_FIGURE]
    names = [r[0] for r in results[0].rows]
    styles = {
        "no tuning, step size 0.5": ("tab:gray", "o"),
        "proxy data, then one run": ("tab:brown", "P"),
        "grid of 8, composition": ("black", "s"),
        "random search, Poisson, mean 8": ("tab:blue", "^"),
        "random search, logarithmic, mean 8": ("tab:green", "D"),
    }
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name in names:
        accs = [next(row[3].mean for row in r.rows if row[0] == name) for r in results]
        totals = [next(row[2] for row in r.rows if row[0] == name) for r in results]
        if name == "grid of 8, single-run ε reported":
            claimed = [r.rows[0][2] for r in results]  # the one-run ε, as the no-tuning row reports it
            ax.plot(claimed, accs, color="tab:red", marker="x", ls="--", label="grid of 8, single-run ε claimed")
            ax.plot(totals, accs, color="tab:red", marker="x", label="the same runs at their true ε")
            for e, t, a in zip(claimed, totals, accs, strict=True):
                ax.annotate(
                    "", xy=(t, a), xytext=(e, a), arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 0.8}
                )
            continue
        color, marker = styles[name]
        ax.plot(totals, accs, color=color, marker=marker, label=name)
    ax.set_xscale("log")
    ticks = [1, 2, 4, 8]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_xlabel(r"total $\varepsilon$ at $\delta = 10^{-5}$ (log scale)")
    ax.set_ylabel("test accuracy of the released model")
    ax.set_title("Tuning the step size of the running example (1,000 steps per run)")
    ax.legend(fontsize=7.5, loc="lower right")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="directory for the figures")
    args = parser.parse_args()
    seed: int = args.seed
    out: Path = args.out

    at_table = print_table(np.random.default_rng(seed))

    out.mkdir(parents=True, exist_ok=True)
    fig_rng = np.random.default_rng(seed + 1)
    figures = {
        "hpt_leak.svg": plot_leak(),
        "hpt_overhead.svg": plot_overhead(),
        "hpt_k_distributions.svg": plot_k_distributions(),
        "hpt_accuracy.svg": plot_accuracy(fig_rng, at_table),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
