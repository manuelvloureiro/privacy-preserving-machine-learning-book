"""Check the derivations in the cryptography appendix and draw its figures.

    uv run python code/appendix-derivations/apx_cryptography.py
    uv run python code/appendix-derivations/apx_cryptography.py --seed 1 --out /tmp/figures

Prints the numbers quoted in book-full/appendix-derivations/09-cryptography.md (they match exactly
with the default seed), then saves the appendix's figures.

Two derivations are checked:

- the GGM tree: a deliberately broken toy PRG, an adaptive three-query adversary, the hybrids H_j,
  the q-sample distinguishers B_j that hand out their challenge strings lazily, and the single-sample
  distinguisher whose advantage is exactly the adversary's divided by n*q;
- the search-to-decision reduction for LWE: re-randomizing a sample with a guess, and recovering a toy
  secret with a simulated decision oracle.

Every scheme here is a toy with insecure parameters, and NumPy's generator is not a cryptographic source.
"""

import argparse
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure
from shared.stats import normal_cdf

configure("apx_cryptography")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

# --- The GGM tree: a toy PRG and the hybrids ------------------------------------------------
#
# Labels are lam-bit integers. The toy PRG is G(s) = (s, f(s)) for a fixed random permutation f:
# its right half is a public function of its left half, so it is broken on purpose, which makes
# every advantage large enough to measure. Inputs are n-bit integers, read from the top bit down:
# x_1 is the top bit, so the node reached after i bits is x >> (n - i).

N_BITS = 3  # input length n
LAM = 8  # label length lambda
Q = 3  # query budget q


def toy_prg_table(rng: np.random.Generator, lam: int) -> IntArray:
    """The permutation f in the toy PRG G(s) = (s, f(s))."""
    return rng.permutation(2**lam).astype(np.int64)


def prg_half(f: IntArray, labels: IntArray, bit: IntArray) -> IntArray:
    """G_bit(label): the left half is the label itself, the right half is f(label)."""
    return np.where(bit == 0, labels, f[labels])


def prefix(x: IntArray, depth: int, n: int) -> IntArray:
    """The node at the given depth on the path of input x: its first `depth` bits."""
    return x >> (n - depth)


def bit_at(x: IntArray, i: int, n: int) -> IntArray:
    """The i-th bit x_i of the input, counting from 1 at the top."""
    return (x >> (n - i)) & 1


def descend(f: IntArray, labels: IntArray, x: IntArray, depth: int, n: int) -> IntArray:
    """Walk from the node at `depth` down to the leaf with G, choosing halves by x_{depth+1}, ..., x_n."""
    for i in range(depth + 1, n + 1):
        labels = prg_half(f, labels, bit_at(x, i, n))
    return labels


class Oracle(Protocol):
    def query(self, x: IntArray) -> IntArray: ...


class HybridOracle:
    """H_j: every node at depth j gets an independent uniform label, drawn in advance; G below it."""

    def __init__(self, rng: np.random.Generator, f: IntArray, j: int, n: int, lam: int, trials: int) -> None:
        self.f, self.j, self.n = f, j, n
        self.table = rng.integers(0, 2**lam, size=(trials, 2**j))
        self.rows = np.arange(trials)

    def query(self, x: IntArray) -> IntArray:
        labels = self.table[self.rows, prefix(x, self.j, self.n)]
        return descend(self.f, labels, x, self.j, self.n)


class ReductionOracle:
    """B_j's simulation: the c-th distinct depth-(j-1) node a query passes through gets challenge pair c."""

    def __init__(self, f: IntArray, j: int, n: int, challenges: IntArray) -> None:
        trials, q, _ = challenges.shape
        self.f, self.j, self.n, self.q = f, j, n, q
        self.challenges = challenges  # shape (trials, q, 2): q pairs (left half, right half) per run
        self.ticket = np.full((trials, 2 ** (j - 1)), -1, dtype=np.int64)  # node -> index of its pair
        self.used = np.zeros(trials, dtype=np.int64)  # how many pairs have been handed out
        self.rows = np.arange(trials)

    def query(self, x: IntArray) -> IntArray:
        parent = prefix(x, self.j - 1, self.n)
        current = self.ticket[self.rows, parent]
        new = current < 0
        current = np.where(new, self.used, current)
        assert int(current.max()) < self.q, "more distinct parents than challenge strings"
        self.ticket[self.rows, parent] = current
        self.used = self.used + new
        pair = self.challenges[self.rows, current]  # shape (trials, 2)
        labels = pair[self.rows, bit_at(x, self.j, self.n)]  # the half chosen by x_j
        return descend(self.f, labels, x, self.j, self.n)


def adversary(rng: np.random.Generator, f: IntArray, oracle: Oracle, trials: int, lam: int) -> BoolArray:
    """An adaptive three-query adversary; True means "random function".

    It asks 000, then 001, then 100 or 110 depending on the top bit of the first answer. Against the
    real F_k these satisfy y2 = f(y1), and y3 = f(y1) (for 100) or y3 = f(f(y1)) (for 110). It flips
    a coin to pick one of the two checks and says "random" when the chosen check fails.
    """
    zeros = np.zeros(trials, dtype=np.int64)
    y1 = oracle.query(zeros + 0b000)
    y2 = oracle.query(zeros + 0b001)
    top = (y1 >> (lam - 1)) & 1
    y3 = oracle.query(np.where(top == 0, 0b100, 0b110).astype(np.int64))
    check1: BoolArray = y2 == f[y1]
    check_100: BoolArray = y3 == f[y1]
    check_110: BoolArray = y3 == f[f[y1]]
    check2: BoolArray = np.where(top == 0, check_100, check_110)
    coin = rng.integers(0, 2, size=trials)
    return np.where(coin == 0, ~check1, ~check2)


def adversary_exact(lam: int) -> list[float]:
    """Pr[A says "random"] in H_0, ..., H_3, by the case analysis in the appendix."""
    miss = 1 - 2.0**-lam  # an independent uniform label fails a fixed equality with this probability
    return [0.0, 0.5 * miss, 0.5 * miss, miss]


def prg_pairs(rng: np.random.Generator, f: IntArray, shape: tuple[int, ...], lam: int) -> IntArray:
    """Samples of G(U): pairs (u, f(u)) for a uniform seed u."""
    u = rng.integers(0, 2**lam, size=shape)
    return np.stack([u, f[u]], axis=-1)


def uniform_pairs(rng: np.random.Generator, shape: tuple[int, ...], lam: int) -> IntArray:
    """Samples of U_{2 lam}: two independent uniform halves."""
    return rng.integers(0, 2**lam, size=(*shape, 2))


def rate_in_hybrid(rng: np.random.Generator, f: IntArray, j: int, trials: int) -> float:
    oracle = HybridOracle(rng, f, j, N_BITS, LAM, trials)
    return float(adversary(rng, f, oracle, trials, LAM).mean())


def rate_in_reduction(rng: np.random.Generator, f: IntArray, j: int, prg: bool, trials: int) -> float:
    """Pr[B_j = 1] when all q challenge pairs come from G(U) (prg=True) or are uniform."""
    shape = (trials, Q)
    challenges = prg_pairs(rng, f, shape, LAM) if prg else uniform_pairs(rng, shape, LAM)
    oracle = ReductionOracle(f, j, N_BITS, challenges)
    return float(adversary(rng, f, oracle, trials, LAM).mean())


def rate_single_sample(rng: np.random.Generator, f: IntArray, prg: bool, trials: int) -> float:
    """Pr[B = 1] for the one-sample distinguisher: j and i uniform, the challenge z in position i,
    uniform pairs before it and PRG pairs after it (the chapter's hybrid lemma). Each (j, i) gets
    the same number of runs, which is the uniform choice of j and i, stratified."""
    total = 0.0
    for j in range(1, N_BITS + 1):
        for i in range(1, Q + 1):
            challenges = np.concatenate(
                [
                    uniform_pairs(rng, (trials, i - 1), LAM),
                    prg_pairs(rng, f, (trials, 1), LAM) if prg else uniform_pairs(rng, (trials, 1), LAM),
                    prg_pairs(rng, f, (trials, Q - i), LAM),
                ],
                axis=1,
            )
            oracle = ReductionOracle(f, j, N_BITS, challenges)
            total += float(adversary(rng, f, oracle, trials, LAM).mean())
    return total / (N_BITS * Q)


@dataclass(frozen=True)
class GGMCheck:
    hybrid: list[float]  # Pr[A = 1] in H_0, ..., H_n
    exact: list[float]
    b_prg: list[float]  # Pr[B_j = 1] on q PRG pairs, j = 1, ..., n
    b_uniform: list[float]  # Pr[B_j = 1] on q uniform pairs
    single_prg: float
    single_uniform: float


def ggm_check(rng: np.random.Generator, trials: int) -> GGMCheck:
    f = toy_prg_table(rng, LAM)
    return GGMCheck(
        hybrid=[rate_in_hybrid(rng, f, j, trials) for j in range(N_BITS + 1)],
        exact=adversary_exact(LAM),
        b_prg=[rate_in_reduction(rng, f, j, True, trials) for j in range(1, N_BITS + 1)],
        b_uniform=[rate_in_reduction(rng, f, j, False, trials) for j in range(1, N_BITS + 1)],
        single_prg=rate_single_sample(rng, f, True, trials),
        single_uniform=rate_single_sample(rng, f, False, trials),
    )


# --- LWE: re-randomizing with a guess ---------------------------------------------------------


def centered(v: IntArray, p: int) -> IntArray:
    """Representatives of v mod p in (-p/2, p/2], so that small errors look small."""
    r = np.mod(v, p)
    return np.where(r > p // 2, r - p, r)


def rounded_normal(rng: np.random.Generator, sigma: float, size: int) -> IntArray:
    """The discretized Gaussian error: a normal with standard deviation sigma, rounded to an integer."""
    return np.rint(rng.normal(0.0, sigma, size=size)).astype(np.int64)


def lwe_samples(
    rng: np.random.Generator, s: IntArray, p: int, m: int, sigma: float
) -> tuple[IntArray, IntArray, IntArray]:
    """m LWE samples (a, b = <a, s> + e mod p), returned with their errors e for checking."""
    a = rng.integers(0, p, size=(m, s.size))
    e = rounded_normal(rng, sigma, m)
    return a, np.mod(a @ s + e, p), e


def rerandomize(
    rng: np.random.Generator, a: IntArray, b: IntArray, k: int, g: int, p: int
) -> tuple[IntArray, IntArray]:
    """Add a fresh uniform r to coordinate k of a, and r*g to b, independently for each sample."""
    r = rng.integers(0, p, size=b.size)
    a2 = a.copy()
    a2[:, k] = np.mod(a2[:, k] + r, p)
    return a2, np.mod(b + r * g, p)


def residual(a: IntArray, b: IntArray, s: IntArray, p: int) -> IntArray:
    """b - <a, s> mod p, centered."""
    return centered(b - a @ s, p)


def small_case_residuals(p: int, e: int, g: int, s_k: int) -> list[int]:
    """The residual e + r(g - s_k) mod p for r = 0, ..., p - 1: the table in the appendix's idea."""
    return [(e + r * (g - s_k)) % p for r in range(p)]


def tv_from_uniform(values: IntArray, p: int) -> float:
    """Total variation distance between the empirical distribution of values mod p and uniform on Z_p."""
    counts = np.bincount(np.mod(values, p), minlength=p) / values.size
    return float(0.5 * np.abs(counts - 1 / p).sum())


# --- LWE: search from a decision oracle --------------------------------------------------------

DecisionOracle = Callable[[IntArray, IntArray], BoolArray]


def accept_prob_lwe(sigma: float, threshold: int) -> float:
    """Pr[|e| <= threshold] for e a normal with standard deviation sigma, rounded to an integer."""
    return 2 * normal_cdf((threshold + 0.5) / sigma) - 1


def oracle_advantage(sigma: float, threshold: int, p: int, noise: float) -> float:
    """Advantage of the simulated oracle: with probability `noise` it answers a fair coin, otherwise it
    accepts iff the residual is within the threshold. On a uniform pair the residual is uniform on Z_p."""
    return (1 - noise) * (accept_prob_lwe(sigma, threshold) - (2 * threshold + 1) / p)


def make_simulated_oracle(
    rng: np.random.Generator, s: IntArray, p: int, threshold: int, noise: float
) -> DecisionOracle:
    """A stand-in for a decision-LWE algorithm, run on one pair per call. It is built with the secret,
    which a real algorithm would not have; the search below calls it only as a black box."""

    def oracle(a: IntArray, b: IntArray) -> BoolArray:
        small = np.abs(residual(a, b, s, p)) <= threshold
        coin = rng.random(b.size) < 0.5
        return np.where(rng.random(b.size) < noise, coin, small)

    return oracle


def repetitions_needed(eps: float, d: int, p: int, beta: float) -> int:
    """Runs per estimate so that all d*p + 1 estimates are within eps/4 with probability >= 1 - beta:
    Hoeffding 2 exp(-2 N (eps/4)^2) per estimate, union bound over d*p + 1 estimates."""
    return math.ceil(8 * math.log(2 * (d * p + 1) / beta) / eps**2)


@dataclass(frozen=True)
class SearchResult:
    secret: list[int]
    rates: NDArray[np.float64]  # acceptance rate for each coordinate k (rows) and guess g (columns)
    baseline: float  # acceptance rate on uniform pairs
    samples_used: int


def search_secret(
    rng: np.random.Generator,
    oracle: DecisionOracle,
    sample: Callable[[int], tuple[IntArray, IntArray]],
    d: int,
    p: int,
    eps: float,
    runs: int,
) -> SearchResult:
    """Recover s coordinate by coordinate: guess g for s_k, re-randomize fresh samples with it, and keep
    the guess whose acceptance rate differs from the uniform baseline by more than eps/2."""
    baseline = float(oracle(rng.integers(0, p, size=(runs, d)), rng.integers(0, p, size=runs)).mean())
    rates = np.zeros((d, p))
    secret: list[int] = []
    for k in range(d):
        for g in range(p):
            a, b = sample(runs)  # fresh samples for every guess, so the runs are independent
            a2, b2 = rerandomize(rng, a, b, k, g, p)
            rates[k, g] = float(oracle(a2, b2).mean())
        hits = [g for g in range(p) if abs(rates[k, g] - baseline) > eps / 2]
        secret.append(hits[0] if len(hits) == 1 else -1)  # -1 marks a failed coordinate
    return SearchResult(secret=secret, rates=rates, baseline=baseline, samples_used=d * p * runs)


# Parameters of the LWE checks.
D_LWE, P_LWE, SIGMA = 4, 97, 3.0
THRESHOLD, NOISE, BETA = 3, 0.9, 0.01


def lwe_search(rng: np.random.Generator, s: IntArray) -> tuple[SearchResult, float, int]:
    eps = oracle_advantage(SIGMA, THRESHOLD, P_LWE, NOISE)
    runs = repetitions_needed(eps, D_LWE, P_LWE, BETA)
    oracle = make_simulated_oracle(rng, s, P_LWE, THRESHOLD, NOISE)

    def sample(m: int) -> tuple[IntArray, IntArray]:
        a, b, _ = lwe_samples(rng, s, P_LWE, m, SIGMA)
        return a, b

    return search_secret(rng, oracle, sample, D_LWE, P_LWE, eps, runs), eps, runs


# --- Table ------------------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    trials = 200_000
    c = ggm_check(rng, trials)
    nq = N_BITS * Q
    print(f"[GGM toy] n={N_BITS} lambda={LAM} q={Q} trials={trials:,}")
    for j in range(N_BITS + 1):
        print(f"[GGM H_{j}] Pr[A=1] simulated={c.hybrid[j]:.4f}  exact={c.exact[j]:.4f}")
    for j in range(1, N_BITS + 1):
        print(
            f"[GGM B_{j}] on PRG pairs={c.b_prg[j - 1]:.4f} (H_{j - 1}={c.hybrid[j - 1]:.4f})  "
            f"on uniform pairs={c.b_uniform[j - 1]:.4f} (H_{j}={c.hybrid[j]:.4f})"
        )
    adv_a = c.exact[-1] - c.exact[0]
    gap = c.single_uniform - c.single_prg
    print(
        f"[GGM single-sample B] Pr[B=1] on G(U)={c.single_prg:.4f}  on uniform={c.single_uniform:.4f}  "
        f"advantage={gap:.4f}  predicted Adv(A)/(nq)={adv_a / nq:.4f}  nq*advantage={nq * gap:.3f}  "
        f"Adv(A)={adv_a:.4f}"
    )

    p_small, s_small, e_small = 5, 3, 1
    for g in (3, 1):
        print(
            f"[LWE small p={p_small} s_1={s_small} e={e_small} g={g}] residuals over r=0..4: "
            f"{small_case_residuals(p_small, e_small, g, s_small)}"
        )

    s = rng.integers(0, P_LWE, size=D_LWE)
    m = 20_000
    a, b, _ = lwe_samples(rng, s, P_LWE, m, SIGMA)
    right = residual(*rerandomize(rng, a, b, 0, int(s[0]), P_LWE), s, P_LWE)
    wrong = residual(*rerandomize(rng, a, b, 0, int((s[0] + 1) % P_LWE), P_LWE), s, P_LWE)
    print(
        f"[LWE re-randomize d={D_LWE} p={P_LWE} sigma={SIGMA:g} m={m:,}] Pr[|residual|<={THRESHOLD}]: "
        f"correct guess={float(np.mean(np.abs(right) <= THRESHOLD)):.3f} (exact {accept_prob_lwe(SIGMA, THRESHOLD):.3f})  "
        f"wrong guess={float(np.mean(np.abs(wrong) <= THRESHOLD)):.3f} (exact {(2 * THRESHOLD + 1) / P_LWE:.3f})"
    )

    result, eps, runs = lwe_search(rng, s)
    off = np.array(
        [[abs(result.rates[k, g] - result.baseline) for g in range(P_LWE) if g != s[k]] for k in range(D_LWE)]
    )
    on = [abs(result.rates[k, int(s[k])] - result.baseline) for k in range(D_LWE)]
    print(
        f"[LWE search] secret={[int(v) for v in s]}  recovered={result.secret}  "
        f"Pr[accept LWE]={accept_prob_lwe(SIGMA, THRESHOLD):.4f}  eps={eps:.4f}  runs per guess N={runs:,}  "
        f"oracle calls={(D_LWE * P_LWE + 1) * runs:,}  samples={result.samples_used:,}"
    )
    print(
        f"[LWE search gaps] threshold eps/2={eps / 2:.4f}  smallest correct gap={min(on):.4f}  "
        f"largest wrong gap={float(off.max()):.4f}  baseline={result.baseline:.4f}"
    )

    successes = 0
    repeats = 10
    for _ in range(repeats):
        s_new = rng.integers(0, P_LWE, size=D_LWE)
        res, _, _ = lwe_search(rng, s_new)
        successes += res.secret == [int(v) for v in s_new]
    print(f"[LWE search repeats] {successes}/{repeats} random secrets recovered exactly (target failure <= {BETA})")


# --- Figures --------------------------------------------------------------------------------

COLORS = {
    "unused": "#d9d9d9",
    "uniform": "#e6a532",
    "computed": "#3a78b5",
    "off": "white",
}


def node_xy(depth: int, index: int, n: int) -> tuple[float, float]:
    """Position of a tree node: leaves spread evenly, each parent centred above its children."""
    width = 2 ** (n - depth)
    return (index * width + (width - 1) / 2, -depth)


def bits(index: int, depth: int) -> str:
    return format(index, f"0{depth}b") if depth > 0 else "∅"


def draw_ggm_panel(ax: Axes, j: int, n: int, queries: list[int]) -> None:
    on_path = {(depth, x >> (n - depth)) for x in queries for depth in range(n + 1)}
    for depth in range(n):
        for index in range(2**depth):
            for child in (2 * index, 2 * index + 1):
                used = (depth + 1, child) in on_path
                x0, y0 = node_xy(depth, index, n)
                x1, y1 = node_xy(depth + 1, child, n)
                ax.plot([x0, x1], [y0, y1], color="black" if used else "#c8c8c8", lw=2.2 if used else 0.8, zorder=1)
    tickets: dict[int, int] = {}
    if j >= 1:
        for x in queries:
            tickets.setdefault(x >> (n - j + 1), len(tickets) + 1)
    for depth in range(n + 1):
        for index in range(2**depth):
            x, y = node_xy(depth, index, n)
            text_color = "black"
            if (depth, index) not in on_path:
                color, edge, text_color = COLORS["off"], "#b0b0b0", "#909090"
            elif depth < j:
                color, edge = COLORS["unused"], "#808080"
            elif depth == j:
                color, edge = COLORS["uniform"], "black"
            else:
                color, edge, text_color = COLORS["computed"], "black", "white"
            ax.add_patch(Circle((x, y), 0.4, facecolor=color, edgecolor=edge, lw=1.2, zorder=2))
            ax.text(x, y, bits(index, depth), ha="center", va="center", fontsize=8, color=text_color, zorder=3)
            if depth == j - 1 and index in tickets:
                ax.text(
                    x - 0.5,
                    y + 0.25,
                    rf"$y_{tickets[index]}$",
                    ha="right",
                    va="bottom",
                    fontsize=12,
                    color="#b03030",
                    zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
                )
    ax.set_xlim(-0.7, 2**n - 0.3)
    ax.set_ylim(-n - 0.6, 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    names = {0: "the real $F_k$", n: "a random function"}
    ax.set_title(f"$H_{j}$" + (f": {names[j]}" if j in names else ""), fontsize=12)


def plot_ggm_hybrids() -> Figure:
    queries = [0b000, 0b001, 0b100]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for j, ax in enumerate(axes.flat):
        draw_ggm_panel(ax, j, N_BITS, queries)
    handles = [
        Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor=COLORS["uniform"], markeredgecolor="black"),
        Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor=COLORS["computed"], markeredgecolor="black"),
        Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor=COLORS["unused"], markeredgecolor="#808080"),
        Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor="white", markeredgecolor="#b0b0b0"),
    ]
    labels = [
        "uniform label, drawn independently",
        "computed with G from its parent",
        "on a query path, never used",
        "not on any query path",
    ]
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.suptitle(
        r"Queries 000, 001, 100. A red $y_c$ marks the node at depth $j-1$ that $\mathcal{B}_j$ gives challenge string $c$"
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.13, hspace=0.25, wspace=0.05)
    return fig


def plot_ggm_check(rng: np.random.Generator, trials: int = 50_000) -> Figure:
    c = ggm_check(rng, trials)
    js = list(range(N_BITS + 1))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(js, c.exact, "k-", lw=1, label="exact, $H_j$")
    ax.plot(js, c.hybrid, "o", color=COLORS["computed"], markersize=9, label="simulated, $H_j$")
    ax.plot(
        [j - 1 for j in range(1, N_BITS + 1)],
        c.b_prg,
        "^",
        color="#b03030",
        markersize=7,
        label=r"$\mathcal{B}_j$ on $Q$ PRG outputs",
    )
    ax.plot(
        range(1, N_BITS + 1),
        c.b_uniform,
        "s",
        color=COLORS["uniform"],
        markersize=6,
        label=r"$\mathcal{B}_j$ on $Q$ uniform strings",
    )
    ax.set(
        xticks=js,
        xticklabels=[f"$H_{j}$" for j in js],
        ylim=(-0.05, 1.08),
        xlabel="hybrid",
        ylabel='Pr[adversary says "random"]',
        title=r"Each $\mathcal{B}_j$ reproduces $H_{j-1}$ or $H_j$ exactly",
    )
    ax.legend(loc="upper left", fontsize=9)
    return fig


def plot_lwe_residuals(rng: np.random.Generator, m: int = 20_000) -> Figure:
    s = rng.integers(0, P_LWE, size=D_LWE)
    a, b, _ = lwe_samples(rng, s, P_LWE, m, SIGMA)
    right = residual(*rerandomize(rng, a, b, 0, int(s[0]), P_LWE), s, P_LWE)
    wrong = residual(*rerandomize(rng, a, b, 0, int((s[0] + 1) % P_LWE), P_LWE), s, P_LWE)
    # A composite modulus: p = 96 and a guess off by 48, which is not invertible mod 96.
    p_bad = 96
    s_bad = np.mod(s, p_bad)
    a_bad, b_bad, _ = lwe_samples(rng, s_bad, p_bad, m, SIGMA)
    bad = residual(*rerandomize(rng, a_bad, b_bad, 0, int((s_bad[0] + p_bad // 2) % p_bad), p_bad), s_bad, p_bad)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    panels = [
        (right, P_LWE, f"p = {P_LWE}, correct guess: still LWE"),
        (wrong, P_LWE, f"p = {P_LWE}, guess off by 1: uniform"),
        (bad, p_bad, f"p = {p_bad}, guess off by {p_bad // 2}: two bumps"),
    ]
    for ax, (values, p, title) in zip(axes, panels, strict=True):
        lo, hi = -(p // 2) + (1 if p % 2 == 0 else 0), p // 2
        counts = np.bincount(values - lo, minlength=hi - lo + 1) / values.size
        ax.bar(range(lo, hi + 1), counts, width=1.0, color=COLORS["computed"])
        ax.axhline(1 / p, color="k", ls="--", lw=1, label="uniform, 1/p")
        ax.set(title=title, xlabel="residual after re-randomizing (centred mod p)", xlim=(lo - 1, hi + 1))
    axes[0].set(ylabel="fraction of samples", ylim=(0, 0.16))
    axes[0].legend(loc="upper right", fontsize=9)
    return fig


def plot_lwe_search(rng: np.random.Generator) -> Figure:
    s = rng.integers(0, P_LWE, size=D_LWE)
    result, eps, runs = lwe_search(rng, s)
    fig, axes = plt.subplots(1, D_LWE, figsize=(12, 3.4), sharey=True)
    guesses = np.arange(P_LWE)
    for k, ax in enumerate(axes):
        rates = result.rates[k]
        right = guesses == s[k]
        ax.plot(guesses[~right], rates[~right], ".", color="#9a9a9a", markersize=4, label="wrong guesses")
        ax.plot(guesses[right], rates[right], "o", color="#b03030", markersize=7, label="correct guess")
        ax.axhline(result.baseline, color="k", lw=1, label="uniform pairs")
        ax.axhspan(result.baseline - eps / 2, result.baseline + eps / 2, color=COLORS["uniform"], alpha=0.25, lw=0)
        ax.set(title=f"coordinate $s_{k + 1} = {int(s[k])}$", xlabel="guess g", xlim=(-2, P_LWE + 1))
    axes[0].set(ylabel=f"acceptance rate ({runs:,} runs)", ylim=(0.40, 0.55))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(r"Only the correct guess leaves the shaded band, $\pm\nu/2$ around the uniform rate")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
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
        "apx_crypto_ggm_hybrids.svg": plot_ggm_hybrids(),
        "apx_crypto_ggm_check.svg": plot_ggm_check(rng),
        "apx_crypto_lwe_residuals.svg": plot_lwe_residuals(rng),
        "apx_crypto_lwe_search.svg": plot_lwe_search(rng),
    }
    for name, fig in figures.items():
        if name not in ("apx_crypto_ggm_hybrids.svg", "apx_crypto_lwe_search.svg"):
            fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
