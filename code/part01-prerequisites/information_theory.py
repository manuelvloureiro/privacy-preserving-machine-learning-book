"""Check the formulas in the information-theory chapter and draw its figures.

    uv run python code/part01-prerequisites/information_theory.py
    uv run python code/part01-prerequisites/information_theory.py --seed 1 --out /tmp/figures

Prints every number quoted in 05-information-theory.md (they match exactly with the
default seed), then saves the chapter's figures. All logarithms are base 2 (bits).

A joint distribution is a matrix J[x][y] = Pr[X = x, Y = y], and a channel is a matrix
C[x][y] = Pr[Y = y | X = x] whose rows sum to 1. Rows index the secret or input, columns
the observation or output.
"""

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure

configure("information_theory")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


# --- Shannon entropy ------------------------------------------------------------------------


def plogp(p: float) -> float:
    """-p log2 p, with the convention 0 log 0 = 0."""
    return -p * math.log2(p) if p > 0 else 0.0


def entropy(p: Vector) -> float:
    """H(p) = -Σ p(x) log2 p(x)."""
    return sum(plogp(px) for px in p)


def binary_entropy(p: float) -> float:
    """h(p) = H(Bernoulli(p))."""
    return plogp(p) + plogp(1 - p)


def marginal_x(joint: Matrix) -> list[float]:
    return [sum(row) for row in joint]


def marginal_y(joint: Matrix) -> list[float]:
    return [sum(row[y] for row in joint) for y in range(len(joint[0]))]


def joint_entropy(joint: Matrix) -> float:
    return sum(plogp(pxy) for row in joint for pxy in row)


def entropy_given_y(joint: Matrix, y: int) -> float:
    """H(X | Y = y): the entropy of the posterior after one particular observation."""
    py = sum(row[y] for row in joint)
    return entropy([row[y] / py for row in joint])


def conditional_entropy(joint: Matrix) -> float:
    """H(X | Y) = Σ_y p(y) H(X | Y = y)."""
    return sum(py * entropy_given_y(joint, y) for y, py in enumerate(marginal_y(joint)) if py > 0)


# --- Mutual information ---------------------------------------------------------------------


def mutual_information(joint: Matrix) -> float:
    """I(X; Y) = H(X) - H(X | Y)."""
    return entropy(marginal_x(joint)) - conditional_entropy(joint)


def mutual_information_kl(joint: Matrix) -> float:
    """I(X; Y) = D(P_XY || P_X P_Y) = Σ p(x,y) log2 p(x,y) / (p(x) p(y))."""
    px, py = marginal_x(joint), marginal_y(joint)
    return sum(
        pxy * math.log2(pxy / (px[x] * py[y])) for x, row in enumerate(joint) for y, pxy in enumerate(row) if pxy > 0
    )


def joint_from(prior: Vector, channel: Matrix) -> list[list[float]]:
    """J[x][y] = π(x) C[x][y]."""
    return [[px * c for c in row] for px, row in zip(prior, channel, strict=True)]


def compose(first: Matrix, second: Matrix) -> list[list[float]]:
    """The channel X -> Y -> Z obtained by feeding the output of `first` into `second`."""
    return [[sum(row[y] * second[y][z] for y in range(len(row))) for z in range(len(second[0]))] for row in first]


def transpose(m: Matrix) -> list[list[float]]:
    return [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]


# --- Channels and capacity ------------------------------------------------------------------


def bsc(p: float) -> list[list[float]]:
    """Binary symmetric channel: the input bit is flipped with probability p."""
    return [[1 - p, p], [p, 1 - p]]


def bsc_capacity(p: float) -> float:
    return 1 - binary_entropy(p)


def bsc_information(q: float, p: float) -> float:
    """I(X; Y) for the BSC(p) with input prior Pr[X = 1] = q."""
    return mutual_information(joint_from([1 - q, q], bsc(p)))


def z_channel(p: float) -> list[list[float]]:
    """Input 0 always arrives as 0; input 1 arrives as 0 with probability p."""
    return [[1.0, 0.0], [p, 1 - p]]


def symmetric_channel(m: int, eps: float) -> list[list[float]]:
    """M-ary symmetric channel: correct with prob 1 - eps, else uniform over the other m - 1 symbols."""
    return [[1 - eps if x == y else eps / (m - 1) for y in range(m)] for x in range(m)]


def shift_channel(m: int, eps: float) -> list[list[float]]:
    """Correct with prob 1 - eps, else the next symbol x + 1 (mod m)."""
    return [[1 - eps if y == x else (eps if y == (x + 1) % m else 0.0) for y in range(m)] for x in range(m)]


def symmetric_channel_capacity(m: int, eps: float) -> float:
    """log2 m - h(eps) - eps log2(m - 1): uniform input is optimal by symmetry."""
    return math.log2(m) - binary_entropy(eps) - eps * math.log2(m - 1)


def blahut_arimoto(channel: Matrix, tol: float = 1e-12, max_iter: int = 100_000) -> tuple[float, list[float]]:
    """Capacity (bits) and an optimal input distribution, by the Blahut–Arimoto iteration.

    Stops when the standard upper bound max_x D(C_x || q) and the lower bound I(r; C)
    are within tol, so the returned value is certified to that accuracy.
    """
    c = np.asarray(channel, dtype=np.float64)
    r = np.full(c.shape[0], 1 / c.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.where(c > 0, np.log2(np.where(c > 0, c, 1.0)), 0.0)
    lower = 0.0
    for _ in range(max_iter):
        q = r @ c  # output distribution
        logq = np.log2(np.where(q > 0, q, 1.0))
        d = np.sum(c * (logc - logq), axis=1)  # D(C_x || q) for each input x, in bits
        lower = float(r @ d)  # = I(r; C)
        upper = float(np.max(d))
        if upper - lower < tol:
            break
        r = r * np.exp2(d)
        r = r / r.sum()
    return lower, [float(v) for v in r]


# --- Min-entropy and vulnerability ----------------------------------------------------------


def vulnerability(prior: Vector) -> float:
    """V(X) = max_x p(x): success probability of the best single guess."""
    return max(prior)


def min_entropy(prior: Vector) -> float:
    return -math.log2(vulnerability(prior))


def posterior_vulnerability(joint: Matrix) -> float:
    """Smith's V(X | Y) = Σ_y max_x p(x, y): success of the best guess after seeing y."""
    return sum(max(row[y] for row in joint) for y in range(len(joint[0])))


def conditional_min_entropy(joint: Matrix) -> float:
    return -math.log2(posterior_vulnerability(joint))


def averaged_min_entropy(joint: Matrix) -> float:
    """Σ_y p(y) H_∞(X | Y = y): the naive average, which is NOT Smith's definition."""
    total = 0.0
    for y, py in enumerate(marginal_y(joint)):
        if py > 0:
            total += py * -math.log2(max(row[y] for row in joint) / py)
    return total


def map_guesses(joint: Matrix) -> list[int]:
    """For each observation y, the secret value with the largest p(x, y)."""
    return [max(range(len(joint)), key=lambda x: joint[x][y]) for y in range(len(joint[0]))]


def map_error(joint: Matrix) -> float:
    """Error probability of the MAP guesser, 1 - V(X | Y)."""
    return 1 - posterior_vulnerability(joint)


def simulate_map_success(rng: np.random.Generator, prior: Vector, channel: Matrix, trials: int) -> float:
    """Draw (X, Y), let the guesser see Y and answer its MAP guess; return the success rate."""
    guesses: NDArray[np.int64] = np.asarray(map_guesses(joint_from(prior, channel)), dtype=np.int64)
    x: NDArray[np.int64] = rng.choice(len(prior), size=trials, p=np.asarray(prior, dtype=np.float64))
    u = rng.random(trials)
    cum = np.cumsum(np.asarray(channel, dtype=np.float64), axis=1)[x]
    y: NDArray[np.int64] = np.minimum((u[:, None] >= cum).sum(axis=1), len(channel[0]) - 1)
    hits: NDArray[np.bool_] = guesses[y] == x
    return float(hits.mean())


def min_entropy_leakage(prior: Vector, channel: Matrix) -> float:
    """log2 V(X | Y) / V(X)."""
    return math.log2(posterior_vulnerability(joint_from(prior, channel)) / vulnerability(prior))


def min_capacity(channel: Matrix) -> float:
    """log2 Σ_y max_x C[x][y]; equals the min-entropy leakage at the uniform prior."""
    return math.log2(sum(max(row[y] for row in channel) for y in range(len(channel[0]))))


def shannon_leakage(prior: Vector, channel: Matrix) -> float:
    return mutual_information(joint_from(prior, channel))


def uniform(n: int) -> list[float]:
    return [1 / n] * n


def pin_distribution(p_popular: float, size: int = 10_000) -> list[float]:
    """One popular PIN with probability p_popular, the rest uniform."""
    return [p_popular] + [(1 - p_popular) / (size - 1)] * (size - 1)


# --- Shannon versus min-entropy: two families of deterministic channels ----------------------
# The secret X is uniform on n-bit strings.
#   reveal-if-zero(j): output X if its lowest j bits are all 0, else output ⊥.
#   reveal-low(m): always output the lowest m bits of X.


def reveal_if_zero_channel(n: int, j: int) -> list[list[float]]:
    """Explicit 2^n x (2^n + 1) channel matrix; the last column is ⊥. For checking small n."""
    size = 2**n
    return [
        [1.0 if (y == x and x % 2**j == 0) or (y == size and x % 2**j != 0) else 0.0 for y in range(size + 1)]
        for x in range(size)
    ]


def reveal_low_channel(n: int, m: int) -> list[list[float]]:
    return [[1.0 if y == x % 2**m else 0.0 for y in range(2**m)] for x in range(2**n)]


def reveal_if_zero_shannon(n: int, j: int) -> float:
    """I(X; Y) = H(Y) = 2^-j n + (1 - 2^-j) log2(1 / (1 - 2^-j))."""
    a = 2.0**-j
    bottom = 0.0 if j == 0 else -(1 - a) * math.log1p(-a) / math.log(2)
    return a * n + bottom


def reveal_if_zero_min(n: int, j: int) -> float:
    """log2(V(X|Y) / V(X)) = log2(2^(n-j) + 1) when ⊥ can occur (j >= 1), and n when j = 0."""
    return float(n) if j == 0 else math.log2(2 ** (n - j) + 1)


def reveal_low_leakage(m: int) -> float:
    """Shannon and min-entropy leakage agree: both are m bits."""
    return float(m)


# --- Fano's inequality ----------------------------------------------------------------------


def fano_lhs(pe: float, m: int) -> float:
    """h(P_e) + P_e log2(m - 1): increasing in P_e on [0, (m - 1)/m]."""
    return binary_entropy(pe) + pe * math.log2(m - 1)


def fano_bound(h_cond: float, m: int) -> float:
    """Smallest P_e in [0, (m-1)/m] with h(P_e) + P_e log2(m - 1) >= H(X | Y), by bisection."""
    lo, hi = 0.0, (m - 1) / m
    if h_cond <= 0:
        return 0.0
    if fano_lhs(hi, m) <= h_cond:
        return hi
    for _ in range(200):
        mid = (lo + hi) / 2
        if fano_lhs(mid, m) >= h_cond:
            hi = mid
        else:
            lo = mid
    return hi


def weak_fano_bound(h_cond: float, m: int) -> float:
    """P_e >= (H(X | Y) - 1) / log2 m, from h(P_e) <= 1 and log2(m - 1) <= log2 m."""
    return max(0.0, (h_cond - 1) / math.log2(m))


def fano_sample_bound(n: int, kl_bits: float, m: int) -> float:
    """Fano's method: P_e >= 1 - (n κ + 1) / log2 m when I(X; Y^n) <= n κ and X is uniform on m values."""
    return max(0.0, 1 - (n * kl_bits + 1) / math.log2(m))


# --- Worked examples used in the text ------------------------------------------------------

# Spam filter: X = (ham, spam), Y = (word absent, word present).
SPAM_JOINT: list[list[float]] = [[3 / 4, 1 / 8], [0.0, 1 / 8]]

# The guessing walk-through: four secret values, three observations.
GUESS_PRIOR: list[float] = [0.4, 0.3, 0.2, 0.1]
GUESS_CHANNEL: list[list[float]] = [
    [0.8, 0.2, 0.0],
    [0.1, 0.8, 0.1],
    [0.0, 0.5, 0.5],
    [0.0, 0.2, 0.8],
]


# --- The chapter's table --------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Shannon entropy ==")
    print(
        f"h(0.5)={binary_entropy(0.5):.4f}  h(0.11)={binary_entropy(0.11):.4f}  h(1/8)={binary_entropy(1 / 8):.4f}  "
        f"h(1/4)={binary_entropy(1 / 4):.4f}"
    )
    j = SPAM_JOINT
    print(
        f"[spam] H(X)={entropy(marginal_x(j)):.4f}  H(Y)={entropy(marginal_y(j)):.4f}  "
        f"H(X,Y)={joint_entropy(j):.4f}  H(X|Y=0)={entropy_given_y(j, 0):.4f}  "
        f"H(X|Y=1)={entropy_given_y(j, 1):.4f}  H(X|Y)={conditional_entropy(j):.4f}"
    )
    pin = pin_distribution(0.1)
    print(
        f"[PIN] uniform: H={entropy(uniform(10_000)):.4f}  "
        f"popular 0.1: H={entropy(pin):.4f}  H_min={min_entropy(pin):.4f}"
    )

    print("== Mutual information ==")
    print(
        f"[spam] I=H(X)-H(X|Y)={mutual_information(j):.4f}  I=KL={mutual_information_kl(j):.4f}  "
        f"I via Y,X swapped={mutual_information(transpose(j)):.4f}"
    )
    noisy = compose([[1.0, 0.0], [0.0, 1.0]], bsc(0.1))  # Y -> Z through a BSC(0.1)
    jz = [[sum(j[x][y] * noisy[y][z] for y in range(2)) for z in range(2)] for x in range(2)]
    print(
        f"[spam, word detector flips w.p. 0.1] I(X;Z)={mutual_information(jz):.4f}  "
        f"<= I(X;Y)={mutual_information(j):.4f}"
    )

    print("== Channel capacity ==")
    for p in (0.0, 0.11, 0.25, 0.5):
        ba, r = blahut_arimoto(bsc(p))
        print(
            f"[BSC p={p}] 1-h(p)={bsc_capacity(p):.4f}  Blahut-Arimoto={ba:.4f}  optimal Pr[X=1]={r[1]:.4f}  "
            f"min-capacity={min_capacity(bsc(p)):.4f}"
        )
    print(f"[BSC p=0.11, prior q=0.2] I={bsc_information(0.2, 0.11):.4f}")
    ba, r = blahut_arimoto(z_channel(0.5))
    print(f"[Z-channel p=0.5] Blahut-Arimoto={ba:.4f}  log2(5/4)={math.log2(5 / 4):.4f}  optimal Pr[X=1]={r[1]:.4f}")
    ba, _ = blahut_arimoto(symmetric_channel(8, 0.3))
    print(
        f"[symmetric M=8 eps=0.3] formula={symmetric_channel_capacity(8, 0.3):.4f}  Blahut-Arimoto={ba:.4f}  "
        f"min-capacity={min_capacity(symmetric_channel(8, 0.3)):.4f}"
    )

    print("== Min-entropy and guessing ==")
    jg = joint_from(GUESS_PRIOR, GUESS_CHANNEL)
    print("[guess] joint rows: " + "  ".join("(" + ", ".join(f"{v:.2f}" for v in row) + ")" for row in jg))
    print("[guess] p(y)=" + ", ".join(f"{v:.2f}" for v in marginal_y(jg)) + f"  MAP guesses={map_guesses(jg)}")
    print(
        f"[guess] V(X)={vulnerability(GUESS_PRIOR):.4f}  H_min(X)={min_entropy(GUESS_PRIOR):.4f}  "
        f"V(X|Y)={posterior_vulnerability(jg):.4f}  H_min(X|Y)={conditional_min_entropy(jg):.4f}  "
        f"averaged H_min(X|Y=y)={averaged_min_entropy(jg):.4f}"
    )
    sim = simulate_map_success(rng, GUESS_PRIOR, GUESS_CHANNEL, trials=1_000_000)
    print(f"[guess] simulated MAP success over 1,000,000 draws={sim:.4f}")
    print(
        f"[guess] min-entropy leakage={min_entropy_leakage(GUESS_PRIOR, GUESS_CHANNEL):.4f}  "
        f"Shannon leakage={shannon_leakage(GUESS_PRIOR, GUESS_CHANNEL):.4f}  H(X)={entropy(GUESS_PRIOR):.4f}"
    )
    print(
        f"[guess] uniform prior: min-entropy leakage={min_entropy_leakage(uniform(4), GUESS_CHANNEL):.4f}  "
        f"min-capacity={min_capacity(GUESS_CHANNEL):.4f}  Shannon capacity={blahut_arimoto(GUESS_CHANNEL)[0]:.4f}"
    )
    best = max(
        min_entropy_leakage(list(pr), GUESS_CHANNEL)
        for pr in (
            (a / 20, b / 20, c / 20, (20 - a - b - c) / 20)
            for a in range(1, 18)
            for b in range(1, 19 - a)
            for c in range(1, 20 - a - b)
        )
    )
    print(f"[guess] max leakage over a grid of priors={best:.4f}")

    print("== Shannon versus min-entropy ==")
    for n in (4, 32):
        print(
            f"[reveal if X=0, n={n}] Shannon={reveal_if_zero_shannon(n, n):.4g}  "
            f"min-entropy={reveal_if_zero_min(n, n):.4f}"
        )
    n = 64
    print(
        f"[n={n}] reveal-if-low-3-zero: Shannon={reveal_if_zero_shannon(n, 3):.4f}  "
        f"min-entropy={reveal_if_zero_min(n, 3):.4f}  one-guess success={2.0**-3 + 2.0**-n:.4f}"
    )
    print(
        f"[n={n}] reveal-low-9: Shannon={reveal_low_leakage(9):.4f}  min-entropy={reveal_low_leakage(9):.4f}  "
        f"one-guess success={2.0 ** (9 - n):.3g}"
    )
    small = 6
    gap = 0.0
    for k in range(small + 1):
        cz, cl = reveal_if_zero_channel(small, k), reveal_low_channel(small, k)
        u = uniform(2**small)
        gap = max(
            gap,
            abs(shannon_leakage(u, cz) - reveal_if_zero_shannon(small, k)),
            abs(min_entropy_leakage(u, cz) - reveal_if_zero_min(small, k)),
            abs(shannon_leakage(u, cl) - reveal_low_leakage(k)),
            abs(min_entropy_leakage(u, cl) - reveal_low_leakage(k)),
        )
    print(f"[closed forms vs explicit matrices, n={small}] max gap={gap:.1e}")

    print("== Fano ==")
    for name, ch in (("symmetric", symmetric_channel(8, 0.2)), ("shift", shift_channel(8, 0.2))):
        jf = joint_from(uniform(8), ch)
        hc = conditional_entropy(jf)
        print(
            f"[{name} M=8 eps=0.2] H(X|Y)={hc:.4f}  MAP error={map_error(jf):.4f}  "
            f"Fano={fano_bound(hc, 8):.4f}  weak Fano={weak_fano_bound(hc, 8):.4f}"
        )
    print(f"[8 classes, I<=1 bit] Fano={fano_bound(2.0, 8):.4f}  weak Fano={weak_fano_bound(2.0, 8):.4f}")
    for n_samples in (100, 400):
        print(f"[M=1024, kappa=0.01 bit, n={n_samples}] P_e >= {fano_sample_bound(n_samples, 0.01, 1024):.4f}")


# --- Figures --------------------------------------------------------------------------------


def plot_binary_entropy() -> Figure:
    ps = [i / 1000 for i in range(1001)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ps, [binary_entropy(p) for p in ps], label=r"Shannon $h(p)$")
    ax.plot(ps, [min_entropy([p, 1 - p]) for p in ps], label=r"min-entropy $-\log_2\max(p, 1-p)$")
    ax.plot([0.11], [binary_entropy(0.11)], "ko")
    ax.annotate(
        r"$h(0.11) \approx 0.5$", (0.11, binary_entropy(0.11)), xytext=(0.16, 0.38), arrowprops={"arrowstyle": "->"}
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.05),
        xlabel=r"$p = \Pr[X = 1]$",
        ylabel="bits",
        title="Entropy of a Bernoulli(p) variable",
    )
    ax.legend(loc="lower center")
    return fig


def plot_conditioning(joint: Matrix = SPAM_JOINT) -> Figure:
    labels = ["H(X)", "H(X | Y = 0)\nword absent", "H(X | Y = 1)\nword present", "H(X | Y)\naverage"]
    values = [
        entropy(marginal_x(joint)),
        entropy_given_y(joint, 0),
        entropy_given_y(joint, 1),
        conditional_entropy(joint),
    ]
    colors = ["C0", "C1", "C1", "C2"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(values)), values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    ax.set_xticks(range(len(values)), labels)
    ax.axhline(values[0], color="C0", linestyle="--", linewidth=1)
    ax.set(ylim=(0, 1.15), ylabel="bits", title="One observation can raise entropy; the average cannot")
    return fig


def plot_guessing(prior: Vector = GUESS_PRIOR, channel: Matrix = GUESS_CHANNEL) -> Figure:
    jg = joint_from(prior, channel)
    arr = np.asarray(jg)
    guesses = map_guesses(jg)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.imshow(arr, cmap="Blues", vmin=0, vmax=0.45)
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            best = guesses[y] == x
            ax.text(
                y,
                x,
                f"{arr[x, y]:.2f}",
                ha="center",
                va="center",
                fontweight="bold" if best else "normal",
                color="white" if arr[x, y] > 0.25 else "black",
            )
            if best:
                ax.add_patch(Rectangle((y - 0.48, x - 0.48), 0.96, 0.96, fill=False, edgecolor="C3", linewidth=2.5))
    ax.set_xticks(range(arr.shape[1]), [f"$y_{i + 1}$" for i in range(arr.shape[1])])
    ax.set_yticks(range(arr.shape[0]), [f"$x_{i + 1}$  ($\\pi$ = {p:g})" for i, p in enumerate(prior)])
    ax.set(
        xlabel="observation",
        ylabel="secret",
        title=f"p(x, y); boxed column maxima sum to {posterior_vulnerability(jg):.2f}",
    )
    return fig


def plot_bsc() -> Figure:
    qs = [i / 200 for i in range(201)]
    ps = [i / 500 for i in range(501)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for p in (0.0, 0.05, 0.11, 0.25, 0.5):
        left.plot(qs, [bsc_information(q, p) for q in qs], label=f"p = {p:g}")
    left.axvline(0.5, color="gray", linestyle=":")
    left.set(
        xlim=(0, 1),
        ylim=(0, 1.05),
        xlabel=r"input prior $\pi_1 = \Pr[X = 1]$",
        ylabel=r"$I(X; Y)$ (bits)",
        title="BSC: information is maximal at q = 1/2",
    )
    left.legend(loc="upper right", fontsize="small")
    right.plot(ps, [bsc_capacity(p) for p in ps], label=r"capacity $1 - h(p)$")
    right.plot(ps, [min_capacity(bsc(p)) for p in ps], "--", label=r"min-capacity $\log_2(2\max(p, 1-p))$")
    right.set(
        xlim=(0, 1),
        ylim=(0, 1.05),
        xlabel="flip probability p",
        ylabel="bits per use",
        title="BSC capacity against flip probability",
    )
    right.legend(loc="upper center", fontsize="small")
    return fig


def plot_leakage(n: int = 64) -> Figure:
    js = list(range(n + 1))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        [reveal_low_leakage(m) for m in js],
        [reveal_low_leakage(m) for m in js],
        "o-",
        markersize=3,
        label="reveal the lowest m bits",
    )
    ax.plot(
        [reveal_if_zero_shannon(n, j) for j in js],
        [reveal_if_zero_min(n, j) for j in js],
        "s-",
        markersize=3,
        label="reveal X if its lowest j bits are 0",
    )
    for j in (1, 3, 6):
        ax.annotate(
            f"j = {j}",
            (reveal_if_zero_shannon(n, j), reveal_if_zero_min(n, j)),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize="small",
        )
    ax.annotate(
        f"j = {n}: reveal X only if X = 0",
        (reveal_if_zero_shannon(n, n), reveal_if_zero_min(n, n)),
        xytext=(8, 5),
        fontsize="small",
        arrowprops={"arrowstyle": "->"},
    )
    ax.set(
        xlim=(-1, n + 1),
        ylim=(-1, n + 1),
        xlabel="Shannon leakage I(X; Y) (bits)",
        ylabel=r"min-entropy leakage $\log_2 V(X|Y)/V(X)$ (bits)",
        title=f"Two families of channels on a uniform {n}-bit secret",
    )
    ax.legend(loc="lower right", fontsize="small")
    return fig


def plot_fano(m: int = 8) -> Figure:
    epss = [i / 200 for i in range(161)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (name, make) in zip(
        axes, (("symmetric channel", symmetric_channel), ("shift channel", shift_channel)), strict=True
    ):
        joints = [joint_from(uniform(m), make(m, e)) for e in epss]
        hcs = [conditional_entropy(jj) for jj in joints]
        ax.plot(epss, [map_error(jj) for jj in joints], "k", linewidth=2, label="MAP error (exact)")
        ax.plot(epss, [fano_bound(h, m) for h in hcs], "--", label="Fano bound")
        ax.plot(epss, [weak_fano_bound(h, m) for h in hcs], ":", label=r"$(H(X|Y) - 1)/\log_2 k$")
        ax.set(xlim=(0, 0.8), ylim=(0, 0.85), xlabel=r"noise level $p$", title=f"{name}, k = {m}")
        ax.legend(loc="upper left", fontsize="small")
    axes[0].set_ylabel("probability of a wrong guess")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="directory for the figures")
    args = parser.parse_args()
    seed: int = args.seed
    out: Path = args.out

    print_table(np.random.default_rng(seed))

    # The figures are exact computations, so they need no random generator (seed + 1 is unused).
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "information_theory_binary_entropy.svg": plot_binary_entropy(),
        "information_theory_conditioning.svg": plot_conditioning(),
        "information_theory_bsc.svg": plot_bsc(),
        "information_theory_guessing.svg": plot_guessing(),
        "information_theory_leakage.svg": plot_leakage(),
        "information_theory_fano.svg": plot_fano(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
