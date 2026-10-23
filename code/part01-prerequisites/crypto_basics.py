"""Check the numbers in the cryptography basics chapter and draw its figures.

    uv run python code/part01-prerequisites/crypto_basics.py
    uv run python code/part01-prerequisites/crypto_basics.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 08-crypto-basics.md (they match exactly with the default seed),
then saves the chapter's figures.

Every scheme here is a toy for illustration: the parameters are far too small to be secure,
nothing runs in constant time, and NumPy's generator is not a cryptographic random source.
"""

import argparse
import hashlib
import hmac
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "crypto_basics"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

IntArray = NDArray[np.int64]


def normal_sf(t: float) -> float:
    """Pr[Z >= t] for Z ~ N(0, 1)."""
    return 0.5 * math.erfc(t / math.sqrt(2))


# --- Negligible functions -------------------------------------------------------------------


def security_bits(name: str, lam: float) -> float:
    """-log2 μ(λ) for the four example functions of the chapter."""
    lg = math.log2(lam)
    if name == "2^-lam":
        return lam
    if name == "lam^-log lam":
        return lg * lg
    if name == "2^-sqrt lam":
        return math.sqrt(lam)
    if name == "lam^-10":
        return 10 * lg
    raise ValueError(name)


def first_crossing(above: str, below: str, start: int = 2) -> int:
    """Smallest integer λ >= start at which `above` has at least as many bits as `below`."""
    lam = start
    while security_bits(above, lam) < security_bits(below, lam):
        lam += 1
    return lam


# --- Concrete security ----------------------------------------------------------------------

SECONDS_PER_YEAR = 365.25 * 24 * 3600


def brute_force(ops_per_second: float, years: float, key_bits: int) -> tuple[float, float]:
    """log2 of the number of keys tried, and the probability that one of them is the key."""
    log2_tries = math.log2(ops_per_second * years * SECONDS_PER_YEAR)
    return log2_tries, 2.0 ** (log2_tries - key_bits)


def years_to_exhaust(ops_per_second: float, key_bits: int) -> float:
    return 2.0**key_bits / ops_per_second / SECONDS_PER_YEAR


# --- Birthday bound -------------------------------------------------------------------------


def collision_exact(q: int, n: int) -> float:
    """Pr[some two of q independent uniform draws from n values coincide]."""
    if q > n:
        return 1.0
    log_none = sum(math.log1p(-i / n) for i in range(q))
    return -math.expm1(log_none)


def collision_union_bound(q: int, n: int) -> float:
    return q * (q - 1) / (2 * n)


def collision_exp_lower(q: int, n: int) -> float:
    return -math.expm1(-q * (q - 1) / (2 * n))


def collision_empirical(rng: np.random.Generator, q: int, n: int, trials: int) -> float:
    draws = np.sort(rng.integers(0, n, size=(trials, q)), axis=1)
    return float(np.mean(np.any(draws[:, 1:] == draws[:, :-1], axis=1)))


def half_collision_queries(n: int) -> float:
    """q at which 1 - exp(-q²/(2n)) = 1/2."""
    return math.sqrt(2 * n * math.log(2))


# --- Distinguishing a biased coin: advantage and the hybrid bound --------------------------


def binomial_pmf_array(m: int, p: float) -> NDArray[np.float64]:
    k = np.arange(m + 1)
    log_comb = np.concatenate([[0.0], np.cumsum(np.log(np.arange(m, 0, -1)) - np.log(np.arange(1, m + 1)))])
    return np.exp(log_comb + k * math.log(p) + (m - k) * math.log1p(-p))


def coin_total_variation(m: int, delta: float) -> float:
    """TV between m flips of Bernoulli(1/2 + δ) and m flips of Bernoulli(1/2): the best advantage."""
    return float(0.5 * np.sum(np.abs(binomial_pmf_array(m, 0.5 + delta) - binomial_pmf_array(m, 0.5))))


def coin_majority_advantage(rng: np.random.Generator, m: int, delta: float, trials: int) -> float:
    """Empirical advantage of 'output 1 iff more than half the flips are 1'."""
    biased = rng.binomial(m, 0.5 + delta, size=trials) > m / 2
    fair = rng.binomial(m, 0.5, size=trials) > m / 2
    return float(np.mean(biased) - np.mean(fair))


def coin_hybrid_bound(m: int, delta: float) -> float:
    return min(1.0, m * delta)


# --- IND-CPA: deterministic against randomized encryption -----------------------------------

BLOCK = 16  # bytes


def prf(key: bytes, x: bytes) -> bytes:
    """Toy PRF: HMAC-SHA256 truncated to one block."""
    return hmac.new(key, x, hashlib.sha256).digest()[:BLOCK]


def xor(a: bytes, b: bytes) -> bytes:
    return bytes(u ^ v for u, v in zip(a, b, strict=True))


def enc_deterministic(key: bytes, m: bytes) -> bytes:
    """m XOR F_k(0): decryptable, but the same message always gives the same ciphertext."""
    return xor(m, prf(key, bytes(1)))


def enc_randomized(rng: np.random.Generator, key: bytes, m: bytes, nonce_bits: int) -> tuple[int, bytes]:
    """(r, m XOR F_k(r)) with r uniform on nonce_bits bits."""
    r = int(rng.integers(0, 2**nonce_bits))
    return r, xor(m, prf(key, r.to_bytes(8, "big")))


M0 = bytes(BLOCK)
M1 = bytes([0xFF]) * BLOCK


def cpa_game_deterministic(rng: np.random.Generator, b: int) -> int:
    """One run of the IND-CPA game; the adversary encrypts m0 itself and compares."""
    key = rng.bytes(16)
    c0 = enc_deterministic(key, M0)
    challenge = enc_deterministic(key, M1 if b else M0)
    return 0 if challenge == c0 else 1


def cpa_game_randomized(rng: np.random.Generator, b: int, queries: int, nonce_bits: int) -> int:
    """The adversary asks for `queries` encryptions of m0 and wins outright if a nonce repeats."""
    key = rng.bytes(16)
    pads = dict((r, xor(c, M0)) for r, c in (enc_randomized(rng, key, M0, nonce_bits) for _ in range(queries)))
    r_star, c_star = enc_randomized(rng, key, M1 if b else M0, nonce_bits)
    if r_star in pads:
        return 0 if xor(c_star, pads[r_star]) == M0 else 1
    return 1


def cpa_advantage_deterministic(rng: np.random.Generator, trials: int) -> float:
    ones_b1 = sum(cpa_game_deterministic(rng, 1) for _ in range(trials))
    ones_b0 = sum(cpa_game_deterministic(rng, 0) for _ in range(trials))
    return (ones_b1 - ones_b0) / trials


def cpa_advantage_randomized(rng: np.random.Generator, queries: int, nonce_bits: int, trials: int) -> float:
    ones_b1 = sum(cpa_game_randomized(rng, 1, queries, nonce_bits) for _ in range(trials))
    ones_b0 = sum(cpa_game_randomized(rng, 0, queries, nonce_bits) for _ in range(trials))
    return (ones_b1 - ones_b0) / trials


def nonce_repeat_probability(queries: int, nonce_bits: int) -> float:
    """Exact advantage of the nonce-repeat adversary: Pr[r* is among the q oracle nonces]."""
    return 1 - (1 - 2.0**-nonce_bits) ** queries


# --- Simulation paradigm: additive secret sharing -------------------------------------------


def share_mod(x: int, randomness: tuple[int, ...], q: int) -> tuple[int, ...]:
    """Additive shares of x in Z_q: the n-1 random values, then x minus their sum."""
    return (*randomness, (x - sum(randomness)) % q)


def all_coalitions_uniform(n: int, q: int) -> bool:
    """Count: for every x and every coalition of n-1 holders, each tuple of shares occurs exactly once."""
    for x in range(q):
        for missing in range(n):
            counts: dict[tuple[int, ...], int] = {}
            for r in product(range(q), repeat=n - 1):
                s = share_mod(x, r, q)
                seen = tuple(v for i, v in enumerate(s) if i != missing)
                counts[seen] = counts.get(seen, 0) + 1
            if len(counts) != q ** (n - 1) or set(counts.values()) != {1}:
                return False
    return True


def second_share_mod(rng: np.random.Generator, x: int, q: int, size: int) -> IntArray:
    r = rng.integers(0, q, size=size)
    return (x - r) % q


def second_share_integer(rng: np.random.Generator, x: int, r_max: int, size: int) -> IntArray:
    """x - r over the integers, r uniform on {0, ..., r_max - 1}: no reduction mod q."""
    return x - rng.integers(0, r_max, size=size)


def empirical_tv(a: IntArray, b: IntArray) -> float:
    lo, hi = int(min(a.min(), b.min())), int(max(a.max(), b.max()))
    ha = np.bincount(a - lo, minlength=hi - lo + 1) / a.size
    hb = np.bincount(b - lo, minlength=hi - lo + 1) / b.size
    return float(0.5 * np.sum(np.abs(ha - hb)))


# --- LWE: Gaussian elimination and Regev's bit encryption -----------------------------------


def solve_mod(rows: list[list[int]], rhs: list[int], q: int) -> list[int] | None:
    """Solve A s = b mod q by Gaussian elimination, pivoting only on entries invertible mod q.

    Uses as many rows as needed among the m >= n given; returns None if no invertible pivot exists.
    """
    a = [row[:] + [v] for row, v in zip(rows, rhs, strict=True)]
    n = len(rows[0])
    used: list[int] = []
    for col in range(n):
        pivot = next((i for i in range(len(a)) if i not in used and math.gcd(a[i][col], q) == 1), None)
        if pivot is None:
            return None
        inv = pow(a[pivot][col], -1, q)
        a[pivot] = [v * inv % q for v in a[pivot]]
        for i in range(len(a)):
            if i != pivot and a[i][col]:
                f = a[i][col]
                a[i] = [(v - f * w) % q for v, w in zip(a[i], a[pivot], strict=True)]
        used.append(pivot)
    s = [0] * n
    for col, row in enumerate(used):
        s[col] = a[row][n]
    return s


@dataclass(frozen=True)
class LwePlain:
    secret: list[int]
    noiseless: list[int] | None
    noisy: list[int] | None


def lwe_elimination(rng: np.random.Generator, n: int, m: int, q: int, noise: int) -> LwePlain:
    """Recover s from m samples (a, <a,s> [+ e]) by elimination, with and without noise in [-noise, noise]."""
    s = rng.integers(0, q, size=n)
    a = rng.integers(0, q, size=(m, n))
    e = rng.integers(-noise, noise + 1, size=m)
    exact = (a @ s) % q
    rows = [[int(v) for v in row] for row in a]
    return LwePlain(
        secret=[int(v) for v in s],
        noiseless=solve_mod(rows, [int(v) for v in exact], q),
        noisy=solve_mod(rows, [int(v) for v in (exact + e) % q], q),
    )


def regev_encrypt(s: IntArray, a: IntArray, e: int, m: int, q: int) -> int:
    """b = <a, s> + e + m q/2 mod q; the ciphertext is (a, b)."""
    return int((int(a @ s) + e + m * (q // 2)) % q)


def regev_decrypt(s: IntArray, a: IntArray, b: int, q: int) -> int:
    """1 iff b - <a, s> mod q is closer to q/2 than to 0."""
    d = (b - int(a @ s)) % q
    return 1 if q // 4 < d < 3 * q // 4 else 0


def regev_encrypt_batch(
    rng: np.random.Generator, s: IntArray, m: IntArray, q: int, bound: int
) -> tuple[IntArray, IntArray]:
    """One fresh ciphertext per row of s (one key per trial), noise uniform on [-bound, bound]."""
    a = rng.integers(0, q, size=s.shape)
    e = rng.integers(-bound, bound + 1, size=s.shape[0])
    b = (np.sum(a * s, axis=1) + e + m * (q // 2)) % q
    return a, b


def regev_decrypt_batch(s: IntArray, a: IntArray, b: IntArray, q: int) -> IntArray:
    d = (b - np.sum(a * s, axis=1)) % q
    return ((d > q // 4) & (d < 3 * q // 4)).astype(np.int64)


def regev_sum_failure_rates(
    rng: np.random.Generator, n: int, q: int, bound: int, ks: list[int], trials: int
) -> list[float]:
    """Add k fresh ciphertexts of random bits, decrypt the sum, compare with the XOR of the bits.

    Returns the empirical failure rate at each k in ks (one running sum, checked at each k).
    """
    s = rng.integers(0, q, size=(trials, n))
    a_sum = np.zeros((trials, n), dtype=np.int64)
    b_sum = np.zeros(trials, dtype=np.int64)
    parity = np.zeros(trials, dtype=np.int64)
    rates: list[float] = []
    for k in range(1, max(ks) + 1):
        m = rng.integers(0, 2, size=trials)
        a, b = regev_encrypt_batch(rng, s, m, q, bound)
        a_sum = (a_sum + a) % q
        b_sum = (b_sum + b) % q
        parity ^= m
        if k in ks:
            wrong: NDArray[np.bool_] = regev_decrypt_batch(s, a_sum, b_sum, q) != parity
            rates.append(float(np.mean(wrong)))
    return rates


def regev_worst_case_k(q: int, bound: int) -> float:
    """Sums of k ciphertexts with |e_i| <= bound always decrypt when k < q / (4 bound)."""
    return q / (4 * bound)


def regev_failure_normal(k: int, q: int, bound: int) -> float:
    """Normal approximation: Pr[the summed noise, wrapped mod q, is more than q/4 from 0]."""
    sd = math.sqrt(k * bound * (bound + 1) / 3)  # variance of uniform on [-B, B] is B(B+1)/3
    span = math.ceil(8 * sd / q) + 1
    total = 0.0
    for j in range(-span, span + 1):
        lo, hi = (q / 4 + j * q) / sd, (3 * q / 4 + j * q) / sd
        total += normal_sf(lo) - normal_sf(hi)
    return total


# --- The chapter's table --------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    lam = 128
    names = ("2^-lam", "lam^-log lam", "2^-sqrt lam", "lam^-10")
    print(
        f"[Negligible at lambda={lam}] "
        + "  ".join(f"{nm}=2^-{security_bits(nm, lam):.2f} ({2.0 ** -security_bits(nm, lam):.1e})" for nm in names)
    )
    print(f"[Crossing] 2^-sqrt(lam) first below lam^-10 at lambda={first_crossing('2^-sqrt lam', 'lam^-10', 16)}")

    bits, p = brute_force(1e18, 1.0, 128)
    print(
        f"[Brute force] 1e18 ops/s for a year = 2^{bits:.1f} keys, success={p:.1e}; "
        f"exhausting 2^128 takes {years_to_exhaust(1e18, 128):.2e} years"
    )

    for q, n in ((23, 365), (1000, 2**20), (100_000, 2**32)):
        print(
            f"[Birthday q={q} N={n}] exact={collision_exact(q, n):.3f}  "
            f"union={collision_union_bound(q, n):.3f}  1-exp={collision_exp_lower(q, n):.3f}"
        )
    q, n = 256, 2**16
    print(
        f"[Birthday q={q} N=2^16] exact={collision_exact(q, n):.4f}  "
        f"empirical={collision_empirical(rng, q, n, trials=100_000):.4f}  "
        f"union={collision_union_bound(q, n):.4f}  half at q={half_collision_queries(n):.0f}"
    )

    delta = 0.01
    for m in (1, 100, 10_000):
        print(
            f"[Coin delta={delta} m={m}] majority empirical={coin_majority_advantage(rng, m, delta, 200_000):.3f}  "
            f"best (TV)={coin_total_variation(m, delta):.3f}  hybrid bound={coin_hybrid_bound(m, delta):.3f}"
        )

    trials = 20_000
    print(f"[IND-CPA deterministic] advantage={cpa_advantage_deterministic(rng, trials):.3f}")
    queries, nonce_bits = 16, 8
    print(
        f"[IND-CPA randomized q={queries} r={nonce_bits} bits] "
        f"empirical={cpa_advantage_randomized(rng, queries, nonce_bits, trials):.4f}  "
        f"exact={nonce_repeat_probability(queries, nonce_bits):.4f}  bound q/|R|={queries / 2**nonce_bits:.4f}"
    )

    print(f"[Sharing h=3 p=5] every coalition of 2 sees each pair exactly once: {all_coalitions_uniform(3, 5)}")
    size = 100_000
    tv_mod = empirical_tv(second_share_mod(rng, 0, 11, size), second_share_mod(rng, 7, 11, size))
    tv_int = empirical_tv(second_share_integer(rng, 0, 16, size), second_share_integer(rng, 7, 16, size))
    print(
        f"[Sharing x=0 vs x=7] Z_11 empirical TV={tv_mod:.4f} (exact 0)  "
        f"integers R=16 empirical TV={tv_int:.4f} (exact {7 / 16:.4f})"
    )

    lwe = lwe_elimination(rng, n=16, m=32, q=4096, noise=1)
    ok_clean = sum(u == v for u, v in zip(lwe.secret, lwe.noiseless or [], strict=False))
    ok_noisy = sum(u == v for u, v in zip(lwe.secret, lwe.noisy or [], strict=False))
    print(
        f"[LWE d=16 p=4096] elimination recovers {ok_clean}/16 coordinates without noise, "
        f"{ok_noisy}/16 with e in {{-1,0,1}}"
    )

    q = 4096
    s = np.array([5, 1000, 3000], dtype=np.int64)
    a1, a2 = np.array([17, 2, 1], dtype=np.int64), np.array([1, 1, 1], dtype=np.int64)
    b1, b2 = regev_encrypt(s, a1, -7, 1, q), regev_encrypt(s, a2, 12, 1, q)
    print(
        f"[Regev example] b1={b1} dec={regev_decrypt(s, a1, b1, q)}  b2={b2} dec={regev_decrypt(s, a2, b2, q)}  "
        f"sum b={(b1 + b2) % q} <a,s>={int((a1 + a2) @ s) % q} dec={regev_decrypt(s, a1 + a2, (b1 + b2) % q, q)}"
    )

    bound, ks = 32, [1, 31, 256, 1024, 4096]
    rates = regev_sum_failure_rates(rng, n=4, q=q, bound=bound, ks=ks, trials=20_000)
    print(f"[Regev sums p={q} B={bound}] worst-case safe for k < {regev_worst_case_k(q, bound):.0f}")
    for k, rate in zip(ks, rates, strict=True):
        print(f"[Regev k={k}] empirical failure={rate:.4f}  normal approx={regev_failure_normal(k, q, bound):.4f}")


# --- Figures --------------------------------------------------------------------------------


def plot_negligible() -> Figure:
    lams = [2.0 ** (i / 8) for i in range(8, 8 * 20 + 1)]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = {
        "2^-lam": r"$2^{-\lambda}$",
        "lam^-log lam": r"$\lambda^{-\log_2 \lambda}$",
        "2^-sqrt lam": r"$2^{-\sqrt{\lambda}}$",
        "lam^-10": r"$\lambda^{-10}$ (not negligible)",
    }
    for name, label in labels.items():
        style = "--" if name == "lam^-10" else "-"
        ax.plot(lams, [security_bits(name, lam) for lam in lams], style, label=label)
    ax.axvline(128, color="gray", linestyle=":", label=r"$\lambda = 128$")
    ax.set(
        xscale="log",
        yscale="log",
        ylim=(0.1, 3e6),
        xlabel=r"security parameter $\lambda$",
        ylabel=r"$-\log_2 \mu(\lambda)$ (bits)",
        title="Negligible functions eventually beat every polynomial",
    )
    ax.set_xscale("log", base=2)
    ax.legend(loc="lower right")
    return fig


def plot_birthday(rng: np.random.Generator, n: int = 2**16) -> Figure:
    qs = list(range(10, 801, 10))
    empirical = [collision_empirical(rng, q, n, trials=4_000) for q in qs[::4]]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(qs, [collision_exact(q, n) for q in qs], "k", linewidth=2, label="exact")
    ax.plot(qs[::4], empirical, "o", color="C3", label="simulation")
    ax.plot(qs, [min(1.0, collision_union_bound(q, n)) for q in qs], label=r"union bound $Q(Q-1)/(2N)$")
    ax.plot(qs, [collision_exp_lower(q, n) for q in qs], "--", label=r"lower bound $1 - e^{-Q(Q-1)/(2N)}$")
    ax.axvline(math.sqrt(n), color="gray", linestyle=":", label=r"$Q = \sqrt{N}$")
    ax.set(
        ylim=(0, 1.05),
        xlabel="number of draws Q",
        ylabel="Pr[some collision]",
        title=r"Birthday bound, $N = 2^{16}$ values",
    )
    ax.legend(loc="lower right")
    return fig


def plot_hybrid(rng: np.random.Generator, delta: float = 0.01) -> Figure:
    ms = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ms, [coin_hybrid_bound(m, delta) for m in ms], label=r"hybrid bound $\min(1, m\xi)$")
    ax.plot(ms, [coin_total_variation(m, delta) for m in ms], "k", linewidth=2, label="best distinguisher (TV)")
    ax.plot(
        ms,
        [coin_majority_advantage(rng, m, delta, 100_000) for m in ms],
        "o",
        color="C3",
        label="majority vote (simulation)",
    )
    ax.set(
        xscale="log",
        yscale="log",
        ylim=(5e-3, 1.5),
        xlabel="number of samples m",
        ylabel="distinguishing advantage",
        title=rf"Bern$(1/2 + \xi)$ against Bern$(1/2)$, $\xi = {delta}$",
    )
    ax.legend(loc="lower right")
    return fig


def plot_shares(rng: np.random.Generator, q: int = 11, r_max: int = 16, size: int = 50_000) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for i, x in enumerate((0, 7)):
        shares = second_share_mod(rng, x, q, size)
        freq = np.bincount(shares, minlength=q) / size
        left.bar(np.arange(q) + (i - 0.5) * 0.4, freq, width=0.4, label=f"secret x = {x}")
    left.axhline(1 / q, color="k", linestyle="--", label=f"1/p = 1/{q}")
    left.set(
        xticks=range(q),
        ylim=(0, 0.16),
        xlabel=r"share $(x - r)$ mod $p$",
        ylabel="frequency",
        title=rf"Mod $p = {q}$: the same uniform distribution",
    )
    left.legend(loc="upper right")
    for i, x in enumerate((0, 7)):
        shares = second_share_integer(rng, x, r_max, size)
        values = np.arange(-r_max + 1, 8)
        freq = np.bincount(shares - values[0], minlength=values.size) / size
        right.bar(values + (i - 0.5) * 0.4, freq, width=0.4, label=f"secret x = {x}")
    right.set(
        ylim=(0, 0.16),
        xlabel=r"share $x - r$ over the integers",
        ylabel="frequency",
        title=rf"No reduction, $r < {r_max}$: the support moves with $x$",
    )
    right.legend(loc="upper right")
    return fig


def plot_regev(rng: np.random.Generator, q: int = 4096, bound: int = 32) -> Figure:
    ks = sorted({round(2 ** (i / 4)) for i in range(4 * 13 + 1)})
    rates = regev_sum_failure_rates(rng, n=4, q=q, bound=bound, ks=ks, trials=4_000)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ks, [regev_failure_normal(k, q, bound) for k in ks], "k", linewidth=2, label="normal approximation")
    ax.plot(ks, rates, "o", color="C3", label="simulation")
    ax.axvline(regev_worst_case_k(q, bound), color="C0", linestyle="--", label=r"worst-case limit $k = p/(4B)$")
    ax.axhline(0.5, color="gray", linestyle=":", label="random guess")
    ax.set(
        xscale="log",
        ylim=(-0.02, 0.6),
        xlabel="ciphertexts added k",
        ylabel="decryption failure rate",
        title=rf"Noise growth in Regev sums, $p = {q}$, $B = {bound}$",
    )
    ax.set_xscale("log", base=2)
    ax.legend(loc="upper left")
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
        "crypto_basics_negligible.svg": plot_negligible(),
        "crypto_basics_hybrid.svg": plot_hybrid(rng),
        "crypto_basics_birthday.svg": plot_birthday(rng),
        "crypto_basics_shares.svg": plot_shares(rng),
        "crypto_basics_regev_noise.svg": plot_regev(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
