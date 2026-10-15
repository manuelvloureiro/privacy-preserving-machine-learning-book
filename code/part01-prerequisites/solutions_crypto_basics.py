"""Check the numbers in the worked solutions to the cryptography basics exercises.

    uv run python code/part01-prerequisites/solutions_crypto_basics.py
    uv run python code/part01-prerequisites/solutions_crypto_basics.py --seed 1

Prints the numbers quoted in book-exercises/part01-prerequisites/08-crypto-basics.md (they match exactly with the default
seed). Makes no figures; --out is accepted for symmetry with the chapter scripts.

Every scheme here is a toy for illustration: the parameters are far too small to be secure,
nothing runs in constant time, and NumPy's generator is not a cryptographic random source.
"""

import argparse
import hashlib
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crypto_basics import (
    binomial_pmf_array,
    coin_total_variation,
    collision_exact,
    half_collision_queries,
    normal_sf,
    regev_failure_normal,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

IntArray = NDArray[np.int64]
NoiseSampler = Callable[[np.random.Generator, int], IntArray]


def normal_cdf(t: float) -> float:
    return 1.0 - normal_sf(t)


def normal_quantile(p: float) -> float:
    """z with Pr[Z <= z] = p, by bisection on the normal CDF."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --- Exercise 1: which are negligible? -----------------------------------------------------


def bits_loglog(lam: float) -> float:
    """-log2 of λ^(-log2 log2 λ)."""
    return math.log2(lam) * math.log2(math.log2(lam))


# --- Exercise 2: hybrids for TV, and the sqrt(m) growth ------------------------------------


def coin_tv_normal(m: int, delta: float) -> float:
    """Normal approximation 2Φ(δ√m) - 1: two normals with sd √m/2 whose means differ by mδ."""
    return 2 * normal_cdf(delta * math.sqrt(m)) - 1


def coin_tv_linear(m: int, delta: float) -> float:
    """Small-shift approximation √(2/π) δ √m of 2Φ(δ√m) - 1."""
    return math.sqrt(2 / math.pi) * delta * math.sqrt(m)


# --- Exercise 3: the best threshold --------------------------------------------------------


def likelihood_ratio_crossing(m: int, delta: float) -> float:
    """k* where (1 + 2δ)^k (1 - 2δ)^(m-k) = 1: the two binomial probabilities cross."""
    return m * -math.log1p(-2 * delta) / (math.log1p(2 * delta) - math.log1p(-2 * delta))


def threshold_advantage(m: int, delta: float, threshold: float) -> float:
    """Pr[K > t] under Bin(m, 1/2 + δ) minus the same under Bin(m, 1/2)."""
    k = np.arange(m + 1)
    above = k > threshold
    biased = binomial_pmf_array(m, 0.5 + delta)
    fair = binomial_pmf_array(m, 0.5)
    return float(np.sum(biased[above]) - np.sum(fair[above]))


# --- Exercise 4: birthday for 32 bits ------------------------------------------------------


def first_half_collision(n: int, start: int) -> int:
    """Smallest q >= start with exact collision probability at least 1/2 (start must be below it)."""
    q = start
    while collision_exact(q, n) < 0.5:
        q += 1
    return q


# --- Exercise 5: two-time pad --------------------------------------------------------------


def pad_encrypt(pad: bytes, m: bytes) -> bytes:
    return bytes(u ^ v for u, v in zip(m, pad, strict=True))


def crib_drag(xored: bytes, crib: bytes, position: int) -> bytes:
    """XOR a guessed word into c1 XOR c2 at a position: the other message's bytes there, if the guess is right."""
    return bytes(u ^ v for u, v in zip(xored[position : position + len(crib)], crib, strict=True))


def readable(text: bytes) -> bool:
    return all(c == 32 or 97 <= c <= 122 for c in text)


# --- Exercise 6: toy GGM -------------------------------------------------------------------


class CountingGenerator:
    """Length-doubling generator from SHA-256: G_b(s) = SHA256(s || b), counting hash calls."""

    def __init__(self) -> None:
        self.calls = 0

    def half(self, seed: bytes, bit: int) -> bytes:
        self.calls += 1
        return hashlib.sha256(seed + bytes([bit])).digest()


def ggm_eval(gen: CountingGenerator, key: bytes, bits: list[int]) -> bytes:
    """Walk the GGM tree from the root key, one generator half per input bit."""
    node = key
    for b in bits:
        node = gen.half(node, b)
    return node


def int_to_bits(x: int, n: int) -> list[int]:
    return [(x >> (n - 1 - i)) & 1 for i in range(n)]


def ggm_all_inputs(key: bytes, n: int) -> tuple[int, int, float]:
    """Evaluate F_k on all 2^n inputs: (calls per evaluation, distinct outputs, fraction of 1 bits)."""
    gen = CountingGenerator()
    outputs = [ggm_eval(gen, key, int_to_bits(x, n)) for x in range(2**n)]
    ones = sum(bin(byte).count("1") for out in outputs for byte in out)
    return gen.calls // 2**n, len(set(outputs)), ones / (len(outputs) * 256)


def ggm_consistent(key: bytes, n: int) -> bool:
    """The same input gives the same output from a fresh generator: F_k is a function."""
    x = int_to_bits(2**n - 3, n)
    return ggm_eval(CountingGenerator(), key, x) == ggm_eval(CountingGenerator(), key, x)


# --- Exercise 7: threshold sharing ---------------------------------------------------------


def shamir_share(x: int, c: int, holder: int, p: int) -> int:
    return (x + c * holder) % p


def shamir_reconstruct(i: int, si: int, j: int, sj: int, p: int) -> int:
    """x from shares of holders i != j: c = (s_i - s_j)/(i - j), x = s_i - c i."""
    c = (si - sj) * pow(i - j, -1, p) % p
    return (si - c * i) % p


def shamir_single_share_uniform(p: int) -> bool:
    """For every x and holder, each share value comes from exactly one c."""
    for x in range(p):
        for holder in (1, 2, 3):
            if sorted(shamir_share(x, c, holder, p) for c in range(p)) != list(range(p)):
                return False
    return True


def shamir_pairs_reconstruct(p: int) -> bool:
    """For every x, c and pair of holders, the two shares give back x."""
    for x in range(p):
        for c in range(p):
            for i, j in ((1, 2), (1, 3), (2, 3)):
                si, sj = shamir_share(x, c, i, p), shamir_share(x, c, j, p)
                if shamir_reconstruct(i, si, j, sj, p) != x:
                    return False
    return True


# --- Exercises 8 and 9: Regev sums with other noise and an odd modulus ---------------------


def uniform_noise(bound: int) -> NoiseSampler:
    def sample(rng: np.random.Generator, size: int) -> IntArray:
        return rng.integers(-bound, bound + 1, size=size)

    return sample


def rounded_normal_noise(sigma: float) -> NoiseSampler:
    def sample(rng: np.random.Generator, size: int) -> IntArray:
        return np.rint(rng.normal(0.0, sigma, size=size)).astype(np.int64)

    return sample


def rounded_normal_variance(sigma: float) -> float:
    """Exact variance of round(N(0, σ²)), summing over the integers within 20σ."""
    top = int(20 * sigma) + 1
    total = 0.0
    for j in range(-top, top + 1):
        mass = normal_cdf((j + 0.5) / sigma) - normal_cdf((j - 0.5) / sigma)
        total += j * j * mass
    return total


def failure_normal(k: int, q: int, var: float, shift: float = 0.0) -> float:
    """Pr[N(shift, k var), wrapped mod q, lands more than q/4 from 0]."""
    sd = math.sqrt(k * var)
    span = math.ceil((8 * sd + abs(shift)) / q) + 1
    total = 0.0
    for j in range(-span, span + 1):
        lo, hi = (q / 4 + j * q - shift) / sd, (3 * q / 4 + j * q - shift) / sd
        total += normal_sf(lo) - normal_sf(hi)
    return total


def k_at_failure(q: int, var: float, target: float) -> float:
    """Unwrapped two-sided approximation: (q/4)² / (z² var) with z the (1 - target/2) quantile."""
    z = normal_quantile(1 - target / 2)
    return (q / 4) ** 2 / (z * z * var)


def k_at_failure_wrapped(q: int, var: float, target: float) -> int:
    """Smallest integer k at which the wrapped normal approximation reaches the target rate."""
    lo, hi = 1, 1
    while failure_normal(hi, q, var) < target:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if failure_normal(mid, q, var) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def regev_sum_experiment(
    rng: np.random.Generator, n: int, q: int, noise: NoiseSampler, ks: list[int], trials: int
) -> list[tuple[float, float]]:
    """Add k fresh Regev ciphertexts of random bits (encoding m⌊q/2⌋), decrypt, compare with the XOR.

    Returns (failure rate, mean drift) at each k, where the drift is what the decrypted value
    d = Σb - <Σa, s> has on top of the summed noise and the parity's encoding, centred in (-q/2, q/2].
    """
    half = q // 2
    s = rng.integers(0, q, size=(trials, n))
    a_sum = np.zeros((trials, n), dtype=np.int64)
    b_sum = np.zeros(trials, dtype=np.int64)
    e_sum = np.zeros(trials, dtype=np.int64)
    parity = np.zeros(trials, dtype=np.int64)
    out: list[tuple[float, float]] = []
    for k in range(1, max(ks) + 1):
        m = rng.integers(0, 2, size=trials)
        a = rng.integers(0, q, size=(trials, n))
        e = noise(rng, trials)
        b = (np.sum(a * s, axis=1) + e + m * half) % q
        a_sum = (a_sum + a) % q
        b_sum = (b_sum + b) % q
        e_sum += e
        parity ^= m
        if k in ks:
            d = (b_sum - np.sum(a_sum * s, axis=1)) % q
            decoded = ((4 * d > q) & (4 * d < 3 * q)).astype(np.int64)
            wrapped: IntArray = (d - e_sum - parity * half) % q
            drift: IntArray = np.where(wrapped > q // 2, wrapped - q, wrapped)
            wrong: NDArray[np.bool_] = decoded != parity
            out.append((float(np.mean(wrong)), float(np.mean(drift))))
    return out


def odd_drift_prediction(k: int) -> float:
    """Expected drift -E⌊w/2⌋ for w ~ Bin(k, 1/2): -(k - 1)/4 for k >= 1."""
    return -(k - 1) / 4


# --- The solutions' table ------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print(f"[Ex1] lam^-loglog lam at lambda=128: 2^-{bits_loglog(128):.2f}")

    delta = 0.01
    for m in (1, 100, 400, 2500, 10_000):
        print(
            f"[Ex2 m={m}] TV={coin_total_variation(m, delta):.4f}  normal={coin_tv_normal(m, delta):.4f}  "
            f"linear={coin_tv_linear(m, delta):.4f}  hybrid={min(1.0, m * delta):.4f}"
        )

    m = 10_000
    k_star = likelihood_ratio_crossing(m, delta)
    print(
        f"[Ex3 m={m}] crossing k*={k_star:.4f} (m/2 + m delta/2 = {m / 2 + m * delta / 2:.0f})  "
        f"threshold advantage={threshold_advantage(m, delta, k_star):.4f}  TV={coin_total_variation(m, delta):.4f}  "
        f"majority advantage={threshold_advantage(m, delta, m / 2):.4f}"
    )

    n32 = 2**32
    q_formula = half_collision_queries(n32)
    q_exact = first_half_collision(n32, int(q_formula) - 100)
    print(
        f"[Ex4] sqrt(2 N ln 2)={q_formula:.1f}  smallest q with exact >= 1/2: {q_exact} "
        f"(exact at q-1: {collision_exact(q_exact - 1, n32):.6f}, at q: {collision_exact(q_exact, n32):.6f})"
    )

    m1, m2 = b"meet me at the north gate", b"the password is blue fish"
    pad = rng.bytes(len(m1))
    c1, c2 = pad_encrypt(pad, m1), pad_encrypt(pad, m2)
    xored = pad_encrypt(c1, c2)
    crib = b"the "
    hits = [p for p in range(len(xored) - len(crib) + 1) if readable(crib_drag(xored, crib, p))]
    print(
        f"[Ex5] c1^c2 = m1^m2: {xored == pad_encrypt(m1, m2)}  crib 'the ' readable at positions {hits}: "
        + "  ".join(f"{p}->{crib_drag(xored, crib, p).decode()!r}" for p in hits)
    )

    key = rng.bytes(32)
    calls, distinct, ones = ggm_all_inputs(key, 8)
    print(
        f"[Ex6 n=8] calls per evaluation={calls}  distinct outputs={distinct}/256  "
        f"fraction of 1 bits={ones:.4f}  consistent={ggm_consistent(key, 8)}"
    )

    p = 5
    print(
        f"[Ex7 p={p}] x=2 shares (s1,s2,s3) for c=0..4: "
        + " ".join(str(tuple(shamir_share(2, c, h, p) for h in (1, 2, 3))) for c in range(p))
    )
    print(
        f"[Ex7] single share uniform p=5: {shamir_single_share_uniform(5)}, p=7: {shamir_single_share_uniform(7)}; "
        f"pairs reconstruct p=5: {shamir_pairs_reconstruct(5)}, p=7: {shamir_pairs_reconstruct(7)}; "
        f"holders 1,3 with s1=4, s3=3: x={shamir_reconstruct(1, 4, 3, 3, 5)}"
    )

    q, bound, target, trials = 4096, 32, 0.01, 20_000
    var_u = bound * (bound + 1) / 3
    k_u, k_u_wrapped = k_at_failure(q, var_u, target), k_at_failure_wrapped(q, var_u, target)
    rate_u = regev_sum_experiment(rng, 4, q, uniform_noise(bound), [k_u_wrapped], trials)[0][0]
    print(
        f"[Ex8 uniform B={bound}] var={var_u:.0f}  z={normal_quantile(1 - target / 2):.3f}  k={k_u:.1f}  "
        f"wrapped k={k_u_wrapped} (approx {regev_failure_normal(k_u_wrapped, q, bound):.4f})  "
        f"simulated failure at that k={rate_u:.4f}"
    )
    sigma = 3.2
    var_g = rounded_normal_variance(sigma)
    k_g, k_g_wrapped = k_at_failure(q, var_g, target), k_at_failure_wrapped(q, var_g, target)
    rate_g = regev_sum_experiment(rng, 4, q, rounded_normal_noise(sigma), [k_g_wrapped], trials)[0][0]
    print(
        f"[Ex8 rounded normal sigma={sigma}] var={var_g:.4f}  k={k_g:.0f}  wrapped k={k_g_wrapped}  "
        f"simulated failure at that k={rate_g:.4f}"
    )

    q_odd, ks = 4093, [1, 31, 256, 1024, 4096]
    odd = regev_sum_experiment(rng, 4, q_odd, uniform_noise(bound), ks, trials)
    even = regev_sum_experiment(rng, 4, q, uniform_noise(bound), ks, trials)
    for k, (fail_o, drift_o), (fail_e, drift_e) in zip(ks, odd, even, strict=True):
        print(
            f"[Ex9 k={k}] p={q_odd}: drift={drift_o:.2f} (predicted {odd_drift_prediction(k):.2f})  "
            f"failure={fail_o:.4f} (normal {failure_normal(k, q_odd, var_u, odd_drift_prediction(k)):.4f})  "
            f"p={q}: drift={drift_e:.2f} failure={fail_e:.4f}"
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
