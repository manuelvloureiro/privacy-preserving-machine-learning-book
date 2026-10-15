"""Check the reconstruction attacks of the reconstruction chapter and draw its figures.

    uv run python code/part02-attacks/reconstruction.py
    uv run python code/part02-attacks/reconstruction.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 09-reconstruction.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions: the secret database is a bit vector d in {0,1}^n. A subset-sum query is a 0/1 vector
q of length n and its true answer is <q, d>. A query matrix A has one query per row. The curator
returns a = A d + e, where e is its error. Logarithms are natural unless a name says log2.
"""

import argparse
import itertools
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# The matrices here are small; one BLAS thread is fastest and avoids oversubscription on a busy machine.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "reconstruction"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def normal_quantile(p: float) -> float:
    """Φ⁻¹(p) by bisection on normal_cdf (accurate to about 1e-12)."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def binary_entropy(p: float) -> float:
    """h(p) in bits."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


# --- Subset-sum queries ----------------------------------------------------------------------


def random_bits(rng: np.random.Generator, n: int) -> IntArray:
    """A secret database: n independent fair bits."""
    return rng.integers(0, 2, size=n).astype(np.int64)


def random_queries(rng: np.random.Generator, m: int, n: int) -> FloatArray:
    """m random subsets of [n], each record included independently with probability 1/2."""
    return rng.integers(0, 2, size=(m, n)).astype(np.float64)


def all_subsets(n: int) -> IntArray:
    """Every subset of [n] as a row of a 2^n × n 0/1 matrix (row k is the binary expansion of k)."""
    k = np.arange(2**n, dtype=np.int64)
    return ((k[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1).astype(np.int64)


def hamming(a: IntArray, b: IntArray) -> int:
    """Number of positions where two bit vectors differ."""
    return int(np.count_nonzero(np.not_equal(a, b)))


def fraction_agreeing(a: IntArray, b: IntArray) -> float:
    """Fraction of positions where two bit vectors agree."""
    return 1 - hamming(a, b) / len(a)


def row_distances(rows: IntArray, d: IntArray) -> IntArray:
    """Hamming distance from each row to d."""
    return np.count_nonzero(np.not_equal(rows, d[None, :]), axis=1).astype(np.int64)


# --- The exponential attack ------------------------------------------------------------------


def max_subset_discrepancy(z: IntArray) -> int:
    """max over subsets S of |Σ_{i∈S} z_i| for an integer vector z: the larger of the positive and negative mass."""
    return int(max(z[z > 0].sum(), -z[z < 0].sum()))


def exponential_attack_worst_case(d: IntArray, error: float) -> tuple[int, int]:
    """All candidates c ∈ {0,1}^n that some curator with error ≤ E could make look consistent.

    c survives iff every subset sum of c is within 2E of the same subset sum of d. Checked by brute force
    over all 2^n subsets. Returns (number of surviving candidates, largest Hamming distance to d).
    """
    n = len(d)
    candidates = all_subsets(n)  # every bit vector, as rows
    subsets = all_subsets(n)
    gaps = np.abs((candidates - d[None, :]) @ subsets.T)  # |Σ_S (c - d)| for every (c, S)
    alive = np.all(gaps <= 2 * error + 1e-9, axis=1)
    distances = row_distances(candidates[alive], d)
    return int(alive.sum()), int(distances.max())


def rounding_curator_attack(d: IntArray, width: int) -> tuple[int, int]:
    """The exponential attack against a curator that rounds every subset sum to a multiple of `width`.

    The attacker keeps every c whose subset sums all round to the published answers.
    Returns (number of consistent candidates, largest Hamming distance to d).
    """
    n = len(d)
    candidates = all_subsets(n)
    subsets = all_subsets(n)

    def publish(x: IntArray) -> IntArray:
        return (np.floor((x @ subsets.T) / width + 0.5) * width).astype(np.int64)

    published = publish(d[None, :])
    alive = np.all(publish(candidates) == published, axis=1)
    distances = row_distances(candidates[alive], d)
    return int(alive.sum()), int(distances.max())


# --- The linear-programming decoder: anti-concentration and the union bound -------------------


def rademacher_fourth_moment(a: Sequence[float]) -> float:
    """E[(Σ s_i a_i)^4] for independent fair signs s_i, by enumerating all 2^k sign patterns."""
    total = 0.0
    for signs in itertools.product((-1.0, 1.0), repeat=len(a)):
        total += sum(s * x for s, x in zip(signs, a, strict=True)) ** 4
    return total / 2 ** len(a)


def rademacher_fourth_moment_formula(a: Sequence[float]) -> float:
    """3(Σ a²)² − 2 Σ a⁴."""
    s2 = sum(x * x for x in a)
    return 3 * s2**2 - 2 * sum(x**4 for x in a)


def paley_zygmund_escape(theta: float) -> float:
    """Lower bound (1 − θ)²/3 on Pr[⟨q, z⟩² > θ E⟨q, z⟩²] for a random subset q."""
    return (1 - theta) ** 2 / 3


def escape_threshold(n: int, beta: float, theta: float = 0.25) -> float:
    """√θ · √(βn)/6: a query escapes the band when |⟨q, z⟩| exceeds this (z has βn entries ≥ 1/3)."""
    return math.sqrt(theta) * math.sqrt(beta * n) / 6


def escape_probability(rng: np.random.Generator, z: FloatArray, threshold: float, trials: int) -> float:
    """Monte Carlo Pr[|⟨q, z⟩| > threshold] for a random subset q, in batches of 5000 queries."""
    hits = 0
    for start in range(0, trials, 5000):
        q = rng.integers(0, 2, size=(min(5000, trials - start), len(z))).astype(np.float64)
        hits += int(np.sum(np.abs(q @ z) > threshold))
    return hits / trials


def bad_difference(n: int, beta: float, balanced: bool) -> FloatArray:
    """A difference vector z = c − d with βn entries of size 1/3 and the rest 0.

    balanced=True gives half of them +1/3 and half −1/3 (so E⟨q, z⟩ = 0, the hard case);
    balanced=False makes them all +1/3.
    """
    k = int(round(beta * n))
    z = np.zeros(n)
    z[:k] = 1 / 3
    if balanced:
        z[: k // 2] = -1 / 3
    return z


def lp_queries_needed(n: int, eta: float, escape: float) -> int:
    """Smallest whole m with (n+1)^n (1 − escape)^m ≤ η: the union bound over the grid of step 1/n."""
    return math.ceil((n * math.log(n + 1) + math.log(1 / eta)) / math.log(1 / (1 - escape)))


def lp_error_tolerance(n: int, beta: float) -> float:
    """Largest E with 2E + 1/2 ≤ √(βn)/12, the condition of the decoder's guarantee."""
    return math.sqrt(beta * n) / 24 - 0.25


# --- Least squares ---------------------------------------------------------------------------


def least_squares_estimate(queries: FloatArray, answers: FloatArray) -> FloatArray:
    """The least-squares solution of A x ≈ a, before rounding, from the normal equations AᵀA x = Aᵀa."""
    return np.asarray(np.linalg.solve(queries.T @ queries, queries.T @ answers), dtype=np.float64)


def round_bits(x: FloatArray) -> IntArray:
    """Clip to [0, 1] and round to the nearest bit."""
    return (np.clip(x, 0.0, 1.0) > 0.5).astype(np.int64)


def least_squares_decode(queries: FloatArray, answers: FloatArray) -> IntArray:
    return round_bits(least_squares_estimate(queries, answers))


def ls_coordinate_sd(n: int, m: int, sigma: float) -> float:
    """Standard deviation of one coordinate of the least-squares error under N(0, σ²) noise.

    The error is (AᵀA)⁻¹Aᵀe, with covariance σ²(AᵀA)⁻¹. A row q of A has second-moment matrix
    Σ = (I + 11ᵀ)/4, whose inverse 4(I − 11ᵀ/(n + 1)) has diagonal 4n/(n + 1). The inverse-Wishart
    mean E[(AᵀA)⁻¹] ≈ Σ⁻¹/(m − n − 1) then gives the variance 4σ²n/((n + 1)(m − n − 1)).
    """
    return 2 * sigma * math.sqrt(n / ((n + 1) * (m - n - 1)))


def ls_predicted_fraction(n: int, m: int, sigma: float) -> float:
    """Predicted fraction of bits least squares recovers: a coordinate rounds wrong when its error passes 1/2.

    The error is one-sided for each bit (a 0 goes wrong above 1/2, a 1 below 1/2), so the fraction is
    Φ(1/(2·sd)). Below m = n + 2 the prediction is undefined; return 1/2 (no better than guessing).
    """
    if m <= n + 1:
        return 0.5
    return normal_cdf(0.5 / ls_coordinate_sd(n, m, sigma))


def ls_recovery(
    rng: np.random.Generator, n: int, ms: Sequence[int], sigmas: Sequence[float], reps: int
) -> list[list[float]]:
    """Simulated fraction of bits recovered by least squares, for each number of queries m and noise σ.

    Each repetition draws one database, one query matrix with max(ms) rows and one noise vector per σ;
    the experiment with m queries uses their first m rows. Returns fractions[i][k] for ms[i], sigmas[k].
    """
    totals = [[0.0] * len(sigmas) for _ in ms]
    for _ in range(reps):
        d = random_bits(rng, n)
        a_full = random_queries(rng, max(ms), n)
        exact = a_full @ d
        noise = [rng.normal(0.0, sigma, size=max(ms)) for sigma in sigmas]
        for i, m in enumerate(ms):
            a_matrix = a_full[:m]
            ls_map = np.linalg.solve(a_matrix.T @ a_matrix, a_matrix.T)  # (AᵀA)⁻¹Aᵀ
            for k in range(len(sigmas)):
                estimate = ls_map @ (exact[:m] + noise[k][:m])
                totals[i][k] += fraction_agreeing(round_bits(estimate), d)
    return [[t / reps for t in row] for row in totals]


def queries_for_recovery(fraction: float, sigma_over_sqrt_n: float) -> float:
    """m/n at which the large-n prediction Φ(√(m − n)/(4σ)) reaches `fraction`, for σ = c√n."""
    z = normal_quantile(fraction)
    return 1 + (4 * sigma_over_sqrt_n * z) ** 2


def smallest_singular_value(matrix: FloatArray) -> float:
    """√(smallest eigenvalue of AᵀA)."""
    return float(math.sqrt(np.linalg.eigvalsh(matrix.T @ matrix)[0]))


def worst_case_wrong_bits(m: int, error: float, smin: float) -> float:
    """Bound 4‖e‖²/s_min² ≤ 4 m E²/s_min² on the bits least squares gets wrong under any error |e_j| ≤ E."""
    return 4 * m * error**2 / smin**2


def adversarial_ls_wrong_bits(queries: FloatArray, d: IntArray, error: float) -> int:
    """A curator with error ≤ E that pushes least squares along its weakest direction.

    It adds e = E·sign(A v), with v the right singular vector of the smallest singular value.
    """
    _, vectors = np.linalg.eigh(queries.T @ queries)
    v = vectors[:, 0]
    e = error * np.sign(queries @ v)
    return hamming(least_squares_decode(queries, queries @ d + e), d)


# --- The Fundamental Law: a sampling curator and an information bound -------------------------


def sampling_curator(
    rng: np.random.Generator, queries: FloatArray, d: IntArray, keep: float
) -> tuple[FloatArray, BoolArray]:
    """Answer every query from a secret random sample: keep each record w.p. `keep`, rescale by 1/keep."""
    kept = rng.random(len(d)) < keep
    answers = queries @ (d * kept) / keep
    return np.asarray(answers, dtype=np.float64), kept


def hoeffding_max_error(n: int, m: int, eta: float, keep: float) -> float:
    """With probability ≥ 1 − η, all m sampled answers are within √(n ln(2m/η) / (2 keep²)) of the truth."""
    return math.sqrt(n * math.log(2 * m / eta) / (2 * keep**2))


def information_min_queries(n: int, beta: float, values: int) -> float:
    """Queries needed to recover all but βn bits when each answer takes at most `values` values."""
    return n * (1 - binary_entropy(beta)) / math.log2(values)


def rounded_answer_values(n: int, width: int) -> int:
    """Number of possible answers when a count in {0, ..., n} is rounded to a multiple of `width`."""
    return n // width + 2


# --- A census block --------------------------------------------------------------------------

AGE_MAX = 99
MARRIED_MIN_AGE = 18

Person = tuple[str, str, str, int]  # (sex F/M, race B/W, married Y/N, age)

# A fictional block of seven people, in the style of Garfinkel, Abowd & Martindale (2019).
BLOCK: list[Person] = [
    ("F", "B", "N", 7),
    ("M", "B", "N", 15),
    ("F", "W", "N", 27),
    ("M", "W", "Y", 33),
    ("F", "W", "Y", 35),
    ("F", "B", "Y", 52),
    ("M", "B", "Y", 71),
]
GROUPS: list[str] = ["total", "female", "male", "black", "white", "married"]
ATTRIBUTE: dict[str, tuple[int, str]] = {
    "female": (0, "F"),
    "male": (0, "M"),
    "black": (1, "B"),
    "white": (1, "W"),
    "married": (2, "Y"),
    "single": (2, "N"),
}
PARTITIONS: list[tuple[str, str]] = [("female", "male"), ("black", "white"), ("married", "single")]


@dataclass(frozen=True)
class GroupStats:
    """One row of the published table. Medians are doubled so they stay integers; None means withheld."""

    count: int
    median2: int | None
    total: int | None  # sum of ages, i.e. mean × count


def doubled_median(ages: Sequence[int]) -> int:
    s = sorted(ages)
    k = len(s)
    return 2 * s[k // 2] if k % 2 else s[k // 2 - 1] + s[k // 2]


def in_group(person: Person, group: str) -> bool:
    if group == "total":
        return True
    position, value = ATTRIBUTE[group]
    return person[position] == value


def publish(block: Sequence[Person]) -> dict[str, GroupStats]:
    """The block's published table: count, median age and mean age (as a sum) for each group."""
    table: dict[str, GroupStats] = {}
    for group in GROUPS:
        ages = [p[3] for p in block if in_group(p, group)]
        table[group] = GroupStats(len(ages), doubled_median(ages), sum(ages))
    return table


def withhold(table: dict[str, GroupStats], group: str, statistic: str) -> dict[str, GroupStats]:
    """A copy of the table with one statistic ("median" or "mean") of one group suppressed."""
    out = dict(table)
    row = out[group]
    out[group] = GroupStats(
        row.count, None if statistic == "median" else row.median2, None if statistic == "mean" else row.total
    )
    return out


def sorted_age_tuples(count: int, total: int | None, median2: int | None, low: int) -> IntArray:
    """Every non-decreasing tuple of `count` ages in [low, AGE_MAX] with the given sum and doubled median."""
    grids = (
        np.meshgrid(*[np.arange(low, AGE_MAX + 1)] * count, indexing="ij", sparse=False)
        if total is None
        else np.meshgrid(*[np.arange(low, AGE_MAX + 1)] * (count - 1), indexing="ij", sparse=False)
    )
    cols = [g.ravel() for g in grids]
    if total is not None:
        last = total - sum(cols, start=np.zeros_like(cols[0]))
        cols.append(last)
    tuples = np.stack(cols, axis=1).astype(np.int64)
    ok = np.all(np.diff(tuples, axis=1) >= 0, axis=1) & (tuples[:, -1] <= AGE_MAX) & (tuples[:, 0] >= low)
    tuples = tuples[ok]
    if median2 is not None:
        k = count
        med = 2 * tuples[:, k // 2] if k % 2 else tuples[:, k // 2 - 1] + tuples[:, k // 2]
        tuples = tuples[med == median2]
    return tuples


def side_stats(table: dict[str, GroupStats]) -> dict[str, GroupStats]:
    """Stats for both sides of every partition, filling in a complement's count and sum from the total."""
    total = table["total"]
    out: dict[str, GroupStats] = {}
    for a, b in PARTITIONS:
        ra = table[a]
        rb = table.get(b, GroupStats(total.count - ra.count, None, None))
        sum_a, sum_b = ra.total, rb.total
        if total.total is not None:
            if sum_b is None and sum_a is not None:
                sum_b = total.total - sum_a
            if sum_a is None and sum_b is not None:
                sum_a = total.total - sum_b
        out[a] = GroupStats(ra.count, ra.median2, sum_a)
        out[b] = GroupStats(rb.count, rb.median2, sum_b)
    return out


def valid_splits(ages: IntArray, stats_a: GroupStats, stats_b: GroupStats, low_a: int) -> BoolArray:
    """For each sorted age vector (row) and each way to pick |a| of the 7 people, whether the split fits.

    Returns a (rows × combinations) boolean matrix; combinations are itertools.combinations(range(7), |a|).
    """
    n = ages.shape[1]
    combos = list(itertools.combinations(range(n), stats_a.count))
    result = np.ones((ages.shape[0], len(combos)), dtype=np.bool_)
    for j, idx in enumerate(combos):
        rest = [i for i in range(n) if i not in idx]
        part_a, part_b = ages[:, list(idx)], ages[:, rest]
        ok = part_a[:, 0] >= low_a
        for part, stats in ((part_a, stats_a), (part_b, stats_b)):
            if stats.total is not None:
                ok &= part.sum(axis=1) == stats.total
            if stats.median2 is not None:
                k = part.shape[1]
                med = 2 * part[:, k // 2] if k % 2 else part[:, k // 2 - 1] + part[:, k // 2]
                ok &= med == stats.median2
        result[:, j] = ok
    return result


def consistent_blocks(
    table: dict[str, GroupStats], trace: list[tuple[str, int]] | None = None
) -> set[tuple[Person, ...]]:
    """Every block of people (as a sorted tuple) whose published table equals `table`.

    Enumerates the age vectors of one partition with known medians, then checks every split of the
    seven people for each partition and assembles the blocks. Married people must be at least 18.
    If `trace` is given, the size of the search space after each step is appended to it.
    """
    stats = side_stats(table)
    a, b = max(PARTITIONS, key=lambda p: (stats[p[0]].median2 is not None) + (stats[p[1]].median2 is not None))
    left = sorted_age_tuples(stats[a].count, stats[a].total, stats[a].median2, MARRIED_MIN_AGE if a == "married" else 0)
    right = sorted_age_tuples(stats[b].count, stats[b].total, stats[b].median2, 0)
    pairs = np.concatenate([np.repeat(left, len(right), axis=0), np.tile(right, (len(left), 1))], axis=1)
    ages = np.unique(np.sort(pairs, axis=1), axis=0)
    if trace is not None:
        trace += [(f"{a} age tuples", len(left)), (f"{b} age tuples", len(right)), ("age vectors", len(ages))]
    total = table["total"]
    if total.median2 is not None:
        ages = ages[2 * ages[:, ages.shape[1] // 2] == total.median2]  # seven people: the median is the 4th smallest
        if trace is not None:
            trace.append(("age vectors with the total median", len(ages)))
    n = ages.shape[1]
    split_ok = [valid_splits(ages, stats[p], stats[q], MARRIED_MIN_AGE if p == "married" else 0) for p, q in PARTITIONS]
    combos = [list(itertools.combinations(range(n), stats[p].count)) for p, _ in PARTITIONS]
    alive = np.all([s.any(axis=1) for s in split_ok], axis=0)
    if trace is not None:
        for k, (p, _) in enumerate(PARTITIONS):
            trace.append((f"age vectors with a valid {p} split", int(split_ok[k].any(axis=1).sum())))
        trace.append(("age vectors with valid splits for all three", int(alive.sum())))
    blocks: set[tuple[Person, ...]] = set()
    labels = [("F", "M"), ("B", "W"), ("Y", "N")]
    for r in np.flatnonzero(alive):
        options = [[set(combos[k][j]) for j in np.flatnonzero(split_ok[k][r])] for k in range(3)]
        for s1, s2, s3 in itertools.product(*options):
            people = [
                (labels[0][i not in s1], labels[1][i not in s2], labels[2][i not in s3], int(ages[r, i]))
                for i in range(n)
            ]
            blocks.add(tuple(sorted(people)))
    return blocks


def attribute_pattern(block: Sequence[Person]) -> tuple[tuple[str, str, str], ...]:
    """The block without ages: the sorted list of (sex, race, married) of its people."""
    return tuple(sorted(p[:3] for p in block))


def pattern_summary(blocks: set[tuple[Person, ...]], truth: Sequence[Person]) -> tuple[int, float]:
    """(number of distinct attribute patterns among the blocks, share of blocks with the true pattern)."""
    patterns = [attribute_pattern(b) for b in blocks]
    return len(set(patterns)), patterns.count(attribute_pattern(truth)) / len(patterns)


def withheld_counts(table: dict[str, GroupStats]) -> list[tuple[str, int]]:
    """Number of consistent blocks when each single median or mean is withheld."""
    rows: list[tuple[str, int]] = []
    for group in GROUPS:
        for statistic in ("median", "mean"):
            rows.append((f"{group} {statistic}", len(consistent_blocks(withhold(table, group, statistic)))))
    return rows


# --- Table -----------------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Fact 1: subset sums are inner products ==")
    d4 = np.array([1, 0, 1, 1], dtype=np.int64)
    a4 = all_subsets(4)
    print(f"  d={d4.tolist()}  answers to the 16 subsets: {(a4 @ d4).tolist()}")

    print("== Exponential attack (n = 10, all 1024 subsets) ==")
    d10 = random_bits(rng, 10)
    print(f"  d={d10.tolist()}  ones={int(d10.sum())}")
    for error in (0.0, 0.5, 1.0, 1.5, 2.0):
        count, dist = exponential_attack_worst_case(d10, error)
        print(f"  E={error:g}: consistent candidates={count}  max Hamming distance={dist}  bound 4E={4 * error:g}")
    for width in (2, 4):
        count, dist = rounding_curator_attack(d10, width)
        print(f"  rounding to multiples of {width}: consistent candidates={count}  max Hamming distance={dist}")

    print("== Anti-concentration (Paley-Zygmund) ==")
    print(
        f"  E[(s1+s2+s3)^4]: enumeration={rademacher_fourth_moment([1, 1, 1]):.0f}  "
        f"formula={rademacher_fourth_moment_formula([1, 1, 1]):.0f}"
    )
    print(
        f"  escape bound at theta=1/4: {paley_zygmund_escape(0.25):.4f}  at theta=1/9: {paley_zygmund_escape(1 / 9):.4f}"
    )
    n, beta = 1000, 0.1
    thr = escape_threshold(n, beta)
    print(f"  n={n} beta={beta}: threshold sqrt(beta n)/12={thr:.4f}")
    for balanced in (False, True):
        z = bad_difference(n, beta, balanced)
        p = escape_probability(rng, z, thr, 100_000)
        print(f"  {'balanced' if balanced else 'aligned'} z: Pr[|<q,z>| > threshold]={p:.4f}  (bound 0.1875)")
    print(
        f"  LP decoder guarantee at n={n}, beta={beta}: error tolerance E <= {lp_error_tolerance(n, beta):.4f}  "
        f"queries m >= {lp_queries_needed(n, 0.01, 3 / 16)} ({lp_queries_needed(n, 0.01, 3 / 16) / n:.1f} n)"
    )

    print("== Least squares under Gaussian noise (n = 500, m = 4n) ==")
    n, m = 500, 2000
    ratios = [0.1, 0.25, 0.5, 1.0, 2.0]
    simulated = ls_recovery(rng, n, [m], [c * math.sqrt(n) for c in ratios], reps=4)[0]
    for c, sim in zip(ratios, simulated, strict=True):
        pred = ls_predicted_fraction(n, m, c * math.sqrt(n))
        print(f"  sigma={c:g} sqrt(n)={c * math.sqrt(n):.2f}: recovered={sim:.4f}  predicted={pred:.4f}")
    for c in (2.0, 5.0):
        print(f"  sigma={c:g} sqrt(n): m/n for 90% recovery={queries_for_recovery(0.9, c):.1f}")

    print("== Least squares under worst-case error (n = 500, m = 4n) ==")
    d = random_bits(rng, n)
    a_matrix = random_queries(rng, m, n)
    smin = smallest_singular_value(a_matrix)
    print(f"  smallest singular value={smin:.2f}  (sqrt(m) - sqrt(n))/2={(math.sqrt(m) - math.sqrt(n)) / 2:.2f}")
    for error in (1.0, 2.0, 0.25 * math.sqrt(n)):
        bound = worst_case_wrong_bits(m, error, smin)
        print(
            f"  E={error:.2f}: guaranteed wrong bits <= {bound:.1f}  "
            f"adversarial curator gets {adversarial_ls_wrong_bits(a_matrix, d, error)} wrong"
        )

    print("== Sampling curator (n = 500, m = 2000, keep = 1/2) ==")
    answers, kept = sampling_curator(rng, a_matrix, d, 0.5)
    err = float(np.max(np.abs(answers - a_matrix @ d)))
    guess = least_squares_decode(a_matrix, answers)
    print(
        f"  kept {int(kept.sum())} of {n}  max error={err:.2f} ({err / math.sqrt(n):.2f} sqrt(n))  "
        f"Hoeffding bound (eta=0.05)={hoeffding_max_error(n, m, 0.05, 0.5):.2f}  "
        f"recovered={fraction_agreeing(guess, d):.4f}  ceiling 3/4"
    )
    print(f"  keep=0.9: Hoeffding bound={hoeffding_max_error(n, m, 0.05, 0.9):.2f}  ceiling={0.9 + 0.1 / 2:.2f}")

    print("== Information bound ==")
    n = 1000
    for beta in (0.05, 0.01):
        print(
            f"  n={n} beta={beta}: h(beta)={binary_entropy(beta):.4f}  exact answers m >= "
            f"{information_min_queries(n, beta, n + 1):.1f}"
        )
    for width in (10, 32):
        v = rounded_answer_values(n, width)
        print(f"  n={n} beta=0.05 rounded to {width}: values={v}  m >= {information_min_queries(n, 0.05, v):.1f}")
    v = rounded_answer_values(n, 10)
    print(f"  n={n} beta=0.01 rounded to 10: m >= {information_min_queries(n, 0.01, v):.1f}")

    print("== Census block ==")
    table = publish(BLOCK)
    for group, row in table.items():
        assert row.median2 is not None and row.total is not None
        print(f"  {group}: count={row.count}  median={row.median2 / 2:g}  mean={row.total / row.count:.2f}")
    trace: list[tuple[str, int]] = []
    blocks = consistent_blocks(table, trace)
    print("  search: " + "  ".join(f"{name}={size}" for name, size in trace))
    print(
        f"  consistent blocks with the full table: {len(blocks)}  "
        f"equal to the true block: {tuple(sorted(BLOCK)) in blocks}"
    )
    for group in GROUPS:
        for statistic in ("median", "mean"):
            found = consistent_blocks(withhold(table, group, statistic))
            patterns, share = pattern_summary(found, BLOCK)
            print(
                f"  withhold {group} {statistic}: {len(found)} consistent blocks, {patterns} attribute patterns, "
                f"true pattern in {share:.2f} of them"
            )


# --- Figures ---------------------------------------------------------------------------------


def plot_exponential(rng: np.random.Generator) -> Figure:
    d = random_bits(rng, 10)
    errors = [k / 4 for k in range(11)]
    results = [exponential_attack_worst_case(d, e) for e in errors]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    left.plot(errors, [r[1] for r in results], "ko-", label="largest distance of a consistent candidate")
    left.plot(errors, [min(4 * e, 10) for e in errors], "C1--", label="bound 4E")
    left.set(xlabel="curator error E", ylabel="Hamming distance to d", title="n = 10, all 1024 subset sums")
    left.legend(fontsize=9)
    right.semilogy(errors, [r[0] for r in results], "C0o-")
    right.axhline(2**10, color="gray", linestyle=":", label="all $2^{10}$ candidates")
    right.set(xlabel="curator error E", ylabel="consistent candidates", title="How many databases survive")
    right.legend(fontsize=9)
    return fig


def plot_anticoncentration(rng: np.random.Generator, n: int = 1000, beta: float = 0.1) -> Figure:
    thr = escape_threshold(n, beta)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, balanced in zip(axes, (True, False), strict=True):
        z = bad_difference(n, beta, balanced)
        q = rng.integers(0, 2, size=(20_000, n)).astype(np.float64)
        values = q @ z
        p = float(np.mean(np.abs(values) > thr))
        bins = np.arange(values.min() - 1 / 6, values.max() + 1 / 3, 1 / 3)
        ax.hist(values, bins=bins, color="C0", alpha=0.7, density=True)
        ax.axvspan(-thr, thr, color="C3", alpha=0.25, label=f"band |⟨q, z⟩| ≤ {thr:.2f}")
        name = "balanced: half +1/3, half −1/3" if balanced else "aligned: all +1/3"
        ax.set(xlabel="⟨q, z⟩ for a random subset q", title=f"{name}\nescapes the band w.p. {p:.3f} (bound 0.188)")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("density")
    return fig


def plot_ls_noise(rng: np.random.Generator) -> Figure:
    ratios = [0.05 * k for k in range(1, 41)]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for color, n in zip(("C0", "C1", "C2"), (100, 300, 900), strict=True):
        sims = ls_recovery(rng, n, [4 * n], [c * math.sqrt(n) for c in ratios], reps=2)[0]
        ax.plot(ratios, sims, "o", color=color, markersize=3.5, label=f"simulated, n = {n}, m = 4n")
    big = 10**6
    pred = [ls_predicted_fraction(big, 4 * big, c * math.sqrt(big)) for c in ratios]
    ax.plot(ratios, pred, "k-", label=r"prediction $\Phi(\sqrt{m-n}\,/\,(4\sigma))$")
    ax.axhline(0.5, color="gray", linestyle=":", label="random guessing")
    ax.set(
        xlabel=r"noise scale $\sigma/\sqrt{n}$",
        ylabel="fraction of bits recovered",
        ylim=(0.45, 1.01),
        title="Least squares from m = 4n noisy random subset sums",
    )
    ax.legend(fontsize=8)
    return fig


def plot_ls_queries(rng: np.random.Generator, n: int = 200) -> Figure:
    factors = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0]
    ratios = [0.5, 1.0, 2.0]
    ms = [int(f * n) for f in factors]
    sims = ls_recovery(rng, n, ms, [c * math.sqrt(n) for c in ratios], reps=2)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    fine = np.geomspace(1.0, 64.0, 200)
    for k, (color, c) in enumerate(zip(("C0", "C1", "C2"), ratios, strict=True)):
        ax.plot(
            factors,
            [row[k] for row in sims],
            "o",
            color=color,
            markersize=4,
            label=rf"simulated, $\sigma = {c:g}\sqrt{{n}}$",
        )
        ax.plot(
            fine, [ls_predicted_fraction(n, int(f * n), c * math.sqrt(n)) for f in fine], "-", color=color, linewidth=1
        )
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.set(
        xscale="log",
        xlabel="queries per record m/n",
        ylabel="fraction of bits recovered",
        ylim=(0.45, 1.01),
        title=f"More queries average the noise away (n = {n}; lines: prediction)",
    )
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64], labels=["1", "2", "4", "8", "16", "32", "64"])
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_census(rows: Sequence[tuple[str, int]]) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    colors = ["C3" if c > 1 else "C0" for c in counts]
    ax.bar(range(len(rows)), counts, color=colors)
    ax.axhline(1, color="k", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylim(0.5, max(counts) * 2)
    ax.set_xticks(range(len(rows)), labels=names, rotation=60, ha="right", fontsize=8)
    for i, c in enumerate(counts):
        ax.text(i, c * 1.1, str(c), ha="center", va="bottom", fontsize=8)
    ax.set(ylabel="blocks consistent with the table", title="Withholding one of the twelve medians and means")
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
        "recon_exponential.svg": plot_exponential(rng),
        "recon_anticoncentration.svg": plot_anticoncentration(rng),
        "recon_ls_noise.svg": plot_ls_noise(rng),
        "recon_ls_queries.svg": plot_ls_queries(rng),
        "recon_census.svg": plot_census(withheld_counts(publish(BLOCK))),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
