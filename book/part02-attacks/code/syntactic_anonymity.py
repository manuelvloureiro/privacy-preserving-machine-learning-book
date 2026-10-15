"""Check the numbers in the syntactic anonymity chapter and draw its figures.

    uv run python book/part02-attacks/code/syntactic_anonymity.py
    uv run python book/part02-attacks/code/syntactic_anonymity.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 08-syntactic-anonymity.md (they match exactly with the default
seed), then saves the chapter's figures.

Conventions: a table has n records, each with quasi-identifiers (age, ZIP code, sex) and one
sensitive attribute (a diagnosis). A generalization coarsens each quasi-identifier to a level of its
hierarchy; records with equal generalized quasi-identifiers form an equivalence class. A release
suppresses (deletes) the records of every class that fails the privacy criterion.
"""

import argparse
import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "syntactic_anonymity"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

FIGURES = Path(__file__).resolve().parent.parent / "figures"

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


# --- Toolkit: Poisson tails and the uniform-cells model ---------------------------------------


def poisson_cdf(j: int, lam: float) -> float:
    """Pr[Poisson(lam) <= j]."""
    if j < 0:
        return 0.0
    term, total = math.exp(-lam), 0.0
    for i in range(j + 1):
        total += term
        term *= lam / (i + 1)
    return total


def violating_fraction(k: int, n: float, cells: float) -> float:
    """Fraction of records in classes of size < k when n people fall in `cells` uniform cells.

    A person's class has 1 + Binomial(n - 1, 1/M) members; in the Poisson limit, 1 + Poisson(n/M).
    """
    return poisson_cdf(k - 2, n / cells)


def load_for_violation(k: int, alpha: float) -> float:
    """The people per cell, lam = n/M, at which a fraction alpha of records violate k-anonymity."""
    lo, hi = 1e-9, 1e4
    for _ in range(200):  # the violating fraction decreases in lam: bisect
        mid = math.sqrt(lo * hi)
        if poisson_cdf(k - 2, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# --- The toy release: homogeneity and background knowledge ------------------------------------


# The chapter's toy release, already generalized to 3-anonymity: (ZIP, age, diagnosis).
TOY_RELEASE: list[tuple[str, str, str]] = [
    ("130**", "< 30", "heart disease"),
    ("130**", "< 30", "heart disease"),
    ("130**", "< 30", "heart disease"),
    ("148**", "4*", "cancer"),
    ("148**", "4*", "flu"),
    ("148**", "4*", "heart disease"),
    ("130**", "3*", "cancer"),
    ("130**", "3*", "cancer"),
    ("130**", "3*", "flu"),
]


def toy_classes(rows: Sequence[tuple[str, str, str]]) -> dict[tuple[str, str], list[str]]:
    classes: dict[tuple[str, str], list[str]] = {}
    for zip_code, age, value in rows:
        classes.setdefault((zip_code, age), []).append(value)
    return classes


def posterior(values: Sequence[str], ruled_out: Sequence[str] = ()) -> dict[str, float]:
    """The adversary's posterior on the target's value: its frequency in the class, after
    deleting the values background knowledge rules out."""
    kept = [v for v in values if v not in ruled_out]
    return {v: kept.count(v) / len(kept) for v in sorted(set(kept))}


def distinct_l(counts: Sequence[int]) -> int:
    return sum(1 for c in counts if c > 0)


def entropy_l(counts: Sequence[int]) -> float:
    """2^H of the class's value distribution: the largest l for which it is entropy l-diverse."""
    total = sum(counts)
    return 2 ** -sum(c / total * math.log2(c / total) for c in counts if c > 0)


def recursive_c(counts: Sequence[int], l: int) -> float:
    """The smallest c with r_1 < c (r_l + ... + r_m), or inf if the tail is empty."""
    r = sorted((c for c in counts if c > 0), reverse=True)
    tail = sum(r[l - 1 :])
    return math.inf if tail == 0 else r[0] / tail


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q, strict=True))


def ordered_emd(p: Sequence[float], q: Sequence[float]) -> float:
    """Earth mover's distance on m ordered values at ground distance |i - j|/(m - 1)."""
    m = len(p)
    cumulative, total = 0.0, 0.0
    for a, b in zip(p[:-1], q[:-1], strict=True):
        cumulative += a - b
        total += abs(cumulative)
    return total / (m - 1)


# --- The synthetic hospital table ----------------------------------------------------------

QI_NAMES = ("age", "ZIP", "sex")
QI_RANGES = (89, 799, 1)  # each quasi-identifier's span, for normalized widths


@dataclass(frozen=True)
class Table:
    qi: IntArray  # n x 3 codes: age in years (0..89), ZIP code (0..799, ordered), sex (0, 1)
    diagnosis: IntArray  # n codes into DIAGNOSES


def normalized(weights: Sequence[float]) -> FloatArray:
    w = np.asarray(weights, dtype=np.float64)
    return w / w.sum()


def zipf(k: int, s: float) -> FloatArray:
    return normalized([1 / (j**s) for j in range(1, k + 1)])


AGE_PROBS = normalized([1.0] * 60 + [1.0 - 0.8 * (a - 59) / 30 for a in range(60, 90)])
# Each diagnosis is most common around one age: (peak age, weight). Flu is equally common at every age.
DIAGNOSIS_PEAKS = ((None, 0.25), (5, 1.0), (18, 1.0), (40, 1.0), (55, 1.0), (68, 1.0), (82, 1.0), (30, 0.05))
DIAGNOSES = ("flu", "asthma", "fracture", "depression", "diabetes", "hypertension", "cancer", "HIV")
AGE_SPREAD = 9.0  # years


def diagnosis_probs(age: int) -> FloatArray:
    """The distribution of the diagnosis at a given age: neighbours in age tend to share it."""
    return normalized(
        [w if peak is None else w * math.exp(-(((age - peak) / AGE_SPREAD) ** 2) / 2) for peak, w in DIAGNOSIS_PEAKS]
    )


def marginal_diagnosis() -> FloatArray:
    """The population's diagnosis distribution, averaging over age."""
    return np.asarray(sum(AGE_PROBS[a] * diagnosis_probs(a) for a in range(90)), dtype=np.float64)


def draw_table(rng: np.random.Generator, n: int) -> Table:
    age = rng.choice(90, size=n, p=AGE_PROBS)
    zip_code = (
        100 * rng.choice(8, size=n, p=zipf(8, 1.0))
        + 10 * rng.choice(10, size=n, p=zipf(10, 0.7))
        + rng.integers(0, 10, size=n)
    )
    sex = rng.integers(0, 2, size=n)
    diag = np.array([rng.choice(len(DIAGNOSES), p=diagnosis_probs(int(a))) for a in age], dtype=np.int64)
    return Table(np.stack([age, zip_code, sex], axis=1).astype(np.int64), diag)


def subtable(table: Table, rows: IntArray) -> Table:
    return Table(table.qi[rows], table.diagnosis[rows])


def exact_classes(table: Table) -> IntArray:
    """Classes of identical (ungeneralized) quasi-identifiers."""
    key = table.qi[:, 0] * 10_000 + table.qi[:, 1] * 10 + table.qi[:, 2]
    _, inverse = np.unique(key, return_inverse=True)
    return inverse.astype(np.int64)


# --- Privacy criteria on the diagnoses of one class -------------------------------------------

Criterion = Callable[[IntArray], bool]  # the diagnosis codes of a candidate class -> acceptable


def value_counts(diagnosis: IntArray) -> list[int]:
    return [int(c) for c in np.bincount(diagnosis, minlength=len(DIAGNOSES))]


def k_anonymous(k: int) -> Criterion:
    return lambda d: len(d) >= k


def k_and_distinct_l(k: int, l: int) -> Criterion:
    return lambda d: len(d) >= k and distinct_l(value_counts(d)) >= l


def k_and_recursive_cl(k: int, c: float, l: int) -> Criterion:
    return lambda d: len(d) >= k and recursive_c(value_counts(d), l) < c


def k_and_t_close(k: int, t: float, overall: Sequence[float]) -> Criterion:
    return lambda d: len(d) >= k and total_variation([x / len(d) for x in value_counts(d)], overall) <= t


# --- Mondrian: multidimensional generalization --------------------------------------------


def mondrian(table: Table, criterion: Criterion) -> IntArray:
    """LeFevre, DeWitt & Ramakrishnan's Mondrian: split a box of records at the median of its
    widest (normalized) quasi-identifier while both halves satisfy the criterion. Returns each
    record's class. Every class satisfies the criterion if the whole table does."""
    labels = np.zeros(len(table.diagnosis), dtype=np.int64)
    stack = [np.arange(len(table.diagnosis))]
    leaves = 0
    while stack:
        idx = stack.pop()
        spans = [
            (float(table.qi[idx, d].max() - table.qi[idx, d].min()) / QI_RANGES[d], d) for d in range(len(QI_RANGES))
        ]
        split = None
        for span, d in sorted(spans, reverse=True):
            if span == 0:
                break
            values = table.qi[idx, d]
            median = int(np.sort(values)[len(values) // 2])
            for left in (values < median, values <= median):  # ties go to one side, so boxes don't overlap
                if (
                    0 < int(left.sum()) < len(idx)
                    and criterion(table.diagnosis[idx[left]])
                    and criterion(table.diagnosis[idx[~left]])
                ):
                    split = (idx[left], idx[~left])
                    break
            if split is not None:
                break
        if split is None:
            labels[idx] = leaves
            leaves += 1
        else:
            stack.extend(split)
    return labels


def class_value_counts(cls: IntArray, diagnosis: IntArray) -> IntArray:
    """J x m matrix: how many records of each class have each diagnosis."""
    counts = np.zeros((int(cls.max()) + 1, len(DIAGNOSES)), dtype=np.int64)
    np.add.at(counts, (cls, diagnosis), 1)
    return counts


@dataclass(frozen=True)
class Risk:
    classes: int
    mean_size: float  # mean class size seen by a random record
    width: float  # mean normalized width of a record's box: 0 = exact, 1 = fully generalized
    linkage: float  # J / n: success of the linkage adversary who knows the target is in the release
    certain: float  # fraction of records in a class with one diagnosis
    guess_accuracy: float  # guessing the class's most common diagnosis
    max_tv: float  # largest total variation between a class and the whole table


def risk(table: Table, cls: IntArray) -> Risk:
    n = len(table.diagnosis)
    counts = class_value_counts(cls, table.diagnosis)
    sizes = counts.sum(axis=1)
    overall = counts.sum(axis=0) / n
    distinct = (counts > 0).sum(axis=1)
    widths = np.zeros(n)
    for d, span in enumerate(QI_RANGES):
        lo = np.full(len(sizes), np.iinfo(np.int64).max)
        hi = np.full(len(sizes), np.iinfo(np.int64).min)
        np.minimum.at(lo, cls, table.qi[:, d])
        np.maximum.at(hi, cls, table.qi[:, d])
        widths += (hi - lo)[cls] / span / len(QI_RANGES)
    return Risk(
        classes=len(sizes),
        mean_size=float(np.mean(sizes[cls])),
        width=float(widths.mean()),
        linkage=len(sizes) / n,
        certain=float(sizes[distinct == 1].sum()) / n,
        guess_accuracy=float(counts.max(axis=1).sum()) / n,
        max_tv=max(total_variation(list(row / row.sum()), list(overall)) for row in counts),
    )


def baseline_accuracy(table: Table) -> float:
    """Guessing the most common diagnosis of the whole table, for everyone."""
    return float(np.bincount(table.diagnosis).max()) / len(table.diagnosis)


def imputation_accuracy(table: Table) -> float:
    """The imputation baseline: guess each person's most likely diagnosis from their exact age,
    under the true population law. No release is involved."""
    best = np.array([int(np.argmax(diagnosis_probs(a))) for a in range(90)])
    return int(np.count_nonzero(best[table.qi[:, 0]] == table.diagnosis)) / len(table.diagnosis)


def imputation_max_posterior() -> float:
    """The largest probability the population law ever gives one diagnosis at one age."""
    return max(float(diagnosis_probs(a).max()) for a in range(90))


def size_range(cls: IntArray) -> tuple[int, int]:
    sizes = np.bincount(cls)
    return int(sizes.min()), int(sizes.max())


# --- Composition: two releases about the same person ------------------------------------------


@dataclass(frozen=True)
class TwoHospitals:
    a: Table
    b: Table
    rows_a: IntArray  # population index of each record of A
    rows_b: IntArray


def two_hospitals(rng: np.random.Generator, population: int, size: int) -> TwoHospitals:
    """Two hospitals that each treat a random `size` of the same population."""
    people = draw_table(rng, population)
    rows_a = np.sort(rng.choice(population, size=size, replace=False))
    rows_b = np.sort(rng.choice(population, size=size, replace=False))
    return TwoHospitals(subtable(people, rows_a), subtable(people, rows_b), rows_a, rows_b)


def candidates(table: Table, cls: IntArray) -> BoolArray:
    """n x m: which diagnoses appear in each record's class."""
    return class_value_counts(cls, table.diagnosis)[cls] > 0


def intersection_attack(h: TwoHospitals, criterion: Criterion) -> tuple[IntArray, IntArray, IntArray]:
    """For people treated in both hospitals: the number of diagnoses left by release A alone,
    by B alone, and by intersecting the two."""
    in_a = candidates(h.a, mondrian(h.a, criterion))
    in_b = candidates(h.b, mondrian(h.b, criterion))
    _, ia, ib = np.intersect1d(h.rows_a, h.rows_b, return_indices=True)
    left_a, left_b = in_a[ia], in_b[ib]
    count: Callable[[BoolArray], IntArray] = lambda m: m.sum(axis=1).astype(np.int64)
    return count(left_a), count(left_b), count(left_a & left_b)


def pinned_fraction(candidate_counts: IntArray) -> float:
    """The fraction of targets left with exactly one possible diagnosis."""
    return int(np.count_nonzero(candidate_counts == 1)) / len(candidate_counts)


def pinned_exact(p: Sequence[float], k1: int, k2: int) -> float:
    """Pr[the target's value is the only one in both its classes], for classes of k1 and k2 records
    whose other members draw values independently from p.

    Given the target's value v, let A be the set of other values among the k1 - 1 others of class 1.
    The answer is Σ_v p_v E[(1 - p(A))^(k2 - 1)], and Pr[A = S] comes from inclusion–exclusion:
    Pr[A = S] = Σ_{T ⊆ S} (-1)^{|S| - |T|} (p_v + p(T))^(k1 - 1).
    """
    m = len(p)
    total = 0.0
    for v in range(m):
        others = [u for u in range(m) if u != v]
        expectation = 0.0
        for size in range(len(others) + 1):
            for s in itertools.combinations(others, size):
                prob_s = sum(
                    (-1) ** (len(s) - len(t)) * (p[v] + sum(p[u] for u in t)) ** (k1 - 1)
                    for j in range(len(s) + 1)
                    for t in itertools.combinations(s, j)
                )
                expectation += prob_s * (1 - sum(p[u] for u in s)) ** (k2 - 1)
        total += p[v] * expectation
    return total


def pinned_union_bound(p: Sequence[float], k1: int, k2: int) -> float:
    """1 - E[number of other values in both classes]: a lower bound by Markov's inequality."""
    m = len(p)
    spurious = sum(
        p[v] * sum((1 - (1 - p[u]) ** (k1 - 1)) * (1 - (1 - p[u]) ** (k2 - 1)) for u in range(m) if u != v)
        for v in range(m)
    )
    return 1 - spurious


def pinned_simulated(rng: np.random.Generator, p: Sequence[float], k1: int, k2: int, trials: int) -> float:
    m = len(p)
    probs = np.asarray(p)
    v = rng.choice(m, size=trials, p=probs)
    a = np.zeros((trials, m), dtype=bool)
    b = np.zeros((trials, m), dtype=bool)
    rows = np.arange(trials)
    for _ in range(k1 - 1):
        a[rows, rng.choice(m, size=trials, p=probs)] = True
    for _ in range(k2 - 1):
        b[rows, rng.choice(m, size=trials, p=probs)] = True
    a[rows, v] = False
    return float(np.mean(~(a & b).any(axis=1)))


# --- The chapter's numbers ------------------------------------------------------------------

POPULATION = 10_000
K = 5
KS = (2, 5, 10, 20, 50)
HOSPITAL_POPULATION, HOSPITAL_SIZE = 15_000, 10_000
SKEW_POPULATION = (0.99, 0.01)  # (other, HIV) in the skewness example


def print_table(rng: np.random.Generator) -> None:
    print("== Toy release: homogeneity and background knowledge")
    for (zip_code, age), values in toy_classes(TOY_RELEASE).items():
        counts = [values.count(v) for v in sorted(set(values))]
        print(
            f"class ({zip_code}, {age}): size={len(values)}  posterior={ {v: round(x, 3) for v, x in posterior(values).items()} }  "
            f"distinct l={distinct_l(counts)}  entropy l={entropy_l(counts):.2f}"
        )
    bob = toy_classes(TOY_RELEASE)[("130**", "3*")]
    print(f"Bob, after ruling out flu: {posterior(bob, ['flu'])}")
    toy_values = [row[2] for row in TOY_RELEASE]
    labels = sorted(set(toy_values))
    overall = [toy_values.count(v) / len(toy_values) for v in labels]
    for key, values in toy_classes(TOY_RELEASE).items():
        dist = [values.count(v) / len(values) for v in labels]
        print(f"class {key}: TV to table={total_variation(dist, overall):.3f}")

    print("\n== Recursive (c, l)-diversity: the posterior bound c/(1+c)")
    for c in (1.0, 2.0, 3.0):
        print(f"c={c:.0f}: bound={c / (1 + c):.3f}")

    print("\n== Skewness: a 2-diverse class against a rare value")
    for share in (0.5, 0.1):
        cls = (1 - share, share)
        tv = total_variation(cls, SKEW_POPULATION)
        print(f"class with {share:.0%} HIV: TV={tv:.2f}  posterior ratio={share / SKEW_POPULATION[1]:.0f}")

    print("\n== Uniform cells: how many bits k-anonymity allows (99% of records in classes >= k)")
    print(f"log2 n = {math.log2(POPULATION):.2f}")
    for k in KS:
        lam = load_for_violation(k, 0.01)
        print(
            f"k={k}: people per cell={lam:.2f}  bits allowed={math.log2(POPULATION / lam):.2f}  "
            f"below log2 n by {math.log2(lam):.2f}  (log2 k={math.log2(k):.2f})"
        )

    print("\n== Synthetic hospital table, Mondrian")
    table = draw_table(rng, POPULATION)
    overall = [float(x) for x in np.bincount(table.diagnosis, minlength=len(DIAGNOSES)) / POPULATION]
    print("diagnosis frequencies: " + "  ".join(f"{d}={p:.3f}" for d, p in zip(DIAGNOSES, overall, strict=True)))
    raw = risk(table, exact_classes(table))
    print(f"n={POPULATION}  exact quasi-identifiers: linkage={raw.linkage:.3f}  certain={raw.certain:.3f}")
    print(
        f"baseline: guess the most common diagnosis={baseline_accuracy(table):.3f}  "
        f"imputation from exact age={imputation_accuracy(table):.3f}  "
        f"largest population posterior={imputation_max_posterior():.3f}"
    )
    criteria: list[tuple[str, Criterion]] = [(f"k={k}", k_anonymous(k)) for k in KS]
    criteria += [
        (f"k={K}, distinct l=3", k_and_distinct_l(K, 3)),
        (f"k={K}, recursive (c=2, l=3)", k_and_recursive_cl(K, 2.0, 3)),
        (f"k={K}, t=0.3", k_and_t_close(K, 0.3, overall)),
        (f"k={K}, t=0.15", k_and_t_close(K, 0.15, overall)),
    ]
    for name, criterion in criteria:
        cls = mondrian(table, criterion)
        s = risk(table, cls)
        lo, hi = size_range(cls)
        print(
            f"{name}: classes={s.classes}  sizes {lo}..{hi}  mean size={s.mean_size:.1f}  width={s.width:.3f}  "
            f"linkage={s.linkage:.4f}  certain={s.certain:.4f}  guess={s.guess_accuracy:.3f}  max TV={s.max_tv:.3f}"
        )

    print(f"\n== Composition: two hospitals, {HOSPITAL_SIZE} of {HOSPITAL_POPULATION} people each")
    h = two_hospitals(rng, HOSPITAL_POPULATION, HOSPITAL_SIZE)
    for name, criterion in ((f"k={K}", k_anonymous(K)), (f"k={K}, distinct l=3", k_and_distinct_l(K, 3))):
        alone_a, alone_b, both = intersection_attack(h, criterion)
        print(
            f"{name}: people in both={len(both)}  pinned by A={pinned_fraction(alone_a):.4f}  "
            f"by B={pinned_fraction(alone_b):.4f}  by A and B={pinned_fraction(both):.4f} "
            f"({int(np.count_nonzero(both == 1))} people)  "
            f"mean candidates A={alone_a.mean():.2f}  A and B={both.mean():.2f}"
        )

    print("\n== Composition model: classes drawn from the marginal")
    p = [float(x) for x in marginal_diagnosis()]
    print("marginal: " + "  ".join(f"{d}={x:.3f}" for d, x in zip(DIAGNOSES, p, strict=True)))
    for k in (2, 5, 7, 10, 20):
        exact = pinned_exact(p, k, k)
        sim = pinned_simulated(rng, p, k, k, 200_000)
        print(f"k={k}: pinned exact={exact:.4f}  simulated={sim:.4f}  union bound={pinned_union_bound(p, k, k):.4f}")


# --- Figures --------------------------------------------------------------------------------


def plot_bits() -> Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    excess = np.linspace(-8, 8, 321)
    for k, color in zip((2, 5, 10, 20), ("C0", "C1", "C2", "C3"), strict=True):
        ax.plot(excess, [violating_fraction(k, 1.0, 2.0 ** float(e)) for e in excess], color=color, label=f"k = {k}")
    ax.axhline(0.01, color="k", ls=":", lw=1)
    ax.text(-7.8, 0.03, "1% of records", fontsize=8)
    ax.set(
        xlabel=r"bits of quasi-identifier beyond $\log_2 n$",
        ylabel="fraction of records in classes smaller than k",
        title="Uniform cells: how many bits k-anonymity can keep",
    )
    ax.legend(fontsize=8, loc="center right")
    return fig


def plot_tradeoff(rng: np.random.Generator) -> Figure:
    table = draw_table(rng, POPULATION)
    ks = [2, 3, 5, 7, 10, 15, 20, 30, 50, 100]
    risks = [risk(table, mondrian(table, k_anonymous(k))) for k in ks]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.plot(ks, [r.linkage for r in risks], "C0o-", label="linkage success J / n")
    left.plot(ks, [1 / k for k in ks], "k:", label="bound 1 / k")
    left.plot(ks, [1 / (2 * k) for k in ks], "k--", lw=0.8, label="1 / (2k)")
    left.set(xscale="log", yscale="log", xlabel="k", ylabel="probability", title="Re-identification is held near 1/k")
    left.legend(fontsize=8)
    right.plot(ks, [r.guess_accuracy for r in risks], "C1o-", label="guess the class's most common diagnosis")
    right.plot(ks, [r.certain for r in risks], "C3s-", label="class has a single diagnosis (certain)")
    right.axhline(imputation_accuracy(table), color="C1", ls="--", lw=1, label="imputation from exact age (no release)")
    right.axhline(baseline_accuracy(table), color="k", ls=":", lw=1, label="guess the table's most common diagnosis")
    right.set(
        xscale="log",
        xlabel="k",
        ylabel="fraction of records",
        ylim=(-0.02, 1.02),
        title="Guessing the diagnosis: the release against the population",
    )
    right.legend(fontsize=8)
    return fig


def plot_skew() -> Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(2)
    ax.bar(x - 0.2, SKEW_POPULATION, width=0.4, color="C0", label="whole table")
    ax.bar(x + 0.2, (0.5, 0.5), width=0.4, color="C3", label="one 2-diverse class")
    ax.set_xticks(list(x), ["no HIV", "HIV"])
    ax.set(ylabel="fraction of records", ylim=(0, 1.05), title="A 2-diverse class raises HIV from 1% to 50%")
    ax.legend(fontsize=8)
    return fig


def plot_composition(rng: np.random.Generator) -> Figure:
    p = [float(x) for x in marginal_diagnosis()]
    ks = list(range(2, 26))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.plot(ks, [pinned_exact(p, k, k) for k in ks], "C0-", label="exact")
    sim_ks = [2, 4, 6, 8, 10, 15, 20, 25]
    left.plot(sim_ks, [pinned_simulated(rng, p, k, k, 20_000) for k in sim_ks], "C0o", label="simulated")
    left.plot(ks, [max(0.0, pinned_union_bound(p, k, k)) for k in ks], "C1--", label="union bound")
    left.set(
        xlabel="class size k in both releases",
        ylabel="P(diagnosis pinned down)",
        ylim=(-0.02, 1.02),
        title="Model: classes drawn from the population",
    )
    left.legend(fontsize=8)

    h = two_hospitals(rng, HOSPITAL_POPULATION, HOSPITAL_SIZE)
    alone_a, alone_b, both = intersection_attack(h, k_anonymous(K))
    bins = np.arange(1, len(DIAGNOSES) + 2) - 0.5
    for sizes, color, name in (
        (alone_a, "C0", "hospital A alone"),
        (alone_b, "C2", "hospital B alone"),
        (both, "C3", "A and B intersected"),
    ):
        right.hist(sizes, bins=bins, histtype="step", lw=2, color=color, density=True, label=name)
    right.set(
        xlabel="diagnoses still possible for the target",
        ylabel="fraction of targets",
        title=f"Two {K}-anonymous hospital releases",
    )
    right.legend(fontsize=8)
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
        "anon_bits.svg": plot_bits(),
        "anon_tradeoff.svg": plot_tradeoff(rng),
        "anon_skew.svg": plot_skew(),
        "anon_composition.svg": plot_composition(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
