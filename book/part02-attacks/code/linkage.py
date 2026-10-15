"""Check the numbers in the linkage and re-identification chapter and draw its figures.

    uv run python book/part02-attacks/code/linkage.py
    uv run python book/part02-attacks/code/linkage.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 07-linkage-reidentification.md (they match exactly with the default
seed), then saves the chapter's figures.

Conventions: a population has n people; each person has a combination of quasi-identifier values,
called a cell. A cell has probability p under the model that generates people independently.
A person is unique when nobody else shares their cell. Logarithms are base 2 for bits.
"""

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "linkage"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

FIGURES = Path(__file__).resolve().parent.parent / "figures"

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


# --- Attributes of the synthetic population --------------------------------------------------


@dataclass(frozen=True)
class Attribute:
    name: str
    probs: tuple[float, ...]


def normalized(weights: Sequence[float]) -> tuple[float, ...]:
    total = sum(weights)
    return tuple(w / total for w in weights)


def zipf(k: int, s: float) -> tuple[float, ...]:
    """Probabilities proportional to 1/j^s for j = 1, ..., k: a few common values, many rare ones."""
    return normalized([1 / (j**s) for j in range(1, k + 1)])


def geometric(k: int, r: float) -> tuple[float, ...]:
    return normalized([r**j for j in range(k)])


def uniform(k: int) -> tuple[float, ...]:
    return tuple([1 / k] * k)


# The synthetic census, in the order the attributes are revealed. Marginals are made up but shaped
# like real ones: a skewed region, a flat birth month, a skewed occupation, real-ish blood groups.
ATTRIBUTES: list[Attribute] = [
    Attribute("sex", uniform(2)),
    Attribute("age band", normalized([1.0] * 12 + [0.8, 0.6, 0.4, 0.2])),
    Attribute("region", zipf(40, 1.0)),
    Attribute("birth month", uniform(12)),
    Attribute("marital status", (0.45, 0.35, 0.12, 0.08)),
    Attribute("children", geometric(6, 0.5)),
    Attribute("occupation", zipf(20, 0.8)),
    Attribute("blood group", (0.374, 0.357, 0.085, 0.066, 0.063, 0.034, 0.015, 0.006)),
]


def entropy_bits(probs: Sequence[float]) -> float:
    """Shannon entropy in bits."""
    return -sum(p * math.log2(p) for p in probs if p > 0)


def collision_probability(probs: Sequence[float]) -> float:
    """Σ p²: the chance that two independent people share a value."""
    return sum(p * p for p in probs)


# --- Linkage: success of the adversary who picks a random matching record --------------------


def linkage_success(class_sizes: Sequence[int]) -> float:
    """Average over a random target of 1/(size of the target's class) = (number of classes)/n."""
    n = sum(class_sizes)
    return sum(k * (1 / k) for k in class_sizes) / n


# --- Uniqueness: the birthday-style calculation ----------------------------------------------


def uniform_unique_fraction(n: float, cells: float) -> float:
    """Expected fraction of n people who are unique when each falls in one of `cells` cells uniformly."""
    return math.exp((n - 1) * math.log1p(-1 / cells))


def poisson_unique_fraction(n: float, cells: float) -> float:
    """The approximation e^{-n/M}."""
    return math.exp(-n / cells)


def bits_for_unique_fraction(n: float, fraction: float) -> float:
    """Bits b = log2 M such that e^{-n/M} = fraction under uniform cells."""
    return math.log2(n) - math.log2(math.log(1 / fraction))


def compressed_cells(attributes: Sequence[Attribute]) -> tuple[FloatArray, FloatArray]:
    """The distinct cell probabilities and how many cells have each, for independent attributes.

    The cells are the outer product of the attributes' values. Many cells share a probability
    (every birth month is equally likely), and the formulas below only need each distinct
    probability once, with its multiplicity, so merging them keeps the sums fast.
    """
    probs, mult = np.ones(1), np.ones(1)
    for a in attributes:
        values, counts = np.unique(np.array(a.probs), return_counts=True)
        probs = np.multiply.outer(probs, values).ravel()
        mult = np.multiply.outer(mult, counts.astype(np.float64)).ravel()
        probs, inverse = np.unique(probs, return_inverse=True)
        mult = np.bincount(inverse, weights=mult)
    return probs, mult


def unique_fraction_exact(attributes: Sequence[Attribute], n: int) -> float:
    """E[fraction unique] = Σ_c p_c (1 - p_c)^{n-1}, the chance a random person shares their cell with nobody."""
    p, mult = compressed_cells(attributes)
    return float(np.sum(mult * p * np.exp((n - 1) * np.log1p(-p))))


def unique_fraction_collision_bound(attributes: Sequence[Attribute], n: int) -> float:
    """Jensen lower bound (1 - Σ p²)^{n-1}; Σ p² is a product over independent attributes."""
    c = math.prod(collision_probability(a.probs) for a in attributes)
    return math.exp((n - 1) * math.log1p(-c))


def unique_fraction_entropy(attributes: Sequence[Attribute], n: int) -> float:
    """Rule of thumb: pretend the cells are uniform with the same entropy, exp(-n 2^{-H})."""
    h = sum(entropy_bits(a.probs) for a in attributes)
    return math.exp(-n * 2.0 ** (-h))


def draw_population(rng: np.random.Generator, attributes: Sequence[Attribute], n: int) -> IntArray:
    """Each person's cell, encoded as one integer in mixed radix."""
    code = np.zeros(n, dtype=np.int64)
    for a in attributes:
        values = rng.choice(len(a.probs), size=n, p=np.array(a.probs))
        code = code * len(a.probs) + values
    return code


def cell_counts(code: IntArray) -> IntArray:
    """For each person, how many people (including them) share their cell."""
    _, inverse, counts = np.unique(code, return_inverse=True, return_counts=True)
    return counts[inverse].astype(np.int64)


def simulated_unique_fraction(rng: np.random.Generator, attributes: Sequence[Attribute], n: int) -> float:
    return float(np.mean(np.equal(cell_counts(draw_population(rng, attributes, n)), 1)))


def mcdiarmid_halfwidth(n: int, beta: float) -> float:
    """t with 2 exp(-n t²/2) = beta: the fraction unique has bounded differences 2/n."""
    return math.sqrt(2 * math.log(2 / beta) / n)


# --- Sparse data: Narayanan–Shmatikov scoring -------------------------------------------------


@dataclass(frozen=True)
class Ratings:
    """A users × items rating matrix, stored items × users (0 = not rated, else 1..5)."""

    by_item: NDArray[np.int8]
    popularity: FloatArray  # the model's item popularity weights, used to draw wrong auxiliary items

    @property
    def users(self) -> int:
        return int(self.by_item.shape[1])

    def support(self) -> IntArray:
        """|supp(i)|: how many users rated item i."""
        return np.count_nonzero(self.by_item, axis=1).astype(np.int64)


def draw_ratings(
    rng: np.random.Generator, users: int, items: int, zipf_s: float = 0.8, median_count: float = 30.0
) -> Ratings:
    """Synthetic sparse ratings: Zipf item popularity, log-normal number of ratings per user."""
    popularity = np.array(zipf(items, zipf_s))
    log_pop = np.log(popularity)
    counts = np.clip(np.round(median_count * np.exp(0.8 * rng.standard_normal(users))), 5, items // 4)
    user_bias = 0.5 * rng.standard_normal(users)
    item_quality = 0.5 * rng.standard_normal(items)
    by_item = np.zeros((items, users), dtype=np.int8)
    k_max = int(counts.max())
    for start in range(0, users, 1000):  # Gumbel top-k: each user's k items, without replacement, by popularity
        rows = np.arange(start, min(start + 1000, users))
        keys = log_pop + rng.gumbel(size=(len(rows), items))
        top = np.argpartition(-keys, k_max - 1, axis=1)[:, :k_max]
        order = np.argsort(-np.take_along_axis(keys, top, axis=1), axis=1)
        top = np.take_along_axis(top, order, axis=1)  # each row's k_max best items, best first
        for j, u in enumerate(rows):
            chosen = top[j, : int(counts[u])]
            raw: FloatArray = 3.5 + user_bias[u] + item_quality[chosen] + 0.8 * rng.standard_normal(len(chosen))
            by_item[chosen, u] = np.clip(np.round(raw), 1, 5).astype(np.int8)
    return Ratings(by_item, popularity)


def user_record(r: Ratings, u: int) -> tuple[IntArray, IntArray]:
    items = np.flatnonzero(r.by_item[:, u]).astype(np.int64)
    return items, r.by_item[items, u].astype(np.int64)


def make_aux(
    rng: np.random.Generator, record: tuple[IntArray, IntArray], m: int, wrong: int, popularity: FloatArray
) -> tuple[IntArray, IntArray]:
    """m auxiliary (item, rating) pairs about the target: m - wrong from their record, with ratings
    off by one half the time, and `wrong` items that are made up (drawn by popularity)."""
    items, ratings = record
    pick = rng.choice(len(items), size=min(m - wrong, len(items)), replace=False)
    aux_items = items[pick]
    aux_ratings = np.clip(ratings[pick] + rng.choice([-1, 0, 0, 1], size=len(pick)), 1, 5)
    if wrong > 0:
        fake = rng.choice(len(popularity), size=wrong, replace=False, p=popularity)
        aux_items = np.concatenate([aux_items, fake])
        aux_ratings = np.concatenate([aux_ratings, rng.integers(1, 6, size=wrong)])
    return aux_items.astype(np.int64), aux_ratings.astype(np.int64)


def weights(support: IntArray, users: int, scheme: str) -> FloatArray:
    """Item weights: 'uniform', 'ns' = 1/ln|supp(i)| (Narayanan–Shmatikov), 'lr' = ln(n/|supp(i)|)."""
    s = np.maximum(support, 2).astype(np.float64)
    if scheme == "uniform":
        return np.ones_like(s)
    if scheme == "ns":
        return 1 / np.log(s)
    if scheme == "lr":
        return np.log(users / s)
    raise ValueError(scheme)


def scores(r: Ratings, aux: tuple[IntArray, IntArray], w: FloatArray) -> FloatArray:
    """Score(aux, record) = Σ_{i in aux} w_i · 1[record rated i within 1 of the aux rating]."""
    items, ratings = aux
    block = r.by_item[items].astype(np.int64)  # m × users
    match = (block > 0) & (np.abs(block - ratings[:, None]) <= 1)
    return w[items] @ match.astype(np.float64)


def top_match_success(s: FloatArray, target: int) -> float:
    """Probability that the best score is the target's, with ties broken uniformly at random."""
    best = float(np.max(s))
    if s[target] < best:
        return 0.0
    return 1.0 / int(np.count_nonzero(s == best))


def eccentricity(s: FloatArray) -> float:
    """(best - second best) / standard deviation of all scores."""
    top2 = np.partition(s, -2)[-2:]
    sd = float(np.std(s))
    return float(top2[1] - top2[0]) / sd if sd > 0 else 0.0


def reidentification_rate(rng: np.random.Generator, r: Ratings, m: int, wrong: int, scheme: str, targets: int) -> float:
    w = weights(r.support(), r.users, scheme)
    total = 0.0
    for t in rng.choice(r.users, size=targets, replace=False):
        aux = make_aux(rng, user_record(r, int(t)), m, wrong, r.popularity)
        total += top_match_success(scores(r, aux, w), int(t))
    return total / targets


def aux_bits(r: Ratings, aux_items: IntArray) -> float:
    """Σ log2(n/|supp(i)|): the identifying information in a set of aux items."""
    s = np.maximum(r.support()[aux_items], 1).astype(np.float64)
    return float(np.sum(np.log2(r.users / s)))


def eccentricity_trials(
    rng: np.random.Generator, released: Ratings, outsiders: Ratings, m: int, wrong: int, targets: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Eccentricity and correctness for insiders (in the release) and outsiders (not in it).

    Returns (insider eccentricities, insider top-match-is-target indicators, outsider eccentricities).
    """
    w = weights(released.support(), released.users, "lr")
    ecc_in: list[float] = []
    hit_in: list[float] = []
    ecc_out: list[float] = []
    for t in rng.choice(released.users, size=targets, replace=False):
        s = scores(released, make_aux(rng, user_record(released, int(t)), m, wrong, released.popularity), w)
        ecc_in.append(eccentricity(s))
        hit_in.append(1.0 if int(np.argmax(s)) == int(t) and top_match_success(s, int(t)) == 1.0 else 0.0)
    for t in rng.choice(outsiders.users, size=targets, replace=False):
        s = scores(released, make_aux(rng, user_record(outsiders, int(t)), m, wrong, released.popularity), w)
        ecc_out.append(eccentricity(s))
    return np.array(ecc_in), np.array(hit_in), np.array(ecc_out)


def accept_rates(ecc_in: FloatArray, hit_in: FloatArray, ecc_out: FloatArray, phi: float) -> tuple[float, float]:
    """(correct re-identifications among insiders, false accepts among outsiders) at threshold φ."""
    return float(np.mean((ecc_in >= phi) * hit_in)), float(np.mean(ecc_out >= phi))


def threshold_for_fpr(ecc_out: FloatArray, fpr: float) -> float:
    """Smallest threshold φ (on a 0.01 grid) at which at most a fraction `fpr` of outsiders is accepted."""
    phi = 0.0
    while float(np.mean(ecc_out >= phi)) > fpr:
        phi += 0.01
    return phi


def split_ratings(r: Ratings, released: int) -> tuple[Ratings, Ratings]:
    return (Ratings(r.by_item[:, :released], r.popularity), Ratings(r.by_item[:, released:], r.popularity))


# --- Sampling: is a sample-unique record population-unique? ----------------------------------


def correct_match_probability(p: float, n: int, pi: float) -> float:
    """Pr[population unique | the target's cell appears exactly once in a Bernoulli(π) sample].

    Exact for n independent people: 1 / (1 + (n-1) p (1-π) / (1 - pπ)).
    """
    return 1 / (1 + (n - 1) * p * (1 - pi) / (1 - p * pi))


def correct_match_poisson(lam: float, pi: float) -> float:
    """The Poisson form 1 / (1 + λ(1 - π)), λ the expected number of others in the target's cell."""
    return 1 / (1 + lam * (1 - pi))


def population_unique_given_sample_unique(p: float, n: int, pi: float) -> float:
    """Pr[a sampled record that is alone in its cell in the sample is alone in the population].

    Each of the other n - 1 people is outside the cell given that they are not (in the cell and
    sampled): ((1 - p)/(1 - pπ))^{n-1}.
    """
    return math.exp((n - 1) * (math.log1p(-p) - math.log1p(-p * pi)))


def population_unique_poisson(lam: float, pi: float) -> float:
    """The Poisson form e^{-λ(1 - π)}: no unsampled others in the cell."""
    return math.exp(-lam * (1 - pi))


def sample_unique_correctness_exact(attributes: Sequence[Attribute], n: int, pi: float) -> float:
    """Fraction of sample-unique records that are population-unique, in expectation:

    Σ_c n p π (1-p)^{n-1} / Σ_c n p π (1-pπ)^{n-1}.
    """
    p, mult = compressed_cells(attributes)
    num = float(np.sum(mult * p * np.exp((n - 1) * np.log1p(-p))))
    den = float(np.sum(mult * p * np.exp((n - 1) * np.log1p(-p * pi))))
    return num / den


def sample_unique_correctness_simulated(
    rng: np.random.Generator, code: IntArray, pi: float, repeats: int
) -> tuple[float, int]:
    """Empirical fraction of sample-unique records that are population-unique, and how many there were."""
    pop_count = cell_counts(code)
    hits = total = 0
    for _ in range(repeats):
        sampled = rng.random(len(code)) < pi
        sample_unique = np.equal(cell_counts(code[sampled]), 1)
        hits += int(np.count_nonzero(np.equal(pop_count[sampled][sample_unique], 1)))
        total += int(np.count_nonzero(sample_unique))
    return hits / total, total


def decode(code: IntArray, attributes: Sequence[Attribute]) -> list[IntArray]:
    """Undo the mixed-radix encoding of draw_population: one array of values per attribute."""
    values: list[IntArray] = []
    rest = code.copy()
    for a in reversed(attributes):
        values.append(rest % len(a.probs))
        rest = rest // len(a.probs)
    return values[::-1]


def estimated_correctness(
    rng: np.random.Generator, attributes: Sequence[Attribute], code: IntArray, pi: float
) -> tuple[FloatArray, FloatArray]:
    """The adversary's estimate for each sample-unique record, and whether it really is unique.

    The adversary sees only the sample. It estimates each attribute's marginal from the sample,
    multiplies them to get p̂ for the record's cell (assuming independence), and predicts
    ((1 - p̂)/(1 - p̂π))^{n-1}, the chance that the record is alone in the population. n and π are public.
    """
    n = len(code)
    pop_count = cell_counts(code)
    sampled = rng.random(n) < pi
    sample_code = code[sampled]
    unique = cell_counts(sample_code) == 1
    values = decode(sample_code, attributes)
    p_hat = np.ones(len(sample_code))
    for a, v in zip(attributes, values, strict=True):
        freq = np.bincount(v, minlength=len(a.probs)) / len(v)
        p_hat = p_hat * freq[v]
    predicted = np.exp((n - 1) * (np.log1p(-p_hat) - np.log1p(-p_hat * pi)))
    actual = (pop_count[sampled] == 1).astype(np.float64)
    return predicted[unique], actual[unique]


def calibration(predicted: FloatArray, actual: FloatArray, edges: Sequence[float]) -> list[tuple[float, float, int]]:
    """(mean predicted, observed rate, count) for each bin of predicted correctness."""
    rows: list[tuple[float, float, int]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        inside = (predicted >= lo) & (predicted < hi) if hi < 1 else (predicted >= lo)
        if np.any(inside):
            rows.append((float(np.mean(predicted[inside])), float(np.mean(actual[inside])), int(np.sum(inside))))
    return rows


# --- The chapter's numbers ------------------------------------------------------------------

# Sweeney's back-of-the-envelope, with round numbers: about 40,000 five-digit ZIP codes, 365 birthdays
# in each of about 80 birth years, two sexes, and the 1990 US population of 248.7 million.
ZIP_CODES = 40_000
BIRTH_DATES = 365 * 80
US_1990 = 248_700_000
WORLD = 8_000_000_000
NETFLIX_USERS = 480_189

POPULATION = 10_000  # the synthetic census of the uniqueness section
BIG_POPULATION = 100_000  # the synthetic census of the sampling section
SAMPLING_ATTRIBUTES = 7  # sampling section: the first seven attributes
USERS, OUTSIDERS, ITEMS = 5_000, 500, 4_000  # the synthetic ratings
AUX_SIZES = list(range(1, 11))
PHI = 1.5  # Narayanan–Shmatikov's eccentricity threshold


def print_table(rng: np.random.Generator) -> None:
    p_small, k_small = 1e-4, 9999
    print(
        f"[Fact 2] (1-p)^k={(1 - p_small) ** k_small:.5f}  lower e^(-pk/(1-p))={math.exp(-p_small * k_small / (1 - p_small)):.5f}  "
        f"upper e^(-pk)={math.exp(-p_small * k_small):.5f}"
    )
    print("[Bits]")
    print(f"  log2 of world population 8e9 = {math.log2(WORLD):.1f} bits;  US 1990 = {math.log2(US_1990):.1f} bits")
    zip_bits, dob_bits = math.log2(ZIP_CODES), math.log2(BIRTH_DATES)
    print(f"  ZIP={zip_bits:.1f}  birth date={dob_bits:.1f}  sex=1.0  total={zip_bits + dob_bits + 1:.1f} bits")
    cells = ZIP_CODES * BIRTH_DATES * 2
    print(
        f"  cells={cells:.3g}  n/M={US_1990 / cells:.3f}  uniform unique fraction={uniform_unique_fraction(US_1990, cells):.3f}"
    )
    print(f"  world, 33 bits: n/M={WORLD / 2**33:.3f}  unique fraction={poisson_unique_fraction(WORLD, 2**33):.3f}")
    for f in (0.5, 0.9, 0.99):
        print(
            f"  bits for {f:g} unique: world={bits_for_unique_fraction(WORLD, f):.1f}  "
            f"above log2 n by {bits_for_unique_fraction(WORLD, f) - math.log2(WORLD):.2f}"
        )

    print("[Linkage success on the toy release]")
    sizes = [1, 1, 1, 2, 3]
    print(f"  class sizes {sizes}: success={linkage_success(sizes):.3f} (= {len(sizes)}/{sum(sizes)})")

    print(f"[Synthetic census, n = {POPULATION}]")
    cum_bits = 0.0
    for k in range(1, len(ATTRIBUTES) + 1):
        a = ATTRIBUTES[k - 1]
        cum_bits += entropy_bits(a.probs)
        attrs = ATTRIBUTES[:k]
        cells_k = math.prod(len(x.probs) for x in attrs)
        sims = [simulated_unique_fraction(rng, attrs, POPULATION) for _ in range(20)]
        print(
            f"  k={k} +{a.name}: |values|={len(a.probs)} H={entropy_bits(a.probs):.2f} cum bits={cum_bits:.2f} "
            f"cells={cells_k}  exact={unique_fraction_exact(attrs, POPULATION):.4f}  "
            f"simulated={np.mean(sims):.4f} (sd {np.std(sims):.4f})  "
            f"collision bound={unique_fraction_collision_bound(attrs, POPULATION):.4f}  "
            f"entropy rule={unique_fraction_entropy(attrs, POPULATION):.4f}  "
            f"uniform cells={uniform_unique_fraction(POPULATION, cells_k):.4f}"
        )
    print(f"  log2 n = {math.log2(POPULATION):.2f}")
    print(f"  McDiarmid 95% half-width for the fraction unique: {mcdiarmid_halfwidth(POPULATION, 0.05):.4f}")

    print(f"[Sparse ratings: {USERS} released users, {OUTSIDERS} outsiders, {ITEMS} items]")
    allr = draw_ratings(rng, USERS + OUTSIDERS, ITEMS)
    released, outsiders = split_ratings(allr, USERS)
    sup = released.support()
    counts: IntArray = np.count_nonzero(released.by_item, axis=0).astype(np.int64)
    print(
        f"  ratings={int(np.sum(counts))}  per user: median={np.median(counts):.0f} mean={np.mean(counts):.1f}  "
        f"density={np.sum(counts) / (USERS * ITEMS):.4f}"
    )
    print(
        f"  log2 n = {math.log2(USERS):.2f}  most popular item rated by {int(sup.max())}; "
        f"items rated by <= 10 users: {int(np.sum(sup <= 10))}"
    )
    # bits carried by one random item from a random user's record
    picks: list[float] = []
    for u in rng.choice(USERS, size=2000, replace=False):
        items, _ = user_record(released, int(u))
        i = int(rng.choice(items))
        picks.append(math.log2(USERS / max(int(sup[i]), 1)))
    print(f"  bits in one aux item: median={np.median(picks):.2f} mean={np.mean(picks):.2f}")
    print(
        f"  Netflix: log2 480189 = {math.log2(NETFLIX_USERS):.1f}; a movie rated by 1% carries {math.log2(100):.2f} bits"
    )
    for m, wrong in ((2, 0), (4, 0), (8, 0), (8, 2)):
        rates = "  ".join(
            f"{s}={reidentification_rate(rng, released, m, wrong, s, 400):.3f}" for s in ("uniform", "ns", "lr")
        )
        print(f"  m={m} wrong={wrong}: {rates}")
    ecc_in, hit_in, ecc_out = eccentricity_trials(rng, released, outsiders, 8, 2, 400)
    tp, fp = accept_rates(ecc_in, hit_in, ecc_out, PHI)
    print(
        f"  eccentricity, m=8 wrong=2, phi={PHI:g}: correct accepts={tp:.3f}  outsider accepts={fp:.3f}  "
        f"top match correct={np.mean(hit_in):.3f}"
    )
    print(f"  median eccentricity: insiders={np.median(ecc_in):.2f}  outsiders={np.median(ecc_out):.2f}")
    phi_1 = threshold_for_fpr(ecc_out, 0.01)
    tp1, fp1 = accept_rates(ecc_in, hit_in, ecc_out, phi_1)
    print(
        f"  threshold for <= 1% outsider accepts: phi={phi_1:.2f}  correct accepts={tp1:.3f}  outsider accepts={fp1:.3f}"
    )

    print(f"[Sampling, n = {BIG_POPULATION}, first {SAMPLING_ATTRIBUTES} attributes]")
    attrs = ATTRIBUTES[:SAMPLING_ATTRIBUTES]
    code = draw_population(rng, attrs, BIG_POPULATION)
    print(
        f"  population unique fraction: exact={unique_fraction_exact(attrs, BIG_POPULATION):.4f}  "
        f"this population={np.mean(cell_counts(code) == 1):.4f}"
    )
    for pi in (0.01, 0.1, 0.5):
        sim, count = sample_unique_correctness_simulated(rng, code, pi, 20)
        print(
            f"  pi={pi:g}: P(pop unique | sample unique) exact={sample_unique_correctness_exact(attrs, BIG_POPULATION, pi):.4f}"
            f"  simulated={sim:.4f} ({count} sample uniques)"
        )
    for lam in (0.1, 0.5, 2.0):
        print(
            f"  lambda={lam:g}: kappa at pi=0.01: {correct_match_poisson(lam, 0.01):.3f}  "
            f"pi=0.5: {correct_match_poisson(lam, 0.5):.3f}   "
            f"xi at pi=0.01: {population_unique_poisson(lam, 0.01):.3f}  pi=0.5: {population_unique_poisson(lam, 0.5):.3f}"
        )
    predicted, actual = estimated_correctness(rng, attrs, code, 0.01)
    print(
        f"  adversary's estimate at pi=0.01: {len(predicted)} sample uniques, mean predicted={np.mean(predicted):.3f}  "
        f"actual={np.mean(actual):.3f}"
    )
    for mean_pred, rate, count in calibration(predicted, actual, [0.0, 0.5, 0.8, 0.95, 1.0]):
        print(f"    bin: predicted={mean_pred:.3f}  actual={rate:.3f}  count={count}")


# --- Figures --------------------------------------------------------------------------------


def plot_uniqueness(rng: np.random.Generator) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    ks = list(range(1, len(ATTRIBUTES) + 1))
    exact = [unique_fraction_exact(ATTRIBUTES[:k], POPULATION) for k in ks]
    bound = [unique_fraction_collision_bound(ATTRIBUTES[:k], POPULATION) for k in ks]
    rule = [unique_fraction_entropy(ATTRIBUTES[:k], POPULATION) for k in ks]
    sims = [simulated_unique_fraction(rng, ATTRIBUTES[:k], POPULATION) for k in ks]
    left.plot(ks, exact, "k-", linewidth=2, label=r"exact: $\sum_c p_c(1-p_c)^{n-1}$")
    left.plot(ks, sims, "o", color="C0", label="one simulated population")
    left.plot(ks, bound, "--", color="C2", label=r"lower bound $(1-\sum_c p_c^2)^{n-1}$")
    left.plot(ks, rule, ":", color="C1", label=r"entropy rule $e^{-n2^{-H}}$")
    left.set_xticks(
        ks, [f"+{a.name}" if i else a.name for i, a in enumerate(ATTRIBUTES)], rotation=40, ha="right", fontsize=8
    )
    left.set(
        ylabel="fraction of people who are unique", ylim=(-0.02, 1.02), title=f"Synthetic census, n = {POPULATION:,}"
    )
    left.legend(fontsize=8, loc="upper left")

    excess = np.linspace(-4, 8, 241)
    for n, style in ((1e4, "C0-"), (1e6, "C1--"), (8e9, "k:")):
        ys = [uniform_unique_fraction(n, n * 2.0**x) for x in excess]
        right.plot(excess, ys, style, label=f"n = {n:.0e}".replace("e+0", "e"))
    for f in (0.5, 0.99):
        x = math.log2(1 / math.log(1 / f))
        right.plot([x], [f], "ko", markersize=4)
        right.annotate(f"{f:.0%} unique at {x:+.1f} bits", (x, f), xytext=(x + 0.6, f - 0.12), fontsize=8)
    right.set(
        xlabel=r"bits of quasi-identifier minus $\log_2 n$",
        ylabel="fraction unique",
        title="Uniform cells: one curve for every n",
    )
    right.legend(fontsize=8, loc="lower right")
    return fig


def plot_bits() -> Figure:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    parts = [
        ("5-digit ZIP code", math.log2(ZIP_CODES), "C0"),
        ("date of birth", math.log2(BIRTH_DATES), "C1"),
        ("sex", 1.0, "C2"),
    ]
    start = 0.0
    for name, bits, color in parts:
        ax.barh([0], [bits], left=start, color=color, edgecolor="white")
        if bits > 3:
            ax.text(
                start + bits / 2, 0, f"{name}\n{bits:.1f} bits", ha="center", va="center", fontsize=9, color="white"
            )
        else:
            ax.annotate(
                f"{name}: {bits:.1f} bit",
                (start + bits / 2, -0.4),
                xytext=(start + bits / 2, -0.75),
                ha="center",
                fontsize=8,
                arrowprops={"arrowstyle": "-"},
            )
        start += bits
    for value, label, offset, align in (
        (math.log2(US_1990), "US 1990, 248.7 million", -0.3, "right"),
        (math.log2(WORLD), "world, 8 billion", 0.3, "left"),
    ):
        ax.axvline(value, color="k", linestyle="--")
        ax.text(value + offset, 0.7, f"{label}:\n{value:.1f} bits", fontsize=8, va="center", ha=align)
    ax.set(
        xlim=(0, 40),
        ylim=(-0.9, 0.95),
        xlabel="bits",
        yticks=[],
        title="Sweeney's three attributes carry more bits than it takes to single out one American",
    )
    return fig


def plot_netflix(rng: np.random.Generator) -> Figure:
    allr = draw_ratings(rng, USERS + OUTSIDERS, ITEMS)
    released, outsiders = split_ratings(allr, USERS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    w = weights(released.support(), released.users, "lr")
    insider = int(rng.integers(USERS))
    outsider = int(rng.integers(OUTSIDERS))
    for ax, ratings, who, title in (
        (axes[0], released, insider, "target is in the release"),
        (axes[1], outsiders, outsider, "target is not in the release"),
    ):
        s = scores(released, make_aux(rng, user_record(ratings, who), 8, 2, released.popularity), w)
        ax.hist(s, bins=40, color="C7")
        best = float(np.max(s))
        ax.axvline(best, color="C3", linestyle="--", label=f"best score, eccentricity {eccentricity(s):.1f}")
        if ratings is released:
            ax.axvline(float(s[who]), color="C0", linestyle=":", linewidth=2, label="the target's record")
        ax.set(yscale="log", xlabel="score", ylabel="records", title=title)
        ax.legend(fontsize=8)

    for scheme, style, label in (
        ("uniform", "C7-o", "equal weights"),
        ("ns", "C1-s", r"$1/\ln|\mathrm{supp}(i)|$"),
        ("lr", "C0-^", r"$\ln(n/|\mathrm{supp}(i)|)$"),
    ):
        rates = [reidentification_rate(rng, released, m, 0, scheme, 200) for m in AUX_SIZES]
        axes[2].plot(AUX_SIZES, rates, style, label=label, markersize=4)
    noisy = [reidentification_rate(rng, released, m, m // 4, "lr", 200) for m in AUX_SIZES]
    axes[2].plot(AUX_SIZES, noisy, "C0--", label=r"$\ln(n/|\mathrm{supp}(i)|)$, a quarter of aux wrong")
    axes[2].set(
        xlabel="auxiliary ratings m",
        ylabel="top match is the target",
        ylim=(-0.02, 1.02),
        title="Re-identification rate",
    )
    axes[2].legend(fontsize=8, loc="lower right")
    return fig


def plot_eccentricity_roc(rng: np.random.Generator) -> Figure:
    allr = draw_ratings(rng, USERS + OUTSIDERS, ITEMS)
    released, outsiders = split_ratings(allr, USERS)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    phis = np.linspace(0, 12, 481)
    for m, style in ((4, "C1"), (6, "C2"), (8, "C0")):
        ecc_in, hit_in, ecc_out = eccentricity_trials(rng, released, outsiders, m, m // 4, 300)
        pts = [accept_rates(ecc_in, hit_in, ecc_out, float(phi)) for phi in phis]
        ax.plot([max(fp, 1e-3) for _, fp in pts], [tp for tp, _ in pts], style, label=f"m = {m} ({m // 4} wrong)")
        tp, fp = accept_rates(ecc_in, hit_in, ecc_out, PHI)
        ax.plot([max(fp, 1e-3)], [tp], "o", color=style)
    ax.set(
        xscale="log",
        xlim=(1e-3, 1.05),
        ylim=(-0.02, 1.02),
        xlabel="outsiders wrongly matched (false positive rate)",
        ylabel="insiders correctly re-identified",
        title="Eccentricity threshold φ swept; dots at φ = 1.5",
    )
    ax.legend(fontsize=9, loc="lower right")
    return fig


def plot_sampling(rng: np.random.Generator) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    pis = np.logspace(-3, 0, 31)
    for k, color in ((5, "C1"), (6, "C2"), (7, "C0"), (8, "C3")):
        attrs = ATTRIBUTES[:k]
        left.plot(
            pis,
            [sample_unique_correctness_exact(attrs, BIG_POPULATION, float(p)) for p in pis],
            color=color,
            label=f"{k} attributes",
        )
        code = draw_population(rng, attrs, BIG_POPULATION)
        sim_pis = [0.003, 0.01, 0.03, 0.1, 0.3]
        left.plot(
            sim_pis,
            [sample_unique_correctness_simulated(rng, code, p, 10)[0] for p in sim_pis],
            "o",
            color=color,
            markersize=4,
        )
    left.set(
        xscale="log",
        xlabel="sampling fraction π",
        ylabel="P(population unique | sample unique)",
        ylim=(-0.02, 1.02),
        title=f"Sample uniques that are real uniques, n = {BIG_POPULATION:,}",
    )
    left.legend(fontsize=8, loc="lower right")

    attrs = ATTRIBUTES[:SAMPLING_ATTRIBUTES]
    code = draw_population(rng, attrs, BIG_POPULATION)
    predicted, actual = estimated_correctness(rng, attrs, code, 0.01)
    edges = list(np.linspace(0, 1, 11))
    rows = calibration(predicted, actual, edges)
    right.plot([0, 1], [0, 1], "k:", label="perfect calibration")
    right.plot([r[0] for r in rows], [r[1] for r in rows], "C0o-", label="adversary's estimate, 1% sample")
    right.set(
        xlabel="predicted probability of being unique in the population",
        ylabel="observed fraction unique in the population",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
        title="Which sample uniques are safe? The sample tells",
    )
    right.legend(fontsize=8, loc="upper left")
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
        "link_bits.svg": plot_bits(),
        "link_uniqueness.svg": plot_uniqueness(rng),
        "link_netflix.svg": plot_netflix(rng),
        "link_eccentricity_roc.svg": plot_eccentricity_roc(rng),
        "link_sampling.svg": plot_sampling(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
