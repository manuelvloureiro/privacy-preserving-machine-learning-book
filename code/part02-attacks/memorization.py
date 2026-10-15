"""Check the numbers in the memorization and training-data extraction chapter and draw its figures.

    uv run python code/part02-attacks/memorization.py
    uv run python code/part02-attacks/memorization.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 05-memorization-extraction.md (they match exactly with the default
seed), then saves the chapter's figures.

The model is a character n-gram language model with add-α smoothing, trained on a synthetic
contact list: each line is a user name built from random syllables, a space, and a five-digit pin,
such as "kafimo 48213". The n-gram order n (the model sees the last n - 1 characters) plays the
role of model size. Secret lines are inserted k times each (duplication). A canary
"my pin is ddddd" is inserted k times to measure exposure (Carlini et al. 2019).
"""

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "memorization"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

# --- The corpus -----------------------------------------------------------------------------

ALPHABET = "^\n abcdefghijklmnopqrstuvwxyz0123456789"  # "^" pads the start of a line; "\n" ends it
V = len(ALPHABET)
BOS, EOL = 0, 1
INDEX = {ch: i for i, ch in enumerate(ALPHABET)}
DIGIT_CODES = [INDEX[str(d)] for d in range(10)]
CONSONANTS = "bdfgklmnprstvz"
VOWELS = "aeiou"
PIN_DIGITS = 5
PIN_SPACE = 10**PIN_DIGITS  # |R|, the number of possible pins
CANARY_PREFIX = "my pin is "
ALPHA = 0.1  # add-α smoothing
BACKGROUND_LINES = 5000
COPIES = (1, 2, 4, 8, 16, 32)  # duplication groups for the secret lines
PER_GROUP = 100
ORDERS = (3, 4, 5, 6, 7, 8)
EXPOSURE_ORDERS = (2, 3, 4)
CANARY_COPIES = (0, 1, 2, 4, 8, 16, 32, 64)


def encode(text: str) -> IntArray:
    return np.array([INDEX[ch] for ch in text], dtype=np.int64)


def decode(codes: Sequence[int]) -> str:
    return "".join(ALPHABET[c] for c in codes)


def random_name(rng: np.random.Generator, syllables: int = 3) -> str:
    """A user name of consonant-vowel syllables: 70 syllables, so 343,000 names of three."""
    return "".join(
        CONSONANTS[int(rng.integers(len(CONSONANTS)))] + VOWELS[int(rng.integers(len(VOWELS)))]
        for _ in range(syllables)
    )


def random_pin(rng: np.random.Generator) -> str:
    return "".join(str(int(d)) for d in rng.integers(0, 10, PIN_DIGITS))


def contact_line(name: str, pin: str) -> str:
    return f"{name} {pin}"


@dataclass(frozen=True)
class Secret:
    name: str
    pin: str
    copies: int

    @property
    def prompt(self) -> str:
        return f"{self.name} "

    @property
    def line(self) -> str:
        return contact_line(self.name, self.pin)


def make_corpus(
    rng: np.random.Generator,
    background: int = BACKGROUND_LINES,
    copies: Sequence[int] = COPIES,
    per_group: int = PER_GROUP,
) -> tuple[list[str], list[Secret]]:
    """Background lines (each once) and secret lines (each `copies` times), all with distinct names."""
    names: list[str] = []
    seen: set[str] = set()
    while len(names) < background + per_group * len(copies):
        name = random_name(rng)
        if name not in seen:
            seen.add(name)
            names.append(name)
    lines = [contact_line(name, random_pin(rng)) for name in names[:background]]
    secrets = [
        Secret(name, random_pin(rng), k)
        for j, k in enumerate(copies)
        for name in names[background + j * per_group : background + (j + 1) * per_group]
    ]
    return lines, secrets


def training_set(
    background: Sequence[str], secrets: Sequence[Secret], extra: Sequence[tuple[str, int]] = ()
) -> list[tuple[str, int]]:
    """(line, multiplicity) pairs: background once, secrets `copies` times, extra lines as given."""
    return [(line, 1) for line in background] + [(s.line, s.copies) for s in secrets] + list(extra)


# --- The n-gram model -----------------------------------------------------------------------


@dataclass(frozen=True)
class NgramModel:
    """Counts N(c, w) of symbol w after context c (the last order - 1 symbols), with add-α smoothing:

        p(w | c) = (N(c, w) + α) / (N(c) + α V).

    Pairs are stored as sorted integer keys c·V + w; contexts as sorted codes c.
    """

    order: int
    alpha: float
    keys: IntArray
    counts: FloatArray
    ctx_keys: IntArray
    ctx_counts: FloatArray

    @property
    def stored_ngrams(self) -> int:
        """The number of distinct n-grams with a non-zero count: the model's size."""
        return len(self.keys)


def context_codes(padded: IntArray, order: int) -> IntArray:
    """For rows of symbols padded with order - 1 BOS on the left, the context code of every later position.

    Column j of the result is the code of the order - 1 symbols before position j + order - 1 of `padded`.
    """
    width = padded.shape[1] - (order - 1)
    codes = np.zeros((padded.shape[0], width), dtype=np.int64)
    for i in range(order - 1):
        codes = codes * V + padded[:, i : i + width]
    return codes


def pad(codes: IntArray, order: int) -> IntArray:
    """Prepend order - 1 BOS symbols to a 1-d or 2-d array of symbol codes."""
    rows = np.atleast_2d(codes)
    return np.concatenate([np.full((rows.shape[0], order - 1), BOS, dtype=np.int64), rows], axis=1)


def line_keys(line: str, order: int) -> IntArray:
    """The keys c·V + w of every (context, next symbol) pair in a line, including the final EOL."""
    codes = np.concatenate([encode(line), np.array([EOL], dtype=np.int64)])
    padded = pad(codes, order)
    return (context_codes(padded, order)[0] * V + codes).astype(np.int64)


def aggregate(keys: IntArray, weights: FloatArray) -> tuple[IntArray, FloatArray]:
    unique, inverse = np.unique(keys, return_inverse=True)
    totals = np.zeros(len(unique), dtype=np.float64)
    np.add.at(totals, inverse, weights)
    return unique.astype(np.int64), totals


def train(lines: Sequence[tuple[str, int]], order: int, alpha: float = ALPHA) -> NgramModel:
    """Maximum-likelihood counts plus add-α smoothing; training is counting."""
    if V**order >= 2**62:
        raise ValueError("order too large for 64-bit keys")
    by_length: dict[int, list[tuple[str, int]]] = {}
    for line, m in lines:
        by_length.setdefault(len(line), []).append((line, m))
    key_parts: list[IntArray] = []
    weight_parts: list[FloatArray] = []
    for length, group in by_length.items():  # lines of one length form a matrix: one vectorized pass
        codes = np.full((len(group), length + 1), EOL, dtype=np.int64)
        for i, (line, _) in enumerate(group):
            codes[i, :length] = encode(line)
        keys_matrix = context_codes(pad(codes, order), order) * V + codes
        key_parts.append(keys_matrix.ravel())
        weight_parts.append(np.repeat(np.array([float(m) for _, m in group]), length + 1))
    keys, counts = aggregate(np.concatenate(key_parts), np.concatenate(weight_parts))
    ctx_keys, ctx_counts = aggregate(keys // V, counts)
    return NgramModel(order, alpha, keys, counts, ctx_keys, ctx_counts)


def lookup(sorted_keys: IntArray, values: FloatArray, query: IntArray) -> FloatArray:
    """values[i] where sorted_keys[i] == query, and 0 where the query is absent."""
    idx = np.searchsorted(sorted_keys, query)
    idx = np.minimum(idx, len(sorted_keys) - 1)
    found = sorted_keys[idx] == query
    return np.where(found, values[idx], 0.0)


def next_counts(model: NgramModel, ctx: IntArray) -> FloatArray:
    """N(c, w) for every context in `ctx` (any shape) and every symbol w: shape ctx.shape + (V,)."""
    query = ctx[..., None] * V + np.arange(V, dtype=np.int64)
    return lookup(model.keys, model.counts, query)


def log_probs_of(model: NgramModel, ctx: IntArray, nxt: IntArray) -> FloatArray:
    """ln p(next | ctx), elementwise."""
    pair = lookup(model.keys, model.counts, ctx * V + nxt)
    total = lookup(model.ctx_keys, model.ctx_counts, ctx)
    return np.log(pair + model.alpha) - np.log(total + model.alpha * V)


def continuation_log_probs(model: NgramModel, prompt: str, continuations: IntArray) -> FloatArray:
    """ln p(continuation | prompt) for each row of symbol codes, by the chain rule (natural log)."""
    rows = np.atleast_2d(continuations)
    head = np.tile(encode(prompt), (rows.shape[0], 1))
    full = pad(np.concatenate([head, rows], axis=1), model.order)
    ctx = context_codes(full, model.order)[:, -rows.shape[1] :]
    terms = log_probs_of(model, ctx, rows)
    total = np.zeros(rows.shape[0])
    for j in range(rows.shape[1]):  # fixed summation order: equal counts give bit-identical sums
        total = total + terms[:, j]
    return total


def line_log_prob(model: NgramModel, line: str) -> float:
    """ln p(line + EOL) from the start of a line."""
    codes = np.concatenate([encode(line), np.array([EOL], dtype=np.int64)])
    return float(continuation_log_probs(model, "", codes[None, :])[0])


def lines_log_probs(model: NgramModel, lines: Sequence[str]) -> FloatArray:
    """ln p(line + EOL) for many lines at once: pad to a common length and mask the padding."""
    width = max(len(line) for line in lines) + 1
    codes = np.full((len(lines), width), EOL, dtype=np.int64)
    mask = np.zeros((len(lines), width), dtype=bool)
    for i, line in enumerate(lines):
        codes[i, : len(line)] = encode(line)
        mask[i, : len(line) + 1] = True
    ctx = context_codes(pad(codes, model.order), model.order)
    terms = np.where(mask, log_probs_of(model, ctx, codes), 0.0)
    return terms.sum(axis=1)


def greedy(model: NgramModel, prompts: Sequence[str], steps: int) -> list[str]:
    """Greedy decoding: append the most frequent next symbol (lowest code on ties) `steps` times, or until EOL.

    Smoothing adds the same α to every count, so the argmax of p(w | c) is the argmax of N(c, w).
    """
    out: list[str] = []
    for prompt in prompts:
        codes = list(encode(prompt))
        for _ in range(steps):
            window = ([BOS] * (model.order - 1) + codes)[-(model.order - 1) :]
            ctx = 0
            for c in window:
                ctx = ctx * V + c
            counts = next_counts(model, np.array([ctx], dtype=np.int64))[0]
            codes.append(int(np.argmax(counts)))
            if codes[-1] == EOL:
                break  # the line is over
        out.append(decode(codes[len(prompt) :]))
    return out


def sample_lines(model: NgramModel, rng: np.random.Generator, count: int, max_len: int = 20) -> list[str]:
    """Temperature-1 samples from the start of a line, stopped at EOL (or cut at max_len symbols)."""
    codes = np.full((count, model.order - 1 + max_len), BOS, dtype=np.int64)
    done = np.zeros(count, dtype=bool)
    lengths = np.full(count, max_len)
    for t in range(max_len):
        ctx = context_codes(codes[:, t : t + model.order], model.order)[:, 0]
        weights = next_counts(model, ctx) + model.alpha
        weights[:, BOS] = 0.0  # BOS only pads; it is never generated
        cdf = np.cumsum(weights, axis=1)
        u = rng.random(count) * cdf[:, -1]
        nxt = (cdf < u[:, None]).sum(axis=1).astype(np.int64)
        codes[:, model.order - 1 + t] = nxt
        newly = (~done) & (nxt == EOL)
        lengths[newly] = t
        done |= nxt == EOL
    body = codes[:, model.order - 1 :]
    return [decode([int(c) for c in body[i, : lengths[i]]]) for i in range(count)]


def unseen_context_probs(copies: int, alpha: float = ALPHA) -> tuple[float, float]:
    """At a context that occurs only in the k copies of one line: p(the line's next symbol), p(any other)."""
    return (copies + alpha) / (copies + alpha * V), alpha / (copies + alpha * V)


# --- Discoverable memorization and the copies needed ----------------------------------------


NEVER = 10**9  # k* when no number of copies makes greedy decoding emit the pin


def pin_codes(pin: str) -> IntArray:
    return encode(pin)


def copies_needed(model: NgramModel, secret: Secret) -> int:
    """k*: the fewest copies of the secret line that make greedy decoding from its prompt emit its pin.

    At step t the context c_t is fixed by the prompt and the earlier pin digits. Without the secret,
    the counts are N₋(c_t, w) = N(c_t, w) - k·(times (c_t, w) occurs in the line). With k copies, the
    true digit wins iff N₋(c_t, s_t) + k·m_t > N₋(c_t, w) + k·m_t(w) for every rival w (≥ for a rival
    with a higher code, since greedy breaks ties toward the lowest code). In the usual case m_t = 1
    and no rival from the line itself, k*_t = max_{w≠s_t} N₋(c_t, w) - N₋(c_t, s_t) + 1, minus one
    when ties go the true digit's way. Returns the smallest k ≥ 1 that wins every step, or NEVER.
    """
    own_keys = line_keys(secret.line, model.order)
    own_unique, own_mult = np.unique(own_keys, return_counts=True)
    prompt = encode(secret.prompt)
    pin = pin_codes(secret.pin)
    full = pad(np.concatenate([prompt, pin]), model.order)
    ctx = context_codes(full, model.order)[0, -len(pin) :]
    counts = next_counts(model, ctx)  # (steps, V), with the secret's k copies included
    query = ctx[:, None] * V + np.arange(V, dtype=np.int64)
    own = lookup(own_unique.astype(np.int64), own_mult.astype(np.float64), query)
    without = counts - secret.copies * own  # N₋
    steps = np.arange(len(pin))
    gain = own[steps, pin][:, None] - own  # how much faster the true digit grows than each rival, per copy
    lead = without[steps, pin][:, None] - without  # the true digit's lead without the secret
    need = 1
    for t in range(len(pin)):
        for w in range(V):
            if w == pin[t]:
                continue
            # greedy takes the lowest code on ties: a rival with a higher code only has to be matched
            ties_win = w > pin[t]
            lead_tw, gain_tw = float(lead[t, w]), float(gain[t, w])
            if (lead_tw > 0 or ties_win and lead_tw == 0) and gain_tw >= 0:
                continue  # already ahead, and the copies do not help the rival
            if gain_tw <= 0:
                return NEVER  # behind, and the copies cannot close the gap
            # need lead + k·gain > 0 (or ≥ 0 when ties win)
            bound = -lead_tw / gain_tw
            need = max(need, math.ceil(bound) if ties_win else math.floor(bound) + 1)
    return need


@dataclass(frozen=True)
class Discoverable:
    copies: int
    exact: float  # greedy output equals the pin
    approximate: float  # at most one digit wrong
    predicted: float  # fraction with k* ≤ k


def discoverable_rates(model: NgramModel, secrets: Sequence[Secret]) -> list[Discoverable]:
    outputs = greedy(model, [s.prompt for s in secrets], PIN_DIGITS)
    groups = sorted({s.copies for s in secrets})
    result: list[Discoverable] = []
    for k in groups:
        idx = [i for i, s in enumerate(secrets) if s.copies == k]
        exact = [outputs[i] == secrets[i].pin for i in idx]
        wrong = [sum(a != b for a, b in zip(outputs[i], secrets[i].pin)) for i in idx]
        predicted = [copies_needed(model, secrets[i]) <= k for i in idx]
        result.append(
            Discoverable(k, float(np.mean(exact)), float(np.mean([w <= 1 for w in wrong])), float(np.mean(predicted)))
        )
    return result


def mean_sample_probability(model: NgramModel, secrets: Sequence[Secret], copies: int) -> float:
    """Average over secrets with `copies` copies of p(pin | prompt): the chance one sample reproduces it."""
    chosen = [s for s in secrets if s.copies == copies]
    probs = [math.exp(float(continuation_log_probs(model, s.prompt, pin_codes(s.pin)[None, :])[0])) for s in chosen]
    return float(np.mean(probs))


def samples_needed(prob: float, failure: float) -> int:
    """Fewest independent samples m with (1 - prob)^m ≤ failure; e^{-m·prob} ≤ failure gives m ≈ ln(1/failure)/prob."""
    return math.ceil(math.log(failure) / math.log1p(-prob))


# --- Canaries and exposure ------------------------------------------------------------------


def all_pins() -> IntArray:
    """Every pin in R as rows of digit codes followed by EOL: shape (|R|, PIN_DIGITS + 1)."""
    values = np.arange(PIN_SPACE)
    digits = np.stack([(values // 10 ** (PIN_DIGITS - 1 - j)) % 10 for j in range(PIN_DIGITS)], axis=1)
    codes = np.array(DIGIT_CODES, dtype=np.int64)[digits]
    return np.concatenate([codes, np.full((PIN_SPACE, 1), EOL, dtype=np.int64)], axis=1)


def pin_index(pin: str) -> int:
    return int(pin)


def canary_log_probs(model: NgramModel) -> FloatArray:
    """ln p(r + EOL | "my pin is ") for every pin r in R."""
    return continuation_log_probs(model, CANARY_PREFIX, all_pins())


def rank_of(log_probs: FloatArray, index: int) -> int:
    """Rank of candidate `index`: how many candidates are at least as likely (itself included)."""
    threshold = float(log_probs[index])
    return int(np.sum(log_probs >= threshold))


def exposure(log_probs: FloatArray, index: int) -> float:
    """exposure = log2 |R| - log2 rank."""
    return math.log2(len(log_probs)) - math.log2(rank_of(log_probs, index))


def all_exposures(log_probs: FloatArray) -> FloatArray:
    """The exposure every candidate would have as the canary, from one sort."""
    order = np.sort(log_probs)
    at_least = len(log_probs) - np.searchsorted(order, log_probs, side="left")
    return math.log2(len(log_probs)) - np.log2(at_least.astype(np.float64))


def expected_null_exposure(size: int) -> float:
    """E[log2 |R| - log2 rank] for a rank uniform on 1..|R|: log2 |R| - log2(|R|!)/|R| → log2 e."""
    return math.log2(size) - math.lgamma(size + 1) / math.log(2) / size


def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def exposure_normal_estimate(rng: np.random.Generator, log_probs: FloatArray, index: int, samples: int) -> float:
    """Secret Sharer's extrapolation, with a normal in place of a skew-normal: fit the scores of
    `samples` random candidates, then exposure ≈ -log2 Pr[score ≥ canary score]."""
    pick = rng.integers(0, len(log_probs), samples)
    mu, sd = float(np.mean(log_probs[pick])), float(np.std(log_probs[pick]))
    tail = normal_sf((float(log_probs[index]) - mu) / sd)
    return min(-math.log2(tail), math.log2(len(log_probs))) if tail > 0 else math.log2(len(log_probs))


def exposure_sampling_estimate(rng: np.random.Generator, log_probs: FloatArray, index: int, samples: int) -> float:
    """-log2 of the fraction of `samples` random candidates at least as likely as the canary (capped at log2 samples)."""
    pick = rng.integers(0, len(log_probs), samples)
    threshold = float(log_probs[index])
    frac = max(float(np.mean(log_probs[pick] >= threshold)), 1 / samples)
    return -math.log2(frac)


def canary_exposure(order: int, background: Sequence[tuple[str, int]], canary_pin: str, copies: int) -> float:
    model = train(list(background) + [(CANARY_PREFIX + canary_pin, copies)], order)
    return exposure(canary_log_probs(model), pin_index(canary_pin))


# --- Extraction without the prefix: generate, rank, verify ----------------------------------


@dataclass(frozen=True)
class Extraction:
    samples: int
    distinct: int
    in_training: int  # distinct generated lines that are private training lines
    public_generated: int  # distinct generated lines that are public lines
    by_copies: dict[int, float]  # fraction of secrets of each duplication group that were generated
    precision_likelihood: float  # among the top `top` distinct lines by target likelihood
    precision_ratio: float  # among the top `top` by likelihood ratio target / reference
    top: int


PUBLIC_LINES, PUBLIC_COPIES = 50, 100


def public_lines(rng: np.random.Generator, count: int = PUBLIC_LINES) -> list[str]:
    """Lines that are everywhere: in the target's training set and in the attacker's own data, many times."""
    return [contact_line(random_name(rng), random_pin(rng)) for _ in range(count)]


def with_public(lines: Sequence[tuple[str, int]], public: Sequence[str]) -> list[tuple[str, int]]:
    return list(lines) + [(line, PUBLIC_COPIES) for line in public]


def extraction_attack(
    rng: np.random.Generator,
    target: NgramModel,
    reference: NgramModel,
    training: Sequence[tuple[str, int]],
    public: Sequence[str],
    secrets: Sequence[Secret],
    samples: int,
    top: int,
) -> Extraction:
    """Sample from the target, keep distinct lines, rank them, and check the top against the private training lines.

    A hit is a generated line that is in the target's training set but not public: private data.
    """
    lines = sample_lines(target, rng, samples)
    distinct = sorted(set(lines))
    members = {line for line, _ in training} - set(public)
    hits = [line in members for line in distinct]
    generated = set(distinct)
    by_copies = {
        k: float(np.mean([s.line in generated for s in secrets if s.copies == k]))
        for k in sorted({s.copies for s in secrets})
    }
    lp_target = lines_log_probs(target, distinct)
    lp_ref = lines_log_probs(reference, distinct)

    hit_array = np.array(hits, dtype=np.float64)

    def precision(scores: FloatArray) -> float:
        best = np.argsort(-scores, kind="stable")[:top]
        return float(np.mean(hit_array[best]))

    public_generated = len(set(public) & generated)
    return Extraction(
        samples,
        len(distinct),
        int(sum(hits)),
        public_generated,
        by_copies,
        precision(lp_target),
        precision(lp_target - lp_ref),
        top,
    )


# --- The chapter's numbers ------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    background, secrets = make_corpus(rng)
    training = training_set(background, secrets)
    positions = sum(m * (len(line) + 1) for line, m in training)
    print(
        f"[Corpus] background lines={len(background)}  secrets={len(secrets)} "
        f"({PER_GROUP} per copies group {COPIES})  training lines={sum(m for _, m in training)}  "
        f"symbols={positions}  example line '{background[0]}'"
    )

    models = {n: train(training, n) for n in ORDERS}
    print("[Model size: distinct stored n-grams]")
    print("  " + "  ".join(f"n={n}: {models[n].stored_ngrams}" for n in ORDERS))

    print(f"[Add-alpha at a context seen only in the k copies, alpha={ALPHA:g}, V={V}]")
    for k in (1, 4, 16):
        top, other = unseen_context_probs(k)
        print(
            f"  k={k}: p(true)={top:.4f}  p(other)={other:.4f}  ratio={top / other:.1f}  "
            f"bits per digit={math.log2(top / other):.2f}"
        )

    # One secret, step by step: why k* is a maximum over steps.
    example = next(s for s in secrets if s.copies == 4)
    print(f"[Copies needed, one secret: '{example.line}' (inserted {example.copies} times)]")
    for n in ORDERS:
        print(
            f"  n={n}: k*={copies_needed(models[n], example)}  greedy from '{example.prompt}' -> "
            f"'{greedy(models[n], [example.prompt], PIN_DIGITS)[0].replace(chr(10), '<EOL>')}'"
        )

    print("[Discoverable memorization: greedy from the true prefix; exact / at most one digit wrong / predicted by k*]")
    rates = {n: discoverable_rates(models[n], secrets) for n in ORDERS}
    for n in ORDERS:
        cells = "  ".join(f"k={d.copies}: {d.exact:.2f}/{d.approximate:.2f}/{d.predicted:.2f}" for d in rates[n])
        print(f"  n={n}: {cells}")
    agree = sum(
        int(round(d.exact * PER_GROUP)) == int(round(d.predicted * PER_GROUP)) for n in ORDERS for d in rates[n]
    )
    print(f"  groups where the k* prediction matches greedy exactly: {agree} of {len(ORDERS) * len(COPIES)}")

    n_sample = 5
    print(f"[Sampling from the true prefix, n={n_sample}: mean p(pin | prompt) and samples for 99% success]")
    for k in COPIES:
        prob = mean_sample_probability(models[n_sample], secrets, k)
        print(f"  k={k}: mean p={prob:.2e}  samples for 99%={samples_needed(prob, 0.01)}")

    print("[Exposure null: uninserted canaries]")
    for n in EXPOSURE_ORDERS:
        exps = all_exposures(canary_log_probs(train(training, n)))
        print(
            f"  n={n}: mean exposure of a non-member={float(np.mean(exps)):.3f}  "
            f"Pr[exposure >= 5]={float(np.mean(exps >= 5)):.5f} (bound {2**-5:.5f})"
        )
    tied = all_exposures(canary_log_probs(train(training, 8)))
    print(
        f"  n=8 (ties: unseen contexts give equal scores): mean={float(np.mean(tied)):.3f}  max={float(np.max(tied)):.2f}"
    )
    print(
        f"  uniform rank, no ties: E[exposure]={expected_null_exposure(PIN_SPACE):.4f}  log2 e={math.log2(math.e):.4f}  "
        f"log2|R|={math.log2(PIN_SPACE):.2f}"
    )

    canary_pin = random_pin(rng)
    print(f"[Exposure of the canary '{CANARY_PREFIX}{canary_pin}' against copies]")
    exposures: dict[int, list[float]] = {}
    for n in EXPOSURE_ORDERS:
        exposures[n] = [canary_exposure(n, training, canary_pin, k) for k in CANARY_COPIES]
        cells = "  ".join(f"k={k}: {e:.2f}" for k, e in zip(CANARY_COPIES, exposures[n], strict=True))
        print(f"  n={n}: {cells}")

    n_est, m_est = 3, 10_000
    idx = pin_index(canary_pin)
    print(f"[Estimating exposure, n={n_est}, from {m_est} random candidates]")
    for k in (1, 2, 4, 8):
        lps = canary_log_probs(train(training + [(CANARY_PREFIX + canary_pin, k)], n_est))
        print(
            f"  k={k}: exact={exposure(lps, idx):.2f}  rank={rank_of(lps, idx)}  "
            f"sampling={exposure_sampling_estimate(rng, lps, idx, m_est):.2f}  "
            f"normal fit={exposure_normal_estimate(rng, lps, idx, m_est):.2f}"
        )

    n_attack, samples, top = 6, 20_000, 100
    public = public_lines(rng)
    ref_background, _ = make_corpus(rng, copies=(), per_group=0)
    reference = train(with_public([(line, 1) for line in ref_background], public), n_attack)
    target = train(with_public(training, public), n_attack)
    result = extraction_attack(rng, target, reference, training, public, secrets, samples, top)
    print(
        f"[Extraction without the prefix, n={n_attack}, {len(public)} public lines x {PUBLIC_COPIES}] "
        f"samples={result.samples}  distinct={result.distinct}  private training lines among them={result.in_training}  "
        f"public lines among them={result.public_generated}"
    )
    print("  generated secrets by copies: " + "  ".join(f"k={k}: {v:.2f}" for k, v in result.by_copies.items()))
    print(
        f"  precision of top {top}: by likelihood={result.precision_likelihood:.2f}  "
        f"by ratio to reference={result.precision_ratio:.2f}"
    )


# --- Figures --------------------------------------------------------------------------------


def plot_copies_needed(background: Sequence[str], secrets: Sequence[Secret], order: int = 5) -> Figure:
    """For one secret, the counts at each step of its pin: the true digit against its best rival."""
    model = train(training_set(background, secrets), order)
    secret = next(s for s in secrets if s.copies == 1)
    own_keys = line_keys(secret.line, order)
    own_unique, own_mult = np.unique(own_keys, return_counts=True)
    full = pad(np.concatenate([encode(secret.prompt), pin_codes(secret.pin)]), order)
    ctx = context_codes(full, order)[0, -PIN_DIGITS:]
    counts = next_counts(model, ctx)
    query = ctx[:, None] * V + np.arange(V, dtype=np.int64)
    without = counts - secret.copies * lookup(own_unique.astype(np.int64), own_mult.astype(np.float64), query)
    pin = pin_codes(secret.pin)
    true = without[np.arange(PIN_DIGITS), pin]
    rival_counts = without.copy()
    rival_counts[np.arange(PIN_DIGITS), pin] = -np.inf
    rival = rival_counts.max(axis=1)
    steps = np.arange(PIN_DIGITS)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(steps - 0.2, true, width=0.4, label="true next digit, without the secret")
    ax.bar(steps + 0.2, rival, width=0.4, label="best rival digit")
    for t in range(PIN_DIGITS):
        need = int(rival[t] - true[t] + 1)
        ax.text(t, float(max(true[t], rival[t])) + 0.3, f"gap + 1 = {max(need, 1)}", ha="center", fontsize=9)
    full_text = secret.prompt + secret.pin
    labels = [
        repr(full_text[len(secret.prompt) + t - (order - 1) : len(secret.prompt) + t]) + f" → {secret.pin[t]}"
        for t in steps
    ]
    ax.set_xticks(steps, labels, fontsize=8)
    ax.set(
        xlabel=f"step: context (last {order - 1} characters) → true digit",
        ylabel="count in the training set",
        title=f"Copies needed for greedy decoding to emit '{secret.pin}' after '{secret.prompt.strip()}' (order {order})",
    )
    ax.set_ylim(0, max(float(rival.max()), float(true.max())) + 1.5)
    ax.legend(fontsize=9, loc="upper right")
    return fig


def plot_exposure_histogram(
    rng: np.random.Generator, background: Sequence[str], secrets: Sequence[Secret], order: int = 4
) -> Figure:
    training = training_set(background, secrets)
    canary_pin = random_pin(rng)
    idx = pin_index(canary_pin)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    base = canary_log_probs(train(training, order)) / math.log(2)
    ax.hist(base, bins=120, color="0.75", label=r"all $10^5$ candidate pins, canary not inserted")
    for k, color in ((1, "C0"), (4, "C1"), (16, "C2"), (64, "C3")):
        lps = canary_log_probs(train(training + [(CANARY_PREFIX + canary_pin, k)], order))
        e = exposure(lps, idx)
        ax.axvline(
            float(lps[idx]) / math.log(2),
            color=color,
            linestyle="--",
            label=f"canary inserted {k}×: exposure {e:.1f} bits",
        )
    ax.set(
        xlabel=r"$\log_2 p(r \mid$ 'my pin is'$)$ (bits)",
        ylabel="number of candidates",
        title=f"The canary moves into the right tail as it is repeated (order {order})",
    )
    ax.legend(fontsize=8, loc="upper left")
    return fig


def plot_exposure_vs_copies(
    rng: np.random.Generator, background: Sequence[str], secrets: Sequence[Secret], canaries: int = 3
) -> Figure:
    training = training_set(background, secrets)
    copies = [0, 1, 2, 4, 8, 16, 32, 64]
    pins = [random_pin(rng) for _ in range(canaries)]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = [k if k > 0 else 0.5 for k in copies]
    for j, n in enumerate(EXPOSURE_ORDERS):
        curves = np.array([[canary_exposure(n, training, pin, k) for k in copies] for pin in pins])
        for row in curves:
            ax.plot(xs, row, color=f"C{j}", alpha=0.25, linewidth=1)
        ax.plot(xs, curves.mean(axis=0), "o-", color=f"C{j}", label=f"order {n} (mean of {canaries} canaries)")
    ax.axhline(math.log2(PIN_SPACE), color="k", linestyle=":", label=r"maximum $\log_2 |R|$ = 16.6 bits")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, ["0"] + [str(k) for k in copies[1:]])
    ax.set(
        xlabel="copies of the canary in the training set",
        ylabel="exposure (bits)",
        ylim=(0, 18),
        title="Exposure grows with repetition, faster for larger models",
    )
    ax.legend(fontsize=9, loc="lower right")
    return fig


def plot_discoverable(rng: np.random.Generator) -> Figure:
    background, secrets = make_corpus(rng)
    training = training_set(background, secrets)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for j, n in enumerate(ORDERS):
        rates = discoverable_rates(train(training, n), secrets)
        ks = [d.copies for d in rates]
        left.plot(ks, [d.exact for d in rates], "o-", color=f"C{j}", label=f"order {n}")
        right.plot(ks, [d.approximate for d in rates], "o-", color=f"C{j}", label=f"order {n}")
    for ax, title in (
        (left, "Verbatim: greedy output equals the pin"),
        (right, "Approximate: at most one digit wrong"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(COPIES), [str(k) for k in COPIES])
        ax.set(xlabel="copies of the secret line in the training set", title=title, ylim=(-0.03, 1.3))
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    left.set_ylabel("fraction discoverably memorized")
    left.legend(fontsize=9, loc="upper left", ncol=3)
    return fig


def plot_extraction(rng: np.random.Generator, order: int = 6, samples: int = 20_000) -> Figure:
    background, secrets = make_corpus(rng)
    training = training_set(background, secrets)
    public = public_lines(rng)
    target = train(with_public(training, public), order)
    ref_background, _ = make_corpus(rng, copies=(), per_group=0)
    reference = train(with_public([(line, 1) for line in ref_background], public), order)
    lines = sorted(set(sample_lines(target, rng, samples)))
    members = {line for line, _ in training} - set(public)
    hits = np.array([line in members for line in lines], dtype=np.float64)
    lp_t = lines_log_probs(target, lines)
    lp_r = lines_log_probs(reference, lines)
    tops = np.arange(10, 1001, 10)
    fig, ax = plt.subplots(figsize=(7, 4))
    is_public = np.array([line in set(public) for line in lines], dtype=np.float64)
    by_likelihood = np.argsort(-lp_t, kind="stable")
    by_ratio = np.argsort(-(lp_t - lp_r), kind="stable")
    ax.plot(
        tops, [float(hits[by_likelihood[:m]].mean()) for m in tops], "C0", label="rank by likelihood: private lines"
    )
    ax.plot(
        tops,
        [float(is_public[by_likelihood[:m]].mean()) for m in tops],
        "C0--",
        label="rank by likelihood: public lines",
    )
    ax.plot(
        tops,
        [float(hits[by_ratio[:m]].mean()) for m in tops],
        "C1",
        label="rank by ratio to a reference model: private lines",
    )
    ax.axhline(float(hits.mean()), color="0.5", linestyle=":", label="all distinct generations")
    ax.set(
        xlabel="number of top-ranked generations checked",
        ylabel="fraction of the top-ranked lines",
        ylim=(0, 1.05),
        title=f"Generate, rank, verify: {samples:,} samples from an order-{order} model",
    )
    ax.legend(fontsize=8, loc="upper right")
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
    background, secrets = make_corpus(rng)
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "memo_copies_needed.svg": plot_copies_needed(background, secrets),
        "memo_exposure_histogram.svg": plot_exposure_histogram(rng, background, secrets),
        "memo_exposure_vs_copies.svg": plot_exposure_vs_copies(rng, background, secrets),
        "memo_discoverable.svg": plot_discoverable(rng),
        "memo_extraction.svg": plot_extraction(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
