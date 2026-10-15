"""Check the worked solutions to the exercises of the memorization and training-data extraction chapter.

    uv run python code/part02-attacks/solutions_memorization.py

Prints every number quoted in book-exercises/part02-attacks/05-memorization-extraction.md (seed 0, which rebuilds the
chapter's corpus and canary). Makes no figures; --out is accepted and ignored.
"""

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from memorization import (
    CANARY_PREFIX,
    PIN_SPACE,
    NgramModel,
    all_exposures,
    canary_log_probs,
    expected_null_exposure,
    exposure,
    lines_log_probs,
    make_corpus,
    pin_index,
    random_pin,
    sample_lines,
    samples_needed,
    train,
    training_set,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: expected exposure of a small space -----------------------------------------


def null_exposure_tail(size: int, e: float) -> float:
    """Pr[exposure ≥ e] for a rank uniform on 1..size: floor(size·2^{-e}) / size."""
    return math.floor(size * 2.0**-e) / size


# --- Exercise 2: samples for extraction -----------------------------------------------------


def samples_bound(prob: float, failure: float) -> int:
    """m ≥ ln(1/β)/p, from (1 - p)^m ≤ e^{-mp}."""
    return math.ceil(math.log(1 / failure) / prob)


# --- Exercise 4: copies needed by hand ------------------------------------------------------


def copies_needed_from_counts(steps: Sequence[tuple[int, int]], ties_win: bool = False) -> int:
    """k* = max_t (rival_t - true_t) + 1 for (true, rival) counts without the secret; one less when ties win."""
    return max(1, max(rival - true + (0 if ties_win else 1) for true, rival in steps))


# --- Exercise 5: exposure as a membership test ----------------------------------------------


def planted_exposures(
    order: int, background: list[tuple[str, int]], planted: Sequence[str], held_out: Sequence[str]
) -> tuple[list[float], list[float]]:
    """Exposures of the planted canaries (one copy each, all in one model) and of pins never planted."""
    model = train(background + [(CANARY_PREFIX + pin, 1) for pin in planted], order)
    scores = canary_log_probs(model)
    return (
        [exposure(scores, pin_index(pin)) for pin in planted],
        [exposure(scores, pin_index(pin)) for pin in held_out],
    )


# --- Exercise 6: which secrets an unprompted sampler generates ------------------------------


def generation_prediction(model: NgramModel, lines: Sequence[str], samples: int) -> tuple[float, float]:
    """Mean over lines of p(line) and of Pr[line appears among `samples` samples] = 1 - (1 - p)^m."""
    probs = np.exp(lines_log_probs(model, lines))
    return float(np.mean(probs)), float(np.mean(1 - (1 - probs) ** samples))


def print_table(rng: np.random.Generator) -> None:
    background, secrets = make_corpus(rng)
    training = training_set(background, secrets)
    canary_pin = random_pin(rng)  # the canary of the chapter's exposure table

    print("[Exercise 1: null exposure for |R| = 10]")
    print(
        f"  E[exposure]={expected_null_exposure(10):.4f}  Pr[exposure >= 2]={null_exposure_tail(10, 2):.1f}  "
        f"Pr[exposure >= 1]={null_exposure_tail(10, 1):.1f}"
    )
    print(f"  |R| = 4: E[exposure]={expected_null_exposure(4):.4f}")

    print("[Exercise 2: samples for 99% with p = 0.0442]")
    p = 0.0442
    print(
        f"  bound ln(100)/p={math.log(100) / p:.2f} -> {samples_bound(p, 0.01)}  exact={samples_needed(p, 0.01)}  "
        f"Pr[success] with 102 samples={1 - (1 - p) ** 102:.4f}  with 101={1 - (1 - p) ** 101:.4f}"
    )

    print("[Exercise 3: canary in unique contexts]")
    for order in (6, 8):
        scores = canary_log_probs(train(training + [(CANARY_PREFIX + canary_pin, 1)], order))
        print(
            f"  order {order}, k=1: exposure={exposure(scores, pin_index(canary_pin)):.2f}  "
            f"log2|R|={math.log2(PIN_SPACE):.2f}"
        )
    v, alpha = 39, 0.1
    top = (1 + alpha) / (1 + alpha * v)
    print(f"  k=1: p(true)={top:.4f} >= 1/V={1 / v:.4f}")

    print("[Exercise 4: copies needed from counts]")
    steps = [(1, 2), (0, 2), (0, 5), (0, 1), (0, 1)]
    split = [(1, 2), (0, 2), (0, 2), (0, 1), (0, 1)]
    print(
        f"  per step: {[rival - true + 1 for true, rival in steps]}  k*={copies_needed_from_counts(steps)}  "
        f"after the split: k*={copies_needed_from_counts(split)}  "
        f"if ties went the true digit's way: k*={copies_needed_from_counts(steps, ties_win=True)}"
    )

    print("[Exercise 5: exposure >= 3 as a test, order 3, 20 planted and 20 held-out canaries]")
    planted = [random_pin(rng) for _ in range(20)]
    held_out = [random_pin(rng) for _ in range(20)]
    base = [(line, m) for line, m in training]
    hits, misses = planted_exposures(3, base, planted, held_out)
    threshold = 3.0
    tpr = float(np.mean([e >= threshold for e in hits]))
    fpr = float(np.mean([e >= threshold for e in misses]))
    print(
        f"  bound on FPR={2**-threshold:.3f}  TPR={tpr:.2f}  FPR={fpr:.2f}  "
        f"mean exposure planted={float(np.mean(hits)):.2f}  held out={float(np.mean(misses)):.2f}"
    )
    exps = all_exposures(canary_log_probs(train(base + [(CANARY_PREFIX + pin, 1) for pin in planted], 3)))
    planted_idx = {pin_index(pin) for pin in planted}
    others = np.array([e for i, e in enumerate(exps) if i not in planted_idx])
    print(
        f"  all {len(others)} unplanted pins: fraction with exposure >= 3 = {float(np.mean(others >= threshold)):.4f}"
    )

    print("[Exercise 6: unprompted sampling from the order-6 model (no public lines), 20,000 samples]")
    model = train(training, 6)
    generated = set(sample_lines(model, rng, 20_000))
    for k in (1, 4, 8, 16, 32):
        lines = [sec.line for sec in secrets if sec.copies == k]
        mean_p, predicted = generation_prediction(model, lines, 20_000)
        observed = float(np.mean([line in generated for line in lines]))
        print(
            f"  k={k}: mean p(line)={mean_p:.2e}  expected copies in 20,000={20_000 * mean_p:.3f}  "
            f"predicted fraction generated={predicted:.2f}  observed={observed:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="accepted for a uniform interface; no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
