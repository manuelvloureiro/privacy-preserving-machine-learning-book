"""Short real-library examples on Adult for the reference pages of practitioner Part 3.

    uv run python code/part03-dp-foundations/practitioner_adult_statistics.py

Each section below holds the example of one page, as a marked excerpt (prefix `adult-`), and prints its numbers on
labelled lines, page by page:

- 02 (properties): diffprivlib's BudgetAccountant as a ledger (a mean and a count of age, the total spent, the
  BudgetError on overspending) and the group-privacy calculation that the page's formula gives.
- 05 (selection): the private median of age, with diffprivlib (exponential mechanism over [17, 90]) and with OpenDP's
  Polars median over the candidate ages 17, ..., 90.
- 06 (query release): education x income counts with OpenDP's Polars group-by over public keys.
- 07 (sensitivity): diffprivlib's PrivacyLeakWarning when no bounds are given, turned into an error.
- 09 (trust models): local randomized response on the income label (diffprivlib's Binary at epsilon = ln 3),
  debiased, against a central count at the same epsilon.
- 11 (mean estimation): mean hours per week with loose bounds (1, 99) and tight bounds (20, 60), and the bias that the
  tight bounds bring.

Dataset: Adult (UCI Machine Learning Repository, dataset 2, "Census Income"; Becker & Kohavi, 1996; licence CC BY
4.0), https://archive.ics.uci.edu/static/public/2/adult.zip, loaded by code/practitioner_datasets.py: 48,842 people
from the 1994 US Current Population Survey, one row each, so the privacy unit is the row. delta = 1e-6 where one is
needed (1/n^1.1 = 7.0e-6, rounded down to a power of ten).

What is deterministic and what is not. diffprivlib takes `random_state`, and every diffprivlib call here passes 0 (or
`--seed`), so its outputs are the same on every run. OpenDP draws its noise from a cryptographic generator that cannot
be seeded: its scales and 95% accuracies are deterministic and quotable, its released values are printed under
"released (one run)" and the test checks them against the true values within a few times their accuracy.

Library versions: diffprivlib 0.6.6 (with scikit-learn 1.8.0), opendp[polars] 0.16.0 with polars 1.36.1.
"""

import argparse
import math
import sys
import warnings
from math import exp, log
from pathlib import Path
from typing import Any

import numpy as np
import opendp.prelude as dp
import polars as pl
from diffprivlib import BudgetAccountant, tools
from diffprivlib.mechanisms import Binary
from diffprivlib.utils import BudgetError, PrivacyLeakWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

dp.enable_features("contrib")

AGE_BOUNDS = (17, 90)  # the survey's age range, from its documentation
MAX_ROWS = 100_000  # a public upper bound on the number of rows, which OpenDP's median needs

# The 16 values of education in Adult's documentation (adult.names): a public codebook, not read from the data.
EDUCATION = [
    "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th",
    "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool",
]  # fmt: skip
INCOME = ["<=50K", ">50K"]

# ---------------------------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------------------------


def delta_for(n: int) -> float:
    """The book's default delta: 1/n^1.1, rounded down to a power of ten."""
    return 10.0 ** math.floor(math.log10(n**-1.1))


def group_privacy_simple(eps: float, delta: float, k: int) -> tuple[float, float]:
    """The looser closed form of group privacy, (k eps, k e^((k-1) eps) delta)."""
    return k * eps, k * exp((k - 1) * eps) * delta


# ---------------------------------------------------------------------------------------------------------------
# Page 02: a budget ledger and group privacy
# ---------------------------------------------------------------------------------------------------------------


def page_02(adult: pl.DataFrame, seed: int) -> None:
    print("== 02 ledger ==")
    # excerpt: adult-ledger
    age = adult["age"].to_numpy()
    acc = BudgetAccountant(epsilon=1.0, delta=1e-6)  # the total, fixed in advance
    mean_age = tools.mean(age, epsilon=0.5, bounds=(17, 90), accountant=acc, random_state=seed)
    over_50 = tools.count_nonzero(age > 50, epsilon=0.5, accountant=acc, random_state=seed)
    print(f"private mean age: {mean_age:.2f}  private count over 50: {over_50:,}")
    print(f"spent: {acc.total()}")
    try:
        tools.count_nonzero(age > 60, epsilon=0.1, accountant=acc, random_state=seed)
    except BudgetError as err:
        print(f"refused: {err}")
    # end excerpt
    eps_left, delta_left = acc.remaining()
    print(f"remaining: epsilon={eps_left:.2f} delta={delta_left:.0e}")
    print(f"true mean age: {age.mean():.2f}  true count over 50: {int((age > 50).sum()):,}")

    # excerpt: adult-group-privacy
    def group_privacy(eps: float, delta: float, k: int) -> tuple[float, float]:
        """The guarantee for a group of k records, from an (eps, delta) guarantee per record."""
        return k * eps, delta * sum(exp(j * eps) for j in range(k))

    eps_k, delta_k = group_privacy(0.5, 1e-5, 10)
    print(f"group of 10 at (0.5, 1e-5): ({eps_k:.1f}, {delta_k:.3e})")
    # end excerpt
    eps_s, delta_s = group_privacy_simple(0.5, 1e-5, 10)
    print(f"simple bound k e^((k-1) eps) delta: ({eps_s:.1f}, {delta_s:.3e})")
    print(f"delta multiplier, group of 10 at eps=1: {group_privacy(1.0, 1.0, 10)[1]:.1f}")


# ---------------------------------------------------------------------------------------------------------------
# Page 05: private median
# ---------------------------------------------------------------------------------------------------------------


def page_05(adult: pl.DataFrame, seed: int) -> float:
    """Prints the median examples; returns OpenDP's released median for the one-run block."""
    print("== 05 median ==")
    # excerpt: adult-median-diffprivlib
    age = adult["age"].to_numpy()
    private = tools.median(age, epsilon=1.0, bounds=(17, 90), random_state=seed)
    print(f"true median age: {np.median(age):.1f}  diffprivlib median: {private:.2f}")
    # end excerpt

    # excerpt: adult-median-opendp
    context = dp.Context.compositor(
        data=adult.lazy(),
        privacy_unit=dp.unit_of(contributions=1),  # one row per person
        privacy_loss=dp.loss_of(epsilon=1.0),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=100_000)],  # a public bound on the number of rows
    )
    age_col: Any = pl.col("age")  # OpenDP adds .dp to Polars expressions when it is imported
    query = context.query().select(age_col.dp.median(candidates=list(range(17, 91))))
    print(query.summarize(alpha=0.05))
    released = query.release().collect()  # changes on every run: OpenDP's noise has no seed
    # end excerpt
    summary = query.summarize(alpha=0.05).row(0, named=True)
    print(f"opendp median: {summary['aggregate']}, {summary['distribution']}, scale={summary['scale']:g}")
    return float(released["age"][0])


# ---------------------------------------------------------------------------------------------------------------
# Page 06: a marginal over public keys
# ---------------------------------------------------------------------------------------------------------------


def page_06(adult: pl.DataFrame) -> pl.DataFrame:
    """Prints the marginal's noise; returns the released counts joined with the true ones."""
    print("== 06 marginal ==")
    # excerpt: adult-marginal
    keys = pl.LazyFrame({"education": EDUCATION}).join(pl.LazyFrame({"income": INCOME}), how="cross")
    context = dp.Context.compositor(
        data=adult.lazy(),
        privacy_unit=dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(epsilon=1.0),
        split_evenly_over=1,
        margins=[dp.polars.Margin(by=["education", "income"], invariant="keys")],  # the cells are public
    )
    query = context.query().group_by("education", "income").agg(dp.len(signed=True)).with_keys(keys)
    print(query.summarize(alpha=0.05))
    released = query.release().collect()  # all 32 cells, zeros included; changes on every run
    # end excerpt
    summary = query.summarize(alpha=0.05).row(0, named=True)
    print(
        f"marginal cells: {keys.collect().height}  {summary['distribution']}, scale={summary['scale']:g}"
        f"  95% accuracy={summary['accuracy']:.2f}"
    )
    truth = adult.group_by("education", "income").agg(pl.len().cast(pl.Int64).alias("true"))
    joined = released.join(truth, on=["education", "income"], how="left").with_columns(pl.col("true").fill_null(0))
    smallest = joined.sort("true").row(0, named=True)
    print(f"smallest true cell: {smallest['education']} {smallest['income']} = {smallest['true']}")
    return joined.with_columns(pl.lit(summary["accuracy"]).alias("accuracy"))


# ---------------------------------------------------------------------------------------------------------------
# Page 07: diffprivlib's warning when bounds are missing
# ---------------------------------------------------------------------------------------------------------------


def page_07(adult: pl.DataFrame, seed: int) -> None:
    print("== 07 bounds warning ==")
    # excerpt: adult-bounds-warning
    age = adult["age"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound stops the run
        try:
            tools.median(age, epsilon=1.0)  # no bounds: diffprivlib would read them from the data
        except PrivacyLeakWarning as warning:
            print(f"{type(warning).__name__}: {warning}")
        private = tools.median(age, epsilon=1.0, bounds=(17, 90), random_state=seed)
        print(f"with bounds (17, 90): {private:.2f}")
    # end excerpt


# ---------------------------------------------------------------------------------------------------------------
# Page 09: local randomized response against a central count
# ---------------------------------------------------------------------------------------------------------------


def page_09(adult: pl.DataFrame, seed: int) -> None:
    print("== 09 local vs central ==")
    # excerpt: adult-local-rr
    answers = (adult["income"] == ">50K").cast(pl.Int8).cast(pl.String).to_list()  # "1" or "0"
    n, eps = len(answers), log(3)
    coin = Binary(epsilon=eps, value0="0", value1="1", random_state=seed)
    reports = [coin.randomise(a) for a in answers]  # each person randomizes their own answer
    p = exp(eps) / (1 + exp(eps))  # chance a report is truthful: 0.75
    reported = reports.count("1") / n
    local = (reported - (1 - p)) / (2 * p - 1)  # undo the expected flips
    local_sd = math.sqrt(p * (1 - p) / n) / (2 * p - 1)  # the randomization noise alone
    print(f"true share over 50K: {answers.count('1') / n:.4f}")
    print(f"local estimate: {local:.4f}  sd {local_sd:.4f}")
    # end excerpt
    # excerpt: adult-central-count
    rich = (adult["income"] == ">50K").to_numpy()
    central = tools.count_nonzero(rich, epsilon=log(3), random_state=seed) / len(rich)
    central_sd = math.sqrt(2) / log(3) / len(rich)  # a Laplace count with sensitivity 1
    print(f"central estimate: {central:.4f}  sd {central_sd:.6f}")
    # end excerpt
    print(f"local sd / central sd: {local_sd / central_sd:.0f}")


# ---------------------------------------------------------------------------------------------------------------
# Page 11: the bias of tight bounds
# ---------------------------------------------------------------------------------------------------------------


def page_11(adult: pl.DataFrame, seed: int) -> None:
    print("== 11 bounds and bias ==")
    # excerpt: adult-mean-bounds
    hours = adult["hours-per-week"].to_numpy()
    loose = tools.mean(hours, epsilon=1.0, bounds=(1, 99), random_state=seed)
    tight = tools.mean(hours, epsilon=1.0, bounds=(20, 60), random_state=seed)
    clipped = np.clip(hours, 20, 60).mean()  # what the tight bounds can at best recover
    print(f"true mean hours: {hours.mean():.3f}")
    print(f"bounds (1, 99): {loose:.3f}  bounds (20, 60): {tight:.3f}")
    print(f"bias of (20, 60): {clipped - hours.mean():+.3f}")
    # end excerpt
    outside = float(((hours < 20) | (hours > 60)).mean())
    print(f"people outside (20, 60): {outside:.1%}")
    print(f"noise scale (1, 99): {98 / (1.0 * len(hours)):.4f}  (20, 60): {40 / (1.0 * len(hours)):.4f}")


# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="diffprivlib's random_state (OpenDP's noise has none)")
    args = parser.parse_args()
    seed: int = args.seed

    adult = practitioner_datasets.adult()
    print(f"rows: {adult.height:,}")
    print(f"1/n^1.1 = {adult.height**-1.1:.1e}  delta: {delta_for(adult.height):.0e}")
    page_02(adult, seed)
    median = page_05(adult, seed)
    marginal = page_06(adult)
    page_07(adult, seed)
    page_09(adult, seed)
    page_11(adult, seed)

    print("released (one run):")
    print(f"  opendp median age: {median:g}")
    for row in marginal.sort("education", "income").iter_rows(named=True):
        print(f"  {row['education']} {row['income']}: {row['len']}  true {row['true']}  accuracy {row['accuracy']:.2f}")


if __name__ == "__main__":
    main()
