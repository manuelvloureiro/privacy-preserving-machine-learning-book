"""Re-identification risk of a record-level table: uniqueness on quasi-identifiers, generalization, distinct l.

    uv run python code/part02-attacks/practitioner_table_risk.py

The table step of the practitioner edition's risk-assessment chapter (book-practitioner/part02-attacks/
10-risk-assessment.md, step 5), and the `qi-uniqueness` and `generalize-and-check` excerpts of
07-linkage-reidentification.md and 08-syntactic-anonymity.md.

Data: Adult (UCI Machine Learning Repository, dataset 2, "Census Income"; Becker & Kohavi, 1996; licence CC BY 4.0),
the 48,842 rows of adult.data and adult.test, the same rows as OpenML data_id 1590, loaded by
`practitioner_datasets.adult()` into data/uci/. Each row is one person from the 1994 US Current Population Survey.
Missing values are null; Polars groups nulls together, so a missing native-country is one more value of it.

Quasi-identifiers (QIs), nested: age, sex, race, then + native-country, + marital-status, + education, and
+ hours-per-week to show how one more column undoes everything. For each set it prints the share of records that
are unique on it (equivalence class of size 1), the share in classes smaller than 5, the smallest class size k, and
the number of classes per record (J/n). Adult is a sample of the population, so a record unique in Adult need not
be unique among all people: these are prosecutor-model figures (the attacker knows the person is in the table).

Then it generalizes: age to 10-year bands, native-country to US/other (missing goes to other), education to four
levels (no diploma, high-school graduate, some college or associate degree, bachelor's or higher). It rechecks the
same nested sets, and measures distinct l on occupation (the sensitive stand-in) over the six-QI classes: how
many records sit in a class where everyone has the same occupation, so that knowing the class gives it away.
Missing occupation counts as one value. Pure counting, so the numbers are exact and do not depend on the seed.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from practitioner_datasets import adult

QI_SETS: list[list[str]] = [
    ["age", "sex", "race"],
    ["age", "sex", "race", "native-country"],
    ["age", "sex", "race", "native-country", "marital-status"],
    ["age", "sex", "race", "native-country", "marital-status", "education"],
    ["age", "sex", "race", "native-country", "marital-status", "education", "hours-per-week"],
]
QIS = QI_SETS[3]  # the six quasi-identifiers agreed for the release
SMALL_CLASS = 5  # classes below this size count as small (11 for health data); the excerpt writes 5
SENSITIVE = "occupation"


# ---------------------------------------------------------------------------
# Uniqueness on a set of quasi-identifiers
# ---------------------------------------------------------------------------


def print_risks(df: pl.DataFrame) -> None:
    """Share unique, share in small classes, smallest class and classes per record, for each nested QI set."""
    # excerpt: qi-uniqueness
    n = df.height
    for qis in QI_SETS:
        # One row per equivalence class (records that agree on every quasi-identifier), with its size.
        sizes = df.group_by(qis).len()
        unique = sizes.filter(pl.col("len") == 1).height / n  # share of records alone in their class
        small = sizes.filter(pl.col("len") < 5)["len"].sum() / n  # share in classes smaller than 5
        min_k = sizes["len"].min()  # the table is k-anonymous for this k
        print(
            f"{' + '.join(qis)}: unique={unique:.2%}  in classes <5={small:.2%}  "
            f"min k={min_k}  J/n={sizes.height / n:.3f}"
        )
    # end excerpt


# ---------------------------------------------------------------------------
# Generalization and distinct l
# ---------------------------------------------------------------------------


def generalize_and_check(df: pl.DataFrame) -> pl.DataFrame:
    # excerpt: generalize-and-check
    general = df.with_columns(
        (pl.col("age") // 10 * 10).alias("age"),  # 10-year bands: 30 stands for 30-39
        pl.when(pl.col("native-country") == "United-States")
        .then(pl.lit("US"))
        .otherwise(pl.lit("other"))
        .alias("native-country"),
        # education-num runs from 1 (preschool) to 16 (doctorate); 9 is high-school graduate.
        pl.col("education-num")
        .cut([8, 9, 12], labels=["no diploma", "HS-grad", "some college", "bachelor+"])
        .alias("education"),
    )
    # Recheck, and count the occupations in each class (distinct l).
    classes = general.group_by(QIS).agg(pl.len(), pl.col("occupation").n_unique().alias("l"))
    unique = classes.filter(pl.col("len") == 1).height / general.height
    one_job = classes.filter(pl.col("l") == 1)["len"].sum() / general.height
    print(f"after generalizing: unique={unique:.2%}  in a class with one occupation={one_job:.2%}")
    # end excerpt
    return general


@dataclass(frozen=True)
class Diversity:
    one_value: float  # share of records in classes with one value of SENSITIVE
    one_value_big: float  # ... among classes of size >= SMALL_CLASS, as a share of all records
    min_l: int
    median_l: float


def distinct_l(df: pl.DataFrame, qis: list[str]) -> Diversity:
    classes = df.group_by(qis).agg(pl.len(), pl.col(SENSITIVE).n_unique().alias("l"))
    single = classes.filter(pl.col("l") == 1)
    big_single = single.filter(pl.col("len") >= SMALL_CLASS)
    min_l = classes["l"].min()
    assert isinstance(min_l, int)
    median_l = classes["l"].median()
    assert isinstance(median_l, float)
    return Diversity(single["len"].sum() / df.height, big_single["len"].sum() / df.height, min_l, median_l)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report() -> None:
    df = adult()
    print(f"[Data] Adult: {df.height:,} records")
    print("\n[Raw quasi-identifiers]")
    print_risks(df)

    print("\n[Generalized: age in 10-year bands, country US/other, education in 4 levels]")
    general = generalize_and_check(df)
    print_risks(general)

    d = distinct_l(general, QIS)
    print(f"\n[Distinct l on {SENSITIVE}, generalized classes of the six QIs]")
    print(f"  records in classes with l=1: {d.one_value:.2%}")
    print(f"  records in classes of size >={SMALL_CLASS} with l=1: {d.one_value_big:.2%}")
    print(f"  min l={d.min_l}  median l over classes={d.median_l:g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="unused: the counts are exact")
    parser.parse_args()
    report()


if __name__ == "__main__":
    main()
