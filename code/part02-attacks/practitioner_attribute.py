"""Attribute inference on a census classifier with ART: the attack, the imputation baseline, and the advantage.

    uv run python code/part02-attacks/practitioner_attribute.py

The `attribute-art` excerpt of the practitioner edition's reference page on attribute inference
(book-practitioner/part02-attacks/03-attribute-inference.md).

Data: Adult (UCI Machine Learning Repository, dataset 2, "Census Income"; Becker & Kohavi, 1996; licence CC BY 4.0),
the 48,842 rows of adult.data and adult.test, loaded by `practitioner_datasets.adult()` into data/uci/. Rows with a
missing value are dropped (45,222 remain). The sensitive field is sex, coded 1 for female; the other inputs are age,
years of education, capital gain and loss, hours per week, and one-hot workclass, marital status, occupation,
relationship, race and native country. The survey weight `fnlwgt` and the text column `education`, which repeats
years of education, are left out.

Three disjoint random sets of 4,000 people (seed 0): the target's training members, non-members from the same
population, and the attacker's own records. The target is a scikit-learn random forest (100 trees, fully grown,
random_state 0) predicting whether income exceeds $50K from every input, sex included, wrapped in ART's
`SklearnClassifier`. The attacker holds everyone's other inputs and income, and queries the model for its label.

- ART's `AttributeInferenceBlackBox` learns sex from the other inputs and the model's predicted label, on the
  attacker's records, then guesses it for members and non-members.
- ART's `AttributeInferenceBaselineTrueLabel` learns sex from the other inputs and the true income alone, without
  the model: the imputation baseline.
Both use ART's default random forest as the attack model, seeded through NumPy's global generator. It prints each success rate with an exact 95%
interval (Clopper-Pearson, from scipy.stats.binomtest), and the attribute advantage, members minus non-members, with
a normal-approximation 95% interval. Pure scikit-learn and seeded, so every printed digit is reproducible.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from art.attacks.inference.attribute_inference import AttributeInferenceBaselineTrueLabel, AttributeInferenceBlackBox
from art.estimators.classification import SklearnClassifier
from numpy.typing import NDArray
from scipy.stats import binomtest, norm
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from practitioner_datasets import adult

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

N_EACH = 4_000  # members, non-members and the attacker's own records
NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "marital-status", "occupation", "relationship", "race", "native-country"]
SENSITIVE = 0  # column of x that holds the sensitive field


def features(df: pl.DataFrame) -> tuple[Array, IntArray]:
    """Sex (1 = female) in column 0, then the numeric and one-hot inputs; the label is income over $50K."""
    df = df.drop_nulls()
    female = (df["sex"] == "Female").cast(pl.Float64).to_frame("female")
    dummies = df.select(CATEGORICAL).to_dummies()
    x = pl.concat([female, df.select(NUMERIC).cast(pl.Float64), dummies.cast(pl.Float64)], how="horizontal")
    y = (df["income"] == ">50K").cast(pl.Int64).to_numpy()
    return x.to_numpy(), y


def rate(hits: int, n: int) -> str:
    ci = binomtest(hits, n).proportion_ci(confidence_level=0.95, method="exact")
    return f"{hits / n:.4f} [{ci.low:.4f}, {ci.high:.4f}]"


def report(seed: int) -> None:
    x, y = features(adult())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    members, others, attacker = (order[i * N_EACH : (i + 1) * N_EACH] for i in range(3))
    print(f"[Data] Adult without missing values: {len(y):,} people, {x.shape[1]} inputs; {N_EACH:,} in each set")
    female = x[:, SENSITIVE]
    print(f"share female: members {female[members].mean():.4f}  non-members {female[others].mean():.4f}")

    forest = RandomForestClassifier(random_state=seed, n_jobs=1).fit(x[members], y[members])
    target = SklearnClassifier(forest)
    print(
        f"target: train accuracy {forest.score(x[members], y[members]):.4f}  "
        f"test accuracy {forest.score(x[others], y[others]):.4f}"
    )

    # excerpt: attribute-art
    def label(rows: IntArray) -> Array:  # the label the model returns for each record, as a column
        return np.argmax(target.predict(x[rows]), axis=1).reshape(-1, 1).astype(np.float64)

    np.random.seed(seed)  # ART builds its random forests unseeded, from NumPy's global generator
    attack = AttributeInferenceBlackBox(target, attack_model_type="rf", attack_feature=SENSITIVE)
    attack.fit(x[attacker])  # the attacker's own records, sensitive field included
    baseline = AttributeInferenceBaselineTrueLabel(attack_model_type="rf", attack_feature=SENSITIVE)
    baseline.fit(x[attacker], y[attacker])  # the same records and the true label, without the model
    hits: dict[str, list[int]] = {}
    for name, rows in (("members", members), ("non-members", others)):
        known = np.delete(x[rows], SENSITIVE, axis=1)  # everything but the sensitive field
        guess = attack.infer(known, pred=label(rows), values=[0.0, 1.0])
        imputed = baseline.infer(known, y=y[rows], values=[0.0, 1.0])
        hits[name] = [int(np.count_nonzero(np.equal(g.ravel(), female[rows]))) for g in (guess, imputed)]
    # end excerpt

    for i, method in enumerate(("ART black-box attack", "imputation baseline")):
        a, b = hits["members"][i], hits["non-members"][i]
        print(f"{method}: right on members {rate(a, N_EACH)}  on non-members {rate(b, N_EACH)}")
        p1, p0 = a / N_EACH, b / N_EACH
        half = norm.ppf(0.975) * math.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / N_EACH)
        print(f"{method}: advantage {p1 - p0:+.4f} [{p1 - p0 - half:+.4f}, {p1 - p0 + half:+.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed: int = args.seed
    report(seed)


if __name__ == "__main__":
    main()
