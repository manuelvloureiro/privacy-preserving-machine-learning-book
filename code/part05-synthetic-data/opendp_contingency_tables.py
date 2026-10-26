"""Contingency tables with OpenDP: the chapter "Contingency tables" of the OpenDP book (Part 5).

    uv run python code/part05-synthetic-data/opendp_contingency_tables.py

Follows the documentation's Contingency Tables page, in its order: the context under zero-concentrated DP with a
delta; keys for the categorical column SEX, cuts for AGE and HWUSUAL, and stable keys for ILOSTAT; the Fixed
algorithm with a workload of two Count queries and oneway_split = 1/8, whose one-way scale and threshold are read
before the release; the released ContingencyTable (keys, cuts, thresholds, schema, project, project_melted, std);
the column SEX that no marginal covers, and a second query that adds its marginal to the same model with table=;
and a dev-time check against the true one-way counts of ILOSTAT and the true AGE x HWUSUAL table, which this public
data allows. It prints every number the page quotes, on labelled lines, in the order of the page's sections.

Data: the France Labour Force Survey sample of the OpenDP tutorials, loaded through code/practitioner_datasets.py
(the documentation reads the same file with dp.examples.get_france_lfs_path()). 3,811,932 quarterly responses from
1,682,306 people; a person appears at most once per quarter over 9 years, so at most 36 times, which is the privacy
unit (dp.unit_of(contributions=36)). The budgets are the documentation's: rho = 0.2 with delta = 1e-7 for the
context, rho = 0.1 with delta = 1e-7 for the first table and rho = 0.05 with delta = 0 for the second.

What is deterministic and what is not. OpenDP draws its noise from a cryptographic generator that cannot be seeded,
so the released projections change on every run; they are printed under "released (one run)" and the test checks
them against the true values within three times their 95% accuracy. Everything else is the same on every run: the
budgets, the one-way scale and threshold, the keys, cuts and schema, the standard deviations and accuracies, the
error messages and the true counts. The --seed flag is accepted like in every script of the book, but nothing in
this chapter is sampled.

Library versions: opendp[mbi,polars] 0.16.0 with polars 1.36.1 and mbi 1.1.0, the Private-PGM version that
opendp 0.16.0 pins; mbi fits its models with JAX, which the script runs on the CPU.
"""

import argparse
import math
import os
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import opendp.prelude as dp
import polars as pl
from mbi import estimation
from opendp.extras.polars.contingency_table import ContingencyTableQuery

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # mbi fits its models with JAX; there is no GPU to look for

dp.enable_features("contrib")

CONTRIBUTIONS = 36  # a person answers at most once per quarter, 9 years x 4 quarters
RHO = 0.2  # the documentation's budget for the context
DELTA = 1e-7  # pays for the stable keys of the column without keys or cuts
TABLE_RHO = 0.1  # the first table's share of rho
SEX_RHO = 0.05  # the second table's share of rho
ONEWAY_SPLIT = 1 / 8  # the share of the first table's rho spent on the one-way marginal of ILOSTAT
ALPHA = 0.05  # the error bars are 95% ones
COLUMNS = ("SEX", "AGE", "HWUSUAL", "ILOSTAT")
WORKLOAD = (("AGE", "HWUSUAL"), ("HWUSUAL", "ILOSTAT"))
TRUE_ILOSTAT = {1: 1_517_011, 2: 155_661, 3: 1_450_283, 9: 688_977}  # dev-time counts on this public data
REFIT_STEPS = 10_000  # the dev-time refit of the released marginals; mbi's mirror_descent runs 1,000 by default


def load() -> pl.LazyFrame:
    """The LFS sample as a LazyFrame; the Parquet file is cached in data/."""
    return pl.scan_parquet(practitioner_datasets.france_lfs())


# ---------------------------------------------------------------------------------------------------------------
# The context and the queries
# ---------------------------------------------------------------------------------------------------------------


# excerpt: ctab-context
def make_context(frame: pl.LazyFrame) -> dp.Context:
    """The context: 36 contributions per person, rho = 0.2 of zCDP with delta = 1e-7, spent query by query."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),  # a person answers in at most 36 quarters
        privacy_loss=dp.loss_of(rho=RHO, delta=DELTA),  # the delta pays for the stable keys of ILOSTAT
    )


# end excerpt


# excerpt: ctab-query
def workload_query(context: dp.Context) -> ContingencyTableQuery:
    """The documentation's query: keys for SEX, cuts for AGE and HWUSUAL, stable keys for ILOSTAT, two counts."""
    return (
        context.query(rho=TABLE_RHO, delta=DELTA)
        .select("SEX", "AGE", "HWUSUAL", "ILOSTAT")
        .contingency_table(
            keys={"SEX": [1, 2]},  # public: the codebook's two codes
            cuts={"AGE": [20, 40, 60], "HWUSUAL": [1, 20, 40]},  # left-open bins: (-inf, 20], (20, 40], ...
            algorithm=dp.mbi.Fixed(
                queries=[dp.mbi.Count(("AGE", "HWUSUAL")), dp.mbi.Count(("HWUSUAL", "ILOSTAT"))],
                oneway_split=ONEWAY_SPLIT,  # the share of rho for the one-way marginal of ILOSTAT
            ),
        )
    )


# end excerpt


# excerpt: ctab-adaptive
def sex_query(context: dp.Context, table: dp.mbi.ContingencyTable) -> ContingencyTableQuery:
    """A second query that adds the marginal of SEX to the released table's model; every key is known, so delta = 0."""
    return (
        context.query(rho=SEX_RHO, delta=0.0)
        .select("SEX")
        .contingency_table(
            table=table,  # keep the keys, cuts, marginals and model of the first release
            algorithm=dp.mbi.Fixed(queries=[dp.mbi.Count(("SEX",))]),
        )
    )


# end excerpt


def refusal(build: Any) -> str:
    """The message of the ValueError that building a query raises."""
    try:
        build()
    except ValueError as err:
        return str(err)
    return "built without an error"


def gaussian_scale(rho: float, sensitivity: float = CONTRIBUTIONS) -> float:
    """The Gaussian scale that spends rho of zCDP on a count of the given L2 sensitivity: sensitivity / sqrt(2 rho)."""
    return sensitivity / math.sqrt(2 * rho)


# ---------------------------------------------------------------------------------------------------------------
# Reading the table and the dev-time truth
# ---------------------------------------------------------------------------------------------------------------


def keys_text(series: pl.Series) -> str:
    """A key set as the page prints it: [1, 2, null]."""
    return "[" + ", ".join("null" if k is None else str(k) for k in series.to_list()) + "]"


def accuracy_line(label: str, std: float) -> str:
    return f"std {label}: {std:.2f}  95% accuracy: {dp.gaussian_scale_to_accuracy(std, alpha=ALPHA):.2f}"


def projection(table: dp.mbi.ContingencyTable, attrs: Sequence[str]) -> list[int]:
    """A projection's counts rounded to integers, in the order of the keys."""
    return [round(float(v)) for v in np.asarray(table.project(attrs)).ravel()]


def binned(column: str, cuts: Sequence[float], labels: Sequence[str]) -> pl.Expr:
    """The column cut into OpenDP's left-open bins; a null lands in the last bin, as in the released table."""
    return pl.col(column).cut(cuts, labels=list(labels)).cast(pl.String).fill_null(labels[-1]).alias(column)


def true_two_way(frame: pl.LazyFrame, table: dp.mbi.ContingencyTable) -> pl.DataFrame:
    """The true AGE x HWUSUAL table for the released keys, one row per cell in the projection's order."""
    exprs = [binned(c, table.cuts[c].to_list(), table.keys[c].to_list()) for c in ("AGE", "HWUSUAL")]
    truth = frame.select(exprs).group_by("AGE", "HWUSUAL").agg(pl.len().alias("true")).collect()
    return table.project_melted(("AGE", "HWUSUAL")).join(truth, on=["AGE", "HWUSUAL"], how="left").fill_null(0)


def ilostat_lines(table: dp.mbi.ContingencyTable, indent: str = "  ", counts: Sequence[int] | None = None) -> list[str]:
    """One line per key of ILOSTAT with the released count (or the counts given) against the true one."""
    if counts is None:
        counts = projection(table, ("ILOSTAT",))
    lines: list[str] = []
    for key, count in zip(table.keys["ILOSTAT"].to_list(), counts):
        true = 0 if key is None else TRUE_ILOSTAT[int(key)]
        lines.append(
            f"{indent}ILOSTAT {'null' if key is None else key}: {count:,}  (true {true:,}, off by {count - true:+,})"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="accepted like in every script; nothing here is sampled")
    parser.parse_args()

    print(f"opendp version: {dp.__version__}  mbi version: {version('mbi')}")
    frame = load()

    # Section: from grouped counts to a contingency table. The sizes that make the full table hopeless.
    cells = frame.select(pl.col(c).n_unique() for c in COLUMNS).collect().row(0)
    print("distinct values: " + "  ".join(f"{c}={n}" for c, n in zip(COLUMNS, cells)))
    print(f"cells in the full table of the four raw columns: {math.prod(cells):,}")
    truth = frame.group_by("ILOSTAT").agg(pl.len()).sort("ILOSTAT").collect()
    print("true ILOSTAT counts: " + "  ".join(f"{r['ILOSTAT']}={r['len']:,}" for r in truth.iter_rows(named=True)))
    print(f"ILOSTAT codes in the codebook but not in the data: {sorted({1, 2, 3, 4, 5, 9} - set(truth['ILOSTAT']))}")
    nulls = frame.select(pl.col("AGE").is_null().sum().alias("age"), pl.col("HWUSUAL").is_null().sum().alias("hw"))
    age_nulls, hw_nulls = nulls.collect().row(0)
    print(f"rows with AGE null: {age_nulls:,}  with HWUSUAL null: {hw_nulls:,}")

    # Section: the context. A delta is needed exactly when some column has neither keys nor cuts.
    context = make_context(frame)
    print(f"privacy unit: {dp.unit_of(contributions=CONTRIBUTIONS)!r}")
    print(f"privacy loss: {dp.loss_of(rho=RHO, delta=DELTA)!r}")
    probe = make_context(frame)  # a separate context, so that the refused queries never touch the budget above
    fixed_sex = dp.mbi.Fixed(queries=[dp.mbi.Count(("SEX",))])
    all_keyed = refusal(
        lambda: (
            probe.query(rho=0.01, delta=DELTA)
            .select("SEX")
            .contingency_table(keys={"SEX": [1, 2]}, algorithm=fixed_sex)
        )
    )
    print(f"delta with keys for every column: {all_keyed}")
    unkeyed = refusal(
        lambda: (
            probe.query(rho=0.01, delta=0.0)
            .select("SEX", "ILOSTAT")
            .contingency_table(keys={"SEX": [1, 2]}, algorithm=fixed_sex)
        )
    )
    print(f"no delta with ILOSTAT unkeyed: {unkeyed}")

    # Section: the Fixed algorithm. The one-way scale and threshold are known before anything is released.
    query = workload_query(context)
    oneway_rho = TABLE_RHO * ONEWAY_SPLIT
    marginal_rho = (TABLE_RHO - oneway_rho) / len(WORKLOAD)
    print(
        f"oneway split: {ONEWAY_SPLIT:g} of rho={TABLE_RHO:g} -> rho={oneway_rho:g} one-way, rho={marginal_rho:g} per count"
    )
    print(f"oneway scale: {query.oneway_scale:.2f}  oneway threshold: {query.oneway_threshold}")
    print(f"scale from the formula {CONTRIBUTIONS} / sqrt(2 x {oneway_rho:g}): {gaussian_scale(oneway_rho):.2f}")
    print(
        f"scale of each count of the workload, {CONTRIBUTIONS} / sqrt(2 x {marginal_rho:g}): {gaussian_scale(marginal_rho):.2f}"
    )

    # Section: the released table.
    # excerpt: ctab-release
    table = query.release()
    ilostat_std = table.std(("ILOSTAT",))
    ilostat_accuracy = dp.gaussian_scale_to_accuracy(ilostat_std, alpha=ALPHA)
    # end excerpt
    print("keys: " + "  ".join(f"{c}={keys_text(table.keys[c])}" for c in COLUMNS))
    print("cuts: " + "  ".join(f"{c}={table.cuts[c].to_list()}" for c in ("AGE", "HWUSUAL")))
    print(f"thresholds: {table.thresholds}")
    print(f"schema: {table.schema}")
    print(accuracy_line("ILOSTAT", ilostat_std))
    two_way_std = table.std(("AGE", "HWUSUAL"))
    two_way_accuracy = dp.gaussian_scale_to_accuracy(two_way_std, alpha=ALPHA)
    print(accuracy_line("AGE x HWUSUAL", two_way_std))
    print(accuracy_line("HWUSUAL", table.std(("HWUSUAL",))))
    print(f"std SEX: {refusal(lambda: table.std(('SEX',)))}")

    # Section: adaptive estimation. The second query adds the marginal of SEX to the same model.
    sex_before = projection(table, ("SEX",))
    table2 = sex_query(context, table).release()
    sex_std = table2.std(("SEX",))
    sex_accuracy = dp.gaussian_scale_to_accuracy(sex_std, alpha=ALPHA)
    print(accuracy_line("SEX after the second query", sex_std))
    print(f"marginals in the model: {table2.marginals.cliques()}")
    print(f"privacy loss spent: {context.current_privacy_loss()}")

    # Section: the check against the truth, which this public data allows.
    two_way = true_two_way(frame, table)
    print("true AGE x HWUSUAL (a null AGE or HWUSUAL counts in the last bin):")
    for age in table.keys["AGE"].to_list():
        row = two_way.filter(pl.col("AGE") == age).sort("HWUSUAL")
        print(f"  AGE {age}: " + "  ".join(f"{r['HWUSUAL']}={r['true']:,}" for r in row.iter_rows(named=True)))
    print(
        f"test bounds, 3 x accuracy: {3 * ilostat_accuracy:,.1f} for ILOSTAT, {3 * two_way_accuracy:,.1f} per two-way cell, "
        f"{3 * sex_accuracy:,.1f} for SEX"
    )
    print("released (one run):")
    for line in ilostat_lines(table):
        print(line)
    errors = (two_way["len"] - two_way["true"]).abs()
    worst = two_way.row(int(errors.arg_max() or 0), named=True)
    print(f"  AGE x HWUSUAL largest |released - true|: {errors.max():,.0f}  (accuracy {two_way_accuracy:.2f})")
    print(f"  at AGE {worst['AGE']} HWUSUAL {worst['HWUSUAL']}: {worst['len']:,.0f}  (true {worst['true']:,})")
    # The last bin of both columns is where a null AGE or HWUSUAL lands; a wrong placement would put it 20,819 off.
    last = two_way.filter((pl.col("AGE") == table.keys["AGE"][-1]) & (pl.col("HWUSUAL") == table.keys["HWUSUAL"][-1]))
    last_len, last_true = round(float(last["len"].item())), int(last["true"].item())
    print(f"  last bin of both, the nulls' cell: {last_len:,}  (true {last_true:,}, off by {last_len - last_true:+,})")
    print(f"  SEX before the second query: {sex_before[0]:,} {sex_before[1]:,} {sex_before[2]:,}  (uniform)")
    sex_after = projection(table2, ("SEX",))
    print(
        f"  SEX after: 1: {sex_after[0]:,}  (true 1,828,992)  2: {sex_after[1]:,}  (true 1,982,940)  null: {sex_after[2]:,}  (true 0)"
    )
    print("  ILOSTAT after the second query:")
    for line in ilostat_lines(table2, indent="    "):
        print(line)
    # Dev time only: the four released marginals of the second table refitted from scratch with more mirror-descent
    # steps than mbi's default of 1,000, to show that the rows left in the empty null keys are the fit and not the
    # noise. Post-processing, so it costs nothing.
    refit = estimation.mirror_descent(table2.model.domain, table2.marginals.flatten(), iters=REFIT_STEPS)
    refit_ilostat = [round(float(v)) for v in np.asarray(refit.project(("ILOSTAT",)).values).ravel()]
    refit_sex = [round(float(v)) for v in np.asarray(refit.project(("SEX",)).values).ravel()]
    refit_two_way = np.asarray(refit.project(("AGE", "HWUSUAL")).values).ravel()
    refit_errors = np.abs(refit_two_way - two_way["true"].to_numpy())
    print(f"  the same marginals refitted at {REFIT_STEPS:,} steps (dev time, post-processing):")
    for line in ilostat_lines(table2, indent="    ", counts=refit_ilostat):
        print(line)
    print(
        f"    SEX: 1: {refit_sex[0]:,}  (true 1,828,992)  2: {refit_sex[1]:,}  (true 1,982,940)  null: {refit_sex[2]:,}  (true 0)"
    )
    print(f"    AGE x HWUSUAL largest |refit - true|: {refit_errors.max():,.0f}  (accuracy {two_way_accuracy:.2f})")


if __name__ == "__main__":
    main()
