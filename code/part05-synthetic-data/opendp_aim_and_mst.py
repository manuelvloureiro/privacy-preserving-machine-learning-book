"""Synthetic data with AIM and MST: the chapter "Synthetic data with AIM and MST" of the OpenDP book (Part 5).

    uv run python code/part05-synthetic-data/opendp_aim_and_mst.py

Follows the documentation's Synthetic Data page and the docstrings of dp.mbi.AIM, dp.mbi.MST, dp.mbi.Sequential and
ContingencyTable.synthesize. One context on the Labour Force Survey sample (privacy unit: 36 contributions, budget
rho = 0.3 with delta = 3e-7) releases three contingency tables over SEX, AGE, HWUSUAL and ILOSTAT, each at
rho = 0.1 and delta = 1e-7: one with AIM in a bounded number of rounds, one with MST, and one with Sequential, which
runs a Fixed workload before AIM. The AIM table is turned into synthetic rows with synthesize(), and the one-way and
the measured multi-way marginals of every release are compared with the real data's. It prints every number the
page quotes, on labelled lines, in the order of the page's sections.

Dataset: the France LFS sample of the OpenDP tutorials (3,811,932 rows, 1,682,306 people), loaded by
code/practitioner_datasets.py. The data are public, so the true marginals are computed and compared with the
releases, which a real release could not do.

What is deterministic and what is not. OpenDP draws its noise from a cryptographic generator that cannot be seeded,
so everything a release contains changes on every run: the marginals AIM and MST choose, the noisy counts, the
synthetic rows, the run times and the distances to the real data. These are printed under "one run" labels and the
test checks them structurally (a spanning tree, the guaranteed marginals present) and against bounds. The noise
scales, the threshold, the budget arithmetic, the conversion of rho to epsilon, the facts about the data and the
sizes of the synthetic frames are deterministic. NumPy's global generator, which synthesize() uses for its rounding
and for sampling values inside cut bins, is seeded, so the synthetic frame is a function of the released table.

Versions: opendp[polars] 0.16.0 with polars 1.36.1 and mbi 1.1.0, the Private-PGM version that opendp 0.16.0 pins.
The mbi package prints advice about JAX when it is imported; the script silences those warnings and runs JAX on the
CPU.
"""

import argparse
import math
import os
import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import opendp.prelude as dp
import polars as pl
from opendp.extras.mbi import Algorithm, ContingencyTable
from opendp.extras.polars.contingency_table import ContingencyTableQuery

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # mbi fits its models with JAX; there is no GPU to look for
warnings.filterwarnings("ignore", category=UserWarning, module="mbi")  # mbi's advice about JAX settings at import

dp.enable_features("contrib")

CONTRIBUTIONS = 36  # a person answers at most once per quarter over nine years
RHO = 0.3  # the context's budget: three releases of rho = 0.1 each
DELTA = 3e-7  # three times the slack of one release, which pays for releasing ILOSTAT's keys
RHO_QUERY = 0.1
DELTA_QUERY = 1e-7
EPSILON_DELTA = 1e-7  # the further slack spent on converting rho to epsilon in the statement
COLUMNS = ("SEX", "AGE", "HWUSUAL", "ILOSTAT")
KEYS: dict[str, list[int]] = {"SEX": [1, 2]}  # public: the survey codes men 1 and women 2
CUTS: dict[str, list[int]] = {"AGE": [20, 40, 60], "HWUSUAL": [1, 20, 40]}  # public: left-open bins
ROUNDS = 10  # AIM's default would be 16 rounds per column, 64 here
GUARANTEED = [("SEX", "ILOSTAT"), ("AGE", "ILOSTAT")]  # the marginals Sequential measures first, with Fixed
FILL: dict[str, float] = {"AGE": 0.0, "HWUSUAL": 99.0}  # public codes for a null: the youngest bin; "not applicable"
TRUNCATE = 4  # rows kept per person under the identifier unit
SAMPLE_ROWS = 100_000  # the smaller synthetic frame drawn with method="sample"


def epsilon_of_rho(rho: float, delta: float) -> float:
    """The epsilon that OpenDP's conversion gives for rho-zCDP at delta, read off a Gaussian measurement."""
    space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
    gaussian = dp.m.make_gaussian(*space, scale=1 / math.sqrt(2 * rho))  # sensitivity 1 at this scale is rho-zCDP
    approx = dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(gaussian), delta)
    epsilon, _ = approx.map(1.0)
    return float(epsilon)


def known(value: float | None) -> float:
    """A one-way scale the query reports; it is None only when every column has keys and nothing is released."""
    if value is None:
        raise ValueError("the query releases no one-way marginals")
    return value


def gaussian_scale(sensitivity: float, rho: float) -> float:
    """The scale of Gaussian noise that spends rho on a query of this L2 sensitivity: sensitivity over sqrt(2 rho)."""
    return sensitivity / math.sqrt(2 * rho)


# ---------------------------------------------------------------------------------------------------------------
# The context and the three releases
# ---------------------------------------------------------------------------------------------------------------


# excerpt: synth-context
def make_context(frame: pl.LazyFrame) -> dp.Context:
    """The context: a privacy unit of 36 rows, a budget of rho = 0.3 with delta = 3e-7, one share per release."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),  # neighbours add or remove one person's 36 rows
        privacy_loss=dp.loss_of(rho=RHO, delta=DELTA),  # zCDP with Gaussian noise; delta pays for ILOSTAT's keys
    )


# end excerpt


def fill_nulls() -> list[pl.Expr]:
    """Each null of AGE and HWUSUAL becomes a public code, so that every row lands in one of the cut bins."""
    return [pl.col(column).fill_null(value) for column, value in FILL.items()]


# excerpt: synth-query
def table_query(context: dp.Context, algorithm: Algorithm, fill: bool = False) -> ContingencyTableQuery:
    """The documentation's contingency-table query at rho = 0.1 and delta = 1e-7, with the algorithm plugged in."""
    query = context.query(rho=RHO_QUERY, delta=DELTA_QUERY)
    if fill:
        query = query.with_columns(fill_nulls())  # a row-by-row transformation before the table
    return query.select(*COLUMNS).contingency_table(
        keys=KEYS,  # public categories of SEX; ILOSTAT's are released with the threshold
        cuts=CUTS,  # bins for the two numeric columns
        algorithm=algorithm,
    )


# end excerpt


def measured(table: ContingencyTable) -> list[tuple[tuple[str, ...], float]]:
    """The multi-way marginals a release measured, with the noise standard deviation of each, in release order."""
    out: list[tuple[tuple[str, ...], float]] = []
    for measurement in table.marginals.flatten():
        clique: tuple[str, ...] = tuple(measurement.clique)
        if len(clique) > 1:
            out.append((clique, float(measurement.stddev)))
    return out


def timed_release(query: ContingencyTableQuery) -> tuple[ContingencyTable, float]:
    """Release the table and return it with the seconds the release took."""
    start = time.perf_counter()
    table = query.release()
    return table, time.perf_counter() - start


def spanning_tree(pairs: Sequence[tuple[str, ...]], columns: Sequence[str]) -> bool:
    """Whether the pairs connect every column with no cycle, which is what MST's selection guarantees."""
    if len(pairs) != len(columns) - 1:
        return False
    parent = {column: column for column in columns}

    def root(column: str) -> str:
        while parent[column] != column:
            column = parent[column]
        return column

    for left, right in pairs:
        if root(left) == root(right):
            return False
        parent[root(left)] = root(right)
    return True


# ---------------------------------------------------------------------------------------------------------------
# Synthetic rows and the comparison with the real data
# ---------------------------------------------------------------------------------------------------------------


def synthesize(table: ContingencyTable, seed: int, rows: int | None = None, method: str = "round") -> pl.DataFrame:
    """The table's synthetic rows; NumPy's global generator, which the rounding and the bin sampling use, is seeded."""
    np.random.seed(seed)
    if method == "sample":
        return table.synthesize(rows=rows, method="sample")
    return table.synthesize(rows=rows, method="round")


def bin_labels(cuts: Sequence[float]) -> list[str]:
    """The labels the library gives the left-open bins of a cut column: (-inf, 20.0], (20.0, 40.0], ..."""
    edges = [float(cut) for cut in cuts]
    return [f"({low}, {high}]" for low, high in zip(["-inf", *edges], [*edges, "inf"], strict=True)]


def binned(frame: pl.LazyFrame, keys: dict[str, list[Any]], fill: bool) -> pl.LazyFrame:
    """The real rows under the table's keys and cuts: cut columns become bin labels and a value outside a column's
    keys becomes null, as the library does before it counts. A null in a cut column goes to the last bin, which is
    where the library's multi-way step puts it."""
    if fill:
        frame = frame.with_columns(fill_nulls())
    exprs: list[pl.Expr] = []
    for column in COLUMNS:
        if column in CUTS:
            labels = bin_labels(CUTS[column])
            cut = pl.col(column).cut([float(c) for c in CUTS[column]], labels=labels).cast(pl.String)
            exprs.append(cut.fill_null(labels[-1]).alias(column))
        else:
            known = [key for key in keys[column] if key is not None]
            exprs.append(pl.when(pl.col(column).is_in(known)).then(pl.col(column)).otherwise(None).alias(column))
    return frame.select(exprs)


def marginal(frame: pl.DataFrame, by: Sequence[str]) -> pl.DataFrame:
    """The counts of every combination of the columns `by`."""
    return frame.group_by(list(by)).len()


def distance(real: pl.DataFrame, synthetic: pl.DataFrame, by: Sequence[str]) -> tuple[float, int]:
    """The total variation distance between two marginals, as distributions, and the largest cell error in rows."""
    joined = real.join(synthetic, on=list(by), how="full", coalesce=True, nulls_equal=True, suffix="_synth")
    real_counts = joined["len"].fill_null(0).cast(pl.Int64).to_numpy()
    synth_counts = joined["len_synth"].fill_null(0).cast(pl.Int64).to_numpy()
    p = real_counts / real_counts.sum()
    q = synth_counts / synth_counts.sum()
    return 0.5 * float(np.abs(p - q).sum()), int(np.abs(real_counts - synth_counts).max())


def tv_bound(cells: int, std: float, rows: int) -> float:
    """A bound on the total variation distance when each of `cells` cells is within three standard deviations."""
    return 3 * cells * std / (2 * rows)


def cells(table: ContingencyTable, clique: Sequence[str]) -> int:
    return int(np.prod([len(table.keys[column]) for column in clique]))


def compare(
    name: str,
    kind: str,
    table: ContingencyTable,
    real: pl.DataFrame,
    synthetic: pl.DataFrame,
    cliques: Sequence[tuple[str, ...]],
) -> list[str]:
    """One line per clique: the distance between the synthetic marginal and the real one, both binned, the table's
    own standard deviation for that marginal, and the bound the distance should stay under."""
    lines: list[str] = []
    for clique in cliques:
        tv, worst = distance(marginal(real, clique), marginal(synthetic, clique), clique)
        std = table.std(clique)
        bound = tv_bound(cells(table, clique), std, real.height)
        label = clique[0] if len(clique) == 1 else "(" + ", ".join(clique) + ")"
        lines.append(
            f"{kind} {label} ({name}): tv={tv:.5f} largest cell error={worst:,} rows  std={std:.1f} bound={bound:.5f}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="seed of synthesize()'s rounding and bin sampling")
    args = parser.parse_args()
    seed: int = args.seed

    print(f"opendp version: {dp.__version__}")
    frame = pl.scan_parquet(practitioner_datasets.france_lfs())
    facts = frame.select(
        pl.len().alias("rows"),
        pl.col("AGE").is_null().sum().alias("age_nulls"),
        pl.col("HWUSUAL").is_null().sum().alias("hw_nulls"),
        (pl.col("HWUSUAL") == 99).sum().alias("hw_99"),
    ).collect()
    n_rows = int(facts["rows"].item())
    print(f"rows: {n_rows:,}  AGE nulls: {facts['age_nulls'].item():,}  HWUSUAL nulls: {facts['hw_nulls'].item():,}")
    print(f"HWUSUAL = 99 (not applicable): {facts['hw_99'].item():,}")

    # Section 3: the AIM release, as the documentation writes it.
    context = make_context(frame)
    print(f"privacy unit: {dp.unit_of(contributions=CONTRIBUTIONS)!r}")
    print(f"privacy loss: {dp.loss_of(rho=RHO, delta=DELTA)!r}")
    print(f"query budget: rho={RHO_QUERY:g} delta={DELTA_QUERY:g}")
    # excerpt: synth-aim
    aim_query = table_query(context, dp.mbi.AIM(rounds=ROUNDS))
    oneway_scale, oneway_threshold = aim_query.oneway_scale, aim_query.oneway_threshold
    table_aim, seconds_aim = timed_release(aim_query)
    # end excerpt
    print(f"aim: oneway scale={oneway_scale:.4f} threshold={oneway_threshold}")
    rho_oneway = RHO_QUERY / 2 / len(COLUMNS)  # half the share for one-way marginals, split over the four columns
    rho_round = RHO_QUERY / 2 / ROUNDS  # the other half, split over the rounds
    measure_split = dp.mbi.AIM().measure_split
    print(
        f"aim: rho per one-way marginal={rho_oneway:.4f}  scale {CONTRIBUTIONS}/sqrt(2 rho)={gaussian_scale(CONTRIBUTIONS, rho_oneway):.4f}"
    )
    print(
        f"aim: rounds={ROUNDS}  rho per round={rho_round:.4f}  measure={rho_round * measure_split:.4f}  "
        f"select={rho_round * (1 - measure_split):.4f}  scale of a round's marginal={gaussian_scale(CONTRIBUTIONS, rho_round * measure_split):.4f}"
    )
    print(f"aim: rounds by default={16 * len(COLUMNS)}  max_size={dp.mbi.AIM().max_size:g} MB")
    print(f"aim keys ILOSTAT: {table_aim.keys['ILOSTAT'].to_list()}  thresholds: {table_aim.thresholds}")
    print(f"aim release (one run): {seconds_aim:.1f} s")
    print(f"aim model total (one run): {float(table_aim.model.total):,.1f}  real rows {n_rows:,}")
    for clique, std in measured(table_aim):
        print(f"aim measured: {clique} std={std:.2f}")

    # Section 4: MST.
    # excerpt: synth-mst
    mst_query = table_query(context, dp.mbi.MST())
    table_mst, seconds_mst = timed_release(mst_query)
    # end excerpt
    rho_pairs = RHO_QUERY / 2 * dp.mbi.MST().measure_split / (len(COLUMNS) - 1)
    print(f"mst: oneway scale={mst_query.oneway_scale:.4f} threshold={mst_query.oneway_threshold}")
    print(
        f"mst: pairs={len(COLUMNS) - 1}  rho per pair={rho_pairs:.4f}  scale of a pair={gaussian_scale(CONTRIBUTIONS, rho_pairs):.4f}"
    )
    print(f"mst release (one run): {seconds_mst:.1f} s")
    print(f"mst model total (one run): {float(table_mst.model.total):,.1f}  real rows {n_rows:,}")
    pairs_mst = [clique for clique, _ in measured(table_mst)]
    for clique, std in measured(table_mst):
        print(f"mst measured: {clique} std={std:.2f}")
    print(f"mst pairs form a spanning tree: {spanning_tree(pairs_mst, COLUMNS)}")

    # Section 5: the one-way marginals of the two releases against the real data, which reveals the nulls.
    start = time.perf_counter()
    synthetic_aim = synthesize(table_aim, seed)
    seconds_synth = time.perf_counter() - start
    synthetic_mst = synthesize(table_mst, seed)
    real_raw = binned(frame, table_aim.keys, fill=False).collect()
    binned_aim = binned(synthetic_aim.lazy(), table_aim.keys, fill=False).collect()  # the numbers back in their bins
    binned_mst = binned(synthetic_mst.lazy(), table_mst.keys, fill=False).collect()
    oneway = [(column,) for column in COLUMNS]
    for line in compare("aim", "one-way", table_aim, real_raw, binned_aim, oneway):
        print(line)
    for line in compare("mst", "one-way", table_mst, real_raw, binned_mst, oneway):
        print(line)
    projected = table_aim.project_melted(("HWUSUAL",))
    real_hw = marginal(real_raw, ("HWUSUAL",)).sort("HWUSUAL")
    print("HWUSUAL bins (aim, one run): fitted count against the real count with nulls in the last bin")
    for row in projected.sort("HWUSUAL").iter_rows(named=True):
        true = int(real_hw.filter(pl.col("HWUSUAL") == row["HWUSUAL"])["len"].item())
        print(f"  {row['HWUSUAL']}: fitted {row['len']:,.0f}  real {true:,}  difference {row['len'] - true:+,.0f}")

    # Section 6: Sequential, with the nulls filled.
    # excerpt: synth-sequential
    sequential = dp.mbi.Sequential(
        algorithms=[
            dp.mbi.Fixed(queries=[dp.mbi.Count(by) for by in GUARANTEED]),  # these two marginals, whatever happens
            dp.mbi.AIM(rounds=ROUNDS),  # then AIM chooses the rest
        ],
    )
    sequential_query = table_query(context, sequential, fill=True)
    table_seq, seconds_seq = timed_release(sequential_query)
    # end excerpt
    rho_fixed = RHO_QUERY / 2 / 2 / len(GUARANTEED)  # half the share to the one-way step, half of the rest to Fixed
    print(
        f"sequential: rho per guaranteed marginal={rho_fixed:.4f}  scale={gaussian_scale(CONTRIBUTIONS, rho_fixed):.4f}"
    )
    rho_round_seq = RHO_QUERY / 2 / 2 / ROUNDS
    print(
        f"sequential: rho per AIM round={rho_round_seq:.4f}  scale of a round's marginal={gaussian_scale(CONTRIBUTIONS, rho_round_seq * measure_split):.4f}"
    )
    print(f"sequential release (one run): {seconds_seq:.1f} s")
    print(f"sequential model total (one run): {float(table_seq.model.total):,.1f}  real rows {n_rows:,}")
    cliques_seq = [clique for clique, _ in measured(table_seq)]
    for clique, std in measured(table_seq):
        print(f"sequential measured: {clique} std={std:.2f}")
    print(f"sequential guaranteed marginals present: {all(tuple(by) in cliques_seq for by in GUARANTEED)}")
    synthetic_seq = synthesize(table_seq, seed)
    real_filled = binned(frame, table_seq.keys, fill=True).collect()
    binned_seq = binned(synthetic_seq.lazy(), table_seq.keys, fill=False).collect()
    for line in compare("sequential", "one-way", table_seq, real_filled, binned_seq, oneway):
        print(line)

    # Section 7: the synthetic rows.
    print(
        f"synthesize (aim, one run): shape {synthetic_aim.shape}  model total {float(table_aim.model.total):,.1f}  "
        f"{seconds_synth:.1f} s"
    )
    with pl.Config(tbl_rows=5, tbl_hide_dataframe_shape=True):
        print(synthetic_aim.head(5))
    ranges = synthetic_aim.select(
        pl.col("AGE").min().alias("age_min"),
        pl.col("AGE").max().alias("age_max"),
        pl.col("HWUSUAL").min().alias("hw_min"),
        pl.col("HWUSUAL").max().alias("hw_max"),
    ).row(0)
    print(f"synthetic AGE from {ranges[0]:.2f} to {ranges[1]:.2f}  HWUSUAL from {ranges[2]:.2f} to {ranges[3]:.2f}")
    print(f"synthetic dtypes: {dict(synthetic_aim.schema)}")
    values = {
        column: sorted(synthetic_aim[column].unique().to_list(), key=lambda v: (v is None, v))
        for column in ("SEX", "ILOSTAT")
    }
    print(f"synthetic SEX values: {values['SEX']}  ILOSTAT values: {values['ILOSTAT']}")
    # excerpt: synth-sample
    sample = table_aim.synthesize(rows=SAMPLE_ROWS, method="sample")  # draws rows independently from the model
    # end excerpt
    print(f"synthesize rows={SAMPLE_ROWS:,} method=sample: shape {sample.shape}")
    print(f"budget spent: {context.current_privacy_loss()}")

    # Section 8: the privacy statement.
    epsilon = epsilon_of_rho(RHO, EPSILON_DELTA)
    total_delta = DELTA + EPSILON_DELTA
    whole = dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(context.accountant), total_delta).map(context.d_in)
    print(f"rho={RHO:g} at delta={EPSILON_DELTA:g}: epsilon={epsilon:.4f}")
    print(f"whole context converted: epsilon={float(whole[0]):.4f} delta={float(whole[1]):g}")
    # excerpt: synth-statement
    print(
        f"This release of three contingency tables and the synthetic rows drawn from them satisfies "
        f"(rho = {RHO:g}, delta = {DELTA:g})-zero-concentrated differential privacy, which is "
        f"(epsilon = {epsilon:.2f}, delta = {total_delta:g})-differential privacy, "
        f"for adding or removing all of one person's survey responses, at most {CONTRIBUTIONS} of them, "
        f"under the central model, using OpenDP {dp.__version__} with Gaussian noise. "
        f"The keys of SEX and the cuts of AGE and HWUSUAL were fixed in advance; the keys of ILOSTAT were released "
        f"with a threshold of {oneway_threshold}. Generating the rows is post-processing and spent nothing."
    )
    # end excerpt

    # The identifier unit: the same query under user-level privacy with truncation, built but not released.
    # excerpt: synth-identifier
    context_id = dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=1, identifier="PIDENT"),  # neighbours add or remove one person
        privacy_loss=dp.loss_of(rho=RHO_QUERY, delta=DELTA_QUERY),
    )
    query_id = (
        context_id.query(rho=RHO_QUERY, delta=DELTA_QUERY)
        .truncate_per_group(TRUNCATE)  # at most 4 rows per person, so a person moves a count by at most 4
        .select(*COLUMNS)
        .contingency_table(keys=KEYS, cuts=CUTS, algorithm=dp.mbi.AIM(rounds=ROUNDS))
    )
    # end excerpt
    print(
        f"identifier unit, {TRUNCATE} rows per person: oneway scale={query_id.oneway_scale:.4f} threshold={query_id.oneway_threshold}"
    )
    print(
        f"identifier unit: scale ratio {known(oneway_scale) / known(query_id.oneway_scale):.2f} = {CONTRIBUTIONS}/{TRUNCATE}"
    )

    # Section 9: the measured multi-way marginals against the real data.
    for name, table, synthetic, real in (
        ("aim", table_aim, binned_aim, real_raw),
        ("mst", table_mst, binned_mst, real_raw),
        ("sequential", table_seq, binned_seq, real_filled),
    ):
        cliques = sorted({clique for clique, _ in measured(table)}, key=lambda c: (len(c), c))
        for line in compare(name, "measured", table, real, synthetic, cliques):
            print(line)


if __name__ == "__main__":
    main()
