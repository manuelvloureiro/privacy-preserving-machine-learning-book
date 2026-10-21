"""Release statistics about customers with OpenDP: the workflow of practitioner Part 3, "Release statistics with a library".

    uv run python code/part03-dp-foundations/practitioner_release_statistics.py

Releases, under one budget of (ε = 1, δ = 1e-5) for adding or removing one customer and all their invoice lines
(accounted in zero-concentrated DP with Gaussian noise, then converted by OpenDP):
the number of purchase lines, the revenue and the median line value; lines and revenue per country (a public list
of keys); and the number of customers who bought each product, where the product list is treated as private and
OpenDP's partition selection decides which products appear. It prints every number the page quotes, on labelled
lines, in the order of the page's steps.

Dataset: Online Retail (UCI Machine Learning Repository, dataset 352; Chen, 2015; licence CC BY 4.0),
https://archive.ics.uci.edu/static/public/352/online+retail.zip, loaded by code/practitioner_datasets.py. 541,909
invoice lines of a UK online shop from December 2010 to December 2011. The privacy unit is the customer: lines
without a CustomerID cannot be attributed to anyone and are dropped, which leaves 406,829 lines by 4,372 customers.

What is deterministic and what is not. OpenDP draws its noise from a cryptographic generator that cannot be seeded,
so the released values change on every run; they are printed under "released (one run)" and the test checks them
against the true values within three times their 95% accuracy. Everything else is deterministic and quotable: the
counts, the truncation trade-offs (truncation samples lines with a seeded Polars shuffle), the budget split, the
noise scales, the accuracies, the partition-selection threshold and the true values of the dev-time check.

Library versions: opendp[polars] 0.16.0 with polars 1.36.1.
"""

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import opendp.prelude as dp
import polars as pl
from numpy.typing import NDArray
from opendp.extras.polars import LazyFrameQuery

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

dp.enable_features("contrib")

EPSILON = 1.0
WEIGHTS = [0.25, 0.25, 0.5]  # share of the budget for the scalars, the countries and the products
K_GRID = (10, 25, 50, 100, 200, 500)
K = 100  # purchase lines kept per customer
M_GRID = (1, 5, 10, 20, 50)
M = 50  # products kept per customer in the per-product counts (the best m in the dev-time table)
VALUE_BOUNDS = (0.0, 200.0)  # line value in pounds, from the business (wholesale packs of gift items)
MEDIAN_CANDIDATES = [i / 2 for i in range(401)]  # £0.00, £0.50, ..., £200.00
MAX_LINES = 1_000_000  # a public upper bound on the number of lines, which OpenDP needs for float sums

# The public key list for the per-country table, fixed before looking at the data: the shop's home market, the
# 27 member states of the EU since 2020 and the 4 of EFTA, and five large markets outside Europe, by their ISO 3166
# English short names, plus "Other". (An earlier version used the 38 values that appear in the Country column,
# which is a key list read from the data: "Hong Kong" occurs only on lines without a CustomerID, so its presence
# in the list said something the release should not.)
MARKETS = [
    "United Kingdom",
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France",
    "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
    "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden",
    "Iceland", "Liechtenstein", "Norway", "Switzerland",
    "United States", "Canada", "Australia", "Japan", "China",
    "Other",
]  # fmt: skip
# Schema rules: the source writes some countries in its own spelling. Every other value that is not in MARKETS
# (another country, "Channel Islands", "European Community", "Unspecified") becomes "Other".
SPELLINGS = {"EIRE": "Ireland", "USA": "United States", "Czech Republic": "Czechia"}


def market() -> pl.Expr:
    """The customer's country mapped onto MARKETS, line by line."""
    country = pl.col("Country").replace(SPELLINGS)
    return pl.when(country.is_in(MARKETS)).then(country).otherwise(pl.lit("Other")).alias("Market")


# ---------------------------------------------------------------------------------------------------------------
# Data and the privacy unit
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cleaning:
    lines: int
    no_customer: int
    with_customer: int
    customers: int
    returns: int
    zero_price: int
    purchases: int
    purchase_customers: int


def epsilon_of_rho(rho: float, delta: float) -> float:
    """The ε that OpenDP's conversion gives for ρ-zCDP at δ, read off a Gaussian mechanism with that ρ."""
    gauss = dp.m.make_gaussian(
        dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float), scale=(2 * rho) ** -0.5
    )
    epsilon, _ = dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(gauss), delta).map(1.0)
    return float(epsilon)


def rho_for(epsilon: float, delta: float) -> float:
    """The largest ρ whose conversion at δ stays within ε.

    The target is shaded by one part in a billion, so that composing the split shares back together, which rounds
    in floating point, cannot land a hair above ε (it can reach 1.0000000000000002 otherwise).
    """
    target = epsilon * (1 - 1e-9)
    return float(dp.binary_search(lambda rho: epsilon_of_rho(rho, delta) <= target, bounds=(1e-6, 1.0)))


def delta_for(n: int) -> float:
    """The book's default δ: 1/n^1.1 rounded down to a power of ten."""
    return 10.0 ** math.floor(math.log10(n**-1.1))


def clean(raw: pl.DataFrame) -> tuple[pl.DataFrame, Cleaning]:
    """Keep the purchase lines of identified customers and add the line value in pounds (not yet clipped).

    Every step is a filter on one line at a time, so removing a customer removes only that customer's lines.
    """
    identified = raw.filter(pl.col("CustomerID").is_not_null())
    returns = identified.filter(pl.col("Quantity") <= 0).height
    zero_price = identified.filter(pl.col("Quantity") > 0, pl.col("UnitPrice") <= 0).height
    purchases = identified.filter(pl.col("Quantity") > 0, pl.col("UnitPrice") > 0)
    purchases = purchases.with_columns(market(), value=pl.col("Quantity") * pl.col("UnitPrice"))
    stats = Cleaning(
        lines=raw.height,
        no_customer=raw.height - identified.height,
        with_customer=identified.height,
        customers=identified["CustomerID"].n_unique(),
        returns=returns,
        zero_price=zero_price,
        purchases=purchases.height,
        purchase_customers=purchases["CustomerID"].n_unique(),
    )
    return purchases, stats


def per_customer(lines: pl.DataFrame, column: str | None = None) -> pl.Series:
    """Lines per customer, or distinct values of `column` per customer."""
    agg = pl.len() if column is None else pl.col(column).n_unique()
    return lines.group_by("CustomerID").agg(agg.alias("n"))["n"]


def quantiles(counts: pl.Series, qs: Sequence[float]) -> list[int]:
    return [int(counts.quantile(q, interpolation="lower") or 0) for q in qs]


def truncation_tradeoff(counts: pl.Series, k: int) -> tuple[float, float]:
    """Share of items kept, and share of customers who lose some, when each customer keeps at most k."""
    c = counts.to_numpy()
    return float(np.minimum(c, k).sum() / c.sum()), float((c > k).mean())


# excerpt: relstat-truncate
def sample_per_customer(k: int, seed: int, *by: str) -> pl.Expr:
    """True for at most k randomly chosen lines of each customer (within each value of `by`)."""
    return pl.int_range(pl.len()).shuffle(seed=seed).over("CustomerID", *by) < k


# end excerpt


# ---------------------------------------------------------------------------------------------------------------
# Truncated true values, for the dev-time check
# ---------------------------------------------------------------------------------------------------------------


def clipped_value() -> Any:
    """The line value clipped to VALUE_BOUNDS, as a Polars expression.

    Typed Any because OpenDP adds the `.dp` namespace (`.dp.sum`, `.dp.median`) to Polars expressions at run time,
    which a type checker cannot see.
    """
    low, high = VALUE_BOUNDS
    return pl.col("value").clip(low, high)


def truncate_lines(lines: pl.DataFrame, k: int, seed: int) -> pl.DataFrame:
    """At most k random lines per customer, as the queries keep."""
    return lines.filter(sample_per_customer(k, seed))


def truncate_products(lines: pl.DataFrame, m: int, seed: int) -> pl.DataFrame:
    """One line per customer and product, then at most m of those per customer: m random products each."""
    return lines.filter(sample_per_customer(1, seed, "StockCode")).filter(sample_per_customer(m, seed))


def median_of(values: NDArray[np.float64]) -> float:
    """The smallest candidate at or above the true median, the value the private median aims for."""
    true = float(np.median(values))
    return min(c for c in MEDIAN_CANDIDATES if c >= true)


# ---------------------------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------------------------


def summary_lines(summary: pl.DataFrame, label: str) -> list[str]:
    out: list[str] = []
    for row in summary.iter_rows(named=True):
        accuracy = row["accuracy"]
        acc = "none" if accuracy is None else f"{accuracy:,.1f}"
        extra = f" threshold={row['threshold']:,}" if "threshold" in row and row["threshold"] is not None else ""
        out.append(
            f"{label} {row['column']}: {row['aggregate']}, {row['distribution']}, "
            f"scale={row['scale']:,.2f} 95% accuracy={acc}{extra}"
        )
    return out


def accuracy_of(summary: pl.DataFrame, column: str) -> float:
    return float(summary.filter(pl.col("column") == column)["accuracy"].item())


def product_counts(context: dp.Context, m: int, seed: int) -> LazyFrameQuery:
    """Customers per product, with at most m random products per customer; the keys come from the data."""
    return (
        context.query()
        .filter(sample_per_customer(1, seed, "StockCode"))
        .filter(sample_per_customer(m, seed))
        .group_by("StockCode")
        .agg(dp.len(signed=True).alias("customers"))
    )


def true_product_counts(lines: pl.DataFrame, m: int, seed: int) -> pl.DataFrame:
    return truncate_products(lines, m, seed).group_by("StockCode").agg(pl.len().alias("customers"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="seed of the truncation (OpenDP's noise has none)")
    args = parser.parse_args()
    seed: int = args.seed

    # Step 1: the unit and the budget.
    # excerpt: relstat-unit
    raw = practitioner_datasets.online_retail()
    lines, stats = clean(raw)
    n_customers = stats.customers
    delta = delta_for(n_customers)  # 1/n^1.1, rounded down to a power of ten
    per_line = per_customer(lines)
    print(f"customers: {n_customers:,}  purchase lines: {lines.height:,}")
    print(f"delta: {delta:.0e}")
    p50, p90, p99, top = quantiles(per_line, [0.5, 0.9, 0.99, 1.0])
    print(f"lines per customer: p50={p50} p90={p90} p99={p99} max={top}")
    # end excerpt
    print(f"invoice lines: {stats.lines:,}")
    print(f"lines without a customer id (dropped): {stats.no_customer:,}")
    print(f"lines with a customer id: {stats.with_customer:,} by {stats.customers:,} customers")
    print(f"1/n^1.1 = {n_customers**-1.1:.2e}")
    print(f"epsilon: {EPSILON:g}")
    # Half of δ pays for converting zCDP to (ε, δ), half for the partition-selection threshold.
    rho = rho_for(EPSILON, delta / 2)
    print(f"rho for epsilon={EPSILON:g} at delta/2: {rho:.4f}")

    # Step 2: bound each customer's contribution.
    print(f"returns and cancellations, Quantity <= 0 (dropped): {stats.returns:,}")
    print(f"lines with UnitPrice <= 0 (dropped): {stats.zero_price:,}")
    print(f"purchase lines: {stats.purchases:,} by {stats.purchase_customers:,} customers")
    for k in K_GRID:
        kept, truncated = truncation_tradeoff(per_line, k)
        print(f"k={k}: lines kept {kept:.1%}  customers truncated {truncated:.1%}")
    low, high = VALUE_BOUNDS
    clipped = (lines["value"] > high).mean() or 0.0
    print(f"line value bounds: [{low:g}, {high:g}] pounds  lines clipped: {clipped:.2%}")
    countries = per_customer(lines, "Country")
    print(f"customers with lines in more than one country: {(countries > 1).sum()}")
    per_product = per_customer(lines, "StockCode")
    p50, p90, p99, top = quantiles(per_product, [0.5, 0.9, 0.99, 1.0])
    print(f"products per customer: p50={p50} p90={p90} p99={p99} max={top}")

    # Step 3: the context.
    # excerpt: relstat-context
    context = dp.Context.compositor(
        data=lines.lazy(),
        privacy_unit=dp.unit_of(contributions=1, identifier="CustomerID"),
        privacy_loss=dp.loss_of(rho=rho, delta=delta / 2),  # Gaussian noise; delta/2 for thresholds
        split_by_weights=WEIGHTS,  # scalars, countries, products
        margins=[dp.polars.Margin(by=[], max_length=MAX_LINES)],
    )
    # end excerpt
    for name, (rho_q, delta_q) in zip(("scalars", "countries", "products"), context.d_mids or [], strict=True):
        print(f"budget {name}: rho={rho_q:.4f} delta={delta_q:.2e}")
    total = dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(context.accountant), delta).map(context.d_in)
    print(f"whole release as (epsilon, delta): ({total[0]:.4f}, {total[1]:.0e})")
    assert total[0] <= EPSILON and total[1] <= delta, total

    # Step 4: scalar statistics with error bars. A query takes the context's next share of the budget when it is
    # built and spends it when it is released, so each query is built, summarized and released in turn.
    # excerpt: relstat-scalars
    value = clipped_value()  # the line value, clipped to [low, high]
    scalars = (
        context.query()
        .filter(sample_per_customer(K, seed))  # at most K lines per customer
        .select(
            dp.len(signed=True),
            value.dp.sum((low, high)).alias("revenue"),
            value.dp.median(candidates=MEDIAN_CANDIDATES).alias("median"),
        )
    )
    scalar_summary = scalars.summarize(alpha=0.05)
    scalars_hat = scalars.release().collect()
    # end excerpt
    for line in summary_lines(scalar_summary, "scalars"):
        print(line)

    # Step 5: group by public keys.
    # excerpt: relstat-public-keys
    by_country = (
        context.query()
        .filter(sample_per_customer(K, seed))
        .group_by("Market")
        .agg(dp.len(signed=True), value.dp.sum((low, high)).alias("revenue"))
        .with_keys(pl.LazyFrame({"Market": MARKETS}))  # every key is released, zeros included
    )
    country_summary = by_country.summarize(alpha=0.05)
    countries_hat = by_country.release().collect()
    # end excerpt
    for line in summary_lines(country_summary, "per country"):
        print(line)

    # Step 6: group by unknown keys, with partition selection. OpenDP 0.16's truncate_num_groups is not used:
    # keep="sample" passes summarize() but fails at collect(), after release() has already spent the query's share,
    # and keep="first" keeps k - 1 groups, the first in sort order, which biases the counts toward low StockCodes.
    # One line per customer and product, then at most M lines per customer, keeps M random products instead.
    # First the dev-time choice of m, on this public set.
    product_queries = {m: product_counts(context, m, seed) for m in M_GRID}
    for m, query in product_queries.items():
        kept, truncated = truncation_tradeoff(per_product, m)
        summary = query.summarize(alpha=0.05).row(0, named=True)
        truth = true_product_counts(lines, m, seed)
        above = truth.filter(pl.col("customers") >= summary["threshold"]).height
        print(
            f"m={m}: pairs kept {kept:.1%}  customers truncated {truncated:.1%}  "
            f"scale={summary['scale']:.2f}  threshold={summary['threshold']}  products above it: {above}"
        )
    # excerpt: relstat-partition-selection
    by_product = (
        context.query()
        .filter(sample_per_customer(1, seed, "StockCode"))  # one line per customer and product
        .filter(sample_per_customer(M, seed))  # at most M products per customer
        .group_by("StockCode")
        .agg(dp.len(signed=True).alias("customers"))
    )
    product_summary = by_product.summarize(alpha=0.05)
    products_hat = by_product.release().collect()
    # end excerpt
    for line in summary_lines(product_summary, "per product"):
        print(line)
    threshold = int(product_summary["threshold"].item())

    # Step 7: the dev-time check against the true values (this public dataset only).
    truncated = truncate_lines(lines, K, seed)
    true_lines = truncated.height
    kept_values = truncated["value"].clip(low, high).to_numpy()
    true_revenue = float(kept_values.sum())
    true_median = median_of(kept_values)
    full_revenue = float(lines["value"].to_numpy().sum())
    full_median = float(np.median(lines["value"].to_numpy()))
    print(f"true lines after truncation: {true_lines:,} (all purchase lines: {lines.height:,})")
    print(
        f"true revenue after truncation and clipping: {true_revenue:,.0f} (all lines, unclipped: {full_revenue:,.0f})"
    )
    print(f"truncation and clipping keep {true_revenue / full_revenue:.1%} of the revenue")
    print(f"true median line value after truncation: {true_median:.2f} (all lines: {full_median:.2f})")
    true_country = truncated.group_by("Market").agg(pl.len().alias("len"), clipped_value().sum().alias("revenue"))
    true_country = pl.DataFrame({"Market": MARKETS}).join(true_country, on="Market", how="left").fill_null(0)
    top_countries = true_country.sort("len", descending=True).head(5)
    for row in top_countries.iter_rows(named=True):
        print(f"true {row['Market']}: lines {row['len']:,}  revenue {row['revenue']:,.0f}")
    print(f"keys with no purchase lines: {true_country.filter(pl.col('len') == 0).height} of {len(MARKETS)}")
    other = lines.filter(pl.col("Market") == "Other")
    print(f"lines mapped to Other: {other.height:,} from {other['Country'].n_unique()} source values")
    products = true_product_counts(lines, M, seed)
    product_accuracy = accuracy_of(product_summary, "customers")
    print(f"products with any purchase: {lines['StockCode'].n_unique():,}")
    print(f"products with true count >= threshold: {products.filter(pl.col('customers') >= threshold).height}")
    for factor in (-3, 3):
        cut = threshold + factor * product_accuracy
        above = products.filter(pl.col("customers") >= cut).height
        print(f"products with true count >= threshold{factor:+d}x accuracy: {above}")

    # The release itself: random on every run.
    print("released (one run):")
    lines_hat, revenue_hat, median_hat = scalars_hat.row(0)
    print(f"  lines {lines_hat:,}  revenue {revenue_hat:,.0f}  median {median_hat:.2f}")
    for name in top_countries["Market"]:
        row = countries_hat.filter(pl.col("Market") == name).row(0, named=True)
        print(f"  {name}: lines {row['len']:,}  revenue {row['revenue']:,.0f}")
    country_accuracy = accuracy_of(country_summary, "len")
    compared = countries_hat.join(true_country, on="Market", suffix="_true")
    inside = compared.filter((pl.col("len") - pl.col("len_true")).abs() <= country_accuracy).height
    print(f"  country line counts within their 95% accuracy: {inside} of {compared.height}")
    print(f"  products released: {products_hat.height:,}")
    joined = products_hat.join(products, on="StockCode", suffix="_true").sort("customers", descending=True)
    for row in joined.head(5).iter_rows(named=True):
        print(f"  product {row['StockCode']}: customers {row['customers']}  true {row['customers_true']}")

    # Step 8: the guarantee, written from the variables above.
    # excerpt: relstat-statement
    print(
        f"This release satisfies (epsilon = {EPSILON:g}, delta = {delta:.0e})-differential privacy "
        f"for adding or removing one customer and all their invoice lines, "
        f"with at most {K} lines per customer (sampled at random), at most {M} random products per customer "
        f"in the product counts, and line values clipped to [{low:g}, {high:g}] pounds, "
        f"under the central model, using OpenDP {dp.__version__} with Gaussian noise "
        f"(accounted as {rho:.4f}-zCDP plus {delta / 2:.0e} for thresholds, converted to epsilon and delta). "
        f"The market list was fixed in advance; the product list is not public."
    )
    # end excerpt


if __name__ == "__main__":
    main()
