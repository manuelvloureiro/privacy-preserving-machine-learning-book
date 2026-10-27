# Release statistics with a library

## Why this matters

Suppose you are the analytics engineer of an online shop. The business wants a dashboard of purchase lines, revenue and the typical sale, overall and per country, and of how many customers bought each product. The figures describe customers, so they should come with a differential privacy (DP) guarantee. The difficulty is that one customer appears in many rows, and a guarantee for one row would protect a single purchase, not the person behind a hundred. This chapter does the whole job with OpenDP, from the unit and the budget to the written guarantee.

The data are the Online Retail set from the University of California, Irvine (UCI) Machine Learning Repository, with 541,909 invoice lines of a British online shop. Each line records one product on one invoice, with its quantity, unit price, customer and country. The script [practitioner_release_statistics.py](../../code/part03-dp-foundations/practitioner_release_statistics.py) runs every step in under a minute, with OpenDP 0.16.0 and Polars 1.36.1.

```bash
uv run python code/part03-dp-foundations/practitioner_release_statistics.py
```

OpenDP draws its noise from a cryptographic generator, which cannot be seeded, so the released values change on every run. This page therefore quotes scales, error bars and thresholds, which involve no noise, and shows released values only in a block marked as one run.

## Choose the library {#sec-relstat-library}

The main question is whether a library can bound the contribution of a person who owns many rows (@tbl-relstat-libraries).

| Library | Runs on | Person with many rows | Groups not known in advance | Use it for |
| --- | --- | --- | --- | --- |
| OpenDP | Polars, one machine | Yes, with an identifier column | Yes, by a noisy threshold | Statistics on one machine; the default here |
| diffprivlib | NumPy, scikit-learn | No, one row per person | No | Quick experiments and [simple models](../part05-dp-ml/01-dp-erm.md) |
| PipelineDP | Apache Beam, Spark | Yes, built in | Yes, built in | Pipelines that already run on Beam or Spark |
| Tumult Analytics | Spark | Yes, with an identifier column | Yes, in a separate query | Large releases on Spark |

: Libraries for releasing statistics, and what each can express. {#tbl-relstat-libraries}

PipelineDP and Tumult were not run here, so their rows follow their documentation. Whatever you pick, write no mechanism yourself, and keep [one accountant](02-properties.md#sec-dpprop-budget), which totals the budget spent, for the whole release.

## Decide the unit and the budget {#sec-relstat-unit}

The **unit of privacy** is what the guarantee protects, and for this dashboard it is a customer. The neighbouring datasets therefore differ by adding or removing one customer with all their lines ([The definition of differential privacy](01-definition.md#sec-dpdef-neighbours)). The script drops lines with no customer identifier, which belong to nobody, as well as returns and lines with a zero price. Each filter judges a line by that line alone, so adding or removing a customer still changes only that customer's lines.

```{.python excerpt="relstat-unit"}
raw = practitioner_datasets.online_retail()
lines, stats = clean(raw)
n_customers = stats.customers
delta = delta_for(n_customers)  # 1/n^1.1, rounded down to a power of ten
per_line = per_customer(lines)
print(f"customers: {n_customers:,}  purchase lines: {lines.height:,}")
print(f"delta: {delta:.0e}")
p50, p90, p99, top = quantiles(per_line, [0.5, 0.9, 0.99, 1.0])
print(f"lines per customer: p50={p50} p90={p90} p99={p99} max={top}")
```

```text
customers: 4,372  purchase lines: 397,884
delta: 1e-05
lines per customer: p50=41 p90=208 p99=662 max=7847
```

The slack $\delta$ follows the book's default for $n$ people, $\delta \le 1/n^{1.1}$ rounded down to a power of ten, which gives $10^{-5}$. The 4,372 customers are counted before returns are dropped, which leaves 4,338, but both counts give the same slack. OpenDP logs a general warning that $\delta$ should be at most $10^{-6}$, while this book ties $\delta$ to the number of people. The privacy level is $\varepsilon = 1$ for the whole release, because all its tables describe the same customers and their costs add up ([Post-processing, group privacy and composition](02-properties.md#sec-dpprop-comp)).

The median customer has 41 lines, but under a guarantee per line the largest customer would be a group of 7,847 records. Group privacy multiplies $\varepsilon$ by the size of the group. Printing that maximum is fine only because this set is public, since a bound read off private data leaks.

## Bound each customer's contribution {#sec-relstat-bound}

Write $k$ for the largest number of lines a customer may contribute, so that every query keeps at most $k$ lines of each customer. A small $k$ means little noise but throws data away, while a large $k$ keeps the data but needs more noise, as the script measures on this public set.

```text
k=10: lines kept 10.2%  customers truncated 84.8%
k=25: lines kept 22.4%  customers truncated 64.6%
k=50: lines kept 36.9%  customers truncated 44.0%
k=100: lines kept 55.1%  customers truncated 24.9%
k=200: lines kept 72.7%  customers truncated 10.4%
k=500: lines kept 88.2%  customers truncated 1.8%
```

The default is a high percentile of lines per customer, read from public or historical data, never from the data you release. With neither, spend a small share of the budget on a private high quantile of the lines per customer, over candidates fixed in advance ([Selection](05-selection.md#sec-sel-apps)). This chapter takes $k = 100$, below the 90th percentile of 208 because the noise grows with $k$. It keeps 55.1% of the lines and truncates a quarter of the customers.

Keeping the first $k$ lines by date would bias every statistic towards each customer's early purchases, so the script keeps a random sample of $k$ lines per customer. The filter below does what OpenDP's `truncate_per_group(k, keep="sample")` does, but with a seed, so that the true values of the check stay fixed.

```{.python excerpt="relstat-truncate"}
def sample_per_customer(k: int, seed: int, *by: str) -> pl.Expr:
    """True for at most k randomly chosen lines of each customer (within each value of `by`)."""
    return pl.int_range(pl.len()).shuffle(seed=seed).over("CustomerID", *by) < k
```

Values need bounds as well. A line's value is its quantity times its unit price, which the script clips to between £0 and £200. The lower end holds because returns were dropped, and the upper end is a bound the business sets, which clips 1.02% of the lines. So one customer can move the revenue by at most 100 lines of £200 each. Take value bounds from the domain, never from the data's minimum or maximum ([Beyond global sensitivity](07-sensitivity.md#sec-sens-clip)).

A group-by needs two more bounds: $L_0$, the number of groups one customer may touch, and $L_\infty$, the lines a customer may put in any one group. Countries need neither, because however a customer spreads their $k$ lines over countries, the country counts change by at most $k$ in all, which the noise already covers. When customers spread over many groups, as with country by month, bounding $L_0$ and $L_\infty$ can beat a total $k$. The noise then scales with $L_0L_\infty$ for Laplace and $\sqrt{L_0}\,L_\infty$ for Gaussian. Products need both bounds for partition selection (@sec-relstat-partitions).

## Set up the context {#sec-relstat-context}

OpenDP keeps the whole release in a **context**, which holds the data, the unit, the total budget and its split between queries. Contexts need `dp.enable_features("contrib")`, which switches on components not yet formally vetted, so a reviewer who needs vetted code must check each call in OpenDP's documentation.

```{.python excerpt="relstat-context"}
context = dp.Context.compositor(
    data=lines.lazy(),
    privacy_unit=dp.unit_of(contributions=1, identifier="CustomerID"),
    privacy_loss=dp.loss_of(rho=rho, delta=delta / 2),  # Gaussian noise; delta/2 for thresholds
    split_by_weights=WEIGHTS,  # scalars, countries, products
    margins=[dp.polars.Margin(by=[], max_length=MAX_LINES)],
)
```

The identifier column makes the unit a customer, so `contributions=1` means adding or removing one customer with all their lines, which the truncation filters then bound. For replace, OpenDP takes `changes=` instead. The budget is in **zero-concentrated differential privacy (zCDP)** ([Concentrated differential privacy](../part04-privacy-accounting/03-concentrated-dp.md)). It states privacy as one number, written $\rho$, that adds up across queries and converts to $(\varepsilon, \delta)$ at the end.

With a zCDP budget, the context adds Gaussian noise to every count and sum. That suits the product counts, where one customer touches many groups ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md#sec-lg-choose)). The script's `rho_for` finds $\rho = 0.0284$, the largest value that OpenDP's own conversion turns into $\varepsilon \le 1$ with half of $\delta$. The other half pays for the threshold of @sec-relstat-partitions. Finally, the margin states a public upper bound on the number of lines, which OpenDP needs before it sums floating-point values.

The weights give a quarter of the budget to the overall statistics, a quarter to the countries and half to the products, which need it most. The script prints the split and converts the whole release back to the $(\varepsilon, \delta)$ the statement will claim.

```text
budget scalars: rho=0.0071 delta=1.25e-06
budget countries: rho=0.0071 delta=1.25e-06
budget products: rho=0.0142 delta=2.50e-06
whole release as (epsilon, delta): (1.0000, 1e-05)
```

A query takes its share of the budget when it is built, but spends it only when it is released. Each new query is built with the next unspent share. So if you build all the queries first, every one gets the first share, and the product query gets a quarter's noise but is charged half. So build, check and release each query in turn, as the script does.

## Release overall statistics with error bars {#sec-relstat-scalars}

The first query releases the count of lines, the revenue and the median line value. The context holds the raw lines, so each query truncates them itself.

```{.python excerpt="relstat-scalars"}
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
```

`signed=True` keeps negative noisy counts, and `summarize` reports, before anything is released, each statistic's noise and 95% error bar.

```text
scalars len: Frame Length, Integer Gaussian, scale=1,453.38 95% accuracy=2,850.0
scalars revenue: Sum, Float Gaussian, scale=290,804.33 95% accuracy=569,966.0
scalars median: 0.5-Quantile, GumbelMin, scale=1,453.38 95% accuracy=none
```

With probability 0.95, the released count of lines is within 2,850 of the truth, and the revenue within £569,966. These bars depend only on the bounds and the budget, so you can read them before the release and decide whether the budget is enough.

The output also shows how OpenDP splits a query's share among its statistics: each gets the same noise scale relative to its sensitivity. The count and the median both get 1,453.38, and the revenue, whose sensitivity is 200 times the count's, about 200 times that. A mean, `.dp.mean`, is a sum and a count, so it takes two shares. Its error bar follows from theirs ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md#sec-lg-recipe)). To give one statistic more budget, put it in a query of its own with its own weight. The median is chosen by the exponential mechanism, sampled by `GumbelMin`, among candidates in steps of £0.50 across the value bounds. So its error bar is in ranks, and `summarize` gives none ([Selection](05-selection.md#sec-sel-apps)).

The error bars cover the noise only, not the bias of truncation and clipping, which together keep only 48.5% of the revenue. So the private revenue estimates a truncated total, not the shop's takings. Say so, and choose a larger $k$ if the full total matters more than the noise.

## Group by a public key list {#sec-relstat-public-keys}

Lines and revenue per country come next, and the key list must not come from the data. The distinct values of the Country column would publish which countries appear in the private data, which no noise covers. So the script fixes its keys in advance. They are the United Kingdom, the 27 EU member states, the 4 of the European Free Trade Association, the United States, Canada, Australia, Japan, China and "Other". A rule renames three of the source's spellings, such as `EIRE` to Ireland, and maps every other value to Other.

```{.python excerpt="relstat-public-keys"}
by_country = (
    context.query()
    .filter(sample_per_customer(K, seed))
    .group_by("Market")
    .agg(dp.len(signed=True), value.dp.sum((low, high)).alias("revenue"))
    .with_keys(pl.LazyFrame({"Market": MARKETS}))  # every key is released, zeros included
)
country_summary = by_country.summarize(alpha=0.05)
countries_hat = by_country.release().collect()
```

```text
per country len: Frame Length, Integer Gaussian, scale=1,185.25 95% accuracy=2,324.0
per country revenue: Sum, Float Gaussian, scale=237,698.48 95% accuracy=465,880.5
```

Every key is released, including the 11 with no purchase lines, whose counts are pure noise, so the release says nothing about which markets have customers. Each market's count has a 95% error bar of 2,324, and its revenue one of about £466,000. Germany's true count of 5,379 is still useful, but its revenue of £131,902 is smaller than its error bar. So publish the error bar with every cell, or merge small markets into Other before the release.

## Group by unknown keys with partition selection {#sec-relstat-partitions}

The list of products is not public, and a product bought by one customer would reveal that purchase simply by appearing in the table. So the release decides privately which products to show, by **partition selection**. It releases only the products whose noisy count clears a threshold, so a product with few customers almost never appears. The threshold is what costs $\delta$ ([Beyond global sensitivity](07-sensitivity.md)).

To count customers rather than lines, the query first keeps one line per customer and product, so $L_\infty = 1$. It then keeps at most $m$ products per customer, so $L_0 = m$, and the script measures the trade-off that $m$ sets on this public set.

```text
m=1: pairs kept 1.6%  customers truncated 97.9%  scale=5.93  threshold=29  products above it: 1
m=5: pairs kept 7.9%  customers truncated 92.4%  scale=13.27  threshold=66  products above it: 9
m=10: pairs kept 15.1%  customers truncated 83.6%  scale=18.77  threshold=96  products above it: 19
m=20: pairs kept 27.5%  customers truncated 68.3%  scale=26.54  threshold=138  products above it: 30
m=50: pairs kept 52.7%  customers truncated 38.6%  scale=41.96  threshold=225  products above it: 50
```

The threshold grows with $m$, from 29 to 225, because each customer can push more groups over it, but the counts grow faster. So the number of products whose true count clears the threshold rises from 1 to 50, and the script takes $m = 50$ (`M` in the code), again chosen on public data.

```{.python excerpt="relstat-partition-selection"}
by_product = (
    context.query()
    .filter(sample_per_customer(1, seed, "StockCode"))  # one line per customer and product
    .filter(sample_per_customer(M, seed))  # at most M products per customer
    .group_by("StockCode")
    .agg(dp.len(signed=True).alias("customers"))
)
product_summary = by_product.summarize(alpha=0.05)
products_hat = by_product.release().collect()
```

```text
per product customers: Frame Length, Integer Gaussian, scale=41.96 95% accuracy=83.0 threshold=225
```

A product is released only if its noisy count reaches 225, and a released count has a 95% error bar of 83. Never publish how many products were dropped, or which, since both come from the private data. Products just below the threshold get in when their noise is positive, so treat a count near 225 as a sign of popularity, not as a measurement.

OpenDP's `truncate_num_groups` should bound the groups per customer, but in version 0.16.0 it does not work. With the default `keep="sample"`, `summarize` accepts the query, but `release` fails after spending the query's share, so run each query end to end on public data first. With `keep="first"` it keeps one group fewer than asked, the first in sort order, which biases the counts. The filters above avoid both problems only because $L_\infty = 1$ turns groups into lines. For $L_\infty > 1$ this book knows of no unbiased group bound in this version.

## Check the release before publishing {#sec-relstat-check}

To check that the code does what the plan says, the script compares the private values with the true ones, after the same truncation and clipping.

```text
true lines after truncation: 219,291 (all purchase lines: 397,884)
true revenue after truncation and clipping: 4,318,947 (all lines, unclipped: 8,911,408)
true median line value after truncation: 15.00 (all lines: 11.80)
```

The release itself is random. One run printed the following.

```text
released (one run):
  lines 218,972  revenue 4,397,410  median 14.50
  United Kingdom: lines 196,311  revenue 3,927,959
  Germany: lines 7,151  revenue 643,094
  Spain: lines -546  revenue -37,048
  country line counts within their 95% accuracy: 34 of 38
  products released: 63
```

Read it against the error bars: about 95% of the market counts should fall inside theirs. Negative values are noise on small markets, and rounding them up to zero is post-processing, which costs no budget. Expect more than the 50 products whose true count clears the threshold, since products just below it get in when their noise is positive. Compare the private median with the truncated £15.00, not the £11.80 of all lines.

The check fails when a value lies far outside its error bar, which points to a wrong sensitivity. It also fails when far more or far fewer than 95% of the cells fall inside, or when a released product's true count lies far below the threshold. In production you cannot compare against the truth, because a decision to publish that depends on the true values is itself a release of them. So run this check on historical or public data while you develop, and release the real data once, with the code you checked.

## Write the guarantee down {#sec-relstat-statement}

The script prints the claim from the variables that set the bounds and the budget, so it cannot drift from the code.

```text
This release satisfies (epsilon = 1, delta = 1e-05)-differential privacy for adding or
removing one customer and all their invoice lines, with at most 100 lines per customer
(sampled at random), at most 50 random products per customer in the product counts, and
line values clipped to [0, 200] pounds, under the central model, using OpenDP 0.16.0 with
Gaussian noise (accounted as 0.0284-zCDP plus 5e-06 for thresholds, converted to epsilon
and delta). The market list was fixed in advance; the product list is not public.
```

The statement names everything a reviewer needs to recompute the claim ([Writing the guarantee down](01-definition.md#sec-dpdef-statement)). Next to it, say what it does not cover. Another release about the same customers adds its own budget. If that one is pure $\varepsilon$-DP, you can add the $(\varepsilon, \delta)$ pairs. Or convert it to $\rho = \varepsilon^2/2$, add that to this release's $\rho$ and then convert, whichever total is smaller ([Concentrated differential privacy](../part04-privacy-accounting/03-concentrated-dp.md)). And the figures describe truncated data, which the dashboard should say. [Release and governance](../part05-dp-ml/12-release-governance.md) maps such a claim to data-protection law.

## Checklist

| Step | Default | Check |
| --- | --- | --- |
| Unit and budget | One person, add/remove; one $\varepsilon$; $\delta \le 1/n^{1.1}$ | $n$ counts people, not rows |
| Contribution bound | $k$ from a high percentile of public or historical data; random sample | No bound read from the release data |
| Value bounds | From the domain or the business | Share clipped and bias stated |
| Context | zCDP with Gaussian noise; weights by need | $\rho$ converted to $(\varepsilon, \delta)$ equals the claim |
| Queries | Build, summarize and release one at a time | Scales match the plan; each query run end to end on public data |
| Public keys | Fixed in advance, values mapped by a rule; every key released | No key list built from the data |
| Unknown keys | Partition selection with $L_0$ and $L_\infty$ | Number dropped never published |
| Check | On historical or public data | About 95% inside their error bars |
| Statement | Printed by the releasing code | Unit, bounds, version and public information named |

: The release checklist: each step, its default and its check. {#tbl-relstat-checklist}

## References

- The OpenDP documentation, "Polars" and "Grouping" in its user guide: contexts, margins and partition selection.
- Wilson et al., "Differentially Private SQL with Bounded User Contribution" (PETS 2020): contribution bounding in a production SQL engine.
- Desfontaines, Voss, Gipson & Mandayam, "Differentially private partition selection" (PETS 2022): the threshold and why it needs $\delta$.

**Derivations.** This workflow has no counterpart in the full edition. The results it uses are derived in the full edition's chapters on [the definition](../../book-full/part03-dp-foundations/01-definition.md), [its properties](../../book-full/part03-dp-foundations/02-properties.md), [the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md), [selection](../../book-full/part03-dp-foundations/05-selection.md) and [sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md).
