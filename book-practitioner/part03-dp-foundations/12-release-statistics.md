# Release statistics with a library

## Why this matters

Suppose that you are the analytics engineer of an online shop. The business wants a dashboard of purchase lines, revenue and the typical sale, overall and per country, and of how many customers bought each product. The figures describe customers, so they need a differential privacy (DP) guarantee. But one customer has many rows, and a guarantee for one row protects a single purchase, not the person behind a hundred. This chapter does the job with OpenDP, from the unit to the written guarantee.

The data are the Online Retail set from the University of California, Irvine (UCI) Machine Learning Repository, with 541,909 invoice lines of a British online shop. Each line records one product on one invoice, with its quantity, unit price, customer and country. The script [practitioner_release_statistics.py](../../code/part03-dp-foundations/practitioner_release_statistics.py) does every step in less than a minute, with OpenDP 0.16.0 and Polars 1.36.1.

```bash
uv run python code/part03-dp-foundations/practitioner_release_statistics.py
```

OpenDP takes its noise from a cryptographic generator that has no seed, so the released values change on every run. So this page quotes scales, error bars and thresholds, which contain no noise, and shows released values only in a block marked as one run.

## Choose the library {#sec-relstat-library}

The main question is whether a library can bound the contribution of a person with many rows (@tbl-relstat-libraries).

| Library | Runs on | Person with many rows | Groups not known in advance | Use it for |
| --- | --- | --- | --- | --- |
| OpenDP | Polars, one machine | Yes, with an identifier column | Yes, by a noisy threshold | Statistics on one machine; the default here |
| diffprivlib | NumPy, scikit-learn | No, one row per person | No | Quick experiments and [simple models](../part05-dp-ml/01-dp-erm.md) |
| PipelineDP | Apache Beam, Spark | Yes, built in | Yes, built in | Pipelines that already run on Beam or Spark |
| Tumult Analytics | Spark | Yes, with an identifier column | Yes, in a separate query | Large releases on Spark |

: Libraries for releasing statistics, and what each can express. {#tbl-relstat-libraries}

We did not run PipelineDP and Tumult, so their rows come from their documentation. Whichever library you use, do not write a mechanism yourself. Keep [one accountant](02-properties.md#sec-dpprop-budget), which adds up the budget, for the whole release.

## Decide the unit and the budget {#sec-relstat-unit}

The **unit of privacy**, which the guarantee protects, is a customer. So two neighbouring datasets differ by one customer with all their lines, added or removed ([The definition of differential privacy](01-definition.md#sec-dpdef-neighbours)). The script removes returns, lines with a zero price and lines with no customer identifier, which belong to nobody. Each filter looks only at the line itself, so a customer who is added or removed still changes only their own lines.

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

The slack $\delta$ follows the book's default for $n$ people ([How small δ must be](01-definition.md#sec-dpdef-small-delta)), which gives $10^{-5}$. OpenDP warns that $\delta$ must be at most $10^{-6}$, but this book ties $\delta$ to the number of people. The privacy level is $\varepsilon = 1$ for the whole release, because all its tables describe the same customers and their costs add up ([Post-processing, group privacy and composition](02-properties.md#sec-dpprop-comp)).

With a guarantee per line, the largest customer is a group of 7,847 records, and group privacy multiplies $\varepsilon$ by the size of the group. You can print that maximum only because this set is public.

## Bound each customer's contribution {#sec-relstat-bound}

Write $k$ for the largest number of lines that a customer may give, so that every query keeps at most $k$ lines of each customer. A small $k$ means little noise but removes data, and a large $k$ keeps the data but needs more noise. The script measures this trade-off on the public set.

```text
k=10: lines kept 10.2%  customers truncated 84.8%
k=25: lines kept 22.4%  customers truncated 64.6%
k=50: lines kept 36.9%  customers truncated 44.0%
k=100: lines kept 55.1%  customers truncated 24.9%
k=200: lines kept 72.7%  customers truncated 10.4%
k=500: lines kept 88.2%  customers truncated 1.8%
```

The default $k$ is a high percentile of lines per customer in public or historical data, never in the data that you release. If you have neither, spend a small share of the budget on a private high quantile, over candidates fixed in advance ([Selection](05-selection.md#sec-sel-apps)). This chapter takes $k = 100$, which is less than the 90th percentile of 208 because the noise grows with $k$. It keeps 55.1% of the lines and truncates a quarter of the customers.

The first $k$ lines by date bias every statistic towards early purchases, so the script keeps a random sample of $k$ lines per customer. The filter below does what OpenDP's `truncate_per_group(k, keep="sample")` does, but with a seed, so that the true values of the check stay fixed.

```{.python excerpt="relstat-truncate"}
def sample_per_customer(k: int, seed: int, *by: str) -> pl.Expr:
    """True for at most k randomly chosen lines of each customer (within each value of `by`)."""
    return pl.int_range(pl.len()).shuffle(seed=seed).over("CustomerID", *by) < k
```

Values need bounds too. The value of a line is its quantity times its unit price, and the script clips it to between £0 and £200. The lower end is true because the script removed the returns. The upper end is a bound that the business sets, and it clips 1.02% of the lines. So one customer can move the revenue by at most 100 lines of £200 each. Take value bounds from the domain, never from the minimum or maximum of the data ([Beyond global sensitivity](07-sensitivity.md#sec-sens-clip)).

A group-by can use two more bounds. The bound $L_0$ is the number of groups that one customer may touch. The bound $L_\infty$ is the number of lines that a customer may put in one group. Countries need neither, because the $k$ lines of a customer change the country counts by at most $k$ in total, and the noise already covers that. Products need both bounds for partition selection (@sec-relstat-partitions).

## Set up the context {#sec-relstat-context}

OpenDP keeps the whole release in a **context**, which holds the data, the unit, the total budget and its split between queries. Contexts need `dp.enable_features("contrib")`, which turns on components that are not formally vetted. So a reviewer who needs vetted code must examine each call in the documentation of OpenDP.

```{.python excerpt="relstat-context"}
context = dp.Context.compositor(
    data=lines.lazy(),
    privacy_unit=dp.unit_of(contributions=1, identifier="CustomerID"),
    privacy_loss=dp.loss_of(rho=rho, delta=delta / 2),  # Gaussian noise; delta/2 for thresholds
    split_by_weights=WEIGHTS,  # scalars, countries, products
    margins=[dp.polars.Margin(by=[], max_length=MAX_LINES)],
)
```

The identifier column makes the unit a customer. So `contributions=1` means one customer with all their lines, added or removed, and the truncation filters bound those lines. For replace, OpenDP takes `changes=` instead. The budget is in **zero-concentrated differential privacy (zCDP)** ([Concentrated differential privacy](../part04-privacy-accounting/03-concentrated-dp.md)), which gives privacy as one number, written $\rho$. Values of $\rho$ add up across queries and change to $(\varepsilon, \delta)$ at the end.

With a zCDP budget, the context adds Gaussian noise to every count and sum. That suits the product counts, where one customer touches many groups ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md#sec-lg-choose)). The script's `rho_for` finds $\rho = 0.0284$, the largest value that the conversion of OpenDP turns into $\varepsilon \le 1$ with half of $\delta$. The other half pays for the threshold of @sec-relstat-partitions. The margin gives a public upper bound on the number of lines, which OpenDP needs before it adds floating-point values.

The weights give a quarter of the budget to the overall statistics, a quarter to the countries and half to the products, which need it most. The script prints the split and converts the whole release back to the $(\varepsilon, \delta)$ of the statement.

```text
budget scalars: rho=0.0071 delta=1.25e-06
budget countries: rho=0.0071 delta=1.25e-06
budget products: rho=0.0142 delta=2.50e-06
whole release as (epsilon, delta): (1.0000, 1e-05)
```

A query takes its share of the budget when you build it, but spends the share only when you release it. Each new query gets the next unspent share. So if you build all the queries first, each one gets the first share. The product query then gets the noise of a quarter but costs half. Build, check and release each query in turn, as the script does.

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

With probability 0.95, the released count of lines is within 2,850 of the truth, and the revenue within £569,966. These bars depend only on the bounds and the budget, so you can read them before the release and decide if the budget is sufficient.

The output also shows how OpenDP splits the share of a query among its statistics. Each statistic gets the same noise scale relative to its sensitivity. So the count and the median both get 1,453.38, and the revenue, with 200 times the sensitivity of the count, gets about 200 times that. To give one statistic more budget, put it in a query of its own with its own weight. The exponential mechanism, sampled by `GumbelMin`, chooses the median among candidates in steps of £0.50. So its error bar is in ranks, and `summarize` gives none ([Selection](05-selection.md#sec-sel-apps)).

The error bars cover only the noise, not the bias of truncation and clipping, which together keep only 48.5% of the revenue. So the private revenue estimates a truncated total, not the takings of the shop. Say so in the release. If the full total is more important than the noise, choose a larger $k$.

## Group by a public key list {#sec-relstat-public-keys}

The next query gives lines and revenue per country, and its key list must not come from the data. The distinct values of the Country column tell which countries are in the private data, and no noise covers that. So the script fixes its keys in advance. The 38 keys are the United Kingdom, the 27 EU member states and the 4 states of the European Free Trade Association. Then come the United States, Canada, Australia, Japan, China and "Other". A rule renames three spellings, such as `EIRE` to Ireland, and maps every other value to Other.

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

The query releases every key, also the 11 with no purchase lines, whose counts are only noise. So the release tells nothing about which markets have customers. The count of each market has a 95% error bar of 2,324, and its revenue one of about £466,000. The true count of 5,379 for Germany is still useful, but its revenue of £131,902 is less than its error bar. So publish the error bar with every cell, or merge small markets into Other before the release.

## Group by unknown keys with partition selection {#sec-relstat-partitions}

The list of products is not public. If one customer buys a product, its row in the table shows that purchase. So the release uses **partition selection** to choose privately which products to show. It releases only the products whose noisy count is above a threshold, so a product with few customers almost never appears. The threshold is what costs $\delta$ ([Beyond global sensitivity](07-sensitivity.md)).

To count customers, not lines, the query first keeps one line per customer and product. So $L_\infty$, the lines per customer in one group, is 1. It then keeps at most a number $m$ of products per customer, so $L_0$, the groups per customer, is $m$. The script measures the trade-off that $m$ sets on this public set.

```text
m=1: pairs kept 1.6%  customers truncated 97.9%  scale=5.93  threshold=29  products above it: 1
m=5: pairs kept 7.9%  customers truncated 92.4%  scale=13.27  threshold=66  products above it: 9
m=10: pairs kept 15.1%  customers truncated 83.6%  scale=18.77  threshold=96  products above it: 19
m=20: pairs kept 27.5%  customers truncated 68.3%  scale=26.54  threshold=138  products above it: 30
m=50: pairs kept 52.7%  customers truncated 38.6%  scale=41.96  threshold=225  products above it: 50
```

The threshold grows with $m$, from 29 to 225, because each customer can push more groups over it. But the counts grow faster, so the number of products whose true count is above the threshold goes from 1 to 50. The script takes $m = 50$ (`M` in the code), again chosen on public data.

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

The release shows a product only if its noisy count is 225 or more, and a released count has a 95% error bar of 83. Never publish how many products were left out, or which, because both come from the private data. A product just below the threshold gets in when its noise is positive. So treat a count near 225 as a sign of popularity, not as a measurement.

In OpenDP 0.16.0, `truncate_num_groups` should bound the groups per customer, but it does not work. With the default `keep="sample"`, `summarize` accepts the query, but `release` fails after it spends the share of the query. So run each query end to end on public data first. With `keep="first"`, it keeps one group fewer than asked, and this biases the counts. The filters above avoid both problems only because $L_\infty = 1$ makes groups into lines. For $L_\infty > 1$, this book knows of no unbiased group bound in this version.

## Check the release before publishing {#sec-relstat-check}

To make sure that the code does what the plan says, the script compares the private values with the true values after the same truncation and clipping.

```text
true lines after truncation: 219,291 (all purchase lines: 397,884)
true revenue after truncation and clipping: 4,318,947 (all lines, unclipped: 8,911,408)
true median line value after truncation: 15.00 (all lines: 11.80)
```

The release is random, and one run printed the following.

```text
released (one run):
  lines 218,972  revenue 4,397,410  median 14.50
  United Kingdom: lines 196,311  revenue 3,927,959
  Germany: lines 7,151  revenue 643,094
  Spain: lines -546  revenue -37,048
  country line counts within their 95% accuracy: 34 of 38
  products released: 63
```

Compare it with the error bars, which must hold about 95% of the market counts. Negative values are noise on small markets, and you can round them up to zero, because post-processing costs no budget. Expect more than the 50 products whose true count is above the threshold, for the reason above. Compare the private median with the truncated £15.00, not with the £11.80 of all lines.

The check fails if a value is far outside its error bar, which shows a wrong sensitivity. It also fails if far more or far fewer than 95% of the cells are inside. And it fails if the true count of a released product is far below the threshold. In production you cannot compare with the truth, because a decision to publish that uses the true values is a release of them. So run this check on historical or public data during development. Then release the real data once, with the code that you checked.

## Write the guarantee down {#sec-relstat-statement}

The script prints the claim from the variables that set the bounds and the budget, so the claim always agrees with the code.

```text
This release satisfies (epsilon = 1, delta = 1e-05)-differential privacy for adding or
removing one customer and all their invoice lines, with at most 100 lines per customer
(sampled at random), at most 50 random products per customer in the product counts, and
line values clipped to [0, 200] pounds, under the central model, using OpenDP 0.16.0 with
Gaussian noise (accounted as 0.0284-zCDP plus 5e-06 for thresholds, converted to epsilon
and delta). The market list was fixed in advance; the product list is not public.
```

The statement names all that a reviewer needs to compute the claim again ([Writing the guarantee down](01-definition.md#sec-dpdef-statement)). Next to it, say what it does not cover. Another release about the same customers adds its own budget to the claim. If that release is pure $\varepsilon$-DP, you can add the two $(\varepsilon, \delta)$ pairs. Or you can add its $\rho = \varepsilon^2/2$ to this release's $\rho$ and then convert. Use the smaller total ([Concentrated differential privacy](../part04-privacy-accounting/03-concentrated-dp.md)). Also, the dashboard must say that the figures describe truncated data. [Release and governance](../part05-dp-ml/12-release-governance.md) maps such a claim to data-protection law.

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
