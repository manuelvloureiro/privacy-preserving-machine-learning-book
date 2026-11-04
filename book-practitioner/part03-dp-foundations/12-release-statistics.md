# Release statistics with a library

## Why this matters

An online shop wants a dashboard of purchase lines, revenue and typical sale, overall and by country, plus customer counts by product. Because these statistics describe customers, their release needs a differential privacy (DP) guarantee. A customer may contribute many rows, so a row-level guarantee would protect a purchase rather than the customer. This chapter uses OpenDP to build the release, define its privacy unit and state its guarantee.

The data are Online Retail from the University of California, Irvine (UCI) Machine Learning Repository: 541,909 invoice lines from a British online shop. Each line records a product, quantity, unit price, customer and country. The script [practitioner_release_statistics.py](../../code/part03-dp-foundations/practitioner_release_statistics.py) runs every step in less than a minute with OpenDP 0.16.0 and Polars 1.36.1.

```bash
uv run python code/part03-dp-foundations/practitioner_release_statistics.py
```

OpenDP uses an unseeded cryptographic generator, so released values vary between runs. This page quotes deterministic scales, error bars and thresholds, and marks one-run releases explicitly.

## Choose the library {#sec-relstat-library}

Choose a library that can bound the contribution of a person with many rows (@tbl-relstat-libraries).

| Library | Runs on | Person with many rows | Groups not known in advance | Use it for |
| --- | --- | --- | --- | --- |
| OpenDP | Polars, one machine | Yes, with an identifier column | Yes, by a noisy threshold | Statistics on one machine; the default here |
| diffprivlib | NumPy, scikit-learn | No, one row per person | No | Quick experiments and [simple models](../part05-dp-ml/01-dp-erm.md) |
| PipelineDP | Apache Beam, Spark | Yes, built in | Yes, built in | Pipelines that already run on Beam or Spark |
| Tumult Analytics | Spark | Yes, with an identifier column | Yes, in a separate query | Large releases on Spark |

: Libraries for releasing statistics, and what each can express. {#tbl-relstat-libraries}

PipelineDP and Tumult were not run; their entries are based on their documentation. Do not implement a mechanism yourself. Use [one accountant](02-properties.md#sec-dpprop-budget) to track the total budget for the release.

## Decide the unit and the budget {#sec-relstat-unit}

The **unit of privacy** is a customer. Neighbouring datasets differ by one customer and all their lines, added or removed ([The definition of differential privacy](01-definition.md#sec-dpdef-neighbours)). The script removes returns, zero-price lines and lines without a customer identifier. Each filter uses only the line itself, so adding or removing a customer affects only that customer's lines.

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

The slack $\delta$ follows the book's default for $n$ people ([How small δ must be](01-definition.md#sec-dpdef-small-delta)), giving $10^{-5}$. OpenDP warns that $\delta$ should be at most $10^{-6}$, while this book sets it from the population size. The entire release has privacy level $\varepsilon = 1$: all tables describe the same customers, and their costs compose ([Post-processing, group privacy and composition](02-properties.md#sec-dpprop-comp)).

Under a line-level guarantee, the largest customer represents 7,847 records, and group privacy multiplies $\varepsilon$ by the group size. This maximum is reported only because the dataset is public.

## Bound each customer's contribution {#sec-relstat-bound}

Let $k$ be the maximum number of lines retained per customer in each query. A smaller $k$ reduces noise but discards more data; a larger $k$ retains more data but requires more noise. The script measures this trade-off on the public dataset.

```text
k=10: lines kept 10.2%  customers truncated 84.8%
k=25: lines kept 22.4%  customers truncated 64.6%
k=50: lines kept 36.9%  customers truncated 44.0%
k=100: lines kept 55.1%  customers truncated 24.9%
k=200: lines kept 72.7%  customers truncated 10.4%
k=500: lines kept 88.2%  customers truncated 1.8%
```

Choose $k$ from a high percentile in public or historical data, never from the release data. If neither is available, spend a small part of the budget on a private high quantile over candidates fixed in advance ([Selection](05-selection.md#sec-sel-apps)). This chapter uses $k = 100$, below the public-data 90th percentile of 208 because noise grows with $k$. It retains 55.1% of lines and truncates about a quarter of customers.

Keeping the first $k$ lines by date would bias statistics toward early purchases. The script instead samples $k$ lines per customer. The filter below matches OpenDP's `truncate_per_group(k, keep="sample")` while using a seed to keep validation values reproducible.

```{.python excerpt="relstat-truncate"}
def sample_per_customer(k: int, seed: int, *by: str) -> pl.Expr:
    """True for at most k randomly chosen lines of each customer (within each value of `by`)."""
    return pl.int_range(pl.len()).shuffle(seed=seed).over("CustomerID", *by) < k
```

Bound each line value, defined as quantity times unit price. The script clips values to £0–£200. The lower bound follows from removing returns; the business sets the upper bound, which clips 1.02% of lines. Thus one customer can affect revenue by at most 100 lines of £200 each. Set value bounds from the domain, not the observed minimum or maximum ([Beyond global sensitivity](07-sensitivity.md#sec-sens-clip)).

A grouped query may require two further bounds: $L_0$, the number of groups one customer may affect, and $L_\infty$, the number of lines they may contribute to one group. Country counts need neither because a customer's $k$ lines change the total counts by at most $k$, already covered by the noise. Product partition selection requires both (@sec-relstat-partitions).

## Set up the context {#sec-relstat-context}

OpenDP's **context** stores the data, privacy unit, total budget and allocation across queries. Contexts require `dp.enable_features("contrib")`, which enables components that are not formally vetted. Reviewers who require vetted code must check each call against OpenDP's documentation.

```{.python excerpt="relstat-context"}
context = dp.Context.compositor(
    data=lines.lazy(),
    privacy_unit=dp.unit_of(contributions=1, identifier="CustomerID"),
    privacy_loss=dp.loss_of(rho=rho, delta=delta / 2),  # Gaussian noise; delta/2 for thresholds
    split_by_weights=WEIGHTS,  # scalars, countries, products
    margins=[dp.polars.Margin(by=[], max_length=MAX_LINES)],
)
```

The identifier column defines each customer as one privacy unit. Thus `contributions=1` represents adding or removing one customer with all their lines; truncation bounds those lines. For replace-one neighbours, OpenDP instead takes `changes=`. The budget uses **zero-concentrated differential privacy (zCDP)** ([Concentrated differential privacy](../part04-privacy-accounting/03-concentrated-dp.md)), represented by $\rho$. Add $\rho$ across queries, then convert the total to $(\varepsilon, \delta)$.

With a zCDP budget, the context adds Gaussian noise to each count and sum. This supports product counts, where one customer may affect many groups ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md#sec-lg-choose)). The script's `rho_for` selects $\rho = 0.0284$, the largest value OpenDP converts to $\varepsilon \le 1$ using half of $\delta$. The remaining half pays for the threshold in @sec-relstat-partitions. The margin supplies a public upper bound on line counts, which OpenDP needs before summing floating-point values.

The weights allocate one quarter of the budget to overall statistics, one quarter to countries and one half to products, which need the most. The script prints this allocation and converts the total budget back to the $(\varepsilon, \delta)$ stated in the guarantee.

```text
budget scalars: rho=0.0071 delta=1.25e-06
budget countries: rho=0.0071 delta=1.25e-06
budget products: rho=0.0142 delta=2.50e-06
whole release as (epsilon, delta): (1.0000, 1e-05)
```

OpenDP assigns a query its budget share when the query is built and spends that share on release. Each new query receives the next unspent share. Building all queries before releasing them can therefore assign each the first share; the product query would receive one quarter's noise while costing half the budget. Build, inspect and release each query in sequence, as the script does.

## Release overall statistics with error bars {#sec-relstat-scalars}

The first query releases line count, revenue and median line value. Because the context stores the raw lines, each query applies its own truncation.

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

`signed=True` retains negative noisy counts. Before release, `summarize` reports each statistic's noise scale and 95% error bar.

```text
scalars len: Frame Length, Integer Gaussian, scale=1,453.38 95% accuracy=2,850.0
scalars revenue: Sum, Float Gaussian, scale=290,804.33 95% accuracy=569,966.0
scalars median: 0.5-Quantile, GumbelMin, scale=1,453.38 95% accuracy=none
```

With probability 0.95, the released line count is within 2,850 of its true value and revenue is within £569,966. These error bars depend only on bounds and budget, so inspect them before release to assess whether the budget is sufficient.

The output also shows how OpenDP divides a query's budget among its statistics. Each statistic receives the same noise scale relative to its sensitivity. The count and median therefore have scale 1,453.38, while revenue, with 200 times the count's sensitivity, has about 200 times that scale. To allocate more budget to one statistic, place it in a separate weighted query. `GumbelMin`, an implementation of the exponential mechanism, selects the median from candidates spaced £0.50 apart. Its error is measured in ranks, so `summarize` provides no error bar ([Selection](05-selection.md#sec-sel-apps)).

The error bars cover noise, not bias from truncation and clipping, which together retain only 48.5% of revenue. The private statistic therefore estimates a truncated total, not the shop's full takings. State this in the release. If coverage matters more than noise, increase $k$.

## Group by a public key list {#sec-relstat-public-keys}

The next query reports lines and revenue by country. Its key list must be fixed in advance because the distinct `Country` values reveal which countries occur in private data, and noise does not protect that list. The script uses 38 keys: the United Kingdom, 27 EU member states, four European Free Trade Association states, the United States, Canada, Australia, Japan, China and "Other". A mapping standardizes three spellings, including `EIRE` to Ireland, and maps all remaining values to Other.

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

The query releases every key, including 11 countries with no purchase lines; their counts contain only noise. Thus, the key list does not disclose which markets have customers. Each count has a 95% error bar of 2,324, and each revenue an error bar of about £466,000. Germany's true count of 5,379 is informative, but its revenue of £131,902 is smaller than the error bar. Publish each cell's error bar or combine small markets into Other before release.

## Group by unknown keys with partition selection {#sec-relstat-partitions}

The product list is private: a row for a product would reveal that at least one customer bought it. The release therefore uses **partition selection** to select products privately. It publishes only products whose noisy counts exceed a threshold, making products with few customers unlikely to appear. The threshold incurs $\delta$ ([Beyond global sensitivity](07-sensitivity.md)).

To count customers rather than lines, the query first retains one line per customer-product pair, setting $L_\infty$, the lines per customer in one group, to 1. It then retains at most $m$ products per customer, setting $L_0$, the groups per customer, to $m$. The script evaluates this trade-off on public data.

```text
m=1: pairs kept 1.6%  customers truncated 97.9%  scale=5.93  threshold=29  products above it: 1
m=5: pairs kept 7.9%  customers truncated 92.4%  scale=13.27  threshold=66  products above it: 9
m=10: pairs kept 15.1%  customers truncated 83.6%  scale=18.77  threshold=96  products above it: 19
m=20: pairs kept 27.5%  customers truncated 68.3%  scale=26.54  threshold=138  products above it: 30
m=50: pairs kept 52.7%  customers truncated 38.6%  scale=41.96  threshold=225  products above it: 50
```

The threshold rises from 29 to 225 as $m$ increases because each customer can affect more groups. Product counts rise faster: the number of products whose true count exceeds the threshold grows from 1 to 50. The script uses $m = 50$ (`M` in the code), selected using public data.

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

The release shows a product only when its noisy count is at least 225; released counts have a 95% error bar of 83. Do not publish the number or identity of omitted products because both depend on private data. Positive noise can include a product just below the threshold. Treat counts near 225 as evidence of possible popularity, not as precise measurements.

In OpenDP 0.16.0, `truncate_num_groups` is intended to bound groups per customer but does not work. With default `keep="sample"`, `summarize` accepts the query, but `release` fails after consuming its budget share. Test each query end to end on public data first. With `keep="first"`, the function retains one fewer group than requested and biases counts. The filters above avoid both failures only because $L_\infty = 1$ makes groups equivalent to lines. This book knows of no unbiased group bound for $L_\infty > 1$ in this version.

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

The check should fail when a value lies far outside its error bar, indicating incorrect sensitivity, or when substantially fewer or more than 95% of cells fall inside. It should also fail if a released product's true count is far below the threshold. In production, do not use true values to decide what to publish; that decision would itself release private information. Run these checks during development on historical or public data, then release real data once with the validated code.

## Write the guarantee down {#sec-relstat-statement}

The script constructs the privacy statement from the variables that define bounds and budget, keeping the statement aligned with the implementation.

```text
This release satisfies (epsilon = 1, delta = 1e-05)-differential privacy for adding or
removing one customer and all their invoice lines, with at most 100 lines per customer
(sampled at random), at most 50 random products per customer in the product counts, and
line values clipped to [0, 200] pounds, under the central model, using OpenDP 0.16.0 with
Gaussian noise (accounted as 0.0284-zCDP plus 5e-06 for thresholds, converted to epsilon
and delta). The market list was fixed in advance; the product list is not public.
```

The statement gives reviewers the information needed to recompute the guarantee ([Writing the guarantee down](01-definition.md#sec-dpdef-statement)). Also state what it excludes. Any further release about the same customers adds to the total budget. For a pure $\varepsilon$-DP release, add the two $(\varepsilon, \delta)$ pairs. Alternatively, add its $\rho = \varepsilon^2/2$ to this release's $\rho$, then convert the total; use the smaller result ([Concentrated differential privacy](../part04-privacy-accounting/03-concentrated-dp.md)). The dashboard must also state that its figures use truncated data. [Release and governance](../part05-dp-ml/12-release-governance.md) connects this guarantee to data-protection law.

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
