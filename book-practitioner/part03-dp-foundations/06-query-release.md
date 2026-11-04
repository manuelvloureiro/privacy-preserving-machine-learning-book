# Query release and private multiplicative weights

## Why this matters

Analysts may request every two-way table over several columns or a synthetic dataset that supports repeated queries. Both require differential privacy (DP). These tables are **marginal tables**: each counts records for combinations of attribute values, and each count is a **cell**.

Add noise to each table and calibrate it to the maximum change one person can make to that table. With one record per person, one person affects one cell, so a table costs about as much as one count. Subsequent computation on noisy tables is post-processing. If there are too many queries to noise every cell, fit synthetic data to a selected set of noisy tables.

## Which method to use {#sec-qr-choose}

Choose a method based on query count, record count and the number of attributes per record (@tbl-qr-choose).

| Situation | Use |
| --- | --- |
| A few marginal tables, with a public list of cells | Noisy counts on every cell, one budget share per table |
| Hundreds of tables or thousands of cells | Gaussian noise on every cell, scaled to the whole set of tables |
| A table whose list of values is not public | Noisy counts, keeping only cells that clear a noisy threshold ([Release statistics with a library](12-release-statistics.md)) |
| Far more queries than records, or queries not known in advance | Synthetic data from AIM or MST |
| Queries that arrive one at a time, perhaps adaptively | Online private multiplicative weights |

: Which query-release method to use. {#tbl-qr-choose}

Each person can affect every table, so the combined query is dense. The dense-query rule in [the Laplace and Gaussian mechanisms](03-laplace-gaussian.md#sec-lg-choose) identifies when Gaussian noise becomes more accurate. At privacy level $\varepsilon = 1$ and slack $\delta = 10^{-5}$, Gaussian noise is better for seven or more tables.

## Noise on every cell {#sec-qr-baseline}

Under add/remove neighbours, one record changes one cell of a marginal table by 1, so the table's sensitivity is 1. Laplace noise of scale $1/\varepsilon$ on each cell makes the table $\varepsilon$-DP by [parallel composition](02-properties.md#sec-dpprop-parallel). Under replace neighbours, a record can move between cells, changing two cells and doubling sensitivity.

Releasing several tables of the same records composes their costs. With Laplace noise, splitting the budget across tables makes per-cell error grow in proportion to table count. Gaussian noise scales with the square root of the sum of squared sensitivities. If each person affects $m$ tables, Gaussian error grows as $\sqrt m$.

For example, a table of 10,000 records has eight yes/no attributes, so $2^8 = 256$ possible records. The release is every cell of the marginal table of every set of attributes, 6,560 cells in all. Each cell is a fraction of the records, under replace neighbours. A uniform guess, which ignores the data, has a largest error of 0.232 on these cells. At $\varepsilon = 1$, Laplace noise on every cell is worse, with 0.497. Gaussian noise at $\delta = 10^{-6}$ is much better, with 0.037.

Two errors are common. Calibrating each cell as a separate query spends $\varepsilon$ once per table. Applying basic composition to thousands of cells also wastes budget. Calibrate noise to the complete set of tables.

## Releasing a marginal with OpenDP {#sec-qr-opendp}

The script [practitioner_adult_statistics.py](../../code/part03-dp-foundations/practitioner_adult_statistics.py) releases an education-by-income table from Adult at $\varepsilon = 1$. Adult contains 48,842 people with one row each. The survey documentation lists 16 education levels and two income classes, making the 32 table keys public. Release empty cells as well; omitting one would reveal that its count is zero. Map missing values to a listed key such as "Unknown", because OpenDP drops rows with keys outside the list. In OpenDP, `Margin(invariant="keys")` declares the cell list public, and `with_keys` restricts the release to that list.

```{.python excerpt="adult-marginal"}
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
```

Run `uv run python code/part03-dp-foundations/practitioner_adult_statistics.py` to print the deterministic summary.

```text
shape: (1, 5)
┌────────┬──────────────┬─────────────────┬───────┬──────────┐
│ column ┆ aggregate    ┆ distribution    ┆ scale ┆ accuracy │
│ ---    ┆ ---          ┆ ---             ┆ ---   ┆ ---      │
│ str    ┆ str          ┆ str             ┆ f64   ┆ f64      │
╞════════╪══════════════╪═════════════════╪═══════╪══════════╡
│ len    ┆ Frame Length ┆ Integer Laplace ┆ 1.0   ┆ 3.375618 │
└────────┴──────────────┴─────────────────┴───────┴──────────┘
```

Each cell receives integer Laplace noise with scale 1 ([Discrete and floating-point-safe mechanisms](08-discrete-mechanisms.md)). The 95% accuracy of 3.38 means each noisy count is within 3.38 of the true count with 95% probability. Report this error bar with every cell; it covers noise but not sampling error or data errors. OpenDP noise is unseeded, so the following values are from one run, alongside the true counts.

```text
  Doctorate >50K: 432  true 431  accuracy 3.38
  HS-grad <=50K: 13281  true 13281  accuracy 3.38
  Preschool <=50K: 82  true 82  accuracy 3.38
  Preschool >50K: 0  true 1  accuracy 3.38
```

Noise has little effect on large cells. The smallest true cell—one person with preschool education and high income—is smaller than the noise. Merge rare categories in advance if small counts matter. With `signed=True`, a noisy count may be negative, preserving unbiasedness. Clamping at zero, rounding and scaling cells to a released total are post-processing and cost no additional privacy.

Three implementation details matter. First, deriving cell keys from private data—for example, calling `unique()` on a private column—may reveal rare values. Use partition selection when no public codebook exists ([Release statistics with a library](12-release-statistics.md)). Second, each additional table over the same people needs a budget share; set `split_evenly_over` to the table count. Third, for many tables, specify `dp.loss_of(rho=...)` with a $\rho$ that converts to the desired $(\varepsilon, \delta)$ ([Release statistics with a library](12-release-statistics.md#sec-relstat-context)). OpenDP then uses Gaussian noise and accounts in zero-concentrated differential privacy (zCDP), which composes more tightly ([Concentrated differential privacy](../part04-privacy-accounting/03-concentrated-dp.md)).

## Synthetic data {#sec-qr-synth}

When the query set is too large to noise every cell, or queries are not known in advance, release synthetic data using four steps.

1. **Select** a set of marginal tables to measure, from domain knowledge or privately with the [exponential mechanism](05-selection.md#sec-sel-exp), scoring each table by how badly the current model fits it.
2. **Measure** each whole table with Gaussian noise, scaled to the whole set.
3. **Generate** a compact graphical model that fits the noisy tables.
4. **Sample** synthetic rows from the model.

All steps after measurement are post-processing, so synthetic rows can be queried and shared without further privacy cost. MST and AIM are standard methods developed by McKenna and coauthors; AIM selects tables adaptively. SmartNoise Synth, the `snsynth` package, provides both methods through one interface. It is not installed in this book's environment, so the example below was not run.

```text
# smartnoise-synth 1.0.8: not installed in this book's environment, not run
from snsynth import Synthesizer

synth = Synthesizer.create("aim", epsilon=1.0, delta=1e-6)
synth.fit(adult_pandas, categorical_columns=["education", "sex", "race", "occupation", "income"])
rows = synth.sample(48_842)
```

A synthetic table fits the measured tables by construction. Evaluate it on additional queries, but account for privacy when evaluation uses private data. Selecting the best of several runs based on private data is also a private query, and each run spends the generator's budget again. Evaluate settings on public data with the same structure or charge comparisons to the budget. Synthetic data from a generator without DP are not anonymized.

## Private multiplicative weights {#sec-qr-pmw}

The synthetic-data tools use the select-measure-update loop of the **multiplicative weights exponential mechanism (MWEM)** (Hardt, Ligett and McSherry 2012). MWEM maintains a guessed histogram with one weight per possible record—256 in the eight-attribute example. In a fixed number of rounds, it selects a poorly answered query using the exponential mechanism, measures it with fresh Laplace noise and updates the histogram toward that measurement. The correction count is bounded independently of the query count, so error grows logarithmically with the number of queries.

MWEM is useful for large query sets. In the eight-attribute example at privacy level 1, 38 rounds give a maximum error of 0.078. This is better than Laplace noise but about twice the Gaussian-noise error. Runtime grows with the number of possible records, limiting MWEM to a few attributes.

Fix the round count from the error bound before accessing private data; the bound uses only public quantities. Measure each query with fresh noise because exact selection scores could leak information. Release the average of the guesses, not the last one, because the accuracy guarantee applies only to the average. For sequential queries, an online variant pays only for its corrections ([Adaptive data analysis](../part05-dp-ml/11-adaptive-data-analysis.md)).

## Cheat sheet

| Method | Privacy | Error grows with |
| --- | --- | --- |
| Laplace noise on every cell | $\varepsilon$; one table costs $\varepsilon$ under add/remove | The number of tables |
| Gaussian noise on every cell | $(\varepsilon, \delta)$ or zCDP | The square root of the number of tables |
| Private multiplicative weights | $\varepsilon$ or $(\varepsilon, \delta)$ | The logarithms of the numbers of queries and possible records |
| Select, measure, generate (MST, AIM) | $(\varepsilon, \delta)$ | No closed form; the practical choice for many attributes |

: The query-release methods of this chapter. {#tbl-qr-cheat}

**Rules of thumb**

- Take the list of cells from a codebook, never from the private data.
- Add noise to whole tables, scaled to the set of tables, and post-process freely afterwards.
- Fit synthetic data once, and never pick the best of several runs on the private data.

## References

- McKenna, Mullins, Sheldon & Miklau, "AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data" (VLDB 2022): the select, measure and generate loop as used in practice.
- Hardt, Ligett & McSherry, "A Simple and Practical Algorithm for Differentially Private Data Release" (NeurIPS 2012): MWEM, with experiments on real tables.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/06-query-release.md).
