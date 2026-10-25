# Query release and private multiplicative weights

## Why this matters

Analysts rarely want one statistic. They ask for every two-way table over a handful of columns, or for a synthetic file they can query as often as they like. Both must satisfy differential privacy (DP). Such tables are **marginal tables**, and they are the most common batch. Each one counts the records for every combination of values of a few attributes, and each count is a **cell**.

The idea that makes these releases work is to add noise to whole tables rather than to single cells, scaled to how much one person can change the table. One person sits in one cell of a table, so a whole table costs about as much as one count. Everything computed from the noisy tables afterwards is free, including repairs that make the cells of a table add up to its total. When the queries are too many for noise on every cell, synthetic data fitted to a few noisy tables takes over.

## Which method to use {#sec-qr-choose}

The choice depends on how many queries you have, compared with the number of records, and on how many attributes a record has (@tbl-qr-choose).

| Situation | Use |
| --- | --- |
| A few marginal tables, with a public list of cells | Noisy counts on every cell, one budget share per table |
| Hundreds of tables or thousands of cells | Gaussian noise on every cell, scaled to the whole set of tables |
| A table whose list of values is not public | Noisy counts, keeping only cells that clear a noisy threshold ([Release statistics with a library](12-release-statistics.md)) |
| Far more queries than records, or queries not known in advance | Synthetic data from AIM or MST |
| Queries that arrive one at a time, perhaps adaptively | Online private multiplicative weights |

: Which query-release method to use. {#tbl-qr-choose}

Where "a few" tables end is the dense-query rule of [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md#sec-lg-choose). One person sits in every table, so at a privacy level $\varepsilon$ of 1 and a slack $\delta$ of $10^{-5}$, Gaussian noise wins from 7 tables on.

## Noise on every cell {#sec-qr-baseline}

Under add/remove neighbours, a person with one record changes one cell of a marginal table by 1, so the whole table has sensitivity 1, whatever its number of cells. Laplace noise of scale $1/\varepsilon$ on every cell then makes the table $\varepsilon$-DP, by the parallel composition of [Post-processing, group privacy and composition](02-properties.md#sec-dpprop-parallel). Under replace neighbours a person can move from one cell to another, so two cells change and the sensitivity doubles.

Several tables of the same people compose, because one person sits in one cell of every table. With Laplace noise the budget is split among the tables, so the error per cell grows in proportion to their number. Gaussian noise is calibrated to the square root of the sum of squared changes. For one person in a number $m$ of tables that is $\sqrt m$, so its error grows only like the square root of the number of tables.

An example from the full edition shows the gap. A table of 10,000 records has eight yes/no attributes. The release is every cell of the marginal table of every set of attributes, 6,560 cells in all, each as a fraction of the records, under replace neighbours. A uniform guess, which ignores the data, has a largest error of 0.232 over these cells. At $\varepsilon = 1$, Laplace noise on every cell has a largest error of 0.497, which is worse than ignoring the data. Gaussian noise does much better, with 0.037, at a slack $\delta$ of $10^{-6}$, stricter than the book's default of $10^{-5}$ for 10,000 records.

Two mistakes are common. Scaling each cell's noise to its own sensitivity, as if it were the only query, silently spends $\varepsilon$ once for every table, since each person sits in one cell of each. And splitting $\varepsilon$ evenly over thousands of separately noised cells with basic composition wastes most of the budget. Calibrate to the whole set of tables instead.

## Releasing a marginal with OpenDP {#sec-qr-opendp}

The script [practitioner_adult_statistics.py](../../code/part03-dp-foundations/practitioner_adult_statistics.py) releases the education-by-income table of the Adult census data, 48,842 people with one row each, at $\varepsilon = 1$. The cells come from the survey's documentation, which lists 16 levels of education and two income classes. So the keys are public, and the table has 32 cells, including any that happen to be empty, since omitting one would reveal that nobody has those values. A missing value needs a cell too: map it to a listed key such as "Unknown", because a row whose key is not in the list is dropped. The `Margin` (an OpenDP setting, not a marginal table) with `invariant="keys"` tells OpenDP that the list of cells is public, and `with_keys` pins the release to exactly those cells.

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

Run it with `uv run python code/part03-dp-foundations/practitioner_adult_statistics.py`. The summary is deterministic.

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

Each cell gets integer Laplace noise of scale 1 ([Discrete and floating-point-safe mechanisms](08-discrete-mechanisms.md)). The 95% accuracy of 3.38 means each noisy count lands that close to the truth 95% of the time. Print that error bar beside every cell, and remember that it covers the noise only, not sampling error or mistakes in the data. OpenDP's noise has no seed, so the released counts change on every run, and these four cells come from one run, beside the true counts.

```text
  Doctorate >50K: 432  true 431  accuracy 3.38
  HS-grad <=50K: 13281  true 13281  accuracy 3.38
  Preschool <=50K: 82  true 82  accuracy 3.38
  Preschool >50K: 0  true 1  accuracy 3.38
```

The large cells barely move, but the smallest true cell, one person with a preschool education and a high income, is smaller than the noise. If small cells matter, coarsen the table by merging rare categories in advance, because noise of a fixed size is large only next to small counts. With `signed=True` a count may come out negative, which keeps it unbiased. Clamping it at zero, rounding, and scaling the cells so that each table adds up to a total already released are all post-processing, so they cost nothing.

Three things go wrong. Taking the list of cells from the data, as a `unique()` on the private column does, reveals rare values by itself. When no codebook exists, use partition selection instead ([Release statistics with a library](12-release-statistics.md)). A second table of the same people needs its own share of the budget, so set `split_evenly_over` to the number of tables. And for many tables, give the budget as `dp.loss_of(rho=...)`, with the $\rho$ that converts to your $(\varepsilon, \delta)$, which [Release statistics with a library](12-release-statistics.md#sec-relstat-context) finds with OpenDP's own conversion. OpenDP then accounts in zero-concentrated differential privacy (zCDP) and uses Gaussian noise, which composes far better there than Laplace noise ([Concentrated differential privacy](../part04-privacy-accounting/03-concentrated-dp.md)).

## Synthetic data {#sec-qr-synth}

When the queries far outnumber what noise on every cell can bear, or nobody knows the queries in advance, release synthetic data instead. The practical tools follow four steps.

1. **Select** a set of marginal tables to measure, from domain knowledge or privately with the [exponential mechanism](05-selection.md#sec-sel-exp), scoring each table by how badly the current model fits it.
2. **Measure** each whole table with Gaussian noise, scaled to the whole set.
3. **Generate** a compact graphical model that fits the noisy tables.
4. **Sample** synthetic rows from the model.

Everything after the measurements is post-processing, so the synthetic rows can be queried and shared at no further privacy cost. MST and AIM, both by McKenna and coauthors, are the standard methods. AIM, an adaptive and iterative mechanism, chooses which tables to measure as it goes. SmartNoise Synth, the `snsynth` package, wraps both behind one interface. It is not installed in this book's environment, so nothing on this page produces synthetic data, and the call below was not run. To use it, install it in an environment of its own.

```text
# smartnoise-synth 1.0.8: not installed in this book's environment, not run
from snsynth import Synthesizer

synth = Synthesizer.create("aim", epsilon=1.0, delta=1e-6)
synth.fit(adult_pandas, categorical_columns=["education", "sex", "race", "occupation", "income"])
rows = synth.sample(48_842)
```

A synthetic table is accurate on the tables it measured, by construction, so evaluate it on other queries, but not freely against the private data. Keeping the best of several runs, judged against the private data, is a selection on that data, and each extra run of the generator spends the generator's budget again. So score candidate settings on a public dataset of the same shape, or charge the comparisons to the budget. A generator trained without DP is not anonymised, however synthetic its rows look.

## Private multiplicative weights {#sec-qr-pmw}

The **multiplicative weights exponential mechanism (MWEM)** of Hardt, Ligett and McSherry (2012) is the practical form of private multiplicative weights, and these tools keep its select, measure and update loop. It keeps a guessed histogram with one weight for every possible record, 256 in the example above, starting from uniform, and runs a fixed number of rounds. Each round selects a query that the guess answers badly, with the exponential mechanism. It then measures that query with fresh Laplace noise, and moves the guess toward the measurement with one multiplicative-weights step. An online-learning bound caps the number of corrections that can ever be needed, whatever the number of queries. So the budget pays for corrections rather than for queries, and the error grows only with the logarithm of the number of queries.

That pays off only for very large sets of queries. On the same example at $\varepsilon = 1$, with 38 rounds chosen as the next paragraph says, its largest error is 0.078. That beats Laplace noise but loses to Gaussian noise by a factor of about two. For arbitrary queries, unlike marginals, over $2^{20}$ possible records, its bound on the records needed beats Gaussian noise only near $10^9$ queries. Its running time also grows with the number of possible records, which limits it to a few attributes.

Three rules make an implementation correct. Fix the number of rounds before seeing the data, by minimizing the error bound, which uses only public quantities. Measure with fresh noise, because the selection step's scores are exact and never released, so reusing one would use the data without paying for it. And release the average of the guesses, not the last one, because the accuracy guarantee covers only the average.

## Online release and SmallDB {#sec-qr-online}

A query server that receives queries one at a time, perhaps chosen after earlier answers, can run the online form (Hardt & Rothblum 2010). It answers from the guess while the [sparse vector technique](05-selection.md#sec-sel-svt) finds the guess's error below a threshold, and pays only for updates. It is mainly a theoretical guarantee, used in [Adaptive data analysis](../part05-dp-ml/11-adaptive-data-analysis.md). SmallDB (Blum, Ligett & Roth 2008) picks a whole small dataset with one exponential mechanism. On the same example it would weigh about $10^{4568}$ candidate datasets, so it shows what is possible rather than what you can run.

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
