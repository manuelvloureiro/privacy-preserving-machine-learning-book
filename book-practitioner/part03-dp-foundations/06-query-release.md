# Query release and private multiplicative weights

## Why this matters

Analysts rarely want one statistic. They ask for every two-way table over a handful of columns, or for a synthetic file they can query as often as they like, and both must satisfy differential privacy (DP). Such tables are **marginal tables**: each counts the records for every combination of values of a few attributes, and each count is a **cell**.

These releases work by adding noise to whole tables rather than to single cells, scaled to how much one person can change the table. One person sits in one cell of a table, so a whole table costs about as much as one count. Everything computed from the noisy tables afterwards is free, including repairs that make the cells add up to the total. When the queries are too many for noise on every cell, synthetic data fitted to a few noisy tables takes over.

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

Where "a few" tables end follows from the dense-query rule of [the Laplace and Gaussian mechanisms](03-laplace-gaussian.md#sec-lg-choose). One person sits in every table, so the tables form a dense query, and at a privacy level $\varepsilon$ of 1 and a slack $\delta$ of $10^{-5}$, Gaussian noise wins from 7 tables on.

## Noise on every cell {#sec-qr-baseline}

Under add/remove neighbours, a person with one record changes one cell of a marginal table by 1, so the whole table has sensitivity 1, whatever its number of cells. Laplace noise of scale $1/\varepsilon$ on every cell then makes the table $\varepsilon$-DP, by [parallel composition](02-properties.md#sec-dpprop-parallel). Under replace neighbours a person can move from one cell to another, so two cells change and the sensitivity doubles.

Several tables of the same people do compose, because one person sits in one cell of every table. With Laplace noise the budget is split among the tables, so the error per cell grows in proportion to their number. Gaussian noise is calibrated to the square root of the sum of squared changes, which for one person in $m$ tables is $\sqrt m$, so its error grows only like $\sqrt m$.

An example from the full edition shows the gap. A table of 10,000 records has eight yes/no attributes, so $2^8 = 256$ possible records. The release is every cell of the marginal table of every set of attributes, 6,560 cells in all, each as a fraction of the records, under replace neighbours. A uniform guess, which ignores the data, has a largest error of 0.232 over these cells. At $\varepsilon = 1$, Laplace noise on every cell does worse than that, with 0.497. Gaussian noise does much better, with 0.037, at a slack $\delta$ of $10^{-6}$, stricter than the book's default.

Two mistakes are common. Scaling each cell's noise to its own sensitivity, as if it were the only query, silently spends $\varepsilon$ once for every table. And splitting $\varepsilon$ evenly over thousands of separately noised cells with basic composition wastes most of the budget. Calibrate to the whole set of tables instead.

## Releasing a marginal with OpenDP {#sec-qr-opendp}

The script [practitioner_adult_statistics.py](../../code/part03-dp-foundations/practitioner_adult_statistics.py) releases the education-by-income table of the Adult census data, 48,842 people with one row each, at $\varepsilon = 1$. The survey's documentation lists 16 levels of education and two income classes, so the keys are public and the table has 32 cells. Empty cells are released too, since omitting one would reveal that nobody has those values. Map a missing value to a listed key such as "Unknown", because a row whose key is not in the list is dropped. In OpenDP, a `Margin` (a setting, not a marginal table) with `invariant="keys"` declares the list of cells public, and `with_keys` pins the release to exactly those cells.

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

Each cell gets integer Laplace noise of scale 1 ([Discrete and floating-point-safe mechanisms](08-discrete-mechanisms.md)). The 95% accuracy of 3.38 means each noisy count lands that close to the truth 95% of the time. Print that error bar beside every cell, and remember that it covers the noise only, not sampling error or mistakes in the data. OpenDP's noise has no seed, so these four cells come from one run, beside the true counts.

```text
  Doctorate >50K: 432  true 431  accuracy 3.38
  HS-grad <=50K: 13281  true 13281  accuracy 3.38
  Preschool <=50K: 82  true 82  accuracy 3.38
  Preschool >50K: 0  true 1  accuracy 3.38
```

The large cells barely move, but the smallest true cell, one person with a preschool education and a high income, is smaller than the noise. Noise of a fixed size is large only next to small counts, so if small cells matter, merge rare categories in advance. With `signed=True` a count may come out negative, which keeps it unbiased. Clamping it at zero, rounding, and scaling the cells to add up to a total already released are post-processing, so they cost nothing.

Three things go wrong. Taking the list of cells from the data, as `unique()` on a private column does, reveals rare values by itself. Without a codebook, use partition selection ([Release statistics with a library](12-release-statistics.md)). A second table of the same people needs its own share of the budget, so set `split_evenly_over` to the number of tables. And for many tables, give the budget as `dp.loss_of(rho=...)`, with the $\rho$ that converts to your $(\varepsilon, \delta)$ ([Release statistics with a library](12-release-statistics.md#sec-relstat-context) shows how). OpenDP then uses Gaussian noise, which composes far better, and accounts in zero-concentrated differential privacy (zCDP) ([Concentrated differential privacy](../part04-privacy-accounting/03-concentrated-dp.md)).

## Synthetic data {#sec-qr-synth}

When the queries far outnumber what noise on every cell can bear, or nobody knows them in advance, release synthetic data instead. The practical tools follow four steps.

1. **Select** a set of marginal tables to measure, from domain knowledge or privately with the [exponential mechanism](05-selection.md#sec-sel-exp), scoring each table by how badly the current model fits it.
2. **Measure** each whole table with Gaussian noise, scaled to the whole set.
3. **Generate** a compact graphical model that fits the noisy tables.
4. **Sample** synthetic rows from the model.

Everything after the measurements is post-processing, so the synthetic rows can be queried and shared at no further privacy cost. The standard methods are MST and AIM, both by McKenna and coauthors, and AIM chooses which tables to measure as it goes. SmartNoise Synth, the `snsynth` package, wraps both behind one interface. It is not installed in this book's environment, so the call below was not run; install it in an environment of its own.

```text
# smartnoise-synth 1.0.8: not installed in this book's environment, not run
from snsynth import Synthesizer

synth = Synthesizer.create("aim", epsilon=1.0, delta=1e-6)
synth.fit(adult_pandas, categorical_columns=["education", "sex", "race", "occupation", "income"])
rows = synth.sample(48_842)
```

A synthetic table is accurate on the tables it measured, by construction, so evaluate it on other queries, but not freely against the private data. Keeping the best of several runs, judged against the private data, is a selection on that data, and each extra run spends the generator's budget again. So score candidate settings on a public dataset of the same shape, or charge the comparisons to the budget. And a generator trained without DP is not anonymised, however synthetic its rows look.

## Private multiplicative weights {#sec-qr-pmw}

The synthetic-data tools keep the select, measure and update loop of the **multiplicative weights exponential mechanism (MWEM)** of Hardt, Ligett and McSherry (2012). MWEM is the practical form of private multiplicative weights. It keeps a guessed histogram with one weight for every possible record, 256 in the eight-attribute example, starting from uniform, and runs a fixed number of rounds. Each round selects a query that the guess answers badly, with the exponential mechanism, measures it with fresh Laplace noise, and moves the guess toward the measurement. An online-learning bound caps the number of corrections ever needed, whatever the number of queries. So the budget pays for corrections rather than queries, and the error grows only with the logarithm of the number of queries.

That pays off only for very large sets of queries. On the eight-attribute example at privacy level 1, with 38 rounds, its largest error is 0.078. That beats Laplace noise but loses to Gaussian noise by a factor of about two. Its running time also grows with the number of possible records, which limits it to a few attributes.

Three rules make an implementation correct. Fix the number of rounds before seeing the data, by minimizing the error bound, which uses only public quantities. Measure with fresh noise, because the selection step's scores are exact and never released, so reusing one would use the data without paying for it. And release the average of the guesses, not the last one, because the accuracy guarantee covers only the average. For queries that arrive one at a time, an online form answers from the guess and pays only for its corrections ([Adaptive data analysis](../part05-dp-ml/11-adaptive-data-analysis.md)).

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
