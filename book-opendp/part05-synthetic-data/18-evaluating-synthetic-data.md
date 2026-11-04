# Evaluating and tuning synthetic data

[Synthetic data with AIM and MST](17-aim-and-mst.md) produced a synthetic table and privacy statement. The statement describes privacy, not data quality. This chapter evaluates quality in three ways: compare measured marginals with the real data, test an unmeasured table, and train a classifier on synthetic rows and test it on real rows. It repeats these checks at three budgets and with MST, then lists what to publish with the data.

The documentation ends after generating rows. This chapter adds evaluation methods from the AIM paper and a benchmark of synthetic-data generators. The Labour Force Survey (LFS) sample is public, so it can be compared with its source. For private data, use a public or historical stand-in, never the data being protected.

The script [opendp_evaluating_synthetic_data.py](../../code/part05-synthetic-data/opendp_evaluating_synthetic_data.py) runs on OpenDP 0.16.0 and `mbi` 1.1.0, reusing functions from the previous chapter's script. Budgets, scales, thresholds, conversions to $\varepsilon$ and the classifier trained on real data are deterministic. OpenDP's noise cannot be seeded, so releases, fitted models and distances vary; the page labels them as one run. Four releases took 55 seconds.

```bash
uv run python code/part05-synthetic-data/opendp_evaluating_synthetic_data.py
```

## What good means {#sec-eval-good}

The benchmark by Tao and coauthors evaluates synthetic data with three questions. Do measured marginals match the real ones? Do unmeasured queries, such as a count over all four columns or a share in a small group, remain accurate? Does a model trained on synthetic rows work on real rows? Each test can reveal failures missed by the others.

Only measured marginals have release-based error bars. Their known Gaussian scales let the table's `std` method estimate how far fitted counts may be from truth ([Contingency tables](16-contingency-tables.md)). Unmeasured tables have no such bar, and neither do downstream models. Evaluate the second and third questions against real data. For private releases, use public or historical data of the same kind ([the full edition's chapter on public data](../../book-full/part05-dp-ml/06-public-data.md)).

The evaluation uses the previous chapter's AIM release with nulls filled. It uses the same 36-contribution privacy unit ([The data and the context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context)), four columns, keys and cuts, and ten rounds. The budget is a privacy parameter $\rho$ of 0.1 under zero-concentrated differential privacy (zCDP), with Gaussian noise; a slack $\delta$ of $10^{-7}$ pays for selecting `ILOSTAT` keys. Two functions build the context and query so later sections can vary the budget.

```{.python excerpt="eval-context"}
def make_context(frame: pl.LazyFrame, rho: float) -> dp.Context:
    """A context for one release at budget rho, with the previous chapter's unit and slack."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),  # neighbours add or remove one person's 36 rows
        privacy_loss=dp.loss_of(rho=rho, delta=DELTA),  # zCDP with Gaussian noise; delta pays for ILOSTAT's keys
    )


def table_query(context: dp.Context, rho: float, algorithm: Algorithm) -> ContingencyTableQuery:
    """The previous chapter's query with its nulls filled, at the budget rho, with the algorithm plugged in."""
    return (
        context.query(rho=rho, delta=DELTA)
        .with_columns(synth.fill_nulls())  # every null lands in a public key before anything is counted
        .select(*COLUMNS)
        .contingency_table(keys=synth.KEYS, cuts=synth.CUTS, algorithm=algorithm)
    )
```

The first line releases the table. The next applies the same preparation to real rows: fill nulls, bin numeric values and map codes outside the keys to null. Thus real and synthetic marginals have identical cells.

```{.python excerpt="eval-release"}
aim_query = table_query(make_context(frame, RHO), RHO, dp.mbi.AIM(rounds=ROUNDS))
table_aim, seconds_aim = synth.timed_release(aim_query)
real = synth.binned(frame, table_aim.keys, fill=True).collect()  # the real rows in the table's keys and bins
```

```text
privacy unit: (SymmetricDistance(), 36)
privacy loss: (Approximate(ZeroConcentratedDivergence), (0.1, 1e-07))
aim rho=0.1: oneway scale=227.6840 threshold=1364  scale of a round's marginal=379.4733  epsilon at delta=1e-07: 2.3484
rows: 3,811,932  keys ILOSTAT: [1, 2, 3, 9, None]
aim rho=0.1 release (one run): 13.7 s
aim rho=0.1 measured: ('AGE', 'HWUSUAL', 'ILOSTAT') std=379.47
aim rho=0.1 measured: ('SEX', 'HWUSUAL', 'ILOSTAT') std=379.47
aim rho=0.1 measured: ('SEX', 'AGE', 'HWUSUAL') std=379.47
aim rho=0.1 measured: ('SEX', 'AGE', 'ILOSTAT') std=379.47
aim rho=0.1 measured: ('SEX', 'ILOSTAT') std=189.74
```

The query and scales match the previous chapter: 227.68 for one-way marginals and 379.47 for each round's marginal. At a further $\delta$ of $10^{-7}$, the budget converts to $\varepsilon = 2.35$. In this run, AIM measured all four three-way marginals and then a pair at half the scale.

## The workload error {#sec-eval-workload}

The first test measures each released marginal's distance from the real one. The AIM paper calls the workload-average distance the **workload error**, which AIM seeks to minimize. This is total variation distance: half the sum of absolute differences between cell shares. Equivalently, it is the largest share difference assigned to any set of cells. The worst distance below, 0.00102, bounds the share error for every query over those columns.

The function computes this distance for a list of cliques, the column sets covered by the marginals. It reads model counts with `project_melted` rather than drawing rows, which would add sampling noise (@sec-eval-rows). It also reports the previous chapter's bound: the distance if every marginal cell were three standard deviations from truth.

```{.python excerpt="eval-errors"}
def marginal_errors(
    table: ContingencyTable, real: pl.DataFrame, cliques: Sequence[tuple[str, ...]]
) -> list[MarginalError]:
    """Each clique's distance between the fitted model's marginal and the real one, with the table's own standard
    deviation for it and the bound that three standard deviations in every cell would give."""
    errors: list[MarginalError] = []
    for clique in cliques:
        fitted = table.project_melted(clique)  # the model's counts: no rows are drawn
        tv, worst = distance(synth.marginal(real, clique), fitted, clique)
        std = table.std(clique)
        bound = synth.tv_bound(synth.cells(table, clique), std, real.height)
        errors.append(MarginalError(clique, tv, worst, std, bound))
    return errors
```

| Marginal | Distance | Largest cell error (rows) | `std` | Bound |
| --- | --- | --- | --- | --- |
| `SEX` | 0.00004 | 158 | 196.2 | 0.00023 |
| `AGE` | 0.00005 | 146 | 219.9 | 0.00035 |
| `HWUSUAL` | 0.00006 | 246 | 219.9 | 0.00035 |
| `ILOSTAT` | 0.00013 | 307 | 201.0 | 0.00040 |
| (`SEX`, `ILOSTAT`) | 0.00022 | 425 | 178.9 | 0.00106 |
| (`AGE`, `HWUSUAL`, `ILOSTAT`) | 0.00102 | 509 | 379.5 | 0.01195 |
| (`SEX`, `AGE`, `HWUSUAL`) | 0.00081 | 503 | 379.5 | 0.00717 |
| (`SEX`, `AGE`, `ILOSTAT`) | 0.00072 | 568 | 379.5 | 0.00896 |
| (`SEX`, `HWUSUAL`, `ILOSTAT`) | 0.00065 | 470 | 379.5 | 0.00896 |

: The workload error of the AIM release at ρ = 0.1 in one run: the total variation distance between each one-way and each measured marginal of the model and the real one, the largest cell error, the table's own standard deviation, and the bound from it. {#tbl-eval-workload}

Every distance is at least three times below its bound, and no cell differs by more than a few hundred of 3.8 million rows. One-way standard deviations are smaller than the 227.68 one-way scale because the model combines each one-way measurement with all measured marginals containing that column. Thus `ILOSTAT`, which appears in four marginals, has standard deviation 201.0.

Use this table to check the pipeline as well as the noise. A distance above its bound signals a systematic problem, such as the null handling in the previous chapter. Private data cannot supply the true marginals needed for these distances. The release does supply the `std` column, which provides the publishable error bars.

## A query nobody measured {#sec-eval-unmeasured}

The second test concerns an unmeasured table. AIM considers marginals of at most three columns, so it never measures the 240-cell table over `SEX`, `AGE`, `HWUSUAL` and `ILOSTAT`. The model can still answer the query using a distribution that matches measured marginals and assumes no further relationships ([the full edition's chapter on query release](../../book-full/part03-dp-foundations/06-query-release.md)). But the release provides no error bar for this answer.

```{.python excerpt="eval-unmeasured"}
try:
    four_std = f"{table_aim.std(COLUMNS):.1f}"
except ValueError as err:
    four_std = str(err)  # no measured marginal covers all four columns, so the table has no error bar for it
```

```text
four-way cells: 240  measured by AIM: False
std of the four-way table: attrs (('SEX', 'AGE', 'HWUSUAL', 'ILOSTAT')) are not covered by the query set
four-way (aim rho=0.1, one run): tv=0.00141 largest cell error=450 rows  best measured (SEX, ILOSTAT) tv=0.00022  worst measured tv=0.00102
cells of the share with at least 1,000 real rows: 26
share working by SEX, AGE, HWUSUAL (aim rho=0.1, one run): largest error=2.05 points at {'SEX': 1, 'AGE': '(-inf, 20.0]', 'HWUSUAL': '(1.0, 20.0]'}  real=100.00%  model=97.95%
```

Here the four-way table is farther from truth than any measured marginal. Its error reflects noise in the measured triples and four-column dependence that no triple captures. The four measured triples overlap heavily, leaving little for the model to infer; therefore, the four-way distance stays near the worst measured distance. A release with fewer marginals relies more on model assumptions, as the MST results in @sec-eval-tuning show.

Conditional shares are harder to estimate because they divide one count by another. From the four-way table, the script computes the employment share in 26 combinations of `SEX`, `AGE` band and `HWUSUAL` band with at least a thousand real rows. The largest error here is two percentage points for men under 20 who usually work 1 to 20 hours. All people in this cell work, but the model estimates a share of 97.95%. Noise is added to counts, not individuals, so small groups have larger relative errors. If analysts need small-group estimates, measure those groups, for example with `Fixed` inside `Sequential`.

## Train on synthetic, test on real {#sec-eval-downstream}

The third test asks whether a model trained on synthetic rows performs on real rows, the benchmark's "train on synthetic, test on real" task. The classifier predicts whether `ILOSTAT` equals 1 from `SEX`, `AGE` band and `HWUSUAL` band. It is a scikit-learn logistic regression with eleven indicator columns, one for each feature key.

The script draws 200,000 synthetic rows independently with `method="sample"`. It also shuffles 400,000 real rows with a fixed seed. The first 200,000 train a real-data baseline; the remaining rows test both models.

```{.python excerpt="eval-synthesize"}
rows = synth.synthesize(table, seed, rows=ROWS, method="sample")  # independent draws from the model
rows = synth.binned(rows.lazy(), table.keys, fill=False).collect()  # the numbers back in their bins
model = classifier(rows, table.keys, seed)  # train on synthetic
```

```{.python excerpt="eval-classifier"}
def classifier(train: pl.DataFrame, keys: dict[str, Any], seed: int) -> LogisticRegression:
    """A logistic regression that predicts ILOSTAT == 1 from SEX, the AGE band and the HWUSUAL band, fitted on
    `train`, which is either synthetic rows or real rows."""
    return LogisticRegression(max_iter=1000, random_state=seed).fit(design(train, keys), labels(train))


def accuracy(model: LogisticRegression, test: pl.DataFrame, keys: dict[str, Any]) -> float:
    """The share of the held-out real rows the model labels correctly."""
    return float((model.predict(design(test, keys)) == labels(test)).mean())
```

```text
held-out real rows: 200,000  real training rows: 200,000  synthetic training rows: 200,000
features: 11 indicators
share working in the held-out rows: 0.3976
train on real, test on real: accuracy=0.9313
standard error of an accuracy on 200,000 rows: 0.0006
train on synthetic (aim rho=0.1, one run), test on real: accuracy=0.9313  gap=+0.0000  mean probability difference=0.0015
```

The real-data model classifies 93.13% of held-out rows correctly, while 39.76% of those rows are working. In this run, the synthetic-data model has the same accuracy, so the gap is zero and well within the 0.0006 standard error. The test allows a one-percentage-point gap, which could result from errors in a few small cells.

Accuracy is coarse: models make identical predictions whenever each cell's probability remains on the same side of one half. The script therefore also reports the models' mean predicted-probability difference, 0.0015 in this run. For a single model, direct private fitting ([Statistical modelling](../part04-extending/14-statistical-modelling.md)) is often preferable to training on synthetic data.

## What the budget and the algorithm buy {#sec-eval-tuning}

Tuning compares evaluation results across settings. The script runs AIM with one-fifth the budget, $\rho = 0.02$, and five times the budget, $\rho = 0.5$, then runs MST at the original budget. Each release has its own context because only one would be published. If all four used the same private data, they would compose to $\rho = 0.72$. Tuning on private data therefore spends budget; tune on a public stand-in and release once.

```{.python excerpt="eval-mst"}
mst_query = table_query(make_context(frame, RHO), RHO, dp.mbi.MST())
table_mst, seconds_mst = synth.timed_release(mst_query)
mst_eval = evaluate("mst rho=0.1", table_mst, seconds_mst, real, test_real, baseline, seed)
```

```text
aim rho=0.02: oneway scale=509.1169 threshold=3004  scale of a round's marginal=848.5281  epsilon at delta=1e-07: 0.9951
aim rho=0.5: oneway scale=101.8234 threshold=630  scale of a round's marginal=169.7056  epsilon at delta=1e-07: 5.6709
mst rho=0.1: oneway scale=227.6840 threshold=1364  scale of a pair=207.8461  epsilon at delta=1e-07: 2.3484
mst rho=0.1 measured: ('HWUSUAL', 'ILOSTAT') std=207.85
mst rho=0.1 measured: ('AGE', 'ILOSTAT') std=207.85
mst rho=0.1 measured: ('SEX', 'ILOSTAT') std=207.85
share working by SEX, AGE, HWUSUAL (mst rho=0.1, one run): largest error=24.03 points at {'SEX': 2, 'AGE': '(-inf, 20.0]', 'HWUSUAL': '(-inf, 1.0]'}  real=100.00%  model=75.97%
budget the four contexts add up to if all were released from the same data: rho=0.72
```

The previous chapter's scale formula uses sensitivity 36 divided by the square root of twice the budget. A 25-fold budget increase therefore divides each scale by five; one-way scales fall from 509.12 to 101.82. The smallest budget converts to $\varepsilon$ of 1.00 and the largest to 5.67 ([the full edition's chapter on choosing epsilon](../../book-full/part05-dp-ml/09-choosing-epsilon.md) discusses defensible choices).

| Release | $\varepsilon$ | Seconds | Measured | One-way mean | Worst measured | Four-way | Share error (points) | Probability difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AIM, $\rho = 0.02$ | 1.00 | 6.4 | 4 | 0.00016 | 0.00157 | 0.00252 | 15.64 | 0.0016 |
| AIM, $\rho = 0.1$ | 2.35 | 13.7 | 5 | 0.00007 | 0.00102 | 0.00141 | 2.05 | 0.0015 |
| AIM, $\rho = 0.5$ | 5.67 | 7.1 | 6 | 0.00005 | 0.00052 | 0.00077 | 1.65 | 0.0014 |
| MST, $\rho = 0.1$ | 2.35 | 3.3 | 3 | 0.00021 | 0.00114 | 0.05610 | 24.03 | 0.0280 |

: The same evaluation at three budgets with AIM and at one with MST, in one run: ε at δ = 10⁻⁷, the seconds the release took, the number of marginals measured, the total variation distances of the one-way marginals (mean), of the worst measured marginal and of the four-way table, the largest error of the share working, and the mean difference in predicted probability from the model trained on real rows. {#tbl-eval-tuning}

Larger budgets improve accuracy, but not uniformly. The worst measured marginal and four-way table improve across all three budgets, though less than the scales do. One-way marginals, already accurate at $\rho = 0.1$, improve least. Share error falls most at the low end, from 15.64 to 2.05 points, because noise affects small groups first. With more budget, AIM also measures more marginals: four, five and six in this run. All four classifiers have accuracy 0.9313, so budget changes are not visible in this task.

The algorithm also matters. Here MST measures three pairs, each containing `ILOSTAT`. It therefore treats `SEX`, `AGE` and `HWUSUAL` as independent given employment status. This is false: among working people, hours depend on sex and age. As a result, MST's four-way table is tens of times farther from truth than AIM's at the same budget.

MST's share error reaches 24 points for women under 20 who report at most one usual hour. Predicted probabilities differ by 0.028 on average, although accuracy is unchanged. Choose MST when pairwise queries suffice and release speed matters: it took 3.3 seconds, compared with 13.7 for AIM in this run.

The other settings are bin cuts and number of rounds. Finer cuts increase the number of cells in every marginal without changing noise scale, raising each marginal's error bound. More rounds divide the budget into smaller shares, allowing AIM to measure more marginals at lower precision.

## How many rows, and how to draw them {#sec-eval-rows}

The marginal evaluation reads the model rather than synthetic rows. Drawing a fixed number of independent rows adds sampling error to model error. The script estimates this error for each one-way marginal using the table's shares, for both 200,000 rows and the real dataset's size.

```text
sampling noise on one-way SEX: expected tv at 200,000 rows=0.00090  at 3,811,932 rows=0.00021
sampling noise on one-way AGE: expected tv at 200,000 rows=0.00152  at 3,811,932 rows=0.00035
sampling noise on one-way HWUSUAL: expected tv at 200,000 rows=0.00100  at 3,811,932 rows=0.00023
sampling noise on one-way ILOSTAT: expected tv at 200,000 rows=0.00140  at 3,811,932 rows=0.00032
aim rho=0.1 one-way tv summed (one run): model=0.00029  round=0.00028  sample=0.00745
aim rho=0.1 four-way tv (one run): model=0.00141  round=0.00139  sample=0.00483
mst rho=0.1 four-way tv (one run): model=0.05610  round=0.18410  sample=0.05791
mst rho=0.1 pair ('AGE', 'HWUSUAL') tv (one run): model=0.01504  round=0.12370  sample=0.01261
```

With 200,000 sampled rows, `SEX` sampling error is many times larger than the model error in @tbl-eval-workload, so the sample is less accurate than the release. Sampling error still exceeds model error at the real dataset's size because privacy noise at this budget is small relative to 3.8 million rows.

The default `method="round"` avoids sampling error; in this run, AIM's rounded marginals match the model. In `mbi` 1.1.0, however, each column is drawn by a row's rank within its parent-column group. Columns with the same parents are therefore sorted together and become dependent. In MST's star, `AGE` and `HWUSUAL` share parents; rounding changes their marginal distance from 0.01504 to 0.12370, while sampling keeps it near the model. Use sampling when joint row distributions matter. If marginal accuracy matters, draw as many rows as the model's total. Check pairwise marginals before trusting rounded rows.

## What to publish {#sec-eval-publish}

A synthetic dataset needs a guarantee statement and enough information for users to judge what it supports. [The practitioner edition's chapter on release and governance](../../book-practitioner/part05-dp-ml/12-release-governance.md) gives the same guidance for trained models. The script supplies the numbers and identifies public information.

```{.python excerpt="eval-statement"}
print(
    f"This synthetic dataset of {ROWS:,} rows was generated by AIM (OpenDP {dp.__version__}, mbi "
    f"{importlib.metadata.version('mbi')}, rounds={ROUNDS}, queries up to 3 columns) from one contingency table "
    f"over {', '.join(COLUMNS)} released under (rho = {RHO:g}, delta = {DELTA:g})-zero-concentrated differential "
    f"privacy, which is (epsilon = {epsilon:.2f}, delta = {2 * DELTA:g})-differential privacy, for adding or "
    f"removing all of one person's survey responses, at most {CONTRIBUTIONS} of them, under the central model. "
    f"The keys of SEX and the cuts of AGE and HWUSUAL were fixed in advance; nulls were recoded to "
    f"{synth.FILL}; the keys of ILOSTAT were released with a threshold of {aim_query.oneway_threshold}. "
    f"Drawing the rows by independent sampling is post-processing and spent nothing."
)
```

```text
This synthetic dataset of 200,000 rows was generated by AIM (OpenDP 0.16.0, mbi 1.1.0, rounds=10, queries up to 3 columns) from one contingency table over SEX, AGE, HWUSUAL, ILOSTAT released under (rho = 0.1, delta = 1e-07)-zero-concentrated differential privacy, which is (epsilon = 2.35, delta = 2e-07)-differential privacy, for adding or removing all of one person's survey responses, at most 36 of them, under the central model. The keys of SEX and the cuts of AGE and HWUSUAL were fixed in advance; nulls were recoded to {'AGE': 0.0, 'HWUSUAL': 99.0}; the keys of ILOSTAT were released with a threshold of 1364. Drawing the rows by independent sampling is post-processing and spent nothing.
```

The statement covers the generated rows without an extra privacy cost because drawing from a released model is post-processing ([the full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md)). It covers this table only; other releases from the same people would compose with it.

| Item | What to publish | In this chapter |
| --- | --- | --- |
| The guarantee | $\rho$ and $\delta$, their conversion to $\varepsilon$, the privacy unit and the trust model, printed from the variables | $\rho = 0.1$, $\varepsilon = 2.35$, 36 contributions, central model |
| The columns | Each column with its keys or cuts, the null recoding, and which keys were released | `SEX` keys, `AGE` and `HWUSUAL` cuts, `ILOSTAT` keys at threshold 1364 |
| The algorithm | Its name and parameters, the library versions, and how the rows were drawn | AIM with 10 rounds, OpenDP 0.16.0, `mbi` 1.1.0, 200,000 sampled rows |
| The error bars | The measured marginals and their `std`, which come from the release and cost nothing | @tbl-eval-workload, the `std` column |
| The utility | The marginal errors and the downstream check, measured on public or historical data | Never on the private data itself |

: What to publish with a synthetic dataset, and where this chapter found each item. {#tbl-eval-publish}

The utility row is easy to mishandle. A distance computed from private data depends on that data; publishing it creates a release not covered by this statement. As [Limitations and troubleshooting](../part04-extending/15-limitations-and-troubleshooting.md) explains, the guarantee covers measured outputs, not separate computations. The `std` error bars are safe to publish. The distances here are safe only because the dataset is public.

This final chapter completes the workflow: release statistics, tables, models and synthetic rows through a context; evaluate quality on data you may inspect; and publish each release with a precise guarantee.

## Summary

| Call or concept | What it does | Rule of thumb |
| --- | --- | --- |
| Three questions | Measured marginals, unmeasured queries, a downstream model | Only the first has an error bar from the release |
| `table.project_melted(clique)` | The fitted model's counts for a marginal, without drawing rows | Evaluate the model first, then the rows |
| Total variation distance against `std` | Workload error of a marginal against three standard deviations in every cell | A distance over the bound points to the pipeline, not the noise |
| `table.std` on an uncovered table | Raises `ValueError`: not covered by the query set | No measurement, no error bar; check unmeasured queries on public data |
| Train on synthetic, test on real | A model fitted to synthetic rows, scored on held-out real rows | Accuracy is coarse; compare predicted probabilities too |
| One context per setting | Alternatives that you compare but release only one of | On the same private data their budgets add up; tune on a stand-in |
| The budget | Divides every scale by the square root of its ratio | Small groups and unmeasured tables gain most; one-way marginals least |
| AIM against MST | Triples chosen adaptively, or a tree of pairs | MST is fast but loses what no pair shows |
| `synthesize(rows, method="sample")` | Independent rows, with sampling error of their own | Draw as many rows as the model's total when the marginals must be close |
| `method="round"` in `mbi` 1.1.0 | Matches the model's marginals but ties columns drawn from the same parents | Check the pairs before trusting rounded rows |
| What to publish | Guarantee, columns, algorithm, error bars, utility on public data | Never publish a distance computed on the private data |

: Evaluating and tuning synthetic data, what each step does, and the rule of thumb that goes with it. {#tbl-eval-summary}

## References

- OpenDP documentation: "Synthetic Data" in the statistical modelling guide, "Contingency Tables" in the tabular data guide, and the API reference for `opendp.extras.mbi.ContingencyTable` (`std`, `project_melted`, `synthesize`). The documentation gives no evaluation; this chapter's is its own.
- McKenna, Mullins, Sheldon and Miklau, "AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data" (2022): the workload error and the selection that minimizes it.
- Tao, McKenna, Hay, Machanavajjhala and Miklau, "Benchmarking Differentially Private Synthetic Data Generation Algorithms" (2021): marginal errors and downstream tasks as the measures of quality.
- The practitioner edition's chapter on [release and governance](../../book-practitioner/part05-dp-ml/12-release-governance.md): the statement of a guarantee and the evidence that goes with it.
- The full edition's chapters on [post-processing and composition](../../book-full/part03-dp-foundations/02-properties.md), [query release](../../book-full/part03-dp-foundations/06-query-release.md), [public data](../../book-full/part05-dp-ml/06-public-data.md) and [choosing epsilon](../../book-full/part05-dp-ml/09-choosing-epsilon.md) derive the results this chapter states.
