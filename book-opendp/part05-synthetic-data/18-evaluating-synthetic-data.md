# Evaluating and tuning synthetic data

[Synthetic data with AIM and MST](17-aim-and-mst.md) ended with a synthetic table and a privacy statement. The statement says how little the table reveals about any one person, but it says nothing about whether the table is any good. This chapter measures that. The one idea is that "good" is three separate questions, and only the first has an answer that the release itself can give. You will compare an AIM release with the real data on the marginals it measured, on a table it never measured, and through a classifier trained on its rows. Then you will repeat the comparison at three budgets and for MST, and write down what to publish alongside the rows.

The documentation stops once the rows are generated, so this chapter goes beyond it. Its methods come from the paper that introduced AIM and from a benchmark of private synthetic-data generators, both cited at the end. The Labour Force Survey (LFS) sample is public, and that is what makes the comparison possible. On private data you would run the same comparison on a public or historical stand-in before the release, never on the data you protect.

The script [opendp_evaluating_synthetic_data.py](../../code/part05-synthetic-data/opendp_evaluating_synthetic_data.py) runs every step on OpenDP 0.16.0 with `mbi` 1.1.0, and it imports the previous chapter's script for the functions the two share. The budgets, noise scales, thresholds, conversions to the privacy parameter $\varepsilon$ and the classifier trained on real rows are the same on every run. The releases, the fitted models and every distance to the truth are not, because OpenDP's noise cannot be seeded. So the page shows them in blocks and tables marked as one run. The whole script, with four releases, took 55 seconds in one run.

```bash
uv run python code/part05-synthetic-data/opendp_evaluating_synthetic_data.py
```

## What good means {#sec-eval-good}

A synthetic table can be judged by three questions. The first is whether the marginals the algorithm measured match the real ones. The second is whether queries it never measured come out right, such as a count over all four columns or a share within a small group. The third is whether a model trained on the synthetic rows works on real rows. These are the checks of the benchmark by Tao and coauthors, and each one catches failures that the other two miss.

Only the first question has an error bar that comes from the release. Every measured marginal carried Gaussian noise of a known scale. So the table's `std` method can say how far its fitted counts should be from the truth, as [Contingency tables](16-contingency-tables.md) showed. A table that no measurement covers has no such bar, and `std` refuses it. A downstream model has no bar either. The second and third questions can therefore be answered only against real data. On a real release that means public or historical data of the same kind, which [the full edition's chapter on public data](../../book-full/part05-dp-ml/06-public-data.md) discusses.

The release under evaluation is the previous chapter's AIM release with its nulls filled. It uses the same privacy unit of 36 contributions, the same four columns with their keys and cuts, and ten rounds. The budget is $\rho = 0.1$ under zero-concentrated differential privacy (zCDP), where $\rho$ bounds the privacy loss on average and the noise is Gaussian. A slack $\delta$ of $10^{-7}$ pays for releasing the keys of `ILOSTAT`. Two functions build the context and the query from a budget, so that later sections can change it.

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

The release itself is one line, and the next one prepares the comparison. It puts the real rows through the same steps as the query, filling the nulls, cutting the numbers into bins and mapping codes outside the keys to null. So every real marginal has exactly the cells of the synthetic one.

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

The scales are the previous chapter's, since the query is the same. The one-way marginals carry noise of scale 227.68 and each round's marginal 379.47. Converted at a further $\delta$ of $10^{-7}$, the budget is $\varepsilon = 2.35$. In this run AIM measured all four three-way marginals and then a pair at half the scale.

## The workload error {#sec-eval-workload}

The first question asks how far each measured marginal of the model is from the real one. The AIM paper calls the average of that distance over a workload the **workload error**, and it is the number AIM itself tries to make small. The distance here is the total variation distance of the previous chapter, half the sum over cells of the absolute differences between the two tables' shares. It is the largest difference between the shares that the two tables give to any set of cells. So the worst distance below, 0.00102, means that no question about those columns gets a share that is off by more than that.

The function below computes it for a list of cliques, the sets of columns each marginal spans. It reads the model's own counts with `project_melted` rather than drawing rows, because drawing rows adds noise of its own, which @sec-eval-rows measures. Next to each distance it puts the bound of the previous chapter. That bound is the distance the marginal would have if every one of its cells were off by three of its standard deviations at once.

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

Every distance is under its bound by a factor of three or more. The worst measured marginal, the triple of `AGE`, `HWUSUAL` and `ILOSTAT`, is under it by a factor of 11.7. No cell is off by more than a few hundred rows out of 3.8 million. The one-way standard deviations are not the one-way scale of 227.68, because the model combines each one-way measurement with every measured marginal that contains the column. That is why `ILOSTAT`, which sits in four marginals, gets 201.0. The triples have larger distances than the one-way marginals although their largest cell errors are similar, since they spread the same noise over many more cells.

Read the table as a check of the pipeline, not only of the noise. A distance that breaks its bound means something systematic, and the previous chapter traced one such break to nulls that the two steps of the release counted differently. On private data you cannot fill this table in, because it needs the real marginals. You can fill in the `std` column, which comes from the release, and that is the error bar you publish.

## A query nobody measured {#sec-eval-unmeasured}

The second question asks about a table that no measurement covers. AIM's candidates have at most three columns, so it never measures the four-way table of `SEX`, `AGE`, `HWUSUAL` and `ILOSTAT`, whose 240 cells hold every combination at once. The model still answers it. It fills those cells with the distribution that agrees with the measured marginals and assumes nothing else. [The full edition's chapter on query release](../../book-full/part03-dp-foundations/06-query-release.md) explains why many queries are best answered through one such model. But the table can no longer put a bar on the answer.

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

In this run the four-way table is further from the truth than every measured marginal. Its errors add up from two sources. One is the noise of the overlapping triples it is assembled from. The other is any dependence among all four columns that no triple shows, which the model cannot know about. Here the four measured triples overlap so much that little is left to guess, and the distance stays near the worst measured one. A release that measured fewer marginals would leave more to the model's assumptions, as @sec-eval-tuning shows for MST.

A conditional share is a harder test, because it divides one small count by another. The script reads the share of people working in each combination of `SEX`, `AGE` band and `HWUSUAL` band off the four-way table. It compares the 26 cells with at least a thousand real rows. The largest error in this run was two points, in the cell of men under 20 who usually work 1 to 20 hours. Everyone in that cell works, since a usual number of hours implies a job, but the model gave the cell a share of 97.95%. Noise is added per count, not per person, so a small group gets the largest relative error. When an analyst will ask about small groups, measure them, for instance with `Fixed` inside `Sequential` as the previous chapter did.

## Train on synthetic, test on real {#sec-eval-downstream}

The third question is whether a model trained on the synthetic rows works on real rows. The benchmark calls this "train on synthetic, test on real". The task here is to predict whether a person works, `ILOSTAT` equal to 1, from `SEX`, the `AGE` band and the `HWUSUAL` band. The model is a logistic regression from scikit-learn on eleven indicator columns, one per key of each feature.

The script draws 200,000 synthetic rows with `method="sample"`, which draws each row independently from the model, as a sample from the population would be. It also takes 400,000 real rows in one seeded Polars shuffle. The first 200,000 train the yardstick, a model on real rows, and the rest are held out to score both models.

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

The model trained on real rows labels 93.13% of the held-out rows correctly, against 39.76% working. The model trained on synthetic rows scored the same in this run, so the gap is zero, well inside the standard error of 0.0006. The test allows a gap of one percentage point. That is generous against the standard error, and it is the size of gap a model would show if it got a few small cells wrong.

Accuracy is a coarse check, though. Two models that put every cell's probability on the same side of one half make the same predictions, whatever the probabilities are. So the script also reports the mean difference between the two models' predicted probabilities, 0.0015 in this run, which sees changes that accuracy cannot. A richer model, one with interactions between the features, would also depend on the parts of the table that are hardest to get right. [Statistical modelling](../part04-extending/14-statistical-modelling.md) fits private models directly, and for a single model that is often better than training on synthetic rows.

## What the budget and the algorithm buy {#sec-eval-tuning}

Tuning means running the evaluation under different settings and choosing between them. The script repeats it with AIM at a fifth of the budget, $\rho = 0.02$, and at five times it, $\rho = 0.5$. Then it runs MST at the original budget. Each release gets a context of its own, because the comparison is between alternatives and you would publish only one of them. Released from the same private data, the four would compose to $\rho = 0.72$, and so tuning on the private data spends budget too. Tune on a public stand-in, and release once.

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

The scales follow the formula of the previous chapter, the sensitivity of 36 over the square root of twice the budget. Twenty-five times the budget therefore divides every scale by five, from 509.12 to 101.82 for the one-way marginals. Converted to $\varepsilon$, the smallest budget is 1.00 and the largest 5.67. [The full edition's chapter on choosing epsilon](../../book-full/part05-dp-ml/09-choosing-epsilon.md) discusses which of them a release can defend.

| Release | $\varepsilon$ | Seconds | Measured | One-way mean | Worst measured | Four-way | Share error (points) | Probability difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AIM, $\rho = 0.02$ | 1.00 | 6.4 | 4 | 0.00016 | 0.00157 | 0.00252 | 15.64 | 0.0016 |
| AIM, $\rho = 0.1$ | 2.35 | 13.7 | 5 | 0.00007 | 0.00102 | 0.00141 | 2.05 | 0.0015 |
| AIM, $\rho = 0.5$ | 5.67 | 7.1 | 6 | 0.00005 | 0.00052 | 0.00077 | 1.65 | 0.0014 |
| MST, $\rho = 0.1$ | 2.35 | 3.3 | 3 | 0.00021 | 0.00114 | 0.05610 | 24.03 | 0.0280 |

: The same evaluation at three budgets with AIM and at one with MST, in one run: ε at δ = 10⁻⁷, the seconds the release took, the number of marginals measured, the total variation distances of the one-way marginals (mean), of the worst measured marginal and of the four-way table, the largest error of the share working, and the mean difference in predicted probability from the model trained on real rows. {#tbl-eval-tuning}

The budget buys accuracy everywhere, but not evenly. Across the three budgets, the worst measured marginal and the four-way table improve steadily, though by less than the scales shrink. The one-way marginals, already close at $\rho = 0.1$, improve least. The share error falls fastest at the low end, from 15.64 points to 2.05, because small groups are the first to drown in noise. A larger budget also lets AIM measure more marginals in its rounds, and in this run it measured four, five and six. All four classifiers scored the same accuracy of 0.9313, so for this task the budget makes no visible difference.

The algorithm changes more than the budget does. MST measures three pairs, all of them with `ILOSTAT` in this run, and so treats `SEX`, `AGE` and `HWUSUAL` as independent once employment status is known. That assumption is false, since hours depend on sex and age among those who work. The four-way table comes out tens of times further from the truth than AIM's at the same budget. The share error reaches 24 points, in the cell of women under 20 who report at most one usual hour. The predicted probabilities move by 0.028 on average, while the accuracy still does not change. MST is the better choice when the questions are about pairs and the release must be fast, since in this run it took 3.3 seconds against 13.7 for AIM.

The cuts and the number of rounds are the other two settings. Finer cuts multiply the cells of every marginal, and since every cell carries noise of the same scale, the bound on each marginal grows with them. More rounds give each round a smaller share, so AIM measures more marginals, each less precisely. Evaluate either change on a public stand-in in the same way.

## How many rows, and how to draw them {#sec-eval-rows}

The evaluation of the marginals read the model, not the rows, and this section shows why. Drawing a fixed number of rows independently adds a sampling error on top of the model's error. The script computes its expected size for each one-way marginal from the table's own shares, at 200,000 rows and at the size of the real table.

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

At 200,000 sampled rows the sampling error of `SEX` is many times the model's error in @tbl-eval-workload, so the rows are much worse than the release. Even at the real table's size the sampling error stays above the model's, because the privacy noise at this budget is small against 3.8 million rows. More rows than the model's total add no information, since they only sample the same model more finely.

The default `method="round"` avoids the sampling error, and for AIM its marginals matched the model's in this run. But in `mbi` 1.1.0 it draws each column by the rank of a row within the group of its parent columns. Two columns drawn from the same parents are then sorted together, which ties them. MST's star makes `AGE` and `HWUSUAL` such a pair, and rounding moved their marginal from 0.01504 to 0.12370 while sampling left it near the model. So sample when the joint distribution of the rows matters, and draw as many rows as the model's total if the marginals must be close. Check the pairs before you trust rounded rows.

## What to publish {#sec-eval-publish}

A synthetic dataset needs a statement of its guarantee and enough description for a user to judge what it can answer. [The practitioner edition's chapter on release and governance](../../book-practitioner/part05-dp-ml/12-release-governance.md) builds such a statement for a trained model, and the same rules apply. The numbers come from the variables of the script, never typed in, and the statement names what was public.

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

The statement covers the rows without a further term, because drawing rows from a released model is post-processing, which [the full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md) proves cannot weaken the guarantee. It covers this one table only. The other releases of this chapter, made from the same people, would add to it.

| Item | What to publish | In this chapter |
| --- | --- | --- |
| The guarantee | $\rho$ and $\delta$, their conversion to $\varepsilon$, the privacy unit and the trust model, printed from the variables | $\rho = 0.1$, $\varepsilon = 2.35$, 36 contributions, central model |
| The columns | Each column with its keys or cuts, the null recoding, and which keys were released | `SEX` keys, `AGE` and `HWUSUAL` cuts, `ILOSTAT` keys at threshold 1364 |
| The algorithm | Its name and parameters, the library versions, and how the rows were drawn | AIM with 10 rounds, OpenDP 0.16.0, `mbi` 1.1.0, 200,000 sampled rows |
| The error bars | The measured marginals and their `std`, which come from the release and cost nothing | @tbl-eval-workload, the `std` column |
| The utility | The marginal errors and the downstream check, measured on public or historical data | Never on the private data itself |

: What to publish with a synthetic dataset, and where this chapter found each item. {#tbl-eval-publish}

The last row is the one most often done wrong. A distance computed on the private data depends on that data, so publishing it is a new release that the statement does not cover. [Limitations and troubleshooting](../part04-extending/15-limitations-and-troubleshooting.md) put it as a rule: the guarantee covers what passes through a measurement and nothing computed beside it. The error bars from `std` are safe to publish. The distances in this chapter's tables were safe only because the data are public.

This is the last chapter of the book. You can now state a privacy unit and a budget, release statistics, tables and models through a context, and read every scale and threshold before spending anything. You can build the framework's pieces by hand when the context is not enough, and turn a contingency table into synthetic rows. You can also measure how good those rows are, on data you are allowed to look at, and publish them with a guarantee that says exactly what it covers.

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
