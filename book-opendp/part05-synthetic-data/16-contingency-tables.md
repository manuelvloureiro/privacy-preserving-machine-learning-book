# Contingency tables

Suppose you need a table of usual work hours by age, employment status and sex. A **contingency table** reports a count for every combination of column values. [Grouping](../part02-tabular-data/04-grouping.md) released a two-column table; adding more columns multiplies its cells and noise. Instead, release a few **marginals**, counts over two or three columns, and fit one model to them. The model then answers other queries over those columns without additional budget.

The chapter follows the documentation's contingency-table page. It sets up a differential privacy (DP) context with a zero-concentrated budget (zCDP), specifies public keys and cut points, and releases a fixed workload of marginals. It then inspects the resulting `ContingencyTable`, compares a projection with its error bar and adds a marginal omitted from the first release.

The script [opendp_contingency_tables.py](../../code/part05-synthetic-data/opendp_contingency_tables.py) runs on the Labour Force Survey (LFS) sample with OpenDP 0.16.0 and `mbi` 1.1.0, the pinned Private-PGM version. Budgets, scales, thresholds, keys and error bars are deterministic. OpenDP's noise cannot be seeded, so the page labels released counts as one run.

```bash
uv run python code/part05-synthetic-data/opendp_contingency_tables.py
```

## From grouped counts to marginals {#sec-ctab-marginals}

Grouped counts from [Grouping](../part02-tabular-data/04-grouping.md) become difficult as columns are added. Each column multiplies the number of cells by its number of values; four raw columns in this sample already produce 4,648 cells. Noise per cell does not depend on its record count. As records spread across more cells, counts shrink while noise stays constant, until noise dominates most cells.

A marginal is a contingency table over a subset of columns. After binning, the two-way `AGE` by `HWUSUAL` marginal has sixteen cells with counts large enough to exceed their noise. A workload of low-order marginals preserves relationships within each subset but omits higher-order relationships. The documentation's workload contains `AGE` by `HWUSUAL` and `HWUSUAL` by `ILOSTAT`.

Separately released noisy marginals can disagree. Both marginals above include one-way `HWUSUAL` counts, but their noisy copies differ. OpenDP therefore fits one probability model to all columns, making its marginals jointly approximate the noisy measurements. All subsequent counts come from that model.

The fit uses **Private-PGM** (McKenna, Sheldon and Miklau, 2019) from the `mbi` package. It builds a graphical model with one factor per measured marginal and fits it by mirror descent, which preserves a valid distribution at each step. Because fitting uses only noisy marginals, it is post-processing and costs no additional budget ([the full edition's chapter on query release](../../book-full/part03-dp-foundations/06-query-release.md)).

Install `pip install 'opendp[mbi]'` to get `mbi` 1.1.0 and run the documentation's code. Each algorithm fits its model with `mbi.estimation.mirror_descent`, using 1,000 steps by default. The final check tests whether that suffices for this dataset.

## The context under zCDP, with a delta {#sec-ctab-context}

The context follows [Assessing utility and choosing the noise](../part02-tabular-data/07-utility-and-noise.md) with a larger budget and the book's privacy unit of 36 contributions per person ([The data and the context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context)). Its zCDP budget is a privacy parameter $\rho$ of 0.2, with a slack $\delta$ of $10^{-7}$ for stable-key selection. The chapter's queries use $\rho = 0.1$ and $\rho = 0.05$, leaving part of the budget unspent.

```{.python excerpt="ctab-context"}
def make_context(frame: pl.LazyFrame) -> dp.Context:
    """The context: 36 contributions per person, rho = 0.2 of zCDP with delta = 1e-7, spent query by query."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),  # a person answers in at most 36 quarters
        privacy_loss=dp.loss_of(rho=RHO, delta=DELTA),  # the delta pays for the stable keys of ILOSTAT
    )
```

```text
privacy unit: (SymmetricDistance(), 36)
privacy loss: (Approximate(ZeroConcentratedDivergence), (0.2, 1e-07))
```

The zCDP budget suits a table that releases many counts: Gaussian costs compose by adding their $\rho$ values, more tightly than pure-DP $\varepsilon$ costs ([Combinators](../part03-framework/11-combinators.md) and the full edition's chapter on [concentrated differential privacy](../../book-full/part04-privacy-accounting/03-concentrated-dp.md)). An $\varepsilon$ budget with Laplace noise is also possible, but the error bars below would then be approximate.

The budget includes $\delta$ because `ILOSTAT` has no keys or cuts. OpenDP selects its keys by the stable-key method from [Grouping](../part02-tabular-data/04-grouping.md), showing a key only when its noisy count exceeds a threshold. The threshold limits the probability that one person's rows alone create a key to at most $\delta$. This slack pays only for key selection. OpenDP rejects nonzero $\delta$ when every column has keys or cuts, and rejects $\delta = 0$ when a column has neither. The script demonstrates both refusals.

```text
delta with keys for every column: delta (1e-07) must be zero because keys and cuts span all columns
no delta with ILOSTAT unkeyed: delta (0.0) must be nonzero because keys and cuts don't span all columns
```

The toy-frame snippet demonstrates the first refusal. When all columns have public keys, no budget is spent discovering keys, so the query reports no one-way scale or threshold.

```python
import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")
frame = pl.LazyFrame({"SEX": [1, 2, 1, 2]})
fixed = dp.mbi.Fixed(queries=[dp.mbi.Count(("SEX",))])
unit = dp.unit_of(contributions=1)
pure = dp.Context.compositor(data=frame, privacy_unit=unit, privacy_loss=dp.loss_of(rho=0.1), split_evenly_over=1)
query = pure.query().contingency_table(keys={"SEX": [1, 2]}, algorithm=fixed)
print(query.oneway_scale, query.oneway_threshold)  # None None
slack = dp.loss_of(rho=0.1, delta=1e-7)
approximate = dp.Context.compositor(data=frame, privacy_unit=unit, privacy_loss=slack, split_evenly_over=1)
try:
    approximate.query().contingency_table(keys={"SEX": [1, 2]}, algorithm=fixed)
except ValueError as err:
    print(err)  # delta (1e-07) must be zero because keys and cuts span all columns
```

## Keys and cuts {#sec-ctab-keys}

The model needs a fixed, finite set of values for every column, because its cells are indexed by them. OpenDP takes that set from one of three places. For a categorical column you pass **keys**, the public list of its values, as `keys={"SEX": [1, 2]}` from the survey's codebook. For a numeric column you pass **cuts**, the public bin edges, as `cuts={"AGE": [20, 40, 60]}`, which make four left-open intervals, from `(-inf, 20.0]` to `(60.0, inf]`. For a column with neither, here `ILOSTAT`, the library spends part of the budget on **stable keys**, the values whose noisy counts clear the threshold.

Each of the three sources can miss values, and the library handles every miss the same way. A value outside the key list, such as a code 3 in `SEX`, becomes `null`, and so does a stable key with too small a count. So `null` stands for "something else", and a true null in the data joins it. To keep true nulls apart, recode them to a value of their own with `with_columns` before the query, as the documentation advises. A cut column has no null bin, so a null `AGE` or `HWUSUAL` lands in the last bin. The check at the end confirms that on the 255 rows without an age and the 20,819 without usual hours.

The `ILOSTAT` codebook lists six codes, from 1 for working to 9 for a child under 15. Codes 4 and 5 do not occur here, so stable-key selection should find four codes. The `null` key represents the two absent codes and has no corresponding rows.

## The Fixed algorithm and its workload {#sec-ctab-fixed}

`dp.mbi.Fixed` releases a specified workload: a list of `Count` queries, one per marginal. Use it when you know which relationships to measure. The algorithms in [the next chapter](17-aim-and-mst.md) select marginals instead. The query specifies columns, keys, cuts and algorithm.

```{.python excerpt="ctab-query"}
def workload_query(context: dp.Context) -> ContingencyTableQuery:
    """The documentation's query: keys for SEX, cuts for AGE and HWUSUAL, stable keys for ILOSTAT, two counts."""
    return (
        context.query(rho=TABLE_RHO, delta=DELTA)
        .select("SEX", "AGE", "HWUSUAL", "ILOSTAT")
        .contingency_table(
            keys={"SEX": [1, 2]},  # public: the codebook's two codes
            cuts={"AGE": [20, 40, 60], "HWUSUAL": [1, 20, 40]},  # left-open bins: (-inf, 20], (20, 40], ...
            algorithm=dp.mbi.Fixed(
                queries=[dp.mbi.Count(("AGE", "HWUSUAL")), dp.mbi.Count(("HWUSUAL", "ILOSTAT"))],
                oneway_split=ONEWAY_SPLIT,  # the share of rho for the one-way marginal of ILOSTAT
            ),
        )
    )
```

The query splits its budget between two uses. `oneway_split` assigns a share to one-way marginals for columns without keys; here, it discovers stable keys for `ILOSTAT`. The remaining budget funds the workload, divided equally among its counts unless weights are provided. By default, `Fixed` uses half the budget times the fraction of unkeyed columns. Here that is half of one quarter. The documentation sets the resulting one-eighth share explicitly.

```text
oneway split: 0.125 of rho=0.1 -> rho=0.0125 one-way, rho=0.04375 per count
oneway scale: 227.68  oneway threshold: 1364
```

Inspect both attributes before release. The scale is the standard deviation of Gaussian noise on the one-way `ILOSTAT` counts. The threshold is the minimum noisy count for retaining a key. The privacy unit sets both: one person's 36 rows may share a status, so that person's contribution to one cell can be 36. This is the marginal's sensitivity. At budget $\rho$, the Gaussian scale for a count of sensitivity $\Delta$ is $\Delta / \sqrt{2\rho}$ ([Assessing utility and choosing the noise](../part02-tabular-data/07-utility-and-noise.md)). The script checks the scale for each budget share.

```text
scale from the formula 36 / sqrt(2 x 0.0125): 227.68
scale of each count of the workload, 36 / sqrt(2 x 0.04375): 121.70
```

Each cell in the two measured marginals receives Gaussian noise with standard deviation 121.70. The threshold 1364 follows the method in [Grouping](../part02-tabular-data/04-grouping.md). At scale 227.68, a key supported only by one person's 36 rows exceeds the threshold with probability at most $\delta$. The privacy unit sets sensitivity to 36; truncating to one row per person and year ([Identifier truncation](../part02-tabular-data/05-identifier-truncation.md)) would reduce it.

## The released table {#sec-ctab-table}

```{.python excerpt="ctab-release"}
table = query.release()
ilostat_std = table.std(("ILOSTAT",))
ilostat_accuracy = dp.gaussian_scale_to_accuracy(ilostat_std, alpha=ALPHA)
```

The release returns a `ContingencyTable` containing the model's key sets, cuts, stable-key thresholds and each column's key type.

```text
keys: SEX=[1, 2, null]  AGE=[(-inf, 20.0], (20.0, 40.0], (40.0, 60.0], (60.0, inf]]  HWUSUAL=[(-inf, 1.0], (1.0, 20.0], (20.0, 40.0], (40.0, inf]]  ILOSTAT=[1, 2, 3, 9, null]
cuts: AGE=[20.0, 40.0, 60.0]  HWUSUAL=[1.0, 20.0, 40.0]
thresholds: {'ILOSTAT': 1364}
schema: {'SEX': Int64, 'AGE': String, 'HWUSUAL': String, 'ILOSTAT': Int64}
```

Every key set without cuts ends in `null`, which collects values outside the specified keys. `SEX` therefore has a `null` key, although the data contain only 1 and 2. `ILOSTAT` has four observed codes and `null`. The schema shows that cut columns use string bin labels, so queries name bins rather than numeric values.

`project` sums out every column except those named and returns counts in key order. `project_melted` returns the same projection as a data frame. Both are post-processing. They can project onto unmeasured column sets, but those estimates have no release-based error bar.

`std` estimates a projection's standard deviation; `dp.gaussian_scale_to_accuracy` converts it to an error bar. Projecting a two-way marginal onto one column sums four noisy cells, doubling the standard deviation from 121.70 to 243.40. Thus each `ILOSTAT` count is within 477.06 of its truth with probability 0.95. This estimate excludes the one-way `ILOSTAT` measurement: it releases keys only, so OpenDP assigns no cell-wise precision to it. `HWUSUAL` appears in both measured marginals, and their estimates combine by inverse variance to 181.42.

```text
std ILOSTAT: 243.40  95% accuracy: 477.06
std AGE x HWUSUAL: 121.70  95% accuracy: 238.53
std HWUSUAL: 181.42  95% accuracy: 355.58
```

## Adaptive estimation {#sec-ctab-adaptive}

No measured marginal contains `SEX`, although its keys are known. The model therefore distributes the total evenly across its three keys, and `std` cannot provide an error bar.

```text
std SEX: attrs (('SEX',)) are not covered by the query set
```

A second query adds the missing marginal without rebuilding the table from scratch. Passing the released table as `table=` preserves its keys, cuts, marginals and model, then measures the new marginal and refits the model. All columns now have keys, including those discovered in the first release, so this query must use $\delta = 0$ and spends nothing on key discovery.

```{.python excerpt="ctab-adaptive"}
def sex_query(context: dp.Context, table: dp.mbi.ContingencyTable) -> ContingencyTableQuery:
    """A second query that adds the marginal of SEX to the released table's model; every key is known, so delta = 0."""
    return (
        context.query(rho=SEX_RHO, delta=0.0)
        .select("SEX")
        .contingency_table(
            table=table,  # keep the keys, cuts, marginals and model of the first release
            algorithm=dp.mbi.Fixed(queries=[dp.mbi.Count(("SEX",))]),
        )
    )
```

The new one-way marginal uses $\rho = 0.05$ and sensitivity 36, giving scale 113.84 and a 223.13 error bar for each of the three `SEX` cells. The model now has four marginals, and the context has spent $\rho = 0.15$ of its 0.2 budget. Because the second query's budget was chosen after seeing the first table, the releases compose adaptively ([Combinators](../part03-framework/11-combinators.md)). The new fit starts from the first model and runs another 1,000 mirror-descent steps.

```text
std SEX after the second query: 113.84  95% accuracy: 223.13
marginals in the model: [('ILOSTAT',), ('AGE', 'HWUSUAL'), ('HWUSUAL', 'ILOSTAT'), ('SEX',)]
privacy loss spent: (0.15, 1e-07)
```

## Checking the release against the truth {#sec-ctab-check}

Because the data are public, the script can compare estimates with truth. Polars computes the one-way `ILOSTAT` counts and the `AGE` by `HWUSUAL` table using the documentation's cuts. As in the library, null ages and hours go in the last bin.

```text
true ILOSTAT counts: 1=1,517,011  2=155,661  3=1,450,283  9=688,977
true AGE x HWUSUAL (a null AGE or HWUSUAL counts in the last bin):
  AGE (-inf, 20.0]: (-inf, 1.0]=2,733  (1.0, 20.0]=13,651  (20.0, 40.0]=95,686  (40.0, inf]=1,051,617
  AGE (20.0, 40.0]: (-inf, 1.0]=1,441  (1.0, 20.0]=31,415  (20.0, 40.0]=370,616  (40.0, inf]=258,601
  AGE (40.0, 60.0]: (-inf, 1.0]=1,285  (1.0, 20.0]=46,743  (20.0, 40.0]=437,820  (40.0, inf]=309,224
  AGE (60.0, inf]: (-inf, 1.0]=1,502  (1.0, 20.0]=28,084  (20.0, 40.0]=122,405  (40.0, inf]=1,039,109
```

The following one-run releases can be compared with the error bars above. Their errors also reflect model fitting.

```text
released (one run):
  ILOSTAT 1: 1,515,589  (true 1,517,011, off by -1,422)
  ILOSTAT 2: 155,622  (true 155,661, off by -39)
  ILOSTAT 3: 1,450,332  (true 1,450,283, off by +49)
  ILOSTAT 9: 688,890  (true 688,977, off by -87)
  ILOSTAT null: 1,142  (true 0, off by +1,142)
  AGE x HWUSUAL largest |released - true|: 521  (accuracy 238.53)
  at AGE (60.0, inf] HWUSUAL (40.0, inf]: 1,038,588  (true 1,039,109)
  last bin of both, the nulls' cell: 1,038,588  (true 1,039,109, off by -521)
  SEX before the second query: 1,270,525 1,270,525 1,270,525  (uniform)
  SEX after: 1: 1,828,537  (true 1,828,992)  2: 1,982,506  (true 1,982,940)  null: 610  (true 0)
  ILOSTAT after the second query:
    ILOSTAT 1: 1,516,288  (true 1,517,011, off by -723)
    ILOSTAT 2: 155,608  (true 155,661, off by -53)
    ILOSTAT 3: 1,450,301  (true 1,450,283, off by +18)
    ILOSTAT 9: 688,867  (true 688,977, off by -110)
    ILOSTAT null: 590  (true 0, off by +590)
```

The largest of the sixteen two-way errors is 521 rows: more than twice, but less than three times, the 238.53 error bar. It occurs in the last bin of both columns, which contains rows with null hours or age; this confirms how the library assigns nulls. The `null` keys of `ILOSTAT` and `SEX` have no true rows but receive positive estimates, because fitted counts cannot be negative. Expect upward bias for cells whose true count is zero.

Model fitting explains the larger errors. The working code 1 of `ILOSTAT` is 1,422 rows below truth, more than twice its 477.06 bar. Its empty `null` key instead holds 1,142 rows. The fit starts from a uniform model, which assigns one fifth of the rows to `null`. Because an empty key approaches zero only in the limit, 1,000 steps leave about a thousand rows there, missing from other keys.

The second query shows the same effect for `SEX`. Its fit starts from the first model, where the `null` key holds one third of the rows. After 1,000 steps, it still holds 610 rows, more than twice its bar. The two real keys are therefore short by 455 and 434 rows. The fit also continues to adjust `ILOSTAT`; its working code remains 723 rows below truth.

The script tests the fit by refitting the second table's four released marginals from scratch with 10,000 steps of `mbi.estimation.mirror_descent`, rather than the default 1,000. This uses no data, so it is post-processing and a development check, not part of a release.

```text
  the same marginals refitted at 10,000 steps (dev time, post-processing):
    ILOSTAT 1: 1,516,829  (true 1,517,011, off by -182)
    ILOSTAT 2: 155,587  (true 155,661, off by -74)
    ILOSTAT 3: 1,450,277  (true 1,450,283, off by -6)
    ILOSTAT 9: 688,859  (true 688,977, off by -118)
    ILOSTAT null: 103  (true 0, off by +103)
    SEX: 1: 1,828,829  (true 1,828,992)  2: 1,982,798  (true 1,982,940)  null: 27  (true 0)
    AGE x HWUSUAL largest |refit - true|: 295  (accuracy 238.53)
```

After more steps, the `null` keys of `ILOSTAT` and `SEX` hold 103 and 27 rows, within their bars. Code 1 is 182 rows from truth, and each real `SEX` key is within 163. The largest two-way error is 295 rows, less than three times its bar. Early stopping therefore biases empty cells upward and larger cells downward without warning in the release. A development check against public or historical data reveals this bias.

The script's test requires every two-way cell and count from the longer fit to lie within three error bars. For the default fit, it checks that each `null` key holds less than one tenth of a percent of the rows and that each real key's error is within three bars plus the rows remaining in `null`.

## Summary

| Call or concept | What it does | Rule of thumb |
| --- | --- | --- |
| `dp.loss_of(rho=0.2, delta=1e-7)` | A zCDP budget with a slack for stable keys | Gaussian noise composes tightly over many marginals; the $\delta$ is needed exactly when a column has neither keys nor cuts |
| `keys={"SEX": [1, 2]}` | The public values of a categorical column | Values outside the list become `null` |
| `cuts={"AGE": [20, 40, 60]}` | Left-open bins of a numeric column | Three cuts make four bins; a null lands in the last one |
| A column with neither | Stable keys, bought with `oneway_split` of the budget | Rare values fall below the threshold and join `null` |
| `dp.mbi.Fixed(queries=[dp.mbi.Count(...)], oneway_split=1/8)` | A workload named in advance | The rest of the budget is shared equally over the counts, or by their weights |
| `query.oneway_scale`, `query.oneway_threshold` | The noise and the cut-off of the key discovery | Read them before releasing; both are `None` when every column is keyed |
| `table.keys`, `table.cuts`, `table.thresholds`, `table.schema` | What the model was fitted on | Cut columns come back as bin labels, not numbers |
| `table.project`, `table.project_melted` | Any count over the table's columns, from the model | Post-processing; a column no marginal covers comes back uniform |
| `table.std` with `dp.gaussian_scale_to_accuracy` | The error bar of a covered projection | Raises for an uncovered projection; a cell near zero is biased upward |
| `contingency_table(table=table, ...)` | Adds marginals to an existing model | Every key is then known, so $\delta = 0$ |
| `mbi.estimation.mirror_descent(domain, marginals, iters=...)` | Refits the released marginals with more steps | Post-processing; the default 1,000 steps leave rows in the empty `null` keys on this data, so check the fit on public data |

: The calls and concepts of a contingency-table release, what each does, and the rule of thumb that goes with it. {#tbl-ctab-summary}

## References

- OpenDP documentation: "Contingency Tables" in Getting Started, and the opening of "Synthetic Data", which lists the algorithms and the rules for null values.
- McKenna, Sheldon and Miklau, "Graphical-model based estimation and inference for differential privacy" (2019): Private-PGM, the model that one consistent table is fitted from, and the mirror-descent fit.
- Bun and Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds" (2016): zCDP and the Gaussian mechanism under it.
- The full edition's chapters on [query release](../../book-full/part03-dp-foundations/06-query-release.md), [sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) and [concentrated differential privacy](../../book-full/part04-privacy-accounting/03-concentrated-dp.md) derive the results this chapter states.
