# Contingency tables

Suppose you want to publish not one count but a whole table. It would show how many people of each age group usually work each number of hours, by employment status and sex. A **contingency table** is that table. It counts the records in every combination of the values of several columns. [Grouping](../part02-tabular-data/04-grouping.md) released one for two columns, and this chapter asks what changes when there are four, or forty. The answer is that you no longer release the table itself. You release a few **marginals**, which are the same counts for two or three columns at a time, and fit one model that agrees with all of them. The model then answers any count over those columns, consistently, from the budget already spent.

The chapter follows the documentation's page on contingency tables. You will set up a context under zero-concentrated differential privacy (zCDP) and say which columns have known keys or cut points. Then you will release a fixed workload of marginals and read the `ContingencyTable` that comes back. At the end you will read a projection of the table against its error bar and add a marginal that the first release left out.

The script [opendp_contingency_tables.py](../../code/part05-synthetic-data/opendp_contingency_tables.py) runs every step on OpenDP 0.16.0 with `mbi` 1.1.0, the Private-PGM version OpenDP pins, against the Labour Force Survey (LFS) sample of the earlier chapters. The budgets, scales, thresholds, keys and error bars it prints are the same on every run. The released counts are not, because OpenDP's noise cannot be seeded, so the page shows them in one block marked as one run.

```bash
uv run python code/part05-synthetic-data/opendp_contingency_tables.py
```

## From grouped counts to marginals {#sec-ctab-marginals}

Why not release the full table with the grouped counts of [Grouping](../part02-tabular-data/04-grouping.md)? Two things go wrong as columns are added. The first is size. Each column multiplies the number of cells by its number of values, and the four raw columns of this sample already make 4,648 cells. The second is noise. The noise on a cell does not depend on how many records the cell holds. So spreading the same records over more cells makes each count smaller while its noise stays the same. Soon most cells are noise.

A marginal is the contingency table of a subset of the columns. The two-way marginal of `AGE` and `HWUSUAL` has sixteen cells once the two columns are binned as below, each with a count large enough to survive its noise. Releasing several low-order marginals keeps the relations between the columns that share a marginal and gives up the higher-order ones. The documentation's workload has two marginals, `AGE` with `HWUSUAL` and `HWUSUAL` with `ILOSTAT`.

Noisy marginals released separately do not agree with each other. Both marginals above contain the one-way counts of `HWUSUAL`, and two noisy copies of the same count differ. OpenDP therefore does not hand you the noisy marginals. It fits one probability model over all the columns whose own marginals come as close as possible to the noisy ones, and every count you read comes from that model.

The method is **Private-PGM**, which McKenna, Sheldon and Miklau introduced in 2019 in their paper "Graphical-model based estimation and inference for differential privacy". The model is a graphical model with one factor per measured marginal, fitted by mirror descent, a gradient method that keeps the model a valid distribution at every step. The fitting sees only the noisy marginals, so it is post-processing and costs no budget. Private-PGM is the `mbi` package, and the full edition's chapter on [query release](../../book-full/part03-dp-foundations/06-query-release.md) covers the problem it solves, which is answering many counting queries from a few noisy measurements.

Install the contingency-table extra with `pip install 'opendp[mbi]'`. It brings `mbi` 1.1.0, the version OpenDP 0.16.0 pins, and the documentation's code then runs as written. Every algorithm fits its model with `mbi.estimation.mirror_descent` at that package's default of 1,000 steps. The check at the end of the chapter looks at whether that is enough on this data.

## The context under zCDP, with a delta {#sec-ctab-context}

The context is the one of [Assessing utility and choosing the noise](../part02-tabular-data/07-utility-and-noise.md), with a larger budget. As there, the privacy unit is 36 contributions, because a person answers at most once per quarter over nine years. The budget is a privacy parameter $\rho$ of 0.2 under zero-concentrated differential privacy, together with a slack $\delta$ of $10^{-7}$ for the stable keys. Of that, the two queries of the chapter take $\rho = 0.1$ and $\rho = 0.05$, so part of the budget stays unspent.

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

Why zCDP rather than pure differential privacy? A contingency table is many counts released together. Under zCDP the Gaussian noise on all of them composes by adding up their values of $\rho$. That sum grows more slowly than the privacy parameters $\varepsilon$ of pure differential privacy would add up. [Combinators](../part03-framework/11-combinators.md) showed that rule, and the full edition's chapter on [concentrated differential privacy](../../book-full/part04-privacy-accounting/03-concentrated-dp.md) derives it. The mechanism also accepts a budget in $\varepsilon$, with Laplace noise. The error bars below are then only approximate, because a sum of Laplace draws is not Laplace.

Why a $\delta$? One column, `ILOSTAT`, will have neither keys nor cuts. Its keys will be read from the data, which is the stable-keys release of [Grouping](../part02-tabular-data/04-grouping.md). There a key is shown only when its noisy count clears a threshold, set so that one person alone reaches it with probability at most $\delta$. The slack pays for that threshold and for nothing else, and the library enforces the connection in both directions. Give every column keys or cuts and also a $\delta$, and the query is refused. Leave a column without them and set $\delta = 0$, and it is refused too. The script provokes both refusals on this data.

```text
delta with keys for every column: delta (1e-07) must be zero because keys and cuts span all columns
no delta with ILOSTAT unkeyed: delta (0.0) must be nonzero because keys and cuts don't span all columns
```

The snippet below shows the first refusal on a toy frame, together with one more fact. When every column is keyed, nothing is spent on discovering keys, and the query reports no one-way scale or threshold at all.

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

The model needs a fixed, finite set of values for every column, because its cells are indexed by them. OpenDP takes that set from one of three places. For a categorical column you pass **keys**, the public list of its values, as `keys={"SEX": [1, 2]}` from the survey's codebook. For a numeric column you pass **cuts**, the public bin edges, as `cuts={"AGE": [20, 40, 60]}`. The column is then binned into left-open intervals, four of them for three cuts, labelled `(-inf, 20.0]`, `(20.0, 40.0]`, `(40.0, 60.0]` and `(60.0, inf]`. For a column with neither, here `ILOSTAT`, the library spends part of the budget on **stable keys**, the values whose noisy counts clear the threshold.

Each of the three sources can miss values, and the library handles every miss the same way. A value outside the key list, a code 3 in `SEX` for instance, becomes `null`. A stable key with too small a count is not released and becomes `null` too. So `null` stands for "something else", and a true null in the data joins it. If you need to tell true nulls apart, recode them to a value of their own with `with_columns` before the query, as the documentation advises. A cut column has no null bin, so a null `AGE` or `HWUSUAL` lands in the last bin. The check at the end of the chapter confirms that on the 255 rows without an age and the 20,819 without usual hours.

The codebook of `ILOSTAT` lists six codes, from 1 for working to 9 for a child under 15. Two of them, 4 for conscripts and 5 for other inactive people, do not occur in this sample at all, so the stable keys should find exactly four. The documentation describes codes 4 and 5 as absorbed into `null`. In this sample `null` is their cell, with no rows behind it, and the released key set below shows the same thing.

## The Fixed algorithm and its workload {#sec-ctab-fixed}

`dp.mbi.Fixed` releases a workload you name in advance, as a list of `Count` queries, one per marginal. The algorithms of [the next chapter](17-aim-and-mst.md) choose the marginals themselves, and a fixed workload is the right tool when you already know which relations you need. The query names the columns, the keys and cuts, and the algorithm.

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

The budget of the query is split in two parts. A share `oneway_split` goes to the one-way marginals of the columns without keys, which is where the stable keys of `ILOSTAT` come from. The rest goes to the workload, equally over its counts unless you give them weights. When you do not set the share, `Fixed` uses half the budget times the fraction of columns without keys, which here is a half of a quarter. The documentation sets it to one eighth explicitly, which is the same number, and the script prints the split.

```text
oneway split: 0.125 of rho=0.1 -> rho=0.0125 one-way, rho=0.04375 per count
oneway scale: 227.68  oneway threshold: 1364
```

Read the two attributes before releasing. The scale is the standard deviation of the Gaussian noise on the one-way counts of `ILOSTAT`, and the threshold is the count a noisy key must exceed to be kept. Both follow from the privacy unit. One person's 36 rows can all carry the same status, so adding or removing a person changes one count by up to 36. That number is the sensitivity of the one-way marginal. The Gaussian scale that spends $\rho$ on a count of sensitivity $\Delta$ is $\Delta / \sqrt{2\rho}$, as [Assessing utility and choosing the noise](../part02-tabular-data/07-utility-and-noise.md) stated. The script checks the arithmetic for both parts of the budget.

```text
scale from the formula 36 / sqrt(2 x 0.0125): 227.68
scale of each count of the workload, 36 / sqrt(2 x 0.04375): 121.70
```

So each cell of the two measured marginals carries Gaussian noise with standard deviation 121.70. The threshold 1364 is set as in [Grouping](../part02-tabular-data/04-grouping.md). At scale 227.68, a key held by one person alone, in all 36 of their rows, clears it with probability at most $\delta$. A smaller threshold needs more keys, a larger budget, or a tighter bound on how many groups one person can touch. The sensitivity 36 is the price of the privacy unit. Truncating to one row per person and year, as [Identifier truncation](../part02-tabular-data/05-identifier-truncation.md) does, would lower it. The sensitivity of each marginal is derived like that of any grouped count, in [the full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md).

## The released table {#sec-ctab-table}

```{.python excerpt="ctab-release"}
table = query.release()
ilostat_std = table.std(("ILOSTAT",))
ilostat_accuracy = dp.gaussian_scale_to_accuracy(ilostat_std, alpha=ALPHA)
```

The release returns a `ContingencyTable`. The table holds the key sets the model was fitted on, the cuts, the thresholds that produced any stable keys, and the data type of each column's keys.

```text
keys: SEX=[1, 2, null]  AGE=[(-inf, 20.0], (20.0, 40.0], (40.0, 60.0], (60.0, inf]]  HWUSUAL=[(-inf, 1.0], (1.0, 20.0], (20.0, 40.0], (40.0, inf]]  ILOSTAT=[1, 2, 3, 9, null]
cuts: AGE=[20.0, 40.0, 60.0]  HWUSUAL=[1.0, 20.0, 40.0]
thresholds: {'ILOSTAT': 1364}
schema: {'SEX': Int64, 'AGE': String, 'HWUSUAL': String, 'ILOSTAT': Int64}
```

Every key set that is not a cut set ends in `null`, the cell for values outside the keys. `SEX` has one although the data hold only 1 and 2. The stable keys of `ILOSTAT` are the four codes with rows, plus `null`. The schema shows that a cut column now holds strings, the bin labels, so a query on the table names bins and not numbers.

The method `project` sums the model over every column but the ones you name and returns the counts as an array, in the order of the keys. The method `project_melted` returns the same counts as a data frame with one row per combination of keys. Both are post-processing. You can project onto any set of columns, including ones no marginal measured, which the next section shows is a mixed blessing.

`std` estimates the standard deviation of a projection, and `dp.gaussian_scale_to_accuracy` turns it into an error bar, as for any Gaussian release. Projecting a two-way marginal onto one of its columns sums four noisy cells, so the standard deviation doubles, from 121.70 to 243.40. With probability 0.95, each count of `ILOSTAT` is therefore within 477.06 of the truth. That estimate leaves out the one-way marginal of `ILOSTAT` itself. It was measured only over the keys that were released, a prefix of the marginal rather than the whole of it. OpenDP counts no cell-wise precision for such a measurement. The column `HWUSUAL` sits in both measured marginals, and the two estimates combine by their inverse variances to 181.42.

```text
std ILOSTAT: 243.40  95% accuracy: 477.06
std AGE x HWUSUAL: 121.70  95% accuracy: 238.53
std HWUSUAL: 181.42  95% accuracy: 355.58
```

## Adaptive estimation {#sec-ctab-adaptive}

`SEX` has keys, but no marginal of the workload contains it. The model has an estimate of the total number of rows and nothing about how they split by sex. So it spreads the total evenly over the three keys, and `std` refuses to put a bar on that.

```text
std SEX: attrs (('SEX',)) are not covered by the query set
```

A second query repairs this without starting over. Passing the released table as `table=` tells the library to keep its keys, cuts, marginals and model, measure the new marginals, and refit. Every column now has keys, the ones found by the first release included, so this query must carry $\delta = 0$ and spends nothing on key discovery.

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

The new marginal is a one-way count at $\rho = 0.05$ with the same sensitivity as before, so its scale is 113.84. The three cells of `SEX` now have an error bar of 223.13. The model has four marginals, and the context has spent $\rho = 0.15$ of its 0.2. The second query's budget was chosen after seeing the first query's table, which is the adaptive composition that [Combinators](../part03-framework/11-combinators.md) described. The refit starts from the first model's potentials, a warm start, and runs its own 1,000 steps of mirror descent from there.

```text
std SEX after the second query: 113.84  95% accuracy: 223.13
marginals in the model: [('ILOSTAT',), ('AGE', 'HWUSUAL'), ('HWUSUAL', 'ILOSTAT'), ('SEX',)]
privacy loss spent: (0.15, 1e-07)
```

## Checking the release against the truth {#sec-ctab-check}

The data are public, so the script computes what the table estimates and compares. The true one-way counts of `ILOSTAT` and the true `AGE` by `HWUSUAL` table for the documentation's cuts come from Polars. Every null age or null hours is placed in the last bin, as the library places them.

```text
true ILOSTAT counts: 1=1,517,011  2=155,661  3=1,450,283  9=688,977
true AGE x HWUSUAL (a null AGE or HWUSUAL counts in the last bin):
  AGE (-inf, 20.0]: (-inf, 1.0]=2,733  (1.0, 20.0]=13,651  (20.0, 40.0]=95,686  (40.0, inf]=1,051,617
  AGE (20.0, 40.0]: (-inf, 1.0]=1,441  (1.0, 20.0]=31,415  (20.0, 40.0]=370,616  (40.0, inf]=258,601
  AGE (40.0, 60.0]: (-inf, 1.0]=1,285  (1.0, 20.0]=46,743  (20.0, 40.0]=437,820  (40.0, inf]=309,224
  AGE (60.0, inf]: (-inf, 1.0]=1,502  (1.0, 20.0]=28,084  (20.0, 40.0]=122,405  (40.0, inf]=1,039,109
```

The releases of one run follow, read against the error bars above. The noise is only part of what they show.

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

Start with the cells the noise alone explains. The largest of the sixteen two-way errors is 521 rows, past twice the per-cell bar of 238.53 but inside three times it. That largest error sits in the last bin of both columns, which holds the 20,819 rows without usual hours and the 255 without an age. A null placed in any other bin would have left this cell some 20,000 rows short, so the run still confirms where the library puts a null. The `null` keys of `ILOSTAT` and `SEX` have no rows behind them, and their estimates are positive, because a fitted distribution cannot go negative. Read a cell whose truth may be zero with that in mind.

Now the counts that the noise does not explain. The working code 1 of `ILOSTAT` came out 1,422 rows short, past twice its bar of 477.06. The empty `null` key holds 1,142 rows. Both are the fit, not the noise. The fit starts from the uniform model, the one that spread `SEX` evenly above, so `null` begins with a fifth of all rows. A key with no rows is the slowest cell to fit, because the model can empty it only in the limit. One thousand steps bring it down to about a thousand rows and no further, and the rows it keeps are missing from the other keys.

The second query shows the same thing in `SEX`. Its fit starts from the first model, in which the `null` key of `SEX` holds a third of all rows. After its own thousand steps that key still holds 610 rows, more than twice its bar. The two real keys are short by 455 and 434 rows. Those steps also carried on fitting `ILOSTAT`, so its working code is now 723 rows short and its `null` holds 590.

A refit settles it. The script refits the four released marginals of the second table from scratch, with 10,000 steps of mirror descent instead of the default 1,000, by calling `mbi.estimation.mirror_descent` on them. That touches no data, so it is post-processing and a dev-time check, never part of a release.

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

The refit leaves 103 rows in the `null` key of `ILOSTAT` and 27 in that of `SEX`, inside their bars. Code 1 is within 182 rows of the truth, and each real key of `SEX` is within 163. The largest two-way error is 295 rows, inside three times its bar. A fit that stops early biases the empty cells upward and the largest ones downward, and nothing in the release warns you. A dev-time check against public or historical data is how you notice it.

The test behind the script holds each two-way cell and each refitted count of `ILOSTAT` and `SEX` within three times its bar. A correct run leaves that band less than once in a thousand. The released counts of the default fit it checks as this section describes. Each `null` holds fewer than a tenth of a percent of the rows and more than the refit leaves there. Each real key is within three times its bar plus the rows left in `null`, because those rows were taken from the real keys.

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
