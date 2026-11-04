# Synthetic data with AIM and MST

[Contingency tables](16-contingency-tables.md) released a specified workload, which suits known questions. Synthetic-data users may not know future questions, so a fixed workload can omit important relationships. This chapter uses algorithms that choose marginals: **AIM** selects the marginal its current model fits worst at each round, while **MST** selects a spanning tree of column pairs. Both fit a model to noisy marginals. `synthesize()` draws rows from that model as post-processing, at no additional cost.

The chapter follows the documentation's synthetic-data page and the docstrings for `dp.mbi.AIM`, `dp.mbi.MST`, `dp.mbi.Sequential` and `ContingencyTable.synthesize`. It releases three tables over four survey columns, inspects the selected marginals, generates a synthetic frame of 3.8 million rows and states the guarantee. Comparing synthetic and real marginals reveals a null-handling problem, which the chapter then fixes.

The script [opendp_aim_and_mst.py](../../code/part05-synthetic-data/opendp_aim_and_mst.py) runs on the Labour Force Survey (LFS) sample with OpenDP 0.16.0, pinned `mbi` 1.1.0 and CPU JAX. Noise scales, thresholds, budget calculations, conversions to $\varepsilon$ and data facts are deterministic. OpenDP's noise cannot be seeded, so selected marginals, counts, rows and distances vary; the page labels them as one run. The script takes about 75 seconds.

```bash
uv run python code/part05-synthetic-data/opendp_aim_and_mst.py
```

## When you do not know the questions {#sec-synth-adaptive}

A fixed workload spends budget only on specified marginals. For an omitted column pair, the model assumes independence; in the previous chapter, this made `SEX` uniform. With four columns, measuring every pair and triple would divide the budget among ten marginals, including many the model already fits well. The goal is to spend budget where the model is wrong.

AIM selects marginals adaptively. It first measures every one-way marginal with half the budget, which also releases keys for unkeyed columns. It then fits a model and runs rounds. Each round scores candidate marginals of up to three columns by model error adjusted for measurement noise. Candidates that overlap the workload receive greater weight, so the three-way workload marginals weigh most.

Report-noisy-max, a form of the exponential mechanism, uses one tenth of the round's budget to select a candidate ([the full edition's chapter on selection](../../book-full/part03-dp-foundations/05-selection.md)). The other nine tenths measure that marginal with Gaussian noise, after which AIM refits the model.

Round budgets adapt. Each starts at the total budget divided by the planned number of rounds. If a measurement changes the model by less than its noise, the next round receives four times the share. A privacy filter stops the rounds when the budget is spent. The `rounds` argument sets the plan: 16 per column by default, or 64 here. The script uses 10 to reduce runtime. `max_size` caps model memory in megabytes; it does not bind for these four small columns.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
aim = dp.mbi.AIM()
print(aim.queries, aim.measure_split, aim.max_size, aim.rounds)  # 3 0.9 80.0 None
print(dp.mbi.MST().num_selections, dp.mbi.MST().measure_split)  # None 0.9
print(dp.mbi.Count(("SEX", "ILOSTAT")))  # Count(by=('SEX', 'ILOSTAT'), weight=1.0)
```

`queries` sets either the maximum candidate size (3 by default) or a workload of weighted `dp.mbi.Count` objects. `measure_split=0.9` allocates 90% of each round to measurement. MST uses the same split and adds `num_selections` (@sec-synth-mst).

## The AIM release {#sec-synth-aim}

The context follows the documentation and triples the budget because this chapter releases three tables. It uses the 36-contribution privacy unit from [The data and the context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context). Its budget is a privacy parameter $\rho$ of 0.3 under zero-concentrated differential privacy (zCDP), with Gaussian noise; the parameter bounds privacy loss on average ([the full edition's chapter on concentrated differential privacy (DP)](../../book-full/part04-privacy-accounting/03-concentrated-dp.md)). The slack $\delta$ of $3 \times 10^{-7}$ pays for selecting `ILOSTAT` keys, which the query does not specify.

```{.python excerpt="synth-context"}
def make_context(frame: pl.LazyFrame) -> dp.Context:
    """The context: a privacy unit of 36 rows, a budget of rho = 0.3 with delta = 3e-7, one share per release."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),  # neighbours add or remove one person's 36 rows
        privacy_loss=dp.loss_of(rho=RHO, delta=DELTA),  # zCDP with Gaussian noise; delta pays for ILOSTAT's keys
    )
```

One function builds all three releases, which differ only by algorithm. Each query receives one-third of the budget: $\rho = 0.1$ and $\delta = 10^{-7}$. It selects four columns, supplies public `SEX` keys and `AGE` and `HWUSUAL` cuts, and discovers `ILOSTAT` keys by thresholding. The `fill` argument is used later.

```{.python excerpt="synth-query"}
def table_query(context: dp.Context, algorithm: Algorithm, fill: bool = False) -> ContingencyTableQuery:
    """The documentation's contingency-table query at rho = 0.1 and delta = 1e-7, with the algorithm plugged in."""
    query = context.query(rho=RHO_QUERY, delta=DELTA_QUERY)
    if fill:
        query = query.with_columns(fill_nulls())  # a row-by-row transformation before the table
    return query.select(*COLUMNS).contingency_table(
        keys=KEYS,  # public categories of SEX; ILOSTAT's are released with the threshold
        cuts=CUTS,  # bins for the two numeric columns
        algorithm=algorithm,
    )
```

The first release uses AIM with ten rounds. Before release, inspect the one-way scale and threshold; reading them incurs no cost.

```{.python excerpt="synth-aim"}
aim_query = table_query(context, dp.mbi.AIM(rounds=ROUNDS))
oneway_scale, oneway_threshold = aim_query.oneway_scale, aim_query.oneway_threshold
table_aim, seconds_aim = timed_release(aim_query)
```

```text
privacy unit: (SymmetricDistance(), 36)
privacy loss: (Approximate(ZeroConcentratedDivergence), (0.3, 3e-07))
query budget: rho=0.1 delta=1e-07
aim: oneway scale=227.6840 threshold=1364
aim: rho per one-way marginal=0.0125  scale 36/sqrt(2 rho)=227.6840
aim: rounds=10  rho per round=0.0050  measure=0.0045  select=0.0005  scale of a round's marginal=379.4733
aim keys ILOSTAT: [1, 2, 3, 9, None]  thresholds: {'ILOSTAT': 1364}
```

The budget split determines the scales. Half of the query's $\rho$ funds the four one-way marginals, giving each 0.0125. One person can change at most 36 entries across a marginal, all in one cell in the worst case, so its Euclidean sensitivity is 36. Let $\Delta_2$ denote that sensitivity and $\sigma$ the Gaussian standard deviation. The mechanism is $\rho$-zCDP when
$$\sigma = \frac{\Delta_2}{\sqrt{2\rho}},$$
and that formula gives the 227.68 the query reports.

The other half of the share is divided over the ten rounds. Nine tenths of the 0.005 of a round goes to measuring, and the formula gives 379.47 for the marginal of a round. The threshold of 1364 is the noisy count that a category of `ILOSTAT` must clear to be released. The keys are the four codes in the data plus the null that stands for anything else.

```text
aim release (one run): 13.4 s
aim measured: ('AGE', 'HWUSUAL', 'ILOSTAT') std=379.47
aim measured: ('SEX', 'HWUSUAL', 'ILOSTAT') std=379.47
aim measured: ('SEX', 'AGE', 'HWUSUAL') std=379.47
aim measured: ('SEX', 'AGE', 'HWUSUAL') std=189.74
aim measured: ('SEX', 'AGE', 'ILOSTAT') std=379.47
aim measured: ('SEX', 'HWUSUAL') std=379.47
aim measured: ('AGE', 'HWUSUAL') std=379.47
```

The table's `marginals` attribute lists each measurement, its clique of columns and its noise. Here AIM measured all four three-way marginals before any pair, as the weights predict. It measured one marginal twice, with the second measurement at half the scale. The first changed the model by less than its noise, so the adaptive rule gave the next round four times its share. This exhausted the budget after seven measurements rather than ten. The selected marginals vary across runs; inspect this list after each release.

## A spanning tree of pairs with MST {#sec-synth-mst}

MST, winner of the 2018 NIST synthetic-data contest, is simpler than AIM. After the same one-way step, it scores column pairs by their disagreement with the current model. Noisy maximum selects pairs that connect columns not yet joined. With four columns, the default is three pairs forming a spanning tree. `num_selections` sets the number of pairs; its default is one fewer than the number of columns. MST measures the selected pairs at once with nine tenths of the remaining budget, then fits the model.

```{.python excerpt="synth-mst"}
mst_query = table_query(context, dp.mbi.MST())
table_mst, seconds_mst = timed_release(mst_query)
```

```text
mst: oneway scale=227.6840 threshold=1364
mst: pairs=3  rho per pair=0.0150  scale of a pair=207.8461
mst release (one run): 3.3 s
mst measured: ('HWUSUAL', 'ILOSTAT') std=207.85
mst measured: ('AGE', 'ILOSTAT') std=207.85
mst measured: ('SEX', 'ILOSTAT') std=207.85
mst pairs form a spanning tree: True
```

The one-way step has the same scale and threshold as AIM. The three pairs split nine tenths of the remaining budget, receiving 0.015 each; the scale is 207.85. This is more precise than an AIM round because MST commits to three measurements rather than reserving budget for later rounds. Here the tree is a star centred on `ILOSTAT`, which is most related to the other columns. The script checks that the selected pairs connect all four columns without a cycle.

## A first look at the one-way marginals {#sec-synth-oneway}

Because the data are public, the script compares each synthetic one-way marginal with its real counterpart after binning numeric columns. It reports total variation distance, half the sum of absolute differences between cell shares, the largest cell error in rows and the marginal's standard deviation from `std`.

The last column is a bound. Write $k$ for the number of cells and $n$ for the number of rows. If every cell were off by three standard deviations, the distance would be
$$\frac{3 k \sigma}{2 n}.$$
A well-fitted marginal is usually far below this bound because cell errors have different signs and few reach three standard deviations.

```text
one-way SEX (aim): tv=0.00007 largest cell error=2,813 rows  std=204.6 bound=0.00024
one-way AGE (aim): tv=0.00015 largest cell error=1,547 rows  std=201.1 bound=0.00032
one-way HWUSUAL (aim): tv=0.00200 largest cell error=11,310 rows  std=192.3 bound=0.00030
one-way ILOSTAT (aim): tv=0.00039 largest cell error=1,670 rows  std=792.7 bound=0.00156
one-way SEX (mst): tv=0.00004 largest cell error=2,779 rows  std=204.5 bound=0.00024
one-way AGE (mst): tv=0.00017 largest cell error=1,981 rows  std=204.5 bound=0.00032
one-way HWUSUAL (mst): tv=0.00257 largest cell error=13,620 rows  std=204.5 bound=0.00032
one-way ILOSTAT (mst): tv=0.00040 largest cell error=1,963 rows  std=227.7 bound=0.00045
```

The `SEX` and `AGE` shares fall within their bounds, but their largest cell errors of two to three thousand rows exceed the noise estimates. `HWUSUAL` is inaccurate even as shares: its distance exceeds the bound manyfold in both releases, and one cell differs by more than eleven thousand rows, over fifty standard deviations. Both algorithms measured this marginal at the same scale as the others, so noise does not explain the error. A systematic problem affects the column.

## Nulls and the key set {#sec-synth-nulls}

Every table's model requires a fixed key set for each column. As the documentation explains, keys come from `keys`, `cuts` or stable-key release; values outside the set become null before counting. Cut columns have no null key. Thus the four `HWUSUAL` bins cover its entire key set, although 20,819 real rows have unrecorded hours.

```text
rows: 3,811,932  AGE nulls: 255  HWUSUAL nulls: 20,819
HWUSUAL = 99 (not applicable): 2,294,921
```

In version 0.16.0, nulls follow different paths. For a cut column, the one-way step joins counts to bin labels and omits null rows. The multi-way step assigns a null bin the last index, counting those rows in the bin above 40 hours. The resulting measurements disagree on the row total, so the fitted model compromises. The script compares its one-way `HWUSUAL` marginal with real counts that place nulls in the last bin.

```text
aim model total (one run): 3,806,638.8  real rows 3,811,932
HWUSUAL bins (aim, one run): fitted count against the real count with nulls in the last bin
  (-inf, 1.0]: fitted 8,953  real 6,961  difference +1,992
  (1.0, 20.0]: fitted 121,660  real 119,893  difference +1,767
  (20.0, 40.0]: fitted 1,028,785  real 1,026,527  difference +2,258
  (40.0, inf]: fitted 2,647,240  real 2,658,551  difference -11,311
```

The last bin is eleven thousand rows short, while each other bin is about two thousand rows high. The model's total, estimated from one-way marginals, is five thousand below the real count. The fit has spread the omitted null rows across bins.

The documentation's remedy is to replace true nulls with keys before tabulation, using `select` or `with_columns`. The query's `fill` option does this row by row without changing the privacy unit. The script maps null `HWUSUAL` values to 99, the survey's code for not applicable, and 255 null `AGE` values to zero, the youngest bin. Choosing a replacement is a modelling decision, but both steps must assign every row to the same key.

## Guaranteeing marginals with Sequential {#sec-synth-sequential}

Adaptive selection may omit a marginal the release requires. `Sequential` runs algorithms in order on one table, each using the previous model. It divides the budget equally unless `weights` specifies another split. Placing `Fixed` before AIM guarantees the named marginals, while AIM uses the remaining budget where the model is inaccurate. The third release guarantees `ILOSTAT` by `SEX` and `AGE` pairs and fills the nulls.

```{.python excerpt="synth-sequential"}
sequential = dp.mbi.Sequential(
    algorithms=[
        dp.mbi.Fixed(queries=[dp.mbi.Count(by) for by in GUARANTEED]),  # these two marginals, whatever happens
        dp.mbi.AIM(rounds=ROUNDS),  # then AIM chooses the rest
    ],
)
sequential_query = table_query(context, sequential, fill=True)
table_seq, seconds_seq = timed_release(sequential_query)
```

```text
sequential: rho per guaranteed marginal=0.0125  scale=227.6840
sequential: rho per AIM round=0.0025  scale of a round's marginal=536.6563
sequential release (one run): 12.1 s
sequential model total (one run): 3,811,712.7  real rows 3,811,932
sequential measured: ('SEX', 'ILOSTAT') std=227.68
sequential measured: ('AGE', 'ILOSTAT') std=227.68
sequential measured: ('SEX', 'HWUSUAL', 'ILOSTAT') std=536.66
sequential measured: ('SEX', 'AGE', 'HWUSUAL') std=536.66
sequential measured: ('SEX', 'AGE', 'HWUSUAL') std=268.33
sequential measured: ('SEX', 'AGE', 'ILOSTAT') std=536.66
sequential measured: ('HWUSUAL', 'ILOSTAT') std=536.66
sequential measured: ('HWUSUAL', 'ILOSTAT') std=536.66
sequential measured: ('SEX', 'HWUSUAL') std=536.66
sequential guaranteed marginals present: True
```

The one-way step again receives half the budget, and the remainder is divided equally between the two algorithms. `Fixed` splits its quarter between two marginals, each with scale 227.68. AIM uses the other quarter for ten rounds, so each marginal has more noise than in the first release: 536.66 rather than 379.47. This is the cost of the guarantee; `weights` can change the allocation. The model's total is now 219 rows from truth, within one standard deviation of a one-way measurement, confirming that null handling caused the earlier discrepancy.

```text
one-way SEX (sequential): tv=0.00004 largest cell error=240 rows  std=198.2 bound=0.00023
one-way AGE (sequential): tv=0.00006 largest cell error=233 rows  std=200.7 bound=0.00032
one-way HWUSUAL (sequential): tv=0.00009 largest cell error=273 rows  std=206.2 bound=0.00032
one-way ILOSTAT (sequential): tv=0.00008 largest cell error=360 rows  std=271.5 bound=0.00053
```

Every distance is at least three times below its bound, and no cell differs by more than 360 rows, less than two standard deviations. The largest `HWUSUAL` error fell from eleven thousand rows to 273.

## Generating the rows {#sec-synth-rows}

`ContingencyTable` stores the Private-PGM model fitted in the previous chapter: a graphical model whose marginals jointly approximate the noisy measurements. `synthesize()` draws a frame from this model. It uses only released values, so repeated calls are post-processing and spend no additional budget.

```text
synthesize (aim, one run): shape (3806638, 4)  model total 3,806,638.8  11.9 s
┌─────┬───────────┬──────────┬─────────┐
│ SEX ┆ AGE       ┆ HWUSUAL  ┆ ILOSTAT │
│ --- ┆ ---       ┆ ---      ┆ ---     │
│ i64 ┆ f64       ┆ f64      ┆ i64     │
╞═════╪═══════════╪══════════╪═════════╡
│ 1   ┆ 19.813324 ┆ 0.482324 ┆ 1       │
│ 1   ┆ 19.500078 ┆ 0.94265  ┆ 1       │
│ 1   ┆ 19.338927 ┆ 0.741979 ┆ 9       │
│ 1   ┆ 19.031755 ┆ 0.118982 ┆ 9       │
│ 1   ┆ 19.232563 ┆ 0.676039 ┆ 1       │
└─────┴───────────┴──────────┴─────────┘
synthetic AGE from 19.00 to 61.00  HWUSUAL from 0.00 to 41.00
synthetic dtypes: {'SEX': Int64, 'AGE': Float64, 'HWUSUAL': Float64, 'ILOSTAT': Int64}
synthetic SEX values: [1, 2]  ILOSTAT values: [1, 2, 3, 9]
```

Without `rows`, the frame size is the model's noisy estimate of the row count. The columns retain their original types, and `SEX` and `ILOSTAT` use their key values. By default, `method="round"` scales expected counts to the requested row count and rounds randomly, so each synthetic marginal matches the model within one row per cell. `method="sample"` instead draws rows independently, adding sampling noise of about the square root of each count. Use it when rows should behave like a sample.

```{.python excerpt="synth-sample"}
sample = table_aim.synthesize(rows=SAMPLE_ROWS, method="sample")  # draws rows independently from the model
```

```text
synthesize rows=100,000 method=sample: shape (100000, 4)
```

The first rows have `AGE` just below 20 because the model stores bins, not original values. For cut columns, `synthesize()` draws uniformly within each bin and within one unit of the edge for open-ended bins. Synthetic ages therefore range from 19 to 61. Rows coded 99 receive hours just above 40.

The frame's mean age reflects the bins and sampling rule, not the original data. Use finer cuts when numeric values matter, accepting more cells per marginal and more noise per cell ([Evaluating and tuning synthetic data](18-evaluating-synthetic-data.md)).

The script seeds NumPy's global generator, used for rounding and within-bin draws. Thus the frame is deterministic given the released table.

## The privacy statement {#sec-synth-statement}

The context has spent all three shares. The script builds the release statement from its variables.

```text
budget spent: (0.3, 3e-07)
rho=0.3 at delta=1e-07: epsilon=4.2709
whole context converted: epsilon=4.2709 delta=4e-07
```

Report a $\rho$ budget in $\varepsilon$ using the conversion in [Assessing utility and choosing the noise](../part02-tabular-data/07-utility-and-noise.md). `make_zCDP_to_approxDP` and `make_fix_delta` convert the budget of 0.3 to $\varepsilon = 4.27$ with an additional $\delta$ of $10^{-7}$. Applying both combinators to the context's `accountant` gives the same $\varepsilon$ and total $\delta$: three parts for key selection and one for conversion. The resulting $\varepsilon = 4.27$ guarantee for the three adaptive releases is loose.

```{.python excerpt="synth-statement"}
print(
    f"This release of three contingency tables and the synthetic rows drawn from them satisfies "
    f"(rho = {RHO:g}, delta = {DELTA:g})-zero-concentrated differential privacy, which is "
    f"(epsilon = {epsilon:.2f}, delta = {total_delta:g})-differential privacy, "
    f"for adding or removing all of one person's survey responses, at most {CONTRIBUTIONS} of them, "
    f"under the central model, using OpenDP {dp.__version__} with Gaussian noise. "
    f"The keys of SEX and the cuts of AGE and HWUSUAL were fixed in advance; the keys of ILOSTAT were released "
    f"with a threshold of {oneway_threshold}. Generating the rows is post-processing and spent nothing."
)
```

Post-processing extends the guarantee to the synthetic rows without increasing privacy loss. The statement also records which keys and cuts were public; choosing cuts from private data would leak information. The 36-contribution privacy unit inflates every scale by a factor of 36. [Identifier truncation and bounds](../part02-tabular-data/05-identifier-truncation.md) shows that few people contribute often, so the script builds—but does not release—the same query with an identifier unit and a four-row contribution bound.

```{.python excerpt="synth-identifier"}
context_id = dp.Context.compositor(
    data=frame,
    privacy_unit=dp.unit_of(contributions=1, identifier="PIDENT"),  # neighbours add or remove one person
    privacy_loss=dp.loss_of(rho=RHO_QUERY, delta=DELTA_QUERY),
)
query_id = (
    context_id.query(rho=RHO_QUERY, delta=DELTA_QUERY)
    .truncate_per_group(TRUNCATE)  # at most 4 rows per person, so a person moves a count by at most 4
    .select(*COLUMNS)
    .contingency_table(keys=KEYS, cuts=CUTS, algorithm=dp.mbi.AIM(rounds=ROUNDS))
)
```

```text
identifier unit, 4 rows per person: oneway scale=25.2982 threshold=142
identifier unit: scale ratio 9.00 = 36/4
```

The one-way scale falls nine-fold to 25.30 because each person can now change a count by at most 4 rows. The threshold falls from 1364 to 142, and all other release scales shrink by the same factor. This requires dropping every contribution after a person's fourth row.

## The measured marginals against the real ones {#sec-synth-check}

The final check compares each measured multi-way marginal with its real counterpart, as in the one-way check. These should be the closest marginals because each release spent budget measuring them. The results are from the same run as above.

```text
measured (AGE, HWUSUAL) (aim): tv=0.00200 largest cell error=2,992 rows  std=224.1 bound=0.00141
measured (SEX, HWUSUAL) (aim): tv=0.00200 largest cell error=5,730 rows  std=242.4 bound=0.00114
measured (AGE, HWUSUAL, ILOSTAT) (aim): tv=0.00219 largest cell error=1,382 rows  std=379.5 bound=0.01195
measured (SEX, AGE, HWUSUAL) (aim): tv=0.00200 largest cell error=1,556 rows  std=169.7 bound=0.00321
measured (SEX, AGE, ILOSTAT) (aim): tv=0.00093 largest cell error=699 rows  std=379.5 bound=0.00896
measured (SEX, HWUSUAL, ILOSTAT) (aim): tv=0.00200 largest cell error=1,940 rows  std=379.5 bound=0.00896
measured (AGE, ILOSTAT) (mst): tv=0.00116 largest cell error=2,366 rows  std=207.8 bound=0.00164
measured (HWUSUAL, ILOSTAT) (mst): tv=0.00258 largest cell error=3,700 rows  std=207.8 bound=0.00164
measured (SEX, ILOSTAT) (mst): tv=0.00041 largest cell error=1,085 rows  std=207.8 bound=0.00123
measured (AGE, ILOSTAT) (sequential): tv=0.00027 largest cell error=387 rows  std=221.1 bound=0.00174
measured (HWUSUAL, ILOSTAT) (sequential): tv=0.00068 largest cell error=1,035 rows  std=351.3 bound=0.00276
measured (SEX, HWUSUAL) (sequential): tv=0.00017 largest cell error=338 rows  std=342.9 bound=0.00162
measured (SEX, ILOSTAT) (sequential): tv=0.00019 largest cell error=335 rows  std=218.1 bound=0.00129
measured (SEX, AGE, HWUSUAL) (sequential): tv=0.00094 largest cell error=712 rows  std=240.0 bound=0.00453
measured (SEX, AGE, ILOSTAT) (sequential): tv=0.00099 largest cell error=751 rows  std=536.7 bound=0.01267
measured (SEX, HWUSUAL, ILOSTAT) (sequential): tv=0.00073 largest cell error=567 rows  std=536.7 bound=0.01267
```

In the two releases without null filling, every marginal containing `HWUSUAL` exceeds its bound. Pairs without that column also have larger errors than in the third release because the incorrect total affects every marginal. After null filling, each measured marginal is at least four times below its bound. The bound assumes every cell is three standard deviations from truth at once, an unlikely event; a three-way distance near one thousandth therefore indicates a good fit.

This check does not assess unmeasured queries or models trained on synthetic rows. [Evaluating and tuning synthetic data](18-evaluating-synthetic-data.md) tests both and examines the effects of budget, cuts and rounds.

## Summary

| Call or concept | What it does | Rule of thumb |
| --- | --- | --- |
| `dp.mbi.AIM(rounds=10)` | One-way marginals, then rounds that measure the marginal the model fits worst | Fewer rounds for a quick release; read `table.marginals` to see what it chose |
| `queries`, `measure_split`, `max_size` | The candidates or a weighted workload; the share of a round spent measuring; the memory cap | The defaults serve a handful of columns |
| `dp.mbi.MST()` | One-way marginals, then pairs that form a spanning tree, measured at once | Faster than AIM when pairs are what matter; it never measures a triple |
| `dp.mbi.Sequential(algorithms=[Fixed, AIM])` | Runs the algorithms in turn on one table, splitting the budget evenly or by `weights` | Put the marginals you must have in `Fixed`; AIM spends the rest |
| `query.oneway_scale`, `query.oneway_threshold` | The one-way step's noise scale and key threshold, before anything is spent | Under the identifier unit, 4 rows per person cost a ninth of 36 |
| `table.synthesize(rows, method)` | Rows drawn from the fitted model; `round` matches its marginals, `sample` draws independently | Post-processing, so free; the row count defaults to the model's noisy total |
| Cut columns in the synthetic frame | Values drawn uniformly inside the bin, within one unit of the edge for the open bins | Right as bins, wrong as numbers; give finer cuts if the numbers matter |
| Nulls in a cut column | Dropped by the one-way step and counted in the last bin by the multi-way step in 0.16.0 | Fill them with a public code in `with_columns` before the table |
| Total variation distance against `std` | How far a synthetic marginal is from the real one, against three standard deviations per cell | A measured marginal over the bound points to the data, not the noise |

: The calls and concepts of synthetic data with AIM and MST, what each does, and the rule of thumb that goes with it. {#tbl-synth-summary}

## References

- OpenDP documentation: "Synthetic Data" in the statistical modelling guide, "Contingency Tables" in the tabular data guide, and the API reference for `opendp.extras.mbi` (`AIM`, `MST`, `Sequential`, `Fixed`, `ContingencyTable`).
- McKenna, Mullins, Sheldon and Miklau, "AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data" (2022): the selection score, the budget schedule and the experiments behind the defaults.
- McKenna, Miklau and Sheldon, "Winning the NIST Contest: A scalable and general approach to differentially private synthetic data" (2021): the MST mechanism and the Private-PGM engine that fits the model.
- McKenna, Sheldon and Miklau, "Graphical-model based estimation and inference for differential privacy" (2019): Private-PGM itself, the inference step that both algorithms share.
- The full edition's chapters on [selection](../../book-full/part03-dp-foundations/05-selection.md), [query release](../../book-full/part03-dp-foundations/06-query-release.md) and [concentrated DP](../../book-full/part04-privacy-accounting/03-concentrated-dp.md) derive the results this chapter states.
