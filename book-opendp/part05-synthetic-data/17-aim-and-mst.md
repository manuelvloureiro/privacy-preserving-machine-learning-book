# Synthetic data with AIM and MST

[Contingency tables](16-contingency-tables.md) released the marginals you asked for, which works when you know the questions in advance. The analyst of a synthetic dataset usually does not, and a workload that guesses wrong leaves the relations they care about to chance. So this chapter lets the library choose. The **AIM** algorithm picks its marginals one round at a time, each time the one that its current model gets most wrong. The **MST** algorithm picks a spanning tree of column pairs in one step. Both fit a model to the noisy marginals, and `synthesize()` turns the model into rows, which is post-processing and costs nothing.

The chapter follows the documentation's synthetic-data page and the docstrings of `dp.mbi.AIM`, `dp.mbi.MST`, `dp.mbi.Sequential` and `ContingencyTable.synthesize`. You will release three tables over four survey columns and read which marginals each algorithm chose. Then you will generate a synthetic frame of 3.8 million rows, state the guarantee, and compare the synthetic marginals with the real ones. That comparison shows a problem with nulls, which you will fix.

The script [opendp_aim_and_mst.py](../../code/part05-synthetic-data/opendp_aim_and_mst.py) runs everything on the Labour Force Survey (LFS) sample with OpenDP 0.16.0 and `mbi` 1.1.0, the Private-PGM version that OpenDP pins, with JAX on the CPU. The noise scales, the threshold, the budget arithmetic, the conversion to $\varepsilon$ and the facts about the data are the same on every run. OpenDP's noise cannot be seeded, so the chosen marginals, the counts, the rows and the distances to the real data change. The page shows them in blocks marked as one run. The whole script takes about 75 seconds.

```bash
uv run python code/part05-synthetic-data/opendp_aim_and_mst.py
```

## When you do not know the questions {#sec-synth-adaptive}

A fixed workload spends the budget on the marginals you name and on nothing else. For a pair of columns you did not name, the model answers as if the two columns were independent, as `SEX` came out uniform in the previous chapter. With four columns you could name every pair and every triple. But the budget would then spread over ten marginals, most of which the model fits well anyway. The question is how to spend the budget where the model is wrong.

AIM answers it adaptively. It first measures the one-way marginal of every column with half the budget, which also releases the keys of a column without keys. Then it fits a model and runs rounds. Each round scores every candidate marginal of up to three columns by how far the model's version is from the true one, minus what measuring it would cost in noise. Candidates that overlap the workload weigh more, so the three-way marginals, which are the workload itself, weigh most.

A noisy selection with a tenth of the round's budget picks one candidate, by the report-noisy-max form of the exponential mechanism ([the full edition's chapter on selection](../../book-full/part03-dp-foundations/05-selection.md)). The other nine tenths measure the chosen marginal with Gaussian noise, and AIM fits the model again.

The budget per round adapts. It starts at the budget divided by the number of rounds. When a measurement moves the model by less than its noise, the round taught little, so the next round gets four times the share. The rounds stop when a privacy filter finds the budget spent. The `rounds` argument sets the number of planned rounds, by default sixteen per column, which is 64 here, and the script uses 10 so that a release takes seconds. The `max_size` argument caps the memory of the model in megabytes, which never binds with four small columns.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
aim = dp.mbi.AIM()
print(aim.queries, aim.measure_split, aim.max_size, aim.rounds)  # 3 0.9 80.0 None
print(dp.mbi.MST().num_selections, dp.mbi.MST().measure_split)  # None 0.9
print(dp.mbi.Count(("SEX", "ILOSTAT")))  # Count(by=('SEX', 'ILOSTAT'), weight=1.0)
```

The `queries` argument is either the largest number of columns of a candidate, 3 by default, or a workload of weighted `dp.mbi.Count` objects. The `measure_split` of 0.9 is the share of each round that goes to measuring. MST has the same split and one argument of its own, `num_selections` (@sec-synth-mst).

## The AIM release {#sec-synth-aim}

The context is the documentation's, with three times the budget because this chapter releases three tables. The privacy unit is 36 contributions per person, as in [the data and the context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context). The budget is a privacy parameter $\rho$ of 0.3 under zero-concentrated differential privacy (zCDP), with Gaussian noise. The parameter bounds the privacy loss on average ([the full edition's chapter on concentrated differential privacy (DP)](../../book-full/part04-privacy-accounting/03-concentrated-dp.md)). The slack $\delta$ of $3 \times 10^{-7}$ pays for releasing the keys of `ILOSTAT`, which the query does not name.

```{.python excerpt="synth-context"}
def make_context(frame: pl.LazyFrame) -> dp.Context:
    """The context: a privacy unit of 36 rows, a budget of rho = 0.3 with delta = 3e-7, one share per release."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),  # neighbours add or remove one person's 36 rows
        privacy_loss=dp.loss_of(rho=RHO, delta=DELTA),  # zCDP with Gaussian noise; delta pays for ILOSTAT's keys
    )
```

Every release goes through one function, because the three queries differ only in the algorithm. The query takes a third of the budget, $\rho = 0.1$ with $\delta = 10^{-7}$. It selects the documentation's four columns, gives the keys of `SEX` and the cuts of `AGE` and `HWUSUAL`, and leaves `ILOSTAT` to the threshold. The `fill` argument is for later.

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

The first release uses AIM with ten rounds. Before the release, the query reports the scale and the threshold of its one-way step, which cost nothing to read.

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

The scales follow from the budget split. Half of the query's $\rho$ goes to the one-way step, where AIM measures all four one-way marginals, so each gets 0.0125. One person changes the cells of a marginal by at most 36 rows in all, possibly all in one cell. So the sensitivity in the Euclidean norm is 36. Write $\Delta_2$ for that sensitivity and $\sigma$ for the standard deviation of the Gaussian noise. The mechanism is $\rho$-zCDP when
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

The `marginals` attribute of the table holds every measurement with its clique, the columns it spans, and its noise. In this run AIM measured all four three-way marginals before any pair, as the weights predict. One of them appears twice, the second time at half the scale, which is the adaptive rule at work. A round had moved the model by less than its noise, so the next round took four times the share, which was all that was left. So there were seven measurements and not ten. Another run chooses differently, so read this list after every release.

## A spanning tree of pairs with MST {#sec-synth-mst}

MST, the mechanism that won the 2018 NIST synthetic-data contest, is simpler than AIM. After the same one-way step it scores only pairs of columns, by how far the model's version is from the true one. Then it picks pairs one at a time by noisy maximum, each time among the pairs that join two columns not yet connected. With four columns that gives three pairs, which form a tree that spans every column, hence the name. The `num_selections` argument sets how many pairs to pick, by default one fewer than the number of columns. Then MST measures all the selected pairs at once with nine tenths of the remaining budget, and fits the model.

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

The one-way step is identical, so its scale and threshold are the same. The three pairs share nine tenths of the remaining budget, 0.015 each, and the formula gives 207.85. So each pair is measured more precisely than a round of AIM, because MST commits to three measurements where AIM keeps budget back for rounds it may need. In this run the tree is a star on `ILOSTAT`, the column most related to the other three. The script checks that the pairs connect all four columns without a cycle, as the selection guarantees.

## A first look at the one-way marginals {#sec-synth-oneway}

The data are public, so the script can compare each one-way marginal of the synthetic rows, with their numeric columns put back into bins, with the real one. It reports the total variation distance, which is half the sum of the absolute differences of the shares of the two marginals. It also reports the largest error of any cell in rows, and the standard deviation of that marginal from `std`.

The last column is a bound. Write $k$ for the number of cells of the marginal and $n$ for the number of rows. If every cell were off by three standard deviations at once, the distance would be
$$\frac{3 k \sigma}{2 n}.$$
A well-fitted marginal stays far under that, because the cell errors have different signs and few reach three standard deviations.

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

`SEX` and `AGE` are right as shares, with distances below their bounds. But their largest cell errors of two or three thousand rows are more than the noise explains. `HWUSUAL` is wrong even as shares. Its distance is many times its bound in both releases, and one cell is off by more than eleven thousand rows, over fifty standard deviations. Both algorithms measured that marginal at the same scale as the others, so noise cannot explain it, and something systematic is wrong with the column.

## Nulls and the key set {#sec-synth-nulls}

The model behind every table needs a fixed set of keys for every column, which the documentation's section on nulls explains. The keys come from you, through `keys` and `cuts`, or from the stable-key release. Either way, every value outside the set is replaced by null before counting. A cut column has no null key, so the four bins of `HWUSUAL` are its whole key set, and the column has 20,819 real nulls, hours that were not recorded.

```text
rows: 3,811,932  AGE nulls: 255  HWUSUAL nulls: 20,819
HWUSUAL = 99 (not applicable): 2,294,921
```

In version 0.16.0 those nulls take two different paths. The one-way step counts a cut column by joining its counts onto the bin labels, so it does not count a null row at all. The multi-way step gives a row whose bin is null the last index, so it counts that row in the bin above 40 hours. So the model is fitted to measurements that disagree about the number of rows, and it compromises. The script shows the fitted one-way marginal of `HWUSUAL` against the real counts with the nulls in the last bin.

```text
aim model total (one run): 3,806,638.8  real rows 3,811,932
HWUSUAL bins (aim, one run): fitted count against the real count with nulls in the last bin
  (-inf, 1.0]: fitted 8,953  real 6,961  difference +1,992
  (1.0, 20.0]: fitted 121,660  real 119,893  difference +1,767
  (20.0, 40.0]: fitted 1,028,785  real 1,026,527  difference +2,258
  (40.0, inf]: fitted 2,647,240  real 2,658,551  difference -11,311
```

The last bin is eleven thousand rows short and every other bin is two thousand rows over. The total of the model, which it estimates from the one-way marginals, is five thousand rows below the real count. Those are the missing nulls, spread by the fit.

The documentation's remedy is to turn true nulls into a key before the table, with `select` or `with_columns`, which is what the `fill` argument of the query does. It is a row-by-row transformation inside the query, so it changes nothing about the privacy unit. The script sends the nulls of `HWUSUAL` to 99, the survey's code for not applicable. It sends the 255 nulls of `AGE` to zero, the youngest bin. The code a null gets is a modelling decision, but every row must land in a key, the same way in both steps.

## Guaranteeing marginals with Sequential {#sec-synth-sequential}

Adaptive selection has a cost, because AIM may not measure a marginal that the release must contain. `Sequential` runs a list of algorithms one after another on the same table, each from the model of the previous one. It splits the budget evenly between them unless `weights` says otherwise. So with a `Fixed` workload first and AIM second, the named marginals are always measured, and AIM spends the rest where the model is still wrong. The third release does this with the pairs of `ILOSTAT` with `SEX` and with `AGE`, and it fills the nulls.

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

The one-way step again takes half, and the rest is split evenly between the two algorithms. `Fixed` divides its quarter between its two marginals, at the familiar 227.68. AIM runs its ten rounds on the other quarter, so each of its marginals is noisier than in the first release, 536.66 against 379.47. That is the price of the guarantee, and `weights` can shift it either way. The total of the model is now 219 rows from the truth, within one standard deviation of a one-way measurement, so the nulls were the problem.

```text
one-way SEX (sequential): tv=0.00004 largest cell error=240 rows  std=198.2 bound=0.00023
one-way AGE (sequential): tv=0.00006 largest cell error=233 rows  std=200.7 bound=0.00032
one-way HWUSUAL (sequential): tv=0.00009 largest cell error=273 rows  std=206.2 bound=0.00032
one-way ILOSTAT (sequential): tv=0.00008 largest cell error=360 rows  std=271.5 bound=0.00053
```

Every distance is under its bound by a factor of three or more, and no cell is more than 360 rows off, under two standard deviations. The largest error of `HWUSUAL` fell from eleven thousand rows to 273.

## Generating the rows {#sec-synth-rows}

A `ContingencyTable` is a model, and `synthesize()` turns it into a frame. The model is the Private-PGM fit of the previous chapter, a graphical model over the four columns whose marginals match the noisy measurements as closely as possible at once. Drawing rows from it looks only at released values, so it is post-processing and spends nothing, however often you call it.

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

Without a `rows` argument the frame has as many rows as the total of the model, a noisy estimate of the row count. The four columns keep their original types, and `SEX` and `ILOSTAT` take values from their keys. The default `method="round"` scales the expected counts of the model to the number of rows and rounds them at random. So the synthetic marginals match those of the model to within a row per cell. The alternative, `method="sample"`, draws each row independently, which adds sampling noise of about the square root of each count, and suits rows that should behave like a sample.

```{.python excerpt="synth-sample"}
sample = table_aim.synthesize(rows=SAMPLE_ROWS, method="sample")  # draws rows independently from the model
```

```text
synthesize rows=100,000 method=sample: shape (100000, 4)
```

Look again at the numeric columns, because the first rows all show an `AGE` just under 20. The model knows only which bin a row is in, so for a cut column `synthesize()` draws a value uniformly inside the bin, or within one unit of the edge for the two open-ended bins. Every synthetic age therefore lies between 19 and 61. The rows coded 99 all become hours just above 40.

So a mean age from this frame reflects the bins and the sampling rule, not the data. If the numbers matter, give finer cuts, at the cost of more cells per marginal and more noise per cell ([Evaluating and tuning synthetic data](18-evaluating-synthetic-data.md)).

The script seeds NumPy's global generator, which the rounding and the draws inside a bin use, so the frame is a function of the released table.

## The privacy statement {#sec-synth-statement}

The context has spent its three shares, and the script writes the statement of the whole release from its variables.

```text
budget spent: (0.3, 3e-07)
rho=0.3 at delta=1e-07: epsilon=4.2709
whole context converted: epsilon=4.2709 delta=4e-07
```

A budget in $\rho$ is stated to readers in $\varepsilon$, with the conversion of [Assessing utility and choosing the noise](../part02-tabular-data/07-utility-and-noise.md). The combinators `make_zCDP_to_approxDP` and `make_fix_delta` convert the budget of 0.3 to $\varepsilon = 4.27$, at a further $\delta$ of $10^{-7}$. The same two combinators on the `accountant` of the context give the same $\varepsilon$ with the total $\delta$, three parts for the keys and one for the conversion. So three adaptive releases cost $\varepsilon = 4.27$, which is a loose guarantee.

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

The statement covers the synthetic rows, because post-processing adds nothing to the loss. It names what was public, since a cut chosen by looking at the data would leak it. The unit of 36 contributions is the expensive part, because every scale above carries the factor 36. But [Identifier truncation and bounds](../part02-tabular-data/05-identifier-truncation.md) showed that hardly anyone answers more than a few times. So the script builds the same query under the identifier unit with a truncation to four rows per person, to read the scale, and does not release it.

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

The one-way scale falls nine-fold, to 25.30, because a person now moves a count by at most 4 rows. The threshold falls from 1364 to 142, and every other scale of the release carries the same factor. The cost is the truncation, which drops every row beyond the fourth of anyone who answered more often.

## The measured marginals against the real ones {#sec-synth-check}

The last check compares every measured multi-way marginal of the three releases with the real one, as for the one-way marginals. Each release spent its budget on its measured marginals, so the synthetic data should be closest to the truth there. The lines are from the same run as the lists above.

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

Two things stand out. In the two releases that left the nulls alone, every marginal with `HWUSUAL` breaks its bound. The pairs without it are also further from the truth than in the third release, because the wrong total leaks into every marginal. In the release that filled the nulls, every measured marginal is under its bound by a factor of four or more. The bound counts three standard deviations in every cell at once, which never happens, so a three-way marginal with a distance of one thousandth is a good fit.

This check does not say how the synthetic data answers a question that nobody measured, or how a model trained on it compares with one trained on the real rows. [Evaluating and tuning synthetic data](18-evaluating-synthetic-data.md) asks those questions, and shows what changes with the budget, the cuts or the number of rounds.

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
