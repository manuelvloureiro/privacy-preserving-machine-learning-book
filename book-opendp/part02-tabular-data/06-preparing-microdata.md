# Preparing microdata

Survey data rarely arrive in the form a statistic requires. Work hours are floats, 99 means "not applicable," and null means no answer. Response dates are split across two integer columns. Preparation may require casting, filling, clipping, filtering or recoding. This chapter shows where to perform those steps. OpenDP inspects the full lazy query and accounts for each step, so prepare data inside the query to keep that work within the guarantee.

The chapter follows the documentation's tutorial on preparing microdata and its guides to data types and expressions. It describes the supported Polars contexts and expressions, OpenDP's changes to them, and its treatment of nulls, undefined floats and bounds. The final example prepares Labour Force Survey (LFS) data to estimate the mean difference between actual and usual work hours.

The script [opendp_preparing_microdata.py](../../code/part02-tabular-data/opendp_preparing_microdata.py) runs every step on OpenDP 0.16.0 with Polars 1.36.1. The noise scales, error bars, thresholds, error messages and true values that it prints are the same on every run. The released counts and mean are not, because OpenDP's noise cannot be seeded, so this page shows them in one block marked as one run.

```bash
uv run python code/part02-tabular-data/opendp_preparing_microdata.py
```

## Prepare inside the query {#sec-prep-inside}

A context on a Polars `LazyFrame` reads only its schema, which OpenDP treats as public, and accesses the data only at release. Before execution, OpenDP parses the query plan into a chain of transformations, as [Transformations](../part03-framework/09-transformations.md) explains. It bounds each person's effect at every step, so filters and fills in the plan are included in the accounting.

If you prepare data before creating the context, OpenDP sees only the prepared frame and guarantees privacy for that frame. A fixed row-wise function, such as renaming a column or parsing a date with a known format, preserves each person's rows. The guarantee still applies, but you must justify that claim because OpenDP does not check the preparation. Data-dependent steps are unsafe: imputing with a column mean, bounding by its maximum or filtering at a quantile carries private information into the release outside the accounting. Before creating the context, use only public facts such as the schema, codebook and published sample-size bound.

The context uses the privacy unit and row-count bound from [The data and the context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context): at most 36 rows per person. It splits an approximate differential privacy budget of $\varepsilon=1$ and $\delta=10^{-7}$ evenly over four queries.

```{.python excerpt="prep-context"}
def make_context(frame: pl.LazyFrame) -> dp.Context:
    """The documentation's context: 36 rows per person, (epsilon = 1, delta = 1e-7) split evenly over 4 queries."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(epsilon=EPSILON, delta=DELTA),
        split_evenly_over=QUERIES,
        margins=[dp.polars.Margin(max_length=MAX_LENGTH)],  # public: a bound on the number of rows
    )
```

```text
privacy unit: (SymmetricDistance(), 36)
privacy loss: (Approximate(MaxDivergence), (1.0, 1e-07))
budget per query: epsilon=0.25 delta=2.5e-08
schema: PIDENT=Int64  YEAR=Int64  QUARTER=Int64  SEX=Int64  AGE=Float64  HWUSUAL=Float64  HWACTUAL=Float64  ILOSTAT=Int64
```

At this point, the context knows only the schema and assumes that every column can contain nulls.

## Columns, filters and groups {#sec-prep-contexts}

Polars evaluates expressions in five contexts, which OpenDP divides into two groups. Expressions in `.select` and `.agg` may change row count or order, as aggregations do. Expressions in `.with_columns`, `.filter` and `.group_by` must operate **row by row**. This preserves the privacy unit: each person's 36 input rows remain their own 36 output rows. Sorting or shifting could mix rows from different people, making one person's effect impossible to bound.

The documentation first adds a column with `.with_columns` and groups by it. The row-wise `cut` expression bins usual hours at fixed breaks and replaces `HWUSUAL` with its bin. Because the labels depend only on the breaks, they are public keys.

```{.python excerpt="prep-cut"}
def binned_counts(context: dp.Context) -> LazyFrameQuery:
    """How many responses fall in each band of usual hours: a row-by-row `cut`, then a count per bin."""
    labels = pl.Series("HWUSUAL", cut_labels(BREAKS, left_closed=True), dtype=pl.Categorical)
    return (
        lazy_query(context)
        .with_columns(pl.col("HWUSUAL").cut(breaks=BREAKS, left_closed=True))  # shadows HWUSUAL with its bin
        .group_by("HWUSUAL")
        .agg(dp.len(signed=True))
        .with_keys(pl.LazyFrame([labels]))  # the bin labels are public, so no threshold is needed
    )
```

```text
binned counts: len Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9
without keys: len Frame Length, Integer Laplace, scale=144.00 threshold=2,973
true bins: null=20,819  [-inf, 0)=0  [0, 20)=89,495  [20, 40)=908,061  [40, 60)=406,947  [60, 80)=78,469  [80, 98)=13,220  [98, inf)=2,294,921
```

Scale is 36 rows per person divided by the query's budget share. With public keys, every bin is released, including the empty bin below zero, whose signed count may be negative. Without public keys, OpenDP must determine from the data which bins exist, so it applies a threshold as described in [Grouping](04-grouping.md). Null `HWUSUAL` values form a null bin absent from the key list and are not released. A margin on `HWUSUAL` also applies only to the original column, not the new bin column.

`.filter` removes rows using a boolean row-wise expression, such as the documentation's filter for code 99 before a sum. A filter preserves most margin information but invalidates invariants about keys and group sizes. It also does not prove which values remain: a column filtered to non-null values is still nullable in its domain.

```text
filtered sum: HWUSUAL Sum, Integer Laplace, scale=11,520.00 95% accuracy=34,511.3
```

The tutorial's final context applies `.group_by` and `.agg` *before* private aggregation. In its sample-and-aggregate example, rows enter 1,000 groups by `PIDENT`, and the query releases the mean of each group's minimum usual hours. The inner `.agg` accepts any expression, but privacy adds noise. The script compares mean scales under the current context and with a `Bound` from [Identifier truncation and bounds](05-identifier-truncation.md) that assigns each person's rows to one group.

```text
sample and aggregate: HWUSUAL Sum, Integer Laplace, scale=17,280.00
sample and aggregate: HWUSUAL Length, Integer Laplace, scale=576.00
one group per person: HWUSUAL Sum, Integer Laplace, scale=480.00
one group per person: HWUSUAL Length, Integer Laplace, scale=16.00
```

The first context does not use `PIDENT` to link a person's rows, so it allows 36 changed groups. The inner aggregation doubles noise even with the bound: a changed group counts as one value removed and another added.

## The types OpenDP tracks {#sec-prep-types}

OpenDP assumes columns may contain nulls until an expression proves otherwise. It also tracks **domain descriptors**, facts about values that later aggregations need. Numeric descriptors include lower and upper bounds, set by `.clip` and removed by some later steps. OpenDP supports signed integers and unsigned 32- and 64-bit integers, but rejects `UInt8`. Float descriptors also record whether values may be NaN, an undefined numeric value; a float sum therefore needs both `.fill_null` and `.fill_nan`. Booleans and strings have no descriptors.

Categorical columns store integer codes and a label table. Two operations can make the table data-dependent. An expression such as `.fill_null` may add or remove categories, causing Polars to rewrite codes and emit a data-dependent warning. Also, codes follow the values' order of first appearance, so grouping by them can reveal row order. OpenDP rejects both cases. An `Enum` fixes categories in the schema, avoiding these problems; `cut` is also safe because its encoding depends only on fixed breaks.

```python
import polars as pl
import opendp.prelude as dp

dp.enable_features("contrib")
status = ["working", "absent", "other"] * 100
unit, loss = dp.unit_of(contributions=1), dp.loss_of(epsilon=1.0, delta=1e-7)


def context_for(column: pl.Series) -> dp.Context:
    return dp.Context.compositor(data=pl.LazyFrame([column]), privacy_unit=unit, privacy_loss=loss, split_evenly_over=1)


categorical = context_for(pl.Series("status", status, dtype=pl.Categorical))
try:
    categorical.query().group_by("status").agg(dp.len(signed=True)).summarize()
except dp.OpenDPException as err:
    print(err.message)  # Categories are data-dependent, which may reveal sensitive record ordering. Cast status to string before grouping.
enum = context_for(pl.Series("status", status, dtype=pl.Enum(["working", "absent", "other"])))
print(f"{enum.query().group_by('status').agg(dp.len(signed=True)).summarize()['scale'].item():.4f}")  # 1.0000
```

The error names the fix, a cast to string, and the enum query goes through. Dates, datetimes and times are also supported, with one time zone per datetime column.

## Casting, filling and clipping {#sec-prep-fill}

The documentation recommends casting because floating-point sums incur rounding error. OpenDP accounts for that error, which can increase the noise scale for many rows and wide bounds. The script compares both sum types.

```{.python excerpt="prep-cast"}
def hours_sum(context: dp.Context, as_integers: bool) -> pl.DataFrame:
    """The noise scale of a sum of usual hours, on the float column or after casting it to integers."""
    usual: Any = pl.col("HWUSUAL")  # typed Any: OpenDP adds the `.dp` namespace to expressions at run time
    column = usual.cast(int) if as_integers else usual
    return lazy_query(context).select(column.dp.sum(HOUR_BOUNDS)).summarize(alpha=0.05)
```

```text
float sum: HWUSUAL Sum, Float Laplace, scale=14,405.18 95% accuracy=43,154.1
integer sum: HWUSUAL Sum, Integer Laplace, scale=14,400.00 95% accuracy=43,139.0
```

For integers, the scale is exactly 36 times the upper bound of 100 divided by the budget share; for floats it is slightly larger. Casting also changes error handling. Polars normally raises a data-dependent error when a value cannot be converted. OpenDP instead returns null, so every cast column is nullable and needs `.fill_null` before aggregation.

```python
import polars as pl
import opendp.prelude as dp

dp.enable_features("contrib")
context = dp.Context.compositor(
    data=pl.LazyFrame({"hours": ["40", "35", "n/a"] * 100}),
    privacy_unit=dp.unit_of(contributions=1),
    privacy_loss=dp.loss_of(epsilon=1.0),
    split_evenly_over=1,
    margins=[dp.polars.Margin(max_length=1000)],
)
hours = pl.col("hours").cast(int)  # OpenDP turns "n/a" into a null instead of raising an error
try:
    context.query().select(hours.clip(0, 80).sum().dp.noise()).summarize()
except dp.OpenDPException as err:
    print(err.message)  # input data (hours) might contain nulls. Preprocess your data with `.fill_null`.
query = context.query().select(hours.fill_null(0).clip(0, 80).sum().dp.noise())
print(query.summarize()["scale"].item())  # 80.0
```

Earlier chapters' `.dp.sum` handles imputation and clipping automatically. This snippet prepares the column manually and adds noise to a plain `.sum`, showing the checks OpenDP applies. A sum needs bounds from `.clip`; without them, one person's effect is unbounded. `.fill_null` and `.fill_nan` remove bounds, so fill first and clip afterward. The script applies this sequence to the survey in `.with_columns` and tests incorrect orders.

```{.python excerpt="prep-fill"}
def prepared_sum(context: dp.Context, steps: pl.Expr) -> pl.DataFrame:
    """A sum with noise on a column prepared in with_columns: OpenDP checks the steps left it non-null and bounded."""
    total: Any = pl.col("HWUSUAL").sum()
    return lazy_query(context).with_columns(steps).select(total.dp.noise()).summarize()
```

```text
prepared sum: HWUSUAL Sum, Integer Laplace, scale=14,400.00
cast without fill_null: input data (HWUSUAL) might contain nulls. Preprocess your data with `.fill_null`.
clip before fill_null: input domain must consist of bounded data. Either specify bounds in the input domain or use make_clamp.
fill_null before fill_nan: input data (HWUSUAL) might contain nans. Preprocess your data with `.fill_nan`.
float prepared sum: HWUSUAL Sum, Float Laplace, scale=14,405.18
```

You can fill from another column, such as actual hours for missing usual hours. Both columns must be row-wise so their rows align, and the fill column must contain no nulls.

```text
filled from a column: HWUSUAL Sum, Integer Laplace, scale=14,400.00
filled from a nullable column: fill expression must not be nullable
```

`.drop_nulls` and `.drop_nans` change column length by removing rows, so OpenDP forbids them in a row-wise context.

```text
drop_nulls in with_columns: drop_nulls is not allowed in a row-by-row context
```

Recoding uses `.replace` or `.replace_strict`. `.replace` swaps selected values without changing the type. `.replace_strict` maps every value and may change the type; unmapped values receive `default`. The documentation says OpenDP sets a missing `default` to null, but version 0.16.0 does not. With three toy codes and a mapping for two, `summarize` succeeds but `release` fails. Always supply `default` to `.replace_strict`.

```text
replace_strict without default: summarize scale=2.0; release: InvalidOperation(ErrString("incomplete mapping specified for `replace_strict`\n\nHint: Pass a `default` value to set unmapped values."))
replace_strict with default: released keys one other two
```

## Predicates and arithmetic {#sec-prep-operators}

OpenDP accepts Polars comparisons, conjunctions and arithmetic as row-wise expressions. Both operands of a binary operator must be row-wise, even in `.select` or `.agg`; otherwise a length mismatch could raise a data-dependent error. Tests such as `is_null` and `is_nan` are also row-wise and can define filters or grouping columns. The documentation counts respondents over retirement age by grouping on a comparison.

```{.python excerpt="prep-bool"}
def over_retirement_age(context: dp.Context) -> LazyFrameQuery:
    """How many responses come from people over 64: a boolean expression as the grouping column."""
    return (
        lazy_query(context)
        .group_by(pl.col("AGE") > RETIREMENT_AGE)
        .agg(dp.len(signed=True))
        .with_keys(pl.LazyFrame({"AGE": [True, False]}))  # a null age is in neither group and is dropped
    )
```

```text
over 64: len Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9
true over 64: true=1,190,845  false=2,620,832  null=255
```

Comparisons with nulls return null, so the 255 rows without an age form a third group omitted from the keys. One grouped query answers both sides of the comparison.

Version 0.16.0 does not support `is_in`, which tests membership in a list. Replace it with a disjunction of equality tests; the script uses this form to count employed respondents.

```text
is_in: IsIn { nulls_equal: false } is not currently supported
employed: len Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9
```

The other is the absolute value, in @sec-prep-refused.

## Strings and dates {#sec-prep-dates}

OpenDP supports string parsing into temporal types through `.str.strptime` and its variants `to_date`, `to_datetime` and `to_time`. The `format` argument is required because inferring a format from data could make it depend on one person's rows. OpenDP rejects nanosecond datetimes because the parser can panic on some inputs, producing a data-dependent error.

```text
to_date without format: format must be specified; otherwise Polars will attempt to infer the format from the data, resulting in an unstable transformation
nanosecond datetime: Nanoseconds are not currently supported due to potential panics when parsing inputs. Please open an issue on the OpenDP repository if you would find this functionality useful. Otherwise, consider parsing into micro- or millisecond datetimes instead.
```

The survey stores dates in `YEAR` and `QUARTER`, not a date column. The documentation combines them with `concat_str` before the context because OpenDP rejects concatenation inside a query.

```text
concat_str: Expr is not recognized at this time: ConcatHorizontal { delimiter: "-", ignore_nulls: false }. If you would like to see this supported, please file an issue.
```

Arithmetic is supported, so you can construct the date inside the query. Multiplying the year by 100 and adding three times the quarter gives an integer such as 200503. Parsing the string with `%Y%m` yields the first day of the quarter's final month. `.dt` expressions then extract date components.

```{.python excerpt="prep-date"}
def quarter_end() -> pl.Expr:
    """The last month of the quarter as a date, built from YEAR and QUARTER and parsed with a fixed format."""
    year_month = pl.col("YEAR") * 100 + pl.col("QUARTER") * 3  # 2005 and 1 become 200503
    return year_month.cast(str).str.to_date(format="%Y%m")


def responses_per_month(context: dp.Context) -> LazyFrameQuery:
    """How many responses carry each quarter-end month, read off the parsed date."""
    return (
        lazy_query(context)
        .with_columns(DATE=quarter_end())
        .group_by(pl.col("DATE").dt.month().alias("MONTH"))
        .agg(dp.len(signed=True))
        .with_keys(pl.LazyFrame({"MONTH": pl.Series(QUARTER_END_MONTHS, dtype=pl.Int8)}))
    )
```

```text
first dates: 2005 Q1 -> 2005-03-01  2005 Q2 -> 2005-06-01
per month: len Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9
true per month: 3=953,695  6=958,396  9=933,257  12=966,584
```

Each quarter ends in one of four public months, so the query uses those months as keys without a threshold. Their type is `Int8`, matching the output of `.dt.month`.

## A worked preparation: hours lost to absence {#sec-prep-worked}

The statistic is the mean difference between actual and usual hours for employed respondents. A negative mean means holidays, illness and short weeks outweigh overtime. The query performs each preparation step: it keeps `ILOSTAT=1` (working for pay or profit), casts both hour columns to integers for integer noise (@sec-prep-fill), fills missing usual hours with 0, then keeps rows with known usual hours, excluding 0 (variable hours) and 99 (not applicable).

Fill before filtering because a filter does not prove non-nullness to OpenDP (@sec-prep-contexts). The filtered column still has a nullable domain, and the next step uses it as a fill value, which must not be nullable. Missing actual hours are replaced by usual hours, making the difference zero. Finally, the query clips differences to the public range of ±40 hours, as required for a bounded sum. Using ±80 would double the noise.

```{.python excerpt="prep-worked"}
def prepare[Frame: (LazyFrameQuery, pl.LazyFrame)](frame: Frame) -> Frame:
    """The rows of people who work, with their usual hours known, and the hours they lost that week, clipped."""
    low, high = LOST_BOUNDS
    return (
        frame.filter(pl.col("ILOSTAT") == 1)  # working for pay or profit
        .with_columns(pl.col("HWUSUAL").cast(int).fill_null(0), pl.col("HWACTUAL").cast(int))
        .filter((pl.col("HWUSUAL") > 0) & (pl.col("HWUSUAL") < 99))  # 0 is "varies", 99 "not applicable"
        .with_columns(pl.col("HWACTUAL").fill_null(pl.col("HWUSUAL")))  # unknown actual hours: assume the usual
        .with_columns(LOST=(pl.col("HWACTUAL") - pl.col("HWUSUAL")).clip(low, high))
    )
```

The function accepts a query or a plain `LazyFrame`, so the release and validation use the same preparation. The query ends with `pl.col("LOST").dp.mean(LOST_BOUNDS)`; `summarize` reports its privacy cost.

```text
hours lost: LOST Sum, Integer Laplace, scale=11,520.00 95% accuracy=34,511.3
hours lost: LOST Length, Integer Laplace, scale=288.00 95% accuracy=863.3
hours lost, bounds (-80, 80): LOST Sum, Integer Laplace, scale=23,040.00 95% accuracy=69,022.2
```

The mean uses a noisy sum and count, which split the query's budget share. The sum's scale is 36 rows times the bound of 40, divided by one-eighth of the budget. With bounds of ±80, its scale doubles. Clipping trades bias against noise: narrower bounds reduce noise but can increase bias. [The full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derives the sum's sensitivity.

Because the survey is public, the script measures clipping bias. On private data, use a public or historical copy for this check.

```text
true prepared rows: 1,490,113  true sum of hours lost: -9,289,648  true mean: -6.2342
rows the clip changes: 42,830 (2.874%)  mean without the clip: -6.5319
mean error bar at 95%: 0.0268
```

Clipping changes fewer than 3% of rows but shifts the true mean from -6.53 to -6.23 hours, a bias of about one-third of an hour. This exceeds the noise. The script combines the sum and count accuracies into a mean error bar: the sum's bar plus the mean times the count's bar, divided by the count. The result is 0.027 hours. At this sample size, clipping bias, not noise, limits accuracy.

```text
released (one run):
  bins: [-inf, 0)=-129  [0, 20)=89,491  [20, 40)=907,895  [40, 60)=406,594  [60, 80)=78,436  [80, 98)=13,452  [98, inf)=2,294,708
  over 64: true=1,190,894  false=2,620,771
  per month: 3=953,684  6=958,313  9=933,077  12=966,807
  hours lost: mean -6.2147  (true -6.2342, within 0.0268 with probability 0.95)
budget spent: 4 of 4 queries
```

Every count falls within its error bar of about 432. The empty bin is negative, which signed counts allow. The mean falls within its bar relative to the clipped truth. These four releases use all four budget shares. [Assessing utility and choosing the noise](07-utility-and-noise.md) develops the error-bar analysis further.

## What a refused expression looks like {#sec-prep-refused}

An unsupported expression raises an `OpenDPException` with a variant and message when you build the query, before OpenDP reads any data. Absolute difference in hours is one example.

```text
abs: Expr is not recognized at this time: [(col("HWACTUAL")) - (col("HWUSUAL"))].abs(). If you would like to see this supported, please file an issue.
```

The messages fall into two categories. "Not recognized at this time" means the expression has no implementation, as with `concat_str` and `abs`. Other messages identify a rule, such as a required format or non-null fill, and give the in-query fix. Rewrite unsupported expressions with supported operators when possible, as with `is_in`. Otherwise, apply the expression before the context only if it is a fixed row-wise function. Disclose this step with the release because OpenDP did not check it.

## Summary

| Call or concept | What it does | Rule of thumb |
| --- | --- | --- |
| Preparation inside the query | OpenDP parses every step and accounts for it | Prepare before the context only with fixed row-by-row steps and public facts, and say so |
| `.with_columns`, `.filter`, `.group_by` | Row-by-row contexts: each person's rows stay their own | A new column has no margins; a filter drops key and length invariants and proves nothing about what remains |
| `.select`, `.agg` | May change the number and order of rows | Binary operators and fills inside them still need row-by-row inputs |
| `.cut(breaks)` | Bins a number into categories with a data-independent encoding | Pass the labels as keys to avoid a threshold |
| Domain descriptors | Nullable, may be NaN, bounds | Every column starts nullable; floats need `.fill_null` and `.fill_nan`; `.clip` sets bounds |
| `.cast` | A failed cast returns null instead of raising | Follow it with `.fill_null`; cast to integers for a cheaper sum |
| `.fill_null`, `.fill_nan` | Impute; a column fill must be row-by-row and non-nullable | Fill first, then clip, because filling drops the bounds |
| `pl.Categorical` vs `pl.Enum` | Categories from the data vs fixed in the schema | Group by enums; cast categoricals to string |
| `.replace_strict(mapping)` | Recodes every value | Pass a `default`: in 0.16.0 a missing one fails at release |
| Comparisons, `&`, `|`, arithmetic | Row-by-row predicates and new columns | `is_in` and `abs` are not supported in 0.16.0 |
| `.str.to_date(format=...)` | Parses a string into a date | The format is mandatory; nanosecond datetimes are refused |
| `.dt.month()` and the other components | Reads a part of a date or time | Keys for a component must match its type (`Int8` for a month) |
| `OpenDPException` | Raised when the plan is parsed, before any data are read | "Not recognized" is unimplemented; the other messages state a rule and its fix |

: The contexts and expressions of this chapter, what each does, and the rule of thumb that goes with it. {#tbl-prep-summary}

## References

- OpenDP documentation: "Preparing Microdata" in the tabular data tutorial; "Data Types" and the "Expressions" pages "Boolean", "Columns", "Manipulation", "Operators", "String" and "Temporal" in the Polars user guide.
- Nissim, Raskhodnikova and Smith, "Smooth Sensitivity and Sampling in Private Data Analysis" (2007): the sample and aggregate framework behind the stable `group_by/agg`.
- Gaboardi, Hay and Vadhan, "A Programming Framework for OpenDP" (2020): the stable transformations that each expression is parsed into.
- The full edition's chapter on [sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derives how the bounds of a clipped column fix the noise of its sum and mean.
