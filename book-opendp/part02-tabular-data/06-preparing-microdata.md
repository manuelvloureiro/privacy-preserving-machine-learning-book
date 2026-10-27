# Preparing microdata

Survey data rarely arrive in the form that a statistic needs. Hours of work come as floats, with 99 for "not applicable" and a null for a missing answer. The date of a response is in two integer columns. So before you release anything, you cast, fill, clip, filter and recode. This chapter is about where that work happens. OpenDP reads the whole lazy query before it runs, preparation included, and accounts for every step. So the preparation belongs inside the query, where the guarantee covers it.

The chapter follows the documentation's tutorial on preparing microdata, its page on data types and its pages on expressions. You see which Polars contexts and expressions OpenDP allows, what it changes about them, and how it tracks nulls, undefined floats and bounds. The chapter ends with a worked preparation of the Labour Force Survey (LFS), for the mean difference between actual and usual hours of work.

The script [opendp_preparing_microdata.py](../../code/part02-tabular-data/opendp_preparing_microdata.py) runs every step on OpenDP 0.16.0 with Polars 1.36.1. The noise scales, error bars, thresholds, error messages and true values that it prints are the same on every run. The released counts and mean are not, because OpenDP's noise cannot be seeded, so this page shows them in one block marked as one run.

```bash
uv run python code/part02-tabular-data/opendp_preparing_microdata.py
```

## Prepare inside the query {#sec-prep-inside}

A context on a Polars LazyFrame reads only the frame's schema, which the library treats as public, and touches the data only at a release. OpenDP parses each query, a lazy plan, into a chain of transformations before anything runs, as [Transformations](../part03-framework/09-transformations.md) explains. For each step, the library finds how much one person's rows can change the output. So a filter or a fill in the middle of the plan is part of the accounting.

Preparation before the context is different, because the library then sees only the prepared frame, and its guarantee is about that frame. A fixed function of each row, such as a renamed column or a date parsed with a known format, leaves every person with the same rows. Then the guarantee still holds, but the claim is yours, because the library did not check it. A step that reads the data is worse. An imputation by the column's own mean, a bound at its maximum or a filter at a quantile carries information about everyone into the release, outside the accounting. Before the context, use only public facts, such as the schema, the code book and the published bound on the sample's size.

The context has the privacy unit of [the data and the context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context), up to 36 rows per person, and that chapter's bound on the rows. The budget is approximate differential privacy, with a privacy budget $\varepsilon$ of 1 and a slack $\delta$ of $10^{-7}$, split evenly over four queries.

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

The schema line is all that the context knows about the data at this point, and it assumes that any column can hold nulls.

## Columns, filters and groups {#sec-prep-contexts}

Polars applies an expression in one of five contexts, and OpenDP puts them in two groups. Expressions in `.select` and `.agg` can change the number and the order of rows, because an aggregation does exactly that. Expressions in `.with_columns`, `.filter` and `.group_by` must be **row-by-row**: a function applied to each row on its own. The reason is the privacy unit. A row-by-row expression keeps each person's rows their own, so 36 rows in gives 36 rows out. An expression that sorted or shifted the column would mix people's rows, and the library could not say what one person changes.

The documentation's first example adds a column with `.with_columns` and groups on it. The row-by-row expression `cut` puts the usual hours into bands at fixed breaks, and replaces `HWUSUAL` with its bin. The labels of the bins depend only on the breaks, so they are public and can be the keys.

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

The scale is the 36 rows that one person can add to a bin, divided by the query's share of the budget. With the keys, every bin is released, including the empty one below zero, whose signed count can be negative. Without the keys, the library must learn from the data which bins exist, so the query has a threshold, as in [Grouping](04-grouping.md). The rows with a null `HWUSUAL` fall in a null bin that is not among the keys, so no release has them. Also, a margin that you declared about `HWUSUAL` applies to the original column, and the new column has none.

A `.filter` removes rows with a boolean row-by-row expression, such as the documentation's filter on the code 99 before a sum. A filter keeps most of what the margins say. But it discards any invariant about the group keys and sizes, which were facts about the data before the filter. A filter also tells the library nothing about what remains: a column filtered to non-null values is still nullable in its domain.

```text
filtered sum: HWUSUAL Sum, Integer Laplace, scale=11,520.00 95% accuracy=34,511.3
```

The last context of the tutorial is a `.group_by` and `.agg` *before* the private aggregation. In the documentation's version of sample and aggregate, the rows go into 1,000 groups by `PIDENT`, and the query releases the mean of each group's minimum usual hours. Any expression can go in that inner `.agg`, but the price is noise. The script prints the scales of the mean on the context above, and with a `Bound` from [Identifier truncation and bounds](05-identifier-truncation.md) that puts each person's rows in one group.

```text
sample and aggregate: HWUSUAL Sum, Integer Laplace, scale=17,280.00
sample and aggregate: HWUSUAL Length, Integer Laplace, scale=576.00
one group per person: HWUSUAL Sum, Integer Laplace, scale=480.00
one group per person: HWUSUAL Length, Integer Laplace, scale=16.00
```

The first context does not know that a person's rows share a `PIDENT`, so it allows for 36 changed groups. Even with the bound, the inner aggregation doubles the noise, because a changed group counts as one value removed and another added.

## The types OpenDP tracks {#sec-prep-types}

OpenDP assumes that every column can hold nulls until an expression proves otherwise. It also keeps **domain descriptors** for each column: facts about its values that later aggregations need. For a number it tracks lower and upper bounds, which a `.clip` sets and some later steps drop. The supported integers are the signed ones and the unsigned ones of 32 and 64 bits, so a `UInt8` column is refused. A float column also records whether its values can be NaN, the float for an undefined number, so a float sum needs both `.fill_null` and `.fill_nan`. Booleans and strings have no descriptors.

A categorical column stores an integer code per value, with a table of labels, and two things about that table depend on the data. An expression that could add or remove a category, such as `.fill_null`, makes Polars rewrite the codes and warn, a side effect that depends on the data. And the codes follow the order in which the values first appear, so a group-by on them would show the order of the rows. So the library refuses both. An `Enum` column fixes its categories in the schema, so neither problem occurs, and neither does it with `cut`, whose encoding depends only on the breaks.

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

The documentation's reason to cast is the float sum. OpenDP accounts for the rounding error of float addition, and on many rows with wide bounds that term shows in the noise scale. The script sums the usual hours both ways.

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

On integers the scale is exactly 36 times the upper bound of 100 over the share of the budget, and on floats it is a little more. A cast has one more effect. In Polars, a value that cannot be converted raises an error, which depends on the data, so OpenDP makes a failed cast return a null. So every cast column is nullable, and a statistic on it needs a `.fill_null` first.

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

The `.dp.sum` of the earlier chapters adds the imputation and the clip for you. The snippet instead prepares the column by hand and adds noise to a plain `.sum`, to show what the library checks. The sum needs the bounds that a `.clip` sets, because without bounds one person could move the sum by any amount. But `.fill_null` and `.fill_nan` drop the bounds, so fill first, then clip. The script runs the same preparation on the survey in `.with_columns`, and tries the wrong orders.

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

The fill value can also come from another column, such as the actual hours for missing usual hours. Then the column and the fill must both be row-by-row, so that their rows line up, and the fill must have no nulls.

```text
filled from a column: HWUSUAL Sum, Integer Laplace, scale=14,400.00
filled from a nullable column: fill expression must not be nullable
```

The expressions `.drop_nulls` and `.drop_nans` remove rows and change the column's length, so the library forbids them in a row-by-row context.

```text
drop_nulls in with_columns: drop_nulls is not allowed in a row-by-row context
```

Recoding has two forms. The expression `.replace` swaps listed values and keeps the type. The expression `.replace_strict` maps every value and can change the type, so an unmapped value takes the `default`. The documentation says that OpenDP sets a missing `default` to null, but version 0.16.0 does not. On a toy column of three codes with a mapping for two, `summarize` accepts the query, but `release` fails. So give `replace_strict` a `default` yourself.

```text
replace_strict without default: summarize scale=2.0; release: InvalidOperation(ErrString("incomplete mapping specified for `replace_strict`\n\nHint: Pass a `default` value to set unmapped values."))
replace_strict with default: released keys one other two
```

## Predicates and arithmetic {#sec-prep-operators}

OpenDP accepts the comparison, conjunction and arithmetic operators of Polars, as row-by-row expressions. Both sides of a binary operator must be row-by-row even inside `.select` or `.agg`, because a length mismatch would raise an error that depends on the data. Tests such as `is_null` and `is_nan` are row-by-row too, and they build filters and grouping columns. The documentation counts the respondents over retirement age with a group-by on a comparison.

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

A comparison with a null is null, so the 255 rows without an age form a third group that the keys leave out. The grouped form answers both halves of the question for the price of one.

Version 0.16.0 does not support two operators that you might want. It refuses `is_in`, membership in a list, so write the test as a disjunction of equalities. The script counts the employed in this way.

```text
is_in: IsIn { nulls_equal: false } is not currently supported
employed: len Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9
```

The other is the absolute value, in @sec-prep-refused.

## Strings and dates {#sec-prep-dates}

Of the string expressions, OpenDP supports only parsing into a temporal type, with `.str.strptime` and its variants `to_date`, `to_datetime` and `to_time`. The `format` argument is mandatory, because a format inferred from the data could change with one person's rows. Nanosecond datetimes are refused, because the parser can panic on some inputs, an error that depends on the data.

```text
to_date without format: format must be specified; otherwise Polars will attempt to infer the format from the data, resulting in an unstable transformation
nanosecond datetime: Nanoseconds are not currently supported due to potential panics when parsing inputs. Please open an issue on the OpenDP repository if you would find this functionality useful. Otherwise, consider parsing into micro- or millisecond datetimes instead.
```

The survey has no date column, only `YEAR` and `QUARTER`. The documentation builds one with `concat_str` before the context, because the library refuses concatenation inside a query.

```text
concat_str: Expr is not recognized at this time: ConcatHorizontal { delimiter: "-", ignore_nulls: false }. If you would like to see this supported, please file an issue.
```

But arithmetic is supported, so you can build the date inside the query. The year times 100 plus three times the quarter gives an integer such as 200503. As a string, it parses with the format `%Y%m` into the first day of the quarter's last month. The `.dt` expressions then read the components of the date, from the year to the day.

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

Every quarter ends in one of four public months, so they are the keys, with no threshold. The keys have the type `Int8`, because they must match the type that `.dt.month` returns.

## A worked preparation: hours lost to absence {#sec-prep-worked}

The statistic is the mean difference between the hours that a person with a job actually worked in the survey week and the hours that they usually work. A negative mean means that holidays, illness and short weeks outweigh overtime. Each step of the preparation is a decision about the data, inside the query. First the query keeps `ILOSTAT` equal to 1, working for pay or profit. It casts both hour columns to integers, for the integer noise of @sec-prep-fill. It fills the usual hours with 0. Then it keeps the rows whose usual hours are known, without 0 for hours that vary and 99 for not applicable.

The fill comes before the filter because a filter proves nothing to the library (@sec-prep-contexts). After the filter, no null usual hours remain, but the column is still nullable in its domain. And the next step uses it as a fill value, which must not be nullable. That step gives a row without actual hours its usual hours, so its difference is zero, the least informative choice. Last, the query clips the difference to public bounds of plus or minus 40 hours, because a sum needs bounds. The full range of plus or minus 80 would double the noise.

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

The function takes a query or a plain LazyFrame, so the release and the check use one preparation. The query ends in `pl.col("LOST").dp.mean(LOST_BOUNDS)`, and `summarize` shows what the mean costs.

```text
hours lost: LOST Sum, Integer Laplace, scale=11,520.00 95% accuracy=34,511.3
hours lost: LOST Length, Integer Laplace, scale=288.00 95% accuracy=863.3
hours lost, bounds (-80, 80): LOST Sum, Integer Laplace, scale=23,040.00 95% accuracy=69,022.2
```

A mean is a noisy sum over a noisy count, which split the query's share of the budget. So the sum's scale is 36 rows times the bound of 40, over an eighth of the budget. With bounds of plus or minus 80, the sum's scale doubles, so the clip is a trade: a narrower bound gives less noise and more bias. [The full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derives how the bounds of a sum set its sensitivity.

The survey is public, so the script measures the bias, which on real data you can do only with a public or historical copy.

```text
true prepared rows: 1,490,113  true sum of hours lost: -9,289,648  true mean: -6.2342
rows the clip changes: 42,830 (2.874%)  mean without the clip: -6.5319
mean error bar at 95%: 0.0268
```

The clip changes under three percent of the rows. But these rows move the true mean from -6.53 hours to -6.23, a bias of about a third of an hour. The noise is far smaller. The script turns the two accuracies into an error bar for the mean: the sum's bar plus the mean times the count's bar, all over the count. It comes to 0.027 hours. So on this many rows, the clip limits the result, not the noise.

```text
released (one run):
  bins: [-inf, 0)=-129  [0, 20)=89,491  [20, 40)=907,895  [40, 60)=406,594  [60, 80)=78,436  [80, 98)=13,452  [98, inf)=2,294,708
  over 64: true=1,190,894  false=2,620,771
  per month: 3=953,684  6=958,313  9=933,077  12=966,807
  hours lost: mean -6.2147  (true -6.2342, within 0.0268 with probability 0.95)
budget spent: 4 of 4 queries
```

Every count is within its bar of about 432, and the empty bin is negative, as a signed count can be. The mean is within its bar of the clipped truth. The four releases used the four shares. [Assessing utility and choosing the noise](07-utility-and-noise.md) takes the error bars further.

## What a refused expression looks like {#sec-prep-refused}

An expression that OpenDP cannot parse raises an `OpenDPException`, with a variant and a message, when you build the query and before it reads any data. The absolute value of the difference in hours is an example.

```text
abs: Expr is not recognized at this time: [(col("HWACTUAL")) - (col("HWUSUAL"))].abs(). If you would like to see this supported, please file an issue.
```

The messages on this page are of two kinds. "Not recognized at this time" means that nobody has implemented the expression yet, as with `concat_str` and `abs`. The other messages state a rule, such as a mandatory format or a fill that must not be nullable, and each names its fix inside the query. For an expression without an implementation, rewrite it with supported operators if you can, as `is_in` became a disjunction. If not, apply it before the context, but only if it is a fixed function of each row. Then state that step with the release, because the library did not check it.

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
