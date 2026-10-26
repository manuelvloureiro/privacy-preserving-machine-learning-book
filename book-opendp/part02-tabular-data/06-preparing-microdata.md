# Preparing microdata

Survey data rarely arrive in the form a statistic needs. Hours of work come as floats, with a 99 for "not applicable" and a null where the answer is missing. The date of a response sits in two integer columns, the year and the quarter. Before you release anything you cast, fill, clip, filter and recode. This chapter is about where that work happens. The one idea is that OpenDP reads the whole lazy query before it runs, preparation included, and accounts for every step. So preparation belongs inside the query, where the guarantee covers it, and not in a script that runs before the context exists.

The chapter follows the documentation's tutorial on preparing microdata, its page on data types and its pages on the expressions it accepts. You will see which Polars contexts and expressions OpenDP allows, and what it changes about them and why. You will also see how it tracks nulls, undefined floats and bounds in the domain of a column. The chapter ends with a worked preparation of the Labour Force Survey (LFS) that releases one statistic. It is the mean difference between the hours people actually worked and the hours they usually work. The last section shows what an expression OpenDP refuses looks like.

The script [opendp_preparing_microdata.py](../../code/part02-tabular-data/opendp_preparing_microdata.py) runs every step on OpenDP 0.16.0 with Polars 1.36.1. The noise scales, the error bars, the thresholds, the error messages and the true values it prints are the same on every run. The released counts and the released mean are not, because OpenDP draws its noise from a generator that cannot be seeded. This page therefore shows them in one block marked as one run.

```bash
uv run python code/part02-tabular-data/opendp_preparing_microdata.py
```

## Prepare inside the query {#sec-prep-inside}

A context built around a Polars LazyFrame reads only the frame's schema, which the library treats as public. The data are not touched until a query is released. Each query you submit is a lazy plan, and OpenDP parses it into a chain of transformations before anything runs, the chain that [Transformations](../part03-framework/09-transformations.md) opens. For each step the library works out how much one person's rows can change the output. So a filter or a fill in the middle of the plan is part of the accounting.

Preparation before the context is different. What the library sees is the prepared frame, and its guarantee is a guarantee about that frame. Suppose the step was a fixed function of each row, such as renaming a column or parsing a date with a known format. Then every person still owns the same rows afterwards, and the guarantee means what it says. But the library did not check it, so the claim is yours. Suppose instead that the step read the data. An imputation by the column's own mean, a bound set at its maximum, or a filter at a quantile each carries information about everyone into the release. The accounting never saw that path, so the stated guarantee no longer covers the original data. The facts you may use freely are the public ones, such as the schema, the code book and the published bound on the sample's size.

The context is the one the documentation uses on these pages. The privacy unit is 36 rows, because a person answers at most once a quarter over nine years. So two neighbouring datasets differ by one person's rows, at most 36 of them. The budget is approximate differential privacy with the privacy budget $\varepsilon$ at 1 and the slack $\delta$ at $10^{-7}$, as in [Essential statistics](03-essential-statistics.md), split evenly over four queries. The margin is the public bound on the number of rows from [Identifier truncation and bounds](05-identifier-truncation.md).

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

Each query gets a quarter of the budget. The schema line is all the context knows about the data at this point, and it assumes that any column may hold nulls.

## Columns, filters and groups {#sec-prep-contexts}

Polars applies an expression in one of five contexts, and OpenDP treats them in two groups. Expressions in `.select` and `.agg` may change the number and the order of rows, because an aggregation does exactly that. Expressions in `.with_columns`, `.filter` and `.group_by` must be **row-by-row**, which means they could be written as a function applied to each row on its own. The reason is the privacy unit. A row-by-row expression keeps each person's rows aligned with their contributions, so 36 rows in means 36 rows out. An expression that sorted or shifted the column would mix people's rows, and the library could no longer say what one person changes.

The documentation's first example adds a column with `.with_columns` and groups on it. The expression `cut` bins the usual hours into bands at fixed breaks, which is row-by-row, and it shadows `HWUSUAL` with its bin. The labels of the bins depend only on the breaks, not on the data, so they are public and can be passed as keys.

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

The scale is 144, which is the 36 rows one person can add to a bin, divided by the query's share of the budget. The error bar on each count is about 432. With the keys supplied, every bin is released, including the empty one below zero, whose noisy count can be negative because the count is signed. Without the keys the library would have to learn from the data which bins exist. The same query then carries a threshold of 2,973 below which a bin is dropped, as [Grouping](04-grouping.md) explained. The rows with a null `HWUSUAL` fall in a null bin that is not among the keys, so they are released nowhere. One more thing changes when a column is shadowed: any margin you declared about `HWUSUAL` applied to the original column, and the new column has none.

A `.filter` masks rows with a boolean row-by-row expression. The documentation's example casts the hours to integers, drops the rows coded 99 and sums the rest. Filtering only removes rows, so it keeps most of what the margins say. But it discards any invariant about the group keys and the group sizes, because those were facts about the dataset before the filter. Filtering also tells the library nothing about what remains. A column filtered to non-null values is still nullable in its domain, which the worked example will run into.

```text
filtered sum: HWUSUAL Sum, Integer Laplace, scale=11,520.00 95% accuracy=34,511.3
```

The last context of the tutorial is a `.group_by` followed by `.agg` *before* the private aggregation. The documentation groups the rows a thousand ways by `PIDENT` modulo 1,000 and takes the minimum usual hours in each group. It then releases the mean of those thousand minima, which is its version of sample and aggregate. Any expression may go in that inner `.agg`, which is its appeal, but the price is noise. The script prints the scales of the mean with the context above, and then with a privacy unit that says each person's rows all land in one group. That unit is the `Bound` of [Identifier truncation and bounds](05-identifier-truncation.md).

```text
sample and aggregate: HWUSUAL Sum, Integer Laplace, scale=17,280.00
sample and aggregate: HWUSUAL Length, Integer Laplace, scale=576.00
one group per person: HWUSUAL Sum, Integer Laplace, scale=480.00
one group per person: HWUSUAL Length, Integer Laplace, scale=16.00
```

The first context does not know that a person's rows share a `PIDENT`, so it allows for 36 changed groups, and the bound brings the scales down by that factor. Even then the inner aggregation doubles the noise, because changing one row changes one aggregated row, which counts as removing a value and adding another.

## The types OpenDP tracks {#sec-prep-types}

Every Polars column can hold nulls, and OpenDP assumes that every column does until an expression proves otherwise. Beyond that, the library keeps a few **domain descriptors** for each column, which are facts about its values that later aggregations need. For an integer column it tracks lower and upper bounds, which a `.clip` establishes and which some later steps drop. The supported integer types are the signed ones and the unsigned ones of 32 and 64 bits, so a `UInt8` column is refused. A float column has one more descriptor, whether the values may be NaN, the float that stands for an undefined number. A float sum therefore needs both `.fill_null` and `.fill_nan` before it. Booleans and strings carry no descriptors, and strings are the slowest type to work with.

Categorical columns look like strings but store an integer code per value, with a table of labels. Two things about that table depend on the data, and the library refuses both. First, an expression that could add or remove a category, such as `.fill_null` or a binary operator, makes Polars rewrite the codes and emit a warning. That is a side effect that depends on the data. Second, the codes are assigned in the order the values first appear, so grouping by them would reveal the order of the rows. An `Enum` column fixes its categories in the schema, so neither problem arises. The `cut` expression is fine too, because its encoding depends only on the breaks.

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

The error names the fix, which is to cast to string before grouping, and the enum query goes through. A released enum column comes back typed categorical, because the frame crosses from Rust to Python through Arrow, which does not tell the two apart. Dates, datetimes and times are supported too. A datetime's time zone is part of the domain, so a column must have one zone throughout. The components of a date are read with the `.dt` expressions of @sec-prep-dates.

## Casting, filling and clipping {#sec-prep-fill}

The schema is read off the frame, so in ordinary use the columns are already typed and a cast is a choice, not a repair. The documentation's reason to cast is the float sum. OpenDP accounts for the rounding error of floating-point addition, and on hundreds of thousands of rows with wide bounds that term shows in the noise scale. The script sums the usual hours both ways.

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

On integers the scale is exactly 36 times the upper bound of 100 over the share of the budget, and on floats it is a little more. The difference is small here and grows with the bounds and the number of rows. A cast has one more effect. In Polars a value that cannot be converted raises an error, and whether it raises depends on the data. So OpenDP makes a failed cast return a null instead. The output domain of every cast is therefore nullable, and a statistic on a cast column needs a `.fill_null` first.

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

The `.dp.sum` of the earlier chapters inserts the imputation and the clip for you. The snippet instead prepares the column by hand and adds noise to a plain `.sum`, which shows what the library checks. A `.clip` establishes the bounds descriptor, and the sum looks for it, since without bounds one person could move the sum by any amount. The order of the steps matters, because `.fill_null` and `.fill_nan` drop the bounds. Fill first, then clip. The script runs the same preparation on the survey in `.with_columns` and tries the wrong orders.

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

The fill value need not be a constant. The documentation fills a person's missing usual hours with their actual hours, which is a fill from another column. Then both the column and the fill must be row-by-row, so that they have the same length and line up. The fill must also be free of nulls itself, or the output could still hold one. The script fills `HWACTUAL` with 0 first and then uses it as the fill, which is accepted, and tries the unfilled column, which is not.

```text
filled from a column: HWUSUAL Sum, Integer Laplace, scale=14,400.00
filled from a nullable column: fill expression must not be nullable
```

Two expressions remove rows rather than fill them. `.drop_nulls` and `.drop_nans` guarantee a clean column, but they change its length. So the library forbids them in a row-by-row context, and they cost the invariants on the dataset's size.

```text
drop_nulls in with_columns: drop_nulls is not allowed in a row-by-row context
```

Recoding has two forms. The expression `.replace` swaps listed old values for new ones and keeps the type, while `.replace_strict` maps every value and may change the type. So a value that the second mapping leaves out needs somewhere to go, and it takes the `default`. The documentation says that OpenDP sets a missing `default` to null, so that the query cannot fail on an unexpected value. Version 0.16.0 does not do that. The script maps a toy column of three codes with a mapping for two of them. Then `summarize` accepts the query, but `release` raises the Polars error about an incomplete mapping. Give `replace_strict` a `default` yourself.

```text
replace_strict without default: summarize scale=2.0; release: InvalidOperation(ErrString("incomplete mapping specified for `replace_strict`\n\nHint: Pass a `default` value to set unmapped values."))
replace_strict with default: released keys one other two
```

The last expression of the page is `.to_physical`, which returns the integer codes behind a categorical or temporal column, under the same condition on the encoding.

## Predicates and arithmetic {#sec-prep-operators}

OpenDP accepts all of Polars's comparison, conjunction and arithmetic operators, and treats each as row-by-row. Both sides of a binary operator must be row-by-row even inside `.select` or `.agg`, so that they have the same length and the rows line up. A mismatch would raise an error that depends on the data. The boolean tests are row-by-row too. They are `is_null` and `is_not_null` for any type and `is_nan`, `is_not_nan`, `is_finite` and `is_not_finite` for floats, and `.not_` negates a predicate. These expressions build filters and grouping columns. The documentation counts the respondents over retirement age by grouping on a comparison.

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

A comparison with a null is null, so the 255 rows without an age form a third group that the keys leave out. The same question can be asked as a filter inside the aggregation, `pl.col("HWUSUAL").filter(pl.col("AGE") > 64)`, which is not row-by-row but requires row-by-row inputs. The grouped form answers both halves of the question for the same price.

Two operators you might reach for are not available in 0.16.0. Membership in a list, `is_in`, is refused with a message that says it is not yet supported, so write the test as a disjunction of equalities. The script counts the employed, who are working or absent from work, that way.

```text
is_in: IsIn { nulls_equal: false } is not currently supported
employed: len Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9
```

The absolute value is the other, and @sec-prep-refused shows its message. One more restriction comes from the columns page: an expression selects one column at a time, so a query over several columns lists one expression per column.

## Strings and dates {#sec-prep-dates}

Of the string expressions, OpenDP supports only parsing into a temporal type, with `.str.strptime` and its variants `to_date`, `to_datetime` and `to_time`. The `format` argument is mandatory. Polars can infer a format from the data, but the inferred format could change when one person's rows are added or removed, which makes the step unstable. Parsing into nanosecond datetimes is refused as well, because the parser can panic on some inputs, and a panic is an error that depends on the data.

```text
to_date without format: format must be specified; otherwise Polars will attempt to infer the format from the data, resulting in an unstable transformation
nanosecond datetime: Nanoseconds are not currently supported due to potential panics when parsing inputs. Please open an issue on the OpenDP repository if you would find this functionality useful. Otherwise, consider parsing into micro- or millisecond datetimes instead.
```

The survey has no date column, only `YEAR` and `QUARTER`. The documentation builds one with `concat_str` before the context, because concatenation is not a supported expression. The script confirms that it is refused inside a query.

```text
concat_str: Expr is not recognized at this time: ConcatHorizontal { delimiter: "-", ignore_nulls: false }. If you would like to see this supported, please file an issue.
```

Arithmetic is supported, so the same date can be built inside the query. Multiplying the year by 100 and adding three times the quarter gives an integer such as 200503. Cast to a string, it parses with the format `%Y%m` into the first day of the quarter's last month. The components of the date are then read with the `.dt` expressions. For a date they cover the year, the quarter, the month, the week, the weekday and the day. For a time they cover the hour down to the nanosecond.

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

The four months are public, since every quarter ends in one of them, so the keys are given and the counts cost no threshold. The key series is typed `Int8` because that is what `.dt.month` returns, and keys must match the grouping column's type exactly.

## A worked preparation: hours lost to absence {#sec-prep-worked}

The statistic is the mean difference between the hours a person actually worked in the survey week and the hours they usually work, among people with a job. A negative mean means that holidays, illness and short weeks outweigh overtime. Each step of the preparation is a decision about the data, and each is inside the query. The rows are first restricted to `ILOSTAT` equal to 1, which the code book defines as working for pay or profit. Both hour columns are cast to integers, so that the sum later uses the integer noise of @sec-prep-fill. The usual hours are filled with 0 before the next filter rather than after it. The filter then keeps the rows whose usual hours are known. That excludes 0, the code for hours that vary, and 99, the code for not applicable.

The reason for filling before filtering is the rule from @sec-prep-contexts that a filter proves nothing to the library. After the filter no null usual hours remain, but the column is still nullable in its domain. The next step wants to use it as a fill value, which is refused unless it is non-nullable. A row whose actual hours are missing gets the usual hours instead, which makes its difference zero, the least informative choice. Finally the difference is clipped to the public bounds of plus or minus 40 hours, because a sum needs bounds. The range of possible differences, plus or minus 80, would double the noise.

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

The function takes either a query or a plain LazyFrame. So the script runs one preparation through the context for the release, and on the frame for the dev-time check. The query ends in `pl.col("LOST").dp.mean(LOST_BOUNDS)`, and `summarize` shows what the mean costs.

```text
hours lost: LOST Sum, Integer Laplace, scale=11,520.00 95% accuracy=34,511.3
hours lost: LOST Length, Integer Laplace, scale=288.00 95% accuracy=863.3
hours lost, bounds (-80, 80): LOST Sum, Integer Laplace, scale=23,040.00 95% accuracy=69,022.2
```

A mean is released as a noisy sum over a noisy count, and the query's share of the budget is split between the two. So the sum's scale is the 36 rows times the bound of 40, over an eighth of the budget. The count's scale divides the same 36 rows by the same eighth. With bounds of plus or minus 80 the sum's scale doubles, which is the third line. The clip is the trade of this chapter, since a narrower bound means less noise and more bias. How the sensitivity of a sum follows from its bounds is derived in [the full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md).

Because the survey is public, the script measures the bias at development time, which on real data you can do only with a public or historical copy.

```text
true prepared rows: 1,490,113  true sum of hours lost: -9,289,648  true mean: -6.2342
rows the clip changes: 42,830 (2.874%)  mean without the clip: -6.5319
mean error bar at 95%: 0.0268
```

The preparation keeps 1,490,113 rows and the clip changes 42,830 of them, under three percent. Those rows move the true mean from -6.53 hours to -6.23. So the clip costs a bias of about a third of an hour. The noise is far smaller. The script turns the two accuracies into an error bar for the mean, which is the sum's bar plus the mean times the count's bar, all over the count. It comes to 0.027 hours. On this many rows the clip, not the noise, is what limits the result, and a wider bound would trade some of that bias for noise you can read off the table.

```text
released (one run):
  bins: [-inf, 0)=-129  [0, 20)=89,491  [20, 40)=907,895  [40, 60)=406,594  [60, 80)=78,436  [80, 98)=13,452  [98, inf)=2,294,708
  over 64: true=1,190,894  false=2,620,771
  per month: 3=953,684  6=958,313  9=933,077  12=966,807
  hours lost: mean -6.2147  (true -6.2342, within 0.0268 with probability 0.95)
budget spent: 4 of 4 queries
```

Every count landed within its bar of about 432, and the empty bin came out negative, as a signed count may. The mean landed within its bar of the clipped truth. The four releases used the four shares, and a fifth query would be refused. [Assessing utility and choosing the noise](07-utility-and-noise.md) takes the error bars further, and shows what they leave out.

## What a refused expression looks like {#sec-prep-refused}

An expression that OpenDP cannot parse raises an `OpenDPException` while the query is built, before any data are read. The exception carries a variant and a message. The absolute value of the difference in hours is a case in point.

```text
abs: Expr is not recognized at this time: [(col("HWACTUAL")) - (col("HWUSUAL"))].abs(). If you would like to see this supported, please file an issue.
```

The messages on this page come in two kinds. "Not recognized at this time" means the expression has not been implemented, and the message asks for an issue, which is how `concat_str` and `abs` fail. The others state a rule, such as that a format must be specified or that a fill must not be nullable. Each of those names its fix, which is to prepare differently inside the query. For an unimplemented expression you have two choices. Rewrite it with supported operators where you can, as `is_in` became a disjunction above. Otherwise apply it before the context, as the documentation does with `concat_str`, but only if it is a fixed function of each row. Then state that step alongside the release, because the library did not check it.

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
