# Tabular data

This part does the everyday work of a private analysis, which is statistics on a table. OpenDP does it through Polars, a dataframe library whose lazy queries OpenDP can read before they run. So every filter, cast and aggregation is accounted for in the privacy guarantee. The thread of the part is that every statistic needs two things from you. One is a bound on what one person can contribute, and the other is a share of the budget. The library turns those into a noise scale that you can read before you release anything.

The chapters follow the documentation's tutorial on tabular data. [Essential statistics](03-essential-statistics.md) releases counts, sums, means, medians and quantiles. It also shows how `summarize` reports the noise scale and the error bar of a query. [Grouping](04-grouping.md) releases a table with one row per group, either with a public list of keys or with keys the library selects privately. [Identifier truncation and bounds](05-identifier-truncation.md) handles a person who owns many rows, by truncating their rows per group and bounding the groups they touch. [Preparing microdata](06-preparing-microdata.md) covers the expressions OpenDP allows before an aggregation, from casting and filling nulls to string and date operations, and the data types it understands. [Assessing utility and choosing the noise](07-utility-and-noise.md) turns scales into error bars and back. It then compares Laplace with Gaussian noise under zero-concentrated differential privacy, and shows what the error bars leave out.

| Chapter | Question it answers |
| --- | --- |
| [Essential statistics](03-essential-statistics.md) | How do I release a count, a sum, a mean or a quantile, and how large is its error? |
| [Grouping](04-grouping.md) | How do I release one statistic per group, when the groups are known and when they are not? |
| [Identifier truncation and bounds](05-identifier-truncation.md) | What do I do when one person owns many rows? |
| [Preparing microdata](06-preparing-microdata.md) | Which transformations may I apply before aggregating, and how do types and nulls behave? |
| [Assessing utility and choosing the noise](07-utility-and-noise.md) | How do I read an error bar, and when is Gaussian noise the better choice? |

: The chapters of this part and the questions they answer. {#tbl-p2-chapters}

By the end of this part you will be able to release a table of statistics about a dataset in which each person owns many rows. Every cell will carry its error bar, and the whole table a guarantee you can state.
