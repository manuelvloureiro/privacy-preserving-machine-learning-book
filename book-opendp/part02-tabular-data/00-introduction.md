# Tabular data

This part does the everyday work of a private analysis: statistics on a table. OpenDP reads the lazy queries of Polars, a dataframe library, before they run. So every filter, cast and aggregation is part of the privacy guarantee. The thread of the part is that every statistic needs two things from you: a bound on what one person can contribute, and a share of the budget. The library turns them into a noise scale, which you read before a release.

The chapters follow the documentation's tutorial on tabular data. [Essential statistics](03-essential-statistics.md) releases counts, sums, means, medians and quantiles, with the noise scale and error bar that `summarize` reports. [Grouping](04-grouping.md) releases one row per group, with public keys or keys that the library selects privately. [Identifier truncation and bounds](05-identifier-truncation.md) truncates the rows of a person who owns many, and bounds the groups that the person touches. [Preparing microdata](06-preparing-microdata.md) covers the expressions that OpenDP allows before an aggregation, and the data types it knows. [Assessing utility and choosing the noise](07-utility-and-noise.md) turns scales into error bars and back, and compares Laplace with Gaussian noise under zero-concentrated differential privacy.

| Chapter | Question it answers |
| --- | --- |
| [Essential statistics](03-essential-statistics.md) | How do I release a count, a sum, a mean or a quantile, and how large is its error? |
| [Grouping](04-grouping.md) | How do I release one statistic per group, when the groups are known and when they are not? |
| [Identifier truncation and bounds](05-identifier-truncation.md) | What do I do when one person owns many rows? |
| [Preparing microdata](06-preparing-microdata.md) | Which transformations may I apply before aggregating, and how do types and nulls behave? |
| [Assessing utility and choosing the noise](07-utility-and-noise.md) | How do I read an error bar, and when is Gaussian noise the better choice? |

: The chapters of this part and the questions they answer. {#tbl-p2-chapters}

At the end of this part, you can release a table of statistics on a dataset in which each person owns many rows. Each cell has an error bar, and the whole table has a guarantee.
