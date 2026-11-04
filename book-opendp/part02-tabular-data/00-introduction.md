# Tabular data

This part covers private statistics on tables. OpenDP analyses Polars lazy queries before execution, so filters, casts and aggregations are included in the privacy guarantee. Each statistic needs a bound on each person's contribution and a budget share. OpenDP uses both to set the noise scale.

The chapters follow the documentation's tabular-data tutorial. [Essential statistics](03-essential-statistics.md) releases counts, sums, means, medians and quantiles, then interprets the scales and error bars from `summarize`. [Grouping](04-grouping.md) releases one row per group, using public or privately selected keys. [Identifier truncation and bounds](05-identifier-truncation.md) limits each person's rows and affected groups. [Preparing microdata](06-preparing-microdata.md) covers supported pre-aggregation expressions and data types. [Assessing utility and choosing the noise](07-utility-and-noise.md) converts scales to error bars and compares Laplace and Gaussian noise under zero-concentrated differential privacy.

| Chapter | Question it answers |
| --- | --- |
| [Essential statistics](03-essential-statistics.md) | How do I release a count, a sum, a mean or a quantile, and how large is its error? |
| [Grouping](04-grouping.md) | How do I release one statistic per group, when the groups are known and when they are not? |
| [Identifier truncation and bounds](05-identifier-truncation.md) | What do I do when one person owns many rows? |
| [Preparing microdata](06-preparing-microdata.md) | Which transformations may I apply before aggregating, and how do types and nulls behave? |
| [Assessing utility and choosing the noise](07-utility-and-noise.md) | How do I read an error bar, and when is Gaussian noise the better choice? |

: The chapters of this part and the questions they answer. {#tbl-p2-chapters}

By the end of this part, you can release statistics on data with multiple rows per person, interpret cell-level error bars and state the table's privacy guarantee.
