# Synthetic data

The last part releases a dataset instead of a statistic. A synthetic dataset is a table of made-up rows whose statistics match those of the real table as closely as a privacy budget allows. OpenDP makes one in two steps. First it releases a contingency table, which holds the noisy counts of every combination of a few columns at a time. Then it fits a model to those noisy marginals and samples rows from it, which is post-processing and costs no more budget.

The thread of the part is that two choices decide the quality of the synthetic data: which marginals you measure, and how you split the budget among them. [Contingency tables](16-contingency-tables.md) releases the marginals of a fixed workload with the `Fixed` algorithm. It explains keys, cuts and thresholds, and the `ContingencyTable` object with its error estimates. [Synthetic data with AIM and MST](17-aim-and-mst.md) lets the library choose the marginals, one round at a time with AIM or along a spanning tree with MST, and generates the rows. [Evaluating and tuning synthetic data](18-evaluating-synthetic-data.md) compares the result with the real data on the marginals, on a query outside the workload and on a downstream model. It also shows how the budget and the workload change the result.

| Chapter | Question it answers |
| --- | --- |
| [Contingency tables](16-contingency-tables.md) | How do I release the noisy counts of a set of marginals, and what does the table object hold? |
| [Synthetic data with AIM and MST](17-aim-and-mst.md) | How do I let the library choose the marginals, and how do I generate synthetic rows? |
| [Evaluating and tuning synthetic data](18-evaluating-synthetic-data.md) | How good is the synthetic data, and what changes it? |

: The chapters of this part and the questions they answer. {#tbl-p5-chapters}

At the end of this part you can generate a private synthetic copy of a table, state its guarantee, and measure how far it is from the original.
