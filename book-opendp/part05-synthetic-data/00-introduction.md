# Synthetic data

The last part releases not a statistic but a dataset. A synthetic dataset is a table of made-up rows whose statistics match the real table's as closely as a privacy budget allows. An analyst who never sees the real data can still run queries on it, fit models and draw charts. OpenDP builds one in two steps. First it releases a contingency table, which holds the noisy counts of every combination of a few columns at a time. Then it fits a model to those noisy marginals and samples rows from it. The second step is post-processing, so it costs no further budget. The thread of the part is that the quality of the synthetic data is decided by which marginals are measured and how the budget is split among them.

[Contingency tables](16-contingency-tables.md) releases the marginals of a fixed workload with the `Fixed` algorithm. It explains keys, cuts and the thresholds that handle categories the data reveals, and shows the `ContingencyTable` object with its projections and error estimates. [Synthetic data with AIM and MST](17-aim-and-mst.md) lets the library choose the marginals adaptively with the AIM algorithm, or along a spanning tree with MST, and generates the synthetic rows. [Evaluating and tuning synthetic data](18-evaluating-synthetic-data.md) measures the result against the real data. It checks the marginals, a query the workload did not contain and a downstream model, and shows how the budget and the workload change the answer.

| Chapter | Question it answers |
| --- | --- |
| [Contingency tables](16-contingency-tables.md) | How do I release the noisy counts of a set of marginals, and what does the table object hold? |
| [Synthetic data with AIM and MST](17-aim-and-mst.md) | How do I let the library choose the marginals, and how do I generate synthetic rows? |
| [Evaluating and tuning synthetic data](18-evaluating-synthetic-data.md) | How good is the synthetic data, and what changes it? |

: The chapters of this part and the questions they answer. {#tbl-p5-chapters}

By the end of this part you will be able to generate a differentially private synthetic copy of a table and state its guarantee. You will also be able to measure how far the copy is from the original.
