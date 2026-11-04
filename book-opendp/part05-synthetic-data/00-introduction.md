# Synthetic data

This part releases a dataset rather than a statistic. A synthetic dataset contains generated rows with statistics that approximate the real data, subject to a privacy budget. OpenDP first releases a contingency table of noisy counts for combinations of columns. It then fits a model to the noisy marginals and samples rows, a post-processing step that uses no additional budget.

Synthetic-data quality depends on the measured marginals and budget allocation. [Contingency tables](16-contingency-tables.md) releases fixed-workload marginals with the `Fixed` algorithm and explains keys, cuts, thresholds and `ContingencyTable` error estimates. [Synthetic data with AIM and MST](17-aim-and-mst.md) selects marginals iteratively with AIM or along a spanning tree with MST, then generates rows. [Evaluating and tuning synthetic data](18-evaluating-synthetic-data.md) compares synthetic and real data on measured marginals, an unmeasured query and a downstream model, and examines how the budget and workload affect accuracy.

| Chapter | Question it answers |
| --- | --- |
| [Contingency tables](16-contingency-tables.md) | How do I release the noisy counts of a set of marginals, and what does the table object hold? |
| [Synthetic data with AIM and MST](17-aim-and-mst.md) | How do I let the library choose the marginals, and how do I generate synthetic rows? |
| [Evaluating and tuning synthetic data](18-evaluating-synthetic-data.md) | How good is the synthetic data, and what changes it? |

: The chapters of this part and the questions they answer. {#tbl-p5-chapters}

By the end of this part, you can generate a private synthetic dataset, state its guarantee and measure its differences from the source data.
