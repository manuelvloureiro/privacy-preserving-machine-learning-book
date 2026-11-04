# Query release and its limits

The previous part introduced mechanisms for answering a few questions under differential privacy (DP). This part studies how to answer many queries on one dataset. The first two chapters avoid assigning a separate privacy-budget share to every query. The third calibrates noise to the dataset instead of the worst case. The last proves that pure DP limits improvements over simple methods for marginals and histograms.

| Chapter | Question it answers |
| --- | --- |
| [Private multiplicative weights](01-multiplicative-weights.md) | How many records are needed to answer a large set of linear queries at once? |
| [Sparse vector](02-sparse-vector.md) | How can a stream of queries be answered when the cost should depend only on the few large ones? |
| [Beyond global sensitivity](03-beyond-global-sensitivity.md) | When can the noise follow the sensitivity of the actual dataset instead of the worst case? |
| [Packing lower bounds](04-packing-lower-bounds.md) | How much error must any pure DP mechanism incur on marginals and histograms? |

: The chapters of this part. {#tbl-p3-chapters}
