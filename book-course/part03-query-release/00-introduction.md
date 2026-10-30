# Query release and its limits

The second part built mechanisms that answer one question, or a few, under differential privacy (DP). This part asks what happens when an analyst wants many answers from the same dataset. Its first two chapters answer many queries for far less than one budget share per query. The third chapter lets the noise follow the dataset at hand instead of the worst case. The last proves that, under pure DP, no mechanism can do much better than the simple ones on marginals and histograms.

| Chapter | Question it answers |
| --- | --- |
| [Private multiplicative weights](01-multiplicative-weights.md) | How many records are needed to answer a large set of linear queries at once? |
| [Sparse vector](02-sparse-vector.md) | How can a stream of queries be answered when the cost should depend only on the few large ones? |
| [Beyond global sensitivity](03-beyond-global-sensitivity.md) | When can the noise follow the sensitivity of the actual dataset instead of the worst case? |
| [Packing lower bounds](04-packing-lower-bounds.md) | How much error must any pure DP mechanism incur on marginals and histograms? |

: The chapters of this part. {#tbl-p3-chapters}
