# Query Release and Its Limits

The second part built mechanisms that answer a single question, or a few, under differential privacy (DP). This part asks what happens when an analyst wants many answers from the same dataset, and how far the mechanisms of the second part can be pushed.

The first two chapters answer many queries while paying far less than one budget share per query. [Private multiplicative weights](01-multiplicative-weights.md) answers a large set of linear queries with a number of records that grows only logarithmically in the number of queries and in the size of the universe. It borrows an algorithm from online learning and touches the data only through the exponential and Laplace mechanisms. [Sparse vector](02-sparse-vector.md) handles queries that arrive one at a time. When only a few of them exceed a threshold, it pays for those few and only logarithmically for the rest. It also turns private multiplicative weights into an online algorithm.

The third chapter questions a choice made in every mechanism so far: noise calibrated to the global sensitivity, the worst case over all pairs of neighbouring datasets. [Calibrating to the dataset at hand](03-beyond-global-sensitivity.md) is tempting, and naively it is not private. The chapter shows four ways to do it safely: testing privately that the dataset is far from any dataset with high sensitivity, a stability argument built on that test for the mode and for histograms, bounding the local sensitivity privately before using it, and smoothing the sensitivity across nearby datasets.

The last chapter turns from algorithms to limits. [Packing arguments](04-packing-lower-bounds.md) prove that, under pure DP, no mechanism can answer one-way marginals or histograms with error much smaller than the simple mechanisms already achieve. These lower bounds hold for every algorithm, however clever.

| Chapter | Question it answers |
| --- | --- |
| [Private multiplicative weights](01-multiplicative-weights.md) | How many records are needed to answer a large set of linear queries at once? |
| [Sparse vector](02-sparse-vector.md) | How can a stream of queries be answered when the cost should depend only on the few large ones? |
| [Beyond global sensitivity](03-beyond-global-sensitivity.md) | When can the noise follow the sensitivity of the actual dataset instead of the worst case? |
| [Packing lower bounds](04-packing-lower-bounds.md) | How much error must any pure DP mechanism incur on marginals and histograms? |

: The chapters of this part. {#tbl-p3-chapters}
