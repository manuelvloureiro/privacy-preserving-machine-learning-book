# Differential privacy foundations

Part 2 showed that anonymization and access control do not prevent leakage. This part introduces **differential privacy (DP)**, a property of a release process rather than a dataset. If one person's data changes, the probability of each output changes by at most a factor $e^\varepsilon$, with additive slack $\delta$. A DP claim must state $\varepsilon$, $\delta$ and the unit of privacy, such as a record or a user. This guarantee limits the attacks covered in Part 2.

| Symbol | Meaning |
| --- | --- |
| $\varepsilon$, $\delta$ | Privacy level and slack |
| $\Delta_1f$, $\Delta_2f$ | Sensitivity, the most one record moves the answer |
| $\mathrm{Lap}(b)$, $\mathcal N(0, \sigma^2)$ | Laplace noise of scale $b$, normal noise of standard deviation $\sigma$ |

: Notation used throughout Part 3. {#tbl-dpf-notation}

The workflow has three steps. First, specify the unit of privacy and neighbouring relation ([the definition of differential privacy](01-definition.md)). Next, bound each person's contribution and calibrate a mechanism to that bound ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). Finally, account for every release; only post-processing is free ([Post-processing, group privacy and composition](02-properties.md)). [Release statistics with a library](12-release-statistics.md) applies these steps to a shop's invoices using OpenDP.

The full chapters cover common tasks. [Selection](05-selection.md) chooses the best of several options or a private median; [Query release](06-query-release.md) publishes multiple counts. [Trust models](09-trust-models.md) compares settings where no central party can hold raw data, and [Private mean estimation](11-mean-estimation.md) calculates the sample size required for a private mean. Four reference pages cover advanced composition and sensitivity, discrete noise mechanisms and accuracy lower bounds.

[Part 4](../part04-privacy-accounting/00-introduction.md) then replaces hand composition with accountants, and [Part 5](../part05-dp-ml/00-introduction.md) uses all of it to train models. At the end of the part, you can release a count, sum, median or table with error bars and a privacy guarantee.
