# Differential privacy foundations

Part 2 showed that anonymization and access control do not stop a release from leaking. This part introduces **differential privacy (DP)**, a property of the release process, not of a dataset. If the data of one person changes, the probability of each output changes by at most a factor $e^\varepsilon$, plus a small slack $\delta$. Here $\varepsilon$ is the privacy level. This one condition limits every attack of Part 2. So a DP claim must state $\varepsilon$, $\delta$ and the unit of privacy, such as a record or a user.

| Symbol | Meaning |
| --- | --- |
| $\varepsilon$, $\delta$ | Privacy level and slack |
| $\Delta_1f$, $\Delta_2f$ | Sensitivity, the most one record moves the answer |
| $\mathrm{Lap}(b)$, $\mathcal N(0, \sigma^2)$ | Laplace noise of scale $b$, normal noise of standard deviation $\sigma$ |

: Notation used throughout Part 3. {#tbl-dpf-notation}

The part teaches one workflow in three steps. First, fix the unit of privacy and the neighbouring relation ([the definition of differential privacy](01-definition.md)). Then bound each person's contribution and calibrate a mechanism to it ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). Last, account for every release, because only post-processing is free ([Post-processing, group privacy and composition](02-properties.md)). [Release statistics with a library](12-release-statistics.md) does all three steps with OpenDP on the invoices of a real shop.

The full chapters between them cover common tasks. [Selection](05-selection.md) picks the best of many options or a private median, and [Query release](06-query-release.md) publishes many counts at once. Use [Trust models](09-trust-models.md) when nobody can hold the raw data, and [Private mean estimation](11-mean-estimation.md) to find how many records a private mean needs. Of the four reference pages, [Advanced composition](04-composition.md) and [Beyond global sensitivity](07-sensitivity.md) help when the simple rules are too loose. Read [Discrete mechanisms](08-discrete-mechanisms.md) before you sample noise yourself, and [Lower bounds](10-lower-bounds.md) before you promise an accuracy.

[Part 4](../part04-privacy-accounting/00-introduction.md) then replaces hand composition with accountants, and [Part 5](../part05-dp-ml/00-introduction.md) uses all of it to train models. At the end of the part, you can release a count, a sum, a median or a table with error bars and a guarantee.
