# Differential privacy foundations

Part 2 showed that anonymization and access control do not stop releases from leaking. This part introduces **differential privacy (DP)**, the defence the rest of the book builds on. It is a property of the release process, not of a dataset. Changing one person's data may change the probability of any output by at most a factor $e^\varepsilon$, set by a privacy level $\varepsilon$, plus a small slack $\delta$. That one condition caps every attack of Part 2. So a DP claim must state $\varepsilon$, $\delta$ and what counts as one person's data, the unit of privacy, such as a record or a user.

| Symbol | Meaning |
| --- | --- |
| $\varepsilon$, $\delta$ | Privacy level and slack |
| $\Delta_1f$, $\Delta_2f$ | Sensitivity, the most one record moves the answer |
| $\mathrm{Lap}(b)$, $\mathcal N(0, \sigma^2)$ | Laplace noise of scale $b$, normal noise of standard deviation $\sigma$ |

: Notation used throughout Part 3. {#tbl-dpf-notation}

The part teaches one workflow. First fix the unit of privacy and the neighbouring relation ([the definition of differential privacy](01-definition.md)). Then bound each person's contribution and calibrate a mechanism to it ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). Finally, account for every release, since only post-processing is free ([Post-processing, group privacy and composition](02-properties.md)). The last page, [Release statistics with a library](12-release-statistics.md), runs this workflow with OpenDP on a real shop's invoices, from bounding each customer's lines to the written guarantee.

The full chapters in between cover common tasks. [Selection](05-selection.md) picks the best of many options or a private median, and [Query release](06-query-release.md) publishes many counts at once. [Trust models](09-trust-models.md) apply when nobody can be trusted with the raw data, and [Private mean estimation](11-mean-estimation.md) says how many records a private mean needs.

Four short reference pages cover the rest. [Advanced composition](04-composition.md) and [Beyond global sensitivity](07-sensitivity.md) matter when the simple rules are too loose. Read [Discrete mechanisms](08-discrete-mechanisms.md) before you sample noise yourself, and [Lower bounds](10-lower-bounds.md) before you promise an accuracy.

[Part 4](../part04-privacy-accounting/00-introduction.md) then replaces hand composition with accountants, and [Part 5](../part05-dp-ml/00-introduction.md) applies all of it to training models. The code here runs OpenDP and diffprivlib on the Adult census data and the UCI Online Retail invoices, and every number comes from a snippet or script you can run. By the end, you can release a count, a sum, a median or a table with error bars and a guarantee you can defend.
