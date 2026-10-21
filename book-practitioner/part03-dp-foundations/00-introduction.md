# Differential privacy foundations

The previous part showed that anonymization and access control do not stop releases from leaking. This part introduces **differential privacy (DP)**, the defence the rest of the book builds on. It is a property of the release process, not of a dataset. Changing one person's data may change the probability of any output by at most a factor $e^\varepsilon$, set by a privacy level $\varepsilon$, plus a small slack $\delta$. That one condition caps every attack of Part 2. So every DP claim must state three things, namely $\varepsilon$, $\delta$ and the unit of privacy, such as a record, a user or a user-day.

| Symbol | Meaning |
| --- | --- |
| $\varepsilon$, $\delta$ | Privacy level and slack |
| $\Delta_1f$, $\Delta_2f$ | Sensitivity, the most one record moves the answer |
| $\mathrm{Lap}(b)$, $\mathcal N(0, \sigma^2)$ | Laplace noise of scale $b$, normal noise of standard deviation $\sigma$ |

: Notation used throughout Part 3. {#tbl-dpf-notation}

The part teaches one workflow, and its last page carries it out. First fix the unit of privacy and the neighbouring relation, as [The definition of differential privacy](01-definition.md) explains. Then bound each person's contribution and calibrate a mechanism to it ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). Next, account for every release, as [Post-processing, group privacy and composition](02-properties.md) explains. Post-processing the outputs is free. [Release statistics with a library](12-release-statistics.md) does all of this with OpenDP on a real shop's invoices, from bounding each customer's lines to the written guarantee.

The full chapters between them cover tasks you meet often. [Selection](05-selection.md) picks the best of many options or a private median. [Query release](06-query-release.md) publishes many counts or marginals at once. [Trust models](09-trust-models.md) apply when nobody can be trusted with the raw data, and [Private mean estimation](11-mean-estimation.md) says how many records a private mean needs.

Four short reference pages cover the rest. [Advanced composition](04-composition.md) and [Beyond global sensitivity](07-sensitivity.md) matter when the simple rules are too loose. Read [Discrete mechanisms](08-discrete-mechanisms.md) before you sample noise yourself, and [Lower bounds](10-lower-bounds.md) before you promise an accuracy.

[Part 4](../part04-privacy-accounting/00-introduction.md) then replaces hand composition with accountants, and [Part 5](../part05-dp-ml/00-introduction.md) applies all of it to training models. The code here runs OpenDP and diffprivlib on the Adult census data and on the UCI Online Retail invoices. Every number comes from a snippet or a script you can run. By the end, you will be able to release a count, a sum, a median or a table with error bars and a guarantee you can defend.
