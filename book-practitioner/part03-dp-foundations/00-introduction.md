# Differential privacy foundations

The previous part showed that anonymization and access control do not stop releases from leaking. This part introduces **differential privacy (DP)**, the defence the rest of the book builds on. It is a property of the release process, not of a dataset. Changing one person's data may change the probability of any output by at most a factor $e^\varepsilon$, set by a privacy level $\varepsilon$, plus a small slack $\delta$. That one condition caps every attack of Part 2. So every DP claim must state three things, namely $\varepsilon$, $\delta$ and the unit of privacy, such as a record, a user or a user-day.

| Symbol | Meaning |
| --- | --- |
| $\varepsilon$, $\delta$ | Privacy level and slack |
| $\Delta_1f$, $\Delta_2f$ | Sensitivity, the most one record moves the answer |
| $\mathrm{Lap}(b)$, $\mathcal N(0, \sigma^2)$ | Laplace noise of scale $b$, normal noise of standard deviation $\sigma$ |

: Notation used throughout Part 3. {#tbl-dpf-notation}

The part teaches one workflow. First fix the unit of privacy and the neighbouring relation, as [The definition of differential privacy](01-definition.md) explains. Then clip the data, bound its sensitivity and calibrate a mechanism to it ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). Next, account for every release ([Post-processing, group privacy and composition](02-properties.md) and [Advanced and optimal composition](04-composition.md)). Finally, post-process the outputs freely, since that costs nothing.

You need those four core chapters to ship a private statistic, while the next ones are per task. [Selection](05-selection.md) picks the best of many options. [Query release](06-query-release.md) answers many queries through a synthetic dataset and [Beyond global sensitivity](07-sensitivity.md) handles data whose worst case is far from typical. Read [Discrete and floating-point-safe mechanisms](08-discrete-mechanisms.md) before sampling noise yourself, and [Local, shuffle and distributed trust models](09-trust-models.md) when nobody can be trusted with the raw data. The last two chapters help you size a study. [Lower bounds](10-lower-bounds.md) says when an accuracy is impossible at your $\varepsilon$, and [Private mean estimation](11-mean-estimation.md) says how many records a private mean needs.

The full edition proves every result, and this edition shows how to use them. [Part 4](../part04-privacy-accounting/00-introduction.md) then replaces hand composition with accountants, and [Part 5](../part05-dp-ml/00-introduction.md) applies all of it to training models. The tools used from here on are OpenDP, Tumult Analytics, diffprivlib, Google's `dp_accounting` and Opacus.
