# Differential privacy in machine learning

This part is for anyone who trains, ships or reviews a model under differential privacy (DP). It builds on the Gaussian mechanism and composition from [Part 3](../part03-dp-foundations/00-introduction.md) and an accountant from [Part 4](../part04-privacy-accounting/00-introduction.md). Its audits reuse the membership-inference attacks of [Part 2](../part02-attacks/10-risk-assessment.md).

Every private training method has two halves. The privacy half, the guarantee $\varepsilon$, is computed by an accountant, while the utility half, accuracy, can only be measured. The main private algorithm is stochastic gradient descent (SGD) with clipped per-example gradients and Gaussian noise, called DP-SGD. Its accountant needs just three settings: the noise multiplier, the sampling rate and the number of steps. So every page asks whether the $\varepsilon$ is correct and whether the accuracy at that $\varepsilon$ is good enough.

Three pages are workflows, each a whole job done with real libraries on public data. [Train a model with DP-SGD](13-train-dp-sgd.md) trains a classifier on the Adult census data with Opacus. [Fine-tune a language model privately with LoRA](14-private-llm-finetuning.md) adapts a small public language model to a private text and checks it for memorized secrets. [Release and governance](12-release-governance.md) recomputes the guarantee, writes it down, maps it to the law and lists what to check before a model ships.

The other pages explain the decisions those jobs make, in the order a project meets them. A short page on [private empirical risk minimization](01-dp-erm.md) for convex models leads to [DP-SGD](02-dp-sgd.md). Then come [hyperparameter tuning](03-hyperparameter-tuning.md), the privacy unit in [User-level differential privacy](07-user-level.md) and the budget in [Choosing ε](09-choosing-epsilon.md). [Public data and private fine-tuning](06-public-data.md) explains why a public starting model helps so much. [Disparate impact](08-disparate-impact.md) and [Auditing private training](10-auditing.md) are the checks before release. Three reference pages cover rarer methods: [private prediction](05-private-prediction.md), [correlated noise across steps](04-matrix-mechanisms.md) and [adaptive data analysis](11-adaptive-data-analysis.md).

Every number on these pages comes from a script you can run.

