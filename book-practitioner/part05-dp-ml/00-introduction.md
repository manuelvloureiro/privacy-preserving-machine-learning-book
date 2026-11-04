# Differential privacy in machine learning

This part covers model training, deployment and review under differential privacy (DP). It builds on the Gaussian mechanism and composition in [Part 3](../part03-dp-foundations/00-introduction.md), the accountants in [Part 4](../part04-privacy-accounting/00-introduction.md) and the membership-inference audits in [Part 2](../part02-attacks/10-risk-assessment.md).

Private training produces a privacy guarantee and a model. An accountant computes $\varepsilon$; evaluation measures utility, such as accuracy. The main algorithm is stochastic gradient descent (SGD) with clipped per-example gradients and Gaussian noise (DP-SGD). Its accountant depends on the noise multiplier, sampling rate and step count. The chapters assess both the validity of $\varepsilon$ and model utility at that level.

Three pages present end-to-end workflows using real libraries and public data. [Train a model with DP-SGD](13-train-dp-sgd.md) trains an Adult classifier with Opacus. [Fine-tune a language model privately with LoRA](14-private-llm-finetuning.md) adapts a small public model to private text and tests for memorized information. [Release and governance](12-release-governance.md) recomputes and documents the guarantee, relates it to law, and gives pre-release checks.

The other chapters explain workflow decisions. [Private empirical risk minimization](01-dp-erm.md) covers convex models, followed by [DP-SGD](02-dp-sgd.md), [hyperparameter tuning](03-hyperparameter-tuning.md), [user-level differential privacy](07-user-level.md) and [choosing ε](09-choosing-epsilon.md). [Public data and private fine-tuning](06-public-data.md) covers public starting models. [Disparate impact](08-disparate-impact.md) and [Auditing private training](10-auditing.md) address pre-release checks. Reference pages cover [private prediction](05-private-prediction.md), [correlated noise across steps](04-matrix-mechanisms.md) and [adaptive data analysis](11-adaptive-data-analysis.md).

Scripts check every reported number.
