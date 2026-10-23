# Part 5 — Differential privacy in machine learning

How to train models with differential privacy: where noise can enter training, DP-SGD and its knobs, hyperparameter tuning, DP-FTRL and matrix-factorization mechanisms, private prediction with PATE and in-context learning, public pretraining and private fine-tuning, user-level privacy, disparate impact, choosing $\varepsilon$, auditing a trained model, and adaptive data analysis: why reusing a holdout set overfits and how DP prevents it.

Start with the [introduction](00-introduction.md), which is part of the book. It assumes the [prerequisites](../part01-prerequisites/README.md), the [attacks](../part02-attacks/README.md) part (membership inference), the [DP foundations](../part03-dp-foundations/README.md) and [privacy accounting](../part04-privacy-accounting/README.md), especially amplification by subsampling.

This file is the GitHub overview. It is not part of the rendered book.

## Chapters

| Ch. | Chapter | Covers | Script |
|---|---------|--------|--------|
| 38 | [Private empirical risk minimization](01-dp-erm.md) | Output, objective and gradient perturbation; excess risk and optimal rates; noisy projected gradient descent and the $\sqrt d/(n\varepsilon)$ rate without a ridge; what survives without convexity | [dp_erm.py](../../code/part05-dp-ml/dp_erm.py) |
| 39 | [DP-SGD](02-dp-sgd.md) | The algorithm and its guarantee; clipping; batch size at a fixed budget; ghost clipping; microbatches (twice the sensitivity, $2m$ times the noise) against vectorized per-example gradients; architecture and model size; privacy against accuracy | [dp_sgd.py](../../code/part05-dp-ml/dp_sgd.py) |
| 40 | [Hyperparameter tuning and private selection](03-hyperparameter-tuning.md) | Why tuning leaks; composition; random-stopping and Rényi-DP tuning; private selection | [hyperparameter_tuning.py](../../code/part05-dp-ml/hyperparameter_tuning.py) |
| 41 | [DP-FTRL and matrix factorization](04-matrix-mechanisms.md) | Prefix sums; tree aggregation; the matrix mechanism; correlated noise without amplification | [matrix_mechanisms.py](../../code/part05-dp-ml/matrix_mechanisms.py) |
| 42 | [Private prediction: PATE and in-context learning](05-private-prediction.md) | Subsample-and-aggregate for predictions; noisy argmax; PATE; stable aggregation with a worst-case budget, and why it rarely pays; private in-context learning | [private_prediction.py](../../code/part05-dp-ml/private_prediction.py) |
| 43 | [Public data and private fine-tuning](06-public-data.md) | Why dimension costs; linear probing; low-rank fine-tuning; caveats of public data | [public_data.py](../../code/part05-dp-ml/public_data.py) |
| 44 | [User-level differential privacy](07-user-level.md) | The privacy unit; group privacy; user-level DP-SGD; contribution bounding | [user_level.py](../../code/part05-dp-ml/user_level.py) |
| 45 | [Disparate impact](08-disparate-impact.md) | Clipping bias and noise by group; measuring per-group accuracy; mitigations | [disparate_impact.py](../../code/part05-dp-ml/disparate_impact.py) |
| 46 | [Choosing ε](09-choosing-epsilon.md) | What $\varepsilon$ bounds for an attacker; trade-off curves; deployed values; how to justify a choice | [choosing_epsilon.py](../../code/part05-dp-ml/choosing_epsilon.py) |
| 47 | [Auditing private training](10-auditing.md) | Lower bounds on $\varepsilon$ from attacks; canaries; tight and one-run audits; finding bugs | [auditing.py](../../code/part05-dp-ml/auditing.py) |
| 48 | [Adaptive data analysis](11-adaptive-data-analysis.md) | Holdout reuse and leaderboard attacks; sample splitting; DP implies generalization; the √k rate; Thresholdout | [adaptive_data_analysis.py](../../code/part05-dp-ml/adaptive_data_analysis.py) |

Each script prints the numbers its chapter quotes, except values the text attributes to a paper, and regenerates the chapter's figures in `_build/figures/part05-dp-ml/` (the build copies them into each book):

```bash
uv run python code/part05-dp-ml/<script>.py
```

Every chapter's exercises, each with a worked solution, are in the exercise book, [book-exercises/part05-dp-ml/](../../book-exercises/part05-dp-ml/), one file per chapter. The exercise book is its own website, and `code/part05-dp-ml/solutions_<script>.py` prints the numbers they quote.

## Suggested order

1. Chapters 38 → 39. Every later chapter uses DP-SGD.
2. Chapters 40 and 41, in either order: how the algorithm is run.
3. Chapters 42 to 45, in any order: other ways to spend the budget, public data, the privacy unit, and who pays.
4. Chapter 46, then chapter 47, which checks what the others claim.
5. Chapter 48 at any point after chapter 40: adaptive data analysis, where DP keeps a reused holdout set honest.

## Core readings

- Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016) → chapter 39
- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023) → chapters 39, 40, 46
- De et al., "Unlocking High-Accuracy Differentially Private Image Classification through Scale" (2022) → chapters 39, 43
- Li et al., "Large Language Models Can Be Strong Differentially Private Learners" (ICLR 2022) → chapters 39, 43
- Papernot et al., "Scalable Private Learning with PATE" (ICLR 2018) → chapter 42
- Jagielski, Ullman & Oprea, "Auditing Differentially Private Machine Learning: How Private is Private SGD?" (NeurIPS 2020) → chapter 47
- Steinke, Nasr & Jagielski, "Privacy Auditing with One (1) Training Run" (NeurIPS 2023) → chapter 47
- Dwork, Feldman, Hardt, Pitassi, Reingold & Roth, "The Reusable Holdout: Preserving Validity in Adaptive Data Analysis" (Science 2015) → chapter 48

## Further readings

- Chaudhuri, Monteleoni & Sarwate (JMLR 2011); Bassily, Smith & Thakurta (FOCS 2014) → chapter 38
- Kairouz et al. (ICML 2021); Choquette-Choo et al. (ICML 2023) → chapter 41
- Papernot & Steinke, "Hyperparameter Tuning with Renyi Differential Privacy" (ICLR 2022) → chapter 40
- Tramèr & Boneh (ICLR 2021); Yu et al. (ICLR 2022) → chapter 43
- Bagdasaryan, Poursaeed & Shmatikov (NeurIPS 2019) → chapter 45
- Nasr et al. (S&P 2021); Nasr et al. (USENIX Security 2023) → chapter 47
- Duan et al. (NeurIPS 2023); Wu et al. (ICLR 2024) → chapter 42
- Desfontaines, "A list of real-world uses of differential privacy" → chapter 46
- Chua et al., "Mind the Privacy Unit! User-Level Differential Privacy for Language Model Fine-Tuning" (2024) → chapter 44
- Blum & Hardt, "The Ladder" (ICML 2015); Bassily et al., "Algorithmic Stability for Adaptive Data Analysis" (STOC 2016) → chapter 48

Each chapter ends with its own references, with what each is good for.

## Exercise

Implement DP-SGD from scratch in PyTorch (per-example clipping via `torch.func.vmap`), match Opacus on CIFAR-10 at $\varepsilon = 8$, then audit both models with your LiRA pipeline from Part 2 using inserted canaries.

DP-SGD is implemented from scratch in NumPy, with ghost clipping, in [DP-SGD](02-dp-sgd.md), on a small network whose experiments run in seconds. The audits with inserted canaries are in [Auditing private training](10-auditing.md). The CIFAR-10 run, the comparison with Opacus and the PyTorch implementation are left to you, since PyTorch and Opacus are not dependencies of this repository.

## Exit check

You are done with this part when you can do each of these on paper, without notes:

- [ ] Derive the sensitivity of the regularized ERM minimizer, and calibrate output perturbation to it.
- [ ] Prove that objective perturbation is $\varepsilon$-DP by a change of variables, and say where its budget goes.
- [ ] Show that noisy gradient descent is private for any loss with bounded gradients, and why convexity is needed only for accuracy.
- [ ] Write DP-SGD step by step, and name every condition of its privacy proof that an implementation can break.
- [ ] Explain why only $\eta C$ matters when every gradient is clipped, and what clipping does to the expected update.
- [ ] Explain why larger batches need less noise per averaged gradient at a fixed budget and number of epochs.
- [ ] Compute a per-example gradient norm for a dense layer without forming the gradient.
- [ ] Explain why choosing the best of several private runs leaks, and how to pay for tuning.
- [ ] Write a prefix-sum workload as a matrix factorization and compute its sensitivity and error.
- [ ] Describe PATE and compute the privacy cost of one noisy-argmax query.
- [ ] Explain why fine-tuning fewer parameters can be more accurate under DP.
- [ ] Convert an example-level guarantee into a user-level one, and describe user-level DP-SGD.
- [ ] Explain the two reasons DP-SGD hurts small groups more.
- [ ] Turn an $(\varepsilon, \delta)$ guarantee into a bound on a membership attacker's TPR at a given FPR.
- [ ] Turn an audit's observed TPR and FPR into a lower bound on $\varepsilon$ with a confidence level.
- [ ] Show with one adaptive query why exact holdout answers overfit, and say what sample splitting costs.
- [ ] Prove that a query chosen by an $(\varepsilon, \delta)$-DP algorithm has expected empirical and population values within $e^\varepsilon - 1 + \delta$, using the replace-one swap.
- [ ] Explain why Gaussian noise lets a holdout set answer about n² adaptive queries, and why Thresholdout pays only for overfitting detections.
