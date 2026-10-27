# Amplification by shuffling and by iteration

## What it is {#sec-shuf-rr}

In the [shuffle model](../part03-dp-foundations/09-trust-models.md), each of $n$ users randomizes their own data, so each report satisfies local differential privacy (DP) at a level $\varepsilon_0$. A **shuffler** strips identities and mixes the order, so each report hides among the others. The bag of reports then satisfies central DP at a much smaller $\varepsilon$, which falls like $\sqrt{\ln(1/\delta)/n}$ for a slack $\delta$ that must be above 0. All results here are for [replace-one neighbours](05-subsampling.md#sec-sub-schemes).

For binary [randomized response](../part03-dp-foundations/09-trust-models.md), the analyst learns only the count of 1-reports, so the shuffled guarantee can be computed exactly. For any other local randomizer, use the clone bound of Feldman, McMillan and Talwar (2021). With probability $e^{-\varepsilon_0}$, another user's report looks exactly like one you could have sent, and such **clones** hide yours. The **numeric clone bound** computes this exactly, while the paper's closed form is looser and needs enough users.

## When you meet it

Shuffling appears in telemetry systems such as Prochlo (Bittau et al. 2017), where the decision is how large $\varepsilon_0$ can be. No maintained library accounts for it, so use the full edition's [shuffling_iteration.py](../../code/part04-privacy-accounting/shuffling_iteration.py).

Two other things share the name but get none of this amplification. A data loader that shuffles the training set once per epoch is a sampler ([Match the sampler to the accountant](05-subsampling.md#sec-sub-schemes)), Federated clients that join rounds when available pass through no shuffler either, so account them without amplification or with [matrix mechanisms](../part05-dp-ml/04-matrix-mechanisms.md).

You also meet amplification by iteration, where later training noise protects earlier records, in papers that release only the final model (@sec-shuf-hidden).

## How to use it {#sec-shuf-clones}

Work backwards from the target: compute the numeric clone bound for a few values of $\varepsilon_0$, and keep the largest that meets it.

| $n$ | $\varepsilon_0$ | Exact, randomized response | Clones, numeric | Clones, closed form |
| --- | --- | --- | --- | --- |
| 1,000 | 4 | 1.6910 | 3.9900 | n/a |
| 10,000 | 2 | 0.0870 | 0.1550 | 0.5009 |
| 100,000 | 2 | 0.0247 | 0.0450 | 0.1862 |
| 100,000 | 4 | 0.0847 | 0.1698 | 0.5346 |

: Central $\varepsilon$ at $\delta = 10^{-6}$ after shuffling the reports of $n$ users. "n/a" marks too few users for the closed form. {#tbl-shuf-compare}

The choice of bound can decide feasibility. For 100,000 users and a target of 0.2, the closed form allows $\varepsilon_0$ up to 2 but the numeric bound allows 4, so deploy the numeric bound.

Before reporting a shuffled $\varepsilon$, check the bound's assumptions, because if one fails, such as a leaked order or a colluding shuffler, only $\varepsilon_0$ remains.

- **The order is uniformly random and hidden,** with no metadata such as arrival time or network address.
- **The shuffler does not collude with the analyst.**
- **Only honest users count in $n$,** since colluders reveal their reports to the analyst.
- **The count of users is the number of reports shuffled,** so with several batches it is the smallest batch.
- **Each user sends one report per batch,** or the reports compose like any other releases.

## What goes wrong {#sec-shuf-hidden}

Shortcuts fail where users are few or $\varepsilon_0$ is large. For 1,000 users at $\varepsilon_0 = 4$, the closed form says nothing, and a normal approximation of the count of 1-reports gives 0.9667 against the exact 1.6910.

Amplification by iteration covers one pass of noisy stochastic gradient descent (SGD) that keeps the weights in a fixed convex set (Feldman et al. 2018). It needs two conditions. Only the final model may be released, and every step must be a contraction, never moving two sets of weights further apart. That takes a convex, smooth loss and a small enough step. Even then the gain depends on the record. Over 10,000 records with noise multiplier 1, the middle record gets $\varepsilon = 0.0707$ at $\delta = 10^{-5}$, but the last gains nothing and gets 5.2985, as one Gaussian step does. So report the worst record's value.

Deep learning breaks both conditions. Checkpoints, logged models and a best checkpoint picked from several all release more than the final model, and non-convex losses are not contractions. Annamalai (2024) builds one whose final model leaks as much as all the intermediate models together. So claim amplification by iteration only for convex models, such as logistic regression, that release just their final weights.

## References

- Feldman, McMillan & Talwar, "Hiding Among the Clones" (FOCS 2021): the clone bound and how to compute it.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/07-shuffling-iteration.md).
