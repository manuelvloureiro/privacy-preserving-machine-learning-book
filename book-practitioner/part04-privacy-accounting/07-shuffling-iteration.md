# Amplification by shuffling and by iteration

## What it is {#sec-shuf-rr}

In the [shuffle model](../part03-dp-foundations/09-trust-models.md), each of $n$ users randomizes their data, producing a report that satisfies local differential privacy (DP) at level $\varepsilon_0$. A **shuffler** removes identities and randomizes report order. The combined reports then satisfy central DP at a smaller $\varepsilon$, which scales as $\sqrt{\ln(1/\delta)/n}$ for $\delta>0$. Results here use [replace-one neighbours](05-subsampling.md#sec-sub-schemes).

For binary [randomized response](../part03-dp-foundations/09-trust-models.md), the analyst observes only the count of 1-reports, allowing an exact shuffled guarantee. For other local randomizers, use the clone bound of Feldman, McMillan and Talwar (2021). Another user's report matches a possible target report with probability $e^{-\varepsilon_0}$; these **clones** conceal the target's report. The **numeric clone bound** computes the guarantee directly. The paper's closed form is looser and requires enough users.

## When you meet it

Shuffling is used in telemetry systems such as Prochlo (Bittau et al. 2017), where you must choose $\varepsilon_0$. No maintained library supports this accounting, so use [shuffling_iteration.py](../../code/part04-privacy-accounting/shuffling_iteration.py) from the full edition.

Two other uses of "shuffle" do not provide this amplification. A data loader that shuffles once per epoch is a sampler ([Match the sampler to the accountant](05-subsampling.md#sec-sub-schemes)). Federated clients that join when available do not pass through a shuffler; account for them without amplification or use [matrix mechanisms](../part05-dp-ml/04-matrix-mechanisms.md).

## How to use it {#sec-shuf-clones}

Start with the central target. Evaluate the numeric clone bound over candidate $\varepsilon_0$ values and select the largest value that meets the target.

| $n$ | $\varepsilon_0$ | Exact, randomized response | Clones, numeric | Clones, closed form |
| --- | --- | --- | --- | --- |
| 1,000 | 4 | 1.6910 | 3.9900 | n/a |
| 10,000 | 2 | 0.0870 | 0.1550 | 0.5009 |
| 100,000 | 2 | 0.0247 | 0.0450 | 0.1862 |
| 100,000 | 4 | 0.0847 | 0.1698 | 0.5346 |

: Central $\varepsilon$ at $\delta = 10^{-6}$ after shuffling the reports of $n$ users. "n/a" marks too few users for the closed form. {#tbl-shuf-compare}

The bound can determine whether a design is feasible. For 100,000 users and a target of 0.2, the closed form permits $\varepsilon_0$ up to 2, while the numeric bound permits 4. Use the tighter numeric bound.

Verify these assumptions before reporting shuffled $\varepsilon$. If any fail—for example, if report order leaks—only the local guarantee $\varepsilon_0$ remains.

- The order is uniformly random and hidden, with no metadata such as arrival time or network address.
- The shuffler does not collude with the analyst.
- Only honest users count in the number of users $n$, because colluders reveal their reports to the analyst.
- The count of users is the number of reports shuffled, so with several batches it is the size of the smallest batch.
- Each user sends one report per batch, or the reports compose like any other releases.

## What goes wrong {#sec-shuf-hidden}

Approximations become unreliable with few users or large $\varepsilon_0$. For 1,000 users at $\varepsilon_0=4$, a normal approximation to the count of 1-reports gives 0.9667, compared with the exact value 1.6910.

In amplification by iteration, later training noise also protects records used earlier. The result applies to one pass of noisy stochastic gradient descent (SGD) whose weights remain in a fixed convex set (Feldman et al. 2018). It requires releasing only the final model and making every step a contraction, meaning it cannot increase the distance between weight vectors. This requires a convex, smooth loss and a sufficiently small step size.

The privacy gain varies by record. For 10,000 records and noise multiplier 1, the middle record gets $\varepsilon=0.0707$ at $\delta=10^{-5}$, while the last record gets 5.2985, the cost of one Gaussian step. Report the worst-case value.

Deep learning typically violates both conditions. Checkpoints, logged models and selecting the best of several checkpoints reveal more than the final model alone. Steps on non-convex losses need not be contractions; Annamalai (2024) constructs a loss where the final model leaks as much as all intermediate models. Claim iteration amplification only for convex models, such as logistic regression, when releasing only final weights.

## References

- Feldman, McMillan & Talwar, "Hiding Among the Clones" (FOCS 2021): the clone bound and how to compute it.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/07-shuffling-iteration.md).
