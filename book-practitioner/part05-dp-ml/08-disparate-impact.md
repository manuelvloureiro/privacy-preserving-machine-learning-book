# Disparate impact

## Why this matters

Training the part's running example with differential privacy (DP) at a privacy budget $\varepsilon$ of 4 costs less than one point of test accuracy ([private gradient descent](02-dp-sgd.md)). The private model reaches 0.9548, against 0.9610 without privacy. That number is an average over the whole test set, so it says nothing about who loses the point. Bagdasaryan, Poursaeed & Shmatikov (2019) studied networks trained with the private form of stochastic gradient descent (SGD), called DP-SGD. They found that the lost accuracy falls mostly on the classes and subgroups that are under-represented in the training data, and called this the **disparate impact** of DP on model accuracy.

So a model that is "less than a point worse" on average can be ten points worse for one group of people. Those groups are often the ones a privacy guarantee is meant to protect, such as small populations, people with rare conditions and records that are unusual in other ways.

## Evaluating a private model per group {#sec-fair-measure}

Before you look at any results, decide which groups matter, and give each group its own test records. Then train a non-private baseline with the same data and recipe. A gap between groups may already exist without privacy, so the quantity that DP is responsible for compares each group with itself.

Write $\mathrm{acc}_g$ for a model's accuracy on the test records of group $g$. The **privacy cost** $\pi_g$ is the accuracy the group loses when the same training is made private,
$$\pi_g = \mathrm{acc}_g(\text{non-private}) - \mathbb E\big[\mathrm{acc}_g(\text{private})\big].$$ {#eq-fair-cost}
The average runs over training runs, because private training is random. DP has disparate impact when different groups have different privacy costs, and the difference between two groups' costs is the **privacy cost gap**.

Every accuracy needs an interval. Each test record is either classified correctly or not, so a group's accuracy is a binomial proportion, and the exact Clopper–Pearson interval covers it ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). In SciPy, `scipy.stats.binomtest(correct, m).proportion_ci(method="exact")` returns it, and in statsmodels `proportion_confint(correct, m, alpha=0.05, method="beta")` does. The gap between two groups needs its own interval. Write $\hat p_g$ for a group's measured accuracy and $m_g$ for its number of test records. For a first group $a$ and a second group $b$, measured on disjoint test records with large counts, an approximate 95% interval for the gap is
$$(\hat p_a - \hat p_b) \pm 1.96\sqrt{\frac{\hat p_a(1 - \hat p_a)}{m_a} + \frac{\hat p_b(1 - \hat p_b)}{m_b}}.$$ {#eq-fair-gap}

These intervals are wider than most people expect. Suppose a 2,000-record test split holds a 5% minority, so only 100 of its records come from the minority. On one model trained at $\varepsilon = 4$, 88 of them were correct, which gives an interval 13.6 points wide. Against the majority's 1,812 out of 1,900, the measured gap of 0.074 has the interval $[0.010, 0.138]$ by @eq-fair-gap. So the gap is real, but it could be anywhere from one point to fourteen. The width falls only like one over the square root of the number of records. So a 5-point gap shows only with a few hundred test records from the group, and a 1-point gap only with thousands.

A larger test set does nothing about the second source of variation, which is training itself. The next run gives another model, and a small group's accuracy varies far more between runs. In the running example at $\varepsilon = 4$, the minority's run-to-run standard deviation is almost six times the majority's. So a comparison of two recipes on one run each can easily reverse for the small group, and you should train three or more times.

`fairlearn.metrics.MetricFrame`, given a `sensitive_features` column, computes any metric per group in its `by_group` table, without intervals. The evaluation can also leak. Per-group accuracies on private test data are statistics of that data, so publishing them needs its own privacy analysis ([Choosing ε](09-choosing-epsilon.md)).

## What to expect {#sec-fair-observed}

To show the effect, the full edition gives the running example a minority. Of its 10,000 training records, 500 come from a different region of input space. Their inputs are longer, and their labels follow a rule that the majority's records do not teach. Without privacy the model learns both groups well, with a gap of only 1.2 points, so any large gap after private training is caused by privacy (@tbl-fair-eps).

| Training | Majority accuracy | Minority accuracy | Privacy cost: majority | minority |
| --- | --- | --- | --- | --- |
| Non-private SGD | $0.9529 \pm 0.0018$ | $0.9411 \pm 0.0052$ | — | — |
| Clipping only, $\sigma = 0$ | $0.9581 \pm 0.0015$ | $0.9007 \pm 0.0048$ | $-0.0051$ | 0.0404 |
| $\varepsilon = 8$, $\sigma = 0.917$ | $0.9550 \pm 0.0006$ | $0.8553 \pm 0.0110$ | $-0.0020$ | 0.0858 |
| $\varepsilon = 4$, $\sigma = 1.36$ | $0.9470 \pm 0.0037$ | $0.8477 \pm 0.0211$ | 0.0059 | 0.0934 |
| $\varepsilon = 1$, $\sigma = 4.13$ | $0.8822 \pm 0.0032$ | $0.6133 \pm 0.0173$ | 0.0707 | 0.3278 |

: Test accuracy of each group (mean ± standard deviation over 3 runs) of the running example with a 5% minority, trained with SGD, with clipping alone, and with DP-SGD at three budgets ($\delta = 10^{-5}$, noise multiplier $\sigma$). The privacy cost is @eq-fair-cost. {#tbl-fair-eps}

The minority pays almost all of the cost. At $\varepsilon = 4$ the majority loses 0.59 points and the minority 9.34, which is 15.7 times as much. Even at $\varepsilon = 8$, where the majority loses nothing, the minority loses 8.58 points. An overall accuracy hides this, because a test set with 95% majority records reports a loss of about one point at $\varepsilon = 4$.

The cost depends steeply on the group's share of the data. At $\varepsilon = 4$, a minority with 1% of the training records loses 34.88 points and ends at chance, while one with 20% loses only 2.80.

## Why small groups pay: clipping and noise {#sec-fair-clipping}

Both causes, clipping and noise, are blind to groups. The clipping row of @tbl-fair-eps adds no noise, so it gives no privacy at all. Yet the minority already loses 4.04 points there while the majority gains 0.51, so clipping alone is disparate.

Clipping caps every record's pull on the update at the clipping norm $C$, so it cuts back hardest the groups whose gradients are long. Gradients are long where the model fits a group badly, as for rare patterns, and where the group's inputs are large, as for outlying records. A small group that has not been learned yet has both, so training gives it less weight than its share of the data.

The noise is the same for every group, while a group's signal grows with its size. So a small group's signal is the weakest against the noise, and its privacy cost grows roughly like one over its **share** of the training records. The two causes also compound, since a group whose gradients clipping halves pays like a group half its size.

What you need from this is a diagnosis. Log the median per-example gradient norm and the fraction of records clipped, per group, during training. In Opacus, each parameter's `p.grad_sample` holds the per-example gradients after the backward pass, so their norms can be grouped by a label you keep beside the batch.

## Mitigations and what they cost {#sec-fair-mitigation}

Every fix moves the cost around rather than removing it, as @tbl-fair-mitigation-guide shows.

| Fix | What happens | Use it? |
| --- | --- | --- |
| Upweight the group's loss | Clipping scales the longer gradients back to $C$, so clipped records gain nothing | No |
| Smaller clipping norm | More records clipped, so the group's effective share falls further | No |
| Larger clipping norm, with the step size times $C$ fixed | Less clipping bias, but the noise $\sigma C$ grows relative to short gradients | Rarely; on the running example neither direction helped |
| Clipping norm set from the largest observed gradient | Depends on the data, so it breaks the privacy proof | Only if chosen privately and charged |
| Oversample the group | Its records' $\varepsilon$ is set by their higher sampling rate, so $\sigma$ must rise | Only with that accounting |
| Global scaling with an adaptive bound (Esipova et al. 2023) | The same as a larger clipping norm with a smaller step; keeps the update's direction | Worth a try |
| Group-aware clipping or fairness constraints (research) | Equalizes the groups' clipping or loss | Only with lawful group labels, spending budget on group statistics |
| More data from the group, a public pretrained model, or a larger $\varepsilon$ | Raises the group's signal or lowers the noise | The fixes that work |

: Mitigations for disparate impact, and when to use them. {#tbl-fair-mitigation-guide}

Oversampling raises a group's pull but costs that group's privacy, because a record's guarantee depends on its own chance of being sampled ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md)). The rule must be fixed in advance, with the accountant at the highest rate. Sampling the 5% minority four times as often, at $\varepsilon = 4$ for its records, raises the noise multiplier from 1.36 to 4.71. Without the higher noise, the minority would silently get $\varepsilon = 20.96$, the weakest privacy for the group already worst served. With it, the gap narrows from 9.93 points to 3.99, but by **levelling down**. Both groups lose accuracy, the majority 8.01 points.

Some of the cost cannot be avoided by any algorithm, because learning a small group's own rule is private estimation on its few records ([Lower bounds](../part03-dp-foundations/10-lower-bounds.md)). So the reliable fixes are more data from the group, or a starting point that already serves it, such as a public pretrained model ([Public data and private fine-tuning](06-public-data.md)).

## Code {#sec-fair-code}

This function computes the interval for the accuracy gap, @eq-fair-gap, and prints the one quoted above.

```python
from math import sqrt


def gap_interval(pa: float, ma: int, pb: float, mb: int) -> tuple[float, float]:
    """Approximate 95% interval for the accuracy gap pa - pb, measured on disjoint test sets."""
    se = sqrt(pa * (1 - pa) / ma + pb * (1 - pb) / mb)
    return pa - pb - 1.96 * se, pa - pb + 1.96 * se


print("%.3f %.3f" % gap_interval(0.954, 1_900, 0.880, 100))  # 0.010 0.138
```

The full edition's script trains the running example with a minority and computes every number in this chapter:

```bash
uv run python code/part05-dp-ml/disparate_impact.py
```

## Cheat sheet

| Question | Answer | Conditions |
| --- | --- | --- |
| Does DP hurt this group? | Its privacy cost, @eq-fair-cost, against the non-private baseline | Same data and recipe, averaged over runs |
| Is the gap real? | Clopper–Pearson per group, @eq-fair-gap for the gap | Disjoint test records, large counts |
| How many test records? | A few hundred per group for a 5-point gap, thousands for 1 point | Width falls like $1/\sqrt{m_g}$ |
| How big will the cost be? | Roughly one over the group's share, more with long gradients | Grows as $\varepsilon$ falls |
| Which fix? | More data or a public model, then global scaling | See @tbl-fair-mitigation-guide |

: The decisions of this chapter. {#tbl-fair-cheat}

**Rules of thumb**

- Report accuracy and privacy cost per group, with intervals and the spread over runs, next to the non-private model's.
- Never report overall accuracy alone, and never report the gap alone, because levelling down closes it too.
- Never compare two recipes on one seed each, because a small group's accuracy varies most between runs.
- Oversample a group only with the accountant at its higher sampling rate.
- Treat published per-group metrics on private test data as a release, and remember that the group labels themselves are sensitive.

## Further reading

- Bagdasaryan, Poursaeed & Shmatikov, "Differential Privacy Has Disparate Impact on Model Accuracy" (NeurIPS 2019): the observation.
- Esipova, Ghomi, Luo & Cresswell, "Disparate Impact in Differential Privacy from Gradient Misalignment" (ICLR 2023): clipping's direction error, and global scaling with an adaptive bound.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/08-disparate-impact.md).
