# Disparate impact

## Why this matters

A private model's test accuracy is an average over everyone, so it cannot show which people lose the accuracy that privacy costs. Bagdasaryan, Poursaeed & Shmatikov (2019) trained networks with differential privacy (DP), using the private form of stochastic gradient descent (SGD) called [DP-SGD](02-dp-sgd.md). They found that the lost accuracy fell mostly on under-represented groups, and called this the **disparate impact** of DP.

So a model that is a point worse on average can be ten points worse for one small group, often one that privacy should protect. Before you ship a private model, measure its cost for each group against a non-private baseline, with a confidence interval on every number.

## Evaluating a private model per group {#sec-fair-measure}

Decide which groups matter before you look at any results, so that the results cannot steer the choice. Then train a non-private baseline with the same data and recipe, because a gap between groups may exist without privacy. DP is responsible only for how far each group's own accuracy drops when training becomes private.

Write $\mathrm{acc}_g$ for a model's accuracy on the test records of group $g$. Private training is random, so its accuracy is averaged over several training runs, written $\mathbb E$. The **privacy cost** $\pi_g$ is the accuracy the group loses when the same data and hyperparameters are trained privately,
$$\pi_g = \mathrm{acc}_g(\text{non-private}) - \mathbb E\big[\mathrm{acc}_g(\text{private})\big].$$ {#eq-fair-cost}
DP has disparate impact when groups have different privacy costs.

Every accuracy needs an interval. Each test record is classified correctly or not, so a group's accuracy is a binomial proportion, and its exact Clopper–Pearson interval contains the true accuracy in at least 95% of test samples ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)).

The gap between two groups needs its own interval. Write $\hat p_g$ for group $g$'s measured accuracy and $m_g$ for its number of test records. For two groups $a$ and $b$ measured on disjoint test records, the variances of the two estimates add. So for large counts an approximate 95% interval for the gap is
$$(\hat p_a - \hat p_b) \pm 1.96\sqrt{\frac{\hat p_a(1 - \hat p_a)}{m_a} + \frac{\hat p_b(1 - \hat p_b)}{m_b}}.$$ {#eq-fair-gap}

These intervals are wider than most people expect. Suppose a test split holds only 100 records of a minority, and 88 of them are classified correctly. The majority gets 1,812 of its 1,900 records right, an accuracy of 0.954, so the measured gap is 0.074. SciPy gives the minority's interval, and @eq-fair-gap gives the gap's.

```python
from math import sqrt

from scipy.stats import binomtest

minority = binomtest(88, 100).proportion_ci(confidence_level=0.95, method="exact")
print(f"{minority.low:.3f} {minority.high:.3f}")  # 0.800 0.936

gap, se = 0.954 - 0.880, sqrt(0.954 * 0.046 / 1_900 + 0.880 * 0.120 / 100)
print(f"{gap - 1.96 * se:.3f} {gap + 1.96 * se:.3f}")  # 0.010 0.138
```

So the gap is real, since its interval excludes zero, but it could be anywhere from one point to fourteen. An interval separates a 5-point gap from zero only with a few hundred test records from the group, and a 1-point gap only with thousands.

The privacy cost needs a different interval. Both models are scored on the same test records, so their accuracies are not independent and @eq-fair-gap does not apply. Pair them instead. Among group $g$'s $m_g$ test records, write $u$ for the number only the non-private model gets right, and $v$ for those only the private one gets right. One run's cost is then $(u - v)/m_g$, and for large counts an approximate 95% interval is
$$\frac{u - v}{m_g} \pm \frac{1.96}{m_g}\sqrt{u + v - \frac{(u - v)^2}{m_g}}.$$ {#eq-fair-paired}

Training adds a second source of variation, which no test set removes. A small group's accuracy varies far more between training runs than a large group's, so a one-run comparison of two recipes can reverse. So train each recipe with several seeds, and report the mean cost and its spread.

The evaluation can also leak, because per-group accuracies on private test data are statistics of that data. To publish them, add noise to each group's count of correct predictions and charge it to a budget of its own ([Release statistics with a library](../part03-dp-foundations/12-release-statistics.md)).

## Disparate impact, measured {#sec-fair-observed}

The script [practitioner_dp_sgd.py](../../code/part05-dp-ml/practitioner_dp_sgd.py) of [Train a model with DP-SGD](13-train-dp-sgd.md#sec-trainwf-utility) trains a small network on the Adult census data to predict whether a person's income exceeds \$50K. It trains once without privacy and once with DP-SGD at a privacy level $\varepsilon$ of 4 and a slack $\delta$ of $10^{-5}$. Then it scores both on the test rows of each `sex`, with exact intervals from SciPy, which gives @tbl-fair-adult for seed 0.

| Group | Test records | Non-private accuracy | Accuracy at $\varepsilon = 4$ | Privacy cost |
| --- | --- | --- | --- | --- |
| Female | 5,613 | 0.924 [0.917, 0.931] | 0.922 [0.915, 0.929] | +0.002 |
| Male | 11,229 | 0.808 [0.800, 0.815] | 0.807 [0.800, 0.815] | +0.001 |

: Accuracy on Adult by sex, with exact 95% intervals, for one training run of each model. {#tbl-fair-adult}

Here DP has no measurable disparate impact. Each group loses at most 0.002 in one private run, well inside its intervals. The large gap between the groups is there without privacy too, so DP is not responsible for it. Both groups are large, and the whole model loses almost nothing at this budget.

Small groups tell a different story at a smaller budget. Adult's `race` column has three groups with under 1,000 of the 30,000 training rows. The script [practitioner_adult_studies.py](../../code/part05-dp-ml/practitioner_adult_studies.py) trains the same model three times each without privacy, with clipping only, and at $\varepsilon = 4$ and 1, and scores every group (@tbl-fair-eps).

| Group | Training rows | Test rows | Non-private | Clipping only | $\varepsilon = 4$ | $\varepsilon = 1$ | Privacy cost at $\varepsilon = 1$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White | 25,670 | 14,352 | 0.8368 | 0.8376 | 0.8363 | 0.8331 | 0.0037 |
| Black | 2,876 | 1,650 | 0.9042 | 0.9046 | 0.9079 | 0.9038 | 0.0004 |
| Asian-Pac-Islander | 926 | 532 | 0.8346 | 0.8289 | 0.8315 | 0.8045 | 0.0301 |
| Amer-Indian-Eskimo | 283 | 167 | 0.9281 | 0.9361 | 0.9361 | 0.9281 | 0.0000 |
| Other | 245 | 141 | 0.9149 | 0.9031 | 0.9031 | 0.8936 | 0.0213 |

: Test accuracy on Adult by race, averaged over 3 runs ($\delta = 10^{-5}$, noise multiplier 0.973 at $\varepsilon = 4$ and 2.383 at $\varepsilon = 1$). Costs are fractions of accuracy. {#tbl-fair-eps}

At $\varepsilon = 4$ no group loses more than 1.2 points. At $\varepsilon = 1$ the White group loses 0.37 points, while the Asian-Pac-Islander group loses 3.01 and the Other group 2.13. Black and Amer-Indian-Eskimo people lose nothing, although the second is one of the smallest groups. Nor does the share of high incomes explain it: about 12% of the Black, Amer-Indian-Eskimo and Other groups earn above \$50K, yet only the Other group pays. So no simple feature of a group predicts its cost, and every small group must be checked, whatever the overall accuracy says.

These numbers are only suggestive. At an accuracy of 0.85 the exact interval for 532 test rows is $[0.816, 0.879]$, three points on each side, and for 141 rows it is six. Three training runs do not remove that spread either, so a real check uses the paired interval of @eq-fair-paired and more runs.

## Why small groups pay: clipping and noise {#sec-fair-clipping}

DP-SGD loses accuracy through two steps, clipping and noise, and both treat every record alike. Yet the clipping column of @tbl-fair-eps, which adds no noise and so gives no privacy, already costs the Asian-Pac-Islander group 0.57 points and the Other group 1.18, while the White group gains 0.08. So clipping by itself can treat the groups unequally.

Clipping caps every record's pull on the update at the clipping norm $C$, so it cuts back the groups with long gradients hardest. A group's gradients are long where the model fits it badly, as it does a rare pattern the rest of the data does not teach. So training gives such a group less weight than its share of the records.

The noise is the same for every group, while a group's signal grows with its size. So a small group's signal is the weakest against the noise, and when the group needs a rule of its own, its privacy cost can grow like one over its share of the training records. On Adult, cutting the Asian-Pac-Islander group from 3.1% to 1% of the training rows leaves its cost at $\varepsilon = 1$ at 3.13 points, within the spread of three runs, so measure the cost instead of predicting it from the share. The two causes compound, because the noise acts on the weight that clipping has already shrunk. A group whose gradients clipping halves pays like a group half its size.

To see which cause hurts your model, measure per-example gradient norms by group. Opacus's `GradSampleModule`, the wrapper DP-SGD training already uses, keeps one gradient per row in each parameter's `grad_sample`, so one backward pass gives the norms.

```{.python excerpt="dpsgd-grad-norms"}
wrapped = GradSampleModule(model)  # adds hooks that keep one gradient per row
wrapped.train()
logits: torch.Tensor = wrapped(split.x)
nn.CrossEntropyLoss()(logits, split.y).backward()  # fills p.grad_sample for every weight p
norms = stacked_norms(wrapped)
```

Group the norms by each row's group label, kept beside the batch but not fed to the model. Log each group's median norm and fraction clipped as training goes on, because the groups can separate late. At initialization on Adult, the median norm is 1.444, so $C = 1$ clips every row, whatever its group.

## Mitigations and what they cost {#sec-fair-mitigation}

At a fixed budget, a change to the training recipe only moves the cost between the groups or between clipping and noise (@tbl-fair-mitigation-guide). Only the last row, which changes the data, the starting model or the budget, reduces it.

| Fix | What happens | Use it? |
| --- | --- | --- |
| Upweight the group's loss, or use a smaller clipping norm | Clipping cuts an upweighted gradient back to $C$, and a smaller norm clips more of the group | No |
| Larger clipping norm, with learning rate times $C$ held fixed | Less clipping, but more noise relative to short gradients | Rarely, since it trades one cause for the other |
| Clipping norm set to the largest gradient seen in the data | Breaks the privacy proof, which assumes $C$ is fixed in advance | Only if chosen privately and charged |
| Oversample the group | Its records' $\varepsilon$ is set by their higher sampling rate, so $\sigma$ must rise | Only with that accounting |
| Global scaling with an adaptive bound (Esipova et al. 2023), one shared scale for every gradient | Keeps the update's direction and needs no group labels | Worth a try |
| More data from the group, a public pretrained model ([Public data and private fine-tuning](06-public-data.md)), or a larger $\varepsilon$ | Raises the group's signal or lowers the noise | The fixes that work |

: Mitigations for disparate impact, and when to use them. {#tbl-fair-mitigation-guide}

Oversampling costs the group's privacy, because a record's guarantee depends on its own chance of being sampled ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md)). So run the accountant at the oversampled rate. On Adult, sampling the Asian-Pac-Islander rows four times as often at $\varepsilon = 1$ raises the noise multiplier from 2.383 to 8.828. Every group then loses, the oversampled group most: it falls from 0.8145 to 0.7657, and the White group from 0.8310 to 0.7860, against the same DP-SGD at the ordinary rate. That is **levelling down** with no gain for anyone.

No algorithm avoids some of the cost. Learning a small group's own rule means estimating it privately from that group's few records, and private estimation from few records has an error floor that no mechanism beats ([Lower bounds](../part03-dp-foundations/10-lower-bounds.md)).

## Cheat sheet

| Question | Answer | Conditions |
| --- | --- | --- |
| Does DP hurt this group? | Its privacy cost, @eq-fair-cost, against the non-private baseline | Same data and recipe, averaged over runs |
| How big will the cost be? | Negligible for large groups at moderate $\varepsilon$, roughly one over the group's share for small ones | Grows as $\varepsilon$ falls |

: The decisions of this chapter. {#tbl-fair-cheat}

**Rules of thumb**

- Report accuracy and privacy cost per group, with intervals, next to the non-private model's.
- Never report the gap alone, because levelling down closes it too.

## References

- Bagdasaryan, Poursaeed & Shmatikov, "Differential Privacy Has Disparate Impact on Model Accuracy" (NeurIPS 2019): the observation.
- Esipova, Ghomi, Luo & Cresswell, "Disparate Impact in Differential Privacy from Gradient Misalignment" (ICLR 2023): global scaling.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/08-disparate-impact.md).
