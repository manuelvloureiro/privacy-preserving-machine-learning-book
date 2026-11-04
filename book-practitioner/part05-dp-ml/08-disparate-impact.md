# Disparate impact

## Why this matters

Average test accuracy can hide groups that bear more of privacy's accuracy cost. Differential privacy (DP) applied to stochastic gradient descent (SGD), called DP-SGD ([DP-SGD](02-dp-sgd.md)), can disproportionately reduce accuracy for under-represented groups. Bagdasaryan, Poursaeed & Shmatikov (2019) called this the **disparate impact** of DP.

A one-point average loss can conceal a ten-point loss for a small group. Before release, compare each group's private-model accuracy with a non-private baseline and report confidence intervals.

## Evaluating a private model per group {#sec-fair-measure}

Specify relevant groups before examining results to avoid selecting groups based on outcomes. Train a non-private baseline with the same data and procedure; group differences may exist without privacy. Measure DP's cost as each group's accuracy change under private training.

Let $\mathrm{acc}_g$ be model accuracy on test records from group $g$. Since private training is random, average accuracy over multiple runs, denoted $\mathbb E$. The group's **privacy cost** $\pi_g$ is its accuracy loss under private rather than non-private training with the same data and hyperparameters:
$$\pi_g = \mathrm{acc}_g(\text{non-private}) - \mathbb E\big[\mathrm{acc}_g(\text{private})\big].$$ {#eq-fair-cost}
DP has disparate impact when groups have different privacy costs.

Report an interval for each accuracy. Correct classifications are binomial outcomes, so the exact Clopper–Pearson interval covers the true group accuracy in at least 95% of samples ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)).

Compute a separate interval for differences between groups. Let $\hat p_g$ be measured accuracy for group $g$, and let $m_g$ be the number of its test records. For disjoint test sets, the variances add. This gives the following approximate 95% interval when counts are sufficiently large:
$$(\hat p_a - \hat p_b) \pm 1.96\sqrt{\frac{\hat p_a(1 - \hat p_a)}{m_a} + \frac{\hat p_b(1 - \hat p_b)}{m_b}}.$$ {#eq-fair-gap}

Small test groups produce wide intervals. For example, suppose 88 of 100 minority-group records are classified correctly. The majority group has 1,812 correct predictions among 1,900 records, or accuracy 0.954; the measured gap is 0.074. SciPy gives the minority interval, and @eq-fair-gap gives the gap interval.

```python
from math import sqrt

from scipy.stats import binomtest

minority = binomtest(88, 100).proportion_ci(confidence_level=0.95, method="exact")
print(f"{minority.low:.3f} {minority.high:.3f}")  # 0.800 0.936

gap, se = 0.954 - 0.880, sqrt(0.954 * 0.046 / 1_900 + 0.880 * 0.120 / 100)
print(f"{gap - 1.96 * se:.3f} {gap + 1.96 * se:.3f}")  # 0.010 0.138
```

The gap's interval excludes zero, but its value could range from one to fourteen points. Distinguishing a five-point gap from zero requires hundreds of group test records; a one-point gap requires thousands.

Privacy cost needs a paired interval because both models use the same test records, so their accuracies are dependent and @eq-fair-gap does not apply. For group $g$, let $u$ count records only the non-private model classifies correctly. Let $v$ count records only the private model gets right, and let $m_g$ be the test-record count. One run's cost is $(u-v)/m_g$. For large counts, an approximate 95% interval is
$$\frac{u - v}{m_g} \pm \frac{1.96}{m_g}\sqrt{u + v - \frac{(u - v)^2}{m_g}}.$$ {#eq-fair-paired}

Training introduces additional variation that a larger test set cannot remove. Small-group accuracy varies more across training runs, so a one-run comparison can reverse. Train each procedure with multiple seeds and report mean cost and its spread.

Per-group accuracy on a private test set is itself a statistic of private data. To publish it, add noise to each group's correct-prediction count and allocate a separate privacy budget ([Release statistics with a library](../part03-dp-foundations/12-release-statistics.md)).

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

At $\varepsilon = 4$, no group loses more than 1.2 points. At $\varepsilon = 1$, the White group loses 0.37 points. The Asian-Pac-Islander group loses 3.01 points, and the Other group loses 2.13. Black and Amer-Indian-Eskimo people lose nothing, although the latter is one of the smallest groups. The share of high incomes does not explain the pattern: about 12% of the Black, Amer-Indian-Eskimo and Other groups earn above \$50K. Only the Other group has a measured loss. No simple group feature predicts privacy cost, so check every small group regardless of overall accuracy.

These results are suggestive, not conclusive. At accuracy 0.85, the exact interval for 532 test rows is $[0.816, 0.879]$, or about three points on each side. With 141 rows, the interval is about six points on each side. Three training runs do not remove this test-sampling uncertainty. A real check needs the paired interval of @eq-fair-paired and more runs.

## Why small groups pay: clipping and noise {#sec-fair-clipping}

DP-SGD can lose accuracy through clipping and noise, and both steps treat every record alike. The clipping-only column of @tbl-fair-eps adds no noise and gives no privacy. Clipping alone costs the Asian-Pac-Islander group 0.57 points and the Other group 1.18 points. The White group gains 0.08 points, so clipping can affect groups unequally.

Clipping caps every record's pull on the update at the clipping norm $C$, so it cuts back the groups with long gradients hardest. A group's gradients are long where the model fits it badly, as it does a rare pattern the rest of the data does not teach. So training gives such a group less weight than its share of the records.

The noise is the same for every group, while a group's signal grows with its size. A small group therefore has a weaker signal relative to the noise. When that group needs a rule of its own, its privacy cost can grow inversely with its share of the training records.

On Adult, reducing the Asian-Pac-Islander share from 3.1% to 1% leaves its cost at $\varepsilon = 1$ within the spread of three runs. The measured cost is 3.13 points. Measure the cost rather than predicting it from group size. Clipping and noise compound because noise acts on the update after clipping has reduced it. If clipping halves a group's gradients, its signal is like that of a group half its size.

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

Oversampling costs the group's privacy because a record's guarantee depends on its chance of being sampled ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md)). Run the accountant at the oversampled rate. On Adult, sampling the Asian-Pac-Islander rows four times as often at $\varepsilon = 1$ raises the noise multiplier from 2.383 to 8.828.

Every group then loses accuracy, with the largest loss in the oversampled group. Its accuracy falls from 0.8145 to 0.7657; the White group's falls from 0.8310 to 0.7860. These comparisons use the same DP-SGD at the ordinary sampling rate. This is **levelling down** with no gain for anyone.

No algorithm avoids some privacy cost. A model that learns a small group's rule must estimate it privately from that group's records. With few records, private estimation has an error floor that no mechanism can beat ([Lower bounds](../part03-dp-foundations/10-lower-bounds.md)).

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
