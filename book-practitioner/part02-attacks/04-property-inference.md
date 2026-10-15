# Property inference

## Why this matters

A hospital trains a readmission model on its 1,000 patients and shares it with a research partner. You have checked that no attack can tell, much better than a coin flip, whether any one patient was in the training set. The partner can still learn whether 45% or 55% of the hospital's patients are over 65, and be right almost every time. That secret is about no patient in particular. It is a **property** of the training set as a whole, a statistic such as the fraction $t$ of records in some subgroup, and **property inference** is the attack that recovers it from a trained model (Ateniese et al. 2015; Ganju et al. 2018).

How can a model hide every record and still give away their mix? Think of it as an average over its $n$ records, plus noise. The luck of which records were sampled moves an average by about $1/\sqrt n$; changing one record moves it by about $1/n$, which is lost in that noise, so membership inference gets harder as $n$ grows. Changing the subgroup's share from $t_0$ to $t_1$ alters about $n|t_1 - t_0|$ records at once, and together they move the model by an amount proportional to $|t_1 - t_0|$ that does not shrink with $n$. Against noise of size $1/\sqrt n$, the signal is about $\sqrt n\,|t_1 - t_0|$: property inference gets *easier* with more data. That comparison of scales explains why record-level defences miss properties, why deleting a column does not help, and how to predict an attack's accuracy before running it.

## The property game and its ceiling {#sec-prop-game}

As for every attack in [Threat models and privacy games](01-threat-models.md), write the attack as a game. A challenger flips a fair bit $b$, draws $n$ records from a population in which the subgroup has share $t_b$, trains a model, and hands the attacker the weights (white-box) or answers to queries (black-box). The attacker knows the two shares, has data from both groups, and can train models the way the victim does. It guesses $b$. Suri & Evans (2022) call this **distribution inference**.

There is no target record, and that changes the score. A membership attacker faces many targets, and a few exposed with high confidence matter, so you report the true positive rate (TPR) at a low false positive rate (FPR). Here each dataset carries one secret, so the natural score is plain **accuracy**, the probability of guessing $b$, against a baseline of $\frac12$. With a fair bit it equals $\frac12(1 + \text{advantage})$, where the **advantage** is $\mathrm{TPR} - \mathrm{FPR}$.

What is the most any attacker can win? Its only evidence is that some views are more likely under one share than the other, so the best it can do is name the share that makes its view more likely, and that attacker's advantage is the total variation (TV) distance between the two distributions of the model. The model, being computed from the data, cannot separate the two worlds better than the data do: anyone holding the data could train it and run the same test. When each record shows its group, the only thing in the data that bears on $t$ is the binomial count of subgroup records. So for every model trained on the data,
$$\text{advantage} \le \mathrm{TV}\big(\mathrm{Bin}(n, t_0), \mathrm{Bin}(n, t_1)\big).$$ {#eq-prop-count-ceiling}
Releasing the count attains this ceiling, because it keeps everything the data say about $t$. A model that never sees the group leaks less, and how much less is something you measure.

For the hospital, the count has mean 450 in one world and 550 in the other, with a standard deviation of about 15.7. The means are more than six standard deviations apart, and @eq-prop-count-ceiling gives 0.9985: an attacker who can read the count off the model is right more than 99.9% of the time. At $n = 100$ the means are 45 and 55, the standard deviation is about 5, the distributions overlap much more, and the ceiling falls to 0.6827; 200,000 simulated games reach 0.6811.

Whether this is a harm depends on whom the property describes. It arises when the property describes the *data owner*: one hospital's patient mix, one company's customers, one lab's machines. If the dataset is a small sample of a public population, its property is public anyway.

## Why a property leaks when no record does {#sec-prop-scaling}

On the released count both attackers can be analysed exactly. The membership attacker asks whether its target, a subgroup member, is in the data. The target adds exactly 1 to the count, where the record in its place would add 1 only a fraction $t$ of the time, so the attacker is looking for one extra unit in a count that already wanders by about $\sqrt n$. The property attacker asks whether the share was $t_0$ or $t_1$; the count's fraction centres on one or the other, with a spread of about $\sqrt{\bar t(1 - \bar t)/n}$ around the midpoint share $\bar t = \frac{t_0 + t_1}2$, and what matters is how many spreads separate the centres. With $\Phi$ the standard normal distribution function, the best advantages are approximately
$$\mathrm{Adv}_{\mathrm{mem}} \approx \frac{1 - t}{\sqrt{2\pi(n - 1)t(1 - t)}}, \qquad \mathrm{Adv}_{\mathrm{prop}} \approx 2\Phi\Big(\frac{\sqrt n\,|t_1 - t_0|}{2\sqrt{\bar t(1 - \bar t)}}\Big) - 1.$$
The first is one unit over the spread of the count, so it falls like $1/\sqrt n$. The second depends on $n$ and the gap only through $\sqrt n\,|t_1 - t_0|$, so doubling the gap has the same effect as multiplying $n$ by 4.

@tbl-prop-count puts numbers on both; @sec-prop-code computes them. At $n = 10{,}000$ the membership attacker is right with probability $0.502$, and the property attacker is never wrong to four decimals. The data, the release and the attacker's knowledge are the same; only the question differs (@fig-prop-count-leak).

| $n$ | Property, exact | Property, formula | Membership, exact | Membership, formula |
| --- | --- | --- | --- | --- |
| 100 | 0.6827 | 0.6827 | 0.0398 | 0.0401 |
| 1,000 | 0.9985 | 0.9984 | 0.0126 | 0.0126 |
| 10,000 | 1.0000 | 1.0000 | 0.0040 | 0.0040 |

: Best advantages from a released count of subgroup records. Property: $t_0 = 0.45$ against $t_1 = 0.55$. Membership: one subgroup record, population share $t = 0.5$. {#tbl-prop-count}

![Left: the released fraction of subgroup records at $n = 1{,}000$ under shares 0.45 and 0.55, which barely overlap, and under share 0.5 with and without one target record, which are indistinguishable. Right: the best property advantage for two gaps and the best membership advantage of one record, against $n$. More data protects the record and exposes the property.](figures/prop_count_leak.svg){#fig-prop-count-leak}

So a property is a group of records, and a protection measured one record at a time says little about it. For the hospital, moving from 45% to 55% changes 100 of its 1,000 patients, and differential privacy covers groups only through its group-privacy property, whose guarantee weakens with the size of the group ([Properties of differential privacy](../part03-dp-foundations/02-properties.md)). To hide the property, the release must be noisy on the scale of $n|t_1 - t_0|$ records, and noise that large also destroys what the release was for: hiding the property means hiding the population. Deleting the group column does not escape this, because whatever correlates with the group still changes what the model fits, as the next section shows. And when outsiders contribute training data, audit with poisoning in scope: a few injected records can make the property far easier to infer (Mahloujifar et al. 2022).

## Shadow models and meta-classifiers {#sec-prop-meta}

For a real model nobody can write down the distribution of the trained weights, so the attacker samples it. It draws many datasets with share $t_0$ and many with $t_1$ from auxiliary data, trains a **shadow model** on each exactly as the victim trains, labels each shadow model's parameters with its share, and trains a classifier on those parameter vectors, the **meta-classifier**, which it then applies to the victim's model. This is the shadow-model recipe of [Membership inference](02-membership-inference.md), with whole models as the data points.

Which meta-classifier, and how accurate? A model fitted by minimizing an average loss really does behave like an average: its parameters scatter approximately as a Gaussian around the **population fit** $\theta(t)$, the model infinitely many records would give, with a covariance $\Sigma_1/n$ that shrinks like $1/n$. Between two Gaussians with the same covariance, the quadratic parts of the log-densities cancel and the log-likelihood ratio is linear, so the best test is a hyperplane between the two clouds: train a **linear discriminant**. Its accuracy depends only on how far apart the centres are in units of the noise. For a small gap they differ by the slope $\theta'$ of the population fit times the gap, which gives
$$\text{accuracy} \approx \Phi\big(\kappa\sqrt n\,|t_1 - t_0|/2\big), \qquad \kappa = \sqrt{\theta'^\top\Sigma_1^{-1}\theta'}.$$ {#eq-prop-kappa}
$\kappa$ is how fast the model moves with the share, in units of its per-record noise. If the property does not change what the model fits, $\theta' = 0$ and nothing leaks to first order, however large $n$ is. If it does, the leak is a side effect of fitting the data well, and no care in the training code removes it. $\kappa$ also compares any model with the count, whose $\kappa = 1/\sqrt{\bar t(1 - \bar t)}$ is 2 at $\bar t = \frac12$: a model with a smaller $\kappa$ leaks the property as well as the count would on a fraction $(\kappa\sqrt{\bar t(1 - \bar t)})^2$ of its records, the unit in which Suri & Evans (2022) report leakage.

Here is a model that never sees the group and leaks anyway. Records have two features and a label; in the majority risk rises with the second feature, in the subgroup it falls. A logistic regression sees features and labels only. At share $\frac12$ the two rules cancel in the second weight, and as the subgroup grows its rule wins: the population fit's second weight moves from 0.442 at share 0.25 to −0.442 at share 0.75. Against that weight's per-record noise this gives $\kappa = 0.647$, against 2 for the count, so the model leaks the property as well as $(0.647/2)^2 = 0.105$ of its records would, about one in ten.

With 150 shadow models per hypothesis and 150 fresh target models, the meta-classifier's accuracy matches @eq-prop-kappa in most cells of @tbl-prop-sweep to within two standard errors, each at most 0.029. Plotted against $\sqrt n\,|t_1 - t_0|$, the four gaps fall on one curve (@fig-prop-meta-accuracy): a gap of 0.05 at $n = 1{,}000$ is about as easy as a gap of 0.1 at $n = 200$, since $\sqrt{1000}\times0.05 = 1.58$ against $\sqrt{200}\times0.1 = 1.41$.

| $n$ | gap 0.05 | gap 0.1 | gap 0.2 | gap 0.4 |
| --- | --- | --- | --- | --- |
| 50 | 0.523 (0.546) | 0.520 (0.590) | 0.653 (0.676) | 0.830 (0.818) |
| 100 | 0.523 (0.564) | 0.670 (0.627) | 0.733 (0.741) | 0.863 (0.901) |
| 200 | 0.503 (0.590) | 0.670 (0.676) | 0.803 (0.819) | 0.957 (0.965) |
| 500 | 0.633 (0.641) | 0.787 (0.765) | 0.910 (0.926) | 0.997 (0.998) |
| 1,000 | 0.713 (0.695) | 0.867 (0.847) | 0.980 (0.979) | 1.000 (1.000) |

: Accuracy of the linear meta-classifier against logistic regressions that never see the group, by dataset size $n$ and gap $\lvert t_1 - t_0\rvert$, with the prediction of @eq-prop-kappa in parentheses. {#tbl-prop-sweep}

![Left: meta-classifier accuracy against $n$ for four gaps between the two shares. Right: the same accuracies against $\sqrt n\,\lvert t_1 - t_0\rvert$, where they collapse onto the predicted curve with $\kappa = 0.647$.](figures/prop_meta_accuracy.svg){#fig-prop-meta-accuracy}

The misses are where $n \le 200$ and the gap is at most 0.1. There the true effect is small, the noise in the shadow models swamps it, and the meta-classifier finds the separating direction badly: the attack is weaker than the best test, not the model safer. So an auditor should report the accuracy predicted from $\kappa$ next to the measured one. Shadow models also need data from both groups, the victim's algorithm and its dataset size, or they sample the wrong distribution.

## Neural networks and permutation invariance {#sec-prop-symmetry}

Swap two hidden units of a network, with their incoming and outgoing weights, and it computes the same function. For tanh units, flipping the signs of a unit's weights in and out also changes nothing, because tanh is odd. Training from a symmetric random start keeps these symmetries, so each weight slot holds a random one of the network's units with a random sign. The property shows in *which* units the network has, not in which slot holds which, and averaging slot by slot washes it out: for tanh networks every raw weight has mean zero under both hypotheses, and a linear meta-classifier on raw weights is at chance (Ganju et al. 2018).

The fix is a representation that ignores order and sign. A **canonical form** flips each unit so its outgoing weight is positive and sorts the units by it; ReLU networks have no sign symmetry, so sorting alone suffices. It is fragile when two units have nearly equal outgoing weights, which swap places from one network to the next. A **set representation** sums a function of each unit's parameter vector $u_j$, and a sum ignores order; $\sum_ju_ju_j^\top$ ignores signs too, and a learned DeepSets network (Zaheer et al. 2017) is the general form. **Probe outputs**, the model's answers on fixed inputs, need no fix: they measure the function, which is what the property changes, and a function does not depend on how it is parameterized. @tbl-prop-representations compares them on tanh networks with 4 hidden units, trained on 200 records with share 0.2 or 0.8, 150 shadow and 150 target networks per side.

| Representation | Features | Accuracy |
| --- | --- | --- |
| Raw parameters | 17 | 0.497 |
| Canonical order (flip so $v_j > 0$, sort by $v_j$) | 17 | 0.807 |
| Unit second moments $\sum_j u_ju_j^\top$ | 11 | 0.877 |
| Outputs on 16 probe inputs | 16 | 1.000 |

: The same linear meta-classifier against the same tanh networks, by representation of the network; $v_j$ is unit $j$'s outgoing weight. {#tbl-prop-representations}

The ranking follows the argument, and it means that a meta-classifier at chance on raw weights says the representation is wrong, not that the model is safe, and deeper networks need an invariant representation at every layer. The table also corrects an intuition about access. White-box access is never weaker for the *best* attack, since whoever holds the weights can compute any query answer, but for a fixed simple attack the representation matters more than the access, and here black-box probes beat raw weights. A model served only through an API is not safe from property inference (Zhang, Tople & Ohrimenko 2021).

## Code {#sec-prop-code}

The ceiling of @eq-prop-count-ceiling and the two advantages of @sec-prop-scaling take a few lines:

```python
from math import erf, exp, lgamma, log, sqrt


def binom_pmf(n: int, t: float) -> list[float]:
    """Pr[Bin(n, t) = k] for k = 0..n, in log space so that large n does not overflow."""
    return [exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(t) + (n - k) * log(1 - t)) for k in range(n + 1)]


def property_ceiling(n: int, t0: float, t1: float) -> float:
    """Best advantage of any property attack when the group is visible: TV(Bin(n, t0), Bin(n, t1))."""
    return 0.5 * sum(abs(p - q) for p, q in zip(binom_pmf(n, t0), binom_pmf(n, t1)))


def property_normal(n: int, t0: float, t1: float) -> float:
    """The normal approximation 2 Phi(sqrt(n) |t1 - t0| / (2 sqrt(tbar (1 - tbar)))) - 1."""
    tbar = (t0 + t1) / 2
    z = sqrt(n) * abs(t1 - t0) / (2 * sqrt(tbar * (1 - tbar)))
    return erf(z / sqrt(2))  # equals 2 Phi(z) - 1


def membership_best(n: int, t: float) -> float:
    """Best advantage for one subgroup record from the same count: (1 - t) max_k Pr[Bin(n - 1, t) = k]."""
    return (1 - t) * max(binom_pmf(n - 1, t))


print(f"{property_ceiling(100, 0.45, 0.55):.4f} {property_normal(100, 0.45, 0.55):.4f}")  # 0.6827 0.6827
print(f"{property_ceiling(1000, 0.45, 0.55):.4f} {property_normal(1000, 0.45, 0.55):.4f}")  # 0.9985 0.9984
print(f"{membership_best(100, 0.5):.4f} {membership_best(10000, 0.5):.4f}")  # 0.0398 0.0040
```

The full edition's script runs the shadow-model attacks and reproduces every number in this chapter:

```bash
uv run python code/part02-attacks/property_inference.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| Score | Accuracy $= \frac12(1 + \mathrm{TPR} - \mathrm{FPR})$ | One secret per dataset; baseline $\frac12$ |
| Ceiling, group visible | $\mathrm{TV}(\mathrm{Bin}(n, t_0), \mathrm{Bin}(n, t_1))$ | Attained by releasing the count |
| Property, released count | $2\Phi\big(\sqrt n\,\lvert t_1 - t_0\rvert/(2\sqrt{\bar t(1 - \bar t)})\big) - 1$ | $\to 1$ as $n$ grows |
| Membership, released count | $\approx (1 - t)/\sqrt{2\pi(n - 1)t(1 - t)}$ | $\to 0$ like $1/\sqrt n$ |
| Meta-classifier accuracy | $\Phi(\kappa\sqrt n\,\lvert t_1 - t_0\rvert/2)$ | Linear discriminant on parameters; count has $\kappa = 1/\sqrt{\bar t(1 - \bar t)}$ |
| Neural networks | Raw weights at chance (tanh) | Canonical order, unit second moments, or probe outputs |

: The results of this chapter. {#tbl-prop-cheat}

**Rules of thumb**

- Property inference gets easier with more data, membership inference harder. What matters is $\sqrt n\,|t_1 - t_0|$.
- A property changes about $n|t_1 - t_0|$ records at once. Defences sized for one record do not cover it.
- Audit with shadow models trained like the victim and a representation that respects the network's symmetries; report accuracy next to the prediction from $\kappa$.

## Further reading

- Ganju, Wang, Yang, Gunter & Borisov, "Property Inference Attacks on Fully Connected Neural Networks using Permutation Invariant Representations" (CCS 2018): the practical attack on neural networks, neuron sorting and the DeepSets meta-classifier.
- Suri & Evans, "Formalizing and Estimating Distribution Inference Risks" (PoPETs 2022): the property game as a test between two training distributions, and leakage reported in records.
- Zhang, Tople & Ohrimenko, "Leakage of Dataset Properties in Multi-Party Machine Learning" (USENIX Security 2021): black-box property inference, including properties of attributes removed from the training data.
- Mahloujifar, Ghosh & Chase, "Property Inference from Poisoning" (IEEE S&P 2022): how a few injected records make a property far easier to infer.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/04-property-inference.md).
