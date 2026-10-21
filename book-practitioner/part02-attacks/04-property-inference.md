# Property inference

## Why this matters

A hospital trains a readmission model on its 1,000 patients and shares it with a research partner. Suppose you have checked that no attack on the model can tell, much better than a coin flip, whether any one patient was in the training set. The partner can still learn whether 45% or 55% of the hospital's patients are over 65, and be right almost every time.

None of these secrets is about a single record. Each is a **property** of the training set as a whole, such as the fraction of records in some subgroup. **Property inference** is the attack that recovers it from a trained model. The one fact to keep from this chapter is that a property changes many records at once, so it moves the model far more than any single record does. That is why defences sized for one record do not cover it, and why the attack gets *easier*, not harder, as the dataset grows.

## The property game and its ceiling {#sec-prop-game}

[Threat models and privacy games](01-threat-models.md) writes every attack as a game. In the property game, a challenger flips a fair bit and trains a model on records drawn with one of two subgroup shares, $t_0$ or $t_1$, depending on the bit. The attacker knows both shares, has data from both groups and can train models the way the victim does. It sees the weights or only the model's answers, and guesses the bit. Each dataset carries a single secret, so the natural score is the **accuracy**, the chance that the guess is right. It is tied to the true positive rate (TPR) and the false positive rate (FPR) by
$$\text{accuracy} = \tfrac12(1 + \mathrm{TPR} - \mathrm{FPR}),$$
so a coin flip scores one half.

First ask whether the property is a secret at all. Learning that smokers get more lung cancer is the purpose of a medical study, not a breach. A property is a harm to reveal only when the property describes the *data owner* rather than the world, such as one hospital's patient mix or one lab's machine configuration. When the dataset is a small sample of a public population, its property is public anyway.

There is a ceiling on what any attacker can learn. When each record shows its group, no model can leak more than the count of subgroup records, so the best advantage is the total variation (TV) distance between the two distributions of that count ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). Write $n$ for the number of records. For the hospital's 1,000 patients the ceiling is 0.9985, so it is no comfort, only a scale against which to read a measured attack. An attacker who can add a few records of its own to the training set can make a property far easier to infer (Mahloujifar et al. 2022).

## Why a property leaks when no record does {#sec-prop-scaling}

The simplest release, the count of subgroup records, shows why. One attacker asks whether one subgroup record is in the data, while another asks whether the share was $t_0$ or $t_1$. Write $t$ for the population share in the first question. Then the best membership advantage is about
$$\mathrm{Adv}_{\mathrm{mem}} \approx \frac{1 - t}{\sqrt{2\pi(n - 1)t(1 - t)}},$$ {#eq-prop-membership}
which falls like $1/\sqrt n$. Write $\bar t$ for the midpoint of the two shares and $\Phi$ for the standard normal distribution function. The best property advantage is about
$$\mathrm{Adv}_{\mathrm{prop}} \approx 2\Phi\Big(\frac{\sqrt n\,|t_1 - t_0|}{2\sqrt{\bar t(1 - \bar t)}}\Big) - 1,$$ {#eq-prop-property}
which rises towards 1.

| $n$ | Property | Membership |
| --- | --- | --- |
| 100 | 0.6827 | 0.0398 |
| 1,000 | 0.9985 | 0.0126 |
| 10,000 | 1.0000 | 0.0040 |

: Exact best advantages from a released count of subgroup records, which the formulas match closely. Property: $t_0 = 0.45$ against $t_1 = 0.55$. Membership: one subgroup record, population share $t = 0.5$. {#tbl-prop-count}

In @tbl-prop-count, at 10,000 records, the attacker who asks about one patient is right with probability 0.502, while the one who asks about the share is never wrong to four decimals.

Two practical rules follow. Size and gap enter only through $\sqrt n\,|t_1 - t_0|$, so doubling the gap has the same effect as four times as much data. And moving from share $t_0$ to $t_1$ changes about $n|t_1 - t_0|$ records, which for the hospital is 100 of its 1,000 patients. Differential privacy (DP) protects one record at a time. Its group-privacy guarantee for a group of records multiplies the privacy level $\varepsilon$ by the size of the group, here 100 ([Properties of differential privacy](../part03-dp-foundations/02-properties.md)). So a model trained with DP at a useful $\varepsilon$ still reveals the property, and a small membership risk says nothing about it either.

## Running the attack: shadow models and a meta-classifier {#sec-prop-meta}

Nobody can write down the distribution of a real model's weights, so the attacker samples it instead (Ateniese et al. 2015). The recipe is the shadow-model idea of [Membership inference](02-membership-inference.md), with whole models as the data points.

1. From auxiliary data, draw many datasets of the victim's size with share $t_0$ and as many with share $t_1$.
2. Train a **shadow model** on each one, with the victim's architecture, hyperparameters and training code.
3. Turn each shadow model into a feature vector, either its parameters or its outputs on fixed probe inputs.
4. Train a classifier, the **meta-classifier**, to tell the two shares apart from those vectors, and measure its accuracy on fresh shadow models.
5. Apply it to the victim's model.

A linear meta-classifier is a sensible default. Models fitted by minimizing an average loss on many records have roughly normal parameters, and for those the best possible test is linear. In scikit-learn, `LogisticRegression` from `sklearn.linear_model` serves well, and `LinearDiscriminantAnalysis` from `sklearn.discriminant_analysis` is the linear test itself. Fit either on the shadow features with the share as the label, and read its accuracy with `score` on held-out shadow models. No mainstream privacy library ships a standard property-inference attack, so this short pipeline is the usual tool.

```text
meta = LogisticRegression(max_iter=1000).fit(shadow_features, shadow_shares)
print(meta.score(heldout_features, heldout_shares))   # attack accuracy
```

The full edition also predicts how well the best attack can do. Write $\theta'$ for the slope of the fitted parameters, how fast they move as the share changes, and $\Sigma_1$ for their noise, the covariance that one record would give. Their ratio is a constant $\kappa$, and for a small gap the best accuracy is about
$$\text{accuracy} \approx \Phi\big(\kappa\sqrt n\,|t_1 - t_0|/2\big), \qquad \kappa = \sqrt{\theta'^\top\Sigma_1^{-1}\theta'}.$$ {#eq-prop-kappa}
The released count has $\kappa = 2$ at a midpoint of one half, so a model leaks as much as the count would on a fraction $(\kappa/2)^2$ of the records. If the property does not change what the model fits, nothing leaks to first order. If it does, the leak is a side effect of fitting the data well, and no care in the training code removes it.

For example, the full edition's logistic regression never sees the group but has $\kappa = 0.647$, so it leaks the share about as well as one record in ten would.

What goes wrong is mostly on the attacker's side. With few records and a small gap, noise in the shadow models hides the shift, so the measured attack falls short of the best one and a weak result is not evidence that the model is safe. Report the measured accuracy, and the predicted one next to it when you can estimate $\kappa$. The auxiliary data must also match the victim's population, or the meta-classifier learns the difference between the populations rather than the share.

## Neural networks: choose the right representation {#sec-prop-symmetry}

A neural network can be written down in many ways. Swap two hidden units, together with their incoming and outgoing weights, and the network computes the same function. With the tanh activation, flipping the signs of one unit's weights also changes nothing. Training treats all these equivalent weight vectors alike, so every raw weight has the same mean under both shares, and a meta-classifier fed the flattened weights sees nothing (Ganju et al. 2018).

The fix is a representation that ignores the order of the units. The full edition compares four on small tanh networks, for shares 0.2 against 0.8. A linear meta-classifier on raw parameters scores 0.497, which is chance, while one on the network's outputs at 16 fixed probe inputs scores 1.000. Sorting the units by their outgoing weights recovers most of the signal, but it is fragile when two units have nearly equal weights. Sums over the units, the idea behind DeepSets (Zaheer et al. 2017), need no sorting and extend to deeper networks layer by layer. Probes do best, because the property changes the function the network computes and probes measure it directly.

Probe outputs need only query access, so a model behind an application programming interface is not safe either, and Zhang, Tople & Ohrimenko (2021) attack properties through queries alone.

## Defences and their limits {#sec-prop-defences}

Because a property is a large group of records, there is no cheap defence. A released statistic hides it only if its noise is on the scale of $n|t_1 - t_0|$ records, and that much noise also destroys what the statistic was for. Removing the group column does not help, because whatever correlates with the group still moves the model.

If the property is a secret (@sec-prop-game), the strongest option is not to release a model trained on that mix, and query access is no way out. Pooling data with other owners is no fix, because Zhang, Tople & Ohrimenko (2021) attack exactly that multi-party setting, and in federated learning the same attack can target one client's updates. If you must release, run the attack recipe with your own data as the auxiliary data. Then report its accuracy for the shares an adversary would plausibly compare, and state in the release notes that DP protects individuals only.

## Code {#sec-prop-code}

The functions below compute the ceiling, the TV distance between the two binomial counts, and the best membership advantage, and print numbers of @tbl-prop-count.

```python
from math import exp, lgamma, log


def binom_pmf(n: int, t: float) -> list[float]:
    """Pr[Bin(n, t) = k] for k = 0..n, in log space."""
    return [exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(t) + (n - k) * log(1 - t)) for k in range(n + 1)]


def property_ceiling(n: int, t0: float, t1: float) -> float:
    """Best advantage of any property attack when the group is visible: TV(Bin(n, t0), Bin(n, t1))."""
    return 0.5 * sum(abs(p - q) for p, q in zip(binom_pmf(n, t0), binom_pmf(n, t1)))


def membership_best(n: int, t: float) -> float:
    """Best advantage for one subgroup record from the same count: (1 - t) max_k Pr[Bin(n - 1, t) = k]."""
    return (1 - t) * max(binom_pmf(n - 1, t))


print(f"{property_ceiling(100, 0.45, 0.55):.4f} {property_ceiling(1000, 0.45, 0.55):.4f}")  # 0.6827 0.9985
print(f"{membership_best(100, 0.5):.4f} {membership_best(10000, 0.5):.4f}")  # 0.0398 0.0040
```

The full edition's script, `code/part02-attacks/property_inference.py`, runs the shadow-model attacks and reproduces every other number in this chapter.

## Cheat sheet

| Question | Answer |
| --- | --- |
| How to score an attack | Accuracy $= \frac12(1 + \mathrm{TPR} - \mathrm{FPR})$; a coin flip scores $\frac12$ |
| How the leak scales | Through $\sqrt n\,\lvert t_1 - t_0\rvert$; grows with $n$, while membership falls like $1/\sqrt n$ |
| Predicted accuracy of a model | $\Phi(\kappa\sqrt n\,\lvert t_1 - t_0\rvert/2)$; the count has $\kappa = 2$ at a midpoint of $\frac12$ |
| Which representation for a network | Probe outputs first, then sums over units or canonical order; never raw weights |
| What DP gives | Protection for one record; little for a property, which changes $n\lvert t_1 - t_0\rvert$ records |

: The results of this chapter. {#tbl-prop-cheat}

**Rules of thumb**

- First ask whether the property describes the data owner. If it describes the world, it is not a secret.
- Record-level DP, or a low membership risk, does not imply property privacy.
- A meta-classifier at chance on raw weights means the representation is wrong, not that the model is safe.
- Shadow datasets must come from the target's distribution, or the attack learns the wrong difference.

## Further reading

- Ganju, Wang, Yang, Gunter & Borisov, "Property Inference Attacks on Fully Connected Neural Networks using Permutation Invariant Representations" (CCS 2018): the practical attack on neural networks, neuron sorting and the DeepSets meta-classifier.
- Zhang, Tople & Ohrimenko, "Leakage of Dataset Properties in Multi-Party Machine Learning" (USENIX Security 2021): black-box property inference, including properties of attributes that were removed from the training data.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/04-property-inference.md).
