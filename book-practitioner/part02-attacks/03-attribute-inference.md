# Attribute inference and model inversion

## Why this matters

A hospital's public model predicts a lung disease from a patient's age band and whether they smoke. An insurer knows your age and, from a claim, that you have the disease. It can use the model to guess whether you smoke, but that alone is not a privacy failure, since a published study on smoking would allow the same guess. The question that matters is this. **Does the model reveal more about your attribute because your record was in its training data?**

**Attribute inference** tries to recover one sensitive field of a record, such as smoking status or a genotype, from its other fields and a model. The one idea of this chapter is to separate two kinds of success. **Population inference** works equally on everyone, and no defence that limits what one record contributes can stop it. The individual leak is the extra success on people who were in the training data.

## The attribute game {#sec-attr-game}

"The attack recovered 78% of smoking statuses" means nothing on its own, so Yeom et al. (2018) defined a game ([Threat models and privacy games](01-threat-models.md)). In it, the adversary wants the sensitive attribute $s$ and already knows the public features $x$. It also sees the label $y$, which the model predicts.

The challenger trains a model, then picks the target either from the training set, making it a **member**, or fresh from the population, with equal chance. The adversary sees the target's $x$ and $y$ and queries the model, and then guesses $s$. The score is the **attribute advantage**,
$$\mathrm{Adv}^{\mathrm{attr}} = \Pr[\text{guess right} \mid \text{member}] - \Pr[\text{guess right} \mid \text{non-member}].$$ {#eq-attr-advantage}
Population inference helps members and non-members equally, so it cancels in the difference, and what is left is what the model knows about its own training records.

A toy experiment shows how small the advantage can be. Against 200 small logistic regressions, model inversion is right on 0.7832 of members. Yet the attribute advantage is only 0.0569. Most of the attack's 78% would be there even if the person had never been in the training data.

Any attribute attack is also a membership attack that says "member" when its guess is right, with the same advantage. So a model that leaks nothing about membership leaks nothing about attributes, though the converse fails. As a membership test it is weak, because in the toy experiment being right is only 1.08 times as likely for a member as for a non-member.

## The imputation baseline {#sec-attr-imputation}

The **imputation baseline** is the success of the best possible guess of $s$ from $x$ and $y$, made by someone who knows the population exactly. Jayaraman & Evans (2022) found that many published attacks do no better than imputation, so this is the number to beat.

The best guess for each pair of $x$ and $y$ is the most probable value of $s$, the maximum a posteriori (MAP) guess. Write $p(s, x, y)$ for the probability that a record from the population has these three values. The MAP guess then succeeds with probability
$$\text{baseline} = \sum_{x, y}\max_s p(s, x, y).$$ {#eq-attr-ceiling}
This is also a ceiling for non-members. A model trained without the target is independent of it, so no attacker does better on a non-member. Attacks that ignore the model have zero advantage, because without the model a member is just another draw from the population.

@tbl-attr-smoking works this out for the smoking example. Smokers make up 30% of the population and tend to be older, and the disease is more likely with age and with smoking.

| $p(s, x, y)$ | young, no disease | young, disease | old, no disease | old, disease |
| --- | --- | --- | --- | --- |
| non-smoker ($s = 0$) | **0.378** | 0.042 | **0.196** | 0.084 |
| smoker ($s = 1$) | 0.072 | **0.048** | 0.054 | **0.126** |
| $\Pr[\text{smoker} \mid x, y]$ | 0.1600 | 0.5333 | 0.2160 | 0.6000 |

: The smoking table: the joint distribution of smoking status $s$, age band $x$ and disease $y$, with the column maxima in bold and the probability of smoking in each column. {#tbl-attr-smoking}

With no observation, the best guess is "non-smoker", right 70% of the time. But the chance of smoking crosses one half in the two disease columns, so there the MAP guess becomes "smoker". Its success is the sum of the bold entries, 0.748.

Every one of these numbers comes from the population, not from any model. So dropping the sensitive column from the model's inputs does not stop this inference, because the correlates in $x$ and $y$ remain. As Dwork & Roth (2014) put it, a study showing that smoking causes cancer discloses a fact about the world, not about its participants. Do not count such a finding as a breach. Population inference can still hurt, but it is a question for policy and fairness.

In practice you never know $p(s, x, y)$, so estimate the baseline from data. Take records from the same population that the model never saw, then fit the best imputer you can on half of them and measure its success on the rest. With a few discrete fields, the imputer is a table of the most common $s$ in each cell, as in the snippet of @sec-attr-code. For continuous fields, train instead any good classifier to predict $s$ from $x$ and $y$, such as scikit-learn's `HistGradientBoostingClassifier`. The Adversarial Robustness Toolbox packages this as `AttributeInferenceBaseline`. Your imputer may fall short of the true ceiling, so the estimate errs low. An attack that seems to beat it on non-members points to a weak imputer, not a leak.

## Model inversion: the MAP attack {#sec-attr-inversion}

The imputation guess needs the probability of the label given the other fields, $p(y \mid x, s)$, and a model that predicts the label from them estimates it. Fredrikson et al. (2014) called the attack built on this **model inversion**. Writing $f_y(x, s)$ for the model's confidence in the known label when it is fed the public features and a candidate value of $s$, the **MAP inversion attack** guesses
$$\hat s(x, y) = \arg\max_s\; p(s \mid x)\, f_y(x, s).$$ {#eq-attr-map-inversion}
The first factor comes from the adversary's knowledge of the population, and using the overall frequency $p(s)$ instead throws away what the public features say. The second comes from one query per candidate, which against a scikit-learn model is a call to `predict_proba`, reading the column of the known label.

Take an old patient with the disease, and a calibrated model, one whose confidences equal the population's probabilities. An old person smokes with probability 0.3913, and the model gives the disease a probability of 0.7 for a smoker. So "smoker" scores 0.2739 against 0.1826 for "non-smoker", and the attack guesses "smoker". With a calibrated model, inversion simply *is* imputation, with zero advantage.

A trained model differs from the population in two ways. On members it is too sure of the truth, which is the generalization gap of [Membership inference](02-membership-inference.md) seen through the attribute. On non-members its confidences are noisy, so its guesses fall below the ceiling. In the toy experiment, inversion beats the ceiling of 0.7517 on members but not on non-members. So measuring an attack only on fresh records understates what it achieves on members.

More data closes both gaps. With 25 training records the advantage is 0.1610, and with 1,600 it is only 0.0052.

The same idea can invert a whole input. For example, `MIFace` in the toolbox recovers a typical face of a class, which is a privacy problem when the class is one person.

## Confidence masking {#sec-attr-masking}

Many interfaces round their confidences or return only the top label, a defence that Fredrikson et al. (2015) suggested. Such **confidence masking** cannot help the best attacker, because whoever holds the full answers can apply the mask themselves. But it bounds nothing about the gap between members and non-members, which is the leak, and the gap can shrink or grow.

| Interface | Members | Non-members | Advantage |
| --- | --- | --- | --- |
| Full confidences | 0.7832 | 0.7262 | 0.0569 |
| Confidences rounded to one decimal | 0.7836 | 0.7253 | 0.0583 |
| Top label only | 0.7741 | 0.7092 | 0.0650 |

: MAP inversion success through three interfaces, averaged over 200 models trained on 100 records each. Guessing from $x$ alone gives 0.7045, and imputation with the true population 0.7517. {#tbl-attr-masking}

In @tbl-attr-masking, rounding changes almost nothing. With labels only, non-member success drops close to the guess from $x$ alone, but member success drops less, so the advantage grows. Masking has hurt population inference more than the individual leak, and label-only membership attacks exploit the same asymmetry.

So don't count a drop in overall attack success as a privacy gain. Less overfitting shrinks the member gap, and [differential privacy](../part03-dp-foundations/01-definition.md) bounds it for every attacker and every output format.

## Deciding whether there is a leak {#sec-attr-decision}

An audit of attribute leakage follows four steps.

1. **Estimate the baseline** from held-out data, as in @sec-attr-imputation.
2. **Run MAP inversion**, or your strongest attack, on training members and on as many held-out non-members. The Adversarial Robustness Toolbox's `AttributeInferenceBlackBox` is a ready-made attack. For synthetic data, Anonymeter's `InferenceEvaluator` measures the same risk.
3. **Compare the two rates**, with a confidence interval for each. If member success is not clearly above non-member success, this attack finds no individual leak.
4. **Classify what remains.** Success up to the baseline is population inference, a policy question. Success on members beyond non-members is the individual leak, which more data or differential privacy addresses.

## Code {#sec-attr-code}

The snippet estimates the imputation baseline from held-out records of the smoking population, and its estimate lands close to the exact 0.748.

```python
import random
from collections import Counter
from collections.abc import Hashable

P_SMOKER = 0.3
P_OLD = {0: 0.4, 1: 0.6}  # Pr[old | s]
P_DISEASE = {(0, 0): 0.1, (0, 1): 0.3, (1, 0): 0.4, (1, 1): 0.7}  # Pr[disease | s, old]
Record = tuple[int, Hashable]  # (sensitive value s, the fields the attacker sees)


def draw(rng: random.Random) -> Record:
    s = int(rng.random() < P_SMOKER)
    old = int(rng.random() < P_OLD[s])
    return s, (old, int(rng.random() < P_DISEASE[(s, old)]))


def imputation_baseline(fit: list[Record], test: list[Record]) -> float:
    """Learn the most common s for each observed cell on `fit`, and return its success on `test`."""
    counts: dict[Hashable, Counter[int]] = {}
    for s, seen in fit:
        counts.setdefault(seen, Counter())[s] += 1
    fallback = Counter(s for s, _ in fit).most_common(1)[0][0]
    guess = {seen: c.most_common(1)[0][0] for seen, c in counts.items()}
    return sum(guess.get(seen, fallback) == s for s, seen in test) / len(test)


rng = random.Random(0)
held_out = [draw(rng) for _ in range(20_000)]
print(f"{imputation_baseline(held_out[:10_000], held_out[10_000:]):.4f}")  # 0.7468
```

## Cheat sheet

| Quantity | Formula or rule | What it tells you |
| --- | --- | --- |
| Attribute advantage | Success on members $-$ success on non-members | The individual leak; report both rates too |
| Imputation baseline | $\sum_{x, y}\max_s p(s, x, y)$ | Population inference; estimate on held-out data |
| MAP inversion | $\hat s = \arg\max_s p(s \mid x)\,f_y(x, s)$ | One query per candidate value; use $p(s \mid x)$, not $p(s)$ |
| Masking | Cannot help the best attacker | But the member/non-member gap can grow |

: The results of this chapter. {#tbl-attr-cheat}

**Rules of thumb**

- Report success on members, on non-members, and the imputation baseline. One success rate, such as 80%, says nothing about privacy until you know the other two.
- An attribute attack is a weak membership test, so measure membership with a dedicated attack.
- Rounding confidences or returning labels only changes the attack, not what the model memorized.

## Further reading

- Fredrikson, Jha & Ristenpart, "Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures" (CCS 2015): inversion from confidences, face reconstruction, and rounding as a countermeasure.
- Jayaraman & Evans, "Are Attribute Inference Attacks Just Imputation?" (CCS 2022): the imputation baseline, and what it takes to beat it.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/03-attribute-inference.md).
