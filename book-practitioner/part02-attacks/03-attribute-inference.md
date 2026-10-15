# Attribute inference and model inversion

## Why this matters

A hospital trains a model that predicts whether a patient will develop a lung disease from their age band and whether they smoke. It puts the model behind an interface that anyone can query. An insurer knows your age and, from a claim, that you have the disease. Can it use the model to find out whether you smoke?

It can make a better guess than it could without the model, but that alone is not a privacy failure. If smokers get the disease more often, the insurer learns something about every person with the disease, whether or not their record was used for training. A published study saying that smoking causes the disease would allow the same guess. The question that matters for you is a different one. **Does the model reveal more about your attribute because your record was in its training data?**

**Attribute inference** tries to recover one sensitive field of a record, such as smoking status, a genotype or an income band. The attacker uses the record's other fields and access to a model that takes the sensitive field as one of its inputs. **Model inversion** is the classic way to do it. You query the model once for each possible value of the sensitive field. Then you keep the value whose prediction best explains the person's outcome, such as the diagnosis, which you already know.

This chapter compares the attack with a guesser who never sees the model but knows the population perfectly. That separates what the attack learns about you from what it learns about everyone. No attack can beat the guesser on people outside the training data, so any excess on people inside it is a leak about individuals. Measuring that leak turns attribute inference into a membership question.

## The attribute inference game {#sec-attr-game}

"The attack recovered 78% of smoking statuses" sounds alarming, but the number means nothing on its own. You also need to know how often the attribute could be guessed without the model, and how often the attack succeeds on people who were not in the training set. Yeom et al. (2018) made this precise with a privacy game. As in [Threat models and privacy games](01-threat-models.md), a challenger sets up a hidden situation, the adversary sees only what the rules give it, and its guess is scored.

The game needs a little notation, because a record has three fields that play different roles. The sensitive attribute $s$ is the field the adversary wants, and it takes a few values, such as smoker or non-smoker. The public features $x$ are what the adversary already knows about the person, such as the age band. The label $y$ is what the model predicts, such as the disease. Records are drawn from a population, and the training set consists of records drawn independently from it.

The game runs in three steps. First, the challenger draws a training set and trains a model on it. Second, it flips a fair coin. On heads it picks the target uniformly at random from the training set, so the target is a **member**. On tails it draws a fresh target from the population, so the target is a **non-member**. Third, the adversary receives the target's $x$ and $y$, but not $s$, together with access to the model and whatever it knows about the population. It outputs a guess of $s$. The coin models nothing an attacker does, and only lets the two success rates be compared under one set of rules.

The score measures how much better the attack does on people in the training data than on others. The **attribute advantage** is the success rate on members minus the success rate on non-members:
$$\mathrm{Adv}^{\mathrm{attr}} = \Pr[\text{guess right} \mid \text{member}] - \Pr[\text{guess right} \mid \text{non-member}].$$ {#eq-attr-advantage}
The subtraction is what makes this a measure of privacy. Some success comes from patterns that hold across the population, such as "people with the disease tend to smoke". This chapter calls that **population inference**. It helps members and non-members equally, so it appears in both terms and cancels. What is left is what the model knows about its own training records. Because the advantage hides the success the two groups share, report the two success rates as well.

The advantage also turns any attribute attack into a membership attack, which sees a whole record and the model and says whether the record was in the training set. As [Membership inference](02-membership-inference.md) explains, its true positive rate (TPR) is the chance that it says "member" for a member. Its false positive rate (FPR) is the chance that it says "member" for a non-member, and its advantage is TPR minus FPR. If an attribute attack is right more often on members, then being right is itself evidence of membership. So a membership attack can run the attribute attack, compare its guess with the true $s$ in the record, and say "member" exactly when the guess is right. Its TPR is then the attribute attack's success on members, and its FPR the success on non-members. Its advantage is therefore the attribute advantage.

This reduction gives every attribute attack a ceiling. Picture everything a membership attack could look at, the target record together with the model. Run the game many times, and this view follows one distribution when the target is a member and another when it is not. No test can tell two distributions apart with an advantage above their **total variation distance**, half the total absolute difference between their probabilities ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). The membership attack built above obeys this ceiling, so every attribute attack does too.

So a model that reveals nothing about membership, where the two distributions are equal, gives no attribute advantage to anyone. The opposite direction fails, because a membership leak need not come with an attribute leak. Suppose the sensitive attribute has no bearing on the label. A model can then memorize its training records through their features and labels while its predictions ignore $s$, and no attribute attack gains anything on members.

The membership test built this way is also weak. It outputs one bit, so a "member" verdict can only shift the odds that the target is a member by the ratio of its TPR to its FPR. An attack that is right on most people, members or not, has a ratio close to 1, so its verdict barely shifts those odds. Dedicated membership attacks are judged by their TPR at a low FPR, where each verdict is strong evidence. There, a test built from an attribute attack is much weaker than the attacks of [Membership inference](02-membership-inference.md).

The toy experiment of this chapter (@sec-attr-code) shows how small the advantage can be behind a large success rate. It trains a logistic regression on 100 records, 200 times over. Each time, a model inversion attack, explained in @sec-attr-inversion, guesses a two-valued sensitive attribute for the 100 members and for 100 fresh records. The attack is right on 0.7832 of members and on 0.7262 of non-members. The attribute advantage is therefore only 0.0569. Like every difference on this page, it is computed from the unrounded rates, which is why it is not 0.0570. Its standard error over the 200 models is 0.0045. Read as a membership test, being right is only 1.08 times as likely for a member as for a non-member. Most of the attack's 78% success would be there even if the person had never been in the training data.

## The imputation baseline {#sec-attr-imputation}

The game removes population inference by subtracting it. Measuring it directly is just as useful, because it tells you how much success any model of the same population would give, or any published statistic about it. Statisticians call filling in a missing field from the others **imputation**. The **imputation baseline** is the success rate of the best possible guess of $s$ from $x$ and $y$, made by a guesser who knows the population exactly. Jayaraman & Evans (2022) found that many published attribute inference attacks do no better than imputation from similar data. This baseline is the one an attack must beat.

The best possible guess is easy to describe. Whatever a guesser answers after seeing $x$ and $y$ is right exactly when the true $s$ equals its answer. So for each pair, the best answer is the value of $s$ that is most probable together with it. This is the maximum a posteriori (MAP) guess, "a posteriori" because it is made after seeing the observation.

Write $p(s, x, y)$ for the probability that a record from the population has these three values. For a given pair of $x$ and $y$, the MAP guess is right on the records with that pair and the guessed value of $s$. Their total probability is $\max_s p(s, x, y)$, and adding over all pairs gives the success rate of the MAP guess:
$$V(S \mid X, Y) = \sum_{x, y}\max_s p(s, x, y).$$
[Information theory](../part01-prerequisites/05-information-theory.md) calls this success rate the **vulnerability**, hence the letter $V$. The capital letters stand for the fields as random quantities.

The ceiling for non-members comes from one more fact. An observation that is independent of the record, such as a coin flip, cannot change which value of $s$ is most likely, so it cannot raise the best success. A model trained without the target is such an observation. A non-member is drawn afresh, separately from the training set, and the model depends only on the training set and on the randomness of training. So the best any adversary can do on a non-member is the best guess from $x$ and $y$ alone. This is the **non-member ceiling**. For every adversary in the game,
$$\Pr[\text{guess right} \mid \text{non-member}] \le V(S \mid X, Y).$$ {#eq-attr-ceiling}
The ceiling rests entirely on that independence. A member's record was used to train the model, so for a member the model is not an independent observation, and the ceiling does not apply.

Attacks that never look at the model cannot tell members from non-members at all. The member target is one record of the training set, chosen at random, and each training record is an independent draw from the population. If you ignore the model, a member is therefore just a record drawn from the population, like a fresh one. An attack that uses only $x$ and $y$ has the same success in both worlds, and zero advantage. The MAP guess made with the population's true probabilities, called **Bayes imputation**, is such an attack, and it reaches the ceiling for members and non-members alike. The game guarantees this by drawing the training set and the fresh target from the same population. Without that, members would differ from non-members before anyone looked at the model.

A worked population makes this concrete. Let $s = 1$ mean "smoker", let $x$ be the age band, young or old, and let $y = 1$ mean "has the disease". Smokers make up 30% of the population. They also tend to be older, since a fraction 0.6 of them is old, against 0.4 of non-smokers. The chance of the disease rises with both age and smoking, so for non-smokers it is 0.1 when young and 0.3 when old. For smokers it is higher, 0.4 when young and 0.7 when old.

To get the probability of a full record, multiply three chances. They are the chance of its smoking status, the chance of its age band given that status, and the chance of its disease outcome given both. The result is the joint table, @tbl-attr-smoking. For example, an old smoker with the disease has probability $0.3 \times 0.6 \times 0.7 = 0.126$.

| $p(s, x, y)$ | young, no disease | young, disease | old, no disease | old, disease |
| --- | --- | --- | --- | --- |
| non-smoker ($s = 0$) | **0.378** | 0.042 | **0.196** | 0.084 |
| smoker ($s = 1$) | 0.072 | **0.048** | 0.054 | **0.126** |
| $\Pr[\text{smoker} \mid x, y]$ | 0.1600 | 0.5333 | 0.2160 | 0.6000 |

: The smoking table: the joint distribution of smoking status $s$, age band $x$ and disease $y$, with the column maxima in bold and the posterior probability of smoking in each column. {#tbl-attr-smoking}

The last row divides each column's smoker entry by the column total, which gives the chance that a person in that column smokes. With no observation, the best guess is "non-smoker", which is right 70% of the time. Age alone changes the odds but not the decision. Adding the two columns of each age band shows that a young person smokes with probability 0.2222, and an old person with probability 0.3913. Since both are below one half, the guess stays "non-smoker" in both age bands, and the success stays at 0.7.

With age and disease together, the chance of smoking crosses one half in the two disease columns (@fig-attr-smoking), so the MAP guess becomes "smoker" there. Its success is the sum of the bold entries, the largest in each column, which gives a vulnerability of 0.748. In the terms of [Information theory](../part01-prerequisites/05-information-theory.md), this gain over the blind guess is a **min-entropy leakage** of 0.0957 bits. That is the base-2 logarithm of the ratio of the two success rates.

![The posterior probability of smoking in each cell of the smoking table, with each cell's probability in brackets. Red bars cross $\frac12$, where imputation guesses "smoker"; together they add 0.048 to the 0.7 of the blind guess.](figures/attr_smoking_posterior.svg){#fig-attr-smoking}

Every one of these numbers comes from the population, not from any model. A model trained on 10,000 patients from this population, if it predicts well, lets an insurer reach 0.748 on anyone drawn from it. The ceiling says that nobody outside the training set can be attacked more successfully. Dwork & Roth (2014) make the same point about a study showing that smoking causes cancer, which discloses a fact about the world, not about its participants. A privacy definition that forbade this would forbid learning anything from data.

That does not make population inference harmless, since being guessed to be a smoker hurts whether or not the guess used your record. But no defence that limits what any single record contributes to a model can prevent it, because the harm would remain with your record removed. It is a question for policy and fairness, not for the attacks of this part.

The ceiling is also a property of the population, not of the model, and it can be high. If the public features and the label nearly determine the sensitive attribute, every model and every statistic "reveals" it. That is why attacks should be reported relative to the ceiling.

## Model inversion: the MAP attack {#sec-attr-inversion}

The imputation guess needs the probability of the label given the other two fields, $p(y \mid x, s)$, which the adversary usually does not know. But a model that predicts $y$ from $x$ and $s$ is an estimate of exactly that link. Fredrikson et al. (2014) used this to infer patients' genotypes from a warfarin dosing model, given their demographics and dose. They called the attack **model inversion**, because it runs the model backwards from its output to an input. A model's **confidences** are the probabilities it predicts for each label. Fredrikson, Jha & Ristenpart (2015) showed that releasing them makes the attack much stronger.

The attack is Bayes' rule. Write $f_y(x, s)$ for the model's confidence in the label when it is fed the public features and a candidate value of the sensitive attribute. Bayes' rule says that the most likely value of $s$ maximizes the product of two factors. The first factor, $p(s \mid x)$, says how plausible $s$ is for someone with these features. The adversary supplies it from what it knows about the population. The second factor says how likely the known label is if the attribute were $s$, and the adversary gets it by querying the model at each candidate. The **MAP inversion attack** guesses
$$\hat s(x, y) = \arg\max_s\; p(s \mid x)\, f_y(x, s).$$ {#eq-attr-map-inversion}

Bayes' rule also divides by the probability of the label given $x$, but that is the same for every candidate, so it cannot change which one wins. Fredrikson et al. used the overall frequency $p(s)$ in place of $p(s \mid x)$, which treats the attribute as independent of the public features. That throws away what $x$ says about $s$, and can lose success even when the model's confidences are exactly right.

Everything the attack does well or badly comes from how the model's confidences differ from the population's probabilities. A model whose confidences equal those probabilities is called **calibrated**. Take a calibrated model and an adversary that uses the population's true $p(s \mid x)$. Both factors are then the true ones, so inversion is imputation and reaches the ceiling. The attack queries the model, but it only reads off probabilities that a guesser who knows the population already has. It therefore behaves like a model-free attack, with the same success on members and non-members.

Take an old patient with the disease. From the population, the adversary knows that an old person smokes with probability 0.3913, and so does not with probability 0.6087. A calibrated model, queried at both values of $s$, returns a disease probability of 0.7 for a smoker and 0.3 for a non-smoker. So "smoker" scores $0.3913 \times 0.7 = 0.2739$. "Non-smoker" scores $0.6087 \times 0.3 = 0.1826$, and the attack guesses "smoker". Normalized, the smoker score is 0.6000, the posterior in @tbl-attr-smoking, so the attack is imputation.

A trained model differs from the population in two ways, one for members and one for non-members. On members, the model is too sure of the truth. Training lowers the loss on each training record, which is small when the model gives the record's label a high confidence. It does this at the record's true value of $s$, so the confidence there rises towards 1. The other value of $s$ was never paired with this record, so the model's confidence there is like its confidence on a fresh record. The ratio of the two scores is therefore tilted towards the truth for members. This tilt is the generalization gap of [Membership inference](02-membership-inference.md), the difference between a model's loss on its training records and on fresh ones, seen through the sensitive attribute.

On non-members, the second difference shows. An overfit model's confidences carry the noise of a small training set rather than the population's exact probabilities, so its guesses fall below the ceiling.

The toy experiment shows both effects. Its population has a sensitive attribute that is 1 with probability 0.3, ten public features, and a label from a logistic model whose weights depend on the attribute. Each trained model has one weight vector per value of $s$, so a training record never touches the weights used at its other value. The mean log-loss is 0.2965 on members and 0.6690 on non-members. On a million fresh records, guessing from $x$ alone is right 0.7045 of the time, and Bayes imputation reaches the ceiling of 0.7517.

Over the models of the game, imputation does as well on members as on non-members, up to sampling noise, as the model-free argument predicts. MAP inversion with the trained model is right on 0.7832 of members, which is 0.0315 above the ceiling. But on non-members it is right only 0.7262 of the time, 0.0254 below the ceiling. In this experiment, then, measuring the attack only on fresh records would understate what it achieves on members.

No knowledge of the population could produce success on members above the ceiling. @fig-attr-evidence shows where it comes from instead: for members, training has removed most of the cases where the model strongly prefers the wrong value.

![The evidence of MAP inversion for the true attribute, the log of the score of the true value over the other, for members and non-members of 50 models trained on 100 records each. The attack is right when the evidence is positive; members have far fewer large negative values.](figures/attr_inversion_evidence.svg){#fig-attr-evidence}

More data closes both gaps, members above the ceiling and non-members below it. As the training set grows, the model approaches the population's probabilities, and both success rates approach the ceiling (@fig-attr-success-vs-n). With 25 training records the advantage is 0.1610. It falls to 0.0111 with 400 records. With 1,600 records it is only 0.0052.

![Success of MAP inversion on members and non-members against the training-set size $n$, 40 models per point. Members start above the Bayes imputation ceiling and non-members below it; both converge to the ceiling as the model approaches the population.](figures/attr_success_vs_n.svg){#fig-attr-success-vs-n}

The same idea can invert a whole input rather than one field. Fredrikson, Jha & Ristenpart (2015) ran gradient descent on an image to find the input that a face recognition model most confidently assigns to one person. They recovered a blurred but recognizable image of that person. What such an attack recovers is a typical input of the class, which is population inference about the class. It is a privacy problem because here the class is a single person.

## Confidence masking {#sec-attr-masking}

If confidences make inversion stronger, why release them? Fredrikson, Jha & Ristenpart (2015) suggested rounding the confidences that an interface returns, and many deployed interfaces return only the top label. Such **confidence masking** is cheap and leaves the predicted label unchanged. Does it close the leak?

Masking can never help the best attacker, because it only coarsens what the attacker sees. Whoever holds the full answers can apply the mask themselves and then guess. So any rule that guesses from the masked answers is also a rule on the full answers, and the best rule on the full answers is at least as good. This is the guessing version of the [data processing inequality](../part01-prerequisites/03-divergences.md#sec-div-dpi).

To write this down, let $F$ be the list of answers the adversary gets by querying the model at every candidate value of $s$. Each answer is the model's vector of confidences over the labels. Let $g$ be any masking function applied to these answers, such as rounding, keeping only the top label, or adding noise. Then the best attack on the masked answers does no better than the best attack on the full ones:
$$V\big(S \mid X, Y, g(F)\big) \le V\big(S \mid X, Y, F\big).$$ {#eq-attr-masking}
This holds on members and on non-members separately, since the argument never used which world the target came from. Added noise is no exception, because an adversary with the full answers could draw the noise itself.

The inequality is about the best attacker in each world separately. It says nothing about the difference between members and non-members, which is the leak. Masking can only lower the best success in each world, by amounts that depend on the mask and on how the confidences of members and non-members differ. Nothing forces the two drops to be equal, so the gap can shrink or grow.

Labels alone still leak, for the same reason confidences do. With only the top label, the adversary still sees, for each candidate $s$, whether the predicted label equals the known label $y$. On a member, the model usually predicts the training label at the true $s$, because it fit that record. At the other value it predicts as it would for a fresh record, which matches $y$ less often. So "the label matches at only one candidate" points to that candidate, and on members it usually points to the truth. Non-members rarely get this help. The model fit neither value for them, so its two predicted labels usually agree, and the attack falls back on the guess from $x$ alone. Label-only membership inference (Choquette-Choo et al. 2021) exploits the same asymmetry.

@tbl-attr-masking attacks the same 200 models as before through three interfaces.

| Interface | Members | Non-members | Advantage |
| --- | --- | --- | --- |
| Full confidences | 0.7832 | 0.7262 | 0.0569 |
| Confidences rounded to one decimal | 0.7836 | 0.7253 | 0.0583 |
| Top label only | 0.7741 | 0.7092 | 0.0650 |

: MAP inversion success through three interfaces, averaged over 200 models trained on 100 records each. Guessing from $x$ alone gives 0.7045 and Bayes imputation 0.7517. {#tbl-attr-masking}

Rounding to one decimal changes almost nothing. The attack only compares the scores of the two candidates, and most confidences are far enough from the rounding boundaries that the comparison keeps its sign. With labels only, non-member success drops to 0.7092, close to the 0.7045 of guessing from $x$ alone. Member success drops less, to 0.7741, which is still 0.0224 above the imputation ceiling. So the advantage grows, from 0.0569 to 0.0650 (@fig-attr-masking). Masking has hurt population inference more than the individual leak. The interface now leaks less about the population, and as much as before about the people in the training set.

![MAP inversion success on members and non-members through three interfaces, for models trained on 100 records. The dashed line is Bayes imputation, the dotted line the guess from the public features alone.](figures/attr_masking.svg){#fig-attr-masking}

Small changes mean little either way. MAP inversion with an overfit model is not the best possible attacker, so a mask can raise its success slightly, as rounding does for members here, without contradicting the inequality.

Above all, masking is obfuscation, not a guarantee. It removes one attack's input without bounding what any attack can learn about a member. So don't count a drop in overall attack success as a privacy gain, and check the gap between members and non-members instead. Less overfitting, for example from more training data, shrinks that gap. Part 3 introduces [differential privacy](../part03-dp-foundations/01-definition.md), which bounds the gap itself, for every attacker and every output format.

## Code {#sec-attr-code}

The imputation baseline and the inversion scores of the smoking example take a few lines with the standard library:

```python
from itertools import product
from math import log2

P_SMOKER = 0.3
P_OLD = {0: 0.4, 1: 0.6}  # Pr[old | s]
P_DISEASE = {(0, 0): 0.1, (0, 1): 0.3, (1, 0): 0.4, (1, 1): 0.7}  # Pr[disease | s, old]


def joint(s: int, old: int, disease: int) -> float:
    """Probability of the record (smoker s, old, disease) in the population."""
    p_s = P_SMOKER if s else 1 - P_SMOKER
    p_x = P_OLD[s] if old else 1 - P_OLD[s]
    p_y = P_DISEASE[(s, old)] if disease else 1 - P_DISEASE[(s, old)]
    return p_s * p_x * p_y


def best_guess_success(sees_age: bool, sees_disease: bool) -> float:
    """Success of the MAP guess of s for a guesser who knows the population and sees the chosen fields."""
    cells: dict[tuple[int, int], list[float]] = {}
    for s, old, disease in product((0, 1), repeat=3):
        key = (old if sees_age else -1, disease if sees_disease else -1)
        cells.setdefault(key, [0.0, 0.0])[s] += joint(s, old, disease)
    return sum(max(cell) for cell in cells.values())


def inversion_scores(old: int, confidence: dict[int, float]) -> dict[int, float]:
    """MAP inversion: prior of s given the age band, times the model's confidence in the known label at s."""
    prior = {s: (P_SMOKER if s else 1 - P_SMOKER) * (P_OLD[s] if old else 1 - P_OLD[s]) for s in (0, 1)}
    total = sum(prior.values())
    return {s: prior[s] / total * confidence[s] for s in (0, 1)}


v0, vx, vxy = best_guess_success(False, False), best_guess_success(True, False), best_guess_success(True, True)
print(f"{v0:.4f} {vx:.4f} {vxy:.4f}")  # 0.7000 0.7000 0.7480
print(f"{log2(vxy / v0):.4f}")  # 0.0957
scores = inversion_scores(old=1, confidence={0: 0.3, 1: 0.7})  # a calibrated model, queried for an old patient with the disease
print(f"{scores[1]:.4f} {scores[0]:.4f}")  # 0.2739 0.1826
print(f"{scores[1] / (scores[0] + scores[1]):.4f}")  # 0.6000
```

The full edition's script, [attribute_inference.py](../../code/part02-attacks/attribute_inference.py), plays the game on the toy model and prints every number in this chapter. Each run trains a regularized logistic regression with one weight vector per value of $s$. It then attacks the members and the same number of fresh records with MAP inversion, Bayes imputation and the guess from $x$ alone. The whole run takes a few seconds:

```bash
uv run python code/part02-attacks/attribute_inference.py
```

## Cheat sheet

| Result | Statement | Why it holds |
| --- | --- | --- |
| Attribute advantage | Success on members $-$ success on non-members | Population inference helps both and cancels |
| Reduction to membership | "Member if the guess is right" has TPR $-$ FPR $=$ attribute advantage, at most the total variation distance | Being right more often on members is evidence of membership |
| Non-member ceiling | Success on non-members $\le V(S \mid X, Y) = \sum_{x, y}\max_s p(s, x, y)$ | The model is independent of a fresh record; the MAP guess is the best guess |
| Model-free attacks | Equal success on members and non-members; imputation reaches $V(S \mid X, Y)$ | Ignoring the model, a member is a population record |
| MAP inversion | $\hat s = \arg\max_s p(s \mid x)\,f_y(x, s)$ | Bayes' rule with the model as likelihood |
| Calibrated model, true $p(s \mid x)$ | Inversion is imputation, zero advantage | Its confidences are the population's probabilities |
| Masking | $V(S \mid X, Y, g(F)) \le V(S \mid X, Y, F)$, but the member/non-member gap can grow | Processing cannot help the best guesser |

: The results of this chapter. {#tbl-attr-cheat}

**Rules of thumb**

- Report success on members, on non-members, and the imputation baseline. One success rate says nothing about privacy.
- Success up to the imputation baseline is population inference, and any model or statistic of the population gives it.
- Success on members beyond what non-members get is the individual leak. It comes from overfitting and shrinks as the training set grows.
- An attribute attack is a membership test that outputs one bit, and it is weak at low false positive rates.
- Rounding confidences or returning labels only changes the attack, not what the model memorized.

## Further reading

- Yeom, Giacomelli, Fredrikson & Jha, "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting" (IEEE CSF 2018): the attribute game with members and non-members, and its link to membership and overfitting.
- Fredrikson, Jha & Ristenpart, "Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures" (CCS 2015): inversion from confidences, face reconstruction, and rounding as a countermeasure.
- Jayaraman & Evans, "Are Attribute Inference Attacks Just Imputation?" (CCS 2022): the imputation baseline, and what it takes to beat it.
- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), §1: the smoking example, and why learning about the population is not a privacy breach of the individual.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/03-attribute-inference.md), and its exercise book works out the cost of the marginal prior and the reduction's weakness at low false positive rates. The exceptions are the findings it cites from the literature: Fredrikson et al.'s warfarin and face attacks, their results on confidences and rounding, Jayaraman & Evans's survey of published attacks, and label-only membership inference (Choquette-Choo et al.).
