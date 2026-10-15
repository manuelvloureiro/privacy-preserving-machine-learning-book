# Attribute inference and model inversion

## Why this matters

A hospital trains a model that predicts a lung disease from a patient's age band and whether they smoke, and puts it behind an interface anyone can query. An insurer knows your age and, from a claim, that you have the disease. Can it use the model to find out whether you smoke? It can make a better guess than without the model. But that is not yet a privacy failure: if smokers get the disease more often, a published study saying so would let it make the same guess about anyone with the disease. The question that matters for *you* is **whether the model reveals more about your attribute because your record was in its training data.**

**Attribute inference** asks an adversary to recover one sensitive field of a record, such as smoking status, a genotype or an income band, from the other fields and access to a model. **Model inversion** is the classic way to do it: query the model once for each candidate value of the sensitive field, and keep the value that best explains the label you already know.

The one idea: **compare with the guesser who never sees the model.** Picture a second insurer with no model but perfect knowledge of the population. Its best guess from the visible fields is the **imputation baseline**. For a record that was not in the training set, no attack can beat it; for a record that was, an attack can. Success up to the baseline is population inference; success on members beyond what non-members get is the leak about individuals. So measuring the leak means comparing members with non-members, which makes attribute inference a membership question in disguise. @tbl-attr-uses maps the decisions to the sections.

| Decision | Rule | Section |
| --- | --- | --- |
| How to score an attribute attack | Success on members minus success on non-members | @sec-attr-game |
| How much is just population inference | Compute the imputation baseline from the population | @sec-attr-imputation |
| How the attack works, and what drives it | Bayes' rule with the model as the likelihood; overfitting | @sec-attr-inversion |
| Does rounding confidences or returning labels only help? | It lowers the attack's success but not the individual leak | @sec-attr-masking |

: The decisions behind evaluating an attribute attack, and where this chapter makes each one. {#tbl-attr-uses}

A record is $(s, x, y)$: a sensitive attribute $s$ with a few values, public features $x$ and a label $y$. The model's **confidence** $f_y(x, s)$ is the probability it gives label $y$ at input $(x, s)$.

## The attribute inference game {#sec-attr-game}

**Idea.** "The attack recovered 78% of smoking statuses" means nothing until you know how often the attribute could be guessed without the model, and how often the attack succeeds on people who were not in the training set. Yeom et al. (2018) made this a game: a challenger trains a model, then flips a fair coin to pick the target either from the training set (a **member**) or fresh from the population (a **non-member**). The adversary sees the target's $x$ and $y$ and the model, and guesses $s$.

**Result.** The **attribute advantage** is
$$\mathrm{Adv}^{\mathrm{attr}} = \Pr[\text{guess right} \mid \text{member}] - \Pr[\text{guess right} \mid \text{non-member}].$$
Anything learned from the population helps members and non-members equally and cancels; what is left is what the model knows about its own training records. Any attribute attack is also a membership attack, one that says "member" when its guess is right: its true positive rate (TPR) is the success on members, its false positive rate (FPR) the success on non-members, and its advantage TPR − FPR is the attribute advantage. So the attribute advantage obeys the same ceiling as membership, the total variation distance between the member and non-member views, and a model that leaks nothing about membership leaks nothing about attributes either.

**Example.** In this chapter's toy experiment, a logistic regression with a binary sensitive attribute (prior 0.3), ten public features and a binary label is trained on $n = 100$ records, 200 times. The inversion attack of @sec-attr-inversion is right on 0.7832 of members and 0.7262 of non-members: an advantage of 0.0569, with a standard error of 0.0045. Read as a membership test it fires on 0.7832 of members and 0.7262 of non-members, a likelihood ratio of only 1.08, and its balanced accuracy is 0.5285. Most of its 78% success would be there if the person had never been in the data.

**Pitfalls.** Never report one success rate. Report the success on members, on non-members and the imputation baseline, as a membership attack reports its TPR and FPR. An attack right on 90% of members and 90% of non-members has zero advantage: it is doing population inference. And as a membership attack, an attribute attack is weak at a low FPR, since it outputs one bit and is right on most people either way; to measure membership, use the attacks of [Membership inference](02-membership-inference.md).

## The imputation baseline {#sec-attr-imputation}

**Idea.** Filling in a missing field from the others is **imputation**. The best possible imputation, by a guesser who knows the population exactly, picks for each visible $(x, y)$ the value of $s$ that is most probable in the population: the **maximum a posteriori (MAP)** guess. Its success rate is the baseline an attack must beat. Jayaraman & Evans (2022) found that many published attribute attacks do no better than imputation from similar data.

**Result.** For a non-member, the model was trained on other people only, so it carries no information about this record beyond the population. Every attack therefore satisfies the **non-member ceiling**
$$\Pr[\text{guess right} \mid \text{non-member}] \le V(S \mid X, Y) = \sum_{x, y}\max_s p(s, x, y),$$
where $p(s, x, y)$ is the population probability of a record, and $V$, the **vulnerability**, is the success of the MAP guess: for each visible $(x, y)$, the probability of its most likely $s$, summed. An attack that ignores the model has the same success on members and non-members, so zero advantage, and Bayes imputation reaches the ceiling in both.

**Example.** Smoking has prior 0.3; a fraction 0.4 of non-smokers and 0.6 of smokers are old; the disease has probability 0.1, 0.3, 0.4 and 0.7 for young non-smokers, old non-smokers, young smokers and old smokers. Multiplying gives @tbl-attr-smoking.

| $p(s, x, y)$ | young, no disease | young, disease | old, no disease | old, disease |
| --- | --- | --- | --- | --- |
| non-smoker ($s = 0$) | **0.378** | 0.042 | **0.196** | 0.084 |
| smoker ($s = 1$) | 0.072 | **0.048** | 0.054 | **0.126** |
| $\Pr[\text{smoker} \mid x, y]$ | 0.1600 | 0.5333 | 0.2160 | 0.6000 |

: The smoking table: the joint distribution of smoking status, age band and disease, with the column maxima in bold and the posterior probability of smoking in each column. {#tbl-attr-smoking}

With no observation the best guess is "non-smoker", right 0.7 of the time. Age alone changes the odds but never the decision, so it still gives 0.7. With age and disease, the guess becomes "smoker" in the two disease columns, and the success is the sum of the bold entries, $V(S \mid X, Y) = 0.748$, a min-entropy leakage of 0.0957 bits. A good model trained on 10,000 patients from this population lets an insurer reach 0.748 on *anyone*, and nobody outside the training set can be attacked more successfully than that.

![The posterior probability of smoking in each cell of the smoking table, with each cell's probability in brackets. Red bars cross $\frac12$, where imputation guesses "smoker".](figures/attr_smoking_posterior.svg){#fig-attr-smoking}

**Pitfalls.** The ceiling is a property of the population, not of the model, and it can be high: if $x$ and $y$ nearly determine $s$, every model and every statistic "reveals" $s$. That is population inference, the "smoking causes cancer" case of Dwork & Roth (2014): the study disclosed a fact about the world, not about its participants. It can still harm people, but no mechanism that limits what one record contributes can prevent it, since the harm would remain with your record removed. It is a question for policy and fairness. In an audit you rarely know the population exactly; estimate the baseline with the best imputation model you can fit on data from the same population, and run the attack on held-out non-members as well.

## Model inversion: the MAP attack {#sec-attr-inversion}

**Idea.** Imputation needs the link $p(y \mid x, s)$ between label and attribute, and the adversary usually does not know it. A model that predicts $y$ from $x$ and $s$ is an estimate of exactly that link. Fredrikson et al. (2014) used this on a warfarin dosing model to infer patients' genetic markers from their demographics and dose, and called it **model inversion**. Confidences make it stronger (Fredrikson, Jha & Ristenpart 2015).

**Result.** The **MAP inversion attack** applies Bayes' rule with the model as the likelihood:
$$\hat s(x, y) = \arg\max_s\; p(s \mid x)\, f_y(x, s).$$ {#eq-attr-map-inversion}
The first factor is what the adversary knows about how $s$ relates to the public features; the second is the model's confidence in the known label when fed the candidate $s$. If the model is **calibrated**, its confidences equal the population's probabilities, and inversion *is* imputation: success $V(S \mid X, Y)$ on members and non-members alike. Everything the attack gains on members comes from the model being too sure of each training record at its true $s$, which is overfitting, the generalization gap of [Membership inference](02-membership-inference.md) seen through the sensitive attribute.

**Example.** On the smoking table, take an old patient with the disease. The adversary knows $\Pr[\text{smoker} \mid \text{old}] = 0.3913$ and queries a calibrated model, which returns 0.7 for a smoker and 0.3 for a non-smoker. The scores are 0.2739 for "smoker" and 0.1826 for "non-smoker", so it guesses "smoker", and normalized they give the posterior 0.6000 of @tbl-attr-smoking. On the toy model, Bayes imputation is right 0.7517 of the time. MAP inversion with the trained model is right on 0.7832 of members, 0.0315 *above* that ceiling, which no population knowledge could produce, and on 0.7262 of non-members, 0.0254 below it. The leak shrinks with the training-set size: the advantage is 0.1610 at $n = 25$, 0.0111 at $n = 400$ and 0.0052 at $n = 1600$ (@fig-attr-success-vs-n).

![Success of MAP inversion on members and non-members against the training-set size $n$. Members start above the Bayes imputation ceiling and non-members below it; both converge to the ceiling as the model approaches the population.](figures/attr_success_vs_n.svg){#fig-attr-success-vs-n}

**Pitfalls.** Use $p(s \mid x)$, not the marginal $p(s)$ that Fredrikson et al. used, or the attack throws away what $x$ says about $s$. On non-members an overfit model does *worse* than imputation, so a weak result on fresh records says nothing about members. The same idea can invert a whole input: on a face recognition model whose classes are people, a search for the image that maximizes one class's confidence recovers a blurred, recognizable face. That is population inference about the class, and a privacy problem because the class is one person.

## Confidence masking {#sec-attr-masking}

**Idea.** If confidences make inversion stronger, why release them? Rounding them, or returning only the top label, is cheap and leaves the prediction unchanged. Such **confidence masking** only coarsens what the adversary sees, and whoever sees the full answers could apply the mask themselves.

**Result.** For any masking function $g$, applied to the answers $F$ the adversary gets by querying every candidate $s$, the best attack on the masked answers does no better than the best on the full ones, on members and on non-members separately:
$$V\big(S \mid X, Y, g(F)\big) \le V\big(S \mid X, Y, F\big).$$
The inequality says nothing about the *difference* between members and non-members, and masking need not shrink it. Labels alone still leak: on a member, the model usually predicts the training label at the true $s$, because it fit that record, and an honest label at the other value, so "the label matches at only one candidate" points to the truth.

**Example.** @tbl-attr-masking attacks the same 200 models through three interfaces. Rounding to one decimal changes almost nothing. Returning labels only drops non-member success to 0.7092, close to the 0.7045 of guessing from $x$ alone, while member success stays 0.0224 above the imputation ceiling, so the advantage *grows* from 0.0569 to 0.0650. The interface now leaks less about the population and as much as before about the training set.

| Interface | Members | Non-members | Advantage |
| --- | --- | --- | --- |
| Full confidences | 0.7832 | 0.7262 | 0.0569 |
| Confidences rounded to one decimal | 0.7836 | 0.7253 | 0.0583 |
| Top label only | 0.7741 | 0.7092 | 0.0650 |

: MAP inversion success through three interfaces, averaged over 200 models trained on 100 records each. Guessing from $x$ alone gives 0.7045 and Bayes imputation 0.7517. {#tbl-attr-masking}

![MAP inversion success on members and non-members through three interfaces. The dashed line is Bayes imputation, the dotted line the guess from the public features alone.](figures/attr_masking.svg){#fig-attr-masking}

**Pitfalls.** Masking is obfuscation, not a guarantee: it removes one attack's input without bounding what any attack learns about a member. Don't count a drop in overall attack success as a privacy gain; check the member/non-member gap. What shrinks the gap is less overfitting, for example through more training data, and what bounds it for every attacker and every output format is [differential privacy](../part03-dp-foundations/01-definition.md).

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

The full edition's script, [attribute_inference.py](../../code/part02-attacks/attribute_inference.py), plays the game on the toy model and prints every number in this chapter:

```bash
uv run python code/part02-attacks/attribute_inference.py
```

## Cheat sheet

| Result | Statement | Notes |
| --- | --- | --- |
| Attribute advantage | Success on members $-$ success on non-members | Population inference cancels |
| Reduction to membership | "Member if the guess is right": TPR $-$ FPR $=$ attribute advantage | Bounded by the membership ceiling; weak at low FPR |
| Non-member ceiling | Success on non-members $\le V(S \mid X, Y) = \sum_{x, y}\max_s p(s, x, y)$ | Imputation reaches it; model-free attacks have zero advantage |
| MAP inversion | $\hat s = \arg\max_s p(s \mid x)\,f_y(x, s)$ | Calibrated model: inversion is imputation |
| Masking | $V(S \mid X, Y, g(F)) \le V(S \mid X, Y, F)$ | The member/non-member gap can grow |

: The results of this chapter. {#tbl-attr-cheat}

**Rules of thumb**

- Report success on members, on non-members, and the imputation baseline. One success rate says nothing about privacy.
- Success up to the imputation baseline is population inference; any model or statistic of the population gives it.
- Success on members beyond non-members is the individual leak. It comes from overfitting and shrinks as the training set grows.
- Rounding confidences or returning labels only changes the attack, not what the model memorized.

## Further reading

- Yeom, Giacomelli, Fredrikson & Jha, "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting" (IEEE CSF 2018): the attribute game with members and non-members, and its link to membership and overfitting.
- Fredrikson, Jha & Ristenpart, "Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures" (CCS 2015): inversion from confidences, face reconstruction, and rounding as a countermeasure.
- Jayaraman & Evans, "Are Attribute Inference Attacks Just Imputation?" (CCS 2022): the imputation baseline, and what it takes to beat it.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/03-attribute-inference.md).
