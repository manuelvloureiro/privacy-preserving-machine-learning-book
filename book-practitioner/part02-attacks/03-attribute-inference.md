# Attribute inference and model inversion

## What it is

An **attribute inference** attack uses a trained model and known fields of a person's record to infer a sensitive field, such as smoking status. For example, an insurer with a person's age and diagnosis could query a hospital model that predicts disease from age and smoking. A public study may support the same inference, so attack success alone does not establish a privacy failure. The relevant measure is how much more the model reveals because it trained on the person's record.

Yeom et al. (2018) turned this question into a game ([Threat models and privacy games](01-threat-models.md)). A fair coin decides whether the target is a training record, a **member**, or a fresh draw from the same population. The attacker knows the target's public features and label, queries the model and guesses the sensitive value. Its score is the **attribute advantage**,
$$\mathrm{Adv}^{\mathrm{attr}} = \Pr[\text{guess right} \mid \text{member}] - \Pr[\text{guess right} \mid \text{non-member}].$$
A guess based only on the population, called **population inference**, helps members and non-members equally because both come from the same population. It cancels in the difference, leaving the model's additional information about individuals.

## When you meet it

Evaluate attribute inference when assessing a claim that a model reveals a sensitive field or when a model omits that field. Only additional success on members is a privacy leak. Inference supported by population-level correlations does not require the person's record and is primarily a policy and fairness concern.

## How to use it

First measure the **imputation baseline**: the best prediction of the sensitive value from public features and the label, without querying the model. Many published attacks do no better (Jayaraman & Evans 2022). Estimate the baseline by fitting a classifier to population records that were not used to train the target model.

Next run the attack. **Model inversion** (Fredrikson et al. 2014) scores each candidate sensitive value $s$ using two quantities. The first, $p(s \mid x)$, is the estimated probability of $s$ given the target's public features $x$. The second, $f_y(x, s)$, is the model's confidence in the known label $y$ for input $(x,s)$. The attack chooses the value that maximizes their product:
$$\hat s = \arg\max_s\; p(s \mid x)\, f_y(x, s).$$ {#eq-attr-map-inversion}

Give the attacker every column it could know. A near-copy of the sensitive field, such as relationship status alongside marital status, can raise the baseline. Run the attack on training members and as many held-out non-members as possible. Report the baseline, both success rates and their [confidence intervals](../part01-prerequisites/04-hypothesis-testing.md). Also report an interval for the advantage; [Pre-release privacy risk assessment](10-risk-assessment.md#sec-risk-threat) blocks a release when that interval lies above zero.

The script [practitioner_attribute.py](../../code/part02-attacks/practitioner_attribute.py), run with `uv run python code/part02-attacks/practitioner_attribute.py`, applies the Adversarial Robustness Toolbox (ART) to Adult, a census dataset with 45,222 records. A random forest predicts whether income exceeds \$50K using all columns; sex is the sensitive field. The attacker has 4,000 auxiliary records and knows each target's other columns and income.

```{.python excerpt="attribute-art"}
def label(rows: IntArray) -> Array:  # the label the model returns for each record, as a column
    return np.argmax(target.predict(x[rows]), axis=1).reshape(-1, 1).astype(np.float64)

np.random.seed(seed)  # ART builds its random forests unseeded, from NumPy's global generator
attack = AttributeInferenceBlackBox(target, attack_model_type="rf", attack_feature=SENSITIVE)
attack.fit(x[attacker])  # the attacker's own records, sensitive field included
baseline = AttributeInferenceBaselineTrueLabel(attack_model_type="rf", attack_feature=SENSITIVE)
baseline.fit(x[attacker], y[attacker])  # the same records and the true label, without the model
hits: dict[str, list[int]] = {}
for name, rows in (("members", members), ("non-members", others)):
    known = np.delete(x[rows], SENSITIVE, axis=1)  # everything but the sensitive field
    guess = attack.infer(known, pred=label(rows), values=[0.0, 1.0])
    imputed = baseline.infer(known, y=y[rows], values=[0.0, 1.0])
    hits[name] = [int(np.count_nonzero(np.equal(g.ravel(), female[rows]))) for g in (guess, imputed)]
```

```text
ART black-box attack: right on members 0.8300 [0.8180, 0.8415]  on non-members 0.8423 [0.8306, 0.8534]
ART black-box attack: advantage -0.0123 [-0.0285, +0.0040]
imputation baseline: right on members 0.8307 [0.8188, 0.8423]  on non-members 0.8417 [0.8301, 0.8529]
```

The attack is correct for 0.8300 of members; the baseline is correct for 0.8307. The result is consistent with population inference, and the advantage interval includes zero, so it passes the rule. On a synthetic population in the full edition, inversion is correct for 0.7832 of members and 0.7262 of non-members, for an advantage of 0.0569.

## What goes wrong

The most common reporting error is to give only the member success rate, such as 0.7832, without the baseline and non-member rate. This overstates leakage. Reporting only the rate on fresh records understates it because models are more confident on members ([Membership inference](02-membership-inference.md)).

A model without the sensitive column may still support population inference through correlated fields. **Confidence masking**, which rounds confidence scores or returns only the top label, does not bound the gap between members and non-members; it can increase that gap. In the full edition's experiment, returning only labels raised advantage from 0.0569 to 0.0650 because success fell more on non-members. Reducing overfitting can narrow the gap. [Differential privacy](../part03-dp-foundations/01-definition.md) bounds it for every attacker and output format.

## References

- Fredrikson, Jha & Ristenpart, "Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures" (CCS 2015): inversion from confidences, and rounding as a countermeasure.
- Jayaraman & Evans, "Are Attribute Inference Attacks Just Imputation?" (CCS 2022): the imputation baseline, and what it takes to beat it.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/03-attribute-inference.md).
