# Attribute inference and model inversion

## What it is

An **attribute inference** attack uses a trained model to recover one sensitive field of a person's record, such as smoking status, from the fields the attacker already knows. Take a hospital model that predicts a lung disease from age and smoking. An insurer who knows your age and your diagnosis, the model's label, can query it to guess whether you smoke. But a published study on smoking would allow the same guess, so success alone is not a privacy failure. What matters is whether the model reveals more about you because your record was in its training data.

Yeom et al. (2018) turned this question into a game ([Threat models and privacy games](01-threat-models.md)). A fair coin decides whether the target is a training record, a **member**, or a fresh draw from the same population. The attacker, who knows the target's public features and label, queries the model and guesses the sensitive value. Its score is the **attribute advantage**,
$$\mathrm{Adv}^{\mathrm{attr}} = \Pr[\text{guess right} \mid \text{member}] - \Pr[\text{guess right} \mid \text{non-member}].$$
Members come from the same population as everyone else, so a guess that rests only on the whole population, called **population inference**, helps both equally and cancels in the difference. What is left is the leak about individuals.

## When you meet it

You meet it when someone claims that "the model reveals" a sensitive field, or asks whether dropping a column made a model safe. The Adversarial Robustness Toolbox, a Python library of attacks, offers `AttributeInferenceBlackBox`. The attribute advantage decides how to classify a finding. Success up to what the population allows would be there without the person's record, so it is a question for policy and fairness. But success on members beyond non-members is a privacy leak.

## How to use it

Start with the **imputation baseline**, the success of the best guess of the sensitive value from the public features and label alone, without the model. Jayaraman & Evans (2022) found that many published attacks do no better, so it is the number to beat. To estimate it, take records from the same population that the model never saw. On half of them, fit a classifier such as scikit-learn's `HistGradientBoostingClassifier` to predict the sensitive field from the public features and label, and measure its accuracy on the other half.

Then run the attack. **Model inversion** (Fredrikson et al. 2014) scores each candidate value $s$ of the sensitive field by two numbers. The first is how likely a person with the target's public features $x$ is to have that value, the probability $p(s \mid x)$, which the attacker estimates from population data. The second is $f_y(x, s)$, the model's confidence in the known label $y$ when it is fed $x$ and the candidate. The maximum a posteriori (MAP) inversion attack then guesses
$$\hat s = \arg\max_s\; p(s \mid x)\, f_y(x, s).$$ {#eq-attr-map-inversion}
Against a scikit-learn model, each candidate costs one call to `predict_proba`, from which you read the column of the known label.

Give the attacker every column it could hold, because a near-copy of the sensitive field, such as a relationship column next to marital status, moves the baseline most. Run the attack on training members and on an equal number of held-out non-members. Report the baseline and both success rates, each with a confidence interval ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). Report their difference, the advantage, with its own interval too. [Pre-release privacy risk assessment](10-risk-assessment.md#sec-risk-threat) blocks a release when the advantage's interval lies above zero.

The full edition measures them on a synthetic population, averaging 200 small logistic regressions trained on 100 records each. There, the best population guess is right 0.7517 of the time. Inversion is right on 0.7832 of members, but also on 0.7262 of non-members. So the attribute advantage is only 0.0569, and most of the attack's success on members would be there even if they had never been in the training data. On non-members the true baseline is a ceiling, since the model never saw them. An attack whose interval there lies above your baseline's means your imputer is weak.

## What goes wrong

The commonest error is to report one success rate, such as the 0.7832 on members, without the baseline and the rate on non-members, which overstates the leak. Measuring only on fresh records understates it. A model is too sure of the true label on its own members, which is the generalization gap of [Membership inference](02-membership-inference.md), so the attack does better on them.

Dropping the sensitive column from the inputs does not stop population inference, because its correlates remain. **Confidence masking**, which rounds the confidences or returns only the top label, puts no bound on the gap between members and non-members, and that gap can grow. In the same experiment, returning labels only raised the advantage from 0.0569 to 0.0650, because success fell more on non-members than on members. Less overfitting shrinks the member gap, and [differential privacy](../part03-dp-foundations/01-definition.md) bounds it for every attacker and every output format.

## Further reading

- Fredrikson, Jha & Ristenpart, "Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures" (CCS 2015): inversion from confidences, and rounding as a countermeasure.
- Jayaraman & Evans, "Are Attribute Inference Attacks Just Imputation?" (CCS 2022): the imputation baseline, and what it takes to beat it.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/03-attribute-inference.md).
