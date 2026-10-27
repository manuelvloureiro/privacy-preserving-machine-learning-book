# Threat models and privacy games

## Why this matters

A hospital trains a model on patient records and offers it to doctors through a web service. The question "is the model private?" has no answer until you say against whom. One attacker reads only the labels the model returns. Another attacker holds the weights, knows the other patients or adds crafted records to the training data, so the answers differ. A **threat model** writes these assumptions down. With a threat model, an attack becomes a **game**: the adversary guesses a hidden **secret**, such as whether the model used one patient's record.

## Writing down the threat model {#sec-threat-models}

Write the threat model before anyone runs an attack, because it decides which attacks to run and what result stops the release. In the worksheet, @tbl-threat-worksheet, the first six rows are the questions every threat model answers. The last two rows turn the answers into a test.

| Question | What to write down | Default when unsure |
| --- | --- | --- |
| Goal: what is the secret? | Membership, an attribute of a known record, the records themselves, or a property of the whole dataset | Membership, the goal that differential privacy bounds directly |
| Access: what does the adversary see? | Labels only, scores, the weights, or the training transcript | The most anyone outside the training team can get |
| Behaviour: does it follow the rules? | Honest-but-curious, or malicious (poisoning, tampered models) | Malicious, if it runs any part of the system |
| Trust: who sees the raw data? | Central curator, local (data protected on each device before collection), or distributed | Central, only if the data holder is not an adversary |
| Side information: what does it know? | The population, public datasets, other records, the target's other fields | Every record except the target's |
| Unit: what is one record? | A row, a user, a household | The person, whenever one person has several rows |
| Attack and metric | Membership: the strongest attack's true positive rate (TPR) at a low false positive rate (FPR). A table: unique share and smallest class. A language model: canary exposure and extraction | The row of the goal and artefact in [the risk assessment's thresholds](10-risk-assessment.md#sec-risk-threat) |
| Acceptable result | The number that blocks the release | Fixed before any attack is run |

: The threat-model worksheet. {#tbl-threat-worksheet}

The access row has four levels: labels, scores, weights and the **transcript**. The transcript is everything that training exposed, such as the gradients of each step. Each level is enough to compute the one before it, so in this ordering, the **access chain**, more access never weakens the best attack. The chain compares the *best* attacks, not the ones someone tried, so "the attack we tried failed" proves little ([Membership inference](02-membership-inference.md)).

To fill the access row, read the serving code and the release plan. A service that returns probability vectors gives black-box access, and a checkpoint shared with a partner gives white-box access. A model in several deployments carries the risk of the most exposed one. For the behaviour row, an **honest-but-curious** adversary follows the protocol and analyses only what it sees. A **malicious** adversary may add crafted records to the training data, called **poisoning**, or send tampered models. If the party that collects the data may be an adversary, protect the data before they leave the user ([Local, shuffle and distributed trust models](../part03-dp-foundations/09-trust-models.md)).

You cannot see what else the adversary holds, so assume that it knows every record except the target's. The unit row decides what one guarantee covers: a guarantee per row covers each of a user's 200 rows, but not all 200 together.

For the hospital's classifier, the API returns class probabilities, so the goal is membership and the access is black-box scores. A doctor is honest-but-curious but may query adaptively, the side information is a public sample of the same population, and the unit is the patient. The release stops if the lower end of the TPR's interval at an FPR of $10^{-3}$ exceeds ten times that FPR. Fix that threshold before the attack runs, because a threshold chosen after the results can always pass them.

## Scoring an attack against its baseline {#sec-threat-scoring}

In the membership game, a challenger flips a coin to decide whether to train the model with or without the target's record. The adversary then guesses whether the target is a **member**, that is, whether the model used its record.

The **true positive rate** is the fraction of members that the attack flags. The **false positive rate** is the fraction of non-members that it flags by mistake. Write $\pi_1$ for the chance that the coin makes the target a member, the prior, and $\pi_0 = 1 - \pi_1$ for the opposite. The attack wins with probability
$$\Pr[\text{win}] = \pi_1\,\mathrm{TPR} + \pi_0\,(1 - \mathrm{FPR}).$$
The two rates do not depend on the prior, but this win rate, the accuracy, does. If only one target in ten is a member, the guess "not a member" wins $\pi_0 = 90\%$ of the games without looking at the model. So score every attack against a **baseline**, the best score of a guess that never looks at the model.

With a fair prior, $\pi_1 = \pi_0 = 1/2$, the baseline is one half. The attack's **advantage** is how far its TPR exceeds its FPR. With the fair prior in the win rate,
$$\Pr[\text{win}] = \frac12 + \frac12\,(\mathrm{TPR} - \mathrm{FPR}).$$
So an attack with 55% accuracy at a fair prior has an advantage of 0.1. Unlike the accuracy, the advantage does not change with the prior. Differential privacy caps every attack's advantage, through the total variation distance of [Divergences](../part01-prerequisites/03-divergences.md).

## Reporting: TPR at a low FPR {#sec-threat-advantage}

An attack scores each target and flags those above a threshold. Its **receiver operating characteristic (ROC)** curve plots its TPR against its FPR as that threshold moves, and the advantage is the curve's largest height above the diagonal. The advantage and the area under the curve both hide the region where false positives are rare, which is where privacy breaks. For example, an attack that flags 1% of members with no false alarms has a tiny advantage, yet it identifies those members with certainty. So report membership attacks by their TPR at a low FPR such as $10^{-3}$, as Carlini et al. (2022) do.

A flagged person cares about the chance that the verdict is right, the **precision**,
$$\Pr[\text{member} \mid \text{flagged}] = \frac{\pi_1\,\mathrm{TPR}}{\pi_1\,\mathrm{TPR} + \pi_0\,\mathrm{FPR}}.$$
When members are rare, $\pi_0$ is large, so even a small FPR on the many non-members swamps the true positives. For example, take a count of people with a condition, released with Gaussian noise of standard deviation 1. A member adds 1 to the count, so an adversary who knows every other record flags the target when the count is high. At an FPR of $10^{-3}$, it catches members with probability 0.0183. At a fair prior the two halves of the prior cancel, so its precision is $0.0183/(0.0183 + 0.001) = 0.948$. If only one target in a hundred is a member, the precision drops to 0.156, so never quote a precision without its prior.

Each rate comes from a finite set of targets, so it needs an interval. The recipe of [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md#sec-ht-recipe) plans about $100/\alpha$ non-members for an FPR of $\alpha$, so that about 100 false positives fix the rate. It sets the threshold on one split, measures on another, and puts a Clopper–Pearson interval on each rate.

The snippet below applies a short form of the recipe to the black-box membership attack of the Adversarial Robustness Toolbox (ART). The target is a random forest trained on half of scikit-learn's 569 breast cancer records. The attacker trains its attack on every other record, whose membership it knows, and reads the TPR on the rest at an FPR of at most 0.01. It sets the threshold on those same scores.

```python
import numpy as np
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import SklearnClassifier
from scipy.stats import binomtest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve

np.random.seed(0)  # ART's attack model draws from NumPy's global generator
x, y = load_breast_cancer(return_X_y=True)
member = np.random.default_rng(0).random(len(y)) < 0.5  # the target's training records
target = SklearnClassifier(RandomForestClassifier(random_state=0).fit(x[member], y[member]))
known = np.arange(len(y)) % 2 == 0  # records whose membership the attacker knows, to train its attack
attack = MembershipInferenceBlackBox(target, attack_model_type="rf")
attack.fit(x[known & member], y[known & member], x[known & ~member], y[known & ~member])
fpr, tpr, _ = roc_curve(member[~known], attack.infer(x[~known], y[~known], probabilities=True)[:, 0])
i = np.flatnonzero(fpr <= 0.01)[-1]  # the best TPR whose FPR is at most 0.01
n_in, n_out = int(member[~known].sum()), int((~member[~known]).sum())
tp, fp = round(tpr[i] * n_in), round(fpr[i] * n_out)
print(f"AUC={auc(fpr, tpr):.3f}  TP={tp} of {n_in}  FP={fp} of {n_out}")  # AUC=0.552  TP=3 of 133  FP=1 of 151
t, f = (binomtest(k, n).proportion_ci(method="exact") for k, n in ((tp, n_in), (fp, n_out)))
print(f"TPR 95% CI [{t.low:.4f}, {t.high:.4f}]  FPR 95% CI [{f.low:.4f}, {f.high:.4f}]")  # TPR 95% CI [0.0047, 0.0645]  FPR 95% CI [0.0002, 0.0363]
```

With only 151 non-members, the smallest FPR above zero is one in 151, far above one in a thousand. So the FPR's interval runs from 0.0002 to 0.0363, and it overlaps the TPR's interval, from 0.0047 to 0.0645. So this test cannot show that the attack beats chance at a low FPR, and its area under the ROC curve, 0.552, adds little. At $10^{-3}$ the recipe asks for about 100,000 non-members, which is why audits retrain many models on many records ([Membership inference](02-membership-inference.md)).

So report the TPR at a stated low FPR, with its counts, an interval on each rate and the ROC curve on log axes. Also give the worksheet's threat model, the baseline, the prior behind any precision, and whether the targets were random or chosen worst cases.

## Average case and worst case {#sec-threat-average}

An evaluation that draws the data and the target from the population measures the **average case**. If the adversary knows or picks the other records and the target, the evaluation measures the **worst case**. Audits of real models measure the first, a guarantee must control the second, and the gap can be large. In the noisy count, the adversary who knows the other records has an advantage of 0.3829. If 100 of those records are fair coins unknown to the adversary, their sum blurs the count like extra noise, and the advantage falls to 0.0779. But an adversary who planted the other records knows them, so a defence that relies on the crowd fails against the adversary it most needs to stop.

An average over targets also hides the worst-treated person, because a few exposed outliers barely move it. So treat an average-case number as a lower bound on the risk. Audits close the gap with **canaries**, records built to be as exposed as possible ([Memorization and training-data extraction](05-memorization-extraction.md)). [Differential privacy](../part03-dp-foundations/01-definition.md) bounds the worst case directly.

## Cheat sheet

| Quantity | Formula | Use it for |
| --- | --- | --- |
| Advantage | $\mathrm{TPR} - \mathrm{FPR}$ | Comparing attacks at any prior; 0 means no attack |
| Precision | $\pi_1\mathrm{TPR}/(\pi_1\mathrm{TPR} + \pi_0\mathrm{FPR})$ | What a flagged person faces; state the prior |

: The results of this chapter. {#tbl-threat-cheat}

**Rules of thumb**

- Fill in the worksheet, including the blocking threshold, before running or reading any attack.
- Assume the adversary knows the other records, and audit with worst-case canaries.

## References

- Salem et al., "SoK: Let the Privacy Games Begin! A Unified Treatment of Data Inference Privacy in Machine Learning" (IEEE S&P 2023): every inference attack written as a game.
- Carlini, Chien, Nasr, Song, Terzis & Tramèr, "Membership Inference Attacks From First Principles" (IEEE S&P 2022): the case for reporting TPR at a low FPR on log-scaled ROC curves.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/01-threat-models.md).
