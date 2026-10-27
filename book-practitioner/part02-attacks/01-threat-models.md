# Threat models and privacy games

## Why this matters

A hospital trains a model on patient records and offers it to doctors through a web service. Whether the model is private has no answer until you say against whom. One attacker reads only predicted labels, while another holds the weights, knows the other patients or slips crafted records into the training data. A **threat model** pins these assumptions down. Once it is fixed, an attack becomes a **game**, in which the adversary guesses a hidden **secret**, such as whether one patient's record was used.

## Writing down the threat model {#sec-threat-models}

Write the threat model before anyone runs an attack, because it decides which attacks to run and what result stops the release. In the worksheet, @tbl-threat-worksheet, the first six rows are the questions every threat model answers, and the last two turn the answers into a test.

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

The access row has four levels, each enough to compute the one below it. The weakest is label-only, the model's decisions on inputs the adversary chooses. Next come black-box scores, from which the labels follow, and then white-box access to the weights, from which the scores follow. The strongest is the **transcript**, everything training exposed, such as per-step gradients, whose last entry is the weights. So in this ordering, the **access chain**, more access never weakens the best attack. That compares the *best* attacks, not the ones someone tried, so "the attack we tried failed" proves little ([Membership inference](02-membership-inference.md)).

To fill the access row, read the serving code and the release plan. A service that returns probability vectors gives black-box access, and a checkpoint shared with a partner gives it white-box access. A model in several deployments carries the risk of the most exposed one.

Behaviour and trust decide who the adversary is. An **honest-but-curious** adversary follows the protocol and analyses only what it sees. A **malicious** one may insert crafted records into the training data, called **poisoning**, or send tampered models if it runs part of the system. If whoever collects the data may be an adversary, protection must happen before the data leave the user ([Local, shuffle and distributed trust models](../part03-dp-foundations/09-trust-models.md)).

Side information is the row you control least, because you cannot see what else the adversary holds. So assume it knows every record except the target's. The unit row decides what one guarantee covers: a per-row guarantee covers each of a user's 200 rows, but not all 200 together.

For the hospital's classifier, served through an API that returns class probabilities, the goal is membership and the access is black-box scores. A doctor using it is honest-but-curious but may query adaptively, the side information is a public sample of the same population, and the unit is the patient. The release is blocked if the lower end of the TPR's interval at an FPR of $10^{-3}$ exceeds ten times that FPR. Fix that threshold before the attack runs, because one chosen after seeing the results can always be set where they pass.

## Scoring an attack against its baseline {#sec-threat-scoring}

In the membership game, a challenger flips a coin to decide whether to train the model with or without the target's record. The adversary then guesses whether the target is a **member**, that is, whether its record was used.

The attack's **true positive rate** is the fraction of members it flags, and its **false positive rate** is the fraction of non-members it wrongly flags. Write $\pi_1$ for the chance that the coin makes the target a member, the prior, and $\pi_0 = 1 - \pi_1$ for the opposite. The attack wins with probability
$$\Pr[\text{win}] = \pi_1\,\mathrm{TPR} + \pi_0\,(1 - \mathrm{FPR}).$$
The two rates do not depend on the prior, but this win rate, the accuracy, does. If only one target in ten is a member, always saying "not a member" has both rates at 0 and wins $\pi_0 = 90\%$ of the games. So score every attack against a **baseline**, the best score of a guess that never looks at the model.

With a fair prior, $\pi_1 = \pi_0 = 1/2$, the baseline is one half. The attack's **advantage** is how far its TPR exceeds its FPR, and putting the fair prior into the win rate gives
$$\Pr[\text{win}] = \frac12 + \frac12\,(\mathrm{TPR} - \mathrm{FPR}).$$
So an attack with 55% accuracy at a fair prior has an advantage of 0.1, and unlike the accuracy, the advantage does not change with the prior. No attack's advantage exceeds the total variation distance between the adversary's views with and without the target ([Divergences](../part01-prerequisites/03-divergences.md)), which differential privacy later bounds.

## Reporting: TPR at a low FPR {#sec-threat-advantage}

An attack scores each target and flags those above a threshold. Its **receiver operating characteristic (ROC)** curve plots its TPR against its FPR as that threshold moves. The advantage is the curve's largest height above the diagonal, where TPR equals FPR. Like the area under the curve, it hides the region where false positives are rare, which is where privacy breaks. An attack that flags 1% of members with no false alarms has a tiny advantage, yet identifies those members with certainty. So report membership attacks, as Carlini et al. (2022) do, by their TPR at a low FPR such as $10^{-3}$.

A flagged person cares about a third number, the chance that the verdict is right. That is the **precision**,
$$\Pr[\text{member} \mid \text{flagged}] = \frac{\pi_1\,\mathrm{TPR}}{\pi_1\,\mathrm{TPR} + \pi_0\,\mathrm{FPR}}.$$
When members are rare, $\pi_0$ is large, so even a small FPR applied to the many non-members swamps the true positives. Take a count of people with a condition, released with Gaussian noise of standard deviation 1, so that a member adds 1 to the count. An adversary who knows every other record flags the target when the count is high. Held to an FPR of $10^{-3}$, it catches members with probability 0.0183. At a fair prior the two halves of the prior cancel, so its precision is $0.0183/(0.0183 + 0.001) = 0.948$. If only one target in a hundred is a member, it drops to 0.156, so never quote a precision without its prior.

Each rate is measured on a finite set of targets, so it needs an interval. The recipe of [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md#sec-ht-recipe) plans about $100/\alpha$ non-members for an FPR of $\alpha$, so that about 100 false positives pin the rate down. It sets the threshold on one split, measures on another, and puts a Clopper–Pearson interval on each rate. The snippet below applies it to the black-box membership attack of the Adversarial Robustness Toolbox (ART), against a random forest trained on half of scikit-learn's 569 breast cancer records. The attacker trains its attack on every other record, whose membership it knows, and reads the TPR on the rest at an FPR of at most 0.01. To stay short, it sets the threshold on those same scores.

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

With only 151 non-members, the smallest nonzero FPR the attack can show is one in 151, far above one in a thousand. Even one false positive leaves the FPR's interval running from 0.0002 to 0.0363. The TPR's interval, from 0.0047 to 0.0645, overlaps it. So this test cannot show that the attack beats chance at a low FPR, and its area under the ROC curve, 0.552, adds little. At $10^{-3}$ the recipe asks for about 100,000 non-members, which is why audits retrain many models on many records ([Membership inference](02-membership-inference.md)).

A report you can act on therefore gives the TPR at a stated low FPR, with its counts, intervals on both rates and the ROC curve on log axes. It also states the worksheet's threat model, the baseline, the prior behind any precision, and whether the targets were random records or chosen worst cases.

## Average case and worst case {#sec-threat-average}

An evaluation that draws the data and the target from the population measures the **average case**. One that lets the adversary know or pick the other records and the target measures the **worst case**. Audits of real models measure the first, but a guarantee must control the second, and the gap can be large. In the noisy count, the adversary who knows the other records has an advantage of 0.3829. If instead 100 of those records are unknown to it, each a fair coin, their sum blurs the count like extra noise, and the advantage falls to 0.0779. But an adversary who planted the other records knows them, so a defence that relies on the crowd fails against the adversary it most needs to stop.

Averaging over targets hides the worst-treated person in the same way, because exposing a few outliers barely moves an average over everyone. So treat an average-case number as a lower bound on the risk. Audits close the gap by planting **canaries**, records built to be as exposed as possible ([Memorization and training-data extraction](05-memorization-extraction.md)), and [differential privacy](../part03-dp-foundations/01-definition.md) bounds the worst case directly.

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
