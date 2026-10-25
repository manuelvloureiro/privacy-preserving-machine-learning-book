# Threat models and privacy games

## Why this matters

A hospital trains a model on patient records and offers it to doctors through a web service. Whether the model is private has no answer until you say **against whom**. One attacker reads only predicted labels, while another holds the weights, knows the other patients or slips crafted records into the training data. A **threat model** pins these assumptions down. Without one, "the model leaks" is a claim nobody can test or compare.

Once the threat model is fixed, an attack becomes a **game**, in which the adversary guesses a hidden **secret**, such as whether one patient's record was used. [Pre-release privacy risk assessment](10-risk-assessment.md) later fills in this chapter's worksheet for three real releases and runs the attacks it names.

## Writing down the threat model {#sec-threat-models}

Write the threat model before anyone runs an attack, because it decides which attacks to run and what result stops the release. @tbl-threat-worksheet is the worksheet. Its first six rows are the questions every threat model answers, and the last two turn the answers into a test.

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

The access row has four levels. The weakest is label-only, the model's decisions on inputs the adversary chooses. Next come black-box scores, and then white-box access to the weights. The strongest is the **transcript**, everything training exposed along the way, such as per-step gradients. Each level can compute the one below it. A label is a function of the scores, the scores are a function of the weights, and the weights are the transcript's last entry. So in this ordering, the **access chain**, more access never weakens the best attack.

To fill the access row, read the serving code and the release plan. An "API only" service that returns full probability vectors gives black-box access, and a checkpoint shared with a partner gives the partner white-box access. A model in several deployments at once carries the risk of the most exposed one. The access chain compares only the *best* attack at each level, though, not the ones someone tried. In the full edition's experiment, the model is the centroid of 100 training points in 50 dimensions, used as an anomaly detector. A black-box attack on its raw loss has an advantage (@sec-threat-scoring) of 0.0407. Calibrating each target's score against how typical that target is raises it to 0.2789, as high as the best attack with the weights. So "the attack we tried failed" proves little ([Membership inference](02-membership-inference.md)).

Behaviour and trust decide who the adversary is. An **honest-but-curious** adversary follows the protocol and only analyses what it legitimately sees. A **malicious** one deviates from the protocol. It may insert crafted records into the training data, which is called **poisoning**, or, if it runs part of the system, send tampered models to the other participants. If whoever collects the data is itself a possible adversary, protection must happen before the data leave the user ([Local, shuffle and distributed trust models](../part03-dp-foundations/09-trust-models.md)).

Side information is the row you control least. So assume it knows every record except the target's, and list the public datasets that share columns with yours. The unit row matters as much, because a per-row guarantee covers each of a user's 200 rows but not all 200 together.

For example, take the hospital's classifier, served through an API that returns class probabilities. The goal is membership, the access is black-box scores, and a doctor using the service is honest-but-curious but free to query adaptively. The side information is a public sample from the same population, and the unit is the patient. The release is blocked if the lower end of the TPR's interval at an FPR of $10^{-3}$ exceeds ten times that FPR. Write that threshold down before the attack runs, because a threshold chosen after seeing the results can always be set where they pass.

## Scoring an attack against its baseline {#sec-threat-scoring}

In the membership game, the challenger flips a coin to decide whether to train the model with or without the target's record. The adversary then guesses whether the target is a **member**, that is, whether its record was used.

Two rates describe the guesses. The attack's **true positive rate** is the fraction of members it flags, and its **false positive rate** is the fraction of non-members it wrongly flags. Neither rate depends on how often members occur, but the accuracy, the share of games won, does, which is why accuracy misleads. Write $\pi_1$ for the chance that the coin makes the target a member, the prior, and $\pi_0 = 1 - \pi_1$ for the opposite. The attack then wins with probability
$$\Pr[\text{win}] = \pi_1\,\mathrm{TPR} + \pi_0\,(1 - \mathrm{FPR}).$$
If only one target in ten is a member, always saying "not a member" wins 90% of the games. So score every attack against a **baseline**, the best score of a guess that never looks at the model.

With a fair prior, half members and half non-members, the baseline is one half. The attack's **advantage** is how far its TPR exceeds its FPR, and at a fair prior it sets the win rate as
$$\Pr[\text{win}] = \frac12 + \frac12\,(\mathrm{TPR} - \mathrm{FPR}).$$
So an attack with 55% accuracy at a fair prior has an advantage of 0.1. The advantage does not change with the prior. No attack's advantage exceeds the total variation distance between the adversary's views with and without the target ([Divergences](../part01-prerequisites/03-divergences.md)), which differential privacy later bounds.

## Reporting: TPR at a low FPR {#sec-threat-advantage}

An attack's **receiver operating characteristic (ROC)** curve plots its TPR against its FPR as its threshold moves, and the advantage is the curve's largest height above the diagonal. Like the area under the curve, it says nothing about the region where false positives are rare, which is where privacy breaks. An attack that flags 1% of members with no false alarms has a tiny advantage, yet it identifies those members with certainty. So Carlini et al. (2022) report membership attacks by their TPR at a low FPR such as $10^{-3}$.

A flagged person cares about a third number, the chance that the verdict is right. That is the **precision**,
$$\Pr[\text{member} \mid \text{flagged}] = \frac{\pi_1\,\mathrm{TPR}}{\pi_1\,\mathrm{TPR} + \pi_0\,\mathrm{FPR}}.$$
When members are rare, even a small FPR applied to the many non-members swamps the true positives (Jayaraman et al., 2021). Take a count of people with a condition, released with Gaussian noise of standard deviation 1, so that a member adds 1 to the count. An adversary who knows every other record flags the target when the count is high, with advantage 0.3829. Held to an FPR of $10^{-3}$, it catches members with probability 0.0183. Yet when it fires at a fair prior, its precision is $0.0183/(0.0183 + 0.001) = 0.948$. If only one target in a hundred is a member, the precision drops to 0.156, so never quote a precision without its prior.

The rates you report are measured on a finite set of targets, so each needs an interval. [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md#sec-ht-recipe) gives the recipe. It plans about $100/a$ non-members for an FPR of $a$, so that about 100 false positives pin the rate down. It sets the threshold on one split, measures on another, and puts a Clopper–Pearson interval on each rate. The snippet below reads the TPR at an FPR of $10^{-3}$ and both intervals from seeded stand-in scores, with members' scores shifted up by 1 as in the noisy count. To stay short it sets the threshold on the scores it measures, but a real evaluation sets it on a separate split.

```python
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_curve

rng = np.random.default_rng(0)
n = 20_000  # members and non-members each
scores = np.concatenate([rng.normal(1.0, 1.0, n), rng.normal(0.0, 1.0, n)])  # an attack's scores
is_member = np.concatenate([np.ones(n), np.zeros(n)])
fpr, tpr, _ = roc_curve(is_member, scores)
i = np.flatnonzero(fpr <= 1e-3)[-1]  # the best TPR whose FPR is at most 1e-3
true_pos, false_pos = round(tpr[i] * n), round(fpr[i] * n)
print(f"TPR={tpr[i]:.4f} ({true_pos} of {n})  FPR={fpr[i]:.5f} ({false_pos} of {n})")  # TPR=0.0158 (317 of 20000)  FPR=0.00100 (20 of 20000)
tpr_ci = binomtest(true_pos, n).proportion_ci(method="exact")
fpr_ci = binomtest(false_pos, n).proportion_ci(method="exact")
print(f"TPR 95% CI [{tpr_ci.low:.4f}, {tpr_ci.high:.4f}]")  # TPR 95% CI [0.0142, 0.0177]
print(f"FPR 95% CI [{fpr_ci.low:.5f}, {fpr_ci.high:.5f}]")  # FPR 95% CI [0.00061, 0.00154]
```

The snippet's 20,000 non-members are a fifth of what the rule asks for, so its FPR rests on only 20 false positives. Its interval runs from 0.00061 to 0.00154. So the threshold's true FPR may lie below the measured 0.00100, and a lower FPR goes with a lower TPR. That is how the TPR of 0.0158 can sit below the exact 0.0183.

A report you can act on therefore gives the TPR at a stated low FPR, with its counts, intervals on both rates and the ROC curve on log axes. It also states the threat model, the baseline, the prior behind any precision, and whether the targets were random records or chosen worst cases.

## Average case and worst case {#sec-threat-average}

An evaluation can draw the data and the target from the population, which gives the **average case**. Or it can let the adversary know or pick the other records and the target, which gives the **worst case**. Audits of real models measure the first, but a guarantee must control the second, and the gap can be large. In the noisy count, the adversary who knows the other records has an advantage of 0.3829. If 100 of them are unknown fair coins, their sum blurs the count like extra noise, and the advantage falls to 0.0779. But an adversary who planted the other records knows them, so a defence that relies on the crowd fails against the adversary it most needs to stop.

Averaging over targets hides the worst-treated person in the same way, because exposing a few outliers completely barely moves an average over everyone. So treat an average-case number as a lower bound on the risk. Audits narrow the gap by planting **canaries**, records built to be as exposed as possible ([Memorization and training-data extraction](05-memorization-extraction.md)). [The definition of differential privacy](../part03-dp-foundations/01-definition.md) bounds the worst case directly.

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
