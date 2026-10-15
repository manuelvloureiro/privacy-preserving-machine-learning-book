# Threat models and privacy games

## Why this matters

A hospital trains a model on patient records and offers it to doctors through a web service. Is the model "private"? The question has no answer until you say **against whom**: an attacker who reads labels or one who holds the weights, one who knows the other patients or only the population, one who follows the rules or one who poisons the training data, one who wants to learn that a person was a patient or to rebuild the whole table. A **threat model** answers these questions. Without one, "the model leaks" and "the model is safe" are both untestable, and an attack number cannot be compared with anything.

Once the threat model is fixed, an attack is a **game**: a **challenger** runs the pipeline with a hidden **secret**, such as whether one patient's record was used, hands the **adversary** what the threat model lets it see, its **view**, and the adversary guesses the secret. The one idea of this chapter: **an attack is a guess about a hidden secret, made from a view the threat model fixes, and it is only as good as its margin over a guess made without the view.** When the secret is one bit, the game is a hypothesis test: its power has a ceiling you can compute, and what matters is the low false positive end of its error curve. @tbl-threat-uses maps the decisions to the sections.

| Decision | Rule | Section |
| --- | --- | --- |
| What an attack assumes | Name its goal, access, behaviour, trust and side information before reading its number | @sec-threat-models |
| Whether an attack beats guessing | Compare its success with the best guess made without the view | @sec-threat-games |
| How to score it | True positive rate (TPR) at a low false positive rate (FPR), and precision at a realistic prior | @sec-threat-advantage |
| How much an average-case number tells you | It is a lower bound on the risk to the worst-placed record, never an upper bound | @sec-threat-average |

: The decisions behind stating and scoring an attack, and where this chapter makes each one. {#tbl-threat-uses}

A two-world game has a hidden bit $b$: $b = 1$ is the world in which the target's record is used (it is a "member") and $b = 0$ the world in which it is not. The attack's **TPR** is the fraction of members it flags as members, and its **FPR** is the fraction of non-members it wrongly flags.

## Threat models {#sec-threat-models}

**Idea.** A threat model answers five questions. The **goal**: membership, a sensitive field of a known record (attribute), the records themselves (reconstruction), or a statistic of the whole dataset (property). The **access**, from least to most: the model's decisions on chosen inputs (**label-only**), its scores such as confidences or losses (**black-box**), its parameters (**white-box**), or everything training exposed, such as per-step gradients (the **transcript**). The **behaviour**: an **honest-but-curious** adversary only analyses what it legitimately sees; a **malicious** one may query adaptively, insert crafted records into the training data (**poisoning**) or tamper with the protocol. The **trust**: in the **central** model a trusted curator holds the data; in the **local** model nobody is trusted. And the **side information**: the population, a public dataset, some or all of the other records, the target's non-sensitive fields.

**Result.** Access is ordered, because each level can compute the one below it (weights give any score, a score gives its label), and processing a view never helps an adversary, who could do the processing itself. So the best attack at each level satisfies
$$\text{best advantage from labels} \le \text{from scores} \le \text{from weights} \le \text{from the transcript},$$
and the same holds for the best TPR at every FPR. More side information and permission to act maliciously likewise never make the best adversary weaker. @tbl-threat-deployments reads four common deployments as threat models.

| Deployment | Access | Behaviour | Trust | Typical side information |
| --- | --- | --- | --- | --- |
| Prediction service that returns the top class | label-only | honest-but-curious user, adaptive queries | central | public data from the same population |
| Service that returns confidences or log-probabilities | black-box | honest-but-curious user, adaptive queries | central | public data; the target's record |
| Model weights published for download | white-box | anyone; can run any computation | central | public data; possibly some training records |
| Federated training with a coordinating server | transcript (per-client updates) | server may be malicious | distributed | the global model at every round |

: Four deployments read as threat models. The same model can sit in several rows at once, and its risk is set by the weakest one. {#tbl-threat-deployments}

**Example.** A security team fits the simplest anomaly detector: the mean $\theta$ of $n = 100$ records with $d = 50$ normal features. A point $x$ gets the score $\|x - \theta\|^2$ and is flagged above $\tau = d$. An adversary holding a target $z$ plays the membership game at each access level (@tbl-threat-access). With the score alone, the **calibrated score** $\|z\|^2 - \|z - \theta\|^2$, which corrects for how far $z$ lies from the origin anyway, does as well as the best white-box attack. The raw score ignores that some targets are far from the mean, member or not, and falls almost to the label-only attack.

| View | Attack | Best advantage $\mathrm{TPR} - \mathrm{FPR}$ |
| --- | --- | --- |
| White-box | likelihood ratio of $\theta$ | 0.2789 |
| Black-box | calibrated score $\|z\|^2 - \|z - \theta\|^2$ | 0.2789 |
| Black-box | raw loss $-\|z - \theta\|^2$ | 0.0407 |
| Label-only | not flagged, $\tau = d$ | 0.0400 |

: Membership advantage against the centroid detector with $n = 100$ and $d = 50$, from 100,000 games per world. {#tbl-threat-access}

The biggest gap is between two attacks on the same black-box view: calibrating each target against its own typicality multiplies the advantage by about seven. With more parameters per record the leak grows, to 0.7407 at $d = 500$ (@fig-threat-access).

![Left: the calibrated black-box score for members and non-members of the centroid detector, $n = 100$, $d = 50$. Right: the best advantage at each access level as the dimension $d$ grows. White-box and calibrated black-box coincide; raw loss and label-only are far below.](figures/threat_access.svg){#fig-threat-access}

**Pitfalls.** The chain compares the *best* attacks at each level, not the ones people run. A weak white-box attack can lose to a well-calibrated black-box one, so "the attack we tried failed" proves little. Adding noise to returned scores, rounding them or returning labels only can lower the best advantage but never raise it, and may not lower it much. Side information is the axis you control least, so a defence should survive the strongest assumption: an adversary who knows every record except the target's, which a malicious adversary approaches by planting records.

## Privacy games and baselines {#sec-threat-games}

**Idea.** A game pins down how the secret and data are drawn, what the adversary sees, and how its guess is scored against a **baseline**, the best guess made from the side information alone. A model that lets an attacker guess a rare diagnosis "with 90% accuracy" reveals nothing if 90% of people do not have it. @tbl-threat-games lists the games of this part.

| Game | Secret | Data in the two worlds | Baseline |
| --- | --- | --- | --- |
| Membership | bit $b$ | $b = 1$: target in the training set; $b = 0$: target fresh from the population | $\frac12$ |
| Attribute | sensitive field of a known member | target in the training set; adversary knows its other fields | best guess from the other fields and the population, or the same attack on non-members |
| Reconstruction | the records (or one column) | data drawn from the population | guessing from the population |
| Distinguishing | bit $b$ | adversary chooses two datasets differing in one record | $\frac12$ |
| Property | bit $b$ | data drawn from two populations that differ in a statistic | $\frac12$ |

: The privacy games of this part. The first three are average-case: data come from the population. The distinguishing game is worst-case: the adversary picks the data. {#tbl-threat-games}

**Result.** With a prior $\pi_1$ that the bit is 1 and $\pi_0 = 1 - \pi_1$, an attack wins with probability $\pi_1\,\mathrm{TPR} + \pi_0(1 - \mathrm{FPR})$. At a fair prior this is $\frac12 + \frac12\mathrm{Adv}$, where the **advantage**
$$\mathrm{Adv} = \mathrm{TPR} - \mathrm{FPR}$$
is how much more often the attack says "member" to members than to non-members. An attack that ignores its view scores 0 at any prior, a perfect attack 1. When the bit is rare, the best attack keeps the default guess "0" and overturns it only on views that are more than $\pi_0/\pi_1$ times as likely when the bit is 1 than when it is 0: 9 times as likely at $\pi_1 = 0.1$. Its gain over the default is the probability of those strongly convincing views.

**Example.** An attack with TPR 0.6 and FPR 0.2 wins 0.7 of games at a fair prior and 0.78 at $\pi_1 = 0.1$. The second number looks better but is worse: always saying 0 wins 0.9 there. The advantage, 0.4, is the same at both priors. A team publishes a count of people with a sensitive condition, with unit Gaussian noise, and the adversary knows every other record. At a fair prior the best guess of the target's bit is right with probability 0.6915. If the condition has prior 0.1, the baseline is 0.9 and the best guess reaches only 0.9013: the release rarely produces evidence strong enough to overturn 9-to-1 odds.

**Pitfalls.** Never report accuracy without the baseline, and prefer the advantage, which does not depend on the prior. Attribute inference has two baselines: comparing with a guess from the population asks "does the model help?", comparing members with non-members asks "does it help *because this person was in the data*?". Only the second is a breach of that person's privacy ([Attribute inference and model inversion](03-attribute-inference.md)).

## Advantage, total variation and TPR at a low FPR {#sec-threat-advantage}

**Idea.** A two-world game is a test of "not a member" against "member", so the results of [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) become ceilings on every attack, including ones nobody has invented. Write $P$ and $Q$ for the distributions of the adversary's view without and with the target. The best attack flags views in decreasing order of the likelihood ratio $q/p$, the views most typical of members relative to non-members, until it has spent its false-positive budget.

**Result.** For every attack,
$$\mathrm{Adv} \le \mathrm{TV}(P, Q), \qquad \mathrm{TPR} \le 1 - T(P, Q)(\mathrm{FPR}),$$
where the **total variation (TV)** distance $\mathrm{TV}(P, Q)$ is the largest difference in probability that $P$ and $Q$ give any set of views, and the **trade-off function** $T(P, Q)(\alpha)$ is the smallest fraction of members any attack with FPR $\alpha$ must miss. The first bound caps one point of the attack's receiver operating characteristic (ROC) curve, the plot of TPR against FPR as the threshold moves; the second caps the whole curve. The question a flagged person cares about is the **precision**, the fraction of "member" verdicts that are right:
$$\Pr[\text{member} \mid \text{flagged}] = \frac{\pi_1\,\mathrm{TPR}}{\pi_1\,\mathrm{TPR} + \pi_0\,\mathrm{FPR}}.$$

**Example.** Take the noisy count again, the other 100 bits known. The best attack says "member" when the count exceeds the point where the two bell curves cross, with advantage $\mathrm{TV} = 0.3829$; simulated over 200,000 games per world it scores 0.3809. Now hold the FPR at $10^{-3}$. The threshold rises until one non-member in a thousand passes, and the TPR falls to 0.0183. The attack catches fewer than 2% of members, but when it fires it is right with precision 0.948 at a fair prior. If one target in a hundred is a member, the precision drops to 0.156. @sec-threat-code computes these numbers.

![Left: the released count in the two worlds when the other 100 records are known (solid) and unknown (dashed). Right: the membership ROC curves of the best attacks on log axes. The dotted diagonal is random guessing.](figures/threat_tradeoff.svg){#fig-threat-tradeoff}

**Pitfalls.** The advantage is only the point of the ROC curve where TPR − FPR is largest, and says nothing about the low-FPR end, where privacy breaks. An attack that flags 1% of members with no false alarms has advantage about 0.01, yet identifies those members with certainty; one right on 55% of members and wrong on 45% of non-members has advantage 0.1 and identifies nobody. Report the TPR at a low FPR, such as $10^{-3}$, with the ROC curve on log axes (Carlini et al. 2022), and state the prior when you quote a precision.

## Average case and worst case {#sec-threat-average}

**Idea.** An **average-case** game draws the data and the target from the population, and the adversary knows only the population. A **worst-case** game lets the adversary know, or choose, the other records and the target. Not knowing the other records works like extra noise that blurs the two worlds together. Evaluations usually measure the average case; guarantees have to control the worst case.

**Result.** The best advantage is smallest when the adversary does not know the other records, larger when it knows them, and largest when it chooses them. Both gaps can be extreme: the average-case advantage can be close to 0 while the worst case is 1.

**Example.** On the noisy count with 100 other records (@tbl-threat-avg-worst), not knowing the crowd cuts the advantage by a factor of five, and turns an exact count from a complete giveaway into a small leak.

| Adversary | Best advantage (exact) | Simulated (200,000 games per world) |
| --- | --- | --- |
| Knows the other records ($\sigma = 1$) | 0.3829 | 0.3809 |
| Does not know them, target's bit 1 | 0.0779 | 0.0751 |
| Does not know them, target's bit random | 0.0390 | 0.0398 |
| Knows the other records, exact count ($\sigma = 0$) | 1 | — |
| Does not know them, exact count ($\sigma = 0$) | 0.0796 | — |

: Membership advantage on a noisy count with 100 other records, for adversaries with different knowledge. {#tbl-threat-avg-worst}

A sharper case: a mechanism publishes one record chosen at random among $n = 1000$. Averaged over targets, the advantage is $1/n = 0.001$, yet every run exposes one person completely.

![Best membership advantage on the noisy count against the number $m$ of other records the adversary does not know, for three noise levels (solid), and against an adversary who knows them (dashed). The average case falls as the crowd grows; the worst case does not move.](figures/threat_average_worst.svg){#fig-threat-average-worst}

**Pitfalls.** An audit that attacks random training and held-out records and reports a small average advantage has shown only that the worst-case risk is at least that large; a few outliers can be exposed far more. A defence that relies on the adversary not knowing the crowd fails against one that knows it or planted it. Audits narrow the gap by playing the adversary who chooses the target: they plant **canaries**, records built to be as exposed as possible, and attack those ([Memorization and training-data extraction](05-memorization-extraction.md)). The guarantee of [The definition of differential privacy](../part03-dp-foundations/01-definition.md) bounds the worst case for every pair of datasets that differ in one record, and therefore every membership-style game at once.

## Code {#sec-threat-code}

When the attack score is a unit normal for non-members and shifted for members, the best TPR at any FPR and the precision take a few lines:

```python
from statistics import NormalDist

Z = NormalDist()


def tpr_at_fpr(fpr: float, shift: float) -> float:
    """Best TPR at a given FPR when the attack score is N(0, 1) for non-members, N(shift, 1) for members."""
    return 1 - Z.cdf(Z.inv_cdf(1 - fpr) - shift)


def precision(tpr: float, fpr: float, prior: float) -> float:
    """Fraction of 'member' verdicts that are right when a fraction `prior` of targets are members."""
    return prior * tpr / (prior * tpr + (1 - prior) * fpr)


advantage = 2 * Z.cdf(0.5) - 1  # best TPR - FPR for a shift of 1: the total variation distance
tpr = tpr_at_fpr(1e-3, 1.0)
print(f"{advantage:.4f}")  # 0.3829
print(f"{tpr:.4f}")  # 0.0183
print(f"{precision(tpr, 1e-3, 0.5):.3f}")  # 0.948
print(f"{precision(tpr, 1e-3, 0.01):.3f}")  # 0.156
```

The full edition's script, [threat_models.py](../../code/part02-attacks/threat_models.py), computes every number in this chapter and checks it by simulation:

```bash
uv run python code/part02-attacks/threat_models.py
```

## Cheat sheet

| Result | Statement | Notes |
| --- | --- | --- |
| Winning probability | $\pi_1\mathrm{TPR} + \pi_0(1 - \mathrm{FPR})$; fair prior: $\frac12 + \frac12\mathrm{Adv}$ | Compare with the baseline $\max(\pi_0, \pi_1)$ |
| Advantage | $\mathrm{Adv} = \mathrm{TPR} - \mathrm{FPR}$ | Independent of the prior; 0 for an attack that ignores its view |
| Access chain | labels $\le$ scores $\le$ weights $\le$ transcript, for the best attack | A better attack on the same view can matter more than more access |
| Ceilings | $\mathrm{Adv} \le \mathrm{TV}(P, Q)$; $\mathrm{TPR} \le 1 - T(P, Q)(\mathrm{FPR})$ | Bound every attack, including future ones |
| Precision | $\pi_1\mathrm{TPR}/(\pi_1\mathrm{TPR} + \pi_0\mathrm{FPR})$ | Falls fast when members are rare |
| Average and worst case | not knowing the others $\le$ knowing them $\le$ choosing them | An average-case number is a lower bound |

: The results of this chapter. {#tbl-threat-cheat}

**Rules of thumb**

- Read an attack's threat model before its number.
- Compare every attack with its baseline. Accuracy near the prior is no attack.
- Report the TPR at a low FPR, with the ROC curve on log axes.
- Treat an average-case audit as a lower bound on the worst case.
- Calibrate scores per target before asking for more access.

## Further reading

- Salem et al., "SoK: Let the Privacy Games Begin! A Unified Treatment of Data Inference Privacy in Machine Learning" (IEEE S&P 2023): every inference attack written as a game, with the relations between them.
- Carlini, Chien, Nasr, Song, Terzis & Tramèr, "Membership Inference Attacks From First Principles" (IEEE S&P 2022): why attacks should be reported by TPR at a low FPR on log-scaled ROC curves.
- Jayaraman, Wang, Knipmeyer, Gu & Evans, "Revisiting Membership Inference Under Realistic Assumptions" (PETS 2021): rare members, priors, precision and threshold choice.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/01-threat-models.md).
