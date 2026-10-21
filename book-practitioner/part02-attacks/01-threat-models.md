# Threat models and privacy games

## Why this matters

A hospital trains a model on patient records and offers it to doctors through a web service. Whether the model is private has no answer until you say **against whom**. One attacker reads only predicted labels, while another holds the weights, knows the other patients or slips crafted records into the training data. A **threat model** pins these assumptions down. Without one, "the model leaks" is a claim nobody can test.

Once the threat model is fixed, an attack becomes a **game**. The adversary guesses a hidden **secret**, such as whether one patient's record was used, from what the threat model lets it see. This chapter gives you a worksheet for the threat model, the baseline every attack must beat, and the way to report a result. Behind all three is one idea: an attack counts only by how much it beats a guess made without the model.

## Writing down the threat model {#sec-threat-models}

Before you run or read an attack, answer five questions.

1. **Goal: what is the secret?** A membership attack asks whether a record was in the training data, and an attribute attack seeks a sensitive field of a known record. A reconstruction attack seeks the records themselves, and a property attack a statistic of the whole dataset.
2. **Access: what does the adversary observe?** The weakest is label-only access, the model's decisions on inputs the adversary chooses. Next is black-box access to scores such as confidences or losses, then white-box access to the weights. The strongest is the **transcript**, everything training exposed along the way, such as per-step gradients.
3. **Behaviour: does it follow the rules?** An honest-but-curious adversary only analyses what it legitimately sees. A **malicious** one may choose its queries adaptively or insert crafted records into the training data, which is called **poisoning**.
4. **Trust: who sees the raw data?** In the **central** model a trusted curator holds the data and the adversary sees only the release. In the **local** model nobody is trusted, so each person's data are protected on their own device before collection.
5. **Side information: what does it know already?** It may know the population, a public dataset, some or all of the other records, or the target's non-sensitive fields.

For example, a service that returns only the top class gives label-only access, while published weights give white-box access to anyone. A federated-learning server holds the transcript of per-client updates, and it may be malicious. One model can sit in several such deployments at once, and its risk is set by the most exposed one.

Access is ordered, because each level can compute the one below it. A label is a function of the scores, and the scores are a function of the weights. So for the same model and side information, the best attack can only get stronger along the chain
$$\text{labels} \;\le\; \text{scores} \;\le\; \text{weights} \;\le\; \text{transcript}.$$
The same holds for side information and behaviour, since an adversary can always forget what it knows or choose not to deviate. It also means that rounding or noising an output can never make the best attack stronger, though it may not weaken it much either.

The chain compares the *best* attack at each level, and the attacks people run are often far from it. Against a simple anomaly detector, calibrating each target's score against how typical the target is raised a black-box attack's advantage from 0.0407 to 0.2789, the white-box optimum. So "the attack we tried failed" proves little. [Membership inference](02-membership-inference.md) builds that calibration into a practical attack.

Side information is the axis you control least, so assume an adversary who knows every record except the target's unless you can argue otherwise. Trust decides who the adversary is. If the curator is inside the threat model, protection must happen before the data leave the user ([Local, shuffle and distributed trust models](../part03-dp-foundations/09-trust-models.md)).

## Scoring an attack against its baseline {#sec-threat-scoring}

State every attack in the same four parts: the challenger, the secret, the adversary's view and its guess. The simplest games ask the adversary to guess one hidden bit. In the membership game, the bit says whether the target's record is in the training set, which makes the target a **member**.

Two rates describe an attack. Its **true positive rate (TPR)** is the fraction of members it flags, and its **false positive rate (FPR)** is the fraction of non-members it wrongly flags. Neither rate depends on how often members occur, but the win rate does, which is why it misleads. Write $\pi_1$ for the probability that the target is a member and $\pi_0 = 1 - \pi_1$ for the opposite. The attack then wins with probability
$$\Pr[\text{win}] = \pi_1\,\mathrm{TPR} + \pi_0\,(1 - \mathrm{FPR}).$$
If only one target in ten is a member, always saying "not a member" wins 90% of the games. So an attack that is right "90% of the time" may reveal nothing. So score every attack against a **baseline**, the best score of a guess that never looks at the model.

With a fair prior, half members and half non-members, the baseline is one half. Write the attack's **advantage** as $\mathrm{Adv} = \mathrm{TPR} - \mathrm{FPR}$. The win rate is then
$$\Pr[\text{win}] = \frac12 + \frac12\,\mathrm{Adv}.$$
So an attack with 55% accuracy at a fair prior has advantage 0.1. Whether 55% is a problem depends on where those wins come from, which the next section shows how to read. Because both rates are measured inside one world, the advantage does not change with the prior. No attack's advantage exceeds the total variation distance between the adversary's views in the two worlds ([Divergences](../part01-prerequisites/03-divergences.md)), which differential privacy later bounds.

## Reporting: TPR at a low FPR {#sec-threat-advantage}

The advantage is the best height of the attack's **receiver operating characteristic (ROC)** curve above the diagonal. It says nothing about the region where false positives are rare, which is where privacy breaks. An attack that flags 1% of members with no false alarms has a tiny advantage, but it identifies those members with certainty. But the 55%-accurate attack of the last section may flag members and non-members at similar rates, and then it identifies nobody with confidence. So Carlini et al. (2022) report membership attacks by their TPR at a low FPR such as $10^{-3}$, or lower, on log axes. This part does the same. The full ROC ceiling of the two worlds is their trade-off curve, which [Gaussian differential privacy](../part04-privacy-accounting/04-gaussian-dp.md) turns into a guarantee.

A flagged person cares about a third number, the chance that the verdict is right. That is the **precision**,
$$\Pr[\text{member} \mid \text{flagged}] = \frac{\pi_1\,\mathrm{TPR}}{\pi_1\,\mathrm{TPR} + \pi_0\,\mathrm{FPR}}.$$
When members are rare, even a small FPR applied to the many non-members swamps the true positives (Jayaraman et al., 2021). So a precision is meaningless without the prior it assumes.

Take a count of people with a condition, released with Gaussian noise of standard deviation 1, and an adversary who knows the other records. The count is centred on 50 without the target and on 51 with it. The best attack thresholds the count, and its advantage is 0.3829. Held to an FPR of $10^{-3}$, it catches members with probability 0.0183. That is fewer than 2% of members, but when it fires at a fair prior its precision is 0.948. If only one target in a hundred is a member, the precision drops to 0.156.

A report you can act on states the following.

- The threat model, from the worksheet above, and the game.
- The TPR at a stated low FPR, $10^{-3}$ or lower, and the ROC curve on log axes.
- The number of non-members the FPR rests on, and a confidence interval for both rates, such as a Clopper–Pearson interval ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)).
- The baseline, and the prior behind any precision.
- Whether targets were random records or chosen worst cases.

To compute these, scikit-learn's `roc_curve` returns the FPR and TPR at every threshold from labels and scores, and `auc` gives the area under that curve. Then `numpy.interp` reads off the TPR at your target FPR. `scipy.stats.binomtest(k, n).proportion_ci(method="exact")` gives the Clopper–Pearson interval for a count. An FPR of $10^{-3}$ rests on few false positives unless you have far more than 1,000 non-members. To see a given number of false positives at an FPR $\alpha$, you need about that number divided by $\alpha$.

## Average case and worst case {#sec-threat-average}

An evaluation can draw the data and the target from the population, which gives the **average case**. Or it can let the adversary know or pick the other records and the target, which gives the **worst case**. Audits of real models measure the first, but a guarantee must control the second. The best advantages are ordered:
$$\text{does not know the others} \;\le\; \text{knows them} \;\le\; \text{chooses them}.$$
Not knowing the other records acts like extra noise that blurs the two worlds.

The gap can be large. Suppose the noisy count has 100 other records. The adversary who knows them reaches an advantage of 0.3829, and one who does not reaches only 0.0779. But a malicious adversary who planted the other records knows them, so a defence that relies on the crowd fails against exactly the adversary it most needs to stop.

Averaging over targets hides the worst-treated person in the same way. A mechanism that publishes one record chosen at random from 1,000 has an average-case advantage of only 0.001, yet every run exposes one person completely.

So treat an average-case number as a lower bound on the risk, never as an upper bound. Audits narrow the gap by planting **canaries**, records built to be as exposed as possible ([Memorization and training-data extraction](05-memorization-extraction.md)). [The definition of differential privacy](../part03-dp-foundations/01-definition.md) bounds the worst case directly, for every pair of datasets that differ in one record.

## Code {#sec-threat-code}

The snippet computes the best TPR at any FPR, and the precision, when the attack's score is standard normal for non-members and shifted by 1 for members, as in the noisy count.

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

## Cheat sheet

| Quantity | Formula | Use it for |
| --- | --- | --- |
| Advantage | $\mathrm{TPR} - \mathrm{FPR}$ | Comparing attacks at any prior; 0 means no attack |
| Precision | $\pi_1\mathrm{TPR}/(\pi_1\mathrm{TPR} + \pi_0\mathrm{FPR})$ | What a flagged person faces; state the prior |
| Access chain | labels $\le$ scores $\le$ weights $\le$ transcript | More access never weakens the best attack |
| Knowledge | not knowing $\le$ knowing $\le$ choosing the other records | An average-case number is a lower bound |
| Noisy count, rest known | $2\Phi\big(\frac1{2\sigma}\big) - 1$ | Advantage on one person's bit, noise $\sigma$ |
| Noisy count, $m$ others unknown | $\approx 2\Phi\big(1/(2\sqrt{m/4 + \sigma^2})\big) - 1$ | The unknown others add variance $m/4$ |

: The results of this chapter. $\Phi$ is the standard normal distribution function, `scipy.stats.norm.cdf`. {#tbl-threat-cheat}

**Rules of thumb**

- Fill in the five-question worksheet before running or reading any attack. Two audits with implicit threat models cannot be compared.
- Compare every attack with its baseline. Accuracy near the prior is no attack, and accuracy or the area under the curve alone says nothing.
- A balanced test set hides the precision at a realistic prior, so state the prior with any precision.
- Check what really leaves an "API only" system, because logits, embeddings and shared checkpoints move it up the access chain.
- A better attack on the same view can matter more than more access, so calibrate scores per target.
- Assume the adversary knows the other records, and audit with worst-case canaries.

## Further reading

- Salem et al., "SoK: Let the Privacy Games Begin! A Unified Treatment of Data Inference Privacy in Machine Learning" (IEEE S&P 2023): every inference attack written as a game. Read §§3–5.
- Carlini, Chien, Nasr, Song, Terzis & Tramèr, "Membership Inference Attacks From First Principles" (IEEE S&P 2022): why attacks should be reported by TPR at a low FPR on log-scaled ROC curves.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/01-threat-models.md).
