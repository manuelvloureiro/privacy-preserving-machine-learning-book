# Membership inference

## Why this matters

A hospital trains a model to predict a disease and publishes it. You hold one patient's record. **Was that record in the training set?** If it was, you have learned that the patient was treated there for that disease. This is **membership inference**, the simplest question an attacker can ask about a model, and the standard way to measure how much a model leaks: if an attacker cannot even tell whether a record was used, it can do little else with it.

An attack is scored by its **true positive rate (TPR)**, the fraction of members it flags, and its **false positive rate (FPR)**, the fraction of non-members it wrongly flags. Its **advantage** is TPR minus FPR ([Threat models and privacy games](01-threat-models.md)).

The one idea: **membership inference is a test about one record.** Train a model many times with a target record and look at its output on that record; train it without the record and look again. Training is random, so each case gives a distribution, the record's **IN** and **OUT** distributions. For a typical record they nearly coincide, because the model predicts it well either way; for an atypical record they are far apart. An attack with one threshold for every record confuses "easy record" with "member"; the strong attacks compare each record with itself. @tbl-mia-uses maps the decisions to the sections.

| Decision | Rule | Section |
| --- | --- | --- |
| Is there a cheap first estimate of the leak? | The loss attack's advantage is the generalization gap | @sec-mia-loss |
| How does an attacker learn what members look like? | Train shadow models on data where membership is known | @sec-mia-shadow |
| Which attack should an audit run? | The likelihood ratio attack (LiRA), a per-record test | @sec-mia-lira |
| How to run it on a real model | The shadow-model protocol, with a global variance if shadows are few | @sec-mia-cifar |
| How to report the result | TPR at FPR $10^{-3}$ on log-scaled receiver operating characteristic (ROC) curves, with the count behind it | @sec-mia-eval |
| Who is at risk, and what does not fix it | Outliers; removing them exposes the next layer | @sec-mia-outliers |

: The decisions behind a membership audit, and where this chapter makes each one. {#tbl-mia-uses}

The numbers come from one experiment, the setup of Carlini et al. (2022) scaled down to a linear model: a pool of 1,000 points with 950 typical points and 50 outliers whose label is a coin flip, regularized logistic regression trained on random halves of it, 128 target models and 64 shadow models, so 64,000 member and 64,000 non-member pairs to score.

## The loss attack and the generalization gap {#sec-mia-loss}

**Idea.** A model fits its training data better than other data, so its loss on a training record is usually lower. The simplest attack, due to Yeom et al. (2018), says "member" when the loss is low.

**Result.** For a loss $\ell$ bounded in $[0, B]$, saying "member" with probability $1 - \ell/B$ has advantage
$$\mathrm{Adv} = \frac{\text{expected test loss} - \text{expected training loss}}{B},$$
the expected generalization gap divided by $B$. With the 0-1 loss the attack is "member if classified correctly", with advantage test error minus training error: a leak you can read off without running anything.

**Example.** The training error is 0.1495 and the test error 0.2663. "Member if correctly classified" has TPR 0.8505, FPR 0.7337 and advantage 0.1168, exactly the gap. It identifies nobody with confidence, since most non-members are flagged too (@fig-mia-loss-distributions).

![Distribution of $\phi$, the logit of the target model's confidence in the true label, over member and non-member pairs. The dashed line is the mean-training-loss threshold: the attack says "member" to its right.](figures/mia_loss_distributions.svg){#fig-mia-loss-distributions}

**Pitfalls.** Overfitting implies leakage, but **a small gap does not mean a private model.** The loss attack ignores which record it is looking at, record-aware attacks beat it by a wide margin, and its average hides who is exposed.

## Shadow models {#sec-mia-shadow}

**Idea.** To learn what members' outputs look like without knowing the target's training set, the attacker trains **shadow models** itself, with the same recipe on data from the same population, so it knows who is a member of each (Shokri et al. 2017). It labels their outputs "in" and "out" and trains an **attack model** to classify them.

**Result.** With enough shadow data, the attack model learns the probability of membership given the output: the best attack *among those that treat all records alike*. What shadow models really buy is **calibration**: the attacker sets the threshold at which a chosen fraction of its own shadow non-members pass, and gets that FPR on the real model.

**Example.** Here the attack model amounts to a per-class loss threshold. Set on the shadows for an FPR of 1%, it gives an FPR of 0.0095 on the target models: calibration works. But its TPR there is 0.0098, no better than chance.

**Pitfalls.** Architecture, optimizer, epochs and augmentation all change how strongly a model fits each record, so imitate the target's recipe; a mismatch mostly spoils calibration, since "lower loss means member" transfers across tasks (Salem et al. 2019). A richer input to the attack model does not fix the deeper problem: it never compares a record with itself.

## LiRA: a likelihood-ratio test per record {#sec-mia-lira}

**Idea.** LiRA (Carlini et al. 2022) compares a record's output on the target model with the same record's outputs on shadow models that did and did not train on it, fits a Gaussian to each set, and asks which fits better. The statistic is $\phi = \ln\frac{p_y}{1 - p_y}$, the log-odds of the confidence $p_y$ in the true label, which is close to Gaussian over retrainings while $p_y$ piles up against 1.

**Result.** For a target record $z$, let $\mu_{\mathrm{in}}, \sigma_{\mathrm{in}}$ be the mean and standard deviation of $\phi_z$ over the shadow models that trained on $z$, and $\mu_{\mathrm{out}}, \sigma_{\mathrm{out}}$ over those that did not. **Online LiRA** scores the observed $\phi_z$ by the log of how much more likely it is under the IN fit than under the OUT fit,
$$\ln\Lambda_z = \ln\mathcal N\big(\phi_z;\, \mu_{\mathrm{in}}, \sigma_{\mathrm{in}}^2\big) - \ln\mathcal N\big(\phi_z;\, \mu_{\mathrm{out}}, \sigma_{\mathrm{out}}^2\big),$$ {#eq-mia-lira-online}
where $\mathcal N(\phi; \mu, \sigma^2)$ is the normal density at $\phi$. It says "member" above a threshold that is **the same for every record**, and when the fits are right no attack has a higher TPR at any FPR. **Offline LiRA** uses OUT models only, trained once before the targets are known, and scores $(\phi_z - \mu_{\mathrm{out}})/\sigma_{\mathrm{out}}$, how many OUT standard deviations the output lies above its usual value.

**Example.** Take two kinds of record (@tbl-mia-two-types) and set each attack to a pooled FPR of $10^{-3}$. One threshold on $\phi$ must sit high because typical non-members score high, and its TPR is 0.0048. Offline LiRA gives every record FPR $10^{-3}$, for a TPR of $0.95 \times 0.0048 + 0.05 \times 0.8185 = 0.0455$. Online LiRA moves false-positive budget from typical records, where it buys little, to atypical ones, and reaches 0.0503, the best possible.

| Kind | Share | OUT: $\phi \sim$ | IN: $\phi \sim$ | Separation $s$ | Best TPR at FPR $10^{-3}$, this kind alone |
| --- | --- | --- | --- | --- | --- |
| Typical | 95% | $\mathcal N(3, 1)$ | $\mathcal N(3.5, 1)$ | 0.5 | 0.0048 |
| Atypical | 5% | $\mathcal N(-2, 1)$ | $\mathcal N(2, 1)$ | 4 | 0.8185 |

: Two kinds of record. A typical record is predicted confidently whether or not it was trained on; an atypical one is predicted wrongly unless it was. {#tbl-mia-two-types}

![The logit $\phi$ of one point over the 64 shadow models, split into the 32 that trained on it (IN) and the 32 that did not (OUT), with the fitted Gaussians. Left: a typical point, whose two distributions overlap. Right: an outlier, predicted wrongly unless the model trained on it.](figures/mia_lira_example.svg){#fig-mia-lira-example}

**Pitfalls.** Per-record variances from few shadow models are noisy, and an underestimated spread makes an ordinary output look extreme, creating false positives exactly at the low-FPR end. With few shadows, use one **global variance** shared by all records (@fig-mia-shadow-count). Online LiRA needs new shadows for new targets; offline LiRA trades some power for reuse. Compute $\phi$ from the logits $f$ as $f_y - \ln\sum_{y' \ne y}e^{f_{y'}}$, never from $1 - p_y$, which rounds to 0 for a confident model.

![TPR at FPR $10^{-3}$ of four LiRA variants as the attacker uses 4 to 64 shadow models. With a global variance the attack is strong even with few shadow models; with per-record variances it needs many. The dotted line is the loss attack.](figures/mia_shadow_count.svg){#fig-mia-shadow-count}

## Running LiRA on a real model {#sec-mia-cifar}

**Idea.** At full size, for example on an image classifier trained on CIFAR-10, the protocol is the running example's, with hours of GPU time in place of seconds.

**Result.** The protocol has six steps.

1. **Split.** For each of $N + 1$ models choose a random half of the pool. One model plays the target; rotating it gives many (model, record) pairs.
2. **Train** every model with the target's exact recipe, including data augmentation.
3. **Statistic.** For each model and record, compute $\phi$ from the logits, on the record and on a few augmented copies (flips, small shifts).
4. **Fit** IN and OUT Gaussians per record over the shadow models.
5. **Score** each (target model, record) pair with @eq-mia-lira-online, or the offline score.
6. **Evaluate.** Pool all pairs, sweep one threshold, plot the ROC curve on log axes, and report the TPR at FPR $10^{-3}$ with the number of non-member pairs behind it.

**Example.** With 16 shadow models each record is IN for about 8. On the running example, per-record variances from 16 shadows give a TPR at FPR $10^{-3}$ of only 0.0023, a global variance 0.0151. One target with 25,000 non-member images allows only 25 false positives at that FPR, so rotate over several targets. Carlini et al. report online LiRA about ten times more powerful than earlier attacks at low FPR on CIFAR-10.

**Pitfalls.** A shadow recipe that differs from the target's, for example in augmentation, estimates the wrong distributions. Papers usually choose the threshold on the data that measure the rates; a real attacker sets it on its shadows. To audit the worst case, plant **canaries**, records built to be as exposed as possible, as members.

## Evaluating attacks: TPR at a low FPR {#sec-mia-eval}

**Idea.** Accuracy and the **area under the curve (AUC)**, the probability that a random member outscores a random non-member, average over everyone. A breach happens when an attacker can say *with confidence* that a particular person was in the data, which requires a low FPR.

**Result.** Report the TPR at a fixed low FPR, such as $10^{-3}$, and the whole ROC curve on log axes, where chance is the diagonal at every scale. An FPR is estimated by counting false positives, and a count of $k$ has a relative error of about $1/\sqrt k$, so to measure FPR $\alpha$ with $k$ false positives you need about $k/\alpha$ non-members. Give a confidence interval for each rate, such as the exact **Clopper–Pearson interval**.

**Example.** Attack A recognizes 5% of members with certainty and nobody else; attack B shifts every member's score slightly. They have the same AUC and differ by a factor of 38 at low FPR (@tbl-mia-same-auc). @tbl-mia-attacks scores this chapter's attacks on the running example.

| | AUC | Best advantage | TPR at FPR $10^{-3}$ |
| --- | --- | --- | --- |
| A: 5% of members exposed | 0.5250 | 0.0500 | 0.05095 |
| B: every member shifted by $\mu = 0.0887$ | 0.5250 | 0.0354 | 0.00134 |

: Two attacks with the same AUC. Only the TPR at a low FPR tells them apart. {#tbl-mia-same-auc}

| Attack | AUC | Best advantage | TPR at FPR $10^{-3}$ | TPR at FPR $10^{-2}$ |
| --- | --- | --- | --- | --- |
| Loss threshold (Yeom) | 0.577 | 0.122 | 0.0009 | 0.0103 |
| Shadow attack model (Shokri) | 0.577 | 0.121 | 0.0008 | 0.0104 |
| LiRA offline | 0.632 | 0.189 | 0.0030 | 0.0273 |
| LiRA online | 0.629 | 0.175 | 0.0149 | 0.0386 |
| LiRA online, global variance | 0.623 | 0.169 | 0.0177 | 0.0396 |

: Every attack on the running example: 128 target models, 64 shadow models, 64,000 member and 64,000 non-member pairs. {#tbl-mia-attacks}

By AUC the attacks are close. At FPR $10^{-3}$ the loss and shadow attacks are at chance, and online LiRA finds 952 of the 64,000 members, 17.0 times the loss attack, with TPR interval $[0.0140, 0.0158]$. Its FPR rests on 64 false positives, interval $[0.00077, 0.00128]$.

![ROC curves of the five attacks on log axes. The loss threshold and the shadow attack model follow the diagonal below FPR 0.01; online LiRA stays far above it. The dashed line marks FPR $10^{-3}$.](figures/mia_roc.svg){#fig-mia-roc}

**Pitfalls.** Never report accuracy or AUC alone. Don't report a TPR at an FPR your data cannot resolve: with 5,000 non-members and no false positives, you can only bound the FPR by about $3/5{,}000 = 6\times10^{-4}$. State the prior with any precision ([Threat models and privacy games](01-threat-models.md)).

## Why outliers are most exposed {#sec-mia-outliers}

**Idea.** The low-FPR end of the ROC curve is made of records whose IN and OUT distributions are far apart: those the model cannot predict without seeing them, such as rare or mislabelled records.

**Result.** For a record whose IN and OUT distributions are Gaussians of common width $\sigma_z$, the best TPR of any test on that record at FPR $\alpha$ is
$$\mathrm{TPR}_z(\alpha) = 1 - \Phi\big(\Phi^{-1}(1 - \alpha) - s_z\big), \qquad s_z = \frac{\mu_{\mathrm{in}} - \mu_{\mathrm{out}}}{\sigma_z},$$
where $\Phi$ is the standard normal cumulative distribution function (CDF) and $\Phi^{-1}$ its inverse. The numerator is the record's **self-influence**, how much training on $z$ raises the model's $\phi$ on $z$, and the **separation** $s_z$ measures it in standard deviations. At $\alpha = 10^{-3}$ a separation of 0.5 gives 0.0048 and a separation of 4 gives 0.8185.

**Example.** At FPR $10^{-3}$, online LiRA flags 26% of outlier members and 0.2% of typical ones (@tbl-mia-outliers), so 87% of the members it finds are outliers, from 5% of the pool. Outliers whose coin-flip label happens to agree with the trend are not exposed (@fig-mia-outliers).

| | Typical points (950) | Outliers (50) |
| --- | --- | --- |
| Mean separation $s_z$ | 0.461 | 1.539 |
| Online LiRA at pooled FPR $10^{-3}$: TPR | 0.0020 | 0.2597 |
| Online LiRA at pooled FPR $10^{-3}$: FPR | 0.00086 | 0.00375 |

: The running example, split into typical points and outliers. {#tbl-mia-outliers}

![Each pool point's separation against $\mu_{\mathrm{out}}$, the mean $\phi$ of the shadow models that did not train on it. Outliers (triangles) whose label the rest of the data contradict have the largest separations.](figures/mia_outliers.svg){#fig-mia-outliers}

**Pitfalls.** A low loss is normal for an easy record and strong evidence for a hard one. **Removing the outliers does not fix it**: Carlini et al. (2022b) removed the most exposed records and retrained, and the next layer became exposed, because exposure is relative to the rest of the data. Memorizing rare examples may also be what makes a model accurate on long-tailed data (Feldman 2020). A defence has to bound the influence of the worst record, which is what [differential privacy](../part03-dp-foundations/01-definition.md) does.

## Code {#sec-mia-code}

This reproduces the two-kinds example; the last two lines score a typical record's output 3.244 OUT standard deviations above its OUT mean, exactly on online LiRA's threshold.

```python
from math import log, pi, sqrt
from statistics import NormalDist

Z = NormalDist()


def log_normal_pdf(x: float, mu: float, sd: float) -> float:
    return -0.5 * ((x - mu) / sd) ** 2 - log(sd * sqrt(2 * pi))


def lira_online(phi: float, mu_in: float, sd_in: float, mu_out: float, sd_out: float) -> float:
    """Log of how much more likely the target's phi is under the IN fit than under the OUT fit."""
    return log_normal_pdf(phi, mu_in, sd_in) - log_normal_pdf(phi, mu_out, sd_out)


def lira_offline(phi: float, mu_out: float, sd_out: float) -> float:
    """How many OUT standard deviations the target's phi lies above its OUT mean."""
    return (phi - mu_out) / sd_out


def tpr_at_fpr(fpr: float, separation: float) -> float:
    """Best TPR at a given FPR for a record whose IN and OUT scores are unit normals `separation` apart."""
    return 1 - Z.cdf(Z.inv_cdf(1 - fpr) - separation)


typical, atypical = tpr_at_fpr(1e-3, 0.5), tpr_at_fpr(1e-3, 4.0)
print(f"{typical:.4f}")  # 0.0048
print(f"{atypical:.4f}")  # 0.8185
print(f"{0.95 * typical + 0.05 * atypical:.4f}")  # 0.0455
print(f"{lira_offline(6.244, 3.0, 1.0):.3f}")  # 3.244
print(f"{lira_online(6.244, 3.5, 1.0, 3.0, 1.0):.3f}")  # 1.497
```

The full edition's script, [membership_inference.py](../../code/part02-attacks/membership_inference.py), runs every attack and prints every number in this chapter:

```bash
uv run python code/part02-attacks/membership_inference.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| Loss attack | $\mathrm{Adv}$ = expected gap $/B$; 0-1 loss: test error $-$ train error | Average-case; a lower bound on leakage |
| Shadow attack model | Learns the membership probability of the output | Best record-blind attack; its real use is calibrating thresholds |
| Online LiRA | $\ln\mathcal N(\phi; \mu_{\mathrm{in}}, \sigma_{\mathrm{in}}^2) - \ln\mathcal N(\phi; \mu_{\mathrm{out}}, \sigma_{\mathrm{out}}^2)$, one threshold for all records | Optimal when the fits are right; needs IN models per target |
| Offline LiRA | $(\phi - \mu_{\mathrm{out}})/\sigma_{\mathrm{out}}$ | OUT models only, reusable |
| Per-record best TPR | $1 - \Phi(\Phi^{-1}(1 - \alpha) - s_z)$, $s_z = (\mu_{\mathrm{in}} - \mu_{\mathrm{out}})/\sigma_z$ | Large separation: outliers |
| Measuring FPR $\alpha$ | About $k/\alpha$ non-members for $k$ false positives | Relative error about $1/\sqrt k$ |

: The results of this chapter. {#tbl-mia-cheat}

**Rules of thumb**

- Audit with LiRA, not with a global loss threshold, which is near chance at low FPR.
- Report the TPR at FPR $10^{-3}$ or lower, on log-scaled ROC curves, with the number of non-members behind it. Never accuracy or AUC alone.
- With fewer than about 64 shadow models, use a global variance.
- Expect the exposed records to be outliers, rare subgroups and mislabelled points; removing them exposes the next ones.
- Overfitting implies leakage; no overfitting does not imply its absence.

## Further reading

- Carlini, Chien, Nasr, Song, Terzis & Tramèr, "Membership Inference Attacks From First Principles" (IEEE S&P 2022): LiRA online and offline, the global variance, and the case for TPR at low FPR. The protocol to follow in an audit.
- Shokri, Stronati, Song & Shmatikov, "Membership Inference Attacks against Machine Learning Models" (IEEE S&P 2017): shadow models and attack models.
- Yeom, Giacomelli, Fredrikson & Jha, "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting" (IEEE CSF 2018): the loss attack and its link to the generalization gap.
- Carlini, Jagielski, Zhang, Papernot, Terzis & Tramèr, "The Privacy Onion Effect: Memorization is Relative" (NeurIPS 2022): why removing the most exposed records does not help.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/02-membership-inference.md).
