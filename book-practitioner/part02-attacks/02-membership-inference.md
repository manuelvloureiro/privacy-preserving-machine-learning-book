# Membership inference

## Why this matters

A hospital publishes a model that predicts a disease, and you hold a patient's record. **Was that record in the training set?** If it was, the patient was treated there for that disease. This is **membership inference**, and it is the standard measure of leakage, because a model that cannot hide whether a record was used will hide little else.

The one idea of this chapter is that membership is a test about one record, so an attack must compare each record with itself. The attack to use in an audit is the likelihood ratio attack (LiRA). It compares the model's output on a record with that record's outputs under models trained with and without it.

## Membership as a test about one record {#sec-mia-test}

The game is the one of [Threat models and privacy games](01-threat-models.md). The attacker gets a training member or a fresh record and says "member" or "not". It is scored by its **true positive rate (TPR)**, the share of members it flags, and its **false positive rate (FPR)**, the share of non-members it flags.

Fix one target record, written $z$, and look at the model's output on it, such as its loss. Retrain on fresh random draws of the data, and that output has one distribution over the runs that include $z$ and another over the runs without it. Call them the **IN** and **OUT** distributions of $z$. The best test of membership compares their likelihoods at the observed output ([the Neyman–Pearson lemma](../part01-prerequisites/04-hypothesis-testing.md#sec-ht-np)), and it does well only where the two lie far apart. A typical record's distributions almost overlap, because the rest of the data already predict it. An outlier's lie far apart, because the model gets it right only if it trained on it, so the records an attack finds at a low FPR are mostly outliers.

The numbers in this chapter come from [practitioner_lira.py](../../code/part02-attacks/practitioner_lira.py), which trains 64 logistic-regression classifiers on random halves of scikit-learn's digits dataset, 1,797 images of handwritten digits. Each model takes a turn as the target, while the other 63 serve as the attacker's **shadow models**, copies trained the same way whose training images the attacker knows.

## The quick check: the generalization gap {#sec-mia-loss}

Before any attack, compare the training and test errors. A model fits its training data, so it gets members right more often than fresh records. Yeom et al. (2018) turned that into the attack "member if correctly classified". When members and non-members come from the same population, its advantage, the TPR minus the FPR, equals the gap between the two errors:
$$\mathrm{Adv} = \text{test error} - \text{training error}.$$
On the digits, averaged over the 64 target models, the gap is 0.0422.

So a large gap shows a leak at no cost. But **a small gap does not mean a private model**, because the gap is an average over records, and an average hides who is exposed. The loss attack, which flags every record whose loss falls below one threshold, shows why. On the digits, at an FPR of $10^{-3}$, it reaches a TPR of 0.0009. That is no higher than its FPR, so it does no better than a coin weighted to say "member" one time in a thousand.

The classic attack of Shokri et al. (2017) uses shadow models to learn one rule for all records, so it too fails at a low FPR.

## LiRA: a likelihood-ratio test per example {#sec-mia-lira}

LiRA, from Carlini et al. (2022), uses the shadow models instead to estimate each record's own IN and OUT distributions. Its statistic is the logit of the model's confidence $p_y$ in the true label $y$,
$$\phi = \ln\frac{p_y}{1 - p_y}.$$
LiRA uses the logit because it is close to normal over retrainings, while the confidence itself piles up against 1.

For each record, the attacker fits a normal distribution to the record's logit over the shadow models that trained on it. Its mean is written $\mu_{\mathrm{in}}$ and its standard deviation $\sigma_{\mathrm{in}}$. A second fit, over the shadow models that did not train on the record, has mean $\mu_{\mathrm{out}}$ and standard deviation $\sigma_{\mathrm{out}}$.

**Online LiRA** scores the target model's logit $\phi_z$ on the record by how much more likely it is under the IN fit than under the OUT fit. Writing $\mathcal N(\phi; \mu, \sigma^2)$ for the normal density, the score is
$$\ln\Lambda_z = \ln\mathcal N\big(\phi_z;\, \mu_{\mathrm{in}}, \sigma_{\mathrm{in}}^2\big) - \ln\mathcal N\big(\phi_z;\, \mu_{\mathrm{out}}, \sigma_{\mathrm{out}}^2\big).$$ {#eq-mia-lira-online}
**Offline LiRA** uses only OUT models. Unlike IN models, which must contain the record being tested, they can be trained once and reused for every new record. It asks how many OUT standard deviations the logit lies above its usual value,
$$\frac{\phi_z - \mu_{\mathrm{out}}}{\sigma_{\mathrm{out}}}.$$ {#eq-mia-lira-offline}

Each score already compares a record with its own fits, so one threshold then serves for all records. In the script, `phi` holds the target's logits on the images and `shadow_phi` the shadows' logits, one row per shadow. The array `shadow_member` says which shadow trained on which image.

```{.python excerpt="lira-scores"}
# Fit a Gaussian to each image's φ over the shadows that trained on it (IN) and those that did not (OUT).
phi_in = np.where(shadow_member, shadow_phi, np.nan)  # NaN where the shadow did not train on the image
phi_out = np.where(~shadow_member, shadow_phi, np.nan)
mu_in, sd_in = np.nanmean(phi_in, axis=0), np.nanstd(phi_in, axis=0, ddof=1)
mu_out, sd_out = np.nanmean(phi_out, axis=0), np.nanstd(phi_out, axis=0, ddof=1)
if global_variance:  # one spread shared by all images: steadier with few shadow models
    sd_in = np.full_like(sd_in, np.sqrt(np.nanmean(sd_in**2)))
    sd_out = np.full_like(sd_out, np.sqrt(np.nanmean(sd_out**2)))
# Online: how much more likely the target's φ is under the image's IN fit than under its OUT fit.
online = norm.logpdf(phi, loc=mu_in, scale=sd_in) - norm.logpdf(phi, loc=mu_out, scale=sd_out)
# Offline (OUT shadows only): how many OUT standard deviations φ lies above the OUT mean.
offline = (phi - mu_out) / sd_out
```

The command `uv run python code/part02-attacks/practitioner_lira.py` runs it in seconds. It compares the attacks by their TPR at two low FPRs and by their area under the curve (AUC) of the TPR against the FPR.

```text
  loss attack: AUC=0.518  TPR@0.001=0.0009  TPR@0.01=0.0105
  LiRA offline: AUC=0.577  TPR@0.001=0.0087  TPR@0.01=0.0345
  LiRA online: AUC=0.623  TPR@0.001=0.0318  TPR@0.01=0.0790
  LiRA online, global variance: AUC=0.614  TPR@0.001=0.0250  TPR@0.01=0.0737
```

By AUC the attacks look close. At an FPR of $10^{-3}$ they are not, because online LiRA finds 0.0318 of the members and the loss attack only 0.0009. Both use the same model outputs, so the whole gain comes from comparing each record's output with that record's own fits. Online LiRA beats offline LiRA because its IN fit also tells it how far training on a record moves that record's logit.

Noisy fits are the main practical failure. Each record's variance comes from only about half the shadow models, and an underestimated one makes an ordinary output look like a far-tail event. That output becomes a false positive at exactly the strict thresholds that matter. So Carlini et al. recommend one **global variance**, shared by all records, when shadow models are few. With a global variance 16 shadows give a usable attack, while per-record variances need about 64. With 63 shadows, as here, the per-record fits win.

## Running LiRA on your model {#sec-mia-cifar}

The recipe is the same for any model. Train the target and its shadows each on a random half of the data pool, all with exactly the target's recipe. A shadow trained differently, even with another augmentation, estimates the wrong IN and OUT distributions. Then compute each model's logit on each record and fit the IN and OUT normals. Let each model take a turn as the target, and score every pair of target model and record, for enough non-member pairs to measure a low FPR. The shadow count is the budget decision, since each shadow is a training run. [Pre-release privacy risk assessment](10-risk-assessment.md) runs the whole job on the digits.

For a confident model, $1 - p_y$ rounds to 0 and the logit becomes infinite. So compute it from the class scores $f$ before the softmax, as $f_y - \ln\sum_{y' \ne y}e^{f_{y'}}$, or from the log-probabilities of `log_softmax`, which keep the tail where members and non-members differ. Clipping the probabilities is no substitute. On a nearly separable model most confidences hit the clip, so the fits have no spread, those pairs drop out, and the most exposed records go with them. The script clips only because no digits model comes near the clip. Treat pairs dropped for want of spread as a failed audit.

The most common way a membership audit goes wrong is to draw members and non-members from different distributions. If non-members come from a later period, another site or another preprocessing pipeline, the attack measures that shift rather than the model, so split one pool at random.

When shadows cost too much, the reference-model attack of Zarifzadeh et al. (2024) stays strong with only a few. Returning labels only does not stop the attack, because a label-only attack measures how much an input must change before its label flips (Choquette-Choo et al., 2021). Check which attack a library runs, because TensorFlow Privacy's `run_attacks` and the Adversarial Robustness Toolbox's `MembershipInferenceBlackBox` treat all records alike, while ML Privacy Meter implements LiRA.

To catch a training change that makes a model leak, rerun a cheaper LiRA on every release candidate. Train 16 models with the candidate's own recipe, since shadows cached from an older recipe are stale, and score online LiRA with a global variance. On the digits, `practitioner_lira.py --models 16` runs this check in seconds, and it blocks the model, as the 64-model run does. Apply the release rule of [the risk assessment](10-risk-assessment.md#sec-risk-threat) unchanged, so the check and the decision agree. To catch a drift below that rule, store the last release's interval and alert when a candidate's lower end exceeds its upper end. Comparing point values would alarm on seed noise alone.

## Reading the result {#sec-mia-eval}

Report the TPR at an FPR of $10^{-3}$ or lower, with the receiver operating characteristic (ROC) curve on log axes, because averages hide breaches. Part 1's [recipe for evaluating a detector at a low FPR](../part01-prerequisites/04-hypothesis-testing.md#sec-ht-recipe) says how many non-members that takes and how to put Clopper–Pearson intervals on both rates. It also sets the thresholds on a separate split, a step the script skips, as the literature does.

```text
  LiRA offline: TP=502/57704  TPR=0.0087 [0.0080, 0.0095]  FP=56/57304  FPR=0.00098 [0.00074, 0.00127]  ln(TPR/FPR)=2.19  pass
  LiRA online: TP=1833/57704  TPR=0.0318 [0.0303, 0.0332]  FP=57/57304  FPR=0.00099 [0.00075, 0.00129]  ln(TPR/FPR)=3.46  block
```

The last word on each line applies the [risk assessment's](10-risk-assessment.md#sec-risk-threat) blocking rule. It blocks a release when the lower end of the TPR's 95% interval at an FPR of $10^{-3}$ is above 0.01, ten times that FPR. Online LiRA blocks this model, while offline LiRA alone would have passed it. The logarithm of the ratio of the rates is 3.46 for online LiRA. It is a point estimate of how large the privacy level $\varepsilon$ of differential privacy (Part 3) would have to be to allow this attack, not a certified lower bound. The pairs share models, so they are not independent trials, and a certified bound needs independent ones, such as the canaries of [Auditing](../part05-dp-ml/10-auditing.md#sec-audit-bound).

## Who is exposed, and what helps {#sec-mia-outliers}

Look at the records an attack scores highest, usually outliers, rare subgroups or mislabelled points. Their IN and OUT distributions lie far apart, because the rest of the data contradict their label or their features are unusual. On the digits, the most exposed image is a 9 that the models classify with confidence 0.812 when they trained on it and 0.003 when they did not. Online LiRA flags it at an FPR of $10^{-3}$ in all 33 target models that trained on it.

Removing the exposed records does not help, because then the next layer of records becomes exposed, which Carlini et al. (2022b) call the onion effect. Closing the generalization gap is not enough either, because the low-FPR end is set by the worst records, not the average.

Differential privacy does help, because [private training](../part05-dp-ml/02-dp-sgd.md) limits how far any one record can move the model, and so how far apart every record's IN and OUT distributions can lie. Expect a trade-off, because when many subgroups are rare, memorizing them can be what makes a model accurate (Feldman, 2020).

## Cheat sheet

| Situation | Use | Notes |
| --- | --- | --- |
| A first look, no attack budget | Test error $-$ training error | A lower bound on the leak; a small gap proves nothing |
| A release decision | Online LiRA, @eq-mia-lira-online, 64 shadows | Report TPR at FPR $10^{-3}$ with intervals |
| Few shadow models | Online LiRA with a global variance | 16 shadows are usable this way |

: Which attack or check to use. {#tbl-mia-cheat}

**Rules of thumb**

- Audit with LiRA. A failed weak attack, such as a global loss threshold, is not evidence of safety.

## Further reading

- Carlini, Chien, Nasr, Song, Terzis & Tramèr, "Membership Inference Attacks From First Principles" (IEEE S&P 2022): LiRA, the global variance, and the audit protocol.
- Yeom, Giacomelli, Fredrikson & Jha, "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting" (IEEE CSF 2018): the loss attack and its link to the generalization gap.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/02-membership-inference.md).
