# Membership inference

## Why this matters

A hospital publishes a model that predicts a disease, and you hold a patient's record. **Was that record in the training set?** If it was, the patient was treated there for that disease. This is **membership inference**, the standard measure of leakage, since a model that cannot hide whether a record was used will hide little else.

The one idea of this chapter is that membership is a test about one record. Some records are predicted well whether or not the model trained on them, so their outputs reveal almost nothing. Others are predicted well only if it did. An attack that applies one rule to every record mixes the two kinds and identifies almost nobody. The attack to use in an audit is the likelihood ratio attack (LiRA). It compares the model's output on a record with how that same record fares under models trained with and without it.

## Membership as a test about one record {#sec-mia-test}

The game is the one of [Threat models and privacy games](01-threat-models.md). The attacker gets a training member or a fresh record with equal chance. It is scored by its **true positive rate (TPR)** and **false positive rate (FPR)**, and its advantage is the TPR minus the FPR.

Fix one target record, written $z$, and look at the model's output on it, such as its loss. Over many training runs with $z$, that output has one distribution, and over runs without $z$ it has another. Call them the **IN** and **OUT** distributions of $z$. The best test thresholds their likelihood ratio ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)).

A simple case sets the scale. Suppose the attack's score is normal with standard deviation 1 and mean 0 for non-members, and shifted for members by an amount $s$, called the **separation**. Then, writing $\Phi$ for the standard normal cumulative distribution function and $\alpha$ for the FPR you allow, the best TPR is
$$\mathrm{TPR}(\alpha) = 1 - \Phi\big(\Phi^{-1}(1 - \alpha) - s\big).$$ {#eq-mia-gauss-tpr}
For example, at an FPR of $10^{-3}$ a separation of 0.5 gives a TPR of only 0.0048, but a separation of 4 gives 0.8185. Typical records have small separations and outliers have large ones, so the low-FPR end of any attack is made of outliers.

The numbers below come from the full edition's experiment, which trains logistic regression on random halves of a pool in which 5% of points are outliers. The attacker attacks 128 target models, with 64 shadow models of its own.

## The quick check: the generalization gap {#sec-mia-loss}

Before any attack, compare the training and test errors. A model fits its training data, so its loss is lower on members. Yeom et al. (2018) showed that the attack "member if correctly classified" has an advantage equal to the gap between the errors:
$$\mathrm{Adv} = \text{test error} - \text{training error}.$$
In the running example, the gap between the two errors, and so the advantage, is 0.1168.

This gives a lower bound for free, since a model with a generalization gap leaks membership by at least the gap. The converse fails, because the gap is an average over records, and an average hides who is exposed. Yeom et al. even construct models that leak with no gap at all, so **a small gap does not mean a private model.** At a low FPR the loss attack is near chance, as @tbl-mia-attacks shows.

The classic attack of Shokri et al. (2017) trains **shadow models**, copies of the target trained on the same population, and learns to tell their members from non-members. It learns a rule that treats all records alike, so it is mostly superseded, but shadows remain the way to calibrate a threshold for the target.

## LiRA: a likelihood-ratio test per example {#sec-mia-lira}

LiRA, from Carlini et al. (2022), estimates each record's own IN and OUT distributions from shadow models trained with and without it. Its statistic is the logit of the model's confidence. Writing $p_y$ for the confidence in the true label $y$, the logit is
$$\phi = \ln\frac{p_y}{1 - p_y}.$$
LiRA uses it because it is close to normal over retrainings, while the confidence itself piles up against 1. For each record, fit a normal distribution to its logit under the shadow models that trained on it, with the mean $\mu_{\mathrm{in}}$ and the standard deviation $\sigma_{\mathrm{in}}$. Fit another to the shadow models that did not, with mean $\mu_{\mathrm{out}}$ and standard deviation $\sigma_{\mathrm{out}}$.

**Online LiRA** scores the target model's output $\phi_z$ by how much more likely it is under the IN fit than the OUT fit. Writing $\mathcal N(\phi; \mu, \sigma^2)$ for the normal density, the score is
$$\ln\Lambda_z = \ln\mathcal N\big(\phi_z;\, \mu_{\mathrm{in}}, \sigma_{\mathrm{in}}^2\big) - \ln\mathcal N\big(\phi_z;\, \mu_{\mathrm{out}}, \sigma_{\mathrm{out}}^2\big).$$ {#eq-mia-lira-online}
**Offline LiRA** uses only OUT models, which can be trained once and reused for every target record. It asks how many OUT standard deviations the output lies above its usual value,
$$\frac{\phi_z - \mu_{\mathrm{out}}}{\sigma_{\mathrm{out}}}.$$ {#eq-mia-lira-offline}
Both put one threshold on the score for all records. Online LiRA is stronger, because it weighs each record's evidence by its own separation, while offline LiRA is cheaper. @tbl-mia-attacks compares them with the older attacks by their area under the curve (AUC) and by their TPR at a low FPR.

| Attack | AUC | TPR at FPR $10^{-3}$ |
| --- | --- | --- |
| Loss threshold (Yeom) | 0.577 | 0.0009 |
| Shadow attack model (Shokri) | 0.577 | 0.0008 |
| LiRA offline | 0.632 | 0.0030 |
| LiRA online | 0.629 | 0.0149 |
| LiRA online, global variance | 0.623 | 0.0177 |

: Every attack on the running example: 128 target models, 64 shadow models, 64,000 member and 64,000 non-member pairs. {#tbl-mia-attacks}

The table shows why the AUC is the wrong summary, because by AUC the attacks are close. But at an FPR of $10^{-3}$ the loss and shadow attacks are at chance, since their TPR is no higher than the FPR. Online LiRA's TPR is 17.0 times the loss attack's there, although both use the same statistic, so comparing each record with itself is the whole gain.

The main practical failure is noisy fits. Each record's variance is estimated from half the shadow models. An underestimated variance makes an ordinary output look like a far-tail event, and that produces false positives exactly at the strict thresholds that matter. Carlini et al. therefore recommend one **global variance**, estimated from all records at once, when shadow models are few. For example, with 16 shadow models online LiRA reaches a TPR of only 0.0023 at an FPR of one in a thousand with per-record variances. With a global variance it reaches 0.0151.

## Running LiRA on your model {#sec-mia-cifar}

The full-scale benchmark, CIFAR-10 with 16 shadow models, takes hours of GPU time, but the recipe is the same for any model.

1. **Split the data.** For each of $N + 1$ models, where $N$ is the number of shadows, choose a random half of the training pool. One model plays the target, and rotating that role adds pairs.
2. **Train** every model with exactly the target's recipe, augmentation included.
3. **Compute the logit** for each model and record. Compute it from the class scores $f$ before the softmax, as $f_y - \ln\sum_{y' \ne y}e^{f_{y'}}$, which never forms $1 - p_y$ and so never rounds to 0 for a confident model. Carlini et al. also query a few augmented copies of each input, such as flipped images, and fit over all of them.
4. **Fit** IN and OUT normals for each record over the shadows, with a global variance if you have fewer than about 64.
5. **Score** each pair of model and record with @eq-mia-lira-online, or with @eq-mia-lira-offline if you only have OUT models.
6. **Evaluate.** Pool all pairs, sweep one threshold, and plot the receiver operating characteristic (ROC) curve on log axes.

The cost is $N + 1$ training runs, so the shadow count is the budget decision, and a global variance makes 16 shadows enough.

The members and non-members must come from the same distribution, and violating this is the most common way a membership audit goes wrong. If non-members come from a later period, another site or another preprocessing pipeline, the attack measures that shift rather than the model. So draw both from one pool, split at random, as step 1 does.

When shadows cost too much, there are cheaper options. Offline LiRA reuses one set of OUT models. Reference-model attacks, such as the relative membership inference attack of Zarifzadeh et al. (2024), are designed to stay strong with only a few reference models. Against a service that returns labels only, a label-only attack measures how much an input must be perturbed before its label flips (Choquette-Choo et al., 2021).

For the computation, `scipy.stats.norm.logpdf` gives the two terms of the online score, and `sklearn.metrics.roc_curve` gives the curve. TensorFlow Privacy's membership-inference tests take losses or logits of members and non-members in an `AttackInputData` object, and `run_attacks` runs simple threshold attacks on them. LiRA's reference implementation lives in the same repository. The Adversarial Robustness Toolbox offers `MembershipInferenceBlackBox`, a trained attack model, and `LabelOnlyDecisionBoundary`, and ML Privacy Meter implements reference-model attacks.

## Reading the result {#sec-mia-eval}

Report the TPR at an FPR of $10^{-3}$ or lower, with the ROC curve on log axes ([Threat models and privacy games](01-threat-models.md)), because averages hide breaches. An attack that exposes 5% of members with certainty can have the same AUC as one that nudges every score slightly, yet catch 38 times as many members.

Measuring a low FPR takes many non-members, because an FPR is estimated by counting false positives. A count of $k$ events has a relative error of about $1/\sqrt k$. So to see $k$ false positives at an FPR $\alpha$, you need about $k/\alpha$ non-members. For example, the running example's 64,000 non-members give about 64 false positives at one in a thousand. Even then, the Clopper–Pearson 95% interval for online LiRA's FPR runs from 0.00077 to 0.00128.

Then check three things.

- **The TPR's interval is clear of the FPR.** If not, the attack is at chance there, like the loss attack in @tbl-mia-attacks.
- **The top-scored records make sense.** Inspect them, because they are usually outliers, rare subgroups or mislabelled points.
- **The thresholds were set on separate data.** Setting them on the data that measure the rates is usual but optimistic.

## Who is exposed, and what helps {#sec-mia-outliers}

The exposed records are those whose IN and OUT distributions lie far apart. A record's separation is large when the rest of the data contradict its label or its features are unusual. In the running example, at an FPR of one in a thousand, a share of 0.873 of the members online LiRA finds are outliers.

That points to what does and does not help.

- **Removing the exposed records does not.** Carlini et al. (2022b) removed the most exposed records and retrained, and the next layer became exposed. They call this the onion effect.
- **Closing the generalization gap is not enough.** More data or stronger regularization shrink the average, but the low-FPR end is set by the worst records.
- **Masking the output does not bound the leak,** because label-only attacks still work.
- **Differential privacy does.** It bounds every record's separation at once. [Private training](../part05-dp-ml/02-dp-sgd.md) provides it, and [an audit with planted canaries](../part05-dp-ml/10-auditing.md) checks it.

Expect a trade-off, because memorizing rare cases can be what makes a model accurate on long-tailed data (Feldman, 2020).

## Cheat sheet

| Situation | Use | Notes |
| --- | --- | --- |
| A first look, no attack budget | Test error $-$ training error | A lower bound on the leak; a small gap proves nothing |
| An audit with few shadow models | Online LiRA with a global variance | 16 shadows are usable this way |
| A label-only service | A label-only attack | Masking the scores does not stop membership inference |

: Which attack or check to use. {#tbl-mia-cheat}

**Rules of thumb**

- Audit with LiRA, not with a global loss threshold, which is near chance at a low FPR.
- Draw members and non-members from one pool, split at random, and train shadows with the target's exact recipe.
- Overfitting implies membership leakage, but no overfitting does not imply its absence. A failed weak attack is not evidence of safety.

## Further reading

- Carlini, Chien, Nasr, Song, Terzis & Tramèr, "Membership Inference Attacks From First Principles" (IEEE S&P 2022): LiRA online and offline, the global variance, and the protocol to follow in an audit.
- Yeom, Giacomelli, Fredrikson & Jha, "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting" (IEEE CSF 2018): the loss attack and its link to the generalization gap.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/02-membership-inference.md).
