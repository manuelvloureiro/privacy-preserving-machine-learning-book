# Membership inference

## Why this matters

A hospital publishes a model that predicts a disease, and you hold a patient's record. **Was that record in the training set?** If it was, the patient was treated there for that disease. This is **membership inference**, the standard measure of leakage, because a model that cannot hide whether a record was used hides little else. Membership is a test about one record, so the attack to use in an audit compares each record with itself.

## Membership as a test about one record {#sec-mia-test}

In the game of [Threat models and privacy games](01-threat-models.md), the attacker gets a training member or a fresh record and says "member" or "not". It is scored by its **true positive rate (TPR)**, the share of members it flags, and its **false positive rate (FPR)**, the share of non-members it flags.

Fix a target record, written $z$, and look at the model's output on it, such as its loss. Retrain on fresh random draws of the data, and that output has one distribution over the runs that include $z$ and another over the runs without it. Call them the **IN** and **OUT** distributions of $z$. The best test of membership compares their likelihoods at the observed output ([the Neyman–Pearson lemma](../part01-prerequisites/04-hypothesis-testing.md#sec-ht-np)), so it does well only where the two lie far apart. A typical record's distributions almost overlap, because the rest of the data already predict it. An outlier's lie far apart, because the model gets it right only if it trained on it. So the records an attack finds at a low FPR are mostly outliers.

The numbers in this chapter come from [practitioner_lira.py](../../code/part02-attacks/practitioner_lira.py). It trains 64 logistic-regression classifiers, each on a random half of scikit-learn's 1,797 images of handwritten digits. Each model takes a turn as the target, while the other 63 serve as the attacker's **shadow models**, copies trained the same way whose training images the attacker knows.

## The quick check: the generalization gap {#sec-mia-loss}

Before any attack, compare the training and test errors. A model fits its training data, so it gets members right more often than fresh records, and Yeom et al. (2018) turned that into the attack "member if correctly classified". When members and non-members come from the same population, its advantage, the TPR minus the FPR, equals the gap between the two errors:
$$\mathrm{Adv} = \text{test error} - \text{training error}.$$
On the digits, averaged over the 64 models, the gap is 0.0422.

So a large gap shows a leak at no cost. But **a small gap does not mean a private model**, because the gap averages over records and hides who is exposed. The loss attack, which flags every record whose loss falls below one threshold, shows this. At an FPR of $10^{-3}$ it reaches a TPR of 0.0009 on the digits, no better than a coin that says "member" one time in a thousand.

The classic attack of Shokri et al. (2017) also applies one rule to all records, learned from shadow models. The script runs it from the Adversarial Robustness Toolbox (ART), a Python library of attacks, with four shadow models.

```{.python excerpt="art-shadow-attack"}
# ART copies the target's recipe and trains each shadow on a random half of the images.
shadows = ShadowModels(models.first, num_shadow_models=N_ART_SHADOWS, random_state=seed)
(x_in, y_in, pred_in), (x_out, y_out, pred_out) = shadows.generate_shadow_dataset(x, onehot)
# One attack model learns "member" from a shadow's probabilities and the true label.
attack = MembershipInferenceBlackBox(models.first, attack_model_type="rf")
attack.fit(x_in, y_in, x_out, y_out, pred=pred_in, test_pred=pred_out)
# The same attack model scores every target's probabilities on every image.
scores = [attack.infer(x, onehot, pred=p, probabilities=True)[:, 0] for p in models.probs]
```

Its one rule fails at a low FPR too, finding only 0.0014 of the members at an FPR of $10^{-3}$.

## LiRA: a likelihood-ratio test per record {#sec-mia-lira}

The likelihood ratio attack (LiRA) of Carlini et al. (2022) uses the shadow models to estimate each record's own IN and OUT distributions instead. Its statistic is the logit of the model's confidence $p_y$ in the true label $y$,
$$\phi = \ln\frac{p_y}{1 - p_y}.$$
The logit is close to normal over retrainings, while the confidence itself piles up against 1.

For each record, the attacker fits a normal distribution to the record's logit over the shadow models that trained on it. Its mean is written $\mu_{\mathrm{in}}$ and its standard deviation $\sigma_{\mathrm{in}}$. A second fit, over the shadow models that did not, has mean $\mu_{\mathrm{out}}$ and standard deviation $\sigma_{\mathrm{out}}$.

**Online LiRA** scores the target model's logit $\phi_z$ on the record by how much more likely it is under the IN fit than under the OUT fit. Writing $\mathcal N(\phi; \mu, \sigma^2)$ for the normal density, the score is
$$\ln\Lambda_z = \ln\mathcal N\big(\phi_z;\, \mu_{\mathrm{in}}, \sigma_{\mathrm{in}}^2\big) - \ln\mathcal N\big(\phi_z;\, \mu_{\mathrm{out}}, \sigma_{\mathrm{out}}^2\big).$$ {#eq-mia-lira-online}
**Offline LiRA** uses only OUT models, which need not contain the record being tested, so they can be trained once and reused for every new record. It asks how many OUT standard deviations the logit lies above its usual value,
$$\frac{\phi_z - \mu_{\mathrm{out}}}{\sigma_{\mathrm{out}}}.$$ {#eq-mia-lira-offline}

Each score already compares a record with its own fits, so one threshold serves for all records. In the script, ART's `SklearnClassifier` wraps the scikit-learn recipe, trains every model and returns its probabilities.

```{.python excerpt="lira-shadows"}
# Each image is in each model's training set with probability 1/2.
member = rng.random((n_models, n)) < 0.5
for k in range(n_models):
    model = recipe()  # a fresh, untrained copy of the target's recipe
    model.fit(x[member[k]], onehot[member[k]])
    probs[k] = model.predict(x)  # the probabilities the attacker sees, for every image
    p = np.clip(probs[k, np.arange(n), y], 1e-15, 1 - 1e-15)  # confidence on the true label
    phi[k] = np.log(p) - np.log(1 - p)  # its logit, close to Gaussian over retrainings
    confidence[k] = p
```

The LiRA scores come from a helper, because no library installed with this book computes them. Its `lira_scores`, in [practitioner_privacy_tests.py](../../code/practitioner_privacy_tests.py), fits the two normals per image and returns @eq-mia-lira-online and @eq-mia-lira-offline.

The command `uv run python code/part02-attacks/practitioner_lira.py` runs in seconds. It compares the attacks by their TPR at two low FPRs and by their area under the curve (AUC) of the TPR against the FPR.

```text
  loss attack: AUC=0.518  TPR@0.001=0.0009  TPR@0.01=0.0105
  ART shadow-model attack: AUC=0.570  TPR@0.001=0.0014  TPR@0.01=0.0146
  LiRA offline: AUC=0.577  TPR@0.001=0.0087  TPR@0.01=0.0345
  LiRA online: AUC=0.623  TPR@0.001=0.0318  TPR@0.01=0.0790
  LiRA online, global variance: AUC=0.614  TPR@0.001=0.0250  TPR@0.01=0.0737
```

By AUC the attacks look close, and ART's attack nearly matches offline LiRA. At an FPR of $10^{-3}$ they are not, because online LiRA finds 0.0318 of the members and the loss attack only 0.0009. All of them see the same model outputs, so the whole gain comes from comparing each record's output with that record's own fits. Online LiRA beats offline LiRA because its IN fit also tells it how far training on a record moves that record's logit.

Noisy fits are the main practical failure. Each record's variance comes from only about half the shadow models. An underestimated one makes an ordinary output look like a far-tail event, a false positive at the strict thresholds that matter. With few shadow models, Carlini et al. therefore recommend one **global variance** shared by all records. It makes 16 shadows usable, while per-record variances need about 64.

## Running LiRA on your model {#sec-mia-cifar}

The recipe is the same for any model. Train the target and its shadows each on a random half of the data pool, all with exactly the target's recipe. A shadow trained differently, even with another augmentation, estimates the wrong IN and OUT distributions. Fit the IN and OUT normals to each record's logits, let each model take a turn as the target, and score every pair of target model and record. Each shadow is a training run, so the shadow count is the budget decision. [Pre-release privacy risk assessment](10-risk-assessment.md) runs the whole job on the digits.

For a confident model, the confidence $p_y$ in the true label nears 1, so $1 - p_y$ rounds to 0 and the logit becomes infinite. Compute it instead from the class scores $f$ before the softmax, as $f_y - \ln\sum_{y' \ne y}e^{f_{y'}}$, or from `log_softmax`. Clipping the probabilities is no substitute. On a nearly separable model most confidences hit the clip, so their fits have no spread and those pairs drop out, taking the most exposed records with them. Treat such dropped pairs as a failed audit. The script clips only because no digits model comes near the clip.

The most common mistake is to draw members and non-members from different distributions. If non-members come from a later period, another site or another preprocessing pipeline, the attack measures that shift rather than the model, so split one pool at random.

Two more facts matter in practice. Returning only labels does not stop the attack, because a label-only attack measures how much an input must change before its label flips (Choquette-Choo et al. 2021). And libraries differ: TensorFlow Privacy's `run_attacks` and ART's `MembershipInferenceBlackBox` treat all records alike, while ML Privacy Meter implements LiRA.

To catch a training change that makes a model leak, rerun a cheaper LiRA on every release candidate. Train 16 models with the candidate's own recipe and score online LiRA with a global variance. On the digits, `practitioner_lira.py --models 16` runs in seconds and blocks the model under the release rule of the next section, as the 64-model run does. Compare candidates by their intervals, not their point values, which differ by seed noise alone.

## Reading the result {#sec-mia-eval}

Report the TPR at an FPR of $10^{-3}$ or lower, with the receiver operating characteristic (ROC) curve on log axes. Put Clopper–Pearson intervals on both rates, as in Part 1's [recipe for evaluating a detector at a low FPR](../part01-prerequisites/04-hypothesis-testing.md#sec-ht-recipe). That recipe also sets the thresholds on a separate split, a step the script, like the literature, skips.

```text
  LiRA offline: TP=502/57704  TPR=0.0087 [0.0080, 0.0095]  FP=56/57304  FPR=0.00098 [0.00074, 0.00127]  ln(TPR/FPR)=2.19  pass
  LiRA online: TP=1833/57704  TPR=0.0318 [0.0303, 0.0332]  FP=57/57304  FPR=0.00099 [0.00075, 0.00129]  ln(TPR/FPR)=3.46  block
```

The last word on each line applies the [risk assessment's](10-risk-assessment.md#sec-risk-threat) blocking rule. It blocks a release when, at an FPR of $10^{-3}$, the lower end of the TPR's 95% interval is above ten times that FPR. Online LiRA's lower end, 0.0303, is above 0.01, so it blocks this model, which offline LiRA alone would have passed. For online LiRA, the logarithm of the ratio of the rates is 3.46. It estimates how large the privacy level $\varepsilon$ of differential privacy (Part 3) would have to be to allow this attack. It is not a certified lower bound, because the pairs share models and so are not independent trials. A certified bound needs independent ones, such as the canaries of [Auditing](../part05-dp-ml/10-auditing.md#sec-audit-bound).

## Who is exposed, and what helps {#sec-mia-outliers}

The records an attack scores highest are usually outliers, rare subgroups or mislabelled points, which the rest of the data do not predict. On the digits, the most exposed image is a 9 that the models classify with confidence 0.812 when they trained on it and 0.003 when they did not. Online LiRA flags it at an FPR of $10^{-3}$ in all 33 target models that trained on it.

Removing the exposed records does not help, because the next layer then becomes exposed, which Carlini et al. (2022b) call the onion effect. Closing the generalization gap is not enough either, because the worst records, not the average, set the low-FPR end.

Differential privacy does help. [Private training](../part05-dp-ml/02-dp-sgd.md) limits how far any one record can move the model, and so how far apart every record's IN and OUT distributions can lie. Expect a trade-off, because when many subgroups are rare, memorizing them can be what makes a model accurate (Feldman 2020).

## Cheat sheet

| Situation | Use | Notes |
| --- | --- | --- |
| A first look, no attack budget | Test error $-$ training error | A lower bound on the leak; a small gap proves nothing |
| A release decision | Online LiRA, @eq-mia-lira-online, 64 shadows | Report TPR at FPR $10^{-3}$ with intervals |
| Few shadow models | Online LiRA with a global variance | 16 shadows are usable this way |

: Which attack or check to use. {#tbl-mia-cheat}

**Rules of thumb**

- Audit with LiRA. A failed weak attack, such as a global loss threshold, is not evidence of safety.

## References

- Carlini, Chien, Nasr, Song, Terzis & Tramèr, "Membership Inference Attacks From First Principles" (IEEE S&P 2022): LiRA, the global variance, and the audit protocol.
- Yeom, Giacomelli, Fredrikson & Jha, "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting" (IEEE CSF 2018): the loss attack and its link to the generalization gap.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/02-membership-inference.md).
