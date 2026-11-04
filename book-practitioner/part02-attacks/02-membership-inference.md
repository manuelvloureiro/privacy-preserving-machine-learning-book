# Membership inference

## Why this matters

A hospital publishes a disease-prediction model. A patient may ask whether their record was in the training set; if so, the model's training data included a record of their disease. **Membership inference** tests whether a model used a specified record. It is a standard measure of leakage because a model that reveals record membership may reveal other information about that record. An audit therefore compares each record with the model's behaviour on that record.

## Membership as a test about one record {#sec-mia-test}

In the game described in [Threat models and privacy games](01-threat-models.md), the attacker receives either a training member or a fresh record and classifies it as a member or non-member. Its **true positive rate (TPR)** is the fraction of members it flags; its **false positive rate (FPR)** is the fraction of non-members it flags.

Fix a target record $z$ and measure the model's output on it, such as its loss. Across retrainings on fresh data samples, outputs have an **IN** distribution when the training set includes $z$ and an **OUT** distribution when it excludes $z$. The optimal membership test compares their likelihoods at the observed output ([the Neyman–Pearson lemma](../part01-prerequisites/04-hypothesis-testing.md#sec-ht-np)); it performs best where the distributions differ most.

The IN and OUT distributions for a typical record often overlap because other records predict it. For an outlier, they may differ more because the model predicts it well only after training on it. As a result, low-FPR attacks often identify outliers.

The numbers in this chapter come from [practitioner_lira.py](../../code/part02-attacks/practitioner_lira.py). It trains 64 logistic-regression classifiers, each on a random half of scikit-learn's 1,797 images of handwritten digits. Each model in turn is the target, and the other 63 are the attacker's **shadow models**, copies trained the same way on images the attacker knows.

## The quick check: the generalization gap {#sec-mia-loss}

Before running an attack, compare training and test errors. Because models tend to predict training records better than fresh records, Yeom et al. (2018) proposed the rule "member if the label is correct." When members and non-members come from the same population, the attack's advantage (TPR minus FPR) equals the difference between test and training errors:
$$\mathrm{Adv} = \text{test error} - \text{training error}.$$
On the digits, averaged over the 64 models, the gap is 0.0422.

The gap is a low-cost indicator of leakage. However, **a small gap does not establish privacy**: it averages across records and can hide exposed individuals. For example, a loss-threshold attack flags records below a single loss threshold. On the digits data, its TPR is 0.0009 at FPR $10^{-3}$, no better than random guessing at that rate.

The attack of Shokri et al. (2017) also learns one rule for all records from shadow models. The script runs it with four shadow models using the Adversarial Robustness Toolbox (ART).

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

This global rule also performs poorly at low FPR: it detects 0.0014 of members at FPR $10^{-3}$.

## LiRA: a likelihood-ratio test per record {#sec-mia-lira}

The likelihood ratio attack (LiRA) of Carlini et al. (2022) uses shadow models to estimate each record's IN and OUT distributions. Its statistic is the logit of the model's confidence $p_y$ in the true label $y$:
$$\phi = \ln\frac{p_y}{1 - p_y}.$$
The logit is approximately normal across retrainings, while confidence values can cluster near 1. For each record, the attacker fits a normal distribution to its logits from shadow models that included the record, with mean $\mu_{\mathrm{in}}$ and standard deviation $\sigma_{\mathrm{in}}$. A second fit to models that excluded it has mean $\mu_{\mathrm{out}}$ and standard deviation $\sigma_{\mathrm{out}}$.

**Online LiRA** compares the target model's logit $\phi_z$ with the IN and OUT fits. If $\mathcal N(\phi; \mu, \sigma^2)$ denotes a normal density, the score is
$$\ln\Lambda_z = \ln\mathcal N\big(\phi_z;\, \mu_{\mathrm{in}}, \sigma_{\mathrm{in}}^2\big) - \ln\mathcal N\big(\phi_z;\, \mu_{\mathrm{out}}, \sigma_{\mathrm{out}}^2\big).$$ {#eq-mia-lira-online}
**Offline LiRA** uses only OUT models, which exclude the target record and can be reused for new records. Its score is the logit's distance above the OUT mean in standard deviations:
$$\frac{\phi_z - \mu_{\mathrm{out}}}{\sigma_{\mathrm{out}}}.$$ {#eq-mia-lira-offline}

Each score already compares a record with its own fits, so one threshold works for all records. In the script, ART's `SklearnClassifier` wraps the scikit-learn recipe, trains every model and returns its probabilities.

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

No library installed with this book computes the LiRA scores, so `lira_scores` in [practitioner_privacy_tests.py](../../code/practitioner_privacy_tests.py) fits the two normals per image and returns @eq-mia-lira-online and @eq-mia-lira-offline. The command `uv run python code/part02-attacks/practitioner_lira.py` runs in seconds. It compares the attacks by their TPR at two low FPRs and by their area under the curve (AUC) of the TPR against the FPR.

```text
  loss attack: AUC=0.518  TPR@0.001=0.0009  TPR@0.01=0.0105
  ART shadow-model attack: AUC=0.570  TPR@0.001=0.0014  TPR@0.01=0.0146
  LiRA offline: AUC=0.577  TPR@0.001=0.0087  TPR@0.01=0.0345
  LiRA online: AUC=0.623  TPR@0.001=0.0318  TPR@0.01=0.0790
  LiRA online, global variance: AUC=0.614  TPR@0.001=0.0250  TPR@0.01=0.0737
```

The attacks have similar AUC values, and ART nearly matches offline LiRA. At FPR $10^{-3}$, their TPRs differ: online LiRA detects 0.0318 of members, compared with 0.0009 for the loss attack. All attacks use the same model outputs; LiRA's improvement comes from comparing each record's output with its own IN and OUT fits.

Noisy fits are a practical limitation because only about half the shadow models estimate each record's variance. An underestimated variance makes ordinary outputs appear extreme. With few shadow models, Carlini et al. recommend one **global variance** across records. This can work with 16 shadows; record-specific variances require about 64.

## Running LiRA on your model {#sec-mia-cifar}

For any model, train the target and shadow models on random halves of the same data pool using the target's exact training procedure. A different procedure, including a different augmentation, estimates different IN and OUT distributions. Fit the normal distributions to each record's logits, then treat each model in turn as the target and score every model-record pair. [Pre-release privacy risk assessment](10-risk-assessment.md) compares these results with its release rule.

For a confident model, $p_y$ may be so close to 1 that $1-p_y$ rounds to 0, making the logit infinite. Compute the logit from class scores $f$ before softmax, using $f_y - \ln\sum_{y' \ne y}e^{f_{y'}}$, or use `log_softmax`.

Do not clip probabilities to avoid this problem. On a nearly separable model, many confidences reach the clipping value, leaving no variation for the fits. The affected pairs are then dropped, potentially including the most exposed records. Treat dropped pairs as a failed audit. The script clips only because the digits models' confidences remain well within the clipping range.

Draw members and non-members from the same pool and split it at random. If non-members come from another period, site or preprocessing pipeline, the attack may measure that distribution shift instead of membership.

Returning only labels does not prevent membership inference. A label-only attack measures how much an input must change before its predicted label flips (Choquette-Choo et al. 2021). ML Privacy Meter implements LiRA. TensorFlow Privacy's `run_attacks` and ART's `MembershipInferenceBlackBox` apply the same attack rule to all records.

Training changes can affect leakage, so rerun LiRA on each release candidate. Train 16 models with the candidate's recipe and score online LiRA using a global variance. On the digits, `practitioner_lira.py --models 16` runs in seconds and blocks the model under the next section's rule, as does the 64-model run. Compare candidates using their intervals; point estimates vary with the random seed.

## Reading the result {#sec-mia-eval}

Report the TPR at an FPR of $10^{-3}$ or lower, with the receiver operating characteristic (ROC) curve on log axes. Put a Clopper–Pearson interval on each rate, as in Part 1's [recipe for evaluating a detector at a low FPR](../part01-prerequisites/04-hypothesis-testing.md#sec-ht-recipe). That recipe also sets the thresholds on a separate split, which the script, like the literature, does not do.

```text
  LiRA offline: TP=502/57704  TPR=0.0087 [0.0080, 0.0095]  FP=56/57304  FPR=0.00098 [0.00074, 0.00127]  ln(TPR/FPR)=2.19  pass
  LiRA online: TP=1833/57704  TPR=0.0318 [0.0303, 0.0332]  FP=57/57304  FPR=0.00099 [0.00075, 0.00129]  ln(TPR/FPR)=3.46  block
```

The final word on each line reports the [risk assessment's](10-risk-assessment.md#sec-risk-threat) blocking rule. It blocks release when the lower end of the TPR's 95% interval at FPR $10^{-3}$ exceeds ten times that FPR. Online LiRA's lower end, 0.0303, exceeds 0.01 and blocks this model; offline LiRA alone would pass it. For online LiRA, $\ln(\mathrm{TPR}/\mathrm{FPR}) = 3.46$ estimates the differential privacy level $\varepsilon$ (Part 3) needed to permit this attack. It is not a certified lower bound because pairs share models and are not independent trials, unlike canaries in [Auditing](../part05-dp-ml/10-auditing.md#sec-audit-bound).

## Who is exposed, and what helps {#sec-mia-outliers}

The highest-scoring records are often outliers, members of rare subgroups or mislabeled points that other data do not predict well. On the digits data, the most exposed image is a 9. Models assign it confidence 0.812 when trained on it and 0.003 when not trained on it. Online LiRA therefore flags this 9 at FPR $10^{-3}$ in all 33 target models that included it.

Removing exposed records can expose other records in turn, the "onion effect" described by Carlini et al. (2022b). Reducing the average generalization gap is also insufficient because the low-FPR risk depends on the most exposed records. Differential privacy provides a guarantee: [Private training](../part05-dp-ml/02-dp-sgd.md) limits how much one record can change the model and thus limits the difference between its IN and OUT distributions. This may reduce utility when accurate prediction requires learning rare subgroups (Feldman 2020).

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
