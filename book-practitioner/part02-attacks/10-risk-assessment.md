# Pre-release privacy risk assessment

## Why this matters

Before a model or a table leaves your hands, someone has to decide whether it leaks too much about the people in it. This chapter runs the attacks of this part as the assessment a privacy reviewer would ask for. Its three releases are a classifier behind a scoring API, a language model whose weights go to a partner, and a record-level census extract.

The order matters as much as the attacks. The results that block the release are written down before anything runs, because a threshold chosen after seeing the numbers can be bent to fit them. And a failed attack shows only that this attack failed, so the assessment reports how strong its attacks were.

[`practitioner_lira.py`](../../code/part02-attacks/practitioner_lira.py) and [`practitioner_table_risk.py`](../../code/part02-attacks/practitioner_table_risk.py) run in seconds on public data. The language-model step reuses the output of [`practitioner_lora.py`](../../code/part05-dp-ml/practitioner_lora.py), from [Fine-tune a language model privately with LoRA](../part05-dp-ml/14-private-llm-finetuning.md), which takes a few minutes on an idle CPU.

```bash
uv run python code/part02-attacks/practitioner_lira.py
uv run python code/part02-attacks/practitioner_table_risk.py
uv run python code/part05-dp-ml/practitioner_lora.py
```

## Fix the threat model and the thresholds first {#sec-risk-threat}

Fill in the worksheet of [Threat models and privacy games](01-threat-models.md) once per artefact, because each release can expose the same data differently. By default the privacy unit is the person, and the adversary sees scores, the confidences a model returns, since many services return them.

Then write the blocking rules of @tbl-risk-thresholds into the plan. The membership rule uses the true positive rate (TPR), the share of members an attack flags, and the false positive rate (FPR), the share of non-members it wrongly flags. At an FPR of $10^{-3}$, it blocks when the lower end of the TPR's 95% confidence interval exceeds ten times the FPR. Reading the lower end means a lucky run cannot trigger it, and the factor of ten is a policy choice: an attack may beat chance, but not tenfold.

Several remedies use differential privacy (DP), [Part 3](../part03-dp-foundations/01-definition.md)'s guarantee that caps what any attack learns about one person by a privacy level $\varepsilon$. One is stochastic gradient descent (SGD) with each record's gradient clipped and noise added, called [DP-SGD](../part05-dp-ml/02-dp-sgd.md).

| Artefact | The adversary sees | Attack | Blocks the release when | Then |
| --- | --- | --- | --- | --- |
| Classifier behind a scoring API | Confidences for any input | Online LiRA, 64 models | Lower end of the TPR's 95% interval at FPR $10^{-3}$ is above 0.01 | Regularize, or train with DP-SGD |
| Fine-tuned language model shared with a partner | The weights | Canaries, exposure, greedy extraction | Any secret extracted, or a one-copy canary's extrapolated exposure within 3 bits of the maximum | Deduplicate, or fine-tune with DP |
| Record-level table | The table and a register with the quasi-identifiers | Uniqueness on the agreed quasi-identifiers | Any record unique, or in a class smaller than 5 (11 for counts from US Medicare and Medicaid data) | Generalize and suppress, else release DP statistics |
| Model, sensitive field | The target's other fields and label | Model inversion | Attribute advantage's 95% interval above 0 | Regularize, or train with DP |
| Model whose training mix is secret | Probe outputs | Shadow models, meta-classifier | Accuracy's 95% interval above one half | Do not release that mix |
| Published counts or queries | Every answer | Least squares, linear programming | Either beats the majority guess at the planned noise and query count | DP noise scaled to all answers |
| Federated training | Each client's update | Division and search | Any single update visible | Secure aggregation, client-level DP |
| Any model claimed to be DP | Depends on the claim | Audit with canaries | Certified lower bound on $\varepsilon$ above the claimed $\varepsilon$ | Block: the code or the accounting has a bug |

: The blocking thresholds, written before any attack runs. {#tbl-risk-thresholds}

A canary within 3 bits of the maximum exposure ranks among the 8 likeliest of all possible secrets. The rule reads one-copy canaries because a deduplicated corpus holds each secret once. If yours is not deduplicated, apply it at your data's largest repeat count.

The four middle rows block on any leak beyond the attack's own baseline, which the chapters on [attribute](03-attribute-inference.md), [property](04-property-inference.md) and [reconstruction attacks](09-reconstruction.md) and [gradient inversion](06-gradient-inversion.md) define. In the last row, an audit with independent trials certifies a lower bound on a DP model's $\varepsilon$, and a bound above the claim proves a bug ([Auditing private training](../part05-dp-ml/10-auditing.md#sec-audit-bound)).

## Classifier: the free check {#sec-risk-gap}

Compare the model's accuracy on its training data and on held-out data. Then run the loss attack, which says "member" whenever the loss on the true label is low ([Membership inference](02-membership-inference.md#sec-mia-loss)). The model is a logistic regression on scikit-learn's digits, 1,797 small images of handwritten digits. The script trains 64 copies, each on a random half of the images. Each copy in turn plays the target, the model under assessment, while the others serve as shadow models in the next step.

```text
target 0: train accuracy=1.0000  test accuracy=0.9650
train-test gap, mean over 64 targets=0.0422
ART rule-based attack on target 0: TPR=1.0000  FPR=0.9650  advantage=0.0350
loss attack at FPR 0.001: TPR=0.0009
loss attack at FPR 0.01: TPR=0.0105
```

The first target gets every training image right, but only 0.9650 of its test images. So even the rule "member if correctly classified" beats chance, by the mean gap of 0.0422. That is a free lower bound on the leak, though it does not say who is exposed. The Adversarial Robustness Toolbox (ART), a Python library of attacks, runs that rule as `MembershipInferenceBlackBoxRuleBased`, and its advantage of 0.0350 on the first target is that model's gap. The loss attack, at an FPR of $10^{-3}$, flags no more members than non-members. A weak attack's failure proves little, so this check can never clear a model. A large gap says regularize first, and a small one only that you need the next step.

## Classifier: LiRA with shadow models {#sec-risk-lira}

The likelihood ratio attack (LiRA) of [Membership inference](02-membership-inference.md#sec-mia-lira) compares the target's output on a record with the outputs of **shadow models**. These are trained like the target on random halves of the data, some with that record and some without. LiRA works with the logit $\phi$ of the confidence $p$ in the true label,
$$\phi = \ln\frac{p}{1 - p},$$
which is close to normal across retrainings. For each image, it fits a normal to $\phi$ over the shadows that trained on it (IN) and another over those that did not (OUT). Online LiRA scores the target's $\phi$ by its log-likelihood ratio between the two fits. Offline LiRA uses only the OUT fit, and counts how many OUT standard deviations $\phi$ lies above the OUT mean. ART's `SklearnClassifier` trains every model, and the helper [practitioner_privacy_tests.py](../../code/practitioner_privacy_tests.py) computes the LiRA scores.

Four decisions set the attack's strength. Use online LiRA, the strongest, for a release decision. Train 64 shadows when training is cheap; with 16 or fewer, fit one variance shared by all images, because per-image variances from so few models are too noisy. Put each record in each model by a fair coin, so that every record has IN and OUT shadows. And let each model serve once as the target, with the other 63 as its shadows. That gives 57,704 member and 57,304 non-member pairs of a model and an image.

At an FPR of $10^{-3}$, the 57,304 non-member pairs allow only about 57 false positives, so the script reads the rate with care. It takes the best TPR among thresholds whose FPR stays at or below the target. Then it puts a Clopper–Pearson interval, the exact binomial interval of [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md#sec-ht-estimation), on each rate.

```{.python excerpt="lira-report"}
# Every (target model, image) pair is one trial; members are the positives.
fpr, tpr, thresholds = roc_curve(member.ravel(), scores.ravel())
# The best TPR among thresholds whose FPR does not exceed the target.
i = int(np.flatnonzero(fpr <= target_fpr)[-1])
n_members, n_nonmembers = int(member.sum()), int((~member).sum())
true_pos = round(tpr[i] * n_members)
false_pos = round(fpr[i] * n_nonmembers)
# Exact 95% intervals (Clopper-Pearson) for both rates.
tpr_ci = clopper_pearson(true_pos, n_members)
fpr_ci = clopper_pearson(false_pos, n_nonmembers)
```

```text
  loss attack: AUC=0.518  TPR@0.001=0.0009  TPR@0.01=0.0105
  ART shadow-model attack: AUC=0.570  TPR@0.001=0.0014  TPR@0.01=0.0146
  LiRA offline: AUC=0.577  TPR@0.001=0.0087  TPR@0.01=0.0345
  LiRA online: AUC=0.623  TPR@0.001=0.0318  TPR@0.01=0.0790
[At FPR 0.001: counts, exact 95% intervals, ln(TPR/FPR) as a point estimate, not a certified bound; step 1 blocks if the TPR's lower end > 0.01]
  loss attack: TP=53/57704  TPR=0.0009 [0.0007, 0.0012]  FP=57/57304  FPR=0.00099 [0.00075, 0.00129]  ln(TPR/FPR)=-0.08  pass
  ART shadow-model attack: TP=82/57704  TPR=0.0014 [0.0011, 0.0018]  FP=57/57304  FPR=0.00099 [0.00075, 0.00129]  ln(TPR/FPR)=0.36  pass
  LiRA offline: TP=502/57704  TPR=0.0087 [0.0080, 0.0095]  FP=56/57304  FPR=0.00098 [0.00074, 0.00127]  ln(TPR/FPR)=2.19  pass
  LiRA online: TP=1833/57704  TPR=0.0318 [0.0303, 0.0332]  FP=57/57304  FPR=0.00099 [0.00075, 0.00129]  ln(TPR/FPR)=3.46  block
```

By area under the curve (AUC) the four attacks look alike, but at an FPR of $10^{-3}$ they differ sharply. Online LiRA flags 1,833 members, a TPR of 0.0318. The lower end of its interval, 0.0303, is above 0.01, so the model fails @tbl-risk-thresholds. Offline LiRA would have passed it, and so would ART's attack of Shokri et al. (2017), which learns one rule for all images from four shadow models of its own. So a release decision rests on the strongest attack you can afford.

A model with privacy level $\varepsilon$ keeps the TPR within about $e^{\varepsilon}$ times the FPR. So the last column, ln(TPR/FPR), estimates the $\varepsilon$ each attack implies. It is not a certified lower bound, because pairs that share a model are not independent trials ([Auditing private training](../part05-dp-ml/10-auditing.md#sec-audit-bound)).

The script also prints the ten images an attacker exposes first, those whose IN and OUT fits lie furthest apart. Inspect them, since such records are usually rare or mislabelled.

Three things go wrong here. Shadows trained with another recipe describe another model, which weakens the attack. Too few non-members leave the FPR uncertain, and even here its interval runs from 0.00075 to 0.00129. The script also sets thresholds and measures on the same pairs, which flatters the attack slightly, where [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md#sec-ht-recipe) uses two splits. And a confidence of 1 makes $\phi$ infinite, so compute $\phi$ from the logits, the model's raw outputs. Clipping the confidence instead hides a confident model's most exposed records ([Membership inference](02-membership-inference.md#sec-mia-cifar)).

## Fine-tuned language model: canaries and extraction {#sec-risk-canary}

A language model can repeat a secret word for word, which a membership test does not measure. So the assessment plants **canaries**, random secrets of a fixed format, and checks how well the trained model knows them ([Memorization and training-data extraction](05-memorization-extraction.md#sec-memo-exposure)). The run adapts Pythia-14m with low-rank adaptation (LoRA) to 1,024 sequences of Tiny Shakespeare, once without privacy and once with DP at $\varepsilon = 8$ per sequence.

The canaries should look like the secrets you fear. Each reads "the pin of", a name such as Alice, "is", and five random digits, spaced so that every possible secret is exactly five tokens. Three are inserted once, and three more at each of 4, 16 and 64 copies. Three controls are drawn the same way and never inserted. The canaries make up a quarter of the training set, far denser than real secrets, so even this small model memorizes them. The non-private model is then a positive control, one known to leak.

Each canary is scored in two ways, with the functions of [Fine-tune a language model privately with LoRA](../part05-dp-ml/14-private-llm-finetuning.md#sec-llmwf-memorization). Exposure ranks the secret among random secrets of the same format by the summed log-probability of its five digit tokens. It turns the rank into bits, as $\log_2$ of the number of candidates minus $\log_2$ of the rank. Sum rather than average, since the mean loss that `transformers` returns misranks secrets of different token counts. Extraction gives the model the prefix and lets it write the likeliest token five times, which is greedy decoding. Run it with the partner's decoding settings, and check whether it wrote the secret.

```text
non-private 1 copies: mean exposure 2.0 bits  extrapolated 2.0  extracted 0/3
non-private 4 copies: mean exposure 5.9 bits  extrapolated 5.7  extracted 0/3
non-private 16 copies: mean exposure 10.0 bits  extrapolated 13.0  extracted 0/3
non-private 64 copies: mean exposure 11.0 bits  extrapolated 16.3  extracted 3/3
non-private controls: mean exposure 1.4 bits  extrapolated 1.5  extracted 0/3
```

Ranking among 2,000 random candidates caps the measured exposure at 10.97 bits, below the 16.61 the whole space of secrets allows. So the measured column can never reach the blocking line 3 bits below that maximum. Only the extrapolated column can, by fitting a skew-normal distribution to the candidates' scores and reading how far into its tail each secret lies. A random rank is worth $\log_2 e = 1.44$ bits on average, and the controls' 1.4 bits match it, so the measurement is sound. The one-copy canaries stay far below the line, at 2.0 bits. But greedy decoding writes out all three 64-copy secrets, so the non-private model fails @tbl-risk-thresholds and must not be shared.

The 4-copy canaries show why you measure both, since two reach 6.9 and 8.2 bits yet neither is extracted. The DP run extracts nothing, but its most repeated canaries still average about 4 bits, against 2.2 for its controls. That is because $\varepsilon$ protects one sequence, and a repeated secret is many sequences. So deduplicate before training, whatever else you do.

The usual failures are a canary format unlike your data, no controls, and ranking by a mean loss. For a release, also measure the share of training sequences that greedy decoding completes from their start, the discoverable rate, which the script skips ([Memorization and training-data extraction](05-memorization-extraction.md#sec-memo-definitions)).

## Table: re-identification on quasi-identifiers {#sec-risk-table}

**Quasi-identifiers** are the columns an adversary can find elsewhere, such as age, sex and country in a register ([Linkage and re-identification](07-linkage-reidentification.md)). Records that agree on all of them form an **equivalence class**, and a record alone in its class can be linked to a name. The table is Adult, a census extract from the UCI Machine Learning Repository, with 48,842 records and an agreed set of six columns, from age to education.

For each set of columns, the script prints the share of unique records and the share in classes smaller than 5. It also prints the smallest class size $k$, and J/n, the number of classes per record ([Linkage and re-identification](07-linkage-reidentification.md#sec-link-quasi) shows the loop).

```text
age + sex + race: unique=0.12%  in classes <5=0.75%  min k=1  J/n=0.012
age + sex + race + native-country: unique=3.07%  in classes <5=7.83%  min k=1  J/n=0.060
age + sex + race + native-country + marital-status + education: unique=14.64%  in classes <5=26.83%  min k=1  J/n=0.227
age + sex + race + native-country + marital-status + education + hours-per-week: unique=36.55%  in classes <5=57.05%  min k=1  J/n=0.480
```

On the agreed six, 14.64% of the records are unique, and hours per week, a column that looks harmless, raises that to 36.55%. So the list must include every column an adversary might hold, and when in doubt, test with the column added.

The standard remedy is to generalize: age into 10-year bands, country into United States or other, and education into four levels. The script then rechecks the classes and counts the distinct occupations in each, the distinct $\ell$ of [Syntactic anonymity and how it breaks](08-syntactic-anonymity.md#sec-anon-l), with occupation as the sensitive column.

```text
after generalizing: unique=0.94%  in a class with one occupation=1.20%
age + sex + race + native-country + marital-status + education: unique=0.94%  in classes <5=3.31%  min k=1  J/n=0.032
age + sex + race + native-country + marital-status + education + hours-per-week: unique=10.09%  in classes <5=21.78%  min k=1  J/n=0.175
```

Generalizing cuts the unique share from 14.64% to 0.94%, but the smallest class still has one record, so the table still fails @tbl-risk-thresholds. Suppressing the classes smaller than 5 would remove 3.31% of the records. That would hold only for these columns, since raw hours per week makes 10.09% unique again. And 1.20% of the records sit in a class with a single occupation, which reveals it without any unique match.

These shares count uniqueness in the sample, which fits the prosecutor model, where the adversary knows the target is in the table. Otherwise, estimate population uniqueness ([Linkage and re-identification](07-linkage-reidentification.md#sec-link-sampling)).

Even a table that passes after suppression can be intersected with a second release of the same people ([Syntactic anonymity and how it breaks](08-syntactic-anonymity.md#sec-anon-composition)). And k-anonymity says nothing against richer side information. So when a table cannot pass without losing its use, release statistics with DP, as [Release statistics with a library](../part03-dp-foundations/12-release-statistics.md) does. Plain aggregates are no fallback, because many accurate counts let an attacker rebuild records ([Reconstruction attacks](09-reconstruction.md)).

## Write the decision {#sec-risk-decide}

The last step compares each result with its threshold and writes a release memo, @tbl-risk-memo. A reviewer who did not run the attacks must be able to check it, so it gives each estimated rate with its interval.

| Memo entry | This assessment |
| --- | --- |
| Threat model | Scores from an API; weights shared with a partner; a table linked with a register. Unit: the person |
| Attacks and their strength | Loss attack, offline and online LiRA with 64 models; 12 canaries and 3 controls with exposure and greedy extraction; uniqueness on six quasi-identifiers plus one |
| Results | Online LiRA TPR 0.0318 [0.0303, 0.0332] at FPR 0.00099; all three 64-copy secrets extracted; 14.64% unique, 0.94% after generalizing |
| Verdict | All three blocked |
| Mitigation | Classifier: regularize or train with DP-SGD, then rerun. Language model: deduplicate and fine-tune with DP. Table: release DP statistics |
| Residual risk | Not assessed: [attribute inference](03-attribute-inference.md), [property inference](04-property-inference.md) and [reconstruction from aggregates](09-reconstruction.md). Not applicable, with no federated training: [gradient leakage](06-gradient-inversion.md) |

: A release memo for the three artefacts. {#tbl-risk-memo}

The residual-risk entry, the one most often left out, names the attacks that were not run. Write it even when every attack fails, because a model can pass this assessment and still leak to an attack nobody tried. Only a DP guarantee bounds every attack at once. For a DP model the memo carries that guarantee and its evidence, which [Release and governance](../part05-dp-ml/12-release-governance.md) assembles into the release package.

## Checklist

| Step | Default | Check |
| --- | --- | --- |
| Threat model and thresholds | Person as unit; adversary sees scores; rules of @tbl-risk-thresholds | Written and dated before any attack runs |
| Free check | Train and test accuracy; loss attack | Gap and loss TPR at FPR $10^{-3}$ reported; a pass clears nothing |
| LiRA | Online, 64 models, fair-coin inclusion, target role rotated | TPR's interval at FPR $10^{-3}$ against 0.01; enough non-members; top records inspected |
| Canaries | Secret format, 1 to many copies, uninserted controls | Controls near 1.44 bits; summed log-probability; deployed decoding |
| Table | Every column an adversary may hold; classes of at least 5 | Unique share and smallest class after generalizing; distinct $\ell$ on the sensitive column |
| Decision | Memo with intervals, verdict, mitigation and residual risk | Every attack not run is named |

: The steps of a pre-release risk assessment, their defaults and their checks. {#tbl-risk-checklist}

## References

- Carlini et al., "Membership Inference Attacks From First Principles" (IEEE S&P 2022): LiRA, and why attacks are reported by their TPR at a low FPR.
- Carlini et al., "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks" (USENIX Security 2019): canaries, exposure and the skew-normal extrapolation.
- Rocher, Hendrickx & de Montjoye, "Estimating the success of re-identifications in incomplete datasets using generative models" (Nature Communications 2019): population uniqueness from a sample.

**Derivations.** This workflow has no counterpart in the full edition. It draws on the full edition's chapters on [threat models](../../book-full/part02-attacks/01-threat-models.md), [membership inference](../../book-full/part02-attacks/02-membership-inference.md), [memorization and extraction](../../book-full/part02-attacks/05-memorization-extraction.md), [linkage and re-identification](../../book-full/part02-attacks/07-linkage-reidentification.md) and [syntactic anonymity](../../book-full/part02-attacks/08-syntactic-anonymity.md).
