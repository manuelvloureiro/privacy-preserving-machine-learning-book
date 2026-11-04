# Pre-release privacy risk assessment

## Why this matters

Before release, assess whether a model or table exposes information about individuals. This chapter evaluates three artefacts: a classifier behind a scoring API, a language model whose weights are shared with a partner, and a census extract. It compares attack results from earlier chapters with fixed thresholds and records the decision.

Specify blocking thresholds before running attacks to prevent choosing thresholds that fit the results. A failed attack establishes only that the tested attack failed, so report its strength as well.

The three scripts below produce the results on this page. The first two run in seconds on public data. The third is from [Fine-tune a language model privately with LoRA](../part05-dp-ml/14-private-llm-finetuning.md) and takes a few minutes on an idle CPU.

```bash
uv run python code/part02-attacks/practitioner_lira.py
uv run python code/part02-attacks/practitioner_table_risk.py
uv run python code/part05-dp-ml/practitioner_lora.py
```

## Fix the threat model and the thresholds first {#sec-risk-threat}

Complete the worksheet in [Threat models and privacy games](01-threat-models.md) for each artefact because releases can expose the same data differently. By default, use the person as the privacy unit and assume the adversary sees model scores, as many services return them.

Record the blocking rules in @tbl-risk-thresholds. The membership rule uses two rates: the true positive rate (TPR), the fraction of members flagged, and the false positive rate (FPR), the fraction of non-members incorrectly flagged. At FPR $10^{-3}$, block release if the lower end of the TPR's 95% confidence interval exceeds ten times the FPR. Using the lower bound reduces sensitivity to a lucky run. The factor of ten is a policy choice; it allows some advantage over chance.

Several remedies use differential privacy (DP), which bounds information any attack can learn about one person at privacy level $\varepsilon$ ([Part 3](../part03-dp-foundations/01-definition.md)). One method is stochastic gradient descent (SGD) with clipped per-record gradients and added noise, called [DP-SGD](../part05-dp-ml/02-dp-sgd.md).

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

A canary within 3 bits of maximum exposure ranks among the 8 most likely possible secrets. The rule uses one-copy canaries because a deduplicated corpus contains each secret once. For a corpus that is not deduplicated, apply it at the largest observed repeat count.

The four middle rows block release when an attack exceeds its baseline. [Attribute inference](03-attribute-inference.md), [property inference](04-property-inference.md), [reconstruction attacks](09-reconstruction.md) and [gradient inversion](06-gradient-inversion.md) specify these baselines. For the final row, independent audit trials provide a certified lower bound on a DP model's $\varepsilon$. A bound above the claim indicates an implementation or accounting error ([Auditing private training](../part05-dp-ml/10-auditing.md#sec-audit-bound)).

## Classifier: the free check {#sec-risk-gap}

First compare model accuracy on training and held-out data. The classifier is logistic regression trained on scikit-learn's 1,797 handwritten-digit images, as in [Membership inference](02-membership-inference.md#sec-mia-loss). Across 64 models, the mean gap is 0.0422. The rule "member if correctly classified" therefore has advantage 0.0422, a lower bound on leakage.

This check cannot clear a model because the average gap does not identify exposed records. For example, the loss attack flags records with low loss. At FPR $10^{-3}$, its TPR is 0.0009, no better than random guessing. Failure of a weak attack provides little evidence. A large gap motivates regularization; a small gap requires further testing.

## Classifier: LiRA with shadow models {#sec-risk-lira}

The release decision uses the likelihood ratio attack (LiRA) from [Membership inference](02-membership-inference.md#sec-mia-lira). It compares each target output with those of **shadow models**, trained with and without that record. Fix attack strength before testing; a weaker attack may pass a model that a stronger attack blocks. Follow the defaults in [Running LiRA on your model](02-membership-inference.md#sec-mia-cifar): use online LiRA with 64 shadows when feasible. With at most 16 shadows, fit one variance for all records. Include each record in each model by a fair coin, and evaluate each model in turn as the target.

With these defaults, the digits experiment has 57,704 member pairs and 57,304 non-member pairs of a model and an image. [Reading the result](02-membership-inference.md#sec-mia-eval) reports the attack's hits. Online LiRA has TPR 0.0318 at FPR 0.00099. Its 95% interval has lower bound 0.0303, above the 0.01 threshold, so the classifier fails @tbl-risk-thresholds. Offline LiRA and the Adversarial Robustness Toolbox shadow-model attack would pass it. Base the decision on the strongest affordable attack.

Report the FPR interval as well. At $10^{-3}$, the non-member pairs yield about 57 false positives, giving an interval from 0.00075 to 0.00129. The script sets and evaluates thresholds on the same pairs, which slightly inflates attack performance. Inspect the ten highest-scoring records; they are often rare or mislabeled.

## Fine-tuned language model: canaries and extraction {#sec-risk-canary}

A language model may reproduce a secret, which membership tests do not measure. This assessment inserts **canaries**, random secrets with a fixed format, and measures how strongly the trained model memorizes them. [Fine-tune a language model privately with LoRA](../part05-dp-ml/14-private-llm-finetuning.md#sec-llmwf-memorization) runs this audit using the scoring method in [Memorization and training-data extraction](05-memorization-extraction.md#sec-memo-exposure). It adapts Pythia-14m to 1,024 Tiny Shakespeare sequences, once without privacy and once with DP at $\varepsilon = 8$ per sequence. Canaries with 1 to 64 copies and uninserted controls receive two scores each. Exposure, in bits, ranks the secret by the sum of its tokens' log-probabilities, not their mean. Extraction checks whether greedy decoding with the partner's settings outputs the secret.

Apply the exposure rule to the extrapolated values, not the measured column. Ranking against 2,000 random candidates caps measured exposure at 10.97 bits, while the full secret space allows 16.61 bits. The measured value therefore cannot approach within 3 bits of the maximum. First check the controls: a random rank has mean exposure $\log_2 e = 1.44$ bits, and the controls' 1.4 bits are consistent with that value.

The non-private model fails @tbl-risk-thresholds. Its one-copy canaries have exposure of 2.0 bits, below the threshold, but greedy decoding outputs all three 64-copy secrets. Do not share this model. The DP run extracts no canaries, but its most repeated canaries still average about 4 bits, compared with 2.2 bits for its controls. The guarantee applies to one sequence, while a repeated secret spans multiple sequences. Deduplicate the training data before training.

## Table: re-identification on quasi-identifiers {#sec-risk-table}

**Quasi-identifiers** are the columns an adversary can find elsewhere, such as age, sex and country in a register. Records that agree on all of them form an **equivalence class**, and a record alone in its class can be linked to a name. The table is Adult, a census extract of 48,842 records, and the agreed quasi-identifiers are six columns, from age to education.

[Linkage and re-identification](07-linkage-reidentification.md#sec-link-quasi) measures uniqueness. The agreed six columns make 14.64% of records unique, so the table fails. Adding hours per week raises this to 36.55%. List every column an adversary might know, and test uncertain columns as well.

[Syntactic anonymity and how it breaks](08-syntactic-anonymity.md#sec-anon-k) generalizes age, country and education, then repeats the check. Generalization reduces the unique share to 0.94%, but the smallest class still has one record, so the table fails. Suppressing classes smaller than 5 removes 3.31% of records. This result applies only to the tested columns: adding raw hours per week makes 10.09% of records unique. In addition, 1.20% of records are in a class with one occupation, a distinct $\ell$ of 1 ([Syntactic anonymity](08-syntactic-anonymity.md#sec-anon-l)). Such a class reveals occupation even without identifying a record.

These shares measure uniqueness within the sample, matching the prosecutor model in which the adversary knows the target is in the table. Otherwise, estimate population uniqueness ([Linkage and re-identification](07-linkage-reidentification.md#sec-link-sampling)). A table that passes after suppression may still be vulnerable to intersection with another release of the same people ([Syntactic anonymity](08-syntactic-anonymity.md#sec-anon-composition)). If the table cannot remain useful after anonymization, release differentially private statistics as in [Release statistics with a library](../part03-dp-foundations/12-release-statistics.md). Accurate aggregates are not a safe fallback; they can enable reconstruction ([Reconstruction attacks](09-reconstruction.md)).

## Write the decision {#sec-risk-decide}

Compare each result with its threshold and record it in the release memo (@tbl-risk-memo). Include an interval for each estimated rate so that a reviewer can verify the decision.

| Memo entry | This assessment |
| --- | --- |
| Threat model | Scores from an API; weights shared with a partner; a table linked with a register. Unit: the person |
| Attacks and their strength | Loss attack, offline and online LiRA with 64 models; 12 canaries and 3 controls with exposure and greedy extraction; uniqueness on six quasi-identifiers plus one |
| Results | Online LiRA TPR 0.0318 [0.0303, 0.0332] at FPR 0.00099; all three 64-copy secrets extracted; 14.64% unique, 0.94% after generalizing |
| Verdict | All three blocked |
| Mitigation | Classifier: regularize or train with DP-SGD, then rerun. Language model: deduplicate and fine-tune with DP. Table: release DP statistics |
| Residual risk | Not assessed: [attribute inference](03-attribute-inference.md), [property inference](04-property-inference.md) and [reconstruction from aggregates](09-reconstruction.md). Not applicable, with no federated training: [gradient leakage](06-gradient-inversion.md) |

: A release memo for the three artefacts. {#tbl-risk-memo}

The residual-risk entry lists attacks that were not run. Include it even when all tested attacks fail: an untested attack may still reveal information. A DP guarantee bounds all attacks. For a DP model, include the guarantee and supporting evidence in the memo; [Release and governance](../part05-dp-ml/12-release-governance.md) describes the release package.

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
