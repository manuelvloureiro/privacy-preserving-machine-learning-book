# Pre-release privacy risk assessment

## Why this matters

Before a model or a table leaves your hands, someone has to decide whether it leaks too much about the people in it. This chapter makes that decision for three releases. They are a classifier behind a scoring API, a language model whose weights go to a partner, and a census extract. The attacks themselves run in earlier chapters, and this one reads their results against fixed thresholds and writes the verdict down.

The order matters as much as the attacks. Write down which results block the release before any attack runs, because a threshold that you choose after you see the numbers can bend to fit them. A failed attack shows only that this attack failed, so the assessment also reports how strong its attacks were.

The three scripts below produce every result on this page. The first two run in seconds on public data. The third comes from [Fine-tune a language model privately with LoRA](../part05-dp-ml/14-private-llm-finetuning.md) and takes a few minutes on an idle CPU.

```bash
uv run python code/part02-attacks/practitioner_lira.py
uv run python code/part02-attacks/practitioner_table_risk.py
uv run python code/part05-dp-ml/practitioner_lora.py
```

## Fix the threat model and the thresholds first {#sec-risk-threat}

Fill in the worksheet of [Threat models and privacy games](01-threat-models.md) once per artefact, because each release can expose the same data differently. By default the privacy unit is the person, and the adversary sees scores, the confidences a model returns, because many services return them.

Then write the blocking rules of @tbl-risk-thresholds into the plan. The membership rule uses two rates. The true positive rate (TPR) is the share of members an attack flags, and the false positive rate (FPR) is the share of non-members it wrongly flags. At an FPR of $10^{-3}$, it blocks when the lower end of the TPR's 95% confidence interval exceeds ten times the FPR. The rule reads the lower end, so a lucky run cannot trigger it. The factor of ten is a policy choice: an attack may beat chance, but not tenfold.

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

A canary within 3 bits of the maximum exposure ranks among the 8 likeliest of all possible secrets. The rule reads one-copy canaries because a deduplicated corpus holds each secret once. If your corpus is not deduplicated, apply the rule at your data's largest repeat count.

The four middle rows block on any leak beyond the attack's own baseline. The chapters on [attribute](03-attribute-inference.md), [property](04-property-inference.md) and [reconstruction attacks](09-reconstruction.md) and [gradient inversion](06-gradient-inversion.md) give each baseline. In the last row, an audit with independent trials certifies a lower bound on a DP model's $\varepsilon$. A bound above the claim proves a bug ([Auditing private training](../part05-dp-ml/10-auditing.md#sec-audit-bound)).

## Classifier: the free check {#sec-risk-gap}

Start with the check that costs nothing: compare the model's accuracy on its training data and on held-out data. The classifier is a logistic regression on scikit-learn's 1,797 images of handwritten digits, from [Membership inference](02-membership-inference.md#sec-mia-loss). There, the mean gap over 64 models is 0.0422. So even the rule "member if correctly classified" beats chance by 0.0422, which is a lower bound on the leak.

Do not clear a model on this check, because the gap is an average over records and does not say who is exposed. For example, the loss attack flags every record with a low loss. At an FPR of $10^{-3}$ its TPR is only 0.0009, so it flags no more members than non-members. A weak attack's failure proves little. A large gap says regularize first, and a small one says only that you need the next step.

## Classifier: LiRA with shadow models {#sec-risk-lira}

The release decision rests on the likelihood ratio attack (LiRA) of [Membership inference](02-membership-inference.md#sec-mia-lira). It compares the target's output on each record with the outputs of **shadow models**, copies trained like the target with and without that record. Fix its strength before the run, because a weaker attack can pass a model that a stronger one blocks. The defaults come from [Running LiRA on your model](02-membership-inference.md#sec-mia-cifar). Use online LiRA, the strongest version, with 64 shadows when training is cheap. If you can afford only 16 or fewer, fit one variance for all records. Put each record in each model by a fair coin, and let each model take one turn as the target.

With these defaults, the digits give 57,704 member and 57,304 non-member pairs of a model and an image. [Reading the result](02-membership-inference.md#sec-mia-eval) counts the attack's hits on them. Online LiRA reaches a TPR of 0.0318 at an FPR of 0.00099. The lower end of its 95% interval is 0.0303, which is above 0.01, so the classifier fails @tbl-risk-thresholds. Offline LiRA would have passed it, and so would the shadow-model attack of the Adversarial Robustness Toolbox. So base the decision on the strongest attack you can afford.

Report the interval on the FPR as well. At $10^{-3}$, the non-member pairs allow only about 57 false positives, so the FPR's interval runs from 0.00075 to 0.00129. The script also sets thresholds and measures on the same pairs, which flatters the attack slightly. Inspect the ten records the script prints as most exposed, because such records are usually rare or mislabelled.

## Fine-tuned language model: canaries and extraction {#sec-risk-canary}

A language model can repeat a secret word for word, and a membership test does not measure that. So the assessment plants **canaries**, random secrets of a fixed format, and checks how well the trained model knows them. [Fine-tune a language model privately with LoRA](../part05-dp-ml/14-private-llm-finetuning.md#sec-llmwf-memorization) runs this audit, with the scoring of [Memorization and training-data extraction](05-memorization-extraction.md#sec-memo-exposure). It adapts Pythia-14m to 1,024 sequences of Tiny Shakespeare, once without privacy and once with DP at $\varepsilon = 8$ per sequence. Canaries with 1 to 64 copies and uninserted controls get two scores each. Exposure, in bits, ranks the secret by the sum of its tokens' log-probabilities, not their mean. Extraction checks whether greedy decoding, with the partner's settings, writes the secret out.

Read the exposure rule on the extrapolated column, not the measured one. The rank among 2,000 random candidates caps the measured exposure at 10.97 bits, while the whole space of secrets allows 16.61. So the measured column can never come within 3 bits of the maximum. First check the controls: a random rank is worth $\log_2 e = 1.44$ bits on average, and the controls' 1.4 bits match it, so the measurement is sound.

The non-private model fails @tbl-risk-thresholds. Its one-copy canaries stay far below the line, at 2.0 bits. But greedy decoding writes out all three 64-copy secrets, so the model must not be shared. The DP run extracts nothing. But its most repeated canaries still average about 4 bits, against 2.2 for its controls. The reason is that $\varepsilon$ protects one sequence, and a repeated secret is many sequences. So deduplicate before training, whatever else you do.

## Table: re-identification on quasi-identifiers {#sec-risk-table}

**Quasi-identifiers** are the columns an adversary can find elsewhere, such as age, sex and country in a register. Records that agree on all of them form an **equivalence class**, and a record alone in its class can be linked to a name. The table is Adult, a census extract of 48,842 records, and the agreed quasi-identifiers are six columns, from age to education.

[Linkage and re-identification](07-linkage-reidentification.md#sec-link-quasi) runs the uniqueness check. On the agreed six columns, 14.64% of the records are unique, so the table fails. Hours per week, a column that looks harmless, raises that to 36.55%. So list every column an adversary might hold, and when in doubt, test with the column added.

[Syntactic anonymity and how it breaks](08-syntactic-anonymity.md#sec-anon-k) then generalizes age, country and education and checks again. Generalization cuts the unique share to 0.94%, but the smallest class still has one record, so the table still fails. Suppression of the classes smaller than 5 removes 3.31% of the records. That fix works only for these columns, because raw hours per week makes 10.09% unique again. Also, 1.20% of the records sit in a class with a single occupation, a distinct $\ell$ of 1 ([Syntactic anonymity](08-syntactic-anonymity.md#sec-anon-l)). Such a class reveals the occupation without a unique match.

These shares count uniqueness in the sample, which fits the prosecutor model, where the adversary knows the target is in the table. Otherwise, estimate population uniqueness ([Linkage and re-identification](07-linkage-reidentification.md#sec-link-sampling)). Also, a table that passes after suppression can still be intersected with a second release of the same people ([Syntactic anonymity](08-syntactic-anonymity.md#sec-anon-composition)). So when a table cannot pass and stay useful, release statistics with DP, as [Release statistics with a library](../part03-dp-foundations/12-release-statistics.md) does. Plain aggregates are no fallback, because many accurate counts let an attacker rebuild records ([Reconstruction attacks](09-reconstruction.md)).

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

The residual-risk entry names the attacks that were not run, and it is the entry most often left out. Write it even when every attack fails, because a model can pass this assessment and still leak to an attack nobody tried. Only a DP guarantee bounds every attack at once. For a DP model the memo carries that guarantee and its evidence, which [Release and governance](../part05-dp-ml/12-release-governance.md) puts into the release package.

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
