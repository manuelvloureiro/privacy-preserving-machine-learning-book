# Release and governance

## Why this matters

A model trained with differential privacy (DP) is not finished when training stops. A privacy reviewer or a data protection officer decides whether it may ship. They ask what the guarantee covers, whether its number is right and what it means under the law, and none of them can rerun your training. So the claim must travel with everything needed to check it. This chapter is the book's home for the privacy statement and for the review of someone else's claim.

The model is a small network, trained on 30,000 people from Adult with the private form of stochastic gradient descent (SGD), called DP-SGD. It comes from [Train a model with DP-SGD](13-train-dp-sgd.md), with a target privacy level $\varepsilon$ of 4. The script [practitioner_release.py](../../code/part05-dp-ml/practitioner_release.py) runs in a few seconds. It reads the run record that the training script writes, passed with `--record PATH`, or else a built-in copy of the book's record.


```bash
uv run python code/part05-dp-ml/practitioner_release.py
```

## Assemble the run record {#sec-relgov-record}

The record must hold what the accountant needs. The accountant is the code that turns a run's settings into $\varepsilon$ ([Choosing an accountant](../part04-privacy-accounting/09-choosing-an-accountant.md)). It reads four settings, the noise multiplier $\sigma$, the sampling rate $q$, the number of steps and the slack $\delta$. It also takes four things for granted, so the record states those too. They are the privacy unit, here one person, the neighbouring relation and the sampler that drew each batch. The fourth is that no other use of the private data went uncounted.

The record also gives $n$, the number of units, the seed, Opacus's `secure_mode` and the library versions. It says how the hyperparameters were chosen and whether any preprocessing saw private data. In the excerpt, `data` is the training script's record and `RELEASE_CONTEXT` holds the release owner's additions.

```{.python excerpt="release-record-load"}
merged = RELEASE_CONTEXT | data  # the training record, plus what the release owner adds
rec = load_record(merged)
for key, value in asdict(rec).items():
    print(f"{key}: {value}")
problems = record_problems(merged)  # missing fields, sampler not Poisson, delta above 1/n^1.1, tuning
limitations = record_limitations(merged)  # stated, so not a problem, but the statement must disclose them
print(f"record problems: {len(problems)}")
for problem in problems:
    print(f"  - {problem}")
print(f"record limitations: {len(limitations)}")
for limitation in limitations:
    print(f"  - {limitation}")
```

```text
unit: person (one row of Adult)
neighbours: add/remove
n: 30000
delta: 1e-05
sigma: 0.972900390625
sample_rate: 0.03333333333333333
sampler: poisson
secure_mode: False
seed: 0
steps: 300
...
record problems: 0
record limitations: 1
  - secure_mode=False: Opacus drew the noise from a seeded, non-cryptographic generator so the numbers reproduce; floating-point attacks on the noise are not excluded. Retrain with secure_mode=True to ship
```

The record's $\delta$ of $10^{-5}$ follows the book's rule ([How small δ must be](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta)). Its values of $\varepsilon$ come from the cross-check of [Calibrate and account a DP-SGD run](../part04-privacy-accounting/10-calibrate-dp-sgd.md#sec-calib-crosscheck).

The tuning entry matters most to a reviewer, because the final run's $\varepsilon$ leaves out every run that helped choose a hyperparameter. Here non-private runs chose the learning rate on 2,000 held-out Adult rows that play the part of public data, and $\varepsilon = 4$ was fixed before any private run. So the run's $\varepsilon$ is also the $\varepsilon$ of the whole procedure. Those 2,000 people come from the same survey and get no guarantee, so the statement names them. If you tune on private data, report both values ([Hyperparameter tuning](03-hyperparameter-tuning.md#sec-hpt-practice)).

A record problem stops the release, such as a missing field, a sampler that is not Poisson, a $\delta$ above the rule or unreported tuning. The one limitation here is `secure_mode`, left off so that the book's numbers repeat. The statement discloses it, and a model you ship is retrained with it on.

## Recompute the guarantee {#sec-relgov-recompute}

The release owner recomputes $\varepsilon$ from $\sigma$, $q$ and the steps alone, with a library other than the one that trained. A recomputation inside the training code would reuse any wrong number. Compare the result only with a budget from the same kind of accountant, because a looser accountant gives a larger $\varepsilon$ for the same run. The script uses the privacy loss distribution (PLD) accountant of `dp_accounting`, as in [the cross-check](../part04-privacy-accounting/10-calibrate-dp-sgd.md#sec-calib-crosscheck), and compares it with the looser Rényi differential privacy (RDP) accountant. It also reads $\varepsilon$ at a $\delta$ a hundred times smaller, and the trade-off curve. That curve gives the largest true positive rate (TPR) that any membership test can reach at each false positive rate (FPR) ([Choosing ε](09-choosing-epsilon.md#sec-epsc-curve)).

```{.python excerpt="release-recompute"}
pld = privacy_loss_distribution.from_gaussian_mechanism(sigma, sampling_prob=sample_rate).self_compose(steps)
eps_pld = pld.get_epsilon_for_delta(delta)
eps_pld_small = pld.get_epsilon_for_delta(delta / 100)
eps_rdp = rdp_epsilon(sigma, sample_rate, steps, delta)
eps_rdp_small = rdp_epsilon(sigma, sample_rate, steps, delta / 100)
tpr = {fpr: tpr_bound(pld, fpr) for fpr in FPRS}  # largest TPR of any membership test
```

```text
eps PLD at delta=1e-05: 3.981
eps RDP at delta=1e-05: 4.503
eps PLD at delta=1e-07: 5.191
eps RDP at delta=1e-07: 5.736
TPR bound at FPR 1e-03: 0.0123 (PLD curve)  0.0536 (pair eps, delta only)
TPR bound at FPR 1e-02: 0.0652 (PLD curve)  0.5355 (pair eps, delta only)
record's PRV eps: reported 3.99, recomputed 3.98: consistent
```

The PLD value, 3.981, agrees with the reported 3.99. The curve gives a number a non-specialist can use. At an FPR of $10^{-3}$, no membership test can reach a TPR above 0.0123, while the pair $(3.98, 10^{-5})$ on its own would allow 0.0536.

The script flags a reported $\varepsilon$ more than 0.05 below the recomputation. The subtler failure is a recomputation that agrees on wrong inputs, such as code that shuffles the data into fixed batches while the accountant assumes Poisson sampling ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md)). No number shows that, so the record's sampler, read from the loader, is what catches it.

## Attach the evidence {#sec-relgov-evidence}

An accountant proves a theorem about an algorithm, not about the code that runs it, so the release also needs evidence about the code and the model. Attach each piece in @tbl-relgov-evidence, and say which attacker it models. The curve covers the strongest attacker, who sees every intermediate model, while the audit and the attack cover only the attacks they run.

| Evidence | Source | On the Adult run | What it cannot show |
| --- | --- | --- | --- |
| Independent $\varepsilon$ | This chapter's recomputation | 3.981 against the reported 3.99 | That the sampler matches the code |
| Clipping and noise test | Continuous integration (CI) test of [Auditing private training](10-auditing.md#sec-audit-ci) | Passes on Opacus's optimizer; fails with the variance passed as the standard deviation, or with no noise | Sampling or accounting bugs |
| One-run audit | [Auditing private training](10-auditing.md#sec-audit-onerun): the released pipeline retrained with 1,000 planted canaries | A lower bound on $\varepsilon$ of 0.00 for DP-SGD and for the non-private baseline, against 0.53 for a memorizing control | That the model is private; a low bound proves nothing |
| Black-box attack | Likelihood ratio attack of [Pre-release privacy risk assessment](../part02-attacks/10-risk-assessment.md#sec-risk-lira) | Not run on this model; say so in the evidence | Whether a stronger attacker does better |
| Groups | Per-group accuracy with exact 95% intervals ([Disparate impact](08-disparate-impact.md)) | Women 0.922 [0.915, 0.929], men 0.807 [0.800, 0.815] | Groups too small to measure |

: The evidence attached to a DP model release, and its limits. {#tbl-relgov-evidence}

The audits can only refute a claim. A one-run bound above the claimed 4 would prove a bug, but 0.00 shows only that this attack found nothing, as in the non-private baseline. The audit also tests the released training code, not the released weights.

The group accuracies come from the 16,842 Adult test rows, which are private too. To publish them, add noise to each group's count of correct predictions and charge it to its own budget ([Release statistics with a library](../part03-dp-foundations/12-release-statistics.md)). Or measure them on a test set that is not private.


## Write the statement {#sec-relgov-statement}

The statement goes in the model card's privacy section. The script's `format_statement` builds it from the record, so no number is typed by hand. The block omits lines that repeat the record.

```text
Guarantee: (3.98, 1e-05)-DP per person (one row of Adult), add/remove neighbours,
  for the whole training run on Adult (UCI 2), 30,000 training rows (n = 30,000).
  Also (5.19, 1e-07)-DP. Any membership test has TPR at most 0.0123 at FPR 1e-3.
Mechanism: DP-SGD, poisson sampling q = 0.0333, noise multiplier 0.9729,
  clipping norm 1, 300 steps (10 epochs).
Accounting: prv, Opacus 1.6.0; recomputed with dp_accounting PLD (RDP gives 4.50).
...
Noise generator: Opacus secure_mode=False, seed 0.
...
Not covered: data outside the training set, API outputs that use other private data, other releases.
```

Below the statement, add the evidence of @tbl-relgov-evidence. The last line needs the most care. The guarantee covers the weights and anything computed from them alone, such as the logits that the deployed application programming interface (API) returns, because post-processing cannot weaken DP. An output that also uses other private data, such as a retrieval index, is a new release, and so is a second model trained on the same people.

## Map the claim to regulation {#sec-relgov-law}

This section is not legal advice, since every answer depends on facts beyond the model. In the European Union, Recital 26 of the General Data Protection Regulation (GDPR) exempts anonymous information. It asks about the means of identification "reasonably likely to be used". Pseudonymised data, which only information kept separately links to a person, remain personal data under Article 4(5).

For AI models, Opinion 28/2024 of the European Data Protection Board (EDPB) sets a test, assessed case by case. A model is anonymous only if the likelihood of extracting personal data about its training set is insignificant. The opinion names DP and attack tests, such as membership inference, as measures to weigh, and warns that a test shows resistance only to the attacks tested. The trade-off curve speaks to extraction as well as membership, because it caps reconstruction too ([Choosing ε](09-choosing-epsilon.md#sec-epsc-curve)).

| Legal question | What DP evidence supports | What DP does not answer |
| --- | --- | --- |
| Is the model anonymous (GDPR Recital 26; EDPB Opinion 28/2024)? | A bound on what any attacker can learn about one person's presence or record: the TPR cap and the audits | Whether the bound is "insignificant"; the training data stay personal data |
| Is this only pseudonymisation (GDPR Art. 4(5))? | DP relies on no separately kept key that could re-link the data to people | Removing identifiers before training is pseudonymisation at most |
| What goes in the risk assessment (GDPR Art. 35)? | The statement, the curve, the audits and the group table | Necessity and proportionality (Art. 35(7)(b)), and the lawful basis (Art. 6) |
| Is the risk of identification very small (HIPAA §164.514(b)(1))? | A quantitative bound that an expert can cite, with documented methods | Whether it is "very small"; the expert's determination itself |
| Is the information deidentified (California Consumer Privacy Act, Cal. Civ. Code §1798.140(m))? | A bound on how much one person's record can change what is inferred | Whether that meets the standard, since DP does not stop inferences from population patterns; the business's own duties |
| Does the data governance meet the EU AI Act (Art. 10)? | Documented data handling and per-group evaluation | Data quality, representativeness and bias, which DP can worsen |

: What DP evidence answers, and leaves open, for the legal questions a release meets. {#tbl-relgov-law}

Training still processes personal data, so the GDPR applies to it, lawful basis included. Article 35 requires a data protection impact assessment (DPIA) before processing likely to result in a high risk. The DPIA is the natural home for the statement. In the United States, the safe harbor route of the Health Insurance Portability and Accountability Act (HIPAA) removes listed identifiers (§164.514(b)(2)), which DP does not do. Its expert determination route can cite DP evidence but not be replaced by it. California's law, as amended by the California Privacy Rights Act, adds duties against re-identification for the business.

None of these texts says that DP makes a model anonymous by itself. A DPIA entry built from this chapter's evidence can read as follows, and the controller and counsel fill in the bracketed parts.

> The model was trained on personal data under [lawful basis, Art. 6]. Training used DP-SGD with a guarantee of $(3.98, 10^{-5})$-DP per person, add/remove neighbours, for the whole training run. No membership test against the released weights can exceed a TPR of 0.0123 at an FPR of $10^{-3}$, and the same curve caps reconstruction. The code passed a CI test of its noise and clipping. A one-run audit of the released pipeline found nothing, while a memorizing control certified 0.53. [A shadow-model attack was / was not run.] These tests show resistance only to the attacks run. DP does not remove identifiers and is not pseudonymisation. Whether the bound makes the model anonymous is for [counsel], and we make no claim of automatic anonymity. Not covered: the 2,000 stand-in rows, the test rows, outputs that use other private data, and later model versions.

## Review someone else's claim {#sec-relgov-review}

When someone else hands you a model, run the same steps in reverse. Ask for the record of @sec-relgov-record, and refuse a claim that cannot be recomputed. Recompute $\varepsilon$ from the claim's own parameters, and check its $\delta$ against the number of units. [Matching the accountant to the code](../part04-privacy-accounting/09-choosing-an-accountant.md#sec-choose-match) gives the checks on the accountant and the sampler.

The script reviews a claim made up for this book, with every flaw a review should catch. It claims $\varepsilon = 1$ at $\delta = 10^{-3}$ for a model trained on 50,000 records, over 5,000 steps with noise multiplier 0.8. Its batches of 256 were shuffled and of fixed size but are accounted as Poisson sampling. The claim states no unit, no clipping norm and no tuning.

```{.python excerpt="release-review"}
def review(claim: dict[str, Any]) -> list[str]:
    """Recompute a claim from its own parameters and list what does not hold up."""
    findings = record_problems(claim)  # includes a sampler that is missing or not Poisson
    sigma, q, steps, delta = claim["sigma"], claim["sample_rate"], claim["steps"], claim["delta"]
    pld = privacy_loss_distribution.from_gaussian_mechanism(sigma, sampling_prob=q).self_compose(steps)
    recomputed = {"PLD": pld.get_epsilon_for_delta(delta)}
    if "rdp" in str(claim.get("accountant")).lower():  # also the accountant the claim names
        recomputed["RDP, the accountant named,"] = rdp_epsilon(sigma, q, steps, delta)
    for name, eps in recomputed.items():
        if eps > claim["epsilon"] + 0.05:
            findings.append(f"eps={claim['epsilon']} does not follow: {name} gives {eps:.2f} at its delta")
    delta_ok = delta_for(claim["n"])
    findings.append(f"at delta={delta_ok:.0e} (1/n^1.1) PLD gives eps={pld.get_epsilon_for_delta(delta_ok):.2f}")
    return findings
```

```text
claim: Our model is trained with differential privacy, epsilon = 1.
findings: 10
  - missing field: unit
  - missing field: neighbours
  - missing field: max_grad_norm
  - missing field: secure_mode
  - sampler 'shuffled fixed-size batches' is not Poisson subsampling: Poisson-subsampled accounting does not apply
  - delta=1e-03 is above 1/n^1.1 (use 1e-06)
  - tuning not reported: the cost of choosing hyperparameters is unknown
  - eps=1.0 does not follow: PLD gives 2.17 at its delta
  - eps=1.0 does not follow: RDP, the accountant named, gives 2.55 at its delta
  - at delta=1e-06 (1/n^1.1) PLD gives eps=3.75
```

By the vendor's own parameters the claimed 1 is 2.17, and the RDP accountant it names gives 2.55. At a $\delta$ the book accepts it is 3.75. Even that is too low, because shuffled fixed-size batches get no amplification by Poisson subsampling, so the true $\varepsilon$ can be far above all these numbers.

The script takes $n$ as given, so it cannot see a wrong unit. If the records are messages and a customer sends many, screen the per-customer $\varepsilon$ yourself with [User-level differential privacy](07-user-level.md#sec-ulvl-code). Send the findings back with a request for a corrected record.

## After release {#sec-relgov-after}

Every retraining on the same people is a new release that composes with the old ones. For identical runs, compose the PLD of one run with itself once per version, `pld.self_compose(steps * versions)`. State the budget per model version and in total over a fixed **period**. If the total over a schedule is too large, calibrate each version to a share of it.

Deletion requests need their own answer, because DP deletes nothing. The model is about as likely as one trained without the person, but their row is still in the data. So honour a request by removing the row, from the stand-in and test rows too, and retraining on a fixed schedule. Whether that is fast enough for the GDPR's Article 17 is a question for counsel.

Give each model version its own record and statement, and name in advance the incidents that reopen a release. A failing CI test, an audit bound above the claim, or a bug in the sampler or the accountant needs a corrected statement or a retrained model.

## Checklist

| Step | Default | Check |
| --- | --- | --- |
| Run record | The fields of @sec-relgov-record, with the sampler read from the loader | `record problems: 0`; `secure_mode` on for the shipped model |
| $\delta$ | The book's rule ([How small δ must be](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta)) | $n$ counts units, not rows |
| Tuning | Choices on public data, or every private run counted | Run and procedure $\varepsilon$ both reported |
| Recomputed $\varepsilon$ | PLD in `dp_accounting`, at $\delta$ and $\delta/100$, and TPR from the curve | Reported value not below the recomputation |
| Audits | The evidence of @tbl-relgov-evidence | CI passes on the released commit; no bound above the claim; the control detects leakage; any attack not run is named |
| Groups | Per-group accuracy with exact intervals | Test set not private, or its release accounted |
| Statement | Generated from the record | Every number recomputable; "not covered" filled in |
| Regulation | DPIA entry mapping the evidence to each legal question | Reviewed by counsel; no claim of automatic anonymity |
| Third-party claim | Recompute from its own parameters | Unit, sampler, $\delta$ and tuning all stated |
| After release | Budget per period; retraining and deletion schedule; incident triggers | Versions composed; each has its own record |

: The release checklist: each step, its default and its check. {#tbl-relgov-checklist}

## References

- Near, Darais, Lefkovitz & Howarth, *Guidelines for Evaluating Differential Privacy Guarantees* (NIST SP 800-226, 2025): what to ask of any DP claim.
- European Data Protection Board, *Opinion 28/2024 on certain data protection aspects related to the processing of personal data in the context of AI models* (December 2024): when a model can count as anonymous.
- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023): its reporting section lists what a privacy statement contains.

**Derivations.** This workflow has no counterpart in the full edition.
