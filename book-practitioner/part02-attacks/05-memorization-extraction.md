# Memorization and training-data extraction

## Why this matters

A language model trained on scraped text or support tickets may learn phone numbers, addresses, keys and private messages. Given the prompt "Anna Smith's phone number is", it may generate the real number. This is **training-data extraction**. [Membership inference](02-membership-inference.md) tests whether the model used a specified record; extraction recovers the record itself.

## Which memorization to measure {#sec-memo-definitions}

The memorization definitions in @tbl-memo-definitions produce rates that differ by orders of magnitude. Reports must specify which definition they use. Each definition uses greedy decoding, which selects the most likely next token at each step.

| Definition | The string counts as memorized if | Use it for |
| --- | --- | --- |
| Discoverable (Carlini et al. 2023) | The text before it in the training data makes greedy decoding produce it | Measuring memorization on a model you own |
| Extractable (Carlini et al. 2021) | Some prompt, any at all, makes greedy decoding produce it | The real risk, which attacks find but no test measures |
| $k$-eidetic | It is extractable and appears in at most $k$ documents | Flagging the dangerous case, $k = 1$ |
| Approximate (Ippolito et al. 2023) | Greedy decoding comes within a set edit distance of it | Near copies, such as a number with one digit changed |

: The notions of memorization and when to use each. The first three are verbatim, needing an exact match, and each implies approximate. Discoverable implies extractable. {#tbl-memo-definitions}

Discoverable memorization is measurable because the auditor has the training text used as a prompt. A verbatim rate is a lower bound on leakage because near copies may also reveal information. A filter that blocks only verbatim text can be bypassed (Ippolito et al. 2023).

Deployed systems sample rather than decode greedily, so extraction probability depends on the attacker's sample count. At temperature 1, let $p$ be the model's probability of generating the string after the prompt and let $\beta$ be the attacker's tolerated failure probability. At least $m$ independent samples are sufficient when
$$m \ge \frac{\ln(1/\beta)}{p}.$$ {#eq-memo-samples}
For a string with probability $10^{-3}$ and $\beta = 0.01$, the requirement is about 4,600 samples. Report the sampling budget with the greedy rate and use the deployed decoding settings.

## What gets memorized, and the first defences {#sec-memo-copies}

Memorization increases with the number of copies of a string. In open models, the largest reproduced at least 1% of the training data under the discoverable definition (Carlini et al. 2023). A sequence repeated ten times is generated about a thousand times more often than one seen once (Kandpal, Wallace & Raffel 2022).

The lowest-cost defence is **deduplication**, which reduces repeated copies. Models trained on deduplicated data emit memorized text about ten times less often at the same quality (Lee et al. 2022). Deduplicate before inserting canaries so their copy counts are controlled.

The second defence removes personal data before training. A personally identifiable information (PII) detector, such as Microsoft's Presidio, identifies fields such as names and phone numbers for masking, but can miss patterns it does not cover. Neither deduplication nor detection protects a secret that appears once in a large model. [Fine-tuning with differential privacy](../part05-dp-ml/14-private-llm-finetuning.md) limits how much one training sequence can change the model and provides a guarantee for such singletons.

## Auditing with canaries and exposure {#sec-memo-exposure}

Discoverable memorization is binary, but an audit needs a graded measure. Carlini et al. (2019) insert **canaries**, random secrets with a fixed format, such as "my pin is" followed by five random digits. After training, rank all possible secret values by model likelihood. The planted value's **rank** is the number of guesses required if an attacker tests them in order.

Write $\hat r$ for the planted value and $|\mathcal R|$ for the number of values in the set $\mathcal R$ it could have taken. The **exposure** is the number of bits of search the model saves an attacker,
$$\mathrm{exposure}(\hat r) = \log_2|\mathcal R| - \log_2\operatorname{rank}(\hat r).$$ {#eq-memo-exposure}
Each exposure bit halves the number of required guesses. Rank 1 gives maximum exposure $\log_2|\mathcal R|$ and allows extraction. For five digits, there are $10^5$ values and maximum exposure is 16.61 bits.

Exposure supports an audit because its null behaviour is known. A uniformly random canary that the model never saw has a uniformly random rank. Therefore, for any threshold of $t$ bits,
$$\Pr\big[\mathrm{exposure}(\hat r) \ge t\big] \le 2^{-t}.$$ {#eq-memo-null}
Thus, $2^{-\text{exposure}}$ bounds the p-value: the probability that an uninserted canary appears at least this memorized. A uniform rank gives an uninserted canary mean exposure $\log_2 e$, about 1.44 bits; controls should be near this value.

Draw canaries in the format of secrets present in the data; an unrelated format tests only how the model treats unfamiliar text. Insert each canary a controlled number of times, retain uninserted controls and fine-tune normally. Score each canary and many random values in the same format using their model log-probability after the prefix. The fraction of random values with at least as high a score estimates the **rank fraction**, the rank divided by $|\mathcal R|$. Its negative base-2 logarithm estimates exposure.

Scoring errors can invalidate the audit. Candidate log-probability is the sum over tokens, whereas Hugging Face models return mean loss divided by token count. Scores for candidates with different token lengths are therefore not comparable. Use summed log-probabilities and a format that gives every candidate the same number of tokens. For example, write a pin as "the pin of Alice is 8 6 5 2 3", with each digit represented by one token.

[Fine-tune a language model privately with LoRA](../part05-dp-ml/14-private-llm-finetuning.md#sec-llmwf-memorization) runs this audit on Pythia-14m. It inserts three canaries at each copy count from 1 to 64 into 1,024 Tiny Shakespeare sequences, keeps three controls and ranks each value among 2,000 random candidates. Exposure increases with copy count. Controls have 1.4 bits of exposure, near the expected 1.44. Greedy decoding extracts only the three canaries repeated 64 times, while two canaries repeated four times reach 6.9 and 8.2 bits. Exposure therefore detects risk that extraction tests miss.

The 2,000 candidates cap the measured exposure. If a canary outranks all candidates, its rank fraction is only known to be below about 1 in 2,000, so measured exposure is capped at $\log_2 2000 = 10.97$ bits. To estimate exposure beyond this cap, fit a skew-normal distribution to candidate scores and estimate the canary's rank from its tail, as Carlini et al. (2019) do. Ties also hide memorization: if all candidates have the same score, each has exposure 0.

Use a canary repeated 16 or more times as a positive control. If it does not reach high exposure without privacy, the run is too short to assess memorization. In the cited run, canaries occupy one quarter of the training set, allowing this small model to memorize them. [Pre-release privacy risk assessment](10-risk-assessment.md) specifies the exposure threshold for blocking release.

## Extraction without the prefix: generate, rank, verify {#sec-memo-attack}

A real attacker has access only to the model. Carlini et al. (2021) extracted data from GPT-2 in three steps: they generated many samples, ranked them by likelihood and checked the highest-ranked candidates for memorization. Ranking is difficult because a model may assign high probability to a string either because it is ordinary text or because it memorized it. A **reference model**, trained on similar data but not on the target's training set, estimates ordinary-text likelihood. Write $p_\theta(x)$ for the target model's probability of a string $x$, and $p_{\mathrm{ref}}(x)$ for the reference model's probability. Rank by the log-likelihood ratio
$$\Lambda(x) = \ln p_\theta(x) - \ln p_{\mathrm{ref}}(x).$$ {#eq-memo-ratio}

The ratio discounts strings that any model finds likely, such as software licenses and boilerplate. Raw likelihood ranks these strings highly even though they reveal no memorization. A smaller model from the same family can serve as the reference. For a fine-tuned model, use its base model.

Without the prefix, attackers extract less than the discoverable rate suggests, but alignment does not prevent extraction. Nasr et al. (2023) prompted ChatGPT to repeat one word until it stopped following its aligned behaviour and generated memorized text. They extracted over 10,000 records using about 200 US dollars of queries.

## Cheat sheet

| Question | Answer |
| --- | --- |
| Exposure of a canary | $\log_2\lvert\mathcal R\rvert - \log_2\operatorname{rank}$, at most 16.61 bits for five digits; p-value at most $2^{-\text{exposure}}$; controls average 1.44 bits |
| How to score a candidate | Summed log-probability, in a format where all candidates have the same token count |
| Ranking generations | $\ln p_\theta(x) - \ln p_{\mathrm{ref}}(x)$, never raw likelihood |
| Defences, in order | Deduplicate (`deduplicate-text-datasets`, `datasketch`), scrub PII, then differential privacy for singletons |

: The results of this chapter. {#tbl-memo-cheat}

**Rules of thumb**

- A verbatim rate is a lower bound on the leak, not a measure of it.

## References

- Carlini, Liu, Erlingsson, Kos & Song, "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks" (USENIX Security 2019): canaries, exposure and how to estimate it.
- Carlini, Ippolito, Jagielski, Lee, Tramèr & Zhang, "Quantifying Memorization Across Neural Language Models" (ICLR 2023): how memorization scales with duplication, model size and prompt length.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/05-memorization-extraction.md).
