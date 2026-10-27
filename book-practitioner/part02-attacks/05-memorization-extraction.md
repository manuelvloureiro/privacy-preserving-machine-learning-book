# Memorization and training-data extraction

## Why this matters

A language model trained on scraped text or support tickets sees phone numbers, addresses, keys and private messages. Asked to continue "Anna Smith's phone number is", it may finish with the real number. This is **training-data extraction**. [Membership inference](02-membership-inference.md) needs the record in hand to ask whether it was used, while extraction hands the attacker the record itself.

## Which memorization to measure {#sec-memo-definitions}

The notions of memorization in @tbl-memo-definitions give rates that differ by orders of magnitude, so a report must say which one it measures. All use greedy decoding, which writes out the most likely next token at each step.

| Definition | The string counts as memorized if | Use it for |
| --- | --- | --- |
| Discoverable (Carlini et al. 2023) | The text before it in the training data makes greedy decoding produce it | Measuring memorization on a model you own |
| Extractable (Carlini et al. 2021) | Some prompt, any at all, makes greedy decoding produce it | The real risk, which attacks find but no test measures |
| $k$-eidetic | It is extractable and appears in at most $k$ documents | Flagging the dangerous case, $k = 1$ |
| Approximate (Ippolito et al. 2023) | Greedy decoding comes within a set edit distance of it | Near copies, such as a number with one digit changed |

: The notions of memorization and when to use each. The first three are verbatim, needing an exact match, and each implies approximate. Discoverable implies extractable. {#tbl-memo-definitions}

Discoverable memorization is the one you can measure, because you hold the training text it uses as the prompt. Treat any verbatim rate as a lower bound on the leak, because near copies leak too. A filter that blocks only verbatim training text is easy to bypass (Ippolito et al. 2023).

Deployed systems sample rather than decode greedily, so whether a string comes out depends on how many samples the attacker draws. A sample at temperature 1 writes out a string with probability $p$, the model's probability of that string after the prompt. Write $\beta$ for the chance of failure the attacker accepts. Then a number $m$ of independent samples suffices when
$$m \ge \frac{\ln(1/\beta)}{p}.$$ {#eq-memo-samples}
For a string of probability $10^{-3}$ and $\beta = 0.01$, that is about 4,600 samples. So report a sampling budget alongside the greedy rate, and measure with the decoding settings you deploy.

## What gets memorized, and the first defences {#sec-memo-copies}

Memorization grows with the copies of a string. On open models, the largest reproduced at least 1% of its training data in the discoverable sense (Carlini et al. 2023). A sequence repeated ten times is generated about a thousand times more often than one seen once (Kandpal, Wallace & Raffel 2022).

So the cheapest defence is **deduplication**, which lowers the copies of every repeated string. Models trained on deduplicated data emit memorized text about ten times less often without losing quality (Lee et al. 2022). Deduplicate before you plant the canaries of the next section, so that only they carry known copy counts.

The second defence removes personal data before training. A detector of personally identifiable information (PII), such as Microsoft's Presidio, finds names, phone numbers and similar fields so you can mask them, but it misses whatever its patterns do not cover. Neither defence helps a secret seen once in a large model, where one copy is enough. Only [fine-tuning with differential privacy](../part05-dp-ml/14-private-llm-finetuning.md), which limits how much any one training sequence can change the model, gives a guarantee for such singletons.

## Auditing with canaries and exposure {#sec-memo-exposure}

Discoverable memorization is all or nothing, while an auditor wants a graded answer. Carlini et al. (2019) plant **canaries**, random secrets of a fixed format such as "my pin is" followed by five random digits. After training, you order all the values the secret could have taken from the most to the least likely under the model. The planted value's **rank** in that order is the number of guesses an attacker trying them in turn needs.

Write $\hat r$ for the planted value and $|\mathcal R|$ for the number of values in the set $\mathcal R$ it could have taken. The **exposure** is the number of bits of search the model saves an attacker,
$$\mathrm{exposure}(\hat r) = \log_2|\mathcal R| - \log_2\operatorname{rank}(\hat r).$$ {#eq-memo-exposure}
Each bit halves the guesses the attacker needs. At rank 1 the exposure reaches its maximum, $\log_2|\mathcal R|$, and the secret can be extracted. For five digits, $10^5$ values, the maximum is 16.61 bits.

Exposure works as an audit because its behaviour is known when nothing was memorized. A canary's value is drawn uniformly at random, so a model that never saw it puts it at a uniformly random rank. So for any threshold of $t$ bits,
$$\Pr\big[\mathrm{exposure}(\hat r) \ge t\big] \le 2^{-t}.$$ {#eq-memo-null}
So $2^{-\text{exposure}}$ bounds the p-value, the chance that an uninserted canary would look at least this memorized. The same uniform rank gives an uninserted canary an average exposure of $\log_2 e$, about 1.44 bits, which the controls should match.

To run the audit, draw canaries in the format of the secrets your data could contain, since a format unlike your data measures how the model treats unfamiliar text. Insert each one a chosen number of times, from once to many, keep some never inserted as controls, and fine-tune as usual. Then score each canary's value, and many random values of the same format, by the model's log-probability after the prefix. The share of random values that score at least as high estimates the **rank fraction**, the rank divided by $|\mathcal R|$, and minus its base-2 logarithm is the exposure.

Scoring is where audits most often go wrong. A candidate's log-probability is the sum over its tokens, but the mean loss that Hugging Face models return divides that sum by the number of tokens. Values of different token lengths then get scores on different scales, and the ranking is wrong. So score every candidate by its summed log-probability, in a format where every candidate has the same number of tokens. The script [practitioner_lora.py](../../code/part05-dp-ml/practitioner_lora.py) fine-tunes the small language model Pythia-14m with low-rank adapters (LoRA) from `peft`. It spaces the digits of each pin, as in "the pin of Alice is 8 6 5 2 3", so that each digit is one token. The exposure functions come from the book's module [practitioner_privacy_tests.py](../../code/practitioner_privacy_tests.py), which follows TensorFlow Privacy.

```{.python excerpt="lora-exposure"}
@torch.no_grad()
def secret_scores(model: PeftModel, tokenizer: PreTrainedTokenizerBase, prefix: str, secrets: list[str]) -> Scores:
    """Summed log-probability of each secret's five digit tokens after the prefix."""
    prefix_ids = tokenizer(prefix)["input_ids"]
    rows = [prefix_ids + tokenizer(" " + s)["input_ids"] for s in secrets]
    ids = torch.tensor(rows)  # every secret is five tokens, so the rows have equal length
    logits = model(input_ids=ids[:, :-1], logits_to_keep=5).logits  # the predictions of the five digits
    logprobs = logits.log_softmax(dim=-1).gather(-1, ids[:, -5:, None])
    return logprobs.sum(dim=(1, 2)).double().numpy()


def canary_exposure(
    model: PeftModel, tokenizer: PreTrainedTokenizerBase, canary: Canary, candidates: list[str]
) -> tuple[float, float]:
    """The exposure of the canary's secret among the candidates, measured and extrapolated past the cap."""
    candidate_scores = scores_batched(model, tokenizer, canary.prefix, candidates)
    score = float(scores_batched(model, tokenizer, canary.prefix, [canary.secret])[0])
    return exposure(score, candidate_scores), extrapolated_exposure(score, candidate_scores, SECRET_SPACE)
```

The model trains on 1,024 sequences of Tiny Shakespeare, a public extract of Shakespeare's plays. The script inserts three canaries at each copy count from 1 to 64, keeps three as controls, and ranks each value among 2,000 random candidates. The command `uv run python code/part05-dp-ml/practitioner_lora.py` runs it in about two and a half minutes. Each line gives a group's mean exposure, that exposure extrapolated past the cap the candidates set (explained below), and how many of its secrets greedy decoding wrote out.

```text
non-private 1 copies: mean exposure 2.0 bits  extrapolated 2.0  extracted 0/3
non-private 4 copies: mean exposure 5.9 bits  extrapolated 5.7  extracted 0/3
non-private 16 copies: mean exposure 10.0 bits  extrapolated 13.0  extracted 0/3
non-private 64 copies: mean exposure 11.0 bits  extrapolated 16.3  extracted 3/3
non-private controls: mean exposure 1.4 bits  extrapolated 1.5  extracted 0/3
```

Exposure rises with the copies, and the controls sit at 1.4 bits, as the 1.44 expected for them predicts. Greedy decoding extracts only the three canaries seen 64 times. Yet two of the canaries seen 4 times reach 6.9 and 8.2 bits, which no extraction test notices, so an audit measures exposure and not only extraction. The canaries fill a quarter of the training set here, which lets so small a model memorize them in a short run. In the same script, fine-tuning with differential privacy at a privacy level $\varepsilon = 8$ extracts nothing ([Fine-tune a language model privately](../part05-dp-ml/14-private-llm-finetuning.md)).

The cap comes from sampling the candidates. A canary that beats all 2,000 shows only that its rank fraction is below about 1 in 2,000. So its exposure stops at $\log_2 2000 = 10.97$ bits, which the 64-copy canaries reach. Past the cap, the script fits a skew-normal distribution to the candidates' scores and reads the canary's rank off its tail, as Carlini et al. (2019) do. Ties hide memorization too, because when every candidate gets the same score, every exposure is 0.

A canary repeated 16 or more times is the positive control: if it does not reach a high exposure without privacy, the run is too short to tell you anything. [Pre-release privacy risk assessment](10-risk-assessment.md) sets the exposure that blocks a release.

## Extraction without the prefix: generate, rank, verify {#sec-memo-attack}

A real attacker has only the model. Carlini et al. (2021) extracted data from GPT-2 by generating many samples, ranking them by how likely each was to be memorized, and checking the top ones. Ranking is the hard step, because a model finds a string likely either because it is ordinary text or because it memorized it. A **reference model**, trained on similar data but not on the target's training set, stands in for ordinary text. Write $p_\theta(x)$ for the target model's probability of a string $x$, and $p_{\mathrm{ref}}(x)$ for the reference's. Rank by the log-likelihood ratio
$$\Lambda(x) = \ln p_\theta(x) - \ln p_{\mathrm{ref}}(x).$$ {#eq-memo-ratio}

The ratio cancels the strings that any model finds likely, such as software licences and boilerplate, which raw likelihood would put first and which reveal nothing. A crude reference, such as a smaller model of the same family, works, and to audit your own fine-tuned model, the base model before fine-tuning is a natural one.

Without the prefix, an attacker extracts less than the discoverable rate suggests, but aligned production models are not safe. Nasr et al. (2023) asked ChatGPT to repeat one word forever, until it left its aligned behaviour and wrote memorized text. For about 200 US dollars of queries, they extracted over 10,000 records.

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
