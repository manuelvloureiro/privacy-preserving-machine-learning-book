# Memorization and training-data extraction

## Why this matters

A language model trained on scraped text or support tickets sees phone numbers, addresses, keys and private messages. Ask it to continue "Anna Smith's phone number is", and it may finish the sentence with the real number. This is **training-data extraction**. [Membership inference](02-membership-inference.md) needs the record in hand to ask whether it was used, while extraction hands the attacker the record itself.

One idea runs through the chapter. Take a secret of a fixed format, such as the five digits after "my pin is". Order all its possible values from the most likely to the least likely under the model. The **rank** of the true secret, its position in that order, is the number of guesses an attacker needs if it tries the values in that order. A model has memorized a string when it ranks it far above the strings that could have stood in its place.

## Which memorization to measure {#sec-memo-definitions}

The notions of memorization below give rates that differ by orders of magnitude, so a report must say which one it measures. All use greedy decoding, which writes out the most likely next token at each step.

| Definition | The string counts as memorized if | Use it for |
| --- | --- | --- |
| Discoverable (Carlini et al. 2023) | The text before it in the training data makes greedy decoding produce it | Measuring memorization on a model you own |
| Extractable (Carlini et al. 2021) | Some prompt, any at all, makes greedy decoding produce it | The real risk, which attacks find but no test measures |
| $k$-eidetic | It is extractable and appears in at most $k$ documents | Flagging the dangerous case, $k = 1$ |
| Approximate (Ippolito et al. 2023) | Greedy decoding comes within a set edit distance of it | Near copies, such as a number with one digit changed |

: The notions of memorization and when to use each. The first three are verbatim, needing an exact match, and each implies approximate. Discoverable implies extractable. {#tbl-memo-definitions}

Discoverable memorization is the one you can measure. Carlini et al. (2023) prompt with 50 tokens of a training sequence, decode the next 50 greedily and count exact matches. Treat any verbatim rate as a lower bound on the leak, because near copies leak too. Ippolito et al. (2023) show that a filter blocking verbatim training text is easy to bypass.

Deployed systems sample rather than decode greedily, which turns memorization into a matter of budget. A sample at temperature 1 writes out a string with probability $p$, the model's probability of that string after the prompt. Write $\beta$ for the chance of failure the attacker accepts. Then a number $m$ of independent samples suffices when
$$m \ge \frac{\ln(1/\beta)}{p}.$$ {#eq-memo-samples}
For example, a string of probability $10^{-3}$ comes out within about 4,600 samples, except with probability 0.01. So report a sampling budget alongside the greedy rate, and measure with the decoding settings you deploy.

## What gets memorized, and the first defences {#sec-memo-copies}

On open models, the largest reproduced at least 1% of its training data this way (Carlini et al., 2023). Memorization grew log-linearly, by a roughly fixed amount each time the number of copies of a string, the size of the model or the length of the prompt grew tenfold. Copies matter most in practice, because Kandpal, Wallace & Raffel (2022) found that a sequence repeated ten times is generated about a thousand times more often than one seen once.

That gives the order of defences. **Deduplication** is the cheapest, because it lowers the copies of every repeated string. Lee et al. (2022) found that models trained on deduplicated data emit memorized text about ten times less often without losing quality. Deduplicate before you plant the canaries of the next section, so that only they carry known copy counts.

The second defence is to remove personal data before training. A detector of personally identifiable information (PII), such as Microsoft's Presidio, finds names, phone numbers and similar fields so you can mask them. It misses whatever its patterns do not cover, so measure what remains with canaries rather than trusting the filter. Neither defence helps a secret seen once in a large model, where one copy is enough. Only [fine-tuning with differential privacy](../part05-dp-ml/14-private-llm-finetuning.md), which limits how much any one training sequence can change the model, gives a guarantee for such singletons.

## Auditing with canaries and exposure {#sec-memo-exposure}

Discoverable memorization is all or nothing, while an auditor wants a graded answer. Carlini et al. (2019) proposed planting **canaries**, random secrets of a fixed format such as "my pin is" followed by five random digits. You know each canary's value and the number of values it could have taken, so after training you can see exactly where the model ranks it.

Write $\hat r$ for the planted value, $\mathcal R$ for the set of values it could have taken and $|\mathcal R|$ for their number. The **exposure** is the number of bits of search the model saves an attacker,
$$\mathrm{exposure}(\hat r) = \log_2|\mathcal R| - \log_2\operatorname{rank}(\hat r).$$ {#eq-memo-exposure}
Each bit halves the guesses the attacker needs, and an exposure near the maximum, $\log_2|\mathcal R|$, means the secret can be extracted. For five digits the maximum is 16.61 bits.

Exposure is an audit tool because its behaviour is known when nothing was memorized. Each canary's value is drawn uniformly at random, so a model that never saw it has no reason to favour it, and it lands at a uniformly random rank. So for any threshold of $t$ bits,
$$\Pr\big[\mathrm{exposure}(\hat r) \ge t\big] \le 2^{-t}.$$ {#eq-memo-null}
Then $2^{-\text{exposure}}$ is an upper bound on the p-value, the chance that an uninserted canary would look at least this memorized. The same uniform rank gives an uninserted canary an average exposure of $\log_2 e$, about 1.44 bits, which the controls should match.

To run the audit, draw canaries in the format of the secrets your data could contain, because a format unlike your data measures how the model treats unfamiliar text. Insert each one a chosen number of times, from once to many. Keep some never inserted as controls, then fine-tune as usual. Score each canary's value, and many random values of the same format, by the model's log-probability after the prefix. The share of random values that score at least as high estimates the **rank fraction**, the rank divided by $|\mathcal R|$, and minus its base-2 logarithm is the exposure.

The scoring step is where audits most often go wrong. A candidate's log-probability is the sum over its tokens, and the mean loss that Hugging Face models return divides that sum by the number of tokens. Values that tokenize to different lengths then get scores on different scales, and the ranking is wrong. So score every candidate by its summed log-probability, and choose a format in which every candidate has the same number of tokens. The script [practitioner_lora.py](../../code/part05-dp-ml/practitioner_lora.py) fine-tunes the small language model Pythia-14m with low-rank adapters (LoRA) from the `peft` library. It separates the digits of each pin by spaces, as in "the pin of Alice is 8 6 5 2 3", so that each digit is one token.

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


def exposure(canary_score: float, candidate_scores: Scores) -> float:
    """log2 of 1 / (the share of candidates the model finds at least as likely), canary included."""
    at_least = 1 + np.sum(candidate_scores >= canary_score)
    return math.log2((len(candidate_scores) + 1) / at_least)
```

The model trains on 1,024 sequences of Tiny Shakespeare, a public extract of Shakespeare's plays. The script inserts three canaries at each copy count from 1 to 64, keeps three as controls, and ranks each value among 2,000 random candidates. The command `uv run python code/part05-dp-ml/practitioner_lora.py` runs it in about two and a half minutes on an idle CPU, and several times longer on a busy one. Each line gives a group's mean exposure, then that exposure extrapolated past the cap the candidates set (explained below), and how many of its secrets greedy decoding wrote out.

```text
non-private 1 copies: mean exposure 2.0 bits  extrapolated 2.0  extracted 0/3
non-private 4 copies: mean exposure 5.9 bits  extrapolated 5.7  extracted 0/3
non-private 16 copies: mean exposure 10.0 bits  extrapolated 13.0  extracted 0/3
non-private 64 copies: mean exposure 11.0 bits  extrapolated 16.3  extracted 3/3
non-private controls: mean exposure 1.4 bits  extrapolated 1.5  extracted 0/3
```

Exposure rises with the copies, and the controls sit at 1.4 bits, as they should. Greedy decoding extracts only the three canaries seen 64 times. Yet two of the canaries seen 4 times reach 6.9 and 8.2 bits, while no extraction test notices them. That is why an audit measures exposure and not only extraction. The canaries fill a quarter of the training set here, which lets so small a model memorize them in a short run. In the same script, fine-tuning with differential privacy, at the privacy level of 8 that Part 3 defines, extracts nothing ([Fine-tune a language model privately](../part05-dp-ml/14-private-llm-finetuning.md)).

Two more things go wrong. First, a sample of candidates caps the exposure it can resolve. A canary that beats all 2,000 candidates only shows that its rank fraction is below about 1 in 2,000. So the exposure is capped at 10.97 bits, which the 64-copy canaries reach. Past the cap, Carlini et al. (2019) fit a skew-normal distribution to the candidates' scores and read off its tail. A plain normal fit would overestimate the exposure, because the scores have a heavier right tail than a normal. Second, ties hide memorization, because when every candidate gets the same score, every exposure is 0.

A canary repeated 16 or more times is the positive control. If it does not reach a high exposure without privacy, the run is too short to tell you anything. [Pre-release privacy risk assessment](10-risk-assessment.md) sets the exposure that blocks a release and runs this audit as one of its steps.

## Extraction without the prefix: generate, rank, verify {#sec-memo-attack}

A real attacker has only the model. Carlini et al. (2021) extracted data from GPT-2 by generating many samples, ranking them by how likely each was to be memorized, and checking the top against public sources. The ranking is the hard step, because a model finds a string likely either because it is ordinary text or because it was memorized. A **reference model**, trained on similar data but not on the target's training set, stands in for ordinary text. Write $p_\theta(x)$ for the target's probability of a string $x$, and $p_{\mathrm{ref}}(x)$ for the reference's. Rank by the log-likelihood ratio
$$\Lambda(x) = \ln p_\theta(x) - \ln p_{\mathrm{ref}}(x).$$ {#eq-memo-ratio}

Raw likelihood puts first the strings that any model finds likely, such as software licences and boilerplate, which reveal nothing, and the ratio cancels them. A crude reference works. Carlini et al. (2021) used a smaller model of the same family. They also used the compressed length of the text, `len(zlib.compress(text.encode()))`, which measures how predictable the text is without any model. To audit your own fine-tuned model this way, the base model before fine-tuning is a natural reference.

An attacker without the prefix extracts less than the discoverable rate suggests, but production models are not safe. Nasr et al. (2023) asked ChatGPT to repeat one word forever, until it left the assistant behaviour that alignment training gives it and wrote memorized text. For about 200 US dollars of queries they extracted over 10,000 records, so alignment hides memorization from casual use without removing it.

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

## Further reading

- Carlini, Liu, Erlingsson, Kos & Song, "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks" (USENIX Security 2019): canaries, exposure and how to estimate it.
- Carlini, Ippolito, Jagielski, Lee, Tramèr & Zhang, "Quantifying Memorization Across Neural Language Models" (ICLR 2023): how memorization scales with duplication, model size and prompt length.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/05-memorization-extraction.md).
