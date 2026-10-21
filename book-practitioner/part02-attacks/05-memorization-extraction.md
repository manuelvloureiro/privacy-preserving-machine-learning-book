# Memorization and training-data extraction

## Why this matters

A language model trained on scraped text sees phone numbers, addresses, keys and private messages. Ask it to continue "Anna Smith's phone number is", and it may finish the sentence with the real number. Nobody queried a database, and no breach leaked anything, because the number is stored in the model's weights and comes out through its normal interface. This is **training-data extraction**. [Membership inference](02-membership-inference.md) needs the record in hand to ask whether it was used, while extraction hands the attacker the record itself.

One idea runs through the chapter. Fix the context in which a secret appears, say "my pin is", and list every possible filler from the most likely to the least likely under the model. The **rank** of the true secret is its position in that list, which is also the number of guesses an attacker needs if it tries the fillers in that order. A model has memorized a string when it ranks it far above the strings that could have stood in its place.

## Which memorization to measure {#sec-memo-definitions}

The notions of memorization below give rates that differ by orders of magnitude, so a report must say which one it measures. All use greedy decoding, which writes out the most likely next token at each step.

| Definition | The string counts as memorized if | Attacker it fits | Use it for |
| --- | --- | --- | --- |
| Discoverable (Carlini et al. 2023) | Its training prefix makes greedy decoding produce it | One who holds the prefix | Measuring memorization on a model you own |
| Extractable (Carlini et al. 2021) | Some prompt, any at all, makes greedy decoding produce it | One who has only the model | The real risk; found by attacks, not measured |
| $k$-eidetic | It is extractable and appears in at most $k$ documents | As extractable | Flagging the dangerous case, $k = 1$ |
| Approximate (Ippolito et al. 2023) | Greedy decoding comes within a set edit distance of it | Any reader of the output | Near copies, such as a number with one digit changed |

: The notions of memorization and when to use each. Discoverable implies extractable, and verbatim implies approximate. {#tbl-memo-definitions}

Discoverable memorization is the one you can measure. Carlini et al. (2023) prompt with 50 tokens of a training sequence, decode 50 tokens greedily and count exact matches. Treat any verbatim rate as a lower bound on the leak. Ippolito et al. (2023) show that a filter blocking verbatim training text is easy to bypass with prompts that change the style, while approximate copies still come out.

Deployed systems sample rather than decode greedily, which turns memorization into a matter of budget. A sample at temperature 1 writes out a string $x$ after a prompt with the model's probability $p$ of that string. Write $\beta$ for the chance of failure the attacker accepts. Then a number $m$ of independent samples suffices when
$$m \ge \frac{\ln(1/\beta)}{p}.$$ {#eq-memo-samples}
For example, a string of probability $10^{-3}$ needs about 4,600 samples for 99% confidence. So report a sampling probability or a budget alongside the greedy rate, because a string that greedy decoding misses can still come out after a few thousand queries. Measure with the temperature and decoding settings you deploy.

## What gets memorized: copies, model size and rare data {#sec-memo-copies}

Carlini et al. (2023) ran that test on open models, and the largest reproduced at least 1% of its training data. Memorization grew log-linearly with three things, so it rises by about the same amount each time one of them is multiplied by a fixed factor.

The first is the number of copies of a string. Kandpal, Wallace & Raffel (2022) found that a sequence repeated ten times is generated about a thousand times more often than one seen once.

The second is model size. A larger model separates contexts more finely, so rare strings end up with parameters of their own, and they are memorized from fewer copies. The third is prompt length, since a longer prefix pins down the context. Rare data are exposed first, because an atypical example can be fitted only by memorizing it ([Algorithmic stability and generalization](../part01-prerequisites/07-stability-generalization.md)).

These findings give the order of defences. **Deduplication** is the cheapest, because it lowers the copies of every repeated string. Lee et al. (2022) found that models trained on deduplicated data emit memorized text about ten times less often without losing quality, and released their exact-substring tool as `deduplicate-text-datasets`. For near duplicates, MinHash with locality-sensitive hashing is the usual method, for example `MinHash` and `MinHashLSH` in the `datasketch` package. Deduplication does nothing for a string seen once in a large model, though, because there one copy is enough.

The second defence is to remove personal data before training. A detector of personally identifiable information (PII), such as Microsoft's Presidio, finds names, phone numbers and similar fields so you can mask them. It misses whatever its patterns do not cover, so measure what remains with canaries rather than trusting the filter. Filtering the model's outputs is weaker still, because approximate copies pass a verbatim filter. The only defence with a guarantee for singletons is [training or fine-tuning with differential privacy](../part05-dp-ml/02-dp-sgd.md).

## Auditing with canaries and exposure {#sec-memo-exposure}

Discoverable memorization is all or nothing, while an auditor wants a graded answer. Carlini et al. (2019) proposed planting **canaries**, random secrets of a fixed format such as "my pin is" followed by five random digits. You know each canary's value and the number of values it could have taken, so after training you can see exactly where the model ranks it. Write $\mathcal R$ for the set of possible values, $|\mathcal R|$ for their number and $\hat r$ for the planted value. The **exposure** is the number of bits of search the model saves an attacker:
$$\mathrm{exposure}(\hat r) = \log_2|\mathcal R| - \log_2\operatorname{rank}(\hat r).$$ {#eq-memo-exposure}
Each bit halves the guesses the attacker needs, and an exposure near the maximum, $\log_2|\mathcal R|$, means the secret can actually be extracted.

What makes exposure an audit tool is its known behaviour when nothing was memorized. A canary that was never inserted lands at a uniformly random rank, so for any threshold of $t$ bits,
$$\Pr\big[\mathrm{exposure}(\hat r) \ge t\big] \le 2^{-t}.$$ {#eq-memo-null}
So $2^{-\text{exposure}}$ bounds a p-value, the chance that an uninserted canary would look at least this memorized. The average exposure of an uninserted canary is about 1.44 bits.

The recipe for a real model follows.

1. Draw a few dozen canaries in a format your data could contain, each with a fresh random value. Insert each one a chosen number of times, from once to many.
2. Train as usual, and keep some canaries drawn but never inserted as a control.
3. For each canary, score its value and many random values of the same format by the model's log-probability after the prefix. With Hugging Face `transformers`, pass the tokens as `input_ids` and as `labels`, with the prefix positions of `labels` set to -100. The returned `loss` is then the mean cross-entropy over the value's tokens.
4. Estimate the rank as the share of random values that score at least as high as the canary, and convert it to exposure.
5. Check that the controls average about 1.44 bits, then report the exposure against copies. TensorFlow Privacy ships secret-sharer utilities that compute exposure, but the snippet in @sec-memo-code does the same.

Three things go wrong. First, scoring $m$ random values cannot resolve exposures above $\log_2 m$ bits, because a canary that beats them all only tells you that its rank fraction is below about $1/m$. Carlini et al. (2019) extrapolate by fitting a skew-normal to the scores, because a plain normal fit overestimates exposure. Second, ties hide memorization, because when every candidate gets the same score, every exposure is 0. Third, the canary format must look like your data, or the model may treat it as unlike anything it learned. Comparing two training runs, such as with and without deduplication, then tests the defence.

## Extraction without the prefix: generate, rank, verify {#sec-memo-attack}

A real attacker has only the model. Carlini et al. (2021) extracted data from GPT-2 in three steps. The attacker generates many samples, ranks them by how likely each is to be memorized, and checks the top of the ranking against public sources. The ranking is the hard step, because a model finds a string likely either because it is ordinary text or because it was memorized. A **reference model**, trained on similar data but not on the target's training set, stands in for ordinary text. Write $p_\theta(x)$ for the target's probability of a string $x$, and $p_{\mathrm{ref}}(x)$ for the reference's. Rank by the log-likelihood ratio
$$\Lambda(x) = \ln p_\theta(x) - \ln p_{\mathrm{ref}}(x),$$ {#eq-memo-ratio}
which is the best possible ranking for a fixed rate of false alarms ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)).

Raw likelihood puts first the strings that any model finds likely, such as software licences and boilerplate, which reveal nothing, and the ratio cancels them. A crude reference works. Carlini et al. (2021) used a smaller model of the same family, and also the target's own likelihood on lowercased text and the compressed length of the text, `len(zlib.compress(text.encode()))` from Python's standard library. With Hugging Face `transformers`, `model.generate` with `do_sample=True` draws the samples, and passing `output_scores=True` and `return_dict_in_generate=True` lets `model.compute_transition_scores` return each token's log-probability.

An attacker without the prefix extracts less than the discoverable rate suggests, but production models are still not safe. Nasr et al. (2023) asked ChatGPT to repeat one word forever, and about 200 US dollars of queries extracted over 10,000 unique memorized examples, so alignment hides memorization from casual use without removing it.

## Code {#sec-memo-code}

These functions compute exposure, its p-value bound and the sample budget for an audit.

```python
from math import ceil, log, log2


def exposure(rank: int, space: int) -> float:
    """Bits of guessing the model saves: log2 |R| - log2 rank."""
    return log2(space) - log2(rank)


def null_p_value(exposure_bits: float) -> float:
    """Upper bound on the chance that a canary never inserted scores this high."""
    return 2.0**-exposure_bits


def samples_bound(p: float, beta: float) -> int:
    """Samples that extract a string of probability p, except with probability beta."""
    return ceil(log(1 / beta) / p)


print(f"{exposure(1, 10**5):.2f}")  # 16.61
print(f"{null_p_value(10):.5f}")  # 0.00098
print(samples_bound(1e-3, 0.01))  # 4606
```

The full edition's script, `code/part02-attacks/memorization.py`, runs every experiment in this chapter.

## Cheat sheet

| Question | Answer |
| --- | --- |
| Which rate to report | Discoverable for your own model, approximate as well as verbatim, with the decoding method stated |
| Samples to extract a string of probability $p$ | $\ln(1/\beta)/p$ |
| What drives memorization | Copies, model size and prompt length, each log-linearly; rare data first |
| Exposure of a canary | $\log_2\lvert\mathcal R\rvert - \log_2\operatorname{rank}$; at most 16.61 bits for five digits |
| Is the exposure significant | p-value at most $2^{-\text{exposure}}$; uninserted canaries average 1.44 bits |
| Ranking generations | $\ln p_\theta(x) - \ln p_{\mathrm{ref}}(x)$, never raw likelihood |
| Defences, in order | Deduplicate, scrub PII, then train with differential privacy for a guarantee on singletons |

: The results of this chapter. {#tbl-memo-cheat}

**Rules of thumb**

- A verbatim rate is a lower bound on the leak, not a measure of it.
- Deduplicate first, but know that neither deduplication nor a PII filter removes a secret seen once.
- Plant canaries at several copy counts, not only once, with uninserted controls, and score enough candidates to resolve the exposure you care about.
- Test with the decoding settings you deploy, and count approximate as well as verbatim matches.

## Further reading

- Carlini, Liu, Erlingsson, Kos & Song, "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks" (USENIX Security 2019): canaries, exposure and how to estimate it.
- Carlini, Ippolito, Jagielski, Lee, Tramèr & Zhang, "Quantifying Memorization Across Neural Language Models" (ICLR 2023): how memorization scales with duplication, model size and prompt length.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/05-memorization-extraction.md).
