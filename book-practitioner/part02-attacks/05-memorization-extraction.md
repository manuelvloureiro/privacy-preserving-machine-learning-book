# Memorization and training-data extraction

## Why this matters

A language model trained on scraped text sees phone numbers, addresses, keys and private messages. Ask it to continue "Anna Smith's phone number is" and it may finish with the real number. Nothing was breached: the number is stored in the weights and comes out through the normal interface. That is **training-data extraction**. [Membership inference](02-membership-inference.md) asks whether a record you already hold was in the training set; extraction hands the attacker the record itself, often one it did not know existed.

If you train or audit a generative model, you need to decide what to measure, which strings are at risk, how to test before release, and what an outside attacker could get. The one idea behind all four is the **rank** of a secret under the model. Fix the context in which a secret appears, say "my pin is", and list every possible filler in order of the model's probability. The secret's rank is its position in that list, and it is the number of guesses an attacker needs if it tries the most likely fillers first. Memorization is the model putting a training string far above the strings that could have stood in its place. @tbl-memo-overview maps the decisions to the sections.

| Decision | Rule | Section |
| --- | --- | --- |
| What "memorized" means for your report | Say which definition: discoverable, extractable, verbatim or approximate | @sec-memo-definitions |
| Which strings are at risk | Duplicated strings, and rare strings in large models; deduplicate | @sec-memo-copies |
| How to audit a model before release | Plant canaries and measure exposure, which has a known null | @sec-memo-exposure |
| What an attacker without the prefix gets | Generate, rank by a likelihood ratio to a reference model, verify | @sec-memo-attack |

: The decisions behind a memorization audit, and where this chapter makes each one. {#tbl-memo-overview}

The examples use one toy that runs in seconds: a contact list of 5,000 background lines like `toneka 14524` (a made-up name and a five-digit pin), plus 600 secret lines, 100 each repeated 1, 2, 4, 8, 16 and 32 times. The model is a character $n$-gram model, which predicts each character from the previous $n - 1$ by counting. Its order $n$ plays the role of model size.

## Verbatim, approximate, discoverable, extractable {#sec-memo-definitions}

**Idea.** "The model memorized its training data" can mean several things, which differ in what the attacker must already know and in how close the output must be. Pick one before measuring, because the numbers differ by orders of magnitude.

**Result.** Let greedy decoding append the most likely next symbol at each step.

- A string is **extractable** (Carlini et al. 2021) if *some* prompt makes greedy decoding emit it. This is the real attacker's notion, but measuring it means searching over all prompts.
- It is **$k$-eidetic** if it is extractable and appears in at most $k$ training documents. Small $k$ is the dangerous case: a secret seen once and given back.
- A suffix is **discoverable** (Carlini et al. 2023) if prompting with its own training prefix makes greedy decoding emit it. It assumes the attacker holds the prefix, but it is easy to measure: prompt with each training prefix and compare.
- It is **approximately** memorized (Ippolito et al. 2023) if the output is within some distance of it, such as one wrong digit. The other three are **verbatim**.

Discoverable implies extractable, and verbatim implies approximate. Deployed models sample rather than decode greedily. If one sample reproduces the string with probability $p$, then
$$m \ge \frac{\ln(1/\beta)}{p}$$ {#eq-memo-samples}
samples suffice to extract it with probability at least $1 - \beta$.

**Example.** A string with probability $10^{-3}$ after its prefix comes out with 99% confidence ($\beta = 0.01$) within about 4,600 samples. In the toy, an order-5 model reproduces a secret's pin in one sample with probability $1.4\times10^{-3}$ on average at 4 copies and 0.142 at 32 copies; the exact budgets for a 99% chance are 3,252 and 31 samples.

**Pitfalls.** A verbatim rate is a lower bound on the leak, not a measure of it: a phone number with one digit changed has still leaked. Ippolito et al. (2023) show that a filter blocking verbatim training text is easy to bypass with prompts that change the style, so don't treat such a filter as a defence. Don't compare a discoverable rate with an extractable one. A string greedy decoding never emits may still come out under sampling; @eq-memo-samples gives the attack budget.

## Duplication and model size {#sec-memo-copies}

**Idea.** Memorization is a vote. Each copy of a training line adds one vote for the line's own next symbol in every context the line passes through. Greedy decoding reproduces the line when, at every step, those votes beat the strongest rival from the rest of the corpus, so the number of copies a line needs is set by its most contested step. A larger model reads longer contexts, which fewer training positions share, so rivals have fewer votes. In the limit every line has contexts of its own and is memorized from one copy.

**Result.** In the $n$-gram toy this is exact: a line needs $k^*$ copies, its largest gap to a rival (in counts from the rest of the corpus) plus one, and the fraction of lines memorized at $k$ copies is the fraction with $k^* \le k$. It matched greedy decoding in all 36 cells of @tbl-memo-discoverable. For real models, Carlini et al. (2023) found that discoverable memorization grows log-linearly with duplication, model size and prompt length, and Kandpal, Wallace & Raffel (2022) found that a sequence repeated 10 times is generated about 1,000 times more often than one seen once.

**Example.** @tbl-memo-discoverable gives the toy's discoverable rates. Along a row, at order 5, the verbatim rate climbs from 0 at one copy to 0.70 at 32. Down a column, a line seen once is safe up to order 5, memorized 8% of the time at order 6, 70% at order 7 and 98% at order 8: those singletons are 1-eidetic. At scale, the largest model Carlini et al. (2023) measured, GPT-J, reproduces at least 1% of its training data from 50-token prefixes.

| Order $n$ | $k = 1$ | $k = 2$ | $k = 4$ | $k = 8$ | $k = 16$ | $k = 32$ |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.00 / 0.00 | 0.00 / 0.03 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.02 | 0.00 / 0.05 |
| 4 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.01 | 0.00 / 0.00 | 0.00 / 0.00 | 0.13 / 0.17 |
| 5 | 0.00 / 0.00 | 0.00 / 0.00 | 0.05 / 0.06 | 0.11 / 0.15 | 0.42 / 0.42 | 0.70 / 0.70 |
| 6 | 0.08 / 0.13 | 0.65 / 0.69 | 0.76 / 0.78 | 0.80 / 0.85 | 0.90 / 0.91 | 0.96 / 0.97 |
| 7 | 0.70 / 0.78 | 0.95 / 0.96 | 0.98 / 0.98 | 0.98 / 0.99 | 0.99 / 0.99 | 0.97 / 0.98 |
| 8 | 0.98 / 0.98 | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 |

: Fraction of 100 secret lines per cell whose pin greedy decoding reproduces from the true prefix, verbatim / with at most one digit wrong. Rows are model orders, columns copies in the training set. {#tbl-memo-discoverable}

**Pitfalls.** Deduplicate first: it is cheap, it lowers the vote count of every repeated string, and Lee et al. (2022) found that models trained on deduplicated data emit memorized text about ten times less often without losing quality. But it does nothing for singletons in a high-capacity model, which are memorized from one copy: in the table, deduplicating the order-8 model's data would barely change its 98%. Rare and atypical strings are exposed first, because they reach contexts of their own soonest; this is the high **self-influence** of atypical examples from [Algorithmic stability and generalization](../part01-prerequisites/07-stability-generalization.md). Larger models memorize more and from fewer copies, so re-audit when you scale up.

## Canaries and the exposure metric {#sec-memo-exposure}

**Idea.** An auditor wants a graded answer with a calibrated false alarm rate, not all or nothing. Carlini et al. (2019) plant **canaries**, random secrets of a fixed format such as "my pin is" followed by five random digits, in the training data, and after training measure where the model ranks the planted value among all values of that format. Because you chose the canary at random, you know the answer and the size of the space.

**Result.** With $|\mathcal R|$ possible values and the canary at rank $\operatorname{rank}(\hat r)$, the **exposure** is the number of halvings of the search that the model saves the attacker, in bits:
$$\mathrm{exposure}(\hat r) = \log_2|\mathcal R| - \log_2\operatorname{rank}(\hat r).$$ {#eq-memo-exposure}
It lies between 0 and $\log_2|\mathcal R|$, and each bit halves the attacker's guessing work. Its null is known. If the model does not depend on the canary, for instance because it was never inserted, then for every $e \ge 0$
$$\Pr[\mathrm{exposure} \ge e] \le 2^{-e},$$ {#eq-memo-null}
and the mean exposure tends to $\log_2 e = 1.4427$ bits. Exposure is therefore $-\log_2$ of a p-value: the test "inserted if exposure $\ge e$" has false positive rate (FPR) at most $2^{-e}$, with no shadow models needed, because the canary's randomness does the calibration.

**Example.** For "my pin is 58745", $|\mathcal R| = 10^5$ and the maximum exposure is 16.61 bits. @tbl-memo-exposure gives the exact exposure, computed by scoring all $10^5$ pins. The order-4 model reaches the maximum at 16 copies, while the order-2 model, whose one-character contexts are shared with every line, barely moves even at 64. Uninserted pins behave as @eq-memo-null says: mean 1.443 bits, and a fraction $1/32 = 2^{-5}$ with exposure at least 5.

| Order $n$ | $k = 0$ | $k = 1$ | $k = 2$ | $k = 4$ | $k = 8$ | $k = 16$ | $k = 32$ | $k = 64$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.40 | 0.41 | 0.43 | 0.47 | 0.55 | 0.73 | 1.21 | 2.54 |
| 3 | 0.89 | 4.05 | 4.44 | 4.71 | 5.24 | 6.59 | 9.69 | 15.61 |
| 4 | 3.76 | 9.98 | 12.15 | 15.02 | 15.61 | 16.61 | 16.61 | 16.61 |

: Exposure in bits of the canary "my pin is 58745" against the number of copies $k$ in the training set. The maximum is $\log_2 10^5 = 16.61$. {#tbl-memo-exposure}

![Exposure against copies of the canary for $n$-gram models of order 2, 3 and 4; thin lines are individual canaries and markers their mean, and the dotted line is the maximum. Larger models reach it with fewer copies.](figures/memo_exposure_vs_copies.svg){#fig-memo-exposure-vs-copies}

**Pitfalls.** For large spaces you must estimate the rank. Counting how many of $m$ random candidates score above the canary is reliable but cannot resolve exposures above $\log_2 m$. Fitting a distribution to the scores reaches further (Carlini et al. 2019 fit a skew-normal), but only as far as its tail is right. At order 3 in the toy, with the canary inserted 8 times, counting 10,000 random pins gives 5.35 bits against an exact 5.24, while a normal fit gives 8.28, because the scores have a heavier right tail than a normal. Ties hide memorization: a model that has never seen a context gives every candidate the same score and every exposure 0. Use canaries to compare training runs that differ in one thing: with and without deduplication, before and after an unlearning method, or with private training ([Part 3](../part03-dp-foundations/00-introduction.md)); Carlini et al. (2019) found that differentially private training kept their canaries at the exposure of canaries never inserted.

## Extraction without the prefix: generate, rank, verify {#sec-memo-attack}

**Idea.** A real attacker has only the model. Carlini et al. (2021) extracted training data from GPT-2 in three steps: generate many samples, rank them by how likely they are to be memorized, and verify the top of the ranking against a search or public sources. The ranking is the hard step. A model can find a string likely because text like it is common, or because it memorized it; only the second is a leak.

**Result.** Rank each generation $x$ by the log-likelihood ratio of the target model to a **reference model** trained on similar data that excludes the target's training set,
$$\Lambda(x) = \ln p_\theta(x) - \ln p_{\mathrm{ref}}(x),$$ {#eq-memo-ratio}
not by $\ln p_\theta(x)$ alone. By the Neyman–Pearson lemma of [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md), this is the most powerful ranking at every FPR. It is the same per-example calibration as in membership inference. The reference can be crude: a smaller model of the same family, the zlib-compressed length of the text, or the target's own likelihood on the lowercased text.

**Example.** The attacker draws 20,000 lines from the order-6 toy model, whose training set also contains 50 public lines repeated 100 times each. Among the 14,852 distinct samples are 197 private training lines. Of the top 100 by likelihood, 45 are private lines and most of the rest are public. Ranking by @eq-memo-ratio, with a reference trained on an independent contact list, 85 of the top 100 are private (@fig-memo-extraction). Without the prefix the attacker gets much less than is discoverable: none of the secrets inserted 1, 2 or 4 times appear in the samples, against 76% of the 4-copy secrets discoverable at order 6, because a sample must first produce the right name.

![Fraction of the top-ranked distinct generations that are private training lines (solid) or public lines (dashed), from 20,000 samples of an order-6 model. Ranking by likelihood puts the public lines first; ranking by the likelihood ratio to a reference model puts the private lines first.](figures/memo_extraction.svg){#fig-memo-extraction}

**Pitfalls.** "Less than discoverable" is still a lot. Nasr et al. (2023) made an aligned chat model, which rarely emits training data in normal use, repeat a single word until it diverged into text that looked like pretraining documents; about 200 US dollars of queries extracted over 10,000 unique memorized examples, including personal contact details. So alignment is not a defence, and a red team should test prompts that escape the chat behaviour. Verify outputs against the training set where you have it, and against a large public corpus, as a lower bound, where you don't. Report precision at the top of the ranking, which is the low-FPR operating point that matters, not recall over all memorized strings.

## Code {#sec-memo-code}

Exposure and sample budgets are what you compute in an audit, and they need only logarithms:

```python
from math import ceil, log, log2


def exposure(rank: int, space: int) -> float:
    """Bits of guessing the model saves: log2 |R| - log2 rank."""
    return log2(space) - log2(rank)


def null_p_value(exposure_bits: float) -> float:
    """Upper bound on the chance that a canary never inserted scores this high."""
    return 2.0**-exposure_bits


def samples_needed(p: float, beta: float) -> int:
    """Smallest m with (1 - p)^m <= beta: samples for a 1 - beta chance of drawing a string of probability p."""
    return ceil(log(beta) / log(1 - p))


def samples_bound(p: float, beta: float) -> int:
    """The simpler sufficient budget ln(1 / beta) / p."""
    return ceil(log(1 / beta) / p)


print(f"{exposure(1, 10**5):.2f}")  # 16.61
print(f"{exposure(6018, 10**5):.2f}")  # 4.05
print(f"{null_p_value(10):.5f}")  # 0.00098
print(samples_needed(0.142, 0.01))  # 31
print(samples_bound(1e-3, 0.01))  # 4606
```

Rank 6,018 of $10^5$ is the order-3 canary at one copy. An exposure of 10 bits has a p-value below 0.001, whatever the size of the space. The full edition's script runs every experiment in this chapter:

```bash
uv run python code/part02-attacks/memorization.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| Discoverable | Greedy decoding from the true prefix emits the suffix | Easy to measure; implies extractable |
| Extractable, $k$-eidetic | Some prompt emits it; in at most $k$ documents | The real attacker's notion |
| Approximate | Output within distance $\tau$ | Verbatim rates understate the leak |
| Samples needed | $m \ge \ln(1/\beta)/p$ | $p$ the probability of the string after the prompt |
| Exposure | $\log_2\lvert\mathcal R\rvert - \log_2\operatorname{rank}$ | Bits of guessing saved |
| Exposure null | $\Pr[\mathrm{exposure} \ge e] \le 2^{-e}$; mean $\to 1.443$ bits | $-\log_2$ of a p-value |
| Ranking generations | $\ln p_\theta(x) - \ln p_{\mathrm{ref}}(x)$ | Cancels text that is likely under any model |

: The results of this chapter. {#tbl-memo-cheat}

**Rules of thumb**

- Memorization is a rank: the secret above the strings it could have been.
- Duplication and model size drive memorization. Deduplicate, but expect singletons in large models to be memorized anyway.
- Audit with planted canaries and report exposure, which is graded and has a known null.
- Rank generations against a reference model, never by raw likelihood.
- An attacker without the prefix gets less than the discoverable rate suggests, but production attacks show that "less" is still thousands of records.

## Further reading

- Carlini, Liu, Erlingsson, Kos & Song, "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks" (USENIX Security 2019): canaries, exposure and how to estimate it.
- Carlini et al., "Extracting Training Data from Large Language Models" (USENIX Security 2021): the generate-rank-verify attack and its reference-based rankings.
- Carlini, Ippolito, Jagielski, Lee, Tramèr & Zhang, "Quantifying Memorization Across Neural Language Models" (ICLR 2023): how memorization scales with duplication, model size and prompt length.
- Nasr et al., "Scalable Extraction of Training Data from (Production) Language Models" (2023): discoverable against extractable memorization, and the divergence attack on an aligned chat model.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/05-memorization-extraction.md).
