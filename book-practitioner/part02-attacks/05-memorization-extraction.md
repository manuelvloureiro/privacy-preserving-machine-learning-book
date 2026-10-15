# Memorization and training-data extraction

## Why this matters

A language model trained on scraped text sees phone numbers, addresses, keys and private messages. Ask it to continue "Anna Smith's phone number is" and it may finish with the real number. Nothing was breached: the number is stored in the weights and comes out through the normal interface. That is **training-data extraction**. [Membership inference](02-membership-inference.md) asks whether a record you already hold was in the training set; extraction hands the attacker the record itself, often one it did not know existed.

The idea that organizes the chapter is the **rank** of a secret under the model. A language model gives every string a probability. Fix the context in which a secret appears, say "my pin is", and list every possible filler in order of the model's probability, most likely first. The secret's rank is its position in that list, and it is also the number of guesses an attacker needs if it tries the most likely fillers first. Memorization is the model putting a training string far above the strings that could have stood in its place. Every measure below reads that rank in its own way.

The examples use one toy: a contact list of 5,000 background lines like `toneka 14524` (a made-up name and a five-digit pin), plus 600 secret lines, 100 each repeated 1, 2, 4, 8, 16 and 32 times. The model is a character $n$-gram model, which predicts each character from the previous $n - 1$ by counting. Its order $n$ plays the role of model size.

## Verbatim, approximate, discoverable, extractable {#sec-memo-definitions}

"The model memorized its training data" can mean several things, and the numbers differ by orders of magnitude between them, so a report must say which it measures. The definitions differ in what the attacker is assumed to know. A string is **extractable** (Carlini et al. 2021) if *some* prompt makes greedy decoding, which appends the most likely next symbol at each step, emit it. That is the real attacker, who has only the model, but measuring it means searching over all prompts. A suffix is **discoverable** (Carlini et al. 2023) if prompting with its own training prefix makes greedy decoding emit it. That assumes an attacker who holds the prefix, which is strong, but it is easy to measure: prompt with each training prefix and compare. Discoverable implies extractable, since the prefix is one prompt among all. A string is **$k$-eidetic** if it is extractable and appears in at most $k$ training documents; small $k$ is the worrying case, a secret seen once and given back.

The definitions also differ in how close the output must be. The ones above are **verbatim**; a string is **approximately** memorized (Ippolito et al. 2023) if the output is within some distance of it. Verbatim matching misses near copies, and a phone number with one digit changed has still leaked to anyone who reads it. Ippolito et al. show that prompts which change the style slip approximate copies past a filter that blocks verbatim training text. A verbatim rate is therefore a lower bound on the leak, not a measure of it, and a verbatim filter is not a defence.

Greedy decoding is a convenience for measurement; deployed models sample. A string greedy decoding never emits can still come out. If one sample reproduces it with probability $p$, then $m$ independent samples all miss it with probability $(1 - p)^m$, which is at most $e^{-pm}$, so
$$m \ge \frac{\ln(1/\beta)}{p}$$ {#eq-memo-samples}
samples extract it with probability at least $1 - \beta$. The budget is inversely proportional to $p$: a string with probability $10^{-3}$ after its prefix comes out with 99% confidence ($\beta = 0.01$) within about 4,600 samples. In the toy, an order-5 model reproduces a secret's pin in one sample with probability $1.4\times10^{-3}$ on average at 4 copies and 0.142 at 32 copies, which means 3,252 and 31 samples for a 99% chance. Duplication made the search trivial.

## Duplication and model size {#sec-memo-copies}

Which strings get memorized? In the $n$-gram toy the answer can be read off the counts. Greedy decoding follows the largest count at each step, so memorization is a vote. Each copy of a training line adds one vote for the line's own next symbol in every context the line passes through. The line comes out when, at every step, its votes beat the strongest rival from the rest of the corpus, so the number of copies it needs, $k^*$, is its largest gap to a rival plus one, set by its most contested step. The fraction of lines memorized at $k$ copies is the fraction with $k^* \le k$; this matched greedy decoding exactly in all 36 cells of @tbl-memo-discoverable, because it is a restatement of what greedy decoding does with counts.

Model size works by splitting the vote. A longer context is a finer partition of the training positions: fewer positions share a given context of length 7 than of length 2. So rival counts shrink as the order grows, while the line's own $k$ votes stay with it in every context it passes through. In the limit every line has contexts of its own and is memorized from a single copy. That is the toy version of "larger models memorize more": a larger model can afford to give rare strings parameters of their own.

@tbl-memo-discoverable shows both effects. Along a row, at order 5, the verbatim rate climbs from 0 at one copy to 0.70 at 32. Down a column, a line seen once is safe up to order 5, memorized 8% of the time at order 6, 70% at order 7 and 98% at order 8; those singletons are 1-eidetic.

| Order $n$ | $k = 1$ | $k = 2$ | $k = 4$ | $k = 8$ | $k = 16$ | $k = 32$ |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.00 / 0.00 | 0.00 / 0.03 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.02 | 0.00 / 0.05 |
| 4 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.01 | 0.00 / 0.00 | 0.00 / 0.00 | 0.13 / 0.17 |
| 5 | 0.00 / 0.00 | 0.00 / 0.00 | 0.05 / 0.06 | 0.11 / 0.15 | 0.42 / 0.42 | 0.70 / 0.70 |
| 6 | 0.08 / 0.13 | 0.65 / 0.69 | 0.76 / 0.78 | 0.80 / 0.85 | 0.90 / 0.91 | 0.96 / 0.97 |
| 7 | 0.70 / 0.78 | 0.95 / 0.96 | 0.98 / 0.98 | 0.98 / 0.99 | 0.99 / 0.99 | 0.97 / 0.98 |
| 8 | 0.98 / 0.98 | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 |

: Fraction of 100 secret lines per cell whose pin greedy decoding reproduces from the true prefix, verbatim / with at most one digit wrong. Rows are model orders, columns copies in the training set. {#tbl-memo-discoverable}

For real models, Carlini et al. (2023) found that discoverable memorization grows log-linearly with duplication, model size and prompt length, and that the largest model they measured, GPT-J, reproduces at least 1% of its training data from 50-token prefixes.

The vote picture also says who is exposed first. A line whose name shares little with other names reaches a context of its own sooner, so rare and atypical strings are memorized first; this is the high **self-influence** of atypical examples from [Algorithmic stability and generalization](../part01-prerequisites/07-stability-generalization.md), which can only be fitted by memorizing them. Deduplication lowers $k$ for every repeated string, so it is the cheapest defence: Lee et al. (2022) found that models trained on deduplicated data emit memorized text about ten times less often without losing quality. But it cannot touch a singleton in a large model, where $k = 1$ already suffices; deduplicating the order-8 model's data would barely change its 98%. And because larger models memorize from fewer copies, an audit passed at one size must be redone after scaling up.

## Canaries and the exposure metric {#sec-memo-exposure}

Discoverable memorization is all or nothing; an auditor wants a graded answer with a known false alarm rate. Carlini et al. (2019) plant **canaries**, random secrets of a fixed format such as "my pin is" followed by five random digits, in the training data, then read off the trained model's rank of the canary among all values of that format. Because you chose the canary at random, you know the answer and the size of the space $\mathcal R$. The **exposure** reads the rank on a log scale,
$$\mathrm{exposure}(\hat r) = \log_2|\mathcal R| - \log_2\operatorname{rank}(\hat r).$$ {#eq-memo-exposure}
Before seeing the model, an attacker guessing the canary faces $|\mathcal R|$ equally likely values; guessing in the model's order, it needs $\operatorname{rank}$ tries. Exposure is the difference in bits, the number of halvings of the search the model has done for the attacker.

What makes exposure an audit tool is that its behaviour without memorization is known exactly. If the model does not depend on the canary, for instance because it was never inserted, the canary is just a uniformly random value, so its rank is uniform over the list. The chance of landing in the top fraction $2^{-e}$ of the list is then $2^{-e}$:
$$\Pr[\mathrm{exposure} \ge e] \le 2^{-e}.$$ {#eq-memo-null}
So exposure is $-\log_2$ of a p-value, and the test "inserted if exposure $\ge e$" has false positive rate (FPR) at most $2^{-e}$. "Was this canary inserted?" is a membership question, but where membership inference calibrates its test with shadow models, here the canary's own randomness does the calibration. In the toy, uninserted pins show mean exposure 1.443 bits, the $\log_2 e$ the uniform rank predicts, and a fraction $1/32 = 2^{-5}$ with exposure at least 5.

For "my pin is 58745", $|\mathcal R| = 10^5$ and the maximum exposure is 16.61 bits. @tbl-memo-exposure gives the exact exposure, from scoring all $10^5$ pins. The order-4 model reaches the maximum at 16 copies. The order-2 model barely moves even at 64, because its one-character contexts are shared with every line, the vote-splitting argument run backwards.

| Order $n$ | $k = 0$ | $k = 1$ | $k = 2$ | $k = 4$ | $k = 8$ | $k = 16$ | $k = 32$ | $k = 64$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.40 | 0.41 | 0.43 | 0.47 | 0.55 | 0.73 | 1.21 | 2.54 |
| 3 | 0.89 | 4.05 | 4.44 | 4.71 | 5.24 | 6.59 | 9.69 | 15.61 |
| 4 | 3.76 | 9.98 | 12.15 | 15.02 | 15.61 | 16.61 | 16.61 | 16.61 |

: Exposure in bits of the canary "my pin is 58745" against the number of copies $k$ in the training set. The maximum is $\log_2 10^5 = 16.61$. {#tbl-memo-exposure}

![Exposure against copies of the canary for $n$-gram models of order 2, 3 and 4; thin lines are individual canaries and markers their mean, and the dotted line is the maximum. Larger models reach it with fewer copies.](figures/memo_exposure_vs_copies.svg){#fig-memo-exposure-vs-copies}

For a large space the rank must be estimated. Counting how many of $m$ random candidates score above the canary is reliable, but a canary above all of them only says its rank fraction is below about $1/m$, so it cannot resolve exposures above $\log_2 m$. Fitting a distribution to the scores reaches further, but only as far as its tail is right, and the tail is the part you cannot sample. At order 3 with 8 copies, counting 10,000 random pins gives 5.35 bits against an exact 5.24, while a normal fit gives 8.28, because the scores have a heavier right tail than a normal. Ties are a further trap: a model that has never seen a context gives every candidate the same score, so every rank is the last and every exposure 0, which hides rather than measures.

Because you control what was planted, use canaries to compare training runs that differ in one thing: with and without deduplication, before and after an unlearning method, or with private training ([Part 3](../part03-dp-foundations/00-introduction.md)). Carlini et al. (2019) found that differentially private training kept their canaries at the exposure of canaries never inserted.

## Extraction without the prefix: generate, rank, verify {#sec-memo-attack}

A real attacker has only the model. Carlini et al. (2021) extracted training data from GPT-2 in three steps: generate many samples, rank them by how likely they are to be memorized, and verify the top of the ranking against a search or public sources. The ranking is the hard step, and it is a hypothesis test. A model can find a string likely because text like it is common, or because it memorized it, and only the second is a leak. Ranking by the target's likelihood alone surfaces the common, such as licences and boilerplate, which reveal nothing. A **reference model**, trained on similar data that excludes the target's training set, stands in for "common text", and the right score is how much more likely the target finds $x$ than the reference does:
$$\Lambda(x) = \ln p_\theta(x) - \ln p_{\mathrm{ref}}(x).$$ {#eq-memo-ratio}
Text both models know cancels. By the Neyman–Pearson lemma of [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md), this likelihood ratio is the most powerful ranking at every FPR; it is the per-example calibration of membership inference again. The reference can be crude: a smaller model of the same family, the zlib-compressed length of the text, or the target's own likelihood on the lowercased text.

In the toy, the attacker draws 20,000 lines from the order-6 model, whose training set also contains 50 public lines repeated 100 times each. Among the 14,852 distinct samples are 197 private training lines. Of the top 100 by likelihood only 45 are private, since the public lines are the most likely strings the model knows. Ranked by @eq-memo-ratio, with a reference trained on an independent contact list that also contains the public lines, 85 of the top 100 are private (@fig-memo-extraction).

![Fraction of the top-ranked distinct generations that are private training lines (solid) or public lines (dashed), from 20,000 samples of an order-6 model. Ranking by likelihood puts the public lines first; ranking by the likelihood ratio to a reference model puts the private lines first.](figures/memo_extraction.svg){#fig-memo-extraction}

Without the prefix the attacker gets much less than is discoverable: none of the secrets inserted 1, 2 or 4 times appear in the samples, though 76% of the 4-copy secrets are discoverable at order 6. A sample must first produce the right name before it can produce the pin, and only heavily duplicated lines make their names likely enough.

"Less than discoverable" can still be a great deal. Nasr et al. (2023) attacked an aligned chat model, which rarely emits training data in normal use because alignment taught it to answer as an assistant, not to continue documents. Asked to repeat a single word forever, it diverged into text that looked like pretraining documents; about 200 US dollars of queries extracted over 10,000 unique memorized examples, including personal contact details. Alignment changed how the model usually responds, not what it had memorized, so it is not a defence, and a red team should test prompts that escape the chat behaviour. Verify outputs against the training set, or against a large public corpus as a lower bound. Report precision at the top of the ranking, which is the low-FPR operating point an attacker checking candidates by hand cares about, not recall over all memorized strings.

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

## Further reading

- Carlini, Liu, Erlingsson, Kos & Song, "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks" (USENIX Security 2019): canaries, exposure and how to estimate it.
- Carlini et al., "Extracting Training Data from Large Language Models" (USENIX Security 2021): the generate-rank-verify attack and its reference-based rankings.
- Carlini, Ippolito, Jagielski, Lee, Tramèr & Zhang, "Quantifying Memorization Across Neural Language Models" (ICLR 2023): how memorization scales with duplication, model size and prompt length.
- Nasr et al., "Scalable Extraction of Training Data from (Production) Language Models" (2023): discoverable against extractable memorization, and the divergence attack on an aligned chat model.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/05-memorization-extraction.md).
