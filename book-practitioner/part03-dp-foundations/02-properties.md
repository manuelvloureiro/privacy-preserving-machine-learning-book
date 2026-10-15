# Post-processing, group privacy and composition

## Why this matters

One mechanism is differentially private. Now you use it in a system. You round its output, plot it, train a model on it. Your users each contributed fifty records, and some of them share a household. And you never run just one mechanism: an analyst asks a hundred queries, a training run takes thousands of noisy gradient steps. Which of these steps cost privacy, and how do the costs add up? Three properties of differential privacy (DP) answer that, and they are what make it an engineering tool: you prove each building block once and add up the costs, instead of proving every pipeline from scratch.

The one idea is about the **privacy loss**, from [The definition of differential privacy](01-definition.md): the log of how much more likely an output is when the data are $D$ than when they are a neighbouring dataset $D'$, which differs in one record. Pure $\varepsilon$-DP keeps it between $-\varepsilon$ and $\varepsilon$ for every output. **Along a path of datasets or across a sequence of mechanisms, privacy losses add; under processing, they only average.** Processing pools outputs, and an average of losses in $[-\varepsilon, \varepsilon]$ stays there. Walking from $D$ to a dataset $k$ records away crosses $k$ neighbouring pairs, and the losses of the steps add. Releasing several independent outputs multiplies their probabilities, so their losses add too. @tbl-dpprop-overview maps the decisions to the sections.

| Decision or question | Rule | Section |
| --- | --- | --- |
| Is anything computed from a private output still private? | Yes, for free, if it never reads the data again | @sec-dpprop-post |
| What does a per-record guarantee give a user with $k$ records? | $k\varepsilon$, and a $\delta$ that grows exponentially in $k$ | @sec-dpprop-group |
| What do many releases from the same data cost? | Budgets add, also when each query depends on earlier answers | @sec-dpprop-comp |
| What do releases from disjoint parts of the data cost? | The largest budget, not the sum | @sec-dpprop-parallel |
| Which shortcuts break the guarantee? | Shared noise, data-reading "post-processing", exact side releases | @sec-dpprop-limits |

: The decisions behind combining private releases, and where this chapter covers each. {#tbl-dpprop-overview}

## Post-processing {#sec-dpprop-post}

**Idea.** Any processing of an output pools outputs: several raw outputs lead to the same processed one, whose probability is the sum of theirs. Its likelihood ratio is therefore a weighted average of their ratios, and an average cannot leave the band $[e^{-\varepsilon}, e^\varepsilon]$. This is the **data processing inequality**: processing can only dilute evidence.

**Result.** If $M$ is $(\varepsilon, \delta)$-DP and $f$ is any function, deterministic or with its own coins, that does not read the data, then $f \circ M$, the mechanism that outputs $f(M(D))$, is $(\varepsilon, \delta)$-DP. Its privacy curve $\delta(\varepsilon)$ lies on or below that of $M$ at every $\varepsilon$.

**Example.** Collect a randomized-response answer at $\varepsilon_0 = \ln 3$ (the truth with probability $\frac34$), then flip it with probability $\frac14$ using a fresh coin before publishing. The published 1 has probability 0.625 when the true bit is 1 and 0.375 when it is 0, so the release is $\ln\frac53 = 0.5108$-DP, below $\ln 3 = 1.0986$. For a count with Laplace noise of scale 1, @tbl-dpprop-laplace-post shows what rounding and thresholding do.

| Published | Smallest pure $\varepsilon$ | $\delta(0.25)$ | $\delta(0.5)$ |
| --- | --- | --- | --- |
| Noisy count $c + Z$ | 1.0000 | 0.3127 | 0.2212 |
| Rounded to an integer | 1.0000 | 0.3073 | 0.1967 |
| Only "is $c + Z > \frac12$?" | 0.8318 | 0.3073 | 0.1967 |

: Exact privacy curves of a count with Laplace noise of scale 1, before and after two post-processings. Each row is on or below the one above it. {#tbl-dpprop-laplace-post}

![Left: the privacy curve $\delta(\varepsilon)$ of a count with Laplace noise of scale 1 (black), after rounding (blue), and after keeping only whether it exceeds $\frac12$ (orange); processing only lowers the curve. Right: randomized response at $\varepsilon_0 = \ln 3$ followed by a fresh coin that flips the answer with probability $f$; the dot is $f = \frac14$, where $\varepsilon = \ln\frac53$.](figures/dpprop_postprocessing.svg){#fig-dpprop-post}

**Pitfalls.** The one condition is that $f$ must not touch the data. "Take the noisy top category, then report its exact count" is not post-processing; the second step is a new, non-private mechanism. Rounding and clipping merge few outputs and rarely lower $\varepsilon$ (rounding keeps it at 1.0000 above), so report the guarantee of $M$ and don't count on processing gains. The upside is large: a model trained on DP statistics, synthetic data from a DP model, a dashboard and every attack of [Threat models and attacks](../part02-attacks/00-introduction.md) are post-processing, which is why a DP system privatizes at one narrow point and treats everything after it as public.

## Group privacy {#sec-dpprop-group}

**Idea.** The definition protects one record, but a user may own fifty, and a household moves together. Walk from $D$ to a dataset $k$ records away one record at a time. Each step can multiply an event's probability by $e^\varepsilon$, so $k$ steps multiply it by $e^{k\varepsilon}$. Each step also adds a slack $\delta$, and every later step multiplies the slack already accumulated by $e^\varepsilon$ again.

**Result.** An $(\varepsilon, \delta)$-DP mechanism is, for datasets $k$ records apart,
$$\Big(k\varepsilon,\ \delta_k\Big)\text{-DP}, \qquad \delta_k = \delta\sum_{j=0}^{k-1}e^{j\varepsilon} = \delta\,\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1}.$$ {#eq-dpprop-group}
Pure DP degrades linearly: $k\varepsilon$, tight for a noisy count. The $\delta$ grows exponentially in $k$, and some mechanisms attain the bound, so none better holds in general. The simpler form $k\,e^{(k-1)\varepsilon}\delta$ is looser; use the sum.

**Example.** A per-record guarantee of $(0.5, 10^{-5})$ gives a user with 10 records $(5, 2.272 \times 10^{-3})$ by @eq-dpprop-group (@tbl-dpprop-group). A count with Gaussian noise does better than the bound, but its exact guarantee for that user is still $(5, 7.6 \times 10^{-4})$, which is weak. At $\varepsilon = 1$ and $k = 10$, $\delta$ is multiplied by 12818.3.

| Group size $k$ | $\varepsilon_k = k\varepsilon$ | $\delta_k$, @eq-dpprop-group | Simple form $k e^{(k-1)\varepsilon}\delta$ | Gaussian count, exact |
| --- | --- | --- | --- | --- |
| 1 | 0.5 | $1.000 \times 10^{-5}$ | $1.000 \times 10^{-5}$ | $1.000 \times 10^{-5}$ |
| 2 | 1 | $2.649 \times 10^{-5}$ | $3.297 \times 10^{-5}$ | $2.551 \times 10^{-5}$ |
| 3 | 1.5 | $5.367 \times 10^{-5}$ | $8.155 \times 10^{-5}$ | $4.858 \times 10^{-5}$ |
| 5 | 2.5 | $1.724 \times 10^{-4}$ | $3.695 \times 10^{-4}$ | $1.288 \times 10^{-4}$ |
| 10 | 5 | $2.272 \times 10^{-3}$ | $9.002 \times 10^{-3}$ | $7.592 \times 10^{-4}$ |

: Group privacy from a $(0.5, 10^{-5})$ guarantee per record: the bound, its simple form, and the exact $\delta_k$ of a count with Gaussian noise at $\varepsilon_k = k\varepsilon$. {#tbl-dpprop-group}

**Pitfalls.** A $0.1$-DP guarantee per record is $10$-DP for a user with 100 records, which promises almost nothing. Don't patch this with group privacy after the fact: define neighbours at the user level, bound each user's contribution, and prove DP at that level directly. Group privacy is the fallback when you cannot. And remember that $\delta$ must already be far below one over the number of people, so multiplying it by thousands often makes the group guarantee vacuous.

## Sequential composition {#sec-dpprop-comp}

**Idea.** Run several mechanisms on the same data with independent noise. The probability of the whole transcript of outputs is the product of the steps' probabilities, so the privacy loss of the transcript is the sum of the steps' losses. That stays true when each mechanism is chosen after seeing the earlier outputs, as an analyst does.

**Result.** If $M_1, \dots, M_k$ use independent coins, and each $M_i$ is $(\varepsilon_i, \delta_i)$-DP for every value of the earlier outputs, then releasing all their outputs is
$$\Big(\sum_{i=1}^k\varepsilon_i,\ \sum_{i=1}^k\delta_i\Big)\text{-DP}.$$ {#eq-dpprop-basic}
This is **basic composition**. It holds for **adaptive** composition, where each query depends on earlier answers, as long as the budgets $(\varepsilon_i, \delta_i)$ are fixed in advance. Unlike group privacy, the $\delta$s add plainly, because every mechanism compares the same pair of datasets.

**Example.** Asking one person the same question $k$ times with fresh randomized response at $\varepsilon_0 = \ln 3$ costs $k\ln 3$, and that worst case is real: the transcript where every answer is truthful has exactly that loss. But it is rare, with probability $(\frac34)^k$, and the mean loss is half the bound (@fig-dpprop-rr-loss). Allowing a small $\delta$ writes off such rare transcripts: at $\varepsilon_0 = 0.1$, the basic bound for $k = 1000$ answers is 100, while the exact $\varepsilon$ at $\delta = 10^{-6}$ is 19.345. For counts with Gaussian noise, each $(0.5, 10^{-5})$-DP, @tbl-dpprop-gauss-comp shows the same gap.

| $k$ | Basic: $(k\varepsilon, k\delta)$ | Exact $\varepsilon$ at the same $\delta = k \times 10^{-5}$ |
| --- | --- | --- |
| 1 | $(0.5, 10^{-5})$ | 0.500 |
| 2 | $(1, 2 \times 10^{-5})$ | 0.694 |
| 4 | $(2, 4 \times 10^{-5})$ | 0.966 |
| 10 | $(5, 10^{-4})$ | 1.504 |

: $k$ counts with Gaussian noise, each $(0.5, 10^{-5})$-DP: basic composition against the exact $\varepsilon$ at the same total $\delta$. {#tbl-dpprop-gauss-comp}

![Distribution of the privacy loss of 10 randomized-response answers at $\varepsilon_0 = \ln 3$ each, when the true bit is 1 (blue) or 0 (orange). The basic bound $10\ln 3$ is reached, but only with probability $(3/4)^{10} = 0.056$; the mean under the first dataset is half of it.](figures/dpprop_rr_loss.svg){#fig-dpprop-rr-loss}

**Pitfalls.** Basic composition is valid and simple, and it is the right tool for a handful of releases. For ten Gaussian queries it already overstates $\varepsilon$ by a factor of 3.3, and for thousands of training steps it is useless: use the accountants of [Advanced and optimal composition](04-composition.md) and [Part 4](../part04-privacy-accounting/00-introduction.md), where $\varepsilon$ grows like $\sqrt k$ rather than $k$. If the budgets themselves depend on earlier answers ("stop when the answer is good enough"), @eq-dpprop-basic does not apply; that needs privacy filters, also in [Advanced and optimal composition](04-composition.md).

## Parallel composition {#sec-dpprop-parallel}

**Idea.** If the data are split into disjoint parts and each mechanism sees only its part, one person's record lands in one part, so only one mechanism sees a difference. You pay that part's budget, not the sum.

**Result.** Split the universe of possible records into disjoint parts by a rule fixed in advance, run $M_i$ on the records in part $i$ with independent coins, and suppose each $M_i$ is $(\varepsilon_i, \delta_i)$-DP under add/remove (neighbours differ by one record added or removed). The whole release is
$$\big(\max_i\varepsilon_i,\ \max_i\delta_i\big)\text{-DP under add/remove}.$$ {#eq-dpprop-parallel}
Under replace (one record's value changed), a record that moves between parts changes two of them, which costs up to twice as much: $2\varepsilon$ for equal pure budgets. When each part is a fixed slot, such as one person in local randomized response, a replacement changes one slot and the maximum applies again.

**Example.** A histogram with Laplace noise of scale $1/\varepsilon$ in each bin is $\varepsilon$-DP under add/remove however many bins it has; for two bins the script finds exactly 1.0000 at $\varepsilon = 1$. Under replace it is $2\varepsilon$-DP (2.0000), which is why a histogram's sensitivity is quoted as 2 under replace ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). Mean incomes for 50 states, each $0.5$-DP, cost $0.5$ under add/remove and $1$ under replace, since a person who moves changes two states. Local randomized response at $\ln 3$ is $\ln 3$-DP for any number of people.

**Pitfalls.** The partition must not depend on the data. "The top 10% of earners and the rest" changes when one high earner is added, which can move records across the boundary and change both parts. Use fixed bins, or treat a data-dependent split as a mechanism of its own.

## What composes and what doesn't {#sec-dpprop-limits}

**Idea.** Each property above rests on an assumption: post-processing must not read the data, composition needs fresh noise and the same neighbouring relation for every step, and the budgets must be fixed in advance. Most real failures break one of them.

**Result.** @tbl-dpprop-rules is the checklist.

| Situation | Composes? | Why |
| --- | --- | --- |
| Function of the output, own coins | Yes, at no cost | Post-processing |
| Second query on the same data, fresh noise | Yes, budgets add | Basic composition |
| Second query chosen from the first answer | Yes, budgets add | Adaptive composition, budgets fixed in advance |
| Two queries sharing one noise draw | **No** | The noise cancels in a difference |
| "Post-processing" that reads the data again | **No** | It is a new mechanism, often a non-private one |
| Budgets chosen from earlier outputs | Not by basic composition | Needs privacy filters |
| Mechanisms proved under add/remove, released under replace | Yes, at up to twice the cost | A replacement is a removal plus an addition |
| A DP release plus an exact statistic of the same data | **No** | The exact statistic is not DP; the pair has no guarantee |
| Many records of one person | Yes, at $k\varepsilon$ and $\delta_k$ | Group privacy |

: What composes and what doesn't. {#tbl-dpprop-rules}

**Example.** Release the number of smokers and the number of smokers aged 40 or over, adding the *same* Laplace draw $Z$ to both. Each release alone is 1-DP. But their difference is the exact number of smokers under 40, and it changes when Alice, a smoker aged 30, is removed. An attacker who subtracts the two releases is always right about Alice: in 200,000 simulated runs its accuracy is 1.0000. With fresh noise for each count the same attack reaches 0.6206, and no attack can beat 0.6967.

**Pitfalls.** Shared noise is a total failure, not a small loss, and it hides in ordinary engineering: two mechanisms seeded with the same random seed, a cached noise value re-added to a related query, noise reused across the shards of a distributed system. Returning the same noisy answer to a repeated identical query is fine, because that is post-processing of one release. An exact total published elsewhere can be differenced against your DP release just as the two counts were. And mixing add/remove and replace guarantees inside one accountant is a factor-2 bug.

## Code {#sec-dpprop-code}

Group privacy is where hand calculations go wrong most often, because $\delta$ does not simply scale by $k$. This snippet computes the three budget rules of this chapter:

```python
from math import exp


def group_privacy(eps: float, delta: float, k: int) -> tuple[float, float]:
    """The guarantee for a group of k records, from an (eps, delta) guarantee per record."""
    delta_k = delta * sum(exp(j * eps) for j in range(k))
    return k * eps, delta_k


def basic_composition(budgets: list[tuple[float, float]]) -> tuple[float, float]:
    """The guarantee of several mechanisms run on the same data with fresh noise."""
    return sum(e for e, _ in budgets), sum(d for _, d in budgets)


def parallel_composition(budgets: list[tuple[float, float]]) -> tuple[float, float]:
    """The guarantee of mechanisms run on disjoint parts of the data, under add/remove."""
    return max(e for e, _ in budgets), max(d for _, d in budgets)


eps_k, delta_k = group_privacy(0.5, 1e-5, 10)
print(f"{eps_k:.1f} {delta_k:.3e}")  # 5.0 2.272e-03
print(f"{group_privacy(0.5, 1e-5, 2)[1]:.3e}")  # 2.649e-05
print(f"{group_privacy(1.0, 1.0, 10)[1]:.1f}")  # 12818.3
print(basic_composition([(0.5, 1e-5)] * 10))  # (5.0, 0.0001)
print(parallel_composition([(0.5, 0.0)] * 50))  # (0.5, 0.0)
```

The third line is the factor by which $\delta$ grows for a group of 10 at $\varepsilon = 1$. Basic composition here is a valid upper bound only; for the exact cost of many Gaussian releases, use an accountant. The full edition's script checks every number in this chapter by exact computation and simulation:

```bash
uv run python code/part03-dp-foundations/properties.py
```

## Cheat sheet

| Property | Rule | Notes |
| --- | --- | --- |
| Post-processing | $f \circ M$ is $(\varepsilon, \delta)$-DP | Only if $f$ never reads the data; can lower $\varepsilon$, never raise it |
| Group privacy | $\big(k\varepsilon,\ \delta\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1}\big)$ for $k$ records | Tight in general; prefer user-level neighbours |
| Basic composition | $(\sum\varepsilon_i, \sum\delta_i)$ | Also adaptive, with budgets fixed in advance; loose for many releases |
| Parallel composition | $(\max\varepsilon_i, \max\delta_i)$ under add/remove | Up to twice that under replace; partition fixed in advance |
| Shared noise | No guarantee at all | Every mechanism needs fresh noise |

: The results of this chapter. {#tbl-dpprop-cheat}

**Rules of thumb**

- Privatize once, as early as possible; everything after it is free.
- Budgets add across queries on the same data and take the maximum across disjoint data.
- For users with many records, define the neighbouring relation at the user level instead of relying on group privacy.
- Every mechanism needs its own fresh noise.
- Use basic composition for a handful of releases; for many, use an accountant.

## Further reading

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014): post-processing, group privacy and basic composition (Proposition 2.1, Theorems 2.2 and 3.16), the standard reference.
- McSherry, "Privacy Integrated Queries" (SIGMOD 2009): parallel composition, and a query system built on post-processing and composition as its only rules.
- Near & Abuah, *Programming Differential Privacy*, the chapters on the properties of DP: sequential and parallel composition and post-processing, with code.
- Rogers, Roth, Ullman & Vadhan, "Privacy Odometers and Filters: Pay-as-you-Go Composition" (NeurIPS 2016): what changes when the budgets themselves are chosen adaptively.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/02-properties.md).
