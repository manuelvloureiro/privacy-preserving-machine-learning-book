# The definition of differential privacy

## Why this matters

A model card says "trained with $\varepsilon = 8$, $\delta = 10^{-5}$". What does that promise, to whom, and against what? The attacks of [Threat models and attacks](../part02-attacks/00-introduction.md) all tried to tell, from an output, whether it came from a dataset with a given person in it or without them. **Differential privacy (DP)** asks that no such test work well, for any person and any adversary. It is a property of the algorithm, checked before any data are touched.

The one idea is that DP bounds a likelihood ratio. Run a randomized algorithm $M$, a **mechanism**, on two datasets $D$ and $D'$ that differ in one person, and look at an output $y$. How much more likely $y$ is under $D$ than under $D'$ is the evidence $y$ carries about that person. Its logarithm is the **privacy loss**,
$$L_{D,D'}(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)},$$
where $p_{M(D)}(y)$ is the probability (or density) that $M$ outputs $y$ on $D$. A loss of 0 means $y$ says nothing; a large loss gives the person away. DP keeps this loss small, and every practical reading of $\varepsilon$ and $\delta$ follows. @tbl-dpdef-overview maps the decisions to the sections.

| Decision or question | What to know | Section |
| --- | --- | --- |
| Whom does the guarantee protect? | The neighbouring relation and the unit of privacy | @sec-dpdef-neighbours |
| What do $\varepsilon$ and $\delta$ mean? | A ratio bound plus a slack; a mechanism has a whole curve $\delta(\varepsilon)$ | @sec-dpdef-definition |
| The simplest mechanism, and what privacy costs | Randomized response and its error | @sec-dpdef-rr |
| What can an attacker achieve? | Caps on true positive rate, advantage and posterior belief | @sec-dpdef-testing |
| How small must $\delta$ be? | Much smaller than one over the number of people | @sec-dpdef-small-delta |
| What is outside the guarantee? | Population inferences, correlated records, the implementation | @sec-dpdef-limits |

: The decisions behind reading or making a DP claim, and where this chapter covers each. {#tbl-dpdef-overview}

## Neighbouring datasets and the unit of privacy {#sec-dpdef-neighbours}

**Idea.** The **neighbouring relation** says which pairs of datasets, written $D \simeq D'$, must look alike. It fixes the **unit of privacy**: the thing whose presence or value the output must not reveal.

**Result.** Under **add/remove**, $D'$ is $D$ with one record added or removed; the size is private, and the guarantee hides whether you took part, the question membership inference asks. Under **replace**, one record's value changes; the size is public, and only your record's value is hidden. A replacement is a removal plus an addition, so $\varepsilon$-DP under add/remove gives $2\varepsilon$-DP under replace; the converse fails. Separately, a record can be one example, one user (all of a person's data) or a group of $k$ records. @tbl-dpdef-neighbours checks five candidates against $D = (\text{Alice: smoker}, \text{Bob: non-smoker}, \text{Carol: smoker})$.

| Candidate $D'$ | Add/remove neighbour? | Replace neighbour? |
| --- | --- | --- |
| (Alice: smoker, Bob: non-smoker) | yes: Carol removed | no: size differs |
| (Alice: smoker, Bob: non-smoker, Carol: smoker, Dan: non-smoker) | yes: Dan added | no: size differs |
| (Alice: non-smoker, Bob: non-smoker, Carol: smoker) | no: two steps | yes: Alice's answer changed |
| (Alice: smoker, Bob: non-smoker, Dan: smoker) | no: two steps | yes: Carol's record replaced by Dan's |
| (Alice: non-smoker, Bob: smoker, Carol: smoker) | no | no: two records changed |

: Five candidate datasets, checked against the add/remove and the replace relation. {#tbl-dpdef-neighbours}

**Example.** Release the histogram (smokers, non-smokers) with Laplace noise of scale $1/\varepsilon$ per bin, at $\varepsilon = 1$. Removing Carol changes one bin, and the largest privacy loss is 1.0000. Changing Alice's answer changes both bins, and it is 2.0000.

**Pitfalls.** An $\varepsilon$ without its neighbouring relation and unit of privacy cannot be evaluated, so always ask for both. Example-level DP on data where one person wrote 200 messages protects each message, not the person ([Post-processing, group privacy and composition](02-properties.md) shows how weak the person's guarantee becomes). Neighbours range over every possible record, including a person aged 900, which is why mechanisms clip their inputs.

## Pure and approximate differential privacy {#sec-dpdef-definition}

**Idea.** No event should become much more likely when one person's record changes: at most a factor $e^\varepsilon$, with a small additive slack $\delta$ for events of tiny probability.

**Result.** $M$ is **$(\varepsilon, \delta)$-differentially private** if for every pair of neighbours $D \simeq D'$ and every set $S$ of outputs,
$$\Pr[M(D) \in S] \le e^\varepsilon\,\Pr[M(D') \in S] + \delta.$$ {#eq-dpdef-def}
The probability is over the mechanism's coins, with the data fixed. With $\delta = 0$ this is **pure DP**, and it is the same as bounding the privacy loss of every single output: $|L_{D,D'}(y)| \le \varepsilon$. With $\delta > 0$ it is **approximate DP**: the loss may exceed $\varepsilon$ on rare outputs, and the smallest valid $\delta$ is the **hockey-stick divergence**, the probability mass of $M(D)$ sticking out above $e^\varepsilon$ times that of $M(D')$. An output with loss $L$ counts toward it with weight $(1 - e^{\varepsilon - L})_+$, where $(x)_+ = \max(x, 0)$: nothing if $L \le \varepsilon$, almost nothing just above, in full if $L$ is huge. So a mechanism has a **privacy curve** $\delta(\varepsilon)$, not a single pair. A bounded loss (Laplace noise) lets the curve reach $\delta = 0$; an unbounded one (Gaussian noise) never does (@fig-dpdef-ratio-band).

![The definition as a picture, for a count released with noise at $\varepsilon = 1$. The shaded band runs from $e^{-\varepsilon}$ to $e^\varepsilon$ times the output density on $D'$ (orange). Left: with Laplace noise of scale $1/\varepsilon$, the density on $D$ (blue) stays inside the band, so the release is pure DP. Right: with Gaussian noise of standard deviation 2, it leaves the band in the tail, and the red mass is what $\delta$ must cover.](figures/dpdef_ratio_band.svg){#fig-dpdef-ratio-band}

**Example.** At $\varepsilon = 0.1$, an event of probability 0.5 on $D'$ has probability at most $0.5e^{0.1} = 0.553$ on $D$; at $\varepsilon = 1$ the bound, 1.36, says nothing. For a count with Gaussian noise of standard deviation 2, @tbl-dpdef-gauss gives the exact curve and the simpler but looser tail probability $\Pr[L > \varepsilon]$.

| $\varepsilon$ | 0.5 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| Exact $\delta(\varepsilon)$ | $5.244\times10^{-2}$ | $6.830\times10^{-3}$ | $9.439\times10^{-6}$ | $3.401\times10^{-10}$ |
| Tail $\Pr[L > \varepsilon]$ | $2.266\times10^{-1}$ | $4.006\times10^{-2}$ | $8.842\times10^{-5}$ | $4.462\times10^{-9}$ |
| Tail / exact | 4.32 | 5.87 | 9.37 | 13.12 |

: The exact privacy curve of a count with Gaussian noise of standard deviation 2, against the tail probability of its privacy loss. {#tbl-dpdef-gauss}

**Pitfalls.** $\delta$ is not "the probability that privacy fails": the tail probability overstates it by a factor of 4 to 13, because outputs just past $\varepsilon$ count for almost nothing, so calibrating with the tail wastes budget. Never check approximate DP output by output, which pays $\delta$ once per output instead of once per event: two uniform distributions on 1,000 disjoint outputs pass that check with $\delta = 0.0010$, yet need $\delta = 1.0000$, since one look at the output reveals the dataset. Report the curve or the mechanism's parameters, not just one pair: accountants compose curves.

## Randomized response {#sec-dpdef-rr}

**Idea.** To learn how many people did something embarrassing, let each person flip a coin in private and sometimes lie. Every answer is deniable, yet the population fraction can be recovered, because the lying rate is known.

**Result.** **Randomized response** at level $\varepsilon$ reports a person's bit truthfully with probability $t = e^\varepsilon/(1 + e^\varepsilon)$ and flipped otherwise, so a report is at most $e^\varepsilon$ times more likely under one true bit than the other. Applied to each of $n$ people it is $\varepsilon$-DP under replace. The fraction $\bar y$ of reported ones is pulled toward $\frac12$ in a known way, and undoing that gives an unbiased estimate of the true fraction $\pi$,
$$\hat\pi = \frac{\bar y - (1 - t)}{2t - 1}, \qquad \operatorname{sd}(\hat\pi) = \frac{e^{\varepsilon/2}}{(e^\varepsilon - 1)\sqrt n} \approx \frac{1}{\varepsilon\sqrt n}.$$
Halving $\varepsilon$ doubles the error, and recovering the accuracy takes four times as many people.

**Example.** Warner's two coins (truth on heads, otherwise a second coin decides) give $t = 0.75$, which is randomized response at $\varepsilon = \ln 3 = 1.0986$. In a survey of 1,000 people with true fraction 0.3, the estimate's standard deviation is 0.0274, about twice the sampling error 0.0145 of an ordinary survey of that size. At $\varepsilon = 0.5$ it is 0.0626, and at $\varepsilon = 0.1$ it is 0.316, larger than the fraction being estimated (@fig-dpdef-rr).

![Left: randomized response at $\varepsilon = \ln 3$ as output probabilities; each report is 3 times more likely under one true answer than under the other. Right: estimates of the fraction of "yes" from 4,000 simulated surveys of 1,000 people with true fraction 0.3, at $\varepsilon = \ln 3$ and $\varepsilon = 0.5$, with the normal curves the standard-deviation formula predicts.](figures/dpdef_rr.svg){#fig-dpdef-rr}

**Pitfalls.** Randomized response protects the value of an answer, not the fact that you answered. It is a **local** mechanism, where each person adds noise before the data leave them, and that costs accuracy; [Local, shuffle and distributed trust models](08-trust-models.md) covers the alternatives.

## What $\varepsilon$ means for an attacker {#sec-dpdef-testing}

**Idea.** An attacker guessing whether the data were $D$ or $D'$ is a hypothesis test, and its decision rule is an event. Since @eq-dpdef-def bounds every event, it bounds every attack's error rates, including attacks not yet invented.

**Result.** Write $\alpha$ for an attack's **false positive rate (FPR)** and $\beta$ for its **false negative rate**, so its **true positive rate (TPR)** is $1 - \beta$. An $(\varepsilon, \delta)$-DP mechanism forces every attack, for every pair of neighbours, to satisfy
$$\alpha + e^\varepsilon\beta \ge 1 - \delta, \qquad \mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta.$$ {#eq-dpdef-testing}
An audit inverts the second form: an attack measured at $(\mathrm{FPR}, \mathrm{TPR})$ shows $\varepsilon \ge \ln\big((\mathrm{TPR} - \delta)/\mathrm{FPR}\big)$. Averaged over a fair coin, the attacker's **advantage**, $\mathrm{TPR} - \mathrm{FPR}$, is at most
$$\mathrm{Adv} \le \frac{e^\varepsilon - 1 + 2\delta}{e^\varepsilon + 1},$$ {#eq-dpdef-advantage}
which is $\tanh(\varepsilon/2) \approx \varepsilon/2$ for pure DP, attained by randomized response. And pure DP lets one output multiply an adversary's odds about a person by at most $e^\varepsilon$, so a prior probability $\pi$ becomes at most $e^\varepsilon\pi/(e^\varepsilon\pi + 1 - \pi)$ (@tbl-dpdef-posterior).

| $\varepsilon$ | Advantage bound | Prior 0.01 | Prior 0.1 | Prior 0.5 |
| --- | --- | --- | --- | --- |
| 0.1 | 0.0500 | 0.0110 | 0.1094 | 0.5250 |
| 1 | 0.4621 | 0.0267 | 0.2320 | 0.7311 |
| $\ln 3$ | 0.5000 | 0.0294 | 0.2500 | 0.7500 |

: The largest membership advantage and the largest posterior probability after one output of an $\varepsilon$-DP mechanism, for three priors. {#tbl-dpdef-posterior}

**Example.** At $\varepsilon = 1$ and $\delta = 10^{-5}$, no attack reaches a TPR above 0.002728 at FPR $10^{-3}$. A count with Gaussian noise of standard deviation 2 is $(1, 0.0068)$-DP; the bound allows TPR 0.0095 at that FPR, and the best real attack reaches 0.0048. If an attack reports TPR 0.05 at FPR 0.001 against a model claimed to be $(2, 10^{-5})$-DP, the claim is false (the cap is 0.0074) or the rates are mis-measured (@sec-dpdef-code).

![Left: in the plane of false positive and false negative rate, the region no test can reach (shaded) for $\varepsilon = \ln 3$, and its boundary for $\varepsilon = 1$ with $\delta = 0$ and $\delta = 0.05$; randomized response sits exactly at the corner. Right: the same bound as the largest TPR against FPR on log axes, for $\varepsilon = 1$ with and without $\delta$, and the best TPR against a count with Gaussian noise of standard deviation 2.](figures/dpdef_tradeoff_region.svg){#fig-dpdef-tradeoff-region}

**Pitfalls.** Large $\varepsilon$ promises little: at $\varepsilon = 5$ the advantage bound is 0.9866, an attacker right 99.3% of the time. A model trained at $\varepsilon = 8$ may be safer in practice, but the guarantee does not say so. The advantage hides the low-FPR regime, where the cap is dominated by $\delta$ (@fig-dpdef-tradeoff-region, right). DP bounds the change in belief, not the belief: an adversary already 99% sure stays about that sure. Audits plant a **canary**, a record built to be easy to detect, since an ordinary record gives a loose lower bound.

## Why $\delta$ must be much smaller than $1/n$ {#sec-dpdef-small-delta}

**Idea.** A deliberately terrible mechanism sets the scale. "Release each record with probability $\delta$" publishes records verbatim, yet it is $(0, \delta)$-DP under add/remove, because any one person is affected only with probability $\delta$.

**Result.** Over $n$ records it publishes $n\delta$ on average, and at least one with probability $1 - (1 - \delta)^n \approx 1 - e^{-n\delta}$. A meaningful guarantee needs $\delta \ll 1/n$, where $n$ counts people, not records.

**Example.** With $n = 10{,}000$ records (@tbl-dpdef-release), $\delta = 1/n$ exposes someone with probability 0.6321; in 100,000 simulated runs it did so in a fraction 0.6351.

| $\delta$ | Expected records published, $n\delta$ | $\Pr[\text{at least one}]$ |
| --- | --- | --- |
| $10^{-4} = 1/n$ | 1 | 0.6321 |
| $10^{-6}$ | 0.01 | $9.950\times10^{-3}$ |
| $10^{-8}$ | 0.0001 | $9.999\times10^{-5}$ |

: Exposure under the $(0, \delta)$-DP mechanism that releases each of 10,000 records with probability $\delta$. {#tbl-dpdef-release}

**Pitfalls.** The advantage bound does not warn about this mechanism, because averaging hides a few people fully exposed. $\delta = 10^{-5}$ suits 60,000 training examples but not 10 million users, who need $\delta$ well below $10^{-7}$. A small $\delta$ usually costs little $\varepsilon$: in @tbl-dpdef-gauss, moving from $\varepsilon = 1$ to 3 cuts $\delta$ from $6.830\times10^{-3}$ to $3.401\times10^{-10}$.

## What differential privacy does not protect {#sec-dpdef-limits}

**Idea.** DP compares a world with your record against one without it, not the world before and after a release, and it treats each record as separate.

**Result.** Three kinds of inference are outside the guarantee. First, what the output reveals about the population: if a study shows smokers often get cancer, beliefs about every smoker change, including those not in the data. Second, correlated records: $k$ records that move together form a group, and under $\varepsilon$-DP per record their privacy loss can reach $k\varepsilon$. Third, anything outside the mathematical mechanism: leaky floating-point samplers ([Discrete and floating-point-safe mechanisms](07-discrete-mechanisms.md)), timing side channels, the stored raw data, other releases of the same data.

**Example.** A household of five who always answer alike, in a count with Laplace noise at $\varepsilon = 1$ per record, faces a largest privacy loss of 5.0000. An attacker asking whether the household took part can be right with probability up to 0.959.

**Pitfalls.** Match the unit of privacy to the unit of correlation, with user- or household-level neighbours; correlations you do not know about are outside the guarantee. A DP proof covers the algorithm, not its implementation, which needs separate review.

## Code {#sec-dpdef-code}

This snippet turns a claimed $(\varepsilon, \delta)$ into attacker terms, and a measured attack into a lower bound on $\varepsilon$:

```python
from math import exp, log, sqrt


def advantage_bound(eps: float, delta: float = 0.0) -> float:
    """The largest membership advantage, TPR - FPR, against any (eps, delta)-DP mechanism."""
    return (exp(eps) - 1 + 2 * delta) / (exp(eps) + 1)


def tpr_cap(eps: float, delta: float, fpr: float) -> float:
    """The largest true positive rate any attack can reach at a given false positive rate."""
    return min(1.0, exp(eps) * fpr + delta)


def epsilon_lower_bound(tpr: float, fpr: float, delta: float) -> float:
    """The smallest eps consistent with an attack measured at (fpr, tpr)."""
    return log((tpr - delta) / fpr)


def posterior_bound(eps: float, prior: float) -> float:
    """The largest posterior probability after one output of an eps-DP mechanism."""
    return exp(eps) * prior / (exp(eps) * prior + 1 - prior)


def rr_sd(eps: float, n: int) -> float:
    """Standard deviation of the randomized-response estimate of a fraction from n people."""
    return exp(eps / 2) / ((exp(eps) - 1) * sqrt(n))


def exposure(delta: float, n: int) -> float:
    """Probability that 'publish each record with probability delta' exposes someone."""
    return 1 - (1 - delta) ** n


print(f"{advantage_bound(1.0):.4f}")  # 0.4621
print(f"{tpr_cap(1.0, 1e-5, 1e-3):.6f}")  # 0.002728
print(f"{tpr_cap(2.0, 1e-5, 1e-3):.4f}")  # 0.0074
print(f"{epsilon_lower_bound(0.05, 1e-3, 1e-5):.2f}")  # 3.91
print(f"{posterior_bound(1.0, 0.01):.4f}")  # 0.0267
print(f"{rr_sd(log(3), 1000):.4f}")  # 0.0274
print(f"{exposure(1e-4, 10_000):.4f}")  # 0.6321
```

The attack at TPR 0.05 and FPR 0.001 shows $\varepsilon \ge 3.91$, far above the claimed 2; a real audit must also account for the uncertainty in measured rates. The full edition's script checks every number in the chapter:

```bash
uv run python code/part03-dp-foundations/definition.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| $(\varepsilon, \delta)$-DP | $\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta$ for all $D \simeq D'$, all $S$ | State the neighbouring relation and unit of privacy |
| Pure DP | $\lvert L_{D,D'}(y)\rvert \le \varepsilon$ for every output | Bounded loss; Laplace noise |
| Approximate DP | $\delta(\varepsilon) = \mathbb E_{M(D)}[(1 - e^{\varepsilon - L})_+]$ | A curve, not a pair; $\Pr[L > \varepsilon]$ is a loose upper bound |
| Add/remove vs replace | $\varepsilon$ under add/remove gives $2\varepsilon$ under replace | Only add/remove hides participation |
| Randomized response | Truth with probability $e^\varepsilon/(1 + e^\varepsilon)$; error $\approx 1/(\varepsilon\sqrt n)$ | $\varepsilon$-DP under replace |
| Attack cap | $\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta$ | Audit: $\varepsilon \ge \ln((\mathrm{TPR} - \delta)/\mathrm{FPR})$ |
| Advantage | $\le (e^\varepsilon - 1 + 2\delta)/(e^\varepsilon + 1)$; $\tanh(\varepsilon/2)$ if pure | Tight for randomized response |
| Posterior | $\le e^\varepsilon\pi/(e^\varepsilon\pi + 1 - \pi)$ from prior $\pi$ | Bounds the change in belief, not the belief |
| Choosing $\delta$ | $(0, \delta)$-DP can expose someone with probability $1 - (1 - \delta)^n$ | $\delta \ll 1/n$, $n$ people |

: The results of this chapter. {#tbl-dpdef-cheat}

**Rules of thumb**

- State the neighbouring relation and the unit of privacy with every $\varepsilon$.
- $\varepsilon$ below 1 is strong, a few is moderate, and around 10 the worst-case guarantee says little.
- Pick $\delta \ll 1/n$, with $n$ the number of people, not records.
- Compare a measured membership attack with the cap $e^\varepsilon\,\mathrm{FPR} + \delta$ at low FPR, not with the advantage.
- DP protects your contribution, not secrets that other people's data reveal about you.

## Further reading

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), ch. 2: the definition, randomized response, what DP does and does not promise, and the "release a random record" argument for small $\delta$.
- Kairouz, Oh & Viswanath, "The Composition Theorem for Differential Privacy" (ICML 2015): the hypothesis-testing reading of DP behind the attack cap and auditing.
- Kasiviswanathan & Smith, "On the 'Semantics' of Differential Privacy: A Bayesian Formulation" (Journal of Privacy and Confidentiality, 2014): the posterior reading, for any prior.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/01-definition.md).
