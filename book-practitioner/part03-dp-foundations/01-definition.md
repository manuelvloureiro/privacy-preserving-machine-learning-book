# The definition of differential privacy

## Why this matters

Every attack in [Threat models and attacks](../part02-attacks/00-introduction.md) worked the same way. The attacker looked at what a system released and asked which of two worlds produced it: the one where a given person's record was in the data, or the one where it was not. Membership inference asks this about a trained model, reconstruction asks it about a table of statistics, and each succeeded because the output looked different enough in the two worlds for a test to tell them apart.

So the defence has to make the two worlds look alike, not just to today's attacks but to every test anyone could run. **Differential privacy (DP)** is that requirement written down. It is a property of the algorithm that produces the release, so you can check it before any data are touched, and it holds whatever the attacker knows or computes.

## What "one person" means {#sec-dpdef-neighbours}

The guarantee compares two datasets that differ in one person, and the **neighbouring relation** says what that means. There are two standard choices. Under **add/remove**, the second dataset is the first with one record added or taken away. The attacker's question is then "is this person in the data at all?", which is exactly the membership question, and the dataset's size is private too. Under **replace**, both datasets have the same size and one record's value is swapped for another. Now the attacker is assumed to know who is in the data, and the guarantee only hides what their record says.

Replacing Alice's record by Dan's is removing Alice and then adding Dan: two add/remove steps, each allowed to shift the output by the guaranteed amount. So $\varepsilon$ under add/remove becomes $2\varepsilon$ under replace. The reverse fails, because a replace guarantee never compares datasets of different sizes: publishing the number of records is perfectly private under replace and not private at all under add/remove.

You can see the factor of 2 in a histogram of smokers and non-smokers, noised for $\varepsilon = 1$ per record. Removing Carol changes one bar by one, and the largest privacy loss (defined in the next section) is 1.0000. Changing Alice's answer from smoker to non-smoker moves two bars at once, one down and one up, and the loss doubles to 2.0000.

The second choice is what a record is. If each training example is a record, the guarantee protects each example. A person who wrote 200 messages is then 200 records, and hiding the person means hiding all 200 at once, which the guarantee covers only at a much higher price ([Post-processing, group privacy and composition](02-properties.md)). So always ask of a claimed $\varepsilon$: per what? The answer should be the unit you care about, usually the person.

Neighbours range over every record the data format allows, including a person aged 900. A mechanism must hide even that one, which is why private algorithms clip their inputs first.

## The definition {#sec-dpdef-definition}

Run a randomized algorithm $M$, called a **mechanism**, on neighbouring datasets $D$ and $D'$, and look at an output $y$. The natural measure of what $y$ reveals is how much more likely it is in one world than in the other. Its logarithm is the **privacy loss**,
$$L_{D,D'}(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)},$$
where $p_{M(D)}(y)$ is the probability (or density) of output $y$ on dataset $D$. A loss of 0 tells the attacker nothing; a loss of 3 means the output is $e^3 \approx 20$ times more likely with the record, which is strong evidence.

Why a ratio and not a difference? A change from 0.50 to 0.51 and one from 0 to 0.01 are the same difference, but the second creates an output possible only when the person is present, which identifies them with certainty. Only a ratio, infinite here, catches it.

Differential privacy bounds this ratio. $M$ is **$(\varepsilon, \delta)$-differentially private** if for every pair of neighbours and every set $S$ of outputs,
$$\Pr[M(D) \in S] \le e^\varepsilon\,\Pr[M(D') \in S] + \delta.$$ {#eq-dpdef-def}
The probabilities are over the mechanism's randomness, with the data fixed. With $\delta = 0$ this is **pure DP**: no event becomes more than $e^\varepsilon$ times more likely, which is the same as saying the privacy loss of every single output is at most $\varepsilon$. At $\varepsilon = 0.1$ an event of probability 0.5 in one world has probability at most $0.5e^{0.1} = 0.553$ in the other, so the release barely moves anyone's odds.

Pure DP is too strict for some useful mechanisms. With Gaussian noise, far-out outputs are much more likely in one world than the other, because the normal density falls off faster than any fixed ratio can follow. Those outputs are astronomically rare, and $\delta$ is the allowance for them: the ratio bound may fail, but only on outputs whose total excess probability is at most $\delta$. This is **approximate DP**.

The way $\delta$ is counted matters in practice. The definition allows each output's probability $p$ in one world to be up to $e^\varepsilon$ times its probability $q$ in the other, and only the excess, $p - e^\varepsilon q$, has to be paid for by $\delta$. An output with loss $L$ has $p = e^L q$, so the excess is a fraction $1 - e^{\varepsilon - L}$ of its probability: close to zero when $L$ is just above $\varepsilon$, close to all of it when $L$ is huge. The smallest valid $\delta$, computed this way, is the **hockey-stick divergence** between the two output distributions. So $\delta$ is not "the probability that privacy fails": that probability is larger, because it counts outputs just over the line in full. For a count released with Gaussian noise of standard deviation 2 at $\varepsilon = 1$, that probability is $4.006\times10^{-2}$ while the true $\delta$ is $6.830\times10^{-3}$, nearly six times smaller (@fig-dpdef-ratio-band). If you calibrate with the tail probability, you add noise you didn't need.

![The definition as a picture, for a count released with noise at $\varepsilon = 1$. The shaded band runs from $e^{-\varepsilon}$ to $e^\varepsilon$ times the output density on $D'$ (orange). Left: with Laplace noise of scale $1/\varepsilon$, the density on $D$ (blue) stays inside the band, so the release is pure DP. Right: with Gaussian noise of standard deviation 2, it leaves the band in the tail, and the red mass is what $\delta$ must cover.](figures/dpdef_ratio_band.svg){#fig-dpdef-ratio-band}

Ask for a larger $\varepsilon$ and fewer outputs exceed it, so the required $\delta$ shrinks: for the Gaussian count above, $\delta$ falls from $6.830\times10^{-3}$ at $\varepsilon = 1$ to $3.401\times10^{-10}$ at $\varepsilon = 3$. A mechanism therefore has a whole **privacy curve** $\delta(\varepsilon)$, and any single pair you quote is one point on it. Report the mechanism's parameters along with the pair, because the accountants of [Part 4](../part04-privacy-accounting/00-introduction.md) combine curves, not points.

Never check approximate DP output by output: the slack is shared across a whole event. Two datasets whose outputs are spread over 1,000 disjoint values pass that check with $\delta = 0.0010$, yet one look at any output reveals the dataset, and the true $\delta$ is 1.0000.

## Randomized response {#sec-dpdef-rr}

The oldest mechanism shows how privacy and useful statistics coexist. To learn what fraction of people did something embarrassing, ask each person to flip a coin in private: on heads, answer truthfully; on tails, flip again and answer "yes" on heads, "no" on tails. Any single "yes" might be a coin, so it proves nothing about the person. But the coins behave predictably over many people, so the fraction can be recovered.

This scheme reports the truth with probability $t = 0.75$. A "yes" is then 0.75/0.25 = 3 times more likely from someone whose true answer is yes than from someone whose answer is no, and no report has a larger ratio, so this is pure DP with $\varepsilon = \ln 3 = 1.0986$. In general, **randomized response** at level $\varepsilon$ tells the truth with probability $t = e^\varepsilon/(1 + e^\varepsilon)$, chosen so that the ratio is exactly $e^\varepsilon$.

The lying pulls the share $\bar y$ of "yes" reports towards one half by a known amount, and undoing that pull gives an unbiased estimate of the true fraction $\pi$,
$$\hat\pi = \frac{\bar y - (1 - t)}{2t - 1}, \qquad \operatorname{sd}(\hat\pi) \approx \frac{1}{\varepsilon\sqrt n}\ \text{ for small } \varepsilon.$$
The cost of privacy is visible in the denominator. When $\varepsilon$ is small, $t$ is close to one half, the reports are nearly pure coin flips, and $2t - 1$ is small; dividing by it blows up the noise. Halving $\varepsilon$ roughly doubles the error, and since error falls like $1/\sqrt n$, you need four times as many people to win it back.

In a survey of 1,000 people with true fraction 0.3, the estimate's standard deviation is 0.0274 at $\varepsilon = \ln 3$, about twice the 0.0145 of an ordinary survey of that size. At $\varepsilon = 0.5$ it is 0.0626, and at $\varepsilon = 0.1$ it is 0.316, larger than the fraction being measured (@fig-dpdef-rr).

![Left: randomized response at $\varepsilon = \ln 3$ as output probabilities; each report is 3 times more likely under one true answer than under the other. Right: estimates of the fraction of "yes" from 4,000 simulated surveys of 1,000 people with true fraction 0.3, at $\varepsilon = \ln 3$ and $\varepsilon = 0.5$, with the normal curves the standard-deviation formula predicts.](figures/dpdef_rr.svg){#fig-dpdef-rr}

Because each person randomizes before their answer leaves them, nobody needs to be trusted. That is the **local** model, and the large error is its price; a trusted curator who noises the aggregate once does far better, and [Local, shuffle and distributed trust models](08-trust-models.md) compares the options.

## What $\varepsilon$ means for an attacker {#sec-dpdef-testing}

Whatever an attacker does, they end by deciding "in" or "out", and the set of outputs on which they say "in" is an event. So the definition bounds how much more often they can say "in" when it is true than when it is false.

Write the attacker's **true positive rate (TPR)** for how often they say "in" when the person is in, and their **false positive rate (FPR)** for how often they say "in" when the person is out. Applying @eq-dpdef-def to their "in" event gives, for every possible attack,
$$\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta.$$ {#eq-dpdef-testing}
This is the form to remember: an attacker who must rarely accuse wrongly keeps the FPR small, and the bound then caps the TPR at a small multiple of it. At $\varepsilon = 1$ and $\delta = 10^{-5}$, no attack reaches a TPR above 0.002728 at an FPR of $10^{-3}$.

Read backwards, the same inequality turns an attack into an audit. If someone claims a model is $(2, 10^{-5})$-DP and you run an attack that reaches a TPR of 0.05 at an FPR of 0.001, the claim cannot be right: the cap at that FPR is 0.0074. Solving for $\varepsilon$ shows the attack proves $\varepsilon \ge 3.91$ (@sec-dpdef-code). Audits use **canaries**, records planted to be as easy to detect as possible, because an ordinary record would give a much weaker bound.

Two other readings follow from the same bound. Averaged over a fair coin for "in" or "out", an attacker's **advantage**, $\mathrm{TPR} - \mathrm{FPR}$, is at most $\tanh(\varepsilon/2)$ under pure DP, about $\varepsilon/2$ when $\varepsilon$ is small. And because one output can change the likelihood ratio by at most $e^\varepsilon$, it can multiply anyone's odds that the person is in by at most $e^\varepsilon$, whatever they believed before (@tbl-dpdef-posterior).

| $\varepsilon$ | Advantage bound | Prior 0.01 | Prior 0.1 | Prior 0.5 |
| --- | --- | --- | --- | --- |
| 0.1 | 0.0500 | 0.0110 | 0.1094 | 0.5250 |
| 1 | 0.4621 | 0.0267 | 0.2320 | 0.7311 |
| $\ln 3$ | 0.5000 | 0.0294 | 0.2500 | 0.7500 |

: The largest membership advantage, and the largest posterior probability after one output of an $\varepsilon$-DP mechanism for three prior beliefs. {#tbl-dpdef-posterior}

These readings also show where the guarantee weakens. At $\varepsilon = 5$ the advantage bound is 0.9866, an attacker right 99.3% of the time: a model trained at $\varepsilon = 8$ may resist real attacks, but the guarantee no longer promises it. And the release bounds the change in a belief, not the belief: an adversary already 99% sure from other sources stays about that sure.

## How small $\delta$ must be {#sec-dpdef-small-delta}

A deliberately terrible mechanism shows how small $\delta$ must be. Publish each record, in full, with probability $\delta$. Any one person is exposed only with probability $\delta$, and otherwise the output ignores them, so this is $(0, \delta)$-DP.

But it publishes $n\delta$ records on average, and at least one with probability about $1 - e^{-n\delta}$. With $n = 10{,}000$ and $\delta = 1/n$, someone's record is published verbatim with probability 0.6321. So $\delta$ must be much smaller than $1/n$, where $n$ counts people, not records: $10^{-5}$ suits 60,000 training examples, but 10 million users need $\delta$ well below $10^{-7}$.

This usually costs little: reasonable mechanisms, like the Gaussian count, spread $\delta$ over many outputs of moderate loss, so their privacy curve falls steeply and a slightly larger $\varepsilon$ buys a much smaller $\delta$.

## What differential privacy does not protect {#sec-dpdef-limits}

The guarantee compares the world with your record to the world without it, so anything learned in both worlds is outside it. If a study shows that smokers often get cancer, beliefs about every smoker change, including those never in the data; that is the study doing its job.

It also treats records as separate. If five people in a household always answer alike, an attacker who learns one answer learns all five, and hiding the household means hiding five records at once. Under $\varepsilon = 1$ per record, the household's privacy loss can reach 5.0000, and an attacker asking whether the household took part can be right with probability up to 0.959. Match the unit of privacy to the unit of correlation: protect households or users, not rows.

Finally, a DP proof covers the algorithm, not the code: floating-point noise samplers can leak through their low-order bits ([Discrete and floating-point-safe mechanisms](07-discrete-mechanisms.md)), and timing can leak. The implementation needs its own review.

## Code {#sec-dpdef-code}

This snippet turns a claimed $(\varepsilon, \delta)$ into what an attacker can achieve, and a measured attack into a lower bound on $\varepsilon$:

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

A real audit must also account for the uncertainty in measured rates ([Membership inference](../part02-attacks/02-membership-inference.md)). The full edition's script checks every number in the chapter:

```bash
uv run python code/part03-dp-foundations/definition.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| $(\varepsilon, \delta)$-DP | $\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta$ for all neighbours, all $S$ | State the neighbouring relation and unit of privacy |
| Add/remove vs replace | $\varepsilon$ under add/remove gives $2\varepsilon$ under replace | Only add/remove hides participation |
| Randomized response | Truth with probability $e^\varepsilon/(1 + e^\varepsilon)$; error $\approx 1/(\varepsilon\sqrt n)$ | Local: no trusted collector |
| Attack cap | $\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta$ | Audit: $\varepsilon \ge \ln((\mathrm{TPR} - \delta)/\mathrm{FPR})$ |
| Posterior | Odds multiplied by at most $e^\varepsilon$ | Bounds the change in belief, not the belief |
| Choosing $\delta$ | $\delta \ll 1/n$, $n$ people | $\delta = 1/n$ can expose someone with probability 0.63 |

: The results of this chapter. {#tbl-dpdef-cheat}

## Further reading

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), ch. 2: the definition, randomized response, what DP does and does not promise.
- Kairouz, Oh & Viswanath, "The Composition Theorem for Differential Privacy" (ICML 2015): the hypothesis-testing reading behind the attack cap and auditing.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/01-definition.md).
