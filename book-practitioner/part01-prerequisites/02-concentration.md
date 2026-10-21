# Concentration inequalities

## Why this matters

Almost every number in a machine-learning system is random. Some are random by design, like noise deliberately added to a published count, and others because of how they were measured, like an error rate on a test set. Before you publish or act on such a number, you need an error bar for it. And before you collect data, you need to know how much data buys the error bar you want.

A concentration inequality gives you both. It bounds the chance that a random quantity lands more than some distance from its mean. This chapter calls that event a failure, its chance the **failure probability** $\beta$, and the distance the **width**. You use every inequality the same way: fix $\beta$ and solve for the width, or fix the width and solve for the number of samples. Which inequality to use depends on what you know about the quantity, and the more you know, the narrower the error bar. At $\beta = 0.05$ the choice matters little, but at the one-in-a-million level that later chapters need, the wrong choice makes an error bar a hundred times too wide.

## Mean or variance only: Markov and Chebyshev {#sec-conc-chebyshev}

**Markov's inequality** needs only the average of a quantity $X$ that cannot be negative, such as a cost or a latency. It bounds the chance that $X$ reaches a threshold $a$:
$$\Pr[X \ge a] \le \frac{\mathbb E[X]}{a}.$$
Setting the right side to $\beta$ says that $X$ stays below $\mathbb E[X]/\beta$ except with probability $\beta$. If requests take 2 ms on average, at most a quarter take 8 ms or more. The condition matters, because a quantity that can be negative can hide large values behind negative ones.

**Chebyshev's inequality** adds the variance. Write $\mu$ for the mean and $\sigma^2$ for the variance. The chance of landing a distance $t$ from the mean is then
$$\Pr[|X - \mu| \ge t] \le \frac{\sigma^2}{t^2}.$$
So the width is $\sigma/\sqrt\beta$. For an average of $n$ independent samples, divide it by $\sqrt n$. It holds for every distribution, which makes it safe when you distrust the shape of the data. The price is steep at small $\beta$, though. At a 95% level it asks for 4.5 standard deviations where a normal-shaped tail needs 2.4. At one in a million, it asks for 1,000 where the normal needs 5.3. So use it for a quick 95% error bar and for nothing stricter.

## Noise you added: Laplace and Gaussian {#sec-conc-noise}

When the randomness is noise you added yourself, you know its distribution exactly, so you read the width off its tail and need no inequality. Laplace noise with scale $b$ stays within its width
$$b\ln(1/\beta)$$
of zero, except with probability $\beta$.
For normal noise with standard deviation $\sigma$, multiply $\sigma$ by the normal quantile at $1-\beta/2$, which is 1.96 at 95%. The bound $\sigma\sqrt{2\ln(1/\beta)}$ is a convenient upper limit for mental arithmetic. In code, `scipy.stats.laplace.isf` and `scipy.stats.norm.isf` give the exact quantiles, as does the standard-library snippet in @sec-conc-code.

The typical use is the error bar you publish with a noisy count. If the count carries Laplace noise of scale 2, the published value is within 5.99 of the true one with 95% probability. Later parts choose the noise scale; this chapter only reads off its error bar. Publish that number next to the count, because an analyst who treats it as exact will over-read small differences.

A sum of $k$ noisy counts needs care. Near its centre the total noise looks normal, with standard deviation $b\sqrt{2k}$, but its far tail decays like a single Laplace draw, so the normal approximation understates rare failures. For a sum of 10 draws, it understates the chance of a large total about ninefold. A valid one-sided width is the larger of
$$\sqrt{8kb^2\ln(1/\beta)} \quad\text{and}\quad 4b\ln(1/\beta).$$
For two sides, use $\beta/2$, which gives 17.18 at 95% for 10 counts of scale 1. If that is too wide, simulate the noise a million times and read off the quantile, since you know its distribution exactly.

## Many numbers at once: the union bound {#sec-conc-union}

A dashboard of $k$ noisy counts should be right everywhere at once, not just count by count. The **union bound** says the chance that any of several failures happens is at most the sum of their chances. So give each count the failure probability $\beta/k$. Because the widths above depend on $\beta$ only through a logarithm, this is cheap. For 1,000 counts with Laplace scale 2, the width that covers them all is 19.81, against 5.99 for one. The union bound needs no independence, so it is always valid, and for rare independent failures it is nearly exact.

## Counts: the Chernoff bound {#sec-conc-chernoff}

Batch sizes under random sampling, the records that match a query and the errors on a test set are all counts of independent yes-or-no outcomes. Write $\mu$ for the mean of a count $S$, and $\delta$ for a relative deviation between 0 and 1. The **Chernoff bound** then says
$$\Pr[S \ge (1+\delta)\mu] \le e^{-\delta^2\mu/(2+\delta)}, \qquad \Pr[S \le (1-\delta)\mu] \le e^{-\delta^2\mu/2}.$$
For example, suppose each example joins a batch with probability 0.01, so that the batch size has mean 600. It leaves $600 \pm 20\%$ with probability at most $2.4\times10^{-5}$ per step. Over 1,000 steps, that is about 0.024. That is good enough for sizing memory, but code that silently assumes a fixed batch size will still break on the rare large batch, so handle it explicitly.

Two things go wrong in practice. The bound depends on the expected count, not on the number of trials, so a count of rare events has a wide relative error bar however many trials you run. And it needs independence: with several rows per user, count users.

## Averages of bounded values: Hoeffding, Bernstein, Clopper–Pearson {#sec-conc-hoeffding}

Often you know only that each value lies in an interval of length $R$, as for a clipped loss or a 0/1 mistake. Then **Hoeffding's inequality** bounds the error of an average of $n$ independent values. With probability $1-\beta$, the average is within
$$R\sqrt{\frac{\ln(2/\beta)}{2n}}$$
of its mean. Its main use is planning, because it needs nothing you don't know in advance. Solving for $n$ gives
$$n \ge \frac{R^2\ln(2/\beta)}{2t^2}$$
for a width $t$, so measuring accuracy to within one percentage point at 95% takes 18,445 test examples. When you pick the best of $m$ models on a validation set, replace $\beta$ by $\beta/m$, because the winner is the one most likely to look too good.

Hoeffding's inequality is too loose for small rates, because it assumes the largest variance the range allows. For a classifier with 1 false positive in 1,000 negatives, it adds 0.039 to the measured rate. **Bernstein's inequality** uses the variance $\sigma^2$ instead, with a one-sided width of about
$$\sqrt{\frac{2\sigma^2\ln(1/\beta)}{n}} + \frac{2\ln(1/\beta)}{3n}$$
for values in $[0, 1]$, and adds only 0.0045 here. Plugging in a variance estimated from the same data is a heuristic, though. For rates, use the exact binomial answer instead, the **Clopper–Pearson** interval. For the classifier, its 95% upper bound is 0.0047. With no errors at all it would be 0.0030, the "rule of three": zero failures in $n$ trials bound the rate by about $3/n$. Plan with Hoeffding and report with Clopper–Pearson ([Hypothesis testing](04-hypothesis-testing.md)).

## Functions and adaptive processes: McDiarmid and Azuma {#sec-conc-mcdiarmid}

**McDiarmid's inequality** extends Hoeffding's to any function of independent inputs, as long as changing one input, the $i$-th, moves the result by at most $c_i$. The tail is then $\exp(-2t^2/\sum_i c_i^2)$, and [Algorithmic stability and generalization](07-stability-generalization.md) uses it for the error of a retrained model. The $c_i$ is the worst-case influence of one record, a quantity later parts return to.

**Azuma's inequality** gives the same tail for a running total whose steps may depend on the past but have expected value zero given it, a **martingale**, with $c_i$ replaced by the length of each step's interval. Use it when an analyst or a training loop picks each step after seeing the last, which breaks the independence the other bounds need.

## Code {#sec-conc-code}

The functions below compute the widths, sample sizes and rate bounds of this chapter, and print the numbers quoted above.

```python
import math
from statistics import NormalDist


def laplace_width(b: float, beta: float, k: int = 1) -> float:
    """All k draws of Lap(b) noise stay within this distance of zero, except with probability beta."""
    return b * math.log(k / beta)


def gaussian_width(sigma: float, beta: float, k: int = 1) -> float:
    """The same for N(0, sigma^2) noise, from the exact normal quantile."""
    return sigma * NormalDist().inv_cdf(1 - beta / (2 * k))


def laplace_sum_width(k: int, b: float, beta: float) -> float:
    """Two-sided width for the sum of k independent Lap(b) noise draws, from the bound."""
    log_term = math.log(2 / beta)
    return max(math.sqrt(8 * k * b**2 * log_term), 4 * b * log_term)


def chernoff_two_sided(mu: float, delta: float) -> float:
    """Chance that a count with mean mu lands more than a fraction delta from it (delta <= 1)."""
    return math.exp(-(delta**2) * mu / (2 + delta)) + math.exp(-(delta**2) * mu / 2)


def hoeffding_n(width: float, beta: float, value_range: float = 1.0) -> int:
    """Samples needed for an average to be within width of its mean, except with probability beta."""
    return math.ceil(value_range**2 * math.log(2 / beta) / (2 * width**2))


def rate_upper_bound(errors: int, n: int, beta: float) -> float:
    """Exact (Clopper-Pearson) one-sided upper bound on a rate measured as errors out of n."""

    def cdf(p: float) -> float:
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(errors + 1))

    lo, hi = errors / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) > beta else (lo, mid)
    return hi


print(f"{laplace_width(2, 0.05):.2f}")  # 5.99
print(f"{laplace_width(2, 0.05, k=1000):.2f}")  # 19.81
print(f"{gaussian_width(2, 0.05):.2f}")  # 3.92
print(f"{laplace_sum_width(10, 1, 0.05):.2f}")  # 17.18
print(f"{chernoff_two_sided(600, 0.2):.1e}")  # 2.4e-05
print(hoeffding_n(0.01, 0.05))  # 18445
print(f"{rate_upper_bound(1, 1000, 0.05):.4f}")  # 0.0047
print(f"{rate_upper_bound(0, 1000, 0.05):.4f}")  # 0.0030
```

## Cheat sheet

| What you know | Use | Width at failure probability $\beta$ |
| --- | --- | --- |
| $X \ge 0$ and its mean | Markov | Stays below $\mathbb E[X]/\beta$ |
| The variance $\sigma^2$ | Chebyshev | $\sigma/\sqrt\beta$; only for a quick 95% bar |
| Laplace noise $\mathrm{Lap}(b)$ | Its exact tail | $b\ln(1/\beta)$; $k$ draws: $b\ln(k/\beta)$ |
| Normal noise $\mathcal N(0, \sigma^2)$ | Its exact quantile | $\sigma \cdot z_{1-\beta/2}$, at most $\sigma\sqrt{2\ln(1/\beta)}$ |
| A sum of $k$ Laplace draws | The sum bound, or simulation | $\max\big(\sqrt{8kb^2\ln(1/\beta)},\ 4b\ln(1/\beta)\big)$, one-sided |
| A count with mean $\mu$ | Chernoff | $\approx\sqrt{2\mu\ln(1/\beta)}$, one-sided |
| An average of $n$ values in a range $R$ | Hoeffding | $R\sqrt{\ln(2/\beta)/(2n)}$; plan $n$ with it |
| The same, with a small variance | Bernstein | $\sqrt{2\sigma^2\ln(1/\beta)/n} + 2\ln(1/\beta)/(3n)$, one-sided |
| A rate: errors out of $n$ | Clopper–Pearson | Exact; zero errors give about $3/n$ at 95% |
| One input moves $f$ by at most $c_i$ | McDiarmid | $\sqrt{\sum_i c_i^2\ln(1/\beta)/2}$, one-sided |
| Adaptive steps of length $d_k$ | Azuma | $\sqrt{\sum_k d_k^2\ln(1/\beta)/2}$, one-sided |

: Which bound to use, and the width it gives. Two-sided unless marked; for two sides from a one-sided width, use $\beta/2$. {#tbl-concentration-cheat-sheet}

**Rules of thumb**

- Use the most you know: an exact distribution beats any bound. Knowing only the variance costs a width growing like $1/\sqrt\beta$, a normal-shaped tail like $\sqrt{\ln(1/\beta)}$, and a Laplace-shaped tail like $\ln(1/\beta)$.
- Publish the error bar with every noisy number.
- For $k$ numbers at once, divide $\beta$ by $k$.
- Match the sides: a one-sided width covers only one direction, so for a two-sided error bar use $\beta/2$.
- Hoeffding needs bounded values: clip unbounded data first, and use the clipping range as $R$.
- Plan sample sizes with Hoeffding; report rates with Clopper–Pearson.
- Never use the normal approximation for a guarantee in the far tail.
- Independence is a condition: with several records per person, count people.

## Further reading

- Mitzenmacher & Upfal, *Probability and Computing*, ch. 4: Chernoff bounds with computer-science examples.
- Boucheron, Lugosi & Massart, *Concentration Inequalities* (2013): the reference, including Bernstein's inequality.

**Derivations.** Every result here except Bernstein's inequality is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/02-concentration.md).
