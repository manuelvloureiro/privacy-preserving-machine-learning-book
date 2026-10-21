# Divergences

## Why this matters

Many questions in machine learning come down to putting a number on how different two distributions are. A feature drifts in production, and you want to know how far it moved. A test set changes slightly, and you want to know how far a classifier's accuracy can move. A **divergence** gives that number. It is zero when the two distributions are equal and grows as they become easier to tell apart. The cheat sheet says which divergence answers which question.

All of them summarize one quantity. Call the two distributions $P$ and $Q$. For an outcome $x$, write $p(x)$ and $q(x)$ for the probabilities they give to it. Their quotient is the **likelihood ratio**
$$r(x) = \frac{p(x)}{q(x)}.$$
Its logarithm, written $L(x)$ and called the **log likelihood ratio**, is the evidence that the outcome gives for $P$ over $Q$. Later parts use it to bound how much an output reveals.

Natural logarithms give divergences in nats, and base-2 logarithms give bits, with one nat about 1.443 bits. This chapter uses nats, so check the base of any library you use.

## KL divergence {#sec-div-kl}

The **Kullback–Leibler (KL) divergence** is the average evidence per observation when the data come from $P$:
$$D_{\mathrm{KL}}(P\|Q) = \mathbb E_P\left[\ln\frac{p(X)}{q(X)}\right] = \sum_x p(x)\ln\frac{p(x)}{q(x)}.$$
It is never negative, and it is zero only when the two distributions are equal. It is not symmetric, because it averages under its first argument, so always say which direction you computed. Maximum likelihood minimizes the KL divergence of the model from the data, while variational inference minimizes the reverse direction.

Its most useful property is that it adds up over independent observations:
$$D_{\mathrm{KL}}(P_1\times P_2\,\|\,Q_1\times Q_2) = D_{\mathrm{KL}}(P_1\|Q_1) + D_{\mathrm{KL}}(P_2\|Q_2).$$ {#eq-div-kl-additive}
So $n$ independent observations carry $n$ times the divergence of one. That makes it easy to work with many samples or many steps: add up the divergences, and convert to the quantity you need only at the end. Independence is a condition, though, because correlated observations partly repeat each other.

To compute a KL divergence from data, bin both samples on the same edges and call `scipy.stats.entropy(p, q)`, which normalizes the two histograms and returns nats, or sum `scipy.special.rel_entr(p, q)`. For example, take the feature histograms in @sec-div-code, where the last of four bins was empty last month but holds a tenth of this month's data. The KL divergence of last month from this month is 0.1116 nats, or 0.1610 bits. In the other direction it is infinite, because this month has mass in a bin that last month never used. So add a small pseudocount to every bin or use coarser bins, and report the choice. Compare only estimates made with the same bins and sample sizes, because they change with both.

## Rényi divergence and the max-divergence {#sec-div-renyi}

The KL divergence reports the typical evidence. The **Rényi divergence** is a dial from the typical evidence to the worst case, set by a number $\alpha$ greater than 1 called its order:
$$D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\ln\mathbb E_Q\big[r^\alpha\big].$$
As the order goes to 1 it becomes the KL divergence, and it never decreases as the order grows. As the order goes to infinity, it becomes the **max-divergence**, the largest log ratio that any outcome, or any event $A$, can give:
$$D_\infty(P\|Q) = \ln\max_x r(x) = \ln\max_{A}\frac{P(A)}{Q(A)}.$$
Like the KL divergence, every Rényi divergence adds up over independent observations.

The Rényi divergence also bounds how often the evidence is large. For every order, the chance under $Q$ that one observation gives at least $c$ nats of evidence for $P$ is at most $\exp(-\alpha c + (\alpha - 1)D_\alpha)$, and the full edition derives this tail bound. Different orders give different bounds, and none is best for every question. So in practice you compute the divergence on a grid of orders, add each order's values over independent observations, and keep the order that gives the tightest final bound.

## Shifted normal and Laplace pairs {#sec-div-gauss}

The most common pair in practice is a noise distribution and a copy shifted by $s$, such as the same noise added to two values that differ by $s$. For normal noise with standard deviation $\sigma$, write $\mu = s/\sigma$ for the shift in standard deviations. Then
$$D_{\mathrm{KL}}(P\|Q) = \frac{\mu^2}2, \qquad D_\alpha(P\|Q) = \frac{\alpha\mu^2}2, \qquad D_\infty(P\|Q) = \infty.$$ {#eq-div-gauss-renyi}
At $\mu = 1$, for example, the KL divergence is 0.5. Order 2 gives 1, and order 10 already gives 5. Because the dependence on the shift is quadratic, a small shift costs very little. A drift of 0.1 standard deviations gives a KL divergence of only 0.005.

For Laplace noise with scale $b$, write $t = s/b$ for the shift in scales. The worst case is then just the shift:
$$D_\infty(P\|Q) = t,$$
and the KL divergence is $t + e^{-t} - 1$, which is 0.3679 at $t = 1$. The evidence never exceeds $t$, and it equals $t$ on half the mass of $P$.

This difference decides how the two noises are analysed. Laplace noise has a finite worst-case ratio, which you read straight off the noise scale as $s/b$. Normal noise has an infinite max-divergence, because some rare outputs are overwhelming evidence. So any worst-case statement about normal noise must either leave out a small probability of rare outputs or use a finite Rényi order instead. Which pair is "more different" depends on the divergence, too. At equal variance and a shift of one Laplace scale, the Laplace pair has the larger divergence at small orders, but the normal pair overtakes it above order 2.98.

## Total variation {#sec-div-tv}

The **total variation (TV)** says directly how often a test is right. It is the mass of $P$ that does not overlap with $Q$:
$$\mathrm{TV}(P, Q) = \max_A|P(A) - Q(A)| = \frac12\sum_x|p(x) - q(x)|.$$ {#eq-div-tv}
It is symmetric and lies between 0 and 1. When $P$ and $Q$ are equally likely a priori, the best possible test from one observation is right with probability
$$\Pr[\text{correct}] = \frac12 + \frac12\mathrm{TV}(P, Q).$$ {#eq-div-tv-test}
No classifier can beat this. For two normal curves one standard deviation apart, the total variation is 0.3829, so the best test is right with probability 0.6915.

The total variation also bounds how far any fixed statistic between 0 and 1 can move. Accuracy is such a statistic, so when a test distribution moves by a total variation of at most 0.01, the accuracy of any fixed classifier moves by at most 0.01.

## Hockey-stick divergence {#sec-div-hockey}

Scale $q$ up by a factor $\gamma$ of at least 1, and measure the mass of $p$ that still sticks out above $\gamma q$. That mass is the **hockey-stick divergence**:
$$E_\gamma(P\|Q) = \max_A\big[P(A) - \gamma Q(A)\big] = \sum_x\big(p(x) - \gamma q(x)\big)_+.$$
Here $(y)_+$ is the positive part of $y$. With no scaling it is the total variation. It falls as $\gamma$ grows, and reaches zero once $\gamma$ reaches the largest likelihood ratio.

So the hockey-stick divergence measures how much of the difference between two distributions comes from likelihood ratios above $\gamma$, and later parts use it in that form. It also gives the best test when the two hypotheses are not equally likely, with $\gamma$ set to the prior odds. For a normal pair shifted by $\mu$ standard deviations, with $\Phi$ the standard normal cumulative distribution function,
$$E_\gamma = \Phi\Big(\frac\mu2 - \frac{\ln\gamma}\mu\Big) - \gamma\,\Phi\Big(-\frac\mu2 - \frac{\ln\gamma}\mu\Big).$$ {#eq-div-gauss-hockey}
It is positive for every $\gamma$, because the normal likelihood ratio is unbounded. At $\mu = 1$ it gives 0.1906 for $\gamma = 2$.

## f-divergences {#sec-div-f}

The KL divergence, the total variation and the hockey-stick divergence are all members of one family. Each is the average under $Q$ of a convex function $f$ of the likelihood ratio, with $f(1) = 0$:
$$D_f(P\|Q) = \mathbb E_Q\big[f(r(X))\big].$$
The $\chi^2$ divergence and the squared Hellinger distance are two more members. Every f-divergence obeys the data processing inequality below, and the Rényi divergence, which is an increasing function of one, inherits that.

## The data processing inequality {#sec-div-dpi}

You rarely use raw outputs, because you threshold a score, round a statistic, fine-tune a model, or pass it through any other algorithm. The **data processing inequality (DPI)** says that none of this can make two distributions easier to tell apart. Describe the processing as a **channel** $K$, which maps each input to a distribution over outputs, and write $PK$ and $QK$ for the two output distributions. Then every f-divergence and every Rényi divergence satisfies
$$D_f(PK\|QK) \le D_f(P\|Q).$$
That is because processing averages likelihood ratios together, and averaging can only weaken evidence.

The condition is that the processing behaves the same whatever the input came from, so it must not look at the original data again. Under that condition, thresholding, rounding or retraining on other data can never make the two distributions easier to tell apart. How much processing loses depends on the divergence. For two unit normal curves one standard deviation apart, a single bit that says whether the observation lies below the midpoint keeps 0.3090 of the KL divergence of 0.5. But it keeps all of the total variation, because it is the best test.

Every divergence here is an average or a maximum over outcomes, and only the maximum controls every output. A small KL divergence can hide a few outputs that are overwhelming evidence. So when every output matters, bound the worst case, through the max-divergence, a high Rényi order or the hockey-stick divergence, not only the average.

## Pinsker's inequality {#sec-div-pinsker}

**Pinsker's inequality** converts a KL divergence into a bound on the total variation:
$$\mathrm{TV}(P, Q) \le \sqrt{\tfrac12D_{\mathrm{KL}}(P\|Q)}.$$ {#eq-div-pinsker}
It is a quick conversion, but often loose, and it says nothing once the KL divergence exceeds 2. It also gives a floor on sample sizes. For a drift of 0.1 standard deviations, no test reaches 95% accuracy with fewer than 324 samples, while the exact answer for this normal pair is 1,083.

## Code {#sec-div-code}

The functions below compute a discrete KL divergence, the hockey-stick divergence of a normal pair, and the Pinsker floor on a sample size.

```python
import math
from statistics import NormalDist

Z = NormalDist()


def kl(p: list[float], q: list[float]) -> float:
    """KL divergence of P from Q in nats; infinite if Q misses an outcome that P can produce."""
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi == 0:
                return math.inf
            total += pi * math.log(pi / qi)
    return total


def gauss_hockey(mu: float, gamma: float) -> float:
    """Hockey-stick divergence E_gamma between N(0, 1) and N(mu, 1)."""
    c = math.log(gamma)
    return Z.cdf(mu / 2 - c / mu) - gamma * Z.cdf(-mu / 2 - c / mu)


def min_samples(kl_one: float, accuracy: float) -> int:
    """Pinsker: no test reaches this accuracy (equal priors) with fewer samples."""
    return math.ceil(2 * (2 * accuracy - 1) ** 2 / kl_one)


def gauss_samples(mu: float, accuracy: float) -> int:
    """Samples the best test needs for N(0, 1) against N(mu, 1), exactly."""
    return math.ceil((2 * Z.inv_cdf(accuracy) / mu) ** 2)


before, after = [0.5, 0.3, 0.2, 0.0], [0.4, 0.3, 0.2, 0.1]
print(f"{kl(before, after):.4f} {kl(after, before)}")  # 0.1116 inf
print(f"{kl(before, after) / math.log(2):.4f}")  # 0.1610
print(f"{gauss_hockey(1.0, 2.0):.4f}")  # 0.1906
print(min_samples(0.005, 0.95), gauss_samples(0.1, 0.95))  # 324 1083
```

## Cheat sheet

| Question | Divergence | Normal pair, shift $\mu$ | Laplace pair, shift $t$ |
| --- | --- | --- | --- |
| Average evidence; drift; adds over samples | KL, $\mathbb E_P[\ln r]$ | $\mu^2/2$ | $t + e^{-t} - 1$ |
| Typical to worst-case evidence, order by order | Rényi $D_\alpha$ | $\alpha\mu^2/2$ | finite, at most $t$ |
| Worst-case ratio over all events | $D_\infty$, $\ln\max r$ | $\infty$ | $t$ |
| Best test accuracy on balanced data | TV | $2\Phi(\mu/2) - 1$ | $1 - e^{-t/2}$ |
| Mass of ratios above $\gamma$ | $E_\gamma$ | @eq-div-gauss-hockey | zero once $\ln\gamma \ge t$ |

: Which divergence answers which question, for a normal pair shifted by $\mu$ standard deviations and a Laplace pair shifted by $t$ scales. {#tbl-div-cheat}

**Rules of thumb**

- KL is not symmetric, so state its direction.
- Don't mix nats and bits.
- Normal noise has no finite worst-case ratio, so a worst-case statement about it must leave out rare outputs.
- A small average divergence does not control every output.
- Processing that does not look at the original data never increases any divergence.
- Add divergences over independent observations, and convert to TV or a tail bound at the end.

## Further reading

- van Erven & Harremoës, "Rényi Divergence and Kullback–Leibler Divergence" (IEEE Transactions on Information Theory, 2014): the Rényi family and its properties.
- Cover & Thomas, *Elements of Information Theory* (2nd ed., 2006), ch. 2 and §11.6: KL and Pinsker's inequality.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/03-divergences.md).
