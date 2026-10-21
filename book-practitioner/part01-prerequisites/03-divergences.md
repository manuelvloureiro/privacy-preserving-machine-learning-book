# Divergences

## What it is {#sec-div-hockey}

A **divergence** puts a number on how different two distributions are. It is zero for equal distributions and grows as they become easier to tell apart. For distributions $P$ and $Q$, write $p(x)$ and $q(x)$ for the probabilities they give an outcome $x$. Every divergence summarizes the **likelihood ratio** $p(x)/q(x)$.

The **Kullback–Leibler (KL) divergence** is the average log ratio when the data come from $P$:
$$D_{\mathrm{KL}}(P\|Q) = \sum_x p(x)\ln\frac{p(x)}{q(x)}.$$
It is not symmetric, and it adds up over independent observations.

The **Rényi divergence** is a dial set by an order $\alpha$ above 1:
$$D_\alpha(P\|Q) = \frac{1}{\alpha-1}\ln\sum_x p(x)^\alpha q(x)^{1-\alpha}.$$
It equals the KL divergence as the order approaches 1. Higher orders weigh the outcomes with large ratios more, up to the **max-divergence** $D_\infty$, the largest log ratio that any outcome gives.

The **total variation (TV)** is the mass of $P$ that does not overlap with $Q$:
$$\mathrm{TV}(P, Q) = \frac12\sum_x|p(x) - q(x)|.$$
The hockey-stick divergence generalizes it by scaling $q$ up by a factor $\gamma$ of at least 1 and measuring the mass of $p$ that still sticks out above it. Writing $(y)_+$ for the positive part of $y$, it is
$$E_\gamma(P\|Q) = \sum_x\big(p(x) - \gamma q(x)\big)_+.$$
At $\gamma = 1$ it is the total variation, and it reaches zero once $\gamma$ exceeds the largest likelihood ratio. The KL divergence, the TV and $E_\gamma$ all belong to one family, the f-divergences.

## When you meet it

You meet divergences when a feature drifts between training and serving, or when a test-data shift may move a classifier's accuracy. Papers state results in one divergence that you may need in another.

## How to use it

To measure drift, bin both samples on the same edges and call `scipy.stats.entropy(p, q)`. It normalizes both histograms and returns natural-log units (nats), or bits with `base=2`. Take a feature whose last bin, empty last month, holds a tenth of this month's data. With last month as $P$, the divergence is 0.1116 nats, but the reverse is infinite, because this month has mass where last month had none.

```python
import math

from scipy import stats

before, after = [0.5, 0.3, 0.2, 0.0], [0.4, 0.3, 0.2, 0.1]
print(f"{stats.entropy(before, after):.4f} {stats.entropy(before, after, base=2):.4f}")  # 0.1116 0.1610
print(stats.entropy(after, before))  # inf
print(math.ceil(2 * 0.9**2 / 0.005), math.ceil((2 * stats.norm.ppf(0.95) / 0.1) ** 2))  # 324 1083
```

For a copy shifted by $s$, closed forms replace histograms. For normal noise, write $\sigma$ for the standard deviation and $\mu = s/\sigma$ for the shift measured in it. Then
$$D_{\mathrm{KL}} = \frac{\mu^2}{2}, \qquad D_\alpha = \frac{\alpha\mu^2}{2}, \qquad D_\infty = \infty,$$
the last because the ratio of normal densities is unbounded. Because the shift enters squared, a drift of 0.1 standard deviations costs a KL divergence of only 0.005. For Laplace noise, write $b$ for the scale and $t = s/b$ for the shift in scales. Its max-divergence is $t$, its TV is $1 - e^{-t/2}$, and its KL divergence is $t + e^{-t} - 1$, which is 0.3679 at $t = 1$.

The TV says how well any test can tell $P$ from $Q$. If the two are equally likely in advance, the best test from one observation has an **advantage**, its true-positive rate minus its false-positive rate, equal to the TV, and it is right with probability $\tfrac12 + \tfrac12\mathrm{TV}$. For two normal curves one standard deviation apart, the TV is 0.3829, so no classifier is right more than 0.6915 of the time. When the test distribution moves, a fixed classifier's accuracy moves by at most the TV.

For the same normal pair, the hockey-stick divergence is
$$E_\gamma = \Phi\Big(\frac\mu2 - \frac{\ln\gamma}\mu\Big) - \gamma\,\Phi\Big(-\frac\mu2 - \frac{\ln\gamma}\mu\Big),$$
where $\Phi$ is the standard normal distribution function. At $\mu = 1$ and $\gamma = 2$ it is 0.1906.

**Pinsker's inequality** converts a KL divergence into a bound on the TV:
$$\mathrm{TV}(P, Q) \le \sqrt{\tfrac12D_{\mathrm{KL}}(P\|Q)}.$$
It is quick but loose. For the drift of 0.1 standard deviations, with KL 0.005 per sample, a test right 95% of the time needs a TV of 0.9, so Pinsker asks for at least 324 samples. The exact answer is 1,083, where the TV of the two sample means, $2\Phi(\sqrt n\,\mu/2) - 1$, reaches 0.9 (the snippet's last call).

## What goes wrong

The KL divergence has a direction, so say which one you computed, and never mix nats with bits. An empty bin makes it infinite, so add a pseudocount, such as half a count per bin, or merge bins, and report which you did. Estimates change with the bins and the sample size, so compare only estimates made the same way. And `scipy.spatial.distance.jensenshannon` returns the square root of the Jensen–Shannon divergence, a symmetric relative of KL.

The **data processing inequality** says that binning, thresholding or any other processing of an observation alone cannot make two distributions easier to tell apart. For normals one standard deviation apart, keeping only whether an observation lies below the midpoint cuts the KL divergence from 0.5 to 0.3090. But it keeps all of the TV, because that threshold is already the best test.

A small KL divergence is an average, so it can hide rare outputs that are overwhelming evidence. When every output matters, bound the worst case with the max-divergence, a high Rényi order or the hockey-stick divergence.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/03-divergences.md).
