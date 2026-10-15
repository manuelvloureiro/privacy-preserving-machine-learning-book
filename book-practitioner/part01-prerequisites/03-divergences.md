# Divergences

## Why this matters

How far is a fitted model from the truth? How many samples does it take to see a drift? Later parts ask the same about the outputs of an algorithm run on two different inputs. Each time you need one number for **how different two distributions $P$ and $Q$ are**, and each candidate number answers a different question.

The one idea is the **likelihood ratio** $r(x) = p(x)/q(x)$, how many times more likely an outcome is under $P$ than under $Q$. Its logarithm $L(x) = \ln r(x)$ is the evidence for $P$, in nats. Every divergence is a summary of $r$ (@tbl-div-questions).

| Question | Divergence | Summary of $r$ | Section |
| --- | --- | --- | --- |
| How much evidence does one observation give, on average? | Kullback–Leibler (KL) | $\mathbb E_P[\ln r]$ | @sec-div-kl |
| The same, weighting rare large ratios more, up to the worst case | Rényi $D_\alpha$, max-divergence $D_\infty$ | log of a power mean of $r$; $\ln\max r$ | @sec-div-renyi |
| How often is the best single guess right? | Total variation (TV) | $\mathbb E_Q[(r - 1)_+]$ | @sec-div-tv |
| The same, when one answer must be $\gamma$ times more convincing | Hockey-stick $E_\gamma$ | $\mathbb E_Q[(r - \gamma)_+]$ | @sec-div-tv |
| Can processing help, and how do I turn KL into TV? | Data processing, Pinsker | — | @sec-div-dpi |

: Each divergence answers one question by summarizing the likelihood ratio $r = p/q$; $(y)_+ = \max(y, 0)$. {#tbl-div-questions}

The running example is a die, fair ($Q$) or loaded so that a six comes up half the time ($P$). A six has $r = 3$; any other roll has $r = 0.6$.

## KL divergence {#sec-div-kl}

**Idea.** The evidence from many independent observations of $P$ adds up and drifts upward. The drift per observation, the average evidence, is the KL divergence: how fast data from $P$ rule out $Q$. Maximum likelihood minimizes it over the model $Q$.

**Result.**
$$D_{\mathrm{KL}}(P\|Q) = \mathbb E_P\left[\ln\frac{p(X)}{q(X)}\right] = \sum_x p(x)\ln\frac{p(x)}{q(x)}.$$
It is never negative, zero only when $P = Q$, not symmetric, and additive: $n$ independent samples carry $n$ times the KL of one. For two normals with the same variance, shifted by $\mu$ standard deviations, $D_{\mathrm{KL}} = \mu^2/2$. For $\mathrm{Lap}(0, b)$ against $\mathrm{Lap}(s, b)$ with $t = s/b$, $D_{\mathrm{KL}} = t + e^{-t} - 1$.

**Example.** For the die, $D_{\mathrm{KL}}(P\|Q) = 0.5\ln1.8 = 0.2939$ but $D_{\mathrm{KL}}(Q\|P) = 0.2426$: a loaded die gives itself away faster than a fair one proves its innocence. Telling two distributions apart takes on the order of $1/\mathrm{KL}$ samples.

**Pitfalls.** The direction matters. Fitting one Gaussian to two separated bumps, forward KL (maximum likelihood) covers both bumps, reverse KL (variational inference) locks onto one, and each fit is bad in the other's direction: the forward fit has forward KL 0.724 but reverse KL 2.097, the reverse fit reverse KL 0.693 but forward KL 15.307. KL is infinite if $Q$ gives zero probability to anything $P$ can produce.

## Rényi divergence and the worst case {#sec-div-renyi}

**Idea.** The Rényi order $\alpha$ is a dial: as it grows, rare outcomes with large ratios weigh more, and at $\alpha \to \infty$ only the worst case counts. It keeps KL's additivity and, unlike KL, bounds how often the evidence is large.

**Result.** For $\alpha > 0$, $\alpha \ne 1$,
$$D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\ln\mathbb E_Q\big[r^\alpha\big] = \frac{1}{\alpha - 1}\ln\sum_x p(x)^\alpha q(x)^{1-\alpha}.$$
It is non-decreasing in $\alpha$, tends to KL as $\alpha \to 1$, adds over independent coordinates, and tends to the **max-divergence** $D_\infty(P\|Q) = \ln\max_x r(x) = \ln\max_A P(A)/Q(A)$, the largest log ratio of the probabilities of any event. For the shifted normal pair, $D_\alpha = \alpha\mu^2/2$ and $D_\infty = \infty$. For the shifted Laplace pair, $D_\infty = t$ and
$$D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\ln\left(\frac{\alpha}{2\alpha - 1}e^{(\alpha - 1)t} + \frac{\alpha - 1}{2\alpha - 1}e^{-\alpha t}\right) \qquad (\alpha \ne \tfrac12, 1).$$ {#eq-div-laplace-renyi}
The tail of the evidence under $Q$ is controlled by every order at once: for any $\alpha > 0$,
$$Q[L \ge c] \le \exp\big(-\alpha c + (\alpha - 1)D_\alpha(P\|Q)\big).$$ {#eq-div-renyi-tail}

**Example.** In @tbl-div-laplace-gauss the Laplace pair is further apart at low orders, the normal pair above order 2.98 (@fig-div-gauss-laplace-renyi). For the normal pair with $\mu = 1$, a draw from $Q$ gives at least 2 nats of evidence for $P$ with probability 0.0062; @eq-div-renyi-tail bounds it by 0.1353 at $\alpha = 1$ and 0.0439 at $\alpha = 2.5$.

| | KL | $D_{0.5}$ | $D_2$ | $D_{10}$ | $D_\infty$ |
| --- | --- | --- | --- | --- | --- |
| Laplace, $t = 1$ | 0.3679 | 0.1891 | 0.6191 | 0.9287 | 1.0000 |
| Normal, same variance | 0.2500 | 0.1250 | 0.5000 | 2.5000 | $\infty$ |

: Rényi divergences of a Laplace pair and a normal pair with the same shift and variance. {#tbl-div-laplace-gauss}

![Left: a Laplace pair and a normal pair, both with shift 1 and variance 2. Right: their Rényi divergences against the order $\alpha$. The normal pair's grows linearly without bound; the Laplace pair's levels off at its max-divergence, 1.](figures/divergences_gauss_laplace_renyi.svg){#fig-div-gauss-laplace-renyi}

**Pitfalls.** Finite $D_\alpha$ at every finite order does not mean a bounded likelihood ratio, as the normal pair shows. Which pair is "more different" depends on the order, so state it. A bounded ratio (Laplace) reaches its worst case on half of $P$'s mass here; an unbounded one (normal) has extreme evidence that is rare, and @eq-div-renyi-tail says how rare.

## Total variation and the hockey-stick divergence {#sec-div-tv}

**Idea.** Total variation is the mass of $P$ that sticks out above $Q$; it says how well the best single guess can do. The hockey-stick divergence $E_\gamma$, for $\gamma \ge 1$, is the mass still sticking out above $\gamma Q$: the part of the difference due to likelihood ratios above $\gamma$.

**Result.**
$$\mathrm{TV}(P, Q) = \max_A|P(A) - Q(A)| = \frac12\sum_x|p(x) - q(x)|, \qquad E_\gamma(P\|Q) = \max_A\big[P(A) - \gamma Q(A)\big] = \sum_x\big(p(x) - \gamma q(x)\big)_+.$$
With equal priors, the best guess from one observation is right with probability $\frac12 + \frac12\mathrm{TV}$, and every statistic with values in $[a, b]$ changes its mean by at most $(b - a)\,\mathrm{TV}$. $E_1 = \mathrm{TV}$, $E_\gamma$ falls as $\gamma$ grows, and it reaches zero exactly at $\gamma = e^{D_\infty}$. For the normal pair, with $\Phi$ the standard normal cumulative distribution function (CDF),
$$\mathrm{TV} = 2\Phi(\mu/2) - 1, \qquad E_\gamma = \Phi\Big(\frac\mu2 - \frac{\ln\gamma}\mu\Big) - \gamma\,\Phi\Big(-\frac\mu2 - \frac{\ln\gamma}\mu\Big).$$ {#eq-div-gauss-hockey}
For the Laplace pair, $\mathrm{TV} = 1 - e^{-t/2}$ and $E_\gamma = 1 - e^{-(t - \ln\gamma)/2}$ for $1 \le \gamma \le e^t$, zero beyond.

**Example.** For the normal pair with $\mu = 1$, $\mathrm{TV} = 0.3829$, so the best guess is right with probability 0.6915; $E_2 = 0.1906$ and $E_4 = 0.0692$, never zero (@fig-div-hockey-stick). For $\mathcal N(0, 1)$ against $\mathcal N(0.1, 1)$, one sample has TV 0.0399, but the best test on 400 samples is right with probability 0.8413.

![The hockey-stick divergence $E_\gamma(P\|Q)$ against $\ln\gamma$ for the loaded die, a Laplace pair and a normal pair with the same variance. The die and Laplace curves reach zero at their max-divergences; the normal curve stays positive.](figures/divergences_hockey_stick.svg){#fig-div-hockey-stick}

**Pitfalls.** TV is not additive and is capped at 1: for $n$ samples compute KL or Rényi and convert (@sec-div-dpi). TV says nothing about rare outcomes with large ratios; $E_\gamma$ for large $\gamma$ does.

## Data processing and Pinsker's inequality {#sec-div-dpi}

**Idea.** Binning, features, thresholds and added noise all average likelihood ratios together, pulling strong evidence toward weak. So no processing can make two distributions easier to tell apart. Pinsker's inequality turns an easy KL into a bound on the TV you want to report.

**Result.** The **data processing inequality (DPI)**: if both inputs go through the same channel $K$, then $D(PK\|QK) \le D(P\|Q)$ for KL, TV, $E_\gamma$, every $D_\alpha$ and $D_\infty$, and in fact every **f-divergence** $\mathbb E_Q[f(r)]$ with $f$ convex and $f(1) = 0$. Equality holds when the processing keeps the likelihood ratio. Pinsker and Bretagnolle–Huber:
$$\mathrm{TV} \le \sqrt{\tfrac12D_{\mathrm{KL}}}, \qquad \mathrm{TV} \le \sqrt{1 - e^{-D_{\mathrm{KL}}}}.$$

**Example.** Rounding draws from $\mathcal N(0, 1)$ or $\mathcal N(1, 1)$ into bins of width 0.5, 1 and 2 lowers KL from 0.5 to 0.4898, 0.4618 and 0.3781; a single bit, the midpoint threshold, keeps 0.3090 and all of the TV. For the die, parity drops KL from 0.2939 to 0.0823, while "is it a six?" keeps all of it. @tbl-div-pinsker shows the two bounds at work.

| $n$ | KL of $n$ samples | Exact TV | Pinsker | Bretagnolle–Huber |
| --- | --- | --- | --- | --- |
| 100 | 0.50 | 0.3829 | 0.5000 | 0.6273 |
| 400 | 2.00 | 0.6827 | 1.0000 | 0.9299 |

: Exact total variation between $n$ samples of $\mathcal N(0, 1)$ and of $\mathcal N(0.1, 1)$, and the two bounds computed from KL. {#tbl-div-pinsker}

**Pitfalls.** Pinsker is vacuous once KL reaches 2; above 1.59 use Bretagnolle–Huber. A tiny TV is compatible with infinite KL, so you cannot bound KL by TV. The DPI bounds the *best* test: processing can help a bad classifier, never beat the best test on raw data.

## Code {#sec-div-code}

The closed forms for the two shifted pairs need only the standard library:

```python
from math import erfc, exp, log, sqrt


def normal_cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def gauss_kl(mu: float) -> float:
    """KL between N(0, s^2) and N(mu * s, s^2): half the squared shift in standard deviations."""
    return mu**2 / 2


def gauss_tv(mu: float) -> float:
    return 2 * normal_cdf(mu / 2) - 1


def gauss_hockey(mu: float, gamma: float) -> float:
    """E_gamma between N(0, 1) and N(mu, 1)."""
    c = log(gamma)
    return normal_cdf(mu / 2 - c / mu) - gamma * normal_cdf(-mu / 2 - c / mu)


def laplace_kl(t: float) -> float:
    """KL between Lap(0, b) and Lap(t * b, b)."""
    return t + exp(-t) - 1


def laplace_renyi(alpha: float, t: float) -> float:
    """Renyi divergence of order alpha (not 1/2 or 1) between Lap(0, b) and Lap(t * b, b)."""
    inner = alpha / (2 * alpha - 1) * exp((alpha - 1) * t) + (alpha - 1) / (2 * alpha - 1) * exp(-alpha * t)
    return log(inner) / (alpha - 1)


def laplace_tv(t: float) -> float:
    return 1 - exp(-t / 2)


print(f"{gauss_kl(1.0):.4f} {gauss_tv(1.0):.4f} {gauss_hockey(1.0, 2.0):.4f}")  # 0.5000 0.3829 0.1906
print(f"{laplace_kl(1.0):.4f} {laplace_renyi(2.0, 1.0):.4f} {laplace_tv(1.0):.4f}")  # 0.3679 0.6191 0.3935
print(f"{laplace_renyi(10.0, 1.0):.4f}")  # 0.9287
```

The full edition's script, [divergences.py](../../code/part01-prerequisites/divergences.py), checks every closed form against simulation.

## Cheat sheet

| Divergence | Definition | Normal pair, shift $\mu$ | Laplace pair, shift $t$ | Use it for |
| --- | --- | --- | --- | --- |
| KL | $\mathbb E_P[\ln r]$ | $\mu^2/2$ | $t + e^{-t} - 1$ | Average evidence; sample sizes ($\approx 1/\mathrm{KL}$) |
| Rényi $D_\alpha$ | $\frac{1}{\alpha-1}\ln\mathbb E_Q[r^\alpha]$ | $\alpha\mu^2/2$ | @eq-div-laplace-renyi | From typical to worst-case evidence; tail of $L$ |
| Max-divergence $D_\infty$ | $\ln\max r$ | $\infty$ | $t$ | Worst-case ratio over all events |
| TV | $\frac12\sum\lvert p - q\rvert$ | $2\Phi(\mu/2) - 1$ | $1 - e^{-t/2}$ | Best single guess: $\frac12 + \frac12\mathrm{TV}$ |
| Hockey-stick $E_\gamma$ | $\sum(p - \gamma q)_+$ | @eq-div-gauss-hockey | $\big(1 - e^{-(t - \ln\gamma)/2}\big)_+$ | Mass of large ratios; $E_1 = $ TV |

: The divergences of this chapter, with closed forms for $\mathcal N(0, \sigma^2)$ against $\mathcal N(s, \sigma^2)$ ($\mu = s/\sigma$) and $\mathrm{Lap}(0, b)$ against $\mathrm{Lap}(s, b)$ ($t = s/b$). {#tbl-div-cheat}

**Rules of thumb**

- Compute in KL or Rényi, where independent samples add; convert to TV at the end.
- A bounded likelihood ratio gives a finite max-divergence; an unbounded one gives finite $D_\alpha$ at every finite order but $D_\infty = \infty$.
- Processing never increases a divergence.
- Pinsker is good for small KL and useless beyond 2; switch to Bretagnolle–Huber.

## Further reading

- van Erven & Harremoës, "Rényi Divergence and Kullback–Leibler Divergence" (IEEE Transactions on Information Theory, 2014): the Rényi family and its properties.
- Cover & Thomas, *Elements of Information Theory* (2nd ed., 2006), ch. 2 and §11.6: KL and Pinsker's inequality.
- Polyanskiy & Wu, *Information Theory: From Coding to Learning* (2025), ch. 7: f-divergences and data processing at reference depth.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/03-divergences.md).
