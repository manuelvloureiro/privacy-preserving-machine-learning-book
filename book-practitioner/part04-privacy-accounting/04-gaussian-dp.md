# f-DP and Gaussian differential privacy

## Why this matters

Every privacy guarantee bounds how well an attacker can tell from a mechanism's output whether it ran on a dataset $D$ or on a neighbour $D'$ that differs in one record. Differential privacy (DP) answers with two numbers, and [Rényi differential privacy (RDP)](01-renyi-dp.md) with a curve of divergences. [Zero-concentrated differential privacy (zCDP)](03-concentrated-dp.md) answers with one number. All three summarize a more complete answer, the attacker's best error rates, and this chapter works with that answer directly.

It pays off for two reasons. First, Gaussian noise gives a family of answers called **Gaussian differential privacy (GDP)**. Its one parameter, written $\mu$, reads directly as the best success rate of a membership attack. Gaussian steps compose within the family, so a Gaussian pipeline has an exact guarantee in one number. Second, a central limit theorem (CLT) gives a quick estimate of $\mu$ for training with differentially private stochastic gradient descent (SGD), called DP-SGD. That estimate can understate the true privacy loss, so it must never be the number you report.

## Trade-off curves and f-DP {#sec-gdp-fdp}

The attacker's decision is a hypothesis test ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)), and a test can make two kinds of mistake. Write $a$ for its **type I error**, how often it claims $D'$ when the data was $D$. Write $b$ for its **type II error**, how often it makes the reverse mistake. In a membership attack, where $D'$ contains the target's record, $a$ is the false positive rate and $1 - b$ the true positive rate. The **trade-off function** of a mechanism gives, at each $a$, the smallest $b$ that any test can reach.

A guarantee is a floor under that curve. Write $T\big(M(D), M(D')\big)$ for the trade-off function of a mechanism $M$ on two datasets. Dong, Roth and Su (2022) call the mechanism **$f$-differentially private ($f$-DP)** if, for all neighbours,
$$T\big(M(D), M(D')\big) \ge f.$$

The familiar guarantee with a privacy level $\varepsilon$ and a slack $\delta$ is the special case whose floor is two straight lines,
$$f_{\varepsilon,\delta}(a) = \max\big(0,\; 1 - \delta - e^\varepsilon a,\; e^{-\varepsilon}(1 - \delta - a)\big).$$ {#eq-gdp-feps}
Two straight lines make a crude floor. Randomized response and the Laplace mechanism both carry the label "$\varepsilon = 1$", but only randomized response sits on $f_{1,0}$, while the Laplace curve lies above it in the middle. So when you compare mechanisms, or audit one, compare curves.

## μ-GDP, read as an attack {#sec-gdp-gdp}

The Gaussian mechanism's curve has a simple form. Write $\sigma$ for the noise multiplier, the standard deviation of the noise in units of the query's $\ell_2$ sensitivity. Every Gaussian mechanism then reduces to telling a standard normal draw from one shifted by a distance $\mu = 1/\sigma$. Write $\Phi$ for the standard normal distribution function, so that $\Phi(t)$ is the chance that a standard normal draw is at most $t$. The best test's type II error at type I error $a$ is
$$G_\mu(a) = \Phi\big(\Phi^{-1}(1 - a) - \mu\big).$$ {#eq-gdp-gmu}
A mechanism is **$\mu$-Gaussian differentially private ($\mu$-GDP)** if $G_\mu$ is a floor for it. For the Gaussian mechanism this floor is the exact curve, so nothing is lost.

The parameter $\mu$ reads directly as an attack. Take a count released with $\sigma = 1$, so 1-GDP. At a 1% false positive rate, and whatever the attacker knows about the rest of the data, it detects the target's record at most 9.24% of the time. @tbl-gdp-mu gives this best true positive rate for three values of $\mu$, next to the $\varepsilon$ that each converts to.

| $\mu$ | Best true positive rate at 0.1% false positives | At 1% | At 5% | $\varepsilon$ at $\delta = 10^{-5}$ |
| --- | --- | --- | --- | --- |
| 0.5 | 0.0048 | 0.0339 | 0.1261 | 1.9931 |
| 1 | 0.0183 | 0.0924 | 0.2595 | 4.3772 |
| 2 | 0.1378 | 0.3721 | 0.6388 | 9.9973 |

: Reading $\mu$-GDP as an attack and as $(\varepsilon, \delta)$. The attack column is $1 - G_\mu(a)$; the code below computes all three. {#tbl-gdp-mu}

This table is the most useful way to explain a guarantee to someone who must decide whether it is enough. It is also the ceiling that a [membership inference attack](../part02-attacks/02-membership-inference.md) can reach, and that a [privacy audit](../part05-dp-ml/10-auditing.md) tests against, usually at low false positive rates. As a rough guide, Dong et al. suggest that $\mu \le 0.5$ is a reasonable guarantee and that $\mu \ge 6$ promises almost nothing.

## Converting μ to (ε, δ) {#sec-gdp-duality}

To report $(\varepsilon, \delta)$, you turn the curve into pairs. Its **privacy profile** $\delta(\varepsilon)$ is the smallest slack that goes with each privacy level. For a curve that is unchanged when you swap $D$ and $D'$, the whole profile loses nothing, because together its pairs rebuild the curve. For $\mu$-GDP the profile has a closed form (Dong et al., Corollary 2.13):
$$\delta(\varepsilon) = \Phi\Big(-\frac{\varepsilon}{\mu} + \frac{\mu}{2}\Big) - e^\varepsilon\,\Phi\Big(-\frac{\varepsilon}{\mu} - \frac{\mu}{2}\Big).$$ {#eq-gdp-delta}
This is also the exact $(\varepsilon, \delta)$ curve of the Gaussian mechanism, the analytic calibration of [The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md) (Balle & Wang 2018).

You use it in both directions, with any root finder, such as `scipy.optimize.brentq` on `scipy.stats.norm.cdf`, or the bisection in the code below. To report, fix $\delta$ and solve @eq-gdp-delta for $\varepsilon$ by bisection, which works because the profile falls as $\varepsilon$ grows. For example, 1-GDP is $(4.3772, 10^{-5})$-DP. To calibrate, fix the target pair, bisect for the largest $\mu$ that meets it, and set the noise multiplier to $1/\mu$. For example, a single $(1, 10^{-5})$-DP release needs $\sigma = 3.7306$.

One pair, on its own, loses a lot. At a type I error of 0.1, for example, the pair for 1-GDP allows a type II error as low as 0.0113. The curve $G_1$ itself requires 0.6109.

Do not calibrate with the classical formula instead, which sets the noise for a query of sensitivity $\Delta$ to
$$\sigma = \frac{\Delta\sqrt{2\ln(1.25/\delta)}}{\varepsilon}.$$
It is valid only for $\varepsilon < 1$, and even there it adds more noise than the exact curve requires. For a numerical accountant, `dp_accounting.pld.privacy_loss_distribution.from_gaussian_mechanism` builds the same exact curve as a privacy loss distribution.

## Composition: μ² adds {#sec-gdp-composition}

Dong et al. show that composing $f_i$-DP steps gives a curve that depends only on the $f_i$, even when each step is chosen after seeing the earlier outputs. For Gaussian curves it stays Gaussian (Dong et al., Corollary 3.3):
$$G_{\mu_1} \otimes \dots \otimes G_{\mu_k} = G_\mu, \qquad \mu = \sqrt{\mu_1^2 + \dots + \mu_k^2}.$$ {#eq-gdp-compose}
Here $\otimes$ stands for composition, and the quantity that adds up is $\mu^2$. So full-batch noisy gradient descent is $\sqrt k/\sigma$-GDP after a number $k$ of steps. Since that is the exact curve, no accountant can improve on it. The same holds for every pipeline of Gaussian noise without subsampling, such as analytics queries, the noisy vote counts of private aggregation of teacher ensembles (PATE) and matrix mechanisms.

The zCDP ledger does the same arithmetic, where $\rho$, the zCDP parameter, equals $\mu^2/2$. But its conversion loses some of the curve, while @eq-gdp-delta loses none.

Composing curves rather than pairs matters in practice. Four 1-GDP releases compose to 2-GDP, which is $(9.9973, 10^{-5})$-DP. Suppose you instead turn each release into a pair with a quarter of the $\delta$ budget, $(4.6902, 2.5\times10^{-6})$, and add the pairs by basic composition. You get $\varepsilon = 18.7607$, almost twice as much. So for a pipeline of Gaussian releases without subsampling, keep a ledger of $\mu^2$ and convert once. Curves that are not Gaussian leave their family when composed, so for them you need a [numerical accountant](06-numerical-accounting.md).

## The CLT estimate for DP-SGD {#sec-gdp-sgd}

Most mechanisms are not Gaussian. But a central limit theorem for $f$-DP says that many steps that each leak little compose to nearly a Gaussian curve (Dong et al. 2022). DP-SGD is such a run.

Each step of DP-SGD samples a batch by Poisson sampling, which includes each record independently with a **sampling rate** $q$. It then clips each gradient and adds Gaussian noise ([Amplification by subsampling](05-subsampling.md)). With add/remove neighbours, Bu et al. (2020) show that $k$ such steps are approximately $\mu$-GDP with
$$\mu = q\sqrt{k\big(e^{1/\sigma^2} - 1\big)}.$$ {#eq-gdp-sgd-mu}
It shows that the parameter grows in proportion to the sampling rate and to the square root of the number of steps, and steeply as the noise falls.

The formula is an approximation, not a bound, and it errs on the unsafe side. A CLT controls the bulk of a distribution much better than its tails, where small values of $\delta$ are decided. After the 10,000 steps of this part's running example, the CLT gives $\varepsilon = 6.0071$ at $\delta = 10^{-5}$. A numerical accountant proves that the true $\varepsilon$ is at least 6.1377, so the CLT number would be false (Gopi, Lee & Wutschitz 2021). Opacus offers this accountant as `accountant="gdp"`, implemented by `opacus.accountants.GaussianAccountant`, and it warns that the GDP accountant is experimental and can underestimate the privacy spent. For anything you report, use its default `"prv"` accountant or `"rdp"` ([Choosing an accountant](09-choosing-an-accountant.md)). The formula also assumes Poisson sampling with add/remove neighbours ([Amplification by subsampling](05-subsampling.md)).

## RDP forgets part of the curve {#sec-gdp-rdp}

An RDP curve is a summary of the trade-off curve, and it forgets part of it even for the Gaussian. For 1-GDP the exact $\delta(1)$ is 0.1269. By any [conversion](02-rdp-conversions.md), its RDP curve certifies no better than 0.2149. So for Gaussian noise without subsampling, account in GDP.

## Code {#sec-gdp-code}

A GDP accountant needs only the normal distribution, which Python's `statistics.NormalDist` provides. The snippet computes the attack reading of $\mu$, the conversions in both directions and the CLT estimate.

```python
from math import exp, sqrt
from statistics import NormalDist

N = NormalDist()


def gdp_power(mu: float, fpr: float) -> float:
    """Best true positive rate of any attack on mu-GDP at a given false positive rate."""
    return 1 - N.cdf(N.inv_cdf(1 - fpr) - mu)


def gdp_delta(eps: float, mu: float) -> float:
    """The exact delta(eps) of mu-GDP."""
    return N.cdf(-eps / mu + mu / 2) - exp(eps) * N.cdf(-eps / mu - mu / 2)


def gdp_eps(mu: float, delta: float) -> float:
    """The eps at which mu-GDP reaches delta, by bisection."""
    lo, hi = 0.0, 500.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gdp_delta(mid, mu) > delta else (lo, mid)
    return hi


def gdp_mu(eps: float, delta: float) -> float:
    """The largest mu that meets (eps, delta), by bisection."""
    lo, hi = 1e-6, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gdp_delta(eps, mid) < delta else (lo, mid)
    return lo


def clt_mu_dpsgd(q: float, sigma: float, steps: int) -> float:
    """CLT estimate of mu for Poisson-subsampled DP-SGD; not a bound."""
    return q * sqrt(steps * (exp(1 / sigma**2) - 1))


print(" ".join(f"{gdp_power(m, 0.001):.4f}" for m in (0.5, 1.0, 2.0)))  # 0.0048 0.0183 0.1378
print(f"{gdp_eps(1.0, 1e-5):.4f}")  # 4.3772
mu = gdp_mu(1.0, 1e-5)  # calibrate one release to (1, 1e-5)
print(f"{mu:.4f} {1 / mu:.4f}")  # 0.2681 3.7306
mu = clt_mu_dpsgd(q=0.01, sigma=1.0, steps=10_000)
print(f"{mu:.4f} {gdp_eps(mu, 1e-5):.4f}")  # 1.3108 6.0071
```

## Cheat sheet

| Task | Formula | Notes |
| --- | --- | --- |
| Read a guarantee as an attack | True positive rate $\le 1 - f(a)$ at false positive rate $a$ | The curve is the guarantee |
| Gaussian mechanism | $\mu = 1/\sigma$; $G_\mu(a) = \Phi(\Phi^{-1}(1 - a) - \mu)$ | Exact |
| Convert to $(\varepsilon, \delta)$ | $\delta(\varepsilon) = \Phi(-\frac\varepsilon\mu + \frac\mu2) - e^\varepsilon\Phi(-\frac\varepsilon\mu - \frac\mu2)$ | Bisect for $\varepsilon$ |
| Calibrate | Bisect for the largest $\mu$, then $\sigma = 1/\mu$ | $(1, 10^{-5})$ needs $\sigma = 3.7306$ |
| Compose Gaussian steps | $\mu = \sqrt{\sum_i\mu_i^2}$ | Exact |
| Estimate DP-SGD | $\mu \approx q\sqrt{k(e^{1/\sigma^2} - 1)}$ | Not a bound; under-reports $\varepsilon$ |

: The tasks of this chapter; @tbl-gdp-mu converts $\mu$ to attack rates and to $\varepsilon$. {#tbl-gdp-cheat}

**Rules of thumb**

- The trade-off curve is the guarantee, and $(\varepsilon, \delta)$, RDP and $\rho$ are summaries of it. Summarize as late as possible.
- For Gaussian noise without subsampling, keep a ledger of $\mu^2$ and convert once, exactly.
- Explain a guarantee with its attack reading, and report $\mu$ next to any $(\varepsilon, \delta)$ pair.
- Calibrate from the exact curve, never from the classical $\sqrt{2\ln(1.25/\delta)}$ formula.
- Use the CLT, and Opacus's `gdp` accountant, only for quick estimates, never for the reported guarantee.

## Further reading

- Dong, Roth & Su, "Gaussian Differential Privacy" (Journal of the Royal Statistical Society Series B, 2022): the source of $f$-DP and GDP.
- Gopi, Lee & Wutschitz, "Numerical Composition of Differential Privacy" (NeurIPS 2021): an accountant with error bounds, and examples where the CLT under-reports $\varepsilon$.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/04-gaussian-dp.md).
