# f-DP and Gaussian differential privacy

## What it is {#sec-gdp-gdp}

Every differential privacy (DP) guarantee limits how well an attacker can tell whether the data held a target's record. The attacker runs a [hypothesis test](../part01-prerequisites/04-hypothesis-testing.md), so the fullest guarantee is a **trade-off curve**, the lowest false negative rate any test can reach at each false positive rate. A mechanism is **$f$-DP** when its curve never falls below a given curve $f$ (Dong, Roth and Su 2022). A privacy level $\varepsilon$ with its slack $\delta$ only summarizes that curve.

For Gaussian noise the curve has one parameter. Write $\sigma$ for the noise multiplier, the noise's standard deviation in units of the query's $\ell_2$ sensitivity. In those units the attacker must tell a standard normal draw from one shifted by a distance $\mu$ equal to $1/\sigma$. So the mechanism satisfies Gaussian differential privacy (GDP) with parameter $\mu$, written **$\mu$-GDP**. Write $\Phi$ for the standard normal distribution function and $\Phi^{-1}$ for its inverse. No membership attack at a false positive rate $\alpha$ then reaches a true positive rate above
$$1 - \Phi\big(\Phi^{-1}(1 - \alpha) - \mu\big).$$

## When you meet it

You meet GDP when deciding whether a guarantee is enough, and as a trap in Opacus (@sec-gdp-wrong). @tbl-gdp-mu reads $\mu$ as the best true positive rate (TPR) of an attack at a given false positive rate (FPR), whatever else the attacker knows.

| $\mu$ | Best TPR at 0.1% FPR | At 1% | At 5% | $\varepsilon$ at $\delta = 10^{-5}$ |
| --- | --- | --- | --- | --- |
| 0.5 | 0.0048 | 0.0339 | 0.1261 | 1.9931 |
| 1 | 0.0183 | 0.0924 | 0.2595 | 4.3772 |
| 2 | 0.1378 | 0.3721 | 0.6388 | 9.9973 |

: Reading $\mu$-GDP as an attack and as $(\varepsilon, \delta)$. {#tbl-gdp-mu}

For example, at a false positive rate of 1%, no attacker on a 1-GDP release detects the record in more than 9.24% of the trials where it is present. That is the ceiling for any [membership inference attack](../part02-attacks/02-membership-inference.md), and what a [privacy audit](../part05-dp-ml/10-auditing.md) tests. Dong et al. suggest that $\mu \le 0.5$ is a reasonable guarantee and that $\mu \ge 6$ promises almost nothing, so report $\mu$ next to a Gaussian pipeline's $(\varepsilon, \delta)$.

## How to use it

Gaussian releases compose exactly within GDP, even when each one depends on the earlier ones. Write $\mu_i$ for the parameter of the $i$-th release. The run is then $\mu$-GDP with
$$\mu = \sqrt{\mu_1^2 + \mu_2^2 + \dots + \mu_k^2},$$
so the quantity that adds up is $\mu^2$. No accountant does better for releases on the whole dataset, such as analytics queries and full-batch noisy gradient descent. So keep a ledger of $\mu^2$ and convert once, at the end. For example, four 1-GDP releases compose to 2-GDP, which is $(9.9973, 10^{-5})$-DP. Converting each release first, with a quarter of the slack, and adding the four values of $\varepsilon$ gives 18.7607, almost twice as much.

The snippet computes the attack reading and converts both ways. The `dp_accounting` functions take a noise multiplier, so for $\mu$-GDP you pass $1/\mu$.

```python
from statistics import NormalDist

import dp_accounting

N = NormalDist()


def attack_power(mu: float, fpr: float) -> float:
    """Best true positive rate of any membership attack on mu-GDP at a false positive rate."""
    return 1 - N.cdf(N.inv_cdf(1 - fpr) - mu)


print(" ".join(f"{attack_power(mu, 0.001):.4f}" for mu in (0.5, 1.0, 2.0)))  # 0.0048 0.0183 0.1378
print(f"{dp_accounting.get_epsilon_gaussian(1.0, 1e-5):.4f}")  # 4.3772
print(f"{dp_accounting.get_sigma_gaussian(1.0, 1e-5):.4f}")  # 3.7306
```

The second print converts 1-GDP to $(4.3772, 10^{-5})$-DP. The third finds the noise multiplier of a single $(1, 10^{-5})$-DP release. Both use the exact curve, which asks for less noise than the classical formula of [the Gaussian mechanism](../part03-dp-foundations/03-laplace-gaussian.md).

## What goes wrong {#sec-gdp-wrong}

Opacus offers `accountant="gdp"` for the private form of stochastic gradient descent (SGD), DP-SGD. It estimates $\mu$ with a central limit theorem (CLT), by which many steps that each leak little compose to nearly a Gaussian curve. For a number $T$ of steps, each sampling a record with probability $q$, Bu et al. (2020) estimate
$$\mu \approx q\sqrt{T\big(e^{1/\sigma^2} - 1\big)}.$$
It is an approximation, not a bound, and it errs on the unsafe side, because a small $\delta$ is decided by rare, large losses, whose tails a CLT fits poorly. On the part's running example of 10,000 steps, the CLT gives 6.0071, but a [numerical accountant](06-numerical-accounting.md) proves that the true $\varepsilon$ is at least 6.1377. So report with Opacus's default numerical accountant, `"prv"`.

A mix of Laplace and Gaussian releases needs a numerical accountant, because a curve that is not Gaussian takes the composition out of GDP. Gaussian pipelines without subsampling, though, are best accounted in GDP, since even the best [conversion from a Rényi DP curve](02-rdp-conversions.md) loses ground. For 1-GDP at $\varepsilon = 1$, it certifies a $\delta$ of 0.2149 against the exact 0.1269.

## References

- Dong, Roth & Su, "Gaussian Differential Privacy" (Journal of the Royal Statistical Society Series B, 2022).

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/04-gaussian-dp.md).
