# f-DP and Gaussian differential privacy

## What it is {#sec-gdp-gdp}

Every differential privacy (DP) guarantee limits an attacker's ability to determine whether a target's record was in the data. The attacker performs a [hypothesis test](../part01-prerequisites/04-hypothesis-testing.md), so the most complete guarantee is a **trade-off curve**: the minimum false negative rate at each false positive rate. A mechanism is **$f$-DP** if its curve is nowhere below a specified curve $f$ (Dong, Roth and Su 2022). A privacy level $\varepsilon$ and slack $\delta$ summarize this curve.

For Gaussian noise, the curve has one parameter. Let $\sigma$ be the noise multiplier, the noise standard deviation in units of the query's $\ell_2$ sensitivity. The attacker must distinguish a standard normal draw from one shifted by $\mu = 1/\sigma$. The mechanism therefore satisfies Gaussian differential privacy (GDP) with parameter $\mu$, or **$\mu$-GDP**. Let $\Phi$ be the standard normal distribution function. At false positive rate $\alpha$, no membership attack achieves a true positive rate above
$$1 - \Phi\big(\Phi^{-1}(1 - \alpha) - \mu\big).$$

## When you meet it

Use GDP to assess whether a guarantee is sufficient; Opacus also exposes it as a potentially unsafe option (@sec-gdp-wrong). @tbl-gdp-mu gives the maximum true positive rate (TPR) at each false positive rate (FPR), regardless of the attacker's other information. The same representation can account for stochastic gradient descent (SGD) runs without subsampling.

| $\mu$ | Best TPR at 0.1% FPR | At 1% | At 5% | $\varepsilon$ at $\delta = 10^{-5}$ |
| --- | --- | --- | --- | --- |
| 0.5 | 0.0048 | 0.0339 | 0.1261 | 1.9931 |
| 1 | 0.0183 | 0.0924 | 0.2595 | 4.3772 |
| 2 | 0.1378 | 0.3721 | 0.6388 | 9.9973 |

: Reading $\mu$-GDP as an attack and as $(\varepsilon, \delta)$. {#tbl-gdp-mu}

At 1% FPR, no attacker against a 1-GDP release can identify the record in more than 9.24% of trials where it is present. This bounds any [membership inference attack](../part02-attacks/02-membership-inference.md) and is the limit tested by a [privacy audit](../part05-dp-ml/10-auditing.md). Dong et al. suggest $\mu \le 0.5$ as a reasonable guarantee and note that $\mu \ge 6$ offers little protection. Report $\mu$ alongside $(\varepsilon,\delta)$ for Gaussian pipelines.

## How to use it

Gaussian releases compose exactly in GDP, including adaptive releases. Let $\mu_i$ be the parameter of release $i$. The full run is $\mu$-GDP, where
$$\mu = \sqrt{\mu_1^2 + \mu_2^2 + \dots + \mu_k^2},$$
Thus, $\mu^2$ composes additively. For whole-dataset releases, such as analytics queries and full-batch noisy gradient descent, no accountant gives a tighter bound. Track $\mu^2$ and convert once after composition. Four 1-GDP releases compose to 2-GDP, or $(9.9973,10^{-5})$-DP. Converting each release separately using one quarter of the slack gives a summed $\varepsilon$ of 18.7607, almost twice as large.

The snippet computes the attack ceiling and converts between GDP and $(\varepsilon,\delta)$. The `dp_accounting` functions accept a noise multiplier, so pass $1/\mu$ for $\mu$-GDP.

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

The final line returns the noise multiplier for one $(1,10^{-5})$-DP release. It uses the exact curve, which requires less noise than the classical formula in [the Gaussian mechanism](../part03-dp-foundations/03-laplace-gaussian.md).

## What goes wrong {#sec-gdp-wrong}

Opacus supports `accountant="gdp"` for DP-SGD. It estimates $\mu$ using the central limit theorem (CLT) of Bu et al. (2020), which approximates many small privacy losses by a Gaussian curve. This is an approximation, not a bound, and its error can be unsafe. Small $\delta$ depends on rare, large losses that the CLT models poorly. For this part's 10,000-step example, the CLT gives 6.0071, while a [numerical accountant](06-numerical-accounting.md) establishes that true $\varepsilon$ is at least 6.1377. Use Opacus's default numerical accountant, `"prv"`, for reporting.

Mixed Laplace and Gaussian releases require a numerical accountant because their composition is not Gaussian. For Gaussian pipelines without subsampling, GDP is the tightest accountant. Even the best [conversion from a Rényi DP curve](02-rdp-conversions.md) is looser: for 1-GDP at $\varepsilon = 1$, it gives $\delta = 0.2149$ instead of the exact 0.1269.

## References

- Dong, Roth & Su, "Gaussian Differential Privacy" (Journal of the Royal Statistical Society Series B, 2022).

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/04-gaussian-dp.md).
