# f-DP and Gaussian differential privacy

## Why this matters

Every privacy guarantee answers one question: how well can an attacker who sees the output of a mechanism $M$ tell whether it ran on a dataset $D$ or on a neighbour $D'$ that differs in one record? Differential privacy (DP) answers with two numbers $(\varepsilon, \delta)$, Rényi differential privacy (RDP) with a curve over orders $\alpha$ ([Rényi DP](01-renyi-dp.md)), and zero-concentrated differential privacy (zCDP) with one number $\rho$ ([Concentrated differential privacy](03-concentrated-dp.md)). The complete answer is the **trade-off function** $T(P, Q)$ of the two output distributions $P = M(D)$ and $Q = M(D')$: for each rate $a$ at which a test may wrongly claim $D'$ (its type I error), the smallest rate $b$ at which it wrongly claims $D$ (its type II error) ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). This chapter writes $a$ and $b$ for the two errors, because $\alpha$ is the Rényi order in this part.

The one idea: **the whole trade-off curve is the guarantee, and every other notion is a lossy summary of it.** Two facts make the curve practical. Gaussian noise gives a one-parameter family of curves $G_\mu$ that is closed under composition, so a pipeline of Gaussian steps has an exact guarantee in one number. And a central limit theorem (CLT) says that many small steps compose to a curve close to some $G_\mu$, which gives a quick estimate for differentially private stochastic gradient descent (SGD), DP-SGD for short. @tbl-gdp-overview maps the decisions to the sections.

| Decision | Rule | Section |
| --- | --- | --- |
| Read a guarantee as a bound on attacks | $f$-DP; $(\varepsilon, \delta)$ is two straight lines | @sec-gdp-fdp |
| State the guarantee of Gaussian noise, and report $(\varepsilon, \delta)$ | $\mu = 1/\sigma$; the exact profile $\delta(\varepsilon)$ | @sec-gdp-gdp |
| Compose Gaussian steps | $\mu = \sqrt{\mu_1^2 + \dots + \mu_k^2}$, exactly | @sec-gdp-composition |
| Estimate $\varepsilon$ for DP-SGD quickly | CLT $\mu = q\sqrt{k(e^{1/\sigma^2} - 1)}$; never report it | @sec-gdp-sgd |
| Understand why RDP accounting leaves slack | RDP keeps only part of the curve | @sec-gdp-rdp |

: The decisions this chapter helps with, and where it makes each one. {#tbl-gdp-overview}

## f-DP {#sec-gdp-fdp}

**Idea.** A privacy guarantee should say what an attacker cannot do. In testing terms the strongest such statement is a lower bound on the attacker's type II error at every type I error: however it tunes its test, it must miss at least this often. That bound is a curve, so the definition takes a curve $f$ as its parameter. The usual $(\varepsilon, \delta)$-DP is one particular curve made of two straight lines.

**Result.** A mechanism is **$f$-differentially private ($f$-DP)** if $T(M(D), M(D')) \ge f$ for all neighbours $D \simeq D'$ (Dong, Roth & Su 2022). $(\varepsilon, \delta)$-DP is exactly $f_{\varepsilon,\delta}$-DP for
$$f_{\varepsilon,\delta}(a) = \max\big(0,\; 1 - \delta - e^\varepsilon a,\; e^{-\varepsilon}(1 - \delta - a)\big).$$ {#eq-gdp-feps}
The first line says a test that fires with probability $a$ under $D$ can fire with probability at most $e^\varepsilon a + \delta$ under $D'$. The curve starts at $1 - \delta$, not at 1: a test may catch $D'$ with probability $\delta$ at no type I cost. Post-processing can only raise the curve, so $f$-DP is closed under it.

**Example.** Three mechanisms carry the label "$\varepsilon = 1$". @tbl-gdp-curves gives their smallest type II errors. Randomized response sits exactly on $f_{1,0}$: it is the worst $(1, 0)$-DP mechanism. The Laplace mechanism matches it at small $a$ but is more private in the middle, 0.3679 against 0.3204 at $a = 0.25$. The label cannot tell the two apart; their curves can.

| Type I error $a$ | $f_{1,0}$ (randomized response) | Laplace, $\varepsilon = 1$ | Gaussian, $\sigma = 1$ ($G_1$) |
| --- | --- | --- | --- |
| 0.01 | 0.9728 | 0.9728 | 0.9076 |
| 0.1 | 0.7282 | 0.7282 | 0.6109 |
| 0.25 | 0.3204 | 0.3679 | 0.3724 |

: Smallest type II error at three type I errors for three mechanisms. {#tbl-gdp-curves}

![Trade-off curves of $(1, 0)$-DP (randomized response), $(1, 0.1)$-DP, the Laplace mechanism with $\varepsilon = 1$ and the Gaussian mechanism with $\sigma = 1$. A mechanism is $f$-DP when its curve lies on or above $f$; the dotted diagonal is perfect privacy.](figures/gaussian_dp_eps_delta.svg){#fig-gdp-eps-delta}

**Pitfalls.** An $(\varepsilon, \delta)$ label is a floor under the curve, not the curve, so two mechanisms with the same label can differ a lot in how much they leak in practice. The bound holds for the worst neighbouring pair; most records are better protected. And $\delta$ is a free shot at detection: $f_{1, 0.1}$ allows $b = 0.6282$ at $a = 0.1$, exactly 0.1 below $f_{1,0}$ (@fig-gdp-eps-delta).

## Gaussian differential privacy {#sec-gdp-gdp}

**Idea.** Divide the output of a Gaussian mechanism by the noise standard deviation and shift it, maps that change no test's errors, and you are left with the problem of telling $\mathcal N(0, 1)$ from $\mathcal N(\mu, 1)$. For a vector output only the direction of the shift carries evidence, so the problem is always one-dimensional. Its trade-off curve $G_\mu$ has one parameter, $\mu$: the distance between the two output means in units of the noise. Using that family as the yardstick gives a one-number guarantee that is exact for Gaussian noise.

**Result.** A mechanism is **$\mu$-Gaussian differential privacy (GDP)** if it is $G_\mu$-DP, with
$$G_\mu(a) = \Phi\big(\Phi^{-1}(1 - a) - \mu\big),$$ {#eq-gdp-gmu}
where $\Phi$ is the standard normal distribution function. The Gaussian mechanism with noise multiplier $\sigma$ (the noise standard deviation in units of the $\ell_2$ sensitivity) is $1/\sigma$-GDP, and this is tight. Because a convex curve is the envelope of the lines that touch it, $f$-DP for a symmetric curve is the same as being $(\varepsilon, \delta(\varepsilon))$-DP for every $\varepsilon \ge 0$ at once. For $G_\mu$ that **privacy profile** has a closed form,
$$\delta(\varepsilon) = \Phi\Big(-\frac{\varepsilon}{\mu} + \frac{\mu}{2}\Big) - e^\varepsilon\,\Phi\Big(-\frac{\varepsilon}{\mu} - \frac{\mu}{2}\Big),$$ {#eq-gdp-delta}
the same exact formula as the analytic Gaussian calibration of [The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md). In words, the first term is the power of the test whose curve has slope $-e^\varepsilon$, and the second is $e^\varepsilon$ times its type I error. To report $\varepsilon$ at a target $\delta$, solve @eq-gdp-delta by bisection, since $\delta(\varepsilon)$ falls as $\varepsilon$ grows.

**Example.** A count released with $\sigma = 1$ is 1-GDP. @tbl-gdp-g1 reads it as attack errors: at a 1% false-positive rate, whatever the attacker knows about the rest of the data, it detects the presence of a record at most 9.24% of the time. At $\varepsilon = 1$ the profile gives $\delta(1) = 0.1269$, and @tbl-gdp-eps gives $\varepsilon$ at $\delta = 10^{-5}$ for three values of $\mu$. Dong et al. suggest that $\mu \le 0.5$ is a reasonable guarantee and $\mu \ge 6$ promises almost nothing.

| Type I error $a$ | Smallest type II error $G_1(a)$ | Best power $1 - G_1(a)$ |
| --- | --- | --- |
| 0.01 | 0.9076 | 0.0924 |
| 0.05 | 0.7405 | 0.2595 |
| 0.1 | 0.6109 | 0.3891 |

: The 1-GDP guarantee of the Gaussian mechanism with $\sigma = 1$, read as attack errors. {#tbl-gdp-g1}

| $\mu$ | 0.5 | 1 | 2 |
| --- | --- | --- | --- |
| $\varepsilon$ at $\delta = 10^{-5}$ | 1.9931 | 4.3772 | 9.9973 |

: The $\varepsilon$ of $\mu$-GDP at $\delta = 10^{-5}$. {#tbl-gdp-eps}

**Pitfalls.** "1-GDP" is not the same guarantee as "$(4.3772, 10^{-5})$-DP". The first implies the second, but the single pair allows far more power: at $a = 0.1$ it permits $b = 0.0113$, while $G_1(0.1) = 0.6109$. Report $\mu$, or the noise multiplier, along with any pair. Smaller $\delta$ costs $\varepsilon$ roughly in proportion to $\mu$: at $\mu = 1$, going from $\delta = 10^{-4}$ to $10^{-5}$ to $10^{-6}$ raises $\varepsilon$ from 3.8044 to 4.3772 to 4.8866, and at $\mu = 2$ the steps are about twice as large.

![Left: the privacy profile $\delta(\varepsilon)$ of $\mu$-GDP for $\mu = 0.5, 1, 2$, with the level $\delta = 10^{-5}$ dotted. Right: $G_1$ and six of its $(\varepsilon, \delta(\varepsilon))$ curves; their upper envelope is $G_1$, so the whole profile loses nothing, while one pair keeps a single touching line.](figures/gaussian_dp_duality.svg){#fig-gdp-duality}

## Composition {#sec-gdp-composition}

**Idea.** An attacker who sees two outputs with independent noise faces the pair of joint distributions, and their trade-off curve is the composed guarantee. For Gaussian curves the joint problem is a two-dimensional Gaussian shift, which is again a one-dimensional shift of length $\sqrt{\mu_1^2 + \mu_2^2}$. So $\mu^2$ adds under composition, just as the variances of independent noises add.

**Result.** If each step is $\mu_i$-GDP, even when it is chosen after seeing the earlier outputs, the composition is
$$G_{\mu_1} \otimes \dots \otimes G_{\mu_k} = G_\mu, \qquad \mu = \sqrt{\mu_1^2 + \dots + \mu_k^2},$$ {#eq-gdp-compose}
and no better bound holds in general (Dong et al. 2022). Full-batch noisy gradient descent with noise multiplier $\sigma$ for $k$ steps is therefore exactly $\sqrt k/\sigma$-GDP. zCDP does the same arithmetic, with $\rho = \mu^2/2$; the difference is that GDP keeps the exact curve, while $\rho$ only gives an RDP bound on it.

**Example.** Two 1-GDP releases compose to $G_{1.4142}$, four to $G_2$, and 100 releases with $\sigma = 10$ each to $G_1$. Four 1-GDP releases are therefore $\varepsilon = 9.9973$ at $\delta = 10^{-5}$. If you first turn each release into one $(\varepsilon, \delta)$ point, giving each a quarter of the $\delta$ budget, $(4.6902, 2.5\times10^{-6})$, and then add the points by basic composition, you get $\varepsilon = 18.7607$: almost twice as large.

![Left: 400 draws each from $\mathcal N(0, I_2)$ and $\mathcal N((1, 1), I_2)$, the outputs of two Gaussian releases on $D$ and $D'$; the best test thresholds the projection on the shift. Right: $G_1$ for one release, $G_{\sqrt 2}$ for two, and simulated errors of the projected test.](figures/gaussian_dp_composition.svg){#fig-gdp-composition}

**Pitfalls.** Compose curves, not points: converting each step to $(\varepsilon, \delta)$ first throws away most of each curve. For curves that are not Gaussian the composition has no closed form, and composing $(\varepsilon, \delta)$ guarantees exactly is computationally intractable when the steps differ. That is why [numerical accounting](06-numerical-accounting.md) composes the distribution of the privacy loss by convolution instead. And GDP composes exactly only for Gaussian noise on the full data; a subsampled step's curve is not Gaussian, which is the subject of the next section.

## The central limit theorem and DP-SGD {#sec-gdp-sgd}

**Idea.** The **privacy loss** of an output is the log of how much more likely it is under $D$ than under $D'$, and the loss of a composition is the sum of the per-step losses. A sum of many small independent terms is nearly normal, and a normal privacy loss means a Gaussian curve. So many small steps compose to roughly some $G_\mu$, whatever the steps were. DP-SGD is the case that matters: each step runs on a Poisson sample that includes each record with probability $q$, clips each gradient and adds Gaussian noise with noise multiplier $\sigma$, and when $q$ is small each step leaks little.

**Result.** Under add/remove neighbours, as $k \to \infty$ with $q\sqrt k$ held fixed, $k$ steps of the Poisson-subsampled Gaussian are asymptotically $\mu$-GDP with
$$\mu = q\sqrt{k\big(e^{1/\sigma^2} - 1\big)}$$ {#eq-gdp-sgd-mu}
(Bu, Dong, Long & Su 2020). Compared with full-batch training, $\sqrt k/\sigma$, subsampling multiplies $\mu$ by about $q\sigma\sqrt{e^{1/\sigma^2} - 1}$, which is $q \times 1.3108$ at $\sigma = 1$. This is a limit, used at finite $k$ as an approximation, not a bound.

**Example.** For the part's running example, $q = 0.01$ and $\sigma = 1$, @eq-gdp-sgd-mu gives $\mu = 0.01 \times 1.3108\sqrt k$. @tbl-gdp-sgd turns each $\mu$ into $\varepsilon$ at $\delta = 10^{-5}$ with @eq-gdp-delta and compares it with a numerical accountant's bracket for the true value. At $k = 10{,}000$ the CLT says 6.0071; the true $\varepsilon$ is at least 6.1377.

| Steps $k$ | CLT $\mu$ | CLT $\varepsilon$ at $10^{-5}$ | Numerical accountant $\varepsilon$ | Full batch, $\mu = \sqrt k$ |
| --- | --- | --- | --- | --- |
| 100 | 0.1311 | 0.4575 | [0.7175, 0.7185] | 10.0 |
| 1,000 | 0.4145 | 1.6177 | [1.8232, 1.8332] | 31.6 |
| 10,000 | 1.3108 | 6.0071 | [6.1377, 6.2377] | 100.0 |

: The running DP-SGD example: the CLT's $\mu$ and $\varepsilon$, and a numerical bracket for the true $\varepsilon$. {#tbl-gdp-sgd}

![The CLT for DP-SGD with $q = 0.01$. Left: $\mu = q\sqrt{k(e^{1/\sigma^2} - 1)}$ against the number of steps for three noise multipliers. Right: the $\varepsilon$ at $\delta = 10^{-5}$ it implies. Dots mark the running example at $k = 100$, 1,000 and 10,000.](figures/gaussian_dp_subsampled.svg){#fig-gdp-subsampled}

**Pitfalls.** Never report the CLT value as a guarantee. It under-reported $\varepsilon$ at every $k$ in @tbl-gdp-sgd, worst for short runs (0.4575 against at least 0.7175 at $k = 100$), because a CLT controls the bulk of the loss distribution, while small $\delta$ is decided by its tail. Plugging in exactly computed per-step moments does not help: it gives an even smaller $\varepsilon$, 5.9184 at $k = 10{,}000$. Report a certified accountant's number, and use the CLT for intuition and planning. The result also depends on the sampling scheme and the neighbouring relation: Dong et al.'s version for fixed-size batches under replace-one neighbours has a different constant, $\mu = 1.7101$ at $k = 10{,}000$ here, and the two must not be mixed ([Amplification by subsampling](05-subsampling.md)). Finally, $\mu$ is very sensitive to $\sigma$: $e^{1/\sigma^2} - 1$ is 3.7707 at $\sigma = 0.8$, 1.7183 at $\sigma = 1$ and 0.5596 at $\sigma = 1.5$.

## RDP as a lossy summary of the curve {#sec-gdp-rdp}

**Idea.** An RDP curve records a few exponential moments of the privacy loss. Different loss distributions share those moments, so the best trade-off curve that the RDP values alone can certify, $f_{\mathrm{RDP}}$, has to allow the worst of them. It lies below the true curve even for the Gaussian, where the RDP curve is exact.

**Result.** A mechanism with a given RDP curve is $f_{\mathrm{RDP}}$-DP, and no higher curve follows from the RDP values alone. The best $(\varepsilon, \delta)$ any RDP conversion can report is the profile of $f_{\mathrm{RDP}}$ ([From Rényi DP to (ε, δ)](02-rdp-conversions.md)). So the slack of RDP accounting has two parts: the conversion, which a better formula removes, and the gap between $f_{\mathrm{RDP}}$ and the true curve, which nothing removes.

**Example.** For 1-GDP, with RDP curve $\varepsilon(\alpha) = \alpha/2$, the RDP summary allows an attacker an advantage of 0.4830 where the best possible is 0.3829, and the best $\delta(1)$ it can certify is 0.2149, 1.69 times the exact 0.1269.

**Pitfalls.** For Gaussian noise on the full data, account in GDP, where $\mu^2$ adds and the profile at the end is exact. Converting an RDP or zCDP total leaves this gap on the table.

## Code {#sec-gdp-code}

A GDP accountant needs only the normal distribution function, which the standard library provides through `math.erfc`:

```python
from math import erfc, exp, sqrt


def normal_cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def gdp_delta(eps: float, mu: float) -> float:
    """The exact delta(eps) of mu-GDP."""
    return normal_cdf(-eps / mu + mu / 2) - exp(eps) * normal_cdf(-eps / mu - mu / 2)


def gdp_eps(mu: float, delta: float) -> float:
    """The eps at which mu-GDP reaches delta; delta(eps) falls as eps grows, so bisect."""
    lo, hi = 0.0, 500.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gdp_delta(mid, mu) > delta else (lo, mid)
    return hi


def compose_gdp(*mus: float) -> float:
    return sqrt(sum(m * m for m in mus))


def clt_mu_dpsgd(q: float, sigma: float, steps: int) -> float:
    """Approximate mu of Poisson-subsampled DP-SGD. An estimate, not a bound."""
    return q * sqrt(steps * (exp(1 / sigma**2) - 1))


print(f"{gdp_eps(1.0, 1e-5):.4f}")  # 4.3772
print(f"{gdp_eps(compose_gdp(1, 1, 1, 1), 1e-5):.4f}")  # 9.9973
mu = clt_mu_dpsgd(q=0.01, sigma=1.0, steps=10_000)
print(f"{mu:.4f} {gdp_eps(mu, 1e-5):.4f}")  # 1.3108 6.0071
```

The last line is the CLT estimate for the running example, below the certified lower bound 6.1377. The full edition's script, [gaussian_dp.py](../../code/part04-privacy-accounting/gaussian_dp.py), computes every number in this chapter, including the numerical brackets:

```bash
uv run python code/part04-privacy-accounting/gaussian_dp.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| $f$-DP | $T(M(D), M(D')) \ge f$ for all neighbours | Type I error $a$, type II error $b$ |
| $(\varepsilon, \delta)$-DP as a curve | $f_{\varepsilon,\delta}(a) = \max(0, 1 - \delta - e^\varepsilon a, e^{-\varepsilon}(1 - \delta - a))$ | Two straight lines |
| $\mu$-GDP | $G_\mu(a) = \Phi(\Phi^{-1}(1 - a) - \mu)$ | Gaussian mechanism: $\mu = 1/\sigma$, exact |
| Profile | $\delta(\varepsilon) = \Phi(-\frac\varepsilon\mu + \frac\mu2) - e^\varepsilon\Phi(-\frac\varepsilon\mu - \frac\mu2)$ | Lossless; bisect for $\varepsilon$ |
| Composition | $\mu = \sqrt{\sum_i\mu_i^2}$ | Exact for Gaussian steps |
| DP-SGD, CLT | $\mu \approx q\sqrt{k(e^{1/\sigma^2} - 1)}$ | An approximation; under-reports $\varepsilon$ |
| RDP from the curve | RDP implies only $f_{\mathrm{RDP}}$, below the true curve | Slack no conversion removes |

: The results of this chapter. {#tbl-gdp-cheat}

**Rules of thumb**

- The trade-off curve is the guarantee; $(\varepsilon, \delta)$, RDP and $\rho$ summarize it. Summarize as late as possible.
- For Gaussian noise without subsampling, account in GDP: $\mu^2$ adds, and the $(\varepsilon, \delta)$ at the end is exact.
- $\mu = 1$ means $\varepsilon \approx 4.4$ at $\delta = 10^{-5}$; $\mu = 0.5$ means about 2.
- Use the CLT for DP-SGD to plan, never as the reported guarantee.

## Further reading

- Dong, Roth & Su, "Gaussian Differential Privacy" (Journal of the Royal Statistical Society Series B, 2022): $f$-DP, GDP, duality, composition and the CLTs. The discussion sections explain why the curve is the right object.
- Bu, Dong, Long & Su, "Deep Learning with Gaussian Differential Privacy" (Harvard Data Science Review, 2020): the CLT for Poisson-subsampled DP-SGD, with experiments.
- Gopi, Lee & Wutschitz, "Numerical Composition of Differential Privacy" (NeurIPS 2021): an accountant with error bounds, and examples where the CLT under-reports $\varepsilon$.
- Balle & Wang, "Improving the Gaussian Mechanism for Differential Privacy" (ICML 2018): the exact $(\varepsilon, \delta)$ profile of the Gaussian mechanism and its use for calibration.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/04-gaussian-dp.md).
