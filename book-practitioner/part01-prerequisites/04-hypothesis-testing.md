# Hypothesis testing

## Why this matters

A spam filter, a fraud detector and a drift monitor all face one problem: **an observation came from one of two distributions, and you must decide which.** You need the best decision rule, how good it can be, which summary of it to trust, and how many test cases it takes to measure. Later parts ask how well any test can tell two distributions apart; the answer is the trade-off curve built here.

The one idea is the **likelihood ratio** $\Lambda(x) = q(x)/p(x)$, how many times more likely the observation is under the alternative $Q$ than under the default $P$. The best test thresholds it, and moving the threshold traces a curve that depends only on $P$ and $Q$. @tbl-ht-overview maps the decisions to sections.

| Decision | Rule | Section |
| --- | --- | --- |
| Which test to use | Threshold the likelihood ratio, or any score monotone in it | @sec-ht-np |
| How good the best test can be | The trade-off curve; closed forms for normal and Laplace shifts | @sec-ht-tradeoff |
| Which summary to report | True positive rate at a fixed low false positive rate, not the area under the curve (AUC) or the best advantage | @sec-ht-auc |
| What an alarm means at a given base rate | Posterior odds $=$ prior odds $\times$ likelihood ratio | @sec-ht-bayes |
| How many test cases to measure a rate | About $100/\alpha$; report a Clopper–Pearson interval | @sec-ht-estimation |

: The decisions this chapter covers, and where each is made. {#tbl-ht-overview}

## Errors and the Neyman–Pearson lemma {#sec-ht-np}

**Idea.** Rejecting an outcome $x$ costs $p(x)$ of false alarms and buys $q(x)$ of detection, an exchange rate of $\Lambda(x)$. For the most detection on a fixed false-alarm budget, spend it where the rate is highest: reject where $\Lambda(x)$ is above a cut-off.

**Result.** For the null hypothesis $H_0: X \sim P$ against the alternative $H_1: X \sim Q$, a test $\phi(x) \in [0, 1]$ rejects $H_0$ with probability $\phi(x)$. Its **false positive rate (FPR)**, or type I error, is $\alpha = \mathbb E_P[\phi]$; its **type II error** is $\beta = 1 - \mathbb E_Q[\phi]$, and its **power** or **true positive rate (TPR)** is $1 - \beta$. The **Neyman–Pearson lemma**: the test that rejects when $q(x) > \eta\,p(x)$, keeps $H_0$ when $q(x) < \eta\,p(x)$, and randomizes on ties so that its FPR is exactly $\alpha$, has the smallest $\beta$ of any test with FPR at most $\alpha$. If $\Lambda$ increases with a statistic $s(x)$, the optimal test is a plain threshold on $s$.

**Example.** A spam filter counts suspicious words, capped at 3 (@tbl-ht-spam). To spend a false positive budget of exactly 0.25, reject a count of 3 always and a count of 2 with probability 0.75: power 0.625. Of 200,000 random tests, the best with FPR at most 0.25 has power 0.587.

| $x$ | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| $p(x)$, ham | 0.4 | 0.3 | 0.2 | 0.1 |
| $q(x)$, spam | 0.1 | 0.2 | 0.3 | 0.4 |
| $\Lambda(x) = q(x)/p(x)$ | 0.250 | 0.667 | 1.500 | 4.000 |

: Word-count distributions under ham ($P$) and spam ($Q$), and their likelihood ratio. {#tbl-ht-spam}

**Pitfalls.** A calibrated score $\Pr[H_1 \mid x]$ increases with $\Lambda(x)$, so thresholding it is optimal; a score not monotone in $\Lambda$ can be beaten by re-ranking. On discrete data a threshold cannot hit every FPR: randomize at the boundary or state the FPR you get.

## Trade-off curves {#sec-ht-tradeoff}

**Idea.** The best $\beta$ at every $\alpha$ is a curve that belongs to the pair $P, Q$: the further below the diagonal $1 - \alpha$, the easier the problem. Its slope is minus the likelihood ratio at the threshold, so a bounded ratio means little power at small $\alpha$.

**Result.** The **trade-off function** is $T(\alpha) = \inf\{\beta_\phi : \alpha_\phi \le \alpha\}$. It is non-increasing, convex and at most $1 - \alpha$, the coin flip that ignores the data. Turned upside down, as power against FPR, it is the optimal **receiver operating characteristic (ROC)** curve. For $\mathcal N(0, 1)$ against $\mathcal N(\mu, 1)$, with $\Phi$ the standard normal cumulative distribution function (CDF),
$$T(\alpha) = \Phi\big(\Phi^{-1}(1 - \alpha) - \mu\big).$$ {#eq-ht-gaussian-tradeoff}
Averaging $n$ draws to detect a mean shift $\delta$ under noise $\sigma$ gives the same curve with $\mu = \delta\sqrt n/\sigma$, so power $1 - \beta'$ at level $\alpha$ needs
$$n \ge \left(\frac{\Phi^{-1}(1 - \alpha) + \Phi^{-1}(1 - \beta')}{\delta/\sigma}\right)^2.$$
For $\mathrm{Lap}(0, b)$ against $\mathrm{Lap}(s, b)$, with $\kappa = s/b$, the ratio is clipped to $[e^{-\kappa}, e^\kappa]$ and
$$T(\alpha) = \begin{cases} 1 - e^{\kappa}\alpha, & \alpha < \tfrac12 e^{-\kappa}, \\ e^{-\kappa}/(4\alpha), & \tfrac12 e^{-\kappa} \le \alpha \le \tfrac12, \\ e^{-\kappa}(1 - \alpha), & \alpha > \tfrac12. \end{cases}$$ {#eq-ht-laplace-tradeoff}
In particular, no test can have power above $e^\kappa\alpha$.

**Example.** A monitor averages 100 readings to detect a drift of $0.2\sigma$, so $\mu = 2$: power 0.6388 at $\alpha = 0.05$, only 0.1378 at $10^{-3}$. Power 0.9 at $\alpha = 0.01$ for $\delta/\sigma = 0.25$ needs 209 readings. @tbl-ht-laplace-vs-gaussian compares a Laplace pair ($\kappa = 1$) and a normal pair ($\mu = 1.0301$) with the same best advantage, 0.3935.

| $\alpha$ | Laplace power, $\kappa = 1$ | Normal power, $\mu = 1.0301$ |
| --- | --- | --- |
| $10^{-3}$ | $2.718\times10^{-3}$ | $1.969\times10^{-2}$ |
| $10^{-6}$ | $2.718\times10^{-6}$ | $9.829\times10^{-5}$ |

: Best power at small false positive rates for a Laplace pair and a normal pair that agree in the bulk. {#tbl-ht-laplace-vs-gaussian}

![Left: the log likelihood ratio of $\mathrm{Lap}(1, 1)$ against $\mathrm{Lap}(0, 1)$ is clipped at $\pm1$, while the normal one with the same best advantage is a straight line. Right: best power on log axes. The Laplace curve follows $e\alpha$ for small $\alpha$; the normal curve rises far above it.](figures/hypothesis_testing_laplace_tradeoff.svg){#fig-ht-laplace-tradeoff}

**Pitfalls.** Two pairs can agree in the bulk and differ enormously at small $\alpha$: at $10^{-6}$ the normal pair is 36 times easier, because its unbounded ratio lets a test find rare, overwhelming evidence (@fig-ht-laplace-tradeoff). Plan sample sizes at the level you will actually use.

## One-number summaries {#sec-ht-auc}

**Idea.** The two common one-number summaries both hide the regime of rare false alarms, usually the one that matters.

**Result.** The **advantage** of a test is $\mathrm{TPR} - \mathrm{FPR}$, and the best advantage over all tests equals the **total variation (TV)** distance, $\frac12\sum_x|q(x) - p(x)|$, reached by rejecting where $q > p$. It is $2\Phi(\mu/2) - 1$ for the normal pair and $1 - e^{-\kappa/2}$ for the Laplace pair. The **AUC** of a ROC curve is the probability that a random positive scores above a random negative. If negatives score $\mathcal N(0, 1)$ and positives $\mathcal N(m, \sigma^2)$, then $\mathrm{AUC} = \Phi(m/\sqrt{1 + \sigma^2})$ and
$$\mathrm{TPR}(\mathrm{FPR}) = 1 - \Phi\Big(\frac{\Phi^{-1}(1 - \mathrm{FPR}) - m}{\sigma}\Big).$$

**Example.** Detector A has positives scoring $\mathcal N(1, 1)$; detector B has $\mathcal N(1.5811, 2^2)$. Both have AUC 0.7602. At an FPR of 0.1 their TPRs are 0.3891 and 0.5595, and at $10^{-3}$ they are 0.0183 and 0.2253: B catches 12 times as many positives (@fig-ht-auc-low-fpr).

![ROC curves of two detectors with the same AUC of 0.760. Left, linear axes: the curves cross and look comparable. Right, log axes: at an FPR of $10^{-3}$ detector B has 12 times the TPR of detector A.](figures/hypothesis_testing_auc_low_fpr.svg){#fig-ht-auc-low-fpr}

**Pitfalls.** The AUC weights all FPRs equally, so the region below $10^{-3}$ barely affects it; TV looks at one point of the curve. Report the TPR at a fixed low FPR chosen by the application, and plot ROC curves on log–log axes.

## Base rates: the Bayesian view {#sec-ht-bayes}

**Idea.** With a known base rate, the question becomes how likely $H_1$ is given what you saw, and the likelihood ratio answers it in one multiplication.

**Result.** With prior probability $\pi$ for $H_1$,
$$\frac{\Pr[H_1 \mid x]}{\Pr[H_0 \mid x]} = \frac{\pi}{1 - \pi} \times \Lambda(x).$$
If all you know is that the test fired, the likelihood ratio of that event is $\mathrm{TPR}/\mathrm{FPR}$. The error-minimizing rule rejects when $\Lambda(x) > (1 - \pi)/\pi$.

**Example.** Use the $\mu = 2$ test in a population where 1% are ill. At an FPR of 0.05 a positive has likelihood ratio $0.6388/0.05 = 12.8$, and the chance it is a real case is only 0.114. At an FPR of $10^{-3}$ the ratio is 137.8 and the chance is 0.582.

**Pitfalls.** When positives are rare, the FPR, not the power, decides whether an alarm is worth acting on.

## Measuring rates from finite trials {#sec-ht-estimation}

**Idea.** A measured FPR is $k$ false positives among $n$ negatives, a binomial count whose uncertainty at low rates is large compared with the rate. "TPR at FPR $10^{-3}$" is only as good as the interval around that FPR.

**Result.** The **Clopper–Pearson** interval at confidence $1 - a$ is $[l, u]$ with $\Pr[\mathrm{Bin}(n, l) \ge k] = a/2$ and $\Pr[\mathrm{Bin}(n, u) \le k] = a/2$. It covers the true rate with probability at least $1 - a$. With no events, the 95% upper bound is about $3/n$, the **rule of three**. The relative width is about $1/\sqrt{n\alpha}$, so measuring an FPR of $\alpha$ to within a factor of about 1.2 takes some $100/\alpha$ negatives.

**Example.** The $\mu = 2$ test at its $10^{-3}$ threshold is evaluated on 10,000 negatives and 10,000 positives. One run gives 10 false positives, interval $[0.00048, 0.00184]$, and 1,350 true positives, interval $[0.1284, 0.1419]$: the FPR is known only to a factor of two. To certify an FPR below $10^{-3}$ at 95% confidence you need 2,995 negatives with no false positive at all.

**Pitfalls.** The normal-approximation (Wald) interval is too narrow for a skewed count and collapses to $[0, 0]$ at $k = 0$. Hoeffding's ignores the tiny variance: its half-width here is 0.0136, over 13 times the rate. A threshold tuned on the same data has a true FPR as uncertain as the interval: choose on one split, measure on another.

## Code {#sec-ht-code}

The normal trade-off curve and the Clopper–Pearson interval need only the standard library; bisection on the upper tail keeps precision when $\alpha$ is tiny:

```python
from collections.abc import Callable
from math import erfc, exp, lgamma, log, log1p, sqrt


def normal_cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def upper_quantile(a: float) -> float:
    """The point a standard normal exceeds with probability a, found by bisection on the tail."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 0.5 * erfc(mid / sqrt(2)) > a else (lo, mid)
    return lo


def gaussian_power(alpha: float, mu: float) -> float:
    """Best power at false positive rate alpha between N(0, 1) and N(mu, 1)."""
    return 1 - normal_cdf(upper_quantile(alpha) - mu)


def binom_cdf(k: int, n: int, p: float) -> float:
    """Pr[Binomial(n, p) <= k], summed in log space."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    total = 0.0
    for i in range(k + 1):
        log_pmf = lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * log(p) + (n - i) * log1p(-p)
        total += exp(log_pmf)
    return total


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided interval for a rate after k successes in n trials."""
    a = (1 - confidence) / 2

    def solve(f: Callable[[float], float], target: float) -> float:
        """The rate p at which f, a binomial CDF that falls as p grows, equals target."""
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
        return lo

    # Lower end: Pr[Bin(n, l) >= k] = a, that is Pr[Bin(n, l) <= k - 1] = 1 - a.
    lower = 0.0 if k == 0 else solve(lambda p: binom_cdf(k - 1, n, p), 1 - a)
    # Upper end: Pr[Bin(n, u) <= k] = a.
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p), a)
    return lower, upper


print(f"{gaussian_power(0.05, 2.0):.4f} {gaussian_power(1e-3, 2.0):.4f}")  # 0.6388 0.1378
low, high = clopper_pearson(10, 10_000)
print(f"[{low:.5f}, {high:.5f}]")  # [0.00048, 0.00184]
print(f"{1 - 0.05 ** (1 / 3000):.3e}")  # 9.981e-04
```

The last line is the exact 95% upper bound after no events in 3,000 trials, close to the rule of three's $3/n$.

## Cheat sheet

| Result | Statement |
| --- | --- |
| Errors of a test $\phi$ | $\alpha = \mathbb E_P\phi$ (FPR), $\beta = 1 - \mathbb E_Q\phi$, power $= 1 - \beta$ (TPR) |
| Neyman–Pearson | Rejecting where $q/p > \eta$, randomizing at ties, is most powerful at its level |
| Normal shift $\mu$ | $T(\alpha) = \Phi(\Phi^{-1}(1 - \alpha) - \mu)$; $n$ averaged draws: $\mu = \delta\sqrt n/\sigma$ |
| Laplace shift $\kappa = s/b$ | @eq-ht-laplace-tradeoff; power $\le e^\kappa\alpha$ |
| Best advantage | $\max(\mathrm{TPR} - \mathrm{FPR}) = \mathrm{TV}$; normal $2\Phi(\mu/2) - 1$, Laplace $1 - e^{-\kappa/2}$ |
| AUC | $\Pr[S_+ > S_-]$; binormal $\Phi(m/\sqrt{1 + \sigma^2})$ |
| Bayes | Posterior odds $=$ prior odds $\times\,\Lambda$; an alarm has $\Lambda = \mathrm{TPR}/\mathrm{FPR}$ |
| Clopper–Pearson | $\Pr[\mathrm{Bin}(n, l) \ge k] = \Pr[\mathrm{Bin}(n, u) \le k] = a/2$; zero events: $u \approx 3/n$ |

: The results of this chapter. {#tbl-ht-cheat-sheet}

**Rules of thumb**

- To find the best test, compute the likelihood ratio and threshold it.
- Compare detectors by their TPR at a fixed low FPR, on log axes; AUC and TV hide that regime.
- A bounded likelihood ratio caps power at small $\alpha$: power $\le \Lambda_{\max}\,\alpha$.
- To measure a rate near $\alpha$, plan for about $100/\alpha$ trials and report a Clopper–Pearson interval.

## Further reading

- Casella & Berger, *Statistical Inference* (2nd ed., 2002), §8.3: the Neyman–Pearson lemma and its one-sided extensions.
- Fawcett, "An introduction to ROC analysis" (Pattern Recognition Letters, 2006): ROC curves, AUC as a ranking probability, and convex hulls.
- Brown, Cai & DasGupta, "Interval Estimation for a Binomial Proportion" (Statistical Science, 2001): why the Wald interval fails and what to use instead.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/04-hypothesis-testing.md).
