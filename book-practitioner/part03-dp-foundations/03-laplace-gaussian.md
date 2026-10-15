# The Laplace and Gaussian mechanisms

## Why this matters

You want to publish a statistic with a differential privacy (DP) guarantee: how many patients in a hospital database have diabetes, the mean age of a cohort, a histogram of diagnoses, or a gradient during training. Almost every DP system does it the same way. It computes the exact answer $f(D)$ on the dataset $D$, adds fresh random noise $Z$, and publishes
$$M(D) = f(D) + Z.$$
If the noise is wide compared with how far one person can move the answer, nobody looking at the output can tell whether that person was in the data. The guarantee you are aiming for is the one from [The definition of differential privacy](01-definition.md): for any two **neighbouring** datasets $D$ and $D'$, which differ in one person's record, and any set $S$ of outputs, $\Pr[M(D) \in S] \le e^\varepsilon \Pr[M(D') \in S] + \delta$. A small $\varepsilon$ means one record barely changes what anyone can see, and $\delta$ is a small allowance for that promise to fail.

Three decisions set up every noisy release, and this chapter takes them in order. How far can one record move the answer? That is the sensitivity, and the noise grows with it. Which noise, Laplace or Gaussian? And how much of it? @tbl-lg-overview summarizes the answers.

| Decision | Rule | Section |
| --- | --- | --- |
| How far one record moves the answer | Compute the sensitivity for your neighbouring relation, before you look at the data | @sec-lg-sensitivity |
| One number or a sparse vector | Laplace noise with scale $\Delta_1f/\varepsilon$ | @sec-lg-laplace |
| A dense vector, such as a gradient | Gaussian noise, calibrated exactly to $\Delta_2f$ | @sec-lg-gauss |
| Choosing between them | Compare the error per coordinate | @sec-lg-choose |

: The decisions behind a noisy release, and where this chapter makes each one. {#tbl-lg-overview}

## Sensitivity {#sec-lg-sensitivity}

**Idea.** To hide one record, the noise has to cover the largest change one record can make to the answer. That change is the **sensitivity**. It depends on the query, on how you measure a change in a vector of answers, and on what "differ in one record" means. Under **add/remove** neighbours one dataset has an extra record, so the dataset's size is private. Under **replace** neighbours one record is swapped for another, so the size $n$ is fixed and public.

**Result.** The $\ell_p$ sensitivity is the largest distance between the answers on two neighbouring datasets:
$$\Delta_pf = \max_{D \simeq D'}\big\|f(D) - f(D')\big\|_p, \qquad p \in \{1, 2\}.$$
The $\ell_1$ norm sums the absolute changes over all coordinates, and Laplace noise is calibrated to it. The $\ell_2$ norm is the straight-line length of the change, and Gaussian noise is calibrated to it. The two are never far apart: $\Delta_2f \le \Delta_1f \le \sqrt d\,\Delta_2f$ for a query with $d$ coordinates. They agree when one record moves a single coordinate, and they are furthest apart when it moves all of them. @tbl-lg-sensitivity gives the sensitivities of the standard queries.

| Query | Add/remove $\Delta_1$ | Add/remove $\Delta_2$ | Replace $\Delta_1$ | Replace $\Delta_2$ |
| --- | --- | --- | --- | --- |
| Count $\lvert D\rvert$ | 1 | 1 | 0 | 0 |
| Sum, records in $[0, B]$ | $B$ | $B$ | $B$ | $B$ |
| Mean, records in $[0, B]$, $n$ public | — | — | $B/n$ | $B/n$ |
| Histogram with $k$ bins | 1 | 1 | 2 | $\sqrt2$ |
| $d$ threshold counts $\#\{x \ge j\}$, $j = 1..d$ | $d$ | $\sqrt d$ | $d$ | $\sqrt d$ |

: Sensitivities of the standard queries under the two neighbouring relations. {#tbl-lg-sensitivity}

**Example.** A study releases the mean age of $n = 1000$ patients, with ages clipped to $[0, 100]$, under replace neighbours. The sensitivity is $B/n = 100/1000 = 0.1$: one patient can move the mean by at most a tenth of a year, however old they are.

**Pitfalls.** Sensitivity is a worst case over every dataset the domain allows, not a property of the data you hold. Computing it from your data, for example from the largest value present, leaks information; [Beyond global sensitivity](06-sensitivity.md) shows how, and how to do better. The sum of unbounded values has infinite sensitivity, so clip every value to a fixed range first. Always state the neighbouring relation along with $\varepsilon$: a histogram has $\Delta_1 = 1$ under add/remove and 2 under replace, so the same noise gives guarantees a factor of 2 apart in $\varepsilon$. Under replace, the count of all records has sensitivity 0 because $n$ is public. If the size of your dataset is itself sensitive, use add/remove.

## The Laplace mechanism {#sec-lg-laplace}

**Idea.** The **Laplace distribution** $\mathrm{Lap}(b)$ with scale $b$ has density $\frac1{2b}e^{-|z|/b}$, a two-sided exponential. Its log falls off in a straight line, at slope $1/b$ on each side. Moving the noise's centre by $\Delta$ therefore changes the log-density at every point by at most $\Delta/b$, so no output can be more than $e^{\Delta/b}$ times as likely under one dataset as under its neighbour. That is pure DP with $\varepsilon = \Delta/b$. @fig-lg-privacy-loss shows this for a count.

**Result.** For a query with $\ell_1$ sensitivity $\Delta_1f$, add independent $\mathrm{Lap}(b)$ noise to each of the $d$ coordinates, with
$$b = \frac{\Delta_1f}{\varepsilon}.$$ {#eq-lg-laplace}
The release is $\varepsilon$-DP, and no smaller scale works in general. The error in each coordinate is $b$ on average. With probability at least $1 - \beta$, every one of the $d$ coordinates is within $b\ln(d/\beta)$ of the truth.

![Left column: a count released with $\mathrm{Lap}(1)$ noise on neighbouring datasets with true counts 0 and 1, and the privacy loss $L(y)$, the log of how much more likely the output $y$ is under one dataset than the other. It equals $+\varepsilon$ for every $y \le 0$ and $-\varepsilon$ for every $y \ge 1$, never more. Right column: the same count with $\mathcal N(0, 2^2)$ noise. The privacy loss is a straight line, and exceeds $\varepsilon = 1$ on the shaded tail, which has probability 0.0401 under $M(D)$. The exact $\delta(1) = 0.0068$ is smaller, because the bound subtracts $e^\varepsilon$ times the mass of $M(D')$ on the same tail.](figures/lapgauss_privacy_loss.svg){#fig-lg-privacy-loss}

**Example.** 1,234 of the hospital's 10,000 patients have diabetes. Released with $\varepsilon = 0.1$ under add/remove, the sensitivity is 1, so the noise is $\mathrm{Lap}(10)$. The expected absolute error is 10, and with probability 0.95 the error is below $10\ln 20 = 29.96$, a relative error of 2.4%. A histogram of diagnoses with 20 bins at $\varepsilon = 1$ gets $\mathrm{Lap}(1)$ noise in each bin, and every bin is within $\ln(20/0.05) = 5.99$ of its true count with probability 0.95.

**Pitfalls.** The noise does not shrink as the data grow. A count gets noise of scale $1/\varepsilon$ whether the database has a hundred records or a hundred million, so its relative error is about $1/(n\varepsilon)$: large populations are cheap to protect, small subgroups are expensive. If one record can affect several of your numbers, the sensitivity counts all of them: ten counts that each get $\mathrm{Lap}(1/\varepsilon)$ noise, and that one record can all move, form a $10\varepsilon$-DP release. Rounding the noisy output or clamping negative counts to zero costs no privacy. Sampling the noise is another matter: textbook floating-point Laplace samplers break the guarantee, as [Discrete and floating-point-safe mechanisms](07-discrete-mechanisms.md) shows, so use a vetted DP library rather than `numpy.random.laplace`.

## The Gaussian mechanism {#sec-lg-gauss}

**Idea.** Gaussian noise is calibrated to the $\ell_2$ sensitivity, which can be far smaller than the $\ell_1$ sensitivity. The price: the log of the normal density falls off like $-z^2$, so the privacy loss grows without limit far out in the tail (@fig-lg-privacy-loss, right). Gaussian noise therefore cannot give pure DP. It gives $(\varepsilon, \delta)$-DP, where $\delta$ accounts for the rare outputs whose loss exceeds $\varepsilon$.

**Result.** Only the ratio of sensitivity to noise matters, $\mu = \Delta_2f/\sigma$: how many noise standard deviations apart the answers on two neighbours can be. Adding $\mathcal N(0, \sigma^2)$ noise to each coordinate is $(\varepsilon, \delta)$-DP exactly when $\delta \ge \delta_\mu(\varepsilon)$, where
$$\delta_\mu(\varepsilon) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^\varepsilon\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big)$$ {#eq-lg-analytic}
and $\Phi$ is the standard normal cumulative distribution function. In words, the first term is the probability, on one dataset, of the outputs that favour it by more than $e^\varepsilon$, and the second subtracts $e^\varepsilon$ times their probability on its neighbour; what is left is exactly the slack $\delta$ the definition needs (Balle & Wang 2018). More noise means a smaller $\mu$ and a smaller $\delta_\mu(\varepsilon)$, so the **analytic calibration**, the smallest $\sigma$ that meets a target $(\varepsilon, \delta)$, is found by solving $\delta_\mu(\varepsilon) = \delta$ for $\mu$ by bisection. @sec-lg-code gives the code.

Many papers and libraries still use the **classic calibration** of Dwork & Roth,
$$\sigma = \frac{\Delta_2f\sqrt{2\ln(1.25/\delta)}}{\varepsilon},$$
which is proved only for $\varepsilon < 1$. @tbl-lg-calibration compares the two at $\delta = 10^{-5}$. Below $\varepsilon = 1$ the classic formula uses 1.3 to 1.7 times the noise needed. Above, it is not even guaranteed to be safe, and at $\delta = 10^{-5}$ it uses too little noise from $\varepsilon = 8.42$ on.

| $\varepsilon$ | Classic $\sigma/\Delta_2f$ | Analytic $\sigma/\Delta_2f$ | Ratio |
| --- | --- | --- | --- |
| 0.1 | 48.4481 | 30.7496 | 1.576 |
| 0.5 | 9.6896 | 7.0318 | 1.378 |
| 1 | 4.8448 | 3.7306 | 1.299 |
| 2 | 2.4224 | 1.9938 | 1.215 |
| 4 | 1.2112 | 1.0812 | 1.120 |
| 8 | 0.6056 | 0.6002 | 1.009 |

: The noise the classic formula and the analytic calibration ask for at $\delta = 10^{-5}$, in units of the $\ell_2$ sensitivity. {#tbl-lg-calibration}

**Example.** Release the diabetes count at $\varepsilon = 0.1$ and $\delta = 10^{-5}$. The sensitivity is 1, so the analytic calibration gives $\sigma = 30.75$. The 95% error is $1.96\sigma = 60.27$, against 29.96 for Laplace noise with $\delta = 0$. For a single count, the Laplace mechanism wins on both error and guarantee.

**Pitfalls.** Don't use the classic formula when you can solve @eq-lg-analytic; the solve is a few lines of code. A Gaussian mechanism is not "$(\varepsilon, \delta)$-DP" for one pair only: it satisfies a whole curve of pairs at once. At $\sigma = \Delta_2f$ the curve passes through $\delta = 0.1269$ at $\varepsilon = 1$ and reaches $\delta = 10^{-5}$ at $\varepsilon = 4.3772$ (@fig-lg-delta-curves). When you report a Gaussian release, report $\sigma/\Delta_2f$ (or $\mu$) as well as the pair you chose, because the accountants of [Part 4](../part04-privacy-accounting/00-introduction.md) compose the curve, not the pair. And choose $\delta$ much smaller than $1/n$: a mechanism that publishes each record verbatim with probability $\delta$ is $(0, \delta)$-DP, and at $\delta = 1/n$ it exposes someone with probability about 0.632 ([The definition of differential privacy](01-definition.md)).

![The privacy curve $\delta(\varepsilon)$ of the Gaussian mechanism for four noise levels $\sigma/\Delta_2f$, on a log scale, and of the Laplace mechanism with $b = \Delta_1f$ (dashed). Crosses are the same divergence computed by numerical integration. The Laplace curve drops to zero at $\varepsilon = 1$, its pure-DP level; every Gaussian curve stays positive but falls faster than exponentially. The dotted line marks $\delta = 10^{-5}$.](figures/lapgauss_delta_curves.svg){#fig-lg-delta-curves}

## Laplace or Gaussian? {#sec-lg-choose}

**Idea.** Laplace noise scales with $\Delta_1f$ and Gaussian noise with $\Delta_2f$, and the two sensitivities differ by up to a factor of $\sqrt d$. Which noise is better depends on how many coordinates one record can move.

**Result.** Compare the typical error per released number at the same $\varepsilon$. It is $\sqrt2\,\Delta_1f/\varepsilon$ for Laplace noise and $\Delta_2f\,\sigma_1$ for Gaussian noise, where $\sigma_1$ is the analytic $\sigma$ for unit sensitivity at your $(\varepsilon, \delta)$: 3.7306 at $\varepsilon = 1$, $\delta = 10^{-5}$. For a **sparse** query, where a record moves one coordinate (a histogram), the two sensitivities are equal and Laplace noise wins at every $d$: 1.41 against 3.73 at those parameters. For a **dense** query, where a record moves all $d$ coordinates, the Laplace error grows like $d$ and the Gaussian error like $\sqrt d$, so Gaussian noise wins once $d$ exceeds $(\varepsilon\sigma_1)^2/2$, which is 6.96 at those parameters (@fig-lg-high-dim).

**Example.** An empirical distribution function of ages, read off at $d$ thresholds, is a dense query: one person's age moves every threshold count at or below it. @tbl-lg-high-dim gives the error per count at $\varepsilon = 1$ and $\delta = 10^{-5}$. At 100 thresholds Gaussian noise is 3.79 times more accurate, at 1,000 thresholds 11.99 times.

| $d$ | Laplace | Gaussian | Laplace / Gaussian |
| --- | --- | --- | --- |
| 1 | 1.41 | 3.73 | 0.38 |
| 10 | 14.14 | 11.80 | 1.20 |
| 100 | 141.42 | 37.31 | 3.79 |
| 1000 | 1414.21 | 117.97 | 11.99 |

: Root-mean-square error per count for $d$ threshold counts at $\varepsilon = 1$, $\delta = 10^{-5}$. {#tbl-lg-high-dim}

![Root-mean-square error per count at $\varepsilon = 1$ for the Laplace mechanism (calibrated to $\Delta_1$) and the Gaussian mechanism (analytic calibration to $\Delta_2$, $\delta = 10^{-5}$), as the number of released counts $d$ grows; dots are simulations. Left: a histogram, where each record falls in one bin, so neither error depends on $d$ and Laplace noise is smaller. Right: threshold counts, where a record can move all $d$ counts; the Laplace error grows like $d$, the Gaussian like $\sqrt d$, and they cross near $d = 7$.](figures/lapgauss_high_dim.svg){#fig-lg-high-dim}

**Pitfalls.** The comparison is at equal $\varepsilon$, but the Gaussian release also has a $\delta$. Dense queries are why private training of neural networks uses Gaussian noise. One example's gradient moves every coordinate of a model with millions of parameters, the densest query there is, and clipping each gradient bounds its $\ell_2$ norm, which is exactly what the Gaussian mechanism needs.

## Code {#sec-lg-code}

The analytic calibration needs only the normal distribution function, which the standard library provides through `math.erfc`. This is the whole of it:

```python
from math import erfc, exp, sqrt


def normal_cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def gaussian_delta(eps: float, mu: float) -> float:
    """The exact delta of the Gaussian mechanism at eps, where mu = sensitivity / sigma."""
    return normal_cdf(mu / 2 - eps / mu) - exp(eps) * normal_cdf(-mu / 2 - eps / mu)


def analytic_sigma(eps: float, delta: float, sensitivity: float = 1.0) -> float:
    """The smallest sigma for which the Gaussian mechanism is (eps, delta)-DP."""
    lo, hi = 1e-6, 100.0  # gaussian_delta grows with mu: bisect for gaussian_delta(eps, mu) = delta
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gaussian_delta(eps, mid) < delta else (lo, mid)
    return sensitivity / lo


print(f"{analytic_sigma(1.0, 1e-5):.4f}")  # 3.7306
print(f"{analytic_sigma(0.1, 1e-5):.4f}")  # 30.7496
```

It returns $\sigma_1 = 3.7306$ at $\varepsilon = 1$, $\delta = 10^{-5}$, and 30.7496 at $\varepsilon = 0.1$, the analytic column of @tbl-lg-calibration. For the Laplace mechanism there is nothing to solve: the scale is $\Delta_1f/\varepsilon$. In production, take the noise itself from a DP library that samples it safely, and use this calculation to check the scale it reports.

The full edition's script, [laplace_gaussian.py](../../code/part03-dp-foundations/laplace_gaussian.py), computes every number in this chapter and checks each mechanism by simulation:

```bash
uv run python code/part03-dp-foundations/laplace_gaussian.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| Sensitivity | $\Delta_pf = \max_{D \simeq D'}\lVert f(D) - f(D')\rVert_p$ | $\Delta_2 \le \Delta_1 \le \sqrt d\,\Delta_2$; depends on the neighbouring relation |
| Count, sum in $[0, B]$, mean with public $n$ | $1$; $B$; $B/n$ | Add/remove; either; replace |
| Laplace mechanism | $\mathrm{Lap}(\Delta_1f/\varepsilon)$ per coordinate: $\varepsilon$-DP | All $d$ errors below $b\ln(d/\beta)$ with probability $1 - \beta$ |
| Gaussian mechanism, exact | $\delta_\mu(\varepsilon) = \Phi(\frac\mu2 - \frac\varepsilon\mu) - e^\varepsilon\Phi(-\frac\mu2 - \frac\varepsilon\mu)$, $\mu = \Delta_2f/\sigma$ | Solve for the smallest $\sigma$ by bisection |
| Gaussian mechanism, classic | $\sigma = \Delta_2f\sqrt{2\ln(1.25/\delta)}/\varepsilon$ | Only for $\varepsilon < 1$; 1.3–1.7 times too much noise |
| Error per coordinate | Laplace $\sqrt2\,\Delta_1f/\varepsilon$; Gaussian $\Delta_2f\,\sigma_1(\varepsilon, \delta)$ | Dense queries: Gaussian wins once $d > (\varepsilon\sigma_1)^2/2$ |

: The results of this chapter. {#tbl-lg-cheat}

**Rules of thumb**

- Compute the sensitivity first, from the domain rather than the data, and state the neighbouring relation with every $\varepsilon$.
- One number or a sparse vector: Laplace noise, pure DP. A dense vector: Gaussian noise, calibrated to $\Delta_2f$.
- Calibrate Gaussian noise exactly. At $\varepsilon = 1$, $\delta = 10^{-5}$ it needs $\sigma \approx 3.73\,\Delta_2f$.
- Report $\sigma/\Delta_2f$ for a Gaussian release, not just one $(\varepsilon, \delta)$ pair.
- Sample the noise with a vetted DP library, never with a plain floating-point sampler.

## Further reading

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), §3.3 and Appendix A: the Laplace mechanism and the classic Gaussian calibration.
- Balle & Wang, "Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising" (ICML 2018): the exact privacy curve and the analytic calibration.
- Near & Abuah, *Programming Differential Privacy*, the chapters on sensitivity and on the Laplace and Gaussian mechanisms: the same material with code.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/03-laplace-gaussian.md).
