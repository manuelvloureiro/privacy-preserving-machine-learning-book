# Lower bounds: what privacy costs

## Why this matters

Someone hands you a design for a private release: a histogram over a million locations with a maximum error of five records at $\varepsilon = 1$, say. Before reading their code, you can check whether that accuracy is possible at all. The earlier chapters gave **upper bounds**, what one mechanism achieves. A **lower bound** says that *no* mechanism satisfying differential privacy (DP) can do better. When it matches a known mechanism, that mechanism is optimal, and more accuracy can only come from more data or less privacy. When a claim beats a lower bound, the claim is wrong.

The one idea is that **accuracy is an attack**. An accurate output reveals something about the data: which of many datasets was used, whether a person is in it, what their secret bits are. DP caps every attack. If an accurate mechanism would hand an attacker more than DP allows, it can't be both. @tbl-lower-overview maps the questions to the sections.

| Question | Answer | Section |
| --- | --- | --- |
| How much error must a pure-DP selection or histogram have? | About $\ln(\text{number of candidates})/\varepsilon$; Laplace is within a factor of about 2 | @sec-lower-packing |
| How many records does a private release need? | Privacy adds a term; pure DP pays linearly in the dimension $d$, approximate DP about $\sqrt d$ | @sec-lower-sample |
| Are released aggregates safe without noise? | No: tracing identifies members once $d$ is large next to $n^2$ | @sec-lower-tracing |
| How much noise must answers to many counting queries carry? | Of order $\sqrt n$, or reconstruction recovers the data | @sec-lower-dinur-nissim |
| Is a claimed accuracy possible? | Compare it with the lower bounds | @sec-lower-code |

: The questions lower bounds answer, and where this chapter answers each one. {#tbl-lower-overview}

Throughout, $n$ is the number of records, the error is at most $\alpha$ with probability at least $1 - \beta$, and the examples use $\varepsilon = 1$, $\beta = 0.05$ and, for approximate DP, $\delta = 10^{-6}$.

## Pure DP pays for the number of candidates {#sec-lower-packing}

**Idea.** Suppose the output must point to one of $m$ candidates, such as the most common of a million ZIP codes. For each candidate there is a dataset on which a correct answer points to it. Run the mechanism on one reference dataset instead, say the empty one: its output probabilities add up to 1, so most candidates get a small share. But pure DP extended to datasets $k$ records apart (**group privacy**, [Post-processing, group privacy and composition](02-properties.md)) says each candidate needs a share of about $e^{-k\varepsilon}$. A budget of 1 can't cover $m$ such demands unless $k \gtrsim \ln m/\varepsilon$. This counting argument is **packing**.

**Result.** If an $\varepsilon$-DP mechanism must land in $m$ disjoint regions, each with probability at least $1 - \beta$ on its own dataset, and every such dataset is within $k$ records of one reference dataset, then
$$k \ge \frac{\ln\big(m(1 - \beta)\big)}{\varepsilon}.$$ {#eq-lower-packing}
Under add/remove neighbours, reporting the **mode** of $N$ elements reliably needs the top element to lead by $k \ge \ln(N(1 - \beta))/\varepsilon$ records, and any $\varepsilon$-DP histogram over $N$ bins has, with probability $\beta$, an error in some bin of at least
$$\alpha \ge \frac12\left(\frac{\ln\big(N(1 - \beta)\big)}{\varepsilon} - 1\right).$$ {#eq-lower-histogram}
The Laplace histogram's error is at most $\ln(N/\beta)/\varepsilon$, and report-noisy-max ([Selection](05-selection.md)) finds the mode at a lead of the same order: both are optimal within a factor of about 2 (@fig-lower-histogram).

**Example.** For $N = 10^6$ locations (@tbl-lower-histogram), the most common one must lead by at least 13.76 records before any $\varepsilon$-DP mechanism can report it reliably; report-noisy-max needs 18.44. Every $\varepsilon$-DP histogram has, at the 95% level, an error of at least 6.38 in some bin; Laplace has 16.79.

| $N$ | Mode: lower bound on lead $k$ | Report-noisy-max needs | Histogram: lower bound on error | Laplace error (exact) | Laplace error (union bound) |
| --- | --- | --- | --- | --- | --- |
| $10^3$ | 6.86 | 11.09 | 2.93 | 9.88 | 9.90 |
| $10^6$ | 13.76 | 18.44 | 6.38 | 16.79 | 16.81 |
| $10^9$ | 20.67 | 25.65 | 9.84 | 23.69 | 23.72 |

: Packing lower bounds against the mechanisms that achieve them, $\varepsilon = 1$, $\beta = 0.05$, add/remove neighbours. Errors and leads are in records. {#tbl-lower-histogram}

![$\ell_\infty$ error of $\varepsilon$-DP histograms over $N$ bins at $\varepsilon = 1$, failure probability 0.05. Orange: the Laplace mechanism, exact (line), union bound (dotted, indistinguishable) and simulated (dots). Blue: the packing lower bound; no $\varepsilon$-DP mechanism has an error in the shaded region. Both grow like $\ln N$.](figures/lower_histogram.svg){#fig-lower-histogram}

**Pitfalls.** Packing holds only for pure DP. With $\delta > 0$, group privacy breaks down past about $\ln(1/\delta)/\varepsilon$ records (13.82 at $\delta = 10^{-6}$), and the bound says nothing more. That is real: the stability-based histogram (Vadhan 2017) releases only non-empty bins whose noisy count clears a threshold of order $\ln(1/\delta)/\varepsilon$, so its error doesn't grow with $N$. For a huge, mostly empty universe, approximate DP is how you escape the $\ln N$.

## How many records privacy costs {#sec-lower-sample}

**Idea.** An error of 6 records is large in a dataset of 50 and negligible in a million. Dividing by $n$ turns a lower bound on error into a lower bound on records, which you can set next to the records you need anyway for sampling error. With many attributes, the number of candidates grows exponentially, and pure and approximate DP part ways.

**Result.** An $\varepsilon$-DP histogram with every frequency within $\alpha$ needs
$$n \ge \frac{\ln\big(N(1 - \beta)\big) - \varepsilon}{2\alpha\varepsilon},$$
while sampling error alone needs up to $\ln(2N/\beta)/(2\alpha^2)$. Privacy adds a term that dominates only when $\alpha \gtrsim \varepsilon$: coarse answers at strong privacy.

For the means of $d$ yes/no attributes, coded $\pm1$, each within $\alpha$, pure DP has $2^d$ candidates, so $n \gtrsim d\ln 2/(\alpha\varepsilon)$: linear in $d$. The Gaussian mechanism, at $(\varepsilon, \delta)$, needs only $2\sqrt d\,z/(\mu\alpha)$, where $z$ is the normal quantile with tail $\beta/(2d)$ and $\mu$ is the largest ratio of sensitivity to noise the exact calibration allows (0.2367 at $\varepsilon = 1$, $\delta = 10^{-6}$; [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). Approximate DP has its own lower bound, from **fingerprinting** (Bun, Ullman & Vadhan 2014): $n$ of order $\sqrt{d\ln(1/\delta)}/(\alpha\varepsilon)$ (Steinke & Ullman 2016). So the Gaussian mechanism is optimal up to a logarithm, and $\sqrt d$ is the unavoidable price.

**Example.** A histogram over $10^6$ cells to within $\alpha = 0.1$ needs at least 64 records under $\varepsilon$-DP and at most 875 for sampling; at $\alpha = 0.01$, 638 and 87,522. Privacy is the smaller cost. For attribute means (@tbl-lower-sample), the packing column grows tenfold with each tenfold increase in $d$, the Gaussian column only about $\sqrt{10}$-fold. From $d = 2731$ attributes on, the Gaussian mechanism needs fewer records than *any* pure-DP mechanism (@fig-lower-sample-complexity). This is one reason private machine learning, with thousands of parameters or more, uses the Gaussian mechanism.

| | Pure DP, packing (lower) | Pure DP, Laplace (upper) | Approximate DP, Gaussian (upper) |
| --- | --- | --- | --- |
| $d = 100$ | 683 | 15,202 | 2,941 |
| $d = 1000$ | 6,921 | 198,070 | 10,836 |
| $d = 10{,}000$ | 69,304 | 2,441,215 | 38,570 |
| $d = 100{,}000$ | 693,137 | 29,017,315 | 134,299 |

: Records needed to release $d$ attribute means with $\ell_\infty$ error 0.1 (on the scale $[-1, 1]$) with probability 0.95, at $\varepsilon = 1$ and, for the Gaussian, $\delta = 10^{-6}$. {#tbl-lower-sample}

![Records needed to release $d$ attribute means to $\ell_\infty$ error 0.1 with probability 0.95, at $\varepsilon = 1$ and $\delta = 10^{-6}$, on log–log axes. Orange: pure DP (Laplace upper bound, packing lower bound), both of slope 1. Blue: approximate DP (Gaussian upper bound, fingerprinting lower bound), both of slope ½. The constants of the fingerprinting bound are loose; the slopes are the point.](figures/lower_sample_complexity.svg){#fig-lower-sample-complexity}

**Pitfalls.** Use the fingerprinting bound for its rate, not its numbers: in the simple form the full edition proves, it says $n \ge 5.0$ at $d = 1000$, against 10,836 for the Gaussian mechanism. And keep $\delta$ far below $1/n$: publishing one random record with probability $\delta$ is $(0, \delta)$-DP. And "privacy is only an additive term" fails for small subgroups, where $n$ is the subgroup's size.

## The lower bound is a membership inference attack {#sec-lower-tracing}

**Idea.** In 2008 Homer et al. showed that from the published allele frequencies of a genome-wide association study you could tell whether a given person's DNA was in it. Their test is, up to details, the score in the fingerprinting proof. Each member pulls the released means slightly toward their own record, so a member agrees with the release more than a stranger does. This is the attack of [Membership inference](../part02-attacks/02-membership-inference.md) on statistics instead of models, and it survives noisy means and a population known only from a reference sample (Dwork et al. 2015).

**Result.** Given the release $\hat q$ of $d$ means, with Gaussian noise $\sigma$ per coordinate, the population means $p$ and a target record $y$, the attacker computes $Z(y) = \langle\hat q, y - p\rangle$ and says "member" when it is large. With attribute means drawn uniformly, the members' mean score exceeds the non-members' by a gap, and non-member scores have a variance, of
$$\text{gap} = \frac{2d}{3n}, \qquad \text{variance} = d\left(\frac2{15} + \frac{8}{15n} + \frac{2\sigma^2}3\right).$$ {#eq-lower-tracing}
For roughly normal scores with gap $m$ and standard deviation $s$, the best threshold has advantage $2\Phi\big(m/(2s)\big) - 1$, the **true positive rate** (TPR) on members minus the **false positive rate** (FPR) on non-members, with $\Phi$ the standard normal cumulative distribution function. Without noise the attack works once $d$ is large next to $n^2$; noise of order $\sqrt d/n$, which is what the Gaussian mechanism adds, stops it. Whatever the attack, $(\varepsilon, \delta)$-DP caps it:
$$\mathrm{TPR} - \mathrm{FPR} \le \frac{e^\varepsilon - 1 + 2\delta}{e^\varepsilon + 1}, \qquad \mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta.$$
At $\varepsilon = 1$, $\delta = 0$ the advantage is at most 0.4621.

**Example.** The full edition releases the means of $n = 50$ records with $d = 2000$ attributes. Without noise, @eq-lower-tracing predicts a gap of 26.67 and a spread of 16.97 (simulated: 26.70 and 16.92), and an advantage of 0.568. A quarter of the members are identified at a 1% false-alarm rate (@tbl-lower-tracing). With the noise of a $(1, 10^{-5})$-DP Gaussian mechanism, $\sigma = 6.67$, the TPR at 1% FPR drops to 0.014, under the cap $e^1 \times 0.01 + 10^{-5} = 0.0272$. The attack fades around $\sigma = \sqrt d/n = 0.89$ (@fig-lower-tracing-advantage).

| Noise $\sigma$ | Advantage (simulated) | Advantage (normal approximation) | Worst-case bound | TPR at 1% FPR |
| --- | --- | --- | --- | --- |
| 0 | 0.578 | 0.568 | 1.000 | 0.242 |
| 1.00 | 0.264 | 0.259 | 0.629 | 0.052 |
| 6.67 (DP, $\varepsilon = 1$) | 0.049 | 0.044 | 0.107 | 0.014 |

: The tracing attack on released attribute means, $n = 50$, $d = 2000$. The advantage is the best $\mathrm{TPR} - \mathrm{FPR}$ over thresholds; the worst-case column is the total variation distance between the Gaussian outputs on two neighbouring datasets, which bounds every attack. {#tbl-lower-tracing}

![Tracing scores of members (blue) and non-members (orange) for $n = 50$ records and $d = 2000$ attributes. Left: exact means; the members' scores are shifted by $2d/(3n) = 26.7$. Right: means released with the Gaussian noise of a $(1, 10^{-5})$-DP mechanism; the two distributions nearly coincide.](figures/lower_tracing_scores.svg){#fig-lower-tracing-scores}

![Advantage of the tracing attack against the noise standard deviation, $n = 50$, $d = 2000$. Dots: simulation; solid line: the normal approximation; dashed: the worst-case bound on every attack. The attack fails once $\sigma$ is well above $\sqrt d/n$; the DP Gaussian mechanism adds $\sigma = 6.7$.](figures/lower_tracing_advantage.svg){#fig-lower-tracing-advantage}

**Pitfalls.** Aggregates are not safe because they are aggregates: without noise, tracing reaches advantage 0.5 once $d \approx 0.56\,n^2$. Report the attack as TPR at a low FPR, since identifying a few people with confidence is already a failure. When auditing with this score, take $p$ from a separate reference sample; using the release itself subtracts the member's own contribution and cancels part of the signal.

## Reconstruction: why answers must carry $\sqrt n$ noise {#sec-lower-dinur-nissim}

**Idea.** [Reconstruction attacks](../part02-attacks/09-reconstruction.md) recover almost all of $n$ secret bits from about $n$ random subset counts answered with error growing more slowly than $\sqrt n$ (Dinur & Nissim 2003). DP limits how well any single bit can be guessed, so such a release can't be private: answers to that many counting queries need noise of order $\sqrt n$.

**Result.** With uniform secret bits, an $(\varepsilon, \delta)$-DP release lets any attack recover on average a fraction of at most $(e^\varepsilon + \delta)/(e^\varepsilon + 1)$ of them, 73.11% at $\varepsilon = 1$. Turned around, recovering a fraction $r$ proves
$$\varepsilon \ge \ln\frac{r - \delta}{1 - r}.$$ {#eq-lower-dn}
The Gaussian mechanism matches: $k$ counting queries have $\ell_2$ sensitivity at most $\sqrt k$, so it adds noise of order $\sqrt k$ per answer. The Laplace mechanism, calibrated to $\ell_1$ sensitivity $k$, pays of order $k$.

**Example.** The full edition answers 800 random subset sums of 200 bits with Gaussian noise and decodes by least squares (@tbl-lower-dn). Recovering 0.9607 of the bits proves $\varepsilon \ge 3.20$. At $\varepsilon = 1$ at most 73.11% may be recovered; the attack gets 0.8058 at noise $0.5\sqrt n$ and 0.6710 at $\sqrt n$, so an $\varepsilon = 1$ mechanism needs noise between the two.

| $\sigma/\sqrt n$ | Fraction recovered | $\varepsilon$ at least (attack) | $\varepsilon$ at most (Gaussian mechanism) |
| --- | --- | --- | --- |
| 0.1 | 1.0000 | $\infty$ | 294.17 |
| 0.25 | 0.9607 | 3.20 | 69.24 |
| 0.5 | 0.8058 | 1.42 | 26.36 |
| 1 | 0.6710 | 0.71 | 11.00 |
| 2 | 0.5810 | 0.33 | 4.89 |

: Reconstruction from 800 noisy subset sums of 200 bits, and the privacy loss it proves, at $\delta = 10^{-6}$. {#tbl-lower-dn}

![Left: fraction of 200 secret bits recovered by least squares from 800 subset sums with Gaussian noise of standard deviation $\sigma$, against $\sigma/\sqrt n$. The dotted line is the most that any $\varepsilon = 1$ mechanism allows. Right: the $\varepsilon$ that the reconstruction proves (blue) and the $\varepsilon$ that the Gaussian mechanism guarantees (orange), at $\delta = 10^{-6}$; the true privacy loss of the release lies between them.](figures/lower_dinur_nissim.svg){#fig-lower-dinur-nissim}

**Pitfalls.** An attack proves only a lower bound on $\varepsilon$. Recovering 67% of the bits at $\sigma = \sqrt n$ proves $\varepsilon \ge 0.71$; it doesn't make the release $(0.71, 10^{-6})$-DP, and the mechanism's analysis guarantees only $\varepsilon \le 11.00$. Stronger attacks narrow that gap, which is what privacy auditing does.

## Code {#sec-lower-code}

Checking a claimed accuracy is a few lines of arithmetic: the pure-DP histogram floor and the Laplace error, the records needed for $d$ means under pure DP and with the Gaussian mechanism, and the $\varepsilon$ a reconstruction proves.

```python
from math import exp, log, sqrt
from statistics import NormalDist

PHI = NormalDist().cdf


def histogram_error_floor(bins: int, eps: float, beta: float) -> float:
    """Smallest max-bin error any eps-DP histogram can have with probability 1 - beta (packing)."""
    return max(0.0, (log(bins * (1 - beta)) / eps - 1) / 2)


def laplace_histogram_error(bins: int, eps: float, beta: float) -> float:
    """Max-bin error the Laplace histogram reaches with probability 1 - beta (exact)."""
    return -log(1 - (1 - beta) ** (1 / bins)) / eps


def means_records_floor_pure(d: int, eps: float, alpha: float, beta: float) -> float:
    """Records any eps-DP release of d means in [-1, 1] needs for max error alpha (packing)."""
    return ((d * log(2) + log(1 - 2.0**-d) + log(1 - beta)) / eps - 1) / alpha


def gaussian_mu(eps: float, delta: float) -> float:
    """Largest sensitivity / sigma for which the Gaussian mechanism is (eps, delta)-DP."""
    lo, hi = 1e-6, 100.0
    for _ in range(200):
        mu = (lo + hi) / 2
        d = PHI(mu / 2 - eps / mu) - exp(eps) * PHI(-mu / 2 - eps / mu)
        lo, hi = (mu, hi) if d < delta else (lo, mu)
    return lo


def means_records_gaussian(d: int, eps: float, delta: float, alpha: float, beta: float) -> float:
    """Records the Gaussian mechanism needs for the same release (an upper bound)."""
    z = NormalDist().inv_cdf(1 - beta / (2 * d))
    return 2 * sqrt(d) * z / (gaussian_mu(eps, delta) * alpha)


def epsilon_proved(recovered: float, delta: float) -> float:
    """Lower bound on epsilon from an attack that recovers this fraction of uniform secret bits."""
    return log((recovered - delta) / (1 - recovered))


print(f"{histogram_error_floor(10**6, 1.0, 0.05):.2f} {laplace_histogram_error(10**6, 1.0, 0.05):.2f}")  # 6.38 16.79
print(f"{means_records_floor_pure(1000, 1.0, 0.1, 0.05):.0f} {means_records_gaussian(1000, 1.0, 1e-6, 0.1, 0.05):.0f}")  # 6921 10836
print(f"{epsilon_proved(0.9607, 1e-6):.2f}")  # 3.20
```

Back to the opening claim: a pure $\varepsilon = 1$ histogram over $10^6$ locations with maximum error 5 at 95% confidence is impossible, since no mechanism goes below 6.38. Unless the design uses $\delta > 0$ and exploits sparsity, ask what it is really doing. The full edition's script, [lower_bounds.py](../../code/part03-dp-foundations/lower_bounds.py), computes every number in this chapter and runs the attacks:

```bash
uv run python code/part03-dp-foundations/lower_bounds.py
```

## Cheat sheet

| Result | Statement | Matching mechanism |
| --- | --- | --- |
| Packing (pure DP) | $m$ candidates within $k$ records of one dataset: $k \ge \ln(m(1 - \beta))/\varepsilon$ | — |
| Mode over $N$ elements (pure) | Lead $k \ge \ln(N(1 - \beta))/\varepsilon$ | Report-noisy-max |
| Histogram, $\ell_\infty$ (pure) | $\alpha \ge \frac12(\ln(N(1 - \beta))/\varepsilon - 1)$ | Laplace: $\alpha \le \ln(N/\beta)/\varepsilon$ |
| Packing under $\delta > 0$ | Forces at most $k \approx \ln(1/\delta)/\varepsilon$ | Stability-based histogram |
| Histogram, records | $n \ge (\ln(N(1 - \beta)) - \varepsilon)/(2\alpha\varepsilon)$; sampling needs $\approx\ln(2N/\beta)/(2\alpha^2)$ | Laplace |
| $d$ means (pure) | $n \gtrsim d\ln2/(\alpha\varepsilon)$ | Laplace: $n = 2d\ln(d/\beta)/(\alpha\varepsilon)$ |
| $d$ means (approximate) | $n$ of order $\sqrt{d\ln(1/\delta)}/(\alpha\varepsilon)$ (fingerprinting) | Gaussian: $n = 2\sqrt d\,z/(\mu\alpha)$ |
| Tracing score | Gap $2d/(3n)$; variance $d(\frac2{15} + \frac8{15n} + \frac{2\sigma^2}3)$ | Defeated by $\sigma \gg \sqrt d/n$ |
| Any membership test | $\mathrm{TPR} - \mathrm{FPR} \le \frac{e^\varepsilon - 1 + 2\delta}{e^\varepsilon + 1}$; $\mathrm{TPR} \le e^\varepsilon\mathrm{FPR} + \delta$ | — |
| Reconstruction | Recovering a fraction $r$ proves $\varepsilon \ge \ln\frac{r - \delta}{1 - r}$ | Gaussian with $\sigma \propto \sqrt k$ |

: The lower bounds of this chapter, with the mechanism that matches each. {#tbl-lower-cheat}

**Rules of thumb**

- Pure DP pays $\ln(\text{number of candidates})/\varepsilon$ in error; for $d$-dimensional outputs that is linear in $d$.
- Approximate DP pays about $\sqrt d$ for $d$ statistics, with $\delta$ far below $1/n$.
- Answering $k$ counting queries costs error about $\sqrt k$ per query with $\delta > 0$, and about $k$ without.
- Released aggregates over many attributes leak membership; noise of order $\sqrt d/n$ stops it.
- An attack bounds $\varepsilon$ from below; only the mechanism's analysis bounds it from above.

## Further reading

- Vadhan, "The Complexity of Differential Privacy" (2017), §5: packing, fingerprinting and reconstruction lower bounds side by side, and the stability-based histogram.
- Dwork, Smith, Steinke & Ullman, "Exposed! A Survey of Attacks on Private Data" (Annual Review of Statistics and Its Application, 2017): reconstruction and tracing attacks for a statistics audience.
- Steinke & Ullman, "Between Pure and Approximate Differential Privacy" (Journal of Privacy and Confidentiality, 2016): the tight $\sqrt{d\ln(1/\delta)}$ bound for attribute means.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/09-lower-bounds.md).
