# Concentration inequalities

## Why this matters

A noisy answer, a measured error rate, the size of a sampled minibatch: each is random. The practical question is **how wide an error bar must be to fail with probability at most $\beta$**, and concentration inequalities answer it.

The one idea is that the more you know about a random quantity, the narrower the bar. Knowing only the variance, the width grows like $1/\sqrt\beta$; with a normal-shaped tail, only like $\sqrt{\ln(1/\beta)}$; with an exponential tail, like $\ln(1/\beta)$. When $\beta$ must be tiny, as in the later parts, that difference is everything. @tbl-conc-overview maps each case to its section.

| You need a width for | Use | Section |
| --- | --- | --- |
| Anything with a known variance, at a modest $\beta$ | Chebyshev | @sec-conc-chebyshev |
| One draw of normal or Laplace noise, or the largest of $k$ draws | Exact tails and the union bound | @sec-conc-noise |
| A count of independent events | Chernoff | @sec-conc-chernoff |
| An average of bounded quantities | Hoeffding, or Bernstein when the variance is small | @sec-conc-hoeffding |
| A function of many inputs, or adaptively chosen steps | McDiarmid, Azuma | @sec-conc-mcdiarmid |

: Which inequality gives the width you need, and where it is stated. {#tbl-conc-overview}

## Variance alone: Markov and Chebyshev {#sec-conc-chebyshev}

**Idea.** A non-negative variable cannot often be much larger than its mean, or those values alone would pull the mean up. Applied to the squared deviation, this bounds deviations by the variance.

**Result.** Markov: for $X \ge 0$, $\Pr[X \ge a] \le \mathbb E[X]/a$. Chebyshev: for $X$ with mean $\mu$ and variance $\sigma^2$, $\Pr[|X - \mu| \ge t] \le \sigma^2/t^2$. Setting the bound equal to $\beta$ and solving for $t$ gives the width: $\sigma/\sqrt\beta$. For a mean of $n$ independent samples the variance is $\sigma^2/n$, so the width is $\sigma/\sqrt{n\beta}$.

**Example.** @tbl-width-price compares the variance-only width with the width under a normal-shaped tail, $\sigma\sqrt{2\ln(1/\beta)}$. For Laplace noise with $b = 1$, Chebyshev gives a 95% width of about 6.3, while the exact tail gives $\ln 20 \approx 3.0$; at $\beta = 10^{-6}$ the two are 1,414 and 13.8.

| $\beta$ | Chebyshev, $1/\sqrt\beta$ | Normal-shaped tail, $\sqrt{2\ln(1/\beta)}$ |
| --- | --- | --- |
| $0.05$ | 4.5 | 2.4 |
| $10^{-3}$ | 31.6 | 3.7 |
| $10^{-6}$ | 1,000 | 5.3 |

: Width needed to fail with probability at most $\beta$, in units of $\sigma$. {#tbl-width-price}

**Pitfalls.** Variance-only bounds are fine for a quick 95% error bar and useless at $10^{-5}$. If you know the shape of the noise, use it.

## Normal and Laplace noise {#sec-conc-noise}

**Idea.** For the two standard noise distributions the tail is known, so the width comes straight from it. To control $k$ draws at once, ask each to fail with probability $\beta/k$: by the **union bound**, the chance that any fails is at most the sum, $\beta$.

**Result.** For $X \sim \mathcal N(0, \sigma^2)$, $\Pr[X \ge t] \le \frac12e^{-t^2/(2\sigma^2)}$, and far out the sharper **Mills ratio** bound $\Pr[Z \ge t] \le \varphi(t)/t$ holds for a standard normal $Z$ with density $\varphi$. For $X \sim \mathrm{Lap}(b)$ the tail is exact, $\Pr[|X| \ge t] = e^{-t/b}$. With probability at least $1 - \beta$:

- one normal draw satisfies $|X| < \sigma\sqrt{2\ln(1/\beta)}$, and all of $k$ draws satisfy $|X_i| < \sigma\sqrt{2\ln(k/\beta)}$;
- one Laplace draw satisfies $|X| \le b\ln(1/\beta)$, and all of $k$ draws satisfy $|X_i| \le b\ln(k/\beta)$.

**Example.** For the maximum of 100 $\mathrm{Lap}(1)$ draws at $\beta = 0.05$, the width is $\ln(100/0.05) = 7.60$, and in simulation it fails at rate 0.0486: for independent draws with small tails the union bound is nearly exact (@fig-laplace-max). For a standard normal at $t = 3$, the exact tail is $1.35\times10^{-3}$, the Mills bound 0.00148 and the plain exponential bound 0.0111.

![Left: the width $b\ln(k/\beta)$ for the maximum of $k$ Laplace draws grows like $\ln k$. Right: the probability that the maximum exceeds it stays at $\beta = 0.05$ for every $k$.](figures/laplace_max.svg){#fig-laplace-max}

**Pitfalls.** A sum of $k$ Laplace draws is not normal in its tail. Its bulk looks Gaussian, but large deviations decay only exponentially, and a valid bound is $\Pr[S \ge t] \le \exp(-\min(t^2/(8kb^2), t/(4b)))$. For $k = 10$, $b = 1$, the normal approximation puts $\Pr[S \ge 20]$ at $3.9\times10^{-6}$; the truth is $3.4\times10^{-5}$, nine times more (@fig-laplace-sum). When the answer must be a guarantee, use a bound, not the CLT (central limit theorem).

![Empirical tail of a sum of 10 $\mathrm{Lap}(1)$ draws, against the sub-exponential bound and the normal approximation, which badly underestimates the far tail.](figures/laplace_sum.svg){#fig-laplace-sum}

## Counts: the Chernoff bound {#sec-conc-chernoff}

**Idea.** Apply Markov to $e^{\lambda X}$ instead of $X$ and pick the best $\lambda$. The average of $e^{\lambda X}$ feels every moment, so the bound decays exponentially. For counts of independent yes/no events it depends only on the expected count.

**Result.** Let $S$ be a sum of independent Bernoulli variables with mean $\mu$. For $\delta > 0$, and for $0 < \delta < 1$ in the second bound,
$$\Pr[S \ge (1+\delta)\mu] \le \exp\Big(-\frac{\delta^2\mu}{2+\delta}\Big), \qquad \Pr[S \le (1-\delta)\mu] \le \exp\Big(-\frac{\delta^2\mu}{2}\Big).$$

**Example.** Each of 60,000 examples joins a minibatch independently with probability 0.01, so $\mu = 600$. The batch is more than 20% too large with probability at most $e^{-10.9} \approx 1.8\times10^{-5}$, and more than 20% too small with probability at most $e^{-12} \approx 6.1\times10^{-6}$. Over 1,000 steps the union bound gives about 0.024 for any batch leaving $600 \pm 20\%$.

**Pitfalls.** The bound depends on $\mu$, not on the number of trials: a million trials at $p = 10^{-6}$ expect one event and do not concentrate. Counts overshoot more easily than they undershoot: for large $\delta$ the upper tail decays like $e^{-\delta\mu}$, not $e^{-\delta^2\mu}$.

## Averages of bounded quantities: Hoeffding and Bernstein {#sec-conc-hoeffding}

**Idea.** A variable confined to $[a, b]$ cannot have a heavy tail, so an average of $n$ of them concentrates like a normal with standard deviation $(b - a)/(2\sqrt n)$. It needs only the range, and so assumes the largest variance the range allows.

**Result.** Hoeffding: for the mean $\bar X_n$ of $n$ independent variables in $[a, b]$, $\Pr[\bar X_n - \mu \ge t] \le \exp(-2nt^2/(b-a)^2)$, so with probability at least $1 - \beta$,
$$|\bar X_n - \mu| < (b - a)\sqrt{\frac{\ln(2/\beta)}{2n}}.$$
For a finite set $H$ of models with losses in $[0, 1]$, the union bound puts $|H|$ inside the logarithm: every empirical risk is within $\sqrt{\ln(2|H|/\beta)/(2n)}$ of its true risk. When the variance $\sigma^2$ is much smaller than $(b - a)^2/4$, use **Bernstein's inequality**, whose width for $|X_i - \mu| \le M$ is about $\sqrt{2\sigma^2\ln(1/\beta)/n} + 2M\ln(1/\beta)/(3n)$.

**Example.** A classifier makes 1 false positive on 1,000 negatives, and you want a 95% upper bound on its true rate. One-sided Hoeffding adds about 0.039, some 40 times the measured rate. Bernstein, with the variance estimated as 0.001, adds about 0.0045. The exact binomial (Clopper–Pearson) bound is 0.0047; [Hypothesis testing](04-hypothesis-testing.md) shows how to compute it.

**Pitfalls.** Never use Hoeffding for a small rate. For the mean of 200 Bernoulli(0.1) variables reaching 0.15, Hoeffding gives 0.37, losing even to Chebyshev's 0.18; Chernoff gives 0.135 (@fig-bernoulli-tails). The Bernstein figure is a heuristic, since the variance came from the same data.

![Empirical tail of the mean of 200 Bernoulli(0.1) variables against the Markov, Chebyshev, Hoeffding and Chernoff bounds.](figures/bernoulli_tails.svg){#fig-bernoulli-tails}

## Functions of many inputs: McDiarmid and Azuma {#sec-conc-mcdiarmid}

**Idea.** A trained model's risk is a function of its data, not a sum, and in sequential procedures each step depends on the past. Concentration still holds if no single input or step can move the result much.

**Result.** McDiarmid: if $X_1, \dots, X_n$ are independent and changing only $X_i$ moves $f$ by at most $c_i$, then $\Pr[f(X) - \mathbb E f(X) \ge t] \le \exp(-2t^2/\sum_ic_i^2)$. Azuma gives the same bound for a running total whose steps have mean zero given the past and lie in intervals of fixed lengths $d_k$, with $\sum_kd_k^2$ in place of $\sum_ic_i^2$.

**Example.** An empirical risk with losses in $[0, 1]$ moves by at most $1/n$ when one example changes, so McDiarmid gives $\exp(-2nt^2)$, Hoeffding again. In a game of 200 coin flips where each coin is chosen from the previous flip, Azuma bounds a drift of 20 heads by $1.83\times10^{-2}$; simulation gives $1.12\times10^{-3}$.

**Pitfalls.** The constant $c_i$, a per-record influence, must hold for every possible change of $X_i$, not just typical ones. Both bounds pay for the worst-case variance, like Hoeffding.

## Cheat sheet

| Bound | Assumption | Width at failure probability $\beta$ |
| --- | --- | --- |
| Markov | $X \ge 0$, mean $\mu$ | $\mu/\beta$ |
| Chebyshev | Variance $\sigma^2$ | $\sigma/\sqrt\beta$ (two-sided) |
| Normal $\mathcal N(0, \sigma^2)$ | — | $\sigma\sqrt{2\ln(1/\beta)}$; $k$ draws: $\sigma\sqrt{2\ln(k/\beta)}$ |
| Laplace $\mathrm{Lap}(b)$ | — | $b\ln(1/\beta)$; $k$ draws: $b\ln(k/\beta)$ |
| Chernoff | Sum of independent Bernoullis, mean $\mu$ | $\approx\sqrt{2\mu\ln(1/\beta)}$ for small deviations |
| Hoeffding | Mean of $n$ independent variables in $[a, b]$ | $(b-a)\sqrt{\ln(1/\beta)/(2n)}$ (one-sided) |
| McDiarmid, Azuma | Bounded differences $c_i$ or step lengths $d_i$ | $\sqrt{\sum_i c_i^2\ln(1/\beta)/2}$ |

: The widths of this chapter's bounds. {#tbl-concentration-cheat-sheet}

**Rules of thumb**

- Variance-only widths cost $1/\sqrt\beta$, normal-shaped tails $\sqrt{\ln(1/\beta)}$, exponential tails $\ln(1/\beta)$.
- Controlling $k$ quantities at once adds $\ln k$ inside the logarithm.
- If the variance is far below what the range allows, use Chernoff, Bernstein or an exact interval, not Hoeffding.
- The CLT is not a tail bound.

## Further reading

- Mitzenmacher & Upfal, *Probability and Computing*, ch. 4: the clearest Chernoff chapter, with computer-science examples.
- Vershynin, *High-Dimensional Probability* (2018), ch. 2: sub-Gaussian and sub-exponential variables in depth.
- Boucheron, Lugosi & Massart, *Concentration Inequalities* (2013), ch. 2 and 6: reference depth, including Bernstein's inequality.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/02-concentration.md).
