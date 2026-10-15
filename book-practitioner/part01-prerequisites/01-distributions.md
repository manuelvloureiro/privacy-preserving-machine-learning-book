# Probability distributions

## Why this matters

A handful of distributions do almost all the work in this book. Counts of events, such as errors on a test set or examples drawn into a minibatch, are binomial, or Poisson when the events are rare. Additive noise is usually normal or Laplace. To say how wide a noisy answer is, or how hard two noisy outputs are to tell apart, you need each family's mean, spread and tail.

The one idea is that the families are built from one another: a binomial is a sum of coin flips, a Poisson is a binomial with many unlikely trials, a Laplace is two exponentials back to back, and a normal is what many small independent pieces add up to. The last section asks how different a density is from a copy of itself moved sideways, the question later parts keep asking. @tbl-dist-overview maps each fact to its section.

| What you need | Rule | Section |
| --- | --- | --- |
| The spread of a count or a measured rate | Binomial variance $np(1-p)$; a rate's standard deviation is $\sqrt{p(1-p)/n}$ | @sec-dist-counts |
| How large Laplace noise gets | $\Pr[\lvert X\rvert \ge t] = e^{-t/b}$, variance $2b^2$ | @sec-dist-laplace |
| How large normal noise gets, in one or many dimensions | Norm $\approx \sigma\sqrt d$, but $\mathcal N(0, \sigma^2)$ along any fixed direction | @sec-dist-normal |
| Whether one observation can be decisive | Laplace log ratio bounded by $\lvert s\rvert/b$; normal log ratio unbounded | @sec-dist-shift |

: The facts this chapter supplies, and where each is stated. {#tbl-dist-overview}

## Counts: binomial and Poisson {#sec-dist-counts}

**Idea.** Every measured rate (accuracy, error rate, false positive rate) is a count of yes/no outcomes divided by the number of trials. With independent trials the count is binomial; with many trials that are each unlikely, it is Poisson.

**Result.** A $\mathrm{Binomial}(n, p)$ count has mean $np$ and variance $np(1-p)$, because variances add over independent trials. The measured rate $\hat p = S/n$ has standard deviation $\sqrt{p(1-p)/n}$: four times the data halves the error. A $\mathrm{Poisson}(\mu)$ count has mean and variance both $\mu$, so its relative spread is $1/\sqrt\mu$, and $\mathrm{Binomial}(n, \mu/n)$ approaches it as $n$ grows (@fig-poisson-limit).

![$\mathrm{Binomial}(n, 5/n)$ for $n = 10$ and 100 against $\mathrm{Poisson}(5)$ and its normal approximation. At $n = 100$ the binomial and the Poisson almost coincide.](figures/poisson_limit.svg){#fig-poisson-limit}

**Example.** A classifier errs on each test example independently with probability 0.05. On 2,000 examples the error count has mean 100 and standard deviation about 9.7, so the measured rate is within about half a percentage point of 5%. At an error rate of 0.001 the count has mean 2 and standard deviation about 1.4, a relative error of about 70%, and one test set in seven shows no error at all. A training loop that includes each of 60,000 examples independently with probability 0.01 gets batches whose size is close to $\mathrm{Poisson}(600)$: mean 600, standard deviation about 24.5.

**Pitfalls.** Rare rates are hard to measure: the relative precision is set by the expected count, not by the number of trials. The variance $p(1-p)$ is tiny near 0, and a bound that uses only the range of each trial cannot see that ([Concentration inequalities](02-concentration.md)). Variances add only over independent terms.

## Laplace and exponential {#sec-dist-laplace}

**Idea.** The **exponential distribution** $\mathrm{Exp}(b)$, with density $\frac1be^{-y/b}$ on $y \ge 0$ and mean $b$, is the waiting time for a rare event. The **Laplace distribution** puts one exponential on each side of a centre: symmetric noise with a sharp peak and tails heavier than a normal's.

**Result.** $\mathrm{Lap}(\mu, b)$ with centre $\mu$ and scale $b$ has density
$$f(x) = \frac{1}{2b}e^{-|x - \mu|/b}.$$
Its size $|X - \mu|$ is $\mathrm{Exp}(b)$, so the average absolute error is $b$, the variance is $2b^2$ (standard deviation $\sigma$ means $b = \sigma/\sqrt2$), and the two-sided tail is exactly
$$\Pr[|X - \mu| \ge t] = e^{-t/b}.$$
The difference of two independent $\mathrm{Exp}(b)$ draws is a $\mathrm{Lap}(b)$ draw.

**Example.** Give a Laplace and a normal the same variance (@fig-laplace-vs-normal). The Laplace exceeds three times its scale with probability $e^{-3} \approx 0.050$, the normal only with probability about 0.034, and the gap grows further out: on a log scale the normal density is a parabola and the Laplace density two straight lines.

![Normal and Laplace densities with variance 2, on linear and log scales. The log of the normal density is a parabola; the log of the Laplace density is two straight lines.](figures/laplace_vs_normal.svg){#fig-laplace-vs-normal}

**Pitfalls.** Check whether a library takes the scale $b$ or the rate $1/b$; NumPy's `rng.exponential(b)` takes the scale. Don't apply normal-shaped tail bounds to Laplace noise: its tail is only exponential.

## Normal {#sec-dist-normal}

**Idea.** The normal is what many small independent effects add up to. Sums of independent normals are normal, and in many dimensions it looks the same in every direction.

**Result.** $\mathcal N(\mu, \sigma^2)$ has density $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/(2\sigma^2)}$. Every probability reduces to the standard normal cumulative distribution function (CDF) $\Phi$, through $\Pr[X \le x] = \Phi((x - \mu)/\sigma)$. In code, use $\Phi(x) = \frac12\operatorname{erfc}(-x/\sqrt2)$. $\sum_ia_iX_i$ for independent normals is normal with mean $\sum_ia_i\mu_i$ and variance $\sum_ia_i^2\sigma_i^2$. A vector $X \sim \mathcal N(0, \sigma^2I_d)$ has length concentrated at $\sigma\sqrt d$, with relative spread about $1/\sqrt{2d}$ (@tbl-gaussian-norms), yet its projection onto any fixed unit vector is just $\mathcal N(0, \sigma^2)$.

| $d$ | 1 | 10 | 100 | 1,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| mean of $\|X\|/\sqrt d$ | 0.802 | 0.977 | 0.998 | 1.000 | 1.000 |
| std of $\|X\|/\sqrt d$ | 0.611 | 0.220 | 0.069 | 0.022 | 0.007 |

: Mean and standard deviation of $\|X\|/\sqrt d$ over 2,000 draws of $X \sim \mathcal N(0, I_d)$. {#tbl-gaussian-norms}

The **central limit theorem (CLT)** says the standardized sum of $n$ independent draws with finite variance approaches $\mathcal N(0, 1)$. Its error in the CDF shrinks only like $1/\sqrt n$: for sums of exponentials it is still 0.0243 at $n = 30$.

**Example.** Add $\mathcal N(0, 0.01^2I_d)$ noise to a model with $d = 10^6$ parameters. The noise vector has length about $0.01 \times 1000 = 10$, yet along any fixed direction it is only $\mathcal N(0, 0.01^2)$: large in norm, small in every direction.

**Pitfalls.** The CLT describes the bulk, not the far tail: its error of order $1/\sqrt n$ dwarfs the tiny tail probabilities a guarantee must control. Never compute a small upper tail as $1 - \Phi(x)$, which loses all its digits; use $\frac12\operatorname{erfc}(x/\sqrt2)$.

## A density against its shifted copy {#sec-dist-shift}

**Idea.** You observe $x = \mu + X$ with noise density $f$, where $\mu$ is either 0 or a known shift $s$. The **log ratio** $L(x) = \ln f(x) - \ln f(x - s)$ says how much more likely $x$ is under one candidate. If it is bounded, no single observation can settle the question; if not, some observations settle it almost for certain. Later parts compare distributions in exactly this way.

**Result.** For normal noise $\mathcal N(0, \sigma^2)$ the log ratio is a straight line, unbounded:
$$L(x) = \frac{s^2 - 2sx}{2\sigma^2}.$$
For Laplace noise $\mathrm{Lap}(b)$ it is clipped:
$$L(x) = \frac{|x - s| - |x|}{b}, \qquad |L(x)| \le \frac{|s|}{b},$$
and the bound is reached on whole half-lines, every $x \le 0$ and every $x \ge s$.

**Example.** Take $s = 1$, $\sigma = 1$, $b = 1$ (@tbl-shifted-log-ratio and @fig-shifted-log-ratio). At $x = 10$ the normal log ratio is $-9.5$: the observation is about 13,360 times more likely under $\mu = 1$. The Laplace never says more than $e \approx 2.72$ times, however far out $x$ lies.

| $x$ | $-2$ | $0$ | $0.5$ | $1$ | $3$ | $10$ |
| --- | --- | --- | --- | --- | --- | --- |
| Normal, $L(x) = (1 - 2x)/2$ | 2.5 | 0.5 | 0 | $-0.5$ | $-2.5$ | $-9.5$ |
| Laplace, $L(x) = \lvert x - 1\rvert - \lvert x\rvert$ | 1 | 1 | 0 | $-1$ | $-1$ | $-1$ |

: The log ratio of a density and its copy shifted by $s = 1$, for $\mathcal N(0, 1)$ and $\mathrm{Lap}(1)$. {#tbl-shifted-log-ratio}

![The log ratio $L(x)$ of a density and its copy shifted by $s = 1$: a straight line for the standard normal, clipped at $\pm1$ for $\mathrm{Lap}(1)$.](figures/shifted_log_ratio.svg){#fig-shifted-log-ratio}

**Pitfalls.** "Unbounded" does not mean "often large": the normal's extreme ratios sit in tails with very little probability, and [Divergences](03-divergences.md) and [Hypothesis testing](04-hypothesis-testing.md) say how little and what it buys an observer. Only shift over scale matters: doubling $b$ or $\sigma$ is the same as halving $s$.

## Cheat sheet

| Distribution | Density or probability mass function | Mean | Variance | Tail |
| --- | --- | --- | --- | --- |
| $\mathrm{Binomial}(n, p)$ | $\binom nk p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | Chernoff, see [Concentration inequalities](02-concentration.md) |
| $\mathrm{Poisson}(\mu)$ | $e^{-\mu}\mu^k/k!$ | $\mu$ | $\mu$ | $\approx(e\mu/k)^k$ |
| $\mathrm{Exp}(b)$ | $\frac1be^{-y/b}$, $y \ge 0$ | $b$ | $b^2$ | $e^{-t/b}$ |
| $\mathrm{Lap}(\mu, b)$ | $\frac{1}{2b}e^{-\lvert x-\mu\rvert/b}$ | $\mu$ | $2b^2$ | $\Pr[\lvert X - \mu\rvert \ge t] = e^{-t/b}$ |
| $\mathcal N(\mu, \sigma^2)$ | $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/(2\sigma^2)}$ | $\mu$ | $\sigma^2$ | $\Pr[X - \mu \ge t] \le \frac12e^{-t^2/(2\sigma^2)}$ |

: The five families: density or probability mass function, moments and tail. {#tbl-distributions-cheat-sheet}

**Rules of thumb**

- A measured rate has standard deviation $\sqrt{p(1-p)/n}$; its relative precision is set by the expected count $np$.
- Normal noise in $d$ dimensions has norm $\approx \sigma\sqrt d$ and is $\mathcal N(0, \sigma^2)$ along every fixed direction.
- Laplace shifted by $s$: log ratio at most $|s|/b$. Normal: unbounded, but large only in the tails, where the CLT says nothing useful.

## Further reading

- Blitzstein & Hwang, *Introduction to Probability* (2nd ed., 2019), ch. 3–5 and 7: every distribution here, with worked examples.
- Wasserman, *All of Statistics* (2004), ch. 2–3: a compact reference.
- Vershynin, *High-Dimensional Probability* (2018), §3.1: why the norm of a normal vector concentrates.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/01-distributions.md).
