# Concentration inequalities

## Why this matters

A noisy answer, a measured error rate, the size of a sampled minibatch: each is random, and the practical question is **how wide an error bar must be to fail with probability at most $\beta$**. Concentration inequalities answer it.

The answer depends on how much you know. Knowing only the variance, the width grows like $1/\sqrt\beta$; knowing that the tail falls off like a normal's, only like $\sqrt{\ln(1/\beta)}$; for an exponential tail, like $\ln(1/\beta)$. At 95% the difference is modest, but later parts need $\beta$ far smaller, so this chapter is about using everything you know.

## Variance alone: Markov and Chebyshev {#sec-conc-chebyshev}

A quantity $X \ge 0$ with mean $\mathbb E[X]$ cannot often be much larger than its mean, or those large values alone would pull the mean up. That is **Markov's inequality**, $\Pr[X \ge a] \le \mathbb E[X]/a$. Apply it to the squared deviation $(X - \mu)^2$, whose mean is the variance $\sigma^2$, and you get **Chebyshev's inequality**,
$$\Pr[|X - \mu| \ge t] \le \frac{\sigma^2}{t^2}.$$
Setting the right side to $\beta$ and solving gives the width $\sigma/\sqrt\beta$. For a mean of $n$ independent samples the variance is $\sigma^2/n$, so the width is $\sigma/\sqrt{n\beta}$.

Chebyshev must hold for every distribution with that variance, including ones that hide their spread in a rare, far-off spike, so it shrinks slowly as $\beta$ falls. @tbl-width-price compares its width with the width under a normal-shaped tail, $\sigma\sqrt{2\ln(1/\beta)}$. For Laplace noise with $b = 1$, Chebyshev gives a 95% width of about 6.3, while the exact tail gives $\ln 20 \approx 3.0$; at $\beta = 10^{-6}$ the two are 1,414 and 13.8, so ignoring the noise's shape costs a factor of 100.

| $\beta$ | Chebyshev, $1/\sqrt\beta$ | Normal-shaped tail, $\sqrt{2\ln(1/\beta)}$ |
| --- | --- | --- |
| $0.05$ | 4.5 | 2.4 |
| $10^{-3}$ | 31.6 | 3.7 |
| $10^{-6}$ | 1,000 | 5.3 |

: Width needed to fail with probability at most $\beta$, in units of $\sigma$. {#tbl-width-price}

## Normal and Laplace noise {#sec-conc-noise}

When you know the noise distribution, read the width off its tail. The Laplace tail is exact, $\Pr[|X| \ge t] = e^{-t/b}$, so setting it to $\beta$ gives $|X| \le b\ln(1/\beta)$ with probability $1 - \beta$. The normal tail has no closed form, but $\Pr[X \ge t] \le \frac12e^{-t^2/(2\sigma^2)}$ for $X \sim \mathcal N(0, \sigma^2)$, which gives the width $\sigma\sqrt{2\ln(1/\beta)}$. The shapes explain the growth rates: a tail like $e^{-t}$ reaches $\beta$ at $t = \ln(1/\beta)$, a tail like $e^{-t^2}$ at its square root.

To keep $k$ draws small at once, ask each to fail with probability $\beta/k$. The **union bound** says the chance that any fails is at most the sum, $\beta$, because adding chances can count an outcome twice but never miss it. Replacing $\beta$ by $\beta/k$ adds $\ln k$ inside the logarithm, so all $k$ normal draws stay below $\sigma\sqrt{2\ln(k/\beta)}$ and all $k$ Laplace draws below $b\ln(k/\beta)$: controlling many things at once is cheap. When each draw rarely fails, double failures are rarer still, so the bound is nearly exact: for the maximum of 100 $\mathrm{Lap}(1)$ draws at $\beta = 0.05$ the width is $\ln(100/0.05) = 7.60$, and in simulation it fails at rate 0.0486 (@fig-laplace-max).

![Left: the width $b\ln(k/\beta)$ for the maximum of $k$ Laplace draws grows like $\ln k$. Right: the probability that the maximum exceeds it stays at $\beta = 0.05$ for every $k$.](figures/laplace_max.svg){#fig-laplace-max}

A sum of $k$ Laplace draws has a normal-looking bulk, as the central limit theorem (CLT) predicts, but its far tail keeps the Laplace's exponential decay. A valid bound is Gaussian for small deviations and exponential for large ones:
$$\Pr[S \ge t] \le \exp\Big(-\min\Big(\frac{t^2}{8kb^2}, \frac{t}{4b}\Big)\Big).$$
For $k = 10$ and $b = 1$, the normal approximation puts $\Pr[S \ge 20]$ at $3.9\times10^{-6}$; the truth is $3.4\times10^{-5}$, nine times more (@fig-laplace-sum). When the answer must be a guarantee, use a bound, not the CLT.

![Empirical tail of a sum of 10 $\mathrm{Lap}(1)$ draws, against the sub-exponential bound and the normal approximation, which badly underestimates the far tail.](figures/laplace_sum.svg){#fig-laplace-sum}

## Counts: the Chernoff bound {#sec-conc-chernoff}

A count has no simple exact tail, so apply Markov to $e^{\lambda X}$ instead of $X$ and choose the best $\lambda$. The average of $e^{\lambda X}$ contains every moment of $X$, and for independent terms it is a product of one factor per term, so the bound shrinks exponentially as terms are added. For a sum $S$ of independent yes/no variables with mean $\mu$,
$$\Pr[S \ge (1+\delta)\mu] \le \exp\Big(-\frac{\delta^2\mu}{2+\delta}\Big), \qquad \Pr[S \le (1-\delta)\mu] \le \exp\Big(-\frac{\delta^2\mu}{2}\Big),$$
the second for $0 < \delta < 1$.

The exponent contains the expected count $\mu$, not the number of trials, because the bound keeps each trial's probability and so knows that the count's variance is only about $\mu$. Rare events concentrate only once you expect many: a million trials at $p = 10^{-6}$ expect one event and do not concentrate at all. The tails also differ: for large $\delta$ the upper exponent behaves like $\delta\mu$, not $\delta^2\mu$, since a count can overshoot its mean far more easily than undershoot it.

Each of 60,000 examples joins a minibatch independently with probability 0.01, so $\mu = 600$. The batch is more than 20% too large with probability at most $e^{-10.9} \approx 1.8\times10^{-5}$, and more than 20% too small with probability at most $e^{-12} \approx 6.1\times10^{-6}$. Over 1,000 steps the union bound gives about 0.024 for any batch leaving $600 \pm 20\%$.

## Averages of bounded quantities: Hoeffding and Bernstein {#sec-conc-hoeffding}

A variable confined to an interval $[a, b]$ cannot have a heavy tail, and the most spread out it can be is a fair coin on the two endpoints, with variance $(b - a)^2/4$. **Hoeffding's inequality** says that an average of $n$ such variables concentrates at least as well as that worst case: with probability at least $1 - \beta$,
$$|\bar X_n - \mu| < (b - a)\sqrt{\frac{\ln(2/\beta)}{2n}}.$$
For a finite set $H$ of models with losses in $[0, 1]$, the union bound puts $\ln|H|$ inside the logarithm: every model's training error is within $\sqrt{\ln(2|H|/\beta)/(2n)}$ of its true error.

Assuming the worst-case variance is also its weakness, serious for small rates. A classifier makes 1 false positive on 1,000 negatives, and you want a 95% upper bound on its true rate. One-sided Hoeffding adds about 0.039, some 40 times the measured rate, because it assumes a variance of 0.25 when the true one is about 0.001. **Bernstein's inequality** keeps the variance $\sigma^2$: for $|X_i - \mu| \le M$ its width is about $\sqrt{2\sigma^2\ln(1/\beta)/n} + 2M\ln(1/\beta)/(3n)$, driven by the variance except far out. With the variance estimated as 0.001 it adds about 0.0045. That is a heuristic, since the variance came from the same data. The exact binomial (Clopper–Pearson) bound is 0.0047, and [Hypothesis testing](04-hypothesis-testing.md) shows how to compute it. For the mean of 200 Bernoulli(0.1) variables reaching 0.15, Hoeffding gives 0.37, losing even to Chebyshev's 0.18; Chernoff gives 0.135 (@fig-bernoulli-tails).

![Empirical tail of the mean of 200 Bernoulli(0.1) variables against the Markov, Chebyshev, Hoeffding and Chernoff bounds.](figures/bernoulli_tails.svg){#fig-bernoulli-tails}

## Functions of many inputs: McDiarmid and Azuma {#sec-conc-mcdiarmid}

A trained model's risk is a function of its data, not a sum, and in a sequential procedure each step can depend on the ones before. Concentration survives both if no single input or step can move the result much. **Azuma's inequality** covers a running total whose steps have mean zero given the past and lie in intervals of lengths $d_k$, with a bound of Hoeffding's form, $\exp(-2t^2/\sum_kd_k^2)$. **McDiarmid's inequality** follows by revealing a function's inputs one at a time and tracking your best guess of the result: if the inputs are independent and changing only $X_i$ moves $f$ by at most $c_i$, each revealed input moves the guess by at most $c_i$, and
$$\Pr[f(X) - \mathbb E f(X) \ge t] \le \exp\Big(-\frac{2t^2}{\sum_ic_i^2}\Big).$$
For an empirical risk with losses in $[0, 1]$, one example moves the average by at most $1/n$, so McDiarmid gives $\exp(-2nt^2)$, Hoeffding again. Its real use is for functions that are not sums, such as the risk of a retrained model ([Algorithmic stability and generalization](07-stability-generalization.md)). The per-record influence $c_i$ must hold for every possible change of $X_i$, since the argument charges each input its largest possible move.

Like Hoeffding, both pay for the worst-case variance. Flip 200 times, choosing before each flip a coin with heads probability 0.3 or 0.7 from the previous flip: the drift of heads from what the chosen coins predict is a running total with steps of length 1. Azuma bounds a drift of 20 by $1.83\times10^{-2}$; simulation gives $1.12\times10^{-3}$, because each step's real variance is 0.21, not the 0.25 the interval allows.

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

## Further reading

- Mitzenmacher & Upfal, *Probability and Computing*, ch. 4: the clearest Chernoff chapter, with computer-science examples.
- Vershynin, *High-Dimensional Probability* (2018), ch. 2: sub-Gaussian and sub-exponential variables in depth.
- Boucheron, Lugosi & Massart, *Concentration Inequalities* (2013), ch. 2 and 6: reference depth, including Bernstein's inequality.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/02-concentration.md).
