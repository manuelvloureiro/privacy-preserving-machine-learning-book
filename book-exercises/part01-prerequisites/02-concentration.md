# Exercises: Concentration inequalities

Exercises for [Concentration inequalities](../../book-full/part01-prerequisites/02-concentration.md), each with a worked solution. Try each exercise before opening its solution. The numbers that are not short hand calculations come from `code/part01-prerequisites/solutions_concentration.py`, which you can run with `uv run python code/part01-prerequisites/solutions_concentration.py`.

## Tightness of Markov {#sec-sol-concentration-1}

> **Exercise 1.** **Tightness of Markov.** For given $\mu$ and $a$, construct the distribution attaining $\Pr[X \ge a] = \mu/a$, and explain why no variable with mean $\mu$ can do worse. *(Markov)*

::: {.callout-tip collapse="true" title="Solution"}
Take $0 < \mu \le a$. (If $\mu > a$, the bound $\mu/a$ exceeds 1 and says nothing; the constant $X = \mu$ has $\Pr[X \ge a] = 1$.)

**The construction.**

1. Let $X = a$ with probability $\mu/a$ and $X = 0$ otherwise. This is a valid distribution because $\mu/a \le 1$, and $X \ge 0$.
2. Its mean is $a \cdot \mu/a + 0 = \mu$.
3. $X \ge a$ happens exactly when $X = a$, so $\Pr[X \ge a] = \mu/a$, which is Markov's bound.

**Why no variable can do worse.** Markov's derivation used nothing about $X$ except $X \ge 0$ and $\mathbb E[X] = \mu$. So every non-negative $X$ with mean $\mu$ has $\Pr[X \ge a] \le \mu/a$, and the two-point variable reaches that ceiling.

**Why this variable, and only this one.** The only inequality in the derivation is step 1, $a\,\mathbf 1[X \ge a] \le X$. Equality after averaging needs equality on every outcome (up to probability zero), so $X$ must equal $a$ where $X \ge a$ and $0$ where $X < a$. Take $\mu = 1$ and $a = 4$ and compare the two rows, as in @tbl-die-markov:

| Outcome | $X = 0$ (prob. $3/4$) | $X = 4$ (prob. $1/4$) | Average |
| --- | --- | --- | --- |
| $4\cdot\mathbf 1[X \ge 4]$ | 0 | 4 | $4\Pr[X \ge 4] = 1$ |
| $X$ | 0 | 4 | $1$ |

: The scaled indicator and the variable agree on every outcome of the two-point distribution with $\mu = 1$, $a = 4$, so Markov loses nothing. {#tbl-sol-concentration-markov-tight}

The two rows of @tbl-sol-concentration-markov-tight are identical. On the die, by contrast, every roll except 5 had slack. Any mass strictly between 0 and $a$, or strictly above $a$, creates slack, because it adds to the mean without adding to $\Pr[X \ge a]$ at the full rate.

Non-negativity matters. Without it, $X = a$ with probability $1 - \xi$ and $X = -(a(1-\xi) - \mu)/\xi$ otherwise has mean $\mu$ and $\Pr[X \ge a] = 1 - \xi$, as close to 1 as you like.

*Markov is the best bound that uses only the mean: its worst case puts all the mass at $0$ and $a$.*
:::

## Cantelli's inequality {#sec-sol-concentration-2}

> **Exercise 2.** **Cantelli's inequality.** Show that $\Pr[X - \mu \ge t] \le \sigma^2/(\sigma^2 + t^2)$. *Hint: apply Markov to $(X - \mu + u)^2$ for $u > 0$, then optimize $u$.* *(Generalized Markov)*

::: {.callout-tip collapse="true" title="Solution"}
Let $Y = X - \mu$, so $\mathbb E[Y] = 0$ and $\mathbb E[Y^2] = \sigma^2$. Take $t > 0$ and any $u > 0$.

1. **Event inclusion.** If $Y \ge t$, then $Y + u \ge t + u > 0$, and squaring two positive numbers keeps the order: $(Y + u)^2 \ge (t + u)^2$. So $\Pr[Y \ge t] \le \Pr[(Y + u)^2 \ge (t + u)^2]$. (The function $(y+u)^2$ is not increasing everywhere, so this is only an inclusion, not an equality. That is enough.)
2. **Markov** on the non-negative variable $(Y + u)^2$ with threshold $(t+u)^2$:
$$\Pr[Y \ge t] \le \frac{\mathbb E[(Y + u)^2]}{(t + u)^2} = \frac{\sigma^2 + 2u\,\mathbb E[Y] + u^2}{(t + u)^2} = \frac{\sigma^2 + u^2}{(t + u)^2}.$$
3. **Optimize $u$.** The log of the right side is $\ln(\sigma^2 + u^2) - 2\ln(t + u)$, with derivative $\frac{2u}{\sigma^2 + u^2} - \frac{2}{t + u}$. It vanishes when $u(t + u) = \sigma^2 + u^2$, that is, at $u = \sigma^2/t > 0$. The derivative is negative below this point and positive above it, so it is the minimum.
4. **Substitute** $u = \sigma^2/t$. The numerator is $\sigma^2 + \sigma^4/t^2 = \sigma^2(t^2 + \sigma^2)/t^2$, and the denominator is $(t + \sigma^2/t)^2 = (t^2 + \sigma^2)^2/t^2$. Their ratio is
$$\Pr[X - \mu \ge t] \le \frac{\sigma^2}{\sigma^2 + t^2}.$$

**How it compares with Chebyshev.** Setting $u = 0$ in step 2 gives back $\sigma^2/t^2$. The shift $u$ helps because a one-sided event lets you move the centre of the square towards the side you don't care about. At $t = \sigma$, Chebyshev gives 1 and Cantelli gives $1/2$. For a standard normal at $t = 3$, Cantelli gives $1/10 = 0.1$ against Chebyshev's $0.111$ (@tbl-concentration-results): the gain fades for large $t$, where both decay like $\sigma^2/t^2$.

**It is tight.** Let $Y = t$ with probability $p = \sigma^2/(\sigma^2 + t^2)$ and $Y = -\sigma^2/t$ otherwise. Its mean is $tp - (\sigma^2/t)(1 - p) = \frac{t\sigma^2 - \sigma^2 t}{\sigma^2 + t^2} = 0$, its variance is $t^2 p + (\sigma^4/t^2)(1-p) = \frac{t^2\sigma^2 + \sigma^4}{\sigma^2 + t^2} = \sigma^2$, and $\Pr[Y \ge t] = p$.

*A one-sided question deserves a one-sided bound: shifting before squaring buys up to a factor of 2 over Chebyshev.*
:::

## A direct Hoeffding lemma {#sec-sol-concentration-3}

> **Exercise 3.** **A direct Hoeffding lemma.** For $X = \pm1$ with probability $\frac12$ each, show that $\mathbb E[e^{\lambda X}] = \cosh\lambda \le e^{\lambda^2/2}$. *Hint: compare Taylor series term by term, using $(2k)! \ge 2^k k!$.* *(Sub-Gaussian)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The moment generating function (MGF).** $\mathbb E[e^{\lambda X}] = \frac12 e^{\lambda} + \frac12 e^{-\lambda} = \cosh\lambda$.
2. **Its series.** Add the series of $e^\lambda$ and $e^{-\lambda}$ and halve. Odd powers cancel and even powers survive: $\cosh\lambda = \sum_{k \ge 0}\frac{\lambda^{2k}}{(2k)!}$.
3. **The series of the bound.** Substitute $\lambda^2/2$ into $e^x = \sum_k x^k/k!$: $e^{\lambda^2/2} = \sum_{k \ge 0}\frac{\lambda^{2k}}{2^k k!}$.
4. **The factorial inequality.** Group $(2k)!$ into $k$ pairs of consecutive factors:
$$(2k)! = \prod_{i=1}^{k}(2i - 1)(2i) \ge \prod_{i=1}^{k} 2i = 2^k k!,$$
since each odd factor $2i - 1$ is at least 1.
5. **Compare term by term.** Both series have the same powers $\lambda^{2k} \ge 0$. By step 4 each coefficient of $\cosh\lambda$ is at most the matching coefficient of $e^{\lambda^2/2}$. Summing gives $\cosh\lambda \le e^{\lambda^2/2}$ for every real $\lambda$.

@tbl-sol-concentration-cosh shows the first four terms.

| $k$ | Term of $\cosh\lambda$ | Term of $e^{\lambda^2/2}$ | $(2k)!$ vs $2^k k!$ |
| --- | --- | --- | --- |
| 0 | $1$ | $1$ | $1 = 1$ |
| 1 | $\lambda^2/2$ | $\lambda^2/2$ | $2 = 2$ |
| 2 | $\lambda^4/24$ | $\lambda^4/8$ | $24 > 8$ |
| 3 | $\lambda^6/720$ | $\lambda^6/48$ | $720 > 48$ |

: The first terms of the two series. They agree up to $\lambda^2$, which is why the variance proxy 1 equals the true variance; afterwards the bound's coefficients are larger. {#tbl-sol-concentration-cosh}

So $X$ is 1-sub-Gaussian. Hoeffding's lemma on $[l, u] = [-1, 1]$ gives the same constant, $(u - l)/2 = 1$: this fair coin on the endpoints is exactly the lemma's worst case. The series comparison reaches the same constant without the convexity argument.
:::

## Sample size for a mean {#sec-sol-concentration-4}

> **Exercise 4.** **Sample size for a mean.** How many samples in $[0, 1]$ does Hoeffding need to estimate a mean to within $\pm0.01$ with probability 0.99? *Check: $n \ge \ln(200)/(2\times10^{-4}) \approx 26{,}500$.* *(Hoeffding, Fact 5)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Two-sided Hoeffding** for the mean of $n$ independent variables in $[0, 1]$, so $u - l = 1$: $\Pr[|\bar X_n - \mu| \ge t] \le 2e^{-2nt^2}$ (the one-sided bound, doubled for the lower tail).
2. **Set the failure probability** to $\beta = 1 - 0.99 = 0.01$ and require $2e^{-2nt^2} \le \beta$.
3. **Solve for $n$** (Fact 5): take logs, $-2nt^2 \le \ln(\beta/2)$, so
$$n \ge \frac{\ln(2/\beta)}{2t^2} = \frac{\ln 200}{2 \times 10^{-4}} = 26{,}491.6.$$
4. $n$ is an integer, so **$n = 26{,}492$** samples suffice. This matches the check value.

For comparison, Chebyshev with the largest possible variance on $[0, 1]$, $\sigma^2 \le 1/4$, needs $n \ge \sigma^2/(t^2\beta) = 0.25/(10^{-4} \times 0.01) = 250{,}000$: almost ten times more. The $\ln(1/\beta)$ in Hoeffding against the $1/\beta$ in Chebyshev is the whole difference.

*Halving the error multiplies $n$ by 4; dividing the failure probability by 10 only adds $\ln 10/(2t^2)$.*
:::

## Noisy argmax {#sec-sol-concentration-5}

> **Exercise 5.** **Noisy argmax.** You add independent $\mathrm{Lap}(b)$ noise to each of $k$ scores and pick the largest noisy score. Show that with probability at least $1-\beta$, the true score of the item you pick is within $2b\ln(k/\beta)$ of the true maximum. *(Laplace tail + union bound)*

::: {.callout-tip collapse="true" title="Solution"}
Write $u_1, \dots, u_k$ for the true scores, $Z_1, \dots, Z_k$ for the noise, $i^*$ for an item with the largest true score, and $\hat\imath$ for the item with the largest noisy score $u_i + Z_i$. Let $\tau = b\ln(k/\beta)$.

1. **One draw (Laplace tail).** For each $i$, $\Pr[|Z_i| \ge \tau] = e^{-\tau/b} = \beta/k$.
2. **All draws at once (union bound, Fact 2).** $\Pr[\text{some } |Z_i| \ge \tau] \le k \cdot \beta/k = \beta$. So the event $G$ = "every $|Z_i| < \tau$" has probability at least $1 - \beta$.
3. **The choice.** $\hat\imath$ maximizes the noisy score, so in particular $u_{\hat\imath} + Z_{\hat\imath} \ge u_{i^*} + Z_{i^*}$.
4. **Rearrange:** $u_{i^*} - u_{\hat\imath} \le Z_{\hat\imath} - Z_{i^*}$.
5. **On $G$**, both noise terms are below $\tau$ in absolute value, so $Z_{\hat\imath} - Z_{i^*} < 2\tau$. Hence $u_{i^*} - u_{\hat\imath} < 2b\ln(k/\beta)$ with probability at least $1 - \beta$.

**Why step 2 needs the union bound.** The picked item $\hat\imath$ depends on the noise: it tends to be an item whose noise came out large. So you can't apply the one-draw bound to $Z_{\hat\imath}$ as if $\hat\imath$ were fixed in advance. Bounding all $k$ noise terms at once covers whichever item gets picked.

**Size.** With $k = 100$, $b = 1$ and $\beta = 0.05$, the guarantee is $2\ln 2000 = 15.20$, twice the maximum-of-Laplace bound 7.60 of @tbl-concentration-results. It grows only like $\ln k$. It is loose, because it charges the worst case on both sides: the picked item's noise at $+\tau$ and the best item's at $-\tau$.

*Selecting with noise costs at most twice the largest noise, and the largest of $k$ noises costs only $\ln k$.*
:::

## Chernoff for a Poisson {#sec-sol-concentration-6}

> **Exercise 6.** **Chernoff for a Poisson.** For $N \sim \mathrm{Poisson}(\mu)$, show that $\Pr[N \ge (1+r)\mu]$ satisfies exactly the same bound as the Bernoulli sum. Why is that no coincidence? *(Chernoff)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The MGF.** $\mathbb E[e^{\lambda N}] = \exp(\mu(e^\lambda - 1))$ ([Probability distributions](../../book-full/part01-prerequisites/01-distributions.md)).
2. **Master bound** with $t = (1 + r)\mu$ and $\lambda > 0$: $\Pr[N \ge (1+r)\mu] \le \exp\big(\mu(e^\lambda - 1) - \lambda(1+r)\mu\big)$.
3. **This is the Bernoulli expression.** Step 3 of the Bernoulli-sum derivation reached exactly this exponent. Steps 4 and 5 there worked on that exponent alone, without going back to the Bernoullis. So they apply verbatim: $\lambda = \ln(1 + r)$ gives
$$\Pr[N \ge (1+r)\mu] \le \exp\big(-\mu[(1+r)\ln(1+r) - r]\big) \le \exp\left(-\frac{r^2\mu}{2+r}\right).$$
The lower tail works the same way: the Bernoulli lower-tail step 1 also reduced to $\exp(\lambda s + \mu(e^{-\lambda} - 1))$, which is the Poisson MGF at $-\lambda$.

**Why it is no coincidence.** Step 1 of the Bernoulli derivation replaced each factor $1 + p_i(e^\lambda - 1)$ by $\exp(p_i(e^\lambda - 1))$, the MGF of a $\mathrm{Poisson}(p_i)$. The product of those is the MGF of a sum of independent Poissons, which is $\mathrm{Poisson}(\mu)$. So the Bernoulli-sum bound *is* the Chernoff bound of $\mathrm{Poisson}(\mu)$, used as an upper bound on the MGF of the sum. For the Poisson itself, that step is an equality and nothing is lost. The Poisson is the worst case among sums of independent Bernoullis with mean $\mu$, as far as the MGF can tell. It is also their limit as every $p_i \to 0$, so a bound that depends only on $\mu$ and holds for all Bernoulli sums has to hold for the Poisson anyway.

**How tight.** For $\mu = 10$ and $r = 1$, the exact tail $\Pr[N \ge 20]$ is $3.45\times10^{-3}$. The optimized bound gives $2.10\times10^{-2}$ and the simplified one $3.57\times10^{-2}$: about 6 and 10 times too large. The exponent is right (Cramér's theorem); what the bound misses is a prefactor of order $1/\sqrt\mu$.
:::

## Azuma for the coin game {#sec-sol-concentration-7}

> **Exercise 7.** **Azuma for the coin game.** Modify the script to use a different adaptive rule (for example, pick the 0.7 coin after two heads in a row). Check that the Azuma bound still holds and explain why the rule does not matter. *(Azuma)*

::: {.callout-tip collapse="true" title="Solution"}
**The modification.** Make the rule a parameter. A rule sees only the past (the last two flips and the running martingale) and says, for each simulated game, whether the next flip uses the 0.7 coin:

```text
Rule = Callable[[BoolArray, BoolArray, FloatArray], BoolArray]

def rule_after_two_heads(last: BoolArray, before: BoolArray, m: FloatArray) -> BoolArray:
    return last & before

def adaptive_martingale(rng, rule: Rule, n: int, p_low: float, p_high: float, trials: int) -> FloatArray:
    m = np.zeros(trials)
    last = np.zeros(trials, dtype=bool)
    before = np.zeros(trials, dtype=bool)
    for _ in range(n):
        p = np.where(rule(last, before, m), p_high, p_low)  # chosen from the past only
        x = rng.random(trials) < p
        m += x - p
        before, last = last, x
    return m
```

The solutions script runs four rules, with $n = 200$ flips, $t = 20$ and $10^6$ games each. "When behind" tries to push the martingale up by using the 0.7 coin whenever $W_{k-1} < 0$. "Always 0.7" is not adaptive at all, so its heads count is $\mathrm{Bin}(200, 0.7)$ and $\Pr[W_{200} \ge 20] = \Pr[\text{heads} \ge 160]$ can be computed exactly: $9.28\times10^{-4}$.

| Rule for using the 0.7 coin | Empirical $\Pr[W_{200} \ge 20]$ |
| --- | --- |
| After one head (the chapter's game) | $9.13\times10^{-4}$ |
| After two heads in a row | $1.10\times10^{-3}$ |
| When behind ($W_{k-1} < 0$) | $1.18\times10^{-3}$ |
| Always | $9.37\times10^{-4}$ |
| Azuma bound $e^{-2t^2/n}$, any rule | $1.83\times10^{-2}$ |
| Normal approximation, variance $200 \times 0.21$ | $1.01\times10^{-3}$ |

: Tail of the coin-game martingale at $t = 20$ after 200 flips, under four adaptive rules, with $10^6$ simulated games per rule and seed 0. {#tbl-sol-concentration-azuma-rules}

The Azuma bound holds for every rule in @tbl-sol-concentration-azuma-rules, with a margin of more than 15 times. (The chapter's $1.12\times10^{-3}$ for the first rule came from $10^5$ games; with about 110 games in the tail, its standard error is about $1.1\times10^{-4}$, so it is within two standard errors of $9.13\times10^{-4}$.)

**Why the rule does not matter.** Azuma's derivation used only one fact about each increment: given the past, $U_k$ has mean zero and lies in an interval of length $d_k$. Check it for any rule that decides from the past.

1. The rule fixes $p_k$ before flip $k$, so given the past, $p_k$ is a constant.
2. Given the past, $X_k$ is a $\mathrm{Bern}(p_k)$ flip, so $U_k = X_k - p_k$ has conditional mean $p_k - p_k = 0$.
3. $U_k$ takes the values $1 - p_k$ and $-p_k$, so it lies in $[-p_k, 1 - p_k]$, an interval of length $d_k = 1$.

Concretely, after the history "heads, heads" the two-heads rule picks $p_k = 0.7$: $U_k = +0.3$ with probability 0.7 and $-0.7$ with probability 0.3, with mean $0.21 - 0.21 = 0$. After any other history it picks $0.3$: $U_k = +0.7$ or $-0.3$, again with mean 0. With $d_k = 1$ for all 200 steps, $\sum_k d_k^2 = 200$ and the bound is $e^{-2 \cdot 20^2/200}$ whatever the rule. A rule that peeked at flip $k$ before choosing its coin would break step 2. That is the one thing adaptivity may not do.

**Why the simulated tails barely move either.** Both coins have the same variance, $0.3 \times 0.7 = 0.21$. So whatever the rule picks, each increment has conditional variance 0.21, and the martingale's total variance is $200 \times 0.21 = 42$. The normal approximation with that variance gives $1.01\times10^{-3}$, close to every row. Azuma assumes the worst-case per-step variance $d_k^2/4 = 0.25$ and uses a Chernoff-type exponent, which is why it sits 15 to 20 times above.

*Azuma needs only "mean zero and bounded given the past", which no rule that decides from the past can break.*
:::

## Low-rate estimate {#sec-sol-concentration-8}

> **Exercise 8.** **Low-rate estimate.** You observe 3 false positives in 5,000 trials. Give 95% upper bounds on the rate using Hoeffding, Bernstein, the multiplicative Chernoff bound, and the exact Clopper–Pearson interval (`scipy.stats.beta.ppf`). Explain the ranking. *Check: Clopper–Pearson gives $\approx 0.00155$.* *(Hoeffding vs variance-aware bounds)*

::: {.callout-tip collapse="true" title="Solution"}
Let $p$ be the true rate, $n = 5000$, $s = 3$ the count, $\hat p = s/n = 0.0006$, and $\beta = 0.05$, so $\ln(1/\beta) = \ln 20 \approx 3.00$. An upper bound needs the lower tail: a candidate rate $p$ is ruled out when, were $p$ true, a count as low as 3 would have probability below $\beta$. The 95% upper bound is the largest $p$ not ruled out. For a bound of the form "$\hat p < p - w$ with probability at most $\beta$", that is simply $\hat p + w$ (Fact 5).

1. **Hoeffding** (one-sided, range $[0, 1]$): $w = \sqrt{\ln 20/(2 \times 5000)}$, so the upper bound is $0.0179$.
2. **Bernstein** with $K = 1$, applied to the lower tail (to $-X_i$, which has the same variance and range). The chapter's plug-in version, with $\sigma^2 \approx \hat p(1 - \hat p)$, gives $\hat p + \sqrt{2\hat p(1-\hat p)\ln 20/n} + 2\ln 20/(3n) = 0.00185$. But a candidate $p$ above $\hat p$ has a larger variance $p(1-p)$, so the plug-in is optimistic. The rigorous version uses each candidate's own variance: rule out $p$ when $\exp\big(-n(p - \hat p)^2/(2p(1-p) + \frac23(p - \hat p))\big) < \beta$. The largest $p$ not ruled out, found by bisection, is $0.00256$.
3. **Multiplicative Chernoff**, lower tail $\Pr[S \le (1-r)\mu] \le e^{-r^2\mu/2}$ with $\mu = np$. At $s = (1-r)\mu$ the exponent is $r^2\mu/2 = (\mu - s)^2/(2\mu)$. Rule out $\mu$ when $(\mu - s)^2 > 2\mu\ln 20$. The boundary is the larger root of $\mu^2 - 2(s + \ln 20)\mu + s^2 = 0$:
$$\mu = s + \ln 20 + \sqrt{(\ln 20)^2 + 2s\ln 20},$$
which gives the upper bound $\mu/n = 0.00224$. With the unsimplified exponent $\mu[r + (1-r)\ln(1-r)] = \mu - s + s\ln(s/\mu)$, it is $0.00189$.
4. **Clopper–Pearson.** The largest $p$ with $\Pr[\mathrm{Bin}(5000, p) \le 3] \ge 0.05$. The script finds it by bisection on the binomial cumulative sum; it equals `scipy.stats.beta.ppf(0.95, 4, 4997)`. The result is $0.00155$, matching the check value.

| Method | 95% upper bound on the rate |
| --- | --- |
| Hoeffding | 0.0179 |
| Bernstein (each candidate's own variance) | 0.00256 |
| Chernoff, $e^{-r^2\mu/2}$ | 0.00224 |
| Chernoff, unsimplified exponent | 0.00189 |
| Clopper–Pearson (exact) | 0.00155 |
| Bernstein, plug-in $\hat p(1-\hat p)$ (heuristic) | 0.00185 |

: One-sided 95% upper bounds on a rate after 3 events in 5,000 trials (observed rate 0.0006). {#tbl-sol-concentration-low-rate}

**The ranking** in @tbl-sol-concentration-low-rate follows how much each method knows about the count.

- **Hoeffding** knows only the range, so it assumes the variance $1/4$. The true variance is about $0.0006$, over 400 times smaller. Its bound is 30 times the observed rate and more than 10 times the exact one.
- **Bernstein** knows the variance. That removes most of the gap, but it pays for the range term $\frac23 K(p - \hat p)$ with $K = 1$ and for its constants.
- **Chernoff** knows the whole MGF of a Bernoulli sum, through its Poisson bound. Simplifying the exponent to $r^2\mu/2$ costs some of that.
- **Clopper–Pearson** uses the binomial distribution itself, with no inequality at all. Nothing valid can be smaller.

The plug-in Bernstein value looks better than the rigorous one, but only because it uses the variance at $\hat p$, which is too small for the rates it should be ruling in.

*Near zero, only variance-aware bounds are usable, and with 3 events even the exact bound is 2.6 times the observed rate.*
:::
