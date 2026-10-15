# Exercises: Selection: exponential mechanism, report-noisy-max, sparse vector

Exercises for [Selection: exponential mechanism, report-noisy-max, sparse vector](../05-selection.md), each with a worked solution. Try each exercise before opening its solution. The script `code/solutions_selection.py` computes every number quoted here; run it with `uv run python book/part03-dp-foundations/code/solutions_selection.py`.

The solutions use the chapter's notation. A selection problem has $k$ candidates with scores $u(D, r)$ of sensitivity $\Delta$, and $\mathrm{OPT}(D) = \max_r u(D, r)$. $\mathrm{Lap}(b)$ and $\mathrm{Gumbel}(\beta)$ are given by their scales, and logarithms are natural. AboveThreshold is the sparse vector algorithm of the chapter.

## The factor 2 is needed {#sec-sol-selection-1}

> **Exercise 1.** **The factor 2 is needed.** Use weights $e^{\varepsilon u}$ (no 2) with $\Delta = 1$ on $k$ candidates and the neighbours $u(D) = (1, 0, \dots, 0)$, $u(D') = (0, 1, \dots, 1)$. Compute the privacy loss of the output "candidate 1" and show that it tends to $2\varepsilon$ as $k \to \infty$. *Check: at $\varepsilon = 1$ it is 1.3106 for $k = 3$ and 1.9977 for $k = 1000$.* *(Exponential mechanism, the two factors)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The pair is allowed.** Every score moves by exactly 1, so $\Delta = 1$. The scores are not monotone: candidate 1 goes down while all others go up.
2. **On $D$.** Candidate 1 has weight $e^\varepsilon$ and the other $k - 1$ have weight 1, so $\Pr[1 \mid D] = \dfrac{e^\varepsilon}{e^\varepsilon + k - 1}$.
3. **On $D'$.** Candidate 1 has weight 1 and the others $e^\varepsilon$, so $\Pr[1 \mid D'] = \dfrac{1}{1 + (k - 1)e^\varepsilon}$.
4. **The ratio** splits into the derivation's two factors:
$$\frac{\Pr[1 \mid D]}{\Pr[1 \mid D']} = \underbrace{e^\varepsilon}_{\text{numerator}}\cdot\underbrace{\frac{1 + (k - 1)e^\varepsilon}{e^\varepsilon + k - 1}}_{\text{normalizer}}.$$
5. **The values.** At $\varepsilon = 1$ the script gives a privacy loss of 1.0000 for $k = 2$ (the normalizer factor is exactly 1), 1.3106 for $k = 3$, 1.7761 for $k = 10$ and 1.9977 for $k = 1000$. The closed form and the mechanism's probabilities agree to four decimals.
6. **The limit.** Divide numerator and denominator of the normalizer factor by $k - 1$: it tends to $e^\varepsilon/1 = e^\varepsilon$. The ratio tends to $e^{2\varepsilon}$, so the loss tends to $2\varepsilon$.

With the correct weights $e^{\varepsilon u/2}$ the same calculation gives at most $e^{\varepsilon/2}\cdot e^{\varepsilon/2} = e^\varepsilon$.

*The two factors of the privacy proof are both reached here: candidate 1's weight drops by $e^\varepsilon$, and the normalizer, dominated by the $k - 1$ competitors, rises by nearly $e^\varepsilon$. Monotone scores cannot do both at once, which is why only they may drop the 2.*
:::

## A private median {#sec-sol-selection-2}

> **Exercise 2.** **A private median.** Records take values in $\{0, 1, \dots, 100\}$. Use the exponential mechanism over $r \in \{0, \dots, 100\}$ with score $u(D, r) = -\big|\#\{i : x_i < r\} - \#\{i : x_i > r\}\big|$. Show that $\Delta u = 1$ under add/remove, and use @eq-sel-em-utility to show that with probability at least 0.95 at $\varepsilon = 1$ the output's score is within 15.22 of the best score. When some candidate splits the data evenly, this means $\big|\#\{x_i < r\} - \#\{x_i > r\}\big| \le 15.22$. *Hint: $k = 101$ and $t = \ln 20$.* *(Utility theorem)*

::: {.callout-tip collapse="true" title="Solution"}
**Sensitivity.**

1. Fix a candidate $r$ and write $a = \#\{x_i < r\}$, $c = \#\{x_i > r\}$, so $u = -|a - c|$.
2. Adding a record $x$ increases $a$ by 1 if $x < r$, increases $c$ by 1 if $x > r$, and changes neither if $x = r$. So $a - c$ changes by at most 1.
3. By the reverse triangle inequality, $\big||a' - c'| - |a - c|\big| \le |(a' - c') - (a - c)| \le 1$. Removing a record is the same step backwards. So $\Delta u = 1$.
4. The script checks this by brute force on the dataset below: adding a record of every possible value and removing each distinct value, the largest change of any score is 1.

**Utility.**

1. There are $k = 101$ candidates. With $t = \ln 20$, $e^{-t} = 0.05$.
2. @eq-sel-em-utility gives a gap of at most $\frac{2\Delta}{\varepsilon}(\ln k + t) = 2(\ln 101 + \ln 20) = 2\ln 2020 = 15.22$ with probability at least 0.95.
3. If some candidate $r^*$ has as many records below it as above it, $\mathrm{OPT} = 0$, and the gap is the selected candidate's $|a - c|$ itself.

**A concrete dataset.** Take $n = 1001$ records $x_i = \lfloor 100\,(i/1000)^2\rfloor$ for $i = 0, \dots, 1000$, skewed toward 0. The median is 25. Values repeat (ten records equal 25), so no candidate splits the data evenly: at $r = 25$, 500 records are below and 491 above, so $\mathrm{OPT} = -9$. @tbl-sol-selection-median lists the candidates near the top.

| Candidate $r$ | Records below | Records above | $u(D, r)$ | Gap |
| --- | --- | --- | --- | --- |
| 24 | 490 | 501 | −11 | 2 |
| 25 | 500 | 491 | −9 | 0 |
| 26 | 510 | 481 | −29 | 20 |

: The median score near the true median 25 for the skewed dataset. {#tbl-sol-selection-median}

Only candidates 24 and 25 are within the bound of 15.22. The exact probability of any other output is $4.54\times10^{-5}$, far below the guaranteed 0.05. The mechanism outputs 25 with probability 0.7310, and its expected distance from 25 is 0.2690.

*The utility theorem bounds the error in rank, not in value: with heavily repeated values, the candidates adjacent to the median can be far off in score, and the mechanism almost never picks them.*
:::

## The noisy maximum value {#sec-sol-selection-3}

> **Exercise 3.** **The noisy maximum value.** For independent $G_r \sim \mathrm{Gumbel}(\beta)$, show that $\max_r(u_r + G_r)$ is Gumbel with scale $\beta$ and location $\beta\ln\sum_re^{u_r/\beta}$, so its mean is $\beta\big(\ln\sum_re^{u_r/\beta} + \gamma\big)$ with $\gamma = 0.5772$ the Euler–Mascheroni constant. Check it on the five scores of @tbl-sel-gumbel with $\beta = 2$. *Hint: $\Pr[\max \le x] = \prod_rF(x - u_r)$.* *(Gumbel-max)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The maximum is below $x$ when every term is.** The terms are independent, and $u_r + G_r \le x$ means $G_r \le x - u_r$. With the Gumbel cumulative distribution function (CDF) $F(g) = \exp(-e^{-g/\beta})$,
$$\Pr\Big[\max_r(u_r + G_r) \le x\Big] = \prod_r\exp\big(-e^{-(x - u_r)/\beta}\big) = \exp\Big(-e^{-x/\beta}\sum_re^{u_r/\beta}\Big).$$
2. **Recognize a shifted Gumbel.** Write $\sum_re^{u_r/\beta} = e^{\ell/\beta}$, that is $\ell = \beta\ln\sum_re^{u_r/\beta}$. The CDF is $\exp(-e^{-(x - \ell)/\beta})$: a Gumbel with scale $\beta$ and location $\ell$.
3. **The mean of a Gumbel.** For $G \sim \mathrm{Gumbel}(1)$, substitute $w = e^{-g}$: $\mathbb E G = \int_0^\infty(-\ln w)\,e^{-w}\,dw = -\Gamma'(1) = \gamma$. Scaling and shifting give $\mathbb E\max = \ell + \beta\gamma$.
4. **Check.** For scores $(4, 3, 3, 1, 0)$ and $\beta = 2$: $\ell = 2\ln(e^2 + 2e^{1.5} + e^{0.5} + 1) = 5.8890$, and the mean is $5.8890 + 2\times0.5772 = 7.0434$. A million simulated draws give 7.0430.

The location $\ell$ is the "soft maximum" of the scores: it lies between $\max_ru_r = 4$ and $\max_ru_r + \beta\ln k = 4 + 2\ln5 = 7.2189$.

*The noisy maximum is a smooth function of all the scores, not of the winner's alone. Releasing it is a separate numeric query, which needs its own budget; report-noisy-max releases only the index.*
:::

## Splitting the budget of AboveThreshold {#sec-sol-selection-4}

> **Exercise 4.** **Splitting the budget of AboveThreshold.** With $\rho \sim \mathrm{Lap}(\Delta/\varepsilon_1)$ and $\nu_i \sim \mathrm{Lap}(2\Delta/\varepsilon_2)$, redo the accuracy derivation to get $\alpha = \max\big(2\Delta\ln(2/\beta)/\varepsilon_1,\ 4\Delta\ln(2k/\beta)/\varepsilon_2\big)$, and minimize it over $\varepsilon_1 + \varepsilon_2 = \varepsilon$. *Check: for $k = 1000$, $\beta = 0.05$, $\varepsilon = 1$, $\Delta = 1$, the equal split gives 84.77 and the best split gives 49.76, with $\varepsilon_1 = 0.148$.* *(Laplace tails, union bound)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Threshold noise.** $\Pr[|\rho| \ge \alpha/2] = e^{-(\alpha/2)\varepsilon_1/\Delta}$. This is at most $\beta/2$ when $\alpha \ge 2\Delta\ln(2/\beta)/\varepsilon_1$.
2. **Query noise.** By the union bound, $\Pr[\max_i|\nu_i| \ge \alpha/2] \le k\,e^{-(\alpha/2)\varepsilon_2/(2\Delta)}$. This is at most $\beta/2$ when $\alpha \ge 4\Delta\ln(2k/\beta)/\varepsilon_2$.
3. **Both.** On the event that both noises are below $\alpha/2$, which has probability at least $1 - \beta$, the answers are $\alpha$-accurate exactly as in the chapter. So $\alpha$ is the larger of the two requirements.
4. **Minimize.** Write $a = 2\Delta\ln(2/\beta)$ and $c = 4\Delta\ln(2k/\beta)$. As $\varepsilon_1$ grows, $a/\varepsilon_1$ falls and $c/(\varepsilon - \varepsilon_1)$ rises, so the maximum is smallest where they are equal: $a/\varepsilon_1 = c/\varepsilon_2 = \alpha^*$. Then $\varepsilon_1 + \varepsilon_2 = (a + c)/\alpha^* = \varepsilon$, so
$$\alpha^* = \frac{a + c}{\varepsilon} = \frac{2\Delta\ln(2/\beta) + 4\Delta\ln(2k/\beta)}{\varepsilon}, \qquad \varepsilon_1 = \frac{a}{a + c}\,\varepsilon.$$
5. **Numbers.** With $k = 1000$, $\beta = 0.05$: $a = 2\ln40 = 7.38$ and $c = 4\ln40000 = 42.39$. The equal split gives $\max(14.76, 84.77) = 84.77$, which is Dwork & Roth's $8(\ln k + \ln(2/\beta))$. The best split gives $\alpha^* = 49.76$ with $\varepsilon_1 = 0.148$ and $\varepsilon_2 = 0.852$.

*The threshold noise is drawn once and the query noise $k$ times, so the union bound makes the query noise the bottleneck. Giving it 85% of the budget improves the accuracy by 41% at the same privacy.*
:::

## No query noise {#sec-sol-selection-5}

> **Exercise 5.** **No query noise.** In the two-query counterexample for Algorithm 5 of @tbl-sel-svt-variants, with $T = 0$ and $\rho \sim \mathrm{Lap}(2)$, compute $\Pr[\bot\top \mid D]$ and $\Pr[\bot\top \mid D']$. *Check: 0.1967 and 0.* *(Reading a proof for the step that fails)*

::: {.callout-tip collapse="true" title="Solution"}
1. **On $D$**, $f = (0, 1)$. The first query says $\bot$ when $0 < \rho$, and the second says $\top$ when $1 \ge \rho$. So $\Pr[\bot\top \mid D] = \Pr[0 < \rho \le 1]$.
2. **Laplace CDF.** $\Pr[\rho \le 1] = 1 - \frac12e^{-1/2}$ and $\Pr[\rho \le 0] = \frac12$, so the probability is $\frac12(1 - e^{-1/2}) = 0.1967$. A million simulated runs give 0.1968.
3. **On $D'$**, $f = (1, 0)$. Now $\bot$ needs $1 < \rho$ and $\top$ needs $0 \ge \rho$. No value of $\rho$ does both, so the probability is 0. A million simulated runs never produced $\bot\top$.
4. **Which step fails.** In the AboveThreshold proof, step 3 shifts $\nu_k$ to keep the $\top$ consistent after $\rho$ has been shifted for the $\bot$s. Without $\nu_k$ there is nothing to shift, and the two conditions can become incompatible.

*A probability that is positive on one dataset and zero on its neighbour has infinite privacy loss: one observation of $\bot\top$ proves the data are $D$, not $D'$.*
:::

## Two candidates {#sec-sol-selection-6}

> **Exercise 6.** **Two candidates.** For two candidates whose scores differ by $d$, compute the probability that the better one is selected by the exponential mechanism with weights $e^{u/b}$ and by Laplace report-noisy-max with scale $b$. *Hint: the difference of two independent $\mathrm{Lap}(b)$ variables has $\Pr[Z_1 - Z_2 > x] = \frac12e^{-x/b}(1 + \frac{x}{2b})$ for $x \ge 0$.* *Check: at $d = b$, 0.7311 against 0.7241.* Compare with the 50-model example, where Laplace noise did better. *(Gumbel-max, Laplace differences)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Exponential mechanism.** $\Pr[\text{better}] = \dfrac{e^{d/b}}{e^{d/b} + 1} = \dfrac{1}{1 + e^{-d/b}}$, the logistic function of $d/b$.
2. **The difference of two Laplace variables.** $Z_2$ is symmetric, so $Z_1 - Z_2$ has the law of $Z_1 + Z_2$, whose density is the convolution $\int\frac{1}{4b^2}e^{-|y|/b - |x - y|/b}\,dy$. For $x \ge 0$, split at $y = 0$ and $y = x$: the pieces $y < 0$ and $y > x$ each give $\frac b2e^{-x/b}$, and $0 \le y \le x$ gives $xe^{-x/b}$. So the density is $\frac{1}{4b}\big(1 + \frac{|x|}{b}\big)e^{-|x|/b}$.
3. **Its tail.** $\int_x^\infty\big(1 + \frac yb\big)e^{-y/b}\,dy = (2b + x)\,e^{-x/b}$, so $\Pr[Z_1 - Z_2 > x] = \frac12e^{-x/b}\big(1 + \frac{x}{2b}\big)$.
4. **Laplace report-noisy-max.** The better candidate loses when $Z_2 - Z_1 > d$, so $\Pr[\text{better}] = 1 - \frac12e^{-d/b}\big(1 + \frac{d}{2b}\big)$.
5. **Numbers.** @tbl-sol-selection-two compares the two.

| $d/b$ | 0.5 | 1 | 2 | 4 |
| --- | --- | --- | --- | --- |
| Exponential mechanism | 0.6225 | 0.7311 | 0.8808 | 0.9820 |
| Laplace report-noisy-max | 0.6209 | 0.7241 | 0.8647 | 0.9725 |

: Probability of selecting the better of two candidates. {#tbl-sol-selection-two}

A million simulated runs at $d = b$ give 0.7308 for the Gumbel noisy max and 0.7239 for the Laplace one.

6. **Against the 50 models.** With two candidates the exponential mechanism is better at every gap in the table. On the 50 models of the chapter, with the same scale $b = 2$ at $\varepsilon = 1$, Laplace noise picks the best model with probability 0.4241 against 0.3935 and has the smaller expected gap, 1.40 against 1.54.

*Neither private argmax dominates the other: which one selects better depends on how many competitors are close to the top. Both have the same privacy guarantee, so the choice is about accuracy on the score profiles you expect.*
:::
