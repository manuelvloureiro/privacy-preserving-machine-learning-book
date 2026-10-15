# Exercises: Numerical accounting with privacy loss distributions

Exercises for [Numerical accounting with privacy loss distributions](../06-numerical-accounting.md), each with a worked solution. Try each exercise before opening its solution. The script `code/solutions_numerical_accounting.py` computes every number quoted here; run it with `uv run python book/part04-privacy-accounting/code/solutions_numerical_accounting.py`.

The solutions use the chapter's notation. $P$ and $Q$ are the output distributions, with densities $P(y)$ and $Q(y)$, because $q$ is the sampling rate. $L = \ln(P/Q)$ is the privacy loss, and the privacy loss distribution (PLD) $\omega$ is its law under $Y \sim P$. $\delta(\varepsilon) = \mathbb E_\omega[(1 - e^{\varepsilon - L})_+]$ (@eq-num-delta). The subsampled Gaussian is the one-step pair of stochastic gradient descent (SGD) under differential privacy (DP), called DP-SGD, with $q = 0.01$ and $\sigma = 1$.

## The PLD of the swapped pair {#sec-sol-num-1}

> **Exercise 1.** **The PLD of the swapped pair.** Show that if $\omega$ is the PLD of $(P, Q)$, the PLD of $(Q, P)$ puts mass $e^{-\ell}\omega(d\ell)$ at loss $-\ell$, plus an atom at $+\infty$ of size $1 - \mathbb E_\omega[e^{-L}]$. Deduce that the add-direction PLD of the subsampled Gaussian is determined by the remove-direction one, and that the Gaussian pair's PLD is the same in both directions. *Check: tilting the pessimistic remove PLD ($h = 10^{-5}$) gives an add-direction $\delta(0) = 3.830792\text{e-}03$, against the exact 3.829249e-03.* *(Change of measure)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The swapped loss.** For the pair $(Q, P)$ the privacy loss is $\ln\frac{Q(y)}{P(y)} = -L(y)$, and its PLD is the law of $-L(Y)$ with $Y \sim Q$.
2. **Split $Q$ by the support of $P$.** Where $P(y) > 0$, Fact 1 of the chapter gives $Q(y) = e^{-L(y)}P(y)$. So for any set $A$ of finite losses, $Q[L \in A] = \int_Ae^{-\ell}\omega(d\ell)$, and $-L$ takes the value $-\ell$ with that weight.
3. **The rest of $Q$.** The $Q$-mass where $P(y) = 0$ is $1 - Q(P > 0) = 1 - \mathbb E_\omega[e^{-L}]$. There $Q/P = \infty$, so the swapped loss is $+\infty$.
4. **The subsampled Gaussian.** The add pair $(\mathcal N(0, \sigma^2), \text{mixture})$ is the remove pair (mixture, $\mathcal N(0, \sigma^2)$) swapped. By steps 2 and 3 its PLD is the remove PLD tilted by $e^{-\ell}$ and reflected. One accountant run can serve both directions.
5. **The Gaussian pair.** Tilting $\mathcal N(\frac{\mu^2}2, \mu^2)$ by $e^{-\ell}$ and normalizing gives $\mathcal N(\frac{\mu^2}2 - \mu^2, \mu^2) = \mathcal N(-\frac{\mu^2}2, \mu^2)$: complete the square in $-\frac{(\ell - m)^2}{2v} - \ell = -\frac{(\ell - m + v)^2}{2v} + \text{const}$. Here $\mathbb E[e^{-L}] = e^{-m + v/2} = 1$, so the tilt needs no normalization and there is no atom. Reflecting gives $\mathcal N(\frac{\mu^2}2, \mu^2)$ again.

**Concretely.** The script applies steps 2 and 3 to the discretized PLDs (@tbl-sol-num-swap).

| Pair and value | From the swapped PLD | Discretized directly | Exact |
| --- | --- | --- | --- |
| Gaussian $\mu = 1$, $\delta(1)$, $h = 10^{-3}$ | 0.127282 | 0.127028 | 0.126937 |
| Subsampled, add, $\delta(0)$, $h = 10^{-5}$ | 3.830792e-03 | 3.832688e-03 | 3.829249e-03 |
| Subsampled, add, $\delta(0.005)$, $h = 10^{-5}$ | 9.660230e-04 | 9.652622e-04 | 9.631443e-04 |

: δ computed from the tilted and reflected PLD, from a direct discretization of the same pair, and exactly. {#tbl-sol-num-swap}

The rounded-up remove PLD has $\mathbb E[e^{-L}] = 0.999995$, so the swap puts $5.0\text{e-}06$ at $+\infty$. That mass is not in the true add PLD. It is the price of rounding up, and it keeps the swapped values above the exact ones, as a pessimistic estimate should be.

*The PLD under $Q$ carries no information beyond the PLD under $P$. This is why a single distribution, together with the atom at $+\infty$, describes a pair completely for $(\varepsilon, \delta)$.*
:::

## δ from tail probabilities {#sec-sol-num-2}

> **Exercise 2.** **δ from tail probabilities.** Show that $\delta(\varepsilon) \le P[L > \varepsilon]$ and that $\delta(\varepsilon) \ge (1 - e^{\varepsilon - \varepsilon'})\,P[L > \varepsilon']$ for every $\varepsilon' > \varepsilon$. *Check: for the Gaussian pair with $\mu = 1$ at $\varepsilon = 1$, the upper bound is 0.3085, δ is 0.1269, and the best lower bound is 0.0624, at $\varepsilon' = 1.504$.* *(Monotone weight, Markov-style bounds)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The weight is between 0 and an indicator.** $g(\ell) = (1 - e^{\varepsilon - \ell})_+$ is 0 for $\ell \le \varepsilon$ and in $(0, 1]$ for $\ell > \varepsilon$. So $g(\ell) \le \mathbf 1\{\ell > \varepsilon\}$.
2. **Upper bound.** Average step 1 over $\omega$: $\delta(\varepsilon) = \mathbb E[g(L)] \le P[L > \varepsilon]$.
3. **Lower bound.** $g$ is non-decreasing, so for $\ell > \varepsilon'$, $g(\ell) \ge g(\varepsilon') = 1 - e^{\varepsilon - \varepsilon'}$. Since $g \ge 0$ everywhere, $g(\ell) \ge (1 - e^{\varepsilon - \varepsilon'})\mathbf 1\{\ell > \varepsilon'\}$.
4. **Average.** $\delta(\varepsilon) \ge (1 - e^{\varepsilon - \varepsilon'})P[L > \varepsilon']$. The first factor grows with $\varepsilon'$ and the second shrinks, so there is a best $\varepsilon'$.

**Concretely,** with $L \sim \mathcal N(0.5, 1)$, $P[L > \varepsilon] = \Phi(0.5 - \varepsilon)$ (@tbl-sol-num-tail).

| $\varepsilon$ | $P[L > \varepsilon]$ | δ(ε) | Best lower bound | at $\varepsilon'$ |
| --- | --- | --- | --- | --- |
| 1 | 0.3085 | 0.1269 | 0.0624 | 1.504 |
| 3 | 0.0062 | 0.0015 | 0.0007 | 3.281 |

: The tail bounds of Exercise 2 for the Gaussian pair with $\mu = 1$. {#tbl-sol-num-tail}

*Both bounds are off by a factor of 2 to 4 here, and the upper one gets worse as ε grows; neither is tight. The weight $(1 - e^{\varepsilon - \ell})_+$ matters, which is why accountants evaluate @eq-num-delta and not a tail probability.*
:::

## The sandwich width {#sec-sol-num-3}

> **Exercise 3.** **The sandwich width.** Show that, without truncation, $\varepsilon^\uparrow(\delta) - \varepsilon^\downarrow(\delta) \le kh$, with equality when no loss lies on the grid and $\varepsilon^\downarrow(\delta) > 0$. When is the width zero? *Check: for three-fold randomized response with $\varepsilon_0 = \ln 3$ at $\delta = 0.3$, the width is 1.5000 for $h = 0.5$, 0.3000 for $h = 0.1$, and 0 for $h = \ln 3/2$.* *(Coupling)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Couple the roundings.** Draw the true per-step losses $L_1, \dots, L_k$ and round each up and down. Since $\lceil x\rceil - \lfloor x\rfloor \le 1$, $L_i^\uparrow \le L_i^\downarrow + h$, so the totals satisfy $S^\uparrow \le S^\downarrow + kh$.
2. **Shift δ.** Write $g_\varepsilon(\ell) = (1 - e^{\varepsilon - \ell})_+$. Then $g_\varepsilon(s + c) = g_{\varepsilon - c}(s)$. With step 1 and monotonicity, $\delta^\uparrow(\varepsilon) = \mathbb E[g_\varepsilon(S^\uparrow)] \le \mathbb E[g_\varepsilon(S^\downarrow + kh)] = \delta^\downarrow(\varepsilon - kh)$.
3. **Invert.** At $\varepsilon = \varepsilon^\downarrow(\delta) + kh$, step 2 gives $\delta^\uparrow \le \delta^\downarrow(\varepsilon^\downarrow(\delta)) \le \delta$, so $\varepsilon^\uparrow(\delta) \le \varepsilon^\downarrow(\delta) + kh$.
4. **Equality.** If no loss lies on the grid, $\lceil x\rceil - \lfloor x\rfloor = 1$ for every loss, so $S^\uparrow = S^\downarrow + kh$ exactly, step 2 is an equality, and the two δ curves are shifts of each other. Their crossing points with level δ differ by exactly $kh$, unless the lower one is cut off at $\varepsilon = 0$.
5. **Zero width.** If every loss is on the grid, both roundings return the true PLD and the width is 0. For randomized response, the losses $\pm\ln 3$ are on the grid when $h = \ln 3/j$ for an integer $j$.

**Concretely,** for three-fold randomized response (@tbl-sol-num-width).

| $h$ | $\mathbb E[e^{-L}]$, up | $\mathbb E[e^{-L}]$, down | $\varepsilon^\uparrow$ | $\varepsilon^\downarrow$ |
| --- | --- | --- | --- | --- |
| 0.5 | 0.8469 | 1.3963 | 3.2583 | 1.7583 |
| 0.1 | 0.9292 | 1.0270 | 2.0583 | 1.7583 |
| $\ln 3/2 = 0.5493$ | 1.0000 | 1.0000 | 2.0541 | 2.0541 |

: Three-fold randomized response with $\varepsilon_0 = \ln 3$: validity of the rounded PLDs and $\varepsilon$ at $\delta = 0.3$ for three grids. {#tbl-sol-num-width}

The widths are $3.2583 - 1.7583 = 1.5$ and $2.0583 - 1.7583 = 0.3$, that is $3h$, and 0 on the grid that contains $\pm\ln 3$, where both sides equal the true value 2.0541. The optimistic value 1.7583 is the same for $h = 0.5$ and $h = 0.1$. At $\delta = 0.3$ only the top atom $j = 3$ matters, and both grids round each step's $\ln 3$ down to 1.0, so the top total is 3.0 in both: $0.4219\,(1 - e^{\varepsilon - 3}) = 0.3$ gives $\varepsilon = 1.7583$. The subsampled Gaussian (remove direction) shows the same width on finer grids: $kh = 0.1000$ for $h = 10^{-3}$, $k = 100$, and 1.0000 for $k = 1000$; 0.0100 and 0.1000 for $h = 10^{-4}$.

*The sandwich width is known before you run the accountant. Choose $h$ from it.*
:::

## Wraparound {#sec-sol-num-4}

> **Exercise 4.** **Wraparound.** Show that the $k$-th power of the discrete Fourier transform on a circular window of $N$ bins computes the PLD of the total loss reduced modulo $Nh$ into the window, and that the error in $\delta(\varepsilon)$ is at most the mass of the composed PLD outside the window. Use Chebyshev's inequality to choose a window for ten Gaussian steps with $\mu = 1$. *Check: a window of width 20 around the mean has mass $1.57\text{e-}03$ outside it (Chebyshev: 0.1000), and an actual δ(1) error of $9.27\text{e-}06$.* *(Circular convolution, Chebyshev)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Circular convolution.** By Fact 3 of the chapter, multiplying transforms of length $N$ convolves with indices added modulo $N$. Applied $k - 1$ times, total index $n$ lands at $n \bmod N$. Placing the window at grid points $s, \dots, s + N - 1$ means each total loss $\ell$ lands at the unique point of the window congruent to $\ell$ modulo $Nh$.
2. **Mass inside the window is untouched.** A total that is already in the window is its own residue, so its mass is in the right place.
3. **Error bound.** Write $\delta = \sum_\ell m_\ell\,g(\ell)$ with $0 \le g \le 1$. Wraparound moves only the mass $m_{\text{out}}$ outside the window, and each moved unit changes its weight $g$ by at most 1 in absolute value. So $|\delta_{\text{circ}} - \delta| \le m_{\text{out}}$.
4. **Chebyshev.** The composed PLD has mean $k\mu^2/2 = 5$ and variance $k\mu^2 = 10$. For a window of width $W$ centred at the mean, $P[|S - 5| \ge W/2] \le 10/(W/2)^2$. For this to be at most $\tau$ you need $W \ge 2\sqrt{10/\tau}$. For an error of at most $\tau = 10^{-5}$, that is $W \ge 2\sqrt{10/10^{-5}} = 2000$, a hundred times wider than the width 20 that already gives an error of $9.27\text{e-}06$ (@tbl-sol-num-wrap). Chebyshev is safe but very wasteful here.

**Concretely,** with $h = 10^{-3}$ (@tbl-sol-num-wrap):

| Window width | Mass outside (exact) | Chebyshev bound | Actual δ(1) error |
| --- | --- | --- | --- |
| 10 | 1.14e-01 | 0.4000 | 1.59e-02 |
| 20 | 1.57e-03 | 0.1000 | 9.27e-06 |
| 30 | 2.10e-06 | 0.0444 | 1.81e-08 |
| 40 | 2.54e-10 | 0.0250 | 2.86e-12 |

: Ten Gaussian steps with $\mu = 1$ composed on circular windows centred at the mean: mass outside the window, its Chebyshev bound, and the actual error in δ(1). {#tbl-sol-num-wrap}

Every actual error is below the mass outside the window, as step 3 requires, and usually far below it. Mass wrapped from one tail partly offsets mass wrapped from the other, and much of it lands where the weight $g$ changes little. Chebyshev is very loose for a Gaussian, whose tails fall like $e^{-x^2/2}$, but it needs only the variance, which is $k$ times the one-step variance for any PLD.

*A window that is too short gives a wrong answer with no warning. Pad, or size the window from a tail bound.*
:::

## Heterogeneous composition {#sec-sol-num-5}

> **Exercise 5.** **Heterogeneous composition.** After 1000 DP-SGD steps ($q = 0.01$, $\sigma = 1$), you release one more statistic with the Gaussian mechanism at $\sigma = 10$. Compute $\varepsilon$ at $\delta = 10^{-5}$ for the whole release with PLDs, and explain why you cannot simply add the two values of $\varepsilon$. *Check: the bounds are $[1.8649, 1.8749]$ on the grid $h = 10^{-5}$, against $[1.8232, 1.8332]$ without the final release.* *(Convolution of different PLDs)*

::: {.callout-tip collapse="true" title="Solution"}
1. **PLD of the final release.** With sensitivity 1 and $\sigma = 10$, the Gaussian PLD is $\mathcal N(\mu^2/2, \mu^2)$ with $\mu = 0.1$. It is the same for both directions (Exercise 1).
2. **Compose.** For each direction, convolve the 1000-step DP-SGD PLD with the Gaussian PLD, both rounded the same way (@eq-num-conv; the rounding argument of @eq-num-sandwich holds for any mix of PLDs).
3. **Read off $\varepsilon$.** Solve $\delta(\varepsilon) = 10^{-5}$ on each result and take the larger direction (@tbl-sol-num-hetero).

| Rounding | Remove | Add | Larger |
| --- | --- | --- | --- |
| Pessimistic | 1.8749 | 1.4836 | 1.8749 |
| Optimistic | 1.8649 | 1.4736 | 1.8649 |

: ε at $\delta = 10^{-5}$ for 1000 DP-SGD steps followed by one Gaussian release with $\sigma = 10$. {#tbl-sol-num-hetero}

4. **Why not add values of $\varepsilon$.** Adding $(\varepsilon_1, \delta_1)$ and $(\varepsilon_2, \delta_2)$ is basic composition. It would need a δ for each part, and the Gaussian's loss is unbounded, so on its own it needs $\varepsilon = 0.3407$ at $\delta = 10^{-5}$, while the convolution adds only $1.8749 - 1.8332 = 0.0417$. The convolution instead adds a loss with mean $0.005$ and standard deviation 0.1 to the DP-SGD loss, which barely moves the tail. On the pessimistic side the final release costs 0.0417 in $\varepsilon$, about an eighth of its stand-alone 0.3407.

*PLDs compose any mix of mechanisms exactly; only their discretization costs anything.*
:::
