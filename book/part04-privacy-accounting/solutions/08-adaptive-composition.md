# Exercises: Adaptive composition

Exercises for [Adaptive composition: filters, odometers and individual accounting](../08-adaptive-composition.md), each with a worked solution. Try each exercise before opening its solution. The script `code/solutions_adaptive_composition.py` computes every number quoted here; run it with `uv run python book/part04-privacy-accounting/code/solutions_adaptive_composition.py`.

The solutions use the chapter's notation. Step $t$ releases $y_t$; $L_{1:t}$ is the privacy loss after $t$ steps, evaluated under $P = M(D)$; $\varepsilon_t(\alpha)$ is the Rényi differential privacy (RDP) spend of step $t$ at order $\alpha$, chosen from the earlier outputs; and $N_t = \exp\big((\alpha - 1)(L_{1:t} - \sum_{j\le t}\varepsilon_j(\alpha))\big)$ is the supermartingale of @eq-adapt-supermartingale. Guarantees are stated in differential privacy (DP), RDP and zero-concentrated differential privacy (zCDP).

## Stopping after one step {#sec-sol-adapt-1}

> **Exercise 1.** **Stopping after one step.** At $\alpha = 2$, run randomized response with $\varepsilon = \ln 3$ on one person's bit, and run a second, identical step only if $y_1 = 1$; after $y_1 = 0$ the run stops. List the three transcripts with their $P$, $Q$, spend and $N_T$, check that $\mathbb E[N_T] = 1$, and compute the joint $D_2$. *Check: $D_2 = 1.6740$, below $B = 1.6946$.* *(Optional stopping, Rényi filter)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The steps.** Randomized response with $\varepsilon = \ln 3$ reports the true bit with probability $e^{\ln 3}/(1 + e^{\ln 3}) = 3/4$. Under $P$ the bit is 1, under $Q$ it is 0. Its RDP at $\alpha = 2$ is $\ln\big((3/4)^2/(1/4) + (1/4)^2/(3/4)\big) = \ln(9/4 + 1/12) = 0.8473$ (@eq-rdp-rr).
2. **The stopping time.** The run takes step 2 only if $y_1 = 1$. That decision is a function of $y_1$, so $T$ is a stopping time (Fact 2 of the chapter), and the spend along a path is $2 \times 0.8473 = 1.6946$ if $y_1 = 1$ and $0.8473$ if $y_1 = 0$. Every path stays within $B = 1.6946$.
3. **The three transcripts.** Multiply the step probabilities along each path, and take $L = \ln(P/Q)$ and $N_T = e^{(2-1)(L - \text{spend})}$:

| Transcript | $P$ | $Q$ | $L$ | spend | $N_T$ |
| --- | --- | --- | --- | --- | --- |
| (1, 1) | 0.5625 | 0.0625 | +2.1972 | 1.6946 | 1.6531 |
| (1, 0) | 0.1875 | 0.1875 | +0.0000 | 1.6946 | 0.1837 |
| (0, stop) | 0.2500 | 0.7500 | −1.0986 | 0.8473 | 0.1429 |

: The stopped two-step run: randomized response with $\ln 3$, repeated only after a 1. {#tbl-sol-adapt-stop}

4. **Optional stopping.** $\mathbb E_P[N_T] = 0.5625 \times 1.6531 + 0.1875 \times 0.1837 + 0.25 \times 0.1429 = 1.0000$. For randomized response the conditional moment of each step equals $e^{(\alpha-1)\varepsilon_t(\alpha)}$ exactly, so $N$ is a martingale and stopping it keeps its mean at exactly 1.
5. **The divergence.** $D_2(P\|Q) = \ln\sum P^2/Q = 1.6740$, which is at most the budget 1.6946, as the Rényi filter (@eq-adapt-renyi-filter) promises. It is above the one-step value 0.8473, because the costly branch carries most of the mass under $P$.

*The table is the derivation of the Rényi filter in miniature: $\mathbb E[e^{L}] = \mathbb E[N_T e^{\text{spend}}] \le e^{B}\,\mathbb E[N_T] = e^{B}$, and the inequality is strict here only because one branch spends less than $B$.*
:::

## The zCDP route to the DP filter {#sec-sol-adapt-2}

> **Exercise 2.** **The zCDP route to the DP filter.** Show that the Whitehouse statistic @eq-adapt-whitehouse equals $\rho + 2\sqrt{\rho\ln(1/\delta)}$ with $\rho = \frac12\sum_t\varepsilon_t^2$, and that the filter allows $\lfloor 2\rho^*/\varepsilon_0^2\rfloor$ steps of $\varepsilon_0$. *Check: at $\varepsilon_g = 1$, $\delta = 10^{-6}$, $\rho^* = 0.017469$ and 349 steps of $0.01$.* *(zCDP conversion)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Substitute.** With $\rho = \frac12 V$ and $V = \sum_t\varepsilon_t^2$: $2\sqrt{\rho\ln(1/\delta)} = 2\sqrt{\frac12V\ln(1/\delta)} = \sqrt{2V\ln(1/\delta)}$. So $\rho + 2\sqrt{\rho\ln(1/\delta)} = \frac12V + \sqrt{2\ln(1/\delta)V}$, the left side of @eq-adapt-whitehouse.
2. **Where $\rho + 2\sqrt{\rho\ln(1/\delta)}$ comes from.** Mironov's conversion of the line $\rho\alpha$ is $\rho\alpha + \ln(1/\delta)/(\alpha - 1)$. With $a = \alpha - 1$ this is $\rho + \rho a + \ln(1/\delta)/a$, minimized at $a = \sqrt{\ln(1/\delta)/\rho}$ with value $\rho + 2\sqrt{\rho\ln(1/\delta)}$ (@eq-cdp-simple).
3. **Invert.** Write $\ell = \ln(1/\delta)$ and $u = \sqrt\rho$. Then $u^2 + 2u\sqrt\ell = \varepsilon_g$ is $(u + \sqrt\ell)^2 = \varepsilon_g + \ell$, so $u = \sqrt{\ell + \varepsilon_g} - \sqrt\ell$ and $\rho^* = (\sqrt{\ell + \varepsilon_g} - \sqrt\ell)^2$ (@eq-cdp-inverse). Because $\rho \mapsto \rho + 2\sqrt{\rho\ell}$ is increasing, the filter continues exactly while $\rho \le \rho^*$.
4. **Count steps.** $k$ steps of $\varepsilon_0$ have $\rho = k\varepsilon_0^2/2$, so $\rho \le \rho^*$ means $k \le 2\rho^*/\varepsilon_0^2$, and the largest such integer is $\lfloor 2\rho^*/\varepsilon_0^2\rfloor$.
5. **Numbers.** At $\delta = 10^{-6}$, $\ell = 13.8155$, and with $\varepsilon_g = 1$, $\rho^* = 0.017469$ (converting it back gives 1.0000). For $\varepsilon_0 = 0.01$, $2\rho^*/\varepsilon_0^2$ gives 349 steps. The statistic is 0.9994 at $k = 349$ and 1.0009 at $k = 350$, so the 350th step is refused. The conversion is attained at the order $1 + \sqrt{\ell/\rho^*} = 29.12$.

*Because the halting rule $\rho \le \rho^*$ does not mention $\alpha$, the same run is a Rényi filter at every order, which is what lets the conversion pick the order afterwards.*
:::

## The Ville boundary is crossed with probability exactly δ {#sec-sol-adapt-3}

> **Exercise 3.** **The Ville boundary is crossed with probability exactly $\delta$.** For Gaussian steps at a fixed order, write $L_{1:t}$ as Brownian motion with drift in variance time, and use $\Pr[\exists\,s: W_s \ge a + bs] = e^{-2ab}$ (derived from Ville's inequality in @sec-adapt-odometers) to show the fixed-order Ville boundary is crossed with probability $\delta$. Then show that at a single fixed time the crossing probability is at most $\Phi(-\sqrt{2\ln(1/\delta)})$. *Check: 0.00719 at $\delta = 0.05$.* *(Ville, Brownian motion)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One step's loss.** A Gaussian step with shift $\mu$ (in units of the noise) has privacy loss $L_t = \mu y_t - \mu^2/2$ with $y_t \sim \mathcal N(\mu, 1)$ under $P$, so $L_t \sim \mathcal N(\mu^2/2, \mu^2)$, independent across steps when $\mu$ is fixed.
2. **Variance time.** Let $s = t\mu^2$. Then $L_{1:t} = s/2 + W_s$, where $W$ is a standard Brownian motion observed at the times $\mu^2, 2\mu^2, \dots$ (sums of independent normals with variance $\mu^2$ each).
3. **The boundary.** At order $\alpha$, each step spends $\alpha\mu^2/2$ (@eq-rdp-gauss), so the Ville boundary with one order is $S_t(\alpha) + \ln(1/\delta)/(\alpha - 1) = \alpha s/2 + \ln(1/\delta)/(\alpha - 1)$.
4. **Crossing.** $L_{1:t}$ exceeds it when $W_s > (\alpha - 1)s/2 + \ln(1/\delta)/(\alpha - 1)$: a line $a + bs$ with $a = \ln(1/\delta)/(\alpha - 1)$ and $b = (\alpha - 1)/2$. In continuous time the crossing probability is $e^{-2ab} = e^{-\ln(1/\delta)} = \delta$, for every $\alpha$. Watching only at the discrete times can only miss crossings, so Ville's bound $\delta$ is attained in the limit of small steps. With $\alpha = 2$ the script's formula gives 0.05 at $\delta = 0.05$ and $10^{-5}$ at $\delta = 10^{-5}$, and the finite horizon $s = 1000$ changes neither at the printed precision.
5. **One fixed time.** At a fixed $s$, $\Pr[W_s > a + bs] = \Phi\big(-(a + bs)/\sqrt s\big)$. The exponent $(a + bs)/\sqrt s = a/\sqrt s + b\sqrt s$ is smallest at $s = a/b$, where it equals $2\sqrt{ab} = 2\sqrt{\ln(1/\delta)/2} = \sqrt{2\ln(1/\delta)}$. So the probability at any single time is at most $\Phi(-\sqrt{2\ln(1/\delta)})$: 0.00719 at $\delta = 0.05$ and $7.99\times10^{-7}$ at $\delta = 10^{-5}$.

*At a fixed time, Mironov's conversion overstates the tail by a factor of 7.0 at $\delta = 0.05$ and 12.5 at $\delta = 10^{-5}$. That slack is exactly what makes the same boundary valid at all times at once.*
:::

## A half-norm record {#sec-sol-adapt-4}

> **Exercise 4.** **A half-norm record.** A record whose clipped gradient has norm $C/2$ at every step, with $\sigma = 10$. Using @eq-adapt-individual-rdp, compute its spend after 100 steps at $\alpha = 5.8$, its $\varepsilon$ at $\delta = 10^{-5}$ at that order and optimized over the grid, and how many steps the individual filter with $B = 2.900$ allows it. *Check: spend 0.7250, $\varepsilon = 3.124$ at $\alpha = 5.8$, 400 steps.* *(Individual RDP)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Per-step spend.** By @eq-adapt-individual-rdp with $\|\bar g\| = C/2$: $\alpha(C/2)^2/(2\sigma^2C^2) = \alpha/(8\sigma^2)$, a quarter of the worst case $\alpha/(2\sigma^2)$. At $\alpha = 5.8$ and $\sigma = 10$ this is $0.007250$.
2. **After 100 steps.** The spends add (the norms here do not even depend on the past): $100 \times 0.007250 = 0.7250$.
3. **$\varepsilon$ at the fixed order.** Mironov's conversion (@eq-rdp-mironov): $0.7250 + \ln(10^5)/4.8 = 3.124$, against the worst case 5.299 at the same order.
4. **$\varepsilon$ with the order optimized.** The record's curve is $\alpha \times 100/(8\sigma^2)$, a zCDP line with $\rho = 0.125$; the best order on the grid is 10.6 and gives 2.524. That number would be a guarantee only for a run whose filter was set at order 10.6.
5. **Steps until the filter excludes it.** The budget $B = 2.900$ is the worst-case spend of 100 steps at $\alpha = 5.8$. At $0.007250$ per step, the record can take part in $2.900/0.007250 = 400$ steps, four times as many as a record always at the clip.

*A record with half-size gradients can take part in four times as many steps for the same guarantee; the spend scales with the square of the norm.*
:::

## The price of choosing the order late {#sec-sol-adapt-5}

> **Exercise 5.** **The price of choosing the order late.** For a zCDP spend $\rho = 1$ at $\delta = 10^{-5}$, compare the fixed-time conversion over the part's grid with the Ville odometer over that grid, over the ten orders $\{1.5, 2, 3, 4, 6, 8, 12, 16, 32, 64\}$, and at the single order that is optimal for $\rho = 0.25$. *Check: 7.786, 9.137, 8.605 and 9.493.* *(Union bound, Ville)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Fixed time.** For the curve $\alpha\rho$, Mironov's conversion over the grid is $\min_\alpha[\alpha + \ln(10^5)/(\alpha - 1)] = 7.786$, at $\alpha = 4.4$. Over all real orders it is $\rho + 2\sqrt{\rho\ln(1/\delta)}$ (Exercise 2, step 2), also 7.786: the grid is fine enough.
2. **Ville over the grid.** @eq-adapt-ville-odometer replaces $\ln(1/\delta)$ by $\ln(|\Lambda|/\delta)$ with $|\Lambda| = 154$, which is the conversion at $\delta/154$. That gives 9.137; the continuous formula $\rho + 2\sqrt{\rho\ln(154/\delta)}$ gives 9.136.
3. **Ville over ten orders.** With $|\Lambda| = 10$ the union costs only $\ln 10$ instead of $\ln 154$, but the best order on the coarse grid is a little off the optimum. The result, 8.605, is still better than the fine grid.
4. **One order, chosen for the wrong spend.** For $\rho = 0.25$ the best grid order is 7.8. A Ville odometer at that order alone pays no union, but at $\rho = 1$ it gives $7.8 + \ln(10^5)/6.8 = 9.493$, worse than both grids.
5. **The trade-off.** A single order is free at its design point and poor elsewhere; a fine grid adapts to any spend but pays $\ln|\Lambda|$. A coarse grid, a few orders per factor of two in spend, gets most of both.

*The cost of an odometer over a filter is the union bound over the orders you keep open; Rogers et al.'s lower bound says some such $\ln\ln$-type cost is unavoidable.*
:::
