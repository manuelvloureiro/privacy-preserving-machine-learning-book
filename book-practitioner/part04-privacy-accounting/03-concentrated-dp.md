# Concentrated differential privacy

## Why this matters

Suppose you run a pipeline of Gaussian releases, such as noisy counts or the noisy marginals behind a synthetic dataset. You would like to keep its differential privacy (DP) budget as a single number that you add to after each release. [Rényi DP](01-renyi-dp.md) describes each mechanism by a whole curve, its **Rényi differential privacy (RDP) curve**, with one value for each order of the Rényi divergence. The Gaussian mechanism's curve is a straight line through the origin, so one number, its slope, carries all of it.

**Zero-concentrated differential privacy (zCDP)** is the bookkeeping built on that number, and it is the most widely used form of concentrated differential privacy (CDP). It is the right ledger for releases built from Gaussian noise on the full data. It is the wrong ledger for training on subsampled minibatches, because it throws away almost all of the privacy that subsampling buys.

## The zCDP ledger {#sec-cdp-definition}

A mechanism is zCDP when its RDP curve lies under a line through the origin. To state it, write $D_\alpha$ for the Rényi divergence of order $\alpha$, and call two datasets neighbours when they differ in one person's record. Then a mechanism $M$ is **$\rho$-zCDP** if, for all neighbouring datasets and all orders above 1,
$$D_\alpha\big(M(D)\,\|\,M(D')\big) \le \rho\alpha.$$ {#eq-cdp-def}
Two properties make this a ledger. Composition adds the slopes, even when each step is chosen after seeing the earlier outputs, and processing an output never raises its $\rho$.

The Gaussian mechanism is the case the definition was built for. Write $\sigma$ for the standard deviation of its noise and $\Delta$ for the query's $\ell_2$ sensitivity, the most one record can move the query's value. Its RDP curve is exactly a line, so one step costs
$$\rho = \frac{\Delta^2}{2\sigma^2}.$$
With the noise multiplier convention, which measures $\sigma$ in units of the sensitivity, a step with $\sigma = 1$ costs $0.5$. Doubling the noise multiplier cuts the cost to a quarter.

The best-known user is the 2020 United States Census TopDown algorithm (Abowd et al. 2022). It answers many queries with the discrete Gaussian mechanism ([Discrete and floating-point-safe mechanisms](../part03-dp-foundations/08-discrete-mechanisms.md)), which keeps counts as integers. Among libraries, Tumult Analytics offers a `RhoZCDPBudget`. In OpenDP, `dp.c.make_zCDP_to_approxDP` turns a zCDP measurement into one with a privacy profile, and `dp.c.make_fix_delta` then fixes the slack so that you can read off the privacy level.

The Census example carries two warnings. First, $\rho$ is not the privacy parameter $\varepsilon$ of an $(\varepsilon, \delta)$ guarantee, where $\delta$ is the slack. A $0.5$-zCDP step converts to about $\varepsilon = 4.7$ at $\delta = 10^{-5}$ (@sec-cdp-conversion), so a reader who mistakes one for the other badly overrates the protection. Second, $\rho$ covers only what passed through the noise. The Census Bureau released some statistics exactly, as a matter of policy, and these **invariants** sit outside the guarantee. A report should list such exact releases next to the budget.

## Charging other kinds of step {#sec-cdp-pure}

A real pipeline mixes mechanisms, and @tbl-cdp-charge gives the charge in $\rho$ for the common ones.

| Step | Charge in $\rho$ | Notes |
| --- | --- | --- |
| Gaussian, sensitivity $\Delta$, noise $\sigma$ | $\Delta^2/(2\sigma^2)$ | Exact |
| Any pure $\varepsilon$-DP step | $\varepsilon^2/2$, or sharply $\varepsilon\tanh(\varepsilon/2)$ | Poor fit once $\varepsilon \ge 2$ |
| Exponential mechanism at $\varepsilon$ | $\varepsilon^2/8$ | Four times cheaper |
| An $(\varepsilon, \delta)$-DP step | $\varepsilon^2/2$, plus $\delta$ in a second column | Approximate zCDP, @sec-cdp-group |
| Any step, for a group of $m$ records | $m^2$ times the step's $\rho$ | Exact for the Gaussian |
| Poisson-subsampled Gaussian | $1/(2\sigma^2)$, whatever the sampling rate | No amplification, @sec-cdp-tcdp |

: What each kind of step costs in a zCDP ledger. {#tbl-cdp-charge}

A pure $\varepsilon$-DP step, such as a Laplace count or a randomized-response bit, changes the probability of any output by at most a factor $e^\varepsilon$ between neighbours. The sharp charge $\varepsilon\tanh(\varepsilon/2)$ is the smallest that holds for all such steps, because randomized response is the worst case, and at $\varepsilon = 1$ it is 0.4621. The exponential mechanism's charge of $\varepsilon^2/8$ is due to Cesar and Rogers (2021).

The general rule breaks down for large $\varepsilon$. A pure DP step's RDP curve never rises above $\varepsilon$, while the line keeps climbing and passes that cap at order $2/\varepsilon$. So a $5$-DP step charged $\varepsilon^2/2 = 12.5$ sits above its own cap at every order. Converting that charge back gives a worse $\varepsilon$ than the 5 you started with, so keep such steps in RDP, where an accountant can charge them the exact curve of randomized response.

## From ρ to (ε, δ) {#sec-cdp-conversion}

You always report $(\varepsilon, \delta)$ in the end. Write $\ell$ for $\ln(1/\delta)$. Applying [Mironov's conversion](02-rdp-conversions.md) to a line gives the **simple conversion** of Bun and Steinke (2016). It says that a $\rho$-zCDP mechanism is $(\varepsilon, \delta)$-DP with
$$\varepsilon = \rho + 2\sqrt{\rho\,\ell}.$$ {#eq-cdp-simple}
The **improved conversion** of Canonne, Kamath and Steinke (2020) adds two negative correction terms at each order. It has no closed form, so you minimize it numerically, as the code below does. For the Gaussian the true $\varepsilon$ is also known, from its privacy profile in [f-DP and Gaussian differential privacy](04-gaussian-dp.md). @tbl-cdp-conversion compares the three.

| $\rho$ | Simple @eq-cdp-simple | Improved | Exact, Gaussian with the same $\rho$ |
| --- | --- | --- | --- |
| 0.1 | 2.246 | 1.914 | 1.760 |
| 0.5 | 5.299 | 4.728 | 4.377 |
| 1 | 7.786 | 7.077 | 6.573 |
| 5 | 20.174 | 19.047 | 17.857 |

: $\varepsilon$ at $\delta = 10^{-5}$ for a $\rho$-zCDP mechanism: the simple and improved conversions, and the exact $\varepsilon$ of the Gaussian mechanism with the same $\rho$. {#tbl-cdp-conversion}

At $\rho = 0.5$, the simple formula overshoots the truth by 21%, and the improved one by 8%. So use the improved conversion, or the exact Gaussian profile when every step is Gaussian.

For calibration you run the conversion backwards, from a target $(\varepsilon, \delta)$ to the largest total $\rho$ that meets it. Solving the simple conversion for $\rho$ shows that the target is met by any
$$\rho \le \Big(\sqrt{\varepsilon + \ell} - \sqrt{\ell}\Big)^2.$$ {#eq-cdp-inverse}
Then split $\rho$ over the queries. If query number $i$ has $\ell_2$ sensitivity $\Delta_i$ and spends $\rho_i$, its noise needs the standard deviation
$$\sigma_i = \frac{\Delta_i}{\sqrt{2\rho_i}}.$$
For example, a release that must be $(1, 10^{-5})$-DP may spend at most $\rho = 0.02082$. A single query of sensitivity 1 then needs $\sigma = 4.901$. If you split the same budget equally over four queries, each gets a quarter of $\rho$ and so needs twice the noise, 9.801.

That recipe works because the split happens in $\rho$, which adds, and never in $\varepsilon$, which does not. Suppose you have a budget of $\rho = 1$ and have spent half of it. Each half alone converts to 4.728, and two of those are far more than the whole run's 7.077, because the conversion grows like the square root of $\rho$. So keep budgets in $\rho$, subtract in $\rho$, and convert once, at the end. In OpenDP, `dp.binary_search_param` does the last step of calibration for you, by searching for the noise scale that meets a target. Use the $\ell_2$ sensitivity here, because the $\ell_1$ sensitivity of a vector query is never smaller and so overpays in noise.

## Groups and approximate zCDP {#sec-cdp-group}

A household, a user with many records or a hospital contributes several records, so you may need the guarantee for a group. Take a $\rho$-zCDP mechanism $M$ and two datasets that differ in $m$ records. Bun and Steinke (2016) show that
$$D_\alpha\big(M(D)\,\|\,M(D')\big) \le m^2\rho\,\alpha.$$ {#eq-cdp-group}
The square is exact for the Gaussian, since a group shifts the query $m$ times as far. For example, a user with 4 records in a $0.1$-zCDP release is protected at $\rho = 1.6$. Plain RDP's group bound is much weaker ([Rényi DP](01-renyi-dp.md)).

Some useful mechanisms, such as propose-test-release and stability-based histograms, are $(\varepsilon, \delta)$-DP but cannot be zCDP at all. **Approximate zCDP** keeps them in the ledger by adding a second column. Each such step adds $\varepsilon^2/2$ to $\rho$ and its slack to the second column, and at the end the second column's total is added to the $\delta$ of the conversion.

## Subsampled steps: keep them out of zCDP {#sec-cdp-tcdp}

Differentially private stochastic gradient descent (SGD), DP-SGD for short, runs each step on a Poisson sample of the data. Each record enters the batch independently with probability $q$, the sampling rate, and this makes a step far more private than the Gaussian mechanism inside it ([Amplification by subsampling](05-subsampling.md)). But zCDP cannot see the gain. The subsampled step's RDP curve is tiny at small orders and bends up towards the unsubsampled line at large ones. A line that covers the whole curve must therefore have slope $1/(2\sigma^2)$, whatever the sampling rate.

The cost of ignoring this is enormous. On this part's running example of 10,000 steps with $q = 0.01$ and $\sigma = 1$, zCDP reports $\varepsilon = 5475.8$. RDP accounting of the same run gives 6.719, because the conversion only uses the small orders where the curve is still tiny.

**Truncated concentrated differential privacy (tCDP)** asks for the line only on low orders, so it sees some of the amplification. It is rarely used, because it does no better than RDP.

So if a pipeline mixes Gaussian releases with a DP-SGD run, keep all of it in RDP. A $\rho$-zCDP step is just the RDP line $\rho\alpha$, so it joins an RDP ledger without loss. In Google's `dp_accounting` library, for example, you add it to an `RdpAccountant` as a `GaussianDpEvent` with noise multiplier $1/\sqrt{2\rho}$. The training steps go in next to it as a `PoissonSampledDpEvent`.

## Code {#sec-cdp-code}

The whole zCDP ledger is a few lines of code. You charge each Gaussian step its $\rho$, add the charges, and convert once, and the last line splits a budget over four queries.

```python
from math import log, sqrt


def zcdp_gaussian(sigma: float, sensitivity: float = 1.0) -> float:
    return sensitivity**2 / (2 * sigma**2)


def zcdp_to_eps_improved(rho: float, delta: float) -> float:
    """Minimized over 20,001 log-spaced orders, the full edition's fine grid."""
    best = float("inf")
    for i in range(20001):
        alpha = 1 + 10 ** (-4 + 8 * i / 20000)
        eps = rho * alpha + log(1 / delta) / (alpha - 1) + log(1 - 1 / alpha) - log(alpha) / (alpha - 1)
        best = min(best, eps)
    return best


def rho_for_target(eps: float, delta: float) -> float:
    """Largest rho whose simple conversion stays within eps at delta."""
    return (sqrt(eps + log(1 / delta)) - sqrt(log(1 / delta))) ** 2


def sigma_for_rho(rho: float) -> float:
    return 1 / sqrt(2 * rho)


rho = 10 * zcdp_gaussian(sigma=1.0)  # ten Gaussian steps with sigma = 1
print(f"{rho:.1f}")  # 5.0
print(f"{zcdp_to_eps_improved(rho, 1e-5):.3f}")  # 19.047
budget = rho_for_target(1.0, 1e-5)  # the whole release must be (1, 1e-5)-DP
print(f"{budget:.5f} {sigma_for_rho(budget):.3f}")  # 0.02082 4.901
print(f"{sigma_for_rho(budget / 4):.3f}")  # 9.801
```

## Cheat sheet

Here $\ell$ stands for $\ln(1/\delta)$.

| Task | Formula | Notes |
| --- | --- | --- |
| Charge a Gaussian step | $\rho = \Delta^2/(2\sigma^2)$ | Exact |
| Convert | Improved conversion, or $\rho + 2\sqrt{\rho\ell}$ | 8% and 21% above exact at $\rho = 0.5$ |
| Calibrate | $\rho = (\sqrt{\varepsilon + \ell} - \sqrt\ell)^2$, then $\sigma_i = \Delta_i/\sqrt{2\rho_i}$ | Split the budget in $\rho$ |
| Protect a group of $m$ | $m^2\rho$ | Exact for the Gaussian |
| Account a subsampled step | Not in zCDP | Use RDP or a privacy loss distribution |

: The tasks of this chapter; @tbl-cdp-charge gives the charge of each kind of step. {#tbl-cdp-cheat}

**Rules of thumb**

- Keep budgets in $\rho$, which adds, and convert to $(\varepsilon, \delta)$ once, at the end, with the improved conversion.
- Never report $\rho$ as if it were $\varepsilon$. A $0.5$-zCDP step is about $4.7$-DP at $\delta = 10^{-5}$.
- Report the conversion you used, and any statistics released exactly.
- Calibrate Gaussian noise with the $\ell_2$ sensitivity, not the $\ell_1$.
- Charge a pure DP step $\varepsilon^2/2$ only for small $\varepsilon$. From about 2 up, keep it in RDP at its own $\varepsilon$.
- A pipeline with any DP-SGD in it belongs in RDP or a numerical accountant, not in zCDP.

## Further reading

- Bun & Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds" (TCC 2016): the definition and every rule of this chapter, in Sections 1–3.
- Canonne, Kamath & Steinke, "The Discrete Gaussian for Differential Privacy" (NeurIPS 2020): the discrete Gaussian and the improved conversion.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/03-concentrated-dp.md).
