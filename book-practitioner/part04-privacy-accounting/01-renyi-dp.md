# Rényi differential privacy

## Why this matters

Model training may use thousands of private stochastic gradient descent (SGD) steps, but its report gives one differential privacy (DP) guarantee, $(\varepsilon,\delta)$. Part 3's advanced composition can be too pessimistic ([Composition](../part03-dp-foundations/04-composition.md)). For example, 1,000 full-batch noisy gradient-descent steps, each with Gaussian noise 20 times the maximum per-record gradient change, give $\varepsilon = 68.01$ at $\delta = 10^{-5}$ under advanced composition, nine times the exact value 7.51. Calibration to this loose bound adds unnecessary noise.

**Rényi differential privacy (RDP)** tracks a curve of privacy costs for each step. The curves compose without spending $\delta$; convert the total to $(\varepsilon,\delta)$ once, after composition. Even Mironov's original conversion (@sec-rdp-conversion) gives 8.84 for this run, 18% above the exact value. Opacus and Google's `dp_accounting` provide RDP accountants.

## The RDP curve {#sec-rdp-definition}

A DP guarantee compares mechanism outputs on neighbouring datasets, which differ in one person's record. RDP measures the distance between the output distributions using the **Rényi divergence** from [Divergences](../part01-prerequisites/03-divergences.md). Its **order** $\alpha > 1$ controls sensitivity to rare outputs: low orders emphasize typical outputs, while high orders give more weight to outputs much likelier on one dataset.

A mechanism is **$(\alpha,\varepsilon)$-RDP** if its divergence is at most $\varepsilon$ at order $\alpha$ for every neighbouring pair. The smallest such value at each order defines its **RDP curve** $\varepsilon(\alpha)$; higher curves mean weaker privacy. The curve is nondecreasing. At infinite order it equals the largest output privacy loss, bounded by pure DP, where $\delta=0$. Accountants use known curves for standard mechanisms on a grid of orders; you do not need to compute divergences directly.

## RDP curves of the basic mechanisms {#sec-rdp-curves}

Accounting requires the mechanism's RDP curve. The basic mechanisms have closed-form curves (Mironov 2017). For the **Gaussian mechanism**, let $\sigma$ be its noise multiplier: the noise standard deviation divided by the query's $\ell_2$ sensitivity. Its curve is
$$\varepsilon(\alpha) = \frac{\alpha}{2\sigma^2}.$$
@tbl-rdp-curves gives the curves of the Gaussian, Laplace and randomized-response mechanisms at a few orders, each at sensitivity 1.

| Mechanism | $\varepsilon(1.5)$ | $\varepsilon(2)$ | $\varepsilon(10)$ | $\varepsilon(64)$ | $\varepsilon(\infty)$ |
| --- | --- | --- | --- | --- | --- |
| Gaussian, $\sigma = 1$ | 0.7500 | 1.0000 | 5.0000 | 32.0000 | $\infty$ |
| Laplace, scale $b = 1$ | 0.5129 | 0.6191 | 0.9287 | 0.9891 | 1 |
| Randomized response, which reports a bit truthfully with probability 0.75 | 0.7340 | 0.8473 | 1.0666 | 1.0940 | 1.0986 |

: RDP curves of the three basic mechanisms with sensitivity 1. {#tbl-rdp-curves}

The Laplace and randomized-response curves approach their pure-DP levels. The Gaussian curve has no finite limit, so the Gaussian mechanism is not pure $\varepsilon$-DP for any finite $\varepsilon$. Its curve remains finite at each finite order, where accountants perform conversions. These curves describe full-dataset steps. For DP-SGD, each step uses a random batch, and libraries compute the smaller subsampled curves ([Amplification by subsampling](05-subsampling.md)).

## Composition adds curves {#sec-rdp-composition}

Training composes mechanisms sequentially, often adaptively: each gradient step depends on the current model. RDP adds their curves order by order. Let $\varepsilon_i$ be the curve for step $i$. A run of $k$ steps has curve $\varepsilon_M$ satisfying
$$\varepsilon_M(\alpha) \le \sum_{i=1}^k \varepsilon_i(\alpha) \qquad\text{for every order } \alpha.$$
Different mechanisms can be mixed because composition adds their values at each common order.

The sum requires two conditions. All curves must use the same neighbouring relation, such as add/remove or replace-one, and the same privacy unit, such as one record or one user. Also, each step's curve must be fixed before the run. Steps may depend on previous outputs, but data-dependent noise or stopping makes the curve data-dependent. Early stopping based on validation performance therefore requires [Adaptive composition](08-adaptive-composition.md).

For Gaussian steps, each curve is the line $\alpha/(2\sigma^2)$. Adding $k$ such curves gives the cost of one step with noise multiplier $\sigma/\sqrt k$. Thus, $k$ full-dataset steps cost as much as one step with $\sqrt k$ times less noise. For 10,000 steps at noise multiplier 1, this is equivalent to one step at multiplier $1/\sqrt{10{,}000}=0.01$, whose $\varepsilon$ at $\delta=10^{-5}$ is 5425.5. DP-SGD reduces this cost by using small random batches. The running example uses 10,000 steps at multiplier 1, each sampling 1% of the data.

## Post-processing and group privacy {#sec-rdp-properties}

Post-processing is free in RDP: a computation on the output that does not access the data again cannot increase the curve. Releasing the trained model or its predictions therefore adds no cost beyond training.

The generic RDP group rule is loose. For Gaussian noise and a user with $m$ records, use the group's sensitivity, at most $m$ times the per-record sensitivity. The curve is
$$\varepsilon_{\mathrm{group}}(\alpha) = \frac{m^2\alpha}{2\sigma^2},$$
This is $m^2$ times the per-record curve; ten records per user therefore cost 100 times as much. Clip each user's total contribution to one norm and account per user ([User-level privacy](../part05-dp-ml/07-user-level.md)).

## Compose, then convert once {#sec-rdp-conversion}

After composition, convert the curve to the reported $(\varepsilon,\delta)$ pair. Both libraries use the **improved conversion** (Balle et al. 2020; independently, Canonne, Kamath and Steinke 2020). Given curve value $\varepsilon(\alpha)$ at order $\alpha$ and target $\delta$, the conversion gives
$$\varepsilon' = \varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}.$$
Report the smallest $\varepsilon'$ on the order grid, or 0 if it is negative. The minimizing order is the best order.

Older papers and code may use Mironov's original conversion, which gives 7.43 rather than 6.71 for the DP-SGD example. The original paper's **moments accountant** (Abadi et al. 2016) uses an RDP curve with that conversion. Its $\varepsilon$ is looser but valid ([From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md)).

## Check the order grid {#sec-rdp-orders}

A conversion can use only the orders on the grid of the accountant, so the grid must contain the best order. That order moves with the run, towards 1 for long runs with little noise and up for short runs with much noise. For the noisy gradient descent at the start of the chapter, it falls from 128 after one step to 4 after 1000.

The default grid of Opacus, `RDPAccountant.DEFAULT_ALPHAS`, stops at order 63, and that of `dp_accounting`, `DEFAULT_RDP_ORDERS`, at 1024. A best order at either end of the grid means that the true optimum is outside it, so give a longer list, with `alphas=[...]` in Opacus or `orders=[...]` in `dp_accounting`. Opacus warns you, but `dp_accounting` does not, so compare the returned order with the ends of the list yourself.

## Accounting with a library {#sec-rdp-library}

In practice, you describe each step to a library, and it composes and converts. In `dp_accounting`, a DP-SGD step is a `GaussianDpEvent` inside a `PoissonSampledDpEvent`, which samples each record independently, with probability 0.01 in the running example. This snippet composes the 10,000 steps and prints $\varepsilon$ with the best order.

```python
import dp_accounting

step = dp_accounting.PoissonSampledDpEvent(0.01, dp_accounting.GaussianDpEvent(1.0))
accountant = dp_accounting.rdp.RdpAccountant()
accountant.compose(step, count=10_000)
eps, order = accountant.get_epsilon_and_optimal_order(1e-5)
print(f"{eps:.4f} {order:g}")  # 6.7128 4.1
```

In Opacus, the accountant is attached to the optimizer, which calls `step` after every noisy update. The same calls by hand give the same $\varepsilon$ to the fourth decimal, at the same order, 4.1, which is well inside both grids.

```python
from opacus.accountants import RDPAccountant

accountant = RDPAccountant()
for _ in range(10_000):  # what the optimizer does at each step of training
    accountant.step(noise_multiplier=1.0, sample_rate=0.01)
eps, order = accountant.get_privacy_spent(delta=1e-5)
print(f"{eps:.4f} {order:g}")  # 6.7127 4.1
```

Opacus's defaults mix two accountants, so name the one that you want. In Opacus 1.6.0, `PrivacyEngine` uses `"prv"`, a [numerical accountant](06-numerical-accounting.md), by default, and `accountant="rdp"` selects this class. But `get_noise_multiplier`, which solves for the noise multiplier $\sigma$, uses RDP by default. Both give valid upper bounds, but different ones, and the numerical accountant gives 6.19 for the running example.

Two more checks catch most mistakes. The event must match the sampler of your data loader, because shuffled batches of fixed size get none of the amplification of a Poisson event ([Amplification by subsampling](05-subsampling.md#sec-sub-schemes)). And a manual call to `step` on top of the optimizer's own call charges each update twice.

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| RDP curve | $\varepsilon(\alpha)$ over orders $\alpha > 1$ | Never decreases; $\varepsilon(\infty)$ is pure DP |
| Gaussian, noise multiplier $\sigma$ | $\varepsilon(\alpha) = \alpha/(2\sigma^2)$ | $\sigma$ in units of the $\ell_2$ sensitivity |
| Composition | $\varepsilon_M(\alpha) \le \sum_i\varepsilon_i(\alpha)$ | Same relation and unit; curves fixed in advance |
| Group of $m$, Gaussian | $m^2\alpha/(2\sigma^2)$ | The generic group rule is loose |
| Improved conversion | $\varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}$, best $\alpha$ | What Opacus and `dp_accounting` use |

: The results of this chapter. {#tbl-rdp-cheat}

## References

- Mironov, "Rényi Differential Privacy" (IEEE CSF 2017): the definition, composition, the curves of the basic mechanisms and group privacy.
- Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016): DP-SGD and the moments accountant.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/01-renyi-dp.md).
