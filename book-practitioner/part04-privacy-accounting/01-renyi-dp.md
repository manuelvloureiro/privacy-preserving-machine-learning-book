# Rényi differential privacy

## Why this matters

Training a model runs thousands of private steps, but you report one differential privacy (DP) guarantee for all of them, a privacy level $\varepsilon$ with a small slack $\delta$. The advanced composition of Part 3 adds up the pairs of the steps, and the result is far too pessimistic ([Composition](../part03-dp-foundations/04-composition.md)). Take 1000 steps of full-batch noisy gradient descent, each with Gaussian noise whose standard deviation is 20 times the most one record can move the gradient. At $\delta = 10^{-5}$, advanced composition gives $\varepsilon = 68.01$, nine times the exact 7.51. So a run calibrated to it adds more noise than it needs.

**Rényi differential privacy (RDP)** fixes this, because it keeps a whole curve of privacy costs for each step instead of one point. The curves of the steps add, with no $\delta$ spent along the way, and you convert the total into $(\varepsilon, \delta)$ once, at the end. Even with Mironov's original conversion (@sec-rdp-conversion), the same run then costs 8.84, only 18% above the exact value. Opacus and Google's `dp_accounting` both have an RDP accountant.

## The RDP curve {#sec-rdp-definition}

A DP guarantee compares the outputs of a mechanism on two neighbouring datasets, which differ in one person's record. RDP measures the distance between those two output distributions with the **Rényi divergence** of [Divergences](../part01-prerequisites/03-divergences.md). That divergence has a dial, the **order** $\alpha$, which is any number above 1. A small order looks at the typical output. A large order gives more weight to the rare outputs that are much more likely on one dataset than on the other.

A mechanism is **$(\alpha, \varepsilon)$-RDP** if, at order $\alpha$, the divergence is at most $\varepsilon$ for every pair of neighbours. The smallest such $\varepsilon$ at each order gives the **RDP curve** $\varepsilon(\alpha)$ of the mechanism, and a higher curve means less privacy. The curve never decreases with the order. At order infinity, it is the largest privacy loss of any output, which is what pure DP, the case $\delta = 0$, bounds. You never compute the divergence yourself, because accountants know the curves of the standard mechanisms on a grid of orders.

## RDP curves of the basic mechanisms {#sec-rdp-curves}

To account for a mechanism, you need its curve, and the basic mechanisms have one in closed form (Mironov 2017). For the **Gaussian mechanism**, write $\sigma$ for its noise multiplier. That is the standard deviation of the noise in units of the $\ell_2$ sensitivity, the most one record can move the answer. Its curve is the straight line
$$\varepsilon(\alpha) = \frac{\alpha}{2\sigma^2}.$$
@tbl-rdp-curves gives the curves of the Gaussian, Laplace and randomized-response mechanisms at a few orders, each at sensitivity 1.

| Mechanism | $\varepsilon(1.5)$ | $\varepsilon(2)$ | $\varepsilon(10)$ | $\varepsilon(64)$ | $\varepsilon(\infty)$ |
| --- | --- | --- | --- | --- | --- |
| Gaussian, $\sigma = 1$ | 0.7500 | 1.0000 | 5.0000 | 32.0000 | $\infty$ |
| Laplace, scale $b = 1$ | 0.5129 | 0.6191 | 0.9287 | 0.9891 | 1 |
| Randomized response, which reports a bit truthfully with probability 0.75 | 0.7340 | 0.8473 | 1.0666 | 1.0940 | 1.0986 |

: RDP curves of the three basic mechanisms with sensitivity 1. {#tbl-rdp-curves}

The Laplace and randomized-response curves level off at their pure DP level. The Gaussian curve has no limit, so the Gaussian mechanism is not $\varepsilon$-DP for any $\varepsilon$. But an accountant converts only at finite orders, where the curve is finite. These curves are for steps on the whole dataset. The private form of stochastic gradient descent (SGD), called DP-SGD, runs each step on a random batch, and libraries compute its much smaller curves for you ([Amplification by subsampling](05-subsampling.md)).

## Composition adds curves {#sec-rdp-composition}

Training runs thousands of mechanisms in sequence, and each may depend on what the earlier ones released, as a gradient step depends on the current model. RDP composes these mechanisms by adding their curves, order by order. Write $\varepsilon_i$ for the curve of the $i$-th step. Then a whole run of $k$ steps has a curve $\varepsilon_M$ with
$$\varepsilon_M(\alpha) \le \sum_{i=1}^k \varepsilon_i(\alpha) \qquad\text{for every order } \alpha.$$
Different mechanisms mix freely, because you only add numbers at the same order.

Two conditions come with the sum. First, every curve must describe the same neighbouring relation, such as add/remove or replace-one, and the same unit of privacy, such as one record or one user. Second, the curve of each step must be fixed before the run. Steps may adapt to earlier outputs, but if the data choose the noise level or the number of steps, the curves depend on the data too. An early stop when a validation metric looks good is such a choice, and it needs [Adaptive composition](08-adaptive-composition.md) instead.

The sum is simplest for Gaussian steps, because their curve is the straight line $\alpha/(2\sigma^2)$. Adding $k$ such lines multiplies the slope by $k$, which gives the line of one step with noise multiplier $\sigma/\sqrt k$. So $k$ Gaussian steps cost as much privacy as one step with $\sqrt k$ times less noise, which is bad news for training. If 10,000 steps with noise multiplier 1 each used the whole dataset, they would cost as much as one step with noise multiplier $1/\sqrt{10{,}000} = 0.01$, whose $\varepsilon$ at $\delta = 10^{-5}$ is 5425.5, no guarantee at all. DP-SGD escapes this by running each step on a small random sample. This part's running example is such a run: 10,000 steps with noise multiplier 1, each on a random 1% of the data.

## Post-processing and group privacy {#sec-rdp-properties}

Post-processing is free in RDP, because a computation on the output that does not look at the data again cannot make the curve larger. So the release of the trained model or its predictions costs nothing beyond the training run.

Groups of records are harder, and the generic group rule of RDP is loose. So when one user gives $m$ records and the noise is Gaussian, use the sensitivity of the group instead, which is at most $m$ times that of one record. The curve becomes
$$\varepsilon_{\mathrm{group}}(\alpha) = \frac{m^2\alpha}{2\sigma^2},$$
which is $m^2$ times the curve of one record, so ten records per user cost a hundred times as much. The fix is to clip the whole contribution of each user to one clipping norm and to account per user ([User-level privacy](../part05-dp-ml/07-user-level.md)).

## Compose, then convert once {#sec-rdp-conversion}

After composition, you have a curve, but you report an $(\varepsilon, \delta)$ pair. Both libraries turn the curve into a pair with the **improved conversion** (Balle et al. 2020, and independently Canonne, Kamath and Steinke 2020). From the value $\varepsilon(\alpha)$ of the curve at one order and a target $\delta$, it gives the privacy level
$$\varepsilon' = \varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}.$$
Report the smallest $\varepsilon'$ over the grid of orders, or 0 if that is negative. Libraries call the order that gave it the best order.

Older papers and code use Mironov's original conversion, which gives 7.43 instead of 6.71 on the DP-SGD running example. The **moments accountant** of the original DP-SGD paper (Abadi et al. 2016) is the RDP curve with that conversion, so its $\varepsilon$ is loose but not wrong ([From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md)).

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
