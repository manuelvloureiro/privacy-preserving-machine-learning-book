# Rényi differential privacy

## Why this matters

Training a model runs thousands of private steps, and you report one differential privacy (DP) guarantee for all of them. It is a privacy level $\varepsilon$ with a small additive slack $\delta$. The tools of Part 3 turn each step into one such pair and then add the pairs up with advanced composition ([Composition](../part03-dp-foundations/04-composition.md)). That route is far too pessimistic. As an illustration, take 1000 steps of full-batch noisy gradient descent. Each step adds Gaussian noise to a gradient, with a standard deviation 20 times the most one record can move it. At $\delta = 10^{-5}$, advanced composition gives this run $\varepsilon = 68.01$, while its exact value is 7.51 (@sec-rdp-composition shows how). So the bound is nine times too large, and a run calibrated to it adds more noise than it needs.

**Rényi differential privacy (RDP)** fixes this by keeping a whole curve of privacy costs for each step instead of one point. The curves of the steps simply add, with no $\delta$ spent along the way, and you convert the total into $(\varepsilon, \delta)$ once, at the end. Even with Mironov's original conversion (@sec-rdp-conversion), the same run then costs 8.84, only 18% above the exact value. RDP is what the RDP accountants of Opacus, for PyTorch, and of Google's `dp_accounting` track.

## The RDP curve {#sec-rdp-definition}

A DP guarantee compares the outputs of a mechanism on two neighbouring datasets, which differ in one person's record. RDP measures how far apart those two output distributions are with the **Rényi divergence** of [Divergences](../part01-prerequisites/03-divergences.md). That divergence has a dial, called the **order** and written $\alpha$, which is any number above 1. A small order looks at the typical output, and a large order weighs more heavily the rare outputs that are much likelier on one dataset than on the other.

A mechanism is **$(\alpha, \varepsilon)$-RDP** if, at order $\alpha$, the divergence is at most $\varepsilon$ for every pair of neighbours. The smallest such $\varepsilon$ at each order traces out the mechanism's **RDP curve** $\varepsilon(\alpha)$, and a higher curve means less privacy. The curve never decreases with the order. At order infinity it is the largest privacy loss of any output, which is what pure DP, the case $\delta = 0$, bounds. So an $\varepsilon$-DP mechanism is $(\alpha, \varepsilon)$-RDP at every order. You never compute the divergence yourself, because accountants store the curve's values on a grid of orders.

## RDP curves of the basic mechanisms {#sec-rdp-curves}

To account for a mechanism you need its curve, and the basic mechanisms have one in closed form (Mironov 2017). The one you meet most is the **Gaussian mechanism**. Write $\sigma$ for its noise multiplier, the noise standard deviation in units of the $\ell_2$ sensitivity, which is the most one record can move the answer. Its curve is the straight line
$$\varepsilon(\alpha) = \frac{\alpha}{2\sigma^2}.$$
@tbl-rdp-curves gives the closed-form curves of the Gaussian, Laplace and randomized-response mechanisms at a few orders, each at sensitivity 1.

| Mechanism | $\varepsilon(1.5)$ | $\varepsilon(2)$ | $\varepsilon(10)$ | $\varepsilon(64)$ | $\varepsilon(\infty)$ |
| --- | --- | --- | --- | --- | --- |
| Gaussian, $\sigma = 1$ | 0.7500 | 1.0000 | 5.0000 | 32.0000 | $\infty$ |
| Laplace, scale $b = 1$ | 0.5129 | 0.6191 | 0.9287 | 0.9891 | 1 |
| Randomized response, which reports a bit truthfully with probability 0.75 | 0.7340 | 0.8473 | 1.0666 | 1.0940 | 1.0986 |

: RDP curves of the three basic mechanisms with sensitivity 1. {#tbl-rdp-curves}

The Laplace and randomized-response curves level off at their pure DP levels. The Gaussian curve grows without limit, so the Gaussian mechanism is not $\varepsilon$-DP for any $\varepsilon$. But it is RDP at every finite order, and an accountant only ever converts at finite orders. These curves are for steps that use the whole dataset. The private form of stochastic gradient descent (SGD), called DP-SGD, uses a random batch at each step. Libraries compute the much smaller curves of its steps for you ([Amplification by subsampling](05-subsampling.md)).

## Composition adds curves {#sec-rdp-composition}

Training runs thousands of mechanisms in sequence, and each may depend on what the earlier ones released, as a gradient step depends on the current model. RDP composes by adding curves order by order. Write $\varepsilon_i$ for the curve of the $i$-th step. Then a whole run of $k$ steps has a curve $\varepsilon_M$ with
$$\varepsilon_M(\alpha) \le \sum_{i=1}^k \varepsilon_i(\alpha) \qquad\text{for every order } \alpha.$$
Different mechanisms mix freely, because you only add numbers at the same order.

Two conditions come with the sum. First, every curve must describe the same neighbouring relation, such as add/remove or replace-one, and the same unit of privacy, such as one record or one user. For example, the RDP accountant of `dp_accounting` 0.6.0 composes a `PoissonSampledDpEvent` under its default add/remove relation. But it refuses a `RandomizedResponseDpEvent` there, since it computes that curve only for replaced records. Second, each step's curve must be fixed in advance. Steps may adapt to earlier outputs, but the noise level and the number of steps must not be chosen from the data during the run. Otherwise the curves themselves would depend on the data. If you stop when a validation metric looks good, you need [Adaptive composition](08-adaptive-composition.md) instead.

For Gaussian steps the sum has a useful shortcut. Adding $k$ copies of the straight line gives the curve of a single Gaussian step with noise multiplier $\sigma/\sqrt k$, whose exact $\varepsilon$ you can compute. The shortcut also shows why DP-SGD needs subsampling. This part's running example is a DP-SGD run of 10,000 steps with noise multiplier 1, each on a random 1% of the data. If every step used the full dataset, the run would act like one step with noise multiplier 0.01. Its exact $\varepsilon$ at $\delta = 10^{-5}$ would be 5425.5, which is no guarantee at all.

## Post-processing and group privacy {#sec-rdp-properties}

Post-processing is free in RDP. Any computation on the output that does not look at the data again leaves the curve no larger at any order. So releasing the trained model or its predictions costs nothing beyond the training run.

Groups are harder, because when one user contributes several records, you need a guarantee for all of them at once. RDP has a generic group rule, but it is loose, so for Gaussian noise use the group's sensitivity instead. A group of $m$ records moves the answer by at most $m$ times as much as one record, so the curve becomes
$$\varepsilon_{\mathrm{group}}(\alpha) = \frac{m^2\alpha}{2\sigma^2},$$
which is $m^2$ times the single-record curve. That grows fast. So when users have many records, clip each user's whole contribution and account per user ([User-level privacy](../part05-dp-ml/07-user-level.md)), which keeps the sensitivity at one clipping norm.

## Compose, then convert once {#sec-rdp-conversion}

After composing, you hold a curve, but you report an $(\varepsilon, \delta)$ pair. A conversion turns each point of the curve into a pair, and you keep the order that gives the smallest $\varepsilon$, which libraries report as the best order. Use the **improved conversion**, which both widely used libraries apply (Balle et al. 2020, and independently Canonne, Kamath and Steinke 2020). From the curve's value $\varepsilon(\alpha)$ at one order and a target $\delta$, it gives the privacy level
$$\varepsilon' = \varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}.$$
You report the smallest $\varepsilon'$ over the grid, or 0 if that comes out negative.

Older papers and code use Mironov's original conversion, which gives larger numbers from the same curve, such as the 8.84 above. On the DP-SGD running example, Mironov's conversion gives 7.43 and the improved one 6.71. The exact value is lower still, as [From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md) shows. The **moments accountant** of the original DP-SGD paper (Abadi et al. 2016) is the same RDP curve in a rescaled form, converted with Mironov's formula. So its $\varepsilon$ is loose but not wrong.

## Check the order grid {#sec-rdp-orders}

The best order moves with the run. Long runs with little noise want orders close to 1, while short runs with much noise want large orders. In the illustration above, it falls from 128 for one step to 4 for 1000 steps. A missing order can only raise $\varepsilon$, so the grid must cover the order the run needs.

The Opacus default list `RDPAccountant.DEFAULT_ALPHAS` stops at order 63, while the `DEFAULT_RDP_ORDERS` of `dp_accounting` go on to 1024. A best order at either end of the grid means that the true optimum lies outside it and the reported $\varepsilon$ is too large. Then pass a longer list and convert again, with `alphas=[...]` in Opacus or `orders=[...]` in `dp_accounting`. Opacus warns when the best order is at an end of its list. `dp_accounting` does not, so there you compare the returned order with the ends of the list yourself.

## Accounting with a library {#sec-rdp-library}

In practice you describe each step to a library and let it compose and convert. The `dp_accounting` library describes each mechanism as an event. A DP-SGD step is then a `GaussianDpEvent` inside a `PoissonSampledDpEvent`, which samples each record independently, with probability 0.01 in the running example. This snippet composes the 10,000 steps and prints $\varepsilon$ with the best order.

```python
import dp_accounting

step = dp_accounting.PoissonSampledDpEvent(0.01, dp_accounting.GaussianDpEvent(1.0))
accountant = dp_accounting.rdp.RdpAccountant()
accountant.compose(step, count=10_000)
eps, order = accountant.get_epsilon_and_optimal_order(1e-5)
print(f"{eps:.4f} {order:g}")  # 6.7128 4.1
```

Opacus works differently, because its accountant is attached to the optimizer, which calls `step` after every noisy update. The next snippet does the same by hand, and it gives the same $\varepsilon$ up to the fourth decimal at the same order, 4.1. Both grids step by tenths below order 11, so that order is well inside them.

```python
from opacus.accountants import RDPAccountant

accountant = RDPAccountant()
for _ in range(10_000):  # what the optimizer does at each step of training
    accountant.step(noise_multiplier=1.0, sample_rate=0.01)
eps, order = accountant.get_privacy_spent(delta=1e-5)
print(f"{eps:.4f} {order:g}")  # 6.7127 4.1
```

In training, `PrivacyEngine(accountant="rdp")` selects this class. Ask for it explicitly when you want RDP numbers, since the default accountant of Opacus 1.6.0 is `"prv"`, a [numerical accountant](06-numerical-accounting.md) of the privacy loss distribution. The function `get_noise_multiplier`, which solves for $\sigma$, defaults to RDP instead, so name the accountant there too and use the same one you report with. An RDP number is a valid upper bound, but it sits above the numerical accountant's value for the same run, which is 6.19 for the running example.

Two more checks catch most mistakes. Make sure the event matches the sampler your data loader runs, because shuffled fixed-size batches get none of a Poisson event's amplification ([Amplification by subsampling](05-subsampling.md#sec-sub-schemes)). And count each step once, since a manual call to `step` on top of the optimizer's own charges each update twice.

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| RDP curve | $\varepsilon(\alpha)$ over orders $\alpha > 1$ | Never decreases; $\varepsilon(\infty)$ is pure DP |
| Gaussian, noise multiplier $\sigma$ | $\varepsilon(\alpha) = \alpha/(2\sigma^2)$ | $\sigma$ in units of the $\ell_2$ sensitivity |
| Composition | $\varepsilon_M(\alpha) \le \sum_i\varepsilon_i(\alpha)$ | Same relation and unit; curves fixed in advance |
| Group of $m$, Gaussian | $m^2\alpha/(2\sigma^2)$ | The generic group rule is loose |
| Improved conversion | $\varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}$, best $\alpha$ | What Opacus and `dp_accounting` use |

: The results of this chapter. {#tbl-rdp-cheat}

**Rules of thumb**

- Compose in RDP and convert once at the end. Never convert each step to a pair first.
- Check that the best order is not at either end of the grid, and extend the grid if it is.
- In Opacus, name the accountant in both `PrivacyEngine` and `get_noise_multiplier`.

## References

- Mironov, "Rényi Differential Privacy" (IEEE CSF 2017): the definition, composition, the curves of the basic mechanisms and group privacy.
- Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016): DP-SGD and the moments accountant.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/01-renyi-dp.md).
