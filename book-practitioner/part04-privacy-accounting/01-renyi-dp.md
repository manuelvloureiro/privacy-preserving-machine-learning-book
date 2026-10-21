# Rényi differential privacy

## Why this matters

Suppose you train a model with 1000 steps of full-batch noisy gradient descent. Each step adds Gaussian noise of standard deviation 20 to a quantity that one person's record can move by at most 1. You want the differential privacy (DP) guarantee of the whole run, a privacy level $\varepsilon$ at a small slack $\delta = 10^{-5}$. The tools of Part 3 turn each step into one such pair and then apply advanced composition ([Composition](../part03-dp-foundations/04-composition.md)). They give $\varepsilon = 68.01$, while the true answer is 7.51, so you would pay for a bound nine times too large in noise or in accuracy.

**Rényi differential privacy (RDP)** fixes this by keeping a whole curve for each step instead of one point. The curves of the steps simply add, with no $\delta$ spent along the way, and you convert the total into $(\varepsilon, \delta)$ once, at the end. For the same 1000 steps, this gives $\varepsilon = 8.84$, only 18% above the truth. It is what the RDP accountants of Opacus and Google's `dp_accounting` library track.

## The RDP curve {#sec-rdp-definition}

Call $P$ and $Q$ the output distributions of a mechanism on two neighbouring datasets, which differ in one person's record. The **Rényi divergence** of [Divergences](../part01-prerequisites/03-divergences.md) measures how far apart they are. It has a parameter called the **order**, written $\alpha$, which is any number above 1. Writing $p$ and $q$ for the densities of the two distributions, it is
$$D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\ln\int p^\alpha q^{1-\alpha}\,dy.$$
A larger order weighs more heavily the outputs that are much likelier under $P$ than under $Q$. So a small order looks at the typical case, and a large order at the worst case.

A mechanism is **$(\alpha, \varepsilon)$-RDP** if this divergence is at most $\varepsilon$ for every pair of neighbours. The smallest such $\varepsilon$, as a function of the order, is the mechanism's **RDP curve** $\varepsilon(\alpha)$, and a higher curve means less privacy. Each point of it bounds one exponential moment of the privacy loss, the logarithm of $p/q$, and [Numerical accounting](06-numerical-accounting.md) works with that loss directly.

The curve never decreases with the order. Its two ends tie it to familiar quantities. As the order falls towards 1, it tends to at least the Kullback–Leibler divergence between neighbours, the average privacy loss. At order infinity it is the largest privacy loss of any output, so an $\varepsilon$-DP mechanism is $(\alpha, \varepsilon)$-RDP at every order. Accountants store the whole curve on a grid of orders. This part uses a grid of 154 orders, the default list of Opacus with 128, 256 and 512 added.

## RDP curves of the basic mechanisms {#sec-rdp-curves}

To account for a mechanism you need its curve, and the three basic mechanisms have one in closed form (Mironov 2017). Each formula below assumes sensitivity 1, the most one record can move the answer. For the **Gaussian mechanism**, write $\sigma$ for the noise multiplier, the noise standard deviation in units of the $\ell_2$ sensitivity. Its curve is the straight line
$$\varepsilon(\alpha) = \frac{\alpha}{2\sigma^2}.$$
The **Laplace mechanism** adds noise whose typical size is its scale $b$, and its curve is
$$\varepsilon(\alpha) = \frac{1}{\alpha - 1}\ln\left(\frac{\alpha}{2\alpha - 1}e^{(\alpha - 1)/b} + \frac{\alpha - 1}{2\alpha - 1}e^{-\alpha/b}\right).$$
**Randomized response**, which reports a bit truthfully with probability $p$ and flips it otherwise, has a closed-form curve too, given in the full edition. @tbl-rdp-curves gives the three curves at a few orders.

| Mechanism | $\varepsilon(1.5)$ | $\varepsilon(2)$ | $\varepsilon(10)$ | $\varepsilon(64)$ | $\varepsilon(\infty)$ |
| --- | --- | --- | --- | --- | --- |
| Gaussian, $\sigma = 1$ | 0.7500 | 1.0000 | 5.0000 | 32.0000 | $\infty$ |
| Laplace, $b = 1$ | 0.5129 | 0.6191 | 0.9287 | 0.9891 | 1 |
| Randomized response, $p = 0.75$ | 0.7340 | 0.8473 | 1.0666 | 1.0940 | 1.0986 |

: RDP curves of the three basic mechanisms with sensitivity 1. {#tbl-rdp-curves}

The Laplace and randomized-response curves level off at their pure DP levels. The Gaussian curve grows without limit, so the Gaussian mechanism is not $\varepsilon$-DP for any $\varepsilon$, but it is RDP at every finite order, and that is all an accountant needs.

These curves are for steps that use the whole dataset. A private step of stochastic gradient descent (SGD), called DP-SGD, works on a random batch and has a much smaller curve, which [Amplification by subsampling](05-subsampling.md) computes. You rarely type these formulas in, because libraries hold them. The `rdp_bank` module of autodp, for example, has `RDP_gaussian`, `RDP_laplace` and `RDP_randresponse`, which return the curve at one order.

## Composition adds curves {#sec-rdp-composition}

Training runs thousands of mechanisms in sequence, and each may depend on what the earlier ones released, as a gradient step depends on the current model. RDP composes by adding curves order by order. Write $\varepsilon_i$ for the curve of step $i$, and $\varepsilon_M$ for the curve of a whole run of $k$ steps. Then
$$\varepsilon_M(\alpha) \le \sum_{i=1}^k \varepsilon_i(\alpha) \qquad\text{for every order } \alpha.$$
The sum is exact when the steps are independent. Different mechanisms mix freely, because you only add numbers at the same order.

A pipeline that mixes mechanisms, such as a Gaussian count, a Laplace histogram and a randomized-response survey, is accounted the same way. In `dp_accounting` you wrap the events in a `ComposedDpEvent`, wrap a repeated one in a `SelfComposedDpEvent`, and compose the result. All the curves you add must describe the same notion of neighbours and the same unit of privacy, though. In `dp_accounting` 0.6.0, for example, the RDP accountant refuses a `RandomizedResponseDpEvent` under its default add-or-remove `neighboring_relation`, which a `PoissonSampledDpEvent` needs ([Amplification by subsampling](05-subsampling.md) explains the relations).

The condition that matters is that each step's curve is fixed in advance. Steps may adapt to earlier outputs, but the noise level and the number of steps must not be chosen from the data during the run. If they are, for example by stopping when a validation metric looks good, you need [Adaptive composition](08-adaptive-composition.md).

For Gaussian steps the sum has a useful shortcut, because adding $k$ copies of the straight line gives the curve of a single Gaussian step with noise multiplier $\sigma/\sqrt k$. The shortcut also shows why DP-SGD needs subsampling. If this part's running example, 10,000 steps with noise multiplier 1, used the full dataset at every step, it would act like one step with noise multiplier 0.01. Its exact $\varepsilon$ would be 5425.5, which is no guarantee at all.

## Post-processing and group privacy {#sec-rdp-properties}

Post-processing is free in RDP. If you apply any randomized map $g$ to the output without looking at the data again, the curve of the result is no larger at any order:
$$\varepsilon_{g\circ M}(\alpha) \le \varepsilon_M(\alpha).$$
So releasing the trained model or its predictions costs nothing beyond the training run, as long as $g$ never reads the data.

Groups are harder. When one user contributes $m$ records, you need a guarantee for all of them at once. RDP has a generic group rule, Mironov's, under which each doubling of the group halves the order and triples $\varepsilon$. It is loose, and for the Gaussian mechanism it overstates the exact value 1.5 times for two records and 3.375 times for eight.

So for Gaussian noise, skip the generic rule and use the group's sensitivity. A group of $m$ records moves the answer by at most $m$ times as much as one record, so the curve becomes
$$\varepsilon_{\text{group}}(\alpha) = \frac{m^2\alpha}{2\sigma^2},$$
which is $m^2$ times the single-record curve. Better still, clip per user and account at the user level ([User-level privacy](../part05-dp-ml/07-user-level.md)).

## Convert once, at the end {#sec-rdp-conversion}

After composing, you hold a curve, but you report an $(\varepsilon, \delta)$ pair. A conversion tries every order of the grid and keeps the one that gives the smallest $\varepsilon$, which libraries report as the best order. [From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md) gives the formulas. The workflow is what matters here, because you compose the curves of all steps first and convert once, at the end.

The **moments accountant** of the original DP-SGD paper (Abadi et al. 2016) is this same workflow with different bookkeeping. It tracks $\lambda\,\varepsilon(\lambda + 1)$, the logarithm of the moment generating function of the privacy loss, and its conversion is Mironov's, the original RDP conversion. So read an older paper's moments-accountant $\varepsilon$ as an RDP number. Similarly, `compute_dp_sgd_privacy_statement` in TensorFlow Privacy 0.9.0 uses the RDP accountant of `dp_accounting` by default.

@tbl-rdp-compare shows what composing curves gains for runs of $k$ Gaussian steps with $\sigma = 20$. It compares advanced composition with RDP, converted by Mironov's conversion, and with the exact value from the Gaussian's privacy curve ([f-DP and Gaussian differential privacy](04-gaussian-dp.md)).

| $k$ | Advanced | RDP (best order) | Exact |
| --- | --- | --- | --- |
| 1 | 0.847 | 0.251 (128) | 0.160 |
| 10 | 3.270 | 0.771 (31) | 0.561 |
| 100 | 13.423 | 2.524 (10.6) | 1.993 |
| 1000 | 68.008 | 8.838 (4) | 7.511 |

: Total $\varepsilon$ at $\delta = 10^{-5}$ for $k$ Gaussian steps with $\sigma = 20$: advanced composition, RDP with Mironov's conversion (best order on the grid in brackets), and exact. {#tbl-rdp-compare}

Advanced composition is 5 to 9 times the exact value, because each step was collapsed to one point before composing, while RDP gets tighter as the run grows. So converting each step to an $(\varepsilon, \delta)$ pair first and composing those pairs throws the gain away.

The best order moves with the run, from 128 for one step to 4 for 1000 steps. So a grid must cover orders from just above 1 to a few hundred, as the default grid of either library does. A best order on the edge of the grid means that the true optimum lies outside it, and the reported $\varepsilon$ is too large.

## Accounting with a library {#sec-rdp-library}

In production, you describe each step to a library and let it compose and convert. Google's `dp_accounting` describes each mechanism as an event, so a DP-SGD step is a `GaussianDpEvent` inside a `PoissonSampledDpEvent`, and an `RdpAccountant` composes the events and converts the total. For example, these lines account for the running example in version 0.6.0.

```text
import dp_accounting

step = dp_accounting.PoissonSampledDpEvent(
    sampling_probability=0.01,
    event=dp_accounting.GaussianDpEvent(noise_multiplier=1.0),
)
accountant = dp_accounting.rdp.RdpAccountant()       # or RdpAccountant(orders=[...])
accountant.compose(step, count=10_000)
eps, order = accountant.get_epsilon_and_optimal_order(target_delta=1e-5)
```

Opacus works differently, because its accountant is attached to the optimizer. There, `PrivacyEngine(accountant="rdp")` selects the class `opacus.accountants.RDPAccountant`, which records each update through `step(noise_multiplier=..., sample_rate=...)`. Its method `get_privacy_spent(delta=...)` then returns $\varepsilon$ with the best order. Outside training, the functions `compute_rdp` and `get_privacy_spent` of `opacus.accountants.analysis.rdp` do the same in two calls. Ask for RDP explicitly when you want these numbers, since the default accountant of Opacus 1.6.0 is a numerical one.

Three checks catch most mistakes. First, if the best order the library returns sits at either end of the grid, pass a longer list of orders. Second, make sure the event matches the sampler your data loader runs, because a Poisson event for a shuffled fixed-size batch is a different mechanism ([Amplification by subsampling](05-subsampling.md)). Third, count each step once. In Opacus the optimizer already steps the accountant after every update, so a manual call to `step` on top of it charges each update twice.

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| $(\alpha, \varepsilon)$-RDP | $D_\alpha(M(D)\Vert M(D')) \le \varepsilon$ for all neighbours | A curve over orders; $\varepsilon(\infty)$ is pure DP |
| Gaussian, noise multiplier $\sigma$ | $\varepsilon(\alpha) = \alpha/(2\sigma^2)$ | Exact; $\sigma$ in units of the $\ell_2$ sensitivity |
| Composition | $\varepsilon_M(\alpha) \le \sum_i\varepsilon_i(\alpha)$ | Curves fixed in advance |
| $k$ full-batch Gaussian steps | One Gaussian with $\sigma/\sqrt k$ | Not for subsampled steps |
| Group of $m$, Gaussian | $m^2\alpha/(2\sigma^2)$ | The generic group rule is loose |

: The results of this chapter. {#tbl-rdp-cheat}

**Rules of thumb**

- Compose in RDP and convert once at the end. Never convert each step to $(\varepsilon_0, \delta_0)$ first.
- Keep a grid of orders from just above 1 to a few hundred, and check that the best order is not at its edge.
- Plan the noise level and the number of steps before training, or use an adaptive accountant.
- For a group of records under Gaussian noise, use the group's sensitivity, not the generic group rule.
- Add only curves computed under the same neighbouring relation and the same unit of privacy.

## Further reading

- Mironov, "Rényi Differential Privacy" (IEEE CSF 2017): the definition, composition, the curves of the basic mechanisms, group privacy and the conversion.
- Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016): DP-SGD and the moments accountant.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/01-renyi-dp.md).
