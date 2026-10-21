# Numerical accounting with privacy loss distributions

## Why this matters

You trained a model with differential privacy (DP), using the private form of stochastic gradient descent (SGD) called DP-SGD. The run took 10,000 steps, with sampling rate 0.01 and noise multiplier 1, and now you need the smallest $\varepsilon$ for which the whole run is $(\varepsilon, 10^{-5})$-DP. Rényi differential privacy (RDP) gives an upper bound that may be loose by an amount you cannot see ([Rényi DP](01-renyi-dp.md)). The central limit theorem (CLT) of [f-DP and Gaussian DP](04-gaussian-dp.md) gives an approximation that may land on either side of the truth.

A privacy loss distribution (PLD) accountant computes the answer itself. It returns an upper bound and a lower bound, and you know in advance how far apart they can be. On this run it brackets the true $\varepsilon$ between 6.1377 and 6.2377, while RDP says 6.7127.

## The privacy loss distribution {#sec-num-pld}

Call your dataset $D$ and a neighbouring dataset $D'$. The mechanism's output then has a distribution $P$ on the first and $Q$ on the second. The **privacy loss** of an output $y$ is the logarithm of how much likelier it is on $D$,
$$L(y) = \ln\frac{P(y)}{Q(y)}.$$
The loss is positive when the output favours $D$ and negative when it favours $D'$. The output is random, so its loss is too, and the distribution of the loss of an output drawn from $P$ is the **PLD**.

The PLD is all you need to compute the slack $\delta$. The definition of DP lets each output be up to $e^\varepsilon$ times likelier under $D$ for free, and $\delta$ pays for the excess ([The definition of differential privacy](../part03-dp-foundations/01-definition.md)). Write $(x)_+$ for $x$ when it is positive and 0 otherwise. Then for every $\varepsilon$,
$$\delta(\varepsilon) = \mathbb E\big[(1 - e^{\varepsilon - L})_+\big] = P[L > \varepsilon] - e^{\varepsilon}\,Q[L > \varepsilon].$$ {#eq-num-delta}
In words, $\delta$ averages a weight over the losses. The weight is zero for losses up to $\varepsilon$ and rises towards 1 above it, so $\delta$ reads only the upper tail of the PLD.

For the Gaussian mechanism the PLD has a closed form, which makes it the check for any accountant. Write $\mu$ for the query's sensitivity divided by the noise standard deviation, which is the shift in units of the noise. The PLD is then the normal distribution with mean $\mu^2/2$ and variance $\mu^2$. Write $\Phi$ for the standard normal distribution function. Putting the two normal tails into @eq-num-delta gives
$$\delta(\varepsilon) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^{\varepsilon}\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big).$$ {#eq-num-gauss}
This is the exact $\delta$ of [the Gaussian mechanism](../part03-dp-foundations/03-laplace-gaussian.md), and the privacy curve of $\mu$-Gaussian differential privacy (GDP). Since $\delta$ falls as $\varepsilon$ grows, you solve it for $\varepsilon$ by bisection, as the snippet in the Code section does.

## Composition is convolution {#sec-num-convolution}

When steps run independently, the privacy loss of the whole run is the sum of the per-step losses, as [Composition](../part03-dp-foundations/04-composition.md) showed. The distribution of a sum of independent variables is the **convolution** of their distributions. So the PLD of $k$ steps is the $k$-fold convolution of the per-step PLDs, and this step loses nothing. So all the error of a PLD accountant comes from computing the convolution on a machine.

Gaussian steps give reference values that every accountant must reproduce. The means and the variances of normal losses add, so $k$ Gaussian steps with shift $\mu$ act like one step with shift $\sqrt k\,\mu$. Ten steps with $\mu = 1$ are therefore $(17.8566, 10^{-5})$-DP, and an accountant that misses this value should not be trusted on DP-SGD.

DP-SGD steps are not independent, because each gradient depends on the model that earlier outputs produced. The fix is a **dominating pair**, a fixed pair of distributions that is at least as easy to tell apart as the step, whatever happened before. Zhu, Dong and Wang (2022) prove that convolving the PLDs of dominating pairs gives a valid $\delta(\varepsilon)$ for the adaptive run. For the Gaussian and the Poisson-subsampled Gaussian, the pairs that libraries use lose nothing. You need a filter instead if the noise level or the number of steps is chosen from the outputs ([Adaptive composition](08-adaptive-composition.md)).

## Pessimistic and optimistic discretization {#sec-num-discretize}

A computer puts the losses on a grid, the multiples of a spacing $h$. Rounding changes $\delta$, but the weight in @eq-num-delta never decreases as the loss grows, so the direction of the rounding fixes the direction of the error. The **pessimistic** discretization rounds every loss up, and the **optimistic** one rounds every loss down. The order survives any number of convolutions, so for $k$ steps and every $\varepsilon$,
$$\delta_{\text{optimistic}}(\varepsilon) \;\le\; \delta(\varepsilon) \;\le\; \delta_{\text{pessimistic}}(\varepsilon).$$ {#eq-num-sandwich}

The width of this sandwich is known before you run anything. Each step's two roundings are at most one grid step apart, so after $k$ steps the two values of $\varepsilon$ at your target $\delta$ are at most $kh$ apart. For a smooth PLD the width is almost exactly $kh$, so the error grows linearly in the number of steps.

The two sides are not equally strong. The rounded-up PLD still belongs to a real pair of distributions, which dominates the original, so the pessimistic value is a guarantee even for adaptive runs. The optimistic value is only a lower bound on what the mechanism costs, so it is an error bar and a referee, never a number to report. The midpoint of the two sides, or rounding each loss to the nearest grid point, is a good estimate but is not a bound in either direction.

## Composition by FFT {#sec-num-fft}

Production accountants convolve with the fast Fourier transform (FFT), padding each convolution so that no mass wraps around the window. If you write your own, pad too, because a wrapped result can err in either direction and is then no bound at all.

Two refinements of the grid are in use. The privacy loss random variable (PRV) accountant of Gopi, Lee and Wutschitz (2021) rounds each loss to the nearest grid point, after shifting the grid to keep the true mean. Its rounding errors then partly cancel, growing like $\sqrt k\,h$ instead of $kh$, and it still proves error bounds on $\varepsilon$. **Connect the dots** (Doroshenko et al. 2022), which Google's `dp_accounting` uses by default, builds the best pessimistic PLD on a given grid, so it allows coarser grids.

## The subsampled Gaussian and DP-SGD {#sec-num-dpsgd}

A DP-SGD step with Poisson sampling reduces to a one-dimensional pair ([Amplification by subsampling](05-subsampling.md)). In units of the clipping norm, the output without the record is $\mathcal N(0, \sigma^2)$, normal noise centred at 0 whose standard deviation is the noise multiplier $\sigma$. When the dataset holds the record, it is drawn with the sampling rate $q$, and when drawn it shifts the centre by 1. So with the record the output follows the mixture $(1 - q)\,\mathcal N(0, \sigma^2) + q\,\mathcal N(1, \sigma^2)$. A guarantee for adding or removing one record must cover both orders of this pair. The **remove** direction puts the mixture in the role of $P$, and the **add** direction swaps the two.

The two directions behave very differently. A rare large output betrays that the record was used, so the remove direction's loss has a long right tail, while the add direction's loss stays below $-\ln(1 - q)$. So one step at $\delta = 10^{-5}$ costs 0.1995 in the remove direction and only 0.0091 in the add direction. Neither direction is the larger in general, so compose both and report the larger. This pair also assumes Poisson sampling, and a loader that forms batches another way needs a different accountant ([Amplification by subsampling](05-subsampling.md#sec-sub-schemes)).

| $k$ | PLD upper | PLD lower | RDP | GDP CLT |
| --- | --- | --- | --- | --- |
| 1 | 0.1995 | 0.1994 | 0.9555 | 0.0368 |
| 1000 | 1.8332 | 1.8232 | 2.1014 | 1.6177 |
| 10000 | 6.2377 | 6.1377 | 6.7127 | 6.0071 |

: DP-SGD with $q = 0.01$ and $\sigma = 1$: $\varepsilon$ at $\delta = 10^{-5}$ after $k$ steps. The PLD columns use the grid $h = 10^{-5}$, each the larger of the two directions. RDP uses the improved conversion. {#tbl-num-dpsgd}

@tbl-num-dpsgd shows what each accountant costs. RDP is a valid bound but a loose one, 6.7127 against the PLD's 6.2377 after the full run. The CLT is worse, because it falls below the PLD lower bound at every $k$. So quoting the CLT's 1.6177 after 1000 steps would claim a guarantee that the run does not have. The reason is that the subsampled PLD has a heavier right tail than a Gaussian, which is exactly the part that $\delta$ reads.

A PLD accountant is only as good as its grid. On the coarse grid $h = 10^{-3}$, the 1000-step bracket is a whole unit wide, and its upper end of 2.3279 is worse than RDP. A tenfold finer grid narrows the bracket tenfold but costs about ten times the memory.

## Running it with a library {#sec-num-libraries}

A maintained library should land inside @tbl-num-dpsgd's bracket. In Google's `dp_accounting`, the function `from_gaussian_mechanism` of the module `dp_accounting.pld.privacy_loss_distribution` builds the subsampled Gaussian. Its grid spacing is `value_discretization_interval`, whose default is $10^{-4}$. Its argument `pessimistic_estimate` chooses the side of the rounding. Its default neighbouring relation is adding or removing one record, so it covers both directions and reports the larger. Then `self_compose` composes the steps and `get_epsilon_for_delta` reads off $\varepsilon$:

```text
from dp_accounting.pld import privacy_loss_distribution as pld_lib

step = pld_lib.from_gaussian_mechanism(standard_deviation=1.0, sampling_prob=0.01,
    value_discretization_interval=1e-5, pessimistic_estimate=True)
upper = step.self_compose(10_000).get_epsilon_for_delta(1e-5)
```

Run it again with `pessimistic_estimate=False` for the lower bound.

Its event interface, `dp_accounting.pld.PLDAccountant`, composes a `PoissonSampledDpEvent` that wraps a `GaussianDpEvent`, and `get_epsilon(target_delta)` returns the pessimistic value. Microsoft's `prv_accountant` package takes the error you accept instead of a grid spacing. Its `PRVAccountant` takes a list of mechanisms, such as `PoissonSubsampledGaussianMechanism`, with `max_self_compositions`, `eps_error` and `delta_error`. Then `compute_epsilon(delta, num_self_compositions)` returns a lower bound, an estimate and an upper bound, and you report the upper one. The accountant is set up for at most `max_self_compositions` steps of each mechanism, so make that at least the number of steps you will run. Opacus uses the same accountant by default, through `PrivacyEngine(accountant="prv")`. If two libraries' brackets fail to overlap, one of them has a bug ([Choosing an accountant](09-choosing-an-accountant.md#sec-choose-check)).

A numerical accountant is slower than RDP, but one run of it is cheap next to training. The cost adds up when you calibrate the noise, because a search evaluates the accountant at many noise levels. So search with RDP, then confirm the final noise level with the PLD.

## Code

The snippet solves @eq-num-gauss for $\varepsilon$, where $k$ Gaussian steps with noise multiplier $\sigma$ and sensitivity 1 have $\mu = \sqrt k/\sigma$.

```python
from math import erfc, exp, sqrt


def normal_cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def gaussian_delta(eps: float, mu: float) -> float:
    """Exact delta(eps) of the Gaussian PLD N(mu^2/2, mu^2)."""
    return normal_cdf(mu / 2 - eps / mu) - exp(eps) * normal_cdf(-mu / 2 - eps / mu)


def gaussian_epsilon(delta: float, mu: float) -> float:
    lo, hi = 0.0, 1000.0  # delta(eps) falls as eps grows: bisect for delta(eps) = delta
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gaussian_delta(mid, mu) > delta else (lo, mid)
    return hi


print(f"{gaussian_epsilon(1e-5, 1.0):.4f}")  # 4.3772
print(f"{gaussian_epsilon(1e-5, sqrt(10)):.4f}")  # 17.8566
print(f"{gaussian_epsilon(1e-5, 10.0):.4f}")  # 91.8173
```

## Cheat sheet

| Question | Answer | Notes |
| --- | --- | --- |
| Gaussian reference | PLD $\mathcal N(\mu^2/2, \mu^2)$; $k$ steps: $\mu \to \sqrt k\,\mu$ | 4.3772, 17.8566, 91.8173 at $\mu = 1, \sqrt{10}, 10$ |
| Which side to report | Pessimistic (rounded up) | Optimistic is the error bar; the midpoint is no bound |
| Grid spacing $h$ | Width in $\varepsilon$ at most $kh$ | Choose $h$ for your largest $k$, then check the width |
| Subsampled Gaussian | Compose add and remove, report the larger | Needs Poisson sampling |

: Numerical accounting at a glance. {#tbl-num-cheat}

**Rules of thumb**

- Report the pessimistic $\varepsilon$, or a PRV accountant's upper bound, never the estimate or the optimistic value. Give the optimistic value as the error bar.
- Check the width of the bracket rather than trusting the default grid, because the width grows with the number of steps.
- Use RDP for a quick bound and never quote the CLT as a guarantee, because it can fall below the PLD lower bound.

## Further reading

- Gopi, Lee & Wutschitz, "Numerical Composition of Differential Privacy" (NeurIPS 2021): the PRV accountant, with explicit error bounds.
- Zhu, Dong & Wang, "Optimal Accounting of Differential Privacy via Characteristic Function" (AISTATS 2022): accounting with the characteristic function of the PLD, dominating pairs and the two directions of Poisson subsampling.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/06-numerical-accounting.md).
