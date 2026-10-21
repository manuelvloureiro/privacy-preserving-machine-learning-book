# Amplification by subsampling

## Why this matters

Most private models are trained with stochastic gradient descent (SGD) made to satisfy differential privacy (DP), a method called DP-SGD. Each step clips every example's gradient, sums the clipped gradients and adds Gaussian noise, whose standard deviation is the clipping norm times the **noise multiplier** $\sigma$. Suppose every record paid the full cost of a Gaussian mechanism at every step. With $\sigma = 1$, ten thousand steps would then cost a privacy level of $\varepsilon = 5425.5$ ([Rényi DP](01-renyi-dp.md)).

DP-SGD escapes this cost because each step looks only at a small random batch. Write $q$ for the **sampling rate**, the probability that a given record lands in the batch. An attacker must then first guess whether the record was used at all, and that uncertainty is worth a lot. This effect is called **amplification by subsampling**. It depends on exactly how the batch was drawn, so the accountant, the code that adds up a run's privacy cost, must assume the sampler the training code actually runs. Getting that wrong is the largest single error in DP-SGD accounting.

## One step under Poisson sampling {#sec-sub-eps-delta}

The simplest sampler is **Poisson sampling**, in which each record enters the batch by its own coin flip, with probability $q$, independently of every other record. Pair it with add/remove neighbours, datasets that differ by one record being present or absent. Balle, Barthe and Gaboardi (2018) give the tight result. If a step is $(\varepsilon, \delta)$-DP on its own, where $\delta$ is the slack, then the same step run on a Poisson sample is $(\varepsilon', \delta')$-DP with
$$\varepsilon' = \ln\big(1 + q(e^\varepsilon - 1)\big), \qquad \delta' = q\delta.$$ {#eq-sub-amplify}

The formula behaves very differently for small and large $\varepsilon$. For small $\varepsilon$ the amplified cost is about $q\varepsilon$, so a 1% sample makes a step about a hundred times cheaper. With that sample, for example, a 1-DP step costs 0.0170. But for large $\varepsilon$ the amplified cost is about
$$\varepsilon' \approx \varepsilon - \ln(1/q),$$
so sampling only subtracts a fixed amount. With the same sample, a 10-DP step still costs 5.3993. The switch happens where $\varepsilon$ is near $\ln(1/q)$.

The result has two conditions. Each record's coin must be independent of the others, so that the record is drawn with probability exactly $q$ whatever else is in the batch. And the batch must stay secret, because an attacker who sees the list of records drawn knows whether the target was used. The rule applies to any $(\varepsilon, \delta)$-DP step, not only to the Gaussian. So a pure DP or randomized-response release run on a secret Poisson sample is amplified too. For example, 1-DP randomized response on a 10% sample costs 0.1586. Per-step pairs compose badly over thousands of steps, though (@sec-sub-dpsgd).

## Match the sampler to the accountant {#sec-sub-schemes}

Most training code does not draw Poisson batches. Some draws a fresh random batch of a fixed size, written $m$, from the $n$ records at each step. Much more code shuffles the data once per epoch and cuts the list into consecutive batches, which is called **shuffle and partition**. Each sampler pairs naturally with a neighbouring relation, and @tbl-sub-matching gives what to account for under each pairing.

| Sampling | Neighbours | What to account for |
| --- | --- | --- |
| Poisson, rate $q$ | add/remove | A Poisson accountant with $q$ and $\sigma$ |
| Fixed size $m$ of $n$ | add/remove | A Poisson accountant with rate $m/n$ and noise multiplier $\sigma/2$ (Lebeda et al. 2024) |
| Fixed size $m$ of $n$ | replace-one | Per step, @eq-sub-amplify with $q = m/n$ and $\sigma/2$; no Poisson shortcut |
| Poisson, rate $q$ | replace-one | About twice the add/remove cost for pure DP at small $\varepsilon$ |
| Shuffle and partition | zero-out | No amplification formula; the Poisson number can be far too small |

: What to account for under each combination of sampling scheme and neighbouring relation. {#tbl-sub-matching}

The factor of two in the noise comes from the gradient sum. Adding or removing a record moves the clipped sum by at most the clipping norm. But a record that enters a fixed-size batch pushes another one out, so the sum can move by twice as much. Twice the shift with the same noise is the same as the original shift with half the noise multiplier, and that is not a small change. This part's running example takes ten thousand steps with $q = 0.01$ and $\sigma = 1$. At $\delta = 10^{-5}$, the Poisson accountant gives it $\varepsilon = 6.71$. With fixed-size batches under add/remove it must be run at $\sigma = 0.5$, and it then gives 47.42, about seven times more.

Shuffle and partition is worse, because the source of amplification is missing. Within an epoch the record is in exactly one batch and is never absent, so the attacker only has to find which batch held it. Chua et al. (2024) build a dataset where the target's batch stands out, under zero-out neighbours, which set the target's gradient to 0. A test on it proves that one shuffled epoch of the running example has $\varepsilon$ of at least 4.06, while a Poisson accountant reports at most 1.21. So a Poisson number reported for shuffled training can be provably false.

So read the data loader before you trust a reported $\varepsilon$. A PyTorch `DataLoader` with `shuffle=True` and a `batch_size` is shuffle and partition, and a `tf.data` pipeline that shuffles and then batches is not Poisson sampling either. Opacus replaces the loader with a Poisson one by default: `PrivacyEngine.make_private` takes `poisson_sampling=True` unless you turn it off, and `opacus.data_loader.DPDataLoader.from_data_loader` converts a loader by hand. A Poisson loader gives batches of varying size, which is a quick check that it is in use. When the largest batches do not fit in memory, Opacus's `BatchMemoryManager` splits them into physical batches without changing the accounting ([Private stochastic gradient descent](../part05-dp-ml/02-dp-sgd.md)). If you cannot use Poisson sampling, report under the matching row of @tbl-sub-matching, or say plainly that $\varepsilon$ assumes a sampler you did not run.

## The accountant for DP-SGD {#sec-sub-dpsgd}

Accountants for DP-SGD work in Rényi differential privacy (RDP), which describes a mechanism by a curve $\varepsilon(\alpha)$ over orders $\alpha > 1$ ([Rényi DP](01-renyi-dp.md)). Curves add over steps, even adaptive ones. So a run of $k$ steps has the curve $k\,\varepsilon(\alpha)$, which is [converted](02-rdp-conversions.md) to $(\varepsilon, \delta)$ once.

Mironov, Talwar and Zhang (2019) show that one Poisson-subsampled Gaussian step has, at integer orders, the curve
$$\varepsilon(\alpha) = \frac{\ln A_\alpha}{\alpha - 1}, \qquad A_\alpha = \sum_{j=0}^\alpha\binom\alpha j(1 - q)^{\alpha - j}q^j\exp\Big(\frac{j^2 - j}{2\sigma^2}\Big).$$ {#eq-sub-binomial}
Here the index $j$ counts the imagined copies of the run that used the record, and accountants compute fractional orders by numerical integration. On the running example the curve is tiny at low orders. At order 2, for example, it is $1.7181 \times 10^{-4}$, against 1 without sampling. @tbl-sub-dpsgd shows what the accountant gives over a run, against the per-step route that amplifies each step with @eq-sub-amplify and then composes the pairs.

| Steps $k$ | RDP, Mironov's conversion | RDP, improved conversion | Per step, basic composition | Per step, advanced composition |
| --- | --- | --- | --- | --- |
| 1,000 | 2.54 | 2.10 | 839.39 | 1258.64 |
| 10,000 | 7.43 | 6.71 | 11296.93 | 24379.55 |

: The $\varepsilon$ at $\delta = 10^{-5}$ of DP-SGD with $q = 0.01$, $\sigma = 1$ under four accounting routes. {#tbl-sub-dpsgd}

The per-step route fails because each step keeps a tiny share of $\delta$, which forces a large per-step $\varepsilon$ that sampling barely reduces. So never compose per-step pairs for DP-SGD. The best order also falls as the run grows, to about 4 at ten thousand steps. So make sure the accountant's grid includes small and fractional orders, because a missing order can only raise $\varepsilon$.

In practice you use a library. Opacus's `RDPAccountant` and Google's `dp_accounting.rdp.RdpAccountant` both compute @eq-sub-binomial, and in `dp_accounting` you describe a step as `PoissonSampledDpEvent(sampling_probability, GaussianDpEvent(noise_multiplier))`. For fixed-size batches it has a `SampledWithoutReplacementDpEvent` for the replace-one relation, while Microsoft's `prv_accountant` describes the same step as a `PoissonSubsampledGaussianMechanism`. The value 6.71 is still an upper bound, and [numerical accounting](06-numerical-accounting.md) computes the tight value.

## Small q and the design of a run {#sec-sub-small-q}

For small $q$ the curve has a simple leading term, which is what makes the trade-offs between batch size, noise and epochs visible:
$$\varepsilon(\alpha) \approx \frac{\alpha q^2\big(e^{1/\sigma^2} - 1\big)}{2} \approx \frac{\alpha q^2}{2\sigma^2}\ \text{for large } \sigma.$$ {#eq-sub-small-q}
The per-step cost is quadratic in $q$, while a run of $E$ epochs takes $E/q$ steps. So the total RDP is about
$$\frac{E\,q\,\alpha}{2\sigma^2}.$$
At a fixed number of epochs, smaller batches are therefore cheaper in privacy.

That gives the recipe for planning a run. Fix the batch size $B$ and the number of epochs $E$ that your accuracy needs. Then the sampling rate is $q = B/n$ and the number of steps is $k = En/B$, where $n$ is the size of the dataset you actually train on, after any filtering. A wrong $n$ gives a wrong $q$, and so a wrong $\varepsilon$. Let the accountant solve for the noise multiplier that meets the target. Opacus does this with `get_noise_multiplier(target_epsilon=..., target_delta=..., sample_rate=..., epochs=...)`, or in one call with `make_private_with_epsilon`. Then check the result with a second accountant ([Choosing an accountant](09-choosing-an-accountant.md)).

Use @eq-sub-small-q for planning only, never inside an accountant, because at high orders it is far too small. The closed-form bound of the original **moments accountant** (Abadi et al. 2016) has the same shape and is outdated.

## Other mechanisms {#sec-sub-generic}

For RDP, unlike for @eq-sub-amplify, no rule returns every mechanism's subsampled curve exactly. General bounds for other mechanisms exist (Zhu & Wang 2019), but they are looser, so use the exact sum for the Gaussian.

## Code {#sec-sub-code}

The accountant below handles Poisson sampling and add/remove neighbours. It computes @eq-sub-binomial in log space at integer orders only, so its value for the running example is a little above the 6.71 of fractional orders.

```python
from math import exp, lgamma, log


def subsampled_gaussian_rdp(alpha: int, q: float, sigma: float) -> float:
    """RDP at integer order alpha of one Poisson-subsampled Gaussian step (sensitivity 1)."""
    logs = [
        lgamma(alpha + 1) - lgamma(j + 1) - lgamma(alpha - j + 1)
        + (alpha - j) * log(1 - q) + j * log(q) + (j * j - j) / (2 * sigma**2)
        for j in range(alpha + 1)
    ]
    top = max(logs)
    log_a = top + log(sum(exp(t - top) for t in logs))  # log-sum-exp avoids overflow
    return log_a / (alpha - 1)


def dpsgd_eps(q: float, sigma: float, steps: int, delta: float) -> tuple[float, int]:
    """Compose over the steps, convert with the improved conversion at the best integer order."""
    best = (float("inf"), 0)
    for alpha in range(2, 64):
        rdp = steps * subsampled_gaussian_rdp(alpha, q, sigma)
        eps = rdp + log((alpha - 1) / alpha) - (log(delta) + log(alpha)) / (alpha - 1)
        best = min(best, (eps, alpha))
    return best


print(f"{subsampled_gaussian_rdp(2, 0.01, 1.0):.4e}")  # 1.7181e-04
print("%.3f %d" % dpsgd_eps(0.01, 1.0, 10_000, 1e-5))  # 6.719 4
```

## Cheat sheet

| Situation | What to use | Notes |
| --- | --- | --- |
| One step, Poisson, add/remove | $\varepsilon' = \ln(1 + q(e^\varepsilon - 1))$, $\delta' = q\delta$ | About $q\varepsilon$ for small $\varepsilon$, $\varepsilon - \ln(1/q)$ for large |
| DP-SGD, Poisson | RDP or numerical accountant with $q$, $\sigma$ | 6.71 for the running example |
| DP-SGD, fixed-size batches, add/remove | Poisson accountant with $m/n$ and $\sigma/2$ | About seven times the Poisson $\varepsilon$ here |
| DP-SGD, shuffle and partition | No valid Poisson number | Switch to Poisson sampling |

: Which accounting to use for each sampler, and how to plan a run. {#tbl-sub-cheat}

**Rules of thumb**

- Account for the sampler you run, and state the neighbouring relation next to $\varepsilon$.
- The sampler can move $\varepsilon$ by a factor of seven, while the choice of conversion moves it by about 10%.
- Never compose per-step $(\varepsilon, \delta)$ pairs for DP-SGD. Compose RDP curves or privacy loss distributions, and convert once.
- Keep the small-$q$ formula for planning, never for the reported number.
- Compute $q$ from the dataset you actually train on, after filtering.
- A Poisson add/remove number does not hold for a replace-one claim.

## Further reading

- Balle, Barthe & Gaboardi, "Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences" (NeurIPS 2018): amplification for each sampler and neighbouring relation.
- Lebeda, Regehr, Kamath & Steinke, "Avoiding Pitfalls for Privacy Accounting of Subsampled Mechanisms under Composition" (IEEE SaTML 2025): fixed-size batches and other accounting pitfalls.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/05-subsampling.md).
