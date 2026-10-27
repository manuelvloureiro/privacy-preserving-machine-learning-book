# Amplification by subsampling

## Why this matters

Most private training uses stochastic gradient descent (SGD) made to satisfy differential privacy (DP), a method called DP-SGD. Each step clips every record's gradient to a maximum length, the **clipping norm**, sums the clipped gradients and adds Gaussian noise. The noise's standard deviation is the clipping norm times the **noise multiplier** $\sigma$. Suppose every record paid the cost of a full Gaussian step each time. Then ten thousand steps at $\sigma = 1$ would cost a privacy level $\varepsilon$ of 5425.5 at a slack $\delta$ of $10^{-5}$, no guarantee at all ([Rényi DP](01-renyi-dp.md)).

DP-SGD avoids this cost because each step uses only a small random batch. Write $q$ for the **sampling rate**, the probability that a given record is in the batch. An attacker must then first guess whether the step used the record at all, and that doubt is worth a lot. This **amplification by subsampling** depends on how the batch was drawn. So the accountant, the code that adds up a run's privacy cost, must assume the sampler that the training code runs. A wrong sampler is the largest single error in DP-SGD accounting, far larger than the choice of accountant (@sec-sub-dpsgd).

## One step under Poisson sampling {#sec-sub-eps-delta}

The simplest sampler is **Poisson sampling**: each record enters the batch by its own coin flip, with probability $q$, independently of every other record. It goes with **add/remove** neighbours, two datasets of which one has an extra record. For this pair, Balle, Barthe and Gaboardi (2018) give the best possible one-step result. Suppose a step is $(\varepsilon, \delta)$-DP when it uses every record, where $\delta$ is the small slack of Part 3. Then the same step on a Poisson sample is $(\varepsilon', \delta')$-DP with
$$\varepsilon' = \ln\big(1 + q(e^\varepsilon - 1)\big), \qquad \delta' = q\delta.$$ {#eq-sub-amplify}

Small and large $\varepsilon$ behave very differently. For small $\varepsilon$ the amplified cost is about $q\varepsilon$. So a 1% sample makes a step about a hundred times cheaper, and a 1-DP step costs 0.0170. For large $\varepsilon$ the amplified cost is about
$$\varepsilon' \approx \varepsilon - \ln(1/q),$$
so sampling only subtracts a fixed amount, and a 10-DP step on the same sample still costs 5.3993. The change happens where $\varepsilon$ is near $\ln(1/q)$. So amplification helps most when each step is already fairly private, which is why DP-SGD takes many cheap steps.

The result has two conditions. First, each record's coin must be independent of the others, so that every record is drawn with probability exactly $q$, whatever else is in the batch. Second, the batch must stay secret, because an attacker who reads the list of drawn records, for example in a log, knows whether the step used the target.

## Match the sampler to the accountant {#sec-sub-schemes}

Most training code does not draw Poisson batches. Some draws a new random batch of a fixed size $B$ from the $n$ records at each step. More often, it shuffles the data once per epoch and cuts it into consecutive batches, which is **shuffle and partition**.

Each sampler needs its own accounting under each neighbouring relation. Besides add/remove, two relations are in use. Under **replace-one**, the two datasets have the same size and differ in one record's value, and under **zero-out**, the target's gradient becomes zero. @tbl-sub-matching gives DP-SGD's accounting for each case as events of Google's `dp_accounting` library, which a Rényi differential privacy (RDP) or privacy loss distribution (PLD) accountant then composes.

| Sampler | Neighbours | What to account for | `dp_accounting` step |
| --- | --- | --- | --- |
| Poisson, rate $q$ | add/remove | A Poisson accountant with $q$ and $\sigma$ | `PoissonSampledDpEvent(q, GaussianDpEvent(σ))` |
| Poisson, rate $q$ | replace-one | The same event, at $\sigma$, in a PLD accountant built with `NeighboringRelation.REPLACE_ONE`, which models the swap itself | `PoissonSampledDpEvent(q, GaussianDpEvent(σ))` |
| Fixed size $B$ of $n$ | add/remove | A Poisson accountant with rate $B/n$ and noise multiplier $\sigma/2$ | `PoissonSampledDpEvent(B/n, GaussianDpEvent(σ/2))` |
| Fixed size $B$ of $n$ | replace-one | Sampling without replacement at noise multiplier $\sigma/2$, in the RDP accountant only | `SampledWithoutReplacementDpEvent(n, B, GaussianDpEvent(σ/2))` |
| Shuffle and partition | zero-out | No amplification: each epoch is one full Gaussian release | `GaussianDpEvent(σ)` once per epoch |

: What to account for under each sampler and neighbouring relation in DP-SGD. {#tbl-sub-matching}

The factor of two for fixed-size batches comes from the gradient sum. Under Poisson sampling, to add or remove a record moves the clipped sum by at most the clipping norm. A fixed-size batch has no space for an extra record. So a drawn record takes the place of another, and the sum can move twice as far (Lebeda et al. 2025). Twice the shift with the same noise is the same as the original shift with half the noise multiplier. Replace-one swaps a record by definition, so its shift doubles too, and for it only the RDP accountant of `dp_accounting` covers fixed-size batches.

Shuffle and partition is worse, because it has no source of amplification. In an epoch the record is in exactly one batch and is never absent, so the attacker only has to find which batch held it. Chua et al. (2024) build a dataset in which the target's gradient points against all the others, so that its batch stands out. There, at $q = 0.01$ and $\sigma = 1$, one shuffled epoch provably has an $\varepsilon$ of at least 4.06, but a Poisson accountant reports at most 1.21. The honest bound is the table's last row, one full Gaussian release per epoch, because each epoch uses the record exactly once.

So read the data loader before you trust a reported $\varepsilon$. A PyTorch `DataLoader` with `shuffle=True` and a `batch_size` is shuffle and partition. Opacus replaces the loader with a Poisson one by default, because `PrivacyEngine.make_private` sets `poisson_sampling=True` unless you turn it off. Its `batch_sampler` is then a `UniformWithReplacementSampler`, which flips one coin per record despite its name. So batch sizes change from step to step, which you can check in the training log. If you cannot use Poisson sampling, report under the matching row of @tbl-sub-matching. Name the sampler next to $\varepsilon$.

## Accounting each sampler {#sec-sub-dpsgd}

Never account DP-SGD step by step with @eq-sub-amplify. That route gives each step a tiny share of $\delta$, and a tiny $\delta$ forces a large per-step $\varepsilon$, which sampling barely reduces, as for the 10-DP step above. On 10,000 steps, advanced composition then gives 24379.55, worse even than 5425.5 without sampling. An RDP accountant instead adds the steps' Rényi curves order by order and converts the total to $(\varepsilon, \delta)$ once, which gives 6.71 ([Rényi DP](01-renyi-dp.md)). A PLD accountant is tighter still ([Numerical accounting](06-numerical-accounting.md)).

The snippet below accounts that run under four rows of @tbl-sub-matching, with 50,000 records in batches of 500, so $q = 0.01$. It uses $\delta = 10^{-5}$ to compare with the rest of the part, not the smaller $\delta$ of a report ([How small δ must be](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta)).

```python
from dp_accounting import GaussianDpEvent, NeighboringRelation, PoissonSampledDpEvent
from dp_accounting import SampledWithoutReplacementDpEvent, SelfComposedDpEvent
from dp_accounting.pld import PLDAccountant
from dp_accounting.rdp import RdpAccountant

n, batch, sigma, steps, delta = 50_000, 500, 1.0, 10_000, 1e-5  # q = 0.01, 100 epochs
poisson = SelfComposedDpEvent(PoissonSampledDpEvent(batch / n, GaussianDpEvent(sigma)), steps)
fixed_size = SelfComposedDpEvent(PoissonSampledDpEvent(batch / n, GaussianDpEvent(sigma / 2)), steps)  # add/remove
shuffled = SelfComposedDpEvent(GaussianDpEvent(sigma), 100)  # one Gaussian release per epoch, no amplification
pld = [PLDAccountant(value_discretization_interval=1e-3).compose(e) for e in (poisson, fixed_size, shuffled)]  # 1e-3: the PLD's grid spacing
print(" ".join(f"{accountant.get_epsilon(delta):.2f}" for accountant in pld))  # 6.19 43.37 91.82
replace_one = RdpAccountant(neighboring_relation=NeighboringRelation.REPLACE_ONE)
swap = SampledWithoutReplacementDpEvent(n, batch, GaussianDpEvent(sigma / 2))  # replace-one: the sum moves by up to 2C
replace_one.compose(SelfComposedDpEvent(swap, steps))
print(f"{replace_one.get_epsilon(delta):.2f}")  # 118.73
```

Poisson sampling costs 6.19. With the same noise, fixed-size batches cost 43.37 under add/remove neighbours, about seven times more, and shuffling costs 91.82 under zero-out. The replace-one value, 118.73, protects a record's value, not its presence, so compare it only with other replace-one numbers. Thus the sampler changes $\varepsilon$ sevenfold or more, while the accountant changes it by less than 10%, from 6.71 by RDP to 6.19 by PLD.

Before you compose an unusual event, check it with `PLDAccountant().supports(event)`, which returns `False` for a `SampledWithoutReplacementDpEvent`, for example.

## Batch size, epochs and noise {#sec-sub-small-q}

Amplification also changes how you plan a run. Write $E$ for the number of epochs, so that a run takes $E/q$ steps at sampling rate $q$. For small $q$ and a large noise multiplier, the run's Rényi curve, its cost at each order $\alpha$, is then roughly
$$\varepsilon(\alpha) \approx \frac{E\,q\,\alpha}{2\sigma^2}.$$
So at a fixed number of epochs a smaller sampling rate costs less privacy, because each step costs about $q^2$ while the number of steps grows only as $1/q$. Use this only to plan, never as the reported number, because at high orders the true curve rises far above it (from about order 9 at $\sigma = 1$ and $q = 0.01$).

Privacy is not the only cost, because the noise also changes the averaged gradient. With $C$ for the clipping norm, the noise on the average of $B$ records has standard deviation $\sigma C/B$. So compare plans by $\sigma/B$, not by $\sigma$. At the same $\varepsilon$, a ten times larger batch needs a larger $\sigma$, but its $\sigma/B$ is six times smaller ([Calibrate and account a DP-SGD run](10-calibrate-dp-sgd.md)).

So plan in this order. First fix the batch size $B$ and the epochs $E$ that your accuracy needs. They set $q = B/n$ and the number of steps, where $n$ is the size of the dataset you actually train on, after any filters. Then let the accountant find $\sigma$. Name the numerical accountant in the Opacus call `get_noise_multiplier`, because its default is RDP ([Calibrate and account a DP-SGD run](10-calibrate-dp-sgd.md)).

## Cheat sheet

| Situation | What to use | Notes |
| --- | --- | --- |
| One step, Poisson, add/remove | $\varepsilon' = \ln(1 + q(e^\varepsilon - 1))$, $\delta' = q\delta$ | About $q\varepsilon$ for small $\varepsilon$ |
| DP-SGD, Poisson loader | PLD or RDP accountant with $q$ and $\sigma$ | 6.19 by PLD for the running example |
| DP-SGD, fixed-size batches, add/remove | Poisson accountant with $B/n$ and $\sigma/2$ | 43.37, about seven times the Poisson value |
| DP-SGD, fixed-size batches, replace-one | `SampledWithoutReplacementDpEvent` at $\sigma/2$ in the RDP accountant | Not comparable with add/remove numbers |
| DP-SGD, shuffle and partition | One Gaussian release per epoch, under zero-out | A Poisson number does not apply; better, switch the loader to Poisson sampling |

: Which accounting to use for each sampler. {#tbl-sub-cheat}

**Rules of thumb**

- Account for the sampler you run, check that its batches vary in size, and state the neighbouring relation next to $\varepsilon$.

## References

- Balle, Barthe & Gaboardi, "Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences" (NeurIPS 2018): amplification for each sampler and neighbouring relation.
- Lebeda, Regehr, Kamath & Steinke, "Avoiding Pitfalls for Privacy Accounting of Subsampled Mechanisms under Composition" (IEEE SaTML 2025): fixed-size batches and other accounting pitfalls.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/05-subsampling.md).
