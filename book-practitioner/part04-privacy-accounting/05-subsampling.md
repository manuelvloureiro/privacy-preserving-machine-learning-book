# Amplification by subsampling

## Why this matters

Most private models are trained with stochastic gradient descent (SGD) made to satisfy differential privacy (DP), a method called DP-SGD. Each step clips every record's gradient to a maximum length, the **clipping norm**, sums the clipped gradients and adds Gaussian noise. The noise has a standard deviation equal to the clipping norm times the **noise multiplier** $\sigma$. Suppose every record paid the full cost of a Gaussian mechanism at every step. Then ten thousand steps with $\sigma = 1$ would have a privacy level $\varepsilon$ of 5425.5 at $\delta = 10^{-5}$ ([Rényi DP](01-renyi-dp.md)), which is no guarantee at all.

DP-SGD escapes this cost because each step looks only at a small random batch. Write $q$ for the **sampling rate**, the probability that a given record lands in the batch. An attacker must then first guess whether the record was used at all, and that uncertainty is worth a lot. This effect is called **amplification by subsampling**, and it depends on exactly how the batch was drawn. So the accountant, the code that adds up a run's privacy cost, must assume the sampler that the training code actually runs. Getting that wrong is the largest single error in DP-SGD accounting, far larger than the difference between one accountant and another (@sec-sub-dpsgd).

## One step under Poisson sampling {#sec-sub-eps-delta}

The simplest sampler is **Poisson sampling**, in which each record enters the batch by its own coin flip, with probability $q$, independently of every other record. It goes with **add/remove** neighbours, two datasets of which one has an extra record. For that pairing, Balle, Barthe and Gaboardi (2018) give the best possible result for one step. As in Part 3, $\delta$ is the guarantee's small additive slack. If the step is $(\varepsilon, \delta)$-DP when it sees every record, then the same step run on a Poisson sample is $(\varepsilon', \delta')$-DP with
$$\varepsilon' = \ln\big(1 + q(e^\varepsilon - 1)\big), \qquad \delta' = q\delta.$$ {#eq-sub-amplify}

The formula behaves very differently for small and large $\varepsilon$. For small $\varepsilon$ the amplified cost is about $q\varepsilon$, so a 1% sample makes a step about a hundred times cheaper. With that sample, for example, a 1-DP step costs 0.0170. For large $\varepsilon$, though, the amplified cost is about
$$\varepsilon' \approx \varepsilon - \ln(1/q),$$
so sampling only subtracts a fixed amount. That is why a 10-DP step on the same sample still costs 5.3993. The switch happens where $\varepsilon$ is near $\ln(1/q)$. So amplification helps most when each step is already fairly private, which is why DP-SGD takes many cheap steps.

The result has two conditions. Each record's coin must be independent of the others, so that every record is drawn with probability exactly $q$ whatever else is in the batch. And the batch must stay secret, because an attacker who sees the list of records drawn, for example in a readable log, knows whether the target was used.

## Match the sampler to the accountant {#sec-sub-schemes}

Most training code does not draw Poisson batches. Some code draws a fresh random batch of a fixed size $B$ from the $n$ records at each step. Much more code shuffles the data once per epoch and cuts the list into consecutive batches, which is called **shuffle and partition**.

Each sampler needs its own accounting under each neighbouring relation. Under **replace-one**, the datasets have the same size and differ in one record's value, and under **zero-out**, the target's gradient is set to zero. @tbl-sub-matching gives the accounting for DP-SGD. It writes each step as an event of Google's `dp_accounting` library, which a Rényi differential privacy (RDP) or privacy loss distribution (PLD) accountant then composes.

| Sampler | Neighbours | What to account for | `dp_accounting` step |
| --- | --- | --- | --- |
| Poisson, rate $q$ | add/remove | A Poisson accountant with $q$ and $\sigma$ | `PoissonSampledDpEvent(q, GaussianDpEvent(σ))` |
| Poisson, rate $q$ | replace-one | The same event, at $\sigma$, in a PLD accountant built with `NeighboringRelation.REPLACE_ONE`, which models the swap itself | `PoissonSampledDpEvent(q, GaussianDpEvent(σ))` |
| Fixed size $B$ of $n$ | add/remove | A Poisson accountant with rate $B/n$ and noise multiplier $\sigma/2$ | `PoissonSampledDpEvent(B/n, GaussianDpEvent(σ/2))` |
| Fixed size $B$ of $n$ | replace-one | Sampling without replacement at noise multiplier $\sigma/2$, in the RDP accountant only | `SampledWithoutReplacementDpEvent(n, B, GaussianDpEvent(σ/2))` |
| Shuffle and partition | zero-out | No amplification: each epoch is one full Gaussian release | `GaussianDpEvent(σ)` once per epoch |

: What to account for under each sampler and neighbouring relation in DP-SGD. {#tbl-sub-matching}

The factor of two for fixed-size batches comes from the gradient sum. Under Poisson sampling, adding or removing a record moves the clipped sum by at most the clipping norm. A fixed-size batch has no room for an extra record, so a drawn record takes another's place, and the sum can move by twice as much (Lebeda et al. 2025). Twice the shift with the same noise is the same as the original shift with half the noise multiplier. Replace-one swaps a record by definition, so its shift doubles too, but the Poisson accountant does not cover that relation. The RDP accountant of `dp_accounting` does, given the replace-one relation and the halved noise multiplier.

Shuffle and partition is worse, because the source of amplification is missing. Within an epoch the record is in exactly one batch and is never absent, so the attacker only has to find which batch held it. Chua et al. (2024) build a dataset in which the target's gradient points against all the others, so its batch's sum stands out. Take this part's running example, with a 1% sampling rate and $\sigma = 1$, under zero-out at $\delta = 10^{-5}$. On that dataset, one shuffled epoch provably has an $\varepsilon$ of at least 4.06, while a Poisson accountant reports at most 1.21. So a Poisson number reported for shuffled training can be provably false. The honest bound for shuffled training is the last row of the table, one full Gaussian release per epoch, since each epoch uses the record exactly once.

So read the data loader before you trust a reported $\varepsilon$. A PyTorch `DataLoader` with `shuffle=True` and a `batch_size` is shuffle and partition. Opacus replaces the loader with a Poisson one by default, since `PrivacyEngine.make_private` takes `poisson_sampling=True` unless you turn it off. You can also convert a loader by hand with `opacus.data_loader.DPDataLoader.from_data_loader`. Either way, the loader's `batch_sampler` is then a `UniformWithReplacementSampler`, which despite its name flips one coin per record. So its batches vary in size from step to step, a quick check in the training log. When the largest batches do not fit in memory, Opacus's `BatchMemoryManager` splits each one into pieces that fit. The optimizer still takes one noisy step, and the accountant counts one step, per sampled batch ([Train a model with DP-SGD](../part05-dp-ml/13-train-dp-sgd.md)). If you cannot use Poisson sampling, report under the matching row of @tbl-sub-matching, and name the sampler next to $\varepsilon$.

## Accounting each sampler {#sec-sub-dpsgd}

DP-SGD is never accounted step by step with @eq-sub-amplify. That route splits $\delta$ into a tiny share per step, and a tiny $\delta$ forces a large per-step $\varepsilon$, which sampling barely reduces. On the running example's 10,000 steps it gives 24379.55 with advanced composition, worse even than 5425.5 without sampling. An RDP accountant instead adds the steps' Rényi curves order by order and converts the total to $(\varepsilon, \delta)$ once, which gives 6.71 ([Rényi DP](01-renyi-dp.md)). A PLD accountant is tighter still ([Numerical accounting](06-numerical-accounting.md)).

The snippet below accounts the running example under the four rows of @tbl-sub-matching, with a training set of $n = 50{,}000$ records and batches of 500. It uses $\delta = 10^{-5}$, the value at which this part compares accountants. A report for 50,000 records would use the book's rule, $\delta \le 1/n^{1.1}$ rounded down to a power of ten, which gives $10^{-6}$ ([Report and review an ε claim](09-choosing-an-accountant.md)).

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

Poisson sampling costs 6.19. The same noise with fixed-size batches costs 43.37 under add/remove, about seven times more, and shuffling with no amplification costs 91.82 under zero-out. The replace-one value, 118.73, protects a different thing, the value of a record rather than its presence, so compare it only with other replace-one numbers. The sampler moves $\varepsilon$ by a factor of seven or more. On this run the accountant moves it by less than 10%, from 6.71 by RDP to 6.19 by PLD, but the gap is wider on shorter runs, 14.6% at 1,000 steps ([Report and review an ε claim](09-choosing-an-accountant.md)).

Two more checks belong here. `PLDAccountant().supports(event)` returns `False` for a `SampledWithoutReplacementDpEvent` under either relation, so call `supports` before you compose an unusual event. And accountants use an exact curve for the subsampled Gaussian but looser general bounds for other mechanisms (Zhu & Wang 2019), so a subsampled step of another mechanism carries more slack.

## Batch size, epochs and noise {#sec-sub-small-q}

Amplification also shapes how you plan a run. Write $E$ for the number of epochs, so that a run takes $E/q$ steps. For a small sampling rate and a large noise multiplier, the run's Rényi cost at an order $\alpha$ is roughly
$$\varepsilon(\alpha) \approx \frac{E\,q\,\alpha}{2\sigma^2}.$$
Use this only for planning, never as the reported number, because at high orders the true curve rises far above it (from about order 9 at $\sigma = 1$ and $q = 0.01$). It says that at a fixed number of epochs, a smaller sampling rate costs less privacy. Each step costs about $q^2$, and the number of steps grows only as $1/q$.

Privacy is not the only cost, though, because the noise lands on the averaged gradient. With $C$ for the clipping norm, the noise on the sum of a batch of $B$ records has standard deviation $\sigma C$, so the noise on their average has $\sigma C/B$. So compare plans by $\sigma/B$, not by $\sigma$. The planning table of [Calibrate and account a DP-SGD run](10-calibrate-dp-sgd.md) shows the trade-off on the 30,000 training rows of the Adult census data. For $\varepsilon = 4$ over 10 epochs, a batch of 100 needs $\sigma = 0.62$ and a batch of 1,000 needs $\sigma = 0.97$. Yet the larger batch trains with far less noise, since $\sigma/B$, which the table lists times 1,000, falls from 6.16 to 0.97.

Plan in this order, then. First fix the batch size $B$ and the number of epochs $E$ that your accuracy needs. They set the sampling rate $q$, which is $B/n$, and the number of steps $E/q$. Here $n$ is the size of the dataset you actually train on, after any filtering, because a wrong $n$ gives a wrong $\varepsilon$. Then let the accountant solve for $\sigma$. [Calibrate and account a DP-SGD run](10-calibrate-dp-sgd.md) does this with Opacus's `get_noise_multiplier`, naming the numerical accountant because the default is RDP, and checks the result with a second accountant.

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

- Account for the sampler you run, and state the neighbouring relation next to $\varepsilon$.
- Check that the loader's batches vary in size. Constant batch sizes mean it is not Poisson sampling.

## Further reading

- Balle, Barthe & Gaboardi, "Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences" (NeurIPS 2018): amplification for each sampler and neighbouring relation.
- Lebeda, Regehr, Kamath & Steinke, "Avoiding Pitfalls for Privacy Accounting of Subsampled Mechanisms under Composition" (IEEE SaTML 2025): fixed-size batches and other accounting pitfalls.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/05-subsampling.md).
