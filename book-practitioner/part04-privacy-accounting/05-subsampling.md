# Amplification by subsampling

## Why this matters

Private training often uses stochastic gradient descent (SGD) adapted to differential privacy (DP), called DP-SGD. Each step clips record gradients to a maximum length, the **clipping norm**, sums them and adds Gaussian noise. The noise standard deviation equals the clipping norm times the **noise multiplier** $\sigma$. If every step used the full dataset, 10,000 steps at $\sigma=1$ would cost $\varepsilon=5425.5$ at $\delta=10^{-5}$, providing no useful guarantee ([Rényi DP](01-renyi-dp.md)).

DP-SGD reduces this cost by using a random batch at each step. Let $q$ be the **sampling rate**, the probability that a record is included. Uncertainty about whether the record was sampled provides **amplification by subsampling**. The amplification depends on the sampling method, so the accountant must match the training sampler. A sampler mismatch causes larger accounting errors than the choice of accountant (@sec-sub-dpsgd).

## One step under Poisson sampling {#sec-sub-eps-delta}

Under **Poisson sampling**, each record independently enters the batch with probability $q$. This sampler pairs with **add/remove** neighbours, which differ by one extra record. Balle, Barthe and Gaboardi (2018) give the tightest one-step bound for this relation. If a full-dataset step is $(\varepsilon,\delta)$-DP, where $\delta$ is the slack defined in Part 3, then the same step on a Poisson sample is $(\varepsilon',\delta')$-DP, with
$$\varepsilon' = \ln\big(1 + q(e^\varepsilon - 1)\big), \qquad \delta' = q\delta.$$ {#eq-sub-amplify}

The amplification depends on the step's $\varepsilon$. For small $\varepsilon$, the cost is about $q\varepsilon$: sampling 1% makes a 1-DP step cost 0.0170, about one hundred times less. For large $\varepsilon$, the cost is approximately
$$\varepsilon' \approx \varepsilon - \ln(1/q),$$
so subsampling subtracts only a fixed amount; a 10-DP step on the same sample still costs 5.3993. The transition occurs near $\varepsilon=\ln(1/q)$. Subsampling therefore helps most when each step is already fairly private, as in DP-SGD's many small steps.

This result requires two conditions. Each record must be sampled independently with probability exactly $q$, regardless of other batch members. The batch must also remain secret; an attacker who sees the sampled records, such as in a log, knows whether a step included the target.

## Match the sampler to the accountant {#sec-sub-schemes}

Most training code uses another sampler. Some implementations draw a new fixed-size batch of $B$ from $n$ records at every step. Others shuffle once per epoch and divide the data into consecutive batches, a method called **shuffle and partition**.

Each sampler requires accounting matched to its neighbouring relation. Under **replace-one**, datasets have the same size and differ in one record's value. Under **zero-out**, the target's gradient is set to zero. @tbl-sub-matching lists the corresponding DP-SGD events in Google's `dp_accounting`; a Rényi differential privacy (RDP) or privacy loss distribution (PLD) accountant composes them.

| Sampler | Neighbours | What to account for | `dp_accounting` step |
| --- | --- | --- | --- |
| Poisson, rate $q$ | add/remove | A Poisson accountant with $q$ and $\sigma$ | `PoissonSampledDpEvent(q, GaussianDpEvent(σ))` |
| Poisson, rate $q$ | replace-one | The same event, at $\sigma$, in a PLD accountant built with `NeighboringRelation.REPLACE_ONE`, which models the swap itself | `PoissonSampledDpEvent(q, GaussianDpEvent(σ))` |
| Fixed size $B$ of $n$ | add/remove | A Poisson accountant with rate $B/n$ and noise multiplier $\sigma/2$ | `PoissonSampledDpEvent(B/n, GaussianDpEvent(σ/2))` |
| Fixed size $B$ of $n$ | replace-one | Sampling without replacement at noise multiplier $\sigma/2$, in the RDP accountant only | `SampledWithoutReplacementDpEvent(n, B, GaussianDpEvent(σ/2))` |
| Shuffle and partition | zero-out | No amplification: each epoch is one full Gaussian release | `GaussianDpEvent(σ)` once per epoch |

: What to account for under each sampler and neighbouring relation in DP-SGD. {#tbl-sub-matching}

The factor of two for fixed-size batches follows from the gradient sum. Under Poisson sampling, adding or removing one record changes the clipped sum by at most one clipping norm. A fixed-size batch has no extra slot, so a sampled record replaces another and the sum can change by twice as much (Lebeda et al. 2025). This is equivalent to halving the noise multiplier. Replace-one also swaps records and doubles the change; for fixed-size batches, only `dp_accounting`'s RDP accountant supports this relation.

Shuffle and partition provides no amplification because each record appears in exactly one batch per epoch. An attacker only needs to identify that batch. Chua et al. (2024) construct a dataset where the target gradient points against all others, making its batch distinguishable. At $q=0.01$ and $\sigma=1$, one shuffled epoch has $\varepsilon$ at least 4.06, while a Poisson accountant reports at most 1.21. The valid bound is one full Gaussian release per epoch, as shown in the table, because each record is used once.

Inspect the data loader before accepting a reported $\varepsilon$. A PyTorch `DataLoader` with `shuffle=True` and `batch_size` uses shuffle and partition. By default, Opacus replaces it with Poisson sampling: `PrivacyEngine.make_private` sets `poisson_sampling=True`. Its `batch_sampler` is a `UniformWithReplacementSampler` that independently samples each record, despite its name. Batch sizes therefore vary, which you can verify in the training log. If using another sampler, account for the matching row of @tbl-sub-matching and report the sampler with $\varepsilon$.

## Accounting each sampler {#sec-sub-dpsgd}

Do not account for DP-SGD by applying @eq-sub-amplify separately to every step. This gives each step a very small share of $\delta$, which raises its per-step $\varepsilon$ and limits amplification. For 10,000 steps, advanced composition then gives 24379.55, worse than the 5425.5 bound without subsampling. An RDP accountant instead composes the Rényi curves and converts once, giving 6.71 ([Rényi DP](01-renyi-dp.md)). A PLD accountant is tighter ([Numerical accounting](06-numerical-accounting.md)).

The snippet accounts for the run under four rows of @tbl-sub-matching, using 50,000 records and batches of 500 ($q=0.01$). It uses $\delta=10^{-5}$ for comparison within this part, not the smaller reporting value ([How small δ must be](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta)).

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

Poisson sampling costs 6.19. With the same noise, fixed-size batches cost 43.37 under add/remove neighbours, about seven times more; shuffle and partition costs 91.82 under zero-out. The replace-one value, 118.73, protects a record's value rather than its presence, so compare it only with other replace-one results. The sampler changes $\varepsilon$ by at least sevenfold, while the accountant changes it by less than 10%: RDP gives 6.71 and PLD gives 6.19.

Before composing an unfamiliar event, check `PLDAccountant().supports(event)`. For example, it returns `False` for `SampledWithoutReplacementDpEvent`.

## Batch size, epochs and noise {#sec-sub-small-q}

Subsampling also affects run planning. Let $E$ be the number of epochs; at sampling rate $q$, the run takes $E/q$ steps. For small $q$ and a large noise multiplier, its Rényi cost at order $\alpha$ is approximately
$$\varepsilon(\alpha) \approx \frac{E\,q\,\alpha}{2\sigma^2}.$$
At a fixed epoch count, a smaller sampling rate costs less privacy: each step costs about $q^2$, while the step count grows as $1/q$. Use this approximation only for planning. At high orders, the true curve rises substantially above it, beginning around order 9 for $\sigma=1$ and $q=0.01$.

Privacy is not the only cost: noise also affects the averaged gradient. If $C$ is the clipping norm, noise on an average over $B$ records has standard deviation $\sigma C/B$. Compare plans using $\sigma/B$, not $\sigma$ alone. At the same $\varepsilon$, a batch ten times larger needs a larger $\sigma$, but its $\sigma/B$ is six times smaller ([Calibrate and account a DP-SGD run](10-calibrate-dp-sgd.md)).

Plan in this order: choose batch size $B$ and epochs $E$ based on model accuracy. They determine $q=B/n$ and the step count, where $n$ is the filtered training-set size. Then use an accountant to find $\sigma$. Specify the numerical accountant in Opacus's `get_noise_multiplier`, whose default is RDP ([Calibrate and account a DP-SGD run](10-calibrate-dp-sgd.md)).

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
