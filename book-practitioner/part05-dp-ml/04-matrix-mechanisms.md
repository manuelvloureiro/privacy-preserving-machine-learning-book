# DP-FTRL and matrix factorization

## What it is {#sec-mfac-what}

Gradient descent without momentum accumulates clipped gradients, so weights after each step depend on their **prefix sum**. Differential privacy (DP) applied to stochastic gradient descent (SGD) yields [DP-SGD](02-dp-sgd.md), which adds fresh noise at every step. Prefix-sum noise then grows like a random walk. Matrix mechanisms correlate noise across steps so some noise cancels in later sums.

Place the step gradients in the rows of matrix $x$. Their prefix sums are $Ax$, where $A$ has ones on and below its diagonal. The **matrix factorization mechanism** factors $A$ into decoder $L$ and encoder $R$, adds Gaussian noise $z$ to the encoded gradients, then decodes:
$$\hat S = L(Rx + z) = Ax + Lz.$$
Each record affects only the encoder column for its step, so privacy depends on the encoder's largest column norm. Accuracy depends on the decoder: row $t$ determines how noise affects prefix sum $t$.

The first method applied follow the regularized leader (FTRL) to binary-tree sums and is called DP-FTRL (Kairouz et al. 2021). The common **square-root factorization** uses the lower-triangular square root of $A$ for both factors. At equal privacy over 100 steps, its final-sum noise standard deviation is about one quarter that of independent noise.

## When you meet it {#sec-mfac-when}

Use matrix mechanisms when the pipeline cannot use the Poisson sampler assumed by DP-SGD accounting. Examples include fixed batches from a shuffled dataset, data processed in arrival order and federated devices that report when available. Treating these as Poisson understates $\varepsilon$ ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md#sec-sub-schemes)); matrix mechanisms do not require subsampling.

Without sampling, independent noise can sharply reduce accuracy; correlated noise can recover much of it. On the full edition's synthetic task, ten fixed-order epochs at $\varepsilon=2$ give test accuracy 0.8641 with independent noise, 0.9491 with the banded square root of @sec-mfac-wrong and 0.9493 with Poisson-sampled DP-SGD. Use Opacus with Poisson sampling when possible; otherwise consider a banded matrix mechanism, as Google did for Gboard models (Xu et al. 2023).

## How to use it {#sec-mfac-use}

Google's JAX Privacy implements banded matrix factorization. The block below was not run, because JAX is not in this book's environment. It cycles through 100 fixed groups of records, one per step, so each record takes part once every 100 steps.

```text
# jax-privacy==2.0.0 (not run)
from jax_privacy.experimental.execution_plan import BandMFExecutionPlanConfig

config = BandMFExecutionPlanConfig.default(
    num_bands=100,       # at most the 100 steps between a record's turns
    iterations=1000,     # ten epochs of 100 steps
    epsilon=2.0,
    delta=1e-5,
    sampling_prob=1.0,   # all of the step's group joins: no amplification claimed
    l2_clip_norm=1.0,
)
plan = config.plan       # calibrates the noise multiplier with dp_accounting
```

The training loop samples batches with `plan.batch_selection_strategy`, clips and sums gradients with `plan.clipped_grad`, and adds noise with `plan.noise_addition_transform`. Use `dp_accounting` on `plan.dp_event` to calculate reported $\varepsilon$. The module is experimental; pin its version.

## What goes wrong {#sec-mfac-wrong}

Repeated participation creates the main risk. Across $E$ epochs, one record affects $E$ encoder columns. Nearby square-root columns overlap, increasing sensitivity faster than $\sqrt E$. **Banding** retains $b$ encoder diagonals: the main diagonal and $b-1$ below it. It requires each pair of a record's participations to be separated by at least $b$ steps (**$b$-min-separation**). The columns then share no rows, bounding sensitivity by $\sqrt E$ times the longest column, as with independent noise.

Data order is part of the mechanism. Fix it independently of private data and reuse the same permutation every epoch. Reshuffling can place a record at the end of one epoch and the start of the next, violating the separation condition.

The neighbouring relation also affects sensitivity. This analysis leaves a neighbour's slot empty, so one step changes by at most the clipping norm. Replacing a record with another doubles sensitivity; report the relation with each $\varepsilon$.

`dp_accounting` 0.6.0 enforces this restriction. Its tree-aggregation event is supported only by a Rényi differential privacy (RDP) accountant using `NeighboringRelation.REPLACE_SPECIAL`, where a record is replaced by an empty slot. Under DP-SGD's default add/remove relation, it raises `UnsupportedEventError`; the privacy loss distribution (PLD) accountant also rejects it. Therefore, DP-FTRL and DP-SGD runs cannot share an add/remove ledger.

## References

- Choquette-Choo et al., "(Amplified) Banded Matrix Factorizations: A Unified Approach to Private Training" (NeurIPS 2023): banding, min-separation and amplification.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/04-matrix-mechanisms.md).
