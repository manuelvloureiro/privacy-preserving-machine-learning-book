# DP-FTRL and matrix factorization

## What it is {#sec-mfac-what}

Gradient descent without momentum adds up its steps, so its weights after each step are fixed by the sum of all clipped gradients so far, a **prefix sum**. Stochastic gradient descent (SGD) with differential privacy (DP), [DP-SGD](02-dp-sgd.md), adds fresh noise at every step, and the noise in the prefix sums then grows like a random walk. Matrix mechanisms correlate the noise across steps, so that each step's noise partly cancels earlier noise in the sums.

Stack the gradient sums of the steps as the rows of a matrix $x$. The prefix sums are then $Ax$, where $A$ has ones on and below the diagonal. The **matrix factorization mechanism** writes $A$ as a decoder $L$ times an encoder $R$. It adds a matrix of Gaussian noise, written $z$, to the encoded gradients and decodes the result,
$$\hat S = L(Rx + z) = Ax + Lz.$$
A record enters only through the column of $R$ for the step it joins, so privacy depends on the longest column of the encoder, called its sensitivity. Accuracy depends on the decoder, whose row $t$ carries the noise into the $t$-th prefix sum.

The first such method ran the follow the regularized leader (FTRL) optimizer on sums released by a binary tree, and is called DP-FTRL (Kairouz et al. 2021). The usual choice today is the **square-root factorization**, which uses the lower-triangular square root of $A$ for both factors. At equal privacy over one epoch of 100 steps, it leaves about a quarter of independent noise's standard deviation in the final sum.

## When you meet it {#sec-mfac-when}

You meet matrix mechanisms when your pipeline cannot honestly claim Poisson sampling, the random batches that DP-SGD's accountant assumes. Fixed batches cut from a shuffled order fail it, and so do data read in arrival order and, in federated learning, phones that check in when they can. Accounting them as Poisson understates $\varepsilon$ ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md#sec-sub-schemes)), and a matrix mechanism needs no sampling.

Giving up amplification is expensive with independent noise, and correlated noise wins most of it back. On the full edition's synthetic task, trained in a fixed order for ten epochs at $\varepsilon = 2$, independent noise reaches a test accuracy of 0.8641. The banded square root of @sec-mfac-wrong reaches 0.9491, against 0.9493 for DP-SGD with Poisson sampling. Google trained its Gboard keyboard models this way (Xu et al. 2023).

So if you can implement Poisson sampling, use DP-SGD with Opacus. If you cannot, use a banded matrix mechanism.

## How to use it {#sec-mfac-use}

Google's JAX Privacy implements banded matrix factorization. The block below was not run, since JAX is not in this book's environment. It cycles through 100 fixed groups of records, one per step, so each record takes part once every 100 steps.

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

The training loop draws each batch from `plan.batch_selection_strategy` and sums its clipped gradients with `plan.clipped_grad`. It passes the sum through `plan.noise_addition_transform` before the optimizer sees it. A `dp_accounting` accountant turns `plan.dp_event` into the $\varepsilon$ you report. The module is marked experimental, so pin the version.

## What goes wrong {#sec-mfac-wrong}

The pitfalls come from records that take part more than once. Over $k$ epochs a record enters through $k$ columns of the encoder. The square root's columns for nearby steps are nonzero in the same rows, so they add up and the sensitivity grows faster than $\sqrt k$. **Banding** keeps a number $b$ of the encoder's diagonals, the main one and those just below it. It needs every two participations of a record to be at least $b$ steps apart, a condition called **$b$-min-separation**. The record's columns then share no rows, so the sensitivity is at most $\sqrt k$ times the longest column, as for independent noise.

The data order is therefore part of the mechanism. Fix it before training, independently of the data, and keep the same permutation every epoch. Reshuffling each epoch can put a record at the end of one epoch and the start of the next, one step apart, which breaks the banded bound.

The neighbouring relation changes the numbers too. The analysis here leaves a neighbour's slot empty, so a step moves by at most the clipping norm. If a neighbour replaces the record with another instead, every sensitivity doubles. So state the relation your accountant assumes next to every $\varepsilon$.

The accounting library enforces this. In `dp_accounting` 0.6.0, the event for tree aggregation, `SingleEpochTreeAggregationDpEvent`, is accepted only by the Rényi differential privacy (RDP) accountant built with `NeighboringRelation.REPLACE_SPECIAL`, under which a neighbour replaces the record with an empty one. An accountant built for add/remove, the default for DP-SGD, raises `UnsupportedEventError`, and the privacy loss distribution (PLD) accountant does not support the event at all. So DP-FTRL and DP-SGD runs cannot share one add/remove ledger.

## Further reading

- Choquette-Choo et al., "(Amplified) Banded Matrix Factorizations: A Unified Approach to Private Training" (NeurIPS 2023): banding, min-separation and amplification.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/04-matrix-mechanisms.md).
