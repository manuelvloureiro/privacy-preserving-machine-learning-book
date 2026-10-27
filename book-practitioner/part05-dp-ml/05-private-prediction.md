# Private prediction: PATE and in-context learning

## What it is {#sec-pate-what}

Private prediction applies differential privacy (DP) to each *answer* instead of to the model. You split the private records into disjoint shards and train an ordinary model, a **teacher**, on each one. The teachers stay secret. Each query gets the winner of their vote after noise is added to the counts, private at a level $\varepsilon$ with a small slack $\delta$. A record lives in one shard, so it can move one teacher's vote to another label. That changes the counts by at most $\sqrt2$ in Euclidean length.

**GNMax** adds Gaussian noise with a standard deviation $\sigma$ to each count and releases only the label with the largest noisy count. In Rényi differential privacy (RDP), each answer costs a straight line of slope $1/\sigma^2$ in the order $\alpha$, and the slopes add
 ([Rényi DP](../part04-privacy-accounting/01-renyi-dp.md)). So write $\rho(\varepsilon, \delta)$ for the largest slope that still converts to your target. A fixed number $Q$ of answers, even adaptively chosen, then needs the noise
$$\sigma = \sqrt{\frac{Q}{\rho(\varepsilon, \delta)}}.$$ {#eq-pate-sigma}
For example, at $\delta = 10^{-5}$ a budget of $\varepsilon = 1$ allows a slope of 0.0306, so 1,000 answers need $\sigma = 181$.

## When you meet it {#sec-pate-when}

You meet private prediction when you cannot train a private model. The common case is a large language model (LLM) behind an interface. Its prompt holds private records, and its answers leak them. The fix splits the records into disjoint groups, prompts the model once per group, and releases the GNMax label (Duan et al. 2023). At $\varepsilon = 1$, Wu et al. (2024) answered 10,000 sentiment queries this way as accurately as one non-private prompt.

Otherwise the choice is between private answers and a private model, whose accuracy is fixed. The vote's noise grows with $\sqrt Q$, so answers win up to a **break-even** number of queries, and the model wins beyond it. On the full edition's 10,000-record synthetic task, the vote beats noisy stochastic gradient descent (SGD), [DP-SGD](02-dp-sgd.md), only on the first 18 queries at $\varepsilon = 1$. So compare the two at your budget and expected number of queries, with the DP-SGD baseline of [Train a model with DP-SGD](13-train-dp-sgd.md).

**PATE** (private aggregation of teacher ensembles) spends the budget once. It labels public, unlabelled inputs by GNMax and trains a new model, the **student**, on them. Only the student is released, at exactly the cost of its labels, so PATE needs public inputs from the private data's distribution. Papernot et al.'s MNIST student comes within a percentage point of non-private accuracy, but at a data-dependent $\varepsilon$ (@sec-pate-wrong). On the synthetic task, DP-SGD beats the best student at every budget.

## How to use it {#sec-pate-use}

`dp_accounting` accounts for the vote. One GNMax answer is a `GaussianDpEvent` whose noise multiplier, the noise over the sensitivity, is $\sigma/\sqrt2$.

```python
import numpy as np
from dp_accounting import GaussianDpEvent, SelfComposedDpEvent
from dp_accounting.pld import PLDAccountant
from dp_accounting.rdp import RdpAccountant

sigma, queries = 181.0, 1000  # noise per count, answers to release
event = SelfComposedDpEvent(GaussianDpEvent(sigma / np.sqrt(2)), queries)  # sensitivity sqrt(2)
rdp = RdpAccountant().compose(event).get_epsilon(1e-5)
pld = PLDAccountant().compose(event).get_epsilon(1e-5)
print(f"{rdp:.3f} {pld:.3f}")  # 0.999 0.914
```

The RDP accountant confirms $\varepsilon = 0.999$. The privacy loss distribution (PLD) accountant, exact for Gaussian noise, gives 0.914, so solving for $\sigma$ with it gives less noise. To solve, pass a function from $\sigma$ to this event to `dp_accounting.calibrate_dp_mechanism`. For many answers the PLD is much slower, and RDP is the fallback. When each query draws a fresh Poisson sample of records, as in Wu et al., wrap each event in a `PoissonSampledDpEvent`.

## What goes wrong {#sec-pate-wrong}

The guarantee needs each record in exactly one shard. So assign shards by a hash of the record's or the user's identifier. Never assign them by position in a file, where one inserted record shifts every later shard. Release the label only, and fix $Q$ in advance.

Noise this large wipes out close votes. With two labels and a gap $g$ between the counts, the vote returns the plurality with probability $\Phi(g/(\sqrt2\sigma))$, where $\Phi$ is the standard normal distribution function.

For example, the noise for 1,000 answers at $\varepsilon = 4$ is $\sigma = 51.8$. Then a lopsided vote of 98 to 2 is right with probability 0.9051, but a close vote of 52 to 48 only about half the time.
So measure the noiseless vote first, since no budget beats it. Adding teachers widens the gaps, since $\sigma$ does not depend on their number, until the shards get too small to train on.

Finally, beware of a data-dependent $\varepsilon$. When teachers agree, a sharper analysis charges less, but that smaller $\varepsilon$ is computed from the votes and leaks them. Until it is released with noise, it is a measurement, not a guarantee.

**Confident-GNMax** answers only queries on which a noisy check finds agreement, the cheap and accurate ones.

## References

- Papernot et al., "Scalable Private Learning with PATE" (ICLR 2018): GNMax, its data-dependent analysis and Confident-GNMax.
- Wu et al., "Privacy-Preserving In-Context Learning for Large Language Models" (ICLR 2024): subsampled voting over prompts.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/05-private-prediction.md).
