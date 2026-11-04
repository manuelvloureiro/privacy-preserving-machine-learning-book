# Private prediction: PATE and in-context learning

## What it is {#sec-pate-what}

Private prediction applies differential privacy (DP) to each answer rather than to stochastic gradient descent (SGD) training. Split private records into disjoint shards and train an ordinary **teacher** model on each. Keep teachers private. For each query, add noise to vote counts and release the winning label. The privacy level is $\varepsilon$, and $\delta$ is the failure-probability allowance. Each record belongs to one shard and can change one vote from one label to another, changing the count vector by at most $\sqrt2$ in Euclidean norm.

Let $\sigma$ denote the standard deviation of the added Gaussian noise. **GNMax** adds this noise to each count and releases the label with the largest noisy count. In Rényi differential privacy (RDP), each answer costs a line with slope $1/\sigma^2$ at order $\alpha$, the Rényi order parameter. Slopes compose additively ([Rényi DP](../part04-privacy-accounting/01-renyi-dp.md)). Let $\rho(\varepsilon,\delta)$ be the largest slope that converts to the target guarantee. For a fixed number $Q$ of answers, including adaptively chosen queries, the required noise is
$$\sigma = \sqrt{\frac{Q}{\rho(\varepsilon, \delta)}}.$$ {#eq-pate-sigma}
For example, set target privacy level $\varepsilon = 1$ and slack $\delta = 10^{-5}$. The allowed slope is 0.0306. Then 1,000 answers require $\sigma = 181$.

## When you meet it {#sec-pate-when}

Use private prediction when private model training is impractical. One example is a large language model (LLM) interface whose answers could expose records in prompts. Split records into disjoint groups, query the model once per group, and release a GNMax label (Duan et al. 2023). Wu et al. (2024) answered 10,000 sentiment queries with accuracy matching one non-private prompt. Their privacy level was $\varepsilon=1$.

The alternative is a private model with fixed accuracy. Vote noise grows as $\sqrt Q$, so private answers are preferable only up to a **break-even** query count. On the full edition's synthetic task, the data contains 10,000 records. Voting outperforms private SGD ([DP-SGD](02-dp-sgd.md)) only for the first 18 queries at privacy level $\varepsilon=1$. Compare both approaches at the expected query count and budget, using [Train a model with DP-SGD](13-train-dp-sgd.md) as the baseline.

**PATE** (private aggregation of teacher ensembles) labels public, unlabelled inputs with GNMax and trains a **student** model on those labels. Only the student is released, at the privacy cost of its labels. PATE therefore requires public inputs from the private data distribution. Papernot et al.'s MNIST student comes within one percentage point of non-private accuracy, but its $\varepsilon$ is data-dependent (@sec-pate-wrong). On the synthetic task, DP-SGD outperforms the best student at every budget.

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

The RDP accountant confirms $\varepsilon = 0.999$. The privacy loss distribution (PLD) accountant, exact for Gaussian noise, gives 0.914, so solving for $\sigma$ with it gives less noise. To solve, pass a function from $\sigma$ to this event to `dp_accounting.calibrate_dp_mechanism`. For many answers the PLD is much slower, so RDP is the fallback. When each query draws a fresh Poisson sample of records, as in Wu et al., wrap each event in a `PoissonSampledDpEvent`.

## What goes wrong {#sec-pate-wrong}

The guarantee needs each record in exactly one shard. So assign shards by a hash of the record's or the user's identifier, never by position in a file, where one inserted record shifts every later shard. Release the label only, and fix $Q$ in advance.

Noise this large wipes out close votes. With two labels and a gap $g$ between the counts, the vote returns the plurality with probability $\Phi(g/(\sqrt2\sigma))$, where $\Phi$ is the standard normal distribution function.

For example, 1,000 answers at privacy level $\varepsilon = 4$ require noise $\sigma = 51.8$. With that noise, a vote of 98 to 2 is correct with probability 0.9051. For a close vote of 52 to 48, accuracy is only about one half.
Measure the noiseless vote first, because no privacy budget can improve it. Adding teachers widens vote gaps because $\sigma$ does not depend on the teacher count. This helps only until the shards become too small to train on.

Finally, beware of a data-dependent $\varepsilon$. When teachers agree, a sharper analysis charges less, but that smaller $\varepsilon$ is computed from the votes and leaks them. Until it is released with noise, it is a measurement, not a guarantee.

**Confident-GNMax** answers only queries on which a noisy check finds agreement, the cheap and accurate ones.

## References

- Papernot et al., "Scalable Private Learning with PATE" (ICLR 2018): GNMax, its data-dependent analysis and Confident-GNMax.
- Wu et al., "Privacy-Preserving In-Context Learning for Large Language Models" (ICLR 2024): subsampled voting over prompts.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/05-private-prediction.md).
