# Reconstruction attacks

## What it is

A **reconstruction attack** recovers records from published aggregates because each count is a linear equation in the values of the people counted. Suppose each of $n$ people has a secret bit, represented in vector $d$. A **subset-sum query** counts the people in a set whose bit is 1. The attacker's $m$ queries form a binary matrix $A$, one query per row. The **curator** publishes the true answers plus an error vector $e$:
$$a = Ad + e.$$

**Least squares** finds real-valued records whose query answers are closest to the published answers, then rounds each value to 0 or 1. Suppose each query includes each person with probability one half and the noise is normal with standard deviation $\sigma$. Least squares then recovers approximately the following fraction
$$\Phi\left(\frac{\sqrt{m - n}}{4\sigma}\right)$$ {#eq-recon-ls-random}
of the bits, where $\Phi$ is the standard normal distribution function. The baseline is the **majority guess**, which assigns the more common value to every person and is correct on half of balanced bits. With several times more queries than people, noise well below $\sqrt n$ can reveal the data. If the curator answers all $2^n$ subset sums, only error of order $n$ prevents reconstruction (Dinur and Nissim 2003).

These results form the **Fundamental Law of Information Recovery**: too many accurate answers can destroy privacy. Preventing reconstruction alone does not establish privacy. A curator could answer queries using a secret random half of the population, preventing reconstruction while revealing that half exactly. Membership inference also requires far fewer statistics ([Lower bounds](../part03-dp-foundations/10-lower-bounds.md)).

## When you meet it

Every published count, cross-tabulation, dashboard cell and query answer contributes an equation. The Census Bureau attacked its 2010 publications and reconstructed a record for every person. About 46% of records matched a real person exactly on block, sex, age, race and ethnicity. Commercial data confirmed the identities of about 17% of the population, motivating the 2020 Census to adopt differential privacy. Query systems are also vulnerable: because Diffix noise did not increase with query count, Cohen and Nissim (2020) averaged it away.

## How to use it

To audit an aggregate release, count its equations. If the cells or allowed queries outnumber the people, protection depends on the noise. Substitute the noise level and query count into @eq-recon-ls-random. If predicted recovery exceeds the majority baseline substantially, run the attack. The snippet uses OpenDP's Gaussian mechanism and least squares. Because OpenDP noise cannot be seeded, the snippet checks whether the recovered share is within tolerance of the prediction rather than printing that share.

```python
from statistics import NormalDist

import numpy as np
import opendp.prelude as dp

dp.enable_features("contrib")
rng = np.random.default_rng(0)
n, m = 500, 2000  # people with a secret bit each, and random subset-sum queries
sigma = 0.25 * np.sqrt(n)  # noise sd added to each answer
secret = rng.integers(0, 2, n)
queries = rng.integers(0, 2, (m, n))
space = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.l2_distance(T=float)
gaussian = dp.m.make_gaussian(*space, scale=sigma)  # OpenDP's Gaussian mechanism on all m answers
answers = np.array(gaussian((queries @ secret).astype(float).tolist()))
estimate = np.linalg.lstsq(queries, answers, rcond=None)[0]  # the textbook attack: least squares
recovered = np.mean(np.clip(estimate, 0, 1).round() == secret)
predicted = NormalDist().cdf(np.sqrt(m - n) / (4 * sigma))
print(f"predicted {predicted:.4f}")  # predicted 0.9584
print(f"recovered within 0.04 of it: {abs(recovered - predicted) < 0.04}")  # recovered within 0.04 of it: True
```

Query count matters as much as noise. In the full edition's simulation, noise of twice $\sqrt n$ and four queries per person recovered 0.5875 of the bits. At the same noise level, @eq-recon-ls-random predicts that 106.1 queries per person recover 90%. Fixed noise therefore becomes insufficient as the query count grows; differential privacy scales noise with the number of answers.

For nonrandom queries, use Dinur and Nissim's linear-programming decoder, which constrains unknowns to $[0,1]$ with `scipy.optimize.linprog`. For published tables, express each cell as a constraint on integer unknowns and solve with `scipy.optimize.milp`. Count the feasible solutions: one solution is enough to reconstruct records. The release fails if either decoder outperforms the majority baseline.

## What goes wrong

Rounding, household swapping and small-cell suppression do not provide noise. Rounding hides nothing because an attacker can round candidate answers in the same way. Swapping leaves most constraints exact. A suppressed cell adds a constraint that its count is below the threshold. Sticky noise—returning the same answer to a repeated query—prevents only duplicate queries, which reconstruction does not require.

Test the union of all releases, not each table separately. Least squares alone is insufficient because adversarially chosen errors affect it more than random noise. Before publication, apply both decoders to the release and record the results in [Pre-release privacy risk assessment](10-risk-assessment.md).

## References

- Cohen & Nissim, "Linear Program Reconstruction in Practice" (Journal of Privacy and Confidentiality, 2020): the attack on Diffix.
- Garfinkel, Abowd & Martindale, "Understanding Database Reconstruction Attacks on Public Data" (2019): a census block rebuilt with a solver.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/09-reconstruction.md).
