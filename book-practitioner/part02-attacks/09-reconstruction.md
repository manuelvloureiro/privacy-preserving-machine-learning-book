# Reconstruction attacks

## What it is

A **reconstruction attack** rebuilds the records behind published aggregates, because every count is a linear equation in the secret values of the people counted. Suppose each of $n$ people has one secret bit, listed in $d$. A **subset-sum query** asks how many people in a set have bit 1. The attacker's $m$ queries are the rows of a matrix $A$ of zeros and ones, and the **curator** who answers publishes the true answers plus an error vector $e$,
$$a = Ad + e.$$

**Least squares** finds the real numbers whose answers come closest to the published ones, then clips and rounds each to 0 or 1. Let each query include every person with probability one half, and let the noise be normal with standard deviation $\sigma$. Then least squares recovers about a fraction
$$\Phi\left(\frac{\sqrt{m - n}}{4\sigma}\right)$$ {#eq-recon-ls-random}
of the bits, where $\Phi$ is the standard normal distribution function. The baseline is the **majority guess**, which gives everyone the commoner value: one half for balanced bits. With a few times more queries than people, noise well below $\sqrt n$ therefore reveals the data. Answering all $2^n$ subset sums needs error of order $n$ to stop reconstruction (Dinur and Nissim 2003).

These results are the **Fundamental Law of Information Recovery**: answering too many questions too accurately destroys privacy. But stopping reconstruction is not privacy. A curator that answers from a secret random half of the people stops it, yet gives that half away exactly. And learning only whether someone took part needs far fewer statistics (Homer et al. 2008, [Lower bounds](../part03-dp-foundations/10-lower-bounds.md)).

## When you meet it

Every published count, cross-tabulation, dashboard cell or query answer is such an equation. The Census Bureau attacked its own 2010 publications and rebuilt a record for every person. About 46% of them matched a real person exactly on block, sex, age, race and ethnicity. Linked to commercial data, they confirmed the identity of about 17% of the population, so the 2020 Census moved to differential privacy. Query systems fall too: the noise of the commercial system Diffix did not grow with the number of queries, so Cohen and Nissim (2020) averaged it away.

## How to use it

To audit an aggregate release, first count the equations. If the cells or allowed queries outnumber the people, only the noise protects them. Put your noise level and number of queries into @eq-recon-ls-random, and if the prediction is far above the majority guess, run the attack. The snippet draws the noise from OpenDP's Gaussian mechanism, as a release would, and runs least squares. OpenDP's noise cannot be seeded, so it checks the share recovered against the prediction instead of printing it.

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

The number of queries matters as much as the noise. In the full edition's simulation, with noise of twice $\sqrt n$, four queries per person recovered only 0.5875 of the bits. Yet solving @eq-recon-ls-random at that noise shows that 106.1 queries per person recover 90%. So fixed noise falls to enough queries, and the noise must grow with the number of answers, as differential privacy makes it.

For queries that are not random, the linear-programming decoder of Dinur and Nissim fits unknowns between 0 and 1 with `scipy.optimize.linprog`. For published tables, write each cell as a constraint on whole-number unknowns, solve with `scipy.optimize.milp`, and count the solutions, since a single one rebuilds the records. The release fails when either decoder beats the majority guess.

## What goes wrong

Rounding, household swapping and small-cell suppression are not noise. Rounding hides nothing, because the attacker can round the answers of each candidate dataset too. Swapping leaves most constraints exact, and a suppressed cell adds one, that the count is below the threshold. Sticky noise, which repeats the same answer to a repeated query, only stops repeats, which the attack never needs.

Test the union of all releases, not one table at a time. And do not trust least squares alone, since errors chosen against it hurt it more than random noise. Reconstruct your own release with both decoders before publishing it, and record the result in [Pre-release privacy risk assessment](10-risk-assessment.md).

## References

- Cohen & Nissim, "Linear Program Reconstruction in Practice" (Journal of Privacy and Confidentiality, 2020): the attack on Diffix.
- Garfinkel, Abowd & Martindale, "Understanding Database Reconstruction Attacks on Public Data" (2019): a census block rebuilt with a solver.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/09-reconstruction.md).
