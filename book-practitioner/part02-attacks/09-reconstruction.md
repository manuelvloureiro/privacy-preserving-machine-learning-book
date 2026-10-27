# Reconstruction attacks

## What it is

A **reconstruction attack** rebuilds the records behind published aggregates, because every count is a linear equation in the secret values of the people counted. For example, each of $n$ people has one secret bit, listed in $d$. A **subset-sum query** asks how many people in a set have bit 1. So the attacker's $m$ queries form a matrix $A$ of zeros and ones, one query per row. The **curator** who answers them publishes the true answers plus an error vector $e$,
$$a = Ad + e.$$

**Least squares** finds the real numbers with answers closest to the published ones, then rounds each to 0 or 1. Let each query include each person with probability one half. Let the noise be normal with standard deviation $\sigma$. Then least squares recovers about a fraction
$$\Phi\left(\frac{\sqrt{m - n}}{4\sigma}\right)$$ {#eq-recon-ls-random}
of the bits, where $\Phi$ is the standard normal distribution function. The baseline is the **majority guess**, which gives everyone the more common value, so it is right on one half of balanced bits. So with a few times more queries than people, noise well below $\sqrt n$ reveals the data. If the curator answers all $2^n$ subset sums, only error of order $n$ stops reconstruction (Dinur and Nissim 2003).

These results are the **Fundamental Law of Information Recovery**: too many answers that are too accurate destroy privacy. But a release that stops reconstruction is not private for that reason. A curator that answers from a secret random half of the people stops it, yet gives that half away exactly. And a test of membership needs far fewer statistics ([Lower bounds](../part03-dp-foundations/10-lower-bounds.md)).

## When you meet it

Every published count, cross-tabulation, dashboard cell or query answer is such an equation. The Census Bureau attacked its own 2010 publications and rebuilt a record for every person. About 46% of them matched a real person exactly on block, sex, age, race and ethnicity. With commercial data, they confirmed the identity of about 17% of the population, so the 2020 Census moved to differential privacy. Query systems fall too: the noise of Diffix did not grow with the number of queries, so Cohen and Nissim (2020) averaged it away.

## How to use it

To audit an aggregate release, first count the equations. If the cells or allowed queries outnumber the people, only the noise protects the people. Put your noise level and number of queries into @eq-recon-ls-random. If the prediction is far above the majority guess, run the attack. The snippet draws the noise from OpenDP's Gaussian mechanism and runs least squares. OpenDP's noise cannot be seeded, so the snippet compares the share recovered with the prediction instead of printing it.

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

The number of queries matters as much as the noise. In the full edition's simulation, with noise of twice $\sqrt n$, four queries per person recovered only 0.5875 of the bits. But @eq-recon-ls-random shows that at that noise, 106.1 queries per person recover 90%. So fixed noise falls to enough queries, and the noise must grow with the number of answers, as differential privacy makes it.

For queries that are not random, use the linear-programming decoder of Dinur and Nissim, which fits unknowns between 0 and 1 with `scipy.optimize.linprog`. For published tables, write each cell as a constraint on whole-number unknowns. Solve the constraints with `scipy.optimize.milp`. Then count the solutions, because a single solution rebuilds the records. The release fails if either decoder beats the majority guess.

## What goes wrong

Rounding, household swapping and small-cell suppression are not noise. Rounding hides nothing, because the attacker can round the answers of each candidate dataset too. Swapping leaves most constraints exact, and a suppressed cell even adds a constraint: the count is below the threshold. And sticky noise, the same answer to a repeated query, stops only repeats, which the attack never needs.

Test the union of all releases, not one table at a time. Do not trust least squares alone, because errors chosen against it hurt it more than random noise. Before you publish, reconstruct your own release with the two decoders. Record the result in [Pre-release privacy risk assessment](10-risk-assessment.md).

## References

- Cohen & Nissim, "Linear Program Reconstruction in Practice" (Journal of Privacy and Confidentiality, 2020): the attack on Diffix.
- Garfinkel, Abowd & Martindale, "Understanding Database Reconstruction Attacks on Public Data" (2019): a census block rebuilt with a solver.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/09-reconstruction.md).
