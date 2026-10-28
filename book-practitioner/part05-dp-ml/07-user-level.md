# User-level differential privacy

## Why this matters

Every differential privacy (DP) guarantee in this part so far protects one record. Stochastic gradient descent (SGD) with clipping and noise, this part's [DP-SGD](02-dp-sgd.md), gives each person of its Adult run a privacy loss $\varepsilon$ of 3.98. But a messaging user contributes hundreds of messages, and together a person's records can reveal far more than any one of them.

A DP guarantee protects whatever differs between two neighbouring datasets, so **the neighbouring relation sets the privacy unit**. Adult has no household identifier, so to show what a larger unit costs, this chapter makes one up. It gives the 30,000 training rows to 3,000 households, very unevenly, each household a cluster of similar rows. The median household holds 4 rows and the largest 359. A household is the user of this chapter, and a row its record.

## The privacy unit and δ {#sec-ulvl-unit}

Take a mechanism $M$, and call two datasets neighbours, $D$ and $D'$, when they differ by adding or removing all the records of one user. The mechanism is **user-level** $(\varepsilon, \delta)$-DP, with a small slack $\delta$, if every such pair and every set $S$ of outputs satisfy
$$\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta.$$
**Example-level** DP is the same inequality for datasets that differ in one record. So every property and accountant of [Part 4](../part04-privacy-accounting/00-introduction.md) holds for user-level DP unchanged. What changes is the sensitivity, because one user can move the computation with all of their records at once.

The right unit, such as a record, a user, a household or a device, answers the question "whose participation do I want to hide?". Report it with every $\varepsilon$ ([Release and governance](12-release-governance.md#sec-relgov-statement)). The unit also sets the slack, because the book's rule for $\delta$ ([How small δ must be](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta)) now counts users. For 3,000 households it gives $10^{-4}$ (@sec-ulvl-code), so this chapter's $10^{-5}$ is stricter than it needs to be.

The textbook answer for a person with $k$ records is **group privacy** ([Post-processing, group privacy and composition](../part03-dp-foundations/02-properties.md)). An example-level $(\varepsilon, \delta)$-DP mechanism is also user-level DP for users with at most $k$ records, at level $k\varepsilon$ and with slack
$$\delta_k = \delta\,\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1}.$$ {#eq-ulvl-group}
The fraction grows like $e^{(k-1)\varepsilon}$, so never read a record-level $\delta$ as a user's. To hold a user's slack at $10^{-5}$, the per-record $\delta$ must be much smaller, and so the per-record $\varepsilon$ larger. This chapter accounts with the full edition's Rényi accountant, which puts the Adult run at 4.50 per row. A household with two rows therefore gets $\varepsilon = 12.43$, not twice 4.50, and for ten rows no per-row $\delta$ is small enough.

## What an example-level model gives a user {#sec-ulvl-group-sgd}

Group privacy treats DP-SGD as a black box, but you know how a user's records enter each step. Each step includes every record independently with the sampling rate $q$. So it sees some record of a user with $k$ records with probability about $kq$. While $kq$ is small, two of their records rarely share a batch. So **a user with $k$ records looks like one record sampled at rate $kq$**. To screen a user's exposure, rerun the record-level accountant at rate $kq$ (@sec-ulvl-code). An exact accountant for a user with $k$ records also exists (Charles et al. 2024).

| Rows of the household $k$ | 1 | 2 | 5 | 10 | 20 |
| --- | --- | --- | --- | --- | --- |
| Exact user accountant | 4.59 | 10.02 | 29.96 | 74.82 | 202.89 |
| One record at rate $kq$ | 4.50 | 9.25 | 25.37 | 57.57 | 138.67 |

: The household-level $\varepsilon$ at $\delta = 10^{-5}$ of the Adult run's example-level DP-SGD ($q = 1/30$, $\sigma = 0.973$, 300 steps) for a household with $k$ rows, from the full edition's Rényi accountant (4.59 at one record comes from its coarser grid of orders). {#tbl-ulvl-k}

Ten rows already cost 74.82, which protects nothing. Yet the exact accountant beats group privacy for every user with more than one record, so never use group privacy for DP-SGD.

The rate-$kq$ screen is always low, by 8% at two rows and by 23% at ten. The reason is that it ignores steps that draw several of the household's rows together, and with $q = 1/30$ such steps are common. So report the exact value when you have an implementation. The libraries this book uses offer only the screen, so report its value, say how low it can be, and keep $kq$ small. It is not a pessimistic bound, because a person's similar records do push the gradient partly the same way.


## User-level DP-SGD {#sec-ulvl-dpsgd}

The better route makes the user the unit that DP-SGD samples, clips and accounts for (McMahan et al. 2018). Starting from weights chosen without looking at the data, each step does five things.

1. **Sample users.** Put each user in the batch independently with probability $q$, now the user sampling rate.
2. **Compute one vector per user.** From that user's records alone, compute their average gradient, or the change in weights after a few training steps on their records.
3. **Clip per user.** Rescale that vector to length at most the clipping norm $C$.
4. **Add noise.** Sum the clipped vectors and add Gaussian noise of standard deviation $\sigma C$, for a noise multiplier $\sigma$.
5. **Step.** Divide by the expected number of sampled users and take a gradient step.

One user moves the sum by at most $C$, however many records they have. So the guarantee is user-level DP with exactly the accountant of example-level DP-SGD, run at the user sampling rate.

That has two consequences. First, **you pay for having fewer units**. With 300 steps and about 100 units per batch, sampling rows needs $\sigma = 0.581$ for $\varepsilon = 4$. The same batch drawn from ten times fewer households needs a ten times higher rate, and so $\sigma = 1.03$. Second, **records inside a user are free**, since you can average them, train on them for several steps or subsample them without changing $\varepsilon$.

Carry a user identifier through the pipeline, and split training and test data by user. Opacus clips each row of a batch, so write a data loader that yields users and clip each user's vector yourself. Three things break the guarantee:

1. A user's vector depends on other users' data, for example through normalization statistics computed over all users.
2. One person is spread over several accounts, so the unit is the account and the person is again a group.
3. Users are iterated in a fixed shuffled order instead of Poisson-sampled, which is a different mechanism that this accountant does not cover ([Matrix factorization](04-matrix-mechanisms.md)).

## Contribution bounding {#sec-ulvl-bounding}

A pipeline built around records can still protect users. **Contribution bounding** keeps at most $k$ records of each user, the cap, and runs example-level DP-SGD on what remains. The result is user-level DP with the $\varepsilon$ the user accountant gives for $k$ records. The condition is that each user's kept records are chosen without looking at other users' data, for example at random. Otherwise adding one user could change which records other users keep, and neighbours would differ by more than one user's records. To calibrate the noise for a cap, solve for $\sigma$ with `dp_accounting.calibrate_dp_mechanism` at rate $kq$ ([Train a model with DP-SGD](13-train-dp-sgd.md#sec-trainwf-tune)).

```python
import polars as pl

rows = pl.DataFrame({"user": ["a", "a", "a", "a", "b", "c", "c"], "x": [1, 2, 3, 4, 5, 6, 7]})
k = 2  # the cap, fixed from public knowledge of the population
capped = rows.sample(fraction=1.0, shuffle=True, seed=0).group_by("user").head(k)
print(capped.height)  # 5
print(capped.group_by("user").len()["len"].max())  # 2
```

Choosing the cap trades noise against bias (Amin et al. 2019). A larger cap keeps more data but needs more noise, while a smaller cap discards records and reweights the data toward light users. On Adult's households at a household-level $\varepsilon = 4$, a cap of fifty rows instead of one multiplies the noise by 4.2, and accuracy falls from 0.8362 to 0.7943.

So take the cap from public knowledge of the population, or choose it privately and count the cost, for example with a private quantile of records per user ([Beyond global sensitivity](../part03-dp-foundations/07-sensitivity.md)). Trying caps on test accuracy spends privacy that the accountant does not see.

## Heterogeneous users {#sec-ulvl-hetero}

User-level training also changes the objective. Averaging per user weights users equally, while example-level training weights them by their number of records ([Disparate impact](08-disparate-impact.md)). If heavy users should count for more, sum a user's gradients before clipping. Their influence then rises with their records until clipping caps the sum at $C$, and $\varepsilon$ does not change.

## Which strategy {#sec-ulvl-run}

@tbl-ulvl-run compares three routes on Adult's households, each with 300 steps, about 100 units per batch and the learning rate 0.2. That rate is the best of 0.1, 0.2, 0.5 and 1 for route (b) at $\varepsilon = 4$, and also the workflow's non-private rate. Strategy (a) runs example-level DP-SGD on all rows and accounts for the largest household.

| Household-level $\varepsilon$ | (a) All rows, $k = 359$ | (b) Household-level DP-SGD | (c) At most 2 rows per household |
| --- | --- | --- | --- |
| 1 | 0.5730 | 0.8212 | 0.7929 |
| 2 | 0.6179 | 0.8374 | 0.8207 |
| 4 | 0.6431 | 0.8432 | 0.8309 |
| 8 | 0.6838 | 0.8445 | 0.8348 |

: Test accuracy at the same household-level budget, $\delta = 10^{-5}$, one run each. The non-private model reaches 0.847, and always predicting the larger class 0.759. {#tbl-ulvl-run}

Keeping the row as the unit fails. Strategy (a) stays below always predicting the larger class, because the largest household forces a noise multiplier between 13.4 and 84.6 on everyone. Household-level DP-SGD is the best at every budget, and its cost is compute, since each step processes all the rows of about 100 households. The cap is the fallback for a pipeline that must stay row-based. How far it trails depends on how rows are spread over households, and here it keeps only 5,394 of the 30,000.

## Code {#sec-ulvl-code}

The rate-$kq$ screen needs only a record-level accountant, such as Google's `dp_accounting`. This snippet screens households with one and two rows, and computes the book's $\delta$ for 3,000 households.

```python
import math

import dp_accounting


def screen(k: int, q: float = 1 / 30, sigma: float = 0.973, steps: int = 300) -> float:
    """The epsilon of a user with k records, screened as one record sampled at rate k * q."""
    step = dp_accounting.PoissonSampledDpEvent(k * q, dp_accounting.GaussianDpEvent(sigma))
    accountant = dp_accounting.rdp.RdpAccountant()
    accountant.compose(dp_accounting.SelfComposedDpEvent(step, steps))
    return accountant.get_epsilon(1e-5)


print(f"{screen(1):.2f}")  # 4.50
print(f"{screen(2):.2f}")  # 9.25
print(10 ** math.floor(math.log10(3_000**-1.1)))  # 0.0001
```

Its 9.25 matches the table's screen, but two Rényi accountants with different orders can differ in the last digits, so quote the value you ran and name its accountant. With $k = 1$ and $q$ the user sampling rate, the same code accounts for user-level DP-SGD.

## Cheat sheet

| Situation | Use | What you get |
| --- | --- | --- |
| Example-level model already trained | User accountant, or the rate-$kq$ screen | A user's $\varepsilon$ grows about linearly in $k$, then faster for large $kq$ |
| Records can be grouped by user | User-level DP-SGD | DP-SGD's accountant over users |
| Pipeline must stay record-based | Contribution bounding with a public cap | User accountant for $k$ records; loses data |
| Heavy users should count more | Sum before clipping instead of averaging | Same $\varepsilon$, different objective |

: Which route to a user-level guarantee. {#tbl-ulvl-cheat}

**Rules of thumb**

- Name the privacy unit with every $\varepsilon$, and set $\delta$ from the number of users.
- Never use group privacy for DP-SGD; screen a user at rate $kq$ instead.

## References

- Charles et al., "Fine-Tuning Large Language Models with User-Level Differential Privacy" (arXiv 2407.07737, 2024): the tight user accountant, and sampling records against sampling users.
- Chua et al., "Mind the Privacy Unit! User-Level Differential Privacy for Language Model Fine-Tuning" (Conference on Language Modeling 2024): contribution bounding against user-level DP-SGD.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/07-user-level.md).
