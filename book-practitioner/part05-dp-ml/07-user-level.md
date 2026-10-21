# User-level differential privacy

## Why this matters

Every differential privacy (DP) guarantee in this part so far protects one record. Stochastic gradient descent (SGD) with clipping and noise, this part's [DP-SGD](02-dp-sgd.md), gives each record of its running example a privacy loss $\varepsilon$ of 6.71. But a messaging user contributes hundreds of messages, and together a person's records can reveal far more than any one of them.

The fix rests on one idea. A DP guarantee protects whatever differs between two neighbouring datasets, so **the neighbouring relation sets the privacy unit**. For the examples, this chapter gives the running example's 10,000 synthetic records to 1,000 users, very unevenly. The median user holds 4 records and the largest 348.

## The privacy unit and δ {#sec-ulvl-unit}

Take a mechanism $M$, and call two datasets neighbours, $D$ and $D'$, when they differ by adding or removing all the records of one user. The mechanism is **user-level** $(\varepsilon, \delta)$-DP, with a small slack $\delta$, if every such pair and every set $S$ of outputs satisfy
$$\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta.$$
**Example-level** DP is the same inequality for datasets that differ in one record. So user-level DP is not a new notion, and every property and accountant of [Part 4](../part04-privacy-accounting/00-introduction.md) holds for it unchanged. What changes is the sensitivity, because one user can now move the computation with all of their records at once.

The right unit, whether a record, a user, a household or a device, answers the question "whose participation do I want to hide?" Report the unit with every $\varepsilon$ ([Report and review an ε claim](../part04-privacy-accounting/09-choosing-an-accountant.md)). For a language model it is usually a document or a user, not a sentence ([Fine-tune a language model privately](14-private-llm-finetuning.md)). The unit also sets the slack, because the book's default $\delta$ ([Choosing ε](09-choosing-epsilon.md)) now counts users. For 1,000 users it gives $10^{-4}$ (@sec-ulvl-code). This chapter's $10^{-5}$ is therefore stricter than it needs to be.

The textbook answer for a person with $k$ records is **group privacy** ([Post-processing, group privacy and composition](../part03-dp-foundations/02-properties.md)). It says that an example-level $(\varepsilon, \delta)$-DP mechanism is also user-level DP for users with at most $k$ records. The level becomes $k\varepsilon$, and the slack becomes
$$\delta_k = \delta\,\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1}.$$ {#eq-ulvl-group}
The slack is worse, because the fraction grows like $e^{(k-1)\varepsilon}$, which is why a record-level $\delta$ must never be read as a user's. Holding a user's slack at $10^{-5}$ needs a much smaller per-record $\delta$, and so a larger per-record $\varepsilon$. On the running example a user with two records therefore gets $\varepsilon = 18.61$, not twice 6.71. For ten records no per-record $\delta$ is small enough, and group privacy gives no guarantee at all.

## What an example-level model gives a user {#sec-ulvl-group-sgd}

Group privacy treats DP-SGD as a black box, but you know exactly how a user's records enter each step. Each step includes every record independently with the sampling rate $q$, so a step sees some record of the user's with probability about $kq$. While $kq$ is small, two of their records rarely share a batch, and **a user with $k$ records looks like one record sampled at rate $kq$**. So you can screen a user's exposure by rerunning the record-level accountant at the rate $kq$, as the snippet in @sec-ulvl-code does. An exact accountant for a user with $k$ records also exists (Charles et al. 2024).

| Records of the user $k$ | 1 | 2 | 5 | 10 | 50 |
| --- | --- | --- | --- | --- | --- |
| Exact user accountant | 6.72 | 15.54 | 52.36 | 146.20 | 2161.43 |
| One record at rate $kq$ | 6.71 | 15.32 | 50.11 | 134.50 | 1673.08 |

: The user-level $\varepsilon$ at $\delta = 10^{-5}$ of the running example's example-level DP-SGD ($q = 0.01$, $\sigma = 1$, 10,000 steps) for a user with $k$ records, from the full edition's Rényi accountant (6.72 at one record comes from its coarser grid of orders). {#tbl-ulvl-k}

At ten records the exact accountant gives 146.20, no protection to speak of. Still, it beats group privacy for every user with more than one record, so never use group privacy for DP-SGD. The rate-$kq$ screen stays within 8% of the exact accountant up to ten records but always comes out a little low. So report the exact value when you have an implementation of it. The libraries this book uses offer only the screen. Then report the screen's value, call it a screen that can be a few percent low, and keep $kq$ small. Treat it as the guarantee, not as a pessimistic bound, because a person's similar records do push the gradient partly the same way.

## User-level DP-SGD {#sec-ulvl-dpsgd}

The better route makes the user the unit that DP-SGD samples, clips and accounts for (McMahan et al. 2018), as federated learning does with a device as the user. Starting from weights chosen without looking at the data, each step does five things.

1. **Sample users.** Put each user in the batch independently with probability $q$, now the user sampling rate.
2. **Compute one vector per user.** From that user's records alone, compute their average gradient, or the change in weights after a few training steps on their records.
3. **Clip per user.** Rescale that vector to length at most the clipping norm $C$.
4. **Add noise.** Sum the clipped vectors and add Gaussian noise of standard deviation $\sigma C$, for a noise multiplier $\sigma$.
5. **Step.** Divide by the expected number of sampled users and take a gradient step.

One user moves the sum by at most $C$, however many records they have. So the guarantee is user-level DP with exactly the accountant of example-level DP-SGD, run at the user sampling rate.

Changing the unit has two consequences. First, **you pay for having fewer units**. To see how much, take a shorter schedule on the running example, 1,000 steps with about 100 units per batch. Sampling records then needs $\sigma = 0.778$ for $\varepsilon = 4$. Sampling users instead needs a rate ten times higher, so the same budget needs $\sigma = 3.78$. Second, **records inside a user are free**, since they can be averaged, used for several training steps or subsampled without changing $\varepsilon$.

Start with the data plumbing, and carry a user identifier through the pipeline to sample by. Split training and test data by user too, so that no user's records sit on both sides. Opacus clips each row of a batch, so for user-level training you write a data loader that yields users and clip each user's vector yourself. Three things break the guarantee:

1. A user's vector depends on other users' data, for example through normalization statistics computed over all users.
2. One person is spread over several accounts, so the unit is the account and the person is again a group.
3. Users are iterated in a fixed shuffled order instead of Poisson-sampled, which is a different mechanism that this accountant does not cover ([Matrix factorization](04-matrix-mechanisms.md)).

## Contribution bounding {#sec-ulvl-bounding}

A pipeline built around records can still protect users. **Contribution bounding** keeps at most $k$ records of each user, the cap, and runs example-level DP-SGD on what remains. The result is user-level DP with the $\varepsilon$ that the user accountant gives for $k$ records. The condition is that each user's kept records are chosen without looking at other users' data, for example at random. Otherwise adding one user could change which records other users keep, and neighbours would differ by more than one user's records. In Polars, shuffle the rows with a fixed seed and keep the first $k$ of each user. Releases of statistics bound customers the same way, on the Online Retail invoice data ([Release statistics with a library](../part03-dp-foundations/12-release-statistics.md#sec-relstat-bound)). To calibrate the noise for a cap, solve for $\sigma$ with `dp_accounting.calibrate_dp_mechanism` at the rate $kq$, as the tuning step of [Train a model with DP-SGD](13-train-dp-sgd.md#sec-trainwf-tune) does at rate $q$.

```python
import polars as pl

rows = pl.DataFrame({"user": ["a", "a", "a", "a", "b", "c", "c"], "x": [1, 2, 3, 4, 5, 6, 7]})
k = 2  # the cap, fixed from public knowledge of the population
capped = rows.sample(fraction=1.0, shuffle=True, seed=0).group_by("user").head(k)
print(capped.height)  # 5
print(capped.group_by("user").len()["len"].max())  # 2
```

Choosing the cap trades noise against bias (Amin et al. 2019). A larger cap keeps more data but needs more noise, while a smaller cap discards records and reweights the data toward light users. On the running example at a user-level $\varepsilon = 4$, raising the cap from one record to fifty multiplies the noise by 5.8. That is far less than fifty, because the batch keeps its size and so the sampling rate falls as the cap rises. Accuracy still falls from 0.8917 to 0.6401. So take the cap from public knowledge of the population. Or choose it privately and count the cost, for example from a private quantile of records per user ([Beyond global sensitivity](../part03-dp-foundations/07-sensitivity.md)). Trying caps on test accuracy spends privacy that the accountant does not see.

## Heterogeneous users {#sec-ulvl-hetero}

User-level training also changes the objective, because averaging per user weights users equally, while example-level training weights them by their number of records ([Disparate impact](08-disparate-impact.md)). If heavy users should count for more, sum a user's gradients instead of averaging them before clipping. Their influence then rises with their records until clipping caps the sum at $C$, and $\varepsilon$ does not change.

## Which strategy {#sec-ulvl-run}

@tbl-ulvl-run compares three routes on the running example, each with 1,000 steps and about 100 units per batch. Strategy (a) runs example-level DP-SGD on all records and accounts for the heaviest user.

| User-level $\varepsilon$ | (a) All records, $k = 348$ | (b) User-level DP-SGD | (c) At most 2 records per user |
| --- | --- | --- | --- |
| 1 | 0.5318 | 0.8065 | 0.7084 |
| 2 | 0.5582 | 0.9028 | 0.8374 |
| 4 | 0.5713 | 0.9347 | 0.8810 |
| 8 | 0.5478 | 0.9442 | 0.9333 |

: Test accuracy at the same user-level budget, $\delta = 10^{-5}$, one run each. The non-private model reaches 0.9612. {#tbl-ulvl-run}

Keeping the record as the unit fails. Strategy (a) stays near chance on this two-class task at every budget, because the heaviest user forces a noise multiplier between 70.3 and 449 on everyone. User-level DP-SGD is the best at every budget, and at $\varepsilon = 4$ it is 2.7 points below the non-private model. Its cost is compute, since each step processes all the records of about 100 users. So the cap is the fallback when the pipeline must stay record-based, and on this task it trails by 5 to 10 points at the smaller budgets. How far it trails depends on how records are spread over users, so measure it on your data.

## Code {#sec-ulvl-code}

The rate-$kq$ screen needs only a record-level accountant, such as Google's `dp_accounting`. This snippet screens users with one and two records, and computes the default $\delta$ for 1,000 users.

```python
import math

import dp_accounting


def screen(k: int, q: float = 0.01, sigma: float = 1.0, steps: int = 10_000) -> float:
    """The epsilon of a user with k records, screened as one record sampled at rate k * q."""
    step = dp_accounting.PoissonSampledDpEvent(k * q, dp_accounting.GaussianDpEvent(sigma))
    accountant = dp_accounting.rdp.RdpAccountant()
    accountant.compose(dp_accounting.SelfComposedDpEvent(step, steps))
    return accountant.get_epsilon(1e-5)


print(f"{screen(1):.2f}")  # 6.71
print(f"{screen(2):.2f}")  # 15.35
print(10 ** math.floor(math.log10(1_000**-1.1)))  # 0.0001
```

Its 15.35 is a little above the table's 15.32, because the two accountants compute the Rényi guarantee at different orders, so quote the one you ran and name it. With $k = 1$ and $q$ the user sampling rate, the same code accounts for user-level DP-SGD.

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

## Further reading

- Charles et al., "Fine-Tuning Large Language Models with User-Level Differential Privacy" (arXiv 2407.07737, 2024): the tight user accountant, and sampling examples against sampling users.
- Chua et al., "Mind the Privacy Unit! User-Level Differential Privacy for Language Model Fine-Tuning" (Conference on Language Modeling 2024): contribution bounding against user-level DP-SGD.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/07-user-level.md).
