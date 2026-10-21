# User-level differential privacy

## Why this matters

Every differential privacy (DP) guarantee in this part so far protects one record. On the running example, stochastic gradient descent (SGD) with clipping and noise, the [DP-SGD](02-dp-sgd.md) of this part, reaches a privacy loss $\varepsilon$ of 6.71 per record. But people do not contribute one record each. A messaging user contributes hundreds of messages and a patient dozens of visits, and together a person's records can reveal far more than any one of them.

The fix rests on one idea: **the neighbouring relation is the privacy unit**. Whatever differs between two neighbouring datasets is what the guarantee protects. So to protect people, make neighbours differ in all of one person's records, bound how far one person can move the computation, and account with that bound. This chapter shows what a record-level $\varepsilon$ means for a person, and three ways to train with a guarantee for the person instead. To make it concrete, the running example's 10,000 records are spread very unevenly over 1,000 users, so that the largest holds 348.

## The privacy unit {#sec-ulvl-unit}

Take a mechanism $M$, and two datasets $D$ and $D'$ that differ by adding or removing all the records of one user. The mechanism is **user-level** $(\varepsilon, \delta)$-DP if, for every such pair and every set $S$ of outputs,
$$\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta.$$
**Example-level** DP is the same inequality for datasets that differ in one record. User-level DP is therefore not a new notion, and every property and accountant of [Part 4](../part04-privacy-accounting/00-introduction.md) holds for it unchanged. Only the sensitivity changes. The right unit, whether a record, a user, a household or a device, answers the question "whose participation do I want to hide?" Report it with every $\varepsilon$ ([Choosing an accountant](../part04-privacy-accounting/09-choosing-an-accountant.md)).

The textbook answer for a person with $k$ records is **group privacy** ([Post-processing, group privacy and composition](../part03-dp-foundations/02-properties.md)). An example-level $(\varepsilon, \delta)$-DP mechanism is user-level $(k\varepsilon, \delta_k)$-DP for users with at most $k$ records, where the person's slack is
$$\delta_k = \delta\,\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1}.$$ {#eq-ulvl-group}
Ten records protected at $\varepsilon = 1$ each give their owner $\varepsilon = 10$, and the slack is worse, because it multiplies the per-record $\delta$ by up to $e^{(k-1)\varepsilon}$. On the running example, even with the per-record $\delta$ tuned so that the user's slack is $10^{-5}$, a user with two records gets $\varepsilon = 18.61$, and a user with ten gets no guarantee at all.

## What an example-level model gives a user {#sec-ulvl-group-sgd}

Group privacy treats DP-SGD as a black box, but you know exactly how a user's records enter each step. Each step includes every record independently with the sampling rate $q$. For a user with $k$ records, what matters is the chance that some record of theirs is in the batch. While $kq$ is small, **a user with $k$ records looks like one record sampled at rate $kq$**. So you can screen a user's exposure by rerunning the record-level accountant at the rate $kq$. The user's $\varepsilon$ then grows roughly linearly in their number of records. An exact accountant for a user with $k$ records also exists (Charles et al. 2024), and it is a little worse than the screen.

| Records of the user $k$ | 1 | 2 | 5 | 10 | 50 |
| --- | --- | --- | --- | --- | --- |
| Exact user accountant | 6.72 | 15.54 | 52.36 | 146.20 | 2161.43 |
| Group privacy, @eq-ulvl-group | 6.71 | 18.61 | 118.12 | none | none |
| One record at rate $kq$ | 6.71 | 15.32 | 50.11 | 134.50 | 1673.08 |

: The user-level $\varepsilon$ at $\delta = 10^{-5}$ of the running example's example-level DP-SGD ($q = 0.01$, $\sigma = 1$, 10,000 steps) for a user with $k$ records. "None" means no $(\varepsilon, 10^{-5})$ guarantee. {#tbl-ulvl-k}

At ten records there is no protection to speak of. Still, the direct accountant beats group privacy for every user with more than one record, so never use group privacy for DP-SGD.

The rate-$kq$ screen is within 8% of the exact accountant up to ten records. It is slightly optimistic, so report the exact value, and do not discount it as pessimistic, because a person's similar records have partly aligned gradients.

## User-level DP-SGD {#sec-ulvl-dpsgd}

The better route makes the user the unit that DP-SGD samples, clips and accounts for (McMahan, Ramage, Talwar & Zhang 2018). Starting from weights chosen without looking at the data, each step does five things.

1. **Sample users.** Put each user in the batch independently with probability $q$, now the user sampling rate.
2. **Compute one vector per user.** From each sampled user's records alone, compute their average gradient, or the change in weights after a few local SGD steps.
3. **Clip per user.** Rescale that vector to length at most the clipping norm $C$.
4. **Add noise.** Sum the clipped vectors and add Gaussian noise with standard deviation $\sigma C$ per coordinate, where $\sigma$ is the noise multiplier.
5. **Step.** Divide by the expected number of sampled users and take a gradient step.

Because one user moves the sum by at most $C$, however many records they have, the guarantee is user-level DP with exactly the accountant of example-level DP-SGD. A common variant clips each record instead and keeps at most $k$ records per user. It has the guarantee of DP-SGD with noise multiplier $\sigma/k$, so it needs $k$ times more noise.

Changing the unit has two consequences. First, **you pay for the number of units**. On the running example, with 1,000 steps and an expected batch of 100 units, sampling records needs $\sigma = 0.778$ for $\varepsilon = 4$. Sampling users must use a rate ten times higher, and then it needs $\sigma = 3.78$. Second, **records inside a user are free**. They can be averaged, used for several local steps or subsampled without changing $\varepsilon$. More records make each user's vector a better estimate, so the signal improves while the noise stays fixed (Levy et al. 2021).

The data plumbing comes first. Carry a user identifier through the pipeline, and shard and sample by it. Per-example libraries such as Opacus clip each row of a batch, so with them you write a data loader that yields users and compute each sampled user's clipped vector yourself. Federated frameworks are built around this pattern, for example TensorFlow Federated's `tff.learning.dp_aggregator`, with a device as the user. Whatever the implementation, three things break the proof:

1. A user's vector depends on other users' data, for example through normalization by population statistics.
2. One person is spread over several accounts, so the unit is the account and the person is again a group.
3. Users are iterated in a fixed shuffled order instead of Poisson-sampled, which is a different mechanism ([Matrix factorization](04-matrix-mechanisms.md)).

## Contribution bounding {#sec-ulvl-bounding}

If your pipeline is built around records, you can keep it and still protect users. **Contribution bounding** keeps at most $k$ records of each user, the cap, and runs example-level DP-SGD on what remains. The result is user-level DP with the $\varepsilon$ that the user accountant gives for $k$ records, and users with fewer records are covered too. The condition is that each user's kept records are chosen without looking at other users' data, for example at random. Otherwise adding one user could change which records others keep. Analytics libraries apply the same bound to queries, for example PipelineDP's `max_partitions_contributed` and `max_contributions_per_partition` parameters, and Tumult Analytics' `MaxRowsPerID` truncation.

Choosing the cap trades noise against bias (Amin et al. 2019). A larger cap keeps more data but needs more noise, while a smaller cap discards records and reweights the data toward light users. On the running example at a user-level $\varepsilon = 4$, raising the cap from 1 to 50 multiplies the noise by 5.8. Accuracy then falls from 0.8917 to 0.6401. The best cap depends on the data, so take it from public knowledge of the population or choose it privately ([Hyperparameter tuning](03-hyperparameter-tuning.md)). Trying caps on test accuracy spends privacy that the accountant does not see.

The two techniques also combine. You can cap each user's records only to bound the compute, and still clip one vector per user. The cap then costs data but no noise.

## Heterogeneous users {#sec-ulvl-hetero}

User-level training changes the objective, not only the noise. Example-level training weights each user by their number of records, while user-level training with averaged vectors weights users equally. For a keyboard model, weighting people equally keeps a few very active users from defining everyone's suggestions. For a model meant to fit the overall distribution of records, it under-weights heavy users, and a per-user cap discards their extra records outright ([Disparate impact](08-disparate-impact.md)).

If heavy users should count for more, sum a user's gradients instead of averaging them before clipping. Their influence then rises with their records until the sum reaches $C$. Both choices stay inside the per-user computation, so $\varepsilon$ does not change.

## Which strategy {#sec-ulvl-run}

@tbl-ulvl-run compares three routes on the running example, each with 1,000 steps and an expected batch of 100 units. Strategy (a) runs example-level DP-SGD on all records and accounts for the heaviest user.

| User-level $\varepsilon$ | (a) All records, $k = 348$ | (b) User-level DP-SGD | (c) At most 2 records per user |
| --- | --- | --- | --- |
| 1 | 0.5318 ($\sigma = 449$) | 0.8065 ($\sigma = 12.9$) | 0.7084 ($\sigma = 14.4$) |
| 2 | 0.5582 ($\sigma = 237$) | 0.9028 ($\sigma = 6.89$) | 0.8374 ($\sigma = 7.62$) |
| 4 | 0.5713 ($\sigma = 128$) | 0.9347 ($\sigma = 3.78$) | 0.8810 ($\sigma = 4.18$) |
| 8 | 0.5478 ($\sigma = 70.3$) | 0.9442 ($\sigma = 2.18$) | 0.9333 ($\sigma = 2.4$) |

: Test accuracy and noise multiplier at the same user-level budget, $\delta = 10^{-5}$, one run each. The non-private model reaches 0.9612. {#tbl-ulvl-run}

Keeping the record as the unit fails. Strategy (a) stays near chance at every budget, because the heaviest user dictates the noise for everyone. User-level DP-SGD is the best at every budget, and at $\varepsilon = 4$ it is 2.7 points below the non-private model. Its cost is compute, since each step processes all the records of about 100 users. The cap is the fallback when the pipeline must stay record-based. It trails by 5 to 10 points at the smaller budgets and comes close at $\varepsilon = 8$.

Single runs vary by about a point, so read the large gaps, not the small ones. Published comparisons on language models mostly agree. Chua et al. (2024) find that user-level DP-SGD generally beats contribution bounding for fine-tuning, while Charles et al. (2024) find that sampling examples sometimes wins.

## Code {#sec-ulvl-code}

The rate-$kq$ screen needs only a record-level accountant. In Google's `dp_accounting`, it is one subsampled Gaussian step composed over the run:

```text
import dp_accounting

step = dp_accounting.PoissonSampledDpEvent(
    sampling_probability=k * q,
    event=dp_accounting.GaussianDpEvent(noise_multiplier=sigma),
)
accountant = dp_accounting.rdp.RdpAccountant()
accountant.compose(dp_accounting.SelfComposedDpEvent(step, steps))
print(accountant.get_epsilon(target_delta=1e-5))
```

With $k = 1$ and $q$ the user sampling rate, the same code accounts for user-level DP-SGD. The script [user_level.py](../../code/part05-dp-ml/user_level.py) builds the user population, implements the exact user accountant and trains the three strategies.

## Cheat sheet

| Situation | Use | What you get |
| --- | --- | --- |
| Example-level model already trained | User accountant, or the rate-$kq$ screen | User's $\varepsilon$ grows roughly linearly in $k$, then faster |
| You can group records by user | User-level DP-SGD | Records' guarantee, with users as units; pays for the number of users |
| Pipeline must stay record-based | Contribution bounding with a public cap | User accountant for $k$ records; loses data |
| Heavy users should count more | Sum before clipping instead of averaging | Same $\varepsilon$, different objective |

: Which route to a user-level guarantee. {#tbl-ulvl-cheat}

**Rules of thumb**

- Name the privacy unit with every $\varepsilon$.
- Screen a user's exposure under an example-level model by rerunning the accountant at rate $kq$.
- Don't use group privacy for DP-SGD.
- Prefer sampling and clipping users, and fall back on a public cap.

## Further reading

- Charles, Ganesh, McKenna, McMahan, Mitchell, Pillutla & Rush, "Fine-Tuning Large Language Models with User-Level Differential Privacy" (arXiv 2407.07737, 2024): the tight user accountant, and sampling examples against sampling users.
- Chua, Ghazi, Huang, Kamath, Kumar, Liu, Manurangsi, Sinha & Zhang, "Mind the Privacy Unit! User-Level Differential Privacy for Language Model Fine-Tuning" (Conference on Language Modeling 2024): contribution bounding against user-level DP-SGD.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/07-user-level.md).
