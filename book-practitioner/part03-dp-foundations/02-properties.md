# Post-processing, group privacy and composition

## Why this matters

A private mechanism is never the whole system. Its output gets rounded, plotted or fed into a model, people may own many records each, and an analyst rarely asks only one question. You need to know what guarantee holds for what you actually publish, without a new proof for every pipeline.

Three properties of differential privacy (DP) give that answer. The first, post-processing, says that anything computed from a private output without reading the data again is free. The second, group privacy, says what a guarantee about one record promises a person who owns several. And composition says that running several mechanisms on the same data costs at most the sum of their budgets. Together they turn privacy into bookkeeping, because you prove each building block once and then keep a ledger of what you spend.

## Post-processing {#sec-dpprop-post}

Take a mechanism $M$ with privacy level $\varepsilon$ and [slack](01-definition.md#sec-dpdef-divergences) $\delta$, so that it is $(\varepsilon, \delta)$-DP. Apply to its output any function $f$, which may use randomness of its own but never reads the data. Then the processed release is just as private,
$$f(M(D)) \text{ is } (\varepsilon, \delta)\text{-DP}.$$ {#eq-dpprop-post}
Processing can blur the evidence that an output carries about the data, and so lower the privacy loss, but it can never raise it.

This is what makes DP practical. Rounding a noisy count, clamping it at zero, or adjusting noisy subtotals so that they add up to their noisy total costs no extra budget. So a private system adds noise at one narrow point, such as noisy counts or noisy gradients, and treats everything after it as public. The same rule covers attackers, because a membership-inference attack is a function of the output. So [the definition's](01-definition.md#sec-dpdef-testing) cap on how often any attack can succeed holds even for attacks invented after the release.

The condition is that $f$ never reads the data. "Take the noisy top category, then report its exact count" breaks it, because the second step is a new and non-private mechanism.

## Group privacy {#sec-dpprop-group}

The definition protects one record, but a customer of an online shop has dozens of invoice lines. If the guarantee is per record and a person owns $k$ records, you need a guarantee for datasets that differ in $k$ records. You can walk from one dataset to the other by changing one record at a time, and each of the $k$ steps costs $\varepsilon$. So for pure DP, where $\delta = 0$, $\varepsilon$-DP per record becomes $k\varepsilon$-DP for the group. With a slack, the slack grows much faster than $k$. For a group of $k$ records, an $(\varepsilon, \delta)$-DP mechanism is
$$\Big(k\varepsilon,\ \delta_k\Big)\text{-DP}, \qquad \delta_k = \delta\,\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1} \le k\,e^{(k-1)\varepsilon}\delta.$$ {#eq-dpprop-group}

The fraction is the sum $\delta\,(1 + e^\varepsilon + \dots + e^{(k-1)\varepsilon})$, and the simpler form on the right is looser. Take $\varepsilon = 0.5$ and $\delta = 10^{-5}$ per record. Then a person with 10 records gets a slack of $2.272\times10^{-3}$. The simple form would claim $9.002\times10^{-3}$, about four times too much. The growth is far worse at $\varepsilon = 1$ per record, where the fraction multiplies $\delta$ by 12818.3 for a group of ten. That hurts, because $\delta$ was chosen far below one over the number of people in the first place ([How small δ must be](01-definition.md#sec-dpdef-small-delta)).

Don't compute group guarantees by hand, because OpenDP computes them from the mechanism itself. A measurement's `map(d_in=k)` is its guarantee when neighbouring datasets differ in $k$ records, here for a count with Laplace noise and one with Gaussian noise.

```{.python excerpt="adult-group-privacy"}
count = dp.t.make_count(dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance())
laplace = count >> dp.m.then_laplace(scale=2.0)  # epsilon = 0.5 per record
print(f"Laplace count: epsilon {laplace.map(d_in=1):.1f} per record, {laplace.map(d_in=10):.1f} for 10")

def gaussian(scale: float) -> dp.Measurement:  # a Gaussian count, as a privacy curve
    return dp.c.make_zCDP_to_approxDP(count >> dp.m.then_gaussian(scale=scale))

def at_1e5(scale: float) -> dp.Measurement:
    return dp.c.make_fix_delta(gaussian(scale), delta=1e-5)

scale = dp.binary_search_param(at_1e5, d_in=1, d_out=(0.5, 1e-5))  # (0.5, 1e-5) per record
curve = gaussian(scale).map(d_in=10)  # the privacy curve for a group of 10 records
print(f"Gaussian count, group of 10: epsilon {curve.epsilon(1e-5):.2f} at delta 1e-5")
# delta_k is the slack that the group-privacy formula gives the group
print(f"Gaussian count, group of 10: epsilon {curve.epsilon(delta_k):.2f} at delta {delta_k:.3e}")
```

```text
Laplace count: epsilon 0.5 per record, 5.0 for 10
Gaussian count, group of 10: epsilon 6.43 at delta 1e-5
Gaussian count, group of 10: epsilon 4.57 at delta 2.272e-03
```

For the Laplace count, `map(d_in=10)` gives $k\varepsilon = 5.0$, as the formula does. The Gaussian count, calibrated to $(0.5, 10^{-5})$ per record, does better than the formula, because OpenDP reads the group's guarantee off the mechanism's whole privacy curve. At the formula's slack its group gets $\varepsilon = 4.57$, which is below the formula's 5.0, and keeping the original slack instead costs more. Read a group guarantee off the library's map, then, and fall back on the formula only for a mechanism the library cannot map.

So if you care about people, don't patch a per-record guarantee with group privacy. Make the person the unit of privacy, so that neighbouring datasets differ in all of one person's records. Then bound how many records each person may contribute before any noise is added, as [Release statistics with a library](12-release-statistics.md) does on real data. Group privacy is the fallback for when you cannot, and for records of different people that reveal each other, such as the members of a household who share an address and an income.

## Sequential composition {#sec-dpprop-comp}

An analyst asks several queries of the same data, each answered with fresh noise, and an adversary sees all the answers together. Say there are $k$ mechanisms, and write $(\varepsilon_i, \delta_i)$ for the budget of the $i$-th. Releasing all their outputs is then
$$\Big(\sum_{i=1}^k\varepsilon_i,\ \sum_{i=1}^k\delta_i\Big)\text{-DP}.$$ {#eq-dpprop-basic}
This is **basic composition**. It needs every mechanism to draw fresh noise and to use the same [neighbouring relation](01-definition.md#sec-dpdef-neighbours), add/remove or replace.

The rule also covers **adaptive** composition, where each query is chosen after reading the earlier answers, as long as each budget is fixed in advance. Budgets chosen from the answers are another matter, as in "keep going until the estimate looks good". They need a privacy filter, a rule that decides when to stop as the budget is spent ([Adaptive composition](../part04-privacy-accounting/08-adaptive-composition.md#sec-adapt-dp-filters)).

Basic composition is loose for many releases. Take ten counts with Gaussian noise, each calibrated to $(0.5, 10^{-5})$. Basic composition charges them $(5, 10^{-4})$, but at the same total slack their exact $\varepsilon$ is only 1.504. So once releases number more than a handful, use an accountant, which computes the total from the mechanisms themselves ([Advanced and optimal composition](04-composition.md), [Part 4](../part04-privacy-accounting/00-introduction.md)).

## Parallel composition {#sec-dpprop-parallel}

Per-region statistics use disjoint subsets of the data. When each person has one record, each region can spend the full budget, because that record falls in one region only. To use this, fix in advance a public rule that sends every possible record to exactly one part, and run a mechanism $M_i$ with independent noise on the $i$-th part. If each $M_i$ is $(\varepsilon_i, \delta_i)$-DP under add/remove, the whole release is
$$\big(\max_i\varepsilon_i,\ \max_i\delta_i\big)\text{-DP under add/remove}.$$ {#eq-dpprop-parallel}

This is why a histogram costs one $\varepsilon$, not one per bin. With Laplace noise of scale $1/\varepsilon$ on every bin, the histogram is $\varepsilon$-DP however many bins it has. Under replace, a record can move from one bin to another, so two bins change and the cost doubles to $2\varepsilon$. A person with records in several bins breaks the rule too, and then you must either charge each bin as a separate query or first bound the number of bins a person may touch.

The rule fixed in advance is the condition that is easiest to break. A split into "the top 10% of earners and the rest" moves its boundary when one high earner is added, so records cross it and both parts change. The same goes for partitioning on the list of keys that happen to appear in the data, because a key that appears only through one person's record reveals that this person is present. So use fixed bins or a public key list. When the keys are not public, release only the keys whose noisy count clears a threshold, a step called **partition selection** ([Release statistics with a library](12-release-statistics.md)).

## Keeping a budget {#sec-dpprop-budget}

Fix a total budget for a dataset before the first release, and charge every release against it. Libraries keep this ledger for you. In the Python library diffprivlib, a `BudgetAccountant` holds a total, and each diffprivlib function you hand it charges its own $\varepsilon$ and $\delta$. A call that would exceed the total raises a `BudgetError` instead of answering. The script [practitioner_adult_statistics.py](../../code/part03-dp-foundations/practitioner_adult_statistics.py) runs it on the Adult census data, 48,842 people with one row each, for which the default slack is $10^{-6}$. It spends half the budget on the mean age and half on the number of people over 50, and then asks for one more count.

```{.python excerpt="adult-ledger"}
age = adult["age"].to_numpy()
acc = BudgetAccountant(epsilon=1.0, delta=1e-6)  # the total, fixed in advance
mean_age = tools.mean(age, epsilon=0.5, bounds=(17, 90), accountant=acc, random_state=seed)
over_50 = tools.count_nonzero(age > 50, epsilon=0.5, accountant=acc, random_state=seed)
print(f"private mean age: {mean_age:.2f}  private count over 50: {over_50:,}")
print(f"spent: {acc.total()}")
try:
    tools.count_nonzero(age > 60, epsilon=0.1, accountant=acc, random_state=seed)
except BudgetError as err:
    print(f"refused: {err}")
```

Run it with `uv run python code/part03-dp-foundations/practitioner_adult_statistics.py`. With `random_state=0` it prints the following.

```text
private mean age: 38.64  private count over 50: 9,813
spent: (epsilon=1.0, delta=0.0)
refused: Privacy spend of (0.1,0) not permissible; will exceed remaining privacy budget. Use BudgetAccountant.remaining() to check remaining budget.
```

The true values are 38.64 and 9,808. Both calls add Laplace noise, so no $\delta$ was spent and the $10^{-6}$ stays available for a Gaussian query. The accountant applies basic composition to every call, so it overcharges calls on disjoint parts of the data, and it trusts each call's own guarantee. For example, diffprivlib's mean divides by the exact number of rows, treating it as public. That number is public only when neighbours have the same size, which is the replace relation, so this ledger holds under replace neighbours.

Most real failures break a condition of the three properties, and the damage is usually total rather than a small loss. @tbl-dpprop-rules lists the common cases.

| Situation | Composes? | What to do |
| --- | --- | --- |
| A function of the output, with its own coins | Yes, for free | Process as much as you like |
| Another query on the same data, fresh noise | Yes, budgets add | Charge it to the ledger |
| A query chosen from earlier answers | Yes, budgets add | Fix its budget in advance |
| Two releases that share one noise draw | No | Draw fresh noise for every query |
| "Post-processing" that reads the data again | No | Charge it as a new mechanism |
| A DP release beside an exact statistic of the same data | No, the exact one has no guarantee | Release the statistic privately too |
| A split whose boundaries depend on the data | No, not in parallel | Fix the bins in advance |
| Guarantees under add/remove and replace in one ledger | Only after conversion | Convert to replace: $2\varepsilon$ and $(1 + e^\varepsilon)\delta$ |

: What composes and what doesn't. {#tbl-dpprop-rules}

Shared noise is the clearest case. Release the number of smokers and the number of smokers aged 40 or over, adding the same Laplace draw to both. With noise of scale 1, each release alone is 1-DP, but their difference cancels the noise and gives the exact number of smokers under 40, so the pair is not DP at all. In code, the bug hides in two mechanisms seeded with the same random seed, in a cached noise value re-added to a related query, and in noise reused across shards. A fixed seed such as the `random_state=0` above reproduces a book's numbers, but in production it makes the noise reproducible and so removable. Returning the same noisy answer to a repeated identical query is fine, because that is post-processing of one release.

## Cheat sheet

| Property | Rule | Condition |
| --- | --- | --- |
| Post-processing | $f(M(D))$ is $(\varepsilon, \delta)$-DP | $f$ never reads the data |
| Group privacy | $\big(k\varepsilon,\ \delta\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1}\big)$ for $k$ records | Prefer the person as the unit, with bounded contributions |
| Basic composition | $(\sum\varepsilon_i, \sum\delta_i)$ | Fresh noise, one neighbouring relation, budgets fixed in advance |
| Parallel composition | $(\max\varepsilon_i, \max\delta_i)$ | Add/remove, parts fixed in advance; $2\varepsilon$ under replace |

: The results of this chapter. {#tbl-dpprop-cheat}

**Rules of thumb**

- Add noise once, as early as possible, and everything computed from the noisy output after it is free.
- Fix the total budget before the first release, and let one accountant refuse what exceeds it.
- Never share a random seed between mechanisms, and never fix one in production.

## References

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014): post-processing, group privacy and basic composition, the standard reference.
- McSherry, "Privacy Integrated Queries" (SIGMOD 2009): parallel composition, and a query system built on post-processing and composition as its only rules.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/02-properties.md).
