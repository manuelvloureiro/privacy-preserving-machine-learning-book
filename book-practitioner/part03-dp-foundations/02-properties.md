# Post-processing, group privacy and composition

## Why this matters

A private mechanism is never the whole system. Its output is rounded or goes into a model, a person often owns many records, and an analyst seldom asks only one question. Three properties of differential privacy (DP) cover these cases, which are post-processing, group privacy and composition. With them you prove each building block once and keep a ledger of what you spend.

## Post-processing {#sec-dpprop-post}

Take a mechanism $M$ that is DP with privacy level $\varepsilon$ and [slack](01-definition.md#sec-dpdef-divergences) $\delta$.
 Apply to its output a function $f$ that never reads the data, although it can use random numbers of its own. Then the processed release is just as private,
$$f(M(D)) \text{ is } (\varepsilon, \delta)\text{-DP}.$$ {#eq-dpprop-post}

So it costs nothing to round a noisy count, to clamp it at zero, or to make noisy subtotals agree with their noisy total. So a private system adds noise at one narrow point, such as counts or gradients, and treats everything after it as public. The rule also covers attackers, because a membership-inference attack is a function of the output. So [the definition's](01-definition.md#sec-dpdef-testing) limit on the success of an attack holds even for attacks invented after the release.


The condition is that $f$ never reads the data. "Take the noisy top category, then report its exact count" breaks it, because the second step is a new and non-private mechanism.

## Group privacy {#sec-dpprop-group}

The definition protects one record, but a customer of an online shop has dozens of invoice lines. For a person with $k$ records, the guarantee must cover datasets that differ in $k$ records. You can go from one such dataset to the other in $k$ steps that each change one record and each cost $\varepsilon$. So under pure DP, where $\delta = 0$, $\varepsilon$-DP per record becomes $k\varepsilon$-DP for the group. But the slack grows much faster than $k$. For a group of $k$ records, an $(\varepsilon, \delta)$-DP mechanism is
$$\Big(k\varepsilon,\ \delta_k\Big)\text{-DP}, \qquad \delta_k = \delta\,\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1} \le k\,e^{(k-1)\varepsilon}\delta.$$ {#eq-dpprop-group}

The form on the right is simpler but looser. With $\varepsilon = 0.5$ and $\delta = 10^{-5}$ per record, a person with 10 records gets a slack of $2.272\times10^{-3}$. That slack is too large, because the book chooses $\delta$ far below one over the number of people ([How small δ must be](01-definition.md#sec-dpdef-small-delta)).

OpenDP calculates group guarantees from the mechanism itself. A measurement's `map(d_in=k)` gives its guarantee when neighbouring datasets differ in $k$ records, here for a Laplace count and a Gaussian count.

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

For the Laplace count, `map(d_in=10)` gives $k\varepsilon = 5.0$, the same as the formula. The Gaussian count, set to $(0.5, 10^{-5})$ per record, does better, because OpenDP gets the group's guarantee from the whole privacy curve of the mechanism. At the formula's slack of $2.272\times10^{-3}$, the group gets $\varepsilon = 4.57$, less than the formula's 5.0. So take a group guarantee from the library's map, and use the formula only for a mechanism that the library cannot map.

Better still, make the person the unit of privacy. Then neighbouring datasets differ in all of one person's records. Bound the number of records that each person can contribute before you add noise,
 as [Release statistics with a library](12-release-statistics.md) does on real data. Use group privacy when you cannot do this, and for records of different people that reveal each other, such as a household with one address and one income.

## Sequential composition {#sec-dpprop-comp}

An analyst asks several queries of the same data, each with fresh noise, and an adversary sees all the answers together. For $k$ mechanisms, write $(\varepsilon_i, \delta_i)$ for the budget of the $i$-th. Then the release of all their outputs is
$$\Big(\sum_{i=1}^k\varepsilon_i,\ \sum_{i=1}^k\delta_i\Big)\text{-DP}.$$ {#eq-dpprop-basic}
This is **basic composition**. It needs every mechanism to draw fresh noise and to use the same [neighbouring relation](01-definition.md#sec-dpdef-neighbours), add/remove or replace.

The rule also covers **adaptive** composition, where the analyst chooses each query from the earlier answers, but only if each budget is fixed in advance.
 A budget chosen from the answers, as in "continue until the estimate looks good", needs a privacy filter. That is a rule that decides when to stop ([Adaptive composition](../part04-privacy-accounting/08-adaptive-composition.md#sec-adapt-dp-filters)).

For many releases, basic composition gives too much. Ten Gaussian counts at $(0.5, 10^{-5})$ each cost $(5, 10^{-4})$ by basic composition, but at that slack their exact $\varepsilon$ is only 1.504. So for more than a few releases, use an accountant. It calculates the total from the mechanisms themselves ([Advanced and optimal composition](04-composition.md), [Part 4](../part04-privacy-accounting/00-introduction.md)).

## Parallel composition {#sec-dpprop-parallel}

Statistics per region use disjoint subsets of the data. When each person has one record, that record is in one region only. So each region can spend the full budget. To use this, fix in advance a public rule that puts every possible record in exactly one part. Then run a mechanism $M_i$ with independent noise on the $i$-th part. If each $M_i$ is $(\varepsilon_i, \delta_i)$-DP under add/remove, the whole release is
$$\big(\max_i\varepsilon_i,\ \max_i\delta_i\big)\text{-DP under add/remove}.$$ {#eq-dpprop-parallel}

So a histogram with Laplace noise of scale $1/\varepsilon$ on every bin is $\varepsilon$-DP for any number of bins. Under replace, a record can move from one bin to another, so two bins change and the cost doubles to $2\varepsilon$. A person with records in several bins also breaks the rule. Then charge each bin as a separate query, or first bound the number of bins that a person can touch.

The condition that is easiest to break is the rule fixed in advance. A split into "the top 10% of earners and the rest" moves its boundary when one high earner is added, so both parts change. A split on the keys that appear in the data also fails, because a key that only one person's record contains reveals that person. So use fixed bins or a public list of keys. When the keys are not public, release only the keys whose noisy count is above a threshold, a step called **partition selection** ([Release statistics with a library](12-release-statistics.md)).

## Keeping a budget {#sec-dpprop-budget}

Fix a total budget for a dataset before the first release, and charge every release against it.
 In the Python library diffprivlib, a `BudgetAccountant` holds the total, and each function that you give it charges its own $\varepsilon$ and $\delta$. A call that would go above the total raises a `BudgetError` and gives no answer. The script [practitioner_adult_statistics.py](../../code/part03-dp-foundations/practitioner_adult_statistics.py) uses it on the Adult census data, 48,842 people with one row each. For these people, the [default slack](01-definition.md#sec-dpdef-small-delta) is $10^{-6}$. It spends the budget on two queries and then asks for one more count.

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


The true values are 38.64 and 9,808. The two calls add Laplace noise, so they spend no $\delta$, and the $10^{-6}$ stays available for a Gaussian query. The ledger has two limits.
 First, it applies basic composition to every call, so it charges too much for calls on disjoint parts of the data. Second, it trusts the guarantee of each call, and diffprivlib's mean divides by the exact number of rows as if that number were public. The number is public only when neighbours have the same size, so this ledger holds only under replace neighbours.

Most real failures break a condition of the three properties, and then the guarantee is usually lost completely (@tbl-dpprop-rules).

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

Shared noise is the clearest case of a broken condition. Suppose that you release the number of smokers and the number of smokers aged 40 or more, with the same Laplace draw of scale 1 on each. Each release alone is 1-DP, but their difference cancels the noise and gives the exact number of smokers under 40. So the pair is not DP at all. In code, the bug hides in a shared random seed, a cached noise value or noise that several shards use. A fixed seed such as `random_state=0` above makes a book's numbers repeatable, but in production it makes the noise known and so removable. The same noisy answer to a repeated identical query is safe, because it is post-processing of one release.


## Cheat sheet

| Property | Rule | Condition |
| --- | --- | --- |
| Post-processing | $f(M(D))$ is $(\varepsilon, \delta)$-DP | $f$ never reads the data |
| Group privacy | $\big(k\varepsilon,\ \delta\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1}\big)$ for $k$ records | Prefer the person as the unit, with bounded contributions |
| Basic composition | $(\sum\varepsilon_i, \sum\delta_i)$ | Fresh noise, one neighbouring relation, budgets fixed in advance |
| Parallel composition | $(\max\varepsilon_i, \max\delta_i)$ | Add/remove, parts fixed in advance; $2\varepsilon$ under replace |

: The results of this chapter. {#tbl-dpprop-cheat}

**Rules of thumb**

- Add noise once, as early as possible. All that you calculate from the noisy output is then free.
- Fix the total budget before the first release, and let one accountant refuse what goes above it.
- Do not share a random seed between mechanisms, and do not fix one in production.

## References

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014): post-processing, group privacy and basic composition, the standard reference.
- McSherry, "Privacy Integrated Queries" (SIGMOD 2009): parallel composition, and a query system built on post-processing and composition as its only rules.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/02-properties.md).
