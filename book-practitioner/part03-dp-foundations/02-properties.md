# Post-processing, group privacy and composition

## Why this matters

A private mechanism is one component of a system. Its output may be rounded or used to fit a model; one person may contribute multiple records; and an analyst may ask multiple questions. Three properties of differential privacy (DP) address these cases: post-processing, group privacy and composition. They allow separate analysis of each mechanism and tracking of the total privacy budget.

## Post-processing {#sec-dpprop-post}

Let $M$ be a mechanism that is DP with privacy level $\varepsilon$ and [slack](01-definition.md#sec-dpdef-divergences) $\delta$. Apply a function $f$ to its output. The function may use additional randomness but must not access the data. The processed release remains equally private:
$$f(M(D)) \text{ is } (\varepsilon, \delta)\text{-DP}.$$ {#eq-dpprop-post}

Rounding a noisy count, clamping it at zero or reconciling noisy subtotals with their total therefore costs no additional privacy. A system can add noise to counts or gradients, then treat subsequent computations as post-processing. The rule also applies to attacks: membership inference is a function of the output, so [the definition's](01-definition.md#sec-dpdef-testing) limit on attack success applies to future attacks.


The condition is that $f$ never accesses the data. For example, selecting the noisy top category and then reporting its exact count violates post-processing because the second step is a new, non-private mechanism.

## Group privacy {#sec-dpprop-group}

The definition protects one record, but an online-shop customer may have dozens of invoice lines. For a person with $k$ records, the guarantee must cover datasets that differ in $k$ records. Transforming one dataset into the other requires $k$ single-record changes, each costing $\varepsilon$. Under pure DP, where $\delta = 0$, per-record $\varepsilon$-DP therefore becomes group-level $k\varepsilon$-DP. Under approximate DP, the slack can grow faster than $k$. For a group of $k$ records, an $(\varepsilon, \delta)$-DP mechanism is
$$\Big(k\varepsilon,\ \delta_k\Big)\text{-DP}, \qquad \delta_k = \delta\,\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1} \le k\,e^{(k-1)\varepsilon}\delta.$$ {#eq-dpprop-group}

The right-hand form is simpler but looser. At $\varepsilon = 0.5$ and $\delta = 10^{-5}$ per record, a person with 10 records has slack $2.272\times10^{-3}$. This exceeds the book's default, which sets $\delta$ far below one divided by the number of people ([How small δ must be](01-definition.md#sec-dpdef-small-delta)).

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

For the Laplace count, `map(d_in=10)` gives $k\varepsilon = 5.0$, matching the formula. The Gaussian count, calibrated to $(0.5, 10^{-5})$ per record, gives a tighter group bound because OpenDP uses the mechanism's full privacy curve. At the formula's slack of $2.272\times10^{-3}$, the group has $\varepsilon = 4.57$, below the formula's 5.0. Use the library's group mapping when available; otherwise use the formula.

Prefer the person as the privacy unit, so neighbouring datasets differ in all of one person's records. Bound each person's contribution before adding noise, as in [Release statistics with a library](12-release-statistics.md). Use group privacy when a person-level guarantee is unavailable or when records from different people reveal information about each other, such as shared household address and income.

## Sequential composition {#sec-dpprop-comp}

Suppose an analyst asks several queries of the same data, each with fresh noise, and an adversary sees all answers. For $k$ mechanisms with budgets $(\varepsilon_i, \delta_i)$, the release of all outputs is
$$\Big(\sum_{i=1}^k\varepsilon_i,\ \sum_{i=1}^k\delta_i\Big)\text{-DP}.$$ {#eq-dpprop-basic}
This is **basic composition**. It needs every mechanism to draw fresh noise and to use the same [neighbouring relation](01-definition.md#sec-dpdef-neighbours), add/remove or replace.

The rule also covers **adaptive** composition, where each query depends on earlier answers, provided every query's budget is fixed in advance. If the budget depends on the answers—for example, continuing until an estimate meets a target—a privacy filter must determine when to stop ([Adaptive composition](../part04-privacy-accounting/08-adaptive-composition.md#sec-adapt-dp-filters)).

For many releases, basic composition can be loose. Ten Gaussian counts at $(0.5, 10^{-5})$ each cost $(5, 10^{-4})$ under basic composition, but their exact $\varepsilon$ at that slack is 1.504. For multiple releases, use an accountant that composes the mechanisms themselves ([Advanced and optimal composition](04-composition.md), [Part 4](../part04-privacy-accounting/00-introduction.md)).

## Parallel composition {#sec-dpprop-parallel}

Statistics by region use disjoint data subsets. If each person has one record, each record belongs to one region, so each region can use the full budget. The partition must follow a public rule fixed in advance that assigns every possible record to exactly one part. Run a mechanism $M_i$ with independent noise on each part. If each $M_i$ is $(\varepsilon_i, \delta_i)$-DP under add/remove, the full release is
$$\big(\max_i\varepsilon_i,\ \max_i\delta_i\big)\text{-DP under add/remove}.$$ {#eq-dpprop-parallel}

A histogram with Laplace noise of scale $1/\varepsilon$ on each bin is therefore $\varepsilon$-DP for any number of bins. Under replace, a record can move between two bins, doubling the cost to $2\varepsilon$. The rule also fails when a person contributes records to multiple bins. Charge each bin as a separate query or bound the number of bins each person can affect.

The fixed-partition condition is easy to violate. A split into "the top 10% of earners and the rest" changes its boundary when one high earner is added, so both parts change. A partition based on observed keys also fails: a key present in only one person's record identifies that person. Use fixed bins or a public key list. If the keys are not public, release only keys whose noisy counts exceed a threshold, using **partition selection** ([Release statistics with a library](12-release-statistics.md)).

## Keeping a budget {#sec-dpprop-budget}

Set a total budget before the first release and charge every release against it. In diffprivlib, a `BudgetAccountant` tracks the total, and each function charges its $\varepsilon$ and $\delta$. A call that would exceed the total raises `BudgetError` and returns no answer. The script [practitioner_adult_statistics.py](../../code/part03-dp-foundations/practitioner_adult_statistics.py) uses it on Adult, a dataset with 48,842 people and one row per person. The [default slack](01-definition.md#sec-dpdef-small-delta) is $10^{-6}$. The script spends the budget on two queries, then attempts a third count.

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


The true values are 38.64 and 9,808. Laplace noise uses no $\delta$, leaving $10^{-6}$ available for a Gaussian query. The ledger has two limitations. First, it applies basic composition to each call, overcharging calls on disjoint data subsets. Second, it trusts each call's guarantee. diffprivlib's mean divides by the exact row count, which is public only when neighbouring datasets have equal size. The ledger therefore holds only under replace neighbours.

Most composition failures violate a condition of one of these properties, often invalidating the guarantee (@tbl-dpprop-rules).

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

Shared noise violates the fresh-randomness condition. Suppose a release gives the number of smokers and the number of smokers aged 40 or more, using the same Laplace draw of scale 1 for both. Each release alone is 1-DP, but subtracting them cancels the noise and reveals the exact number of smokers under 40. The pair is not DP. Code can reuse noise through a shared random seed, cached value or shard-level draw. A fixed seed such as `random_state=0` makes examples reproducible, but in production it makes the noise known and removable. Releasing the same noisy answer to an identical query is safe because it is post-processing of one release.


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
