# Post-processing, group privacy and composition

## Why this matters

A private mechanism is never the whole system. Its output gets rounded, plotted or fed into a model, the people in your data may each own many records, and an analyst rarely asks only one question. So you need to know what guarantee holds for what you actually publish in each case, without writing a new proof for every pipeline.

Three properties of differential privacy (DP) give that answer. The first, post-processing, says that anything computed from a private output without reading the data again is free. The second, group privacy, says what a guarantee about one record promises for a person who owns several. And composition says that running several mechanisms on the same data costs at most the sum of their budgets. Together they turn privacy into bookkeeping, because you can prove each building block once and then keep a ledger of what you spend.

## Post-processing {#sec-dpprop-post}

Take a private mechanism, written $M$. Write $\varepsilon$ for its privacy level and $\delta$ for its slack, so that it is $(\varepsilon, \delta)$-DP. Apply to its output any function $f$, which may use random coins of its own but never reads the data. Then the processed release is just as private,
$$f(M(D)) \text{ is } (\varepsilon, \delta)\text{-DP}.$$ {#eq-dpprop-post}
Processing can merge outputs that point in different directions, and so it can lower the privacy loss, but it can never raise it.

This is what makes DP practical. Rounding a noisy count, clamping it at zero, or making noisy totals consistent with their parts costs no extra budget. A model trained on private synthetic data, a plot and a dashboard all inherit the guarantee with no further proof too. So a private system adds noise at one narrow point, such as noisy counts or noisy gradients, and treats everything after it as public. The same rule covers attackers. A membership-inference attack is a function of the output, so the attack cap of the definition holds even for attacks invented after the release.

The condition is that $f$ never reads the data. "Take the noisy top category, then report its exact count" breaks it, because the second step is a new and non-private mechanism. So does "publish the noisy count, unless the true count is 0", which peeks at the raw data to decide. A finer histogram computed after an odd bin catches your eye is a second mechanism too, whose cost adds by composition.

## Group privacy {#sec-dpprop-group}

The definition protects one record, but a user of a messaging app contributes hundreds of messages. If the guarantee is per record and a person owns $k$ records, you need a guarantee for datasets that differ in $k$ records. For pure DP, the losses of the $k$ steps from one dataset to the other add up, so $\varepsilon$-DP per record becomes $k\varepsilon$-DP for the group. With a slack, each step also adds its own $\delta$, and every later step multiplies the earlier ones by $e^\varepsilon$. For a group of $k$ records, an $(\varepsilon, \delta)$-DP mechanism is
$$\Big(k\varepsilon,\ \delta_k\Big)\text{-DP}, \qquad \delta_k = \delta\,\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1} \le k\,e^{(k-1)\varepsilon}\delta.$$ {#eq-dpprop-group}
The simpler form on the right is looser, so compute the sum when you can.

The price is steep. A user with 50 rows under a row-level guarantee of $\varepsilon$ is protected only at $50\varepsilon$. The slack grows even faster, because $\delta$ is multiplied by 12818.3 for a group of 10 at $\varepsilon = 1$ per record. That hurts, since $\delta$ was already chosen much smaller than one over the number of records. For example, a guarantee of $(0.5, 10^{-5})$ per record becomes $(5, 2.272\times10^{-3})$ for a user with 10 records.

So if you care about people, don't patch a per-record guarantee with group privacy. Define neighbours as datasets that differ in one person's records, and bound each person's contribution, for example by capping how many records a user may add. Then prove DP at that level directly. Group privacy is the fallback for when you cannot, and for correlated records such as households.

## Sequential composition {#sec-dpprop-comp}

An analyst asks several queries of the same data, each answered with fresh noise, and an adversary sees all the answers together. Write $M_i$ for the $i$-th of $k$ mechanisms and $(\varepsilon_i, \delta_i)$ for its budget. Releasing all their outputs is then
$$\Big(\sum_{i=1}^k\varepsilon_i,\ \sum_{i=1}^k\delta_i\Big)\text{-DP}.$$ {#eq-dpprop-basic}
This is **basic composition**. It holds because independent outputs have independent noise, so their privacy losses add. It needs every mechanism to use fresh coins and the same neighbouring relation.

The rule also covers **adaptive** composition, where each query is chosen after reading the earlier answers, as long as each budget is fixed in advance. For example, an eleventh query that runs only if the first ten look interesting is covered, provided its budget was fixed beforehand. Budgets chosen from the answers, as in "keep going until the estimate looks good", are another matter. They need a privacy filter, a rule that decides when to stop as the budget is spent ([Adaptive composition](../part04-privacy-accounting/08-adaptive-composition.md#sec-adapt-dp-filters)). Pure DP with basic composition is the exception, since stopping before a fixed total is exceeded is always safe there.

Basic composition is loose for many releases. Ten Gaussian counts at $(0.5, 10^{-5})$ each cost an exact $\varepsilon$ of 1.504, where basic composition charges 5. So use the accountants of [Advanced and optimal composition](04-composition.md) and [Part 4](../part04-privacy-accounting/00-introduction.md) once releases number more than a handful.

## Parallel composition {#sec-dpprop-parallel}

A histogram counts people in disjoint bins, and per-region statistics use disjoint subsets of the data. If you split the data by region, each region can spend the full budget, because one person's record falls in one region only. Split the possible records into disjoint parts by a public rule fixed in advance, so that each person lands in exactly one part. Then run a mechanism $M_i$ with independent coins on the records in the $i$-th part. If each $M_i$ is $(\varepsilon_i, \delta_i)$-DP under add/remove, the whole release is
$$\big(\max_i\varepsilon_i,\ \max_i\delta_i\big)\text{-DP under add/remove}.$$ {#eq-dpprop-parallel}

This is why a histogram costs one $\varepsilon$, not one per bin. With Laplace noise of scale $1/\varepsilon$ on every bin, the histogram is $\varepsilon$-DP however many bins it has. Under replace, a record can move from one bin to another, so two bins change and the cost doubles to $2\varepsilon$. The cost goes back to the maximum when each part is a fixed slot that belongs to one person, as in local randomized response.

The rule fixed in advance is the condition that is easiest to break. A split into "the top 10% of earners and the rest" moves its boundary when one high earner is added, so records cross it and both parts change. The same goes for partitioning on the list of keys that happen to appear in the data, because the list itself reveals who is present. Use fixed bins or a public key list, or treat a data-dependent split as a mechanism of its own.

## Keeping a budget {#sec-dpprop-budget}

Fix a total budget for a dataset before the first release, and charge every release against it. Suppose a dashboard over one patient table has a total of $(1, 10^{-5})$. A histogram of 20 diagnoses with $\varepsilon = 0.5$ on every bin costs only 0.5, by parallel composition. A mean age, released as a noisy sum and a noisy count at 0.125 each, costs 0.25 by sequential composition. A set of Gaussian counts at $(0.25, 10^{-5})$ uses the rest. Any further query must then be refused, or paid for from a new budget, and the snippet in @sec-dpprop-code keeps exactly this ledger. Libraries keep the same ledger. OpenDP's `Context.compositor` takes a total budget and a number of queries. Its `context.query()` then charges each one. diffprivlib's `BudgetAccountant` tracks spending across its mechanisms, and Tumult Analytics sessions hold a total budget and evaluate each query against it.

Most real failures break a condition of the three properties, and the damage is usually total rather than a small loss. @tbl-dpprop-rules lists the common cases.

| Situation | Composes? | What to do |
| --- | --- | --- |
| A function of the output, with its own coins | Yes, for free | Process as much as you like |
| Another query on the same data, fresh noise | Yes, budgets add | Charge it to the ledger |
| A query chosen from earlier answers | Yes, budgets add | Fix its budget in advance |
| Two releases that share one noise draw | No | Draw fresh noise for every query |
| "Post-processing" that reads the data again | No | Charge it as a new mechanism |
| A DP release beside an exact statistic of the same data | No | Release the statistic privately too |
| A split whose boundaries depend on the data | No, not in parallel | Fix the bins in advance |
| Guarantees under add/remove and replace in one ledger | Only after conversion | Convert, at $2\varepsilon$ |

: What composes and what doesn't. {#tbl-dpprop-rules}

Shared noise is the clearest case. Release the number of smokers and the number of smokers aged 40 or over, adding the same Laplace draw to both. Each release alone is 1-DP, but their difference cancels the noise and gives the exact number of smokers under 40, so the pair is not DP at all. In code, the bug hides in two mechanisms seeded with the same random seed and in a cached noise value re-added to a related query. It also hides in noise reused across the shards of a distributed system. Returning the same noisy answer to a repeated identical query is fine, because that is post-processing of one release.

## Code {#sec-dpprop-code}

The ledger below charges each release against a total fixed in advance, and refuses any release that would exceed it. It also computes group privacy, where hand calculations go wrong most often.

```python
from math import exp


def group_privacy(eps: float, delta: float, k: int) -> tuple[float, float]:
    """The guarantee for a group of k records, from an (eps, delta) guarantee per record."""
    return k * eps, delta * sum(exp(j * eps) for j in range(k))


class Ledger:
    """Basic composition for releases from one dataset, with a total fixed in advance."""

    def __init__(self, eps_total: float, delta_total: float) -> None:
        self.eps_total, self.delta_total = eps_total, delta_total
        self.eps_spent, self.delta_spent = 0.0, 0.0

    def charge(self, name: str, eps: float, delta: float = 0.0) -> None:
        if self.eps_spent + eps > self.eps_total or self.delta_spent + delta > self.delta_total:
            raise ValueError(f"{name}: over budget")
        self.eps_spent += eps
        self.delta_spent += delta

    def charge_parallel(self, name: str, budgets: list[tuple[float, float]]) -> None:
        """Mechanisms on disjoint parts of the data, under add/remove, cost the largest budget."""
        self.charge(name, max(e for e, _ in budgets), max(d for _, d in budgets))


ledger = Ledger(1.0, 1e-5)
ledger.charge_parallel("histogram, 20 bins", [(0.5, 0.0)] * 20)
ledger.charge("noisy sum", 0.125)
ledger.charge("noisy count", 0.125)
ledger.charge("Gaussian counts", 0.25, 1e-5)
print(ledger.eps_spent, ledger.delta_spent)  # 1.0 1e-05
try:
    ledger.charge("one more count", 0.1)
except ValueError as err:
    print(err)  # one more count: over budget
eps_k, delta_k = group_privacy(0.5, 1e-5, 10)
print(f"{eps_k:.1f} {delta_k:.3e}")  # 5.0 2.272e-03
print(f"{group_privacy(1.0, 1.0, 10)[1]:.1f}")  # 12818.3
```

## Cheat sheet

| Property | Rule | Condition |
| --- | --- | --- |
| Post-processing | $f(M(D))$ is $(\varepsilon, \delta)$-DP | $f$ never reads the data |
| Group privacy | $\big(k\varepsilon,\ \delta\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1}\big)$ for $k$ records | Prefer user-level neighbours |
| Basic composition | $(\sum\varepsilon_i, \sum\delta_i)$ | Fresh noise, one neighbouring relation, budgets fixed in advance |
| Parallel composition | $(\max\varepsilon_i, \max\delta_i)$ | Add/remove, parts fixed in advance; $2\varepsilon$ under replace |

: The results of this chapter. {#tbl-dpprop-cheat}

**Rules of thumb**

- Add noise once, as early as possible, and everything computed from the noisy output after it is free.
- Fix the total budget before the first release, and keep a ledger.
- Budgets add across queries on the same data, and take the maximum across disjoint parts of it.
- Use basic composition for a handful of releases, and an accountant when there are many.
- Never share a random seed between mechanisms, and never partition on the keys found in the data.

## Further reading

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014): post-processing, group privacy and basic composition, the standard reference.
- McSherry, "Privacy Integrated Queries" (SIGMOD 2009): parallel composition, and a query system built on post-processing and composition as its only rules.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/02-properties.md).
