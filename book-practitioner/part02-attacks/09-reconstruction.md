# Reconstruction attacks

## Why this matters

A hospital keeps one sensitive bit per patient, whether they tested positive, and never releases it. It only answers counting questions, "how many of patients 3, 17, 42, … tested positive?", and adds a little noise to every answer. Is that safe? Dinur and Nissim (2003) showed that it is not, unless the noise is large. A table of aggregates is a system of equations whose unknowns are the people in it, and enough equations with small enough errors determine the unknowns.

The one idea: **every answer is a linear equation in the secret bits, and the error of the answers is the only thing standing between the attacker and the solution.** For a random query, the error must be of order $\sqrt n$, where $n$ is the number of people, or the attacker recovers almost every bit. And errors that are independent from answer to answer are averaged away by asking more. Dwork and Roth call this the **Fundamental Law of Information Recovery**: overly accurate answers to too many questions destroy privacy. If you run or audit a system that answers aggregate queries, this chapter tells you how to attack it. @tbl-recon-uses maps the questions to the sections.

| Question | Rule | Section |
| --- | --- | --- |
| How accurate is too accurate? | Error well below $\sqrt n$ on random queries, or below $n$ on all queries, reveals the data | @sec-recon-threshold |
| What attack do I run, and how well does it do? | Least squares; fraction recovered $\approx \Phi(\sqrt{m - n}/(4\sigma))$ | @sec-recon-ls |
| Does stopping reconstruction mean privacy? | No; and any attack needs a minimum number of queries | @sec-recon-law |
| We publish tables, not query answers | Every table cell is a constraint; overlapping tables pin down the microdata | @sec-recon-census |

: The questions behind a reconstruction audit, and where this chapter answers each. {#tbl-recon-uses}

Throughout, the secret is a bit vector $d = (d_1, \dots, d_n)$, one bit per person. A **subset-sum query** asks how many people in a set $S$ have bit 1. The **curator**, the party holding $d$, answers $m$ queries, each with an error, and "error at most $E$" means every answer is within $E$ of the truth.

## How accurate is too accurate? {#sec-recon-threshold}

**Idea.** An attacker rules out a candidate database $c$ whenever some answer $c$ would give differs from the published one by more than the curator's error allows. A query aimed exactly at the bits $c$ got wrong moves the answer by the number of wrong bits. A random query, which includes each person with probability $\frac12$, cannot aim; its sum over $k$ wrong bits behaves like $k$ coin flips and moves by about $\sqrt k$. So the error must reach these scales to hide a wrong guess.

**Result.** If the attacker may ask all $2^n$ subset sums, any candidate consistent with every answer has at most $4E$ wrong bits, and this is tight. With only random queries, Dinur and Nissim showed that error $E$ well below $\sqrt n$ on a number of queries slightly above $n$ (of order $n\log n$ in the version proved in the full edition) lets a **linear-program decoder** recover all but a small fraction of the bits in polynomial time: it finds any vector with entries between 0 and 1 whose answers are within $E$ of every published answer, and rounds it. A curator whose error stays below these scales is called **blatantly non-private**.

**Example.** With $n = 10^6$ people and every subset sum answered with error at most 1000, the exponential attack gets at most 4,000 bits wrong, 0.4% of them, although each answer is off by up to 1000. For the linear-program decoder at $n = 1000$ and a target of at most 100 wrong bits, the proof tolerates error $E \le 0.1667$ and needs $m \ge 33{,}296$ queries, about $33n$. Those constants are far from what the attack needs in practice: least squares with $m = 4n$ already works at errors of about $0.2\sqrt n$ (@sec-recon-ls).

**Pitfalls.** Don't read the proved constants as a safety margin; the scaling is what matters, and attacks beat the constants easily. Random queries are not needed either: Cohen and Nissim (2020) ran a linear-program reconstruction against a commercial query system that added noise to its answers and recovered most of a secret column. A deterministic distortion is not noise: a curator that rounds every answer to a multiple of 4 has error at most 2, but an attacker who simulates the rounding on each candidate recovers $d$ exactly. And a few wildly wrong answers don't save the curator: a variant of the decoder that minimises total absolute error tolerates a constant fraction of them (Dwork, McSherry and Talwar, 2007).

## Least squares under random and worst-case error {#sec-recon-ls}

**Idea.** The attack you would actually run needs neither a solver nor a value of $E$. **Least squares** picks the vector whose answers come closest to the published ones in squared distance, then clips each coordinate to $[0, 1]$ and rounds it. Its error is a linear function of the curator's, so you can predict its success.

**Result.** With independent Gaussian noise of standard deviation $\sigma$ on each of $m$ random subset sums, each decoded coordinate is off by a normal error with standard deviation about $2\sigma/\sqrt{m - n}$: the noise averaged over the $m - n$ equations beyond the $n$ needed, doubled because a random query includes each person only half the time. A bit is wrong when that error crosses $\frac12$, so the expected fraction recovered is
$$\Phi\left(\frac{\sqrt{m - n}}{4\sigma}\right),$$ {#eq-recon-ls-random}
where $\Phi$ is the standard normal cumulative distribution function. At a fixed ratio $m/n$ only $\sigma/\sqrt n$ matters, and at a fixed $\sigma$ the fraction tends to 1 as $m$ grows. To hold the attack at a fixed accuracy, the noise must grow like $\sqrt m$. Against worst-case errors $\lvert e_j\rvert \le E$, least squares gets at most $4mE^2/s_{\min}(A)^2$ bits wrong, where $s_{\min}(A)$ is the smallest singular value of the query matrix, the factor by which it shrinks the direction it shrinks most.

**Example.** @tbl-recon-ls averages over four random databases of $n = 500$ bits with $m = 2000$ queries.

| $\sigma/\sqrt n$ | 0.1 | 0.25 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- |
| $\sigma$ | 2.24 | 5.59 | 11.18 | 22.36 | 44.72 |
| Recovered, simulated | 1.0000 | 0.9655 | 0.8130 | 0.6550 | 0.5875 |
| Recovered, @eq-recon-ls-random | 1.0000 | 0.9585 | 0.8069 | 0.6676 | 0.5858 |

: Least-squares reconstruction of $n = 500$ bits from $m = 2000$ random subset sums with Gaussian noise. {#tbl-recon-ls}

Noise of order $\sqrt n$ is not automatically safe: at $\sigma = 0.25\sqrt n$, 96.55% of the bits come back. A curator that answers unlimited queries with $\sigma = 2\sqrt n$ looks safe at $m = 4n$ (0.5875 recovered) but gives up 90% of the bits at $m = 106.1\,n$ (@fig-recon-ls-queries).

![Fraction of bits recovered by least squares at $n = 200$ as the number of queries grows, for three noise levels. Lines are @eq-recon-ls-random; dots are simulations. Near $m = n$ the system is barely determined and the noise is amplified; beyond that, more queries average the noise away, and even at $\sigma = 2\sqrt n$ the attack recovers most bits with $64n = 12\,800$ queries.](figures/recon_ls_queries.svg){#fig-recon-ls-queries}

**Pitfalls.** A bound on the error of each answer protects nothing when the number of answers is unbounded; the error budget has to cover all of them together. Worst-case errors hurt more than random ones of the same size: at $E = 0.25\sqrt n = 5.59$ a curator that pushes every answer along the decoder's weakest direction causes 143 wrong bits, against about 17 for Gaussian noise of the same scale. So when a curator's errors are not independent random noise, test it with errors chosen against the decoder, not only with Gaussian noise of the same size. Near $m = n$ the system is barely determined and least squares amplifies noise; an attacker simply asks more.

## The Fundamental Law: what stops reconstruction, and what doesn't {#sec-recon-law}

**Idea.** Is error of order $\sqrt n$ enough to stop reconstruction, and how few queries can any attack use? A curator answering from a secret random sample settles the first; counting bits of information, the second.

**Result.** A **sampling curator** keeps each person with probability $p$ in a secret set, answers every query from that set and scales up by $1/p$. With probability $1 - \eta$, all $m$ answers have error at most $\sqrt{n\ln(2m/\eta)/(2p^2)}$, of order $\sqrt{n\log m}$, and for a uniformly random $d$ no attacker recovers more than a fraction $p + \frac{1 - p}{2}$ of the bits on average: three quarters at $p = \frac12$. The people left out never affect any answer. In the other direction, if each answer takes at most $V$ values, any attack that gets at most a fraction $\beta$ of the bits wrong needs at least
$$m \ge \frac{n\,(1 - h(\beta))}{\log_2 V}$$ {#eq-recon-info}
queries, where $h(\beta) = -\beta\log_2\beta - (1 - \beta)\log_2(1 - \beta)$ is the **binary entropy**: $n$ secret bits, less the uncertainty a guess may keep, divided by what one answer can carry.

**Example.** At $n = 500$, $m = 2000$ and $p = \frac12$, the sampling curator keeps 250 people, its largest error is 28.00, or $1.25\sqrt n$, and least squares recovers 0.7560 of the bits, right at the ceiling. For the information bound at $n = 1000$ and 5% wrong bits, exact counts ($V = 1001$) need at least 71.6 queries; rounding every count to a multiple of 10 raises the bound to 106.9, and to a multiple of 32, to 141.5. These are floors for any attack, whatever its running time.

**Pitfalls.** Stopping blatant reconstruction is not privacy. The sampling curator answers exactly for the people in its sample, and an attacker with enough queries learns all of their bits: half the people are fully exposed so that the other half can be fully hidden. A guarantee has to hold for every individual, which is what [The definition of differential privacy](../part03-dp-foundations/01-definition.md) provides, and it has to control both halves of the law, the error scale and the number of answers. That is why differential privacy tracks a budget spent across queries. Coarser answers raise the information bound, but it is only a floor on what an attack needs, not a defence.

## Reconstructing a census block {#sec-recon-census}

**Idea.** Statistical offices publish many tables from the same confidential records. Each cell is a query, and a constraint solver searches for all databases consistent with them. Garfinkel, Abowd and Martindale (2019) showed this on a toy block of seven people.

**Result.** @tbl-recon-census-published is everything published about a fictional block of seven people with an age, a sex, a race and a marital status (nobody under 18 married). Means to two decimals give exact sums: $34.29 \times 7 = 240.03$, so the ages total 240. Exactly one block is consistent with these eighteen numbers, and it is the true one (@tbl-recon-census-block).

| Group | Count | Median age | Mean age |
| --- | --- | --- | --- |
| Total | 7 | 33 | 34.29 |
| Female | 4 | 31 | 30.25 |
| Male | 3 | 33 | 39.67 |
| Black | 4 | 33.5 | 36.25 |
| White | 3 | 33 | 31.67 |
| Married | 4 | 43.5 | 47.75 |

: The published statistics for a fictional census block of seven people. A median over an even count is the average of the two middle ages. {#tbl-recon-census-published}

| Person | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Age | 7 | 15 | 27 | 33 | 35 | 52 | 71 |
| Sex | F | M | F | M | F | F | M |
| Race | B | B | W | W | W | B | B |
| Married | no | no | no | yes | yes | yes | yes |

: The block reconstructed from @tbl-recon-census-published, which is the true block. {#tbl-recon-census-block}

**Example.** The search narrows fast: of 14,790 age vectors that fit the sex statistics, 387 admit a race split, 33 a marital split, and one all three. Withholding a single statistic rarely helps (@fig-recon-census): eight of the twelve medians and means leave the block unique when withheld, because the rest imply them. The worst case for the attacker, withholding the married mean, still leaves only 45 blocks, and 51% of them have the true sex, race and marital pattern of all seven people. At scale, the Census Bureau's own reconstruction of the 2010 Census, described by Abowd (2021), rebuilt a record for each of the 308,745,538 people counted; about 46% matched a real person exactly on block, sex, age, race and ethnicity, and linking to commercial data confirmed the identity of about 17% of the population.

![Number of seven-person blocks consistent with @tbl-recon-census-published when one median or mean is withheld (log scale). Blue bars: the block is still unique. Red bars: a few candidates remain, still far fewer than the possible blocks.](figures/recon_census.svg){#fig-recon-census}

**Pitfalls.** Real publications are more exposed than the toy, because real tables cross-classify many variables at many geographic levels, so the same people appear in thousands of cells: the "too many questions" of the Fundamental Law. The usual protections leave most constraints exact: swapping a small fraction of households between blocks, the main protection of the 2010 Census, changes some answers, and coarsening ages or suppressing small cells removes some constraints. The Census Bureau's response was to move the 2020 Census to a method based on differential privacy.

## Code

To audit a query system or plan a release, three numbers come first: what fraction least squares would recover at your noise level, how many queries per person push it to a target, and the fewest queries any attack needs. The snippet computes all three, using the finite-$m$ variance $4\sigma^2 n/((n + 1)(m - n - 1))$ that the full edition derives, which is about $4\sigma^2/(m - n)$.

```python
from math import log2, sqrt
from statistics import NormalDist

PHI = NormalDist()


def ls_recovered(n: int, m: int, sigma: float) -> float:
    """Predicted fraction of n secret bits that least squares recovers from m noisy random subset sums."""
    sd = 2 * sigma * sqrt(n / ((n + 1) * (m - n - 1)))  # spread of each decoded bit's error
    return PHI.cdf(0.5 / sd)  # a bit comes out wrong when its error crosses 1/2


def queries_per_person(target: float, sigma_over_sqrt_n: float) -> float:
    """m/n at which least squares recovers a fraction `target` of the bits, for noise sigma = c * sqrt(n)."""
    return 1 + (4 * sigma_over_sqrt_n * PHI.inv_cdf(target)) ** 2


def binary_entropy(p: float) -> float:
    return -p * log2(p) - (1 - p) * log2(1 - p)


def min_queries(n: int, wrong: float, values: int) -> float:
    """Fewest answers, each taking at most `values` values, from which any attack gets all but n * wrong bits."""
    return n * (1 - binary_entropy(wrong)) / log2(values)


n, m = 500, 2000
print(" ".join(f"{ls_recovered(n, m, c * sqrt(n)):.4f}" for c in (0.1, 0.25, 0.5, 1, 2)))  # 1.0000 0.9585 0.8069 0.6676 0.5858
print(f"{queries_per_person(0.9, 2.0):.1f}")  # 106.1
print(f"{min_queries(1000, 0.05, 1001):.1f}")  # 71.6
print(f"{min_queries(1000, 0.05, 1000 // 10 + 2):.1f}")  # 106.9
print(f"{min_queries(1000, 0.05, 1000 // 32 + 2):.1f}")  # 141.5
```

The first line is the prediction row of @tbl-recon-ls. The full edition's script, [reconstruction.py](../../code/part02-attacks/reconstruction.py), runs the attacks themselves: the exponential attack, least squares under random and worst-case error, the sampling curator and the census-block search.

```bash
uv run python code/part02-attacks/reconstruction.py
```

## Cheat sheet

| Setting | Result | Notes |
| --- | --- | --- |
| All $2^n$ subset sums, error $\le E$ | Any consistent candidate has $\le 4E$ wrong bits | Tight; error $o(n)$ is blatantly non-private |
| Random subset sums, error $\le E$ | Error well below $\sqrt n$ on about $n\log n$ queries: a linear program recovers almost all bits | Proved constants are pessimistic |
| Least squares, Gaussian noise $\sigma$ | Fraction recovered $\approx \Phi(\sqrt{m - n}/(4\sigma))$ | Noise must grow like $\sqrt m$ to hold it fixed |
| Least squares, worst-case error | Wrong bits $\le 4mE^2/s_{\min}(A)^2$ | Adversarial errors hurt more than random ones |
| Sampling curator | Error $O(\sqrt{n\log m})$; recovery $\le p + \frac{1 - p}{2}$ | Stops reconstruction, exposes the sampled people |
| Information bound | $m \ge n(1 - h(\beta))/\log_2 V$ | Exact counts: of order $n/\log n$ queries |
| Published tables | Each cell is a constraint | Swapping, coarsening and suppression leave most constraints exact |

: The regimes of reconstruction and the rules for each. {#tbl-recon-cheat}

**Rules of thumb**

- The threat scale of a random subset-sum query is $\sqrt n$, of the worst query $n$. Errors well below that reveal the database.
- Independent noise per answer is averaged away by more answers; budget the error over all answers, not per answer.
- Rounding, swapping and suppression are not noise: an attacker can simulate them or route around them.
- Stopping reconstruction is not the same as privacy; ask what the system guarantees for each individual.

## Further reading

- Dinur & Nissim, "Revealing Information while Preserving Privacy" (PODS 2003): the original attacks and the notion of blatant non-privacy.
- Dwork, Smith, Steinke & Ullman, "Exposed! A Survey of Attacks on Private Data" (Annual Review of Statistics and Its Application, 2017): a readable survey of reconstruction and tracing attacks.
- Garfinkel, Abowd & Martindale, "Understanding Database Reconstruction Attacks on Public Data" (CACM 2019): the seven-person block reconstructed with a solver.
- Cohen & Nissim, "Linear Program Reconstruction in Practice" (Journal of Privacy and Confidentiality, 2020): the attack against a deployed query system.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/09-reconstruction.md).
