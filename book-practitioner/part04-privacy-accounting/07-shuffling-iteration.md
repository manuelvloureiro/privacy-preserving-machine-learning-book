# Amplification by shuffling and by iteration

## Why this matters

Uncertainty is worth privacy, as [subsampling](05-subsampling.md) showed, and two other kinds of it matter in practice. The first is about which report is yours. In the shuffle model ([Local, shuffle and distributed trust models](../part03-dp-foundations/09-trust-models.md)), each of $n$ users randomizes their own data, and a **shuffler** strips identities and mixes the order. Each report satisfies local differential privacy (DP) at a level $\varepsilon_0$ that is usually large. But your report hides among the others, so the bag of reports satisfies central DP with an $\varepsilon$ that falls like $1/\sqrt n$.

The second is uncertainty about what happened inside training. If only the final model is released, later noise also covers a record used early, and on a convex loss the privacy loss can even stop growing. Of the two, shuffling is a deployable tool with a computable guarantee, used for example in Prochlo (Bittau et al. 2017). Hidden-state amplification, by contrast, rarely applies to deep learning. All results use replace-one neighbours, two datasets of the same size that differ in the value of one record.

## Shuffled binary randomized response, exactly {#sec-shuf-rr}

For the simplest local randomizer, the shuffled guarantee can be computed exactly. Each user holds a bit and sends it through binary randomized response ([The definition of differential privacy](../part03-dp-foundations/01-definition.md#sec-dpdef-rr)). The user reports the true bit with probability $\theta = e^{\varepsilon_0}/(1 + e^{\varepsilon_0})$ and the flipped bit otherwise. After shuffling, the analyst learns only how many reports are 1, and your report moves that count by at most one.

So the guarantee is the privacy of a pair of count distributions. Let your bit be 0 in $D$ and 1 in $D'$, and suppose the other users all hold 0. Each of them reports 1 with probability $1 - \theta$, so the number of their 1-reports is binomial, and we write $b(y)$ for its chance of being $y$. The count is then distributed as
$$P(y) = \theta\,b(y) + (1 - \theta)\,b(y - 1), \qquad Q(y) = (1 - \theta)\,b(y) + \theta\,b(y - 1),$$ {#eq-shuf-rr-pair}
when you hold 0 and when you hold 1. Its exact privacy curve is a sum over the possible counts, the same sum as in [Numerical accounting](06-numerical-accounting.md#sec-num-pld). You compute it in both directions and keep the larger, which the snippet in @sec-shuf-code does in a few seconds. Other values for the other users change it by less than $10^{-4}$ in the full edition's checks.

For intuition, the count is close to normal, and your report shifts its mean against a spread that grows like $\sqrt n$. So the shuffled count approximately satisfies Gaussian differential privacy (GDP) ([f-DP and Gaussian DP](04-gaussian-dp.md)), with a shift in standard deviations of
$$\mu \approx \frac{2\sinh(\varepsilon_0/2)}{\sqrt n}.$$ {#eq-shuf-rr-clt}
Don't deploy with this approximation, because it fails when the others send few 1-reports. For example, with 1000 users at $\varepsilon_0 = 4$, it gives 0.9667 where the exact value is 1.6910.

Shuffling never improves pure DP, the guarantee whose slack $\delta$ is zero. An all-zero count is still $e^{\varepsilon_0}$ times likelier when you hold 0, however many users there are, and shuffling only makes it rare. So any shuffled guarantee below $\varepsilon_0$ needs $\delta > 0$.

## The clone bound for any local randomizer {#sec-shuf-clones}

Real deployments send histograms, categories or noisy vectors, so you need a bound for every $\varepsilon_0$-DP **local randomizer**, the randomized map from one user's data to their report. The bound to use is that of Feldman, McMillan & Talwar (2021). Its idea is that each other user is, with probability $e^{-\varepsilon_0}$, a **clone** whose report looks exactly like one of your two possible reports.

That reduction gives a small pair of count distributions, whose exact $\delta(\varepsilon)$ is the **numeric clone bound**. It holds for every $\varepsilon_0$-DP randomizer, even one chosen adaptively, as when each user randomizes their own gradient in shuffled stochastic gradient descent (SGD). The same paper also gives a closed form, which needs enough clones to hide among,
$$\varepsilon_0 \le \ln\frac{n}{16\ln(2/\delta)}.$$
When that holds, the shuffled output is $(\varepsilon, \delta)$-DP with
$$\varepsilon \le \ln\left(1 + \frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1}\left(\frac{8\sqrt{e^{\varepsilon_0}\ln(4/\delta)}}{\sqrt n} + \frac{8e^{\varepsilon_0}}{n}\right)\right).$$ {#eq-shuf-clone-bound}
For large $n$ its leading term gives the shape to remember, because the central $\varepsilon$ falls like $\sqrt{\ln(1/\delta)/n}$ with a constant that grows like $e^{\varepsilon_0/2}$. Where the condition fails, though, the closed form says nothing.

| $n$ | $\varepsilon_0$ | Exact, randomized response | Clones, numeric | Clones, closed form |
| --- | --- | --- | --- | --- |
| 1,000 | 4 | 1.6910 | 3.9900 | n/a |
| 10,000 | 2 | 0.0870 | 0.1550 | 0.5009 |
| 100,000 | 2 | 0.0247 | 0.0450 | 0.1862 |
| 100,000 | 4 | 0.0847 | 0.1698 | 0.5346 |

: Central $\varepsilon$ at $\delta = 10^{-6}$ after shuffling $n$ reports that are each $\varepsilon_0$-DP. "n/a" marks where the condition of @eq-shuf-clone-bound fails. {#tbl-shuf-compare}

@tbl-shuf-compare shows what generality costs. Where many clones are expected, the numeric clone bound is 1.44 to 2.00 times the exact value for randomized response. The closed form is 3.10 to 4.75 times the numeric bound where it applies. So use the closed form to reason about scaling, and compute the numeric bound to deploy.

## Choosing ε₀ and deploying a shuffler {#sec-shuf-deploy}

The practical question runs backwards, because you want the largest $\varepsilon_0$ that meets a target central guarantee, which means the least noise per report. So compute the bound for a few candidate values of $\varepsilon_0$ and keep the largest that meets the target. The choice of bound can decide feasibility, and a larger $\varepsilon_0$ buys accuracy. Take 100,000 users and a target of 0.2 at $\delta = 10^{-6}$. Among the values in @tbl-shuf-compare, the closed form allows $\varepsilon_0$ up to 2, while the numeric clone bound allows 4.

Before you report a shuffled $\varepsilon$, check that the deployment meets the assumptions of the bound.

- **The order is uniformly random and hidden.** With a fixed or leaked order, or metadata such as arrival time or network address, the analyst knows which report is yours, and the guarantee falls back to $\varepsilon_0$.
- **The shuffler is trusted not to collude with the analyst.** If they collude, again only $\varepsilon_0$ remains.
- **The other users are honest.** Users who collude with the analyst reveal their own reports, so count only honest users in $n$.
- **The computation uses the number of reports actually shuffled.** The bound gets worse as $n$ falls, so use the smallest batch you will release, not the number of users enrolled.
- **Each user sends one report per batch.** Otherwise the reports compose like any other releases.

## Amplification by iteration {#sec-shuf-iteration}

Suppose projected noisy SGD makes one pass over $n$ records and releases only the final model. Then the noise of every later step protects a record used early (Feldman et al. 2018). The guarantee needs every step to be a **contraction**, a map that never pushes two models further apart. Write $\beta$ for the smoothness of the loss, the most its gradient changes per unit of distance. A gradient step on a convex loss is then a contraction when the step size is at most $2/\beta$. A projection onto a convex set is one too. With Lipschitz losses as well, the record used at the $t$-th step gets one Gaussian step's Rényi differential privacy (RDP) curve. That curve is divided by the $n - t + 1$ noisy steps from $t$ to the end.

So the guarantee depends on the position, and a report must state the worst one. For a pass over 10,000 records with noise multiplier 1, the last record gets $\varepsilon = 5.2985$ at $\delta = 10^{-5}$, converted as in [RDP conversions](02-rdp-conversions.md#sec-conv-improved). That is a single Gaussian step, with no amplification at all, while the record in the middle gets 0.0707.

## Hidden state: privacy that stops growing {#sec-shuf-hidden}

Full-batch noisy gradient descent (GD) uses every record at every step, so composition charges in proportion to the number of steps. On a bounded convex set, though, the runs on $D$ and $D'$ can never drift further apart than the set's **diameter** $d_K$, the largest distance between two of its points. Altschuler & Talwar (2022) prove that the privacy loss of the last iterate then stops growing. Write $s$ for the per-step shift between the runs and $\sigma$ for the per-step noise, both in parameter units. When $d_K/s$ is an integer, the RDP of the last iterate stays below
$$\varepsilon(\alpha) \le \frac{2\alpha d_Ks}{\sigma^2}$$ {#eq-shuf-plateau}
however long you train.

For example, on the unit interval with $s = 0.01$ and $\sigma = 0.1$, the bound stops at 4.0000 at order 2, while composition charges 20.0000 for 2000 steps. **Strong convexity** gives a plateau without a bounded set, because each step then shrinks the gap between the runs. The plateau arrives after about as many steps as the loss's condition number (Chourasia, Ye & Shokri 2021).

These results need two conditions, and deep learning breaks both. No mainstream library, Opacus or `dp_accounting` included, offers last-iterate accounting. The first is that **only the last iterate is released**. Checkpoints, logged models or picking the best checkpoint put you back to composition. The second is that every step is a contraction, which requires a convex loss. Annamalai (2024) constructs a non-convex loss for which the last iterate of differentially private SGD (DP-SGD) leaks as much as all iterates together. So never report a hidden-state $\varepsilon$ for non-convex training, and account for it with the ordinary DP-SGD composition instead. Use it for convex models, such as private logistic regression, that release only their final weights.

## Code {#sec-shuf-code}

This snippet computes the exact central $\varepsilon$ of shuffled binary randomized response from @eq-shuf-rr-pair, so you can use it to choose $\varepsilon_0$. With SciPy, `scipy.stats.binom.pmf` gives the binomial probabilities directly.

```python
from math import exp, lgamma, log


def log_binom_pmf(k: int, n: int, p: float) -> float:
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(p) + (n - k) * log(1 - p)


def shuffled_rr_pair(n: int, eps0: float) -> tuple[list[float], list[float]]:
    theta = exp(eps0) / (1 + exp(eps0))
    b = [exp(log_binom_pmf(y, n - 1, 1 - theta)) for y in range(n)] + [0.0]
    P = [theta * b[y] + (1 - theta) * (b[y - 1] if y else 0.0) for y in range(n + 1)]
    Q = [(1 - theta) * b[y] + theta * (b[y - 1] if y else 0.0) for y in range(n + 1)]
    return P, Q


def hockey_stick(P: list[float], Q: list[float], eps: float) -> float:
    return sum(max(p - exp(eps) * q, 0.0) for p, q in zip(P, Q))


def shuffled_rr_epsilon(n: int, eps0: float, delta: float) -> float:
    P, Q = shuffled_rr_pair(n, eps0)
    lo, hi = 0.0, eps0  # shuffling never does worse than the local eps0
    for _ in range(50):
        mid = (lo + hi) / 2
        worst = max(hockey_stick(P, Q, mid), hockey_stick(Q, P, mid))
        lo, hi = (mid, hi) if worst > delta else (lo, mid)
    return hi


print(f"{shuffled_rr_epsilon(10_000, 2.0, 1e-6):.4f}")  # 0.0870
print(f"{shuffled_rr_epsilon(100_000, 4.0, 1e-6):.4f}")  # 0.0847
print(f"{shuffled_rr_epsilon(1_000, 4.0, 1e-6):.4f}")  # 1.6910
```

## Cheat sheet

| Situation | Accountant | Conditions and pitfalls |
| --- | --- | --- |
| Binary randomized response behind a shuffler | Exact count pair, @eq-shuf-rr-pair | The normal approximation fails with few 1-reports |
| Any $\varepsilon_0$-DP randomizer behind a shuffler | Numeric clone bound | 1.44 to 2.36 times the exact value for randomized response |
| One-pass noisy SGD, last iterate | $\varepsilon_t(\alpha) = \frac{\alpha}{2\tilde\sigma^2(n - t + 1)}$ | Convex, smooth, Lipschitz; report the last record's value |
| Full-batch GD on a bounded convex set | Plateau $\frac{2\alpha d_Ks}{\sigma^2}$ | Only the last iterate released |
| Non-convex training, or checkpoints released | Ordinary composition | No hidden-state amplification |

: Which accountant for which kind of amplification. All use replace-one neighbours. {#tbl-shuf-cheat}

**Rules of thumb**

- Choose $\varepsilon_0$ with the numeric bound at the smallest batch size you will release.
- Don't deploy with the closed form or the normal approximation when users are few or $\varepsilon_0$ is large, because both fail there.
- A leaked order, a colluding shuffler or identifying metadata turns the guarantee back into $\varepsilon_0$.
- Never claim hidden-state privacy for deep learning, and for convex models release only the final weights.

## Further reading

- Feldman, McMillan & Talwar, "Hiding Among the Clones: A Simple and Nearly Optimal Analysis of Privacy Amplification by Shuffling" (FOCS 2021): the clone bound and how to compute it. The full edition's script, [shuffling_iteration.py](../../code/part04-privacy-accounting/shuffling_iteration.py), computes it too.
- Annamalai, "It's Our Loss: No Privacy Amplification for Hidden State DP-SGD With Non-Convex Loss" (AISec 2024): why hidden-state results fail for deep learning.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/07-shuffling-iteration.md).
