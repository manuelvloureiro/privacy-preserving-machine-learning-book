# Amplification by shuffling and by iteration

## Why this matters

[Amplification by subsampling](05-subsampling.md) showed that uncertainty is worth privacy: a step that uses your record only with probability $q$ leaks far less than its noise alone suggests. Two other kinds of uncertainty matter in practice, and each needs its own accountant.

The first is uncertainty about **which report is yours**. In the shuffle model, $n$ users randomize their own data, and a trusted shuffler strips identities and mixes the order. Each report alone satisfies local differential privacy (DP) at a level $\varepsilon_0$, usually large to keep reports useful. But your report hides among the others, and the bag is central DP with an $\varepsilon$ that falls like $1/\sqrt n$. This is not the shuffle-and-partition batching used to train models in the central model, where no comparable theorem applies ([Amplification by subsampling](05-subsampling.md)).

The second is uncertainty about **what happened inside training**. Noisy gradient descent (GD) on a convex loss forgets: each later step adds fresh noise and pulls nearby parameters together. If only the final model is released, early records are better hidden than composition says, and for some problems the loss stops growing with the number of steps. These results are research-level with strict conditions; this chapter gives their ideas and when they apply.

The one idea behind every result: what the adversary sees can be generated from a much simpler pair of distributions by a procedure that does not depend on your record, so by post-processing the simple pair's privacy carries over. For shuffling the simple pair is a pair of counts; for iteration, a pair of Gaussians. Throughout, neighbouring datasets differ by **replacing** one record. @tbl-shuf-overview maps the decisions to the sections.

| Decision or result | Rule | Section |
| --- | --- | --- |
| Central $\varepsilon$ of shuffled binary randomized response | Compute it exactly from the count distribution | @sec-shuf-rr |
| Central $\varepsilon$ for any local randomizer behind a shuffler | The numeric clone bound; the closed form only for scaling | @sec-shuf-clones |
| Releasing only the last model of a one-pass convex run | Per-record guarantee that depends on position | @sec-shuf-iteration |
| Training many epochs, releasing only the last model | Plateau only with a bounded domain or strong convexity, never for deep learning | @sec-shuf-hidden |

: The decisions behind amplification by shuffling and by iteration, and where this chapter makes each one. {#tbl-shuf-overview}

## Shuffled randomized response, exactly {#sec-shuf-rr}

**Idea.** Each user reports a bit with binary [randomized response](../part03-dp-foundations/01-definition.md): the true bit with probability $\theta = e^{\varepsilon_0}/(1 + e^{\varepsilon_0})$, the flipped bit otherwise. Once the order is gone, the only thing the bag tells the analyst is the **number of 1-reports**. Your report moves that count by at most one, and the other users' random flips hide the move.

**Result.** Let your bit be 0 in $D$ and 1 in $D'$, and let the others all hold 0, so their number of 1-reports is binomial, $\mathrm{Bin}(n - 1, 1 - \theta)$, with probability mass function $b$. The count has distribution
$$P(y) = \theta\,b(y) + (1 - \theta)\,b(y - 1), \qquad Q(y) = (1 - \theta)\,b(y) + \theta\,b(y - 1),$$ {#eq-shuf-rr-pair}
mixing the two cases "your report is 0" and "your report is 1". The exact $\delta(\varepsilon)$ is the larger of the two hockey-stick divergences $\sum_y(P(y) - e^\varepsilon Q(y))_+$ and the same with $P$ and $Q$ swapped, computed in milliseconds even for $n = 10^5$ (@sec-shuf-code). Other configurations of the other users' bits give $\varepsilon$ within $10^{-4}$ of this pair in every case checked. For intuition, the count is nearly normal, and the central guarantee is roughly that of a Gaussian mechanism with
$$\mu \approx \frac{2\sinh(\varepsilon_0/2)}{\sqrt n}:$$ {#eq-shuf-rr-clt}
it falls like $1/\sqrt n$ and grows like $e^{\varepsilon_0/2}$.

**Example.** A company asks 10,000 users whether they used a feature, through randomized response with $\varepsilon_0 = 2$ and a shuffler. Each report alone is 2-DP: it is the truth with probability $\theta = 0.8808$. The shuffled bag is $(0.0870, 10^{-6})$-DP. The shuffler bought central-model privacy without anyone seeing raw data.

![Left: the number of 1-reports among $n = 30$ shuffled randomized-response reports with $\varepsilon_0 = 2$, when you hold 0 (blue) and 1 (orange); your report moves the count by at most one. Right: the exact $\delta(\varepsilon)$ for one unshuffled report and for $n = 10, 100, 1000$ shuffled reports. Every curve reaches 0 only at $\varepsilon = \varepsilon_0$, but for large $n$ it is negligible far earlier.](figures/shuffling_iteration_counts.svg){#fig-shuf-counts}

**Pitfalls.** Shuffling never improves pure DP: if nobody reports a 1, the likelihood ratio is still $e^{\varepsilon_0}$, whatever $n$. It only makes such outcomes rare and moves them into $\delta$ (@fig-shuf-counts), so a shuffled guarantee always needs $\delta > 0$. Don't use the normal approximation when there are few 1-reports to hide among: at $n = 1000$ and $\varepsilon_0 = 4$ the others send about 18 of them on average, and the approximation gives 0.9667 against the exact 1.6910. Compute the exact value instead.

## The clone bound for any local randomizer {#sec-shuf-clones}

**Idea.** Users often send histograms, $k$-ary answers or noisy vectors, so you need a bound for every $\varepsilon_0$-DP local randomizer. The argument of Feldman, McMillan & Talwar (2021) is that any such randomizer, whatever its user holds, produces with probability $e^{-\varepsilon_0}$ a report distributed exactly like yours could be: a **clone**, equally likely to look like your $D$-version or your $D'$-version. After shuffling, the adversary can do no better than count how many reports look like each version, and the clones blur that count. About $ne^{-\varepsilon_0}$ others are clones.

**Result.** The shuffled output is a post-processing of a pair of two-dimensional counts (their Theorem 3.2). Its exact $\delta(\varepsilon)$ is the **numeric clone bound**, valid for every $\varepsilon_0$-DP randomizer, even adaptively chosen ones. Their closed form (Theorem 3.1) bounds it analytically:
$$\varepsilon \le \ln\left(1 + \frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1}\left(\frac{8\sqrt{e^{\varepsilon_0}\ln(4/\delta)}}{\sqrt n} + \frac{8e^{\varepsilon_0}}{n}\right)\right), \qquad\text{valid when } \varepsilon_0 \le \ln\frac{n}{16\ln(2/\delta)}.$$ {#eq-shuf-clone-bound}
In words, the central $\varepsilon$ shrinks like $1/\sqrt n$, with a constant growing like $e^{\varepsilon_0/2}$ and a factor $\sqrt{\ln(1/\delta)}$, provided the expected number of clones is large enough. @tbl-shuf-compare compares the bounds with the exact value for binary randomized response, which no general bound can beat.

| $n$ | $\varepsilon_0$ | Exact, binary randomized response | Normal approx., @eq-shuf-rr-clt | Clones, numeric | Clones, closed form |
| --- | --- | --- | --- | --- | --- |
| 1,000 | 1 | 0.1266 | 0.1212 | 0.1824 | 0.5662 |
| 1,000 | 2 | 0.3233 | 0.2890 | 0.5455 | n/a |
| 1,000 | 4 | 1.6910 | 0.9667 | 3.9900 | n/a |
| 10,000 | 1 | 0.0357 | 0.0352 | 0.0530 | 0.2140 |
| 10,000 | 2 | 0.0870 | 0.0844 | 0.1550 | 0.5009 |
| 10,000 | 4 | 0.3146 | 0.2816 | 0.6009 | n/a |
| 100,000 | 1 | 0.0101 | 0.0101 | 0.0153 | 0.0726 |
| 100,000 | 2 | 0.0247 | 0.0244 | 0.0450 | 0.1862 |
| 100,000 | 4 | 0.0847 | 0.0822 | 0.1698 | 0.5346 |

: Central $\varepsilon$ at $\delta = 10^{-6}$ after shuffling $n$ reports that are each $\varepsilon_0$-DP. "n/a" marks where the condition of @eq-shuf-clone-bound fails. {#tbl-shuf-compare}

**Example.** You collect a bit from $n = 10^5$ users behind a shuffler and want central $\varepsilon \le 0.2$ at $\delta = 10^{-6}$. The closed form allows $\varepsilon_0 = 2$ (0.1862). The numeric clone bound allows $\varepsilon_0 = 4$ (0.1698), and for binary randomized response the exact value at $\varepsilon_0 = 4$ is 0.0847. At $\varepsilon_0 = 4$ each report is the truth with probability 0.9820, and the noise adds only $0.0190/n$ to the variance of the estimated proportion, against up to $0.25/n$ from sampling alone. The choice of accountant decides whether the deployment is feasible.

**Pitfalls.** The closed form is 3.10 to 4.75 times the numeric bound where it applies: use it to reason about scaling, and compute the numeric bound to deploy. Its condition matters; at $\delta = 10^{-6}$ it caps $\varepsilon_0$ at 1.460 for $n = 1000$, 3.763 for $n = 10^4$ and 6.066 for $n = 10^5$. The older bound of Erlingsson et al. (2019) grows like $e^{3\varepsilon_0}$ instead of $e^{\varepsilon_0/2}$, and in @tbl-shuf-compare amplifies at all only at $n = 10^5$, $\varepsilon_0 = 1$. Every result here assumes a uniformly random order the analyst cannot see: if the order leaks, the guarantee falls back to $\varepsilon_0$.

## Amplification by iteration {#sec-shuf-iteration}

**Idea.** Projected noisy stochastic gradient descent (SGD) makes one pass over the data, using record $x_t$ at step $t$. If every iterate is released, $x_t$ is protected only by the noise of step $t$. If only the final model is released, the noise of every later step also covers it. Feldman, Mironov, Talwar & Thakurta (2018) track how far apart the runs on $D$ and $D'$ might be. The step using your record pushes them apart; each later noisy step "pays off" part of the gap at a cost quadratic in the payment, so spreading it evenly is cheapest. The later steps must never push points apart (they must be **contractions**), which holds for a gradient step on a convex, $\beta$-smooth loss with step size $\eta \le 2/\beta$, followed by projection onto a convex set.

**Result.** For convex, Lipschitz, smooth losses and noise multiplier $\tilde\sigma$ (noise per unit of the gradient's replace-one sensitivity), releasing only the last iterate gives the record used at step $t$ of $n$ the Rényi differential privacy (RDP) curve
$$\varepsilon_t(\alpha) = \frac{\alpha}{2\tilde\sigma^2}\cdot\frac{1}{n - t + 1}:$$ {#eq-shuf-fmtt}
one Gaussian step's RDP curve, divided by the number of noisy steps from $t$ to the end. The shifted-divergence machinery behind it is in the full edition.

**Example.** One pass over $n = 10{,}000$ records with $\tilde\sigma = 1$, converted to $(\varepsilon, 10^{-5})$ with Mironov's RDP conversion ([Rényi DP](01-renyi-dp.md)). The last record gets $\varepsilon = 5.2985$, as if every iterate had been released. The record used 100 steps before the end gets 0.4849, the one in the middle 0.0707, and the first 0.0481.

![The $\mu$ of the last iterate against the step $t$ at which your record is used, for a one-pass run of $T = 100$ steps with per-step shift equal to the noise standard deviation. Releasing all iterates gives $\mu = 1$ for every record (dotted). The bound @eq-shuf-fmtt (red) is exact in the worst case; for typical contractions (blue, orange) the true value is smaller.](figures/shuffling_iteration_positions.svg){#fig-shuf-positions}

**Pitfalls.** The guarantee is per record and depends on position, so a report must state the worst record's value, which is the last one's and gets no amplification at all. Randomizing the stopping time spreads the benefit across records. The bound covers one pass in a fixed order without sampling; combining it with subsampling or several passes needs other theorems (Altschuler & Talwar 2022). And it needs convexity: a non-contractive step can amplify a small gap faster than the noise can pay it off.

## Hidden state: privacy that stops growing {#sec-shuf-hidden}

**Idea.** Full-batch noisy GD uses every record at every step, so composition charges every step and the privacy loss grows linearly in $T$. If only the final model is released, two things can stop that growth. If the parameters live in a **bounded convex set**, the two runs can never be further apart than its diameter, so only the last stretch of steps needs to hide the gap. If the loss is **strongly convex**, each step is a strict contraction that shrinks the gap even when no noise pays for it. Either way the last iterate forgets the distant past.

**Result.** For a bounded domain, Altschuler & Talwar (2022) prove that the last iterate's RDP grows like composition up to a number of steps $\bar T$, proportional to the diameter, and then stays flat; their lower bound shows the shape is right. For a quadratic loss the answer is exact: with contraction factor $c < 1$ per step and per-step shift $s$ under noise $\sigma$, the last iterate tends to a Gaussian mechanism with
$$\mu_\infty = \frac{s}{\sigma}\sqrt{\frac{1 + c}{1 - c}},$$ {#eq-shuf-last-iterate}
reached after about $1/(1 - c)$ steps; for a gradient step of size $1/\beta$ on a $\lambda$-strongly convex, $\beta$-smooth loss, that is the condition number $\beta/\lambda$. With $c = 1$ there is no plateau: hiding the state buys nothing without contraction.

**Example.** Run 10,000 steps of full-batch noisy GD with $\sigma/s = 10$ on a quadratic with $c = 0.99$. Releasing every iterate is $\mu = 10$, which is $\varepsilon = 91.8173$ at $\delta = 10^{-5}$: no guarantee at all. Releasing only the final model is at the plateau, $\mu = 1.4107$, which is $\varepsilon = 6.5534$. The noise is the same; only what the adversary sees differs. On the unit interval with a linear loss (@fig-shuf-hidden, right), after 2000 steps composition charges an RDP of 20.0000 at $\alpha = 2$ and the bounded-domain bound 4.0000, while the hard instance actually reaches 0.3786: the bound's shape is right and its constant loose.

![Left: the exact $\mu$ of the last iterate of one-dimensional full-batch noisy GD against the number of steps $T$, for $c = 1$ (on top of composition), $c = 0.99$ and $c = 0.9$; with contraction it plateaus. Right: the Rényi divergence of order 2 of the last iterate for a linear loss on $[0, 1]$: composition, the bounded-domain bound, and the exact value for two projected random walks.](figures/shuffling_iteration_hidden_state.svg){#fig-shuf-hidden}

**Pitfalls.** All of this needs **only the last iterate released**. Checkpoints, logged intermediate models, or a model published after each epoch put you back to composition, and so does picking the best of several checkpoints. It also needs **contractive steps**, which require convexity. For non-convex losses, Annamalai (2024) built a loss for which the last iterate of DP-SGD leaks as much as all iterates together. Attacks that see only the final model have fallen short of the DP-SGD bound (Nasr et al. 2021), but that is an observation about attacks, not a guarantee. Do not report a hidden-state $\varepsilon$ for deep learning.

## Code {#sec-shuf-code}

This standard-library snippet builds @eq-shuf-rr-pair in log space, takes the larger hockey-stick divergence over both directions, and bisects for $\varepsilon$:

```python
from math import exp, lgamma, log


def log_binom_pmf(k: int, n: int, p: float) -> float:
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(p) + (n - k) * log(1 - p)


def shuffled_rr_pair(n: int, eps0: float) -> tuple[list[float], list[float]]:
    """Distribution of the number of 1-reports when you hold 0 (P) or 1 (Q); the n - 1 others hold 0."""
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

It reproduces the exact column of @tbl-shuf-compare. The full edition's script, [shuffling_iteration.py](../../code/part04-privacy-accounting/shuffling_iteration.py), also computes the numeric clone bound and every iteration result:

```bash
uv run python code/part04-privacy-accounting/shuffling_iteration.py
```

## Cheat sheet

| Result | Statement | Use |
| --- | --- | --- |
| Shuffled binary randomized response | Count pair @eq-shuf-rr-pair; $\mu \approx 2\sinh(\varepsilon_0/2)/\sqrt n$ | Compute exactly; the approximation fails with few 1-reports |
| Numeric clone bound | Exact $\delta(\varepsilon)$ of the clone count pair | Deploy with it for any $\varepsilon_0$-DP randomizer; 1.44–2.36 times the exact value for randomized response |
| Closed-form clone bound | @eq-shuf-clone-bound, needs $\varepsilon_0 \le \ln\frac{n}{16\ln(2/\delta)}$ | Scaling only; 3.10–4.75 times the numeric bound |
| One-pass noisy SGD, last iterate | $\varepsilon_t(\alpha) = \frac{\alpha}{2\tilde\sigma^2(n - t + 1)}$ | Convex, smooth losses; report the last record's value |
| Bounded domain or strong convexity | RDP of the last iterate plateaus; quadratic limit $\frac s\sigma\sqrt{\frac{1+c}{1-c}}$ | Convex training, last iterate only |

: The results of this chapter. All use replace-one neighbours. {#tbl-shuf-cheat}

**Rules of thumb**

- Shuffling turns $\varepsilon_0$-local reports into a central $\varepsilon$ that falls like $1/\sqrt n$, but never lowers the pure-DP level.
- For binary randomized response, compute the exact value; for other randomizers, the numeric clone bound.
- Shuffling reports is not shuffle-and-partition batching in DP-SGD: don't carry these results over.
- Amplification by iteration needs convex, smooth losses and a hidden state; the record used last gets none.
- Never claim hidden-state privacy for non-convex training or when checkpoints are released.

## Further reading

- Feldman, McMillan & Talwar, "Hiding Among the Clones: A Simple and Nearly Optimal Analysis of Privacy Amplification by Shuffling" (FOCS 2021): the clone reduction and bound, and the numerical method to compute it.
- Feldman, Mironov, Talwar & Thakurta, "Privacy Amplification by Iteration" (FOCS 2018): the shifted Rényi divergence and the one-pass noisy SGD guarantee.
- Altschuler & Talwar, "Privacy of Noisy Stochastic Gradient Descent: More Iterations without More Privacy Loss" (NeurIPS 2022): the bounded-domain plateau, with a matching lower bound.
- Annamalai, "It's Our Loss: No Privacy Amplification for Hidden State DP-SGD With Non-Convex Loss" (AISec 2024): why the hidden-state results do not extend to deep learning.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/07-shuffling-iteration.md).
