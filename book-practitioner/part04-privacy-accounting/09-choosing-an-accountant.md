# Choosing an accountant

## Why this matters

You trained a model with differential privacy (DP), using stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, called DP-SGD: each record joined each batch independently with probability $q = 0.01$ (Poisson sampling), gradients were clipped to norm $C$, and noise of standard deviation $\sigma C$ was added to their sum, with **noise multiplier** $\sigma = 1$, for $k = 10{,}000$ steps. Now you must write "the model is $(\varepsilon, 10^{-5})$-DP" in a paper or model card. The procedure that produces $\varepsilon$ is an **accountant**. On this one run, composing per-step guarantees gives 11296.93, Rényi differential privacy (RDP) gives 7.4287 or 6.7127 depending on the conversion, the Gaussian differential privacy (GDP) central limit theorem (CLT) gives 6.0071, and a privacy loss distribution (PLD) accountant gives the interval $[6.1377, 6.2377]$. Which do you report, and what else must you say?

The one idea: accountants make different kinds of statement. An **upper bound** has a proof that the run is $(\varepsilon, \delta)$-DP: perhaps loose, but true. An **approximation** has no proof and may land below the truth, making the claim false. A certified **lower bound** says the true $\varepsilon$ is at least this large: not a guarantee, but a referee for every other number. The rule follows: **report the tightest upper bound whose assumptions hold for the code you ran, with enough detail that someone else can recompute it.** @tbl-choose-overview maps the decisions to the sections.

| Decision | Rule | Section |
| --- | --- | --- |
| Which accountant for DP-SGD | Pessimistic PLD, with the optimistic value as error bar; never the CLT | @sec-choose-compare |
| How much noise for a target $\varepsilon$ | Calibrate $\sigma$ with the accountant you will report | @sec-choose-inverse |
| Which accountant for other mechanisms | The one whose summary describes that mechanism's privacy loss exactly | @sec-choose-decision |
| Does the number apply to my code? | Check sampling, neighbours, privacy unit, tuning, adaptivity | @sec-choose-match |
| What a privacy statement must contain | Setting, unit, relation, accountant, parameters, $\varepsilon$ at several $\delta$ | @sec-choose-report |
| How to check an accountant | Cases with known answers, and properties every correct one has | @sec-choose-check |

: The decisions behind reporting the privacy of a training run, and where this chapter makes each one. {#tbl-choose-overview}

## Every accountant on one run {#sec-choose-compare}

**Idea.** Every accountant starts from what one step reveals about one record. Along that record's clipped gradient, in units of $C$, the output is $\mathcal N(0, \sigma^2)$ without the record and the mixture $(1 - q)\mathcal N(0, \sigma^2) + q\mathcal N(1, \sigma^2)$ with it, since the record is sampled with probability $q$. Accountants differ in what they keep of this pair; the more they keep, the tighter they are.

**Result.** The part's accountants, grouped by what they keep:

1. **Per step, then composition** ([Amplification by subsampling](05-subsampling.md)): an amplified $(\varepsilon, \delta)$ pair per step, then basic or advanced composition. An upper bound.
2. **RDP with Mironov's or the improved conversion** ([Rényi DP](01-renyi-dp.md), [From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md)): $k$ times the exact RDP curve of the subsampled Gaussian, converted at the best order of a grid. Upper bounds; the improved one is always smaller.
3. **RDP with the optimal conversion**: the tightest upper bound obtainable from the curve alone.
4. **The GDP CLT** ([f-DP and Gaussian differential privacy](04-gaussian-dp.md)): treats the run as $\mu$-GDP with $\mu = q\sqrt{k(e^{1/\sigma^2} - 1)}$, a large-$k$ limit with no error bound in the tail, where $\delta$ is read. Not a bound.
5. **PLD, pessimistic and optimistic** ([Numerical accounting with privacy loss distributions](06-numerical-accounting.md)): the whole distribution of the privacy loss on a grid of width $h$, composed numerically, losses rounded up (upper bound) or down (lower bound), worse of the add and remove directions. The two are at most $kh$ apart, tail truncation aside.

Every correct implementation satisfies two chains: RDP Mironov $\ge$ RDP improved $\ge$ RDP optimal $\ge$ true $\varepsilon$, and PLD upper $\ge$ true $\varepsilon \ge$ PLD lower. Theory does not order the chains against each other. @tbl-choose-main gives every value on the running example.

| $k$ | Basic | Advanced | RDP, Mironov | RDP, improved | RDP, optimal | GDP CLT | PLD upper | PLD lower |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.20 | 1.13 | 1.3175 | 0.9555 | 0.4899 | 0.0368 | 0.1995 | 0.1994 |
| 10 | 3.64 | 7.88 | 1.4145 | 1.0353 | 0.7951 | 0.1304 | 0.3800 | 0.3799 |
| 100 | 58.01 | 78.10 | 1.6118 | 1.2141 | 1.1538 | 0.4575 | 0.7185 | 0.7175 |
| 1,000 | 839.39 | 1258.64 | 2.5380 | 2.1014 | 2.1009 | 1.6177 | 1.8332 | 1.8232 |
| 10,000 | 11296.93 | 24379.55 | 7.4287 | 6.7127 | 6.7127 | 6.0071 | 6.2377 | 6.1377 |

: DP-SGD with $q = 0.01$, $\sigma = 1$: $\varepsilon$ at $\delta = 10^{-5}$ after $k$ steps under every accountant. The PLD columns use the grid $h = 10^{-5}$. {#tbl-choose-main}

![Left: $\varepsilon$ at $\delta = 10^{-5}$ against the number of steps for DP-SGD with $q = 0.01$, $\sigma = 1$, under every accountant (the two PLD curves overlap). Right: each divided by the PLD upper bound. Ratios above 1 are valid but loose; the shaded region below 1 is where only the PLD lower bound may lie, and the CLT lies there at every $k$.](figures/choosing_an_accountant_eps_vs_k.svg){#fig-choose-eps-k}

**Example.** Suppose you ran 1,000 steps. Depending on the accountant, your paper says $\varepsilon = 2.54$ (RDP with Mironov's conversion, as in older code), 2.10 (RDP with the improved conversion), 1.83 (PLD) or 1.62 (CLT). The first three are true, and 1.83 is the most informative. The fourth is false: the optimistic PLD proves $\varepsilon \ge 1.8232$, so the run is not $(1.62, 10^{-5})$-DP.

**Pitfalls.** Per-step composition is useless beyond a handful of steps: over a thousand times too large at $k = 10{,}000$, and advanced composition is even worse than basic here. RDP's excess has two sources. In short runs it is mostly the conversion's fixed toll $\ln(1/\delta)/(\alpha - 1)$: at $k = 1$ the improved conversion overstates the PLD by 379.1%. In long runs it is the curve, which keeps only moments of the loss: 7.6% at $k = 10{,}000$, which only a numerical accountant removes. The CLT lies below the certified lower bound at every $k$ in the table, so never report it as a guarantee. The PLD bracket is $kh = 0.1$ wide at $k = 10{,}000$; if that is too wide, refine $h$.

## Calibrating the noise for a target ε {#sec-choose-inverse}

**Idea.** In practice you fix the budget, the steps and the batch size, and ask how much noise to add. A loose accountant overstates $\varepsilon$ at every $\sigma$, so its slack becomes extra noise: the averaged gradient's noise has standard deviation $\sigma C/(qn)$ for $n$ records, so every percent of $\sigma$ is a percent of gradient noise.

**Result.** For an accountant $A$ reporting $\varepsilon_A(\sigma)$, the noise it requires for a target $\varepsilon^*$ is
$$\sigma_A(\varepsilon^*) = \min\{\sigma > 0 : \varepsilon_A(\sigma) \le \varepsilon^*\}.$$ {#eq-choose-inverse}
More noise never raises a correct accountant's $\varepsilon$, so this is the root of a decreasing function, found by bisection. Return the end of the final bracket where $\varepsilon_A \le \varepsilon^*$ was checked, never the midpoint. @tbl-choose-sigma gives the answers for the running example.

| Target $\varepsilon$ | Advanced | RDP, Mironov | RDP, improved | GDP CLT | PLD upper | PLD lower |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 24.682 | 4.974 | 4.126 | 3.796 | 3.988 | 3.654 |
| 3 | 10.505 | 1.854 | 1.662 | 1.549 | 1.584 | 1.547 |
| 8 | 5.711 | 0.962 | 0.917 | 0.868 | 0.885 | 0.880 |

: The noise multiplier each accountant requires for a target $\varepsilon$ at $\delta = 10^{-5}$, with $k = 10{,}000$ steps at $q = 0.01$. {#tbl-choose-sigma}

**Example.** With $n = 50{,}000$ records, the expected batch holds 500. At $\varepsilon = 8$ the RDP accountant (improved) sets $\sigma = 0.917$ and the PLD accountant $\sigma = 0.885$: 3.6% more noise for the same certified budget. Across the targets, advanced composition asks for 6.19 to 6.63 times the PLD's noise, Mironov's conversion 8.6 to 24.7% more, and the improved conversion 3.4 to 4.9% more.

![Each accountant's noise multiplier for a target $\varepsilon$ ($k = 10{,}000$, $q = 0.01$, $\delta = 10^{-5}$), divided by the one the PLD upper bound requires. The CLT and the PLD lower bound fall in the shaded region: less noise than any certified accountant can justify.](figures/choosing_an_accountant_noise.svg){#fig-choose-noise}

**Pitfalls.** Calibrate with the accountant you will report. The CLT asks for 2.0 to 4.8% less noise than the certified bound needs, and at $\varepsilon = 8$ that is provably too little: at its $\sigma = 0.8676$ the optimistic PLD already gives 8.2594. At small targets the PLD's fixed width $kh$ is a large fraction of $\varepsilon$, so at $\varepsilon = 1$ it brackets $\sigma$ only to $[3.654, 3.988]$; refine $h$.

## Where each accountant is tight {#sec-choose-decision}

**Idea.** An accountant is tight when its summary describes the mechanism's privacy loss exactly, so the ranking changes with the mechanism.

**Result.** @tbl-choose-decision is the decision table.

| Mechanism | Use | Avoid |
| --- | --- | --- |
| Gaussian, no subsampling | GDP, or zero-concentrated differential privacy (zCDP), converted exactly: $\mu^2$ adds and the profile is exact | Per-step $(\varepsilon, \delta)$ composition |
| Poisson-subsampled Gaussian (DP-SGD) | Pessimistic PLD, optimistic value as error bar | zCDP, which cannot express amplification; the CLT as a guarantee |
| Pure-DP steps (randomized response, Laplace, exponential mechanism) | PLD (exact for finite outputs), or RDP with the mechanism's own curve | Advanced composition when a tighter accountant is available |
| Adaptive noise, stopping or per-record budgets | RDP filter at a fixed order ([Adaptive composition](08-adaptive-composition.md)) | Plugging realized $\varepsilon_t$ into a fixed-parameter theorem |
| Local randomizers behind a shuffler | Exact computation or a numeric bound ([Amplification by shuffling and by iteration](07-shuffling-iteration.md)) | The Poisson-subsampling accountant |
| Only the last iterate released | Amplification by iteration, for convex smooth losses with no released checkpoints | Any hidden-state claim for non-convex training |

: Which accountant to use for which mechanism. {#tbl-choose-decision}

**Example.** A team releases 100 noisy counts with the Gaussian mechanism at $\sigma = 10$ and reports $\varepsilon = 5.2985$ from an old RDP accountant with Mironov's conversion. The run is exactly $\mu$-GDP with $\mu = 1$, so its exact $\varepsilon$ is 4.3772; the improved RDP conversion gives 4.7285. The team overstated $\varepsilon$ by 21.0%. For 100 steps of randomized response at $\varepsilon_0 = 0.1$, the exact PLD gives 4.3068, RDP 4.6154, zCDP 4.7263, advanced composition 5.8502 and basic composition 10.0000.

**Pitfalls.** zCDP is exact for the plain Gaussian but cannot express the gain from subsampling, so it is the wrong tool for DP-SGD. The PLD accountant assumes the per-step pair is fixed in advance; with adaptive choices use a filter.

## Matching the accountant to the code {#sec-choose-match}

**Idea.** The accountant takes $q$, $\sigma$ and $k$ and trusts that the code does what its theorem assumes. A mismatch makes the number a statement about a different algorithm. Ponomareva et al. (2023) note that many papers report the Poisson-sampling number while training with shuffled fixed-size batches.

**Result.** Before reporting $\varepsilon$, check @tbl-choose-checklist against the code.

| Item | The accountant assumes | Check in the code | If it does not match |
| --- | --- | --- | --- |
| Sampling scheme | Poisson sampling at rate $q$ | Each record included independently, variable batch sizes? | Fixed-size random batches: account with noise multiplier $\sigma/2$ (as Poisson under add/remove, Lebeda et al. 2024). Shuffle and partition: report the guarantee without amplification, for zero-out neighbours |
| Neighbouring relation | Add/remove one record | Add/remove, replace-one or zero-out? | Replace-one costs about twice as much per step for small $\varepsilon$ |
| Privacy unit | One example | Can one person contribute several examples? | Clip and sample per user, or pay group privacy |
| Released outputs | Every noisy gradient | Checkpoints, logs? | Nothing: the guarantee already covers them |
| Hyperparameter tuning | Not counted | Was anything chosen from results on the private data? | Account for the tuning runs, or tune on public data |
| Adaptive parameters | $\sigma$, $q$, $k$ fixed in advance | Early stopping, noise schedules, per-record budgets? | Use a filter |
| Implementation | Exact Gaussian noise, exact clipping | Floating-point noise, accumulation across devices | Outside the accountant: verify the implementation |

: A checklist for matching the accountant to the training code. {#tbl-choose-checklist}

**Example.** A script uses a data loader with `shuffle=True` and `batch_size=500` on 50,000 records for 100 epochs, and calls an RDP accountant with `sample_rate=0.01`. That sampler is shuffle and partition, which has no amplification theorem, so the reported 6.71 does not apply. What holds is 100 unamplified Gaussian steps, one per epoch, for zero-out neighbours (one record's gradient replaced by zero): $\mu = 10$ GDP, $\varepsilon = 91.8173$; under replace-one, $\varepsilon = 284.39$. Fixed-size random batches, accounted correctly, give 47.42 (RDP) and 43.42 (PLD) instead of 6.71 and 6.24. Switching the loader to Poisson sampling, a few lines of code, makes 6.24 honest. Separately, tuning 10 configurations on the private data and releasing the best costs all 10 runs composed: $\varepsilon = 27.23$ with RDP.

**Pitfalls.** The sampling scheme is the largest single risk: a factor of 6.96 in $\varepsilon$ from fixed-size batches, against at most 1.19 between the RDP and PLD accountants at the same 10,000 steps. Example-level DP protects each example, not each user. And floating-point noise sampling can break a mechanism that is correct on paper ([Discrete and floating-point-safe mechanisms](../part03-dp-foundations/07-discrete-mechanisms.md)).

## What to report {#sec-choose-report}

**Idea.** An $\varepsilon$ without its assumptions cannot be checked or compared. NIST SP 800-226 (2025) and Ponomareva et al. (2023) say what a privacy statement should contain.

**Result.** Report the setting (central or local, and who is trusted); what is covered (one run or tuning too), the unit of privacy and the neighbouring relation; the accountant, its assumptions and whether they hold; and code where possible. For DP-SGD add what recomputing $\varepsilon$ needs: the sampling scheme, $q$ (or batch size and $n$), $k$ (or epochs), $\sigma$, $C$, and the accountant's version and discretization. Choose $\delta \ll 1/n$ for $n$ units of privacy ([The definition of differential privacy](../part03-dp-foundations/01-definition.md)): Ponomareva et al. suggest $n^{-1.1}$, and NIST gives $1/n^2$ as a common recommendation and calls values above $10^{-5}$ suspicious. Report $\varepsilon$ at several $\delta$, since a single pair cannot be converted to a smaller $\delta$, and give the RDP, zCDP or GDP parameters behind any conversion.

**Example.** For $n = 50{,}000$, $1/n = 2.0\times10^{-5}$, so $\delta = 10^{-5}$ is only half of $1/n$. @tbl-choose-profile gives the privacy profile from one composed PLD. The RDP accountant's original parameter is its curve: at the best order 4.1, $\varepsilon(4.1) = 3.7336$, which the improved conversion turns into 6.7127.

| $\delta$ | PLD upper | PLD lower | RDP, improved |
| --- | --- | --- | --- |
| $2.0\times10^{-5}$ ($1/n$) | 6.0068 | 5.9068 | 6.4884 |
| $1.0\times10^{-5}$ | 6.2377 | 6.1377 | 6.7127 |
| $6.8\times10^{-6}$ ($n^{-1.1}$) | 6.3641 | 6.2641 | 6.8353 |
| $1.0\times10^{-6}$ | 6.9574 | 6.8574 | 7.4140 |
| $4.0\times10^{-10}$ ($1/n^2$) | 9.0502 | 8.9378 | 9.4540 |

: The running example's privacy profile: $\varepsilon$ at several $\delta$ for $k = 10{,}000$, $q = 0.01$, $\sigma = 1$, $n = 50{,}000$. {#tbl-choose-profile}

A filled-in statement:

> **Privacy guarantee.** The model was trained with DP-SGD in the central model: the training server is trusted to run the code below, and the guarantee covers every intermediate noisy gradient, hence every checkpoint and the final model. **Unit of privacy:** one training example; each person contributes at most one example. **Neighbouring relation:** add or remove one example. **Training:** Poisson sampling with rate $q = 0.01$ ($n = 50{,}000$ examples, expected batch size 500), $k = 10{,}000$ steps (100 epochs), per-example clipping norm $C = 1$, Gaussian noise multiplier $\sigma = 1$. **Accountant:** privacy loss distributions, pessimistic discretization on the grid $h = 10^{-5}$, both add and remove directions. **Guarantee:** $(6.24, 10^{-5})$-DP and $(9.06, 4.0\times10^{-10})$-DP. The optimistic discretization gives 6.1377 and 8.9378, so the accountant's own error is at most 0.1000 and 0.1124. For comparison, the RDP accountant with the improved conversion gives 6.7127 at $\delta = 10^{-5}$. **Not covered:** hyperparameters were tuned on a public proxy dataset; no tuning on the private data was performed. **If the sampling assumption failed** (for example, shuffled fixed-size batches), the guarantee without amplification is $(91.82, 10^{-5})$-DP for zero-out neighbours.

**Pitfalls.** Round upper bounds up: 9.0502 becomes 9.06, because 9.05 would be a claim the accountant has not certified. An RDP number has no matching lower bound, so it carries no error bar; the PLD does. If any checklist item is in doubt, state the fallback guarantee that does not rely on it, as the statement does for sampling.

## Checking an accountant {#sec-choose-check}

**Idea.** An accountant is code, and code has bugs. Before trusting any accountant's number, run checks whose answers you know: cases where the truth is exact, and properties every correct accountant has.

**Result.** Four checks. (1) With $q = 1$, $k$ steps are exactly $\sqrt k/\sigma$-GDP, so every upper bound must lie above that closed form. (2) The PLD bracket for the plain Gaussian must contain the same closed form. (3) Both chains of @sec-choose-compare must hold at every $k$, and no reported upper bound may fall below the optimistic PLD. (4) The composed pessimistic PLD, evaluated at its own $\varepsilon$, must give back its $\delta$, and $\varepsilon$ must never decrease with $k$.

**Example.** For $\sigma = 1$, $k = 10$, $q = 1$, the exact $\varepsilon$ is 17.8566. The RDP accountant gives 19.0536, above it as it must be. The CLT formula gives $\mu = 4.1452$ instead of 3.1623 and $\varepsilon = 25.5853$: it fails, because it assumes small $q$. On the plain Gaussian with $\mu = 1$ and $k = 10$ at $h = 10^{-3}$, the PLD bracket $[17.8516, 17.8616]$ contains 17.8566.

**Pitfalls.** Compare against an independent library: Google's `dp_accounting` (RDP and PLD), Opacus (RDP, GDP, and a privacy loss random variable (PRV) accountant, its default), and Microsoft's `prv_accountant`. The full edition's numbers agree with Opacus 1.6.0, `dp_accounting` 0.6.0 and `prv_accountant` 0.2.0 to the printed precision. Two RDP accountants agree when their orders and conversion agree; if they differ by about $\frac{\ln\alpha}{\alpha - 1} + \ln\frac{\alpha}{\alpha - 1}$ (0.3612 at order 10), one uses Mironov's conversion; on the running example that is the difference between 7.43 and 6.71. Fractional orders are a weak spot: `dp_accounting` 0.6.0 returns a loose bound there (49.43 instead of 47.42 for the fixed-batch case), so compare at integer orders. PLD and PRV brackets from different libraries must overlap; if they do not, one has a bug.

## Code {#sec-choose-code}

The snippet is a small RDP accountant for DP-SGD (exact subsampled-Gaussian RDP at integer orders, improved conversion), the GDP closed form used by the checks, and the noise calibration of @eq-choose-inverse.

```python
from math import e, erfc, exp, lgamma, log, sqrt


def normal_cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def gdp_epsilon(mu: float, delta: float) -> float:
    """Exact epsilon at delta of a mu-GDP mechanism, by bisection on its delta(eps) curve."""

    def gdp_delta(eps: float) -> float:
        return normal_cdf(mu / 2 - eps / mu) - exp(eps) * normal_cdf(-mu / 2 - eps / mu)

    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gdp_delta(mid) > delta else (lo, mid)
    return hi


def subsampled_gaussian_rdp(alpha: int, q: float, sigma: float) -> float:
    """Exact RDP at an integer order of one Poisson-subsampled Gaussian step."""
    if q == 1.0:
        return alpha / (2 * sigma**2)
    terms = [  # log of each term of the binomial sum, combined with log-sum-exp
        lgamma(alpha + 1) - lgamma(j + 1) - lgamma(alpha - j + 1)
        + (alpha - j) * log(1 - q) + j * log(q) + (j * j - j) / (2 * sigma**2)
        for j in range(alpha + 1)
    ]
    top = max(terms)
    return (top + log(sum(exp(t - top) for t in terms))) / (alpha - 1)


def rdp_epsilon(q: float, sigma: float, steps: int, delta: float) -> float:
    """Upper bound on DP-SGD's epsilon: RDP on integer orders 2..64, improved conversion."""
    return min(
        steps * subsampled_gaussian_rdp(a, q, sigma) + log((a - 1) / a) - (log(delta) + log(a)) / (a - 1)
        for a in range(2, 65)
    )


def sigma_for_target(eps_target: float, q: float, steps: int, delta: float) -> float:
    """Smallest noise multiplier meeting the target, by bisection; returns the end that was checked."""
    lo, hi = 0.3, 50.0
    while hi - lo > 1e-4:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if rdp_epsilon(q, mid, steps, delta) <= eps_target else (mid, hi)
    return hi


q, sigma, steps, delta = 0.01, 1.0, 10_000, 1e-5
print(f"{rdp_epsilon(q, sigma, steps, delta):.4f}")  # 6.7194
print(f"{gdp_epsilon(q * sqrt(steps * (e - 1)), delta):.4f}")  # 6.0071
print(f"{sigma_for_target(3.0, q, steps, delta):.3f}")  # 1.665
print(f"{gdp_epsilon(sqrt(10) / sigma, delta):.4f}")  # 17.8566
print(f"{rdp_epsilon(1.0, sigma, 10, delta):.4f}")  # 19.8017
print(f"{gdp_epsilon(sqrt(10 * (e - 1)), delta):.4f}")  # 25.5853
print(f"{gdp_epsilon(sqrt(100) / sigma, delta):.4f}")  # 91.8173
```

The first three lines are the running example: the RDP bound, the CLT value (below the PLD lower bound, so not a guarantee) and the noise for $\varepsilon = 3$. The rest are checks: the exact $q = 1$ value, the RDP bound above it, the CLT failing it, and the shuffle-and-partition fallback. Integer orders give 6.7194, 1.665 and 19.8017 where the full edition's fractional orders give 6.7127, 1.662 and 19.0536: a coarser grid costs tightness, never validity. The full edition's script, [choosing_an_accountant.py](../../code/part04-privacy-accounting/choosing_an_accountant.py), runs every accountant, PLD included, in about two and a half minutes:

```bash
uv run python code/part04-privacy-accounting/choosing_an_accountant.py
```

## Cheat sheet

| Accountant | Kind | $\varepsilon$ at $k = 10{,}000$ | $\sigma$ for $\varepsilon = 3$ | When to use it |
| --- | --- | --- | --- | --- |
| Per step + advanced composition | Upper bound | 24379.55 | 10.505 | Never for DP-SGD |
| RDP, Mironov conversion | Upper bound | 7.4287 | 1.854 | Never: the improved conversion is free |
| RDP, improved conversion | Upper bound | 6.7127 | 1.662 | Quick bounds, adaptive filters, many heterogeneous mechanisms |
| PLD, pessimistic | Upper bound, within $kh$ of the truth | 6.2377 | 1.584 | The number to report |
| PLD, optimistic | Lower bound | 6.1377 | 1.547 | The error bar, and the referee for other accountants |
| GDP CLT | Approximation | 6.0071 | 1.549 | Intuition only; under-reports here |

: The accountants on the running example ($q = 0.01$, $\sigma = 1$, $\delta = 10^{-5}$), what kind of statement each makes, and when to use it. {#tbl-choose-cheat}

**Rules of thumb**

- Report the pessimistic PLD with the optimistic value beside it. RDP with the improved conversion is fine for adaptivity or a quick bound; expect it to overstate $\varepsilon$ by 5 to 10% for long runs and far more for short ones.
- Never report a CLT number as a guarantee. Check every reported number against the PLD lower bound.
- Calibrate $\sigma$ with the accountant you will report.
- Check the sampling scheme first: fixed-size or shuffled batches accounted as Poisson is the most expensive error in practice.
- Report the privacy unit, the neighbouring relation, $q$, $k$, $\sigma$, $C$, the accountant and its discretization, $\varepsilon$ at several $\delta$ including one well below $1/n$, and what was not accounted for.

## Further reading

- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023): matching accounting to the data pipeline, setting $\delta$, and what to report.
- Near, Darais, Lefkovitz & Howarth, "Guidelines for Evaluating Differential Privacy Guarantees" (NIST SP 800-226, 2025): what an evaluator will ask of your claim.
- Lebeda, Regehr, Kamath & Steinke, "Avoiding Pitfalls for Privacy Accounting of Subsampled Mechanisms under Composition" (IEEE SaTML 2025): fixed-size batches versus Poisson sampling.
- Gopi, Lee & Wutschitz, "Numerical Composition of Differential Privacy" (NeurIPS 2021): the PRV accountant with certified error bounds, and where the CLT under-reports.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/09-choosing-an-accountant.md).
