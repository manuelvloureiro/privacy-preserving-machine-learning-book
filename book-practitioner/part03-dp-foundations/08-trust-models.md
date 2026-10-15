# Local, shuffle and distributed trust models

## Why this matters

Every mechanism so far in this part assumed a **trusted curator** that holds the raw data and releases only a mechanism's output. The guarantee of differential privacy (DP) covers that output, not the curator's database. Often the curator is exactly the party users don't want to see their data: a browser vendor collecting usage statistics, a keyboard app learning new words.

So the practical question is: **if you can't trust a curator, what do you trust instead, and what does it cost in accuracy?** @fig-trust-models shows the four standard answers. The **central** model trusts the curator. The **local** model trusts nobody: each person randomizes their own record on their device. The **shuffle** model trusts a shuffler only to unlink reports from senders. The **distributed** model trusts cryptography and an honest majority of clients.

![The four trust models. Blue boxes hold one record each, green boxes must be trusted, and red boxes see what is released and are not trusted. Only the central model gives raw records to a single party.](figures/trust_models.svg){#fig-trust-models}

The one idea is *where the noise enters*. The further upstream it is added, the fewer parties you trust and the more noise reaches the answer. A local report must hide its owner from everyone, so each of the $n$ reports carries about as much noise as the central mechanism adds once. Independent noises add in variance, so the total noise grows like $\sqrt n$ while the count grows like $n$. @tbl-trust-decisions maps the decisions to the sections.

| Decision | Rule | Section |
| --- | --- | --- |
| Can you afford local DP at all? | Its error is about $\sqrt n$ times central; no protocol avoids this | @sec-trust-local |
| Which local frequency oracle? | $k$-ary randomized response below $k = 3e^{\varepsilon_0} + 2$ values, optimized unary encoding above | @sec-trust-frequency |
| How large a local $\varepsilon_0$ a shuffler allows | Invert the amplification bound for your $n$ and target $(\varepsilon, \delta)$ | @sec-trust-shuffle |
| How much noise each client adds under secure aggregation | Split the central Gaussian variance; over-provision for colluders and dropouts | @sec-trust-distributed |
| Which model to deploy | Compare errors at the same central guarantee; state the trust model next to $(\varepsilon, \delta)$ | @sec-trust-compare |

: The decisions behind a deployment without a fully trusted curator, and where this chapter makes each one. {#tbl-trust-decisions}

## Central and local differential privacy {#sec-trust-local}

**Idea.** In the local model each person runs a **local randomizer** $R$ on their record and sends only the result. The receiver may know whose report it is, so the report must look about the same whatever the record.

**Result.** $R$ is an **$\varepsilon_0$-DP local randomizer** if for every pair of records $x, x'$ and every set $S$ of reports,
$$\Pr[R(x) \in S] \le e^{\varepsilon_0}\Pr[R(x') \in S].$$ {#eq-trust-ldp}
Replacing one record changes only one report, so the collection is $\varepsilon_0$-DP under replace neighbours, against everyone, the collector included.

The basic randomizer is **randomized response**: report your true bit with probability $t = e^{\varepsilon_0}/(1 + e^{\varepsilon_0})$, the flipped bit otherwise ([The definition of differential privacy](01-definition.md)). Debiasing the mean report $\bar y$ as $\hat\pi = (\bar y - (1 - t))/(2t - 1)$ gives an unbiased estimate of the fraction of ones, with
$$\operatorname{sd}(\hat\pi) = \frac{e^{\varepsilon_0/2}}{(e^{\varepsilon_0} - 1)\sqrt n}.$$
No local protocol does better than order $1/((e^{\varepsilon_0} - 1)\sqrt n)$ (Duchi, Jordan & Wainwright 2018); the central Laplace mechanism reaches order $1/(n\varepsilon)$. @tbl-trust-who summarizes the four models.

| Model | Who must be trusted | What the untrusted party sees | Error of a count, $\varepsilon$ fixed |
| --- | --- | --- | --- |
| Central | The curator, with all raw records | $M(D)$ | Order $1/\varepsilon$, independent of $n$ |
| Local | Nobody | Every report $R(x_i)$ | Order $\sqrt n/\varepsilon$ |
| Shuffle | The shuffler, to unlink reports from senders | The multiset of reports | Order $\sqrt{\ln(1/\delta)}/\varepsilon$ |
| Distributed | The cryptography, and enough honest clients | The noisy sum | Order $\sqrt{\ln(1/\delta)}/\varepsilon$ |

: The four trust models: who is trusted, what the untrusted party sees, and the order of the error of a count. {#tbl-trust-who}

**Example.** Of $n = 10{,}000$ people, 3,000 hold a 1, and each uses randomized response at $\varepsilon = \ln 3$, so $t = 0.75$. The estimate's standard deviation is 0.00866, or 86.6 people on the count. A trusted curator adding $\mathrm{Lap}(1/\varepsilon)$ noise has standard deviation 1.287: the local survey is 67.3 times noisier at the same $\varepsilon$. The gap grows like $\sqrt n$ (@fig-trust-rr-gap). At $\varepsilon = 1$ it is 21 at $n = 1{,}000$, 215 at $n = 100{,}000$ and 2146 at $n = 10^7$. For a standard deviation of 0.001 on a fraction, randomized response needs 920,674 people and the central Laplace mechanism 1,415.

![Standard deviation of an estimated fraction at $\varepsilon = 1$, against the number of people. Randomized response (line and simulated dots) falls like $1/\sqrt n$ and stays a constant factor above the lower bound for any local protocol; the central Laplace mechanism falls like $1/n$.](figures/trust_rr_gap.svg){#fig-trust-rr-gap}

**Pitfalls.** Local DP is affordable only for frequent items in very large populations. The guarantee doesn't cover metadata: a network address stored with a report links it to the person. And $\varepsilon_0$ is per report: a user who reports daily pays $\varepsilon_0$ daily by composition ([Post-processing, group privacy and composition](02-properties.md)).

## Frequency estimation {#sec-trust-frequency}

**Idea.** In **frequency estimation** each person holds one of $k$ values and the analyst wants the fraction $f_v$ holding each value $v$. In every scheme in use, a **frequency oracle**, a report "supports" $v$ with probability $p$ if the person holds $v$ and a smaller $q$ otherwise; debiasing the count of supporting reports estimates $f_v$.

**Result.** With $C_v$ the number of reports supporting $v$, the unbiased estimate and its exact variance are
$$\hat f_v = \frac{C_v/n - q}{p - q}, \qquad \operatorname{Var}\hat f_v = \frac{f_vp(1-p) + (1 - f_v)q(1-q)}{n(p - q)^2}.$$ {#eq-trust-oracle}
**$k$-ary randomized response** reports the true value with probability $p = e^{\varepsilon_0}/(e^{\varepsilon_0} + k - 1)$ and each other value with probability $q = 1/(e^{\varepsilon_0} + k - 1)$. **Optimized unary encoding** sends a one-hot vector of $k$ bits, keeping a 1 with probability $p = \frac12$ and turning a 0 into a 1 with probability $q = 1/(e^{\varepsilon_0} + 1)$ (Wang et al. 2017). Utility is decided by the noise floor, the variance for a rare value ($f_v \to 0$):
$$k\text{-ary: } \frac{e^{\varepsilon_0} + k - 2}{n(e^{\varepsilon_0} - 1)^2}, \qquad \text{optimized unary: } \frac{4e^{\varepsilon_0}}{n(e^{\varepsilon_0} - 1)^2}.$$
The $k$-ary variance grows linearly in $k$; the unary one doesn't, at the price of a $k$-bit report. $k$-ary randomized response wins exactly when $k < 3e^{\varepsilon_0} + 2$ (@tbl-trust-oracles, @fig-trust-frequency-oracles).

| $k$ | $k$-ary randomized response | Symmetric unary | Optimized unary |
| --- | --- | --- | --- |
| 2 | 0.921 | 3.918 | 3.683 |
| 8 | 2.953 | 3.918 | 3.683 |
| 32 | 11.082 | 3.918 | 3.683 |
| 128 | 43.596 | 3.918 | 3.683 |

: $n$ times the variance of the estimated frequency of a rare value, at $\varepsilon_0 = 1$. Symmetric unary encoding is basic one-shot RAPPOR. $k$-ary randomized response wins below $k = 3e + 2 = 10.15$. {#tbl-trust-oracles}

![$n$ times the variance of a rare value's estimated frequency, against the domain size $k$, at $\varepsilon_0 = 1$ and $\varepsilon_0 = 3$. $k$-ary randomized response grows linearly in $k$; the unary encodings are flat; the dotted line is the crossover $k = 3e^{\varepsilon_0} + 2$.](figures/trust_frequency_oracles.svg){#fig-trust-frequency-oracles}

**Example.** A keyboard app wants the frequency of each of $k = 128$ emoji at $\varepsilon_0 = 1$ from $n = 10^6$ users. A rare emoji's standard deviation is the square root of the table entry over $n$: 0.0066 for $k$-ary randomized response, 0.0019 for optimized unary encoding. Emoji used by fewer than about 0.6% of users, three standard deviations, are lost either way. With $k = 8$, below the crossover, $k$-ary randomized response is better: 0.0017.

**Pitfalls.** These oracles keep the $\sqrt n$ gap. For large $k$, hashing schemes (local hashing, Hadamard response) reach the unary variance with reports of about $\log k$ bits. For values that don't change, memoize: RAPPOR (Erlingsson, Pihur & Korolova 2014) fixes one permanent randomized response per value and adds fresh noise around it in each report, so averaging recovers at most the permanent randomized value.

## The shuffle model {#sec-trust-shuffle}

**Idea.** The local model loses $\sqrt n$ because the analyst knows whose report is whose. In the **shuffle model** a **shuffler** strips identifiers and timing and forwards the locally randomized reports in random order. It is trusted only to unlink. Anonymity amplifies privacy: every other person, with probability $e^{-\varepsilon_0}$, sends a report that looks exactly like one the target might have sent, so the target hides among about $ne^{-\varepsilon_0}$ such clones ([Amplification by shuffling and by iteration](../part04-privacy-accounting/07-shuffling-iteration.md)).

**Result.** By Theorem 3.1 of Feldman, McMillan & Talwar (2021), shuffling $n$ reports from $\varepsilon_0$-DP local randomizers is $(\varepsilon, \delta)$-DP under replace neighbours with
$$\varepsilon \le \ln\left(1 + \frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1}\left(\frac{8\sqrt{e^{\varepsilon_0}\ln(4/\delta)}}{\sqrt n} + \frac{8e^{\varepsilon_0}}{n}\right)\right),$$ {#eq-trust-fmt}
provided $\varepsilon_0 \le \ln\big(n/(16\ln(2/\delta))\big)$, that is, enough expected clones. It falls like $1/\sqrt n$ (@fig-trust-shuffle-bound). Inverted, a target $\varepsilon$ allows $\varepsilon_0$ up to roughly $\ln(\varepsilon^2n/\ln(1/\delta))$, close to central accuracy. The closed form is loose: @tbl-trust-shuffle compares it with the paper's numerical bound, valid for every $\varepsilon_0$-DP randomizer, and with the exact $\varepsilon$ of shuffled binary randomized response, below which no general bound can go.

| $\varepsilon_0$ | Closed form, @eq-trust-fmt | Numeric (Theorem 3.2) | Exact, shuffled binary randomized response |
| --- | --- | --- | --- |
| 0.5 | 0.0939 | 0.0204 | 0.0162 |
| 1 | 0.2140 | 0.0530 | 0.0357 |
| 2 | 0.5009 | 0.1550 | 0.0870 |
| 3 | 0.8241 | 0.3207 | 0.1692 |

: Central $\varepsilon$ after shuffling the reports of $n = 10^4$ people, each $\varepsilon_0$-DP, at $\delta = 10^{-6}$. The condition of the theorem requires $\varepsilon_0 \le 3.763$ here. {#tbl-trust-shuffle}

![Central $\varepsilon$ at $\delta = 10^{-6}$ after shuffling, against $n$, for $\varepsilon_0 = 1$ and $\varepsilon_0 = 3$: the closed form of Feldman, McMillan & Talwar (lines, drawn where its condition holds), their numeric bound (dots) and the exact $\varepsilon$ of shuffled binary randomized response (crosses). The dotted lines mark $\varepsilon_0$, the guarantee without shuffling.](figures/trust_shuffle_bound.svg){#fig-trust-shuffle-bound}

**Example.** For central $\varepsilon = 1$ at $\delta = 10^{-6}$, bisection on @eq-trust-fmt (@sec-trust-code) allows $\varepsilon_0 = 3.504$ at $n = 10^4$, 5.701 at $n = 10^5$ and 7.992 at $n = 10^6$; at $n = 10^3$ the condition caps it at 1.460. At 7.992 a single report is almost the truth, yet the collection is $(1, 10^{-6})$-DP.

**Pitfalls.** The closed form is 2.6 to 4.6 times the numerical bound in @tbl-trust-shuffle: reason with it, deploy with a numerical accountant. If the shuffler colludes with the analyst, the guarantee falls back to the local $\varepsilon_0$, so keep $\varepsilon_0$ reasonable. If the adversary controls $m$ senders, use $n - m$ in place of $n$. And the theorem covers only the multiset: timing, message sizes and message counts must not leak, which is why real shufflers batch and pad.

## Distributed DP {#sec-trust-distributed}

**Idea.** The **distributed model** adds the central amount of noise once, in pieces. Each client adds a share, and a **secure aggregation** protocol (Bonawitz et al. 2017), built on additive secret sharing ([Cryptography basics](../part01-prerequisites/08-crypto-basics.md)), reveals only the sum to the server.

**Result.** If each of $n$ clients adds $\mathcal N(0, \sigma^2/n)$ noise, the sum carries $\mathcal N(0, \sigma^2)$, and the server's view is as private as the central Gaussian mechanism with noise $\sigma$ ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). If an adversary knows the noise of $t$ clients, because they collude or dropped out, it subtracts their shares, leaving a Gaussian mechanism with noise
$$\sigma\sqrt{\frac{n - t}{n}}.$$
To tolerate a fraction $\rho$ of such clients, have each add variance $\sigma^2/(n(1 - \rho))$. Secure aggregation works on integers modulo some $q$, so in practice clients round their values and add integer noise: **Skellam** shares, which sum exactly to Skellam noise close to a Gaussian (Agarwal, Kairouz & Liu 2021), or discrete Gaussian shares (Kairouz, Liu & Steinke 2021).

**Example.** A server aggregates a histogram through 100 clients at $(1, 10^{-6})$-DP. Under replace a changed value moves two counts by 1, so $\Delta_2 = \sqrt2$, and the analytic calibration gives total noise $\sigma = 5.975$, or $0.597$ per client. If 20% of clients collude with the server, the actual guarantee is $\varepsilon = 1.128$; at 50%, 1.455; with only 10% honest, 3.525 (@fig-trust-collusion).

![The actual $\varepsilon$ at $\delta = 10^{-6}$ of distributed Gaussian noise calibrated for a target $\varepsilon$, against the fraction of the total noise that the adversary does not know. At fraction 1 every client is honest and the target is met; below it, colluding or dropped clients leave less noise than planned.](figures/trust_collusion.svg){#fig-trust-collusion}

**Pitfalls.** Decide up front how many clients may collude or drop out, and over-provision. Laplace noise doesn't split this way: $n$ shares of $\mathrm{Lap}(b/n)$ sum to variance $2b^2/n$, not $2b^2$. Choose the modulus large enough that the true sum never wraps around. And secure aggregation hides inputs without checking them: a malicious client can send a vector with no noise, and bounding that needs zero-knowledge proofs or verifiable aggregation.

## Comparing the four models {#sec-trust-compare}

**Idea.** To choose, fix the central guarantee and compare errors. The full edition's script estimates an 8-value histogram at $(1, 10^{-6})$ in all four models, each with its best mechanism from this chapter.

**Result.** Squared error falls like $1/n^2$ for central and distributed, like $1/n$ for local, and shuffle bends from the local slope to the central one as $n$ grows (@tbl-trust-matched, @fig-trust-matched-error).

| $n$ | Central | Distributed | Shuffle | Local |
| --- | --- | --- | --- | --- |
| $10^3$ | $6.40\times10^{-5}$ | $2.86\times10^{-4}$ | $9.35\times10^{-3}$ | $2.71\times10^{-2}$ |
| $10^4$ | $6.40\times10^{-7}$ | $2.86\times10^{-6}$ | $4.88\times10^{-5}$ | $2.71\times10^{-3}$ |
| $10^5$ | $6.40\times10^{-9}$ | $2.86\times10^{-8}$ | $4.76\times10^{-7}$ | $2.71\times10^{-4}$ |

: Expected squared $\ell_2$ error of an 8-value histogram of fractions at the same central $(\varepsilon, \delta) = (1, 10^{-6})$ in all four trust models. {#tbl-trust-matched}

![Expected squared $\ell_2$ error of an 8-value histogram against the number of people, at the same central $(\varepsilon, \delta) = (1, 10^{-6})$ in all four trust models. Lines are exact; dots are simulations. Central and distributed fall like $1/n^2$, local like $1/n$; shuffle bends from the local to the central slope as $n$ grows.](figures/trust_matched_error.svg){#fig-trust-matched-error}

**Example.** At $n = 10^5$ the local error is about 42,000 times the central one. At $n = 10^4$ shuffling is about 56 times better than local. Distributed noise is about 4.5 times the central Laplace error, because Gaussian noise is wider than Laplace noise in low dimension.

**Pitfalls.** Every column has the same $(\varepsilon, \delta)$ against a different adversary: only the local model protects against the collector. A report of "$\varepsilon = 1$" without its trust model can't be compared with anything.

## Code {#sec-trust-code}

Two calculations recur in local and shuffled deployments: which frequency oracle to use, and how large a local $\varepsilon_0$ the shuffler allows.

```python
from math import exp, log, log1p, sqrt


def rare_variance_krr(eps0: float, k: int) -> float:
    """n times the variance of a rare value's frequency under k-ary randomized response."""
    return (exp(eps0) + k - 2) / (exp(eps0) - 1) ** 2


def rare_variance_oue(eps0: float) -> float:
    """n times the variance of a rare value's frequency under optimized unary encoding."""
    return 4 * exp(eps0) / (exp(eps0) - 1) ** 2


def shuffle_cap(n: int, delta: float) -> float:
    """The largest eps0 the shuffling theorem covers: ln(n / (16 ln(2/delta)))."""
    return log(n / (16 * log(2 / delta)))


def shuffle_epsilon(eps0: float, n: int, delta: float) -> float:
    """Central epsilon after shuffling n eps0-DP reports (Feldman, McMillan & Talwar 2021)."""
    if eps0 > shuffle_cap(n, delta):
        raise ValueError("eps0 is above the theorem's cap")
    e0 = exp(eps0)
    bracket = 8 * sqrt(e0 * log(4 / delta)) / sqrt(n) + 8 * e0 / n
    return log1p((e0 - 1) / (e0 + 1) * bracket)


def max_local_epsilon(eps: float, n: int, delta: float) -> float:
    """The largest eps0 whose shuffled reports are (eps, delta)-DP by the closed form."""
    lo, hi = 0.0, shuffle_cap(n, delta)
    if shuffle_epsilon(hi, n, delta) <= eps:
        return hi
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if shuffle_epsilon(mid, n, delta) <= eps else (lo, mid)
    return lo


print(f"{rare_variance_krr(1.0, 128):.3f} {rare_variance_oue(1.0):.3f}")  # 43.596 3.683
print(f"{shuffle_epsilon(1.0, 10_000, 1e-6):.4f}")  # 0.2140
print(" ".join(f"{max_local_epsilon(1.0, n, 1e-6):.3f}" for n in (10**3, 10**4, 10**5, 10**6)))  # 1.460 3.504 5.701 7.992
```

The full edition's script, [trust_models.py](../../code/part03-dp-foundations/trust_models.py), computes every number in this chapter, including the numerical shuffle bound:

```bash
uv run python code/part03-dp-foundations/trust_models.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| Local randomizer | $\Pr[R(x) \in S] \le e^{\varepsilon_0}\Pr[R(x') \in S]$ for all $x, x'$ | Private against everyone, the collector included; replace neighbours |
| Randomized response | $\operatorname{sd} = e^{\varepsilon_0/2}/((e^{\varepsilon_0} - 1)\sqrt n)$ | Optimal among local protocols up to a constant |
| Local versus central | Local error of order $1/((e^{\varepsilon_0} - 1)\sqrt n)$; central $\sqrt2/(n\varepsilon)$ | The gap grows like $\sqrt n$ |
| Frequency oracle | $\hat f_v = (C_v/n - q)/(p - q)$, variance @eq-trust-oracle | Unbiased, exact variance |
| $k$-ary randomized response | Rare variance $(e^{\varepsilon_0} + k - 2)/(n(e^{\varepsilon_0} - 1)^2)$ | Best for $k < 3e^{\varepsilon_0} + 2$ |
| Optimized unary encoding | Rare variance $4e^{\varepsilon_0}/(n(e^{\varepsilon_0} - 1)^2)$ | Independent of $k$; $k$-bit reports |
| Shuffling | @eq-trust-fmt, for $\varepsilon_0 \le \ln(n/(16\ln(2/\delta)))$ | 2.6–4.6 times loose at $n = 10^4$; deploy with a numerical bound |
| Distributed Gaussian | $n$ shares $\mathcal N(0, \sigma^2/n)$ sum to $\mathcal N(0, \sigma^2)$ | $t$ colluders leave $\sigma\sqrt{(n - t)/n}$ |

: The results of this chapter. {#tbl-trust-cheat}

**Rules of thumb**

- Local DP costs a factor of about $\sqrt n$ in error over central DP.
- Use $k$-ary randomized response for $k < 3e^{\varepsilon_0} + 2$ values and optimized unary encoding (or a hashing variant) above. Memoize reports of values that don't change.
- A shuffler lets each report use a large $\varepsilon_0$, roughly up to $\ln(\varepsilon^2n/\ln(1/\delta))$. Compute the central $\varepsilon$ numerically before deploying.
- With distributed noise, decide how many clients may collude or drop out, and over-provision for it.
- Always state the trust model next to $(\varepsilon, \delta)$.

## Further reading

- Wang, Blocki, Li & Jha, "Locally Differentially Private Protocols for Frequency Estimation" (USENIX Security 2017): frequency oracles, optimized unary encoding and local hashing, compared.
- Feldman, McMillan & Talwar, "Hiding Among the Clones" (FOCS 2021): the closed form used here and the numerical bound to deploy with.
- Kairouz, Liu & Steinke, "The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation" (ICML 2021): distributed noise end to end, with integer noise and modular arithmetic.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/08-trust-models.md).
