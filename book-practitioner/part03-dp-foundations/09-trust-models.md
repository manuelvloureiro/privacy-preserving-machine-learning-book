# Local, shuffle and distributed trust models

## Why this matters

Every mechanism so far in this part assumed a **trusted curator**, which holds the raw data and releases only the mechanism's output. In this **central model**, the differential privacy (DP) guarantee covers the output, not the curator's database, its engineers or a court order for the data. Often the curator cannot be trusted that far, as when a browser vendor collects usage statistics from its own users.

The other three models trust less. In the **local model** nobody is trusted, and each person randomizes their own record before it leaves their device. In the **shuffle model** a shuffler is trusted only to cut the link between each report and its sender. In the **distributed model** cryptography and enough honest participants are trusted. The further upstream the noise enters, the fewer parties you trust and the more noise reaches the answer.

## Choosing a trust model {#sec-trust-compare}

Start from whom you cannot trust, then pay what that model costs. At a fixed privacy level, the error of a count stays constant in the central model, but in the local model it grows with the square root of the number of people. For a small histogram at the same guarantee, the full edition finds that at a hundred thousand people the local model's squared error is about 42,000 times the central one. Whatever you choose, state the trust model next to the privacy parameters, because the same numbers protect against different adversaries.

## Central and local differential privacy {#sec-trust-local}

In the local model, each person sends only a **report**, the output of a **local randomizer** $R$ run on their own record. Write $\varepsilon_0$ for the privacy level of one report. The randomizer is $\varepsilon_0$-DP when every set of reports $S$ is at most $e^{\varepsilon_0}$ times as likely under one record as under any other:
$$\Pr[R(x) \in S] \le e^{\varepsilon_0}\Pr[R(x') \in S].$$ {#eq-trust-ldp}
With independent randomness per person, the whole collection is then $\varepsilon_0$-DP in the central sense. The guarantee covers only the randomized value, so a network address stored next to a report links it back to the person.

The basic local randomizer is **randomized response** on one bit. Each person reports their true bit with probability $e^{\varepsilon_0}/(1 + e^{\varepsilon_0})$, and the flipped bit otherwise. The analyst undoes the known flipping to get an unbiased estimate $\hat\pi$ of the fraction of ones, whose standard deviation for $n$ people is
$$\operatorname{sd}(\hat\pi) = \frac{e^{\varepsilon_0/2}}{(e^{\varepsilon_0} - 1)\sqrt n}.$$
For example, survey ten thousand people at the level $\ln 3$. The estimated count then has a standard deviation of 86.6 people, while a trusted curator adding Laplace noise would be off by 1.287. So the local survey is 67.3 times noisier.

That factor is the model's, not the protocol's. No local protocol can estimate a fraction with error below order $1/((e^{\varepsilon_0} - 1)\sqrt n)$, and randomized response is optimal up to a constant. That is why local DP is used for the most frequent items in large populations, as in Apple's, Google's and Microsoft's telemetry, and almost never for finer ones. On a small user base its estimates are mostly noise. OpenDP provides randomized response as `dp.m.make_randomized_response_bool` for one bit and `dp.m.make_randomized_response` for a list of values.

## Frequency estimation: $k$-ary randomized response and unary encoding {#sec-trust-frequency}

The workhorse of deployed local DP is **frequency estimation**. Each person holds one of a known list of values, such as emoji, and the analyst wants the fraction holding each one. Write $k$ for the number of values.

In a **frequency oracle**, each report supports a value with a probability that is $p$ if the person holds it and a smaller $q$ otherwise. Count the reports $C_v$ that support a value $v$. The debiased estimate of its fraction is then
$$\hat f_v = \frac{C_v/n - q}{p - q}.$$
There are two standard designs, and the pure-ldp Python package implements both. The first, **$k$-ary randomized response**, reports the true value with probability $e^{\varepsilon_0}/(e^{\varepsilon_0} + k - 1)$, and any other value otherwise. The second, **optimized unary encoding** (Wang et al. 2017), writes the value as $k$ bits with a single 1. It keeps that 1 with probability one half, and it turns each 0 into a 1 with probability $1/(e^{\varepsilon_0} + 1)$.

Rare values decide between them, because their variance is the noise floor below which nothing can be seen. Times $n$, the two floors are
$$k\text{-ary: } \frac{e^{\varepsilon_0} + k - 2}{(e^{\varepsilon_0} - 1)^2}, \qquad \text{optimized unary: } \frac{4e^{\varepsilon_0}}{(e^{\varepsilon_0} - 1)^2}.$$
So $k$-ary randomized response wins exactly when $k < 3e^{\varepsilon_0} + 2$. Unary encoding's floor does not depend on $k$, and its price is a report of $k$ bits. To size $n$, note that a value is visible only if its frequency is about three times the square root of the floor divided by $n$.

For example, a keyboard app wants the frequency of each of 128 emoji from a million users at $\varepsilon_0 = 1$. A rare emoji's estimate has a standard deviation of 0.0019 under optimized unary encoding, against 0.0066 under $k$-ary randomized response. An estimate must stand about three standard deviations above zero to be seen, so even the better design loses every emoji used by fewer than about 0.6% of users.

## Large and unknown domains: hashing and heavy hitters {#sec-trust-heavy}

Unary reports need one bit per possible value, which **local hashing** avoids. Each person draws a hash function from a large public family, maps their value into one of $w$ buckets, and sends the hash index with a $w$-ary randomized response of their bucket. Choose $w$ from the privacy level, because near $w = e^{\varepsilon_0} + 1$ the variance equals that of optimized unary encoding, with reports of a few bits.

An oracle also needs the analyst to name each value in advance. Finding frequent values with no dictionary is the **heavy hitters** problem, and the standard answer grows prefixes (Bassily et al. 2017). Split the people into one group per symbol position, and let group $\ell$ report the first $\ell$ symbols of its value by local hashing. The analyst keeps the short prefixes whose estimates clear a threshold, extends each survivor by one symbol, and tests those on the next group. Each threshold comes from the normal tail of an absent value's estimate, so that a level raises few false alarms.

Deployed systems use these pieces. Google's RAPPOR, in Chrome, sends a unary encoding of a hashed value and tests candidates that the analyst supplies (Erlingsson, Pihur & Korolova 2014). Apple's count-mean sketch hashes each value and sends a unary encoding of its bucket.

## Numbers in one bit: local mean estimation {#sec-trust-mean}

Telemetry also collects numbers, such as minutes in an app, each between 0 and a known bound $m$. Ding, Kulkarni & Yekhanin (2017) showed that one bit per person is enough. A person with value $x$ sends a bit that is 1 with a probability rising in a straight line, from the lie probability of randomized response at 0 to its truth probability at $m$:
$$\Pr[Y = 1] = \frac{1}{e^{\varepsilon_0} + 1} + \frac{x}{m}\cdot\frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1}.$$ {#eq-trust-onebit}
The analyst undoes the line to get an unbiased mean. For a million users of an app at $\varepsilon_0 = 1$, the one-bit estimate has a standard deviation of 1.420 minutes. Fix the bound before collection and clip to it on the device, because a bound taken from the data would let one record change everyone's randomizer.

## Collecting every day: memoization {#sec-trust-repeated}

Telemetry asks the same question every day, and fresh noise averages away. By basic [composition](02-properties.md), each report of the same value adds its $\varepsilon_0$ to the cost. After a hundred daily reports of one bit at the level $\ln 3$, the bit is no longer in doubt.

The fix is **memoization**. The device draws one permanent randomized response per value and reuses it, so all reports of an unchanged value cost only that one response's level $\varepsilon_\infty$. RAPPOR's homepage collection has $\varepsilon_\infty = 2.043$. Memoization fails when the value changes, since each new value costs another $\varepsilon_\infty$. A user who changes their homepage every month reaches $\varepsilon = 24.5$ in a year.

For numbers that drift a little, Microsoft's Windows 10 telemetry uses **$\alpha$-point rounding**, named after a secret random threshold $\alpha$. A report then changes only when the value crosses that threshold. State such a guarantee as one about changes in behaviour, not as a per-round $\varepsilon$ times the number of rounds.

## The shuffle model and amplification by shuffling {#sec-trust-shuffle}

In the **shuffle model** (Cheu et al. 2019), each person sends a local report to a **shuffler**. The shuffler strips identifiers and timing and forwards the reports in random order, so each report hides among the others ([Amplification by shuffling and by iteration](../part04-privacy-accounting/07-shuffling-iteration.md)). Feldman, McMillan & Talwar (2021) give the bound to compute with. It needs enough people to hide among, which means
$$\varepsilon_0 \le \ln\frac{n}{16\ln(2/\delta)}.$$
When this holds, the shuffled reports are $(\varepsilon, \delta)$-DP for a small slack $\delta$, with
$$\varepsilon \le \ln\left(1 + \frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1}\left(\frac{8\sqrt{e^{\varepsilon_0}\ln(4/\delta)}}{\sqrt n} + \frac{8e^{\varepsilon_0}}{n}\right)\right).$$ {#eq-trust-fmt}

In practice you solve the bound for the local level. Roughly, a central $\varepsilon$ allows $\varepsilon_0$ up to about $\ln(\varepsilon^2 n/\ln(1/\delta))$. For a central $\varepsilon = 1$ at $\delta = 10^{-6}$, ten thousand people may each randomize at $\varepsilon_0 = 3.504$, but at a thousand people the condition caps it at 1.460. The closed form is several times looser than the numerical bound of the same paper, so plan with the closed form, and deploy with that numerical bound, for which the authors publish code. When you quote the shuffled $\varepsilon$, say that it assumes a trusted shuffler.

Three things undo the amplification. A shuffler that colludes with the analyst leaves each report at its local $\varepsilon_0$, so keep that level reasonable. Fake senders provide no cover, so count only honest ones in @eq-trust-fmt. And timing and message sizes must not leak, which is why real shufflers batch and pad their traffic.

## Distributed DP: splitting the noise {#sec-trust-distributed}

The **distributed model** adds the central amount of noise once, in pieces. Each client adds a small share of noise to its value, and **secure aggregation** reveals only the sum to the server ([Cryptography basics](../part01-prerequisites/08-crypto-basics.md)).

Gaussian noise splits cleanly. Write $\sigma$ for the standard deviation of the central noise, and suppose each of the $n$ clients adds normal noise of variance $\sigma^2/n$. The shares then sum to the central noise of variance $\sigma^2$, so the server's view is as private as the central [Gaussian mechanism](03-laplace-gaussian.md). The small shares are also the weakness. When some number $t$ of the clients collude with the server or drop out, only the honest clients' noise is left, of standard deviation
$$\sigma\sqrt{\frac{n - t}{n}}.$$
So decide in advance the largest fraction $\rho$ of clients that may collude or drop out, and have each client add variance $\sigma^2/(n(1 - \rho))$. For example, take a histogram summed through 100 clients and calibrated to $\varepsilon = 1$. If a fifth of the clients collude, the guarantee weakens to 1.128, and with only a tenth honest it is 3.525.

Secure aggregation works on integers modulo a large number, so clients round their scaled vectors and add integer noise ([Discrete mechanisms](08-discrete-mechanisms.md)). The usual choices are Skellam noise (Agarwal, Kairouz & Liu 2021) and the distributed discrete Gaussian (Kairouz, Liu & Steinke 2021), which TensorFlow Federated implements in `tff.aggregators.DistributedDpSumFactory`.

## Cheat sheet

| Situation | Use | Rule |
| --- | --- | --- |
| Nobody may see raw records | Local randomized response | Only for frequent items in very large populations |
| Frequencies over $k$ values | $k$-ary randomized response or optimized unary encoding | $k$-ary if $k < 3e^{\varepsilon_0} + 2$ |
| A huge or unknown domain | Local hashing and growing prefixes | About $e^{\varepsilon_0} + 1$ buckets |
| A bounded number | One-bit mean | Fix the bound before collection |
| The same question every day | Memoization, $\alpha$-point rounding for drifting numbers | Each new value costs another $\varepsilon_\infty$ |
| An anonymizing relay | Shuffle model | Plan with @eq-trust-fmt, deploy with a numerical bound |
| Federated sums | Distributed Gaussian, Skellam or discrete Gaussian | Over-provision for colluding and dropped clients |

: Which trust model and protocol to use, by situation. {#tbl-trust-cheat}

**Rules of thumb**

- Local DP costs about $\sqrt n$ in error over central DP.
- Always state the trust model next to $(\varepsilon, \delta)$.

## Further reading

- Wang, Blocki, Li & Jha, "Locally Differentially Private Protocols for Frequency Estimation" (USENIX Security 2017): frequency oracles and local hashing, compared.
- Feldman, McMillan & Talwar, "Hiding Among the Clones" (FOCS 2021): the shuffling bounds, closed form and numerical.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/09-trust-models.md).
