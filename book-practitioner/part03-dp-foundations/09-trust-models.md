# Local, shuffle and distributed trust models

## Why this matters

Every mechanism so far in this part assumed a **trusted curator**, which holds the raw data and releases only the mechanism's output. In this **central model**, the differential privacy (DP) guarantee protects what the curator publishes, but not against a breach of its database, a curious engineer or a court order. A browser vendor that collects usage statistics is often not trusted that far.

Three other models trust less. In the **local model** nobody is trusted, and each person randomizes their own record before it leaves their device. The **shuffle model** trusts a shuffler only to cut the link between each report and its sender. The **distributed model** trusts cryptography that reveals only a sum, run by enough honest participants. The closer to each person the noise is added, the fewer parties you trust, and the more noise reaches the answer.

## Choosing a trust model {#sec-trust-compare}

That extra noise is the price of trusting less. As the population grows, the noise on a private count stays fixed in the central model but grows with its square root in the local model. On the full edition's eight-value histogram over a hundred thousand people, the local model's squared error is about 42,000 times the central one at the same guarantee. The shuffle and distributed models close most of that gap.

So use the central model whenever a curator can be trusted with the raw data, as [Release statistics with a library](12-release-statistics.md) does end to end. Choose local only when nobody may see a raw record and the population is large enough, which you check by solving the standard deviation of @sec-trust-local for the number of people. Either way, state the trust model next to the privacy parameters, because the same $\varepsilon$ protects against a different adversary in each model.

## Central and local differential privacy {#sec-trust-local}

In the local model, each person sends only a **report**, the output of a **local randomizer** $R$ run on their own record. Write $\varepsilon_0$ for the privacy level of one report. The randomizer is $\varepsilon_0$-DP when no set of possible reports becomes more than $e^{\varepsilon_0}$ times as likely when the record changes. In symbols, every set of reports $S$ and every two records $x$ and $x'$ satisfy
$$\Pr[R(x) \in S] \le e^{\varepsilon_0}\Pr[R(x') \in S].$$ {#eq-trust-ldp}
Changing one person's record changes only their own report, so the whole collection is then $\varepsilon_0$-DP in the central sense, with neighbours that differ in one replaced record. The guarantee covers only the randomized value, so a network address stored with a report links it back to the person.

The basic local randomizer is [randomized response](01-definition.md) on one bit: each person reports their true bit with a probability $p = e^{\varepsilon_0}/(1 + e^{\varepsilon_0})$, and the flipped bit otherwise. The analyst undoes the expected flips by subtracting $1 - p$ from the share of reported ones and dividing by $2p - 1$. The estimate's standard deviation then depends only on the level and the number of people $n$:
$$\operatorname{sd}(\hat\pi) = \frac{e^{\varepsilon_0/2}}{(e^{\varepsilon_0} - 1)\sqrt n}.$$

The script [practitioner_adult_statistics.py](../../code/part03-dp-foundations/practitioner_adult_statistics.py) runs both models at $\varepsilon_0 = \ln 3$ on the 48,842 people of the Adult census data, where each bit says whether a person's income is over \$50K. In the local model, each bit goes through diffprivlib's `Binary` mechanism, and the analyst sees only the reports.

```{.python excerpt="adult-local-rr"}
answers = (adult["income"] == ">50K").cast(pl.Int8).cast(pl.String).to_list()  # "1" or "0"
n, eps = len(answers), log(3)
coin = Binary(epsilon=eps, value0="0", value1="1", random_state=seed)
reports = [coin.randomise(a) for a in answers]  # each person randomizes their own answer
p = exp(eps) / (1 + exp(eps))  # chance a report is truthful: 0.75
reported = reports.count("1") / n
local = (reported - (1 - p)) / (2 * p - 1)  # undo the expected flips
local_sd = math.sqrt(p * (1 - p) / n) / (2 * p - 1)  # the randomization noise alone
print(f"true share over 50K: {answers.count('1') / n:.4f}")
print(f"local estimate: {local:.4f}  sd {local_sd:.4f}")
```

In the central model, a trusted curator counts the true bits with diffprivlib's `tools.count_nonzero` at the same level, which adds Laplace noise with standard deviation $\sqrt2/\varepsilon$ to the count.

```{.python excerpt="adult-central-count"}
rich = (adult["income"] == ">50K").to_numpy()
central = tools.count_nonzero(rich, epsilon=log(3), random_state=seed) / len(rich)
central_sd = math.sqrt(Laplace(epsilon=log(3), sensitivity=1).variance(0)) / len(rich)  # the count's noise
print(f"central estimate: {central:.4f}  sd {central_sd:.6f}")
```

Its seed-0 run, `uv run python code/part03-dp-foundations/practitioner_adult_statistics.py`, prints this.

```text
true share over 50K: 0.2393
local estimate: 0.2359  sd 0.0039
central estimate: 0.2393  sd 0.000026
local sd / central sd: 149
```

So at the same guarantee, the local estimate is 149 times noisier, and the factor grows like the square root of the population. It belongs to the model, not to randomized response: no local protocol can estimate a fraction with an error below order $1/((e^{\varepsilon_0} - 1)\sqrt n)$, and randomized response reaches that order. That is why Apple, Google and Microsoft use local DP only for the most frequent items in very large populations.

## Frequency estimation: $k$-ary randomized response and unary encoding {#sec-trust-frequency}

The workhorse of deployed local DP is **frequency estimation**. Each person holds one value from a known list, such as an emoji, and the analyst wants the share holding each one. A report votes for a value with some probability $p_1$ if the person holds it, and with a smaller probability $p_0$ otherwise. If $c_v$ of the $n$ reports vote for a value $v$, removing the false votes on average gives the unbiased estimate
$$\hat f_v = \frac{c_v/n - p_0}{p_1 - p_0}.$$

Two designs set $p_1$ and $p_0$ for a list of $k$ values. In **$k$-ary randomized response**, each person reports their true value with probability $e^{\varepsilon_0}/(e^{\varepsilon_0} + k - 1)$, and another value otherwise. **Optimized unary encoding** sends one bit per value, keeping the 1 of the value held only half the time but almost never turning a 0 into a 1. Most values of a long list are rare, and a rare value's variance sets the smallest share that can be seen. Comparing those variances, use $k$-ary randomized response when $k < 3e^{\varepsilon_0} + 2$, and unary encoding otherwise, at the price of a $k$-bit report. The pure-ldp Python package implements both.

Take a keyboard app that wants the share of each of 128 emoji from a million users at $\varepsilon_0 = 1$. A rare emoji's estimate has a standard deviation of 0.0019 under unary encoding, against 0.0066 under $k$-ary randomized response. By convention, an estimate must stand three standard deviations above zero to be told from noise. So even the better design loses every emoji used by fewer than about $3 \times 0.0019$, or 0.6%, of users. Size the population from that floor before collecting. For domains too large to list, local hashing and heavy-hitter protocols send a few bits per report instead of one per value (Wang et al. 2017; Bassily et al. 2017).

## Numbers in one bit: local mean estimation {#sec-trust-mean}

For numbers between 0 and a known bound, such as minutes in an app, one bit per person is enough (Ding, Kulkarni & Yekhanin 2017). The chance of sending a 1 rises in a straight line with the value, so the analyst can invert that line to get an unbiased mean. For a million users, a bound of 1,440 minutes and $\varepsilon_0 = 1$, the mean has a standard deviation of 1.420 minutes. Fix the bound before collection and clip to it on the device, because a bound computed from the reports would let one person's value change everyone's randomizer.

## Collecting every day: memoization {#sec-trust-repeated}

Telemetry asks the same question every day. An analyst who averages the daily reports of an unchanged value averages away their fresh noise, and [composition](02-properties.md) counts that leak by adding each report's $\varepsilon_0$. The fix is **memoization**: the device draws one permanent randomized response per value and reuses it, so an unchanged value costs only that response's level, written $\varepsilon_\infty$. RAPPOR, Google's collection of Chrome users' homepages, has $\varepsilon_\infty = 2.043$.

Memoization fails when the value changes, since each new value costs another $\varepsilon_\infty$. A user who changes their homepage every month reaches $12 \times 2.043 = 24.5$ in a year.

## The shuffle model and amplification by shuffling {#sec-trust-shuffle}

In the **shuffle model** (Cheu et al. 2019), each person sends a local report to a **shuffler**, which strips identifiers and timing and forwards the reports in random order. Each report then hides among the others, so the collection meets central $(\varepsilon, \delta)$-DP at an $\varepsilon$ far below $\varepsilon_0$ ([Amplification by shuffling and by iteration](../part04-privacy-accounting/07-shuffling-iteration.md)). The closed-form bound of Feldman, McMillan & Talwar (2021) gives that $\varepsilon$ only when there are enough people to hide among:
$$\varepsilon_0 \le \ln\frac{n}{16\ln(2/\delta)}.$$

In practice you fix the central target and solve for the largest $\varepsilon_0$ it allows. For a central $(1, 10^{-6})$, ten thousand people may each randomize at $\varepsilon_0 = 3.504$, while at a thousand people the condition above binds first and caps it at 1.460. Both use a $\delta$ below the book's default $1/n^{1.1}$. The closed form overstates $\varepsilon$ by a factor of 2.6 to 4.6 at ten thousand people, so plan with it, and deploy with the paper's published numerical bound.

Three things undo the amplification. A shuffler that colludes with the analyst leaves each report at $\varepsilon_0$, so choose an $\varepsilon_0$ you could defend on its own, within the tiers of [the definition](01-definition.md#sec-dpdef-testing). Fake senders give no cover, so count only honest ones in $n$. And real shufflers batch and pad their traffic, so that timing and sizes do not leak.

## Distributed DP: splitting the noise {#sec-trust-distributed}

In the **distributed model**, each client adds a small share of the central noise, and **secure aggregation** reveals only the sum to the server ([Cryptography basics](../part01-prerequisites/08-crypto-basics.md)). Gaussian noise splits cleanly, because variances of independent shares add. If each of $n$ clients adds normal noise of variance $\sigma^2/n$, the shares sum to central noise with standard deviation $\sigma$, as private as the central [Gaussian mechanism](03-laplace-gaussian.md).

The small shares are also the weakness. A client that colludes with the server can reveal its share, and one that drops out adds none. With $t$ such clients, only the honest clients' noise is left, with standard deviation
$$\sigma\sqrt{\frac{n - t}{n}}.$$
So fix in advance the largest fraction $\rho$ of clients that may collude or drop out, and have each client add variance $\sigma^2/(n(1 - \rho))$. The honest clients alone then supply the full noise.

For example, a server sums a histogram of about a hundred thousand users through 100 clients and targets $(1, 10^{-6})$, the default $\delta$ for that many users. A user who changes value moves one count down and another up, so the $\ell_2$ sensitivity is $\sqrt 2$. The analytic Gaussian calibration of [the Laplace and Gaussian mechanisms](03-laplace-gaussian.md) gives a total $\sigma$ of 5.975, or $5.975/\sqrt{100} = 0.597$ per client. Without over-provisioning, a fifth of the clients colluding weakens the guarantee to $\varepsilon = 1.128$ at the same $\delta$, and nine-tenths to 3.525.

Secure aggregation works on integers, so clients add integer noise, Skellam or the distributed discrete Gaussian of [Discrete mechanisms](08-discrete-mechanisms.md), as TensorFlow Federated's `tff.aggregators.DistributedDpSumFactory` does. And it does not check inputs, so a malicious client can still send any vector.

## Cheat sheet

| Situation | Use | Rule |
| --- | --- | --- |
| A curator can be trusted | Central Laplace or Gaussian | The default |
| Nobody may see raw records | Local randomized response | Frequent items, very large populations |
| Frequencies over $k$ values | $k$-ary randomized response or unary encoding | $k$-ary if $k < 3e^{\varepsilon_0} + 2$ |
| The same question daily | Memoization | Each new value costs $\varepsilon_\infty$ |
| An anonymizing relay | Shuffle model | Deploy with a numerical bound |
| Federated sums | Distributed Gaussian or Skellam | Over-provision for colluders |

: Which trust model and protocol to use, by situation. {#tbl-trust-cheat}

**Rules of thumb**

- Local noise exceeds central noise by a factor that grows like $\sqrt n$, 149 on Adult at $\ln 3$.

## References

- Wang, Blocki, Li & Jha, "Locally Differentially Private Protocols for Frequency Estimation" (USENIX Security 2017): frequency oracles and local hashing, compared.
- Feldman, McMillan & Talwar, "Hiding Among the Clones" (FOCS 2021): the shuffling bounds, closed form and numerical.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/09-trust-models.md).
