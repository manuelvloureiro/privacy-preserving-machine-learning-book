# Local, shuffle and distributed trust models

## Why this matters

Every mechanism so far in this part assumed a **trusted curator**, which holds the raw data and releases only the output of the mechanism. In this **central model**, the differential privacy (DP) guarantee protects what the curator publishes. It does not protect against a breach of its database, a curious engineer or a court order. A browser vendor that collects usage statistics often does not have that much trust.

Three other models trust less. In the **local model**, you trust nobody, and each person randomizes their own record before it leaves their device. The **shuffle model** trusts a shuffler only to cut the link between each report and its sender. The **distributed model** trusts cryptography that shows only a sum, run by enough honest participants. The closer to each person you add the noise, the fewer parties you trust, and the more noise reaches the answer.

## Choosing a trust model {#sec-trust-compare}

Less trust costs more noise. As the population grows, the noise on a private count stays fixed in the central model, but grows with its square root in the local model. The shuffle and distributed models close most of that gap.

So use the central model when you can trust a curator with the raw data, as [Release statistics with a library](12-release-statistics.md) does end to end. Choose local only when nobody can see a raw record and the population is large enough, which you check with the standard deviation of @sec-trust-local. In all cases, state the trust model next to the privacy parameters, because the same $\varepsilon$ protects against a different adversary in each model.

## Central and local differential privacy {#sec-trust-local}

In the local model, each person sends only a **report**, the output of a **local randomizer** $R$ on their own record. Write $\varepsilon_0$ for the privacy level of one report. The randomizer is $\varepsilon_0$-DP when no set of possible reports becomes more than $e^{\varepsilon_0}$ times as likely when the record changes. That is, all sets of reports $S$ and all pairs of records $x$ and $x'$ satisfy
$$\Pr[R(x) \in S] \le e^{\varepsilon_0}\Pr[R(x') \in S].$$ {#eq-trust-ldp}
A change to one person's record changes only their own report, so the whole collection is $\varepsilon_0$-DP in the central sense, with neighbours that differ in one replaced record. The guarantee covers only the randomized value, so a network address stored with a report links the report back to the person.

The basic local randomizer is [randomized response](01-definition.md) on one bit. Each person reports their true bit with a probability $p = e^{\varepsilon_0}/(1 + e^{\varepsilon_0})$, and the flipped bit otherwise. To remove the expected flips, the analyst subtracts $1 - p$ from the share of reported ones and divides by $2p - 1$. The standard deviation of the estimate then depends only on the level and the number of people $n$:
$$\operatorname{sd}(\hat\pi) = \frac{e^{\varepsilon_0/2}}{(e^{\varepsilon_0} - 1)\sqrt n}.$$

The script [practitioner_adult_statistics.py](../../code/part03-dp-foundations/practitioner_adult_statistics.py) runs the two models at $\varepsilon_0 = \ln 3$ on the 48,842 people of the Adult census data. Each bit says whether a person's income is more than \$50K. In the local model, diffprivlib's `Binary` mechanism randomizes each bit, and the analyst sees only the reports.

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

In the central model, a trusted curator counts the true bits with diffprivlib's `tools.count_nonzero` at the same level. This call adds Laplace noise with standard deviation $\sqrt2/\varepsilon$ to the count.

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

So at the same guarantee, the local estimate is 149 times noisier, and the factor grows like the square root of the population. This cost comes from the model, not from randomized response, because no local protocol can estimate a fraction with an error below order $1/((e^{\varepsilon_0} - 1)\sqrt n)$, and randomized response reaches that order. That is why Apple, Google and Microsoft use local DP only for the most frequent items in very large populations.

## Frequency estimation: $k$-ary randomized response and unary encoding {#sec-trust-frequency}

Most deployed local DP does **frequency estimation**. Each person has one value from a known list, such as an emoji, and the analyst wants the share of people with each value. A report votes for a value with some probability $p_1$ if the person has it, and with a smaller probability $p_0$ otherwise. Suppose that $c_v$ of the $n$ reports vote for a value $v$. When you remove the false votes on average, you get the unbiased estimate
$$\hat f_v = \frac{c_v/n - p_0}{p_1 - p_0}.$$

Two designs set $p_1$ and $p_0$ for a list of $k$ values. In **$k$-ary randomized response**, each person reports their true value with probability $e^{\varepsilon_0}/(e^{\varepsilon_0} + k - 1)$, and another value otherwise. **Optimized unary encoding** sends one bit per value, and keeps the 1 of the person's value only half the time, but almost never changes a 0 into a 1. The variance for a rare value sets the smallest share that you can see, so compare the designs by it. Use $k$-ary randomized response when $k < 3e^{\varepsilon_0} + 2$, and unary encoding otherwise, at the cost of a $k$-bit report. The pure-ldp Python package has the two designs.

Take a keyboard app that wants the share of each of 128 emoji from a million users at $\varepsilon_0 = 1$. The estimate for a rare emoji has a standard deviation of 0.0019 under unary encoding, against 0.0066 under $k$-ary randomized response. By convention, an estimate must be three standard deviations above zero before you can tell it from noise. So even the better design loses all emoji that fewer than about $3 \times 0.0019$, or 0.6%, of users choose. Size the population from that floor before you collect. For lists too large to send, local hashing and heavy-hitter protocols send a few bits per report (Wang et al. 2017; Bassily et al. 2017).

## Numbers in one bit: local mean estimation {#sec-trust-mean}

For numbers between 0 and a known bound, such as minutes in an app, one bit per person is enough (Ding, Kulkarni & Yekhanin 2017). The chance of a 1 increases in a straight line with the value, so the analyst inverts that line to get an unbiased mean. For a million users, a bound of 1,440 minutes and $\varepsilon_0 = 1$, the mean has a standard deviation of 1.420 minutes. Fix the bound before collection and clip to it on the device. A bound from the reports would let one person's value change the randomizer of all users.

## Collecting every day: memoization {#sec-trust-repeated}

Telemetry asks the same question every day. If a value does not change, an analyst can average its daily reports to remove their fresh noise. [Composition](02-properties.md) counts that leak as the sum of the $\varepsilon_0$ of each report. The fix is **memoization**. The device draws one permanent randomized response per value and sends it each day, so an unchanged value costs only the level of that response, written $\varepsilon_\infty$. RAPPOR, Google's collection of the homepages of Chrome users, has $\varepsilon_\infty = 2.043$.

Memoization fails when the value changes, because each new value costs another $\varepsilon_\infty$. A user who changes their homepage every month gets to $12 \times 2.043 = 24.5$ in a year.

## The shuffle model and amplification by shuffling {#sec-trust-shuffle}

In the **shuffle model** (Cheu et al. 2019), each person sends a local report to a **shuffler**. The shuffler removes identifiers and timing, and sends the reports on in random order. Each report then hides among the others, so the collection meets central $(\varepsilon, \delta)$-DP at an $\varepsilon$ far below $\varepsilon_0$ ([Amplification by shuffling and by iteration](../part04-privacy-accounting/07-shuffling-iteration.md)). The closed-form bound of Feldman, McMillan & Talwar (2021) gives that $\varepsilon$ only when there are enough people to hide among:
$$\varepsilon_0 \le \ln\frac{n}{16\ln(2/\delta)}.$$

In practice, you fix the central target and solve for the largest $\varepsilon_0$ that it allows. For a central $(1, 10^{-6})$, ten thousand people can each randomize at $\varepsilon_0 = 3.504$. But at a thousand people, the condition above limits it to 1.460. These targets use a $\delta$ below the book's default ([How small δ must be](01-definition.md#sec-dpdef-small-delta)). At ten thousand people, the closed form gives an $\varepsilon$ 2.6 to 4.6 times too large, so plan with it, but deploy with the paper's numerical bound.

Three things remove the amplification. A shuffler that colludes with the analyst leaves each report at $\varepsilon_0$. So choose an $\varepsilon_0$ that you can defend on its own, within the tiers of [the definition](01-definition.md#sec-dpdef-testing). Fake senders give no cover, so count only honest ones in $n$. And timing and sizes can leak, so real shufflers batch and pad their traffic.

## Distributed DP: splitting the noise {#sec-trust-distributed}

In the **distributed model**, each client adds a small share of the central noise, and **secure aggregation** shows only the sum to the server ([Cryptography basics](../part01-prerequisites/08-crypto-basics.md)). Gaussian noise splits cleanly, because the variances of independent shares add. If each of $n$ clients adds normal noise of variance $\sigma^2/n$, the sum has noise with standard deviation $\sigma$. That is as private as the central [Gaussian mechanism](03-laplace-gaussian.md).

But the small shares are also the weakness. A client that colludes with the server can show its share, and a client that drops out adds none. With $t$ such clients, only the noise of the honest clients is left, with standard deviation
$$\sigma\sqrt{\frac{n - t}{n}}.$$
So fix in advance the largest fraction $\rho$ of clients that may collude or drop out, and have each client add variance $\sigma^2/(n(1 - \rho))$. The honest clients alone then supply the full noise.

For example, a server sums a histogram of about a hundred thousand users through 100 clients, at $(1, 10^{-6})$. A user who changes value moves one count down and another up, so the $\ell_2$ sensitivity is $\sqrt 2$. The analytic Gaussian calibration of [the Laplace and Gaussian mechanisms](03-laplace-gaussian.md) gives a total $\sigma$ of 5.975. That is $5.975/\sqrt{100} = 0.597$ per client. Without over-provisioning, if a fifth of the clients collude, the guarantee weakens to $\varepsilon = 1.128$ at the same $\delta$.

Secure aggregation works on integers, so clients add integer noise. This noise is Skellam or the distributed discrete Gaussian of [Discrete mechanisms](08-discrete-mechanisms.md), as in TensorFlow Federated's `tff.aggregators.DistributedDpSumFactory`. Secure aggregation also does not check inputs, so a malicious client can still send any vector.

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
