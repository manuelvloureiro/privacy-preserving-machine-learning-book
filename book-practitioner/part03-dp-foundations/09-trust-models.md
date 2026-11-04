# Local, shuffle and distributed trust models

## Why this matters

Earlier mechanisms in this part assume a **trusted curator** who holds raw data and releases only the mechanism's output. In this **central model**, differential privacy (DP) protects released outputs, not data exposed by a breach, employee or court order. A browser vendor collecting usage statistics may not be able to rely on a trusted curator.

Three models reduce reliance on a curator, with different protections. In the **local model**, each person randomizes their record on-device, so no central party sees the raw record. In the **shuffle model**, a shuffler breaks the link between each report and its sender. In the **distributed model**, cryptography and enough honest participants allow a server to reveal only a sum. Each model reduces trust in intermediaries but increases noise.

## Choosing a trust model {#sec-trust-compare}

Reduced trust generally increases noise. For a private count, central-model noise remains fixed as the population grows, while local-model noise grows with the square root of population size. Shuffle and distributed models narrow this gap.

Use the central model when a curator may hold raw data, as in [Release statistics with a library](12-release-statistics.md). Use the local model when no party may see raw records and the population is large enough, based on the standard deviation in @sec-trust-local. Report the trust model with the privacy parameters: the same $\varepsilon$ protects against different adversaries in different models.

## Central and local differential privacy {#sec-trust-local}

In the local model, each person sends a **report**, the output of a **local randomizer** $R$ applied to their record. Let $\varepsilon_0$ denote the privacy level of one report. The randomizer is $\varepsilon_0$-DP if changing a record increases the probability of any report set by at most $e^{\varepsilon_0}$. Write $x$ and $x'$ for two possible records. Let $S$ denote a set of reports. The guarantee is
$$\Pr[R(x) \in S] \le e^{\varepsilon_0}\Pr[R(x') \in S].$$ {#eq-trust-ldp}
Changing one person's record changes only that person's report, so the report collection is $\varepsilon_0$-DP under replace-one-record neighbours. This guarantee protects the randomized value, not identifying metadata. For example, a network address stored with a report may identify its sender.

The basic local randomizer applies [randomized response](01-definition.md) to one bit. Let $p$ be the probability that a report is truthful: $p = e^{\varepsilon_0}/(1 + e^{\varepsilon_0})$. Each person reports the true bit with probability $p$ and its complement otherwise. To correct for flips, the analyst subtracts $1-p$ from the fraction of reported ones and divides by $2p-1$. If $n$ is the number of people, the estimate's standard deviation is
$$\operatorname{sd}(\hat\pi) = \frac{e^{\varepsilon_0/2}}{(e^{\varepsilon_0} - 1)\sqrt n}.$$

The script [practitioner_adult_statistics.py](../../code/part03-dp-foundations/practitioner_adult_statistics.py) compares the models on Adult, a census dataset of 48,842 people. It uses $\varepsilon_0 = \ln 3$ for local randomization. Each bit indicates whether a person's income exceeds \$50K. The local model uses diffprivlib's `Binary` mechanism, so the analyst receives only randomized reports.

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

For the central-model estimate, a trusted curator applies diffprivlib's `tools.count_nonzero` to the true bits at the same privacy level. The call adds Laplace noise with standard deviation $\sqrt2/\varepsilon$ to the count.

```{.python excerpt="adult-central-count"}
rich = (adult["income"] == ">50K").to_numpy()
central = tools.count_nonzero(rich, epsilon=log(3), random_state=seed) / len(rich)
central_sd = math.sqrt(Laplace(epsilon=log(3), sensitivity=1).variance(0)) / len(rich)  # the count's noise
print(f"central estimate: {central:.4f}  sd {central_sd:.6f}")
```

Running `uv run python code/part03-dp-foundations/practitioner_adult_statistics.py` with seed 0 gives:

```text
true share over 50K: 0.2393
local estimate: 0.2359  sd 0.0039
central estimate: 0.2393  sd 0.000026
local sd / central sd: 149
```

At the same guarantee, local noise is 149 times larger, and the ratio grows with the square root of population size. This cost is inherent to local DP, not specific to randomized response: no local protocol estimates a fraction with error below order $1/((e^{\varepsilon_0} - 1)\sqrt n)$, and randomized response achieves that order. Apple, Google and Microsoft therefore use local DP mainly to estimate frequent items across very large populations.

## Frequency estimation: $k$-ary randomized response and unary encoding {#sec-trust-frequency}

Most deployed local-DP systems perform **frequency estimation**. Each person has a category, such as an emoji, from a known list; the analyst estimates each category's share. Let $v$ denote one category. A report votes for $v$ with probability $p_1$ when the person has that category and with smaller probability $p_0$ otherwise. Let $c_v$ count the reports for $v$ among the $n$ reports. Correcting for expected false votes gives the unbiased estimate
$$\hat f_v = \frac{c_v/n - p_0}{p_1 - p_0}.$$

Two designs set $p_1$ and $p_0$ for a list of $k$ values. In **$k$-ary randomized response**, a person reports the true value with probability $e^{\varepsilon_0}/(e^{\varepsilon_0} + k - 1)$ and otherwise reports another value. **Optimized unary encoding** sends one bit per value, retains the bit for the person's value about half the time and rarely flips a 0 to 1. Compare the designs' variances for rare values, which determine the smallest detectable share. Use $k$-ary randomized response when $k < 3e^{\varepsilon_0} + 2$; otherwise use unary encoding, which requires a $k$-bit report. The pure-ldp Python package implements both.

Consider a keyboard app estimating 128 emoji frequencies among one million users, at $\varepsilon_0 = 1$. For a rare emoji, unary encoding has standard deviation 0.0019; $k$-ary randomized response has 0.0066. A common detection rule requires an estimate to exceed zero by three standard deviations. Even the better design cannot reliably detect emoji used by fewer than about $3 \times 0.0019$, or 0.6%, of users. Check the population size before collection. For larger lists, local hashing and heavy-hitter protocols reduce report length to a few bits (Wang et al. 2017; Bassily et al. 2017).

## Numbers in one bit: local mean estimation {#sec-trust-mean}

One bit per person can estimate a mean when each value lies in a known interval, such as app-use minutes (Ding, Kulkarni & Yekhanin 2017). The probability of reporting 1 increases linearly with the value. The analyst inverts this relation to estimate the mean without bias. With one million users and a bound of 1,440 minutes, the standard deviation is 1.420 minutes at $\varepsilon_0 = 1$. Set the bound before collection and clip values on-device. Otherwise, one person's value could affect every user's randomizer through the estimated bound.

## Collecting every day: memoization {#sec-trust-repeated}

Telemetry asks the same question every day. If a value does not change, an analyst can average its daily reports to remove their fresh noise. [Composition](02-properties.md) counts that leak as the sum of the $\varepsilon_0$ of each report. **Memoization** prevents this averaging attack: the device draws one permanent randomized response per value and sends it each day. An unchanged value then costs only the level of that response, written $\varepsilon_\infty$. RAPPOR, Google's collection of the homepages of Chrome users, has $\varepsilon_\infty = 2.043$.

Memoization fails when the value changes, because each new value costs another $\varepsilon_\infty$. A user who changes their homepage every month gets to $12 \times 2.043 = 24.5$ in a year.

## The shuffle model and amplification by shuffling {#sec-trust-shuffle}

In the **shuffle model** (Cheu et al. 2019), each person sends a local report to a **shuffler**. The shuffler removes identifiers and timing, then sends the reports in random order. Because each report is hidden among the others, the collection can satisfy central DP with privacy level $\varepsilon$ far below the local level $\varepsilon_0$. The guarantee also has slack $\delta$ ([Amplification by shuffling and by iteration](../part04-privacy-accounting/07-shuffling-iteration.md)). Feldman, McMillan & Talwar (2021) give a closed-form bound that requires enough people:
$$\varepsilon_0 \le \ln\frac{n}{16\ln(2/\delta)}.$$

Fix the central target, then solve for the largest allowed $\varepsilon_0$. The target here is privacy level 1 and slack $10^{-6}$. At a population of 10,000, the bound allows $\varepsilon_0 = 3.504$. At 1,000 people, it allows $\varepsilon_0 = 1.460$. These targets use a $\delta$ below the book's default ([How small δ must be](01-definition.md#sec-dpdef-small-delta)). For 10,000 people, the closed-form bound overestimates the privacy level by a factor between 2.6 and 4.6. Use it for planning, but use the paper's numerical bound for deployment.

Three conditions can remove amplification. If the shuffler colludes with the analyst, each report retains privacy level $\varepsilon_0$. Choose a local level that is acceptable on its own under [the definition](01-definition.md#sec-dpdef-testing). Fake senders provide no protection; therefore, count only honest participants in $n$. Timing and message sizes may also leak information, so practical shufflers batch and pad traffic.

## Distributed DP: splitting the noise {#sec-trust-distributed}

In the **distributed model**, each client adds a share of the central noise, and **secure aggregation** reveals only the sum to the server ([Cryptography basics](../../book-full/part01-prerequisites/08-crypto-basics.md)). Independent Gaussian variances add. If each of $n$ clients adds normal noise with variance $\sigma^2/n$, the sum has standard deviation $\sigma$, matching the central [Gaussian mechanism](03-laplace-gaussian.md).

Small shares create a vulnerability: colluding clients can reveal their noise, and dropped clients add none. Let $t$ be the number of clients that are unavailable or colluding. The remaining noise has standard deviation
$$\sigma\sqrt{\frac{n - t}{n}}.$$
Set in advance the largest fraction $\rho$ of clients that may collude or drop out. Have each client add variance $\sigma^2/(n(1 - \rho))$ so the remaining honest clients provide the required total noise.

For example, a server aggregates a histogram for about 100,000 users through 100 clients. The target is $(1, 10^{-6})$. Changing one user's value decrements one count and increments another, giving $\ell_2$ sensitivity $\sqrt 2$. Analytic Gaussian calibration in [the Laplace and Gaussian mechanisms](03-laplace-gaussian.md) gives total standard deviation $\sigma = 5.975$. This is $5.975/\sqrt{100} = 0.597$ per client. If a fifth of the clients collude and noise is not over-provisioned, the guarantee weakens to $\varepsilon = 1.128$ at the same $\delta$.

Secure aggregation operates on integers, so clients add integer-valued noise, such as Skellam noise or the distributed discrete Gaussian in [Discrete mechanisms](08-discrete-mechanisms.md). TensorFlow Federated provides `tff.aggregators.DistributedDpSumFactory` for this task. However, secure aggregation does not validate inputs, so a malicious client can still send an arbitrary vector.

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
