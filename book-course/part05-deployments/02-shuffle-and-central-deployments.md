# The Shuffle Model and Central Deployments

The local model of [the previous chapter](01-local-dp.md) asks no one to be trusted, and pays with an error $\sqrt n$ times larger than the central model's. This chapter first studies a middle ground for differential privacy (DP), the shuffle model, in which a party that only mixes messages lets each person add far less noise. It then turns to deployments in the central model, where a company holds the data and releases statistics or answers queries. These are an analytics service that answers queries from outside analysts, and two large public data releases.

Each deployment brings a piece of theory with it. The shuffle model comes with amplification by shuffling and a counting protocol whose error does not grow with $n$. The query service comes with a sharper way to account for the exponential mechanism and a fast way to run it. The data releases come with zero-concentrated differential privacy and with a choice between protecting each action and protecting each person.

## Encode, shuffle, analyze {#sec-shuf-prochlo}

Google's Prochlo system started from a failure of local privacy. Its designers wanted to learn which applications call which system interfaces, so that rarely used interfaces could be retired and old ones tracked. With a thousand applications and a hundred interfaces there are $10^5$ pairs to count. By their estimate, a local protocol such as RAPPOR would need reports from about a trillion people to count them usefully, more people than exist.

Their architecture has three stages, **encode, shuffle, analyze**. Each person *encodes* their data on their own device, for instance with a local randomizer. A *shuffler* collects the encoded messages, strips their metadata such as arrival times and network addresses, and forwards them in a uniformly random order. The *analyst* sees only the shuffled batch and decodes statistics from it. The shuffler's only job is to break the link between a message and its sender.

Someone must run the shuffler, and the users must trust it not to reveal the order. Prochlo names three options: a trusted third party, a split of the work between several parties, or trusted hardware. It implements the last one, running the shuffle inside a trusted execution environment (TEE), an Intel SGX enclave, which is hardware designed so that even the machine's owner cannot see the computation inside. Trust then rests on the hardware vendor, and side-channel attacks such as Spectre have shown that such hardware can leak. Splitting the work is the idea of a mix network, in which several servers each shuffle and re-encrypt the batch, so that the order stays hidden unless all of them collude. In Prochlo's experiments on word frequencies, RAPPOR recovered less than $5\%$ as many distinct words as the shuffled system.

## The shuffle model {#sec-shuf-model}

Prochlo motivated a formal model that sits between the local and central ones. The trust barrier passes after the shuffler: the adversary sees the shuffled messages but not who sent which.

::: {#def-shuf-model name="Shuffle model"}
A protocol in the **shuffle model** gives each person the same randomizer $R\colon \mathcal X \to \mathcal Y$. Person $i$ sends $Y_i = R(X_i)$, using randomness independent of everyone else's, and the **shuffler** outputs $(Y_{\pi(1)}, \dots, Y_{\pi(n)})$ for a permutation $\pi$ drawn uniformly at random. The protocol is **$(\varepsilon, \delta)$-DP in the shuffle model** if this shuffled output, as a mechanism on the dataset $D$, is $(\varepsilon,\delta)$-DP in the central sense.
:::

The shuffled output carries the same information as the multiset of messages, because the uniform order adds nothing. A local protocol is also a shuffle protocol with the same guarantee, since the shuffled output is a post-processing of the messages. The interesting protocols are those whose messages are *not* private one by one but become private once mixed.

The benchmark task is again a sum of bits. In the central model, a count plus Laplace noise of scale $1/\varepsilon$ has error of order $1/\varepsilon$. In the local model each person adds noise of that size, and the $n$ independent noises add up to a standard deviation of order $\sqrt n/\varepsilon$, which no local protocol can beat by [the previous chapter's lower bound](01-local-dp.md). The question is how close the shuffle model can get to $1/\varepsilon$.

## Amplification by shuffling {#sec-shuf-amplification}

The first answer is general. Shuffling makes any local randomizer more private, just as subsampling did in [an earlier chapter](../part04-learning-statistics/02-private-erm.md). Here is the result in the form in which it was first proved. If each person's message is $\varepsilon_0$-private on its own, the shuffled batch is private with a parameter smaller by a factor of order $\sqrt n$.

::: {#thm-shuf-amplification name="Amplification by shuffling"}
Let $n \ge 1000$, $\delta \in (0, \frac1{100})$, and let $R$ be an $\varepsilon_0$-local randomizer with $0 < \varepsilon_0 < \frac12$. Then the shuffled output of $R(X_1), \dots, R(X_n)$ is $(\varepsilon, \delta)$-DP with

$$
\varepsilon = 12\,\varepsilon_0\sqrt{\frac{\ln(1/\delta)}{n}} .
$$
:::
<!-- proof: cited -->

The proof is in the 2019 paper by Erlingsson and coauthors listed in the references (Theorem 7 and Corollary 9 there). It shuffles the records before randomizing them, which gives the same output distribution when everyone uses the same randomizer, and which also allows each randomizer to depend on the messages before it. The proof moves the one differing record to the front and swaps it into a uniformly random position, and then bounds how much each successive message reveals about that position.

The gain is a factor of order $\sqrt n$, but it applies only to randomizers that are already very private. Few deployments want $\varepsilon_0 \le \frac12$ locally, because that is the regime where local accuracy is worst. The useful regime is $\varepsilon_0$ of order $\ln n$, which the theorem does not cover. A later analysis by Feldman, McMillan and Talwar (2021) does, and still yields a constant $\varepsilon$ there. It covers every $\varepsilon_0 \le \ln\big(n/(16\ln(2/\delta))\big)$, with $\varepsilon = O\big((1 - e^{-\varepsilon_0})\sqrt{e^{\varepsilon_0}\ln(1/\delta)/n}\big)$. Its proof splits each other person's randomizer into a mixture, in which with probability $e^{-\varepsilon_0}$ the message looks like a "clone" of what the differing person might have sent, and the count of clones hides that person. The next section carries out this idea in full for one simple randomizer.

## Counting bits in the shuffle model {#sec-shuf-bits}

For a sum of bits, a direct protocol reaches error of order $\sqrt{\ln(1/\delta)}/\varepsilon$, independent of $n$. Its randomizer is randomized response with a twist: almost everyone tells the truth.

::: {#def-shuf-rr name="Shuffled randomized response"}
Fix a coin probability $\nu \in (0,1]$. Person $i$, with bit $X_i$, sends $Y_i = X_i$ with probability $1 - \nu$, and otherwise sends a fresh fair coin. The analyst sees the shuffled reports, counts their ones as $Z = \sum_{i=1}^n Y_i$, and estimates the count $c = \sum_{i=1}^n X_i$ by

$$
\tilde c = \frac{Z - n\nu/2}{1 - \nu}.
$$
:::

In the local model, $\nu$ would be a constant. Here it will be of order $\ln(1/\delta)/(\varepsilon^2 n)$, so only about $\ln(1/\delta)/\varepsilon^2$ people in total send coins. The coins of the others hide any one person, much as the clones above. Two lemmas prepare the proof, and both use $\tau = \tanh(\varepsilon/2)$, the number for which $(1+\tau)/(1-\tau) = e^\varepsilon$. The first says that the number of ones among $s$ fair coins hides whether one more bit, of any distribution, is added to it, as long as $s$ is large.

::: {#lem-shuf-binomial}
Let $\varepsilon > 0$, let $\tau = \tanh(\varepsilon/2)$, and let $s \ge 2/\tau$ be an integer. Let $N$ be binomial with $s$ trials and success probability $\frac12$, and let $B$ and $B'$ be Bernoulli variables independent of $N$, with any parameters. Then for every set $S$ of integers,

$$
\Pr[N + B \in S] \le e^{\varepsilon}\,\Pr[N + B' \in S] + 2e^{-s\tau^2/8}.
$$
:::

::: {.proof}
1. Write $P(k) = \Pr[N = k] = \binom sk 2^{-s}$. For $1 \le k \le s$, the ratio of consecutive probabilities is $P(k-1)/P(k) = k/(s+1-k)$.
2. Let $K$ be the set of integers $k$ with $|k - \frac{s+1}2| \le \frac{s+1}2\tau$. Writing $k = \frac{s+1}2(1+v)$ with $|v| \le \tau$, the ratio becomes $(1+v)/(1-v)$, which lies in $[e^{-\varepsilon}, e^{\varepsilon}]$. Since $\tau < 1$, every $k \in K$ satisfies $1 \le k \le s$.
3. For $k \in K$ and any Bernoulli parameter $b$, $\Pr[N + B = k] = (1-b)P(k) + bP(k-1)$ is at most $\max(P(k), P(k-1))$. By step 2 this is at most $e^\varepsilon\min(P(k), P(k-1))$, which is at most $e^{\varepsilon}\Pr[N + B' = k]$.
4. Outside $K$, use $|N + B - \frac{s+1}2| \le |N - \frac s2| + \frac12$. So $N + B \notin K$ requires $|N - \frac s2| > \frac{(s+1)\tau - 1}{2}$, and this threshold is at least $s\tau/4$ because $s\tau \ge 2$. Hoeffding's inequality for $s$ fair coins gives $\Pr[N + B\notin K] \le 2\exp(-2(s\tau/4)^2/s) = 2e^{-s\tau^2/8}$.
5. Split $S$ into $S \cap K$, bounded by step 3, and the rest, bounded by step 4.
:::

The second lemma ensures that enough people send coins. It is a lower tail for the binomial distribution, sharper than Hoeffding's inequality when the mean is small.

::: {#lem-shuf-chernoff}
Let $N$ be binomial with mean $\mu$. Then

$$
\Pr\!\left[N \le \frac\mu2\right] \le \exp\!\left(-\frac{(1-\ln 2)\,\mu}{2}\right).
$$
:::

::: {.proof}
1. Let $N$ have $m$ trials with success probability $p$, so $\mu = mp$. For $\lambda > 0$, Markov's inequality applied to $e^{-\lambda N}$ gives $\Pr[N \le a] \le e^{\lambda a}\,\mathbb E[e^{-\lambda N}]$.
2. The trials are independent, so $\mathbb E[e^{-\lambda N}] = (1 - p + pe^{-\lambda})^m$. Using $1 + z \le e^z$, this is at most $\exp\big(\mu(e^{-\lambda} - 1)\big)$.
3. Take $a = \mu/2$ and $\lambda = \ln 2$. The exponent becomes $\mu\ln 2/2 - \mu/2$.
:::

The privacy theorem chooses the coin probability $\nu$ so that, by @lem-shuf-chernoff, enough people send coins with high probability. Those coins then hide the one person in question, by @lem-shuf-binomial.

::: {#thm-shuf-rr-privacy name="Privacy of shuffled randomized response"}
Let $\varepsilon > 0$, $\delta \in (0,1)$, $\tau = \tanh(\varepsilon/2)$ and $n \ge 1 + 16\ln(4/\delta)/\tau^2$, and set

$$
\nu = \frac{16\ln(4/\delta)}{\tau^2(n-1)} .
$$ {#eq-shuf-q}

Then shuffled randomized response is $(\varepsilon,\delta)$-DP in the shuffle model.
:::

::: {.proof}
1. *Reduce to the count.* Given the count $Z$, every arrangement of the shuffled bits with $Z$ ones is equally likely. So the shuffled output and $Z$ are functions of each other, up to independent randomness, and it suffices to prove that $Z$ is $(\varepsilon,\delta)$-DP.
2. *Isolate one person.* The count does not depend on the order of the people, so we may take neighbours $D \sim D'$ that differ by replacing the record of person $1$. Let $J$ be the set of persons $i \ge 2$ who send coins. Its law does not depend on the data, and given $J$,

$$
Z = C_J + N + Y_1, \qquad C_J = \sum_{i \ge 2,\ i \notin J} X_i,
$$

where $N$ counts the ones among the $|J|$ fair coins. Both $C_J$ and the law of $N$ are the same under $D$ and $D'$, and only the Bernoulli variable $Y_1$ changes.
3. *Enough coins.* Let $s_0 = 8\ln(4/\delta)/\tau^2$. The size $|J|$ is binomial with mean $(n-1)\nu = 2s_0$. By @lem-shuf-chernoff, $\Pr[|J| < s_0] \le \exp(-(1-\ln 2)\,8\ln(4/\delta)/\tau^2)$. Since $8(1-\ln 2) > 2$ and $\tau < 1$, this is at most $(\delta/4)^2 \le \delta/4$.
4. *Each large $J$ is private.* If $|J| \ge s_0$, then $|J|\tau \ge 8\ln 4/\tau > 2$, so @lem-shuf-binomial applies. Its error term is $2e^{-|J|\tau^2/8} \le 2e^{-\ln(4/\delta)} = \delta/2$. Shifting by the constant $C_J$ preserves the inequality, so for every event $S$, $\Pr_D[Z \in S \mid J] \le e^\varepsilon\Pr_{D'}[Z\in S \mid J] + \delta/2$.
5. *Average over $J$.* Summing over sets with $|J| \ge s_0$ and adding the probability of the others,

$$
\Pr_D[Z \in S] \le e^{\varepsilon}\Pr_{D'}[Z \in S] + \frac\delta2 + \frac\delta4 \le e^{\varepsilon}\Pr_{D'}[Z \in S] + \delta.
$$
:::

The coins also add error to the count. But only about $n\nu$ people send them, so the error depends on $n\nu$, which @eq-shuf-q keeps independent of $n$.

::: {#thm-shuf-rr-accuracy name="Accuracy of shuffled randomized response"}
For every dataset, $\tilde c$ is an unbiased estimate of $c$ with variance at most $n\nu/(2(1-\nu)^2)$. With $\nu$ as in @eq-shuf-q and $\beta \in (0,1)$, with probability at least $1 - \beta$,

$$
|\tilde c - c| \le \frac{1}{1-\nu}\sqrt{\frac{n\nu}{2\beta}} \le \frac{4}{(1-\nu)\,\tau}\sqrt{\frac{\ln(4/\delta)}{\beta}} .
$$

For $\varepsilon \le 1$ and $\nu \le \frac12$, the right-hand side is at most $18\sqrt{\ln(4/\delta)/\beta}\,/\varepsilon$.
:::

::: {.proof}
1. Each report is $1$ with probability $(1-\nu)X_i + \nu/2$, so $\mathbb E[Z] = (1-\nu)c + n\nu/2$, and $\tilde c$ is unbiased.
2. Each report is a Bernoulli variable with parameter $\nu/2$ or $1 - \nu/2$, so its variance is $\frac \nu2(1 - \frac \nu2) \le \frac \nu2$. The reports are independent, so $\operatorname{Var}(Z) \le n\nu/2$ and $\operatorname{Var}(\tilde c) \le n\nu/(2(1-\nu)^2)$.
3. Chebyshev's inequality gives the first bound. For the second, $n\nu = 16\ln(4/\delta)\,n/(\tau^2(n-1)) \le 32\ln(4/\delta)/\tau^2$ when $n \ge 2$.
4. The function $\tanh$ is concave on $[0,\infty)$, so $\tau \ge \tanh(\frac12)\,\varepsilon = 0.462\,\varepsilon$ for $\varepsilon \le 1$. With $1/(1-\nu) \le 2$, the constant is $8/0.462 < 18$.
:::

So the shuffled count has error $O(\sqrt{\ln(1/\delta)}/\varepsilon)$, which does not grow with $n$. It is within a factor $\sqrt{\ln(1/\delta)}$ of the central model and far from the local model's $\sqrt n/\varepsilon$, as @fig-shuf-counts shows. The example below compares the three models at one setting.

::: {#exm-shuf-rr}
Take $\varepsilon = 1$, $\delta = 10^{-6}$ and $n = 10^6$. Then $\tau = 0.4621$, and @eq-shuf-q gives $\nu = 0.001139$, so about $1{,}139$ people send coins. The standard deviation of $\tilde c$ is at most $23.9$. Local randomized response at the same $\varepsilon$ has standard deviation $\sqrt n\,e^{1/2}/(e - 1) = 959.5$, and central Laplace noise has $\sqrt 2 = 1.414$.
:::

![Error of three private counts of $n$ bits at $\varepsilon = 1$, as the 95th percentile of the absolute error over 2,000 simulated runs on data with half the bits equal to one. The local error grows as $\sqrt n$. The shuffled error, with $\nu$ from @eq-shuf-q and $\delta = 10^{-6}$, stays near 47, and the central Laplace error stays near 3.](figures/shuf-counts.svg){#fig-shuf-counts}

The price is trust in the shuffler. Each message on its own is randomized response that keeps the true bit with probability $1 - \nu/2$, so it is only $\varepsilon_0$-locally private with $\varepsilon_0 = \ln((2-\nu)/\nu) = 7.47$ in the example. A shuffler that leaked the order would expose almost every bit. That $\varepsilon_0$ is below $\ln\big(n/(16\ln(2/\delta))\big) = 8.37$, so this protocol lies in the range that the clone analysis of the previous section covers. Prochlo's engineering, with its anonymization and mixing, is a practice already common in industry, and the shuffle model shows that, combined with a little local noise, it yields a formal guarantee close to the central one.

## A query service: LinkedIn {#sec-shuf-linkedin}

LinkedIn's audience engagements service lets advertisers ask how members who match a target audience engage with content, such as which articles or topics they click on most. It runs in the central model: LinkedIn holds the data and answers each query with a private mechanism. Its design rests on research done for it, of which this section covers two pieces, a sharper analysis of the exponential mechanism and a fast way to run it.

### Bounded range {#sec-shuf-br}

Recall [the exponential mechanism](../part02-dp-basics/05-exponential-mechanism.md). Given a score $u(D, r)$ for each candidate $r$ in a countable set, whose value changes by at most $\Delta u$ between neighbours, it outputs $r$ with probability proportional to $\exp(\varepsilon u(D,r)/(2\Delta u))$, and it is $\varepsilon$-DP. Write

$$
L(y) = \ln\frac{\Pr[M(D) = y]}{\Pr[M(D') = y]}
$$

for the privacy loss of output $y$ between neighbours $D$ and $D'$. Pure $\varepsilon$-DP says $|L(y)| \le \varepsilon$, an interval of width $2\varepsilon$. The exponential mechanism satisfies a stronger condition.

::: {#def-shuf-br name="Bounded range"}
A mechanism $M$ with a countable output set is **$\varepsilon$-bounded range** if for all neighbours $D \sim D'$ and all outputs $y, y'$,

$$
\Pr[M(D) = y]\,\Pr[M(D') = y'] \le e^{\varepsilon}\,\Pr[M(D') = y]\,\Pr[M(D) = y'] .
$$
:::

When the outputs have positive probability, this says $L(y) - L(y') \le \varepsilon$. So the privacy losses lie in an interval of width $\varepsilon$, wherever that interval sits. The next result shows that the two notions are within a factor of two of each other.

::: {#prp-shuf-br-dp}
An $\varepsilon$-bounded range mechanism is $\varepsilon$-DP. An $\varepsilon$-DP mechanism with a countable output set is $2\varepsilon$-bounded range.
:::

::: {.proof}
1. *Same support.* Let $M$ be $\varepsilon$-bounded range. Suppose some output $y$ has $\Pr[M(D) = y] > 0$ but $\Pr[M(D') = y] = 0$. With this $y$, the right-hand side of the definition is zero for every $y'$, and the left-hand side is $\Pr[M(D) = y]$ times $\Pr[M(D') = y']$. So the definition forces $\Pr[M(D') = y'] = 0$ for every $y'$, which is impossible. So $M(D)$ and $M(D')$ give positive probability to the same outputs, and $L$ is finite on them.
2. *Both signs occur.* Both distributions sum to one over the common support, so some output $y_+$ has $L(y_+) \ge 0$ and some output $y_-$ has $L(y_-) \le 0$.
3. *Width $\varepsilon$ around zero.* For every $y$, bounded range gives $L(y) \le L(y_-) + \varepsilon \le \varepsilon$ and $L(y) \ge L(y_+) - \varepsilon \ge -\varepsilon$. The pointwise lemma of [the randomized response chapter](../part02-dp-basics/01-randomized-response.md) turns $|L| \le \varepsilon$ into $\varepsilon$-DP.
4. *Converse.* If $|L| \le \varepsilon$ everywhere, then $L(y) - L(y') \le 2\varepsilon$.
:::

The general conversion loses that factor of two, but the exponential mechanism does not. Its privacy losses already lie in an interval of width $\varepsilon$.

::: {#prp-shuf-em-br}
The exponential mechanism with parameter $\varepsilon$ is $\varepsilon$-bounded range.
:::

::: {.proof}
1. Write $Z_D = \sum_r \exp(\varepsilon u(D,r)/(2\Delta u))$ for the normalizing constant. Then $L(r) = \frac{\varepsilon}{2\Delta u}\big(u(D,r) - u(D',r)\big) + \ln(Z_{D'}/Z_D)$.
2. The constant cancels in a difference: $L(r) - L(r') = \frac{\varepsilon}{2\Delta u}\big[(u(D,r) - u(D',r)) - (u(D,r') - u(D',r'))\big]$.
3. Each bracketed difference is at most $\Delta u$ in absolute value, so $L(r) - L(r') \le \varepsilon$.
:::

The factor of two matters when many selections are composed. Two lemmas make the composition precise. The first converts a tail bound on the privacy loss into a privacy guarantee. It holds for any output space, with $L$ the logarithm of the ratio of probability mass functions or of densities.

::: {#lem-shuf-tail}
Let $\varepsilon' \ge 0$ and $\delta \in [0,1]$. If for all neighbours $D \sim D'$ the loss satisfies $\Pr_{y \sim M(D)}[L(y) > \varepsilon'] \le \delta$, then $M$ is $(\varepsilon', \delta)$-DP.
:::

::: {.proof}
Fix an event $S$ and let $G = \{y : L(y) \le \varepsilon'\}$. On $S \cap G$ the probability under $D$ is at most $e^{\varepsilon'}$ times that under $D'$, so $\Pr[M(D) \in S\cap G] \le e^{\varepsilon'}\Pr[M(D')\in S]$. The rest of $S$ has probability at most $\Pr[M(D) \notin G] \le \delta$ under $D$.
:::

The second lemma bounds the mean loss, using Hoeffding's lemma: a random variable $W$ in an interval of length $\ell$ satisfies $\mathbb E[e^{\lambda(W - \mathbb E W)}] \le e^{\lambda^2\ell^2/8}$ for every real $\lambda$. This is the step at the heart of the proof of Hoeffding's inequality.

::: {#lem-shuf-mean-loss}
If $M$ is $\varepsilon$-bounded range and $D \sim D'$, then $\mathbb E_{y\sim M(D)}[L(y)] \le \varepsilon^2/8$.
:::

::: {.proof}
1. Under $y \sim M(D)$, the variable $W = -L(y)$ lies in an interval of length $\varepsilon$, and $\mathbb E[e^{W}] = \sum_y \Pr[M(D) = y]\,\frac{\Pr[M(D')=y]}{\Pr[M(D)=y]} = 1$.
2. Hoeffding's lemma with $\lambda = 1$ gives $1 = \mathbb E[e^W] \le e^{\mathbb E W + \varepsilon^2/8}$, so $\mathbb E[L] = -\mathbb E[W] \le \varepsilon^2/8$.
:::

The two lemmas combine into a composition theorem. The total privacy loss of $k$ steps is a sum of $k$ losses, each with a small mean and a range of width $\varepsilon$. So the sum concentrates around $k\varepsilon^2/8$, and @lem-shuf-tail turns that concentration into a guarantee.

::: {#thm-shuf-br-composition name="Composition of bounded-range mechanisms"}
Run $k$ mechanisms in sequence on the same dataset, where the $j$-th may be chosen as a function of the outputs of the first $j-1$, and each is $\varepsilon$-bounded range whatever those outputs are. For every $\delta \in (0,1)$, the sequence of outputs is $(\varepsilon', \delta)$-DP with

$$
\varepsilon' = \frac{k\varepsilon^2}{8} + \varepsilon\sqrt{\frac{k\ln(1/\delta)}{2}} .
$$
:::

::: {.proof}
1. The probability of an output sequence is the product of the conditional probabilities of its terms, so the loss is a sum $L = \sum_{j=1}^k L_j$, where $L_j$ is the loss of the $j$-th output given the earlier ones.
2. Given the earlier outputs, $L_j$ lies in an interval of length $\varepsilon$, by bounded range, and its conditional mean is at most $\varepsilon^2/8$, by @lem-shuf-mean-loss. Let $V_j$ be $L_j$ minus its conditional mean, so $L \le \sum_j V_j + k\varepsilon^2/8$.
3. For $\lambda > 0$, Hoeffding's lemma applied conditionally gives $\mathbb E[e^{\lambda V_j} \mid \text{earlier outputs}] \le e^{\lambda^2\varepsilon^2/8}$. Taking conditional expectations from the last term to the first gives $\mathbb E[e^{\lambda\sum_j V_j}] \le e^{k\lambda^2\varepsilon^2/8}$.
4. By Markov's inequality, $\Pr[\sum_j V_j \ge t] \le e^{-\lambda t + k\lambda^2\varepsilon^2/8}$. Choosing $\lambda = 4t/(k\varepsilon^2)$ makes this $e^{-2t^2/(k\varepsilon^2)}$, which equals $\delta$ at $t = \varepsilon\sqrt{k\ln(1/\delta)/2}$.
5. So $\Pr[L > \varepsilon'] \le \delta$ for every pair of neighbours, and @lem-shuf-tail finishes the proof.
:::

[Advanced composition](../part02-dp-basics/04-advanced-composition.md) gives $\varepsilon\sqrt{2k\ln(1/\delta)} + k\varepsilon\tanh(\varepsilon/2)$ for $k$ mechanisms that are only known to be $\varepsilon$-DP. The square-root term of @thm-shuf-br-composition is half as large. Its linear term $k\varepsilon^2/8$ is also smaller, about a quarter of $k\varepsilon\tanh(\varepsilon/2) \approx k\varepsilon^2/2$ for small $\varepsilon$. A numerical example shows the size of the gain.

::: {#exm-shuf-br}
Compose $k = 100$ exponential mechanisms, each with $\varepsilon = 0.1$, at $\delta = 10^{-6}$. Basic composition gives $10.00$, advanced composition $5.76$, and @thm-shuf-br-composition gives $2.75$.
:::

### Top-$k$ selection with Gumbel noise {#sec-shuf-gumbel}

An advertiser usually wants the few most popular items in a histogram, such as the top topics among an audience, not every count. LinkedIn's algorithms split along two lines. The **domain** of items may be known in advance, or not. A person's **contribution** may be **$m$-restricted**, meaning each person adds one to at most $m$ counts, or unrestricted. Hires are $1$-restricted, since a hire adds one to one employer, but skills are not, since one person may list many.

With $m$-restricted contributions, replacing one person changes the vector of counts by at most $2m$ in $\ell_1$ norm. Laplace noise of scale $2m/\varepsilon$ on every count, followed by taking the top $k$, is then $\varepsilon$-DP. Its noise grows with $m$. For unrestricted contributions, a better route is to select with the exponential mechanism, whose score for an item is its count. A count changes by at most one between neighbours, however many counts a person touches. **Peeling** selects the top item, removes it, and repeats $k$ times. Each round is the exponential mechanism, so by @prp-shuf-em-br and @thm-shuf-br-composition its cost depends on $k$ and not on $m$.

Sampling from the exponential mechanism needs its normalizing constant, and peeling needs it $k$ times. Adding noise of the right shape and taking a maximum avoids both. The right shape is the Gumbel distribution.

::: {#def-shuf-gumbel name="Gumbel distribution"}
The **Gumbel distribution** with scale $b > 0$ has cumulative distribution function $\Pr[G \le g] = \exp(-e^{-g/b})$ for real $g$.
:::

The next theorem is the reason for this choice. Adding independent Gumbel noise to each score and taking the largest gives exactly the exponential mechanism's distribution, and taking the $k$ largest gives exactly peeling.

::: {#thm-shuf-gumbel name="Gumbel-max trick"}
Let $u_1, \dots, u_d$ be scores, let $G_1, \dots, G_d$ be independent Gumbel variables with scale $b$, and write $w_i = e^{u_i/b}$. Then

1. the index of the largest noisy score $u_i + G_i$ is $i$ with probability $w_i/\sum_{j} w_j$;
2. the indices of the $k$ largest noisy scores, in decreasing order, have the distribution of $k$ rounds of peeling, where each round picks a remaining item $i$ with probability proportional to $w_i$.
:::

::: {.proof}
1. *Exponential race.* Let $E_i = e^{-(u_i + G_i)/b}$. Then $\Pr[E_i > t] = \Pr[G_i < -b\ln t - u_i] = \exp(-w_i t)$ for $t > 0$. So the $E_i$ are independent exponential variables with rates $w_i$, and the noisy scores in decreasing order are the $E_i$ in increasing order.
2. *The first finisher.* With $W = \sum_j w_j$, the chance that $E_i$ is the smallest is

$$
\int_0^\infty w_i e^{-w_i t}\prod_{j \ne i} e^{-w_j t}\,dt = \frac{w_i}{W}.
$$

This proves the first claim.
3. *Memorylessness.* Condition on $E_i$ being smallest, with value $t$. This is the event that $E_j > t$ for every $j \ne i$, and given it, the excesses $E_j - t$ are independent exponential variables with the original rates $w_j$, because $\Pr[E_j > t + s \mid E_j > t] = e^{-w_j s}$.
4. *Induction.* By step 3, the order of the remaining $E_j$ is a fresh exponential race among the remaining items. Step 2 applied to it shows that the next item is chosen with probability proportional to its $w_j$, and repeating gives the second claim.
:::

Counts change by at most one between neighbours, so take $b = 2/\varepsilon_0$. Then $w_i = e^{\varepsilon_0 u_i/2}$, and the first claim is the exponential mechanism with parameter $\varepsilon_0$. By the second claim, adding Gumbel noise once and reporting the $k$ largest noisy counts *is* peeling, so it inherits peeling's guarantee from @thm-shuf-br-composition. It draws $d$ noise values and selects the top $k$ once, in time of order $d$ plus the time to select the top $k$. Peeling would take time of order $kd$.

### Queries from outside {#sec-shuf-outside}

Unlike the deployments of the previous chapter, here outside analysts choose the queries, and they may be careless or malicious. Each advertiser has a privacy budget for its own queries. If two advertisers with budgets $\varepsilon$ each combined their answers, basic composition would only give $2\varepsilon$ about the members, and with many advertisers the guarantee would fade. The mechanisms alone cannot rule this out when each advertiser has its own budget. LinkedIn relies instead on a vetting process and on agreements in which advertisers undertake not to share results. The service also sets its parameters by modelling the attacks an analyst might mount, deliberately or by accident, and choosing $\varepsilon$ and the number of queries allowed per month so that those attacks fail. By that accounting, its monthly privacy loss is lower than the reported values for the local deployments of the previous chapter, though those systems protect very different data.

LinkedIn used the same algorithms for its public labour market insights, which show which employers are hiring, which job titles are growing, and which skills new hires have, by region and industry. Hires by employer and by title are $1$-restricted counts, so Laplace noise suits them. Skills are unrestricted, so their top lists come from Gumbel noise and peeling.

## Data releases {#sec-shuf-releases}

The last two deployments do not answer queries. They publish a fixed dataset once, computed in the central model.

### Mobility reports {#sec-shuf-mobility}

During the Covid-19 pandemic, Google published community mobility reports, built from the location history of users who had opted in. For each region and day, they gave the change in visits to six kinds of places, such as retail and recreation, grocery stores and pharmacies, parks, transit stations, workplaces and residences, relative to a baseline from before the pandemic. Weekly patterns are visible. Visits to workplaces fell on weekdays but hardly changed at weekends, when most people stay away from work anyway, and time at home rose. The counts behind these percentages are noised with Laplace noise to satisfy differential privacy. To bound each person's influence, a person counts at most once per category and place, and at most four such pairs per day. A person with more has four pairs kept at random and the rest dropped, which is the same idea as $m$-restricted contribution. Regions smaller than $3$ square kilometres, and metrics whose noisy count of people is below $100$, are discarded, and so are changes that the noise makes too uncertain.

### A URL dataset with zero-concentrated privacy {#sec-shuf-urls}

Facebook and the academic organization Social Science One released a dataset on how users interact with web links shared on Facebook. It covers more than 38 million URLs, and researchers obtain access through a proposal process. The raw data are events, such as "a woman aged 35 to 44 in Germany shared this URL on this date". The release aggregates them by URL, month, country, age and gender, and publishes noisy counts of views, clicks, shares, likes and other reactions. A noisy count can be negative, a visible sign of the noise.

The noise is Gaussian, and the guarantee is zero-concentrated differential privacy, a variant of differential privacy that suits Gaussian noise. It is stated with the Rényi divergence. For an order $\alpha > 1$ and two distributions $P$ and $Q$, write $p$ and $q$ for their densities, or their mass functions on a countable set. The **Rényi divergence** of order $\alpha$ is

$$
D_\alpha(P\,\|\,Q) = \frac{1}{\alpha - 1}\ln\int p(y)^\alpha q(y)^{1-\alpha}\,dy ,
$$

with a sum in place of the integral on a countable set. It is zero when $P = Q$ and grows as they move apart. Zero-concentrated privacy asks that the divergence between the outputs on neighbours grow at most linearly in the order $\alpha$.

::: {#def-shuf-zcdp name="Zero-concentrated differential privacy"}
Let $\rho \ge 0$. A mechanism $M$ is **$\rho$-zero-concentrated differentially private** if for all neighbours $D \sim D'$ and all $\alpha > 1$,

$$
D_\alpha\big(M(D)\,\|\,M(D')\big) \le \rho\,\alpha .
$$
:::

It has one parameter, like pure privacy, but composes like advanced composition, as the next three results show. It lies between the two: an $\varepsilon$-DP mechanism is $(\varepsilon^2/2)$-zero-concentrated, and the converse direction holds with a $\delta$.

::: {#lem-shuf-renyi-product}
For product distributions, $D_\alpha(P_1\times P_2\,\|\,Q_1\times Q_2) = D_\alpha(P_1\,\|\,Q_1) + D_\alpha(P_2\,\|\,Q_2)$. So releasing the outputs of a $\rho_1$- and a $\rho_2$-zero-concentrated mechanism, run with independent coins, is $(\rho_1 + \rho_2)$-zero-concentrated.
:::

::: {.proof}
The integral of $p^\alpha q^{1-\alpha}$ over a product space factorizes into the two integrals, and the logarithm turns the product into a sum.
:::

The same additivity holds when the second mechanism is chosen after seeing the first output, as Bun and Steinke (2016) show. The second result gives the guarantee of the Gaussian mechanism, which is why the release uses it.

::: {#prp-shuf-gaussian name="Gaussian mechanism"}
Let $f\colon \mathcal X^n \to \mathbb R^d$ change by at most $\Delta_2 f$ in Euclidean norm between neighbours. Then $M(D) = f(D) + Z$, with $Z$ a vector of $d$ independent normal variables of mean $0$ and variance $\sigma^2$, is $(\Delta_2 f)^2/(2\sigma^2)$-zero-concentrated.
:::

::: {.proof}
1. In one dimension, let $p$ and $q$ be normal densities with means $a$ and $b$ and variance $\sigma^2$. Completing the square,

$$
\alpha(y-a)^2 + (1-\alpha)(y-b)^2 = \big(y - \alpha a - (1-\alpha)b\big)^2 + \alpha(1-\alpha)(a-b)^2 .
$$

2. The first term integrates against the normal density to one, so $\int p^\alpha q^{1-\alpha} = \exp\big(\alpha(\alpha-1)(a-b)^2/(2\sigma^2)\big)$, and $D_\alpha = \alpha(a-b)^2/(2\sigma^2)$.
3. In $d$ dimensions the coordinates are independent, so by @lem-shuf-renyi-product the divergences add to $\alpha\|f(D) - f(D')\|_2^2/(2\sigma^2) \le \alpha(\Delta_2 f)^2/(2\sigma^2)$.
:::

The third result converts a zero-concentrated guarantee into an $(\varepsilon, \delta)$ guarantee, the form in which a release usually states its budget.

::: {#prp-shuf-zcdp-dp}
A $\rho$-zero-concentrated mechanism is $(\rho + 2\sqrt{\rho\ln(1/\delta)}, \delta)$-DP for every $\delta \in (0,1)$.
:::

::: {.proof}
1. If $\rho = 0$, every Rényi divergence is $0$, so $M(D)$ and $M(D')$ have the same distribution and the claim holds. Otherwise let $y \sim M(D)$, with density $p$, and let $L = \ln(p(y)/q(y))$, where $q$ is the density of $M(D')$. For $\alpha > 1$, $\mathbb E[e^{(\alpha-1)L}] = \int p^\alpha q^{1-\alpha} = e^{(\alpha-1)D_\alpha} \le e^{(\alpha-1)\alpha\rho}$.
2. Markov's inequality gives $\Pr[L > t] \le e^{(\alpha-1)(\alpha\rho - t)}$.
3. Take $t = \rho + 2\sqrt{\rho\ln(1/\delta)}$ and $\alpha = 1 + \sqrt{\ln(1/\delta)/\rho}$. Writing $\ell = \ln(1/\delta)$, the exponent is $\sqrt{\ell/\rho}\,\big(\rho + \sqrt{\rho\ell} - \rho - 2\sqrt{\rho\ell}\big) = -\ell$, so $\Pr[L > t] \le \delta$.
4. @lem-shuf-tail finishes the proof.
:::

The release's budget raises a question that recurs in deployments: what is a neighbour? **User-level** privacy hides everything one person did, and **action-level** privacy hides one action, such as one click. User-level privacy is the goal, but a few users act thousands of times, and hiding them would drown every count in noise. The release instead protects every action strongly enough that almost all users get a user-level guarantee too. A user whose actions number at most a high percentile of the activity distribution enjoys a stated user-level guarantee, and heavier users still have the action-level one.

::: {#exm-shuf-budget}
Suppose each action adds one to a single cell, and the actions of one user fall in different cells. Replacing one action changes two counts by one each, a Euclidean change of $\sqrt 2$. Replacing all $k$ actions of a user changes at most $2k$ counts, a change of at most $\sqrt{2k}$. Then by @prp-shuf-gaussian, $k$ actions cost $k$ times the action-level $\rho$. To give users with at most $20$ actions $(0.5, 10^{-6})$-DP, solve $\rho + 2\sqrt{\rho\ln 10^6} = 0.5$ for $\rho = 0.00444$ at the user level. The action level is then $\rho = 0.000222$, which Gaussian noise of standard deviation $\sigma = \sqrt2/\sqrt{2\rho} = 67.1$ achieves. A single action alone is protected at $(0.111, 10^{-6})$-DP.
:::

Many other deployments exist. The largest is the US Census Bureau's use of differential privacy for the 2020 census, a system complex enough to need a separate treatment.

## Summary {#sec-shuf-summary}

| Result | Statement |
| --- | --- |
| @def-shuf-model | the adversary sees the messages in a uniformly random order, not who sent which |
| @thm-shuf-amplification | shuffling turns $\varepsilon_0$-local into $\big(12\varepsilon_0\sqrt{\ln(1/\delta)/n}, \delta\big)$-DP, for $\varepsilon_0 < \frac12$ and $n \ge 1000$ |
| @thm-shuf-rr-privacy | randomized response with $\nu = 16\ln(4/\delta)/(\tau^2(n-1))$ is $(\varepsilon,\delta)$-DP once shuffled |
| @thm-shuf-rr-accuracy | its count has error $O(\sqrt{\ln(1/\delta)/\beta}/\varepsilon)$, independent of $n$ |
| @prp-shuf-br-dp | $\varepsilon$-bounded range implies $\varepsilon$-DP, which implies $2\varepsilon$-bounded range |
| @prp-shuf-em-br | the exponential mechanism is $\varepsilon$-bounded range |
| @thm-shuf-br-composition | $k$ bounded-range steps cost $k\varepsilon^2/8 + \varepsilon\sqrt{k\ln(1/\delta)/2}$ |
| @thm-shuf-gumbel | Gumbel noise and a maximum sample the exponential mechanism, and the top $k$ is peeling |
| @prp-shuf-gaussian | Gaussian noise with sensitivity $\Delta_2 f$ is $(\Delta_2 f)^2/(2\sigma^2)$-zero-concentrated |
| @prp-shuf-zcdp-dp | $\rho$-zero-concentrated implies $(\rho + 2\sqrt{\rho\ln(1/\delta)}, \delta)$-DP |

: The chapter's results. {#tbl-shuf-summary}

## References {#sec-shuf-notes}

- Andrea Bittau, Úlfar Erlingsson, Petros Maniatis, Ilya Mironov, Ananth Raghunathan, David Lie, Mitch Rudominer, Ushasree Kode, Julien Tinnes and Bernhard Seefeld, "Prochlo: Strong Privacy for Analytics in the Crowd", 2017. The encode, shuffle, analyse architecture and its implementation with hardware enclaves.
- Albert Cheu, Adam Smith, Jonathan Ullman, David Zeber and Maxim Zhilyaev, "Distributed Differential Privacy via Shuffling", 2019. The shuffle model as defined here, a one-message protocol for sums of bits with error $O(\sqrt{\ln(1/\delta)}/\varepsilon)$, and a protocol with many one-bit messages for sums of numbers in $[0,1]$ with error $O(\ln(n/\delta)/\varepsilon)$. The constants in @thm-shuf-rr-privacy are this chapter's own.
- Úlfar Erlingsson, Vitaly Feldman, Ilya Mironov, Ananth Raghunathan, Kunal Talwar and Abhradeep Thakurta, "Amplification by Shuffling: From Local to Central Differential Privacy via Anonymity", 2019. @thm-shuf-amplification.
- Vitaly Feldman, Audra McMillan and Kunal Talwar, "Hiding Among the Clones: A Simple and Nearly Optimal Analysis of Privacy Amplification by Shuffling", 2021. Amplification for $\varepsilon_0$ up to about $\ln n$, by the clone decomposition.
- Ryan Rogers, Subbu Subramaniam, Sean Peng, David Durfee, Seunghyun Lee, Santosh Kumar Kancha, Shraddha Sahay and Parvez Ahammad, "LinkedIn's Audience Engagements API: A Privacy Preserving Data Analytics System at Scale", 2020. The query service, its algorithms for known and unknown domains, and its choice of parameters.
- David Durfee and Ryan Rogers, "Practical Differentially Private Top-$k$ Selection with Pay-what-you-get Composition", 2019. Bounded range, Gumbel-noise top-$k$ selection and its equivalence with peeling.
- Jinshuo Dong, David Durfee and Ryan Rogers, "Optimal Differential Privacy Composition for Exponential Mechanisms", 2020. The optimal composition of bounded-range mechanisms, and efficiently computable bounds that improve slightly on @thm-shuf-br-composition.
- Emil Julius Gumbel, *Statistical Theory of Extreme Values and Some Practical Applications*, 1954. The Gumbel distribution and its use for maxima.
- Ryan Rogers, Adrian Rivera Cardoso, Koray Mancuhan, Akash Kaura, Nikhil Gahlawat, Neha Jain, Paul Ko and Parvez Ahammad, "A Members First Approach to Enabling LinkedIn's Labor Market Insights at Scale", 2020. The labour market release and its choice of algorithms.
- Ahmet Aktay and coauthors, "Google Covid-19 Community Mobility Reports: Anonymization Process Description", 2020. How the mobility reports bound contributions and add noise.
- Solomon Messing, Christina DeGregorio, Bennett Hillenbrand, Gary King, Saurav Mahanti, Zagreb Mukerjee, Chaya Nayak, Nate Persily, Bogdan State and Arjun Wilkins, "Facebook Privacy-Protected Full URLs Data Set", 2020. The URL release, its Gaussian noise and its action-level and user-level budget, set so that the 99th percentile of users by activity gets the user-level guarantee.
- Cynthia Dwork and Guy N. Rothblum, "Concentrated Differential Privacy", 2016, and Mark Bun and Thomas Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds", 2016. Concentrated privacy, and its zero-concentrated form with @prp-shuf-gaussian, @prp-shuf-zcdp-dp and adaptive composition.
- John M. Abowd and coauthors, "The 2020 Census Disclosure Avoidance System TopDown Algorithm", 2022. The census deployment.
