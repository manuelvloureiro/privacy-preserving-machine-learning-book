# Local, shuffle and distributed trust models

## Why this matters

Every mechanism so far in this part assumed a **trusted curator**: someone who holds the raw data, runs the mechanism, and releases only its output. The guarantee of differential privacy (DP) is about that output. It says nothing about the curator, the database it keeps, the engineer who can read it, or the court order that can demand it. [Threat models](../part02-attacks/01-threat-models.md) put it in one line: trust decides who the adversary is. In the central model the curator is outside the threat model. In many deployments it cannot be. A browser vendor collecting usage statistics, a keyboard app learning new words, or a phone operating system counting crashes is exactly the party users do not want to see their data.

This chapter answers one question: **how much accuracy does each weaker trust assumption cost, and what do you have to trust instead?** There are four answers, drawn in @fig-trust-models. They are four parallel options, so they are listed:

- **Central.** Trust the curator with everything. The noise is added once, to the answer, and the error does not grow with the number of people $n$.
- **Local.** Trust nobody. Each person randomizes their own record before it leaves their device. The noise is added $n$ times, and for counting the error is a factor of about $\sqrt n$ worse than central. No protocol can avoid this.
- **Shuffle.** Trust a shuffler only to cut the link between reports and senders. Local reports become anonymous, and anonymity amplifies privacy: reports randomized at a weak local level $\varepsilon_0$ give a strong central $\varepsilon$, so the error is close to central again.
- **Distributed.** Trust cryptography. Each person adds a small share of the noise, and a secure aggregation protocol reveals only the sum. The total noise equals the central noise, as long as enough people are honest.

![The four trust models. Blue boxes hold one record each, green boxes must be trusted, and red boxes see what is released and are not trusted. Only the central model gives raw records to a single party.](figures/trust_models.svg){#fig-trust-models}

### The one idea: where the noise enters

All four models release a noisy statistic. They differ only in where the noise is added and who sees the data before it is. The further upstream the noise enters, the fewer parties you trust, and the more noise reaches the answer.

Here is why. A local report has to hide its owner from everyone, including the analyst, so each report carries noise of the size the central mechanism adds once in total. When you add up $n$ independent noises, their variances add, so the total noise has standard deviation of order $\sqrt n$ times one report's. The signal, the true count, grows like $n$. So the relative error of a local count shrinks only like $\sqrt n/n = 1/\sqrt n$, while a central count with one dose of noise has relative error of order $1/n$. The shuffle and distributed models move the noise back downstream without giving the data back to one party: anonymity hides which report is whose, and cryptography hides everything except the sum.

@tbl-trust-uses lists the results and where each one is used.

| Result | Where it is used |
| --- | --- |
| A local randomizer makes the whole protocol $\varepsilon$-DP | Deciding what a locally private deployment promises, and against whom |
| Unbiased estimation from randomized reports | Every local frequency estimate; debiasing any linear randomizer |
| The local lower bound: error $\gtrsim 1/((e^\varepsilon - 1)\sqrt n)$ | Knowing when local DP is hopeless, and why the other models exist |
| $k$-ary randomized response and unary encoding | Frequency estimation over large domains, as in browser and phone telemetry |
| Amplification by shuffling (Feldman, McMillan & Talwar 2021) | Choosing the local $\varepsilon_0$ that a shuffler turns into a target central $\varepsilon$ |
| Distributed noise with secure aggregation | Central accuracy without a trusted curator, in federated settings; how collusion erodes it |

: The results of this chapter and where each one is used. {#tbl-trust-uses}

The chapter uses the definition, the hockey-stick form and randomized response from [The definition of differential privacy](01-definition.md), post-processing from [Post-processing, group privacy and composition](02-properties.md), and the [Laplace and Gaussian mechanisms](03-laplace-gaussian.md). From Part 1 it uses concentration ([Concentration inequalities](../part01-prerequisites/02-concentration.md)), Pinsker's inequality ([Divergences](../part01-prerequisites/03-divergences.md)) and additive secret sharing ([Cryptography basics](../part01-prerequisites/08-crypto-basics.md)). Each of them is restated where it is used. The derivation of amplification by shuffling belongs to [Privacy accounting](../part04-privacy-accounting/00-introduction.md); here it is a cited result that you compute with.

**The guarantee, restated.** A randomized mechanism $M$ is **$(\varepsilon, \delta)$-DP** if, for every pair of neighbouring datasets $D, D'$ and every set $S$ of possible outputs, $\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta$. In words: changing one person's record changes the probability of any outcome by at most a factor $e^\varepsilon$, up to a small slack $\delta$. With $\delta = 0$ this is **pure** $\varepsilon$-DP.

**Notation.** $n$ people hold records $x_1, \dots, x_n$ from a domain $\mathcal X$. A **local randomizer** $R$ maps one record to a random report, and $p_R(y \mid x)$ is the probability that it outputs report $y$ when the record is $x$. When local and central guarantees appear together, $\varepsilon_0$ is the local level of one report and $(\varepsilon, \delta)$ the central guarantee of everything released. Neighbouring datasets differ by **replacing** one record, in every model of this chapter. The reason is that a report has to hide its owner's value, while the number of reports already reveals who took part, so no randomizer of the value could hide participation; [The definition of differential privacy](01-definition.md) makes the same point for randomized response. Frequencies are fractions in $[0, 1]$; an error on a fraction becomes $n$ times larger on a count.

## Toolkit: three facts every derivation uses

### Fact 1 — Debiasing a linear randomizer

A randomizer shrinks the signal: the mean of a report moves less with the true value than the true value itself does. If you know by how much it shrinks, you can undo it. Precisely, suppose a report $Y$ has mean $\mathbb E[Y] = a + b\,\theta$ for an unknown quantity $\theta$ and known numbers $a$ and $b \ne 0$. Then subtracting the offset and dividing by the slope,
$$\hat\theta = \frac{Y - a}{b},$$
is unbiased for $\theta$, with $\operatorname{Var}\hat\theta = \operatorname{Var}(Y)/b^2$.

The proof is two lines. Linearity of expectation gives $\mathbb E\hat\theta = (a + b\theta - a)/b = \theta$. Subtracting a constant does not change a variance, and scaling by $1/b$ multiplies it by $1/b^2$.

Every local estimator in this chapter is of this form. The slope $b$ is the gap between "report 1 when the true value is $v$" and "report 1 when it is not". A strongly private randomizer makes that gap small, and the estimator then has to scale the report up by $1/b$, which multiplies the noise variance by $1/b^2$. That is where the cost of local privacy comes from.

### Fact 2 — Variances of independent reports add

For independent reports $Y_1, \dots, Y_n$, the variance of a sum is the sum of the variances, and dividing by $n$ divides the variance by $n^2$:
$$\operatorname{Var}\Big(\frac1n\sum_iY_i\Big) = \frac{1}{n^2}\sum_i\operatorname{Var}Y_i.$$
A Bernoulli report, which is 1 with probability $\pi_i$ and 0 otherwise, has variance $\pi_i(1 - \pi_i)$. Together with Fact 1, this gives the exact variance of every frequency estimate below, with no approximation.

### Fact 3 — A test built from an estimator

A good estimator can tell two hypotheses apart, so if two hypotheses are hard to tell apart, no estimator can be good. This fact turns that into a number, and it is how the chapter proves lower bounds on error.

It uses the **total variation (TV) distance** between two distributions $P$ and $Q$ of the data: half the total absolute difference between their probability tables, $\mathrm{TV}(P, Q) = \frac12\sum_z|P(z) - Q(z)|$. From [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md), every test has false positive rate plus false negative rate at least $1 - \mathrm{TV}$. The reason is short: a test that decides "$Q$" on a set $A$ of data has false positive rate $P(A)$ and false negative rate $1 - Q(A)$, so their sum is $1 - (Q(A) - P(A))$, and $Q(A) - P(A)$ can never exceed TV, which is the largest such difference.

Now the fact. Suppose an estimator $\hat\theta$ is within $\alpha/2$ of the truth with probability at least $1 - \beta$, both when the truth is $\theta_0$ and when it is $\theta_1 = \theta_0 + \alpha$. Use it to test: decide $\theta_1$ if $\hat\theta > \theta_0 + \alpha/2$, the midpoint between the two values. When the truth is $\theta_0$, the test is wrong only if the estimate lands more than $\alpha/2$ above the truth, which happens with probability at most $\beta$. When the truth is $\theta_1$, the test is wrong only if the estimate lands at least $\alpha/2$ below it, again with probability at most $\beta$ (an estimate exactly at the midpoint can be sent either way; with continuous estimates it has probability 0). So both error rates are at most $\beta$, their sum is at most $2\beta$, and the bound $1 - \mathrm{TV} \le 2\beta$ gives that the TV distance between the two distributions of the data is at least $1 - 2\beta$.

Read the other way round, this turns an upper bound on how much $n$ reports can reveal (a small TV) into a lower bound on any estimator's error.

## Central and local differential privacy

### Context

In the central model a curator holds $D = (x_1, \dots, x_n)$ and releases $M(D)$. In the **local model** there is no curator. Each person runs a randomizer on their own record and sends only the result. The question of this section: what does a local randomizer promise, to whom, and what does the change cost on the simplest task, counting?

### Statement

A local randomizer has to make each person's report look about the same whatever their record is, because whoever receives it may know whose it is. So the requirement is the DP definition applied to a dataset of one record, with every pair of records as neighbours. A randomizer $R: \mathcal X \to \mathcal Y$ is an **$\varepsilon_0$-DP local randomizer** if for every pair of records $x, x' \in \mathcal X$ and every event $S \subseteq \mathcal Y$,
$$\Pr[R(x) \in S] \le e^{\varepsilon_0}\Pr[R(x') \in S].$$ {#eq-trust-ldp}
This is the [definition of differential privacy](01-definition.md) for one record. The **local protocol** releases the transcript $(R(x_1), \dots, R(x_n))$, with independent randomness for each person. Then:

1. The transcript is $\varepsilon_0$-DP in the central sense, under replace.
2. Each report on its own is $\varepsilon_0$-DP against anyone who sees it, including the party that collects it.

### Derivation

The idea is that replacing one person's record changes only that person's report, so the privacy of the whole transcript is the privacy of one report.

1. **The transcript factorizes.** People use independent randomness, and the probability of independent events is the product of their probabilities: $p(y_1, \dots, y_n \mid D) = \prod_i p_R(y_i \mid x_i)$.
2. **Neighbours differ in one factor.** If $D$ and $D'$ differ only in record $j$, every factor with $i \ne j$ is the same under both and cancels in the ratio of the two probabilities. What remains, the **privacy loss** of the transcript, is $\ln\frac{p_R(y_j \mid x_j)}{p_R(y_j \mid x_j')}$. Applying @eq-trust-ldp to the single outcome $S = \{y_j\}$, in both directions, bounds it by $\varepsilon_0$ in absolute value. A bound on every single outcome's ratio extends to every event by adding the outcomes in the event. This is the argument [The definition of differential privacy](01-definition.md) made for randomized response, for any $R$.
3. **Who sees the report does not matter.** @eq-trust-ldp involves only the distribution of one report. It holds for the network, the collector, the analyst and anyone they hand the data to. Any later processing of the reports, including the curator's estimate, is covered by **post-processing** ([Post-processing, group privacy and composition](02-properties.md)): a function applied to a DP output, without access to the data, is DP with the same parameters, because it can only blur the output's dependence on the data, never sharpen it.

### Interpretation

The guarantee no longer depends on the curator. A breach of the collector's database, a malicious insider or a subpoena reveals reports, and reports are already private. That is why the large deployments of the 2010s (Google's RAPPOR in Chrome, Apple's telemetry in iOS and macOS) used local DP.

But every person pays separately. Each report must hide its owner from everyone, so each carries noise of the size the central mechanism adds once in total. @tbl-trust-who sets out what you trust in each model and what it costs. The error column is derived in the sections below; the local entry follows from the example right after the table, where the standard deviation of a count is $n$ times that of a fraction, $n \cdot 1/(\varepsilon\sqrt n) = \sqrt n/\varepsilon$.

| Model | Who must be trusted | What the untrusted party sees | Error of a count, $\varepsilon$ fixed |
| --- | --- | --- | --- |
| Central | The curator, with all raw records | $M(D)$ | Order $1/\varepsilon$, independent of $n$ |
| Local | Nobody | Every report $R(x_i)$ | Order $\sqrt n/\varepsilon$ |
| Shuffle | The shuffler, to unlink reports from senders | The multiset of reports | Order $\sqrt{\ln(1/\delta)}/\varepsilon$ |
| Distributed | The cryptography, and enough honest clients | The noisy sum | Order $\sqrt{\ln(1/\delta)}/\varepsilon$ |

: The four trust models: who is trusted, what the untrusted party sees, and the order of the error of a count. {#tbl-trust-who}

### Example: one survey, two trust models

Of $n = 10{,}000$ people, 3,000 hold a 1. Each person uses **randomized response** at $\varepsilon = \ln 3$: report the true bit with the keep probability $t = e^\varepsilon/(1 + e^\varepsilon) = 3/4 = 0.75$, and the flipped bit otherwise. The mean $\bar y$ of the reports has expectation $(1 - t) + \pi(2t - 1)$, where $\pi$ is the true fraction of ones, so by Fact 1 the estimate $\hat\pi = (\bar y - (1 - t))/(2t - 1)$ is unbiased. Every report is a Bernoulli variable with variance $t(1 - t)$, whatever its true bit, so by Fact 2 and Fact 1 its standard deviation is $\sqrt{t(1-t)}/((2t - 1)\sqrt n)$, which [The definition of differential privacy](01-definition.md) rewrites as $\sqrt{e^\varepsilon/(n(e^\varepsilon - 1)^2)}$. With $e^\varepsilon = 3$ and $(e^\varepsilon - 1)^2 = 4$ this is $\sqrt{0.75/10{,}000} = 0.00866$. In 20,000 simulated surveys the mean estimate is 0.3000 and the standard deviation 0.00861. On the count, that is a standard deviation of $10{,}000 \times 0.00866 = 86.6$ people.

A trusted curator would instead release the true count plus Laplace noise $\mathrm{Lap}(1/\varepsilon)$. The **Laplace mechanism** adds noise of scale $b = \Delta_1/\varepsilon$, where the sensitivity $\Delta_1$ is the most one person can move the answer; under replace, changing one bit moves the count by at most 1, so $b = 1/\varepsilon$. A $\mathrm{Lap}(b)$ variable has variance $2b^2$, so the standard deviation is $\sqrt2/\varepsilon = 1.4142/1.0986 = 1.287$. The local survey is $86.6/1.287 = 67.3$ times noisier, at the same $\varepsilon$, on the same people.

### Check yourself

- A company runs randomized response at $\varepsilon_0 = 1$ on each user's device, and then also stores each user's network address with the report. Is the report still $1$-DP? *(As a function of the sensitive bit, yes: @eq-trust-ldp is about the bit. But the stored address links the report to the person, so anything else learned about them later can be combined with it. The guarantee covers the randomized value, not the metadata.)*
- Why does local DP use the replace relation and not add/remove? *(Each report is sent by a known device. Whether a report exists reveals participation, which no randomizer of the value can hide.)*

## The price of local privacy: a $\sqrt n$ gap

### Context

The example showed a factor of 67 at $n = 10{,}000$. Is that a weakness of randomized response, or of the local model? This section shows it is the model: no locally private protocol can estimate a fraction with error below order $1/((e^{\varepsilon_0} - 1)\sqrt n)$, while the central Laplace mechanism reaches order $1/(n\varepsilon)$. The ratio grows like $\sqrt n$.

### Statement

**Randomized response (upper bound).** The calculation of the previous example, at a general level $\varepsilon_0$, gives a standard deviation that does not depend on the data, because every report has the same variance $t(1-t)$ whatever its true bit. For every dataset, the unbiased estimate of a fraction has
$$\operatorname{sd}(\hat\pi) = \frac{e^{\varepsilon_0/2}}{(e^{\varepsilon_0} - 1)\sqrt n}.$$

**Any local protocol (lower bound).** The lower bound needs a way to measure how much a set of reports reveals. It uses the **Kullback–Leibler (KL) divergence**, $D_{\mathrm{KL}}(M_0\|M_1) = \sum_y m_0(y)\ln\frac{m_0(y)}{m_1(y)}$: the expected log-likelihood ratio in favour of $M_0$ when the data come from $M_0$, which is 0 when the two distributions are equal and grows as they separate. KL is convenient because it adds over independent observations.

The key step is that a local randomizer contracts this information. Let each person's record be drawn independently from a population $P_0$ or $P_1$, and let each person apply any $\varepsilon_0$-DP local randomizer (possibly a different one per person). Because the randomizer can change each report's probability by at most a factor $e^{\varepsilon_0}$, the report distributions $M_0, M_1$ under the two populations can differ only a little, however different the populations are:
$$D_{\mathrm{KL}}(M_0\|M_1) \le (e^{\varepsilon_0} - 1)^2\,\mathrm{TV}(P_0, P_1)^2.$$ {#eq-trust-kl-contraction}
As a consequence, $n$ reports cannot tell apart two populations whose fractions are close. Take $\mathrm{Bern}(\tfrac12)$ and $\mathrm{Bern}(\tfrac12 + \alpha)$, populations in which a fraction $\frac12$ or $\frac12 + \alpha$ hold a 1. No estimator built from $n$ reports is within $\alpha/2$ of the true fraction with probability $1 - \beta$ under both, unless
$$\frac\alpha2 \ge \frac{1 - 2\beta}{(e^{\varepsilon_0} - 1)\sqrt{2n}}.$$ {#eq-trust-lower}

### Derivation

The upper bound is the randomized-response calculation of the previous example, which used Facts 1 and 2. The lower bound has two halves. Steps 1 to 4 show that one report carries little information about the population (@eq-trust-kl-contraction). Steps 5 to 7 add up $n$ reports and turn "little information" into "large error" with Fact 3. Write $m_j(y) = \sum_xP_j(x)\,p_R(y \mid x)$ for the probability of report $y$ under population $j$: draw a record from $P_j$, then randomize it. Assume a finite domain for simplicity.

1. **Pin the report probabilities in a band.** Fix a report $y$ and let $r(y) = \min_xp_R(y \mid x)$ be its smallest probability over all records. By @eq-trust-ldp, no record makes $y$ more than $e^{\varepsilon_0}$ times as likely as any other, so every $p_R(y \mid x)$ lies in $[r(y), e^{\varepsilon_0}r(y)]$. Subtracting the floor, $g(x) = p_R(y \mid x) - r(y)$ lies in $[0, (e^{\varepsilon_0} - 1)r(y)]$.
2. **Only the difference of the populations matters.** Both populations are probability distributions, so $\sum_x(P_0(x) - P_1(x)) = 1 - 1 = 0$. Subtracting the constant $r(y)$ from $p_R(y \mid x)$ therefore changes nothing: $m_0(y) - m_1(y) = \sum_x(P_0(x) - P_1(x))\,p_R(y \mid x) = \sum_x(P_0(x) - P_1(x))\,g(x)$.
3. **Bound the sum.** Since the differences $P_0(x) - P_1(x)$ add to 0, their positive parts and their negative parts have the same total size, and each total is half the sum of absolute values, which is $\mathrm{TV}(P_0, P_1)$. With $g$ between 0 and $G = (e^{\varepsilon_0} - 1)r(y)$, the sum is largest when $g = G$ on the positive parts and $g = 0$ on the negative ones, giving $\mathrm{TV}\cdot G$, and smallest the other way round, giving $-\mathrm{TV}\cdot G$. Finally, $r(y) \le m_1(y)$, because $m_1(y)$ is an average of the numbers $p_R(y \mid x)$, each at least $r(y)$. So
$$|m_0(y) - m_1(y)| \le (e^{\varepsilon_0} - 1)\,\mathrm{TV}(P_0, P_1)\,m_1(y).$$
4. **From the pointwise bound to KL.** The tool is the inequality $\ln u \le u - 1$, which holds because the line $u - 1$ is tangent to the concave curve $\ln u$ at $u = 1$ and a concave curve lies below its tangents. Applied to $u = m_0/m_1$:
$$D_{\mathrm{KL}}(M_0\|M_1) = \sum_ym_0\ln\frac{m_0}{m_1} \le \sum_ym_0\Big(\frac{m_0}{m_1} - 1\Big) = \sum_y\frac{(m_0 - m_1)^2}{m_1} \le (e^{\varepsilon_0} - 1)^2\,\mathrm{TV}^2\sum_ym_1.$$
The middle equality holds because both sides equal $\sum_ym_0^2/m_1 - 1$: the left is $\sum m_0^2/m_1 - \sum m_0$, and expanding the square on the right gives $\sum m_0^2/m_1 - 2\sum m_0 + \sum m_1$, with $\sum m_0 = \sum m_1 = 1$. The last inequality squares step 3. The last sum is 1, which is @eq-trust-kl-contraction.
5. **Add over people.** Records and randomness are independent across people, so the reports are independent. For independent observations the log of the joint likelihood ratio is the sum of the individual log ratios, so the KL divergence, its expectation, adds ([Divergences](../part01-prerequisites/03-divergences.md)). The two Bernoulli populations differ by $\alpha$ in the probability of a 1 and by $\alpha$ in the probability of a 0, so $\mathrm{TV}(\mathrm{Bern}(\frac12), \mathrm{Bern}(\frac12 + \alpha)) = \frac12(\alpha + \alpha) = \alpha$. So $n$ reports have KL at most $n(e^{\varepsilon_0} - 1)^2\alpha^2$.
6. **Pinsker.** Fact 3 needs TV, not KL. **Pinsker's inequality** ([Divergences](../part01-prerequisites/03-divergences.md)) converts one into the other: $\mathrm{TV} \le \sqrt{\mathrm{KL}/2}$ for any two distributions. TV does not add over independent samples, which is why the derivation goes through KL. Here $\mathrm{TV}(M_0^n, M_1^n) \le \sqrt{\mathrm{KL}/2} \le \alpha(e^{\varepsilon_0} - 1)\sqrt{n/2}$, where $M_j^n$ is the distribution of all $n$ reports.
7. **Fact 3.** An estimator within $\alpha/2$ with probability $1 - \beta$ under both populations forces $\mathrm{TV} \ge 1 - 2\beta$. Combining with step 6, $1 - 2\beta \le \alpha(e^{\varepsilon_0} - 1)\sqrt{n/2}$, so $\alpha \ge (1 - 2\beta)\sqrt2/((e^{\varepsilon_0} - 1)\sqrt n)$. Halving both sides, and using $\sqrt2/2 = 1/\sqrt2$, gives @eq-trust-lower.

### Interpretation

Randomized response is optimal up to a constant. Divide its standard deviation by the lower bound: the factor $(e^{\varepsilon_0} - 1)\sqrt n$ cancels, and what is left is $e^{\varepsilon_0/2}\sqrt2/(1 - 2\beta)$. At $\varepsilon_0 = 1$ and $\beta = 0.1$ this is $1.6487 \times 1.4142/0.8 = 2.91$, the same as the ratio $3.03\times10^{-2}/1.04\times10^{-2}$ of the two values in @tbl-trust-results. No local protocol can do more than a constant factor better.

Local noise is a contraction. @eq-trust-kl-contraction says a local randomizer shrinks the information about the population by a factor $(e^{\varepsilon_0} - 1)^2 \approx \varepsilon_0^2$ for small $\varepsilon_0$. Since KL adds over people, you need about $1/\varepsilon_0^2$ times as many people to learn the same thing. Centrally, privacy costs an additive $1/(n\varepsilon)$ in the error of a fraction, which vanishes faster than the sampling error $1/\sqrt n$ that any survey has anyway.

The bound is loose by a constant. For one randomized-response report at $\varepsilon_0 = 1$, comparing $\mathrm{Bern}(0.5)$ with $\mathrm{Bern}(0.6)$, the exact KL is 0.00429, and @eq-trust-kl-contraction gives $(e - 1)^2 \times 0.1^2 = 2.952 \times 0.01 = 0.02952$, about seven times more. The slack comes from step 1: it allows every report probability to sit anywhere in the band $[r(y), e^{\varepsilon_0}r(y)]$, and randomized response does not use all of it. Duchi, Jordan & Wainwright (2018) prove the same contraction with a factor 4 and use it for minimax rates (the best worst-case error any estimator can reach) of many local estimation problems; Chan, Shi & Song (2012) prove the matching $\Omega(\sqrt n)$ lower bound for counting.

Interaction does not help much. The derivation allowed each person a different randomizer. Lower bounds of the same order also hold for **sequentially interactive** protocols, in which each person still speaks once but their randomizer may be chosen after seeing the earlier reports.

### Example: how the gap grows

@fig-trust-rr-gap plots the three curves at $\varepsilon = 1$: randomized response, the lower bound, and the central Laplace mechanism, whose error on a fraction is the count's $\sqrt2/\varepsilon$ divided by $n$. At $n = 1{,}000$, randomized response has standard deviation $3.03\times10^{-2}$, the central Laplace mechanism $\sqrt2/1000 = 1.41\times10^{-3}$, a ratio of 21. Dividing the two formulas, the ratio is $e^{1/2}\sqrt n/(\sqrt2(e - 1)) = 0.678\sqrt n$, so it grows like $\sqrt n$: at $n = 100{,}000$ the ratio is 215, and at $n = 10^7$ it is 2146. Each factor of 100 in $n$ multiplies it by $\sqrt{100} = 10$. For a standard deviation of 0.001 at $\varepsilon = 1$, randomized response needs $(e^{1/2}/((e - 1) \times 0.001))^2$, that is 920,674 people, and the central Laplace mechanism needs $\sqrt2/n \le 0.001$, that is 1,415 ([Exercise 3](solutions/08-trust-models.md#sec-sol-trust-3)). This is why local DP is used for the most frequent items of very large populations, and almost never for anything finer.

![Standard deviation of an estimated fraction at $\varepsilon = 1$, against the number of people. Randomized response (line and simulated dots) falls like $1/\sqrt n$ and stays a constant factor above the lower bound for any local protocol; the central Laplace mechanism falls like $1/n$.](figures/trust_rr_gap.svg){#fig-trust-rr-gap}

### Check yourself

- In steps 2 and 3, why may $g$ be shifted by $r(y)$, a quantity that depends on $y$? *(For fixed $y$ it is a constant in $x$, and $\sum_x(P_0 - P_1)(x)\cdot c = 0$ for any constant $c$.)*
- The central Laplace mechanism at $\varepsilon = 1$ has fraction error $\sqrt2/n$. At what $n$ does the sampling error $\sqrt{\pi(1-\pi)/n}$ of a fraction near $\frac12$ exceed it, and what does that say about the cost of central privacy? *(Near $\frac12$ the sampling error is $\sqrt{0.25/n} = 0.5/\sqrt n$. It exceeds $\sqrt2/n$ when $0.5/\sqrt n > \sqrt2/n$, that is $\sqrt n > 2\sqrt2$, or $n > 8$. Beyond a handful of people, central privacy costs less than sampling already does.)*

## Frequency estimation: $k$-ary randomized response and unary encoding

### Context

Counting one bit is the simplest task. The workhorse of deployed local DP is **frequency estimation**: each person holds one of $k$ values (a home page, an emoji, a crashing app), and the analyst wants the fraction $f_v$ of people holding each value $v$. Two natural generalizations of randomized response behave very differently as $k$ grows.

### Statement

Both schemes fit one template, and analysing the template once gives both. The idea is that each report either "supports" value $v$ or not, and supports it more often when the person really holds $v$. Then the number of supporting reports is a shrunken version of the true count, and Fact 1 unshrinks it.

**A pure frequency oracle.** Suppose that for each value $v$ there is a 0/1 "support" event in the report that has probability $p$ when the person holds $v$ and $q < p$ otherwise, independently across people. Let $C_v$ be the number of reports that support $v$. Then the debiased estimate and its exact variance are
$$\hat f_v = \frac{C_v/n - q}{p - q}, \qquad \operatorname{Var}\hat f_v = \frac{f_vp(1-p) + (1 - f_v)q(1-q)}{n(p - q)^2}.$$ {#eq-trust-oracle}

**$k$-ary randomized response.** Report the true value with probability $p = \frac{e^{\varepsilon_0}}{e^{\varepsilon_0} + k - 1}$, and each of the other $k - 1$ values with probability $q = \frac{1}{e^{\varepsilon_0} + k - 1}$. The two probabilities are chosen so that $p/q = e^{\varepsilon_0}$ and $p + (k - 1)q = 1$. The report supports $v$ if it equals $v$. It is an $\varepsilon_0$-DP local randomizer.

**Unary encoding.** Encode the value as a one-hot vector of $k$ bits (bit $v$ is 1 and all others are 0), and send each bit independently through its own randomized response: a 1 is sent as 1 with probability $p$, a 0 is sent as 1 with probability $q$. The report supports $v$ if bit $v$ is 1. Changing the record changes two bits of the vector, so the privacy level is the product of the two bits' worst ratios:
$$\varepsilon_0 = \ln\frac{p(1 - q)}{q(1 - p)}.$$
Two choices of $(p, q)$ are used. The **symmetric** one sets $p = e^{\varepsilon_0/2}/(e^{\varepsilon_0/2} + 1)$ and $q = 1 - p$, which is basic one-shot RAPPOR. The **optimized** one sets $p = \frac12$ and $q = 1/(e^{\varepsilon_0} + 1)$ (Wang et al. 2017).

**Variances for a rare value.** The regime that decides utility is $f_v \to 0$. Most values of a large domain are rare, and the variance at $f_v = 0$ is the noise floor: a value whose frequency is below a few standard deviations of it cannot be told apart from 0. Setting $f_v = 0$ in @eq-trust-oracle gives
$$k\text{-ary: } \frac{e^{\varepsilon_0} + k - 2}{n(e^{\varepsilon_0} - 1)^2}, \qquad \text{symmetric unary: } \frac{e^{\varepsilon_0/2}}{n(e^{\varepsilon_0/2} - 1)^2}, \qquad \text{optimized unary: } \frac{4e^{\varepsilon_0}}{n(e^{\varepsilon_0} - 1)^2}.$$
$k$-ary randomized response beats optimized unary encoding exactly when $k < 3e^{\varepsilon_0} + 2$.

### Derivation

1. **The oracle is unbiased (Fact 1).** $C_v$ is a sum of $n$ independent Bernoulli variables, one per person: $nf_v$ of them, the holders of $v$, have mean $p$, and the rest have mean $q$. So $\mathbb E[C_v/n] = f_vp + (1 - f_v)q = q + f_v(p - q)$, which is linear in $f_v$ with offset $a = q$ and slope $b = p - q$. Fact 1 gives the estimate in @eq-trust-oracle.
2. **Its variance (Fact 2).** The Bernoulli variances are $p(1-p)$ for the $nf_v$ holders and $q(1-q)$ for the others. Their sum is $n\big(f_vp(1-p) + (1 - f_v)q(1-q)\big)$. Dividing by $n^2$ (Fact 2, for the mean $C_v/n$) and by $(p - q)^2$ (Fact 1, for the debiasing) gives @eq-trust-oracle.
3. **$k$-ary randomized response is $\varepsilon_0$-DP.** Any single report $y$ has probability $p$ (if $y$ is the record) or $q$ (if not) under any record, so the ratio between two records is at most $p/q = e^{\varepsilon_0}$, which is @eq-trust-ldp for single outcomes.
4. **Unary encoding is private at the stated level.** Changing the record from $x$ to $x'$ changes the one-hot vector in two bits, $x$ and $x'$. All other bits have the same distribution under both records and cancel in the ratio. Bit $x$ goes from "a 1" to "a 0", so its contribution to the ratio is $p/q$ if it is sent as 1 and $(1-p)/(1-q)$ if sent as 0. Bit $x'$ goes the other way: $q/p$ or $(1-q)/(1-p)$. Since $p > q$, the largest product is $\frac pq\cdot\frac{1-q}{1-p}$, when bit $x$ is sent as 1 and bit $x'$ as 0. Plugging in the two choices gives exactly $e^{\varepsilon_0}$. For the symmetric one, $q = 1 - p$ makes both factors equal to $p/(1 - p) = e^{\varepsilon_0/2}$, and their product is $e^{\varepsilon_0}$. For the optimized one, $p = \frac12$ makes $p/(1 - p) = 1$, and $(1 - q)/q = e^{\varepsilon_0}$. The script checks both at $\varepsilon_0 = 1$.
5. **Rare-value variances.** At $f_v = 0$, @eq-trust-oracle is $q(1-q)/(n(p - q)^2)$. For $k$-ary: $q(1-q) = \frac{e^{\varepsilon_0} + k - 2}{(e^{\varepsilon_0} + k - 1)^2}$ and $p - q = \frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + k - 1}$; the factors $(e^{\varepsilon_0} + k - 1)^2$ cancel. For optimized unary: $q(1-q) = \frac{e^{\varepsilon_0}}{(e^{\varepsilon_0} + 1)^2}$ and $p - q = \frac12 - \frac{1}{e^{\varepsilon_0} + 1} = \frac{e^{\varepsilon_0} - 1}{2(e^{\varepsilon_0} + 1)}$; the factors $(e^{\varepsilon_0} + 1)^2$ cancel and the 2 squared leaves the 4. The symmetric case is binary randomized response at level $\varepsilon_0/2$, whose rare-value variance is the $k = 2$ case of the $k$-ary formula with $\varepsilon_0/2$ in place of $\varepsilon_0$.
6. **Crossover.** The $k$-ary and optimized unary variances have the same denominator, so the first is smaller exactly when $e^{\varepsilon_0} + k - 2 < 4e^{\varepsilon_0}$, that is when $k < 3e^{\varepsilon_0} + 2$.

### Interpretation

$k$-ary randomized response spreads its noise over the domain. With many values the true one is reported with probability $\approx e^{\varepsilon_0}/k$, so the report is almost uniform and carries little signal. Its rare-value variance has $k$ in the numerator, so it grows linearly in $k$ (@fig-trust-frequency-oracles).

Unary encoding avoids this by paying $\varepsilon_0$ for two bits, once, whatever $k$ is. Its variance does not depend on $k$, at the price of a $k$-bit report. Optimized unary encoding spends the budget asymmetrically: it keeps a 1 only half the time but almost never invents one. That helps because almost all bits of a one-hot vector are 0, so the noise that matters most is the false 1s, and those are the ones it suppresses. For large $k$, hashing schemes (local hashing, Hadamard response) reach the same variance with reports of $O(\log k)$ bits.

RAPPOR shows what a deployment adds. Erlingsson, Pihur & Korolova (2014) deployed a unary encoding over a Bloom filter (a hashed bit vector) in Chrome. It adds a **permanent** randomized response, memoized per value, on top of a fresh one per report. The reason is composition. By basic composition ([Post-processing, group privacy and composition](02-properties.md)), releasing several outputs about the same record, each $\varepsilon_0$-DP, costs the sum of their levels, because the privacy losses of independent releases add. Without memoization, a user who reports the same value every day would pay $\varepsilon_0$ per day, and averaging the reports would reveal the value. With it, the daily reports are fresh noise around a fixed randomized value, and averaging can recover at most that randomized value.

The variance still hides a $\sqrt n$. Every row of @tbl-trust-oracles must be divided by $n$, and the square root is again of order $1/\sqrt n$: these oracles are optimal local protocols, still with the gap of the previous section.

| $k$ | $k$-ary randomized response | Symmetric unary | Optimized unary |
| --- | --- | --- | --- |
| 2 | 0.921 | 3.918 | 3.683 |
| 8 | 2.953 | 3.918 | 3.683 |
| 32 | 11.082 | 3.918 | 3.683 |
| 128 | 43.596 | 3.918 | 3.683 |

: $n$ times the variance of the estimated frequency of a rare value, at $\varepsilon_0 = 1$. $k$-ary randomized response wins below $k = 3e + 2 = 10.15$. {#tbl-trust-oracles}

![$n$ times the variance of a rare value's estimated frequency, against the domain size $k$, at $\varepsilon_0 = 1$ and $\varepsilon_0 = 3$. $k$-ary randomized response grows linearly in $k$; the unary encodings are flat; the dotted line is the crossover $k = 3e^{\varepsilon_0} + 2$.](figures/trust_frequency_oracles.svg){#fig-trust-frequency-oracles}

### Example: an emoji survey

A keyboard app wants the frequency of each of $k = 128$ emoji at $\varepsilon_0 = 1$ from $n = 10^6$ users. The standard deviation of a rare emoji's estimate is the square root of the table entry divided by $n$. $k$-ary randomized response gives $\sqrt{43.596/10^6} = 0.0066$ per emoji; optimized unary encoding gives $\sqrt{3.683/10^6} = 0.0019$. An estimate has to stand about three standard deviations above 0 to be told apart from noise, and $3 \times 0.0019 \approx 0.006$, so emoji used by fewer than about 0.6% of users are lost in the noise either way. With $k = 8$ choices instead, below the crossover, $k$-ary randomized response is better: $\sqrt{2.953/10^6} = 0.0017$.

### Check yourself

- What is $k$-ary randomized response at $k = 2$, and why does @tbl-trust-oracles give 0.921 for it? *(It is binary randomized response. The rare-value formula becomes $e^{\varepsilon_0}/(e^{\varepsilon_0} - 1)^2$, the randomized-response factor, which is $e/(e - 1)^2 = 0.921$ at $\varepsilon_0 = 1$.)*
- In unary encoding, why does flipping the two bits of $x$ and $x'$ not cost $2\varepsilon_0$, given that each bit is a randomized response? *(The level $\varepsilon_0$ is defined as the log of the product of the two worst ratios, so it already accounts for both bits. The symmetric scheme gives each bit $\varepsilon_0/2$.)*

## The shuffle model and amplification by shuffling

### Context

The local model loses a factor $\sqrt n$ because each report must hide its owner from an analyst who knows whose report it is. Remove that knowledge, and the reports hide each other. In the **shuffle model** (Bittau et al. 2017, the encode–shuffle–analyze architecture of Prochlo; Cheu et al. 2019), each person sends a locally randomized report to a **shuffler**, which strips identifiers and timing and forwards a uniformly random permutation of the reports. The analyst sees the multiset: which reports were sent, but not by whom. The shuffler can be a separate organization, a mix network, or a trusted execution environment. It must be trusted only to unlink, not to keep data secret: it sees reports, which are already $\varepsilon_0$-DP.

### Statement

The claim is that anonymity amplifies privacy: the more people's reports the target's report is mixed with, the less the multiset reveals about the target, so the central $\varepsilon$ falls as $n$ grows. Erlingsson et al. (2019) showed that shuffling $\varepsilon_0$-DP reports gives central $(\varepsilon, \delta)$-DP with $\varepsilon = O(\varepsilon_0\sqrt{\ln(1/\delta)/n})$ for $\varepsilon_0 < 1$. The bound to compute with is Theorem 3.1 of Feldman, McMillan & Talwar (2021), quoted here with the range spaces of the randomizers left out. In it, $[n] = \{1, \dots, n\}$ and $z_{1:i-1}$ is the list of the first $i - 1$ outputs:

> For any domain $\mathcal D$, let $\mathcal R^{(i)}$ for $i \in [n]$ be a sequence of algorithms such that $\mathcal R^{(i)}(z_{1:i-1}, \cdot)$ is an $\varepsilon_0$-DP local randomizer for all values of auxiliary inputs $z_{1:i-1}$. Let $\mathcal A_s$ be the algorithm that, given a dataset $x_{1:n}$, samples a uniform random permutation $\pi$ over $[n]$, then sequentially computes $z_i = \mathcal R^{(i)}(z_{1:i-1}, x_{\pi(i)})$ for $i \in [n]$ and outputs $z_{1:n}$. Then for any $\delta \in [0, 1]$ such that $\varepsilon_0 \le \log\big(\frac{n}{16\log(2/\delta)}\big)$, $\mathcal A_s$ is $(\varepsilon, \delta)$-DP, where

$$\varepsilon \le \ln\left(1 + \frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1}\left(\frac{8\sqrt{e^{\varepsilon_0}\ln(4/\delta)}}{\sqrt n} + \frac{8e^{\varepsilon_0}}{n}\right)\right).$$ {#eq-trust-fmt}

The theorem is stated for replace neighbours, with $\log$ the natural logarithm. Shuffling the outputs of a fixed randomizer is the special case in which every $\mathcal R^{(i)}$ ignores its auxiliary input. Read the formula from the inside out: the bracket falls like $1/\sqrt n$, it is multiplied by a factor below 1 that depends only on $\varepsilon_0$, and $\ln(1 + u) \approx u$ when $u$ is small, so for large $n$ the central $\varepsilon$ is about that product.

### Why it works, in outline

The full derivation is in [Amplification by shuffling and by iteration](../part04-privacy-accounting/07-shuffling-iteration.md). Its first step is elementary and carries the intuition: every other person's report has some chance of looking exactly like a report the target might have sent, and after shuffling those look-alikes hide the target's report.

Fix the two values $x^0, x^1$ that the target person might hold. For any other person's value $x$ and any report $y$, @eq-trust-ldp says $y$ is at most $e^{\varepsilon_0}$ times less likely under $x$ than under any other value, so $p_R(y \mid x) \ge e^{-\varepsilon_0}p_R(y \mid x^0)$ and $p_R(y \mid x) \ge e^{-\varepsilon_0}p_R(y \mid x^1)$. A number at least as large as two others is at least as large as their average, so $p_R(y \mid x) \ge \frac{e^{-\varepsilon_0}}{2}\big(p_R(y \mid x^0) + p_R(y \mid x^1)\big)$. That lets you split the report distribution of $x$ into three pieces:
$$R(x) = \frac{e^{-\varepsilon_0}}{2}R(x^0) + \frac{e^{-\varepsilon_0}}{2}R(x^1) + (1 - e^{-\varepsilon_0})\,\mathrm{LO}(x),$$
where the leftover distribution $\mathrm{LO}(x)$ is whatever remains after removing the first two pieces, rescaled to total mass 1. It is non-negative by the inequality, and its weight $1 - e^{-\varepsilon_0}$ is what the first two pieces leave of the total mass 1.

Read the split as a recipe for drawing the report: with probability $e^{-\varepsilon_0}/2$ report as if holding $x^0$, with the same probability as if holding $x^1$, and otherwise draw from the leftover. So every other person, with probability $e^{-\varepsilon_0}$, sends a report that looks exactly like a report of $x^0$ or of $x^1$: a **clone** of the target. The number of clones is a binomial count with mean about $(n-1)e^{-\varepsilon_0}$, split evenly between the two hypotheses at random. After shuffling, the analyst sees the target's report among those clones and cannot tell which one is the target's.

To turn this into $(\varepsilon, \delta)$, the proof uses the **hockey-stick divergence** of [The definition of differential privacy](01-definition.md), $E_\gamma(P\|Q) = \sum_y\max\big(P(y) - \gamma Q(y), 0\big)$, the total by which $P$ exceeds $\gamma$ times $Q$. A mechanism is $(\varepsilon, \delta)$-DP exactly when $E_{e^\varepsilon} \le \delta$ for the output distributions of every pair of neighbours, because the worst event $S$ in the definition is the set of outputs where $P > e^\varepsilon Q$. The rest of the proof bounds the hockey-stick divergence of the resulting clone counts with binomial concentration ([Concentration inequalities](../part01-prerequisites/02-concentration.md)): a binomial count rarely strays from its mean by more than a few standard deviations, so the even split between $x^0$ and $x^1$ clones is rarely lopsided enough to expose the target. It gains the extra factor $\frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1}$ from writing the target's own report as a randomized response.

@fig-trust-clones does the split for $4$-ary randomized response at $\varepsilon_0 = \ln 3$, where $p = 3/(3 + 3) = \frac12$ and $q = 1/6$. A person holding value 3 reports $(\frac16, \frac16, \frac12, \frac16)$. The clone weight is $e^{-\varepsilon_0}/2 = \frac16$, so the clone part is $\frac16\big(R(1) + R(2)\big) = \frac16(\frac23, \frac23, \frac13, \frac13) = (0.1111, 0.1111, 0.0556, 0.0556)$. Subtracting it from the person's own distribution leaves $(0.0556, 0.0556, 0.4444, 0.1111)$, of total mass $0.6667 = 1 - e^{-\varepsilon_0}$.

![The report distribution of a person holding value 3 under 4-ary randomized response at $\varepsilon_0 = \ln 3$, split into a part that looks like a report of value 1 (blue), a part that looks like value 2 (orange), and a leftover (grey). With probability $e^{-\varepsilon_0} = 1/3$, this person is a clone of a target whose value is 1 or 2.](figures/trust_clones.svg){#fig-trust-clones}

### Interpretation

**The scaling.** The factor in front of the bracket is a hyperbolic tangent: dividing top and bottom by $e^{\varepsilon_0/2}$ gives $\frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1} = \frac{e^{\varepsilon_0/2} - e^{-\varepsilon_0/2}}{e^{\varepsilon_0/2} + e^{-\varepsilon_0/2}} = \tanh(\varepsilon_0/2)$. For large $n$ the second term of the bracket, of order $1/n$, is negligible next to the first, and $\ln(1 + u) \approx u$, so @eq-trust-fmt is about $\tanh(\varepsilon_0/2)\cdot 8\sqrt{e^{\varepsilon_0}\ln(4/\delta)/n}$. It falls like $1/\sqrt n$ (@fig-trust-shuffle-bound). For small $\varepsilon_0$, $\tanh(\varepsilon_0/2) \approx \varepsilon_0/2$ and $e^{\varepsilon_0} \approx 1$, so it is of order $\varepsilon_0\sqrt{\ln(1/\delta)/n}$; for large $\varepsilon_0$, $\tanh \approx 1$, so it is of order $\sqrt{e^{\varepsilon_0}\ln(1/\delta)/n}$.

**Large local $\varepsilon_0$ becomes affordable.** Invert the large-$\varepsilon_0$ form: a target central $\varepsilon$ allows $e^{\varepsilon_0}$ of order $\varepsilon^2n/\ln(1/\delta)$. Randomized response at that level has error $e^{\varepsilon_0/2}/((e^{\varepsilon_0} - 1)\sqrt n) \approx 1/\sqrt{ne^{\varepsilon_0}}$ on a fraction, since $e^{\varepsilon_0} - 1 \approx e^{\varepsilon_0}$ when $\varepsilon_0$ is large. Substituting $e^{\varepsilon_0} \approx \varepsilon^2n/\ln(1/\delta)$ gives $1/\sqrt{\varepsilon^2n^2/\ln(1/\delta)} = \sqrt{\ln(1/\delta)}/(\varepsilon n)$: the central rate $1/(\varepsilon n)$, up to $\sqrt{\ln(1/\delta)}$. Multiplied by $n$, this is the shuffle row of @tbl-trust-who.

**The condition matters.** $\varepsilon_0 \le \ln(n/(16\ln(2/\delta)))$ is the same as $e^{\varepsilon_0} \le n/(16\ln(2/\delta))$, or $ne^{-\varepsilon_0} \ge 16\ln(2/\delta)$: there must be enough expected clones. Below that, the target may have too few clones to hide among, and the theorem says nothing.

**The constants are loose.** @tbl-trust-shuffle compares @eq-trust-fmt with two computed values at $n = 10^4$ and $\delta = 10^{-6}$. The **numeric** column computes the exact hockey-stick divergence of the pair of clone counts that Theorem 3.2 of the same paper reduces the problem to: the clone argument above turns any $\varepsilon_0$-DP randomizer into a comparison of two distributions of counts, and that comparison can be computed exactly. So the numeric column is still a valid bound for every $\varepsilon_0$-DP randomizer. The **exact** column is the true $\varepsilon$ of shuffled binary randomized response on five dataset pairs (the other $n - 1$ people hold 0, 2500, 5000, 7499 or 9999 ones), maximized. Since binary randomized response is one of the randomizers the theorem covers, no valid general bound can be below it. Dividing the columns, the closed form is 2.6 to 4.6 times the numeric bound (for example $0.2140/0.0530 = 4.0$ at $\varepsilon_0 = 1$), and the numeric bound is at most 1.9 times the exact value. Use the closed form to reason, and a numerical accountant to deploy.

| $\varepsilon_0$ | Closed form, @eq-trust-fmt | Numeric (Theorem 3.2) | Exact, shuffled binary randomized response |
| --- | --- | --- | --- |
| 0.5 | 0.0939 | 0.0204 | 0.0162 |
| 1 | 0.2140 | 0.0530 | 0.0357 |
| 2 | 0.5009 | 0.1550 | 0.0870 |
| 3 | 0.8241 | 0.3207 | 0.1692 |

: Central $\varepsilon$ after shuffling the reports of $n = 10^4$ people, each $\varepsilon_0$-DP, at $\delta = 10^{-6}$. The condition of the theorem requires $\varepsilon_0 \le 3.763$ here. {#tbl-trust-shuffle}

![Central $\varepsilon$ at $\delta = 10^{-6}$ after shuffling, against $n$, for $\varepsilon_0 = 1$ and $\varepsilon_0 = 3$: the closed form of Feldman, McMillan & Talwar (lines, drawn where its condition holds), the numeric bound of their Theorem 3.2 (dots) and the exact $\varepsilon$ of shuffled binary randomized response (crosses). The dotted lines mark $\varepsilon_0$, the guarantee without shuffling.](figures/trust_shuffle_bound.svg){#fig-trust-shuffle-bound}

### Example: choosing the local level

You want central $\varepsilon = 1$ at $\delta = 10^{-6}$. The right-hand side of @eq-trust-fmt increases with $\varepsilon_0$, so the largest allowed $\varepsilon_0$ is where it equals 1, and bisection finds it: halve an interval of $\varepsilon_0$ values, keeping the half where the bound crosses 1. Each person may randomize at $\varepsilon_0 = 3.504$ when $n = 10^4$, at $5.701$ when $n = 10^5$, and at $7.992$ when $n = 10^6$. At $\varepsilon_0 = 7.992$, a single report is almost the true value: $k$-ary randomized response over 8 values keeps the truth with probability $e^{7.992}/(e^{7.992} + 7) = 0.998$. Yet the shuffled collection is $(1, 10^{-6})$-DP. @sec-trust-code shows what this buys in accuracy.

### Trust and its limits

The amplification holds only as long as the shuffler does its one job, and three things can undo it.

The first is collusion. If the shuffler colludes with the analyst, the analyst sees who sent what, and the guarantee falls back to the local $\varepsilon_0$ of each report, which may be large. Shuffling is a defence in depth, not a replacement for a reasonable $\varepsilon_0$.

The second is fake senders. If the adversary controls $m$ of the senders, it knows their reports and can remove them from the multiset. Only the $n - m$ honest reports provide clones, so use $n - m$ in place of $n$ in @eq-trust-fmt ([Exercise 5](solutions/08-trust-models.md#sec-sol-trust-5)).

The third is side channels. The theorem covers only the multiset. Timing, message sizes and the number of messages per person must not reveal anything either, which is why real shufflers batch and pad.

### Check yourself

- Why does the theorem allow each randomizer to depend on the earlier outputs $z_{1:i-1}$? *(So that adaptive protocols, such as a server that updates the query after each batch, are covered. The clone argument needs only that each step is $\varepsilon_0$-DP in its record, whatever the auxiliary input.)*
- At $n = 10^4$ and $\delta = 10^{-6}$, the theorem's condition allows $\varepsilon_0 \le 3.763$. How many clones does the target expect at that limit? *(At the limit the condition holds with equality, so $ne^{-\varepsilon_0} = 16\ln(2/\delta) = 16\times14.5 = 232$, since $\ln(2\times10^6) = 14.5$.)*

## Distributed DP: splitting the noise

### Context

The shuffle model still adds noise to every report. The **distributed model** adds the central amount of noise once, in pieces. Each client adds a small share of noise to its own value, and a **secure aggregation** protocol reveals only the sum of the noisy values to the server. In federated settings this is how a server can learn a noisy model update without seeing any client's update.

Treat secure aggregation as a black box here. It is a cryptographic protocol built from the additive secret sharing of [Cryptography basics](../part01-prerequisites/08-crypto-basics.md): each value is split into random shares that add up to it, so that any incomplete set of shares looks uniformly random and reveals nothing. The property used below is **simulation**: everything the server sees during the protocol could be generated from the sum alone, plus randomness of its own, so it learns nothing beyond the sum. Practical versions (Bonawitz et al. 2017) use pairwise masks that cancel in the sum and tolerate clients that drop out.

### Statement

The idea is that independent Gaussian noises add up to a Gaussian whose variance is the sum of theirs. So $n$ small Gaussians of variance $\sigma^2/n$ add up to exactly the one Gaussian of variance $\sigma^2$ that the central mechanism would have added.

Recall the **Gaussian mechanism**: release $f(D) + \mathcal N(0, \sigma^2I)$, where the **$\ell_2$ sensitivity** $\Delta_2$ is the largest Euclidean distance $\|f(D) - f(D')\|_2$ between neighbours. Its privacy depends only on the ratio $\Delta_2/\sigma$, and a larger $\sigma$ gives a smaller $\varepsilon$ at each $\delta$. Let $n$ clients hold vectors $x_i$ with a sum $f(D) = \sum_ix_i$ of $\ell_2$ sensitivity $\Delta_2$. Each client sends $x_i + Z_i$ with $Z_i \sim \mathcal N(0, \frac{\sigma^2}{n}I)$ independent, through secure aggregation. Then:

1. The server's view is $(\varepsilon, \delta)$-DP whenever the Gaussian mechanism with noise $\sigma$ on $f$ is.
2. If an adversary knows the noise of $t$ clients (they collude with it, or they dropped out after their noise was planned for), the guarantee is that of the Gaussian mechanism with noise $\sigma\sqrt{(n - t)/n}$.

### Derivation

1. **The noise adds up to the central noise.** Independent normals add their means and their variances, and their sum is again normal: $\sum_iZ_i \sim \mathcal N(0, n\cdot\frac{\sigma^2}{n}I) = \mathcal N(0, \sigma^2I)$. So the sum of the noisy values, $f(D) + \sum_iZ_i$, is exactly the output of the central Gaussian mechanism on $f(D)$.
2. **The protocol reveals only the sum.** By the simulation property, the server's view is a function (a simulator) of the sum and independent randomness. A function of a DP output is DP with the same parameters (post-processing, [Post-processing, group privacy and composition](02-properties.md)), so the view is as private as the sum.
3. **Colluders subtract what they know.** $t$ colluding clients know their own $x_i$ and $Z_i$ and can remove them from the sum. What remains is $\sum_{\text{honest}}x_i$ plus $\sum_{\text{honest}}Z_i$, a sum of $n - t$ normals of variance $\sigma^2/n$ each, which by step 1 is $\mathcal N(0, \frac{n - t}{n}\sigma^2I)$. That is a Gaussian mechanism with the smaller noise $\sigma\sqrt{(n - t)/n}$. The privacy of a Gaussian at a given $\sigma$ is the exact $(\varepsilon, \delta)$ curve of [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md).

### From real noise to integers: Skellam and the distributed discrete Gaussian

Secure aggregation adds integers modulo $q$, because only in a finite set like that can shares be uniformly random. Adding a fixed number modulo $q$ is a bijection, a one-to-one shuffle of the set, and a bijection maps the uniform distribution to itself ([Cryptography basics](../part01-prerequisites/08-crypto-basics.md)); there is no uniform distribution on the real numbers or on all integers for this to work with. Real-valued Gaussian noise cannot be sent. So the clients scale and round their vectors to integers, add integer noise, and the modulus is chosen large enough that the true sum never wraps around.

The integer noise must keep the property that made step 1 work: the sum of the shares must have a known distribution with the central privacy. Two answers are used.

The first is **Skellam noise** (Agarwal, Kairouz & Liu 2021). $\mathrm{Sk}(\mu)$ is the difference $N_1 - N_2$ of two independent $\mathrm{Poisson}(\mu)$ variables. Each Poisson has mean and variance $\mu$, so the difference has mean 0 and variance $\mu + \mu = 2\mu$. Sums of independent Poissons are Poisson with the summed mean. The reason is the **moment generating function** (MGF), $\mathbb E[e^{\lambda N}]$, which determines a distribution: for $\mathrm{Poisson}(\mu)$ it is $e^{\mu(e^\lambda - 1)}$, the MGF of a sum of independent variables is the product of their MGFs, and multiplying these exponentials adds the $\mu$s. So if each of $n$ clients adds $\mathrm{Sk}(\mu/n) = N_{1i} - N_{2i}$, the total is $\sum_iN_{1i} - \sum_iN_{2i}$, a difference of two independent $\mathrm{Poisson}(\mu)$ variables, which is exactly $\mathrm{Sk}(\mu)$. Agarwal, Kairouz & Liu show that its privacy is close to that of a Gaussian with the same variance $2\mu$, with a correction that vanishes as $\mu$ grows. This section uses the Gaussian calibration as that approximation and does not derive the correction.

The second is **the distributed discrete Gaussian** (Kairouz, Liu & Steinke 2021). A discrete Gaussian puts probability proportional to $e^{-z^2/(2\sigma^2)}$ on each integer $z$, and each client adds one. Unlike Skellam, the sum of discrete Gaussians is not a discrete Gaussian, but it is close to one when each share's variance is not too small, and the paper bounds the gap.

### Interpretation

Distributed noise gives central accuracy, with a $\delta$. The distributed sum has the error of the central Gaussian mechanism. Against the central Laplace mechanism on a low-dimensional histogram it still loses a constant factor (see @sec-trust-code), because the Gaussian has to pay for its $\delta$ with a wider spread. Its advantage is in high dimension. The Laplace mechanism is calibrated to the $\ell_1$ sensitivity, the sum of absolute changes, and the Gaussian to the $\ell_2$ sensitivity. For a $d$-dimensional change spread evenly over all coordinates, the $\ell_1$ norm is $\sqrt d$ times the $\ell_2$ norm, so in the worst case the $\ell_2$ sensitivity of a $d$-dimensional model update is $\sqrt d$ times smaller.

The guarantee depends on how many are honest. @fig-trust-collusion shows how fast it erodes. Deployments over-provision the noise: if at most a fraction $\rho$ of clients may collude or drop out, each adds $\frac{\sigma^2}{n(1-\rho)}$. Then the $n(1 - \rho)$ honest clients alone contribute $n(1-\rho)\cdot\frac{\sigma^2}{n(1-\rho)} = \sigma^2$, enough for the target, and when everyone is honest the server gets $\frac{1}{1-\rho}$ times the variance it needs.

Malicious clients are a different problem. A client can send any vector it likes, including one without noise or with an enormous value. Secure aggregation hides the inputs but does not check them; bounding them needs zero-knowledge proofs or verifiable aggregation, outside this chapter.

### Example: 100 clients

A server aggregates a histogram from users through 100 clients and wants $(\varepsilon, \delta) = (1, 10^{-6})$. Under replace, a user who changes value moves one count down by 1 and another up by 1, so $\Delta_2 = \sqrt{1^2 + 1^2} = \sqrt2$. The **analytic Gaussian calibration** of [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md) computes the exact $\delta$ of the Gaussian mechanism at a given $\sigma$ and picks the smallest $\sigma$ that meets the target. It gives total noise $\sigma = 5.975$, so each client adds noise of standard deviation $5.975/\sqrt{100} = 0.597$, because 100 variances of $\sigma^2/100$ add to $\sigma^2$. If 20% of the clients collude with the server, the remaining noise is $5.975\sqrt{0.8}$ and gives $\varepsilon = 1.128$; at 50%, $\varepsilon = 1.455$; if only 10% are honest, $\varepsilon = 3.525$. The script also checks the two pieces of the pipeline: 100 Skellam shares with total variance 25 sum to a variable with empirical mean 0.004 and variance 25.15, and the two-server additive sharing of the vectors $(3, -2)$, $(5, 7)$, $(0, 1)$ modulo $2^{20}$ reveals the sum $(8, 6)$.

![The actual $\varepsilon$ at $\delta = 10^{-6}$ of distributed Gaussian noise calibrated for a target $\varepsilon$, against the fraction of the total noise that the adversary does not know. At fraction 1 every client is honest and the target is met; below it, colluding or dropped clients leave less noise than planned.](figures/trust_collusion.svg){#fig-trust-collusion}

### Check yourself

- Why can't each client simply add $\mathcal N(0, \sigma^2)$, the full central noise? *(It would work for privacy, but the sum would carry $n\sigma^2$ of variance, $n$ times too much: that is the local model again, with its $\sqrt n$ loss.)*
- Is the sum of $n$ independent $\mathrm{Lap}(b/n)$ variables a $\mathrm{Lap}(b)$ variable? *(No. Each has variance $2(b/n)^2$, so the sum has variance $n \cdot 2b^2/n^2 = 2b^2/n$, not $2b^2$, and by the central limit theorem it is close to normal for large $n$. Splitting Laplace noise needs a different decomposition, as a difference of Gamma variables ([Exercise 6](solutions/08-trust-models.md#sec-sol-trust-6)).)*

## Code: frequency estimation in four trust models {#sec-trust-code}

The script [code/trust_models.py](code/trust_models.py) computes every number in this chapter and draws its figures. Each estimator, bound and accountant is a small typed function. The central comparison estimates a histogram over $k = 8$ values with Zipf frequencies ($f_v \propto 1/(v + 1)$) at the same central guarantee $(\varepsilon, \delta) = (1, 10^{-6})$ under replace in all four models. The four set-ups are:

- **Central:** the Laplace mechanism on the counts, $\mathrm{Lap}(2/\varepsilon)$ per count (the histogram has $\ell_1$ sensitivity 2 under replace, since one count goes down by 1 and another up by 1). It is pure $\varepsilon$-DP, which is stronger than required.
- **Distributed:** one-hot vectors plus Skellam shares through secure aggregation, with total variance from the Gaussian calibration at $\Delta_2 = \sqrt2$. The simulation draws the aggregated Skellam noise directly, which Poisson additivity makes exact.
- **Shuffle:** the better local oracle at the largest $\varepsilon_0$ that @eq-trust-fmt allows, capped by its condition.
- **Local:** the better local oracle at $\varepsilon_0 = \varepsilon = 1$.

The error is the expected squared $\ell_2$ distance between the estimated and the true histogram of fractions, computed exactly from @eq-trust-oracle and the noise variances, and checked against 200 simulated runs. For instance, the central Laplace noise has variance $2 \times 2^2 = 8$ per count, so 8 counts give $64$ on the counts and $64/n^2$ on the fractions, which is $6.40\times10^{-5}$ at $n = 10^3$. The distributed noise has variance $5.975^2 = 35.7$ per count, and $8 \times 35.7/10^6 = 2.86\times10^{-4}$. From the repository root:

```bash
uv run python book/part03-dp-foundations/code/trust_models.py
```

@tbl-trust-results shows the output (seed 0).

| Experiment | Result |
| --- | --- |
| Randomized response, $\varepsilon = \ln 3$, $n = 10^4$, 3,000 ones | sd 0.00866 (simulated: mean 0.3000, sd 0.00861). Count sd: local 86.6, central 1.287, ratio 67.3 |
| Fraction sd at $\varepsilon = 1$ | $n = 10^3$: local $3.03\times10^{-2}$, central $1.41\times10^{-3}$, ratio 21, lower bound $1.04\times10^{-2}$. $n = 10^5$: ratio 215. $n = 10^7$: ratio 2146 |
| One report, $\mathrm{Bern}(0.5)$ against $\mathrm{Bern}(0.6)$, $\varepsilon_0 = 1$ | KL 0.00429; bound @eq-trust-kl-contraction 0.02952 |
| Frequency oracles, $\varepsilon_0 = 1$ | See @tbl-trust-oracles; both unary encodings have $\varepsilon_0 = 1.0000$ |
| Clones, 4-ary, $\varepsilon_0 = \ln 3$ | Leftover mass 0.6667 |
| Shuffling, $n = 10^4$, $\delta = 10^{-6}$ | See @tbl-trust-shuffle. Largest $\varepsilon_0$ for $\varepsilon = 1$: 1.460 ($n = 10^3$, the condition's cap), 3.504 ($10^4$), 5.701 ($10^5$), 7.992 ($10^6$) |
| Distributed Gaussian, $(1, 10^{-6})$, 100 clients | $\sigma = 5.975$ total, 0.597 per client. $\varepsilon$ with 80%, 50%, 10% honest: 1.128, 1.455, 3.525 |
| Matched $(1, 10^{-6})$, $k = 8$, $n = 10^3$ | Central $6.40\times10^{-5}$; distributed $2.86\times10^{-4}$; shuffle $9.35\times10^{-3}$; local $2.71\times10^{-2}$ |
| Matched, $n = 10^4$ | Central $6.40\times10^{-7}$; distributed $2.86\times10^{-6}$; shuffle $4.88\times10^{-5}$; local $2.71\times10^{-3}$ |
| Matched, $n = 10^5$ | Central $6.40\times10^{-9}$; distributed $2.86\times10^{-8}$; shuffle $4.76\times10^{-7}$; local $2.71\times10^{-4}$ |

: Output of the chapter's script (seed 0). The matched rows give the expected squared $\ell_2$ error of the histogram; the script also prints the simulated values, which agree within the Monte Carlo error of 200 runs. {#tbl-trust-results}

[tests/test_trust_models.py](../../tests/test_trust_models.py) checks that the output still matches.

![Expected squared $\ell_2$ error of an 8-value histogram against the number of people, at the same central $(\varepsilon, \delta) = (1, 10^{-6})$ in all four trust models. Lines are exact; dots are simulations. Central and distributed fall like $1/n^2$, local like $1/n$; shuffle bends from the local to the central slope as $n$ grows.](figures/trust_matched_error.svg){#fig-trust-matched-error}

**What to take from it**

Local is a different regime. At $n = 10^5$ the local error is $2.71\times10^{-4}/6.40\times10^{-9} \approx 42{,}000$ times the central error. Squared errors scale like $1/n$ locally and like $1/n^2$ centrally, so the gap grows linearly in $n$ (@fig-trust-matched-error).

Shuffling recovers most of it. At $n = 10^4$ the shuffle error is $2.71\times10^{-3}/4.88\times10^{-5} \approx 56$ times smaller than local, and from $n = 10^4$ on it falls like $1/n^2$, parallel to central, because the allowed $\varepsilon_0$ keeps growing with $n$. At $n = 10^3$ the theorem's condition caps $\varepsilon_0$ at 1.460, and shuffle helps only by a factor of 3 ($2.71\times10^{-2}/9.35\times10^{-3}$). Its remaining distance from central is the loose constant of @eq-trust-fmt and the $\sqrt{\ln(1/\delta)}$; a numerical bound narrows it.

Distributed is central Gaussian. It is $2.86\times10^{-4}/6.40\times10^{-5} \approx 4.5$ times the central Laplace error for this 8-value histogram, because Gaussian noise for $\delta = 10^{-6}$ is wider than Laplace noise in low dimension. It needs no trusted party, only honest-majority cryptography.

Every model's guarantee is the same $(\varepsilon, \delta)$, against a different adversary. The local model protects against the collector; the others protect only against parties that do not see what the trusted component sees. Report both.

## References

- Warner, "Randomized Response: A Survey Technique for Eliminating Evasive Answer Bias" (*Journal of the American Statistical Association* 1965): the original local randomizer.
- Kasiviswanathan, Lee, Nissim, Raskhodnikova & Smith, "What Can We Learn Privately?" (FOCS 2008): defines the local model and relates it to statistical-query learning.
- Duchi, Jordan & Wainwright, "Minimax Optimal Procedures for Locally Private Estimation" (*Journal of the American Statistical Association* 2018; conference version 2013): the KL contraction behind the local lower bound, and optimal local estimators.
- Chan, Shi & Song, "Optimal Lower Bound for Differentially Private Multi-Party Aggregation" (2012): the $\Omega(\sqrt n)$ lower bound for counting without a trusted party.
- Erlingsson, Pihur & Korolova, "RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response" (CCS 2014): the first large deployment, with memoization.
- Wang, Blocki, Li & Jha, "Locally Differentially Private Protocols for Frequency Estimation" (USENIX Security 2017): the frequency-oracle framework, optimized unary encoding and local hashing.
- Bittau et al., "Prochlo: Strong Privacy for Analytics in the Crowd" (2017): the encode–shuffle–analyze architecture.
- Cheu, Smith, Ullman, Zeber & Zhilyaev, "Distributed Differential Privacy via Shuffling" (EUROCRYPT 2019): the shuffle model as a model of computation, and its separations from local and central.
- Erlingsson, Feldman, Mironov, Raghunathan, Talwar & Thakurta, "Amplification by Shuffling: From Local to Central Differential Privacy via Anonymity" (SODA 2019): the first amplification theorem.
- Balle, Bell, Gascón & Nissim, "The Privacy Blanket of the Shuffle Model" (CRYPTO 2019): a different decomposition of the randomizer, and tight bounds for specific protocols.
- Feldman, McMillan & Talwar, "Hiding Among the Clones: A Simple and Nearly Optimal Analysis of Privacy Amplification by Shuffling" (FOCS 2021): the closed form used here, the clone reduction and its numerical version.
- Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving Machine Learning" (CCS 2017): the secure aggregation protocol used in federated learning.
- Kairouz, Liu & Steinke, "The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation" (ICML 2021) and Agarwal, Kairouz & Liu, "The Skellam Mechanism for Differentially Private Federated Learning" (NeurIPS 2021): integer noise that survives secure aggregation.

## Cheat sheet

@tbl-trust-cheat-sheet collects the results. $\varepsilon_0$ is the local level of one report, and errors are on fractions.

| Result | Statement | Note |
| --- | --- | --- |
| Local randomizer | $\Pr[R(x) \in S] \le e^{\varepsilon_0}\Pr[R(x') \in S]$ for all $x, x'$ | The transcript is $\varepsilon_0$-DP under replace, against everyone |
| Randomized response | $\operatorname{sd} = e^{\varepsilon_0/2}/((e^{\varepsilon_0} - 1)\sqrt n)$ | Data-independent; $\approx 1/(\varepsilon_0\sqrt n)$ for small $\varepsilon_0$ |
| KL contraction | $D_{\mathrm{KL}}(M_0\|M_1) \le (e^{\varepsilon_0} - 1)^2\mathrm{TV}(P_0, P_1)^2$ | Holds for any local randomizer |
| Local lower bound | error $\ge (1 - 2\beta)/((e^{\varepsilon_0} - 1)\sqrt{2n})$ | Central Laplace: $\sqrt2/(n\varepsilon)$; the gap is $\sqrt n$ |
| Frequency oracle | $\hat f_v = (C_v/n - q)/(p - q)$, variance @eq-trust-oracle | Unbiased; exact variance |
| $k$-ary randomized response | $p = e^{\varepsilon_0}/(e^{\varepsilon_0} + k - 1)$; rare variance $(e^{\varepsilon_0} + k - 2)/(n(e^{\varepsilon_0} - 1)^2)$ | Best for $k < 3e^{\varepsilon_0} + 2$ |
| Optimized unary encoding | $p = \frac12$, $q = 1/(e^{\varepsilon_0} + 1)$; rare variance $4e^{\varepsilon_0}/(n(e^{\varepsilon_0} - 1)^2)$ | Independent of $k$; $k$-bit reports |
| Clones | $R(x) = \frac{e^{-\varepsilon_0}}{2}R(x^0) + \frac{e^{-\varepsilon_0}}{2}R(x^1) + (1 - e^{-\varepsilon_0})\mathrm{LO}(x)$ | The intuition behind amplification |
| Shuffling (Feldman, McMillan & Talwar 2021) | @eq-trust-fmt, for $\varepsilon_0 \le \ln(n/(16\ln(2/\delta)))$ | $\varepsilon \approx 8\tanh(\varepsilon_0/2)\sqrt{e^{\varepsilon_0}\ln(4/\delta)/n}$; 2.6–4.6 times loose at $n = 10^4$ |
| Distributed Gaussian | $n$ shares $\mathcal N(0, \sigma^2/n)$ sum to $\mathcal N(0, \sigma^2)$ | $t$ colluders leave $\sigma\sqrt{(n - t)/n}$ |
| Skellam | $n$ shares $\mathrm{Sk}(\mu/n)$ sum to $\mathrm{Sk}(\mu)$, variance $2\mu$ | Integer noise for secure aggregation |

: Summary of the results of this chapter. {#tbl-trust-cheat-sheet}

**Rules of thumb**

- Local DP costs a factor of about $\sqrt n$ in error over central DP. Use it only for frequent items in very large populations.
- For frequency estimation, use $k$-ary randomized response when $k < 3e^{\varepsilon_0} + 2$ and optimized unary encoding (or a hashing variant) otherwise. Memoize reports of values that do not change.
- A shuffler lets each report use a large $\varepsilon_0$, roughly up to $\ln(\varepsilon^2n/\ln(1/\delta))$. Compute the central $\varepsilon$ numerically before deploying; the closed form is several times loose.
- With distributed noise, decide how many clients may collude or drop out, and over-provision the noise for it.
- Always state the trust model next to $(\varepsilon, \delta)$: the same numbers protect against different adversaries.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](solutions/08-trust-models.md).
