# Exercises: Query release and private multiplicative weights

Exercises for [Query release and private multiplicative weights](../../book-full/part03-dp-foundations/06-query-release.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part03-dp-foundations/solutions_query_release.py` computes every number quoted here; run it with `uv run python code/part03-dp-foundations/solutions_query_release.py`.

The solutions use the chapter's notation. The universe has $N$ possible records, a dataset of $n$ records is its histogram $p$, and a linear query $q$ has answer $\langle q, p\rangle$. $\mathcal Q$ is the set of queries, the workload, and $|\mathcal Q|$ its size. Multiplicative weights with learning rate $\eta$ (also called the step size) updates each weight by the factor $1 - \eta\ell$, where $\ell$ is the expert's loss, and after $T$ rounds its regret is at most $\eta T + \ln N/\eta$ (@eq-qr-regret). Private multiplicative weights runs $T$ rounds, each with two steps of budget $\varepsilon_0$, and its error bound @eq-qr-pmw contains the term $L = \ln|\mathcal Q| + 2\ln(2T/\beta)$, where $\beta$ is the failure probability. Every exercise that needs one uses $\beta = 0.05$; under approximate differential privacy (DP) it uses $\delta = 10^{-6}$. Logarithms are natural unless written $\log_2$.

## The halving algorithm {#sec-sol-qr-1}

> **Exercise 1.** **The halving algorithm.** $N$ experts predict a yes/no outcome each day, and one of them is always right. You follow the majority of the experts who have never been wrong, and you drop every expert as soon as it errs. Show that you make at most $\log_2N$ mistakes, whatever the predictions. *Check: with 1,024 experts the bound is 10; in 200 simulated runs of 200 days with random experts, the most mistakes was 8.* *(Potential argument)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The perfect expert survives.** It is never wrong, so it is never dropped, and at least one expert always remains.
2. **A mistake halves the survivors.** Break ties in the vote either way. You are wrong only when at least half of the surviving experts were wrong. All of those are dropped, so at least half the survivors go.
3. **Count.** Start with $N$ survivors. After $M$ mistakes at most $N/2^M$ remain, and by step 1 that is at least 1. So $2^M \le N$, which is $M \le \log_2N$.
4. **Numbers.** With $N = 1024$ the bound is $\log_2 1024 = 10$. In the script's 200 runs, with random predictions from the other experts, the most mistakes was 8 and the average 4.29.

*The days on which you are right cost nothing, and the days on which you are wrong pay for themselves by shrinking the pool. Multiplicative weights replaces "drop" with "shrink the weight", which is the same argument with the total weight in place of the number of survivors.*
:::

## Weighted majority {#sec-sol-qr-2}

> **Exercise 2.** **Weighted majority.** Now no expert need be perfect, and the best one is wrong $\mathrm{OPT}$ times. Give every expert weight 1, follow the weighted majority, and halve the weight of every expert that is wrong. Show that you make at most $(\mathrm{OPT} + \log_2N)/\log_2(4/3) \approx 2.41(\mathrm{OPT} + \log_2N)$ mistakes. *Check: $1/\log_2(4/3) = 2.4094$; with 1,024 experts and $\mathrm{OPT} = 5$ the bound is 36.14, and the most mistakes in 200 simulated runs was 15.* *(Upper and lower bounds on the total weight)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Lower bound.** The best expert's weight is halved once per mistake, so at the end it is $2^{-\mathrm{OPT}}$, and the total weight $W$ is at least that.
2. **Upper bound.** When you are wrong, at least half the total weight was on wrong experts, and that half is halved. So the total weight falls to at most $\frac12 + \frac12\cdot\frac12 = \frac34$ of what it was. After $M$ mistakes, $W \le N(3/4)^M$.
3. **Combine.** $2^{-\mathrm{OPT}} \le N(3/4)^M$. Taking $\log_2$ of both sides, $-\mathrm{OPT} \le \log_2N - M\log_2(4/3)$, so $M \le (\mathrm{OPT} + \log_2N)/\log_2(4/3)$.
4. **Numbers.** $\log_2(4/3) = 0.4150$, so the constant is 2.4094. With $N = 1024$ and $\mathrm{OPT} = 5$ the bound is $2.4094 \times 15 = 36.14$. The script's 200 runs with random experts never exceeded 15 mistakes.

*The factor 2.41 in front of $\mathrm{OPT}$ is the price of a deterministic learner: an adversary can make the weighted majority wrong whenever the vote is close. The randomized version of the chapter pays only $\mathrm{OPT}$ plus a term that grows like the square root of the number of rounds.*
:::

## The learning rate in the mistake bound {#sec-sol-qr-3}

> **Exercise 3.** **The learning rate in the mistake bound.** In the derivation of @eq-qr-mistakes, keep a general learning rate $\eta < \alpha$ instead of $\alpha/2$. Show that the number of corrections satisfies $U < \ln N/(\eta(\alpha - \eta))$, and that $\eta = \alpha/2$ minimizes this bound. *Check: for $N = 256$ and $\alpha = 0.05$, the bound is 8,872 at $\eta = 0.025$ and 11,830 at both $\eta = 0.0125$ and $\eta = 0.0375$.* *(Regret to mistakes)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The same two bounds on the regret.** Each correction has regret more than $\alpha$, so after $U$ corrections the total is more than $\alpha U$. The regret bound @eq-qr-regret, applied to those $U$ rounds ($T = U$), gives at most $\eta U + \ln N/\eta$.
2. **Solve for $U$.** $\alpha U < \eta U + \ln N/\eta$ gives $(\alpha - \eta)U < \ln N/\eta$, so $U < \ln N/(\eta(\alpha - \eta))$ for $\eta < \alpha$.
3. **The best learning rate.** The bound is smallest when $\eta(\alpha - \eta)$ is largest. That product is a downward parabola in $\eta$ with roots at 0 and $\alpha$, so its maximum is at the midpoint $\eta = \alpha/2$, where it equals $\alpha^2/4$. The bound is then $4\ln N/\alpha^2$.
4. **Numbers.** With $\ln 256 = 5.545$ and $\alpha = 0.05$: at $\eta = 0.025$ the bound is $5.545/(0.025 \times 0.025) = 8872$. At $\eta = 0.0125$ or $0.0375$ the product is $0.0125 \times 0.0375$ in both cases, and the bound is 11,830.

*A step that is too small learns slowly from each correction, and a step that is too large pays too much regret on each one. The parabola is symmetric, so the two kinds of error cost the same at equal distances from $\alpha/2$.*
:::

## The pure-DP sample size {#sec-sol-qr-4}

> **Exercise 4.** **The pure-DP sample size.** Under pure DP each step gets $\varepsilon_0 = \varepsilon/(2T)$, and the error bound @eq-qr-pmw becomes $A/\sqrt T + CT$ with $A = 2\sqrt{\ln N}$ and $C = 4L/(n\varepsilon)$. The term $L$ grows with $T$ only through a logarithm, so treat it as a constant. Find the $T$ that minimizes the bound, show that the minimum is $3(4L\ln N/(n\varepsilon))^{1/3}$, and deduce $n = 108L\ln N/(\alpha^3\varepsilon)$. *Check: for $N = 2^{20}$, $|\mathcal Q| = 10^6$, $\alpha = 0.05$ and $\varepsilon = 1$, a search over $T$ that lets $L$ vary needs $5.14\times10^8$ records, at $T = 48{,}610$ and $L = 42.78$. The closed form with that $L$ gives $5.12\times10^8$.* *(Balancing two terms)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The derivative.** The derivative of $A T^{-1/2} + CT$ with respect to $T$ is $-\frac A2T^{-3/2} + C$. It is zero at $T^{3/2} = A/(2C)$, that is $T = (A/(2C))^{2/3}$, and the function is convex, so this is the minimum.
2. **The minimum.** At that $T$, $A T^{-1/2} = A(2C/A)^{1/3} = 2^{1/3}A^{2/3}C^{1/3}$ and $CT = C(A/(2C))^{2/3} = 2^{-2/3}A^{2/3}C^{1/3}$. Since $2^{1/3} = 2 \times 2^{-2/3}$, the two coefficients add to $3 \times 2^{-2/3}$, and the sum is $\frac{3}{2^{2/3}}A^{2/3}C^{1/3}$.
3. **Substitute.** $A^{2/3} = 2^{2/3}(\ln N)^{1/3}$, so the minimum is $3(\ln N)^{1/3}C^{1/3} = 3\big(4L\ln N/(n\varepsilon)\big)^{1/3}$.
4. **Records.** Setting this equal to $\alpha$ and cubing, $\alpha^3 = 27 \times 4L\ln N/(n\varepsilon)$, so $n = 108L\ln N/(\alpha^3\varepsilon)$.
5. **Numbers.** The script searches over $T$, recomputing $L$ for each, and finds $5.14\times10^8$ records at $T = 48{,}610$, where $L = \ln 10^6 + 2\ln(2 \times 48{,}610/0.05) = 42.78$ with $\beta = 0.05$. The closed form with that $L$ gives $108 \times 42.78 \times \ln 2^{20}/0.05^3 = 5.12\times10^8$. The same search under approximate DP, as in @tbl-qr-records, needs $1.56\times10^7$ records, about 33 times fewer.

*Pure DP pays $\alpha^{-3}$ where approximate DP pays $\alpha^{-2}$, because basic composition charges each round $1/T$ of the budget, while advanced composition charges only about $1/\sqrt T$. It is the same $\sqrt k$ saving as in [Advanced and optimal composition](../../book-full/part03-dp-foundations/04-composition.md).*
:::

## A small dataset by sampling {#sec-sol-qr-5}

> **Exercise 5.** **A small dataset by sampling.** SmallDB rests on a claim: a random sample of $m = \lceil2\ln(2|\mathcal Q|)/\alpha^2\rceil$ records, drawn from the data, has error at most $\alpha/2$ on every query with positive probability. Take the chapter's running example, a table with eight yes/no attributes and 6,560 marginal queries, and $\alpha = 0.1$. Compute $m$, and check the claim by drawing samples from the example's histogram. *Check: $m = 1897$. Over 200 samples, the largest error is 0.0261 on average and at most 0.0451, so every sample is within $\alpha/2 = 0.05$. With only 200 records per sample, 1% of the samples stay within 0.05.* *(Hoeffding's inequality and the union bound)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The size.** $m = \lceil2\ln(2 \times 6560)/0.1^2\rceil = \lceil200\ln 13{,}120\rceil = \lceil1896.4\rceil = 1897$.
2. **Why it works.** For one query, a sample's answer is an average of $m$ independent values in $[0, 1]$ with mean equal to the true answer. Hoeffding's inequality says that such an average is off by more than $s$ with probability at most $2e^{-2ms^2}$, which at $s = \alpha/2$ is $2e^{-m\alpha^2/2}$. With $m = 1897$ this is $2e^{-9.485} = 1.52\times10^{-4}$, and the union bound over 6,560 queries gives at most 0.997, which is below 1.
3. **The check.** The script draws 200 samples of 1,897 records from the example's histogram and computes each sample's largest error over the 6,560 queries. The average is 0.0261 and the largest is 0.0451, so every sample is within 0.05.
4. **Fewer records.** With 200 records instead, the average largest error is 0.0834, and only 1% of samples stay within 0.05.

*The union bound only just certifies that a good sample exists, with a failure bound of 0.997. In fact every one of the 200 samples was good. The union bound adds up 6,560 failure probabilities as if the failures never overlapped, but the queries share records, and each query's own failure probability is far below the Hoeffding bound. SmallDB's search over all $N^m$ datasets is where the cost lies, not the existence of a good one.*
:::

## Where private multiplicative weights overtakes Gaussian noise {#sec-sol-qr-6}

> **Exercise 6.** **Where private multiplicative weights overtakes Gaussian noise.** Using @eq-qr-gauss-n and the bound of private multiplicative weights with its best number of rounds, find the number of queries at which private multiplicative weights needs fewer records for error 0.05, at $\varepsilon = 1$ and $\delta = 10^{-6}$. Explain why shrinking the universe from $2^{20}$ to $2^{10}$ records moves the crossover so little. *Check: about $1.1\times10^9$ queries for $N = 2^{20}$ and $5.1\times10^8$ for $N = 2^{10}$.* *(Comparing sample complexities)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The two requirements.** Gaussian noise on every answer needs $n = \sqrt{|\mathcal Q|}\,z/(\mu\alpha)$ by @eq-qr-gauss-n. Here $\mu$ is a constant fixed by $(\varepsilon, \delta)$, and $z$, a normal quantile at the tail $\beta/(2|\mathcal Q|)$, grows only like $\sqrt{\ln|\mathcal Q|}$, so the requirement grows like $\sqrt{|\mathcal Q|\ln|\mathcal Q|}$. Private multiplicative weights needs about $64\sqrt{\ln N\ln(1/\delta)}\,L/(\alpha^2\varepsilon)$ by @eq-qr-pmw-n, which grows like $\ln|\mathcal Q|$ through $L$.
2. **The crossover.** At small $|\mathcal Q|$ the Gaussian mechanism needs far fewer records (@tbl-qr-records), and its requirement grows faster, so the two curves cross once. A bisection on $|\mathcal Q|$ in the script finds the crossover at about $1.1\times10^9$ queries for $N = 2^{20}$.
3. **A smaller universe.** With $N = 2^{10}$ the factor $\sqrt{\ln N}$ falls by $\sqrt{10/20} = 0.71$, so private multiplicative weights needs about 0.71 times as many records. The Gaussian requirement grows like $\sqrt{|\mathcal Q|}$, up to the slowly growing $z$, so it reaches a level 0.71 times lower at about $0.71^2 = 0.5$ times as many queries. The script finds $5.1\times10^8$. So a universe 1,024 times smaller moves the crossover by only a factor of 2, because the universe enters private multiplicative weights only through $\sqrt{\ln N}$.

*The advantage of private multiplicative weights is real but asymptotic. It needs about a billion queries without structure (queries whose answers one record can all change) at these settings, which is why it matters most for releases such as all marginals of a wide table, where the number of queries grows exponentially with the number of attributes.*
:::
