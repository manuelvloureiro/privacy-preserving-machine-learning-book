# Sparse Vector

Queries often arrive one at a time, and each must be answered before the next is seen. Under differential privacy (DP), answering each of them with its own noise makes the error grow polynomially with the number of queries. Often the analyst only needs to know which queries exceed a threshold, and few of them do. This chapter shows that the cost then depends on the number of large queries and only logarithmically on the total. The tool is the sparse vector technique. Its privacy proof is short but delicate, because it rests on adding noise to the threshold as well as to the queries. At the end, the technique turns private multiplicative weights into an algorithm that answers queries as they arrive.

## Answering queries online {#sec-svt-online}

The setting is a stream of queries $f_1, f_2, \dots, f_k \colon \mathcal X^n \to \mathbb R$, each with sensitivity at most $1$: changing one record changes $f_i(X)$ by at most $1$. Counting queries, which count the records with some property, are the typical example. Query $f_i$ arrives at step $i$ and must be answered before $f_{i+1}$ is revealed.

The [Laplace mechanism](../part02-dp-basics/02-laplace-mechanism.md) answers each query with noise from the Laplace distribution $\mathrm{Lap}(b)$ of scale $b$, whose density at $z$ is $e^{-|z|/b}/(2b)$. For a sensitivity-$1$ query it is $\varepsilon$-DP when $b = 1/\varepsilon$. Its tails are $\Pr[|\mathrm{Lap}(b)| \ge tb] = e^{-t}$. Under pure DP with a total budget $\varepsilon$, basic composition forces the scale $k/\varepsilon$ on each of $k$ answers, so the error is of order $k/\varepsilon$. Under approximate DP, the [advanced composition theorem](../part02-dp-basics/04-advanced-composition.md) brings this down to order $\sqrt{k\log(1/\delta)}/\varepsilon$. Both are polynomial in $k$, and in the worst case no algorithm that answers every query can do much better.

A weaker question has a cheaper answer: which queries are large? If all $k$ queries were known in advance, the [exponential mechanism](../part02-dp-basics/05-exponential-mechanism.md) with score $f_i(X)$ would pick a query whose value is within $2(\ln k + t)/\varepsilon$ of the largest, except with probability $e^{-t}$. Repeating it $c$ times with budget $\varepsilon/c$ each finds $c$ large queries with error of order $c\log k/\varepsilon$, or $\sqrt{c\log(1/\delta)}\log k/\varepsilon$ with advanced composition. That is logarithmic in $k$, but it needs the whole list. Online, the first query cannot be compared with queries not yet seen, so the question becomes absolute.

> Given a public threshold $T$, report the first $c$ queries in the stream with $f_i(X) \ge T$, and stop.

The idea is to run the Laplace mechanism on each query and report only whether the noisy value exceeds the threshold. The essential detail is to compare it with a noisy threshold $\hat T$, not with $T$. Getting the details exactly right is notoriously hard, and several published variants are not private.

## AboveThreshold {#sec-svt-above}

The case $c = 1$ finds the first large query and halts.

::: {#def-svt-above name="AboveThreshold"}
The inputs are a dataset $X$, a stream of sensitivity-$1$ queries $f_1, f_2, \dots$, a threshold $T \in \mathbb R$ and $\varepsilon > 0$.

1. Draw $\hat T = T + \mathrm{Lap}(2/\varepsilon)$.
2. For each query $f_i$ in turn, draw $\nu_i \sim \mathrm{Lap}(4/\varepsilon)$. If $f_i(X) + \nu_i \ge \hat T$, output $\top$ and halt. Otherwise output $\bot$.

All noise draws are independent.
:::

The symbol $\top$ means "above the threshold" and $\bot$ means "below". Nothing in the algorithm depends on the number of queries, so the stream may even be infinite, and the privacy guarantee does not change. The accuracy guarantee does, because more noise draws mean a larger largest one.

The queries may be fixed in advance or chosen by an analyst who sees the earlier answers. Before the algorithm halts every answer is $\bot$, so in the second case each query is still a fixed function of the analyst's own coins, which do not depend on the dataset. The proof below applies to the stream fixed by each value of those coins, and averaging the inequality over the coins keeps it.

::: {#thm-svt-above-privacy}
AboveThreshold is $\varepsilon$-DP.
:::

::: {.proof}
1. Fix neighbours $X \sim X'$ and a stream, and write $A$ and $A'$ for the outputs on $X$ and $X'$. The outputs are $a = \bot^{t-1}\top$, which halts at query $t$, and the output that never halts: $\bot^k$ for a stream of finite length $k$, or $\bot^\infty$ for an infinite one. These form a countable set, so by the [pointwise lemma for discrete outputs](../part02-dp-basics/01-randomized-response.md) it suffices to show $\Pr[A = a] \le e^\varepsilon\Pr[A' = a]$ for each of them.
2. *Fix the noise of the first $t-1$ queries.* Condition on $\nu_i = v_i$ for $i < t$, the same values on both datasets, and let

$$
g(X) = \max_{i < t}\big(f_i(X) + v_i\big),
$$

with $g = -\infty$ if $t = 1$. With the $v_i$ fixed, $g$ is a number, not a random variable. Each $f_i$ has sensitivity $1$, so $|g(X) - g(X')| \le 1$.
3. The output is $a$ exactly when every earlier query falls below $\hat T$ and query $t$ does not, that is, when $g(X) < \hat T \le f_t(X) + \nu_t$. Write $\phi_2$ and $\phi_4$ for the densities of $\hat T$ and $\nu_t$. Then

$$
\Pr[A = a] = \int\!\!\int \phi_2(\tau)\,\phi_4(v)\,\mathbf 1\big[g(X) < \tau \le f_t(X) + v\big]\,d\tau\,dv .
$$

4. *Change variables to move from $X$ to $X'$.* Substitute

$$
\tau = \tau' + g(X) - g(X'), \qquad v = v' + g(X) - g(X') + f_t(X') - f_t(X).
$$

This is a shift, so $d\tau\,dv = d\tau'\,dv'$. The condition $g(X) < \tau$ becomes $g(X') < \tau'$, and $\tau \le f_t(X) + v$ becomes $\tau' \le f_t(X') + v'$, so the indicator becomes $\mathbf 1[g(X') < \tau' \le f_t(X') + v']$.
5. *Pay for the shifts.* By step 2, $|\tau - \tau'| \le 1$. By step 2 and the sensitivity of $f_t$, $|v - v'| \le 2$. A Laplace density of scale $b$ changes by at most a factor $e^{s/b}$ when its argument moves by $s$, so $\phi_2(\tau) \le e^{\varepsilon/2}\phi_2(\tau')$ and $\phi_4(v) \le e^{2\varepsilon/4}\phi_4(v') = e^{\varepsilon/2}\phi_4(v')$. Hence

$$
\Pr[A = a] \le e^{\varepsilon}\int\!\!\int \phi_2(\tau')\,\phi_4(v')\,\mathbf 1\big[g(X') < \tau' \le f_t(X') + v'\big]\,d\tau'\,dv' = e^\varepsilon\Pr[A' = a],
$$

where both probabilities are conditional on the fixed $v_1,\dots,v_{t-1}$.
6. The $v_i$ have the same distribution on both datasets, so averaging step 5 over them gives the unconditional inequality.
7. *The output that never halts.* It occurs exactly when $\hat T > \sup_i(f_i(X) + \nu_i)$, the supremum over all queries of the stream. Fixing all $\nu_i$ makes this supremum a number $g(X)$, possibly $+\infty$, with $|g(X) - g(X')| \le 1$ when it is finite, and the output has probability $0$ on both datasets when it is not. Shifting $\tau$ by at most $1$ as in steps 4 and 5 gives the factor $e^{\varepsilon/2} \le e^\varepsilon$.
:::

The two noises do different jobs. The noise on $\hat T$ pays, in one draw, for all the $\bot$ answers: the proof fixes their noise and absorbs the change in their maximum by shifting the threshold. The noise $\nu_t$ pays for the single $\top$. So the privacy cost does not grow with the number of $\bot$ answers.

The output carries no numbers, so accuracy means making no mistakes, with some slack around the threshold.

::: {#def-svt-accurate name="Accuracy for threshold answers"}
An algorithm that answers a stream of queries with $\top$ or $\bot$ is **$(\alpha,\beta)$-accurate** on $X$ if, with probability at least $1-\beta$, every answered query satisfies

$$
a_i = \top \implies f_i(X) \ge T - \alpha \qquad\text{and}\qquad a_i = \bot \implies f_i(X) \le T + \alpha .
$$
:::

::: {#thm-svt-above-accuracy name="Accuracy of AboveThreshold"}
Let $\beta \in (0,1)$, and let $f_1, \dots, f_k$ be a stream with $f_i(X) \le T - \alpha$ for every $i < k$, where

$$
\alpha = \frac{8\big(\ln k + \ln(2/\beta)\big)}{\varepsilon}.
$$ {#eq-svt-above-alpha}

Then AboveThreshold is $(\alpha,\beta)$-accurate on $X$, and with probability at least $1-\beta$ it does not halt before query $k$.
:::

::: {.proof}
1. *The good event.* Let $G$ be the event that $|\hat T - T| < \alpha/2$ and $|\nu_i| < \alpha/2$ for every $i \le k$.
2. *Accuracy on $G$.* If $a_i = \top$, then $f_i(X) \ge \hat T - \nu_i > (T - \alpha/2) - \alpha/2 = T - \alpha$. If $a_i = \bot$, then $f_i(X) < \hat T - \nu_i < T + \alpha$.
3. *No early halt on $G$.* By step 2, a $\top$ at query $i$ needs $f_i(X) > T - \alpha$, which the hypothesis rules out for $i < k$.
4. *$G$ is likely.* The threshold noise has scale $2/\varepsilon$, so the tail formula gives $\Pr[|\hat T - T| \ge \alpha/2] = e^{-\varepsilon\alpha/4}$. Each $\nu_i$ has scale $4/\varepsilon$, so a union bound gives $\Pr[\max_{i\le k}|\nu_i| \ge \alpha/2] \le k e^{-\varepsilon\alpha/8}$.
5. With @eq-svt-above-alpha, $ke^{-\varepsilon\alpha/8} = k\cdot\frac{1}{k}\cdot\frac{\beta}{2} = \beta/2$. Also $e^{-\varepsilon\alpha/4} = (\beta/(2k))^2 \le \beta/2$. So $\Pr[G] \ge 1-\beta$.
:::

The error grows only logarithmically in $k$, like the exponential mechanism's, but the queries can arrive online.

::: {#exm-svt-numbers}
Take $k = 10^6$ counting queries, $\varepsilon = 1$ and $\beta = 0.05$. Then @eq-svt-above-alpha gives $\alpha = 8(\ln 10^6 + \ln 40) = 8(13.82 + 3.69) = 140.0$. On a dataset of $n = 10^5$ records, a query is placed correctly unless its count lies within $140$ of the threshold. Answering every query with the Laplace mechanism and basic composition would instead add noise of scale $10^6$, ten times the size of the dataset.
:::

## Sparse vector {#sec-svt-sparse}

To find $c$ large queries, restart AboveThreshold after each $\top$, with a fresh threshold, until $c$ of them have been found.

::: {#def-svt-sparse name="Sparse vector"}
The inputs are those of AboveThreshold and a count $c \ge 1$, with either $\delta = 0$ or $\delta \in (0,1/2]$. Set

$$
\sigma = \frac{2c}{\varepsilon} \ \text{ if } \delta = 0, \qquad \sigma = \frac{\sqrt{32c\ln(1/\delta)}}{\varepsilon} \ \text{ if } \delta > 0 .
$$

1. Draw $\hat T = T + \mathrm{Lap}(\sigma)$ and set $\mathrm{count} = 0$.
2. For each query $f_i$ in turn, draw $\nu_i \sim \mathrm{Lap}(2\sigma)$. If $f_i(X) + \nu_i < \hat T$, output $\bot$. Otherwise output $\top$ and add $1$ to $\mathrm{count}$. If $\mathrm{count} = c$, halt. If not, draw a fresh $\hat T = T + \mathrm{Lap}(\sigma)$ and continue.
:::

Between two $\top$ answers the algorithm runs AboveThreshold with parameter $\varepsilon' = 2/\sigma$, whose noise scales are $2/\varepsilon' = \sigma$ and $4/\varepsilon' = 2\sigma$. The fresh threshold matters: reusing $\hat T$ after a $\top$ would break this equivalence, and with it the proof.

::: {#thm-svt-sparse-privacy}
Sparse vector with $\delta = 0$ is $\varepsilon$-DP for every $\varepsilon > 0$. With $\varepsilon \in (0,1]$ and $\delta \in (0,1/2]$ it is $(\varepsilon,\delta)$-DP.
:::

::: {.proof}
1. The output is the concatenation of at most $c$ runs of AboveThreshold, each with parameter $\varepsilon' = 2/\sigma$ and fresh noise. Each run takes the rest of the stream, which may depend on the earlier outputs, so the whole is an adaptive composition of $c$ mechanisms. A run that never starts, because the algorithm halted, has a fixed output. Each run is $\varepsilon'$-DP by @thm-svt-above-privacy.
2. If $\delta = 0$, then $\varepsilon' = \varepsilon/c$, and basic composition gives $c\varepsilon' = \varepsilon$.
3. If $\delta > 0$, then $\varepsilon' = 2\varepsilon/\sqrt{32c\ln(1/\delta)} = \varepsilon/\big(2\sqrt{2c\ln(1/\delta)}\big)$. The [budget-splitting lemma](01-multiplicative-weights.md) says that $k$ adaptively composed mechanisms, each $\varepsilon/(2\sqrt{2k\ln(1/\delta)})$-DP, are together $(\varepsilon,\delta)$-DP when $\varepsilon \le 1$ and $\delta \le 1/2$. Apply it with $k = c$.
:::

::: {#thm-svt-sparse-accuracy name="Accuracy of sparse vector"}
Let $\beta \in (0,1)$, let $f_1, \dots, f_k$ be a stream, and let

$$
\alpha = 4\sigma\ln(2k/\beta) .
$$

With probability at least $1-\beta$, every answered query satisfies the two implications of @def-svt-accurate. If moreover fewer than $c$ of the queries have $f_i(X) > T - \alpha$, the algorithm answers all $k$ queries. In terms of the parameters, the accuracy is

$$
\alpha = \frac{8c\big(\ln k + \ln(2/\beta)\big)}{\varepsilon} \ \text{ if } \delta = 0, \qquad \alpha = \frac{\sqrt{512\,c\ln(1/\delta)}\,\big(\ln k + \ln(2/\beta)\big)}{\varepsilon} \ \text{ if } \delta > 0 .
$$ {#eq-svt-sparse-alpha}
:::

::: {.proof}
1. Let $G$ be the event that every threshold drawn and every $\nu_i$ with $i \le k$ lies within $\alpha/2$ of its centre. On $G$, steps 2 and 3 of the proof of @thm-svt-above-accuracy apply to each query, with whichever threshold is current. So the implications hold, and a $\top$ needs $f_i(X) > T - \alpha$.
2. If fewer than $c$ queries have $f_i(X) > T-\alpha$, then on $G$ there are fewer than $c$ answers $\top$, and the algorithm never halts.
3. The query noise has scale $2\sigma$, and $\alpha/(4\sigma) = \ln(2k/\beta)$. A union bound over the $k$ draws bounds its failure by $k e^{-\alpha/(4\sigma)} = \beta/2$.
4. At most $k+1$ thresholds are drawn, each with scale $\sigma$. Their failure is at most $(k+1)e^{-\alpha/(2\sigma)} = (k+1)\beta^2/(4k^2) \le \beta/2$, since $k + 1 \le 2k$ and $\beta \le 1$.
5. Substituting the two values of $\sigma$ into $4\sigma\ln(2k/\beta)$ gives @eq-svt-sparse-alpha, because $4\sqrt{32} = \sqrt{512}$.
:::

With $c = 1$ and $\delta = 0$ this is @thm-svt-above-accuracy again. The error is polynomial in the number $c$ of large queries, which every algorithm must pay in some form, and logarithmic in the total number $k$. So sparse vector can scan a very long stream, as long as few of its queries are large.

## Numeric sparse vector {#sec-svt-numeric}

Sparse vector already pays in proportion to the number of large queries. It can therefore also release their values, at the cost of a constant factor.

::: {#def-svt-numeric name="Numeric sparse vector"}
Run sparse vector with $\delta = 0$ and budget $\varepsilon/2$, so $\sigma = 4c/\varepsilon$. Whenever it would output $\top$ for query $f_i$, output instead $a_i = f_i(X) + \mathrm{Lap}(2c/\varepsilon)$, with fresh noise.
:::

::: {#thm-svt-numeric}
Numeric sparse vector is $\varepsilon$-DP. Let $\beta\in(0,1)$ and $\alpha = 16c\ln(4k/\beta)/\varepsilon$. With probability at least $1-\beta$, every answered query satisfies

$$
a_i = \bot \implies f_i(X) \le T + \alpha, \qquad a_i \in \mathbb R \implies f_i(X) \ge T - \alpha \ \text{ and } \ |a_i - f_i(X)| \le \alpha ,
$$

and if fewer than $c$ queries have $f_i(X) > T - \alpha$, all $k$ are answered.
:::

::: {.proof}
1. *Privacy.* The algorithm is an adaptive composition of at most $c$ runs of AboveThreshold, each with parameter $2/\sigma = \varepsilon/(2c)$, and at most $c$ Laplace mechanisms. Each Laplace mechanism measures one sensitivity-$1$ query, chosen from the earlier outputs, with scale $2c/\varepsilon$, so it is $\varepsilon/(2c)$-DP. Basic composition over the $2c$ mechanisms gives $\varepsilon$.
2. *Accuracy of the labels.* @thm-svt-sparse-accuracy with $\sigma = 4c/\varepsilon$ and failure probability $\beta/2$ gives accuracy $4\sigma\ln(4k/\beta) = \alpha$ for the labels, and no early halt, with probability at least $1-\beta/2$.
3. *Accuracy of the values.* At most $k$ values are released, one per query. By the tail formula and a union bound, some noise exceeds $\alpha$ with probability at most $k\,e^{-\alpha\varepsilon/(2c)} = k\,(\beta/(4k))^8 \le \beta/2$.
:::

The fresh noise in the released value matters. Releasing $f_i(X) + \nu_i$, the value already compared with the threshold, is not private, and published variants that did so were wrong.

## Online private multiplicative weights {#sec-svt-online-pmw}

The [previous chapter](01-multiplicative-weights.md) built private multiplicative weights for a fixed set of linear queries $q \colon \mathcal X \to [0,1]$, with $q(X) = \frac1n\sum_k q(X_k)$. Its key steps were the following. The histogram $p_X$ of the dataset, the fraction of records equal to each $x$, satisfies $q(X) = \langle q, p_X\rangle$. Multiplicative weights over the experts $\mathcal X$ with learning rate $\gamma$ and losses $s_tq^t$, where $s_t$ is the sign of the error $\langle q^t, p^t\rangle - q^t(X)$, has total error at most $\gamma U + \ln|\mathcal X|/\gamma$ after $U$ updates. And the offline algorithm used the exponential mechanism to find a query with a large error.

Online, the queries arrive one at a time, and the exponential mechanism cannot be used. Sparse vector replaces it. The algorithm keeps a distribution $\hat p$ and answers each query from it while it is accurate. When sparse vector reports a large error, the algorithm answers with a noisy true value and updates $\hat p$.

These queries have sensitivity $1/n$, not $1$. Running sparse vector on $f_i/\Delta$ with threshold $T/\Delta$ handles sensitivity $\Delta$: it is the same algorithm with every noise scale multiplied by $\Delta$, so it has the same privacy and $\Delta$ times the accuracy $\alpha$.

::: {#def-svt-online-pmw name="Online private multiplicative weights"}
The inputs are a dataset $X\in\mathcal X^n$, a stream of linear queries $f_1, f_2, \dots$, an accuracy $\alpha \in (0,1]$, and $\varepsilon \in (0,1]$ and $\delta\in(0,1/2]$. Set

$$
c = \left\lceil\frac{64\ln|\mathcal X|}{\alpha^2}\right\rceil, \qquad \gamma = \frac\alpha8, \qquad \varepsilon_0 = \frac{\varepsilon}{4\sqrt{c\ln(1/\delta)}} ,
$$

let $\hat p$ be uniform on $\mathcal X$, set $\mathrm{count} = 0$, and draw $\hat T = \alpha/2 + \mathrm{Lap}(2/(\varepsilon_0n))$. For each query $f_i$ in turn:

1. Let $e_i = |\langle f_i, \hat p\rangle - f_i(X)|$ and draw $\nu_i \sim \mathrm{Lap}(4/(\varepsilon_0n))$.
2. If $e_i + \nu_i < \hat T$, answer $\langle f_i, \hat p\rangle$.
3. Otherwise answer $a_i = f_i(X) + Z_i$ with $Z_i \sim\mathrm{Lap}(1/(\varepsilon_0 n))$. Let $s_i$ be the sign of $\langle f_i, \hat p\rangle - a_i$, update $\hat p(x) \propto \hat p(x)\big(1 - \gamma s_if_i(x)\big)$, and add $1$ to $\mathrm{count}$. If $\mathrm{count} = c$, halt. Otherwise draw a fresh $\hat T = \alpha/2 + \mathrm{Lap}(2/(\varepsilon_0n))$.
:::

Steps 1 to 3 follow numeric sparse vector, run on the errors $e_i$ with threshold $\alpha/2$, except that the fresh measurement in step 3 is of $f_i(X)$ rather than of $e_i$. The answer in step 2 uses only $\hat p$, which depends on the earlier outputs alone, so it costs no privacy.

::: {#thm-svt-online-pmw-privacy}
Online private multiplicative weights is $(\varepsilon,\delta)$-DP.
:::

::: {.proof}
1. Between two updates $\hat p$ is fixed, so the errors $e_i$ form a stream of queries determined by the earlier outputs and the analyst. Each $e_i$ has sensitivity $1/n$, since $\langle f_i,\hat p\rangle$ does not depend on $X$ and $f_i(X)$ changes by at most $1/n$.
2. Each stretch between updates is therefore AboveThreshold with parameter $\varepsilon_0$, rescaled to sensitivity $1/n$, and so $\varepsilon_0$-DP by @thm-svt-above-privacy. Each measurement in step 3 is the Laplace mechanism on a query of sensitivity $1/n$ with scale $1/(\varepsilon_0n)$, so it is also $\varepsilon_0$-DP. The answers of step 2 and the updates are post-processing.
3. There are at most $c$ stretches and $c$ measurements, so the algorithm is an adaptive composition of $2c$ mechanisms, each $\varepsilon_0$-DP. Since $\varepsilon_0 = \varepsilon/(2\sqrt{2\cdot 2c\ln(1/\delta)})$, the budget-splitting lemma with $k = 2c$ makes it $(\varepsilon,\delta)$-DP.
:::

::: {#thm-svt-online-pmw-accuracy name="Accuracy of online private multiplicative weights"}
Let $|\mathcal X|\ge 2$, $\beta\in(0,1)$ and a stream of $k$ queries, and let the other parameters be as in @def-svt-online-pmw. If

$$
n \ge \frac{128\sqrt{c\ln(1/\delta)}\,\ln\big((2c+k)/\beta\big)}{\alpha\varepsilon},
$$ {#eq-svt-online-n}

then with probability at least $1-\beta$ the algorithm answers all $k$ queries, each within $\alpha$ of $f_i(X)$.
:::

::: {.proof}
1. *The good event.* Let $G$ be the event that every threshold noise, every $\nu_i$ and every $Z_i$ is less than $\alpha/8$ in absolute value. These are at most $2c + k$ draws, because there are at most $c$ thresholds and $c$ measurements. The widest scale is $4/(\varepsilon_0n)$, so each draw fails with probability at most $e^{-\alpha\varepsilon_0n/32}$. Condition @eq-svt-online-n is the same as $\alpha\varepsilon_0n/32 \ge \ln((2c+k)/\beta)$, by the value of $\varepsilon_0$. A union bound gives $\Pr[G] \ge 1-\beta$, and the rest assumes $G$.
2. *Answers from $\hat p$.* In step 2, $e_i < \hat T - \nu_i < \alpha/2 + \alpha/8 + \alpha/8 = 3\alpha/4$, so the answer is within $\alpha$.
3. *Measured answers.* In step 3, $|a_i - f_i(X)| = |Z_i| < \alpha/8$. Also $e_i \ge \hat T - \nu_i > \alpha/2 - \alpha/4 = \alpha/4$. The difference $\langle f_i,\hat p\rangle - a_i$ is the true error minus $Z_i$, and $|Z_i| < \alpha/8 < e_i$, so $s_i$ is the sign of the true error.
4. *Few updates.* By step 3, the updates are those of multiplicative weights for queries with $\gamma = \alpha/8$, and each has error above $\alpha/4$. If there are $U$ of them, the regret bound gives

$$
\frac\alpha4U < \frac\alpha8U + \frac{8\ln|\mathcal X|}{\alpha}, \qquad\text{that is,}\qquad U < \frac{64\ln|\mathcal X|}{\alpha^2} \le c .
$$

So the count never reaches $c$, the algorithm never halts, and all $k$ queries are answered.
:::

As in the previous chapter, $c \le 66\ln|\mathcal X|/\alpha^2$. So condition @eq-svt-online-n asks for

$$
n = O\!\left(\frac{\sqrt{\log|\mathcal X|\,\log(1/\delta)}\;\log\frac{k + \log|\mathcal X|/\alpha^2}{\beta}}{\alpha^2\varepsilon}\right)
$$

records. This matches the offline algorithm, with the length $k$ of the stream in place of the number $|Q|$ of queries. Sparse vector thus turns an algorithm that searches with the exponential mechanism into one that answers queries as they arrive.

## Summary {#sec-svt-summary}

| Result | Statement |
| --- | --- |
| @thm-svt-above-privacy | AboveThreshold, with noise on the threshold and on each query, is $\varepsilon$-DP |
| @thm-svt-above-accuracy | it finds the first large query with error $8(\ln k + \ln(2/\beta))/\varepsilon$ |
| @thm-svt-sparse-privacy | sparse vector, $c$ restarts with fresh thresholds, is $\varepsilon$-DP or $(\varepsilon,\delta)$-DP |
| @thm-svt-sparse-accuracy | error $8c(\ln k + \ln(2/\beta))/\varepsilon$, or $\sqrt{512c\ln(1/\delta)}(\ln k+\ln(2/\beta))/\varepsilon$ |
| @thm-svt-numeric | releasing the large values with fresh noise costs a constant factor |
| @thm-svt-online-pmw-privacy | online private multiplicative weights is $(\varepsilon,\delta)$-DP |
| @thm-svt-online-pmw-accuracy | it answers $k$ online queries to accuracy $\alpha$ from $n = O(\sqrt{\log\lvert\mathcal X\rvert\log(1/\delta)}\log k/(\alpha^2\varepsilon))$ records, up to lower-order logarithmic terms |

: The chapter's results. {#tbl-svt-summary}

## Notes {#sec-svt-notes}

- Cynthia Dwork, Moni Naor, Omer Reingold, Guy N. Rothblum and Salil Vadhan, "On the Complexity of Differentially Private Data Release: Efficient Algorithms and Hardness Results", 2009. The first use of the sparse vector idea.
- Aaron Roth and Tim Roughgarden, "Interactive Privacy via the Median Mechanism", 2010, and Moritz Hardt and Guy N. Rothblum, "A Multiplicative Weights Mechanism for Privacy-Preserving Data Analysis", 2010. Online query answering with sparse vector, including online private multiplicative weights.
- Cynthia Dwork and Aaron Roth, *The Algorithmic Foundations of Differential Privacy*, 2014. Section 3.6 presents AboveThreshold, sparse vector and its numeric variant in the form used here, and Section 4.2 the online multiplicative weights mechanism.
- Min Lyu, Dong Su and Ninghui Li, "Understanding the Sparse Vector Technique for Differential Privacy", 2017. A survey of published variants, several of them not private, including those that reuse the threshold or release the compared noisy values.
