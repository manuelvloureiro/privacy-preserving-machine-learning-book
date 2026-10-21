# What ε means, and converting between definitions

A **mechanism** $M$ is a randomized algorithm: it takes a dataset and returns a random output. Two datasets, written $D$ and $D'$, are **neighbours** when they differ in one person's record. The definition of privacy has two parameters, a multiplicative limit $\varepsilon$ and a small additive slack $\delta$. The mechanism is **$(\varepsilon, \delta)$-differentially private** when, for every pair of neighbours and every set $S$ of outputs,
$$\Pr[M(D) \in S] \le e^\varepsilon\,\Pr[M(D') \in S] + \delta.$$

The **privacy loss** of an output $y$ is the log of how much more likely $y$ is when the data are $D$ than when they are $D'$. Write $p(y)$ and $q(y)$ for the probability (or density) of $y$ under $M(D)$ and $M(D')$. Then the privacy loss $L$ is
$$L(y) = \ln\frac{p(y)}{q(y)}.$$
Pure differential privacy (DP), the case $\delta = 0$, is written $\varepsilon$-DP. It is the same as asking that $|L(y)| \le \varepsilon$ for every output. One direction holds because a single output is itself a set of outputs, and the other because a set's probability is the sum over its outputs. The absolute value appears because neighbours come in both orders, so the definition also holds with $D$ and $D'$ swapped. @sec-apx-cal-dp builds these definitions slowly, with examples.

This appendix answers two practical questions. The first is what a number like $\varepsilon = 1$ promises. The first two sections read the definition as a limit on any test an attacker can run, and as a limit on how far an attacker's belief can move. The second question is how to report $(\varepsilon, \delta)$ when your accountant tracks something else. An **accountant** is the code that adds up the privacy cost of many releases from the same data, which is called composition. Accountants track Rényi differential privacy (RDP), zero-concentrated differential privacy (zCDP) or Gaussian differential privacy (GDP), because those compose more simply, and the last three sections convert each of them to $(\varepsilon, \delta)$. @tbl-apx-conv-map lists the five results and the chapter sections that state or use them.

| Section | What it derives | Stated or used in |
| --- | --- | --- |
| @sec-apx-conv-testing | Every test has $\mathrm{FPR} + e^\varepsilon\,\mathrm{FNR} \ge 1 - \delta$ | @sec-dpdef-testing, @sec-dpdef-advantage, @sec-gdp-fdp |
| @sec-apx-conv-bayes | Posterior odds are at most $e^\varepsilon$ times the prior odds | @sec-dpdef-bayes |
| @sec-apx-conv-rdp | RDP gives $\varepsilon = \varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1)$ | @sec-rdp-conversion, @sec-conv-mironov, @sec-conv-improved |
| @sec-apx-conv-zcdp | $\rho$-zCDP gives $\varepsilon = \rho + 2\sqrt{\rho\ln(1/\delta)}$ | @sec-cdp-definition, @sec-cdp-conversion |
| @sec-apx-conv-gdp | $\mu$-GDP: the curve $G_\mu$ and its exact $(\varepsilon, \delta)$ | @sec-gdp-gdp, @sec-gdp-duality, @sec-gdp-composition, @sec-lg-analytic |

: The results of this appendix and the chapter sections that state or use them. {#tbl-apx-conv-map}

**A running example.** Most sections below use one mechanism, the Gaussian mechanism. It adds normal noise to a query, and the noise has standard deviation $\sigma$. Here the query's **sensitivity** $\Delta$ is the most its value can move between neighbours, measured as a Euclidean distance when the query returns a vector.

Dividing the output by $\sigma$, and then subtracting the query's value on $D$, changes nothing about privacy, because you can undo both steps. After them, the output on $D$ is a unit normal centred at 0, and the output on $D'$ is a unit normal centred a distance $\mu = \Delta/\sigma$ away, for the neighbours whose query values are the full sensitivity apart. A smaller shift is harder to detect, so this pair is the worst one (@sec-apx-conv-gdp shows why). So the two output distributions are
$$P = \mathcal N(0, 1) \text{ on } D, \qquad Q = \mathcal N(\mu, 1) \text{ on } D'.$$
Taking the log of the ratio of the two normal densities, the $y^2$ terms cancel and the privacy loss is a straight line in the output,
$$L(y) = -\frac{y^2}{2} + \frac{(y - \mu)^2}{2} = \frac{\mu^2}{2} - \mu y.$$
When $y$ is drawn from the distribution $P$, so that $y$ is standard normal, the loss is a shifted and scaled normal variable, which is again normal. It is therefore normal with mean $\mu^2/2$ and variance $\mu^2$. Throughout, the standard normal distribution function $\Phi$ gives the probability that a standard normal variable $Z$ is at most $x$,
$$\Phi(x) = \Pr[Z \le x],$$
and $\Phi^{-1}$ is its inverse. Unless a section says otherwise, the numbers use $\delta = 10^{-5}$, and logarithms are natural.

## What ε means for any test {#sec-apx-conv-testing}

### The question

An attacker sees the output and must guess whether the data were $D$ or $D'$, for example whether your record was used. It may know everything else and compute anything. What does $\varepsilon = 1$ stop it from doing?

### The answer

Any guessing rule can be written as a set $S$ of outputs: the attacker says "$D'$" when the output lands in $S$, and "$D$" otherwise. It can be wrong in two ways. Its **false positive rate (FPR)** is how often it says "$D'$" when the data were $D$, so $\mathrm{FPR} = \Pr[M(D) \in S]$. Its **false negative rate (FNR)** is how often it says "$D$" when the data were $D'$, so $\mathrm{FNR} = \Pr[M(D') \notin S]$. If $M$ is $(\varepsilon, \delta)$-DP, then every test, against every pair of neighbours, satisfies both
$$\mathrm{FPR} + e^\varepsilon\,\mathrm{FNR} \ge 1 - \delta \qquad\text{and}\qquad e^\varepsilon\,\mathrm{FPR} + \mathrm{FNR} \ge 1 - \delta.$$ {#eq-apx-conv-testing}
The chapter on the definition (@sec-dpdef-testing) writes $\alpha$ and $\beta$ for the two error rates. This page needs $\alpha$ later for something else, so it writes the rates out. There are two inequalities because the definition holds in both orders of the pair.

### A worked example

Take $\varepsilon = 1$ and $\delta = 0$, and an attacker who accepts 5% false positives. The first inequality says $0.05 + e\,\mathrm{FNR} \ge 1$, so $\mathrm{FNR} \ge 0.95/e = 0.3495$. The second says $e \times 0.05 + \mathrm{FNR} \ge 1$, so $\mathrm{FNR} \ge 1 - 0.1359 = 0.8641$. Both must hold, so the attacker misses at least **86.4%** of the time, and it catches $D'$ at most 13.6% of the time.

Notice which inequality did the work. The first one alone would allow a miss rate as low as 35%. At a low FPR the second one is much stronger. Low FPR is exactly where attackers want to operate, because an attack that rarely raises a false alarm is one whose accusations can be trusted. For comparison, a coin that says "$D'$" 5% of the time, ignoring the output, catches $D'$ 5% of the time. So $\varepsilon = 1$ lets the best attacker catch $D'$ at most $0.136/0.05 = e \approx 2.7$ times as often as that coin.

Other false positive rates give other limits. At $\mathrm{FPR} = 0.01$ the attacker misses at least 0.9728 of the time, and at $\mathrm{FPR} = 0.25$ at least 0.3204. With $\delta = 10^{-5}$ the limit at $\mathrm{FPR} = 0.05$ drops only to 0.864076, because $\delta$ lowers each of the two bounds by exactly $\delta$.

### Why it works

Each inequality is the definition applied to one set. The proof also uses one rule twice: the probability that the output is outside $S$ is one minus the probability that it is inside, so $\Pr[M(D) \notin S] = 1 - \mathrm{FPR}$ and $\Pr[M(D') \in S] = 1 - \mathrm{FNR}$.

1. **The first inequality.** Apply the definition to the set of outputs where the test says "$D$", the complement of $S$. It gives $\Pr[M(D) \notin S] \le e^\varepsilon\Pr[M(D') \notin S] + \delta$. The left side is $1 - \mathrm{FPR}$ and the probability on the right is $\mathrm{FNR}$. So $1 - \mathrm{FPR} \le e^\varepsilon\,\mathrm{FNR} + \delta$, which rearranges to the first inequality.
2. **The second inequality.** If $D$ and $D'$ are neighbours, so are $D'$ and $D$, so the definition also holds with the two datasets swapped. Apply it in that order to $S$ itself: $\Pr[M(D') \in S] \le e^\varepsilon\Pr[M(D) \in S] + \delta$. The left side is $1 - \mathrm{FNR}$ and the probability on the right is $\mathrm{FPR}$, which gives the second inequality.
3. **Tests that flip a coin.** An attacker may also randomize, saying "$D'$" with some probability that depends on the output. In that case, think of its coin $U$ as part of the output. Releasing $(M(D), U)$, with $U$ independent of the data, is still $(\varepsilon, \delta)$-DP, because for each value of $U$ the inequality holds for the matching slice of the set, and averaging both sides over $U$ keeps it. So steps 1 and 2 apply to randomized tests too.

**A tiny case shows that the bound is reached.** In randomized response, one person reports a bit: the true bit with probability $e^\varepsilon/(1 + e^\varepsilon)$, the flipped bit otherwise. Each report is exactly $e^\varepsilon$ times as likely under one value of the bit as under the other, so the mechanism is $\varepsilon$-DP. Let the bit be 1 under $D$ and 0 under $D'$, and let the test say "$D'$" when the report is 0. It errs when the report is flipped, in either world, so $\mathrm{FPR} = \mathrm{FNR} = 1/(1 + e^\varepsilon)$, which is 0.2689 at $\varepsilon = 1$. Then
$$\mathrm{FPR} + e^\varepsilon\,\mathrm{FNR} = \frac{1 + e^\varepsilon}{1 + e^\varepsilon} = 1,$$
so the first inequality holds with equality, and by symmetry so does the second. So randomized response meets both bounds at once, and neither inequality can be strengthened if it is to hold for every $\varepsilon$-DP mechanism.

### Checking it

The script for this page (see @sec-apx-conv-code) lists every possible test against three people who each run randomized response with $\varepsilon = \ln 3$, for neighbours that differ in the first person's bit. There are 8 report vectors, and a test is any subset of them, so there are $2^8 = 256$ tests. The smallest slack, the amount by which a test clears the tighter of the two inequalities, is exactly 0: no test breaks the bound, and several tests meet it.

@fig-apx-conv-testing-region draws the forbidden region, with the FPR across and the FNR up. Each pair of lines is the lower edge of where a test can be, and a smaller $\varepsilon$ pushes that edge up towards the diagonal $\mathrm{FPR} + \mathrm{FNR} = 1$, which is what a test that ignores the output achieves. The dots are randomized response, which sits on the corner of each region.

![The false positive and false negative rates that tests can reach under $\varepsilon$-DP with $\delta = 0$, for $\varepsilon = 0.5$, 1 and 2. The shaded area below each pair of lines is forbidden for every test. The dots mark randomized response at the same $\varepsilon$, the cross marks the worked example at 5% false positives, and the dotted diagonal is guessing without looking at the output.](figures/conv_testing_region.svg){#fig-apx-conv-testing-region}

### Recipe and pitfalls

To turn a guarantee into an attack limit, pick the false positive rate you care about. The attacker's true positive rate (TPR) is the fraction of the time it correctly says "$D'$", so $\mathrm{TPR} = 1 - \mathrm{FNR}$. The second inequality caps it at
$$\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta.$$
Auditing, which tests a privacy claim by attacking the system, runs this in reverse. If an attack measures a TPR and an FPR against a trained model, the mechanism's $\varepsilon$ must be at least $\ln\big((\mathrm{TPR} - \delta)/\mathrm{FPR}\big)$.

Three things go wrong in practice. The bound is for the worst mechanism with this $\varepsilon$, and most mechanisms sit strictly inside it, so a weak attack says little about how private the mechanism really is. At very small FPR the $\delta$ term dominates, since at $\mathrm{FPR} = 0$ the bound still allows a TPR of $\delta$, so $\delta$ must be tiny. And the guarantee covers only the mechanism's own kind of neighbours. A guarantee for neighbours that add or remove one record says less about neighbours that replace one record with another. @sec-dpdef-testing proves the converse, that the two inequalities for all tests are the same as $(\varepsilon, \delta)$-DP, and @sec-dpdef-advantage turns them into a bound on membership advantage.

## What ε means for an attacker's belief {#sec-apx-conv-bayes}

### The question

People often ask a different question: "after the release, how sure can someone be about my record?" Take the strongest attacker. It knows every record except yours, so it knows the data are one of two neighbours, $D$ or $D'$, which differ only in your record. Before the release it believes $D$ with some prior probability $\pi$. How large can its belief be afterwards?

### The answer

The **odds** of $D$ are its probability divided by the probability of $D'$, so the prior odds are $\pi/(1 - \pi)$. If $M$ is $\varepsilon$-DP, then after any output $y$ the posterior odds are within a factor $e^\varepsilon$ of the prior odds, and the posterior probability is capped:
$$\frac{\Pr[D \mid y]}{\Pr[D' \mid y]} \le e^\varepsilon\,\frac{\pi}{1 - \pi}, \qquad \Pr[D \mid y] \le \frac{e^\varepsilon\pi}{e^\varepsilon\pi + 1 - \pi}.$$ {#eq-apx-conv-posterior}
The same argument with $e^{-\varepsilon}$ gives the matching lower limits.

### A worked example

Take $\varepsilon = 1$ and a prior of 1%, so the prior odds are $0.01/0.99 = 0.010101$. The output can multiply them by at most $e$, which brings them to $0.027457$. Any odds $o$ turn back into a probability as $o/(1 + o)$, so the posterior is at most $0.027457/1.027457 = 0.0267$. In words, a suspicion of 1% can rise to at most **2.7%**. With a prior of 50% the prior odds are 1, so the posterior odds are at most $e$ and the posterior is at most $e/(1 + e) = 0.7311$, or **73%**.

@tbl-apx-conv-posterior gives the cap for other values. Small priors stay small unless $\varepsilon$ is large: at $\varepsilon = 4$ a prior of 1% can rise to 36%.

| $\varepsilon$ | Prior 0.01 | Prior 0.5 |
| --- | --- | --- |
| 0.5 | 0.0164 | 0.6225 |
| 1 | 0.0267 | 0.7311 |
| 2 | 0.0695 | 0.8808 |
| 4 | 0.3555 | 0.9820 |

: The largest posterior probability of $D$ after one output of an $\varepsilon$-DP mechanism, from @eq-apx-conv-posterior. {#tbl-apx-conv-posterior}

### Why it works

The argument is Bayes' rule written in odds. A table with the two hypotheses as rows shows it (@tbl-apx-conv-bayes-table). Each row multiplies the prior by the chance of seeing $y$ under that hypothesis, and the posterior divides each product by their sum $Z$.

| Hypothesis | Prior | Chance of $y$ | Prior × chance | Posterior |
| --- | --- | --- | --- | --- |
| $D$ | $\pi$ | $p(y)$ | $\pi\,p(y)$ | $\pi\,p(y)/Z$ |
| $D'$ | $1 - \pi$ | $q(y)$ | $(1 - \pi)\,q(y)$ | $(1 - \pi)\,q(y)/Z$ |

: Bayes' rule for two hypotheses, where $Z = \pi\,p(y) + (1 - \pi)\,q(y)$ is the total chance of seeing $y$. {#tbl-apx-conv-bayes-table}

1. **Divide the two posteriors.** The sum $Z$ is the same in both rows, so it cancels in the ratio. What is left is the prior odds times the ratio of the chances, which is $e^{L(y)}$:
$$\frac{\Pr[D \mid y]}{\Pr[D' \mid y]} = \frac{\pi}{1 - \pi}\cdot\frac{p(y)}{q(y)} = \frac{\pi}{1 - \pi}\,e^{L(y)}.$$
2. **Use pure DP.** Under $\varepsilon$-DP, $L(y) \le \varepsilon$ for every output, so $e^{L(y)} \le e^\varepsilon$. This gives the odds bound of @eq-apx-conv-posterior.
3. **Turn odds into a probability.** The map $o \mapsto o/(1 + o)$ grows with $o$, so the largest odds give the largest probability. Put in $o = e^\varepsilon\pi/(1 - \pi)$ and multiply top and bottom by $1 - \pi$ to get the probability bound.

With numbers, take randomized response at $\varepsilon = 1$, your bit 1 under $D$, and a report of 1. Then $p(y) = e/(1 + e) = 0.7311$ and $q(y) = 0.2689$. With a prior of 1%, the products are $0.01 \times 0.7311 = 0.007311$ and $0.99 \times 0.2689 = 0.266252$. Their sum is $Z = 0.273563$, so the posterior is $0.007311/0.273563 = 0.0267$, exactly the cap.

**With δ.** Under $(\varepsilon, \delta)$-DP the privacy loss is not bounded everywhere, so the bound cannot hold for every output. It is tempting to say that it holds except on outputs of total probability $\delta$, but that is false in general. For example, @sec-apx-conv-gdp shows that the Gaussian running example with $\mu = 1$ is $(4.3772, 10^{-5})$-DP. Yet, when the data are $D$, its privacy loss exceeds 4.3772 with probability $5.284\times10^{-5}$, more than five times $\delta$.

What does hold is the same bound at a larger level $\varepsilon' > \varepsilon$, except on a set of outputs whose probability is a small multiple of $\delta$. To prove it, let $B$ be the set of outputs whose loss exceeds $\varepsilon'$, and write $P(B)$ and $Q(B)$ for its probability under $D$ and $D'$.

1. The definition, applied to $B$, says $P(B) - e^\varepsilon Q(B) \le \delta$.
2. On $B$ each output has $q(y) < e^{-\varepsilon'}p(y)$, because its loss exceeds $\varepsilon'$. Adding over $B$ gives $Q(B) < e^{-\varepsilon'}P(B)$.
3. Put step 2 into step 1. Since $Q(B)$ is less than $e^{-\varepsilon'}P(B)$, replacing it by $e^{-\varepsilon'}P(B)$ makes the left side smaller, and the left side was already at most $\delta$. So $P(B)(1 - e^{\varepsilon - \varepsilon'}) < \delta$, and therefore
$$P(B) < \frac{\delta}{1 - e^{\varepsilon - \varepsilon'}}.$$ {#eq-apx-conv-bad-set}

Outside $B$ the loss is at most $\varepsilon'$, so steps 1 to 3 of the pure case go through with $\varepsilon'$ in place of $\varepsilon$. So the posterior bound holds at level $\varepsilon'$, except on outputs whose probability under $D$ is at most $\delta/(1 - e^{\varepsilon - \varepsilon'})$. With $\varepsilon' = \varepsilon + 1$ that exception is $\delta/(1 - e^{-1}) = 1.582\times10^{-5}$. For the Gaussian example the true probability of a loss above $4.3772 + 1$ is $5.381\times10^{-7}$, well inside the bound.

### Checking it

The script draws 2,000,000 people whose bit is 1 with probability 1% and runs randomized response at $\varepsilon = 1$. Among the reports of 1, the fraction that came from a true 1 is 0.0267, the cap of @eq-apx-conv-posterior. Randomized response reaches the cap, so the cap cannot be lowered.

@fig-apx-conv-posterior shows the mechanism of the proof. In log odds, the output adds $L(y)$ to the prior's log odds, so $\varepsilon$-DP keeps the posterior inside a band of half-width $\varepsilon$ around the diagonal. In probabilities the same band bends. The reason is that turning log odds back into a probability squeezes large log odds towards 1 and very negative ones towards 0, while leaving the middle stretched.

![The largest (solid) and smallest (dashed) posterior of $D$ after one output of an $\varepsilon$-DP mechanism, for $\varepsilon = 0.5$, 1, 2 and 4. Left: on the scale of log odds the posterior stays within a band of half-width $\varepsilon$ around the prior (dotted diagonal). Right: the same band on the scale of probabilities, with the two worked examples at $\varepsilon = 1$ marked.](figures/conv_posterior.svg){#fig-apx-conv-posterior}

### Recipe and pitfalls

To explain $\varepsilon$ to someone who thinks in probabilities, ask for their prior $\pi$ and give them the cap $e^\varepsilon\pi/(e^\varepsilon\pi + 1 - \pi)$. The guarantee limits the *change* in belief, not the belief itself. An attacker who is already 99% sure stays about that sure, and DP promises only that the output does not make it much surer. The bound is per person and per release. For correlated records, such as a household, first work out the guarantee for the whole group, and for repeated releases first add up their privacy cost. With $\delta > 0$, quote the bound at a larger level $\varepsilon'$ together with the exception of @eq-apx-conv-bad-set. @sec-dpdef-bayes states the result and discusses weaker attackers.

## From Rényi DP to (ε, δ) {#sec-apx-conv-rdp}

### The question

Your accountant tracks RDP because RDP composes by adding. At the end it hands you a curve, one number for each order $\alpha$ (defined next), and you must report $(\varepsilon, \delta)$. How do you turn the curve into $\varepsilon$?

The **Rényi divergence** of order $\alpha > 1$ is a log average of a power of the likelihood ratio $p/q$. Written with the privacy loss, with $y$ drawn from $P = M(D)$, it is
$$D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\ln\mathbb E_P\big[e^{(\alpha - 1)L}\big].$$
The expectation is the average of $(p/q)^{\alpha - 1}$ weighted by $p$, which is the integral of $p^\alpha q^{1 - \alpha}$, the form in which it is often defined. A mechanism has **RDP curve** $\varepsilon(\alpha)$ when $D_\alpha(M(D)\|M(D')) \le \varepsilon(\alpha)$ for every pair of neighbours. Multiplying by $\alpha - 1$ and exponentiating, this says that
$$\mathbb E_P\big[e^{(\alpha - 1)L}\big] \le e^{(\alpha - 1)\varepsilon(\alpha)}.$$ {#eq-apx-conv-mgf}
In words, the curve bounds the moment generating function (MGF) of the privacy loss, the average of $e^{sL}$, at the point $s = \alpha - 1$.

### The answer

For every order $\alpha > 1$, a mechanism with RDP curve $\varepsilon(\alpha)$ is $(\varepsilon, \delta)$-DP with
$$\varepsilon = \varepsilon(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}.$$ {#eq-apx-conv-rdp}
Every order gives a valid guarantee, so you report the smallest $\varepsilon$ over the orders you have computed, usually a fixed grid. This is Mironov's conversion.

### A worked example

For the Gaussian running example with $\mu = \Delta/\sigma = 1$, the curve is $\varepsilon(\alpha) = \alpha/2$ (step 4 below derives it). With $\delta = 10^{-5}$, the second term's numerator is $\ln(1/\delta) = 11.5129$. So at order 2 the conversion gives $1 + 11.5129 = 12.5129$, and at order 32 it gives $16 + 11.5129/31 = 16.3714$. Between the two, at order 5.8, it gives $2.9 + 11.5129/4.8 = 5.2985$, which is the best possible (step 5). Yet the exact $\varepsilon$ of this mechanism at $\delta = 10^{-5}$ is 4.3772 (@sec-apx-conv-gdp), so the conversion overstates it by 0.9213.

### Why it works

The proof has two moves. Markov's inequality turns the MGF bound into a bound on the chance that the loss is large. Then a small loss everywhere except on a rare set gives $(\varepsilon, \delta)$-DP.

1. **Markov's inequality on the loss.** Fix a target level $\varepsilon$. Markov's inequality says that a quantity $X$ that is never negative exceeds a level $t$ with probability at most $\mathbb E[X]/t$. The reason is that those outcomes alone already contribute at least $t$ times their probability to the average. The quantity $e^{(\alpha - 1)L}$ is never negative, and it exceeds $e^{(\alpha - 1)\varepsilon}$ exactly when $L > \varepsilon$, since $\alpha - 1 > 0$. So, using @eq-apx-conv-mgf in the last step,
$$\Pr_P[L > \varepsilon] \le \frac{\mathbb E_P\big[e^{(\alpha - 1)L}\big]}{e^{(\alpha - 1)\varepsilon}} \le e^{(\alpha - 1)(\varepsilon(\alpha) - \varepsilon)}.$$
2. **Solve for ε.** Set the right side equal to $\delta$ and take logs: $(\alpha - 1)(\varepsilon(\alpha) - \varepsilon) = -\ln(1/\delta)$. Solving for $\varepsilon$ gives @eq-apx-conv-rdp. So at that $\varepsilon$, the loss exceeds $\varepsilon$ with probability at most $\delta$.
3. **A rare large loss gives (ε, δ)-DP.** Take any set $S$ of outputs and split it in two. On the part where $L \le \varepsilon$, each output has $p(y) \le e^\varepsilon q(y)$, so that part has $P$-probability at most $e^\varepsilon$ times its $Q$-probability, which is at most $e^\varepsilon Q(S)$. The other part lies inside the set where $L > \varepsilon$, which has $P$-probability at most $\delta$ by step 2. Adding the two parts gives $P(S) \le e^\varepsilon Q(S) + \delta$. RDP also bounds the divergence with the two datasets swapped, because swapped neighbours are still neighbours, so the same argument gives the other direction. @sec-apx-cal-gauss-classic uses the same split to calibrate the Gaussian mechanism. Note the direction: a rare large loss is enough for $(\varepsilon, \delta)$-DP, but @sec-apx-conv-bayes showed that $(\varepsilon, \delta)$-DP does not force the large loss to be that rare.

A tiny case shows step 3 at work, before the Gaussian. Take three outputs with probabilities $p = (0.6, 0.39, 0.01)$ under $D$ and $q = (0.6, 0.3999, 0.0001)$ under $D'$, and $\varepsilon = 0.1$. Only the third output has a loss above 0.1, namely $\ln 100 = 4.605$, and its probability under $D$ is 0.01. Step 3 then promises $(0.1, 0.01)$-DP. Checking all 8 sets directly, the largest value of $P(S) - e^{0.1}Q(S)$ is 0.00989, just below the promised 0.01.

4. **The Gaussian's curve.** Under $P$ the loss is normal with mean $\mu^2/2$ and variance $\mu^2$, so its MGF is that of a normal variable. For a normal variable $X$ with mean $m$ and variance $v$, that MGF is
$$\mathbb E[e^{sX}] = e^{sm + s^2v/2}.$$
So with $s = \alpha - 1$, $m = \mu^2/2$ and $v = \mu^2$, the exponent is $(\alpha - 1)\mu^2/2 + (\alpha - 1)^2\mu^2/2$, which factors as $(\alpha - 1)\,\alpha\mu^2/2$. Dividing by $\alpha - 1$ then gives $\varepsilon(\alpha) = \alpha\mu^2/2$, a straight line through the origin. @sec-apx-ccomp-rdp derives it in more detail and shows why curves add under composition.
5. **The best order.** The conversion is $g(\alpha) = \alpha\mu^2/2 + \ell/(\alpha - 1)$, where $\ell = \ln(1/\delta)$. The first term grows with the order and the second shrinks, so the best order is in between. The slope is $g'(\alpha) = \mu^2/2 - \ell/(\alpha - 1)^2$, which is zero when $(\alpha - 1)^2 = 2\ell/\mu^2$. The first term is a straight line and the second curves upward, so their sum curves upward and this point is its minimum:
$$\alpha^* = 1 + \frac{\sqrt{2\ell}}{\mu}, \qquad \varepsilon = \frac{\mu^2}{2} + \mu\sqrt{2\ell}.$$
For the second formula, the first term at $\alpha^*$ is $\mu^2/2 + \mu\sqrt{2\ell}/2$, and the second is $\ell\mu/\sqrt{2\ell} = \mu\sqrt{2\ell}/2$. At $\mu = 1$, $\sqrt{2 \times 11.5129} = 4.7985$, so $\alpha^* = 5.7985$ and $\varepsilon = 0.5 + 4.7985 = 5.2985$.

**The improved conversion.** Step 3 is wasteful. The smallest valid $\delta$ is the largest value of $P(S) - e^\varepsilon Q(S)$ over sets $S$. That largest value comes from the set of outputs with $L > \varepsilon$, and each such output adds $p - e^\varepsilon q = p\,(1 - e^{\varepsilon - L})$. So the exact $\delta$ counts an output with $L > \varepsilon$ only in proportion $1 - e^{\varepsilon - L}$, which is small when $L$ is just above $\varepsilon$, while step 3 charges it in full. The improved conversion (Balle et al. 2020; Canonne, Kamath & Steinke 2020) bounds that proportion by the smallest multiple of $e^{(\alpha - 1)L}$ that lies above it, whose average @eq-apx-conv-mgf controls. It gives
$$\varepsilon = \varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}.$$
@sec-conv-improved derives it. For the Gaussian with $\mu = 1$ its best order is 5.43, where it gives 4.7284. It saves 0.5701 of the 0.9213 by which Mironov's conversion overstates the exact 4.3772.

### Checking it

The script averages $e^{2L}$ over 2,000,000 Gaussian outputs; the log of that average, divided by 2, estimates the Rényi divergence of order 3 as 1.4965, against $\alpha\mu^2/2 = 1.5$ from step 4. At Mironov's $\varepsilon = 5.2985$ the chance that the loss exceeds $\varepsilon$ is $7.992\times10^{-7}$, below the $10^{-5}$ that step 1 guarantees. The exact $\delta$ at that $\varepsilon$ is smaller still, $1.300\times10^{-7}$. So the true $\delta$ is 76.9 times smaller than the $10^{-5}$ you report. At the improved conversion's 4.7284 the exact $\delta$ is $2.098\times10^{-6}$, so it too holds with room to spare.

@fig-apx-conv-rdp-order draws the conversion as a function of the order. The dashed line and the dashed hyperbola are its two terms, and their sum has a single minimum at $\alpha^* = 5.80$. The improved conversion lies below at every order, and neither reaches the exact value. The reason is that the RDP curve keeps only the MGF values of the privacy loss, so a conversion from it must allow for every loss distribution with those values.

![The two terms of Mironov's conversion for the Gaussian mechanism with $\mu = 1$ at $\delta = 10^{-5}$: the Rényi curve $\alpha/2$, which grows, and $\ln(1/\delta)/(\alpha - 1)$, which shrinks. Their sum (red) is smallest at $\alpha^* = 5.80$. The improved conversion (purple) and the exact $\varepsilon$ of the Gaussian (dotted) are shown for comparison.](figures/conv_rdp_order.svg){#fig-apx-conv-rdp-order}

### Recipe and pitfalls

Compose in RDP, by adding the curves at each order of a fixed grid, such as the 154 orders from 1.1 to 512 that Part 4 uses. Then convert once, at the end, taking the smallest $\varepsilon$ over the grid. Use the improved formula, because it is never worse and needs one more line of code. It can return a negative number when $\varepsilon(\alpha)$ is tiny, and then you report 0.

Three things go wrong. Converting each step to $(\varepsilon, \delta)$ and then adding the steps' $\varepsilon$'s (basic composition) wastes most of the advantage of RDP. A grid can miss the best order. The conversion is flat near its minimum, so a grid point close to the best order loses little, but a grid with no point near it can lose a lot. And the curve must hold for every pair of neighbours under the relation you claim. @sec-conv-mironov measures where each step of the proof loses, and @sec-conv-order explains how to choose the grid.

## From zero-concentrated DP to (ε, δ) {#sec-apx-conv-zcdp}

### The question

When every step adds Gaussian noise, the RDP curve is a straight line through the origin. A line is fixed by its slope, so you could keep one number instead of a curve. What is that number, and what $(\varepsilon, \delta)$ does it give?

### The answer

A mechanism is **$\rho$-zCDP** when its RDP curve lies under the line of slope $\rho$ through the origin: $\varepsilon(\alpha) \le \rho\alpha$ for every order $\alpha > 1$. The word "concentrated" in the name says that the privacy loss then has tails no heavier than a normal variable's. Such a mechanism is $(\varepsilon, \delta)$-DP with
$$\varepsilon = \rho + 2\sqrt{\rho\ln(1/\delta)}.$$ {#eq-apx-conv-zcdp}
Three facts make $\rho$ easy to track. First, the Gaussian mechanism with sensitivity $\Delta$ and noise $\sigma$ is $\rho$-zCDP with $\rho = \Delta^2/(2\sigma^2)$. Second, running several mechanisms adds their $\rho$'s. Third, an $\varepsilon$-DP mechanism, such as a Laplace count, is $(\varepsilon^2/2)$-zCDP (Bun & Steinke 2016, Proposition 1.4). For calibration, though, you need the reverse. Solving @eq-apx-conv-zcdp for $\rho$, a target $(\varepsilon, \delta)$ is met by any
$$\rho \le \Big(\sqrt{\varepsilon + \ln(1/\delta)} - \sqrt{\ln(1/\delta)}\Big)^2.$$ {#eq-apx-conv-zcdp-inverse}

### A worked example

You answer ten count queries, each with $\Delta = 1$ and Gaussian noise of $\sigma = 5$. Each costs $\rho = 1/(2 \times 25) = 0.02$, so the ten cost 0.2. You also release one Laplace count at $\varepsilon = 0.5$, which costs $0.5^2/2 = 0.125$. So the total is $\rho = 0.325$, and the whole release is $(\varepsilon, 10^{-5})$-DP with
$$\varepsilon = 0.325 + 2\sqrt{0.325 \times 11.5129} = 4.1937.$$
Going the other way, a single Gaussian query that must be $(1, 10^{-5})$-DP can have $\rho$ up to $(\sqrt{12.5129} - \sqrt{11.5129})^2 = 0.02082$. That means $\sigma = 1/\sqrt{2\rho} = 4.901$ for a sensitivity of 1.

### Why it works

1. **Plug the line into the RDP conversion.** By @eq-apx-conv-rdp with $\varepsilon(\alpha) = \rho\alpha$, every order gives a valid $\varepsilon$, namely $g(\alpha) = \rho\alpha + \ell/(\alpha - 1)$, with $\ell = \ln(1/\delta)$.
2. **Find where the slope is zero.** The slope is $g'(\alpha) = \rho - \ell/(\alpha - 1)^2$. It is zero when $(\alpha - 1)^2 = \ell/\rho$, that is, at $\alpha^* = 1 + \sqrt{\ell/\rho}$.
3. **Check that it is a minimum.** The term $\rho\alpha$ is a straight line, and $\ell/(\alpha - 1)$ curves upward for $\alpha > 1$. Their sum curves upward, so its only flat point is its lowest point.
4. **Evaluate.** At $\alpha^*$ the first term is $\rho + \rho\sqrt{\ell/\rho} = \rho + \sqrt{\rho\ell}$. The second is $\ell/\sqrt{\ell/\rho} = \sqrt{\rho\ell}$. Their sum is $\rho + 2\sqrt{\rho\ell}$, which is @eq-apx-conv-zcdp.
5. **The Gaussian's ρ.** Step 4 of @sec-apx-conv-rdp showed that the Gaussian's curve is exactly the line $\alpha\mu^2/2$ with $\mu = \Delta/\sigma$. So $\rho = \mu^2/2 = \Delta^2/(2\sigma^2)$.
6. **Composition.** RDP curves add at each order when mechanisms run one after another (@sec-apx-ccomp-rdp). Lines through the origin add to a line whose slope is the sum of the slopes, so the $\rho$'s add.
7. **Pure DP.** An $\varepsilon$-DP mechanism has a privacy loss between $-\varepsilon$ and $\varepsilon$. A variable confined to an interval has light tails, and Hoeffding's lemma, which bounds the MGF of such a variable by a normal one's, turns that into the line $\alpha\varepsilon^2/2$. @sec-cdp-pure proves it, along with the sharper slope $\varepsilon\tanh(\varepsilon/2)$.
8. **The inverse.** Write $x = \sqrt\rho$, so that @eq-apx-conv-zcdp reads $x^2 + 2x\sqrt\ell = \varepsilon$. Adding $\ell$ to both sides then completes the square, $(x + \sqrt\ell)^2 = \varepsilon + \ell$, and taking the positive root and squaring it gives @eq-apx-conv-zcdp-inverse.

The one-step Gaussian with $\Delta = 1$ and $\sigma = 1$ is $0.5$-zCDP, and it shows that this section and the last agree. Step 2 gives $\alpha^* = 1 + \sqrt{11.5129/0.5} = 5.7985$, and @eq-apx-conv-zcdp gives $0.5 + 2\sqrt{0.5 \times 11.5129} = 5.2985$, the same as the minimum found for the curve $\alpha/2$ in @sec-apx-conv-rdp.

### Checking it

For a Gaussian mechanism the zCDP line is exact, so the only slack is in the conversion. @tbl-apx-conv-zcdp compares @eq-apx-conv-zcdp with the improved RDP conversion on the same line, minimized over 20,001 orders, and with the exact $\varepsilon$ of the Gaussian with $\mu = \sqrt{2\rho}$. The simple formula is 13% to 27% above the exact value, more so at small $\rho$. The improved conversion removes between 49% and 68% of that gap.

| $\rho$ | $\mu = \sqrt{2\rho}$ | Simple, @eq-apx-conv-zcdp | Improved conversion | Exact Gaussian |
| --- | --- | --- | --- | --- |
| 0.125 | 0.5000 | 2.5243 | 2.1657 | 1.9931 |
| 0.2 | 0.6325 | 3.2349 | 2.8136 | 2.5944 |
| 0.5 | 1.0000 | 5.2985 | 4.7284 | 4.3772 |
| 5 | 3.1623 | 20.1743 | 19.0473 | 17.8566 |

: The $\varepsilon$ at $\delta = 10^{-5}$ of a $\rho$-zCDP mechanism, from the simple formula and the improved conversion, against the exact $\varepsilon$ of the Gaussian mechanism with the same $\rho$. {#tbl-apx-conv-zcdp}

### Recipe and pitfalls

Keep the budget in $\rho$, add it up, and convert once at the end. Do not add up the $\varepsilon$'s of the parts instead. Because $\varepsilon$ grows like the square root of $\rho$, the $\varepsilon$'s of two halves add to more than the $\varepsilon$ of the whole. Pure-DP steps enter at $\varepsilon^2/2$, which is close to the best possible slope for small $\varepsilon$ and wasteful for large ones.

The big pitfall is subsampling. Differentially private stochastic gradient descent (SGD), DP-SGD for short, runs each step on a random sample of the data, which makes each step much more private at small orders. But the sampled step's RDP curve bends sharply upward at large orders. Any line that stays above the whole curve is then as steep as the line without sampling, so zCDP loses the whole gain. Accountants for DP-SGD therefore use RDP curves or numerical methods (@sec-cdp-tcdp). @sec-cdp-conversion states the conversion with its improved version.

## Gaussian DP and its exact (ε, δ) curve {#sec-apx-conv-gdp}

### The question

For the Gaussian mechanism, both conversions above leave slack. Is there one number that describes its privacy exactly, and turns into the exact $(\varepsilon, \delta)$?

### The answer

Write $a$ for a test's false positive rate. A mechanism is **$\mu$-GDP** when no pair of neighbours is easier to tell apart than $\mathcal N(0, 1)$ from $\mathcal N(\mu, 1)$. In terms of errors, every test with false positive rate $a$ has a false negative rate of at least
$$G_\mu(a) = \Phi\big(\Phi^{-1}(1 - a) - \mu\big).$$ {#eq-apx-conv-gmu}
The Gaussian mechanism is $\mu$-GDP with $\mu = \Delta/\sigma$, with no slack. It is $(\varepsilon, \delta)$-DP exactly when $\delta$ is at least
$$\delta(\varepsilon) = \Phi\Big(-\frac{\varepsilon}{\mu} + \frac{\mu}{2}\Big) - e^\varepsilon\,\Phi\Big(-\frac{\varepsilon}{\mu} - \frac{\mu}{2}\Big).$$ {#eq-apx-conv-gdp-delta}
When several GDP mechanisms run one after another, their $\mu$'s add like the sides of a right-angled triangle, $\mu = \sqrt{\mu_1^2 + \dots + \mu_k^2}$.

### A worked example

Take $\mu = 1$, the Gaussian mechanism with $\sigma = \Delta$. The best test says "$D'$" when the output is above a threshold (step 1 below). An attacker who accepts 5% false positives uses the threshold $\Phi^{-1}(0.95) = 1.6449$, and misses with probability at least $G_1(0.05) = \Phi(1.6449 - 1) = 0.7405$. At $\varepsilon = 1$, @eq-apx-conv-gdp-delta gives $\delta(1) = \Phi(-0.5) - e\,\Phi(-1.5) = 0.308538 - e \times 0.066807 = 0.126937$. To report $\varepsilon$ at $\delta = 10^{-5}$, solve @eq-apx-conv-gdp-delta for $\varepsilon$ by bisection, which repeatedly halves an interval that contains the answer. This works because $\delta(\varepsilon)$ falls as $\varepsilon$ grows: a larger $\varepsilon$ asks less of every test. @tbl-apx-conv-gdp-eps gives the result for three values of $\mu$.

| $\mu$ | 0.5 | 1 | 2 |
| --- | --- | --- | --- |
| $\delta(1)$ | 0.0068 | 0.1269 | 0.5099 |
| $\varepsilon$ at $\delta = 10^{-5}$ | 1.9931 | 4.3772 | 9.9973 |

: The exact $\delta$ at $\varepsilon = 1$ and the exact $\varepsilon$ at $\delta = 10^{-5}$ of $\mu$-GDP, from @eq-apx-conv-gdp-delta. {#tbl-apx-conv-gdp-eps}

Composition is just as quick. Four releases with $\sigma = 2\Delta$ each have $\mu_i = 0.5$, so together they have $\mu = \sqrt{4 \times 0.25} = 1$. They are 1-GDP, and so $(4.3772, 10^{-5})$-DP.

### Why it works

The first three steps find the best test, which gives the curve $G_\mu$. The next three find the $(\varepsilon, \delta)$ guarantee that the curve implies, and the last one handles composition.

1. **The best test thresholds the output.** The Neyman–Pearson lemma says that, among tests with the same false positive rate, the one with the fewest misses says "$D'$" on the outputs where $q(y)/p(y)$ is largest. The reason is that each output in the region costs $p(y)$ in false positives and buys $q(y)$ in detections, so the best buys go first. For the pair $\mathcal N(0, 1)$ and $\mathcal N(\mu, 1)$, the ratio is $q(y)/p(y) = e^{-L(y)} = e^{\mu y - \mu^2/2}$, which grows with $y$. So the best test says "$D'$" when $y$ is above a threshold $t$.
2. **Its two errors.** Under $D$, $y$ is standard normal, so $a = \Pr[Z > t] = 1 - \Phi(t)$, and the threshold for a given $a$ is $t = \Phi^{-1}(1 - a)$. Under $D'$, $y = \mu + Z$, so the test misses when $\mu + Z \le t$, which has probability $\Phi(t - \mu)$.
3. **Substitute.** Putting $t = \Phi^{-1}(1 - a)$ into $\Phi(t - \mu)$ gives @eq-apx-conv-gmu. For the Gaussian mechanism, a pair of neighbours whose query values differ by $s \le \Delta$ gives the pair $\mathcal N(0, 1)$ and $\mathcal N(s/\sigma, 1)$, with shift $s/\sigma \le \Delta/\sigma$. A larger shift is easier to detect, because $G_\mu$ falls as $\mu$ grows, so the worst pair has $\mu = \Delta/\sigma$.
4. **One (ε, δ) guarantee is one line.** By @sec-apx-conv-testing, $(\varepsilon, \delta)$-DP asks every test to have $\mathrm{FNR} \ge 1 - \delta - e^\varepsilon a$. The best tests have $\mathrm{FNR} = G_\mu(a)$, so the condition is $\delta \ge 1 - G_\mu(a) - e^\varepsilon a$ for every $a$, and the smallest valid $\delta$ is the largest value of the right side. The other inequality of @eq-apx-conv-testing gives the same number. Reflecting the output, $y \mapsto \mu - y$, swaps the two normals, so it turns each test into one with the two error rates swapped, and the two inequalities into each other.
5. **Find the touching point.** Lowering the threshold by a small step $h$ adds the outputs just below $t$ to the region. That raises $a$ by about $p(t)h$ and lowers the miss rate by about $q(t)h$, so the curve's slope at that point is $-q(t)/p(t) = -e^{\mu t - \mu^2/2}$. As $a$ grows the threshold drops, so this slope gets flatter. The quantity $1 - G_\mu(a) - e^\varepsilon a$ therefore rises while the miss rate falls faster than $e^\varepsilon$ per unit of $a$, and falls afterwards. So it is largest where the slope equals $-e^\varepsilon$. Setting $\mu t - \mu^2/2 = \varepsilon$ gives $t = \varepsilon/\mu + \mu/2$.
6. **Read off δ.** At that threshold, $a = 1 - \Phi(t) = \Phi(-t)$ and the detection rate is $1 - \Phi(t - \mu) = \Phi(\mu - t)$. So $\delta(\varepsilon) = \Phi(\mu - t) - e^\varepsilon\Phi(-t)$, which is @eq-apx-conv-gdp-delta.
7. **Composition.** Each release adds independent noise, so $k$ releases give a vector of $k$ outputs, $\mathcal N(0, I)$ under $D$ and $\mathcal N(m, I)$ under $D'$ with $m = (\mu_1, \dots, \mu_k)$. Only the direction of $m$ carries evidence, and along it the shift is the length of $m$, $\sqrt{\mu_1^2 + \dots + \mu_k^2}$. @sec-apx-ccomp-gaussian proves this rotation step.

At $\mu = 1$ and $\varepsilon = 1$, step 5 gives $t = 1 + 0.5 = 1.5$. Step 6 then gives $a = \Phi(-1.5) = 0.066807$ and a detection rate of $\Phi(-0.5) = 0.308538$, which are the numbers of the worked example. @sec-apx-cal-gauss-exact reaches the same formula by a different route. There, $\Phi(-0.5)$ is the chance that the privacy loss exceeds $\varepsilon$, and $e\,\Phi(-1.5)$ is the correction subtracted from it. With $\mu = \Delta/\sigma$ the two derivations give the same curve.

### Checking it

Searching over 20,001 thresholds for the largest $1 - G_1(a) - e\,a$ gives 0.126937, the value of @eq-apx-conv-gdp-delta. A simulation of the threshold test at $a = 0.05$ with 1,000,000 draws per world gives a false positive rate of 0.0497 and a false negative rate of 0.7413, against the exact 0.7405.

@fig-apx-conv-gdp shows both halves of the derivation. On the left, each curve $G_\mu$ has a dashed line of slope $-e$ that just touches it. The line starts at $1 - \delta(1)$ on the vertical axis, so the gap below 1 is the exact $\delta$ at $\varepsilon = 1$. On the right, the whole curve $\delta(\varepsilon)$ falls steeply, and the dots mark where it crosses $10^{-5}$.

![Left: the trade-off curves $G_\mu$ for $\mu = 0.5$, 1 and 2, the smallest false negative rate of any test at each false positive rate. Each dashed line has slope $-e$ and touches its curve at the dot; it meets the vertical axis at $1 - \delta(1)$. Right: the exact $\delta(\varepsilon)$ of $\mu$-GDP on a log scale, with the $\varepsilon$ at $\delta = 10^{-5}$ marked.](figures/conv_gdp.svg){#fig-apx-conv-gdp}

### Recipe and pitfalls

For a pipeline of Gaussian releases, compute $\mu_i = \Delta_i/\sigma_i$ for each, combine them as $\mu = \sqrt{\sum\mu_i^2}$, and solve @eq-apx-conv-gdp-delta for $\varepsilon$ at your $\delta$. Nothing is lost along the way, so this $\varepsilon$ is exact.

The formula is exact only for Gaussian noise with the $\ell_2$ sensitivity $\Delta$, the largest Euclidean distance between the query's values on neighbours. For subsampled DP-SGD, GDP comes from a central limit theorem and is only an approximation (@sec-gdp-sgd). And one reported $(\varepsilon, \delta)$ pair keeps one touching line and forgets the rest of the curve. @sec-gdp-duality shows that the whole family of lines rebuilds $G_\mu$.

## Summary

@tbl-apx-conv-summary puts the four definitions side by side. Each one tracks a different summary of the privacy loss, and each converts to $(\varepsilon, \delta)$ at the end.

| Definition | What you track | How it composes | How to get $(\varepsilon, \delta)$ |
| --- | --- | --- | --- |
| $(\varepsilon, \delta)$-DP | Two numbers | Add the $\varepsilon$'s and the $\delta$'s (basic composition) | Nothing to do |
| RDP | A curve $\varepsilon(\alpha)$ over orders | Add the curves at each order | Smallest $\varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1)$ over orders, or the improved formula |
| $\rho$-zCDP | One number $\rho$ | Add the $\rho$'s | $\rho + 2\sqrt{\rho\ln(1/\delta)}$ |
| $\mu$-GDP | One number $\mu$ | Add the $\mu^2$'s | Solve @eq-apx-conv-gdp-delta for $\varepsilon$ |

: The four ways to track privacy, how each composes and how each converts to $(\varepsilon, \delta)$. {#tbl-apx-conv-summary}

For the Gaussian mechanism with $\mu = 1$ at $\delta = 10^{-5}$, the three conversions give 5.2985 (Mironov's, which equals the zCDP formula here), 4.7284 (the improved conversion) and 4.3772 (GDP, exact).

## Code {#sec-apx-conv-code}

The script `code/appendix-derivations/apx_core_conversions.py` checks every number on this page. It enumerates every test against three people's randomized response, simulates the posterior and the threshold test, and checks the Gaussian's Rényi divergence by Monte Carlo. It also minimizes both RDP conversions over orders, inverts the exact Gaussian $\delta(\varepsilon)$ by bisection, and computes $\Phi^{-1}$ by bisection on $\Phi$, which it evaluates with the standard library's complementary error function. Run it with

```bash
uv run python code/appendix-derivations/apx_core_conversions.py
```

| Check | Printed | Expected |
| --- | --- | --- |
| Least slack over all 256 tests, $\varepsilon = \ln 3$ | 0.000000 | at least 0 |
| Posterior from 2,000,000 randomized-response draws, prior 1%, $\varepsilon = 1$ | 0.0267 | 0.0267 |
| Rényi divergence of order 3 by Monte Carlo, $\mu = 1$ | 1.4965 | 1.5 |
| Three outputs: largest $P(S) - e^{0.1}Q(S)$ | 0.00989 | at most 0.01 |
| $\Pr[L > 5.2985]$ for the Gaussian, $\mu = 1$ | $7.992\times10^{-7}$ | at most $10^{-5}$ |
| $\delta(1)$ for $\mu = 1$ by search over thresholds | 0.126937 | 0.126937 |
| Threshold test at $a = 0.05$, $\mu = 1$, simulated miss rate | 0.7413 | 0.7405 |

: The checks the script prints, with seed 0. {#tbl-apx-conv-code}

Every bound holds, and the ones that should be tight are tight. Randomized response reaches both the testing bound and the posterior cap. The Gaussian's privacy loss is normal, which is why its RDP curve, its zCDP parameter and its GDP curve are all exact descriptions. So for the RDP curve and the zCDP parameter, all the slack comes from the final conversion to $(\varepsilon, \delta)$.

## References

- Dwork, C. & Roth, A. *The Algorithmic Foundations of Differential Privacy* (2014). The definitions, the privacy loss and the basic properties.
- Wasserman, L. & Zhou, S. "A statistical framework for differential privacy" (2010), and Kairouz, P., Oh, S. & Viswanath, P. "The composition theorem for differential privacy" (2015). The reading of $(\varepsilon, \delta)$-DP as a limit on every test.
- Kasiviswanathan, S. P. & Smith, A. "On the 'semantics' of differential privacy: a Bayesian formulation" (2014). The posterior reading, including weaker attackers.
- Mironov, I. "Rényi differential privacy" (2017). RDP and the conversion of @eq-apx-conv-rdp.
- Balle, B., Barthe, G., Gaboardi, M., Hsu, J. & Sato, T. "Hypothesis testing interpretations and Rényi differential privacy" (2020), and Canonne, C., Kamath, G. & Steinke, T. "The discrete Gaussian for differential privacy" (2020). The improved conversion.
- Bun, M. & Steinke, T. "Concentrated differential privacy: simplifications, extensions, and lower bounds" (2016). zCDP, its conversion and the bound for pure DP.
- Dong, J., Roth, A. & Su, W. "Gaussian differential privacy" (2022). $f$-DP, GDP, the exact $(\varepsilon, \delta)$ curve and composition.
- Balle, B. & Wang, Y.-X. "Improving the Gaussian mechanism for differential privacy: analytical calibration and optimal denoising" (2018). The exact $\delta$ of the Gaussian mechanism as a calibration rule.
