# Hypothesis testing

## Why this matters

A spam filter sees an email and must say "ham" or "spam". A fraud detector sees a transaction and must say "normal" or "fraud". A medical test reads a biomarker and must say "healthy" or "ill". A monitoring job sees a batch of sensor readings and must say "calibrated" or "drifted". Each one is the same problem: **an observation came from one of two distributions, and you must decide which.** Hypothesis testing is the mathematics of that decision. It starts by naming the two kinds of mistake a decision rule can make and measuring how often it makes each. It then asks what the best possible rule is and how good it can be. Finally, it asks how to summarize and compare real rules, and how to estimate their error rates from a finite amount of data.

The whole chapter rests on one idea. Write $p(x)$ for the probability of observing $x$ under the first distribution and $q(x)$ for its probability under the second. The ratio $q(x)/p(x)$, called the **likelihood ratio**, says how many times more likely the observation is under the second distribution than under the first, and **it is the only thing about the observation that matters.** The Neyman–Pearson lemma, proved below, says that the best rule compares this ratio with a threshold. Moving the threshold trades one kind of mistake for the other, and the best rules at all thresholds together trace a curve that is fixed by the two distributions alone. Every one-number summary of a rule, from the best advantage to the area under the curve (AUC) to the probability that an alarm is right, is read off that curve. @tbl-ht-uses lists the results and where each is used. Later parts ask how well *any* test can distinguish two distributions; the trade-off curve built here is the answer.

| Result | Typical use |
| --- | --- |
| Type I/II errors, power | Reading any detector's confusion matrix |
| Neyman–Pearson lemma | Designing the optimal detector; proving that nothing beats it |
| Likelihood-ratio test | Score thresholds, sequential and multi-sample tests |
| Trade-off function $T(P,Q)$ | The whole error trade-off between two distributions, as one curve |
| Gaussian and Laplace trade-off curves | Closed-form reference curves for shifted noise |
| Best advantage $=$ total variation | A one-number summary of distinguishability |
| Receiver operating characteristic (ROC) curve, AUC, true positive rate at a low false positive rate | Comparing classifiers and detectors honestly |
| Bayesian view | Turning error rates into posterior probabilities at a given base rate |
| Confidence intervals for rates | Reporting measured error rates with their uncertainty |

: The results in this chapter and where they are used. {#tbl-ht-uses}

The distributions used here are in [Probability distributions](01-distributions.md), the tail bounds and confidence intervals in [Concentration inequalities](02-concentration.md), and total variation and Kullback–Leibler (KL) divergence in [Divergences](03-divergences.md). Each is restated below where it is used.

## Toolkit: four facts every derivation uses

The derivations below reuse four facts. Each is stated and proved once here, with a small running example.

**The running example.** A spam filter counts suspicious words in an email, capped at 3. The default assumption is that an email is ham; this is the **null hypothesis** $H_0$, under which the count $X$ has distribution $P$. The **alternative** $H_1$ is that the email is spam, under which $X$ has distribution $Q$. @tbl-ht-spam gives both, and their likelihood ratio $\Lambda(x) = q(x)/p(x)$, with the alternative on top so that large values point to $H_1$. ([Divergences](03-divergences.md) writes the ratio the other way up, as $r(x) = p(x)/q(x)$, and its logarithm as $L(x)$; here $\Lambda$ is always the plain ratio $q/p$.) Spam tends to have more suspicious words, so the ratio grows with the count: a count of 3 is four times as likely in spam as in ham, a count of 0 a quarter as likely.

| $x$ | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| $p(x)$, ham | 0.4 | 0.3 | 0.2 | 0.1 |
| $q(x)$, spam | 0.1 | 0.2 | 0.3 | 0.4 |
| $\Lambda(x) = q(x)/p(x)$ | 0.250 | 0.667 | 1.500 | 4.000 |

: Word-count distributions under ham ($P$) and spam ($Q$), and their likelihood ratio. {#tbl-ht-spam}

### Fact 1 — A test is a function, and its errors are expectations

The simplest decision rule picks a set $R$ of outcomes, its **rejection region**, and says "spam" (rejects $H_0$) exactly when $x$ lands in $R$. It is useful to allow a little more: on some outcomes the rule may say "spam" only with some probability, by flipping a biased coin. So a **test** is a function $\phi: \mathcal X \to [0, 1]$ on the set $\mathcal X$ of possible outcomes, where $\phi(x)$ is the probability of rejecting $H_0$ after observing $x$. A **deterministic test** takes only the values 0 and 1, so it is the indicator of its rejection region: $\phi(x) = 1$ on $R$ and 0 off it. A **randomized test** may take values strictly between 0 and 1.

A test makes two kinds of mistake, and both are averages of $\phi$. The first is to reject when the email is ham. Given $X = x$, the test rejects with probability $\phi(x)$, so averaging over $X$ drawn from $P$ gives the probability of rejecting under $H_0$. The second is to keep $H_0$ when the email is spam: averaging $\phi$ over $X$ drawn from $Q$ gives the probability of rejecting under $H_1$, and one minus that is the probability of failing to. Writing $\mathbb E_P$ and $\mathbb E_Q$ for averages under $P$ and $Q$, the two error probabilities are
$$\alpha_\phi = \mathbb E_P[\phi(X)], \qquad \beta_\phi = 1 - \mathbb E_Q[\phi(X)].$$
For a deterministic test this is the familiar fact that the probability of an event is the average of its indicator, $\Pr[X \in R] = \mathbb E\big[\mathbf 1[X \in R]\big]$: the indicator is 1 on the event and 0 off it, so its average is the probability mass of the event ([Concentration inequalities](02-concentration.md) builds Markov's inequality on the same fact).

For example, the test "reject when $X \ge 2$" has $\phi = (0, 0, 1, 1)$ on the outcomes $0, 1, 2, 3$. Its first error is the ham mass on $\{2, 3\}$, $\alpha_\phi = 0.2 + 0.1 = 0.3$. It rejects spam with probability $\mathbb E_Q\phi = 0.3 + 0.4 = 0.7$, so its second error is $\beta_\phi = 0.3$.

### Fact 2 — Change of measure

The likelihood ratio converts averages under $P$ into averages under $Q$. The reason is simple: wherever $p(x) > 0$, the spam probability $q(x)$ equals the ham probability $p(x)$ multiplied by $\Lambda(x) = q(x)/p(x)$. So any average under $Q$ can be computed as an average under $P$ in which each outcome is reweighted by $\Lambda$. The one catch is outcomes that $P$ never produces, where $p(x) = 0$ and the ratio is undefined; those have to be counted separately. For any function $g \ge 0$,
$$\mathbb E_Q[g(X)] = \mathbb E_P[g(X)\Lambda(X)] + \mathbb E_Q\big[g(X)\mathbf 1[p(X) = 0]\big].$$
To prove it, split the sum (or integral) $\sum_x g(x)q(x)$ into the points with $p(x) > 0$, where $q(x) = \Lambda(x)p(x)$ turns the sum into $\sum_x g(x)\Lambda(x)p(x) = \mathbb E_P[g\Lambda]$, and the points with $p(x) = 0$, which give the second term.

In the spam example $p > 0$ everywhere, so only the first term appears. For "reject when $X = 3$", $\phi = (0, 0, 0, 1)$ and $\mathbb E_P[\phi \Lambda] = 0.1 \times 4 = 0.4$, which is exactly $\mathbb E_Q\phi$, the spam mass at 3. One consequence is used later: taking $g = 1$, the left side is 1 and the second term is $\Pr_Q[p = 0]$, the $Q$-probability of the outcomes $P$ never produces, so $\mathbb E_P[\Lambda] = 1 - \Pr_Q[p = 0] = \Pr_Q[p > 0] \le 1$.

### Fact 3 — Mixing tests mixes their errors

Suppose you flip a coin that lands heads with probability $\lambda \in [0, 1]$, and run test $\phi_1$ on heads and $\phi_2$ on tails. On outcome $x$ this rejects with probability $\lambda\phi_1(x) + (1 - \lambda)\phi_2(x)$, so it is the test $\phi = \lambda\phi_1 + (1 - \lambda)\phi_2$, which takes values in $[0, 1]$. Its errors are the same mixture of the two tests' errors,
$$\alpha_\phi = \lambda \alpha_{\phi_1} + (1 - \lambda)\alpha_{\phi_2}, \qquad \beta_\phi = \lambda \beta_{\phi_1} + (1 - \lambda)\beta_{\phi_2},$$
because by Fact 1 both errors are averages of $\phi$, and the average of a weighted sum is the weighted sum of the averages. Geometrically, if you plot each test as a point $(\alpha_\phi, \beta_\phi)$, every point on the segment between two achievable points is achievable too. So the set of achievable error pairs is convex, and randomization fills in the segments between deterministic tests.

### Fact 4 — The inverse normal distribution function

Write $\Phi$ for the standard normal cumulative distribution function (CDF): $\Phi(x)$ is the probability that a standard normal variable is at most $x$. Write $\Phi^{-1}$ for its inverse, so $\Phi^{-1}(u)$ is the point below which a standard normal has probability $u$. The normal density is symmetric about 0, so the area above $x$ equals the area below $-x$: $1 - \Phi(x) = \Phi(-x)$. Applying $\Phi^{-1}$ to both sides of this identity, with $x = \Phi^{-1}(\alpha)$, gives $\Phi^{-1}(1 - \alpha) = -\Phi^{-1}(\alpha)$. The values used below are $\Phi^{-1}(0.9) = 1.2816$, $\Phi^{-1}(0.95) = 1.6449$, $\Phi^{-1}(0.975) = 1.9600$, $\Phi^{-1}(0.99) = 2.3263$ and $\Phi^{-1}(0.999) = 3.0902$.

$\Phi$ has no closed-form inverse, so these values are computed numerically. The script solves $1 - \Phi(x) = \alpha$ by bisection, evaluating the upper tail as $\frac12\operatorname{erfc}(x/\sqrt2)$ with the complementary error function $\operatorname{erfc}$. Solving for the tail directly, rather than computing $\Phi^{-1}$ at $1 - \alpha$, keeps full precision when $\alpha$ is tiny, because $1 - \alpha$ then rounds to a number very close to 1 and loses the digits that matter.

## Simple-versus-simple testing {#sec-ht-setup}

### Context

The simplest testing problem has exactly two candidate distributions, each fully specified: **simple** hypotheses, as opposed to composite ones such as "the mean is positive", which allow a whole family of distributions. Most practical detection problems reduce to it once you fix what "normal" and "abnormal" data look like. This section fixes the vocabulary.

### Definitions

You observe $X$ and test whether it came from $P$ or from $Q$:
$$H_0: X \sim P \qquad\text{against}\qquad H_1: X \sim Q.$$

A test can go wrong in two ways, and each has a name. A **type I error**, or false positive, is rejecting $H_0$ when it is true: flagging a ham email as spam. Its probability is $\alpha_\phi$ from Fact 1, also called the **size** of the test or its **false positive rate (FPR)**. A **type II error**, or false negative, is keeping $H_0$ when $H_1$ is true: letting spam through. Its probability is $\beta_\phi$, the **false-negative rate**. (These are the letters of Neyman and Pearson; a page where $\alpha$ or $\beta$ already means something else writes the two errors as $\mathrm{FPR}$ and $\mathrm{FNR}$.)

The complement of each error is a success rate. The **power** $1 - \beta_\phi$ is the probability of detecting $H_1$ when it holds; in classification it is called the **true positive rate (TPR)**, **recall** or **sensitivity**. The complement of the first error, $1 - \alpha_\phi$, is the **specificity**, the probability of correctly passing a ham email. Finally, a test has **level** $\alpha$ if its false positive rate is at most $\alpha$, $\alpha_\phi \le \alpha$: the level is a budget, the size is what the test actually spends.

The convention is asymmetric on purpose. $H_0$ is the default ("this email is fine", "this patient is healthy"), and a false alarm is the error you control first. You fix the level, then look for the most powerful test at that level.

### The two errors pull in opposite directions

For each rejection region in the spam example, Fact 1 gives the errors directly: $\alpha$ is the ham mass $p$ summed over the region, and the power is the spam mass $q$ summed over it. For the region $\{2, 3\}$, $\alpha = 0.2 + 0.1 = 0.3$ and the power is $0.3 + 0.4 = 0.7$, so $\beta = 0.3$. @tbl-ht-spam-tests lists the threshold tests "reject when $X \ge k$" and two regions that are not thresholds.

| Rejection region | $\alpha$ | $\beta$ | Power |
| --- | --- | --- | --- |
| none | 0.00 | 1.00 | 0.00 |
| $\{3\}$ | 0.10 | 0.60 | 0.40 |
| $\{2, 3\}$ | 0.30 | 0.30 | 0.70 |
| $\{1, 2, 3\}$ | 0.60 | 0.10 | 0.90 |
| all | 1.00 | 0.00 | 1.00 |
| $\{0\}$ | 0.40 | 0.90 | 0.10 |
| $\{0, 3\}$ | 0.50 | 0.50 | 0.50 |

: Errors of deterministic tests in the spam example. The first five are thresholds on the count; the last two are not. {#tbl-ht-spam-tests}

Going down the first five rows, the rejection region grows: each added outcome adds its ham mass to $\alpha$ and its spam mass to the power, so $\alpha$ rises and $\beta$ falls. You cannot lower both errors by changing the region; you can only trade one for the other.

The last two rows are bad regions, and a useful yardstick shows why. A test that ignores the email and rejects with probability $\alpha$ by flipping a coin has, by Fact 1, size $\alpha$ and power $\alpha$, so $\beta = 1 - \alpha$. Any test worth using must beat this. The region $\{0\}$ does not: at $\alpha = 0.4$ it has $\beta = 0.9$, worse than the coin flip's $1 - 0.4 = 0.6$, because it rejects on the outcome most typical of ham. The region $\{0, 3\}$ does exactly as well as the coin flip, $\beta = 0.5 = 1 - 0.5$.

### Why randomize

The threshold tests reach only $\alpha \in \{0, 0.1, 0.3, 0.6, 1\}$. If your spam policy allows a false-positive rate of 0.25, no deterministic threshold test uses that budget: $\{3\}$ spends only 0.1, and $\{2, 3\}$ overspends at 0.3. A randomized test spends it exactly: reject on $\{3\}$ always and on $\{2\}$ with probability $0.75$. This is the same as flipping a coin that picks the region $\{3\}$ with probability 0.25 and the region $\{2, 3\}$ with probability 0.75 (count 3 is then rejected either way, and count 2 with probability 0.75). So it is the mixture $0.25 \cdot \mathbf 1[x \in \{3\}] + 0.75 \cdot \mathbf 1[x \in \{2, 3\}]$, and by Fact 3 its size is $\alpha = 0.25 \times 0.1 + 0.75 \times 0.3 = 0.25$ exactly. In practice, randomization matters for discrete data, where the threshold can only jump from one outcome to the next. For continuous data such as scores, moving a threshold changes $\alpha$ smoothly, so a threshold can hit any $\alpha$ on its own.

@fig-ht-errors shows the two errors for a continuous problem: a biomarker that is $\mathcal N(0, 1)$ in healthy patients and $\mathcal N(2, 1)$ in ill ones, with the test "reject when $x > \tau$". The false positive rate $\alpha$ is the area under the healthy curve to the right of $\tau$, and the miss rate $\beta$ is the area under the ill curve to the left of $\tau$. Moving $\tau$ right shrinks the blue area $\alpha$ and grows the orange area $\beta$.

![The two errors of the threshold test "reject when $x > \tau$" for $P = \mathcal N(0, 1)$ against $Q = \mathcal N(2, 1)$, with $\tau$ set so that $\alpha = 0.05$. The blue area under $P$ beyond $\tau$ is $\alpha$; the orange area under $Q$ before $\tau$ is $\beta = 0.361$.](figures/hypothesis_testing_errors.svg){#fig-ht-errors}

### Example: a medical screening test

A screening test flags a patient when the biomarker exceeds $\tau = 1.645$, the threshold drawn in @fig-ht-errors. A healthy patient's biomarker is standard normal, so it exceeds $\tau$ with probability $1 - \Phi(1.645)$, and by Fact 4 that is 0.05 because $\Phi^{-1}(0.95) = 1.6449$. So one healthy patient in 20 is sent for a follow-up. An ill patient's biomarker is centred at 2, so it falls below $\tau$ with probability $\beta = \Phi(\tau - 2) = \Phi(-0.355) = 0.361$. The test's sensitivity is therefore $1 - \beta = 0.6388$: about two ill patients in three are caught. Lowering the threshold catches more ill patients and alarms more healthy ones. Which point to pick is a policy decision. What the mathematics can say is which points are *achievable*, and that is the subject of @sec-ht-np.

### Check yourself

- A test rejects with probability $\alpha$ regardless of the data. What are its errors? *($\alpha_\phi = \alpha$ and $\beta_\phi = 1 - \alpha$. This "coin flip" test is the baseline every useful test must beat.)*
- Why is the test that rejects on $\{0\}$ in @tbl-ht-spam-tests worse than the coin flip at the same $\alpha$? *(It rejects where $\Lambda(x) = 0.25 < 1$: on that region ham is four times as likely as spam, so it spends false-positive budget where spam is rare.)*

## The Neyman–Pearson lemma {#sec-ht-np}

### Context

Among all tests at level $\alpha$, which one is the most powerful? The space of tests is huge: every function $\phi: \mathcal X \to [0, 1]$. The Neyman–Pearson lemma says that you only need to look at one family, thresholds of the likelihood ratio, and that the best test in that family beats every other test.

The intuition is a budget. Each outcome $x$ you decide to reject costs $p(x)$ of false-positive budget and buys $q(x)$ of power. The exchange rate at $x$ is $q(x)/p(x) = \Lambda(x)$, units of power per unit of budget. If you want the most power for a fixed budget, you should spend it on the outcomes with the best exchange rate first, and that means rejecting exactly where $\Lambda(x)$ is above some cut-off.

### Statement

Let $P$ and $Q$ have densities (or probability mass functions) $p$ and $q$. A **likelihood ratio (LR) test** with threshold $\tau \ge 0$ rejects wherever $q(x)$ is more than $\tau$ times $p(x)$, keeps $H_0$ wherever it is less, and on the tie set, where $q(x) = \tau\,p(x)$ exactly, may reject with any probability $\gamma(x) \in [0, 1]$:
$$\phi^*(x) = \begin{cases} 1 & \text{if } q(x) > \tau\,p(x), \\ \gamma(x) & \text{if } q(x) = \tau\,p(x), \\ 0 & \text{if } q(x) < \tau\,p(x). \end{cases}$$ {#eq-ht-lr-test}
Writing the condition as $q > \tau p$ rather than $\Lambda > \tau$ also covers points where $p(x) = 0$: there any $q(x) > 0$ counts as above every threshold. The lemma has two parts.

1. **Optimality.** Every test $\phi$ with $\alpha_\phi \le \alpha_{\phi^*}$ has $\beta_\phi \ge \beta_{\phi^*}$: no test that spends at most as much false-positive budget misses less.
2. **Existence.** For every $\alpha \in (0, 1]$ there is an LR test with a constant $\gamma$ and size exactly $\alpha$.

Together they say that the most powerful level-$\alpha$ test is the LR test of size $\alpha$. (For $\alpha = 0$ the best test is $\mathbf 1[p(x) = 0]$: with no budget at all, you can only reject where $H_0$ is impossible.)

### Derivation of optimality

The plan is to compare $\phi^*$ with any other test $\phi$ outcome by outcome, weighting each outcome by $q(x) - \tau\,p(x)$, which is positive exactly where the LR test rejects. A pointwise comparison then adds up to a comparison of the errors.

1. **A pointwise inequality.** For every $x$,
$$\big(\phi^*(x) - \phi(x)\big)\big(q(x) - \tau\,p(x)\big) \ge 0.$$
To see it, check the three cases of @eq-ht-lr-test. If $q > \tau p$, then $\phi^* = 1 \ge \phi$, so both factors are $\ge 0$. If $q < \tau p$, then $\phi^* = 0 \le \phi$, so both are $\le 0$. If $q = \tau p$, the second factor is 0. This is the only place the form of $\phi^*$ is used, and it does not care what $\gamma$ is.
2. **Sum (or integrate) over $x$.** A sum of non-negative terms is non-negative. Expanding the product, $\sum_x\phi(x)q(x) = \mathbb E_Q[\phi]$ and $\sum_x\phi(x)p(x) = \mathbb E_P[\phi]$ by Fact 1, and the same for $\phi^*$, so
$$\mathbb E_Q[\phi^*] - \mathbb E_Q[\phi] \ge \tau\big(\mathbb E_P[\phi^*] - \mathbb E_P[\phi]\big).$$
3. **Use the level constraint.** The other test spends no more budget: $\mathbb E_P[\phi] = \alpha_\phi \le \alpha_{\phi^*} = \mathbb E_P[\phi^*]$. Together with $\tau \ge 0$, the right side is $\ge 0$.
4. **Read off the errors.** So $\mathbb E_Q[\phi^*] \ge \mathbb E_Q[\phi]$. Both sides are powers, $1 - \beta_{\phi^*} \ge 1 - \beta_\phi$, which is $\beta_\phi \ge \beta_{\phi^*}$.

Step 2 is the budget intuition of the Context written out. $\tau$ is the exchange rate at the cut-off: every unit of false-positive probability spent at $x$ buys $\Lambda(x)$ units of power. The LR test spends its budget only where the exchange rate is at least $\tau$, so no other allocation of the same budget buys more power.

### Derivation of existence

The idea is to lower the cut-off on $\Lambda$ until the outcomes strictly above it use at most $\alpha$ of the budget, while including the outcomes at the cut-off would use at least $\alpha$. Then randomizing on the outcomes exactly at the cut-off fills the gap. To make this precise, let $X \sim P$ and let $G(c) = \Pr_P[\Lambda(X) > c]$ for $c \ge 0$ be the ham probability of the outcomes whose ratio exceeds $c$, which is the size of the test that rejects strictly above $c$. (Points with $p(x) = 0$ have probability 0 under $P$, so $\Lambda(X)$ is finite.)

1. **How $G$ behaves.** Raising $c$ removes outcomes from $\{\Lambda > c\}$, so $G$ is non-increasing. It is right-continuous, because as $c' \downarrow c$ the sets $\{\Lambda > c'\}$ grow to $\{\Lambda > c\}$, and the probability of a growing sequence of sets tends to the probability of their union. In the same way, as $c' \uparrow c$ the sets $\{\Lambda > c'\}$ shrink to $\{\Lambda \ge c\}$, so the left limit is $G(c^-) = \Pr_P[\Lambda(X) \ge c]$.
2. **Choose the cut-off.** Set $\tau = \inf\{c \ge 0 : G(c) \le \alpha\}$, the smallest cut-off whose strict rejection region fits in the budget. Right-continuity gives $G(\tau) \le \alpha$. Just left of $\tau$, $G > \alpha$, so $\Pr_P[\Lambda \ge \tau] \ge \alpha$ (when $\tau = 0$ this reads $1 \ge \alpha$).
3. **Randomize on the ties.** The LR test with constant $\gamma$ rejects $\{\Lambda > \tau\}$ always and $\{\Lambda = \tau\}$ with probability $\gamma$, so its size is $G(\tau) + \gamma\Pr_P[\Lambda = \tau]$. If $\Pr_P[\Lambda = \tau] > 0$, pick $\gamma = \big(\alpha - G(\tau)\big)/\Pr_P[\Lambda = \tau]$ to make the size exactly $\alpha$. It lies in $[0, 1]$: it is $\ge 0$ because $G(\tau) \le \alpha$, and $\le 1$ because $G(\tau) + \Pr_P[\Lambda = \tau] = \Pr_P[\Lambda \ge \tau] \ge \alpha$, both by step 2. If $\Pr_P[\Lambda = \tau] = 0$, step 2 gives $G(\tau) = \alpha$ and any $\gamma$ works.

**Concrete: the spam example at $\alpha = 0.25$.** The distinct ratios, from the largest, are 4 (mass 0.1 under $P$), 1.5 (mass 0.2), 0.667 and 0.25. For any cut-off $c$ with $1.5 \le c < 4$, only the count 3 has a ratio above $c$, so $G(c) = 0.1$, which fits in the budget. Just below 1.5 the count 2 joins, and $G = 0.1 + 0.2 = 0.3$, which does not. So $\tau = 1.5$. The test rejects the count 3 always and must spend the remaining $0.25 - 0.1 = 0.15$ on the tie set $\{2\}$, whose ham mass is 0.2: $\gamma = (0.25 - 0.1)/0.2 = 0.75$. This is the randomized test of @sec-ht-setup. Its power is the spam mass it rejects, $0.4 + 0.75 \times 0.3 = 0.625$, so $\beta = 0.375$.

The script checks the lemma by brute force. It draws 200,000 random tests, $\phi(x)$ uniform on $[0, 1]$ for each $x$; 11,727 of them have size at most 0.25, and the most powerful of those has power 0.587, below 0.625. @fig-ht-discrete-tradeoff plots 400 random tests: every one lies on or above the curve traced by the LR tests.

![Errors $(\alpha, \beta)$ of 400 random tests on the spam example (grey), against the curve of likelihood-ratio tests. Black dots are the deterministic thresholds, the red square is the randomized test of size 0.25, and the dotted line is the coin-flip test.](figures/hypothesis_testing_discrete_tradeoff.svg){#fig-ht-discrete-tradeoff}

### Interpretation

**Only the likelihood ratio matters.** The optimal test looks at $x$ only through $\Lambda(x)$, so two observations with the same likelihood ratio are interchangeable for testing. Everything else about $x$ is noise.

**Monotone ratios give thresholds on $x$.** Often $\Lambda(x)$ increases with some simple statistic $s(x)$, such as the word count in the spam example. Then "the ratio is above its threshold" is the same event as "the statistic is above some threshold $\tau$", $\{s > \tau\}$, and the optimal test is a plain threshold on $s$. The word count, the Gaussian biomarker and the Laplace shift below all work this way. This is why thresholding a well-calibrated score is the right thing to do.

**The slope of the curve is $-\tau$.** Plot the LR tests as points $(\alpha, \beta)$. Moving along a segment of this curve means rejecting one more outcome $x$ with a little more probability. That spends $p(x)$ of false-positive budget per unit of extra rejection probability and gains $q(x)$ of power, so $\beta$ falls by $\Lambda(x)$ for each unit $\alpha$ rises: the slope of the segment is $-\Lambda(x)$. In @fig-ht-discrete-tradeoff the segments have slopes $-4$, $-1.5$, $-0.667$ and $-0.25$: the likelihood ratios, largest first. Because the LR test adds outcomes in decreasing order of exchange rate, each segment is less steep than the last, and a curve whose slope keeps increasing is convex.

**Many observations.** For $n$ independent observations the joint probability of the sample is the product of the individual probabilities, under $P$ and under $Q$. So the likelihood ratio of the whole sample is $\prod_i \Lambda(x_i)$, and the LR test thresholds its logarithm $\sum_i \ln \Lambda(x_i)$, a sum of independent terms. Sums of independent terms concentrate, so the errors decay exponentially in $n$. With the false positive rate held fixed, the type II error decays like $e^{-n\,D_{\mathrm{KL}}(P\|Q)}$, where the KL divergence $D_{\mathrm{KL}}(P\|Q) = \mathbb E_P[\ln(p(X)/q(X))]$ is the average log-ratio by which a ham observation favours ham over spam (the Chernoff–Stein lemma; Cover & Thomas, ch. 11, @sec-apx-tail-cramer-stein in the appendices, and [Exercise 8](../../book-exercises/part01-prerequisites/04-hypothesis-testing.md#sec-sol-ht-8)).

**What the lemma needs.** Both hypotheses must be simple. A composite alternative such as "the mean is positive" contains many distributions, each with its own LR test, and one test is optimal against all of them at once only in special cases. The main one is a one-sided test in a family with a monotone likelihood ratio, where every member's LR test is the same threshold on one statistic (Karlin–Rubin; Casella & Berger, §8.3).

### Example: why a classifier's score should be a likelihood ratio

A fraud model outputs a score $s(x)$. Suppose the score is calibrated, $s(x) = \Pr[\text{fraud} \mid x]$, with a base rate $\pi$ of fraud. Bayes' rule says that the odds of fraud after seeing $x$ are the odds before seeing it, multiplied by the likelihood ratio (@sec-ht-bayes derives this): $s/(1 - s) = \frac{\pi}{1 - \pi}\Lambda(x)$. The odds $s/(1 - s)$ grow with $s$, so $s$ is an increasing function of $\Lambda(x)$. Thresholding $s$ is then an LR test, and Neyman–Pearson says no other use of $x$ can catch more fraud at the same false-alarm rate. If the score is not monotone in $\Lambda$, re-ranking the transactions by an estimate of $\Lambda$ can only help.

### Check yourself

- Where did the proof use that $\phi$ takes values in $[0, 1]$? *(In step 1: when $q > \tau p$ it needs $\phi \le 1 = \phi^*$, and when $q < \tau p$ it needs $\phi \ge 0 = \phi^*$.)*
- In the spam example, what is the most powerful test at level $\alpha = 0.3$, and does it need randomization? *(Reject $\{2, 3\}$, with power 0.70. No: 0.3 is exactly the $P$-mass of the outcomes with $\Lambda \ge 1.5$.)*

## Trade-off functions and ROC curves {#sec-ht-tradeoff}

### Context

A single test gives one point $(\alpha, \beta)$. To compare two *problems* (how hard is it to tell $P$ from $Q$?) you want the best $\beta$ at every $\alpha$ at once. That is a curve, and it depends only on $P$ and $Q$, not on any particular test.

### Definition

For each false positive budget $\alpha$, look at every test that stays within it and record the smallest miss rate any of them achieves. As a function of $\alpha$, that is the **trade-off function** of $P$ and $Q$:
$$T(P, Q)(\alpha) = \inf\{\beta_\phi : \alpha_\phi \le \alpha\}, \qquad \alpha \in [0, 1].$$ {#eq-ht-tradeoff}
The infimum ($\inf$) is the largest number that is at most every $\beta_\phi$ in the set; it is the minimum when the minimum is attained. By the Neyman–Pearson lemma it is attained, by the LR test of size $\alpha$, so in practice you compute $T$ by sweeping the LR threshold. Write $T$ for $T(P, Q)$ when the pair is clear.

The **ROC curve** of the optimal test plots its TPR against its FPR, that is, power against size: $\alpha \mapsto 1 - T(\alpha)$. It holds the same information as $T$, turned upside down. A particular *score*, thresholded at every possible value, also traces an ROC curve, and by Neyman–Pearson that curve lies on or below the optimal one.

### Statement: four properties

Every trade-off function is

1. **non-increasing,**
2. **at most $1 - \alpha$:** $T(\alpha) \le 1 - \alpha$,
3. **convex,**
4. **continuous** on $[0, 1]$.

In words: a bigger budget never hurts, the coin flip is always available, mixing tests fills in segments, and there are no jumps.

Conversely, every function $f: [0, 1] \to [0, 1]$ with these four properties is $T(P, Q)$ for some pair $P, Q$, so these four properties are exactly what "a possible error trade-off" means. The standard construction takes $P$ uniform on $[0, 1]$, so that any test rejecting an interval of length $\alpha$ has size $\alpha$, and designs $Q$ so that rejecting the top interval $(1 - \alpha, 1]$ misses exactly $f(\alpha)$. Concretely, let $Q$ have CDF $G(x) = f(1 - x)$ on $[0, 1]$ (so $G(1) = f(0)$, and $G(0) = f(1) = 0$ since property 2 forces $f(1) \le 0$), and put its remaining mass $1 - f(0)$ at a point outside $[0, 1]$, where $P$ has no mass. The slope of $G$ at $x$ is minus the slope of $f$ at $1 - x$. Because $f$ is convex its slope increases with its argument, so as $x$ grows and $1 - x$ shrinks, the slope of $G$, which is the density of $Q$ on $[0, 1]$, is non-decreasing. The likelihood ratio is that density on $[0, 1]$ (since $p = 1$ there) and infinite at the outside point, so it is non-decreasing, and the tests "reject the outside point and every $x > 1 - \alpha$" are LR tests. Each has size $\alpha$, and its type II error is the $Q$-mass it does not reject, $\Pr_Q[0 \le X \le 1 - \alpha] = G(1 - \alpha) = f(\alpha)$.

### Derivation

1. **Non-increasing.** If $\alpha \le \alpha'$, every test with $\alpha_\phi \le \alpha$ also has $\alpha_\phi \le \alpha'$. The infimum in @eq-ht-tradeoff is over a larger set at $\alpha'$, so it is no larger.
2. **At most $1 - \alpha$.** The coin-flip test $\phi \equiv \alpha$ is always within budget, with $\alpha_\phi = \alpha$ and $\beta_\phi = 1 - \alpha$ (Fact 1), so the infimum is at most $1 - \alpha$.
3. **Convex.** Convexity says the curve at a mixed budget is no higher than the same mixture of the curve's values, and Fact 3 provides exactly that mixture. Fix $\alpha_1, \alpha_2$, $\lambda \in [0, 1]$ and a small $h > 0$. Pick tests $\phi_i$ with $\alpha_{\phi_i} \le \alpha_i$ and $\beta_{\phi_i} \le T(\alpha_i) + h$, which exist by the definition of the infimum. By Fact 3, the mixture $\lambda\phi_1 + (1 - \lambda)\phi_2$ has size at most $\lambda \alpha_1 + (1 - \lambda)\alpha_2$ and type II error at most $\lambda T(\alpha_1) + (1 - \lambda)T(\alpha_2) + h$. So $T(\lambda \alpha_1 + (1 - \lambda)\alpha_2) \le \lambda T(\alpha_1) + (1 - \lambda)T(\alpha_2) + h$. Let $h \to 0$.
4. **Continuous.** Inside $(0, 1)$ this comes free from convexity: a convex function on an interval is continuous at every interior point, because its chords pin it between two lines on either side of the point (any analysis text, e.g. Rockafellar, *Convex Analysis*, §10). At $\alpha = 1$, the test $\phi \equiv 1$ has $\beta = 0$, so $T(1) = 0$, and property 2 squeezes $0 \le T(\alpha) \le 1 - \alpha \to 0$.

    At $\alpha = 0$ convexity alone allows a jump, so the argument needs Fact 2. First, $T(0) = 1 - \Pr_Q[p = 0]$: a test of size 0 must have $\phi = 0$ wherever $p > 0$, so the most it can do is reject everywhere $p = 0$. Now take any test with $\alpha_\phi \le \alpha$ and any cut-off $K > 0$, and bound its power. By Fact 2 the power is $\mathbb E_Q[\phi] = \mathbb E_P[\phi \Lambda] + \mathbb E_Q[\phi\,\mathbf 1[p = 0]]$, and the second term is at most $\Pr_Q[p = 0]$. Split $\mathbb E_P[\phi \Lambda]$ by whether $\Lambda \le K$: where $\Lambda \le K$ it is at most $K\,\mathbb E_P[\phi] \le K\alpha$, and where $\Lambda > K$ it is at most $\mathbb E_P[\Lambda\,\mathbf 1[\Lambda > K]]$ because $\phi \le 1$. So
    $$1 - \beta_\phi = \mathbb E_Q[\phi] \le \Pr_Q[p = 0] + \mathbb E_P[\phi \Lambda] \le \Pr_Q[p = 0] + K\alpha + \mathbb E_P\big[\Lambda\,\mathbf 1[\Lambda > K]\big].$$
    Hence $T(\alpha) \ge T(0) - K\alpha - \mathbb E_P[\Lambda\,\mathbf 1[\Lambda > K]]$. Since $\mathbb E_P[\Lambda] \le 1$ is finite (Fact 2), the part of that average coming from $\Lambda > K$ tends to 0 as $K \to \infty$. So choose $K$ large to make the last term small, then $\alpha$ small to make $K\alpha$ small: $T(\alpha) \to T(0)$.

### Interpretation

The two extremes of the curve are the two extremes of difficulty. $T(\alpha) = 1 - \alpha$ for all $\alpha$ means no test beats a coin flip, which happens exactly when $P = Q$. $T \equiv 0$ means some test makes no errors at all, which happens when $P$ and $Q$ live on disjoint sets.

Between the extremes, the further $T$ bends below the diagonal, the easier the problem. A pair $(P, Q)$ is **harder to distinguish** than $(P', Q')$ if $T(P, Q) \ge T(P', Q')$ everywhere: at every budget, the best test misses more. This is only a partial order, because two curves can cross, and then neither problem is uniformly harder.

The curve need not start at 1. If $Q$ puts mass where $P$ has none, a test catches that mass with no false alarms, so $T(0) = 1 - \Pr_Q[p = 0]$ is below 1 ([Exercise 6](../../book-exercises/part01-prerequisites/04-hypothesis-testing.md#sec-sol-ht-6)). And the curve depends on which distribution is the null: swapping the roles of $P$ and $Q$ reflects the curve across the diagonal, so $T(Q, P)$ is the inverse function of $T(P, Q)$ when $T(P, Q)$ is strictly decreasing ([Exercise 2](../../book-exercises/part01-prerequisites/04-hypothesis-testing.md#sec-sol-ht-2)).

### Check yourself

- Why can a trade-off function never be concave and below the diagonal on part of $[0, 1]$ and above it elsewhere? *(Property 2 forbids it from ever being above the diagonal $1 - \alpha$.)*
- A deterministic score gives a staircase ROC on a finite test set. Why can you always join adjacent points with straight segments? *(Fact 3: randomizing between the two thresholds achieves every point on the segment.)*

## The Gaussian trade-off curve {#sec-ht-gaussian}

### Context

The most common pair in practice is a shift of a normal distribution: a score, a biomarker or a sample mean that is normal under both hypotheses with the same spread but different centres. Its trade-off curve has a closed form, worked out here in full.

### Statement

For a unit-variance normal under $H_0$ and the same normal shifted right by $\mu$ under $H_1$, the best test is a threshold on $x$. At budget $\alpha$ the threshold sits where $P$ has upper tail $\alpha$, which is $\Phi^{-1}(1 - \alpha)$, and the miss rate is the probability that $Q$, centred $\mu$ higher, falls below it. So for $P = \mathcal N(0, 1)$ and $Q = \mathcal N(\mu, 1)$ with $\mu \ge 0$,
$$T(P, Q)(\alpha) = \Phi\big(\Phi^{-1}(1 - \alpha) - \mu\big).$$ {#eq-ht-gaussian-tradeoff}
More generally, $\mathcal N(m_0, \sigma^2)$ against $\mathcal N(m_1, \sigma^2)$ has this curve with $\mu = |m_1 - m_0|/\sigma$. The reason is that standardizing the observation, $x \mapsto (x - m_0)/\sigma$, is a one-to-one relabelling of the outcomes, so every test on the original scale is a test on the new scale with the same two errors, and on the new scale the pair is $\mathcal N(0, 1)$ against $\mathcal N(\pm\mu, 1)$.

### Derivation

The plan is to compute the likelihood ratio, see that it increases with $x$, and then read both errors of the threshold test off normal tails.

1. **The likelihood ratio.** With the standard normal density $\varphi(x) = e^{-x^2/2}/\sqrt{2\pi}$, the density of $Q$ is $\varphi(x - \mu)$. The constants cancel in the ratio, and expanding the square, $x^2 - (x - \mu)^2 = 2\mu x - \mu^2$:
$$\Lambda(x) = \frac{\varphi(x - \mu)}{\varphi(x)} = \exp\Big(\frac{x^2 - (x - \mu)^2}{2}\Big) = \exp\Big(\mu x - \frac{\mu^2}{2}\Big).$$
2. **The LR test is a threshold on $x$.** For $\mu > 0$, $\Lambda$ is strictly increasing in $x$, so the likelihood ratio exceeds a cut-off $c$ exactly when $x > \tau$, where $\tau = (\ln c + \mu^2/2)/\mu$ solves $\mu \tau - \mu^2/2 = \ln c$. Ties $\Lambda(x) = c$ happen only at the single point $x = \tau$, which has probability 0, so no randomization is needed. By Neyman–Pearson, the tests $\phi_\tau(x) = \mathbf 1[x > \tau]$ are optimal.
3. **Size fixes the threshold.** $\alpha = \Pr_P[X > \tau] = 1 - \Phi(\tau)$, so $\tau = \Phi^{-1}(1 - \alpha)$.
4. **Type II error.** Under $Q$, $X - \mu \sim \mathcal N(0, 1)$, so the test misses with probability $\beta = \Pr_Q[X \le \tau] = \Pr[X - \mu \le \tau - \mu] = \Phi(\tau - \mu)$.
5. **Substitute** step 3 into step 4 to get @eq-ht-gaussian-tradeoff.

**Checks.** At $\mu = 0$ the two hypotheses coincide, and indeed $\Phi(\Phi^{-1}(1 - \alpha)) = 1 - \alpha$, the diagonal. As $\mu \to \infty$, $T \to 0$ for every $\alpha > 0$. And $T$ is symmetric about the diagonal, $T(T(\alpha)) = \alpha$ ([Exercise 1](../../book-exercises/part01-prerequisites/04-hypothesis-testing.md#sec-sol-ht-1)). In short: Fact 4 turns $\Phi^{-1}(1 - T(\alpha))$ into $-\Phi^{-1}(T(\alpha)) = -\big(\Phi^{-1}(1 - \alpha) - \mu\big)$, so $T(T(\alpha)) = \Phi\big(-\Phi^{-1}(1 - \alpha)\big) = 1 - (1 - \alpha) = \alpha$.

### Interpretation

**One number describes the problem.** The curve depends on the two normals only through $\mu$, the distance between the means in units of the standard deviation. @fig-ht-gaussian-tradeoff shows the family. Crosses mark simulated thresholds at $\mu = 1$, with 20,000 draws per point.

**Near $\alpha = 0$ the curve is vertical.** The slope of the curve at the point with threshold $\tau$ is $-\Lambda(\tau)$, as in the slope rule of @sec-ht-np. Here it follows from the two errors: moving the threshold from $\tau$ to $\tau + \mathrm d\tau$ lowers $\alpha$ by $p(\tau)\,\mathrm d\tau$ and raises $\beta$ by $q(\tau)\,\mathrm d\tau$, so $T'(\alpha) = d\beta/d\alpha = -q(\tau)/p(\tau) = -\Lambda(\tau)$. Small $\alpha$ means a large threshold, and $\Lambda(\tau) = e^{\mu \tau - \mu^2/2} \to \infty$ as $\tau \to \infty$. So the first sliver of false-positive budget buys power at an unbounded exchange rate. The Laplace pair of @sec-ht-laplace behaves differently.

**Averaging $n$ observations.** Take $n$ independent draws from $\mathcal N(0, \sigma^2)$ against $\mathcal N(h, \sigma^2)$. Step 1, with the spread $\sigma$ put back, gives each draw a log likelihood ratio $(h x_i - h^2/2)/\sigma^2$, and the log ratios of independent draws add up to $\sum_i(h x_i - h^2/2)/\sigma^2$. This depends on the data only through $\sum_i x_i$, that is, through the sample mean $\bar x$, so the best test thresholds $\bar x$. The sample mean is $\mathcal N(0, \sigma^2/n)$ against $\mathcal N(h, \sigma^2/n)$, a shift of $h$ with standard deviation $\sigma/\sqrt n$. So the curve is @eq-ht-gaussian-tradeoff with
$$\mu = \frac{h\sqrt n}{\sigma}.$$
Distinguishability grows like $\sqrt n$, the same rate at which a sample mean concentrates.

![The Gaussian trade-off curve $\Phi(\Phi^{-1}(1 - \alpha) - \mu)$ for $\mu = 0, 0.5, 1, 2, 3$. Crosses are simulated errors of threshold tests at $\mu = 1$.](figures/hypothesis_testing_gaussian_tradeoff.svg){#fig-ht-gaussian-tradeoff}

### Example: detecting sensor drift

A monitoring job averages $n = 100$ readings from a sensor with noise $\sigma$ and must detect a drift of $h = 0.2\sigma$ in the mean. Then $\mu = h\sqrt n/\sigma = 0.2 \times 10 = 2$. @tbl-ht-gaussian evaluates @eq-ht-gaussian-tradeoff at this shift and at $\mu = 1$, for two levels. Each row sets the threshold from the level by Fact 4 and subtracts $\mu$; for example, at $\mu = 2$ and $\alpha = 0.05$, $\beta = \Phi(1.6449 - 2) = \Phi(-0.3551) = 0.3612$.

| $\mu$ | $\alpha$ | Threshold $\Phi^{-1}(1 - \alpha)$ | $\beta = T(\alpha)$ | Power |
| --- | --- | --- | --- | --- |
| 1 | 0.05 | 1.6449 | 0.7405 | 0.2595 |
| 1 | $10^{-3}$ | 3.0902 | 0.9817 | 0.0183 |
| 2 | 0.05 | 1.6449 | 0.3612 | 0.6388 |
| 2 | $10^{-3}$ | 3.0902 | 0.8622 | 0.1378 |

: The Gaussian trade-off curve at two shifts and two levels. {#tbl-ht-gaussian}

At a 5% false-alarm rate the job catches the drift with probability 0.6388 (@tbl-ht-gaussian). A simulation with 1,000,000 draws per hypothesis gives $\hat\alpha = 0.0503$ and $\hat\beta = 0.3617$. If false alarms must be rare, $\alpha = 10^{-3}$, the threshold moves out to 3.0902 and the power drops to $1 - \Phi(3.0902 - 2) = 0.1378$: a stricter threshold costs a lot of power when $\mu$ is moderate.

**How many readings?** Turn the question around: what shift $\mu$, and so what $n$, gives power at least $1 - \beta'$ at level $\alpha$? The power target is $T(\alpha) \le \beta'$, that is, $\Phi(\Phi^{-1}(1 - \alpha) - \mu) \le \beta'$. Applying the increasing function $\Phi^{-1}$ to both sides gives $\Phi^{-1}(1 - \alpha) - \mu \le \Phi^{-1}(\beta')$, and by Fact 4, $\Phi^{-1}(\beta') = -\Phi^{-1}(1 - \beta')$. So the condition is $\mu \ge \Phi^{-1}(1 - \alpha) + \Phi^{-1}(1 - \beta')$. Substituting $\mu = h\sqrt n/\sigma$ and solving for $n$,
$$n \ge \left(\frac{\Phi^{-1}(1 - \alpha) + \Phi^{-1}(1 - \beta')}{h/\sigma}\right)^2.$$
For power 0.9 at $\alpha = 0.01$ and $h/\sigma = 0.25$, the two quantiles are $\Phi^{-1}(0.99) = 2.3263$ and $\Phi^{-1}(0.9) = 1.2816$, so $n \ge ((2.3263 + 1.2816)/0.25)^2 = 208.3$, and you need $n = 209$ readings.

### Check yourself

- Two classifiers produce normal scores with equal spread. Classifier A separates the class means by 2 standard deviations, classifier B by 1. How many independent B-scores must you average to match one A-score? *(Four: $\mu$ grows like $\sqrt n$, and $1 \times \sqrt 4 = 2$.)*
- Why does step 2 need $\mu > 0$, and what changes for $\mu < 0$? *(For $\mu < 0$ the ratio is decreasing in $x$, so the optimal test rejects for small $x$. The curve is the same with $|\mu|$.)*

## The Laplace trade-off curve {#sec-ht-laplace}

### Context

Replace the normal by a Laplace distribution: the same shift problem, but with exponential tails. The Laplace distribution $\mathrm{Lap}(m, b)$ with centre $m$ and scale $b$ has density $\frac1{2b}e^{-|x - m|/b}$. The trade-off curve changes shape at the ends, and the reason is the likelihood ratio. Shifting a Laplace density changes its exponent $-|x - m|/b$ by at most $m_1/b$ for a shift of $m_1$ (moving the centre by $m_1$ changes the distance $|x - m|$ by at most $m_1$), so the shifted density is within a factor $e^{m_1/b}$ of the original everywhere ([Probability distributions](01-distributions.md)). A shifted normal is not: its likelihood ratio, computed in @sec-ht-gaussian, grows without bound.

### Statement

Because the likelihood ratio is capped, no test can buy power faster than the cap allows, and the curve's two ends are straight lines with the cap as their slope. In between, where the ratio does vary, the curve bends. For $P = \mathrm{Lap}(0, b)$ and $Q = \mathrm{Lap}(m_1, b)$ with $m_1 > 0$, write $\kappa = m_1/b$ for the shift in units of the scale. Then
$$T(P, Q)(\alpha) = \begin{cases} 1 - e^{\kappa}\alpha, & \alpha < \tfrac12 e^{-\kappa}, \\[2pt] \dfrac{e^{-\kappa}}{4\alpha}, & \tfrac12 e^{-\kappa} \le \alpha \le \tfrac12, \\[6pt] e^{-\kappa}(1 - \alpha), & \alpha > \tfrac12. \end{cases}$$ {#eq-ht-laplace-tradeoff}
The three pieces correspond to a threshold above the spam-side centre $m_1$, between the two centres, and below the ham-side centre 0.

### Derivation

Rescale $x$ by $b$, which changes no test's errors (as for the normal), so that $b = 1$ and $m_1 = \kappa$. The CDF of $\mathrm{Lap}(0, 1)$ is $F(x) = \frac12 e^x$ for $x < 0$ and $1 - \frac12 e^{-x}$ for $x \ge 0$, obtained by integrating the density $\frac12 e^{-|x|}$ on each side of 0. The plan is the same as for the normal: find the likelihood ratio, see that thresholds on $x$ are LR tests, and read the errors off $F$, now in three cases.

1. **The likelihood ratio.** The normalizing constants cancel, so $\ln \Lambda(x) = -|x - \kappa| + |x| = |x| - |x - \kappa|$. Removing the absolute values on each side of 0 and $\kappa$ gives
$$\ln \Lambda(x) = \begin{cases} -\kappa, & x \le 0, \\ 2x - \kappa, & 0 < x < \kappa, \\ \kappa, & x \ge \kappa. \end{cases}$$
It is non-decreasing in $x$ and clipped to $[-\kappa, \kappa]$ (@fig-ht-laplace-tradeoff, left).
2. **Thresholds on $x$ are LR tests.** Because $\Lambda$ is non-decreasing, the region $\{x > \tau\}$ always consists of every point with $\Lambda > c$, for the cut-off $c = \Lambda(\tau)$, plus part of the tie set $\{\Lambda = c\}$. When $\tau$ falls in a flat piece, the tie set is a whole half-line with positive probability, and rejecting only the part of it with $x > \tau$ is one allowed choice of $\gamma(x)$ in @eq-ht-lr-test. So every threshold test is optimal, and its errors $\alpha = 1 - F(\tau)$ and $\beta = \Pr_Q[X \le \tau] = F(\tau - \kappa)$ trace $T$.
3. **Case $\tau \ge \kappa$.** Both $\tau$ and $\tau - \kappa$ are $\ge 0$. Then $\alpha = \frac12 e^{-\tau} \le \frac12 e^{-\kappa}$, and $\beta = 1 - \frac12 e^{-(\tau - \kappa)} = 1 - e^{\kappa}\cdot\frac12 e^{-\tau} = 1 - e^{\kappa}\alpha$.
4. **Case $0 \le \tau < \kappa$.** Now $\tau \ge 0$ but $\tau - \kappa < 0$. Then $\alpha = \frac12 e^{-\tau} \in (\frac12 e^{-\kappa}, \frac12]$, and $\beta = \frac12 e^{\tau - \kappa}$. Substituting $e^\tau = 1/(2\alpha)$ from the first equation: $\beta = \frac12 e^{-\kappa}/(2\alpha) = e^{-\kappa}/(4\alpha)$.
5. **Case $\tau < 0$.** Both are negative. Then $\alpha = 1 - \frac12 e^\tau > \frac12$, so $\frac12 e^\tau = 1 - \alpha$, and $\beta = \frac12 e^{\tau - \kappa} = e^{-\kappa}(1 - \alpha)$.

**Checks.** The pieces agree at the joins: at $\alpha = \frac12 e^{-\kappa}$ both give $\frac12$, and at $\alpha = \frac12$ both give $\frac12 e^{-\kappa}$. At $\kappa = 0$ all three pieces are $1 - \alpha$, the diagonal of two identical distributions.

### Interpretation

**The ends are straight lines.** Their slopes are $-e^{\kappa}$ and $-e^{-\kappa}$. That is the slope rule of @sec-ht-np at work: the slope of $T$ is $-\Lambda$ at the threshold, and on the two flat pieces of the likelihood ratio $\Lambda$ is the constant $e^{\kappa}$ or $e^{-\kappa}$. Because $\Lambda \le e^{\kappa}$ everywhere, no test can buy power faster than $e^{\kappa}$ per unit of false-positive rate. Reading the first piece as power: for small $\alpha$, the best power is exactly $1 - T(\alpha) = e^{\kappa}\alpha$.

**Gaussian against Laplace.** To compare the two shapes fairly, match the two problems by their best advantage, the largest gap $\mathrm{TPR} - \mathrm{FPR}$ any test achieves. @sec-ht-tv shows that it equals $1 - e^{-\kappa/2}$ for the Laplace pair and $2\Phi(\mu/2) - 1$ for the Gaussian pair. At $\kappa = 1$ the Laplace value is $1 - e^{-1/2} = 0.3935$, and solving $2\Phi(\mu/2) - 1 = 0.3935$ gives a Gaussian pair with $\mu = 1.0301$. In the bulk the two curves nearly coincide. At small $\alpha$ they do not, as @tbl-ht-laplace-vs-gaussian shows: the Laplace power is $e\alpha$, and the Gaussian power is $1 - \Phi(\Phi^{-1}(1 - \alpha) - 1.0301)$.

| $\alpha$ | Laplace power, $\kappa = 1$ | Gaussian power, $\mu = 1.0301$ |
| --- | --- | --- |
| $10^{-3}$ | $2.718\times10^{-3}$ | $1.969\times10^{-2}$ |
| $10^{-6}$ | $2.718\times10^{-6}$ | $9.829\times10^{-5}$ |

: Best power at small false-positive rates for two pairs with the same total variation. {#tbl-ht-laplace-vs-gaussian}

At $\alpha = 10^{-6}$ the Gaussian test is 36 times more powerful, $9.829\times10^{-5}/2.718\times10^{-6} = 36.2$ (@tbl-ht-laplace-vs-gaussian). Its unbounded likelihood ratio lets a test find rare observations that are overwhelmingly more likely under $Q$; the clipped Laplace ratio forbids that. @fig-ht-laplace-tradeoff plots both on log axes.

![Left: the log likelihood ratio of $\mathrm{Lap}(1, 1)$ against $\mathrm{Lap}(0, 1)$ is clipped at $\pm1$, while the Gaussian one with the same total variation is a straight line. Right: best power $1 - T(\alpha)$ on log axes. The Laplace curve follows $e\alpha$ for small $\alpha$; the Gaussian curve rises far above it.](figures/hypothesis_testing_laplace_tradeoff.svg){#fig-ht-laplace-tradeoff}

### Example: absolute-error residuals

A regression model with absolute-error loss has residuals that are roughly Laplace with scale $b$. You suspect a batch of inputs carries a systematic offset of $m_1 = b$, so $\kappa = 1$. The first breakpoint of @eq-ht-laplace-tradeoff is $\frac12 e^{-1} = 0.1839$ and the second is $\frac12$, so each level falls on a different piece. At $\alpha = 0.05$, below the first breakpoint, the first piece gives $\beta = 1 - e \times 0.05 = 0.8641$, power 0.1359. At $\alpha = 0.25$, between the breakpoints, the middle piece gives $\beta = e^{-1}/(4 \times 0.25) = e^{-1} = 0.3679$, power 0.6321. At $\alpha = 0.75$, above $\frac12$, the last piece gives $\beta = e^{-1} \times 0.25 = 0.0920$.

A simulation of the threshold test at $\alpha = 0.25$ with 1,000,000 draws per hypothesis gives $\hat\alpha = 0.2498$ and $\hat\beta = 0.3685$.

### Check yourself

- On which piece of @eq-ht-laplace-tradeoff does the threshold $\tau$ lie between the two medians $0$ and $m_1$? *(The middle piece, $\frac12 e^{-\kappa} \le \alpha \le \frac12$, where $\Lambda$ is strictly increasing.)*
- Why can a test on the Laplace pair at level $\alpha = 10^{-3}$ never have power above $e^{\kappa} \times 10^{-3}$? *(Fact 2: power $= \mathbb E_P[\phi \Lambda] \le e^{\kappa}\,\mathbb E_P[\phi] \le e^{\kappa}\alpha$, since $\Lambda \le e^{\kappa}$ everywhere.)*

## Best advantage equals total variation {#sec-ht-tv}

### Context

Sometimes you want one number for "how distinguishable are $P$ and $Q$?". A natural one is the **advantage** of a test, $1 - \alpha_\phi - \beta_\phi = \mathrm{TPR} - \mathrm{FPR}$: how much more often it fires under $Q$ than under $P$. A test that ignores the data, such as the coin flip, fires equally often under both and has advantage 0. (In medicine, $\mathrm{TPR} - \mathrm{FPR}$ at a threshold is Youden's index.) Its maximum over all tests turns out to be the total variation (TV) distance of [Divergences](03-divergences.md). That distance is half the total absolute difference between the two probability tables, $\frac12\sum_x|q(x) - p(x)|$, or equivalently the largest difference $\Pr_Q[A] - \Pr_P[A]$ in the probability the two distributions give to any one event $A$.

### Statement

The advantage of a test is a sum over outcomes of $\phi(x)$ times $q(x) - p(x)$, so the best test fires exactly on the outcomes where $q$ exceeds $p$, and its advantage is the total excess of $q$ over $p$. That excess is the total variation distance:
$$\max_\phi\,(1 - \alpha_\phi - \beta_\phi) = \mathrm{TV}(P, Q) = \sup_A\,\big(\Pr_Q[A] - \Pr_P[A]\big) = \frac12\sum_x |q(x) - p(x)|,$$ {#eq-ht-tv}
with the sum replaced by an integral for densities. The maximum is attained by the LR test with $\tau = 1$: reject where $q > p$.

### Derivation

1. **Advantage as one sum.** By Fact 1, the power is $\mathbb E_Q\phi$ and the false positive rate is $\mathbb E_P\phi$, so $1 - \alpha_\phi - \beta_\phi = \mathbb E_Q\phi - \mathbb E_P\phi = \sum_x \phi(x)\big(q(x) - p(x)\big)$.
2. **Maximize term by term.** Each $\phi(x)$ appears in one term only, so each can be chosen separately. A term is largest with $\phi(x) = 1$ where $q(x) > p(x)$ and $\phi(x) = 0$ where $q(x) < p(x)$. So the maximum is $\sum_{x: q > p}(q(x) - p(x)) = \Pr_Q[A^*] - \Pr_P[A^*]$ with $A^* = \{q > p\}$. A deterministic $\phi(x) = \mathbf 1[x \in A]$ gives $\Pr_Q[A] - \Pr_P[A]$, and the best of these is $A^*$, so the maximum is also $\sup_A (\Pr_Q[A] - \Pr_P[A])$.
3. **Rewrite as half the $L^1$ distance.** Both tables sum to 1, so $\sum_x(q - p) = 1 - 1 = 0$, and the positive and negative parts of $q - p$ are equal: $\sum_{q > p}(q - p) = \sum_{q < p}(p - q)$. Their sum is $\sum_x|q - p|$, so each is half of it.

On the trade-off curve, the advantage at level $\alpha$ is $1 - \alpha - T(\alpha)$: the vertical gap between the diagonal and the curve. Its derivative is $-1 - T'(\alpha)$, which is zero where the slope of $T$ is $-1$. By the slope rule of @sec-ht-np the slope is $-\Lambda$ at the threshold, so this is the point where the LR threshold is $\tau = 1$ (@fig-ht-tv-advantage).

![The Gaussian trade-off curve at $\mu = 1$, the diagonal $1 - \alpha$, and the advantage $1 - \alpha - T(\alpha)$ (dashed). The red segment is the largest gap, at $\alpha = 0.3085$, and its length is the total variation 0.3829.](figures/hypothesis_testing_tv_advantage.svg){#fig-ht-tv-advantage}

### Example: three pairs

**Spam words.** The differences $|q - p|$ in @tbl-ht-spam are 0.3, 0.1, 0.1 and 0.3, so $\mathrm{TV} = \frac12(0.3 + 0.1 + 0.1 + 0.3) = 0.40$. Spam is more likely than ham at the counts 2 and 3, so the best test rejects $\{2, 3\}$, and indeed its advantage is $1 - 0.3 - 0.3 = 0.4$ (@tbl-ht-spam-tests).

**Gaussian.** The best test rejects where $q > p$, that is, where $\Lambda(x) = e^{\mu x - \mu^2/2} > 1$, which is $x > \mu/2$: the point halfway between the two means, where the two bell curves cross. That test has $\alpha = 1 - \Phi(\mu/2)$ and $\beta = \Phi(\mu/2 - \mu) = \Phi(-\mu/2) = 1 - \Phi(\mu/2)$, so $\mathrm{TV} = 1 - 2(1 - \Phi(\mu/2)) = 2\Phi(\mu/2) - 1$. At $\mu = 1$ this is $2\Phi(0.5) - 1 = 0.3829$, reached at $\alpha = 1 - \Phi(0.5) = 0.3085$. Maximizing $1 - \alpha - T(\alpha)$ over a grid of 100,001 values of $\alpha$ gives the same 0.3829 at $\alpha = 0.3085$.

**Laplace.** Here it is easiest to maximize the gap on the curve directly. On the middle piece of @eq-ht-laplace-tradeoff the gap is $1 - \alpha - \frac{e^{-\kappa}}{4\alpha}$, and setting its derivative to zero, $-1 + \frac{e^{-\kappa}}{4\alpha^2} = 0$, gives $\alpha = \frac12 e^{-\kappa/2}$. There $\frac{e^{-\kappa}}{4\alpha} = \frac12 e^{-\kappa/2}$ as well, so $\mathrm{TV} = 1 - e^{-\kappa/2}$. At $\kappa = 1$ that is 0.3935, at $\alpha = 0.3033$; the grid search agrees.

### Interpretation

TV is the best that a summary using a single threshold, and weighting the two errors equally, can say. It measures the problem at one point of the curve, where $\tau = 1$, and says nothing about the ends. @tbl-ht-laplace-vs-gaussian is the warning: two pairs with the same TV, 0.3935, differ by a factor of 36 in power at $\alpha = 10^{-6}$. If the regime you care about is "very few false alarms", TV is the wrong summary, and the whole curve is the right one.

### Check yourself

- Why is the advantage of any test at most 1, and when is it exactly 1? *(It is at most $\Pr_Q[A^*] - \Pr_P[A^*] \le 1$, with equality only if $\Pr_Q[A^*] = 1$ and $\Pr_P[A^*] = 0$: the distributions have disjoint supports.)*
- Is the coin-flip test's advantage ever negative? Is any test's? *(The coin flip has advantage 0. A test can have negative advantage, like rejecting $\{0\}$ in the spam example, but then its complement $1 - \phi$ has positive advantage.)*

## ROC summaries: why AUC hides the low-FPR regime {#sec-ht-auc}

### Context

Real detectors are compared on held-out data by their ROC curves, the plot of TPR against FPR as the detector's threshold sweeps from strict to lenient. The curve is usually summarized by the **AUC**, the area under it. In many applications only a small corner of the curve matters: a fraud team can review a tiny fraction of transactions, a screening programme can afford few false alarms, a spam filter must almost never discard real mail. This section shows that two detectors with the same AUC can differ by an order of magnitude in that corner.

### Statement: AUC is a ranking probability

The area under the ROC curve has a direct meaning: it is the probability that the detector scores a randomly chosen positive above a randomly chosen negative. For a score $S$ that is $S_- \sim P$ on negatives and $S_+ \sim Q$ on positives (continuous and independent),
$$\mathrm{AUC} = \int_0^1 \mathrm{TPR}\,d(\mathrm{FPR}) = \Pr[S_+ > S_-].$$

### Derivation

The idea is to parametrize the curve by the threshold and recognize the area integral as an average over a negative's score.

1. **Parametrize the curve by the threshold.** The test "positive when $S > \tau$" has $\mathrm{FPR}(\tau) = \Pr[S_- > \tau]$ and $\mathrm{TPR}(\tau) = \Pr[S_+ > \tau]$. As $\tau$ goes from $+\infty$ to $-\infty$, the FPR goes from 0 to 1.
2. **Change variables.** $\mathrm{FPR}(\tau) = \int_\tau^\infty p_-(s)\,\mathrm ds$, where $p_-$ is the density of $S_-$, so $\mathrm d\,\mathrm{FPR}(\tau) = -p_-(\tau)\,\mathrm d\tau$. Substituting, and flipping the limits to absorb the minus sign,
$$\mathrm{AUC} = \int_{+\infty}^{-\infty}\Pr[S_+ > \tau]\,\big(-p_-(\tau)\big)\,\mathrm d\tau = \int_{-\infty}^{\infty}\Pr[S_+ > \tau]\,p_-(\tau)\,\mathrm d\tau.$$
3. **Read it as an average.** Fix a negative's score $\tau$. The chance that an independent positive beats it is $\Pr[S_+ > \tau]$. The last integral averages that chance over $\tau$ drawn from the negatives' density, so it is $\mathbb E\big[\Pr[S_+ > S_- \mid S_-]\big] = \Pr[S_+ > S_-]$: the probability that a random positive outscores a random negative.

**The binormal model.** A common model has normal scores in both classes, $S_- \sim \mathcal N(0, 1)$ and $S_+ \sim \mathcal N(m, \sigma^2)$. The difference of two independent normals is normal, with the means subtracted and the variances added, so $S_+ - S_- \sim \mathcal N(m, 1 + \sigma^2)$. The AUC is the probability that this difference is positive, and the ROC curve follows by setting the threshold from the FPR:
$$\mathrm{AUC} = \Phi\big(m/\sqrt{1 + \sigma^2}\big), \qquad \mathrm{TPR}(\mathrm{FPR}) = 1 - \Phi\Big(\frac{\Phi^{-1}(1 - \mathrm{FPR}) - m}{\sigma}\Big).$$
The second formula repeats steps 3 and 4 of the Gaussian derivation in @sec-ht-gaussian: the threshold is $\tau = \Phi^{-1}(1 - \mathrm{FPR})$, and a positive exceeds it when $(S_+ - m)/\sigma > (\tau - m)/\sigma$, now with spread $\sigma$ under $Q$.

### Example: same AUC, different detectors

Detector A has positives that score $\mathcal N(1, 1)$, so $m = 1$, $\sigma = 1$ and $\mathrm{AUC} = \Phi(1/\sqrt2) = 0.7602$. Detector B has positives that score $\mathcal N(m, 2^2)$, more spread out, with $m$ chosen to give the same AUC: $m/\sqrt5 = 1/\sqrt2$ gives $m = \sqrt{5/2} = 1.5811$ and the same AUC, 0.7602. @tbl-ht-auc compares them using the binormal ROC formula. For example, at FPR $10^{-3}$ the threshold is 3.0902; A's TPR is $1 - \Phi(3.0902 - 1) = 0.0183$, and B's is $1 - \Phi\big((3.0902 - 1.5811)/2\big) = 1 - \Phi(0.7546) = 0.2253$.

| Detector | AUC (formula / simulated) | TPR at FPR $= 0.1$ | TPR at FPR $= 10^{-3}$ |
| --- | --- | --- | --- |
| A, $\mathcal N(1, 1)$ | 0.7602 / 0.7605 | 0.3891 | 0.0183 |
| B, $\mathcal N(1.5811, 2^2)$ | 0.7602 / 0.7601 | 0.5595 | 0.2253 |

: Two binormal detectors with equal AUC. The simulated AUC counts correctly ranked pairs among 1,000,000. {#tbl-ht-auc}

At an FPR of $10^{-3}$, detector B catches 12 times as many positives, $0.2253/0.0183 = 12.3$ (@tbl-ht-auc). On linear axes the two ROC curves cross and look like comparable detectors; on log axes the gap at low FPR is obvious (@fig-ht-auc-low-fpr). B's positives are more spread out, so a few of them score far above every negative. Those are exactly the cases a strict threshold catches.

![ROC curves of two detectors with the same AUC of 0.760. Left, linear axes: the curves cross and enclose equal areas. Right, log axes: at FPR $= 10^{-3}$ detector B has 12 times the TPR of detector A.](figures/hypothesis_testing_auc_low_fpr.svg){#fig-ht-auc-low-fpr}

### Interpretation

The AUC hides this because it averages over all FPRs equally. The region FPR $\le 10^{-3}$ is a tenth of a percent of the horizontal axis, so it barely affects the area, and most of the AUC comes from FPRs nobody would operate at. The remedy is to report the TPR at a fixed low FPR, chosen by the application, and to plot ROC curves on log–log axes. A log–log ROC curve shows exactly the behaviour near zero that separated the Gaussian and Laplace curves in @sec-ht-laplace.

Neither curve here is the best possible for its detector. B's score threshold is not the optimal test for its own $P$ and $Q$: with unequal variances the log likelihood ratio, $\ln$ of the $\mathcal N(m, \sigma^2)$ density over the $\mathcal N(0, 1)$ density, has the term $\frac12(1 - 1/\sigma^2)s^2$, which is quadratic in the score $s$ and positive for $\sigma = 2$. So the ratio is large at both ends, and the optimal test also flags very *low* scores. Neyman–Pearson says the optimal ROC is at least as good as the one plotted.

An extreme case is in [Exercise 5](../../book-exercises/part01-prerequisites/04-hypothesis-testing.md#sec-sol-ht-5): a detector with AUC only 0.525 that catches 5% of positives at FPR $10^{-3}$.

### Check yourself

- A detector's ROC curve lies on the diagonal except for a jump to TPR $0.05$ at FPR $0$. What is its AUC, and is it useless? *(About $0.5 + 0.05 \times 0.5 \approx 0.525$. It is not useless: it catches 5% of positives with no false alarms.)*
- Why does the AUC not depend on any threshold, while TPR at FPR $10^{-3}$ does? *(The AUC integrates over all thresholds. TPR at a fixed FPR evaluates one threshold, the one whose FPR is $10^{-3}$.)*

## The Bayesian view {#sec-ht-bayes}

### Context

Neyman–Pearson treats the two hypotheses asymmetrically and uses no prior. In many problems you do know a base rate: 1% of patients are ill, one transaction in a thousand is fraud. Then the question becomes "given this observation, how likely is $H_1$?", and the likelihood ratio answers it.

### Statement

Bayes' rule says how evidence updates a belief, and in terms of odds it is a single multiplication. Write $\pi = \Pr[H_1]$ for the prior probability of $H_1$, the base rate. The odds of $H_1$ against $H_0$ after seeing $x$ are the odds before seeing it, multiplied by the likelihood ratio of $x$:
$$\underbrace{\frac{\Pr[H_1 \mid x]}{\Pr[H_0 \mid x]}}_{\text{posterior odds}} = \underbrace{\frac{\pi}{1 - \pi}}_{\text{prior odds}} \times \underbrace{\frac{q(x)}{p(x)}}_{\Lambda(x)}.$$ {#eq-ht-posterior-odds}
If you want to make as few mistakes as possible, you should say $H_1$ whenever the posterior odds exceed 1, that is, exactly when $\Lambda(x) > (1 - \pi)/\pi$. So the decision that minimizes the probability of error is again an LR test, with the threshold set by the prior.

### Derivation

1. **Bayes' rule.** The probability of $H_1$ and $x$ together is $\pi q(x)$, and of $H_0$ and $x$ together is $(1 - \pi)p(x)$. Dividing each by the total probability of $x$, $m(x) = \pi q(x) + (1 - \pi)p(x)$, gives $\Pr[H_1 \mid x] = \pi q(x)/m(x)$ and $\Pr[H_0 \mid x] = (1 - \pi)p(x)/m(x)$. Dividing one by the other cancels $m(x)$ and gives @eq-ht-posterior-odds.
2. **Error of a test.** A test errs when $H_0$ holds and it rejects, with probability $(1 - \pi)\alpha_\phi$, or when $H_1$ holds and it misses, with probability $\pi \beta_\phi$. By Fact 1, writing $\alpha_\phi = \sum_x\phi p$ and $\beta_\phi = 1 - \sum_x\phi q$,
$$(1 - \pi)\alpha_\phi + \pi \beta_\phi = \pi + \sum_x \phi(x)\big((1 - \pi)p(x) - \pi q(x)\big).$$
3. **Minimize term by term.** Each $\phi(x)$ appears in one term, so set $\phi(x) = 1$ where the bracket is negative and 0 where it is positive. The bracket is negative where $\pi q(x) > (1 - \pi)p(x)$, or $\Lambda(x) > (1 - \pi)/\pi$.
4. **The minimum error on the curve.** Because the optimal test is an LR test, it lies on the trade-off curve, so the minimum error is $\min_\alpha\big[(1 - \pi)\alpha + \pi T(\alpha)\big]$. With equal priors, $\pi = \frac12$, it is $\frac12\min_\alpha(\alpha + T(\alpha))$. Minimizing $\alpha + T(\alpha)$ is maximizing the advantage $1 - \alpha - T(\alpha)$, whose maximum is TV by @eq-ht-tv, so the minimum error is $\frac12(1 - \mathrm{TV})$. For the Gaussian pair at $\mu = 1$ that is $\frac12(1 - 0.3829) = 0.3085$.

The same holds for any outcome you observe, not only for $x$. If the evidence is just "the test fired", its likelihood ratio is the probability of firing under $H_1$ over that under $H_0$, $\Pr[\text{fire} \mid H_1]/\Pr[\text{fire} \mid H_0] = (1 - \beta)/\alpha$, the power divided by the false positive rate.

### Example: screening at 1% prevalence

Use the biomarker test with $\mu = 2$ from @tbl-ht-gaussian, in a population where 1% are ill (prior odds $1/99$). To get the probability of illness after a positive result, multiply the prior odds by the likelihood ratio of a positive, then turn odds $o$ back into a probability $o/(1 + o)$.

At $\alpha = 0.05$ the power is 0.6388, so a positive result has likelihood ratio $0.6388/0.05 = 12.8$. The posterior odds are $12.8/99 = 0.129$, so $\Pr[\text{ill} \mid \text{positive}] = 0.129/1.129 = 0.114$. Nine in ten positives are false alarms. At $\alpha = 10^{-3}$ the power is 0.1378, the likelihood ratio is $0.1378/0.001 = 137.8$, the posterior odds are $137.8/99 = 1.392$, and $\Pr[\text{ill} \mid \text{positive}] = 1.392/2.392 = 0.582$.

The stricter threshold catches fewer ill patients but makes a positive result far more informative (@fig-ht-posterior). This is the second reason low-FPR behaviour matters: when positives are rare, the false-positive rate, not the power, decides whether an alarm is worth acting on.

**The Bayes rule here.** Minimizing the error probability at $\pi = 0.01$ means rejecting when $\Lambda(x) > (1 - \pi)/\pi = 99$. For the Gaussian, $\Lambda(x) = e^{\mu x - \mu^2/2} > 99$ means $x > \mu/2 + \ln(99)/\mu = 1 + 4.5951/2 = 3.2976$. That test has $\alpha = 1 - \Phi(3.2976) = 4.88\times10^{-4}$ and power only $1 - \Phi(3.2976 - 2) = 0.0972$: when errors of both kinds count equally and the disease is rare, it is cheapest to rarely say "ill". If missing a case costs $c$ times as much as a false alarm, the quantity to minimize becomes $(1 - \pi)\alpha + c\pi \beta$, the same argument gives the threshold $(1 - \pi)/(c\pi)$, and the test moves along the same curve.

![Posterior probability of illness after a positive result, as a function of the prior, for the $\mu = 2$ biomarker test at two thresholds. At 1% prevalence (dashed line) the stricter test gives 0.582 against 0.114.](figures/hypothesis_testing_posterior.svg){#fig-ht-posterior}

### Check yourself

- If the prior doubles from 1% to 2%, roughly how does the posterior odds after a positive change? *(It roughly doubles: posterior odds scale linearly with prior odds, $2/98$ against $1/99$.)*
- Why is the Bayes-optimal test an LR test even though Bayes and Neyman–Pearson ask different questions? *(Both trade off the two errors linearly: Bayes with fixed weights $1 - \pi$ and $\pi$, Neyman–Pearson with the exchange rate $\tau$, which plays the role of a Lagrange multiplier on the false-positive budget. Each linear trade-off is minimized by a threshold on $\Lambda$.)*

## Estimating error rates from finite trials {#sec-ht-estimation}

### Context

Everything so far assumed you know $P$ and $Q$. In practice you measure a detector's FPR and TPR on test data: $k$ false positives among $n$ negatives. Each negative is falsely flagged independently with probability $p$, so the count $k$ is $\mathrm{Bin}(n, p)$, the number of successes in $n$ independent trials that each succeed with probability $p$ ([Probability distributions](01-distributions.md)). The measured rate $k/n$ is therefore uncertain, and at low rates the uncertainty is large relative to the rate itself. A claim such as "TPR at FPR $10^{-3}$" is only as good as the interval around it.

### Statement: the Clopper–Pearson interval

The interval is built by asking which true rates would make the observed count unsurprising. A true rate far above $k/n$ would make a count as small as $k$ very unlikely; the upper end $u$ is the rate at which seeing $k$ or fewer has probability exactly $\alpha/2$. Likewise the lower end $l$ is the rate at which seeing $k$ or more has probability $\alpha/2$. For $k$ successes in $n$ trials and confidence level $1 - \alpha$, let $l$ and $u$ solve
$$\Pr[\mathrm{Bin}(n, l) \ge k] = \frac \alpha2, \qquad \Pr[\mathrm{Bin}(n, u) \le k] = \frac \alpha2,$$
with $l = 0$ if $k = 0$ and $u = 1$ if $k = n$. Then $\Pr[l \le p \le u] \ge 1 - \alpha$ for every true rate $p$.

### Derivation (upper end; the lower end is symmetric)

The interval is random, because it depends on the observed count, and the claim is about how often it misses the fixed true rate $p$. The plan is to show that it misses above, $p > u$, with probability at most $\alpha/2$. Write $F_p(k) = \Pr[\mathrm{Bin}(n, p) \le k]$ for the probability of at most $k$ successes at rate $p$, and let $K \sim \mathrm{Bin}(n, p)$ be the observed count.

1. **Monotonicity.** $F_p(k)$ is continuous and strictly decreasing in $p$ for $k < n$: a larger success probability makes small counts less likely (@sec-apx-tail-clopper in the appendices proves it). So for each $k$ there is a single rate $u(k)$ with $F_{u(k)}(k) = \alpha/2$, and the true rate lies at or above the upper end, $p \ge u(K)$, exactly when $F_p(K) \le F_{u(K)}(K) = \alpha/2$.
2. **A CDF evaluated at its own draw is rarely small.** It remains to show $\Pr[F_p(K) \le \alpha/2] \le \alpha/2$. Let $k^*$ be the largest $k$ with $F_p(k) \le \alpha/2$. Since $F_p$ is increasing in $k$, $F_p(K) \le \alpha/2$ exactly when $K \le k^*$, so $\Pr[F_p(K) \le \alpha/2] = \Pr[K \le k^*] = F_p(k^*) \le \alpha/2$. (If no such $k^*$ exists, the probability is 0.)
3. **Combine.** So $\Pr[p \ge u(K)] \le \alpha/2$, and the same argument gives $\Pr[p \le l(K)] \le \alpha/2$ at the lower end. The probability that at least one of two events happens is at most the sum of their probabilities (the union bound), so the interval misses with probability at most $\alpha$, and its coverage is at least $1 - \alpha$.

The interval is **exact** in the sense that it uses the binomial distribution itself rather than an approximation, and **conservative**: its coverage is at least $1 - \alpha$, often more.

**Zero counts and the rule of three.** If $k = 0$, the probability of seeing no successes at rate $u$ is $(1 - u)^n$, so the one-sided upper bound at level $0.95$ solves $(1 - u)^n = 0.05$. That gives $u = 1 - 0.05^{1/n}$. For large $n$, $0.05^{1/n} = e^{-\ln(20)/n} \approx 1 - \ln(20)/n$, so $u \approx \ln(20)/n \approx 3/n$, since $\ln 20 = 3.00$ to two decimals. With $n = 3{,}000$ negatives and no false positives, $u = 9.981\times10^{-4}$. Turning this around, to certify an FPR below $10^{-3}$ at 95% confidence you need $(1 - 10^{-3})^n \le 0.05$, that is, at least $\ln 20/(-\ln(1 - 10^{-3})) = 2{,}994.2$, so 2,995, negatives with *no* false positive at all.

### Example: measuring TPR at FPR $10^{-3}$

Take the $\mu = 2$ biomarker, with the threshold $\tau = 3.0902$ that gives a true FPR of $10^{-3}$ and a true TPR of 0.1378. Test it on $n = 10{,}000$ negatives and 10,000 positives. One simulated run gives @tbl-ht-one-run.

| Quantity | Count | Estimate | 95% Clopper–Pearson interval |
| --- | --- | --- | --- |
| False positives | 10 | 0.0010 | $[0.00048, 0.00184]$ |
| True positives | 1,350 | 0.1350 | $[0.1284, 0.1419]$ |

: One simulated evaluation of the $\mu = 2$ test at its $10^{-3}$ threshold, with 10,000 examples per class. {#tbl-ht-one-run}

The TPR is pinned down to about $\pm 0.007$, half the interval's width of $0.1419 - 0.1284 = 0.0135$. The FPR is not: the data are consistent with a true FPR anywhere from about half to almost twice the target (@tbl-ht-one-run). The number of negatives limits what a low-FPR claim can mean.

Two other common ways to build the FPR interval from $k = 10$, $n = 10{,}000$ do worse. The **Wald** interval uses the normal approximation to the binomial, $\hat p \pm 1.96\sqrt{\hat p(1 - \hat p)/n}$ with $\hat p = k/n$, where 1.96 is $\Phi^{-1}(0.975)$. Here the half-width is $1.96\sqrt{0.001 \times 0.999/10{,}000} = 0.00062$, giving $[0.00038, 0.00162]$. That is too narrow and symmetric for a skewed count, and at $k = 0$ it collapses to $[0, 0]$, claiming certainty from no data. The **Hoeffding** interval comes from Hoeffding's inequality ([Concentration inequalities](02-concentration.md)), which says that the average of $n$ independent variables in $[0, 1]$ strays from its mean by at least $v$ with probability at most $2e^{-2nv^2}$. Setting that to 0.05 gives the two-sided half-width $v = \sqrt{\ln(2/0.05)/(2n)} = \sqrt{\ln(40)/20{,}000} = 0.0136$, more than 13 times the rate being measured. A bound that uses only the range $[0, 1]$ ignores the tiny variance $p(1 - p)$.

Clopper–Pearson is conservative, not loose: over 2,000 simulated evaluations with true FPR $10^{-3}$ and $n = 10{,}000$, its interval covered the true rate 97.20% of the time, above the nominal 95%. @fig-ht-rate-ci shows how slowly the interval tightens with $n$ when the observed rate stays at $10^{-3}$.

![95% Clopper–Pearson intervals for an observed false-positive rate of exactly $10^{-3}$, as the number of negatives grows from $10^3$ to $10^6$, with the upper end of the two-sided Hoeffding interval for comparison.](figures/hypothesis_testing_rate_ci.svg){#fig-ht-rate-ci}

### Interpretation

What sets the relative width of the interval is mainly the expected count $k = np$, the number of false positives you expect to see, not $n$ alone. A count of $k$ fluctuates by about $\sqrt k$, so its relative uncertainty is about $1/\sqrt k$. Ten events give a factor-of-two uncertainty, $[0.00048, 0.00184]$ around $10^{-3}$. A hundred events, at $n = 100{,}000$, give about $\pm 20\%$: $[0.00081, 0.00122]$. This gives a rule of thumb: to measure an FPR of $p$ to within a factor of about 1.2, you need on the order of $100/p$ negatives, $10^5$ for $p = 10^{-3}$.

The threshold is also estimated. If you choose the threshold on the same data so that the empirical FPR is $10^{-3}$, the true FPR of that threshold is uncertain in exactly the way shown above. Choose thresholds on one split and measure rates on another.

### Check yourself

- You see 0 false positives in 300 negatives. What can you claim at 95% confidence? *(FPR $\le$ about $3/300 = 0.01$, by the rule of three. Nothing about $10^{-3}$.)*
- Why is Hoeffding so much wider than Clopper–Pearson at low rates? *(Hoeffding assumes the worst-case variance $\frac14$ for a variable in $[0, 1]$; the true variance of a false-positive indicator is $p(1 - p) \approx 10^{-3}$.)*

## Code: check every result by computation and simulation

The script [code/hypothesis_testing.py](../../code/part01-prerequisites/hypothesis_testing.py) computes every number quoted in this chapter and draws every figure. Each error rate, trade-off curve, interval and simulation is a small typed function; the script calls them once for the table and loops them over parameters for the figures. It uses only the standard library and NumPy: $\Phi$ comes from `math.erfc`, and $\Phi^{-1}$ and the Clopper–Pearson ends are found by bisection. From the repository root:

```bash
uv run python code/part01-prerequisites/hypothesis_testing.py
```

**Output** (seed 0), summarized in @tbl-ht-code:

| Experiment | Result |
| --- | --- |
| Spam example, LR test at $\alpha = 0.25$ | $\tau = 1.5$, $\gamma = 0.75$, power 0.625 |
| 200,000 random tests on the spam example | 11,727 with size $\le 0.25$; best power 0.587 |
| Gaussian $\mu = 2$, $\alpha = 0.05$ | $\beta = 0.3612$ exact; simulated $\hat\alpha = 0.0503$, $\hat\beta = 0.3617$ |
| Gaussian $\mu = 2$, $\alpha = 10^{-3}$ | power 0.1378 |
| Laplace $\kappa = 1$, $\alpha = 0.25$ | $\beta = 0.3679$ exact; simulated $\hat\alpha = 0.2498$, $\hat\beta = 0.3685$ |
| Same TV 0.3935, $\alpha = 10^{-6}$ | Laplace power $2.718\times10^{-6}$; Gaussian ($\mu = 1.0301$) $9.829\times10^{-5}$ |
| Best advantage, Gaussian $\mu = 1$ | Formula 0.3829; grid maximum 0.3829 at $\alpha = 0.3085$ |
| Equal-AUC detectors A and B | AUC 0.7602 both; TPR at $10^{-3}$: 0.0183 and 0.2253 |
| Screening at 1% prevalence, $\mu = 2$ | Posterior after a positive: 0.114 at $\alpha = 0.05$; 0.582 at $\alpha = 10^{-3}$ |
| Clopper–Pearson, $k = 10$, $n = 10{,}000$ | $[0.00048, 0.00184]$; coverage over 2,000 runs 0.9720 |

: Results printed by the script with seed 0. {#tbl-ht-code}

[tests/test_hypothesis_testing.py](../../tests/test_hypothesis_testing.py) checks that the output still matches @tbl-ht-code and every other number in the text.

Four things stand out. First, Neyman–Pearson is not just a theorem: a random search over 200,000 tests never comes close to the LR test at the same level. Second, the closed-form curves match simulation to within sampling error, for both the Gaussian and the Laplace pairs, so the derivations can be trusted at the levels where simulation is too expensive. Third, summaries can agree while the curves differ: equal TV or equal AUC says nothing about power at $10^{-3}$ or $10^{-6}$, which is why the whole curve is the object to look at. Fourth, low rates are expensive to measure: ten false positives pin the FPR down only to a factor of two.

## References

- Lehmann & Romano, *Testing Statistical Hypotheses* (3rd ed., 2005), ch. 3: most powerful tests and power functions, the rigorous Neyman–Pearson lemma (§3.2, Theorem 3.2.1) with the converse that every most powerful test is an LR test almost everywhere, and monotone likelihood ratios. The trade-off function is the smallest type II error as a function of the level, that is, one minus the power of the most powerful test.
- Casella & Berger, *Statistical Inference* (2nd ed., 2002), §8.3: a gentler proof of the lemma (Theorem 8.3.12) and the Karlin–Rubin theorem for one-sided composite tests.
- Wasserman, *All of Statistics* (2004), ch. 10: a compact tour of tests, power and p-values.
- Cover & Thomas, *Elements of Information Theory* (2nd ed., 2006), §11.7–11.9: Neyman–Pearson from the information-theory side, the Chernoff–Stein lemma and Chernoff information.
- Fawcett, "An introduction to ROC analysis" (Pattern Recognition Letters, 2006): ROC curves (the trade-off function turned upside down), AUC as a ranking probability, and the convex hull of a classifier's ROC, which is the convexity property proved here.
- Clopper & Pearson, "The use of confidence or fiducial limits illustrated in the case of the binomial" (Biometrika, 1934), and Brown, Cai & DasGupta, "Interval Estimation for a Binomial Proportion" (Statistical Science, 2001): the exact interval, and why the Wald interval fails.

## Cheat sheet

@tbl-ht-cheat-sheet collects every result in the chapter.

| Result | Statement | Where it comes from |
| --- | --- | --- |
| Errors of a test $\phi$ | $\alpha_\phi = \mathbb E_P\phi$, $\beta_\phi = 1 - \mathbb E_Q\phi$, power $= 1 - \beta_\phi$ | Fact 1 |
| Neyman–Pearson | The size-$\alpha$ test that rejects when $q/p > \tau$ (randomizing at $=$) is most powerful at level $\alpha$ | $(\phi^* - \phi)(q - \tau p) \ge 0$ pointwise |
| Trade-off function | $T(\alpha) = \inf\{\beta_\phi : \alpha_\phi \le \alpha\}$; non-increasing, $\le 1 - \alpha$, convex, continuous | Mixing tests; change of measure |
| Slope of $T$ | $T'(\alpha) = -\Lambda$ at the threshold | $\mathrm d\beta/\mathrm d\alpha = -q(\tau)/p(\tau)$ |
| Gaussian $\mathcal N(0, 1)$ vs $\mathcal N(\mu, 1)$ | $T(\alpha) = \Phi(\Phi^{-1}(1 - \alpha) - \mu)$; $n$ draws: $\mu = h\sqrt n/\sigma$ | LR increasing in $x$ |
| Laplace $\mathrm{Lap}(0, b)$ vs $\mathrm{Lap}(m_1, b)$ | $1 - e^\kappa \alpha$; $e^{-\kappa}/(4\alpha)$; $e^{-\kappa}(1 - \alpha)$, with $\kappa = m_1/b$ | LR clipped to $[e^{-\kappa}, e^\kappa]$ |
| Best advantage | $\max_\phi(1 - \alpha_\phi - \beta_\phi) = \mathrm{TV}(P, Q)$; Gaussian $2\Phi(\mu/2) - 1$; Laplace $1 - e^{-\kappa/2}$ | Reject where $q > p$ |
| AUC | $\Pr[S_+ > S_-]$; binormal $\Phi(m/\sqrt{1 + \sigma^2})$ | Change of variables in $\int\mathrm{TPR}\,\mathrm d\mathrm{FPR}$ |
| Bayes | Posterior odds $=$ prior odds $\times \Lambda(x)$; minimum error at $\pi = \frac12$ is $\frac12(1 - \mathrm{TV})$ | Bayes' rule |
| Clopper–Pearson | $\Pr[\mathrm{Bin}(n, l) \ge k] = \Pr[\mathrm{Bin}(n, u) \le k] = \alpha/2$; zero counts: $u \approx 3/n$ | Inverting binomial tails |

: Every result in this chapter. {#tbl-ht-cheat-sheet}

**Rules of thumb**

- To find the best test, compute the likelihood ratio and threshold it. If it is monotone in a statistic, threshold the statistic.
- Look at the whole trade-off curve, on log axes if false positives must be rare. TV and AUC are single points or averages of it.
- A bounded likelihood ratio caps power at small $\alpha$: power $\le \Lambda_{\max}\,\alpha$.
- To measure a rate near $p$, plan for about $100/p$ trials, and report a Clopper–Pearson interval.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part01-prerequisites/04-hypothesis-testing.md).
