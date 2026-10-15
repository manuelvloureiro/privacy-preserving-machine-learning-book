# Hypothesis testing

## Why this matters

A spam filter reads an email and must say "ham", meaning a legitimate email, or "spam". A fraud detector sees a transaction and must say "normal" or "fraud". A monitoring job reads a batch of sensor readings and must say whether the sensor has drifted. These are all the same problem. An observation came from one of two distributions, and you must decide which one. Hypothesis testing is the study of that decision.

Later parts of this book ask how well *any* test could tell two distributions apart, not just the test someone happened to build. This chapter covers three things you need for that. The first is what the best possible test looks like. The second is how good that test can be. The third is how to measure a real test's error rates from a finite number of trials.

The central tool is a number computed from each observation. It says how many times more likely the observation is under one distribution than under the other, and it is called the **likelihood ratio**. The best test compares this number with a cut-off. A stricter cut-off raises fewer false alarms but misses more real cases. Listing the best pair of error rates you can reach at every cut-off gives a curve, and that curve depends only on the two distributions. Much of what goes wrong in practice comes from summarizing the curve by one number. A single number hides how a test behaves when false alarms must be rare, and that is often the case that matters.

## Tests and their two errors {#sec-ht-setup}

The chapter's running example is a spam filter that counts suspicious words in an email. Any count above 3 is recorded as 3. The default assumption is that an email is ham, and this default is called the **null hypothesis**, written $H_0$. Under it, the count has a distribution we call $P$. We write $p(x)$ for the probability, under $P$, that the count equals $x$.

The **alternative hypothesis** $H_1$ is that the email is spam. Under it, the count has a distribution $Q$, and $q(x)$ is the probability of the count $x$. @tbl-ht-spam gives both distributions. Its last row is the likelihood ratio, written $\Lambda(x) = q(x)/p(x)$, and a large ratio points to spam. Spam tends to contain more suspicious words, so the ratio grows with the count. A count of 3 is four times as likely in spam as in ham, and a count of 0 is a quarter as likely.

| $x$ | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| $p(x)$, ham | 0.4 | 0.3 | 0.2 | 0.1 |
| $q(x)$, spam | 0.1 | 0.2 | 0.3 | 0.4 |
| $\Lambda(x) = q(x)/p(x)$ | 0.250 | 0.667 | 1.500 | 4.000 |

: Word-count distributions under ham ($P$) and spam ($Q$), and their likelihood ratio. {#tbl-ht-spam}

[Divergences](03-divergences.md) writes the ratio the other way up, with $p$ on top. In this chapter the ratio always has the alternative on top.

A **test** is a rule that looks at the count and decides whether to say "spam", which is called rejecting $H_0$. The simplest test picks a set of counts and rejects exactly when the count lands in it. We will also let a test reject some counts only with some probability, by flipping a biased coin. Such a **randomized test** can spend a false-alarm budget exactly, as the example below shows, where a plain set of counts often cannot. We describe any test by a function $\phi$, where $\phi(x)$ is the probability of rejecting after seeing the count $x$. A test that never flips a coin takes only the values 0 and 1.

A test can make two kinds of mistake. The first is to reject when the email is ham, flagging a good email as spam. This is a **type I error**, or false positive. Its probability is the **false positive rate (FPR)** of the test, written $\alpha$.

The second mistake is to keep $H_0$ when the email is spam, letting spam through. This is a **type II error**, or false negative, and its probability is written $\beta$.

The **power** of a test is $1 - \beta$, the probability that it catches spam. In classification the power is called the **true positive rate (TPR)**, or recall.

A test has **level** $\alpha$ if its FPR is at most $\alpha$. The level is a budget, and the FPR is what the test actually spends. Both error rates are averages of $\phi$. On a ham email the test rejects with probability $\phi(x)$. Averaging $\phi$ over counts drawn from $P$, that is, summing $\phi(x)\,p(x)$ over all counts, gives the FPR. Averaging $\phi$ over counts drawn from $Q$ gives the power in the same way. Take the test "reject when the count is at least 2". Its FPR is the ham probability of the counts 2 and 3, which is $0.2 + 0.1 = 0.3$. Its power is the spam probability of the same counts, $0.3 + 0.4 = 0.7$. Its type II error is therefore 0.3.

The two errors pull in opposite directions. Adding a count to the rejection set adds its ham probability to the FPR and its spam probability to the power. Adding a count therefore raises the FPR and lowers the type II error, and growing the set cannot lower both. The convention treats the two errors asymmetrically. The null hypothesis is the default, and a false alarm, such as blocking a real email, is the costly error you budget for. You fix the level first, and then you look for the most powerful test at that level.

A useful yardstick is a test that ignores the email and rejects with probability $\alpha$ by flipping a coin. It rejects ham and spam equally often, so its FPR and its power are both $\alpha$. Any test worth using must beat it. The test that rejects only a count of 0 does not. Its FPR is 0.4, the same as a coin that rejects 40% of emails. Its power is only 0.1, while the coin's power is 0.4. The test rejects the count most typical of ham, where spam is four times rarer.

A plain set of counts can spend only certain budgets, because each count brings its whole ham probability with it. The sets that reject every count from some cut-off upward have FPRs of 0, 0.1, 0.3, 0.6 and 1. Suppose your policy allows an FPR of 0.25. Rejecting only a count of 3 spends just 0.1. Adding the count 2 brings the FPR to 0.3, which overspends. The randomized test that always rejects 3, and rejects 2 with probability 0.75, spends exactly 0.25.

The randomized test is a coin flip between two plain tests. With probability 0.25 it runs "reject 3", and otherwise it runs "reject 2 or 3". Both error rates are averages of $\phi$, so its errors are the same weighted average of the two tests' errors. Its FPR is a quarter of 0.1 plus three quarters of 0.3, which is exactly 0.25. Plot each test as a point, its FPR against its type II error. Mixing tests in this way makes every point on the segment between two achievable points achievable too.

Randomization is needed only for discrete data, where moving a threshold makes the FPR jump from one count to the next. For a continuous score, moving the threshold changes the FPR smoothly, so a plain threshold can hit any budget.

@fig-ht-errors shows a continuous example. A biomarker is normal with variance 1 in both groups, centred at 0 in healthy patients and at 2 in ill ones. A screening test flags patients above a threshold of 1.645, which flags 5% of healthy patients. Ill patients fall below the threshold with probability 0.361. The test therefore catches about two ill patients in three. Lowering the threshold catches more ill patients and alarms more healthy ones.

![The two errors of the threshold test "reject when $x > t$" for $P = \mathcal N(0, 1)$ against $Q = \mathcal N(2, 1)$, with $t$ set so that $\alpha = 0.05$. The blue area under $P$ beyond $t$ is $\alpha$; the orange area under $Q$ before $t$ is $\beta = 0.361$.](figures/hypothesis_testing_errors.svg){#fig-ht-errors}

Which point on the plot of FPR against type II error to pick is a policy decision. What the mathematics can say is which points are achievable at all, and the next section answers that.

## The Neyman–Pearson lemma {#sec-ht-np}

Among all tests with a given level, which one has the most power? There are infinitely many tests, one for every function $\phi$ with values between 0 and 1. The answer is that you only need to look at one family.

The reasoning is about a budget. Rejecting a count $x$ costs $p(x)$ of false-positive budget, and it buys $q(x)$ of power. The exchange rate at $x$ is therefore $q(x)/p(x)$, which is the likelihood ratio $\Lambda(x)$. To get the most power for a fixed budget, spend it on the counts with the best exchange rate first. That means rejecting exactly where the likelihood ratio is above some cut-off.

A **likelihood ratio (LR) test** does exactly this. It picks a threshold, written $\eta$, and rejects wherever $q(x)$ is more than $\eta\,p(x)$. It keeps $H_0$ wherever $q(x)$ is less. On ties, where the two are equal, it may reject with any probability. Writing the rule this way, rather than as a comparison of the ratio with the threshold, also handles outcomes that $P$ never produces. At such an outcome $p(x)$ is zero, so any positive $q(x)$ is above $\eta\,p(x)$ whatever the threshold. Rejecting it costs no budget at all.

The **Neyman–Pearson lemma** says that this family contains the best tests, in two parts. First, no test whose FPR is at most that of an LR test has a smaller type II error. Second, for every level between 0 and 1 there is an LR test whose FPR is exactly that level. To reach the level exactly, it may need to randomize on the ties. Together, the two parts say that the most powerful test at level $\alpha$ is the LR test whose FPR is exactly $\alpha$.

The budget picture also explains why no other test can beat it. Wherever the LR test rejects, each unit of budget buys at least $\eta$ units of power. Wherever it keeps $H_0$, each unit would buy at most $\eta$. Any other test with the same budget must move some spending from counts where the ratio is above $\eta$ to counts where it is below. Each such move gives up at least as much power as it gains.

Back to the spam filter with a budget of 0.25. The count 3 has the highest ratio, 4, and rejecting it costs 0.1. The count 2 comes next, with a ratio of 1.5. Rejecting it fully would cost another 0.2 and overspend. The cut-off therefore sits at the ratio of the count 2, which is 1.5. Counts above it are always rejected, and the count 2, tied with the cut-off, is rejected only in part. The LR test spends the remaining 0.15 on the count 2 by rejecting it with probability 0.75. This is the randomized test from @sec-ht-setup. Its power is 0.625.

A brute-force check confirms that no other test does better. It draws 200,000 random tests, each picking every $\phi(x)$ uniformly between 0 and 1. Of these, 11,727 stay within the budget of 0.25. The most powerful of them reaches 0.587. @fig-ht-discrete-tradeoff plots 400 random tests. Every one lies on or above the line traced by the LR tests, which the next section calls the trade-off curve.

![Errors $(\alpha, \beta)$ of 400 random tests on the spam example (grey), against the curve of likelihood-ratio tests. Black dots are the deterministic thresholds, the red square is the randomized test of size 0.25, and the dotted line is the coin-flip test.](figures/hypothesis_testing_discrete_tradeoff.svg){#fig-ht-discrete-tradeoff}

The best test looks at an observation only through its likelihood ratio. Two observations with the same ratio are interchangeable for testing, and everything else about them is noise.

Often the likelihood ratio grows with some simple statistic, such as the word count. Then "the ratio is above $\eta$" is the same event as "the statistic is above some threshold". The best test is then a plain threshold on that statistic, which is why thresholding a score works. A fraud model whose score is calibrated is an example. Calibrated means that the score is the probability of fraud given the transaction. That probability grows with the likelihood ratio, as @sec-ht-bayes shows. Thresholding a calibrated score is therefore an LR test, and no other use of the data catches more fraud at the same false-alarm rate. A score may instead rank transactions differently from the likelihood ratio. Re-ranking them by an estimate of the ratio then helps, provided the estimate puts the transactions in the same order as the true ratio.

With several independent observations, the probability of the whole sample is the product of the individual probabilities. The likelihood ratio of the sample is therefore the product of the individual ratios. The LR test thresholds its logarithm, which is a sum of independent terms. Sums of independent terms concentrate ([Concentration inequalities](02-concentration.md)). As the number of observations $n$ grows, the best test can therefore make both errors small.

The Chernoff–Stein lemma makes this precise for one setting. Hold the FPR fixed at any level, and ask how fast the type II error falls as $n$ grows. It falls like $e^{-n\,\mathrm{KL}(P\|Q)}$. Here $\mathrm{KL}(P\|Q)$ is the Kullback–Leibler (KL) divergence of [Divergences](03-divergences.md). It is the average log-ratio by which a ham observation favours ham over spam.

The Neyman–Pearson lemma needs both hypotheses to be **simple**. That means each one names a single, fully specified distribution. A **composite** hypothesis, such as "the mean is positive", names a whole family of distributions. Each member has its own LR test, so there is usually no single test that is best against all of them at once. The main exception is a **one-sided test**, which asks whether a parameter such as the mean is above a given value. Suppose that for every member of the family the likelihood ratio grows with the same statistic. Then every member's LR test is the same threshold on that statistic. That one threshold is then best against all of them.

## Trade-off curves {#sec-ht-tradeoff}

A single test gives one pair of errors. To say how hard it is to tell $P$ from $Q$, you want the best type II error at every level at once. For each budget $\alpha$, look at every test whose FPR stays within it, and record the smallest type II error any of them achieves. As a function of the budget, this is the **trade-off function** of $P$ and $Q$, written $T(\alpha)$. By the Neyman–Pearson lemma, the LR test at each level achieves the best value. In practice you therefore compute the curve by sweeping the LR threshold.

The **receiver operating characteristic (ROC)** curve of the best test plots its power against its FPR, which is $1 - T(\alpha)$ against $\alpha$. It holds the same information as the trade-off curve, turned upside down. A particular score, thresholded at every value, also traces an ROC curve. By the lemma, that curve lies on or below the best one.

Every trade-off curve has four properties, and each has a simple reason. It never increases, because a bigger budget still allows every test that a smaller one allowed. It lies on or below the line $1 - \alpha$, because the coin-flip test is always available. It is convex, meaning that it bends upward like a bowl, because mixing two tests reaches every point between them. And it has no jumps. Strictly between the ends this follows from convexity. At a budget of 1 it follows from the bound $1 - \alpha$, which forces the curve down to 0. At a budget of zero the full edition gives a separate argument.

The four properties also work in reverse. Any function with all four is the trade-off curve of some pair of distributions. So these properties are the only constraints a trade-off curve has to satisfy.

The slope of the curve has a meaning too. Moving along it means rejecting one more count with a little more probability. That spends $p(x)$ of budget and buys $q(x)$ of power. The curve therefore falls by the likelihood ratio $\Lambda(x)$ for each unit the FPR rises. In other words, the slope at a point is minus the likelihood ratio at the threshold. The spam curve shows this. Its four segments in @fig-ht-discrete-tradeoff have slopes $-4$, $-1.5$, $-0.667$ and $-0.25$, which are minus the four ratios of @tbl-ht-spam. The LR test adds counts in decreasing order of their ratio, so each segment is less steep than the last. That is another way to see why the curve is convex.

The two extremes of the curve are the two extremes of difficulty. If the curve is the line $1 - \alpha$, no test beats a coin flip, which happens exactly when $P$ and $Q$ are the same. If it is zero everywhere, some test makes no errors at all, which happens when the two distributions never produce the same outcome. In between, the further the curve bends below the line, the easier the problem.

One problem is harder than another if its curve is nowhere below the other's, so that the best test misses at least as often at every budget. Two curves can cross, though. Then neither problem is harder at every level, and you have to know the level you care about.

Two more details are worth knowing. The curve's value at $\alpha = 0$ need not be 1. If $Q$ puts some probability on outcomes that $P$ never produces, a test can catch that probability with no false alarms. And the curve depends on which distribution is the null. Swapping $P$ and $Q$ reflects the curve across the line where the two errors are equal.

## The Gaussian trade-off curve {#sec-ht-gaussian}

The most common pair of hypotheses in practice is a normal distribution and a shifted copy of it. A score, a biomarker or a sample mean is normal under both hypotheses, with the same spread but different centres, and its curve has a closed form.

The formula uses the standard normal cumulative distribution function (CDF), written $\Phi$. The value $\Phi(x)$ is the probability that a standard normal variable is at most $x$. Its inverse $\Phi^{-1}$ gives the point below which a standard normal has a given probability. For example, $\Phi^{-1}(0.95) = 1.6449$. A standard normal exceeds that point with probability 0.05.

Take $P = \mathcal N(0, 1)$, the standard normal, against $Q = \mathcal N(\mu, 1)$, the same distribution shifted right by $\mu$. Dividing the two densities, the squared terms in $x$ cancel, and the likelihood ratio is $e^{\mu x - \mu^2/2}$. This grows with $x$, so the best test is a threshold on $x$, as in @fig-ht-errors. To spend a budget $\alpha$, the threshold must leave probability $\alpha$ of $P$ above it. That point is $\Phi^{-1}(1 - \alpha)$. The test misses when $Q$, centred $\mu$ higher, falls below that threshold. The curve is therefore
$$T(\alpha) = \Phi\big(\Phi^{-1}(1 - \alpha) - \mu\big).$$ {#eq-ht-gaussian-tradeoff}
The same curve holds for any two normal distributions with a common standard deviation. There $\mu$ is the distance between their means, measured in standard deviations. Shifting and rescaling the observation changes no test's errors, so only that distance matters. @fig-ht-gaussian-tradeoff shows the family.

![The Gaussian trade-off curve $\Phi(\Phi^{-1}(1 - \alpha) - \mu)$ for $\mu = 0, 0.5, 1, 2, 3$. Crosses are simulated errors of threshold tests at $\mu = 1$.](figures/hypothesis_testing_gaussian_tradeoff.svg){#fig-ht-gaussian-tradeoff}

Near $\alpha = 0$ the normal curve is vertical, because its slope is minus the likelihood ratio at the threshold. A small budget means a high threshold, and the normal likelihood ratio grows without limit as the threshold rises. The first sliver of false-alarm budget therefore buys power at an unlimited exchange rate. The Laplace pair in the next section behaves differently.

Averaging many readings makes the problem easier. Suppose you take $n$ independent readings, each normal with the same standard deviation $\sigma$ under both hypotheses. The alternative shifts their mean by an amount $\delta$. Because the readings are normal with a common spread, each reading's log likelihood ratio is a straight-line function of the reading. Their sum then depends on the data only through the sample mean, and the best test thresholds the sample mean. The sample mean has standard deviation $\sigma/\sqrt n$, so the curve is @eq-ht-gaussian-tradeoff with
$$\mu = \frac{\delta\sqrt n}{\sigma}.$$
How easily you can tell the two apart grows like $\sqrt n$, the rate at which a sample mean concentrates.

For example, a monitoring job averages 100 readings and must detect a drift of $0.2\sigma$ in the mean. The shift is then $\mu = 2$. At a false-alarm rate of 5%, the threshold sits 1.6449 standard deviations of the sample mean above the healthy mean. The job then catches the drift with probability 0.6388. A simulation with a million draws per hypothesis agrees. Its FPR is 0.0503, and its type II error is 0.3617.

| $\mu$ | $\alpha$ | Threshold $\Phi^{-1}(1 - \alpha)$ | $\beta = T(\alpha)$ | Power |
| --- | --- | --- | --- | --- |
| 1 | 0.05 | 1.6449 | 0.7405 | 0.2595 |
| 1 | $10^{-3}$ | 3.0902 | 0.9817 | 0.0183 |
| 2 | 0.05 | 1.6449 | 0.3612 | 0.6388 |
| 2 | $10^{-3}$ | 3.0902 | 0.8622 | 0.1378 |

: The Gaussian trade-off curve at two shifts and two levels. {#tbl-ht-gaussian}

If false alarms must be rare, at a rate of $10^{-3}$, the threshold moves out to 3.0902. The power drops to 0.1378 (@tbl-ht-gaussian). A strict threshold costs a lot of power when the shift is moderate.

You can also turn the question around and ask how many readings you need. Say you want power at least $1 - \beta'$ at level $\alpha$. The curve must then be at most $\beta'$ at that level. The curve is $\Phi$ evaluated at the threshold minus the shift. That value of $\Phi$ is at most $\beta'$ only when the threshold minus the shift is at most $-\Phi^{-1}(1 - \beta')$. The shift must therefore be at least $\Phi^{-1}(1 - \alpha) + \Phi^{-1}(1 - \beta')$. Substituting $\mu = \delta\sqrt n/\sigma$ and solving for $n$ gives
$$n \ge \left(\frac{\Phi^{-1}(1 - \alpha) + \Phi^{-1}(1 - \beta')}{\delta/\sigma}\right)^2.$$
Take a target power of 0.9 at a level of 0.01, with a drift of $0.25\sigma$. The quantile for the level is 2.3263. The quantile for the power is 1.2816. The bound then comes to 208.3, so you need 209 readings.

## The Laplace trade-off curve {#sec-ht-laplace}

Replace the normal by a Laplace distribution, which has exponential tails. The Laplace distribution $\mathrm{Lap}(m, b)$ has the centre $m$ and the scale $b$. Its density is proportional to $e^{-|x - m|/b}$ ([Probability distributions](01-distributions.md)). Take $P = \mathrm{Lap}(0, b)$ against $Q = \mathrm{Lap}(s, b)$, a shift by $s$. Write $\kappa = s/b$ for the shift measured in units of the scale.

The key fact is that the likelihood ratio of this pair is capped. Moving the centre by $s$ changes the distance from any point to the centre by at most $s$. The exponent therefore changes by at most $\kappa$, and the shifted density is within a factor $e^\kappa$ of the original everywhere. The likelihood ratio stays between $e^{-\kappa}$ and $e^{\kappa}$. A shifted normal has no such cap, as the previous section showed.

The cap shapes the whole curve, because the slope of the curve is minus the likelihood ratio at the threshold. Above the centre $s$ of $Q$ the ratio sits at its cap $e^\kappa$, and below the centre 0 of $P$ it sits at its floor $e^{-\kappa}$. The two ends of the curve are therefore straight lines, with slopes $-e^{\kappa}$ and $-e^{-\kappa}$. Between the two centres the ratio rises with $x$, and the curve bends. The full edition derives the formula for that middle piece from the Laplace tails. The full curve is
$$T(\alpha) = \begin{cases} 1 - e^{\kappa}\alpha, & \alpha < \tfrac12 e^{-\kappa}, \\[2pt] \dfrac{e^{-\kappa}}{4\alpha}, & \tfrac12 e^{-\kappa} \le \alpha \le \tfrac12, \\[6pt] e^{-\kappa}(1 - \alpha), & \alpha > \tfrac12. \end{cases}$$ {#eq-ht-laplace-tradeoff}
The three pieces correspond to a threshold above the centre of $Q$, between the two centres, and below the centre of $P$.

The first piece is the one that matters most. Since the ratio is never above $e^\kappa$, no test can buy power faster than $e^\kappa$ per unit of FPR. No test at level $\alpha$ can therefore have power above $e^\kappa\alpha$. For small budgets, the best test reaches exactly that power.

As an example, suppose a regression model's residuals are roughly Laplace with scale $b$. You suspect that a batch of inputs carries an offset equal to $b$, so $\kappa = 1$. The first breakpoint of the curve is then $\tfrac12 e^{-1}$, which is 0.1839. A level of 0.05 lies below it, on the first piece, and there the best power is 0.1359. A level of 0.25 lies on the middle piece. There the type II error is $e^{-1}$, which is 0.3679. A simulation of the threshold test with a million draws per hypothesis gives 0.3685.

To compare the two shapes fairly, match a Laplace pair and a normal pair by their best advantage. That is the largest gap between TPR and FPR that any test achieves (@sec-ht-tv). At $\kappa = 1$ the Laplace pair's best advantage is 0.3935. A normal pair with $\mu = 1.0301$ has the same. In the middle of the curve the two nearly coincide. At small levels they do not (@tbl-ht-laplace-vs-gaussian).

| $\alpha$ | Laplace power, $\kappa = 1$ | Normal power, $\mu = 1.0301$ |
| --- | --- | --- |
| $10^{-3}$ | $2.718\times10^{-3}$ | $1.969\times10^{-2}$ |
| $10^{-6}$ | $2.718\times10^{-6}$ | $9.829\times10^{-5}$ |

: Best power at small false positive rates for a Laplace pair and a normal pair with the same best advantage. {#tbl-ht-laplace-vs-gaussian}

At a level of $10^{-6}$, the best test on the normal pair has 36 times the power of the best test on the Laplace pair. The unbounded normal likelihood ratio lets a test find rare observations that are overwhelmingly more likely under $Q$. The capped Laplace ratio rules that out (@fig-ht-laplace-tradeoff). Two problems that look equally hard on average can therefore differ enormously when false alarms must be rare. Plan for the level you will actually use.

![Left: the log likelihood ratio of $\mathrm{Lap}(1, 1)$ against $\mathrm{Lap}(0, 1)$ is clipped at $\pm1$, while the normal one with the same best advantage is a straight line. Right: best power on log axes. The Laplace curve follows $e\alpha$ for small $\alpha$; the normal curve rises far above it.](figures/hypothesis_testing_laplace_tradeoff.svg){#fig-ht-laplace-tradeoff}

## Best advantage and total variation {#sec-ht-tv}

Sometimes you want a single number for how distinguishable two distributions are. A natural choice is the **advantage** of a test, its TPR minus its FPR. It says how much more often the test fires under $Q$ than under $P$. The coin-flip test fires equally often under both, so its advantage is 0. In medicine, the advantage at a threshold is called Youden's index.

The advantage is a sum over outcomes of $\phi(x)$ times $q(x) - p(x)$. Each term can be made as large as possible on its own. The best test therefore fires exactly on the outcomes where $q$ exceeds $p$, which is the LR test with threshold 1. Its advantage is the total amount by which $q$ exceeds $p$. That total is the **total variation (TV)** distance of [Divergences](03-divergences.md). It equals half the sum of $|q(x) - p(x)|$ over all outcomes.

Take the spam example. The gaps between the two rows of @tbl-ht-spam are 0.3, 0.1, 0.1 and 0.3. Half their sum, the TV distance, is 0.40. Spam is more likely than ham at the counts 2 and 3, so the best test rejects those two counts. Its FPR is 0.3, and its type II error is also 0.3. Its advantage is therefore 0.4, as it should be.

For the normal pair, the best test rejects above $\mu/2$, halfway between the means, where the two bell curves cross. Its advantage is $2\Phi(\mu/2) - 1$. At $\mu = 1$ that is 0.3829. For the Laplace pair the best advantage is $1 - e^{-\kappa/2}$. At $\kappa = 1$ that is 0.3935.

On the trade-off curve, the advantage at level $\alpha$ is the vertical gap between the line $1 - \alpha$ and the curve. The gap is largest where the curve's slope is $-1$, which is where the LR threshold is 1 (@fig-ht-tv-advantage).

![The Gaussian trade-off curve at $\mu = 1$, the diagonal $1 - \alpha$, and the advantage $1 - \alpha - T(\alpha)$ (dashed). The red segment is the largest gap, at $\alpha = 0.3085$, and its length is the total variation 0.3829.](figures/hypothesis_testing_tv_advantage.svg){#fig-ht-tv-advantage}

The TV distance answers one question well. It tells you how small the sum of the two error rates can be, when one error counts as much as the other. That question picks out a single point of the curve, the one where the threshold is 1. It says nothing about the ends of the curve. The Laplace and normal pairs of @tbl-ht-laplace-vs-gaussian share the same TV distance, yet their powers at the smallest level differ by a factor of 36. If false alarms must be rare, the TV distance is the wrong summary, and the whole curve is the right one.

## ROC summaries: why AUC hides the low-FPR regime {#sec-ht-auc}

Real detectors are compared on held-out data by their ROC curves. The negatives in that data are drawn under $H_0$, and the positives under $H_1$. The usual summary of an ROC curve is the **area under the curve (AUC)**. In many applications only a small corner of the curve matters. A fraud team can review only a tiny fraction of transactions, and a spam filter must almost never discard real mail. Two detectors with the same AUC can differ by a factor of ten in that corner.

The AUC has a direct meaning. Write $S_-$ for the score of a random negative and $S_+$ for the score of a random positive, drawn independently. The AUC is the probability that $S_+$ is above $S_-$. In words, it is the chance that the detector ranks a random positive above a random negative.

The reason is easiest to see on a test set, by sweeping the threshold down from the top. The FPR rises by the same small step each time the threshold passes the score of a negative. At that moment, the TPR is the share of positives that score above that negative. The area under the curve adds up these TPRs, each times the same step. That is their average over the negatives, which is the chance that a positive beats a random negative. This reading assumes continuous scores, so that a positive and a negative never tie.

A common model, called the binormal model, has normal scores in both classes. Negatives score $\mathcal N(0, 1)$. Positives score $\mathcal N(m, \sigma^2)$, with the mean $m$ and the standard deviation $\sigma$. The difference of the two scores is then normal, with mean $m$ and variance $1 + \sigma^2$. The AUC is the chance that this difference is positive:
$$\mathrm{AUC} = \Phi\big(m/\sqrt{1 + \sigma^2}\big).$$
The ROC curve follows from setting the threshold from the FPR, as in @sec-ht-gaussian:
$$\mathrm{TPR}(\mathrm{FPR}) = 1 - \Phi\Big(\frac{\Phi^{-1}(1 - \mathrm{FPR}) - m}{\sigma}\Big).$$

Now compare two detectors. Detector A has positives that score $\mathcal N(1, 1)$. Detector B has positives that score $\mathcal N(1.5811, 2^2)$, which are more spread out, with the mean chosen to give the same AUC. Both have an AUC of 0.7602. @tbl-ht-auc shows their TPRs at two FPRs.

| Detector | AUC (formula / simulated) | TPR at FPR $= 0.1$ | TPR at FPR $= 10^{-3}$ |
| --- | --- | --- | --- |
| A, $\mathcal N(1, 1)$ | 0.7602 / 0.7605 | 0.3891 | 0.0183 |
| B, $\mathcal N(1.5811, 2^2)$ | 0.7602 / 0.7601 | 0.5595 | 0.2253 |

: Two detectors with normal scores in both classes and equal AUC. The simulated AUC counts correctly ranked pairs among 1,000,000. {#tbl-ht-auc}

At an FPR of $10^{-3}$, detector B catches 12 times as many positives as detector A. B's positives are more spread out, so a few of them score far above every negative. Those are exactly the cases a strict threshold catches. On linear axes the two ROC curves cross and look comparable. On log axes the gap at low FPR is obvious (@fig-ht-auc-low-fpr).

![ROC curves of two detectors with the same AUC of 0.760. Left, linear axes: the curves cross and look comparable. Right, log axes: at an FPR of $10^{-3}$ detector B has 12 times the TPR of detector A.](figures/hypothesis_testing_auc_low_fpr.svg){#fig-ht-auc-low-fpr}

The AUC hides this because it weighs all FPRs equally. The region of FPRs below $10^{-3}$ is a tenth of a percent of the horizontal axis, so it barely changes the area. Most of the AUC comes from FPRs nobody would operate at.

An extreme case makes the point. Picture a detector whose ROC curve jumps to a TPR of 0.05 at an FPR of 0 and then follows the diagonal. Its AUC is only about 0.525. Yet it catches 5% of positives with no false alarms. Report the TPR at a fixed low FPR chosen by the application, and plot ROC curves on log–log axes, where that corner is visible.

For detector A, thresholding the score is already the best test. Its positives and negatives have the same spread, so its likelihood ratio grows with the score, as in @sec-ht-gaussian. Detector B is different. Its positives are more spread out than its negatives, since their standard deviation is 2 against 1. Its log likelihood ratio therefore contains a positive term in the square of the score, and the ratio is large at both ends. The best test for B would also flag very *low* scores. By the Neyman–Pearson lemma, B's best ROC curve is at least as good as the one plotted.

## The Bayesian view {#sec-ht-bayes}

The Neyman–Pearson lemma uses no prior. But often you know a base rate, such as 1% of patients being ill or one transaction in a thousand being fraud. Then the useful question is how likely $H_1$ is, given the observation. The likelihood ratio answers it with one multiplication.

Write $\pi$ for the base rate, the probability of $H_1$ before seeing anything. The odds of $H_1$ are its probability divided by the probability of $H_0$. Bayes' rule says that the odds after seeing $x$ are the odds before, multiplied by the likelihood ratio:
$$\frac{\Pr[H_1 \mid x]}{\Pr[H_0 \mid x]} = \frac{\pi}{1 - \pi} \times \Lambda(x).$$ {#eq-ht-posterior-odds}
To see why, note that the chance that $H_1$ holds and you see $x$ is $\pi\,q(x)$. The chance that $H_0$ holds and you see $x$ is $(1 - \pi)\,p(x)$. Dividing each by the total chance of seeing $x$ gives the two posterior probabilities. That total cancels in their ratio.

The rule also tells you how to decide. To make as few mistakes as possible, say $H_1$ whenever its posterior odds exceed 1. That happens exactly when $\Lambda(x)$ is above $(1 - \pi)/\pi$. The decision with the fewest errors is therefore again an LR test, with its threshold set by the base rate.

Unequal costs change only the threshold. If a miss costs $c$ times as much as a false alarm, the threshold becomes $(1 - \pi)/(c\pi)$. The test then moves along the same trade-off curve.

When both hypotheses are equally likely beforehand, the fewest errors means the smallest sum of the two error rates. Minimizing that sum is the same as maximizing the advantage. Take one minus the TV distance, and halve the result. That is the smallest possible probability of error. For the normal pair at $\mu = 1$, it is 0.3085.

The same rule applies to any evidence, not only to the raw observation. If all you know is that the test fired, the likelihood ratio of that event is the chance of firing under $H_1$ divided by the chance under $H_0$. That is the TPR divided by the FPR.

Take the biomarker test with $\mu = 2$ in a population where 1% are ill. At an FPR of 0.05 the power is 0.6388. A positive result is then 12.8 times as likely in an ill patient as in a healthy one. Even so, a positive patient is ill with probability only 0.114, so nine in ten positives are false alarms. At an FPR of $10^{-3}$ the power is 0.1378. A positive result is then 137.8 times as likely in an ill patient. A positive patient is ill with probability 0.582 (@fig-ht-posterior).

![Posterior probability of illness after a positive result, as a function of the prior, for the $\mu = 2$ biomarker test at two thresholds. At 1% prevalence (dashed line) the stricter test gives 0.582 against 0.114.](figures/hypothesis_testing_posterior.svg){#fig-ht-posterior}

The stricter threshold catches fewer ill patients, but it makes each positive result far more informative. This is a second reason that behaviour at low FPR matters. When positives are rare, the FPR, not the power, decides whether an alarm is worth acting on.

The test with the fewest errors at this base rate is stricter still. It rejects when the ratio exceeds 99. For this pair, that means a reading above 3.2976. Its FPR is $4.88\times10^{-4}$. Its power is only 0.0972. When both kinds of error count equally and the disease is rare, it is cheapest to rarely say "ill".

## Measuring rates from finite trials {#sec-ht-estimation}

Everything so far assumed that you know $P$ and $Q$. In practice you measure a detector's FPR on test data, where you count the false positives among the negatives. Write $n$ for the number of negatives and $k$ for the number of false positives among them. If each negative is falsely flagged independently, with the same probability $\alpha$, then $k$ follows a binomial distribution ([Probability distributions](01-distributions.md)). The intervals below are computed from that distribution, so they are only as valid as this model.

The measured rate $k/n$ is uncertain, and at low rates the uncertainty is large compared with the rate itself. When the rate is small, a count of $k$ events fluctuates by about $\sqrt k$. Its relative uncertainty is then about $1/\sqrt k$. The relative precision of a measured rate therefore depends mainly on how many events you see, not on $n$ alone.

The **Clopper–Pearson interval** turns the observed count into a range of plausible true rates. It asks which true rates would make the observed count unsurprising. A true rate far above $k/n$ would make a count as small as $k$ very unlikely. The upper end $u$ is therefore the rate at which seeing $k$ or fewer events has a small probability. Likewise, the lower end $l$ is the rate at which seeing $k$ or more events has that small probability.

For a two-sided interval at confidence $1 - a$, each end gets half of the allowed failure probability $a$. The two ends solve
$$\Pr[\mathrm{Bin}(n, l) \ge k] = \frac a2, \qquad \Pr[\mathrm{Bin}(n, u) \le k] = \frac a2.$$
By convention, the lower end is 0 when no events are seen, and the upper end is 1 when every trial is an event. The interval contains the true rate with probability at least $1 - a$.

The reason is that each end can miss on only one side. The true rate lies above the upper end only when the observed count was unusually small for it. By the choice of $u$, that happens with probability at most $a/2$. The same holds for the lower end. The union bound, which says that the chance of either of two events is at most the sum of their chances, then gives at most $a$ in total.

The interval is called exact because it uses the binomial distribution itself, not an approximation. It is also conservative, because its coverage is at least $1 - a$ and often more. A simulation checks this, with a true FPR of $10^{-3}$ and 10,000 negatives. Over 2,000 runs, the 95% interval covered the true rate 97.20% of the time.

Often you need only an upper bound, for example to claim that an FPR is below a target. The upper end on its own misses with probability at most $a/2$. The upper end of a two-sided interval at confidence $1 - a$ is therefore a one-sided upper bound at confidence $1 - a/2$. A two-sided 90% interval gives each end a failure probability of 0.05. Its upper end is therefore a one-sided 95% upper bound. You compute it by solving $\Pr[\mathrm{Bin}(n, u) \le k] = 0.05$. For 1 false positive in 1,000 negatives, this bound is 0.0047 (the code below prints it).

With no events at all, the chance of seeing none at rate $u$ is $(1 - u)^n$. The one-sided 95% upper bound solves $(1 - u)^n = 0.05$. For a small rate, $(1 - u)^n$ is close to $e^{-nu}$, so the bound is about $\ln(20)/n$. Since $\ln 20$ is about 3, the bound is about $3/n$, a shortcut called the **rule of three**. With 3,000 negatives and no false positives, the bound is $9.981\times10^{-4}$. Turned around, certifying an FPR below $10^{-3}$ at 95% confidence takes 2,995 negatives with no false positive at all.

Here is what this means for the monitor with $\mu = 2$, run at its $10^{-3}$ threshold. Its true FPR is $10^{-3}$, and its true TPR is 0.1378. Test it on 10,000 negatives and 10,000 positives. @tbl-ht-one-run shows one simulated run.

| Quantity | Count | Estimate | 95% Clopper–Pearson interval |
| --- | --- | --- | --- |
| False positives | 10 | 0.0010 | $[0.00048, 0.00184]$ |
| True positives | 1,350 | 0.1350 | $[0.1284, 0.1419]$ |

: One simulated evaluation of the $\mu = 2$ test at its $10^{-3}$ threshold, with 10,000 examples per class. {#tbl-ht-one-run}

The TPR is pinned down to about $\pm 0.007$. The FPR is not, because it rests on only 10 events. The data are consistent with a true FPR anywhere from about half to almost twice the target. With 100,000 negatives you would expect about 100 false positives. The interval would then be $[0.00081, 0.00122]$, about 20% on either side. This gives a rule of thumb. You need on the order of $100/\alpha$ negatives to measure an FPR of $\alpha$ to within a factor of about 1.2. @fig-ht-rate-ci shows how slowly the interval tightens as the number of negatives grows.

![95% Clopper–Pearson intervals for an observed false-positive rate of exactly $10^{-3}$, as the number of negatives grows from $10^3$ to $10^6$, with the upper end of the two-sided Hoeffding interval for comparison.](figures/hypothesis_testing_rate_ci.svg){#fig-ht-rate-ci}

Two other common intervals do worse on the same 10 false positives. The **Wald interval** uses a normal approximation to the binomial. Here it gives $[0.00038, 0.00162]$, which is too narrow. It is also symmetric, while a count of rare events is skewed to the right. And with no events it collapses to $[0, 0]$, claiming certainty from no data.

The Hoeffding interval comes from Hoeffding's inequality ([Concentration inequalities](02-concentration.md)). That inequality uses only the fact that each outcome lies between 0 and 1. It therefore assumes the largest variance such an outcome can have, a quarter, while the true variance of a rare false positive is tiny. Here its half-width is 0.0136, more than 13 times the rate being measured.

One last trap is that the threshold is often estimated too. Suppose you choose the threshold on the same data so that the measured FPR is $10^{-3}$. The choice favours thresholds whose measured FPR happens to be low by chance. The chosen threshold therefore looks better on that data than it is, and its true FPR is as uncertain as the interval above. Choose thresholds on one split of the data, and measure their rates on another.

## Code {#sec-ht-code}

The normal trade-off curve and the Clopper–Pearson interval need only the standard library. The quantile is found by bisection on the upper tail, which keeps full precision when $\alpha$ is tiny. Computing $\Phi^{-1}(1 - \alpha)$ directly would round $1 - \alpha$ to a number very close to 1 and lose the digits that matter.

```python
from collections.abc import Callable
from math import erfc, exp, lgamma, log, log1p, sqrt


def normal_cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def upper_quantile(a: float) -> float:
    """The point a standard normal exceeds with probability a, found by bisection on the tail."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 0.5 * erfc(mid / sqrt(2)) > a else (lo, mid)
    return lo


def gaussian_power(alpha: float, mu: float) -> float:
    """Best power at false positive rate alpha between N(0, 1) and N(mu, 1)."""
    return 1 - normal_cdf(upper_quantile(alpha) - mu)


def binom_cdf(k: int, n: int, p: float) -> float:
    """Pr[Binomial(n, p) <= k], summed in log space."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    total = 0.0
    for i in range(k + 1):
        log_pmf = lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * log(p) + (n - i) * log1p(-p)
        total += exp(log_pmf)
    return total


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided interval for a rate after k successes in n trials."""
    tail = (1 - confidence) / 2  # a/2 in the text

    def solve(f: Callable[[float], float], target: float) -> float:
        """The rate p at which f, a binomial CDF that falls as p grows, equals target."""
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
        return lo

    # Lower end: Pr[Bin(n, l) >= k] = tail, that is Pr[Bin(n, l) <= k - 1] = 1 - tail.
    lower = 0.0 if k == 0 else solve(lambda p: binom_cdf(k - 1, n, p), 1 - tail)
    # Upper end: Pr[Bin(n, u) <= k] = tail.
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p), tail)
    return lower, upper


def upper_bound(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided upper bound: the upper end of the two-sided interval at confidence 2 * confidence - 1."""
    return clopper_pearson(k, n, 2 * confidence - 1)[1]


print(f"{gaussian_power(0.05, 2.0):.4f} {gaussian_power(1e-3, 2.0):.4f}")  # 0.6388 0.1378
low, high = clopper_pearson(10, 10_000)
print(f"[{low:.5f}, {high:.5f}]")  # [0.00048, 0.00184]
print(f"{upper_bound(1, 1_000):.4f}")  # 0.0047
print(f"{upper_bound(0, 3_000):.3e}")  # 9.981e-04
```

The last two lines are one-sided 95% upper bounds. The first is for 1 false positive in 1,000 negatives. The second is for no false positives in 3,000, close to the rule of three's $3/n$.

## Cheat sheet

| Result | Statement |
| --- | --- |
| Errors of a test $\phi$ | FPR $\alpha$ is the average of $\phi$ under $P$; power $1 - \beta$ (TPR) is its average under $Q$ |
| Neyman–Pearson | The LR test, rejecting where $q > \eta p$ and randomizing at ties, is most powerful at its level |
| Trade-off curve $T(\alpha)$ | Best $\beta$ at level $\alpha$; non-increasing, at most $1 - \alpha$, convex, continuous; slope $-\Lambda$ at the threshold |
| Normal shift $\mu$ | $T(\alpha) = \Phi(\Phi^{-1}(1 - \alpha) - \mu)$; $n$ averaged readings: $\mu = \delta\sqrt n/\sigma$ |
| Laplace shift $\kappa = s/b$ | @eq-ht-laplace-tradeoff; power at most $e^\kappa\alpha$ |
| Best advantage | $\max(\mathrm{TPR} - \mathrm{FPR}) = \mathrm{TV}$; normal $2\Phi(\mu/2) - 1$, Laplace $1 - e^{-\kappa/2}$ |
| AUC | $\Pr[S_+ > S_-]$; binormal $\Phi(m/\sqrt{1 + \sigma^2})$ |
| Bayes | Posterior odds $=$ prior odds $\times\,\Lambda$; an alarm has $\Lambda = \mathrm{TPR}/\mathrm{FPR}$ |
| Clopper–Pearson | $\Pr[\mathrm{Bin}(n, l) \ge k] = \Pr[\mathrm{Bin}(n, u) \le k] = a/2$ |
| One-sided upper bound | 95% one-sided $=$ upper end of the two-sided 90% interval; zero events: $u \approx 3/n$ |

: The results of this chapter. {#tbl-ht-cheat-sheet}

**Rules of thumb**

- To find the best test, compute the likelihood ratio and threshold it. If the ratio grows with a statistic, threshold the statistic.
- Compare detectors by their TPR at a fixed low FPR, on log axes. AUC and TV hide that regime.
- A capped likelihood ratio caps power at small levels: power is at most the cap times $\alpha$.
- To measure a rate near $\alpha$, plan for about $100/\alpha$ trials, and report a Clopper–Pearson interval.

## Further reading

- Casella & Berger, *Statistical Inference* (2nd ed., 2002), §8.3: the Neyman–Pearson lemma and the Karlin–Rubin theorem for one-sided composite tests.
- Cover & Thomas, *Elements of Information Theory* (2nd ed., 2006), §11.7–11.9: the Chernoff–Stein lemma, on how fast errors fall with many observations.
- Fawcett, "An introduction to ROC analysis" (Pattern Recognition Letters, 2006): ROC curves, AUC as a ranking probability, and convex hulls.
- Brown, Cai & DasGupta, "Interval Estimation for a Binomial Proportion" (Statistical Science, 2001): why the Wald interval fails and what to use instead.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/04-hypothesis-testing.md), except two that it cites: the Chernoff–Stein lemma and the exception for one-sided composite tests (Karlin–Rubin).
