# How much noise to add

Every differentially private release in this book answers the same practical question. You have a number, or a list of numbers, computed from people's data, and you want to publish it with random noise added. How much noise is enough? This appendix answers that question from scratch for the mechanisms the book uses most. They are Laplace noise for pure differential privacy (DP), Gaussian noise for approximate DP, which allows a small failure probability, randomized response for a single bit, and the exponential mechanism for choosing among candidates. It assumes only that you know what a probability density, an average, a logarithm and an exponential are. Everything else, including the definition of differential privacy itself, is built up on the page.

The results sit together because they all rest on one quantity, the **privacy loss**. It measures how much more likely an output is when one person's record is in the data than when it is changed. Each mechanism is calibrated by bounding the privacy loss, everywhere for pure DP and except on outputs of small total probability for approximate DP. The chapters derive the same formulas with more general tools. @tbl-apx-cal-map shows where each one is stated and used, and its symbols are defined in the sections it points to.

| Section of this appendix | What it derives | Where the chapters state or use it |
| --- | --- | --- |
| @sec-apx-cal-dp | The definition, the privacy loss and the exact $\delta$ | @sec-dpdef-neighbours, @sec-dpdef-definition, @sec-dpdef-divergences |
| @sec-apx-cal-sensitivity | Sensitivity of counts, sums, means and histograms | @sec-lg-sensitivity |
| @sec-apx-cal-laplace | Laplace noise of scale $\Delta_1/\varepsilon$ gives $\varepsilon$-DP | @sec-lg-laplace |
| @sec-apx-cal-rr | Randomized response and its unbiased estimate | @sec-dpdef-rr |
| @sec-apx-cal-exponential | The exponential mechanism and its utility | @sec-sel-exp, @sec-sel-utility |
| @sec-apx-cal-gauss-classic | The classic Gaussian formula $\sigma = \Delta_2\sqrt{2\ln(1.25/\delta)}/\varepsilon$ | @sec-lg-gauss-classic |
| @sec-apx-cal-gauss-exact | The exact Gaussian $\delta(\varepsilon)$ and the smallest $\sigma$ | @sec-lg-analytic, @sec-lg-high-dim |
| @sec-apx-cal-tradeoffs | Computing $\delta$ in code, and how $\varepsilon$, $\delta$, $\sigma$ and $\Delta_2$ trade off | @sec-lg-analytic, @sec-epsc-delta |

: The calibration formulas derived in this appendix and the chapter sections that state or use them. {#tbl-apx-cal-map}

## What differential privacy promises, and the privacy loss {#sec-apx-cal-dp}

### The question

Before you can ask how much noise is enough, you need to know what "enough" means. What exactly must the noisy output satisfy, and how do you check it for one mechanism?

### The answer

A dataset is a collection of records, one per person. Two datasets $D$ and $D'$ are **neighbours** if they differ in one person's record. The book uses two versions of this. Under **add/remove**, $D'$ is $D$ with one record added or removed, so the two have different numbers of records. The size $n$ is then itself protected, because the output must not reveal which size is true. Under **replace-one**, $D'$ is $D$ with one record changed, so $n$ is the same on both and treated as public. This appendix uses add/remove unless it says otherwise, as the chapters do for counts and histograms.

A **mechanism** $M$ is a randomized algorithm: run on a dataset $D$, it returns a random output $M(D)$. Write $p_D(y)$ for the probability (or density) that $M(D)$ outputs $y$, and $p_{D'}(y)$ for the same on the neighbour. An **event** $S$ is any set of outputs, such as "the released count is above 1240". The privacy parameter $\varepsilon \ge 0$ says how much one record may change the probability of an event: by at most a factor $e^\varepsilon$, so a small $\varepsilon$ means a factor close to 1. Formally, $M$ is **$\varepsilon$-differentially private** ($\varepsilon$-DP) if for every pair of neighbours and every event $S$,
$$\Pr[M(D) \in S] \le e^\varepsilon\,\Pr[M(D') \in S].$$

The weaker version allows a small slack $\delta$ on top of the factor. A mechanism is **$(\varepsilon, \delta)$-differentially private** if, for every pair of neighbours and every event $S$,
$$\Pr[M(D) \in S] \le e^\varepsilon\,\Pr[M(D') \in S] + \delta.$$ {#eq-apx-cal-def}
Read it as "the factor $e^\varepsilon$ holds, except for a $\delta$ of probability". Both definitions must hold with $D$ and $D'$ swapped too, since a neighbour of $D'$ is $D$.

The quantity that decides both is the **privacy loss** of an output $y$. It is the log of how much more likely $y$ is on $D$ than on $D'$:
$$L(y) = \ln\frac{p_D(y)}{p_{D'}(y)}.$$ {#eq-apx-cal-loss}
A loss of 0 means $y$ is equally likely on both datasets and says nothing about which one was used. A loss of $\varepsilon$ means $y$ is $e^\varepsilon$ times more likely on $D$. Two facts, proved below, turn the definitions into statements about $L$. First, if $L(y) \le \varepsilon$ for every output $y$ and every ordered pair of neighbours, the mechanism is $\varepsilon$-DP. Second, call the outputs where the loss exceeds $\varepsilon$ the **bad set**, $B = \{y : L(y) > \varepsilon\}$. The smallest $\delta$ that works for one pair at a given $\varepsilon$ is the probability of the bad set on $D$, minus $e^\varepsilon$ times its probability on $D'$:
$$\delta(\varepsilon) = \Pr[M(D) \in B] - e^\varepsilon\,\Pr[M(D') \in B].$$ {#eq-apx-cal-exact-delta}

### A worked example

Take a mechanism with three possible outputs, and the probabilities in @tbl-apx-cal-three on a dataset $D$ and its neighbour $D'$.

| Output $y$ | 1 | 2 | 3 |
| --- | --- | --- | --- |
| $p_D(y)$ | 0.5 | 0.3 | 0.2 |
| $p_{D'}(y)$ | 0.2 | 0.3 | 0.5 |
| Ratio $p_D(y)/p_{D'}(y)$ | 2.5 | 1 | 0.4 |
| Privacy loss $L(y)$ | 0.9163 | 0 | $-0.9163$ |
| $p_D(y) - 2\,p_{D'}(y)$ | 0.1 | $-0.3$ | $-0.8$ |

: A mechanism with three outputs on two neighbouring datasets. The last row uses $e^\varepsilon = 2$. {#tbl-apx-cal-three}

The largest ratio is 2.5, at output 1, so this pair is $\varepsilon$-DP with $\varepsilon = \ln 2.5 = 0.9163$ and with nothing smaller. The script that comes with this appendix (see the Code section at the end) checks all 7 non-empty events, and none has a larger ratio of probabilities than 2.5.

Now ask for a smaller $\varepsilon$, so that the factor $e^\varepsilon$ is 2 instead of 2.5. Output 1 then breaks the factor, because its ratio is 2.5. It is the only output in the bad set, because only there is the ratio above 2. So @eq-apx-cal-exact-delta gives $\delta = 0.5 - 2 \times 0.2 = 0.1$. As a check, the script computes $\Pr[M(D) \in S] - 2\Pr[M(D') \in S]$ for every event $S$. The largest value is indeed 0.1000, reached by the event that holds output 1 alone. @fig-apx-cal-excess draws it: $\delta$ is the red part of the first bar, the probability that sticks out above $e^\varepsilon$ times the neighbour's.

![The three-output example. Blue bars are the probabilities on $D$, orange bars are $e^\varepsilon = 2$ times the probabilities on $D'$. Only output 1 has more probability on $D$ than the orange bar allows, and the excess, in red, is the exact $\delta = 0.1$.](figures/cal_excess.svg){#fig-apx-cal-excess}

Note that $\delta$ is not the probability of the bad set, which here is 0.5. Most of that probability is allowed by the factor $e^\varepsilon$, and only the excess counts.

### Why it works

1. **A pointwise bound gives the event bound.** Suppose $L(y) \le \varepsilon$ at every output, that is $p_D(y) \le e^\varepsilon p_{D'}(y)$. The probability of an event is the sum of the probabilities of its outputs, so summing this inequality over the outputs in $S$ gives $\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S]$. For a density, the sum is an integral, and the same step works. So bounding the loss at every output is enough for $\varepsilon$-DP.
2. **The slack for one event is a sum over its outputs.** Moving the $e^\varepsilon$ term to the left in @eq-apx-cal-def, the event $S$ needs $\delta \ge \Pr[M(D) \in S] - e^\varepsilon\Pr[M(D') \in S]$. The right side is the sum, over the outputs $y$ in $S$, of $p_D(y) - e^\varepsilon p_{D'}(y)$.
3. **The worst event takes exactly the outputs with a positive term.** To make that sum as large as possible, include every output whose term is positive and leave out every other one. A term is positive exactly when $p_D(y) > e^\varepsilon p_{D'}(y)$, that is when $L(y) > \varepsilon$. So the worst event is the bad set $B$, and the smallest $\delta$ that covers every event is its sum, @eq-apx-cal-exact-delta.
4. **The probability of the bad set is always a valid $\delta$.** Dropping the term $e^\varepsilon\Pr[M(D') \in B]$, which is never negative, gives $\delta(\varepsilon) \le \Pr[M(D) \in B] = \Pr[L > \varepsilon]$, where the loss is computed at an output drawn from $M(D)$. This simpler bound is what the classic Gaussian formula of @sec-apx-cal-gauss-classic uses.

### Checking it

The script lists every event of the three-output example and confirms both numbers by brute force. The largest ratio over events is 2.5000, the same as the largest ratio over single outputs. The largest excess at $e^\varepsilon = 2$ is 0.1000, the same as the sum over the bad set. The probability of the bad set, 0.5000, is five times larger than the exact $\delta$, which shows how loose step 4 can be.

### Recipe and pitfalls

To check a mechanism, write down its privacy loss for a pair of neighbours. If the loss never exceeds $\varepsilon$, you have $\varepsilon$-DP. If it does, the exact $\delta$ is the excess mass of @eq-apx-cal-exact-delta, maximized over pairs of neighbours.

The most common mistake is to apply $\delta$ to each output instead of each event. "Every output has $p_D(y) \le e^\varepsilon p_{D'}(y) + \delta$" pays the slack once per output, and with many outputs that allows a mechanism that reveals everything (@sec-dpdef-definition gives the example). The second mistake is to forget the neighbouring relation. A replacement is a removal followed by an addition, so a guarantee under replace-one can cost up to twice the $\varepsilon$ of one under add/remove. Every guarantee should therefore say which relation it uses. @sec-dpdef-divergences writes the same exact $\delta$ as a divergence, and @sec-apx-conv-testing explains what $\varepsilon$ means for an attacker.

## Sensitivity: how much one person can move the answer {#sec-apx-cal-sensitivity}

### The question

Noise has to hide the effect of any one person. So before choosing the noise you need to know how far one person can move the answer. How do you compute that for the queries you actually run?

### The answer

Let $f$ be the query, a function that takes a dataset and returns $d$ numbers. The **sensitivity** of $f$ is the largest change that one person can cause, over all pairs of neighbours. Because the answer is a list of numbers, the change $v = f(D) - f(D')$ is a list too, and its size can be measured in two ways. The **$\ell_1$ norm** $\|v\|_1 = |v_1| + \dots + |v_d|$ adds up the absolute changes, and Laplace noise is calibrated to it. The **$\ell_2$ norm** $\|v\|_2 = \sqrt{v_1^2 + \dots + v_d^2}$ is the straight-line length of the change, and Gaussian noise is calibrated to it. So there are two sensitivities, one for each norm:
$$\Delta_1 = \max_{D \sim D'}\|f(D) - f(D')\|_1, \qquad \Delta_2 = \max_{D \sim D'}\|f(D) - f(D')\|_2.$$
@tbl-apx-cal-sens gives them for the standard queries. Values are **clipped** to a range $[0, C]$: any value above $C$ is replaced by $C$ before the query runs, so no record can contribute more than $C$.

| Query | Add/remove $\Delta_1$ | Add/remove $\Delta_2$ | Replace-one $\Delta_1$ | Replace-one $\Delta_2$ |
| --- | --- | --- | --- | --- |
| Count of records | 1 | 1 | 0 | 0 |
| Sum of values clipped to $[0, C]$ | $C$ | $C$ | $C$ | $C$ |
| Mean of $n$ clipped values, $n$ public | no useful bound | no useful bound | $C/n$ | $C/n$ |
| Histogram, each record in one bin | 1 | 1 | 2 | $\sqrt2$ |
| $d$ counts that one record can all change by 1 | $d$ | $\sqrt d$ | $d$ | $\sqrt d$ |

: Sensitivities of the standard queries under the two neighbouring relations. {#tbl-apx-cal-sens}

### A worked example

A study releases the mean age of $n = 1000$ patients, with ages clipped to $[0, 100]$, under replace-one. One patient's age can change from 0 to 100 at most, so the sum moves by at most $C = 100$. The mean is the sum divided by the fixed $n$, so it moves by at most $100/1000 = 0.1$ years. That is the sensitivity, $\Delta_1 = \Delta_2 = 0.1$, however old the patients actually are.

### Why it works

Each row asks what the single worst record can do.

1. **Count.** Adding or removing a person changes the count by exactly 1. Replacing a person leaves it unchanged, which is why under replace-one the count of all records is public anyway.
2. **Sum of clipped values.** Adding a person with value $x$ in $[0, C]$ changes the sum by $x$, at most $C$. Replacing $x$ by $x'$ changes it by $|x - x'|$, again at most $C$. Both bounds are reached with $x = C$ and $x' = 0$.
3. **Mean.** Under replace-one, the mean is the sum divided by the public $n$, so its sensitivity is the sum's divided by $n$, which is $C/n$. Under add/remove the divisor changes too, and the change depends on $n$, which is private. The script finds that for datasets of 1 to 3 values in $[0, 3]$ one record can move the mean by 1.5, from the mean of $\{0\}$ to the mean of $\{0, 3\}$. The usual fix is to release a noisy sum and a noisy count separately and divide them.
4. **Histogram.** Under add/remove the person lands in one bin, so one bin changes by 1. Both norms of the change are 1. Under replace-one one bin loses 1 and another gains 1, so the $\ell_1$ norm is $1 + 1 = 2$ and the $\ell_2$ norm is $\sqrt{1^2 + 1^2} = \sqrt2$.
5. **Counts that one person can all touch.** Suppose each person answers $d$ yes/no questions and you release the number of "yes" answers to each. One person who says yes to all $d$ moves every count by 1, so the $\ell_1$ norm of the change is $d$ and the $\ell_2$ norm is $\sqrt{d \cdot 1^2} = \sqrt d$.

The last row shows why the two norms matter. When one person spreads their effect over many coordinates, $\Delta_1$ grows like $d$ but $\Delta_2$ only like $\sqrt d$. Since the noise is scaled to the sensitivity, as the next sections show, this is why Gaussian noise wins for long vectors (@sec-lg-high-dim).

### Checking it

The script lists every dataset of 1 to 3 records and every neighbour under both relations, and takes the largest change. Values are in $\{0, 1, 2, 3\}$, so $C = 3$ and the histogram has 4 bins, and the vector query uses $d = 3$ questions. It reproduces every row of @tbl-apx-cal-sens. For example, the histogram has $\ell_1$ sensitivity 1 under add/remove and 2 under replace-one, with $\ell_2$ sensitivity 1.4142 under replace-one. The three counts have 3 and 1.7321. For the mean of exactly 3 records under replace-one it finds $3/3 = 1.0000$.

### Recipe and pitfalls

First fix the neighbouring relation, then ask what the worst single record can do to each coordinate, and add the changes up in the norm your noise needs. Compute the sensitivity before you look at the data, because it is a worst case over everything the domain allows, not over the data you hold.

Unbounded data has infinite sensitivity. If ages, incomes or gradients can be as large as anyone likes, one record can move a sum by any amount, and no finite noise hides it. Clip every value to a fixed range first, and calibrate to the clipping norm. A second pitfall is a person who contributes many records: if one person can add 5 ratings, a count of ratings has sensitivity 5 per person, not 1. @sec-apx-sgd-clipping applies the same reasoning to clipped gradients.

## Laplace noise: scale $\Delta_1/\varepsilon$ gives $\varepsilon$-DP {#sec-apx-cal-laplace}

### The question

You want to publish a count, or a few numbers, with pure $\varepsilon$-DP. How wide must the Laplace noise be, and how far will the published answer be from the truth?

### The answer

The **Laplace distribution** is a two-sided exponential, peaked at 0. With scale $b$, its typical size, it puts density $\frac{1}{2b}e^{-|z|/b}$ at each value $z$. The **Laplace mechanism** adds independent Laplace noise to each of the $d$ numbers, with the scale set to the $\ell_1$ sensitivity divided by $\varepsilon$:
$$M(D) = f(D) + (Z_1, \dots, Z_d), \qquad Z_i \sim \mathrm{Lap}(b), \qquad b = \frac{\Delta_1}{\varepsilon}.$$ {#eq-apx-cal-laplace}
It is $\varepsilon$-DP. For one number, the error is the noise itself, and it exceeds a threshold $\tau$ with probability exactly $e^{-\tau/b}$. So with probability 0.95 the error is at most $b\ln 20$.

### A worked example

A hospital has 1,234 patients with diabetes. On the neighbouring dataset $D'$ there is one more, so the two true counts are 1234 and 1235, and $\Delta_1 = 1$. With $\varepsilon = 0.5$, the scale is $b = 1/0.5 = 2$.

The privacy loss at an output $y$ is the log of the ratio of the two Laplace densities. The factors $\frac{1}{2b}$ cancel, and the log of a ratio of exponentials is the difference of the exponents, so
$$L(y) = \frac{|y - 1235| - |y - 1234|}{b}.$$
@tbl-apx-cal-laplace evaluates it at five outputs.

| Output $y$ | 1230 | 1234 | 1234.5 | 1235 | 1240 |
| --- | --- | --- | --- | --- | --- |
| $p_D(y)$ | 0.0338 | 0.2500 | 0.1947 | 0.1516 | 0.0124 |
| $p_{D'}(y)$ | 0.0205 | 0.1516 | 0.1947 | 0.2500 | 0.0205 |
| Ratio | 1.6487 | 1.6487 | 1.0000 | 0.6065 | 0.6065 |
| $L(y)$ | 0.5 | 0.5 | 0 | $-0.5$ | $-0.5$ |

: The Laplace mechanism for a count with $b = 2$: the two densities and the privacy loss at five outputs. {#tbl-apx-cal-laplace}

At $y = 1230$ the loss is $(5 - 4)/2 = 0.5$, and at $y = 1240$ it is $(5 - 6)/2 = -0.5$. Below both true counts the ratio is stuck at $e^{0.5} = 1.6487$, above both it is stuck at $e^{-0.5} = 0.6065$, and between them it moves from one to the other. It never goes further, however far out the output lands.

For accuracy, the 95% error bound is $b\ln 20 = 2 \times 2.9957 = 5.99$. So the published count is within about 6 of 1234 in 95% of releases. The bound scales with $1/\varepsilon$, so it is 29.96 at $\varepsilon = 0.1$, where the scale is 10, and 3.00 at $\varepsilon = 1$.

### Why it works

1. **Only the shift matters.** Write $a = f(D)$ and $a' = f(D')$ for the two true answers. For one number, the privacy loss at $y$ is $(|y - a'| - |y - a|)/b$, as in the example.
2. **Bound it with the triangle inequality.** The distance from $y$ to $a'$ is at most the distance from $y$ to $a$ plus the distance from $a$ to $a'$, so $|y - a'| - |y - a| \le |a - a'|$. In the example, with $y = 1230$, this says $5 - 4 \le 1$.
3. **The shift is at most the sensitivity.** So $L(y) \le |a - a'|/b \le \Delta_1/b$, and with $b = \Delta_1/\varepsilon$ that is $\varepsilon$. This holds at every output $y$.
4. **Several numbers.** With $d$ coordinates and independent noise, the density of the output is the product of the $d$ coordinate densities. The log of a product is a sum, so the privacy loss is the sum of the coordinate losses. Each is at most $|a_i - a'_i|/b$ by step 2, and their sum is at most $\|a - a'\|_1/b \le \Delta_1/b = \varepsilon$.
5. **Conclude.** The loss is at most $\varepsilon$ at every output, for every pair of neighbours in either order. So by step 1 of @sec-apx-cal-dp the mechanism is $\varepsilon$-DP.
6. **Accuracy.** The absolute error $|Z|$ has density $\frac{1}{b}e^{-z/b}$ for $z \ge 0$, twice the Laplace density on one side. Its integral from $\tau$ to infinity is $e^{-\tau/b}$. Setting $e^{-\tau/b} = 0.05$ and solving gives $\tau = b\ln 20$. The same density has average $\int_0^\infty z\frac1be^{-z/b}\,\mathrm dz = b$, so the typical error is $b$.

The bound of step 2 is reached, with equality, at every output below both true answers. So with a smaller scale $b$ the loss there would be $1/b > \varepsilon$, and those outputs would break the factor $e^\varepsilon$. For the count above, the event "the output is at most 1234" has probability 0.5 on $D$ and $\frac12e^{-1/b} = 0.3033$ on $D'$, a ratio of exactly $e^{0.5}$.

### Checking it

@fig-apx-cal-laplace-ratio draws the two densities and their ratio for the example. The ratio is flat at $e^{\varepsilon}$ on the left and at $e^{-\varepsilon}$ on the right, and it never leaves that band. @fig-apx-cal-loss draws the privacy loss itself. It is flat outside the two true answers and a straight line between them, and it stays between $-0.5$ and $0.5$. On a grid of 70,001 outputs from 1200 to 1270, the script finds the loss in exactly $[-0.5000, 0.5000]$.

![Top: the output densities of a count released with Laplace noise of scale 2, on a dataset with true count 1234 and on its neighbour with 1235. Bottom: their ratio on a log scale, which stays between $e^{-0.5}$ and $e^{0.5}$ at every output and reaches both ends.](figures/cal_laplace_ratio.svg){#fig-apx-cal-laplace-ratio}

![The privacy loss $L(y)$ of the same count against the released value, for Laplace noise of scale 2 and for Gaussian noise of standard deviation 2. The Laplace loss is flat outside the two true answers and never leaves $[-0.5, 0.5]$. The Gaussian loss is a straight line that leaves any band.](figures/cal_privacy_loss.svg){#fig-apx-cal-loss}

For several numbers, the script takes three yes/no counts that one person can all raise by 1, so $\Delta_1 = 3$ and $b = 3/0.5 = 6$. Over 100,000 random outputs, the largest privacy loss is 0.5000, the $\varepsilon$ that step 4 promises.

For accuracy, a million draws of $\mathrm{Lap}(2)$ give a mean absolute error of 2.0026 against the exact 2. Their 95th percentile is 5.9938 against $b\ln 20 = 5.9915$. The error exceeds 5.9915 in a fraction 0.0501 of draws.

### Recipe and pitfalls

1. Compute the $\ell_1$ sensitivity $\Delta_1$ under your neighbouring relation.
2. Set $b = \Delta_1/\varepsilon$ and add independent $\mathrm{Lap}(b)$ noise to every coordinate.
3. Quote the error: $b$ on average, and $b\ln 20$ with probability 0.95 for each coordinate.

The noise does not depend on the number of people, so a count of a million people and a count of ten get the same noise. Small groups are therefore expensive. Use fresh noise for every release, because reusing the same noise draw lets an analyst subtract it out. Releasing the same count $k$ times with fresh noise costs $k\varepsilon$, which @sec-apx-ccomp-basic derives. @sec-lg-laplace adds the bound on the largest error over $d$ coordinates.

## Randomized response: one bit, $\varepsilon$-DP {#sec-apx-cal-rr}

### The question

Each person holds one sensitive bit, such as "I have used a drug", and nobody is trusted to see the true bits. How often should each person tell the truth, and how do you estimate the fraction of ones from the noisy reports?

### The answer

Each person reports their true bit with probability $t$ and the flipped bit with probability $1 - t$, where
$$t = \frac{e^\varepsilon}{1 + e^\varepsilon}.$$
This is $\varepsilon$-DP under replace-one. If $\bar y$ is the fraction of reported ones among $n$ people, an unbiased estimate of the true fraction $\pi$ of ones is
$$\hat\pi = \frac{\bar y - (1 - t)}{2t - 1}, \qquad \operatorname{sd}(\hat\pi) = \frac{\sqrt{t(1 - t)}}{(2t - 1)\sqrt n}.$$

### A worked example

At $\varepsilon = 0.5$, $t = e^{0.5}/(1 + e^{0.5}) = 0.6225$. A true 1 is reported as 1 with probability 0.6225, and a true 0 is reported as 1 with probability 0.3775. Their ratio is $0.6225/0.3775 = 1.6487 = e^{0.5}$.

Survey $n = 1000$ people, of whom a fraction $\pi = 0.3$ have the bit. The expected fraction of reported ones is $0.3 \times 0.6225 + 0.7 \times 0.3775 = 0.4510$. The estimate subtracts $1 - t = 0.3775$ and divides by $2t - 1 = 0.2449$, which turns this back into 0.3. Its standard deviation is $\sqrt{0.6225 \times 0.3775}/(0.2449\sqrt{1000}) = 0.0626$.

### Why it works

1. **Privacy.** There are two inputs and two outputs. A report of 1 has probability $t$ if the bit is 1 and $1 - t$ if it is 0, so its privacy loss is $\ln(t/(1 - t))$. With $t = e^\varepsilon/(1 + e^\varepsilon)$, the ratio is $t/(1 - t) = e^\varepsilon$. A report of 0 has the inverse ratio, so $|L| \le \varepsilon$ at both outputs.
2. **Many people.** People flip their own coins independently, so the probability of all the reports is a product with one factor per person. Under replace-one, neighbours differ in one person's bit, so every other factor is the same on both datasets and cancels in the ratio. The loss is that one person's loss.
3. **The expected report.** A person with bit 1 reports 1 with probability $t$, and a person with bit 0 with probability $1 - t$. Averaging over the population gives $\mathbb E[\bar y] = \pi t + (1 - \pi)(1 - t) = (1 - t) + \pi(2t - 1)$.
4. **Unbias it.** Solving that equation for $\pi$ gives $\hat\pi$. It is a linear function of $\bar y$, so its average is the true $\pi$.
5. **The spread.** Each report is a 0/1 variable with variance $t(1 - t)$, whatever the true bit, and the reports are independent. So $\bar y$ has variance $t(1 - t)/n$. Dividing by $2t - 1$ divides the standard deviation by $2t - 1$.

### Checking it

Ten thousand simulated surveys give a mean estimate of 0.2996 and a standard deviation of 0.0629, against 0.3 and 0.0626 from the formulas.

### Recipe and pitfalls

Set $t = e^\varepsilon/(1 + e^\varepsilon)$, collect the reports, and compute $\hat\pi$. Expect an error of about $1/(\varepsilon\sqrt n)$ for small $\varepsilon$. The reason is that $e^\varepsilon \approx 1 + \varepsilon$ there, so $t \approx (1 + \varepsilon)/2$, which makes $2t - 1 \approx \varepsilon/2$ and $t(1 - t) \approx \frac14$.

The price of trusting nobody is large. If one trusted party could see the true bits, it could add $\mathrm{Lap}(1/\varepsilon)$ noise to their count instead, and the error on the fraction would fall like $1/n$, not like $1/\sqrt n$. The guarantee also holds only under replace-one. Under add/remove the neighbour has one report more or fewer, and the number of reports, which is published, shows the difference at once. @sec-dpdef-rr compares the two designs, and Warner's original survey design is the same mechanism.

## The exponential mechanism: choosing the best candidate {#sec-apx-cal-exponential}

### The question

Sometimes the output is not a number but a choice: the best model, the most common diagnosis, the winner of a vote. Adding noise to a name makes no sense. How do you choose privately, and how good is the choice?

### The answer

Fix a list of candidates $\mathcal R$ in advance, and write $|\mathcal R|$ for their number. For each candidate $r$, a score $u(D, r)$ says how good it is on dataset $D$, where higher is better. Let $\Delta u$ be the sensitivity of the score, the most that one person can change any one candidate's score.

The **exponential mechanism** gives each candidate a weight $\exp(\varepsilon u(D, r)/(2\Delta u))$, which grows with its score. It then picks a candidate with probability equal to its share of the total weight $W(D)$, the **normalizer**:
$$\Pr[M_E(D) = r] = \frac{\exp\big(\frac{\varepsilon\,u(D, r)}{2\Delta u}\big)}{W(D)}, \qquad W(D) = \sum_{s \in \mathcal R}\exp\Big(\frac{\varepsilon\,u(D, s)}{2\Delta u}\Big).$$
Dividing by the normalizer makes the probabilities add up to 1. The mechanism is $\varepsilon$-DP, and its choice is rarely far below the best score $\mathrm{OPT}(D)$. For every $t > 0$,
$$\Pr\Big[u(D, M_E(D)) \le \mathrm{OPT}(D) - \frac{2\Delta u}{\varepsilon}(\ln |\mathcal R| + t)\Big] \le e^{-t}.$$

### A worked example

Six people vote among options A, B, C and D. On $D$ the counts are $(3, 2, 1, 0)$, and on the neighbour $D'$ one more person votes for B, so the counts are $(3, 3, 1, 0)$. The score is the count, so $\Delta u = 1$. Take $\varepsilon = 1$, so each weight is $e^{u/2}$.

On $D$ the weights are $e^{1.5} = 4.4817$, $e^{1} = 2.7183$, $e^{0.5} = 1.6487$ and $e^0 = 1$, which sum to $W(D) = 9.8487$. So A is chosen with probability $4.4817/9.8487 = 0.4551$. On $D'$ B's weight rises to $e^{1.5}$ and the normalizer to $W(D') = 11.6121$. @tbl-apx-cal-em lists all four probabilities.

| Option | A | B | C | D |
| --- | --- | --- | --- | --- |
| $\Pr[M_E(D) = r]$ | 0.4551 | 0.2760 | 0.1674 | 0.1015 |
| $\Pr[M_E(D') = r]$ | 0.3859 | 0.3859 | 0.1420 | 0.0861 |
| Ratio | 1.1790 | 0.7151 | 1.1790 | 1.1790 |

: The exponential mechanism with $\varepsilon = 1$ on a vote and its neighbour with one more vote for B. {#tbl-apx-cal-em}

For A, C and D the score did not change, so the ratio is just the change in the normalizer, $11.6121/9.8487 = 1.1790$. For B, its own weight is also $e^{1/2}$ times larger on $D'$ than on $D$, so its ratio is $1.1790 \times e^{-1/2} = 1.1790 \times 0.6065 = 0.7151$. The largest privacy loss on this pair is $|\ln 0.7151| = 0.3353$, well below $\varepsilon = 1$.

### Why it works

**Privacy.** The probability of $r$ is a weight divided by a total, and one person can change both. So split the ratio of the probabilities on $D$ and $D'$ into two factors:
$$\frac{\Pr[M_E(D) = r]}{\Pr[M_E(D') = r]} = \exp\Big(\frac{\varepsilon\big(u(D, r) - u(D', r)\big)}{2\Delta u}\Big) \cdot \frac{W(D')}{W(D)}.$$

1. **The own-weight factor.** One person changes $r$'s score by at most $\Delta u$, so the exponent is at most $\varepsilon\Delta u/(2\Delta u) = \varepsilon/2$. The factor is at most $e^{\varepsilon/2}$.
2. **The normalizer factor.** Every candidate's score on $D'$ is at most $\Delta u$ above its score on $D$, so every term of $W(D')$ is at most $e^{\varepsilon/2}$ times the matching term of $W(D)$. Adding up the terms, $W(D') \le e^{\varepsilon/2}W(D)$.
3. **Multiply.** The ratio is at most $e^{\varepsilon/2} \cdot e^{\varepsilon/2} = e^\varepsilon$. The candidate and the pair were arbitrary, and the same argument works with $D$ and $D'$ swapped, so by step 1 of @sec-apx-cal-dp the mechanism is $\varepsilon$-DP.

The 2 in the exponent is there because both factors can be large at once. One person can raise $r$'s score and lower all the others, so each factor needs its own half of the budget.

**Utility.** Call a candidate poor if its score is at least $c$ below the best.

1. **Poor candidates are light.** Each has weight at most $e^{\varepsilon(\mathrm{OPT} - c)/(2\Delta u)}$, since the weight grows with the score.
2. **The normalizer is heavy.** It contains the weight of a best candidate, so $W(D) \ge e^{\varepsilon\,\mathrm{OPT}/(2\Delta u)}$.
3. **Divide.** There are at most $|\mathcal R|$ poor candidates, so together they have probability at most $|\mathcal R|\,e^{\varepsilon(\mathrm{OPT} - c)/(2\Delta u)}/e^{\varepsilon\,\mathrm{OPT}/(2\Delta u)} = |\mathcal R|\,e^{-\varepsilon c/(2\Delta u)}$.
4. **Solve for $c$.** Setting $|\mathcal R|\,e^{-\varepsilon c/(2\Delta u)} = e^{-t}$ and taking logs gives $c = \frac{2\Delta u}{\varepsilon}(\ln |\mathcal R| + t)$.

### Checking it

@fig-apx-cal-exponential shows the vote. On the right, each ratio is drawn as its two factors, and both stay below $e^{\varepsilon/2}$. Only B's score moved, so for A, C and D the own-weight factor is exactly 1.

![Left: the probability that the exponential mechanism with $\varepsilon = 1$ chooses each option, on a vote with counts (3, 2, 1, 0) and on its neighbour with one more vote for B. Right: each ratio of the two probabilities, split into the change of the option's own weight and the change of the normalizer. Each factor stays below $e^{\varepsilon/2}$, so the ratio stays below $e^\varepsilon$.](figures/cal_exponential.svg){#fig-apx-cal-exponential}

The utility bound is weak for so small an example. The probability of choosing C or D, at least $c = 2$ below A, is 0.2689. Step 3 with $|\mathcal R| = 4$ bounds it by $4e^{-1} = 1.4715$, which says nothing. It helps when there are many candidates. Take 50 models scored by their correct predictions, $900, 899, \dots, 851$, so that $\Delta u = 1$. With $\varepsilon = 1$ and $t = 3$, the bound says the gap exceeds $2(\ln 50 + 3) = 13.82$ with probability at most $e^{-3} = 0.0498$. The exact probability is $9.12\times10^{-4}$, much smaller, because most models are far below the best and much lighter than the bound assumes.

### Recipe and pitfalls

1. Fix the list of candidates before looking at the data, and choose a score $u(D, r)$.
2. Find the score's sensitivity $\Delta u$.
3. Sample $r$ with probability proportional to $\exp(\varepsilon u(D, r)/(2\Delta u))$.

The candidate list must not depend on the data, because the proof compares the same list on $D$ and $D'$. Watch numerical overflow too: subtract the largest score before taking exponentials, which changes no probability. When every score moves the same way under each change of one person, as counts under add/remove do, the two factors cannot both be large, and the 2 can be dropped (@sec-sel-exp). @sec-sel-utility adds the expected gap and the applications.

## Gaussian noise, the classic formula {#sec-apx-cal-gauss-classic}

### The question

Gaussian noise is calibrated to the $\ell_2$ sensitivity, which is much smaller than the $\ell_1$ sensitivity for long vectors such as gradients. But its privacy loss has no upper limit, so it can never give pure $\varepsilon$-DP. How much Gaussian noise gives $(\varepsilon, \delta)$-DP, and where does the textbook formula come from?

### The answer

Add independent normal noise with mean 0 and standard deviation $\sigma$ to each coordinate. For $0 < \varepsilon < 1$ and $0 < \delta < 1$, the **Gaussian mechanism** with
$$\sigma = \frac{\Delta_2\sqrt{2\ln(1.25/\delta)}}{\varepsilon}$$ {#eq-apx-cal-classic}
is $(\varepsilon, \delta)$-DP. The proof below needs $\delta \le 0.7448$, which every useful $\delta$ satisfies.

### A worked example

Release the diabetes count with $\Delta_2 = 1$, $\varepsilon = 0.5$ and $\delta = 10^{-5}$. Then $\ln(1.25/10^{-5}) = \ln 125{,}000 = 11.7361$, and its double has square root $c = \sqrt{23.4722} = 4.8448$. So $\sigma = 4.8448/0.5 = 9.6896$. The 95% error of normal noise is 1.96 standard deviations, here 18.99, against 5.99 for Laplace noise at the same $\varepsilon$. For a single count Laplace noise is better, and Gaussian noise pays off only for long vectors.

### Why it works

The plan is to find the privacy loss, which turns out to be a normal random variable. Then show that it exceeds $\varepsilon$ with probability at most $\delta$, and that this is enough. Write $\Phi(x)$ for the standard normal cumulative distribution function, the probability that a normal variable with mean 0 and standard deviation 1 is below $x$, and $\varphi(x) = e^{-x^2/2}/\sqrt{2\pi}$ for its density.

1. **The privacy loss is a straight line in the noise.** Take one number, with true answers $a$ on $D$ and $a + s$ on $D'$, where $s$ is the shift between them. An output from $M(D)$ is $y = a + z$, where $z$ is the noise. Its density on $D$ is proportional to $e^{-z^2/(2\sigma^2)}$ and on $D'$ to $e^{-(z - s)^2/(2\sigma^2)}$, because $y$ sits $z - s$ away from $D'$'s answer. The log of the ratio is the difference of the exponents:
$$L = \frac{(z - s)^2 - z^2}{2\sigma^2} = \frac{s^2 - 2sz}{2\sigma^2}.$$
2. **So the loss is normal.** The noise $z$ is normal with mean 0 and standard deviation $\sigma$, and $L$ is a constant minus a multiple of $z$. So $L$ is normal with mean $s^2/(2\sigma^2)$ and standard deviation $2s\sigma/(2\sigma^2) = s/\sigma$. Write $\mu = s/\sigma$ for the shift measured in noise standard deviations. Then $L$ has mean $\mu^2/2$ and standard deviation $\mu$. For a vector the same holds with $s = \|f(D) - f(D')\|_2$. The reason is that expanding the squared distances in the same way leaves only the noise's component along the direction of the shift, and that component is a single normal number with standard deviation $\sigma$.
3. **A small chance of a large loss is enough.** Let $B$ be the bad set, the outputs with $L > \varepsilon$, and suppose its probability on $D$ is at most $\delta$. Split any event $S$ into its part inside $B$ and its part outside. The part inside has probability at most $\delta$, because it is part of $B$. On the part outside, every output has $p_D(y) \le e^\varepsilon p_{D'}(y)$, so summing over it gives at most $e^\varepsilon\Pr[M(D') \in S]$. Adding the two parts gives
$$\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta,$$
which is @eq-apx-cal-def. The pair in the other order has the same loss distribution, by the symmetry of the normal curve, so it needs nothing new.
4. **The bad event is a normal tail.** Standardize the loss by subtracting its mean $\mu^2/2$ and dividing by its standard deviation $\mu$. The result is a standard normal variable, and $L > \varepsilon$ exactly when it exceeds the point $t = (\varepsilon - \mu^2/2)/\mu = \varepsilon/\mu - \mu/2$. By the symmetry of the normal curve, $\Pr[L > \varepsilon] = \Phi(-t)$. A larger shift makes $t$ smaller and the tail heavier, so the worst pair has the largest shift, $s = \Delta_2$.
5. **Put in the formula.** Write $c = \sqrt{2\ln(1.25/\delta)}$, so @eq-apx-cal-classic is $\sigma = c\Delta_2/\varepsilon$. Then $\mu = \Delta_2/\sigma = \varepsilon/c$, and
$$t = \frac{\varepsilon}{\mu} - \frac{\mu}{2} = c - \frac{\varepsilon}{2c}.$$
6. **Bound the normal tail.** For $t > 0$, every point $z$ beyond $t$ has $z/t \ge 1$, so the tail $\int_t^\infty\varphi(z)\,\mathrm dz$ is at most $\int_t^\infty\frac{z}{t}\varphi(z)\,\mathrm dz$. The derivative of $\varphi(z)$ is $-z\varphi(z)$, so the second integral equals $\varphi(t)/t$. This is the **Mills-ratio bound**:
$$\Phi(-t) \le \frac{e^{-t^2/2}}{t\sqrt{2\pi}}.$$
7. **Expand $t^2$.** Squaring step 5 gives $t^2 = c^2 - \varepsilon + \varepsilon^2/(4c^2)$, which is at least $c^2 - \varepsilon$. So $e^{-t^2/2} \le e^{-c^2/2}e^{\varepsilon/2}$. By the definition of $c$, $e^{-c^2/2} = \delta/1.25$. Together with step 6,
$$\Pr[L > \varepsilon] \le \delta \cdot \frac{e^{\varepsilon/2}}{1.25\sqrt{2\pi}\,t}.$$
8. **This is where 1.25 and $\varepsilon < 1$ come in.** The bound is at most $\delta$ when the fraction is at most 1, that is when $t \ge e^{\varepsilon/2}/(1.25\sqrt{2\pi})$. For $\varepsilon < 1$, the right side is below $e^{1/2}/(1.25\sqrt{2\pi}) = 0.5262$. It remains to show that $t$ is that large. For $\varepsilon < 1$, step 5 gives $t = c - \varepsilon/(2c) \ge c - 1/(2c)$. This lower bound grows with $c$, and it equals 0.5262 at $c = 1.0176$, the positive root of $c^2 - 0.5262c - \frac12 = 0$. Since $c$ grows as $\delta$ shrinks, $c \ge 1.0176$ holds whenever $\delta \le 1.25e^{-1.0176^2/2} = 0.7448$. So the constant 1.25 absorbs the factor $e^{\varepsilon/2}/(t\sqrt{2\pi})$, and the condition $\varepsilon < 1$ keeps $e^{\varepsilon/2}$ below $e^{1/2}$.

Step 3 is the reason a tail probability is enough. The rare outputs with a large loss are paid for by $\delta$, and every other output keeps the factor $e^\varepsilon$.

### Checking it

The proof made three approximations in a row. Step 4 replaced the exact $\delta$ by the bad-set probability, step 6 replaced that by the Mills bound, and step 7 loosened it further with the constants. @fig-apx-cal-classic-chain follows each of them for $\varepsilon$ up to 1 at $\delta = 10^{-5}$. At $\varepsilon = 0.5$ the numbers are in @tbl-apx-cal-chain.

| Quantity at the classic $\sigma = 9.6896$ | Value |
| --- | --- |
| Target $\delta$ | $10^{-5}$ |
| After expanding $t^2$, step 7 | $8.55\times10^{-7}$ |
| Mills bound $\varphi(t)/t$, step 6 | $8.54\times10^{-7}$ |
| Bad event $\Pr[L > \varepsilon] = \Phi(-t)$, step 4 | $8.21\times10^{-7}$ |
| Exact $\delta(\varepsilon)$, from @sec-apx-cal-gauss-exact | $1.61\times10^{-8}$ |

: Each step of the classic proof at $\varepsilon = 0.5$, $\delta = 10^{-5}$, $\Delta_2 = 1$, where $t = 4.7932$. {#tbl-apx-cal-chain}

So the classic $\sigma$ delivers a $\delta$ 622 times smaller than asked for. The constants of step 8 cost a factor of 11.7, and dropping the correction term of @eq-apx-cal-exact-delta costs a factor of 51.0. The Mills ratio itself costs almost nothing.

![The $\delta$ that the classic $\sigma$ delivers at each $\varepsilon$ up to 1, for a target $\delta = 10^{-5}$, at each step of the proof. The bound after step 7, the Mills bound and the bad-event probability nearly coincide, about ten times below the target. The exact $\delta$ is a further 25.8 times smaller at $\varepsilon = 1$ and 505.9 times smaller at $\varepsilon = 0.05$.](figures/cal_classic_chain.svg){#fig-apx-cal-classic-chain}

### Recipe and pitfalls

1. Compute the $\ell_2$ sensitivity $\Delta_2$.
2. For $\varepsilon < 1$, set $\sigma = \Delta_2\sqrt{2\ln(1.25/\delta)}/\varepsilon$.
3. Add independent normal noise with that $\sigma$ to every coordinate.

The formula is valid but wasteful, as the table shows. The proof says nothing for $\varepsilon \ge 1$, and @sec-apx-cal-gauss-exact shows that the formula eventually becomes wrong there. Choose $\delta$ much smaller than $1/n$. A mechanism that publishes each person's record in full with probability $\delta$ satisfies the definition, and with $n$ people and $\delta = 1/n$ it publishes about one person's record every time. @sec-lg-gauss-classic gives the same proof with the vector case in full.

## Gaussian noise, the exact calibration {#sec-apx-cal-gauss-exact}

### The question

The classic formula overshoots by a large factor, and it says nothing for $\varepsilon \ge 1$. What is the smallest $\sigma$ that gives $(\varepsilon, \delta)$-DP, for any $\varepsilon$?

### The answer

Measure the sensitivity in units of the noise, $\mu = \Delta_2/\sigma$. The Gaussian mechanism is $(\varepsilon, \delta)$-DP exactly when $\delta \ge \delta(\varepsilon)$, where
$$\delta(\varepsilon) = \Phi\Big(\frac{\Delta_2}{2\sigma} - \frac{\varepsilon\sigma}{\Delta_2}\Big) - e^\varepsilon\,\Phi\Big(-\frac{\Delta_2}{2\sigma} - \frac{\varepsilon\sigma}{\Delta_2}\Big) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^\varepsilon\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big).$$ {#eq-apx-cal-exact}
This $\delta(\varepsilon)$ falls as $\sigma$ grows. So the smallest $\sigma$ for a target $(\varepsilon, \delta)$ is the one solution of $\delta(\varepsilon) = \delta$, which a computer finds by bisection.

### A worked example

To see the two terms, take a case where both terms are large enough to see, $\mu = 1$ and $\varepsilon = 1$. The first term is $\Phi(0.5 - 1) = \Phi(-0.5) = 0.3085$, the probability of the bad set on $D$. The second is $e \cdot \Phi(-1.5) = 0.1816$, the correction. Their difference is the exact $\delta = 0.1269$.

Now the count again, with $\Delta_2 = 1$, $\varepsilon = 0.5$ and $\delta = 10^{-5}$. Bisection finds $\sigma = 7.0318$, and putting it back into @eq-apx-cal-exact gives $\delta = 1.000\times10^{-5}$. The classic formula gave 9.6896, which is 1.378 times more noise than needed. At the classic $\sigma$ the true $\delta$ is $1.61\times10^{-8}$. With the exact $\sigma$ the 95% error is $1.96 \times 7.0318 = 13.78$, against 18.99 with the classic one.

### Why it works

1. **Start from the exact $\delta$.** By @eq-apx-cal-exact-delta, the smallest $\delta$ for one pair of neighbours is $\Pr[M(D) \in B] - e^\varepsilon\Pr[M(D') \in B]$, where $B$ is the set of outputs with $L > \varepsilon$.
2. **The first term.** Step 4 of @sec-apx-cal-gauss-classic found $\Pr[M(D) \in B] = \Phi(-t)$ with $t = \varepsilon/\mu - \mu/2$, which is $\Phi(\mu/2 - \varepsilon/\mu)$.
3. **The second term.** Measure the output in units of $\sigma$, starting from $D$'s true answer. The bad set is the outputs below $\mu/2 - \varepsilon/\mu$, by step 4 of @sec-apx-cal-gauss-classic. On $D'$ the output is centred at $\mu$ instead of 0, so it falls below that point with probability $\Phi(\mu/2 - \varepsilon/\mu - \mu) = \Phi(-\mu/2 - \varepsilon/\mu)$.
4. **Combine.** Subtracting $e^\varepsilon$ times the second term from the first gives @eq-apx-cal-exact for this pair.
5. **The largest shift is the worst pair, and more noise always helps.** The plan is to show that $\delta$ grows with $\mu$, by computing its derivative. Write $\alpha = \mu/2 - \varepsilon/\mu$ and $\beta = -\mu/2 - \varepsilon/\mu$ for the two arguments, so that $\delta = \Phi(\alpha) - e^\varepsilon\Phi(\beta)$. Their derivatives in $\mu$ are $\frac12 + \varepsilon/\mu^2$ and $-\frac12 + \varepsilon/\mu^2$, which differ by exactly 1. Also $\alpha^2 - \beta^2 = (\alpha - \beta)(\alpha + \beta) = \mu \cdot (-2\varepsilon/\mu) = -2\varepsilon$, so $\varphi(\beta) = e^{-\varepsilon}\varphi(\alpha)$. The derivative of $\delta$ in $\mu$ is therefore
$$\varphi(\alpha)\Big(\frac12 + \frac{\varepsilon}{\mu^2}\Big) - e^\varepsilon e^{-\varepsilon}\varphi(\alpha)\Big(-\frac12 + \frac{\varepsilon}{\mu^2}\Big) = \varphi(\alpha) > 0.$$
So $\delta$ grows with the shift $\mu$, and the worst pair of neighbours is the one with the largest shift, $\Delta_2$. And since $\mu = \Delta_2/\sigma$, a larger $\sigma$ means a smaller $\mu$ and a smaller $\delta$.
6. **"Exactly when."** The worst pair needs $\delta \ge \delta(\varepsilon)$ by step 1, and every other pair needs less by step 5. So $\delta \ge \delta(\varepsilon)$ is both necessary and enough.
7. **Solve by bisection.** As $\sigma$ grows from near 0 to very large, $\delta(\varepsilon)$ falls steadily from near 1 to near 0, so it crosses the target exactly once. Bisection keeps an interval of $\sigma$ values with the crossing inside, evaluates $\delta$ at its midpoint, and keeps the half that still contains the crossing. Each step halves the interval, and the script takes its midpoints in $\ln\sigma$ because $\sigma$ can range over many orders of magnitude.

Intuitively, step 5 says that adding more noise can only make the two datasets harder to tell apart. A Gaussian mechanism with a larger $\sigma$ is the one with a smaller $\sigma$ plus extra independent noise, and anything you can detect after adding that extra noise yourself, you could detect before.

### Checking it

@fig-apx-cal-bad-region draws the derivation at $\mu = 1$ and $\varepsilon = 1$. On the left, the bad set is the outputs left of $-0.5\sigma$, where the density on $D$ rises above $e^\varepsilon$ times the density on $D'$. The red area between the two curves is $\delta$, and the orange area under it is the correction that the classic proof throws away. On the right, the same $\delta$ appears as the part of the privacy loss distribution beyond $L = \varepsilon$.

![The exact $\delta$ of the Gaussian mechanism at $\mu = 1$, $\varepsilon = 1$. Left: the output density on $D$ and $e^\varepsilon$ times the density on $D'$, with the output measured in units of $\sigma$. On the bad set, left of the dotted line, the red excess is $\delta = 0.1269$ and the orange area is the correction $e^\varepsilon\Pr[M(D') \in B] = 0.1816$. Right: the same excess in terms of the privacy loss, which is normal with mean $\mu^2/2$ and standard deviation $\mu$ on $D$.](figures/cal_bad_region.svg){#fig-apx-cal-bad-region}

The script checks the closed form in three ways. First, it integrates the excess $p_D - e^\varepsilon p_{D'}$ over the bad set numerically (by Simpson's rule, on a fine grid), and at $\mu = 1$, $\varepsilon = 1$ this gives 0.1269. The same $\delta$ is also the average of $(1 - e^{\varepsilon - L})_+$ over losses drawn on $D$, where $(x)_+$ means $\max(x, 0)$ (@sec-dpdef-divergences shows why). A Monte Carlo average of it over a million random draws gives 0.1270. Second, over 22 pairs of $\varepsilon$ and $\mu$, the largest relative difference between the closed form and the numerical integral is $3.1\times10^{-6}$. Third, a numerical derivative in $\mu$ agrees with $\varphi(\alpha)$ of step 5 to within $10^{-9}$ on a grid of 16 pairs.

@tbl-apx-cal-sigma compares the two calibrations at $\delta = 10^{-5}$, and @fig-apx-cal-sigma draws them over a wider range.

| $\varepsilon$ | Classic $\sigma/\Delta_2$ | Exact $\sigma/\Delta_2$ | Classic / exact |
| --- | --- | --- | --- |
| 0.1 | 48.4481 | 30.7496 | 1.576 |
| 0.25 | 19.3792 | 13.2855 | 1.459 |
| 0.5 | 9.6896 | 7.0318 | 1.378 |
| 1 | 4.8448 | 3.7306 | 1.299 |
| 2 | 2.4224 | 1.9938 | 1.215 |
| 4 | 1.2112 | 1.0812 | 1.120 |
| 8 | 0.6056 | 0.6002 | 1.009 |

: The classic and the exact Gaussian noise at $\delta = 10^{-5}$. {#tbl-apx-cal-sigma}

![Left: the noise $\sigma/\Delta_2$ needed for $(\varepsilon, 10^{-5})$-DP by the classic formula and by the exact calibration, on log axes. Right: their ratio. For $\varepsilon$ from 0.05 to 1 the classic formula uses 1.30 to 1.68 times the necessary noise. Beyond $\varepsilon = 1$, in the shaded region, its proof says nothing, and beyond $\varepsilon = 8.42$ it adds too little noise.](figures/cal_sigma_vs_eps.svg){#fig-apx-cal-sigma}

For $\varepsilon < 1$ the classic formula is valid but loose, by a factor that grows as $\varepsilon$ shrinks. Above 1 its proof no longer applies. The formula still happens to give enough noise at $\delta = 10^{-5}$ up to $\varepsilon = 8.42$, but beyond that point its true $\delta$ exceeds the target, so it is wrong there, not just unproved.

### Recipe and pitfalls

1. Compute the $\ell_2$ sensitivity $\Delta_2$.
2. Solve $\delta(\varepsilon) = \delta$ for $\mu$ by bisection, using @eq-apx-cal-exact with $\Phi(x)$ computed as `0.5 * math.erfc(-x / math.sqrt(2))` in Python. Then set $\sigma = \Delta_2/\mu$.
3. Check the answer by putting $\sigma$ back into @eq-apx-cal-exact.

Because only $\mu = \Delta_2/\sigma$ enters, you can solve once for $\Delta_2 = 1$ and scale: at $\varepsilon = 1$ and $\delta = 10^{-5}$ the noise is $3.7306\,\Delta_2$. Never use the classic formula for $\varepsilon \ge 1$. One Gaussian mechanism satisfies a whole curve of $(\varepsilon, \delta)$ pairs, so quote $\mu$ or the curve rather than one pair. @sec-lg-analytic derives the same formula through a divergence between the two output distributions, and @sec-apx-ccomp-gaussian shows that many Gaussian releases combine into one. @sec-apx-cal-tradeoffs works through the formula by hand and in code, and shows how the noise changes when you move $\varepsilon$, $\delta$ or the number of releases.

## How ε, δ, σ and the sensitivity trade off {#sec-apx-cal-tradeoffs}

### The question

The last section gave a formula for the exact $\delta$ of the Gaussian mechanism and a way to solve it for $\sigma$. In practice you start from what you can give up, a privacy budget of $\varepsilon$ and $\delta$, or from the noise you can afford, and you want to know how the four quantities move together. They are the privacy parameter $\varepsilon$, the slack $\delta$, the noise standard deviation $\sigma$ and the $\ell_2$ sensitivity $\Delta_2$. How do you compute $\delta$ from normal probabilities, by hand and in code? And if you halve $\varepsilon$, make $\delta$ a thousand times smaller or release the answer ten times, how much more noise do you need?

### The answer

Everything follows from the exact formula of @sec-apx-cal-gauss-exact. Write $\Phi(x)$ for the probability that a standard normal variable is below $x$, and $\mu = \Delta_2/\sigma$ for the sensitivity measured in units of the noise. The mechanism is $(\varepsilon, \delta)$-DP exactly when $\delta$ is at least
$$\delta(\varepsilon) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^\varepsilon\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big).$$
Six facts about it answer the questions.

1. **Only the ratio $\mu = \Delta_2/\sigma$ matters.** The formula sees $\Delta_2$ and $\sigma$ only through $\mu$. Doubling the sensitivity and doubling $\sigma$ therefore gives the same guarantee, and you can solve once for $\Delta_2 = 1$ and multiply the resulting $\sigma$ by $\Delta_2$.
2. **One $\sigma$ gives a whole curve.** For a fixed $\mu$, $\delta(\varepsilon)$ falls as $\varepsilon$ grows. So a Gaussian mechanism is $(\varepsilon, \delta(\varepsilon))$-DP for every $\varepsilon$ at once, not for one pair.
3. **The noise falls roughly like $1/\varepsilon$, then more slowly.** Write $z_\delta$ for the point a standard normal variable exceeds with probability $\delta$, so that $\Phi(-z_\delta) = \delta$. The exact $\delta$ is never more than the probability of the bad set, a normal tail. So asking only that this tail be $\delta$ gives a simple **tail rule**, which always gives enough noise:
$$\sigma_{\text{tail}} = \Delta_2\,\frac{z_\delta + \sqrt{z_\delta^2 + 2\varepsilon}}{2\varepsilon},$$ {#eq-apx-cal-tail-rule}
The rule is about $\Delta_2 z_\delta/\varepsilon$ when $\varepsilon$ is small compared with $z_\delta^2/2$, and about $\Delta_2/\sqrt{2\varepsilon}$ when it is large. At $\delta = 10^{-5}$ it uses 1.39 times the exact noise for $\varepsilon = 0.1$. The excess shrinks as $\varepsilon$ grows, to 1.02 times at $\varepsilon = 32$.
4. **The noise grows only slowly as $\delta$ shrinks.** The point $z_\delta$ grows like $\sqrt{2\ln(1/\delta)}$, and by fact 3 the noise is roughly proportional to $z_\delta$. Near $\delta = 10^{-5}$, each factor of 10 in $\delta$ adds only about 13% to 15% more noise at $\varepsilon$ of 0.5 or 1.
5. **Repeating costs $\sqrt k$.** Releasing $k$ answers, each with sensitivity $\Delta_2$ and its own independent noise $\sigma$, gives exactly the guarantee of one release with $\mu\sqrt k$. Then $k$ releases need $\sqrt k$ times the noise of one for the same $(\varepsilon, \delta)$.
6. **Choose $\delta$ from the number of people $n$.** Take $\delta$ well below $1/n$, for example $1/n^2$. The reason, shown in step 7 below, is that a mechanism that publishes each record with probability $\delta$ is $(0, \delta)$-DP, and with $\delta = 1/n$ it publishes someone's record with probability about 0.63.

### A worked example

**By hand.** Take $\Delta_2 = 1$, $\sigma = 4$ and $\varepsilon = 1$. @tbl-apx-cal-byhand follows the formula one number at a time, for a dataset $D$ and a neighbour $D'$. First compute $\mu$, then the two arguments of $\Phi$, which @sec-apx-cal-gauss-exact called $\alpha$ and $\beta$. Then look up $\Phi$ of each in a normal table or compute it, and finally combine them. The second term, $e^\varepsilon\Phi(\beta)$, is the **correction** that turns the probability of the bad set into the exact $\delta$.

| Step | Formula | Value |
| --- | --- | --- |
| Sensitivity in noise units | $\mu = \Delta_2/\sigma$ | 0.2500 |
| First argument | $\alpha = \mu/2 - \varepsilon/\mu = 0.125 - 4$ | $-3.8750$ |
| Second argument | $\beta = -\mu/2 - \varepsilon/\mu = -0.125 - 4$ | $-4.1250$ |
| Probability of the bad set on $D$ | $\Phi(\alpha)$ | $5.3312\times10^{-5}$ |
| Same set on $D'$ | $\Phi(\beta)$ | $1.8537\times10^{-5}$ |
| Correction | $e^\varepsilon\,\Phi(\beta) = 2.7183 \times 1.8537\times10^{-5}$ | $5.0388\times10^{-5}$ |
| Exact $\delta$ | $\Phi(\alpha) - e^\varepsilon\Phi(\beta)$ | $2.9243\times10^{-6}$ |

: The exact $\delta$ of the Gaussian mechanism with $\Delta_2 = 1$, $\sigma = 4$ and $\varepsilon = 1$, one step at a time. {#tbl-apx-cal-byhand}

So noise of standard deviation 4 on a count, whose sensitivity is 1, gives $(1, 2.92\times10^{-6})$-DP. Notice how much the correction removes. The first term, the probability of the bad set, is 18.2 times the final $\delta$, so using it alone would overstate $\delta$ eighteen-fold.

**By computer.** Most languages have no $\Phi$ built in, but they all have the **complementary error function** $\operatorname{erfc}$, which Python calls `math.erfc`. In terms of it,
$$\Phi(x) = \tfrac12\operatorname{erfc}\big(-x/\sqrt2\big), \qquad 1 - \Phi(x) = \tfrac12\operatorname{erfc}\big(x/\sqrt2\big).$$
Use these forms, and never compute a small tail as 1 minus something close to 1. A double-precision number carries about 16 significant digits. So when $\Phi(x)$ is within $10^{-16}$ of 1, the subtraction $1 - \Phi(x)$ keeps none of the digits that matter. @tbl-apx-cal-tails shows this at $x = 8$ and $x = 9$. The textbook form $\Phi(x) = \frac12(1 + \operatorname{erf}(x/\sqrt2))$, with the plain error function $\operatorname{erf}$, has the same problem for negative $x$, because $\operatorname{erf}$ is then close to $-1$.

| Way of computing the upper tail at $x$ | $x = 8$ | $x = 9$ |
| --- | --- | --- |
| $1 - \Phi(x)$, by subtraction | $6.6613\times10^{-16}$ | 0 |
| $\Phi(-x)$ with the $\operatorname{erf}$ form | $6.1062\times10^{-16}$ | 0 |
| $\frac12\operatorname{erfc}(x/\sqrt2)$ | $6.2210\times10^{-16}$ | $1.1286\times10^{-19}$ |

: The probability that a standard normal variable exceeds $x$, computed three ways in double precision. Only the last is right. {#tbl-apx-cal-tails}

The same cancellation spoils $\delta$ itself when $\delta$ is tiny. Then $\alpha$ and $\beta$ are far out in the lower tail, where $\operatorname{erf}$ is close to $-1$ and the erf form adds 1 to it. The script that comes with this appendix calibrates $\sigma$ at $\varepsilon = 1$ for four targets. It puts each $\sigma$ back into the formula, computing $\Phi$ once with $\operatorname{erfc}$ and once with $\operatorname{erf}$, and @tbl-apx-cal-precision shows the result. With $\operatorname{erfc}$, every target comes back correct to five digits. With $\operatorname{erf}$, the target $10^{-12}$ is already off in the fifth digit. The error then grows fast, so that $10^{-15}$ comes back 4% too small and $10^{-18}$ comes back 55 times too large.

| Target $\delta$ | $\sigma/\Delta_2$ | $\delta$ with $\operatorname{erfc}$ | $\delta$ with $\operatorname{erf}$ | First term / $\delta$ |
| --- | --- | --- | --- | --- |
| $10^{-9}$ | 5.4953 | $1.0000\times10^{-9}$ | $1.0000\times10^{-9}$ | 32.5 |
| $10^{-12}$ | 6.5578 | $1.0000\times10^{-12}$ | $9.9997\times10^{-13}$ | 45.4 |
| $10^{-15}$ | 7.4870 | $1.0000\times10^{-15}$ | $9.6226\times10^{-16}$ | 58.5 |
| $10^{-18}$ | 8.3220 | $1.0000\times10^{-18}$ | $5.5511\times10^{-17}$ | 71.7 |

: The exact $\delta$ at $\varepsilon = 1$, recomputed at the calibrated $\sigma$ with $\Phi$ from $\operatorname{erfc}$ and from $\operatorname{erf}$. The last column is how much larger the first term is than the difference. {#tbl-apx-cal-precision}

Even with $\operatorname{erfc}$, the formula subtracts two nearly equal terms, and a subtraction loses digits too. The last column says how many. When the first term is $r$ times the difference, the subtraction loses about $\log_{10} r$ digits. Here $r$ is at most 72, and $\log_{10} 72 = 1.86$, so the subtraction loses at most 2 of the 16 digits. To check this independently, the script also integrates the excess $p_D - e^\varepsilon p_{D'}$ over the bad set directly. Measure the noise in units of $\sigma$ as $y$, so that its privacy loss is $L(y) = \mu^2/2 - \mu y$. The integrand is then $\varphi(y)\,(1 - e^{\varepsilon - L(y)})$, which is never a difference of two tails. That integral agrees with the $\operatorname{erfc}$ formula to about one part in $10^{12}$ at all four targets.

**Scaling.** Now take a sum of values clipped to $[0, 2]$, so that $\Delta_2 = 2$, with $\sigma = 8$. The ratio is still $\mu = 0.25$, so $\delta$ is again $2.9243\times10^{-6}$, and the same holds for $\Delta_2 = 0.5$ with $\sigma = 2$.

**The curve.** The same $\sigma = 4$ gives a whole curve of guarantees. It is $(1, 2.92\times10^{-6})$-DP, and also $(2, 5.09\times10^{-17})$-DP and $(0.9263, 10^{-5})$-DP. @fig-apx-cal-delta-curves draws the curve for four noise levels. Each falls steeply as $\varepsilon$ grows, and each crosses $\delta = 10^{-5}$ at its own $\varepsilon$.

![The exact $\delta(\varepsilon)$ of the Gaussian mechanism against $\varepsilon$, on a log scale, for four noise levels $\mu = \Delta_2/\sigma$. A single $\sigma$ is $(\varepsilon, \delta(\varepsilon))$-DP for every $\varepsilon$ on its curve. The dots mark the $\varepsilon$ each one gives at $\delta = 10^{-5}$: 0.93 for $\sigma = 4\Delta_2$, 1.99 for $2\Delta_2$, 4.38 for $\Delta_2$ and 10.00 for $\Delta_2/2$.](figures/cal_delta_curves.svg){#fig-apx-cal-delta-curves}

### Why it works

1. **Only $\mu$ matters.** In the formula, $\Delta_2$ and $\sigma$ appear only in $\mu/2$ and $\varepsilon/\mu$. So any sensitivity and noise with the same ratio give the same curve. In particular, the smallest $\sigma$ for a target is $\Delta_2$ times the smallest $\sigma$ for a sensitivity of 1. The reason behind it is that measuring the output in units of $\sigma$ turns every Gaussian mechanism into one with noise 1 and shift $\mu$.
2. **The curve falls.** Write $\alpha = \mu/2 - \varepsilon/\mu$ and $\beta = -\mu/2 - \varepsilon/\mu$ as before, and $\varphi$ for the standard normal density. Step 5 of @sec-apx-cal-gauss-exact showed that $e^\varepsilon\varphi(\beta) = \varphi(\alpha)$. Both arguments fall by $1/\mu$ for each unit of $\varepsilon$, so differentiating in $\varepsilon$ gives
$$\frac{\mathrm d\delta}{\mathrm d\varepsilon} = -\frac{\varphi(\alpha)}{\mu} - e^\varepsilon\Phi(\beta) + \frac{e^\varepsilon\varphi(\beta)}{\mu} = -e^\varepsilon\,\Phi(\beta) < 0.$$
The first and last terms cancel, and what is left is negative. So a larger $\varepsilon$ always comes with a smaller $\delta$, and one mechanism traces a curve. The same step 5 showed that $\delta$ grows with $\mu$. More noise means a smaller $\mu$, so it lowers the whole curve.
3. **A safe rule from the bad set alone.** Step 4 of @sec-apx-cal-dp showed that the exact $\delta$ is never more than the probability of the bad set, which step 4 of @sec-apx-cal-gauss-classic computed:
$$\Pr[L > \varepsilon] = \Phi(\alpha).$$
So any $\sigma$ that makes $\Phi(\alpha) = \delta$ is enough. Since $\Phi(-z_\delta) = \delta$, that means $\alpha = -z_\delta$, which is $\mu/2 - \varepsilon/\mu = -z_\delta$. Multiplying by $2\mu$ gives the quadratic $\mu^2 + 2z_\delta\mu - 2\varepsilon = 0$, whose positive root is $\mu = \sqrt{z_\delta^2 + 2\varepsilon} - z_\delta$. Then $\sigma = \Delta_2/\mu$, and multiplying the top and bottom by $\sqrt{z_\delta^2 + 2\varepsilon} + z_\delta$ gives @eq-apx-cal-tail-rule. Solving the same equation for $\varepsilon$ instead gives the matching rule for the $\varepsilon$ of a given $\sigma$:
$$\varepsilon_{\text{tail}} = z_\delta\mu + \frac{\mu^2}{2}.$$
4. **Its two regimes.** When $2\varepsilon$ is small compared with $z_\delta^2$, the square root is close to $z_\delta$, so $\sigma_{\text{tail}} \approx \Delta_2 z_\delta/\varepsilon$, which is proportional to $1/\varepsilon$. When $2\varepsilon$ is large compared with $z_\delta^2$, the square root is close to $\sqrt{2\varepsilon}$ and dominates $z_\delta$, so $\sigma_{\text{tail}} \approx \Delta_2/\sqrt{2\varepsilon}$. The bend between the two is near $\varepsilon = z_\delta^2/2$. At $\delta = 10^{-5}$, where $z_\delta = 4.2649$, that is $\varepsilon = 9.09$. The classic formula has the same $1/\varepsilon$ shape with $\sqrt{2\ln(1.25/\delta)}$ in place of $z_\delta$, but it never bends, which is why it fails at large $\varepsilon$ (@sec-apx-cal-gauss-exact).
5. **Why $\delta$ costs little.** By the Mills-ratio bound of @sec-apx-cal-gauss-classic, $\Phi(-z_\delta)$ is close to $e^{-z_\delta^2/2}/(z_\delta\sqrt{2\pi})$. So $\delta$ shrinks like $e^{-z_\delta^2/2}$, and $z_\delta$ grows only like $\sqrt{2\ln(1/\delta)}$. Dividing $\delta$ by 10 adds about $2\ln 10 = 4.61$ to $z_\delta^2$. At $\delta = 10^{-5}$ this makes $z_\delta$ grow from 4.2649 to 4.7534, a factor of 1.115, and the tail rule's noise of about $\Delta_2 z_\delta/\varepsilon$ grows by the same factor. The exact noise grows a little faster, as the checks below show.
6. **Repeating adds the losses.** Suppose the $k$ questions are fixed in advance, each answer has sensitivity $\Delta_2$, and each gets independent noise $\sigma$. Together the $k$ answers are one vector, with $\ell_2$ sensitivity $\sqrt{k\Delta_2^2} = \sqrt k\,\Delta_2$. Step 2 of @sec-apx-cal-gauss-classic showed that a vector release behaves like one number shifted by the $\ell_2$ length of the change. So the $k$ releases have the exact curve of one release with $\mu\sqrt k$. To keep $\mu\sqrt k$ at the value one release needs, $\sigma$ must grow by $\sqrt k$. When a question may depend on earlier answers, the $k$ answers are no longer one fixed vector query, so this argument does not apply as it stands. @sec-apx-ccomp-gaussian proves the same result for that case.
7. **Why $\delta$ must be below $1/n$.** Take the mechanism that publishes each record in full with probability $\delta$, independently. One record changes the output only when its own coin publishes it, so the two probabilities of any event differ by at most $\delta$, and the mechanism is $(0, \delta)$-DP. Each of $n$ records stays hidden with probability $1 - \delta$, so someone is exposed with probability $1 - (1 - \delta)^n$. With $\delta = 1/n$ this is about 0.63. The reason is that $n\ln(1 - 1/n)$ is close to $-1$ for large $n$, so
$$(1 - 1/n)^n = e^{n\ln(1 - 1/n)} \approx e^{-1},$$
and $1 - e^{-1} = 0.63$. A guarantee that allows this mechanism promises little, so $n\delta$ must be far below 1.

### Checking it

@tbl-apx-cal-grid is the reference table of this section. It gives the exact $\sigma/\Delta_2$ for five values of $\varepsilon$ and four of $\delta$, each found by bisection and checked by putting it back into the formula. Read along a row to see the cost of a smaller $\delta$, and up a column to see the cost of a smaller $\varepsilon$.

| $\varepsilon$ | $\delta = 10^{-3}$ | $\delta = 10^{-5}$ | $\delta = 10^{-7}$ | $\delta = 10^{-9}$ |
| --- | --- | --- | --- | --- |
| 0.1 | 17.4044 | 30.7496 | 41.3295 | 50.2098 |
| 0.5 | 4.6101 | 7.0318 | 8.9957 | 10.6739 |
| 1 | 2.5747 | 3.7306 | 4.6787 | 5.4953 |
| 2 | 1.4452 | 1.9938 | 2.4491 | 2.8445 |
| 4 | 0.8231 | 1.0812 | 1.2978 | 1.4878 |

: The smallest $\sigma/\Delta_2$ for $(\varepsilon, \delta)$-DP, from the exact calibration. Multiply by $\Delta_2$ to get $\sigma$. {#tbl-apx-cal-grid}

Moving along a row, to a smaller $\delta$, changes the noise much less than moving up a column, to a smaller $\varepsilon$. At $\varepsilon = 0.5$, for example, a $\delta$ ten thousand times smaller costs 1.518 times the noise. Halving $\varepsilon$ instead, to 0.25, costs 1.889 times the noise (@tbl-apx-cal-sigma has the value at 0.25). @fig-apx-cal-tradeoff-map shows the whole plane. On the left each line is one noise level, and the lines are nearly vertical because $\sigma$ depends mostly on $\varepsilon$. On the right, the product $\varepsilon\sigma/\Delta_2$ would be a flat line if $\sigma$ were exactly proportional to $1/\varepsilon$.

![Left: contour lines of the exact $\sigma/\Delta_2$ over $\varepsilon$ (log scale) and $\log_{10}\delta$, with $\delta = 10^{-5}$ dotted. The lines are nearly vertical, so $\varepsilon$ decides the noise far more than $\delta$ does. Right: the product $\varepsilon\sigma/\Delta_2$ against $\varepsilon$ for three values of $\delta$, exact (solid) and by the tail rule (dashed). The tail rule is nearly flat for small $\varepsilon$, where it is close to $z_\delta/\varepsilon$, and the exact product sags below it there, because the correction term matters most at small $\varepsilon$. Both rise at large $\varepsilon$, where the noise falls more slowly than $1/\varepsilon$.](figures/cal_tradeoff_map.svg){#fig-apx-cal-tradeoff-map}

**Noise against $\varepsilon$.** @tbl-apx-cal-tail-rule compares the tail rule with the exact calibration at $\delta = 10^{-5}$. The rule is always enough, as step 3 promises. Its excess over the exact noise is 39% at $\varepsilon = 0.1$, and it shrinks steadily as $\varepsilon$ grows. The reason is the correction term that the rule ignores. At $\varepsilon = 0.1$ and the exact $\sigma$, the probability of the bad set is 111 times the final $\delta$, so the correction removes almost all of it. At $\varepsilon = 8$ it is only 3.6 times the final $\delta$, and the correction matters much less. In the exact calibration, the product $\varepsilon\sigma/\Delta_2$ grows by only a factor of 1.213 from $\varepsilon = 0.1$ to $\varepsilon = 1$. So "$\sigma$ is proportional to $1/\varepsilon$" is a fair first guess below $\varepsilon = 1$. It fails at large $\varepsilon$, where the product keeps growing, up to 6.5579 at $\varepsilon = 32$.

| $\varepsilon$ | Exact $\sigma/\Delta_2$ | $\varepsilon\sigma/\Delta_2$ | Tail rule | Tail / exact | $z_\delta/\varepsilon$ | $1/\sqrt{2\varepsilon}$ |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 30.7496 | 3.0750 | 42.7658 | 1.391 | 42.6489 | 2.2361 |
| 0.5 | 7.0318 | 3.5159 | 8.6454 | 1.229 | 8.5298 | 1.0000 |
| 1 | 3.7306 | 3.7306 | 4.3791 | 1.174 | 4.2649 | 0.7071 |
| 2 | 1.9938 | 3.9876 | 2.2439 | 1.125 | 2.1324 | 0.5000 |
| 4 | 1.0812 | 4.3246 | 1.1728 | 1.085 | 1.0662 | 0.3536 |
| 8 | 0.6002 | 4.8018 | 0.6320 | 1.053 | 0.5331 | 0.2500 |
| 32 | 0.2049 | 6.5579 | 0.2083 | 1.016 | 0.1333 | 0.1250 |

: The noise for $(\varepsilon, 10^{-5})$-DP: the exact calibration, the tail rule of @eq-apx-cal-tail-rule and its two limits, with $z_\delta = 4.2649$. {#tbl-apx-cal-tail-rule}

The last two columns show the two regimes of step 4. At $\varepsilon = 0.1$ the rule is almost exactly $z_\delta/\varepsilon$. At $\varepsilon = 32$ the rule has moved well away from $z_\delta/\varepsilon$ and towards $1/\sqrt{2\varepsilon}$, but it is still above that limit, because $z_\delta$ is not yet small next to $\sqrt{2\varepsilon} = 8$.

**Noise against $\delta$.** Starting from $\delta = 10^{-5}$, making $\delta$ ten times smaller raises the exact noise by a factor of 1.146 at $\varepsilon = 0.5$. There $\sigma/\Delta_2$ goes from 7.0318 to 8.0576, while at $\varepsilon = 1$ the factor is 1.132. Both are a little above the 1.115 that step 5 predicted from $z_\delta$ alone, so the estimate from $z_\delta$ alone is slightly low.

**The privacy a noise level buys.** Turning the question around, @tbl-apx-cal-eps-of-sigma gives the $\varepsilon$ that a chosen $\sigma$ buys at $\delta = 10^{-5}$, exactly and by the tail rule $\varepsilon_{\text{tail}} = z_\delta\mu + \mu^2/2$. For large $\sigma$ the $z_\delta\mu$ term dominates, so $\varepsilon$ falls like $1/\sigma$, the inverse of the noise rule. Again the rule is safe, because it never reports a smaller $\varepsilon$ than the exact one.

| $\sigma/\Delta_2$ | 0.5 | 1 | 2 | 5 | 10 | 20 |
| --- | --- | --- | --- | --- | --- | --- |
| Exact $\varepsilon$ | 9.9973 | 4.3772 | 1.9931 | 0.7255 | 0.3407 | 0.1600 |
| Tail rule $z_\delta\mu + \mu^2/2$ | 10.5298 | 4.7649 | 2.2574 | 0.8730 | 0.4315 | 0.2145 |

: The $\varepsilon$ that a given noise level buys at $\delta = 10^{-5}$. {#tbl-apx-cal-eps-of-sigma}

**Repeating.** Suppose ten releases must together be $(1, 10^{-5})$-DP. Then by step 6 each needs $\sqrt{10}$ times the noise of one release, so $\sigma = 11.7973$. As a check, putting $\mu = \sqrt{10}/11.7973$ back into the formula gives $\delta = 1.000\times10^{-5}$. The tempting alternative is to split the budget instead. Basic composition, which @sec-apx-ccomp-basic derives, says that the $\varepsilon$s and the $\delta$s of separate releases add up, so ten $(0.1, 10^{-6})$-DP releases are together $(1, 10^{-5})$-DP. That needs $\sigma = 36.3047$, which is 3.08 times more noise. With 100 releases the gap grows to 9.70 times. Splitting gives each release only $\varepsilon/k$, and the noise grows roughly like $1/\varepsilon$, so it grows like $k$ rather than $\sqrt k$.

**Choosing $\delta$ from $n$.** @tbl-apx-cal-delta-n gives the noise at $\varepsilon = 1$ for four choices of $\delta$, and how often the publish-a-record mechanism of step 7 would expose someone. A fixed $\delta = 10^{-5}$ is fine for a thousand people, with exposure 0.00995. For a million people it is ten times $1/n$, and the exposure is 0.99995. A rule sometimes used to get "a little below $1/n$", $\delta = 1/n^{1.1}$, is not far enough below, since its exposure is still 0.328 for ten thousand people. By contrast, $\delta = 1/n^2$ makes $n\delta = 1/n$, which keeps the exposure near $1/n$, and its noise is modest. For ten thousand people it needs only 1.367 times the noise of $\delta = 10^{-5}$.

| $n$ | $\delta = 10^{-5}$ | $\delta = 1/n$ | $\delta = 1/n^{1.1}$ | $\delta = 1/n^2$ | Exposure at $10^{-5}$ | Exposure at $1/n^{1.1}$ |
| --- | --- | --- | --- | --- | --- | --- |
| 1,000 | 3.7306 | 2.5747 | 2.7660 | 4.2247 | 0.00995 | 0.394 |
| 10,000 | 3.7306 | 3.1857 | 3.4106 | 5.1003 | 0.0952 | 0.328 |
| 100,000 | 3.7306 | 3.7306 | 3.9833 | 5.8678 | 0.632 | 0.271 |
| 1,000,000 | 3.7306 | 4.2247 | 4.5013 | 6.5578 | 0.99995 | 0.222 |

: The exact $\sigma/\Delta_2$ at $\varepsilon = 1$ for four ways of choosing $\delta$, and the probability $1 - (1 - \delta)^n$ that publishing each record with probability $\delta$ exposes someone. At $\delta = 1/n$ that probability is 0.632 for every $n$ shown. {#tbl-apx-cal-delta-n}

### Recipe and pitfalls

Decide the quantities in this order, because the later steps use the earlier choices.

1. **Fix the neighbouring relation and compute $\Delta_2$**, as in @sec-apx-cal-sensitivity. Clip first if the data are unbounded.
2. **Pick $\delta$ from the number of people $n$**, well below $1/n$. The choice $\delta = 1/n^2$ is cheap, as @tbl-apx-cal-delta-n shows, and @sec-epsc-delta discusses other rules.
3. **Pick $\varepsilon$.** It decides the noise far more than $\delta$ does, so spend your judgement here.
4. **Count the releases $k$** that must share the budget. If they are Gaussian releases of the same sensitivity, solve for one and multiply $\sigma$ by $\sqrt k$.
5. **Solve for $\sigma$ by bisection** on the exact formula with $\Delta_2 = 1$, computing $\Phi$ with $\operatorname{erfc}$, and multiply by $\Delta_2$. @tbl-apx-cal-grid gives a first estimate, and @eq-apx-cal-tail-rule gives a safe closed form if you cannot run bisection.
6. **Check by plugging back.** Put $\mu = \Delta_2/\sigma$ into the formula and confirm that $\delta(\varepsilon)$ is at most your target.
7. **Report $\mu$, or the curve,** not only one pair, so that a reader can find $\delta$ at any $\varepsilon$.

The common mistakes follow from the facts above. Quoting $\varepsilon$ without $\delta$ says nothing, because one Gaussian mechanism has every $\varepsilon$ on its curve. A $\delta$ of $1/n$ or more allows a mechanism that publishes a whole record. Computing $1 - \Phi(x)$ by subtraction, or $\Phi$ with $\operatorname{erf}$, silently loses accuracy from about $\delta = 10^{-12}$ and gives nonsense by $10^{-18}$. The classic formula is unproved above $\varepsilon = 1$ and wrong above 8.42 at $\delta = 10^{-5}$. And forgetting composition is the most expensive mistake of all, because $k$ releases at the $\sigma$ of one are only $(\varepsilon, \delta)$-DP for a much larger $\varepsilon$. Splitting the budget evenly fixes that, but at about $k$ times the noise of one release instead of $\sqrt k$ times. @sec-apx-ccomp-gaussian and @sec-apx-ccomp-compare compare the ways to account for repeated releases.

## All the recipes in one table {#sec-apx-cal-recipes}

@tbl-apx-cal-recipes collects the calibrations of this appendix. Every row assumes you have fixed the neighbouring relation first, since the sensitivity depends on it.

| Mechanism | Sensitivity | Noise or rule | Guarantee |
| --- | --- | --- | --- |
| Laplace | $\Delta_1$, the $\ell_1$ norm | $\mathrm{Lap}(b)$ per coordinate, $b = \Delta_1/\varepsilon$ | $\varepsilon$-DP; 95% error $b\ln 20$ per coordinate |
| Randomized response | one bit, under replace-one | truth with probability $e^\varepsilon/(1 + e^\varepsilon)$ | $\varepsilon$-DP; error about $1/(\varepsilon\sqrt n)$ |
| Exponential mechanism | $\Delta u$ of the score | choose $r$ with probability $\propto e^{\varepsilon u(D, r)/(2\Delta u)}$ | $\varepsilon$-DP; gap $\le \frac{2\Delta u}{\varepsilon}(\ln |\mathcal R| + t)$ with probability $1 - e^{-t}$ |
| Gaussian, classic | $\Delta_2$, the $\ell_2$ norm | $\sigma = \Delta_2\sqrt{2\ln(1.25/\delta)}/\varepsilon$ | $(\varepsilon, \delta)$-DP for $\varepsilon < 1$; loose |
| Gaussian, exact | $\Delta_2$, the $\ell_2$ norm | $\sigma = \Delta_2/\mu$ with $\delta(\varepsilon) = \delta$ by bisection | $(\varepsilon, \delta)$-DP for every $\varepsilon$; tight |
| Gaussian, $k$ releases | $\Delta_2$ of each release | $\sqrt k$ times the exact $\sigma$ of one, with $\delta$ well below $1/n$ | $(\varepsilon, \delta)$-DP for all $k$ together |

: The calibration recipes of this appendix. {#tbl-apx-cal-recipes}

## Code

The script computes every number on this page and draws every figure. It checks the three-output example against all events, finds every sensitivity by brute force over small datasets, and simulates the Laplace error and randomized response. For the Gaussian mechanism it compares the closed form with numerical integration and with Monte Carlo, checks the derivative of step 5, and solves the exact calibration by bisection. For the trade-offs it computes $\Phi$ three ways and checks tiny values of $\delta$ against an integral over the bad set. It also tabulates the noise as each quantity moves. From the repository root:

```bash
uv run python code/appendix-derivations/apx_core_calibration.py
```

@tbl-apx-cal-code summarizes its output with the default random seed, 0.

| Check | Result |
| --- | --- |
| Three outputs, every event | Largest ratio 2.5000; largest excess at $e^\varepsilon = 2$ is 0.1000; bad set 0.5000 |
| Sensitivities, every dataset of 1 to 3 records | Every row of @tbl-apx-cal-sens; mean under add/remove up to 1.5000 |
| Laplace loss, count, $b = 2$ | In exactly $[-0.5000, 0.5000]$; event ratio 1.6487 |
| Laplace loss, three counts, $b = 6$ | Largest over 100,000 outputs 0.5000 |
| Laplace error, $b = 2$ | Mean 2.0026 (exact 2); 95th percentile 5.9938 (formula 5.9915) |
| Randomized response, $\varepsilon = 0.5$, $n = 1000$ | Mean 0.2996, standard deviation 0.0629 (formula 0.0626) |
| Exponential mechanism, vote | Largest privacy loss 0.3353 at $\varepsilon = 1$ |
| Classic chain, $\varepsilon = 0.5$, $\delta = 10^{-5}$ | $8.55\times10^{-7}$, $8.54\times10^{-7}$, $8.21\times10^{-7}$, exact $1.61\times10^{-8}$ |
| Exact $\delta$, $\mu = 1$, $\varepsilon = 1$ | 0.1269 (Simpson 0.1269, Monte Carlo 0.1270) |
| Exact calibration, $\varepsilon = 0.5$, $\delta = 10^{-5}$ | $\sigma = 7.0318$ against classic 9.6896 |
| Classic formula fails at $\delta = 10^{-5}$ | For $\varepsilon > 8.42$ |
| Exact $\delta$ by hand, $\Delta_2 = 1$, $\sigma = 4$, $\varepsilon = 1$ | $2.9243\times10^{-6}$ (bad-set integral the same) |
| $\Phi$ from $\operatorname{erf}$ at $\delta = 10^{-18}$ | $5.5511\times10^{-17}$, against $1.0000\times10^{-18}$ from $\operatorname{erfc}$ |
| Tail rule against exact, $\delta = 10^{-5}$ | 1.391 times at $\varepsilon = 0.1$, 1.016 times at $\varepsilon = 32$ |
| Ten releases at $(1, 10^{-5})$ | $\sigma = 11.7973$ each, against 36.3047 when the budget is split |

: The script's output with the default seed. {#tbl-apx-cal-code}

[tests/test_appendix_core_calibration.py](../../tests/test_appendix_core_calibration.py) checks that the output still matches. Three things stand out. Every formula agrees with brute force or simulation to within sampling error. The Laplace and exponential mechanisms meet their bounds exactly on some outputs, so their noise cannot be reduced. And the classic Gaussian formula is safe but wasteful for small $\varepsilon$, and unsafe for very large $\varepsilon$, which is why the exact calibration should be the default.

## References

- Dwork, McSherry, Nissim & Smith, "Calibrating Noise to Sensitivity in Private Data Analysis" (TCC 2006): sensitivity and the Laplace mechanism.
- Warner, "Randomized Response: A Survey Technique for Eliminating Evasive Answer Bias" (1965): randomized response.
- McSherry & Talwar, "Mechanism Design via Differential Privacy" (FOCS 2007): the exponential mechanism.
- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014): §3.3 for the Laplace mechanism, §3.4 for the exponential mechanism and Appendix A for the classic Gaussian proof.
- Balle & Wang, "Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising" (ICML 2018): the exact $\delta(\varepsilon)$ of the Gaussian mechanism and its calibration by bisection.
- Near & Abuah, *Programming Differential Privacy*: the same mechanisms with code, and the practical pitfalls of sensitivity.
