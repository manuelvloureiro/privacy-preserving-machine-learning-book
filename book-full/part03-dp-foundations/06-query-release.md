# Query release and private multiplicative weights

## Why this matters

Many private releases are not one statistic but a large batch of simple ones. A statistics office publishes every cross-tabulation of a census file, and a dashboard shows hundreds of counts. Before it trains a model, a team computes how often every combination of a few features occurs. Each of these numbers is a **linear query**: the average, over the records, of a function that gives each record a value between 0 and 1. Counts and frequencies are the usual case, where the function is 1 for the records that match and 0 for the rest. The tools so far answer such a batch by adding noise to every answer, and the noise has to grow with the number of queries. Under pure differential privacy (DP), the Laplace mechanism's noise grows in proportion to the number of queries. Under approximate DP, the Gaussian mechanism's noise grows like its square root ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). With millions of queries, either is useless.

The growth is not only an artefact of these mechanisms. The [reconstruction attack](../part02-attacks/09-reconstruction.md) of Dinur and Nissim shows that answers that are all very accurate, to many more queries than there are records, give the data away. But that attack needs every answer to be within about $1/\sqrt n$ of the truth, where $n$ is the number of records. Between "too accurate to be private" and "too noisy to be useful" there is a wide gap, and this chapter fills it. It shows how to answer an arbitrary set of linear queries with an error that grows only with the *logarithm* of the number of queries and the logarithm of the number of possible records. As a bonus, the answer is a **synthetic dataset**: a histogram over all possible records that can be published and that answers every query in the set. It also answers queries outside the set, although the guarantee covers only the queries in it.

### The one idea: learn the data from its mistakes

Keep a public guess of the data, a histogram over all possible records that starts out uniform. Find a query that the guess gets badly wrong, measure that one query privately, and nudge the guess toward the measured answer. Each nudge is a step of the **multiplicative weights** method, which comes from online learning. Its guarantee bounds its **regret**, how much worse it does than the best fixed choice in hindsight. That bound says that only about $\ln N/\alpha^2$ nudges are ever needed before the guess is within a target error $\alpha$ on every query, where $N$ is the number of possible records. The number of queries does not appear. So the privacy budget is split over the corrections instead of over every query. The privacy tools come from the previous chapters. In each round the exponential mechanism of [Selection](05-selection.md) finds a badly answered query, and the Laplace mechanism measures it. [Advanced composition](04-composition.md) adds up the cost of the rounds. When the queries arrive one at a time, the sparse vector technique of [Selection](05-selection.md) replaces the exponential mechanism.

@tbl-qr-overview lists the results of the chapter and where each is used.

| Result | Where it is used |
| --- | --- |
| Noise on every answer: error grows like $\lvert Q\rvert$ or $\sqrt{\lvert Q\rvert}$ | The baseline; small or structured sets of queries |
| Multiplicative weights and its regret bound | The engine of everything below; online learning, and the [matrix-factorization training](../part05-dp-ml/04-matrix-mechanisms.md) of Part 5 |
| Regret to mistakes: at most $4\ln N/\alpha^2$ corrections | Why the number of rounds does not depend on the number of queries |
| Offline private multiplicative weights | Synthetic data for a fixed set of queries, such as all marginals of a table |
| Online private multiplicative weights | Interactive query servers; queries that depend on earlier answers |
| SmallDB | What the exponential mechanism can do without regard to running time; the limits of efficient release |

: The results of this chapter and where each is used. {#tbl-qr-overview}

**How to read this chapter.** The baseline section sets the target. The next two sections are about learning, with no privacy at all: multiplicative weights, and how its regret bound turns into a bound on the number of corrections. Read them before the private algorithm, which is those two sections with two lines replaced. The online version and SmallDB are variations and can be read in either order.

**What you need.** The exponential mechanism, its utility bound and the sparse vector technique are in [Selection](05-selection.md). Advanced composition is in [Advanced and optimal composition](04-composition.md), and the Laplace and Gaussian mechanisms are in [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md). Hoeffding's inequality and the union bound are in [Concentration inequalities](../part01-prerequisites/02-concentration.md). Each is restated where it is used.

**Notation.** The **universe** $X$ is the set of all possible records, and $N = |X|$ is its size. A dataset $D$ holds $n$ records, $x_1, \dots, x_n$. Its **histogram** $p$ records how often each value occurs. For each element $x$ of the universe, $p(x)$ is the fraction of the records that equal $x$.

A linear query is a function $q$ from the universe to $[0, 1]$. Its answer on $D$ is the average of its values over the records, $q(D) = \frac1n\sum_iq(x_i)$. The set of queries to answer is $Q$, and $|Q|$ is its size. Neighbouring datasets differ by replacing one record, and $n$ is public, so one answer moves by at most $1/n$.

The **error** of a guessed histogram $\hat p$ on a query is the distance between its answer and the true one. The target is an error of at most $\alpha$ on every query, except with a small failure probability $\beta$. Logarithms are natural.

**The running example.** A table has $n = 10{,}000$ records with eight yes/no attributes, so the universe has $N = 2^8 = 256$ elements. The records come from a mixture of three groups with different attribute rates. The queries are every cell of every marginal table: for every set of attributes and every combination of their values, the fraction of records with those values. Each nonempty set of attributes gives a table, so there are $2^8 - 1 = 255$ tables. A cell fixes each attribute to 0, to 1, or leaves it out, which gives $3^8$ choices, minus the one that leaves every attribute out, so there are $3^8 - 1 = 6560$ cells. The uniform histogram, which uses no data at all, has a largest error of 0.232 over these queries.

## Toolkit: three facts every derivation uses

### Fact 1 — A linear query is an inner product with the histogram

The answer to a linear query depends on the records only through how many of them take each value. So group the records by value: each element $x$ of the universe contributes $q(x)$ once for every record equal to it, which is $np(x)$ times. Dividing by $n$,
$$q(D) = \frac1n\sum_{i=1}^nq(x_i) = \sum_{x \in X}q(x)\,p(x) = \langle q, p\rangle.$$ {#eq-qr-inner}
Here $\langle q, p\rangle$ is the inner product of two vectors indexed by the universe.

A concrete case: two yes/no attributes, so the universe is $\{00, 01, 10, 11\}$, and four records $01, 11, 11, 10$. @tbl-qr-inner lists the histogram and the query "the first attribute is 1".

| Element $x$ | 00 | 01 | 10 | 11 |
| --- | --- | --- | --- | --- |
| Records equal to $x$ | 0 | 1 | 1 | 2 |
| $p(x)$ | 0 | 1/4 | 1/4 | 1/2 |
| $q(x)$: first attribute is 1 | 0 | 0 | 1 | 1 |

: A four-record dataset as a histogram, and one query. {#tbl-qr-inner}

The inner product is $0 + 0 + \frac14 + \frac12 = \frac34$, and indeed three of the four records have a first attribute of 1.

Two consequences matter. Any distribution $\hat p$ over the universe, not only a real dataset, gives an answer $\langle q, \hat p\rangle$ to every query, so a histogram is a synthetic dataset. And the error of $\hat p$ on $q$ is $|\langle q, \hat p - p\rangle|$, which is linear in the difference of the two histograms.

### Fact 2 — Two bounds on the logarithm

The analysis of multiplicative weights tracks the logarithm of a product of factors $1 - x$, and it needs two bounds on $\ln(1 - x)$, one from above and one from below. From above, the logarithm is concave, so it lies below its tangent at 0: $\ln(1 - x) \le -x$ for every $x < 1$. From below, near 0 it falls short of the tangent by at most $x^2$:
$$\ln(1 - x) \ge -x - x^2 \qquad \text{for } |x| \le \tfrac12.$$ {#eq-qr-log-lower}

To see the lower bound, let $g(x) = \ln(1 - x) + x + x^2$, so that $g(0) = 0$. Its derivative is $g'(x) = -\frac1{1 - x} + 1 + 2x = \frac{x(1 - 2x)}{1 - x}$. On $[0, \frac12]$ the derivative is at least 0, so $g$ rises from 0 and stays non-negative. On $[-\frac12, 0]$ it is at most 0, so $g$ falls toward $g(0) = 0$ and is non-negative there too. The two ends of the range show how much room is left. At $x = \frac12$, $\ln\frac12 = -0.6931$ lies above $-x - x^2 = -0.75$. At $x = -\frac12$, $\ln\frac32 = 0.4055$ lies above $-x - x^2 = 0.25$. Outside $[-\frac12, \frac12]$ the bound can fail, and at $x = 0.9$ it does, because $\ln 0.1 = -2.30$ lies below $-x - x^2 = -1.71$. That is why the step size will be kept small.

### Fact 3 — The privacy tools, restated

Three tools from earlier chapters do the private work. Each comes with a tail bound that the accuracy proofs combine by a union bound, which says that the probability that at least one of several events happens is at most the sum of their probabilities.

- The **Laplace mechanism** takes an answer of sensitivity $\Delta$, the most that one record can move it. To be $\varepsilon_0$-DP for a per-step budget $\varepsilon_0$, it adds noise $\mathrm{Lap}(b)$ of scale $b = \Delta/\varepsilon_0$. Its noise exceeds $b\ln(1/\beta')$ in absolute value with probability exactly $\beta'$.
- The **exponential mechanism** picks one of $k$ candidates, each with a score $u$ of sensitivity $\Delta$, with probability proportional to $e^{\varepsilon_0u/(2\Delta)}$, and is $\varepsilon_0$-DP. Its utility bound ([Selection](05-selection.md#sec-sel-utility)) says that, except with probability $\beta'$, the chosen candidate's score is within $\frac{2\Delta}{\varepsilon_0}\big(\ln k + \ln(1/\beta')\big)$ of the best.
- **Advanced composition** ([Advanced and optimal composition](04-composition.md#sec-comp-advanced)) says that $k$ steps that are each $\varepsilon_0$-DP, chosen adaptively, are $(\varepsilon', \delta)$-DP with $\varepsilon' = \varepsilon_0\sqrt{2k\ln(1/\delta)} + k\varepsilon_0\tanh(\varepsilon_0/2)$.

The derivations need a closed-form per-step budget. If each of $k$ steps gets
$$\varepsilon_0 = \frac{\varepsilon}{2\sqrt{2k\ln(1/\delta)}},$$ {#eq-qr-step-eps}
then the whole is $(\varepsilon, \delta)$-DP whenever $\varepsilon \le 8\ln(1/\delta)$. The condition is mild: at $\delta = 10^{-6}$ it allows any $\varepsilon$ up to 110. To see the claim, substitute $\varepsilon_0$ into $\varepsilon'$. The first term is exactly $\varepsilon/2$. The second term, $k\varepsilon_0\tanh(\varepsilon_0/2)$, is at most $k\varepsilon_0^2/2$, because $\tanh$ never exceeds its argument. Substituting $\varepsilon_0$ turns $k\varepsilon_0^2/2$ into $\varepsilon^2/(16\ln(1/\delta))$, and the condition $\varepsilon \le 8\ln(1/\delta)$ makes this at most $\varepsilon/2$. When $\varepsilon_0$ is small the second term is far below $\varepsilon/2$, so the closed form leaves up to half the budget unspent. The chapter's script also solves for the largest $\varepsilon_0$ exactly, and the Example of @sec-qr-pmw compares the two.

## Answering every query: the baseline {#sec-qr-baseline}

### Context

The direct way to release the answers to a set of queries is to add noise to each. Before improving on it, it helps to know exactly what it costs and why the cost grows with the number of queries.

### Statement

Write $f(D)$ for the vector of the $|Q|$ answers. Replacing one record moves each answer by at most $1/n$. The $\ell_1$ norm of the change, the sum of the absolute changes, is therefore at most $|Q|/n$. The $\ell_2$ norm, the square root of the sum of their squares, is at most $\sqrt{|Q|}/n$, because the squares add up to at most $|Q|/n^2$. The Laplace mechanism on the vector is $\varepsilon$-DP, and with probability at least $1 - \beta$ its largest error is below $\alpha$ once
$$n \ge \frac{|Q|\ln(|Q|/\beta)}{\alpha\varepsilon}.$$ {#eq-qr-laplace-n}
The Gaussian mechanism is calibrated through a number $\mu = \mu(\varepsilon, \delta)$, the largest ratio of sensitivity to noise standard deviation that the exact calibration of [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md) allows. Dividing the $\ell_2$ sensitivity by $\mu$ gives the noise it needs, a standard deviation $\sigma = \sqrt{|Q|}/(n\mu)$ on every answer, and then it is $(\varepsilon, \delta)$-DP. Its largest error is below $\alpha$ with probability at least $1 - \beta$ once
$$n \ge \frac{\sqrt{|Q|}\,z}{\mu\alpha},$$ {#eq-qr-gauss-n}
where $z$ is the number that a standard normal variable exceeds with probability $\beta/(2|Q|)$.

A query set with structure can do better. For all the cells of a marginal table, replacing one record lowers one cell by $1/n$ and raises another by $1/n$, so each table contributes only $2/n$ to the $\ell_1$ sensitivity, whatever its number of cells.

### Derivation

1. **Laplace.** Each answer gets noise of scale $b = |Q|/(n\varepsilon)$. One noise exceeds $b\ln(|Q|/\beta)$ in absolute value with probability $\beta/|Q|$, and by the union bound some noise does with probability at most $\beta$. Setting $b\ln(|Q|/\beta) = \alpha$ and solving for $n$ gives @eq-qr-laplace-n.
2. **Gaussian.** A normal variable exceeds $z$ standard deviations in absolute value with probability $2\Phi(-z)$, where $\Phi$ is the standard normal cumulative distribution function. By the union bound, some answer's noise exceeds $\sigma z$ with probability at most $2|Q|\Phi(-z) = \beta$, which is how $z$ was chosen. Setting $\sigma z = \alpha$ and solving for $n$ gives @eq-qr-gauss-n.

### Interpretation

The records needed grow like $|Q|$ under pure DP and like $\sqrt{|Q|}$, times a slowly growing $z$, under approximate DP. Put differently, at a fixed $n$ the error per answer grows at the same rates. For a few hundred queries on a large dataset that is fine, and nothing in this chapter beats it there. But a table with 30 yes/no attributes has $3^{30} - 1 \approx 2\times10^{14}$ marginal cells, and no dataset is large enough to answer them all with independent noise. The algorithms below pay instead for the logarithm of $|Q|$ and the logarithm of $N$.

### Example: the running example

With $n = 10{,}000$ and $\varepsilon = 1$, the 255 marginal tables give an $\ell_1$ sensitivity of $2 \times 255/n = 0.0510$, so the Laplace mechanism adds noise of scale 0.0510 to each of the 6,560 cells. Its largest error, averaged over 20 runs, is 0.497, worse than the 0.232 of the uniform histogram, which uses no data. The Gaussian mechanism at $\delta = 10^{-6}$ has $\ell_2$ sensitivity $\sqrt{2 \times 255}/n = 0.00226$. With $\mu = 0.2367$ its noise has standard deviation 0.0095, and its average largest error is 0.037. Both use the structure of the marginals. Treated as 6,560 unrelated queries, the $\ell_1$ sensitivity would be $|Q|/n = 0.656$, about 13 times larger.

### Check yourself

- Why does the $\ell_2$ sensitivity grow like $\sqrt{|Q|}$ while the $\ell_1$ sensitivity grows like $|Q|$? *(Each of the $|Q|$ answers moves by up to $1/n$. The $\ell_1$ norm adds the $|Q|$ moves, while the $\ell_2$ norm adds their squares, $|Q|/n^2$, and takes a square root.)*
- A colleague answers 6,560 marginal cells with Laplace noise of scale $1/(n\varepsilon)$ each, "because each cell has sensitivity $1/n$". What is the actual guarantee? *(Composing 6,560 steps of $\varepsilon$-DP gives $6560\varepsilon$ by basic composition, or $510\varepsilon$ if you use the structure of the 255 tables, each with $\ell_1$ sensitivity $2/n$. Each cell's noise must instead be scaled to the sensitivity of the whole vector.)*

## Multiplicative weights and its regret bound {#sec-qr-mw}

### Context

Forget privacy for two sections. The method behind this chapter comes from a prediction game. Each day, $N$ **experts** make predictions, and you must decide whom to follow before you see the outcome. In the easiest version one expert is always right, and you do not know which. Then a simple rule works: follow the majority of the experts who have never been wrong, and drop every expert who errs. Whenever you are wrong, at least half of the remaining experts were wrong too, so at least half are dropped. The perfect expert is never dropped, so you can be wrong at most $\log_2N$ times. This is the **halving algorithm**. In 20 simulated runs with 1,024 random experts and one perfect one, over 200 days, it never made more than 6 mistakes, against the bound $\log_2 1024 = 10$.

Real experts are never perfect, and dropping an expert for one mistake is too harsh. **Multiplicative weights** keeps every expert but multiplies its weight by a factor below 1 each time it does badly. The question is how much worse this can be than the best expert in hindsight, a quantity called **regret**.

### Statement

The game has $T$ rounds. In round $t$ the learner chooses a probability distribution $p_t$ over the $N$ experts. Then a **loss vector** $\ell_t$ is revealed, which gives each expert a loss between $-1$ and 1, and the learner pays the average loss under its distribution, $\langle\ell_t, p_t\rangle$. The losses may be chosen by an adversary who knows the learner's rule.

**Multiplicative weights** keeps a positive weight $w_t(x)$ on each expert $x$, and every expert starts with weight 1. A step size $\gamma$, at most $\frac12$, sets how strongly one round's loss changes a weight. In round $t$ the learner divides each weight by the total weight $W_t = \sum_xw_t(x)$ and plays the resulting distribution $p_t(x) = w_t(x)/W_t$. After the losses are revealed, it updates each weight by
$$w_{t+1}(x) = w_t(x)\big(1 - \gamma\ell_t(x)\big).$$ {#eq-qr-mw}
An expert with a positive loss loses weight, and one with a negative loss gains it.

Compare the learner with any distribution $\pi$ over the experts, fixed in hindsight. Its **regret** against $\pi$ is the learner's total loss minus the total loss of $\pi$, and it satisfies
$$\sum_{t=1}^T\langle\ell_t, p_t\rangle - \sum_{t=1}^T\langle\ell_t, \pi\rangle \le \gamma T + \frac{\ln N}{\gamma}.$$ {#eq-qr-regret}
A larger step size raises the first term and lowers the second, so the best step size makes them equal. That step size, and the regret it gives, are
$$\gamma = \sqrt{\frac{\ln N}{T}}, \qquad \text{regret} \le 2\sqrt{T\ln N}.$$
Because $\gamma$ may be at most $\frac12$, this choice needs $T \ge 4\ln N$.

### Derivation

The proof follows the total weight $W_t$. It bounds $\ln W_{T+1}$ from above by the learner's total loss and from below by any one expert's total loss, and then compares the two bounds.

1. **The total weight shrinks with the learner's loss.** Summing @eq-qr-mw over the experts, $W_{t+1} = \sum_xw_t(x) - \gamma\sum_xw_t(x)\ell_t(x)$. Dividing by $W_t$ turns the weights into $p_t$, so $W_{t+1} = W_t\big(1 - \gamma\langle\ell_t, p_t\rangle\big)$.
2. **Upper bound.** Start from $W_1 = N$ and take logarithms of the product of the factors in step 1. Each factor's $\gamma\langle\ell_t, p_t\rangle$ is at most $\frac12$, so the upper bound of Fact 2 applies:
$$\ln W_{T+1} = \ln N + \sum_t\ln\big(1 - \gamma\langle\ell_t, p_t\rangle\big) \le \ln N - \gamma\sum_t\langle\ell_t, p_t\rangle.$$
3. **Lower bound from one expert.** The total weight is at least the weight of any single expert $x$, and that weight is the product of its own factors. Each $\gamma\ell_t(x)$ lies in $[-\frac12, \frac12]$, so the lower bound of Fact 2 applies, and $\ell_t(x)^2 \le 1$:
$$\ln W_{T+1} \ge \ln w_{T+1}(x) = \sum_t\ln\big(1 - \gamma\ell_t(x)\big) \ge -\gamma\sum_t\ell_t(x) - \gamma^2T.$$
4. **Compare.** Putting steps 2 and 3 together and rearranging, $\gamma\sum_t\langle\ell_t, p_t\rangle \le \ln N + \gamma\sum_t\ell_t(x) + \gamma^2T$. Dividing by $\gamma$ gives @eq-qr-regret for the comparison with the single expert $x$.
5. **Any distribution.** The inequality of step 4 holds for every expert $x$, and its left side does not depend on $x$. Multiply it by $\pi(x)$ and add over the experts. Since the $\pi(x)$ sum to 1, the left side and the constant terms are unchanged, and $\sum_x\pi(x)\ell_t(x) = \langle\ell_t, \pi\rangle$. This is @eq-qr-regret.
6. **Choose the step size.** The bound $\gamma T + \ln N/\gamma$ is smallest where the two terms are equal, at $\gamma = \sqrt{\ln N/T}$, and there it equals $2\sqrt{T\ln N}$. Fact 2 needs $\gamma \le \frac12$, that is $T \ge 4\ln N$.

### Interpretation

The regret grows like $\sqrt T$, so the average regret per round, $2\sqrt{\ln N/T}$, goes to zero. Over many rounds the learner does as well as the best fixed distribution, against any sequence of losses, without knowing in advance which distribution that is. The number of experts enters only through $\ln N$, which is what makes the method useful with exponentially many experts. The weights also look familiar. Each factor $1 - \gamma\ell$ is close to $e^{-\gamma\ell}$ when $\gamma$ is small, so after $t$ rounds each weight is close to $e^{-\gamma\sum_s\ell_s(x)}$, and the distribution is a softmax of minus the cumulative losses. That is the shape of the exponential mechanism, with cumulative loss in place of score. The version that uses the factor $e^{-\gamma\ell}$ exactly is called Hedge. Online learning also views Hedge as follow the regularized leader (FTRL). In each round that rule plays the distribution that minimizes the total loss so far plus a penalty, and for Hedge the penalty is the negative entropy of the distribution. [DP-FTRL](../part05-dp-ml/04-matrix-mechanisms.md) uses the same view with a squared-norm penalty to train models privately.

The bound guards against the worst sequence of losses, chosen by an adversary. The sequences met in practice are rarely that hostile, as the example of the next section shows.

### Example: two experts by hand

Take $N = 2$ experts and $\gamma = \frac12$ over $T = 3$ rounds, with losses $\ell_1 = (1, 0)$, $\ell_2 = (0, 1)$ and $\ell_3 = (1, 0)$. @tbl-qr-mw-hand follows the weights.

| Round $t$ | Weights $w_t$ | Distribution $p_t$ | Loss $\ell_t$ | Learner pays $\langle\ell_t, p_t\rangle$ |
| --- | --- | --- | --- | --- |
| 1 | $(1, 1)$ | $(1/2, 1/2)$ | $(1, 0)$ | $1/2$ |
| 2 | $(1/2, 1)$ | $(1/3, 2/3)$ | $(0, 1)$ | $2/3$ |
| 3 | $(1/2, 1/2)$ | $(1/2, 1/2)$ | $(1, 0)$ | $1/2$ |

: Multiplicative weights on two experts with step size 1/2. {#tbl-qr-mw-hand}

The learner pays $\frac12 + \frac23 + \frac12 = \frac53$ in total. Expert 2 lost only 1, so the regret against it is $\frac23$. The bound @eq-qr-regret is $\frac12 \times 3 + \ln 2/\frac12 = 1.5 + 1.386 = 2.886$.

### Check yourself

- Why may the losses be chosen by an adversary who knows the learner's rule? *(Steps 1–5 hold for every sequence of losses in $[-1, 1]$. Nothing in them assumes the losses are random or independent of the weights.)*
- What goes wrong in step 3 if a loss can be 3? *(Then $\gamma\ell$ can exceed $\frac12$, and the lower bound of Fact 2 fails; with $\gamma\ell \ge 1$ a weight can even become zero or negative. Losses must be rescaled into $[-1, 1]$ first.)*

## From regret to a mistake bound: learning a histogram {#sec-qr-mistakes}

### Context

Now turn the game into an algorithm for releasing answers, still without privacy. Let the experts be the elements of the universe, so that the learner's distribution $p_t$ is a guessed histogram. Compare it with the true histogram $p$, which is one particular distribution $\pi$ over the experts. Then choose the losses so that each round's regret against $p$ is the error of the guess on one query. A regret bound becomes a bound on how many large errors the guess can make.

### Statement

Run multiplicative weights over the universe, starting from the uniform histogram. In each round, look for a query $q_t \in Q$ on which the current guess has error more than $\alpha$. If there is none, stop: the guess is within $\alpha$ on every query. Otherwise let $s_t$ be $+1$ if the guess's answer is too high and $-1$ if it is too low, and use the loss $\ell_t = s_tq_t$. With step size $\gamma = \alpha/2$, the number $U$ of rounds before the algorithm stops satisfies
$$U \le \frac{4\ln N}{\alpha^2}.$$ {#eq-qr-mistakes}
The bound does not depend on $|Q|$.

### Derivation

1. **Each round's regret is the error.** By Fact 1, $\langle\ell_t, p_t - p\rangle = s_t\big(\langle q_t, p_t\rangle - \langle q_t, p\rangle\big)$. The sign $s_t$ was chosen to make this the absolute value, so it equals the error of $p_t$ on $q_t$, which is more than $\alpha$.
2. **Total regret is large.** Adding over the $U$ rounds, the regret against $p$ is more than $\alpha U$.
3. **Total regret is small.** The losses $s_tq_t$ lie in $[-1, 1]$, and $\gamma = \alpha/2 \le \frac12$, so @eq-qr-regret with $\pi = p$, the true histogram, bounds the same regret by $\gamma U + \ln N/\gamma = \alpha U/2 + 2\ln N/\alpha$.
4. **Combine.** The regret is more than $\alpha U$ by step 2 and at most $\alpha U/2 + 2\ln N/\alpha$ by step 3, so
$$\alpha U < \frac{\alpha U}{2} + \frac{2\ln N}{\alpha}.$$
Subtracting $\alpha U/2$ from both sides and dividing by $\alpha/2$ gives $U < 4\ln N/\alpha^2$, which is @eq-qr-mistakes.

### Interpretation

The update moves the guess toward the truth on the query it got wrong. If the guess's answer is too high, the loss is $+q_t$, and the weights of the records that the query counts shrink, which lowers the answer. Why should a few such corrections fix every query at once? The regret bound gives the reason. Roughly, each correction moves the guess closer to the true histogram, as measured by the relative entropy between the two, and that distance starts at no more than $\ln N$. The number of queries never enters, because the algorithm never looks at the queries it gets right.

The algorithm plays the adversary of the prediction game against itself. Easy queries would teach the guess nothing, so it always picks one that is badly wrong. That choice, and the sign, are the only two places where it touches the data. Both depend on the data only through answers to queries, each of sensitivity $1/n$, which is what makes the algorithm easy to privatize.

The bound is very loose. It assumes that each correction only just clears $\alpha$ and that the regret is as large as @eq-qr-regret allows. In practice the first corrections fix errors several times larger than $\alpha$, and they fix many queries at once.

![Non-private multiplicative weights on the running example with $\alpha = 0.05$ and step size $\gamma = \alpha/2$. Left: the largest error over all 6,560 queries before each update, until it falls below $\alpha$. Right: the regret against the true histogram, which must exceed $\alpha$ times the number of updates, and its bound $\gamma t + \ln N/\gamma$.](figures/qr_mw.svg){#fig-qr-mw}

### Example: the running example

With $\alpha = 0.05$ the bound @eq-qr-mistakes allows $4\ln 256/0.05^2 = 8872$ corrections. The algorithm needs 148 (@fig-qr-mw). The first three queries it corrects have errors 0.232, 0.228 and 0.226, more than four times $\alpha$. After 148 rounds the regret is 16.70, against a bound of 225.5.

### Check yourself

- If the universe doubles from 256 to 512 elements, how does the bound on corrections change? *(It grows from $4\ln 256/\alpha^2$ to $4\ln 512/\alpha^2$, by the factor $\ln 512/\ln 256 = 9/8$. A doubling of the universe adds only $4\ln 2/\alpha^2$ corrections.)*
- Why does the derivation compare with the true histogram $p$ rather than with the best single expert? *(The regret bound holds for every distribution $\pi$ over the experts, and $p$ is a distribution over the universe. Its loss $\langle\ell_t, p\rangle$ is exactly the true answer times $s_t$, which is what turns regret into error.)*

## Private multiplicative weights {#sec-qr-pmw}

### Context

The algorithm of the previous section touches the data in two places: it finds a query with a large error, and it finds the sign of that error. Make both private and nothing else changes. The exponential mechanism picks a query whose error is close to the largest, and the Laplace mechanism measures the chosen query's answer, which gives the sign. The private selection can miss and the measurement can be off, so the algorithm cannot tell for sure when to stop. Instead it runs a fixed number of rounds $T$, chosen in advance, and releases the average of its guesses, which is what the analysis can bound (step 5 of the derivation). This is the structure of the **multiplicative weights exponential mechanism (MWEM)** of Hardt, Ligett & McSherry (2012), the practical form of the private multiplicative weights mechanism of Hardt & Rothblum (2010).

### Statement

**Private multiplicative weights (offline).** Input: the dataset $D$, the queries $Q$, a number of rounds $T \ge 4\ln N$ and a per-step budget $\varepsilon_0$. Start from the uniform histogram $p_1$ and set $\gamma = \sqrt{\ln N/T}$. For $t = 1, \dots, T$:

1. **Select.** Run the exponential mechanism over $Q$ with budget $\varepsilon_0$ and score $|\langle q, p_t\rangle - q(D)|$, the error of the current guess, which has sensitivity $1/n$. Call the chosen query $q_t$.
2. **Measure.** Release $m_t = q_t(D) + \mathrm{Lap}\big(1/(n\varepsilon_0)\big)$.
3. **Update.** Let $s_t = +1$ if $\langle q_t, p_t\rangle > m_t$ and $-1$ otherwise, and apply @eq-qr-mw with the loss $s_tq_t$ to get $p_{t+1}$.

Output the average histogram $\bar p = \frac1T\sum_{t=1}^Tp_t$.

**Privacy.** The algorithm is $2T\varepsilon_0$-DP by basic composition. So $\varepsilon_0 = \varepsilon/(2T)$ gives $\varepsilon$-DP, and $\varepsilon_0$ from @eq-qr-step-eps with $k = 2T$ steps gives $(\varepsilon, \delta)$-DP.

**Accuracy.** With probability at least $1 - \beta$, the largest error of the average histogram over the queries in $Q$ is at most
$$\max_{q \in Q}\big|\langle q, \bar p\rangle - q(D)\big| \le 2\sqrt{\frac{\ln N}{T}} + \frac{2}{n\varepsilon_0}\Big(\ln|Q| + 2\ln\frac{2T}\beta\Big).$$ {#eq-qr-pmw}
The first term falls with more rounds. The second grows with them, because each round gets less budget. Choosing $T$ to balance them gives two sample sizes, both written with $L = \ln|Q| + 2\ln(2T/\beta)$, which depends on $T$ only through a logarithm. With $(\varepsilon, \delta)$-DP, error $\alpha$ is guaranteed once
$$n \ge \frac{64\sqrt{\ln N\,\ln(1/\delta)}\,L}{\alpha^2\varepsilon}.$$ {#eq-qr-pmw-n}
With $\varepsilon$-DP it is guaranteed once
$$n \ge \frac{108\,L\ln N}{\alpha^3\varepsilon}.$$

### Derivation

The proof runs the regret argument of the previous section on the event where the two private steps work as intended in every round.

1. **Privacy.** In round $t$, the guess $p_t$ depends only on earlier outputs, so given the history it is a fixed histogram. Then the score $|\langle q, p_t\rangle - q(D)|$ moves by at most $1/n$ between neighbours, and the selection is $\varepsilon_0$-DP. The measurement is a Laplace mechanism on an answer of sensitivity $1/n$, so it is $\varepsilon_0$-DP too. The update and the average use only released outputs, so they are post-processing. The $2T$ steps are chosen adaptively, which the composition theorems allow ([Advanced and optimal composition](04-composition.md#sec-comp-adaptive)).
2. **The good event.** Write $\mathrm{err}_t(q) = |\langle q, p_t\rangle - q(D)|$ for the error of round $t$'s guess on query $q$. Give each of the $2T$ private steps the failure probability $\beta' = \beta/(2T)$ of Fact 3. The exponential mechanism's utility bound, with $|Q|$ candidates and sensitivity $1/n$, then says that the selection picks a query whose error is within
$$c_E = \frac{2}{n\varepsilon_0}\Big(\ln|Q| + \ln\frac{2T}\beta\Big)$$
of the largest, except with probability $\beta/(2T)$. The Laplace tail says that the measurement's noise is below
$$c_L = \frac{1}{n\varepsilon_0}\ln\frac{2T}\beta,$$
also except with probability $\beta/(2T)$. A union bound over the $2T$ steps makes all of this hold at once except with probability $\beta$.
3. **Each round's loss is nearly the largest error.** If the sign $s_t$ is right, then $\langle s_tq_t, p_t - p\rangle = \mathrm{err}_t(q_t)$, as in the previous section. If it is wrong, the measured and true answers lie on different sides of the guess's answer, so the true error is at most the noise, $\mathrm{err}_t(q_t) \le c_L$. Then the loss is $-\mathrm{err}_t(q_t)$. Because $\mathrm{err}_t(q_t) \le c_L$, the loss falls short of $\mathrm{err}_t(q_t)$ by at most $2c_L$, so it is still at least $\mathrm{err}_t(q_t) - 2c_L$. Either way, by step 2,
$$\langle s_tq_t, p_t - p\rangle \ge \mathrm{err}_t(q_t) - 2c_L \ge \max_q\mathrm{err}_t(q) - c_E - 2c_L.$$
4. **Sum and use the regret bound.** Adding over the $T$ rounds, the regret against the true histogram is at least $\sum_t\max_q\mathrm{err}_t(q) - T(c_E + 2c_L)$. By @eq-qr-regret with $\gamma = \sqrt{\ln N/T}$ it is at most $2\sqrt{T\ln N}$. Dividing by $T$,
$$\frac1T\sum_{t=1}^T\max_q\mathrm{err}_t(q) \le 2\sqrt{\frac{\ln N}{T}} + c_E + 2c_L.$$
5. **The average histogram.** For any query $q$, by Fact 1, $\langle q, \bar p\rangle - q(D) = \frac1T\sum_t\langle q, p_t - p\rangle$. Its absolute value is at most $\frac1T\sum_t\mathrm{err}_t(q)$, which is at most the left side of step 4. Finally $c_E + 2c_L = \frac{2}{n\varepsilon_0}\big(\ln|Q| + 2\ln\frac{2T}\beta\big)$, which gives @eq-qr-pmw.
6. **Sample size, approximate DP.** Take the per-step budget of @eq-qr-step-eps with $k = 2T$ steps, $\varepsilon_0 = \varepsilon/(4\sqrt{T\ln(1/\delta)})$. With it, the bound @eq-qr-pmw has the shape $A/\sqrt T + B\sqrt T$. The first term comes from the regret, with $A = 2\sqrt{\ln N}$, and the second from the noise, with $B = 8\sqrt{\ln(1/\delta)}\,L/(n\varepsilon)$. A sum of this shape is smallest where its two terms are equal, at $T = A/B$, and there it equals
$$2\sqrt{AB} = 8\big(\ln N\ln(1/\delta)\big)^{1/4}\sqrt{\frac{L}{n\varepsilon}}.$$
Setting this to $\alpha$ and solving for $n$ gives @eq-qr-pmw-n. The term $L$ depends on $T$ too, but only through a logarithm, so the chapter's script finds the best $T$ by searching over it.
7. **Sample size, pure DP.** With $\varepsilon_0 = \varepsilon/(2T)$, the second term of @eq-qr-pmw grows linearly, as $CT$ with $C = 4L/(n\varepsilon)$, while the first is still $A/\sqrt T$. Their sum is smallest where its derivative $-A/(2T^{3/2}) + C$ is zero, at $T = (A/(2C))^{2/3}$. At that $T$ the second term is half the first, so the sum is $3CT$, which works out to $3(4L\ln N/(n\varepsilon))^{1/3}$. Setting this to $\alpha$ and solving gives the pure-DP sample size.

### Interpretation

**Logarithms where the baseline had powers.** The number of queries enters @eq-qr-pmw-n only through $L$, which grows like $\ln|Q|$, and the universe only through $\sqrt{\ln N}$, where the baseline paid $|Q|$ or $\sqrt{|Q|}$. The price is in the accuracy. At a fixed $n$ the error of private multiplicative weights falls like $n^{-1/2}$ under approximate DP and like $n^{-1/3}$ under pure DP, while the error of independent noise falls like $1/n$ for a fixed set of queries. So private multiplicative weights wins only when the queries are many compared with the records. Its guaranteed error never goes below about $1/\sqrt n$, and that is no accident: below that scale, the reconstruction attack would recover the data from answers to many queries.

**How close to the best possible.** The $\ln N$ cannot be avoided under pure DP: the packing argument of [Lower bounds](10-lower-bounds.md#sec-lower-sample), a later chapter of this part, shows that releasing a histogram over $N$ cells to error $\alpha$ needs about $\ln N/(2\alpha\varepsilon)$ records. Under approximate DP, Bun, Ullman & Vadhan (2014) used the fingerprinting argument of the same chapter to show that, for some families of queries, no algorithm needs substantially fewer records than private multiplicative weights.

**The running time is proportional to $N$.** Each round updates a weight for every element of the universe. With 8 attributes that is 256 weights, but with 50 it is $2^{50}$, which is out of reach. Some cost of this kind is unavoidable in general. Ullman & Vadhan (2011) showed that, under standard cryptographic assumptions, no efficient algorithm can produce private synthetic data that is accurate for every two-way marginal. Their proof uses probabilistically checkable proof (PCP) techniques from complexity theory. Practical synthetic-data methods keep the select–measure–update loop but replace the full histogram with a compact model, and the Laplace measurement with Gaussian measurements of whole marginal tables (McKenna, Mullins, Sheldon & Miklau 2022).

**The constants are poor.** On the running example the bound @eq-qr-pmw is above 1, which says nothing about errors that cannot exceed 1. The real error is much smaller, as the example shows. The bound assumes that the selection misses by its worst amount and that every correction only just clears the error it fixes.

![Largest error over the 6,560 marginal queries of the running example, as the number of records grows, averaged over 10 runs: Laplace and Gaussian noise on every answer, using the marginals' structure, and private multiplicative weights with pure DP and the number of rounds that minimizes its bound. The dotted line is the uniform histogram, which uses no data.](figures/qr_error_vs_n.svg){#fig-qr-error-vs-n}

### Example: the running example, and the scale of the bound

With $n = 10{,}000$, $\varepsilon = 1$ and pure DP, the number of rounds that minimizes @eq-qr-pmw is $T = 38$, so each step gets $\varepsilon_0 = 1/76 = 0.0132$. @tbl-qr-pmw compares three choices of $T$, averaged over 20 runs.

| Rounds $T$ | Budget per step $\varepsilon_0$ | Largest error | Bound @eq-qr-pmw |
| --- | --- | --- | --- |
| 10 | 0.0500 | 0.098 | 1.572 |
| 38 | 0.0132 | 0.078 | 1.120 |
| 100 | 0.0050 | 0.123 | 1.486 |

: Private multiplicative weights on the running example, $n = 10{,}000$, $\varepsilon = 1$. {#tbl-qr-pmw}

The bound picks a good $T$, although its values are useless as numbers. With 38 rounds the largest error is 0.078, against 0.497 for Laplace noise on every cell and 0.037 for Gaussian noise. So on this example the Gaussian mechanism wins. @fig-qr-error-vs-n shows the trend over $n$. Private multiplicative weights beats Laplace noise throughout, but it beats Gaussian noise only below a few thousand records: 6,560 queries are not many enough.

With $\delta > 0$, advanced composition gives each step more budget once $T$ is large. At $T = 100$ and $\delta = 10^{-6}$, pure DP gives each step $\varepsilon_0 = 0.00500$. The closed form @eq-qr-step-eps gives 0.00673, although in total it spends only $0.505\varepsilon$ of the budget. The largest $\varepsilon_0$ that advanced composition allows is 0.01322. At $T = 1000$ the same three budgets, in the same order, are 0.00050, 0.00213 and 0.00418.

The guarantees separate only at a much larger scale. @tbl-qr-records gives the records each method needs for error 0.05 on every query, with probability 0.95. The setting is $\varepsilon = 1$, $\delta = 10^{-6}$, a universe of $N = 2^{20}$ records, and queries with no structure. The Laplace and SmallDB columns are pure DP, and the Gaussian and private multiplicative weights columns are $(\varepsilon, \delta)$-DP. For private multiplicative weights the table solves @eq-qr-pmw for $n$, after a search for the $T$ that makes $n$ smallest. At $10^6$ queries the search finds $T = 81{,}552$ and $L = 43.81$, and it needs $1.56\times10^7$ records. The closed form @eq-qr-pmw-n gives almost the same, $1.55\times10^7$.

| Queries $\lvert Q\rvert$ | Laplace | Gaussian | Private multiplicative weights (bound) | SmallDB (bound) |
| --- | --- | --- | --- | --- |
| $10^3$ | $1.98\times10^5$ | $1.08\times10^4$ | $1.31\times10^7$ | $6.74\times10^6$ |
| $10^6$ | $3.36\times10^8$ | $4.61\times10^5$ | $1.56\times10^7$ | $1.29\times10^7$ |
| $10^9$ | $4.74\times10^{11}$ | $1.76\times10^7$ | $1.81\times10^7$ | $1.90\times10^7$ |

: Records needed for error 0.05 on every query with probability 0.95, $\varepsilon = 1$, $\delta = 10^{-6}$, $N = 2^{20}$. {#tbl-qr-records}

The bounds for private multiplicative weights hardly move as the number of queries grows a millionfold, while the Laplace column grows a millionfold and the Gaussian column a thousandfold. The private multiplicative weights and Gaussian columns cross near $10^9$ queries (@fig-qr-sample-complexity).

![Records needed for error 0.05 on every query, with probability 0.95, at $\varepsilon = 1$ and $\delta = 10^{-6}$ over a universe of $2^{20}$ records, as the number of queries grows: Laplace and Gaussian noise on every answer, and the bounds of private multiplicative weights and SmallDB. Log–log axes.](figures/qr_sample_complexity.svg){#fig-qr-sample-complexity}

### Check yourself

- Why does the algorithm release the average of its guesses rather than the last one? *(The regret bound controls the sum over rounds of the largest errors, so it controls the average guess, through step 5. It says nothing about any single round, and the last guess can be worse than the average.)*
- Why can step 2 not reuse the selection's work, the exact error the exponential mechanism used as the chosen query's score? *(The exponential mechanism releases only the index of the query. Its scores are exact, never noised and never released, so using one would use the data without paying for it. The measurement must be a fresh Laplace answer, with its own budget.)*

## Online private multiplicative weights {#sec-qr-online}

### Context

The offline algorithm needs the whole set of queries in advance, because the exponential mechanism chooses among them. An interactive query server instead receives queries one at a time, and an analyst may choose each query after seeing the answers to the earlier ones. The sparse vector technique of [Selection](05-selection.md#sec-sel-svt) is made for this. Recall its basic form, AboveThreshold. It compares each incoming query's noisy value with one noisy threshold and answers $\bot$ ("below") or $\top$ ("above"). It halts at the first $\top$, and the $\bot$ answers before it cost nothing. Its comparisons can be wrong only for queries close to the threshold. With probability $1 - \beta$, every query it calls $\bot$ lies below the threshold plus a **margin**, and every query it calls $\top$ lies above the threshold minus that margin. For $k$ queries of sensitivity $\Delta$ at budget $\varepsilon$ the margin is $8\Delta(\ln k + \ln(2/\beta))/\varepsilon$ (@eq-sel-svt-alpha). The variant Sparse restarts it after each $\top$, up to a fixed number of $\top$ answers.

Here the query asks whether the current guess's error is above a threshold. Queries the guess already answers well are answered from the guess, for free. Only the queries that it gets wrong cost privacy, and the mistake bound says there are few of them. This is the original form of private multiplicative weights, due to Hardt & Rothblum (2010).

### Statement

**Online private multiplicative weights.** Input: the dataset, a target error $\alpha$, privacy parameters $(\varepsilon, \delta)$, and a stream of at most $k$ queries, with $k$ fixed in advance. Set the budget of updates to $c = \lceil16\ln N/\alpha^2\rceil$, the step size to $\gamma = \alpha/4$, and the guess $\hat p$ to the uniform histogram. Give each of the at most $c$ runs of AboveThreshold inside Sparse, and each of the at most $c$ measurements, the budget $\varepsilon_0$ of @eq-qr-step-eps with $2c$ steps. For each incoming query $f_i$, the score is the current guess's error on it, $g_i = |\langle f_i, \hat p\rangle - f_i(D)|$, which has sensitivity $1/n$. Run Sparse on these scores with threshold $3\alpha/4$, allowing up to $c$ answers $\top$, and act on its answer:

- If Sparse answers $\bot$, answer $\langle f_i, \hat p\rangle$ from the guess.
- If it answers $\top$, measure $m_i = f_i(D) + \mathrm{Lap}(1/(n\varepsilon_0))$, answer $m_i$, and update $\hat p$ by @eq-qr-mw with the loss $s_if_i$, where $s_i$ is the sign of $\langle f_i, \hat p\rangle - m_i$.
- After the $c$-th $\top$, halt.

With these budgets the algorithm is $(\varepsilon, \delta)$-DP.

Accuracy needs enough records to keep the noise small. Suppose that, with probability at least $1 - \beta$, every AboveThreshold run has margin at most $\alpha/4$ and every measurement has noise at most $\alpha/4$. On that event every answer is within $\alpha$ of the truth. The algorithm also never uses up its $c$ answers $\top$, so it answers the whole stream. By @eq-sel-svt-alpha with failure probability $\beta/(2c)$ per run, and the Laplace tail of Fact 3, this needs
$$n \ge \frac{32\big(\ln k + \ln(4c/\beta)\big)}{\alpha\varepsilon_0}, \qquad \varepsilon_0 = \frac{\varepsilon}{2\sqrt{4c\ln(1/\delta)}}.$$ {#eq-qr-online-n}
Because $\sqrt c$ is about $4\sqrt{\ln N}/\alpha$, the per-step budget $\varepsilon_0$ shrinks like $\alpha\varepsilon/\sqrt{\ln N\ln(1/\delta)}$, up to constants. The records needed therefore grow like $\sqrt{\ln N\ln(1/\delta)}/(\alpha^2\varepsilon)$ times a logarithm, as in the offline algorithm, but the logarithm is now of the stream length $k$ instead of $|Q|$.

### Derivation

1. **Sensitivity.** Between two updates the guess $\hat p$ is fixed, because it changes only after a measurement, which is a released output. Given the history, each score $g_i$ is the distance between a fixed number and an answer of sensitivity $1/n$, so it has sensitivity $1/n$.
2. **Privacy.** The runs of AboveThreshold and the measurements alternate: a run ends with its $\top$, then that query is measured, then the next run starts with the updated guess. So the algorithm is a sequence of at most $2c$ steps, each $\varepsilon_0$-DP and each chosen as a function of the earlier outputs. By advanced composition and @eq-qr-step-eps it is $(\varepsilon, \delta)$-DP.
3. **Answers.** On the good event, a $\bot$ means $g_i < 3\alpha/4 + \alpha/4 = \alpha$, so the guess's answer is within $\alpha$. A $\top$ is answered by the measurement, within $\alpha/4$.
4. **Updates are real corrections.** A $\top$ means $g_i \ge 3\alpha/4 - \alpha/4 = \alpha/2$. The measurement's noise is at most $\alpha/4$, less than the error, so the measured answer lies on the same side of the guess's answer as the true one, and the sign $s_i$ is right. As in the previous section, the update's regret against the true histogram is $g_i \ge \alpha/2$.
5. **Few updates.** After $U$ updates the regret is at least $\alpha U/2$, and by @eq-qr-regret with $\gamma = \alpha/4$ it is at most $\alpha U/4 + 4\ln N/\alpha$. So $\alpha U/4 \le 4\ln N/\alpha$, which is $U \le 16\ln N/\alpha^2 \le c$. The budget of $c$ answers $\top$ is never exhausted.
6. **Records.** Half of the failure probability $\beta$ goes to the $c$ runs of AboveThreshold, $\beta/(2c)$ each, and half to the $c$ measurements. With budget $\varepsilon_0$, sensitivity $1/n$ and failure probability $\beta/(2c)$, the margin of one run is $8(\ln k + \ln(4c/\beta))/(n\varepsilon_0)$, and asking for at most $\alpha/4$ gives @eq-qr-online-n. A measurement's noise exceeds $\ln(2c/\beta)/(n\varepsilon_0)$ with probability $\beta/(2c)$. That level is smaller than the margin of a run, so the condition that keeps the margin below $\alpha/4$ keeps it below $\alpha/4$ too. The union bound over the $c$ runs and the $c$ measurements gives total failure probability $\beta$.

### Interpretation

The online algorithm pays for the same corrections as the offline one, but it finds them differently. The exponential mechanism searches a known list for the worst query, while the sparse vector technique waits for a bad query to arrive. The threshold must leave room between "answered well" and "corrected": a $\bot$ only promises an error below $\alpha$, while a $\top$ promises one above $\alpha/2$. The gap between the two is what lets every update make progress.

Two features matter in practice. The analyst may choose each query after seeing the earlier answers, and the guarantee still holds, because every step in the derivation conditions on the history. This is the setting of [Adaptive data analysis](../part05-dp-ml/11-adaptive-data-analysis.md), where the same machinery protects against overfitting to a reused holdout set. And the budget $c$ is set by the worst case, so it is far larger than the number of corrections a real stream needs. The offline algorithm needed 148 corrections on the running example, against an online budget of 35,490.

### Example: a large query server

For a universe of $N = 2^{20}$ records and $\alpha = 0.05$, the budget of corrections is $c = 88{,}723$. To answer a stream of $k = 10^6$ queries at $\varepsilon = 1$ and $\delta = 10^{-6}$, with probability 0.95, @eq-qr-online-n asks for $8.39\times10^7$ records. That is about five times the offline bound for $10^6$ queries in @tbl-qr-records. The difference is the price of the worst-case budget $c$ and of the wider margin that the threshold needs.

### Check yourself

- Why is the threshold $3\alpha/4$ and not $\alpha$? *(A $\top$ must certify an error of at least $\alpha/2$, so that the update makes progress, while a $\bot$ must certify an error below $\alpha$. With margin $\alpha/4$, the threshold $3\alpha/4$ gives both. At threshold $\alpha$, a $\bot$ would only certify an error below $5\alpha/4$.)*
- Could the algorithm answer a $\top$ query with $\langle f_i, \hat p\rangle$ and skip the measurement? *(No. On a $\top$ the guess is known to be off by at least $\alpha/2$, so its answer could miss the target. The measurement is also what gives the sign for the update.)*

## SmallDB: the exponential mechanism over small datasets {#sec-qr-smalldb}

### Context

Before private multiplicative weights, Blum, Ligett & Roth (2008) were the first to answer many linear queries with error that grows only with $\ln|Q|$. Their algorithm, **SmallDB**, makes a single choice with the exponential mechanism. The candidates are not queries but whole small datasets. It is the exponential mechanism used as a learner, exactly like the classifier selection of [Selection](05-selection.md#sec-sel-apps), with datasets in place of classifiers.

### Statement

**SmallDB** runs the exponential mechanism over small datasets: every dataset $S$ of $m = \lceil2\ln(2|Q|)/\alpha^2\rceil$ records drawn from the universe. The score of a candidate $S$ is minus its largest error on the queries, $-\max_{q \in Q}|q(S) - q(D)|$, so the candidates with small errors are the likely ones. This score has sensitivity $1/n$, so SmallDB is $\varepsilon$-DP. The chosen dataset has error at most $\alpha$ on every query, with probability at least $1 - \beta$, once
$$n \ge \frac{4\big(m\ln N + \ln(1/\beta)\big)}{\alpha\varepsilon} \approx \frac{8\ln(2|Q|)\ln N}{\alpha^3\varepsilon}.$$ {#eq-qr-smalldb}

### Derivation

1. **A small dataset exists.** Draw $m$ records independently from the true histogram $p$ and call the result $S$. For one query, $q(S)$ is an average of $m$ independent values in $[0, 1]$ with mean $q(D)$, so by Hoeffding's inequality it is more than $\alpha/2$ away with probability at most $2e^{-2m(\alpha/2)^2} = 2e^{-m\alpha^2/2}$. By the union bound, some query is that far off with probability at most $2|Q|e^{-m\alpha^2/2}$. With $m > 2\ln(2|Q|)/\alpha^2$ this is below 1. So with positive probability no query is off by more than $\alpha/2$, and therefore some dataset of $m$ records has error at most $\alpha/2$ on every query. The best score is at least $-\alpha/2$.
2. **Count the candidates.** A dataset of $m$ records is a list of $m$ elements of the universe, so there are at most $N^m$ candidates, and $\ln N^m = m\ln N$.
3. **The selection is nearly the best.** By the utility bound of the exponential mechanism with $\Delta = 1/n$ and failure probability $\beta$, the chosen dataset's score is within $\frac{2}{n\varepsilon}\big(m\ln N + \ln(1/\beta)\big)$ of the best. Its error is therefore at most $\alpha/2$ plus that amount.
4. **Records.** Asking that the second amount be at most $\alpha/2$ gives @eq-qr-smalldb.

### Interpretation

SmallDB has the same logarithm of $|Q|$ as private multiplicative weights, and with pure DP it has the same $\alpha^{-3}$. It is simpler to analyze, because it makes one choice instead of $T$. But it is hopeless to run: the exponential mechanism must weigh $N^m$ candidates. Its role is to show what the exponential mechanism can achieve when running time is ignored. The private learner of [Selection](05-selection.md#sec-sel-apps) picks, from a finite class, a classifier with few training mistakes. SmallDB picks, from the finite class of small datasets, one with small errors on the queries, and the same utility bound controls both.

### Example: the running example

For the 6,560 marginal queries at $\alpha = 0.1$, a dataset of $m = 1897$ records suffices, and there are $256^{1897} \approx 10^{4568}$ candidates. @eq-qr-smalldb asks for $4.21\times10^5$ records at $\varepsilon = 1$. For a universe of $2^{20}$ and $\alpha = 0.05$, the last column of @tbl-qr-records shows that SmallDB's bound is close to that of private multiplicative weights.

### Check yourself

- Why does the existence argument in step 1 need no algorithm? *(It shows that a random sample of $m$ records has the property with positive probability, so some dataset has it. The exponential mechanism then finds a nearly as good one by trying all of them.)*
- Where does the $\alpha^{-3}$ come from? *(From $m$, which grows like $1/\alpha^2$, times the $1/(\alpha\varepsilon)$ of the exponential mechanism's error.)*

## Code: check every algorithm by computation and simulation

The script [code/query_release.py](../../code/part03-dp-foundations/query_release.py) computes every number in this chapter and draws every figure. It builds the running example's table, runs the baselines, the halving algorithm, non-private multiplicative weights and private multiplicative weights, and evaluates every bound. From the repository root:

```bash
uv run python code/part03-dp-foundations/query_release.py
```

**Output** (seed 0), summarized in @tbl-qr-code:

| Experiment | Result |
| --- | --- |
| Running example | $N = 256$, $\lvert Q\rvert = 6560$ cells in 255 tables; uniform histogram error 0.232 |
| Baselines, $n = 10{,}000$ | Laplace scale 0.0510, largest error 0.497; Gaussian $\mu = 0.2367$, standard deviation 0.0095, largest error 0.037 |
| Halving, 1,024 experts | At most 6 mistakes in 20 runs; bound 10 |
| Non-private multiplicative weights, $\alpha = 0.05$ | 148 corrections against the bound 8,872; regret 16.70 against 225.5 |
| Private multiplicative weights, $\varepsilon = 1$ | @tbl-qr-pmw; best $T = 38$ |
| Budget per step at $\delta = 10^{-6}$ | $T = 100$: 0.00500, 0.00673, 0.01322; $T = 1000$: 0.00050, 0.00213, 0.00418 |
| Records needed | @tbl-qr-records; closed form $1.55\times10^7$ at $10^6$ queries |
| SmallDB, $\alpha = 0.1$ | $m = 1897$; $10^{4568}$ candidates; $4.21\times10^5$ records |
| Online private multiplicative weights, $\alpha = 0.05$ | $c = 88{,}723$ for $N = 2^{20}$, $8.39\times10^7$ records for $k = 10^6$; $c = 35{,}490$ for $N = 256$ |

: The script's output with the default seed. {#tbl-qr-code}

[tests/test_query_release.py](../../tests/test_query_release.py) checks that the output still matches.

Three things stand out. First, the algorithms work much better than their bounds say. Multiplicative weights needs 148 corrections where the bound allows 8,872, and private multiplicative weights reaches an error of 0.078 where its bound exceeds 1. Second, the bounds still pick sensible parameters, such as the number of rounds, because they get the shape of the trade-off right even when the constants are poor. Third, the advantage of private multiplicative weights is asymptotic: on a few thousand structured queries, Gaussian noise on every answer is better. The method pays off when the queries are far more numerous than the records, which is the setting of synthetic data and of open-ended query servers.

## References

- Hardt & Rothblum, "A Multiplicative Weights Mechanism for Privacy-Preserving Data Analysis" (FOCS 2010). Private multiplicative weights in its online form, with the sparse vector technique, and the $\sqrt{\ln N}\ln|Q|/\alpha^2$ sample complexity.
- Hardt, Ligett & McSherry, "A Simple and Practical Algorithm for Differentially Private Data Release" (NeurIPS 2012). MWEM, the offline form with the exponential mechanism and an averaged output, and its experiments on real tables.
- Blum, Ligett & Roth, "A Learning Theory Approach to Non-Interactive Database Privacy" (STOC 2008; Journal of the ACM 2013). SmallDB, the first release with error logarithmic in the number of queries.
- Arora, Hazan & Kale, "The Multiplicative Weights Update Method: a Meta-Algorithm and Applications" (Theory of Computing, 2012). The regret bound of @eq-qr-regret and the many uses of the method.
- Littlestone & Warmuth, "The Weighted Majority Algorithm" (Information and Computation, 1994). The halving and weighted majority algorithms.
- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), Chapter 4. SmallDB and online private multiplicative weights, with full proofs.
- Gupta, Roth & Ullman, "Iterative Constructions and Private Data Release" (TCC 2012). The general select–measure–update framework behind both algorithms.
- Bun, Ullman & Vadhan, "Fingerprinting Codes and the Price of Approximate Differential Privacy" (STOC 2014). Lower bounds showing that private multiplicative weights is close to optimal for some query families.
- Ullman & Vadhan, "PCPs and the Hardness of Generating Private Synthetic Data" (TCC 2011). Why efficient private synthetic data for all two-way marginals is impossible in general.
- McKenna, Mullins, Sheldon & Miklau, "AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data" (VLDB 2022). The select–measure–update loop with graphical models and Gaussian measurements, as used in practice.

## Cheat sheet

A dataset is a histogram $p$ over a universe of $N$ records, and a linear query $q : X \to [0, 1]$ has answer $\langle q, p\rangle$ of sensitivity $1/n$. @tbl-qr-cheat collects the results for error $\alpha$ on every query in $Q$, with probability $1 - \beta$.

| Method | Privacy | Records needed |
| --- | --- | --- |
| Laplace on every answer | $\varepsilon$ | $\lvert Q\rvert\ln(\lvert Q\rvert/\beta)/(\alpha\varepsilon)$ |
| Gaussian on every answer | $(\varepsilon, \delta)$ | $\sqrt{\lvert Q\rvert}\,z/(\mu\alpha)$ |
| Multiplicative weights (no privacy) | — | At most $4\ln N/\alpha^2$ corrections; regret $\le \gamma T + \ln N/\gamma$ |
| Offline private multiplicative weights | $\varepsilon$ or $(\varepsilon, \delta)$ | Error $2\sqrt{\ln N/T} + \frac{2}{n\varepsilon_0}(\ln\lvert Q\rvert + 2\ln\frac{2T}\beta)$; $n \approx 64\sqrt{\ln N\ln(1/\delta)}\,L/(\alpha^2\varepsilon)$ |
| Online private multiplicative weights | $(\varepsilon, \delta)$ | $c = 16\ln N/\alpha^2$ corrections; $n \ge 32(\ln k + \ln(4c/\beta))/(\alpha\varepsilon_0)$ |
| SmallDB | $\varepsilon$ | $4(m\ln N + \ln(1/\beta))/(\alpha\varepsilon)$, $m = 2\ln(2\lvert Q\rvert)/\alpha^2$; time $N^m$ |

: The query-release results of this chapter. {#tbl-qr-cheat}

**Rules of thumb**

- For hundreds or thousands of queries, add Gaussian noise to every answer, scaled to the structure of the query set.
- Reach for private multiplicative weights when the queries far outnumber the records, or when you want a synthetic dataset that answers queries you have not thought of yet.
- Spend the budget on corrections, not on queries, because the number of rounds depends on $\ln N/\alpha^2$, never on $|Q|$.
- Release the average of the guesses, measure with fresh noise, and choose $T$ by the bound, not by looking at the data.
- The running time is proportional to the universe, so beyond a few dozen attributes, replace the histogram with a compact model.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part03-dp-foundations/06-query-release.md).
