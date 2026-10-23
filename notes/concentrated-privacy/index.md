---
title: "From Differential Privacy to Concentrated Privacy"
subtitle: "How the privacy cost of many releases adds up, derived from first principles"
author: "Technical study notes"
date: "23 October 2026"
abstract: |
  When you publish many statistics about the same people, how much privacy do they lose in total? This note answers that question step by step, for a reader who knows basic probability and nothing about privacy. It defines differential privacy, finds the random quantity that every privacy guarantee controls, called the privacy loss, and shows why many small losses add up to much less than their worst case. That observation leads to two newer definitions, concentrated and zero-concentrated differential privacy, which make the saving easy to compute. Every result is proved in the text, or cited where its proof needs more machinery than the note builds.
---

A dataset holds records about people, and a statistic computed from it, such as a count or an average, reveals a little about each of them. Differential privacy limits how much. It asks that the published output be almost as likely whether or not any one person's record is in the data. Then nobody can learn much about that person from the output, because the output would have looked much the same without them. Random noise is what makes this possible. Without noise, adding one person to a count changes the answer by exactly one, which anyone comparing two releases could see. With noise, the two possible answers overlap, and a single output cannot reliably tell them apart.

One release is rarely the end. An analyst who publishes a hundred statistics about the same people gives an attacker a hundred chances to learn something, so the real question is how the privacy cost of many releases adds up. The simplest answer is that the costs add, so that a hundred releases cost a hundred times as much as one. That answer is safe but very pessimistic, because it charges every release its worst case. This note shows that the cost of a release is a random quantity, which is usually far below its worst case and sometimes even negative. Over many releases, the total is the sum of small average costs, plus a random fluctuation that grows only with the square root of the number of releases. The saving is large when each release is very private and there are many releases.

You need only basic probability to follow the note: densities, expectations, variances, the normal distribution and Markov's inequality. Everything else is defined where it is first used. @tbl-map shows the question each section answers, and @tbl-symbols lists the symbols for reference.

| Section | The question it answers |
|:---|:---|
| @sec-loss | What does a privacy guarantee compare, output by output? |
| @sec-pure | What is the strongest guarantee, and how much does it cost over many releases? |
| @sec-approx | What guarantee does Gaussian noise give, since its loss is unbounded? |
| @sec-rare | How do you prove such a guarantee? |
| @sec-advanced | Why do many small losses add up to much less than their worst case? |
| @sec-cdp | Can the facts behind that saving become the definition of privacy? |
| @sec-zcdp | What definition makes the saving easy to compute? |
| @sec-compare | How do the four definitions compare? |
| @sec-worked | What do they give for a hundred Gaussian releases? |

: The question each section answers. {#tbl-map}

| Symbol | Meaning |
|:---|:---|
| $D$, $D'$; $D \simeq D'$ | two neighbouring datasets, which means that one contains a person's record and the other does not |
| $M$ | a mechanism, the random procedure that computes the released output |
| $f$, $\Delta f$ | the statistic to release, and its sensitivity: how much one record can change it |
| $P$, $Q$ | the distributions of the released output when the data are $D$ and $D'$ |
| $p$, $q$ | their densities |
| $L$ | the privacy loss, $\ln(p/q)$ at the released output |
| $\varepsilon$, $\delta$ | the parameters of a differential-privacy guarantee |
| $\delta'$ | a small probability that you choose yourself when you derive one guarantee from another |
| $E_{e^\varepsilon}$, $\delta(\varepsilon)$ | the hockey-stick divergence, and the smallest $\delta$ that works for a given $\varepsilon$, called the privacy profile |
| $b$, $\sigma$ | the scale of Laplace noise, and the standard deviation of Gaussian noise |
| $k$ | the number of releases |
| $\mu$, $\tau$ | the two parameters of concentrated differential privacy, a mean and a spread |
| $\xi$, $\rho$ | the two parameters of zero-concentrated differential privacy |
| $\lambda$, $\alpha$ | the argument of a moment generating function, and the order of a Rényi divergence |

: The symbols the note uses. Each is also explained where it first appears. {#tbl-symbols}

# The privacy loss {#sec-loss}

Every definition in this note compares two probability distributions of the released output, one for each of two datasets. This section introduces them, and then the single quantity that summarizes how they differ at each possible output. That quantity, the privacy loss, is what the rest of the note studies.

## Two output distributions

Take two datasets $D$ and $D'$ that are the same except that $D$ contains one person's record and $D'$ does not. Such datasets are called **neighbouring**, written $D \simeq D'$. Some authors instead call two datasets neighbouring when one record has been replaced by another. The derivations below work for either choice, but the numbers in the examples depend on it, and this note uses adding or removing a record throughout.

A **mechanism** $M$ is a randomized procedure that takes a dataset and releases an output. Because it is random, running it on $D$ gives an output drawn from a probability distribution, which we call $P$. Running it on $D'$ gives an output drawn from another distribution, $Q$. Both depend on the pair of datasets, although the notation does not show it. Privacy is about how different $P$ and $Q$ are. If they were identical, the output would reveal nothing about whether the person is in the data. If they differ a lot, it reveals a lot.

The running example of this note is a counting query. Suppose the statistic $f(D)$ is the number of people in dataset $D$ who have some medical condition. Adding or removing one person changes the count by at most one. In general, the **sensitivity** of a statistic is the most it can change between neighbouring datasets. For a statistic $f$ it is written $\Delta f$:

$$\Delta f = \max_{D \simeq D'} |f(D) - f(D')|.$$

So a count has $\Delta f = 1$. A mechanism for releasing the count adds random noise to the true count. With noise drawn from the Laplace distribution of scale $b$, which has density $e^{-|z|/b}/(2b)$, the output on dataset $D$ has density

$$p(y) = \frac{1}{2b}\exp\left(-\frac{|y - f(D)|}{b}\right).$$

This is a peaked curve centred at the true count $f(D)$. On the neighbouring dataset the output density $q$ is the same curve centred at $f(D')$ instead.

## The privacy loss

To see how much a single output reveals, imagine an observer who knows everything about the data except whether the person's record is in it. For this observer only two datasets are possible, $D$ and $D'$, and before seeing the output they hold some odds for $D$ against $D'$. Bayes' rule says how seeing the output $y$ changes those odds. The new odds are the old odds multiplied by the ratio of the two densities at $y$:

$$\frac{\Pr[D \mid y]}{\Pr[D' \mid y]} = \frac{\Pr[D]}{\Pr[D']}\cdot\frac{p(y)}{q(y)}.$$ {#eq-bayes}

The ratio $p(y)/q(y)$ is therefore exactly how much the output moves the observer's belief. We take its logarithm, because later sections combine many releases, and the logarithm turns the product of their ratios into a sum. The logarithm is called the **privacy loss** at $y$:

$$L(y) = \ln\frac{p(y)}{q(y)}.$$

A privacy loss of $0$ means the output is equally likely under both datasets and teaches the observer nothing. A loss of $1$ means that the output multiplies the odds for $D$ by $e \approx 2.7$. A negative loss moves the odds towards $D'$. If $q(y) = 0$ while $p(y) > 0$, the output is impossible under $D'$, so it proves that the data are $D$, and we set $L(y) = +\infty$.

For the counting example, take Laplace noise of scale $b = 1$, and let the true counts be $f(D) = 1$ and $f(D') = 0$. The constants $1/(2b)$ cancel in the ratio, and the logarithm of a ratio of exponentials is the difference of the exponents, so $L(y) = |y| - |y - 1|$. @tbl-laplace-loss evaluates it at a few outputs.

| Output $y$ | $-2$ | $0$ | $0.5$ | $1$ | $3$ |
|:---|---:|---:|---:|---:|---:|
| Privacy loss $L(y)$ | $-1$ | $-1$ | $0$ | $1$ | $1$ |

: The privacy loss of the Laplace mechanism with scale $1$, when the true counts are $1$ and $0$. {#tbl-laplace-loss}

An output of $1$ or more points towards $D$, an output of $0$ or less points towards $D'$, and an output halfway between says nothing. The loss never goes beyond $\pm 1$, however extreme the output. That bound is what makes this mechanism private, as @sec-pure shows.

## The privacy loss as a random variable

The output is random, so the privacy loss of the output is random too. When the data are $D$, the output $Y$ is drawn from $P$, and $L(Y)$ is a random variable. Every definition in this note is a statement about its distribution. Pure differential privacy bounds its largest value. Approximate differential privacy tolerates large values on a small part of the outputs, in a sense that @sec-rare makes precise. Concentrated differential privacy controls how large the loss is on average, and how far above its average it can stray. Each definition is also required for the swapped pair $(D', D)$, whose privacy loss is $-L$, so it constrains small values of $L$ as well.

One technical point recurs. A density can be changed on a set of outputs of probability zero without changing any probability. So a statement about $L(y)$ can only be required for "almost every" $y$, that is, except on a set of outputs with probability zero. Such a set cannot affect any guarantee.

# Pure differential privacy {#sec-pure}

The strongest natural guarantee is that the privacy loss is small at every output. This section states that guarantee, shows how much Laplace noise achieves it, and derives what it costs over many releases.

## The definition and what it says about the loss

The definition is usually stated for events rather than single outputs. An **event** $S$ is a set of possible outputs, such as "the released count is above 100".

::: {#def-pure name="Pure differential privacy"}
Let $\varepsilon \ge 0$. A mechanism is **$\varepsilon$-differentially private**, or $\varepsilon$-DP, if for every pair of neighbouring datasets $D \simeq D'$, in either order, and every event $S$,
$$\Pr_P[S] \le e^{\varepsilon}\, \Pr_Q[S].$$
:::

In words, no event can be much more likely under one dataset than under its neighbour. For small $\varepsilon$ the factor $e^\varepsilon$ is about $1 + \varepsilon$, so with $\varepsilon = 0.1$ no event becomes more than about 10% more likely. Because the definition covers both orders of the pair, it also gives $\Pr_Q[S] \le e^\varepsilon \Pr_P[S]$.

The definition talks about events, but the observer of @eq-bayes updates on a single output, through the privacy loss. The next result says that the two views agree. It is the reason the rest of the note can work with the privacy loss alone.

::: {#thm-pure-loss name="Pure DP bounds the loss"}
A mechanism is $\varepsilon$-DP if and only if, for every pair of neighbouring datasets, the privacy loss satisfies $-\varepsilon \le L(y) \le \varepsilon$ for almost every output $y$.
:::

The probability of an event is the integral of the density over it. So a bound on the density ratio at every output integrates to a bound on every event, and conversely a set of outputs where the ratio is too large is itself an event that breaks the definition.

::: {.proof}
**From the loss to events.** The upper bound $L \le \varepsilon$ says that $p(y) \le e^\varepsilon q(y)$ for almost every $y$. Integrating both sides over any event $S$ gives $\Pr_P[S] \le e^\varepsilon \Pr_Q[S]$, because a set of probability zero contributes nothing to an integral. The lower bound $L \ge -\varepsilon$ says that $q(y) \le e^\varepsilon p(y)$, and integrating it gives the other order, $\Pr_Q[S] \le e^\varepsilon \Pr_P[S]$.

**From events to the loss.** Let $B$ be the set of outputs where $p(y) > e^\varepsilon q(y)$. On $B$ the density $p$ is positive, so if $B$ has positive length, it has positive probability under $P$, and the integrand $p - e^\varepsilon q$ is positive throughout it. Its integral over $B$ is then positive:
$$\Pr_P[B] - e^\varepsilon \Pr_Q[B] = \int_B \bigl(p(y) - e^\varepsilon q(y)\bigr)\,\mathrm d y > 0.$$
That contradicts the definition for the event $B$. So $B$ has probability zero, which means that $L \le \varepsilon$ for almost every output. The same argument with $D$ and $D'$ swapped gives $L \ge -\varepsilon$.
:::

## The Laplace mechanism {#sec-laplace}

How much Laplace noise makes a statistic $\varepsilon$-DP? By @thm-pure-loss it is enough to bound the privacy loss at every output. The answer is a noise scale proportional to the sensitivity.

::: {#thm-laplace name="Laplace mechanism"}
Let $f$ have sensitivity $\Delta f$. The mechanism that releases $f(D) + Z$, with $Z$ drawn from the Laplace distribution of scale $b = \Delta f/\varepsilon$, is $\varepsilon$-DP.
:::

The proof first computes the privacy loss, then bounds it.

::: {.proof}
**The log-ratio.** On dataset $D$ the output has the Laplace density centred at $f(D)$, and on $D'$ the same density centred at $f(D')$. Both have the normalizing constant $1/(2b)$, so it cancels in the ratio. What remains is a ratio of two exponentials, whose logarithm is the difference of their exponents:
$$L(y) = \ln\frac{p(y)}{q(y)} = \frac{|y - f(D')| - |y - f(D)|}{b}.$$ {#eq-laplace-loss}
The numerator compares how far the output $y$ lies from each of the two true answers.

**Bounding it.** That difference of distances can never exceed the distance between the two answers. To see this, write $y - f(D') = \bigl(y - f(D)\bigr) + \bigl(f(D) - f(D')\bigr)$ and apply the triangle inequality to the sum:
$$|y - f(D')| \le |y - f(D)| + |f(D) - f(D')|.$$
Subtracting $|y - f(D)|$ from both sides, and using the definition of the sensitivity, gives
$$|y - f(D')| - |y - f(D)| \le |f(D) - f(D')| \le \Delta f.$$
Dividing by $b$ and substituting into @eq-laplace-loss bounds the loss at every output:
$$L(y) \le \frac{\Delta f}{b} = \varepsilon.$$
Swapping $D$ and $D'$ turns the loss into $-L$, and the same argument shows $-L \le \varepsilon$. @thm-pure-loss turns the two bounds into $\varepsilon$-DP.
:::

For the counting query, $\Delta f = 1$, so noise of scale $b = 1$ gives $1$-DP. @tbl-laplace-loss shows the bound being reached: every output at or below $0$, or at or above $1$, has a loss of exactly $\pm 1$.

## Many releases: the losses add {#sec-basic}

Now release several statistics, one after another. The analyst may choose each statistic after seeing the earlier answers, which is called **adaptive** composition. The $i$-th release is then a mechanism $M_i$ that depends on the dataset and on the earlier outputs $y_1, \dots, y_{i-1}$, which we call the **history**. The analyst's choices depend only on the history, never directly on the data, so for a given history they are the same whichever dataset is in use.

The key fact is that the privacy loss of the whole sequence of outputs is the sum of the losses of its rounds. The probability of a sequence of outputs is the probability of the first, times the probability of the second given the first, and so on. Write $p_i(y_i \mid \text{history})$ for the density of round $i$'s output given its history, on dataset $D$, and $q_i$ for the same on $D'$. Then the density of the whole sequence is a product, on each dataset:

$$p(y_1, \dots, y_k) = \prod_{i=1}^k p_i(y_i \mid y_1, \dots, y_{i-1}).$$

The logarithm of a ratio of two such products is a sum of logarithms of ratios. So the total privacy loss is the sum of the per-round losses $L_i$, each computed for its own history:

$$L = \sum_{i=1}^k L_i, \qquad L_i = \ln\frac{p_i(y_i \mid \text{history})}{q_i(y_i \mid \text{history})}.$$ {#eq-loss-sum}

::: {#thm-basic name="Basic composition"}
Suppose that for every history, the $i$-th release, viewed as a mechanism on the dataset, is $\varepsilon_i$-DP. Then the sequence of $k$ releases is $(\varepsilon_1 + \dots + \varepsilon_k)$-DP.
:::

::: {.proof}
For every history, round $i$ is $\varepsilon_i$-DP, so @thm-pure-loss gives $|L_i| \le \varepsilon_i$. By @eq-loss-sum the total loss is then at most $\varepsilon_1 + \dots + \varepsilon_k$ in absolute value, and @thm-pure-loss, applied to the whole sequence, turns that bound into the guarantee.
:::

So a hundred releases at $\varepsilon = 0.1$ each cost $\varepsilon = 10$. That is a weak guarantee: by @eq-bayes, a loss of $10$ multiplies an observer's odds by $e^{10} \approx 22{,}000$. The theorem is correct, but it charges every round its largest possible loss, all in the same direction. The actual loss of a round is random, and it is usually smaller.

The counting example shows by how much. When the data are $D$, the output is the true count $1$ plus Laplace noise of scale $1$. With probability $\tfrac12$ it lands at or above $1$, where the loss is $+1$. With probability $\tfrac12 e^{-1} = 0.184$ it lands at or below $0$, where the loss is $-1$. In between, the loss lies strictly between $-1$ and $1$, and those outputs add $0.052$ to the average. So the loss averages $0.5 - 0.184 + 0.052 = 0.368$, well below its maximum of $1$, and it is negative more than a sixth of the time. Over many rounds these smaller and opposite losses partly cancel. @sec-advanced turns that observation into a much better bound.

## Post-processing

Analysts transform published numbers: they round them, combine them and feed them into models. A guarantee is only useful if these transformations cannot undo it. They cannot, for any transformation that does not look at the data again.

Suppose you apply a function $g$ to the output. The event "$g$ of the output lands in $S$" is the same event as "the output lands in the set of points that $g$ maps into $S$". The definition applies to that set too, so $g$ of the output is $\varepsilon$-DP. A randomized $g$, whose randomness is independent of the data, is an average of deterministic functions, and averaging both sides of an inequality preserves it. The same argument works for the approximate guarantee of the next section.

# Approximate differential privacy {#sec-approx}

Pure DP needs the privacy loss to be bounded. The other standard noise distribution, the Gaussian, does not give a bounded loss. This section shows why, defines the weaker guarantee that Gaussian noise does satisfy, and says exactly what its new parameter $\delta$ measures.

## The Gaussian mechanism's privacy loss {#sec-gauss-loss}

Gaussian noise is the other standard way to release a statistic. Its privacy loss turns out to be a normal random variable, which later sections exploit. Let the mechanism release $Y = f(D) + Z$, where $Z$ is normal with mean $0$ and standard deviation $\sigma$. Write $d = f(D) - f(D')$ for the difference between the two true answers.

To compute the privacy loss, start from the two normal densities. Their normalizing constants are equal, so the log-ratio is the difference of the exponents:

$$L(Y) = \frac{(Y - f(D'))^2 - (Y - f(D))^2}{2\sigma^2}.$$

When the data are $D$, the output is $Y = f(D) + Z$. So $Y - f(D) = Z$ and $Y - f(D') = Z + d$. Substituting these and expanding $(Z + d)^2 = Z^2 + 2dZ + d^2$, the $Z^2$ terms cancel:

$$L(Y) = \frac{(Z + d)^2 - Z^2}{2\sigma^2} = \frac{d^2}{2\sigma^2} + \frac{d}{\sigma^2}\,Z.$$ {#eq-gauss-loss}

The loss is a constant plus a multiple of the normal noise $Z$, so it is itself normal. Its mean is the constant $d^2/(2\sigma^2)$. Its standard deviation is the multiplier $|d|/\sigma^2$ times the standard deviation $\sigma$ of $Z$, which is $|d|/\sigma$. For a count with $\sigma = 5$ and $d = 1$, the mean is $0.02$ and the standard deviation is $0.2$. That is a small loss on average, with typical values well inside $\pm 1$.

But a normal variable is unbounded, so there is no $\varepsilon$ that bounds this loss at every output. By @thm-pure-loss the Gaussian mechanism is not $\varepsilon$-DP for any finite $\varepsilon$. Large losses do occur, but only with tiny probability, and a useful definition has to allow for that.

## The definition

::: {#def-approx name="Approximate differential privacy"}
Let $\varepsilon \ge 0$ and $0 \le \delta \le 1$. A mechanism is **$(\varepsilon, \delta)$-DP** if for every pair of neighbouring datasets $D \simeq D'$, in either order, and every event $S$,
$$\Pr_P[S] \le e^\varepsilon \Pr_Q[S] + \delta.$$
:::

The only change from @def-pure is the additive slack $\delta$. It lets an event be more likely under $D$ than the factor $e^\varepsilon$ allows, by at most $\delta$ in probability.

The slack must be tiny, far below one over the number of people in the data. To see why, consider a mechanism that publishes each person's record in full, independently with probability $\delta$. Take an event $S$, and split it according to whether the extra person's record is published. If it is not, the output has the same distribution under $D$ and $D'$, so that part of $S$ is equally likely under both. The part where it is published is impossible under $D'$, and has probability at most $\delta$ under $D$. So $\Pr_P[S] \le \Pr_Q[S] + \delta$, and the mechanism is $(0, \delta)$-DP. Yet on a dataset of $n$ people it publishes about $n\delta$ records in full. In practice $\delta$ is therefore set to $10^{-5}$ or smaller.

## What $\delta$ measures {#sec-exact-delta}

Given a mechanism and an $\varepsilon$, what is the smallest $\delta$ that works? It is the largest amount by which any event breaks the pure guarantee, the largest value of $\Pr_P[S] - e^\varepsilon \Pr_Q[S]$. The next result computes it in terms of the privacy loss. @sec-rare uses it to explain what $\delta$ does and does not tell you, and @sec-worked uses it to find the exact guarantee of the Gaussian mechanism.

To state it, write $u_+ = \max(u, 0)$ for the positive part of a number, and $\Pr_P[L > \varepsilon]$ for the probability under $P$ of the set of outputs where the loss exceeds $\varepsilon$.

::: {#thm-hockey name="The exact δ"}
For a pair of output distributions $P$ and $Q$, the largest value of $\Pr_P[S] - e^\varepsilon \Pr_Q[S]$ over all events $S$ is
$$\int \bigl(p(y) - e^\varepsilon q(y)\bigr)_+\,\mathrm d y.$$
It can also be written in terms of the privacy loss in two ways:
$$\Pr_P[L > \varepsilon] - e^\varepsilon\, \Pr_Q[L > \varepsilon] = \mathbb E_P\Bigl[\bigl(1 - e^{\varepsilon - L}\bigr)_+\Bigr].$$
The mechanism is $(\varepsilon, \delta)$-DP exactly when this quantity is at most $\delta$ for every pair of neighbouring datasets, in both orders.
:::

The best event for breaking the inequality collects every output where $p$ exceeds $e^\varepsilon q$, and nothing else.

::: {.proof}
**The best event.** Write the quantity to maximize as an integral over the event:
$$\Pr_P[S] - e^\varepsilon \Pr_Q[S] = \int_S \bigl(p(y) - e^\varepsilon q(y)\bigr)\,\mathrm d y.$$
Adding to $S$ a set of outputs where the integrand is negative lowers the integral, and adding a set where it is positive raises it. So the integral is largest for the event where the integrand is positive, which is the event $\{L > \varepsilon\}$, and its value there is the integral of the positive part.

**The first form.** Split that integral into its two terms. The integral of $p$ over $\{L > \varepsilon\}$ is $\Pr_P[L > \varepsilon]$, and the integral of $e^\varepsilon q$ over it is $e^\varepsilon \Pr_Q[L > \varepsilon]$.

**The second form.** On the event $\{L > \varepsilon\}$ the definition of the loss gives $q = e^{-L}p$, so the integrand is $p\,(1 - e^{\varepsilon - L})$. Its integral is the average under $P$ of $1 - e^{\varepsilon - L}$ over the outputs where that factor is positive, which is $\mathbb E_P[(1 - e^{\varepsilon - L})_+]$. Outputs where $q = 0$ have $L = +\infty$ and count with factor $1$.
:::

This quantity is known as the **hockey-stick divergence**, written $E_{e^\varepsilon}(P \| Q)$, after the shape of the graph of $u \mapsto u_+$. The second form is the one to remember. It says that $\delta$ is an average over the outputs with a loss above $\varepsilon$, in which each output is discounted by the factor $1 - e^{\varepsilon - L}$. That factor is close to $0$ when the loss only just exceeds $\varepsilon$, and close to $1$ when the loss is far above it. Outputs that are impossible under $D'$ count in full.

# Proving a guarantee by showing that large losses are rare {#sec-rare}

Every guarantee in the rest of the note is proved the same way. You show that the privacy loss exceeds some value $\varepsilon$ only with a small probability, and you conclude that the mechanism is $(\varepsilon, \delta)$-DP with $\delta$ equal to that probability. This section proves that the conclusion is valid. It then shows that it only works in that direction, which is why @sec-composing-delta needs a separate argument for mechanisms that already have a $\delta$.

The probability in these arguments is not a property of the mechanism but a number you pick when you apply a theorem: a smaller one gives a smaller slack and a larger $\varepsilon$. To keep it apart from the $\delta$ of a mechanism's existing guarantee, the note writes it $\delta'$. In the final guarantee it plays the role of $\delta$.

::: {#lem-tail name="Rare large losses give approximate DP"}
Suppose that for every pair of neighbouring datasets, in both orders, the privacy loss satisfies $\Pr_P[L > \varepsilon] \le \delta'$. Then the mechanism is $(\varepsilon, \delta')$-DP.
:::

The proof splits an event into the outputs with a small loss, where the pure guarantee holds, and the rest, which is rare.

::: {.proof}
Fix an event $S$. Split it into the outputs with a loss of at most $\varepsilon$ and those with a larger loss:
$$S_{\text{small}} = S \cap \{L \le \varepsilon\}, \qquad S_{\text{large}} = S \cap \{L > \varepsilon\}.$$
On $S_{\text{small}}$ the density bound $p \le e^\varepsilon q$ holds, because that is what $L \le \varepsilon$ means. Outputs where $q = 0$ and $p > 0$ have an infinite loss, so they all lie in $S_{\text{large}}$. Integrating the density bound over $S_{\text{small}}$ gives
$$\Pr_P[S_{\text{small}}] \le e^\varepsilon \Pr_Q[S_{\text{small}}].$$
Since $S_{\text{small}}$ is part of $S$, its probability under $Q$ is at most $\Pr_Q[S]$, so $\Pr_P[S_{\text{small}}] \le e^\varepsilon \Pr_Q[S]$. The other part is rare by assumption: $\Pr_P[S_{\text{large}}] \le \Pr_P[L > \varepsilon] \le \delta'$. Adding the two bounds gives $\Pr_P[S] \le e^\varepsilon \Pr_Q[S] + \delta'$.
:::

The lemma does not work in reverse: a mechanism can be $(\varepsilon, \delta)$-DP while its loss exceeds $\varepsilon$ far more often than with probability $\delta$. The reason is the discount in @thm-hockey. A tail probability counts every output with a loss above $\varepsilon$ in full, while $\delta$ counts an output whose loss only just exceeds $\varepsilon$ at a tiny fraction of its probability.

A mechanism with only two possible outputs makes this concrete. @tbl-two-outputs gives their probabilities under the two datasets, for some small $\delta$.

| Output | Probability under $P$ | Probability under $Q$ |
|:---|:---:|:---:|
| $0$ | $\tfrac12$ | $\tfrac12 + \delta$ |
| $1$ | $\tfrac12$ | $\tfrac12 - \delta$ |

: A mechanism that is $(0, \delta)$-DP, although its loss is positive half the time. {#tbl-two-outputs}

Take $\varepsilon = 0$. With two outputs, the integral in @thm-hockey becomes a sum over them. Output $0$ contributes nothing, because it is less likely under $P$ than under $Q$. Output $1$ contributes the difference $\tfrac12 - (\tfrac12 - \delta) = \delta$. So the pair is $(0, \delta)$-DP in this order, and the same computation gives $\delta$ in the other order. Yet output $1$ is more likely under $P$, so its loss is positive, and it occurs with probability $\tfrac12$. The loss there is $\ln\bigl(\tfrac12 / (\tfrac12 - \delta)\bigr)$, only about $2\delta$, which is why @thm-hockey discounts it almost entirely. So an $(\varepsilon, \delta)$-DP guarantee does not tell you that the loss exceeds $\varepsilon$ with probability at most $\delta$, and a proof must never assume that it does.

# Advanced composition {#sec-advanced}

@thm-basic charged $k$ releases at $\varepsilon$ each a total of $k\varepsilon$, as if every round reached its largest loss and all the losses pushed in the same direction. This section shows that the total loss of all the rounds is, with high probability, much smaller.

## The idea

Think of the total loss @eq-loss-sum as a random walk with $k$ steps, each between $-\varepsilon$ and $\varepsilon$. Two facts keep the walk short.

First, each step is small on average. The average loss is never negative, because outputs that favour $D$ are more likely when the data are $D$. But Step 1 below shows that it is at most about $\varepsilon^2/2$, far below the worst case $\varepsilon$ when $\varepsilon$ is small. So the averages of the $k$ steps add up to about $k\varepsilon^2/2$.

Second, each step fluctuates around its average, and fluctuations partly cancel. A sum of $k$ independent fluctuations, each of size up to $\varepsilon$, typically has size about $\varepsilon\sqrt{k}$, not $k\varepsilon$. The rounds here need not be independent, since each may depend on the earlier outputs, but Step 3 shows that the same bound holds. Together, the total loss is at most about $k\varepsilon^2/2 + \varepsilon\sqrt{k}$, with the second term multiplied by a factor that grows slowly as the probability of failure $\delta'$ shrinks.

::: {#thm-advanced name="Advanced composition"}
Suppose that for every history, each of $k$ releases is $\varepsilon$-DP. Then for every $\delta'$ between $0$ and $1$, the sequence of releases is $(\varepsilon_{\mathrm{total}}, \delta')$-DP with
$$\varepsilon_{\mathrm{total}} = \varepsilon\sqrt{2k\ln(1/\delta')} + \frac{k\varepsilon(e^\varepsilon - 1)}{2}.$$
:::

The first term is the fluctuation, and the factor $\sqrt{2\ln(1/\delta')}$ is the slowly growing multiplier: halving $\delta'$ increases it only a little. The second term is the sum of the averages. It is close to $k\varepsilon^2/2$ when $\varepsilon$ is small, because $e^\varepsilon - 1$ is then close to $\varepsilon$. Dwork, Rothblum and Vadhan (2010) proved the theorem with a second term twice as large, and Dwork and Rothblum (2016) gave the form above.

The proof has four steps. Step 1 bounds the average loss of one round. Step 2 states the two probability tools that the rest of the proof needs. Step 3 shows that the sum of the fluctuations is rarely large. Step 4 adds the averages back and uses @lem-tail to turn the result into a guarantee.

## Step 1: the average loss of one round is small {#sec-mean}

Fix one round and its history, and write $P$ and $Q$ for that round's two output distributions. The average privacy loss under $P$ is

$$\mathbb E_P[L] = \int p(y)\ln\frac{p(y)}{q(y)}\,\mathrm d y.$$

This average is known as the **Kullback–Leibler divergence** of $P$ from $Q$, written $D_{\mathrm{KL}}(P \| Q)$. To bound it we use what $\varepsilon$-DP says about the likelihood ratio $R = p/q$, so the first move is to write the average in terms of $R$. Under $\varepsilon$-DP, $q$ is positive wherever $p$ is, so we may divide by it, and $p = qR$ turns the integral into an average under $Q$:

$$\mathbb E_P[L] = \int q(y)\,R(y)\ln R(y)\,\mathrm d y = \mathbb E_Q[R \ln R].$$

Two facts about $R$ are now available. Because the round is $\varepsilon$-DP, @thm-pure-loss puts $R$ between $r_- = e^{-\varepsilon}$ and $r_+ = e^\varepsilon$. And its average under $Q$ is one, because $\mathbb E_Q[R] = \int q\,(p/q) = \int p = 1$. The question is how large $\mathbb E_Q[R\ln R]$ can be under these two constraints.

The function $r \mapsto r\ln r$ is convex, since its second derivative is $1/r > 0$. A convex function lies below the straight line, or chord, joining two of its points. The chord from $r_-$ to $r_+$ gives each point between them the weights that place it on the segment, so for every $R$ between $r_-$ and $r_+$,

$$R\ln R \le \frac{r_+ - R}{r_+ - r_-}\,r_-\ln r_- + \frac{R - r_-}{r_+ - r_-}\,r_+\ln r_+.$$

The right-hand side is a straight line in $R$, so its average under $Q$ is the same line evaluated at the average of $R$, which is $1$:

$$\mathbb E_Q[R\ln R] \le \frac{(r_+ - 1)\,r_-\ln r_- + (1 - r_-)\,r_+\ln r_+}{r_+ - r_-}.$$

It remains to simplify this fraction. Substitute $\ln r_- = -\varepsilon$ and $\ln r_+ = \varepsilon$ in the numerator:

$$\varepsilon\bigl((1 - r_-)\,r_+ - (r_+ - 1)\,r_-\bigr) = \varepsilon\bigl(r_+ + r_- - 2\,r_- r_+\bigr) = \varepsilon\,(r_+ + r_- - 2),$$

where the last step uses $r_- r_+ = 1$. So the bound is $\varepsilon\,(r_+ + r_- - 2)/(r_+ - r_-)$. Both numerator and denominator factor neatly in terms of $s = e^{\varepsilon/2}$, because $r_+ + r_- - 2 = (s - 1/s)^2$ and $r_+ - r_- = (s - 1/s)(s + 1/s)$, as expanding the right-hand sides shows. The common factor $s - 1/s$ cancels:

$$\mathbb E_P[L] \le \varepsilon\,\frac{s - 1/s}{s + 1/s} = \varepsilon\,\frac{e^{\varepsilon/2} - e^{-\varepsilon/2}}{e^{\varepsilon/2} + e^{-\varepsilon/2}} = \varepsilon\tanh\frac{\varepsilon}{2}.$$ {#eq-mean-bound}

The last step is the definition of the hyperbolic tangent, $\tanh u = (e^u - e^{-u})/(e^u + e^{-u})$. For $u \ge 0$ it satisfies $\tanh u \le u$, so the bound is at most $\varepsilon^2/2$, the size the idea promised. Multiplying the numerator and denominator of the middle fraction by $e^{\varepsilon/2}$ gives $\tanh(\varepsilon/2) = (e^\varepsilon - 1)/(e^\varepsilon + 1)$. Since $e^\varepsilon + 1 \ge 2$, the bound is also at most $\varepsilon(e^\varepsilon - 1)/2$, the looser form that @thm-advanced uses, following the literature. For the counting example of @sec-basic, with $\varepsilon = 1$, the exact average loss is $0.368$, the bound @eq-mean-bound gives $0.462$, and the looser form gives $0.859$.

## Step 2: two probability tools

Step 3 has to show that a sum of random fluctuations is rarely large. The standard way is through the moment generating function $\mathbb E[e^{\lambda X}]$ of a random variable $X$. It suits sums because the exponential of a sum is a product of exponentials, and Step 3 shows how to average that product one factor at a time. Two facts do the work.

The first turns a bound on the moment generating function into a bound on a tail probability. For every $\lambda > 0$, the event $X \ge t$ is the same as the event $e^{\lambda X} \ge e^{\lambda t}$. Markov's inequality, applied to the positive variable $e^{\lambda X}$, bounds the probability of that event by $\mathbb E[e^{\lambda X}]/e^{\lambda t}$. This is **Chernoff's bound**:

$$\Pr[X \ge t] \le e^{-\lambda t}\,\mathbb E\bigl[e^{\lambda X}\bigr] \qquad\text{for every } \lambda > 0.$$ {#eq-chernoff}

The second bounds the moment generating function of a bounded variable with mean zero. It is **Hoeffding's lemma**, which this note cites without proof (Boucheron, Lugosi and Massart, 2013, Lemma 2.2). If $X$ has mean $0$ and always lies between two numbers $\ell$ and $h$, then for every real $\lambda$

$$\mathbb E\bigl[e^{\lambda X}\bigr] \le \exp\left(\frac{\lambda^2 (h - \ell)^2}{8}\right).$$ {#eq-hoeffding}

A normal variable with mean zero and standard deviation $(h - \ell)/2$ has moment generating function exactly equal to this bound. So the lemma says that a bounded variable has tails no heavier than that normal variable. The lemma is a statement about a single distribution, so it also applies to the conditional distribution given a history, provided the conditional mean is zero and $\ell$ and $h$ are fixed by the history.

## Step 3: the fluctuations rarely add up to much

Return to the $k$ rounds. All probabilities in this step are for the data $D$, so the outputs are drawn from $P$. Let $m_i$ be the average loss of round $i$ given its history. Step 1 bounds it, in the form @thm-advanced uses, by

$$m_i \le m_0, \qquad m_0 = \frac{\varepsilon(e^\varepsilon - 1)}{2}.$$

The fluctuation of round $i$ is $X_i = L_i - m_i$, and it has mean zero given the history. Because $L_i$ lies between $-\varepsilon$ and $\varepsilon$, and $m_i$ is a number fixed by the history, the fluctuation lies between $-\varepsilon - m_i$ and $\varepsilon - m_i$. That interval has width $2\varepsilon$ and is fixed by the history. Hoeffding's lemma @eq-hoeffding with $h - \ell = 2\varepsilon$ then gives, for every history,

$$\mathbb E\bigl[e^{\lambda X_i} \mid \text{history}\bigr] \le e^{\lambda^2\varepsilon^2/2}.$$

The rounds need not be independent, because each may depend on the outputs before it. The argument handles that by averaging over one round at a time, starting from the last. Let $S_j = X_1 + \dots + X_j$ be the sum of the first $j$ fluctuations. An average can be computed in two stages, first over round $j$ given everything before it, then over everything before it. This is the tower rule for conditional expectations. In the first stage, $S_{j-1}$ is fixed by the outputs before round $j$, so it comes out as a constant factor:

$$\mathbb E\bigl[e^{\lambda S_j}\bigr] = \mathbb E\Bigl[e^{\lambda S_{j-1}}\;\mathbb E\bigl[e^{\lambda X_j} \mid y_1, \dots, y_{j-1}\bigr]\Bigr] \le e^{\lambda^2\varepsilon^2/2}\;\mathbb E\bigl[e^{\lambda S_{j-1}}\bigr].$$

Repeating this for $j = k, k - 1, \dots, 1$ multiplies $k$ such factors:

$$\mathbb E\bigl[e^{\lambda S_k}\bigr] \le e^{k\lambda^2\varepsilon^2/2}.$$

Chernoff's bound @eq-chernoff turns this into a tail bound. For every $t > 0$ and every $\lambda > 0$,

$$\Pr[S_k \ge t] \le \exp\left(-\lambda t + \frac{k\lambda^2\varepsilon^2}{2}\right).$$

The bound holds for every positive $\lambda$, so we pick the one that makes it smallest. The exponent is a parabola in $\lambda$, smallest where its derivative $-t + k\lambda\varepsilon^2$ is zero, at $\lambda = t/(k\varepsilon^2)$. Substituting that value, the exponent becomes $-t^2/(k\varepsilon^2) + t^2/(2k\varepsilon^2) = -t^2/(2k\varepsilon^2)$. So

$$\Pr[S_k \ge t] \le \exp\left(-\frac{t^2}{2k\varepsilon^2}\right).$$

To make this probability equal to $\delta'$, choose $t = \varepsilon\sqrt{2k\ln(1/\delta')}$. This is the first term of @thm-advanced.

## Step 4: add the averages back and conclude

The total loss is the sum of the fluctuations plus the sum of the averages, and each average is at most $m_0$:

$$L = S_k + \sum_{i=1}^k m_i \le S_k + k m_0.$$

So the total loss exceeds $t + k m_0$ only if $S_k$ exceeds $t$, which by Step 3 has probability at most $\delta'$. The value $t + k m_0$ is exactly $\varepsilon_{\mathrm{total}}$, so

$$\Pr_P[L > \varepsilon_{\mathrm{total}}] \le \delta'.$$

The assumptions are the same for the swapped pair $(D', D)$, so the same bound holds in the other order. @lem-tail now gives the $(\varepsilon_{\mathrm{total}}, \delta')$-DP guarantee, which proves @thm-advanced.

## How much it saves

Take $k = 100$ releases at $\varepsilon = 0.1$, and choose $\delta' = 10^{-5}$, so that $\ln(1/\delta') = 11.513$. The first term is then $0.1\sqrt{200 \times 11.513} = 4.80$, and the second is only $0.53$. Their total of $5.32$ is about half of the $10$ that basic composition charges.

The saving grows with the number of releases, because the fluctuation grows like $\sqrt{k}$ while basic composition grows like $k$. But it shrinks as $\varepsilon$ grows, because the second term grows faster than $\varepsilon$. When $\varepsilon = 1$, for example, the second term is $k(e - 1)/2 = 0.86\,k$, almost as large as the $k$ that basic composition charges in total. @sec-worked shows a case where advanced composition is worse than basic composition for this reason.

## Rounds that already have a $\delta$ {#sec-composing-delta}

The Gaussian mechanism is not $\varepsilon$-DP for any $\varepsilon$, only $(\varepsilon, \delta)$-DP. To compose Gaussian releases with the theorems above, as @sec-worked does, you need versions of them for rounds that each have a slack $\delta_0$. They exist, and they cost an extra $k\delta_0$ in the slack:

- by basic composition, $k$ rounds that are each $(\varepsilon, \delta_0)$-DP are $(k\varepsilon,\, k\delta_0)$-DP;
- by advanced composition, they are $(\varepsilon_{\mathrm{total}},\, k\delta_0 + \delta')$-DP, with $\varepsilon_{\mathrm{total}}$ as in @thm-advanced.

The proofs above do not extend directly. Steps 1 and 3 need the loss of every round to lie between $-\varepsilon$ and $\varepsilon$. And @sec-rare showed that an $(\varepsilon, \delta_0)$ guarantee does not even imply that the loss exceeds $\varepsilon$ with probability at most $\delta_0$. The proofs instead rest on a decomposition. Call two distributions **$\varepsilon$-indistinguishable** if each event's probabilities under them are within a factor $e^\varepsilon$ of each other, as in @def-pure. Then the two output distributions of an $(\varepsilon, \delta_0)$-DP round can be written as mixtures. With probability $1 - \delta_0$ the output comes from a pair of $\varepsilon$-indistinguishable distributions, and with probability $\delta_0$ from an arbitrary pair. The arguments above then apply to the first parts, and each round's second part is charged to the slack. This note cites that decomposition rather than proving it (Dwork, Rothblum and Vadhan, 2010; Kairouz, Oh and Viswanath, 2015).

# Concentrated differential privacy {#sec-cdp}

The proof of advanced composition used only two facts about each round's loss. Its average is small, and its fluctuation around the average has tails no heavier than those of a normal variable. The worst-case bound $|L| \le \varepsilon$ served only to derive these two facts. The Gaussian mechanism has both properties directly: by @eq-gauss-loss its loss is normal, with average $d^2/(2\sigma^2)$, which is small when $\sigma$ is large. Yet its loss is unbounded. Dwork and Rothblum (2016) therefore made the two facts the definition of privacy, so that Gaussian noise is covered without a worst case.

## The definition

The second fact needs a name. A random variable $X$ is **$\tau$-subgaussian** if its fluctuation around its mean has a moment generating function no larger than that of a normal variable with mean zero and standard deviation $\tau$:

$$\mathbb E\bigl[e^{\lambda(X - \mathbb E[X])}\bigr] \le e^{\lambda^2\tau^2/2} \quad\text{for every real } \lambda.$$

A normal variable with standard deviation $\tau$ meets this with equality. And Hoeffding's lemma @eq-hoeffding says that a variable confined to an interval of width $2\varepsilon$ is $\varepsilon$-subgaussian. Requiring the bound for negative $\lambda$ as well as positive controls fluctuations below the mean as well as above it.

::: {#def-cdp name="Concentrated differential privacy"}
A mechanism is **$(\mu, \tau)$-CDP** if for every pair of neighbouring datasets, in either order, the privacy loss $L$ under $P$ has mean at most $\mu$ and is $\tau$-subgaussian.
:::

The parameter $\mu$ bounds the average loss, and $\tau$ bounds how far the loss spreads around that average. In the literature this definition is also called mean-concentrated differential privacy.

The Gaussian mechanism satisfies it. Its loss @eq-gauss-loss is normal with mean $d^2/(2\sigma^2)$ and standard deviation $|d|/\sigma$, and bounding $|d|$ by the sensitivity $\Delta f$ shows that it is $((\Delta f)^2/(2\sigma^2),\; \Delta f/\sigma)$-CDP. Every $\varepsilon$-DP mechanism, such as the Laplace mechanism, satisfies it too. Its average loss is at most $\varepsilon\tanh(\varepsilon/2)$ by @eq-mean-bound, and its loss lies between $-\varepsilon$ and $\varepsilon$, so it is $\varepsilon$-subgaussian by Hoeffding's lemma. So an $\varepsilon$-DP mechanism is $(\varepsilon\tanh(\varepsilon/2),\; \varepsilon)$-CDP. Dwork and Rothblum state the slightly weaker mean $\varepsilon(e^\varepsilon - 1)/2$.

## From CDP to approximate DP

A CDP guarantee is useful only if it says how likely large losses are, so that @lem-tail can turn it into an approximate-DP guarantee. The tail bound comes from Chernoff's bound, exactly as in Step 3 of @sec-advanced.

::: {#thm-cdp-dp name="CDP gives approximate DP"}
A $(\mu, \tau)$-CDP mechanism is, for every $\delta'$ between $0$ and $1$,
$$\bigl(\mu + \tau\sqrt{2\ln(1/\delta')},\; \delta'\bigr)\text{-DP}.$$
:::

::: {.proof}
**A tail bound for the fluctuation.** Apply Chernoff's bound @eq-chernoff to $X = L - \mathbb E_P[L]$, and use the subgaussian bound on its moment generating function. For every $t > 0$ and every $\lambda > 0$,
$$\Pr[L - \mathbb E_P[L] \ge t] \le \exp\left(-\lambda t + \frac{\lambda^2\tau^2}{2}\right).$$
The exponent is smallest at $\lambda = t/\tau^2$, where it equals $-t^2/\tau^2 + t^2/(2\tau^2) = -t^2/(2\tau^2)$. Choosing $t = \tau\sqrt{2\ln(1/\delta')}$ makes the bound equal to $\delta'$.

**From the fluctuation to the loss.** The mean $\mathbb E_P[L]$ is at most $\mu$, so a loss above $\mu + t$ is a fluctuation above $t$. Hence $\Pr_P[L > \mu + t] \le \delta'$, in both orders, and @lem-tail gives the guarantee.
:::

## Composing CDP mechanisms

What happens when $k$ rounds are each CDP? The argument of Step 3 in @sec-advanced carries over, and it gives an approximate-DP guarantee for the whole sequence. What it does not give is a CDP guarantee for the sequence, and the reason motivates @sec-zcdp.

::: {#thm-cdp-comp name="Composition of CDP"}
Suppose that for every history, round $i$ is $(\mu_i, \tau_i)$-CDP. Then for every $\lambda > 0$ the total loss satisfies
$$\mathbb E\Bigl[e^{\lambda(L - \sum_i \mu_i)}\Bigr] \le \exp\Bigl(\frac{\lambda^2}{2}\sum_i \tau_i^2\Bigr).$$
Consequently, for every $\delta'$ between $0$ and $1$, the sequence is $\bigl(\sum_i \mu_i + \sqrt{\textstyle\sum_i \tau_i^2}\,\sqrt{2\ln(1/\delta')},\; \delta'\bigr)$-DP.
:::

::: {.proof}
**One round.** Let $m_i$ be the mean of round $i$'s loss given its history, which is at most $\mu_i$. For $\lambda > 0$, replacing $m_i$ by the larger number $\mu_i$ lowers the exponent, so it can only make the average smaller. The subgaussian bound then gives
$$\mathbb E\bigl[e^{\lambda(L_i - \mu_i)} \mid \text{history}\bigr] \le \mathbb E\bigl[e^{\lambda(L_i - m_i)} \mid \text{history}\bigr] \le e^{\lambda^2\tau_i^2/2}.$$

**All rounds.** Averaging over one round at a time, as in Step 3 of @sec-advanced, multiplies these bounds and gives the first claim.

**The guarantee.** The proof of @thm-cdp-dp used the moment generating function only for $\lambda > 0$, and only after shifting the loss by a number at least as large as its mean. The first claim provides exactly that, with $\sum_i \mu_i$ in place of $\mu$ and $\sqrt{\sum_i \tau_i^2}$ in place of $\tau$. Repeating that proof gives the second claim.
:::

For $k$ identical rounds the guarantee is $k\mu + \tau\sqrt{2k\ln(1/\delta')}$. That is the sum of the averages plus a fluctuation that grows like $\sqrt{k}$, the same shape as advanced composition.

Is the sequence itself $(\sum_i \mu_i, \sqrt{\sum_i \tau_i^2})$-CDP? The proof does not show it, and the obstacle is worth understanding. @def-cdp centres the loss at its own mean and needs the bound for every $\lambda$, negative ones included. The proof's bound fails both requirements. It centres the total loss at $\sum_i \mu_i$, which is only an upper bound on its mean. And for $\lambda < 0$, replacing $m_i$ by the larger $\mu_i$ raises the exponent instead of lowering it, so the first inequality of the proof runs the wrong way. Dwork and Rothblum (2016) state a composition theorem for CDP, and Bun and Steinke (2016) discuss what makes such statements delicate. This note needs neither, because @thm-cdp-comp already gives the approximate-DP guarantee. The lesson is about which statement composes cleanly. It is a bound for positive $\lambda$ only, centred at a fixed number rather than at the mean, and zero-concentrated differential privacy makes exactly that its definition.

# Zero-concentrated differential privacy {#sec-zcdp}

Bun and Steinke (2016) define privacy by a bound on the moment generating function of the loss itself, for every positive argument, with no centring. They phrase it through a family of divergences between $P$ and $Q$, called Rényi divergences, rather than directly. The divergences are standard objects with known properties, and two of those properties come for free: guarantees survive post-processing, and pure DP appears as a limiting case. This section first shows that a Rényi divergence is the moment generating function of the privacy loss in disguise.

## Rényi divergence and the privacy loss

For a number $\alpha > 1$, called the **order**, the **Rényi divergence** of $P$ from $Q$ is

$$D_\alpha(P \| Q) = \frac{1}{\alpha - 1}\ln\int q(y)\left(\frac{p(y)}{q(y)}\right)^{\alpha}\,\mathrm d y.$$

It is set to $+\infty$ if $P$ gives positive probability to outputs that $Q$ makes impossible. Inside the integral is the likelihood ratio raised to the power $\alpha$, averaged under $Q$. The next lemma rewrites that average in terms of the privacy loss.

::: {#lem-renyi name="Rényi divergence is a moment generating function of the loss"}
For every $\alpha > 1$,
$$D_\alpha(P \| Q) = \frac{1}{\alpha - 1}\ln\mathbb E_P\bigl[e^{(\alpha - 1)L}\bigr].$$
:::

::: {.proof}
The ratio inside the integral is $p/q = e^L$, so its $\alpha$-th power is $e^{\alpha L}$, and the integral is the average of $e^{\alpha L}$ under $Q$. Split $e^{\alpha L}$ as $e^{L}\cdot e^{(\alpha - 1)L}$. Since $q\,e^{L} = q\,(p/q) = p$, the first factor turns the average under $Q$ into an average under $P$:
$$\int q\,e^{\alpha L} = \int q\,e^{L}\,e^{(\alpha - 1)L} = \int p\,e^{(\alpha - 1)L} = \mathbb E_P\bigl[e^{(\alpha - 1)L}\bigr].$$
Taking the logarithm and dividing by $\alpha - 1$ gives the claim.
:::

So the Rényi divergence of order $\alpha$ is the logarithm of the moment generating function of the loss at $\lambda = \alpha - 1$, divided by $\lambda$. The family spans the features of the loss that earlier sections used. As $\alpha$ decreases to $1$, $D_\alpha$ tends to the average loss, the Kullback–Leibler divergence of @sec-mean. As $\alpha$ grows, larger losses get more weight, and as $\alpha \to \infty$, $D_\alpha$ tends to the largest loss, the quantity that pure DP bounds. This note cites both limits rather than proving them (van Erven and Harremoës, 2014).

## The definition

::: {#def-zcdp name="Zero-concentrated differential privacy"}
Let $\xi, \rho \ge 0$. A mechanism is **$(\xi, \rho)$-zCDP** if for every pair of neighbouring datasets, in either order, and every $\alpha > 1$,
$$D_\alpha(P \| Q) \le \xi + \rho\,\alpha.$$
When $\xi = 0$ it is called **$\rho$-zCDP**.
:::

The bound is a straight line in $\alpha$, and the next calculation shows why that is the natural shape: it is the shape the Gaussian mechanism produces. Substitute @lem-renyi and write $\lambda = \alpha - 1$, so that $\alpha = 1 + \lambda$. Multiplying by $\lambda$ gives, for every $\lambda > 0$,

$$\ln\mathbb E_P\bigl[e^{\lambda L}\bigr] \le \lambda\bigl(\xi + \rho(1 + \lambda)\bigr) = \lambda(\xi + \rho) + \rho\lambda^2.$$

Moving the linear term to the left-hand side and taking exponentials,

$$\mathbb E_P\bigl[e^{\lambda(L - \xi - \rho)}\bigr] \le e^{\rho\lambda^2} \quad\text{for every } \lambda > 0.$$ {#eq-zcdp-mgf}

This is the form that @sec-cdp found to compose cleanly. It holds for $\lambda > 0$ only, and it centres the loss at the fixed number $\xi + \rho$ rather than at its mean. The right-hand side is the moment generating function $e^{\lambda^2 v/2}$ of a normal variable with variance $v = 2\rho$. So in its upper tail the loss behaves like a normal variable with mean $\xi + \rho$ and variance $2\rho$.

## The Gaussian mechanism

For the Gaussian mechanism the bound in @def-zcdp holds with equality, which makes $\rho$ a natural measure of its privacy.

::: {#thm-gauss-zcdp name="Gaussian mechanism"}
Releasing $f(D) + Z$, with $Z$ normal of mean $0$ and standard deviation $\sigma$, is $\rho$-zCDP with $\rho = (\Delta f)^2/(2\sigma^2)$, and no smaller $\rho$ works.
:::

::: {.proof}
**The moment generating function.** By @eq-gauss-loss the loss is normal with mean $d^2/(2\sigma^2)$ and variance $d^2/\sigma^2$. A normal variable with mean $m$ and variance $v$ has $\ln\mathbb E[e^{\lambda X}] = \lambda m + \lambda^2 v/2$, so here
$$\ln\mathbb E_P\bigl[e^{\lambda L}\bigr] = \frac{\lambda d^2}{2\sigma^2} + \frac{\lambda^2 d^2}{2\sigma^2} = \lambda(1 + \lambda)\,\frac{d^2}{2\sigma^2}.$$

**The divergence.** Dividing by $\lambda = \alpha - 1$, and using $1 + \lambda = \alpha$, @lem-renyi gives
$$D_\alpha(P \| Q) = \alpha\,\frac{d^2}{2\sigma^2} \le \alpha\,\frac{(\Delta f)^2}{2\sigma^2}.$$

**Both orders, and tightness.** Swapping $D$ and $D'$ only changes the sign of $d$, and the result depends on $d^2$, so the same bound holds in the other order. Equality holds for a pair with $|d| = \Delta f$, so no smaller $\rho$ works.
:::

For a count with $\sigma = 5$, each release is $0.02$-zCDP. Doubling $\sigma$ divides $\rho$ by four.

## Composition: the parameters add

::: {#thm-zcdp-comp name="Composition of zCDP"}
Suppose that for every history, round $i$ is $(\xi_i, \rho_i)$-zCDP. Then the sequence of rounds is $\bigl(\sum_i \xi_i,\; \sum_i \rho_i\bigr)$-zCDP.
:::

::: {.proof}
**One round.** Fix $\alpha > 1$ and write $\lambda = \alpha - 1$. For every history, @lem-renyi turns the definition into a bound on the moment generating function: multiply $D_\alpha \le \xi_i + \rho_i\alpha$ by $\lambda$ and take exponentials.
$$\mathbb E\bigl[e^{\lambda L_i} \mid \text{history}\bigr] \le e^{\lambda(\xi_i + \rho_i\alpha)}.$$

**All rounds.** The total loss is the sum of the round losses, by @eq-loss-sum. Averaging over one round at a time, as in Step 3 of @sec-advanced, multiplies these bounds:
$$\mathbb E_P\bigl[e^{\lambda L}\bigr] \le e^{\lambda\sum_i(\xi_i + \rho_i\alpha)}.$$

**Back to divergences.** Taking the logarithm, dividing by $\lambda$ and applying @lem-renyi to the whole sequence gives $D_\alpha \le \sum_i \xi_i + \alpha\sum_i \rho_i$. This holds for every $\alpha > 1$, and the same argument works in the other order, which is the claim.
:::

No step needed the average of a round given its history, so the obstacle of @sec-cdp does not arise. Composition is just addition.

## From zCDP to approximate DP

::: {#thm-zcdp-dp name="zCDP gives approximate DP"}
A $(\xi, \rho)$-zCDP mechanism with $\rho > 0$ is, for every $\delta'$ between $0$ and $1$,
$$\bigl(\xi + \rho + 2\sqrt{\rho\ln(1/\delta')},\; \delta'\bigr)\text{-DP}.$$
:::

::: {.proof}
**A tail bound.** Apply Chernoff's bound @eq-chernoff to $X = L - \xi - \rho$, using @eq-zcdp-mgf. For every $t > 0$ and every $\lambda > 0$,
$$\Pr_P[L > \xi + \rho + t] \le \exp\bigl(-\lambda t + \rho\lambda^2\bigr).$$

**The best $\lambda$.** The exponent is smallest where its derivative $-t + 2\rho\lambda$ is zero, at $\lambda = t/(2\rho)$, which is positive because $\rho > 0$. Its value there is $-t^2/(2\rho) + t^2/(4\rho) = -t^2/(4\rho)$.

**The guarantee.** Choosing $t = 2\sqrt{\rho\ln(1/\delta')}$ makes $t^2/(4\rho)$ equal to $\ln(1/\delta')$, so the tail probability is at most $\delta'$. This holds in both orders, and @lem-tail gives the guarantee.
:::

Composition (@thm-zcdp-comp, with every $\xi_i = 0$) followed by this conversion gives the main formula of the note. A sequence of $k$ rounds, each $\rho$-zCDP, is

$$\bigl(k\rho + 2\sqrt{k\rho\ln(1/\delta')},\; \delta'\bigr)\text{-DP}.$$ {#eq-zcdp-total}

The parameter $\rho$ adds up linearly, and the square root appears only once, in the final conversion. That conversion is not exact, because it loses in two places. Chernoff's bound overestimates the tail even at the best $\lambda$, and @lem-tail counts every output with a loss above $\varepsilon$ in full, without the discount of @thm-hockey. Sharper conversions exist (Bun and Steinke, 2016; Canonne, Kamath and Steinke, 2020).

The case $\rho = 0$, which the theorem excludes, is simpler. Then the divergence is at most $\xi$ at every order, and the limit $\alpha \to \infty$ cited above shows that the largest loss is at most $\xi$, so the mechanism is $\xi$-DP.

## Pure DP and CDP give zCDP

Both earlier definitions fit inside zCDP, so a mix of Laplace and Gaussian releases can be accounted for in one currency and converted once at the end.

::: {#thm-pure-zcdp name="Pure DP gives zCDP"}
An $\varepsilon$-DP mechanism is $(\varepsilon^2/2)$-zCDP.
:::

::: {.proof}
**The fluctuation.** The loss lies between $-\varepsilon$ and $\varepsilon$, so its fluctuation $L - \mathbb E_P[L]$ has mean zero and lies in an interval of width $2\varepsilon$. Hoeffding's lemma @eq-hoeffding bounds its moment generating function by $e^{\lambda^2\varepsilon^2/2}$. Adding back the mean, for every $\lambda > 0$,
$$\ln\mathbb E_P\bigl[e^{\lambda L}\bigr] = \lambda\,\mathbb E_P[L] + \ln\mathbb E_P\bigl[e^{\lambda(L - \mathbb E_P[L])}\bigr] \le \lambda\,\mathbb E_P[L] + \frac{\lambda^2\varepsilon^2}{2}.$$

**The mean.** By @eq-mean-bound the mean is at most $\varepsilon\tanh(\varepsilon/2)$, which is at most $\varepsilon^2/2$. Since $\lambda > 0$, substituting this upper bound keeps the inequality:
$$\ln\mathbb E_P\bigl[e^{\lambda L}\bigr] \le \frac{\varepsilon^2}{2}\bigl(\lambda + \lambda^2\bigr) = \frac{\varepsilon^2}{2}\,\lambda(1 + \lambda).$$

**The divergence.** Dividing by $\lambda = \alpha - 1$, and using $1 + \lambda = \alpha$, @lem-renyi gives $D_\alpha(P \| Q) \le \alpha\varepsilon^2/2$. The same holds in the other order.
:::

The proof needs the sharper mean bound $\varepsilon\tanh(\varepsilon/2)$. The looser form $\varepsilon(e^\varepsilon - 1)/2$ exceeds $\varepsilon^2/2$ and would give a slightly worse $\rho$.

With this theorem, the example of @sec-advanced becomes a zCDP calculation. A hundred $0.1$-DP releases are each $0.005$-zCDP, so $0.5$-zCDP in total, and @eq-zcdp-total with $\delta' = 10^{-5}$ gives $\varepsilon = 5.30$. That is slightly better than the $5.32$ of advanced composition.

::: {#thm-cdp-zcdp name="CDP gives zCDP"}
A $(\mu, \tau)$-CDP mechanism is $\bigl(\max\{0, \mu - \tau^2/2\},\; \tau^2/2\bigr)$-zCDP.
:::

::: {.proof}
**The moment generating function.** Split the loss into its mean and its fluctuation. For $\lambda > 0$, the mean is at most $\mu$ and the fluctuation is $\tau$-subgaussian, so
$$\ln\mathbb E_P\bigl[e^{\lambda L}\bigr] = \lambda\,\mathbb E_P[L] + \ln\mathbb E_P\bigl[e^{\lambda(L - \mathbb E_P[L])}\bigr] \le \lambda\mu + \frac{\lambda^2\tau^2}{2}.$$

**The divergence.** Dividing by $\lambda = \alpha - 1$, @lem-renyi gives
$$D_\alpha(P \| Q) \le \mu + (\alpha - 1)\frac{\tau^2}{2} = \Bigl(\mu - \frac{\tau^2}{2}\Bigr) + \alpha\,\frac{\tau^2}{2}.$$
@def-zcdp needs a constant term of at least zero. If $\mu < \tau^2/2$, the constant here is negative, and replacing it by $0$ keeps the bound true while meeting that requirement.
:::

For the Gaussian mechanism, $\mu = \tau^2/2 = (\Delta f)^2/(2\sigma^2)$, so this returns exactly @thm-gauss-zcdp. The converse, from zCDP to CDP, does not follow from these arguments. A zCDP bound controls the loss only through @eq-zcdp-mgf, for positive $\lambda$, while CDP needs a centred bound for every $\lambda$. Approximate DP, finally, does not fit inside zCDP at all. A mechanism with $\delta > 0$ may produce an output that is impossible under a neighbouring dataset, and then every Rényi divergence is infinite.

## Post-processing

As with differential privacy, nothing computed from a zCDP output can be less private. The reason is that a Rényi divergence can only shrink when the same transformation is applied to both outputs, a property called the data-processing inequality. This note cites it without proof (van Erven and Harremoës, 2014). Bun and Steinke (2016) show that CDP, as defined in @def-cdp, lacks this property, which is another reason to prefer zCDP.

# Comparing the four definitions {#sec-compare}

The four definitions control different features of the same random variable, the privacy loss. @tbl-compare puts them side by side, with the results of this note that support each entry.

| Definition | What it bounds | How it composes | Conversion to $(\varepsilon, \delta)$-DP | Post-processing |
|:---|:---|:---|:---|:---|
| $\varepsilon$-DP | the largest loss, $\lvert L\rvert \le \varepsilon$ | $\varepsilon$ adds (@thm-basic); advanced (@thm-advanced) | is already DP | yes |
| $(\varepsilon, \delta)$-DP | the exact $\delta$ of @thm-hockey | $\varepsilon$ and $\delta$ add; advanced with $k\delta_0$ (@sec-composing-delta, cited) | is already DP | yes |
| $(\mu, \tau)$-CDP | the mean and the spread of $L$ | a one-sided bound (@thm-cdp-comp) | @thm-cdp-dp | no (cited) |
| $(\xi, \rho)$-zCDP | every Rényi divergence, $D_\alpha \le \xi + \rho\alpha$ | $\xi$ and $\rho$ add (@thm-zcdp-comp) | @thm-zcdp-dp | yes (cited) |

: The four definitions, what each controls, and how each composes. {#tbl-compare}

Never compare an $\varepsilon$ with a $\rho$ as numbers. They measure different features of the loss, and only the conversions relate them: a $\rho$ of $2$, for example, means an $\varepsilon$ of about $11.6$ at $\delta' = 10^{-5}$, as the next section computes.

# A hundred Gaussian releases {#sec-worked}

This section applies every route of the note to one task, so that you can see how much each one gives away. A count with sensitivity $\Delta f = 1$ is released $k = 100$ times, with Gaussian noise of standard deviation $\sigma = 5$. The analyst may choose each release after seeing the earlier ones, provided each one has sensitivity at most $1$. The aim is the smallest $\varepsilon$ for which the hundred releases together are $(\varepsilon, 10^{-5})$-DP. In the conversions, that final slack is the $\delta'$ you choose, so $\delta' = 10^{-5}$ and $\ln(1/\delta') = 11.513$.

**Through zCDP.** By @thm-gauss-zcdp each release is $\rho$-zCDP with $\rho = 1/(2 \times 25) = 0.02$. By @thm-zcdp-comp the sequence is $2$-zCDP. Then @eq-zcdp-total gives

$$\varepsilon = 2 + 2\sqrt{2 \times 11.513} = 2 + 9.597 \approx 11.60.$$

**Through CDP.** Each release is $(0.02, 0.2)$-CDP. @thm-cdp-comp, with $\sum_i \mu_i = 2$ and $\sqrt{\sum_i \tau_i^2} = \sqrt{100 \times 0.04} = 2$, gives $\varepsilon = 2 + 2\sqrt{2 \times 11.513} \approx 11.60$ again. The two routes agree because for Gaussian noise $\tau^2 = 2\rho$ in every round. Then the CDP fluctuation term $\sqrt{\sum_i\tau_i^2}\,\sqrt{2\ln(1/\delta')} = \sqrt{2k\rho}\,\sqrt{2\ln(1/\delta')}$ equals the zCDP term $2\sqrt{k\rho\ln(1/\delta')}$.

**The exact answer.** @thm-hockey gives the smallest $\delta$ for each $\varepsilon$, and for Gaussian noise it has a closed form. Write $\gamma = |d|/\sigma$ for the distance between the two true answers, in units of the noise's standard deviation, and $W$ for a standard normal variable.

Under $P$, @eq-gauss-loss gives $L = \gamma^2/2 + (d/\sigma)(Z/\sigma)$, where $Z/\sigma$ is standard normal. A standard normal variable is symmetric about zero, so multiplying it by $d/\sigma$ has the same distribution as multiplying it by $\gamma$, whatever the sign of $d$. So under $P$ the loss is distributed as $\gamma^2/2 + \gamma W$.

Under $Q$, the output is $Y = f(D') + Z$, so $Y - f(D') = Z$ and $Y - f(D) = Z - d$. Substituting into the loss and expanding, the $Z^2$ terms cancel again:

$$L = \frac{Z^2 - (Z - d)^2}{2\sigma^2} = -\frac{d^2}{2\sigma^2} + \frac{d}{\sigma^2}\,Z.$$

By the same symmetry, under $Q$ the loss is distributed as $-\gamma^2/2 + \gamma W$. The first form of @thm-hockey needs the probability that the loss exceeds $\varepsilon$ under each distribution. Under $P$, that is the probability that $W$ exceeds $\varepsilon/\gamma - \gamma/2$. Under $Q$, it is the probability that $W$ exceeds $\varepsilon/\gamma + \gamma/2$. Writing $\Phi$ for the standard normal distribution function, the probability that $W$ exceeds $c$ is $\Phi(-c)$, so

$$\delta(\varepsilon) = \Phi\Bigl(\frac{\gamma}{2} - \frac{\varepsilon}{\gamma}\Bigr) - e^{\varepsilon}\,\Phi\Bigl(-\frac{\gamma}{2} - \frac{\varepsilon}{\gamma}\Bigr).$$ {#eq-gauss-exact}

Swapping $D$ and $D'$ changes only the sign of $d$, so the same formula holds in the other order. The function $\delta(\varepsilon)$ is called the **privacy profile** of the mechanism, and @eq-gauss-exact is the exact profile of the Gaussian mechanism (Balle and Wang, 2018).

When the hundred releases are fixed in advance, the total loss is a sum of $100$ independent losses, one per release. Under $P$ each of them is normal with mean $0.02$ and variance $0.04$. Means and variances of independent variables add, so the sum is normal with mean $2$ and variance $4$. That is the distribution of $\gamma^2/2 + \gamma W$ with $\gamma = 2$. Under $Q$ the same argument gives $-\gamma^2/2 + \gamma W$ with $\gamma = 2$. So the hundred releases together have exactly the loss of a single Gaussian release with $\gamma = 2$. Solving @eq-gauss-exact numerically for $\delta = 10^{-5}$ with $\gamma = 2$, by bisection, gives $\varepsilon = 10.00$. Releases chosen adaptively are no worse. That follows from the composition theorem of Gaussian differential privacy, a framework built around this exact profile. This note cites it (Dong, Roth and Su, 2022).

**Through per-release $(\varepsilon, \delta)$ guarantees.** One release has $\gamma = 0.2$, and @eq-gauss-exact gives its $\varepsilon_0$ for any per-release slack $\delta_0$. Basic composition, from @sec-composing-delta, splits the total slack evenly, $\delta_0 = 10^{-5}/100 = 10^{-7}$. That gives $\varepsilon_0 = 0.932$ per release and $93.2$ in total.

Advanced composition splits the total slack into $k\delta_0 + \delta' = 10^{-5}$. A grid search over the split finds the best at $\delta_0 = 8.5 \times 10^{-8}$ and $\delta' = 1.5 \times 10^{-6}$, since $100 \times 8.5 \times 10^{-8} + 1.5 \times 10^{-6} = 10^{-5}$. That gives $\varepsilon_0 = 0.938$ per release and $121.6$ in total. @tbl-worked collects the four answers.

| Route | Per release | Total $\varepsilon$ at $\delta = 10^{-5}$ |
|:---|:---|---:|
| Basic composition | $(0.932,\ 10^{-7})$-DP | $93.2$ |
| Advanced composition | $(0.938,\ 8.5 \times 10^{-8})$-DP, $\delta' = 1.5 \times 10^{-6}$ | $121.6$ |
| zCDP or CDP, then convert | $\rho = 0.02$ | $11.60$ |
| Exact | $\gamma = 0.2$ | $10.00$ |

: A hundred Gaussian releases of a count, with $\sigma = 5$, by each route. {#tbl-worked}

Advanced composition does worse than basic composition here. The per-release $\varepsilon_0$ is close to $1$, so its second term $k\varepsilon_0(e^{\varepsilon_0} - 1)/2 = 73.0$ is already most of the basic total, as @sec-advanced warned. The concentrated definitions avoid the problem because they never replace the Gaussian loss by a worst case. They work with its actual mean of $0.02$ and spread of $0.2$ per release, and come within 16% of the exact answer.

Every number in @tbl-worked is a guarantee, an upper bound that holds for every pair of neighbouring datasets. None of them is the loss of a particular run, which is random and usually much smaller.

# What the note proves and what it cites

Every theorem and lemma of the note is proved in the text, apart from the results below. They are cited because their proofs need more machinery than the note builds, and each is used only where the text says so:

- Hoeffding's lemma, in Step 2 of @sec-advanced and in the CDP and zCDP bounds for pure DP;
- the mixture decomposition behind composition for rounds with $\delta_0 > 0$, in @sec-composing-delta;
- the limits of the Rényi divergence as $\alpha \to 1$ and $\alpha \to \infty$, and its data-processing inequality, in @sec-zcdp;
- the failure of post-processing for CDP, in @sec-zcdp;
- the adaptive composition theorem of Gaussian differential privacy, for the exact answer in @sec-worked.

The note does not prove that CDP is closed under composition in the full sense of @def-cdp, and it does not rely on it.

# Sources

1. C. Dwork, G. N. Rothblum and S. Vadhan (2010), *Boosting and Differential Privacy*. The advanced composition theorem and the bound on the average loss. <https://privacytools.seas.harvard.edu/sites/g/files/omnuum6656/files/privacytools/files/05670947.pdf>
2. C. Dwork and G. N. Rothblum (2016), *Concentrated Differential Privacy*. The definition of CDP, the improved second term, and the analysis of Gaussian noise. <https://arxiv.org/abs/1603.01887>
3. M. Bun and T. Steinke (2016), *Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds*. The definition of zCDP, its conversions and composition, the Gaussian mechanism, and the relation to CDP. <https://arxiv.org/abs/1605.02065>
4. P. Kairouz, S. Oh and P. Viswanath (2015), *The Composition Theorem for Differential Privacy*. The optimal composition of $(\varepsilon, \delta)$ guarantees and the decomposition of an $(\varepsilon, \delta)$ pair. <https://proceedings.mlr.press/v37/kairouz15.pdf>
5. B. Balle and Y.-X. Wang (2018), *Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising*. The exact privacy profile @eq-gauss-exact. <https://arxiv.org/abs/1805.06530>
6. J. Dong, A. Roth and W. J. Su (2022), *Gaussian Differential Privacy*. The composition of Gaussian mechanisms under adaptive choice. <https://arxiv.org/abs/1905.02383>
7. C. Canonne, G. Kamath and T. Steinke (2020), *The Discrete Gaussian for Differential Privacy*. A sharper conversion from zCDP to approximate DP. <https://arxiv.org/abs/2004.00010>
8. T. van Erven and P. Harremoës (2014), *Rényi Divergence and Kullback–Leibler Divergence*. The properties of Rényi divergence, including its limits and the data-processing inequality. <https://arxiv.org/abs/1206.2459>
9. S. Boucheron, G. Lugosi and P. Massart (2013), *Concentration Inequalities: A Nonasymptotic Theory of Independence*. Hoeffding's lemma, with proof (Lemma 2.2).
