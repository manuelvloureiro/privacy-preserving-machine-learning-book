# Property inference

A hospital trains a readmission model on its 1,000 patients and shares it with a research partner. Suppose you have checked that no attack on the model can tell, much better than a coin flip, whether any one patient was in the training set. The partner can still learn whether 45% or 55% of the hospital's patients are over 65, and be right almost every time. That secret is not about a record. It is a **property** of the training set as a whole, a statistic of the dataset such as the fraction of records that belong to some subgroup. **Property inference** is the attack that recovers such a statistic from a trained model. Ateniese et al. (2015) introduced it, and Ganju et al. (2018) carried it to neural networks.

This chapter asks how well an attacker can learn a property from a model, and why it can learn a property when it cannot learn a record. The one idea is that a model behaves like a noisy average over its $n$ records. Changing one record moves an average by about $1/n$, while the randomness of which records were sampled moves it by about $1/\sqrt n$, the usual error of an average, so a single record is lost in the noise. A property is different, because raising the subgroup's share from one value $t_0$ to another value $t_1$ changes about $n|t_1 - t_0|$ records at once. Each moves the average by about $1/n$, so together they move it by about the gap $|t_1 - t_0|$, which does not shrink with $n$. Divided by the noise $1/\sqrt n$, the signal is about $\sqrt n\,|t_1 - t_0|$. So property inference gets easier with more data, while membership inference gets harder.

The running example is the hospital, whose records state each patient's age group. An attacker's **advantage** is how much more often it answers "55%" when that is right than when it is wrong. At $n = 1{,}000$, no attacker on any model trained on these records has advantage above 0.9985, and simply publishing the number of patients over 65 reaches that ceiling. On the same published number, the best attacker asking whether one patient took part has advantage 0.0126. Models trained on a synthetic population then show the practical attack. @tbl-prop-symbols lists the symbols, each defined where it first appears.

| Symbol | Meaning |
| --- | --- |
| $\mathcal D_t$; $t_0$, $t_1$, $\bar t$ | population with subgroup share $t$; the two candidate shares; their midpoint |
| $D \sim \mathcal D_t^n$; $S$ | dataset of $n$ records; its number of subgroup records |
| $\mathcal T$; $\theta$; $P_{n,t}$ | training algorithm; trained parameters; their distribution |
| $\theta(t)$; $\Sigma_1(t)$ | population fit; its noise covariance per record |
| $\gamma$; $\kappa$ | separation of two Gaussians in standard deviations; its rate in $\sqrt n\,\lvert t_1 - t_0\rvert$ |
| $\Phi$ | standard normal distribution function |
| $h$; $u_j$; $g$ | hidden units; parameters of unit $j$; a symmetry of the units |

: The symbols of this chapter. {#tbl-prop-symbols}

## The property-inference game {#sec-prop-game}

To ask whether an attacker can learn the property, you must fix who draws the data, what the attacker sees and knows, and how its guess is scored. [Threat models and privacy games](01-threat-models.md) writes attacks as games between a **challenger**, who runs the training pipeline with a hidden secret, and an **adversary**, who tries to guess the secret (Salem et al. 2023). In the property game the secret is one bit about the population that produced the training set. This section sets up that game and finds the most any attacker can win, because every attack later in the chapter is measured against that ceiling.

Records come from a population that mixes a majority group $A$ and a subgroup $B$. Write $\mathcal D_A$ and $\mathcal D_B$ for the distributions of a record from each group. A population in which a fraction $t$ of people belong to $B$ is the mixture $\mathcal D_t = (1 - t)\mathcal D_A + t\,\mathcal D_B$. To draw a record from it, first pick $B$ with probability $t$ and $A$ otherwise, then draw from that group.

::: {#def-prop-game name="Property-inference game"}
Fix a dataset size $n$, two shares $t_0 \ne t_1$, and a training algorithm $\mathcal T$ that maps a dataset to parameters and may use random choices of its own, such as its initialization. The **property-inference game** runs in four steps.

1. The challenger draws a hidden bit $b$, which is 0 or 1 with probability $\frac12$ each.
2. It draws a dataset $D \sim \mathcal D_{t_b}^n$ of $n$ independent records and trains $\theta = \mathcal T(D)$.
3. The adversary receives a view of $\theta$: the parameters (white-box), or answers to queries (black-box). It knows $\mathcal D_A$, $\mathcal D_B$, $n$, $t_0$, $t_1$ and $\mathcal T$.
4. It outputs a guess $b'$ and wins if $b' = b$.
:::

With a fair bit, an attacker who learns nothing scores exactly $\frac12$. Knowing $\mathcal D_A$, $\mathcal D_B$ and $\mathcal T$ means, in practice, holding auxiliary data from both groups and being able to train models the way the victim does. Ateniese et al. and Ganju et al. assume exactly this, and the shadow models of @sec-prop-meta rely on it. Suri & Evans (2022) study the game in this form and call it **distribution inference**.

The attack is scored as in [Threat models and privacy games](01-threat-models.md). Its **true positive rate** $\mathrm{TPR} = \Pr[b' = 1 \mid b = 1]$ is how often it says 1 when the bit is 1, and its **false positive rate** $\mathrm{FPR} = \Pr[b' = 1 \mid b = 0]$ is how often it says 1 when the bit is 0. Its **advantage** is $\mathrm{TPR} - \mathrm{FPR}$. Half the games have $b = 1$, where the attacker wins with probability $\mathrm{TPR}$, and half have $b = 0$, where it wins with probability $1 - \mathrm{FPR}$. So its accuracy is

$$\Pr[b' = b] = \tfrac12\big(1 + \mathrm{TPR} - \mathrm{FPR}\big).$$

Write $P_{n,t}$ for the distribution of $\theta$, over both the random data and the training randomness. Two results from Part 1 bound what the attacker can do. The first says that the best advantage at telling two distributions apart is their total variation (TV) distance, half the total absolute difference between their probabilities (an integral of densities for continuous views). The test that attains it says 1 wherever the second distribution is more likely than the first ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md#sec-ht-tv)).

The second result is the **data processing inequality** ([Divergences](../part01-prerequisites/03-divergences.md#sec-div-dpi)). Passing a view through any random function whose behaviour does not depend on the secret cannot increase its TV distance. Equality holds when the output is a **sufficient statistic**, meaning that the likelihood ratio of the view between the two worlds can be computed from the output alone, because the best test needs nothing else.

The model is computed from the data, so the attacker can never learn more from $\theta$ than from $D$ itself, and when each record shows its group only the count of subgroup records matters.

::: {#thm-prop-ceiling name="The ceiling on property inference"}
In the game of @def-prop-game, every adversary has
$$\mathrm{TPR} - \mathrm{FPR} \le \mathrm{TV}(P_{n,t_0}, P_{n,t_1}) \le \mathrm{TV}(\mathcal D_{t_0}^n, \mathcal D_{t_1}^n).$$ {#eq-prop-ceiling}
If in addition $\mathcal D_A$ and $\mathcal D_B$ have disjoint supports, so that no record could come from both groups, then the right-hand side equals
$$\mathrm{TV}\big(\mathrm{Bin}(n, t_0), \mathrm{Bin}(n, t_1)\big).$$ {#eq-prop-count-ceiling}
:::

::: {.proof}
The plan is to bound the advantage by the TV distance of the models, then by that of the datasets, and then to show that with visible groups the count $S$ of subgroup records carries everything the dataset says about the share.

1. *The game is a test.* The view has distribution $P_{n,t_0}$ when $b = 0$ and $P_{n,t_1}$ when $b = 1$. No test between two distributions has advantage above their TV distance, which gives the left inequality of @eq-prop-ceiling.
2. *Training is a channel.* The view is a randomized function of $D$, and its randomness does not depend on $b$. By the data processing inequality such processing cannot increase the TV distance, which gives the right inequality.
3. *Write the likelihood ratio of a dataset.* To find what in $D$ matters, compare how likely $D$ is under the two shares. Write $\mathcal D_A(x)$ and $\mathcal D_B(x)$ also for the densities of the two groups. With disjoint supports, a record $x$ from $B$ has density $t\,\mathcal D_B(x)$ under $\mathcal D_t$ (pick group $B$, then draw $x$), and a record from $A$ has $(1 - t)\mathcal D_A(x)$. The records are independent, so the density of $D$ is the product over its records. In the ratio of the density under $t_1$ to that under $t_0$, the factors $\mathcal D_A(x)$ and $\mathcal D_B(x)$ cancel. Each of the $S$ subgroup records leaves a factor $t_1/t_0$, and each of the $n - S$ others leaves $(1 - t_1)/(1 - t_0)$, so the likelihood ratio is
$$\Big(\frac{t_1}{t_0}\Big)^S\Big(\frac{1 - t_1}{1 - t_0}\Big)^{n - S}.$$
4. *Conclude that the count is sufficient.* The likelihood ratio depends on $D$ only through $S$, so $S$ is sufficient, and the data processing inequality from $D$ to $S$ holds with equality. Each record is from the subgroup with probability $t$, independently, so $S$ has law $\mathrm{Bin}(n, t)$, and the TV distance between the datasets is @eq-prop-count-ceiling.
:::

The theorem says that **releasing the count is the worst case**. When the group is visible in each record, no model trained on $D$ can leak more about the share than the count does, because the count already carries everything $D$ says about it. A model that never sees the group, like the logistic regression of @sec-prop-meta, is still bound by this ceiling, since it is computed from the same records, and @sec-prop-meta measures how far below it falls.

The secret belongs to a group, not a person, and that changes how the attack is scored. A membership attacker faces many targets, and exposing a few of them with high confidence is already a harm, so its score is the true positive rate at a low false positive rate. Here each dataset carries one secret, so the plain accuracy is the natural score.

::: {#exm-prop-hospital}
The hospital has $n = 1{,}000$ patients, and whether 45% or 55% of them are over 65 is the secret. If age group is part of each record, @eq-prop-count-ceiling gives the ceiling $\mathrm{TV}(\mathrm{Bin}(1000, 0.45), \mathrm{Bin}(1000, 0.55)) = 0.9985$ for every model trained on the data. The reason it is so close to 1 is that the count has mean 450 in one world and 550 in the other, with standard deviation $\sqrt{1000 \times 0.45 \times 0.55} \approx 15.7$. The two means are more than six standard deviations apart, so the two distributions barely overlap. An attacker on the count itself reaches the ceiling and is right more than 99.9% of the time.

At $n = 100$ the means are 45 and 55 and the standard deviation is about 5, so the ceiling falls to $0.6827$. Over 200,000 simulated games, the likelihood-ratio test on the count has advantage 0.6811, within sampling error of the formula.
:::

## Why a property leaks when no record does {#sec-prop-scaling}

The opening claimed that more data hides each record better and the property worse. This section proves it for the simplest release of all, the count $S$ of subgroup records. Two attackers look at the same count. One asks whether a target, a member of the subgroup, is in $D$. The other asks whether $D$ was drawn with share $t_0$ or $t_1$. For a count, both best advantages can be computed exactly, and they move in opposite directions as $n$ grows.

The membership attacker sees the count with or without one extra subgroup record. Step 4 of the proof of its advantage, @prp-prop-membership below, needs to know how far the distribution of a count moves when the count is shifted by one, and the answer depends on the shape of the distribution. A probability mass function is **unimodal** if it rises to a most likely value, the mode, and then falls.

::: {#lem-prop-shift name="Shifting a unimodal count by one"}
Let $X$ be an integer-valued random variable with finite support, whose probability mass function $p_k = \Pr[X = k]$ is unimodal. Then
$$\mathrm{TV}\big(\mathrm{Law}(X + 1), \mathrm{Law}(X)\big) = \max_k p_k,$$
where $\mathrm{Law}(\cdot)$ means "the distribution of".
:::

::: {.proof}
1. *Write the TV distance as steps.* The shifted variable takes the value $k$ with probability $\Pr[X + 1 = k] = p_{k-1}$. So the TV distance is half the total size of the steps between neighbouring probabilities,
$$\mathrm{TV}\big(\mathrm{Law}(X + 1), \mathrm{Law}(X)\big) = \frac12\sum_k|p_k - p_{k-1}|.$$
2. *Add the climb and the descent.* Let $k^*$ be the mode. The support is finite, so the probabilities start and end at 0. Up to $k^*$ every step goes up, so the steps add to the climb from 0 to the top, $p_{k^*}$. After $k^*$ every step goes down, so they add to the descent back to 0, also $p_{k^*}$.
3. *Halve.* The total is $2p_{k^*}$, and half of it is $p_{k^*} = \max_kp_k$. @tbl-prop-shift walks through $X \sim \mathrm{Bin}(4, \frac12)$.
:::

| $k$ | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| $16\Pr[X = k]$ | 1 | 4 | 6 | 4 | 1 | 0 |
| $16\Pr[X + 1 = k]$ | 0 | 1 | 4 | 6 | 4 | 1 |
| $16 \times$ difference | 1 | 3 | 2 | −2 | −3 | −1 |

: Shifting $X \sim \mathrm{Bin}(4, \frac12)$ by one. The positive differences add up to $6/16$, the height of the mode, and so do the negative ones. Half the total absolute difference is $\frac12 \times \frac{12}{16} = \frac{6}{16} = 0.375 = \max_k\Pr[X = k]$. {#tbl-prop-shift}

Unimodality matters, because a distribution that rises twice climbs and descends twice, so its steps add up to more than twice its peak. The next proposition shows that the membership advantage is the height of the binomial's most likely value times $1 - t$, the chance that a fresh record is not from the subgroup. That height shrinks like $1/\sqrt n$, because the count's standard deviation grows like $\sqrt n$ and its probability is spread over that many likely values.

::: {#prp-prop-membership name="Membership from a released count"}
Let the dataset hold $n - 1$ records from $\mathcal D_t$ plus one more, either a target from the subgroup or a fresh record from $\mathcal D_t$, and let the release be the count $S$ of subgroup records. Write $X \sim \mathrm{Bin}(n - 1, t)$ for the count among the $n - 1$ other records. The best advantage at telling the two cases apart is
$$\mathrm{Adv}_{\mathrm{mem}}(n) = (1 - t)\max_k\Pr[X = k] \approx \frac{1 - t}{\sqrt{2\pi(n - 1)t(1 - t)}}.$$ {#eq-prop-membership}
:::

::: {.proof}
The plan is to write the two views as distributions of the count, to notice that one of them is a mixture containing the other, and to reduce the TV distance to the effect of shifting a binomial by one.

1. *Write the two views.* With the target in $D$, the count is $S = X + 1$. With a fresh record, the count is $X + 1$ when the fresh record is from the subgroup, which happens with probability $t$, and $X$ otherwise.
2. *See the second view as a mixture.* So the law of the second view is $t\,\mathrm{Law}(X + 1) + (1 - t)\,\mathrm{Law}(X)$, and a fraction $t$ of the time the non-member world looks exactly like the member world.
3. *Mixing scales the TV distance.* Write $P = \mathrm{Law}(X + 1)$ and $Q = \mathrm{Law}(X)$, with mass functions $p$ and $q$. In the TV distance between $P$ and $tP + (1 - t)Q$, the part $tp(x)$ cancels from each term, which leaves a factor $1 - t$:
$$\mathrm{TV}\big(P, tP + (1 - t)Q\big) = \frac12\sum_x(1 - t)|p(x) - q(x)| = (1 - t)\,\mathrm{TV}(P, Q).$$
The best advantage is this TV distance, so it is $(1 - t)\,\mathrm{TV}(\mathrm{Law}(X + 1), \mathrm{Law}(X))$.
4. *Shift the binomial.* A binomial mass function is unimodal, so @lem-prop-shift turns the TV distance into $\max_k\Pr[X = k]$, which gives the exact form of @eq-prop-membership.
5. *Estimate the height of the mode.* By the normal approximation of [Probability distributions](../part01-prerequisites/01-distributions.md), $X$ is close to a normal with the same mean and variance $(n - 1)t(1 - t)$. A normal density with variance $s^2$ has height $1/\sqrt{2\pi s^2}$ at its peak. The probability of the most likely integer is close to that height, because the integers are spaced one apart and the density is nearly flat over a unit interval at the top. That gives the approximate form. @sec-apx-prob-stirling in the appendices proves both facts and bounds the error of the second.
:::

The property attacker sees the same count, but its two worlds centre the fraction $S/n$ on $t_0$ and on $t_1$, with spread about $\sqrt{t(1 - t)/n}$ at share $t$. Its advantage therefore depends on how many spreads apart the two centres are, and that number grows like $\sqrt n$.

::: {#prp-prop-property name="The property from a released count"}
Let the release be the count $S$ of a dataset drawn from $\mathcal D_{t_0}^n$ or $\mathcal D_{t_1}^n$, with the groups visible, and let $\bar t = \frac{t_0 + t_1}2$ be the midpoint share. The best advantage is
$$\mathrm{Adv}_{\mathrm{prop}}(n) = \mathrm{TV}\big(\mathrm{Bin}(n, t_0), \mathrm{Bin}(n, t_1)\big) \approx 2\Phi\Big(\frac{\sqrt n\,|t_1 - t_0|}{2\sqrt{\bar t(1 - \bar t)}}\Big) - 1.$$ {#eq-prop-property}
:::

::: {.proof}
1. *Start from the exact answer.* The count is sufficient (@thm-prop-ceiling), so the best advantage is the TV distance between the two binomials.
2. *Approximate by two normals.* By the normal approximation, $S/n$ is close to $\mathcal N(t, t(1 - t)/n)$. The two variances are close when the gap is small, so replace both by the one at the midpoint, $\bar t(1 - \bar t)/n$. Measured in units of that standard deviation, the two means are a distance
$$\gamma = \frac{\sqrt n\,|t_1 - t_0|}{\sqrt{\bar t(1 - \bar t)}}$$
apart, so the problem is two unit-variance normals with means $\gamma$ apart.
3. *Use the Gaussian pair.* For two unit-variance normals with means $\gamma$ apart, the densities cross halfway between the means, and the best advantage is $2\Phi(\gamma/2) - 1$ ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md#sec-ht-tv)). Substituting $\gamma$ gives @eq-prop-property.
:::

The two results say that **more data protects records and exposes properties**. The membership advantage falls like $1/\sqrt n$, a line of slope $-\frac12$ on the log axes of @fig-prop-count-leak (right). The property advantage climbs to 1 once $\sqrt n\,|t_1 - t_0|$ is a few times $\sqrt{\bar t(1 - \bar t)}$, sooner for a larger gap. Neither argument is special to counts, because most fitted models behave like averages of their records, as @sec-prop-meta shows.

The reason is that **a property is a group of records**. Moving from $t_0$ to $t_1$ changes about $n|t_1 - t_0|$ records, which for the hospital is $1{,}000 \times 0.1 = 100$ of its 1,000 patients. A protection measured one record at a time says little about a change to 100 records at once, and [Part 3](../part03-dp-foundations/02-properties.md) makes this precise with group privacy, which extends a guarantee for one record to a group at a cost that grows with the group's size. To hide the property, a released count must carry noise of about $n|t_1 - t_0|$, the distance between its values in the two worlds, and that much noise also hides the count the release was for.

![Left: the released fraction $S/n$ at $n = 1{,}000$ under $t = 0.45$ and $t = 0.55$, and under $t = 0.5$ with and without one target from the subgroup; the last two curves are indistinguishable. Right: the best property advantage for two gaps, and the best membership advantage of one record, against $n$.](figures/prop_count_leak.svg){#fig-prop-count-leak}

::: {#exm-prop-two-attackers}
@tbl-prop-count gives both advantages for the hospital's shares $t_0 = 0.45$ and $t_1 = 0.55$, and for membership at $t = 0.5$. You can check the first row by hand. For the property, $\bar t = 0.5$, so $\sqrt{\bar t(1 - \bar t)} = 0.5$ and the argument of $\Phi$ in @eq-prop-property is $\sqrt{100} \times 0.1/(2 \times 0.5) = 1$. Then $2\Phi(1) - 1 = 2(0.8413) - 1 = 0.6827$. For membership, @eq-prop-membership gives $0.5/\sqrt{2\pi \times 99 \times 0.25} = 0.5/12.47 = 0.0401$. Both approximations are within 1% of the exact values.

At $n = 10{,}000$, an attacker who wants to know whether one patient is in the data has advantage 0.0040, so with a fair bit it is right with probability $\frac12 + \frac12(0.0040) = 0.502$. An attacker who wants the share of patients over 65 is never wrong to four decimals. The data, the release and the attacker's knowledge are the same, and only the question differs.
:::

| $n$ | Property, exact | Property, @eq-prop-property | Membership, exact | Membership, @eq-prop-membership |
| --- | --- | --- | --- | --- |
| 100 | 0.6827 | 0.6827 | 0.0398 | 0.0401 |
| 1,000 | 0.9985 | 0.9984 | 0.0126 | 0.0126 |
| 10,000 | 1.0000 | 1.0000 | 0.0040 | 0.0040 |

: Best advantages from a released count of subgroup records. Property: $t_0 = 0.45$ against $t_1 = 0.55$. Membership: one subgroup record, population share $t = 0.5$. {#tbl-prop-count}

## Shadow models and meta-classifiers {#sec-prop-meta}

For a real model nobody can write down $P_{n,t}$, the distribution of the trained parameters, so the best test of @thm-prop-ceiling is out of reach. Ateniese et al. (2015) gave the recipe every later attack uses: if you cannot compute the two distributions, sample from them. This section states that recipe, and then answers the two questions it leaves open, which classifier to train and how accurate it can be.

::: {#def-prop-shadow name="Shadow-model attack"}
The attacker draws many datasets of $n$ records from $\mathcal D_{t_0}$ and from $\mathcal D_{t_1}$, using auxiliary data, and trains a **shadow model** on each with the victim's algorithm $\mathcal T$. It labels each shadow model's parameters with the bit that produced them, trains a classifier on these labelled parameter vectors, the **meta-classifier**, and applies it to the victim's model.
:::

This is the shadow-model idea of [Membership inference](02-membership-inference.md), with whole models as the data points. To say which meta-classifier is best, you need the distribution of the parameters, and for models fitted by minimizing an average loss it is approximately Gaussian.

Let $\theta$ minimize the average loss $\frac1n\sum_i\ell(\theta; x_i)$ over the training records, and let $\theta(t)$ minimize the expected loss under $\mathcal D_t$, the model you would get from infinitely many records. How noisy $\theta$ is around $\theta(t)$ depends on two things. A sharply curved loss pins the minimizer down, and records whose gradients point in different directions pull it about.

::: {#prp-prop-mest name="Fitted parameters are approximately Gaussian"}
Let the average curvature of the loss be $H = \mathbb E_{\mathcal D_t}\nabla^2\ell(\theta(t); x)$ and the second moment of the per-record gradients $G = \mathbb E_{\mathcal D_t}\big[\nabla\ell(\theta(t); x)\nabla\ell(\theta(t); x)^\top\big]$, with $H$ invertible, and define the **sandwich covariance** $\Sigma_1(t) = H^{-1}GH^{-1}$, named for $G$ sitting between two copies of $H^{-1}$. Under the regularity conditions of van der Vaart (1998, ch. 5), for large $n$,
$$\theta \approx \mathcal N\big(\theta(t), \Sigma_1(t)/n\big).$$
:::

::: {.proof}
This is the asymptotic normality of **M-estimators**, estimators defined as the minimizer of an average loss. The plan is to write the fit as a linear function of an average of per-record gradients, which is approximately Gaussian. Van der Vaart gives the conditions for each approximation, and @sec-apx-opt-mest proves them for logistic regression.

1. *The fit solves an average equation.* At a minimizer the gradient is zero, so $\frac1n\sum_i\nabla\ell(\theta; x_i) = 0$.
2. *Linearize around $\theta(t)$.* For large $n$, $\theta$ is close to $\theta(t)$, so a first-order Taylor expansion of the gradient is accurate. The sample average of the Hessians is close to its expectation $H$ by the law of large numbers, so $0 \approx \frac1n\sum_i\nabla\ell(\theta(t); x_i) + H(\theta - \theta(t))$. Solving for $\theta$,
$$\theta - \theta(t) \approx -H^{-1}\,\frac1n\sum_i\nabla\ell(\theta(t); x_i).$$
So the fit is an average, as in @sec-prop-scaling, in which each record pulls by its gradient scaled by the inverse curvature.
3. *The average gradient has mean zero.* $\theta(t)$ minimizes the expected loss, so the expected gradient there is zero. Each gradient therefore has mean zero, so its second moment $G$ is its covariance, and an average of $n$ independent copies has covariance $G/n$.
4. *Apply the central limit theorem.* The average gradient is approximately $\mathcal N(0, G/n)$. Multiplying a Gaussian vector by the symmetric matrix $-H^{-1}$ multiplies its covariance by $H^{-1}$ on both sides, which gives $\mathcal N(0, H^{-1}GH^{-1}/n)$.
:::

Under the two hypotheses, then, the parameters are two Gaussians, whose covariances are nearly equal when the gap is small. Step 2 of the next theorem's proof needs the best test between two such Gaussians, and the following lemma gives it. In $d$ dimensions the noise can be larger in some directions than in others, so a shift along a noisy direction counts for less than the same shift along a quiet one. The **Mahalanobis distance** measures a shift in exactly this way, in standard deviations along its own direction.

::: {#lem-prop-gauss name="Two Gaussians with a shared covariance"}
Let $P = \mathcal N(m_0, \Sigma)$ and $Q = \mathcal N(m_1, \Sigma)$ on $\mathbb R^d$ with $\Sigma$ invertible, with densities $p$ and $q$, and let
$$\gamma = \sqrt{(m_1 - m_0)^\top\Sigma^{-1}(m_1 - m_0)}$$
be their Mahalanobis distance. Then the log-likelihood ratio is linear in $\theta$,
$$\ln\frac{q(\theta)}{p(\theta)} = \omega^\top\Big(\theta - \frac{m_0 + m_1}2\Big), \qquad \omega = \Sigma^{-1}(m_1 - m_0),$$
and the best test with a fair bit has accuracy
$$\Phi(\gamma/2).$$ {#eq-prop-gauss-acc}
:::

::: {.proof}
The plan is to write out the two log-densities, let the quadratic parts cancel because the covariances are equal, and find the distribution of what is left.

1. *Expand the two quadratics.* A Gaussian log-density is $-\frac12(\theta - m)^\top\Sigma^{-1}(\theta - m)$ plus a constant that depends only on $\Sigma$, so the constants cancel in $\ln q - \ln p$. Multiplying out, the terms $\theta^\top\Sigma^{-1}\theta$ appear with opposite signs and cancel too, which leaves $(m_1 - m_0)^\top\Sigma^{-1}\theta - \frac12(m_1^\top\Sigma^{-1}m_1 - m_0^\top\Sigma^{-1}m_0)$.
2. *Factor the constant.* Because $\Sigma^{-1}$ is symmetric, the cross terms $m_1^\top\Sigma^{-1}m_0$ and $m_0^\top\Sigma^{-1}m_1$ are equal, so the bracketed constant factors as $(m_1 - m_0)^\top\Sigma^{-1}(m_1 + m_0)$, and with its factor $-\frac12$ it becomes $-(m_1 - m_0)^\top\Sigma^{-1}\frac{m_0 + m_1}2$. Pulling out $\omega^\top = (m_1 - m_0)^\top\Sigma^{-1}$ gives the linear form.
3. *Find its distribution.* Call the linear form $\Lambda$. As a linear function of a Gaussian it is Gaussian. Under $Q$ its mean is $\omega^\top\frac{m_1 - m_0}2 = \frac{\gamma^2}2$, by the definition of $\gamma$, and under $P$ the same computation gives $-\frac{\gamma^2}2$. Under both, its variance is $\omega^\top\Sigma\omega = (m_1 - m_0)^\top\Sigma^{-1}(m_1 - m_0) = \gamma^2$.
4. *Compute the accuracy.* The best test says 1 where $q > p$, that is, where $\Lambda > 0$. Under $Q$ the threshold 0 lies $\frac{\gamma^2/2}{\gamma} = \frac\gamma2$ standard deviations below the mean, so the test is right with probability $\Phi(\gamma/2)$. Under $P$ it is right when $\Lambda < 0$, which by symmetry has the same probability. Averaging the two worlds gives @eq-prop-gauss-acc.
:::

Putting the lemma together with @prp-prop-mest gives the accuracy of the best attack on a fitted model, and says that the best meta-classifier is linear.

::: {#thm-prop-meta name="The best attack on a fitted model"}
Suppose the parameters satisfy @prp-prop-mest under both hypotheses, and that the gap is small enough to replace $\Sigma_1(t_0)$ and $\Sigma_1(t_1)$ by their average $\bar\Sigma_1$. Then the best attack is a linear test in $\theta$, and its accuracy is
$$\Phi(\gamma/2), \qquad \gamma = \sqrt{n\,\big(\theta(t_1) - \theta(t_0)\big)^\top\bar\Sigma_1^{-1}\big(\theta(t_1) - \theta(t_0)\big)}.$$ {#eq-prop-delta}
For a small gap, with $\theta'(\bar t) = \frac{\mathrm d\theta}{\mathrm dt}$ the slope of the population fit at the midpoint,
$$\gamma \approx \kappa\sqrt n\,|t_1 - t_0|, \qquad \kappa = \sqrt{\theta'(\bar t)^\top\Sigma_1(\bar t)^{-1}\theta'(\bar t)}.$$ {#eq-prop-kappa}
:::

::: {.proof}
1. *Two Gaussians.* Under the two hypotheses $\theta$ is approximately $\mathcal N(\theta(t_b), \bar\Sigma_1/n)$, for $b = 0, 1$.
2. *Apply the lemma.* @lem-prop-gauss with $m_b = \theta(t_b)$ and $\Sigma = \bar\Sigma_1/n$ says the best test is linear, with accuracy $\Phi(\gamma/2)$. Since $\Sigma^{-1} = n\,\bar\Sigma_1^{-1}$, the factor $n$ comes out of the Mahalanobis distance, which gives @eq-prop-delta.
3. *Linearize in the share.* For a small gap the population fit changes almost linearly in $t$, so $\theta(t_1) - \theta(t_0) \approx \theta'(\bar t)(t_1 - t_0)$, and the covariance hardly changes, so $\bar\Sigma_1 \approx \Sigma_1(\bar t)$. Substituting both into @eq-prop-delta gives $\gamma^2 \approx n(t_1 - t_0)^2\,\theta'(\bar t)^\top\Sigma_1(\bar t)^{-1}\theta'(\bar t)$, whose square root is @eq-prop-kappa.
:::

The count of @sec-prop-scaling fits the same mould, which checks the theorem against an exact answer.

::: {#cor-prop-count name="The released fraction"}
For the released fraction $\theta = S/n$, the constant is $\kappa = 1/\sqrt{\bar t(1 - \bar t)}$, which is 2 at $\bar t = \frac12$, and the advantage $2\Phi(\gamma/2) - 1$ is @eq-prop-property.
:::

::: {.proof}
1. *The fraction is a fitted model.* Take the loss $\ell(\theta; x) = \frac12(\theta - a)^2$, where $a \in \{0, 1\}$ marks the subgroup. The minimizer of its average is the mean of the marks, $\theta = S/n$, and its population version is $\theta(t) = t$, so $\theta' = 1$.
2. *Compute the sandwich.* The second derivative of the loss is $H = 1$. The gradient $t - a$ has variance $G = t(1 - t)$, the variance of a $\mathrm{Bern}(t)$ variable. So $\Sigma_1 = t(1 - t)$, and @eq-prop-kappa gives $\kappa = 1/\sqrt{\bar t(1 - \bar t)}$.
3. *Turn accuracy into advantage.* Advantage is twice accuracy minus 1, so $\Phi(\gamma/2)$ becomes $2\Phi(\gamma/2) - 1$, which with this $\kappa$ is @eq-prop-property.
:::

**Two numbers set the leak.** The slope $\theta'$ says how much the property moves what the model learns, and $\Sigma_1$ says how noisy that is. If the property does not affect what the model fits, then $\theta' = 0$ and, to first order, nothing leaks however large $n$ is. The constant also compares any model with the count. By @cor-prop-count the count on $n'$ records has $\gamma = \sqrt{n'}\,|t_1 - t_0|/\sqrt{\bar t(1 - \bar t)}$. Setting this equal to the model's $\kappa\sqrt n\,|t_1 - t_0|$ and solving for $n'$ shows that the model leaks as much as the count on $n' = n(\kappa\sqrt{\bar t(1 - \bar t)})^2$ records, which is how Suri & Evans (2022) measure leakage.

The Gaussian picture fails where its approximations do. With small $n$ the parameters are skewed, and the linear meta-classifier is no longer optimal. The attacker's own sampling adds a second shortfall, because a meta-classifier trained on few shadow models estimates the direction $\omega$ with error. The meta-classifier then falls short of $\Phi(\gamma/2)$ most when the true effect is small, since then the noise in the shadow models swamps the shift it is looking for.

::: {#exm-prop-logistic}
Each record has two features $x \sim \mathcal N(0, I_2)$ and a label. The majority labels with $\Pr[y = 1 \mid x] = \sigma(2x_1 + x_2)$ and the subgroup with $\sigma(2x_1 - x_2)$, where $\sigma(z) = 1/(1 + e^{-z})$ is the logistic function. Think of a symptom $x_2$ that raises risk in most patients and lowers it in a minority. **The group is never a feature**, so the model sees $(x_1, x_2, y)$ only. A logistic regression with weights $(\theta_1, \theta_2)$ and an intercept is trained on $n$ records by Newton's method, with a tiny ridge penalty $10^{-3}$. The chapter's script (@sec-prop-code) computes $\theta(t)$, shown in @tbl-prop-theta, and $\Sigma_1(t)$ exactly, by Gauss–Hermite quadrature over $x$ (a numerical rule for integrals against a Gaussian).

The property moves $\theta_2$. At $t = \frac12$ the two rules cancel in $\theta_2$, because half the population pushes it up and half down. As $t$ grows, the subgroup's rule wins and $\theta_2$ goes negative, with slope, computed by the script, $\theta'(\frac12) = (0, -1.704, 0)$ for $(\theta_1, \theta_2, \text{intercept})$. Mixing the rules also shrinks $\theta_1$ from 2 to 1.704, but symmetrically in $t$, so $\theta_1$ is flat at $\frac12$ and carries no information there. The intercept is 0 for every $t$, so it carries none either.

The noise $\Sigma_1(\frac12)$ has diagonal $(12.59, 6.94, 6.04)$ and zero off-diagonal entries. Flipping the sign of $x_2$ swaps the two groups' rules and leaves the population at $t = \frac12$ unchanged, which makes $\theta_2$ uncorrelated with the rest. Flipping the sign of $x$ and swapping the labels flips only the intercept, which makes it uncorrelated with the weights. Two thousand models trained on $n = 1{,}000$ records at $t = \frac12$ test this. Their variances, times $n$, are $(11.92, 6.80, 6.08)$, and their mean $\theta_1$ is 1.693, so the Gaussian picture is right to within about 5%. Most of the difference in the variance of $\theta_1$ comes from the ridge penalty, which the population fit omits; including it lowers the predicted value from 12.59 to 11.83 (@sec-apx-opt-mest).

Only $\theta_2$ has a nonzero slope and $\Sigma_1$ is diagonal, so @eq-prop-kappa reduces to the slope of $\theta_2$ over its standard deviation, $\kappa = 1.704/\sqrt{6.94} = 0.647$, against 2 for the count (@cor-prop-count). By the comparison with the count above, the model leaks the property as well as $(0.647/2)^2 = 0.105$ of its records would, about one record in ten.
:::

| $t$ | 0 | 0.25 | 0.5 | 0.75 | 1 |
| --- | --- | --- | --- | --- | --- |
| $\theta_1(t)$ | 2.000 | 1.767 | 1.704 | 1.767 | 2.000 |
| $\theta_2(t)$ | 1.000 | 0.442 | 0.000 | −0.442 | −1.000 |

: The population fit $\theta(t)$ of the logistic regression, against the subgroup share $t$. The intercept is 0 for every $t$. {#tbl-prop-theta}

In @fig-prop-parameter-cloud, models trained at $t_0 = 0.4$ and $t_1 = 0.6$ overlap at $n = 100$ and separate along $\theta_2$ at $n = 1{,}000$.

![The parameters $(\theta_1, \theta_2)$ of 300 logistic regressions trained with subgroup share $t = 0.4$ and 300 with $t = 0.6$, at $n = 100$ (left) and $n = 1{,}000$ (right), with the linear meta-classifier fitted to them. The clouds shrink like $1/\sqrt n$; the gap between their centres stays fixed.](figures/prop_parameter_cloud.svg){#fig-prop-parameter-cloud}

For the attack itself, the script trains 150 shadow models per hypothesis for each $n$ and gap, with $t_0$ and $t_1$ centred on $\frac12$. It fits the linear test of @lem-prop-gauss, with means and covariance estimated from the shadow models' three parameters, and measures its accuracy on 150 fresh target models per hypothesis (@tbl-prop-sweep). Each accuracy averages 300 outcomes, so its standard error is at most $\sqrt{0.25/300} = 0.029$. Most cells agree with $\Phi(\gamma/2)$ to within two standard errors. The exceptions are small effects ($n \le 200$, gap $\le 0.1$), where the meta-classifier finds the direction of $\omega$ badly.

| $n$ | gap 0.05 | gap 0.1 | gap 0.2 | gap 0.4 |
| --- | --- | --- | --- | --- |
| 50 | 0.523 (0.546) | 0.520 (0.590) | 0.653 (0.676) | 0.830 (0.818) |
| 100 | 0.523 (0.564) | 0.670 (0.627) | 0.733 (0.741) | 0.863 (0.901) |
| 200 | 0.503 (0.590) | 0.670 (0.676) | 0.803 (0.819) | 0.957 (0.965) |
| 500 | 0.633 (0.641) | 0.787 (0.765) | 0.910 (0.926) | 0.997 (0.998) |
| 1,000 | 0.713 (0.695) | 0.867 (0.847) | 0.980 (0.979) | 1.000 (1.000) |

: Accuracy of the meta-classifier against logistic regressions, by dataset size $n$ and gap $\lvert t_1 - t_0\rvert$, with the prediction $\Phi(\gamma/2)$ of @eq-prop-delta in parentheses. {#tbl-prop-sweep}

The scaling law holds too. By @eq-prop-kappa the accuracy depends on $n$ and the gap only through $\sqrt n\,|t_1 - t_0|$, and plotted against that product the four curves fall on one (@fig-prop-meta-accuracy). So a gap of 0.05 at $n = 1{,}000$ is about as easy as a gap of 0.1 at $n = 200$ or so, since $\sqrt{1000}\times0.05 = 1.58$ against $\sqrt{200}\times0.1 = 1.41$.

![Left: accuracy of the meta-classifier against $n$ for four gaps $\lvert t_1 - t_0\rvert$ (150 shadow and 150 target models per hypothesis, seed 1). Right: the same accuracies against $\sqrt n\,\lvert t_1 - t_0\rvert$, with the Gaussian prediction $\Phi(\kappa\sqrt n\,\lvert t_1 - t_0\rvert/2)$, $\kappa = 0.647$.](figures/prop_meta_accuracy.svg){#fig-prop-meta-accuracy}

## Permutation invariance {#sec-prop-symmetry}

A logistic regression has one parameter vector per function, since different weights give different predictions, but a neural network has many. Swap two hidden units, together with their incoming and outgoing weights, and the network computes exactly the same function. Ganju et al. (2018) observed that a meta-classifier fed the flattened weights then sees each unit in an arbitrary position of the vector, and does poorly. This section works out why for one hidden layer of tanh units, and which representations bring the property back.

::: {#def-prop-network name="A one-layer tanh network and its symmetries"}
A network with $h$ hidden units computes
$$f_\theta(x) = \sigma\big(v^\top\tanh(Wx + \beta) + c\big),$$
where row $j$ of the matrix $W$ and entry $\beta_j$ of the hidden bias vector $\beta$ compute unit $j$'s input, $v_j$ is its outgoing weight, and $c$ is the output bias. Unit $j$'s parameters are $u_j = (W_{j\cdot}, \beta_j, v_j)$. A **symmetry** is a matrix $g = F\Pi$, where the permutation matrix $\Pi$ reorders the units and the diagonal matrix $F$ with entries $\pm1$ flips the sign of some of them. It acts on $\theta$ by applying $g$ to the rows of $W$ and to the entries of $\beta$ and $v$, leaving $c$ alone.
:::

There are $h!$ orders and $2^h$ sign patterns, so a generic network, one whose units are nonzero and pairwise neither equal nor opposite, has $h!\,2^h$ parameter vectors that compute the same function: $4! \times 2^4 = 384$ for $h = 4$. The next proposition shows that the symmetry of the function carries over to the distribution of the trained weights, provided training treats every arrangement of the units alike, and that this has a sharp consequence for a linear attacker.

::: {#prp-prop-symmetry name="Symmetries hide the property from raw weights"}
Every symmetry $g$ leaves the network's function unchanged,
$$f_{(gW,\ g\beta,\ gv,\ c)} = f_{(W,\ \beta,\ v,\ c)}.$$ {#eq-prop-symmetry}
Suppose the initial parameters have the same law as their image under every $g$ (for example, independent entries drawn from a distribution symmetric about 0) and training is gradient descent. Then, under both hypotheses of the game,

1. the law of the trained parameters is invariant under every $g$;
2. every coordinate of every $u_j$ has mean zero, so no linear score of the raw parameters has a mean that depends on the hypothesis through them;
3. the second-moment features $\sum_j u_ju_j^\top$ are invariant under every $g$.
:::

::: {.proof}
The plan is to show that $g$ passes through tanh, so it leaves the output unchanged, then that it passes through every step of gradient descent, and then to read off what this means for averages.

1. *tanh commutes with $g$.* tanh acts entry by entry, so permuting its input permutes its output in the same way. tanh is odd, $\tanh(-z) = -\tanh(z)$, so flipping the sign of an entry flips the sign of its output. Hence $\tanh(gz) = g\tanh(z)$.
2. *The output is unchanged.* By step 1 the symmetry can be pulled out of tanh, and $g^\top g = \Pi^\top F^2\Pi = I$, because each entry of $F$ is $\pm1$ and undoing a reordering gives back the original. So $(gv)^\top\tanh(gWx + g\beta) = v^\top g^\top g\tanh(Wx + \beta) = v^\top\tanh(Wx + \beta)$, which is @eq-prop-symmetry.
3. *Gradient descent commutes with $g$.* Let $\mathcal L$ be the training loss, and let $g$ also denote its action on the whole parameter vector, which only reorders entries and flips signs, so it is linear and orthogonal. Step 2 says $\mathcal L(g\theta) = \mathcal L(\theta)$ for every $\theta$. Differentiating both sides with the chain rule gives $g^\top\nabla\mathcal L(g\theta) = \nabla\mathcal L(\theta)$, so $\nabla\mathcal L(g\theta) = g\nabla\mathcal L(\theta)$. One step with learning rate $\eta$ from $g\theta$ therefore lands at $g\theta - \eta g\nabla\mathcal L(\theta) = g(\theta - \eta\nabla\mathcal L(\theta))$. Repeating, if $T$ steps from $\theta_0$ reach $\theta_T$, then from $g\theta_0$ they reach $g\theta_T$.
4. *The law is invariant.* The initial parameters $\theta_0$ have the same law as $g\theta_0$. Training maps them to $\theta_T$ and, by step 3, to $g\theta_T$, so $\theta_T$ and $g\theta_T$ have the same law for every dataset, and averaging over datasets under either hypothesis keeps this. That is claim 1.
5. *Raw means vanish.* Take the $g$ that flips unit $j$ only. By claim 1, $u_j$ and $-u_j$ have the same law, so $\mathbb E[u_j] = -\mathbb E[u_j]$, which forces $\mathbb E[u_j] = 0$ under both hypotheses. A linear score $w^\top\theta$ has means $w^\top m_0$ and $w^\top m_1$ under the two hypotheses, where $m_0$ and $m_1$ are the mean parameter vectors. Their difference $m_1 - m_0$ is zero in every unit coordinate, so the score's mean shift comes only from the output bias $c$, which no $g$ touches. That is claim 2.
6. *Second moments are invariant.* A permutation only reorders the terms of $\sum_ju_ju_j^\top$, which does not change the sum. A sign flip turns $u_j$ into $-u_j$, and $(-u_j)(-u_j)^\top = u_ju_j^\top$. That is claim 3.
:::

Raw weights hide the property because each position in the parameter vector holds a random one of the network's units, with a random sign. The property changes the set of units the network learns, not their positions, so averaging position by position washes it out. There are three ways to undo the symmetry. A **canonical form** picks one parameter vector per function: flip each unit so that $v_j > 0$, then sort the units by $v_j$. Sorting is the first remedy of Ganju et al., but it is fragile, because two units with nearly equal $v_j$ swap places from one network to the next. A **set representation** feeds each unit through the same map $\phi$, adds the results and applies a second map, $\rho\big(\sum_j\phi(u_j)\big)$, which does not depend on the order (Zaheer et al. 2017). The features $\sum_ju_ju_j^\top$ are the case $\phi(u) = uu^\top$, with $\rho$ the identity. **Query features** use the network's outputs on fixed probe inputs, which are invariant automatically and need only black-box access.

The example below shows that **black-box can beat white-box**. Whoever holds the weights can compute any query output, so white-box access is never weaker for the best attack. But a fixed linear meta-classifier is not the best attack, and for it a better representation can matter more than more access. Zhang, Tople & Ohrimenko (2021) attack properties through queries alone.

::: {#exm-prop-tanh}
The data are as in @exm-prop-logistic, with $t_0 = 0.2$ and $t_1 = 0.8$. Each network has $h = 4$ tanh units and is trained on $n = 200$ records by 300 steps of full-batch gradient descent from a random start. For each hypothesis, 150 shadow networks fit the linear discriminant and 150 target networks measure its accuracy.
Each unit has four parameters (two input weights, a bias and an output weight), so the raw parameters number $4 \times 4 + 1 = 17$ with the output bias. The symmetric second-moment matrix has 10 distinct entries, which with the output bias gives 11 features. To see which features carry signal, @tbl-prop-representations also reports each representation's largest **standardized mean gap**: for each feature, the difference of its means under $t_0$ and $t_1$, divided by its pooled standard deviation.

Raw parameters are at chance, as claim 2 predicts. Their largest standardized gap, 0.34, is noise: a difference of two means of 150 standardized values has standard deviation $\sqrt{2/150} = 0.115$, and the largest of 17 is typically two to three times that. Canonical order lifts the accuracy to 0.807, mostly through one feature (@fig-prop-symmetries, left). Second moments do better with fewer features, because they need no tie-breaking. Probe outputs are best, because they measure the function the property changes directly.
:::

| Representation | Features | Largest standardized mean gap | Accuracy |
| --- | --- | --- | --- |
| Raw parameters | 17 | 0.34 | 0.497 |
| Canonical order (flip so $v_j > 0$, sort by $v_j$) | 17 | 0.66 | 0.807 |
| Unit second moments $\sum_j u_ju_j^\top$ | 11 | 2.24 | 0.877 |
| Outputs on 16 probe inputs | 16 | 2.81 | 1.000 |

: The linear meta-classifier against tanh networks, by representation. The standardized gap of a feature is the difference of its means under $t_0$ and $t_1$, over its pooled standard deviation, on the shadow networks. {#tbl-prop-representations}

![Left: for each representation of 100 tanh networks per hypothesis ($t = 0.2$ against $t = 0.8$), the standardized mean gap of every feature. Right: accuracy of the linear meta-classifier on 100 further networks per hypothesis (seed 1). Raw parameters carry no mean signal; invariant representations do.](figures/prop_symmetries.svg){#fig-prop-symmetries}

## Code {#sec-prop-code}

The script [code/property_inference.py](../../code/part02-attacks/property_inference.py) computes every number in this chapter and draws every figure. The binomial advantages are exact sums, and the logistic regression's population fit and sandwich covariance come from Gauss–Hermite quadrature on a $40\times40$ grid. From the repository root:

```bash
uv run python code/part02-attacks/property_inference.py
```

| Experiment | Result |
| --- | --- |
| Released count, $t_0 = 0.45$, $t_1 = 0.55$ | Property advantage 0.6827, 0.9985, 1.0000 at $n = 100$, 1,000, 10,000 (@tbl-prop-count) |
| Released count, membership at $t = 0.5$ | 0.0398, 0.0126, 0.0040 at the same $n$ |
| Simulated property game, $n = 100$ | Advantage 0.6811 over 200,000 games (exact 0.6827) |
| Logistic regression, population view at $t = \frac12$ | $\theta = (1.704, 0, 0)$; $\theta' = (0, -1.704, 0)$; $\Sigma_1$ diagonal $(12.59, 6.94, 6.04)$; $\kappa = 0.647$ |
| 2,000 logistic regressions, $n = 1{,}000$, $t = \frac12$ | Mean $\theta_1 = 1.693$; $n\times$ variances $(11.92, 6.80, 6.08)$ |
| Meta-classifier sweep | @tbl-prop-sweep |
| tanh networks, $h = 4$, $n = 200$, $t = 0.2$ against $0.8$ | Accuracy 0.497 (raw), 0.807 (canonical), 0.877 (second moments), 1.000 (probes); symmetry check passed |

: The script's output with the default seed. {#tbl-prop-code}

[tests/test_property_inference.py](../../tests/test_property_inference.py) checks that the output still matches.

## Summary {#sec-prop-summary}

| Result | Statement | Use |
| --- | --- | --- |
| @thm-prop-ceiling | $\mathrm{Adv} \le \mathrm{TV}(P_{n,t_0}, P_{n,t_1}) \le \mathrm{TV}(\mathcal D_{t_0}^n, \mathcal D_{t_1}^n)$, the binomial TV when groups are visible | The ceiling; a released count is the worst case; report accuracy |
| @lem-prop-shift | A unimodal count shifted by one moves by its largest probability | Membership on a count |
| @prp-prop-membership | $(1 - t)\max_k\Pr[\mathrm{Bin}(n - 1, t) = k]$ | Membership gets harder like $1/\sqrt n$ |
| @prp-prop-property | $2\Phi\big(\sqrt n\,\lvert t_1 - t_0\rvert/(2\sqrt{\bar t(1 - \bar t)})\big) - 1$ | Properties get easier with $n$; defences sized for one record do not cover them |
| @prp-prop-mest | $\theta \approx \mathcal N(\theta(t), H^{-1}GH^{-1}/n)$ | The law of a fitted model |
| @lem-prop-gauss | Best accuracy $\Phi(\gamma/2)$, by a linear test | Train a linear meta-classifier |
| @thm-prop-meta, @cor-prop-count | Accuracy $\Phi(\kappa\sqrt n\,\lvert t_1 - t_0\rvert/2)$; $\kappa = 2$ for the count at $\bar t = \frac12$ | Predict an attack from $\theta'$ and $\Sigma_1$; dropping the group column does not hide the property |
| @prp-prop-symmetry | $h!\,2^h$ equivalent parameter vectors; raw means vanish for tanh | Use canonical order, sums over units or probe outputs |

: The results of this chapter, and where each is used. {#tbl-prop-summary}

## References {#sec-prop-references}

- Ateniese, Mancini, Spognardi, Villani, Vitali & Felici, "Hacking Smart Machines with Smarter Ones: How to Extract Meaningful Data from Machine Learning Classifiers" (International Journal of Security and Networks, 2015). The first property-inference attack: shadow models trained with and without the property, and a meta-classifier on their parameters.
- Ganju, Wang, Yang, Gunter & Borisov, "Property Inference Attacks on Fully Connected Neural Networks using Permutation Invariant Representations" (CCS 2018). The permutation problem for neural networks, neuron sorting, and the DeepSets meta-classifier.
- Suri & Evans, "Formalizing and Estimating Distribution Inference Risks" (PoPETs 2022). The property game as a test between two training distributions, and leakage measured in records.
- Zhang, Tople & Ohrimenko, "Leakage of Dataset Properties in Multi-Party Machine Learning" (USENIX Security 2021). Black-box property inference from query outputs alone.
- Mahloujifar, Ghosh & Chase, "Property Inference from Poisoning" (IEEE S&P 2022). Adding a few records makes a property far easier to infer.
- Melis, Song, De Cristofaro & Shmatikov, "Exploiting Unintended Feature Leakage in Collaborative Learning" (IEEE S&P 2019). Properties inferred from the updates of collaborative training.
- Salem, Cherubin, Evans, Köpf, Paverd, Suri, Tople & Zanella-Béguelin, "SoK: Let the Privacy Games Begin! A Unified Treatment of Data Inference Privacy in Machine Learning" (IEEE S&P 2023). Property inference as one game among the others of this part.
- Zaheer, Kottur, Ravanbakhsh, Póczos, Salakhutdinov & Smola, "Deep Sets" (NeurIPS 2017). The $\rho(\sum_j\phi(u_j))$ form of permutation-invariant functions.
- van der Vaart, *Asymptotic Statistics* (1998), ch. 5. Asymptotic normality of M-estimators and the sandwich covariance $H^{-1}GH^{-1}$, with the regularity conditions this chapter skips.
