# Property inference

## Why this matters

A hospital trains a readmission model on its 1,000 patients and shares it with a research partner. Suppose you have checked that no attack on the model can tell, much better than a coin flip, whether any one patient was in the training set. The partner can still learn whether 45% or 55% of the hospital's patients are over 65, and be right almost every time. A company that shares a fraud model may reveal the share of its customers in a market it has not announced. A model trained on a lab's machines may reveal whether those machines had been patched. None of these secrets is about a record. Each is a **property** of the training set as a whole: a statistic $t(D)$ of the dataset $D$, such as the fraction of records that belong to some subgroup.

**Property inference** is the attack that recovers such a statistic from a trained model. Ateniese et al. (2015) introduced it. In their examples, the attacker learned whether a speech recognizer had been trained on speakers with a particular accent, and whether a network-traffic classifier had seen a particular kind of traffic. Ganju et al. (2018) carried the attack to fully connected neural networks, and showed that the attacker must first deal with a nuisance: the same network can be written with its hidden units in any order.

This chapter works through four questions, each building on the one before. It starts by writing the attack as a game, because only then can you ask what the most any attacker can win is. With that ceiling in hand, it explains why a property leaks when no single record does, and why the leak grows with the size of the dataset. Real models have no formula for their distribution, so it then shows how a practical attacker builds its test by training models of its own, and predicts how well that test does. Finally it turns to neural networks, whose raw weights hide the property from a simple attacker, and finds the representation that brings it back.

### The one idea: a model is a noisy statistic, and a property moves it further than any record does

A trained model $\theta$ is a function of its training set, plus training noise. Think of it as an average over the $n$ records. Changing one record out of $n$ moves an average by about $1/n$. The randomness of which records were sampled moves it by about $1/\sqrt n$, the usual size of the error of an average. Since $1/n$ is much smaller than $1/\sqrt n$ when $n$ is large, a single record is lost in the noise. That is why membership inference gets *harder* with more data.

A property is different. Raising the subgroup's share from $t_0$ to $t_1$ changes about $n|t_1 - t_0|$ records at once. Each moves the average by about $1/n$, so together they move $\theta$ by an amount proportional to $|t_1 - t_0|$ that does not shrink with $n$. Against noise of size $1/\sqrt n$, the signal-to-noise ratio is about $|t_1 - t_0|/(1/\sqrt n) = \sqrt n\,|t_1 - t_0|$, so property inference gets *easier* with more data.

Everything else in this chapter is that argument made precise. The attack is a [hypothesis test](../part01-prerequisites/04-hypothesis-testing.md) that decides whether the model was trained under $t_0$ or under $t_1$, and the attacker approximates the best such test by training models of its own. @tbl-prop-uses lists the results.

| Result | Where it is used |
| --- | --- |
| The property game; advantage at most the total variation (TV) distance between the models, and between the datasets | Stating what a property attack achieves and what no attack can exceed |
| Membership advantage $\approx 1/\sqrt n$, property advantage $\to 1$ for the same released count | Why protecting records does not protect properties |
| Trained parameters $\approx \mathcal N(\theta(t), \Sigma_1/n)$; best accuracy $\Phi(\kappa\sqrt n\,\lvert t_1 - t_0\rvert/2)$ | Predicting an attack's accuracy from the model and the data; the $\sqrt n\,\lvert t_1 - t_0\rvert$ scaling law |
| Shadow models and a linear meta-classifier | The practical attack, when the law of $\theta$ is unknown |
| Hidden-unit symmetries; invariant representations | Attacking neural networks through their weights or through their outputs |

: The results of this chapter and where each is used. {#tbl-prop-uses}

**What this chapter assumes.** Every tool the derivations need is restated where it is used, so you can read this chapter on its own. For more detail: the game view, the advantage (how much more often an attack says 1 when the answer is 1 than when it is 0) and the fact that processing the view cannot help the attacker are in [Threat models and privacy games](01-threat-models.md). Shadow models first appear in [Membership inference](02-membership-inference.md). The TV distance, the data processing inequality and sufficient statistics are in [Divergences](../part01-prerequisites/03-divergences.md). The Neyman–Pearson lemma and the Gaussian trade-off curve are in [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md). The binomial distribution and its normal approximation are in [Probability distributions](../part01-prerequisites/01-distributions.md).

**Conventions.** Records come from a population that mixes a majority group $A$ and a subgroup $B$. Write $\pi_A$ and $\pi_B$ for the distributions of a record from each group. A population in which a fraction $t$ of people belong to $B$ is the mixture $\pi_t = (1 - t)\pi_A + t\,\pi_B$: to draw a record, first pick $B$ with probability $t$ and $A$ otherwise, then draw from that group. A dataset $D$ is $n$ independent records from $\pi_t$, written $D \sim \pi_t^n$. A training algorithm $T$ maps it to parameters $\theta = T(D)$, possibly using random choices of its own (initialization, mini-batch order). $P_{n,t}$ is the distribution of $\theta$, over both the random data and the training randomness. The secret is which of two shares, $t_0$ or $t_1$, was used. A challenger picks it with a hidden bit $b$, which is 0 or 1 with probability $\frac12$ each (a **fair bit**); $b = 0$, share $t_0$, is the null hypothesis, and $b = 1$, share $t_1$, the alternative. The attacker outputs a guess $b'$. Its **true positive rate** (TPR), $\mathrm{TPR} = \Pr[b' = 1 \mid b = 1]$ is how often it says 1 when the bit is 1, and its **false positive rate** (FPR), $\mathrm{FPR} = \Pr[b' = 1 \mid b = 0]$ is how often it says 1 when the bit is 0. Its **advantage** is $\mathrm{TPR} - \mathrm{FPR}$: how much more often it says 1 in the world where 1 is right. Its **accuracy** is $\Pr[b' = b]$, the probability that the guess matches the hidden bit, whichever value the bit has. It is not the probability that $b$ is 1. With a fair bit, half the games have $b = 1$, where the attacker wins with probability TPR, and half have $b = 0$, where it wins with probability $1 - \mathrm{FPR}$, so $\Pr[b' = b] = \frac12\mathrm{TPR} + \frac12(1 - \mathrm{FPR}) = \frac12(1 + \text{advantage})$. An attacker who ignores the model says 1 equally often in both worlds, so its advantage is 0 and its accuracy is $\frac12$: that is the coin-flip baseline. $\Phi(z) = \Pr[Z \le z]$ is the standard normal distribution function, for $Z \sim \mathcal N(0, 1)$.

## Toolkit: three facts every derivation uses

### Fact 1 — The best advantage is the total variation distance

The attacker sees a view, such as the trained parameters, whose distribution is $P$ when $b = 0$ and $Q$ when $b = 1$. Some views are more likely in one world than in the other, and that difference is all the attacker has to go on. The best it can do is say 1 exactly on the views that are more likely under $Q$. How well that works is measured by the **total variation distance**, half the total absolute difference between the two probability tables $p$ and $q$:
$$\mathrm{TV}(P, Q) = \tfrac12\sum_x|p(x) - q(x)|.$$
(For continuous views, the sum is an integral over densities.) The fact is that every attack has advantage at most $\mathrm{TV}(P, Q)$, and the **likelihood-ratio test** "guess 1 when $q > p$" attains it. With a fair bit, the best accuracy is therefore $\frac12 + \frac12\mathrm{TV}(P, Q)$.

The argument takes two lines. An attack may randomize, so describe it by $\phi(x)$, the probability that it says 1 on view $x$. Its TPR is $\sum_x\phi(x)q(x)$ and its FPR is $\sum_x\phi(x)p(x)$, so its advantage is $\sum_x\phi(x)(q(x) - p(x))$. Each term is largest when $\phi(x) = 1$ where $q(x) > p(x)$ and $\phi(x) = 0$ elsewhere, and the maximum is the total excess of $q$ over $p$. Both tables sum to 1, so the excess of $q$ over $p$ equals the excess of $p$ over $q$, and each is half the total absolute difference: the TV distance. This is the best-advantage result of [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md), restated for games in [Threat models and privacy games](01-threat-models.md).

A Gaussian case gives a number you will see again. Let $P = \mathcal N(0, 1)$ and $Q = \mathcal N(1, 1)$. The two densities cross halfway between the means, at $\frac12$, and $q > p$ to the right of it, so the best test says 1 when the view exceeds $\frac12$. Its TPR is $\Pr[\mathcal N(1, 1) > \frac12] = \Phi(\frac12)$, because $\frac12$ is half a standard deviation below the mean 1. Its FPR is $\Pr[\mathcal N(0, 1) > \frac12] = 1 - \Phi(\frac12)$. So $\mathrm{TV} = \Phi(\frac12) - (1 - \Phi(\frac12)) = 2\Phi(\frac12) - 1 = 0.3829$, and the best accuracy is $\frac12 + \frac12(0.3829) = 0.6915$, which is $\Phi(\frac12)$ itself. This fact sets the ceiling on the property game and turns every advantage in this chapter into an accuracy.

### Fact 2 — Training cannot add information about the property

The attacker sees the model, not the data, and the model is computed from the data. Whatever the attacker could learn from $\theta$, someone holding $D$ could learn too: they could train $\theta$ themselves, using the same algorithm and fresh random choices, and run the attacker's test on it. So the best test on $\theta$ can be no better than the best test on $D$. In terms of Fact 1, if $\theta = T(D)$, where $T$ may use randomness that does not depend on $b$, then
$$\mathrm{TV}(P_{n,t_0}, P_{n,t_1}) \le \mathrm{TV}(\pi_{t_0}^n, \pi_{t_1}^n).$$
This is the **data processing inequality** of [Divergences](../part01-prerequisites/03-divergences.md): passing a view through any channel whose behaviour does not depend on the secret cannot increase its TV distance. It is the same as the "processing cannot help the adversary" fact of [Threat models and privacy games](01-threat-models.md).

Equality holds when $\theta$ is a **sufficient statistic** for $t$, which means that the likelihood ratio between the two worlds can be computed from $\theta$ alone, so $\theta$ keeps everything in $D$ that bears on $t$. The two extreme cases show the range. A model that ignores its data entirely has $P_{n,t_0} = P_{n,t_1}$, and TV 0. Releasing the dataset itself attains the ceiling. Fact 2 gives the ceiling in the game section, and it is why a released count turns out to be the worst case.

### Fact 3 — Two Gaussians with a shared covariance: the best test is linear

The trained parameters of many models are approximately Gaussian, as @sec-prop-meta shows. So the most useful special case of Fact 1 is two Gaussians in $\mathbb R^d$ with the same covariance $\Sigma$ and different means, $P = \mathcal N(m_0, \Sigma)$ and $Q = \mathcal N(m_1, \Sigma)$. How hard are they to tell apart? In one dimension, the answer depends on how far apart the means are, measured in standard deviations. In $d$ dimensions the noise can be larger in some directions than in others, so a shift along a noisy direction counts for less than the same shift along a quiet one. The **Mahalanobis distance** measures the shift in exactly this way, in standard deviations along its own direction:
$$\Delta = \sqrt{(m_1 - m_0)^\top\Sigma^{-1}(m_1 - m_0)}.$$
For $d = 1$ and $\Sigma = \sigma^2$, it is $|m_1 - m_0|/\sigma$.

The fact has two parts. First, the log of the likelihood ratio is a linear function of $\theta$: it measures how far $\theta$ lies past the midpoint of the two means, along a direction $w$,
$$\ln\frac{q(\theta)}{p(\theta)} = w^\top\Big(\theta - \frac{m_0 + m_1}2\Big), \qquad w = \Sigma^{-1}(m_1 - m_0).$$
So the best test of Fact 1, "say 1 when $q > p$", is a linear test: say 1 when $\theta$ lies on $m_1$'s side of a hyperplane through the midpoint. Second, its accuracy with a fair bit depends only on the Mahalanobis distance,
$$\Phi(\Delta/2).$$ {#eq-prop-gauss-acc}

*Derivation.* The idea is to write out the two Gaussian log-densities, watch the quadratic parts cancel because the covariances are equal, and then find the distribution of what is left.

1. **Expand the two quadratics.** A Gaussian log-density is $-\frac12(\theta - m)^\top\Sigma^{-1}(\theta - m)$ plus a constant that depends only on $\Sigma$, and that constant is the same for $P$ and $Q$. So $\ln q - \ln p = -\frac12(\theta - m_1)^\top\Sigma^{-1}(\theta - m_1) + \frac12(\theta - m_0)^\top\Sigma^{-1}(\theta - m_0)$. Multiplying out, the terms $\theta^\top\Sigma^{-1}\theta$ appear with opposite signs and cancel, which leaves $(m_1 - m_0)^\top\Sigma^{-1}\theta - \frac12(m_1^\top\Sigma^{-1}m_1 - m_0^\top\Sigma^{-1}m_0)$.
2. **Factor the constant.** $m_1^\top\Sigma^{-1}m_1 - m_0^\top\Sigma^{-1}m_0 = (m_1 - m_0)^\top\Sigma^{-1}(m_1 + m_0)$, because when you multiply out the right-hand side, the cross terms $m_1^\top\Sigma^{-1}m_0$ and $m_0^\top\Sigma^{-1}m_1$ are equal ($\Sigma^{-1}$ is symmetric) and cancel. Pulling out $w^\top = (m_1 - m_0)^\top\Sigma^{-1}$ gives the linear form.
3. **Find its distribution.** $L = w^\top(\theta - \frac{m_0 + m_1}2)$ is a linear function of a Gaussian, so it is Gaussian. Under $Q$, $\theta$ has mean $m_1$, so $L$ has mean $w^\top(m_1 - \frac{m_0 + m_1}2) = w^\top\frac{m_1 - m_0}2 = \frac{\Delta^2}2$, by the definition of $\Delta$. Under $P$ the same computation with $m_0$ gives $-\frac{\Delta^2}2$. Under both, its variance is $w^\top\Sigma w = (m_1 - m_0)^\top\Sigma^{-1}\Sigma\Sigma^{-1}(m_1 - m_0) = \Delta^2$, so its standard deviation is $\Delta$.
4. **Compute the accuracy of the best test.** By Fact 1, the best test guesses 1 when $q > p$, that is, when $L > 0$. Under $Q$, $L$ has mean $\frac{\Delta^2}2$ and standard deviation $\Delta$, so 0 lies $\frac{\Delta^2/2}{\Delta} = \frac\Delta2$ standard deviations below the mean, and the test is right with probability $\Pr[\mathcal N(\Delta^2/2, \Delta^2) > 0] = \Phi(\Delta/2)$. Under $P$, $L$ has mean $-\frac{\Delta^2}2$, and the test is right when $L < 0$, which lies $\frac\Delta2$ standard deviations above that mean: the same probability, $\Phi(\Delta/2)$. Averaging the two worlds gives @eq-prop-gauss-acc.

In one dimension with $\Sigma = 1$, $\Delta = |m_1 - m_0|$, and @eq-prop-gauss-acc is the Gaussian pair of Fact 1: $\Delta = 1$ gives $\Phi(0.5) = 0.6915$. In many dimensions, only the length of the mean shift measured with $\Sigma^{-1}$ matters, so a shift along a direction where $\theta$ is noisy counts for little. This fact gives the accuracy of every attack on a model whose parameters are approximately Gaussian, and it says which meta-classifier to train: a linear one.

## The property-inference game {#sec-prop-game}

### Context

"Can the attacker learn the property?" needs the same care as any attack: who draws the data, what the attacker sees and knows, and how its guess is scored. [Threat models and privacy games](01-threat-models.md) writes attacks as games between a **challenger**, who runs the training pipeline with a hidden secret, and an **adversary**, who tries to guess the secret. It follows Salem et al. (2023). The property game is the version in which the secret is one bit about the population that produced the training set: was the subgroup's share $t_0$ or $t_1$? Suri & Evans (2022) call it **distribution inference** and study it in this form.

### Statement

The **property-inference game** with parameters $(n, t_0, t_1)$ runs in four steps. The secret is drawn at random by a fair bit so that neither answer is right more often by default, and an attacker who learns nothing scores exactly $\frac12$.

1. The challenger draws a fair bit $b$.
2. It draws a dataset $D \sim \pi_{t_b}^n$ and trains $\theta = T(D)$.
3. The adversary receives a view of $\theta$: the parameters (white-box), or answers to queries (black-box). It knows $\pi_A$, $\pi_B$, $n$, $t_0$, $t_1$ and the training algorithm $T$.
4. It outputs $b'$ and wins if $b' = b$.

What limits the adversary? Its view is a model whose distribution is $P_{n,t_0}$ or $P_{n,t_1}$, so by Fact 1 its advantage is at most the TV distance between them. The model is computed from the data, so by Fact 2 that distance is at most the TV distance between the two dataset distributions. For every adversary,
$$\mathrm{TPR} - \mathrm{FPR} \le \mathrm{TV}(P_{n,t_0}, P_{n,t_1}) \le \mathrm{TV}(\pi_{t_0}^n, \pi_{t_1}^n).$$ {#eq-prop-ceiling}
The right-hand side becomes concrete when the group of each record can be read off the record ($\pi_A$ and $\pi_B$ have disjoint supports: no record could come from both groups). Then the only thing in $D$ that says anything about $t$ is how many records come from the subgroup. Call that count $S$. It is a sufficient statistic, and $S$ is binomial: each of the $n$ records is from the subgroup with probability $t$, independently. So the right-hand side is exactly
$$\mathrm{TV}\big(\mathrm{Bin}(n, t_0), \mathrm{Bin}(n, t_1)\big).$$ {#eq-prop-count-ceiling}

### Derivation

1. **The game is a test.** The view has distribution $P_{n,t_0}$ when $b = 0$ and $P_{n,t_1}$ when $b = 1$. By Fact 1, no attacker's advantage exceeds the TV distance between them. This is the left inequality of @eq-prop-ceiling.
2. **Training is a channel.** The view is a randomized function of $D$, with randomness that does not depend on $b$. Fact 2 says such processing cannot increase the TV distance, which gives the right inequality.
3. **Write the likelihood ratio of a dataset.** To find what in $D$ matters, compare how likely $D$ is under the two shares. With disjoint supports, a record $x$ from $B$ has density $t\,\pi_B(x)$ under $\pi_t$ (pick group $B$, then draw $x$), and a record from $A$ has $(1 - t)\pi_A(x)$. The records are independent, so the density of $D$ is the product over its $n$ records. Divide the $t_1$ version by the $t_0$ version. The factors $\pi_A(x)$ and $\pi_B(x)$ are the same in both and cancel. Each of the $S$ subgroup records leaves a factor $t_1/t_0$, and each of the $n - S$ others a factor $(1 - t_1)/(1 - t_0)$:
$$\frac{d\pi_{t_1}^n}{d\pi_{t_0}^n}(D) = \Big(\frac{t_1}{t_0}\Big)^S\Big(\frac{1 - t_1}{1 - t_0}\Big)^{n - S}.$$
4. **Conclude that the count is sufficient.** The likelihood ratio depends on $D$ only through $S$, so $S$ is sufficient: the best test of Fact 1 can be computed from $S$ alone, and the data processing inequality of Fact 2 holds with equality. The TV distance between the datasets therefore equals the TV distance between the laws of $S$, which are $\mathrm{Bin}(n, t_0)$ and $\mathrm{Bin}(n, t_1)$. That is @eq-prop-count-ceiling.

### Interpretation

The first lesson is that **releasing the count is the worst case**. When the group is visible in each record, no model trained on $D$ can leak more about $t$ than the count of subgroup records, because the count already carries everything $D$ says about $t$. A model that does not see the group, like the logistic regression of @sec-prop-meta, leaks less, and how much less is a number you can measure.

The game also says **what the attacker needs**. It gives the adversary $\pi_A$, $\pi_B$ and $T$. In practice this means auxiliary data from both groups and the ability to train models the way the victim does. Ateniese et al. and Ganju et al. assume exactly this. It is the same assumption that makes shadow models work in [Membership inference](02-membership-inference.md): the attacker trains its own copies of the victim's pipeline on data it controls.

**The secret belongs to a group, not a person**, and that changes how the attack is scored. The game has no target record. Its baseline is a coin flip, and the score is accuracy, not the true positive rate at a low false positive rate. In membership inference, the attacker faces many targets, and a few records exposed with high confidence matter even if most are not, so the end of the curve with few false positives is what counts. Here each dataset carries one secret, so the plain probability of guessing it is the natural score.

You may ask whether this **is an attack at all**. Learning that smokers get more lung cancer is the purpose of a medical study, not a breach. Property inference is a harm when the property describes the *data owner* rather than the world: the share of a company's customers in a region, the demographics of one hospital's patients, the configuration of one lab's machines. When $D$ is a small sample from a public population, its property is public knowledge anyway.

Finally, the game has **variants**. The secret can be a real number (estimate $t$ rather than choose between $t_0$ and $t_1$), or the presence of any record with a given feature. Some games fix the subgroup count at exactly $\lfloor tn\rfloor$ instead of drawing it at random. With a fixed count and the group visible, the ceiling is 1, because the count itself differs between the two worlds and reveals the bit. The attacker's difficulty then comes only from the training noise and from the group being hidden.

### Example: one hospital's patients

A hospital has $n = 1{,}000$ patients. Whether 45% or 55% of them are over 65 is the secret. If age group is part of each record, @eq-prop-count-ceiling gives the ceiling $\mathrm{TV}(\mathrm{Bin}(1000, 0.45), \mathrm{Bin}(1000, 0.55)) = 0.9985$ for every model trained on the data. You can see why it is so close to 1. The count has mean 450 in one world and 550 in the other, and its standard deviation is $\sqrt{1000 \times 0.45 \times 0.55} \approx 15.7$, so the two means are more than six standard deviations apart and the two distributions barely overlap. A model from which the count can be read off reaches the ceiling: by Fact 1 its attacker is right with probability $\frac12 + \frac12(0.9985)$, more than 99.9% of the time.

At $n = 100$ the means are 45 and 55, the standard deviation is about 5, and the distributions overlap much more. The ceiling is $0.6827$, and 200,000 simulated games with the likelihood-ratio test on the count give an advantage of 0.6811, within sampling error of the formula.

### Check yourself

- Why can no attacker on any model trained on $D$ beat @eq-prop-count-ceiling when the group is visible in every record? *(The model is a randomized function of $D$, so by the data processing inequality its TV is at most that of $D$, and the count $S$ is sufficient, so the TV of $D$ equals that of $S$.)*
- A defender trains on a dataset from which the group column has been deleted. Does @eq-prop-ceiling still hold, and does @eq-prop-count-ceiling? *(@eq-prop-ceiling holds for any training algorithm. @eq-prop-count-ceiling still bounds it, because the defender's dataset is a function of the original one; it may no longer be attained.)*

## Why a property leaks when no record does {#sec-prop-scaling}

### Context

Take the simplest model of all: the training algorithm releases the count $S$ of subgroup records. Two attackers look at the same release. One asks about a record: "is my target, a member of the subgroup, in $D$?" The other asks about the property: "was $D$ drawn with $t_0$ or $t_1$?" For a count, both best advantages can be computed exactly, and they move in opposite directions as $n$ grows. This makes the one idea of the chapter exact.

### Statement

**Membership.** In the membership game, $D$ holds $n - 1$ records from $\pi_t$ plus one more: either the target (a subgroup member) or a fresh record from $\pi_t$. Let $X \sim \mathrm{Bin}(n - 1, t)$ be the subgroup count among the $n - 1$ other records. The target adds exactly 1 to the count, and a fresh record adds 1 only a fraction $t$ of the time. So the target is detectable only through how much one extra unit changes the distribution of a count that already varies by about $\sqrt n$. That change is governed by the height of the most likely count, and the best advantage is
$$\mathrm{Adv}_{\mathrm{mem}}(n) = (1 - t)\max_k\Pr[X = k] \approx \frac{1 - t}{\sqrt{2\pi(n - 1)t(1 - t)}}.$$ {#eq-prop-membership}

**Property.** Under the two hypotheses the fraction $S/n$ centres on $t_0$ or on $t_1$, with standard deviation about $\sqrt{t(1 - t)/n}$. The best advantage is the TV distance between the two binomials, and by the normal approximation it depends on how many standard deviations apart the two centres are. With the midpoint share $\bar t = \frac{t_0 + t_1}2$,
$$\mathrm{Adv}_{\mathrm{prop}}(n) = \mathrm{TV}\big(\mathrm{Bin}(n, t_0), \mathrm{Bin}(n, t_1)\big) \approx 2\Phi\Big(\frac{\sqrt n\,|t_1 - t_0|}{2\sqrt{\bar t(1 - \bar t)}}\Big) - 1.$$ {#eq-prop-property}

The first falls like $1/\sqrt n$. The second rises to 1 once $\sqrt n\,|t_1 - t_0|$ is a few times $\sqrt{\bar t(1 - \bar t)}$.

### Derivation

**Membership.** The plan is to write the two views as distributions of the count, notice that one of them is a mixture containing the other, and reduce the TV distance to the effect of shifting a binomial by one.

1. **Write the two views.** With the target in $D$, $S = X + 1$. With a fresh record, $S = X + B'$, where $B' \sim \mathrm{Bernoulli}(t)$ is 1 when the fresh record is from the subgroup.
2. **See the second view as a mixture.** $X + B'$ equals $X + 1$ with probability $t$ and $X$ with probability $1 - t$, so its law is $t\,\mathcal L(X + 1) + (1 - t)\,\mathcal L(X)$, where $\mathcal L(\cdot)$ means "the distribution of". A fraction $t$ of the time the non-member world looks exactly like the member world.
3. **Mixing scales the TV distance.** For any $P$ and $Q$, $\mathrm{TV}(P, tP + (1 - t)Q) = \frac12\sum_x|p(x) - tp(x) - (1 - t)q(x)| = \frac12\sum_x|(1 - t)(p(x) - q(x))| = (1 - t)\mathrm{TV}(P, Q)$. By Fact 1, the best advantage is therefore $(1 - t)\,\mathrm{TV}(\mathcal L(X + 1), \mathcal L(X))$.
4. **Shift a single-peaked distribution by one.** Write $p_k = \Pr[X = k]$. Then $\Pr[X + 1 = k] = p_{k-1}$, so $\mathrm{TV}(\mathcal L(X + 1), \mathcal L(X)) = \frac12\sum_k|p_k - p_{k-1}|$, half the total size of the steps between neighbouring probabilities. A binomial probability mass function is **unimodal**: it rises to its most likely value, the mode $k^*$, and then falls. So the upward steps add up to the climb from 0 to the top, $p_{k^*}$, and the downward steps add up to the descent back to 0, also $p_{k^*}$. Half of the total is $p_{k^*} = \max_k p_k$. @tbl-prop-shift walks through $X \sim \mathrm{Bin}(4, \frac12)$.
5. **Estimate the height of the mode.** By the normal approximation of [Probability distributions](../part01-prerequisites/01-distributions.md), a binomial with many trials is close to a normal with the same mean and variance, here $\mathcal N((n - 1)t, (n - 1)t(1 - t))$. A normal density with variance $s^2$ has height $1/\sqrt{2\pi s^2}$ at its peak, here $1/\sqrt{2\pi(n - 1)t(1 - t)}$. The probability of the most likely integer is close to that peak height, because the integers are spaced one apart and the density is nearly flat over a unit interval at the top. That gives @eq-prop-membership.

| $k$ | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| $16\Pr[X = k]$ | 1 | 4 | 6 | 4 | 1 | 0 |
| $16\Pr[X + 1 = k]$ | 0 | 1 | 4 | 6 | 4 | 1 |
| $16 \times$ difference | 1 | 3 | 2 | −2 | −3 | −1 |

: Shifting $X \sim \mathrm{Bin}(4, \frac12)$ by one. The positive differences add up to $6/16$, the height of the mode, and so do the negative ones. Half the total absolute difference is $\frac12 \times \frac{12}{16} = \frac{6}{16} = 0.375 = \max_k\Pr[X = k]$. {#tbl-prop-shift}

**Property.** Here the plan is to use the exact answer from the game section, then approximate both binomials by normals with a common spread so that the Gaussian pair of Fact 1 applies.

1. **Start from the exact answer.** By @sec-prop-game, the released count $S$ is a sufficient statistic, so the best advantage is the TV distance between the two binomials.
2. **Approximate by two normals.** By the normal approximation, $S/n$ is close to $\mathcal N(t, t(1 - t)/n)$. The two variances $t_0(1 - t_0)/n$ and $t_1(1 - t_1)/n$ are close when the gap is small, so replace both by the one at the midpoint, $\bar t(1 - \bar t)/n$. Now divide $S/n$ by its standard deviation $\sqrt{\bar t(1 - \bar t)/n}$. The problem becomes a pair of unit-variance Gaussians whose means are $\mu = \sqrt n\,|t_1 - t_0|/\sqrt{\bar t(1 - \bar t)}$ apart.
3. **Use the Gaussian pair.** As in the example of Fact 1, two unit-variance Gaussians with means $\mu$ apart have densities that cross at the midpoint, $\mu/2$ from each mean, so the best test has $\mathrm{TPR} = \Phi(\mu/2)$ and $\mathrm{FPR} = 1 - \Phi(\mu/2)$, and the TV distance is $2\Phi(\mu/2) - 1$. Substituting $\mu$ gives @eq-prop-property.

**The general pattern.** Neither argument is special to counts. Most estimators behave like averages: $\theta \approx \theta(\pi) + \frac1n\sum_i\psi(x_i)$, where $\theta(\pi)$ is the value the estimator would take on the whole population and $\psi(x)$ is the **influence** of one record $x$, how far it pulls the estimate (@sec-prop-meta derives this for logistic regression). One record $z$ shifts $\theta$ by $\psi(z)/n$, while the sampling noise has size $\sigma/\sqrt n$, where $\sigma$ is the standard deviation of $\psi(x)$. The shift measured in noise units is $(|\psi(z)|/n)/(\sigma/\sqrt n) = |\psi(z)|/(\sigma\sqrt n) \to 0$. The property instead shifts the centre from $\theta(\pi_{t_0})$ to $\theta(\pi_{t_1})$, a fixed distance, so in noise units the shift grows like $\sqrt n$.

### Interpretation

The headline is that **more data protects records and exposes properties**. @fig-prop-count-leak (right) shows both advantages against $n$ on log axes. The membership advantage falls along a line of slope $-\frac12$, which is what $1/\sqrt n$ looks like on log axes. The property advantage climbs to 1, sooner for a larger gap $|t_1 - t_0|$.

The reason is that **a property is a group of records**. Moving from $t_0$ to $t_1$ changes about $n|t_1 - t_0|$ records: for the hospital, $1{,}000 \times 0.1 = 100$ of its 1,000 patients. A protection measured one record at a time says little about a change to 100 records at once. [Part 3](../part03-dp-foundations/02-properties.md) makes this precise with its group-privacy property, whose guarantee weakens with the size of the group.

It follows that **hiding the property means hiding the population**. To push the property advantage down, the released statistic must be noisy on the scale of $n|t_1 - t_0|$, not on the scale of one record. That much noise also destroys what the statistic was for ([Exercise 6](solutions/04-property-inference.md#sec-sol-property-inference-6)).

**The approximations are good here.** At $n = 100$ the normal approximation to the property advantage is exact to four decimals, and the one to the membership advantage is off by less than 1% (@tbl-prop-count).

![Left: the released fraction $S/n$ at $n = 1{,}000$ under $t = 0.45$ and $t = 0.55$, and under $t = 0.5$ with and without one target from the subgroup; the last two curves are indistinguishable. Right: the best property advantage for two gaps, and the best membership advantage of one record, against $n$.](figures/prop_count_leak.svg){#fig-prop-count-leak}

### Example: the same release, two attackers

@tbl-prop-count gives both advantages for $t_0 = 0.45$, $t_1 = 0.55$, and for membership at $t = 0.5$. You can check the first row by hand. For the property, $\bar t = 0.5$, so $\sqrt{\bar t(1 - \bar t)} = 0.5$ and the argument of $\Phi$ in @eq-prop-property is $\sqrt{100} \times 0.1/(2 \times 0.5) = 1$; then $2\Phi(1) - 1 = 2(0.8413) - 1 = 0.6827$. For membership, @eq-prop-membership gives $0.5/\sqrt{2\pi \times 99 \times 0.25} = 0.5/12.47 = 0.0401$.

| $n$ | Property, exact | Property, @eq-prop-property | Membership, exact | Membership, @eq-prop-membership |
| --- | --- | --- | --- | --- |
| 100 | 0.6827 | 0.6827 | 0.0398 | 0.0401 |
| 1,000 | 0.9985 | 0.9984 | 0.0126 | 0.0126 |
| 10,000 | 1.0000 | 1.0000 | 0.0040 | 0.0040 |

: Best advantages from a released count of subgroup records. Property: $t_0 = 0.45$ against $t_1 = 0.55$. Membership: one subgroup record, population share $t = 0.5$. {#tbl-prop-count}

At $n = 10{,}000$, an attacker who wants to know whether one patient is in the data has advantage 0.0040: with a fair bit it is right with probability $\frac12 + \frac12(0.0040) = 0.502$. An attacker who wants the share of patients over 65 is never wrong to four decimals. The data, the release and the attacker's knowledge are the same; only the question differs.

### Check yourself

- In step 4 of the membership derivation, why does the argument need the probability mass function to be unimodal? *(For one that rises and falls several times, the climbs and descents add up to more than twice the maximum. For $X$ uniform on $\{0, 2\}$, $\mathrm{TV}(X, X + 1) = 1$ while the maximum is $\frac12$.)*
- Doubling the gap $|t_1 - t_0|$ has the same effect on @eq-prop-property as multiplying $n$ by what? *(By 4: the formula depends on $n$ and the gap only through $\sqrt n\,|t_1 - t_0|$.)*

## Shadow models and meta-classifiers {#sec-prop-meta}

### Context

For a real model nobody can write down $P_{n,t}$, the distribution of the trained parameters, so the likelihood-ratio test of Fact 1 is out of reach. Ateniese et al. (2015) gave the recipe every later attack uses: if you cannot compute the two distributions, sample from them. Draw many datasets from $\pi_{t_0}$ and from $\pi_{t_1}$, using auxiliary data. Train a **shadow model** on each, exactly as the victim trains. Label each shadow model's parameters with the bit that produced them, and train a classifier on these labelled parameter vectors: the **meta-classifier**. Then apply it to the victim's model. This is the shadow-model idea of [Membership inference](02-membership-inference.md), with whole models as the data points.

Which meta-classifier should you train, and how accurate can it be? For models fitted by minimizing an average loss, the parameters turn out to be approximately Gaussian with a covariance that shrinks like $1/n$. Fact 3 then answers both questions: train a linear classifier, and expect accuracy $\Phi(\Delta/2)$.

### Statement

The setting is a model fitted by minimizing an average loss. Let $\theta$ minimize the average loss $\frac1n\sum_i\ell(x_i, \theta)$ over the training records, and let $\theta(t)$ minimize the expected loss under $\pi_t$: the model you would get from infinitely many records. How noisy $\theta$ is around $\theta(t)$ depends on two matrices. $H$ is the average curvature of the loss at $\theta(t)$: a sharply curved loss pins the minimizer down. $G$ is the covariance of the per-record gradients at $\theta(t)$: how much each record pulls in its own direction. Together they give $\Sigma_1(t)$, the covariance of the fit for a single record's worth of data (the subscript 1):
$$H = \mathbb E_{\pi_t}\nabla^2\ell(x, \theta(t)), \qquad G = \mathbb E_{\pi_t}\big[\nabla\ell(x, \theta(t))\nabla\ell(x, \theta(t))^\top\big], \qquad \Sigma_1(t) = H^{-1}GH^{-1}.$$
$\Sigma_1$ is called the **sandwich covariance**, because $G$ sits between two copies of $H^{-1}$. Then, for large $n$:

1. **The parameters are approximately Gaussian,** centred on the population fit with covariance shrinking like $1/n$: $\theta \approx \mathcal N\big(\theta(t), \Sigma_1(t)/n\big)$.
2. **The best attack is linear,** and its accuracy is that of Fact 3 with the shift $\theta(t_1) - \theta(t_0)$ measured against the noise $\bar\Sigma_1/n$:
$$\Phi(\Delta/2), \qquad \Delta = \sqrt{n\,\big(\theta(t_1) - \theta(t_0)\big)^\top\bar\Sigma_1^{-1}\big(\theta(t_1) - \theta(t_0)\big)},$$ {#eq-prop-delta}
where $\bar\Sigma_1$ is the average of $\Sigma_1(t_0)$ and $\Sigma_1(t_1)$.
3. **For a small gap, only $\sqrt n\,|t_1 - t_0|$ matters.** The shift is then about the slope $\theta'(\bar t) = \frac{d\theta}{dt}$ times the gap, so $\Delta$ is the gap times $\sqrt n$ times a constant $\kappa$, which is the length of the slope measured against the noise:
$$\Delta \approx \kappa\sqrt n\,|t_1 - t_0|, \qquad \kappa = \sqrt{\theta'(\bar t)^\top\Sigma_1(\bar t)^{-1}\theta'(\bar t)}.$$ {#eq-prop-kappa}
For the released fraction $\theta = S/n$, $\kappa = 1/\sqrt{\bar t(1 - \bar t)}$, which is $1/\sqrt{0.25} = 2$ at $\bar t = \frac12$, and @eq-prop-delta becomes @eq-prop-property.

### Derivation

**The parameters are approximately Gaussian.** This is the standard asymptotic normality of **M-estimators**, estimators defined as the minimizer of an average loss (van der Vaart 1998, ch. 5). The idea: near its minimum, the fitted parameter is a linear function of an average of per-record gradients, and an average is approximately Gaussian. Four steps make this precise.

1. **The fit solves an average equation.** At a minimizer the gradient is zero, so $\frac1n\sum_i\nabla\ell(x_i, \theta) = 0$.
2. **Linearize around $\theta(t)$.** For large $n$, $\theta$ is close to $\theta(t)$, so a first-order Taylor expansion of the gradient around $\theta(t)$ is accurate: $0 \approx \frac1n\sum_i\nabla\ell(x_i, \theta(t)) + H(\theta - \theta(t))$. Here the sample average of the Hessians has been replaced by its expectation $H$, which the law of large numbers allows for large $n$. Solving for $\theta$,
$$\theta - \theta(t) \approx -H^{-1}\,\frac1n\sum_i\nabla\ell(x_i, \theta(t)).$$
This is the average form of @sec-prop-scaling, with influence $\psi(x) = -H^{-1}\nabla\ell(x, \theta(t))$: each record pulls the fit by its gradient, scaled by the inverse curvature.
3. **The average gradient has mean zero.** $\theta(t)$ minimizes the expected loss, so the expected gradient there is zero, $\mathbb E_{\pi_t}\nabla\ell(x, \theta(t)) = 0$. Each gradient therefore has mean zero and covariance $G$, and an average of $n$ independent copies has covariance $G/n$.
4. **Apply the central limit theorem.** An average of many independent terms is approximately Gaussian, so the average gradient is approximately $\mathcal N(0, G/n)$. Multiplying a Gaussian vector by the matrix $-H^{-1}$ multiplies its covariance by $H^{-1}$ on both sides ($H$ is symmetric), which gives $\mathcal N(0, H^{-1}GH^{-1}/n)$.

**The best attack and its accuracy.** Under the two hypotheses, $\theta$ is approximately $\mathcal N(\theta(t_0), \Sigma_1(t_0)/n)$ and $\mathcal N(\theta(t_1), \Sigma_1(t_1)/n)$. Fact 3 needs a shared covariance; when the two covariances are close, replace both by $\bar\Sigma_1/n$. Fact 3 then applies with $m_b = \theta(t_b)$ and $\Sigma = \bar\Sigma_1/n$. Since $\Sigma^{-1} = n\,\bar\Sigma_1^{-1}$, the factor $n$ comes out of the Mahalanobis distance, which gives @eq-prop-delta. Fact 3 also says the best test is linear in $\theta$: a **linear discriminant**, which is the meta-classifier to train.

**The scaling law.** For a small gap, the population fit changes almost linearly in $t$, so $\theta(t_1) - \theta(t_0) \approx \theta'(\bar t)(t_1 - t_0)$, and the covariance hardly changes, so $\bar\Sigma_1 \approx \Sigma_1(\bar t)$. Substituting into @eq-prop-delta gives $\Delta^2 \approx n(t_1 - t_0)^2\,\theta'(\bar t)^\top\Sigma_1(\bar t)^{-1}\theta'(\bar t)$, whose square root is @eq-prop-kappa.

**The released fraction.** The count fits the same mould, which checks the general formula against @sec-prop-scaling. Take $\ell(x, \theta) = \frac12(\theta - a)^2$, where $a \in \{0, 1\}$ marks the subgroup. The minimizer of the average of $\frac12(\theta - a_i)^2$ is the mean of the $a_i$, which is $\theta = S/n$. Its population version is $\theta(t) = \mathbb E[a] = t$, so $\theta' = 1$. The second derivative of the loss is $H = 1$, and the gradient $\theta(t) - a = t - a$ has variance $G = t(1 - t)$, the variance of a Bernoulli($t$). So $\Sigma_1 = t(1 - t)$ and $\kappa = 1/\sqrt{t(1 - t)}$. With this $\kappa$, the accuracy $\Phi(\Delta/2)$ turns into the advantage $2\Phi(\Delta/2) - 1$ of @eq-prop-property, because advantage is twice accuracy minus 1.

### Interpretation

**Two numbers set the leak.** $\theta'(t)$ says how much the property moves what the model learns, and $\Sigma_1$ says how noisy that is. If the property does not affect what the model fits, then $\theta' = 0$ and, to first order, nothing leaks, however large $n$ is. If it does, the leak is a side effect of fitting the data well, and no amount of care in the training code removes it.

**$\kappa$ compares any model with the count.** With the group visible, the count has $\kappa = 1/\sqrt{\bar t(1 - \bar t)}$. Suppose a model has a smaller $\kappa$. How many records $n'$ would the count need to leak as much as the model does on $n$? Setting the two values of $\Delta$ equal, $\kappa\sqrt n = \sqrt{n'}/\sqrt{\bar t(1 - \bar t)}$, so $n' = n(\kappa\sqrt{\bar t(1 - \bar t)})^2$, a fraction of the real $n$. Suri & Evans (2022) measure leakage in exactly these units, as the number of records an attacker would need to see directly to match the model ([Exercise 3](solutions/04-property-inference.md#sec-sol-property-inference-3)).

**Where the Gaussian picture fails.** The derivation used a Taylor expansion and the central limit theorem, both of which need large $n$. When $n$ is small or the model is far from its large-$n$ regime, the parameters are skewed and the linear meta-classifier is no longer optimal. A second gap comes from the attacker's own sampling. When the meta-classifier is trained on few shadow models, it estimates its direction $w$ with error, and falls short of $\Phi(\Delta/2)$. That shortfall is largest when the true effect is small, because then the noise in the shadow models swamps the shift it is trying to find.

**The real pipeline.** Ganju et al. (2018) attack networks trained on census income data and other datasets, with many shadow models per hypothesis and a neural network as the meta-classifier. The toy below keeps every step but runs in seconds.

### Example: logistic regression with a hidden subgroup

Each record has two features $x \sim \mathcal N(0, I_2)$ and a label. The majority labels with $\Pr[y = 1 \mid x] = \sigma(2x_1 + x_2)$, the subgroup with $\sigma(2x_1 - x_2)$, where $\sigma(z) = 1/(1 + e^{-z})$ is the logistic function. Think of a symptom $x_2$ that raises risk in most patients and lowers it in a minority. **The group is never a feature**: the model sees $(x_1, x_2, y)$ only. A logistic regression with weights $(w_1, w_2)$ and an intercept is trained on $n$ records, by Newton's method with a tiny ridge penalty $10^{-3}$.

The script computes $\theta(t)$ and $\Sigma_1(t)$ exactly, by Gauss–Hermite quadrature over $x$ (a numerical rule for integrals against a Gaussian), as @tbl-prop-theta shows.

| $t$ | 0 | 0.25 | 0.5 | 0.75 | 1 |
| --- | --- | --- | --- | --- | --- |
| $w_1(t)$ | 2.000 | 1.767 | 1.704 | 1.767 | 2.000 |
| $w_2(t)$ | 1.000 | 0.442 | 0.000 | −0.442 | −1.000 |

: The population fit $\theta(t)$ of the logistic regression, against the subgroup share $t$. The intercept is 0 for every $t$. {#tbl-prop-theta}

To use @eq-prop-kappa you need the slope $\theta'$ and the noise $\Sigma_1$ at the midpoint $t = \frac12$. Start with the slope, which says **the property moves $w_2$**. At $t = \frac12$ the two rules cancel in $w_2$: half the population pushes it up and half down. As $t$ grows, the subgroup's rule wins and $w_2$ goes negative, with slope $\theta'(\frac12) = (0, -1.704, 0)$ for $(w_1, w_2, \text{intercept})$. The table agrees: between $t = 0.25$ and $t = 0.75$, $w_2$ falls by $0.442 - (-0.442) = 0.884$ over a step of 0.5 in $t$, a slope of $-1.768$, close to the slope at $\frac12$. Mixing the rules also makes the labels noisier, which shrinks $w_1$ from 2 to 1.704. But the table shows it shrinks symmetrically in $t$ (1.767 at both 0.25 and 0.75), so $w_1$ is flat at $\frac12$ and carries no information there. The intercept is 0 for every $t$, so it carries none either.

Next, **the noise.** $\Sigma_1(\frac12)$ has diagonal $(12.59, 6.94, 6.04)$ for $(w_1, w_2, \text{intercept})$, and zero off-diagonal entries. The zeros come from two symmetries of the population at $t = \frac12$. Flipping the sign of $x_2$ swaps the two groups' rules, and with half of each the mixture is unchanged. The flip turns a fitted $(w_1, w_2, \text{intercept})$ into $(w_1, -w_2, \text{intercept})$, so the two are equally likely, and $w_2$ is therefore uncorrelated with the other two parameters. Flipping the sign of $x$ and swapping the labels $y \to 1 - y$ also leaves the population unchanged and flips only the sign of the intercept, which makes the intercept uncorrelated with the weights. To test the Gaussian picture, the script trains two thousand models on $n = 1{,}000$ records at $t = \frac12$. $n$ times their variances are $(11.92, 6.80, 6.08)$, and their mean $w_1$ is 1.693. The Gaussian picture is right to within about 5%; the rest is the bias of a finite sample and of the ridge penalty.

Now **the constant.** Only $w_2$ has a nonzero slope, and $\Sigma_1$ is diagonal, so @eq-prop-kappa reduces to the slope of $w_2$ over its standard deviation: $\kappa = 1.704/\sqrt{6.94} = 0.647$, against 2 for the count. By the comparison in the interpretation, the model leaks the property as well as $(0.647/2)^2 = 0.105$ of its records would: about one record in ten.

@fig-prop-parameter-cloud shows the trained models for $t_0 = 0.4$ and $t_1 = 0.6$. At $n = 100$ the two clouds overlap. At $n = 1{,}000$ they have separated along $w_2$, and a line splits them.

![The parameters $(w_1, w_2)$ of 300 logistic regressions trained with subgroup share $t = 0.4$ and 300 with $t = 0.6$, at $n = 100$ (left) and $n = 1{,}000$ (right), with the linear meta-classifier fitted to them. The clouds shrink like $1/\sqrt n$; the gap between their centres stays fixed.](figures/prop_parameter_cloud.svg){#fig-prop-parameter-cloud}

Now the attack itself. For each $n$ and gap, with $t_0, t_1$ centred on $\frac12$, the script trains 150 shadow models per hypothesis, fits the linear discriminant on their three parameters, and measures its accuracy on 150 fresh target models per hypothesis (@tbl-prop-sweep). Each accuracy is an average over 300 right-or-wrong outcomes, each with variance at most $\frac12 \cdot \frac12 = 0.25$, so its standard error is at most $\sqrt{0.25/300} = 0.029$.

| $n$ | gap 0.05 | gap 0.1 | gap 0.2 | gap 0.4 |
| --- | --- | --- | --- | --- |
| 50 | 0.523 (0.546) | 0.520 (0.590) | 0.653 (0.676) | 0.830 (0.818) |
| 100 | 0.523 (0.564) | 0.670 (0.627) | 0.733 (0.741) | 0.863 (0.901) |
| 200 | 0.503 (0.590) | 0.670 (0.676) | 0.803 (0.819) | 0.957 (0.965) |
| 500 | 0.633 (0.641) | 0.787 (0.765) | 0.910 (0.926) | 0.997 (0.998) |
| 1,000 | 0.713 (0.695) | 0.867 (0.847) | 0.980 (0.979) | 1.000 (1.000) |

: Accuracy of the meta-classifier against logistic regressions, by dataset size $n$ and gap $\lvert t_1 - t_0\rvert$, with the prediction $\Phi(\Delta/2)$ of @eq-prop-delta in parentheses. {#tbl-prop-sweep}

**The prediction holds.** Most cells agree with $\Phi(\Delta/2)$ to within two standard errors. The exceptions are at small effects ($n \le 200$, gap $\le 0.1$), where the meta-classifier, fitted to 300 noisy shadow models, finds the direction of $w$ badly. There the attack is weaker than the best test, as the interpretation predicted.

**The scaling law holds.** @eq-prop-kappa says the accuracy depends on $n$ and the gap only through $\sqrt n\,|t_1 - t_0|$. Plotted against that product, the four curves fall on one, the curve $\Phi(\kappa\sqrt n\,|t_1 - t_0|/2)$ with $\kappa = 0.647$ (@fig-prop-meta-accuracy). So a gap of 0.05 at $n = 1{,}000$ is about as easy as a gap of 0.1 at $n = 200$ or so: $\sqrt{1000}\times0.05 = 1.58$, against $\sqrt{200}\times0.1 = 1.41$.

![Left: accuracy of the meta-classifier against $n$ for four gaps $\lvert t_1 - t_0\rvert$ (150 shadow and 150 target models per hypothesis, seed 1). Right: the same accuracies against $\sqrt n\,\lvert t_1 - t_0\rvert$, with the Gaussian prediction $\Phi(\kappa\sqrt n\,\lvert t_1 - t_0\rvert/2)$, $\kappa = 0.647$.](figures/prop_meta_accuracy.svg){#fig-prop-meta-accuracy}

### Check yourself

- If the subgroup used the majority's label rule $\sigma(2x_1 + x_2)$, what would $\kappa$ be? *(Zero. $\pi_A = \pi_B$, so $\pi_t$ does not depend on $t$, $\theta' = 0$, and no attack beats a coin flip; the property is then undetectable even from the data.)*
- Why does the best meta-classifier ignore the intercept and $w_1$ in this example, even though they vary from model to model? *(Near $t = \frac12$ their means do not depend on $t$, and they are uncorrelated with $w_2$, so they add noise without signal; the linear discriminant gives them weight $\Sigma^{-1}(m_1 - m_0)$, which is zero in those coordinates.)*

## Permutation invariance {#sec-prop-symmetry}

### Context

A logistic regression has one parameter vector per function: different weights give different predictions. A neural network has many. Swap two hidden units, together with their incoming and outgoing weights, and the network computes exactly the same function. Ganju et al. (2018) observed that a meta-classifier fed the flattened weights must then learn the same pattern in every slot and every order, and does poorly. Their remedy is a representation that does not depend on the order of the units. This section works it out for a network with one hidden layer of tanh units, which has a second symmetry as well: because tanh is odd, flipping the signs of a unit's weights in and out also leaves the function unchanged.

### Statement

Let $f_\theta(x) = \sigma\big(v^\top\tanh(Wx + b) + c\big)$ be a network with $h$ hidden units. Row $j$ of $W$ and entry $b_j$ compute unit $j$'s input, $v_j$ is its outgoing weight, and $c$ is the output bias. Collect unit $j$'s parameters as $u_j = (W_{j\cdot}, b_j, v_j)$. The two symmetries are a reordering of the units and a sign flip of some of them. A **permutation matrix** $\Pi$ reorders the entries of a vector, and a diagonal matrix $S$ with entries $\pm1$ flips the sign of some of them. Applying $g = S\Pi$ to the rows of $W$ and the entries of $b$ and $v$ reorders the units and flips some. It does not change the function:
$$f_{(gW,\ gb,\ gv,\ c)} = f_{(W,\ b,\ v,\ c)}.$$ {#eq-prop-symmetry}
There are $h!$ orders and $2^h$ sign patterns, so each generic network has $h!\,2^h$ parameter vectors: $4! \times 2^4 = 24 \times 16 = 384$ for $h = 4$. Suppose the initialization is invariant under every $g$ (for example, independent entries drawn from a distribution symmetric about 0) and training is gradient descent. Then the symmetry carries over from the function to the distribution of the trained weights, with three consequences:

1. the law of the trained parameters is invariant under every $g$, under both hypotheses;
2. every coordinate of every $u_j$ has mean zero under both hypotheses, so a linear meta-classifier on the raw parameters can use none of them;
3. the second-moment features $\sum_j u_ju_j^\top$ are invariant under every $g$, and keep information that the raw means lose.

### Derivation

The plan is to show that $g$ passes through tanh, so it leaves the network's output unchanged; then that it passes through every step of gradient descent; and then to read off what this means for averages.

1. **tanh commutes with $g$.** tanh acts entry by entry, so permuting its input permutes its output in the same way. tanh is odd, $\tanh(-z) = -\tanh(z)$, so flipping the sign of an entry flips the sign of its output. Hence $\tanh(gz) = g\tanh(z)$.
2. **The output is unchanged.** Using step 1, $(gv)^\top\tanh(gWx + gb) = v^\top g^\top g\tanh(Wx + b) = v^\top\tanh(Wx + b)$, because $g^\top g = \Pi^\top S^2\Pi = I$: $S^2 = I$ since each entry is $\pm1$, and $\Pi^\top\Pi = I$ since undoing a reordering gives back the original. This is @eq-prop-symmetry. Distinct $\Pi$ and $S$ give distinct parameter vectors when the units are distinct and no $v_j$ is 0, which gives the count $h!\,2^h$.
3. **Gradient descent commutes with $g$.** Let $\mathcal L$ be the training loss and let $g$ also denote its action on the whole parameter vector, which is linear and orthogonal (it only reorders entries and flips signs). Step 2 says $\mathcal L(g\theta) = \mathcal L(\theta)$ for every $\theta$. Differentiating both sides with the chain rule gives $g^\top\nabla\mathcal L(g\theta) = \nabla\mathcal L(\theta)$, so $\nabla\mathcal L(g\theta) = g\nabla\mathcal L(\theta)$. One step of gradient descent from $g\theta$ therefore lands at $g\theta - \eta g\nabla\mathcal L(\theta) = g(\theta - \eta\nabla\mathcal L(\theta))$, where $\eta$ is the step size: $g$ applied to the step from $\theta$. Repeating, if gradient descent from $\theta_0$ reaches $\theta_k$, then from $g\theta_0$ it reaches $g\theta_k$.
4. **The law is invariant.** By assumption, the initial parameters $\theta_0$ have the same law as $g\theta_0$. Training maps $\theta_0$ to $\theta_k$ and, by step 3, $g\theta_0$ to $g\theta_k$, so $\theta_k$ and $g\theta_k$ have the same law, for every dataset. Averaging over datasets drawn under either hypothesis keeps this.
5. **Raw means vanish.** Take the $g$ that flips unit $j$ only. Invariance says $u_j$ and $-u_j$ have the same law, so $\mathbb E[u_j] = \mathbb E[-u_j] = -\mathbb E[u_j]$, which forces $\mathbb E[u_j] = 0$, under both hypotheses. The linear discriminant of Fact 3 puts weight $\Sigma^{-1}(m_1 - m_0)$ on the features, where $m_0$ and $m_1$ are the mean parameter vectors under the two hypotheses; their difference is zero in every unit coordinate, so the weight is zero apart from the output bias $c$, which no $g$ touches.
6. **Second moments are invariant.** A permutation only reorders the terms of $\sum_ju_ju_j^\top$, which does not change the sum. A sign flip turns $u_j$ into $-u_j$, and $(-u_j)(-u_j)^\top = u_ju_j^\top$.

### Interpretation

**Raw weights hide the property from a linear attacker.** Because of the symmetries, each unit slot holds a random one of the network's units, with a random sign. The property shows in *which* units the network has, not in which slot holds which unit, so averaging over networks slot by slot washes it out.

There are **three ways to undo the symmetry**. A **canonical form** picks one parameter vector per function: flip each unit so that $v_j > 0$, then sort the units by $v_j$. Ganju et al. sort neurons this way as a baseline. It is fragile, because two units with nearly equal $v_j$ swap places from one network to the next. A **set representation** feeds each unit through the same map $\phi$, adds the results, and applies a second map: $\rho\big(\sum_j\phi(u_j)\big)$. The sum does not depend on the order of the units. Ganju et al. use this DeepSets form (Zaheer et al. 2017), with $\phi$ and $\rho$ learned networks. The features $\sum_ju_ju_j^\top$ are the case $\phi(u) = uu^\top$, which is also unchanged by sign flips. **Query features** use the network's outputs on fixed probe inputs. A function does not depend on how it is parameterized, so these are invariant automatically, and they need only black-box access.

**Deeper networks** have the same problem at each layer. Each hidden layer has its own permutations, and they compose, so a set representation is applied layer by layer.

Finally, **black-box can beat white-box.** Fact 2 of [Threat models and privacy games](01-threat-models.md) says that processing the view cannot help the attacker: whoever holds the weights can compute any query output, so white-box access is never weaker *for the best attack*. That says nothing about a fixed simple attack. For a linear meta-classifier, a better representation of the same model can matter more than more access, and the example shows probe outputs beating raw weights. Zhang, Tople & Ohrimenko (2021) attack properties through queries alone.

### Example: a tanh network with four hidden units

The data are as in @sec-prop-meta, with $t_0 = 0.2$ and $t_1 = 0.8$. Each network has $h = 4$ tanh units and is trained on $n = 200$ records by 300 steps of full-batch gradient descent from a random start. On each side, 150 shadow networks fit the linear discriminant and 150 target networks measure its accuracy. The script also checks @eq-prop-symmetry directly: a network with permuted and sign-flipped units gives the same logits on 1,000 inputs, to $10^{-12}$.

Each unit has four parameters (two input weights, a bias and an output weight), so the raw parameters number $4 \times 4 + 1 = 17$, counting the output bias $c$. The canonical form reorders the same 17. The second-moment matrix $\sum_j u_ju_j^\top$ is a symmetric $4 \times 4$ matrix with $4 \times 5/2 = 10$ distinct entries, plus $c$, which gives 11. To see which features carry signal, @tbl-prop-representations also reports each representation's largest **standardized mean gap**: for each feature, the difference of its means under $t_0$ and $t_1$, divided by its pooled standard deviation.

| Representation | Features | Largest standardized mean gap | Accuracy |
| --- | --- | --- | --- |
| Raw parameters | 17 | 0.34 | 0.497 |
| Canonical order (flip so $v_j > 0$, sort by $v_j$) | 17 | 0.66 | 0.807 |
| Unit second moments $\sum_j u_ju_j^\top$ | 11 | 2.24 | 0.877 |
| Outputs on 16 probe inputs | 16 | 2.81 | 1.000 |

: The linear meta-classifier against tanh networks, by representation. The standardized gap of a feature is the difference of its means under $t_0$ and $t_1$, over its pooled standard deviation, on the shadow networks. {#tbl-prop-representations}

**Raw parameters are at chance**, as step 5 predicts. Their largest standardized gap, 0.34, is the size expected from noise alone. A standardized feature has variance 1, so its mean over 150 networks has variance $1/150$, and the difference of two such means has variance $2/150$, a standard deviation of $\sqrt{2/150} = 0.115$. The largest of 17 such gaps is typically two to three times that.

**Canonical order recovers most of the signal,** with one feature carrying most of it (@fig-prop-symmetries, left). **Second moments do better with fewer features,** because they need no tie-breaking between units with nearly equal $v_j$. **Probe outputs are best.** The property changes the function the network computes, and probe outputs measure that function directly, without passing through any parameterization.

![Left: for each representation of 100 tanh networks per hypothesis ($t = 0.2$ against $t = 0.8$), the standardized mean gap of every feature. Right: accuracy of the linear meta-classifier on 100 further networks per hypothesis (seed 1). Raw parameters carry no mean signal; invariant representations do.](figures/prop_symmetries.svg){#fig-prop-symmetries}

### Check yourself

- Would the raw means vanish for a network with ReLU units instead of tanh? *(Not by this argument. ReLU is not odd, so the sign flip is not a symmetry; only the permutations remain. They force every unit slot to have the same mean, but that common mean can differ between $t_0$ and $t_1$.)*
- Why does a linear meta-classifier on the sum $\sum_ju_j$ fail for tanh networks, even though the sum is permutation invariant? *(The sign flips still apply: $\mathbb E[u_j] = 0$ for every $j$, so the sum has mean zero under both hypotheses. An invariant feature needs to be even in each $u_j$, like $u_ju_j^\top$.)*

## Code: check every result by computation and simulation

The script [code/property_inference.py](code/property_inference.py) computes every number in this chapter and draws every figure. The binomial advantages are exact sums. The population fit and the sandwich covariance of the logistic regression are computed by Gauss–Hermite quadrature on a $40\times40$ grid. Shadow and target models are trained in batches with NumPy: logistic regressions by Newton's method, tanh networks by full-batch gradient descent. From the repository root:

```bash
uv run python book/part02-attacks/code/property_inference.py
```

**Output** (seed 0), summarized in @tbl-prop-code:

| Experiment | Result |
| --- | --- |
| Released count, $t_0 = 0.45$, $t_1 = 0.55$ | Property advantage 0.6827, 0.9985, 1.0000 at $n = 100$, 1,000, 10,000 (@tbl-prop-count) |
| Released count, membership at $t = 0.5$ | 0.0398, 0.0126, 0.0040 at the same $n$ |
| Simulated property game, $n = 100$ | Advantage 0.6811 over 200,000 games (exact 0.6827) |
| Logistic regression, population view at $t = \frac12$ | $\theta = (1.704, 0, 0)$; $\theta' = (0, -1.704, 0)$; $\Sigma_1$ diagonal $(12.59, 6.94, 6.04)$; $\kappa = 0.647$ |
| 2,000 logistic regressions, $n = 1{,}000$, $t = \frac12$ | Mean $w_1 = 1.693$; $n\times$ variances $(11.92, 6.80, 6.08)$ |
| Meta-classifier sweep | @tbl-prop-sweep |
| tanh networks, $h = 4$, $n = 200$, $t = 0.2$ against $0.8$ | Accuracy 0.497 (raw), 0.807 (canonical), 0.877 (second moments), 1.000 (probes); symmetry check passed |

: The script's output with the default seed. {#tbl-prop-code}

[tests/test_property_inference.py](../../tests/test_property_inference.py) checks that the output still matches.

**What to take from it.** Four things stand out. First, more data helps the property attacker: the same release that hides each record better as $n$ grows reveals the property better. Second, the Gaussian picture predicts real attacks: two numbers computed from the population, $\theta'$ and $\Sigma_1$, predict the meta-classifier's accuracy across a 20-fold range of $n$ and an 8-fold range of gaps. Third, hiding the group column does not hide the property: the logistic regression never sees the group, and still leaks the property as well as a tenth of its records would. Fourth, representation matters before access does: the same networks are opaque to a linear attacker in raw form and transparent once the hidden-unit symmetries are removed.

## References

- Ateniese, Mancini, Spognardi, Villani, Vitali & Felici, "Hacking Smart Machines with Smarter Ones: How to Extract Meaningful Data from Machine Learning Classifiers" (International Journal of Security and Networks, 2015). The first property-inference attack: shadow models trained with and without the property, and a meta-classifier on their parameters. Read it for the recipe and its threat model.
- Ganju, Wang, Yang, Gunter & Borisov, "Property Inference Attacks on Fully Connected Neural Networks using Permutation Invariant Representations" (CCS 2018). The permutation problem for neural networks, neuron sorting, and the DeepSets meta-classifier. The source of @sec-prop-symmetry.
- Suri & Evans, "Formalizing and Estimating Distribution Inference Risks" (PoPETs 2022). The property game as a test between two training distributions, and leakage measured in records. [Exercise 3](solutions/04-property-inference.md#sec-sol-property-inference-3) follows their definition.
- Zhang, Tople & Ohrimenko, "Leakage of Dataset Properties in Multi-Party Machine Learning" (USENIX Security 2021). Black-box property inference from query outputs, including properties of attributes that were removed from the training data.
- Mahloujifar, Ghosh & Chase, "Property Inference from Poisoning" (IEEE S&P 2022). An attacker who can add a few records to the training set can make a property far easier to infer.
- Melis, Song, De Cristofaro & Shmatikov, "Exploiting Unintended Feature Leakage in Collaborative Learning" (IEEE S&P 2019). Properties of other participants' data, inferred from the updates they share during collaborative training.
- Salem, Cherubin, Evans, Köpf, Paverd, Suri, Tople & Zanella-Béguelin, "SoK: Let the Privacy Games Begin! A Unified Treatment of Data Inference Privacy in Machine Learning" (IEEE S&P 2023). Property inference as one game among the others of this part.
- Zaheer, Kottur, Ravanbakhsh, Póczos, Salakhutdinov & Smola, "Deep Sets" (NeurIPS 2017). The $\rho(\sum_j\phi(u_j))$ form of permutation-invariant functions.
- van der Vaart, *Asymptotic Statistics* (1998), ch. 5. Asymptotic normality of M-estimators and the sandwich covariance $H^{-1}GH^{-1}$, with the regularity conditions this chapter skips.

## Cheat sheet

$\pi_t$ mixes the majority and a subgroup of share $t$; $D \sim \pi_t^n$; $\theta = T(D) \sim P_{n,t}$; $\bar t = \frac{t_0 + t_1}2$. @tbl-prop-cheat collects the results.

| Result | Formula | Note |
| --- | --- | --- |
| Ceiling on any attack | $\mathrm{Adv} \le \mathrm{TV}(P_{n,t_0}, P_{n,t_1}) \le \mathrm{TV}(\pi_{t_0}^n, \pi_{t_1}^n)$ | Equals $\mathrm{TV}(\mathrm{Bin}(n, t_0), \mathrm{Bin}(n, t_1))$ when the group is visible |
| Property, released count | $2\Phi\big(\sqrt n\,\lvert t_1 - t_0\rvert/(2\sqrt{\bar t(1 - \bar t)})\big) - 1$ | $\to 1$ as $n$ grows |
| Membership, released count | $(1 - t)\max_k\Pr[\mathrm{Bin}(n - 1, t) = k] \approx (1 - t)/\sqrt{2\pi nt(1 - t)}$ | $\to 0$ like $1/\sqrt n$ |
| Two Gaussians, shared $\Sigma$ | Best accuracy $\Phi(\Delta/2)$; linear test $w = \Sigma^{-1}(m_1 - m_0)$ | $\Delta$ the Mahalanobis distance |
| Trained parameters | $\theta \approx \mathcal N(\theta(t), H^{-1}GH^{-1}/n)$ | M-estimators, large $n$ |
| Meta-classifier accuracy | $\Phi(\kappa\sqrt n\,\lvert t_1 - t_0\rvert/2)$, $\kappa = \lvert\theta'\rvert_{\Sigma_1^{-1}}$ | Count: $\kappa = 1/\sqrt{\bar t(1 - \bar t)}$ |
| Hidden-unit symmetry | $f_{(gW, gb, gv, c)} = f_{(W, b, v, c)}$, $g = S\Pi$ | $h!\,2^h$ copies; raw means vanish for tanh |

: Every result of the chapter. {#tbl-prop-cheat}

**Rules of thumb**

- Property inference gets easier with more data, membership inference harder. What matters for a property is $\sqrt n\,|t_1 - t_0|$.
- A property changes about $n|t_1 - t_0|$ records at once. Defences sized for one record do not cover it.
- Removing the sensitive column does not remove the property. Whatever correlates with it moves the model.
- Train the meta-classifier on a representation that respects the model's symmetries: canonical order, sums over units, or outputs on probe inputs.
- Report accuracy for property attacks: there is one secret per dataset, not a population of targets.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](solutions/04-property-inference.md).
