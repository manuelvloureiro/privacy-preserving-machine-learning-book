# Exercises: Algorithmic stability and generalization

Exercises for [Algorithmic stability and generalization](../../book-full/part01-prerequisites/07-stability-generalization.md), each with a worked solution. Try each exercise before opening its solution. The numbers come from [code/solutions_stability.py](../../code/part01-prerequisites/solutions_stability.py); run it from the repository root with `uv run python code/part01-prerequisites/solutions_stability.py`. Exercises 5 and 7 are about stochastic gradient descent (SGD), Exercise 5 compares it with regularized empirical risk minimization (ERM), and Exercise 6 is about nearest neighbour (NN) rules.

## Removal against replacement {#sec-sol-stability-1}

> **Exercise 1.** **Removal against replacement.** Define removal stability by $|\ell(\mathcal T(D); z) - \ell(\mathcal T(D^{\setminus i}); z)| \le \gamma_{\text{remove}}$, where $D^{\setminus i}$ drops $z_i$. Show that $\mathcal T$ is then $2\gamma_{\text{remove}}$-uniformly stable in the replace-one sense. *Hint: go from $D$ to $D^{(i)}$ through $D^{\setminus i}$.* *(Triangle inequality)*

::: {.callout-tip collapse="true" title="Solution"}
The key observation is that $D$ and $D^{(i)}$ have the same set with $z_i$ dropped. Take $n = 3$ and $i = 2$:

| Training set | Entries |
| --- | --- |
| $D$ | $(z_1, z_2, z_3)$ |
| $D^{(2)}$ | $(z_1, z_2', z_3)$ |
| $D^{\setminus 2}$ | $(z_1, z_3)$ |
| $(D^{(2)})^{\setminus 2}$ | $(z_1, z_3)$ |

: Dropping record 2 from $D$ and from $D^{(2)}$ gives the same set. {#tbl-sol-stability-removal}

As @tbl-sol-stability-removal shows, dropping position 2 erases the only place where the two sets differ. In general $(D^{(i)})^{\setminus i} = D^{\setminus i}$.

1. **Insert the midpoint.** For any test point $z$, add and subtract $\ell(\mathcal T(D^{\setminus i}); z)$:
$$\ell(\mathcal T(D); z) - \ell(\mathcal T(D^{(i)}); z) = \big[\ell(\mathcal T(D); z) - \ell(\mathcal T(D^{\setminus i}); z)\big] + \big[\ell(\mathcal T(D^{\setminus i}); z) - \ell(\mathcal T(D^{(i)}); z)\big].$$
2. **First bracket.** Removal stability applied to the training set $D$ and index $i$ bounds it by $\gamma_{\text{remove}}$ in absolute value.
3. **Second bracket.** Removal stability holds for *every* training set, so apply it to $D^{(i)}$ and index $i$. Since $(D^{(i)})^{\setminus i} = D^{\setminus i}$, it bounds $|\ell(\mathcal T(D^{(i)}); z) - \ell(\mathcal T(D^{\setminus i}); z)|$ by $\gamma_{\text{remove}}$.
4. **Triangle inequality.** $|\ell(\mathcal T(D); z) - \ell(\mathcal T(D^{(i)}); z)| \le 2\gamma_{\text{remove}}$ for every $D$, $i$, $z_i'$ and $z$. This is @eq-stability-uniform with $\gamma = 2\gamma_{\text{remove}}$.

For a randomized $\mathcal T$, the same steps go through with $\mathbb E_{\mathcal T}$ in front of each loss, because the triangle inequality holds for the expected differences too.

*Removal and replacement stability are interchangeable up to a factor of 2, which is why the Bousquet–Elisseeff bound can be stated in either form.*
:::

## The renaming table for every $p$ {#sec-sol-stability-2}

> **Exercise 2.** **The renaming table for every $p$.** Redo @tbl-stability-renaming with general $p$ and show that the expected gap is $p(1 - p)$. *Check: $0.21$ at $p = 0.3$, and the largest gap is $0.25$, at $p = \frac12$.* *(Renaming trick)*

::: {.callout-tip collapse="true" title="Solution"}
The algorithm is $\mathcal T(D) = z_1/2$ and the loss is $\ell(\theta; z) = (\theta - z)^2$, so the model is $0$ after $z_1 = 0$ and $\frac12$ after $z_1 = 1$. The losses in each row do not depend on $p$; only the probabilities do.

| $(z_1, z_1')$ | Probability | Train $\ell(\mathcal T(D); z_1)$ | Test $\ell(\mathcal T(D); z_1')$ | Renamed $\ell(\mathcal T(D^{(1)}); z_1)$ |
| --- | --- | --- | --- | --- |
| $(0, 0)$ | $(1 - p)^2$ | $0$ | $0$ | $0$ |
| $(0, 1)$ | $(1 - p)p$ | $0$ | $1$ | $\tfrac14$ |
| $(1, 0)$ | $p(1 - p)$ | $\tfrac14$ | $\tfrac14$ | $1$ |
| $(1, 1)$ | $p^2$ | $\tfrac14$ | $\tfrac14$ | $\tfrac14$ |

: Every outcome of the one-example case for a general $p = \Pr[z_1 = 1]$. {#tbl-sol-stability-renaming}

1. **Expected training loss.** Only the rows with $z_1 = 1$ contribute, each with loss $\frac14$: $\frac14\big(p(1 - p) + p^2\big) = \frac p4$.
2. **Expected test loss.** $1\cdot(1 - p)p + \frac14 p(1 - p) + \frac14p^2 = p(1 - p) + \frac p4$.
3. **Expected renamed loss.** $\frac14(1 - p)p + 1\cdot p(1 - p) + \frac14p^2 = p(1 - p) + \frac p4$, the same as step 2. The values $1$ and $\frac14$ have swapped between the rows $(0, 1)$ and $(1, 0)$, which have the same probability $p(1 - p)$ (@lem-stability-swap).
4. **Expected gap.** Subtract step 1 from step 2: $\mathbb E[\mathrm{gap}] = p(1 - p)$. By step 3 the right side of @eq-stability-identity gives the same value.
5. **Largest gap.** $\frac{\mathrm d}{\mathrm dp}\,p(1 - p) = 1 - 2p$ vanishes at $p = \frac12$, and the second derivative is $-2 < 0$, so the maximum is $\frac12\cdot\frac12 = 0.25$.

At $p = 0.3$ the gap is $0.3 \times 0.7 = 0.21$, the check value. @tbl-sol-stability-renaming-values lists the enumerated expectations from the script, which agree with the formulas. A grid search over $p \in \{0, 0.001, \dots, 1\}$ finds the largest gap $0.2500$ at $p = 0.500$.

| $p$ | Train $\frac p4$ | Test | Renamed | Gap | $p(1 - p)$ |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.0250 | 0.1150 | 0.1150 | 0.0900 | 0.0900 |
| 0.3 | 0.0750 | 0.2850 | 0.2850 | 0.2100 | 0.2100 |
| 0.5 | 0.1250 | 0.3750 | 0.3750 | 0.2500 | 0.2500 |
| 0.7 | 0.1750 | 0.3850 | 0.3850 | 0.2100 | 0.2100 |
| 0.9 | 0.2250 | 0.3150 | 0.3150 | 0.0900 | 0.0900 |

: Enumerated expectations of the one-example case for several $p$. {#tbl-sol-stability-renaming-values}

The stability is $\gamma = 0.75$ for every $p$, since it is a worst case over outcomes and does not see $p$. The gap never exceeds $0.25$, a third of $\gamma$.

*The gap is largest when $z_1$ is least predictable: a coin with $p = \frac12$ makes the training point most different, on average, from a fresh one.*
:::

## The sample mean {#sec-sol-stability-3}

> **Exercise 3.** **The sample mean.** Let $z_i \in [0, 1]$ with variance $\sigma^2$, $\mathcal T(D) = \frac1n\sum_iz_i$ and $\ell(\theta; z) = (\theta - z)^2$. Show that $\gamma = 2/n$, and that the expected gap is exactly $2\sigma^2/n$. *Check: for fair coins and $n = 10$, the gap is $0.05$ and the bound $0.2$.* *(Uniform stability, bias of the training loss)*

::: {.callout-tip collapse="true" title="Solution"}
Write $\bar z = \mathcal T(D)$ and $\mu = \mathbb E z_i$.

**Stability.**

1. **The model moves by at most $1/n$.** $D$ and $D^{(i)}$ differ in one term of the average, so $\theta - \theta' = \mathcal T(D) - \mathcal T(D^{(i)}) = (z_i - z_i')/n$, and $|\theta - \theta'| \le 1/n$ because $z_i, z_i' \in [0, 1]$.
2. **Factor the loss difference.** $(\theta - z)^2 - (\theta' - z)^2 = (\theta - \theta')(\theta + \theta' - 2z)$.
3. **Bound the second factor.** $\theta$, $\theta'$ and $z$ all lie in $[0, 1]$, so $\theta + \theta' - 2z \in [-2, 2]$.
4. **Combine.** $|\ell(\theta; z) - \ell(\theta'; z)| \le \frac1n \cdot 2 = \frac2n$ for every $D$, $i$, $z_i'$ and $z$. So $\mathcal T$ is $\gamma$-uniformly stable with $\gamma = 2/n$.

The bound is tight up to $1/n^2$. Steps 1 and 3 cannot be tight at once: $|\theta - \theta'| = 1/n$ forces $\{z_i, z_i'\} = \{0, 1\}$, so $\theta$ and $\theta'$ cannot both equal $z \pm 1$. The best case puts every other point at 1, $z_i = 0$, $z_i' = 1$ and $z = 0$: then $\theta = (n-1)/n$, $\theta' = 1$, and the loss difference is $1 - ((n-1)/n)^2 = (2n - 1)/n^2$. The script maximizes over all corner configurations and finds $0.1900 = 19/100$ at $n = 10$, against the bound $0.2000$.

**Expected gap.**

5. **Expected training loss.** Expand around $\mu$: $\sum_i(z_i - \bar z)^2 = \sum_i(z_i - \mu)^2 - n(\bar z - \mu)^2$, because the cross term is $-2(\bar z - \mu)\sum_i(z_i - \mu) = -2n(\bar z - \mu)^2$. Take expectations, using $\mathbb E(z_i - \mu)^2 = \sigma^2$ and $\operatorname{Var}(\bar z) = \sigma^2/n$ (independence): $\mathbb E\sum_i(z_i - \bar z)^2 = n\sigma^2 - \sigma^2$. Divide by $n$:
$$\mathbb E\big[\mathcal L(\bar z; D)\big] = \frac{n - 1}{n}\sigma^2.$$
6. **Expected test loss.** A fresh $z$ is independent of $\bar z$, so $\mathbb E(\bar z - z)^2 = \operatorname{Var}(\bar z - z) + (\mathbb E\bar z - \mathbb E z)^2 = \frac{\sigma^2}n + \sigma^2 + 0$:
$$\mathbb E\big[\mathcal L_{\mathcal D}(\bar z)\big] = \frac{n + 1}{n}\sigma^2.$$
7. **Gap.** Subtract: $\mathbb E[\mathrm{gap}] = \frac{n + 1 - (n - 1)}{n}\sigma^2 = \frac{2\sigma^2}n$.

**Concrete case, $n = 2$ fair coins** ($\sigma^2 = \frac14$). The outcomes $(0, 0)$ and $(1, 1)$ have $\bar z = z_1 = z_2$ and training loss 0. The outcomes $(0, 1)$ and $(1, 0)$ have $\bar z = \frac12$ and training loss $\frac14$. So the expected training loss is $\frac12 \cdot \frac14 = 0.125 = \frac{1}{2}\sigma^2$, as step 5 says. The test loss is $\frac32\sigma^2 = 0.375$, and the gap $0.25 = 2\sigma^2/2$. The script's exact enumeration gives train $0.1250$, test $0.3750$, gap $0.2500$.

**Check.** For fair coins, $\sigma^2 = \frac14$. At $n = 10$ the gap is $2 \cdot \frac14/10 = 0.05$ and the bound $2/10 = 0.2$. Exact enumeration over the number of ones gives train $0.2250 = \frac{9}{10}\cdot\frac14$, test $0.2750 = \frac{11}{10}\cdot\frac14$ and gap $0.0500$.

*The sample mean overfits by $2\sigma^2/n$, the familiar bias of the uncorrected sample variance counted twice: once because the training points pull $\bar z$ towards themselves, once because a fresh point does not. Stability gets the $1/n$ rate right and is loose by the factor $(2/n)/(2\sigma^2/n) = 1/\sigma^2$, which is at least 4 because $\sigma^2 \le \frac14$ on $[0, 1]$.*
:::

## Leave-one-out is unbiased {#sec-sol-stability-4}

> **Exercise 4.** **Leave-one-out is unbiased.** Show that $\mathbb E[\ell(\mathcal T(D^{\setminus i}); z_i)] = \mathbb E[\mathcal L_{\mathcal D}(\mathcal T(\tilde D))]$, where $\tilde D$ is a training set of size $n - 1$. Why does this make leave-one-out cross-validation an almost unbiased estimate of the test loss? *(Independence, step 1 of the proof of @thm-stability-identity)*

::: {.callout-tip collapse="true" title="Solution"}
1. **$z_i$ is a fresh point for $\mathcal T(D^{\setminus i})$.** $D^{\setminus i}$ consists of the $z_j$ with $j \ne i$, which are independent of $z_i$. So $z_i$ plays the role of $z_i'$ in step 1 of the proof of @thm-stability-identity.
2. **Condition on $D^{\setminus i}$.** The model $\mathcal T(D^{\setminus i})$ is then fixed and $z_i \sim \mathcal D$ is independent of it, so $\mathbb E[\ell(\mathcal T(D^{\setminus i}); z_i) \mid D^{\setminus i}] = \mathcal L_{\mathcal D}(\mathcal T(D^{\setminus i}))$. Take expectations: $\mathbb E[\ell(\mathcal T(D^{\setminus i}); z_i)] = \mathbb E[\mathcal L_{\mathcal D}(\mathcal T(D^{\setminus i}))]$.
3. **$D^{\setminus i}$ is a training set of size $n - 1$.** It is $n - 1$ i.i.d. draws from $\mathcal D$, so it has the same distribution as $\tilde D$, and $\mathbb E[\mathcal L_{\mathcal D}(\mathcal T(D^{\setminus i}))] = \mathbb E[\mathcal L_{\mathcal D}(\mathcal T(\tilde D))]$.

**Why leave-one-out is almost unbiased.** The leave-one-out estimate is $\frac1n\sum_i\ell(\mathcal T(D^{\setminus i}); z_i)$. By linearity and the result above, its expectation is exactly $\mathbb E[\mathcal L_{\mathcal D}(\mathcal T(\tilde D))]$: the expected test loss of the algorithm trained on $n - 1$ points. The only bias is the difference between training on $n - 1$ and on $n$ points. If $\mathcal T$ has removal stability $\gamma_{\text{remove}}$ (Exercise 1), apply it to $D$ and index $n$: $|\ell(\mathcal T(D); z) - \ell(\mathcal T(D^{\setminus n}); z)| \le \gamma_{\text{remove}}$ for every $z$, so $|\mathcal L_{\mathcal D}(\mathcal T(D)) - \mathcal L_{\mathcal D}(\mathcal T(D^{\setminus n}))| \le \gamma_{\text{remove}}$, and since $D^{\setminus n}$ has the law of $\tilde D$,
$$\big|\mathbb E[\mathcal L_{\mathcal D}(\mathcal T(D))] - \mathbb E[\mathcal L_{\mathcal D}(\mathcal T(\tilde D))]\big| \le \gamma_{\text{remove}}.$$
For a stable algorithm this bias is of order $1/n$. Compare the training loss: it evaluates $\mathcal T(D)$ on points it was fitted to, and its bias is the whole gap.

**Concrete case: the sample mean of Exercise 3.** The mean of the other $n - 1$ points has variance $\sigma^2/(n - 1)$ and is independent of $z_i$, so the expected leave-one-out loss is $\sigma^2 + \frac{\sigma^2}{n - 1} = \frac{n}{n - 1}\sigma^2$. That is the test loss $\frac{m + 1}{m}\sigma^2$ of Exercise 3 at size $m = n - 1$. For fair coins and $n = 10$, exact enumeration gives:

| Estimate | Value |
| --- | --- |
| Leave-one-out | 0.27778 |
| Test loss at size $n - 1 = 9$ | 0.27778 |
| Test loss at size $n = 10$ | 0.27500 |
| Training loss | 0.22500 |

: Expected losses of the sample mean of 10 fair coins. {#tbl-sol-stability-loo}

The leave-one-out estimate overshoots the test loss at size 10 by $\sigma^2/(n(n - 1)) = 0.25/90 \approx 0.0028$, while the training loss undershoots it by $0.05$ (@tbl-sol-stability-loo).

*Leaving the evaluated point out removes the dependence that biases the training loss; what remains is the small cost of training on one fewer record. (Unbiasedness says nothing about variance: the $n$ terms share most of their training data, so they are correlated.)*
:::

## SGD with a strongly convex loss {#sec-sol-stability-5}

> **Exercise 5.** **SGD with a strongly convex loss.** If each $\ell(\cdot; z)$ is $\lambda$-strongly convex and $\beta_s$-smooth (on a bounded domain, since a strongly convex function cannot be Lipschitz everywhere), the gradient step with $\eta \le 2/(\beta_s + \lambda)$ is $(1 - \eta\beta_s\lambda/(\beta_s + \lambda))$-contractive (Hardt, Recht & Singer 2016, Lemma 3.7). With constant $\eta$, show that $\mathbb E r_T \le \frac{2L}{n}\cdot\frac{\beta_s + \lambda}{\beta_s\lambda}$ for every $T$. *Check: for $\lambda \ll \beta_s$ this is about $2L/(\lambda n)$, the ERM bound: with strong convexity, training longer no longer costs stability.* *(SGD coupling, geometric series)*

::: {.callout-tip collapse="true" title="Solution"}
Run SGD on $D$ and $D^{(i)}$ with the same index sequence, as in the proof of @thm-stability-sgd, and write $\kappa = 1 - \eta\beta_s\lambda/(\beta_s + \lambda) < 1$ for the contraction factor and $\psi_j(\theta) = \theta - \eta\nabla\ell(\theta; z_j)$ for the gradient step on record $j$.

1. **Same record picked** ($I_t \ne i$, probability $1 - 1/n$). Both runs apply the same map $\psi_{I_t}$, which is $\kappa$-contractive: $r_{t+1} \le \kappa\,r_t$.
2. **Differing record picked** ($I_t = i$, probability $1/n$). Split the difference so that the contraction still applies:
$$\theta_{t+1} - \theta_{t+1}' = \big[\psi_i(\theta_t) - \psi_i(\theta_t')\big] + \eta\big[\nabla\ell(\theta_t'; z_i') - \nabla\ell(\theta_t'; z_i)\big].$$
The first bracket has norm at most $\kappa\,r_t$; the second at most $2\eta L$, since both gradients have norm at most $L$ on the domain. So $r_{t+1} \le \kappa\,r_t + 2\eta L$.
3. **Take expectations** over $I_t$, which is independent of $r_t$: $\mathbb E r_{t+1} \le \kappa\,\mathbb E r_t + \frac{2\eta L}n$. (In step 3 of the proof of @thm-stability-sgd, without strong convexity, the factor in front of $\mathbb E r_t$ was 1; here it is $\kappa$ in both cases.)
4. **Unroll** from $r_0 = 0$. After one step, $\mathbb E r_1 \le \frac{2\eta L}n$; after two, $\le \frac{2\eta L}n(1 + \kappa)$; after $T$,
$$\mathbb E r_T \le \frac{2\eta L}n\sum_{k=0}^{T-1}\kappa^k = \frac{2\eta L}n\cdot\frac{1 - \kappa^T}{1 - \kappa}.$$
5. **Bound the geometric series** by its infinite sum, $\frac{1}{1 - \kappa} = \frac{\beta_s + \lambda}{\eta\beta_s\lambda}$. The learning rate cancels:
$$\mathbb E r_T \le \frac{2\eta L}n\cdot\frac{\beta_s + \lambda}{\eta\beta_s\lambda} = \frac{2L}n\cdot\frac{\beta_s + \lambda}{\beta_s\lambda} \quad\text{for every } T.$$

**Check.** If $\lambda \ll \beta_s$, then $\frac{\beta_s + \lambda}{\beta_s\lambda} = \frac1\lambda + \frac1{\beta_s} \approx \frac1\lambda$, so $\mathbb E r_T \lesssim \frac{2L}{\lambda n}$. That is the weight bound of @thm-stability-erm. Multiplying by $L$, because the loss is $L$-Lipschitz, gives $\gamma \le 2L^2/(\lambda n)$, the regularized ERM bound, independent of $T$.

**A numerical check.** Take the running example with the regularizer moved into each per-example loss, $\ell(\theta; z) + \frac\lambda2\|\theta\|^2$, with $\lambda = 0.1$, $n = 100$ and $\eta = 0.25$. This loss is $\lambda$-strongly convex and $\beta_s$-smooth with $\beta_s = \frac14 + \lambda = 0.35$, and $\eta \le 2/(\beta_s + \lambda) \approx 4.44$. The logistic part of the gradient has norm at most $\|x\| = 1$, so, starting from 0 and with $\eta\lambda \le 1$, an SGD step gives $\|\theta_{t+1}\| \le (1 - \eta\lambda)\|\theta_t\| + \eta$, so the iterates stay in the ball $\|\theta\| \le 1/\lambda$, where the gradient norm is at most $1 + \lambda\cdot\frac1\lambda = 2$: $L = 2$. Then $\kappa = 0.98056$ and the bound for every $T$ is $\frac{4}{100}\cdot\frac{0.45}{0.035} = 0.5143$. Averaged over 100 coupled pairs of runs, with $z_1$ replaced by a fresh draw:

| Epochs | Mean $\| \theta_T - \theta_T'\|$ | Strongly convex bound $\frac{2\eta L}n\frac{1 - \kappa^T}{1 - \kappa}$ | Convex bound $\frac{2\eta L T}{n}$ |
| --- | --- | --- | --- |
| 1 | 0.0451 | 0.4421 | 1.0 |
| 5 | 0.0591 | 0.5143 | 5.0 |
| 20 | 0.0528 | 0.5143 | 20.0 |
| 40 | 0.0584 | 0.5143 | 40.0 |
| 100 | 0.0554 | 0.5143 | 100.0 |

: Divergence of coupled SGD runs on $\lambda$-regularized logistic regression ($\lambda = 0.1$, $n = 100$, $\eta = 0.25$) against the two bounds. {#tbl-sol-stability-sgd}

The measured divergence levels off after a few epochs at about 0.055 (it fluctuates by a few thousandths across rows, which is Monte Carlo noise), and the strongly convex bound levels off at 0.5143, while the convex bound grows linearly with $T$ (@tbl-sol-stability-sgd).

*With strong convexity, the contraction forgets old differences at a geometric rate, so each new use of the differing record adds a shrinking amount and the divergence saturates.*
:::

## $k$-nearest neighbours {#sec-sol-stability-6}

> **Exercise 6.** **$k$-nearest neighbours.** Change the script to use $k = 15$ neighbours with majority vote, and compare its gap with 1-NN for $n = 200$ and $n = 1000$. Then show that its worst-case uniform stability is still $\gamma = 1$. What does this say about uniform stability as a measure? *(Unstable algorithms, worst case against average case)*

::: {.callout-tip collapse="true" title="Solution"}
**The experiment.** The solution script replaces the nearest-neighbour search by the $k$ largest inner products (the features are unit vectors, so these are the $k$ nearest points) and predicts the sign of the sum of their labels; $k = 15$ is odd, so there are no ties. A training point counts as one of its own neighbours. Both rules run on the same 10 training sets and the same 2,000 test points:

| $n$ | 1-NN train | 1-NN test | 1-NN gap | 15-NN train | 15-NN test | 15-NN gap |
| --- | --- | --- | --- | --- | --- | --- |
| 200 | 0.000 | 0.379 | 0.379 | 0.246 | 0.316 | 0.070 |
| 1,000 | 0.000 | 0.369 | 0.369 | 0.243 | 0.289 | 0.046 |

: 0-1 error of 1-NN and 15-NN in the running example, averaged over 10 training sets. {#tbl-sol-stability-knn}

The 15-NN gap is five to eight times smaller than the 1-NN gap, and it shrinks with $n$, while the 1-NN gap stays near its limit 0.338 (@tbl-sol-stability-knn). The 15-NN training error is close to the Bayes error 0.250, so it is an honest estimate, and its test error is better than 1-NN's too.

**Worst-case stability is still 1.** The 0-1 loss lies in $\{0, 1\}$, so $\gamma \le 1$ always. To show $\gamma = 1$, build one training set where one replacement flips the loss at one test point.

1. **The training set.** Fix a test point $z = (x, +1)$. Put 15 distinct training points close to $x$, 8 labelled $+1$ and 7 labelled $-1$, and put the other $n - 15$ points far away.
2. **Before.** The 15 nearest neighbours of $x$ are the close points. The vote is $8 - 7 = +1$, so the prediction is $+1$ and the loss at $z$ is 0.
3. **The replacement.** Replace one of the $+1$ close points, $z_i = (x_i, +1)$, by $z_i' = (x_i, -1)$, the same point with its label flipped. The neighbours of $x$ do not change, but the vote becomes $7 - 8 = -1$.
4. **After.** The prediction is $-1$ and the loss at $z$ is 1. So $|\ell(\mathcal T(D); z) - \ell(\mathcal T(D^{(i)}); z)| = 1$, and $\gamma = 1$.

The script builds this configuration (15 points near $e_1$, test point $e_1$) and confirms the loss goes from 0 to 1.

**What it says about uniform stability.** Uniform stability is a worst case over training sets, replacements and test points. It cannot tell 1-NN from 15-NN, although one has a gap of 0.37 and the other of 0.05. The configuration in step 1, a test point with a vote exactly on the edge, is rare under the data distribution; @eq-stability-identity says the expected gap is the *average* replace-one effect, which for 15-NN is small. An average-case notion of stability, which averages over the data instead of taking the worst case, is what explains why $k$-NN generalizes.

*Uniform stability is sufficient for a small gap but far from necessary: it misses algorithms that are stable on typical data.*
:::

## The decaying learning rate {#sec-sol-stability-7}

> **Exercise 7.** **The decaying learning rate.** For $\eta_t = c/t$, show that the bound @eq-stability-hrs is at most $\frac{2L^2c(1 + \ln T)}{n}$. With $T = n$ and constant $\eta = 1/\sqrt n$ instead, show $\gamma \le 2L^2/\sqrt n$, and explain why @eq-stability-be is then vacuous while the bounds of Feldman & Vondrák and of Bousquet, Klochkov & Zhivotovskiy are not. *(SGD stability, high-probability rates)*

::: {.callout-tip collapse="true" title="Solution"}
**Decaying step.** The bound needs $\eta_t \le 2/\beta_s$ for every $t$; the largest step is $\eta_1 = c$, so take $c \le 2/\beta_s$.

1. **Sum the steps.** $\sum_{t=1}^T\eta_t = c\sum_{t=1}^T\frac1t$.
2. **Compare with an integral.** For $t \ge 2$, $\frac1t \le \int_{t-1}^t\frac{\mathrm ds}s$, because $1/s \ge 1/t$ on $[t - 1, t]$. Summing, $\sum_{t=2}^T\frac1t \le \int_1^T\frac{\mathrm ds}s = \ln T$, so $\sum_{t=1}^T\frac1t \le 1 + \ln T$.
3. **Substitute** into @eq-stability-hrs: $\gamma \le \frac{2L^2}n\cdot c(1 + \ln T) = \frac{2L^2c(1 + \ln T)}n$.

For $c = 1$ and $T = n = 10{,}000$, the harmonic sum is $9.7876$, below $1 + \ln T = 10.2103$, and the bound is $0.002042$ (with $L = 1$). The learning rates shrink fast enough that $\gamma$ grows only logarithmically in $T$.

**Constant step $1/\sqrt n$ for $T = n$ steps.** $\sum_{t=1}^n\eta_t = n\cdot\frac1{\sqrt n} = \sqrt n$, so @eq-stability-hrs gives $\gamma \le \frac{2L^2}n\sqrt n = \frac{2L^2}{\sqrt n}$.

**Why @eq-stability-be is vacuous.** Its stability term is $2n\gamma\sqrt{\ln(1/\beta)/(2n)} = \gamma\sqrt{2n\ln(1/\beta)}$. With $\gamma = 2L^2/\sqrt n$ the $\sqrt n$ cancels and the term is $2L^2\sqrt{2\ln(1/\beta)}$, a constant that does not fall with $n$. For $L = 1$ and $\beta = 0.05$ it is $4.895$. With $K = 1$ the gap is at most 1 anyway, so a width near 5 says nothing. This is the $\gamma \sim 1/\sqrt n$ regime of @fig-stability-high-probability.

**Why the improved bounds are not.** Plug $\gamma = 2/\sqrt n$ into their orders ($L = K = 1$):

- Feldman & Vondrák (2018): $\sqrt{(\gamma + 1/n)\ln(1/\beta)} \approx \sqrt{2\ln(1/\beta)}\,n^{-1/4}$, which goes to zero.
- Bousquet, Klochkov & Zhivotovskiy (2020): $\gamma\ln n\ln(1/\beta) + \sqrt{\ln(1/\beta)/n} = \frac{2\ln n\ln(1/\beta)}{\sqrt n} + \sqrt{\frac{\ln(1/\beta)}n}$, which is $\tilde O(1/\sqrt n)$.

Both depend on $\gamma$ linearly or through a square root, not through $n\gamma$, so they only need $\gamma \to 0$. @tbl-sol-stability-rates shows the three at $\beta = 0.05$. The last two columns drop the unknown constants, so they show the rate, not a usable bound.

| $n$ | $\gamma = 2/\sqrt n$ | Width of @eq-stability-be | $\sqrt{(\gamma + 1/n)\ln(1/\beta)}$ | $\gamma\ln n\ln(1/\beta) + \sqrt{\ln(1/\beta)/n}$ |
| --- | --- | --- | --- | --- |
| 1,000 | 0.0632 | 4.997 | 0.439 | 1.364 |
| 10,000 | 0.0200 | 4.928 | 0.245 | 0.569 |
| 1,000,000 | 0.0020 | 4.899 | 0.077 | 0.085 |

: High-probability width with $\gamma = 2/\sqrt n$, $K = 1$, $\beta = 0.05$: the Bousquet–Elisseeff width, and the orders of the Feldman–Vondrák (2018) and Bousquet–Klochkov–Zhivotovskiy (2020) bounds with constants dropped. {#tbl-sol-stability-rates}

*One pass of SGD with step $1/\sqrt n$ has $\gamma \sim 1/\sqrt n$, exactly where McDiarmid's worst-case charging breaks down and the sharper bounds are needed.*
:::

## Self-influence and label noise {#sec-sol-stability-8}

> **Exercise 8.** **Self-influence and label noise.** Change the label probability to $\sigma(20x_1)$, which makes labels less noisy, and rerun the self-influence experiment at $\lambda = 0.01$. Predict before running: does the gap go up or down, and what happens to the share of records against the trend? *(Memorization)*

::: {.callout-tip collapse="true" title="Solution"}
**Prediction.**

1. **Share against the trend.** A record goes against the trend when $y x_1 < 0$, which happens with probability $\sigma(-s|x_1|)$ given $x_1$, for label probability $\sigma(sx_1)$. That is $\min(p, 1 - p)$, so the expected share against the trend equals the Bayes error. Raising $s$ from 5 to 20 makes $\sigma(-s|x_1|)$ smaller at every $x_1 \ne 0$, so the share should fall well below a quarter. By quadrature it falls from $0.250$ to $0.079$.
2. **Gap.** @exm-stability-self-influence found that records against the trend have the largest self-influence, and @eq-stability-identity makes the gap the average self-influence. Fewer such records should mean a smaller gap. Against-trend records should still be the most memorized ones, since they are still the ones the trend cannot fit.

**Result.** The solution script repeats the chapter's experiment ($n = 40$, $\lambda = 0.01$, 100 training sets, each record replaced by a fresh draw in turn) with $\Pr[y = 1 \mid x] = \sigma(20x_1)$:

| Label probability | Gap | Average self-influence | Share against trend | Self-influence against | Self-influence with |
| --- | --- | --- | --- | --- | --- |
| $\sigma(5x_1)$ (chapter) | $0.1482 \pm 0.0073$ | $0.1422 \pm 0.0025$ | 0.24 | 0.284 | 0.097 |
| $\sigma(20x_1)$ | $0.0896 \pm 0.0051$ | $0.0918 \pm 0.0016$ | 0.082 | 0.254 | 0.077 |

: Self-influence of regularized logistic regression, $n = 40$, $\lambda = 0.01$, for two levels of label noise, with ± one standard error. {#tbl-sol-stability-noise}

Both predictions hold (@tbl-sol-stability-noise). The gap falls by about 40%, and the gap and the average self-influence still agree within their errors, as the identity requires. The share against the trend drops to 0.082, close to the quadrature value 0.079. Against-trend records are still about three times more memorized than the others (0.254 against 0.077), so most of the drop in the gap comes from there being fewer of them. The average training loss is 0.3083 and the test loss 0.3979.

*The gap is driven by the records whose label could have gone either way; less label noise means fewer of them to memorize.*
:::

## The best of $k$ models on a validation set {#sec-sol-stability-9}

> **Exercise 9.** **The best of $k$ models on a validation set.** You trained $k = 100$ models on other data, and each is right on a new record with probability $p = 0.8$, independently of the others. You score them on a validation set of $m = 1{,}000$ records and report the best score. Using the derivation of @eq-stability-fixed-questions, show that the winner's validation accuracy overstates its true accuracy by at most $\sqrt{\ln k/(2m)}$ on average. Compare this bound with a simulation and with a normal approximation, and show that scoring the winner again on a fresh test set of $m$ records removes the bias. *Check: the bound is 0.0480; the simulated bias is 0.0313 and the normal approximation 0.0317; on the fresh test set the bias is $-0.0001 \pm 0.0002$.* *(Maximal inequality)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One model.** The validation accuracy of model $j$ is an average of $m$ independent values, 1 for a right prediction and 0 for a wrong one. Write $Z_j$ for its accuracy minus $p$. By Hoeffding's lemma, $\mathbb E\,e^{sZ_j} \le e^{s^2/(8m)}$ for every real $s$. The models were trained on other data, so each is fixed before the validation set is drawn, and the lemma applies.
2. **The winner.** The winner's score minus $p$ is $\max_jZ_j$, because all models have the same true accuracy. Only the largest value matters, so the maximum runs over $k$ variables, not the $2k$ of @eq-stability-fixed-questions, which also covered errors in the other direction.
3. **The bound.** Steps 2 to 4 of the proof of @thm-stability-fixed-questions, with $k$ in place of $2k$, gives $\mathbb E\max_jZ_j \le \sqrt{\ln k/(2m)} = \sqrt{\ln 100/2{,}000} = 0.0480$.
4. **Simulation.** The script draws each model's validation score as a binomial count of right predictions out of 1,000, with success probability 0.8, for 4,000 validation sets. The winner's score exceeds 0.8 by 0.0313 on average.
5. **Normal approximation.** Each score has standard deviation $\sqrt{p(1 - p)/m} = 0.01265$. The largest of 100 independent standard normals has mean 2.5076, computed by numerical integration, so the largest score exceeds $p$ by about $2.5076 \times 0.01265 = 0.0317$. The bound of step 3 is larger mainly because Hoeffding's lemma uses the worst variance, $\frac14$ per record, while the true variance is $p(1 - p) = 0.16$.
6. **Fresh test set.** The winner is chosen before the test set is drawn, so on the test set it is a fixed model, and its test accuracy is an unbiased estimate of $p$. The simulation gives a bias of $-0.0001 \pm 0.0002$.

*Choosing among models that are fixed in advance costs only $\sqrt{\ln k}$. The bias is real, about 3 points here, but a fresh measurement of the single winner removes it.*
:::
