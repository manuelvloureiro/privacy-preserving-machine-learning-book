# Private empirical risk minimization

## Why this matters

Most of machine learning is **empirical risk minimization (ERM)**: choose the parameters $w$ that minimize the average loss on the training data, usually plus a penalty on the size of $w$. The trained weights are a function of every training record, and [Membership inference](../part02-attacks/02-membership-inference.md) showed that they give some of those records away. Differential privacy (DP) promises the opposite: the distribution of whatever you release changes by at most a factor $e^\varepsilon$, plus a slack $\delta$, when one record is replaced. To keep that promise for a trained model, randomness has to enter somewhere between the data and the released weights. This chapter asks where.

There are three natural places, and each gives a classic algorithm. **Output perturbation** trains as usual and adds noise to the final weights. **Objective perturbation** adds a random linear term to the objective, then minimizes it exactly. **Gradient perturbation** adds noise to every gradient during training. All three were analysed for convex losses between 2008 and 2014, by Chaudhuri, Monteleoni and Sarwate, by Kifer, Smith and Thakurta, and by Bassily, Smith and Thakurta, and the third is the ancestor of the algorithm almost every private deep-learning system runs today: stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, DP-SGD, the subject of the [next chapter](02-dp-sgd.md).

### The one idea: privacy needs a bound on how far one record can move what you release

Every DP mechanism in this book works the same way. Find a quantity whose change, when one record is replaced, is bounded by a known number, its **sensitivity**. Then add noise large enough to hide a change of that size. The three methods differ only in *which* quantity they bound.

Output perturbation bounds the minimizer itself. Strong convexity, the curvature a ridge penalty $\frac\lambda2\|w\|^2$ adds, guarantees that replacing one of $n$ records moves the minimizer by at most $2L/(\lambda n)$, where $L$ bounds the length of each record's loss gradient ([Optimization](../part01-prerequisites/06-optimization.md)). Objective perturbation bounds the *gradient* of the objective at every point, which also moves by at most $2L/n$, and turns that into a bound on the output's density through a change of variables. Gradient perturbation bounds the gradient at the points the algorithm actually visits, one step at a time, and composes the steps.

The order matters because each step back toward the data needs fewer assumptions. Output perturbation needs an exact, unique minimizer of a strongly convex objective. Objective perturbation needs that and a loss with bounded curvature. Gradient perturbation, for its privacy, needs only that each record's gradient is bounded: no convexity, no exact solution. That is why it is the one that survives in deep learning.

@tbl-dperm-overview lists the results and where each is used.

| Result | Where it is used |
| --- | --- |
| Output perturbation: noise of scale $2L/(\lambda n)$ on the minimizer | Convex models trained to convergence; the baseline the others are measured against |
| Sensitivity of an approximate minimizer | Why output perturbation needs the exact solution, and how much an inexact one costs |
| Objective perturbation: a random linear term, minimized exactly | Private logistic regression and linear support vector machines; why curvature in the data helps |
| Noisy gradient descent: Gaussian noise on every gradient | The privacy argument of [DP-SGD](02-dp-sgd.md); every iterative private trainer |
| Excess risk of all three, and the optimal rates | Choosing between them; what no method can beat ([Lower bounds](../part03-dp-foundations/09-lower-bounds.md)) |
| What survives without convexity | Why deep learning uses gradient perturbation, with clipping |

: The results of this chapter and where each is used. {#tbl-dperm-overview}

**Notation.** A training set $D = (z_1, \dots, z_n)$ holds $n$ records $z_i = (x_i, y_i)$ with features $x_i \in \mathbb R^d$ and labels $y_i \in \{-1, +1\}$. The loss of weights $w$ on a record is $\ell(w, z)$, and the regularized objective is
$$J(w; D) = \frac1n\sum_{i=1}^n\ell(w, z_i) + \frac\lambda2\|w\|^2,$$
with minimizer $w^*(D)$. Neighbouring datasets $D \simeq D'$ differ by **replacing** one record, $z$ in $D$ by $z'$ in $D'$, so both have size $n$; the next chapter switches to adding or removing one, and says why. $L$ bounds the length of every gradient $\nabla_w\ell(w, z)$, $\|\cdot\|$ is the Euclidean norm, and a released, private version of the weights is written $\tilde w$. The **excess risk** of a release is $J(\tilde w; D) - J(w^*; D)$, how much worse it fits the training objective than the exact minimizer.

**The running example.** Regularized logistic regression, with loss $\ell(w, (x, y)) = \ln(1 + e^{-y\langle w, x\rangle})$, in $d = 10$ dimensions with ridge $\lambda = 0.01$. The features are uniform on the unit sphere, $\|x\| = 1$, and the labels come from a logistic model whose true weights have length 5. The derivative of $\ln(1 + e^{-m})$ in the margin $m = y\langle w, x\rangle$ lies in $(-1, 0)$, so each gradient has length at most $\|x\| = 1$ and $L = 1$. Every method targets $\varepsilon = 1$, with $\delta = 10^{-5}$ for those that need a $\delta$, on $n$ from 1,000 to 100,000 records.

**How to read this chapter.** Read the toolkit, then @sec-dperm-output, which sets the pattern. @sec-dperm-objective has the chapter's one long derivation; its one-dimensional example shows each step on numbers. @sec-dperm-gradient is the one to know well, because the next chapter builds on it. @sec-dperm-compare and @sec-dperm-nonconvex say which method to use when.

## Toolkit: five facts every derivation uses

### Fact 1 — The Gaussian mechanism, calibrated exactly

To release a vector-valued function $f(D)$ whose value moves by at most $\Delta$ in Euclidean length between neighbours (its **$\ell_2$ sensitivity**), add independent normal noise $\mathcal N(0, \sigma^2)$ to each coordinate. What matters for privacy is only the ratio $\mu = \Delta/\sigma$, the shift measured in noise standard deviations: rotate so the shift lies along one axis, and the other coordinates have the same distribution under both datasets. The mechanism is then $(\varepsilon, \delta)$-DP exactly when
$$\delta \ge \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^\varepsilon\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big),$$ {#eq-dperm-gauss-delta}
where $\Phi$ is the standard normal distribution function ([The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md), Balle & Wang 2018). The right side increases with $\mu$, so for each $(\varepsilon, \delta)$ there is a largest admissible ratio $\mu(\varepsilon, \delta)$, and the smallest noise is $\sigma = \Delta/\mu(\varepsilon, \delta)$. For $\varepsilon = 1$ and $\delta = 10^{-5}$, $\mu = 0.2681$, so $\sigma = 3.7306\,\Delta$. The classic formula $\sigma = \Delta\sqrt{2\ln(1.25/\delta)}/\varepsilon$ gives $4.8448\,\Delta$, 30% more noise for the same guarantee; this chapter always uses the exact calibration.

### Fact 2 — The regularized minimizer moves by at most $2L/(\lambda n)$

[Optimization](../part01-prerequisites/06-optimization.md) proves this, and the argument is short enough to restate. Replacing $z$ by $z'$ changes $J$ by $h(v) = \frac1n(\ell(v, z') - \ell(v, z))$, whose gradient has length at most $2L/n$: one gradient leaves, another arrives. At the new minimizer $w' = w^*(D')$, the old objective's gradient is therefore $\nabla J(w'; D) = -\nabla h(w')$, of length at most $2L/n$. The ridge makes $J(\cdot; D)$ **$\lambda$-strongly convex**: its gradient grows by at least $\lambda$ per unit of distance in every direction, so a point where the gradient has length $g$ is within $g/\lambda$ of the minimizer. Hence
$$\|w^*(D) - w^*(D')\| \le \frac{2L}{\lambda n}.$$ {#eq-dperm-sens}
The same argument gives a bound for any point, not only the exact minimizer: if $\|\nabla J(w; D)\| \le g$, then $\|w - w^*(D)\| \le g/\lambda$.

### Fact 3 — Noise with density proportional to $e^{-\|b\|/s}$

Pure $\varepsilon$-DP, with $\delta = 0$, needs noise whose density falls off no faster than exponentially in every direction. The standard choice in $d$ dimensions has density $\nu(b) \propto e^{-\|b\|/s}$: it depends on $b$ only through its length. If $f$ has $\ell_2$ sensitivity $\Delta$ and $s = \Delta/\varepsilon$, then $f(D) + b$ is $\varepsilon$-DP. At any output $y$, the ratio of the two output densities is $e^{(\|y - f(D')\| - \|y - f(D)\|)/s}$. By the triangle inequality the exponent is at most $\|f(D) - f(D')\|/s \le \Delta/s = \varepsilon$.

To sample it, split $b$ into a direction and a length. The density depends only on the length $r$, and the sphere of radius $r$ has surface area proportional to $r^{d - 1}$, so the length has density proportional to $r^{d-1}e^{-r/s}$. That is a **gamma distribution** with shape $d$ and scale $s$, and the direction is uniform on the sphere. Its moments are $\mathbb E\|b\| = ds$ and $\mathbb E\|b\|^2 = d(d + 1)s^2$. Compare the Gaussian, whose squared length has mean $d\sigma^2$: the pure-DP noise is longer by a factor that grows with $d$.

### Fact 4 — Change of variables for densities

If a random vector $b$ has density $\nu$ and $w$ is obtained from $b$ by a smooth invertible map, write the inverse as $b = \phi(w)$. Then $w$ has density
$$p(w) = \nu\big(\phi(w)\big)\,\big|\det\nabla\phi(w)\big|.$$ {#eq-dperm-cov}
The determinant corrects for stretching: if $\phi$ squeezes a small box around $w$ into a box of half the volume around $b$, the probability of that box is the same, so the density at $w$ is half the density at $b$. In one dimension this is the familiar $p(w) = \nu(\phi(w))|\phi'(w)|$.

### Fact 5 — Adaptive Gaussian steps compose into one Gaussian

Run $T$ Gaussian mechanisms in sequence, each allowed to depend on the outputs of the earlier ones, with sensitivity-to-noise ratios $\mu_1, \dots, \mu_T$. The whole sequence is exactly as private as a single Gaussian mechanism with ratio
$$\mu = \sqrt{\mu_1^2 + \dots + \mu_T^2}.$$ {#eq-dperm-gdp}
This is composition of Gaussian differential privacy (GDP) from [f-DP and Gaussian differential privacy](../part04-privacy-accounting/04-gaussian-dp.md). The reason is that the privacy loss of each step is a normal random variable with variance $\mu_t^2$ and mean $\mu_t^2/2$, given the past. Their sum is normal with variance $\sum_t\mu_t^2$ and half that as its mean, which is the privacy loss of one Gaussian with ratio $\mu$. So $T$ equal steps with ratio $\mu_{\text{step}}$ give $\mu = \sqrt T\,\mu_{\text{step}}$, and @eq-dperm-gauss-delta converts the total to $(\varepsilon, \delta)$ with no slack.

## Output perturbation {#sec-dperm-output}

### Context

The simplest private trainer treats training as a black box: compute the exact minimizer $w^*(D)$, then add noise. Fact 2 says the minimizer is a function with $\ell_2$ sensitivity $2L/(\lambda n)$, so this is just the Gaussian mechanism (or Fact 3's mechanism) applied to it. Chaudhuri & Monteleoni (2008) introduced it for logistic regression. Two questions remain: how much accuracy does it cost, and what happens when, as always in practice, the minimizer is only computed approximately?

### Statement

Release
$$\tilde w = w^*(D) + Z, \qquad Z \sim \mathcal N\big(0, \sigma^2I\big), \qquad \sigma = \frac{2L}{\lambda n\,\mu(\varepsilon, \delta)}.$$ {#eq-dperm-output}
This is $(\varepsilon, \delta)$-DP for replace-one neighbours. Replacing $Z$ by noise with density $\propto e^{-\|b\|/s}$ and $s = 2L/(\lambda n\varepsilon)$ gives pure $\varepsilon$-DP. If the objective has Hessian $H$ at $w^*$, the excess risk is, to second order,
$$\mathbb E\big[J(\tilde w) - J(w^*)\big] \approx \frac12\,\mathbb E\big[Z^\top HZ\big] = \frac{\mathbb E\|Z\|^2}{2d}\operatorname{tr}H,$$ {#eq-dperm-output-risk}
for any noise whose direction is uniform, so $\frac{\sigma^2}{2}\operatorname{tr}H$ for the Gaussian. An approximate minimizer $w$ with $\|\nabla J(w; D)\| \le g$ has sensitivity at most
$$\frac{2L}{\lambda n} + \frac{2g}{\lambda}.$$ {#eq-dperm-approx}

### Derivation

1. **Privacy.** By Fact 2, $w^*$ moves by at most $\Delta = 2L/(\lambda n)$ between neighbours. Adding Gaussian noise with $\sigma = \Delta/\mu(\varepsilon, \delta)$ is $(\varepsilon, \delta)$-DP by Fact 1, and adding noise $\propto e^{-\|b\|/s}$ with $s = \Delta/\varepsilon$ is $\varepsilon$-DP by Fact 3. Nothing else about the data reaches the output.
2. **Expand the objective around its minimizer.** The gradient vanishes at $w^*$, so a second-order Taylor expansion gives $J(w^* + Z) - J(w^*) \approx \frac12Z^\top HZ$, with $H = \nabla^2J(w^*)$. The noise is small next to the scale on which the curvature changes in the cases of interest; @sec-dperm-compare checks the approximation against simulation.
3. **Average over the direction.** Write $Z = \|Z\|u$ with $u$ a unit vector. For a uniformly random direction, $\mathbb E[u_iu_j]$ is $1/d$ when $i = j$, because the $d$ squared coordinates of a unit vector add up to 1 and are exchangeable, and 0 otherwise, by symmetry under flipping one coordinate's sign. So $\mathbb E[u^\top Hu] = \sum_iH_{ii}/d = \operatorname{tr}H/d$, and if the length is independent of the direction, as for both noises here, $\mathbb E[Z^\top HZ] = \mathbb E\|Z\|^2\operatorname{tr}H/d$. For the Gaussian, $\mathbb E\|Z\|^2 = d\sigma^2$. This is @eq-dperm-output-risk.
4. **Approximate minimizers.** Let $w$ and $w'$ be the outputs of the optimizer on $D$ and $D'$, with gradients of length at most $g$. By the last line of Fact 2, each is within $g/\lambda$ of its exact minimizer. The triangle inequality along the path $w \to w^*(D) \to w^*(D') \to w'$ gives $\|w - w'\| \le g/\lambda + 2L/(\lambda n) + g/\lambda$, which is @eq-dperm-approx. The stopping rule must be a gradient check, not a step count: only the gradient bound is guaranteed on every dataset.

### Interpretation

**The noise is calibrated to the worst case, the damage to the actual case.** The sensitivity $2L/(\lambda n)$ uses only the curvature $\lambda$ that the ridge guarantees. The damage, @eq-dperm-output-risk, is measured by the actual curvature $H$, which includes the data's own. Put the two together for the Gaussian: the excess risk is about $\frac{\operatorname{tr}H}{2}\cdot\frac{4L^2}{\lambda^2n^2\mu^2}$. If every eigenvalue of $H$ is close to some $h$, this is $\frac{2dL^2}{\mu^2n^2}\cdot\frac{h}{\lambda^2}$. The factor $h/\lambda^2$ grows as the ridge shrinks, whatever the data look like. A small ridge buys a better non-private fit and costs a lot of noise.

**It falls like $1/n^2$.** The noise standard deviation falls like $1/n$, so its square, and the excess risk with it, falls like $1/n^2$. The statistical error of the non-private minimizer falls only like $1/n$, so for large $n$ privacy eventually costs less than sampling does, in this convex, low-dimensional setting.

**It needs the exact solution.** @eq-dperm-approx adds $2g/\lambda$ to the sensitivity. To keep that below the exact term, the optimizer must reach a gradient of length $g \le L/n$, which shrinks with the dataset. An optimizer stopped after a fixed number of steps has no such guarantee on every dataset, and then the sensitivity bound, and the privacy proof, do not apply. Wu et al. (2017) analyse output perturbation of SGD run for a fixed number of passes directly, with a different sensitivity bound.

![Left: two neighbouring regularized logistic objectives in one dimension ($n = 20$, $\lambda = 0.5$), their minimizers (vertical lines) 0.083 apart, below the bound $2L/(\lambda n) = 0.2$. Right: the released weight's density under each dataset with Gaussian noise for $\varepsilon = 1$, $\delta = 10^{-5}$; the noise must hide a shift as large as the bound, not the actual shift.](figures/erm_output_1d.svg){#fig-dperm-output-1d}

### Example: one dimension, then the running example

Take one feature, $n = 20$ records and $\lambda = 0.5$ (@fig-dperm-output-1d). Replacing the last record, which has $x = 1$, by the same feature with the opposite label moves the minimizer from 0.2128 to 0.1296, by 0.0832. The bound is $2/(0.5 \times 20) = 0.2$, and that is what the noise is calibrated to: $\sigma = 3.7306 \times 0.2 = 0.7461$. The two output densities overlap almost completely, as they must.

In the running example ($d = 10$, $\lambda = 0.01$), $n = 10{,}000$ gives sensitivity $2/(0.01 \times 10{,}000) = 0.02$ and $\sigma = 0.07461$ per coordinate, next to a minimizer of length 2.886. The Hessian at $w^*$ has eigenvalues between 0.0263 and 0.0330, and trace 0.3129. @eq-dperm-output-risk predicts $\frac12 \times 0.07461^2 \times 0.3129 = 8.711 \times 10^{-4}$, and 20 simulated releases average $9.274 \times 10^{-4}$. The pure-DP version, with noise $\propto e^{-\|b\|/s}$ and $s = 0.02$, has $\mathbb E\|b\|^2 = 110 \times 0.02^2$; the prediction $6.885 \times 10^{-4}$ and the simulated $8.321 \times 10^{-4}$ differ by more, because the squared length of gamma noise varies a lot from draw to draw and 20 draws average it only roughly.

Stopping early is costly. With a gradient tolerance of $g = 10^{-6}$ the sensitivity is 0.0202, essentially exact. With $g = 10^{-4}$ it doubles to 0.04, and with $g = 10^{-3}$ it is 0.22, eleven times the exact value. A tolerance that looks tight, $10^{-3}$, multiplies the noise by eleven and the excess risk by about 121.

### Check yourself

- You halve the ridge $\lambda$. By what factor does the Gaussian noise grow, and roughly by what factor does the excess risk grow if the data's curvature dominates $H$? *(The noise doubles, since $\sigma \propto 1/\lambda$. With $H$ unchanged the excess risk, proportional to $\sigma^2\operatorname{tr}H$, grows fourfold.)*
- Why must the stopping rule be "stop when $\|\nabla J\| \le g$" rather than "stop after 50 Newton steps"? *(The privacy proof needs the sensitivity bound on every pair of neighbouring datasets. A gradient check guarantees it everywhere; a step count only guarantees it on datasets where 50 steps happen to be enough.)*

## Objective perturbation {#sec-dperm-objective}

### Context

Output perturbation wastes the data's curvature: it pays for the worst case $\lambda$ and is judged on the actual $H$. Chaudhuri, Monteleoni & Sarwate (2011) put the noise inside the optimization instead. Add a random linear term $\frac1n\langle b, w\rangle$ to the objective and minimize exactly. A linear term is a constant force; the objective's curvature, all of it, resists that force. Where the data add curvature, the minimizer moves less. The price is a subtler privacy proof, because the output is no longer "something fixed plus noise".

### Statement

Let the loss have the form $\ell(w, (x, y)) = g(y\langle w, x\rangle)$ with $g$ convex and twice differentiable, $|g'| \le 1$ and $0 \le g'' \le c$, and let $\|x\| \le 1$; so the gradients have length at most $L = 1$. Logistic regression qualifies with $c = \frac14$. Set
$$\varepsilon' = \varepsilon - 2\ln\Big(1 + \frac{c}{n\lambda}\Big),$$ {#eq-dperm-eps-prime}
draw $b$ with density $\nu(b) \propto e^{-\varepsilon'\|b\|/2}$ (Fact 3 with $s = 2/\varepsilon'$), and release
$$\tilde w = \arg\min_w\Big[J(w; D) + \frac1n\langle b, w\rangle\Big].$$ {#eq-dperm-objective}
If $\varepsilon' > 0$, this is $\varepsilon$-DP for replace-one neighbours (Chaudhuri, Monteleoni & Sarwate 2011, Algorithm 2). If not, increase the ridge to $\lambda + \lambda_+$ with $\lambda_+ = \frac{c}{n(e^{\varepsilon/4} - 1)} - \lambda$, which makes the logarithm $\varepsilon/4$ and lets $\varepsilon' = \varepsilon/2$.

The minimizer moves, to first order, by $-H^{-1}b/n$, so the excess risk is about
$$\mathbb E\big[J(\tilde w) - J(w^*)\big] \approx \frac{1}{2n^2}\mathbb E\big[b^\top H^{-1}b\big] = \frac{\mathbb E\|b\|^2}{2dn^2}\operatorname{tr}H^{-1}.$$ {#eq-dperm-objective-risk}

### Derivation

The plan is to write the density of $\tilde w$ explicitly with Fact 4, then bound the ratio of the two densities, for $D$ and $D'$, at every point. The ratio splits into a noise factor and a stretching factor, and each consumes part of $\varepsilon$.

1. **From the output back to the noise.** The perturbed objective is $\lambda$-strongly convex, so it has exactly one minimizer, where its gradient is zero: $\nabla J(\tilde w; D) + b/n = 0$. Solving for $b$,
$$b = \phi_D(\tilde w) = -n\nabla J(\tilde w; D) = -\sum_{i=1}^n\nabla\ell(\tilde w, z_i) - n\lambda\tilde w.$$
Every $w$ is the output for exactly one $b$, namely $\phi_D(w)$, so the map from noise to output is invertible and $\phi_D$ is its inverse.
2. **The density of the output.** By Fact 4, $p_D(w) = \nu(\phi_D(w))\,|\det\nabla\phi_D(w)|$, and the same for $D'$ with $\phi_{D'}$. The ratio at any $w$ is
$$\frac{p_D(w)}{p_{D'}(w)} = \underbrace{\frac{\nu(\phi_D(w))}{\nu(\phi_{D'}(w))}}_{\text{noise}}\cdot\underbrace{\frac{|\det\nabla\phi_D(w)|}{|\det\nabla\phi_{D'}(w)|}}_{\text{Jacobian}}.$$
3. **The noise factor is at most $e^{\varepsilon'}$.** The sums in $\phi_D$ and $\phi_{D'}$ share every term except the replaced record's, so $\phi_{D'}(w) - \phi_D(w) = \nabla\ell(w, z) - \nabla\ell(w, z')$, of length at most $2$. The noise factor is $e^{\varepsilon'(\|\phi_{D'}(w)\| - \|\phi_D(w)\|)/2}$, and by the triangle inequality the exponent is at most $\varepsilon'\cdot 2/2 = \varepsilon'$.
4. **The Jacobian is the Hessian.** $\nabla\phi_D(w) = -n\nabla^2J(w; D) = -A$, with $A = \sum_ig''(m_i)x_ix_i^\top + n\lambda I$, where $m_i = y_i\langle w, x_i\rangle$ is the margin; the chain rule gives each loss's Hessian as $g''(m_i)$ times $y_i^2x_ix_i^\top = x_ix_i^\top$. Every term is positive semi-definite and $n\lambda I$ is added, so all eigenvalues of $A$ are at least $n\lambda$. For $D'$ the matrix is $A' = A + E$ with $E = g''(m')x'x'^\top - g''(m)xx^\top$: one rank-one term in, one out.
5. **The Jacobian factor is at most $(1 + c/(n\lambda))^2$.** The ratio of determinants is $\det(A')/\det(A) = \det(I + A^{-1}E) = \det(I + M)$ with $M = A^{-1/2}EA^{-1/2}$, since $A^{-1}E$ and $M$ are similar matrices. $M$ is symmetric and is the difference of two rank-one positive semi-definite matrices, $g''(m')\tilde x'\tilde x'^\top - g''(m)\tilde x\tilde x^\top$ with $\tilde x = A^{-1/2}x$. A rank-one matrix $vv^\top$ has one nonzero eigenvalue, $\|v\|^2$, and here $\|\tilde x\|^2 = x^\top A^{-1}x \le \|x\|^2/(n\lambda) \le 1/(n\lambda)$. So $M$ has at most two nonzero eigenvalues, each between $-a$ and $a$ with $a = c/(n\lambda)$, and $\det(I + M) = (1 + \lambda_1)(1 + \lambda_2) \le (1 + a)^2$. Exchanging the roles of $D$ and $D'$ gives the same bound for the inverse ratio, because $A'$ also has all eigenvalues at least $n\lambda$.
6. **Add the two budgets.** The log of the density ratio is at most $\varepsilon' + 2\ln(1 + c/(n\lambda)) = \varepsilon$ by @eq-dperm-eps-prime, at every $w$ and in both directions. A bound on the density ratio everywhere is pure $\varepsilon$-DP.
7. **When $\varepsilon' \le 0$.** The Jacobian factor alone would exceed $\varepsilon$. A larger ridge shrinks $a$: with $\lambda + \lambda_+$ in place of $\lambda$, $c/(n(\lambda + \lambda_+)) = e^{\varepsilon/4} - 1$, so $2\ln(1 + a) = \varepsilon/2$, and the noise gets the other half.
8. **Excess risk.** To first order in $b$, the perturbed minimizer solves $H(\tilde w - w^*) + b/n = 0$, so $\tilde w - w^* = -H^{-1}b/n$. Substituting this into the second-order expansion $J(\tilde w) - J(w^*) \approx \frac12(\tilde w - w^*)^\top H(\tilde w - w^*)$ of step 2 of @sec-dperm-output gives $\frac{1}{2n^2}b^\top H^{-1}HH^{-1}b = \frac{1}{2n^2}b^\top H^{-1}b$. Averaging over the uniform direction, as in step 3 there, replaces $b^\top H^{-1}b$ by $\mathbb E\|b\|^2\operatorname{tr}H^{-1}/d$, which is @eq-dperm-objective-risk.

**Walk-through in one dimension.** With $d = 1$ everything in steps 1–5 is a number, and the script computes both densities exactly on a grid of outputs. Take $n = 5$ records with features in $[-1, 1]$, $\lambda = 0.1$ and $\varepsilon = 1$. Then $c/(n\lambda) = 0.25/0.5 = 0.5$, the Jacobian budget is $2\ln 1.5 = 0.8109$, and the noise gets $\varepsilon' = 0.1891$. @fig-dperm-objective-density shows one dataset: the two densities, and their log ratio, which stays inside $\pm\varepsilon$. Over 500 random datasets and replacements, the largest log ratio anywhere on the grid is 0.3779. Its noise part never exceeds 0.0992, about half its budget of 0.1891, and its Jacobian part never exceeds 0.3395, well under its budget of 0.8109, and under even $\ln 1.5 = 0.4055$, half of it.

![Objective perturbation in one dimension, $n = 5$, $\lambda = 0.1$, $\varepsilon = 1$. Left: the exact densities of the released weight under a dataset and its neighbour, computed by change of variables. Right: their log ratio, the privacy loss, which stays well inside $\pm\varepsilon$.](figures/erm_objective_density.svg){#fig-dperm-objective-density}

### Interpretation

**Where the slack is.** Both factors in step 2 are bounded by worst cases that cannot happen together, and mostly cannot happen at all for logistic regression. The noise factor reaches $e^{\varepsilon'}$ only if the old and new record have full-length gradients pointing in opposite directions. A logistic gradient has full length only when its margin is very negative, and two records whose gradients point in opposite directions have opposite margins, so at most one of them can be saturated: the difference has length at most about 1, not 2, which is why the noise part stops at half its budget. The Jacobian bound charges both the record that leaves and the one that arrives the full curvature $c$, while their eigenvalues $\lambda_1 \ge 0 \ge \lambda_2$ have opposite signs, so the product $(1 + \lambda_1)(1 + \lambda_2)$ is at most $1 + a$: the Jacobian needs only half its budget ([Exercise 6](../../book-exercises/part05-dp-ml/01-dp-erm.md#sec-sol-dperm-6)). The analysis is safe, not tight.

**Curvature helps.** Compare @eq-dperm-objective-risk with @eq-dperm-output-risk. In output perturbation the noise has length about $1/(\lambda n)$ and the damage is weighted by $H$; in objective perturbation the noise has length about 1, is divided by $n$, and the damage is weighted by $H^{-1}$. If every eigenvalue of $H$ is close to $h$, the ratio of the two excess risks is about $(h/\lambda)^2$ (with $\varepsilon' \approx \varepsilon$). When the ridge supplies all the curvature, $h = \lambda$ and the two are equivalent. When the data supply more, objective perturbation wins quadratically.

**What it needs.** Exact minimization, twice-differentiable losses (the hinge loss of a support vector machine must be smoothed first), a bound $c$ on the curvature, and a loss that depends on $w$ through one inner product, which makes each record's Hessian rank one. Kifer, Smith & Thakurta (2012) extend it to Gaussian noise with $(\varepsilon, \delta)$-DP and to constrained problems, and Iyengar et al. (2019) give a version that tolerates approximate minimization by adding a little output noise on top.

### Example: the running example at $n = 10{,}000$

At $n = 10{,}000$ the Jacobian budget is $2\ln(1 + 0.25/100) = 0.0050$, so $\varepsilon' = 0.9950$: almost all of $\varepsilon$ goes to the noise. (At $n = 100$ the split would be $\varepsilon' = 0.5537$, and the Jacobian would take almost half.) The noise has $\mathbb E\|b\|^2 = d(d+1)(2/\varepsilon')^2$, and $\operatorname{tr}H^{-1} = 320.7$, so @eq-dperm-objective-risk predicts an excess risk of $7.127 \times 10^{-5}$; 20 simulated releases average $8.156 \times 10^{-5}$. The pure-DP output perturbation of @sec-dperm-output, with the same $\varepsilon$, had $8.321 \times 10^{-4}$: objective perturbation is ten times better. The rule of thumb $(h/\lambda)^2$ predicts this: the objective's curvature, ridge included, lies between 0.0263 and 0.0330, about three times the ridge's 0.01, and $3^2 = 9$.

### Check yourself

- Why does the proof need the minimizer to be unique? *(Step 1 inverts the map from noise to output. If two weights both minimized the perturbed objective, the output would not be a function of $b$, and Fact 4 would not apply.)*
- If you double $n$, what happens to the share of $\varepsilon$ that the Jacobian consumes? *(It roughly halves: $2\ln(1 + c/(n\lambda)) \approx 2c/(n\lambda)$ for large $n$.)*

## Gradient perturbation: noisy gradient descent {#sec-dperm-gradient}

### Context

The first two methods release one point and need the exact minimizer. Real training is iterative and stops early. Gradient perturbation, analysed by Song, Chaudhuri & Sarwate (2013) and Bassily, Smith & Thakurta (2014), makes each step of the optimizer private and composes the steps. Here is its simplest form, full-batch gradient descent with Gaussian noise. Its privacy argument uses nothing about convexity; convexity enters only when you ask how accurate it is.

### Statement

Start from $w_0 = 0$ and run $T$ steps of
$$w_{t+1} = w_t - \eta\big(\nabla J(w_t; D) + \xi_t\big), \qquad \xi_t \sim \mathcal N(0, \sigma^2I)\ \text{independent}, \qquad \sigma = \frac{2L}{n}\cdot\frac{\sqrt T}{\mu(\varepsilon, \delta)}.$$ {#eq-dperm-noisy-gd}
Releasing all the iterates $w_1, \dots, w_T$ is $(\varepsilon, \delta)$-DP for replace-one neighbours, for any loss whose per-record gradients have length at most $L$, convex or not.

For a $\lambda$-strongly convex, $\beta$-smooth objective (curvature between $\lambda$ and $\beta$ everywhere) and step $\eta = 1/\beta$, the excess risk of the last iterate is approximately
$$\mathbb E\big[J(w_T) - J(w^*)\big] \approx \Big(1 - \frac\lambda\beta\Big)^T\big(J(w_0) - J(w^*)\big) + \frac{d\sigma^2}{4\beta},$$ {#eq-dperm-gd-risk}
an optimization error that shrinks with $T$ plus a noise floor that grows with $T$, because $\sigma^2 \propto T$.

### Derivation

1. **One step is a Gaussian mechanism.** Given $w_t$, the step releases $\nabla J(w_t; D) + \xi_t$. The ridge term $\lambda w_t$ is the same on both datasets, and the averages differ only in the replaced record, so the gradient moves by $\frac1n\|\nabla\ell(w_t, z) - \nabla\ell(w_t, z')\| \le 2L/n$. Its ratio is $\mu_{\text{step}} = 2L/(n\sigma)$. The point $w_t$ was itself computed from earlier noisy gradients, but it is public at that moment, which is what adaptive composition allows.
2. **Compose.** By Fact 5, $T$ such steps are exactly as private as one Gaussian with $\mu = \sqrt T\cdot 2L/(n\sigma)$. Setting this to $\mu(\varepsilon, \delta)$ gives the $\sigma$ of @eq-dperm-noisy-gd. The iterates are computed from the noisy gradients and $w_0$ alone, so releasing them is post-processing.
3. **Where convexity was not used.** Steps 1 and 2 bounded the gradient's change by $2L/n$ at the point $w_t$, whatever $w_t$ is. Nothing required a minimizer, convexity or smoothness.
4. **Optimization error.** Without noise, gradient descent with step $1/\beta$ on a $\lambda$-strongly convex, $\beta$-smooth function shrinks $J(w_t) - J(w^*)$ by at least a factor $1 - \lambda/\beta$ per step ([Optimization](../part01-prerequisites/06-optimization.md)), which gives the first term.
5. **The noise floor.** Near the minimizer, the gradient is about $H(w_t - w^*)$, so the error $e_t = w_t - w^*$ evolves as $e_{t+1} \approx (I - \eta H)e_t - \eta\xi_t$. Along an eigenvector of $H$ with curvature $h$, the error's variance $v$ settles where one step leaves it unchanged: $v = (1 - \eta h)^2v + \eta^2\sigma^2$, so $v = \eta^2\sigma^2/(1 - (1 - \eta h)^2) = \eta\sigma^2/(h(2 - \eta h))$. Its cost is $\frac12hv = \eta\sigma^2/(2(2 - \eta h))$, about $\eta\sigma^2/4$ when $\eta h$ is small. Summing over the $d$ directions with $\eta = 1/\beta$ gives $d\sigma^2/(4\beta)$, and the curvature $h$ has cancelled: directions with little curvature wander farther but cost less per unit of wandering.

### Interpretation

**The step count is a privacy parameter.** Every step spends budget, so the noise per step grows like $\sqrt T$ and the noise floor like $T$. Too few steps leave optimization error, too many drown the solution in noise. @fig-dperm-noisy-gd-steps shows the trade-off on the running example. And you cannot pick the best $T$ by trying several and keeping the one with the best validation score without paying for it: that selection is itself a computation on the data, the subject of [Hyperparameter tuning](03-hyperparameter-tuning.md).

**A public rule for $T$.** @eq-dperm-gd-risk can be evaluated without looking at the data, using only public quantities: $\lambda$, the curvature bound $\beta = \frac14 + \lambda$ for logistic loss with unit features, $J(w_0) = \ln 2$ for $w_0 = 0$ (each loss is $\ln 2$ at $w = 0$, and $J(w^*) \ge 0$), $d$, $n$ and the budget. The rule picks the $T$ that minimizes it. It is pessimistic, because it only knows that the curvature is at least $\lambda$.

**Why it is the one that generalizes.** Gradient perturbation needs a gradient bound, not a minimizer. For a neural network no such bound holds, because gradients can be arbitrarily long, but you can force one by **clipping**: rescale each record's gradient to length at most $C$ before averaging, exactly as [Beyond global sensitivity](../part03-dp-foundations/06-sensitivity.md#sec-sens-clip) clips a value before averaging it. That, with minibatches instead of full batches, is DP-SGD.

![Excess risk of noisy gradient descent at $n = 10{,}000$, $(\varepsilon, \delta) = (1, 10^{-5})$, against the number of steps $T$, averaged over datasets, with the same steps without noise (dashed). The noiseless error falls geometrically and reaches machine precision; the noisy run is limited by noise that grows with $T$, so its error is smallest near $T = 50$.](figures/erm_noisy_gd_steps.svg){#fig-dperm-noisy-gd-steps}

### Example: choosing $T$ for the running example

For $n = 10{,}000$, the public rule picks $T = 218$, which gives per-step noise $\sigma = 0.01102$ on the mean gradient and an estimated excess risk of $1.301 \times 10^{-3}$. Simulated runs average $1.384 \times 10^{-3}$, close to the estimate.

@fig-dperm-noisy-gd-steps shows that the rule overshoots. With the actual curvature of the data, between 0.0263 and 0.0330 rather than the 0.01 the rule assumes, noiseless gradient descent is already within $2.679 \times 10^{-11}$ of the minimum after 100 steps. Measured over 10 datasets, the excess risk is $1.027 \times 10^{-3}$ at $T = 20$, $2.875 \times 10^{-4}$ at $T = 50$, $7.137 \times 10^{-4}$ at $T = 100$ and $6.018 \times 10^{-3}$ at $T = 1{,}000$. The best $T$ is near 50, four times smaller than the rule's, and gives five times less excess risk. You may only use that knowledge if you obtained it privately, or from public data that resembles yours.

| $n$ | Steps $T$ (public rule) | Per-step $\sigma$ on the mean gradient | Estimated excess risk |
| --- | --- | --- | --- |
| 1,000 | 100 | 0.07461 | $6.725 \times 10^{-2}$ |
| 3,000 | 156 | 0.03106 | $1.080 \times 10^{-2}$ |
| 10,000 | 218 | 0.01102 | $1.301 \times 10^{-3}$ |
| 30,000 | 274 | 0.004117 | $1.779 \times 10^{-4}$ |
| 100,000 | 335 | 0.001366 | $1.930 \times 10^{-5}$ |

: Noisy gradient descent on the running example: the step count chosen by minimizing @eq-dperm-gd-risk with public quantities only, the resulting noise, and the estimated excess risk. {#tbl-dperm-gd}

The step count grows slowly with $n$, like $\ln n$, because the optimization error shrinks geometrically while the target, the noise floor, shrinks like $1/n^2$ (@tbl-dperm-gd).

### Check yourself

- Noisy gradient descent with $T = 100$ steps is $(1, 10^{-5})$-DP. You run it for $T = 400$ steps with the same per-step noise. What is $\mu$ now? *(Twice as large: $\mu = \sqrt T\,\mu_{\text{step}}$, and $\sqrt{400/100} = 2$. So $\mu = 2 \times 0.2681 = 0.5362$, a weaker guarantee.)*
- In step 1 of the derivation, why is it allowed that the point $w_t$ at which the gradient is evaluated depends on the data? *(Because $w_t$ is computed from earlier released outputs only. Adaptive composition covers mechanisms chosen based on earlier outputs, so the bound holds conditionally on the past, at whatever $w_t$ it produced.)*

## How the three compare {#sec-dperm-compare}

### Context

Each method is private by its own argument. Which should you use? Theory answers by rates, how the excess risk falls with $n$, $d$ and $\varepsilon$, and by lower bounds that no method can beat. Simulation answers for the running example. They agree on the rate and disagree on the constant, and the constant is what you choose by.

### Statement

For $\lambda$-strongly convex objectives with $L$-Lipschitz losses, Bassily, Smith & Thakurta (2014) show that the best achievable expected excess risk under $(\varepsilon, \delta)$-DP is
$$\Theta\Big(\frac{L^2\,d\,\ln(1/\delta)}{\lambda\,n^2\varepsilon^2}\Big)$$ {#eq-dperm-rate-strong}
up to logarithmic factors in $n$, and that a version of noisy gradient descent attains it. For convex objectives over a ball of radius $R$, without strong convexity, the rate is $\Theta\big(LR\sqrt{d\ln(1/\delta)}/(n\varepsilon)\big)$ up to logarithms. The lower bounds come from the fingerprinting codes of [Lower bounds](../part03-dp-foundations/09-lower-bounds.md): a private algorithm whose output is too accurate would let an adversary trace which records were used.

On the running example, @tbl-dperm-compare gives the measured excess risk of the four private releases.

| $n$ | Output, Gaussian | Output, pure | Objective, pure | Noisy gradient descent (public $T$) |
| --- | --- | --- | --- | --- |
| 1,000 | $8.170 \times 10^{-2}$ | $7.060 \times 10^{-2}$ | $1.009 \times 10^{-2}$ | $6.558 \times 10^{-2}$ |
| 3,000 | $1.211 \times 10^{-2}$ | $7.206 \times 10^{-3}$ | $6.303 \times 10^{-4}$ | $8.888 \times 10^{-3}$ |
| 10,000 | $9.274 \times 10^{-4}$ | $8.321 \times 10^{-4}$ | $8.156 \times 10^{-5}$ | $1.384 \times 10^{-3}$ |
| 30,000 | $9.973 \times 10^{-5}$ | $8.943 \times 10^{-5}$ | $7.135 \times 10^{-6}$ | $1.538 \times 10^{-4}$ |
| 100,000 | $8.543 \times 10^{-6}$ | $6.379 \times 10^{-6}$ | $8.798 \times 10^{-7}$ | $1.621 \times 10^{-5}$ |
| Slope in $\log n$ | $-2.01$ | $-2.00$ | $-2.01$ | $-1.79$ |

: Mean excess risk $J(\tilde w) - J(w^*)$ over 20 datasets of the running example ($d = 10$, $\lambda = 0.01$, $\varepsilon = 1$, $\delta = 10^{-5}$ for the Gaussian methods), and the slope of a straight-line fit of $\log$ excess risk against $\log n$. {#tbl-dperm-compare}

### Derivation

The upper bounds for output and objective perturbation are @eq-dperm-output-risk and @eq-dperm-objective-risk with the noise of each method substituted. Both have $\mathbb E\|\text{noise}\|^2 \propto 1/n^2$ (the output noise has scale $1/(\lambda n)$; the objective noise has scale 1 and is divided by $n$), which gives the $1/n^2$. For noisy gradient descent, the noise floor $d\sigma^2/(4\beta)$ has $\sigma^2 \propto T/n^2$, and the optimization error must be pushed below it, which takes $T \propto \frac\beta\lambda\ln n$ steps; this gives $1/n^2$ times a logarithm. The proofs of the optimal rate and of the lower bound are long; Bassily, Smith & Thakurta (2014) give both.

### Interpretation

**The rate is the same; the constants are not.** All three methods fall like $1/n^2$, and noisy gradient descent a little slower, $-1.79$, because its step count grows with $\ln n$. On this problem objective perturbation is about ten times better than output perturbation at every $n$, the $(h/\lambda)^2$ of @sec-dperm-objective, and noisy gradient descent with the public step rule is the worst of the three for $n \ge 10{,}000$. Tuned privately to $T \approx 50$, it would sit between the two perturbation methods.

**Pure against approximate DP.** At $d = 10$ the pure-DP noise of Fact 3 and the Gaussian are close: their mean squared lengths are $d(d+1)s^2 = 110\,\Delta^2$ at $\varepsilon = 1$ and $d\sigma^2 = 10 \times 3.7306^2\,\Delta^2 = 139.2\,\Delta^2$, which is why the pure-DP column is slightly better here. At $d = 100$ they are $10{,}100\,\Delta^2$ against $1{,}392\,\Delta^2$: pure DP costs a factor that grows like $d$, the price of $\delta = 0$ that [Lower bounds](../part03-dp-foundations/09-lower-bounds.md) proves unavoidable.

**Accuracy is forgiving.** Excess risk is a fine measure, but you care about predictions. On held-out data the non-private model has accuracy 0.747 at $n = 1{,}000$; output perturbation drops it to 0.681, noisy gradient descent to 0.695 and objective perturbation only to 0.736. By $n = 10{,}000$ all four are within 0.002 of the non-private 0.753. A low-dimensional convex model on ten thousand records can be made private almost for free.

![Excess risk of the four private releases against the dataset size on the running example, log-log, with a line of slope $-2$. All fall like $1/n^2$; objective perturbation is about ten times better than output perturbation throughout.](figures/erm_comparison.svg){#fig-dperm-comparison}

### Example: reading the rate

@eq-dperm-rate-strong predicts how much data you need. Suppose an excess risk of $10^{-4}$ is acceptable, and output perturbation gives $9.274 \times 10^{-4}$ at $n = 10{,}000$. The rate $1/n^2$ says that multiplying $n$ by $\sqrt{9.274} = 3.05$ brings it to about $10^{-4}$; @tbl-dperm-compare confirms it, with $9.973 \times 10^{-5}$ at $n = 30{,}000$. Equivalently, halving $\varepsilon$ costs the same as halving $n$, because $n$ and $\varepsilon$ appear only as the product $n\varepsilon$: to keep the same accuracy at half the budget, you need twice the data.

### Check yourself

- In @eq-dperm-rate-strong, why do $n$ and $\varepsilon$ appear only as the product $n\varepsilon$? *(Every sensitivity in this chapter is proportional to $1/n$ and every noise to sensitivity over $\varepsilon$, so the noise scale is proportional to $1/(n\varepsilon)$.)*
- Objective perturbation beat output perturbation by a factor of ten. Name a problem on which it would not help at all. *(Any problem whose data add no curvature beyond the ridge, such as a linear loss: then $H = \lambda I$, $h = \lambda$, and the ratio $(h/\lambda)^2$ is 1.)*

## When the model is not convex {#sec-dperm-nonconvex}

### Context

A neural network's loss is not convex in its weights. What survives of the three methods?

### Statement

Output perturbation and objective perturbation have no general privacy guarantee for non-convex objectives: both need a unique minimizer that depends continuously on the data. Noisy gradient descent keeps its guarantee, @eq-dperm-noisy-gd, for any loss whose per-record gradients have length at most $L$, and loses only its accuracy guarantee. When no such $L$ exists, clipping each per-record gradient to length $C$ makes $L = C$.

### Derivation

1. **The minimizer can jump.** [Optimization](../part01-prerequisites/06-optimization.md) gives a one-dimensional loss with two wells, $\frac14(w^2 - 1)^2 - yw$ with labels $y = \pm1$, whose empirical minimizer sits in the well favoured by the majority label. With 501 labels $+1$ and 500 labels $-1$, changing one label flips the majority and moves the minimizer by 2.0010. No $1/n$ appears: the sensitivity of the minimizer is a constant, so output perturbation would need noise with $\sigma = 3.7306 \times 2.0010 = 7.465$ at $\varepsilon = 1$, more than three times the distance between the wells, whatever the dataset size.
2. **The objective-perturbation proof breaks at step 1.** Without strong convexity the perturbed objective can have several minimizers, the map from noise to output is no longer invertible, and Fact 4 cannot be applied.
3. **Gradient perturbation survives.** Its derivation in @sec-dperm-gradient used only the gradient bound at the visited points (step 3 there).
4. **Clipping supplies the bound.** Replace each per-record gradient $\nabla\ell(w, z_i)$ by $\nabla\ell(w, z_i)\min(1, C/\|\nabla\ell(w, z_i)\|)$, which has length at most $C$ and is unchanged when it is already shorter. The average of clipped gradients changes by at most $2C/n$ when one record is replaced, and the derivation goes through with $L = C$.

### Interpretation

What the non-convex case loses is the guarantee of accuracy, not of privacy. Noisy gradient descent on a neural network is private whatever the network does; whether it learns anything is an empirical question, and the answer depends on the clipping norm, the noise and the number of steps, which the [next chapter](02-dp-sgd.md) studies. Clipping also changes what is being optimized: the average of clipped gradients is not the gradient of $J$, so the algorithm no longer minimizes $J$, a bias that the next chapter measures.

Full-batch gradient descent is also too slow for large datasets: each step touches every record. Using a random minibatch at each step makes each step cheaper and, by [Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md), more private per step, because each record is used only with probability $q$. Clipping, noise, minibatches and a subsampled-Gaussian accountant together are DP-SGD.

### Example: two wells and output perturbation

Take the double-well loss with $n = 1{,}001$, whose minimizer sits in one of the two wells near $\pm1$. Output perturbation with $\sigma = 7.465$ releases a weight whose standard deviation is more than seven times the distance from either well to the origin; the released value carries almost no information about which well the data chose, which is exactly what privacy demands, since one record decides it. Noisy gradient descent with clipping still moves toward whichever well the noisy average gradient points to; on datasets where many records agree, the average gradient is large compared with the noise and the method finds the majority well with high probability.

### Check yourself

- Why does clipping not break the privacy proof, even though it changes the gradients? *(The proof needs only that each step releases a function of the data with bounded sensitivity plus Gaussian noise. The clipped average is such a function, with sensitivity $2C/n$, whatever it is the gradient of.)*
- Does noisy gradient descent on a non-convex loss need to find the global minimum to be private? *(No. Privacy holds for every run, whatever point it reaches; finding a good point is a question of accuracy only.)*

## Code: check every number by computation

The script [code/dp_erm.py](../../code/part05-dp-ml/dp_erm.py) implements the three methods for regularized logistic regression with exact Newton solves, computes the exact output densities of objective perturbation in one dimension, and runs the comparison. From the repository root:

```bash
uv run python code/part05-dp-ml/dp_erm.py
```

Its public functions are used by later chapters of this part: `make_logistic_data`, `fit_logistic`, `logistic_objective`, `logistic_grad`, `excess_risk`, `gaussian_delta(eps, mu)`, `gaussian_mu(eps, delta)` and `gaussian_sigma(sensitivity, eps, delta)`.

**Output** (seed 0), summarized in @tbl-dperm-code:

| Experiment | Result |
| --- | --- |
| Gaussian calibration, $\varepsilon = 1$, $\delta = 10^{-5}$ | $\mu = 0.2681$; exact $\sigma = 3.7306\,\Delta$ against classic $4.8448\,\Delta$ |
| Output perturbation, one dimension | Minimizers 0.0832 apart, bound 0.2 |
| Output perturbation, $n = 10{,}000$ | Predicted excess risk $8.711 \times 10^{-4}$, measured $9.274 \times 10^{-4}$ |
| Approximate minimizer, $g = 10^{-3}$ | Sensitivity 0.22 against 0.02 exact |
| Objective perturbation, one dimension, 500 datasets | Largest log ratio 0.3779 against $\varepsilon = 1$; noise part 0.0992 of 0.1891, Jacobian part 0.3395 of 0.8109 |
| Objective perturbation, $n = 10{,}000$ | Predicted $7.127 \times 10^{-5}$, measured $8.156 \times 10^{-5}$ |
| Noisy gradient descent, $n = 10{,}000$ | Public rule $T = 218$, measured $1.384 \times 10^{-3}$; best $T \approx 50$ gives $2.875 \times 10^{-4}$ |
| Slopes in $\log n$ | $-2.01$, $-2.00$, $-2.01$, $-1.79$ |

: The script's output with the default seed. {#tbl-dperm-code}

[tests/test_dp_erm.py](../../tests/test_dp_erm.py) checks that the output still matches.

**What to take from it.** Three things. First, the privacy proofs can be checked numerically where the output distribution is computable: in one dimension the change-of-variables density is exact, and the log ratio stays inside its bound with room to spare. Second, the second-order predictions of excess risk match simulation to within the sampling error, so you can predict the cost of privacy before training. Third, all methods have the same $1/n^2$ rate on this problem, and the constant, a factor of ten here, comes from how each method uses the data's curvature.

## References

- Chaudhuri & Monteleoni, "Privacy-preserving logistic regression" (NeurIPS 2008). Output and objective perturbation for logistic regression, the first private ERM algorithms.
- Chaudhuri, Monteleoni & Sarwate, "Differentially Private Empirical Risk Minimization" (JMLR 2011). The general analysis of both: output perturbation (Algorithm 1), objective perturbation (Algorithm 2) and their excess-risk bounds. The reference for @sec-dperm-output and @sec-dperm-objective.
- Kifer, Smith & Thakurta, "Private Convex Empirical Risk Minimization and High-dimensional Regression" (COLT 2012). Objective perturbation with Gaussian noise, constraints and non-smooth losses.
- Song, Chaudhuri & Sarwate, "Stochastic gradient descent with differentially private updates" (IEEE GlobalSIP 2013). Gradient perturbation for SGD, with minibatches.
- Bassily, Smith & Thakurta, "Private Empirical Risk Minimization: Efficient Algorithms and Tight Error Bounds" (FOCS 2014). Noisy gradient methods that attain the optimal rates, and the matching lower bounds; the reference for @sec-dperm-compare.
- Wu, Li, Kumar, Chaudhuri, Jha & Naughton, "Bolt-on Differential Privacy for Scalable Stochastic Gradient Descent-based Analytics" (SIGMOD 2017). Output perturbation for SGD run for a fixed number of passes, without an exact minimizer.
- Iyengar, Near, Song, Thakkar, Thakurta & Wang, "Towards Practical Differentially Private Convex Optimization" (IEEE S&P 2019). Approximate minima perturbation, an objective perturbation that tolerates inexact solvers, and a benchmark of all three methods.
- Balle & Wang, "Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising" (ICML 2018). The exact calibration of Fact 1.
- Dong, Roth & Su, "Gaussian Differential Privacy" (JRSS-B 2022). The composition of Fact 5.

## Cheat sheet

$n$ records, ridge $\lambda$, per-record gradients of length at most $L$, replace-one neighbours, $\mu(\varepsilon, \delta)$ the largest admissible Gaussian ratio (0.2681 at $\varepsilon = 1$, $\delta = 10^{-5}$), $H$ the Hessian at the minimizer. @tbl-dperm-cheat collects the results.

| Result | Formula | Needs | Guarantee |
| --- | --- | --- | --- |
| Sensitivity of the minimizer | $2L/(\lambda n)$ | $\lambda$-strong convexity, exact minimizer | — |
| Approximate minimizer | $2L/(\lambda n) + 2g/\lambda$ | $\|\nabla J\| \le g$ checked at stopping | — |
| Output perturbation | $w^* + \mathcal N(0, \sigma^2I)$, $\sigma = 2L/(\lambda n\mu)$ | As above | $(\varepsilon, \delta)$-DP |
| Output, pure | $w^* + b$, $\nu(b) \propto e^{-\|b\|\lambda n\varepsilon/(2L)}$ | As above | $\varepsilon$-DP |
| Excess risk, output | $\approx\mathbb E\|Z\|^2\operatorname{tr}H/(2d)$ | Second order | — |
| Objective perturbation | $\arg\min J + \langle b, w\rangle/n$, $\nu(b) \propto e^{-\varepsilon'\|b\|/2}$, $\varepsilon' = \varepsilon - 2\ln(1 + c/(n\lambda))$ | Loss $g(y\langle w, x\rangle)$, $|g'| \le 1$, $0 \le g'' \le c$, $\|x\| \le 1$, exact minimizer | $\varepsilon$-DP |
| Excess risk, objective | $\approx\mathbb E\|b\|^2\operatorname{tr}H^{-1}/(2dn^2)$ | Second order | — |
| Noisy gradient descent | $\sigma = (2L/n)\sqrt T/\mu$ on the mean gradient | Per-record gradients $\le L$ (or clipped) | $(\varepsilon, \delta)$-DP, all iterates |
| Excess risk, noisy gradient descent | $\approx(1 - \lambda/\beta)^T(J(w_0) - J^*) + d\sigma^2/(4\beta)$ | Strong convexity, smoothness, $\eta = 1/\beta$ | — |
| Optimal rate | $\tilde\Theta(L^2d\ln(1/\delta)/(\lambda n^2\varepsilon^2))$ | Strongly convex | Bassily, Smith & Thakurta (2014) |

: Every result of the chapter. {#tbl-dperm-cheat}

**Rules of thumb**

- All three methods give excess risk falling like $1/(n\varepsilon)^2$ on strongly convex problems. Halving $\varepsilon$ costs as much as halving the data.
- Output perturbation pays for the ridge's curvature, objective perturbation benefits from the data's. If the data add curvature $h$ beyond $\lambda$, expect objective perturbation to win by about $(h/\lambda)^2$.
- Output and objective perturbation need the exact minimizer; stop on a gradient tolerance well below $L/n$, never on a step count.
- The number of steps of noisy gradient descent is part of the privacy budget. Choose it from public quantities, or pay for choosing it.
- Only gradient perturbation is private without convexity. Clip per-record gradients to get the bound it needs.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part05-dp-ml/01-dp-erm.md).
