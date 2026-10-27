# Private empirical risk minimization

Most of machine learning is **empirical risk minimization (ERM)**. You choose the parameters $\theta$ that minimize the average loss on the training data, usually plus a penalty on the size of $\theta$. The trained weights are a function of every training record, and [Membership inference](../part02-attacks/02-membership-inference.md) showed that they give some of those records away. An exact minimizer can even release a record outright: the median of an odd number of values minimizes the average absolute loss, and it *is* one of the values. Differential privacy (DP) promises the opposite: the distribution of whatever you release changes by at most a factor $e^\varepsilon$, plus a slack $\delta$, when one record is replaced. So randomness has to enter somewhere between the data and the released weights, and this chapter asks where.

The idea that organizes the answer is the one behind every mechanism of this book. Find a quantity that one record can move only by a known amount, its **sensitivity**, and add noise large enough to hide a move of that size. There are three natural places for the noise. **Output perturbation** trains as usual and adds noise to the minimizer, which a ridge penalty keeps from moving far. **Objective perturbation** adds a random linear term to the objective and minimizes it exactly, which needs a bound on how far one record moves the objective's gradient. **Gradient perturbation** adds noise to every gradient during training, so it needs that bound only at the points it visits. The earlier in training the noise enters, the fewer assumptions it needs, and the third needs no convexity at all, which is why the [next chapter](02-dp-sgd.md) builds on it.
The running example is regularized logistic regression in $d = 10$ dimensions, with a ridge of weight 0.01. The features are uniform on the unit sphere, and the labels come from a logistic model whose true weights have length 5. Every method targets $\varepsilon = 1$, with $\delta = 10^{-5}$ for those that need a $\delta$, on 1,000 to 100,000 records. At 10,000 records, output perturbation with Gaussian noise raises the training loss above the exact minimizer's by $9.274 \times 10^{-4}$, and objective perturbation by ten times less.

## Output perturbation {#sec-dperm-output}

The simplest private trainer computes the exact minimizer and adds noise to it, as Chaudhuri & Monteleoni (2008) did for logistic regression. This section asks how much noise hides one record's effect on the minimizer, what the noise costs in accuracy, and what changes when the minimizer is only approximate.

A training set $D$ holds $n$ records. Each record $z_i = (x_i, y_i)$ pairs a feature vector $x_i \in \mathbb R^d$ with a label $y_i \in \{-1, +1\}$, and $\ell(\theta; z)$ is the loss of the parameters on one record. The objective adds a ridge penalty of weight $\lambda$ to the average loss,
$$\mathcal L(\theta; D) = \frac1n\sum_{i=1}^n\ell(\theta; z_i) + \frac\lambda2\|\theta\|^2,$$
and its minimizer is $\theta^*(D)$.

Neighbouring datasets $D \sim D'$ differ by **replacing** one record, $z$ in $D$ by $z'$ in $D'$, so both have size $n$. The **excess risk** of a released $\tilde\theta$ is $\mathcal L(\tilde\theta; D) - \mathcal L(\theta^*; D)$, how much worse it fits the objective than the exact minimizer.

Two earlier results do most of the work. First, if each loss is convex with gradients of length at most $L$, replacing one record moves the minimizer by at most $2L/(\lambda n)$ (@thm-opt-erm-sensitivity in [Optimization](../part01-prerequisites/06-optimization.md)). That is because the objective's gradient moves by at most $2L/n$, and the ridge's curvature $\lambda$ turns a small gradient into a small distance.

The second result says how much Gaussian noise hides a move of $\Delta_2 f$, the most that one record can move a function $f$ in Euclidean length. With normal noise of standard deviation $\sigma$ per coordinate, only the ratio $\mu = \Delta_2 f/\sigma$ matters, because a rotation puts the move along one axis without changing the noise. The release is $(\varepsilon, \delta)$-DP exactly when
$$\delta \ge \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^\varepsilon\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big),$$ {#eq-dperm-gauss-delta}
where $\Phi$ is the standard normal distribution function (@thm-lg-analytic). The right side grows with $\mu$, so each budget has a largest admissible ratio, written $\mu(\varepsilon, \delta)$, and the least noise is $\sigma = \Delta_2 f/\mu(\varepsilon, \delta)$ (@cor-lg-calibration). For $\varepsilon = 1$ and $\delta = 10^{-5}$, $\mu = 0.2681$, so $\sigma = 3.7306\,\Delta_2 f$, while the classic formula $\Delta_2 f\sqrt{2\ln(1.25/\delta)}/\varepsilon$ of @thm-lg-classic gives 30% more, $4.8448\,\Delta_2 f$.

Gaussian noise can never give pure $\varepsilon$-DP, with $\delta = 0$, because the log ratio of two shifted Gaussian densities grows without bound in the tails. Pure DP needs a density that falls off only exponentially in every direction, and the theorem below offers this noise as its pure option.

::: {#lem-dperm-norm-noise name="Noise with density proportional to $e^{-\|b\|/s}$"}
Let a noise vector $b \in \mathbb R^d$ have density $\nu(b) \propto e^{-\|b\|/s}$, with scale $s > 0$. If $f$ has $\ell_2$ sensitivity $\Delta_2 f$ and $s = \Delta_2 f/\varepsilon$, then $f(D) + b$ is $\varepsilon$-DP. The length $\|b\|$ has the gamma distribution with shape $d$ and scale $s$, the direction is uniform and independent of the length, and
$$\mathbb E\|b\| = ds, \qquad \mathbb E\|b\|^2 = d(d + 1)s^2.$$
:::

::: {.proof}
1. *Privacy.* At an output $y$ the two densities share their constant, so their ratio is $e^{(\|y - f(D')\| - \|y - f(D)\|)/s}$. By the triangle inequality the difference of distances is at most $\|f(D) - f(D')\| \le \Delta_2 f$. So the ratio is at most $e^{\Delta_2 f/s} = e^\varepsilon$ at every output, which is pure $\varepsilon$-DP by the pointwise criterion of [the definition of differential privacy](../part03-dp-foundations/01-definition.md).
2. *Length and direction.* The density depends on $b$ only through its length $r$. The sphere of radius $r$ has surface area proportional to $r^{d-1}$, so the length has density proportional to
$$r^{d - 1}e^{-r/s},$$
the gamma density with shape $d$ and scale $s$. At a given length every direction has the same density, so the direction is uniform and independent of the length.
3. *Moments.* A gamma variable with shape $d$ and scale $s$ has mean $ds$ and variance $ds^2$. The second moment is the variance plus the squared mean, $ds^2 + d^2s^2 = d(d + 1)s^2$.
:::

A Gaussian vector's squared length has mean $d\sigma^2$, against $d(d + 1)s^2$ here, so in high dimension the pure-DP noise is longer by a factor that grows with $d$.

::: {#thm-dperm-output name="Output perturbation"}
Let each loss $\ell(\cdot; z)$ be convex and differentiable, with gradients of length at most $L$, let $\lambda > 0$, and let neighbours replace one record. Release
$$\tilde\theta = \theta^*(D) + Z, \qquad Z \sim \mathcal N\big(0, \sigma^2I\big), \qquad \sigma = \frac{2L}{\lambda n\,\mu(\varepsilon, \delta)}.$$ {#eq-dperm-output}
This release is $(\varepsilon, \delta)$-DP. Replacing $Z$ by noise $b$ with density $\propto e^{-\|b\|/s}$ and $s = 2L/(\lambda n\varepsilon)$ gives pure $\varepsilon$-DP.
:::

::: {.proof}
1. *Sensitivity.* By @thm-opt-erm-sensitivity the minimizer moves by at most $\Delta_2\theta^* = 2L/(\lambda n)$ when one record is replaced.
2. *Gaussian noise.* With $\sigma = \Delta_2\theta^*/\mu(\varepsilon, \delta)$ the ratio of sensitivity to noise is $\mu(\varepsilon, \delta)$, which satisfies @eq-dperm-gauss-delta, so the release is $(\varepsilon, \delta)$-DP.
3. *Pure noise.* With $s = \Delta_2\theta^*/\varepsilon$, @lem-dperm-norm-noise makes the release $\varepsilon$-DP.
:::

How much accuracy does the noise cost? @sec-apx-learn-erm in the appendices answers exactly, and a second-order answer is simpler. Near the minimizer the objective is a bowl, whose shape is given by the **Hessian** $H = \nabla^2\mathcal L(\theta^*)$, the matrix of second derivatives. A small random step costs half its squared length times the bowl's curvature averaged over the step's direction.

::: {#prp-dperm-output-risk name="Excess risk of output perturbation"}
Let $\mathcal L$ be twice differentiable with Hessian $H$ at $\theta^*$, and let the noise $Z$ have a uniformly random direction, independent of its length. To second order in $Z$,
$$\mathbb E\big[\mathcal L(\tilde\theta) - \mathcal L(\theta^*)\big] \approx \frac12\,\mathbb E\big[Z^\top HZ\big] = \frac{\mathbb E\|Z\|^2}{2d}\operatorname{tr}H,$$ {#eq-dperm-output-risk}
which is $\frac{\sigma^2}{2}\operatorname{tr}H$ for the Gaussian.
:::

::: {.proof}
1. *Expand around the minimizer.* The gradient vanishes at $\theta^*$, so a second-order Taylor expansion leaves only the quadratic term,
$$\mathcal L(\theta^* + Z) - \mathcal L(\theta^*) \approx \tfrac12Z^\top HZ.$$
2. *Average over the direction.* Write $Z = \|Z\|u$ with $u$ a unit vector. The $d$ squared coordinates of $u$ add up to 1 and are exchangeable, so each averages $1/d$. Flipping the sign of one coordinate leaves a uniform direction uniform, so the product of two different coordinates averages to zero. Hence
$$\mathbb E[u^\top Hu] = \frac1d\sum_iH_{ii} = \frac{\operatorname{tr}H}{d}.$$
3. *Put the length back.* The length is independent of the direction, so $\mathbb E[Z^\top HZ] = \mathbb E\|Z\|^2\operatorname{tr}H/d$. For the Gaussian $\mathbb E\|Z\|^2 = d\sigma^2$.
:::

Every solver stops at an approximate minimizer, so the theorem's exact $\theta^*$ never exists in practice. What a solver can guarantee is that the gradient's length is below a tolerance $\tau$, and the next result adds that slack to the sensitivity.

::: {#prp-dperm-approx name="Sensitivity of an approximate minimizer"}
In the setting of @thm-dperm-output, let the optimizer return, on every dataset, a point $\theta$ with $\|\nabla\mathcal L(\theta; D)\| \le \tau$. Then its outputs $\theta$ and $\theta'$ on two neighbouring datasets satisfy
$$\|\theta - \theta'\| \le \frac{2L}{\lambda n} + \frac{2\tau}{\lambda}.$$ {#eq-dperm-approx}
:::

::: {.proof}
1. *Each output is near its minimizer.* The objective is $\lambda$-strongly convex, so part (a) of @cor-opt-strong puts $\theta$ within $\tau/\lambda$ of $\theta^*(D)$, and $\theta'$ within $\tau/\lambda$ of $\theta^*(D')$.
2. *The minimizers are near each other.* By @thm-opt-erm-sensitivity they are at most $2L/(\lambda n)$ apart.
3. *Add the three distances.* The triangle inequality along the path $\theta \to \theta^*(D) \to \theta^*(D') \to \theta'$ gives @eq-dperm-approx.
:::

The noise is calibrated to the worst case, and the damage is measured on the actual case. The sensitivity uses only the ridge's curvature $\lambda$, while the damage is weighted by $H$, which includes the data's own. If every eigenvalue of $H$ is close to some $h$, the Gaussian excess risk is about $\frac{2dL^2}{\mu^2n^2}\cdot\frac{h}{\lambda^2}$, so a small ridge buys a better non-private fit and costs a lot of noise. The excess risk falls like $1/n^2$, faster than the error of order $1/n$ that sampling causes even without privacy, so for large $n$ privacy costs less than sampling does.

The method also needs a nearly exact solution. To keep the extra term of @eq-dperm-approx below the exact one, the optimizer must reach $\tau \le L/n$, a tolerance that shrinks with the dataset. The stopping rule must be a gradient check, not a step count, because a fixed number of steps may leave a large gradient on some dataset, and the proof needs the bound on every one. Wu et al. (2017) bound the sensitivity of stochastic gradient descent (SGD) run for a fixed number of passes instead.

![Left: two neighbouring regularized logistic objectives in one dimension ($n = 20$, $\lambda = 0.5$), their minimizers (vertical lines) 0.083 apart, below the bound $2L/(\lambda n) = 0.2$. Right: the released weight's density under each dataset with Gaussian noise for $\varepsilon = 1$, $\delta = 10^{-5}$; the noise must hide a shift as large as the bound, not the actual shift.](figures/erm_output_1d.svg){#fig-dperm-output-1d}

::: {#exm-dperm-output}
Start with one feature, $n = 20$ records and $\lambda = 0.5$ (@fig-dperm-output-1d). Replacing the last record, which has $x = 1$, by the same feature with the opposite label moves the minimizer from 0.2128 to 0.1296, by 0.0832. With $L = 1$, since a logistic gradient is never longer than its unit feature, the bound is $2/(0.5 \times 20) = 0.2$, and the noise is calibrated to the bound, $\sigma = 3.7306 \times 0.2 = 0.7461$, so the two output densities overlap almost completely.

In the running example ($d = 10$, $\lambda = 0.01$, $L = 1$), $n = 10{,}000$ gives sensitivity $2/(0.01 \times 10{,}000) = 0.02$ and $\sigma = 0.07461$ per coordinate, next to a minimizer of length 2.886. The Hessian at $\theta^*$ has eigenvalues between 0.0263 and 0.0330, and trace 0.3129. @eq-dperm-output-risk predicts $\frac12 \times 0.07461^2 \times 0.3129 = 8.711 \times 10^{-4}$, and 20 simulated releases average $9.274 \times 10^{-4}$. The pure-DP version, with $s = 0.02$, predicts $6.885 \times 10^{-4}$ and averages $8.321 \times 10^{-4}$. The gap is wider because the squared length of gamma noise varies a lot, and 20 draws average it only roughly.

Stopping early is costly. A gradient tolerance of $\tau = 10^{-6}$ gives sensitivity 0.0202, essentially exact, and $\tau = 10^{-4}$ doubles it to 0.04. At $\tau = 10^{-3}$ it is 0.22, eleven times the exact value, which multiplies the excess risk by about 121.
:::

## Objective perturbation {#sec-dperm-objective}

Output perturbation wastes the data's curvature, because it pays for the worst case $\lambda$ and is judged on the actual $H$. Chaudhuri, Monteleoni & Sarwate (2011) put the noise inside the optimization instead. This section asks whether a randomly tilted objective, minimized exactly, is private, and what it costs.

The tilt is a random linear term $\frac1n\langle b, \theta\rangle$. It acts as a constant force, and all of the objective's curvature resists it, so where the data add curvature the minimizer moves less. The output is no longer something fixed plus noise, so the proof is subtler, and it needs each record's Hessian to have rank one. So the loss must depend on $\theta$ through one inner product, $\ell(\theta; (x, y)) = \psi(y\langle \theta, x\rangle)$, for a scalar function $\psi$ of the **margin** $m = y\langle\theta, x\rangle$. Logistic regression has $\psi(m) = \ln(1 + e^{-m})$, whose derivative lies in $(-1, 0)$ and whose second derivative is at most $c = \frac14$. The budget $\varepsilon$ is split in two. One part pays for how differently the two datasets stretch the noise into an output, which a bound $c$ on $\psi''$ controls, and the rest, written $\varepsilon'$, goes to the noise.

::: {#thm-dperm-objective name="Objective perturbation"}
Let the loss be $\ell(\theta; (x, y)) = \psi(y\langle \theta, x\rangle)$, with $\psi$ convex and twice differentiable, $|\psi'| \le 1$ and $0 \le \psi'' \le c$, and let every feature have $\|x\| \le 1$. Let $\lambda > 0$ and let neighbours replace one record. Set
$$\varepsilon' = \varepsilon - 2\ln\Big(1 + \frac{c}{n\lambda}\Big),$$ {#eq-dperm-eps-prime}
draw $b$ with density $\nu(b) \propto e^{-\varepsilon'\|b\|/2}$, and release
$$\tilde\theta = \arg\min_\theta\Big[\mathcal L(\theta; D) + \frac1n\langle b, \theta\rangle\Big].$$ {#eq-dperm-objective}
If $\varepsilon' > 0$, the release is $\varepsilon$-DP (Chaudhuri, Monteleoni & Sarwate 2011, Algorithm 2). If $\varepsilon' \le 0$, adding $\lambda_+ = \frac{c}{n(e^{\varepsilon/4} - 1)} - \lambda$ to the ridge and drawing the noise with $\varepsilon/2$ in place of $\varepsilon'$ makes it $\varepsilon$-DP.
:::

::: {.proof}
The plan is to write the density of $\tilde\theta$ explicitly, then bound the ratio of the two densities, for $D$ and $D'$, at every point. The ratio splits into a noise factor and a stretching factor, and each consumes part of $\varepsilon$.

1. *From the output back to the noise.* The perturbed objective is $\lambda$-strongly convex, so it has exactly one minimizer, where its gradient is zero: $\nabla\mathcal L(\tilde\theta; D) + b/n = 0$. Solving for $b$ gives
$$b = \phi_D(\tilde\theta) = -n\nabla\mathcal L(\tilde\theta; D) = -\sum_{i=1}^n\nabla\ell(\tilde\theta; z_i) - n\lambda\tilde\theta.$$
Every $\theta$ is the output for exactly one noise vector, $\phi_D(\theta)$, so the map from noise to output is invertible and $\phi_D$ is its inverse.
2. *The density of the output.* When a random vector with density $\nu$ is mapped smoothly and invertibly, the image has density $\nu$ at the preimage times the absolute determinant of the inverse map's Jacobian. The determinant corrects for stretching: a small box around $\theta$ and its preimage around $b$ carry the same probability, so their densities differ by the ratio of their volumes. Applied to $D$ and $D'$, the ratio of the two output densities at any $\theta$ is
$$\frac{p_D(\theta)}{p_{D'}(\theta)} = \underbrace{\frac{\nu(\phi_D(\theta))}{\nu(\phi_{D'}(\theta))}}_{\text{noise}}\cdot\underbrace{\frac{|\det\nabla\phi_D(\theta)|}{|\det\nabla\phi_{D'}(\theta)|}}_{\text{Jacobian}}.$$
3. *The noise factor is at most $e^{\varepsilon'}$.* The sums in $\phi_D$ and $\phi_{D'}$ share every term except the replaced record's, so $\phi_{D'}(\theta) - \phi_D(\theta) = \nabla\ell(\theta; z) - \nabla\ell(\theta; z')$, of length at most 2, since each gradient $\psi'(m)\,y\,x$ has length at most $|\psi'|\,\|x\| \le 1$. This is the setting of @lem-dperm-norm-noise with sensitivity 2 and scale $s = 2/\varepsilon'$, so the noise factor is at most $e^{2/s} = e^{\varepsilon'}$.
4. *The Jacobian is the Hessian.* Differentiating $\phi_D$ gives $\nabla\phi_D(\theta) = -A$, with
$$A = \sum_i\psi''(m_i)\,x_ix_i^\top + n\lambda I,$$
where $m_i$ is the margin of record $i$. The chain rule gives each loss's Hessian as $\psi''(m_i)$ times $y_i^2x_ix_i^\top = x_ix_i^\top$. Every term is positive semi-definite, so all eigenvalues of $A$ are at least $n\lambda$. For $D'$ the matrix is $A' = A + E$, with $E = \psi''(m')x'x'^\top - \psi''(m)xx^\top$: one rank-one term in, one out.
5. *The Jacobian factor is at most $(1 + c/(n\lambda))^2$.* Bound first the ratio $\det(A')/\det(A)$, which is the factor for the pair taken in the other order. It is $\det(I + A^{-1}E)$, which equals $\det(I + K)$ for the similar, symmetric matrix $K = A^{-1/2}EA^{-1/2}$. With $\tilde x = A^{-1/2}x$, $K$ is the difference of the rank-one terms $\psi''(m')\tilde x'\tilde x'^\top$ and $\psi''(m)\tilde x\tilde x^\top$. A rank-one matrix $vv^\top$ has one nonzero eigenvalue, $\|v\|^2$, and here $\|\tilde x\|^2 = x^\top A^{-1}x \le 1/(n\lambda)$. So $K$ has at most two nonzero eigenvalues $\kappa_1$ and $\kappa_2$. Adding the positive term can raise an eigenvalue by at most its own eigenvalue, and subtracting the other can lower one by at most its own, so each lies between $-a$ and $a$, with $a = c/(n\lambda)$. Hence
$$\det(I + K) = (1 + \kappa_1)(1 + \kappa_2) \le (1 + a)^2.$$
The same argument with $D$ and $D'$ exchanged bounds $\det(A)/\det(A')$, the Jacobian factor itself, by $(1 + a)^2$, because $A'$ also has all eigenvalues at least $n\lambda$.
6. *Add the two budgets.* The log of the density ratio is at most $\varepsilon' + 2\ln(1 + c/(n\lambda))$, which is $\varepsilon$ by @eq-dperm-eps-prime, at every $\theta$ and in both directions. A bound on the density ratio everywhere is pure $\varepsilon$-DP.
7. *When $\varepsilon' \le 0$.* The Jacobian factor alone would then exceed $\varepsilon$, and a larger ridge shrinks it. With $\lambda + \lambda_+$ in place of $\lambda$, $a = e^{\varepsilon/4} - 1$, so the Jacobian takes $2\ln(1 + a) = \varepsilon/2$ and the noise gets the other half.
:::

The analysis is safe but not tight. The noise factor reaches $e^{\varepsilon'}$ only if both records have full-length gradients in opposite directions. A logistic gradient has full length only at a very negative margin, and two records whose gradients point in opposite directions have margins of opposite sign, so at most one of them can. The Jacobian bound charges both records the full curvature $c$, but the record that arrives raises the determinant while the one that leaves lowers it, so $\ln(1 + a)$ suffices, half the charge.

The method needs exact minimization, which no computer delivers, and even rounding can leak ([Discrete mechanisms](../part03-dp-foundations/08-discrete-mechanisms.md)). It also needs twice-differentiable losses, so a support vector machine's hinge loss must be smoothed first. Iyengar et al. (2019) tolerate inexact solvers by adding a little output noise.

The excess risk follows the second-order argument of output perturbation, except that the noise now enters as a force, which the curvature $H$ converts into a displacement by dividing by it.

::: {#prp-dperm-objective-risk name="Excess risk of objective perturbation"}
Under the assumptions of @prp-dperm-output-risk, with the noise $b$ of @thm-dperm-objective, to second order in $b$,
$$\mathbb E\big[\mathcal L(\tilde\theta) - \mathcal L(\theta^*)\big] \approx \frac{1}{2n^2}\mathbb E\big[b^\top H^{-1}b\big] = \frac{\mathbb E\|b\|^2}{2dn^2}\operatorname{tr}H^{-1}.$$ {#eq-dperm-objective-risk}
:::

::: {.proof}
1. *The displacement.* To first order in $b$ the gradient of $\mathcal L$ at $\tilde\theta$ is $H(\tilde\theta - \theta^*)$, and the perturbed minimizer makes it cancel the force:
$$\tilde\theta - \theta^* = -\frac1nH^{-1}b.$$
2. *Its cost.* Substituting the displacement into the quadratic of step 1 of @prp-dperm-output-risk gives $\frac{1}{2n^2}b^\top H^{-1}HH^{-1}b = \frac{1}{2n^2}b^\top H^{-1}b$.
3. *Average over the direction.* The direction of $b$ is uniform and independent of its length (@lem-dperm-norm-noise), so step 2 of @prp-dperm-output-risk applies with $H^{-1}$ in place of $H$.
:::

Curvature now helps. Output perturbation's noise scales like $1/(\lambda n)$ and is weighted by $H$, while here the noise does not depend on $n$ or $\lambda$, is divided by $n$, and is weighted by $H^{-1}$. If every eigenvalue of $H$ is close to $h$, output perturbation's excess risk is about $(h/\lambda)^2$ times objective perturbation's, when $\varepsilon' \approx \varepsilon$. When the ridge supplies all the curvature the two are equivalent, and when the data supply more, objective perturbation wins quadratically.

![Objective perturbation in one dimension, $n = 5$, $\lambda = 0.1$, $\varepsilon = 1$. Left: the exact densities of the released weight under a dataset and its neighbour, computed by change of variables. Right: their log ratio, the privacy loss, which stays well inside $\pm\varepsilon$.](figures/erm_objective_density.svg){#fig-dperm-objective-density}

::: {#exm-dperm-objective}
In one dimension the chapter's script computes both output densities exactly on a grid. Take logistic loss ($c = \frac14$) on $n = 5$ records with features in $[-1, 1]$, $\lambda = 0.1$ and $\varepsilon = 1$. Then $c/(n\lambda) = 0.5$, the Jacobian budget is $2\ln 1.5 = 0.8109$, and the noise gets $\varepsilon' = 0.1891$ (@fig-dperm-objective-density). Over 500 random datasets and replacements, the largest log ratio anywhere on the grid is 0.3779. Its noise part never exceeds 0.0992, about half its budget of 0.1891, and its Jacobian part never exceeds 0.3395, below even $\ln 1.5 = 0.4055$, which is half the Jacobian budget of 0.8109.

In the running example at $n = 10{,}000$, the Jacobian budget is $2\ln(1 + 0.25/100) = 0.0050$, so the noise gets $\varepsilon' = 0.9950$. The noise has $\mathbb E\|b\|^2 = d(d+1)(2/\varepsilon')^2$, and $\operatorname{tr}H^{-1} = 320.7$, so @eq-dperm-objective-risk predicts an excess risk of $7.127 \times 10^{-5}$, and 20 simulated releases average $8.156 \times 10^{-5}$. Pure-DP output perturbation with the same $\varepsilon$ had $8.321 \times 10^{-4}$, so objective perturbation is ten times better. The rule $(h/\lambda)^2$ predicts this, because the objective's curvature, between 0.0263 and 0.0330, is about three times the ridge's 0.01, and $3^2 = 9$.
:::

## Gradient perturbation: noisy gradient descent {#sec-dperm-gradient}

The first two methods need the exact minimizer, but real training is iterative and stops early. Gradient perturbation, proposed by Williams & McSherry (2010), makes each step of the optimizer private and composes the steps. This section asks how much noise per step makes a whole run of full-batch gradient descent private, and how accurate the run is.

Each step will be a Gaussian mechanism applied at a point that earlier steps chose. $T$ such adaptive Gaussian mechanisms, with ratios $\mu_1, \dots, \mu_T$, are exactly as private as one Gaussian mechanism with ratio $\sqrt{\mu_1^2 + \dots + \mu_T^2}$ ([f-DP and Gaussian differential privacy](../part04-privacy-accounting/04-gaussian-dp.md#sec-gdp-composition)). Given the past, each step's privacy loss is normal with variance $\mu_t^2$ and mean $\mu_t^2/2$, so their sum is normal with variance $\sum_t\mu_t^2$ and half that as its mean, which is the privacy loss of one Gaussian mechanism.

::: {#thm-dperm-noisy-gd name="Noisy gradient descent"}
Let every per-record gradient have length at most $L$, with the loss convex or not, and let neighbours replace one record. Start from $\theta_0 = 0$ and run $T$ steps of
$$\theta_{t+1} = \theta_t - \eta\big(\nabla \mathcal L(\theta_t; D) + \xi_t\big), \qquad \xi_t \sim \mathcal N(0, \sigma^2I)\ \text{independent}, \qquad \sigma = \frac{2L}{n}\cdot\frac{\sqrt T}{\mu(\varepsilon, \delta)},$$ {#eq-dperm-noisy-gd}
with learning rate $\eta$. Releasing all the iterates $\theta_1, \dots, \theta_T$ is $(\varepsilon, \delta)$-DP.
:::

::: {.proof}
1. *One step is a Gaussian mechanism.* Given $\theta_t$, the step releases $\nabla\mathcal L(\theta_t; D) + \xi_t$. The ridge term $\lambda\theta_t$ is the same on both datasets, and the averages differ only in the replaced record. So the gradient moves by at most $\frac1n\|\nabla\ell(\theta_t; z) - \nabla\ell(\theta_t; z')\| \le 2L/n$, and the step's ratio is $\mu_{\text{step}} = 2L/(n\sigma)$. The point $\theta_t$ depends on the data only through earlier released outputs, which is what adaptive composition allows.
2. *Compose.* By the composition of Gaussian mechanisms, the $T$ steps are as private as one Gaussian with ratio $\sqrt T\cdot 2L/(n\sigma)$. Setting this to $\mu(\varepsilon, \delta)$ gives the $\sigma$ of @eq-dperm-noisy-gd. The iterates are computed from the noisy gradients and $\theta_0$ alone, so releasing them is post-processing.
3. *No convexity was used.* Steps 1 and 2 bounded the gradient's change by $2L/n$ at the point $\theta_t$, whatever $\theta_t$ is. Nothing required a minimizer, convexity or smoothness.
:::

Convexity enters only for accuracy. On a $\lambda$-strongly convex, **$\beta$-smooth** objective, one whose curvature lies between $\lambda$ and $\beta$ everywhere, gradient descent with learning rate $\eta = 1/\beta$ shrinks the error by a fixed factor per step, while the noise keeps the iterate from settling.

::: {#prp-dperm-gd-risk name="Excess risk of noisy gradient descent"}
Let the objective be $\lambda$-strongly convex and $\beta$-smooth, and let $\eta = 1/\beta$. The excess risk of the last iterate is approximately
$$\mathbb E\big[\mathcal L(\theta_T) - \mathcal L(\theta^*)\big] \approx \Big(1 - \frac\lambda\beta\Big)^T\big(\mathcal L(\theta_0) - \mathcal L(\theta^*)\big) + \frac{d\sigma^2}{4\beta}.$$ {#eq-dperm-gd-risk}
:::

::: {.proof}
1. *Optimization error.* Without noise, each step shrinks $\mathcal L(\theta_t) - \mathcal L(\theta^*)$ by at least a factor $1 - \lambda/\beta$. @thm-opt-gd-quadratic proves the same factor for the distance on a quadratic, and @sec-apx-opt-convexity proves it for the excess risk of every such objective. Over $T$ steps this gives the first term.
2. *How the noise moves the error.* Near the minimizer the gradient is about $H(\theta_t - \theta^*)$, so the error $e_t = \theta_t - \theta^*$ evolves as
$$e_{t+1} \approx (I - \eta H)e_t - \eta\xi_t.$$
3. *The steady state in one direction.* Along an eigenvector of $H$ with curvature $h$, the error's variance $v$ settles where one step leaves it unchanged, $v = (1 - \eta h)^2v + \eta^2\sigma^2$. Solving gives $v = \eta\sigma^2/(h(2 - \eta h))$.
4. *Its cost.* That direction costs $\frac12hv = \eta\sigma^2/(2(2 - \eta h))$, about $\eta\sigma^2/4$ when $\eta h$ is small and at most twice that when $\eta h$ reaches 1. Summing over the $d$ directions with $\eta = 1/\beta$ gives $d\sigma^2/(4\beta)$.
:::

The curvature $h$ cancels in step 4, because flat directions wander farther but cost less per unit of wandering. What does not cancel is $T$, so the step count is a privacy parameter: every step spends budget, so the noise floor grows like $T$, while too few steps leave optimization error (@fig-dperm-noisy-gd-steps). @eq-dperm-gd-risk can be minimized from public quantities: $\lambda$, the curvature bound $\beta = \frac14 + \lambda$ of logistic loss with unit features, $\mathcal L(\theta_0) = \ln 2$ at $\theta_0 = 0$, $d$, $n$ and the budget. This **public rule** for $T$ is pessimistic, because it knows only that the curvature is at least $\lambda$. Picking $T$ by validation score is itself a computation on the data ([Hyperparameter tuning](03-hyperparameter-tuning.md)).

![Excess risk of noisy gradient descent at $n = 10{,}000$, $(\varepsilon, \delta) = (1, 10^{-5})$, against the number of steps $T$, averaged over datasets, with the same steps without noise (dashed). The noiseless error falls geometrically and reaches machine precision; the noisy run is limited by noise that grows with $T$, so its error is smallest near $T = 50$.](figures/erm_noisy_gd_steps.svg){#fig-dperm-noisy-gd-steps}

::: {#exm-dperm-noisy-gd}
For $n = 10{,}000$, the public rule picks $T = 218$. That gives per-step noise $\sigma = 0.01102$ on the mean gradient and an estimated excess risk of $1.301 \times 10^{-3}$, and simulated runs average $1.384 \times 10^{-3}$. @tbl-dperm-gd gives the rule's choice at every dataset size. The step count grows slowly, like $\ln n$, because the optimization error shrinks geometrically while its target, the noise floor, shrinks like $1/n^2$.

| $n$ | Steps $T$ (public rule) | Per-step $\sigma$ on the mean gradient | Estimated excess risk |
| --- | --- | --- | --- |
| 1,000 | 100 | 0.07461 | $6.725 \times 10^{-2}$ |
| 3,000 | 156 | 0.03106 | $1.080 \times 10^{-2}$ |
| 10,000 | 218 | 0.01102 | $1.301 \times 10^{-3}$ |
| 30,000 | 274 | 0.004117 | $1.779 \times 10^{-4}$ |
| 100,000 | 335 | 0.001366 | $1.930 \times 10^{-5}$ |

: Noisy gradient descent on the running example: the step count chosen by minimizing @eq-dperm-gd-risk with public quantities only, the resulting noise, and the estimated excess risk. {#tbl-dperm-gd}

@fig-dperm-noisy-gd-steps shows that the rule overshoots, because the data's actual curvature, between 0.0263 and 0.0330, is well above 0.01. Over 10 datasets the excess risk is smallest near $T = 50$, at $2.875 \times 10^{-4}$, five times less than the rule's, and it rises to $6.018 \times 10^{-3}$ at $T = 1{,}000$. But you may use that only if you learned it privately or from similar public data.
:::

## Noisy projected gradient descent without a ridge {#sec-dperm-convex}

Every method so far leaned on the ridge, for its privacy or for its accuracy. But a linear classifier trained without a ridge inside a fixed ball of weights is convex with bounded gradients, not strongly convex. This section asks how accurate a private trainer can be on such a model, and where the dimension enters.
The tool is the guarantee of projected SGD from [Optimization](../part01-prerequisites/06-optimization.md). Projected SGD keeps the weights in a closed convex set $\mathcal W$ fixed in advance, here the ball of radius $R$, by replacing each new point with its **projection** $\Pi_{\mathcal W}$, the nearest point of the set. It steps along a random vector $g_t$ that is right on average given the past, with squared length at most $G^2$ on average. If the objective is convex and its minimizer over $\mathcal W$ lies within $R$ of the start, then $T$ steps with learning rate $\eta = R/(G\sqrt T)$ make the average $\bar\theta_T$ of the visited points satisfy
$$\mathbb E\big[\mathcal L(\bar\theta_T)\big] - \mathcal L(\theta^*) \le \frac{RG}{\sqrt T}$$
(@thm-opt-sgd-convex). Independent noise with mean zero only adds its second moment to $G^2$ (@prp-opt-noisy).

::: {#thm-dperm-convex name="Noisy projected gradient descent"}
Let the loss be convex in $\theta$ with per-record gradients of length at most $L$, let the objective be the plain average $\mathcal L(\theta; D) = \frac1n\sum_i\ell(\theta; z_i)$, with no ridge, and let neighbours replace one record. Keep the weights in the ball $\mathcal W = \{\|\theta\| \le R\}$. Start from $\theta_1 = 0$ and run $T$ steps of
$$\theta_{t+1} = \Pi_{\mathcal W}\Big(\theta_t - \eta\big(\nabla \mathcal L(\theta_t; D) + \xi_t\big)\Big), \qquad \xi_t \sim \mathcal N(0, \sigma^2I), \qquad \sigma = \frac{2L}{n}\cdot\frac{\sqrt T}{\mu(\varepsilon, \delta)},$$ {#eq-dperm-projected-gd}
with $\eta = R/(G\sqrt T)$ and $G^2 = L^2 + d\sigma^2$, and release $\bar\theta = \frac1T\sum_{t=1}^T\theta_t$. The release is $(\varepsilon, \delta)$-DP, and its excess risk over the best weights in the ball is at most
$$\mathbb E\big[\mathcal L(\bar\theta; D)\big] - \min_{\theta\in\mathcal W}\mathcal L(\theta; D) \le RL\sqrt{\frac1T + \frac{4d}{n^2\mu(\varepsilon, \delta)^2}}.$$ {#eq-dperm-convex-bound}
Once $T \ge n^2\mu^2/(4d)$ this is at most $2\sqrt2\,RL\sqrt d/(n\mu)$, and for $\varepsilon < 1$
$$\mathbb E\big[\mathcal L(\bar\theta; D)\big] - \min_{\theta\in\mathcal W}\mathcal L(\theta; D) \le \frac{4RL\sqrt{d\ln(1.25/\delta)}}{n\varepsilon}.$$ {#eq-dperm-convex-rate}
:::

::: {.proof}
The plan is to prove privacy as for noisy gradient descent, check the two conditions of the SGD guarantee for the noisy gradient, and then see that the step count drops out of the noise term.

1. *Privacy.* Each step releases a noisy gradient whose noiseless part moves by at most $2L/n$, as in step 1 of the proof of @thm-dperm-noisy-gd, and the $T$ steps compose into one Gaussian with ratio $\mu(\varepsilon, \delta)$. The projection uses only the public ball, and the average uses only the iterates, so both are post-processing.
2. *The noisy gradient is unbiased.* Write $g_t = \nabla\mathcal L(\theta_t; D) + \xi_t$. Given the past, $\xi_t$ is fresh noise with mean zero, so $g_t$ averages to the true gradient.
3. *Its second moment is at most $L^2 + d\sigma^2$.* The gradient of $\mathcal L$ is an average of vectors of length at most $L$, so its squared length is at most $L^2$. The noise is independent with mean zero, so by @prp-opt-noisy it adds its own second moment, one $\sigma^2$ per coordinate, $d\sigma^2$ in all.
4. *Apply the SGD guarantee.* The minimizer over the ball and the start $\theta_1 = 0$ both lie in the ball, so they are at most $R$ apart. With $\eta = R/(G\sqrt T)$, @thm-opt-sgd-convex bounds the excess risk of $\bar\theta$ by
$$\frac{RG}{\sqrt T} = R\sqrt{\frac{L^2 + d\sigma^2}{T}}.$$
5. *The step count cancels from the noise term.* By @eq-dperm-projected-gd, $\sigma^2 = 4L^2T/(n^2\mu^2)$, so $d\sigma^2/T = 4L^2d/(n^2\mu^2)$ whatever $T$ is. Factoring out $L^2$ gives @eq-dperm-convex-bound.
6. *Choose $T$.* The first term under the root falls with $T$ and the second does not. When $T \ge n^2\mu^2/(4d)$ the first is at most the second, and the root is at most $\sqrt{8d/(n^2\mu^2)} = 2\sqrt2\,\sqrt d/(n\mu)$.
7. *In terms of $\varepsilon$.* For $\varepsilon < 1$ the classic noise $\Delta_2 f\sqrt{2\ln(1.25/\delta)}/\varepsilon$ is $(\varepsilon, \delta)$-DP (@thm-lg-classic). The exact calibration uses the least noise that is $(\varepsilon, \delta)$-DP, so its ratio is at least the classic one,
$$\frac1\mu \le \frac{\sqrt{2\ln(1.25/\delta)}}{\varepsilon}.$$
Substituting this into step 6 gives $2\sqrt2\cdot\sqrt2 = 4$ in front, which is @eq-dperm-convex-rate.
:::

Privacy noise is just more gradient variance. The SGD guarantee cares only about the second moment of $g_t$, and Gaussian noise raises it by one $\sigma^2$ for every coordinate the optimizer moves, which [Public data and private fine-tuning](06-public-data.md) exploits by training fewer coordinates. Unlike in @sec-dperm-gradient, more steps never hurt, because step 5 cancels the growth of the noise. The step count only has to reach $n^2\mu^2/(4d)$, but each full-batch step costs $n$ gradient evaluations, so Bassily, Smith & Thakurta use one random record per step instead, which by [Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md) also needs less noise.

The rate cannot be beaten. As long as $\sqrt d \le n\varepsilon$, every $(\varepsilon, \delta)$-DP algorithm has excess risk of order $RL\sqrt d/(n\varepsilon)$ on some dataset, by the fingerprinting argument of [Lower bounds](../part03-dp-foundations/10-lower-bounds.md) (Bassily, Smith & Thakurta 2014). @sec-apx-learn-erm derives this lower bound with explicit constants, and @eq-dperm-convex-rate matches it up to $\sqrt{\ln(1/\delta)}$ and constants. It is a worst case, reached by losses nearly linear near their minimizer.

![Noisy projected gradient descent on unregularized logistic regression in the ball of radius 5, $d = 10$, $(1, 10^{-5})$-DP, with $T = \lceil n^2\mu^2/(4d)\rceil$ steps: the measured excess risk over the best weights in the ball (circles) against the bound @eq-dperm-convex-bound (squares), averaged over 3 datasets drawn separately from those of the table. The bound falls like $1/n$; the measured excess risk falls faster.](figures/erm_convex.svg){#fig-dperm-convex}

::: {#exm-dperm-convex}
Take the running example's data, with $d = 10$ and $L = 1$, drop the ridge, and keep the weights in the ball of radius $R = 5$. At $\varepsilon = 1$ and $\delta = 10^{-5}$ the exact ratio is $\mu = 0.2681$, and the classic ratio $\varepsilon/\sqrt{2\ln(1.25/\delta)}$ is smaller, 0.2064, as step 7 requires. With a thousand records the rule $T = \lceil n^2\mu^2/(4d)\rceil$ gives 1,797 steps, and the noise on the mean gradient is $\sigma = 0.3163$. The noise adds $d\sigma^2 = 10 \times 0.3163^2$ to the second moment, so $G^2 = 2.0004$, about twice the noiseless $L^2 = 1$. The learning rate is then 0.0834, and @eq-dperm-convex-bound evaluates to 0.1668.

As $T$ grows, the first term under the root vanishes and the bound falls to $2RL\sqrt d/(n\mu) = 0.1180$, a factor $\sqrt2$ below its value at the threshold. It is 0.5137 after 100 steps, 0.1973 after 1,000 and 0.1190 after 100,000, within 1% of that limit. @tbl-dperm-convex runs the algorithm for three dataset sizes, five datasets each.

| $n$ | Steps $T$ | $\sigma$ | Measured excess risk | Bound @eq-dperm-convex-bound | Bound / measured |
| --- | --- | --- | --- | --- | --- |
| 300 | 162 | 0.3166 | $1.119 \times 10^{-1}$ | $5.558 \times 10^{-1}$ | 5.0 |
| 1,000 | 1,797 | 0.3163 | $3.028 \times 10^{-2}$ | $1.668 \times 10^{-1}$ | 5.5 |
| 3,000 | 16,167 | 0.3162 | $7.297 \times 10^{-3}$ | $5.561 \times 10^{-2}$ | 7.6 |

: Noisy projected gradient descent without a ridge on the running example's data, ball of radius 5, $(1, 10^{-5})$-DP: mean excess risk over 5 datasets against the bound. {#tbl-dperm-convex}

The bound holds at every $n$ and is loose by a factor of 5 to 8. Its log-log slope in $n$ is $-1.00$, while the measured slope is $-1.18$, steeper because the logistic loss has curvature near its minimizer that the bound does not use (@fig-dperm-convex). The rule's $\sigma$ is the same at every $n$, because $T \propto n^2$ and $\sigma \propto \sqrt T/n$.
:::

## How the three compare {#sec-dperm-compare}

Each method is private by its own argument, so which should you use? Theory answers by rates, how the excess risk falls with $n$, $d$ and $\varepsilon$, and simulation answers for the running example. This section shows that they agree on the rate and differ in the constant, which is what you choose by.

With a guaranteed curvature the rate improves. The convex rate of @sec-dperm-convex falls like $1/(n\varepsilon)$, and for a $\lambda$-strongly convex objective it is replaced by its square, divided by $\lambda$.

::: {#thm-dperm-rates name="Optimal rate for strongly convex objectives"}
For $\lambda$-strongly convex objectives whose losses have gradients of length at most $L$, the best expected excess risk achievable under $(\varepsilon, \delta)$-DP is of order
$$\frac{L^2\,d}{\lambda\,n^2\varepsilon^2},$$ {#eq-dperm-rate-strong}
up to logarithmic factors in $n$ and $1/\delta$.
:::

<!-- proof: cited -->

Bassily, Smith & Thakurta (2014) prove both sides, the upper bound with noisy stochastic gradient descent and the lower bound with fingerprinting codes, and @sec-apx-learn-erm in the appendices proves both with explicit constants. Output and objective perturbation fall at this rate in $n$ because their noise has second moment proportional to $1/n^2$. Noisy gradient descent loses a logarithm, because pushing its optimization error below a noise floor $\propto T/n^2$ takes $T \propto \ln n$ steps.

| $n$ | Output, Gaussian | Output, pure | Objective, pure | Noisy gradient descent (public $T$) |
| --- | --- | --- | --- | --- |
| 1,000 | $8.170 \times 10^{-2}$ | $7.060 \times 10^{-2}$ | $1.009 \times 10^{-2}$ | $6.558 \times 10^{-2}$ |
| 3,000 | $1.211 \times 10^{-2}$ | $7.206 \times 10^{-3}$ | $6.303 \times 10^{-4}$ | $8.888 \times 10^{-3}$ |
| 10,000 | $9.274 \times 10^{-4}$ | $8.321 \times 10^{-4}$ | $8.156 \times 10^{-5}$ | $1.384 \times 10^{-3}$ |
| 30,000 | $9.973 \times 10^{-5}$ | $8.943 \times 10^{-5}$ | $7.135 \times 10^{-6}$ | $1.538 \times 10^{-4}$ |
| 100,000 | $8.543 \times 10^{-6}$ | $6.379 \times 10^{-6}$ | $8.798 \times 10^{-7}$ | $1.621 \times 10^{-5}$ |
| Slope in $\ln n$ | $-2.01$ | $-2.00$ | $-2.01$ | $-1.79$ |

: Mean excess risk $\mathcal L(\tilde\theta) - \mathcal L(\theta^*)$ over 20 datasets of the running example ($d = 10$, $\lambda = 0.01$, $\varepsilon = 1$, $\delta = 10^{-5}$ for the Gaussian methods), and the slope of a straight-line fit of $\ln$ excess risk against $\ln n$. {#tbl-dperm-compare}

On the running example (@tbl-dperm-compare, @fig-dperm-comparison) all four releases fall like $1/n^2$, and noisy gradient descent a little slower, $-1.79$, because its step count grows with $\ln n$. Objective perturbation is about ten times better than output perturbation at every $n$, the $(h/\lambda)^2$ of @sec-dperm-objective. Noisy gradient descent with the public step rule is the worst of the three for $n \ge 10{,}000$, and tuned privately to $T \approx 50$ it would sit between the two perturbation methods. Accuracy on held-out data suffers even less. The non-private model has accuracy 0.747 at $n = 1{,}000$, which output perturbation drops to 0.681, noisy gradient descent to 0.695 and objective perturbation only to 0.736. By $n = 10{,}000$ all four private releases are within 0.002 of the non-private 0.753.

The pure-DP column is slightly better because, by @lem-dperm-norm-noise, at $d = 10$ and $\varepsilon = 1$ its noise has mean squared length $d(d + 1)(\Delta_2 f)^2 = 110\,(\Delta_2 f)^2$, against $d \times 3.7306^2(\Delta_2 f)^2 = 139.2\,(\Delta_2 f)^2$ for the Gaussian. At $d = 100$ it is $10{,}100\,(\Delta_2 f)^2$ against $1{,}392\,(\Delta_2 f)^2$, the price of $\delta = 0$ that [Lower bounds](../part03-dp-foundations/10-lower-bounds.md) proves unavoidable.

A deployed model is judged not on the training objective but by its **population risk**, the expected loss on a new record from the same distribution. For convex Lipschitz losses over a ball, Bassily, Feldman, Talwar & Thakurta (2019) show that the best population excess risk is of order
$$LR\Big(\frac{1}{\sqrt n} + \frac{\sqrt{d\ln(1/\delta)}}{n\varepsilon}\Big).$$
The first term is what any learner pays for seeing only $n$ samples, so once $n$ exceeds about $d\ln(1/\delta)/\varepsilon^2$, privacy adds less than sampling already costs.

![Excess risk of the four private releases against the dataset size on the running example, log-log, with a line of slope $-2$. All fall like $1/n^2$; objective perturbation is about ten times better than output perturbation throughout.](figures/erm_comparison.svg){#fig-dperm-comparison}

::: {#exm-dperm-compare}
@eq-dperm-rate-strong predicts how much data you need. Suppose an excess risk of $10^{-4}$ is acceptable, and output perturbation gives $9.274 \times 10^{-4}$ at $n = 10{,}000$. The rate $1/n^2$ says that multiplying $n$ by $\sqrt{9.274} = 3.05$ brings it to about $10^{-4}$, and @tbl-dperm-compare confirms it, with $9.973 \times 10^{-5}$ at $n = 30{,}000$. Halving $\varepsilon$ costs as much as halving $n$, because every noise scale in this chapter is proportional to $1/(n\varepsilon)$.
:::

## When the model is not convex {#sec-dperm-nonconvex}

A neural network's loss is not convex in its parameters, and the privacy proofs of output and objective perturbation both need a unique minimizer that moves only a little when one record changes. This section asks what survives of the three methods, and why the next chapter keeps only one.

Without convexity the minimizer can jump. [Optimization](../part01-prerequisites/06-optimization.md) averages the per-record loss $\frac14(\theta^2 - 1)^2 - y\theta$, with labels $y = \pm1$, into a risk with two wells, minimized in the well of the majority label. With 501 labels $+1$ and 500 labels $-1$, changing one label flips the majority and moves the minimizer by 2.0010. No $1/n$ appears, so output perturbation would need noise whose scale does not shrink with the dataset.

Objective perturbation breaks at step 1 of its proof, because the perturbed objective can have several minimizers and the map from noise to output is no longer invertible. Neel, Roth, Vietri & Wu (2020) prove a variant private without convexity, but only when an oracle returns the global minimizer of the perturbed objective. Gradient descent on a neural network finds a local minimum, about which that guarantee says nothing. Gradient perturbation survives, once clipping supplies the gradient bound its proof needs, just as [Beyond global sensitivity](../part03-dp-foundations/07-sensitivity.md#sec-sens-clip) clips a value before averaging it.

::: {#prp-dperm-clipped name="Noisy gradient descent with clipping"}
For any differentiable loss, convex or not, replace each per-record gradient by its clipped version $\nabla\ell(\theta; z_i)\min(1, C/\|\nabla\ell(\theta; z_i)\|)$, for a clipping norm $C$. Noisy gradient descent with the average of clipped gradients in place of $\nabla\mathcal L$, and with $L = C$ in @eq-dperm-noisy-gd, is $(\varepsilon, \delta)$-DP for replace-one neighbours.
:::

::: {.proof}
1. *The clipped gradients are bounded.* A clipped gradient has length at most $C$, and is unchanged when it is already shorter.
2. *The step's sensitivity.* The average of clipped gradients loses one term of length at most $C/n$ and gains another, so it moves by at most $2C/n$.
3. *The rest of the proof.* The proof of @thm-dperm-noisy-gd used only this bound at the visited points (its step 3), so it goes through with $L = C$.
:::

The non-convex case loses the guarantee of accuracy, not of privacy. Noisy gradient descent on a neural network is private whatever the network does, and whether it learns is an empirical question. Clipping also changes what is optimized, because the average of clipped gradients is not the gradient of $\mathcal L$, and the [next chapter](02-dp-sgd.md) measures that bias. Random minibatches, which make each step cheaper and more private, complete DP-SGD, stochastic gradient descent with per-example clipping and Gaussian noise.

::: {#exm-dperm-two-wells}
Take the double-well loss with $n = 1{,}001$ records. Output perturbation at $\varepsilon = 1$ needs $\sigma = 3.7306 \times 2.0010 = 7.465$, more than three times the distance between the wells, whatever the dataset size. The released value then says almost nothing about which well the data chose. Privacy demands exactly this, since one record decides the well, but the release is useless. Noisy gradient descent with clipping still moves toward the well the noisy average gradient points to. When many records agree, that gradient is large compared with the noise, and the method finds the majority well with high probability.
:::

## Code {#sec-dperm-code}

The script [code/part05-dp-ml/dp_erm.py](../../code/part05-dp-ml/dp_erm.py) runs the three methods on regularized logistic regression with exact Newton solves, computes the exact one-dimensional densities of objective perturbation, and checks noisy projected gradient descent against its bound. From the repository root:

```bash
uv run python code/part05-dp-ml/dp_erm.py
```

| Experiment | Result |
| --- | --- |
| Gaussian calibration, $\varepsilon = 1$, $\delta = 10^{-5}$ | $\mu = 0.2681$; exact $\sigma = 3.7306\,\Delta_2 f$ against classic $4.8448\,\Delta_2 f$ |
| Output perturbation, one dimension | Minimizers 0.0832 apart, bound 0.2 |
| Output perturbation, $n = 10{,}000$ | Predicted excess risk $8.711 \times 10^{-4}$, measured $9.274 \times 10^{-4}$ |
| Approximate minimizer, $\tau = 10^{-3}$ | Sensitivity 0.22 against 0.02 exact |
| Objective perturbation, one dimension, 500 datasets | Largest log ratio 0.3779 against $\varepsilon = 1$; noise part 0.0992 of 0.1891, Jacobian part 0.3395 of 0.8109 |
| Objective perturbation, $n = 10{,}000$ | Predicted $7.127 \times 10^{-5}$, measured $8.156 \times 10^{-5}$ |
| Noisy gradient descent, $n = 10{,}000$ | Public rule $T = 218$, measured $1.384 \times 10^{-3}$; best $T \approx 50$ gives $2.875 \times 10^{-4}$ |
| Noisy projected gradient descent, no ridge, $n = 1{,}000$ | $T = 1{,}797$, bound 0.1668, measured $3.028 \times 10^{-2}$ |
| Slopes in $\ln n$ | $-2.01$, $-2.00$, $-2.01$, $-1.79$ with a ridge; $-1.18$ measured and $-1.00$ for the bound without |

: The script's output with the default seed. {#tbl-dperm-code}

[tests/test_dp_erm.py](../../tests/test_dp_erm.py) checks that the output still matches.

## Summary {#sec-dperm-summary}

| Result | Statement | Use |
| --- | --- | --- |
| @lem-dperm-norm-noise | Noise $\propto e^{-\Vert b\Vert/s}$ with $s = \Delta_2 f/\varepsilon$ is $\varepsilon$-DP; $\mathbb E\Vert b\Vert^2 = d(d+1)s^2$ | Pure-DP output and objective perturbation; costs a factor growing like $d$ against the Gaussian |
| @thm-dperm-output | $\theta^* + \mathcal N(0, \sigma^2I)$ with $\sigma = 2L/(\lambda n\mu)$ is $(\varepsilon, \delta)$-DP | Convex models trained to convergence; the baseline the others are measured against |
| @prp-dperm-output-risk | Excess risk $\approx\mathbb E\Vert Z\Vert^2\operatorname{tr}H/(2d)$ | Predicts the cost of privacy; it pays for the ridge's curvature and falls like $1/n^2$ |
| @prp-dperm-approx | Sensitivity $2L/(\lambda n) + 2\tau/\lambda$ at gradient tolerance $\tau$ | Stop on a gradient check with $\tau$ well below $L/n$, never on a step count |
| @thm-dperm-objective | Minimizing $\mathcal L + \langle b, \theta\rangle/n$ is $\varepsilon$-DP, with $\varepsilon - 2\ln(1 + c/(n\lambda))$ left for the noise | Private logistic regression and smoothed support vector machines; needs an exact minimizer |
| @prp-dperm-objective-risk | Excess risk $\approx\mathbb E\Vert b\Vert^2\operatorname{tr}H^{-1}/(2dn^2)$ | Beats output perturbation by about $(h/\lambda)^2$ when the data add curvature $h$ |
| @thm-dperm-noisy-gd | Noise $\sigma = (2L/n)\sqrt T/\mu$ on the mean gradient; all iterates $(\varepsilon, \delta)$-DP, convex or not | The privacy argument of [DP-SGD](02-dp-sgd.md); every iterative private trainer |
| @prp-dperm-gd-risk | Optimization error shrinking like $(1 - \lambda/\beta)^T$, plus a noise floor $d\sigma^2/(4\beta)$ | The step count is part of the budget; choose it from public quantities, or pay for choosing it |
| @thm-dperm-convex | No ridge, ball of radius $R$: excess risk $\le 4RL\sqrt{d\ln(1.25/\delta)}/(n\varepsilon)$ | The $\sqrt d/(n\varepsilon)$ rate, optimal up to logarithms; noise is more gradient variance; [Public data](06-public-data.md) |
| @thm-dperm-rates | Strongly convex: $L^2d/(\lambda n^2\varepsilon^2)$, up to logarithms | All three methods reach it; halving $\varepsilon$ costs as much as halving the data |
| @prp-dperm-clipped | Clipped noisy gradient descent is private for any loss | Why deep learning uses gradient perturbation, with clipping |

: The results of this chapter, and where each is used. {#tbl-dperm-summary}

## References {#sec-dperm-references}

- Chaudhuri & Monteleoni, "Privacy-preserving logistic regression" (NeurIPS 2008). Output and objective perturbation for logistic regression, the first private ERM algorithms.
- Chaudhuri, Monteleoni & Sarwate, "Differentially Private Empirical Risk Minimization" (JMLR 2011). The general analysis of both: output perturbation (Algorithm 1), objective perturbation (Algorithm 2) and their excess-risk bounds. The reference for @sec-dperm-output and @sec-dperm-objective.
- Kifer, Smith & Thakurta, "Private Convex Empirical Risk Minimization and High-dimensional Regression" (COLT 2012). Objective perturbation with Gaussian noise, constraints and non-smooth losses.
- Williams & McSherry, "Probabilistic Inference and Differential Privacy" (NeurIPS 2010). The first use of noisy gradient steps for private learning.
- Song, Chaudhuri & Sarwate, "Stochastic gradient descent with differentially private updates" (IEEE GlobalSIP 2013). Gradient perturbation for SGD, with minibatches.
- Bassily, Smith & Thakurta, "Private Empirical Risk Minimization: Efficient Algorithms and Tight Error Bounds" (FOCS 2014). Noisy stochastic gradient methods that attain the optimal rates, and the matching lower bounds; the reference for @sec-dperm-convex and @sec-dperm-compare.
- Bassily, Feldman, Talwar & Thakurta, "Private Stochastic Convex Optimization with Optimal Rates" (NeurIPS 2019). The optimal population risk under DP, the rate quoted in @sec-dperm-compare.
- Neel, Roth, Vietri & Wu, "Oracle Efficient Private Non-Convex Optimization" (ICML 2020). Objective perturbation without convexity, given an optimization oracle; the caveat of @sec-dperm-nonconvex.
- Mironov, "On Significance of the Least Significant Bits for Differential Privacy" (CCS 2012). Floating-point noise sampling breaks DP; why exact minimization is a real assumption.
- Wu, Li, Kumar, Chaudhuri, Jha & Naughton, "Bolt-on Differential Privacy for Scalable Stochastic Gradient Descent-based Analytics" (SIGMOD 2017). Output perturbation for SGD run for a fixed number of passes, without an exact minimizer.
- Iyengar, Near, Song, Thakkar, Thakurta & Wang, "Towards Practical Differentially Private Convex Optimization" (IEEE S&P 2019). Approximate minima perturbation, an objective perturbation that tolerates inexact solvers, and a benchmark of all three methods.
- Balle & Wang, "Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising" (ICML 2018). The exact calibration @eq-dperm-gauss-delta.
- Dong, Roth & Su, "Gaussian Differential Privacy" (JRSS-B 2022). The composition of Gaussian mechanisms used in @sec-dperm-gradient.
