# Private learning and tuning

This appendix derives five results that the chapters on private machine learning state but do not prove in full. All five are about learning under differential privacy (DP). Each one takes a guarantee for a small building block and follows it through a learning procedure. The block is one noisy gradient step, one noisy vote, one training run, one record of a user, or one step's noise. The chapters cite papers for these proofs, sketch them, or check them only numerically. Here each is derived step by step from facts the book has already met, and each is checked by a script. @tbl-apx-learn-map lists them.

| Derivation | Where the chapter states or uses it |
| --- | --- |
| The best excess risk of private convex empirical risk minimization, from above and below, and exact bounds for output and objective perturbation | @sec-dperm-compare, @sec-dperm-convex, @sec-dperm-output and @sec-dperm-objective |
| The data-dependent Rényi bound for a noisy vote of teachers | @sec-pate-data-dependent |
| The Rényi bound for the best of a random number of training runs | @sec-hpt-rdp |
| The worst case for a user with $k$ records in private gradient descent with record-level sampling | @sec-ulvl-group-sgd |
| How one step of private gradient descent changes the loss of one group | @sec-fair-noise |

: The derivations of this appendix and the chapter sections that state or use them. {#tbl-apx-learn-map}

The page keeps one notation throughout, even where a source paper uses other letters. The model's parameters are $\theta$, the Rényi order is always $\alpha$, the learning rate is $\eta$ and the sampling rate is $q$. The letter $\lambda$ is a ridge coefficient in the first section and a curvature in the last, and $\mu$ is a ratio of sensitivity to noise. Each section names its letters again where they first appear.

## The optimal rates of private convex learning {#sec-apx-learn-erm}

### What the chapter uses

[Private empirical risk minimization](../part05-dp-ml/01-dp-erm.md) trains a model by **empirical risk minimization (ERM)**: it minimizes the average loss on $n$ training records, plus a ridge penalty. For a dataset $D$ the objective is
$$\mathcal L(\theta; D) = \frac1n\sum_{i=1}^n\ell(\theta; z_i) + \frac\lambda2\|\theta\|^2.$$
Here $\theta$ is the vector of model parameters in $d$ dimensions, $z_i$ is the $i$-th record, and we write $\mathcal L(\theta)$ when the dataset is fixed. The loss $\ell$ is convex, and its gradient has length at most a constant $L$. The ridge coefficient is $\lambda$, and the minimizer of the objective $\mathcal L$ is $\theta^*$.

A private algorithm releases parameters $\tilde\theta$, and its **excess risk** is $\mathcal L(\tilde\theta) - \mathcal L(\theta^*)$, how much worse they fit than the minimizer. Neighbouring datasets differ by replacing one record, and the guarantee is DP with privacy parameters $\varepsilon$ and $\delta$.

@sec-dperm-compare states in @eq-dperm-rate-strong that the best expected excess risk any $(\varepsilon, \delta)$-DP algorithm can reach is of order
$$\frac{L^2d}{\lambda n^2\varepsilon^2},$$
up to logarithms. @sec-dperm-convex states the matching lower bound of order $RL\sqrt d/(n\varepsilon)$ for losses without a ridge, with weights in a ball of radius $R$. The chapter says that "the proofs of the optimal rate and of the lower bound are long" and cites Bassily, Smith & Thakurta (2014) for both. It also gives the excess risk of output perturbation, @eq-dperm-output-risk, and of objective perturbation, @eq-dperm-objective-risk, only to second order in the noise.

This section proves three things. The first is an upper bound of the right order, with explicit constants, for noisy projected gradient descent. The second is a lower bound for every private algorithm, again with explicit constants, which proves the rate up to those constants. The third is a pair of exact two-sided bounds that replace the chapter's second-order formulas.

The upper bound uses the exact Gaussian calibration of the chapter. There, $\mu(\varepsilon, \delta)$ is the largest ratio of sensitivity to noise at which one Gaussian release is $(\varepsilon, \delta)$-DP, given by @eq-dperm-gauss-delta. The algorithm is noisy projected gradient descent, started from $\theta_1 = 0$ and kept in the ball of radius $R = L/\lambda$. It runs for $T$ steps, and the learning rate at step $t$ is $\eta_t = 2/(\lambda(t + 1))$. Each gradient gets Gaussian noise of standard deviation $\sigma = (2L/n)\sqrt T/\mu(\varepsilon, \delta)$ per coordinate. The release $\tilde\theta$ is the average of the iterates in which iterate $t$ has weight proportional to $t$. If the number of steps is $T \ge n^2\mu^2/d$, then
$$\mathbb E\big[\mathcal L(\tilde\theta)\big] - \mathcal L(\theta^*) \le \frac{16L^2d}{\lambda n^2\mu(\varepsilon, \delta)^2} \le \frac{32L^2d\ln(1.25/\delta)}{\lambda n^2\varepsilon^2}.$$ {#eq-apx-learn-upper}
The second form holds for $\varepsilon < 1$.

The lower bound uses a number $K$ that measures how many records a private mean needs. The fingerprinting bound of [Lower bounds](../part03-dp-foundations/10-lower-bounds.md) says that estimating $d$ means of $\pm1$ attributes privately to a constant accuracy needs a number of records proportional to
$$K = \frac{\sqrt d}{(e^\varepsilon - 1) + 4\delta\sqrt d}.$$
The denominator is how much one record can influence a correlation score under DP, and the numerator is the spread of that score. The proof below builds hard datasets with $m$ real records out of $n$, for an integer $m$ that it is free to choose. For every $(\varepsilon, \delta)$-DP algorithm there is a dataset of $n$ records, with linear losses whose gradients have length at most $L$, on which
$$\mathbb E\big[\mathcal L(\tilde\theta)\big] - \mathcal L(\theta^*) \ge \frac{L^2}{2\lambda}\max_{1 \le m \le n}\Big[\frac mn\Big(\frac13 - \frac{m}{2K}\Big)\Big]^2 \ge \frac{L^2K^2}{2592\,\lambda n^2}\quad\text{when } 3 \le K \le 3n.$$ {#eq-apx-learn-lower}
The range for $K$ makes the best choice of $m$, about $K/3$, a whole number between 1 and $n$. When $\delta$ is small enough that $4\delta\sqrt d \le e^\varepsilon - 1$, the number $K$ is at least $\sqrt d/(2(e^\varepsilon - 1))$. Then, still for $3 \le K \le 3n$, the bound is at least $L^2d/(10368\,\lambda n^2(e^\varepsilon - 1)^2)$. For $\varepsilon \le 1$, $e^\varepsilon - 1$ is between $\varepsilon$ and $1.72\varepsilon$, so this is of the order of @eq-dperm-rate-strong. Without a ridge, over the ball of radius $R$ and with the same $K$, every private algorithm has excess risk at least
$$\max_{1\le m\le n}\frac{mLR}{16n}\Big(\frac23 - \frac mK\Big)^2 \ge \frac{LRK}{729\,n}\quad\text{when } \tfrac92 \le K \le \tfrac92n,$$ {#eq-apx-learn-lower-convex}
which is of order $RL\sqrt d/(n\varepsilon)$.

The exact bounds sandwich the excess risk between two multiples of the noise's average squared length. Write $\beta_s$ for the largest curvature of $\mathcal L$, for example $\beta_s = \frac14 + \lambda$ for the chapter's logistic regression with features of length at most 1. Output perturbation adds noise $Z$ to $\theta^*$, and objective perturbation adds $\frac1n\langle b, \theta\rangle$ to the objective. Then
$$\frac\lambda2\mathbb E\|Z\|^2 \le \mathbb E\big[\mathcal L(\theta^* + Z) - \mathcal L(\theta^*)\big] \le \frac{\beta_s}2\mathbb E\|Z\|^2, \qquad \frac{\lambda\,\mathbb E\|b\|^2}{2\beta_s^2n^2} \le \mathbb E\big[\mathcal L(\tilde\theta) - \mathcal L(\theta^*)\big] \le \frac{\mathbb E\|b\|^2}{2\lambda n^2}.$$ {#eq-apx-learn-sandwich}

### The idea

The upper bound treats privacy noise as one more source of gradient noise. Stochastic gradient descent (SGD) on a strongly convex objective converges at a rate that depends on the noise only through the average squared length $G^2$ of the noisy gradient. Privacy adds $d\sigma^2$ to $G^2$, one $\sigma^2$ per coordinate. The noise per step must grow like $\sqrt T$ to keep the total privacy fixed, so $d\sigma^2$ grows like $T$. But the convergence bound divides $G^2$ by $T$, so the number of steps cancels from the noise term. What is left is $d$ times the noise of one full-budget release, which is the rate.

The lower bound turns a private trainer into a private mean estimator, which the fingerprinting argument says cannot be accurate with few records. Take the loss $-\langle \theta, x\rangle$ on records $x$, plus the ridge. Then the objective is a bowl centred on the mean of the records divided by $\lambda$, and the excess risk of any $\theta$ is $\frac\lambda2$ times its squared distance from that centre. A trainer with small excess risk therefore outputs a good estimate of the mean of the records.

The rate in $n$ comes from padding with records equal to zero. @fig-apx-learn-reduction shows the move in two dimensions. Four real records sit at corners of a square, and four zero records sit at the origin. The mean of all eight is half the mean of the four real ones, so multiplying the trainer's output by $n/m = 2$ estimates the real mean. That multiplication also doubles the trainer's error. So a trainer whose error is $e$ on $n$ records gives an estimator whose error is $(n/m)e$ on $m$ records. Fingerprinting says the estimator's error is at least a constant when $m$ is about $K$ or less, so the trainer's error is at least that constant times $m/n$, about $K/n$. Squaring a distance of order $K/n$ gives an excess risk of order $K^2/n^2$.

![The reduction in two dimensions, with $L = \lambda = 1$. Four real records sit at corners of the square (one corner holds two), and four zero records at the origin. The minimizer $\theta^*$ of the objective is the mean of all eight, and the grey circles are levels of the excess risk. Multiplying a private output by $n/m = 2$ estimates the mean of the four real records, with twice the output's error.](figures/learn_erm_reduction.svg){#fig-apx-learn-reduction}

The exact bounds hold because a strongly convex, smooth function lies between two parabolas around its minimizer. The excess risk of any point is at least $\frac\lambda2$ and at most $\frac{\beta_s}2$ times its squared distance from $\theta^*$. The chapter's second-order formula sits between the two, with the true curvature in place of $\lambda$ or $\beta_s$.

### Derivation

**The upper bound.** Three facts from earlier chapters do the work, and each is restated where it is used.

1. **Privacy.** Each step releases the noisy gradient $\nabla\mathcal L(\theta_t) + \xi_t$, where the noise $\xi_t$ is fresh, normal, with mean zero and variance $\sigma^2$ in each coordinate. Replacing one record changes the average of the loss gradients by at most $2L/n$, because one gradient of length at most $L$ leaves and another arrives. The ridge term $\lambda \theta_t$ is the same on both datasets. So each step is a Gaussian mechanism with ratio $2L/(n\sigma)$. By Fact 5 of [Private empirical risk minimization](../part05-dp-ml/01-dp-erm.md), $T$ adaptive Gaussian steps are exactly as private as one Gaussian release with ratio $\sqrt T\cdot 2L/(n\sigma)$, and the choice of $\sigma$ makes this $\mu(\varepsilon, \delta)$. The projection and the average use only released values, so they are post-processing.
2. **The ball contains the minimizer.** The gradient of $\mathcal L$ is zero at $\theta^*$, so $\lambda \theta^* = -\frac1n\sum_i\nabla\ell(\theta^*; z_i)$. The right side is an average of vectors of length at most $L$, so $\|\theta^*\| \le L/\lambda = R$. The minimizer over the ball is therefore the overall minimizer.
3. **The noisy gradient's second moment.** On the ball, $\|\nabla\mathcal L(\theta)\| \le L + \lambda\|\theta\| \le 2L$. The noise has mean zero and is independent of $\theta_t$, so the cross term vanishes when the square is expanded:
$$\mathbb E\|\nabla\mathcal L(\theta_t) + \xi_t\|^2 = \|\nabla\mathcal L(\theta_t)\|^2 + \mathbb E\|\xi_t\|^2 \le 4L^2 + d\sigma^2 = G^2.$$
4. **The bound for SGD on a strongly convex objective.** [Optimization](../part01-prerequisites/06-optimization.md) proves @eq-sgd-strong. It is about a function $F$ that is $\lambda$-strongly convex on a closed convex set containing its minimizer. The random step directions must be right on average, with average squared length at most $G^2$. Then projected steps with learning rates $2/(\lambda(t + 1))$ and the average weighted by $t$ satisfy
$$\mathbb E[F(\tilde\theta_T)] - F^* \le \frac{2G^2}{\lambda(T + 1)}.$$
The proof follows the squared distance to the minimizer, and the noise enters only through $G^2$. The ridge makes $\mathcal L$ $\lambda$-strongly convex, and step 2 puts the minimizer in the ball, so the bound applies to $\mathcal L$.
5. **Substitute the noise.** The calibration gives $\sigma^2 = 4L^2T/(n^2\mu^2)$. So
$$\frac{2G^2}{\lambda(T + 1)} = \frac{8L^2}{\lambda(T + 1)} + \frac{8L^2d}{\lambda n^2\mu^2}\cdot\frac{T}{T + 1} \le \frac{8L^2}{\lambda(T + 1)} + \frac{8L^2d}{\lambda n^2\mu^2}.$$
The noise term no longer depends on $T$.
6. **Choose $T$.** When $T \ge n^2\mu^2/d$, the first term is at most the second, so the sum is at most $16L^2d/(\lambda n^2\mu^2)$. This is the first form of @eq-apx-learn-upper.
7. **In terms of $\varepsilon$.** For $\varepsilon < 1$ the classic Gaussian calibration is $(\varepsilon, \delta)$-DP. It adds noise $\Delta\sqrt{2\ln(1.25/\delta)}/\varepsilon$ to a function whose sensitivity is $\Delta$. The exact calibration uses the least noise that works, so its ratio is at least the classic one, and
$$\frac1{\mu^2} \le \frac{2\ln(1.25/\delta)}{\varepsilon^2}.$$
Substituting this doubles the constant to 32.

The other schedule of that section, learning rates $1/(\lambda t)$ with the uniform average, has the bound $G^2(1 + \ln T)/(2\lambda T)$. The same substitution gives at most $4L^2d(1 + \ln T)/(\lambda n^2\mu^2)$ for $T \ge n^2\mu^2/d$. Since $T$ grows like $n^2$, the factor $1 + \ln T$ grows like $\ln n$. This is the logarithm the chapter mentions, and the weighted average removes it.

**The strongly convex lower bound.** The plan is to build a family of hard datasets and an estimator, and then to apply the fingerprinting bound.

1. **A loss whose excess risk is a distance.** Take the linear loss $\ell(\theta; x) = -\langle \theta, x\rangle$ on records with $\|x\| \le L$, so every loss gradient, $-x$, has length at most $L$. With the ridge, completing the square gives
$$\mathcal L(\theta) = \frac\lambda2\|\theta\|^2 - \langle \theta, \bar x\rangle = \frac\lambda2\Big\|\theta - \frac{\bar x}\lambda\Big\|^2 - \frac{\|\bar x\|^2}{2\lambda},$$
where $\bar x$ is the mean of the records. So $\theta^* = \bar x/\lambda$, and the excess risk of any $\theta$ is exactly $\frac\lambda2\|\theta - \theta^*\|^2$.
2. **Hard datasets.** The fingerprinting bound uses random records $y_i \in \{-1, +1\}^d$ with random attribute means, as in @sec-lower-fingerprinting, so the hard datasets are built from them. Fix an integer $m$ between 1 and $n$. The dataset has $m$ real records $x_i = (L/\sqrt d)\,y_i$, each of length $L$, and $n - m$ records equal to zero. The zeros add nothing to the sum, so the mean of all $n$ records is $\bar x = (m/n)(L/\sqrt d)\,\bar y$, where $\bar y$ is the mean of the $y_i$.
3. **From a trainer to a mean estimator.** Given the $m$ records $y_i$, build the dataset of step 2 and run the private trainer to get $\tilde\theta$. The estimator $M$ rescales $\tilde\theta$ so that $\theta^*$ maps to $\bar y$, then clips each coordinate to $[-1, 1]$:
$$M = \operatorname{clip}_{[-1, 1]^d}\Big(\frac{n\lambda\sqrt d}{mL}\,\tilde\theta\Big).$$
Replacing one $y_i$ replaces one record of the padded dataset, so $M$ is $(\varepsilon, \delta)$-DP by post-processing. Clipping never moves a coordinate away from $\bar y$, because $\bar y$ lies in $[-1, 1]^d$. And by the Cauchy–Schwarz inequality the $\ell_1$ norm is at most $\sqrt d$ times the Euclidean norm. So
$$\|M - \bar y\|_1 \le \sqrt d\cdot\frac{n\lambda\sqrt d}{mL}\|\tilde\theta - \theta^*\| = \frac{dn\lambda}{mL}\|\tilde\theta - \theta^*\|.$$
4. **Average error.** Write $E$ for the trainer's expected excess risk on these datasets. By step 1, $\mathbb E\|\tilde\theta - \theta^*\|^2 = 2E/\lambda$, and the average of a length is at most the square root of the average squared length. So the average error per attribute, $a = \mathbb E\|M - \bar y\|_1/d$, satisfies
$$a \le \frac nm\cdot\frac{\sqrt{2\lambda E}}{L}.$$
5. **The fingerprinting bound.** @eq-lower-fp says that an $(\varepsilon, \delta)$-DP estimator of $d$ attribute means from $m$ records, with outputs in $[-1, 1]^d$ and average error $a$ per attribute, needs
$$m \ge \Big(\frac23 - 2a\Big)K.$$
Its proof plays two facts against each other. An accurate estimate must correlate with the records by a total of order $d$. DP lets each record carry only about $(e^\varepsilon - 1)\sqrt d + 4\delta d$ of that total. Rearranging the inequality gives $a \ge \frac13 - \frac{m}{2K}$, and step 4 turns it into
$$\frac{\sqrt{2\lambda E}}{L} \ge \frac mn\Big(\frac13 - \frac{m}{2K}\Big).$$
6. **Choose $m$.** This holds for every $m$, so it holds for the best one. When the right side is negative it says nothing, so only values of $m$ below $2K/3$ matter, where both sides are non-negative. Squaring and multiplying by $L^2/(2\lambda)$ gives the first form of @eq-apx-learn-lower. The bracket is largest near $m = K/3$. If $K \ge 3$, the integer $m = \lfloor K/3\rfloor$ lies between $K/6$ and $K/3$, so $\frac mn(\frac13 - \frac m{2K}) \ge \frac{K}{6n}\cdot\frac16 = \frac{K}{36n}$. Squaring and multiplying by $L^2/(2\lambda)$ gives $L^2K^2/(2592\lambda n^2)$. The condition $K \le 3n$ keeps this $m$ at most $n$. The trainer's expected excess risk on the random datasets is at least this, so some dataset has at least this much.

**The convex lower bound.** Without a ridge in the objective, the same idea works if the ridge is moved into the loss and its size is chosen from $m$.

1. **The loss.** On the ball of radius $R$, take $\ell(\theta; x) = \frac\lambda2\|\theta\|^2 - \langle \theta, x\rangle$ with $\|x\| \le L/2$ and $\lambda = mL/(2nR)$. Its gradient $\lambda \theta - x$ has length at most $\lambda R + L/2 \le L$, because $m \le n$. So it is a convex loss with gradients of length at most $L$ on the ball.
2. **The datasets.** Use $m$ real records $x_i = (L/(2\sqrt d))\,y_i$ and $n - m$ zero records. As before, $\mathcal L(\theta) = \frac\lambda2\|\theta - \bar x/\lambda\|^2$ plus a constant, and now $\bar x/\lambda = (R/\sqrt d)\,\bar y$. That point has length at most $R$, so it is the minimizer over the ball, and the excess risk is again $\frac\lambda2\|\theta - \theta^*\|^2$.
3. **The estimator.** Output $M = \operatorname{clip}_{[-1, 1]^d}((\sqrt d/R)\,\tilde\theta)$. As in steps 3 and 4 above, $a \le \sqrt{2E/\lambda}/R = \sqrt{4nE/(mLR)}$.
4. **Fingerprinting.** The condition $m \ge (\frac23 - 2a)K$ rearranges to $2a \ge \frac23 - \frac mK$, so
$$E \ge \frac{mLR}{16n}\Big(\frac23 - \frac mK\Big)^2,$$
which is @eq-apx-learn-lower-convex before the choice of $m$.
5. **Choose $m$.** The right side is largest at $m = 2K/9$. If $K \ge 9/2$, the integer $m = \lfloor 2K/9\rfloor$ lies between $K/9$ and $2K/9$, so $\frac23 - \frac mK \ge \frac49$ and the bound is at least $\frac{LR}{16n}\cdot\frac K9\cdot\frac{16}{81} = \frac{LRK}{729n}$. The condition $K \le \frac92n$ keeps $m \le n$.

**The exact bounds.** Two inequalities from [Optimization](../part01-prerequisites/06-optimization.md) give all four bounds. A $\beta_s$-smooth function lies below its tangent plus $\frac{\beta_s}2$ times the squared distance, by the descent lemma @eq-descent-lemma. A $\lambda$-strongly convex function lies above its tangent plus $\frac\lambda2$ times the squared distance. Here $\beta_s$ must bound the curvature of $\mathcal L$ everywhere, not only at $\theta^*$. For the loss $g(y\langle \theta, x\rangle)$ with $0 \le g'' \le c$ and $\|x\| \le 1$, the Hessian is at most $(c + \lambda)I$.

1. **Output perturbation.** At $\theta^*$ the gradient is zero, so the tangent is flat. The two inequalities give $\frac\lambda2\|Z\|^2 \le \mathcal L(\theta^* + Z) - \mathcal L(\theta^*) \le \frac{\beta_s}2\|Z\|^2$ for every noise vector $Z$. Averaging gives the first pair of @eq-apx-learn-sandwich. For Gaussian noise, $\mathbb E\|Z\|^2 = d\sigma^2$.
2. **Objective perturbation, from above.** The perturbed objective $\mathcal L_b(\theta) = \mathcal L(\theta) + \frac1n\langle b, \theta\rangle$ is $\lambda$-strongly convex and minimized at $\tilde\theta$. Strong convexity at its minimizer, where the gradient is zero, gives $\mathcal L_b(\theta^*) \ge \mathcal L_b(\tilde\theta) + \frac\lambda2\|\theta^* - \tilde\theta\|^2$. Writing out $\mathcal L_b$ and moving terms,
$$\mathcal L(\tilde\theta) - \mathcal L(\theta^*) \le \frac1n\langle b, \theta^* - \tilde\theta\rangle - \frac\lambda2\|\tilde\theta - \theta^*\|^2 \le \frac{\|b\|}{n}t - \frac\lambda2t^2, \qquad t = \|\tilde\theta - \theta^*\|,$$
by the Cauchy–Schwarz inequality. The right side is a downward parabola in $t$ whose largest value is $\|b\|^2/(2\lambda n^2)$. This holds for every draw of $b$, so averaging over $b$ gives the upper bound of @eq-apx-learn-sandwich.
3. **Objective perturbation, from below.** The gradient of $\mathcal L_b$ is zero at its minimizer $\tilde\theta$, so the gradient of $\mathcal L$ is $-b/n$ there, and it is zero at $\theta^*$. A $\beta_s$-smooth function's gradient changes by at most $\beta_s$ per unit of distance, so $\|b\|/n \le \beta_s t$. Strong convexity of $\mathcal L$ at $\theta^*$ gives $\mathcal L(\tilde\theta) - \mathcal L(\theta^*) \ge \frac\lambda2t^2 \ge \frac{\lambda\|b\|^2}{2\beta_s^2n^2}$. Averaging over $b$ gives the lower bound.
4. **Compare with the chapter.** The second-order formulas average $\frac12Z^\top HZ$ and $\frac1{2n^2}b^\top H^{-1}b$, where $H$ is the Hessian at $\theta^*$. Every eigenvalue of $H$ lies between $\lambda$ and $\beta_s$. So the exact upper bounds are the chapter's formulas with $H$ replaced by its worst case, $\beta_s I$ for output perturbation and $\lambda I$ for objective perturbation.

Step 1 also shows what output perturbation achieves. The minimizer moves by at most $2L/(\lambda n)$ when one record is replaced (Fact 2 of the chapter), so the Gaussian noise has standard deviation $2L/(\lambda n\mu)$, and the upper bound is $2\beta_s dL^2/(\lambda^2n^2\mu^2)$. This is the rate of @eq-apx-learn-upper times the condition number $\beta_s/\lambda$, so output perturbation reaches the optimal rate only when $\beta_s/\lambda$ is bounded by a constant. Objective perturbation with the chapter's pure-DP noise, whose density is proportional to $e^{-\varepsilon'\|b\|/2}$ with $\varepsilon'$ the part of the budget left for the noise, has $\mathbb E\|b\|^2 = d(d + 1)(2/\varepsilon')^2$, which grows like $d^2$, the price of $\delta = 0$.

### Checking it

The script computes the bounds on the chapter's running example. It is logistic regression in $d = 10$ dimensions with ridge $\lambda = 0.01$ and $L = 1$. The budget is $\varepsilon = 1$ and $\delta = 10^{-5}$, so $\mu = 0.2681$. At $n = 10{,}000$ the rule $T = \lceil n^2\mu^2/d\rceil$ asks for 718,515 steps, with noise $\sigma = 0.6325$ on the mean gradient. The upper bound @eq-apx-learn-upper is $2.2268 \times 10^{-3}$. The uniform average's bound is $8.0638 \times 10^{-3}$, larger by about $(1 + \ln T)/4$, the ratio of the two closed forms. That many full-batch steps are too slow for the script, so it runs the algorithm at smaller $n$, on two freshly drawn datasets each. With 300 records the measured excess risk is $2.2640 \times 10^{-1}$, and the bound is 10.9 times larger. The ratio grows to 16.5 at 1,000 records, where the measured value is $1.3489 \times 10^{-2}$, and to 23.0 at 3,000 records, where it is $1.0754 \times 10^{-3}$. It is loose for the reason the chapter gives: the logistic loss has more curvature than $\lambda$ near its minimizer, which the bound does not use.

The lower bound is far below. The running example has $K = 1.8402$, which is below 3, so only the first form of @eq-apx-learn-lower applies, at $m = 1$. At $n = 10{,}000$ it gives $1.8991 \times 10^{-9}$, a factor $1.17 \times 10^6$ below the upper bound. In high dimension the factor settles near $4.3 \times 10^5$: at $n = 10{,}000$ it is $4.27 \times 10^5$ for $d = 1{,}000$ and $4.28 \times 10^5$ for $d = 10{,}000$. The convex bound @eq-apx-learn-lower-convex, with $R = 5$, is $8.6 \times 10^3$ times below the chapter's upper bound @eq-dperm-convex-rate at both dimensions. These constants are as poor as those of the fingerprinting bound itself, which the lower-bounds chapter already calls loose. What the proof establishes is the shape: both bounds fall like $1/n^2$ and grow like $d$, as @fig-apx-learn-rates shows.

![Left: excess risk against the number of records on the running example ($d = 10$, $\lambda = 0.01$, $(1, 10^{-5})$-DP), log–log. The upper bound @eq-apx-learn-upper (solid), noisy projected gradient descent measured on two datasets per point (circles), and the lower bound @eq-apx-learn-lower for every private algorithm (dashed). Right: the two bounds against the dimension at $n = 10{,}000$. Both fall like $1/n^2$ and grow like $d$; the constants are far apart.](figures/learn_erm_rates.svg){#fig-apx-learn-rates}

The exact bounds bracket the measurements. At $n = 10{,}000$, output perturbation with Gaussian noise has the bounds $2.7835 \times 10^{-4}$ and $7.2372 \times 10^{-3}$ from @eq-apx-learn-sandwich. The chapter's second-order value is $8.7070 \times 10^{-4}$, and 20 releases average $9.6417 \times 10^{-4}$. Objective perturbation has the bounds $3.2872 \times 10^{-7}$ and $2.2221 \times 10^{-4}$, the second-order value $7.1266 \times 10^{-5}$, and the measured $7.1178 \times 10^{-5}$. On 50 draws of the noise, every single release also obeys the per-draw inequalities of steps 2 and 3. The upper bounds are 7.5 and 3.1 times the measurements. The lower bounds are much weaker, because $\lambda = 0.01$ is far below the data's curvature. @fig-apx-learn-sandwich shows why: along any direction, the objective lies between the two parabolas, and close to its second-order approximation.

![The running example's objective at $n = 10{,}000$ along a random direction $u$ through its minimizer (black), between the parabolas of strong convexity, $\frac\lambda2t^2$, and smoothness, $\frac{\beta_s}2t^2$ (dashed). The second-order approximation $\frac{h_u}2t^2$, with $h_u$ the curvature along $u$, is dotted.](figures/learn_erm_sandwich.svg){#fig-apx-learn-sandwich}

### What it does not cover

The lower bound loses large constants in three places: the fingerprinting bound itself, the factor $\sqrt d$ between the $\ell_1$ and Euclidean errors, and the choice of $m$. It also loses a factor $\sqrt{\ln(1/\delta)}$, which Steinke & Ullman (2016) recover with a sharper distribution of the attribute means. It needs $\delta$ small compared with $1/\sqrt d$, so that $K$ is of order $\sqrt d/(e^\varepsilon - 1)$, and it holds for replace neighbours. The upper bound uses full gradients, so it costs about $n^3\mu^2/d$ gradient evaluations. Bassily, Smith & Thakurta (2014) use one random record per step and subsampling, which is much cheaper and adds logarithmic factors; that analysis stays cited.

## The data-dependent bound for a noisy vote {#sec-apx-learn-pate}

### What the chapter uses

[Private prediction](../part05-dp-ml/05-private-prediction.md) answers a query with GNMax: each of many teachers votes for a label, Gaussian noise of standard deviation $\sigma$ is added to each label's count, and the label with the highest noisy count is released. This is the vote of **private aggregation of teacher ensembles (PATE)**. GNMax satisfies **Rényi differential privacy (RDP)** with cost $\alpha/\sigma^2$ at every order $\alpha$. When the teachers agree, the answer is almost certain, and @sec-pate-data-dependent charges it less with the bound @eq-pate-dd-gauss of Papernot et al. (2018, Theorem 6).

Write $P = M(D)$ and $P' = M(D')$ for the distributions of the answer on the dataset and on a neighbour. Let $i^*$ be the likely answer, and let $\tilde\beta$ be a known bound on the probability $\beta = P(\text{not } i^*)$ that the answer is something else, a failure probability (Papernot et al. write it $q$). The bound uses two auxiliary orders, $\alpha_2 = \sigma\sqrt{\ln(1/\tilde\beta)}$ and $\alpha_1 = \alpha_2 + 1$, with the RDP costs of GNMax at those orders, $\varepsilon_1 = \alpha_1/\sigma^2$ and $\varepsilon_2 = \alpha_2/\sigma^2$. Theorem 6 requires $\alpha \le \alpha_1$, $\alpha_2 > 1$, and the condition
$$\tilde\beta \le \frac{e^{(\alpha_2 - 1)\varepsilon_2}}{\Big(\frac{\alpha_1}{\alpha_1 - 1}\cdot\frac{\alpha_2}{\alpha_2 - 1}\Big)^{\alpha_2}}.$$
Under these, the order-$\alpha$ divergence of the answer on the dataset from the answer on any neighbour satisfies
$$D_\alpha\big(P\|P'\big) \le \frac{1}{\alpha - 1}\ln\Big[(1 - \tilde\beta)A^{\alpha - 1} + \tilde\beta B^{\alpha - 1}\Big],\quad A = \frac{1 - \tilde\beta}{1 - (\tilde\beta e^{\varepsilon_2})^{(\alpha_2 - 1)/\alpha_2}},\quad B = \frac{e^{\varepsilon_1}}{\tilde\beta^{1/(\alpha_1 - 1)}}.$$
Here $D_\alpha$ is the Rényi divergence of order $\alpha$, defined in @eq-pate-renyi by
$$e^{(\alpha - 1)D_\alpha(P\|P')} = \sum_yP(y)^\alpha P'(y)^{1 - \alpha}.$$
The chapter derives the simpler pure-DP form @eq-pate-dd-pure and says of the Gaussian form that "its proof is in Papernot et al. (2018, Appendix A)". In step 4 of its derivation, it also uses that the pure-DP bound increases in $\beta$, which step 7 below proves.

This section proves the Gaussian bound with the true $\beta$ in place of $\tilde\beta$. It then shows that the largest value over all $\beta \le \tilde\beta$ is a valid bound with no condition at all, and checks that this largest value is the value at $\tilde\beta$ in the chapter's setting. It also proves that the pure-DP bound always increases in $\beta$, which closes the gap in step 4.

### The idea

Split the Rényi sum into the likely answer and all the others. On the likely answer, the two distributions are both close to 1, so the term is close to 1. It can only be large if the neighbour puts much more probability off $i^*$ than $P$ does. A Rényi guarantee at a high order $\alpha_2$, applied to the single event "not $i^*$", forbids that. On the other answers, $P$ has total mass only $\beta$. Their likelihood ratios could be large, but a Rényi guarantee at order $\alpha_1 \ge \alpha$ controls their average, and the average of a smaller power is controlled by the average of a larger one.

A small example shows the split. Take three answers with $P = (0.99, 0.008, 0.002)$ and $P' = (0.985, 0.01, 0.005)$, and order $\alpha = 2$. The Rényi sum is $0.99^2/0.985 + 0.008^2/0.01 + 0.002^2/0.005 = 0.99503 + 0.0064 + 0.0008 = 1.00223$. The first term is near $1 - \beta = 0.99$, inflated a little because $P'$ has slightly less mass on $i^*$. The other two are $\beta = 0.01$ times an average ratio below 1. The bound treats the two parts separately and replaces each by its worst case.

### Derivation

The orders $\alpha_1$ and $\alpha_2$ are fixed numbers in steps 1 to 4, chosen before $\beta$ enters. The bound holds for any order $\alpha > 1$ and any orders with $\alpha \le \alpha_1$, $\alpha_2 > 1$ and $\beta e^{\varepsilon_2} < 1$. The last one keeps the denominator of $A$ positive.

1. **Split the sum.** Write $\beta' = P'(\text{not } i^*)$. Then
$$e^{(\alpha - 1)D_\alpha(P\|P')} = (1 - \beta)^\alpha(1 - \beta')^{1 - \alpha} + \sum_{y \ne i^*}P(y)^\alpha P'(y)^{1 - \alpha}.$$
2. **The neighbour cannot move much mass off $i^*$.** GNMax is $(\alpha_2, \varepsilon_2)$-RDP for every pair of neighbours, in either order, because the neighbour relation is symmetric. So $D_{\alpha_2}(P'\|P) \le \varepsilon_2$. Post-processing never increases a Rényi divergence, and reducing an answer to "is it $i^*$?" is post-processing. The reduced distributions are $(1 - \beta', \beta')$ and $(1 - \beta, \beta)$, and their Rényi sum is at least its "not $i^*$" term:
$$\beta'^{\alpha_2}\beta^{1 - \alpha_2} \le e^{(\alpha_2 - 1)\varepsilon_2}, \qquad\text{so}\qquad \beta' \le (\beta e^{\varepsilon_2})^{(\alpha_2 - 1)/\alpha_2}.$$
3. **The likely answer's term.** The exponent $1 - \alpha$ is negative, so a smaller base gives a larger power. Step 2 bounds $1 - \beta'$ from below, so
$$(1 - \beta)^\alpha(1 - \beta')^{1 - \alpha} \le (1 - \beta)^\alpha\big(1 - (\beta e^{\varepsilon_2})^{(\alpha_2 - 1)/\alpha_2}\big)^{1 - \alpha} = (1 - \beta)A^{\alpha - 1},$$
with $A$ as in the statement, $\beta$ in place of $\tilde\beta$.
4. **The other answers.** Write their sum as $\beta$ times an average under $P$ given "not $i^*$". With the likelihood ratio $r(y) = P(y)/P'(y)$,
$$\sum_{y\ne i^*}P(y)^\alpha P'(y)^{1 - \alpha} = \beta\sum_{y \ne i^*}\frac{P(y)}\beta\,r(y)^{\alpha - 1}.$$
Now set $s = (\alpha - 1)/(\alpha_1 - 1)$, which lies in $(0, 1]$ because $\alpha \le \alpha_1$. Then $r^{\alpha - 1} = (r^{\alpha_1 - 1})^s$. The function $t \mapsto t^s$ is concave, so by Jensen's inequality the average of $X^s$ is at most $(\text{average of }X)^s$. For example, with $s = \frac12$, the average of $\sqrt1$ and $\sqrt9$ is 2, below $\sqrt5$. The average of $r^{\alpha_1 - 1}$ given "not $i^*$" is $\frac1\beta\sum_{y \ne i^*}P(y)^{\alpha_1}P'(y)^{1 - \alpha_1}$. That sum is part of the full Rényi sum of order $\alpha_1$, which RDP bounds by $e^{(\alpha_1 - 1)\varepsilon_1}$. So
$$\sum_{y\ne i^*}P(y)^\alpha P'(y)^{1 - \alpha} \le \beta\Big(\frac{e^{(\alpha_1 - 1)\varepsilon_1}}\beta\Big)^{s} = \beta\Big(\frac{e^{\varepsilon_1}}{\beta^{1/(\alpha_1 - 1)}}\Big)^{\alpha - 1} = \beta B^{\alpha - 1}.$$
5. **Collect.** Adding steps 3 and 4 and taking $\frac1{\alpha - 1}\ln$ gives the bound with the true $\beta$:
$$D_\alpha(P\|P') \le V(\beta) = \frac{1}{\alpha - 1}\ln\Big[(1 - \beta)A(\beta)^{\alpha - 1} + \beta B(\beta)^{\alpha - 1}\Big].$$ {#eq-apx-learn-pate-q}
6. **From $\beta$ to $\tilde\beta$.** Only the bound $\tilde\beta \ge \beta$ is known, so take the largest value of $V$ over all $\beta$ in $(0, \tilde\beta]$. That is a valid bound, with no condition beyond those listed before step 1. When $V$ increases in $\beta$, the largest value is $V(\tilde\beta)$, which is @eq-pate-dd-gauss. The condition of Theorem 6 is what Papernot et al. use to prove that $V$ increases. The script checks it directly instead: at the chapter's $\sigma = 40$ and $\alpha = 8$, $V$ increases on $(0, \tilde\beta]$ at every vote gap (the difference between the top two counts) where the condition holds.
7. **The pure-DP form increases in $\beta$.** For a mechanism that is pure DP with parameter $\varepsilon_0$, write $\kappa = e^{\varepsilon_0}$ for its largest ratio of probabilities. The same split gives the bracket of @eq-pate-dd-pure,
$$f(\beta) = (1 - \beta)^\alpha(1 - \kappa\beta)^{1 - \alpha} + \beta\,\kappa^{\alpha - 1},$$
on $0 \le \beta < 1/\kappa$. Its first term has the form $y\,\psi(x/y)$, with the power function $\psi(t) = t^\alpha$, $x = 1 - \beta$ and $y = 1 - \kappa\beta$. That is the **perspective** of the convex function $\psi$, and a perspective is jointly convex in $(x, y)$ (@sec-apx-tail-perspective).

   The reason is short. A convex function is the largest of its tangent lines, so $\psi(t) = \max(at + c)$ over the slopes $a$ and intercepts $c$ of its tangents. Then $y\psi(x/y)$ is the largest of the linear functions $ax + cy$, and a maximum of linear functions is convex. Here $x$ and $y$ are linear in $\beta$, so the first term is convex in $\beta$, and the second term is linear. So $f$ is convex, and its slope only grows. At $\beta = 0$ the slope is
$$f'(0) = \kappa^{\alpha - 1} + (\alpha - 1)\kappa - \alpha,$$
which is 0 at $\kappa = 1$ and grows with $\kappa$. So $f' \ge 0$ on the whole range, and the bound is largest at $\beta = \tilde\beta$, in every case.
8. **Why $\alpha_2 = \sigma\sqrt{\ln(1/\tilde\beta)}$.** The bound is small when $A$ is close to 1, that is when step 2's bound on $\beta'$ is close to $\tilde\beta$. With $\varepsilon_2 = \alpha_2/\sigma^2$, the logarithm of that bound is
$$\Big(1 - \frac1{\alpha_2}\Big)\Big(\ln\tilde\beta + \frac{\alpha_2}{\sigma^2}\Big) = \ln\tilde\beta + \frac{\alpha_2}{\sigma^2} + \frac{\ln(1/\tilde\beta)}{\alpha_2} - \frac1{\sigma^2}.$$
The middle two terms trade a larger order against a larger cost, and their sum is smallest at $\alpha_2 = \sigma\sqrt{\ln(1/\tilde\beta)}$, where it equals $2\sqrt{\ln(1/\tilde\beta)}/\sigma$. So this choice makes the neighbour's failure probability as close to $\tilde\beta$ as the argument allows: at most $\tilde\beta\,e^{2\sqrt{\ln(1/\tilde\beta)}/\sigma - 1/\sigma^2}$. The choice $\alpha_1 = \alpha_2 + 1$ is Papernot et al.'s, and any $\alpha_1 \ge \alpha$ gives a valid bound.

### Checking it

Take the chapter's example: two labels, $\sigma = 40$, order $\alpha = 8$ and a gap of 150 votes. On two labels the difference of the two noises is normal with variance $2\sigma^2$, so the answer is wrong with probability $\tilde\beta = \beta = \frac12\operatorname{erfc}(150/80) = 4.0050 \times 10^{-3}$. The orders are $\alpha_2 = 93.981$ and $\alpha_1 = 94.981$. Their costs are $\varepsilon_2 = 0.05874$ and $\varepsilon_1 = 0.05936$. Step 2 allows the neighbour a failure probability of at most $4.5014 \times 10^{-3}$. That is 1.1240 times $\tilde\beta$, which is the factor $e^{2\sqrt{\ln(1/\tilde\beta)}/\sigma - 1/\sigma^2}$ of step 8. The likely answer's term is $0.999477$ and the other answers' term is $9.1545 \times 10^{-3}$. The bound is $1.2278 \times 10^{-3}$, the value the chapter quotes, and the largest value over $\beta \le \tilde\beta$ is the same. One teacher that changes its vote moves the gap by 2, so the neighbours have gaps of 148 and 152. Against the worse of the two, the exact divergence in this direction is $2.2362 \times 10^{-4}$, 5.49 times smaller.

Over gaps from 0 to 300 in steps of 5, the condition holds from a gap of 20 on, and at all 57 such gaps $V$ increases on $(0, \tilde\beta]$ and the exact two-label divergence lies below the bound. At $\sigma = 40$ the condition allows any $\tilde\beta$ up to $0.36321$. With three labels and votes $(200, 40, 10)$, the union bound of @eq-pate-qtilde gives $\tilde\beta = 2.7303 \times 10^{-3}$ and the bound $8.7544 \times 10^{-4}$. The exact worst divergence over the six ways one teacher can move its vote is $1.5188 \times 10^{-4}$, and the data-independent cost is $5.0000 \times 10^{-3}$. The script also checks step 7 numerically: on a grid of orders and values of $\varepsilon_0$, the pure-DP bracket is convex and increasing in $\beta$.

@fig-apx-learn-pate-terms splits the bound into its two parts. Below a gap of 116 the bound exceeds $\alpha/\sigma^2$ and is not used. Above it, the other answers' term is the larger one: the cost of an answer is mostly the chance that the noise overturns the vote, weighted by how much an overturned vote reveals. @fig-apx-learn-pate-q shows the two facts behind step 6. With the orders fixed, $V$ rises steadily up to $\tilde\beta$. And the largest value over $\beta \le \tilde\beta$, as a function of $\tilde\beta$, falls far below the data-independent cost as the vote becomes certain.

![The data-dependent cost of one GNMax answer at order 8 with $\sigma = 40$, against the gap between the two vote counts, log scale. The bound (red) is the logarithm of 1 plus the two dashed parts, divided by $\alpha - 1$: the likely answer's term minus $1 - \tilde\beta$, and the other answers' term minus $\tilde\beta$. Black: the data-independent cost $\alpha/\sigma^2$. Dotted: the exact cost on two labels.](figures/learn_pate_terms.svg){#fig-apx-learn-pate-terms}

![Left: the bound @eq-apx-learn-pate-q as a function of the true failure probability $\beta$, with the orders set for gaps of 150 and 200 votes; it increases up to $\tilde\beta$ (dotted). Right: the largest value over $\beta \le \tilde\beta$ against $\tilde\beta$, at order 8 and $\sigma = 40$, with the data-independent cost and the largest $\tilde\beta$ the condition of Theorem 6 allows.](figures/learn_pate_q.svg){#fig-apx-learn-pate-q}

### What it does not cover

The proof that the condition of Theorem 6 makes $V$ increasing stays cited to Papernot et al. (2018, Appendix A). Taking the largest value over $\beta \le \tilde\beta$ is a valid replacement that needs no condition, and it costs only a one-dimensional search. The union bound $\tilde\beta$ of @eq-pate-qtilde for many labels is the paper's Proposition 7 and is not derived here. The bound is for one direction, $D_\alpha(M(D)\|M(D'))$ with $D$ the actual dataset, which is the direction the chapter's accounting uses.

## The best of a random number of runs {#sec-apx-learn-tuning}

### What the chapter uses

[Hyperparameter tuning](../part05-dp-ml/03-hyperparameter-tuning.md) releases the best of $K$ independent training runs, with $K$ drawn at random before the search. Each run is an algorithm $Q$ with RDP curve $\varepsilon(\alpha)$ at order $\alpha$. The number of runs $K$ follows a **truncated negative binomial** distribution with parameters $\nu > -1$ and $\omega \in (0, 1)$ (Papernot & Steinke write $\eta$ for $\nu$, $\gamma$ for $\omega$ and $\lambda$ for the order), whose generating function is $f(x) = \mathbb E[x^K]$. Its mean is $\mathbb E[K]$, and the case $\nu = 0$ is the logarithmic distribution. @sec-hpt-rdp states Papernot & Steinke's (2022) Theorem 2 as @eq-hpt-tnb: for every $\hat\alpha \ge 1$, the best-of release $M$ satisfies
$$\varepsilon_M(\alpha) \le \varepsilon(\alpha) + (1 + \nu)\Big(1 - \frac1{\hat\alpha}\Big)\varepsilon(\hat\alpha) + \frac{(1 + \nu)\ln(1/\omega)}{\hat\alpha} + \frac{\ln\mathbb E[K]}{\alpha - 1}.$$

The chapter derives the generic bound @eq-hpt-generic. For neighbours with run distributions $Q, Q'$,
$$D_\alpha\big(M(D)\|M(D')\big) \le D_\alpha(Q\|Q') + \frac1{\alpha - 1}\ln\Big[f'(p)^\alpha f'(p')^{1 - \alpha}\Big],$$
where $p$ and $p'$ are the probabilities that one and the same randomized test accepts a run's output, on $D$ and on $D'$. The chapter takes outputs to be finite and $\Pr[K = 0] = 0$. Its step 9 then says that the truncated negative binomial case follows by bounding the rejection probabilities through the run's RDP at a second order $\hat\alpha$, and that "Papernot & Steinke (2022, Appendix B.2) carry out the algebra". The chapter also quotes their Corollary 3, $(2 + \nu)\varepsilon$-DP for an $\varepsilon$-DP run, and Corollary 4 for zero-concentrated DP runs. This section carries out that algebra and derives both corollaries.

### The idea

For this distribution, $f'(x)$ is a power of one simple quantity, $a = \omega + (1 - \omega)(1 - x)$. Here $1 - x$ is the probability that the test rejects, and $\omega$ is a floor that comes from the distribution of $K$, not from the data. So the worst-case term of the generic bound is a power of $\omega a'^{\alpha - 1}/a^\alpha$, with $a'$ the same quantity on $D'$. That ratio is large only if $a'$ is much larger than $a$.

Two facts stop that. The run's RDP guarantee says that the neighbour cannot reject much more often than $D$ does: in Rényi terms the rejection probability on $D'$ is at most a power of that on $D$. And $a$ is never below $\omega$, so dividing by it costs at most a power of $1/\omega$. The floor $\omega$ is the "few runs" explanation of the chapter: it caps how much the event "every run was rejected" can say about the data.

### Derivation

The steps continue the chapter's derivation, whose steps 1 to 5 give @eq-hpt-generic. Write $r = 1 - p$ and $r' = 1 - p'$ for the probabilities that the test rejects a run's output, on $D$ and $D'$.

1. **The generating function.** The weights of the distribution are $\Pr[K = k] = (1 - \omega)^k\binom{k + \nu - 1}{k}/(\omega^{-\nu} - 1)$ for $k \ge 1$, because $\prod_{j = 0}^{k - 1}\frac{j + \nu}{j + 1} = \binom{k + \nu - 1}k$. The negative binomial series $\sum_{k \ge 0}\binom{k + \nu - 1}kz^k = (1 - z)^{-\nu}$, with the $k = 0$ term removed, gives
$$f(x) = \frac{(1 - (1 - \omega)x)^{-\nu} - 1}{\omega^{-\nu} - 1}, \qquad f'(x) = \mathbb E[K]\,\omega^{\nu + 1}\,a(x)^{-(\nu + 1)}, \quad a(x) = 1 - (1 - \omega)x.$$
The mean is the slope at 1, as in the chapter:
$$\mathbb E[K] = f'(1) = \frac{\nu(1 - \omega)}{\omega(1 - \omega^\nu)}.$$
Since $a(1) = \omega$, the formula for $f'$ is consistent at $x = 1$. For $\nu = 0$ the same formulas hold in the limit, where $f$ becomes $\ln(1 - (1 - \omega)x)/\ln\omega$.
2. **The worst-case term.** Write $a = a(p) = \omega + (1 - \omega)r$ and $a' = a(p') = \omega + (1 - \omega)r'$. The constant $\mathbb E[K]\omega^{\nu + 1}$ appears with total power $\alpha + (1 - \alpha) = 1$, so
$$f'(p)^\alpha f'(p')^{1 - \alpha} = \mathbb E[K]\omega^{\nu + 1}\big(a^{-\alpha}a'^{\alpha - 1}\big)^{\nu + 1} = \mathbb E[K]\Big(\frac{\omega\,a'^{\alpha - 1}}{a^\alpha}\Big)^{1 + \nu}.$$
3. **The neighbour cannot reject much more often.** The run is $(\hat\alpha, \hat\varepsilon)$-RDP with $\hat\varepsilon = \varepsilon(\hat\alpha)$, for both orders of every pair of neighbours. The test's verdict, reject or accept, is post-processing of a run's output (Fact 2 of the chapter), so $D_{\hat\alpha}$ between the verdict's distributions on $D'$ and on $D$ is at most $\hat\varepsilon$. Their Rényi sum is at least its "reject" term, so
$$r'^{\hat\alpha}r^{1 - \hat\alpha} \le e^{(\hat\alpha - 1)\hat\varepsilon}, \qquad\text{so}\qquad r' \le e^{s\hat\varepsilon}r^s, \quad s = 1 - \frac1{\hat\alpha}.$$
4. **Carry it to $a'$.** Since $e^{s\hat\varepsilon} \ge 1$, step 3 gives
$$a' \le \omega + (1 - \omega)e^{s\hat\varepsilon}r^s \le e^{s\hat\varepsilon}\big(\omega\cdot 1^s + (1 - \omega)r^s\big) \le e^{s\hat\varepsilon}\big(\omega + (1 - \omega)r\big)^s = e^{s\hat\varepsilon}a^s.$$
The last inequality is Jensen's inequality for the concave function $t \mapsto t^s$, with weights $\omega$ and $1 - \omega$ at the points 1 and $r$. For example, with $s = \frac12$, $\omega = \frac12$ and $r = 0$, the middle bracket is $\frac12$ and the right side is $\sqrt{1/2} = 0.71$.
5. **Bound the ratio.** Substituting step 4,
$$\frac{\omega\,a'^{\alpha - 1}}{a^\alpha} \le e^{(\alpha - 1)s\hat\varepsilon}\,\omega\,a^{s(\alpha - 1) - \alpha}.$$
The exponent of $a$ is $s(\alpha - 1) - \alpha = -1 - (\alpha - 1)/\hat\alpha$, which is negative. So the power is largest where $a$ is smallest, and $a \ge \omega$. Replacing $a$ by $\omega$ gives
$$\frac{\omega\,a'^{\alpha - 1}}{a^\alpha} \le e^{(\alpha - 1)(1 - 1/\hat\alpha)\hat\varepsilon}\,\omega^{-(\alpha - 1)/\hat\alpha}.$$
6. **Assemble.** Raise step 5 to the power $1 + \nu > 0$, which keeps the inequality, and multiply by $\mathbb E[K]$ as in step 2. Taking $\frac1{\alpha - 1}\ln$ gives
$$\frac1{\alpha - 1}\ln\Big[f'(p)^\alpha f'(p')^{1 - \alpha}\Big] \le (1 + \nu)\Big(1 - \frac1{\hat\alpha}\Big)\varepsilon(\hat\alpha) + \frac{(1 + \nu)\ln(1/\omega)}{\hat\alpha} + \frac{\ln\mathbb E[K]}{\alpha - 1}.$$
Adding $D_\alpha(Q\|Q') \le \varepsilon(\alpha)$ in @eq-hpt-generic gives @eq-hpt-tnb. At $\hat\alpha = 1$ step 3 says nothing, but then $s = 0$ and the trivial bound $a' \le 1 = a^0$ gives the same formula, with the term $(1 + \nu)\ln(1/\omega)$.
7. **Corollary 3, pure DP.** An $\varepsilon$-DP run has $\varepsilon(\alpha) \le \varepsilon$ at every order, so the first term is at most $\varepsilon$. Let $\hat\alpha \to \infty$: the second term tends to $(1 + \nu)\varepsilon$ and the third to 0. Then let $\alpha \to \infty$: the last term tends to 0, and the order-$\infty$ divergence is the largest log ratio of probabilities. So the release is $(2 + \nu)\varepsilon$-DP.
8. **Corollary 4, zero-concentrated DP.** A run that satisfies zero-concentrated DP with parameter $\rho$, a number that measures its privacy loss, has the RDP curve $\varepsilon(\alpha) = \rho\alpha$. Then $(1 - 1/\hat\alpha)\varepsilon(\hat\alpha) = \rho(\hat\alpha - 1)$, and the terms with $\hat\alpha$ are $(1 + \nu)\big(\rho\hat\alpha + \ln(1/\omega)/\hat\alpha\big) - (1 + \nu)\rho$. For positive numbers $u$ and $v$, the sum $u\hat\alpha + v/\hat\alpha$ is smallest at $\hat\alpha = \sqrt{v/u}$, where it equals $2\sqrt{uv}$. Here the best second order is
$$\hat\alpha = \sqrt{\frac{\ln(1/\omega)}{\rho}},$$
which is at least 1 when $\rho \le \ln(1/\omega)$. So
$$\varepsilon_M(\alpha) \le \rho\alpha + \frac{\ln\mathbb E[K]}{\alpha - 1} + 2(1 + \nu)\sqrt{\rho\ln(1/\omega)} - (1 + \nu)\rho.$$
By the same rule, applied to $\rho(\alpha - 1) + \ln\mathbb E[K]/(\alpha - 1)$, the first two terms are smallest at the order
$$\alpha^* = 1 + \sqrt{\frac{\ln\mathbb E[K]}{\rho}},$$
where they equal $\rho + 2\sqrt{\rho\ln\mathbb E[K]}$. Adding up gives the chapter's closed form, $2\sqrt{\rho\ln\mathbb E[K]} + 2(1 + \nu)\sqrt{\rho\ln(1/\omega)} - \nu\rho$, at $\alpha^*$. The Rényi divergence grows with the order, so the bound at $\alpha^*$ also holds at every order $\alpha \le \alpha^*$.

### Checking it

The chapter checks @eq-hpt-tnb on randomized response with $\varepsilon = 0.5$ and a logarithmic number of runs with mean 10, for which $\omega = 0.02692$. At order $\alpha = 5$ its terms are $0.3842$ for one run, $0.5756$ for $\ln\mathbb E[K]/(\alpha - 1)$, and, at $\hat\alpha = 100.95$, $0.4904$ and $0.0358$ for the two middle terms. Their sum is $1.4860$, against an exact divergence of $0.3912$. @fig-apx-learn-tuning-terms draws the four terms against $\hat\alpha$.

The figure shows why $\hat\alpha = 100.95$ is not the best choice. Randomized response is pure DP, so as $\hat\alpha$ grows the second term rises to $(1 + \nu)\varepsilon = 0.5$ and the third falls to 0. The sum keeps falling, to its limit $1.4599$, and that limit is the value in the chapter's table. A grid of orders that stops at $\hat\alpha = 100.95$ still gives a valid bound, just slightly above the best one. The same happens at the table's other orders: the limits, which the table gives, are $3.0299$ at order 2 and $1.0962$ at order 20, against $3.0561$ and $1.1224$ at the end of the grid.

![The four terms of @eq-hpt-tnb at order $\alpha = 5$ against the second order $\hat\alpha$, for randomized response with $\varepsilon = 0.5$ and a logarithmic number of runs with mean 10. The sum (black) keeps falling toward its limit (dashed); the dot marks the end of the chapter's grid of orders.](figures/learn_tuning_terms.svg){#fig-apx-learn-tuning-terms}

The script also checks every step of the chain on 20,000 random cases of $\omega$, $\alpha$, $\hat\alpha$, $\hat\varepsilon$ and rejection probabilities that obey step 3, and none fails. For Corollary 4 with $\rho = 0.1$ and a geometric number of runs ($\nu = 1$) with mean 10, so $\omega = 0.1$, the closed form gives $2.7791$. Minimizing the theorem over $\hat\alpha$ numerically at $\alpha^* = 5.7985$ gives the same, with the best $\hat\alpha = 4.7985$.

@fig-apx-learn-tuning-square shows where the slack is. The shaded region is every pair of rejection probabilities that order-4 RDP allows, and the red curve is the looser constraint that step 3 keeps. On the shaded region the ratio $\omega a'^{\alpha - 1}/a^\alpha$ is at most $7.67$, while step 5 bounds it by $89.63$. Most of the gap comes from replacing $a$ by $\omega$, which is exact only when the test rejects nothing on $D$.

![The unit square of rejection probabilities $(r, r')$ on $D$ and $D'$. Shaded: the pairs that order-$\hat\alpha$ RDP with $\hat\alpha = 4$ and $\hat\varepsilon = 0.5$ allows. Red: the bound $r' \le e^{(1 - 1/\hat\alpha)\hat\varepsilon}r^{1 - 1/\hat\alpha}$ of step 3. Grey: levels of $\omega a'^{\alpha - 1}/a^\alpha$ at $\alpha = 5$, $\omega = 0.05$, with its largest value on the shaded region marked.](figures/learn_tuning_square.svg){#fig-apx-learn-tuning-square}

### What it does not cover

The derivation keeps the chapter's simplification to finite outputs and $\Pr[K = 0] = 0$. Papernot & Steinke (2022) handle the general case, where an empty search releases a fixed output, and that stays cited.

## A user with many records in record-level DP-SGD {#sec-apx-learn-user}

### What the chapter uses

[User-level differential privacy](../part05-dp-ml/07-user-level.md) asks what DP-SGD, the private SGD of the earlier chapters, guarantees to a user who owns $k$ records. DP-SGD samples each record into a batch independently with probability $q$, clips each record's gradient to length at most $C$, and adds Gaussian noise of standard deviation $\sigma C$ to the sum. In units of $C$, @sec-ulvl-group-sgd compares the output without the user, $P_0 = \mathcal N(0, \sigma^2)$, with the output with the user, the mixture @eq-ulvl-mixture:
$$P_{(k)} = \sum_{m=0}^kb_m\,\mathcal N(m, \sigma^2), \qquad b_m = \binom kmq^m(1 - q)^{k - m}.$$
Its **Rényi moment** at integer order $\alpha$ is the average $A_\alpha = \mathbb E_{y\sim P_0}[(P_{(k)}(y)/P_0(y))^\alpha]$, so that $D_\alpha(P_{(k)}\|P_0) = \ln A_\alpha/(\alpha - 1)$. The chapter computes it as @eq-ulvl-pair:
$$A_\alpha = \mathbb E\Big[\exp\Big(\frac1{\sigma^2}\sum_{1 \le i < j \le \alpha}N_iN_j\Big)\Big],$$
where $N_1, \dots, N_\alpha$ are independent copies of the number $N \sim \mathrm{Bin}(k, q)$ of the user's records in a batch. The chapter derives $A_\alpha$ for this pair, in which every drawn record pushes the sum by the full $C$ in the same direction. But it takes from Charles et al. (2024, Theorem 3) that this aligned case is the worst over all configurations of the user's gradients, and gives only the intuition.

This section proves that claim for the divergence $D_\alpha(P\|P_0)$ of the output with the user from the output without, at integer orders. The user's gradients may point in any directions in any number of dimensions. The Rényi moment is never larger than @eq-ulvl-pair, and it equals it when the gradients are aligned.

### The idea

The Rényi moment of order $\alpha$ is an average over $\alpha$ independent draws of the user's subset of records. Each draw shifts the output by the sum $v$ of the drawn gradients. The moment turns out to be the average of the exponential of the sum of inner products $\langle v_i, v_j\rangle$ between different draws. An inner product is at most the product of the lengths, and a sum of $m$ gradients of length at most 1 has length at most $m$. So each inner product is at most the product of the two counts, which is exactly what @eq-ulvl-pair has. Gradients that point the same way reach that maximum.

A user with two records and orthogonal gradients $e_1$ and $e_2$ shows the difference. @tbl-apx-learn-user-pairs lists the inner product of two independent draws for each pair of subsets, against the aligned case where both gradients are $e_1$.

| Draw 1 | Draw 2 | Orthogonal gradients | Aligned gradients |
| --- | --- | --- | --- |
| $\{1\}$ | $\{1\}$ | 1 | 1 |
| $\{1\}$ | $\{2\}$ | 0 | 1 |
| $\{1\}$ | $\{1, 2\}$ | 1 | 2 |
| $\{1, 2\}$ | $\{1, 2\}$ | 2 | 4 |

: The inner product $\langle v_1, v_2\rangle$ of the summed gradients of two independent draws of a two-record user's subset, for orthogonal and for aligned unit gradients. Draws that include no record give 0 in both cases. {#tbl-apx-learn-user-pairs}

Every entry of the orthogonal column is at most the aligned one, and the exponential of a larger number is larger. So the aligned moment is larger.

### Derivation

1. **One step, with the past fixed.** By adaptive composition, it is enough to bound one step for every possible history. Given the history, the model is fixed, so the user's clipped gradients $g_1, \dots, g_k$ are fixed vectors in $\mathbb R^d$ with $\|g_a\| \le 1$ in units of $C$. The user's drawn subset $S$ contains each record independently with probability $q$, and the user adds $v_S = \sum_{a \in S}g_a$ to the sum.
2. **The other records do not matter.** The other records' contribution $U$ is random but has the same distribution with and without the user, and it is independent of $S$. So both outputs are mixtures over $U$ of a pair shifted by $U$. The quantity $e^{(\alpha - 1)D_\alpha(P\|Q)} = \sum_yQ(y)\big(P(y)/Q(y)\big)^\alpha$ is a perspective of $t \mapsto t^\alpha$, so it is jointly convex in the pair (@sec-apx-tail-perspective). Its value at a mixture of pairs is therefore at most the average of its values at the pairs. A shift by $U$ changes nothing, so it suffices to take $U = 0$: compare $P_0 = \mathcal N(0, \sigma^2I)$ with $P = \sum_S\pi(S)\,\mathcal N(v_S, \sigma^2I)$, where $\pi(S) = q^{|S|}(1 - q)^{k - |S|}$.
3. **The likelihood ratio.** Two normal densities with the same covariance $\sigma^2I$ differ by $\|y\|^2 - \|y - v\|^2 = 2\langle v, y\rangle - \|v\|^2$ in the exponent, so
$$\frac{P(y)}{P_0(y)} = \sum_S\pi(S)\exp\Big(\frac{2\langle v_S, y\rangle - \|v_S\|^2}{2\sigma^2}\Big).$$
This is the chapter's @eq-ulvl-ratio, with vectors in place of numbers.
4. **The $\alpha$-th power is an average over independent copies.** As in step 4 of the chapter, multiplying out $\alpha$ copies of the sum gives one term for every choice $(S_1, \dots, S_\alpha)$, with weight $\pi(S_1)\cdots\pi(S_\alpha)$. That is an average over independent subsets. With $u = v_{S_1} + \dots + v_{S_\alpha}$,
$$\Big(\frac{P(y)}{P_0(y)}\Big)^\alpha = \mathbb E\Big[\exp\Big(\frac{2\langle u, y\rangle - \sum_i\|v_{S_i}\|^2}{2\sigma^2}\Big)\Big].$$
5. **Average over $y$ with the Gaussian moment generating function.** Fact 3 of the chapter gives the **moment generating function (MGF)** of a normal variable $Y$ with mean 0 and variance $\sigma^2$. For every real $s$,
$$\mathbb E[e^{sY}] = e^{s^2\sigma^2/2}.$$
Under $P_0$ the coordinates of $y$ are independent normal variables of this kind. So $\mathbb E[e^{\langle u, y\rangle/\sigma^2}]$ is a product of one-dimensional MGFs, one per coordinate $j$, with $s = u_j/\sigma^2$:
$$\mathbb E_{y \sim P_0}\big[e^{\langle u, y\rangle/\sigma^2}\big] = \prod_je^{u_j^2/(2\sigma^2)} = e^{\|u\|^2/(2\sigma^2)}.$$
Everything is positive, so the two averages can be exchanged, and
$$A_\alpha = \mathbb E\Big[\exp\Big(\frac{\|u\|^2 - \sum_i\|v_{S_i}\|^2}{2\sigma^2}\Big)\Big] = \mathbb E\Big[\exp\Big(\frac1{\sigma^2}\sum_{i < j}\langle v_{S_i}, v_{S_j}\rangle\Big)\Big],$$ {#eq-apx-learn-user-moment}
because $\|u\|^2 = \sum_i\|v_{S_i}\|^2 + 2\sum_{i<j}\langle v_{S_i}, v_{S_j}\rangle$ and the squares cancel.
6. **Bound each inner product.** By the Cauchy–Schwarz inequality, $\langle v_{S_i}, v_{S_j}\rangle \le \|v_{S_i}\|\,\|v_{S_j}\|$. By the triangle inequality, $\|v_S\| \le \sum_{a\in S}\|g_a\| \le |S|$. So $\langle v_{S_i}, v_{S_j}\rangle \le |S_i|\,|S_j|$.
7. **The aligned value.** The exponential is increasing and every weight is positive, so step 6 can be applied inside the average of @eq-apx-learn-user-moment. The counts $|S_1|, \dots, |S_\alpha|$ are independent $\mathrm{Bin}(k, q)$ variables, so the bound is @eq-ulvl-pair. Equality holds when every $g_a$ is the same unit vector $e$, because then $v_S = |S|e$ and both inequalities of step 6 are equalities.
8. **Compose.** RDP curves add over steps, even when each step depends on the earlier outputs. So after $T$ steps the user's divergence of this direction is at most $\frac{T}{\alpha - 1}\ln A_\alpha$, with $A_\alpha$ from @eq-ulvl-pair.

### Checking it

The script computes @eq-apx-learn-user-moment exactly, by summing over all subsets, for $k = 5$, $q = 0.05$ and $\sigma = 1$. When the five gradients are aligned, $A_2 = 1.232011$, so the order-2 divergence is $0.208648$, the value of @eq-ulvl-pair. When every pair of gradients meets at 60 degrees, $A_2 = 1.066665$, and at 90 degrees $A_2 = 1.021664$. @fig-apx-learn-user-angle draws the divergence against the common angle, at orders 2 and 3. It is largest at angle 0, where it equals the aligned value. On 200 random configurations of four gradients of random length in three dimensions, no Rényi moment of order 2 or 3 exceeds the aligned one.

The identity of step 5 is checked against a direct integral. For three unit gradients in the plane at angles 0, 60 and 150 degrees, with $q = 0.3$ and $\sigma = 1$, a two-dimensional quadrature of $\mathbb E_{P_0}[(P/P_0)^2]$ gives $1.516754$, and @eq-apx-learn-user-moment gives $1.516754$.

![The Rényi divergence of the output with a user of five records from the output without, as a fraction of its aligned value, when every pair of the user's gradients meets at the same angle ($q = 0.05$, $\sigma = 1$). Orders 2 and 3. Both are largest when the gradients are aligned.](figures/learn_user_angle.svg){#fig-apx-learn-user-angle}

@fig-apx-learn-user-pairs shows which pairs of counts carry @eq-ulvl-pair. Pairs in which either copy draws no record contribute $0.948825$ in total, their probability, because their weight $e^{N_1N_2/\sigma^2}$ is 1. The pair $(1, 1)$ adds $0.112710$, the two mixed pairs $(1, 2)$ and $(2, 1)$ add $0.064501$ together, and $(2, 2)$ adds $0.025084$. Even the pair $(5, 5)$ contributes $7.032 \times 10^{-3}$. Its probability is only $q^{10}$, below $10^{-13}$, but its weight is $e^{25}$. This is why the user's cost grows faster than linearly in $k$ once the higher terms wake up, as the right panel shows.

![Left: the probability of each pair of counts $(m_1, m_2)$ of a user's records in two independent batches ($k = 5$, $q = 0.05$), and its contribution to $A_2$ after weighting by $e^{m_1m_2/\sigma^2}$ with $\sigma = 1$, log scale. Right: the exact order-2 RDP of one step against the number of records $k$, with the small-$kq$ approximation $(kq)^2(e^{1/\sigma^2} - 1)$.](figures/learn_user_pairs.svg){#fig-apx-learn-user-pairs}

### What it does not cover

The argument needs an integer order, because step 4 multiplies out the $\alpha$-th power. For non-integer orders, for the other direction $D_\alpha(P_0\|P)$, and for the privacy loss distribution (PLD) form that the tightest accountants use, the claim that the aligned pair is the worst case stays cited to Charles et al. (2024, Theorem 3). In the other direction the ratio is raised to the negative power $1 - \alpha$, so the expansion of step 4 does not exist.

## How one step of DP-SGD changes a group's loss {#sec-apx-learn-fairness}

### What the chapter uses

[Disparate impact](../part05-dp-ml/08-disparate-impact.md) splits the records into groups $g$ with shares $s_g$, and asks why DP-SGD hurts small groups more. In @sec-fair-noise it describes in words a result of Tran, Dinh & Fioretto (2021). The expected change in a group's loss in one step has a clipping term, driven by the group's gradient norms, and a noise term proportional to $\eta^2\sigma^2C^2$ times the trace of the group's loss Hessian. Here $\eta$ is the learning rate, $\sigma$ the noise multiplier and $C$ the clipping norm. The chapter gives no formula. This section derives the decomposition and shows that it is exact for quadratic losses. It also shows that it contains the chapter's noise floor @eq-fair-floor.

Write $\mathcal L_g$ for group $g$'s average loss, $\nabla\mathcal L_g$ for its gradient (the group's average gradient $\bar g_g$ of the chapter), and $H_g$ for its Hessian. The unclipped training gradient is $\nabla\mathcal L = \sum_hs_h\nabla\mathcal L_h$, the share-weighted average of the groups' gradients.

One step moves the weights by $-\eta u$, with update $u = G_t/B + \tau\xi$. Here $G_t$ is the sum of the batch's clipped gradients, $B$ the expected batch size, $\xi$ standard normal noise and $\tau = \sigma C/B$ the noise per coordinate. By @eq-fair-groups, the update's mean is $m = \sum_hs_hm_h$, with $m_h$ the group's average clipped gradient. Write $\Sigma_B$ for the covariance of $G_t/B$, which comes from sampling the batch, and $\Delta\mathcal L_g$ for the change in $\mathcal L_g$ over the step. Then, to second order in $\eta$,
$$\mathbb E[\Delta\mathcal L_g] \approx \underbrace{-\eta\langle\nabla\mathcal L_g, \nabla\mathcal L\rangle}_{\text{descent}}\;\underbrace{-\;\eta\langle\nabla\mathcal L_g, m - \nabla\mathcal L\rangle}_{\text{clipping}} + \frac{\eta^2}2\Big(\underbrace{m^\top H_gm}_{\text{curvature}} + \underbrace{\operatorname{tr}(H_g\Sigma_B)}_{\text{sampling}} + \underbrace{\tau^2\operatorname{tr}H_g}_{\text{noise}}\Big),$$ {#eq-apx-learn-fair-step}
and the approximation is exact when $\mathcal L_g$ is quadratic. The noise term is $\eta^2\sigma^2C^2\operatorname{tr}H_g/(2B^2)$, the form the chapter describes.

### The idea

One step moves the weights by a random vector. A group's loss responds to the average move through its gradient, to first order, and to the spread of the move through its curvature, to second order. Clipping changes the average move. It shortens the gradients of groups with long gradients, so their share of the average falls and their loss falls more slowly. The noise does not change the average, but it adds spread. That spread is the same in every direction, so it raises a group's loss in proportion to the group's total curvature, the trace of its Hessian.

One dimension shows both effects. Take $\mathcal L(\theta) = \frac h2\theta^2$ and a step $\theta - \eta(m + \tau\xi)$, with $\xi$ standard normal. The new loss averages to $\frac h2\big((\theta - \eta m)^2 + \eta^2\tau^2\big)$, because a squared normal variable has mean equal to its variance. Expanded, that is $\mathcal L(\theta) - \eta h\theta m + \frac{\eta^2}2(hm^2 + h\tau^2)$. Here $h\theta$ is the gradient, so the first-order term is the gradient times the average move, and the noise adds $\frac{\eta^2}{2}h\tau^2$ whatever the average move is.

### Derivation

1. **The update's mean and spread.** By @eq-fair-groups, $\mathbb E[G_t/B] = m$. The noise $\tau\xi$ has mean zero, covariance $\tau^2I$, and is independent of the batch. So $u$ has mean $m$ and covariance $\Sigma_B + \tau^2I$.
2. **Expand the group's loss.** A second-order Taylor expansion around the current weights $\theta$ gives
$$\mathcal L_g(\theta - \eta u) = \mathcal L_g(\theta) - \eta\langle\nabla\mathcal L_g(\theta), u\rangle + \frac{\eta^2}2u^\top H_gu + R,$$
where the remainder $R$ involves third derivatives. If $\mathcal L_g$ is quadratic, $R = 0$ and $H_g$ is constant.
3. **Average.** The first-order term averages to $-\eta\langle\nabla\mathcal L_g, m\rangle$. For the second-order term, write $u = m + e$ with $\mathbb E[e] = 0$. Then $u^\top H_gu = m^\top H_gm + 2m^\top H_ge + e^\top H_ge$, and the middle term averages to zero. The last term averages to $\operatorname{tr}(H_g\operatorname{Cov}(u))$, because $\mathbb E[e^\top H_ge] = \sum_{jk}(H_g)_{jk}\mathbb E[e_je_k]$. With step 1, that is $\operatorname{tr}(H_g\Sigma_B) + \tau^2\operatorname{tr}H_g$.
4. **Split off clipping.** Write $m = \nabla\mathcal L + (m - \nabla\mathcal L)$. The first part is what the step would do without clipping. The second, $m - \nabla\mathcal L = \sum_hs_h(m_h - \nabla\mathcal L_h)$, is the bias that clipping adds. A group whose records are all shorter than $C$ has $m_h = \nabla\mathcal L_h$ and adds nothing to it. This gives the descent and clipping terms of @eq-apx-learn-fair-step.
5. **The sampling term.** Under Poisson sampling, each record $i$ enters the batch independently with probability $q$, so $G_t = \sum_ib_ic_i$ with $c_i$ the clipped gradient and $b_i$ a coin with variance $q(1 - q)$. With $B = qn$,
$$\Sigma_B = \frac{q(1 - q)}{B^2}\sum_ic_ic_i^\top = \frac{1 - q}{qn^2}\sum_ic_ic_i^\top.$$
It vanishes for full batches, $q = 1$.
6. **The noise term.** Substituting $\tau = \sigma C/B$ gives $\frac{\eta^2}{2}\tau^2\operatorname{tr}H_g = \eta^2\sigma^2C^2\operatorname{tr}H_g/(2B^2)$. It does not depend on the group's share, its gradients or the clipping weights, only on its curvature.
7. **The chapter's noise floor.** Take the chapter's quadratic model: group $g$ owns $d_g$ weights $\theta_g$, $\mathcal L_g = \frac\lambda2\|v\|^2$ with $v = \theta_g - \theta_g^*$, full batches and no clipping. On group $g$'s weights the mean update is $s_g\lambda v$ and the group's gradient is $\lambda v$, while $H_g = \lambda I$ on its $d_g$ weights. With $a = \eta s_g\lambda$, @eq-apx-learn-fair-step is exact and reads
$$\mathbb E[\Delta\mathcal L_g] = -\eta s_g\lambda^2\|v\|^2 + \frac{\eta^2}2\big(s_g^2\lambda^3\|v\|^2 + \tau^2d_g\lambda\big) = -a(2 - a)\mathcal L_g + \frac{\eta^2\tau^2d_g\lambda}{2}.$$
The loss stops changing on average when the two terms balance, at
$$\mathbb E[\mathcal L_g] = \frac{\eta^2\tau^2d_g\lambda}{2a(2 - a)} = \frac{d_g\eta\tau^2}{2s_g(2 - \eta s_g\lambda)},$$
which is @eq-fair-floor. The descent term shrinks as the loss falls, and the noise term does not, which is why the loss settles at a floor rather than at zero. The same balance, direction by direction with share 1, gives the per-direction cost $\eta s^2/(2(2 - \eta h_j))$ of @eq-pub-noise-cost, where $s$ is the noise per coordinate and $h_j$ a curvature.

### Checking it

The script uses the chapter's clipping example (@fig-fair-clipping-toy): 95 majority records with gradient $(0.2, 0)$ and 5 minority records with gradient $(0, 4)$. It gives each group the quadratic loss $\frac12\|\theta - \theta_g^*\|^2$, whose Hessian is the identity, placed so that its gradient at the current weights is the group's gradient. All 100 records are in every batch, so the sampling term is zero. The noise is the running example's at $\varepsilon = 4$, $\sigma = 1.36$, with $B = 100$ and $\eta = 0.5$.

At $C = 0.5$ the majority is not clipped, so its clipping term is 0. Its descent term is $-0.01900$, its curvature term $0.00459$ and its noise term $1.156 \times 10^{-5}$, for a total of $-0.01440$. The minority's descent term is $-0.40000$, but clipping adds back $0.35000$. With the same curvature and noise terms, its total is $-0.04540$. So clipping has cancelled seven eighths of the minority's progress in this step.

The formula is exact here because the losses are quadratic. To check it, the script raises the noise multiplier to $\sigma = 100$, so that the noise term matters, and averages the minority's loss change over 2,000,000 noise draws. The average is $0.01707$, against $0.01709$ from @eq-apx-learn-fair-step. Step 7 is checked on the chapter's model with 51 weights per group, $\lambda = 1$, $\eta = 0.5$ and $\tau = 0.0068$. For a group with share 0.05 the floor is $5.9702 \times 10^{-3}$, and the expected change there is 0.

@fig-apx-learn-fair-terms shows the four terms against the clipping norm. For the minority, the clipping term is almost as large as the descent term until $C$ approaches the minority's gradient length 4. The noise term grows like $C^2$ and is tiny per step at this budget. It matters over many steps because it does not shrink when the gradients do, which is the noise floor of step 7.

![The terms of @eq-apx-learn-fair-step for one full-batch step of the chapter's two-group example, against the clipping norm $C$, log–log: the loss decrease without clipping or noise, and the increases from clipping, from the curvature of the mean step and from the noise ($\sigma = 1.36$, $B = 100$, $\eta = 0.5$). The dotted line marks each group's gradient length.](figures/learn_fair_terms.svg){#fig-apx-learn-fair-terms}

### What it does not cover

For a loss that is not quadratic, the remainder of step 2 is of order $\eta^3$ times the third derivatives, so @eq-apx-learn-fair-step is accurate only for small steps. The decomposition is for one step at the current weights. It does not say how the terms evolve over training, which the chapter's feedback loop and noise floor address. Tran, Dinh & Fioretto (2021) also relate the trace of a group's Hessian to its distance from the decision boundary and its input norms, and that relation stays cited.

## Code

The script [apx_learning.py](../../code/appendix-derivations/apx_learning.py) computes every number on this page and draws every figure. It runs noisy projected gradient descent and output and objective perturbation on the running example. It evaluates the data-dependent PATE bound against exact divergences on two and three labels. It sums the Rényi moment of a user's subsets exactly, and checks the identity against a quadrature. For the tuning and fairness sections, it checks every inequality on random cases and by Monte Carlo. From the repository root:

```bash
uv run python code/appendix-derivations/apx_learning.py
```

@tbl-apx-learn-code collects the main results.

| Check | Value | Compared with |
| --- | --- | --- |
| Upper bound @eq-apx-learn-upper, $n = 3{,}000$ | $2.4742 \times 10^{-2}$ | measured $1.0754 \times 10^{-3}$ |
| Lower bound @eq-apx-learn-lower, $n = 10{,}000$ | $1.8991 \times 10^{-9}$ | upper bound $2.2268 \times 10^{-3}$ |
| Output perturbation, $n = 10{,}000$ | $[2.7835 \times 10^{-4},\ 7.2372 \times 10^{-3}]$ | measured $9.6417 \times 10^{-4}$ |
| Objective perturbation, $n = 10{,}000$ | $[3.2872 \times 10^{-7},\ 2.2221 \times 10^{-4}]$ | measured $7.1178 \times 10^{-5}$ |
| PATE bound, gap 150 | $1.2278 \times 10^{-3}$ | exact $2.2362 \times 10^{-4}$ |
| PATE bound, votes $(200, 40, 10)$ | $8.7544 \times 10^{-4}$ | exact $1.5188 \times 10^{-4}$ |
| Tuning bound at order 5, limit in $\hat\alpha$ | $1.4599$ | exact $0.3912$ |
| Corollary 4, closed form | $2.7791$ | numerical minimum $2.7791$ |
| User moment $A_2$, aligned | $1.232011$ | 60 degrees: $1.066665$ |
| Moment identity, quadrature | $1.516754$ | formula $1.516754$ |
| Minority's step, formula | $0.01709$ | Monte Carlo $0.01707$ |

: The main numbers the script prints, each against the value it is compared with. {#tbl-apx-learn-code}

**What to take from it.** Every bound holds on every check. The upper bounds are 3.1 to 23.0 times the measured values. The lower bounds prove the rates but not useful constants. The two places where the chapters relied on numerical evidence now have proofs, or a rigorous replacement. The pure-DP PATE bound always increases in $\beta$, and the Gaussian one can always be replaced by its largest value over $\beta \le \tilde\beta$. The tuning bound for a pure-DP run is best in the limit of large $\hat\alpha$, slightly below its value at the end of any finite grid of orders, and the chapter's table now reports that limit.

## References

- Bassily, Smith & Thakurta, "Private Empirical Risk Minimization: Efficient Algorithms and Tight Error Bounds" (FOCS 2014). The optimal rates and lower bounds of the first section, with stochastic gradient algorithms that use one record per step.
- Steinke & Ullman, "Between Pure and Approximate Differential Privacy" (Journal of Privacy and Confidentiality, 2016). The fingerprinting lemma used by the lower bounds, and the sharper version with the factor $\sqrt{\ln(1/\delta)}$.
- Lacoste-Julien, Schmidt & Bach, "A Simpler Approach to Obtaining an O(1/t) Convergence Rate for the Projected Stochastic Subgradient Method" (arXiv 2012). The weighted average that removes the logarithm from the strongly convex rate.
- Chaudhuri, Monteleoni & Sarwate, "Differentially Private Empirical Risk Minimization" (JMLR 2011). Output and objective perturbation, whose excess risk the exact bounds bracket.
- Papernot, Song, Mironov, Raghunathan, Talwar & Erlingsson, "Scalable Private Learning with PATE" (ICLR 2018). Theorem 6 and Proposition 10, and the proof of monotonicity under their condition (Appendix A).
- Papernot & Steinke, "Hyperparameter Tuning with Renyi Differential Privacy" (ICLR 2022). Theorem 2 and Corollaries 3 and 4, with the general case of an empty search (Appendix B).
- Charles, Ganesh, McKenna, McMahan, Mitchell, Pillutla & Rush, "Fine-Tuning Large Language Models with User-Level Differential Privacy" (arXiv 2407.07737, 2024). Theorem 3, the aligned worst case for both directions and all orders.
- Tran, Dinh & Fioretto, "Differentially Private Empirical Risk Minimization under the Fairness Lens" (NeurIPS 2021). The per-step decomposition of a group's loss change, and its link to the distance from the decision boundary.
