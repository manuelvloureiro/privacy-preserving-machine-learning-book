# Public data and private fine-tuning

The previous chapters trained models with differential privacy (DP) from random weights, by stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, called DP-SGD ([DP-SGD](02-dp-sgd.md)). On this part's running example, described below, it costs little at a privacy budget of $\varepsilon = 8$ and a great deal at $\varepsilon = 1$ and below. The most accurate private models are not trained that way. They are **pretrained** without privacy on a large **public** dataset, such as text or images from the web, and then **fine-tuned** privately on the sensitive data (Tramèr & Boneh 2021; Yu et al. 2022; Li et al. 2022). This chapter asks why that helps so much. It also asks why fine-tuning only a small part of the model, such as the last layer or a low-rank update, often beats fine-tuning all of it under DP.

The one idea is that DP-SGD pays for noise in every trainable direction, while the useful part of the gradient, the signal, does not grow with the number of directions. The noise has the same size in each coordinate it updates, so its length grows with the number of trainable weights. The signal has length at most the clipping norm, and in practice it is concentrated in a few directions. Public data move the starting point close to a good model, and they tell you, at no privacy cost, which few directions to train. Training only those cuts the noise without cutting the signal, unless the task needs a direction that was left out.

The running example is the part's. Its inputs are 50-dimensional standard normal vectors, each labelled by the sign of a fixed random teacher network with 8 hidden tanh units. The network being trained, the student, has 64 hidden tanh units and $p = 3{,}329$ weights, and it trains on $n = 10{,}000$ private records. A public task labels 50,000 more inputs with a teacher close to the private one (@sec-pub-run). Fine-tuning runs DP-SGD for $T = 10{,}000$ steps, with sampling rate $q = 0.01$ and clipping norm $C = 0.5$, at $\delta = 10^{-5}$, and neighbouring private datasets $D \sim D'$ differ by adding or removing one record. Pretrained on the public task, the student is right on 0.9035 of the private test inputs before any private training. At $\varepsilon = 1$, private training from scratch reaches 0.9246, and private fine-tuning of a small low-rank adapter reaches 0.9616. @tbl-pub-symbols lists the symbols.

| Symbol | Meaning |
| --- | --- |
| $n$, $p$, $d$ | Private records; weights of the model; trainable weights, the rest frozen |
| $C$, $\sigma$, $q$, $B = qn$ | Clipping norm, noise multiplier, sampling rate, expected batch size |
| $T$, $\eta$ | Number of steps; learning rate |
| $s = \sigma C/B$ | Noise standard deviation per coordinate on the averaged gradient |
| $H$, $h_j$, $d_{\text{eff}}$ | Curvature of the loss at the start; its eigenvalues; the effective dimension |
| $D_{\text{pub}}$, $D$ | Public dataset; private dataset |

: The symbols of this chapter. {#tbl-pub-symbols}

## The cost of each trainable direction {#sec-pub-dimension}

Every bound on private learning in this part grows with the dimension. [Private empirical risk minimization](01-dp-erm.md#sec-dperm-convex) shows that the best excess risk under $(\varepsilon, \delta)$-DP grows like $\sqrt d$ for convex Lipschitz losses and like $d$ for strongly convex ones. Yet large pretrained language models with hundreds of millions of weights are fine-tuned privately with good results. Before choosing which weights to train, you need to know what each trainable direction really costs.

The privacy side does not depend on that choice. [DP-SGD](02-dp-sgd.md#sec-dpsgd-algorithm) proves that each step's clipped sum has sensitivity $C$ whatever the model, as long as clipping acts on the vector of trainable coordinates, and its accountant turns $q$, $\sigma$ and $T$ alone into $(\varepsilon, \delta)$. Accuracy does depend on the choice, because the noise lands on every trainable coordinate. The next lemma measures that noise, in all $d$ coordinates and in a subspace. Its second half is what later proofs use to treat one direction at a time.

::: {#lem-pub-noise name="Gaussian noise in d coordinates and in a subspace"}
Let $\xi \sim \mathcal N(0, s^2I_d)$ be a vector of $d$ independent normal coordinates with mean 0 and standard deviation $s$. Its squared length has mean $ds^2$ and relative spread $\sqrt{2/d}$, so its length is close to $s\sqrt d$ when $d$ is large. In every orthonormal basis the coordinates of $\xi$ are again independent $\mathcal N(0, s^2)$. So if $P_m$ is the orthogonal projection onto a fixed $m$-dimensional subspace, $P_m\xi$ has standard deviation $s$ along each of $m$ orthonormal directions of the subspace, and length close to $s\sqrt m$.
:::

::: {.proof}
1. *The squared length.* It is a sum of $d$ independent terms $\xi_i^2$, each with mean $s^2$ and variance $\mathbb E[\xi_i^4] - s^4 = 2s^4$, since a normal's fourth moment is $3s^4$. So the sum has mean $ds^2$ and standard deviation $\sqrt{2d}\,s^2$, a share $\sqrt{2/d}$ of the mean, and Chebyshev's inequality (@thm-conc-chebyshev) keeps it near the mean.
2. *Rotation invariance.* The density of $\xi$ is proportional to $e^{-\|\xi\|^2/(2s^2)}$, which depends only on the length. A change of orthonormal basis keeps lengths, so the density keeps its product form in the new coordinates. @sec-apx-prob-convolution in the appendices proves this in full.
3. *Projection.* Choose the new basis so that its first $m$ vectors span the subspace. Then $P_m\xi$ keeps those $m$ coordinates, which by step 2 are independent $\mathcal N(0, s^2)$, and step 1 with $m$ in place of $d$ gives their length.
:::

In DP-SGD the noise on the averaged gradient is the step's noise $\xi_t \sim \mathcal N(0, \sigma^2C^2I_d)$ divided by the expected batch size $B$, the fixed denominator DP-SGD uses. Its standard deviation per coordinate is $s = \sigma C/B$, so by @lem-pub-noise its length is
$$\frac{\|\xi_t\|}B \approx s\sqrt d = \frac{\sigma C\sqrt d}B.$$ {#eq-pub-noise-length}

What the noise costs is the loss it adds, and its length alone does not say that. To find the loss, follow the noise on a loss that is quadratic near the pretrained weights, a local model that holds while fine-tuning stays near them. There the noise's contribution to the weights is a sum of independent Gaussian kicks, each shrunk by the curvature. Steep directions forget old kicks, while flat ones accumulate kicks but charge little for them.

::: {#thm-pub-noise-cost name="The noise cost of T noisy steps"}
Let the training loss be a quadratic in the weights $\theta$, $\mathcal L(\theta) = \mathcal L(\theta^*) + \frac12(\theta - \theta^*)^\top H(\theta - \theta^*)$. Assume that its curvature $H$, a symmetric matrix on the $d$ trainable coordinates, has eigenvalues $h_j$ with $0 \le h_j < 2/\eta$, so that each step shrinks every direction's error. Run $T$ steps of noisy gradient descent $\theta_{t+1} = \theta_t - \eta(\nabla\mathcal L(\theta_t) + \xi_t)$ with independent $\xi_t \sim \mathcal N(0, s^2I_d)$, and the same steps without noise from the same start. The noise raises the expected loss after $T$ steps by the **noise cost**
$$\mathcal E = \sum_{j=1}^d\frac{\eta s^2}{2(2 - \eta h_j)}\Big(1 - (1 - \eta h_j)^{2T}\Big) \approx \frac{\eta s^2}{4}\,d_{\text{eff}}, \qquad d_{\text{eff}} = \sum_{j=1}^d\Big(1 - (1 - \eta h_j)^{2T}\Big),$$ {#eq-pub-noise-cost}
where the approximation holds when $\eta h_j \ll 1$. The **effective dimension** $d_{\text{eff}}$ satisfies
$$d_{\text{eff}} \le \min\big(d,\ 2\eta T\operatorname{tr}H\big).$$ {#eq-pub-deff-bound}
:::

::: {.proof}
The plan is to write the noise's effect on the weights as a linear recursion, turn the loss increase into a variance, compute that variance one eigenvector at a time, and bound the sum.

1. *Separate the noise.* The gradient of the quadratic is $H(\theta - \theta^*)$. Write $\bar\theta_t$ for the iterate without noise and $e_t = \theta_t - \bar\theta_t$ for the difference. Subtracting the two update rules, the terms with $\theta^*$ cancel, so
$$e_{t+1} = (I - \eta H)e_t - \eta\xi_t, \qquad e_0 = 0.$$
So $e_t$ is a linear function of the noise alone, a Gaussian vector with mean zero.
2. *The loss increase as a variance.* Expanding the quadratic around $\bar\theta_T$ gives the loss without noise, a term linear in $e_T$ and a quadratic term:
$$\mathcal L(\theta_T) = \mathcal L(\bar\theta_T) + \nabla\mathcal L(\bar\theta_T)^\top e_T + \tfrac12e_T^\top He_T.$$
The middle term has mean zero, because $e_T$ has mean zero and $\bar\theta_T$ is not random. So $\mathcal E = \frac12\mathbb E[e_T^\top He_T]$.
3. *One direction at a time.* In the eigenbasis of $H$ the recursion splits into one scalar recursion per eigenvector, and by @lem-pub-noise the noise coordinates in that basis are still independent $\mathcal N(0, s^2)$. Along an eigenvector with curvature $h$, the error obeys $e_{t+1} = (1 - \eta h)e_t - \eta\xi$. Each kick shrinks by the factor $1 - \eta h$ at every later step, and the kicks are independent, so their variances add up to a geometric sum:
$$v_T = \eta^2s^2\sum_{t=0}^{T-1}(1 - \eta h)^{2t} = \eta^2s^2\,\frac{1 - (1 - \eta h)^{2T}}{1 - (1 - \eta h)^2}.$$
4. *Its cost.* The quadratic term of step 2 charges the direction $\frac12hv_T$. The denominator of $v_T$ factors as $\eta h(2 - \eta h)$, so for $h > 0$ the factor $h$ cancels, and for $h = 0$ both sides below are 0:
$$\frac12hv_T = \frac{\eta s^2}{2(2 - \eta h)}\Big(1 - (1 - \eta h)^{2T}\Big).$$
Summing over the $d$ directions gives @eq-pub-noise-cost, and when $\eta h \ll 1$ the first factor is $\eta s^2/4$.
5. *Which directions pay.* When $2\eta hT \gg 1$ the bracket is close to 1. The direction is at its steady state, the case [Private empirical risk minimization](01-dp-erm.md#sec-dperm-gradient) analyses, and it costs $\eta s^2/4$ whatever its curvature, because stiffer directions wander less but charge more. When $2\eta hT \ll 1$ the bracket is about $2\eta hT$, since $(1 - x)^{2T} \approx 1 - 2Tx$ for small $x$. The noise there has random-walked a distance of variance $T\eta^2s^2$, charged at the small rate $h/2$.
6. *The bound.* Each bracket is at most 1, which gives $d_{\text{eff}} \le d$. Bernoulli's inequality $(1 - x)^{2T} \ge 1 - 2Tx$ holds for $0 \le x \le 1$, and for larger $x$ its right side is negative while the even power on the left is not. So each bracket is at most $2\eta h_jT$, and summing over $j$ gives $d_{\text{eff}} \le 2\eta T\operatorname{tr}H$.
:::

So the count of trainable weights is an upper bound, not the price. Noise in a direction where the loss is flat costs nothing, however long it is. By step 5, what you pay for is roughly the number of directions with $2\eta hT$ above 1, that is with curvature above $1/(2\eta T)$, the directions the noise has time to fill. A pretrained model whose loss is curved in few directions can therefore be fine-tuned in all its weights at a cost set by those few. Li, Liu, Hashimoto, Inan, Kulkarni, Lee & Thakurta (2022) explain the success of private fine-tuning of large language models this way. And since $d_{\text{eff}}$ grows with $\eta T$, a longer or more aggressive run fills more flat directions with noise.

Fewer trainable weights still help, for two reasons. Freezing weights restricts $H$ to fewer coordinates, whose eigenvalues interlace with the full ones, so $d_{\text{eff}}$ has fewer and smaller terms and, when every $\eta h_j \le 1$ so that each term grows with $h_j$, can only fall. And with fewer coordinates each per-example gradient is shorter, so fewer gradients are shrunk by clipping before they are averaged, and the signal can be longer. Against both stands the **bias** of freezing: a direction the task needs but that is frozen cannot move. Note too that a loss increase is not an accuracy loss, so use @eq-pub-noise-cost to compare strategies, not to predict accuracy.

![Left: at $\varepsilon = 1$ ($\sigma = 4.13$, $B = 100$, $C = 0.5$), the noise length $\sigma C\sqrt d/B$ (line) against the number of trainable weights $d$, and the length of the average clipped gradient at the pretrained model (points) for the last layer, low-rank adapters of ranks 1 to 16, and all weights. Right: their ratio. The noise grows like $\sqrt d$ while the signal shrinks, because longer gradient vectors are clipped more.](figures/pub_noise_signal.svg){#fig-pub-noise-signal}

::: {#exm-pub-dimension}
@tbl-pub-noise gives the noise length @eq-pub-noise-length for the three trainable sets of @sec-pub-run, with $B = 100$ and $C = 0.5$, and the noise multiplier $\sigma$ that the accountant gives for each budget.

| $\varepsilon$ | $\sigma$ | All weights, $d = 3{,}329$ | Rank-4 adapter and last layer, $d = 525$ | Last layer, $d = 65$ |
| --- | --- | --- | --- | --- |
| 0.5 | 7.72 | 2.227 | 0.884 | 0.311 |
| 1 | 4.13 | 1.191 | 0.473 | 0.166 |
| 2 | 2.28 | 0.658 | 0.261 | 0.092 |
| 8 | 0.917 | 0.265 | 0.105 | 0.037 |

: Length of the noise on the averaged gradient, $\sigma C\sqrt d/B$, for three numbers of trainable weights. {#tbl-pub-noise}

The signal is the average clipped gradient over the private records at the pretrained weights. Its length is 0.0130 when all weights are trained, 0.0207 for the adapter and 0.0250 for the last layer alone (@fig-pub-noise-signal). The last layer's 65 coordinates carry a *longer* signal than all 3,329 together, because each record's gradient restricted to them is shorter and less often clipped. At $\varepsilon = 1$ the noise is 92.0 times the signal for all weights, 22.9 times for the adapter and 6.7 times for the last layer. A signal of constant direction outgrows a random walk of noise after $(\text{noise}/\text{signal})^2$ steps ([DP-SGD](02-dp-sgd.md#sec-dpsgd-algorithm)), here 8,455, 523 and 44 steps.

The curvature tells a less pessimistic story. The chapter's script (@sec-pub-code), as a diagnostic outside any private release, measures the **Gauss–Newton curvature** of the private loss, the Hessian of the logistic loss with the network replaced by its linearization at the pretrained weights. It is positive semi-definite, and near a good fit it is close to the full Hessian. Estimated from 2,000 private records, it has largest eigenvalue 0.723 and trace 16.310 over all weights, and 0.142 and 0.681 for the last layer. With $\eta = 0.1$ and $T = 10{,}000$, only 406 of the 3,329 directions have curvature above $1/(2\eta T) = 5 \times 10^{-4}$, and $d_{\text{eff}} = 451$, against 60.2 of 65 for the last layer. The full model's noise is $\sqrt{3{,}329/65} = 7.2$ times longer than the last layer's, so counting weights would charge it the squared ratio, $3{,}329/65 = 51$ times the cost, but @eq-pub-noise-cost charges only $451/60.2 = 7.5$ times. At $\varepsilon = 1$, where $s = 0.0206$, @eq-pub-noise-cost predicts a loss increase of 0.0048 for all weights at $\eta = 0.1$, against 0.0355 if every direction had reached its steady state. For the last layer alone it predicts 0.00064.
:::

## The last layer on frozen features {#sec-pub-features}

The simplest way to use a pretrained model is to freeze everything except the last layer. For the running example that is logistic regression on the 64 hidden activations plus a bias, 65 weights, and it turns private deep learning back into the convex problem of [Private empirical risk minimization](01-dp-erm.md). Tramèr & Boneh (2021) showed that this recipe beats end-to-end private training on standard image benchmarks at moderate budgets, even with handcrafted features, and that features learned on public data do better still. This section asks what the private part costs, and what limits it.

::: {#def-pub-probe name="Linear probing"}
Let $\phi\colon \mathcal X \to \mathbb R^d$ be a feature map computed from public data only, here the pretrained network's $d - 1$ hidden tanh activations and a constant 1. **Linear probing** freezes $\phi$ and trains a linear classifier $\theta \in \mathbb R^d$ on the features, with labels $y \in \{-1, 1\}$ and the logistic loss
$$\ell(\theta; (x, y)) = \ln\big(1 + e^{-y\langle\theta, \phi(x)\rangle}\big).$$
:::

The privacy of a probe rests on one fact: as far as the private data are concerned, the features are a fixed function. Step 3 of the probe's proof needs this, and so do @sec-pub-subspace and @sec-pub-caveats, so the next lemma states it once.

::: {#lem-pub-free name="Computations on public data are free"}
Let $a$ be the output of any computation that reads only the public dataset $D_{\text{pub}}$, with coins independent of the private dataset $D$. Suppose that for every fixed value of $a$ the mechanism $D \mapsto M(D; a)$ is $(\varepsilon, \delta)$-DP. Then for every fixed $D_{\text{pub}}$, the mechanism $D \mapsto M(D; a)$ with $a$ computed from $D_{\text{pub}}$ is $(\varepsilon, \delta)$-DP.
:::

::: {.proof}
1. *The public output does not move.* Fix $D_{\text{pub}}$, neighbours $D \sim D'$ and an event $S$. The computation never reads the private data, so $a$ has the same distribution $\rho$ under $D$ and under $D'$.
2. *Average the guarantee.* Each probability is an average over $a \sim \rho$ of the probability for a fixed $a$. The DP inequality holds for each fixed $a$, and averaging both sides over the same $\rho$ keeps it:
$$\mathbb E_{a \sim \rho}\Pr[M(D; a) \in S] \le \mathbb E_{a \sim \rho}\big[e^\varepsilon\Pr[M(D'; a) \in S] + \delta\big] = e^\varepsilon\,\mathbb E_{a \sim \rho}\Pr[M(D'; a) \in S] + \delta.$$
:::

In [DP-SGD](02-dp-sgd.md#sec-dpsgd-algorithm) this is the assumption that the initial weights do not depend on the private data.

::: {#prp-pub-probe name="The cost of a private linear probe"}
In the setting of @def-pub-probe, the following hold.

1. Every per-record gradient has length at most $\|\phi(x)\| < \sqrt d$, so the loss is $\sqrt d$-Lipschitz in $\theta$ and clipping is not needed for privacy.
2. Write $\mathcal L_{\mathcal D}$ for the population risk, $\mathcal L_{\text{best}}$ for the best population risk of any model, $\theta^\phi$ for the best linear model on the features and $\tilde\theta$ for the private probe. The excess risk splits into a **feature gap** and a **privacy cost**,
$$\mathcal L_{\mathcal D}(\tilde\theta) - \mathcal L_{\text{best}} = \underbrace{\mathcal L_{\mathcal D}(\theta^\phi) - \mathcal L_{\text{best}}}_{\text{features}} + \underbrace{\mathcal L_{\mathcal D}(\tilde\theta) - \mathcal L_{\mathcal D}(\theta^\phi)}_{\text{privacy}},$$ {#eq-pub-probe-split}
and the feature gap depends on neither $n$ nor $\varepsilon$.
3. The privacy cost is that of private convex empirical risk minimization (ERM) in $d$ dimensions. For losses that are $L$-Lipschitz over a ball of radius $R$, noisy projected gradient descent keeps it below a constant times $LR\sqrt{d\ln(1/\delta)}/(n\varepsilon)$, and no private algorithm does better than order $LR\sqrt d/(n\varepsilon)$.
:::

::: {.proof}
1. *The gradient bound.* By the chain rule, $\nabla_\theta\ell = \ell'(u)\,y\,\phi(x)$, where $u = y\langle\theta, \phi(x)\rangle$ is the margin and $\ell'(u) = -1/(1 + e^u)$ lies in $(-1, 0)$. Each tanh activation lies in $(-1, 1)$ and the constant is 1, so $\|\phi(x)\|^2 < (d - 1) + 1 = d$.
2. *The split.* Adding and subtracting $\mathcal L_{\mathcal D}(\theta^\phi)$ gives @eq-pub-probe-split. Its first difference compares two non-private quantities, so it contains no noise and does not depend on $n$ or $\varepsilon$.
3. *The privacy term.* By @lem-pub-free the features are a fixed function, so the probe is ordinary convex ERM on the records $(\phi(x_i), y_i)$, with Lipschitz constant $\sqrt d$, or $C$ if you clip. [Private empirical risk minimization](01-dp-erm.md#sec-dperm-convex) proves the upper bound for noisy projected gradient descent and cites the lower bound of Bassily, Smith & Thakurta (2014).
:::

Features cap accuracy, and privacy only decides how close you get to the cap. The feature gap is paid even with infinite data and no privacy, while the privacy cost falls like $\sqrt d/(n\varepsilon)$. With few poor features the probe is private, cheap and bad, which is the point of Tramèr & Boneh's title: private learning needs better features, or much more data. End-to-end training can learn better features, but it pays for every weight to do so.

So linear probing is the safe first baseline. The problem is convex, so the optimizer cannot wander into a bad region, and every method of [Private empirical risk minimization](01-dp-erm.md) applies. Clipping still pays, because the bound $\sqrt d$ is reached only by a record with every activation saturated and a large negative margin. A record classified with a wide margin has $|\ell'(u)|$ close to 0, so clipping at a smaller $C$ biases only a few gradients and divides the noise by $\sqrt d/C$. [Exercise 3](../../book-exercises/part05-dp-ml/06-public-data.md#sec-sol-pub-3) compares the two on the running example.

::: {#exm-pub-probe}
The pretrained network's 64 hidden units plus a constant give $d = 65$ features. The largest feature norm on the private training records is 7.035, below the bound $\sqrt{65} = 8.062$. The pretrained network with its public last layer labels 0.9035 of the private test inputs correctly. Fitting the last layer exactly and without privacy on the 10,000 private records raises this to 0.9536, the probe's ceiling $\theta^\phi$ in @eq-pub-probe-split. A network trained without privacy from scratch reaches 0.9606, so the feature gap is 0.007 in accuracy. The private probe reaches 0.9478 at $\varepsilon = 0.5$ and 0.9514 at $\varepsilon = 8$ (@tbl-pub-run). At every budget it is within 0.006 of its ceiling, and at no budget can it pass it.
:::

## Low-rank adapters {#sec-pub-lora}

The last layer is often not enough, because features pretrained on one task are not exactly right for another. Training all weights fixes the features but pays for every weight. Low-rank adaptation (LoRA) sits in between (Hu et al. 2022). Yu et al. (2022) showed that private fine-tuning of language models with it matches or beats private full fine-tuning at a fraction of the cost. To run it under DP-SGD you need to know how many weights it trains, what its per-example gradients are, and where it can move each matrix.

::: {#def-pub-lora name="Low-rank adapter"}
Take a pretrained $d_{\text{out}} \times d_{\text{in}}$ weight matrix $W_0$ and a rank $r$. A **low-rank adapter** freezes $W_0$ and trains two factors $U \in \mathbb R^{d_{\text{out}} \times r}$ and $V \in \mathbb R^{d_{\text{in}} \times r}$, so that the layer uses
$$W = W_0 + UV^\top.$$
Training starts from $U = 0$ and a random $V$, so the model starts exactly at the pretrained one. The adapter trains
$$d = r(d_{\text{in}} + d_{\text{out}})$$ {#eq-pub-lora-count}
weights, against $d_{\text{in}}d_{\text{out}}$ for the matrix.
:::

The adapter saves noise only while $d$ is below $d_{\text{in}}d_{\text{out}}$, and [Exercise 1](../../book-exercises/part05-dp-ml/06-public-data.md#sec-sol-pub-1) finds the rank where it stops. For the matrix itself, record $i$'s gradient is the outer product $\nu_ia_i^\top$ of the gradient $\nu_i$ with respect to the layer's output and the layer's input $a_i$. Its length is $\|\nu_i\|\,\|a_i\|$, so clipping can compute it without forming the gradient, which is ghost clipping ([DP-SGD](02-dp-sgd.md#sec-dpsgd-ghost)). The factors' gradients follow by the chain rule and stay outer products.

::: {#prp-pub-lora name="Gradients and steps of a low-rank adapter"}
For record $i$, let $\nu_i$ be the gradient of its loss with respect to the layer's output and $a_i$ the layer's input. The adapter's per-example gradients are outer products,
$$\nabla_U\ell_i = \nu_i\,(V^\top a_i)^\top, \qquad \nabla_V\ell_i = a_i\,(U^\top\nu_i)^\top,$$ {#eq-pub-lora-grad}
with lengths $\|\nu_i\|\,\|V^\top a_i\|$ and $\|a_i\|\,\|U^\top\nu_i\|$. Take one step of size $\eta/B$ on both factors with the noisy sums of clipped gradients (the bar marks clipping) $G_U = \sum_i\overline{\nabla_U\ell_i} + \xi_U$ and $G_V = \sum_i\overline{\nabla_V\ell_i} + \xi_V$. It changes $W$ to $W'$ with
$$W' - W = -\frac\eta B\big(G_UV^\top + UG_V^\top\big) + O\Big(\frac{\eta^2}{B^2}\Big).$$ {#eq-pub-lora-step}
From $U = 0$, the change $W' - W$ has rank at most $r$, and to first order its rows lie in the span of the columns of $V$. The privacy guarantee is that of DP-SGD with the same $q$, $\sigma$ and $T$.
:::

::: {.proof}
The plan is to push the matrix gradient through the product $UV^\top$, read the lengths off the outer products, expand one step, and set $U = 0$.

1. *The gradient for $U$.* Write $G_i = \nu_ia_i^\top$ for record $i$'s gradient with respect to $W$, through which alone the loss depends on the factors. A small change $\mathrm dU$ changes $W$ by $\mathrm dU\,V^\top$, and so the loss by $\operatorname{tr}(G_i^\top\mathrm dU\,V^\top)$. The cyclic property of the trace moves $V^\top$ to the front:
$$\operatorname{tr}(G_i^\top\mathrm dU\,V^\top) = \operatorname{tr}(V^\top G_i^\top\mathrm dU) = \langle G_iV, \mathrm dU\rangle.$$
So $\nabla_U\ell_i = G_iV = \nu_i(V^\top a_i)^\top$.
2. *The gradient for $V$.* A change $\mathrm dV$ changes $W$ by $U\,\mathrm dV^\top$ and the loss by $\operatorname{tr}(G_i^\top U\,\mathrm dV^\top) = \langle G_i^\top U, \mathrm dV\rangle$, so $\nabla_V\ell_i = G_i^\top U = a_i(U^\top\nu_i)^\top$.
3. *Lengths.* An outer product of two vectors has length equal to the product of their lengths. So record $i$'s whole gradient norm comes from vectors of dimension $d_{\text{out}}$, $d_{\text{in}}$ and $r$, without forming a matrix.
4. *The step.* After the step, $W' = W_0 + (U - \frac\eta BG_U)(V - \frac\eta BG_V)^\top$. Expanding the product and subtracting $W = W_0 + UV^\top$ gives
$$W' - W = -\frac\eta B\big(G_UV^\top + UG_V^\top\big) + \frac{\eta^2}{B^2}G_UG_V^\top,$$
which is @eq-pub-lora-step.
5. *From $U = 0$.* By step 2, $\nabla_V\ell_i = 0$ for every record, so $G_V = \xi_V$ is pure noise and the first-order change is $-\frac\eta BG_UV^\top$, whose rows are combinations of the columns of $V$. The second-order term $\frac{\eta^2}{B^2}G_U\xi_V^\top$ tilts them slightly. The whole change is a $d_{\text{out}} \times r$ matrix times an $r \times d_{\text{in}}$ one, so its rank is at most $r$.
6. *Privacy.* Clipping acts on the vector of all trainable coordinates, so the clipped sum has sensitivity $C$ and the guarantee of [DP-SGD](02-dp-sgd.md#sec-dpsgd-algorithm) applies.
:::

An adapter is DP-SGD in a smaller space that moves. Its noise lives in $d = r(d_{\text{in}} + d_{\text{out}})$ coordinates, so @eq-pub-noise-length applies with that $d$, and noise in the other directions of $W$ never arises. But the restriction costs something at the start. By step 5 of the proof, with $U = 0$ the first update is the true gradient times $VV^\top$, which for a random $V$ of small rank is close to a multiple of the projection onto a random $r$-dimensional subspace. By symmetry, such a subspace captures on average only $r/d_{\text{in}}$ of a fixed vector's energy, since each of the $d_{\text{in}}$ directions of a random basis gets an equal share. Once $U \neq 0$ the gradient for $V$ is no longer zero, so $V$ turns toward the directions the gradients use. So low-rank adaptation works when the change the new task needs is itself close to low rank, as Hu et al. (2022) observed for language models.

Full fine-tuning is not ruled out. Li, Tramèr, Liang & Hashimoto (2022) fine-tuned all the weights of language models with DP-SGD and matched or beat the parameter-efficient methods, those that train few weights such as adapters, once the hyperparameters were right. @thm-pub-noise-cost explains why this can work: full fine-tuning pays $d_{\text{eff}}$, not $p$. Which wins depends on the model, the data and the tuning. The adapter's other advantages are certain, since it stores only $d$ numbers per task and clips much smaller gradients.

::: {#exm-pub-lora}
The running example's first layer maps 51 inputs (50 features and a constant) to 64 hidden units, which is 3,264 weights. A rank-4 adapter trains $4 \times (51 + 64) = 460$ of them, and with the 65 weights of the last layer that gives $d = 525$. Rank 1 gives 180 and rank 16 gives 1,905. On 256 private records, the ghost norms of @eq-pub-lora-grad agree with explicitly formed per-example gradients to a relative difference of $3.3 \times 10^{-16}$. [Exercise 4](../../book-exercises/part05-dp-ml/06-public-data.md#sec-sol-pub-4) checks the rank of the first step.
:::

## Gradients near a public subspace {#sec-pub-subspace}

@thm-pub-noise-cost says noise costs little in directions the loss does not use, and near a good fit those are the directions the private gradients do not point along. If you knew the directions the gradients do use, you could add noise only there. Public data give an estimate: compute per-example gradients on public records, find their top principal directions, and project the private gradients onto them. Zhou, Wu & Banerjee (2021) call this **projected DP-SGD**. The **gradient embedding perturbation** of Yu, Zhang, Chen & Liu (2021) also releases the part outside the subspace with separate noise. This section asks how much a projection saves and what it loses.

A projection chosen from public data is a fixed map, so it shrinks the noise and the signal but leaves the guarantee alone.

::: {#prp-pub-proj name="Projected DP-SGD"}
Let $P_m$ be the orthogonal projection onto an $m$-dimensional subspace computed from public data only. Each step releases
$$P_m\Big(\sum_{i \in S_t}\bar g_i\Big) + \mathcal N(0, \sigma^2C^2P_m),$$ {#eq-pub-proj}
the projected sum of the clipped gradients $\bar g_i$ of the batch $S_t$ plus noise of standard deviation $\sigma C$ along each of $m$ orthonormal directions of the subspace. This has the same $(\varepsilon, \delta)$ as DP-SGD with the same $q$, $\sigma$ and $T$. Let $\bar g_D$ be the average clipped gradient over the private records, and $f_m = \|P_m\bar g_D\|^2/\|\bar g_D\|^2$ the share of its energy in the subspace. The kept signal has length $\sqrt{f_m}\,\|\bar g_D\|$, the noise on the averaged gradient has length about $s\sqrt m$, and the signal outgrows the noise after about
$$T_{\text{even}}(m) = \Big(\frac{s\sqrt m}{\sqrt{f_m}\,\|\bar g_D\|}\Big)^2$$ {#eq-pub-break-even}
steps, against $(s\sqrt p/\|\bar g_D\|)^2$ without projection. The lost share $1 - f_m$ is a bias that no number of steps recovers.
:::

::: {.proof}
1. *Sensitivity.* A projection never lengthens a vector, because $x = P_mx + (x - P_mx)$ with orthogonal parts, so $\|x\|^2 = \|P_mx\|^2 + \|x - P_mx\|^2$. Adding or removing one record therefore changes the projected sum by at most $C$.
2. *The mechanism.* In an orthonormal basis of the subspace, @eq-pub-proj is a vector of $m$ coordinates and sensitivity at most $C$, plus independent $\mathcal N(0, \sigma^2C^2)$ noise on each. With Poisson sampling it is the subsampled Gaussian of DP-SGD. By @lem-pub-free the public subspace is a fixed parameter, so the accountant of [DP-SGD](02-dp-sgd.md#sec-dpsgd-algorithm) gives the same $(\varepsilon, \delta)$.
3. *Signal and noise.* Dividing by $B$, the kept signal is $P_m\bar g_D$, of length $\sqrt{f_m}\,\|\bar g_D\|$ by the definition of $f_m$. The noise has $m$ coordinates of standard deviation $s$, so by @lem-pub-noise its length is about $s\sqrt m$.
4. *Break-even.* Over $j$ steps a signal of constant direction adds up to $j$ times its length, and independent noise to $\sqrt j$ times its length. Setting $j\sqrt{f_m}\,\|\bar g_D\| = \sqrt j\,s\sqrt m$ and solving for $j$ gives @eq-pub-break-even.
:::

A projection trades bias for variance, and the public data fix the exchange rate. The noise falls by $\sqrt{p/m}$ and the signal by $\sqrt{f_m}$. If the private gradients lie near the public subspace, $f_m$ is close to 1 for $m \ll p$ and the trade is excellent. If the public data are unrelated, $f_m \approx m/p$ as for a random subspace, and the projection removes as much signal as noise. The subspace must come from public data. One found from the private gradients themselves leaks information about the records, and @lem-pub-free no longer applies.

A fixed subspace is also a cap, since training can then move only within it. So projected DP-SGD recomputes the subspace as training proceeds, and gradient embedding perturbation pays a little more noise to release the residual. The same concentration explains why full fine-tuning can work: gradients near a low-dimensional subspace come from a loss curved in few directions, which makes $d_{\text{eff}}$ small.

![Left: eigenvalues of the Gauss–Newton curvature of the private loss at the pretrained weights, estimated from 2,000 private records, on log scales, with the thresholds $1/(2\eta T)$ for $\eta = 0.1$ and $0.5$ and $T = 10{,}000$; directions below a threshold contribute little noise cost. Right: the fraction of the energy of the private average clipped gradient that lies in the top $m$ principal directions of 500 public per-example gradients (solid), and in $m$ random directions (dashed).](figures/pub_subspace.svg){#fig-pub-subspace}

::: {#exm-pub-subspace}
At the pretrained weights the average clipped private gradient has length 0.0130. @tbl-pub-subspace gives the share of its energy captured by the top principal directions of 500 clipped public per-example gradients, and by random subspaces of the same dimension.

| Dimension $m$ | 1 | 4 | 16 | 64 | 256 | 3,329 (all) |
| --- | --- | --- | --- | --- | --- | --- |
| Public top-$m$ directions, $f_m$ | 0.596 | 0.623 | 0.655 | 0.732 | 0.887 | 1 |
| $m$ random directions | 0.001 | 0.001 | 0.004 | 0.022 | 0.080 | 1 |
| Break-even steps at $\varepsilon = 1$, @eq-pub-break-even | 4 | 16 | 62 | 222 | 733 | 8,455 |

: How much of the private average clipped gradient lies in public subspaces, at the pretrained weights, and the steps after which the kept signal outgrows the noise at $\varepsilon = 1$ ($s = 0.0206$). {#tbl-pub-subspace}

A single public direction holds 60% of the private gradient's energy, and 64 directions, 2% of the dimensions, hold 73%. At $\varepsilon = 1$ the kept signal outgrows the noise after 222 steps in the 64-dimensional subspace, against 8,455 steps for all weights. The cost is the 27% of the energy outside it, which a fixed projection never learns. The curvature in the left panel of @fig-pub-subspace tells the same story of few important directions. Its eigenvalues fall from 0.723 to below $5 \times 10^{-4}$ within the first 406 directions, so a run with $\eta = 0.1$ fills only the directions above that threshold line.
:::

## Pretraining and fine-tuning the running example {#sec-pub-run}

The previous sections predict which strategy should win. This section measures how much a related public dataset buys on the running example at each budget, and which part of the model to fine-tune. The public task is built with care, because an identical task would leave nothing to learn privately and an unrelated one would teach nothing.

::: {#def-pub-task name="The public task and pretraining"}
The private teacher has a first layer $W_1^{\text{priv}}$ (8 hidden units, 51 inputs) and an output layer $W_2^{\text{priv}}$. The **public teacher** has first layer $W_1^{\text{priv}} + 0.15Z_1$ and output layer $\cos(0.3)\,W_2^{\text{priv}} + \sin(0.3)\,Z_2$, where $Z_1$ and $Z_2$ are fixed standard normal draws. So it has the same hidden units, each slightly perturbed, read out with weights turned by 0.3 radians toward a random vector. It labels 50,000 inputs from the private input distribution. **Pretraining** trains the 64-unit student on them by SGD without privacy, for 10,000 steps with batches of 100 and learning rate 0.5.
:::

::: {#def-pub-methods name="The four fine-tuning strategies"}
From the pretrained weights, DP-SGD with $q = 0.01$, $C = 0.5$ and $T = 10{,}000$ trains one of three sets. **Full** trains all 3,329 weights, **last layer** trains $d = 65$, and **adapter** trains a rank-4 adapter $W_1 = W_1^{\text{pre}} + UV^\top$ on the first layer and the last layer, $d = 525$. The baseline, **scratch**, trains all weights from a random initialization, as in [DP-SGD](02-dp-sgd.md). The private test set of 20,000 records is split in half. Each method's learning rate is chosen from three values on the first half at $\varepsilon = 1$ and used at every budget, and accuracy is reported on the second half.
:::

The four strategies differ in where they start and what they train, but not in what they cost, which is what makes the comparison fair.

::: {#prp-pub-methods name="One guarantee for every strategy"}
At each budget, the four strategies of @def-pub-methods are $(\varepsilon, \delta)$-DP with respect to the private training set, with the same noise multiplier: $\sigma = 7.72$, 4.13, 2.28 and 0.917 for $\varepsilon = 0.5$, 1, 2 and 8 at $\delta = 10^{-5}$.
:::

::: {.proof}
1. *Fixed starting points.* The pretrained weights and the adapter's random $V$ come from public data and independent coins, so by @lem-pub-free they are fixed parameters of each private run.
2. *The same accountant.* Each method clips the vector of its trainable coordinates, so each step's clipped sum has sensitivity $C$ and the guarantee of [DP-SGD](02-dp-sgd.md#sec-dpsgd-algorithm) applies. It depends only on $q$, $\sigma$ and $T$, which are the same for every method, so the accountant finds the same $\sigma$ for each budget.
:::

The proposition covers the training set and not the learning-rate choice, which looked at the private validation half without paying for it. [Hyperparameter tuning](03-hyperparameter-tuning.md) shows how to tune privately. Here the comparison is of methods at their best, as in [DP-SGD](02-dp-sgd.md#sec-dpsgd-batch).

::: {#exm-pub-run}
The public teacher agrees with the private one on 0.9004 of the private test inputs. Pretraining reaches 0.9861 training accuracy on the public task, and 0.9035 on the private task, so it is useful but not sufficient. @tbl-pub-lr shows the learning-rate selection, and @tbl-pub-run the results.

| Method | Learning rates tried, accuracy on the validation half | Chosen |
| --- | --- | --- |
| Scratch | 0.25: 0.9318; 0.5: 0.9303; 1: 0.8582 | 0.25 |
| Full | 0.03: 0.9559; 0.1: 0.9567; 0.3: 0.9433 | 0.1 |
| Last layer | 0.1: 0.9479; 0.3: 0.9499; 1: 0.9460 | 0.3 |
| Adapter, rank 4 | 0.03: 0.9556; 0.1: 0.9619; 0.3: 0.9472 | 0.1 |

: Learning-rate selection at $\varepsilon = 1$, one run each, on the validation half of the private test set. {#tbl-pub-lr}

| | Scratch | Full | Last layer | Adapter, rank 4 |
| --- | --- | --- | --- | --- |
| Trainable weights $d$ | 3,329 | 3,329 | 65 | 525 |
| $\varepsilon = 0.5$ | $0.8589 \pm 0.0075$ | $0.9395 \pm 0.0006$ | $0.9478 \pm 0.0010$ | $0.9464 \pm 0.0005$ |
| $\varepsilon = 1$ | $0.9246 \pm 0.0050$ | $0.9599 \pm 0.0013$ | $0.9497 \pm 0.0000$ | $0.9616 \pm 0.0001$ |
| $\varepsilon = 2$ | $0.9518 \pm 0.0010$ | $0.9663 \pm 0.0001$ | $0.9519 \pm 0.0005$ | $0.9653 \pm 0.0002$ |
| $\varepsilon = 8$ | $0.9611 \pm 0.0012$ | $0.9677 \pm 0.0001$ | $0.9514 \pm 0.0005$ | $0.9655 \pm 0.0001$ |
| Non-private | 0.9606 | 0.9704 | 0.9536 | — |

: Test accuracy on the report half of the private test set, mean and standard deviation over two runs, $\delta = 10^{-5}$. The pretrained model without fine-tuning has 0.9035. Non-private: SGD for 1,000 steps from scratch and from the pretrained weights, and the exact logistic regression on the frozen features. {#tbl-pub-run}

Pretraining is worth about a factor of eight in $\varepsilon$. Every fine-tuning method at $\varepsilon = 0.5$ (0.9395 to 0.9478) beats training from scratch at $\varepsilon = 1$ (0.9246). The adapter at $\varepsilon = 1$ (0.9616) matches training from scratch at $\varepsilon = 8$ (0.9611), and full fine-tuning at $\varepsilon = 2$ (0.9663) beats it.
Which weights to train depends on the budget. At $\varepsilon = 0.5$ the last layer is best, because its noise is $\sqrt{3{,}329/65} = 7.2$ times shorter than full fine-tuning's and its ceiling, 0.9536, is not yet binding. From $\varepsilon = 1$ on, it is stuck near that ceiling (0.9497 to 0.9519), while the adapter and full fine-tuning keep improving, since they can correct the features. The adapter is never more than 0.0022 below the best method, so the middle choice is the robust one. Full fine-tuning is behind at $\varepsilon = 0.5$, by 0.0083, but best at $\varepsilon = 2$ and 8, as Li et al. (2022) found for language models.

The learning rate matters as much as the method. In @tbl-pub-lr full fine-tuning loses 1.3 percentage points at a learning rate three times too large, and the adapter 1.5 points. Their best learning rate, 0.1, is 2.5 times smaller than for training from scratch, which is @thm-pub-noise-cost at work: a larger $\eta$ fills more flat directions with noise and undoes what pretraining provided.

The adapter's rank matters less. At $\varepsilon = 1$, two runs each give $0.9590 \pm 0.0038$ at rank 1 ($d = 180$), $0.9624 \pm 0.0023$ at rank 4 ($d = 525$) and $0.9586 \pm 0.0031$ at rank 16 ($d = 1{,}905$). A higher rank follows more of the directions the task needs but adds noise, and over a sixteenfold range of ranks neither force wins: the differences are about the size of the run-to-run spread. Rank 4's mean over these two seeds, 0.9624, even differs from the 0.9616 of @tbl-pub-run by about as much, so differences between pretrained methods smaller than about 0.005 should not be read as rankings.
:::

![Test accuracy of the four strategies against $\varepsilon$ at $\delta = 10^{-5}$, one run each with the learning rates of @tbl-pub-lr and a different seed from @tbl-pub-run, with the pretrained model's accuracy before fine-tuning (dotted). The pretrained methods are far ahead of training from scratch at small budgets; the last layer stays near its ceiling from $\varepsilon = 1$ on, and training from scratch passes it at larger budgets.](figures/pub_accuracy.svg){#fig-pub-accuracy}

In the single runs of @fig-pub-accuracy the order of the pretrained methods at $\varepsilon = 0.5$ and 1 changes, as the run-to-run spread predicts.

## What public data does not buy {#sec-pub-caveats}

Everything so far treated the public data as free, and formally that is right. But "public" is a label someone chose, and the guarantee says nothing about the records under that label. Tramèr, Kamath & Carlini (2024) argue that the field's reliance on web-scraped pretraining data needs scrutiny. Before you report a guarantee for a fine-tuned model, you need to know exactly what it covers.

::: {#thm-pub-guarantee name="What the guarantee covers"}
Let $M(D_{\text{pub}}, D)$ pretrain on $D_{\text{pub}}$ without privacy and fine-tune on $D$ with an $(\varepsilon, \delta)$-DP algorithm whose randomness is independent of the pretraining. Then for every fixed $D_{\text{pub}}$, the mechanism $D \mapsto M(D_{\text{pub}}, D)$ is $(\varepsilon, \delta)$-DP. There is no guarantee of any kind with respect to $D_{\text{pub}}$. A person with a record $x$ in both datasets is protected only for the copy in $D$, while the copy in $D_{\text{pub}}$ can be revealed completely.
:::

::: {.proof}
1. *Fixed public data.* With $D_{\text{pub}}$ fixed, pretraining produces weights $\theta_0$, or a distribution over them, that do not depend on $D$. Fine-tuning is $(\varepsilon, \delta)$-DP for every fixed $\theta_0$, so @lem-pub-free gives the first claim.
2. *Nothing for the public data.* Changing one record of $D_{\text{pub}}$ can change $\theta_0$ arbitrarily, and the fine-tuned model inherits whatever $\theta_0$ encodes. Training without privacy can memorize individual records ([Memorization and extraction](../part02-attacks/05-memorization-extraction.md)), and no noise was calibrated to that change.
3. *A record in both.* DP compares $D$ with $D$ without $x$, with $D_{\text{pub}}$ the same in both. So a test of whether $x$ was in $D$ obeys @thm-dpdef-testing: its true positive rate is at most $e^\varepsilon$ times its false positive rate plus $\delta$. A test of whether $x$ is in the training data at all can use the copy in $D_{\text{pub}}$, which nothing bounds.
:::

The first gap is overlap. Web-scale "public" datasets contain personal data, such as posts, profiles and leaked documents. A private set of medical notes or emails may overlap with them, and the people in both are not protected at all. Deduplicating against the public data protects copies, not people, since near-duplicates and paraphrases survive exact matching. Public is not the same as non-sensitive either, since a model can link public records in ways their publication did not allow. And large models are often fine-tuned by whoever has the compute, a trust question DP does not address ([Trust models](../part03-dp-foundations/09-trust-models.md)).

The second gap is distribution shift. Public data help as much as they resemble the private task. Benchmarks whose "private" set is close to well-curated public data can overstate what the method does on genuinely sensitive data, such as medical images, rare languages or internal logs. @tbl-pub-run used a public task close to the private one by design. [Exercise 5](../../book-exercises/part05-dp-ml/06-public-data.md#sec-sol-pub-5) shows the gain shrinking as the public teacher drifts away, and for the last layer turning into a loss against training from scratch. So when you report $\varepsilon$, say what the public data are, whether overlap with the private data was checked, and that the guarantee covers the private data given the pretrained model.

::: {#exm-pub-two-copies}
Take the guarantee at $\varepsilon = 1$ and $\delta = 10^{-5}$. Any test of whether a record was in the private fine-tuning set that has false positive rate 0.01 has true positive rate at most $e \times 0.01 + 10^{-5} = 0.0272$. If the same record is also in the public set, a membership test against the pretrained model, such as the likelihood-ratio attack of [Membership inference](../part02-attacks/02-membership-inference.md), is bounded by nothing. It may still fail in practice, but no theorem says it will. If instead the record appears twice in the private set, group privacy (@thm-dpprop-group) bounds the pair by $(2\varepsilon, 2e^\varepsilon\delta) = (2, 5.44 \times 10^{-5})$, a weaker but real guarantee. So a copy in the public set is worse than a duplicate in the private set.
:::

## Code {#sec-pub-code}

The script [code/public_data.py](../../code/part05-dp-ml/public_data.py) builds the public task, pretrains the student, implements the last-layer and low-rank-adapter versions of DP-SGD with their per-example gradients (ghost norms included), estimates the curvature and the public gradient subspace, and runs the comparison. It takes about four minutes on a laptop. From the repository root:

```bash
uv run python code/part05-dp-ml/public_data.py
```

| Experiment | Result |
| --- | --- |
| Public task | Teacher agreement 0.9004; pretrained student 0.9035 on the private task |
| Noise length at $\varepsilon = 1$ | 1.191, 0.473 and 0.166 for $d = 3{,}329$, 525 and 65 |
| Signal at the pretrained weights | 0.0130, 0.0207 and 0.0250; break-even 8,455, 523 and 44 steps |
| Curvature | Largest eigenvalue 0.723, trace 16.310; $d_{\text{eff}} = 451$ against 60.2 for the last layer at $\eta = 0.1$ |
| Public subspace | 60% of the private gradient's energy in one direction, 73% in 64 |
| Adapter ghost norms | Agree with explicit gradients to $3.3 \times 10^{-16}$ |
| Fine-tuning, $\varepsilon = 0.5$ | Last layer 0.9478, adapter 0.9464, full 0.9395, scratch 0.8589 |
| Fine-tuning, $\varepsilon = 1$ | Adapter 0.9616, full 0.9599, last layer 0.9497, scratch 0.9246 |

: The script's output with the default seed. {#tbl-pub-code}

[tests/test_public_data.py](../../tests/test_public_data.py) checks that the output still matches.

## Summary {#sec-pub-summary}

| Result | Statement | Use |
| --- | --- | --- |
| @lem-pub-noise | Squared length $ds^2$; a projection keeps $m$ independent coordinates | Noise length $\sigma C\sqrt d/B$: count the trainable weights first |
| @thm-pub-noise-cost | Cost $\approx\frac{\eta s^2}4d_{\text{eff}}$, $d_{\text{eff}} \le \min(d, 2\eta T\operatorname{tr}H)$, on a quadratic | Think in effective dimension: noise where the loss is flat is cheap; keep $\eta$ small |
| @lem-pub-free | A computation on public data is a fixed parameter of the mechanism | Pretraining and tuning on public data cost nothing |
| @prp-pub-probe | Convex ERM with $L \le \sqrt d$; feature gap plus privacy cost of order $\sqrt d/(n\varepsilon)$ | Best at very small $\varepsilon$; measure its ceiling on public data that resemble the private task |
| @prp-pub-lora | $r(d_{\text{in}} + d_{\text{out}})$ weights; outer-product gradients; first step of rank $r$ | At moderate budgets, the robust middle choice |
| @prp-pub-proj | Same $(\varepsilon, \delta)$; break-even $(s\sqrt m/(\sqrt{f_m}\Vert\bar g_D\Vert))^2$; bias $1 - f_m$ | Projected DP-SGD; the subspace must come from public data |
| @prp-pub-methods | Every strategy has the $\sigma$ of DP-SGD with the same $q$ and $T$ | A related public model is worth about a factor of eight in $\varepsilon$ here |
| @thm-pub-guarantee | $(\varepsilon, \delta)$-DP in $D$ for every fixed $D_{\text{pub}}$; nothing for $D_{\text{pub}}$ | Check for overlap, and say what the public data are when you report $\varepsilon$ |

: The results of this chapter, and where each is used. {#tbl-pub-summary}

## References {#sec-pub-references}

- Tramèr & Boneh, "Differentially Private Learning Needs Better Features (or Much More Data)" (ICLR 2021). Linear models on handcrafted ScatterNet features beat end-to-end private networks at moderate budgets; features from public data do better. The source of @sec-pub-features.
- Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang & Chen, "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022). The adapter of @sec-pub-lora, without privacy.
- Yu, Naik, Backurs, Gopi, Inan, Kamath, Kulkarni, Lee, Manoel, Wutschitz, Yekhanin & Zhang, "Differentially Private Fine-tuning of Language Models" (ICLR 2022). Private fine-tuning with LoRA and other parameter-efficient methods; reports, for example, 87.8% accuracy on a language-inference benchmark at $\varepsilon = 6.7$ against 90.2% without privacy, for a large RoBERTa model.
- Li, Tramèr, Liang & Hashimoto, "Large Language Models Can Be Strong Differentially Private Learners" (ICLR 2022). Full fine-tuning with the right hyperparameters matches parameter-efficient methods; ghost clipping for sequence models.
- Li, Liu, Hashimoto, Inan, Kulkarni, Lee & Thakurta, "When Does Differentially Private Learning Not Suffer in High Dimensions?" (NeurIPS 2022). Dimension-free bounds when gradients concentrate in few directions, and the measurement that fine-tuning gradients are dominated by a few principal components. The background of @sec-pub-dimension.
- Zhou, Wu & Banerjee, "Bypassing the Ambient Dimension: Private SGD with Gradient Subspace Identification" (ICLR 2021). Projected DP-SGD with a subspace estimated from public gradients.
- Yu, Zhang, Chen & Liu, "Do Not Let Privacy Overbill Utility: Gradient Embedding Perturbation for Private Learning" (ICLR 2021). Projection onto a public anchor subspace plus a separately noised residual.
- Tramèr, Kamath & Carlini, "Position: Considerations for Differentially Private Learning with Large-Scale Public Pretraining" (ICML 2024). What the public-pretraining paradigm does and does not protect, and how benchmarks can mislead. The source of @sec-pub-caveats.
- Bassily, Smith & Thakurta, "Private Empirical Risk Minimization: Efficient Algorithms and Tight Error Bounds" (FOCS 2014). The rates for private convex ERM used in @sec-pub-features.
- De, Berrada, Hayes, Smith & Balle, "Unlocking High-Accuracy Differentially Private Image Classification through Scale" (arXiv 2022). Private fine-tuning of image models pretrained on large public datasets, among other practices.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part05-dp-ml/06-public-data.md).
