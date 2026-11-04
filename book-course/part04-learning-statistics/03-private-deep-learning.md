# Private deep learning

Private stochastic gradient descent (SGD) in [the previous chapter](02-private-erm.md) assumes a convex, Lipschitz loss and has loose privacy constants. Neural-network losses generally satisfy neither assumption. This chapter adapts SGD to differential privacy (DP) by clipping each gradient, gives a tighter privacy accountant based on privacy-loss moments, and privatizes arbitrary learners by voting among models trained on disjoint data.

## Neural networks {#sec-dpdl-networks}

As in the previous chapter, each record contains a feature vector and label, and training minimizes total loss over parameter vector $\theta$:

$$
\mathcal L(\theta; D) = \sum_{i=1}^n \ell(\theta; x_i, y_i).
$$

Here a feature vector $x$ lies in $\mathbb R^{m_0}$, and a label is one of $K$ classes, $y \in \{1, \dots, K\}$. A neural network maps a feature vector to a score for each class. Training adjusts its parameters $\theta$. A multilayer perceptron stacks linear maps and fixed nonlinear functions.

::: {#def-dpdl-network name="Multilayer perceptron"}
A network with $J$ layers sets $v^{(0)} = x$ and, for $j = 1, \dots, J$,

$$
v^{(j)} = \phi\big(W^{(j)}v^{(j-1)} + c^{(j)}\big),
$$

where $W^{(j)}$ is an $m_j \times m_{j-1}$ matrix of weights, $c^{(j)} \in \mathbb R^{m_j}$ is a vector of offsets, and the **activation function** $\phi\colon \mathbb R \to \mathbb R$ is applied to each coordinate. The last layer has no activation and outputs $o \in \mathbb R^K$, one score per class. The parameter vector $\theta$ collects all weights and offsets.
:::

Without activations, the network is a single linear map. Common activations include the rectified linear unit $\phi(u)=\max(0,u)$ and bounded $\tanh(u)=(e^u-e^{-u})/(e^u+e^{-u})$.

The **softmax** maps scores to class probabilities, $\hat y_k=e^{o_k}/\sum_{j=1}^K e^{o_j}$. The predicted class maximizes $\hat y_k$. The **cross-entropy** loss is the negative log probability assigned to the true class:

$$
\ell(\theta; x, y) = -\ln \hat y_y = -\sum_{k=1}^K \mathbf 1[k = y]\,\ln \hat y_k ,
$$

where $\hat y$ depends on $\theta$ through every layer. The previous chapter's analysis does not apply: this loss is not convex in $\theta$, and its gradients are not bounded. In a two-layer network, the second-layer weights multiply the first-layer gradient and can be arbitrarily large. Thus one gradient can have unbounded sensitivity.

## Differentially private SGD {#sec-dpdl-dpsgd}

Two changes make noisy SGD practical. First, **clipping** limits each gradient's norm and directly bounds sensitivity. Second, each step uses a random batch, called a **lot**, formed by Poisson sampling.

::: {#def-dpdl-clip name="Clipping"}
For a clipping norm $C$, a positive number, the clipping of a vector $g$ is

$$
\mathrm{clip}_C(g) = g\cdot\min\left(1, \frac{C}{\|g\|_2}\right).
$$
:::

Each clipped vector has norm at most $C$.

DP-SGD clips each gradient in the lot, adds Gaussian noise to their sum, and divides by the expected lot size $qn$. The **noise multiplier** $\sigma$ is the noise standard deviation divided by the clipping norm $C$; unlike the previous chapter's $\sigma$, it is not itself a standard deviation.

::: {#def-dpdl-dpsgd name="DP-SGD"}
Let the parameter vector $\theta$ have $d$ coordinates. Fix a sampling rate $q$ in $(0, 1]$, a noise multiplier $\sigma > 0$, a clipping norm $C$, a number of steps $T$, learning rates $\eta_t$ and a start $\theta_0$. For $t = 1, \dots, T$:

1. Form the lot $B_t$ by including each index $i \in \{1,\dots,n\}$ independently with probability $q$.
2. Compute the noisy average of clipped gradients,
$$
\tilde g_t = \frac{1}{qn}\left(\sum_{i\in B_t}\mathrm{clip}_C\big(\nabla\ell(\theta_{t-1}; z_i)\big) + b_t\right), \qquad b_t \sim \mathcal N(0, \sigma^2C^2 I_d).
$$
3. Step: $\theta_t = \theta_{t-1} - \eta_t\,\tilde g_t$.

Output $\theta_T$.
:::

Unlike the previous chapter's private SGD, each step uses a lot with expected size $qn$. Clipping bounds each gradient by construction, and the noise is calibrated to $C$, the maximum contribution of one record.

::: {#lem-dpdl-clip}
Adding or removing one record changes the clipped sum $\sum_{i\in B}\mathrm{clip}_C(\nabla\ell(\theta; z_i))$ by a vector of norm at most $C$. Replacing one record changes it by at most $2C$.
:::

::: {.proof}
Adding or removing a record adds or removes one clipped vector, of norm at most $C$. A replacement does both, and the triangle inequality gives $2C$.
:::

Clipping biases the average gradient. Consequently, the earlier convergence theorem no longer applies even to convex losses, although clipped SGD works well in practice.

Privacy amplification depends on the sampling scheme. **Poisson sampling** includes each record independently with probability $q$, producing a random lot size. Even when $q=1/n$, a lot may be empty or contain multiple records, unlike a single-record sample. Fixed-size sampling without replacement requires a different analysis. The results below use Poisson sampling and add/remove neighbours, which differ by one record; @lem-dpdl-clip gives sensitivity $C$ for this relation.

Implementations often shuffle once and divide the data into fixed-size batches. These batches are dependent, so the Poisson-sampling analysis does not apply.

## Privacy accounting {#sec-dpdl-accounting}

Without subsampling, each DP-SGD step is a Gaussian mechanism with noise multiplier $\sigma$, so the previous chapter's analysis applies. If a step is $(\varepsilon_0,\delta_0)$-DP, Poisson subsampling gives approximately $(2q\varepsilon_0,q\delta_0)$-DP. Advanced composition over $T$ steps then gives

$$
\Big(O\big(q\varepsilon_0\sqrt{T\ln(1/\delta)}\big),\ Tq\delta_0 + \delta\Big)\text{-DP}.
$$

Abadi and coauthors derived a tighter bound for this algorithm, requiring less noise for the same guarantee.

::: {#thm-dpdl-abadi name="Moments accountant, asymptotic form"}
There are constants $c_1$ and $c_2$ such that, for every sampling rate $q$ and number of steps $T$, every $\varepsilon < c_1q^2T$ and every $\delta > 0$, DP-SGD is $(\varepsilon, \delta)$-DP under add/remove neighbours if

$$
\sigma \ge c_2\,\frac{q\sqrt{T\ln(1/\delta)}}{\varepsilon}.
$$
:::
<!-- proof: cited -->

Abadi and coauthors (2016, Theorem 1) prove this result; the rest of the section derives its ingredients. Compared with composition, the bound reduces $\sigma$ by a factor of order $\sqrt{\ln(1/\delta_0)}$ and avoids the accumulated failure probability $Tq\delta_0$, as @cor-dpdl-gaussian-composition shows. These factors can determine whether a privacy guarantee is useful.

The accountant starts from the privacy loss. For a mechanism $M$ whose output has density $p_D$ on dataset $D$, the **privacy loss** at an output $y$ is $\ln\big(p_D(y)/p_{D'}(y)\big)$. It is a random variable when $y$ is drawn from $M(D)$. By the privacy loss criterion of [the approximate DP chapter](../part02-dp-basics/03-approximate-dp.md), $(\varepsilon,\delta)$-DP follows if, for every ordered pair of neighbours, the loss exceeds $\varepsilon$ with probability at most $\delta$. But tail probabilities do not compose simply, and moments do, by addition. So the accountant takes, for an order $\lambda > 0$, the logarithm of the average $\lambda$-th power of the ratio $p_D(y)/p_{D'}(y)$, for the worst pair of neighbours.

::: {#def-dpdl-moments name="Log-moments of the privacy loss"}
Let $p_D$ be the density of the output $M(D)$. For $\lambda > 0$, the $\lambda$-th log-moment of $M$ is

$$
\kappa_M(\lambda) = \sup_{D \sim D'}\ \ln\ \mathbb E_{y\sim M(D)}\left[\left(\frac{p_D(y)}{p_{D'}(y)}\right)^{\lambda}\right].
$$
:::

Up to a factor, it is the Rényi divergence of order $\alpha = \lambda + 1$ between the two output distributions. A bound on it for every pair of neighbours is Rényi differential privacy (RDP). By Markov's inequality, a bound on one log-moment already gives approximate DP.

::: {#thm-dpdl-convert name="From moments to approximate DP"}
For every $\lambda > 0$ and $\varepsilon \ge 0$, the mechanism $M$ is $(\varepsilon, \delta)$-DP with

$$
\delta = \exp\big(\kappa_M(\lambda) - \lambda\varepsilon\big).
$$
:::

::: {.proof}
1. Fix neighbours $D \sim D'$ and write $L = \ln(p_D(Y)/p_{D'}(Y))$ for the privacy loss random variable, with $Y \sim M(D)$. The map $u \mapsto e^{\lambda u}$ is increasing, so $\Pr[L > \varepsilon] \le \Pr[e^{\lambda L} \ge e^{\lambda\varepsilon}]$.
2. Markov's inequality bounds the right side by $\mathbb E[e^{\lambda L}]\,e^{-\lambda\varepsilon}$, and $\mathbb E[e^{\lambda L}] \le e^{\kappa_M(\lambda)}$ by definition.
3. So the privacy loss exceeds $\varepsilon$ with probability at most $\delta$, and the privacy loss criterion gives $(\varepsilon,\delta)$-DP.
:::

The conversion is useful because log-moments are easy to track over many steps. The next lemma shows that they add up when mechanisms run in sequence, even when the outputs of the previous mechanisms choose the next one.

::: {#lem-dpdl-compose name="Composition of moments"}
Let $M_1$ have output $y_1$, and let $M_2(\cdot\,; y_1)$ be a mechanism chosen as a function of $y_1$ with $\kappa_{M_2(\cdot;y_1)}(\lambda) \le \kappa_2$ for every $y_1$. The mechanism $M$ that outputs $(y_1, y_2)$ satisfies $\kappa_M(\lambda) \le \kappa_{M_1}(\lambda) + \kappa_2$. By induction, $T$ adaptive steps have log-moments at most the sum of theirs.
:::

::: {.proof}
1. The density of $(y_1, y_2)$ on $D$ is $p_D(y_1)\,p_D(y_2 \mid y_1)$, so the ratio for neighbours factors as $r_1(y_1)\,r_2(y_2 \mid y_1)$. Here $r_1$ is the density ratio of $y_1$, and $r_2$ is the density ratio of $y_2$ given $y_1$.
2. The expectation over $y_2$ first, given $y_1$, gives
$$
\mathbb E\big[r_1^\lambda r_2^\lambda\big] = \mathbb E_{y_1}\Big[r_1(y_1)^\lambda\ \mathbb E_{y_2}\big[r_2(y_2\mid y_1)^\lambda \,\big|\, y_1\big]\Big] \le e^{\kappa_2}\,\mathbb E_{y_1}\big[r_1(y_1)^\lambda\big] \le e^{\kappa_2 + \kappa_{M_1}(\lambda)}.
$$
3. The logarithm and the supremum over neighbours give the claim.
:::

For the Gaussian mechanism the log-moments have a closed form.

::: {#lem-dpdl-gaussian}
Let $M(D) = f(D) + b$ with $b \sim \mathcal N(0, \sigma^2 I_d)$, where $f$ has $\ell_2$ sensitivity $\Delta_2 f$. Then for every $\lambda > 0$,

$$
\kappa_M(\lambda) = \frac{\lambda(\lambda + 1)(\Delta_2 f)^2}{2\sigma^2}.
$$
:::

::: {.proof}
1. Fix neighbours, write $\mu = f(D)$ and $\mu' = f(D')$, and let $r = \|\mu - \mu'\|_2 \le \Delta_2 f$. For $y = \mu + \sigma \xi$ with $\xi$ standard normal, the privacy loss is
$$
\frac{\|y - \mu'\|^2 - \|y - \mu\|^2}{2\sigma^2} = \frac{r^2}{2\sigma^2} + \frac{\langle \xi, \mu - \mu'\rangle}{\sigma}.
$$
2. So the privacy loss is normal with mean $m = r^2/(2\sigma^2)$ and variance $r^2/\sigma^2 = 2m$.
3. A normal variable $Z$ with mean $m$ and variance $s^2$ has $\mathbb E[e^{\lambda Z}] = e^{\lambda m + \lambda^2s^2/2}$. Here the exponent is $\lambda m + \lambda^2 m = \lambda(\lambda+1)r^2/(2\sigma^2)$.
4. This increases with $r$, and the supremum is reached at $r = \Delta_2 f$.
:::

The three results give a guarantee for many Gaussian steps without sampling. The log-moments of the steps add up, and the conversion turns their sum into $(\varepsilon,\delta)$-DP at the best order $\lambda$.

::: {#cor-dpdl-gaussian-composition}
Let $\rho = (\Delta_2 f)^2/(2\sigma^2)$. Any adaptive sequence of $T$ Gaussian mechanisms, each with sensitivity $\Delta_2 f$ and noise $\mathcal N(0,\sigma^2I_d)$, is $(\varepsilon,\delta)$-DP for every $\delta \in (0,1)$, with

$$
\varepsilon = T\rho + 2\sqrt{T\rho\ln(1/\delta)}.
$$
:::

::: {.proof}
1. By @lem-dpdl-gaussian and @lem-dpdl-compose, the sequence has $\kappa(\lambda) \le \lambda(\lambda+1)T\rho$.
2. By @thm-dpdl-convert, it is $(\varepsilon,\delta)$-DP whenever $\lambda(\lambda+1)T\rho - \lambda\varepsilon \le \ln\delta$, that is, whenever $\varepsilon \ge (\lambda+1)T\rho + \ln(1/\delta)/\lambda$.
3. The right side is smallest at $\lambda = \sqrt{\ln(1/\delta)/(T\rho)}$, where it equals $T\rho + 2\sqrt{T\rho\ln(1/\delta)}$.
:::

For comparison, take the same $T$ Gaussian steps with $\Delta_2 f = 1$, so that $T\rho = T/(2\sigma^2)$. Advanced composition first needs a guarantee per step. The Gaussian mechanism of [the approximate DP chapter](../part02-dp-basics/03-approximate-dp.md) gives $\varepsilon_0$ with $\sigma = \sqrt{2\ln(1/\delta_0) + \varepsilon_0}/\varepsilon_0$, so $\varepsilon_0$ is about $\sqrt{2\ln(1/\delta_0)}/\sigma$. Then it gives $\varepsilon \approx \varepsilon_0\sqrt{2T\ln(1/\delta)}$, which is $2\sqrt{T\ln(1/\delta_0)\ln(1/\delta)}/\sigma$. The corollary gives $\sqrt{2T\ln(1/\delta)}/\sigma$ plus a lower-order term. That is smaller by a factor of about $\sqrt{2\ln(1/\delta_0)}$, with no failure probability $T\delta_0$ to pay.

DP-SGD adds Poisson sampling to each step, and its log-moments come from a one-dimensional pair of distributions. By @lem-dpdl-clip, after a division by $C$, a step compares the noise distribution $\mathcal N(0, \sigma^2)$ with the mixture $(1-q)\,\mathcal N(0, \sigma^2) + q\,\mathcal N(1, \sigma^2)$. In the mixture, the extra record is sampled with probability $q$.

::: {#thm-dpdl-sampled name="Log-moments of the sampled Gaussian"}
For every integer $\lambda \ge 1$, one step of DP-SGD with sampling rate $q$ and noise multiplier $\sigma$ has, under add/remove neighbours,

$$
\kappa(\lambda) \le \ln A_{\lambda+1}, \qquad A_{a} = \sum_{k=0}^{a}\binom{a}{k}(1-q)^{a-k}q^k\exp\left(\frac{k^2 - k}{2\sigma^2}\right).
$$
:::
<!-- proof: cited -->

Mironov, Talwar and Zhang (2019, Section 3.3) prove the result. The binomial theorem expands the $a$th power of the mixture's density ratio and yields $A_a$. Each term is a Gaussian integral, $\mathbb E[e^{k\xi/\sigma^2}]=e^{k^2/(2\sigma^2)}$ for $\xi\sim\mathcal N(0,\sigma^2)$. The main step shows that this ordered pair is worst among all neighbouring pairs and both directions. For $\sigma\ge1$, small $q$, and moderate $\lambda$, the bound is $q^2\lambda(\lambda+1)/((1-q)\sigma^2)+O(q^3\lambda^3/\sigma^3)$, which implies @thm-dpdl-abadi.

These pieces together give the **moments accountant**. It computes the log-moment of one step for every $\lambda$, multiplies by $T$ by @lem-dpdl-compose, and converts with @thm-dpdl-convert at the best $\lambda$:

$$
\varepsilon(\delta) = \min_{\lambda \ge 1}\ \frac{T\ln A_{\lambda+1} + \ln(1/\delta)}{\lambda}.
$$ {#eq-dpdl-accountant}

The accountant depends only on $q$, $\sigma$, $T$, and $\delta$, not on the data. You can therefore set the noise or number of steps to meet a privacy target before training. @fig-dpdl-accounting compares it with composition for one setting.

![Privacy parameter $\varepsilon$ at $\delta = 10^{-5}$ of DP-SGD with sampling rate $q = 0.01$ and noise multiplier $\sigma = 4$, against the number of epochs of $1/q$ steps. Dashed: each step is $(\varepsilon_0,\delta_0)$-DP by @cor-dpdl-gaussian-composition with $T = 1$, Poisson sampling makes it $(\ln(1 + q(e^{\varepsilon_0}-1)), q\delta_0)$-DP, and advanced composition combines the steps, with $\delta$ split evenly. Solid: the moments accountant @eq-dpdl-accountant.](figures/dpdl-accounting.svg){#fig-dpdl-accounting}

::: {#exm-dpdl-accounting}
With $q = 0.01$, $\sigma = 4$ and $\delta = 10^{-5}$, @fig-dpdl-accounting shows the two accountings of the same algorithm. After $100$ epochs, or $T = 10{,}000$ steps, the composition route certifies $\varepsilon = 22.08$ and the moments accountant certifies $\varepsilon = 1.26$. After $400$ epochs the values are $61.41$ and $2.58$. The first is no guarantee at all, and the second is a reasonable one.
:::

These analyses use add/remove neighbours, whereas this book generally uses replace-one neighbours. A replacement is a removal and an addition, so conversion between the relations costs about twice $\varepsilon$.

## DP-SGD in practice {#sec-dpdl-practice}

DP-SGD differs from ordinary SGD in accuracy, architecture, and tuning.

**Accuracy.** On MNIST, a well-tuned DP-SGD model reached $98.1\%$ accuracy at $\varepsilon=2.93$ and $\delta=10^{-5}$, compared with about $99.8\%$ without privacy (Papernot and coauthors, 2020). On CIFAR-10, it reached $66.2\%$ at $\varepsilon=7.53$, a weaker result in both accuracy and privacy.

**Architecture.** Bounded activations such as $\tanh$ can outperform the rectified linear unit under DP-SGD. Network size also matters: the noise vector has expected squared norm $d\sigma^2C^2$, while the clipped-gradient sum has norm at most $qnC$. Thus the signal-to-noise ratio is at most about $qn/(\sigma\sqrt d)$ and decreases as the parameter count $d$ grows. Use a network large enough for the task, but no larger.

**Hyperparameters.** Tuning the learning rate, lot size, clipping norm, or epoch count on private data spends privacy budget. Ignoring this cost overstates the final model's privacy. Private selection can choose among many private runs at a small constant factor in $\varepsilon$ (Liu and Talwar). Public data can support tuning or pretraining, leaving fewer parameters to train privately.

## Private aggregation of teacher ensembles {#sec-dpdl-pate}

DP-SGD modifies one training algorithm. Private aggregation of teacher ensembles (PATE) treats the learner as a black box, so improvements to non-private learning also apply. PATE instead requires both a sensitive labelled training set and **public unlabelled** feature vectors.

PATE uses **sample and aggregate**: split the data into disjoint parts, train one non-private model on each part, then combine their outputs privately. Each record affects only one part, which limits the aggregation's sensitivity. PATE trains $m$ classifiers, called **teachers**, and aggregates them by voting.

For a public feature vector $x$, each teacher votes for a label. Let $c_k$ be the number of votes for label $k$. The **mode** $a$ is the most-voted label, with ties resolved in favour of the smallest label. The **gap** $\Gamma=c_a-\max_{k\ne a}c_k$ measures the margin over the next label. The private mode releases $a$ only when a noisy gap exceeds a threshold.

::: {#def-dpdl-mode name="Private mode of the teachers"}
Let $\varepsilon > 0$ and $\delta \in (0, 1/2)$. Split the dataset into $m$ fixed disjoint parts by position, train one teacher on each, and collect their votes on $x$. Let $a$ be the mode of the votes and $\Gamma$ its gap, as above. Draw $Z \sim \mathrm{Lap}(2/\varepsilon)$, Laplace noise with scale $2/\varepsilon$. If

$$
\Gamma + Z > \tau, \qquad \tau = 2 + \frac{2}{\varepsilon}\ln\frac{1}{2\delta},
$$

output the mode $a$. Otherwise output a symbol $\bot$ for "no answer", or a uniformly random label.
:::

When the votes agree, the mode is stable and can be released exactly. This is propose-test-release from [the chapter on beyond global sensitivity](../part03-query-release/03-beyond-global-sensitivity.md): privately test whether the data is far from one that changes the answer, then release the answer without noise if the test passes. The threshold here is lower so the failure probability is $\delta$, rather than $\delta/2$.

::: {#thm-dpdl-mode}
The private mode is $(\varepsilon,\delta)$-DP. The teachers' training algorithm may be randomized, with coins independent of everything else.
:::

::: {.proof}
1. Fix the teachers' coins; a guarantee for every fixed choice of coins implies one for random coins. Neighbours differ in one record, which lies in one part, so at most one teacher's vote changes. That moves one vote from one label to another, and changes each count by at most $1$.
2. *The gap has sensitivity $2$.* If the mode stays the same, both terms of $\Gamma$ change by at most $1$. If the mode changes from $a$ on $D$ to $a' \ne a$ on $D'$, then $c_{a'} + 1 \ge c'_{a'} \ge c'_a \ge c_a - 1$, so $\Gamma \le c_a - c_{a'} \le 2$, and likewise $\Gamma' \le 2$. In both cases $|\Gamma - \Gamma'| \le 2$. So, by the Laplace mechanism, the indicator that the test passes is $\varepsilon$-DP.
3. *Same mode.* The output is a function of the test indicator and of the common mode, so it is $\varepsilon$-DP by post-processing.
4. *Different modes.* By step 2, $\Gamma \le 2$. The test then passes on $D$ with probability at most $\Pr[Z > \tau - 2] = \tfrac12 e^{-\varepsilon(\tau-2)/2} = \delta$. Outside that event the output is $\bot$, so for every set $S$ of outputs,
$$
\Pr[M(D)\in S] \le \Pr[M(D) = \bot]\,\mathbf 1[\bot\in S] + \delta \le e^\varepsilon\Pr[M(D') = \bot]\,\mathbf 1[\bot\in S] + \delta \le e^\varepsilon\Pr[M(D')\in S] + \delta,
$$
where the middle step is the $\varepsilon$-DP of the test from step 2. Replacing $\bot$ by a random label is post-processing.
:::

Accuracy depends on teacher quality. For independent records and disjoint training subsets, teachers' votes on a fixed $x$ are independent.

::: {#prp-dpdl-mode-accuracy}
Let $\beta \in (0,1)$ be a failure probability, and let $\tau$ be the threshold of @def-dpdl-mode. Suppose each teacher, independently, votes for the correct label $y$ of $x$ with probability at least $3/4$. If

$$
m \ge \max\left(4\tau + \frac8\varepsilon\ln\frac1\beta,\ 32\ln\frac2\beta\right),
$$

then the private mode outputs $y$ with probability at least $1 - \beta$.
:::

::: {.proof}
1. Let $V$ be the number of correct votes. Its mean is at least $3m/4$, and Hoeffding's inequality for independent indicators gives $\Pr[V \le 5m/8] \le e^{-2m(1/8)^2} = e^{-m/32} \le \beta/2$.
2. If $V > 5m/8$, then $y$ has a majority, so it is the mode, and every other label has at most $m - V$ votes. So $\Gamma \ge 2V - m > m/4$.
3. The test then fails with probability $\Pr[Z \le \tau - \Gamma] \le \tfrac12 e^{-\varepsilon(m/4 - \tau)/2}$, because $m/4 > \tau$. The first lower bound on $m$ makes this at most $\beta/2$.
4. A union bound over the two failure events gives the claim.
:::

::: {#exm-dpdl-teachers}
With $\varepsilon = 1$, $\delta = 10^{-5}$ and $\beta = 0.01$, the threshold is $\tau = 2 + 2\ln(50{,}000) = 23.64$. The two lower bounds are $94.56 + 8\ln 100 = 131.4$ and $32\ln 200 = 169.5$, so $170$ teachers suffice. If a teacher needs $n_0$ records to be right three times in four, the private prediction needs $170\,n_0$.
:::

The required number of teachers grows as $\ln(1/\delta)/\varepsilon$. This sample-and-aggregate cost is about a factor $1/\varepsilon$ more data than the non-private algorithm requires, often more than a task-specific private method.

Answering $k$ queries with separate private-mode runs costs about $\varepsilon\sqrt{2k\ln(1/\delta')}$ by advanced composition. [Sparse vector](../part03-query-release/02-sparse-vector.md) is more efficient when most queries are stable: it pays only for answers above a threshold. Here, each query tests whether the next point's vote gap is small. The method therefore pays for unstable points, returning random labels for them, while releasing stable modes exactly. Its privacy cost grows with the square root of the number of unstable queries, rather than all $k$ (Bassily, Thakkar and Thakurta).

The labelled pairs $(x_1,\hat y_1),\dots,(x_k,\hat y_k)$ use public features and private labels. PATE trains a non-private **student** on them. If labeling is $(\varepsilon,\delta)$-DP and the features are independent of private data, the student inherits that guarantee by post-processing and can answer further queries without additional cost. PATE thus converts a bounded number of private predictions into a private model.

## Summary {#sec-dpdl-summary}

| Result | Statement |
| --- | --- |
| @lem-dpdl-clip | one record changes the clipped sum by at most $C$, or $2C$ under replacement |
| @thm-dpdl-abadi | DP-SGD is $(\varepsilon,\delta)$-DP with $\sigma = O\big(q\sqrt{T\ln(1/\delta)}/\varepsilon\big)$ |
| @thm-dpdl-convert | a log-moment bound $\kappa(\lambda)$ gives $(\varepsilon, e^{\kappa(\lambda)-\lambda\varepsilon})$-DP |
| @lem-dpdl-compose | log-moments add up under adaptive composition |
| @lem-dpdl-gaussian | the Gaussian mechanism has $\kappa(\lambda) = \lambda(\lambda+1)(\Delta_2 f)^2/(2\sigma^2)$ |
| @cor-dpdl-gaussian-composition | $T$ Gaussian steps give $\varepsilon = T\rho + 2\sqrt{T\rho\ln(1/\delta)}$ |
| @thm-dpdl-sampled | a sampled Gaussian step has $\kappa(\lambda) \le \ln A_{\lambda+1}$ |
| @eq-dpdl-accountant | the moments accountant for $T$ steps of DP-SGD |
| @thm-dpdl-mode | releasing the teachers' mode when its noisy gap exceeds $\tau$ is $(\varepsilon,\delta)$-DP |
| @prp-dpdl-mode-accuracy | $O(\ln(1/\delta)/\varepsilon + \ln(1/\beta)/\varepsilon)$ accurate teachers suffice |

: The chapter's results. {#tbl-dpdl-summary}

## References {#sec-dpdl-notes}

- Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar and Li Zhang, "Deep Learning with Differential Privacy", 2016, which introduces DP-SGD with clipping and Poisson sampling, and the moments accountant.
- Ilya Mironov, "Rényi Differential Privacy", 2017, the privacy notion built on the moments of the privacy loss, with its composition and conversion rules.
- Ilya Mironov, Kunal Talwar and Li Zhang, "Rényi Differential Privacy of the Sampled Gaussian Mechanism", 2019, which proves @thm-dpdl-sampled.
- Borja Balle, Gilles Barthe and Marco Gaboardi, "Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences", 2018, the amplification by Poisson sampling of @fig-dpdl-accounting.
- Nicolas Papernot, Abhradeep Thakurta, Shuang Song, Steve Chien and Úlfar Erlingsson, "Tempered Sigmoid Activations for Deep Learning with Differential Privacy", 2020, on bounded activations and the accuracy figures quoted above.
- Jingcheng Liu and Kunal Talwar, "Private Selection from Private Candidates", 2019, on hyperparameter selection at a constant factor in privacy.
- Kobbi Nissim, Sofya Raskhodnikova and Adam Smith, "Smooth Sensitivity and Sampling in Private Data Analysis", 2007, which introduces sample and aggregate.
- Nicolas Papernot, Martín Abadi, Úlfar Erlingsson, Ian Goodfellow and Kunal Talwar, "Semi-Supervised Knowledge Transfer for Deep Learning from Private Training Data", 2017, and Nicolas Papernot, Shuang Song, Ilya Mironov, Ananth Raghunathan, Kunal Talwar and Úlfar Erlingsson, "Scalable Private Learning with PATE", 2018, the PATE framework and its refinements.
- Raef Bassily, Om Thakkar and Abhradeep Thakurta, "Model-Agnostic Private Learning", 2018, the analysis of PATE with the sparse vector technique.
- Abhradeep Thakurta and Adam Smith, "Differentially Private Feature Selection via Stability Arguments, and the Robustness of the Lasso", 2013, on the release of a stable answer, as in @thm-dpdl-mode.
