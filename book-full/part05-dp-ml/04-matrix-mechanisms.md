# DP-FTRL and matrix factorization

Stochastic gradient descent (SGD) with differential privacy (DP), the DP-SGD of [the previous chapter](02-dp-sgd.md), owes much of its privacy to **Poisson sampling**. A record is in each batch only with probability $q$, and the accountant credits the run for that uncertainty. On the running example of this part, with $q = 0.01$, this amplification turns a noise multiplier of 1 into $\varepsilon = 6.71$ over 10,000 steps. But fixed batches, streaming pipelines and federated servers cannot honestly claim Poisson sampling, and the Poisson accountant then understates $\varepsilon$ ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md#sec-sub-schemes)). Without amplification, for $\varepsilon = 2$ over five passes, the noise multiplier grows from 0.928 to 4.458 and test accuracy falls from 0.9460 to 0.8941.

How can you train privately without random batches? The one idea is that training releases running sums of gradients, so noise added at one step can partly cancel the noise added before it. A matrix $A$ describes the release of all running sums, and a factorization $A = LR$ splits the release into an encoder $R$, which sets the privacy, and a decoder $L$, which sets the error. The first such algorithm ran the follow the regularized leader (FTRL) optimizer on sums noised through a binary tree, and is called DP-FTRL. That tree, the square-root factorization and the best known mechanisms are all choices of $L$ and $R$.

The running example is the part's network from [DP-SGD](02-dp-sgd.md), with $d = 3{,}329$ weights, $n = 10{,}000$ records, clipping norm $C = 0.5$ and $\delta = 10^{-5}$. The fixed-order runs cut one public random permutation, drawn independently of the data, into 100 batches of 100. So each record takes part once per epoch, 100 steps apart. @tbl-mfac-symbols lists the symbols.

| Symbol | Meaning |
| --- | --- |
| $T$, $t$; $E$ | number of steps and a step; number of epochs |
| $x_t$, stacked as the rows of $x$; $S_t$, $\hat S_t$ | clipped gradient sum of step $t$; its prefix sum $x_1 + \dots + x_t$, and the private estimate |
| $A$ | the $T \times T$ lower-triangular matrix of ones, so that $Ax$ stacks the prefix sums |
| $A = LR$; $r_j$, $\ell_t$ | a factorization into encoder $R$ and decoder $L$; column $j$ of $R$, row $t$ of $L$ |
| $z$; $s$ | Gaussian noise; its standard deviation in units of the clipping norm $C$ |
| $\mu$; $\mu(\varepsilon, \delta)$ | sensitivity over noise standard deviation; the largest $\mu$ whose Gaussian release is $(\varepsilon, \delta)$-DP |
| $\Delta_2 R$; $e_t$ | the encoder's sensitivity in units of $C$; the error of $\hat S_t$ per unit of privacy |
| $c_k$; $h_k$ | the coefficients of the square-root factorization and of its inverse |

: The symbols of this chapter. {#tbl-mfac-symbols}

## Training is a stream of prefix sums {#sec-mfac-prefix}

What does a training run have to release, and what does privacy cost when nothing is sampled? Gradient descent adds up its steps, so training is the private release of running sums, a classic problem with much better mechanisms than fresh noise on every step. Write $B$ for the batch size and $\eta$ for the learning rate.

::: {#prp-mfac-prefix name="Training releases prefix sums"}
Run gradient descent from $\theta_0$ with a constant learning rate $\eta$ and no momentum, $\theta_t = \theta_{t-1} - \frac\eta B\hat x_t$, where $\hat x_t$ is the possibly noisy clipped gradient sum of step $t$. Then
$$\theta_t = \theta_0 - \frac\eta B\hat S_t, \qquad \hat S_t = \sum_{s \le t}\hat x_s,$$ {#eq-mfac-weights}
so every model is post-processing of the prefix sums, and its noise is $\eta/B$ times theirs. The FTRL algorithm with linear losses $\langle\hat x_s, \theta\rangle$, $\theta_t = \arg\min_\theta\big(\langle\hat S_t, \theta\rangle + \frac B{2\eta}\|\theta - \theta_0\|^2\big)$, produces the same weights. Releasing all $T$ prefix sums is the linear workload
$$S = Ax, \qquad A = \begin{pmatrix}1 & & & \\ 1 & 1 & & \\ \vdots & & \ddots & \\ 1 & 1 & \cdots & 1\end{pmatrix} \in \mathbb R^{T \times T},$$ {#eq-mfac-workload}
where the $x_t$ are stacked as the rows of the $T \times d$ matrix $x$, and row $t$ of $Ax$ is $S_t$.
:::

::: {.proof}
1. *Unroll the recursion.* The updates $\theta_u - \theta_{u-1} = -\frac\eta B\hat x_u$ for $u = 1, \dots, t$ telescope when added, which gives @eq-mfac-weights.
2. *FTRL.* The FTRL objective is a linear term plus a quadratic. Its gradient is $\hat S_t + \frac B\eta(\theta - \theta_0)$, and setting it to zero gives $\theta = \theta_0 - \frac\eta B\hat S_t$, the same point.
3. *Post-processing.* Every model is a fixed function of the released prefix sums, and post-processing never weakens a guarantee (@thm-dpprop-post).
:::

Why release every prefix sum, and not only the last? Training is adaptive, because the gradient $x_t$ is computed at $\theta_{t-1}$, which depends on the earlier noisy prefix sums. So the intermediate models must be private too, and the workload is all $T$ rows of $A$, released one at a time as the stream arrives. With momentum or a learning-rate schedule the weights are still fixed linear combinations of the noisy gradients, with another lower-triangular matrix in place of $A$, which Denisov, McMahan, Rush, Smith & Thakurta (2022) factor too. This chapter keeps plain gradient descent.

Two facts from earlier chapters price every mechanism here. The first is about privacy. Release a matrix $f(D)$ plus independent normal noise of standard deviation $sC$ on every entry. Measure the change in $f$ between neighbours in the Frobenius norm, the length of the matrix read as one long vector. Then only the ratio $\mu$ of that change to $sC$ matters. The release is $(\varepsilon, \delta)$-DP exactly when $\delta$ is at least the Gaussian privacy profile of @thm-lg-analytic at $\mu$, and it satisfies $\mu$-[Gaussian differential privacy](../part04-privacy-accounting/04-gaussian-dp.md#sec-gdp-gdp) (GDP). The profile grows with $\mu$, so each target allows a largest ratio $\mu(\varepsilon, \delta)$. At $\delta = 10^{-5}$ these are $\mu(2, \delta) = 0.502$ and $\mu(8, \delta) = 1.666$.

The second fact is about error. A weighted sum of independent normals is normal, with variance the sum of the variances times the weights squared (@prp-dist-normal-sums). So if $z$ has independent entries of standard deviation $sC$ and $\ell$ is a fixed row vector, each coordinate of $\ell z$ is normal with variance
$$s^2C^2\|\ell\|^2 = s^2C^2\sum_j\ell_j^2.$$ {#eq-mfac-linear}

Neighbouring datasets $D \sim D'$ differ by one added or removed record, whose slot in the public order is then empty in one stream. Under replace-one neighbours, a contribution could change by $2C$, and every sensitivity would double. The two obvious mechanisms put the noise at opposite ends, on each step's gradient sum or on each prefix sum.

::: {#prp-mfac-naive name="The two naive mechanisms"}
Let a record take part in one step $\tau$, fixed in advance, and add a vector of length at most $C$ to $x_\tau$. Then each of the following is $\mu$-GDP.

- **Independent noise on each step**, $\hat S_t = \sum_{u \le t}(x_u + z_u)$ with noise of standard deviation $C/\mu$ per coordinate. The error of $\hat S_t$ has standard deviation $\sqrt t\,C/\mu$.
- **Independent noise on each prefix sum**, $\hat S_t = S_t + z_t$ with noise of standard deviation $\sqrt T\,C/\mu$ per coordinate, which is also the error of every $\hat S_t$.
:::

::: {.proof}
The plan is to find, for each mechanism, how far one record moves the noisy quantities, which fixes the noise by the first fact, and then to read the error off @eq-mfac-linear.

1. *Independent noise on each step: privacy.* The record changes only $x_\tau$, by a vector of length at most $C$, so the noisy gradients $x_u + z_u$ form one Gaussian mechanism with ratio $\mu$, and the prefix sums are post-processing of them.
2. *Its error.* The error is $\hat S_t - S_t = z_1 + \dots + z_t$, which is @eq-mfac-linear with $\ell = (1, \dots, 1, 0, \dots, 0)$, carrying $t$ ones. So its standard deviation is $\sqrt t\,C/\mu$.
3. *Independent noise on each prefix sum: privacy.* The record's gradient appears in every $S_t$ with $t \ge \tau$. So $T - \tau + 1$ of the released vectors move by the same vector, and the total Frobenius change is $\sqrt{T - \tau + 1}\,C$. The worst position, $\tau = 1$, gives $\sqrt T\,C$, and the noise must be $\sqrt T\,C/\mu$.
4. *Its error.* Each $\hat S_t$ carries one noise vector of that standard deviation.
:::

DP-SGD run in a fixed order without sampling is the first mechanism, so its model at step $t$ carries a random walk of $t$ noise vectors. Neither mechanism uses the structure of $A$. The first ignores that the noise accumulates, and the second ignores that most of $S_t$ was already released in $S_{t-1}$.

::: {#exm-mfac-one-epoch}
Take one epoch, $T = 100$ steps, and normalize $\mu = 1$ and $C = 1$. Independent noise on each step gives the final prefix sum an error standard deviation of $\sqrt{100} = 10$. Averaged over the 100 prefix sums, its root-mean-square error is
$$\sqrt{(1 + 2 + \dots + 100)/100} = \sqrt{50.5} = 7.11.$$
Independent noise on each prefix sum has error 10 on every one.

At $\varepsilon = 2$, $\mu = 0.502$, so the first mechanism needs a noise scale of $1/0.502 = 1.994$ per step, and its final prefix sum carries noise of standard deviation 19.94, both in units of $C$. DP-SGD with Poisson sampling at the same $\varepsilon$ needs only $\sigma = 0.827$, and its final prefix sum carries $0.827\sqrt{100} = 8.27$. So amplification buys a factor of 2.4 here. @sec-mfac-run shows that correlated noise can beat even that without any sampling.
:::

## The matrix mechanism {#sec-mfac-mechanism}

The two naive mechanisms put the noise on the inputs or on the answers. The **matrix mechanism** of Li, Hay, Rastogi, Miklau & McGregor (2010) puts it in between, on a linear transformation of the inputs from which a linear map recovers the answers. How private and how accurate is such a release, even on an adaptive stream, and how accurate can any such release be?

::: {#def-mfac-mechanism name="Matrix mechanism"}
A **factorization** of the prefix-sum matrix is a pair $A = LR$, with an **encoder** $R \in \mathbb R^{m \times T}$ and a **decoder** $L \in \mathbb R^{T \times m}$. Write $r_j$ for column $j$ of $R$ and $\ell_t$ for row $t$ of $L$. The matrix mechanism with noise scale $s$ releases
$$\hat S = L(Rx + z) = Ax + Lz, \qquad z_{jk} \sim \mathcal N(0, s^2C^2) \text{ independent}.$$ {#eq-mfac-mechanism}
It is **streaming** when $L$ and $R$ are lower-triangular, so that $\hat S_t$ needs only the inputs and noise up to step $t$.
:::

The encoder decides where each step's gradient lands, so it sets the privacy, and the decoder decides how much noise reaches each answer, so it sets the error.

::: {#thm-mfac-privacy name="Privacy and error of the matrix mechanism"}
Let a record take part in one step, at a position fixed in advance. Then the release @eq-mfac-mechanism is $\mu$-GDP with
$$\mu = \frac{\Delta_2 R}{s}, \qquad \Delta_2 R = \max_j\|r_j\|,$$ {#eq-mfac-privacy}
the largest Euclidean length of a column of $R$. If the factorization is streaming, this holds even when each $x_t$ is chosen after seeing $\hat S_1, \dots, \hat S_{t-1}$. Each coordinate of $\hat S_t - S_t$ is normal with mean zero and standard deviation $sC\|\ell_t\|$. At a fixed privacy level $\mu$ this is
$$\frac C\mu\,e_t, \qquad e_t = \Delta_2 R\,\|\ell_t\|,$$ {#eq-mfac-error}
and the **error per unit of privacy** $e_t$ does not change if $R$ is multiplied by any number $c > 0$ and $L$ divided by it.
:::

::: {.proof}
The plan is to compute how far one record moves the noisy encoding $Rx + z$, extend that to adaptive inputs, and then read the error off the decoder.

1. *Reduce to the encoding.* $L$ is fixed, so $\hat S$ is post-processing of $Rx + z$ (@thm-dpprop-post), and $LRx = Ax$ by the factorization.
2. *Sensitivity for fixed inputs.* Neighbouring streams differ in the record's contribution $g$ to $x_\tau$, a vector of length at most $C$, so row $\tau$ of $x$ changes by $g$ and nothing else. Then $Rx$ changes by $r_\tau g^\top$, column $\tau$ of $R$ times the row $g$. Its entries are $r_{j\tau}g_k$, so its squared Frobenius norm factorizes, and its Frobenius norm is $\|r_\tau\|\,\|g\| \le \|r_\tau\|C$. The worst position gives $\Delta_2 R\,C$, and with noise of standard deviation $sC$ the ratio is $\Delta_2 R/s$.
3. *Adaptive inputs.* With lower-triangular factors, row $j$ of $Rx + z$ depends on $x_1, \dots, x_j$ only, so it can be released at step $j$. Because $R$ is lower-triangular, rows before $\tau$ do not involve $x_\tau$, so the earlier outputs have the same distribution under both streams, and conditioning on them fixes $g$. Given the same history, the other records' gradients agree in both runs, so each later row of $Rx + z$ has its mean shifted by the matching row of $r_\tau g^\top$. These are Gaussian mechanisms released one after another, with squared shifts adding up to at most $\|r_\tau\|^2C^2$. Adaptive Gaussian steps compose into one Gaussian mechanism whose squared ratio is the sum of theirs ([Gaussian DP](../part04-privacy-accounting/04-gaussian-dp.md#sec-gdp-composition)), so the ratio is at most $\Delta_2 R/s$ again. Denisov et al. (2022, Theorem 2.1) prove the general statement, and @sec-apx-mat-streaming in the appendices proves the Gaussian case for several participations.
4. *Error.* The error is $\hat S_t - S_t = \ell_tz$, so @eq-mfac-linear gives standard deviation $sC\|\ell_t\|$. Substituting $s = \Delta_2 R/\mu$ from @eq-mfac-privacy gives @eq-mfac-error. Replacing $R$ by $cR$ and $L$ by $L/c$ multiplies $\Delta_2 R$ by $c$ and $\|\ell_t\|$ by $1/c$, so their product is unchanged.
:::

So privacy lives in the columns and error in the rows. Streaming factors cost nothing, because any factorization has a lower-triangular one with the same distribution of $\hat S$ (Denisov et al. 2022, Proposition 2.2), which @sec-apx-mat-streaming derives by Gram–Schmidt.

When $R$ is square and invertible, the mechanism is noisy SGD with correlated noise. Then $L = AR^{-1}$ and $\hat S = A(x + R^{-1}z)$, so the mechanism adds the noise $\zeta = R^{-1}z$ to the gradients and then sums. An implementation makes each $\zeta_t$ on the fly from the stored past noise.

How small can the error be? The squared row lengths of $L$ add up to its squared Frobenius norm $\|L\|_F^2$, so the root-mean-square error over the $T$ prefix sums is
$$\Big(\frac1T\sum_te_t^2\Big)^{1/2} = \frac{\Delta_2 R\,\|L\|_F}{\sqrt T}.$$
The bound below uses the **trace norm** $\|A\|_* = \sum_k\nu_k(A)$, the sum of the singular values $\nu_k(A)$ of $A$.

::: {#thm-mfac-lower name="The trace-norm lower bound"}
For every factorization $A = LR$,
$$\frac{\Delta_2 R\,\|L\|_F}{\sqrt T} \ge \frac{\|A\|_*}T.$$ {#eq-mfac-lower}
For the prefix-sum matrix, $\nu_k(A) = 1/\big(2\sin\frac{(2k - 1)\pi}{4T + 2}\big)$ for $k = 1, \dots, T$.
:::

::: {.proof}
The plan is to bound $\|A\|_*$ by the product $\|L\|_F\|R\|_F$, then bound $\|R\|_F$ by its longest column.

1. *Duality.* For any matrix $U$ whose largest singular value is at most 1, $\operatorname{tr}(U^\top A) \le \|A\|_*$, with equality for a $U$ built from the singular vectors of $A$. This dual description of the trace norm is proved in @sec-apx-mat-prefix in the appendices.
2. *Cauchy–Schwarz.* The trace $\operatorname{tr}(U^\top LR)$ is the Frobenius inner product of $L^\top U$ and $R$, so the Cauchy–Schwarz inequality gives
$$\operatorname{tr}(U^\top LR) \le \|L^\top U\|_F\,\|R\|_F.$$
3. *Remove $U$.* The matrix $U^\top$ does not lengthen any vector, since its singular values are at most 1. Applied to each column of $L$, this gives $\|L^\top U\|_F = \|U^\top L\|_F \le \|L\|_F$. With step 1, $\|L\|_F\|R\|_F \ge \|A\|_*$.
4. *Bound $\|R\|_F$ by the sensitivity.* $R$ has $T$ columns, each of length at most $\Delta_2 R$, so $\|R\|_F^2 \le T(\Delta_2 R)^2$. With step 3, this gives $\Delta_2 R\,\|L\|_F \ge \|A\|_*/\sqrt T$, which is @eq-mfac-lower after dividing by $\sqrt T$.
5. *The singular values.* The inverse $A^{-1}$ has 1 on the diagonal and $-1$ just below it. The eigenvalues of $(A^{-1})^\top A^{-1}$ are $4\sin^2\frac{(2k - 1)\pi}{4T + 2}$, because that product is the second-difference matrix with a 1 in its last diagonal entry, whose eigenvectors are sine waves; @sec-apx-mat-prefix derives them. The singular values of $A$ are the reciprocals of their square roots.
:::

Step 4 is tight only when all columns of $R$ are equally long, so unequal columns waste privacy on the short ones.

::: {#exm-mfac-gap}
@tbl-mfac-single collects the error per unit of privacy for one participation. At $T = 100$ the naive mechanisms have root-mean-square errors 7.11 and 10, while @eq-mfac-lower says no factorization can go below 2.18. The other rows are derived in @sec-mfac-tree and @sec-mfac-sqrt.

| Mechanism | $T = 100$: $\Delta_2 R$ | root-mean-square $e_t$ | max $e_t$ | $T = 10{,}000$: $\Delta_2 R$ | root-mean-square $e_t$ | max $e_t$ |
| --- | --- | --- | --- | --- | --- | --- |
| Independent noise ($R = I$) | 1 | 7.11 | 10.00 | 1 | 70.71 | 100.00 |
| Noise on prefix sums ($R = A$) | 10 | 10.00 | 10.00 | 100 | 100.00 | 100.00 |
| Binary tree | 2.828 | 5.05 | 6.93 | 3.873 | 9.84 | 13.96 |
| Square root | 1.591 | 2.37 | 2.53 | 2.000 | 3.84 | 4.00 |
| Optimal (Frobenius) | 1 | 2.24 | 2.45 | — | — | — |
| Lower bound @eq-mfac-lower | — | 2.18 | — | — | 3.63 | — |

: Error of each released prefix sum per unit of privacy ($\mu = 1$, $C = 1$), for one participation per record. The optimal factorization is normalized to columns of length 1 and was computed for $T \le 256$ only. {#tbl-mfac-single}

Over a hundredfold longer run, the independent-noise error grows tenfold, the tree's by a factor of 2 and the square root's by 1.6. So at $T = 10{,}000$, independent noise is 18 times worse than the square root.
:::

## Tree aggregation and DP-FTRL {#sec-mfac-tree}

The first mechanism with error far below $\sqrt T$ was the **binary tree**, which Dwork, Naor, Pitassi & Rothblum (2010) and Chan, Shi & Song (2010) found independently for counting events in a stream. Kairouz, McMahan, Song, Thakkar, Thakurta & Xu (2021) used it to release the prefix sums of gradients, with FTRL as the optimizer, and called the result **DP-FTRL**. How does the tree work as a factorization, and how good is it?

::: {#def-mfac-tree name="Binary tree mechanism"}
Build a binary tree over the steps $1, \dots, T$. Level 0 has one node per step, and a node at level $l$ covers one of the blocks $1, \dots, 2^l$, then $2^l + 1, \dots, 2\cdot2^l$, and so on, of $2^l$ consecutive steps. There are $H = \lceil\log_2T\rceil + 1$ levels. The mechanism releases each node's sum of gradients plus independent noise, and estimates each prefix sum as
$$\hat S_t = \sum_{\text{nodes in the decomposition of }t}(\text{node sum} + \text{noise}),$$
where the decomposition of $t$ takes one node for each 1 in the binary expansion of $t$.
:::

::: {#prp-mfac-tree name="Error of the tree"}
The binary tree is a streaming factorization of $A$, with
$$\Delta_2 R = \sqrt H, \qquad e_t = \sqrt H\sqrt{\operatorname{popcount}(t)},$$ {#eq-mfac-tree}
where $\operatorname{popcount}(t)$ is the number of 1s in the binary expansion of $t$, at most $H - 1$ for $t \le T$. Both factors grow like $\sqrt{\ln T}$, so the error grows like $\ln T$.
:::

::: {.proof}
The plan is to check the decomposition, write the tree as an encoder and decoder, and count.

1. *The decomposition.* Write $t$ in binary. Going from the highest bit down, each 1 at bit $l$ takes the next block of $2^l$ steps. That block starts right after the sum of the higher bits, which is a multiple of $2^{l+1}$, so it is a node of the tree. For example, $7 = 4 + 2 + 1$ uses the nodes covering steps 1 to 4, 5 and 6, and 7, while $8$ uses a single node covering 1 to 8.
2. *The factorization.* $R$ has one row per node, with 1 in the columns of the steps the node covers, so it is $m \times T$ with $m$ about $2T$. Row $t$ of $L$ has a 1 for each node of the decomposition of $t$. Then $LR = A$, because by step 1 those nodes cover each step $1, \dots, t$ exactly once.
3. *Sensitivity.* A step lies in exactly one node per level, so column $j$ of $R$ has $H$ ones and length $\sqrt H$.
4. *Error.* Row $t$ of $L$ has $\operatorname{popcount}(t)$ ones, so $\|\ell_t\| = \sqrt{\operatorname{popcount}(t)}$, and @eq-mfac-error gives @eq-mfac-tree.
5. *Streaming.* Each node can be released once its last step has arrived, and $\hat S_t$ uses only completed nodes. With the rows of $R$ ordered by completion time, the factors are streaming, so @thm-mfac-privacy applies to adaptive training.
:::

The tree sums at most $\log_2t + 1$ noise terms for the $t$-th prefix instead of $t$, but each is larger by the factor $\sqrt H$, because every step is hidden in $H$ nodes. That trade is a large win for long runs. For short runs $\sqrt H$ is not small next to $\sqrt t$, and the tree loses.

The tree is not the end. Its error jumps around, because $t = 127 = 1111111_2$ uses seven nodes and $t = 128$ uses one (@fig-mfac-errors, left). Every step also pays for being in top-level nodes that few prefixes use. The square-root factorization of the next section has smaller error everywhere.

::: {#exm-mfac-tree}
For $T = 100$ the tree has $H = 8$ levels, so $\Delta_2 R = \sqrt8 = 2.828$. The largest popcount up to 100 is 6, at $t = 63$, so the largest error per unit of privacy is $\sqrt8\sqrt6 = 6.93$. The root-mean-square error is 5.05 (@tbl-mfac-single). For $T = 10{,}000$ there are 15 levels and $\Delta_2 R = 3.873$. The maximum error is then 13.96 and the root-mean-square error 9.84, against 70.71 for independent noise.
:::

DP-FTRL runs FTRL, which by @eq-mfac-weights is gradient descent on noisy prefix sums, on the prefix sums that the tree releases. Kairouz et al. (2021) also add momentum and restart the tree after a number of steps. They report that DP-FTRL without amplification compares well with amplified DP-SGD. Xu et al. (2023) use it to train production language models for the Gboard keyboard in federated learning, where devices cannot be Poisson-sampled.

## The square-root factorization {#sec-mfac-sqrt}

The tree has a tall, thin encoder and a sparse decoder. A symmetric choice is $L = R$, a matrix whose square is $A$. Does $A$ have a lower-triangular square root, what does it look like, and how close does it come to the best possible factorization?

Power series find the square root. A lower-triangular **Toeplitz** matrix is constant along each diagonal, so its first column $(c_0, \dots, c_{T-1})$ fixes it. Identify that column with the series $c(u) = \sum_kc_ku^k$ in a formal variable $u$. The proof of the next theorem needs one fact, which turns a matrix equation into an equation between series.

::: {#lem-mfac-toeplitz name="Lower-triangular Toeplitz matrices multiply like power series"}
The product of the lower-triangular Toeplitz matrices of two series $a(u)$ and $b(u)$ is the lower-triangular Toeplitz matrix of $a(u)b(u)$, truncated after $u^{T-1}$.
:::

::: {.proof}
1. Entry $(i, k)$ of the first matrix is $a_{i-k}$ and entry $(k, j)$ of the second is $b_{k-j}$, both zero above the diagonal. So entry $(i, j)$ of the product is $\sum_ka_{i-k}b_{k-j}$ over $j \le k \le i$. This sum depends only on $i - j$, and it is the coefficient of $u^{i-j}$ in $a(u)b(u)$.
:::

::: {#thm-mfac-sqrt name="The square-root factorization"}
The lower-triangular matrix $R = A^{1/2}$ with $R^2 = A$ is the Toeplitz matrix with entries
$$R_{ij} = c_{i-j} \text{ for } i \ge j, \qquad c_k = \binom{2k}k4^{-k}: \quad c = 1, \tfrac12, \tfrac38, \tfrac5{16}, \tfrac{35}{128}, \dots,$$ {#eq-mfac-sqrt}
the coefficients of the power series $(1 - u)^{-1/2} = \sum_kc_ku^k$, computed by $c_0 = 1$ and $c_k = c_{k-1}\frac{2k - 1}{2k}$. With $L = R$,
$$(\Delta_2 R)^2 = \|\ell_T\|^2 = \sum_{k<T}c_k^2, \qquad 1 + \frac{\ln\frac{2T+1}3}\pi \le \sum_{k<T}c_k^2 \le 1 + \frac{0.8 + \ln\frac{4T - 3}5}\pi,$$ {#eq-mfac-sqrt-error}
and the largest error per unit of privacy is $\max_te_t = e_T = \sum_{k<T}c_k^2$, which grows like $\ln(T)/\pi$. As noisy SGD, the noise added to the gradients is $\zeta = R^{-1}z$, where $R^{-1}$ is the Toeplitz matrix of the coefficients $h_k$ of $(1 - u)^{1/2}$, $h = 1, -\frac12, -\frac18, -\frac1{16}, -\frac5{128}, \dots$.
:::

::: {.proof}
The plan is to write $A$ as a series, take its square root as a series, and then read the sensitivity and the error off the coefficients.

1. *$A$ is the series $1/(1 - u)$.* Its first column is all ones, and $1 + u + u^2 + \dots = 1/(1 - u)$.
2. *A square root.* The series $(1 - u)^{-1/2}$ squares to $1/(1 - u)$, so by @lem-mfac-toeplitz its Toeplitz matrix squares to $A$. Its coefficients come from the binomial series. The $k$-th one is
$$\frac{\frac12\cdot\frac32\cdots\frac{2k-1}2}{k!} = \frac{1\cdot3\cdots(2k - 1)}{2^kk!}.$$
Multiplying top and bottom by $2\cdot4\cdots2k = 2^kk!$ gives $\frac{(2k)!}{4^k(k!)^2} = \binom{2k}k4^{-k}$, and the ratio of consecutive terms is $\frac{2k-1}{2k}$.
3. *The inverse.* By @lem-mfac-toeplitz again, $R^{-1}$ is the matrix of $1/(1 - u)^{-1/2} = (1 - u)^{1/2}$. The binomial series gives its coefficients $1, -\frac12, -\frac18, -\frac1{16}, -\frac5{128}, \dots$, all negative after the first.
4. *Sensitivity and error.* Column $j$ of $R$ holds $c_0, \dots, c_{T-j}$, so the first column is the longest, with squared length $\sum_{k<T}c_k^2$. Row $t$ of $L = R$ holds $c_{t-1}, \dots, c_0$, so its squared length is $\sum_{k<t}c_k^2$, largest at $t = T$.
5. *Bounds.* A Wallis-type inequality gives $\frac1{\pi(k + 1/2)} \le c_k^2 \le \frac1{\pi(k + 1/4)}$ for $k \ge 1$. @sec-apx-mat-prefix in the appendices proves it, and since both bounding functions decrease, each sum compares with an integral. For the lower bound,
$$\sum_{k=1}^{T-1}\frac1{k + 1/2} \ge \int_1^T\frac{\mathrm dv}{v + 1/2} = \ln\frac{2T + 1}3.$$
For the upper bound, the first term is kept apart and the rest is bounded by an integral,
$$\sum_{k=1}^{T-1}\frac1{k + 1/4} \le \frac1{5/4} + \int_1^{T-1}\frac{\mathrm dv}{v + 1/4} = 0.8 + \ln\frac{4T - 3}5.$$
Multiplying both sums by $1/\pi$ and adding the $k = 0$ term, 1, gives @eq-mfac-sqrt-error.
:::

The noise added to step $t$'s gradient is $z_t - \frac12z_{t-1} - \frac18z_{t-2} - \dots$, fresh noise minus a decaying share of the noise already added. At the same noise scale, each step's noise is a little larger than independent noise, by the factor $\big(\sum_kh_k^2\big)^{1/2}$. The script sums 100,000 coefficients and finds $\sum_kh_k^2 = 1.2732 = 4/\pi$, so the factor tends to 1.128. But the negative coefficients pull the accumulated noise back toward zero, so the running sums no longer drift (@fig-mfac-noise-paths).

The square root is close to optimal. Against the lower bound of 2.18 at $T = 100$ and 3.63 at $T = 10{,}000$, its root-mean-square error is 2.37 and 3.84, within 9% and 6%, and @sec-apx-mat-prefix proves that the ratio tends to 1. Fichtenberger, Henzinger & Upadhyay (2023) and Henzinger, Upadhyay & Upadhyay (2023) bound its maximum and mean squared errors. A finer expansion, which this chapter does not prove, gives $\sum_{k<T}c_k^2 \approx (\ln T + \gamma + 4\ln 2)/\pi$ with $\gamma$ Euler's constant, which is 2.532 at $T = 100$ against the exact 2.531.

![Left: one coordinate of the noise added to each of the first 60 steps' gradients, independent (identity factorization) and square-root correlated, both calibrated to $\mu = 1$ for one participation. Right: the noise in the prefix sums over 1,000 steps, for independent noise, the binary tree and the square root, with bands of two standard deviations. Independent noise drifts like a random walk, and the square root's stays within a narrow band.](figures/mfac_noise_paths.svg){#fig-mfac-noise-paths}

::: {#exm-mfac-sqrt}
At $T = 4$ the coefficients are $c = (1, \frac12, \frac38, \frac5{16})$, so
$$R = \begin{pmatrix}1 & & & \\ \frac12 & 1 & & \\ \frac38 & \frac12 & 1 & \\ \frac5{16} & \frac38 & \frac12 & 1\end{pmatrix}.$$
Row 4 times column 1 of $R$ is $\frac5{16} + \frac38\cdot\frac12 + \frac12\cdot\frac38 + \frac5{16} = 1$, as $R^2 = A$ requires. The first column has squared length
$$1 + \frac14 + \frac9{64} + \frac{25}{256} = \frac{381}{256} = 1.488,$$
so $\Delta_2 R = 1.220$, and the last prefix sum has error 1.488 per unit of privacy, against $\sqrt4 = 2$ for independent noise. At $T = 100$ the same computation gives $\sum_kc_k^2 = 2.531$, so $\Delta_2 R = 1.591$. The final prefix sum's error is then 2.53, a quarter of independent noise's 10.
:::

### The optimal factorization

How much better than the square root can any factorization do? By scale invariance the best encoder can have columns of length at most 1, and it then minimizes $\|L\|_F$. Write $X = R^\top R$ for the inner products of the columns of $R$. For a given $X$ the best decoder, the least-squares solution of $LR = A$, gives $\|L\|_F^2 = \operatorname{tr}(A^\top AX^{-1})$, which is convex in $X$.

::: {#thm-mfac-optimal name="The optimal factorization"}
The problem of minimizing $\operatorname{tr}(A^\top AX^{-1})$ over positive definite $X$ with diagonal at most 1 has the solution
$$X = V^{-1/2}(V^{1/2}A^\top AV^{1/2})^{1/2}V^{-1/2},$$
where $V$ is the diagonal matrix of a positive vector $v$ that is a fixed point of $v \mapsto \operatorname{diag}\big((V^{1/2}A^\top AV^{1/2})^{1/2}\big)$. The optimal value is $\sum_iv_i$. In words, the optimal $X$ is the matrix square root of $A^\top A$, rescaled on both sides by a diagonal matrix chosen so that every column of $R$ has length exactly 1.
:::
<!-- proof: cited -->

The proof is Denisov et al. (2022, Theorem 3.2), who iterate the fixed-point map to find $v$, and @sec-apx-mat-frobenius in the appendices gives it in full. A lower-triangular $R$ with $R^\top R = X$ comes from a Cholesky factorization of $X$ with rows and columns reversed (Denisov et al. 2022).

The script's iteration converges in 155 steps at $T = 100$, to a root-mean-square error of 2.24 against 2.37 for the square root, and in 193 steps at $T = 256$, to 2.52 against 2.67. Each iteration costs cubic time, so the square root, free and nearly as good, is the usual default for long runs. @fig-mfac-errors shows the errors across one run and against the horizon, and @fig-mfac-matrices shows that the square root and the optimal factors look alike at $T = 16$.

![Left: standard deviation of the error of each released prefix sum $S_t$ for $T = 256$, one participation, at $\mu = 1$ and $C = 1$, for five factorizations. Right: the root-mean-square error over a run against its length $T$, with the trace-norm lower bound of @eq-mfac-lower dotted. The optimal factorization is computed up to $T = 256$.](figures/mfac_errors.svg){#fig-mfac-errors}

![The encoder $R$ (top) and decoder $L$ (bottom) of three factorizations of the $16 \times 16$ prefix-sum matrix. The tree's encoder has one row per node (31 rows), and its decoder picks the nodes of each prefix. The square root uses the same matrix for both, and the optimal factorization's encoder has columns of length exactly 1.](figures/mfac_matrices.svg){#fig-mfac-matrices}

## Several participations {#sec-mfac-epochs}

So far each record took part in one step. Training usually makes several passes, and then a record's gradient enters $x$ at several steps, each time through a different column of $R$. How large is the sensitivity then, and how should the factorization change to keep it small? The answer depends on which steps a record can occupy and on how the columns of $R$ overlap.

::: {#def-mfac-banded name="Participation pattern, separation and band"}
A record's **participation pattern** is the set of steps $\tau_1 < \dots < \tau_k$ at which it contributes, with vectors $g_1, \dots, g_k$ of length at most $C$, possibly chosen adaptively. The pattern has **$b$-min-separation** if any two participations are at least $b$ steps apart. A lower-triangular encoder is **$b$-banded** if its nonzero entries lie on the main diagonal and the $b - 1$ diagonals below it, that is $R_{ij} = 0$ whenever $i - j \ge b$.
:::

::: {#thm-mfac-multi name="Sensitivity under several participations"}
For a record with participation pattern $\tau_1, \dots, \tau_k$, the change in $Rx$ is $\sum_ir_{\tau_i}g_i^\top$. If $R^\top R \ge 0$ entrywise, the sensitivity in units of $C$ is
$$(\Delta_2 R)^2 = \frac1{C^2}\max_{\text{patterns}}\ \sup_{\|g_i\| \le C}\Big\|\sum_ir_{\tau_i}g_i^\top\Big\|_F^2 = \max_{\text{patterns}}\Big\|\sum_ir_{\tau_i}\Big\|^2,$$ {#eq-mfac-multi}
the largest length of a sum of participating columns, and the release is $\mu$-GDP with $\mu = \Delta_2 R/s$. For independent noise, $R = I$, this gives $\Delta_2 R = \sqrt k$. For a $b$-banded encoder and patterns with $b$-min-separation, whatever the sign of the entries of $R^\top R$,
$$\Delta_2 R = \Big(\sum_i\|r_{\tau_i}\|^2\Big)^{1/2} \le \sqrt k\max_j\|r_j\|.$$
:::

::: {.proof}
The plan is to expand the squared norm of the change into inner products, find the worst contributions, and evaluate the two special cases.

1. *The change in $Rx$.* Row $\tau_i$ of $x$ changes by $g_i$, so $Rx$ changes by $\sum_ir_{\tau_i}g_i^\top$, a sum of the column-times-row products of @thm-mfac-privacy.
2. *Expand the norm.* The Frobenius inner product of two such products, $ab^\top$ and $a'b'^\top$, is $(a\cdot a')(b\cdot b')$. So
$$\Big\|\sum_ir_{\tau_i}g_i^\top\Big\|_F^2 = \sum_{i,j}(r_{\tau_i}\cdot r_{\tau_j})(g_i\cdot g_j).$$
3. *The worst contributions.* Each $g_i\cdot g_j$ lies in $[-C^2, C^2]$. The entries of $R^\top R$ are the inner products of columns, so if they are nonnegative, the sum is largest when every $g_i\cdot g_j = C^2$, that is when all $g_i$ are the same vector of length $C$. The sum is then $C^2\|\sum_ir_{\tau_i}\|^2$. The condition holds for the identity, the tree and the square root, because their $R$ has nonnegative entries. Adaptive choices of the $g_i$ cannot exceed this supremum. Denisov et al. (2022, Theorem 2.1) state their adaptive result for exactly this sensitivity, and @sec-apx-mat-streaming in the appendices proves it for Gaussian noise and a lower-triangular $R$.
4. *Independent noise.* The columns of $I$ are orthonormal, so the squared length of a sum of $k$ of them is $k$.
5. *Banded encoders.* Column $\tau$ of a $b$-banded $R$ is nonzero only in rows $\tau, \dots, \tau + b - 1$. Participations at least $b$ apart have columns with disjoint supports, so the cross terms of step 2 vanish whatever the $g_i$, leaving $\sum_i\|r_{\tau_i}\|^2\|g_i\|^2$.
:::

Correlated noise makes repeated participation more expensive. Under independent noise, $k$ participations cost $\sqrt k$, as Gaussian composition says. Under the square root, the columns of nearby steps overlap and point in similar directions. So a record's contributions pile up, and the sensitivity grows faster than $\sqrt k$. Under the tree it is worse still, because a top node holds several participations of the same record. For the optimal factorization, @eq-mfac-multi applies only if $X$ has no negative entries. Banding restores $\sqrt k$: it cuts the square root's encoder to its first $b$ diagonals, which removes the overlaps. Its decoder $L = AR^{-1}$ is less accurate for one participation, but with many the saving in sensitivity is larger. Choquette-Choo and coauthors (2023) optimize factorizations under a participation constraint, and add amplification to banded ones.

So the data order is part of the mechanism. A fixed permutation repeated every epoch guarantees a separation of exactly $n/B$ steps. A new shuffle every epoch can put a record's two participations one step apart across an epoch boundary, which breaks the banded bound. The order must be fixed independently of the data, but it need not be random.

::: {#exm-mfac-epochs}
@tbl-mfac-epochs takes the running example's order, with separation $b = 100$, so over $E$ epochs every record takes part $E$ times. The banded encoder keeps the first 100 coefficients of the square root. Each cell gives the sensitivity and then $e_T$ of @eq-mfac-error with that sensitivity.

| Epochs $E$ | Independent: $\Delta_2 R$ / final $e_T$ | Tree | Square root | Banded square root |
| --- | --- | --- | --- | --- |
| 1 | 1.000 / 10.00 | 2.828 / 4.90 | 1.591 / 2.53 | 1.591 / 2.53 |
| 2 | 1.414 / 20.00 | 4.690 / 8.12 | 2.531 / 4.20 | 2.250 / 4.16 |
| 5 | 2.236 / 50.00 | 8.944 / 21.91 | 5.080 / 8.86 | 3.558 / 8.56 |
| 10 | 3.162 / 100.00 | 16.186 / 39.65 | 9.154 / 16.54 | 5.031 / 15.69 |
| 20 | 4.472 / 200.00 | 30.430 / 74.54 | 17.191 / 32.09 | 7.115 / 29.85 |

: Sensitivity (in units of $C$) and error of the final prefix sum per unit of privacy, for $E$ participations 100 steps apart, $T = 100E$. {#tbl-mfac-epochs}

The banded sensitivity at 20 epochs is $\sqrt{20}\times1.591 = 7.115$, as step 5 of the proof of @thm-mfac-multi predicts. The full square root's is 17.191, more than twice as large, because its long columns overlap. But the final errors are close, 29.85 against 32.09, because the banded decoder is less accurate. Both are between a sixth and a seventh of independent noise's 200.
:::

## The running example: matrix mechanisms against DP-SGD {#sec-mfac-run}

The previous sections compared mechanisms by the error in released sums. Does that ranking carry over to trained networks, and how do mechanisms without sampling compare with DP-SGD with sampling at the same $\varepsilon$? The comparison needs one number per method, computed before training: the noise its final model carries.

::: {#prp-mfac-noise name="Noise at equal privacy"}
Fix a target $(\varepsilon, \delta)$ and $E$ participations 100 steps apart, so $T = 100E$. A fixed-order mechanism with encoder $R$ and the multi-epoch sensitivity $\Delta_2 R$ of @thm-mfac-multi needs the noise scale
$$s = \frac{\Delta_2 R}{\mu(\varepsilon, \delta)},$$
and its final prefix sum carries noise of standard deviation $s\|\ell_T\|$ per coordinate, in units of $C$. For independent noise this is $s = \sqrt E/\mu(\varepsilon, \delta)$. DP-SGD with noise multiplier $\sigma$ carries $\sigma\sqrt T$ in its final prefix sum.
:::

::: {.proof}
1. *Noise scale.* By @thm-mfac-multi the release is $\mu$-GDP with $\mu = \Delta_2 R/s$, and it meets the target exactly when $\mu \le \mu(\varepsilon, \delta)$. The smallest such $s$ is the one stated, and for $R = I$ the sensitivity is $\sqrt E$.
2. *Final prefix noise.* By @thm-mfac-privacy the error of $\hat S_T$ has standard deviation $sC\|\ell_T\|$, which is $s\|\ell_T\|$ in units of $C$.
3. *DP-SGD.* Its final prefix sum is a sum of $T$ independent noise vectors of standard deviation $\sigma$ in units of $C$, so by @eq-mfac-linear it carries $\sigma\sqrt T$.
:::

For DP-SGD, the Rényi differential privacy (RDP) accountant of [DP-SGD](02-dp-sgd.md) finds the smallest $\sigma$ for $q = 0.01$, $T$ steps and the target $\varepsilon$. Training then runs @eq-mfac-weights with each factorization's noise added to the gradients. For the square-root and banded encoders that noise is $R^{-1}z$, a convolution in time. For the tree it is the difference of consecutive tree estimates.

::: {#exm-mfac-run}
@tbl-mfac-noise gives the noise each method needs for $\varepsilon = 2$ and 8 at $\delta = 10^{-5}$, and the noise its final prefix sum carries. @tbl-mfac-run gives the test accuracy at the better of two learning rates, 0.5 and 2, averaged over three runs.

| $\varepsilon$, epochs | DP-SGD $\sigma$ | Independent $s$ | Tree $s$ | Square root $s$ | Banded $s$ | Final prefix noise: DP-SGD / independent / tree / square root / banded |
| --- | --- | --- | --- | --- | --- | --- |
| 2, 1 | 0.827 | 1.994 | 5.639 | 3.172 | 3.172 | 8.27 / 19.94 / 9.77 / 5.05 / 5.05 |
| 2, 5 | 0.928 | 4.458 | 17.833 | 10.129 | 7.093 | 20.75 / 99.69 / 43.68 / 17.67 / 17.06 |
| 2, 10 | 1.03 | 6.305 | 32.273 | 18.251 | 10.031 | 32.57 / 199.38 / 79.05 / 32.98 / 31.28 |
| 8, 1 | 0.501 | 0.600 | 1.698 | 0.955 | 0.955 | 5.01 / 6.00 / 2.94 / 1.52 / 1.52 |
| 8, 5 | 0.573 | 1.342 | 5.369 | 3.049 | 2.135 | 12.81 / 30.01 / 13.15 / 5.32 / 5.14 |
| 8, 10 | 0.616 | 1.898 | 9.716 | 5.495 | 3.020 | 19.48 / 60.02 / 23.80 / 9.93 / 9.42 |

: Noise needed by each method at equal $\varepsilon$ ($\delta = 10^{-5}$, 100 steps per epoch) and the resulting noise in the final prefix sum, all in units of $C$. DP-SGD uses Poisson sampling with $q = 0.01$ and the RDP accountant, and the others use a fixed order and exact Gaussian accounting. {#tbl-mfac-noise}

| $\varepsilon$, epochs | DP-SGD (Poisson) | Independent | Tree | Square root | Banded square root |
| --- | --- | --- | --- | --- | --- |
| 2, 1 | 0.9343 ± 0.0044 | 0.9066 ± 0.0031 | 0.9134 ± 0.0038 | 0.9361 ± 0.0046 | 0.9358 ± 0.0019 |
| 2, 5 | 0.9460 ± 0.0021 | 0.8941 ± 0.0071 | 0.9183 ± 0.0084 | 0.9430 ± 0.0011 | 0.9456 ± 0.0019 |
| 2, 10 | 0.9493 ± 0.0012 | 0.8641 ± 0.0130 | 0.9096 ± 0.0074 | 0.9398 ± 0.0029 | 0.9491 ± 0.0024 |
| 8, 1 | 0.9371 ± 0.0016 | 0.9395 ± 0.0018 | 0.9412 ± 0.0006 | 0.9461 ± 0.0022 | 0.9434 ± 0.0017 |
| 8, 5 | 0.9524 ± 0.0026 | 0.9420 ± 0.0029 | 0.9461 ± 0.0012 | 0.9497 ± 0.0025 | 0.9542 ± 0.0014 |
| 8, 10 | 0.9520 ± 0.0006 | 0.9410 ± 0.0032 | 0.9484 ± 0.0004 | 0.9530 ± 0.0015 | 0.9549 ± 0.0024 |

: Test accuracy (mean ± standard deviation over three runs) at equal $\varepsilon$, $\delta = 10^{-5}$, batch 100, clipping norm 0.5, best of learning rates 0.5 and 2. Without noise, fixed-order training with clipping reaches 0.9420, 0.9580 and 0.9606 after 1, 5 and 10 epochs. {#tbl-mfac-run}

At $\varepsilon = 2$, fixed-order SGD with independent noise loses 2.8 points to DP-SGD after one epoch and 8.5 points after ten. It gets worse with more epochs because its noise scale grows like $\sqrt E$, from 1.994 to 6.305. The tree recovers between a quarter and a half of the gap, and the square-root encoders nearly all of it. At ten epochs the banded encoder reaches 0.9491, within 0.0002 of DP-SGD.
:::

Matrix mechanisms can even beat amplified DP-SGD for short runs at large $\varepsilon$, where amplification gains little. At $\varepsilon = 8$ and one epoch, the square root's final prefix sum carries 1.52 against DP-SGD's 5.01, and it wins outright. DP-SGD keeps its lead at small $\varepsilon$ over many epochs. There its noise multiplier grows only from 0.827 to 1.03, while every fixed-order mechanism pays for ten participations, and only banding keeps up. In every row, the method with the least final-prefix noise is the most accurate, or within the run-to-run spread of it.

The comparison is slightly tilted toward the fixed-order methods. DP-SGD uses the RDP accountant, which overstates $\varepsilon$ and so gives it more noise than it needs ([Choosing an accountant](../part04-privacy-accounting/09-choosing-an-accountant.md)), while the others use the exact Gaussian curve. A learning rate chosen by test accuracy is itself a leak ([Hyperparameter tuning](03-hyperparameter-tuning.md)), and this experiment does it only to compare methods at their best. The runs are short, but the gaps of @tbl-mfac-epochs widen with more epochs. @fig-mfac-accuracy shows the same pattern across budgets at five epochs.

![Test accuracy of the running example after 5 epochs (500 steps of batch 100) at learning rate 1, against $\varepsilon$ at $\delta = 10^{-5}$, one run per point. DP-SGD uses Poisson sampling, and the other four use a fixed data order and differ only in how their noise is correlated.](figures/mfac_accuracy.svg){#fig-mfac-accuracy}

A federated server cannot sample devices with a known rate. But if it can make sure that a device contributes at most once every $b$ rounds, the banded matrix mechanism gives a provable guarantee. At the running example's scale it is as accurate as DP-SGD with sampling would be. The privacy unit there is a user, which [User-level differential privacy](07-user-level.md) treats.

## Code {#sec-mfac-code}

The script [code/matrix_mechanisms.py](../../code/part05-dp-ml/matrix_mechanisms.py) builds the identity, prefix, tree, square-root and optimal factorizations and computes their sensitivities and errors. Its structured formulas agree with dense matrices at $T = 256$ to a relative difference of $3.3 \times 10^{-16}$. It also computes multi-epoch sensitivities, and trains the running example with each mechanism's noise, against DP-SGD from [code/dp_sgd.py](../../code/part05-dp-ml/dp_sgd.py). From the repository root:

```bash
uv run python code/part05-dp-ml/matrix_mechanisms.py
```

| Experiment | Result |
| --- | --- |
| Square-root coefficients and inverse | $c = 1, 0.5, 0.375, 0.3125, \dots$; $h = 1, -0.5, -0.125, -0.0625, \dots$; $R^2 = A$ exactly at $T = 4$ |
| One participation, $T = 100$ | Root-mean-square error 7.11 (independent), 5.05 (tree), 2.37 (square root), 2.24 (optimal), bound 2.18 |
| One participation, $T = 10{,}000$ | 70.71, 9.84, 3.84; bound 3.63 |
| 20 epochs, sensitivity | 4.472 (independent), 30.430 (tree), 17.191 (square root), 7.115 (banded) |
| Accuracy, $\varepsilon = 2$, 10 epochs | DP-SGD 0.9493, independent 0.8641, tree 0.9096, square root 0.9398, banded 0.9491 |
| Accuracy, $\varepsilon = 8$, 1 epoch | DP-SGD 0.9371, independent 0.9395, square root 0.9461 |

: The script's output with the default seed. {#tbl-mfac-code}

[tests/test_matrix_mechanisms.py](../../tests/test_matrix_mechanisms.py) checks that the output still matches. The script takes about two minutes of processor time, most of it for the 198 training runs behind @tbl-mfac-run.

## Summary {#sec-mfac-summary}

| Result | Statement | Use |
| --- | --- | --- |
| @prp-mfac-prefix | $\theta_t = \theta_0 - \frac\eta B\hat S_t$; the workload is $Ax$ | Every model is post-processing of the prefix sums |
| @prp-mfac-naive | Noise per step: error $\sqrt t\,C/\mu$; noise per prefix sum: $\sqrt T\,C/\mu$ | The first is DP-SGD without amplification |
| @def-mfac-mechanism | $\hat S = L(Rx + z) = Ax + Lz$ | With $R$ invertible, noisy SGD with noise $R^{-1}z$ |
| @thm-mfac-privacy | $\mu = \Delta_2 R/s$ with $\Delta_2 R = \max_j\Vert r_j\Vert$; error $\frac C\mu\Delta_2 R\Vert\ell_t\Vert$ | Comparing correlated-noise schemes, also on adaptive streams |
| @thm-mfac-lower | $\Delta_2 R\,\Vert L\Vert_F/\sqrt T \ge \Vert A\Vert_*/T$ | No factorization beats it; 2.18 at $T = 100$ |
| @prp-mfac-tree | $\Delta_2 R = \sqrt H$, $e_t = \sqrt H\sqrt{\operatorname{popcount}(t)}$ | DP-FTRL; worth it only beyond a few dozen steps |
| @thm-mfac-sqrt | $c_k = \binom{2k}k4^{-k}$; $(\Delta_2 R)^2 = \max_te_t = \sum_{k<T}c_k^2 \approx (\ln T + 3.35)/\pi$ | The default for one pass: within about 10% of the best error, at no cost |
| @thm-mfac-optimal | $X = R^\top R$ from a fixed point in $v$ | Slightly better than the square root, at cubic cost per iteration |
| @thm-mfac-multi | $\Delta_2 R = \max\Vert\sum_ir_{\tau_i}\Vert$ if $R^\top R \ge 0$; banded: $(\sum_i\Vert r_{\tau_i}\Vert^2)^{1/2}$ | Compute it for your actual participation pattern; for several passes, fix the order and use a band no wider than the separation |
| @prp-mfac-noise | $s = \Delta_2 R/\mu(\varepsilon, \delta)$; final noise $s\Vert\ell_T\Vert$, against $\sigma\sqrt T$ | Compare methods by final-prefix noise. Without honest Poisson sampling, fix the order and use a matrix mechanism; keep DP-SGD for small $\varepsilon$ over many epochs |

: The results of this chapter, and where each is used. {#tbl-mfac-summary}

## References {#sec-mfac-references}

- Dwork, Naor, Pitassi & Rothblum, "Differential Privacy Under Continual Observation" (STOC 2010), and Chan, Shi & Song, "Private and Continual Release of Statistics" (ACM Transactions on Information and System Security, 2011; conference version 2010). The binary tree for private running counts.
- Li, Hay, Rastogi, Miklau & McGregor, "Optimizing Linear Counting Queries Under Differential Privacy" (PODS 2010); journal version Li, Miklau, Hay, McGregor & Rastogi, "The Matrix Mechanism" (VLDB Journal 2015). Factoring a query workload to choose where noise goes.
- Kairouz, McMahan, Song, Thakkar, Thakurta & Xu, "Practical and Private (Deep) Learning without Sampling or Shuffling" (ICML 2021). DP-FTRL with tree aggregation, and its comparison with amplified DP-SGD.
- Denisov, McMahan, Rush, Smith & Thakurta, "Improved Differential Privacy for SGD via Optimal Private Linear Operators on Adaptive Streams" (NeurIPS 2022). The matrix mechanism for adaptive streams (Theorem 2.1), lower-triangular factorizations, and the fixed point for the optimal factorization.
- Fichtenberger, Henzinger & Upadhyay, "Constant Matters: Fine-grained Error Bound on Differentially Private Continual Observation" (ICML 2023), and Henzinger, Upadhyay & Upadhyay, "Almost Tight Error Bounds on Differentially Private Continual Counting" (SODA 2023). The square-root factorization and how close it is to optimal, for the maximum and the mean squared error.
- Choquette-Choo, McMahan, Rush & Thakurta, "Multi-Epoch Matrix Factorization Mechanisms for Private Machine Learning" (ICML 2023). Sensitivity under several participations and factorizations optimized for it.
- Choquette-Choo, Ganesh, McKenna, McMahan, Rush, Thakurta & Xu, "(Amplified) Banded Matrix Factorizations: A Unified Approach to Private Training" (NeurIPS 2023). Banded encoders, min-separation, and amplification for matrix mechanisms.
- Xu, Zhang, Andrew, Choquette-Choo, Kairouz, McMahan, Rosenstock & Zhang, "Federated Learning of Gboard Language Models with Differential Privacy" (ACL 2023, industry track). DP-FTRL in a production federated system.
