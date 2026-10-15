# DP-FTRL and matrix factorization

## Why this matters

Stochastic gradient descent (SGD) with differential privacy (DP), the [DP-SGD](02-dp-sgd.md) of this part, gets its privacy from two sources. Clipping and Gaussian noise make each training step a Gaussian mechanism. On top of that, **Poisson sampling** puts each record into a batch only with some probability $q$, independently of the other records. The accountant then credits the run for the uncertainty about whether a record was used at all ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md)). This amplification matters a great deal. This part's running example trains a small network on 10,000 records, sampling each with probability $q = 0.01$. There, amplification is what turns a noise multiplier of 1 into a privacy loss $\varepsilon$ of 6.71 over 10,000 steps.

Many training pipelines cannot honestly claim Poisson sampling. Data loaders shuffle the dataset once, or once per epoch, and cut it into fixed batches. Streaming pipelines read the data in the order it arrives. In federated learning the records live on phones, and the server cannot choose which devices check in, so it cannot sample them independently with a known probability. Applying the Poisson accountant to such pipelines can understate $\varepsilon$ considerably ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md#sec-sub-schemes)). The honest alternative, giving up amplification, is expensive. Take the running example at $\varepsilon = 2$ over five passes through the data. With Poisson sampling, DP-SGD needs a noise multiplier of 0.928, but without amplification the multiplier has to grow to 4.458. Test accuracy falls from 0.9460 to 0.8941 as a result.

This chapter asks how to train privately without trusting random batches, and whether noise that is correlated across steps can do better than independent noise. It can, by a lot. The next section shows that training releases running sums of gradients, so the noise can be placed where those running sums need it. The noise added at one step can then partly cancel the noise added before it. The first algorithm to exploit this used the follow the regularized leader (FTRL) optimizer, and is called DP-FTRL. Its modern form, the **matrix factorization mechanism**, contains it as a special case. On the running example it closes nearly all of the gap to amplified DP-SGD without any sampling.

## Training is a stream of prefix sums {#sec-mfac-prefix}

Start with what a training run actually releases. At each step $t$, the per-example gradients of the batch are clipped to length at most the clipping norm $C$ and then summed. Write $x_t$ for that sum and $\hat x_t$ for its noisy version. Plain gradient descent, without momentum, uses a constant step size $\eta$ and a fixed batch size $B$. At each step it moves the weights by $\eta/B$ times $\hat x_t$, and these moves add up over the run. So the weights after step $t$ are the starting weights, minus the running total of the noisy gradient sums scaled by the same factor:
$$w_{t+1} = w_0 - \frac\eta B\hat S_t, \qquad \hat S_t = \hat x_1 + \dots + \hat x_t.$$ {#eq-mfac-weights}

The running total $\hat S_t$ is the noisy **prefix sum**, a private estimate of the true prefix sum $S_t = x_1 + \dots + x_t$. Every model along the way is a fixed function of the noisy prefix sums. A computation on a private output that does not touch the data again is post-processing, which cannot weaken privacy ([Post-processing, group privacy and composition](../part03-dp-foundations/02-properties.md)). So releasing the prefix sums privately makes every model private, and the error in a model is $\eta/B$ times the error in its prefix sum.

Why release every prefix sum, and not only the last one? One prefix sum would be enough if only the final model mattered. But training is adaptive, because the gradient at step $t$ is computed at the weights $w_t$, which depend on the earlier noisy prefix sums. Each intermediate model is used to compute what is released later, so it must itself be private. The task is therefore to release all $T$ prefix sums of a run of $T$ steps, one at a time, as the gradients arrive. Momentum and learning-rate schedules lead to similar tasks (Denisov et al. 2022), but this chapter keeps plain gradient descent.

Two mechanisms suggest themselves, and each fails in its own way. To compare them, recall that the privacy of a Gaussian mechanism depends only on one ratio, its sensitivity divided by its noise standard deviation ([The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md)). This chapter writes $\mu$ for that ratio and compares mechanisms at the same $\mu$. Assume for now that each record takes part in a single step, as in one pass over the data.

The first mechanism adds **independent noise to each step's gradient sum**, which is DP-SGD without amplification. One record changes one gradient sum by at most $C$, so noise with standard deviation $C/\mu$ in each coordinate hides it. But the noise in $\hat S_t$ is then a sum of $t$ independent noise vectors, a random walk. The variances of independent terms add, so the walk's standard deviation is $\sqrt t$ times that of one step, $\sqrt t\,C/\mu$.

The second mechanism adds **independent noise to each prefix sum** directly. Every released sum then has the same error, but the sensitivity is much larger. A record used at step 1 appears in all $T$ prefix sums, and changing it moves each of them by the same vector of length up to $C$. Squared lengths add across the released sums, so the total change has length $\sqrt T\,C$. The noise must then be $\sqrt T\,C/\mu$, and every prefix sum is as noisy as the worst one of the first mechanism. Neither exploits the structure of the problem, since most of $S_t$ was already released in $S_{t-1}$.

Take one epoch of the running example, which is $T = 100$ steps. The target is a privacy loss of $\varepsilon = 2$, with the slack $\delta$ set to $10^{-5}$ here and throughout this chapter. Independent noise on each step then needs a noise scale of 1.994 per step, in units of $C$, and its final prefix sum carries ten times that, 19.94. DP-SGD with Poisson sampling needs a noise multiplier of only 0.827 for the same budget, so its final prefix sum carries 8.27. Amplification buys a factor of 2.4 here, and the rest of the chapter shows that correlated noise can beat even that, without any sampling.

## The matrix mechanism {#sec-mfac-mechanism}

The two naive mechanisms put the noise at opposite ends, on the inputs or on the answers. The **matrix mechanism** of Li et al. (2010) puts it in between. To see how, write the prefix sums as a matrix product. Stack the gradient sums as the rows of a matrix $x$, with one row per step and one column per model weight. Then let $A$ be the $T \times T$ matrix with ones on and below the diagonal and zeros above it. Each row of $A$ starts with a run of ones, so each row of the product $Ax$ is a prefix sum. Releasing all the prefix sums is therefore the same as releasing $Ax$.

Now split $A$ into two factors, $A = LR$. The **encoder** $R$ turns the gradients into intermediate quantities $Rx$, and the mechanism adds Gaussian noise $z$, a matrix of the same shape, to those. The **decoder** $L$ then turns the noisy quantities back into prefix sums. Because $LR = A$, the release is the true prefix sums plus decoded noise:
$$\hat S = L(Rx + z) = Ax + Lz.$$ {#eq-mfac-mechanism}

The entries of $z$ are independent and normal, with standard deviation $sC$. The number $s$ is the **noise scale**, measured in units of $C$ like DP-SGD's noise multiplier. Multiplying the noisy quantities by the fixed matrix $L$ is post-processing, so the release is exactly as private as $Rx + z$.

Privacy is decided by the columns of $R$. Suppose a record takes part in a single step $\tau$, whose position in the data order is fixed in advance. It changes that step's gradient sum by its clipped gradient, a vector $g$ of length at most $C$, and it changes nothing else. In the product $Rx$, that change becomes column $\tau$ of the encoder times $g$. Measure the size of this shift as the length of all its entries read as one long vector. For a column times a vector, that length is the product of their two lengths, so the shift is at most the column's length times $C$. The sensitivity of the encoder, in units of $C$, is therefore the length of its longest column. Write $r_j$ for the $j$-th column of $R$. The mechanism's ratio of sensitivity to noise is then
$$\mu = \frac{\operatorname{sens}(R)}{s}, \qquad \operatorname{sens}(R) = \max_j\|r_j\|.$$ {#eq-mfac-privacy}

This sensitivity depends on what counts as neighbouring datasets. Here the record's slot in the fixed data order is public, and the neighbouring dataset leaves that slot empty, so it contributes a zero gradient. If neighbours instead replace the record by another one, its gradient can change by up to $2C$, and every sensitivity in this chapter doubles.

To meet a target $(\varepsilon, \delta)$, use the exact Gaussian calibration of [f-DP and Gaussian DP](../part04-privacy-accounting/04-gaussian-dp.md). That formula gives the exact privacy of a Gaussian mechanism with a given ratio, so it yields the largest admissible ratio $\mu(\varepsilon, \delta)$. Then set the noise scale to $s = \operatorname{sens}(R)/\mu(\varepsilon, \delta)$. For $\varepsilon = 2$ at this chapter's $\delta$, that largest ratio is 0.502.

Error is decided by the rows of $L$. The error in a released prefix sum is the matching row of $L$ times the noise. Write $\ell_t$ for the row that produces the $t$-th prefix sum. Its error is a weighted sum of independent normal entries, and its variance is the noise variance times the sum of the squared weights. So each coordinate of the error has standard deviation $sC\|\ell_t\|$. Substituting $s = \operatorname{sens}(R)/\mu$ turns this into $C/\mu$ times the quantity
$$e_t = \operatorname{sens}(R)\,\|\ell_t\|.$$ {#eq-mfac-error}
This chapter calls $e_t$ the **error per unit of privacy**, because it is the error's standard deviation when $\mu = 1$ and $C = 1$. Scaling the encoder up and the decoder down by the same factor leaves $e_t$ unchanged. So only the product of column length and row length matters, and a good factorization balances the two.

The two naive mechanisms are the extreme factorizations. Independent noise on each step is $R = I$, the identity, with $L = A$, so its columns have length 1 and its rows have length $\sqrt t$. Noise on each prefix sum is the reverse, $R = A$ with $L = I$, so its first column has length $\sqrt T$ and its rows have length 1.

No factorization can do arbitrarily well. A matrix's singular values measure how much it stretches vectors in its different directions, and a standard routine such as `numpy.linalg.svd` computes them. The sum of the singular values of $A$ is called its trace norm, written $\|A\|_*$, and it depends on $A$ alone. For every factorization, the root-mean-square of the errors over the run is at least that norm divided by the number of steps:
$$\Big(\frac1T\sum_te_t^2\Big)^{1/2} \ge \frac{\|A\|_*}T.$$ {#eq-mfac-lower}
@tbl-mfac-single shows how close each mechanism comes, and the next two sections explain the rows for the tree and the square root.

| Mechanism | $T = 100$: sens$(R)$ | root-mean-square $e_t$ | max $e_t$ | $T = 10{,}000$: sens$(R)$ | root-mean-square $e_t$ | max $e_t$ |
| --- | --- | --- | --- | --- | --- | --- |
| Independent noise ($R = I$) | 1 | 7.11 | 10.00 | 1 | 70.71 | 100.00 |
| Noise on prefix sums ($R = A$) | 10 | 10.00 | 10.00 | 100 | 100.00 | 100.00 |
| Binary tree | 2.828 | 5.05 | 6.93 | 3.873 | 9.84 | 13.96 |
| Square root | 1.591 | 2.37 | 2.53 | 2.000 | 3.84 | 4.00 |
| Optimal | 1 | 2.24 | 2.45 | — | — | — |
| Lower bound | — | 2.18 | — | — | 3.63 | — |

: Error of each released prefix sum per unit of privacy ($\mu = 1$, $C = 1$), for one participation per record. The optimal factorization minimizes the root-mean-square error; it is scaled so that its columns have length 1 and was computed for $T \le 256$ only. {#tbl-mfac-single}

The privacy argument above assumed gradients fixed in advance, which training does not provide. The argument still holds on one condition, which is that both factors are lower-triangular, with no entries above the diagonal. Each row of $Rx + z$ then depends only on the gradients up to its own step, so it can be released at that step. Before the record's step its presence changes nothing, and after it each released row is a Gaussian step chosen in the light of earlier ones. Such steps compose like a single Gaussian mechanism whose squared ratios add ([f-DP and Gaussian DP](../part04-privacy-accounting/04-gaussian-dp.md#sec-gdp-composition)). The squared shifts of the rows add up to the squared length of the record's column. So the total ratio is the same as for fixed inputs, as Denisov et al. (2022) prove in general.

Without the condition, the mechanism cannot run during training, because a factor with entries above the diagonal would need noise or data from future steps. The condition costs nothing, though, because for any factorization there is a lower-triangular one whose release has the same distribution (Denisov et al. 2022).

When $R$ is square and invertible, the mechanism is simply noisy SGD with correlated noise. The decoder must then be $L = AR^{-1}$, which makes the release equal to $A(x + R^{-1}z)$. In words, the mechanism adds the correlated noise $\zeta$, which is $R^{-1}z$, to the gradients and then sums them as usual. Since $R^{-1}$ is lower-triangular too, each step's correlated noise $\zeta_t$ is a fixed combination of the fresh noise vectors drawn so far. An implementation therefore stores those vectors and generates $\zeta_t$ on the fly, without ever forming a $T \times T$ matrix.

## Tree aggregation and DP-FTRL {#sec-mfac-tree}

The first mechanism with error far below $\sqrt T$ was the **binary tree**. Two groups found it independently, as a way to count events in a stream (Dwork et al. 2010; Chan et al. 2010). Build a binary tree over the steps $1, \dots, T$. Each leaf is one step, each node on the next level covers two consecutive steps, each node on the level above that covers four, and so on. A tree over $T$ steps therefore has $\lceil\log_2T\rceil + 1$ levels. The mechanism releases every node's gradient sum with its own independent noise, so its encoder has one row per node, about twice as many rows as steps.

Any prefix of steps can be assembled from a few nodes. Write $t$ in binary, and take one node for each 1 in its binary expansion. For example, seven is four plus two plus one. The prefix up to step 7 therefore uses three nodes, one covering the first four steps, one covering the next two, and one covering step 7 alone. So the noisy prefix sum collects one noise term for each 1 in the binary expansion of $t$. That is at most $\log_2 t + 1$ terms, instead of the $t$ terms of independent noise.

The price is paid in the encoder. Each step lies in exactly one node per level, so its column in $R$ holds a 1 for each level and zeros elsewhere. That column's length is the square root of the number of levels, and the noise has to be that much larger. Write $\operatorname{popcount}(t)$ for the number of 1s in the binary expansion of $t$. Combining the two effects gives the error per unit of privacy of the $t$-th prefix sum:
$$e_t = \sqrt{\lceil\log_2T\rceil + 1}\,\sqrt{\operatorname{popcount}(t)}.$$ {#eq-mfac-tree}
Both factors grow like $\sqrt{\log T}$, so the error grows like $\log T$, far more slowly than the $\sqrt t$ of independent noise. The tree also works during training, because a node can be released as soon as its last step has arrived, and $\hat S_t$ uses only completed nodes.

A tree over one epoch of the running example has 8 levels, so its sensitivity is 2.828. Among these steps, the largest popcount is 6, reached at step 63. That gives a largest error per unit of privacy of 6.93, against 10 for independent noise. Trading a larger noise per node for far fewer noise terms per prefix is a large win for long runs, but not for short ones. With only 8 steps, the tree's root-mean-square error is worse than that of independent noise.

The tree is not the end of the story, for two reasons. First, its error jumps around. The prefix up to step 127 uses seven nodes, because its binary expansion is all ones. The next prefix uses only one node (@fig-mfac-errors, left). Second, its encoder is wasteful, because few prefixes use the nodes at the top levels, yet every step pays for being in them. The square-root factorization of the next section has a smaller error at every step.

Kairouz et al. (2021) used the tree with FTRL, an optimizer that in this setting produces exactly the same weights as gradient descent on the noisy prefix sums. They also added momentum and restarted the tree after a number of steps, and called the method DP-FTRL. They report that DP-FTRL, without any amplification, compares favourably with amplified DP-SGD in their experiments, and that it allows much more flexible data access. Xu et al. (2023) describe how production language models for the Gboard keyboard were trained with DP-FTRL in federated learning, where Poisson sampling of devices is not available.

## The square-root factorization {#sec-mfac-sqrt}

The tree treats the two factors very differently, with an encoder that has a row for every node and a decoder that picks a few nodes per prefix. A natural symmetric choice is to use the same matrix for both, $L = R$, so that $R$ is a square root of $A$. For the mechanism to work during training, this square root must be lower-triangular. Such a square root exists, and a convenient fact gives it in closed form.

The fact concerns lower-triangular matrices that are constant along each diagonal, which are called Toeplitz matrices. Such a matrix is described completely by its first column. Read the entries of that column as the coefficients of a power series in a variable $u$, so that the column $1, 2, 3, \dots$ becomes $1 + 2u + 3u^2 + \dots$. Then multiplying two such matrices gives the matrix whose series is the product of their series. The prefix-sum matrix $A$ has a first column of ones, so it corresponds to the series $1 + u + u^2 + \dots$, which is $1/(1 - u)$. A matrix whose square is $A$ therefore corresponds to a series whose square is $1/(1 - u)$, namely $(1 - u)^{-1/2}$. The $k$-th coefficient of that series is
$$c_k = \binom{2k}k4^{-k}, \qquad c = 1, \tfrac12, \tfrac38, \tfrac5{16}, \tfrac{35}{128}, \dots.$$ {#eq-mfac-sqrt}
In the matrix $R$, the coefficient $c_0$ sits on the diagonal, $c_1$ just below it, and so on.

For this factorization, the sensitivity and the largest error come from the same sum. Column $j$ of $R$ holds $c_0, \dots, c_{T-j}$, so the first column is the longest. Row $t$ of $L = R$ holds $c_{t-1}, \dots, c_0$, so the last row is the longest. Both have squared length $\sum_{k<T}c_k^2$. The largest error, $e_T$, is the sensitivity times the last row's length, which is the square root of that sum times itself:
$$\operatorname{sens}(R)^2 = \max_t e_t = \sum_{k<T}c_k^2.$$ {#eq-mfac-sqrt-error}

This sum grows like $\ln(T)/\pi$. A known bound on the coefficients (quoted by Fichtenberger, Henzinger & Upadhyay 2023) shows that their squares shrink roughly like $1/(\pi k)$, and adding up such terms gives a logarithm. So the square root's error grows logarithmically, like the tree's, but it stays much smaller than the tree's at both lengths in @tbl-mfac-single. For one epoch of the running example the sum is 2.531, and the sensitivity is its square root, 1.591. The final prefix sum's error per unit of privacy is then 2.53, a quarter of independent noise's 10.

Seen as noisy SGD, the square root adds the correlated noise $\zeta = R^{-1}z$ to the gradients. The inverse $R^{-1}$ corresponds to the series $(1 - u)^{1/2}$, whose coefficients after the first are all negative. The noise added to step $t$'s gradient is therefore
$$\zeta_t = z_t - \tfrac12z_{t-1} - \tfrac18z_{t-2} - \tfrac1{16}z_{t-3} - \dots,$$
which is fresh noise minus a decaying share of the noise already injected. Each step's noise is a little larger than a single fresh draw. Its standard deviation tends to $\sqrt{4/\pi} = 1.128$ times the noise scale, a value the full edition's script finds numerically. But the negative coefficients pull the accumulated noise back toward zero, so the running sums no longer drift like a random walk. In @fig-mfac-noise-paths, the per-step noises of the two mechanisms look alike, but their running sums do not.

![Left: one coordinate of the noise added to each of the first 60 steps' gradients, independent (identity factorization) and square-root correlated, both calibrated to $\mu = 1$ for one participation. Right: the noise in the prefix sums over 1,000 steps, for independent noise, the binary tree and the square root, with bands of two standard deviations. Independent noise drifts like a random walk; the square root's stays within a narrow band.](figures/mfac_noise_paths.svg){#fig-mfac-noise-paths}

The square root comes close to the lower bound of @eq-mfac-lower. Over one epoch of the running example, its root-mean-square error is 2.37, within 9% of the bound. For longer runs the gap shrinks, to 6% over 10,000 steps. No factorization beats the bound, so the square root is also that close to the best possible error. Denisov et al. (2022) characterize the best factorization as the fixed point of an iteration on matrices, so it can be computed. On one epoch, the factorization it finds reaches 2.24, against 2.37 for the square root. But each iteration takes time cubic in $T$, so the square root, free and nearly as good, is the usual default for long runs. @fig-mfac-errors compares the errors of all five factorizations, and @fig-mfac-matrices shows that the square root and the optimal factorization even look alike.

![Left: standard deviation of the error of each released prefix sum $S_t$ for $T = 256$, one participation, at $\mu = 1$ and $C = 1$, for five factorizations. Right: the root-mean-square error over a run against its length $T$, with the trace-norm lower bound dotted; the optimal factorization is computed up to $T = 256$.](figures/mfac_errors.svg){#fig-mfac-errors}

![The encoder $R$ (top) and decoder $L$ (bottom) of three factorizations of the $16 \times 16$ prefix-sum matrix. The tree's encoder has one row per node (31 rows), and its decoder picks the nodes of each prefix. The square root uses the same matrix for both; the optimal factorization's encoder has columns of length exactly 1.](figures/mfac_matrices.svg){#fig-mfac-matrices}

## Several participations {#sec-mfac-epochs}

Training usually makes several passes over the data, so a record takes part in $k$ steps. Its gradient enters the encoder once through each of the $k$ matching columns of $R$. Its clipped gradients may differ between participations and may depend on earlier releases, but the formula below takes the worst gradients over every choice, so it covers that case too.

The worst case is easy to find when every pair of columns of $R$ has a nonnegative inner product. Any encoder with nonnegative entries qualifies, such as the identity, the tree and the square root.

Here is why the condition matters. The squared length of the record's total shift is a sum of terms, one for each pair of participations. Each term is the inner product of the two columns times the inner product of the two gradients. A gradient inner product is at most $C^2$, reached when the two gradients are the same vector of full length. So when every column inner product is nonnegative, the shift is largest when the record contributes the same gradient every time. The sensitivity is then the length of the sum of the record's columns, maximized over the participation patterns it could have. With $\tau_1, \dots, \tau_k$ the steps of one pattern,
$$\operatorname{sens}(R) = \max_{\text{patterns}}\Big\|\sum_{i=1}^kr_{\tau_i}\Big\|.$$ {#eq-mfac-multi}
As before, this is in units of $C$, and the ratio is $\mu = \operatorname{sens}(R)/s$. If some column inner products are negative, aligned gradients are no longer necessarily the worst case. The worst case must then be computed with the signs taken into account, and it can be larger than this formula gives.

Correlated noise makes repeated participation more expensive. For independent noise, the columns of the identity are at right angles, so $k$ participations cost $\sqrt k$, the usual composition of Gaussian mechanisms. For the square root, the columns of nearby steps overlap and point in similar directions. A record that appears again then adds its contribution on top of its earlier one, and the sensitivity grows faster than $\sqrt k$. The tree is worse still, because a node covering many steps contains several participations of the same record. Those contributions add up inside the node before the length is taken, so the squared sensitivity grows like the square of their number.

**Banding** removes the overlaps. A banded encoder keeps only the first $b$ diagonals of the square root and sets every entry further below the diagonal to zero. Suppose also that any two participations of a record are at least $b$ steps apart, a condition called **$b$-min-separation**. Each column is then zero wherever the column of another participation is nonzero, so the cross terms vanish, whatever the gradients are. The sensitivity is back to the square root of the sum of the squared column lengths, which is at most $\sqrt k$ times the longest column, as for independent noise. Without the separation, two participations closer than $b$ steps would overlap again, and the bound would no longer hold.

The price of banding is accuracy for a single participation. The band no longer spreads the noise across the whole run, so the banded decoder, $L = AR^{-1}$, has longer rows than the square root's. A single participation therefore gets a larger error. With many participations, though, the saving in sensitivity dominates. Choquette-Choo, McMahan, Rush & Thakurta (2023) optimize factorizations directly for multi-epoch training. A second paper by Choquette-Choo et al. (2023) shows that banded factorizations can also use amplification by sampling. They then match DP-SGD where amplification is strong and beat it elsewhere.

The running example uses a fixed order of this kind. One public random permutation of the record slots, drawn independently of the data, is kept for every epoch and cut into 100 batches of 100. Each epoch is then 100 steps, and each record takes part once per epoch, exactly 100 steps apart. @tbl-mfac-epochs uses this order for $E$ epochs, so the run has $T = 100E$ steps and every record takes part $E$ times. The banded encoder keeps the first 100 coefficients of the square root.

| Epochs $E$ | Independent: sens / final $e_T$ | Tree | Square root | Banded square root |
| --- | --- | --- | --- | --- |
| 1 | 1.000 / 10.00 | 2.828 / 4.90 | 1.591 / 2.53 | 1.591 / 2.53 |
| 2 | 1.414 / 20.00 | 4.690 / 8.12 | 2.531 / 4.20 | 2.250 / 4.16 |
| 5 | 2.236 / 50.00 | 8.944 / 21.91 | 5.080 / 8.86 | 3.558 / 8.56 |
| 10 | 3.162 / 100.00 | 16.186 / 39.65 | 9.154 / 16.54 | 5.031 / 15.69 |
| 20 | 4.472 / 200.00 | 30.430 / 74.54 | 17.191 / 32.09 | 7.115 / 29.85 |

: Sensitivity (in units of $C$) and error of the final prefix sum per unit of privacy, for $E$ participations 100 steps apart, $T = 100E$. {#tbl-mfac-epochs}

At 20 epochs, the banded sensitivity is 7.115. That is exactly $\sqrt{20}$ times the single-participation value, as the banding argument predicts. The full square root's sensitivity is 17.191, more than twice as large, because its long columns overlap. Yet the final errors are close, 29.85 against 32.09, because the banded decoder is less accurate.

So the data order is part of the mechanism, because the sensitivity depends on which steps a record can occupy. A fixed permutation repeated every epoch guarantees a separation of exactly $n/B$ steps, the number of records $n$ divided by the batch size, which is the length of an epoch. Reshuffling every epoch can put a record at the end of one epoch and at the start of the next, one step apart, which breaks the banded bound. The order must be chosen independently of the data and fixed in the analysis, but it need not be random.

## The running example: matrix mechanisms against DP-SGD {#sec-mfac-run}

The previous sections compared mechanisms by the error in the released sums, but what matters is the accuracy of the trained network. The running example is the network of [DP-SGD](02-dp-sgd.md), trained on $n = 10{,}000$ records with clipping norm $C = 0.5$, and each matrix mechanism trains it in the fixed order above. DP-SGD trains it with Poisson sampling at $q = 0.01$, so its expected batch is also 100, and it is accounted with the Rényi differential privacy (RDP) accountant. All methods are compared at the same $\varepsilon$.

@tbl-mfac-noise gives the noise each method needs and the noise that ends up in the final prefix sum, computed as in the earlier sections. DP-SGD's comes from its accountant, and the fixed-order methods' from @tbl-mfac-epochs and $\mu(\varepsilon, \delta)$. @tbl-mfac-run gives the test accuracy, averaged over three runs, with each method at the better of two step sizes.

| $\varepsilon$, epochs | DP-SGD $\sigma$ | Independent $s$ | Tree $s$ | Square root $s$ | Banded $s$ | Final prefix noise: DP-SGD / independent / tree / square root / banded |
| --- | --- | --- | --- | --- | --- | --- |
| 2, 1 | 0.827 | 1.994 | 5.639 | 3.172 | 3.172 | 8.27 / 19.94 / 9.77 / 5.05 / 5.05 |
| 2, 5 | 0.928 | 4.458 | 17.833 | 10.129 | 7.093 | 20.75 / 99.69 / 43.68 / 17.67 / 17.06 |
| 2, 10 | 1.03 | 6.305 | 32.273 | 18.251 | 10.031 | 32.57 / 199.38 / 79.05 / 32.98 / 31.28 |
| 8, 1 | 0.501 | 0.600 | 1.698 | 0.955 | 0.955 | 5.01 / 6.00 / 2.94 / 1.52 / 1.52 |
| 8, 5 | 0.573 | 1.342 | 5.369 | 3.049 | 2.135 | 12.81 / 30.01 / 13.15 / 5.32 / 5.14 |
| 8, 10 | 0.616 | 1.898 | 9.716 | 5.495 | 3.020 | 19.48 / 60.02 / 23.80 / 9.93 / 9.42 |

: Noise needed by each method at equal $\varepsilon$ ($\delta = 10^{-5}$, 100 steps per epoch) and the resulting noise in the final prefix sum, all in units of $C$. DP-SGD's noise multiplier is $\sigma$; it uses Poisson sampling with $q = 0.01$ and the RDP accountant. The others use a fixed order and exact Gaussian accounting. {#tbl-mfac-noise}

| $\varepsilon$, epochs | DP-SGD (Poisson) | Independent | Tree | Square root | Banded square root |
| --- | --- | --- | --- | --- | --- |
| 2, 1 | 0.9343 ± 0.0044 | 0.9066 ± 0.0031 | 0.9134 ± 0.0038 | 0.9361 ± 0.0046 | 0.9358 ± 0.0019 |
| 2, 5 | 0.9460 ± 0.0021 | 0.8941 ± 0.0071 | 0.9183 ± 0.0084 | 0.9430 ± 0.0011 | 0.9456 ± 0.0019 |
| 2, 10 | 0.9493 ± 0.0012 | 0.8641 ± 0.0130 | 0.9096 ± 0.0074 | 0.9398 ± 0.0029 | 0.9491 ± 0.0024 |
| 8, 1 | 0.9371 ± 0.0016 | 0.9395 ± 0.0018 | 0.9412 ± 0.0006 | 0.9461 ± 0.0022 | 0.9434 ± 0.0017 |
| 8, 5 | 0.9524 ± 0.0026 | 0.9420 ± 0.0029 | 0.9461 ± 0.0012 | 0.9497 ± 0.0025 | 0.9542 ± 0.0014 |
| 8, 10 | 0.9520 ± 0.0006 | 0.9410 ± 0.0032 | 0.9484 ± 0.0004 | 0.9530 ± 0.0015 | 0.9549 ± 0.0024 |

: Test accuracy (mean ± standard deviation over three runs) at equal $\varepsilon$, $\delta = 10^{-5}$, batch 100, clipping norm 0.5, best of step sizes 0.5 and 2. Without noise, fixed-order training with clipping reaches 0.9420, 0.9580 and 0.9606 after 1, 5 and 10 epochs. {#tbl-mfac-run}

Without amplification, correlated noise is essential. At $\varepsilon = 2$, fixed-order SGD with independent noise loses 2.8 percentage points of accuracy to DP-SGD after one epoch. After ten epochs it loses 8.5 points. It even gets worse with more training, because each extra epoch adds a participation, so its noise scale grows like $\sqrt E$, from 1.994 at one epoch to 6.305 at ten. The tree recovers about a quarter to a half of the gap. The square root and the banded square root recover nearly all of it, and at ten epochs the banded encoder reaches 0.9491, within 0.02 points of DP-SGD.

Matrix mechanisms can even beat amplified DP-SGD for short runs at large $\varepsilon$, where amplification gains little. At $\varepsilon = 8$ and one epoch, DP-SGD needs a noise multiplier of 0.501. Independent noise without sampling needs barely more, 0.600. With correlated noise, the square root's final prefix sum carries noise of only 1.52, against 5.01 for DP-SGD. So the square root wins outright, with an accuracy of 0.9461 against 0.9371.

DP-SGD keeps its lead at small $\varepsilon$ over many epochs, where amplification is strongest. At the smaller budget, ten epochs raise DP-SGD's noise multiplier only from 0.827 to 1.03, while every fixed-order mechanism pays for ten participations. The full square root then falls behind, because its overlapping columns make repeated participation expensive, and only banding keeps up.

One number is a good guide to these results. In every row of the two tables, the method with the least noise in its final prefix sum is the most accurate, or within the run-to-run spread of the best. Since that noise needs no training to compute, it is a cheap way to choose a method. @fig-mfac-accuracy shows the same pattern across privacy budgets at five epochs.

![Test accuracy of the running example after 5 epochs (500 steps of batch 100) at step size 1, against $\varepsilon$ at $\delta = 10^{-5}$, one run per point. DP-SGD uses Poisson sampling; the other four use a fixed data order and differ only in how their noise is correlated.](figures/mfac_accuracy.svg){#fig-mfac-accuracy}

The comparison is slightly tilted, in three ways worth knowing. DP-SGD is accounted with the RDP accountant, which overstates $\varepsilon$ somewhat compared with a numerical accountant ([Choosing an accountant](../part04-privacy-accounting/09-choosing-an-accountant.md)), so a tighter accountant would give DP-SGD somewhat less noise. The step size was also chosen per method by test accuracy, which itself leaks information ([Hyperparameter tuning](03-hyperparameter-tuning.md)). Finally, the runs are short, at most 1,000 steps. The noise gaps of @tbl-mfac-epochs keep widening with the number of epochs, so longer runs should separate the methods more, not less.

All of this is what makes matrix mechanisms the tool for federated learning. There a record is a device, and the server cannot sample devices with a known rate, so the Poisson accountant does not apply. Suppose the server can make sure a device contributes at most once every $b$ rounds. Then a banded mechanism whose band is at most $b$ gives a provable guarantee, and at the running example's scale it is as accurate as DP-SGD with sampling would be. That is why production federated training of language models uses DP-FTRL (Xu et al. 2023). It is also why later work optimizes factorizations for user-level privacy, where the privacy unit is a user rather than an example ([User-level differential privacy](07-user-level.md)).

## Code

A correlated-noise mechanism is a few lines on top of noisy SGD, and the part that takes care is the sensitivity. This snippet computes the square root's coefficients, its single-participation errors, the multi-epoch sensitivities of @tbl-mfac-epochs and one resulting noise scale.

```python
from math import ceil, erfc, exp, log2, sqrt


def sqrt_coeffs(t: int) -> list[float]:
    """c_k = C(2k, k)/4^k: the first column of the lower-triangular square root of A."""
    r = [1.0]
    for k in range(1, t):
        r.append(r[-1] * (2 * k - 1) / (2 * k))
    return r


def toeplitz_sensitivity(c: list[float], t: int, sep: int, epochs: int) -> float:
    """Length of the sum of the encoder columns a record touches, worst slot (entries of c >= 0)."""
    best = 0.0
    for offset in range(sep):
        steps = range(offset, t, sep)[:epochs]
        col = [sum(c[i - p] for p in steps if 0 <= i - p < len(c)) for i in range(t)]
        best = max(best, sqrt(sum(v * v for v in col)))
    return best


def tree_sensitivity(t: int, sep: int, epochs: int) -> float:
    """A record's contributions to one tree node add up before the norm is taken."""
    best = 0.0
    for offset in range(sep):
        steps = range(offset, t, sep)[:epochs]
        total = 0
        for level in range(ceil(log2(t)) + 1):
            counts: dict[int, int] = {}
            for p in steps:
                counts[p >> level] = counts.get(p >> level, 0) + 1
            total += sum(n * n for n in counts.values())
        best = max(best, sqrt(total))
    return best


def phi(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def gaussian_mu(eps: float, delta: float) -> float:
    """Largest sensitivity-to-noise ratio for which a Gaussian mechanism is (eps, delta)-DP."""
    lo, hi = 1e-9, 100.0
    for _ in range(200):
        mu = (lo + hi) / 2
        if phi(mu / 2 - eps / mu) - exp(eps) * phi(-mu / 2 - eps / mu) > delta:
            hi = mu
        else:
            lo = mu
    return lo


T = 100
r = sqrt_coeffs(T)
sens = sqrt(sum(v * v for v in r))  # the first column is the longest
rows = [sqrt(sum(v * v for v in r[:t])) for t in range(1, T + 1)]  # row t of L = R
print(f"{sens:.3f} {sens * rows[-1]:.2f}")  # 1.591 2.53
print(f"{sens * sqrt(sum(x * x for x in rows) / T):.2f}")  # 2.37
print(f"{sqrt(sum(range(1, T + 1)) / T):.2f}")  # 7.11

E = 20  # epochs of 100 steps, each record exactly 100 steps apart
print(f"{toeplitz_sensitivity(sqrt_coeffs(T * E), T * E, T, E):.3f}")  # 17.191
print(f"{toeplitz_sensitivity(r, T * E, T, E):.3f}")  # 7.115
print(f"{tree_sensitivity(T * E, T, E):.3f}")  # 30.430
print(f"{toeplitz_sensitivity(r, T * 10, T, 10) / gaussian_mu(2.0, 1e-5):.3f}")  # 10.031
```

The banded encoder is the square root cut to its first 100 coefficients, which is why it reuses `r`. The last line is its noise scale for $\varepsilon = 2$ over ten epochs, as in @tbl-mfac-noise. The script [matrix_mechanisms.py](../../code/part05-dp-ml/matrix_mechanisms.py) builds every factorization, including the optimal one, and trains the running example with each. Run it from the repository root:

```bash
uv run python code/part05-dp-ml/matrix_mechanisms.py
```

## Cheat sheet

| Result | Formula | Conditions |
| --- | --- | --- |
| Training releases prefix sums | $w_{t+1} = w_0 - \frac\eta B\hat S_t$ | Gradient descent or FTRL, no momentum |
| Matrix mechanism | $\hat S = L(Rx + z) = Ax + Lz$, with $A = LR$ | $L$ and $R$ lower-triangular for adaptive training |
| Privacy | $\mu = \operatorname{sens}(R)/s$, with $\operatorname{sens}(R) = \max_j\|r_j\|$; set $s = \operatorname{sens}(R)/\mu(\varepsilon, \delta)$ | One participation, slot fixed in advance; adaptive inputs need lower-triangular factors |
| Error | Standard deviation $sC\|\ell_t\|$ per coordinate; $e_t = \operatorname{sens}(R)\,\|\ell_t\|$ per unit of privacy | Any factorization |
| Lower bound | Root-mean-square $e_t \ge \|A\|_*/T$ | Any factorization |
| Independent noise | $R = I$: sensitivity 1, error $\sqrt t$ | DP-SGD without amplification |
| Binary tree | Sensitivity $\sqrt{\lceil\log_2T\rceil + 1}$; error grows like $\log T$ | DP-FTRL |
| Square root | $c_k = \binom{2k}k4^{-k}$; $\operatorname{sens}(R)^2$ = max error $= \sum_{k<T}c_k^2 \approx (\ln T + 3.35)/\pi$ | $L = R$ |
| Several participations | Sensitivity = largest length, over the possible participation patterns, of the sum of the participating columns | Nonnegative inner products between columns |
| Banded encoder | Sensitivity at most $\sqrt k$ times the longest column, for $k$ participations | Band $\le$ separation of participations |

: The results of this chapter. {#tbl-mfac-cheat}

**Rules of thumb**

- If you cannot implement Poisson sampling, do not account as if you did. Fix the data order in advance and use a matrix mechanism.
- For one pass over the data, use the square-root factorization. Its root-mean-square error is within about 10% of the lower bound, and so of the best possible error, and it costs nothing to compute.
- For several passes, fix the order so that participations are evenly separated, and use a banded encoder whose band is at most the separation.
- Compute the sensitivity for the participation pattern you actually have, because reshuffling every epoch changes it.
- Compare methods by the noise in the final prefix sum at equal $\varepsilon$. On the running example, the method with the least of it was the most accurate, or within the run-to-run spread of the best, in every configuration.

## Further reading

- Kairouz, McMahan, Song, Thakkar, Thakurta & Xu, "Practical and Private (Deep) Learning without Sampling or Shuffling" (ICML 2021): DP-FTRL with tree aggregation, and its comparison with amplified DP-SGD.
- Denisov, McMahan, Rush, Smith & Thakurta, "Improved Differential Privacy for SGD via Optimal Private Linear Operators on Adaptive Streams" (NeurIPS 2022): the matrix mechanism for adaptive streams, and the optimal factorization.
- Choquette-Choo, Ganesh, McKenna, McMahan, Rush, Thakurta & Xu, "(Amplified) Banded Matrix Factorizations: A Unified Approach to Private Training" (NeurIPS 2023): banded encoders, min-separation, and amplification for matrix mechanisms.
- Xu, Zhang, Andrew, Choquette-Choo, Kairouz, McMahan, Rosenstock & Zhang, "Federated Learning of Gboard Language Models with Differential Privacy" (ACL 2023, industry track): DP-FTRL in a production federated system.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/04-matrix-mechanisms.md), except the following, which it cites, states or checks numerically: the theorem that lower-triangular factorizations stay private for adaptive streams, for one or several participations, and that a lower-triangular factorization loses nothing (Denisov et al. 2022); the fixed point that characterizes the optimal factorization (Denisov et al. 2022); the bound on the squared square-root coefficients (quoted from Fichtenberger et al. 2023) and the approximation $(\ln T + 3.35)/\pi$, stated without proof; the limit $\sqrt{4/\pi}$, checked numerically; and the experimental results of the papers cited.
