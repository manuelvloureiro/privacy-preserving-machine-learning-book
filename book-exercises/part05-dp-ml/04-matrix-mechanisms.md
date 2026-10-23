# Exercises: DP-FTRL and matrix factorization

Exercises for [the chapter on matrix factorization mechanisms](../../book-full/part05-dp-ml/04-matrix-mechanisms.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part05-dp-ml/solutions_matrix_mechanisms.py` computes every number quoted here; run it with `uv run python code/part05-dp-ml/solutions_matrix_mechanisms.py`.

The solutions use the chapter's notation. To train with differential privacy (DP) without sampling, a run releases the prefix sums $S_t = x_1 + \dots + x_t$ of the clipped gradient sums, the workload $Ax$ with $A$ the $T \times T$ lower-triangular matrix of ones. A matrix mechanism factors $A = LR$ and releases $L(Rx + z)$; its sensitivity $\Delta_2 R$ is the largest column length of the encoder $R$ for one participation, and its error per unit of privacy at step $t$ is $e_t = \Delta_2 R\|\ell_t\|$, with $\ell_t$ the $t$-th row of the decoder $L$ (@eq-mfac-error). The follow the regularized leader (FTRL) algorithm run on prefix sums released by a binary tree is DP-FTRL. Stochastic gradient descent (SGD) with per-example clipping, Gaussian noise and Poisson sampling is DP-SGD, and its accountant works with Rényi differential privacy (RDP).

## When does the tree beat independent noise? {#sec-sol-mfac-1}

> **Exercise 1.** **When does the tree beat independent noise?** Compute the root-mean-square error per unit of privacy, $\big(\frac1T\sum_te_t^2\big)^{1/2}$, of the binary tree and of independent noise on each step, for $T = 8$ by hand. Then find the smallest $T$ for which the tree is better. *Check: 2.550 against 2.121 at $T = 8$; the tree first wins at $T = 28$.* *(Comparing factorizations by their error)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Independent noise.** $R = I$, so $\Delta_2 R = 1$, and row $t$ of $L = A$ has $t$ ones, so $e_t = \sqrt t$. The mean of $e_t^2$ over $t = 1, \dots, 8$ is $(1 + 2 + \dots + 8)/8 = 36/8 = 4.5$, and the root-mean-square error is $\sqrt{4.5} = 2.121$.
2. **The tree.** With $T = 8$ there are $\lceil\log_28\rceil + 1 = 4$ levels, so each step is in 4 nodes and $\Delta_2 R = 2$ (@eq-mfac-tree). Prefix $t$ uses one node per 1 in its binary expansion: the popcounts of $1, \dots, 8$ are $1, 1, 2, 1, 2, 2, 3, 1$, with mean $13/8$. So $e_t^2 = 4\operatorname{popcount}(t)$ has mean $6.5$, and the root-mean-square error is $\sqrt{6.5} = 2.550$.
3. **Why the tree loses.** Every step pays for being in 4 nodes, a factor of 2 in noise, while the prefix sums are too short for the saving in the number of noise terms to make up for it.
4. **The crossover.** Computing both errors for each $T$, the tree is first better at $T = 28$: 3.789 against 3.808. From there on independent noise grows like $\sqrt{T/2}$ and the tree like $\log_2T$, and the gap widens quickly, to 70.71 against 9.84 at $T = 10{,}000$ (@tbl-mfac-single).

*The tree is a long-run device. For a few dozen steps, plain independent noise is as good; the square-root factorization beats both at every length.*
:::

## The square root at T = 4 {#sec-sol-mfac-2}

> **Exercise 2.** **The square root at T = 4.** Write down the $4 \times 4$ square-root encoder $R$ of @eq-mfac-sqrt, check that $R^2 = A$, and compute $\Delta_2 R$ and the error of the last prefix sum per unit of privacy. Then write the noise added to step 4's gradient, $(R^{-1}z)_4$, in terms of $z_1, \dots, z_4$, and compute its standard deviation relative to the noise scale. *Check: $\sum_{k<4}r_k^2 = 1.4883$, $\Delta_2 R = 1.2200$; the step-4 noise has standard deviation 1.1267 times the noise scale.* *(Power series and Toeplitz matrices)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The coefficients.** $r_0 = 1$ and $r_k = r_{k-1}\frac{2k-1}{2k}$ give $1, \frac12, \frac38, \frac5{16}$, and $R$ is the lower-triangular Toeplitz matrix with first column $(1, \frac12, \frac38, \frac5{16})$.
2. **$R^2 = A$.** By the correspondence with power series, entry $(i, j)$ of $R^2$ is the coefficient of $x^{i-j}$ in $r(x)^2$: $1$; $\frac12 + \frac12 = 1$; $\frac38 + \frac14 + \frac38 = 1$; $\frac5{16} + \frac3{16} + \frac3{16} + \frac5{16} = 1$. All entries on and below the diagonal are 1, which is $A$; the script finds $R^2 - A = 0$ exactly.
3. **Sensitivity.** The first column is the longest: $1 + \frac14 + \frac9{64} + \frac{25}{256} = \frac{381}{256} = 1.4883$, so $\Delta_2 R = 1.2200$.
4. **Last error.** $L = R$, and its last row holds the same four numbers in reverse order, so $\|\ell_4\| = 1.2200$ and $e_4 = 1.2200^2 = 1.4883$, against $\sqrt4 = 2$ for independent noise.
5. **The step noise.** $R^{-1}$ is the Toeplitz matrix of $(1 - x)^{1/2}$, with coefficients $1, -\frac12, -\frac18, -\frac1{16}$. So $(R^{-1}z)_4 = z_4 - \frac12z_3 - \frac18z_2 - \frac1{16}z_1$.
6. **Its spread.** The $z$'s are independent, so by @eq-mfac-linear the variance is the noise scale squared times $1 + \frac14 + \frac1{64} + \frac1{256} = \frac{325}{256} = 1.2695$, a standard deviation of 1.1267 times the noise scale.

*Each step's noise is only 13% larger than independent noise of the same scale, yet the prefix sums are much more accurate, because the step noises are negatively correlated and partly cancel when summed.*
:::

## The optimal factorization for two steps {#sec-sol-mfac-3}

> **Exercise 3.** **The optimal factorization for two steps.** For $T = 2$, every encoder with columns of length 1 has $X = R^\top R = \begin{pmatrix}1 & \rho\\ \rho & 1\end{pmatrix}$ for some $\rho \in (-1, 1)$, and the best decoder gives $\|L\|_F^2 = \operatorname{tr}(A^\top AX^{-1})$. Find the $\rho$ that minimizes it, and compare the minimum with independent noise, the square root and the lower bound @eq-mfac-lower (in squared form, $(\Delta_2 R)^2\|L\|_F^2 \ge \|A\|_*^2/T$). *Check: $\rho = 0.3820$, minimum $2.6180 = (3 + \sqrt5)/2$; independent noise 3, square root 2.8125, bound 2.5.* *(Optimizing a factorization)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The objective.** $A^\top A = \begin{pmatrix}2 & 1\\ 1 & 1\end{pmatrix}$ and $X^{-1} = \frac1{1 - \rho^2}\begin{pmatrix}1 & -\rho\\ -\rho & 1\end{pmatrix}$, so $\operatorname{tr}(A^\top AX^{-1}) = \frac{2 - \rho - \rho + 1}{1 - \rho^2} = \frac{3 - 2\rho}{1 - \rho^2}$.
2. **Minimize.** The derivative vanishes when $-2(1 - \rho^2) + 2\rho(3 - 2\rho) = 0$, that is $\rho^2 - 3\rho + 1 = 0$, whose root in $(-1, 1)$ is $\rho = (3 - \sqrt5)/2 = 0.3820$.
3. **The minimum.** Substituting, $\frac{3 - 2\rho}{1 - \rho^2} = 2.6180 = (3 + \sqrt5)/2$. The fixed-point iteration of the chapter finds the same $X_{12} = 0.3820$ and $\|L\|_F^2 = 2.6180$.
4. **Independent noise.** $\rho = 0$ gives $3 = \|A\|_F^2$.
5. **The square root.** $R = L = \begin{pmatrix}1 & 0\\ \frac12 & 1\end{pmatrix}$: $(\Delta_2 R)^2 = 1 + \frac14 = 1.25$ and $\|L\|_F^2 = 2.25$, product 2.8125. Its columns have unequal lengths, 1.25 and 1 squared, which wastes privacy on the short one.
6. **The bound.** The singular values of $A$ are $1.6180$ and $0.6180$ (the golden ratio and its reciprocal), so $\|A\|_* = \sqrt5$ and $\|A\|_*^2/T = 2.5$.

*So $2.5 \le 2.6180 \le 2.8125 \le 3$: even at two steps the optimum is close to the bound, and the square root sits between it and independent noise. The optimal encoder's columns are positively correlated, $\rho > 0$: noise put into the first step's quantity is partly reused for the second prefix sum.*
:::

## How close is the square root to the lower bound? {#sec-sol-mfac-4}

> **Exercise 4.** **How close is the square root to the lower bound?** Compute the trace-norm lower bound @eq-mfac-lower and the square root's root-mean-square error per unit of privacy for $T = 100$, 1,000 and 10,000. By how much does each grow per tenfold increase of $T$? *Check: bound 2.178, 2.902, 3.634; square root 2.371, 3.102, 3.836; ratio 1.089, 1.069, 1.056.* *(Lower bounds from singular values)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The bound.** The singular values of $A$ are $1/(2\sin\frac{(2k-1)\pi}{4T + 2})$; summing them and dividing by $T$ gives 2.178, 2.902 and 3.634.
2. **The square root.** Its error per unit of privacy at step $t$ is $\Delta_2 R\|\ell_t\| = \big(\sum_{k<T}r_k^2\big)^{1/2}\big(\sum_{k<t}r_k^2\big)^{1/2}$; the root mean square over $t$ is 2.371, 3.102 and 3.836.
3. **The ratios.** 1.089, 1.069 and 1.056: the square root's excess over the bound shrinks, in relative terms, as $T$ grows.
4. **Growth per decade.** The bound grows by 0.724 and 0.732 per decade, the square root by 0.731 and 0.734. Both approach $\ln(10)/\pi = 0.733$ per decade, the growth of $\frac1\pi\ln T$: for small $k$, $\sin\frac{(2k-1)\pi}{4T+2} \approx \frac{(2k-1)\pi}{4T}$, so the $k$-th singular value is $\nu_k \approx \frac{2T}{\pi(2k-1)}$, and the sum of $\frac{2}{\pi(2k - 1)}$ over $k$ up to about $T$ grows like $\frac1\pi\ln T$.

*No factorization can make the error stop growing: the lower bound itself grows like $\ln(T)/\pi$, and the square root matches that rate with a constant overhead of about 0.2.*
:::

## Two participations under the square root {#sec-sol-mfac-5}

> **Exercise 5.** **Two participations under the square root.** With $T = 200$ and a record in steps 1 and 101 (separation 100), use @eq-mfac-multi to compute the square-root encoder's sensitivity from the two columns' lengths and their inner product. Compare with the banded encoder that keeps the first 100 coefficients. *Check: $\|r_1\|^2 = 2.7524$, $\|r_{101}\|^2 = 2.5314$, cross term 1.1211, sensitivity 2.5308; banded 2.2500.* *(Sensitivity under several participations)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The two columns.** Column 1 of the $200 \times 200$ square-root encoder holds $r_0, \dots, r_{199}$, with squared length 2.7524. Column 101 holds $r_0, \dots, r_{99}$ starting in row 101, with squared length 2.5314.
2. **Their inner product.** In rows 101 to 200 the first column holds $r_{100}, \dots, r_{199}$ and the second $r_0, \dots, r_{99}$, so twice the inner product is $2\sum_{k<100}r_kr_{k+100} = 1.1211$.
3. **Sensitivity.** The entries are nonnegative, so the worst case aligns the two gradients, and $(\Delta_2 R)^2 = 2.7524 + 2.5314 + 1.1211 = 6.4048$, a sensitivity of 2.5308. Starting the record at step 1 is the worst offset; the script's maximum over offsets agrees.
4. **Banded.** Cutting the encoder to its first 100 diagonals leaves column 1 with $r_0, \dots, r_{99}$ in rows 1 to 100 and column 101 with the same in rows 101 to 200: no overlap, no cross term, so $(\Delta_2 R)^2 = 2 \times 2.5314$ and the sensitivity is $2.2500 = \sqrt2 \times 1.591$.
5. **Reading it.** The cross term adds 21% to the squared sensitivity of the full square root (1.1211 out of 5.2838). With more participations the cross terms multiply: at 20 epochs the full square root's sensitivity is 17.191 against the banded 7.115 (@tbl-mfac-epochs).

*Correlated noise makes a record's repeated contributions pile up in the same noisy quantities. Banding with a band no wider than the separation keeps them apart, so participations compose as in independent noise.*
:::

## How much amplification would DP-SGD need? {#sec-sol-mfac-6}

> **Exercise 6.** **How much amplification would DP-SGD need?** For one epoch of the running example ($T = 100$, batch 100) at $\varepsilon = 2$, $\delta = 10^{-5}$, the chapter reports the standard deviation of the noise in the final prefix sum: 8.27 for DP-SGD with Poisson sampling, 19.94 for fixed-order independent noise and 5.05 for the square root, in units of $C$. Derive the first two from the noise multipliers. Then find the noise multiplier DP-SGD would need to match the square root's final noise, and the $\varepsilon$ it would then have. *Check: $\sigma = 0.505$, $\varepsilon = 7.82$.* *(Comparing mechanisms at equal privacy)*

::: {.callout-tip collapse="true" title="Solution"}
1. **DP-SGD.** The RDP accountant gives $\sigma = 0.827$ for $q = 0.01$, 100 steps and $\varepsilon = 2$. The final prefix sum carries 100 independent noises of standard deviation $\sigma$, so $0.827\sqrt{100} = 8.27$.
2. **Independent noise without sampling.** One participation, sensitivity 1, $\mu(2, 10^{-5}) = 0.502$, so the noise scale is $1/0.502 = 1.994$ and the final prefix noise is $1.994 \times 10 = 19.94$.
3. **The square root.** Its sensitivity is 1.591, so its noise scale is $1.591/0.502 = 3.172$, and its last decoder row has length 1.591, giving $3.172 \times 1.591 = 5.05$.
4. **Matching it with DP-SGD.** DP-SGD's final noise is $\sigma\sqrt{100}$, so it would need $\sigma = 5.05/10 = 0.505$. At that noise multiplier the accountant reports $\varepsilon = 7.82$ for 100 steps at $q = 0.01$.

*To carry as little noise in its final model as the square root does at $\varepsilon = 2$, DP-SGD would have to spend nearly four times the budget, even with amplification. For a single short pass, correlated noise beats sampling; over many epochs at small $\varepsilon$ the comparison turns, as @tbl-mfac-noise shows.*
:::
