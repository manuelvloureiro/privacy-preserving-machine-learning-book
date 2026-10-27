# Sums, limits and counting

This appendix proves six facts about sums of random variables and about counting, which the chapters use without a full proof. The first finds the distribution of a sum of two independent variables by adding up the ways the sum can arise, so the book does not need the theorem that a moment generating function pins down its distribution. The second finds the most likely value of a binomial count, the height of that peak, and the size of $n!$. The third uses the size of $k!$ to show how fast a Poisson count becomes unlikely far above its mean. The fourth proves the central limit theorem (CLT) by swapping one term at a time for a normal one. The fifth shows that the length of a normal vector in $d$ dimensions rarely exceeds $\sqrt d$ by much, and the sixth finds the tail of the difference of two Laplace variables. Each is a statement about a sum or a count, and the proofs share three tools: they add up the ways a sum can happen, compare neighbouring terms through their ratio, and apply Markov's inequality to an exponential.

@tbl-apx-prob-map lists where each result is stated or used.

| Derivation | Where the chapters state or use it |
| --- | --- |
| @sec-apx-prob-convolution: sums by convolution, rotation invariance of the normal | @sec-dist-mgf and the sections on the Poisson, Laplace and normal in [Probability distributions](../part01-prerequisites/01-distributions.md); @sec-div-gauss; @sec-lg-analytic; @sec-lower-packing |
| @sec-apx-prob-stirling: Stirling's formula, the binomial mode and its height | @sec-threat-games; @sec-threat-average; @sec-prop-scaling; @sec-memo-exposure |
| @sec-apx-prob-poisson-tail: the tail of a Poisson count | @sec-dist-poisson |
| @sec-apx-prob-clt: the central limit theorem | The normal section of [Probability distributions](../part01-prerequisites/01-distributions.md); @sec-gdp-clt and every normal approximation in the book |
| @sec-apx-prob-gauss-norm: the length of a normal vector | @sec-mean-multi |
| @sec-apx-prob-laplace-diff: the difference of two Laplace variables | @sec-pate-stable; @sec-lower-packing |

: The derivations of this appendix and the chapter sections that state or use them. {#tbl-apx-prob-map}

## A sum's distribution, found by convolution {#sec-apx-prob-convolution}

### What the chapter uses

[Probability distributions](../part01-prerequisites/01-distributions.md) identifies four sums of independent variables, and @sec-div-gauss and @sec-lg-analytic use the rotation invariance of a normal vector $X \sim \mathcal N(0, \sigma^2 I_d)$, whose $d$ coordinates are independent $\mathcal N(0, \sigma^2)$ variables. As in the chapter, $\mathrm{Exp}(b)$ and $\mathrm{Lap}(b)$ have scale $b$, an equals sign between distributions means that the left side has the distribution on the right, and in the last identity each $X_i$ is $\mathcal N(\mu_i, \sigma_i^2)$ with constant weights $a_i$:
$$\begin{aligned}
&\mathrm{Poisson}(\mu_1) + \mathrm{Poisson}(\mu_2) = \mathrm{Poisson}(\mu_1 + \mu_2), \qquad \mathrm{Exp}(b) - \mathrm{Exp}(b) = \mathrm{Lap}(b),\\
&\mathrm{Exp}(1) + \dots + \mathrm{Exp}(1) \;(n \text{ terms}) = \mathrm{Gamma}(n, 1), \qquad \textstyle\sum_i a_iX_i \sim \mathcal N\big(\sum_i a_i\mu_i, \sum_i a_i^2\sigma_i^2\big).
\end{aligned}$$
A rotation leaves the distribution of $X$ unchanged, so its coordinates along two perpendicular unit vectors are independent $\mathcal N(0, \sigma^2)$ variables. The chapter finds the four sums by multiplying moment generating functions (MGFs), and then relies on a theorem it takes as given in @sec-dist-mgf, that two distributions with the same MGF near 0 are the same. This section proves all five facts without that theorem, and also the density $\frac\varepsilon4(1 + \varepsilon|z|)e^{-\varepsilon|z|}$ of the difference of two $\mathrm{Lap}(1/\varepsilon)$ variables that @sec-lower-packing uses.

### The idea

To find the chance that a sum takes a value, list every way it can take that value and add up their chances. Roll two fair dice and ask for a total of 4. The first die shows 1, 2 or 3, and the second must then show 3, 2 or 1. Each pair has chance $\frac16 \cdot \frac16$, because the dice are independent, so the total is $3/36$. @tbl-apx-prob-dice writes this out.

| First die $x$ | 1 | 2 | 3 | 4, 5, 6 |
| --- | --- | --- | --- | --- |
| Second die must show $4 - x$ | 3 | 2 | 1 | impossible |
| Chance of this way | $1/36$ | $1/36$ | $1/36$ | 0 |

: The ways two dice can total 4. Adding the last row gives $\Pr[\text{total} = 4] = 3/36$. {#tbl-apx-prob-dice}

For densities the list of ways becomes an integral, the **convolution**. Take independent variables $X$ and $Y$ with densities $f$ and $g$. The first variable takes a value $x$, the second must make up the rest, $s - x$, and independence multiplies the two densities, so the density of $X + Y$ at $s$ is
$$f_{X+Y}(s) = \int f(x)\,g(s - x)\,dx.$$ {#eq-apx-prob-conv}
For counts the integral becomes a sum over $x$. Each sum in the chapter is one such integral.

### Derivation

1. **Why @eq-apx-prob-conv holds.** Fix a small width $\kappa > 0$ and the interval of that width around $s$. The sum lands in it when $X$ is near some value $x$ and $Y$ lands within $\kappa$ of $s - x$. Given $X = x$, that has chance about $g(s - x)\kappa$, because $Y$ is independent of $X$, so its density does not change. Averaging over $x$ with the density $f$ gives $\kappa\int f(x)g(s - x)\,dx$, and dividing by $\kappa$ gives the density. For counts the same step reads $\Pr[X + Y = k] = \sum_j\Pr[X = j]\Pr[Y = k - j]$, as with the dice.

2. **Poisson.** Take independent $N_1 \sim \mathrm{Poisson}(\mu_1)$ and $N_2 \sim \mathrm{Poisson}(\mu_2)$. The sum of step 1, with $\Pr[N = j] = e^{-\mu}\mu^j/j!$, is
$$\Pr[N_1 + N_2 = k] = \sum_{j=0}^k e^{-\mu_1}\frac{\mu_1^j}{j!}\,e^{-\mu_2}\frac{\mu_2^{k-j}}{(k-j)!} = \frac{e^{-(\mu_1+\mu_2)}}{k!}\sum_{j=0}^k\binom kj\mu_1^j\mu_2^{k-j}.$$
The second form multiplies and divides by $k!$, and uses $\binom kj = k!/(j!(k-j)!)$. By the binomial theorem the last sum is $(\mu_1 + \mu_2)^k$, so $\Pr[N_1 + N_2 = k] = e^{-(\mu_1+\mu_2)}(\mu_1 + \mu_2)^k/k!$, the $\mathrm{Poisson}(\mu_1 + \mu_2)$ probability.

3. **Exponential minus exponential.** Let $Y_1, Y_2 \sim \mathrm{Exp}(b)$ be independent, each with density $f(y) = \frac1be^{-y/b}$ for $y \ge 0$ and 0 for $y < 0$. The difference $Y_1 - Y_2$ equals $x$ when $Y_1 = y$ and $Y_2 = y - x$, so its density at $x$ is $\int f(y)f(y - x)\,dy$. This is @eq-apx-prob-conv for the sum $Y_1 + (-Y_2)$, whose second term has density $f(-z)$, so the copy of $f$ is shifted but not flipped. The integrand is non-zero only where $y \ge 0$ and $y - x \ge 0$, that is for $y \ge \max(0, x)$, and there
$$f(y)f(y - x) = \frac{1}{b^2}e^{-y/b}e^{-(y - x)/b} = \frac{1}{b^2}e^{x/b}e^{-2y/b}.$$
The integral of $e^{-2y/b}$ from a point $c$ to infinity is $\frac b2e^{-2c/b}$. With $c = \max(0, x)$ the density of the difference is therefore $\frac{1}{2b}e^{x/b}e^{-2\max(0, x)/b}$. That is $\frac1{2b}e^{-x/b}$ when $x \ge 0$ and $\frac1{2b}e^{x/b}$ when $x < 0$, which together make $\frac1{2b}e^{-|x|/b}$, the $\mathrm{Lap}(b)$ density.

4. **Sums of exponentials.** Write $f_n(x) = x^{n-1}e^{-x}/(n-1)!$ for $x \ge 0$, the $\mathrm{Gamma}(n, 1)$ density. For $n = 1$ it is $e^{-x}$, the $\mathrm{Exp}(1)$ density. Suppose that the sum of $n$ independent $\mathrm{Exp}(1)$ variables has density $f_n$, and add one more. In @eq-apx-prob-conv, put the sum at $x$ and write $y$ for the value of the first term. Both terms are non-negative, so $y$ lies between 0 and $x$, and the two exponentials combine to $e^{-x}$, which leaves the integral:
$$\int_0^x\frac{y^{n-1}e^{-y}}{(n-1)!}\,e^{-(x - y)}\,dy = \frac{e^{-x}}{(n-1)!}\int_0^xy^{n-1}\,dy = \frac{e^{-x}}{(n-1)!}\cdot\frac{x^n}{n} = \frac{x^ne^{-x}}{n!}.$$
The result is $f_{n+1}$, so by induction the claim holds for every $n$.

5. **Sums of normals.** Start with independent $X_1 \sim \mathcal N(0, \sigma_1^2)$ and $X_2 \sim \mathcal N(0, \sigma_2^2)$, and write $v = \sigma_1^2 + \sigma_2^2$. In @eq-apx-prob-conv the two exponents add to $-\frac{y^2}{2\sigma_1^2} - \frac{(s - y)^2}{2\sigma_2^2}$. Completing the square in $y$ rewrites this sum as
$$-\frac{v}{2\sigma_1^2\sigma_2^2}\Big(y - \frac{\sigma_1^2}{v}s\Big)^2 - \frac{s^2}{2v},$$
which you can check by expanding both forms. The first term is the exponent of a normal density in $y$ with variance $\sigma_1^2\sigma_2^2/v$, so its integral over $y$ is $\sqrt{2\pi\sigma_1^2\sigma_2^2/v}$. Multiplying that by the two normalizing factors $1/(\sigma_1\sqrt{2\pi})$ and $1/(\sigma_2\sqrt{2\pi})$ leaves $1/\sqrt{2\pi v}$. So the sum has density $e^{-s^2/(2v)}/\sqrt{2\pi v}$, the $\mathcal N(0, v)$ density. Two simpler facts extend this to weighted sums with any means. First, $aX$ is $\mathcal N(a\mu, a^2\sigma^2)$ when $X \sim \mathcal N(\mu, \sigma^2)$ and $a > 0$, because $aX \le t$ is the event $X \le t/a$, and the derivative of its probability in $t$ is the density of $X$ at $t/a$ times $1/a$. A negative $a$ also flips the sign, which the symmetry of the normal about its mean absorbs. Second, adding a constant only moves the mean. So the two-term rule, applied one term at a time, gives the rule for $\sum_ia_iX_i$.

6. **Rotation invariance.** The coordinates of $X \sim \mathcal N(0, \sigma^2I_d)$ are independent, so its joint density is the product of $d$ normal densities, and the exponents add up to the squared length $\|x\|^2 = x_1^2 + \dots + x_d^2$:
$$p(x) = \prod_{i=1}^d\frac{e^{-x_i^2/(2\sigma^2)}}{\sigma\sqrt{2\pi}} = (2\pi\sigma^2)^{-d/2}e^{-\|x\|^2/(2\sigma^2)}.$$ {#eq-apx-prob-joint}
So the density depends on $x$ only through its distance from the origin, and its contours are spheres. In this appendix a **rotation** is any square matrix $O$ whose rows are perpendicular unit vectors, also called an orthogonal matrix (it includes reflections). It keeps every length, $\|Ox\| = \|x\|$, because the coordinates of $Ox$ are the components of $x$ along the rows of $O$. It also keeps every volume, because a matrix scales volumes by $|\det O|$, and $\det O = \pm1$ since $O^\top O = I$. Now let $Y = OX$. For any region $A$, $Y$ lands in $A$ exactly when $X$ lands in $O^{-1}A$, and the substitution $x = O^{-1}y$, which keeps volume, gives $\Pr[Y \in A] = \int_A p(O^{-1}y)\,dy$. By @eq-apx-prob-joint, $p(O^{-1}y)$ depends only on $\|O^{-1}y\| = \|y\|$, so it equals $p(y)$, and $Y$ has the same density as $X$.

7. **Perpendicular projections are independent.** Let $u$ and $v$ be perpendicular unit vectors. Choose a rotation $O$ whose first two rows are $u$ and $v$, with any unit vectors that complete a perpendicular set as the other rows. The first two coordinates of $Y = OX$ are then $\langle u, X\rangle$ and $\langle v, X\rangle$. By step 6, $Y \sim \mathcal N(0, \sigma^2I_d)$, so these two coordinates are independent $\mathcal N(0, \sigma^2)$ variables. This also proves the chapter's remark that zero covariance means independence for jointly normal variables, if "jointly normal" means "linear functions of one vector $G \sim \mathcal N(0, I_d)$ of independent standard normals". Two such variables $A = \langle a, G\rangle$ and $B = \langle b, G\rangle$ have covariance $\mathbb E[AB] = \sum_{i,j}a_ib_j\,\mathbb E[G_iG_j] = \langle a, b\rangle$, because $\mathbb E[G_iG_j]$ is 1 when $i = j$ and 0 otherwise. When that covariance is 0, $a/\|a\|$ and $b/\|b\|$ are perpendicular unit vectors, so $A$ and $B$ are multiples of two independent normals, and adding constant means keeps them independent.

8. **Two Laplace variables with the same scale.** The density $\frac\varepsilon4(1 + \varepsilon|z|)e^{-\varepsilon|z|}$ is the equal-scale case of the convolution in @sec-apx-prob-laplace-diff (step 5 there).

One more claim of the chapter needs no new proof. The chapter says that the sum of two independent $\mathrm{Lap}(b)$ variables is not Laplace, because its MGF $1/(1 - b^2\lambda^2)^2$ is not a Laplace MGF. That argument uses only the easy direction, that two variables with the same distribution have the same MGF, so it does not need the uniqueness theorem.

### Checking it

The script of this appendix (see the Code section at the end) evaluates each convolution numerically and compares it with the closed form. For Poisson counts with means 2 and 3, the largest gap to $\mathrm{Poisson}(5)$ is $3.3 \times 10^{-16}$, which is rounding error. The difference of two exponentials, integrated on a grid, is within $7.2 \times 10^{-5}$ of the Laplace density, and five exponentials are within $1.1 \times 10^{-3}$ of the Gamma density. Both gaps are the error of the grid, so they shrink as the grid gets finer. Normals with variances 1 and 4 match the density with variance 5 to $1.2 \times 10^{-12}$.

@fig-apx-prob-convolution draws the integral of step 3. In each of the first three panels, the green area under the product of $f(y)$ and its copy shifted by $x$ is the density of the difference at $x$. The overlap is largest, 0.5, at $x = 0$, where the two curves coincide. Moving the copy either way shrinks the overlap at the same exponential rate, which gives the two-sided Laplace shape of the last panel.

![The density of $Y_1 - Y_2$ for independent $\mathrm{Exp}(1)$ variables at $x = -1$, 0 and 1.5, as the area under $f(y)f(y - x)$ (green), next to the $\mathrm{Lap}(1)$ density with those three areas marked.](figures/apx_prob_convolution.svg){#fig-apx-prob-convolution}

For rotation invariance, the script projects a million draws from $\mathcal N(0, I_2)$ onto $u = (\cos\phi, \sin\phi)$ and the perpendicular $v$, with $\phi = \pi/6$. The projections have variances 0.9983 and 1.0010 and correlation $-0.0005$. Independence is a stronger claim than zero correlation, so the script also checks a joint probability. Both projections exceed 1 with chance 0.0251, and the product of the two separate chances is $(1 - \Phi(1))^2 = 0.0252$, where $\Phi$ is the standard normal cumulative distribution function (CDF). @fig-apx-prob-rotation shows the circular contours and the two matching histograms.

![Left: contours of the $\mathcal N(0, I_2)$ density, which are circles, with 400 sample points and two perpendicular unit vectors $u$ and $v$. Right: histograms of the projections $\langle u, X\rangle$ and $\langle v, X\rangle$ over 200,000 draws, against the $\mathcal N(0, 1)$ density.](figures/apx_prob_rotation.svg){#fig-apx-prob-rotation}

### What it does not cover

The derivations identify each sum that the book needs, one at a time. They do not prove the general theorem that an MGF finite near 0 determines the distribution, which needs a way to recover a distribution from its transform. That theorem stays cited to Blitzstein & Hwang (2019) and Billingsley (1995), and nothing in the book depends on it beyond the cases proved here.

## Stirling's formula and the peak of a binomial {#sec-apx-prob-stirling}

### What the chapter uses

Write $X \sim \mathrm{Bin}(n, p)$ for the number of successes in $n$ independent trials with success probability $p$, and write $\bar p = 1 - p$ and $p_k = \Pr[X = k] = \binom nkp^k\bar p^{n-k}$. Four sections of the attacks part use four facts about these probabilities, which this section proves. First, the probabilities rise and then fall, with the most likely count at $\lfloor(n+1)p\rfloor$. So shifting $\mathrm{Bin}(m, \frac12)$ by one has total variation (TV) distance equal to the largest weight, as @sec-threat-average (@eq-threat-avg-worst) says, and the binomial is unimodal, as @sec-prop-scaling (@eq-prop-membership) says. Second, Stirling's formula holds with an explicit error $r_n$:
$$n! = \sqrt{2\pi n}\,\Big(\frac ne\Big)^ne^{r_n}, \qquad 0 < r_n \le \frac{1}{12n}.$$ {#eq-apx-prob-stirling}
@sec-memo-exposure (@eq-memo-null-mean) uses its rough form $\ln N! = N\ln N - N + O(\ln N)$. Third, given the exact sum of $m$ fair secret bits, the majority guess of @sec-threat-games recovers exactly the fraction
$$\frac12 + \frac12\binom{2j}{j}4^{-j}, \qquad j = \lfloor m/2\rfloor,$$ {#eq-apx-prob-recon}
of the bits on average, which is $\frac12 + 1/\sqrt{2\pi m}$ up to a relative error that vanishes. The chapter gets that approximation from the CLT, which alone does not give it, because the CLT is a statement about probabilities, not averages. Fourth, when $np\bar p \ge 4$, the height of the mode is close to the peak $1/\sqrt{2\pi np\bar p}$ of the normal curve with the same mean and variance, which @sec-prop-scaling uses and which gives the height $\sqrt{2/(\pi m)}$ of @sec-threat-average:
$$\Big|\ln\big(\max_kp_k\cdot\sqrt{2\pi np\bar p}\big)\Big| \le \frac{3.5}{np\bar p}.$$ {#eq-apx-prob-peak}

### The idea

The shape of the binomial comes from comparing each probability with the next one. Going from $k$ to $k + 1$ successes multiplies the probability by a ratio that gets smaller as $k$ grows. While the ratio is above 1 the probabilities climb, and once it drops below 1 they fall, so there is one peak. With $n = 4$ and $p = \frac12$ the ratios are $4, \frac32, \frac23, \frac14$, so the probabilities $\frac1{16}, \frac4{16}, \frac6{16}, \frac4{16}, \frac1{16}$ climb twice and fall twice.

### Derivation

The proof first finds the peak from the ratio of neighbours (steps 1 and 2). It then compares $\ln n!$ with the integral of $\ln x$ (step 3), shows that the remaining error settles to a constant (step 4), and finds that constant with an integral due to Wallis (steps 5 and 6). Steps 7 and 8 apply the result to the fraction recovered and to the height of the peak.

1. **The ratio of neighbours.** Dividing the binomial formula at $k + 1$ by the one at $k$, most factors cancel:
$$\frac{p_{k+1}}{p_k} = \frac{n - k}{k + 1}\cdot\frac p{\bar p}.$$
As $k$ grows the numerator $n - k$ falls and the denominator $k + 1$ rises, so the ratio decreases. It is above 1 exactly when $(n - k)p > (k + 1)\bar p$, which rearranges to $k < (n + 1)p - 1$. So the probabilities rise while $k + 1 < (n + 1)p$ and do not rise after. The last count reached by a rise is the largest whole number not above $(n + 1)p$, so the largest probability is at $k^* = \lfloor(n + 1)p\rfloor$. When $(n + 1)p$ is a whole number the ratio there is exactly 1, and $k^* - 1$ ties with $k^*$.

2. **Shifting by one costs the height of the peak.** This completes the argument of @sec-threat-average. The shifted list is the distribution of $X + 1$, with $\Pr[X + 1 = k] = p_{k-1}$, and the TV distance is the total of $p_{k-1} - p_k$ over the $k$ where it is positive. By step 1 those are exactly the $k$ after the peak, $k = k^* + 1, \dots, n + 1$, with $p_{n+1} = 0$. In the sum $(p_{k^*} - p_{k^*+1}) + (p_{k^*+1} - p_{k^*+2}) + \dots + (p_n - p_{n+1})$, each negative part cancels the positive part of the next difference, and only $p_{k^*}$ survives. So the TV distance equals $\max_kp_k$.

3. **A rough Stirling formula.** The function $\ln x$ increases, so on the interval $[k - 1, k]$ it lies below $\ln k$, and on $[k, k + 1]$ it lies above. Summing over $k = 2, \dots, n$, and using $\ln 1 = 0$, gives
$$\int_1^n\ln x\,dx \le \ln n! \le \int_1^{n+1}\ln x\,dx.$$
The integral of $\ln x$ is $x\ln x - x$, so the two sides are $n\ln n - n + 1$ and $(n + 1)\ln(n + 1) - n$. The upper side exceeds $n\ln n - n$ by $\ln(n + 1) + n\ln(1 + 1/n)$, and the second part is at most 1 because $\ln(1 + u) \le u$. So $\ln n!$ is $n\ln n - n$ up to an error between 0 and $\ln(n + 1) + 1$, which is the form @sec-memo-exposure uses:
$$\ln n! = n\ln n - n + O(\ln n).$$

4. **The error settles to a constant.** Define $d_n = \ln n! - (n + \frac12)\ln n + n$, the error left after taking out $n\ln n - n$ and also $\frac12\ln n$. The extra $\frac12\ln n$ is what makes the remaining error change by a summable amount, as the next lines show. Subtracting $d_{n+1}$ from $d_n$, the factorials leave $-\ln(n + 1)$ and most terms cancel:
$$d_n - d_{n+1} = \Big(n + \frac12\Big)\ln\frac{n + 1}{n} - 1.$$
Write $u = 1/(2n + 1)$. Then $\frac{n + 1}{n} = \frac{1 + u}{1 - u}$ and $n + \frac12 = \frac1{2u}$. Subtracting the power series of $\ln(1 - u)$ from that of $\ln(1 + u)$ cancels the even powers and gives $\ln\frac{1 + u}{1 - u} = 2(u + \frac{u^3}3 + \frac{u^5}5 + \cdots)$. Multiplying by $\frac1{2u}$ and subtracting 1 gives
$$d_n - d_{n+1} = \frac{u^2}{3} + \frac{u^4}{5} + \frac{u^6}{7} + \cdots.$$
Every term is positive, so $d_n$ decreases. The denominators 5, 7, … are larger than 3, so the sum is at most $\frac{u^2}3(1 + u^2 + u^4 + \cdots) = \frac{u^2}{3(1 - u^2)}$. Since $1 - u^2 = \frac{4n(n+1)}{(2n+1)^2}$, this equals $\frac{1}{12n(n + 1)}$, which is $\frac1{12}\big(\frac1n - \frac1{n+1}\big)$. Adding these steps from $n$ onwards, the right sides telescope, so $d_n - d_{n+k} < \frac1{12n}$ for every $k$. A decreasing sequence that never drops more than a fixed amount has a limit, so $d_n$ tends to a constant $d_\infty$, with $0 < d_n - d_\infty \le \frac1{12n}$. Once step 6 shows that $d_\infty = \ln\sqrt{2\pi}$, this is @eq-apx-prob-stirling with $r_n = d_n - d_\infty$.

5. **Wallis's integrals.** Let $I_n = \int_0^{\pi/2}\sin^nx\,dx$, so $I_0 = \pi/2$ and $I_1 = 1$. Integrating by parts, with $\sin^nx = \sin^{n-1}x\cdot\sin x$, gives $I_n = (n - 1)\int_0^{\pi/2}\sin^{n-2}x\cos^2x\,dx$. Writing $\cos^2x = 1 - \sin^2x$ turns this into $(n - 1)(I_{n-2} - I_n)$, so $I_n = \frac{n-1}{n}I_{n-2}$. Running this down to $I_0$ or $I_1$ gives products of fractions such as $\frac{1\cdot3\cdots(2j-1)}{2\cdot4\cdots2j}$, which equals $\binom{2j}j4^{-j}$. So
$$I_{2j} = \frac\pi2\binom{2j}{j}4^{-j}, \qquad I_{2j+1} = \frac{1}{(2j + 1)\binom{2j}{j}4^{-j}}.$$
Since $0 \le \sin x \le 1$, the integrals decrease in $n$. So $1 \le I_{2j}/I_{2j+1} \le I_{2j-1}/I_{2j+1} = \frac{2j+1}{2j}$, and the ratio tends to 1. The ratio is $\frac\pi2(2j + 1)\big(\binom{2j}j4^{-j}\big)^2$, so
$$\binom{2j}{j}4^{-j} = \frac{1 + o(1)}{\sqrt{\pi j}},$$ {#eq-apx-prob-central}
where $o(1)$ is a quantity that tends to 0 as $j$ grows.

6. **The constant.** Write $\ln\binom{2j}{j} = \ln(2j)! - 2\ln j!$, and replace each $\ln n!$ by $d_n + (n + \frac12)\ln n - n$, which is the definition of step 4 solved for $\ln n!$. With $\ln(2j) = \ln 2 + \ln j$, the terms in $j\ln j$ and $j$ cancel, and what is left is
$$\ln\Big(\binom{2j}{j}4^{-j}\Big) = d_{2j} - 2d_j + \tfrac12\ln 2 - \tfrac12\ln j.$$
As $j$ grows, $d_{2j} - 2d_j \to -d_\infty$, and by @eq-apx-prob-central the left side is $-\frac12\ln(\pi j) + o(1)$. Matching the two gives $-d_\infty + \frac12\ln 2 = -\frac12\ln\pi$, so $d_\infty = \frac12\ln(2\pi) = \ln\sqrt{2\pi}$.

7. **The fraction recovered, exactly.** The majority guess gets $\max(K, m - K)$ bits right, so the expected fraction is $\frac12 + \mathbb E|K - \frac m2|/m$, with $K \sim \mathrm{Bin}(m, \frac12)$. One identity does all the work:
$$\Big(k - \frac m2\Big)\binom mk = \frac m2\Big[\binom{m-1}{k-1} - \binom{m-1}{k}\Big].$$
It holds because $\binom{m-1}{k-1} = \frac km\binom mk$ and $\binom{m-1}{k} = \frac{m-k}m\binom mk$. By symmetry about $\frac m2$, $\mathbb E|K - \frac m2|$ is twice the sum of $(k - \frac m2)\binom mk2^{-m}$ over $k > \frac m2$. Let $k_0$ be the first whole number above $\frac m2$. After the identity replaces each term, the sum from $k_0$ to $m$ telescopes as in step 2, and only $\frac m2\binom{m-1}{k_0-1}2^{-m}$ is left. So $\mathbb E|K - \frac m2|/m = \binom{m-1}{k_0-1}2^{-m}$. Now write $j = \lfloor m/2\rfloor$, so that $k_0 = j + 1$. When $m = 2j$ is even, $\binom{2j-1}{j} = \frac12\binom{2j}{j}$, and when $m = 2j + 1$ is odd, $\binom{2j}{j}2^{-2j-1}$ appears directly. Either way $\mathbb E|K - \frac m2|/m = \frac12\binom{2j}{j}4^{-j}$, which is @eq-apx-prob-recon. By @eq-apx-prob-central and $j \approx m/2$, the excess over $\frac12$ is $\frac{1 + o(1)}{2\sqrt{\pi m/2}} = \frac{1 + o(1)}{\sqrt{2\pi m}}$, which is the chapter's approximation.

8. **The height of the peak for any $p$.** Write $k = k^*$ for the mode and $a = k/n$ for the fraction of successes there. Replace each of $n!$, $k!$ and $(n - k)!$ in $p_k$ by @eq-apx-prob-stirling. The square roots give $\sqrt{n/(2\pi k(n - k))}$, and the powers $n^n$, $k^k$, $(n-k)^{n-k}$, $p^k$ and $\bar p^{n-k}$ combine into $\exp\big(-k\ln\frac{k}{np} - (n - k)\ln\frac{n-k}{n\bar p}\big)$. Multiplying by $\sqrt{2\pi np\bar p}$ and grouping gives three factors:
$$p_k\sqrt{2\pi np\bar p} = \underbrace{\sqrt{\frac{p\bar p}{a(1 - a)}}}_{A}\cdot\underbrace{e^{-n\,\mathrm{kl}(a\|p)}}_{B}\cdot\underbrace{e^{r_n - r_k - r_{n-k}}}_{R}.$$
Here $\mathrm{kl}(a\|p) = a\ln\frac ap + (1 - a)\ln\frac{1-a}{\bar p}$ is the Kullback–Leibler (KL) divergence between coins with success probabilities $a$ and $p$, which is never negative. By step 1, $k^*$ lies between $(n + 1)p - 1$ and $(n + 1)p$, so the mode is within one count of the mean: the offset $\xi = k^* - np$ satisfies $|\xi| < 1$. That makes each factor close to 1.

    - *Factor $B$.* The logarithm is concave, so by Jensen's inequality $\mathrm{kl}(a\|p) \le \ln\big(\frac{a^2}p + \frac{(1-a)^2}{\bar p}\big) = \ln\big(1 + \frac{(a-p)^2}{p\bar p}\big) \le \frac{(a-p)^2}{p\bar p}$. With $a - p = \xi/n$, this gives $0 \le -\ln B = n\,\mathrm{kl}(a\|p) \le \xi^2/(np\bar p) < 1/(np\bar p)$.
    - *Factor $R$.* By @eq-apx-prob-stirling, $|\ln R| \le \frac1{12}\big(\frac1n + \frac1k + \frac1{n-k}\big)$. When $np\bar p \ge 4$, $k > np - 1 \ge np/2$ and $n - k > n\bar p - 1 \ge n\bar p/2$. With $\frac1n, \frac1{np}, \frac1{n\bar p} \le \frac1{np\bar p}$, that gives $|\ln R| \le \frac{1}{12}\cdot\frac{5}{np\bar p}$.
    - *Factor $A$.* With $a = p + \xi/n$ and $1 - a = \bar p - \xi/n$, multiplying out gives $a(1 - a) = p\bar p(1 + x)$ with $x = \frac{\xi(\bar p - p)}{np\bar p} - \frac{\xi^2}{n^2p\bar p}$, so $|x| \le \frac{2}{np\bar p} \le \frac12$. For $|x| \le \frac12$, $|\ln(1 + x)| \le 2|x|$, so $|\ln A| = \frac12|\ln(1 + x)| \le \frac{2}{np\bar p}$.

    Adding the three logarithms gives $(2 + 1 + \frac5{12})/(np\bar p) < 3.5/(np\bar p)$, which is @eq-apx-prob-peak. With $n - 1$ trials and the chapter's subgroup share $t$ as the success probability, it gives @eq-prop-membership, and for $p = \frac12$ the height $\sqrt{2/(\pi m)}$ of @sec-threat-average.

### Checking it

@fig-apx-prob-shift-tv shows step 2 for $\mathrm{Bin}(10, \frac12)$. The green bars, where the shifted list is larger, start right after the peak and add up to 0.2461, the height of the mode. For $\mathrm{Bin}(100, \frac12)$ the same total is 0.0796, the value in @sec-threat-average. The script also checks step 1 by brute force: for every $n \le 60$ and eight values of $p$, $\lfloor(n+1)p\rfloor$ is a most likely count in all 480 cases.

![Top: the $\mathrm{Bin}(10, \frac12)$ probabilities and the same list moved one place right. Bottom: their difference at each $k$. The positive differences (green) all lie after the peak and add up to the height of the mode.](figures/apx_prob_shift_tv.svg){#fig-apx-prob-shift-tv}

@fig-apx-prob-stirling shows that Stirling's error $r_n$ is close to its bound $1/(12n)$ everywhere. At $n = 10$ it is $8.331 \times 10^{-3}$ against $8.333 \times 10^{-3}$, so $1/(12n)$ is nearly the exact error, not just a bound. The rough bounds of step 3 are far weaker.

![The error of Stirling's formula, $r_n = \ln n! - \ln\big(\sqrt{2\pi n}(n/e)^n\big)$, against $n$ on log axes, with the bound $1/(12n)$ and the gap between the rough bounds of step 3.](figures/apx_prob_stirling.svg){#fig-apx-prob-stirling}

For the fraction of bits recovered, @eq-apx-prob-recon gives 0.62305, 0.53979 and 0.51261 for $m = 10$, 100 and 1000, which match the sums over the binomial table, against 0.62616, 0.53989 and 0.51262 for the approximation $\frac12 + 1/\sqrt{2\pi m}$.

@fig-apx-prob-peak shows @eq-apx-prob-peak. The ratio of the mode height to $1/\sqrt{2\pi np\bar p}$ approaches 1 for $p = 0.5$, 0.2 and 0.05, and its logarithm falls like $1/(np\bar p)$. The constant 3.5 of the proof is loose: over every $n \le 20{,}000$ and six values of $p$ with $np\bar p \ge 4$, the script finds $np\bar p\,|\ln(\text{ratio})|$ at most 0.187.

![Left: the height of the binomial mode divided by the normal peak height $1/\sqrt{2\pi np\bar p}$, against $n$, for three values of $p$. Right: the absolute logarithm of that ratio against $np\bar p$, with the proved bound $3.5/(np\bar p)$.](figures/apx_prob_peak.svg){#fig-apx-prob-peak}

## How fast a Poisson count falls far above its mean {#sec-apx-prob-poisson-tail}

### What the chapter uses

@sec-dist-poisson in [Probability distributions](../part01-prerequisites/01-distributions.md) defines a Poisson count $N \sim \mathrm{Poisson}(\mu)$ with mean $\mu > 0$, which takes each value $k = 0, 1, 2, \dots$ with probability
$$\Pr[N = k] = e^{-\mu}\frac{\mu^k}{k!}.$$
It says that far above the mean these probabilities fall like $(e\mu/k)^k$, faster than any exponential but slower than the tail of a normal variable with the same variance. This section proves that for every count $k \ge 1$,
$$\Pr[N = k] \le \Big(\frac{e\mu}{k}\Big)^k,$$ {#eq-apx-prob-poisson-pmf}
and that, once $k$ is larger than $e\mu$, the whole tail obeys the same bound up to a constant factor:
$$\Pr[N \ge k] \le \frac{(e\mu/k)^k}{1 - e\mu/k}.$$ {#eq-apx-prob-poisson-tail}

### The idea

The probability of the count $k$ has $k!$ in its denominator, so a lower bound on $k!$ gives an upper bound on the probability. The lower bound comes from the power series of the exponential, which writes $e^k$ as a sum of positive terms $k^j/j!$, one of which is $k^k/k!$. A sum of positive terms is at least any one of its terms, so $k^k/k!$ is at most $e^k$. For $k = 3$, for example, the term is $27/6 = 4.5$, while the whole sum is $e^3 \approx 20.09$.

### Derivation

1. **A lower bound on the factorial.** The power series of the exponential at $k$ is $e^k = \sum_{j \ge 0}k^j/j!$. Every term is positive, so the sum is at least its term $j = k$, which is $k^k/k!$. Solving $k^k/k! \le e^k$ for $k!$ gives
$$k! \ge \Big(\frac ke\Big)^k.$$
2. **Bound one probability.** Replacing $k!$ in the denominator by this smaller number can only make the fraction larger, so
$$\Pr[N = k] \le e^{-\mu}\mu^k\Big(\frac ek\Big)^k.$$
3. **Drop the constant.** The mean $\mu$ is positive, so the factor $e^{-\mu}$ is below 1, and dropping it gives @eq-apx-prob-poisson-pmf.
4. **Faster than any exponential.** To compare the bound with an exponential decay, write it as an exponential, using $x^k = e^{k\ln x}$ and $\ln(1/x) = -\ln x$:
$$\Big(\frac{e\mu}{k}\Big)^k = e^{-k\ln(k/(e\mu))}.$$
The exponent is $k$ times a logarithm that keeps growing with $k$. So for any constant $c > 0$, once $k > e\mu\,e^c$ the logarithm exceeds $c$ and the bound is below $e^{-ck}$. The exponent grows like $k\ln k$, while that of $e^{-ck}$ grows only like $k$.
5. **A matching lower bound.** The comparison with the normal needs to know that the probabilities are not much smaller than the bound. The factorial $k! = 1 \cdot 2 \cdots k$ is a product of $k$ numbers, none larger than $k$, so $k! \le k^k$. Putting this larger number in the denominator gives
$$\Pr[N = k] \ge e^{-\mu}\Big(\frac{\mu}{k}\Big)^k = e^{-\mu - k\ln(k/\mu)}.$$
So the exponent of $\Pr[N = k]$ lies between $k\ln(k/(e\mu))$ and $\mu + k\ln(k/\mu)$, and both grow like $k\ln k$.
6. **Slower than a normal.** A normal variable with the Poisson's own mean $\mu$ and variance $\mu$ has density proportional to $e^{-(k - \mu)^2/(2\mu)}$ at $k$. After expanding the square, the leading term of the exponent is $k^2/(2\mu)$, so far out its tail falls like $e^{-k^2/(2\mu)}$. An exponent that grows like $k^2$ eventually overtakes one that grows like $k\ln k$, so by step 5 the Poisson probabilities fall more slowly than the normal tail.
7. **The whole tail.** Assume $k > e\mu$ and write $q = e\mu/k$, which is then below 1. The tail $\Pr[N \ge k]$ is the sum of $\Pr[N = j]$ over $j \ge k$, and by @eq-apx-prob-poisson-pmf each term is at most $(e\mu/j)^j$. For $j \ge k$ the base $e\mu/j$ is at most $q$, so each term is at most $q^j$. These terms form a geometric series with first term $q^k$ and ratio $q$, so
$$\Pr[N \ge k] \le \sum_{j \ge k}q^j = \frac{q^k}{1 - q},$$
which is @eq-apx-prob-poisson-tail. Once $k \ge 2e\mu$, the ratio $q$ is at most $\frac12$, so the factor $1/(1 - q)$ is at most 2.

### Checking it

Take $\mu = 4$, the chapter's example of a count with standard deviation 2. Then $e\mu \approx 10.87$, so the tail bound @eq-apx-prob-poisson-tail applies from $k = 11$. @tbl-apx-prob-poisson-tail compares the bounds with the exact values at $k = 20$ and $k = 40$.

| $k$ | $\Pr[N = k]$ | Bound @eq-apx-prob-poisson-pmf | Ratio | $\Pr[N \ge k]$ | Bound @eq-apx-prob-poisson-tail | Ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | $8.28 \times 10^{-9}$ | $5.09 \times 10^{-6}$ | 615 | $1.02 \times 10^{-8}$ | $1.11 \times 10^{-5}$ | 1093 |
| 40 | $2.71 \times 10^{-26}$ | $2.35 \times 10^{-23}$ | 867 | $3.01 \times 10^{-26}$ | $3.23 \times 10^{-23}$ | 1075 |

: Exact probabilities of $N \sim \mathrm{Poisson}(4)$ against the two bounds of this section. {#tbl-apx-prob-poisson-tail}

The bound on one probability is loose by a factor of several hundred, and Stirling's formula @eq-apx-prob-stirling accounts for the loss. It gives $k! = \sqrt{2\pi k}\,(k/e)^ke^{r_k}$, so step 1 drops the factor $\sqrt{2\pi k}\,e^{r_k}$ and step 3 drops $e^\mu$, and their product is the ratio in the table. That product grows only like $\sqrt k$, so in the exponent it is a small constant beside $k\ln(k/(e\mu))$, and the bound has the right rate.

The normal tail is far smaller. A normal variable with the same mean 4 and standard deviation 2 lies above 20, eight standard deviations out, with probability about $6.22 \times 10^{-16}$, against $1.02 \times 10^{-8}$ for the Poisson count. So a normal approximation to a count understates its far tail by many orders of magnitude. In @fig-apx-prob-poisson-tail, on a log scale, the two bounds run parallel to the exact tail, a constant factor above it, while the normal tail bends away below it.

![The tail $\Pr[N \ge k]$ of $N \sim \mathrm{Poisson}(4)$ on a log scale (black), with the bound @eq-apx-prob-poisson-pmf on one probability (dashed), the tail bound @eq-apx-prob-poisson-tail from $k = 11$ (orange), the Chernoff bound $e^{-\mu}(e\mu/k)^k$ (dotted), and the tail of a normal variable with mean 4 and variance 4 (dash-dotted).](figures/apx_prob_poisson_tail.svg){#fig-apx-prob-poisson-tail}

### What it does not cover

The Chernoff method of [Concentration inequalities](../part01-prerequisites/02-concentration.md) gives a sharper tail bound, because it keeps the factor $e^{-\mu}$ that step 3 drops and does not need $k > e\mu$. Steps 3 and 4 of the proof of @thm-conc-chernoff-bernoulli, applied to the Poisson MGF $\exp\big(\mu(e^\lambda - 1)\big)$, give, for every $k > \mu$,
$$\Pr[N \ge k] \le e^{-\mu}\Big(\frac{e\mu}{k}\Big)^k.$$
For the count with mean 4 it gives $9.32 \times 10^{-8}$ at $k = 20$, about 9 times the exact tail. This section's bound is simpler, and it shows that the rate $(e\mu/k)^k$ comes from the growth of $k!$.

## The central limit theorem, by swapping one term at a time {#sec-apx-prob-clt}

### What the chapter uses

The normal section of [Probability distributions](../part01-prerequisites/01-distributions.md) states the CLT without proof, and almost every normal approximation in the book rests on it, including the CLT for composition in @sec-gdp-clt, whose terms are not identically distributed. Take independent $X_1, \dots, X_n$ with means $\mu_i$, variances $\sigma_i^2$ and finite third moments. Write $Y_i = X_i - \mu_i$ for the centred terms and $s^2 = \sigma_1^2 + \dots + \sigma_n^2$ for the variance of their sum. The error is controlled by the **Lyapunov ratio**
$$L = \frac{\mathbb E|Y_1|^3 + \dots + \mathbb E|Y_n|^3}{s^3},$$
which is $\mathbb E|X - \mu|^3/(\sigma^3\sqrt n)$ for $n$ copies of one distribution with variance $\sigma^2$, so it falls like $1/\sqrt n$. This section proves that the standardized sum $S = (Y_1 + \dots + Y_n)/s$ has a CDF within $2L^{1/4}$ of the standard normal CDF $\Phi$:
$$\big|\Pr[S \le z] - \Phi(z)\big| \le 2L^{1/4} \quad\text{for every } z.$$ {#eq-apx-prob-clt}
So the gap tends to 0 whenever $L$ does, which proves the CLT when the third moments are finite.

### The idea

Compare the sum with a sum of normals, one term at a time. Next to each $Y_i$, put a normal variable $G_i$ with the same mean 0 and variance $\sigma_i^2$, so that the sum of the $G_i$, divided by $s$, is exactly $\mathcal N(0, 1)$. Then replace $G_1$ by $Y_1$, then $G_2$ by $Y_2$, and so on, until only the $Y_i$ remain. Each swap changes the average of a smooth function little, because the first two terms of a Taylor expansion depend on a term only through its mean and variance, which $Y_i$ and $G_i$ share. What is left after each swap is a third-order term of size about $n^{-3/2}$, so the $n$ swaps together cost about $n^{-1/2}$.

### Derivation

A probability $\Pr[S \le z]$ is the average of an indicator, which jumps, so it cannot be expanded in a Taylor series. The proof therefore first compares the averages $\mathbb E h(S)$ and $\mathbb E h(Z)$ of a smooth **test function** $h$ (steps 1 to 4), and then squeezes the indicator between two smooth ramps and pays for the width of the ramp (steps 5 to 7).

1. **The hybrids.** Let $G_1, \dots, G_n$ be independent normal variables, independent of the $X_i$, with $G_i \sim \mathcal N(0, \sigma_i^2)$. For $k = 0, \dots, n$, let
$$H_k = \frac{Y_1 + \dots + Y_k + G_{k+1} + \dots + G_n}{s}.$$
Then $H_n = S$. By step 5 of @sec-apx-prob-convolution, $H_0 = (G_1 + \dots + G_n)/s$ is exactly $\mathcal N(0, 1)$, since its variance is $s^2/s^2 = 1$. Write $Z = H_0$.

2. **Telescoping.** For any test function $h$, the change from $Z$ to $S$ is a sum of single swaps, because the middle terms cancel in pairs:
$$\mathbb E h(S) - \mathbb E h(Z) = \sum_{k=1}^n\big(\mathbb E h(H_k) - \mathbb E h(H_{k-1})\big).$$

3. **One swap.** The hybrids $H_k$ and $H_{k-1}$ differ only in term $k$. So if $W$ stands for the other terms divided by $s$, then $H_k = W + Y_k/s$ and $H_{k-1} = W + G_k/s$, and $W$ is independent of both $Y_k$ and $G_k$. Now suppose that $h$ has two derivatives, and that its second derivative $h''$ changes by at most $\gamma|a - b|$ between any two points $a$ and $b$, for a constant $\gamma$. With this condition, Taylor's theorem says that, for every $w$ and $y$,
$$\Big|h(w + y) - h(w) - h'(w)y - \tfrac12h''(w)y^2\Big| \le \frac\gamma6|y|^3.$$
Apply it with $y = Y_k/s$, take expectations, and do the same with $G_k$. The term $h(W)$ is common to both. The first-order term averages to 0 for both, because $W$ is independent of $Y_k$ and $Y_k$ has mean 0, so $\mathbb E[h'(W)Y_k] = \mathbb E[h'(W)]\,\mathbb E[Y_k] = 0$, and likewise for $G_k$. The second-order term averages to $\frac12\mathbb E[h''(W)]\,\sigma_k^2$ for both, because $Y_k$ and $G_k$ have the same variance. So these terms cancel and
$$\big|\mathbb E h(H_k) - \mathbb E h(H_{k-1})\big| \le \frac{\gamma}{6s^3}\big(\mathbb E|Y_k|^3 + \mathbb E|G_k|^3\big).$$

4. **Sum the swaps.** For a normal variable, integrating $|x|^3$ against the density gives $\mathbb E|G_k|^3 = 2\sqrt{2/\pi}\,\sigma_k^3 \approx 1.596\,\sigma_k^3$. And $\sigma_k^3 = (\mathbb E Y_k^2)^{3/2} \le \mathbb E|Y_k|^3$ by Jensen's inequality, because $x \mapsto x^{3/2}$ is convex. So each swap costs at most $\frac{\gamma}{6s^3}(1 + 2\sqrt{2/\pi})\mathbb E|Y_k|^3$, and by step 2, with $c = (1 + 2\sqrt{2/\pi})/6 \approx 0.433$,
$$\big|\mathbb E h(S) - \mathbb E h(Z)\big| \le c\,\gamma\,L.$$ {#eq-apx-prob-smooth}

5. **A smooth ramp.** The function $\rho(u) = 6u^5 - 15u^4 + 10u^3$ rises from $\rho(0) = 0$ to $\rho(1) = 1$. Its first two derivatives are 0 at both ends, and its third derivative is at most 60 in size on $[0, 1]$. Set $\rho = 0$ below 0 and $\rho = 1$ above 1. For a width $\kappa > 0$ and a point $z$, the upper ramp $h^+(x) = 1 - \rho((x - z)/\kappa)$ is 1 up to $z$ and 0 beyond $z + \kappa$. Its second derivative changes by at most $\gamma = 60/\kappa^3$ per unit, so @eq-apx-prob-smooth applies with that $\gamma$. The lower ramp $h^-(x) = h^+(x + \kappa)$ is the same ramp moved left by $\kappa$.

6. **Sandwich the indicator.** At every point, $h^-(x) \le \mathbf 1[x \le z] \le h^+(x)$, and averaging keeps the order. The upper ramp is at most $\mathbf 1[x \le z + \kappa]$, so
$$\Pr[S \le z] \le \mathbb E h^+(S) \le \mathbb E h^+(Z) + c\gamma L \le \Phi(z + \kappa) + c\gamma L.$$
The normal density is at most $1/\sqrt{2\pi}$, so $\Phi(z + \kappa) \le \Phi(z) + \kappa/\sqrt{2\pi}$. The lower ramp gives the matching bound from below, so
$$\big|\Pr[S \le z] - \Phi(z)\big| \le \frac{\kappa}{\sqrt{2\pi}} + \frac{60\,c\,L}{\kappa^3}.$$

7. **Choose the width.** A narrow ramp fits the indicator closely but has a large third derivative. The best balance sets the derivative in $\kappa$ to zero, which gives $\kappa^4 = 180\,c\,L\sqrt{2\pi}$. At that width the second term is a third of the first, and the bound is $\frac43\kappa/\sqrt{2\pi} = 1.99\,L^{1/4}$. That proves @eq-apx-prob-clt.

**Many dimensions.** @sec-apx-opt-mest and other sections use the CLT for random vectors. It follows from the one-dimensional theorem by the Cramér–Wold device: random vectors converge in distribution to a normal vector if and only if every fixed linear combination of the coordinates converges to the matching normal variable. Each linear combination is a sum of independent scalar terms, so the theorem above applies to it. The Cramér–Wold device itself rests on a uniqueness theorem for the Fourier transform of a distribution, so it stays cited (Billingsley, 1995).

### Checking it

@fig-apx-prob-lindeberg follows the swaps for $n = 30$ $\mathrm{Exp}(1)$ terms with $h = \sin$, whose third derivative is at most 1 in size. The script computes every hybrid's average $\mathbb E\sin(H_k)$ exactly from a closed form, with no simulation. The error grows in steps of at most 0.00122 to a total of 0.0365. Step 3 bounds each step by 0.00407 and the total by 0.1220, about three times the truth.

![The error $|\mathbb E\,h(H_k) - \mathbb E\,h(Z)|$ after swapping $k$ of $n = 30$ normal terms for centred $\mathrm{Exp}(1)$ terms, with $h = \sin$, computed exactly, against the sum of the per-swap bounds of step 3.](figures/apx_prob_lindeberg.svg){#fig-apx-prob-lindeberg}

@fig-apx-prob-smoothing draws the sandwich of step 6 at $z = 0$. The shaded area under the normal density is the price $\Phi(\kappa) - \Phi(0)$ of a ramp of width $\kappa = 0.5$.

![The indicator $\mathbf 1[x \le 0]$ between the upper and lower smooth ramps of widths $\kappa = 0.5$ and $0.2$, over the $\mathcal N(0, 1)$ density. The shaded area is $\Phi(0.5) - \Phi(0)$.](figures/apx_prob_smoothing.svg){#fig-apx-prob-smoothing}

The CDF bound is valid but slow. For the same 30 terms, @eq-apx-prob-clt gives 1.6201, which says nothing, while the true largest gap is 0.0243 and the Berry–Esseen bound $0.48L$ gives 0.2116. Even at $n = 10^8$ the bound is still 0.2478. So the proof gets the right answer in the limit, but not the right rate.

### What it does not cover

The proof assumes finite third moments, but the CLT holds with finite variance alone. Lindeberg's original argument handles that case by cutting each term into a bounded part and a small remainder. The rate is also weak, because a smooth ramp of width $\kappa$ must pay $\kappa^{-3}$ in its third derivative. The Berry–Esseen theorem improves $L^{1/4}$ to a constant times $L$, with a constant below 0.48 for identical terms (Shevtsova, 2011) and a constant of the same size for terms that are not identical. Its proof uses Fourier analysis, so it stays cited.

## The length of a normal vector stays near $\sqrt d$ {#sec-apx-prob-gauss-norm}

### What the chapter uses

Let $Z \sim \mathcal N(0, I_d)$, a vector of $d$ independent standard normal coordinates, and let $t \ge 0$. @sec-mean-multi uses
$$\Pr\big[\|Z\| \ge \sqrt d + t\big] \le e^{-t^2/2}$$ {#eq-apx-prob-norm}
to choose the radius $\tau_d$ that every record stays within and the margin for the noise in each round of @eq-mean-shrink. The chapter cites Boucheron, Lugosi & Massart (2013, ch. 5), and this section proves the bound with only the Chernoff method.

### The idea

The squared length $S = \|Z\|^2 = Z_1^2 + \dots + Z_d^2$ is a sum of $d$ independent terms, each with mean 1, so its mean is $d$. That is the setting of the Chernoff bound. Markov's inequality applied to $e^{\lambda(S - d)}$ gives a product of $d$ identical factors, each with a closed form. A well-chosen $\lambda$ then gives a bound on the squared length due to Laurent & Massart (2000), and square roots turn it into @eq-apx-prob-norm.

### Derivation

1. **One factor.** For $\lambda < \frac12$, combining the two exponents in the normal integral gives
$$\mathbb E\,e^{\lambda Z_1^2} = \int\frac{e^{-(1 - 2\lambda)x^2/2}}{\sqrt{2\pi}}\,dx = (1 - 2\lambda)^{-1/2},$$
because the integrand is a normal density with variance $1/(1 - 2\lambda)$, without its factor $\sqrt{1 - 2\lambda}$. So the logarithm of the MGF of $Z_1^2 - 1$ is $-\lambda - \frac12\ln(1 - 2\lambda)$.

2. **A simple upper bound.** The series $-\frac12\ln(1 - 2\lambda) = \sum_{k\ge1}\frac{(2\lambda)^k}{2k}$ starts with the term $\lambda$, which the $-\lambda$ cancels. For $0 \le \lambda < \frac12$ each later term is at most $\frac14(2\lambda)^k$, because $2k \ge 4$. Summing the geometric series gives
$$-\lambda - \tfrac12\ln(1 - 2\lambda) = \sum_{k\ge2}\frac{(2\lambda)^k}{2k} \le \frac{\lambda^2}{1 - 2\lambda}.$$

3. **The Chernoff bound.** The $d$ terms are independent, so the MGF of $S - d$ is the product of $d$ factors. Markov's inequality applied to $e^{\lambda(S - d)}$ gives, for any level $\ell > 0$ and any $0 < \lambda < \frac12$,
$$\Pr[S - d \ge \ell] \le \exp\Big(-\lambda\ell + \frac{d\lambda^2}{1 - 2\lambda}\Big).$$

4. **Choose the level and $\lambda$.** Write the level through a free parameter $x > 0$ as $\ell = 2\sqrt{dx} + 2x$, because that makes the final exponent simple. Write $r = \sqrt{x/d}$, so that $\sqrt{dx} = dr$ and $x = dr^2$. Choose $\lambda = r/(1 + 2r)$, which is below $\frac12$, so that $1 - 2\lambda = 1/(1 + 2r)$. Then $\frac{d\lambda^2}{1 - 2\lambda} = \frac{dr^2}{1 + 2r}$ and $\lambda\ell = \frac{2dr^2(1 + r)}{1 + 2r}$, so the exponent of step 3 is
$$\frac{-2dr^2(1 + r) + dr^2}{1 + 2r} = -dr^2\,\frac{1 + 2r}{1 + 2r} = -x.$$
So $\Pr[S - d \ge 2\sqrt{dx} + 2x] \le e^{-x}$.

5. **Back to the length.** Take $x = t^2/2$. If $\|Z\| \ge \sqrt d + t$, then squaring gives $S \ge d + 2\sqrt d\,t + t^2$. Here $2\sqrt{dx} = \sqrt2\sqrt d\,t$, which is at most $2\sqrt d\,t$, and $2x = t^2$, so $S - d \ge 2\sqrt{dx} + 2x$. So the event $\|Z\| \ge \sqrt d + t$ lies inside the event of step 4, and its probability is at most $e^{-x} = e^{-t^2/2}$.

### Checking it

The squared length $\|Z\|^2$ has the chi-squared distribution, whose tail the script computes exactly with the incomplete gamma function. @tbl-apx-prob-norm compares it with the bound.

| $t$ | $d = 1$ | $d = 10$ | $d = 100$ | $d = 1000$ | bound $e^{-t^2/2}$ |
| --- | --- | --- | --- | --- | --- |
| 1 | $4.55 \times 10^{-2}$ | $6.75 \times 10^{-2}$ | $7.51 \times 10^{-2}$ | $7.76 \times 10^{-2}$ | $6.07 \times 10^{-1}$ |
| 2 | $2.70 \times 10^{-3}$ | $2.96 \times 10^{-3}$ | $2.63 \times 10^{-3}$ | $2.44 \times 10^{-3}$ | $1.35 \times 10^{-1}$ |
| 3 | $6.33 \times 10^{-5}$ | $3.84 \times 10^{-5}$ | $1.98 \times 10^{-5}$ | $1.37 \times 10^{-5}$ | $1.11 \times 10^{-2}$ |

: Exact $\Pr[\|Z\| \ge \sqrt d + t]$ for $Z \sim \mathcal N(0, I_d)$, against the bound of @eq-apx-prob-norm. {#tbl-apx-prob-norm}

The bound holds in every dimension, and over $d$ up to 10,000 and $t$ up to 6 the exact tail is never more than 0.498 of it. It is loose, though. @fig-apx-prob-norm-tails shows that for large $d$ the exact tail falls like $e^{-t^2}$, twice as fast in the exponent. @fig-apx-prob-norm-density shows why: as $d$ grows, $\|Z\| - \sqrt d$ settles to a normal distribution with variance $\frac12$, so the spread of the length stays of order 1 while $\sqrt d$ grows. For @sec-mean-multi, what matters is that the bound does not depend on $d$.

![Exact $\Pr[\|Z\| \ge \sqrt d + t]$ for $d = 1$, 10, 100 and 1000 on a log scale, against the bound $e^{-t^2/2}$.](figures/apx_prob_norm_tails.svg){#fig-apx-prob-norm-tails}

![The density of $\|Z\| - \sqrt d$ for $d = 1$, 10, 100 and 1000, with a normal density of variance $\frac12$ for comparison. The length sits within a distance of order 1 from $\sqrt d$ in every dimension.](figures/apx_prob_norm_density.svg){#fig-apx-prob-norm-density}

## The difference of two Laplace variables {#sec-apx-prob-laplace-diff}

### What the chapter uses

@sec-pate-stable compares a Laplace noise $\nu \sim \mathrm{Lap}(b_1)$ with an independent noise $\rho \sim \mathrm{Lap}(b_2)$, where $b_1 = 4/\varepsilon_0$ and $b_2 = 2/\varepsilon_0$ for a budget $\varepsilon_0$. To keep the chance that any of $P$ tests wrongly passes below $\delta_{\text{bad}}$, it sets a threshold $\tau$ with $P\cdot\Pr[\nu - \rho > \tau] \le \delta_{\text{bad}}$, using the tail of @eq-pate-stable-threshold:
$$\Pr[\nu - \rho > \tau] = \frac{b_1^2e^{-\tau/b_1} - b_2^2e^{-\tau/b_2}}{2(b_1^2 - b_2^2)}, \qquad \tau \ge 0,\ b_1 \ne b_2.$$ {#eq-apx-prob-lapdiff}
The chapter checks the formula by simulation, and @sec-lower-packing uses the equal-scale case $b_1 = b_2 = 1/\varepsilon$, with density $\frac\varepsilon4(1 + \varepsilon|z|)e^{-\varepsilon|z|}$. This section does the integral for both.

### The idea

A Laplace variable is symmetric, so $\rho' = -\rho$ is again $\mathrm{Lap}(b_2)$, and $\nu - \rho = \nu + \rho'$ is a sum of two independent Laplace variables. Its density at $s$ is the convolution @eq-apx-prob-conv. Each Laplace density has a corner at 0, so the integral splits into three ranges of the integration variable $y$: below 0, between 0 and $s$, and above $s$. On each range both absolute values have a fixed sign, so the integrand is a single exponential with a one-line integral.

### Derivation

1. **Three pieces.** Fix $s \ge 0$, and write the density of $\nu + \rho'$ at $s$ as $\frac{1}{4b_1b_2}\int e^{-|y|/b_1 - |s - y|/b_2}\,dy$.
    - For $y < 0$, the exponent is $y/b_1 - (s - y)/b_2$. Its integral over $y < 0$ is $e^{-s/b_2}\cdot\frac{b_1b_2}{b_1 + b_2}$, because $\int_{-\infty}^0e^{y(1/b_1 + 1/b_2)}\,dy = \frac{1}{1/b_1 + 1/b_2}$.
    - For $y > s$, the exponent is $-y/b_1 - (y - s)/b_2$, and the same computation gives $e^{-s/b_1}\cdot\frac{b_1b_2}{b_1 + b_2}$.
    - For $0 \le y \le s$, the exponent is $-y/b_1 - (s - y)/b_2 = -s/b_2 + y(1/b_2 - 1/b_1)$. With $b_1 \ne b_2$ its integral is $\frac{b_1b_2}{b_1 - b_2}\big(e^{-s/b_1} - e^{-s/b_2}\big)$.

2. **The density.** Divide the three pieces by $4b_1b_2$ and collect the two exponentials. The coefficient of $e^{-s/b_1}$ is $\frac14\big(\frac1{b_1 + b_2} + \frac1{b_1 - b_2}\big) = \frac{b_1}{2(b_1^2 - b_2^2)}$, and that of $e^{-s/b_2}$ is $-\frac{b_2}{2(b_1^2 - b_2^2)}$. The density is symmetric, so for every $s$,
$$f(s) = \frac{b_1e^{-|s|/b_1} - b_2e^{-|s|/b_2}}{2(b_1^2 - b_2^2)}.$$
As a check, at $s = 0$ this is $\frac{1}{2(b_1 + b_2)}$, which is positive.

3. **The tail.** For $\tau \ge 0$, integrate $f$ from $\tau$ to infinity, using $\int_\tau^\infty b\,e^{-s/b}\,ds = b^2e^{-\tau/b}$. This gives @eq-apx-prob-lapdiff. At $\tau = 0$ it gives $\frac12$, as symmetry requires.

4. **Far out, the wider noise wins.** For large $\tau$ the term with the larger scale $b_1$ dominates, and the tail is about $\frac{b_1^2}{2(b_1^2 - b_2^2)}e^{-\tau/b_1}$. With $b_1 = 2b_2$ that is $\frac23e^{-\tau/b_1}$, which is $\frac43$ times the tail $\frac12e^{-\tau/b_1}$ of $\nu$ alone. So the narrower noise $\rho$ adds only a constant factor. To make $P\cdot\Pr[\nu - \rho > \tau] \le \delta_{\text{bad}}$ for $P$ queries, the threshold must be about $b_1\ln\big(2P/(3\delta_{\text{bad}})\big)$, which is the $4\ln(P/\delta_{\text{bad}})/\varepsilon_0$ growth the chapter describes.

5. **Equal scales.** When $b_1 = b_2 = b$, the middle piece of step 1 has the constant integrand $e^{-s/b}$ over an interval of length $s$, so it is $s\,e^{-s/b}$. The other two pieces are $\frac b2e^{-s/b}$ each. Dividing by $4b^2$,
$$f(s) = \frac{1}{4b}\Big(1 + \frac{|s|}{b}\Big)e^{-|s|/b}.$$
With $b = 1/\varepsilon$ this is the density $\frac\varepsilon4(1 + \varepsilon|z|)e^{-\varepsilon|z|}$ that @sec-lower-packing uses, and the factor $1 + |s|/b$ is why it is not a Laplace density. Integrating from $\tau$ gives the tail $\frac14(2 + \tau/b)e^{-\tau/b}$, which is also the limit of @eq-apx-prob-lapdiff as $b_1 \to b_2$.

### Checking it

Take the chapter's budget $\varepsilon_0 = 1$, so $b_1 = 4$ and $b_2 = 2$. At $\tau = 8$, @eq-apx-prob-lapdiff and a numerical integral of the density of step 2 both give 0.0872, and a million simulated pairs give 0.0867. Solving $P\cdot\Pr[\nu - \rho > \tau] = 5 \times 10^{-6}$ by bisection gives the chapter's thresholds, $\tau = 74.8$ for a pool of $P = 1{,}000$ queries and $\tau = 84.0$ for $P = 10{,}000$. The equal-scale density of step 5 matches a numerical convolution to $7.5 \times 10^{-9}$.

@fig-apx-prob-laplace-diff shows the tail on a log scale. It runs parallel to the tail of $\nu$ alone and slightly above it, as step 4 says. The threshold sits where the tail reaches $\delta_{\text{bad}}/P = 5 \times 10^{-10}$, far beyond the reach of a simulation, so only the closed form sets it accurately.

![The tail $\Pr[\nu - \rho > \tau]$ for $\nu \sim \mathrm{Lap}(4)$ and $\rho \sim \mathrm{Lap}(2)$ on a log scale: the formula (line), one million simulated pairs (dots), and the tails of $\nu$ and $\rho$ alone (dashed). The dotted lines mark the threshold for a pool of 10,000 queries and the level $5 \times 10^{-10}$.](figures/apx_prob_laplace_diff.svg){#fig-apx-prob-laplace-diff}

## Code

The script `code/appendix-derivations/apx_probability.py` checks every result of this appendix against an exact value, a numerical integral or a simulation, and draws the figures. Run it with

```bash
uv run python code/appendix-derivations/apx_probability.py
```

@tbl-apx-prob-code lists the main numbers it prints, with the default seed 0.

| Check | Result |
| --- | --- |
| Poisson, exponential, Gamma and normal sums against their closed forms | largest gaps $3.3 \times 10^{-16}$, $7.2 \times 10^{-5}$, $1.1 \times 10^{-3}$, $1.2 \times 10^{-12}$ |
| Projections of $\mathcal N(0, I_2)$ at $\phi = \pi/6$ | variances 0.9983 and 1.0010, correlation $-0.0005$ |
| Mode at $\lfloor(n + 1)p\rfloor$ | 480 of 480 cases |
| Stirling error at $n = 10$ against $1/(12n)$ | $8.331 \times 10^{-3}$ against $8.333 \times 10^{-3}$ |
| Fraction recovered, $m = 10$, 100, 1000 | 0.62305, 0.53979, 0.51261 |
| Shift TV of $\mathrm{Bin}(100, \frac12)$ and its mode height | both 0.0796 |
| Largest $np\bar p\,|\ln(\text{ratio})|$ for $np\bar p \ge 4$ | 0.187 (proved: at most 3.5) |
| $\mathrm{Poisson}(4)$ at $k = 20$ and 40: $\Pr[N = k]$ and its bound | $8.28 \times 10^{-9}$ and $5.09 \times 10^{-6}$; $2.71 \times 10^{-26}$ and $2.35 \times 10^{-23}$ |
| $\mathrm{Poisson}(4)$ at $k = 20$ and 40: $\Pr[N \ge k]$ and its bound | $1.02 \times 10^{-8}$ and $1.11 \times 10^{-5}$; $3.01 \times 10^{-26}$ and $3.23 \times 10^{-23}$ |
| Ratio $e^\mu\sqrt{2\pi k}\,e^{r_k}$ at $k = 20$ and 40 | 614.6 and 867.4 ($e^\mu\sqrt{2\pi k}$ alone: 612.0 and 865.6) |
| Normal tail with mean 4 and variance 4, above 20 and 40 | $6.22 \times 10^{-16}$ and $9.74 \times 10^{-73}$ |
| Chernoff bound on $\Pr[N \ge 20]$ | $9.32 \times 10^{-8}$, 9.1 times the exact tail |
| Swapping 30 $\mathrm{Exp}(1)$ terms, $h = \sin$ | error 0.0365, bound 0.1220 |
| CDF gap for 30 $\mathrm{Exp}(1)$ terms | exact 0.0243; bound 1.6201; Berry–Esseen 0.2116 |
| Normal length tail, largest ratio to $e^{-t^2/2}$ | 0.498 |
| Laplace difference tail at $\tau = 8$ | formula 0.0872, integral 0.0872, simulation 0.0867 |
| Thresholds for $P = 1{,}000$ and $10{,}000$ | 74.8 and 84.0 |

: The numbers printed by the appendix's script. {#tbl-apx-prob-code}

Two lessons stand out. The closed forms match their numerical checks to the accuracy of the grid or the simulation, so the derivations prove the formulas the chapters quote. And the bounds differ a lot in how tight they are. Stirling's $1/(12n)$ is nearly exact, and the peak-height bound is about 19 times too large in its constant. The norm bound loses a factor of 2 in the exponent for large $d$, and the CLT bound is far too weak to use for numbers. In the chapters, the CLT is used for its limit, while concrete error sizes come from the exact computations or from Berry–Esseen.

## References

- Blitzstein, J. K. & Hwang, J. (2019). *Introduction to Probability*, 2nd ed. Convolutions, the MGF and the uniqueness theorem at the level of this book.
- Billingsley, P. (1995). *Probability and Measure*, 3rd ed. Proofs of uniqueness for MGFs and characteristic functions, and the Cramér–Wold device.
- Feller, W. (1968). *An Introduction to Probability Theory and Its Applications*, Vol. 1, 3rd ed. The binomial ratio argument, Stirling's formula and Wallis's product.
- Robbins, H. (1955). "A remark on Stirling's formula". *American Mathematical Monthly* 62(1). The bounds $\frac1{12n+1} < r_n < \frac1{12n}$ on the error of Stirling's formula.
- Lindeberg, J. W. (1922). "Eine neue Herleitung des Exponentialgesetzes in der Wahrscheinlichkeitsrechnung". *Mathematische Zeitschrift* 15. The replacement argument, under the Lindeberg condition.
- Durrett, R. (2019). *Probability: Theory and Examples*, 5th ed. The CLT under the Lindeberg condition and its standard proof.
- Shevtsova, I. (2011). "On the absolute constants in the Berry–Esseen type inequalities for identically distributed summands". arXiv:1111.6554. The constant 0.48 quoted above.
- Laurent, B. & Massart, P. (2000). "Adaptive estimation of a quadratic functional by model selection". *Annals of Statistics* 28(5). The chi-squared tail bound of @sec-apx-prob-gauss-norm, in Lemma 1.
- Boucheron, S., Lugosi, G. & Massart, P. (2013). *Concentration Inequalities: A Nonasymptotic Theory of Independence*. Chapter 5 proves the same normal-length bound through Gaussian concentration of functions that change slowly.
