# Composition of trade-off functions

Two chapters of Part 4 compose differential privacy (DP) guarantees step by step, [f-DP and Gaussian DP](../part04-privacy-accounting/04-gaussian-dp.md) and [Numerical accounting](../part04-privacy-accounting/06-numerical-accounting.md), and they lean on five results that this appendix derives. Each result is about how well an attacker can tell two output distributions apart, which the chapters measure by the trade-off curve of the attacker's two kinds of error. The first says that any curve with the four properties of a trade-off curve is the curve of some pair of distributions. The next two say that running mechanisms one after another, even when each is chosen after seeing the earlier outputs, leaks no more than running fixed, independent copies of the worst step, once with trade-off curves and once with hockey-stick divergences. The last two are central limit theorems: with many small steps, the composed curve comes close to the curve of two normal distributions with the same variance. The first three share one move, which replaces an attacker's test on one pair, piece by piece, by a test on another pair that does at least as well. The central limit theorems instead follow the best tests of the composed pair and watch their errors approach those of the two normal distributions.

@tbl-apx-comp-map lists the five derivations and where the book states or uses each one.

| Derivation | What it proves | Stated or used in |
| --- | --- | --- |
| @sec-apx-comp-canonical | Every convex, non-increasing, continuous $f \le 1 - \alpha$ is the trade-off curve of a pair | @sec-ht-tradeoff, @sec-gdp-fdp, @sec-gdp-composition |
| @sec-apx-comp-fdp | Composition of $f$-DP, adaptive steps included, and the tensor product is well defined | @sec-gdp-composition (@eq-gdp-compose) |
| @sec-apx-comp-dominating | Dominating pairs survive adaptive composition | @sec-num-convolution, @sec-num-discretize, @sec-choose-compare |
| @sec-apx-comp-clt | The $f$-DP central limit theorem with its Berry–Esseen band | @sec-gdp-clt (@eq-gdp-clt-mu) |
| @sec-apx-comp-sgd-clt | The central limit theorem for the Poisson-subsampled Gaussian | @sec-gdp-sgd (@eq-gdp-sgd-mu), @sec-num-dpsgd, @sec-choose-compare |

: The derivations of this appendix and the chapter sections that state or use them. {#tbl-apx-comp-map}

**Notation.** The notation is that of [f-DP and Gaussian DP](../part04-privacy-accounting/04-gaussian-dp.md). A mechanism $M$ runs either on a dataset $D$ or on a neighbouring dataset $D'$, written $D \sim D'$, which adds or removes one record (the results hold for any neighbouring relation). Write $P = M(D)$ and $Q = M(D')$ for the distributions of its output, with densities, or probabilities for discrete outputs, $p$ and $q$.

A **test** $\phi$ looks at the output $y$ and fires, deciding that the data was $D'$, with probability $\phi(y)$. How often it fires on $D$ is its type I error $\alpha = \mathbb E_P[\phi]$. How often it fires on $D'$ is its **power** $\mathbb E_Q[\phi]$, and how often it stays silent there is its type II error $\beta = 1 - \mathbb E_Q[\phi]$. The **trade-off function** $T(P, Q)$ gives, at each type I error $\alpha$, the smallest type II error that any test with type I error at most $\alpha$ can reach. A **pair** always means such a pair of distributions $(P, Q)$, and its curve means $T(P, Q)$. A mechanism is $f$-DP when the curve of every neighbouring pair lies on or above the curve $f$. The privacy loss of an output measures how much more likely the output is on $D$ than on $D'$,
$$L(y) = \ln\frac{p(y)}{q(y)}.$$

One fact from [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) carries most of the weight. The **Neyman–Pearson lemma** (@sec-ht-np) says that the best test at any type I error is a likelihood-ratio test, which fires where $q/p$ is above a threshold and stays silent where it is below. On outputs exactly at the threshold it may fire with any probability, which lets the test spend its type I error exactly when the threshold outputs carry positive mass. So for every $\alpha \in (0, 1]$ such a test exists with type I error exactly $\alpha$, and its type II error is $T(P, Q)(\alpha)$. At $\alpha = 0$ the best test fires only on outputs that $P$ never produces. In terms of the loss, $q/p = e^{-L}$, so the best tests fire when $L$ is small.

## Every trade-off function has a canonical pair {#sec-apx-comp-canonical}

### What the chapter uses

Every trade-off curve is non-increasing, convex and continuous, and lies on or below the line $1 - \alpha$ of the test that fires at random (@sec-gdp-fdp). The converse holds too. Let $f\colon [0, 1] \to [0, 1]$ have these four properties. Let $P_f$ be the uniform distribution on $[0, 1]$, and let $Q_f$ put mass $f(1 - x)$ on each interval $[0, x]$,
$$Q_f\big([0, x]\big) = f(1 - x) \quad\text{for } x \in [0, 1],$$
plus an **atom**, a single point carrying positive probability, at a point outside $[0, 1]$, say at 2, which carries the missing mass $1 - f(0)$. Then
$$T(P_f, Q_f) = f.$$ {#eq-apx-comp-canonical}
The pair $(P_f, Q_f)$ is the **canonical pair** of $f$. @sec-ht-tradeoff gives this construction in one paragraph, and the composition theorem of the next section needs it, because it starts from curves $f_1, \dots, f_k$ known only to have the four properties.

### The idea

The Gaussian curve $G_\mu$ is the curve of $\mathcal N(0, 1)$ against $\mathcal N(\mu, 1)$, and a mechanism whose neighbouring pairs all have curves on or above it satisfies Gaussian differential privacy (GDP) with parameter $\mu$, written $\mu$-GDP. For $f = G_1$ the canonical pair is easy to recognize. Let $Z$ be standard normal and $\Phi$ its distribution function. Then $P_f$ is the law, meaning the distribution, of $\Phi(Z)$, which is uniform on $[0, 1]$ because $\Pr[\Phi(Z) \le x] = \Pr[Z \le \Phi^{-1}(x)] = x$. And $Q_f$ is the law of $\Phi(Z + 1)$. Applying the increasing function $\Phi$ to both $\mathcal N(0, 1)$ and $\mathcal N(1, 1)$ relabels the outputs one to one, so it changes no test's errors, and the pair still has curve $G_1$. @fig-apx-comp-canonical shows this pair and the test at $\alpha = 0.1$.

### Derivation

The plan is to show that $Q_f$ is a distribution whose density rises from left to right (steps 1 and 2), so that the likelihood ratio rises too and the tests that fire on a top interval are the best tests (steps 3 and 4). Their errors are then $\alpha$ and $f(\alpha)$ (step 5).

1. **$Q_f$ is a probability distribution.** Write $F(x) = f(1 - x)$. As $x$ grows, $1 - x$ shrinks, and $f$ is non-increasing, so $F$ is non-decreasing. Property $f(\alpha) \le 1 - \alpha$ at $\alpha = 1$ forces $f(1) = 0$, so $F(0) = 0$, and $F(1) = f(0) \le 1$. The atom adds the missing mass $1 - f(0)$, so the total is 1.
2. **$Q_f$ has a non-decreasing density on $[0, 1]$.** This step uses three standard facts about a convex function, which are cited (Rockafellar 1970, §§23–24). It has a slope $f'(\alpha)$ at every point except at most countably many kinks, its slope never decreases as $\alpha$ grows, and it is the integral of its slope. So $F$ has density $F'(x) = -f'(1 - x)$, which is minus the slope of $f$ at $1 - x$. As $x$ grows, $1 - x$ shrinks, so the slope $f'(1 - x)$ can only become more negative, and the density $-f'(1 - x)$ can only grow. At a kink the density has a jump, and either one-sided value can stand for it.
3. **The likelihood ratio is non-decreasing.** The density of $P_f$ is 1 on $[0, 1]$, so the likelihood ratio $q/p$ equals the density of $Q_f$ there. At the atom, $P_f$ has no mass, so the ratio is infinite.
4. **The top-interval tests are likelihood-ratio tests.** Fix $\alpha \in (0, 1]$ and let $\phi_\alpha$ fire on the atom and on every $x > 1 - \alpha$. Take the threshold $\tau$ to be the density just to the left of $1 - \alpha$, which is well defined even when $1 - \alpha$ is a kink, because a non-decreasing function has a limit from each side. By step 3, the ratio is at least $\tau$ wherever $\phi_\alpha$ fires and at most $\tau$ wherever it is silent. Outputs with ratio exactly $\tau$ are ties, and the lemma allows any behaviour on ties. So $\phi_\alpha$ is a likelihood-ratio test, and no test with type I error at most $\alpha$ has a smaller type II error.
5. **Read off the errors.** The type I error of $\phi_\alpha$ is the uniform mass of $(1 - \alpha, 1]$, which is $\alpha$, since the atom has no $P_f$-mass. Its type II error is the $Q_f$-mass where it is silent, $Q_f([0, 1 - \alpha]) = F(1 - \alpha) = f(\alpha)$. So $T(P_f, Q_f)(\alpha) = f(\alpha)$ for $\alpha \in (0, 1]$. At $\alpha = 0$ the best test fires on the atom only, and it misses $F(1) = f(0)$.

### Checking it

The script of this appendix, [apx_composition.py](../../code/appendix-derivations/apx_composition.py), coarsens each canonical pair into 1000 equal cells of $[0, 1]$, plus the atom, and computes the trade-off curve of this finite pair exactly, by sorting the cells by likelihood ratio. Reporting only the cell is post-processing, so the coarse curve can only lie above $f$, and it still touches $f$ at the cell edges, where the top-interval tests are still available. For $f = G_1$ the largest gap is $2.0\times10^{-3}$, near $\alpha = 0$, where $G_1$ is steepest. The second example is the curve of an $(\varepsilon, \delta)$-DP guarantee, made of two straight lines,
$$f_{\varepsilon, \delta}(\alpha) = \max\big(0,\; 1 - \delta - e^\varepsilon \alpha,\; e^{-\varepsilon}(1 - \delta - \alpha)\big).$$
With $\varepsilon = 1$ and $\delta = 0.1$ its largest gap is $8.3\times10^{-5}$. Both gaps shrink as the cells shrink, as @eq-apx-comp-canonical predicts. In @fig-apx-comp-canonical (right), the density of $Q_f$ steps from $e^{-1}$ to $e$, minus the two slopes of $f_{1, 0.1}$, at $1 - \alpha^*$, where the two lines meet at $\alpha^* = 0.9/(1 + e) = 0.2420$.

![The canonical pair of two trade-off functions. Left: for $f = G_1$, the uniform $P_f$ and the density of $Q_f$, which is minus the slope of $f$ at $1 - x$; the test that fires on $(0.9, 1]$ has type I error 0.1 (blue area) and power $1 - G_1(0.1) = 0.3891$ (orange area). Right: for $f_{1, 0.1}$, the density of $Q_f$ steps from $e^{-1}$ to $e$ at $x = 1 - 0.2420$, and the missing mass $\delta = 0.1$ sits in an atom outside $[0, 1]$ that every good test fires on.](figures/apx_comp_canonical.svg){#fig-apx-comp-canonical}

## Composition of f-DP, adaptive steps included {#sec-apx-comp-fdp}

### What the chapter uses

Two independent mechanisms have a joint output with the **product** distribution $P \times P'$, whose coordinates are drawn independently. The **tensor product** of two trade-off functions $f = T(P, Q)$ and $g = T(P', Q')$ is the curve of the product pair,
$$f \otimes g = T(P \times P', Q \times Q'),$$
and it does not depend on which pairs represent $f$ and $g$. Suppose $M_1$ is $f_1$-DP and, for every value $y_1$ of its output, the second mechanism $M_2(\cdot\,; y_1)$ is $f_2$-DP. Then the pair of outputs $(y_1, y_2)$ satisfies
$$T\big((M_1, M_2)(D), (M_1, M_2)(D')\big) \ge f_1 \otimes f_2 \quad\text{for all neighbours } D \sim D',$$ {#eq-apx-comp-tensor}
and the same holds for $k$ steps with $f_1 \otimes \dots \otimes f_k$. @thm-gdp-composition states this (Dong, Roth & Su 2022, Theorem 3.2), proves the non-adaptive case through Blackwell's theorem, and sketches the adaptive case. This section proves the adaptive case, and with it that the tensor product is well defined, using only the Neyman–Pearson lemma.

### The idea

Take the adaptive example of [Rényi DP](../part04-privacy-accounting/01-renyi-dp.md) (@tbl-rdp-adaptive), two runs of randomized response, which reports a person's bit truthfully with some keep-probability and flips it otherwise. One person's bit is 1 in $D$ and 0 in $D'$. The first step keeps the bit with probability 0.75, and the second step's keep-probability is 0.75 if the first output was 1 and 0.6 if it was 0. A keep-probability closer to $\frac12$ hides the bit better, so each step's curve lies on or above $f$, the curve of randomized response with keep-probability 0.75. @tbl-apx-comp-adaptive lists the four joint outputs.

| $(y_1, y_2)$ | $p$, bit 1 ($D$) | $q$, bit 0 ($D'$) | $q/p$ |
| --- | --- | --- | --- |
| (1, 1) | 0.5625 | 0.0625 | 0.111 |
| (1, 0) | 0.1875 | 0.1875 | 1 |
| (0, 1) | 0.15 | 0.3 | 2 |
| (0, 0) | 0.1 | 0.45 | 4.5 |

: Two adaptive randomized responses: keep-probability 0.75, then 0.75 or 0.6 depending on $y_1$. {#tbl-apx-comp-adaptive}

The test that fires only on $(0, 0)$ has type I error $p(0, 0) = 0.1$ and type II error $1 - 0.45 = 0.55$. On the branch $y_1 = 0$ it fires when $y_2 = 0$, which happens with probability 0.4 on $D$ and 0.6 on $D'$. The best test of the fixed pair at type I error 0.4 fires on output 0, at type I error 0.25 and power 0.75, and on output 1 with probability 0.2, which adds the remaining 0.15 of type I error and $0.2 \times 0.25 = 0.05$ of power. So its power is 0.8, more than the branch test's 0.6. Swapping it in keeps the joint type I error at $0.25 \times 0.4 = 0.1$ and raises the power to $0.75 \times 0.8 = 0.6$, so the type II error drops from 0.55 to 0.4, which is exactly $f \otimes f$ at 0.1.

### Derivation

The plan is to swap the attacker's test on each branch, one value of the first output at a time, for the best test of a fixed pair with curve $f_2$ (steps 1 to 3). After the swap the second output no longer depends on the first, so the joint pair is a product, and the first coordinate is swapped the same way (step 4). Write the two joint distributions with the first output's law and the second output's law given the first: $P(y_1, y_2) = P_1(y_1)P_2(y_2 \mid y_1)$ and $Q(y_1, y_2) = Q_1(y_1)Q_2(y_2 \mid y_1)$. The assumption is that $T(P_1, Q_1) \ge f_1$, and that $T\big(P_2(\cdot \mid y_1), Q_2(\cdot \mid y_1)\big) \ge f_2$ for every $y_1$. By @sec-apx-comp-canonical there are pairs $(\tilde P_i, \tilde Q_i)$ with curve exactly $f_i$.

1. **The replacement step.** Suppose $T(P, Q) \ge T(\tilde P, \tilde Q)$. Then for every test $\phi$ of $P$ against $Q$ there is a test $\tilde\phi$ of $\tilde P$ against $\tilde Q$ with the same type I error and a type II error no larger:
$$\mathbb E_{\tilde P}[\tilde\phi] = \mathbb E_P[\phi] \qquad\text{and}\qquad 1 - \mathbb E_{\tilde Q}[\tilde\phi] \le 1 - \mathbb E_Q[\phi].$$ {#eq-apx-comp-replace}
To find it, let $\alpha = \mathbb E_P[\phi]$ and take $\tilde\phi$ to be the likelihood-ratio test of $\tilde P$ against $\tilde Q$ with type I error exactly $\alpha$, which exists by the Neyman–Pearson lemma. For $\alpha = 0$ it is the test that fires on every output where $\tilde p = 0$, which is best because firing anywhere else would cost type I error. Its type II error is $T(\tilde P, \tilde Q)(\alpha)$, which is at most $T(P, Q)(\alpha)$ by assumption. That is at most the type II error of $\phi$, because $\phi$ is one of the tests the minimum in $T(P, Q)(\alpha)$ ranges over.
2. **Swap the second coordinate, one branch at a time.** Fix any joint test $\phi(y_1, y_2)$. For a fixed $y_1$, the function $\phi(y_1, \cdot)$ is a test of $P_2(\cdot \mid y_1)$ against $Q_2(\cdot \mid y_1)$, and that pair's curve is at least $f_2 = T(\tilde P_2, \tilde Q_2)$. Step 1 gives a test $\tilde\phi_{y_1}$ of $\tilde P_2$ against $\tilde Q_2$ with the same type I error on this branch and at least the same power. Now replace the second output too. Write $z_2$ for an output of the fixed pair, drawn from $\tilde P_2$ or $\tilde Q_2$ independently of $y_1$, and define the new joint test $\phi'(y_1, z_2) = \tilde\phi_{y_1}(z_2)$. It is a test of the pair $P_1 \times \tilde P_2$ against $Q_1 \times \tilde Q_2$, in which the second output no longer depends on the first.
3. **Average over the first output.** The joint type I error of $\phi$ is the average over $y_1 \sim P_1$ of the branch type I errors, and step 2 kept each of these. So $\phi'$ has the same joint type I error:
$$\mathbb E_{P_1 \times \tilde P_2}[\phi'] = \mathbb E_{y_1 \sim P_1}\big[\mathbb E_{\tilde P_2}\tilde\phi_{y_1}\big] = \mathbb E_{y_1 \sim P_1}\big[\mathbb E_{P_2(\cdot \mid y_1)}\phi(y_1, \cdot)\big] = \mathbb E_P[\phi].$$
The joint power is the average over $y_1 \sim Q_1$ of the branch powers, and step 2 raised or kept each of them. So $\mathbb E_{Q_1 \times \tilde Q_2}[\phi'] \ge \mathbb E_Q[\phi]$, and the joint type II error did not grow.
4. **Swap the first coordinate.** The pair is now a product, so for each fixed $z_2$, the function $\phi'(\cdot, z_2)$ is a test of $P_1$ against $Q_1$. Step 1, with $T(P_1, Q_1) \ge f_1 = T(\tilde P_1, \tilde Q_1)$, swaps it for a test of $\tilde P_1$ against $\tilde Q_1$ with the same type I error and at least the same power. Averaging over $z_2 \sim \tilde P_2$ and $z_2 \sim \tilde Q_2$, as in step 3, gives a test $\phi''$ of $\tilde P_1 \times \tilde P_2$ against $\tilde Q_1 \times \tilde Q_2$ with the type I error of $\phi$ and a type II error no larger. The order of the two swaps matters: before step 2 the two outputs were dependent, so fixing the second output would have changed the distribution of the first.
5. **Conclude.** Take any $\alpha$ and any joint test $\phi$ with type I error at most $\alpha$. Steps 2 to 4 give a test $\phi''$ on the product pair with the same type I error and a type II error no larger. So the type II error of $\phi$ is at least $T(\tilde P_1 \times \tilde P_2, \tilde Q_1 \times \tilde Q_2)$ evaluated at that type I error. Trade-off curves are non-increasing, so it is also at least that curve at $\alpha$, which is $f_1 \otimes f_2(\alpha)$. This is @eq-apx-comp-tensor.
6. **The tensor product does not depend on the pairs.** Steps 2 to 5 never used that $P_2(\cdot \mid y_1)$ changes with $y_1$. So they also prove the non-adaptive statement: if $T(P_i, Q_i) \ge T(\tilde P_i, \tilde Q_i)$ for $i = 1, 2$, then $T(P_1 \times P_2, Q_1 \times Q_2) \ge T(\tilde P_1 \times \tilde P_2, \tilde Q_1 \times \tilde Q_2)$. If both pairs have curve exactly $f_i$, the assumption holds in both directions, so the two product curves are each at least the other, and they are equal.
7. **Many steps, by induction.** The $k$-fold tensor product $f_1 \otimes \dots \otimes f_k$ is the curve of the $k$-fold product pair, and by step 6 it does not depend on the pairs chosen. Suppose the first $k - 1$ steps together are $f_1 \otimes \dots \otimes f_{k-1}$-DP, which is the curve of the product pair $\tilde P_1 \times \dots \times \tilde P_{k-1}$ against $\tilde Q_1 \times \dots \times \tilde Q_{k-1}$. Treat the first $k - 1$ outputs as one first output, and step $k$, chosen after seeing all of them, as the second step. Steps 2 to 5 then give the curve of $\tilde P_1 \times \dots \times \tilde P_k$ against $\tilde Q_1 \times \dots \times \tilde Q_k$, which is $f_1 \otimes \dots \otimes f_k$.

### Checking it

For the example of @tbl-apx-comp-adaptive, the script computes both curves exactly, by sorting the outputs by likelihood ratio. On a grid of 100,001 type I errors the adaptive curve is never below the product curve. The two touch from $\alpha = 0.25$ on and differ most, by 0.2812, at $\alpha = 0.0625$, because the branch $y_1 = 0$ keeps the bit only with probability 0.6 and so leaks less than the theorem charges for it. The script also runs steps 2 and 4 on 20,000 random tests on the four joint outputs. No test had its type I error changed by more than $3.3\times10^{-16}$, which is rounding, and none had its type II error raised. @fig-apx-comp-replacement shows the replacement on single tests, and it shows that step 4 matters too. The test that fires only on $(1, 1)$ has type II error 0.9375 at type I error 0.5625, step 2 lowers it to 0.7708, and step 4 lowers it to 0.1042.

![Test replacement for two adaptive randomized responses. Left: step 2 on each branch, where the attacker's test (dot) is swapped for the best test of the fixed pair with curve $f_2$ at the same type I error (square). On branch $y_1 = 1$ the branch pair is the fixed pair itself, and on branch $y_1 = 0$ (keep 0.6) its curve lies above $f_2$. Right: four joint tests on the adaptive pair, after step 2 (triangles, shown when step 4 moves them further) and after steps 2 and 4 (squares); each moves straight down, onto or above $f_1 \otimes f_2$. The adaptive pair's own curve (grey, dashed) lies above $f_1 \otimes f_2$.](figures/apx_comp_replacement.svg){#fig-apx-comp-replacement}

### What it does not cover

With continuous outputs, the averages in step 3 make sense only if the joint test $\phi'$ is a measurable function of $(y_1, z_2)$, which roughly means that its averages are well defined. Choosing the canonical pair as $(\tilde P_2, \tilde Q_2)$ settles this. Its best test at type I error $\alpha$ is "fire when $z_2 > 1 - \alpha$ or $z_2$ is the atom", so $\phi'(y_1, z_2)$ is that formula with $\alpha$ replaced by the branch's type I error, which is itself an average of $\phi$ over $y_2$ and so a measurable function of $y_1$. For finite outputs, as in the examples, there is nothing to check. The proof does not need Blackwell's theorem, which says more: the curve of one pair lies on or above the curve of another exactly when the first pair is a post-processing of the second. That stronger statement stays cited.

## Dominating pairs survive adaptive composition {#sec-apx-comp-dominating}

### What the chapter uses

The **hockey-stick divergence** with a weight $\gamma \ge 0$ is
$$E_\gamma(P\|Q) = \sum_y\big(P(y) - \gamma Q(y)\big)_+,$$
where $P(y)$ and $Q(y)$ are the probabilities of output $y$ and $(x)_+$ is $x$ when $x$ is positive and 0 otherwise. (This section writes sums for outputs with finitely or countably many values. For continuous outputs every sum becomes an integral of densities.) At $\gamma = e^\varepsilon$ it is the smallest $\delta$ for which the pair satisfies $P(S) \le e^\varepsilon Q(S) + \delta$ for every set of outputs $S$. A pair $(P, Q)$ **dominates** a mechanism $M$ if $E_\gamma(M(D)\|M(D')) \le E_\gamma(P\|Q)$ for all neighbours and all $\gamma \ge 0$. Suppose the pair $(P_i, Q_i)$ dominates step $i$ for every value of the earlier outputs, and let $M$ be the adaptive composition of the $k$ steps. Then for every $\gamma \ge 0$,
$$E_\gamma\big(M(D)\,\big\|\,M(D')\big) \le E_\gamma\big(P_1 \times \dots \times P_k\,\big\|\,Q_1 \times \dots \times Q_k\big).$$ {#eq-apx-comp-dominating}
@sec-num-convolution cites this to Zhu, Dong & Wang (2022, Theorem 10). It is what lets a privacy loss distribution (PLD), rounded up to a grid, bound adaptive runs in step 6 of @sec-num-discretize, and what makes the PLD accountants of @sec-choose-compare valid for the private form of stochastic gradient descent (SGD), DP-SGD, whose steps depend on the model so far.

### The idea

In this section tests $\phi$ fire for $P$, as in the numerical-accounting chapter, because $E_\gamma(P\|Q)$ rewards finding $P$. Think of $E_\gamma$ as a best net gain: a test earns one unit each time it fires on a draw from $P$ and pays $\gamma$ each time it fires on a draw from $Q$. Draw each test as a point, with how often it fires under $Q$ across and under $P$ up. The upper edge $g$ of the region of all tests is concave, and a line of slope $\gamma$ that touches it meets the vertical axis at height $E_\gamma$. A concave edge is the lowest of its touching lines, so the values $E_\gamma$ at all slopes fix the edge, and domination puts one region inside the other.

Randomized response with keep-probability 0.75 shows the touching lines. Its upper edge runs from $(0, 0)$ to $(0.25, 0.75)$ with slope 3, and then to $(1, 1)$ with slope $1/3$. A line of slope $\gamma$ between $1/3$ and 3 touches the corner $(0.25, 0.75)$ and meets the vertical axis at $0.75 - 0.25\gamma$. The hockey-stick formula gives the same value, $E_\gamma = (0.75 - 0.25\gamma)_+ + (0.25 - 0.75\gamma)_+$, whose second term is 0 once $\gamma \ge 1/3$. At $\gamma = 1$, for example, both give 0.5.

### Derivation

The plan is to write $E_\gamma$ as a best gain (step 1) and the upper edge of the testing region as the lowest of its touching lines (steps 2 and 3). Domination then gives every test a counterpart on the dominating pair that is no worse in either error (step 4), and the branch-by-branch swap of @sec-apx-comp-fdp goes through (steps 5 and 6).

1. **The best gain is the hockey-stick divergence.** For every $\gamma \ge 0$,
$$E_\gamma(P\|Q) = \max_\phi\big(\mathbb E_P[\phi] - \gamma\,\mathbb E_Q[\phi]\big),$$
where $\phi$ ranges over tests with values in $[0, 1]$. The gain is $\sum_y\phi(y)\big(P(y) - \gamma Q(y)\big)$, and each term is largest when $\phi(y) = 1$ where $P(y) > \gamma Q(y)$ and $\phi(y) = 0$ elsewhere. That choice gives $\sum_y(P(y) - \gamma Q(y))_+$. This is @lem-div-best-event, and it is also @eq-gdp-gain with the roles of the two distributions swapped.
2. **The upper edge of the region.** For a pair $(\tilde P, \tilde Q)$ and $c \in [0, 1]$, let
$$g(c) = \max\big\{\mathbb E_{\tilde P}[\tilde\phi] : \mathbb E_{\tilde Q}[\tilde\phi] \le c\big\}.$$
This is the Neyman–Pearson problem with the two distributions swapped, so the maximum is reached by a likelihood-ratio test that fires where $\tilde p/\tilde q$ is large. A bigger budget allows more tests, so $g$ is non-decreasing. To see that $g$ is concave, take tests $\tilde\phi_1$ and $\tilde\phi_2$ that reach $g$ at budgets $c_1$ and $c_2$, and a weight $\lambda \in [0, 1]$. The mixed test $\lambda\tilde\phi_1 + (1 - \lambda)\tilde\phi_2$ mixes both of their averages ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). So it fires under $\tilde Q$ at most $\lambda c_1 + (1 - \lambda)c_2$ of the time and under $\tilde P$ exactly $\lambda g(c_1) + (1 - \lambda)g(c_2)$ of the time. So $g$ at the mixed budget is at least the mixed value, which is what concave means.
3. **The edge is the lowest of its touching lines.** For every $c \in [0, 1]$,
$$g(c) = \inf_{\gamma \ge 0}\big[E_\gamma(\tilde P\|\tilde Q) + \gamma c\big].$$ {#eq-apx-comp-envelope}
The right side is never below $g(c)$: for any test with $\mathbb E_{\tilde Q}[\tilde\phi] \le c$, step 1 gives $\mathbb E_{\tilde P}[\tilde\phi] \le E_\gamma + \gamma\,\mathbb E_{\tilde Q}[\tilde\phi] \le E_\gamma + \gamma c$, because $\gamma \ge 0$. To see that the right side comes down to $g(c)$, use a line that touches $g$ at $c$. For $0 < c < 1$, let $\gamma^*$ be the slope of $g$ just to the right of $c$. It is finite, because the slope of a concave function never increases, so it is at most the slope of the chord from 0 to $c$, which is at most $1/c$. It is at least 0 because $g$ is non-decreasing. A concave function lies below each of its touching lines (Rockafellar 1970, §23), so $g(c') \le g(c) + \gamma^*(c' - c)$ for every $c'$. Now any test $\tilde\phi$ has $\mathbb E_{\tilde P}[\tilde\phi] \le g(c')$ with $c' = \mathbb E_{\tilde Q}[\tilde\phi]$, so its gain at price $\gamma^*$ is at most $g(c') - \gamma^*c' \le g(c) - \gamma^*c$. By step 1, $E_{\gamma^*} + \gamma^*c \le g(c)$. At $c = 1$, the slope $\gamma^* = 0$ works, since $E_0 = 1 = g(1)$. At $c = 0$, the edge is $g(0) = \tilde P[\tilde q = 0]$, the mass that $\tilde P$ puts where $\tilde Q$ puts none, and $E_\gamma$ falls to exactly this value as $\gamma \to \infty$, because every output with $\tilde q > 0$ eventually contributes 0. So at $c = 0$ the infimum is reached only in the limit, which is why @eq-apx-comp-envelope uses $\inf$ rather than $\min$.
4. **Domination puts one region inside the other.** Suppose $E_\gamma(P\|Q) \le E_\gamma(\tilde P\|\tilde Q)$ for every $\gamma \ge 0$. Then for every test $\phi$ of $P$ against $Q$ there is a test $\tilde\phi$ of $\tilde P$ against $\tilde Q$ with
$$\mathbb E_{\tilde P}[\tilde\phi] \ge \mathbb E_P[\phi] \qquad\text{and}\qquad \mathbb E_{\tilde Q}[\tilde\phi] \le \mathbb E_Q[\phi].$$
The reason is that, by step 1 and the assumption, $\mathbb E_P[\phi] \le E_\gamma(P\|Q) + \gamma\,\mathbb E_Q[\phi] \le E_\gamma(\tilde P\|\tilde Q) + \gamma\,\mathbb E_Q[\phi]$ for every $\gamma \ge 0$. Taking the smallest right side, @eq-apx-comp-envelope gives $\mathbb E_P[\phi] \le g(\mathbb E_Q[\phi])$. The test that reaches $g$ at the budget $c = \mathbb E_Q[\phi]$ is the $\tilde\phi$ wanted.
5. **Swap branch by branch.** Take two adaptive steps. The letters $P_i$ and $Q_i$ already name the dominating pairs, so the run's own laws are $A_1(y_1)A_2(y_2 \mid y_1)$ on $D$ and $B_1(y_1)B_2(y_2 \mid y_1)$ on $D'$. The fixed pair $(P_2, Q_2)$ dominates $(A_2(\cdot \mid y_1), B_2(\cdot \mid y_1))$ on every branch, and $(P_1, Q_1)$ dominates $(A_1, B_1)$. Fix $\gamma \ge 0$ and a joint test $\phi$. Its gain is
$$\sum_{y_1}\Big(A_1(y_1)\,\mathbb E_{A_2(\cdot \mid y_1)}\phi(y_1, \cdot) - \gamma\,B_1(y_1)\,\mathbb E_{B_2(\cdot \mid y_1)}\phi(y_1, \cdot)\Big).$$
On each branch, step 4 gives a test $\tilde\phi_{y_1}$ of $P_2$ against $Q_2$ that fires at least as often under $P_2$ and at most as often under $Q_2$. The weights $A_1(y_1)$ and $\gamma\,B_1(y_1)$ are not negative, so the swap raises or keeps the first part of each term and lowers or keeps the subtracted part. So the gain does not fall, and the new test is a test of $A_1 \times P_2$ against $B_1 \times Q_2$.

   That pair is a product, so for each fixed second output the test is a test of $A_1$ against $B_1$. Step 4 swaps each of these tests for a test of $P_1$ against $Q_1$ in the same way, with the weights now the probabilities of the second output under $P_2$ and $\gamma Q_2$, which are again not negative. The result is a test of $P_1 \times P_2$ against $Q_1 \times Q_2$ with at least the gain of $\phi$.
6. **Conclude, and induct.** By step 1, no test of the product pair gains more than $E_\gamma$ of the product pair, so neither does $\phi$. Step 1 also says that the best gain over joint tests $\phi$ is $E_\gamma$ of the run's joint pair, so that divergence is at most $E_\gamma$ of the product pair, which is @eq-apx-comp-dominating for two steps. Treating the first $k - 1$ steps as one step, as in step 7 of @sec-apx-comp-fdp, extends it to $k$ steps.
7. **The same theorem in the language of curves.** Domination at every $\gamma \ge 0$ is the same as an order between trade-off curves, and the proof goes both ways through step 4. Switch back to tests that fire for $Q$, as in @sec-apx-comp-fdp, by reading $1 - \phi$ and $1 - \tilde\phi$. First suppose that the pair $(\tilde P, \tilde Q)$ dominates $(P, Q)$. For any $\phi$, step 4 gives a $\tilde\phi$, and the type I errors of $1 - \phi$ and $1 - \tilde\phi$ are $1 - \mathbb E_P[\phi]$ and $1 - \mathbb E_{\tilde P}[\tilde\phi]$, while their type II errors are $\mathbb E_Q[\phi]$ and $\mathbb E_{\tilde Q}[\tilde\phi]$. So the test $1 - \tilde\phi$ is no worse in either error, and since trade-off curves are non-increasing this gives $T(P, Q) \ge T(\tilde P, \tilde Q)$.

   Now suppose instead that $T(P, Q) \ge T(\tilde P, \tilde Q)$, and take any test $\phi$ that fires for $P$. The replacement step of @sec-apx-comp-fdp, applied to $1 - \phi$, gives a test $\chi$ of $\tilde P$ against $\tilde Q$ with the same type I error and a type II error no larger. Read back as a test that fires for $P$, $\tilde\phi = 1 - \chi$ fires under $\tilde P$ exactly as often as $\phi$ fires under $P$, and under $\tilde Q$ at most as often as $\phi$ fires under $Q$. So its gain is at least that of $\phi$ at every $\gamma \ge 0$, and by step 1, $E_\gamma(P\|Q) \le E_\gamma(\tilde P\|\tilde Q)$. So the theorem of this section is the theorem of @sec-apx-comp-fdp, read through hockey-stick divergences.

### Checking it

The three-step adaptive run of [Adaptive composition](../part04-privacy-accounting/08-adaptive-composition.md) (@tbl-adapt-three-step) starts with randomized response with keep-probability 0.75, followed by a strong step (keep 0.75) and a weak step (keep 0.6), in an order set by the first output. Randomized response with keep-probability 0.75 dominates both kinds of step, so the product of three such pairs dominates the run. The script checks $E_\gamma$ of both at 502 values of $\gamma$ in $[0, e^4]$, including 100 values below 1, and the product is never below the run. @tbl-apx-comp-three lists four of them.

| $\varepsilon = \ln\gamma$ | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| Adaptive run, $\delta(\varepsilon)$ | 0.5750 | 0.3926 | 0.1528 | 0.0000 |
| Product of three dominating pairs | 0.6875 | 0.4190 | 0.3064 | 0.1080 |

: $\delta(\varepsilon) = E_{e^\varepsilon}(P\|Q)$ of the three-step adaptive run and of the product of its dominating pairs. {#tbl-apx-comp-three}

The run's $\delta$ reaches 0 at its largest privacy loss. Along any path, the first step and the strong step each multiply the likelihood ratio by at most $0.75/0.25 = 3$, and the weak step by at most $0.6/0.4 = 1.5$, so the largest loss is $\ln(3 \times 3 \times 1.5) = \ln 13.5 = 2.6027$. The product's $\delta$ reaches 0 only at $3\ln 3 = 3.2958$, because it charges for three strong steps while no path of the run uses more than two (@fig-apx-comp-three-step). The script also checks step 4 directly: of 100,000 random tests on the run's eight transcripts, none lies above the product's upper edge $g$.

![The hockey-stick divergence $E_\gamma(P\|Q)$ against $\ln\gamma$ for the three-step adaptive run (dashed) and for the product of three randomized responses with keep-probability 0.75 (solid). The product lies on or above the run at every $\gamma \ge 0$, including $\gamma < 1$ (left of the dotted line).](figures/apx_comp_three_step.svg){#fig-apx-comp-three-step}

@fig-apx-comp-regions draws the regions for a pair with a curved edge. The dominating pair has $P = \mathcal N(1, 1)$ and $Q = \mathcal N(0, 1)$, and the dominated pair has $P = \mathcal N(0.6, 1)$ and the same $Q$, so its region lies inside. Three touching lines of the outer region, with slopes 0.5, 1 and $e$, meet the vertical axis at $E_\gamma = 0.5953$, 0.3829 and 0.1269, and the inner pair's values, 0.5253, 0.2358 and 0.0190, are all lower.

![Testing regions in the plane of $(\mathbb E_Q\phi, \mathbb E_P\phi)$. The grey region holds every test of $\mathcal N(1, 1)$ against $\mathcal N(0, 1)$; the region of $\mathcal N(0.6, 1)$ against $\mathcal N(0, 1)$ (green) lies inside it. Each dashed line has slope $\gamma$ and touches the outer edge at a dot; its height at 0 is $E_\gamma$ of the outer pair. Crosses on the vertical axis mark $E_\gamma$ of the inner pair, below each line.](figures/apx_comp_regions.svg){#fig-apx-comp-regions}

### What it does not cover

The theorem needs domination at every $\gamma \ge 0$, not only at $\gamma \ge 1$, because in step 3 the touching line at a large budget $c$ can have a slope below 1. The values of $E_\gamma$ below $\gamma = 1$ describe the reversed pair $(Q, P)$, because
$$E_\gamma(P\|Q) = 1 - \gamma + \gamma\,E_{1/\gamma}(Q\|P).$$
To see it, split each term as $(P - \gamma Q)_+ = (P - \gamma Q) + (\gamma Q - P)_+$, and note that $(\gamma Q - P)_+ = \gamma(Q - P/\gamma)_+$. Summing over $y$, the first part gives $1 - \gamma$, since $P$ and $Q$ each sum to 1, and the second gives $\gamma\,E_{1/\gamma}(Q\|P)$. So domination below $\gamma = 1$ is domination of the reversed pair at $1/\gamma > 1$. The theorem also assumes one fixed pair per step for every history. A pair that may change with the history is not covered, and that version stays with Zhu, Dong & Wang (2022).

## The f-DP central limit theorem {#sec-apx-comp-clt}

### What the chapter uses

@thm-gdp-clt in @sec-gdp-clt, the central limit theorem (CLT) for composition, describes the composition of symmetric curves $f_i = T(P_i, Q_i) = T(Q_i, P_i)$ through the moments of each step's privacy loss $L_i$ under $P_i$ (@lem-gdp-moments). The composed loss $L = L_1 + \dots + L_k$ has mean $K = \sum_i\mathrm{kl}(f_i)$ and variance $V = \sum_i\kappa_2(f_i) - \sum_i\mathrm{kl}(f_i)^2$ under $P$, and the summed third central moments are $\|\bar\kappa_3\|_1 = \sum_i\bar\kappa_3(f_i)$. Set
$$\mu = \frac{2K}{\sqrt V}, \qquad w = \frac{0.56\,\|\bar\kappa_3\|_1}{V^{3/2}},$$
where 0.56 is the Berry–Esseen constant of step 3 and $w$, the chapter's $\xi$, is the width of a band around the Gaussian curve $G_\mu(\alpha) = \Phi\big(\Phi^{-1}(1 - \alpha) - \mu\big)$, with $\Phi$ the standard normal distribution function. If $w < \frac12$, then for all $\alpha \in [w, 1 - w]$,
$$G_\mu(\alpha + w) - w \le f_1 \otimes \dots \otimes f_k(\alpha) \le G_\mu(\alpha - w) + w.$$ {#eq-apx-comp-band}
This is Dong, Roth & Su (2022), Theorem 3.4. In the limit form (their Theorem 3.5), take a fresh list of $k$ symmetric curves for each $k$, and suppose that as $k$ grows $\sum\mathrm{kl} \to K_\infty$, the largest mean tends to 0, the summed second moments tend to some $s^2 > 0$, and the summed third absolute moments tend to 0. Then the composed curve tends to $G_{2K_\infty/s}$. The chapter's proof matches the composed loss with a Gaussian pair and points here for the passage from the loss to the curve.

### The idea

Compose $k = 100$ randomized responses, each $\varepsilon_0$-DP with $\varepsilon_0 = 0.1$. Each step's loss is $\pm 0.1$, with mean $\varepsilon_0\tanh(\varepsilon_0/2) = 0.004996$ by step 8. So $K = 0.4996$, $V = 0.9975$ and $\mu = 2 \times 0.4996/\sqrt{0.9975} = 1.0004$. Measured in units of $\sqrt V$ from the mean under $P$, the loss under $Q$ is the same law shifted left by $\mu$, and @fig-apx-comp-clt-cdfs (right) shows its two step-shaped distribution functions next to $\Phi(x)$ and $\Phi(x + \mu)$. A best test's type I error is a value of the first and its power a value of the second, each within $w$ of the normal curve, which is what the band says.

### Derivation

Let $(P_i, Q_i)$ be pairs with curves $f_i$. By @sec-apx-comp-fdp, the composed curve $f_1 \otimes \dots \otimes f_k$ is the curve of the product pair $P = \prod P_i$ against $Q = \prod Q_i$, whose privacy loss is $L = \sum_i L_i$ with independent terms. Steps 1 and 2 restate what the chapter proves about the law of $L$. Steps 3 to 6 turn it into the band, and steps 7 and 8 give the limit forms.

1. **The functionals are moments of the loss.** By @lem-gdp-moments, $\mathrm{kl}(f_i) = \mathbb E_P[L_i]$, $\kappa_2(f_i) = \mathbb E_P[L_i^2]$ and $\bar\kappa_3(f_i) = \mathbb E_P|L_i - \mathbb E_PL_i|^3$, and a curve fixes the law of its loss under $P$. Means and variances of independent terms add, so $K$ and $V$ are the mean and variance of $L$ under $P$.
2. **Symmetry mirrors the law.** As in step 3 of the proof of @thm-gdp-clt, the swapped pair $(Q_i, P_i)$ has loss $-L_i$ and the same curve $f_i$, so by step 1, $-L_i$ under $Q_i$ has the law of $L_i$ under $P_i$. Adding the independent steps, $-L$ under $Q$ has the law of $L$ under $P$, with mean $K$, variance $V$ and the same third absolute central moments.
3. **Berry–Esseen under both laws.** The Berry–Esseen theorem bounds how far a standardized sum of independent terms can be from normal. Take such a sum $S$, write $m$ for its mean and $V$ for its variance, and write $\rho_3$ for the sum of the third absolute central moments of its terms. Then for every $x$ (Shevtsova 2010, with the constant 0.56)
$$\Big|\Pr\Big[\frac{S - m}{\sqrt V} \le x\Big] - \Phi(x)\Big| \le \frac{0.56\,\rho_3}{V^{3/2}}.$$
The same bound holds with $<$ in place of $\le$: as $x$ rises towards a point $x_0$ from below, $\Pr[\cdot \le x]$ rises to $\Pr[\cdot < x_0]$, while $\Phi(x)$ moves continuously to $\Phi(x_0)$, so the bound survives the limit. Under $P$, with $S = L$, the right side is exactly $w$. Under $Q$ the bound follows from step 2. For every $y$,
$$\Pr_Q\Big[\frac{L + K}{\sqrt V} < y\Big] = \Pr_P\Big[\frac{L - K}{\sqrt V} > -y\Big] = 1 - \Pr_P\Big[\frac{L - K}{\sqrt V} \le -y\Big],$$
which is within $w$ of $1 - \Phi(-y) = \Phi(y)$, by the symmetry of the normal density.
4. **The errors of a best test.** By the Neyman–Pearson lemma, a best test fires when $L < \tau$ and, if $L$ has an atom at $\tau$, fires there with some probability. So its type I error $\alpha$ lies between $\Pr_P[L < \tau]$ and $\Pr_P[L \le \tau]$. Write $x = (\tau - K)/\sqrt V$ for the threshold in standard units under $P$. Both ends are within $w$ of $\Phi(x)$ by step 3, so $|\alpha - \Phi(x)| \le w$. Its power lies between $\Pr_Q[L < \tau]$ and $\Pr_Q[L \le \tau]$. The event $L < \tau$ is the event $(L + K)/\sqrt V < x + 2K/\sqrt V$, so by step 3 the power is within $w$ of $\Phi(x + \mu)$ with $\mu = 2K/\sqrt V$. In terms of the type II error $\beta = 1 - \text{power}$, and using $1 - \Phi(t) = \Phi(-t)$,
$$|\alpha - \Phi(x)| \le w, \qquad |\beta - \Phi(-x - \mu)| \le w.$$
This is where $\mu$ comes from: under $P$ and under $Q$ the loss has the same spread, and its two means $K$ and $-K$ are $2K/\sqrt V$ standard deviations apart, as the means of the Gaussian pair behind $G_\mu$ are $\mu$ apart.
5. **Remove the threshold.** The Gaussian curve is $G_\mu(\alpha) = \Phi(-\Phi^{-1}(\alpha) - \mu)$, because $\Phi^{-1}(1 - \alpha) = -\Phi^{-1}(\alpha)$. From $\Phi(x) \ge \alpha - w$ and the fact that $\Phi$ is increasing, $x \ge \Phi^{-1}(\alpha - w)$. Then $\Phi(-x - \mu) \le \Phi(-\Phi^{-1}(\alpha - w) - \mu) = G_\mu(\alpha - w)$, so $\beta \le G_\mu(\alpha - w) + w$. In the same way, $\Phi(x) \le \alpha + w$ gives $x \le \Phi^{-1}(\alpha + w)$ and $\beta \ge G_\mu(\alpha + w) - w$. The first bound needs $\alpha - w \ge 0$ and the second needs $\alpha + w \le 1$, so that $\Phi^{-1}$ is applied inside $[0, 1]$, which is why the band holds on $[w, 1 - w]$.
6. **Every point of the curve.** For every type I error $\alpha$, the Neyman–Pearson lemma gives a best test of size exactly $\alpha$, and its type II error is the composed curve at $\alpha$. Steps 4 and 5 apply to that test, so @eq-apx-comp-band holds at every $\alpha \in [w, 1 - w]$.
7. **The limit form.** Take a fresh list of $k$ symmetric curves for each $k$, with the four conditions. First, $V \to s^2$. Each mean $\mathrm{kl}_i$ is a Kullback–Leibler divergence and so cannot be negative, so each squared mean is at most the largest mean times the mean. So $\sum_i\mathrm{kl}_i^2 \le \max_i\mathrm{kl}_i\sum_i\mathrm{kl}_i \to 0 \cdot K_\infty = 0$, and $V$ has the same limit as $\sum_i\kappa_2$.

   Second, $w \to 0$, because a third absolute central moment is at most 8 times the third absolute moment. To see it, measure size by $\|X\|_3 = (\mathbb E|X|^3)^{1/3}$, which obeys the triangle inequality. The mean $m_i$ of $L_i$ is at most $\|L_i\|_3$ in size, since an average is at most the average size, and the average size is at most $\|L_i\|_3$. So $\|L_i - m_i\|_3 \le \|L_i\|_3 + |m_i| \le 2\|L_i\|_3$, and cubing gives the factor 8. Hence $\sum\bar\kappa_3 \le 8\sum\kappa_3 \to 0$, so $w \to 0$ and $\mu \to 2K_\infty/s$.

   Third, the band closes. Since $\mu$ converges, it stays below some bound $\mu_{\max}$. The function $(\mu, \alpha) \mapsto G_\mu(\alpha)$ is uniformly continuous on $[0, \mu_{\max}] \times [0, 1]$, so both edges of the band tend to $G_{2K_\infty/s}$ uniformly in $\alpha$. Below $\alpha = w$, the composed curve and $G_\mu$ are both non-increasing and at most 1, so each lies between its value at $w$ and 1. Both values at $w$ tend to 1 as $w \to 0$, so the curve converges there too, and the same argument with 0 in place of 1 covers $\alpha > 1 - w$.
8. **Steps that are $\varepsilon_i$-DP.** Such a step is $f_{\varepsilon_i, 0}$-DP, and $f_{\varepsilon, 0}$ is the curve of randomized response. Its loss is $+\varepsilon$ with probability $t = e^\varepsilon/(1 + e^\varepsilon)$ and $-\varepsilon$ otherwise. So $\mathrm{kl} = \varepsilon(2t - 1)$, and $2t - 1 = (e^\varepsilon - 1)/(e^\varepsilon + 1) = \tanh(\varepsilon/2)$. The loss always has size $\varepsilon$, so $\kappa_2 = \varepsilon^2$ and $\kappa_3 = \varepsilon^3$. For $x \ge 0$, $x - x^3/3 \le \tanh x \le x$, and with $x = \varepsilon/2$ this gives
$$\Big|\varepsilon\tanh\frac\varepsilon2 - \frac{\varepsilon^2}{2}\Big| \le \frac{\varepsilon^4}{24}.$$
If $\sum\varepsilon_i^2 \to \mu^2$ and $\max\varepsilon_i \to 0$, then $\sum\mathrm{kl} \to \mu^2/2$, because the error is at most $\max\varepsilon_i^2\sum\varepsilon_i^2/24 \to 0$. Also $\sum\kappa_2 \to \mu^2$ and $\sum\kappa_3 \le \max\varepsilon_i\sum\varepsilon_i^2 \to 0$. Step 7, with $K_\infty = \mu^2/2$ and $s = \mu$, gives the limit $G_{2(\mu^2/2)/\mu} = G_\mu$. By @sec-apx-comp-fdp the composed mechanism is $\bigotimes f_{\varepsilon_i, 0}$-DP, so its guarantee tends to $\mu$-GDP.

### Checking it

The script composes $k$ randomized responses with $\varepsilon_0 = 1/\sqrt k$, exactly, from the binomial law of the number of truthful reports. @tbl-apx-comp-clt compares the composed curve with the band. Its "room" columns give the smallest distance, over $\alpha \in [w, 1 - w]$, between the curve and each edge of the band.

| $k$ | $\mu$ | $w$ | Smallest room above the lower edge | Smallest room below the upper edge | $\max_\alpha\lvert f(\alpha) - G_\mu(\alpha)\rvert$ | Actual $\sup_x\lvert F(x) - \Phi(x)\rvert$ |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 1.0042 | 0.1837 | 0.2151 | 0.2420 | 0.0137 | 0.1331 |
| 100 | 1.0004 | 0.0562 | 0.0611 | 0.0647 | 0.0014 | 0.0405 |
| 1000 | 1.0000 | 0.0177 | 0.0187 | 0.0193 | 0.0001 | 0.0127 |

: The band of @eq-apx-comp-band for $k$ randomized responses with $\varepsilon_0 = 1/\sqrt k$. The last column is the distance between the distribution function $F$ of $(L - K)/\sqrt V$ under $P$ and $\Phi$, the quantity that Berry–Esseen bounds by $w$. {#tbl-apx-comp-clt}

The exact curve always stays inside the band, with room to spare (@fig-apx-comp-clt-band). The Berry–Esseen step is nearly sharp here: the last column is about 0.72 times $w$ at every $k$, because the distribution function of the loss jumps at each of its $k + 1$ values. Yet the curve is much closer to $G_\mu$, because a jump moves both errors of a test at once, right and down, roughly along the falling curve $G_\mu$ rather than away from it. So the band is valid but wide, and its width shrinks only like $1/\sqrt k$. At $k = 1$, $w = 0.7663$ is above $\frac12$ and the theorem says nothing.

![The band of the $f$-DP central limit theorem for $k$ randomized responses with $\varepsilon_0 = k^{-1/2}$. Grey: the region between $G_\mu(\alpha + w) - w$ and $G_\mu(\alpha - w) + w$; blue: $G_\mu$; black: the exact composed curve. Dotted lines mark $\alpha = w$ and $\alpha = 1 - w$, where the band starts to apply. Left: $k = 10$, $w = 0.1837$. Right: $k = 100$, $w = 0.0562$.](figures/apx_comp_clt_band.svg){#fig-apx-comp-clt-band}

![Distribution functions of the standardized composed loss $(L - K)/\sqrt V$ of $k$ randomized responses, under $P$ (blue steps) and under $Q$ (red steps), against $\Phi(x)$ and $\Phi(x + \mu)$ (dashed). The law under $Q$ is the mirror image of the law under $P$, shifted left by $\mu = 2K/\sqrt V$. Left: $k = 10$. Right: $k = 100$.](figures/apx_comp_clt_cdfs.svg){#fig-apx-comp-clt-cdfs}

### What it does not cover

The proof needs every $f_i$ to be symmetric, because step 2 gets the law under $Q$ from the law under $P$. Without symmetry the two laws have to be handled separately, as in @sec-apx-comp-sgd-clt. The Berry–Esseen theorem and its constant 0.56 are cited (Shevtsova 2010), and the central limit theorem itself, without a rate, is derived in @sec-apx-prob-clt. The band is additive, so it says nothing useful about type I errors below $w$, which are the errors that decide a small $\delta$. The $\delta$ that goes with $\varepsilon$ is the largest value of $1 - f(\alpha) - e^\varepsilon \alpha$ over $\alpha$, and for a large $\varepsilon$ the cost $e^\varepsilon \alpha$ pushes that largest value to a tiny $\alpha$. This is why the chapter treats the CLT only as an approximation in the tails.

## The central limit theorem for the subsampled Gaussian {#sec-apx-comp-sgd-clt}

### What the chapter uses

Each step of DP-SGD with Poisson subsampling includes each record independently with probability $q$, the sampling rate, clips each record's gradient, and adds Gaussian noise with noise multiplier $\sigma$. Under add/remove neighbours, where $D'$ is $D$ with one target record added, one step reduces to the pair (@sec-gdp-sgd)
$$P = \mathcal N(0, \sigma^2), \qquad Q = (1 - q)\,\mathcal N(0, \sigma^2) + q\,\mathcal N(1, \sigma^2).$$
This order is the **add direction**, and the reversed order $(Q, P)$ is the **remove direction**. Since $q$ names the sampling rate, this section writes the densities as $P(y)$ and $Q(y)$. Write $v = e^{1/\sigma^2} - 1$ and $P^k$, $Q^k$ for the $k$-fold products. If $k \to \infty$ and $q\sqrt k \to c > 0$ with $\sigma$ fixed, then
$$T(P^k, Q^k) \to G_{c\sqrt v} \quad\text{uniformly on } [0, 1],$$
which is @thm-gdp-sgd, $\mu = q\sqrt{kv}$ (Bu, Dong, Long & Su 2020, Theorem 5). The pair is not symmetric, so @eq-apx-comp-band does not apply as stated, and this section treats $P$ and $Q$ separately and bounds the remainders of the chapter's expansion by $O(q^3)$. @sec-num-dpsgd and @sec-choose-compare use the formula as an approximation.

### The idea

The skeleton is that of @sec-apx-comp-clt, with the law under $Q$ computed on its own instead of as a mirror image. To leading order the two laws are mirror images anyway. At $q = 0.01$ and $\sigma = 1$, $v = e - 1 = 1.7183$, and numerical integration gives one step's loss mean $8.1890\times10^{-5}$ and variance $1.6009\times10^{-4}$ under $P$, and $-8.3812\times10^{-5}$ and $1.7163\times10^{-4}$ under $Q$. The leading terms are $\pm\frac{q^2}{2}v = \pm8.5914\times10^{-5}$ and $q^2v = 1.7183\times10^{-4}$, and the errors, of order $q^3$, add up over $k$ steps to order $kq^3 \to 0$.

### Derivation

The plan is to bound the mean, variance and third moment of one step's loss under each law with remainders of order $q^3$ (steps 1 to 6), add them over $k$ steps (step 7), and then follow the best tests as in @sec-apx-comp-clt, standardizing each law with its own mean and variance (steps 8 and 9).

1. **The loss.** Steps 1 and 2 of the proof of @thm-gdp-sgd give the likelihood ratio $Q(y)/P(y) = 1 + qW$ with $W = e^{(y - 1/2)/\sigma^2} - 1$, so the privacy loss is
$$L = \ln\frac{P(y)}{Q(y)} = -\ln(1 + qW).$$
Since $W > -1$, the number $x = qW$ is above $-q$, so $x \ge -\frac12$ whenever $q \le \frac12$, which step 3 needs. From here on, $q \le \frac12$.
2. **All moments of $W$ are finite.** Under $P$, $y \sim \mathcal N(0, \sigma^2)$, and $(1 + W)^n = e^{ny/\sigma^2}e^{-n/(2\sigma^2)}$. The moment generating function $\mathbb E[e^{\lambda y}] = e^{\lambda^2\sigma^2/2}$ at $\lambda = n/\sigma^2$ gives, for any whole number $n$,
$$\mathbb E_P\big[(1 + W)^n\big] = e^{n(n - 1)/(2\sigma^2)}.$$
The cases $n = 1$ and $n = 2$ give $\mathbb E_P[W] = 0$ and $\mathbb E_P[W^2] = v$, as in the chapter. The absolute moments are bounded too, because $|W|^n \le (1 + W)^n + 1$ when $W \ge -1$, since a negative $W$ has $|W| \le 1$. So $\mathbb E_P|W|^n \le e^{n(n - 1)/(2\sigma^2)} + 1$, which at $\sigma = 1$ is 21.09 for the third moment and 404.43 for the fourth. Averages under $Q$ are averages under $P$ weighted by the likelihood ratio, $\mathbb E_Q[h(W)] = \mathbb E_P[(1 + qW)h(W)]$ for any function $h$, so they are finite as well.
3. **The logarithm with a bounded remainder.** For $x \ge -\frac12$,
$$\ln(1 + x) = x - \frac{x^2}{2} + R(x), \qquad |R(x)| \le \frac23|x|^3.$$
The remainder $R$ starts at $R(0) = 0$, and its slope is $R'(x) = \frac{1}{1 + x} - 1 + x = \frac{x^2}{1 + x}$. For $x \ge 0$ that slope lies between 0 and $x^2$, so integrating from 0 gives $0 \le R(x) \le x^3/3$. For $-\frac12 \le x < 0$ the denominator $1 + x$ is at least $\frac12$, so the slope is at most $2x^2$, and integrating gives $|R(x)| \le \frac23|x|^3$. The logarithm itself has slope $1/(1 + x) \le 2$ on the same range, so
$$|\ln(1 + x)| \le 2|x| \quad\text{for } x \ge -\tfrac12.$$
4. **The means.** Under $P$, by steps 2 and 3, $L = -qW + \frac{q^2W^2}{2} - R(qW)$ has
$$\Big|\mathbb E_P[L] - \frac{q^2}{2}v\Big| \le \frac23q^3\,\mathbb E_P|W|^3.$$
Under $Q$, weighting by $1 + qW$ gives $\mathbb E_Q[L] = -\frac{q^2}{2}v + \frac{q^3}{2}\mathbb E_P[W^3] - \mathbb E_P[(1 + qW)R(qW)]$, because $\mathbb E_P[(1 + qW)(-qW + \frac{q^2W^2}{2})] = -q^2v + \frac{q^2}{2}v + \frac{q^3}{2}\mathbb E_P[W^3]$. The middle term is at most $\frac{q^3}{2}\mathbb E_P|W|^3$ in size. By the bound on $R$ and $|1 + qW| \le 1 + q|W|$, the last is at most $\frac23q^3\,\mathbb E_P|W|^3 + \frac23q^4\,\mathbb E_P[W^4]$. Adding the two gives
$$\Big|\mathbb E_Q[L] + \frac{q^2}{2}v\Big| \le q^3\Big(\frac76\,\mathbb E_P|W|^3 + \frac23\,q\,\mathbb E_P[W^4]\Big).$$
5. **The variances.** Write $L = -qW + r$, where $|r| \le \frac{q^2W^2}{2} + \frac23q^3|W|^3$. Then $\mathbb E_P[L^2] = q^2v - 2q\,\mathbb E_P[Wr] + \mathbb E_P[r^2]$. The middle term is at most $q^3\,\mathbb E_P|W|^3 + \frac43q^4\,\mathbb E_P[W^4]$ in size, and the last is of order $q^4$. The squared mean is of order $q^4$ by step 4. So $\operatorname{Var}_P(L) = q^2v + O(q^3)$, where $O(q^3)$ means at most a constant times $q^3$, the constant depending only on $\sigma$ through the moments of step 2. Under $Q$, $\mathbb E_Q[L^2] = \mathbb E_P[L^2] + q\,\mathbb E_P[WL^2]$, and $|L| \le 2q|W|$ by step 3 makes the extra term at most $4q^3\,\mathbb E_P|W|^3$. So $\operatorname{Var}_Q(L) = q^2v + O(q^3)$ as well.
6. **The third moments.** By step 3, $|L| \le 2q|W|$, so $\mathbb E_P|L|^3 \le 8q^3\,\mathbb E_P|W|^3$, and under $Q$ the extra weight adds at most $8q^4\,\mathbb E_P[W^4]$. As in step 7 of @sec-apx-comp-clt, a third absolute central moment is at most 8 times the third absolute moment. So under both laws $\mathbb E|L - \mathbb E L|^3 = O(q^3)$.
7. **Add up $k$ steps.** Under $P$ the summed loss has mean $K_P = k\,\mathbb E_P[L]$ and variance $V_P = k\,\operatorname{Var}_P(L)$, and likewise $K_Q$ and $V_Q$ under $Q$. With $q\sqrt k \to c$, $kq^2 \to c^2$ and $kq^3 = kq^2 \cdot q \to 0$. So steps 4 and 5 give
$$K_P \to \frac{c^2v}{2}, \qquad K_Q \to -\frac{c^2v}{2}, \qquad V_P \to c^2v, \qquad V_Q \to c^2v.$$
By step 6 the summed third central moments are $O(kq^3)$, while $V^{3/2}$ tends to $(c^2v)^{3/2} > 0$. So the Berry–Esseen bounds $w_P$ and $w_Q$ under the two laws, each the summed third central moments over $V^{3/2}$, tend to 0.
8. **The errors of a best test.** As in step 4 of @sec-apx-comp-clt, a best test fires when the summed loss $S$ is below a threshold $\tau$, and at $\tau$ itself it fires with some probability. Write $x = (\tau - K_P)/\sqrt{V_P}$ for the threshold in standard units under $P$. Then step 7's Berry–Esseen bound under $P$ gives $|\alpha - \Phi(x)| \le w_P$, as before.

   Under $Q$ the loss must be standardized with its own mean and variance. Since $\tau = K_P + x\sqrt{V_P}$, the event $S < \tau$ is the event
$$\frac{S - K_Q}{\sqrt{V_Q}} < \omega_kx + \mu_k, \qquad \omega_k = \sqrt{\frac{V_P}{V_Q}}, \qquad \mu_k = \frac{K_P - K_Q}{\sqrt{V_Q}}.$$
By step 7, the scale ratio $\omega_k$ tends to 1 and the shift $\mu_k$ tends to $c^2v/(c\sqrt v) = c\sqrt v = \mu$. So the Berry–Esseen bound under $Q$ puts the power within $w_Q$ of $\Phi(\omega_kx + \mu_k)$.

   It remains to show that $\Phi(\omega_kx + \mu_k)$ differs from $\Phi(x + \mu)$ by at most some $\xi_k$ that tends to 0 and does not depend on $x$. Fix a large number $A$ and split the values of $x$ in two. For $|x| \le A$ the difference is at most $(|\omega_k - 1|A + |\mu_k - \mu|)/\sqrt{2\pi}$, because the normal density is at most $1/\sqrt{2\pi}$. For $|x| > A$, take $k$ large enough that $\omega_k \ge \frac12$ and $0 \le \mu_k \le \mu + 1$. Then both arguments are at least $A/2$ when $x > A$ and at most $\mu + 1 - A/2$ when $x < -A$, so both values lie within $\Phi(\mu + 1 - A/2)$ of the same end, 1 or 0. Choosing $A$ large first makes the second part small, and then choosing $k$ large makes the first part small. So $|\beta - \Phi(-x - \mu)| \le w_Q + \xi_k$.
9. **Conclude.** Steps 5 and 6 of @sec-apx-comp-clt, with $w = \max(w_P, w_Q + \xi_k)$, put the curve of $P^k$ against $Q^k$ in a band around $G_\mu$ whose width tends to 0, and step 7 there turns this into uniform convergence. The remove direction swaps the roles of $P$ and $Q$. Its loss is $-L$, and its means are $-K_Q \to \frac{c^2v}{2}$ under $Q$ and $-K_P \to -\frac{c^2v}{2}$ under $P$, so the same argument gives the same limit. Finally, DP-SGD chooses each step after seeing the model so far. But clipping bounds each step's effect, so each step's curve is at least the curve of this pair (@sec-gdp-sgd), and by @sec-apx-comp-fdp the whole run is $T(P, Q)^{\otimes k}$-DP, a curve that tends to $G_\mu$.

### Checking it

Step 4 says that $(\mathbb E_P[L] - \frac{q^2}{2}v)/q^3$ stays bounded, by $\frac23\,\mathbb E_P|W|^3 \le 14.06$ at $\sigma = 1$. By numerical integration it is $-3.619$, $-4.024$, $-4.292$ and $-4.453$ at $q = 0.02$, 0.01, 0.005 and 0.0025, so it settles near a constant, and the remainder really is of order $q^3$. The Berry–Esseen bound under $P$ is 0.3045, 0.0963 and 0.0304 at $k = 100$, 1000 and 10,000, shrinking like $1/\sqrt k$. The actual distance at $k = 100$, computed from a numerical PLD, is between 0.0312 and 0.0343, so the bound is about nine times too large. @fig-apx-comp-sgd-cdfs shows the standardized laws, shifted by $\mu = 0.1311$, 0.4145 and 1.3108.

![Distribution functions of the standardized summed loss $(S - K_P)/\sqrt{V_P}$ of $k$ steps of the Poisson-subsampled Gaussian with $q = 0.01$ and $\sigma = 1$, under $P$ (blue) and $Q$ (red), computed from a numerical PLD, against $\Phi(x)$ and $\Phi(x + \mu)$ with $\mu = q\sqrt{k(e^{1/\sigma^2} - 1)}$. From left to right, $k = 100$, 1,000 and 10,000.](figures/apx_comp_sgd_cdfs.svg){#fig-apx-comp-sgd-cdfs}

The theorem controls the bulk, not the tail, and @fig-apx-comp-sgd-tails shows what that costs at $\delta = 10^{-5}$. It plots $\delta(\varepsilon)$ for each direction, from a numerical PLD that rounds each step's loss to a grid of spacing $h = 10^{-5}$, against the Gaussian profile with the CLT's $\mu$. @tbl-apx-comp-sgd-eps gives $\varepsilon$ at $\delta = 10^{-5}$ as brackets. Rounding every loss up gives the upper end, which is a valid bound, and rounding down gives the lower end, so the two ends differ by at most $kh$.

| $k$ | CLT $\varepsilon$ | Add direction | Remove direction |
| --- | --- | --- | --- |
| 100 | 0.4575 | [0.3377, 0.3387] | [0.7175, 0.7185] |
| 1,000 | 1.6177 | [1.4117, 1.4217] | [1.8232, 1.8332] |
| 10,000 | 6.0071 | [5.5615, 5.6615] | [6.1377, 6.2377] |

: $\varepsilon$ at $\delta = 10^{-5}$ for the running DP-SGD example, from the CLT and from a numerical PLD in each direction of neighbours. {#tbl-apx-comp-sgd-eps}

The two directions err in opposite ways. In the remove direction the loss is $\ln(1 + qW)$ under the mixture, and the rare outputs where the record was in the batch and the noise pushed $y$ high give it a long upper tail, so $\delta$ lies above the Gaussian's. In the add direction the loss $-\ln(1 + qW)$ can never exceed $-\ln(1 - q)$ per step, so its upper tail is shorter than a Gaussian's. An accountant must take the worse direction, so the CLT under-reports $\varepsilon$ at every $k$ shown, by 0.13 even at $k = 10{,}000$.

![The privacy profile $\delta(\varepsilon)$ on a log scale for $k$ steps of the Poisson-subsampled Gaussian with $q = 0.01$, $\sigma = 1$: the remove direction (red) and the add direction (blue) from a numerical PLD, rounded up, and the Gaussian profile of the CLT's $\mu$ (dashed). The dotted line is $\delta = 10^{-5}$. Left: $k = 1{,}000$. Right: $k = 10{,}000$.](figures/apx_comp_sgd_tails.svg){#fig-apx-comp-sgd-tails}

### What it does not cover

The theorem is a limit with no rate in the tail. Its band is at most $\max(w_P, w_Q + \xi_k)$ wide, and both parts stay far wider than $10^{-5}$, so the theorem does not certify any $(\varepsilon, \delta)$ at a small $\delta$, which is the chapter's warning. Batches of fixed size under replace-one neighbours, where $D'$ swaps one record of $D$ for another, lead to a different formula for $\mu$ (Dong et al. 2022, Theorem 5.2 and Corollary 5.4), and that result stays cited.

## Code

The script [code/appendix-derivations/apx_composition.py](../../code/appendix-derivations/apx_composition.py) checks every derivation numerically. It builds the coarsened canonical pairs, computes exact trade-off curves of finite pairs by sorting likelihood ratios, and runs the test replacement on random tests. It computes hockey-stick divergences of the adaptive runs and their dominating products, the exact composed curves of randomized response against the Berry–Esseen band, and the moments and numerical PLDs of the subsampled Gaussian. It uses the standard library and NumPy only and runs in under a minute. From the repository root:

```bash
uv run python code/appendix-derivations/apx_composition.py
```

**Output** (seed 0):

| Check | Result |
| --- | --- |
| Canonical pair, 1000 cells: largest gap to $f$ | $G_1$: $2.0\times10^{-3}$; $f_{1, 0.1}$: $8.3\times10^{-5}$ |
| Adaptive randomized response against $f \otimes f$ | never below; largest gap 0.2812 at $\alpha = 0.0625$ |
| Test replacement on 20,000 random tests | type I error unchanged (to $3.3\times10^{-16}$), type II never larger, mean drop 0.4013 |
| Three-step run against the product, 502 values of $\gamma$ | never above; $\delta(1)$ 0.3926 against 0.4190 |
| 100,000 random tests against the product's upper edge | none above; smallest margin 0.0726 |
| CLT band, $k = 100$ randomized responses | $\mu = 1.0004$, $w = 0.0562$; curve inside, at most 0.0014 from $G_\mu$ |
| Subsampled Gaussian, mean remainder over $q^3$ at $q = 0.0025$ | $-4.453$ under $P$, bound 14.06 |
| DP-SGD, $k = 10{,}000$: $\varepsilon$ at $10^{-5}$ | CLT 6.0071; remove direction $[6.1377, 6.2377]$ |

: Results printed by the script with seed 0. {#tbl-apx-comp-code}

[tests/test_appendix_composition.py](../../tests/test_appendix_composition.py) checks that the output still matches @tbl-apx-comp-code and every other number on this page.

**What to take from it.** The replacement step never fails: on every random test the type I error is kept and the type II error does not grow, so composition is never worse than the product of the fixed pairs that bound each step. The adaptive examples show how loose that can be when the branches differ. The Berry–Esseen band always holds and is nearly sharp for the distribution functions, but the curve itself is far closer to $G_\mu$. The subsampled-Gaussian expansions have remainders of order $q^3$, as proved. The CLT still under-reports $\varepsilon$ at $\delta = 10^{-5}$, because the tail of the loss is longer than a Gaussian's in the direction that matters.

## References

- Dong, Roth & Su, "Gaussian Differential Privacy" (Journal of the Royal Statistical Society Series B, 2022; arXiv:1905.02383). The source of $f$-DP, the tensor product (Definition 3.1), adaptive composition (Theorem 3.2, Appendix C) and the CLTs (Theorems 3.4–3.6, Appendix D). Read it for the original proofs, which this appendix follows in spirit.
- Zhu, Dong & Wang, "Optimal Accounting of Differential Privacy via Characteristic Function" (AISTATS 2022). Dominating pairs (Definition 7) and their adaptive composition (Theorem 10), including pairs that may depend on the history.
- Bu, Dong, Long & Su, "Deep Learning with Gaussian Differential Privacy" (Harvard Data Science Review, 2020). The CLT for the Poisson-subsampled Gaussian (Theorem 5), proved here in @sec-apx-comp-sgd-clt.
- Shevtsova, "An improvement of convergence rate estimates in the Lyapunov theorem" (Doklady Mathematics, 2010). The constant 0.56 in the Berry–Esseen bound for sums of independent, non-identically distributed terms.
- Feller, *An Introduction to Probability Theory and Its Applications*, Volume II (1971), chapter 16, section 5. The Berry–Esseen theorem and Lyapunov's central limit theorem for triangular arrays, with proofs.
- Rockafellar, *Convex Analysis* (1970), §§23–24. One-sided derivatives and supporting lines of convex and concave functions, used in @sec-apx-comp-canonical and in step 3 of @sec-apx-comp-dominating.
