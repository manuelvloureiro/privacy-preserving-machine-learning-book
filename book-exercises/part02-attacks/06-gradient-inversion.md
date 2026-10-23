# Exercises: Gradient inversion

Exercises for [Gradient inversion](../../book-full/part02-attacks/06-gradient-inversion.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part02-attacks/solutions_gradient_inversion.py` computes every number quoted here; run it with `uv run python code/part02-attacks/solutions_gradient_inversion.py`.

The solutions use the chapter's notation: the network $a = W_1x + b_1$, $h = \sigma(a)$, $z = W_2h + b_2$, $p = \mathrm{softmax}(z)$, $\ell = -\ln p_y$, with backward signals $\gamma_2 = p - e_y$ and $\gamma_1 = (W_2^\top\gamma_2)\odot\sigma'(a)$, and the three Toolkit facts: the outer-product form of a fully connected layer's gradient (Fact 1), the softmax gradient $p - e_y$ (Fact 2), and backpropagation and averaging (Fact 3). The numerical checks use the chapter's $8\times8$ images of four shapes and random networks with 32 hidden units.

## Logistic regression leaks its input {#sec-sol-gradinv-1}

> **Exercise 1.** **Logistic regression leaks its input.** For logistic regression with a bias, loss $\ln(1 + e^{-y(\theta^\top x + b)})$ with $y \in \{-1, +1\}$, show that $\nabla_\theta\ell = (\nabla_b\ell)\,x$, so that $x = \nabla_\theta\ell/\nabla_b\ell$ whenever $\nabla_b\ell \ne 0$. When is $\nabla_b\ell = 0$? *Check: with $\theta = (1, -1)$, $b = 0$, $x = (2, 1)$ and $y = +1$, $\nabla_b\ell = -0.2689$, $\nabla_\theta\ell = (-0.5379, -0.2689)$, and the ratio is $(2, 1)$.* *(Chain rule, Fact 1)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One scalar carries the loss.** Write $a = \theta^\top x + b$, a fully connected layer with one output unit. The loss is $\ell = \ln(1 + e^{-ya})$.
2. **Its backward signal.** $\gamma = \frac{\partial\ell}{\partial a} = \frac{-ye^{-ya}}{1 + e^{-ya}} = -y\,\sigma(-ya)$, where $\sigma$ is the sigmoid.
3. **Fact 1 with $m = 1$.** $\nabla_\theta\ell = \gamma x$ and $\nabla_b\ell = \gamma$. So $\nabla_\theta\ell = (\nabla_b\ell)x$, and dividing by $\nabla_b\ell$ gives $x$.
4. **When the ratio fails.** $\gamma = -y\sigma(-ya)$ is never exactly zero, because the sigmoid is strictly positive. It is tiny when the example is classified correctly with a large margin ($ya \gg 0$), and then in floating point it can underflow to 0. So the attack is exact in exact arithmetic for every example, and fails in practice only for examples the model already fits very confidently.
5. **Check.** $a = 1\cdot2 - 1\cdot1 + 0 = 1$ and $y = +1$, so $\gamma = -\sigma(-1) = -1/(1 + e) = -0.2689$. Then $\nabla_\theta\ell = -0.2689\cdot(2, 1) = (-0.5379, -0.2689)$, and the ratio is $(-0.5379, -0.2689)/(-0.2689) = (2, 1)$.

*A linear model with a bias is the one-unit case of @eq-gradinv-ratio: its gradient is the input, scaled by how wrong the model is.*
:::

## No bias in the first layer {#sec-sol-gradinv-2}

> **Exercise 2.** **No bias in the first layer.** Remove $b_1$ from the network. Show that $\nabla_{W_1}\ell$ still has rank one and that its top right singular vector $u$ is $\pm x/\| x\|$. The attacker then knows $x = cu$ for an unknown scalar $c$; describe a one-dimensional search for $c$, and explain why the positive homogeneity of ReLU ($\max(ca, 0) = c\max(a, 0)$ for $c > 0$) does not make $c$ unidentifiable. *Check: on an image, the script finds $|\cos(u, x)| = 1.000000$, and the search recovers $x$ to a relative error of $4.0\times10^{-16}$.* *(Rank one, Fact 1)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Rank one.** Fact 1 does not need a bias: $\nabla_{W_1}\ell = \gamma_1x^\top$ still. If $\gamma_1 \ne 0$, every row is a multiple of $x^\top$, so the rank is one.
2. **The singular vector.** A rank-one matrix $\gamma_1x^\top$ has the singular value decomposition $\|\gamma_1\|\| x\|\cdot\frac{\gamma_1}{\|\gamma_1\|}\big(\frac{x}{\| x\|}\big)^\top$, so its only right singular vector with a nonzero singular value is $u = \pm x/\| x\|$ (the sign is arbitrary in any decomposition). Hence $x = cu$ with $c = \pm\| x\|$.
3. **The search.** For each candidate $c$, compute the gradient that the input $cu$ would produce, $G(cu)$, and its mismatch $R(c) = \| G(cu) - g\|^2$. The true $c$ gives $R = 0$. The script evaluates $R$ on a grid of 6,001 values of $c$ in $[-30, 30]$ and refines the best one by golden-section search.
4. **Why the scale is identifiable.** Positive homogeneity says $h(cu) = c\,h(u)$ for $c > 0$ when $b_1 = 0$. It does not make the gradient independent of $c$. The logits become $z = cW_2h(u) + b_2$, so the probabilities $p$, and with them $\gamma_2 = p - e_y$ and $\gamma_1$, change with $c$; and $\nabla_{W_1}\ell = \gamma_1(cu)^\top$ carries an explicit factor $c$ as well. So different $c$ give different gradients, and only the true one matches. A negative $c$ switches on a different set of units, so it gives a different gradient too: the sign is identifiable.
5. **Check.** On an image, $|\cos(u, x)| = 1.000000$. The search returns $c = -3.3719$; the singular vector happened to come out as $-x/\| x\|$, and $\| x\| = 3.3719$. The reconstruction $cu$ has relative error $4.0\times10^{-16}$. At $-c$ the mismatch is $R = 10.8$, far from 0.

*Removing the bias removes the one-line division, not the leak: the gradient still contains the input's direction exactly, and one scalar is easy to search for.*
:::

## Labels with tanh {#sec-sol-gradinv-3}

> **Exercise 3.** **Labels with tanh.** With $h = \tanh(a)$ the hidden activations can be negative. Show that the rule "the label is the row of $\nabla_{W_2}\ell$ with the most negative sum" can fail, and that the rule "the label is the row whose sum has the opposite sign to all the others" is correct whenever $K \ge 3$ and $\sum_jh_j \ne 0$. *Check: on 1,000 random images and networks with $K = 4$, the first rule is right 0.488 of the time and the second 1.000.* *(Fact 2)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Row sums.** By Fact 1 at the last layer, row $k$ of $\nabla_{W_2}\ell$ is $\gamma_{2,k}h^\top$, so its sum is $\gamma_{2,k}S$ with $S = \sum_jh_j$.
2. **Signs of $\gamma_2$.** By Fact 2, $\gamma_{2,y} = p_y - 1 < 0$ and $\gamma_{2,k} = p_k > 0$ for $k \ne y$.
3. **The first rule fails when $S < 0$.** Then row $y$ has sum $(p_y - 1)S > 0$ and every other row has sum $p_kS < 0$. The most negative sum belongs to the class with the largest $p_k$ among the wrong ones, never to $y$. With tanh and random weights, $S$ is negative about half the time, so the rule is right about half the time.
4. **The second rule.** Whatever the sign of $S \ne 0$, row $y$'s sum has the sign of $-S$ and all $K - 1$ others have the sign of $S$. With $K \ge 3$ there are at least two others, so the majority sign is that of $S$ and the odd one out is $y$. With $K = 2$ there is one row of each sign, and without knowing the sign of $S$ the attacker cannot tell which is which.
5. **Check.** On 1,000 random images and networks with $K = 4$, the most-negative rule is right 0.488 of the time, close to the $\frac12$ of step 3, and the odd-one-out rule 1.000 of the time.

*What leaks is not "negative means label" but "the label's row has the opposite sign": a property of $p - e_y$ that no activation function can hide.*
:::

## Label counts in a batch {#sec-sol-gradinv-4}

> **Exercise 4.** **Label counts in a batch.** Derive @eq-gradinv-counts, and show that the attacker's rounded estimate $\mathrm{round}\big(B(\bar p_k - (\nabla_{b_2}\ell)_k)\big)$ equals $n_k$ for every class if and only if $\big|\sum_i(p_{ik} - \bar p_k)\big| < \frac12$ for every $k$ (ignoring ties at exactly $\frac12$). *Check: for $B = 16$ over 200 batches, the largest count error has median 1.102 with $\bar p = 1/K$ and 0.233 with $\bar p$ averaged over 200 of the attacker's images.* *(Fact 3)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Average the bias gradients.** By Fact 3 and Fact 2, $\nabla_{b_2}\ell = \frac1B\sum_i(p_i - e_{y_i})$.
2. **Count.** The $k$-th entry of $\sum_ie_{y_i}$ is the number of $i$ with $y_i = k$, that is $n_k$. So $B(\nabla_{b_2}\ell)_k = \sum_ip_{ik} - n_k$, which rearranges to @eq-gradinv-counts.
3. **The attacker's estimate.** Replacing $\sum_ip_{ik}$ by $B\bar p_k$ gives $\hat n_k = B\bar p_k - B(\nabla_{b_2}\ell)_k = n_k - \sum_i(p_{ik} - \bar p_k)$. Its error is $e_k = -\sum_i(p_{ik} - \bar p_k)$.
4. **Rounding.** $n_k$ is an integer, so $\mathrm{round}(n_k + e_k) = n_k$ exactly when $|e_k| < \frac12$ (at $|e_k| = \frac12$ it depends on the tie rule). All counts are right exactly when this holds for every $k$.
5. **Why it degrades with $B$.** $e_k$ is a sum of $B$ deviations of individual predictions from the guess. If the guess is the right mean and the deviations are roughly independent with standard deviation $s$, the sum has standard deviation about $s\sqrt B$ and eventually exceeds $\frac12$. A wrong guess adds a bias $B(\mathbb E p_k - \bar p_k)$ that grows linearly in $B$, which is why the uniform guess fails much sooner.
6. **Check.** Over 200 batches of 16, the median of $\max_k|e_k|$ is 1.102 with $\bar p = 1/K$ and 0.233 with $\bar p$ averaged over 200 of the attacker's own images. All errors are below $\frac12$ in 0.105 and 0.950 of the batches. (These are fresh random batches, so the rates differ slightly from the 0.095 and 0.975 of the chapter.)

*The bias gradient of a batch is its label histogram, minus a term the attacker can predict by running the model on data of its own.*
:::

## Isolation when inputs are correlated {#sec-sol-gradinv-5}

> **Exercise 5.** **Isolation when inputs are correlated.** Suppose each unit fires on a fraction $\rho$ of the population, with $\rho$ uniform on $[0, 1]$ across units, and given $\rho$ the examples of a batch switch the unit on independently with probability $\rho$. Show that a unit isolates a given example with probability $\int_0^1\rho(1 - \rho)^{B-1}\,d\rho = \frac{1}{B(B + 1)}$, and compare with $2^{-B}$. How many units does an even chance of isolating a given example take at $B = 10$? *Check: 76 units, against 710 under independence.* *Hint: the integral is a Beta function, $\int_0^1\rho^{a-1}(1-\rho)^{b-1}d\rho = \frac{(a-1)!(b-1)!}{(a+b-1)!}$.* *(Isolation, Fact 3)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Condition on $\rho$.** Given $\rho$, the unit is on for example $i$ with probability $\rho$ and off for each of the other $B - 1$ with probability $1 - \rho$, independently: $\rho(1 - \rho)^{B-1}$.
2. **Average over $\rho$.** $\rho$ is uniform on $[0, 1]$, so the unconditional probability is $\int_0^1\rho(1 - \rho)^{B-1}d\rho$.
3. **Evaluate.** This is the Beta integral with $a = 2$, $b = B$: $\frac{1!\,(B - 1)!}{(B + 1)!} = \frac{1}{B(B + 1)}$.
4. **Compare.** $2^{-B}$ falls exponentially and $1/(B(B + 1))$ only like $1/B^2$. At $B = 4$ they are close ($6.25\times10^{-2}$ against $5.00\times10^{-2}$); at $B = 10$ the second is 9 times larger ($9.09\times10^{-3}$ against $9.77\times10^{-4}$); at $B = 16$ it is 240 times larger ($3.68\times10^{-3}$ against $1.53\times10^{-5}$). The reason: a unit with small $\rho$ fires on almost nothing, and when it does fire on a batch it is likely to fire on one example only.
5. **Width for an even chance.** With $m$ units, each isolating example $i$ independently with probability $r$, the chance that some unit isolates it is $1 - (1 - r)^m$. Setting it to $\frac12$ gives $m = \ln2/(-\ln(1 - r))$, rounded up. For $r = 2^{-10}$ that is 710; for $r = 1/110$ it is 76.
6. **Check against the images.** On the images, with 1,024-unit networks, the measured fraction of (unit, example) pairs where the unit fires on that example only is $5.27\times10^{-2}$ at $B = 4$, $8.09\times10^{-3}$ at $B = 10$ and $2.52\times10^{-3}$ at $B = 16$. These are close to the uniform-fraction model and orders of magnitude above $2^{-B}$ at the larger batch sizes. This is why the open squares of @fig-gradinv-isolation decay so slowly.

*Independence is the best case for the client. Real data make units fire on many or on few examples, and the "few" case isolates examples polynomially rather than exponentially often.*
:::

## Several local steps {#sec-sol-gradinv-6}

> **Exercise 6.** **Several local steps.** A client holding one example runs $T$ steps of gradient descent with learning rate $\eta$ on it, starting from $\theta_0$, and sends $\theta_0 - \theta_T$. Show that the first-layer part of the update is still rank one and that $x = (U_1)_{j,:}/(v_1)_j$ for every unit with $(v_1)_j \ne 0$, where $U_1$ and $v_1$ are the changes in $W_1$ and $b_1$. What changes with a batch of two? *Check: with $T = 5$ and $\eta = 0.5$, the ratio recovers $x$ to a relative error of $1.4\times10^{-16}$.* *(Fact 1, Fact 3)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One step.** At step $t$, with parameters $\theta_t$, Fact 1 gives $\nabla_{W_1}\ell(\theta_t) = \gamma_1^{(t)}x^\top$ and $\nabla_{b_1}\ell(\theta_t) = \gamma_1^{(t)}$, where $\gamma_1^{(t)}$ is the backward signal under $\theta_t$. The input $x$ is the same at every step; only $\gamma_1^{(t)}$ changes.
2. **Sum the steps.** $U_1 = W_1^{(0)} - W_1^{(T)} = \eta\sum_{t=0}^{T-1}\gamma_1^{(t)}x^\top = \eta\,\bar\gamma_1\,x^\top$ with $\bar\gamma_1 = \sum_t\gamma_1^{(t)}$, the first-layer backward signal summed over the steps, and likewise $v_1 = b_1^{(0)} - b_1^{(T)} = \eta\,\bar\gamma_1$.
3. **Rank one and the ratio.** $U_1$ is an outer product, so it has rank one, and for $\bar\gamma_{1,j} \ne 0$ the ratio of row $j$ to $(v_1)_j$ is $x^\top$. Multiple local steps change $\bar\gamma_1$, not the factor $x$.
4. **A batch of two.** Write $\gamma_1^{(t)}(x_i)$ for the first-layer backward signal of example $i$ at step $t$. Each step's gradient is $\frac12\big(\gamma_1^{(t)}(x_1)x_1^\top + \gamma_1^{(t)}(x_2)x_2^\top\big)$, and the sum over steps is $u_1x_1^\top + u_2x_2^\top$ (up to the factors $\frac\eta2$), where $u_i = \sum_t\gamma_1^{(t)}(x_i)$ is example $i$'s signal summed over the steps. Unit $j$ still isolates example 1 if $u_{2,j} = 0$, which for ReLU requires the unit to stay off on $x_2$ at every step. As the weights move, a unit can switch on or off part-way, so isolation over $T$ steps is a little rarer than for one step.
5. **Check.** With $T = 5$ and $\eta = 0.5$ on one image, the ratio recovers $x$ to a relative error of $1.4\times10^{-16}$, and the update's second singular value is $1.4\times10^{-16}$ times its first. Over 200 batches of two, unit ratios of the update recover 0.963 of the examples exactly after one step and 0.917 after five.

*Local steps change the backward signals, not the inputs they multiply, so a client that trains on its data before sending an update has hidden nothing about a single example and little about a small batch.*
:::
