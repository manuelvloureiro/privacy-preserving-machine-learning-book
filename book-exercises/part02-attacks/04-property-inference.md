# Exercises: Property inference

Exercises for [Property inference](../../book-full/part02-attacks/04-property-inference.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part02-attacks/solutions_property_inference.py` computes every number quoted here; run it with `uv run python code/part02-attacks/solutions_property_inference.py`.

The solutions use the chapter's notation. $\mathcal D_t$ mixes a majority group and a subgroup of share $t$, a dataset is $n$ independent records from $\mathcal D_t$, and the attacker must tell $t_0$ from $t_1$, with $\bar t = \frac{t_0 + t_1}2$. Its advantage is the true positive rate minus the false positive rate, its accuracy with a fair bit is $\frac12(1 + \text{advantage})$, and the best advantage is the total variation (TV) distance between the two laws of what it sees. $\Phi$ is the standard normal distribution function.

## Shifting a unimodal pmf {#sec-sol-prop-1}

> **Exercise 1.** **Shifting a unimodal pmf.** Prove that $\mathrm{TV}(\mathrm{Law}(X), \mathrm{Law}(X + 1)) = \max_k\Pr[X = k]$ for any integer-valued $X$ whose pmf rises to a mode and then falls, and deduce @eq-prop-membership. *Check: for $X \sim \mathrm{Bin}(99, \frac12)$ both sides are 0.0796, and the membership advantage at $n = 100$ is half of that, 0.0398.* Find a pmf for which the identity fails. *(Total variation, membership)*

::: {.callout-tip collapse="true" title="Solution"}
Write $p_k = \Pr[X = k]$, and let $k^*$ be a mode: $p_k$ is non-decreasing for $k \le k^*$ and non-increasing for $k \ge k^*$.

1. **TV as a sum of differences.** $X + 1$ takes the value $k$ with probability $p_{k-1}$. So
$$\mathrm{TV}(\mathrm{Law}(X), \mathrm{Law}(X + 1)) = \frac12\sum_k|p_k - p_{k-1}|,$$
with $p_k = 0$ outside the support.
2. **The climb telescopes.** For $k \le k^*$ the differences are non-negative, so $\sum_{k \le k^*}|p_k - p_{k-1}| = \sum_{k \le k^*}(p_k - p_{k-1}) = p_{k^*}$. All terms cancel except the last, and the pmf starts from 0.
3. **The descent telescopes.** For $k > k^*$ the differences are non-positive, so $\sum_{k > k^*}|p_k - p_{k-1}| = \sum_{k > k^*}(p_{k-1} - p_k) = p_{k^*}$, since the pmf ends at 0.
4. **Add and halve.** The total is $2p_{k^*}$, and half of it is $p_{k^*} = \max_kp_k$.
5. **Membership.** Step 3 of the chapter's membership derivation reduced the membership advantage to $(1 - t)\,\mathrm{TV}(\mathrm{Law}(X + 1), \mathrm{Law}(X))$ with $X \sim \mathrm{Bin}(n - 1, t)$. A binomial pmf is unimodal: the ratio $p_{k+1}/p_k = \frac{(n - 1 - k)t}{(k + 1)(1 - t)}$ decreases in $k$, so it crosses 1 at most once. Steps 1–4 give $(1 - t)\max_kp_k$, which is @eq-prop-membership.
6. **Check.** For $X \sim \mathrm{Bin}(99, \frac12)$ the script finds $\mathrm{TV}(X, X + 1) = 0.0796 = \max_kp_k$. With $1 - t = \frac12$, the membership advantage at $n = 100$ is $0.0398$, which matches the game computed directly.

**A pmf where it fails.** Let $X$ be uniform on $\{0, 2\}$. Then $X + 1$ is uniform on $\{1, 3\}$, the two supports are disjoint, and the TV is 1. The largest probability is $\frac12$. The pmf goes up, down, up and down again, so the climbs and descents in steps 2 and 3 add up to $4 \times \frac12 = 2$, not $2\max_kp_k = 1$.

*A shift by one record is invisible exactly to the extent that the distribution is spread out; for a unimodal count, the height of the mode measures how spread out it is.*
:::

## How much data does the property attacker need? {#sec-sol-prop-2}

> **Exercise 2.** **How much data does the property attacker need?** With $t_0 = 0.45$ and $t_1 = 0.55$, use @eq-prop-property to find the $n$ at which the best attacker on the released count reaches accuracy 0.95. Then find the smallest such $n$ exactly. *Check: the approximation gives $n = 270.6$; the exact answer is 269.* *(Normal approximation)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Accuracy from advantage.** By @eq-prop-property, the best accuracy is approximately
$$\frac12 + \frac12\Big(2\Phi\Big(\frac{\sqrt n\,|t_1 - t_0|}{2\sqrt{\bar t(1 - \bar t)}}\Big) - 1\Big) = \Phi\Big(\frac{\sqrt n\,|t_1 - t_0|}{2\sqrt{\bar t(1 - \bar t)}}\Big).$$
2. **Invert $\Phi$.** Setting it to 0.95 gives $\sqrt n = 2\sqrt{\bar t(1 - \bar t)}\,\Phi^{-1}(0.95)/|t_1 - t_0|$. Here $\bar t = 0.5$, $\sqrt{\bar t(1 - \bar t)} = 0.5$, $\Phi^{-1}(0.95) = 1.6449$ and $|t_1 - t_0| = 0.1$, so $\sqrt n = 2 \times 0.5 \times 1.6449/0.1 = 16.449$.
3. **Square.** $n = 270.6$, so the approximation says 271 records.
4. **Exact.** The script computes $\frac12 + \frac12\mathrm{TV}(\mathrm{Bin}(n, 0.45), \mathrm{Bin}(n, 0.55))$ for each $n$. The smallest $n$ with accuracy at least 0.95 is 269, where the accuracy is 0.9501; at $n = 268$ it is 0.9495.

The approximation is off by two records. It uses the variance at the midpoint, $0.25$, for both hypotheses, while the true variances are $0.45 \times 0.55 = 0.2475$, slightly smaller, and it smooths over the integer steps of the count.

*A few hundred records suffice to tell a 45% share from a 55% share with 95% accuracy. A dataset of that size is small by any standard.*
:::

## Leakage in records {#sec-sol-prop-3}

> **Exercise 3.** **Leakage in records.** Suri & Evans measure a model's leakage by the number $k$ of records, with their groups visible, that an attacker would need to see to match the model's accuracy. Show that for a small gap, $k \approx n(\kappa\sqrt{\bar t(1 - \bar t)})^2$, which is $n(\kappa/2)^2$ at $\bar t = \frac12$. *Check: for the logistic regression at $n = 1{,}000$ and gap 0.1, the predicted accuracy is 0.8467 and the exact $k$ is 105, against $n(\kappa/2)^2 = 104.6$.* *(Scaling law)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The records attacker.** With $k$ records and their groups, the count of subgroup records is sufficient (the game section of the chapter), so the best accuracy is that of the count. By Exercise 2, step 1, it is approximately $\Phi\big(\sqrt k\,|t_1 - t_0|/(2\sqrt{\bar t(1 - \bar t)})\big)$.
2. **The model attacker.** By @eq-prop-kappa and @eq-prop-gauss-acc, the best accuracy from a model trained on $n$ records is approximately $\Phi\big(\kappa\sqrt n\,|t_1 - t_0|/2\big)$.
3. **Equate the arguments.** $\Phi$ is increasing, so the accuracies match when $\sqrt k/\sqrt{\bar t(1 - \bar t)} = \kappa\sqrt n$. The gap cancels, and
$$k = n\,\kappa^2\,\bar t(1 - \bar t).$$
4. **At $\bar t = \frac12$:** $\bar t(1 - \bar t) = \frac14$, so $k = n(\kappa/2)^2$.
5. **Check.** For the logistic regression, $\kappa = 0.647$ and $(\kappa/2)^2 = 0.105$. At $n = 1{,}000$ and $t_0, t_1 = 0.45, 0.55$, @eq-prop-delta predicts accuracy 0.8467. The smallest $k$ for which the exact binomial attacker reaches 0.8467 is 105, and the formula gives $1000 \times 0.1046 = 104.6$. At $n = 200$ the prediction is 0.6763, matched by 21 records, against $20.9$ from the formula.

*The logistic regression, which never sees the group, gives the property away as well as one record in ten would with its group label attached.*
:::

## Symmetries of a tanh network {#sec-sol-prop-4}

> **Exercise 4.** **Symmetries of a tanh network.** Show that $h!\,2^h$ parameter vectors represent the same generic network, and evaluate it for $h = 8$ and $h = 100$. Verify that $\sum_ju_ju_j^\top$ is invariant and that $\sum_ju_j$ is not invariant under sign flips. *Check: $h = 8$ gives 10,321,920.* *(Symmetry)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Every $g = F\Pi$ preserves the function.** This is @eq-prop-symmetry: $\tanh(gz) = g\tanh(z)$, and $g^\top g = I$.
2. **Count the $g$.** There are $h!$ permutations $\Pi$ and $2^h$ sign patterns $F$. Each pair gives a different matrix $g$, because $g$ sends the $j$-th basis vector to $\pm$ the $\pi(j)$-th, which determines both $\pi$ and the sign.
3. **Distinct $g$ give distinct parameters, for a generic network.** Suppose $g\theta = g'\theta$. Then $r = g'^{-1}g$ fixes $\theta$: it sends each unit $u_j$ to $\pm u_{\pi(j)}$ and gets $u_j$ back. If no two units are equal or opposite, this forces $\pi(j) = j$. And if no unit is 0, the sign must be $+$, since $u_j = -u_j$ would mean $u_j = 0$. So $r = I$, and $g = g'$. Generic parameters (drawn from a density) have distinct, non-opposite, non-zero units with probability 1.
4. **Evaluate.** $h = 4$: $24 \times 16 = 384$. $h = 8$: $40{,}320 \times 256 = 10{,}321{,}920$. $h = 100$: $100! \times 2^{100} = 1.1831\times10^{188}$.
5. **Second moments are invariant.** $\sum_j(gu)_j(gu)_j^\top$: a permutation reorders the terms of the sum, and a sign flip replaces $u_ju_j^\top$ by $(-u_j)(-u_j)^\top = u_ju_j^\top$.
6. **The plain sum is not.** Flipping unit 1 changes $\sum_ju_j$ by $-2u_1 \ne 0$. It is invariant under permutations only.
7. **Numerical check.** The script draws a random network with $h = 4$, applies the permutation $(3, 1, 4, 2)$ and flips units 2 and 3. The logits on 500 inputs agree to $10^{-12}$, the second-moment features agree to $10^{-12}$, and the raw weights differ by more than 0.1.

*For a network of realistic width, the number of equivalent parameter vectors is astronomically larger than any number of shadow models, so a meta-classifier cannot learn the symmetry from examples; it has to be built in.*
:::

## Several released models {#sec-sol-prop-5}

> **Exercise 5.** **Several released models.** The victim releases $k$ models, each trained on its own dataset of $n$ records from the same $\mathcal D_t$. Show that the best attack sums the $k$ linear scores, and that its accuracy is $\Phi(\sqrt k\,\gamma/2)$, where $\gamma$ is the Mahalanobis distance of @eq-prop-delta for one model. Compare with one model trained on $kn$ records. *Check: for the logistic regression at $n = 200$ and gap 0.2, $\gamma = 1.827$; with $k = 4$ the predicted accuracy is 0.966.* *(Fact 3, additivity)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The models are independent given $b$.** Each is trained on its own dataset, so under hypothesis $b$ the models $\theta_1, \dots, \theta_k$ are independent, each approximately $\mathcal N(m_b, \Sigma)$ with $m_b = \theta(t_b)$ and $\Sigma = \bar\Sigma_1/n$.
2. **Log-likelihood ratios add.** The joint density is a product, so the joint log-likelihood ratio is $\sum_iL(\theta_i)$, with $L(\theta) = \omega^\top(\theta - \frac{m_0 + m_1}2)$ from Fact 3. The best attack thresholds this sum at 0: it adds the linear scores.
3. **Its distribution.** Each $L(\theta_i)$ has mean $\pm\gamma^2/2$ and variance $\gamma^2$ (Fact 3, step 3). The sum of $k$ independent copies has mean $\pm k\gamma^2/2$ and variance $k\gamma^2$.
4. **Accuracy.** Under hypothesis 1 the attack is right when the sum is positive, with probability $\Phi\big((k\gamma^2/2)/(\sqrt k\,\gamma)\big) = \Phi(\sqrt k\,\gamma/2)$. Hypothesis 0 is symmetric.
5. **One model on $kn$ records.** By @eq-prop-delta, $\gamma$ grows like $\sqrt n$, so a single model trained on $kn$ records has Mahalanobis distance $\sqrt k\,\gamma$: the same accuracy.
6. **Check.** At $n = 200$ and $t_0, t_1 = 0.4, 0.6$, @eq-prop-delta gives $\gamma = 1.827$, so one model gives $\Phi(0.913) = 0.819$. Four models give $\gamma\sqrt4 = 3.653$ and accuracy $\Phi(1.827) = 0.966$, and one model on 800 records also has $\gamma = 3.653$. Simulation, with the meta-classifier fitted on 150 shadow models per side and scores summed over groups of 4 target models: accuracy 0.817 for one model and 0.970 for four.

*Every release adds evidence. A victim who retrains and republishes a model on fresh data from the same population leaks the property as if it had trained one model on all of the data.*
:::

## Noise sized for one record {#sec-sol-prop-6}

> **Exercise 6.** **Noise sized for one record.** The hospital adds $\mathcal N(0, \sigma^2)$ noise to the released count, with $n = 10{,}000$, $t_0 = 0.45$, $t_1 = 0.55$. Using normal approximations, find the property and membership advantages at $\sigma = 10$, and the $\sigma$ at which the property advantage falls to 0.1. *Check: at $\sigma = 10$ the property advantage is still 1.0000; hiding the property needs $\sigma \approx 3{,}979$.* *(Normal approximation, group size)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The released value.** $S + N$ is approximately normal, with variance $n\bar t(1 - \bar t) + \sigma^2 = 2{,}500 + \sigma^2$ under both hypotheses (variance at the midpoint, as in @eq-prop-property). Write $s = \sqrt{2500 + \sigma^2}$.
2. **Property.** The means differ by $n|t_1 - t_0| = 1{,}000$. The Gaussian pair has standardized shift $1000/s$, so the advantage is $2\Phi(1000/(2s)) - 1$. At $\sigma = 10$, $s = 50.99$ and the argument of $\Phi$ is 9.81: the advantage is 1.0000.
3. **Membership.** The target shifts $S$ by 1 with probability $1 - \bar t$ (step 2 of the chapter's membership derivation), so the advantage is $(1 - \bar t)(2\Phi(1/(2s)) - 1)$. Without noise this is 0.0040, and at $\sigma = 10$ it is 0.0039.
4. **Noise that hides the property.** Set $2\Phi(1000/(2s)) - 1 = 0.1$, so $\Phi(1000/(2s)) = 0.55$ and $1000/(2s) = \Phi^{-1}(0.55) = 0.1257$. Then $s = 3{,}979$, and $\sigma = \sqrt{s^2 - 2500} = 3{,}979$ to the nearest unit.
5. **What that noise costs.** The count is about 5,000, and the noise now has standard deviation about 4,000. The release says almost nothing about anything. The membership advantage has dropped to 0.000050, far below anything a membership attacker would care about.

*Noise of size 10 hides one record and leaves the property in plain view. To hide a property carried by 1,000 records, the noise must be on the scale of 1,000 records, a group-sized defence with a group-sized cost.*
:::
