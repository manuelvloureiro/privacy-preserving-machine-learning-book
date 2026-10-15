# Disparate impact

## Why this matters

[The chapter on private gradient descent](02-dp-sgd.md) showed that training the part's running example with differential privacy (DP) at $\varepsilon = 4$ costs less than one point of test accuracy: 0.9548 against 0.9610 without privacy. That number is an average over the test set. It says nothing about who loses the point. Bagdasaryan, Poursaeed & Shmatikov (2019) asked that question and found an uncomfortable answer: when a network is trained with stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, DP-SGD, the accuracy it loses is concentrated on the classes and subgroups that are under-represented in the training data. In their gender classifier, for example, the drop in accuracy was much larger on Black faces than on white faces, and the gap between groups that the non-private model already had grew under DP. They called this the **disparate impact** of DP on model accuracy.

This matters in two ways. A model that is "less than a point worse" can be ten points worse on a group of people, and an average hides it. And the groups that pay are often the ones a privacy guarantee is meant to protect: small populations, rare conditions, unusual records. This chapter explains why it happens, from the two ingredients of DP-SGD, measures it on the running example, and asks what can be done about it.

### The one idea: every record gets the same budget of influence, and every group gets the same noise

DP-SGD caps each record's pull on the update at $C$, then adds noise whose size depends only on $C$ and $\sigma$, not on the data. Both rules are blind to groups, and that is exactly why they are not neutral. A group's pull on the expected update is at most its share of the records times $C$, however badly the model fits it, so a group whose gradients are large is cut back more than a group whose gradients are small: that is the **clipping** mechanism. And the noise is the same for everyone, while a group's signal is proportional to its size, so a small group's signal-to-noise ratio is smaller by its share: that is the **noise** mechanism. Small groups and groups the model finds hard suffer from both, and the two compound, because clipping shrinks a group's effective share and the noise then acts on the shrunk share.

@tbl-fair-overview lists the results and where each is used.

| Result | Where it is used |
| --- | --- |
| Measuring disparate impact: privacy cost per group | Reporting a private model; [choosing ε](09-choosing-epsilon.md) |
| Clipping reweights groups and turns the update | Choosing $C$; group-aware clipping |
| The noise floor of a small group | Deciding whether a group is large enough to train on privately |
| Confidence intervals for per-group accuracy | Every per-group evaluation, private or not |
| Mitigations and what each costs | Training a private model that serves every group |

: The results of this chapter and where each is used. {#tbl-fair-overview}

**Notation.** As in [DP-SGD](02-dp-sgd.md), $n$ records $z_i = (x_i, y_i)$ train a model with weights $w$, $g_i$ is record $i$'s gradient, $C$ the clipping norm, $\sigma$ the noise multiplier, $q$ the sampling rate, $B = qn$ the expected batch size, $T$ the number of steps and $\eta$ the step size. The records are split into **groups** $g = 1, \dots, K$; group $g$ has $n_g$ records and **share** $s_g = n_g/n$. A record's **clipping weight** is $\min(1, C/\|g_i\|)$, the factor clipping multiplies its gradient by, and a group's clipping weight $w_g$ is its average over the group's records. $\mathrm{acc}_g$ is a model's accuracy on test records of group $g$. A subscript $i$ always names a record and a subscript $g$ or $h$ a group, so $g_i$ is a record's gradient and $\bar g_g$, defined below, a group's average gradient.

**The running example.** The part's running example classifies 50-dimensional inputs with a two-layer network ([DP-SGD](02-dp-sgd.md)). This chapter adds a minority group. Of the $n = 10{,}000$ training records, 9,500 are the first records of the running example's fixed training set: standard normal inputs, labelled by the random teacher network. The other 500, a share of 5%, come from a different region and follow a different rule. Their inputs are $x = 8u + z$, where $u$ is a fixed unit vector and $z$ is standard normal in a fixed 15-dimensional subspace orthogonal to $u$, and their label is the sign of one coordinate of $z$. So the minority's records lie far from the majority's, have larger inputs (average length 8.89 against 7.04), and need a rule that the majority does not teach. The non-private model learns both: it reaches 0.9529 on the majority and 0.9411 on the minority. Each group is tested on 20,000 fresh records of its own, so that the test error is small compared with the effects being measured. DP-SGD runs with the running example's settings: $q = 0.01$, $C = 0.5$, $\eta = 0.5$, $T = 10{,}000$ steps and $\delta = 10^{-5}$.

**How to read this chapter.** @sec-fair-observed defines what to measure and shows the effect. @sec-fair-clipping and @sec-fair-noise explain it, one mechanism each; they are the core of the chapter. @sec-fair-measure is about measuring per-group accuracy honestly, and applies to any model. @sec-fair-mitigation discusses what can be done and what each fix costs.

## Toolkit: four facts every derivation uses

### Fact 1 — The expected update is a reweighted average gradient

Each DP-SGD step adds up the clipped gradients of a Poisson batch, adds Gaussian noise of standard deviation $\sigma C$ to each coordinate, and divides by the public expected batch size $B = qn$. Each record is in the batch with probability $q$, so the expected sum is $q$ times the sum over all records, and dividing by $qn$ gives
$$\mathbb E\Big[\frac{G_t}{B}\Big] = \frac1n\sum_{i=1}^n\min\Big(1, \frac C{\|g_i\|}\Big)g_i,$$
the average gradient with each record weighted by its clipping weight ([DP-SGD](02-dp-sgd.md#sec-dpsgd-clipping)). The noise on the averaged update has standard deviation $\tau = \sigma C/B$ per coordinate, whatever the data. In the running example at $\varepsilon = 4$, $\sigma = 1.36$ and $\tau = 1.36 \times 0.5/100 = 0.0068$.

### Fact 2 — A drift beats a random walk after (noise/signal)² steps

Suppose each step moves the weights by a signal of length $\ell_{\text{sig}}$ that keeps pointing the same way, plus independent noise of length $\ell_{\text{noise}}$. After $k$ steps the signals add up to length $k\ell_{\text{sig}}$. The noise vectors are independent with mean zero, so their squared lengths add, and their sum has length about $\ell_{\text{noise}}\sqrt k$. The signal is longer once $k\ell_{\text{sig}} > \ell_{\text{noise}}\sqrt k$, that is after
$$k > \Big(\frac{\ell_{\text{noise}}}{\ell_{\text{sig}}}\Big)^2$$
steps. [DP-SGD](02-dp-sgd.md#sec-dpsgd-algorithm) uses this to explain why training works when every step's noise is ten times longer than its signal: at $\varepsilon = 4$ the noise on the averaged gradient has length $\tau\sqrt p$ for $p = 3{,}329$ weights, and the signal emerges after a few hundred steps.

### Fact 3 — A shrinking recursion with noise settles at a fixed variance

Let a number evolve as $x_{t+1} = (1 - a)x_t + e_t$, where $0 < a < 2$ and the $e_t$ are independent with mean zero and variance $v$. Each step shrinks $x$ toward zero by the factor $1 - a$ and adds fresh noise. Because the noise is independent of the current value, variances add: $\operatorname{Var}x_{t+1} = (1 - a)^2\operatorname{Var}x_t + v$. Since $(1 - a)^2 < 1$, this converges, and the limit $V$ solves $V = (1 - a)^2V + v$:
$$V = \frac{v}{1 - (1 - a)^2} = \frac{v}{a(2 - a)}.$$
For example, with $a = 0.1$ and $v = 1$ the variance settles at $1/(0.1 \times 1.9) = 5.26$: a weak pull toward zero lets the noise accumulate to more than five times one step's variance. Halving the pull roughly doubles the stationary variance. The number of steps to forget the starting point is about $1/a$, since $(1 - a)^t \approx e^{-at}$.

### Fact 4 — An accuracy measured on $m$ records is a binomial proportion

If a model classifies each test record of a group correctly with probability $p$, independently, then the number correct out of $m$ is binomial with mean $mp$ and variance $mp(1 - p)$, and the measured accuracy $k/m$ has standard deviation $\sqrt{p(1 - p)/m}$. The **Clopper–Pearson interval** ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md#sec-ht-estimation)) inverts the binomial distribution exactly: at the 95% level its upper end $u$ is the accuracy at which $k$ or fewer correct answers would have probability 0.025, and its lower end $l$ the accuracy at which $k$ or more would have probability 0.025. It covers the true accuracy with probability at least 0.95. The smaller the group, the wider the interval, in proportion to $1/\sqrt m$.

## Disparate impact, measured {#sec-fair-observed}

### Context

What exactly should you measure to say that DP hurts one group more than another, and does it happen on the running example? An accuracy gap between groups is not enough, because the non-private model may already have one. The question is how much *privacy* adds to each group's error.

### Statement

The **privacy cost** of group $g$ is the accuracy the group loses when the same training recipe is made private:
$$\pi_g = \mathrm{acc}_g(\text{non-private}) - \mathbb E\big[\mathrm{acc}_g(\text{private})\big],$$ {#eq-fair-cost}
where the expectation is over the randomness of private training (Esipova et al. 2023 use this definition; Bagdasaryan et al. 2019 compare the same differences). DP has **disparate impact** when the privacy costs of groups differ, $\pi_a \ne \pi_b$, and the **privacy cost gap** $|\pi_a - \pi_b|$ measures how much.

On the running example with a 5% minority, @tbl-fair-eps gives each group's accuracy over three training runs per row.

| Training | Majority accuracy | Minority accuracy | Gap | Privacy cost: majority | minority |
| --- | --- | --- | --- | --- | --- |
| Non-private SGD | $0.9529 \pm 0.0018$ | $0.9411 \pm 0.0052$ | 0.0119 | — | — |
| Clipping only, $\sigma = 0$ | $0.9581 \pm 0.0015$ | $0.9007 \pm 0.0048$ | 0.0574 | $-0.0051$ | 0.0404 |
| $\varepsilon = 8$, $\sigma = 0.917$ | $0.9550 \pm 0.0006$ | $0.8553 \pm 0.0110$ | 0.0997 | $-0.0020$ | 0.0858 |
| $\varepsilon = 4$, $\sigma = 1.36$ | $0.9470 \pm 0.0037$ | $0.8477 \pm 0.0211$ | 0.0993 | 0.0059 | 0.0934 |
| $\varepsilon = 2$, $\sigma = 2.28$ | $0.9366 \pm 0.0082$ | $0.7574 \pm 0.0379$ | 0.1791 | 0.0163 | 0.1836 |
| $\varepsilon = 1$, $\sigma = 4.13$ | $0.8822 \pm 0.0032$ | $0.6133 \pm 0.0173$ | 0.2689 | 0.0707 | 0.3278 |

: Test accuracy of each group (mean ± standard deviation over 3 runs) of the running example with a 5% minority, trained with DP-SGD at four budgets ($\delta = 10^{-5}$), with clipping alone, and without privacy. The privacy cost is @eq-fair-cost. {#tbl-fair-eps}

### Derivation

This statement is a definition and a measurement, so its "derivation" is the experiment's design, and each choice in it is there to make the comparison fair.

1. **Same data, same recipe.** Every row trains on the same 10,000 records with the same sampling rate, step size and number of steps. The only differences are the clipping (absent in the non-private row) and the noise multiplier, which the accountant of [DP-SGD](02-dp-sgd.md) sets for each $\varepsilon$.
2. **Large test sets.** Each group is tested on 20,000 records, so by Fact 4 a measured accuracy near 0.9 has a standard deviation of $\sqrt{0.9 \times 0.1/20{,}000} = 0.0021$. The differences in the table are much larger than that.
3. **Several runs.** Private training is random, and its spread differs between groups. Each row averages three runs and reports their standard deviation; the privacy cost subtracts the average private accuracy from the average non-private accuracy.
4. **A non-private model that serves both groups.** The minority's non-private accuracy, 0.9411, is within 1.2 points of the majority's, 0.9529. Any large gap in the private rows is therefore caused by privacy, not inherited from the data.

### Interpretation

**The minority pays almost all of it.** At $\varepsilon = 4$ the majority loses 0.59 points of accuracy and the minority 9.34, 15.7 times as much. At $\varepsilon = 8$, a budget at which [DP-SGD](02-dp-sgd.md) found the running example nearly as accurate as non-private training, the majority loses nothing (its accuracy is 0.20 points higher) and the minority loses 8.58 points. The ratio is smaller at $\varepsilon = 1$, 4.6, only because there the majority starts to lose too: 7.07 points against the minority's 32.78. An average over a test set drawn like the training set, 95% majority, would report a loss of $0.95 \times 0.0059 + 0.05 \times 0.0934 = 0.0103$ at $\varepsilon = 4$, about one point.

**Both mechanisms are visible.** The clipping-only row has no noise at all, yet the minority loses 4.04 points while the majority gains 0.51: clipping alone is disparate. Adding noise then widens the gap steadily as $\varepsilon$ falls. The next two sections explain each part.

**The minority's runs also vary more.** At $\varepsilon = 4$ the standard deviation over runs is 0.0211 for the minority and 0.0037 for the majority. A single private run says less about a small group than about a large one, a point @sec-fair-measure returns to.

![Test accuracy of each group of the running example against $\varepsilon$ (log scale, one run per point, $\delta = 10^{-5}$), with each group's non-private accuracy dashed. The majority is close to its non-private accuracy down to $\varepsilon = 2$; the minority falls far below it at every budget.](figures/fair_privacy_accuracy.svg){#fig-fair-privacy-accuracy}

### Example: what the published experiments found

Bagdasaryan, Poursaeed & Shmatikov (2019) trained classifiers with DP-SGD on images and text and found the same pattern: the under-represented classes and subgroups, and the more complex ones, lost the most accuracy, and an existing gap between groups grew. Esipova et al. (2023) report, for MNIST digits where class 8 was undersampled to about 1% of the training set, a privacy cost of 8.9 points for the typical class 2 and 57.9 points for class 8 under DP-SGD (their Table 2, $\varepsilon = 5.90$). On the running example, @fig-fair-privacy-accuracy shows the same shape: the majority's curve is the curve of [DP-SGD](02-dp-sgd.md), the minority's lies far below it.

### Check yourself

- Why is the privacy cost a better measure than the gap between the private model's group accuracies? *(Because part of the gap may be inherited from non-private training, which DP did not cause. The privacy cost subtracts each group's own non-private accuracy.)*
- At $\varepsilon = 4$, a test set with the training proportions shows an overall loss of under one point. What would a test set with equal numbers of both groups show? *(The average of the two privacy costs, $(0.0059 + 0.0934)/2 = 0.0497$: about five points. The overall number depends on the test set's mix as much as on the model.)*

## Clipping reweights the groups {#sec-fair-clipping}

### Context

Clipping without noise changes nothing about privacy (without noise there is none), yet in @tbl-fair-eps it costs the minority 4.04 points. How does a rule that treats every record the same way end up treating groups differently?

### Statement

Split the expected update of Fact 1 by group. Write $\bar g_g = \frac1{n_g}\sum_{i\in g}g_i$ for group $g$'s average gradient and $m_g = \frac1{n_g}\sum_{i\in g}\min(1, C/\|g_i\|)\,g_i$ for its average *clipped* gradient. Then
$$\mathbb E\Big[\frac{G_t}{B}\Big] = \sum_gs_g\,m_g, \qquad \|s_g\,m_g\| \le s_g\,C:$$ {#eq-fair-groups}
each group pulls with its share times its average clipped gradient, and that pull is never longer than its share times $C$, however large the group's gradients are.

When the records within each group have parallel gradients, each group's average clipped gradient is its average gradient scaled by a number, and the expected update becomes the gradient of a **reweighted** mixture of the groups:
$$\mathbb E\Big[\frac{G_t}{B}\Big] = \Big(\sum_hs_h\,c_h\Big)\sum_g\tilde s_g\,\bar g_g, \qquad \tilde s_g = \frac{s_g\,c_g}{\sum_hs_h\,c_h},$$ {#eq-fair-effective}
where $c_g = \|m_g\|/\|\bar g_g\|$ is the fraction of the group's average gradient that survives clipping and $\tilde s_g$ is its **effective share**. When all of a group's records have the same norm, $c_g$ is the group's clipping weight $w_g$.

### Derivation

1. **Split the sum.** Fact 1's expected update is $\frac1n\sum_i\min(1, C/\|g_i\|)g_i$. Group the records: $\frac1n\sum_g\sum_{i\in g}(\cdots) = \sum_g\frac{n_g}n\cdot\frac1{n_g}\sum_{i\in g}(\cdots) = \sum_gs_gm_g$.
2. **Bound the pull.** Each clipped gradient has length at most $C$, and the length of an average is at most the average of the lengths (the triangle inequality), so $\|m_g\| \le C$ and $\|s_gm_g\| \le s_gC$.
3. **Parallel gradients.** If every $g_i$ in group $g$ points along the same unit vector $e_g$, then $g_i = \|g_i\|e_g$ and $\min(1, C/\|g_i\|)g_i = \min(\|g_i\|, C)e_g$. Averaging, $m_g$ and $\bar g_g$ both point along $e_g$, so $m_g = c_g\bar g_g$ with $c_g = \|m_g\|/\|\bar g_g\|$.
4. **Factor out the total.** $\sum_gs_gc_g\bar g_g = (\sum_hs_hc_h)\sum_g\frac{s_gc_g}{\sum_hs_hc_h}\bar g_g$, which is @eq-fair-effective. The effective shares are non-negative and add up to 1.
5. **Equal norms.** If all of group $g$'s records also have the same norm $\nu_g$, then $c_g = \min(\nu_g, C)/\nu_g = \min(1, C/\nu_g) = w_g$.

### Interpretation

**Two errors: length and direction.** @eq-fair-effective separates what clipping does into two parts. The factor $\sum_hs_hc_h$ shortens the update, which a larger step size can undo. The effective shares change *which* mixture of the groups is being descended, and no step size undoes that. Esipova et al. (2023) make this distinction for a whole batch: they compare training with the clipped batch gradient's length but the true direction against the true length but the clipped direction, and find the direction error far more harmful, especially to the under-represented class (their Table 1). In @fig-fair-clipping-toy, clipping shortens the update and also turns it toward the majority.

**Which groups are cut back.** A group's effective share falls when its clipping weight falls, that is when its gradients are long compared with $C$. Gradients are long for two reasons. The model fits the group badly: for the logistic loss, the derivative with respect to the output is $-y\,\mathrm{sigmoid}(-yf)$, close to 1 in size for a misclassified record and close to 0 for a record classified with a wide margin. And the group's inputs are large: a dense layer's per-example gradient is the outer product of the back-propagated error and the layer's input, whose norm is the product of the two norms ([DP-SGD](02-dp-sgd.md#sec-dpsgd-ghost)), so at the same error a record with a longer input has a longer gradient. Tran, Dinh & Fioretto (2021) identify both factors, gradient norms and the input norms behind them, as the source of the clipping part of disparate impact. A small group the model has not yet learned has both.

**A feedback loop.** Early in training every record is clipped and every group has a similar weight. As the majority is learned, its gradients shrink below $C$ and its weight rises toward 1, while the minority, not yet learned, keeps long gradients and a low weight. Its effective share falls below its true share, so it is learned more slowly, so its gradients stay long. The loop ends only when the minority's gradients fall below $C$ too.

**Where training stops.** Clipped gradient descent stops where the expected update is zero, $\sum_gs_gm_g = 0$, not where the true average gradient $\sum_gs_g\bar g_g$ is zero. If the groups' gradients conflict, the stopping point is that of the reweighted objective, which has moved toward the majority.

![The clipping example in two dimensions: 95 majority records with gradient (0.2, 0) and 5 minority records with gradient (0, 4). Without clipping the two groups pull about equally hard (black); clipping at C = 0.5 cuts the minority's pull eightfold and turns the update toward the majority (red). Scaling every gradient by the same factor 1/8 instead keeps the direction (blue).](figures/fair_clipping_toy.svg){#fig-fair-clipping-toy}

### Example: two dimensions, then the running example

Take 95 majority records whose gradient is $(0.2, 0)$, already short, and 5 minority records whose gradient is $(0, 4)$, long because the model fits them badly. Without clipping the average gradient is $(0.95 \times 0.2, 0.05 \times 4) = (0.19, 0.2)$, of length 0.2759: the five minority records pull as hard as the ninety-five others, and the minority's component is a fraction $0.2/0.39 = 0.513$ of the sum of the two components. Clip at $C = 0.5$. The majority's gradients are untouched; the minority's shrink to length 0.5. The average becomes $(0.19, 0.05 \times 0.5) = (0.19, 0.025)$, of length 0.1916, and the minority's fraction falls to 0.116. The update has turned by 39.0 degrees toward the majority. Here $c_{\text{maj}} = 1$ and $c_{\text{min}} = 0.5/4 = 0.125$, and @eq-fair-effective gives the minority an effective share of $0.05 \times 0.125/(0.05 \times 0.125 + 0.95) = 0.0065$ instead of 0.05. Scaling every gradient by $C/Z$ with $Z = 4$, the largest norm, instead gives $(0.0238, 0.0250)$: eight times shorter, but at an angle of 0.0 degrees to the true average.

In the running example, @tbl-fair-weights follows each group's gradients during training with clipping and no noise, on 1,000 majority and the 500 minority training records.

| Step | Median norm: majority | minority | Clipped: majority | minority | Weight $w_g$: majority | minority | Effective share of the minority | Test accuracy: majority | minority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3.4 | 3.7 | 1.00 | 1.00 | 0.165 | 0.163 | 0.0495 | 0.5064 | 0.5019 |
| 2,500 | $3.8 \times 10^{-4}$ | 2.7 | 0.18 | 0.70 | 0.869 | 0.424 | 0.0251 | 0.9569 | 0.6724 |
| 5,000 | $1.4 \times 10^{-4}$ | 0.71 | 0.14 | 0.54 | 0.891 | 0.579 | 0.0331 | 0.9607 | 0.8137 |
| 7,500 | $7.7 \times 10^{-5}$ | 0.30 | 0.13 | 0.45 | 0.906 | 0.651 | 0.0364 | 0.9621 | 0.8661 |
| 10,000 | $7.0 \times 10^{-5}$ | 0.16 | 0.11 | 0.39 | 0.917 | 0.702 | 0.0387 | 0.9613 | 0.8900 |

: Per-group gradient statistics during training with clipping at $C = 0.5$ and no noise. "Clipped" is the fraction of records with gradient norm above $C$. The effective share is $s\,w_{\text{min}}/(s\,w_{\text{min}} + (1 - s)\,w_{\text{maj}})$ with $s = 0.05$, @eq-fair-effective with $c_g = w_g$. {#tbl-fair-weights}

At step 0 both groups are fully clipped, with similar weights, and the minority's effective share, 0.0495, is its true share. By step 2,500 the majority's median gradient norm has fallen to $3.8 \times 10^{-4}$, though 18% of its records, those still near the decision boundary, are clipped. The minority's median norm is still 2.7 and 70% of its records are clipped. Its effective share has fallen to 0.0251, half its true share, exactly while it most needs to be learned: its test accuracy is 0.6724 against the majority's 0.9569. It recovers only as the minority is learned, to 0.0387 at the end. The same happens with noise (@fig-fair-clipping-training): at $\varepsilon = 4$ the minority's effective share is 0.0261 at step 2,500 and 0.0358 at the end, where its weight is 0.637 against the majority's 0.902. The noise keeps the minority's gradients long, so clipping keeps cutting them back.

![Left: the median per-example gradient norm of each group during non-private training (solid) and DP-SGD at $\varepsilon = 4$ (dashed), on a log scale, with the clipping norm dotted. The majority's gradients fall far below $C$ within the first thousand steps; the minority's stay above it much longer. Right: each group's mean clipping weight during DP-SGD at $\varepsilon = 4$.](figures/fair_clipping_training.svg){#fig-fair-clipping-training}

### Check yourself

- If you multiply the minority's loss by 10, does its pull on the update grow tenfold? *(Not for clipped records. Their gradients grow tenfold and are then rescaled back to length $C$; only records whose upweighted gradient stays below $C$ gain. [Exercise 1](../../book-exercises/part05-dp-ml/08-disparate-impact.md#sec-sol-fair-1) works it out.)*
- Why does a larger clipping norm reduce the clipping part of disparate impact? *(It raises the clipping weights of the groups with long gradients more than those of groups whose gradients are already below $C$, whose weight is already 1, so the effective shares move back toward the true shares.)*

## The noise floor of a small group {#sec-fair-noise}

### Context

Noise is added to the sum, the same for everyone. Why should it hurt a small group more than a large one? Because what the noise competes with is each group's signal, and a group's signal in the sum is proportional to its size.

### Statement

**A drift that is a fraction of the population's.** If the part of the expected update that serves group $g$ is a fraction $s_g$ of what the whole population's signal would be, and the noise is unchanged, Fact 2 says that group's signal emerges from the noise only after
$$k_g = \frac1{s_g^2}\Big(\frac{\ell_{\text{noise}}}{\ell_{\text{sig}}}\Big)^2$$ {#eq-fair-breakeven}
steps, $1/s_g^2$ times as many as for a signal the whole population shares.

**A quadratic model of the noise floor.** Suppose group $g$'s loss depends on its own block of $k$ weights, which the other groups' losses do not touch, as $L_g = \frac\lambda2\|w_g - w_g^*\|^2$, and the training objective is the share-weighted average $\sum_gs_gL_g$. Run gradient descent with noise of standard deviation $\tau$ per coordinate on the averaged gradient, as DP-SGD adds. Then group $g$'s excess loss settles at
$$\mathbb E[L_g] = \frac{k\,\eta\,\tau^2}{2\,s_g\,(2 - \eta s_g\lambda)} \approx \frac{k\,\eta\,\tau^2}{4\,s_g},$$ {#eq-fair-floor}
and it takes about $1/(\eta s_g\lambda)$ steps to get there. With clipping, $s_g$ is replaced by $s_gc_g$, the group's share times the fraction of its gradient that survives clipping; when the majority is hardly clipped, this is close to the effective share $\tilde s_g$ of @eq-fair-effective.

### Derivation

1. **The group's gradient.** The objective's gradient with respect to $w_g$ is $s_g\lambda(w_g - w_g^*)$: only group $g$'s loss depends on these weights, and it enters with weight $s_g$.
2. **One step.** With $x = w_g - w_g^*$ and noise $\tau\xi$ on the averaged gradient, $\xi$ standard normal, the update $w_g \leftarrow w_g - \eta(s_g\lambda x + \tau\xi)$ gives $x \leftarrow (1 - \eta s_g\lambda)x - \eta\tau\xi$, coordinate by coordinate.
3. **Stationary variance.** This is Fact 3 with $a = \eta s_g\lambda$ and $v = \eta^2\tau^2$: each coordinate settles at variance $\eta^2\tau^2/(a(2 - a)) = \eta\tau^2/(s_g\lambda(2 - \eta s_g\lambda))$.
4. **Excess loss.** $L_g = \frac\lambda2\sum_jx_j^2$ over the $k$ coordinates, so its expectation is $\frac\lambda2k$ times the variance, which is @eq-fair-floor. For a small group $\eta s_g\lambda$ is small and $2 - \eta s_g\lambda \approx 2$.
5. **Time scale.** By Fact 3 the starting point is forgotten after about $1/a = 1/(\eta s_g\lambda)$ steps.
6. **The break-even law.** If the population's signal has length $\ell_{\text{sig}}$ per step, the part that serves group $g$ has length $s_g\ell_{\text{sig}}$ and the noise length is still $\ell_{\text{noise}}$; Fact 2 with $s_g\ell_{\text{sig}}$ in place of $\ell_{\text{sig}}$ gives @eq-fair-breakeven.
7. **With clipping.** By @eq-fair-effective, with parallel gradients within the group, group $g$'s term in the expected update is $s_gc_g\bar g_g$ instead of $s_g\bar g_g$, while the noise is unchanged. Treating $c_g$ as constant over the steps considered, steps 1 to 6 apply with $s_gc_g$ in place of $s_g$. If the majority's $c$ is close to 1, then $\sum_hs_hc_h$ is close to 1 and $s_gc_g$ is close to $\tilde s_g$.

### Interpretation

**The floor scales like one over the share.** A group with a twentieth of the records sits at about twenty times the excess loss of a group with all of them. The curvature $\lambda$ cancels: a group whose loss is sharper is pulled back harder, but each unit of noise also costs it more, and the two effects balance. What does not cancel is the share, because the noise is not scaled by it. It also takes the small group $1/s_g$ times longer to reach even that floor.

**Why waiting does not help.** At a fixed budget, more steps require more noise per step. Under the small-$q$ law of [DP-SGD](02-dp-sgd.md#sec-dpsgd-batch), multiplying the number of steps by $m$ multiplies the noise variance per step by about $m$, so the break-even count of @eq-fair-breakeven grows as fast as the run ([Exercise 3](../../book-exercises/part05-dp-ml/08-disparate-impact.md#sec-sol-fair-3)).

**The two mechanisms compound.** Clipping lowers a group's pull from $s_g$ to $s_gc_g$; the noise floor scales like one over the pull. A group that is 5% of the data and whose gradients clipping halves pays the floor of a group of 2.5%.

**How good is the model.** The quadratic model is exact for what it assumes, as the simulation below confirms, but a network is not quadratic and its groups do not use disjoint weights. Where the minority's task overlaps the majority's, the majority's gradient does part of the minority's work, and the penalty is smaller. Tran, Dinh & Fioretto (2021) derive a related per-step decomposition for DP-SGD without assuming disjoint weights: the expected change in a group's loss has a clipping term, driven by the group's gradient norms, and a noise term proportional to $\eta^2\sigma^2C^2$ times the trace of the group's loss Hessian, which they relate to the group's distance to the decision boundary and its input norms. The quadratic model makes the role of the share explicit, which their per-step terms leave implicit.

![Left: each group's test accuracy of the running example against the minority's share of the 10,000 training records (one run per point), non-private and at $\varepsilon = 4$. Right: the stationary excess loss of a group in the quadratic model, @eq-fair-floor, against its share, with the noise of $\varepsilon = 4$, compared with simulation.](figures/fair_group_size.svg){#fig-fair-group-size}

### Example: the model, then the running example

At $\varepsilon = 4$ the noise on the averaged gradient has $\tau = 0.0068$ per coordinate. Take $k = 51$ weights per group, $\lambda = 1$ and $\eta = 0.5$. @tbl-fair-floor compares @eq-fair-floor with a simulation of 40,000 noisy steps, averaged over the second half.

| Share $s_g$ | Excess loss, @eq-fair-floor | Simulated | Steps to settle, $1/(\eta s_g)$ |
| --- | --- | --- | --- |
| 0.95 | $4.07 \times 10^{-4}$ | $4.07 \times 10^{-4}$ | 2 |
| 0.2 | $1.55 \times 10^{-3}$ | $1.55 \times 10^{-3}$ | 10 |
| 0.05 | $5.97 \times 10^{-3}$ | $5.99 \times 10^{-3}$ | 40 |
| 0.01 | $2.96 \times 10^{-2}$ | $2.97 \times 10^{-2}$ | 200 |

: The noise floor of a group in the quadratic model with $\varepsilon = 4$ noise ($\tau = 0.0068$, $k = 51$, $\lambda = 1$, $\eta = 0.5$). {#tbl-fair-floor}

The formula and the simulation agree to within 1%, and the floor grows from $4.07 \times 10^{-4}$ for a group with 95% of the data to $2.96 \times 10^{-2}$ for a group with 1%, a factor of 73. The approximate form of @eq-fair-floor would say 95, the ratio of the shares; the exact form is lower because the factor $2 - \eta s_g\lambda$ is 1.525 for the large group rather than 2. The break-even law is harsher. In [DP-SGD](02-dp-sgd.md) the averaged clipped gradient of the running example has length about 0.026 during training, and at $\varepsilon = 4$ the noise has length $\tau\sqrt{3{,}329} = 0.392$. A signal shared by the whole population emerges after $(0.392/0.026)^2 = 228$ steps. A signal that is 20% of it needs 5,693 steps, 5% needs 91,085, nine times the entire run, and 1% needs 2,277,115. A group whose rule the rest of the data does not teach is in this position.

In the network, @tbl-fair-size varies the minority's share with everything else fixed.

| Minority share | Records | Non-private: majority | minority | $\varepsilon = 4$: majority | minority | Privacy cost: majority | minority |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1% | 100 | $0.9556 \pm 0.0015$ | $0.8498 \pm 0.0021$ | $0.9558 \pm 0.0036$ | $0.5009 \pm 0.0102$ | $-0.0002$ | 0.3488 |
| 5% | 500 | $0.9529 \pm 0.0018$ | $0.9411 \pm 0.0052$ | $0.9470 \pm 0.0037$ | $0.8477 \pm 0.0211$ | 0.0059 | 0.0934 |
| 20% | 2,000 | $0.9395 \pm 0.0016$ | $0.9787 \pm 0.0004$ | $0.9355 \pm 0.0019$ | $0.9507 \pm 0.0038$ | 0.0041 | 0.0280 |

: Each group's test accuracy for three sizes of the minority, non-private and at $\varepsilon = 4$ (mean ± standard deviation over 2 runs; the 5% row is from @tbl-fair-eps, over 3 runs). {#tbl-fair-size}

The minority's privacy cost falls steeply with its share: 34.88 points at 1%, 9.34 at 5%, 2.80 at 20%. Multiplying the share by 5, from 1% to 5%, divides the cost by $0.3488/0.0934 = 3.7$, and multiplying it by 4, from 5% to 20%, divides it by $0.0934/0.0280 = 3.3$: a little less than the ratio of the shares, which is what @eq-fair-floor predicts for the loss. Accuracy is not loss, and it saturates: at 1% the private model is at chance on the minority, 0.5009, and cannot lose more. The majority's cost stays below one point throughout. With 20% of the data, the minority is no longer a minority in any sense that matters to the noise, and its privacy cost is within three points of the majority's (@fig-fair-group-size, left).

### Check yourself

- The noise multiplier is doubled. By how much does the noise floor of @eq-fair-floor grow, and does it grow more for the small group? *(By a factor of 4, since it is proportional to $\tau^2$, for every group alike. The gap between groups grows by the same factor in absolute terms; the ratio between them, $s_{\text{major}}/s_{\text{minor}}$, is unchanged.)*
- Why does the curvature $\lambda$ drop out of the approximate noise floor? *(The pull toward the optimum and the cost of a displacement are both proportional to $\lambda$; the stationary variance falls like $1/\lambda$ and the loss per unit of variance rises like $\lambda$.)*

## Measuring per-group accuracy {#sec-fair-measure}

### Context

Every number so far was measured on 20,000 test records per group. Real evaluations are not so generous: a test split of 2,000 records with a 5% minority holds 100 minority records. How precisely can a group's accuracy, and the gap between two groups, be measured, and how much does a single training run say?

### Statement

The test-set error of a group's accuracy is binomial (Fact 4): with $k$ correct of $m$ records, report the Clopper–Pearson interval. For the gap between two groups measured on disjoint test records, the two estimates are independent and their variances add, so an approximate 95% interval is
$$(\hat p_a - \hat p_b) \pm 1.96\sqrt{\frac{\hat p_a(1 - \hat p_a)}{m_a} + \frac{\hat p_b(1 - \hat p_b)}{m_b}},$$ {#eq-fair-gap}
with $\hat p_g$ the measured accuracies and $m_g$ the numbers of test records. Training randomness is a second, separate source of variation: a group's accuracy varies from run to run, more for small groups, and is estimated only by training several times.

### Derivation

1. **One group.** Each test record is classified correctly or not, independently given the model, with probability equal to the group's true accuracy $p_g$. The count is binomial, and Clopper–Pearson inverts it exactly ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md#sec-ht-estimation)).
2. **The gap.** The two counts come from disjoint test records, so they are independent given the model. The variance of a difference of independent estimates is the sum of their variances, $p_a(1 - p_a)/m_a + p_b(1 - p_b)/m_b$. For large counts the difference is approximately normal by the central limit theorem, and 1.96 standard deviations on each side cover 95%.
3. **Two sources.** Conditional on the trained model, the spread comes from the test set alone. Over training runs, the model itself varies; by the law of total variance the spread of a single run's measured accuracy is the test-set variance plus the run-to-run variance of the true accuracy. Only repeated training estimates the second.

### Interpretation

**Small groups need intervals.** The width of the Clopper–Pearson interval falls like $1/\sqrt m$ (@fig-fair-intervals). A 5-point gap between groups can be seen only with a few hundred minority test records, and a 1-point gap only with thousands ([Exercise 4](../../book-exercises/part05-dp-ml/08-disparate-impact.md#sec-sol-fair-4)). Report the interval, not only the point estimate, for every group.

**Report the spread over runs.** In @tbl-fair-eps the run-to-run standard deviation at $\varepsilon = 4$ is 0.0211 for the minority against 0.0037 for the majority, almost six times as large. A comparison of two training recipes on one run each can easily be reversed for the small group. This is the same lesson as for the overall accuracy of private models ([DP-SGD](02-dp-sgd.md#sec-dpsgd-run)), sharper for small groups.

**The evaluation leaks too.** Per-group accuracies computed on private test data are statistics of that data. If they are published, they need their own privacy analysis, which [Choosing ε](09-choosing-epsilon.md) and the selection mechanisms of [Hyperparameter tuning](03-hyperparameter-tuning.md) address.

![Width of the 95% Clopper–Pearson interval for a group's accuracy against the number of test records in the group, for observed accuracies 0.95 and 0.7, on log scales. The width falls like one over the square root of the number of records.](figures/fair_intervals.svg){#fig-fair-intervals}

### Example: one model, three test sets

Train the running example at $\varepsilon = 4$ once. On all 20,000 minority test records its accuracy is 0.8641. @tbl-fair-ci measures it on random subsets.

| Test records | Correct | Estimate | 95% Clopper–Pearson interval |
| --- | --- | --- | --- |
| 100 minority | 88 | 0.880 | $[0.800, 0.936]$ |
| 500 minority | 424 | 0.848 | $[0.813, 0.878]$ |
| 2,000 minority | 1,742 | 0.871 | $[0.856, 0.885]$ |
| 1,900 majority | 1,812 | 0.954 | $[0.943, 0.963]$ |

: Minority accuracy of one model trained at $\varepsilon = 4$, measured on random subsets of the minority's test records, with the majority on 1,900 records for comparison. {#tbl-fair-ci}

With 100 minority records, the split a 2,000-record test set with a 5% minority would give, the estimate is 0.880, 1.6 points above the true 0.8641, and the interval is 13.6 points wide. The true accuracy is inside it, as it should be 95% of the time, and the non-private model's 0.9411 from @tbl-fair-eps lies only just above its upper end, 0.936: on this test set alone the minority's privacy cost could be anywhere from about half a point to fourteen points. With 500 records the interval is 6.5 points wide and still contains the true value. The majority's interval, from 1,900 records at a higher accuracy, is 2 points wide. By @eq-fair-gap the gap between the groups, measured on the 1,900 and the 100 records, is $0.954 - 0.880 = 0.074$ with a standard error of $\sqrt{0.954 \times 0.046/1{,}900 + 0.88 \times 0.12/100} = 0.0329$, so its 95% interval is $0.074 \pm 1.96 \times 0.0329 = 0.074 \pm 0.064$, from 0.010 to 0.138: the gap is real, but it could be anywhere from one point to fourteen. And this is one training run: by @tbl-fair-eps the minority's accuracy at $\varepsilon = 4$ varies by 0.0211 from run to run on top of this.

### Check yourself

- A report gives a private model's accuracy as 0.84 on a subgroup of 60 test records and 0.85 non-private on the same records. What can you conclude? *(Almost nothing about this subgroup. Each accuracy's 95% interval is about $2 \times 1.96\sqrt{0.84 \times 0.16/60} = 0.19$ wide, nineteen times the one-point difference. Because both models were tested on the same records, the difference could be measured more precisely from the records on which they disagree, but the report does not say how many there are.)*
- Why doesn't a larger test set fix the run-to-run variation? *(It removes the test-set error for the model at hand, but a different training run produces a different model. Only repeated training estimates how much a group's accuracy varies with the training noise.)*

## Mitigations and what they cost {#sec-fair-mitigation}

### Context

Both mechanisms come from DP-SGD's two knobs. Can you choose the knobs, or change the algorithm, so that small groups pay less, without spending more privacy? Each fix in this section moves cost around rather than removing it, and it is worth knowing where to.

### Statement

**Global scaling is a large clipping norm.** Scaling every per-example gradient by $C/Z$, where no gradient is longer than $Z$, and adding noise $\sigma C$, is the same algorithm as DP-SGD with clipping norm $Z$, the same noise multiplier and step size $\eta C/Z$:
$$\frac\eta B\Big(\frac CZ\sum_{i\in S_t}g_i + \sigma C\xi\Big) = \frac{\eta C/Z}B\Big(\sum_{i\in S_t}g_i + \sigma Z\xi\Big).$$ {#eq-fair-global}

**Oversampling a group costs that group's privacy.** If group $g$'s records are sampled at rate $vq$ instead of $q$, its pull on the update grows by the factor $v$, and each of its records has the privacy of the subsampled Gaussian at rate $vq$:
$$\varepsilon_g = \varepsilon_{\text{acct}}(vq, \sigma, T, \delta) \;>\; \varepsilon_{\text{acct}}(q, \sigma, T, \delta),$$ {#eq-fair-oversample}
where $\varepsilon_{\text{acct}}$ is the accountant of [DP-SGD](02-dp-sgd.md). The algorithm's guarantee is the largest over records, so a fixed target $\varepsilon$ requires the noise multiplier for rate $vq$.

### Derivation

1. **Global scaling.** Factor $C/Z$ out of the left-hand side: $\frac CZ\sum_ig_i + \sigma C\xi = \frac CZ(\sum_ig_i + \sigma Z\xi)$. With clipping norm $Z$ and no gradient longer than $Z$, DP-SGD's clipped sum is $\sum_ig_i$ and its noise $\sigma Z\xi$, which is the right-hand side. Both have sensitivity-to-noise ratio $1/\sigma$, so the same accountant applies ([Exercise 6](../../book-exercises/part05-dp-ml/08-disparate-impact.md#sec-sol-fair-6)).
2. **Oversampling: the pull.** The expected clipped sum is $\sum_iq_i\min(1, C/\|g_i\|)g_i$, with $q_i$ record $i$'s sampling rate. With $q_i = vq$ for group $g$ and division by the public $B = qn$, group $g$'s term becomes $vs_gm_g$.
3. **Oversampling: the privacy.** The add/remove analysis of record $x$ in [DP-SGD](02-dp-sgd.md#sec-dpsgd-algorithm) compares the output with and without $x$; the mixture depends only on $x$'s own inclusion probability, because the other records' terms are the same in both. A record sampled at rate $vq$ therefore has the privacy of the subsampled Gaussian at rate $vq$. The sampling rate may depend on the record's own group, which is part of the record, but the rule must be fixed in advance, not chosen by looking at the data.
4. **The guarantee.** $(\varepsilon, \delta)$-DP must hold for every record, so it is the largest of the per-record values, which is @eq-fair-oversample's.

### Interpretation

**Global scaling trades direction for noise.** By @eq-fair-global, scaling all gradients by the same factor is the large-$C$ end of the clipping-norm curve of [DP-SGD](02-dp-sgd.md#sec-dpsgd-clipping) with $\eta C$ fixed: nothing is clipped, so the update points in the true direction (the blue arrow of @fig-fair-clipping-toy), but the noise $\sigma Z$ is as large as possible relative to the gradients. Esipova et al. (2023) propose a version that clips the rare gradients above $Z$ instead of discarding them and adapts $Z$ during training from a privately estimated count of how many gradients exceed it; they report that it reduces the privacy cost gap and improves accuracy compared with DP-SGD and with earlier mitigations on MNIST, CelebA and two tabular datasets. Because it treats every record alike, it needs no group labels.

**Group-aware methods need group labels.** Xu, Du & Wu (2021) adjust each group's contribution according to how much that group is being clipped, giving groups that are clipped more a larger clipping bound, so that the groups' clipping biases are equalized. Tran, Dinh & Fioretto (2021) add to the training objective penalties on the differences between groups in the quantities their decomposition identifies. Both need each training record's group, which is itself sensitive and may not be available or lawful to collect, and the per-group statistics they use must be computed privately.

**Oversampling buys little at a fixed budget.** Fixed at $\varepsilon = 4$ for every record, sampling the minority four times as often requires $\sigma = 4.71$ instead of 1.36: the minority's pull grows by 4 and the noise by $4.71/1.36 = 3.46$. Its signal-to-noise ratio improves by only $4/3.46 = 1.16$, while the majority's falls by 3.46, and the majority's records are then far more private than the target, $\varepsilon = 0.86$, which is wasted budget ([Exercise 5](../../book-exercises/part05-dp-ml/08-disparate-impact.md#sec-sol-fair-5)). Oversampling without changing $\sigma$ would silently give the minority $\varepsilon = 20.96$: the group that is already disadvantaged in accuracy would also get the weakest privacy.

**Some of the cost is unavoidable.** Learning a group's own rule privately is, in part, a private estimation problem on that group's $n_g$ records, and private estimation has a price that no algorithm avoids. [Lower bounds](../part03-dp-foundations/09-lower-bounds.md) proves it for the simplest case: releasing $d$ attribute means with constant accuracy under $(\varepsilon, \delta)$-DP needs at least about $\sqrt d/(e^\varepsilon - 1)$ records, by the fingerprinting argument, whatever the mechanism. Put differently, the error that $n_g$ records allow shrinks only like $1/(n_g\varepsilon)$, and the Gaussian mechanism, with error about $\sqrt{d\ln(1/\delta)}/(n_g\varepsilon)$, is close to the best possible. A smaller group is a smaller dataset, and no clever clipping makes it larger. Feldman (2020) argues the deeper point for any learner: when the data have a long tail of small subpopulations, fitting them well requires memorizing their few examples, which is exactly what DP forbids. What helps is more data from the group, a larger $\varepsilon$ (with its own costs), or a starting point that already serves the group, such as a public pretrained model ([Public data and private fine-tuning](06-public-data.md)).

### Example: the clipping norm and oversampling on the running example

@tbl-fair-mitigation compares three changes at $\varepsilon = 4$ for every record.

| Change | $\sigma$ | Majority accuracy | Minority accuracy | Gap |
| --- | --- | --- | --- | --- |
| None: $C = 0.5$, $\eta = 0.5$ | 1.36 | $0.9470 \pm 0.0037$ | $0.8477 \pm 0.0211$ | 0.0993 |
| Smaller clipping norm: $C = 0.05$, $\eta = 5$ | 1.36 | $0.9541 \pm 0.0008$ | $0.8313 \pm 0.0157$ | 0.1229 |
| Larger clipping norm: $C = 5$, $\eta = 0.05$ | 1.36 | $0.9463 \pm 0.0027$ | $0.8133 \pm 0.0166$ | 0.1330 |
| Oversample the minority four times | 4.71 | $0.8669 \pm 0.0079$ | $0.8270 \pm 0.0111$ | 0.0399 |

: Mitigations at $\varepsilon = 4$ on the running example with a 5% minority (mean ± standard deviation). The first row is from @tbl-fair-eps (3 runs); the clipping norm rows keep $\eta C = 0.25$, 2 runs each; the oversampling row samples minority records at rate $4q$ and sets $\sigma$ so that they meet $\varepsilon = 4$, 3 runs. {#tbl-fair-mitigation}

None of the three changes helps the minority. A ten times smaller clipping norm clips the minority more, and its accuracy falls by 1.64 points while the majority's rises by 0.71: the clipping mechanism of @sec-fair-clipping at work. A ten times larger one clips almost nothing, so the update's direction is right, but with $\eta C$ fixed the noise in each step, $\eta\sigma C/B$, is unchanged while the step along every gradient shorter than 0.5 is ten times smaller: the noise is ten times larger relative to that part of the signal, and the minority falls by 3.44 points: the noise mechanism of @sec-fair-noise at work. The running example's $C = 0.5$ sits between the two failures. These differences are of the size of the run-to-run spread, 0.0157 to 0.0211, so only their direction is clear, not their size.

Oversampling narrows the gap from 9.93 to 3.99 points, but by **levelling down**: the minority's accuracy is 2.07 points lower than without it, and the majority's 8.01 points lower. A smaller gap is not by itself a fairer model. Report both groups' accuracy and privacy cost, never the gap alone.

### Check yourself

- Why does oversampling a group require more noise, while upweighting its loss does not? *(Upweighting leaves every record's contribution bounded by $C$ and its sampling rate at $q$, so the accountant is unchanged, but clipping undoes it. Oversampling raises the group's sampling rate, which the subsampled Gaussian's privacy depends on.)*
- A team proposes to fix disparate impact by setting $C$ to the largest per-example gradient norm observed in the training data. What is wrong? *(The clipping norm then depends on the data, and the sensitivity argument of DP-SGD assumes it is fixed in advance. A data-dependent $C$ must itself be chosen privately, as in Esipova et al.'s adaptive threshold, and the estimate charged to the budget.)*

## Code: check every number by computation

The script [code/disparate_impact.py](../../code/part05-dp-ml/disparate_impact.py) builds the running example with a minority group, trains it with the model, clipping and accountant of [DP-SGD](02-dp-sgd.md) through a training loop that samples each group at its own rate and logs per-group gradient statistics, and computes the two-dimensional example, the quadratic model and the Clopper–Pearson intervals. From the repository root:

```bash
uv run python code/part05-dp-ml/disparate_impact.py
```

**Output** (seed 0), summarized in @tbl-fair-code:

| Experiment | Result |
| --- | --- |
| Per-group accuracy, 5% minority | Privacy cost at $\varepsilon = 4$: 0.0059 majority, 0.0934 minority; clipping alone: $-0.0051$ and 0.0404 |
| Two-dimensional clipping example | Minority's fraction of the update 0.513 → 0.116; the update turns by 39.0 degrees |
| Clipping weights during training | Minority's effective share 0.0495 at the start, 0.0251 at step 2,500, 0.0387 at the end |
| Quadratic noise floor | Formula and simulation agree; $4.07 \times 10^{-4}$ at share 0.95, $2.96 \times 10^{-2}$ at 0.01 |
| Group size at $\varepsilon = 4$ | Minority privacy cost 0.3488 at 1%, 0.0934 at 5%, 0.0280 at 20% |
| Clopper–Pearson on 100 minority records | 88 correct, $[0.800, 0.936]$, against a true accuracy of 0.8641 |
| Mitigations at $\varepsilon = 4$ | No change of $C$ helps the minority; oversampling four times needs $\sigma = 4.71$ and levels both groups down |

: The script's output with the default seed. {#tbl-fair-code}

[tests/test_disparate_impact.py](../../tests/test_disparate_impact.py) checks that the output still matches.

**What to take from it.** The disparate impact of DP-SGD needs no special data to appear: a 5% group that lives in another region of input space and follows its own rule is enough. Its two causes can be separated by switching the noise off, and each follows the arithmetic of this chapter: clipping lowers the group's effective share while the majority is learned, and the noise penalizes a small share roughly in proportion to one over the share. Neither knob of DP-SGD removes the effect at a fixed budget; it only moves it between the groups and between the two mechanisms.

## References

- Bagdasaryan, Poursaeed & Shmatikov, "Differential Privacy Has Disparate Impact on Model Accuracy" (NeurIPS 2019). The observation this chapter explains: DP-SGD's accuracy loss falls on under-represented and complex groups, and existing gaps grow.
- Tran, Dinh & Fioretto, "Differentially Private Empirical Risk Minimization under the Fairness Lens" (NeurIPS 2021). Per-step decompositions of a group's excess risk into clipping and noise terms, for output perturbation and DP-SGD, and the role of gradient norms, input norms, the loss Hessian and the distance to the decision boundary.
- Esipova, Ghomi, Luo & Cresswell, "Disparate Impact in Differential Privacy from Gradient Misalignment" (ICLR 2023). Separates clipping's direction error from its length error, finds the direction error the main cause, and proposes global scaling with an adaptive threshold.
- Xu, Du & Wu, "Removing Disparate Impact on Model Accuracy in Differentially Private Stochastic Gradient Descent" (KDD 2021). Group-dependent clipping that equalizes the groups' clipping bias.
- Feldman, "Does Learning Require Memorization? A Short Tale about a Long Tail" (STOC 2020). Why fitting small subpopulations well requires memorizing their examples, and so conflicts with privacy.
- Clopper & Pearson, "The use of confidence or fiducial limits illustrated in the case of the binomial" (Biometrika, 1934). The exact interval of @sec-fair-measure.

## Cheat sheet

Group $g$ has share $s_g$, average gradient $\bar g_g$ and clipping weight $w_g$, the mean of $\min(1, C/\|g_i\|)$ over its records; $\tau = \sigma C/B$. @tbl-fair-cheat collects the results.

| Result | Formula | Conditions |
| --- | --- | --- |
| Privacy cost of a group | $\pi_g = \mathrm{acc}_g(\text{non-private}) - \mathbb E[\mathrm{acc}_g(\text{private})]$ | Same data and recipe |
| A group's pull | $s_gm_g$, of length at most $s_gC$ | Any gradients |
| Effective share | $\tilde s_g = s_gc_g/\sum_hs_hc_h$, with $c_g = w_g$ for equal norms | Parallel gradients within each group |
| Break-even steps | $(\ell_{\text{noise}}/\ell_{\text{sig}})^2/s_g^2$ | Group's signal a fraction $s_g$ of the population's |
| Noise floor | $\mathbb E[L_g] = k\eta\tau^2/(2s_g(2 - \eta s_g\lambda)) \approx k\eta\tau^2/(4s_g)$ | Quadratic loss on the group's own $k$ weights |
| Accuracy gap interval | $\hat p_a - \hat p_b \pm 1.96\sqrt{\sum_g\hat p_g(1 - \hat p_g)/m_g}$ | Disjoint test records, large counts |
| Global scaling | Same as clipping norm $Z$, step $\eta C/Z$ | No gradient above $Z$ |
| Oversampling at $v q$ | Those records' $\varepsilon$ is the accountant's at rate $vq$ | Rate fixed in advance per group |

: Every result of the chapter. {#tbl-fair-cheat}

**Rules of thumb**

- Always report accuracy per group, with Clopper–Pearson intervals and the spread over several training runs, next to the non-private model's.
- Expect the privacy cost of a group to grow like one over its share of the data, and faster for groups with long gradients: large inputs or a rule the rest of the data does not teach.
- Clipping alone is disparate. A larger clipping norm (with $\eta C$ fixed) reduces the clipping part and raises the noise part.
- Do not upweight a group's loss to help it: clipping undoes it. Do not oversample a group without accounting at its higher sampling rate.
- When a group is too small to learn privately, get more data from it or start from a public model that already serves it.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part05-dp-ml/08-disparate-impact.md).
