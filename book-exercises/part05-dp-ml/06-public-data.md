# Exercises: Public data and private fine-tuning

Exercises for [the chapter on public data and private fine-tuning](../../book-full/part05-dp-ml/06-public-data.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part05-dp-ml/solutions_public_data.py` computes every number quoted here; run it with `uv run python code/part05-dp-ml/solutions_public_data.py`.

The solutions use the chapter's notation. Models are trained with differential privacy (DP) by stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, DP-SGD. $p$ is the number of weights, $d$ the number of trainable weights, $C$ the clipping norm, $\sigma$ the noise multiplier, $B$ the expected batch size, $s = \sigma C/B$ the noise standard deviation per coordinate on the averaged gradient, $\eta$ the learning rate and $T$ the number of steps. The running example has $n = 10{,}000$ private records, $p = 3{,}329$, $q = 0.01$, $C = 0.5$ and $T = 10{,}000$; its student is pretrained on a related public task, and the last layer has $d = 65$ weights.

## Counting the adapter's weights {#sec-sol-pub-1}

> **Exercise 1.** **Counting the adapter's weights.** A low-rank adapter of rank $r$ on a $d_{\text{out}} \times d_{\text{in}}$ matrix trains $r(d_{\text{in}} + d_{\text{out}})$ weights (@eq-pub-lora-count). For a $768 \times 768$ projection, compute the number of adapter weights, their share of the matrix's weights, and the factor by which the noise on the averaged gradient is shorter than for the full matrix at the same $\sigma$, for $r = 1$, 8 and 64. Repeat for the running example's first layer, $51 \to 64$, at $r = 4$. At which rank does the adapter stop saving anything? *Check: 1,536, 12,288 and 98,304 weights; noise 19.60, 6.93 and 2.45 times shorter; 2.66 times for the running example.* *(Counting dimensions)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Counts.** $r(768 + 768) = 1{,}536r$: 1,536, 12,288 and 98,304. The matrix has $768^2 = 589{,}824$ weights, so the shares are 0.0026, 0.0208 and 0.1667.
2. **Noise.** By @eq-pub-noise-length the noise length is $s\sqrt d$, so the ratio of full to adapter is $\sqrt{589{,}824/d}$: 19.60, 6.93 and 2.45.
3. **The running example.** $4 \times (51 + 64) = 460$ of 3,264 weights, a share of 0.1409, and a noise ratio of $\sqrt{3{,}264/460} = 2.66$. The small layer leaves less to save: the share $r(d_{\text{in}} + d_{\text{out}})/(d_{\text{in}}d_{\text{out}})$ falls with the layer's size.
4. **Break-even rank.** The adapter has fewer weights than the matrix while $r(d_{\text{in}} + d_{\text{out}}) < d_{\text{in}}d_{\text{out}}$, that is $r < d_{\text{in}}d_{\text{out}}/(d_{\text{in}} + d_{\text{out}})$: 384 for the square $768 \times 768$ matrix, and $3{,}264/115 = 28.4$ for the running example.

*The saving grows with the layer's width, which is why adapters matter most for large models. Counting weights gives the noise length; @sec-pub-dimension says the cost depends on the effective dimension, which can be much smaller for both.*
:::

## The noise cost of a finite run {#sec-sol-pub-2}

> **Exercise 2.** **The noise cost of a finite run.** In @eq-pub-noise-cost, a direction of curvature $h$ reaches the share $1 - (1 - \eta h)^{2T}$ of its steady-state noise cost after $T$ steps. (a) Compute this share for $h = 10^{-5}$, $10^{-4}$, $10^{-3}$ and $0.1$, with $\eta = 0.1$ and $T = 10{,}000$. (b) For the spectra $h_j = 0.7\,j^{-a}$, $j = 1, \dots, 3{,}329$, with $a = 1$, 1.5 and 2, compute the trace, $d_{\text{eff}}$ and the bound $2\eta T\operatorname{tr}H$ of @eq-pub-deff-bound. *Check: (a) 0.0198, 0.1813, 0.8647 and 1.0000; (b) $d_{\text{eff}} = 2{,}079$, 286 and 65.* *(Geometric sums and spectra)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Why this share.** Step 3 of the derivation of @eq-pub-noise-cost gives the variance after $T$ steps as the steady-state variance $\eta^2s^2/(1 - (1 - \eta h)^2)$ times $1 - (1 - \eta h)^{2T}$, the partial sum of a geometric series with ratio $(1 - \eta h)^2$.
2. **(a)** With $\eta T = 1{,}000$: $h = 10^{-5}$ gives $1 - (1 - 10^{-6})^{20{,}000} \approx 1 - e^{-0.02} = 0.0198$; $h = 10^{-4}$ gives $1 - e^{-0.2} = 0.1813$; $h = 10^{-3}$ gives $1 - e^{-2} = 0.8647$; $h = 0.1$ gives $1 - 0.99^{20{,}000}$, which is 1 to four decimals. The threshold $2\eta hT = 1$ sits at $h = 5 \times 10^{-4}$, between the second and the third.
3. **(b) Traces.** $0.7\sum_jj^{-a}$: 6.081 for $a = 1$ (a harmonic sum, about $0.7\ln 3{,}329$ plus a constant), 1.804 for $a = 1.5$ and 1.151 for $a = 2$ (close to $0.7\pi^2/6 = 1.151$).
4. **(b) Effective dimension.** Summing the shares: $d_{\text{eff}} = 2{,}079$, 286 and 65. Roughly, $d_{\text{eff}}$ counts the eigenvalues above $5 \times 10^{-4}$: $0.7j^{-a} > 5 \times 10^{-4}$ when $j < 1{,}400^{1/a}$, that is $j < 1{,}400$, 125 and 37, plus the partial contributions of the smaller ones.
5. **(b) The bound.** $2\eta T\operatorname{tr}H = 2{,}000\operatorname{tr}H$: 12,163, 3,609 and 2,302, all larger than $d_{\text{eff}}$ by a factor between 6 and 35, and the first two larger than $p$ itself.

*How fast the spectrum decays decides what full fine-tuning costs: with $a = 2$, all 3,329 weights cost as much noise as 65 weights would at their steady state. The trace bound is valid but loose, because it charges every direction as if it were flat.*
:::

## A private linear probe without clipping {#sec-sol-pub-3}

> **Exercise 3.** **A private linear probe without clipping.** The last layer of the running example is logistic regression on 65 frozen features, each in $(-1, 1)$ or equal to 1, so every per-record gradient has length at most $L = \sqrt{65}$ (@sec-pub-features). Train it privately by full-batch noisy gradient descent, the method of [Private empirical risk minimization](../../book-full/part05-dp-ml/01-dp-erm.md) with replace-one neighbours, per-step noise $\sigma = 2L\sqrt T/(n\mu)$ on the mean gradient and $\mu = 0.2681$ for $(\varepsilon, \delta) = (1, 10^{-5})$, learning rate 1, from the pretrained last layer, with $T = 100$ and 400 steps: once with $L = \sqrt{65}$ and no clipping, once clipping each gradient to $C = 1$. *Check: per-step noise 0.0602 and 0.0075 at $T = 100$; accuracy 0.9383 without clipping, 0.9410 with.* *(Lipschitz bound against clipping)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The bound.** $\nabla_\theta\ell = \ell'(m)y\phi(x)$ with $|\ell'| < 1$, so the gradient is shorter than $\|\phi(x)\| < \sqrt{65} = 8.062$. On the private records the largest feature norm is 7.035, so the bound is not far from attained in norm, but $|\ell'(m)|$ is small for most records.
2. **Noise without clipping.** $\sigma = 2 \times 8.062 \times \sqrt{100}/(10{,}000 \times 0.2681) = 0.0602$ per coordinate, a noise vector of length $0.0602 \times \sqrt{65} = 0.4850$ on the mean gradient. With $T = 400$, $\sqrt T$ doubles it: 0.1203 and 0.9700.
3. **Noise with clipping at 1.** The sensitivity uses $C = 1$ in place of 8.062, so the noise is 8.062 times smaller: 0.0075 at $T = 100$ (length 0.0602) and 0.0149 at $T = 400$.
4. **Accuracy.** On the report half of the private test set, the unclipped probe reaches 0.9383 at $T = 100$ and 0.9361 at $T = 400$; the clipped one 0.9410 and 0.9454. More steps help the clipped version, whose noise is small enough for the extra progress to outweigh the extra noise, and hurt the unclipped one.
5. **Why clipping wins.** Clipping at 1 changes only the gradients of records with $|\ell'(m)|\,\|\phi(x)\| > 1$, the few badly classified ones, and divides the noise by 8. The bias it introduces is smaller than the noise it removes.

*A known Lipschitz constant makes clipping unnecessary for privacy, not useless for accuracy. Both runs are below the chapter's DP-SGD probe at $\varepsilon = 1$ (0.9497): each full-batch step of noisy gradient descent uses every record, so it gets none of the amplification that Poisson sampling gives DP-SGD, and the replace-one relation doubles the sensitivity to $2L/n$.*
:::

## The adapter's first step {#sec-sol-pub-4}

> **Exercise 4.** **The adapter's first step.** Start a rank-4 adapter on the running example's first layer from $U = 0$ and a random $V$, at the pretrained weights. Take one noisy step on a batch of 100 private records with $\sigma = 4.13$, $C = 0.5$ and step $\eta/B$ applied to both factors. Show that the change $W' - W$ of the matrix has rank at most 4, and that its rows lie in the span of the columns of $V$ up to a term of relative size about $(\eta/B)\,\|\xi_V\|/\|V\|$. Check numerically for $\eta/B = 0.001$ (the chapter's learning rate 0.1 with $B = 100$), 0.003 and 1. *Check: rank 4 each time; parts outside the span of relative size 0.016, 0.041 and 0.97.* *(Low-rank updates)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The gradients at $U = 0$.** By @eq-pub-lora-grad, $\nabla_V\ell_i = a_i(U^\top\nu_i)^\top = 0$ for every record, so the noisy sum for $V$ is pure noise, $G_V = \xi_V$. The sum for $U$ is $G_U = \sum_i\overline{\nu_i(V^\top a_i)^\top} + \xi_U$, with $\nu_i$ the gradient of record $i$'s loss with respect to the layer's output.
2. **The step.** $U' = -\frac\eta BG_U$ and $V' = V - \frac\eta B\xi_V$, so, with $W = W_0$ before the step and $W' = W_0 + U'V'^\top$ after it, $W' - W = U'V'^\top = -\frac\eta BG_UV^\top + \frac{\eta^2}{B^2}G_U\xi_V^\top$.
3. **Rank.** $W' - W = U'V'^\top$ is a $64 \times 4$ matrix times a $4 \times 51$ matrix, so its rank is at most 4. Numerically it is 4 in all three cases.
4. **Span.** The first term's rows are combinations of the columns of $V$. The second is smaller by a factor of about $\frac\eta B\|\xi_V\|/\|V\|$. Here $\xi_V$ has $51 \times 4 = 204$ entries of standard deviation $\sigma C = 2.065$, so $\|\xi_V\| \approx 2.065\sqrt{204} = 29.5$, and $V$ has entries of standard deviation $1/\sqrt{51}$, so $\|V\| \approx 2$. For $\eta/B = 0.001$ the ratio is about $0.001 \times 29.5/2 = 0.015$, close to the measured 0.016; for 0.003 about 0.044 against 0.041. At $\eta/B = 1$ the second term dominates and 0.97 of the change lies outside the span.
5. **Reading it.** In realistic training, the adapter's first steps are the private gradient of $W$ projected onto the random input subspace spanned by $V$, plus noise inside that subspace. The subspace then turns, slowly, as $U$ grows and $\nabla_V$ stops being zero.

*Starting from $U = 0$ makes the model start exactly at the pretrained one, but it also means the first steps can only move $W$ within a random subspace. Initializing $V$ from public data, for example from the top principal directions of public inputs to the layer, is one way to make those first steps count.*
:::

## How related must the public data be? {#sec-sol-pub-5}

> **Exercise 5.** **How related must the public data be?** Rebuild the public task with four degrees of mismatch between the public and the private teacher: shift and rotation $(0, 0)$, $(0.15, 0.3)$ (the chapter's), $(0.5, 1.0)$ and $(1.0, \pi/2)$. For each, pretrain the student, and report the public teacher's agreement with the private labels, the pretrained student's accuracy on the private task, and the accuracy after private last-layer fine-tuning at $\varepsilon = 1$ (learning rate 0.3). Compare with training from scratch at $\varepsilon = 1$, 0.9246 (@tbl-pub-run). *Check: agreements 1.0000, 0.9004, 0.6378 and 0.4866; after fine-tuning 0.9749, 0.9507, 0.9054 and 0.8586.* *(Distribution shift)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Identical teachers.** The public task is the private task with five times more data and no privacy. The pretrained student reaches 0.9784 on the private test inputs, more than any private method in the chapter; fine-tuning its last layer privately gives 0.9749, slightly *worse*. There is nothing left to learn, and the noise can only move the model away.
2. **The chapter's mismatch.** Agreement 0.9004, pretrained 0.9035, fine-tuned 0.9507: the features are good, the readout is wrong, and the private last layer fixes most of it.
3. **Large mismatch.** At $(0.5, 1.0)$ the public labels agree with the private ones on only 0.6378 of the inputs, and the pretrained student is at 0.6334; the private probe lifts it to 0.9054, which is still below training from scratch at the same budget (0.9246). The features are partly relevant, but their ceiling is too low.
4. **Unrelated.** At $(1.0, \pi/2)$ the public teacher is close to independent of the private one (agreement 0.4866, about chance), and the probe reaches 0.8586: random-ish features of a tanh network still support a mediocre linear classifier, but pretraining has bought nothing.
5. **The pattern.** The value of public data is set by how much the private task shares with it, not by how much public data there is: the four public sets are equally large. Below some level of relatedness, a strategy that freezes the features loses to training from scratch.

*Before relying on a public model, measure how well its frozen features do on the private task's distribution, using public data that resemble it if you can, or a small private budget if you cannot. Evaluations in which public and private data come from nearly the same distribution flatter the method.*
:::

## A record in both datasets {#sec-sol-pub-6}

> **Exercise 6.** **A record in both datasets.** A fine-tuning run is $(1, 10^{-5})$-DP with respect to the private dataset. (a) Bound the true positive rate of any test of whether a given record was in the private dataset, at false positive rates 0.001 and 0.01. (b) The record appears twice in the private dataset. What does group privacy guarantee? (c) The record appears once in the private dataset and once in the public pretraining data. What is guaranteed now? *Check: (a) 0.00273 and 0.02719; (b) $(2, 5.44 \times 10^{-5})$.* *(What the guarantee covers)*

::: {.callout-tip collapse="true" title="Solution"}
1. **(a)** An $(\varepsilon, \delta)$-DP mechanism makes every test between neighbouring datasets satisfy $\text{TPR} \le e^\varepsilon\,\text{FPR} + \delta$ ([The definition](../../book-full/part03-dp-foundations/01-definition.md)). With $\varepsilon = 1$: $e \times 0.001 + 10^{-5} = 0.00273$ and $e \times 0.01 + 10^{-5} = 0.02719$.
2. **(b)** Removing both copies changes the dataset by two records. Group privacy for a group of two gives $(2\varepsilon, 2e^\varepsilon\delta) = (2, 5.44 \times 10^{-5})$ ([Post-processing, group privacy and composition](../../book-full/part03-dp-foundations/02-properties.md)): the test's power bound becomes $e^2\,\text{FPR} + 5.44 \times 10^{-5}$, weaker but finite.
3. **(c)** The guarantee compares the private dataset with and without the record, the public data held fixed (@sec-pub-caveats). Whether the record is in the public data is not something the mechanism randomizes over: the pretrained weights depend on it without noise. So nothing bounds a test that targets the public copy, such as a membership inference attack on the pretrained model. The private copy is still covered by (a).
4. **Comparison.** A duplicate inside the private dataset weakens the guarantee by a known factor; a copy in the public data removes it for that person, as far as the theorem goes.

*Deduplicate the private dataset against the public data before fine-tuning, and when that is not possible, do not describe the people in the overlap as protected.*
:::
