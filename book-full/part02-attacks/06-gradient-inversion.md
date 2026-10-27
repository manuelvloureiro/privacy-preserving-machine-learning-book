# Gradient inversion

A phone keyboard learns from what its users type, but the raw text never leaves the phones. A server sends the current model $\theta$ to each phone, the phone computes the gradient of the loss on its own few examples, and only that gradient goes back. The server averages the gradients of many phones and takes a step. This is the setting of **federated learning (FL)**, and it is often described as private because "the data stay on the device". This chapter asks one question about it: **what does a single client's gradient reveal about the client's data?**

The answer is often everything, and one idea explains why. A gradient is not a summary chosen to hide the data. In a fully connected layer, each row of the weight gradient is the layer's input multiplied by a single number, the rate at which the loss changes with that row's output. The bias gradient holds the same numbers on their own. So with one example, dividing a row of the weight gradient by the matching entry of the bias gradient gives back the input exactly. With a batch, the gradient is a sum of such rows, one per example, and the attacker's work is to pull the sum apart. The chapter builds its attacks in four steps. One division recovers the input of one example, and the signs of the last layer's gradient give its label. When no division applies, **Deep Leakage from Gradients (DLG)** searches for an input whose gradient matches the one observed. Last, the chapter asks which of these attacks survive when the client averages over a batch.

The running example comes in two sizes. A network with three inputs and two hidden units is small enough to compute by hand, and the division gives back its input $(0.5, 1, 0.2)$ exactly, once for each hidden unit. The experiments then use synthetic $8\times8$ images and a network with 32 hidden units, whose gradient has 2,212 entries against 64 unknown pixels. There the division recovers an image to rounding error, DLG recovers one to about 13 digits after 600 steps, and the same division, applied unit by unit, still recovers all four images of a batch of 4.

| Symbol | Meaning |
| --- | --- |
| $x \in \mathbb R^d$; $y$ | an input with $d$ pixels; its label, one of $K$ classes |
| $W_1, b_1$; $W_2, b_2$ | the weights and biases of the first and the second layer |
| $a$, $h$, $z$, $p$ | pre-activations, hidden activations, logits and predicted probabilities |
| $\sigma$; $m$ | the activation function; the number of hidden units |
| $\ell$ | the cross-entropy loss of one example |
| $\gamma_1$, $\gamma_2$ | the backward signals $\partial\ell/\partial a$ and $\partial\ell/\partial z$ |
| $e_y$; $\odot$; $A_{j,:}$ | the one-hot vector of the label; the entrywise product; row $j$ of a matrix $A$ |
| $B$; $g$ | the batch size; the gradient the client sends |
| $R$ | the matching loss of Deep Leakage from Gradients |

: The symbols of this chapter. {#tbl-gradinv-symbols}

## The reconstruction game {#sec-gradinv-game}

Before any attack can be judged, you need to know who sees what and what counts as success. This section fixes both for one round of federated training, in the vocabulary of [Threat models and privacy games](01-threat-models.md), and then fixes the network that every experiment uses.

The server is **honest-but-curious** in the sense of that chapter: it runs the protocol as specified and only analyses what it receives. Its access is the **transcript**: it sees each client's own update, and it knows the architecture and every parameter of $\theta$ because it chose them. The derivative of an average is the average of the derivatives, so the client's gradient is the average of its per-example gradients.

::: {#def-gradinv-game name="The gradient reconstruction game"}
Fix a model $\theta$, known to the server, and a population of labelled examples. The challenger plays the client, and the adversary is the server. The game has three steps.

1. The challenger draws a batch of $B$ labelled examples $(x_1, y_1), \dots, (x_B, y_B)$ from the population. The inputs are the secret.
2. The client sends the gradient of its average loss with respect to all the parameters,
$$g = \frac1B\sum_{i=1}^B\nabla_\theta\ell(\theta; x_i, y_i).$$
3. The adversary sees $(\theta, g)$ and outputs reconstructions $\hat x_1, \dots, \hat x_B$.

Each true input $x$ is matched to its closest reconstruction $\hat x$ (two inputs may share one), and the pair is scored by its **relative error** $\|\hat x - x\|/\|x\|$.
:::

The relative error is 0 for a perfect reconstruction and 1 for a mistake as large as the input itself. The matching is needed because an average does not change when you reorder the examples, so the gradient carries no record of which example came first. The baseline is a guess from the population alone, such as the average input of the population. The attacks below either recover the client's exact pixels or, for large batches, produce blends of them.

All experiments use one small network, written in NumPy. An input first goes through a fully connected layer that gives one number per hidden unit. An activation is applied to each unit, and a second fully connected layer turns the result into one score per class, called a **logit**. The **softmax** turns the logits into probabilities by exponentiating and normalizing, so that every probability is positive and they sum to 1. The loss is the **cross-entropy**, minus the log of the probability the model gives to the true label, which is small when the model is confidently right.

::: {#def-gradinv-network name="The network"}
An input $x \in \mathbb R^d$ has a label $y$, one of $K$ classes. It passes through a first layer with weights $W_1 \in \mathbb R^{m\times d}$ and biases $b_1$, one per hidden unit. An activation $\sigma$ is applied to each of the $m$ hidden units. A second layer with weights $W_2 \in \mathbb R^{K\times m}$ and biases $b_2$ gives one logit per class, and then
$$a = W_1x + b_1, \quad h = \sigma(a), \quad z = W_2h + b_2, \quad p = \mathrm{softmax}(z), \quad \ell = -\ln p_y,$$
where the softmax has entries
$$p_k = \frac{e^{z_k}}{\sum_{k'}e^{z_{k'}}}.$$
The **backward signals** are the rates at which the loss changes with each hidden unit's pre-activation and with each logit, $\gamma_1 = \partial\ell/\partial a$ and $\gamma_2 = \partial\ell/\partial z$.
:::

The activation $\sigma$ is either the ReLU $\max(a, 0)$, which passes positive values and replaces negative ones by 0, or the sigmoid $1/(1 + e^{-a})$, which squashes every value smoothly into $(0, 1)$. Texts on backpropagation usually write the backward signals as $\delta$, but this book keeps $\delta$ for the privacy parameter of the later parts.

::: {#exm-gradinv-images}
The client's inputs are synthetic $8\times8$ grey-scale images, so $d = 64$. Each shows one of $K = 4$ shapes (a horizontal bar, a vertical bar, a diagonal and a square outline), shifted at random and with pixel noise. The network has $m = 32$ hidden units and is at its random initialization, as in the first round of training. The first layer has $64\cdot32$ weights and 32 biases, and the second has $32\cdot4$ weights and 4 biases, so the network has
$$64\cdot32 + 32 + 32\cdot4 + 4 = 2{,}212$$
parameters. The server therefore receives 2,212 numbers about an example that has only 64 unknown pixels.
:::

## What a gradient is made of {#sec-gradinv-gradient}

Every attack in this chapter reads the input out of the gradient, so you first need to know where in the gradient the input sits. This section derives the gradient in two results. The first shows that in a fully connected layer the input appears in every row of the weight gradient. The second gives the backward signals that scale those rows.

The reason the input appears is the chain rule. The weight $W_{jk}$ multiplies the input $x_k$ and feeds only the output $a_j$, so the loss changes with it at the rate at which the loss changes with $a_j$, times $x_k$. That is what flows backward times what flowed forward.

::: {#prp-gradinv-outer name="The gradient of a fully connected layer is an outer product"}
Let a fully connected layer compute $a = Wx + b$ with $W \in \mathbb R^{m\times d}$, and let the loss $\ell$ depend on $W$ and $b$ only through $a$, with backward signal $\gamma = \partial\ell/\partial a$. Then
$$\nabla_W\ell = \gamma\,x^\top, \qquad \nabla_b\ell = \gamma.$$ {#eq-gradinv-outer}
:::

::: {.proof}
1. *Each weight feeds one output.* Written out, output $j$ is $a_j = \sum_kW_{jk}x_k + b_j$. The weight $W_{jk}$ appears in $a_j$ only, with coefficient $x_k$, so $\partial a_j/\partial W_{jk} = x_k$.
2. *The chain rule multiplies the two rates.* The loss depends on $W_{jk}$ only through $a_j$, so
$$\frac{\partial\ell}{\partial W_{jk}} = \frac{\partial\ell}{\partial a_j}\,\frac{\partial a_j}{\partial W_{jk}} = \gamma_jx_k,$$
which is the $(j, k)$ entry of $\gamma x^\top$.
3. *The bias has coefficient 1.* The bias $b_j$ enters $a_j$ with coefficient 1, so the same rule gives $\partial\ell/\partial b_j = \gamma_j$.
:::

A matrix whose $(j, k)$ entry is $\gamma_jx_k$ is called the outer product of $\gamma$ and $x$, and its row $j$ is the input multiplied by the number $\gamma_j$. The signal $\gamma$ depends on everything above the layer (the rest of the network, the loss, the label), but the factor $x^\top$ does not. So the rest of the network can only rescale the rows, never change their direction.

To use @prp-gradinv-outer at the first layer you need its backward signal $\gamma_1$, and the chain rule obtains it by sending the last layer's signal $\gamma_2$ back through the network. At the last layer the signal has a simple form. Raising the logit of a wrong class $k$ raises the loss at the rate $p_k$, the probability the model already gives that class, and raising the logit of the true class lowers the loss at the rate $1 - p_y$. Each hidden unit then collects the logits' signals, weighted by its connections to them, and multiplies by the slope of its activation.

::: {#prp-gradinv-backward name="The backward signals"}
For the network of @def-gradinv-network, write $e_y$ for the one-hot vector of the label, with 1 in position $y$ and 0 elsewhere, and $\odot$ for the entrywise product of two vectors. Then
$$\gamma_2 = p - e_y,$$ {#eq-gradinv-softmax}
so the entry of $\gamma_2$ at the true label is negative and every other entry is positive, and
$$\gamma_1 = (W_2^\top\gamma_2)\odot\sigma'(a).$$ {#eq-gradinv-backprop}
:::

::: {.proof}
The plan is to differentiate the loss in the logits, read off the signs, and then pass the result back through the second layer and the activation.

1. *Remove the fraction.* Since $p_y = e^{z_y}/\sum_ke^{z_k}$, taking minus its log gives
$$\ell = -z_y + \ln\sum_ke^{z_k}.$$
2. *Differentiate in one logit.* The first term depends only on $z_y$, so it contributes $-1$ when $k = y$ and 0 otherwise. The second term is the log of a sum, whose derivative is the derivative of the sum, $e^{z_k}$, divided by the sum itself, which is $p_k$. Together,
$$\frac{\partial\ell}{\partial z_k} = p_k - \mathbf 1[k = y],$$
which is entry $k$ of $p - e_y$.
3. *Read off the signs.* Every $e^{z_k}$ is positive, so every $p_k$ lies strictly between 0 and 1. The entry at the label is $p_y - 1 < 0$, and every other entry is $p_k > 0$.
4. *Pass back through the second layer.* Logit $k$ is $z_k = \sum_jW_{2,kj}h_j + b_{2,k}$, so $h_j$ enters each logit with coefficient $W_{2,kj}$, and the chain rule sums over the logits:
$$\frac{\partial\ell}{\partial h_j} = \sum_k\gamma_{2,k}W_{2,kj} = (W_2^\top\gamma_2)_j.$$
5. *Pass through the activation.* The output $h_j = \sigma(a_j)$ changes with $a_j$ at the rate $\sigma'(a_j)$, so multiplying step 4 by it gives entry $j$ of @eq-gradinv-backprop.
:::

For ReLU the slope is $\sigma'(a_j) = \mathbf 1[a_j > 0]$, which is 1 if the unit is switched on and 0 if it is switched off. So a unit that is switched off on an input has $\gamma_{1,j} = 0$ and, by @prp-gradinv-outer, a zero row in the first-layer weight gradient. The sign pattern of $\gamma_2$ is what label recovery uses in @sec-gradinv-labels.

::: {#exm-gradinv-tiny}
Take $d = 3$ inputs, $m = 2$ ReLU units and $K = 2$ classes, with
$$W_1 = \begin{pmatrix}0.4 & -0.2 & 0.1\\ 0.3 & 0.5 & -0.4\end{pmatrix},\quad b_1 = \begin{pmatrix}0.1\\ -0.1\end{pmatrix},\quad W_2 = \begin{pmatrix}1 & -1\\ 0.5 & 0.5\end{pmatrix},\quad b_2 = 0,$$
and the example $x = (0.5, 1, 0.2)$ with label $y = 1$, the first class. @tbl-gradinv-tiny walks through the forward pass, the backward pass and the gradient.

| Step | Quantity | Value |
| --- | --- | --- |
| Forward | $a = W_1x + b_1$ | $(0.2 - 0.2 + 0.02 + 0.1,\ 0.15 + 0.5 - 0.08 - 0.1) = (0.12, 0.47)$ |
| | $h = \max(a, 0)$ | $(0.12, 0.47)$, both units on |
| | $z = W_2h$ | $(0.12 - 0.47,\ 0.06 + 0.235) = (-0.35, 0.295)$ |
| | $p = \mathrm{softmax}(z)$ | $(0.3441, 0.6559)$ |
| Backward | $\gamma_2 = p - e_y$ | $(-0.6559, 0.6559)$ |
| | $\gamma_1 = (W_2^\top\gamma_2)\odot\sigma'(a)$ | $(1\cdot(-0.6559) + 0.5\cdot0.6559,\ (-1)\cdot(-0.6559) + 0.5\cdot0.6559) = (-0.3279, 0.9838)$ |
| Gradient | $\nabla_{W_1}\ell = \gamma_1x^\top$ | rows $(-0.1640, -0.3279, -0.0656)$ and $(0.4919, 0.9838, 0.1968)$ |
| | $\nabla_{b_1}\ell = \gamma_1$ | $(-0.3279, 0.9838)$ |

: Forward pass, backward pass and gradient of a two-unit ReLU network, by @prp-gradinv-backward and @prp-gradinv-outer. The server sees only the "Gradient" rows. {#tbl-gradinv-tiny}

The first probability is $p_1 = e^{-0.35}/(e^{-0.35} + e^{0.295}) = 1/(1 + e^{0.645}) = 1/(1 + 1.906) = 0.3441$, so $p = (0.3441, 0.6559)$. The two entries of $\gamma_2$ sum to 0, and only the one at the true class is negative. Both units are on, since both entries of $a$ are positive, so $\gamma_1$ is just $W_2^\top\gamma_2$, and the two rows of the weight gradient are $-0.3279$ times $x$ and $0.9838$ times $x$.
:::

## Exact recovery from a fully connected layer {#sec-gradinv-exact}

The server sees $g$. Is there any computation, short of a search, that gives back the client's input? For the most common first layer of all, a fully connected layer with a bias, there is: one division. The observation goes back at least to Phong et al. (2018), who pointed it out for exactly this federated setting.

The division works because of @prp-gradinv-outer. Row $j$ of the first-layer weight gradient is the input $x$ scaled by the number $\gamma_{1,j}$, and entry $j$ of the bias gradient is that same number, so dividing the first by the second removes the scale.

::: {#thm-gradinv-exact name="Exact recovery from one example"}
Suppose the client sends the gradient of a single example, and the first layer is fully connected with a bias, $a = W_1x + b_1$, whatever the layers above it and the loss. Then for every unit $j$ with $(\nabla_{b_1}\ell)_j \ne 0$,
$$x = \frac{(\nabla_{W_1}\ell)_{j,:}}{(\nabla_{b_1}\ell)_j}.$$ {#eq-gradinv-ratio}
Moreover, if $\gamma_1 \ne 0$, the matrix $\nabla_{W_1}\ell$ has **rank one**, and its row space is the line through $x$.
:::

::: {.proof}
1. *Write out what the server sees.* @prp-gradinv-outer applied to the first layer gives $(\nabla_{W_1}\ell)_{j,:} = \gamma_{1,j}\,x^\top$ and $(\nabla_{b_1}\ell)_j = \gamma_{1,j}$.
2. *Divide one by the other.* When $\gamma_{1,j} \ne 0$, the ratio is $\gamma_{1,j}x^\top/\gamma_{1,j} = x^\top$. Nothing about the rest of the network enters, because the depth, the activations above, the loss and the label all sit inside the number $\gamma_{1,j}$, and that number cancels.
3. *Read off the rank.* All rows of $\nabla_{W_1}\ell = \gamma_1x^\top$ are multiples of the same vector $x^\top$, and at least one is nonzero when $\gamma_1 \ne 0$. A matrix whose rows are all multiples of one nonzero vector has rank one, and its row space is spanned by that vector.
:::

The attack is exact and needs no optimization, no prior knowledge of the data and no label. It also checks itself, because every usable unit gives the same answer, so agreement between the ratios tells the attacker it has the input. Deeper in the network, the same division gives an intermediate feature vector.

Four things get in its way, and only one stops it fully. With ReLU, switched-off units give nothing, but one active unit is enough. Without a bias there is no denominator, but the rank-one structure still gives the direction of $x$, so only the scale remains unknown, one number that matching gradients as in @sec-gradinv-dlg finds. A convolutional first layer really breaks the outer-product form. It applies the same small filter at every position of the image, so its gradient is a sum of outer products over all the image patches, and only the search of @sec-gradinv-dlg can undo it. A batch of more than one example also breaks the formula, but only partly, as @sec-gradinv-batch shows.

![Exact recovery from the first-layer gradient of a ReLU network with 32 hidden units. Left: the client's $8\times8$ input. Middle: $\nabla_{W_1}\ell$, one row per hidden unit and one column per pixel (blue positive, red negative), next to $\nabla_{b_1}\ell$. Every nonzero row is the image times the matching bias entry; switched-off units give zero rows. Right: one row divided by its bias entry, which is the input exactly.](figures/gradinv_rank_one.svg){#fig-gradinv-rank-one}

::: {#exm-gradinv-exact}
In the hand network of @exm-gradinv-tiny, the server never sees $a$, $h$, $z$ or $p$. It sees only the gradient rows of @tbl-gradinv-tiny. Dividing the first row by $-0.3279$ and the second by $0.9838$ gives $(0.5, 1, 0.2)$ both times, which is the input.

On the images, @fig-gradinv-rank-one shows the attack on one input. Of the 32 ReLU units, 13 are switched on and have a nonzero bias gradient. Every one of the 13 ratios equals the image to a relative error of at most $6.7\times10^{-17}$, which is rounding error. To check the rank, look at the **singular values** of $\nabla_{W_1}\ell$. They measure how much of the matrix lies along each of its independent directions, so a rank-one matrix has exactly one that is not zero. Here the largest is $1.6155$, and the second is $9.5\times10^{-17}$ times the first. The gradient is rank one, as @thm-gradinv-exact says.
:::

## Label recovery from the last layer {#sec-gradinv-labels}

The gradient depends on the label too, and the label is often the most sensitive part: the diagnosis, the word typed next, the class of a photo. Zhao, Mopuri & Bilen (2020) noticed that the label can be read off the last layer in closed form. Their attack, iDLG ("improved DLG"), then searches only for the input, which makes the search of @sec-gradinv-dlg easier and more reliable.

All three rules below come from @prp-gradinv-backward: the last layer's backward signal is a vector of probabilities with 1 subtracted at the true label. A batch gradient averages these vectors, so the subtracted 1s add up, before the division by the batch size, to the number of examples of each class.

::: {#prp-gradinv-labels name="Labels from the last layer"}
Let the network of @def-gradinv-network keep its last-layer bias $b_2$. For a single example, the label is the position of the only negative entry of the last-layer bias gradient,
$$y = \arg\min_k(\nabla_{b_2}\ell)_k.$$ {#eq-gradinv-labels}
If in addition the hidden activations satisfy $h \ge 0$ and $h \ne 0$, as for ReLU or the sigmoid, then row $y$ of $\nabla_{W_2}\ell$ is the only row whose entries are all non-positive, and the only row with a negative sum.

For a batch of $B$ examples, write $n_k$ for the number of examples of class $k$ and $p_{ik}$ for the probability the model gives class $k$ on example $i$. Then
$$n_k = \sum_{i=1}^Bp_{ik} - B\,(\nabla_{b_2}\ell)_k.$$ {#eq-gradinv-counts}
:::

::: {.proof}
1. *The bias rule.* @prp-gradinv-outer at the last layer gives $\nabla_{b_2}\ell = \gamma_2$, and @prp-gradinv-backward gives $\gamma_2 = p - e_y$, with entry $p_y - 1 < 0$ at the label and $p_k > 0$ elsewhere. The smallest entry is therefore at $y$.
2. *The weight rule.* @prp-gradinv-outer gives $(\nabla_{W_2}\ell)_{k,:} = \gamma_{2,k}\,h^\top$. With $h \ge 0$, row $y$ is a negative number times non-negative entries, so it is non-positive entrywise, and every other row is $p_kh^\top \ge 0$. Summing a row multiplies $\gamma_{2,k}$ by $\sum_jh_j$, which is positive because $h \ge 0$ and $h \ne 0$, so the sum keeps the sign of $\gamma_{2,k}$.
3. *The counts.* The batch gradient is the average of the per-example ones, so $\nabla_{b_2}\ell = \frac1B\sum_i(p_i - e_{y_i})$. The vector $e_{y_i}$ has a 1 in position $k$ exactly when example $i$ has class $k$, so $\sum_ie_{y_i}$ counts the examples of each class, and entry $k$ is
$$(\nabla_{b_2}\ell)_k = \frac1B\Big(\sum_ip_{ik} - n_k\Big).$$
Multiplying by $B$ and solving for $n_k$ gives @eq-gradinv-counts.
:::

For one example the bias rule makes no errors, at any point of training, because the sign argument never depends on how good the model is. A confident model gives a tiny negative entry, which floating point hides only when $p_y$ rounds to 1 and the entry becomes 0. That is one reason attacks are run on models early in training. Without a bias, and even for an activation such as tanh whose outputs can be negative, step 2 of the proof still shows that row $y$ of $\nabla_{W_2}\ell$ is a negative multiple of $h$ and every other row a positive one, so the label's row points the opposite way.

For a batch, the gradient leaks the label histogram. @eq-gradinv-counts is exact, so the only unknown is the total prediction $\sum_ip_{ik}$, and the attacker can estimate it well. If it holds data of its own from a similar population, which this attack assumes, it can run the model on them and average the predictions into a guess $\bar p$, then replace the total by $B\bar p_k$ and round. The true counts are integers, so rounding gives them exactly whenever the estimate of the total is off by less than $\frac12$ for every class. The cruder guess $\bar p_k = 1/K$ of evenly spread predictions does worse, because even an untrained network favours some classes over others.

![Labels from the last-layer bias gradient. Left: for one example, $\nabla_{b_2}\ell = p - e_y$ has a single negative entry, at the true label (red). Right: for a batch of 16, the counts $B(\bar p - \nabla_{b_2}\ell)$ with $\bar p$ the model's average prediction on the attacker's own images (green) are close to the true counts (grey); with $\bar p = 1/K$ (orange) they are off by up to about half an example. The batch is a different random draw from the one in @tbl-gradinv-counts.](figures/gradinv_labels.svg){#fig-gradinv-labels}

::: {#exm-gradinv-labels}
In the hand network of @exm-gradinv-tiny, the last-layer bias gradient is $\gamma_2 = (-0.6559, 0.6559)$. Its smallest entry is the first, so the server reads off the label 1. On 1,000 random images and random networks, both the bias rule and the weight rule name the label correctly in a fraction 1.000 of the trials.

For a batch, take 16 images with true counts $(4, 2, 4, 6)$ over the four shapes. @tbl-gradinv-counts reads them off the bias gradient of the 32-unit ReLU network, with three guesses for the total prediction in @eq-gradinv-counts.

| Guess of $\sum_ip_{ik}$ | Estimated counts | Rounded | All correct? |
| --- | --- | --- | --- |
| Exact, $\sum_ip_{ik}$ (not available to the attacker) | $(4.00, 2.00, 4.00, 6.00)$ | $(4, 2, 4, 6)$ | yes |
| $B\bar p$, $\bar p$ averaged over 200 of the attacker's own images | $(3.88, 2.00, 4.19, 5.92)$ | $(4, 2, 4, 6)$ | yes |
| $B/K = 4$ for every class | $(2.12, 3.01, 4.27, 6.60)$ | $(2, 3, 4, 7)$ | no |

: Label counts of one batch of 16 recovered from $\nabla_{b_2}\ell$ by @eq-gradinv-counts, under three guesses of the summed predictions. {#tbl-gradinv-counts}

With the averaged guess every estimate is within $\frac12$ of the truth, the largest miss being $4.19 - 4 = 0.19$, so rounding fixes it. With the uniform guess the first class is off by $4 - 2.12 = 1.88$, and rounding cannot help.

Over 200 random batches and networks, the averaged guess gets all the rounded counts right in 1.000 of batches of 4 and 0.975 of batches of 16. For batches of 64 the rate falls to 0.525, and with the uniform guess the three rates are 0.925, 0.095 and 0.000. The rates fall because the error of the estimate is a sum of $B$ deviations, one per example, and that sum grows with $B$. At $B = 16$ the largest error over the four classes, before rounding, has a median over batches of 0.233 with the averaged guess, against 1.102 with the uniform one.
:::

## Deep Leakage from Gradients {#sec-gradinv-dlg}

The closed forms need a fully connected layer with a bias. Most image models start with convolutions, and many architectures have normalization layers, attention, or no biases at all. Zhu, Liu & Han (2019) gave a generic attack that needs none of these: **guess an input, compute its gradient, and move the guess until the gradients match**. Zhu et al. also searched for a soft label, a probability vector standing in for the unknown label, and that joint search often failed. With the label fixed by @eq-gradinv-labels, the search is over the input only.

To turn "move the guess until the gradients match" into something you can optimize, you need one number that says how badly a guess matches. DLG uses the squared distance between the gradient the guessed batch would produce and the observed gradient, summed over all the gradient's entries. It is zero for a perfect match and positive otherwise.

::: {#def-gradinv-dlg name="Deep Leakage from Gradients"}
Given the model $\theta$, the observed gradient $g$ and the labels $y_1, \dots, y_B$, write $G(x') = \frac1B\sum_i\nabla_\theta\ell(\theta; x_i', y_i)$ for the gradient that a guessed batch $x' = (x_1', \dots, x_B')$ would produce. The **matching loss** is $R(x') = \|G(x') - g\|^2$, and DLG outputs
$$\hat x_1, \dots, \hat x_B = \arg\min_{x_1', \dots, x_B'}R(x'), \qquad R(x') = \Big\|\frac1B\sum_{i=1}^B\nabla_\theta\ell(\theta; x_i', y_i) - g\Big\|^2,$$ {#eq-gradinv-dlg}
searching by a gradient method that starts from random noise and repeatedly moves the guess a little downhill on $R$.
:::

For a batch, the label rules of @prp-gradinv-labels give only the counts of each class, not which example has which label. That is enough, because the guesses are interchangeable: the attacker hands out the recovered labels to the guesses in any order, and the search fits each guess to its label.

Two facts make this a sensible search. The true batch scores the best value the matching loss can take, and the slope of the matching loss can be computed, as long as the network can be differentiated twice. The second fact needs the second derivatives because $G$ is itself made of first derivatives of the network.

::: {#prp-gradinv-dlg name="The true batch is a global minimizer"}
The true batch $(x_1, \dots, x_B)$ has $R = 0$, the smallest value $R$ takes. If the network is twice differentiable, then $R$ is differentiable in the guess, with
$$\nabla_{x'}R = 2\,J_G(x')^\top\big(G(x') - g\big),$$ {#eq-gradinv-dlg-slope}
where $J_G$ is the **Jacobian** of $G$, the matrix of rates at which each gradient entry changes with each input pixel.
:::

::: {.proof}
1. *The true batch is a global minimizer.* $R$ is a squared length, so $R \ge 0$ for every guess. Plugging in the true inputs reproduces $g$ exactly, so $R = 0$ there, and no guess can do better.
2. *The slope of a squared length.* A squared length $\|u\|^2$ changes with $u$ at the rate $2u$. Here $u = G(x') - g$, and $g$ does not depend on the guess.
3. *The chain rule passes the slope back to the guess.* $u$ changes with $x'$ at the rates collected in $J_G(x')$, so the chain rule multiplies the transpose of that matrix by $2u$, which is @eq-gradinv-dlg-slope. The entries of $J_G$ are derivatives of the gradient, so they are second derivatives of the network, and they exist because the network is twice differentiable.
:::

The need for second derivatives is why Zhu et al. replaced ReLU by the sigmoid, and why the experiments here do the same. The ReLU's slope is 0 or 1, constant on each side of zero, so its own derivative is zero almost everywhere. The factor $\sigma'(a)$ in @eq-gradinv-backprop then does not change as the guess moves, and the search gets no signal through it. The sigmoid's slope varies smoothly, so it does. No one builds $J_G$, which has one row per parameter and one column per pixel. Automatic-differentiation libraries instead run backpropagation through the backward pass itself, which is called **double backpropagation**. That is all DLG needs, so it works for convolutional networks and on real images: Zhu et al. recover pictures from the CIFAR-100 image dataset and text tokens from single-example gradients.

@prp-gradinv-dlg does not say that the search finds the true batch. $R$ is not convex in the guess, and for such a function [Optimization](../part01-prerequisites/06-optimization.md) shows that a gradient method with a suitable step only drives the slope to zero. That gives a **stationary point**, where the surface is flat, which may or may not be the global minimum, so success is an empirical question. The final value of $R$ tells you which case you are in: $R = 0$ up to rounding means an exact match, while a value stuck well above zero means the search is stuck, and restarts or image priors may help.

The minimizer is also plausibly unique, since matching $g$ imposes one equation per parameter, 2,212 here against $64B$ unknown pixels. That count is not a proof. For a single example, though, the sigmoid network still has a fully connected first layer with a bias, so @thm-gradinv-exact shows that its gradient determines the input. Geiping et al. (2020) compared gradients by a cosine distance and penalized differences between neighbouring pixels, and so recovered recognizable images from trained networks and from batches.

![Deep Leakage from Gradients on one $8\times8$ image, sigmoid network with 32 hidden units. Top: the reconstruction $\hat x$ at selected iterations of Adam, starting from uniform noise, and the true input. Bottom: the matching loss $R(\hat x)$ and the relative error, on a log scale. Both fall geometrically until they reach rounding error.](figures/gradinv_dlg_progress.svg){#fig-gradinv-dlg-progress}

::: {#exm-gradinv-dlg}
The chapter's script (@sec-gradinv-code) runs DLG on one image with the sigmoid network of 32 hidden units, computing the slope of @eq-gradinv-dlg-slope by double backpropagation written out by hand. A finite-difference check, which nudges one pixel slightly and sees how much $R$ changes, agrees with it to within $1.2\times10^{-10}$. The search starts from uniform noise and runs 600 steps of the Adam optimizer, a variant of gradient descent that scales each pixel's step by a running estimate of the size of that pixel's gradient, with learning rate 0.05 decaying linearly to 0. The label comes from @eq-gradinv-labels.

| Iteration | 0 | 10 | 50 | 100 | 200 | 400 | 600 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Relative error | $1.6$ | $0.67$ | $7.5\times10^{-2}$ | $5.7\times10^{-3}$ | $3.4\times10^{-5}$ | $7.8\times10^{-10}$ | $5.8\times10^{-14}$ |
| Matching loss $R$ | $1.3$ | $0.20$ | $2.5\times10^{-3}$ | $1.3\times10^{-5}$ | $4.6\times10^{-10}$ | $2.5\times10^{-19}$ | $1.3\times10^{-27}$ |

: Deep Leakage from Gradients on one image, sigmoid network with 32 hidden units: relative error of the reconstruction and the matching loss over the iterations. {#tbl-gradinv-dlg}

By iteration 50 the image is visually correct, with 7.5% error, and by iteration 600 it is exact to about 13 digits (@fig-gradinv-dlg-progress). The matching loss falls about as the square of the error, which is what you should expect close to the answer. Near the true input $x$ the map $G$ is nearly linear, so the mismatch is about the Jacobian times the error, and its squared length is quadratic in the error:
$$R(x') \approx \|J_G(x)(x' - x)\|^2.$$
In the table, between iterations 50 and 100 the error shrinks by a factor $0.075/0.0057 \approx 13$ and $R$ by $0.0025/0.000013 \approx 190$, close to $13^2 \approx 170$.
:::

## Batches and averaging {#sec-gradinv-batch}

Clients rarely send the gradient of one example. They send the average over a batch, or the change in parameters after several local steps. Averaging is often said to protect the individual examples, and this section measures that claim. Averaging does destroy the one-line formula of @eq-gradinv-ratio, and it does make DLG harder, but the span of the batch survives, and so does every example that some unit "sees alone".

The structure of a batch gradient follows from @prp-gradinv-outer. Each example contributes one outer product, whose rows are multiples of its own input, so every row of the sum is a weighted combination of the $B$ inputs.

::: {#prp-gradinv-batch name="The gradient of a batch"}
Take a batch of $B$ examples and a fully connected first layer, and write $\gamma_{1,i}$ for the backward signal of example $i$ at that layer, with entries $\gamma_{1,ij}$, one per unit $j$. Then
$$\nabla_{W_1}\ell = \frac1B\sum_{i=1}^B\gamma_{1,i}\,x_i^\top, \qquad \operatorname{rank}\nabla_{W_1}\ell \le B, \qquad \text{every row} \in \operatorname{span}\{x_1, \dots, x_B\}.$$ {#eq-gradinv-batch}
For every unit $j$ with $(\nabla_{b_1}\ell)_j \ne 0$, the unit ratio is a weighted combination of the inputs,
$$\frac{(\nabla_{W_1}\ell)_{j,:}}{(\nabla_{b_1}\ell)_j} = \sum_{i=1}^B\frac{\gamma_{1,ij}}{\sum_{i'}\gamma_{1,i'j}}\,x_i^\top,$$ {#eq-gradinv-batch-ratio}
with weights that sum to 1 (some may be negative). It equals $x_i$ exactly when $\gamma_{1,ij} \ne 0$ and $\gamma_{1,i'j} = 0$ for every $i' \ne i$.
:::

::: {.proof}
1. *A sum of outer products.* The batch gradient is the average of the per-example gradients, and by @prp-gradinv-outer each of those is $\gamma_{1,i}x_i^\top$.
2. *Rank and span.* Stack the $x_i^\top$ as the rows of a matrix $X$ with $B$ rows, and the $\gamma_{1,i}^\top$ as the rows of a matrix $\Gamma$. The sum of outer products is then the product $\frac1B\Gamma^\top X$. A product has rank at most that of either factor, and $X$ has only $B$ rows, so the rank is at most $B$. Row $j$ of the product is $\frac1B\sum_i\Gamma_{ij}x_i^\top$, a combination of the inputs, so it lies in their span.
3. *The ratios.* By @prp-gradinv-outer again, entry $j$ of the bias gradient is $\frac1B\sum_i\gamma_{1,ij}$. Dividing row $j$ by it cancels the factors $\frac1B$ and gives @eq-gradinv-batch-ratio. The weights are each example's share of the same total, so they sum to 1, and if only $\gamma_{1,ij}$ is nonzero the weight on $x_i$ is 1 and all the others are 0.
:::

For ReLU the condition of the last sentence has a simple form. By @prp-gradinv-backward, $\gamma_{1,i'j} = 0$ whenever unit $j$ is switched off on $x_{i'}$, so a unit that is switched on for example $i$ and for no other example gives back $x_i$ exactly. Such a unit **isolates** example $i$, and @sec-gradinv-isolation asks how often that happens. The span leaks even when nothing is isolated. When the gradient has rank $B$, as it does for the batch of 4 below, its rows span exactly the span of the inputs. Every example is then a combination of $B$ known directions, so a further search such as DLG has $B$ unknowns per example instead of $d$: 4 instead of 64 here, and 4 instead of hundreds of thousands for real images.

The span leaks even when nothing is isolated. When the gradient has rank $B$, as it does for the batch of 4 below, its rows span exactly the span of the inputs. Every example is then a combination of $B$ directions the attacker can read off the gradient, so a search such as DLG can look for each example as $B$ unknown coefficients of those directions instead of $d$ unknown pixels. That is 4 unknowns instead of 64 here, or of hundreds of thousands for real images.

![Deep Leakage from Gradients against batch size, sigmoid network with 32 hidden units. Left: relative error of the closest reconstruction for every example (dots) and the median (red), on a log scale; the dotted line marks an error of 0.05. Right: a batch of 4, true images (top) and their closest reconstructions (bottom). Some examples are recovered and others are blends.](figures/gradinv_dlg_batch.svg){#fig-gradinv-dlg-batch}

::: {#exm-gradinv-batch}
A batch of 4 images through the 32-unit ReLU network gives a first-layer gradient whose singular values, relative to the largest, are $1.0$, $0.84$, $0.69$, $0.039$ and then $1.5\times10^{-16}$. Four directions are nonzero and the rest are rounding error, so the rank is 4, as @eq-gradinv-batch says. Every row lies in the span of the four images to within $4.6\times10^{-16}$, and the unit ratios recover all 4 of the 4 images exactly.

DLG does much worse on the same kind of batch. @tbl-gradinv-dlg-batch runs it on the sigmoid network for 6 random networks and batches of each size.

| $B$ | Median relative error | Examples recovered (error $< 0.05$) | Median final matching loss |
| --- | --- | --- | --- |
| 1 | 0.000 | 1.00 | $1.5\times10^{-27}$ |
| 2 | 0.000 | 0.67 | $1.3\times10^{-27}$ |
| 4 | 0.363 | 0.29 | $3.3\times10^{-5}$ |
| 8 | 0.711 | 0.00 | $7.7\times10^{-5}$ |

: Deep Leakage from Gradients on batches, sigmoid network with 32 hidden units, 600 Adam steps, 6 trials per batch size. {#tbl-gradinv-dlg-batch}

Batches of one and two are recovered to rounding error in most runs. From $B = 4$ the median run gets stuck, with the matching loss near $10^{-5}$ instead of $10^{-27}$, and the reconstructions are blends of the batch (@fig-gradinv-dlg-batch). The failure lies in the optimization, because the information is there. At $B = 4$ there are $64\cdot4 = 256$ unknown pixels against 2,212 equations. On a ReLU network of the same width, which also averages four examples, the unit ratios recovered all four images, so averaging has made the search harder without hiding the inputs.
:::

## How often a unit isolates an example {#sec-gradinv-isolation}

Every isolated example comes out exactly, so the share of a batch that a gradient exposes is the share isolated by at least one unit. How does that share depend on the batch size and on the width of the layer? To answer, you need a model of when units switch on, and the simplest one treats each unit's response to each example as a fair coin.

For a single unit, the coin model is exact in one case. Suppose the first-layer bias is zero and the inputs are drawn from a distribution that is symmetric about zero, such as $\mathcal N(0, I)$. Unit $j$, with weights $(W_1)_{j,:}$, is on for $x_i$ when $(W_1)_{j,:}\,x_i > 0$. Symmetry makes this product equally likely to be positive or negative, and independent inputs make the signs independent across the batch. So each unit's on/off pattern over the batch is a sequence of fair coin flips. Different units look at the same inputs, though, so independence between units is only approximate, and real data are not symmetric at all.

::: {#prp-gradinv-isolation name="Isolation under independent units"}
Take a ReLU first layer with $m$ units and a batch of $B$ examples. Suppose that each unit is on for each example with probability $\frac12$, independently across examples and across units, and that every unit $j$ active on an example $i$ has $(W_2^\top\gamma_{2,i})_j \ne 0$, where $\gamma_{2,i}$ is the last-layer signal of example $i$. Then each example of the batch is recovered exactly by some unit ratio with probability
$$P_{\text{iso}}(B, m) = 1 - (1 - 2^{-B})^m.$$ {#eq-gradinv-isolation}
:::

::: {.proof}
1. *One unit.* By @prp-gradinv-batch and the ReLU slope, unit $j$ gives back $x_i$ exactly when it is on for $x_i$ and off for each of the other $B - 1$ examples, since an active unit has a nonzero signal by hypothesis. Each of these $B$ events has probability $\frac12$, and they are independent, so their probability together is $2^{-B}$.
2. *Many units.* The example is missed only if all $m$ units miss it. The units are independent by hypothesis, so this has probability $(1 - 2^{-B})^m$, and subtracting from 1 gives @eq-gradinv-isolation.
:::

The formula pits an exponential in $B$ against a count that grows linearly in $m$. A unit isolates a given example with probability $2^{-B}$, so among $m$ units the expected number that isolate it is $m2^{-B}$, and almost every example is isolated when $m$ is much larger than $2^B$. To find the width that gives an even chance, require the miss probability $(1 - 2^{-B})^m$ to be at most $\frac12$ and take logs. The log of $1 - 2^{-B}$ is negative, so dividing by it flips the inequality and gives
$$m \ge \frac{\ln2}{-\ln(1 - 2^{-B})}.$$
At $B = 10$ the denominator is $-\ln(1 - 2^{-10}) = 0.000977$, close to $2^{-10}$ itself, so the bound is $0.6931/0.000977 \approx 709.4$, so you need $m \ge 710$ units. Real first layers have thousands of units, so batches of 10 are not safe from this attack.

Data whose examples resemble one another, such as images of a few shapes, are easier for the attacker, not harder. With such images, a unit tends to fire on many examples of a batch or on few. The number of examples it fires for is more spread out than the coin model predicts, which statisticians call **overdispersion**. In particular some units fire only on rare inputs, and when such a unit fires on a large batch it usually fires on just one example, so isolation is much more common at large $B$. Take one extreme model, in which each unit has its own firing rate $r$, drawn uniformly between 0 and 1. A unit with rate $r$ isolates a given example with probability $r(1 - r)^{B-1}$, and averaging over $r$ gives $\int_0^1r(1 - r)^{B-1}\,dr = 1/(B(B + 1))$. That shrinks only like $1/B^2$ instead of halving with each extra example.

![Fraction of a batch recovered exactly by first-layer unit ratios, for ReLU networks with $m = 64$, 256 and 1024 hidden units. Lines: @eq-gradinv-isolation. Filled circles: Gaussian inputs and zero first-layer bias, where each unit's on/off pattern is a sequence of fair coin flips. Open squares: the $8\times8$ images, which are correlated, so units fire on many or few of them and isolation decays much more slowly.](figures/gradinv_isolation.svg){#fig-gradinv-isolation}

::: {#exm-gradinv-isolation}
For the batch of 4 in @exm-gradinv-batch, the network has $m = 32$ units and $1 - 2^{-4} = 15/16$, so @eq-gradinv-isolation predicts
$$P_{\text{iso}}(4, 32) = 1 - (15/16)^{32} = 1 - e^{32\ln(15/16)} = 1 - e^{-2.065} = 0.873$$
per example under independence. All 4 of the 4 images were recovered. @tbl-gradinv-isolation compares @eq-gradinv-isolation with simulation over a range of widths and batch sizes, with 20 random networks and batches per cell (@fig-gradinv-isolation shows the full range).

| $m$ | $B$ | @eq-gradinv-isolation | Gaussian inputs | Images |
| --- | --- | --- | --- | --- |
| 64 | 4 | 0.984 | 0.975 | 0.838 |
| 64 | 8 | 0.222 | 0.188 | 0.550 |
| 64 | 16 | 0.001 | 0.000 | 0.144 |
| 256 | 8 | 0.633 | 0.562 | 0.769 |
| 256 | 12 | 0.061 | 0.046 | 0.542 |
| 256 | 16 | 0.004 | 0.006 | 0.375 |
| 1024 | 12 | 0.221 | 0.196 | 0.850 |
| 1024 | 16 | 0.016 | 0.009 | 0.691 |

: Fraction of batch examples recovered exactly from first-layer unit ratios of a random ReLU network, against @eq-gradinv-isolation. {#tbl-gradinv-isolation}

For Gaussian inputs the formula matches to within the simulation noise, overestimating slightly in seven of the eight rows because units share inputs. For images the formula is far too reassuring about privacy. At $B = 16$ it predicts that 1.6% of the batch is exposed in the 1024-unit network, and 69.1% is.

At $B = 4$, by contrast, the images are recovered less often than the formula predicts, 0.838 against 0.984 for $m = 64$. Overdispersion cuts both ways. The same spread that makes "on for exactly one image" more common at large $B$ also makes "on for all four images" and "on for none" common at small $B$, and units like that isolate nothing.
:::

## Code {#sec-gradinv-code}

The script [code/part02-attacks/gradient_inversion.py](../../code/part02-attacks/gradient_inversion.py) computes every number in this chapter and draws every figure. From the repository root:

```bash
uv run python code/part02-attacks/gradient_inversion.py
```

| Experiment | Result |
| --- | --- |
| Hand example (@tbl-gradinv-tiny) | $p = (0.3441, 0.6559)$, $\gamma_1 = (-0.3279, 0.9838)$; both unit ratios give $(0.5, 1, 0.2)$; label 1 (index 0 in the script) |
| Exact recovery, one image, ReLU, $m = 32$ | 13 usable units of 32; largest relative error $6.7\times10^{-17}$; $s_2/s_1 = 9.5\times10^{-17}$; 2,212 gradient entries for 64 pixels |
| Labels, one example, 1,000 trials | Bias rule 1.000; weight rule 1.000 |
| Label counts, all exact, 200 batches | Averaged guess: $B = 4$: 1.000, $B = 16$: 0.975, $B = 64$: 0.525. Uniform guess: 0.925, 0.095, 0.000 |
| DLG, one image (@tbl-gradinv-dlg) | Relative error $5.8\times10^{-14}$ after 600 steps; matching loss $1.3\times10^{-27}$ |
| Double backpropagation against finite differences | Largest gap $1.2\times10^{-10}$ |
| DLG on batches (@tbl-gradinv-dlg-batch) | Recovered: $B = 1$: 1.00, $B = 2$: 0.67, $B = 4$: 0.29, $B = 8$: 0.00 |
| Batch of 4, ReLU | Rank 4; rows in the span to $4.6\times10^{-16}$; 4 of 4 images isolated |
| Width for isolation probability $\frac12$ at $B = 10$ | 710 |

: The script's output with the default seed. Here $s_1$ and $s_2$ are the two largest singular values of the first-layer gradient. {#tbl-gradinv-code}

[tests/test_gradient_inversion.py](../../tests/test_gradient_inversion.py) checks that the output still matches.

## Summary {#sec-gradinv-summary}

| Result | Statement | Use |
| --- | --- | --- |
| @def-gradinv-game | An honest-but-curious server sees $(\theta, g)$ and outputs reconstructions, scored by relative error | Every attack here needs no malice |
| @prp-gradinv-outer | $\nabla_W\ell = \gamma x^\top$ and $\nabla_b\ell = \gamma$ for a fully connected layer | A fully connected layer's input sits in its gradient |
| @prp-gradinv-backward | $\gamma_2 = p - e_y$ and $\gamma_1 = (W_2^\top\gamma_2)\odot\sigma'(a)$ | The sign pattern for labels; a switched-off ReLU unit has a zero row |
| @thm-gradinv-exact | One example: $x = (\nabla_{W_1}\ell)_{j,:}/(\nabla_{b_1}\ell)_j$, and $\nabla_{W_1}\ell$ has rank one | The input costs one division; fails for a convolutional first layer |
| @prp-gradinv-labels | $y = \arg\min_k(\nabla_{b_2}\ell)_k$; $n_k = \sum_ip_{ik} - B(\nabla_{b_2}\ell)_k$ | The label costs one sign; batch counts are exact for small batches |
| @def-gradinv-dlg, @prp-gradinv-dlg | Minimize $R(x') = \Vert G(x') - g\Vert^2$; the true batch has $R = 0$ | No closed form needed, only a twice-differentiable network; a final $R$ well above zero means the search is stuck |
| @prp-gradinv-batch | $\operatorname{rank}\nabla_{W_1}\ell \le B$, rows in the span of the batch; unit ratios are weighted combinations | Batching hinders search, not structure: the span and isolated examples leak |
| @prp-gradinv-isolation | $P_{\text{iso}} = 1 - (1 - 2^{-B})^m$ for independent units | Most of a batch is exposed once $m$ is much larger than $2^B$; real data are isolated far more often |

: The results of this chapter, and where each is used. {#tbl-gradinv-summary}

## References {#sec-gradinv-references}

- Zhu, Liu & Han, "Deep Leakage from Gradients" (NeurIPS 2019). The gradient-matching attack of @sec-gradinv-dlg, with reconstructions of images and text from single-example and small-batch gradients; replaces ReLU by the sigmoid to keep the objective twice differentiable.
- Zhao, Mopuri & Bilen, "iDLG: Improved Deep Leakage from Gradients" (arXiv 2020). Label recovery from the sign of the last-layer gradient, and DLG with the label fixed.
- Phong, Aono, Hayashi, Wang & Moriai, "Privacy-Preserving Deep Learning via Additively Homomorphic Encryption" (IEEE Transactions on Information Forensics and Security, 2018). Points out that the input of a fully connected layer with a bias can be read off its gradient, @eq-gradinv-ratio.
- Geiping, Bauermeister, Dröge & Moeller, "Inverting Gradients — How easy is it to break privacy in federated learning?" (NeurIPS 2020). Cosine-similarity matching with a total-variation image prior; reconstructions from trained networks, from several local steps and from batches; the proof that fully connected layers leak their inputs in closed form.
- Fowl, Geiping, Czaja, Goldblum & Goldstein, "Robbing the Fed: Directly Obtaining Private Data in Federated Learning with Modified Models" (ICLR 2022). A malicious server inserts a fully connected block whose units bin inputs by a feature, so that most examples of a large batch are isolated.
- Boenisch, Dziedzic, Schuster, Shamsabadi, Shumailov & Papernot, "When the Curious Abandon Honesty: Federated Learning Is Not Private" (IEEE EuroS&P 2023). Trap weights that make first-layer units fire for single examples; exact recovery from large batches by a malicious server.
- Wainakh et al., "User-Level Label Leakage from Gradients in Federated Learning" (PETS 2022). Label counts from batch gradients, as in @eq-gradinv-counts.
- Huang, Gupta, Song, Li & Arora, "Evaluating Gradient Inversion Attacks and Defenses in Federated Learning" (NeurIPS 2021). How the strength of these attacks depends on assumptions such as known labels, batch normalization statistics and untrained models.
- McMahan, Moore, Ramage, Hampson & Agüera y Arcas, "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS 2017). The federated averaging protocol, where clients send updates after several local steps.
