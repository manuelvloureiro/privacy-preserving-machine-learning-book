# Gradient inversion

## Why this matters

A phone keyboard learns from what its users type, and the raw text never leaves the phones. The server sends the current model to each phone, the phone computes the gradient of the loss on its own few examples, and only that gradient goes back. This is **federated learning (FL)**, often described as private because "the data stay on the device". If you design, review or red-team such a system, the question is: **what does one client's gradient reveal about the client's data?**

Often, everything. The one idea is that the gradient of a fully connected layer is an **outer product**. The layer computes $a = Wx + b$, and the weight $W_{jk}$ multiplies the input $x_k$, so the loss changes with $W_{jk}$ at the rate $\delta_jx_k$, where $\delta_j = \partial\ell/\partial a_j$ is the backward signal from everything above the layer. Collected into a matrix,
$$\nabla_W\ell = \delta\,x^\top, \qquad \nabla_b\ell = \delta.$$ {#eq-gradinv-outer}
Row $j$ of the weight gradient is the input $x$ scaled by the single number $\delta_j$, so the layer's input sits in every row. The rest of the network, the loss and the label only rescale the rows. With one example the scale divides out; with a batch the gradient is a sum of one outer product per example, and the attacker's job is to pull the sum apart. @tbl-gradinv-overview maps what the attacker recovers to the sections.

| What the server sees | What it recovers | Section |
| --- | --- | --- |
| One example, fully connected first layer with a bias | The input, exactly, by one division | @sec-gradinv-exact |
| The last layer's gradient | The label from a sign; for a batch, the label counts | @sec-gradinv-labels |
| Any differentiable model, no closed form | The input, by Deep Leakage from Gradients (DLG) | @sec-gradinv-dlg |
| A batch average | The span of the batch, every example a unit isolates, and small batches by search | @sec-gradinv-batch |

: What a server recovers from a client's gradient, and where this chapter shows it. {#tbl-gradinv-overview}

**The threat model.** The server is **honest-but-curious** in the sense of [Threat models and privacy games](01-threat-models.md): it follows the protocol, sends the real model, and only analyses what it receives. It knows the architecture and every parameter, because it chose them, and it sees each client's update, $g = \frac1B\sum_i\nabla_\theta\ell(\theta; x_i, y_i)$, the average gradient over the client's $B$ examples. Reconstructions are scored by the **relative error** $\lVert\hat x - x\rVert/\lVert x\rVert$, with each true example matched to its closest reconstruction, since an average carries no order. The experiments use $8\times8$ images of four shapes and a network with one hidden layer of 32 units at its random initialization: 2,212 gradient entries against 64 unknown pixels per example.

## Exact recovery from a fully connected layer {#sec-gradinv-exact}

**Idea.** By @eq-gradinv-outer, row $j$ of the first layer's weight gradient is $\delta_{1,j}x^\top$, and entry $j$ of its bias gradient is the same number $\delta_{1,j}$. Dividing one by the other cancels it (Phong et al. 2018).

**Result.** For a single example and a fully connected first layer with a bias, every unit $j$ with a nonzero bias gradient gives
$$x = \frac{(\nabla_{W_1}\ell)_{j,:}}{(\nabla_{b_1}\ell)_j}.$$ {#eq-gradinv-ratio}
The weight gradient has rank one, and its rows all lie on the line through $x$. The attack needs no search, no prior on the data and no label, and it checks itself: all usable units give the same answer.

**Example.** A two-unit ReLU network with $W_1 = \begin{pmatrix}0.4 & -0.2 & 0.1\\ 0.3 & 0.5 & -0.4\end{pmatrix}$, $b_1 = (0.1, -0.1)$, $W_2 = \begin{pmatrix}1 & -1\\ 0.5 & 0.5\end{pmatrix}$ and no output bias sees $x = (0.5, 1, 0.2)$ with the first class as label. The backward signal at the first layer is $\delta_1 = (-0.3279, 0.9838)$, so the two rows of the weight gradient are $(-0.1640, -0.3279, -0.0656)$ and $(0.4919, 0.9838, 0.1968)$. Dividing each by its bias entry gives $(0.5, 1, 0.2)$ twice; @sec-gradinv-code runs it. On an $8\times8$ image and the 32-unit network, 13 units are switched on, and every one of the 13 ratios equals the image to a relative error of at most $6.7\times10^{-17}$, rounding error (@fig-gradinv-rank-one).

![Exact recovery from the first-layer gradient of a ReLU network with 32 hidden units. Left: the client's input. Middle: the weight gradient, one row per hidden unit, next to the bias gradient; every nonzero row is the image times its bias entry, and switched-off units give zero rows. Right: one row divided by its bias entry, which is the input exactly.](figures/gradinv_rank_one.svg){#fig-gradinv-rank-one}

**Pitfalls.** Depth does not protect: twenty layers above the first and a custom loss only change $\delta_1$, which cancels. ReLU units switched off on $x$ give nothing, but one active unit is enough. Removing the bias does not help much: the rank-one structure still gives the direction of $x$, and a one-dimensional search finds its scale. The same division works at any fully connected layer, where it yields that layer's input, an intermediate feature vector. What really breaks the closed form is a convolutional first layer, whose gradient is a sum of outer products over all image patches, and a batch of more than one example, which breaks it only partly (@sec-gradinv-batch).

## Label recovery from the last layer {#sec-gradinv-labels}

**Idea.** With softmax and cross-entropy, the last layer's backward signal is the predicted probabilities minus the one-hot label, $p - e_y$. Softmax probabilities lie strictly between 0 and 1, so the entry at the true label, $p_y - 1$, is the only negative one (Zhao, Mopuri & Bilen 2020).

**Result.** For one example, the label is the position of the smallest, and only negative, entry of the last-layer bias gradient,
$$y = \arg\min_k(\nabla_{b_2}\ell)_k.$$ {#eq-gradinv-labels}
Without a bias, if the hidden activations are non-negative (ReLU or sigmoid), the label is the only row of the last-layer weight gradient with a negative sum. For a batch of $B$, the bias gradient gives the label counts: the number $n_k$ of examples of class $k$ is
$$n_k = \sum_{i=1}^Bp_{ik} - B\,(\nabla_{b_2}\ell)_k,$$ {#eq-gradinv-counts}
where the attacker estimates the model's total predicted probability for class $k$, $\sum_ip_{ik}$, as $B$ times the model's average prediction on data of its own, and rounds.

**Example.** For one example both rules name the label correctly in 1,000 of 1,000 trials. A batch of 16 has true counts $(4, 2, 4, 6)$. With the average prediction taken over 200 of the attacker's own images, @eq-gradinv-counts gives $(3.88, 2.00, 4.19, 5.92)$, which rounds to the truth. Guessing a uniform prediction gives $(2.12, 3.01, 4.27, 6.60)$, which does not. Over 200 random batches, the averaged guess gets all counts exact for 1.000 of batches of 4, 0.975 of batches of 16 and 0.525 of batches of 64 (@fig-gradinv-labels).

![Labels from the last-layer bias gradient. Left: for one example it has a single negative entry, at the true label. Right: for a batch of 16, the counts estimated with the model's average prediction on the attacker's own images are close to the true counts; with a uniform guess they are off by up to about half an example.](figures/gradinv_labels.svg){#fig-gradinv-labels}

**Pitfalls.** Labels are often the most sensitive part of an example: the diagnosis, the next word typed. A batch leaks its label histogram even when no input is recovered, and the counts stay close when not exact. The rule is exact at any point of training, because the sign never depends on how good the model is; only a probability that rounds to 1 in floating point hides it. Knowing the label also makes inversion easier, because the search of @sec-gradinv-dlg then runs over the input only.

## Deep Leakage from Gradients {#sec-gradinv-dlg}

**Idea.** Where no closed form exists, as with convolutions, normalization layers or no biases, guess an input, compute the gradient it would produce, and move the guess until the gradients match. This is DLG (Zhu, Liu & Han 2019).

**Result.** With the labels fixed by @sec-gradinv-labels, DLG minimizes the **matching loss**, the squared distance between the gradient of the guessed batch and the observed $g$:
$$\hat x = \arg\min_{x'}R(x'), \qquad R(x') = \Big\lVert\frac1B\sum_{i=1}^B\nabla_\theta\ell(\theta; x_i', y_i) - g\Big\rVert^2.$$ {#eq-gradinv-dlg}
It starts from random noise and runs a gradient method on $R$, which automatic differentiation provides by differentiating through the backward pass. The true batch has $R = 0$, the smallest possible value. $R$ is not convex, so the search may stop at a point that is not the input, but the final value tells you which: $R$ at rounding error means an exact match, $R$ stuck well above zero means the search failed.

**Example.** On one image, a sigmoid network with 32 hidden units, and 600 steps of the Adam optimizer from uniform noise, the relative error is $7.5\times10^{-2}$ at step 50, already visually correct, and $5.8\times10^{-14}$ at step 600, with matching loss $1.3\times10^{-27}$ (@fig-gradinv-dlg-progress).

![Deep Leakage from Gradients on one image, sigmoid network with 32 hidden units. Top: the reconstruction at selected iterations, starting from noise, and the true input. Bottom: the matching loss and the relative error on a log scale; both fall until they reach rounding error.](figures/gradinv_dlg_progress.svg){#fig-gradinv-dlg-progress}

**Pitfalls.** The search needs second derivatives of the network, which is why Zhu et al. replaced ReLU by the sigmoid: ReLU's second derivative is zero almost everywhere, so the search gets no signal through it. A failed run is an optimization failure, not proof of privacy; restarts, better optimizers and image priors help. Geiping et al. (2020) match gradients by cosine distance and add a penalty on differences between neighbouring pixels, and recover recognizable images from trained networks and from batches. A small final $R$ is not certain success either; a loss that stopped falling well above rounding error means restart.

## Batches and averaging {#sec-gradinv-batch}

**Idea.** Clients usually send a batch average, and averaging is often said to protect individuals. It does break the one-line formula and makes the search harder, but three things survive: the span of the batch, every example that some unit "sees alone", and a search that still works for small batches.

**Result.** The first-layer gradient is a sum of $B$ outer products, so its rank is at most $B$ and every row is a combination of the $B$ inputs: the server learns the subspace that contains the batch. A unit's ratio from @eq-gradinv-ratio is now a weighted average of the inputs, and it equals one input exactly when the unit is active for that example only, when it **isolates** it. If each of $m$ ReLU units fires on each example like an independent fair coin, each example is recovered exactly with probability
$$P_{\text{iso}}(B, m) = 1 - (1 - 2^{-B})^m.$$ {#eq-gradinv-isolation}
That is exponential in $B$ and linear in $m$: an even chance at $B = 10$ needs 710 units, and real first layers have thousands.

**Example.** A batch of 4 images through the 32-unit ReLU network gives a rank-4 gradient, and the unit ratios recover all 4 images exactly, where the formula predicts 0.873 per example. @tbl-gradinv-isolation tests the formula. For Gaussian inputs it matches. For the images it is far too optimistic about privacy at large batches: at $B = 16$ in the 1024-unit network it predicts 1.6% exposed, and 69.1% is. Similar images make units fire on many or few of them, so units that fire on exactly one are common (@fig-gradinv-isolation). DLG on the sigmoid network recovers all single images, 0.67 of examples in batches of 2, 0.29 at $B = 4$ and none at $B = 8$, where the matching loss stalls near $10^{-5}$ and the reconstructions are blends.

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

![Fraction of a batch recovered exactly by first-layer unit ratios, for ReLU networks with 64, 256 and 1024 hidden units. Lines: the independence formula. Filled circles: Gaussian inputs, which follow it. Open squares: the images, which are correlated, so isolation decays much more slowly.](figures/gradinv_isolation.svg){#fig-gradinv-isolation}

**Pitfalls.** Batching is not a defence. The label histogram, the span of the batch and every isolated example come out exactly, and DLG's failure at $B = 8$ is a stuck search, not missing information: there are still 2,212 equations. Sending the parameter change after several local steps does not help either: for one example, every step's first-layer gradient is a multiple of the same $x^\top$, so the division still recovers $x$. A malicious server does much better than an honest one: by choosing the weights it can plant units that fire on rare inputs only (Boenisch et al. 2023) or that bin inputs by a feature (Fowl et al. 2022), so that most of a large batch is isolated by design. Defences have to change what the server sees. Differential privacy ([Part 3](../part03-dp-foundations/00-introduction.md)) bounds what any update reveals, and secure aggregation, which reveals only the sum over many clients, is introduced in [Local, shuffle and distributed trust models](../part03-dp-foundations/08-trust-models.md).

## Code {#sec-gradinv-code}

The hand example of @sec-gradinv-exact, with the label rule and the isolation formula, in plain Python. The script indexes classes from 0, so label 0 is the first class.

```python
from math import ceil, exp, log

Vector = list[float]
Matrix = list[Vector]


def client_gradient(w1: Matrix, b1: Vector, w2: Matrix, x: Vector, y: int) -> tuple[Matrix, Vector, Vector]:
    """What a client sends for one example: the gradients of W1, b1 and b2 (one ReLU layer, softmax, cross-entropy)."""
    a = [sum(wk * xk for wk, xk in zip(row, x)) + bj for row, bj in zip(w1, b1)]
    h = [max(aj, 0.0) for aj in a]
    z = [sum(wk * hk for wk, hk in zip(row, h)) for row in w2]
    total = sum(exp(zk) for zk in z)
    delta2 = [exp(zk) / total - (1.0 if k == y else 0.0) for k, zk in enumerate(z)]  # p - e_y
    delta1 = [sum(delta2[k] * w2[k][j] for k in range(len(w2))) * (1.0 if a[j] > 0 else 0.0) for j in range(len(a))]
    grad_w1 = [[dj * xk for xk in x] for dj in delta1]  # outer product delta1 x^T
    return grad_w1, delta1, delta2  # the bias gradients equal the backward signals


def recover_input(grad_w1: Matrix, grad_b1: Vector) -> Vector:
    """Exact recovery: a row of the weight gradient divided by its bias entry."""
    j = next(j for j, g in enumerate(grad_b1) if g != 0.0)
    return [g / grad_b1[j] for g in grad_w1[j]]


def recover_label(grad_b2: Vector) -> int:
    """The only negative entry of the last-layer bias gradient is the label."""
    return min(range(len(grad_b2)), key=lambda k: grad_b2[k])


def isolation(batch: int, units: int) -> float:
    """Chance that some ReLU unit fires for one given example of the batch only, under the coin-flip model."""
    return 1 - (1 - 2.0**-batch) ** units


def width_for_even_odds(batch: int) -> int:
    """Smallest number of units that isolates each example with probability at least 1/2."""
    return ceil(log(2) / -log(1 - 2.0**-batch))


w1 = [[0.4, -0.2, 0.1], [0.3, 0.5, -0.4]]
b1 = [0.1, -0.1]
w2 = [[1.0, -1.0], [0.5, 0.5]]
gw1, gb1, gb2 = client_gradient(w1, b1, w2, x=[0.5, 1.0, 0.2], y=0)
print([round(v, 4) for v in recover_input(gw1, gb1)])  # [0.5, 1.0, 0.2]
print([round(v, 4) for v in gb2], recover_label(gb2))  # [-0.6559, 0.6559] 0
print(f"{isolation(4, 32):.3f}")  # 0.873
print(width_for_even_odds(10))  # 710
```

The full edition's script runs every attack in this chapter on the image network, including DLG with a hand-written double backpropagation, in about half a minute:

```bash
uv run python code/part02-attacks/gradient_inversion.py
```

## Cheat sheet

| Result | Formula | Needs | Fails when |
| --- | --- | --- | --- |
| Outer product | $\nabla_W\ell = \delta x^\top$, $\nabla_b\ell = \delta$ | A fully connected layer | — |
| Exact recovery, one example | $x = (\nabla_{W_1}\ell)_{j,:}/(\nabla_{b_1}\ell)_j$ | First layer fully connected with bias | Convolutional first layer; batches (partly) |
| Label, one example | $y = \arg\min_k(\nabla_{b_2}\ell)_k$ | Softmax and cross-entropy | $p_y$ rounds to 1 |
| Label counts, batch | $n_k = \sum_ip_{ik} - B(\nabla_{b_2}\ell)_k$ | Last-layer bias; a guess of $\sum_ip_{ik}$ | Large $B$ with a poor guess |
| DLG | $\min_{x'}\lVert G(x') - g\rVert^2$, $G$ the gradient of the guessed batch | Twice-differentiable network; labels | Search stuck (final $R \gg 0$), larger batches |
| Span of a batch | Rank $\le B$, rows in $\operatorname{span}\{x_i\}$ | Fully connected first layer | — |
| Isolation | $1 - (1 - 2^{-B})^m$ under independence | ReLU units, random weights | Real data: much slower decay |

: The results of this chapter. {#tbl-gradinv-cheat}

**Rules of thumb**

- One example through a fully connected layer with a bias: the input costs one division, the label one sign.
- Where there is no closed form, match gradients; the final matching loss tells you whether it worked.
- Batching helps against search but not against structure. Label counts, the span and every isolated example leak, and with real data that is most of a batch of 10 or more on a wide layer.
- An honest server can already do all of this; a malicious one can do it for large batches by design. Protect updates with differential privacy and secure aggregation, not with batching or local steps.

## Further reading

- Zhu, Liu & Han, "Deep Leakage from Gradients" (NeurIPS 2019): the gradient-matching attack on images and text.
- Geiping, Bauermeister, Dröge & Moeller, "Inverting Gradients — How easy is it to break privacy in federated learning?" (NeurIPS 2020): cosine matching with an image prior, and attacks on trained networks, local steps and batches.
- Huang, Gupta, Song, Li & Arora, "Evaluating Gradient Inversion Attacks and Defenses in Federated Learning" (NeurIPS 2021): how attack strength depends on assumptions such as known labels and untrained models; read it before trusting a published attack number.
- Boenisch, Dziedzic, Schuster, Shamsabadi, Shumailov & Papernot, "When the Curious Abandon Honesty: Federated Learning Is Not Private" (IEEE EuroS&P 2023): trap weights and exact recovery from large batches by a malicious server.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/06-gradient-inversion.md).
