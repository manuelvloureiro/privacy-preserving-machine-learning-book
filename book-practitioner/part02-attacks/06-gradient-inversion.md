# Gradient inversion

## Why this matters

A phone keyboard learns from what its users type, and the raw text never leaves the phones. Instead, a server sends the current model to each phone, the phone computes the gradient of the loss on its own few examples, and it sends back only that gradient. The server averages the gradients of many phones and takes a training step. This is **federated learning (FL)**, often called private because "the data stay on the device".

This chapter asks what a single client's gradient reveals about its data, and the answer is often everything. A gradient holds one derivative for every parameter, and many of them are simple functions of the input, so the input can be read back. Averaging over a batch protects much less than people expect. So treat a per-client update as being as sensitive as the data it came from.

## The attacker and what it sees {#sec-gradinv-setting}

The attacker is the server. It is **honest-but-curious** in the sense of [Threat models and privacy games](01-threat-models.md), so it follows the protocol, sees each client's own update and knows every parameter, since it chose them. A client holds a batch of $B$ examples, each an input with its label. Write $\theta$ for all the parameters, and $\ell(\theta; x_i, y_i)$ for the loss on example $i$ with input $x_i$ and label $y_i$. The client sends the average gradient
$$g = \frac1B\sum_{i=1}^B\nabla_\theta\ell(\theta; x_i, y_i).$$

The numbers in this chapter come from the full edition's experiments, on small grey-scale images and a network with one hidden layer of 32 units. Its gradient gives 2,212 numbers, one per parameter, against 64 unknown pixels per image.

## Exact recovery from a fully connected layer {#sec-gradinv-exact}

The most common first layer, a fully connected layer with a bias, gives its input back by one division. Each unit $j$ multiplies the input $x$ by its weights and adds a bias. Write $\delta_j$ for the rate at which the loss responds to that unit's output, its backward signal. The chain rule then makes the weight gradient and the bias gradient
$$\nabla_W\ell = \delta\,x^\top, \qquad \nabla_b\ell = \delta.$$ {#eq-gradinv-outer}
So row $j$ of the weight gradient is the input scaled by $\delta_j$, and entry $j$ of the bias gradient is that same scale. For a single example, dividing one by the other gives the input exactly, for every unit whose bias gradient is not zero:
$$x = \frac{(\nabla_{W_1}\ell)_{j,:}}{(\nabla_{b_1}\ell)_j}.$$ {#eq-gradinv-ratio}

Everything above the layer, including the depth of the network, the loss and the label, sits inside the scale $\delta_j$ and cancels. So the attack needs no optimization and no knowledge of the data, and it checks itself, because every usable unit gives the same answer. In the experiment, every switched-on unit gives back the image to rounding error. At a deeper fully connected layer, the same division gives that layer's input.

Switched-off ReLU units give nothing, but one active unit is enough. Without a bias, every row still points along $x$, so a one-dimensional search for the scale finds it. What stops the division is a convolutional first layer, because its filter is reused at every position of the image, so its gradient is a sum that no division separates. For such models the attacker searches instead (@sec-gradinv-dlg).

## Label recovery from the last layer {#sec-gradinv-labels}

The label is often the most sensitive part of an example, such as a diagnosis. It can be read off the last layer without any search (Zhao, Mopuri & Bilen 2020). The rule needs a last layer with a bias, followed by the softmax and the cross-entropy loss. Write $p$ for the predicted probabilities and $e_y$ for the vector with a 1 at the position of the label $y$. The last layer's bias gradient is then $p - e_y$, which is negative only at the true label. So for a single example
$$y = \arg\min_k(\nabla_{b_2}\ell)_k.$$ {#eq-gradinv-labels}
Without the bias, the same sign pattern shows in the last layer's weight gradient when the hidden values cannot be negative. The one failure is a very confident model, whose true-label probability rounds to exactly 1, so the sign is lost.

A batch hides which example has which label, but not how many examples there are of each class. Write $n_k$ for the number of examples of class $k$, and $p_{ik}$ for the probability the model gives that class on example $i$. Then
$$n_k = \sum_{i=1}^Bp_{ik} - B\,(\nabla_{b_2}\ell)_k.$$ {#eq-gradinv-counts}
The server does not know the sum of predictions, but it knows the model. So it estimates the sum from its own data of the same kind, and rounds the result. With that estimate, every count came out exact in all batches of 4, but larger batches miss more often, and the share exact falls to 52.5% for batches of 64.

## Deep Leakage from Gradients {#sec-gradinv-dlg}

When the division does not apply, the attacker searches. **Deep Leakage from Gradients (DLG)** (Zhu, Liu & Han 2019) guesses an input and computes the gradient it would produce. It then moves the guess until that gradient matches the observed one. Write $R$ for the **matching loss**, the squared distance between the two gradients. With the labels from @sec-gradinv-labels, the attack looks for the guesses that make it smallest:
$$\hat x_1, \dots, \hat x_B = \arg\min_{x_1', \dots, x_B'}R(x'), \qquad R(x') = \Big\lVert\frac1B\sum_{i=1}^B\nabla_\theta\ell(\theta; x_i', y_i) - g\Big\rVert^2.$$ {#eq-gradinv-dlg}

In PyTorch the attack is a short loop. Its key call is `torch.autograd.grad` with `create_graph=True`, which keeps the gradient differentiable so that the optimizer can follow the slope of $R$.

```text
import torch
import torch.nn.functional as F

loss = F.cross_entropy(model(x_guess), y_known)
grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
R = sum(((g - g_obs) ** 2).sum() for g, g_obs in zip(grads, observed_grads))
R.backward()   # then one optimizer step on x_guess; repeat
```

Three things decide whether it works. The activation must have a slope that varies, so Zhu et al. replaced the ReLU by the sigmoid. The search can get stuck, because $R$ is not convex, but the final value of $R$ tells you which case you are in. A loss at rounding error means an exact match, while one that stops falling above it means a restart from a new random guess. Finally, a failed run is an optimization failure, not a lack of information. Geiping et al. (2020) compared gradients by direction and added a smoothness penalty, and recovered recognizable images from trained networks and from batches. In the full edition's experiment, one image comes back exact to about 13 digits after 600 steps of the Adam optimizer.

## Batches and averaging {#sec-gradinv-batch}

Clients usually send the average over a batch, or the parameter change after several local steps. Averaging breaks the division and makes the search harder, but much of the structure survives, because each example contributes one outer product to the first-layer gradient, so for a batch of $B$
$$\nabla_{W_1}\ell = \frac1B\sum_{i=1}^B\delta_{1,i}\,x_i^\top, \qquad \operatorname{rank}\nabla_{W_1}\ell \le B.$$ {#eq-gradinv-batch}
Every row of the gradient is a combination of the $B$ inputs. So the server learns the few directions that every example is built from, which already rules out almost every input.

The division still recovers an example exactly when one ReLU unit is switched on for that example and off for the rest of the batch, because the unit then **isolates** the example. Suppose the $m$ hidden units switched on like independent fair coins. Each example would then be isolated with probability $1 - (1 - 2^{-B})^m$, so an even chance at a batch of 10 needs only 710 units. Real data are worse than that formula, because images of the same kind make a unit fire on many examples or on few. With 1,024 units and a batch of 16, the formula predicts that 1.6% of the batch is exposed, while in fact 69.1% is. So wide layers with real data expose most examples of batches of 10 or more.

DLG suffers more from batching, and from a batch of 4 the median run gets stuck. That is a harder search, not less information, since the isolation attack recovered all four images of such a batch.

Several local steps, as in federated averaging, hamper the search but do not protect a single example. Every step's first-layer gradient is an outer product with the same input, so the division still works on the total change. And a malicious server that chooses the weights can isolate most examples of even a large batch by design. It plants units that fire only on rare inputs or sort inputs into narrow bins (Boenisch et al. 2023; Fowl et al. 2022).

## What to do as a deployer {#sec-gradinv-defences}

For an FL deployment, ask who can see one client's update, using @tbl-gradinv-decisions.

| Setting | What leaks to the server | What to do |
| --- | --- | --- |
| One example per update, fully connected first layer with bias | The input exactly, and the label | Never deploy without protection |
| Small batches, any architecture | Labels or their counts, isolated examples, most of the rest by search | Treat each update as raw data |
| Large batches or several local steps | Label counts, the span of the batch, isolated examples | Not a defence on its own |
| Server not trusted to choose weights | Most examples of any batch | Verify the model clients receive |
| Updates protected by differential privacy (DP) | Bounded by the privacy level | The only protection with a guarantee |

: Gradient leakage by FL setting, and the response. {#tbl-gradinv-decisions}

Two protections change the picture. **Secure aggregation** is a cryptographic protocol that lets the server see only the sum over many clients, never one client's update. That removes the per-client view that every attack here uses. It needs a minimum number of clients per sum, though, because a sum over very few clients is close to a single update. So larger cohorts help, and clients that drop out mid-round must not leave a small group exposed.

Secure aggregation alone is not enough, because the sum and the final model still leak and a server can still choose malicious weights. For a guarantee, add [training with DP](../part05-dp-ml/02-dp-sgd.md). In DP federated averaging, each client's update is clipped to a fixed norm and noise is added to the sum. That bounds what any update reveals, whoever sees it. Opacus provides per-example clipping and noise for PyTorch through its `PrivacyEngine`, and Flower and TensorFlow Federated both offer secure aggregation and differentially private aggregation of client updates. Heuristic defences, such as pruning small gradient entries or mixing inputs, raise the cost of the search but carry no guarantee.

To check your own setup, attack it with your own architecture, batch size and number of local steps. Geiping's open-source `breaching` framework implements the published attacks for PyTorch models, and the full edition's script, `code/part02-attacks/gradient_inversion.py`, runs every attack in this chapter.

## Cheat sheet

| Attack | Formula | Needs | Fails when |
| --- | --- | --- | --- |
| Exact recovery, one example | $x = (\nabla_{W_1}\ell)_{j,:}/(\nabla_{b_1}\ell)_j$ | Fully connected first layer with bias | Convolutional first layer; batches, in part |
| Label, one example | $y = \arg\min_k(\nabla_{b_2}\ell)_k$ | Softmax and cross-entropy | $p_y$ rounds to 1 |
| Label counts, batch | $n_k = \sum_ip_{ik} - B(\nabla_{b_2}\ell)_k$ | Last-layer bias; an estimate of $\sum_ip_{ik}$ | Large $B$ with a poor estimate |
| Deep Leakage from Gradients | Minimize the matching loss $R$ | A smooth network; the labels | Search stuck (final $R \gg 0$); larger batches |
| Span and isolation, batch | Rank $\le B$; isolated examples recovered exactly | Fully connected first layer, ReLU units | Never safe; real data are isolated often |
| Secure aggregation alone | Hides single updates | Enough clients per sum | The sum and the model still leak; add DP |

: The results of this chapter. {#tbl-gradinv-cheat}

**Rules of thumb**

- "We share gradients, not data" is not privacy, because a per-client update is as sensitive as the data it came from.
- Do not rely on batch size or local steps, which leave the label counts, the span and every isolated example exposed.
- Do not trust a server that controls the model's weights.
- Secure aggregation needs large cohorts and safe handling of dropouts, and only DP bounds what any update reveals.

## Further reading

- Zhu, Liu & Han, "Deep Leakage from Gradients" (NeurIPS 2019): the gradient-matching attack, with reconstructions of images and text.
- Geiping, Bauermeister, Dröge & Moeller, "Inverting Gradients — How easy is it to break privacy in federated learning?" (NeurIPS 2020): cosine matching with an image prior, and attacks on trained networks, local steps and batches.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/06-gradient-inversion.md).
