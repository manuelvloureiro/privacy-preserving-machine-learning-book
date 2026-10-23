# Gradient inversion

## Why this matters

A phone keyboard learns from what its users type, and the raw text never leaves the phones. Instead, a server sends the current model to each phone, the phone computes the gradient of the loss on its own few records, and it sends back only that gradient. The server averages them and takes a training step. This is **federated learning (FL)**, often called private because "the data stay on the device".

Before you deploy it, ask what a single client's gradient reveals about its data. Often, everything. A gradient holds one derivative for every parameter, and many of them are simple functions of the input, so the input can be read back. Averaging over a batch protects much less than people expect (@sec-gradinv-batch), so treat a per-client update as being as sensitive as the data it came from.

## The attacker and what it sees {#sec-gradinv-setting}

The attacker is the server, which sees each client's own update and knows every parameter, since it sent the model out. In the terms of [Threat models and privacy games](01-threat-models.md), its access is therefore the **transcript**, everything training exposes along the way. Assume first that it is **honest-but-curious**, so it only analyses what it receives. What it receives is simple, because each client holds a batch of $B$ records and sends the average of their gradients. Write $\ell_i$ for the loss on record $i$, which depends on all the parameters $\theta$. The client then sends
$$g = \frac1B\sum_{i=1}^B\nabla_\theta\ell_i.$$

## Exact recovery from a fully connected layer {#sec-gradinv-exact}

The most common first layer, a fully connected layer with a bias, gives its input back by one division. Write $W_1$ and $b_1$ for this first layer's weights and bias. Each unit $j$ multiplies the input $x$ by its row of weights and adds its bias. Its backward signal $u_j$ is how fast the loss responds to its output, and the chain rule turns it into the weight and bias gradients
$$\nabla_{W_1}\ell = u\,x^\top, \qquad \nabla_{b_1}\ell = u.$$ {#eq-gradinv-outer}
So row $j$ of the weight gradient is the input scaled by $u_j$, and entry $j$ of the bias gradient is that same scale. For a single record, dividing one by the other gives the input exactly, for every unit whose bias gradient is not zero:
$$x = \frac{(\nabla_{W_1}\ell)_{j,:}}{(\nabla_{b_1}\ell)_j}.$$ {#eq-gradinv-ratio}

Everything above the layer, including the depth of the network, the loss and the label, sits inside the scale $u_j$ and cancels. So the attack needs no optimization and no knowledge of the data. The snippet plays one client with one image of scikit-learn's bundled digits in PyTorch. Its last line reads the label, as @sec-gradinv-labels explains.

```python
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits

torch.manual_seed(0)
images, labels = load_digits(return_X_y=True)
x = torch.tensor(images[[7]] / 16, dtype=torch.float32)  # one client's image, pixels in [0, 1]
y = torch.tensor(labels[[7]])
first, last = torch.nn.Linear(64, 32), torch.nn.Linear(32, 10)
F.cross_entropy(last(torch.relu(first(x))), y).backward()  # the gradient the client sends
# Row j of the weight gradient is the input times the bias gradient of unit j: divide them.
j = int(first.bias.grad.abs().argmax())
recovered = first.weight.grad[j] / first.bias.grad[j]
print(f"image recovered: {torch.allclose(recovered, x[0], atol=1e-6)}")  # image recovered: True
# Only the true label's logit has a negative gradient (softmax minus one-hot).
print(f"recovered label {int(last.bias.grad.argmin())}, true label {int(y)}")  # recovered label 7, true label 7
```

The snippet picks the unit whose bias gradient is largest in size, because a switched-off ReLU unit has a zero gradient, and one active unit is enough. What stops the division is a convolutional first layer. Its filter is reused at every position of the image, so its gradient adds up one such product per image patch, and no division separates the terms of a sum. The attacker then has to search instead (@sec-gradinv-dlg).

## Label recovery from the last layer {#sec-gradinv-labels}

The label is often the most sensitive part of an example, such as a diagnosis. It can be read off the last layer without any search (Zhao, Mopuri & Bilen 2020), if that layer has a bias and feeds the softmax and cross-entropy loss. Write $p$ for the predicted probabilities and $e_y$ for the vector with a 1 at the position of the label $y$. The gradient of the last layer's bias $b_2$ is then $p - e_y$, which is negative only at the true label. So for a single record
$$y = \arg\min_k(\nabla_{b_2}\ell)_k,$$ {#eq-gradinv-labels}
which is the snippet's `argmin`. The one failure is a very confident model. If its true-label probability $p_y$ rounds to 1 in floating point, the label's entry becomes zero, so no entry is negative.

A batch hides which record has which label, but not how many records there are of each class. Each entry of the bias gradient is the class's average predicted probability minus its share of the batch. So a server that estimates the average prediction on data of its own can read off each class's count. In the full edition's experiment, every batch of 4 gave all its counts exactly. Even among batches of 64, 52.5% of the batches did.

## Deep Leakage from Gradients {#sec-gradinv-dlg}

When the division does not apply, the attacker searches. **Deep Leakage from Gradients (DLG)** (Zhu, Liu & Han 2019) guesses an input and computes the gradient it would produce, then moves the guess until that gradient matches the observed one. Write $x_i'$ for the guessed inputs, $y_i$ for their labels, read first as in @sec-gradinv-labels, and $\ell(\theta; x', y)$ for the loss of one guess. The **matching loss** $R$ is the squared distance between the guesses' average gradient and the observed one, so the attack looks for the guesses that make it smallest:
$$\hat x_1, \dots, \hat x_B = \arg\min_{x_1', \dots, x_B'}R(x'), \qquad R(x') = \Big\|\frac1B\sum_{i=1}^B\nabla_\theta\ell(\theta; x_i', y_i) - g\Big\|^2.$$ {#eq-gradinv-dlg}
In PyTorch the attack is a short loop around one call. With `create_graph=True`, `torch.autograd.grad` returns the guess's gradient in a form that can itself be differentiated. An optimizer such as Adam can then follow the slope of $R$ back to the input.

Two things decide whether the search works. The activation's slope must vary, because the search differentiates a gradient, and the ReLU's slope is constant on each side of zero. So Zhu et al. used the sigmoid. And because $R$ is not convex, the search can get stuck. So run it until $R$ stops falling, and judge the final value against the squared size $\|g\|^2$ of the observed gradient $g$, in the precision you compute in.

The full edition's experiment runs 600 steps of Adam in double precision, with a learning rate starting at 0.05 and decaying linearly to zero. There a successful run ends near $10^{-27}$ and a stuck one near $10^{-5}$, and one image comes back exact to about 13 significant digits. Another input can also give a small $R$, so restart from several random guesses, and trust an image only when the restarts agree on it.

## Batches and averaging {#sec-gradinv-batch}

Clients usually send the average over a batch, or the change in their parameters after several training steps on the device (**local steps**). Averaging breaks the division and makes the search harder, but much of the structure survives. Each record adds one outer product to the first-layer gradient, so every row of that gradient is a combination of the $B$ inputs. The server therefore learns the **span** of the batch, the set of all such combinations, which contains every record.

The division also still recovers an example exactly when some ReLU unit is switched on for that example only. That unit then **isolates** the record. If units switched on like independent fair coins, a unit would isolate a given record with probability $2^{-B}$, so isolation would become rare as the batch grows. But real images make a unit fire on many records or on few, so far more units fire on just one. Take the full edition's network with 1,024 hidden units and batches of 16 synthetic $8\times8$ images. The coin model predicts that 1.6% of the batch is exposed, but 69.1% was.

The share exposed depends on both the batch size and the width of the first layer, so report both. @tbl-gradinv-isolation gives the shares from the same experiment.

| Hidden units | Batch size | Share of the batch recovered exactly |
| --- | --- | --- |
| 64 | 4 | 0.838 |
| 64 | 8 | 0.550 |
| 64 | 16 | 0.144 |
| 1,024 | 16 | 0.691 |

: Isolation attack on synthetic images, from the full edition. {#tbl-gradinv-isolation}

A larger batch or a narrower layer lowers the share, but a malicious server undoes both. So measure at your own width and batch size, and treat any isolated record as a breach.

DLG suffers more from batching. On the full edition's sigmoid network with 32 hidden units, the median run from a batch of 4 gets stuck. That is a harder search, not less information: the isolation attack recovered all four images of such a batch. Local steps do not protect a single record either. Each step's first-layer gradient is an outer product with the same input, and only the backward signal changes, so the division still works on the total change. And a **malicious** server that chooses the weights can plant units that fire only on rare inputs, each isolating one record. It then recovers most records of even a large batch by design (Boenisch et al. 2023; Fowl et al. 2022).

## What to do as a deployer {#sec-gradinv-defences}

@tbl-gradinv-decisions reads off the risk of an FL deployment from who can see one client's update.

| Setting | What leaks to the server | What to do |
| --- | --- | --- |
| One record per update, fully connected first layer with bias | The input exactly, and the label | Never deploy without protection |
| Batches of any size, or several local steps | Label counts, the span of the batch, isolated records, and the rest by search in small batches | Treat each update as raw data |
| Server not trusted to choose weights | Most records of any batch | Batch size is no defence; treat each update as raw data |
| Updates protected by differential privacy (DP) | Bounded per client by the DP guarantee | The only protection with a guarantee |

: Gradient leakage by FL setting, and the response. {#tbl-gradinv-decisions}

**Secure aggregation** is a cryptographic protocol that lets the server see only the sum over many clients, never one client's update. That removes the per-client view that every attack here uses, but a sum over very few clients is close to a single update. So set a minimum cohort size, and make sure that clients dropping out mid-round cannot shrink a sum below it.

Secure aggregation alone is not enough. The sum is still a batch average that the attacks above can read, the final model still leaks like any model, and a malicious server can still choose its weights. For a guarantee, add DP with the client as the unit of privacy. Each client's update is clipped, that is, scaled down to a fixed maximum norm, and noise is added to the sum. That bounds what any one client's data can change ([User-level privacy](../part05-dp-ml/07-user-level.md)). Flower and TensorFlow Federated offer both protections. Heuristic defences such as gradient pruning raise the cost of the search but carry no guarantee.

To assess your own deployment, start by finding out what the server sees, single updates or only sums, and with what minimum cohort. Then note the client batch size, the number of local steps, and the type and width of the first layer. Run the attacks on your own architecture at that batch size. Geiping's open-source `breaching` framework implements the published search attacks for PyTorch. If any client's update is visible on its own, block the deployment until secure aggregation or client-level DP is in place. [Pre-release privacy risk assessment](10-risk-assessment.md) records this verdict next to the other attacks.

## Cheat sheet

| Attack | Formula | Needs | Fails when |
| --- | --- | --- | --- |
| Exact recovery, one record | $x = (\nabla_{W_1}\ell)_{j,:}/(\nabla_{b_1}\ell)_j$ | Fully connected first layer with bias | Convolutional first layer; batches, in part |
| Label, one record | $y = \arg\min_k(\nabla_{b_2}\ell)_k$ | Softmax and cross-entropy | $p_y$ rounds to 1 |
| Deep Leakage from Gradients | Minimize the matching loss $R$ | An activation whose slope varies; the labels | Search stuck (final $R \gg 0$); larger batches |
| Isolation, batch | Isolated records recovered exactly | Fully connected first layer, ReLU units | Rarely on real data |

: The results of this chapter. {#tbl-gradinv-cheat}

**Rules of thumb**

- Secure aggregation needs a minimum cohort, and only client-level DP bounds what any update reveals.

## Further reading

- Zhu, Liu & Han, "Deep Leakage from Gradients" (NeurIPS 2019): the gradient-matching attack.
- Geiping, Bauermeister, Dröge & Moeller, "Inverting Gradients — How easy is it to break privacy in federated learning?" (NeurIPS 2020): cosine matching with an image prior, on trained networks, local steps and batches.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/06-gradient-inversion.md).
