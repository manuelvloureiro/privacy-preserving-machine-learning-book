# Gradient inversion

## Why this matters

In **federated learning (FL)**, a server sends the current model to each client. Each client computes the loss gradient on its records and returns the gradient; the server averages the gradients and updates the model. This arrangement is sometimes described as private because raw data remain on the device.

However, a client's gradient may reveal its data because many gradient components are simple functions of the input. Averaging over a batch provides limited protection (@sec-gradinv-batch). Treat each client update as sensitive as the underlying data.

## The attacker and what it sees {#sec-gradinv-setting}

The server knows the model parameters and sees each client's update. Its access is the training **transcript** defined in [Threat models and privacy games](01-threat-models.md). First assume the server is **honest-but-curious**: it follows the protocol but analyses all received information. A client with a batch of $B$ records sends the mean gradient. Let $\ell_i$ be the loss on record $i$, which depends on model parameters $\theta$. The client sends
$$g = \frac1B\sum_{i=1}^B\nabla_\theta\ell_i.$$

## Exact recovery from a fully connected layer {#sec-gradinv-exact}

For a fully connected first layer with a bias, one division recovers the input. Let $W_1$ and $b_1$ be the weights and bias. Unit $j$ multiplies input $x$ by its weight row and adds its bias. Let $u_j$ be the loss derivative with respect to the unit's output. The chain rule gives the weight and bias gradients
$$\nabla_{W_1}\ell = u\,x^\top, \qquad \nabla_{b_1}\ell = u.$$ {#eq-gradinv-outer}
Thus, row $j$ of the weight gradient is $x$ scaled by $u_j$, and unit $j$'s bias gradient is $u_j$. For one record, their ratio recovers the input exactly whenever the bias gradient is nonzero:
$$x = \frac{(\nabla_{W_1}\ell)_{j,:}}{(\nabla_{b_1}\ell)_j}.$$ {#eq-gradinv-ratio}

The network depth, loss and label affect $u_j$, which cancels in the ratio. The attack therefore requires neither optimization nor prior knowledge of the input. The snippet simulates one client using an image from scikit-learn's digits dataset in PyTorch. Its final line recovers the label, as described in @sec-gradinv-labels.

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

The snippet uses the unit with the largest bias gradient because an inactive ReLU unit has zero gradient; one active unit suffices. A convolutional first layer does not permit this division. It reuses each filter across image positions, so its gradient sums products from multiple patches. The attacker must then use a search method (@sec-gradinv-dlg).

## Label recovery from the last layer {#sec-gradinv-labels}

The label, such as a diagnosis, may be the most sensitive part of a record. It can be recovered from the last layer without search (Zhao, Mopuri & Bilen 2020) when that layer has a bias and feeds a softmax with cross-entropy loss. Let $p$ be the predicted probabilities and $e_y$ the vector with 1 at the true label $y$. The last-layer bias gradient $b_2$ is $p-e_y$. Since each entry of $p$ is between 0 and 1, only the true-label entry is negative. For one record,
$$y = \arg\min_k(\nabla_{b_2}\ell)_k.$$ {#eq-gradinv-labels}
This is the snippet's `argmin`. If the model is so confident that $p_y$ rounds to 1, the true-label gradient becomes zero and the rule no longer identifies a negative entry.

A batch hides the label of each record but may reveal each class count. Each bias-gradient entry equals that class's average predicted probability minus its fraction of the batch. A server that estimates the average prediction can recover the class counts. In the full edition's experiment, all counts were exact for every batch of 4 and for 52.5% of batches of 64.

## Deep Leakage from Gradients {#sec-gradinv-dlg}

When division does not apply, the attacker can use **Deep Leakage from Gradients (DLG)** (Zhu, Liu & Han 2019). DLG guesses inputs, computes their gradients and adjusts the inputs until the average gradient matches the observed gradient $g$. Let $x_i'$ be the guessed inputs, $y_i$ their labels (recovered from the last layer), and $\ell(\theta;x',y)$ the loss of a guess. The attack minimizes the **matching loss** $R$, the squared distance between the guessed mean gradient and $g$:
$$\hat x_1, \dots, \hat x_B = \arg\min_{x_1', \dots, x_B'}R(x'), \qquad R(x') = \Big\|\frac1B\sum_{i=1}^B\nabla_\theta\ell(\theta; x_i', y_i) - g\Big\|^2.$$ {#eq-gradinv-dlg}
In PyTorch, `create_graph=True` in `torch.autograd.grad` makes the guessed gradient differentiable. An optimizer such as Adam can then follow the gradient of $R$ with respect to the input.

Two conditions affect the search. First, the activation's slope must vary because the search differentiates a gradient. ReLU has constant slope on either side of zero, so Zhu et al. used sigmoid. Second, $R$ is nonconvex and the search may get stuck. Stop when $R$ no longer decreases, then compare it with the observed gradient's squared norm $\|g\|^2$. In double precision, a successful run in the full edition reaches about $10^{-27}$, while a stuck run reaches about $10^{-5}$. Because other inputs may also give small $R$, use several random restarts and accept a reconstruction only when the results agree.

## Batches and averaging {#sec-gradinv-batch}

Clients usually send a batch average or a parameter change after several on-device training steps (**local steps**). Averaging prevents the direct division and slows search, but preserves structure. Each record adds an outer product to the first-layer gradient, so each gradient row is a combination of the $B$ inputs. The server therefore learns their **span**, the set of their linear combinations, which contains every record.

The division still recovers a record exactly if a ReLU unit activates only for that record; other records then contribute nothing to its gradient row. Such a unit **isolates** the record. Isolation is more common than a count of random units predicts because similar images may activate the same units. In the full edition's network with 1,024 hidden units, a naive estimate predicts exposure for 1.6% of records in a batch of 16; the measured share is 69.1% (@tbl-gradinv-isolation). Larger batches reduce this share, while wider first layers increase it. Measure the risk at the deployed width and batch size, and treat any isolated record as exposed.

| Hidden units | Batch size | Share of the batch recovered exactly |
| --- | --- | --- |
| 64 | 4 | 0.838 |
| 64 | 8 | 0.550 |
| 64 | 16 | 0.144 |
| 1,024 | 16 | 0.691 |

: Isolation attack on synthetic images, from the full edition. {#tbl-gradinv-isolation}

Batches make DLG less reliable: in the full edition's sigmoid-network experiment, the median search on a batch of 4 gets stuck. However, the isolation attack still recovered all four images in such a batch. Local training steps also fail to protect a single record. At each step, the first-layer gradient is an outer product with the same input, so the ratio still recovers that input from the total parameter change. A **malicious** server that chooses the weights can plant units that activate only on rare inputs. Each unit can then isolate one record, allowing recovery of most records in a batch of any size (Boenisch et al. 2023, Fowl et al. 2022).

## What to do as a deployer {#sec-gradinv-defences}

The risk of an FL deployment depends on who can see each client's update (@tbl-gradinv-decisions).

| Setting | What leaks to the server | What to do |
| --- | --- | --- |
| One record per update, fully connected first layer with bias | The input exactly, and the label | Never deploy without protection |
| Batches of any size, or several local steps | Label counts, the span of the batch, isolated records, and the rest by search in small batches | Treat each update as raw data |
| Server not trusted to choose weights | Most records of any batch | Batch size is no defence; treat each update as raw data |
| Updates protected by differential privacy (DP) | Bounded per client by the DP guarantee | The only protection with a guarantee |

: Gradient leakage by FL setting, and the response. {#tbl-gradinv-decisions}

**Secure aggregation** lets the server see only the sum of updates from multiple clients, preventing the single-client access required by the attacks above. A sum from a very small cohort may still reveal an individual update. Set a minimum cohort size and prevent client dropouts from reducing the sum below it.

Secure aggregation alone does not provide a formal guarantee. The aggregate remains a batch average that may be vulnerable to the attacks above, and a malicious server may still choose the model weights. Add DP at the client level for a guarantee: clip each client's update to a fixed maximum norm and add noise to the sum. This bounds the effect of any one client's data ([User-level privacy](../part05-dp-ml/07-user-level.md)). Flower and TensorFlow Federated offer both protections. Gradient pruning and similar defences may hinder attacks but provide no guarantee.

Assess whether the server sees individual updates or only aggregates, and identify the minimum cohort size. Run the attacks using the deployed architecture, batch size and number of local steps. Geiping's open-source PyTorch framework `breaching` implements these attacks. If the server can see an individual update, block deployment until secure aggregation or client-level DP is in place. Record the decision in [Pre-release privacy risk assessment](10-risk-assessment.md).

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

## References

- Zhu, Liu & Han, "Deep Leakage from Gradients" (NeurIPS 2019): the gradient-matching attack.
- Geiping, Bauermeister, Dröge & Moeller, "Inverting Gradients — How easy is it to break privacy in federated learning?" (NeurIPS 2020): cosine matching with an image prior, on trained networks, local steps and batches.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/06-gradient-inversion.md).
