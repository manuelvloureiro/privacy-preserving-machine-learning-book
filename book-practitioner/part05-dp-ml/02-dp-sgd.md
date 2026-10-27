# DP-SGD

## Why this matters

Nearly every deep-learning system that satisfies differential privacy (DP) runs one algorithm. It is stochastic gradient descent (SGD) with clipped per-record gradients and Gaussian noise at every step, called DP-SGD (Abadi et al. 2016). Clipping limits how far one record moves the model at each step, and the noise hides that move. An **accountant** adds up the cost of the steps into the guarantee's $\varepsilon$, with a small additive slack $\delta$.

The accountant sees only the noise, the sampling rate (the chance that a record joins a batch) and the number of steps, never the model. So those three decide $\varepsilon$, and you can tune the clipping norm and learning rate for accuracy alone. Most failures come from code that breaks an assumption of the privacy proof, so the checks matter as much as the settings. [Train a model with DP-SGD](13-train-dp-sgd.md) does the whole job in Opacus.

## The algorithm and its guarantee {#sec-dpsgd-algorithm}

A training set holds $n$ records. The gradient of one record's loss is its **per-example gradient**, written $g_i$ for record $i$. Each step of DP-SGD does five things.

1. Sample a batch. Each record enters independently with probability $q$, the **sampling rate**, which is called Poisson sampling.
2. Compute the per-example gradient of each record in the batch.
3. Clip each gradient longer than the **clipping norm** $C$ to length $C$.
4. Sum the clipped gradients, and add Gaussian noise with standard deviation $\sigma C$ to each coordinate, where the **noise multiplier** $\sigma$ measures the noise in units of $C$.
5. Divide the noisy sum by the expected batch size $B = qn$, a fixed public number, and step against it, scaled by the **learning rate** $\eta$.

Write $\theta_t$ for the weights at step $t$ and $S_t$ for its batch. With Gaussian noise $\xi_t$ drawn afresh on each coordinate, the update is
$$\theta_{t+1} = \theta_t - \frac{\eta}{B}\Big(\sum_{i\in S_t}g_i\min\Big(1, \frac{C}{\|g_i\|}\Big) + \xi_t\Big), \qquad \xi_t \sim \mathcal N(0, \sigma^2C^2I).$$
Releasing every intermediate set of weights then satisfies $(\varepsilon, \delta)$-DP for **add/remove neighbours**, datasets that differ by one record ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md)).

This part's running example (@sec-dpsgd-run) trains a small network on 10,000 synthetic records. It runs 10,000 steps with $\sigma = 1$, and each record joins a batch with probability $q = 0.01$. For this run, the Rényi differential privacy (RDP) accountant, a standard and slightly loose upper bound, gives $\varepsilon = 6.71$ at the $\delta = 10^{-5}$ of the book's rule ([How small δ must be](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta)). To pick $\sigma$ for a target $\varepsilon$, let the accountant you will report solve for it ([Calibrate and account a DP-SGD run](../part04-privacy-accounting/10-calibrate-dp-sgd.md#sec-calib-plan)).

## The implementation checklist {#sec-dpsgd-checklist}

The guarantee rests on four properties of your code. Code that breaks one still trains and reports as before.

First, one record must move the clipped sum by at most $C$. Batch normalization breaks this, because its statistics let one record change every other record's gradient. So private models use group or layer normalization, which works on each record alone. With `strict=False`, Opacus's validator lists the layers to replace instead of raising an error.

```python
from opacus.validators import ModuleValidator
from torchvision.models import resnet18

model = resnet18(weights=None)
print(len(ModuleValidator.validate(model, strict=False)))  # 20
model = ModuleValidator.fix(model)
print(len(ModuleValidator.validate(model, strict=False)))  # 0
```

Also fix $C$ in advance, from public data or a private estimate. A $C$ read off the current batch makes the noise depend on the data.

Second, the accountant must assume the sampler the code actually runs ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md#sec-sub-schemes)). The Poisson accountant's saving comes from each record's chance of being left out of a batch. A loader that shuffles once per epoch and cuts fixed batches puts every record in exactly one batch per epoch. That chance is then gone, and the true $\varepsilon$ can be far above the Poisson number.

Fixed-size batches drawn afresh at every step can still use a Poisson accountant, at the batch's share of the data and half your noise multiplier. The noise counts as halved because a record that enters a fixed-size batch pushes another out, so the sum can move by $2C$. This raises the running example's $\varepsilon$ from 6.71 to 47.42, and it holds only for add/remove neighbours.

Third, divide the noisy sum by the public number $B$, never by the actual batch size. The actual size is one larger when the protected record is drawn, so dividing by it reveals whether it was.

Fourth, the accountant must count every step that touched the data, including those of runs discarded in [tuning](03-hyperparameter-tuning.md).

Beyond the proof, state the neighbouring relation next to every $\varepsilon$, and make data-dependent preprocessing private too. A person with many records needs [user-level privacy](07-user-level.md). For a model you release, turn on Opacus's `secure_mode=True`, because floating-point sampling is not the exact Gaussian noise of the proof ([Discrete and floating-point-safe mechanisms](../part03-dp-foundations/08-discrete-mechanisms.md)).

## Clipping and the learning rate {#sec-dpsgd-clipping}

Clipping must happen per record. PyTorch's `torch.nn.utils.clip_grad_norm_` clips a whole batch's gradient, which bounds no single record and so gives no privacy at any threshold.

The clipping norm $C$ does not change $\varepsilon$, but it costs accuracy either way. A small $C$ clips more gradients, so records with large gradients count for less, and they are often rare records ([Disparate impact](08-disparate-impact.md)). A large $C$ means more noise, since the noise is $\sigma C$.

When every gradient is clipped, each has length exactly $C$ and the noise is proportional to $C$ too. So the step depends only on the product $\eta C$ of the learning rate and the clipping norm. Clip hard, and tune that one number. On the running example, holding $\eta C$ at 0.25 keeps test accuracy at 0.9599 or above for every $C$ up to 0.3. Holding $\eta$ at 0.5 instead, $C = 10$ drops it to 0.8347.

Noise and clipping also move the best learning rate. Noise lowers it ([Optimization](../part01-prerequisites/06-optimization.md)), while clipping raises it, because clipped gradients are shorter. The two do not cancel predictably, so search the step again whenever $\sigma$ or $C$ changes, and count the search's runs in $\varepsilon$.

Gradient norms also drift. The running example's median per-example norm falls from 3.592 to 0.006 in a thousand steps, so a $C$ that clips everything early clips only a minority later. Adaptive clipping moves $C$ toward a quantile of the norms, measured by a small private count, and Opacus offers it with `clipping="adaptive"`. The count spends part of the budget, so the gradient gets more noise at the same $\sigma$.

## Batch size at a fixed budget {#sec-dpsgd-batch}

Averaging the noisy sum over $B$ records divides the added noise by $B$, so larger batches dilute it. At a fixed budget and number of epochs, a larger batch also means fewer steps. So the noise multiplier must rise to keep $\varepsilon$, but more slowly than $B$ does. For a small sampling rate $q$ and a $\sigma$ well above 1, the accountant's cost over a fixed number of epochs depends on $q/\sigma^2$. So $\sigma$ only has to grow like $\sqrt q$. Since $q = B/n$, with $n$ the number of records, the noise on the averaged gradient falls with the batch size,
$$\sigma \propto \sqrt{\frac Bn}, \qquad \frac{\sigma C}{B} \propto \frac{C}{\sqrt{Bn}}.$$ {#eq-dpsgd-batch}

So choose the largest batch your hardware allows, with a larger learning rate for the fewer steps. When it does not fit in memory, Opacus's `BatchMemoryManager` splits the sampled batch into physical batches that fit, and still steps once per sampled batch.


## Making it fast {#sec-dpsgd-ghost}

Computed one record at a time, DP-SGD can run tens or hundreds of times slower than ordinary training. Two fixes cost no accuracy. Vectorized per-example gradients, from `torch.func.vmap` or Opacus's hooks, compute the same clipped sum in parallel. **Ghost clipping** also saves memory, because it gets each record's gradient norm from each layer's inputs and errors without the gradient itself. Opacus turns it on with `grad_sample_mode="ghost"`.

The third fix, microbatching, costs accuracy. It clips the average gradient of each group of $m$ records. One record can change its group's clipped average by up to $2C$, so the noise doubles. The update then averages over $B/m$ groups rather than $B$ records, so the noise on each coordinate of the averaged update becomes
$$2m\cdot\frac{\sigma C}{B}.$$ {#eq-dpsgd-microbatch}
On the running example at $\sigma = 1$, groups of 4 cut test accuracy from 0.9589 to 0.9336. So use microbatching only as a last resort.

## What privacy costs in accuracy {#sec-dpsgd-run}

@tbl-dpsgd-run trains the running example, a network with one hidden layer of 64 tanh units, at four budgets with $C$ and $\eta$ both 0.5.


| | Noise multiplier | Test accuracy (mean ± standard deviation over 3 runs) |
| --- | --- | --- |
| $\varepsilon = 0.5$ | 7.72 | $0.8209 \pm 0.0180$ |
| $\varepsilon = 1$ | 4.13 | $0.9175 \pm 0.0161$ |
| $\varepsilon = 4$ | 1.36 | $0.9548 \pm 0.0019$ |
| $\varepsilon = 8$ | 0.917 | $0.9604 \pm 0.0019$ |
| Clipped, no noise | 0 | $0.9649 \pm 0.0009$ |
| Non-private SGD (1,000 steps) | — | $0.9610 \pm 0.0005$ |

: Test accuracy of the running example under DP-SGD at four budgets, with $\delta = 10^{-5}$ and the RDP accountant. {#tbl-dpsgd-run}

Clipping without noise scores slightly above non-private SGD here, so the whole loss comes from the noise, and it grows steeply as $\varepsilon$ shrinks. Going from $\varepsilon = 8$ to 4 costs half a point, while going from 1 to 0.5 costs almost 10. The noise grows in total length with the square root of the number of trained weights, while the clipped signal stays at most $C$, so a smaller model or [public pretraining](06-public-data.md) often helps.


## Cheat sheet

| Setting | Start with | Check |
| --- | --- | --- |
| $\delta$ | The book's rule | $n$ counts people |
| Batch size | 256 to 4,096, as large as fits | Noise on the average is $\sigma C/B$ |
| Epochs | Fixed before calibrating; Adult uses 10 | $\sigma$ recomputed if they change |
| Noise multiplier | From the accountant you report | A second accountant gives the same $\varepsilon$ |
| Clipping norm | 1, or the median norm on public data | Per record, never `clip_grad_norm_` |
| Optimizer, learning rate | SGD with momentum 0.9; several times the non-private learning rate (ten on Adult), chosen on public data | Retuned when $\sigma$ or $C$ changes |
| Sampler | Poisson | Fixed-size batches: account at $\sigma/2$ (add/remove only) |
| Report | $\varepsilon$, $\delta$, neighbours, accountant | Every run counted |

: DP-SGD starting settings and checks. {#tbl-dpsgd-cheat}

**Rules of thumb**

- Fix $\varepsilon$, $\delta$ and the number of epochs first, since they are the contract. Then choose the batch, let the accountant set $\sigma$, and tune the learning rate last.
- Find the $\varepsilon$ below which accuracy collapses on your task before you promise a budget.

## References

- Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016): DP-SGD and the moments accountant.
- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023): what to clip, how to tune and what to report.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/02-dp-sgd.md).
