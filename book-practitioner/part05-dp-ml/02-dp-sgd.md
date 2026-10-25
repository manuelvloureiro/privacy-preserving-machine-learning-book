# DP-SGD

## Why this matters

Nearly every deep-learning system that satisfies differential privacy (DP) runs one algorithm. It is stochastic gradient descent (SGD) with each record's gradient clipped to a fixed length and Gaussian noise added at every step, called DP-SGD (Abadi et al. 2016). Clipping limits how far one record moves each step, whatever the model, and the noise hides that move ([Private empirical risk minimization](01-dp-erm.md)). An **accountant** then adds up the cost of every step into one privacy level, the $\varepsilon$ of the guarantee. It comes with $\delta$, a small additive slack that the guarantee allows on top of $\varepsilon$.

The accountant sees only the noise, the sampling rate (the chance that a record joins a batch) and the number of steps, never the model. So those three decide $\varepsilon$, and the clipping norm and the learning rate are free to be tuned for accuracy. Most failures come from code that breaks an assumption of the privacy proof, so the checks matter as much as the settings. [Train a model with DP-SGD](13-train-dp-sgd.md) does the whole job in Opacus.

## The algorithm and its guarantee {#sec-dpsgd-algorithm}

A training set holds $n$ records. The gradient of one record's loss is its **per-example gradient**, written $g_i$ for record $i$. DP-SGD repeats five operations at every step.

1. Sample a batch. Each record enters independently with probability $q$, the **sampling rate**, which is called Poisson sampling.
2. Compute the per-example gradient of each record in the batch.
3. Clip each gradient longer than the **clipping norm** $C$ to length $C$.
4. Sum the clipped gradients, and add Gaussian noise with standard deviation $\sigma C$ to each coordinate of the sum, where the **noise multiplier** $\sigma$ measures the noise in units of $C$.
5. Divide the noisy sum by the expected batch size $B = qn$, a fixed public number, and take a step in the opposite direction, scaled by the **learning rate** $\eta$.

Write $\theta_t$ for the weights at step $t$ and $S_t$ for its batch. The Gaussian noise $\xi_t$ is drawn afresh, independently on each coordinate, and the update is
$$\theta_{t+1} = \theta_t - \frac{\eta}{B}\Big(\sum_{i\in S_t}g_i\min\Big(1, \frac{C}{\|g_i\|}\Big) + \xi_t\Big), \qquad \xi_t \sim \mathcal N(0, \sigma^2C^2I).$$
Releasing every intermediate set of weights then satisfies $(\varepsilon, \delta)$-DP for **add/remove neighbours**, datasets that differ because one has a record the other lacks ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md)).

This part's running example trains a small network on 10,000 synthetic records (@sec-dpsgd-run). It runs 10,000 steps with $\sigma = 1$, and each record joins a batch with probability 0.01. The Rényi differential privacy (RDP) accountant, a standard upper bound that is slightly loose, gives it $\varepsilon = 6.71$ at $\delta = 10^{-5}$. To pick $\sigma$ for a target $\varepsilon$ instead, let the accountant you will report solve for it. [Train a model with DP-SGD](13-train-dp-sgd.md#sec-trainwf-calibrate) does this in one Opacus call ([Calibrate and account a DP-SGD run](../part04-privacy-accounting/10-calibrate-dp-sgd.md)).

Choose $\delta$ by the book's default, $\delta \le 1/n^{1.1}$ rounded down to a power of ten. That keeps it safely below $1/n$ ([Choosing ε](09-choosing-epsilon.md)). Here $n$ counts people, and for the running example's 10,000 records, one per person, the rule gives its $10^{-5}$.

## The implementation checklist {#sec-dpsgd-checklist}

The guarantee rests on four properties of your code, and code that breaks one still trains and reports as before.

First, one record must move the clipped sum by at most $C$. Batch normalization breaks this, because its statistics let one record change every other record's gradient. So private models use group or layer normalization, which normalize each record on its own. With `strict=False`, Opacus's validator lists the layers to replace instead of raising an error.

```python
from opacus.validators import ModuleValidator
from torchvision.models import resnet18

model = resnet18(weights=None)
print(len(ModuleValidator.validate(model, strict=False)))  # 20
model = ModuleValidator.fix(model)
print(len(ModuleValidator.validate(model, strict=False)))  # 0
```

The clipping norm must also be fixed in advance, from public data or from a private estimate. A $C$ read off the current batch makes the size of the noise depend on the data, and that dependence leaks information the accountant does not count.

Second, the accountant must assume the sampler the code actually runs ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md#sec-sub-schemes)). The Poisson accountant's saving comes from each record's chance of being left out. A loader that shuffles once per epoch and cuts fixed batches puts every record in exactly one batch per epoch. That chance is then gone, and the true $\varepsilon$ can be far above the Poisson number.

Batches of a fixed size, drawn afresh at every step, can still use a Poisson accountant. Give it the batch's share of the data as the rate and half your noise multiplier, because a record that enters a fixed-size batch pushes another out, so the sum can move by $2C$. That is expensive, since halving $\sigma$ raises the running example's $\varepsilon$ from 6.71 to 47.42. The rule holds only for add/remove neighbours, and Part 4 gives what to account for under the others.

Third, divide the noisy sum by the public number $B$, never by the actual batch size. The actual size is one larger when the protected record is drawn, so dividing by it reveals whether it was, and the accountant does not count that leak.

Fourth, the accountant must count every step that touched the data, including those of runs discarded during [tuning](03-hyperparameter-tuning.md).

Beyond the proof, state the neighbouring relation next to every $\varepsilon$, and make data-dependent preprocessing private too. The guarantee covers one record, so a person with many records needs [user-level privacy](07-user-level.md). And the proof assumes exact Gaussian noise, which ordinary floating-point sampling does not produce, so turn on Opacus's `secure_mode=True` for a model you release ([Discrete and floating-point-safe mechanisms](../part03-dp-foundations/08-discrete-mechanisms.md)).

## Clipping and the learning rate {#sec-dpsgd-clipping}

Clipping must happen per record. PyTorch's `torch.nn.utils.clip_grad_norm_` clips the gradient of a whole batch, which bounds nothing about any single record, so it gives no privacy at any threshold.

The clipping norm does not change $\varepsilon$, but it sets the two ways clipping costs accuracy. A small $C$ clips more gradients, so records with large gradients count for less, and they are often rare records ([Disparate impact](08-disparate-impact.md)). A large $C$ instead means more noise, since the noise is $\sigma C$. When every gradient is clipped, each has length exactly $C$, and the noise is proportional to $C$ too. So only the product $\eta C$ of the learning rate and the clipping norm matters. The practical recipe is to clip hard and tune that one number. On the running example at $\sigma = 1$, holding $\eta C$ at 0.25 keeps test accuracy at 0.9599 or above for every $C$ up to 0.3. With the learning rate held at 0.5 instead, $C = 10$ drops it to 0.8347.

Noise also changes the best learning rate. On the logistic problem of [Optimization](../part01-prerequisites/06-optimization.md), adding noise moves the best learning rate on a grid down from 0.178 to 0.056. Clipping pulls the other way, because clipped gradients are shorter, which is why private runs often start from a step several times the non-private one. The two effects do not cancel predictably. So search the step again whenever $\sigma$ or $C$ changes, and count the search's runs in $\varepsilon$.

Gradient norms also drift. The running example's median per-example norm falls from 3.592 to 0.006 in a thousand steps, so a $C$ that clips everything early clips only a minority later. Adaptive clipping is the private estimate that the checklist allows, since it moves $C$ toward a quantile of the norms measured by a small private count. Opacus offers it with `clipping="adaptive"`, but this is not a drop-in switch. It takes five more parameters, such as `target_unclipped_quantile`, the share of gradients to leave unclipped. And because part of the noise budget goes to the count, the gradient gets more noise at the same $\sigma$.

## Batch size at a fixed budget {#sec-dpsgd-batch}

Averaging the noisy sum over $B$ records divides the added noise by $B$, so larger batches dilute it. At a fixed budget and number of epochs, a larger batch means fewer steps, and the noise multiplier must rise to keep $\varepsilon$, but more slowly than $B$ does. For a small sampling rate and a noise multiplier well above 1, the accountant's cost at a fixed number of epochs depends on $q/\sigma^2$, so the noise multiplier only has to grow like $\sqrt q$. Since $q$ is $B/n$, the noise on the averaged gradient falls with the batch size,
$$\sigma \propto \sqrt{\frac Bn}, \qquad \frac{\sigma C}{B} \propto \frac{C}{\sqrt{Bn}}.$$ {#eq-dpsgd-batch}
Small batches are worse still, because they need a noise multiplier near or below 1, where subsampling saves much less.

So choose the largest batch your hardware allows, with a larger learning rate for the fewer steps. When the batch does not fit in memory, Opacus's `BatchMemoryManager` splits the logical batch, the one the sampler draws, into physical batches that fit. It still steps once per logical batch, so the sampling rate and the accounting stay the same.

## Making it fast {#sec-dpsgd-ghost}

Computed one record at a time, DP-SGD can run tens or hundreds of times slower than ordinary training. Of three fixes, the first two are free. Vectorized per-example gradients, from `torch.func.vmap` or Opacus's hooks, compute the same clipped sum in parallel. **Ghost clipping** also saves memory, because it gets each record's gradient norm from each layer's inputs and back-propagated errors without forming the gradient. Opacus turns it on with `grad_sample_mode="ghost"`, and `make_private` then also returns a loss function to train with.

Microbatching is the costly fix. It clips the average gradient of each fixed group of $m$ records instead of each record's gradient. One record can then change its group's clipped average by up to $2C$, so the noise must double. And since the update divides it by the $B/m$ groups rather than by $B$ records, the noise on each coordinate of the averaged update becomes
$$2m\cdot\frac{\sigma C}{B}.$$ {#eq-dpsgd-microbatch}
On the running example at $\sigma = 1$, groups of 4 cut test accuracy from 0.9589 with per-example clipping to 0.9336, so use it only as a last resort.

## What privacy costs in accuracy {#sec-dpsgd-run}

The running example is a network with one hidden layer of 64 tanh units. @tbl-dpsgd-run trains it at four budgets, with $C$ and $\eta$ both 0.5.

| | Noise multiplier | Test accuracy (mean ± standard deviation over 3 runs) |
| --- | --- | --- |
| $\varepsilon = 0.5$ | 7.72 | $0.8209 \pm 0.0180$ |
| $\varepsilon = 1$ | 4.13 | $0.9175 \pm 0.0161$ |
| $\varepsilon = 4$ | 1.36 | $0.9548 \pm 0.0019$ |
| $\varepsilon = 8$ | 0.917 | $0.9604 \pm 0.0019$ |
| Clipped, no noise | 0 | $0.9649 \pm 0.0009$ |
| Non-private SGD (1,000 steps) | — | $0.9610 \pm 0.0005$ |

: Test accuracy of the running example under DP-SGD at four budgets, with $\delta = 10^{-5}$ and the RDP accountant. {#tbl-dpsgd-run}

Clipping without noise scores slightly above non-private SGD here, so the whole loss comes from the noise, and it grows steeply as $\varepsilon$ shrinks. Going from $\varepsilon = 8$ to 4 costs half a point, while halving it from 1 to 0.5 costs almost 10. The total length of the noise on a step also grows with the square root of the number of trained weights. The clipped signal cannot exceed $C$, so a smaller model or [public pretraining](06-public-data.md) often helps.

## Cheat sheet

| Setting | Start with | Check |
| --- | --- | --- |
| $\delta$ | $1/n^{1.1}$ rounded down to a power of ten | $n$ counts people |
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
