# DP-SGD

## Why this matters

Nearly every deep-learning system with differential privacy (DP) uses the differentially private version of stochastic gradient descent (SGD), called DP-SGD: it clips per-record gradients and adds Gaussian noise at each step (Abadi et al. 2016). Clipping bounds each record's effect on the model; noise obscures that effect. An **accountant** composes step costs into privacy level $\varepsilon$ and slack $\delta$.

The accountant uses only the noise, sampling rate and number of steps; it does not inspect the model. These parameters determine $\varepsilon$, while clipping norm and learning rate can be tuned for accuracy. Most failures arise when code violates a proof assumption, so implementation checks are as important as parameter choices. [Train a model with DP-SGD](13-train-dp-sgd.md) demonstrates the full Opacus workflow.

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

This part's running example (@sec-dpsgd-run) is the run of [Train a model with DP-SGD](13-train-dp-sgd.md). A network with one hidden layer of 64 units learns from 30,000 people of the Adult census survey whether a person's income exceeds \$50K. It runs 300 steps with $\sigma = 0.973$, and each person joins a batch with probability $q = 1/30$, so a batch holds about 1,000 people. At the $\delta = 10^{-5}$ of the book's rule ([How small δ must be](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta)), the accountant Opacus uses gives $\varepsilon = 3.99$ for this run. The Rényi differential privacy (RDP) accountant, a standard and slightly loose upper bound, gives 4.50. To pick $\sigma$ for a target $\varepsilon$, let the accountant you will report solve for it ([Calibrate and account a DP-SGD run](../part04-privacy-accounting/10-calibrate-dp-sgd.md#sec-calib-plan)).

## The implementation checklist {#sec-dpsgd-checklist}

The guarantee requires four implementation properties. Violating one does not necessarily stop training or reporting.

First, one record may change the clipped sum by at most $C$. Batch normalization violates this condition because its statistics let one record affect other records' gradients. Use group or layer normalization instead; these operate on each record separately. With `strict=False`, Opacus's validator lists incompatible layers rather than raising an error.

```python
from opacus.validators import ModuleValidator
from torchvision.models import resnet18

model = resnet18(weights=None)
print(len(ModuleValidator.validate(model, strict=False)))  # 20
model = ModuleValidator.fix(model)
print(len(ModuleValidator.validate(model, strict=False)))  # 0
```

Set $C$ in advance using public data or a private estimate. Estimating $C$ from the current batch makes the noise data-dependent.

Second, match the accountant to the implemented sampler ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md#sec-sub-schemes)). Poisson accounting benefits from the chance that each record is absent from a batch. A loader that shuffles once per epoch and divides data into fixed batches includes every record exactly once per epoch, eliminating that amplification. Its true $\varepsilon$ may greatly exceed the Poisson estimate.

Fresh fixed-size batches can be accounted with a Poisson accountant using the batch's sampling rate and half the noise multiplier. A sampled record replaces another, so the sum can change by $2C$. In the running example, `dp_accounting`'s numerical accountant gives $\varepsilon=23.03$, compared with 3.98 for Poisson sampling; this bound uses add/remove neighbours.

Third, divide the noisy sum by the public expected batch size $B$, not the realized size. The realized size is one larger when the protected record is included, so dividing by it reveals that inclusion.

Fourth, account for every step that accessed the data, including discarded runs during [tuning](03-hyperparameter-tuning.md).

State the neighbouring relation with each $\varepsilon$, and make data-dependent preprocessing private. Use [user-level privacy](07-user-level.md) when one person contributes multiple records. For released models, enable Opacus's `secure_mode=True`; floating-point sampling does not produce the exact Gaussian noise assumed by the proof ([Discrete and floating-point-safe mechanisms](../part03-dp-foundations/08-discrete-mechanisms.md)).

## Clipping and the learning rate {#sec-dpsgd-clipping}

Clip each record's gradient separately. PyTorch's `torch.nn.utils.clip_grad_norm_` clips the batch gradient, which does not bound any single record's contribution and provides no privacy guarantee.

The clipping norm $C$ does not change $\varepsilon$, but it affects accuracy. Small $C$ clips more gradients, reducing the contribution of high-gradient records, which may be rare ([Disparate impact](08-disparate-impact.md)). Large $C$ increases noise because its standard deviation is $\sigma C$.

When all gradients are clipped to length $C$, the noise also scales with $C$. The update therefore depends on the product $\eta C$, where $\eta$ is the learning rate. If clipping is aggressive, tune this product. In the running example at $\varepsilon=4$, fixing $\eta C=2$ keeps test accuracy between 0.8446 and 0.8464 for $C$ from 0.1 to 10. Fixing $\eta=2$ instead gives accuracy 0.7869 at $C=10$.

Noise and clipping both affect the optimal learning rate. Noise lowers it ([Optimization](../../book-full/part01-prerequisites/06-optimization.md)); clipping raises it because gradients become shorter. Their effects do not cancel predictably. Retune the learning rate when $\sigma$ or $C$ changes, and include all search runs in the privacy cost.

Gradient norms change during training. In the running example, the median per-example norm on public rows falls from 1.444 to 0.044 over 300 steps. Thus, $C=1$ clips every row initially but only one quarter at the end. Adaptive clipping adjusts $C$ toward a gradient-norm quantile using a private count; Opacus enables it with `clipping="adaptive"`. This count consumes budget, increasing gradient noise at the same $\sigma$.

## Batch size at a fixed budget {#sec-dpsgd-batch}

Averaging the noisy sum over $B$ records divides the added noise by $B$, so larger batches dilute it. At a fixed budget and number of epochs, a larger batch also means fewer steps. So the noise multiplier must rise to keep $\varepsilon$, but more slowly than $B$ does. For a small sampling rate $q$ and a $\sigma$ well above 1, the accountant's cost over a fixed number of epochs depends on $q/\sigma^2$. So $\sigma$ only has to grow like $\sqrt q$. Since $q = B/n$, with $n$ the number of records, the noise on the averaged gradient falls with the batch size,
$$\sigma \propto \sqrt{\frac Bn}, \qquad \frac{\sigma C}{B} \propto \frac{C}{\sqrt{Bn}}.$$ {#eq-dpsgd-batch}

So choose the largest batch your hardware allows, with a larger learning rate for the fewer steps. When it does not fit in memory, Opacus's `BatchMemoryManager` splits the sampled batch into physical batches that fit, and still steps once per sampled batch.


## Making it fast {#sec-dpsgd-ghost}

Computed one record at a time, DP-SGD can run tens or hundreds of times slower than ordinary training. Two fixes cost no accuracy. Vectorized per-example gradients, from `torch.func.vmap` or Opacus's hooks, compute the same clipped sum in parallel. **Ghost clipping** also saves memory, because it gets each record's gradient norm from each layer's inputs and errors without the gradient itself. Opacus turns it on with `grad_sample_mode="ghost"`.

The third fix, microbatching, costs accuracy. It clips the average gradient of each group of $m$ records. One record can change its group's clipped average by up to $2C$, so the noise doubles. The update then averages over $B/m$ groups rather than $B$ records, so the noise on each coordinate of the averaged update becomes
$$2m\cdot\frac{\sigma C}{B}.$$ {#eq-dpsgd-microbatch}
On the running example at $\sigma = 0.973$, groups of 4 cut test accuracy from 0.8444 to 0.8206, more than a quarter of the way down to always predicting the larger class. So use microbatching only as a last resort.

## What privacy costs in accuracy {#sec-dpsgd-run}

@tbl-dpsgd-run trains the running example at five budgets with $C = 1$ and $\eta = 2$, ten times the best non-private learning rate on public data, and scores it on the 16,842 test rows. Always predicting the larger class, an income of at most \$50K, scores 0.759.

| | Noise multiplier | Test accuracy (mean ± standard deviation over 3 runs) |
| --- | --- | --- |
| $\varepsilon = 0.25$ | 8.125 | $0.7933 \pm 0.0200$ |
| $\varepsilon = 1$ | 2.383 | $0.8406 \pm 0.0023$ |
| $\varepsilon = 2$ | 1.431 | $0.8440 \pm 0.0014$ |
| $\varepsilon = 4$ | 0.973 | $0.8447 \pm 0.0010$ |
| $\varepsilon = 8$ | 0.722 | $0.8452 \pm 0.0012$ |
| Clipped, no noise | 0 | $0.8454 \pm 0.0005$ |
| Non-private SGD | — | $0.8449 \pm 0.0029$ |

: Test accuracy of the running example under DP-SGD at five budgets, with $\delta = 10^{-5}$ and Opacus's accountant. {#tbl-dpsgd-run}

Clipping without noise scores as well as non-private SGD here, so the whole loss comes from the noise, and it grows steeply as $\varepsilon$ shrinks. Going from $\varepsilon = 8$ to 1 costs half a point, while going from 1 to 0.25 costs almost 5, and the spread between runs grows eightfold. The noise grows in total length with the square root of the number of trained weights, while the clipped signal stays at most $C$, so a smaller model or [public pretraining](06-public-data.md) often helps.


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
