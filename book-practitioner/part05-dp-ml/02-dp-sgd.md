# DP-SGD

## Why this matters

Nearly every deep-learning system that satisfies differential privacy (DP) runs one algorithm. It is stochastic gradient descent (SGD) with each example's gradient clipped to a fixed length and Gaussian noise added to every step, called DP-SGD (Abadi et al. 2016). Clipping bounds one record's influence for any model ([Private empirical risk minimization](01-dp-erm.md)). An **accountant** then adds up the cost of every step into one privacy level $\varepsilon$ with a small slack $\delta$.

The privacy analysis never looks inside the model, so the settings split in two. The noise, the sampling rate and the number of steps decide $\varepsilon$, while the clipping norm and the step size are free to be tuned for accuracy. Most failures come from code that silently breaks the proof, so a checklist matters as much as the settings.

## The algorithm and its guarantee {#sec-dpsgd-algorithm}

A training set holds $n$ records, and the gradient of one record's loss is its **per-example gradient**, written $g_i$ for record $i$. DP-SGD repeats five operations at every step.

1. Sample a batch. Each record enters independently with probability $q$, the **sampling rate**, which is called Poisson sampling.
2. Compute the per-example gradient of each record in the batch.
3. Clip each gradient longer than the **clipping norm** $C$ to length $C$.
4. Add Gaussian noise with standard deviation $\sigma C$ to each coordinate of the sum, where the **noise multiplier** $\sigma$ measures the noise in units of $C$.
5. Divide by the expected batch size, the fixed number $B = qn$, and step against the result, scaled by the step size $\eta$.

Write $w_t$ for the weights at step $t$ and $S_t$ for its batch. Then the update, with fresh Gaussian noise $\xi_t$, is
$$w_{t+1} = w_t - \frac{\eta}{B}\Big(\sum_{i\in S_t}g_i\min\Big(1, \frac{C}{\|g_i\|}\Big) + \xi_t\Big), \qquad \xi_t \sim \mathcal N(0, \sigma^2C^2I).$$
Releasing every intermediate set of weights is then $(\varepsilon, \delta)$-DP for **add/remove neighbours**, datasets that differ because one has a record the other lacks. The Rényi differential privacy (RDP) accountant computes $\varepsilon$ from the sampling rate, the noise multiplier and the number of steps alone ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md)).

This part's running example takes 10,000 steps with $q = 0.01$ and $\sigma = 1$, which costs $\varepsilon = 6.71$ at $\delta = 10^{-5}$. To pick $\sigma$ for a target, fix the rate and the number of steps and let the accountant solve for it. With the same settings, $\varepsilon = 1$ needs a noise multiplier of 4.13.

## The implementation checklist {#sec-dpsgd-checklist}

The guarantee rests on four conditions, each a property of your code that a reviewer can check. Code that breaks one still trains and reports as before.

First, one record must move the clipped sum by at most $C$. Batch normalization breaks this, because its statistics let one record change every other record's gradient, so private models use group or layer normalization instead. The clipping norm must also be fixed in advance, since a $C$ set from the current batch makes the noise depend on the data.

Second, the accountant must assume the sampler the code actually runs ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md#sec-sub-schemes)). Many data loaders shuffle once per epoch and cut the data into fixed batches. A record is then never absent from an epoch, and the true $\varepsilon$ can be far above the Poisson number. Fixed-size batches drawn afresh at every step are covered under add/remove if you give a Poisson accountant the batch's fraction of the data as its rate and half your noise multiplier.

Third, divide the noisy sum by the public number $B$, never by the actual batch size, because the actual size changes when the target record is added.

Fourth, the accountant must count every step that touched the data, including those of runs discarded during [tuning](03-hyperparameter-tuning.md).

Beyond the proof, state the neighbouring relation next to every $\varepsilon$, since under replacement one record can move the sum by $2C$. Preprocessing that looks at the data, such as normalizing by the mean, must itself be private. And a person with many records needs [user-level privacy](07-user-level.md).

## Clipping and the step size {#sec-dpsgd-clipping}

The clipping norm leaves privacy alone but controls both of its costs. A small $C$ clips more gradients, so records with large gradients count for less, and they are often rare examples ([Disparate impact](08-disparate-impact.md)). A large $C$ instead means more noise, since the noise is $\sigma C$.

When every gradient is clipped, only the product $\eta C$ of the step size and the clipping norm matters. So the practical recipe is to clip hard and tune one number. On the running example, holding $\eta C$ at 0.25 keeps test accuracy at 0.9599 or above for every $C$ up to 0.3. With the step size held fixed instead, a clipping norm of 10 drops accuracy to 0.8347.

What goes wrong is that gradient norms drift. The median per-example gradient norm of the running example falls from 3.592 to 0.006 in a thousand steps, so a $C$ that clips everything early clips only a minority later. Adaptive clipping, which tracks a quantile of the norms privately, follows the drift for a small share of the budget, and Opacus offers it as `clipping="adaptive"`.

## Batch size at a fixed budget {#sec-dpsgd-batch}

Suppose the privacy budget and the number of epochs, passes over the data, are fixed. Should you take many small steps or a few large ones? For a small sampling rate and a noise multiplier well above 1, the noise multiplier must grow like $\sqrt q$. Since $q$ is $B/n$, the noise on the averaged gradient falls with the batch size,
$$\sigma \propto \sqrt{\frac Bn}, \qquad \frac{\sigma C}{B} \propto \frac{C}{\sqrt{Bn}}.$$ {#eq-dpsgd-batch}
Small batches are even worse than this law says, because they need a noise multiplier near or below 1, where each step costs more than the law assumes. For example, take a budget of $\varepsilon = 4$ over 100 epochs. Going from a batch of 10 to one of 1,000 then cuts the noise per averaged gradient by a factor of 18.6, where the law predicts 10.

Fewer steps are the price, and a larger step size pays it. At scale, De et al. (2022) and Anil et al. (2022) report that very large batches are a main ingredient of accurate models. So choose the largest batch that your hardware and your epoch count allow, and scale the step size up with it.

## Making it fast {#sec-dpsgd-ghost}

Computed one record at a time, DP-SGD can run up to two orders of magnitude slower than ordinary training. Of the three fixes, two are free. Vectorized per-example gradients, through `torch.func.vmap` with `grad`, JAX's `vmap` or Opacus's backward hooks, compute exactly the same clipped sum in parallel, so they change only the running time. **Ghost clipping** saves memory as well. A dense layer's per-example gradient is the outer product of the layer's input and the error flowing back into it. Its length is therefore the product of their lengths, so the norms come without forming the gradients.

Microbatching is the costly fix. It averages fixed groups of records and clips the averages, but one record can then flip its group's clipped average, so the sensitivity doubles to $2C$. Write $m$ for the number of records per group. The noise on each coordinate of the averaged update then becomes
$$2m\cdot\frac{\sigma C}{B}$$ {#eq-dpsgd-microbatch}
at the same $\varepsilon$. On the running example, groups of 4 cut test accuracy from 0.9589 to 0.9336, so use it only as a last resort, and check that your library charges for it.

## What privacy costs in accuracy {#sec-dpsgd-run}

The running example is a network with one hidden layer of 64 tanh units, trained on 10,000 synthetic records with $C$ and $\eta$ both 0.5. @tbl-dpsgd-run trains it at four budgets.

| | Noise multiplier | Test accuracy (mean ± standard deviation over 3 runs) |
| --- | --- | --- |
| $\varepsilon = 0.5$ | 7.72 | $0.8209 \pm 0.0180$ |
| $\varepsilon = 1$ | 4.13 | $0.9175 \pm 0.0161$ |
| $\varepsilon = 4$ | 1.36 | $0.9548 \pm 0.0019$ |
| $\varepsilon = 8$ | 0.917 | $0.9604 \pm 0.0019$ |
| Clipped, no noise | 0 | $0.9649 \pm 0.0009$ |

: Test accuracy of the running example under DP-SGD at four budgets, with $\delta = 10^{-5}$. {#tbl-dpsgd-run}

Clipping alone costs nothing here, so the whole loss comes from the noise, and it grows steeply as $\varepsilon$ shrinks. Going from $\varepsilon = 8$ to 4 costs half a point, while halving it from 1 to 0.5 costs almost 10.

Model size matters because the noise grows with the number of trained weights while the clipped signal does not. Write $p$ for the number of trained weights. The noise vector on the averaged update is then about
$$\frac{\sigma C\sqrt p}{B}$$
long, which in the running example is about ten times the length of the clipped signal. So the best private model is often smaller than the best non-private one, unless [public pretraining](06-public-data.md) does most of the work.

## The Opacus workflow {#sec-dpsgd-opacus}

To make a PyTorch training loop private, Opacus wraps an ordinary model, optimizer and data loader. First, `ModuleValidator.fix(model)` replaces batch normalization with group normalization. A `PrivacyEngine` then returns private versions of all three. Its `make_private_with_epsilon` picks the noise multiplier for a target $\varepsilon$, $\delta$ and number of epochs, while `make_private` takes `noise_multiplier` directly. Both take `max_grad_norm`, the clipping norm, and by default replace the loader with one that draws Poisson batches (`poisson_sampling=True`). The engine's default accountant is the privacy loss random variable (PRV) accountant, `"prv"`, and `accountant="rdp"` selects the Rényi one.

```text
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

model = ModuleValidator.fix(model)               # BatchNorm -> GroupNorm
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
engine = PrivacyEngine()                         # accountant="prv" by default
model, optimizer, loader = engine.make_private_with_epsilon(
    module=model, optimizer=optimizer, data_loader=loader,
    target_epsilon=4.0, target_delta=1e-5, epochs=epochs, max_grad_norm=C)
for epoch in range(epochs):
    for x, y in loader:                          # Poisson batches
        optimizer.zero_grad()
        criterion(model(x), y).backward()        # per-example gradients
        optimizer.step()                         # clip, add noise, step
print(engine.get_epsilon(delta=1e-5))
```

To size the noise before training, `opacus.accountants.utils.get_noise_multiplier(target_epsilon=..., target_delta=..., sample_rate=..., epochs=...)` solves for $\sigma$. When a large batch does not fit in memory, `BatchMemoryManager` from `opacus.utils.batch_memory_manager` splits it into smaller physical batches but still steps once per logical batch, so the accounting is unchanged. Passing `grad_sample_mode="ghost"` turns on ghost clipping, and `make_private` then also returns a wrapped loss function. In TensorFlow, TensorFlow Privacy's `DPKerasSGDOptimizer` takes `l2_norm_clip`, `noise_multiplier` and `num_microbatches`, and the last should equal the batch size.

Tune in a fixed order. First fix $\varepsilon$, a $\delta$ below $1/n$ and the number of epochs, since they are the contract. Then choose a large batch, thousands of examples if you can, and let the accountant set the noise multiplier. Finally, set $C$ near or below the median per-example gradient norm, measured on public or proxy data, and search the step size, which now stands for $\eta C$. Count that search as [Hyperparameter tuning](03-hyperparameter-tuning.md) explains.

When a private model is much worse than its non-private twin, the noise is too long for the signal. Change the batch size first, then train fewer weights, for instance by starting from a public model. If training diverges, lower $\eta C$. And if $\varepsilon$ looks too good, look for a shuffling loader, a microbatch setting or uncounted tuning runs.

## Cheat sheet

| Setting or check | Rule |
| --- | --- |
| What sets $\varepsilon$ | The sampling rate, the noise multiplier and the number of steps only |
| Sampler | Poisson; fixed-size batches use rate $m/n$ and $\sigma/2$; never shuffle-and-partition |
| Model code | Group or layer normalization; divide by $B = qn$ |
| Clipping norm | Fixed in advance, below the typical gradient norm; then tune $\eta C$ |
| Batch size | As large as you can, with the step size scaled up |
| Speed | Vectorize or ghost-clip; microbatches of $m$ multiply the noise by $2m$ |
| Model size | Noise length about $\sigma C\sqrt p/B$; train fewer weights |
| Report | $\varepsilon$, $\delta$, the neighbouring relation, every run counted |

: DP-SGD settings and checks. {#tbl-dpsgd-cheat}

**Rules of thumb**

- Read the data loader and the model before trusting an $\varepsilon$.
- Find the $\varepsilon$ below which accuracy collapses on your task before you promise a budget.
- Report $\varepsilon$ for every step of training, not for the last epoch, and at the $\delta$ you state.
- Choose $C$ and the number of epochs on public data, or pay for the runs that chose them.

## Further reading

- Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016): DP-SGD and the moments accountant.
- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023): what to clip, how to tune and what to report.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/02-dp-sgd.md).
