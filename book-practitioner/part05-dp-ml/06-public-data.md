# Public data and private fine-tuning

## Why this matters

Training from random weights with the differential privacy (DP) version of stochastic gradient descent (SGD), [DP-SGD](02-dp-sgd.md), loses a great deal of accuracy at a privacy loss $\varepsilon$ of 1 and below. The most accurate private models are built differently. They start from a model **pretrained** without privacy on a large public dataset, and only **fine-tune** it privately on the sensitive data. Such models are far more accurate than those trained privately from scratch, as Tramèr & Boneh (2021) and later work report.

Fine-tuning only a small part of the model, such as the last layer or a low-rank update to each matrix, often helps further. That is because DP-SGD adds noise to every weight it trains, while the useful signal is bounded by clipping and usually lies in a few directions. This chapter gives the order in which to try the options and what public data does *not* protect.

## How many weights to train {#sec-pub-dimension}

The privacy argument of DP-SGD needs only that one record changes the clipped gradient sum by at most the clipping norm $C$, whichever weights are trained. So the choice of trainable weights leaves $\varepsilon$ unchanged and changes only how much noise lands on the model. Public data are free for a related reason, because anything computed from public data alone, such as the starting weights, is a fixed input as far as the private data are concerned. The condition is that nothing in it was chosen by looking at the private data. That includes normalization statistics and tokenizers, which are fine when fitted on public data and a leak when fitted on private data.

The noise grows with the number $k$ of trainable weights. Write $\sigma$ for the noise multiplier and $B$ for the expected batch size. Then the noise on the averaged gradient has length about
$$\text{noise length} \approx \frac{\sigma C\sqrt k}{B}.$$ {#eq-pub-noise-length}
The signal, the average clipped gradient, is at most $C$ long whatever $k$ is, and it can even grow when fewer weights are trained, because shorter gradients are clipped less.

This part's running example is a small network with 3,329 weights, pretrained on a related public task and then fine-tuned on 10,000 private records. It trains for 10,000 steps with a clipping norm of 0.5 and a batch size of 100, at a slack $\delta$ of $10^{-5}$. At $\varepsilon = 1$ the noise multiplier is 4.13, and the noise length is 1.191 for all weights against 0.166 for the 65 weights of the last layer.

Measured against the signal, the noise is 92.0 times larger for all weights but only 6.7 times for the last layer. So the signal needs about 8,455 steps to show through the noise for all weights, and only 44 for the last layer.

Counting weights overstates the cost, though. Noise hurts only in directions in which the loss is curved enough to notice it within the run (Li, Liu et al. 2022). Write $h_j$ for the curvature of the loss in its $j$-th principal direction. With step size $\eta$, the number of directions the noise has time to fill is the **effective dimension**
$$d_{\text{eff}} = \sum_{j=1}^k\Big(1 - (1 - \eta h_j)^{2T}\Big),$$ {#eq-pub-noise-cost}
and the noise raises the loss in proportion to it. On the running example, full fine-tuning has an effective dimension of 451 out of 3,329 weights. So its noise is 51 times larger in squared length than the last layer's, but it costs only about 7.5 times more. A larger step size or a longer run fills more flat directions with noise, which is why private fine-tuning wants smaller step sizes than training from scratch.

Freezing weights has a cost too, a **bias**, because no amount of private data can move the model along a frozen direction that the task needs.

## Linear probing first {#sec-pub-features}

The simplest option freezes everything but the last layer and trains a linear classifier on the frozen features, which is **linear probing**. Tramèr & Boneh (2021) showed that it beats end-to-end private training on standard image benchmarks at moderate budgets. It turns private deep learning back into the convex problem of [Private empirical risk minimization](01-dp-erm.md), so it is the safe first baseline.

Its accuracy is capped by the features. The error of a private probe is the privacy cost of convex empirical risk minimization (ERM) in $k$ dimensions plus the **feature gap**. That gap is how far the best linear model on the features falls short of the best model of any kind, and it does not shrink with more records or a larger $\varepsilon$. The best possible probe reaches 0.9536, against 0.9606 for a network trained non-privately from scratch. The private probe stays within 0.006 of its ceiling at every budget, but it can never pass it. So measure the probe's non-private ceiling first, and if it already meets your target, stop there. To train the probe privately, feed the frozen features to `diffprivlib.models.LogisticRegression`, or run DP-SGD on the one layer.

## Low-rank adapters and full fine-tuning {#sec-pub-lora}

When the features are not quite right, the probe stops at its ceiling, and you need to change the earlier layers. **Low-rank adaptation** (LoRA, Hu et al. 2022) freezes each pretrained matrix $W_0$ and learns a correction $UV^\top$ whose rank, the number $r$ of columns of the two factors, is small. For a matrix with $d_{\text{in}}$ inputs and $d_{\text{out}}$ outputs, that is
$$k = r(d_{\text{in}} + d_{\text{out}})$$ {#eq-pub-lora-count}
trainable weights instead of $d_{\text{in}}d_{\text{out}}$.

Training starts with the factor $U$ at zero, so the model begins exactly at the pretrained one, and with the factor $V$ random. That random draw must be independent of the private data. The adapter's per-example gradients are outer products, so the ghost clipping of [DP-SGD](02-dp-sgd.md#sec-dpsgd-ghost) applies and no per-example matrix is ever formed. Yu et al. (2022) showed that private fine-tuning of language models with LoRA matches or beats private full fine-tuning, at a fraction of the cost.

Full fine-tuning is not ruled out. Li, Tramèr et al. (2022) matched or beat the adapters by fine-tuning all the weights of language models, with large batches, a step size tuned for private training, and a fine-tuning objective that resembles the pretraining one. The evidence on accuracy goes both ways, but the adapter is cheaper to store, compute and clip.

In PyTorch, the usual stack is Hugging Face's `peft` library for the adapter and Opacus for DP-SGD. Build a `LoraConfig(r=..., target_modules=...)` and wrap the base model with `get_peft_model(model, config)`, so that only the adapter's weights require gradients. Then pass the model, its optimizer and data loader to the `make_private` method of Opacus's `PrivacyEngine`, which computes per-sample gradients only for the trainable parameters. For large language models, the `private-transformers` package of Li et al. and Microsoft's `dp-transformers` wrap this setup, with ghost clipping to save memory.

A third option projects each private gradient onto a subspace estimated from public gradients and adds noise only there (Zhou, Wu & Banerjee 2021). It keeps the same $\varepsilon$ and trades noise for the bias of the signal it projects away.

## Which method at which budget {#sec-pub-run}

On the running example, DP-SGD fine-tunes the last layer, a rank-4 adapter on the first layer plus the last layer, or all weights, against a baseline trained from scratch. Each method's step size was chosen at $\varepsilon = 1$ on half of the private test set, and accuracy is reported on the other half.

| | Scratch | Full | Last layer | Adapter, rank 4 |
| --- | --- | --- | --- | --- |
| Trainable weights $k$ | 3,329 | 3,329 | 65 | 525 |
| Step size | 0.25 | 0.1 | 0.3 | 0.1 |
| $\varepsilon = 0.5$ | 0.8589 | 0.9395 | 0.9478 | 0.9464 |
| $\varepsilon = 1$ | 0.9246 | 0.9599 | 0.9497 | 0.9616 |
| $\varepsilon = 2$ | 0.9518 | 0.9663 | 0.9519 | 0.9653 |
| $\varepsilon = 8$ | 0.9611 | 0.9677 | 0.9514 | 0.9655 |

: Test accuracy (mean of two runs), $\delta = 10^{-5}$. The pretrained model without fine-tuning has 0.9035. {#tbl-pub-run}

On this task, pretraining is worth about a factor of eight in $\varepsilon$, since the adapter at $\varepsilon = 1$ matches training from scratch at $\varepsilon = 8$.

Which weights to train depends on the budget. At $\varepsilon = 0.5$ the last layer is best, because its noise is 7.2 times shorter than full fine-tuning's and its ceiling is not yet binding. From $\varepsilon = 1$ on, it is stuck near that ceiling, while the adapter and full fine-tuning keep improving. The adapter is never more than 0.0022 below the best method at any budget, so it is the robust middle choice.

The step size matters as much as the method. A step size three times too large costs full fine-tuning 1.3 points of accuracy. Their best step size is 2.5 times smaller than the one for training from scratch. The adapter's rank, by contrast, is cheap to get roughly right, since ranks from 1 to 16 differ by less than the run-to-run spread.

## What public data does not buy {#sec-pub-caveats}

The guarantee is exactly this. For every fixed public dataset, the map from the private dataset to the fine-tuned model is $(\varepsilon, \delta)$-DP. With respect to the public dataset, there is no guarantee of any kind. Non-private pretraining can memorize individual records ([Memorization and training-data extraction](../part02-attacks/05-memorization-extraction.md)), and the fine-tuned model inherits whatever the pretrained weights encode (Tramèr, Kamath & Carlini 2024).

This matters most for a person whose record is in both datasets, because only the private copy is protected, and a membership attack against the pretrained model ([Membership inference](../part02-attacks/02-membership-inference.md)) has no bound at all. Web-scale "public" datasets contain posts, profiles and leaked documents, so a private set of medical notes or emails may well overlap with them.

Before you train and when you report, go through this checklist.

1. Deduplicate the private set against the public data. That protects copies, not people, because exact matching misses paraphrases and records about the same person.
2. Check that the public data resemble the private task, because benchmarks with a "private" set close to curated public data overstate what you will get on genuinely sensitive data.
3. Choose the public model, the step size, the adapter rank and the architecture on public data, or pay for every private run. Keeping the best of three privately fine-tuned models spends the budget of all three runs, unless you use a private selection procedure ([Hyperparameter tuning](03-hyperparameter-tuning.md)).
4. Say what the public data are, how they were obtained and whether overlap was checked, and state that the guarantee covers the private data given the pretrained model.

## Code {#sec-pub-code}

Before choosing what to fine-tune, compare the candidates' noise lengths at the noise multiplier your accountant gives.

```python
from math import sqrt


def noise_length(sigma: float, clip: float, batch: float, trainable: int) -> float:
    return sigma * clip * sqrt(trainable) / batch


def lora_weights(rank: int, d_in: int, d_out: int) -> int:
    return rank * (d_in + d_out)


adapter = lora_weights(4, 51, 64) + 65  # first-layer adapter plus the 65-weight last layer
print(adapter)  # 525
print(f"{noise_length(4.13, 0.5, 100, 3329):.3f}")  # 1.191
print(f"{noise_length(4.13, 0.5, 100, adapter):.3f}")  # 0.473
print(f"{noise_length(4.13, 0.5, 100, 65):.3f}")  # 0.166
```

## Cheat sheet

| Situation | Train | Why |
| --- | --- | --- |
| Very small budget, or a probe already meets the target | Last layer (linear probe) | Fewest weights, convex, known ceiling |
| Moderate budget, features not quite right | Low-rank adapter plus last layer | Never far from the best; cheap to clip and store |
| Larger budget, large batches, tuned step size | All weights | Pays for the effective dimension, not every weight |

: Which weights to fine-tune privately. The noise length is $\sigma C\sqrt k/B$ for $k$ trainable weights, whatever they are. {#tbl-pub-cheat}

**Rules of thumb**

- Start from a model pretrained on public data that resembles the private task.
- Measure the linear probe's ceiling before anything else.
- Use large batches and a step size smaller than for training from scratch, and tune them on public data.
- Never call the pretrained model private, since nothing protects its training data.
- Expect little from a public model far from the task.

## Further reading

- Tramèr & Boneh, "Differentially Private Learning Needs Better Features (or Much More Data)" (ICLR 2021): linear probing on good features.
- Li, Tramèr, Liang & Hashimoto, "Large Language Models Can Be Strong Differentially Private Learners" (ICLR 2022), and Yu et al., "Differentially Private Fine-tuning of Language Models" (ICLR 2022): full and parameter-efficient private fine-tuning, and the hyperparameters that make each work.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/06-public-data.md).
