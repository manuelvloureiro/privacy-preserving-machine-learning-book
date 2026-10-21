# Public data and private fine-tuning

## Why this matters

The differential privacy (DP) version of stochastic gradient descent (SGD), [DP-SGD](02-dp-sgd.md), loses much accuracy when it trains a model from random weights. The loss is large at a privacy loss $\varepsilon$ of 1 or below.

The most accurate private models avoid that. They start from a model **pretrained** without privacy on a large public dataset, and only **fine-tune** it privately on the sensitive data. That makes them far more accurate than models trained privately from scratch (Tramèr & Boneh 2021).

Fine-tuning only a small part of the model often helps further. DP-SGD adds noise to every weight it trains, while the useful signal does not grow with their number, so fewer weights mean less noise for the same signal. This chapter says which weights to train, and what public data does *not* protect, and the workflow [Fine-tune a language model privately with LoRA](14-private-llm-finetuning.md) does the job end to end.

## How many weights to train {#sec-pub-dimension}

Which weights you train leaves $\varepsilon$ unchanged, because the guarantee of DP-SGD needs only that one record changes the clipped gradient sum by at most the clipping norm $C$. The choice changes only how much noise lands on the model. Anything else computed from public data alone, such as the starting weights, also costs no privacy, provided nothing in it was chosen by looking at the private data. So a tokenizer or normalization statistics fitted on public data are free, while fitted on the private data they leak.

The noise grows with the number $k$ of trainable weights. Write $\sigma$ for the noise multiplier and $B$ for the expected batch size, expected because DP-SGD's Poisson sampling draws batches of random size. Then the noise on the averaged gradient has length about
$$\text{noise length} \approx \frac{\sigma C\sqrt k}{B}.$$ {#eq-pub-noise-length}
The signal, the average clipped gradient, is at most $C$ long whatever $k$ is. It can even get longer when fewer weights train, because clipping then cuts each record's gradient less.

The running example of [DP-SGD](02-dp-sgd.md), a small network with 3,329 weights, is pretrained on a public task close to the private one and then fine-tuned on the 10,000 private records. It trains for 10,000 steps with batches of 100 and a clipping norm of 0.5, and at $\varepsilon = 1$ the noise multiplier is 4.13. Compared with the signal at the pretrained weights, the noise is then 92.0 times longer when all weights train. It is only 6.7 times longer when the 65 weights of the last layer train. The gap is wider than the formula predicts, because the signal is longer when only the last layer trains (0.0250 against 0.0130).

Counting weights overstates the cost of training them all, though. Noise does harm only along the directions in which the loss is curved, because a step along a flat direction barely changes the loss. Near a pretrained model few directions are curved (Li, Liu et al. 2022). So on the running example, full fine-tuning has 51 times the last layer's weights, but its noise is predicted to raise the training loss only about 7.5 times as much.

A larger step size lets the noise into more of the nearly flat directions within a run, so private fine-tuning wants smaller steps than training from scratch. Training fewer weights has its own cost, a **bias**, because no amount of private data can move the model along a frozen direction that the task needs.

## Linear probing first {#sec-pub-features}

The simplest option freezes everything but the last layer and trains a linear classifier on the frozen features, which is **linear probing**. Because the features are fixed, the probe's loss is convex. Linear probing thus turns private deep learning back into the problem of [Private empirical risk minimization](01-dp-erm.md), which makes it the safe first baseline.

The error of a private probe is the error that privacy adds to convex empirical risk minimization (ERM), which grows with the number $k$ of features, plus the **feature gap**. That gap is how far the best linear model on the features falls short of the best model of any kind. No number of records or size of $\varepsilon$ closes it. For example, the running example's probe fitted without privacy, its ceiling, has a test accuracy of 0.9536, while a network trained without privacy from scratch reaches 0.9606. The private probe stays within 0.006 of that ceiling, but never passes it.

Measure the probe's ceiling first, by fitting it without privacy on labelled public data that resemble the task. If it already meets your target, stop there. To train the probe privately, feed the frozen features to the `LogisticRegression` of diffprivlib, IBM's library of private scikit-learn models, or run DP-SGD on the one layer.

## Low-rank adapters and full fine-tuning {#sec-pub-lora}

When the features are not quite right, the probe stops at a ceiling below your target, and only changing the earlier layers can lift it. **Low-rank adaptation** (LoRA, Hu et al. 2022) freezes each pretrained matrix and learns a small correction to it. The correction is the product of two thin matrices that each have $r$ columns, where $r$ is the rank. A matrix with $d_{\text{in}}$ inputs and $d_{\text{out}}$ outputs has $d_{\text{in}}d_{\text{out}}$ weights, while its adapter has only
$$k = r(d_{\text{in}} + d_{\text{out}}).$$ {#eq-pub-lora-count}

The adapter comes from Hugging Face's `peft` library. This snippet puts a rank-4 adapter on the first layer of a network shaped like the running example, and trains its last layer in full.

```python
import torch
from peft import LoraConfig, get_peft_model

model = torch.nn.Sequential(torch.nn.Linear(50, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))
print(sum(p.numel() for p in model.parameters()))  # 3329
config = LoraConfig(r=4, lora_alpha=8, target_modules=["0"], modules_to_save=["2"], lora_dropout=0.0)
adapted = get_peft_model(model, config)  # freezes every pretrained weight
print(sum(p.numel() for p in adapted.parameters() if p.requires_grad))  # 521
```

In the snippet, `target_modules` and `modules_to_save` name layers by their index in the `Sequential`, and `lora_alpha` scales the correction. It trains 521 weights, the adapter's $4 \times (50 + 64) = 456$ and the last layer's 65. The run in @sec-pub-run treats the first layer's biases as a 51st input, so its adapter has $4 \times (51 + 64) = 460$ weights, and 525 with the last layer. Opacus's `PrivacyEngine`, used as in [DP-SGD](02-dp-sgd.md), then computes per-sample gradients only for the parameters that require them, so the frozen weights cost nothing. Call `model.train()` before `make_private`, since Opacus rejects a model left in evaluation mode, as a pretrained model often is after you measure it.

The adapter's second factor starts at zero, so training begins exactly at the pretrained model. The first factor starts random, and that draw must not depend on the private data.

Yu et al. (2022) showed that private fine-tuning of language models with LoRA matches or beats private full fine-tuning, at a fraction of the compute and memory. Each adapter factor is an ordinary linear layer, so the ghost clipping of [DP-SGD](02-dp-sgd.md#sec-dpsgd-ghost) applies to it. It gets each record's gradient norm without ever forming the per-example gradient.

Full fine-tuning is not ruled out. Li, Tramèr et al. (2022) matched the adapters by fine-tuning all the weights of language models. They used large batches and a step size tuned for private training, so the evidence goes both ways. A third option adds noise only in the few directions in which public gradients mostly lie, and throws away the private signal outside them (Zhou, Wu & Banerjee 2021).

## Which method at which budget {#sec-pub-run}

The numbers in this section come from the full edition's synthetic running example. No practitioner script reproduces them, so treat them as one task's evidence. On it, DP-SGD fine-tunes the last layer, a rank-4 adapter plus the last layer, or all weights, against a model trained privately from scratch. Each method's step size was chosen at $\varepsilon = 1$ on half of the private test set, and accuracy is reported on the other half. That choice looked at private data without paying for it, so the table compares methods at their best rather than what a real run could claim ([Hyperparameter tuning](03-hyperparameter-tuning.md)).

| | Scratch | Full | Last layer | Adapter, rank 4 |
| --- | --- | --- | --- | --- |
| Trainable weights $k$ | 3,329 | 3,329 | 65 | 525 |
| Step size | 0.25 | 0.1 | 0.3 | 0.1 |
| $\varepsilon = 0.5$ | 0.8589 | 0.9395 | 0.9478 | 0.9464 |
| $\varepsilon = 1$ | 0.9246 | 0.9599 | 0.9497 | 0.9616 |
| $\varepsilon = 2$ | 0.9518 | 0.9663 | 0.9519 | 0.9653 |
| $\varepsilon = 8$ | 0.9611 | 0.9677 | 0.9514 | 0.9655 |

: Test accuracy (mean of two runs), $\delta = 10^{-5}$. The pretrained model without fine-tuning has 0.9035. {#tbl-pub-run}

Here pretraining is worth about a factor of eight in $\varepsilon$, since the adapter at $\varepsilon = 1$ matches training from scratch at $\varepsilon = 8$.

At $\varepsilon = 0.5$ the last layer is best, because its noise is 7.2 times shorter than full fine-tuning's and its ceiling is not yet binding. From $\varepsilon = 1$ on, it is stuck near its ceiling while the others keep improving. The adapter is never more than 0.0022 below the best method at any budget, so it is the robust middle choice here. That ranking depends on the task. A public task further from the private one can reverse it even when the probe's ceiling looks fine. So rerun the comparison on a public stand-in for your task before you commit.

The step size matters as much as the method, since a step three times too large costs full fine-tuning 1.3 percentage points of accuracy. The adapter's rank, by contrast, is cheap to get roughly right, because at $\varepsilon = 1$ ranks 1, 4 and 16 differ by about the run-to-run spread.

## What public data does not buy {#sec-pub-caveats}

Whatever the public data are, training on the private data is $(\varepsilon, \delta)$-DP, where $\delta$ is the small additive slack the guarantee allows. The public data get no guarantee of any kind. Non-private pretraining can memorize individual records ([Memorization and training-data extraction](../part02-attacks/05-memorization-extraction.md)), and the fine-tuned model inherits whatever the pretrained weights encode (Tramèr, Kamath & Carlini 2024).

A person in both datasets is protected only in the private copy. A membership attack against the pretrained model ([Membership inference](../part02-attacks/02-membership-inference.md)) has no bound at all. Web-scale "public" datasets contain posts, profiles and leaked documents, so private notes or emails may well overlap with them.

Text raises a second question, which unit the guarantee protects, called the privacy unit. A language model trains on fixed-length sequences, but one person's documents span many of them, so a guarantee per sequence protects far less than a person ([User-level differential privacy](07-user-level.md)).

Check three things before you train and when you report.

1. Check that the public data resemble the private task, and deduplicate the private set against them. Deduplication protects copies, not people, since exact matching misses paraphrases.
2. Choose the public model and the hyperparameters on public data. Otherwise every private run costs privacy ([Hyperparameter tuning](03-hyperparameter-tuning.md)).
3. Name the public data and the overlap check, and state that the guarantee covers the private data given the pretrained model.

## Cheat sheet

| Situation | Train | Why |
| --- | --- | --- |
| Tiny budget, or the probe meets the target | Last layer (linear probe) | Fewest weights; convex |
| Moderate budget, features not quite right | Low-rank adapter plus last layer | Never far from the best |
| Larger budget, large batches, tuned step size | All weights | Noise costs only along curved directions |

: Where to start when fine-tuning privately; the ranking depends on the task. {#tbl-pub-cheat}

**Rule of thumb.** Measure the linear probe's ceiling before anything else. Then use large batches and a smaller step size than for training from scratch. Tune the step size by DP-SGD runs at the target noise on public data, since a non-private sweep does not show how the noise moves the best step.

## Further reading

- Tramèr & Boneh, "Differentially Private Learning Needs Better Features (or Much More Data)" (ICLR 2021): linear probing on good features.
- Li, Tramèr, Liang & Hashimoto, "Large Language Models Can Be Strong Differentially Private Learners" (ICLR 2022), and Yu et al., "Differentially Private Fine-tuning of Language Models" (ICLR 2022): full and adapter-based private fine-tuning.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/06-public-data.md).
