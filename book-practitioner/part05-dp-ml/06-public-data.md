# Public data and private fine-tuning

## Why this matters

Differential privacy (DP) applied to stochastic gradient descent (SGD), or DP-SGD ([DP-SGD](02-dp-sgd.md)), can cause substantial accuracy loss when training from random initialization, especially at privacy loss $\varepsilon \le 1$. More accurate private models often start from weights **pretrained** without privacy on a large public dataset, then **fine-tune** them on sensitive data (Tramèr & Boneh 2021).

Fine-tuning fewer weights can further improve accuracy. DP-SGD adds noise to every trained weight, while the useful signal does not grow with the number of weights. Training fewer weights therefore reduces noise relative to signal. [Fine-tune a language model privately with LoRA](14-private-llm-finetuning.md) demonstrates this workflow.

## How many weights to train {#sec-pub-dimension}

The trainable weights do not change $\varepsilon$: DP-SGD requires only that one record changes the clipped sum by at most clipping norm $C$. The choice affects the noise added to the model. Other computations based only on public data are also free, provided they do not depend on private data. Examples include initial weights, tokenizers and normalization statistics.

Noise magnitude grows with the number $d$ of trainable weights. Let $\sigma$ be the noise multiplier and $B$ the expected batch size; Poisson sampling gives variable batch sizes. The averaged-gradient noise has length about
$$\text{noise length} \approx \frac{\sigma C\sqrt d}{B}.$$ {#eq-pub-noise-length}
The average clipped-gradient signal has length at most $C$, independent of $d$. Training fewer weights omits gradient components for frozen weights, although clipping may then reduce the remaining components less.

Consider the Adult network in [DP-SGD](02-dp-sgd.md), with 6,018 weights. Pretrain it without privacy on the 1,500 public stand-in rows used by [Train a model with DP-SGD](13-train-dp-sgd.md), then fine-tune on 30,000 private rows. At $\varepsilon=1$, noise is 1.6 times the signal when all weights train, but 0.4 times the signal when only the 130 last-layer weights train. This difference is smaller than the formula's factor of 6.8 because the last-layer signal is also shorter (0.0726 rather than 0.1137).

Parameter count overstates the cost of full fine-tuning. Noise affects loss mainly along directions with curvature; steps along flat directions have little effect. Near a pretrained model, few directions are curved (Li, Liu et al. 2022). In the running example, full fine-tuning uses 46 times as many weights as last-layer training, but its noise should increase loss only 16 times as much. Larger learning rates expose more nearly flat directions to noise, so private fine-tuning often needs smaller steps than training from scratch.

Training fewer weights has its own cost, a **bias**, because no private data can move the model along a frozen direction that the task needs.

## Linear probing first {#sec-pub-features}

The simplest option freezes all but the last layer and trains a linear classifier on the fixed features, a method called **linear probing**. Its loss is convex, so it reduces private deep learning to [Private empirical risk minimization](01-dp-erm.md) and provides a useful baseline.

A private probe has two error sources. Privacy error for convex empirical risk minimization (ERM) grows with feature count $d$. The **feature gap** is the difference between the best linear model on the features and the best model of any type; more data or a larger $\varepsilon$ cannot remove it. In this example, the non-private probe reaches test accuracy 0.8431, while a non-private network trained from scratch reaches 0.8498. Across budgets in @sec-pub-run, private-probe accuracy stays within 0.002 of its ceiling, comparable to run-to-run variation.

Measure the probe's ceiling by fitting it without privacy on labelled public data similar to the task. If it meets the target, use the probe. For private training, pass frozen features to diffprivlib's `LogisticRegression` ([Private empirical risk minimization](01-dp-erm.md)) or run DP-SGD on the final layer.

## Low-rank adapters and full fine-tuning {#sec-pub-lora}

When the probe's ceiling is below your target, only changing the earlier layers can lift it. **Low-rank adaptation** (LoRA, Hu et al. 2022) freezes each pretrained matrix and learns a small correction to it. The correction is the product of two thin matrices that each have $r$ columns, where $r$ is the rank. A matrix with $d_{\text{in}}$ inputs and $d_{\text{out}}$ outputs has $d_{\text{in}}d_{\text{out}}$ weights, while its adapter has only
$$d = r(d_{\text{in}} + d_{\text{out}}).$$ {#eq-pub-lora-count}

The adapter comes from Hugging Face's `peft` library. This snippet puts a rank-4 adapter on the first layer of the running example's network, and trains its last layer in full.

```python
import torch
from peft import LoraConfig, get_peft_model

model = torch.nn.Sequential(torch.nn.Linear(91, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
print(sum(p.numel() for p in model.parameters()))  # 6018
config = LoraConfig(r=4, lora_alpha=8, target_modules=["0"], modules_to_save=["2"], lora_dropout=0.0)
adapted = get_peft_model(model, config)  # freezes every pretrained weight
print(sum(p.numel() for p in adapted.parameters() if p.requires_grad))  # 750
```

In the snippet, `target_modules` and `modules_to_save` name layers by their index in the `Sequential`, and `lora_alpha` scales the correction. It trains 750 weights, the adapter's $4 \times (91 + 64) = 620$ and the last layer's 130. Opacus's `PrivacyEngine` computes per-example gradients only for the parameters that require them, so the frozen weights cost nothing. Call `model.train()` before `make_private`, because Opacus rejects a model in evaluation mode, as a pretrained model often is after you measure it.

The adapter's second factor starts at zero, so training begins exactly at the pretrained model. The first factor starts random, and that draw must not depend on the private data.

Yu et al. (2022) showed that private fine-tuning of language models with LoRA matches or beats private full fine-tuning, at a fraction of the compute and memory. Each adapter factor is an ordinary linear layer, so the ghost clipping of [DP-SGD](02-dp-sgd.md#sec-dpsgd-ghost) applies to it. Still, Li, Tramèr et al. (2022) matched the adapters with full fine-tuning, using large batches and a learning rate tuned for private training.

## Which method at which budget {#sec-pub-run}

The script [practitioner_adult_studies.py](../../code/part05-dp-ml/practitioner_adult_studies.py) pretrains the running example on the public stand-in rows, then fine-tunes it with Opacus on the private rows, for the workflow's 300 steps. DP-SGD fine-tunes the last layer, a rank-4 adapter plus the last layer, or all weights, against a model trained privately from scratch. Each method's learning rate was chosen at $\varepsilon = 1$ from 0.1, 0.3, 1 and 3, on half of the test rows, and accuracy comes from the other half. That choice looked at private data without paying, so the table shows each method at its best, not what a real run could claim ([Hyperparameter tuning](03-hyperparameter-tuning.md)).

| | Scratch | Full | Last layer | Adapter, rank 4 |
| --- | --- | --- | --- | --- |
| Trainable weights $d$ | 6,018 | 6,018 | 130 | 750 |
| Learning rate | 1 | 1 | 1 | 0.3 |
| $\varepsilon = 0.25$ | 0.8260 | 0.8233 | 0.8416 | 0.8396 |
| $\varepsilon = 0.5$ | 0.8402 | 0.8404 | 0.8432 | 0.8456 |
| $\varepsilon = 1$ | 0.8434 | 0.8459 | 0.8440 | 0.8469 |
| $\varepsilon = 4$ | 0.8450 | 0.8491 | 0.8435 | 0.8468 |

: Accuracy on half of the Adult test rows (mean of two runs), $\delta = 10^{-5}$. The pretrained model without fine-tuning scores 0.8383. {#tbl-pub-run}

Adult is an easy task, so every gap here is under two points, but the pattern is the one larger tasks show. Pretraining is worth about a factor of eight in $\varepsilon$, since the adapter at $\varepsilon = 0.5$ matches training from scratch at $\varepsilon = 4$.

At $\varepsilon = 0.25$ the last layer is best, because its noise is 6.8 times shorter than full fine-tuning's, and its ceiling does not yet bind. From $\varepsilon = 0.5$ on it stays near its ceiling, while first the adapter and then full fine-tuning pass it. The adapter is never more than 0.0023 below the best method, so it is the robust middle choice here. But a public task further from the private one can reverse the ranking even when the probe's ceiling looks fine, so rerun the comparison on a public stand-in for your task.

The learning rate matters as much as the method, since a step three times too large costs full fine-tuning 1.1 percentage points of accuracy at $\varepsilon = 1$. The adapter's rank matters little, since at $\varepsilon = 1$ ranks 1, 4 and 16 differ by about the run-to-run spread.

## What public data does not buy {#sec-pub-caveats}

Whatever the public data are, training on the private data is $(\varepsilon, \delta)$-DP, where $\delta$ is the small additive slack the guarantee allows. The public data get no guarantee of any kind. Non-private pretraining can memorize individual records ([Memorization and training-data extraction](../part02-attacks/05-memorization-extraction.md)), and the fine-tuned model inherits whatever the pretrained weights encode (Tramèr, Kamath & Carlini 2024).

So a person in both datasets has protection only in the private copy. A membership attack against the pretrained model ([Membership inference](../part02-attacks/02-membership-inference.md)) has no bound at all. Web-scale "public" datasets contain posts, profiles and leaked documents, so private notes or emails may well overlap with them. For text, also mind the privacy unit, because one person's documents span many training sequences ([User-level differential privacy](07-user-level.md)).

Check three things before you train and when you report.

1. Check that the public data resemble the private task, and deduplicate the private set against them. Deduplication protects copies, not people, since exact matching misses paraphrases.
2. Choose the public model and the hyperparameters on public data. Otherwise every private run costs privacy ([Hyperparameter tuning](03-hyperparameter-tuning.md)).
3. Name the public data and the overlap check, and state that the guarantee covers the private data given the pretrained model.

## Cheat sheet

| Situation | Train | Why |
| --- | --- | --- |
| Tiny budget, or the probe meets the target | Last layer (linear probe) | Fewest weights; convex |
| Moderate budget, features not quite right | Low-rank adapter plus last layer | Never far from the best |
| Larger budget, large batches, tuned learning rate | All weights | Noise costs only along curved directions |

: Where to start when fine-tuning privately; the ranking depends on the task. {#tbl-pub-cheat}

**Rule of thumb.** Measure the linear probe's ceiling before anything else. Then use large batches and a smaller learning rate than for training from scratch. Tune the learning rate with DP-SGD runs at the target noise on public data, because a non-private sweep does not show how the noise moves the best step.

## References

- Tramèr & Boneh, "Differentially Private Learning Needs Better Features (or Much More Data)" (ICLR 2021): linear probing on good features.
- Li, Tramèr, Liang & Hashimoto, "Large Language Models Can Be Strong Differentially Private Learners" (ICLR 2022), and Yu et al., "Differentially Private Fine-tuning of Language Models" (ICLR 2022): full and adapter-based private fine-tuning.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/06-public-data.md).
