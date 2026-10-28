# Public data and private fine-tuning

## Why this matters

The differential privacy (DP) version of stochastic gradient descent (SGD), [DP-SGD](02-dp-sgd.md), loses much accuracy when it trains a model from random weights, most at a privacy loss $\varepsilon$ of 1 or below. The most accurate private models avoid this. They start from a model **pretrained** without privacy on a large public dataset, and only **fine-tune** it privately on the sensitive data (Tramèr & Boneh 2021).

Fine-tuning only a small part of the model often helps further. DP-SGD adds noise to every weight it trains, while the useful signal does not grow with their number, so fewer weights mean less noise for the same signal. The workflow [Fine-tune a language model privately with LoRA](14-private-llm-finetuning.md) does the job end to end.

## How many weights to train {#sec-pub-dimension}

Which weights you train leaves $\varepsilon$ unchanged, because DP-SGD's guarantee needs only that one record moves the clipped sum by at most the clipping norm $C$. The choice changes only how much noise lands on the model. Anything else computed only from public data is also free, if no choice in it looked at the private data. Examples are the starting weights, a tokenizer and normalization statistics.

The noise grows with the number $d$ of trainable weights. Write $\sigma$ for the noise multiplier and $B$ for the expected batch size, expected because DP-SGD's Poisson sampling draws batches of random size. Then the noise on the averaged gradient has length about
$$\text{noise length} \approx \frac{\sigma C\sqrt d}{B}.$$ {#eq-pub-noise-length}
The signal, the average clipped gradient, is at most $C$ long whatever $d$ is. When fewer weights train it loses the frozen weights' part of the gradient, though clipping then cuts the rest less.

Take the running example of [DP-SGD](02-dp-sgd.md), the Adult network with 6,018 weights. Pretrain it without privacy on the 1,500 public stand-in rows of [Train a model with DP-SGD](13-train-dp-sgd.md), then fine-tune it on the 30,000 private rows. At $\varepsilon = 1$ and the pretrained weights, the noise is 1.6 times longer than the signal when all weights train. It is only 0.4 times as long when the 130 weights of the last layer train. That gap is narrower than the formula's factor of 6.8, because the signal is shorter when only the last layer trains (0.0726 against 0.1137).

But counting weights overstates the cost of training them all. Noise does harm only along the directions in which the loss is curved, because a step along a flat direction barely changes the loss. Near a pretrained model, few directions are curved (Li, Liu et al. 2022). So on the running example, full fine-tuning trains 46 times as many weights as the last layer, but its noise should raise the loss only 16 times as much. A larger learning rate lets the noise into more of the nearly flat directions, so private fine-tuning wants smaller steps than training from scratch.

Training fewer weights has its own cost, a **bias**, because no private data can move the model along a frozen direction that the task needs.

## Linear probing first {#sec-pub-features}

The simplest option freezes everything but the last layer and trains a linear classifier on the frozen features, which is **linear probing**. Because the features are fixed, the probe's loss is convex. So linear probing turns private deep learning back into [Private empirical risk minimization](01-dp-erm.md), which makes it the safe first baseline.

A private probe's error has two parts. One is the error that privacy adds to convex empirical risk minimization (ERM), which grows with the number $d$ of features. The other is the **feature gap**, how far the best linear model on the features falls short of the best model of any kind, which no number of records or size of $\varepsilon$ closes. For example, the running example's probe without privacy, its ceiling, has a test accuracy of 0.8431, while a non-private network trained from scratch reaches 0.8498. At every budget of @sec-pub-run the private probe stays within 0.002 of that ceiling, on either side, which is the spread between runs.

So measure the probe's ceiling first, by fitting it without privacy on labelled public data that resemble the task. If it already meets your target, stop there. To train the probe privately, feed the frozen features to diffprivlib's `LogisticRegression` ([Private empirical risk minimization](01-dp-erm.md)), or run DP-SGD on the one layer.

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
