# Fine-tune a language model privately with LoRA

## Why this matters

You have a pretrained language model and private text, such as support tickets or clinical notes. You want the model to learn from that text without being able to repeat it. A model fine-tuned without privacy can write out a phone number or a password that was often in its training text ([Memorization and training-data extraction](../part02-attacks/05-memorization-extraction.md)). The fix is to fine-tune with the differential privacy (DP) version of stochastic gradient descent (SGD), called DP-SGD. It costs least when it trains only a small adapter on the frozen model, which is often as accurate as training every weight ([Public data and private fine-tuning](06-public-data.md)). Here the adapter is low-rank adaptation (LoRA), from Hugging Face's `peft` library, and Opacus does the private training.

The model is Pythia-14m, a public language model of 14 million parameters from EleutherAI. The private text is Tiny Shakespeare, a million characters of Shakespeare's plays. The script fine-tunes the model twice, once without privacy and once at a privacy loss $\varepsilon$ of 8 for each sequence of training text. First it plants made-up secrets, called canaries, in the text, so that you can see what each model memorized. The script [practitioner_lora.py](../../code/part05-dp-ml/practitioner_lora.py) runs in a few minutes without a GPU.

```bash
uv run python code/part05-dp-ml/practitioner_lora.py
```

## Fix the privacy unit, n and δ {#sec-llmwf-unit}

The guarantee protects one unit, so decide first what the unit is. A language model trains on fixed-length sequences of tokens, the word pieces its tokenizer cuts text into. The script cuts the text into sequences of 32 tokens, about one or two lines of verse, and makes each sequence the unit. To keep the run short, it trains on the first 1,024 sequences and holds out the last 128.

```text
sequences of 32 tokens: 10,632  training n=1,024  held out 128
delta=1e-04 (privacy unit: one training sequence)
```

For $n = 1{,}024$ sequences, the book's rule for the slack $\delta$ gives $10^{-4}$ ([How small δ must be](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta)).

A sequence is the right unit for this demonstration, but rarely in production. One person's document spans many sequences, and one user writes many documents, so a guarantee per sequence protects a person much less than it seems. In production, make the unit a document or a user, and bound how many sequences each contributes. Then set $\delta$ from the number of users ([User-level differential privacy](07-user-level.md)).

## Plant canaries {#sec-llmwf-canaries}

To check memorization later, you need secrets whose presence you control. So the script writes **canaries** into the training text. Each is a sentence such as "the pin of Alice is 8 6 5 2 3", whose last five digits are the secret. There are four groups of three canaries, inserted 1, 4, 16 and 64 times each. Three more, the controls, are drawn the same way but never inserted, so they show what an unseen secret scores in the audit of @sec-llmwf-memorization.

```{.python excerpt="lora-canary-insert"}
def insert_canaries(
    train: torch.Tensor, canaries: list[Canary], tokenizer: PreTrainedTokenizerBase, rng: np.random.Generator
) -> torch.Tensor:
    """Write each copy of each canary over the start of a different training sequence."""
    train = train.clone()
    total = sum(c.copies for c in canaries)
    rows = iter(rng.choice(len(train), size=total, replace=False).tolist())
    for canary in canaries:
        tokens = torch.tensor(tokenizer(canary.text)["input_ids"])
        for _ in range(canary.copies):
            train[next(rows), : len(tokens)] = tokens
    return train
```

The 255 copies take up a quarter of the training set. That is far denser than secrets in real data, on purpose, because it lets a model this small memorize within a short run, so the audit has a positive control.

```text
canary sequences: 255 (3 canaries each at (1, 4, 16, 64) copies)
tokens per secret: [5]
candidates: 2,000  exposure cap 10.97 bits  (full space 16.61)
```

The digits are separated by spaces so that every possible secret is exactly five tokens long, which the second line confirms over 1,000 random secrets.

## Load a public model and tokenizer {#sec-llmwf-public}

Everything not trained privately must come from outside the private data. That includes the tokenizer, whose vocabulary would leak rare words if it were built from the private text. So the weights and the tokenizer both come from the same public checkpoint, Pythia-14m, pinned to one revision. The script loads the model in float32, because `transformers` loads some checkpoints in float16, which is far slower on a CPU.

The guarantee then covers the private text given the pretrained model, and says nothing about the model's own training data. Pythia was pretrained on the Pile, which includes Project Gutenberg's books and so very likely Shakespeare's plays. That is harmless here, but with real data, check for overlap with the public model's training data ([Public data and private fine-tuning](06-public-data.md#sec-pub-caveats)).

## Choose what to train {#sec-llmwf-lora}

DP-SGD adds noise to every weight it trains. So the noise on the averaged gradient grows with the square root of the number of trained weights, but the clipped signal does not ([Public data and private fine-tuning](06-public-data.md#sec-pub-dimension)). That makes the choice of trainable weights a main lever on accuracy, with the learning rate. LoRA freezes each pretrained matrix and learns a correction to it, the product of two thin factors whose width is the rank.

```{.python excerpt="lora-config"}
attention_and_mlp = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]  # every linear layer
config = LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=attention_and_mlp, lora_dropout=0.0)
lora_model = get_peft_model(model, config)  # freezes the pretrained weights
trainable_count = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
total_count = sum(p.numel() for p in lora_model.parameters())
print(f"trainable parameters: LoRA {trainable_count:,}  full fine-tuning {total_count - trainable_count:,}")
```

```text
full model parameters: 14,067,712
trainable parameters: LoRA 98,304  full fine-tuning 14,067,712
```

By default the adapters have rank 8 and sit on every linear layer, in attention and in the feed-forward blocks. They train 98,304 weights, against about 14 million for full fine-tuning. `lora_alpha` scales the adapters' correction, set to the usual twice the rank. For a real task, choose the rank and the layers on public data, because a choice made from private runs costs privacy ([Hyperparameter tuning](03-hyperparameter-tuning.md)).

## Make the model Opacus-compatible {#sec-llmwf-opacus}

Opacus needs one gradient per sequence, and layers that mix the records of a batch, such as batch normalization, make that impossible. Opacus's `ModuleValidator.validate` lists such layers before you train.

```text
validator errors: 0
```

Pythia has none, and only the adapters train in any case. They are ordinary linear layers, which Opacus's default hooks handle, so the script passes `grad_sample_mode="hooks"`. Opacus computes per-sequence gradients only for the weights that require gradients, so the 14 million frozen weights add no cost. Call `model.train()` before you wrap a model you have just evaluated, because Opacus rejects one in evaluation mode. After training, the hooks stay attached and slow down everything the model does next, so the script calls `remove_hooks()` before it evaluates the private model.

## Calibrate and train {#sec-llmwf-train}

The accountant finds the noise for a fixed target, here $\varepsilon = 8$ over 3 epochs with an expected batch of 32 sequences. The sampling rate $q$, the share of the data in an average batch, is then $32/1{,}024$, so the 3 epochs take 96 steps. The accountant is the privacy loss random variable (PRV) accountant, which computes $\varepsilon$ almost exactly ([Choosing an accountant](../part04-privacy-accounting/09-choosing-an-accountant.md)).

```{.python excerpt="lora-private"}
privacy_engine = PrivacyEngine(accountant="prv")
private_model, optimizer, loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=loader,  # replaced by Poisson sampling with rate BATCH / n
    target_epsilon=epsilon,
    target_delta=delta,
    epochs=EPOCHS,
    max_grad_norm=MAX_GRAD_NORM,
    grad_sample_mode="hooks",
)
print(f"noise multiplier: {optimizer.noise_multiplier:.3f}")
```

The training loop is an ordinary one, with Poisson batches, in which each sequence joins each batch independently with probability $q$.

```{.python excerpt="lora-step"}
for (batch,) in loader:  # each batch holds a random number of sequences
    optimizer.zero_grad()
    loss = private_model(input_ids=batch, labels=batch).loss
    loss.backward()  # Opacus's hooks keep one gradient per sequence
    optimizer.step()  # clips each one to MAX_GRAD_NORM, sums, adds noise
    steps += 1
```

```text
noise multiplier: 0.553
sample rate q=0.03125  steps=96  noise multiplier=0.553
epsilon (Opacus PRV)=7.99  epsilon (dp_accounting PLD)=7.98  at delta=1e-04
```

The noise multiplier is the noise's standard deviation in units of the clipping norm, `MAX_GRAD_NORM`, which the script sets to 1. The last line cross-checks the $\varepsilon$ with the privacy loss distribution (PLD) accountant of Google's `dp_accounting`, as in [Calibrate and account a DP-SGD run](../part04-privacy-accounting/10-calibrate-dp-sgd.md#sec-calib-crosscheck), and the two agree.

Both runs use AdamW at a learning rate of $10^{-3}$, and the private run differs only by clipping, noise and Poisson sampling. That learning rate, the rank and the epochs were chosen in a few non-private trial runs on the same text. A real run must choose them on similar public text, such as other public-domain plays, or pay for the trials in $\varepsilon$ ([Hyperparameter tuning](03-hyperparameter-tuning.md)).

Two more settings change for a real run. With more data, use batches in the thousands, and let Opacus's `BatchMemoryManager` split them into physical batches that fit in memory ([Train a model with DP-SGD](13-train-dp-sgd.md)). And a model you release needs `secure_mode=True` in `PrivacyEngine`. Its secure noise cannot be seeded, so the script leaves it off.

## Evaluate {#sec-llmwf-evaluate}

A language model is judged by how well it predicts held-out text. The script measures the mean cross-entropy per token on the 128 held-out sequences, and its exponential, the **perplexity**. A perplexity of 159.3 means the model is as unsure as if it chose among about 159 equally likely tokens. So lower is better.

```text
held-out loss pretrained: 5.306 nats/token  perplexity 201.6
held-out loss LoRA non-private: 5.071 nats/token  perplexity 159.3
held-out loss LoRA DP eps=8: 5.044 nats/token  perplexity 155.1
```

Both fine-tuned models improve on the pretrained one. On one seed, the private model's slightly lower perplexity means only that DP costs nothing measurable here. Both gains are modest, because a quarter of the training sequences begin with canaries rather than Shakespeare.

Do not expect privacy to be this cheap on every task. This run is short, trains few weights, and uses text that is very likely in the pretraining data already. With a harder task or a smaller $\varepsilon$, compare the private model with both baselines at several budgets, as [Train a model with DP-SGD](13-train-dp-sgd.md#sec-trainwf-utility) does.

## Check memorization {#sec-llmwf-memorization}

The canaries show what perplexity cannot, whether the model will repeat a secret. The script measures each canary in two ways. The first is extraction. It gives the model the prefix "the pin of Alice is", lets it write the likeliest next token five times, and checks whether the result is the secret.

```{.python excerpt="lora-extract"}
@torch.no_grad()
def greedy_completion(model: PeftModel, tokenizer: PreTrainedTokenizerBase, prefix: str) -> str:
    """The five tokens the model writes after the prefix when it always picks the likeliest one."""
    ids = tokenizer(prefix, return_tensors="pt")["input_ids"]
    for _ in range(5):
        next_token = model(input_ids=ids).logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_token], dim=1)
    return tokenizer.batch_decode(ids[:, -5:])[0].strip()
```

The second is the **exposure** of [Memorization and training-data extraction](../part02-attacks/05-memorization-extraction.md#sec-memo-exposure). It ranks the canary's secret among random secrets of the same format by how likely the model finds each, and turns the rank into bits. The script's likelihood is the summed log-probability of the five digits. Each bit halves the number of guesses an attacker needs.

```{.python excerpt="lora-exposure"}
@torch.no_grad()
def secret_scores(model: PeftModel, tokenizer: PreTrainedTokenizerBase, prefix: str, secrets: list[str]) -> Scores:
    """Summed log-probability of each secret's five digit tokens after the prefix."""
    prefix_ids = tokenizer(prefix)["input_ids"]
    rows = [prefix_ids + tokenizer(" " + s)["input_ids"] for s in secrets]
    ids = torch.tensor(rows)  # every secret is five tokens, so the rows have equal length
    logits = model(input_ids=ids[:, :-1], logits_to_keep=5).logits  # the predictions of the five digits
    logprobs = logits.log_softmax(dim=-1).gather(-1, ids[:, -5:, None])
    return logprobs.sum(dim=(1, 2)).double().numpy()


def canary_exposure(
    model: PeftModel, tokenizer: PreTrainedTokenizerBase, canary: Canary, candidates: list[str]
) -> tuple[float, float]:
    """The exposure of the canary's secret among the candidates, measured and extrapolated past the cap."""
    candidate_scores = scores_batched(model, tokenizer, canary.prefix, candidates)
    score = float(scores_batched(model, tokenizer, canary.prefix, [canary.secret])[0])
    return exposure(score, candidate_scores), extrapolated_exposure(score, candidate_scores, SECRET_SPACE)
```

No library installed with this book computes exposure, so `exposure` and `extrapolated_exposure` come from the book's helper module [practitioner_privacy_tests.py](../../code/practitioner_privacy_tests.py), after TensorFlow Privacy's versions. With 2,000 random candidates, a canary that beats them all reaches a cap of 10.97 bits. A secret the model has never seen has a random rank and averages about 1.44 bits ($\log_2 e$). Each line below gives a group's mean exposure, the same past the cap, and how many of its three secrets greedy decoding wrote out.

```text
non-private 1 copies: mean exposure 2.0 bits  extrapolated 2.0  extracted 0/3
non-private 4 copies: mean exposure 5.9 bits  extrapolated 5.7  extracted 0/3
non-private 16 copies: mean exposure 10.0 bits  extrapolated 13.0  extracted 0/3
non-private 64 copies: mean exposure 11.0 bits  extrapolated 16.3  extracted 3/3
non-private controls: mean exposure 1.4 bits  extrapolated 1.5  extracted 0/3
DP 1 copies: mean exposure 0.9 bits  extrapolated 0.9  extracted 0/3
DP 4 copies: mean exposure 0.6 bits  extrapolated 0.6  extracted 0/3
DP 16 copies: mean exposure 4.2 bits  extrapolated 4.3  extracted 0/3
DP 64 copies: mean exposure 4.0 bits  extrapolated 3.9  extracted 0/3
DP controls: mean exposure 2.2 bits  extrapolated 2.2  extracted 0/3
```

The non-private model, the positive control, leaks. Exposure rises with the number of copies, from 2.0 bits for one copy to the cap for 64, and the model writes out all three 64-copy secrets. The extrapolated column sees past the cap by fitting a skew-normal distribution, a normal with one heavier tail, to the candidates' scores. It puts the 64-copy secrets near the 16.61 bits of the whole space.

The 4-copy canaries are the instructive case. Two of the three reach 6.9 and 8.2 bits, yet none is extracted, so exposure detects memorization that extraction alone would miss.

Greedy decoding extracts nothing from the private model, and its canaries with 1 or 4 copies score no higher than its controls. The controls' 2.2 bits are within the scatter of a mean over three secrets. Still, DP limits the effect without erasing it, since the canaries with 16 and 64 copies still average about 4 bits.

The guarantee allows that, because $\varepsilon = 8$ protects one sequence, while a canary with many copies is many sequences. Group privacy bounds a group of units at about their number times $\varepsilon$. So 16 copies get $16 \times 8$, far too much to promise anything ([User-level differential privacy](07-user-level.md)). A secret that a user repeats across many documents is the same case, one more reason to make the unit the user.

Two cautions apply. Because the canaries are so dense, a real non-private model may leak less than this one, and a real audit should plant fewer copies. And three canaries per group is a small sample. For a fuller assessment before release, run the attacks of [Pre-release privacy risk assessment](../part02-attacks/10-risk-assessment.md). For a statistical lower bound on $\varepsilon$, run the one-run audit of [Auditing private training](10-auditing.md).

## Variants {#sec-llmwf-variants}

Three variants matter for real models. The block below shows ghost clipping, LoRA on GPT-2, and packages that wrap the whole loop, with pinned versions. It was not run.

```text
# opacus==1.6.0: ghost clipping, when per-sequence gradients do not fit in memory
model, optimizer, criterion, loader = privacy_engine.make_private_with_epsilon(
    module=model, optimizer=optimizer, data_loader=loader, criterion=torch.nn.CrossEntropyLoss(),
    target_epsilon=8.0, target_delta=1e-4, epochs=3, max_grad_norm=1.0, grad_sample_mode="ghost",
)
logits = model(input_ids=batch).logits[:, :-1]                 # predict each next token
loss = criterion(logits.flatten(0, 1), batch[:, 1:].flatten(), shape=logits.shape)
loss.backward()                                                # norms first, then the clipped sum

# peft==0.21.2: GPT-2 stores its attention as Conv1D, with the weight transposed
LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], fan_in_fan_out=True)

# dp-transformers==1.0.0 (Microsoft): Opacus inside the Hugging Face Trainer
# private-transformers (Li et al.): install from its GitHub repository, pinned to a commit
```

Ghost clipping gets each sequence's gradient norm without storing the gradient, so memory no longer holds one gradient per sequence ([DP-SGD](02-dp-sgd.md#sec-dpsgd-ghost)).

## Report {#sec-llmwf-report}

[Release and governance](12-release-governance.md#sec-relgov-statement) builds the statement from the run's record. For this run, it adds three things. The guarantee covers the private text given the pretrained model, named with its revision. The hyperparameters came from non-private trials on the private text, which a real release could not claim for free. And the canary audit is attached as evidence, with its caveats.

What ships is the adapter. With `--save-adapter FOLDER`, the script saves it after `remove_hooks()` and loads it back onto the same pinned checkpoint.

```{.python excerpt="lora-save"}
private.save_pretrained(path)  # only the adapter weights and config, after Opacus's hooks are removed
base_model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32)  # the same pinned checkpoint
reloaded = PeftModel.from_pretrained(base_model, path)
```

The folder holds only the adapter, so the statement names the base checkpoint and revision it fits. The guarantee covers the adapter and anything computed from it.

## Checklist

| Step | Default | Check |
| --- | --- | --- |
| Unit, $n$, $\delta$ | Document or user in production; $\delta$ by the book's rule | $n$ counts units, not tokens or lines |
| Canaries | A few per group, inserted 1 to many times, plus controls | Controls near 1.44 bits, allowing for the scatter of a few |
| Public model and tokenizer | One public checkpoint, pinned revision, float32 | Nothing fitted on private text; overlap checked |
| What to train | LoRA, rank 8, on every linear layer | Trainable count printed; chosen on public data |
| Opacus | Validator; hooks; `remove_hooks()` after training | No validator errors |
| Calibrate and train | PRV accountant; Poisson sampling; large batches | PLD agrees with PRV |
| Evaluate | Held-out perplexity against the pretrained and non-private models | Private model beats the pretrained one |
| Memorization | Exposure and greedy extraction, non-private and private | Nothing extracted; one-copy canaries no higher than the controls |
| Report and save | Unit, $\varepsilon$, $\delta$, accountant, public model, tuning; adapter saved after `remove_hooks()` | Statement names what is not covered and the base revision |

: The steps of fine-tuning a language model privately, their defaults and their checks. {#tbl-llmwf-checklist}

## References

- Yu et al., "Differentially Private Fine-tuning of Language Models" (ICLR 2022): LoRA and other adapters with DP-SGD, and their accuracy.
- Li, Tramèr, Liang & Hashimoto, "Large Language Models Can Be Strong Differentially Private Learners" (ICLR 2022): full fine-tuning with ghost clipping, and the hyperparameters that make it work.
- Carlini et al., "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks" (USENIX Security 2019): canaries and exposure.

**Derivations.** This workflow draws on the full edition's chapters on [public data and private fine-tuning](../../book-full/part05-dp-ml/06-public-data.md), [DP-SGD](../../book-full/part05-dp-ml/02-dp-sgd.md) and [memorization and training-data extraction](../../book-full/part02-attacks/05-memorization-extraction.md).
