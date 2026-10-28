# Train a model with DP-SGD

## Why this matters

You have a PyTorch training loop on sensitive data, and it must become private. The algorithm is stochastic gradient descent (SGD) with differential privacy (DP), called DP-SGD, which clips each record's gradient and adds noise at every step ([DP-SGD](02-dp-sgd.md)). Opacus does it with one call that wraps the model. The hard part is the decisions around that call. A mistake in one of them leaves code that trains as before but reports a false privacy level $\varepsilon$.

This chapter trains on Adult, a census survey of 48,842 people from the UCI Machine Learning Repository, to predict whether a person earns more than $50,000 a year. [Calibrate and account a DP-SGD run](../part04-privacy-accounting/10-calibrate-dp-sgd.md) plans the privacy side of this run, and this chapter keeps training, tuning and utility. The script [practitioner_dp_sgd.py](../../code/part05-dp-ml/practitioner_dp_sgd.py) runs every step except tuning, which [practitioner_tuning.py](../../code/part05-dp-ml/practitioner_tuning.py) does. Each runs in one to two minutes without a GPU.

```bash
uv run python code/part05-dp-ml/practitioner_dp_sgd.py
uv run python code/part05-dp-ml/practitioner_tuning.py
```

## Fix the privacy unit, n and δ {#sec-trainwf-unit}

The guarantee protects exactly one record, so first decide what a record is ([User-level differential privacy](07-user-level.md)). Adult has no person identifier, so one row counts as one person because of how the survey was collected. When your data has an identifier, count distinct identifiers, not rows.

The script splits the rows with seed 0 into 30,000 private training rows and 16,842 test rows. The other 2,000, the stand-in rows from here on, play the part of public data. The script chooses settings on them but never trains on them. They come from the same survey, so this is an assumption for the demonstration. In production the role belongs to public data, and without any, tuning costs privacy (@sec-trainwf-tune).

The slack $\delta$ depends on the number of people $n$ in the training set, not in the whole dataset. The book's rule ([How small δ must be](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta)) gives $\delta = 10^{-5}$ for $n = 30{,}000$.

## Preprocess from public facts {#sec-trainwf-preprocess}

Every statistic computed from the private rows is a release that the accountant does not count. A scaler fitted on the training set leaks each column's mean and spread. An encoder fitted on it leaks which categories occur, even a country that only one person comes from. So the script takes its category lists from the dataset's documentation, and scales each number by bounds the survey publishes, such as ages from 17 to 90. Capital gains and losses go on a log scale first, and a missing value becomes its own category.

```{.python excerpt="dpsgd-preprocess"}
def features(rows: pl.DataFrame) -> Array:
    """One-hot categories from the documented lists, numbers scaled by public bounds."""
    rows = rows.with_columns(pl.col("capital-gain", "capital-loss").log1p())
    numeric = [
        np.clip((rows[name].to_numpy() - low) / (high - low), 0.0, 1.0) for name, (low, high) in NUMERIC_BOUNDS.items()
    ]
    encoder = OneHotEncoder(categories=list(CATEGORIES.values()), sparse_output=False)
    text = rows.select(pl.col(list(CATEGORIES)).fill_null("?")).to_numpy()
    encoder.fit(text[:1])  # the categories are given, so fitting learns nothing from the rows
    return np.hstack([np.column_stack(numeric), encoder.transform(text)])
```

This gives 91 features for a network with one hidden layer of 64 units and 6,018 weights. To check this step in other code, find every `fit` call and ask what data it sees.

## Train the baselines {#sec-trainwf-baselines}

A private accuracy means little on its own, so train three baselines with the same model, batch size, epochs and optimizer. The non-private model shows the best you can expect. A model with clipping and no noise separates the cost of clipping from the cost of noise. The majority class shows the floor, where a collapsed private model lands.

```text
non-private: test accuracy 0.847
clip only (C=1, no noise): test accuracy 0.845
always '<=50K': test accuracy 0.759
```

Clipping alone costs little here. If clip-only accuracy were far below non-private, no budget would fix it, because the problem would be the clipping norm or the learning rate.

## Check the model {#sec-trainwf-validate}

Opacus computes one gradient per record, and some layers break that. Batch normalization is the common case, because its statistics let one record change every other record's gradient. The validator lists such layers before you train.

```{.python excerpt="dpsgd-validate"}
errors = ModuleValidator.validate(model, strict=False)
print(f"validator errors: {len(errors)}")
```

Here it finds none. For a model with such layers, `ModuleValidator.fix` replaces batch normalization with group normalization ([DP-SGD](02-dp-sgd.md#sec-dpsgd-checklist) shows it on a ResNet).

## Choose the clipping norm, batch and epochs {#sec-trainwf-clip}

Fix these settings before you touch the private data, because choosing them on it costs privacy. The usual guide for the clipping norm $C$, the length each record's gradient is cut to, is the typical gradient norm on public data. So the script measures it on the stand-in rows.

```{.python excerpt="dpsgd-grad-norms"}
wrapped = GradSampleModule(model)  # adds hooks that keep one gradient per row
wrapped.train()
logits: torch.Tensor = wrapped(split.x)
nn.CrossEntropyLoss()(logits, split.y).backward()  # fills p.grad_sample for every weight p
norms = stacked_norms(wrapped)
```

```text
at initialisation: median norm 1.444  90th percentile 1.538
fraction clipped at C=1: 1.000
```

The median rule would give a $C$ of about 1.4. The script keeps the usual default $C = 1$, just below the median, so every gradient is clipped at the start. That costs nothing, because when every gradient is clipped, only the learning rate times $C$ matters ([DP-SGD](02-dp-sgd.md#sec-dpsgd-clipping)).

Make the batch as large as you can afford. The reason is that the noise on the averaged gradient is $\sigma C/B$, where $B$ is the expected batch size and the noise multiplier $\sigma$ is the noise's standard deviation in units of $C$. With $B = 1{,}000$, Opacus sets the sampling rate $q$, the chance that a row joins a batch, to $1/30 = 0.0333$. Ten epochs then make 300 steps. Fix the epochs now, because the noise is calibrated to them.

The optimizer is SGD with momentum 0.9. The script picks its learning rate on the stand-in rows too. It trains a non-private model on 1,500 of them at six rates and scores each on the other 500.

```text
  lr=0.05: 0.814  lr=0.1: 0.816  lr=0.2: 0.826  lr=0.5: 0.792  lr=1: 0.812  lr=2: 0.768
non-private learning rate 0.2; DP-SGD learning rate 2 (x10)
```

DP-SGD then uses ten times the winning rate, a rule of thumb rather than a measurement. Private runs usually need a larger step, because clipped gradients are shorter ([DP-SGD](02-dp-sgd.md#sec-dpsgd-clipping)). No private row was used, so this choice costs no privacy.

## Calibrate the noise {#sec-trainwf-calibrate}

The target is a privacy level of $\varepsilon = 4$ at the slack $\delta = 10^{-5}$ of @sec-trainwf-unit. For this $q$ and these epochs, Opacus's `get_noise_multiplier` finds the smallest $\sigma$ that meets it. With its privacy loss random variable (PRV) accountant it gives $\sigma = 0.973$, and [Calibrate and account a DP-SGD run](../part04-privacy-accounting/10-calibrate-dp-sgd.md#sec-calib-plan) shows the call. That $\sigma$ holds only for those epochs and that $q$, so recompute it if you change either.

## Train {#sec-trainwf-train}

The privacy engine wraps the model, the optimizer and the data loader. Its optimizer clips each record's gradient and adds the noise. Its loader draws Poisson batches, in which each row joins each batch independently with probability $q$, as the accountant assumes. The script's own loader cuts fixed batches, so `make_private` must replace it.

```{.python excerpt="dpsgd-setup"}
engine = PrivacyEngine(accountant="prv")
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
model, optimizer, data_loader = engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=sigma,
    max_grad_norm=clip,
)  # Poisson sampling is the default: each row joins each batch with probability q
```

The training loop runs inside `BatchMemoryManager`, which splits each logical batch of about 1,000 rows into physical batches of at most 500. The optimizer still steps once per logical batch, so the accounting does not change.

```{.python excerpt="dpsgd-train-loop"}
with BatchMemoryManager(
    data_loader=data_loader, max_physical_batch_size=PHYSICAL_BATCH, optimizer=optimizer
) as batches:
    for x, y in batches:  # Opacus splits each logical batch; it steps once per logical batch
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
```

```text
steps=300  q=0.0333
epsilon (Opacus PRV) = 3.99
test accuracy at epsilon=4: 0.846
```

The script leaves `secure_mode` off, so that its noise is seeded and its numbers repeat, and its record says so. For a model you release, pass `secure_mode=True` to `PrivacyEngine`. It draws the noise from a secure generator and guards against the floating-point flaws of ordinary Gaussian samples ([Discrete and floating-point-safe mechanisms](../part03-dp-foundations/08-discrete-mechanisms.md)).

## Cross-check ε {#sec-trainwf-crosscheck}

An $\varepsilon$ from one library is a claim about that library's arithmetic. So the script recomputes it with Google's `dp_accounting`, whose privacy loss distribution (PLD) accountant gives 3.98 against Opacus's 3.99. The looser Rényi differential privacy (RDP) accountant gives 4.50, so a statement must name its accountant. [Calibrate and account a DP-SGD run](../part04-privacy-accounting/10-calibrate-dp-sgd.md#sec-calib-crosscheck) runs this check and says what a disagreement means.

## Tune honestly {#sec-trainwf-tune}

This run chose every hyperparameter on a public stand-in, so tuning cost nothing. Without public data, tuning runs on the private rows, and every private run that influences the release costs privacy, the discarded ones too ([Hyperparameter tuning](03-hyperparameter-tuning.md)).

The tuning script measures that cost for six learning rates at a total $\varepsilon$ of 4. A grid runs all six rates and pays for each by composition. A random search runs a Poisson number of models with mean 4, each at a rate drawn from the six, and releases only the best. Only RDP handles a random number of runs, so the script uses it throughout, and its single-run $\sigma$ is larger than the PRV value above.

```{.python excerpt="tuning-cost"}
def random_search(sigma: float) -> dp_accounting.DpEvent:
    """A Poisson number of runs with mean 4 (shape=inf selects the Poisson case); only the best is released."""
    return dp_accounting.dp_event.RepeatAndSelectDpEvent(one_run(sigma), mean=MEAN_RUNS, shape=math.inf)


def sigma_for(strategy: Callable[[float], dp_accounting.DpEvent], epsilon: float, delta: float) -> float:
    """The smallest noise multiplier per run that keeps the whole strategy within (epsilon, delta).

    The search starts at 0.1 and tries 1 first.
    """
    return dp_accounting.calibrate_dp_mechanism(
        dp_accounting.rdp.RdpAccountant, strategy, epsilon, delta, dp_accounting.LowerEndpointAndGuess(0.1, 1.0)
    )
```

```text
one run, no tuning: sigma=1.029
grid of 6, all runs counted: sigma=1.816
random search, Poisson mean 4 runs: sigma=1.256
naive grid of 6 at sigma=1.029: reports epsilon=4.00
naive grid, true epsilon of all 6 runs = 9.84
```

The naive grid is the common mistake. It trains six models at the single-run noise, reports the $\varepsilon$ of one, and actually spends 9.84. The random search costs a modest rise in $\sigma$, from 1.029 to 1.256.

With seed 0, the search drew only two runs, both at the learning rate 0.05. So its model reaches only 0.828 on the test set, against 0.846 for the run tuned on public data. Do not redraw until you like the answer, because the guarantee depends on the randomness of the draw. So spend private runs on tuning only when no public data can stand in.

## Measure utility and groups {#sec-trainwf-utility}

Train at several budgets with everything else fixed, and compare against the baselines. Fix the released budget beforehand, as the script does with $\varepsilon = 4$, because the script scores these runs on private rows that no $\varepsilon$ counts ([Choosing ε](09-choosing-epsilon.md)).


| Run | Noise multiplier | Test accuracy |
| --- | --- | --- |
| Non-private | — | 0.847 |
| Clipped, no noise | 0 | 0.845 |
| $\varepsilon = 8$ | 0.722 | 0.846 |
| $\varepsilon = 4$ | 0.973 | 0.846 |
| $\varepsilon = 2$ | 1.431 | 0.845 |
| $\varepsilon = 1$ | 2.383 | 0.843 |
| $\varepsilon = 0.25$ | 8.125 | 0.793 |
| Always "≤50K" | — | 0.759 |

: Test accuracy on Adult against the privacy budget, with $\delta = 10^{-5}$ and one seed. {#tbl-trainwf-utility}

Privacy costs almost nothing on Adult down to $\varepsilon = 1$. Only at $\varepsilon = 0.25$ does accuracy drop, by about five points, and the other differences are too small to read on one seed.

Adult is easy. Its training set is large, so a batch of 1,000 still has a small sampling rate and divides the noise by 1,000. Its model is small, and the noise's total length grows with the square root of the number of trained weights. So expect a steeper curve with millions of weights trained from scratch, a few thousand records, a rare class or many more epochs. Even here, five points are lost at $\varepsilon = 0.25$ (@tbl-trainwf-utility), and public pretraining is the usual remedy: on the 1,500 public stand-in rows it wins most of them back ([Public data and private fine-tuning](06-public-data.md#sec-pub-run)).

An average can hide a group that pays. So check accuracy by group with exact (Clopper–Pearson) intervals, which stay valid for small groups ([Disparate impact](08-disparate-impact.md)).

```{.python excerpt="dpsgd-per-group"}
in_group = sex == group
correct, total = int(right[in_group].sum()), int(in_group.sum())
interval = binomtest(correct, total).proportion_ci(confidence_level=0.95, method="exact")
```

```text
non-private Female: 0.924 [0.917, 0.931]  (5,186/5,613)
non-private Male: 0.808 [0.800, 0.815]  (9,071/11,229)
DP Female: 0.922 [0.915, 0.929]  (5,177/5,613)
DP Male: 0.807 [0.800, 0.815]  (9,065/11,229)
```

At $\varepsilon = 4$ neither group pays a measurable price, since each private interval almost covers its non-private one. The gap between the groups exists without privacy too, so it is the baseline for the private model.

## When it goes wrong {#sec-trainwf-troubleshoot}

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| The validator reports errors | Batch normalization or an unsupported custom layer | `ModuleValidator.fix`, then build the optimizer on the fixed model's weights and retrain the baseline |
| "Model needs to be in training mode" | A model last used for evaluation | `model.train()` before `make_private` |
| Out of memory | Per-example gradients for a large physical batch | Lower `max_physical_batch_size`, or use ghost clipping: pass `grad_sample_mode="ghost"` and `criterion=` to `make_private`, which then returns the model, optimizer, loss function and loader, and train with that loss |
| Ten times slower than usual | Per-example gradients for every weight | Train fewer weights: freeze a public backbone, use adapters |
| Accuracy near the majority class | Noise too large for the signal | Check $\sigma C/B$; raise the batch, or train fewer epochs so that the accountant needs a smaller $\sigma$; start from a public model |
| $\varepsilon$ looks too good | A shuffling loader with `poisson_sampling=False`, uncounted tuning runs, or epochs changed after calibration | Check that the loader is Opacus's `DPDataLoader`, as the record's sampler does; `engine.get_epsilon` and the cross-check report the same $\varepsilon$ either way. Recount runs and epochs |

: Common DP-SGD failures and their fixes. {#tbl-trainwf-trouble}

Accuracy and privacy fail independently. A run whose $\varepsilon$ is wrong usually trains normally, so only the cross-check and the audit of [Auditing private training](10-auditing.md) catch it.

## Record the run {#sec-trainwf-record}

A reviewer can only check a claim whose inputs are on record. So the script ends by writing a record of the run. [Release and governance](12-release-governance.md#sec-relgov-record) lists its fields and turns it into a statement. The script reads the sampler from the loader Opacus returned and `secure_mode` from the engine, so the record cannot claim Poisson sampling that the code did not do.

## Checklist

| Step | Default | Check |
| --- | --- | --- |
| Unit, $n$, $\delta$ | Person; $\delta$ by the book's rule | $n$ counts people in the training set |
| Preprocessing | Public categories and bounds | No `fit` on private rows |
| Baselines | Non-private, clip-only, majority class | Clip-only close to non-private |
| Model | No batch normalization | `ModuleValidator.validate` returns nothing |
| Clipping, batch, epochs | $C$ near or below the median norm on public data; large $B$; epochs fixed | Chosen without private data |
| Noise and cross-check | As in [Calibrate and account a DP-SGD run](../part04-privacy-accounting/10-calibrate-dp-sgd.md) | Epochs and $q$ match the run |
| Training | Poisson sampling, `BatchMemoryManager`, `secure_mode=True` for release | The loader is Opacus's |
| Tuning | On public data; else random search with a Poisson number of runs | Every private run counted |
| Utility and groups | Several budgets; per-group exact intervals | Compared with the baselines |
| Record | Every parameter and version written | $\varepsilon$ recomputable from it |

: The steps of training with DP-SGD, their defaults and their checks. {#tbl-trainwf-checklist}

## References

- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023): the decisions of this chapter, with what to report.
- The Opacus documentation and tutorials (opacus.ai): the privacy engine, `BatchMemoryManager`, ghost clipping and the validator.

**Derivations.** This workflow draws on the full edition's chapters on [DP-SGD](../../book-full/part05-dp-ml/02-dp-sgd.md), [hyperparameter tuning](../../book-full/part05-dp-ml/03-hyperparameter-tuning.md), [disparate impact](../../book-full/part05-dp-ml/08-disparate-impact.md) and [choosing ε](../../book-full/part05-dp-ml/09-choosing-epsilon.md).
