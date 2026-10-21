# Train a model with DP-SGD

## Why this matters

You have a PyTorch training loop on sensitive data, and it must become private. The algorithm is stochastic gradient descent (SGD) with differential privacy (DP), called DP-SGD, which clips each example's gradient and adds Gaussian noise at every step ([DP-SGD](02-dp-sgd.md)). The library is Opacus, and the hard part is not the call that wraps the model but the dozen decisions around it, made in the right order. Most of them are inputs to the privacy guarantee, so they must be settled before the first training step. A mistake in any of them leaves code that trains exactly as before but reports a false $\varepsilon$. That is the privacy level, which bounds what the model can reveal about any one person.

This chapter does the whole job on a real dataset, Adult from the UCI Machine Learning Repository. It holds 48,842 people from a census survey, and the task is to predict whether a person earns more than $50,000 a year. Each section is one step of the job. The script [practitioner_dp_sgd.py](../../code/part05-dp-ml/practitioner_dp_sgd.py) runs every step except tuning, which [practitioner_tuning.py](../../code/part05-dp-ml/practitioner_tuning.py) does. Each runs in one to two minutes without a GPU. Its options `--epsilon` and `--epochs` change the main run; the defaults, 4 and 10, give the numbers below.

```bash
uv run python code/part05-dp-ml/practitioner_dp_sgd.py
uv run python code/part05-dp-ml/practitioner_tuning.py
```

## Fix the privacy unit, n and δ {#sec-trainwf-unit}

The first decision is what one record is, because the guarantee protects exactly that ([User-level differential privacy](07-user-level.md)). In Adult each row is one person, so protecting a row protects a person. Adult has no person identifier, so this rests on how the survey was collected. When your data has one, count distinct identifiers, not rows. If a person has many rows, such as visits or messages, together they move the model much further than one row, and you need user-level training.

The script splits the rows with seed 0. It keeps 30,000 rows as the private training set and 16,842 as the test set. The other 2,000 rows, the stand-in rows from here on, play the part of public data. The script uses them to choose the clipping norm, the length each example's gradient is cut to, and the learning rate, and it never trains the model on them. Treating them as public is an assumption made for the demonstration, since they come from the same survey, and in production this role belongs to genuinely public data. Without public data, the tuning runs on the private rows and its cost is counted in $\varepsilon$, as @sec-trainwf-tune shows.

Next comes $\delta$, the small additive slack that the guarantee allows on top of $\varepsilon$. It is fixed from the number of people $n$ in the private training set and nothing else. The book's default is $\delta \le 1/n^{1.1}$, rounded down to a power of ten, which keeps $\delta$ safely below $1/n$ ([Choosing ε](09-choosing-epsilon.md)).

```{.python excerpt="dpsgd-delta"}
def delta_for(n: int) -> float:
    """The book's default: delta <= 1/n^1.1, rounded down to a power of ten."""
    return 10.0 ** math.floor(math.log10(n**-1.1))
```

Here $n$ is the 30,000 people of the private training set, not the 48,842 of the whole dataset, and the script prints $\delta = 10^{-5}$. The usual mistake is quoting a $\delta$ above $1/n$.

## Preprocess from public facts {#sec-trainwf-preprocess}

Every statistic computed from the private rows is itself a release, and the accountant, the code that adds up the privacy cost of training, does not count it. A scaler fitted on the training set leaks each column's mean and spread. An encoder fitted on it leaks which categories occur, even a country only one person comes from. So the script builds its features from facts that are public before any data is seen.

The category lists are copied from the dataset's documentation. The numeric columns are scaled by public bounds. Ages run from 17 to 90, the largest age the survey records, and hours per week run from 1 to 99. Capital gains and losses go on a log scale, capped at 99,999, the largest amount the survey records. A missing value becomes its own category, so no row is dropped.

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

This gives 91 features. The model is a small network with one hidden layer of 64 units, which has 6,018 weights in all. To check this step in someone else's code, search for every `fit` call and ask what data it sees.

## Train the baselines {#sec-trainwf-baselines}

A private accuracy means little on its own, so train three baselines with the same model, batch size, epochs and optimizer. The non-private model shows the best you can hope for. A model with clipping and no noise separates the cost of clipping from the cost of noise. And always predicting the majority class shows the floor, where a private model whose training collapsed lands.

```text
non-private: test accuracy 0.847
clip only (C=1, no noise): test accuracy 0.845
always '<=50K': test accuracy 0.759
```

Clipping alone costs little here. Had clip-only accuracy been far below non-private, no privacy budget would fix it, because the problem would be the clipping norm or the step size.

## Check the model {#sec-trainwf-validate}

Opacus computes one gradient per example, and some layers make that impossible or break the guarantee. Batch normalization is the common case, because its statistics mix the examples of a batch, so one record changes every other record's gradient. The validator lists such layers before you train.

```{.python excerpt="dpsgd-validate"}
errors = ModuleValidator.validate(model, strict=False)
print(f"validator errors: {len(errors)}")
```

Here it finds none. For a model with such layers, `ModuleValidator.fix` replaces batch normalization with group normalization ([DP-SGD](02-dp-sgd.md#sec-dpsgd-checklist) shows it on a ResNet).

## Choose the clipping norm, batch and epochs {#sec-trainwf-clip}

These settings are fixed before the private data is touched, because choosing them on it costs privacy. Start with the clipping norm $C$. The usual guide is the typical per-example gradient norm on public data or data like it. So the script measures it on the stand-in rows with Opacus's per-example gradients.

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

The median rule would give a $C$ of about 1.4. The script keeps the usual default $C = 1$ instead, just below the median, so every gradient is clipped at the start, and that costs nothing. When every gradient is clipped, each has length exactly $C$ and the noise scales with $C$ too, so the step size times $C$ is the one number that matters ([DP-SGD](02-dp-sgd.md#sec-dpsgd-clipping)).

Next comes the expected batch size $B$, which should be as large as you can afford. It is an expected size because the sampler draws every batch at random (@sec-trainwf-train). The reason is that the noise on the averaged gradient is $\sigma C/B$, where the noise multiplier $\sigma$ is the noise's standard deviation in units of $C$. The script takes a batch of 1,000. Opacus sets the sampling rate $q$, the chance that a row joins a batch, to one over the number of batches per epoch, $1/\lceil n/B\rceil$. That is $B/n = 0.0333$ here, but not when $B$ does not divide $n$, so plan with Opacus's value. An epoch is $1/q$ steps, so 10 epochs give 300 steps. At $\sigma = 1$ the noise on each coordinate of the averaged gradient is then only 0.0010. Fix the number of epochs now, since the noise is calibrated to it.

The optimizer is SGD with momentum 0.9, and its learning rate is chosen on the stand-in rows too. The script trains a non-private model on 1,500 of them at six rates, scores each on the other 500, and keeps the best.

```text
  lr=0.05: 0.814  lr=0.1: 0.816  lr=0.2: 0.826  lr=0.5: 0.792  lr=1: 0.812  lr=2: 0.768
non-private learning rate 0.2; DP-SGD learning rate 2 (x10)
```

DP-SGD then uses ten times the winning rate. That factor is a rule of thumb rather than a measurement. Clipped gradients are shorter and noise changes the best step, so private runs usually start from a step several times the non-private one ([DP-SGD](02-dp-sgd.md#sec-dpsgd-clipping)). No private row has been touched, so this choice costs no privacy.

## Calibrate the noise {#sec-trainwf-calibrate}

Fix the target $\varepsilon$, $\delta$, $q$ and the number of epochs, and let the accountant solve for the smallest noise multiplier that meets them. Use the accountant you will report, here the privacy loss random variable (PRV) accountant, which computes $\varepsilon$ numerically and almost exactly. A looser accountant overestimates $\varepsilon$, and calibrating with it pays for the overestimate with extra noise ([Calibrate and account a DP-SGD run](../part04-privacy-accounting/10-calibrate-dp-sgd.md#sec-calib-plan)).

```{.python excerpt="dpsgd-calibrate"}
sigma = get_noise_multiplier(
    target_epsilon=epsilon,
    target_delta=delta,
    sample_rate=sample_rate,
    epochs=epochs,
    accountant="prv",
)
```

For $\varepsilon = 4$ at $\delta = 10^{-5}$ it prints `sigma=0.973`. The result is only valid for the epochs and sampling rate you gave it. If you later train longer, or change the batch size, recompute $\sigma$ before training, or report the larger $\varepsilon$ the accountant gives for what actually ran.

## Train {#sec-trainwf-train}

The privacy engine wraps the model, the optimizer and the data loader. The optimizer it returns clips each example's gradient and adds the noise, while the loader it returns draws Poisson batches, in which each row joins each batch independently with probability $q$. The accountant assumes exactly this sampling. The loader the script passes in shuffles and cuts fixed batches, so `make_private` replaces it, and that replacement is what makes the assumption true.

```{.python excerpt="dpsgd-setup"}
engine = PrivacyEngine(accountant="prv")
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
model, optimizer, data_loader = engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=sigma,
    max_grad_norm=CLIP,
)  # Poisson sampling is the default: each row joins each batch with probability q
```

The training loop is an ordinary one inside `BatchMemoryManager`. It splits each logical batch of about 1,000 rows into physical batches of at most 500 that fit in memory, and the optimizer steps once per logical batch. So the sampling rate and the accounting do not change.

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

The script leaves `secure_mode` off, so that its noise is seeded and its numbers repeat, and its record says so. For a model you release, pass `secure_mode=True` to `PrivacyEngine`. It draws the noise from a cryptographically secure generator, and it guards against the floating-point flaws of ordinary Gaussian samples ([Discrete and floating-point-safe mechanisms](../part03-dp-foundations/08-discrete-mechanisms.md)).

## Cross-check ε {#sec-trainwf-crosscheck}

An $\varepsilon$ from one library is a claim about that library's arithmetic. Recompute it independently with Google's `dp_accounting`, from the noise multiplier, sampling rate and step count of your plan, while Opacus counts the steps it actually took. Its privacy loss distribution (PLD) accountant, like the PRV accountant, computes $\varepsilon$ almost exactly, but by a different numerical method, so the two should agree to within rounding. Its Rényi differential privacy (RDP) accountant, an older bound that is slightly loose, runs alongside for comparison.

```{.python excerpt="dpsgd-cross-check"}
step = dp_accounting.PoissonSampledDpEvent(sample_rate, dp_accounting.GaussianDpEvent(sigma))
training = dp_accounting.SelfComposedDpEvent(step, steps)
epsilon_pld = dp_accounting.pld.PLDAccountant().compose(training).get_epsilon(delta)
epsilon_rdp = dp_accounting.rdp.RdpAccountant().compose(training).get_epsilon(delta)
```

```text
epsilon PRV (Opacus) = 3.99
epsilon PLD (dp_accounting) = 3.98
epsilon RDP (dp_accounting) = 4.50
```

The PRV and PLD values agree. The RDP accountant gives 4.50 for the same run, and since RDP is a valid but looser bound, a statement must name its accountant. If the two tight accountants disagree by more than rounding, the run did not do what the plan says.

## Tune honestly {#sec-trainwf-tune}

The default is the one this run used. Choose every hyperparameter on public data, or on a public stand-in, and tuning costs nothing. When no public data exists, tuning must run on the private rows. Then every private run whose result influences what you release costs privacy, including the runs you discard ([Hyperparameter tuning](03-hyperparameter-tuning.md)).

The tuning script measures that cost for six learning rates at a total $\varepsilon$ of 4. A single run with no tuning is the reference. A grid runs all six rates and pays for each by composition. A random search runs a number of models drawn from a Poisson distribution with mean 4, gives each a learning rate drawn from the six, and releases only the best. Only the RDP accountant of `dp_accounting` handles a random number of runs, so the script uses RDP throughout. Its single-run noise multiplier is therefore larger than the PRV value of earlier sections.

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

The naive grid is the common mistake. It trains six models at the single-run noise, reports the $\varepsilon$ of one, and actually spends 9.84. The random search pays for its honesty with a modest rise in $\sigma$, from 1.029 to 1.256.

The search itself, with seed 0, drew only two runs, both at the learning rate 0.05. Its model reaches 0.828 on the test set, below the 0.846 of the run tuned on public data. A search with few runs can miss, and you may not redraw until you like the answer, because the guarantee depends on the randomness of the draw. So spend private runs on tuning only when no public data can stand in.

## Measure utility and groups {#sec-trainwf-utility}

Train at several budgets with everything else fixed, and compare against the baselines. Fix the budget you release before these runs, as the script does with $\varepsilon = 4$. The other runs train and are scored on private rows. Choosing $\varepsilon$ from them would be a use of private data that no $\varepsilon$ counts ([Choosing ε](09-choosing-epsilon.md)).

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

Privacy costs almost nothing on Adult down to $\varepsilon = 1$, and only at $\varepsilon = 0.25$ does accuracy drop, by about five points. At budgets of 1 and above, the differences are a thousandth or two on a single seed, too small to read as an effect.

Adult is easy for three reasons. The training set is large, so a batch of 1,000 still has a small sampling rate and divides the noise by 1,000. The model is small, and the noise's total length grows with the square root of the number of trained weights. And the task suits a small network trained for ten epochs. Expect a steeper curve with a model of millions of weights trained from scratch, a few thousand records, a rare class or many more epochs. The running example of [DP-SGD](02-dp-sgd.md#sec-dpsgd-run) already loses several points at $\varepsilon = 1$. Public pretraining is the usual remedy ([Public data and private fine-tuning](06-public-data.md)).

An average can hide a group that pays, so check accuracy by group with exact (Clopper–Pearson) binomial intervals, which stay valid for small groups ([Disparate impact](08-disparate-impact.md)).

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

At $\varepsilon = 4$ neither group pays a measurable price, since each private interval overlaps its non-private one almost entirely. The gap between the groups exists without privacy too, and it is the baseline a private model must be compared against.

## When it goes wrong {#sec-trainwf-troubleshoot}

@tbl-trainwf-trouble maps the common symptoms to their usual cause and fix.

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| The validator reports errors | Batch normalization or an unsupported custom layer | `ModuleValidator.fix`, then build the optimizer on the fixed model's weights and retrain the baseline |
| "Model needs to be in training mode" | A model last used for evaluation | `model.train()` before `make_private` |
| Out of memory | Per-example gradients for a large physical batch | Lower `max_physical_batch_size`, or use ghost clipping: pass `grad_sample_mode="ghost"` and `criterion=` to `make_private`, which then returns the model, optimizer, loss function and loader, and train with that loss |
| Ten times slower than usual | Per-example gradients for every weight | Train fewer weights: freeze a public backbone, use adapters |
| Accuracy near the majority class | Noise too large for the signal | Check $\sigma C/B$; raise the batch, or train fewer epochs so that the accountant needs a smaller $\sigma$; start from a public model |
| $\varepsilon$ looks too good | A shuffling loader with `poisson_sampling=False`, uncounted tuning runs, or epochs changed after calibration | Check that the loader is Opacus's `DPDataLoader`, as the record's sampler does; `engine.get_epsilon` and the cross-check report the same $\varepsilon$ either way. Recount runs and epochs |

: Common DP-SGD failures and their fixes. {#tbl-trainwf-trouble}

Accuracy and privacy fail independently. A run whose $\varepsilon$ is wrong usually trains normally, which is why the cross-check and the audit of [Auditing private training](10-auditing.md) exist.

## Record the run {#sec-trainwf-record}

A reviewer can only check a claim whose inputs are written down, so the script writes a record of the run when it ends. It holds the unit, $n$ and $\delta$, and the accountant's three inputs, the noise multiplier, the sampling rate and the steps. It names the accountant and the library versions, and it gives the PRV, PLD and RDP values of $\varepsilon$. It reads the sampler from the loader Opacus returned and `secure_mode` from the engine, so it cannot claim Poisson sampling the code did not do. It also says how each setting was chosen, and whether that choice is counted in $\varepsilon$. [Release and governance](12-release-governance.md) turns it into a statement.

## Checklist

| Step | Default | Check |
| --- | --- | --- |
| Unit, $n$, $\delta$ | Person; $\delta \le 1/n^{1.1}$ rounded down | $n$ counts people; $\delta < 1/n$ |
| Preprocessing | Public categories and bounds | No `fit` on private rows |
| Baselines | Non-private, clip-only, majority class | Clip-only close to non-private |
| Model | No batch normalization | `ModuleValidator.validate` returns nothing |
| Clipping, batch, epochs | $C$ near or below the median norm on public data; large $B$; epochs fixed | Chosen without private data |
| Noise | `get_noise_multiplier` with the reported accountant | Epochs and $q$ match the run |
| Training | Poisson sampling, `BatchMemoryManager`, `secure_mode=True` for release | The loader is Opacus's |
| Cross-check | PLD from `dp_accounting` | Agrees with PRV within rounding |
| Tuning | On public data; else random search with a Poisson number of runs | Every private run counted |
| Utility and groups | Several budgets; per-group exact intervals | Compared with the baselines |
| Record | Every parameter and version written | $\varepsilon$ recomputable from it |

: The steps of training with DP-SGD, their defaults and their checks. {#tbl-trainwf-checklist}

## Further reading

- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023): the decisions of this chapter, with what to report.
- The Opacus documentation and tutorials (opacus.ai): the privacy engine, `BatchMemoryManager`, ghost clipping and the validator.

**Derivations.** This workflow draws on the full edition's chapters on [DP-SGD](../../book-full/part05-dp-ml/02-dp-sgd.md), [hyperparameter tuning](../../book-full/part05-dp-ml/03-hyperparameter-tuning.md), [disparate impact](../../book-full/part05-dp-ml/08-disparate-impact.md) and [choosing ε](../../book-full/part05-dp-ml/09-choosing-epsilon.md).
