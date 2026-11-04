# Calibrate and account a DP-SGD run

## Why this matters

An **accountant** maps settings for stochastic gradient descent (SGD) with differential privacy (DP) to privacy level $\varepsilon$. Before training, it finds the noise needed for a target; afterward, it reports the run's $\varepsilon$. Both calculations can be wrong without an obvious training signal. Calibration and reporting may use different accountants, the loader may use a different sampler than assumed, or a restart may reset the step count. Any mismatch invalidates the claim without changing the loss curve.

This chapter accounts for DP-SGD on Adult from the UCI Machine Learning Repository; it does not train a model. The planned run uses 30,000 people, batches of 1,000, ten epochs and target $\varepsilon = 4$. [Train a model with DP-SGD](../part05-dp-ml/13-train-dp-sgd.md) performs the training.

The script [practitioner_calibrate.py](../../code/part04-privacy-accounting/practitioner_calibrate.py) runs these steps in one to two minutes with Opacus 1.6.0 and `dp-accounting` 0.6.0. Defaults match Part 5. Options `--n`, `--batch`, `--epochs` and `--epsilon` configure another run, as shown for MNIST in the second command.

```bash
uv run python code/part04-privacy-accounting/practitioner_calibrate.py
uv run python code/part04-privacy-accounting/practitioner_calibrate.py --n 60000 --batch 256 --epochs 10 --epsilon 3 --dataset MNIST --unit image
```

## Fix the privacy unit, n and δ {#sec-calib-unit}

The guarantee protects one record, so define the record first. Each Adult row represents one person. If a person could have multiple rows, $n$ must count people, as in [User-level differential privacy](../part05-dp-ml/07-user-level.md).

Slack $\delta$ bounds the probability of the guarantee's exception. Set it using the book's rule: at most $1/n^{1.1}$, rounded down to a power of ten ([How small δ must be](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta)).

```{.python excerpt="calibrate-delta"}
def delta_for(n: int) -> float:
    """The book's default delta: at most 1/n^1.1, rounded down to a power of ten."""
    return 10.0 ** math.floor(math.log10(n**-1.1))
```

```text
Adult rows: 48,842; Part 5's training split: n=30,000
delta_for(30,000) = 1e-05
delta_for(60,000) = 1e-06
delta_for(569) = 1e-04
```

Here, $n$ counts people in the training split, not all Adult rows, so $\delta$ is $10^{-5}$. Set validation rows aside because the run neither trains on nor protects them. The other examples show that $\delta$ must be recomputed for each dataset.

## Choose the sampler {#sec-calib-sampler}

The accountants assume **Poisson sampling**: each record independently joins a batch with probability $q$, so batch sizes vary. Other samplers can cost much more privacy (@sec-calib-samplers). For an expected batch size $B$, the sampling rate is $q = B/n$. A run of $E$ epochs takes $E/q$ steps, rounded down; the output calls this $T$.

```text
Poisson: B=1000, E=10: q=0.033333, T=300
Opacus DPDataLoader.from_data_loader: q=1/30=0.033333, 30 steps an epoch, 300 in 10 epochs
chosen run uses the loader's q and steps; get_noise_multiplier at that q counts int(E/q)=300: equal True
```

Training uses the Poisson `DPDataLoader` that Opacus creates from the input loader. Its rate is the reciprocal of the input loader's batch count, $\lceil n/B \rceil$, and it takes that many steps per epoch. Here, $B$ divides $n$, so both plans agree. Otherwise, the script reads $q$ and $T$ from the loader and checks that Opacus calibration counts the same steps.

This assumption fails if training uses a custom loader with `shuffle=True` and fixed `batch_size`; the logged batches will all have the same size ([Amplification by subsampling](05-subsampling.md#sec-sub-schemes)).

## Plan the noise {#sec-calib-plan}

Choose batch size, epochs and noise multiplier $\sigma$. Training clips each record's gradient to length at most $C$, the clipping norm, and $\sigma$ is the standard deviation of the noise as a multiple of $C$. For a given target and sampler, an accountant solves for the smallest acceptable $\sigma$. Calibrate with the accountant you will report ([The inverse problem](09-choosing-an-accountant.md#sec-choose-inverse)). The script cross-checks two tight accountants ([Numerical accounting](06-numerical-accounting.md)): Opacus's privacy loss random variable (PRV), accurate to 0.01 in $\varepsilon$, and `dp_accounting`'s privacy loss distribution (PLD), accurate to 0.01 in $\sigma$.

```{.python excerpt="calibrate-sigma"}
def sigma_opacus(epsilon: float, delta: float, q: float, epochs: int) -> float:
    """Opacus: the sigma whose PRV epsilon after `epochs` is at most the target (to 0.01 in epsilon)."""
    return get_noise_multiplier(
        target_epsilon=epsilon, target_delta=delta, sample_rate=q, epochs=epochs, accountant="prv"
    )


def sigma_dp_accounting(epsilon: float, delta: float, q: float, steps: int) -> float:
    """dp_accounting: the same search with a PLD accountant (to 0.01 in sigma)."""

    def training_run(sigma: float) -> DpEvent:
        return SelfComposedDpEvent(PoissonSampledDpEvent(q, GaussianDpEvent(sigma)), steps)

    def accountant() -> PLDAccountant:
        return PLDAccountant(value_discretization_interval=PLD_GRID)

    return calibrate_dp_mechanism(accountant, training_run, epsilon, delta, LowerEndpointAndGuess(0.3, 1.0), tol=0.01)
```

Specify `accountant="prv"` because `get_noise_multiplier` otherwise uses the looser Rényi differential privacy (RDP) accountant and returns more noise. If `dp_accounting` raises an error, reduce 0.3 in `LowerEndpointAndGuess(0.3, 1.0)`, the initial lower search bound.

The table uses a PLD grid of $10^{-3}$, the step to which the accountant rounds the privacy loss, so each search takes about a second. The average over a batch of $B$ records divides the noise on the sum by $B$. The final column, $\sigma/B$ times 1,000, gives average-gradient noise in units of the clipping norm.

```text
dataset             n  delta     B   E  eps      T  sigma PRV  sigma PLD  sigma/B x1000
Adult           30000  1e-05   100  10    4   3000       0.62       0.62           6.16
Adult           30000  1e-05   300  10    4   1000       0.74       0.74           2.45
Adult           30000  1e-05  1000  10    4    300       0.97       0.98           0.97
Adult           30000  1e-05  1000  10    8    300       0.72       0.73           0.72
Adult           30000  1e-05  1000  20    4    600       1.16       1.16           1.16
```

The block shows five of the script's 18 Adult cases. At $\varepsilon = 4$ and ten epochs, batch size 1,000 requires more noise than batch size 100: 0.97 rather than 0.62. Averaging across ten times as many records reduces noise on the average from 6.16 to 0.97. Twenty epochs increase $\sigma$ to 1.16.

The table does not determine batch size: larger batches reduce noise on the average but also reduce the number of steps. Estimate the required step count using a non-private run or public data, then choose the largest batch that preserves those steps and fits memory. Part 5 uses 1,000 ([Batch size at a fixed budget](../part05-dp-ml/02-dp-sgd.md#sec-dpsgd-batch)).

A row from the end of the table shows that small datasets are different.

```text
breast cancer     569  1e-04    64  20    1    177       4.96       4.91          77.51
largest gap between PRV and PLD sigma: 1.6% (Adult, B=300, E=10, eps=8)
```

With 569 records, batch size 64 exceeds one tenth of the dataset, so subsampling provides little amplification. At $\varepsilon = 1$, noise on the average is 77.51. Small datasets may require a larger $\varepsilon$, public pretraining or another method.

The final line compares the two libraries. Across all 24 cases, their $\sigma$ values differ by at most 1.6%. A gap above about 3% suggests mismatched inputs, such as batch rate, epoch-to-step conversion or slack $\delta$.

## Check ε with a second accountant {#sec-calib-crosscheck}

Train with the calibrated, unrounded $\sigma$ as Part 5 does with Opacus's 0.9729. If rounding is necessary, round up to the three decimals in the statement; lower noise increases $\varepsilon$. Before training, compute the planned run's $\varepsilon$ with each available accountant. Include Gaussian differential privacy (GDP) in the comparison, but do not report it as a bound.

```{.python excerpt="calibrate-crosscheck"}
def crosscheck_epsilon(sigma: float, q: float, steps: int, delta: float) -> dict[str, float]:
    """Epsilon of one planned run by five accountants (GDP is the trap: it is not an upper bound)."""
    epsilons: dict[str, float] = {}
    for name, opacus_accountant in [("Opacus RDP", RDPAccountant()), ("Opacus PRV", PRVAccountant())]:
        opacus_accountant.history = [(sigma, q, steps)]  # what `steps` calls of .step() would record
        epsilons[name] = opacus_accountant.get_epsilon(delta)
    run = SelfComposedDpEvent(PoissonSampledDpEvent(q, GaussianDpEvent(sigma)), steps)
    epsilons["dp_accounting RDP"] = RdpAccountant().compose(run).get_epsilon(delta)
    epsilons["dp_accounting PLD"] = PLDAccountant().compose(run).get_epsilon(delta)
    gdp = GaussianAccountant()
    gdp.history = [(sigma, q, steps)]
    epsilons["Opacus GDP"] = gdp.get_epsilon(delta)
    return epsilons
```

```text
chosen cell: Adult, B=1000, E=10, eps=4: sigma=0.972900 (Opacus PRV, as Part 5)
Opacus RDP         eps=4.50
Opacus PRV         eps=3.99
dp_accounting RDP  eps=4.50
dp_accounting PLD  eps=3.98
Opacus GDP         eps=3.34
PLD <= target: True; RDP >= PLD: True; |PRV - PLD|/PLD < 3%: True
crosscheck: pass
PLD bracket (grid 1e-4): lower 3.9656 (optimistic), upper 3.9806 (pessimistic); width 0.0150
```

The tight PRV and PLD accountants agree at 3.99 and 3.98, both meeting the target. The two RDP accountants agree at 4.50, a looser bound above target. GDP's 3.34 is unsafe: it uses a central limit theorem (CLT) approximation rather than a bound and falls below certified results. Do not report it.

The pass line checks three conditions: PLD is at most the target, RDP is at least PLD within grid error, and PRV agrees with PLD within 3%. The PLD check fails if $\sigma$ was rounded down or the step count differs from the plan. Any other failed comparison indicates mismatched accountant inputs.

The final line gives the PLD error interval. Rounding each step's privacy loss up gives 3.9806; rounding down gives 3.9656. The true $\varepsilon$ lies between them. Report the upper value and the lower value as its error bar ([Which accountant](09-choosing-an-accountant.md#sec-choose-decision)). Longer runs widen the interval, so use a finer grid.

## If the loader is not Poisson {#sec-calib-samplers}

Some training loops use non-Poisson samplers with fixed batch sizes. Calculate their costs before training; the script evaluates four samplers from [Amplification by subsampling](05-subsampling.md#sec-sub-dpsgd).

```text
Poisson (PLD)                              eps=3.98
fixed size, add/remove, sigma/2 (PLD)      eps=23.03
fixed size, replace-one (RDP)              eps=52.60
shuffle, zero-out, no amplification (PLD)  eps=18.51
PLDAccountant supports sampling without replacement: add/remove False, replace-one False
```

Fixed-size random batches cost more because an included record replaces another, allowing the clipped sum to move by twice the clipping norm $C$. Under add/remove neighbours, account with a Poisson accountant at $\sigma/2$; the cost rises from 3.98 to 23.03. For replace-one neighbours, only RDP supports the sampler, also at $\sigma/2$ (last line). Its value 52.60 protects record values, not record presence, and is not comparable with add/remove values.

Shuffle-and-partition sampling has no amplification. The valid bound is one Gaussian step per epoch, 18.51 here, under zero-out neighbours, where one gradient is set to zero. State this relation explicitly. The sum changes by at most $C$, as under add/remove, so the values are comparable. Shuffle-and-partition is cheaper than fixed-size random batches here, but the comparison reverses in the 100-epoch run of [Matching the accountant to the code](09-choosing-an-accountant.md#sec-choose-match). Prefer Opacus's Poisson loader when possible.

## Train, stop early and resume {#sec-calib-train}

During training, Opacus's optimizer calls `accountant.step` once per noisy update, or logical batch. This remains true when `BatchMemoryManager` divides it into smaller physical batches. The script simulates 300 calls and computes $\varepsilon$ after each epoch.

```{.python excerpt="calibrate-train"}
def train_epochs(
    accountant: PRVAccountant, sigma: float, q: float, steps_per_epoch: int, epochs: int, delta: float
) -> list[float]:
    """Step the accountant once per logical batch for `epochs` epochs; epsilon at the end of each epoch."""
    epsilons: list[float] = []
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            accountant.step(noise_multiplier=sigma, sample_rate=q)  # what the optimizer's step does
        epsilons.append(float(accountant.get_epsilon(delta)))
    return epsilons
```

If validation loss stalls and training stops after six of ten epochs, the planned guarantee still covers this prefix. Report the planned maximum because stopping depends on released outputs; the realized six-epoch $\varepsilon$ is not a valid claim. A privacy filter initialized at the first step could certify adaptive stopping ([Adaptive composition](08-adaptive-composition.md)). Log $\varepsilon$ each epoch; it must not decrease. A decrease indicates an accountant reset.

Do not train beyond the planned maximum. Adding epochs after inspecting the model makes step count data-dependent. Create a new plan and account from the first step, or plan the longest run in advance and treat shorter runs as early stops.

When a job resumes, preserve accountant state. Save the accountant's `state_dict` with the model and restore it before the next step. Opacus's `PrivacyEngine.save_checkpoint` and `load_checkpoint` handle this.

```{.python excerpt="calibrate-resume"}
state = accountant.state_dict()  # saved with the model; PrivacyEngine.save_checkpoint stores it too
# ... the job stops, and later restarts ...
restarted = PRVAccountant()
restarted.load_state_dict(state)  # without this, epsilon would restart from zero
resumed = train_epochs(restarted, sigma, q, steps_per_epoch, epochs - stop, delta)
steps_recorded = sum(steps for _, _, steps in restarted.history)
```

```text
steps per epoch: 30; planned: 10 epochs = 300 steps
epoch  1: eps=1.70
epoch  2: eps=2.09
epoch  4: eps=2.69
epoch  6: eps=3.18
epsilon never decreases over 10 epochs: True
stopped early at epoch 6: eps=3.18; report the planned maximum: eps=3.99
after resuming at epoch 6: eps at epoch 10=3.99; uninterrupted run: eps=3.99; equal: True
steps recorded: 300; planned (the loader's 10 x 30): 300; equal: True
```

The resumed run ends at 3.99, like a run that never stopped, with all 300 steps in its history. A fresh accountant would have counted only the last four epochs and reported 2.69.

## Write the statement {#sec-calib-statement}

The last step writes the claim with every input that a reviewer needs to recompute it. The script fills the template from the run. It rounds up the PLD's $\varepsilon$ at two values of the slack, the rule's $\delta$ and $1/n^2$, where $n$ is still the number of training records.

```{.python excerpt="calibrate-statement"}
def privacy_statement(
    unit: str, dataset: str, n: int, batch: int, epochs: int, sigma: float, clip: float, delta: float, calibration: str
) -> str:
    """The filled template of the reporting page: every input a reviewer needs to recompute epsilon."""
    q, _, steps = opacus_loader_plan(n, batch, epochs)  # the steps the Poisson loader runs
    run = SelfComposedDpEvent(PoissonSampledDpEvent(q, GaussianDpEvent(sigma)), steps)
    epsilon = round_up(PLDAccountant().compose(run).get_epsilon(delta))
    epsilon_tiny_delta = round_up(PLDAccountant().compose(run).get_epsilon(1 / n**2))
    return f"""\
Unit: one {unit} (a row of {dataset}); neighbouring datasets differ by adding or removing one {unit}
Data: n = {n:,}; Poisson sampling with q = {q:.6f} (B = {batch}); T = {steps:,} steps ({epochs} epochs)
Mechanism: DP-SGD, clipping norm C = {clip}, noise multiplier sigma = {sigma:.3f}
Accountant: PLD, dp-accounting {version("dp-accounting")}; {calibration}
Guarantee: (epsilon = {epsilon:.2f}, delta = {delta:.0e}); epsilon = {epsilon_tiny_delta:.2f} at delta = 1/n^2 = {1 / n**2:.1e}
Not covered: hyperparameter tuning runs (see Part 5), and any model released before this one"""
```

```text
Unit: one person (a row of Adult); neighbouring datasets differ by adding or removing one person
Data: n = 30,000; Poisson sampling with q = 0.033333 (B = 1000); T = 300 steps (10 epochs)
Mechanism: DP-SGD, clipping norm C = 1.0, noise multiplier sigma = 0.973
Accountant: PLD, dp-accounting 0.6.0; calibrated with Opacus 1.6.0 (PRV)
Guarantee: (epsilon = 3.99, delta = 1e-05); epsilon = 6.30 at delta = 1/n^2 = 1.1e-09
Not covered: hyperparameter tuning runs (see Part 5), and any model released before this one
```

The guarantee reads 3.99 because the script rounds the PLD value up, while the cross-check above printed it to the nearest value, 3.98. The unit, the dataset and the clipping norm come from options, because the accountant never sees them. The last line matters most to a reviewer, because it names privacy spent outside this $\varepsilon$. Examples are tuning on the private data ([Hyperparameter tuning](../part05-dp-ml/03-hyperparameter-tuning.md)) and earlier models trained on the same people.

A published statement also gives what the script cannot know ([Write the statement](../part05-dp-ml/12-release-governance.md#sec-relgov-statement)). That is the setting, the grid and bracket of the PLD, and the data outside the run, such as pretrained weights and the unprotected validation rows.

## Checklist

| Step | Default | Check |
| --- | --- | --- |
| Unit, $n$ and $\delta$ | One person; $n$ from the training split, validation rows excluded; $\delta$ by the [book's rule](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta) | $n$ counts people, not rows; $\delta$ recomputed for each dataset |
| Sampler | Poisson; $q$ and $T$ from Opacus's loader, $\lceil n/B \rceil$ steps an epoch | Calibration counts the same steps; batch sizes vary |
| Plan the noise | Calibrate with PRV or PLD, the accountant you report | Two libraries agree within about 3% |
| Cross-check | Train at the calibrated $\sigma$, or round it up | PLD at most the target; RDP above PLD; GDP never reported; PLD bracket reported |
| Other samplers | Keep Opacus's Poisson loader | No `shuffle=True` loader with fixed batches in the training code |
| Train and resume | One accountant step per logical batch; save the accountant's state; any extension inside the plan | $\varepsilon$ never decreases; steps recorded equal the loader's steps |
| Statement | $\varepsilon$ at the planned maximum, rounded up, at $\delta$ and $1/n^2$ | Every input to recompute it; tuning, earlier models, pretraining, preprocessing and validation named |

: The steps of calibrating and accounting a DP-SGD run. {#tbl-calib-checklist}

## References

- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023): planning a run's privacy budget, and what to report.
- Lebeda, Regehr, Kamath & Steinke, "Avoiding Pitfalls for Privacy Accounting of Subsampled Mechanisms under Composition" (IEEE SaTML 2025): why fixed-size and shuffled batches need their own accounting.

**Derivations.** The results this workflow relies on are derived in the full edition's chapters [Amplification by subsampling](../../book-full/part04-privacy-accounting/05-subsampling.md), [Numerical accounting](../../book-full/part04-privacy-accounting/06-numerical-accounting.md) and [Choosing an accountant](../../book-full/part04-privacy-accounting/09-choosing-an-accountant.md).
