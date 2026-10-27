# Calibrate and account a DP-SGD run

## Why this matters

An **accountant** turns a training run's settings into its privacy level $\varepsilon$ under differential privacy (DP). Before the run it finds the noise that meets your target, and after it the $\varepsilon$ the run spent. Both can go wrong silently. The noise may be calibrated with one accountant and reported with another, the loader may not sample as the accountant assumes, or a restart may reset the step count. None changes the loss curve, but each makes the claim false.

This chapter does the accounting, and trains nothing, for a run of stochastic gradient descent (SGD) with DP, called DP-SGD, on the Adult census data from the UCI Machine Learning Repository. It trains on 30,000 people in batches of 1,000 for ten epochs, aiming at $\varepsilon = 4$, and [Train a model with DP-SGD](../part05-dp-ml/13-train-dp-sgd.md) trains it.

The script [practitioner_calibrate.py](../../code/part04-privacy-accounting/practitioner_calibrate.py) runs the seven steps below in one to two minutes, with Opacus 1.6.0 and `dp-accounting` 0.6.0. Its defaults are Part 5's run. Options such as `--n`, `--batch`, `--epochs` and `--epsilon` set your own, as the second command does for MNIST, and only the planning table stays fixed.

```bash
uv run python code/part04-privacy-accounting/practitioner_calibrate.py
uv run python code/part04-privacy-accounting/practitioner_calibrate.py --n 60000 --batch 256 --epochs 10 --epsilon 3 --dataset MNIST --unit image
```

## Fix the privacy unit, n and δ {#sec-calib-unit}

The guarantee protects exactly one record, so first decide what a record is. In Adult each row is one person. If one person could own many rows, the number of records $n$ would have to count people, and training would need the methods of [User-level differential privacy](../part05-dp-ml/07-user-level.md).

The slack $\delta$ is roughly the chance that the guarantee fails. The book's rule sets it to at most $1/n^{1.1}$, rounded down to a power of ten, well below $1/n$ ([Choosing δ](09-choosing-an-accountant.md#sec-choose-delta)).

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

Here $n$ counts the people of the training split, not all of Adult's rows, so $\delta$ is $10^{-5}$. Set the validation rows aside first, because rows that are not trained on are neither in $n$ nor protected. The lines for MNIST and scikit-learn's breast cancer data show that $\delta$ must be recomputed for each dataset.

## Choose the sampler {#sec-calib-sampler}

The accountants here assume **Poisson sampling**, in which each record joins each batch independently with probability $q$, the sampling rate, so batches vary in size. Other samplers cost far more privacy (@sec-calib-samplers). With an expected batch size $B$, the rate is $q = B/n$. A run of $E$ epochs then takes $E/q$ steps, rounded down, which the output calls $T$.

```text
Poisson: B=1000, E=10: q=0.033333, T=300
Opacus DPDataLoader.from_data_loader: q=1/30=0.033333, 30 steps an epoch, 300 in 10 epochs
chosen run uses the loader's q and steps; get_noise_multiplier at that q counts int(E/q)=300: equal True
```

The second line is what training will run, because Opacus replaces your loader with a Poisson `DPDataLoader`. Its rate is one over the original loader's number of batches, $\lceil n/B \rceil$, with that many steps per epoch. Here $B$ divides $n$, so the two lines agree. When it does not, training runs the loader's numbers. So the script takes $q$ and $T$ from the loader, and the third line checks that Opacus's calibration counts the same steps. This step has gone wrong if the training code builds its own loader with `shuffle=True` and a fixed `batch_size`. The sign is that all batches in the logs have the same size ([Amplification by subsampling](05-subsampling.md#sec-sub-schemes)).

## Plan the noise {#sec-calib-plan}

Next come the batch size, the epochs and the noise multiplier $\sigma$. Training clips each record's gradient to length at most $C$, the clipping norm, and $\sigma$ is the noise's standard deviation as a multiple of $C$. Given the target and the sampler, an accountant finds the smallest $\sigma$ that meets them, so calibrate with the accountant you will report ([The inverse problem](09-choosing-an-accountant.md#sec-choose-inverse)). The script uses tight accountants from two libraries, which check each other ([Numerical accounting](06-numerical-accounting.md)). Opacus's `get_noise_multiplier` searches with its privacy loss random variable (PRV) accountant to within 0.01 in $\varepsilon$. Google's `dp_accounting` searches with its privacy loss distribution (PLD) accountant to within 0.01 in $\sigma$.

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

Pass `accountant="prv"` explicitly, because `get_noise_multiplier` defaults to the Rényi differential privacy (RDP) accountant, a valid but looser bound that asks for more noise. In `dp_accounting`, `LowerEndpointAndGuess(0.3, 1.0)` starts the search at a lower end of 0.3, so lower that number if the call raises an error.

The table uses a PLD grid of $10^{-3}$, the step to which the accountant rounds the privacy loss, so each search takes about a second. Averaging over a batch of $B$ divides the noise on the sum by $B$. So the last column, $\sigma/B$ times 1,000, is the noise on the averaged gradient per unit of clipping norm.

```text
dataset             n  delta     B   E  eps      T  sigma PRV  sigma PLD  sigma/B x1000
Adult           30000  1e-05   100  10    4   3000       0.62       0.62           6.16
Adult           30000  1e-05   300  10    4   1000       0.74       0.74           2.45
Adult           30000  1e-05  1000  10    4    300       0.97       0.98           0.97
Adult           30000  1e-05  1000  10    8    300       0.72       0.73           0.72
Adult           30000  1e-05  1000  20    4    600       1.16       1.16           1.16
```

The block shows 5 of the script's 18 Adult cells. At $\varepsilon = 4$ and ten epochs, a batch of 1,000 needs more noise than a batch of 100, with $\sigma$ at 0.97 instead of 0.62. Yet that noise, averaged over ten times as many records, is smaller on the average, 0.97 instead of 6.16. Twenty epochs instead of ten raise $\sigma$ to 1.16.

The table alone cannot choose the batch, because a larger batch lowers the noise on the average but also the number of steps. So learn the steps your model needs from a non-private run or public data. Then take the largest batch that leaves those steps and fits in memory ([Batch size at a fixed budget](../part05-dp-ml/02-dp-sgd.md#sec-dpsgd-batch)), which for Part 5 is 1,000.

Small datasets change the picture, as a row from the end of the table shows.

```text
breast cancer     569  1e-04    64  20    1    177       4.96       4.91          77.51
largest gap between PRV and PLD sigma: 1.6% (Adult, B=300, E=10, eps=8)
```

With 569 records, a batch of 64 is over a tenth of the data, so sampling amplifies little. The noise on the average is then 77.51 at $\varepsilon = 1$, so small datasets need a large $\varepsilon$, public pretraining, or another method.

The block's last line is the libraries' check on each other. Across all 24 cells their $\sigma$ differ by at most 1.6%. A gap above about 3% means the two calls got different inputs, such as a batch size passed as a rate, epochs passed as steps, or another $\delta$.

## Check ε with a second accountant {#sec-calib-crosscheck}

Train with the calibrated $\sigma$ unrounded, as Part 5 does with Opacus's 0.9729. If you round it, round up to the three decimals the statement prints, because less noise means a larger $\varepsilon$. Before training, compute the planned run's $\varepsilon$ with every accountant at hand, three from Opacus and two from `dp_accounting`.

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

The two tight accountants, PRV and the PLD, agree at 3.99 and 3.98, and both meet the target. The two RDP implementations agree at 4.50, above the target, as a looser bound should, and Opacus's warnings about RDP orders concern only that value. The trap is the Gaussian differential privacy (GDP) value of 3.34. It comes from a central limit theorem (CLT), which approximates the run rather than bounding it, and it lands below the certified values, so never report it.

The pass line runs three checks. The PLD value must be at most the target, which fails if $\sigma$ was rounded down or the steps differ from the plan. RDP must lie at or above the PLD, up to its grid error, and PRV must agree with the PLD within 3%. If either of these fails, the accountants were given different runs.

The last line is the PLD's error bar. Rounding each step's privacy loss up to the grid gives the reported 3.9806, and rounding it down gives an optimistic 3.9656, so the true $\varepsilon$ lies between them. Claim the upper value and report the lower one as its error bar ([Which accountant](09-choosing-an-accountant.md#sec-choose-decision)). The bracket widens with the steps, so a long run needs a finer grid.

## If the loader is not Poisson {#sec-calib-samplers}

Not every training loop samples by the Poisson scheme, so price the alternatives before you train. The script accounts the chosen run, with its loader's steps, under the four samplers of [Amplification by subsampling](05-subsampling.md#sec-sub-dpsgd).

```text
Poisson (PLD)                              eps=3.98
fixed size, add/remove, sigma/2 (PLD)      eps=23.03
fixed size, replace-one (RDP)              eps=52.60
shuffle, zero-out, no amplification (PLD)  eps=18.51
PLDAccountant supports sampling without replacement: add/remove False, replace-one False
```

Fixed-size random batches cost more, because a record added to a full batch pushes another out, so the clipped sum can move by twice the clipping norm $C$. Under add/remove neighbours, the Poisson accountant at half the noise multiplier covers them, which turns 3.98 into 23.03. Under replace-one neighbours only RDP accounts them, at $\sigma/2$ too, since the PLD accountant cannot (last line). Its 52.60 protects a record's value rather than its presence, so it does not compare with the add/remove values.

Shuffling once per epoch and cutting the data into batches gets no amplification, so all that is certain is one Gaussian step per epoch, 18.51 here. That holds for zero-out neighbours, which differ by one record's gradient being replaced by zero, and the statement must name that relation. A step's sum then moves by at most $C$, as under add/remove, so the value compares with the add/remove ones. It beats fixed-size batches here, but on the 100-epoch run of [Report and review an ε claim](09-choosing-an-accountant.md#sec-choose-match) the order reverses. So the fix for a non-Poisson loader is the loader: keep the Poisson loader that Opacus's `make_private` installs.

## Train, stop early and resume {#sec-calib-train}

During training, Opacus's optimizer calls `accountant.step` once per noisy update, which is once per logical batch, even when `BatchMemoryManager` splits it into smaller physical ones. The script simulates the run's 300 calls and asks for $\varepsilon$ after each epoch.

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

Suppose the validation loss stalls and the run stops after six of its ten planned epochs. Stopping early is safe, because the stopped run releases a prefix of the planned run, which the planned guarantee covers. But you still report the planned maximum. The $\varepsilon$ at the stopping point is not a valid claim, because the decision to stop depended on what the run released. Only a privacy filter running from the first step could certify it ([Adaptive composition](08-adaptive-composition.md)). Log $\varepsilon$ every epoch anyway, because it must never decrease, so a drop means the accountant was reset.

Training past the maximum is not safe. If you add epochs after seeing the model, the number of steps depends on what the run released, so the longer run's $\varepsilon$ is not a valid claim either. An extension needs a new plan, accounted from the first step. So plan the longest run you might want, and treat stopping sooner as an early stop.

A job that stops and restarts must carry the accountant with it. Save its `state_dict` with the model and load it before the first new step, as Opacus's `PrivacyEngine.save_checkpoint` and `load_checkpoint` do.

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

The last step writes the claim with every input a reviewer needs to recompute it. The script fills the template from the run's parameters, and rounds up the PLD's $\varepsilon$ at the rule's $\delta$ and at $1/n^2$, where $n$ is the training set's size.

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

The guarantee reads 3.99 because the PLD value printed as 3.98 above is rounded up here, not to the nearest value. The unit, dataset and clipping norm come from options, because the accountant never sees them. The last line matters most to a reviewer, because it names privacy spent outside this $\varepsilon$. Tuning on the private data adds its own cost ([Hyperparameter tuning](../part05-dp-ml/03-hyperparameter-tuning.md)), and so does every earlier model trained on the same people.

A published statement also gives what the script cannot know ([The statement](09-choosing-an-accountant.md#sec-choose-report)). That is the setting, the PLD's grid and bracket, and the data outside the run, such as pretrained weights and the unprotected validation rows.

## Checklist

| Step | Default | Check |
| --- | --- | --- |
| Unit, $n$ and $\delta$ | One person; $n$ from the training split, validation rows excluded; $\delta \le 1/n^{1.1}$, rounded down | $n$ counts people, not rows; $\delta$ recomputed for each dataset |
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
