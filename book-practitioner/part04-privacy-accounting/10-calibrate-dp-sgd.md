# Calibrate and account a DP-SGD run

## Why this matters

An **accountant** is the code that turns a training run's settings into its privacy level $\varepsilon$ under differential privacy (DP). Before the run, it finds the noise that meets your target, and after it, the $\varepsilon$ the run spent. Both can go wrong without any visible sign. The noise may be calibrated with one accountant and reported with another, or the data loader may not sample the way the accountant assumes. A restart may also reset the accountant's count of steps. None of these changes the loss curve, but each makes the privacy claim false.

This chapter does that accounting and trains nothing. Its run is stochastic gradient descent (SGD) with DP, called DP-SGD, on the Adult census data from the UCI Machine Learning Repository. [Train a model with DP-SGD](../part05-dp-ml/13-train-dp-sgd.md) trains it. It trains on 30,000 people in batches of 1,000 for ten epochs, aiming at $\varepsilon = 4$. Each section below is one step, and the rules they apply are explained in [Report and review an ε claim](09-choosing-an-accountant.md).

The script [practitioner_calibrate.py](../../code/part04-privacy-accounting/practitioner_calibrate.py) runs all seven steps in one to two minutes, with Opacus 1.6.0 and `dp-accounting` 0.6.0. Its defaults are Part 5's run. Options set your own run: `--n`, `--batch`, `--epochs`, `--epsilon`, `--dataset`, `--unit` and `--delta`, which defaults to the rule of the next section. Only the planning table stays fixed. The second command plans ten epochs on MNIST's 60,000 images, in batches of 256, at a target of 3.

```bash
uv run python code/part04-privacy-accounting/practitioner_calibrate.py
uv run python code/part04-privacy-accounting/practitioner_calibrate.py --n 60000 --batch 256 --epochs 10 --epsilon 3 --dataset MNIST --unit image
```

## Fix the privacy unit, n and δ {#sec-calib-unit}

The first decision is what one record is, because the guarantee protects exactly that. In Adult each row is one person, so protecting a row protects a person. If one person could own many rows, the number of records $n$ would have to count people, not rows, and training would need the user-level methods of [User-level differential privacy](../part05-dp-ml/07-user-level.md).

The failure probability $\delta$ is roughly the small chance that the guarantee does not hold. The book's rule sets it from $n$ as $\delta \le 1/n^{1.1}$, rounded down to a power of ten, which keeps it well below $1/n$ ([Choosing δ](09-choosing-an-accountant.md#sec-choose-delta)).

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

Here $n$ counts the people of the training split, not the whole dataset, and the rule gives $10^{-5}$. Set the validation rows aside first: they are not trained on, so they are neither in $n$ nor protected. The other two lines are for MNIST and scikit-learn's breast cancer data. Adult's $\delta$ copied to MNIST would break the rule.

## Choose the sampler {#sec-calib-sampler}

The accountants on this page assume **Poisson sampling**, in which each record joins each batch independently with probability $q$, the sampling rate, so batches vary in size. It is the default because the accountants certify how much it hides each record, while the common alternatives cost far more (@sec-calib-samplers). With an expected batch size $B$, the rate is $q = B/n$. A run of $E$ epochs then takes $E/q$ steps, rounded down, which the output calls $T$.

```text
Poisson: B=1000, E=10: q=0.033333, T=300
Opacus DPDataLoader.from_data_loader: q=1/30=0.033333, 30 steps an epoch, 300 in 10 epochs
chosen run uses the loader's q and steps; get_noise_multiplier at that q counts int(E/q)=300: equal True
```

The second line is what training will do. Opacus replaces your loader with a Poisson `DPDataLoader`. Its rate is one over the number of batches in the original loader, $\lceil n/B \rceil$, and it runs that many steps per epoch. Here the batch size divides $n$, so the first two lines agree. When it does not, they differ slightly, and the loader's count is what training runs. So the script takes $q$ and $T$ from the loader, and the third line checks that Opacus's calibration counts the same steps. This step has gone wrong if the training code builds its own loader with `shuffle=True` and a fixed `batch_size`, which the Poisson accountant does not describe. Batches that all have the same size in the logs are the sign ([Amplification by subsampling](05-subsampling.md#sec-sub-schemes)).

## Plan the noise {#sec-calib-plan}

Next come the batch size, the epochs and the noise multiplier $\sigma$. Training clips each example's gradient to a length of at most $C$, the clipping norm, and $\sigma$ is the noise's standard deviation as a multiple of $C$. With the target and the sampler fixed, an accountant finds the smallest $\sigma$ that meets them, so calibrate with the accountant you will report ([The inverse problem](09-choosing-an-accountant.md#sec-choose-inverse)). The script uses two libraries that check each other, both with tight accountants ([Numerical accounting](06-numerical-accounting.md)). Opacus's `get_noise_multiplier`, from `opacus.accountants.utils`, searches with its privacy loss random variable (PRV) accountant and stops within 0.01 of the target $\varepsilon$. Google's `dp_accounting` searches with its privacy loss distribution (PLD) accountant and stops within 0.01 in $\sigma$.

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

The table uses a PLD grid of $10^{-3}$, the step to which the accountant rounds the privacy loss, which keeps each search near a second. Its last column is $\sigma/B$ times 1,000. Averaging over a batch of $B$ divides the noise on the gradient sum by $B$, so that column is the noise on the averaged gradient per unit of clipping norm.

```text
dataset             n  delta     B   E  eps      T  sigma PRV  sigma PLD  sigma/B x1000
Adult           30000  1e-05   100  10    4   3000       0.62       0.62           6.16
Adult           30000  1e-05   300  10    4   1000       0.74       0.74           2.45
Adult           30000  1e-05  1000  10    4    300       0.97       0.98           0.97
Adult           30000  1e-05  1000  10    8    300       0.72       0.73           0.72
Adult           30000  1e-05  1000  20    4    600       1.16       1.16           1.16
```

The block shows 5 of the script's 18 Adult cells. Read it along the batch size first. At $\varepsilon = 4$ and ten epochs, $\sigma$ grows from 0.62 at a batch of 100 to 0.97 at 1,000. Yet the noise on the averaged gradient falls from 6.16 to 0.97, because a larger batch divides the noise among more examples. At that batch, twenty epochs raise $\sigma$ from 0.97 to 1.16.

The table cannot choose the batch, because the noise on the average keeps falling as the batch grows, and so does the number of steps. So learn the epochs and steps your model needs from a non-private run or public data. Then take the largest batch that leaves those steps and fits in memory ([Batch size at a fixed budget](../part05-dp-ml/02-dp-sgd.md#sec-dpsgd-batch)). Compare cells by $\sigma/B$, never by $\sigma$ alone, since a smaller batch has the smaller $\sigma$ but the noisier average. The run takes the largest batch with ten epochs at $\varepsilon = 4$, which is the cell Part 5 trains.

The size of the data changes the picture, as a row from the end of the table shows.

```text
breast cancer     569  1e-04    64  20    1    177       4.96       4.91          77.51
largest gap between PRV and PLD sigma: 1.6% (Adult, B=300, E=10, eps=8)
```

With 569 records, a batch of 64 is a large share of the data, so sampling amplifies little. At $\varepsilon = 1$ the noise on the averaged gradient is 77.51, so small datasets need a large $\varepsilon$, public pretraining, or another method.

The two libraries also check each other. Across all 24 cells their $\sigma$ differ by at most 1.6%. A gap above about 3% points to a call given different inputs. Look for a batch size passed as a rate, epochs passed as steps, or a different $\delta$.

## Check ε with a second accountant {#sec-calib-crosscheck}

Train with the $\sigma$ the calibration returned, here Opacus's 0.9729, unrounded, as Part 5 does. If you round it, round up at the precision the statement prints, three decimals, and never down, since less noise means a larger $\varepsilon$. Before training, compute the planned run's $\varepsilon$ with every accountant at hand, three from Opacus and two from `dp_accounting`.

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

PRV and the PLD, the two tight accountants, agree at 3.99 and 3.98, and both meet the target. The two RDP implementations agree with each other at 4.50, above the target. That is expected of RDP, a valid but looser bound. Opacus may warn that an RDP order failed to converge, or that the best order is the largest it tried. Both concern only the RDP value, which you do not report. The Gaussian differential privacy (GDP) value of 3.34 is the trap. It comes from a central limit theorem (CLT), which approximates the run rather than bounding it, and here it lands below the certified values, so never report it.

The pass line runs three checks. The PLD value must be at most the target, which fails if $\sigma$ was rounded down or the steps differ from the plan. RDP should lie at or above the PLD, give or take the PLD's small grid error. And PRV and PLD must agree within 3%. A larger gap in either of the last two means the accountants were given different runs.

The last line is the PLD's error bar. The PLD rounds each step's privacy loss up to its grid for the value it reports, 3.9806. Rounding down gives an optimistic 3.9656, and the true $\varepsilon$ lies between them. The claim is the upper value, and the lower one is reported next to it as the error bar ([Which accountant](09-choosing-an-accountant.md#sec-choose-decision)). The lower value comes from `privacy_loss_distribution.from_gaussian_mechanism` with `pessimistic_estimate=False` and `use_connect_dots=False`, composed over the steps. The bracket widens as the steps grow, so a long run needs a finer grid to keep it narrow.

## If the loader is not Poisson {#sec-calib-samplers}

Not every training loop samples by the Poisson scheme, so price the alternatives before you train. The script builds the same four `dp_accounting` events as the snippet of [Amplification by subsampling](05-subsampling.md#sec-sub-dpsgd), for the chosen run and with the steps its loader runs.

```text
Poisson (PLD)                              eps=3.98
fixed size, add/remove, sigma/2 (PLD)      eps=23.03
fixed size, replace-one (RDP)              eps=52.60
shuffle, zero-out, no amplification (PLD)  eps=18.51
PLDAccountant supports sampling without replacement: add/remove False, replace-one False
```

Under add/remove neighbours, fixed-size random batches are covered by the Poisson accountant at half the noise multiplier. A record added to a full batch pushes another out, so the sum can move by up to $2C$. On this run that turns 3.98 into 23.03. Under replace-one neighbours the sum also moves by up to $2C$, so these batches are accounted at $\sigma/2$ too. Only the RDP accountant handles them, as the last line shows. They give 52.60, but that number protects a record's value rather than its presence, so it cannot be compared with the add/remove values.

Shuffling once per epoch and cutting the data into batches gets no amplification, so all that is certain is one Gaussian step per epoch, 18.51 here. It holds for zero-out neighbours, which differ by one record's gradient being replaced by zero. Under zero-out a step's clipped sum moves by at most $C$, as under add/remove, so this value does compare with the add/remove ones. Name the relation in the statement. It is below 23.03 here, but on the 100-epoch run of [Report and review an ε claim](09-choosing-an-accountant.md#sec-choose-match) the order reverses. So the fix for a non-Poisson loader is the loader, not the accountant: keep the Poisson loader that Opacus's `make_private` installs by default.

## Train, stop early and resume {#sec-calib-train}

During training, Opacus's optimizer calls `accountant.step` once for every noisy update. That is one call per logical batch, even when `BatchMemoryManager` splits it into smaller physical ones. The script simulates the run's 300 calls and asks for $\varepsilon$ at the end of each epoch.

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

Suppose the validation loss stalls and the run stops after six of its ten planned epochs. The $\varepsilon$ to report is still the planned maximum. Stopping early within a maximum fixed in advance is safe, because the stopped run releases a prefix of the planned run, which the planned guarantee covers. The $\varepsilon$ at the stopping point is not a valid claim, because the decision to stop depended on what the run released. Only a privacy filter that ran from the first step could certify a smaller number ([Adaptive composition](08-adaptive-composition.md)), and this book runs none for subsampled steps. Log $\varepsilon$ every epoch anyway, because it must never decrease, and a drop means the accountant was reset.

Training past the maximum is the opposite case. If you decide after seeing the model to add epochs, the number of steps depends on what the run released. The longer run's $\varepsilon$, computed afterwards, is then the same plug-in mistake. An extension needs a new plan, accounted from the first step. If you might want more epochs, plan the longest run you could train before training starts, and treat stopping sooner as an early stop.

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

The resumed run ends at 3.99, the same as a run that never stopped. Its history holds all 300 steps, the loader's 30 an epoch for ten epochs. Had the restart created a fresh accountant, it would have counted only the last four epochs and reported 2.69, the epoch-4 value above.

## Write the statement {#sec-calib-statement}

The last step writes the claim, with every input a reviewer needs to recompute it. The script fills the template from the run's own parameters, reads $\varepsilon$ from the PLD at the rule's $\delta$ and at $1/n^2$, and rounds both up.

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

The guarantee reads 3.99 because the PLD value of 3.98 was rounded up, never to the nearest value. At the stricter $\delta = 1/n^2$, the same run is 6.30. The accountant never sees the unit, the dataset or the clipping norm, so they come from the options `--unit`, `--dataset` and `--clip`. The last line matters most to a reviewer, because it names privacy spent outside the $\varepsilon$ above. Tuning on the private data adds its own cost ([Hyperparameter tuning](../part05-dp-ml/03-hyperparameter-tuning.md)), and so does every earlier model trained on the same people.

A published statement adds what the script cannot know ([The statement](09-choosing-an-accountant.md#sec-choose-report)). It gives the setting, here the central model covering every checkpoint, and the PLD's grid, `dp_accounting`'s default of $10^{-4}$, with the bracket of step 4. It also names the data outside the run. That means any pretrained weights and their training data, any preprocessing fitted on the private rows, and the validation rows, which are not protected. [Release and governance](../part05-dp-ml/12-release-governance.md) builds the release package around it.

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

## Further reading

- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023): planning a run's privacy budget, and what to report.
- Lebeda, Regehr, Kamath & Steinke, "Avoiding Pitfalls for Privacy Accounting of Subsampled Mechanisms under Composition" (IEEE SaTML 2025): why fixed-size and shuffled batches need their own accounting.

**Derivations.** The results this workflow relies on are derived in the full edition's chapters [Amplification by subsampling](../../book-full/part04-privacy-accounting/05-subsampling.md), [Numerical accounting](../../book-full/part04-privacy-accounting/06-numerical-accounting.md) and [Choosing an accountant](../../book-full/part04-privacy-accounting/09-choosing-an-accountant.md).
