# Hyperparameter tuning and private selection

## Why this matters

Stochastic gradient descent (SGD) with clipped gradients and Gaussian noise satisfies differential privacy (DP) at the accountant's reported level for one run; this is [DP-SGD](02-dp-sgd.md). Hyperparameter search trains and evaluates multiple models on private data, then selects one. The reported privacy level $\varepsilon$ must account for the entire search.

Selecting only the best of a fixed number of runs can cost as much as releasing every model. On Adult, six runs each at $\varepsilon=4$ cost 9.84 in total. If the run count is drawn from a distribution fixed in advance, the search can cost only slightly more than one run.

## The procedure and the report {#sec-hpt-practice}

The least costly tuning uses no private data. Runs on public or proxy data and published settings for similar tasks incur no privacy cost. Fix a large batch, epoch count and small clipping norm in advance. When the norm is below typical gradient size, accuracy often depends mainly on the product of clipping norm and learning rate, so search only that product ([DP-SGD](02-dp-sgd.md#sec-dpsgd-clipping)).

If private-data tuning is necessary, specify the complete search in advance: candidate set, distribution for the number of runs (@sec-hpt-rdp), private score (@sec-hpt-select) and validation set. Extending or restarting the search after seeing results makes it data-dependent and invalidates the stated accounting.

Account for all runs. For two or three candidates, compose the cost of each run (@sec-hpt-composition). For larger searches, use random search with a random run count (@sec-hpt-rdp). Include runs that crashed or were used for debugging, and evaluations used for early stopping.

Log the candidate set, sampled run count, and each run's settings and score. Report both the released run's $\varepsilon$, which assumes public hyperparameters, and the whole procedure's $\varepsilon$, which is the release guarantee. An [audit](10-auditing.md) tests only the released model and cannot verify the search cost. At $\delta=10^{-5}$, the book's rule for 30,000 records ([How small δ must be](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta)), the Adult search in @sec-hpt-run reports:

> DP-SGD ran for 300 steps with Poisson sampling at rate 1/30. It clipped gradients to norm 1 and added noise with multiplier 1.256, so one run satisfies $(2.73, 10^{-5})$-DP. The learning rate was chosen by random search over $\{0.05, 0.1, 0.2, 0.5, 1, 2\}$. The number of runs was drawn from a Poisson distribution with mean 4, and each run was scored on 2,000 validation records treated as public. The whole procedure satisfies $(4.00, 10^{-5})$-DP for add/remove neighbours, datasets that differ by one record. It was computed with `dp_accounting` from the Rényi bound of Papernot & Steinke (2022).

## Paying for every run {#sec-hpt-composition}

Choosing among eight learning rates does not limit leakage to three bits. The selected model depends on all run outcomes. Even if each run is labelled only "good" or "bad", a bad selected run implies that all candidates were bad, revealing information about every run. Tuning without privacy and then training once privately can also leak: outliers may change the selected regularization weight, revealing their presence ([Membership inference](../part02-attacks/02-membership-inference.md)).

The safe approach charges for every run because selection uses all run outputs ([Composition](../part03-dp-foundations/04-composition.md)). For DP-SGD, $k$ runs of $T$ steps cost the same as one run of $kT$ steps. One run of [Part 4's example](../part04-privacy-accounting/00-introduction.md)—10,000 steps, sampling rate 0.01 and noise multiplier 1—costs $\varepsilon=6.71$ at $\delta=10^{-5}$. Ten runs cost 27.23, about four times the single-run cost rather than ten times, because conversion to $\varepsilon$ incurs the $\delta$ cost once.

## A random number of runs {#sec-hpt-rdp}

The way out is to draw the number of runs $K$ at random, independently of the data. A search whose runs all came out bad then most likely drew few runs, which says little about the data. So the whole search costs about one run, plus a term that grows with the logarithm of the expected number of runs. For DP-SGD, use the version of Papernot & Steinke (2022), which works with the Rényi differential privacy (RDP) that DP-SGD's accountants already compute ([Rényi DP](../part04-privacy-accounting/01-renyi-dp.md)).

To use it, fix the candidate set and the distribution of $K$, either Poisson or a truncated negative binomial with a shape parameter $\nu$. That family includes the geometric distribution, $\nu = 1$, which is the **random stopping** of Liu & Talwar, and the logarithmic one, $\nu = 0$. Then draw $K$, run that many candidates drawn at random from the set, and release the best one by a private score. To get the cost from Google's `dp_accounting`, wrap the event of one run in a `RepeatAndSelectDpEvent`. Give it $\nu$ as its `shape` (infinite for Poisson), and compose it in an `RdpAccountant`.

NumPy draws $K$ with `rng.poisson(mean)`, `rng.geometric(1 / mean)` or `rng.logseries(p)`. The last takes $p$ rather than a mean, so solve its mean, $-p/((1 - p)\ln(1 - p))$, for $p$.

| Expected runs | 10 | 100 | 1,000 |
| --- | --- | --- | --- |
| Composition | 27.23 | 145.17 | 1035.93 |
| Poisson | 13.67 | 67.77 | 302.57 |
| Geometric ($\nu = 1$) | 11.31 | 14.32 | 16.76 |
| Logarithmic ($\nu = 0$) | 10.08 | 11.95 | 13.46 |

: The $\varepsilon$ at $\delta = 10^{-5}$ of a search over runs of the running example (one run: 6.71), from the full edition's chapter. {#tbl-hpt-overhead}

In @tbl-hpt-overhead, the geometric and logarithmic searches barely grow from 10 to 1,000 runs, while composition grows almost linearly. A smaller $\nu$ gives a cheaper bound, but the search also stops after one run more often. The Poisson bound needs runs that are already very private, which a run at $\varepsilon = 6.71$ is not, so here it grows almost like composition. A random number of runs may also miss the best candidate. With eight candidates and a geometric number of runs with mean eight, it never tries the best one with probability 0.4667.

The theorems have three conditions, and each is easy to break. First, each run must draw its candidate independently from a fixed distribution, as random search does. Bayesian optimization picks each candidate from earlier scores, so it is not covered and needs [adaptive composition](../part04-privacy-accounting/08-adaptive-composition.md). Second, the number of runs must be drawn independently of the data. Third, the score that ranks the runs must be private and counted in each run's cost.

## Selecting by a noisy validation score {#sec-hpt-select}

Sometimes the candidates are fixed in advance, such as public pretrained checkpoints, and only the choice among them uses private data. That is a **selection** problem, which [Selection](../part03-dp-foundations/05-selection.md) solves with the exponential mechanism, a random choice weighted toward high scores. Score each candidate by its number of correct predictions, which one record changes by at most 1, not by an accuracy, which divides by a private set size. The mechanism's shortfall from the best count then grows only with the logarithm of the number of candidates.

For the random-number bounds, add Gaussian noise to each run's count of correct predictions, since an exact validation accuracy leaves the validation records unprotected. With noise of standard deviation 20, one noisy count costs $\varepsilon = 0.182$. If the validation set is disjoint from the training set, each record pays only one of the two costs. So the event for one run is the costlier of its training and its count, not their sum. Otherwise compose both into one event. On the running example the counts are the cheaper, so they add nothing to the total.

## Tuning on Adult {#sec-hpt-run}

The script [practitioner_tuning.py](../../code/part05-dp-ml/practitioner_tuning.py) tunes the learning rate of a small network on 30,000 Adult census records, with 300 steps per run. It asks the accountant for the noise multiplier per run that each strategy can afford.

| Strategy | Noise multiplier per run | Total $\varepsilon$ |
| --- | --- | --- |
| One run, no tuning | 1.029 | 4.00 |
| Grid of 6 at the noise of one run, one run's $\varepsilon$ reported | 1.029 | 9.84 (reported as 4.00) |
| Grid of 6, every run counted | 1.816 | 4.00 |
| Random search, Poisson number of runs with mean 4 | 1.256 | 4.00 |

: Tuning the learning rate on Adult at a total of $\varepsilon = 4$, $\delta = 10^{-5}$, with `dp_accounting`'s RDP accountant. {#tbl-hpt-run}

The second row is the reporting bug of @sec-hpt-composition. This snippet recomputes it, together with the numbers of the report in @sec-hpt-practice.

```python
import dp_accounting


def one_run(sigma: float) -> dp_accounting.DpEvent:
    step = dp_accounting.PoissonSampledDpEvent(1 / 30, dp_accounting.GaussianDpEvent(sigma))
    return dp_accounting.SelfComposedDpEvent(step, 300)


def epsilon(event: dp_accounting.DpEvent) -> float:
    return dp_accounting.rdp.RdpAccountant().compose(event).get_epsilon(1e-5)


search = dp_accounting.dp_event.RepeatAndSelectDpEvent(one_run(1.256), mean=4, shape=float("inf"))
naive_grid = dp_accounting.SelfComposedDpEvent(one_run(1.029), 6)
print(f"{epsilon(one_run(1.256)):.2f}")  # 2.73
print(f"{epsilon(search):.2f}")  # 4.00
print(f"{epsilon(naive_grid):.2f}")  # 9.84
```

With seed 0, the search drew only two runs, both at the smallest learning rate. So its model reaches only 0.828 on the test set, against 0.846 for the run of [Train a model with DP-SGD](13-train-dp-sgd.md), tuned for free on public stand-in rows.

## Cheat sheet

| Situation | Strategy | Cost |
| --- | --- | --- |
| Two or three candidates | Fixed grid, composition | $k$ runs of $T$ steps = one run of $kT$ steps |
| Many candidates | Random search; compute Poisson, geometric and logarithmic for your per-run noise and mean, and take the cheapest | Poisson wins only for very private runs; at one run of 6.71, logarithmic is cheapest at every size in @tbl-hpt-overhead, but it stops after one run more often |
| Fixed candidates | Exponential mechanism on correct counts | Shortfall grows with the log of the candidate count |
| Trained candidates, disjoint validation set | Exponential mechanism or noisy counts | The larger of training and selection costs |

: Which tuning strategy for which search. {#tbl-hpt-cheat}

**Rules of thumb**

- Tune on public or proxy data when you can. Otherwise fix the search before touching the private data, and never extend it.
- Report the whole search's $\varepsilon$, never the selected run's alone.

## References

- Papernot & Steinke, "Hyperparameter Tuning with Renyi Differential Privacy" (ICLR 2022): the Poisson and truncated negative binomial bounds for DP-SGD.
- Liu & Talwar, "Private Selection from Private Candidates" (STOC 2019): random stopping and its bound of three times one run.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/03-hyperparameter-tuning.md).
