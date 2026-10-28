# Hyperparameter tuning and private selection

## Why this matters

Stochastic gradient descent (SGD) with clipped gradients and Gaussian noise satisfies differential privacy (DP) at the level its accountant reports for one run. It is called [DP-SGD](02-dp-sgd.md). But nobody trains once. You try several learning rates, score each model and ship the best, and every run and every score touched private data. So the privacy level $\varepsilon$ you report must pay for the search too.

Releasing only the best of a fixed number of runs can cost as much as releasing all of them. On the Adult census data, a grid of six runs that each meet $\varepsilon = 4$ really costs 9.84. If instead you draw the *number* of runs at random, from a distribution fixed in advance, the whole search costs little more than one run.

## The procedure and the report {#sec-hpt-practice}

The cheapest tuning is the tuning you don't do on private data. Runs on public or proxy data cost nothing, and neither do published settings for similar tasks. Fix a large batch, the number of epochs and a small clipping norm in advance. With the clipping norm below the typical gradient norm, only its product with the learning rate matters, so often only that product needs a search ([DP-SGD](02-dp-sgd.md#sec-dpsgd-clipping)).

If you must tune on private data, fix the search before you touch the data. Write down the candidate set, the distribution of the number of runs (@sec-hpt-rdp), the score, which must be private too (@sec-hpt-select), and the validation set. Trying "a few more" later, or restarting a disappointing search, makes the search depend on the data, which no accountant covers.

Then account for the whole search. For two or three candidates, charge for every run (@sec-hpt-composition), and for more, use random search with a random number of runs (@sec-hpt-rdp). Either way, count every run that touched the private data, including crashed and debugging runs and evaluations for early stopping.

Finally, log the candidate set, the drawn number of runs and each run's settings and score, and report two numbers. The $\varepsilon$ of the released run says what the model alone would leak if its hyperparameters had been public. The $\varepsilon$ of the whole procedure is what the release guarantees, and an [audit](10-auditing.md) cannot check it, because an audit tests only the released model. With the slack $\delta = 10^{-5}$ that the book's rule gives for 30,000 records ([How small δ must be](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta)), the Adult search of @sec-hpt-run reports as follows.

> DP-SGD ran for 300 steps with Poisson sampling at rate 1/30. It clipped gradients to norm 1 and added noise with multiplier 1.256, so one run satisfies $(2.73, 10^{-5})$-DP. The learning rate was chosen by random search over $\{0.05, 0.1, 0.2, 0.5, 1, 2\}$. The number of runs was drawn from a Poisson distribution with mean 4, and each run was scored on 2,000 validation records treated as public. The whole procedure satisfies $(4.00, 10^{-5})$-DP for add/remove neighbours, datasets that differ by one record. It was computed with `dp_accounting` from the Rényi bound of Papernot & Steinke (2022).

## Paying for every run {#sec-hpt-composition}

A common argument says that choosing among eight learning rates reveals at most three bits. It is wrong, because the release is a whole private run, picked because of how it came out. If each run says only "good" or "bad", the best is bad only when every run was bad. So a bad best reveals all of them. Even tuning without privacy and then training once privately leaks. The winning regularization weight can change when you add a few outliers, and so reveal whether they were in the data ([Membership inference](../part02-attacks/02-membership-inference.md)).

The safe answer is to charge for every run, because the best run is computed from all the runs' outputs ([Composition](../part03-dp-foundations/04-composition.md)). For DP-SGD, a grid of $k$ runs, each of $T$ steps, costs exactly as much as one run of $kT$ steps. One run of [Part 4's running example](../part04-privacy-accounting/00-introduction.md), 10,000 DP-SGD steps that each sample a record with probability 0.01 at noise multiplier 1, costs $\varepsilon = 6.71$ at $\delta = 10^{-5}$. Ten such runs cost 27.23, about four times one run rather than ten, because the conversion to $\varepsilon$ pays its $\delta$ cost only once.

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
