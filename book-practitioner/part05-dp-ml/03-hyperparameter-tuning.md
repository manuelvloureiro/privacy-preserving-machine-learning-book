# Hyperparameter tuning and private selection

## Why this matters

Stochastic gradient descent (SGD) with clipped gradients and Gaussian noise satisfies differential privacy (DP) at the level its accountant reports for one run, and is called [DP-SGD](02-dp-sgd.md). But nobody trains once. You try a grid of step sizes, score each model on a validation set and ship the best, and every run and every score touched private data. So the search leaks, and you must pay for it honestly.

The answer depends on one choice. With a fixed number of runs, releasing the best can cost as much as releasing all of them. For example, one run of this part's running example costs a privacy level $\varepsilon$ of 6.71, but a grid of 10 runs costs 27.23. Now draw the *number* of runs at random, from a distribution fixed in advance. The same expected number of runs then costs between 10.08 and 13.67.

## The procedure and the report {#sec-hpt-practice}

First, spend private runs only where you must, because the cheapest tuning is tuning you don't do on private data. Runs on public or proxy data cost nothing, and so do published DP-SGD settings for similar tasks. Fix a large batch, the number of epochs and a small clipping norm in advance. Below the typical gradient norm only the clipping norm's product with the step size matters, so that product is often the only thing left to search ([DP-SGD](02-dp-sgd.md#sec-dpsgd-clipping)). Use tuners such as Optuna or Ray Tune on private data only for random search, with the number of runs drawn first.

Second, fix the search before you touch the private data. Write down the candidate set, the distribution of the number of runs, the score and the validation set. Deciding to "try a few more", or restarting a search whose result disappoints, makes the search depend on the data, and no theorem covers that.

Third, make the score private, with a disjoint validation set and noise on each count of correct predictions (@sec-hpt-select). Fourth, account for the whole search. For two or three candidates, charge for every run by composition (@sec-hpt-composition). For more, use random search with a random number of runs (@sec-hpt-rdp). Count every run that touched the private data, including crashed runs, debugging runs and evaluations used for early stopping.

Log the candidate set, the drawn number of runs, and each run's hyperparameters, accounting settings and noisy score. Opacus's `get_epsilon` reports one run only, so compose the runs in an accountant yourself. If you trained 50 configurations at $\varepsilon = 2$ each, the honest answer is not 2 but the composition of all 50, which grows almost linearly at large $\varepsilon$ (@tbl-hpt-overhead).

Finally, report two numbers. The $\varepsilon$ of the released run says what the model alone would leak, while the $\varepsilon$ of the whole procedure is what the release actually guarantees. For the Poisson search of @sec-hpt-run, a complete report reads as follows.

> DP-SGD with Poisson sampling rate 0.01, clipping norm 0.5, noise multiplier 1.53 and 1,000 steps. One run satisfies $(0.98, 10^{-5})$-DP. The step size was selected by random search over $\{0.0625, 0.125, \dots, 8\}$, with a Poisson number of runs with mean 8. Each run was scored by its correct predictions on a disjoint validation set of 5,000 records, plus Gaussian noise of standard deviation 20. The whole procedure satisfies $(1.99, 10^{-5})$-DP by the Rényi bound of Papernot & Steinke (2022, Theorem 6), for add/remove neighbours at the record level.

An [audit](10-auditing.md) tests only the released model, so it cannot catch a missing tuning charge.

## The best of k runs can cost k runs {#sec-hpt-leak}

A common argument says that choosing among eight step sizes reveals at most three bits. It is wrong, because the release is a whole private run, picked because of how it came out. If each run is a randomized response that comes out "good" or "bad", the best of $k$ runs is bad only if every run is bad. The evidence from that event multiplies across runs, so the best of $k$ runs is exactly $k\varepsilon$-DP.

Tuning without privacy and then training once privately does not escape this. Even a single chosen regularization weight can reveal whether a few outliers were in the training set ([membership inference](../part02-attacks/02-membership-inference.md)).

## Paying by composition {#sec-hpt-composition}

The safe answer is to charge for every run. The best run is computed from the runs' outputs alone, so it costs at most what releasing all of them costs ([Properties of DP](../part03-dp-foundations/02-properties.md)). Rényi differential privacy (RDP) describes each run by a curve $\varepsilon(\lambda)$ over orders $\lambda$ ([Rényi DP](../part04-privacy-accounting/01-renyi-dp.md)). The curves of $k$ runs add, so the whole search has a curve $\varepsilon_A$ of at most
$$\varepsilon_A(\lambda) \le k\,\varepsilon(\lambda).$$ {#eq-hpt-compose}
For DP-SGD this is simple, because $k$ runs of a number of steps $T$ cost exactly as much as one run of $kT$ steps.

In the running example, ten runs cost 4.06 times one run, because the conversion to $\varepsilon$ with a small slack $\delta$ pays its fixed cost only once. To keep ten runs at the budget of one, each run needs a noise multiplier of 2.45, which costs accuracy.

## A random number of runs {#sec-hpt-rdp}

The way past the $k\varepsilon$ barrier is to draw the number of runs at random, independently of the data. Then "every run was bad" is mostly explained by the search stopping early. Liu & Talwar (2019) showed this first for **random stopping**, which stops after each run with a fixed probability and costs $3\varepsilon$ for $\varepsilon$-DP runs.

For DP-SGD, use the version of Papernot & Steinke (2022), which works with RDP curves. First fix the candidate set and a distribution for the number of runs $K$. It is either Poisson or truncated negative binomial, a family with a shape parameter $\eta$. Then draw $K$ and run that many candidates drawn at random from the set. Release the best by a noisy score. The whole search then costs about one run, plus a term that grows only with the logarithm of the expected number of runs,
$$\varepsilon_A(\lambda) \le \varepsilon(\lambda) + \frac{\ln\mathbb E[K]}{\lambda - 1} + (\text{a term set by the shape of } K).$$
For pure $\varepsilon$-DP runs it tends to $(2 + \eta)\varepsilon$, so the logarithmic distribution ($\eta = 0$) gives $2\varepsilon$ and the geometric ($\eta = 1$) gives $3\varepsilon$. Google's `dp_accounting` computes it, by composing in an `RdpAccountant` a `RepeatAndSelectDpEvent(event, mean, shape)` that wraps one run's event. Its shape is infinite for Poisson, 1 for geometric and 0 for logarithmic.

```text
import dp_accounting
run = dp_accounting.SelfComposedDpEvent(
    dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(sigma)), steps)
search = dp_accounting.dp_event.RepeatAndSelectDpEvent(run, mean=8, shape=float("inf"))
acc = dp_accounting.rdp.RdpAccountant()
acc.compose(search)
print(acc.get_epsilon(1e-5))
```

@tbl-hpt-overhead evaluates the bounds on the running example.

| Expected runs | 10 | 100 | 1,000 |
| --- | --- | --- | --- |
| Composition | 27.23 | 145.17 | 1035.93 |
| Poisson | 13.67 | 67.77 | 302.57 |
| Geometric ($\eta = 1$) | 11.31 | 14.32 | 16.76 |
| Logarithmic ($\eta = 0$) | 10.08 | 11.95 | 13.46 |

: The $\varepsilon$ at $\delta = 10^{-5}$ of tuning with DP-SGD runs of the running example (one run: $\varepsilon = 6.71$), by the number of runs and how it is chosen. {#tbl-hpt-overhead}

A smaller $\eta$ gives a cheaper bound but more often stops after a single run, and the Poisson bound suits runs that are already very private. A random number of runs may also miss the best candidate. On a grid of 8 with 8 runs on average, it never tries the best one with probability 0.4667.

The theorems have three conditions, and each is easy to break. First, each run must draw its candidate independently from a fixed distribution, which is random search. So Bayesian optimization, which picks each candidate from earlier scores, is not covered and needs [adaptive composition](../part04-privacy-accounting/08-adaptive-composition.md). Second, the number of runs must be drawn independently of the data. Third, the score that ranks the runs must be part of each run's private output.

## Selecting by a noisy validation score {#sec-hpt-select}

When the candidates are fixed, such as public pretrained checkpoints, only the choice among them uses private data. That is a pure **selection** problem, which [Selection](../part03-dp-foundations/05-selection.md) solved. Score each candidate by its number of correct predictions on a private validation set. One record changes such a count by at most 1, while an accuracy divides by a size that is itself private. Then the **exponential mechanism** at a budget $\varepsilon$ picks a candidate whose expected shortfall from the best count is at most
$$\frac{2}{\varepsilon}(\ln k + 1)$$
correct predictions, among $k$ candidates. On a validation set of $m$ records, that is an accuracy gap of at most $2(\ln k + 1)/(\varepsilon m)$. For example, choosing among eight trained models on 5,000 validation records costs about a tenth of a point of accuracy at a budget of 0.1.

With disjoint training and validation sets, each record sits in one of them, so a search over trained candidates costs the larger of the training cost and the selection cost, not their sum.

The random-number bounds need a private score too. An exact validation accuracy leaves the validation records unprotected, so add Gaussian noise to each run's count of correct predictions. With noise of standard deviation 20, one noisy count costs 0.182. Eight of them stay below the training runs' cost at every order the conversion uses, so they add nothing to the total.

## Tuning the running example {#sec-hpt-run}

@tbl-hpt-run tunes the running example's step size over eight values at a total budget of $\varepsilon = 2$. Each strategy picks the noise multiplier that keeps it within budget.

| Strategy | $\sigma$ | $\varepsilon$ of one run | Total $\varepsilon$ | Test accuracy |
| --- | --- | --- | --- | --- |
| No tuning: step size 0.5 from [DP-SGD](02-dp-sgd.md) | 1.03 | 1.97 | 1.97 | $0.9509 \pm 0.0015$ |
| Grid of 8, single-run $\varepsilon$ reported | 1.03 | 1.97 | 5.60 | $0.9479 \pm 0.0030$ |
| Tune on proxy data, then one run | 1.03 | 1.97 | 1.97 | $0.9487 \pm 0.0025$ |
| Grid of 8, composition | 2.07 | 0.66 | 1.99 | $0.9430 \pm 0.0022$ |
| Random search, Poisson, mean 8 | 1.53 | 0.98 | 1.99 | $0.9433 \pm 0.0060$ |
| Random search, logarithmic, mean 8 | 1.36 | 1.17 | 2.00 | $0.9372 \pm 0.0200$ |

: Tuning the step size of the running example at a total budget of $\varepsilon = 2$, $\delta = 10^{-5}$, with mean and standard deviation of the released model's test accuracy over repetitions of the whole procedure. {#tbl-hpt-run}

The second row is the common reporting bug. A team runs the grid at the noise of a single run and reports that run's 1.97, when the honest cost is 5.60. It then looks better than an honest search of the same grid.

The honest searches lose about a point against one untuned run. Poisson random search beats composition only slightly, because with one run per step size on average it often misses one, and the logarithmic distribution often stops after one run. Tuning on proxy data, a related task labelled by a different random network, is nearly free.

## Cheat sheet

| Situation | Strategy | Cost |
| --- | --- | --- |
| Good hyperparameters from public or proxy data | One private run | One run |
| Two or three candidates | Fixed grid, composition | $k$ runs of $T$ steps = one run of $kT$ steps |
| Many candidates, small per-run budget | Random search, Poisson number of runs | One run plus a term in the log of the mean |
| Many candidates, larger per-run budget | Random search, logarithmic or $\eta = 0.5$ | Near $(2 + \eta)$ runs for pure DP; may try few |
| Fixed candidates, private validation set | Exponential mechanism on correct-prediction counts | Gap at most $2(\ln k + 1)/\varepsilon$ predictions |
| Trained candidates, disjoint validation set | Noisy counts | The larger of training and selection costs |

: Which tuning strategy for which search. {#tbl-hpt-cheat}

**Rules of thumb**

- Fix the search before touching the private data, and never "try a few more" or restart it.
- Report the whole search's $\varepsilon$, never the selected run's alone.
- Account for Bayesian optimization as adaptive composition, not as a fixed candidate set.
- Never select by exact accuracy on a private validation set.

## Further reading

- Papernot & Steinke, "Hyperparameter Tuning with Renyi Differential Privacy" (ICLR 2022): the Poisson and truncated negative binomial bounds for DP-SGD.
- Liu & Talwar, "Private Selection from Private Candidates" (STOC 2019): random stopping and the $3\varepsilon$ bound.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/03-hyperparameter-tuning.md).
