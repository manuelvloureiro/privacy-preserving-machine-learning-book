# Algorithmic stability and generalization

## Why this matters

Your model's training loss is low. **Will its loss on new data be low too, and which training examples does it depend on?** The difference between test loss and training loss is the **generalization gap**. The classical answer counts how rich the model class is; this chapter looks at the training algorithm instead.

The one idea is that **a model generalizes when it does not depend much on any single training example.** The expected gap is exactly the average effect of swapping one example out; an algorithm that limits that effect limits the gap; and you can turn the two knobs that control it: regularization strength, and how long you train with stochastic gradient descent (SGD). Later parts measure how much a trained model depends on a single example, and they use the quantities defined here. @tbl-stability-overview maps the results to the sections.

| Question | Result | Section |
| --- | --- | --- |
| What is the gap, really? | The average replace-one effect; at most the stability $\beta$ | @sec-stability-gap |
| How sure is the bound for the one model I trained? | A high-probability bound, useful when $\beta$ falls faster than $1/\sqrt n$ | @sec-stability-gap |
| How stable is regularized training? | $\beta = 2L^2/(\lambda n)$ | @sec-stability-erm |
| How stable is SGD? | $\beta \le (2L^2/n)\sum_t\eta_t$: stop early | @sec-stability-sgd |
| Which examples carry the gap? | Atypical ones, measured by self-influence | @sec-stability-memorization |

: The questions of this chapter, and where each is answered. {#tbl-stability-overview}

**Notation.** The training set is $S = (z_1, \dots, z_n)$, drawn independently from a distribution $\mathcal D$, and $S^{(i)}$ is $S$ with example $i$ replaced by a fresh draw $z_i'$. The algorithm $A$ returns the model $A(S)$, with test loss $R(A(S))$ and training loss $\hat R_S(A(S))$.

## The gap and uniform stability {#sec-stability-gap}

**Idea.** The test loss can be measured on a fresh point $z_i'$. Swap the names of $z_i$ and $z_i'$, which have the same distribution, and the same number becomes the loss on $z_i$ of a model trained without it. So the gap is how much worse each example does when it is left out.

**Result.** For any algorithm,
$$\mathbb E[\mathrm{gap}] = \frac1n\sum_{i=1}^n\mathbb E\Big[\ell\big(A(S^{(i)}), z_i\big) - \ell\big(A(S), z_i\big)\Big].$$
$A$ is **$\beta$-uniformly stable** if replacing any one example moves the loss of the trained model by at most $\beta$ at every test point, for every training set and replacement (in expectation over $A$'s own randomness, for SGD). Then $|\mathbb E[\mathrm{gap}]| \le \beta$. For the one model you trained, with losses in $[0, M]$, McDiarmid's inequality ([Concentration inequalities](02-concentration.md)) gives, with probability at least $1 - \delta$,
$$R(A(S)) \le \hat R_S(A(S)) + \beta + (2n\beta + M)\sqrt{\frac{\ln(1/\delta)}{2n}}.$$
The term $2n\beta\sqrt{\ln(1/\delta)/(2n)}$ vanishes only if $\beta$ falls faster than $1/\sqrt n$.

**Example.** The running example is logistic regression on features uniform on the unit sphere in 10 dimensions, regularized with $\lambda = 0.01$. With $n = 50$, averaged over 100 training sets, the training loss is 0.4603 and the test loss 0.5781, an expected gap of about 0.119. With $n = 1000$ the gap is about 0.006. At $n = 10{,}000$ and $\delta = 0.05$, the high-probability bound adds 0.065 to the training loss at $\lambda = 1$, where the loss is at most $M = \ln(1 + e) = 1.313$ because the weights satisfy $\|w\| \le 1/\lambda$, and 0.614 at $\lambda = 0.1$.

**Pitfalls.** The identity is exact; the bounds are worst cases, and the high-probability one is loose because it charges every example the full $2\beta$. Sharper bounds (Feldman & Vondrák; Bousquet, Klochkov & Zhivotovskiy) need only $\beta \to 0$. Zero training error is a warning, not a result: a lookup table that memorizes its training set has $\beta = 1$, the largest value for a loss in $[0, 1]$, and a gap that never shrinks.

## Regularized ERM is stable {#sec-stability-erm}

**Idea.** The penalty $\frac\lambda2\|w\|^2$ makes the training objective a bowl with curvature at least $\lambda$, and one example, with weight $1/n$, can only tilt such a bowl so far.

**Result.** For regularized **empirical risk minimization (ERM)** with a convex, $L$-Lipschitz loss, replacing one example moves the minimizer by at most $2L/(\lambda n)$ and the loss at any point by at most
$$\beta = \frac{2L^2}{\lambda n}.$$
Doubling $\lambda$ halves $\beta$, and $\beta$ falls like $1/n$, fast enough for both bounds above. The weights also satisfy $\|w\| \le L/\lambda$. This is the movement bound of [Optimization](06-optimization.md).

**Example.** @tbl-stability-replace-one flips one label in logistic regression with unit-norm features ($L = 1$) and records the largest changes.

| $n$ | $\lambda$ | Bound $2L^2/(\lambda n)$ | Largest loss change | Ratio |
| --- | --- | --- | --- | --- |
| 100 | 1 | 0.02 | 0.00505 | 0.25 |
| 1,000 | 0.1 | 0.02 | 0.00499 | 0.25 |
| 10,000 | 0.01 | 0.02 | 0.00285 | 0.14 |

: Replace-one loss changes of regularized logistic regression against the stability bound. {#tbl-stability-replace-one}

The measured expected gap is a fixed fraction of the bound: about 3% at $\lambda = 0.01$ and 11% to 13% at $\lambda = 1$.

**Pitfalls.** The bound needs a Lipschitz loss: squared loss qualifies only on a bounded set of weights and labels. It ignores the data's own curvature, so with weak regularization it overestimates more (ratio 0.14 against 0.25). Stability is not accuracy: too large a $\lambda$ closes the gap by underfitting, and the test loss is U-shaped in $\lambda$.

## SGD is stable if you stop early {#sec-stability-sgd}

**Idea.** Run SGD on $S$ and on $S^{(i)}$ with the same random order. The two runs only drift apart on the steps that pick example $i$, about once per epoch. On every other step they take a gradient step on the same convex smooth loss, which does not push two points apart as long as the step size $\eta \le 2/\rho$, with $\rho$ the smoothness.

**Result.** For a convex, $\rho$-smooth, $L$-Lipschitz loss and steps $\eta_t \le 2/\rho$, SGD is uniformly stable with
$$\beta \le \frac{2L^2}{n}\sum_{t=1}^T\eta_t$$
(Hardt, Recht & Singer 2016). With a constant step $\eta$ for $E$ epochs this is $2L^2\eta E$: it grows with training time and does not improve with more data at a fixed number of epochs.

**Example.** Unregularized logistic regression with $n = 100$ and $\eta = 0.25$ has bound $0.5E$. The measured gap grows from 0.036 after 1 epoch to 0.121 after 40, and the test loss is lowest after about 5 epochs while the training loss keeps falling (@fig-stability-sgd-epochs).

![Left: train and test loss of SGD on logistic regression ($n = 100$, $\eta = 0.25$) against the number of epochs. Right: the measured gap against the stability bound $2L^2\eta T/n$, on a log scale. Both grow with training time; the bound grows faster.](figures/stability_sgd_epochs.svg){#fig-stability-sgd-epochs}

**Pitfalls.** Training longer lowers the optimization error but raises $\beta$, so early stopping is a trade-off, not a trick. The bound is loose, by a factor of 14 after one epoch and 165 after 40, because on a curved loss the two runs contract between jumps instead of drifting further apart. For non-convex losses a gradient step can push points apart, and the bound grows much faster with $T$.

## Memorization and self-influence {#sec-stability-memorization}

**Idea.** The bracket in the gap identity can be measured example by example. It shows which examples the model fits only because it saw them.

**Result.** The **self-influence**, or **memorization score**, of example $i$ is
$$\mathrm{mem}_i(S) = \ell\big(A(S^{(i)}), z_i\big) - \ell\big(A(S), z_i\big),$$
its loss under a model trained without it minus its loss under a model trained with it. Its average is the expected gap. Typical examples are fitted whether or not they are present; atypical and mislabelled ones are fitted only when present, so they carry most of the gap.

**Example.** With $n = 40$ and $\lambda = 0.01$, the measured gap is $0.1482 \pm 0.0073$ and the average self-influence $0.1422 \pm 0.0025$, equal within error. Examples whose label goes against the trend have average self-influence 0.284, against 0.097 for the rest (@fig-stability-self-influence). The 1-nearest-neighbour classifier, which predicts the label of the closest training point, is the extreme: training error 0, $\beta = 1$, and a test error that falls only to 0.352 at $n = 4{,}000$, towards its limit 0.338.

![Self-influence of each training example against $y_ix_{i1}$, which measures how typical its label is, for $n = 40$ and two values of $\lambda$. Examples whose label goes against the trend have the largest self-influence; regularization flattens the profile. The dashed line is the average, which equals the expected gap.](figures/stability_self_influence.svg){#fig-stability-self-influence}

**Pitfalls.** A small average gap allows a few heavily memorized examples; uniform stability does not. Memorization is not always a defect: Feldman (2020) shows that on data with a long tail of rare subpopulations, fitting the singletons is needed for accuracy. So expect the rarest examples to be the most memorized, and measure them rather than the average.

## Code

There is no snippet: every number here comes from retraining models. The full edition's script, [stability.py](../../code/part01-prerequisites/stability.py), runs all the experiments:

```bash
uv run python code/part01-prerequisites/stability.py
```

## Cheat sheet

| Result | Assumptions | Statement |
| --- | --- | --- |
| Gap identity | None | $\mathbb E[\mathrm{gap}] = \frac1n\sum_i\mathbb E[\mathrm{mem}_i]$ |
| Expected gap | $\beta$-uniformly stable | $\lvert\mathbb E[\mathrm{gap}]\rvert \le \beta$ |
| High probability | Also $0 \le \ell \le M$ | $\mathrm{gap} \le \beta + (2n\beta + M)\sqrt{\ln(1/\delta)/(2n)}$ |
| Regularized ERM | Convex, $L$-Lipschitz loss, $\frac\lambda2\lVert w\rVert^2$ | $\beta = 2L^2/(\lambda n)$ |
| SGD | Convex, $\rho$-smooth, $L$-Lipschitz, $\eta_t \le 2/\rho$ | $\beta \le (2L^2/n)\sum_t\eta_t$ |
| Lookup table, 1-nearest neighbour | 0-1 loss | $\beta = 1$; training error 0 |

: The results of this chapter. {#tbl-stability-cheat}

**Rules of thumb**

- Regularization strength and training time are the two stability knobs: $\beta \propto 1/\lambda$ for ERM, $\beta \propto \eta T$ for SGD.
- A useful high-probability bound needs $\beta$ to fall faster than $1/\sqrt n$.
- Training error 0 means the model may depend entirely on individual examples.
- The examples a model memorizes most are the atypical ones.

## Further reading

- Bousquet & Elisseeff, "Stability and Generalization" (JMLR 2002): uniform stability and the high-probability bound.
- Shalev-Shwartz & Ben-David, *Understanding Machine Learning* (2014), ch. 13: the gap identity and the stability of regularized ERM.
- Hardt, Recht & Singer, "Train Faster, Generalize Better: Stability of Stochastic Gradient Descent" (ICML 2016): SGD stability and early stopping.
- Feldman & Zhang, "What Neural Networks Memorize and Why" (NeurIPS 2020): memorization scores measured on image benchmarks.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/07-stability-generalization.md).
