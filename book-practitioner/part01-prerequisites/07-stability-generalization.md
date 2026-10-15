# Algorithmic stability and generalization

## Why this matters

Your model's training loss is low. Will its loss on new data be low too, and which training examples does it depend on? For a model fixed before you saw the data, the training loss is an unbiased estimate of the test loss. For a model fitted to the data it is biased downward, because training picked the model that looks good on exactly these examples. That bias, test loss minus training loss, is the **generalization gap**.

Instead of counting how rich the model class is, this chapter looks at the training algorithm, and its one idea is that **a model generalizes when it does not depend much on any single training example.** The gap is exactly the average effect of swapping one example out, and two knobs you already turn control that effect: regularization strength, and how long you train with stochastic gradient descent (SGD). Later parts reuse these quantities.

**Notation.** The training set is $S = (z_1, \dots, z_n)$, drawn independently from a distribution $\mathcal D$, and $S^{(i)}$ is $S$ with example $i$ replaced by a fresh draw $z_i'$. The algorithm $A$ returns the model $A(S)$, with test loss $R(A(S))$ and training loss $\hat R_S(A(S))$.

## The gap and uniform stability {#sec-stability-gap}

The test loss of $A(S)$ can be measured on the fresh point $z_i'$, which the model never saw. Swap the names of $z_i$ and $z_i'$. They have the same distribution, so nothing changes on average, but the same number now reads as the loss on $z_i$ of a model trained on $S^{(i)}$, which did not see $z_i$. The training loss is the loss on $z_i$ of the model that did. So for any algorithm,
$$\mathbb E[\mathrm{gap}] = \frac1n\sum_{i=1}^n\mathbb E\Big[\ell\big(A(S^{(i)}), z_i\big) - \ell\big(A(S), z_i\big)\Big]:$$
the expected gap is how much worse each example does when it is left out, averaged over examples.

To bound the gap, bound each bracket. $A$ is **$\beta$-uniformly stable** if replacing any one example moves the loss of the trained model by at most $\beta$ at every test point, for every training set and replacement (in expectation over $A$'s own randomness, for SGD). Then every bracket is at most $\beta$, and so is the expected gap. No model class appears, so the bound holds for large models too.

You train once, though, so you want the gap of your model, not its average over training sets. The gap is a function of $n$ independent examples, none of which can move it by much, and that is the situation of McDiarmid's inequality ([Concentration inequalities](02-concentration.md)). With losses in $[0, M]$, with probability at least $1 - \delta$,
$$R(A(S)) \le \hat R_S(A(S)) + \beta + (2n\beta + M)\sqrt{\frac{\ln(1/\delta)}{2n}}.$$
The $M$ part is the usual $1/\sqrt n$ fluctuation you would pay even for a fixed model. The $2n\beta$ part is the price of learning, and it vanishes only if $\beta$ falls faster than $1/\sqrt n$. It is loose, because it charges every example the full $2\beta$ although their effects largely cancel; sharper bounds (Bousquet, Klochkov & Zhivotovskiy) need only $\beta \to 0$.

The running example is logistic regression on 10-dimensional features uniform on the unit sphere, regularized with $\lambda = 0.01$. With $n = 50$ the training loss is 0.4603 and the test loss 0.5781, a gap of about 0.119; with $n = 1000$ the gap is about 0.006. At $n = 10{,}000$ and $\delta = 0.05$, the high-probability bound adds 0.065 to the training loss at $\lambda = 1$ but 0.614 at $\lambda = 0.1$, mostly through the $2n\beta$ term.

## Regularized ERM is stable {#sec-stability-erm}

Regularized **empirical risk minimization (ERM)** minimizes the training loss plus $\frac\lambda2\|w\|^2$. The penalty makes the objective a bowl with curvature at least $\lambda$, and one example, carrying weight $1/n$, can only tilt such a bowl so far. [Optimization](06-optimization.md) turns that into a number: with a convex, $L$-Lipschitz loss, replacing one example moves the minimizer by at most $2L/(\lambda n)$, so the loss at any point moves by at most $L$ times that,
$$\beta = \frac{2L^2}{\lambda n}.$$
Doubling $\lambda$ halves $\beta$, and $\beta$ falls like $1/n$, faster than $1/\sqrt n$, so both bounds above go to zero as the data grow.

@tbl-stability-replace-one flips one label in logistic regression with unit-norm features ($L = 1$), keeping $\lambda n$ fixed so the bound stays at 0.02.

| $n$ | $\lambda$ | Bound $2L^2/(\lambda n)$ | Largest loss change | Ratio |
| --- | --- | --- | --- | --- |
| 100 | 1 | 0.02 | 0.00505 | 0.25 |
| 1,000 | 0.1 | 0.02 | 0.00499 | 0.25 |
| 10,000 | 0.01 | 0.02 | 0.00285 | 0.14 |

: Replace-one loss changes of regularized logistic regression against the stability bound. {#tbl-stability-replace-one}

The slack comes from the worst cases in the argument: two examples' gradients rarely point in opposite directions at full length, and the data add curvature of their own, which the bound ignores. That second slack grows when the regularizer is weak, which is why the ratio drops to 0.14 at $\lambda = 0.01$. The measured expected gap is smaller still: about 3% of the bound at $\lambda = 0.01$, 11% to 13% at $\lambda = 1$.

The argument needs a Lipschitz loss, so squared loss qualifies only on a bounded set of weights and labels. And stability is not accuracy: a large $\lambda$ closes the gap by pulling the weights towards zero and underfitting, so the test loss is U-shaped in $\lambda$.

## SGD is stable if you stop early {#sec-stability-sgd}

Large models are trained by SGD for a number of steps, not to a minimizer. Run SGD on $S$ and on $S^{(i)}$ with the same random order. On steps that pick a shared example, both runs take a gradient step on the same convex, $\rho$-smooth loss, and such a step does not push two points apart as long as $\eta \le 2/\rho$. Only the steps that pick example $i$ can separate them, by at most $2L\eta_t$, and those come about once per epoch. Adding up the jumps gives (Hardt, Recht & Singer 2016)
$$\beta \le \frac{2L^2}{n}\sum_{t=1}^T\eta_t.$$
With a constant step $\eta$ for $E$ epochs this is $2L^2\eta E$. Stability degrades with training time, and the $n$ cancels: for a fixed number of epochs, more data does not help the bound.

So early stopping is a trade-off, not a trick: training longer lowers the optimization error but raises $\beta$. With unregularized logistic regression, $n = 100$ and $\eta = 0.25$, the measured gap grows from 0.036 after 1 epoch to 0.121 after 40, and the test loss is lowest after about 5 epochs while the training loss keeps falling (@fig-stability-sgd-epochs).

![Left: train and test loss of SGD on logistic regression ($n = 100$, $\eta = 0.25$) against the number of epochs. Right: the measured gap against the stability bound $2L^2\eta T/n$, on a log scale. Both grow with training time; the bound grows faster.](figures/stability_sgd_epochs.svg){#fig-stability-sgd-epochs}

The bound, $0.5E$ here, is loose by a factor of 14 after one epoch and 165 after 40, because the argument only uses that a step does not expand distances; on a curved loss it usually contracts them, so the runs drift apart much more slowly. On a non-convex loss a gradient step can push points apart, and the bound grows much faster with $T$.

## Memorization and self-influence {#sec-stability-memorization}

The bracket in the gap identity can be measured example by example, by retraining without each one: the **self-influence**, or **memorization score**, of example $i$,
$$\mathrm{mem}_i(S) = \ell\big(A(S^{(i)}), z_i\big) - \ell\big(A(S), z_i\big),$$
and by the identity its average is the expected gap. With $n = 40$ and $\lambda = 0.01$, the measured gap is $0.1482 \pm 0.0073$ and the average self-influence $0.1422 \pm 0.0025$, equal within error.

The scores are uneven, and the reason is redundancy. A typical example is fitted whether or not it is present, because its neighbours carry the same information. An atypical or mislabelled one is fitted only if it is there itself. So examples whose label goes against the trend have average self-influence 0.284, against 0.097 for the rest (@fig-stability-self-influence).

![Self-influence of each training example against $y_ix_{i1}$, which measures how typical its label is, for $n = 40$ and two values of $\lambda$. Examples whose label goes against the trend have the largest self-influence; regularization flattens the profile. The dashed line is the average, which equals the expected gap.](figures/stability_self_influence.svg){#fig-stability-self-influence}

The extreme case is the 1-nearest-neighbour classifier, which predicts the label of the closest training point. It fits every training example using that example itself, so its training error is 0 and its $\beta$ is 1, the largest possible for the 0-1 loss. Its test error falls only to 0.352 at $n = 4{,}000$, towards a limit of 0.338, while the stable logistic regression reaches 0.259 on the same data. Zero training error is a warning, not a result.

A small average gap still allows a few heavily memorized examples; only uniform stability caps each. And memorization is not always a defect: Feldman (2020) shows that on data with a long tail of rare subpopulations, fitting the singletons is needed for accuracy. So expect the rarest examples to be the most memorized, and measure them rather than the average.

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
- The examples a model memorizes most are the atypical ones.

## Further reading

- Bousquet & Elisseeff, "Stability and Generalization" (JMLR 2002): uniform stability and the high-probability bound.
- Shalev-Shwartz & Ben-David, *Understanding Machine Learning* (2014), ch. 13: the gap identity and the stability of regularized ERM.
- Hardt, Recht & Singer, "Train Faster, Generalize Better: Stability of Stochastic Gradient Descent" (ICML 2016): SGD stability and early stopping.
- Feldman & Zhang, "What Neural Networks Memorize and Why" (NeurIPS 2020): memorization scores measured on image benchmarks.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/07-stability-generalization.md).
