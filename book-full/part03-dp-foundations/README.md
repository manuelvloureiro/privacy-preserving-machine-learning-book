# Part 3 — Differential privacy foundations

The definition of differential privacy and the tools that make it usable: neighbouring relations, the hypothesis-testing and Bayesian readings, post-processing, group privacy and composition, the Laplace, Gaussian and exponential mechanisms, report-noisy-max and the sparse vector technique, query release with private multiplicative weights, data-dependent sensitivity, advanced and optimal composition, local, shuffle and distributed trust, lower bounds, exact discrete samplers, and private mean estimation. Rényi DP, numerical accounting and amplification come next, in [Part 4](../part04-privacy-accounting/README.md).

Start with the [introduction](00-introduction.md), which is part of the book. Read chapters 18 → 19 → 20 → 21 in order; the rest can be read in any order, though chapter 27 closes the part best.

This file is the GitHub overview. It is not part of the rendered book.

## Chapters

| Ch. | Chapter | Covers | Script |
|-----|---------|--------|--------|
| 18 | [The definition of differential privacy](01-definition.md) | Neighbouring relations; pure and approximate DP; randomized response; divergence, hypothesis-testing and Bayesian forms; membership advantage; why the definition must be multiplicative and randomized; why $\delta \ll 1/n$, and catastrophic against graceful $\delta$; what DP does not protect; the three-question test for a privacy violation | [definition.py](../../code/part03-dp-foundations/definition.py) |
| 19 | [Post-processing, group privacy and composition](02-properties.md) | Post-processing; group privacy for pure and approximate DP; basic and parallel composition; what does not compose | [properties.py](../../code/part03-dp-foundations/properties.py) |
| 20 | [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md) | Sensitivity; the Laplace mechanism and its privacy loss distribution; classic and analytic Gaussian calibration; Laplace against Gaussian in high dimension | [laplace_gaussian.py](../../code/part03-dp-foundations/laplace_gaussian.py) |
| 21 | [Advanced and optimal composition](04-composition.md) | The privacy loss random variable; advanced composition; the exact Kairouz–Oh–Viswanath formula; adaptive composition; Laplace noise over many dense counts | [composition.py](../../code/part03-dp-foundations/composition.py) |
| 22 | [Selection: exponential mechanism, report-noisy-max, sparse vector](05-selection.md) | The exponential mechanism and its utility; private learning and the private median; report-noisy-max; Gumbel noise; AboveThreshold, Sparse and the broken variants | [selection.py](../../code/part03-dp-foundations/selection.py) |
| 23 | [Query release and private multiplicative weights](06-query-release.md) | Noise on every answer against multiplicative weights; the regret bound and the $4\ln N/\alpha^2$ mistake bound; offline private MW (MWEM); online private MW via sparse vector; SmallDB | [query_release.py](../../code/part03-dp-foundations/query_release.py) |
| 24 | [Beyond global sensitivity](07-sensitivity.md) | Local and smooth sensitivity; propose-test-release; subsample-and-aggregate; inverse sensitivity; stability-based release; clipping | [sensitivity.py](../../code/part03-dp-foundations/sensitivity.py) |
| 25 | [Discrete and floating-point-safe mechanisms](08-discrete-mechanisms.md) | The floating-point attack; snapping; exact Bernoulli, discrete Laplace and discrete Gaussian samplers | [discrete_mechanisms.py](../../code/part03-dp-foundations/discrete_mechanisms.py) |
| 26 | [Local, shuffle and distributed trust models](09-trust-models.md) | Local randomizers; the $\sqrt n$ gap; frequency estimation; hashing and heavy hitters; one-bit means; repeated collection and memoization (RAPPOR, Microsoft); the shuffle model; distributed noise behind secure aggregation | [trust_models.py](../../code/part03-dp-foundations/trust_models.py) |
| 27 | [Lower bounds: what privacy costs](10-lower-bounds.md) | Packing, also under approximate DP; sample complexity; fingerprinting; tracing attacks; Dinur–Nissim as a lower bound | [lower_bounds.py](../../code/part03-dp-foundations/lower_bounds.py) |
| 28 | [Private mean estimation](11-mean-estimation.md) | Find the data privately, then clip there: bounded data; coarse histogram then fine estimate ($\ln R/\varepsilon$); shrinking balls in $d$ dimensions ($d/(\alpha\varepsilon)$); clipping heavy tails ($1/(\alpha^2\varepsilon)$) | [mean_estimation.py](../../code/part03-dp-foundations/mean_estimation.py) |

Each script prints the numbers its chapter quotes and regenerates the chapter's figures in `_build/figures/part03-dp-foundations/` (the build copies them into each book):

```bash
uv run python code/part03-dp-foundations/<script>.py
```

Every chapter's exercises, each with a worked solution, are in the exercise book, [book-exercises/part03-dp-foundations/](../../book-exercises/part03-dp-foundations/), one file per chapter. The exercise book is its own website, and `code/part03-dp-foundations/solutions_<script>.py` prints the numbers they quote.

## Core readings

- Dwork & Roth (2014), ch. 1–3 → chapters 18–22; ch. 4 → chapter 23; ch. 8 → chapter 27
- Kamath, *Algorithms for Private Data Analysis* (CS 860, University of Waterloo, 2020), recorded lectures → chapters 18–28
- Near & Abuah, *Programming Differential Privacy* → code alongside every chapter
- Balle & Wang, "Improving the Gaussian Mechanism for Differential Privacy" (ICML 2018) → chapter 20
- Kasiviswanathan et al., "What Can We Learn Privately?" (FOCS 2008) → chapter 26

## Further readings

- Vadhan, *The Complexity of Differential Privacy* (2017) → chapter 27
- Kairouz, Oh & Viswanath, "The Composition Theorem for Differential Privacy" (ICML 2015) → chapter 21
- Nissim, Raskhodnikova & Smith, "Smooth Sensitivity and Sampling in Private Data Analysis" (STOC 2007) → chapter 24
- Canonne, Kamath & Steinke, "The Discrete Gaussian for Differential Privacy" (NeurIPS 2020) → chapter 25
- Hardt & Rothblum, "A Multiplicative Weights Mechanism for Privacy-Preserving Data Analysis" (FOCS 2010) → chapter 23
- Karwa & Vadhan, "Finite Sample Differentially Private Confidence Intervals" (ITCS 2018) → chapter 28
- Biswas, Dong, Kamath & Ullman, "CoinPress: Practical Private Mean and Covariance Estimation" (NeurIPS 2020) → chapter 28
- Kifer & Machanavajjhala, "No Free Lunch in Data Privacy" (SIGMOD 2011) → chapter 18

Each chapter ends with its own references, with what each is good for.

## Exercise

Implement Laplace, Gaussian (classic and analytic calibration), exponential and sparse vector from scratch; then plot $\varepsilon$ after $k$-fold composition of the Gaussian mechanism under basic, advanced and optimal composition. The mechanisms are implemented in [laplace_gaussian.py](../../code/part03-dp-foundations/laplace_gaussian.py) and [selection.py](../../code/part03-dp-foundations/selection.py), and the composition plot is in [Advanced and optimal composition](04-composition.md). Try them yourself before reading the scripts.

## Exit check

You are done with this part when you can do each of these on paper, without notes:

- [ ] State $(\varepsilon, \delta)$-DP for add/remove and replace neighbours, and show that $\varepsilon$ under add/remove gives $2\varepsilon$ under replace.
- [ ] Prove that randomized response with a fair second coin is $\ln 3$-DP, and derive its unbiased estimator.
- [ ] Derive $\mathrm{FPR} + e^\varepsilon\,\mathrm{FNR} \ge 1 - \delta$ and the resulting bound on membership advantage.
- [ ] Explain why $\delta$ must be much smaller than $1/n$.
- [ ] Prove post-processing, group privacy for pure DP, and basic composition.
- [ ] Prove the Laplace mechanism private, and derive the exact $\delta(\varepsilon)$ of the Gaussian mechanism.
- [ ] Prove the exponential mechanism private and state its utility bound; explain why report-noisy-max costs $\varepsilon$ and not $k\varepsilon$.
- [ ] Prove AboveThreshold private, and say what goes wrong if it never halts.
- [ ] Explain why private multiplicative weights pays $\ln|Q|$ where noise on every answer pays $|Q|$, and when Gaussian noise on every answer still wins.
- [ ] Explain why noise scaled to local sensitivity is not private, and what smooth sensitivity fixes.
- [ ] Derive advanced composition from the privacy loss random variable and Azuma's inequality.
- [ ] Show the $\sqrt n$ error gap between local and central DP for counting.
- [ ] Prove a packing lower bound for pure DP.
- [ ] Estimate a mean privately when the range is unknown, and say what clipping costs on heavy-tailed data.
- [ ] Explain the floating-point attack on the Laplace mechanism, and sample a discrete Laplace exactly.
