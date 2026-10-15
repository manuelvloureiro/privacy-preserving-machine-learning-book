# Differential privacy foundations

The previous part showed that models and statistics leak: membership, attributes, training text, whole datasets. This part introduces the defence that the rest of the book is built on, **differential privacy (DP)**, and develops it far enough that you can prove a new mechanism private, calibrate its noise, compose it with others, and say what any such mechanism must cost. The finer tools for tracking privacy over many steps (Rényi DP, numerical accountants, amplification by subsampling) come in the [next part](../part04-privacy-accounting/00-introduction.md), which builds directly on this one.

## Who this part is for

You are expected to have read the [Prerequisites](../part01-prerequisites/00-introduction.md) and [Threat models and attacks](../part02-attacks/00-introduction.md). This part uses the max-divergence and the hockey-stick divergence, the data processing inequality, the trade-off function of a hypothesis test, and tail bounds for the Laplace and normal distributions, all derived in the prerequisites. It uses the games and the advantage of the attacks part, since the definition of DP is a bound on exactly those games.

## The common thread

A randomized algorithm is differentially private when changing one person's record changes the distribution of its output by at most a factor $e^\varepsilon$, up to a small slack $\delta$. Everything in this part follows from reading that one sentence in three ways.

- **As a bound on a divergence.** Pure $\varepsilon$-DP bounds the max-divergence between the outputs on neighbouring datasets, and $(\varepsilon, \delta)$-DP bounds their hockey-stick divergence. The data processing inequality then gives post-processing for free, and the **privacy loss** $L_{D,D'}$, the log of the likelihood ratio, is the quantity whose sum governs composition.
- **As a bound on every attack.** By the hypothesis-testing view, $\mathrm{FPR} + e^\varepsilon\,\mathrm{FNR} \ge 1 - \delta$ for any test of "was this record used". Every game of the attacks part, from membership inference to reconstruction, is therefore capped by a number you can compute before the data is touched.
- **As a budget to spend.** Privacy is lost a little at each release. Noise is calibrated to how much one record can move the answer (its **sensitivity**), losses add up under composition, and lower bounds say how much accuracy no mechanism can avoid paying.

@tbl-dp-map lists the chapters.

| Chapter | The question it answers | Builds on |
| --- | --- | --- |
| [The definition of differential privacy](01-definition.md) | What exactly does DP promise, to whom, and what does it not protect? | 3, 4, 9, 10 |
| [Post-processing, group privacy and composition](02-properties.md) | Which operations keep a guarantee, and how do guarantees add up? | 3, 18 |
| [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md) | How much noise does a numerical query need, and what is the exact privacy of Gaussian noise? | 1, 2, 3, 18, 19 |
| [Advanced and optimal composition](04-composition.md) | How does privacy degrade over many releases, exactly? | 2, 3, 4, 18, 19 |
| [Selection: exponential mechanism, report-noisy-max, sparse vector](05-selection.md) | How do you privately choose the best of many options, or find the few queries above a threshold? | 1, 2, 18–20 |
| [Beyond global sensitivity](06-sensitivity.md) | What if the worst case is far from the data you actually have? | 1, 2, 3, 9, 18–20 |
| [Discrete and floating-point-safe mechanisms](07-discrete-mechanisms.md) | Why do textbook mechanisms break on real computers, and how is that fixed? | 1, 3, 18, 20 |
| [Local, shuffle and distributed trust models](08-trust-models.md) | What changes when no one is trusted with the raw data? | 2, 8, 9, 18, 20 |
| [Lower bounds: what privacy costs](09-lower-bounds.md) | How much accuracy must every private mechanism give up? | 2, 4, 5, 10, 17, 18–20 |

: The chapters of this part, the question each one answers, and the earlier chapters it uses, by number (chapters 1–8 are the prerequisites, 9–17 the attacks part). {#tbl-dp-map}

## How to read this part

**Order.** Chapters 18 to 21 are the core and should be read in order: the definition, the properties that make it usable, the two mechanisms everything else is built from, and advanced and optimal composition, which finishes the story of composition that chapter 19 starts and introduces the privacy loss random variable that the next part uses throughout. Chapters 22 and 23 add mechanisms, for selection and for data whose worst case is far from typical. Chapter 24 is short and practical: read it before you implement any mechanism yourself. Chapter 25 changes who is trusted with the raw data. Chapter 26 is the most theoretical; it ties back to reconstruction and membership attacks and closes the part with what privacy costs.

**Structure of a chapter.** Every chapter has the same shape as in the earlier parts: why it matters, a toolkit, one section per result with a statement, derivation, interpretation, worked example and two *check yourself* questions, then code, references and a cheat sheet. Exercises with worked solutions are collected in the exercise book, a companion website, with a link at the end of each chapter.

**Proofs you can check by computer.** Most privacy claims in this part are checked numerically as well as proved: the exact privacy curve of a mechanism is computed and compared with its bound, a composition formula is compared with brute-force enumeration, and a buggy mechanism is shown to leak by computing its privacy loss. Running the scripts is the fastest way to see how loose each bound is. From the repository root, for example,

```bash
uv run python code/part03-dp-foundations/laplace_gaussian.py
```

## Notation

The chapters share the conventions in @tbl-dp-notation, set in [The definition of differential privacy](01-definition.md). The next part uses the same notation, except that $\sigma$ there is the noise multiplier (the noise standard deviation per unit of sensitivity) and an attacker's two error rates are written $a$ and $b$.

| Symbol | Meaning |
| --- | --- |
| $M$ | A randomized mechanism, mapping a dataset to a random output |
| $D \simeq D'$ | Neighbouring datasets: by default one is the other with one record added or removed; "replace one record" is named explicitly when used |
| $D' = D \cup \{x\}$ | Under add/remove, $D'$ is the dataset that contains the record $x$ and $D$ the one without it. A membership test says "$D'$" when it guesses that the record is in; its false positive rate is the chance of saying "$D'$" when the data were $D$, and its true positive rate the chance of saying "$D'$" when they were $D'$ |
| $\varepsilon$, $\delta$ | The privacy parameters of $(\varepsilon, \delta)$-DP; pure DP is $\delta = 0$ |
| $\Delta_1 f$, $\Delta_2 f$ | The $\ell_1$ and $\ell_2$ sensitivities of a query $f$ |
| $L_{D,D'}(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)}$ | The privacy loss of output $y$; $L_{D,D'}(Y)$ with $Y \sim M(D)$ is the privacy loss random variable |
| $E_{e^\varepsilon}(P \,\|\, Q)$ | The hockey-stick divergence; $(\varepsilon,\delta)$-DP means $E_{e^\varepsilon}(M(D) \,\|\, M(D')) \le \delta$ for all $D \simeq D'$ |
| $\mathrm{Lap}(b)$, $\mathcal N(0, \sigma^2)$ | Laplace noise with scale $b$; normal noise with variance $\sigma^2$ |

: Notation shared by the chapters of this part and the next. {#tbl-dp-notation}

## What you will be able to do

By the end of this part you should be able to:

- state pure and approximate DP for a stated neighbouring relation and unit of privacy, and translate a guarantee into a bound on membership advantage and on posterior odds;
- prove post-processing, group privacy and basic composition, and say what does not compose;
- calibrate the Laplace and Gaussian mechanisms, including the exact (analytic) Gaussian calibration;
- derive advanced composition and compute optimal composition exactly;
- prove the exponential mechanism, report-noisy-max and the sparse vector technique private, and recognise the common broken variants of the last;
- handle data-dependent sensitivity with smooth sensitivity, propose-test-release, subsample-and-aggregate and clipping;
- implement discrete Laplace and discrete Gaussian samplers exactly, and explain why naive floating-point noise is not private;
- design frequency estimation in the local model and explain what shuffling and secure aggregation buy;
- prove packing and fingerprinting lower bounds, and run the tracing attack they come from.

Each of these is a result derived in one of the chapters that follow.
