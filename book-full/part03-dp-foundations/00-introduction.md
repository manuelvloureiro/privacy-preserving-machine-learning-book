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
| [The definition of differential privacy](01-definition.md) | What exactly does DP promise, to whom, and what does it not protect? How do you tell a privacy violation from learning about the population? | 3, 4, 9, 10 |
| [Post-processing, group privacy and composition](02-properties.md) | Which operations keep a guarantee, and how do guarantees add up? | 3, 18 |
| [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md) | How much noise does a numerical query need, and what is the exact privacy of Gaussian noise? | 1, 2, 3, 18, 19 |
| [Advanced and optimal composition](04-composition.md) | How does privacy degrade over many releases, exactly? | 2, 3, 4, 18, 19 |
| [Selection: exponential mechanism, report-noisy-max, sparse vector](05-selection.md) | How do you privately choose the best of many options, or find the few queries above a threshold? | 1, 2, 18–20 |
| [Query release and private multiplicative weights](06-query-release.md) | How do you answer millions of linear queries, or publish a synthetic dataset, when noise on every answer would drown them? | 2, 18, 20–22 |
| [Beyond global sensitivity](07-sensitivity.md) | What if the worst case is far from the data you actually have? | 1, 2, 3, 9, 18–20, 22 |
| [Discrete and floating-point-safe mechanisms](08-discrete-mechanisms.md) | Why do textbook mechanisms break on real computers, and how is that fixed? | 1, 3, 18, 20 |
| [Local, shuffle and distributed trust models](09-trust-models.md) | What changes when no one is trusted with the raw data? | 2, 8, 9, 18, 20 |
| [Lower bounds: what privacy costs](10-lower-bounds.md) | How much accuracy must every private mechanism give up? | 2, 4, 5, 10, 17, 18–20 |
| [Private mean estimation](11-mean-estimation.md) | How many records does a private estimate of a population mean need, and what does not knowing the range, the dimension or the tails cost? | 1, 2, 20–22, 24, 27 |

: The chapters of this part, the question each one answers, and the earlier chapters it uses, by number (chapters 1–8 are the prerequisites, 9–17 the attacks part). {#tbl-dp-map}

## How to read this part

**Order.** Chapters 18 to 21 are the core and should be read in order: the definition, the properties that make it usable, the two mechanisms everything else is built from, and advanced and optimal composition, which finishes the story of composition that chapter 19 starts and introduces the privacy loss random variable that the next part uses throughout. Chapter 22 adds mechanisms for selection, and chapter 23 builds on them to answer many queries at once, by releasing synthetic data with private multiplicative weights. Chapter 24 handles data whose worst case is far from typical. Chapter 25 is short and practical: read it before you implement any mechanism yourself. Chapter 26 changes who is trusted with the raw data. Chapter 27 is the most theoretical; it ties back to reconstruction and membership attacks and says what privacy must cost. Chapter 28 closes the part by putting these tools together on the most common statistical task, estimating a mean, and matches each of its error rates with a lower bound from chapter 27.

**Structure of a chapter.** Every chapter has the same shape as in the earlier parts: why it matters, a toolkit, one section per result with a statement, derivation, interpretation, worked example and two *check yourself* questions, then code, references and a cheat sheet. Exercises with worked solutions are collected in the exercise book, a companion website, with a link at the end of each chapter.

**Proofs you can check by computer.** Most privacy claims in this part are checked numerically as well as proved: the exact privacy profile of a mechanism is computed and compared with its bound, a composition formula is compared with brute-force enumeration, and a buggy mechanism is shown to leak by computing its privacy loss. Running the scripts is the fastest way to see how loose each bound is. From the repository root, for example,

```bash
uv run python code/part03-dp-foundations/laplace_gaussian.py
```

## Notation

The chapters share the conventions in @tbl-dp-notation, set in [the definition of differential privacy](01-definition.md) and fixed for the whole book. The next part uses the same notation. There the noise multiplier, the noise standard deviation per unit of sensitivity, is usually written $\sigma$; it equals the standard deviation when the sensitivity is 1.

| Symbol | Meaning |
| --- | --- |
| $D = (x_1, \dots, x_n)$, $\mathcal X$ | A dataset of $n$ records, each from the data universe $\mathcal X$, so $D \in \mathcal X^n$ |
| $M$; $M_1, \dots, M_k$ | A mechanism, a randomized algorithm whose privacy is analysed; $k$ mechanisms in a composition |
| $y \in \mathcal Y$, $S \subseteq \mathcal Y$ | An output, the output space and an event; $Y = M(D)$ is the random output |
| $D \sim D'$ | Neighbouring datasets: by default one is the other with one record added or removed; "replace one record" is named explicitly when used |
| $D' = D \cup \{x\}$ | Under add/remove, $D'$ is the dataset that contains the record $x$ and $D$ the one without it. A membership test says "$D'$" when it guesses that the record is in; its false positive rate is the chance of saying "$D'$" when the data were $D$, and its true positive rate the chance of saying "$D'$" when they were $D'$ |
| $\varepsilon$, $\delta$ | The privacy parameters: $\varepsilon$-DP is pure DP, $(\varepsilon, \delta)$-DP approximate DP |
| $f$; $\Delta_1 f$, $\Delta_2 f$ | A query $f\colon \mathcal X^n \to \mathbb R^d$ with output dimension $d$; its $\ell_1$ and $\ell_2$ sensitivities, written $\Delta f$ when the norm is clear |
| $L_{D,D'}(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)}$ | The privacy loss of output $y$; $L = L_{D,D'}(Y)$ with $Y \sim M(D)$ is the privacy loss random variable |
| $E_\gamma(P \,\|\, Q)$, $\delta(\varepsilon)$ | The hockey-stick divergence at level $\gamma \ge 1$; $(\varepsilon,\delta)$-DP means $E_{e^\varepsilon}(M(D) \,\|\, M(D')) \le \delta$ for all $D \sim D'$, and the privacy profile $\delta(\varepsilon)$ is the smallest such $\delta$ |
| $\mathrm{Lap}(b)$, $\mathcal N(0, \sigma^2)$ | Laplace noise with scale $b = \Delta_1 f/\varepsilon$; normal noise with standard deviation $\sigma$ |
| $r \in \mathcal R$, $u(D, r)$, $\Delta u$ | The exponential mechanism's candidates, its score function and the score's sensitivity |
| $\mathcal Q$, $\tau$, $\bot$, $\top$ | A workload of counting queries; the sparse vector technique's threshold and its answers "below" and "above" |
| $(\alpha, \beta)$ | An $(\alpha, \beta)$-accurate mechanism errs by at most $\alpha$ with probability at least $1 - \beta$; $\beta$ is a failure probability |
| $\mathrm{FPR}$, $\mathrm{FNR}$ | A test's false positive and false negative rates, its type I and type II errors, written $\alpha$ and $\beta$ on a page without an accuracy guarantee |
| $R\colon \mathcal X \to \mathcal Y$, $\varepsilon_0$ | A local randomizer and its privacy parameter |

: Notation shared by the chapters of this part and the next. {#tbl-dp-notation}

## What you will be able to do

By the end of this part you should be able to:

- state pure and approximate DP for a stated neighbouring relation and unit of privacy, and translate a guarantee into a bound on membership advantage and on posterior odds;
- prove post-processing, group privacy and basic composition, and say what does not compose;
- calibrate the Laplace and Gaussian mechanisms, including the exact (analytic) Gaussian calibration;
- derive advanced composition and compute optimal composition exactly;
- prove the exponential mechanism, report-noisy-max and the sparse vector technique private, and recognize the common broken variants of the last;
- handle data-dependent sensitivity with smooth sensitivity, propose-test-release, subsample-and-aggregate and clipping;
- implement discrete Laplace and discrete Gaussian samplers exactly, and explain why naive floating-point noise is not private;
- design frequency estimation in the local model and explain what shuffling and secure aggregation buy;
- prove packing and fingerprinting lower bounds, and run the tracing attack they come from.

Each of these is a result derived in one of the chapters that follow.
