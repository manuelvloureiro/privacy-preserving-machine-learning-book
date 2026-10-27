# Differential privacy foundations

The previous part showed that models and statistics leak membership, attributes, training text and even whole datasets. This part introduces the defence that the rest of the book uses, **differential privacy (DP)**. It goes far enough that you can prove a new mechanism private, calibrate its noise, compose it with others, and say what any such mechanism must cost. The [next part](../part04-privacy-accounting/00-introduction.md) builds on it with finer tools for many steps: Rényi DP, numerical accountants and amplification by subsampling.

## Who this part is for

This part assumes the [Prerequisites](../part01-prerequisites/00-introduction.md) and [Threat models and attacks](../part02-attacks/00-introduction.md). From [Divergences](../part01-prerequisites/03-divergences.md) it uses the max-divergence, the hockey-stick divergence and the data processing inequality. It also uses the trade-off function of [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) and the Laplace and normal tail bounds of [Concentration inequalities](../part01-prerequisites/02-concentration.md). The definition of DP bounds the games of [Threat models and privacy games](../part02-attacks/01-threat-models.md), so their advantage returns here. Each chapter restates in one sentence every fact it takes from an earlier page, with a link to its proof.

A reader who has done the course can skim the core chapters. The full edition goes beyond the course in the privacy profile, discrete mechanisms, trust models and mean estimation.

## The common thread

A randomized algorithm is differentially private when a change to one person's record changes the distribution of its output by at most a factor $e^\varepsilon$, up to a small slack $\delta$. Everything in this part comes from three readings of that sentence.

- **As a bound on a divergence.** Pure $\varepsilon$-DP bounds the max-divergence between the outputs on neighbouring datasets, and $(\varepsilon, \delta)$-DP bounds their hockey-stick divergence. So the data processing inequality makes post-processing free. The **privacy loss** $L_{D,D'}$ is the log of the likelihood ratio, and its sum controls composition.
- **As a bound on every attack.** Any test of "was this record used" has $\mathrm{FPR} + e^\varepsilon\,\mathrm{FNR} \ge 1 - \delta$. So every game of the attacks part, from membership inference to reconstruction, has a cap that you can compute before you touch the data.
- **As a budget to spend.** Each release loses a little privacy, and its noise is set by how much one record can move the answer, its **sensitivity**. These losses add up under composition, and lower bounds say how much accuracy every mechanism must give up.

@tbl-dp-map lists the chapters.

| Chapter | The question it answers | Builds on |
| --- | --- | --- |
| [The definition of differential privacy](01-definition.md) | What exactly does DP promise, to whom, and what does it not protect? How do you tell a privacy violation from learning about the population? | 3, 4, 9–11, 13 |
| [Post-processing, group privacy and composition](02-properties.md) | Which operations keep a guarantee, and how do guarantees add up? | 2, 3, 9, 10, 18 |
| [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md) | How much noise does a numerical query need, and what is the exact privacy of Gaussian noise? | 1–4, 18, 19 |
| [Advanced and optimal composition](04-composition.md) | How does privacy degrade over many releases, exactly? | 2–4, 18–20 |
| [Selection: exponential mechanism, report-noisy-max, sparse vector](05-selection.md) | How do you privately choose the best of many options, or find the few queries above a threshold? | 1, 2, 18–21 |
| [Query release and private multiplicative weights](06-query-release.md) | How do you answer millions of linear queries, or publish a synthetic dataset, when noise on every answer would drown them? | 2, 17, 18, 20–22 |
| [Beyond global sensitivity](07-sensitivity.md) | What if the worst case is far from the data you actually have? | 1, 2, 3, 9, 18–20, 22 |
| [Discrete and floating-point-safe mechanisms](08-discrete-mechanisms.md) | Why do textbook mechanisms break on real computers, and how is that fixed? | 1, 3, 18–20 |
| [Local, shuffle and distributed trust models](09-trust-models.md) | What changes when no one is trusted with the raw data? | 1–4, 8, 9, 18–20 |
| [Lower bounds: what privacy costs](10-lower-bounds.md) | How much accuracy must every private mechanism give up? | 2, 4, 5, 10, 17–20, 22–24 |
| [Private mean estimation](11-mean-estimation.md) | How many records does a private estimate of a population mean need, and what does not knowing the range, the dimension or the tails cost? | 1, 2, 18–22, 24, 27 |

: The chapters of this part, the question each one answers, and the earlier chapters it uses, by number (chapters 1–8 are the prerequisites, 9–17 the attacks part). {#tbl-dp-map}

## How to read this part

**Order.** Read chapters 18 to 21, the core, in order. They give the definition, the properties that make it usable, the two mechanisms that everything else uses, and advanced and optimal composition. Chapter 21 finishes the composition story of chapter 19 and introduces the privacy loss random variable, which the next part uses throughout. Chapter 22 adds mechanisms for selection, and chapter 23 uses them to answer many queries at once with private multiplicative weights. Chapter 24 handles data whose worst case is far from typical. Read chapter 25 before you implement a mechanism yourself. Chapter 26 changes who holds the raw data, and chapter 27 says what privacy must cost. Chapter 28 applies these tools to a mean and matches each of its error rates with a lower bound from chapter 27.

**Structure of a chapter.** Every chapter opens with its question and a running example with numbers, such as the counting query of [the Laplace and Gaussian mechanisms](03-laplace-gaussian.md) or the dashboard of a hundred counts in [Advanced and optimal composition](04-composition.md). Then comes one section per topic. Its results are numbered environments, so that later chapters can cite them, and each theorem has its proof right after it, in numbered steps. A Code section runs the script that checks every quoted number. A Summary table lists each result with where it is used, and References close the chapter.

**Proofs you can check by computer.** The scripts check most claims of this part with numbers as well as proofs. For example, they compare a mechanism's exact privacy profile with its bound, a composition formula with brute-force enumeration, and a buggy mechanism with its computed privacy loss. Run them to see how loose each bound is. From the repository root, for example, run

```bash
uv run python code/part03-dp-foundations/laplace_gaussian.py
```

## Notation

The chapters share the conventions in @tbl-dp-notation, set in [the definition of differential privacy](01-definition.md), and the next part uses them too. There $\sigma$ usually stands for the noise multiplier, the noise standard deviation per unit of sensitivity, which is the standard deviation when the sensitivity is 1.

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
| $\mathrm{FPR}$, $\mathrm{FNR}$ | A test's false positive and false negative rates, its type I and type II errors, written $\alpha$ and $\beta$ only on a page where neither letter has another meaning |
| $R\colon \mathcal X \to \mathcal Y$, $\varepsilon_0$ | A local randomizer and its privacy parameter |

: Notation shared by the chapters of this part and the next. {#tbl-dp-notation}

## What you will be able to do

At the end of this part you can:

- state pure and approximate DP for a stated neighbouring relation and unit of privacy, and translate a guarantee into a bound on membership advantage and on posterior odds;
- say why $\delta$ must be much smaller than one over the number of records;
- prove post-processing, group privacy and basic composition, and say what does not compose;
- calibrate the Laplace and Gaussian mechanisms, including the exact (analytic) Gaussian calibration;
- derive advanced composition and compute optimal composition exactly;
- prove the exponential mechanism, report-noisy-max and the sparse vector technique private, and recognize the common broken variants of the last;
- answer a large workload of linear queries with private multiplicative weights, and bound its error through the regret of multiplicative weights;
- handle data-dependent sensitivity with smooth sensitivity, propose-test-release, inverse sensitivity, subsample-and-aggregate and clipping;
- implement discrete Laplace and discrete Gaussian samplers exactly, and explain why naive floating-point noise is not private;
- design frequency estimation in the local model and explain what shuffling and secure aggregation buy;
- prove packing and fingerprinting lower bounds, and run the tracing attack they come from;
- estimate a mean privately when its range, its dimension or its tails are unknown, and match each error rate with a lower bound.
