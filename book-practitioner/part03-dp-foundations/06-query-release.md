# Query release and private multiplicative weights

## Why this matters

Many private releases are a large batch of simple statistics rather than a single number. A statistics office, for example, publishes every cross-tabulation of a census file. Each of these numbers is a **linear query**, the average over the records of a function that gives each record a value between 0 and 1. The most common batch is a set of **marginal tables**. Each table counts the records for every combination of values of a few attributes, and each count is a **cell**.

Adding noise to every answer is the direct approach, but the noise has to grow with the number of queries, and with millions of queries the answers drown. **Private multiplicative weights** fills the gap. It answers any set of linear queries with an error that grows only with the logarithm of their number. It also produces a **synthetic dataset**, a weighted list of possible records that answers every query in the set.

## Which method to use {#sec-qr-choose}

The choice depends on how many queries you have, compared with the number of records, and on how many attributes a record has (@tbl-qr-choose).

| Situation | Use |
| --- | --- |
| Hundreds or thousands of queries, especially marginals | Gaussian noise on every answer, scaled to the structure of the set |
| The same, but pure differential privacy is required | Laplace noise on every answer |
| Far more queries than records, or queries not known in advance | Synthetic data |
| The same, with few attributes | Offline private multiplicative weights |
| Queries arrive one at a time, perhaps adaptively | Online private multiplicative weights |
| The same, with dozens of attributes | A graphical-model method such as MST or AIM |

: Which query-release method to use. {#tbl-qr-choose}

## Noise on every answer {#sec-qr-baseline}

Write $n$ for the number of records, which is public, and $|Q|$ for the number of queries in the set $Q$. Replacing one record moves each answer by at most $1/n$. So the $\ell_1$ sensitivity of all the answers together is at most $|Q|/n$, and their $\ell_2$ sensitivity is at most $\sqrt{|Q|}/n$.

Fix a target error $\alpha$ and a failure probability $\beta$. At privacy level $\varepsilon$, the Laplace mechanism of [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md) keeps every error below $\alpha$, with probability at least $1 - \beta$, once
$$n \ge \frac{|Q|\ln(|Q|/\beta)}{\alpha\varepsilon}.$$
The Gaussian mechanism also allows a small slack $\delta$, and it needs two further numbers. Write $\mu$ for the largest ratio of sensitivity to noise standard deviation that its exact calibration allows, which is 0.2367 at $\varepsilon = 1$ and $\delta = 10^{-6}$. And write $z$ for the number that a standard normal variable exceeds with probability $\beta/(2|Q|)$. It then needs
$$n \ge \frac{\sqrt{|Q|}\,z}{\mu\alpha}.$$
So the records needed grow like $|Q|$ under pure differential privacy (DP), and like $\sqrt{|Q|}$ under approximate DP.

Structure in the queries helps a lot. Replacing one record lowers one cell of a marginal table and raises another, so each table adds only $2/n$ to the $\ell_1$ sensitivity, whatever its number of cells. Take a running example of $n = 10{,}000$ records with eight yes/no attributes, so that there are 256 possible records. Every cell of every marginal table makes 6,560 cells in 255 tables. A uniform histogram, which ignores the data, has a largest error of 0.232 over these cells. At $\varepsilon = 1$, Laplace noise on every cell has a largest error of 0.497, which is worse than ignoring the data. Gaussian noise at $\delta = 10^{-6}$ does much better, with 0.037, and on this example nothing in this chapter beats it.

Two mistakes are common. Scaling each cell's noise to its own sensitivity of $1/n$ silently spends $\varepsilon$ once per cell. And splitting $\varepsilon$ evenly over thousands of separately noised queries with basic composition wastes most of the budget. Calibrate to the joint sensitivity instead, or use the [composition](04-composition.md) rules.

## Private multiplicative weights {#sec-qr-pmw}

The engine is **multiplicative weights**, a method from online learning. It keeps a weight on each of $N$ possible records, which together form a guessed histogram, starting from uniform. Each round it takes a query on which the guess is off by more than $\alpha$. It then moves the weights of the records that query counts up or down to correct it. Online learning caps the number of such corrections at
$$\frac{4\ln N}{\alpha^2},$$ {#eq-qr-mistakes}
whatever the number of queries. That is why the privacy budget pays for corrections rather than for queries.

The private version, the **multiplicative weights exponential mechanism (MWEM)** of Hardt, Ligett and McSherry (2012), runs a fixed number $T$ of rounds. Each round has three steps, and the first two each spend a budget $\varepsilon_0$.

1. **Select.** Pick a query with the [exponential mechanism](05-selection.md#sec-sel-exp), using the guess's error on each query as the score. The score has sensitivity $1/n$.
2. **Measure.** Release the chosen query's answer plus fresh Laplace noise of scale $1/(n\varepsilon_0)$.
3. **Update.** Apply one multiplicative-weights step with that query, toward the measurement.

The release is the average of the $T$ guesses, and giving each step $\varepsilon_0 = \varepsilon/(2T)$ makes it $\varepsilon$-DP by basic composition. Write $\bar p$ for the released average. With probability at least $1 - \beta$, every query $q$ in the set has an error of at most
$$\max_q\big|q(\bar p) - q(D)\big| \le 2\sqrt{\frac{\ln N}{T}} + \frac{2}{n\varepsilon_0}\Big(\ln|Q| + 2\ln\frac{2T}\beta\Big).$$ {#eq-qr-pmw}
The first term is the learning error, which falls with more rounds. The second is the privacy noise, which grows with more rounds because each gets less budget. So fix $T$ in advance by minimizing the bound, which uses only public quantities. On the running example at $\varepsilon = 1$ the bound picks 38 rounds, and the largest error is 0.078, against 0.098 with 10 rounds and 0.123 with 100.

So on 6,560 queries, private multiplicative weights beats Laplace noise but loses to Gaussian noise by a factor of about two. Its advantage is that its error grows only with $\ln|Q|$. For queries with no structure over $2^{20}$ possible records, it overtakes Gaussian noise only near $10^9$ queries.

Two implementation rules matter. The measurement must use fresh noise, because reusing the exponential mechanism's internal scores would use the data without paying for it. And release the average of the guesses, not the last one, because the guarantee covers only the average.

## Synthetic data in practice {#sec-qr-synth}

The running time of private multiplicative weights grows with the number of possible records, because every round updates a weight for each one. Some such cost is unavoidable, because under standard cryptographic assumptions no efficient method produces private synthetic data that is accurate for every two-way marginal (Ullman & Vadhan 2011). So practical tools keep the loop but change its parts, in a pipeline of four steps.

1. **Select** a set of marginal tables to measure, from domain knowledge or privately with the exponential mechanism.
2. **Measure** each whole table with Gaussian noise, scaled to the joint sensitivity of the set.
3. **Generate** a compact graphical model that fits the noisy tables, instead of a full histogram.
4. **Sample** a synthetic dataset from the model.

Everything after the measurements is post-processing, so the synthetic data can be queried and shared at no further privacy cost. The same holds for repairs that make the noisy answers consistent, such as rounding negative counts up to zero or forcing the cells of a table to add up to its total. The 2020 US Census TopDown algorithm works this way at national scale. It measures many tables with discrete Gaussian noise, and then post-processes them into consistent, non-negative counts.

MST and AIM, both from McKenna and coauthors, are the standard methods, and their private-pgm library implements both. SmartNoise Synth, the `snsynth` package, wraps MWEM, MST and AIM behind one interface that fits a table at a given $\varepsilon$ and samples synthetic rows. For the noisy-marginal baseline alone, OpenDP and Tumult Analytics provide Gaussian counts with the accounting done for you.

## Online release and SmallDB {#sec-qr-online}

A query server that receives queries one at a time, perhaps chosen adaptively, can run the online form of private multiplicative weights (Hardt & Rothblum 2010). It answers from the guess when the [sparse vector technique](05-selection.md#sec-sel-svt) finds the guess good enough, and pays only for updates. It is mainly a guarantee, used in [Adaptive data analysis](../part05-dp-ml/11-adaptive-data-analysis.md). SmallDB (Blum, Ligett & Roth 2008) picks a whole small dataset with one exponential mechanism. On the running example it would weigh about $10^{4568}$ candidates, so it shows what is possible rather than what you can run.

## Code {#sec-qr-code}

This snippet runs private multiplicative weights with pure DP on four yes/no attributes, so 16 possible records, and all 80 marginal cells.

```python
import itertools
import random
from math import exp, log, sqrt

Query = list[float]  # one value in [0, 1] for each possible record


def answer(q: Query, hist: list[float]) -> float:
    return sum(qx * px for qx, px in zip(q, hist))


def laplace(scale: float, rng: random.Random) -> float:
    return rng.expovariate(1 / scale) - rng.expovariate(1 / scale)


def private_mw(
    hist: list[float], n: int, queries: list[Query], eps: float, rounds: int, rng: random.Random
) -> list[float]:
    """Select (exponential mechanism), measure (Laplace), update (multiplicative weights); pure eps-DP."""
    eps0 = eps / (2 * rounds)  # 2T steps under basic composition
    gamma = sqrt(log(len(hist)) / rounds)
    guess = [1 / len(hist)] * len(hist)
    total = [0.0] * len(hist)
    for _ in range(rounds):
        total = [t + g for t, g in zip(total, guess)]
        errors = [abs(answer(q, guess) - answer(q, hist)) for q in queries]
        top = max(errors)
        weights = [exp(eps0 * n * (e - top) / 2) for e in errors]  # score sensitivity 1/n
        q = rng.choices(queries, weights=weights)[0]
        measured = answer(q, hist) + laplace(1 / (n * eps0), rng)  # fresh noise
        sign = 1.0 if answer(q, guess) > measured else -1.0
        guess = [g * (1 - gamma * sign * qx) for g, qx in zip(guess, q)]
        guess = [g / sum(guess) for g in guess]
    return [t / rounds for t in total]  # release the average guess


d = 4
universe = list(itertools.product((0, 1), repeat=d))
queries = [
    [1.0 if all(x[a] == v for a, v in zip(attrs, vals)) else 0.0 for x in universe]
    for k in range(1, d + 1)
    for attrs in itertools.combinations(range(d), k)
    for vals in itertools.product((0, 1), repeat=k)
]
gen = random.Random(0)
n = 5000
records = [tuple(int(gen.random() < p) for p in (0.8, 0.6, 0.3, 0.2)) for _ in range(n)]
hist = [records.count(x) / n for x in universe]
synth = private_mw(hist, n, queries, 1.0, 20, random.Random(1))
print(len(queries))  # 80
print(f"{max(abs(answer(q, synth) - answer(q, hist)) for q in queries):.4f}")  # 0.0877
print(f"{max(abs(answer(q, [1 / 16] * 16) - answer(q, hist)) for q in queries):.4f}")  # 0.3866
```

The synthetic histogram's largest error over the 80 cells is 0.0877, against 0.3866 for the uniform histogram.

## Cheat sheet

| Method | Privacy | Records for error $\alpha$ on every query |
| --- | --- | --- |
| Laplace on every answer | $\varepsilon$ | $\lvert Q\rvert\ln(\lvert Q\rvert/\beta)/(\alpha\varepsilon)$ |
| Gaussian on every answer | $(\varepsilon, \delta)$ | $\sqrt{\lvert Q\rvert}\,z/(\mu\alpha)$ |
| Offline private multiplicative weights | $\varepsilon$ or $(\varepsilon, \delta)$ | Grows with $\ln\lvert Q\rvert$ and $\ln N$; choose $T$ from @eq-qr-pmw |
| Select, measure, generate (MST, AIM) | $\varepsilon$ or $(\varepsilon, \delta)$ | Not a closed form; the practical choice for many attributes |

: The query-release methods of this chapter. $N$ is the number of possible records, $\lvert Q\rvert$ the number of queries, $\beta$ the failure probability, and $\mu$ and $z$ the Gaussian calibration numbers of @sec-qr-baseline. {#tbl-qr-cheat}

**Rules of thumb**

- For hundreds or thousands of queries, add Gaussian noise to every answer, scaled to the structure of the query set.
- Use private multiplicative weights when the queries far outnumber the records, or when you want a synthetic dataset.
- Fix the number of rounds from the bound, never from the data, then measure with fresh noise and release the average guess.
- Beyond a few dozen attributes, use a graphical-model method such as AIM.
- Synthetic data from a generator trained without DP is not anonymised, however synthetic it looks.
- A generator is accurate on the marginals it was fit to by construction, so evaluate it on other queries.

## Further reading

- Hardt, Ligett & McSherry, "A Simple and Practical Algorithm for Differentially Private Data Release" (NeurIPS 2012): MWEM, with experiments on real tables.
- McKenna, Mullins, Sheldon & Miklau, "AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data" (VLDB 2022): the select–measure–update loop as used in practice.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/06-query-release.md), except the hardness of efficient synthetic data (Ullman & Vadhan), which it cites.
