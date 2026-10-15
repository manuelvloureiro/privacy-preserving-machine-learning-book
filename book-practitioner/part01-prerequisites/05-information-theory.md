# Information theory

## Why this matters

How much does an observation tell you about something you cannot see, such as a keyword about whether an email is spam? Information theory has two answers for two questions. **How many bits, on average, does the observation save you?** Shannon entropy and mutual information answer that. **How much more likely is one best guess to be right?** Min-entropy and vulnerability answer that.

The one idea: every quantity here is the logarithm of a probability, either averaged over outcomes or taken at the most likely one. When a rare outcome reveals a lot, the two kinds disagree by orders of magnitude. @tbl-info-overview maps the questions to the sections.

| Question | Quantity | Section |
| --- | --- | --- |
| How uncertain is a value, in bits? | Entropy $H(X)$, conditional entropy | @sec-info-entropy |
| How much does $Y$ tell you about $X$ on average? | Mutual information; processing never adds to it | @sec-info-mi |
| How often does one best guess succeed? | Vulnerability, min-entropy, min-entropy leakage | @sec-info-min |
| Which measure should you trust? | Shannon against min-entropy leakage | @sec-info-shannon-vs-min |
| Can any estimator do better than a given error? | Fano's inequality | @sec-info-fano |

: The questions of this chapter, the quantity that answers each, and where. {#tbl-info-overview}

Logarithms are base 2 here, so quantities are in **bits**. Natural logarithms give **nats**; one nat is $1/\ln 2 \approx 1.443$ bits.

## Entropy {#sec-info-entropy}

**Idea.** The rarer an outcome, the more it tells you. Its **surprisal** is $\log(1/p(x))$, and **entropy** is the average surprisal.

**Result.** For a discrete $X$ with distribution $p$,
$$H(X) = -\sum_x p(x)\log p(x), \qquad 0 \le H(X) \le \log\lvert\mathcal X\rvert,$$
zero for a constant and largest for a uniform variable. For a coin with heads probability $p$ it is the **binary entropy** $h(p)$. The **conditional entropy** $H(X \mid Y) = \sum_y p(y)\,H(X \mid Y = y)$ is the uncertainty left after seeing $Y$, averaged over what you might see. The **chain rule** adds the pieces, $H(X, Y) = H(Y) + H(X \mid Y)$, and on average seeing $Y$ never hurts: $H(X \mid Y) \le H(X)$.

**Example.** One email in eight is spam. Every spam contains the word "offer", and so does one ham email in seven. Before reading, $H(X) = h(1/8) = 0.5436$ bits. Without the word the email is certainly ham (0 bits); with it, spam and ham are equally likely (1 bit). The word appears a quarter of the time, so $H(X \mid Y) = 0.25$ bits.

**Pitfalls.** Conditioning reduces entropy only on average: in the example, the rare observation raised it from 0.5436 bits to a full bit. Keep bits and nats apart when you compare numbers from different sources.

## Mutual information and data processing {#sec-info-mi}

**Idea.** What $Y$ tells you about $X$ is the average drop in uncertainty it causes. The same number measures how far $X$ and $Y$ are from independent.

**Result.** The **mutual information** is
$$I(X; Y) = H(X) - H(X \mid Y) = D\big(P_{XY}\,\big\|\,P_XP_Y\big),$$
the Kullback–Leibler (KL) divergence from the joint distribution to the product of its marginals ([Divergences](03-divergences.md)). It is symmetric, non-negative, and zero exactly under independence. The **data processing inequality** says that if $Z$ is computed from $Y$ alone, possibly with fresh randomness, then $I(X; Z) \le I(X; Y)$.

**Example.** For the spam keyword, $I(X; Y) = 0.5436 - 0.25 = 0.2936$ bits. Detect the keyword with a tokenizer that errs 10% of the time, and the noisy feature carries $I(X; Z) = 0.1668$ bits, 57% of the clean one.

**Pitfalls.** Denoising, feature extraction and training are all processing: they can make information easier to use, never larger. So a bound on the raw observation covers everything computed from it. But a small mutual information is an average, and it can hide rare outcomes on which $Y$ reveals $X$ completely (@sec-info-shannon-vs-min).

## Min-entropy and one-guess leakage {#sec-info-min}

**Idea.** With one try, only the most likely value matters. After an observation, the best guess is the **maximum a posteriori (MAP)** guess, the value most likely given it.

**Result.** The **vulnerability** of $X$ and its **min-entropy** are
$$V(X) = \max_x p(x), \qquad H_\infty(X) = -\log V(X) \le H(X).$$
After observing $Y$, the MAP guesser succeeds with the **posterior vulnerability**, the sum of the largest entry in each column of the joint table $p(x, y)$:
$$V(X \mid Y) = \sum_y\max_x p(x, y).$$
The **min-entropy leakage** $\mathcal L = \log\big(V(X \mid Y)/V(X)\big)$ is the factor by which the observation multiplies the chance of a correct first guess, in bits: 1 bit doubles it. Its worst case over priors, the **min-capacity**, is $\log\sum_y\max_x C[x, y]$, with column maxima taken in the channel $C[x, y] = p(y \mid x)$.

**Example.** A uniform 4-digit PIN has 13.2877 bits of both entropies. If 10% of people choose 1234, the Shannon entropy barely moves, to 12.4278 bits, but the min-entropy falls to 3.3219 bits: one guess in ten succeeds. For four secret values with prior $(0.4, 0.3, 0.2, 0.1)$, observed through the channel of @sec-info-code, the blind guess succeeds with probability 0.4 and the MAP guesser with 0.66 (@fig-guessing), a leakage of 0.7225 bits; the min-capacity is 1.2630 bits.

![The joint distribution $p(x, y) = \pi(x)C[x, y]$ of four secret values and three observations. The MAP guesser picks the boxed cell in each column, and its success probability is their sum $V(X \mid Y) = 0.66$.](figures/information_theory_guessing.svg){#fig-guessing}

**Pitfalls.** The guesser compares joint probabilities, not channel entries: on $y_3$ it picks $x_3$ over $x_4$, whose channel entry is larger, because $x_3$ is twice as likely a priori. Take the log after averaging success probabilities, not before: averaging the per-observation min-entropies gives a conditional min-entropy of 0.6548 bits here instead of 0.5995, and it is not the log of any success rate. If you do not know the prior, report the min-capacity.

## Shannon against min-entropy leakage {#sec-info-shannon-vs-min}

**Idea.** **Shannon leakage**, the mutual information $I(X; Y)$, averages logs, so an outcome of probability $2^{-j}$ that reveals $n$ bits adds only about $2^{-j}n$. Min-entropy leakage takes the log of a success probability, which that outcome lifts above $2^{-j}$.

**Result.** On a uniform $n$-bit secret, revealing the lowest $m$ bits gives both leakages $m$. Revealing the whole secret when its lowest $j$ bits are zero gives Shannon leakage about $2^{-j}n$ but min-entropy leakage close to $n - j$ (@fig-leakage).

![Shannon leakage against min-entropy leakage for two families of deterministic channels on a uniform 64-bit secret. Revealing the low $m$ bits gives equal leakages (diagonal). Revealing the whole secret with probability $2^{-j}$ gives a small Shannon leakage but a min-entropy leakage close to $64 - j$.](figures/information_theory_leakage.svg){#fig-leakage}

**Example.** In @tbl-two-channels, Shannon ranks the first channel as less leaky, yet a one-shot guesser does better against it by a factor of more than $10^{15}$.

| Channel | Shannon leakage | Min-entropy leakage | One-guess success |
| --- | --- | --- | --- |
| Reveal all if the low 3 bits are zero | 8.1686 bits | 61.0000 bits | 0.1250 |
| Reveal the low 9 bits | 9.0000 bits | 9.0000 bits | $2.78\times10^{-17}$ |

: Shannon and min-entropy leakage of two deterministic channels on a uniform 64-bit secret. {#tbl-two-channels}

**Pitfalls.** Use min-entropy when one try matters. Both measures average over the secret, so neither shows how exposed a single value is: revealing a 32-bit secret only when it is all zeros leaks $7.786\times10^{-9}$ Shannon bits and 1 min-entropy bit, yet exposes that value completely. Later parts build a worst-case measure.

## Fano's inequality {#sec-info-fano}

**Idea.** If $X$ is still uncertain after seeing $Y$, every guess of $X$ from $Y$ must often be wrong.

**Result.** For $X$ with $M$ possible values and any estimator with error probability $P_e$,
$$H(X \mid Y) \le h(P_e) + P_e\log(M - 1), \qquad\text{so}\qquad P_e \ge \frac{H(X \mid Y) - 1}{\log M}.$$
Solve the left form numerically for the smallest $P_e$. For estimation, take $M$ candidate parameters whose sample distributions are within KL divergence $\kappa$ bits of each other: from $n$ samples, $P_e \ge 1 - (n\kappa + 1)/\log M$.

**Example.** A label is uniform over 8 classes (3 bits) and the features carry at most 1 bit about it, so $H(X \mid Y) \ge 2$. Every classifier errs at least 0.3730 of the time; the weak form gives 0.3333. With 1024 candidates and $\kappa = 0.01$ bits, 100 samples leave $P_e \ge 0.8000$ and 400 leave $P_e \ge 0.5000$.

**Pitfalls.** Fano is tight only when errors spread evenly over the wrong values. If each of 8 symbols is shifted to its neighbour with probability 0.2, the bound gives 0.0954 against a true MAP error of 0.2000. The weak form says nothing when $H(X \mid Y) \le 1$.

## Code {#sec-info-code}

Everything in @sec-info-min comes from the joint table. This computes the guessing example:

```python
from math import log2

prior = [0.4, 0.3, 0.2, 0.1]
channel = [  # channel[x][y] = probability of observation y given secret x
    [0.8, 0.2, 0.0],
    [0.1, 0.8, 0.1],
    [0.0, 0.5, 0.5],
    [0.0, 0.2, 0.8],
]


def joint(prior: list[float], channel: list[list[float]]) -> list[list[float]]:
    return [[px * c for c in row] for px, row in zip(prior, channel)]


def mutual_information(p: list[list[float]]) -> float:
    """Shannon leakage I(X; Y) in bits, from the joint table p[x][y]."""
    px = [sum(row) for row in p]
    py = [sum(col) for col in zip(*p)]
    return sum(
        pxy * log2(pxy / (px[x] * py[y]))
        for x, row in enumerate(p)
        for y, pxy in enumerate(row)
        if pxy > 0
    )


def posterior_vulnerability(p: list[list[float]]) -> float:
    """Success probability of the best one-shot guess after seeing y: sum of column maxima."""
    return sum(max(col) for col in zip(*p))


p = joint(prior, channel)
v_prior = max(prior)
v_post = posterior_vulnerability(p)
print(f"{v_prior:.4f} {v_post:.4f}")  # 0.4000 0.6600
print(f"{log2(v_post / v_prior):.4f}")  # 0.7225
print(f"{mutual_information(p):.4f}")  # 0.6865
print(f"{log2(sum(max(col) for col in zip(*channel))):.4f}")  # 1.2630
```

It prints the two vulnerabilities, the min-entropy leakage, the Shannon leakage and the min-capacity. The full edition's script, [information_theory.py](../../code/part01-prerequisites/information_theory.py), computes every number in this chapter:

```bash
uv run python code/part01-prerequisites/information_theory.py
```

## Cheat sheet

| Quantity | Formula | Notes |
| --- | --- | --- |
| Entropy | $H(X) = -\sum_x p(x)\log p(x)$ | $0 \le H \le \log\lvert\mathcal X\rvert$ |
| Conditional entropy | $H(X \mid Y) = \sum_y p(y)H(X \mid Y = y)$ | $\le H(X)$ on average only |
| Mutual information | $I(X; Y) = H(X) - H(X \mid Y) = D(P_{XY}\,\Vert\,P_XP_Y)$ | Symmetric; zero iff independent |
| Data processing | $Z$ computed from $Y$: $I(X; Z) \le I(X; Y)$ | Bound the raw observation |
| Vulnerability, min-entropy | $V(X) = \max_x p(x)$, $H_\infty = -\log V$ | $H_\infty \le H$ |
| Posterior vulnerability | $V(X \mid Y) = \sum_y\max_x p(x, y)$ | Success of the MAP guesser |
| Min-entropy leakage | $\log V(X \mid Y)/V(X)$ | 1 bit doubles one-guess success |
| Min-capacity | $\log\sum_y\max_x C[x, y]$ | Worst case over priors |
| Fano | $H(X \mid Y) \le h(P_e) + P_e\log(M - 1)$ | Lower bound on any estimator's error |

: The quantities of this chapter. {#tbl-info-cheat}

**Rules of thumb**

- Shannon quantities average logs; min-entropy quantities take the log of a best-case probability. Use the one that matches how many tries the adversary gets.
- Both are averages over the secret. Neither says how exposed a single value is.
- Processing never adds information: a bound on the raw observation covers everything computed from it.
- To show an estimator must fail, bound the mutual information between the unknown and the data, then apply Fano.

## Further reading

- Cover & Thomas, *Elements of Information Theory* (2nd ed., 2006), ch. 2: the standard reference for every Shannon result here.
- MacKay, *Information Theory, Inference, and Learning Algorithms* (2003): the same with a machine-learning slant, free online.
- Smith, "On the Foundations of Quantitative Information Flow" (FoSSaCS 2009): vulnerability, and why Shannon leakage is the wrong measure for one-try guessing.
- Alvim et al., *The Science of Quantitative Information Flow* (2020): the full theory of leakage and capacity.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/05-information-theory.md).
