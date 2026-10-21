# Information theory

## Why this matters

You often observe one thing and want to know how much it tells you about another thing you cannot see. A word in an email hints at whether the email is spam, and a noisy sensor reading carries a trace of a machine's true state. Later parts ask how much a published output reveals about hidden data. Information theory turns "how much" into a number, in two different ways, because there are two different questions.

The first question is how many bits the output saves, on average, when you describe the secret. Shannon entropy and mutual information answer it, so they are the language of compression, of the cross-entropy loss and of feature selection. The second question is how much more likely a single best guess becomes once you have seen the output. Min-entropy and vulnerability answer that one, so they are the language of password strength and of anyone who gets one try. The two can rank the same channels in opposite orders, and both are averages, so neither says how exposed one particular value is.

Logarithms here are base 2, so every quantity is in **bits**. One nat, the unit of natural logarithms, is about 1.443 bits. SciPy's `scipy.stats.entropy(pk, base=2)` returns bits, but its default and scikit-learn's `sklearn.metrics.mutual_info_score(labels_true, labels_pred)` return nats, so convert before you compare numbers.

## Entropy {#sec-info-entropy}

The **entropy** of a random variable $X$ is its average surprisal, the logarithm of one over the probability of the value that occurs. Writing $p(x)$ for the probability that $X$ takes the value $x$, it is
$$H(X) = -\sum_x p(x)\log p(x),$$
where an outcome of probability zero adds nothing. It is also the average number of bits per value that the best possible code needs. For a coin that lands heads with probability $p$, it is the **binary entropy** $h(p)$, which is 1 bit for a fair coin and zero for a certain one. Even a coin with an 11% chance of heads still carries $h(0.11) = 0.4999$ bits.

Entropy lies between zero, when one value is certain, and the logarithm of the number of possible values, when every value is equally likely. A uniform 4-digit PIN, for example, has $\log_2 10{,}000 = 13.2877$ bits. The same bound sizes a combination of columns, such as birth date, sex and postcode. Its entropy is at most the logarithm of the number of combinations. Once it approaches the logarithm of the number of rows, many rows have a combination of their own.

For two variables, the **conditional entropy** $H(X \mid Y)$ is the uncertainty about $X$ left after seeing $Y$, averaged over the values of $Y$. It is never larger than $H(X)$, but only on average. A particular observation can increase the uncertainty, for example when a rare word makes a probably-legitimate email look equally likely to be spam.

## Mutual information and data processing {#sec-info-mi}

How much does an output $Y$ tell you about a secret $X$? The **mutual information** $I(X; Y)$ is the drop in uncertainty it causes on average. It is also the Kullback–Leibler (KL) divergence between the joint distribution of the pair and the product of the two separate distributions:
$$I(X; Y) = H(X) - H(X \mid Y) = D\big(P_{XY}\,\big\|\,P_XP_Y\big).$$ {#eq-mi-kl}
So it is never negative, it is zero only when the two variables are independent, and it is symmetric. Unlike a correlation, it detects dependence of any kind. Decision trees call it the **information gain**, and many feature-selection methods rank features by it.

To estimate mutual information from data, `mutual_info_score` puts the observed frequencies of two discrete columns into @eq-mi-kl. For continuous features, `mutual_info_classif` and `mutual_info_regression` in `sklearn.feature_selection` use a nearest-neighbour estimator instead. Both are biased. Plug-in estimates from small samples run high, so independent variables rarely give exactly zero. Compare an estimate with the same estimate on data whose pairing you have shuffled, which shows what independence alone produces.

The **data processing inequality (DPI)** says that processing cannot add information. Suppose a third variable $Z$ is computed from $Y$ alone, possibly with fresh randomness of its own. This is written $X \to Y \to Z$, and then
$$I(X; Z) \le I(X; Y).$$
The condition is essential, because a $Z$ that also looked at $X$ directly could know more. In practice, denoising, feature extraction and training are all processing steps, so a bound on what the raw output reveals covers everything anyone computes from it.

A **channel** is the table of output probabilities for each input, and its **capacity** is the largest mutual information over all input distributions. It is the highest rate at which error-correcting codes can send data reliably. For the binary symmetric channel (BSC), which flips each bit with probability $p$, it is $1 - h(p)$.

## Min-entropy and one-guess leakage {#sec-info-min}

A guesser who gets one try wins with the probability of the most likely value. That probability is the **vulnerability** $V(X)$, and its negative logarithm is the **min-entropy**:
$$V(X) = \max_x p(x), \qquad H_\infty(X) = -\log V(X).$$
Min-entropy is never larger than Shannon entropy, and the two are equal only for a uniform distribution. A value with a min-entropy of $k$ bits is as hard to guess in one try as a uniform string of $k$ bits, whatever its Shannon entropy.

A PIN shows why Shannon entropy is the wrong measure of guessing difficulty. Suppose 10% of people choose 1234, and the rest choose uniformly among the other PINs. Then the Shannon entropy barely moves, to 12.4278 bits. But the min-entropy falls to 3.3219 bits, because the guess "1234" succeeds one time in ten.

Now let the guesser see an output $Y$ through a channel $C$, whose entry $C[x, y]$ is the probability of output $y$ when the secret is $x$. Write $\pi$ for the **prior**, how likely each secret is to begin with, so that each entry of the joint table is $\pi(x)C[x, y]$. The best strategy is the maximum a posteriori (MAP) guess, the secret most likely given the output. Its success probability is called the **posterior vulnerability**, and it is the sum of the column maxima of the joint table:
$$V(X \mid Y) = \sum_y\max_x \pi(x)C[x, y].$$ {#eq-posterior-vulnerability}
The **min-entropy leakage** is the logarithm of the factor by which the output multiplies the chance of a correct guess:
$$\mathcal L = \log\frac{V(X \mid Y)}{V(X)}.$$
So a leakage of 1 bit doubles the guesser's chance, which is a statement you can act on.

This is how to quantify one-guess risk on a small table. List the secret's prior, write down the channel from secret to output, and compute the two vulnerabilities. In the example of @sec-info-code, a secret with four values has a best prior guess that is right with probability 0.4. With the output, the guesser wins with probability 0.66, so the leakage is 0.7225 bits. Two shortcuts give wrong answers. Comparing channel entries instead of joint entries ignores the prior. Averaging the min-entropy of each posterior takes the logarithm too early, and gives 0.6548 bits instead of the correct conditional min-entropy of 0.5995 bits.

When you do not know the prior, use the worst case over all priors. That is the **min-capacity**, which needs only the column maxima of the channel itself, and which a uniform prior attains:
$$\mathcal{ML}(C) = \log\sum_y\max_x C[x, y].$$ {#eq-min-capacity}
For the example channel it is 1.2630 bits. For a channel that reports a hidden bit correctly with probability 0.75 and flips it otherwise, it is 0.5850 bits. So whatever the prior, seeing the output multiplies the chance of guessing the bit by at most 1.5.

## Shannon against min-entropy leakage {#sec-info-shannon-vs-min}

The two leakages agree for a channel that reveals the same bits every time. They disagree, by a lot, for a channel that reveals the secret completely on a few outputs and nothing on the rest. Shannon leakage averages the logarithm, so a rare full reveal contributes little. Min-entropy leakage takes the logarithm of a success probability, which the same reveal dominates. @tbl-two-channels compares two deterministic channels on a uniform 64-bit secret.

| Channel | Shannon leakage | Min-entropy leakage | One-guess success |
| --- | --- | --- | --- |
| Reveal all if the low 3 bits are zero | 8.1686 bits | 61.0000 bits | 0.1250 |
| Reveal the low 9 bits | 9.0000 bits | 9.0000 bits | $2.78\times10^{-17}$ |

: Shannon and min-entropy leakage of two deterministic channels on a uniform 64-bit secret. {#tbl-two-channels}

Shannon leakage says the second channel leaks more, yet a guesser with one try does better against the first by a factor of more than $10^{15}$. So low mutual information does not mean that nothing is revealed. A channel can have little mutual information with its secret and still give some values away completely.

Min-entropy leakage has the same blind spot in a milder form, because it too is an average over secrets. A channel that reveals a 32-bit secret only when it is all zeros has a min-entropy leakage of exactly 1 bit, and a far smaller Shannon leakage. Yet for the one secret made of zeros, it reveals everything. Any measure that averages over the secret can hide a value that is fully exposed. So when every value matters, you need a worst-case bound on the likelihood ratio of each output, which later parts build.

## Fano's inequality {#sec-info-fano}

Mutual information can also show that a problem is hard. **Fano's inequality** concerns a secret that takes one of $M$ values. It says that a guess computed from the output, wrong with probability $P_e$, must satisfy
$$H(X \mid Y) \le h(P_e) + P_e\log(M - 1).$$ {#eq-fano}
So if much uncertainty remains after the output, every guess must often be wrong. For a label uniform over 8 classes, with features that carry at most 1 bit about it, every classifier errs at least 0.3730 of the time. Later parts use the same tool to show that some estimates need many samples.

## Code {#sec-info-code}

This snippet computes the vulnerabilities, the min-entropy leakage and the min-capacity for a secret with four values and three outputs, and the min-capacity of a channel that flips a bit with probability 0.25.

```python
from math import log2

prior = [0.4, 0.3, 0.2, 0.1]
channel = [  # channel[x][y] = probability of observation y given secret x
    [0.8, 0.2, 0.0],
    [0.1, 0.8, 0.1],
    [0.0, 0.5, 0.5],
    [0.0, 0.2, 0.8],
]


def posterior_vulnerability(prior: list[float], channel: list[list[float]]) -> float:
    """Success probability of the best one-shot guess: the sum of the joint table's column maxima."""
    joint = [[px * c for c in row] for px, row in zip(prior, channel)]
    return sum(max(col) for col in zip(*joint))


def min_capacity(channel: list[list[float]]) -> float:
    """Leakage in bits under the worst prior: log of the sum of the channel's column maxima."""
    return log2(sum(max(col) for col in zip(*channel)))


v_prior = max(prior)
v_post = posterior_vulnerability(prior, channel)
print(f"{v_prior:.4f} {v_post:.4f}")  # 0.4000 0.6600
print(f"{log2(v_post / v_prior):.4f}")  # 0.7225
print(f"{min_capacity(channel):.4f}")  # 1.2630
flip = [[0.75, 0.25], [0.25, 0.75]]  # report the hidden bit correctly with probability 0.75
print(f"{min_capacity(flip):.4f} {2 ** min_capacity(flip):.4f}")  # 0.5850 1.5000
```

## Cheat sheet

| Question | Quantity | Formula |
| --- | --- | --- |
| How uncertain is a secret, in bits? | Entropy | $H(X) = -\sum_x p(x)\log p(x)$, at most $\log\lvert\mathcal X\rvert$ |
| How much does an output tell on average? | Mutual information | $I(X; Y) = H(X) - H(X \mid Y)$ |
| Can post-processing reveal more? | Data processing | $I(X; Z) \le I(X; Y)$ when $X \to Y \to Z$ |
| How likely is one guess to succeed? | Vulnerability | $V = \max_x p(x)$, $H_\infty = -\log V$ |
| How much does the output help one guess? | Min-entropy leakage | $\log V(X \mid Y)/V(X)$ |
| The worst case over priors? | Min-capacity | $\log\sum_y\max_x C[x, y]$ |
| Is any single value fully exposed? | None of these | Needs a worst-case bound on each output |

: Which quantity answers which question. {#tbl-info-cheat}

**Rules of thumb**

- Check whether a library returns bits or nats.
- Compare a mutual information estimate with the same estimate on shuffled data, because small-sample estimates run high.
- Measure guessing difficulty with min-entropy, never with the Shannon entropy of a password or PIN distribution.
- Low mutual information does not mean nothing is revealed, and an average leakage never shows that every value is safe.

## Further reading

- Cover & Thomas, *Elements of Information Theory* (2nd ed., 2006), ch. 2 and 7: the standard reference for every Shannon result here.
- Smith, "On the Foundations of Quantitative Information Flow" (FoSSaCS 2009): vulnerability, and why Shannon leakage is the wrong measure for one-try guessing.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/05-information-theory.md).
