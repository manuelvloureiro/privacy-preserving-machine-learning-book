# Information theory

## What it is {#sec-info-what}

Information theory measures how much one quantity tells you about another, with two sets of measures for two questions. The first, Shannon's, asks how many bits an observation saves, on average, when you describe a hidden value. The **entropy** of a random variable $X$ is the average number of bits per value that the best compression code needs. Writing $p(x)$ for the probability of the value $x$, with logarithms base 2, it is
$$H(X) = -\sum_x p(x)\log_2 p(x).$$
The **mutual information** $I(X; Y)$ is the drop in the entropy of $X$ that seeing an observation $Y$ causes on average. From a joint table it is $H(X) + H(Y) - H(X, Y)$, where $H(X, Y)$ is the entropy of the pairs. Unlike a correlation, it is zero only when the two are independent.

The second question asks how much more likely a single best guess becomes. A guesser with one try picks the most likely value, so it wins with the largest probability, called the **vulnerability** $V(X)$. Its negative logarithm is the **min-entropy**,
$$H_\infty(X) = -\log_2 \max_x p(x),$$
which is never larger than the entropy.

## When you meet it {#sec-info-when}

Feature selection often ranks features by mutual information, which decision trees call the **information gain**. Min-entropy matters whenever a value must be hard to guess, such as a password or an identifier. So use Shannon's measures for averages and features, and min-entropy for any value that an attacker gets one guess at.

## How to use it {#sec-info-use}

`scipy.stats.entropy(p, base=2)` returns the entropy of a distribution in bits. For two discrete columns, `sklearn.metrics.mutual_info_score(a, b)` plugs their observed frequencies into the definition of mutual information. Continuous values rarely repeat, so `mutual_info_classif` in `sklearn.feature_selection` estimates it from nearest-neighbour distances instead. Both return nats (natural-log units), so divide by $\ln 2$ to get bits.

The two measures can disagree sharply. A uniform 4-digit PIN has $\log_2 10{,}000 = 13.2877$ bits of entropy. Now suppose 10% of people choose 1234 and the rest choose evenly among the others. The entropy barely moves, to 12.4278 bits. But the min-entropy falls to 3.3219 bits, because the guess "1234" succeeds one time in ten.

To measure how much an output helps one guess, start from two tables. The prior $\pi(x)$ says how likely each secret is, and the channel $K[x, y]$ says how likely each output $y$ is when the secret is $x$. Multiplying each row of the channel by its prior gives the joint table. The best guess after seeing $y$ is the row with the largest entry in column $y$, so its success probability is the sum of the column maxima:
$$V(X \mid Y) = \sum_y\max_x \pi(x)K[x, y].$$ {#eq-posterior-vulnerability}
The logarithm of $V(X \mid Y)/V(X)$ is the **min-entropy leakage**, and 1 bit of it doubles the chance of a correct guess. In the full edition's example with four secrets, the output raises that chance from 0.4 to 0.66, a leakage of $\log_2(0.66/0.4) = 0.7225$ bits.

Without a known prior, use the worst case over priors, the **min-capacity**. It is the logarithm of the sum of the channel's own column maxima, 1.2630 bits in the example.

Mutual information can also show that a task is hard, through the uncertainty $H(X \mid Y) = H(X) - I(X; Y)$ that the output leaves. Take a secret with $m$ possible values, and a guess from the output with error probability $P_e$. Writing $h(P_e)$ for the entropy of a coin with that bias, Fano's inequality says
$$H(X \mid Y) \le h(P_e) + P_e\log_2(m - 1).$$ {#eq-fano}
For example, a label uniform over 8 classes has 3 bits of entropy. If the features carry at most 1 bit about it, 2 bits remain. Solving @eq-fano for $P_e$ numerically, with `scipy.optimize.brentq`, then shows that every classifier errs at least 0.3730 of the time.

## What goes wrong {#sec-info-wrong}

Plug-in estimates of mutual information run high on small samples, so two independent columns rarely give exactly zero. Compare an estimate with the same estimate after shuffling one column, which shows what independence alone produces.

Low mutual information does not mean that little is revealed, just as high entropy did not make the PIN hard to guess. Take a uniform 64-bit secret, and a channel that outputs all of it one time in eight, when its top 3 bits are zero, and a fixed symbol otherwise. Its mutual information is only 8.1686 bits, yet a guesser who sees its output wins one time in eight. Min-entropy leakage catches this channel, at 61 bits, but it too averages over secrets, so it can stay small when one rare value is fully exposed.

Processing an output cannot add information about its source. If a variable $Z$ is computed from $Y$ alone, possibly with fresh randomness, the data-processing inequality says
$$I(X; Z) \le I(X; Y).$$
The condition matters, because a $Z$ that also looked at $X$ can know more.

## References

- Smith, "On the Foundations of Quantitative Information Flow" (FoSSaCS 2009): why one-try guessing needs min-entropy.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/05-information-theory.md).
