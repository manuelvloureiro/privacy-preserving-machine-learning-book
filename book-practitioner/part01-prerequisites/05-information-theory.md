# Information theory

## What it is {#sec-info-what}

Information theory measures how much one quantity tells you about another, with two sets of measures for two questions. The first, Shannon's, asks how many bits an observation saves, on average, when you describe a hidden value. The **entropy** of a random variable $X$ is the average number of bits per value that the best compression code needs. Writing $p(x)$ for the probability of the value $x$, with logarithms base 2, it is
$$H(X) = -\sum_x p(x)\log p(x).$$
The **mutual information** $I(X; Y)$ is the drop in the entropy of $X$ that seeing an observation $Y$ causes on average. From a joint table it is $H(X) + H(Y) - H(X, Y)$, where $H(X, Y)$ is the entropy of the pairs. Unlike a correlation, it is zero only when the two are independent. The most mutual information that a noisy channel can carry, over all input distributions, is its **capacity**.

The second question asks how much more likely a single best guess becomes. A guesser with one try picks the most likely value, so it wins with the largest probability, called the **vulnerability** $V(X)$. Its negative logarithm is the **min-entropy**,
$$H_\infty(X) = -\log \max_x p(x).$$
Min-entropy is never larger than entropy.

## When you meet it {#sec-info-when}

Decision trees call mutual information the **information gain**, and feature selection often ranks features by it. Min-entropy matters instead whenever a value must be hard to guess, such as a password or an identifier. Use Shannon's for averages and features, and min-entropy for any value that an attacker gets one guess at.

## How to use it {#sec-info-use}

`scipy.stats.entropy(p, base=2)` returns the entropy of a distribution in bits. For two discrete columns, `sklearn.metrics.mutual_info_score(a, b)` plugs their observed frequencies into the definition of mutual information. Continuous values rarely repeat, so `mutual_info_classif` in `sklearn.feature_selection` estimates it from nearest-neighbour distances instead. Both return nats (natural-log units), so divide by $\ln 2$ to get bits.

A uniform 4-digit PIN has $\log_2 10{,}000 = 13.2877$ bits of entropy. Now suppose 10% of people choose 1234 and the rest choose evenly among the others. The entropy barely moves, to 12.4278 bits. But the min-entropy falls to 3.3219 bits, because the guess "1234" succeeds one time in ten.

To measure how much an output helps one guess, start from two tables. The prior $\pi(x)$ says how likely each secret is, and the channel $C[x, y]$ says how likely each output $y$ is when the secret is $x$. Multiplying each row of the channel by its prior gives the joint table. The best guess after seeing $y$ is the row with the largest entry in column $y$, so its success probability is the sum of the column maxima:
$$V(X \mid Y) = \sum_y\max_x \pi(x)C[x, y].$$ {#eq-posterior-vulnerability}
The logarithm of $V(X \mid Y)/V(X)$ is the **min-entropy leakage**, and 1 bit of it doubles the chance of a correct guess. In the full edition's example with four secrets, the output raises that chance from 0.4 to 0.66, so the leakage is $\log_2(0.66/0.4) = 0.7225$ bits.

If you do not know the prior, use the worst case over priors, the **min-capacity**. It is the logarithm of the sum of the channel's own column maxima, 1.2630 bits in the example.

Mutual information can also show that a task is hard, through the uncertainty $H(X \mid Y) = H(X) - I(X; Y)$ that it leaves. Take a secret with $M$ possible values, and a guess from the output with error probability $P_e$. Writing $h(P_e)$ for the entropy of a coin with that bias, Fano's inequality says
$$H(X \mid Y) \le h(P_e) + P_e\log(M - 1).$$ {#eq-fano}
For example, a label uniform over 8 classes has 3 bits of entropy. If the features carry at most 1 bit about it, 2 bits remain, and every classifier errs at least 0.3730 of the time. Get such a number by solving @eq-fano for $P_e$ numerically, with `scipy.optimize.brentq`. For a binary label, $M - 1 = 1$ and the last term vanishes.

## What goes wrong {#sec-info-wrong}

Plug-in estimates of mutual information run high on small samples, so two independent columns rarely give exactly zero. Compare an estimate with the same estimate after shuffling one column, which shows what independence alone produces.

Shannon entropy is the wrong way to rate guessing difficulty, as the PIN shows, and low mutual information does not mean that little is revealed. Take a uniform 64-bit secret and a channel that outputs all of it when its top 3 bits are zero, one time in eight, and a fixed symbol otherwise. Its mutual information is 8.1686 bits, less than the 9 bits of a channel that always reveals the low 9 bits. Yet a guesser who sees its output wins one time in eight. Min-entropy leakage catches this channel, at 61 bits, but it too averages over secrets, so it can stay small when one rare value is fully exposed.

Processing an output cannot add information about its source. If a variable $Z$ is computed from $Y$ alone, possibly with fresh randomness, the data-processing inequality says
$$I(X; Z) \le I(X; Y).$$
The condition matters, because a $Z$ that also looked at $X$ can know more.

## Further reading

- Smith, "On the Foundations of Quantitative Information Flow" (FoSSaCS 2009): why one-try guessing needs min-entropy.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/05-information-theory.md).
