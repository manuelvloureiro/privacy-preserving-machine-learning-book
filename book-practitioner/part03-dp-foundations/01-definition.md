# The definition of differential privacy

## Why this matters

Every attack in [Threat models and attacks](../part02-attacks/00-introduction.md) runs an algorithm on one of two datasets that differ in one person, and guesses from the output which one was used. **Differential privacy (DP)** makes the two outputs look alike against every test, including tests nobody has invented yet. It is a property of the algorithm, not of the data, so you can check it before any data are touched. This chapter teaches you to read a privacy claim, which means knowing what it protects, what it promises and what it leaves out.

## Neighbouring datasets {#sec-dpdef-neighbours}

The first question about any claim is what "differ in one record" means, and the **neighbouring relation** answers it. Under **add/remove**, two datasets are neighbours when one is the other with one record added or taken away. The guarantee then hides whether a person is in the data at all. Under **replace**, the two datasets have the same size and differ in the value of one record, so only what the record says is hidden. A replacement is a removal followed by an addition, so a guarantee at a privacy level $\varepsilon$ converts as
$$\varepsilon\text{-DP under add/remove} \implies 2\varepsilon\text{-DP under replace}.$$ {#eq-dpdef-replace}
The reverse fails, since only add/remove hides how many records there are.

You also choose the **unit of privacy**, what one record is. At the **example level**, it is one training example or one event, and at the **user level**, all the data of one person. Under example-level DP, a user who wrote 200 messages is a group of 200 records. The guarantee weakens quickly with the size of a group ([Post-processing, group privacy and composition](02-properties.md)), so check that the claim protects the person. The guarantee must also hide every record the data format allows, even a person aged 900, which is why private algorithms clip their inputs before adding noise.

## The definition {#sec-dpdef-definition}

Call a randomized algorithm that takes a dataset and returns an output a **mechanism**, written $M$. Write $D$ for a dataset and $D'$ for one of its neighbours. The definition asks that no set of outputs become much more likely when one record changes. It measures "much more likely" by a factor $e^\varepsilon$, for a non-negative privacy parameter $\varepsilon$, and allows a small additive slack $\delta$ on top. A mechanism is **$(\varepsilon, \delta)$-differentially private** if every pair of neighbours and every set $S$ of outputs satisfy
$$\Pr[M(D) \in S] \le e^\varepsilon\,\Pr[M(D') \in S] + \delta.$$ {#eq-dpdef-def}
The probabilities come only from the mechanism's own randomness, with the data held fixed.

When $\delta = 0$ the guarantee is called **pure DP**, and otherwise **approximate DP**. One consequence matters in practice. A private mechanism must be random, because a deterministic one gives some output that has probability 0 on a neighbour. So rounding, aggregating or suppressing small cells gives no guarantee by itself, however coarse the result looks.

## The privacy loss and δ {#sec-dpdef-divergences}

For an output $y$, write $p_{M(D)}(y)$ for its probability, or its density, when the mechanism runs on $D$. The **privacy loss** of $y$ is the logarithm of how much more likely the output is on $D$ than on $D'$,
$$L(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)}.$$ {#eq-dpdef-loss}
A loss of 0 means that the output says nothing about which dataset was used. Independent outputs multiply their likelihood ratios, so their losses add, which is why the accountants of Part 4 work with the loss. Pure $\varepsilon$-DP is exactly the condition that every output has a loss between $-\varepsilon$ and $\varepsilon$. Laplace noise meets it, while the loss of Gaussian noise grows without limit in the far tail ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)).

Under approximate DP, $\delta$ covers the outputs whose loss exceeds $\varepsilon$. It counts each one only by how far its probability on $D$ goes beyond the ratio bound. This weighted total is called the **hockey-stick divergence**. So $\delta$ is not the chance that the loss exceeds $\varepsilon$, which is a valid but often loose choice. Take a count with Gaussian noise of standard deviation 2, at $\varepsilon = 1$. Its loss exceeds $\varepsilon$ with probability $4.006\times10^{-2}$. The exact $\delta$ is $6.830\times10^{-3}$, less than a sixth of that.

Each $\varepsilon$ has its own smallest $\delta$, and the function $\delta(\varepsilon)$ is the mechanism's **privacy curve**. A quoted pair is one point on it, so report the noise parameters with the pair, and others can recompute the whole curve. [Numerical accounting](../part04-privacy-accounting/06-numerical-accounting.md) computes these curves from the distribution of the loss.

## Randomized response {#sec-dpdef-rr}

The first private mechanism comes from survey design (Warner, 1965). Each person tells the truth with some probability $t$ a little above one half, and lies otherwise, so a single answer proves nothing about them. A person who answers truthfully on heads, and with a second coin toss on tails, tells the truth with probability $\frac34$, which gives pure DP at $\varepsilon = \ln 3$. In general, **randomized response** at level $\varepsilon$ tells the truth with probability
$$t = \frac{e^\varepsilon}{1 + e^\varepsilon}.$$

The guarantee is $\varepsilon$ per person however many people answer, provided their coins are independent, but only under replace. Write $\bar y$ for the share of "yes" reports among the $n$ people. The estimate of the true fraction and its standard deviation are
$$\hat\pi = \frac{\bar y - (1 - t)}{2t - 1}, \qquad \operatorname{sd}(\hat\pi) = \frac{e^{\varepsilon/2}}{(e^\varepsilon - 1)\sqrt n},$$ {#eq-dpdef-rr-estimate}
which is about $1/(\varepsilon\sqrt n)$ for small $\varepsilon$. For 1,000 people, the two-coin scheme gives 0.0274, about twice the sampling error that the survey already has. That is the price of the **local** model, in which nobody sees a true answer ([Local, shuffle and distributed trust models](09-trust-models.md)). diffprivlib implements the mechanism as `mechanisms.Binary`.

## What ε means for an attacker {#sec-dpdef-testing}

A membership attacker looks at the output and decides whether to accuse the target of being in the data, so it has two error rates ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). Its **false positive rate (FPR)** is how often it accuses a target who is out. Its **true positive rate (TPR)** is how often it accuses one who is in. The outputs on which it accuses form a set, which the definition bounds. So under add/remove, every attack at every FPR satisfies
$$\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta.$$ {#eq-dpdef-testing}

This is the form to use, because Part 2 judged attacks by their TPR at a low FPR. For example, take $\varepsilon = 1$ and $\delta = 10^{-5}$. Then at an FPR of $10^{-3}$, no attack reaches a TPR above 0.002728. As the FPR falls towards 0, the $\delta$ term takes over, so at a very low FPR the guarantee rests almost entirely on $\delta$.

Read backwards, the cap turns an attack into an audit. Measure an attack's FPR and TPR, and solve the cap for the smallest $\varepsilon$ consistent with them,
$$\varepsilon \ge \ln\frac{\mathrm{TPR} - \delta}{\mathrm{FPR}}.$$
Suppose a model is claimed to be $(2, 10^{-5})$-DP, and an attack reaches a TPR of 0.05 at an FPR of 0.001. If those rates are exact, $\varepsilon$ is at least 3.91, so the claim is false. Audits plant a **canary**, a record built to be as easy to detect as possible, because the guarantee must hold for the worst record ([Memorization and extraction](../part02-attacks/05-memorization-extraction.md)).

Two summaries are easier to explain to a lawyer or an auditor. The **membership advantage**, the TPR minus the FPR, says how much better than a coin flip an attacker does. Under $(\varepsilon, \delta)$-DP it is at most
$$\mathrm{TPR} - \mathrm{FPR} \le \frac{e^\varepsilon - 1 + 2\delta}{e^\varepsilon + 1},$$ {#eq-dpdef-advantage}
which is the hyperbolic tangent of $\varepsilon/2$ when $\delta = 0$.
The posterior bound speaks about beliefs. One output can multiply an adversary's odds between two neighbours by at most $e^\varepsilon$, whatever it believed before. So a prior belief $\pi$ that the data are $D$ rises at most to
$$\Pr[D \mid y] \le \frac{e^\varepsilon\pi}{e^\varepsilon\pi + 1 - \pi}.$$ {#eq-dpdef-posterior}
@tbl-dpdef-epsilon turns a claimed $\varepsilon$ into both numbers.

| $\varepsilon$ | Advantage bound | Posterior from a prior of 0.01 | Posterior from a prior of 0.5 |
| --- | --- | --- | --- |
| 0.1 | 0.0500 | 0.0110 | 0.5250 |
| 1 | 0.4621 | 0.0267 | 0.7311 |
| 3 | 0.9051 | 0.1687 | 0.9526 |
| 8 | 0.9993 | 0.9679 | 0.9997 |

: What a pure $\varepsilon$-DP guarantee promises against the strongest attacker. {#tbl-dpdef-epsilon}

So an $\varepsilon$ below 1 is strong, 1 to 5 is moderate, and around 10 the worst-case guarantee says little. Real deployments often sit higher, and [Choosing ε](../part05-dp-ml/09-choosing-epsilon.md) discusses how to set and defend a value. The full trade-off curve between the two error rates is the subject of [f-DP and Gaussian differential privacy](../part04-privacy-accounting/04-gaussian-dp.md).

## How small δ must be {#sec-dpdef-small-delta}

A small $\delta$ barely moves the advantage bound, but the advantage is an average that hides harm to a few people. For example, take a mechanism that publishes each record in full, independently with probability $\delta$. Under add/remove it is $(0, \delta)$-DP, because a record changes the output only when its own coin publishes it. Yet over $n$ records it publishes at least one with probability
$$1 - (1 - \delta)^n \approx 1 - e^{-n\delta}.$$
With 10,000 records and $\delta = 1/n$, someone's full record is published with probability 0.6321.

Treat $\delta$ as a per-person budget for catastrophe. Choose it well below $1/n$, for example $1/n^{1.5}$, and let $n$ count people rather than records.

## What differential privacy does not protect {#sec-dpdef-limits}

DP compares a world with your record and a world without it, so anything that can be learned in both worlds is outside the guarantee. A study that never used Alice's record may show that smokers get cancer at a high rate. That changes what anyone believes about Alice, a smoker, yet the study is 0-DP with respect to her. Correlated records are a second gap. A household of five that always answers alike is protected only at five times the per-record $\varepsilon$, unless the household is the unit of privacy. The third gap is the implementation, which needs its own review ([Discrete and floating-point-safe mechanisms](08-discrete-mechanisms.md)).

When someone reports a privacy violation, ask three questions. Would the inference come out materially different without the person's record? Does it succeed exactly when the model is accurate? And when, and by whom, was the revealing information given up? Only the first decides. If the answer is no, as when public "likes" predict someone's politics, it is not a violation in the DP sense, even when it causes harm. A language model that completes a prompt with one user's identification number does fail it, because a model trained without her messages would not.

## Code {#sec-dpdef-code}

This snippet prints rows of @tbl-dpdef-epsilon and turns a measured attack into a lower bound on $\varepsilon$.

```python
from math import exp, log


def advantage_bound(eps: float) -> float:
    """The largest TPR - FPR against any eps-DP mechanism."""
    return (exp(eps) - 1) / (exp(eps) + 1)


def posterior_bound(eps: float, prior: float) -> float:
    """The largest posterior after one output of an eps-DP mechanism."""
    return exp(eps) * prior / (exp(eps) * prior + 1 - prior)


def epsilon_lower_bound(tpr: float, fpr: float, delta: float) -> float:
    """The smallest eps consistent with an attack measured at (fpr, tpr)."""
    return log((tpr - delta) / fpr)


print(f"{advantage_bound(3.0):.4f}")  # 0.9051
print(f"{advantage_bound(8.0):.4f}")  # 0.9993
print(f"{posterior_bound(8.0, 0.01):.4f}")  # 0.9679
print(f"{posterior_bound(8.0, 0.5):.4f}")  # 0.9997
print(f"{epsilon_lower_bound(0.05, 1e-3, 1e-5):.2f}")  # 3.91
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| $(\varepsilon, \delta)$-DP | $\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta$ | All neighbours, all sets $S$ |
| Add/remove to replace | $\varepsilon \to 2\varepsilon$ | Only add/remove hides participation |
| Advantage and posterior | See @tbl-dpdef-epsilon | Odds move by at most a factor $e^\varepsilon$ |
| Attack cap | $\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta$ | Audit: $\varepsilon \ge \ln((\mathrm{TPR} - \delta)/\mathrm{FPR})$ |
| Choosing $\delta$ | $\delta \ll 1/n$, with $n$ people | $\delta = 1/n$ can expose someone |

: The results of this chapter. {#tbl-dpdef-cheat}

**Rules of thumb**

- Every claim must state $\varepsilon$, $\delta$ and the unit of privacy, with its neighbouring relation.
- When users contribute many rows, use user-level neighbours, not row-level DP.
- An $\varepsilon$ is never a probability of re-identification. It bounds how far any output can shift an attacker's odds.
- A $\delta$ above one over the number of people allows exposing someone.
- A correct inference about a population is not a leak in the DP sense.

## Further reading

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), ch. 2: the definition and what DP promises.
- Kairouz, Oh & Viswanath, "The Composition Theorem for Differential Privacy" (ICML 2015): the hypothesis-testing reading behind the attack cap.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/01-definition.md).
