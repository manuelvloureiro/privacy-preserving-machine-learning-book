# The definition of differential privacy

## Why this matters

Every attack in [Threat models and attacks](../part02-attacks/00-introduction.md) runs an algorithm on one of two datasets that differ in one person, and guesses from the output which one was used. **Differential privacy (DP)** makes the two outputs look alike against every test, including tests nobody has invented yet. It is a property of the algorithm, not of the data, so you can check it before any data are touched, and read any claim of it for what it protects and what it leaves out.

## Neighbouring datasets {#sec-dpdef-neighbours}

The first question about any claim is what "differ in one record" means, and the **neighbouring relation** answers it. Under **add/remove**, two datasets are neighbours when one has a record the other lacks, so the guarantee hides whether a person is in the data at all. Under **replace**, they have the same size and differ in the value of one record, so only what the record says is hidden. A replacement is a removal followed by an addition, two steps that each cost the privacy level $\varepsilon$ of @sec-dpdef-definition, so
$$\varepsilon\text{-DP under add/remove} \implies 2\varepsilon\text{-DP under replace}.$$ {#eq-dpdef-replace}
The reverse fails, because replace never compares datasets of different sizes.

You also choose the **unit of privacy**, what one record is. At the **example level**, it is one training record or one event, and at the **user level**, all the data of one person. Under example-level DP, a user who wrote 200 messages is a group of 200 records, protected only at $200\varepsilon$ ([Post-processing, group privacy and composition](02-properties.md)). So check that the claim protects the person. The guarantee must also hide every record the data format allows, even a person aged 900, whose record could move an average a long way. So private algorithms clip each input to a fixed range before they add noise.

## The definition {#sec-dpdef-definition}

Call a randomized algorithm that takes a dataset and returns an output a **mechanism**, written $M$, and write $D$ for a dataset and $D'$ for one of its neighbours. The definition asks that no set of outputs become more than a factor $e^\varepsilon$ more likely when one record changes, up to a small additive slack $\delta$. Here $\varepsilon$ is a non-negative privacy parameter. A mechanism is **$(\varepsilon, \delta)$-differentially private** if every pair of neighbours and every set $S$ of outputs satisfy
$$\Pr[M(D) \in S] \le e^\varepsilon\,\Pr[M(D') \in S] + \delta.$$ {#eq-dpdef-def}
The probabilities come only from the mechanism's own randomness, with the data held fixed.

When $\delta = 0$ the guarantee is called **pure DP**, and otherwise **approximate DP**. Either way, a private mechanism must be random. A deterministic one that answers differently on two neighbours gives that answer probability 1 on one and 0 on the other, which no factor $e^\varepsilon$ covers. So rounding, aggregating or hiding small table cells gives no guarantee by itself.

## The privacy loss and δ {#sec-dpdef-divergences}

For an output $y$, write $p_{M(D)}(y)$ for its probability, or its density, when the mechanism runs on $D$. The **privacy loss** of $y$ is the logarithm of how much more likely the output is on $D$ than on $D'$,
$$L(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)}.$$ {#eq-dpdef-loss}
Independent outputs multiply their likelihood ratios, so their losses add, which is why Part 4's accountants work with the loss. Pure $\varepsilon$-DP holds exactly when every output has a loss between $-\varepsilon$ and $\varepsilon$. Laplace noise meets that, while the loss of Gaussian noise grows without limit in the far tail ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)).

Under approximate DP, $\delta$ pays for the outputs whose loss exceeds $\varepsilon$. It counts only the part of each one's probability on $D$ above $e^\varepsilon$ times its probability on $D'$. So $\delta$ is not the chance that the loss exceeds $\varepsilon$, and the total is the **hockey-stick divergence** ([Divergences](../part01-prerequisites/03-divergences.md#sec-div-hockey)). Each $\varepsilon$ has its own smallest $\delta$, and the function $\delta(\varepsilon)$ is the mechanism's **privacy curve**, of which a quoted pair is one point. The noise parameters fix the whole curve, so report them with the pair and others can recompute it ([Numerical accounting](../part04-privacy-accounting/06-numerical-accounting.md)).

## Randomized response {#sec-dpdef-rr}

In **randomized response**, a survey design, each person tells the truth with some probability $t$ and lies otherwise, so a single answer proves nothing about them. A person who tosses a coin, answers truthfully on heads and answers by a second toss on tails tells the truth with probability $\frac34$. The true answer is then three times as likely as the false one, which gives pure DP at $\varepsilon = \ln 3$. In general, randomized response at level $\varepsilon$ tells the truth with probability
$$t = \frac{e^\varepsilon}{1 + e^\varepsilon}.$$

If the coins are independent, the guarantee is $\varepsilon$ per person however many people answer, but only under replace, since under add/remove the number of reports reveals who answered. Write $\bar y$ for the share of "yes" reports among the $n$ people. Undoing the lies with $t$ gives the estimate of the true fraction $\pi$ and its standard deviation,
$$\hat\pi = \frac{\bar y - (1 - t)}{2t - 1}, \qquad \operatorname{sd}(\hat\pi) = \frac{e^{\varepsilon/2}}{(e^\varepsilon - 1)\sqrt n}.$$ {#eq-dpdef-rr-estimate}
For 1,000 people the two-coin scheme has a standard deviation of 0.0274. When the true fraction is 0.3, that is about twice the sampling error of 0.0145 the survey already has. This doubling is the price of the **local** model, in which nobody sees a true answer ([Local, shuffle and distributed trust models](09-trust-models.md)).

OpenDP builds the mechanism from its truth probability, here the two-coin 0.75, and `map` turns one changed answer into its $\varepsilon$, $\ln 3$. The `contrib` feature switches on mechanisms not yet formally vetted.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
rr = dp.m.make_randomized_response_bool(prob=0.75)
print(f"{rr.map(1):.4f}")  # 1.0986
```

## What ε means for an attacker {#sec-dpdef-testing}

A membership attacker looks at the output and decides whether to accuse the target of being in the data, so it has two error rates ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). Its **false positive rate (FPR)** is how often it accuses a target who is out, and its **true positive rate (TPR)** how often it accuses one who is in. The outputs on which it accuses form a set, which the definition bounds, so under add/remove every attack satisfies
$$\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta.$$ {#eq-dpdef-testing}

This is the form to use, because Part 2 judged attacks by their TPR at a low FPR. Take $\varepsilon = 1$ and the $\delta = 10^{-5}$ that @sec-dpdef-small-delta recommends for 10,000 people. At an FPR of $10^{-3}$, no attack reaches a TPR above $e \cdot 10^{-3} + 10^{-5} = 0.002728$. As the FPR falls towards 0, the first term vanishes, so at a very low FPR the guarantee rests almost entirely on $\delta$.

Read backwards, the cap turns a measured attack into a lower bound on $\varepsilon$,
$$\varepsilon \ge \ln\frac{\mathrm{TPR} - \delta}{\mathrm{FPR}}.$$
Suppose a model is claimed to be $(2, 10^{-5})$-DP, and an attack reaches a TPR of 0.05 at an FPR of 0.001. If those rates were exact, $\varepsilon$ would be at least about $\ln 50 = 3.91$, and the claim false. A real audit puts confidence bounds on the rates and plants a **canary**, a record built to be as easy to detect as possible ([Auditing](../part05-dp-ml/10-auditing.md)).

Two summaries are easier to explain to a lawyer. The first, the **membership advantage**, is the TPR minus the FPR, how much better than a coin flip an attacker does. Under $(\varepsilon, \delta)$-DP it is at most
$$\mathrm{TPR} - \mathrm{FPR} \le \frac{e^\varepsilon - 1 + 2\delta}{e^\varepsilon + 1}.$$ {#eq-dpdef-advantage}

The second, the posterior bound, speaks about beliefs. Under pure DP, one output can multiply an adversary's odds between two neighbours by at most $e^\varepsilon$. So a prior belief $p$ that the data are $D$ rises at most to
$$\Pr[D \mid y] \le \frac{e^\varepsilon p}{e^\varepsilon p + 1 - p}.$$ {#eq-dpdef-posterior}
@tbl-dpdef-epsilon turns a claimed $\varepsilon$ into both numbers.

| $\varepsilon$ | Advantage bound | Posterior from a prior of 0.01 | Posterior from a prior of 0.5 |
| --- | --- | --- | --- |
| 0.1 | 0.0500 | 0.0110 | 0.5250 |
| 1 | 0.4621 | 0.0267 | 0.7311 |
| 3 | 0.9051 | 0.1687 | 0.9526 |
| 8 | 0.9993 | 0.9679 | 0.9997 |

: What a pure $\varepsilon$-DP guarantee promises against the strongest attacker. {#tbl-dpdef-epsilon}

So an $\varepsilon$ of at most 1 is strong for any mechanism, since a 1% prior rises to under 3%. Between 1 and 10 a claim needs the mechanism's privacy curve as justification, and above 10, where the bounds promise almost nothing, an audit as well ([Choosing ε](../part05-dp-ml/09-choosing-epsilon.md)). For statistics, start from the error you can accept and solve it for $\varepsilon$ ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md#sec-lg-laplace)). The table is for pure DP, so for an $(\varepsilon, \delta)$ claim quote the advantage bound and the cap on the TPR, which both include $\delta$.

## How small δ must be {#sec-dpdef-small-delta}

A small $\delta$ barely moves the advantage bound, but the advantage averages over everyone, so it can hide total harm to a few people. Take a mechanism that publishes each record in full, independently with probability $\delta$. Under add/remove it is $(0, \delta)$-DP, because a record changes the output only when its own coin publishes it. Yet over $n$ records it publishes at least one with probability
$$1 - (1 - \delta)^n \approx 1 - e^{-n\delta}.$$
With 10,000 records and $\delta = 1/n$, that is $1 - e^{-1} = 0.6321$.

So treat $\delta$ as a per-person budget for catastrophe, which may be spent on total failures like these. This book's default keeps $n\delta$, the expected number of people exposed this way, well below 1. Let $n$ count people rather than records, and choose
$$\delta \le \frac{1}{n^{1.1}},$$
rounded down to a power of ten. For 10,000 people the rule gives $10^{-5}$. The publishing mechanism then has $n\delta = 0.1$ and exposes someone with probability only 0.0952.

## What differential privacy does not protect {#sec-dpdef-limits}

DP compares a world with your record and a world without it, so anything that can be learned in both worlds is outside the guarantee. A study that never used Alice's record may show that smokers get cancer at a high rate, which changes what anyone believes about Alice, a smoker, yet the study is 0-DP with respect to her. So a reported harm is a DP violation only if the inference would differ materially without the person's record. Correlated records are a second gap: a household of five that always answers alike is a group of five, unless the household is the unit of privacy. The third gap is the implementation, which needs its own review ([Discrete and floating-point-safe mechanisms](08-discrete-mechanisms.md)).

## Writing the guarantee down {#sec-dpdef-statement}

A reviewer can check only what a claim writes down. So a guarantee names the unit of privacy with its neighbouring relation, and $\varepsilon$ and $\delta$ for the whole release rather than per query. It also names the mechanism, the library version, the trust model and what was treated as public, since public information is not protected. This example uses the **central model**, in which a trusted curator holds the raw data and adds the noise:

> This release satisfies $(1, 10^{-5})$-DP for adding or removing one customer and all their invoice lines, at most 100 per customer, under the central model, with Gaussian noise from OpenDP 0.16.0. The market list was fixed in advance.

[Release statistics with a library](12-release-statistics.md) and [Release and governance](../part05-dp-ml/12-release-governance.md) print such statements from the code that made the release.

## Code {#sec-dpdef-code}

This snippet checks two numbers quoted above.

```python
from math import exp, log

print(f"{(exp(3) - 1) / (exp(3) + 1):.4f}")  # 0.9051
print(f"{log((0.05 - 1e-5) / 1e-3):.2f}")  # 3.91
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| $(\varepsilon, \delta)$-DP | $\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta$ | All neighbours, all sets $S$ |
| Attack cap | $\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta$ | Audit: $\varepsilon \ge \ln((\mathrm{TPR} - \delta)/\mathrm{FPR})$ |
| Choosing $\delta$ | $\delta \le 1/n^{1.1}$, rounded down to a power of ten | $n$ counts people, not records |

: The results of this chapter. {#tbl-dpdef-cheat}

**Rules of thumb**

- When people contribute many rows, the unit of privacy is the person.
- An $\varepsilon$ is never a probability of re-identification. Under pure DP it bounds how far any output can shift an attacker's odds.


## References

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), ch. 2: the definition and what DP promises.
- Kairouz, Oh & Viswanath, "The Composition Theorem for Differential Privacy" (ICML 2015): the hypothesis-testing reading behind the attack cap.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/01-definition.md).
