# The definition of differential privacy

## Why this matters

Attacks in [Threat models and attacks](../part02-attacks/00-introduction.md) compare outputs from datasets that differ in one person. **Differential privacy (DP)** bounds how much the output distribution changes between these datasets, including for tests not yet developed. DP is a property of the algorithm, not the data, so it can be checked before data access.

## Neighbouring datasets {#sec-dpdef-neighbours}

The **neighbouring relation** defines what it means for datasets to differ in one record. Under **add/remove**, neighbours differ by one record, so the guarantee hides whether a person is included. Under **replace**, neighbours have the same size but one record has a different value, so the guarantee hides the record's value. A replacement can be represented as a removal followed by an addition. Each step costs privacy level $\varepsilon$ from @sec-dpdef-definition, so
$$\varepsilon\text{-DP under add/remove} \implies 2\varepsilon\text{-DP under replace}.$$ {#eq-dpdef-replace}
The reverse fails, because replace never compares datasets of different sizes.

Specify the **unit of privacy**, the entity represented by one record. At the **example level**, the unit is one training record or event. At the **user level**, it is all data from one person. A user with 200 messages is a group of 200 records under example-level DP, with a guarantee that scales to $200\varepsilon$ ([Post-processing, group privacy and composition](02-properties.md)). Confirm that the stated unit matches the intended protection. The guarantee must hold for every record allowed by the data format, including an age of 900 that could strongly affect an average. Private algorithms therefore clip each input to a fixed range before adding noise.

## The definition {#sec-dpdef-definition}

A **mechanism** $M$ is a randomized algorithm that takes a dataset and returns an output. Write $D$ for a dataset and $D'$ for one of its neighbours. The definition asks that no set of outputs becomes more than a factor $e^\varepsilon$ more likely when one record changes, up to a small additive slack $\delta$. Here $\varepsilon$ is not negative. A mechanism is **$(\varepsilon, \delta)$-differentially private** if every pair of neighbours and every set $S$ of outputs satisfy
$$\Pr[M(D) \in S] \le e^\varepsilon\,\Pr[M(D') \in S] + \delta.$$ {#eq-dpdef-def}
The probabilities come only from the randomness of the mechanism, and the data stay fixed.

When $\delta = 0$, the guarantee is **pure DP**; otherwise, it is **approximate DP**. A deterministic mechanism that returns different answers on neighbouring datasets assigns probability 1 to an answer on one dataset and 0 on the other. No finite factor $e^\varepsilon$ can bound this change. Rounding, aggregation and suppression of small cells therefore provide no guarantee by themselves.

## The privacy loss and δ {#sec-dpdef-divergences}

For output $y$, let $p_{M(D)}(y)$ be its probability or density when the mechanism runs on $D$. The **privacy loss** of $y$ is the log-likelihood ratio between $D$ and $D'$:
$$L(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)}.$$ {#eq-dpdef-loss}
Independent outputs multiply their likelihood ratios, so their privacy losses add. Part 4's accountants use this property. Pure $\varepsilon$-DP holds exactly when every output has privacy loss between $-\varepsilon$ and $\varepsilon$. Laplace noise meets this condition; Gaussian noise has unbounded privacy loss in the tail ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)).

Under approximate DP, $\delta$ bounds the probability mass where privacy loss exceeds $\varepsilon$. It counts only the probability on $D$ above $e^\varepsilon$ times the corresponding probability on $D'$. Thus, $\delta$ is not the probability that privacy loss exceeds $\varepsilon$; the exact quantity is the **hockey-stick divergence** ([Divergences](../part01-prerequisites/03-divergences.md#sec-div-hockey)). Each $\varepsilon$ has a smallest corresponding $\delta$, and the function $\delta(\varepsilon)$ is the mechanism's **privacy curve**. Report the noise parameters with the guarantee so that others can reproduce the curve ([Numerical accounting](../part04-privacy-accounting/06-numerical-accounting.md)).

## Randomized response {#sec-dpdef-rr}

In **randomized response**, each survey respondent tells the truth with probability $t$ and otherwise lies. A respondent who answers truthfully on the first coin toss and uses a second toss on tails answers truthfully with probability $\frac34$. The true answer is then three times as likely as the false answer, giving pure DP at $\varepsilon = \ln 3$. In general, randomized response at level $\varepsilon$ tells the truth with probability
$$t = \frac{e^\varepsilon}{1 + e^\varepsilon}.$$

With independent randomization, the guarantee is $\varepsilon$ per person for any number of respondents. It holds only under replace: under add/remove, the number of reports reveals who participated. Let $\bar y$ be the fraction of "yes" reports among $n$ people. Correcting for false answers gives an estimate of the true fraction $\pi$ and its standard deviation:
$$\hat\pi = \frac{\bar y - (1 - t)}{2t - 1}, \qquad \operatorname{sd}(\hat\pi) = \frac{e^{\varepsilon/2}}{(e^\varepsilon - 1)\sqrt n}.$$ {#eq-dpdef-rr-estimate}
For 1,000 people, the two-coin scheme has a standard deviation of 0.0274. When the true fraction is 0.3, that is about twice the sampling error of 0.0145 that the survey already has. This increase is the cost of the **local** model, in which no party sees a true answer ([Local, shuffle and distributed trust models](09-trust-models.md)).

OpenDP builds the mechanism from its truth probability, 0.75 for the two coins, and `map` turns one changed answer into its $\varepsilon$, $\ln 3$. The `contrib` feature turns on mechanisms that are not yet formally vetted.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
rr = dp.m.make_randomized_response_bool(prob=0.75)
print(f"{rr.map(1):.4f}")  # 1.0986
```

## What ε means for an attacker {#sec-dpdef-testing}

A membership attacker observes the output and predicts whether the target was included in the data ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). Its **false positive rate (FPR)** is the fraction of non-members it flags; its **true positive rate (TPR)** is the fraction of members it flags. The flagged outputs form a set bounded by the definition, so under add/remove every attack satisfies
$$\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta.$$ {#eq-dpdef-testing}

This form matches Part 2's evaluation of attacks by TPR at low FPR. For $\varepsilon = 1$ and $\delta = 10^{-5}$, the default in @sec-dpdef-small-delta for 10,000 people, an attack at FPR $10^{-3}$ has TPR at most $e \cdot 10^{-3} + 10^{-5} = 0.002728$. As FPR approaches 0, the bound depends almost entirely on $\delta$.

Rearranging the cap gives a lower bound on $\varepsilon$ from a measured attack:
$$\varepsilon \ge \ln\frac{\mathrm{TPR} - \delta}{\mathrm{FPR}}.$$
Suppose a model is claimed to be $(2, 10^{-5})$-DP, and an attack has TPR 0.05 at FPR 0.001. If these rates were exact, then $\varepsilon$ would be at least $\ln 50 = 3.91$, contradicting the claim. A real audit uses confidence bounds on the rates and a **canary**, a record designed to be easy to detect ([Auditing](../part05-dp-ml/10-auditing.md)).

Two summaries aid interpretation. The **membership advantage** is TPR minus FPR, the attack's improvement over random guessing. Under $(\varepsilon, \delta)$-DP, it is at most
$$\mathrm{TPR} - \mathrm{FPR} \le \frac{e^\varepsilon - 1 + 2\delta}{e^\varepsilon + 1}.$$ {#eq-dpdef-advantage}

The second summary bounds posterior belief. Under pure DP, one output can multiply the adversary's odds between two neighbouring datasets by at most $e^\varepsilon$. A prior probability $p$ that the data are $D$ can therefore rise at most to
$$\Pr[D \mid y] \le \frac{e^\varepsilon p}{e^\varepsilon p + 1 - p}.$$ {#eq-dpdef-posterior}
@tbl-dpdef-epsilon turns a claimed $\varepsilon$ into both numbers.

| $\varepsilon$ | Advantage bound | Posterior from a prior of 0.01 | Posterior from a prior of 0.5 |
| --- | --- | --- | --- |
| 0.1 | 0.0500 | 0.0110 | 0.5250 |
| 1 | 0.4621 | 0.0267 | 0.7311 |
| 3 | 0.9051 | 0.1687 | 0.9526 |
| 8 | 0.9993 | 0.9679 | 0.9997 |

: What a pure $\varepsilon$-DP guarantee promises against the strongest attacker. {#tbl-dpdef-epsilon}

An $\varepsilon$ at most 1 gives a strong bound for any mechanism: a 1% prior rises to less than 3%. For $\varepsilon$ between 1 and 10, justify the claim with the mechanism's privacy curve. Above 10, the bounds provide little protection, so include an audit ([Choosing ε](../part05-dp-ml/09-choosing-epsilon.md)). For statistics, specify acceptable error and solve for $\varepsilon$ ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md#sec-lg-laplace)). The table assumes pure DP; for an $(\varepsilon, \delta)$ guarantee, report both the advantage bound and TPR cap, which include $\delta$.

## How small δ must be {#sec-dpdef-small-delta}

A small $\delta$ has little effect on the advantage bound, but an average can hide severe harm to a few people. Consider a mechanism that independently publishes each record in full with probability $\delta$. It is $(0, \delta)$-DP under add/remove because a record changes the output only when its own random draw publishes it. Across $n$ records, the probability of publishing at least one is
$$1 - (1 - \delta)^n \approx 1 - e^{-n\delta}.$$
With 10,000 records and $\delta = 1/n$, that is $1 - e^{-1} = 0.6321$.

Treat $\delta$ as a per-person budget for catastrophic failures such as full record publication. Following Ponomareva et al. (2023), this book's default keeps $n\delta$, the expected number of people exposed this way, well below 1. Let $n$ count people, not records, and choose
$$\delta \le \frac{1}{n^{1.1}},$$
rounded down to a power of ten. For 10,000 people, the rule gives $10^{-5}$. The publishing mechanism then has $n\delta = 0.1$, and it exposes someone with probability only 0.0952.

## What differential privacy does not protect {#sec-dpdef-limits}

DP compares datasets with and without a record, so it does not prevent conclusions that hold in both datasets. A study that never used Alice's record may still show that smokers have a high cancer rate. This can affect beliefs about Alice, but the study is 0-DP with respect to her. A harm violates DP only if the inference would materially change without her record. Correlated records create a second limitation: a household of five whose members' records always change together is a group of five unless the household is the privacy unit. Implementation errors create a third limitation ([Discrete and floating-point-safe mechanisms](08-discrete-mechanisms.md)).

## Writing the guarantee down {#sec-dpdef-statement}

A reviewer can check only a fully specified claim. State the privacy unit and neighbouring relation, and report $\varepsilon$ and $\delta$ for the whole release, not per query. Also name the mechanism, library version, trust model and data treated as public; DP does not protect public data. This example uses the **central model**, in which a trusted curator holds the raw data and adds noise:

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
