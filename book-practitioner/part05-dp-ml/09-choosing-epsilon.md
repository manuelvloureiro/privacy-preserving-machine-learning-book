# Choosing ε

## Why this matters

Every model protected by differential privacy (DP) has a privacy budget $\varepsilon$. The differentially private version of stochastic gradient descent (SGD), [DP-SGD](02-dp-sgd.md), shows how this budget affects accuracy. This chapter relates $\varepsilon$ to risks for people in the data, providing a basis for choosing it rather than following convention.

Membership tests assess whether a record appeared in training. Their true positive rate (TPR) measures correctly identified members; their false positive rate (FPR) measures non-members incorrectly flagged. The guarantee $(\varepsilon,\delta)$ bounds TPR in terms of FPR ([The definition of differential privacy](../part03-dp-foundations/01-definition.md)). Each run also has a tighter trade-off curve of achievable rates. Compute that curve, evaluate it at the threat's FPR and compare it with accuracy at each budget.

## Justifying a choice {#sec-epsc-justify}

A defensible choice specifies what the budget protects before selecting its value.

1. **Unit and period.** Say whether one unit is a record, a user or a device, and whether neighbouring datasets differ by adding or removing one unit or by replacing one. Say which runs the budget covers, since every tuning run spends budget too ([Hyperparameter tuning](03-hyperparameter-tuning.md)).
2. **Threat.** Say whether the concern is membership, reconstruction or attribute inference, and what the attacker knows and sees. This sets the false positive rate at which you read the curve, a low one for an attacker who must rarely accuse a non-member.
3. **Whole guarantee.** Choose $\delta$ by the book's rule (@sec-epsc-delta), then compute the trade-off curve (@sec-epsc-curve) and read it there.
4. **Evidence.** The curve is a theorem about an algorithm, not about your code, so run an audit that could refute it ([Auditing private training](10-auditing.md)).
5. **Utility.** Put accuracy against $\varepsilon$ next to the risk, and pick the smallest budget past the bend, where accuracy stops improving. Draw it on public or proxy data, because private runs that choose $\varepsilon$ are a search whose cost the reported $\varepsilon$ leaves out.
6. **Comparison.** Only now compare with deployments (@sec-epsc-deployed) and with the tiers in the rules of thumb.

The Adult public census survey illustrates step 5. [Train a model with DP-SGD](13-train-dp-sgd.md) trains a small network on 30,000 people at five privacy budgets, each with $\delta=10^{-5}$. @tbl-epsc-adult compares test accuracy with the maximum membership-test TPR at FPR $10^{-3}$, computed from @sec-epsc-curve.

| $\varepsilon$ | Noise multiplier | Test accuracy | Largest TPR at FPR $10^{-3}$ |
| --- | --- | --- | --- |
| 0.25 | 8.125 | 0.793 | 0.0013 |
| 1 | 2.383 | 0.843 | 0.0023 |
| 2 | 1.431 | 0.845 | 0.0046 |
| 4 | 0.973 | 0.846 | 0.0123 |
| 8 | 0.722 | 0.846 | 0.0445 |
| Non-private | — | 0.847 | — |

: Accuracy and worst-case membership risk on Adult, one seed. {#tbl-epsc-adult}

Accuracy increases five points from $\varepsilon=0.25$ to 1, then remains within a few thousandths of the non-private result, comparable to one-seed variation. Risk increases from 0.0023 at $\varepsilon=1$ to 0.0445 at $\varepsilon=8$. For Adult, the utility curve bends near $\varepsilon=1$, making 1 or 2 defensible choices. Because these runs were scored on private rows, the table is illustrative only; choose budgets using public stand-in data. Harder tasks, larger models or smaller datasets may shift the bend to higher $\varepsilon$.

## The pair and the whole curve {#sec-epsc-curve}

A membership test says "member" or "non-member" about one target record. Its TPR is the chance that it says "member" when the target was in the training set, and its FPR the chance when it was not. Every test against an $(\varepsilon, \delta)$-DP mechanism satisfies
$$\mathrm{TPR} \le \min\big\{1,\ e^\varepsilon\,\mathrm{FPR} + \delta,\ 1 - e^{-\varepsilon}(1 - \delta - \mathrm{FPR})\big\}.$$ {#eq-epsc-pair}
The middle term is the one that binds at low FPR, and the last applies the same cap to the members a test misses. Attacks are judged at a low FPR, because an attack that rarely flags a non-member identifies people with confidence ([Membership inference](../part02-attacks/02-membership-inference.md)). Yet at $\varepsilon = 8$ and an FPR of $10^{-3}$, the pair caps the TPR at 0.9997, which constrains almost nothing.

A real training run satisfies DP for a whole range of pairs, one $\delta(\varepsilon)$ for each $\varepsilon$, because its accountant can trade $\varepsilon$ for $\delta$. Each pair gives a line, the middle term of @eq-epsc-pair, and every test must stay under all of these lines at once. The best test reaches the lowest line exactly, which is called the **trade-off curve**:
$$\mathrm{TPR}^*(\mathrm{FPR}) = \min_{\varepsilon}\big(e^\varepsilon\,\mathrm{FPR} + \delta(\varepsilon)\big).$$ {#eq-epsc-duality}
For DP-SGD, a privacy loss distribution (PLD) accountant computes this curve. It can round the privacy loss up or down, and the two curves it gets bracket the true one ([Numerical accounting](../part04-privacy-accounting/06-numerical-accounting.md)).

@tbl-epsc-curve compares the pair with the curve for two of the Adult runs of @tbl-epsc-adult, each sampled at rate 1/30 for 300 steps.

| $\varepsilon$ | Noise multiplier | Pair, FPR $10^{-3}$ | Curve, FPR $10^{-3}$ | Curve, FPR $10^{-2}$ |
| --- | --- | --- | --- | --- |
| 4 | 0.973 | 0.05461 | [0.01221, 0.01234] | [0.06447, 0.06515] |
| 8 | 0.722 | 0.9997 | [0.04414, 0.04452] | [0.155, 0.1563] |

: Largest TPR of a membership test against two Adult runs at $\delta = 10^{-5}$, from the pair and from the curve. {#tbl-epsc-curve}

So a budget of 8 means little as a bare pair, while its curve caps the TPR at 0.0445 at an FPR of $10^{-3}$.

To compute the curve with Google's `dp_accounting`, build the run's PLD, ask it for $\delta$ on a grid of $\varepsilon$, and take the lowest line. The snippet's first print is the upper end of a bracket in @tbl-epsc-curve, and its second gives the last column of @tbl-epsc-adult.

```python
import numpy as np
from dp_accounting.pld import privacy_loss_distribution as pld

EPS_GRID = np.linspace(0.0, 20.0, 2001)


def tpr_at(sigma: float, q: float, steps: int, fpr: float) -> float:
    run = pld.from_gaussian_mechanism(sigma, sampling_prob=q).self_compose(steps)
    deltas = np.array(run.get_delta_for_epsilon(EPS_GRID))
    return float(np.min(np.exp(EPS_GRID) * fpr + deltas))


print(f"{tpr_at(0.722, 1 / 30, 300, 1e-2):.4f}")  # 0.1563
adult = [tpr_at(sigma, 1 / 30, 300, 1e-3) for sigma in (8.125, 2.383, 1.431, 0.973, 0.722)]
print(" ".join(f"{tpr:.4f}" for tpr in adult))  # 0.0013 0.0023 0.0046 0.0123 0.0445
```

The library's default PLD rounds the loss up, so its curve is a valid upper bound. A Rényi differential privacy (RDP) accountant gives a valid but looser curve. The Gaussian differential privacy (GDP) approximation is quick but not a bound, so use it to explain and the PLD to certify.

The curve is a worst case, for an attacker who sees every intermediate model, so a typical record attacked through the final model alone is much safer. The curve bounds reconstruction too, since a guess among $m$ equally likely records is a membership test at an FPR of $1/m$ ([Reconstruction attacks](../part02-attacks/09-reconstruction.md)).


## What an attacker can believe {#sec-epsc-bayes}

Most people ask how sure an attacker can become that a person was in the data. Write $p$ for the attacker's **prior**, its probability of "member" before it sees the model. A privacy loss of at most $\varepsilon$ means that no output is more than $e^\varepsilon$ times as likely with the person's record as without it. So the attacker's belief afterwards is at most
$$\Pr[\text{member} \mid \text{output}] \le \frac{p\,e^\varepsilon}{p\,e^\varepsilon + 1 - p}.$$ {#eq-epsc-posterior}
So DP bounds the change of belief, not the belief itself.

For example, an attacker who starts at 1% can end at 35.55% when $\varepsilon = 4$. That is for the worst output, and in DP-SGD the privacy loss of the released model is usually far below $\varepsilon$.

Base rates matter as much as the loss. An attacker who accuses every person its test flags as a member is right with probability
$$\Pr[\text{member} \mid \text{flag}] = \frac{p\,\mathrm{TPR}}{p\,\mathrm{TPR} + (1 - p)\,\mathrm{FPR}}.$$ {#eq-epsc-precision}
Take the Adult run at $\varepsilon = 8$ and an attacker with a 1% prior who flags at an FPR of $10^{-2}$. The curve caps its TPR there at 0.1563 (@tbl-epsc-curve), so at most 0.1363 of its flags are right, and more than six out of seven are false. An insider with a prior of 0.5 would be right far more often, so a statement about beliefs must name the attacker.

## How small δ must be {#sec-epsc-delta}

For $n$ privacy units, the book's rule sets $\delta \le 1/n^{1.1}$, rounded down to a power of ten ([How small δ must be](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta)). The units are people when one person has many records ([User-level differential privacy](07-user-level.md)). For Adult's 30,000 people it gives $10^{-5}$. That is not far below $1/n$, as always when $n$ is just above a power of ten. NIST SP 800-226, the US guidelines for evaluating DP claims, calls a $\delta$ above $10^{-5}$ suspicious unless justified.

For DP-SGD a smaller $\delta$ is cheap. Take the Adult run at $\varepsilon = 4$. When $\delta$ falls to $10^{-8}$, the PLD accountant puts it at only 5.763. So report $\varepsilon$ at a second, smaller $\delta$ too, and never quote a $\delta$ as large as one over the number of units.

## Comparing with deployed values {#sec-epsc-deployed}

"Industry uses $\varepsilon$ between 1 and 10" hides what every $\varepsilon$ needs, its **privacy unit** and the **period** over which it is spent. The Census Bureau's 2020 redistricting data used zero-concentrated differential privacy (zCDP), which states the guarantee as one number, written $\rho$, that converts to $(\varepsilon, \delta)$ pairs. Its unit was one person's record under the replace-one relation, and persons got $\rho = 2.56$, reported as $\varepsilon = 17.14$ at $\delta = 10^{-10}$.

So before you compare, convert both budgets to the same unit, neighbouring relation and $\delta$. A zCDP parameter $\rho$ converts to a pair by
$$\varepsilon = \rho + 2\sqrt{\rho\ln(1/\delta)}.$$ {#eq-epsc-zcdp}
For the Census figures this formula gives 17.92, and the Bureau's 17.14 comes from a tighter conversion. Since a replacement is a removal followed by an addition, an add/remove guarantee $(\varepsilon, \delta)$ implies only the replace-one guarantee $(2\varepsilon, (1 + e^\varepsilon)\delta)$ ([The definition of differential privacy](../part03-dp-foundations/01-definition.md)). And a budget stated per day adds up over time by composition, to 365 a year at $\varepsilon = 1$ per day.

So it would be wrong to call the Adult run at $\varepsilon = 8$ "less than half the Census budget". At the Census's $\delta = 10^{-10}$ it is at 13.812, and under the Census's replace-one relation that can double.

## Cheat sheet

| Question | Tool | Notes |
| --- | --- | --- |
| What does DP-SGD allow? | The trade-off curve, @eq-epsc-duality | Certify with the PLD, not with GDP |
| How sure can an attacker become? | @eq-epsc-posterior and @eq-epsc-precision | Name the prior; base rates dominate |
| How small a $\delta$? | The book's rule, @sec-epsc-delta | $n$ counts privacy units |
| Comparing with a deployment? | @eq-epsc-zcdp; replace-one doubles $\varepsilon$ | Same unit and $\delta$ |

: Which tool answers which question when choosing $\varepsilon$. {#tbl-epsc-cheat}

**Rules of thumb**

- A budget of at most 1 is strong for any mechanism. A budget between 1 and 10 needs DP-SGD's curve as justification, and a budget above 10 needs the curve and an audit (Ponomareva et al. 2023).
- Never quote $\varepsilon$ without its unit, $\delta$ and period.

## References

- Near, Darais, Lefkovitz & Howarth, *Guidelines for Evaluating Differential Privacy Guarantees* (NIST SP 800-226, 2025): privacy units, $\delta$, and the hazards of large budgets.
- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023): tiers and reporting.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/09-choosing-epsilon.md).
