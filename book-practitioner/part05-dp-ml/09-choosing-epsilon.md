# Choosing ε

## Why this matters

Every model trained with differential privacy (DP) ships with a privacy budget $\varepsilon$. The chapter on private stochastic gradient descent (SGD), [DP-SGD](02-dp-sgd.md), showed what each budget costs in accuracy, which is half of the choice. The other half is what it means for the people in the data. Without it, $\varepsilon$ gets chosen by habit ("everyone uses 8"), and it does not survive a reviewer who asks what the number promises.

An attacker who wants to know whether a record was used for training runs a membership test. The test can catch real members, its true positives, and it can wrongly flag non-members, its false positives. The budget $\varepsilon$, together with its small slack $\delta$ ([The definition of differential privacy](../part03-dp-foundations/01-definition.md)), caps the rate of true positives in terms of the rate of false positives. But each training run also has its own curve of the best rates an attacker can reach, and that curve usually lies far below the cap. Choosing $\varepsilon$ well means computing the curve, reading it at the false positive rate your threat calls for, and setting it beside the accuracy that each budget buys.

## Justifying a choice {#sec-epsc-justify}

A defensible choice says what the budget protects before it picks the number, in six steps.

1. **Unit and period.** Say whether one unit is a record, a user or a device, and whether neighbouring datasets differ by adding or removing one unit or by replacing one. Say which runs and searches the budget covers, since every run used to tune hyperparameters spends budget too ([Hyperparameter tuning](03-hyperparameter-tuning.md)).
2. **Threat.** Say whether the concern is membership, reconstruction or attribute inference, and what the attacker knows and sees. This decides the false positive rate at which you read the curve, a low one for an attacker who must rarely accuse a non-member.
3. **Whole guarantee.** Choose $\delta$ by the book's default (@sec-epsc-delta), then compute the trade-off curve (@sec-epsc-curve) and read it there.
4. **Evidence.** The curve is a theorem about an algorithm, not about your code, so run an audit that could refute it ([Auditing private training](10-auditing.md)).
5. **Utility.** Put accuracy against $\varepsilon$ next to the risk, and pick the smallest budget past the bend, where accuracy stops improving. Draw it on public or proxy data, because private runs that choose $\varepsilon$ are a search whose cost the reported $\varepsilon$ leaves out.
6. **Comparison.** Only now compare with deployments (@sec-epsc-deployed) and with the tiers in the rules of thumb at the end of this page.

Adult, a public census survey, shows step 5 on a real run. [Train a model with DP-SGD](13-train-dp-sgd.md) trains one small network on 30,000 of its people at five budgets, each with $\delta = 10^{-5}$ and with the noise multiplier that meets it. @tbl-epsc-adult sets the test accuracies beside a measure of risk, the largest true positive rate (TPR) that any membership test can reach at a false positive rate (FPR) of $10^{-3}$. The snippet in @sec-epsc-curve computes it.

| $\varepsilon$ | Noise multiplier | Test accuracy | Largest TPR at FPR $10^{-3}$ |
| --- | --- | --- | --- |
| 0.25 | 8.125 | 0.793 | 0.0013 |
| 1 | 2.383 | 0.843 | 0.0023 |
| 2 | 1.431 | 0.845 | 0.0046 |
| 4 | 0.973 | 0.846 | 0.0123 |
| 8 | 0.722 | 0.846 | 0.0445 |
| Non-private | — | 0.847 | — |

: Accuracy and worst-case membership risk on Adult, one seed. {#tbl-epsc-adult}

Accuracy rises by five points from $\varepsilon = 0.25$ to 1, then stays within a few thousandths of the non-private model, which on one seed is noise. The risk keeps growing, though, and faster with each doubling of the budget, from 0.0023 at $\varepsilon = 1$ to 0.0445 at $\varepsilon = 8$. So on Adult the bend sits at $\varepsilon = 1$, and a budget of 1 or 2 is the defensible choice. These runs trained and were scored on private rows, so here the table illustrates step 5; a real choice reads it from a public stand-in. [Release and governance](12-release-governance.md) releases the $\varepsilon = 4$ run to show the steps, and a reviewer would be right to ask why it spends more than accuracy needs. Expect the bend further right on a harder task, a larger model or a smaller dataset.

## The pair and the whole curve {#sec-epsc-curve}

A membership test says "member" or "non-member" about one target record. Its TPR is the chance that it says "member" when the target was in the training set, and its FPR the chance that it says so when the target was not. Every test against an $(\varepsilon, \delta)$-DP mechanism satisfies
$$\mathrm{TPR} \le \min\big\{1,\ e^\varepsilon\,\mathrm{FPR} + \delta,\ 1 - e^{-\varepsilon}(1 - \delta - \mathrm{FPR})\big\}.$$ {#eq-epsc-pair}
The middle term is the one that binds at low FPR, and the last applies the same cap to the members a test misses. Attacks are judged at a low FPR, because an attack that rarely flags a non-member identifies people with confidence ([Membership inference](../part02-attacks/02-membership-inference.md)). Yet at $\varepsilon = 8$ and an FPR of $10^{-3}$, the pair caps the TPR at 0.9997, which constrains almost nothing.

A real training run satisfies DP for a whole range of pairs, one $\delta(\varepsilon)$ for each $\varepsilon$, because its accountant can trade a larger $\varepsilon$ for a smaller $\delta$. Each pair gives a line, the middle term of @eq-epsc-pair, and every test must stay under all of these lines at once. The best test reaches the lowest line exactly, and that line is called the **trade-off curve**:
$$\mathrm{TPR}^*(\mathrm{FPR}) = \min_{\varepsilon}\big(e^\varepsilon\,\mathrm{FPR} + \delta(\varepsilon)\big).$$ {#eq-epsc-duality}
For DP-SGD, a privacy loss distribution (PLD) accountant computes this curve. It can round the privacy loss up or down, and the two curves it gets bracket the true one ([Numerical accounting](../part04-privacy-accounting/06-numerical-accounting.md)). @tbl-epsc-curve compares the pair with the curve for this part's running example from [DP-SGD](02-dp-sgd.md), a synthetic run sampled at rate 0.01 for 10,000 steps.

| $\varepsilon$ | Noise multiplier | Pair, FPR $10^{-3}$ | Curve, FPR $10^{-3}$ | Curve, FPR $10^{-2}$ |
| --- | --- | --- | --- | --- |
| 4 | 1.36 | 0.05461 | [0.01215, 0.01312] | [0.06739, 0.0724] |
| 8 | 0.917 | 0.9997 | [0.05555, 0.0589] | [0.1987, 0.2092] |

: Largest TPR of a membership test against the running example at $\delta = 10^{-5}$, from the pair and from the curve. {#tbl-epsc-curve}

So a budget of 8 means little as a bare pair, while its curve caps the TPR at 0.0589 at an FPR of $10^{-3}$.

With Google's `dp_accounting`, build the run's PLD, ask it for $\delta$ on a grid of $\varepsilon$, and take the lowest line. The snippet's first print, for the running example at $\varepsilon = 4$, falls inside the bracket of @tbl-epsc-curve. Its second uses Adult's sampling rate of 1/30 and 300 steps, and gives the last column of @tbl-epsc-adult.

```python
import numpy as np
from dp_accounting.pld import privacy_loss_distribution as pld

EPS_GRID = np.linspace(0.0, 20.0, 2001)


def tpr_at(sigma: float, q: float, steps: int, fpr: float) -> float:
    run = pld.from_gaussian_mechanism(sigma, sampling_prob=q).self_compose(steps)
    deltas = np.array(run.get_delta_for_epsilon(EPS_GRID))
    return float(np.min(np.exp(EPS_GRID) * fpr + deltas))


print(f"{tpr_at(1.36, 0.01, 10_000, 1e-3):.4f}")  # 0.0126
adult = [tpr_at(sigma, 1 / 30, 300, 1e-3) for sigma in (8.125, 2.383, 1.431, 0.973, 0.722)]
print(" ".join(f"{tpr:.4f}" for tpr in adult))  # 0.0013 0.0023 0.0046 0.0123 0.0445
```

The library's default PLD rounds the loss up, so its curve is a valid upper bound. A Rényi differential privacy (RDP) accountant gives a valid but looser curve. The Gaussian differential privacy (GDP) approximation is quick but not a bound, so use it to explain and the PLD to certify.

The curve is also a worst case. It describes a strong attacker who sees every intermediate model, and a typical record attacked through the final model alone is much safer. It also bounds reconstruction, since guessing which of $m$ equally likely records is the target is a membership test at an FPR of $1/m$ ([Reconstruction attacks](../part02-attacks/09-reconstruction.md)).

## What an attacker can believe {#sec-epsc-bayes}

Most people ask how sure an attacker can become that a given person was in the data. Write $p$ for the attacker's **prior**, its probability of "member" before it sees the model. Suppose no output is more than $e^\varepsilon$ times as likely with the person's record as without it, which is what a privacy loss of at most $\varepsilon$ means. Then the attacker's belief afterwards is at most
$$\Pr[\text{member} \mid \text{output}] \le \frac{p\,e^\varepsilon}{p\,e^\varepsilon + 1 - p}.$$ {#eq-epsc-posterior}
DP bounds the change of belief, not the belief itself. An attacker who starts at 1% can end at 35.55% when $\varepsilon = 4$. That is the worst output, though, and in DP-SGD the privacy loss of the model actually released is usually far below $\varepsilon$.

Base rates matter as much as the loss. An attacker who accuses every person its test flags as a member is right with probability
$$\Pr[\text{member} \mid \text{flag}] = \frac{p\,\mathrm{TPR}}{p\,\mathrm{TPR} + (1 - p)\,\mathrm{FPR}}.$$ {#eq-epsc-precision}
Take the running example's $\varepsilon = 8$ run and an attacker with a 1% prior who flags at an FPR of $10^{-2}$. The curve caps its TPR there at 0.2092 (@tbl-epsc-curve), so its flags are right with probability at most 0.1744, and five out of six are false. An insider with a prior of 0.5 would be right far more often, so a statement about beliefs must name the attacker.

## How small δ must be {#sec-epsc-delta}

A deliberately bad mechanism shows what $\delta$ permits. It publishes each of $n$ records in full, independently with probability $\delta$. It is $(0, \delta)$-DP, because adding or removing one record changes the output only when that record's own coin comes up. Yet it publishes at least one record with probability
$$1 - (1 - \delta)^n \approx 1 - e^{-n\delta}.$$ {#eq-epsc-release}
For 10,000 records at $\delta = 1/n$, that is 0.6321. So $n\delta$ must be below 1. Here $n$ counts privacy units, the thing one guarantee protects, which means people when one person has many records ([User-level differential privacy](07-user-level.md)).

This book uses one default throughout. Take $\delta \le 1/n^{1.1}$ and round it down to a power of ten, which keeps $n\delta$ below 1 for every $n$. It gives $10^{-5}$ for both of this page's runs, and the `delta_for` function of [Train a model with DP-SGD](13-train-dp-sgd.md) computes it. When $n$ is just above a power of ten, as with Adult's 30,000 people, the default is not far below $1/n$. That is one more reason to report a second, smaller $\delta$. NIST SP 800-226, the US guidelines for evaluating DP claims, reports a stricter common recommendation of $1/n^2$, and calls a $\delta$ above $10^{-5}$ suspicious unless justified.

For DP-SGD a smaller $\delta$ is cheap. The RDP accountant that set the running example's $\varepsilon = 4$ run puts it at only 5.209 when $\delta$ falls to $10^{-8}$. So report $\varepsilon$ at a second, smaller $\delta$ too, and never quote a $\delta$ as large as one over the number of units.

## Comparing with deployed values {#sec-epsc-deployed}

"Industry uses $\varepsilon$ between 1 and 10" hides what every $\varepsilon$ needs, its **privacy unit** and the **period** over which it is spent. The Census Bureau's 2020 redistricting data used zero-concentrated differential privacy (zCDP), which states the whole guarantee as one number, written $\rho$, that converts to $(\varepsilon, \delta)$ pairs. The unit was one person's record under the replace-one relation, and persons got $\rho = 2.56$, which the Bureau reported as $\varepsilon = 17.14$ at $\delta = 10^{-10}$. Google's Gboard keyboard models, trained in federated learning on users' phones, had per-device guarantees for a whole training run, from about 1 to 14 at the same $\delta$.

So convert both budgets to the same unit, neighbouring relation and $\delta$ before comparing. A zCDP parameter $\rho$ converts to a pair by
$$\varepsilon = \rho + 2\sqrt{\rho\ln(1/\delta)}.$$ {#eq-epsc-zcdp}
For the Census figures this formula gives 17.92. The Bureau's 17.14 comes from a tighter conversion. Since a replacement is a removal followed by an addition, an add/remove guarantee $(\varepsilon, \delta)$ implies only the replace-one guarantee $(2\varepsilon, (1 + e^\varepsilon)\delta)$ ([The definition of differential privacy](../part03-dp-foundations/01-definition.md)). And a budget stated per day or per report adds up over time by composition, to 365 a year at $\varepsilon = 1$ per day.

So it would be wrong to call the running example's $\varepsilon = 8$ run "less than half the Census budget". At the Census's $\delta = 10^{-10}$ it is at 11.627, and under the Census's replace-one relation that can double.

## Cheat sheet

| Question | Tool | Notes |
| --- | --- | --- |
| What does DP-SGD allow? | The trade-off curve, @eq-epsc-duality | Certify with the PLD, not with GDP |
| How sure can an attacker become? | @eq-epsc-posterior and @eq-epsc-precision | Name the prior; base rates dominate |
| How small a $\delta$? | $\delta \le 1/n^{1.1}$, rounded down to a power of ten | $n$ counts privacy units |
| Comparing with a deployment? | @eq-epsc-zcdp; replace-one doubles $\varepsilon$ | Same unit and $\delta$ |

: Which tool answers which question when choosing $\varepsilon$. {#tbl-epsc-cheat}

**Rules of thumb**

- A budget of at most 1 is strong for any mechanism. A budget between 1 and 10 needs DP-SGD's curve as justification, and a budget above 10 needs the curve and an audit (Ponomareva et al. 2023).
- Never quote $\varepsilon$ without its unit, $\delta$ and period.

## Further reading

- Near, Darais, Lefkovitz & Howarth, *Guidelines for Evaluating Differential Privacy Guarantees* (NIST SP 800-226, 2025): privacy units, $\delta$, and the hazards of large budgets.
- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023): tiers and reporting.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/09-choosing-epsilon.md).
