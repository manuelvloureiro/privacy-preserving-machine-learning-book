# Choosing ε

## Why this matters

Every model trained with differential privacy (DP) ships with a privacy budget $\varepsilon$. The chapter on private stochastic gradient descent (SGD), [DP-SGD](02-dp-sgd.md), measured what each budget costs in accuracy. That is half of the choice, and the other half is what each budget means for the people in the data. Without it, $\varepsilon$ gets chosen by habit ("everyone uses 8"), and the choice does not survive a reviewer who asks what the number promises.

The answer reads privacy as a test. An attacker who wants to know whether a record was used for training runs a membership test, and every test has two error rates. The budget $\varepsilon$, together with its small slack $\delta$, caps one rate in terms of the other. But the training algorithm has its own curve of the best rates an attacker can reach, which is usually much lower. Choosing $\varepsilon$ well means computing that curve, reading it where your threat lives, and writing down what it says.

## Justifying a choice {#sec-epsc-justify}

A defensible choice fixes the claim before the number, in six steps.

1. **Unit and neighbours.** Say whether the privacy unit is a record, a user or a device, and whether neighbours add or remove a unit or replace it. Say over what period the budget is spent, counting every run and search that touched the data ([Hyperparameter tuning](03-hyperparameter-tuning.md)).
2. **Threat.** Say whether the concern is membership, reconstruction or attribute inference, what the attacker knows beforehand, and what it sees, because this decides where you read the curve.
3. **Whole guarantee.** Compute the trade-off curve with a tight accountant (@sec-epsc-curve), choose $\delta$ small (@sec-epsc-delta), and read the curve at the threat's operating points.
4. **Evidence.** The curve is a theorem about an algorithm, not about your code. So run an empirical audit whose lower bound is consistent with the claim ([Auditing private training](10-auditing.md)).
5. **Utility.** Plot accuracy against $\varepsilon$ next to the risk, and pick the smallest budget past the bend where accuracy stops rising steeply.
6. **Comparison.** Only now compare with deployments (@sec-epsc-deployed) and with the tiers of Ponomareva et al. (2023) in the rules of thumb.

On the running example, measure the risk in step 5 as the largest true positive rate (TPR) of a membership test at a false positive rate (FPR) of 1%. The first two doublings of the budget, from 1 to 4, raise test accuracy by 2.66 and then 1.07 points, while that TPR stays below 0.0724. But the next doubling gains only 0.56 points, and it nearly triples the TPR, to 0.2092. So past $\varepsilon = 4$, each point of accuracy is bought with a much larger worst-case risk.

A team that chooses the $\varepsilon = 4$ run can then write the statement in @tbl-epsc-statement, a template for any model card.

| Item | Statement |
| --- | --- |
| Training | DP-SGD with Poisson sampling rate 0.01, noise multiplier 1.36, clipping norm 0.5 and 10,000 steps, on 10,000 records, one record per person |
| Unit and period | One record under add/remove neighbours; this one run, and no other run touched the training data |
| Assumptions | Poisson sampling and per-example clipping as the accountant assumes, checked by an audit |
| Guarantee | The Rényi differential privacy (RDP) accountant gives $(3.993, 10^{-5})$-DP and $(5.209, 10^{-8})$-DP; a privacy loss distribution (PLD) accountant gives $\varepsilon \le 3.729$ at $\delta = 10^{-5}$ |
| Membership | Against an attacker who knows every other record and sees every iterate, any membership test has TPR at most 0.01312 at FPR $10^{-3}$ and at most 0.0724 at FPR $10^{-2}$ |
| Utility | Test accuracy 0.9548, against 0.9610 without privacy |

: A privacy statement for the running example's $\varepsilon = 4$ run, following the six steps. {#tbl-epsc-statement}

## The pair and the whole curve {#sec-epsc-curve}

A membership test says "member" or "non-member" about one target record. Its TPR is the chance that it says "member" when the target was in the training set. Its FPR is the chance that it says so when the target was not. Every test against an $(\varepsilon, \delta)$-DP mechanism satisfies
$$\mathrm{TPR} \le \min\big\{1,\ e^\varepsilon\,\mathrm{FPR} + \delta,\ 1 - e^{-\varepsilon}(1 - \delta - \mathrm{FPR})\big\}.$$ {#eq-epsc-pair}
Attacks are judged at a low FPR, because an attack that rarely flags a non-member identifies people with confidence ([Membership inference](../part02-attacks/02-membership-inference.md)). Yet at $\varepsilon = 8$ the pair caps the TPR at 0.9997, which constrains almost nothing.

A real training run satisfies DP for a whole range of pairs, with one $\delta(\varepsilon)$ for each $\varepsilon$. The best TPR an attacker can reach at a given FPR is the lowest of all their lines, which is called the **trade-off curve**:
$$\mathrm{TPR}^*(\mathrm{FPR}) = \min_{\varepsilon}\big(e^\varepsilon\,\mathrm{FPR} + \delta(\varepsilon)\big).$$ {#eq-epsc-duality}
For DP-SGD, a PLD accountant computes this curve, and its two roundings bracket the truth ([Numerical accounting](../part04-privacy-accounting/06-numerical-accounting.md)). @tbl-epsc-curve compares the answers for the running example, sampled at rate 0.01 for 10,000 steps.

| $\varepsilon$ | Noise multiplier | Pair, FPR $10^{-3}$ | Exact, FPR $10^{-3}$ | Exact, FPR $10^{-2}$ |
| --- | --- | --- | --- | --- |
| 4 | 1.36 | 0.05461 | [0.01215, 0.01312] | [0.06739, 0.0724] |
| 8 | 0.917 | 0.9997 | [0.05555, 0.0589] | [0.1987, 0.2092] |
| 16 | 0.695 | 1 | [0.2714, 0.2808] | [0.5447, 0.5588] |

: The largest TPR of a membership test against the running example's DP-SGD at $\delta = 10^{-5}$, from the pair alone and exactly from the PLD bracket. {#tbl-epsc-curve}

At an FPR of $10^{-3}$, the curve of the $\varepsilon = 8$ run allows a TPR of at most 0.0589, far below the pair's cap. So a budget of 8 means little as a bare pair, but something real for DP-SGD once you report its curve.

Use the PLD bracket to certify, while an RDP accountant also gives a valid but looser curve. The Gaussian differential privacy (GDP) approximation in the snippet below is a quick check, not a bound, since at $\varepsilon = 16$ it reports 0.3234, above the exact value. With Google's `dp_accounting`, build the run's PLD and read the curve through @eq-epsc-duality, passing a grid of $\varepsilon$ to `get_delta_for_epsilon`:

```text
from dp_accounting.pld import privacy_loss_distribution as pld
import numpy as np

run = pld.from_gaussian_mechanism(1.36, sampling_prob=0.01).self_compose(10_000)
eps = np.linspace(0.0, 20.0, 2001)
tpr = np.min(np.exp(eps) * 1e-3 + run.get_delta_for_epsilon(eps))  # TPR at FPR 1e-3
```

For $\varepsilon$ alone, `dp_accounting.pld.PLDAccountant` and Opacus's default `accountant="prv"` give the same kind of tight answer ([Numerical accounting](../part04-privacy-accounting/06-numerical-accounting.md)). The curve describes a strong attacker who sees every intermediate model, and a typical record attacked through the final model is much safer.

The same curve bounds reconstruction ([Reconstruction attacks](../part02-attacks/09-reconstruction.md)). Suppose an attacker knows the target is one of a number $m$ of equally likely records. It then succeeds with probability at most the curve read at an FPR of $1/m$, so at most 0.0589 for a thousand candidates at $\varepsilon = 8$.

## What an attacker can believe {#sec-epsc-bayes}

Most people ask how sure someone can become that they were in the data. Write $p$ for the attacker's **prior**, its probability of "member" before it sees the model. If no output has a privacy loss above $\varepsilon$, its belief afterwards is at most
$$\Pr[\text{member} \mid \text{output}] \le \frac{p\,e^\varepsilon}{p\,e^\varepsilon + 1 - p}.$$ {#eq-epsc-posterior}
So DP bounds the change of belief, not the belief itself. At $\varepsilon = 4$, an attacker who starts at 1% can end at 35.55%. That is the worst output, though, and in DP-SGD the realized loss is usually far below $\varepsilon$.

Base rates matter as much as the loss. An attacker who acts on a flag is right with probability
$$\Pr[\text{member} \mid \text{flag}] = \frac{p\,\mathrm{TPR}}{p\,\mathrm{TPR} + (1 - p)\,\mathrm{FPR}}.$$ {#eq-epsc-precision}
Suppose an attacker with a 1% prior flags at an FPR of $10^{-2}$, against the $\varepsilon = 8$ run. Its flags are then right with probability at most 0.1744, so five out of six are false. A statement about beliefs must therefore name the attacker, because an insider with a prior of 0.5 starts much closer to certainty.

## How small δ must be {#sec-epsc-delta}

A deliberately bad mechanism shows what $\delta$ permits. It publishes each of $n$ records in full, independently with probability $\delta$, and it is $(0, \delta)$-DP. Yet it publishes at least one record with probability
$$1 - (1 - \delta)^n \approx 1 - e^{-n\delta}.$$ {#eq-epsc-release}
For the running example's 10,000 records at $\delta = 1/n$, that is 0.6321. So $n\delta$ must be much smaller than 1. Here $n$ counts privacy units, which means people when one person has many records ([User-level differential privacy](07-user-level.md)). NIST SP 800-226 reports the common recommendation that $\delta$ be at most $1/n^2$, and calls values above $10^{-5}$ suspicious.

For DP-SGD a smaller $\delta$ is cheap, because its privacy loss has a light tail. Moving the $\varepsilon = 4$ run to $\delta = 1/n^2$ raises its budget only to 5.209.

## Comparing with deployed values {#sec-epsc-deployed}

"Industry uses $\varepsilon$ between 1 and 10" hides what every $\varepsilon$ needs, its **privacy unit** and the **period** over which it is spent. The Census Bureau's 2020 redistricting data used zero-concentrated differential privacy (zCDP), whose guarantee is a single number $\rho$. It spent $\rho = 2.56$ on persons, reported as $\varepsilon = 17.14$ at $\delta = 10^{-10}$, with one person's record replaced as the unit. Google's Gboard language models, trained privately in federated learning, had device-level guarantees for a whole training run, from about 1 to 14 at the same $\delta$. Apple's local DP is stated per donation, with QuickType at $\varepsilon = 8$ and two donations a day, so up to 16 a day.

So convert both budgets to the same unit, neighbouring relation and $\delta$ before comparing. A zCDP parameter $\rho$ converts to a pair by
$$\varepsilon = \rho + 2\sqrt{\rho\ln(1/\delta)}.$$ {#eq-epsc-zcdp}
An add/remove guarantee $(\varepsilon, \delta)$ implies only the replace-one guarantee $(2\varepsilon, (1 + e^\varepsilon)\delta)$. A budget per day adds up over time, to 365 a year at $\varepsilon = 1$ per day.

So the running example's $\varepsilon = 8$ run is not "less than half the Census budget", because at $\delta = 10^{-10}$ it is at 11.627, and a replacement can double that.

## Code

This snippet turns a claimed pair into the TPR it allows, and approximates DP-SGD's curve with GDP.

```python
from math import exp, sqrt
from statistics import NormalDist


def pair_tpr_cap(eps: float, delta: float, fpr: float) -> float:
    """The largest TPR any test can reach at a given FPR, from the pair (eps, delta) alone."""
    return min(1.0, exp(eps) * fpr + delta, 1 - exp(-eps) * (1 - delta - fpr))


def gdp_tpr(q: float, steps: int, sigma: float, fpr: float) -> float:
    """GDP estimate of DP-SGD's TPR at an FPR (sampling rate q, noise multiplier sigma); not a bound."""
    mu = q * sqrt(steps * (exp(1 / sigma**2) - 1))
    return NormalDist().cdf(NormalDist().inv_cdf(fpr) + mu)


print(f"{pair_tpr_cap(4, 1e-5, 1e-3):.5f}")  # 0.05461
print(f"{gdp_tpr(0.01, 10_000, 1.36, 1e-3):.5f}")  # 0.01244
```

The full edition's script computes the exact curves and every other number in this chapter, in two to three minutes:

```bash
uv run python code/part05-dp-ml/choosing_epsilon.py
```

## Cheat sheet

| Question | Tool | Notes |
| --- | --- | --- |
| What does DP-SGD allow? | The trade-off curve, @eq-epsc-duality | Certify with the PLD, check with GDP |
| How sure can an attacker become? | @eq-epsc-posterior and @eq-epsc-precision | Name the prior; base rates dominate |
| Can a record be reconstructed? | The curve at FPR $1/m$ | State the prior, "one of $m$" |
| How small a $\delta$? | $\delta \le 1/n^2$ | $n$ counts privacy units |
| Comparing with a deployment? | @eq-epsc-zcdp; replace-one doubles $\varepsilon$ | Same unit and $\delta$ first |

: Which tool answers which question when choosing $\varepsilon$. {#tbl-epsc-cheat}

**Rules of thumb**

- Report the curve, not just the pair, as the TPR at FPR $10^{-3}$ and $10^{-2}$.
- A budget of at most 1 is strong for any mechanism. One up to 10 needs DP-SGD's curve as justification, and one above 10 needs the curve and an audit.
- Default to $\delta \le 1/n^2$, since it costs DP-SGD little, and never use $\delta \ge 1/n$.
- Never read the line $e^\varepsilon\,\mathrm{FPR} + \delta$ as DP-SGD's actual risk, because its curve lies far below it.
- Never quote $\varepsilon$ without its unit, $\delta$ and period, and never compare a daily budget with a lifetime one.

## Further reading

- Near, Darais, Lefkovitz & Howarth, *Guidelines for Evaluating Differential Privacy Guarantees* (NIST SP 800-226, 2025): privacy units, $\delta$, and the hazards of large budgets.
- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023): tiers and reporting.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/09-choosing-epsilon.md).
