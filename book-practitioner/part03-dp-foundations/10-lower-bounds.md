# Lower bounds: what privacy costs

## Why this matters

Sooner or later someone shows you a design that looks too good. It releases a histogram over a million locations with differential privacy (DP) at a privacy parameter $\varepsilon$ of 1 and no slack. It also promises an error of at most five records in every bin, with 95% probability. Before anyone tunes it, you want to know whether any mechanism could keep that promise.

A **lower bound** answers that question for every mechanism at once. It says that no mechanism satisfying the definition can be more accurate than some floor. So use it as a feasibility check. If a requirement is below the floor, the design is impossible, and the only ways out are more data, a coarser answer or a weaker guarantee. If your mechanism is already within a small factor of the floor, stop tuning it, because no algorithm can gain more than that factor.

Accuracy here has two numbers. The **error** $\alpha$ is how far the output may land from the true answer, and the **failure probability** $\beta$ is the chance that it lands further away. Pure DP has no slack, while approximate DP allows a small slack $\delta$ ([The definition of differential privacy](01-definition.md)). The examples use $\varepsilon = 1$, $\beta = 0.05$ and, for approximate DP, $\delta = 10^{-6}$.

## Choosing among many candidates: packing {#sec-lower-packing}

Many releases must single out one of many candidates. A mode names the most common of $N$ possible values, and a histogram over $N$ bins must show which bin holds a spike. Under pure DP, the **packing lemma** prices this. Suppose that each of $m$ candidates has a dataset on which a good answer must point to it. Suppose also that all these datasets lie within $k$ records of one reference dataset. Then an $\varepsilon$-DP mechanism that is right with probability $1 - \beta$ on all of them needs
$$k \ge \frac{\ln\big(m(1 - \beta)\big)}{\varepsilon}.$$ {#eq-lower-packing}
The cost grows only with the logarithm of the number of candidates, but under pure DP it never goes away.

For the mode, the candidates are the $N$ values, and $k$ is the **lead**, how many more times the top value appears than any other. Below a lead of $\ln(N(1 - \beta))/\varepsilon$, no pure-DP mechanism finds the mode reliably. A histogram that is accurate to within $\alpha$ in every bin could find a spike of height $2\alpha + 1$, so its largest bin error obeys
$$\alpha \ge \frac12\left(\frac{\ln\big(N(1 - \beta)\big)}{\varepsilon} - 1\right).$$ {#eq-lower-histogram}

For a million locations, every $\varepsilon$-DP histogram has an error of at least 6.38 records in some bin, so the design from the opening is impossible. The Laplace histogram of [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md) reaches 16.79, and since both numbers grow like $\ln N/\varepsilon$, Laplace is optimal up to a factor of about 2. Report-noisy-max from [Selection](05-selection.md) is optimal for the mode in the same sense, needing a lead of 18.44 where the floor is 13.76. Once a pure-DP histogram or selection uses one of these mechanisms, further tuning can win at most that factor.

The lemma needs pure DP. With a slack $\delta$, the argument stops at a distance of about $\ln(1/\delta)/\varepsilon$, which is 13.82 records here, however many candidates there are. A real mechanism reaches that cap. The stability-based histogram never releases an empty bin, so its error does not grow with $N$ at all ([Beyond global sensitivity](07-sensitivity.md)). So when a huge and mostly empty universe puts the pure-DP floor too high, switch to approximate DP rather than tune.

Packing also prices not knowing where the data lie. Suppose that the records are normal with variance 1, and all you know is that their mean lies within a distance $R$ of zero. Then estimating the mean to within $\alpha$ under pure DP needs at least $\ln(\lfloor R/\alpha\rfloor(1 - \beta))/\varepsilon$ records. At an error of 0.1, the floor is only 16.07 records even when $R$ is a million, and under approximate DP it is about $\ln(1/\delta)/\varepsilon$ whatever $R$ is. A mean whose error grows in step with $R$ is therefore paying for a clumsy mechanism, and [Private mean estimation](11-mean-estimation.md) gives one that pays only the logarithm.

## How many records you need {#sec-lower-sample}

An error in records becomes an error in fractions once you divide by the dataset size $n$, so every floor above is also a floor on $n$. Take a pure-DP histogram whose frequencies must all be within $\alpha$. It needs
$$n \ge \frac{\ln\big(N(1 - \beta)\big) - \varepsilon}{2\alpha\varepsilon}.$$ {#eq-lower-n-histogram}
Sampling costs records too. Estimating every frequency to within $\alpha$ from a random sample takes about $\ln(2N/\beta)/(2\alpha^2)$ records, by Hoeffding's inequality and a union bound ([Concentration inequalities](../part01-prerequisites/02-concentration.md)), and no estimator does much better. You need the larger of the two numbers. For a million cells at an error of 0.1, privacy needs at least 64 records while sampling needs up to 875. At an error of 0.01, privacy needs 638 and sampling 87,522. So privacy dominates only for coarse answers at strong privacy. When a plan is short of records for fine answers, sampling is the cause, and no better mechanism will fix it.

To size $n$ for a target, solve the floor, or the error of the mechanism you plan to use, for $n$ with a root finder such as `scipy.optimize.brentq`. For a model, an accountant such as Opacus's `get_noise_multiplier` or Google's `dp_accounting` gives the noise that a target $\varepsilon$ allows, and you check the resulting error the same way.

Many attributes change the picture. Release the means of $d$ yes/no attributes with an error of at most $\alpha$ in each. Packing over the $2^d$ answer patterns says that a pure-DP mechanism then needs about
$$n \ge \frac{d\ln 2}{\alpha\varepsilon}$$ {#eq-lower-n-marginals}
records, linear in $d$. The Gaussian mechanism is approximate DP and calibrates its noise to the $\ell_2$ sensitivity, so its requirement grows only like $\sqrt{d\ln d}$. @tbl-lower-sample shows the two rates. From $d = 2731$ attributes on, the Gaussian mechanism needs fewer records than any pure-DP mechanism can. So for outputs with many coordinates, such as a model's gradient, use approximate DP, because under pure DP the cost is linear in the dimension.

| | Pure DP, packing (lower) | Pure DP, Laplace (upper) | Approximate DP, Gaussian (upper) |
| --- | --- | --- | --- |
| $d = 1000$ | 6,921 | 198,070 | 10,836 |
| $d = 100{,}000$ | 693,137 | 29,017,315 | 134,299 |

: Records needed to release $d$ attribute means with $\ell_\infty$ error 0.1 (on the scale $[-1, 1]$) with probability 0.95, at $\varepsilon = 1$ and, for the Gaussian, $\delta = 10^{-6}$. {#tbl-lower-sample}

## The floor for approximate DP: fingerprinting {#sec-lower-fingerprinting}

Is $\sqrt d$ the price of approximate DP, or could a cleverer mechanism do better? **Fingerprinting** (Bun, Ullman & Vadhan 2014) says that it is the price. In its tight form (Steinke & Ullman 2016), any $(\varepsilon, \delta)$-DP release of $d$ attribute means with error $\alpha$ needs a number of records of order
$$n \gtrsim \frac{\sqrt{d\ln(1/\delta)}}{\alpha\varepsilon}.$$
The Gaussian mechanism matches this up to a factor of $\sqrt{\ln d}$. So once a release of many means uses well-calibrated Gaussian noise, stop looking for a better mechanism, and look for more records, fewer dimensions or a looser error instead.

Use this bound for its rate, not for its constants, which are poor. It also needs $\delta$ far below $1/n$, and so does every $(\varepsilon, \delta)$ claim. A mechanism that publishes each record outright with probability $\delta$ satisfies the definition, and at $\delta = 1/n$ it publishes about one record. So treat a claim with $\delta$ near $1/n$ as no guarantee at all.

## Tracing attacks on released means {#sec-lower-tracing}

The fingerprinting argument is an attack that has been run on real data. In 2008, Homer et al. showed that the published variant frequencies of a genome-wide study reveal whether a given person's DNA was in it. So the US National Institutes of Health withdrew such summaries that year. The attacker holds a target's record $y$ and the population means $p$. It scores how closely the released means $\hat q$ follow the target's deviations from the population:
$$Z(y) = \sum_{j=1}^d \hat q_j\,(y_j - p_j).$$
A member pulls the release toward its own record, so its score is shifted up, while a non-member's score is centred at zero. Without noise, the attack becomes strong once the number of attributes approaches the square of the number of people. Noise much larger than $\sqrt d/n$ per attribute defeats it, and the Gaussian mechanism adds noise of that order.

On the means of 50 people over 2,000 attributes, the full edition's simulation reaches a true positive rate (TPR) of 0.242 at a false positive rate (FPR) of 1% without noise. With the noise of a $(1, 10^{-5})$-DP Gaussian mechanism, that TPR falls to 0.014.

Use the score to audit any release of many averages, with the population means taken from a separate sample. Report the TPR at a low FPR, as in [Membership inference](../part02-attacks/02-membership-inference.md), and compare it with the cap $\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta$ that the [definition](01-definition.md) puts on every membership test. An audit that beats the cap has found a bug in the mechanism or in its accounting.

## Reconstruction as a floor on noise {#sec-lower-dinur-nissim}

[Reconstruction attacks](../part02-attacks/09-reconstruction.md) recover almost all of $n$ secret bits from about $n$ random counting queries, whenever the noise per answer grows more slowly than $\sqrt n$ (Dinur & Nissim 2003). So a private mechanism that answers that many counting queries must add error of order $\sqrt n$ to each answer, and the Gaussian mechanism adds just that. With many more queries than records, answer from a private synthetic dataset instead ([Query release](06-query-release.md)).

The same argument turns a measured attack into a statement about $\varepsilon$. Suppose the secret bits are independent, and each is equally likely to be 0 or 1. Then the definition's cap on guessing one bit says that no attack on an $(\varepsilon, \delta)$-DP output recovers, on average, more than a fraction $r$ of them, where
$$r \le \frac{e^\varepsilon + \delta}{e^\varepsilon + 1}, \qquad\text{so}\qquad \varepsilon \ge \ln\frac{r - \delta}{1 - r}.$$ {#eq-lower-dn}
At $\varepsilon = 1$, that limit is 73.11%. The bits must be uniform, because with biased bits a guesser beats one half from the prior alone.

For example, recovering 96.07% of the bits proves that $\varepsilon$ is at least 3.20. But an attack bounds $\varepsilon$ only from below, and narrowing the gap to the guarantee is the job of [Auditing private training](../part05-dp-ml/10-auditing.md).

## Cheat sheet

| Release | Floor: no mechanism does better | Closest mechanism |
| --- | --- | --- |
| Mode of $N$ values, pure DP | Lead $k \ge \ln(N(1 - \beta))/\varepsilon$ | Report-noisy-max |
| Histogram over $N$ bins, pure DP | $\alpha \ge \frac12(\ln(N(1 - \beta))/\varepsilon - 1)$ | Laplace, about twice the floor |
| Histogram or mode, approximate DP | About $\ln(1/\delta)/\varepsilon$, whatever $N$ | Stability-based histogram |
| Mean within $R$ of zero, pure DP | $n \ge \ln(\lfloor R/\alpha\rfloor(1 - \beta))/\varepsilon$ | Coarse then fine |
| $d$ means, pure DP | $n \gtrsim d\ln 2/(\alpha\varepsilon)$ | Laplace, within a factor of order $\ln d$ |
| $d$ means, approximate DP | $n \gtrsim \sqrt{d\ln(1/\delta)}/(\alpha\varepsilon)$ | Gaussian, within $\sqrt{\ln d}$ |
| About $n$ counting queries | Error of order $\sqrt n$ per answer | Gaussian |
| A fraction $r$ of uniform bits recovered | Proves $\varepsilon \ge \ln\frac{r - \delta}{1 - r}$ | — |

: The floors of this chapter as feasibility checks, with the mechanism that comes closest to each. {#tbl-lower-cheat}

**Rules of thumb**

- Check a requirement against its floor before tuning. Below the floor, ask for more data, a coarser answer or a weaker guarantee.
- A floor holds for every mechanism, not just for bad ones, so no clever algorithm gets under it.
- Never promise per-cell accuracy for a large contingency table at small $\varepsilon$ without checking the histogram floor, and use approximate DP for outputs with many coordinates.
- Stop tuning once a mechanism is within its known factor of the floor, about 2 for Laplace histograms and $\sqrt{\ln d}$ for Gaussian means.
- When sampling needs more records than privacy does, collect more data instead of changing the mechanism.
- Aggregates are not a defence. Never release many unnoised statistics from a small cohort, and audit releases of many averages with the tracing score.
- Keep $\delta$ far below $1/n$.

## Further reading

- Vadhan, "The Complexity of Differential Privacy" (2017), §5: the three lower bounds side by side, and the stability-based histogram.
- Steinke & Ullman, "Between Pure and Approximate Differential Privacy" (Journal of Privacy and Confidentiality, 2016): the simplified fingerprinting lemma and the tight $\sqrt{d\ln(1/\delta)}$ bound for attribute means.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/10-lower-bounds.md).
