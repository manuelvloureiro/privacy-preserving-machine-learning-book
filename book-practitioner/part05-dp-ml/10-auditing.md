# Auditing private training

## Why this matters

The running example trains a network with differential privacy (DP), using the private form of stochastic gradient descent (SGD) called [DP-SGD](02-dp-sgd.md). Its noise multiplier $\sigma$, the noise's standard deviation in units of the clipping norm, is 1. The Rényi differential privacy (RDP) accountant turns its configuration into a privacy level $\varepsilon$ of 6.71 at a slack $\delta$ of $10^{-5}$. That number is a proof about an algorithm, while what ships is code, and a bug in the noise or the sampling barely changes the accuracy.

An **audit** checks the code by attacking it. It plants records called **canaries**, trains with and without them, and counts how often a membership test guesses right. DP caps how often any test can be right, so a test that is right too often proves that the code is less private than claimed. The reverse never holds, because a weak attack also fails against a leaky model.

## Which audit for which question {#sec-audit-choose}

Use the cheapest audit that answers your question (@tbl-audit-choose). The per-step audit also works as a regression test (@sec-audit-ci).

| Question | Audit | Cost | Blind to |
| --- | --- | --- | --- |
| Does the code clip and add the noise it claims? | Per-step audit with a Dirac canary in every step (@sec-audit-bugs) | Two training runs | Sampling and accounting bugs |
| Is the claimed $\varepsilon$ right, accounting included? | End-to-end white-box audit with the likelihood-ratio statistic (@sec-audit-tight) | Thousands of runs per hypothesis | Bugs smaller than its shortfall, about 3 units here |
| A cheap end-to-end check of a slow model | One-run audit with many white-box canaries (@sec-audit-onerun) | One run | Most bugs, since its best bound here was 2.53 |
| What can a realistic attacker learn from the released model? | Black-box one-run audit with mislabeled canaries (@sec-audit-blackbox) | One run, plus controls | The mechanism's worst case |

: Which audit answers which question. "Here" means the running example, whose claim is 6.71. {#tbl-audit-choose}

## Known bugs, and which audit catches them {#sec-audit-bugs}

Read the code for each bug below, since an audit catches only some of them.

- **Variance passed where a standard deviation is expected.** With the running example's clipping norm of 0.5, this halves the noise. A per-step audit catches it.
- **Noise scaled for the mean but added to the sum,** which makes it too small by the batch size. A per-step audit catches it at once.
- **Clipping applied after averaging, or noise from a weak seed** with only 100 possible values. Nasr et al. (2023) found both in real code with per-step audits.
- **The accountant told the wrong numbers,** such as 20,000 records instead of 10,000 because augmented copies were counted. The noise is right, so only an end-to-end audit can catch it, and only if the error is large.
- **Batches shuffled but accounted as Poisson sampling,** which can make the true $\varepsilon$ several times the reported one ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md)). No cheap audit exercises the sampling, so read that code yourself.

The per-step audit puts a Dirac canary (@sec-audit-canaries) into every step of one run and leaves it out of another. Each step's noisy value on the canary's coordinate, in units of the clipping norm, is one trial, and for correct code it follows
$$x_t \sim \mathcal N(0, \sigma^2) \text{ without the canary}, \qquad x_t \sim \mathcal N(1, \sigma^2) \text{ with it}.$$ {#eq-audit-perstep}
The audit bounds the separation $\mu$ of these two distributions from below with @eq-audit-mu, and correct code has $\mu = 1/\sigma$. On the running example, two runs of 10,000 steps bound $\mu$ by 0.886 for the correct code, against the claimed 1, while the variance bug gives 1.745 and is refuted.

Audits catch errors only in the unsafe direction, so code that adds too much noise passes every audit.

## From an attack to a lower bound {#sec-audit-bound}

A **trial** is one training run with or without the canary, and the attack says "in" when a score, such as minus the canary's loss, is above a threshold. Its **true positive rate (TPR)** is the chance that it says "in" when the canary was trained on. Its **false positive rate (FPR)** is the chance that it says "in" when the canary was left out. DP requires, for every test,
$$\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta, \qquad 1 - \mathrm{FPR} \le e^\varepsilon\,(1 - \mathrm{TPR}) + \delta.$$ {#eq-audit-region}

The rates are measured from $n$ trials with the canary and $n$ without, so the bound must use their pessimistic ends. Write $\bar\alpha$ for the one-sided Clopper–Pearson upper bound on the FPR ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). Write $\underline\tau$ for the lower bound on the TPR, both at level $\beta/2$, where $\beta$ is the chance that the audit is wrong. For a count of `k` events in `n` trials, SciPy gives the upper bound as `scipy.stats.beta.ppf(1 - beta / 2, k + 1, n - k)`, and statsmodels' `proportion_confint(..., method="beta")` gives the same interval. With probability at least $1 - \beta$, the code is then not $(\varepsilon, \delta)$-DP for any level below
$$\varepsilon_{\mathrm{lo}} = \max\Big\{\ln\frac{\underline\tau - \delta}{\bar\alpha},\ \ln\frac{1 - \bar\alpha - \delta}{1 - \underline\tau}\Big\}.$$ {#eq-audit-eps-lo}
The bound needs independent trials and a threshold chosen without looking at the counted trials. So choose the threshold on half of the trials, and count on the other half.

Suppose 1,000 runs with the canary give 500 true positives, and 1,000 without give one false positive. The point estimate is 6.21, but confidence is expensive, and the certified bound is only 4.43. Even a perfect attack hits a ceiling. With $n$ trials per hypothesis, it can certify at most about
$$\varepsilon_{\max}(n) \approx \ln\frac{n}{\ln(2/\beta)}.$$ {#eq-audit-ceiling}
So each extra unit of $\varepsilon$ costs a factor of $e$ in trials. At 95% confidence, 1,000 trials per hypothesis certify at most 5.60, and 10,000 at most 7.90. Certifying the running example's 6.71 takes at least 3,038, so no feasible end-to-end audit can confirm that a large claim is tight.

## Canaries and threat models {#sec-audit-canaries}

An **input-space canary** is a record designed to stand out, such as a mislabeled example. A **gradient-space canary** has a gradient that the auditor writes directly. The strongest is the **Dirac canary** of Nasr et al. (2023). Its gradient has length exactly the clipping norm, all on one coordinate that no other record's gradient touches.

The threat model says what the attacker sees. In the **black-box** model it sees only the final model. In the **white-box** model it sees every intermediate model, and so it can recover each step's noisy gradient sum. Against a Dirac canary seen white-box, the most powerful test thresholds a **likelihood-ratio statistic** built from those sums, which the full edition derives. A simpler score that adds up the canary coordinate's total change is weaker, lowering the most the audit can certify on the running example from 6.01 to 4.35.

## Tight audits of the mechanism {#sec-audit-tight}

An end-to-end white-box audit with Dirac canaries faces the accountant's own worst case, but it still falls far short of the claim, because at small $\delta$ the level is decided at a tiny FPR. For the running example that rate is $8.1\times10^{-8}$, far below the $3/n$ that $n$ trials can measure. Nasr et al. (2023) avoid the tail by assuming a Gaussian trade-off curve. Write $\Phi$ for the standard normal distribution function. A single test then bounds the separation $\mu$ from below,
$$\mu \ge \Phi^{-1}(\mathrm{TPR}) - \Phi^{-1}(\mathrm{FPR}),$$ {#eq-audit-mu}
read at a moderate FPR, where trials are cheap. Its conversion to $\varepsilon$ holds only if the curve really is Gaussian.

On the running example, the rigorous bound @eq-audit-eps-lo reaches 3.63 with 50,000 trials per hypothesis, about half the claim. The Gaussian-shape estimate reaches 5.33.

## One-run auditing {#sec-audit-onerun}

Steinke, Nasr & Jagielski (2023) get many trials from one run. Choose a number $m$ of canaries, include each one by its own independent fair coin, train once and score every canary. For a list of $r$ guesses, guess "in" for the $r/2$ highest scores and "out" for the $r/2$ lowest. Under $\varepsilon$-DP, the number $W$ of correct guesses is no more likely to reach any count $v$ than a binomial count is:
$$\Pr[W \ge v] \le \Pr\big[\mathrm{Bin}(r, p_\varepsilon) \ge v\big], \qquad p_\varepsilon = \frac{e^\varepsilon}{1 + e^\varepsilon}.$$ {#eq-audit-onerun}
The certified bound is the largest level whose tail at the observed count is below $\beta$.

The coins must be independent and fair, and trying several list lengths needs a Bonferroni correction. Under $(\varepsilon, \delta)$-DP, the paper adds a term of at most $2m\delta$.

On the running example, one run with 1,000 white-box Dirac canaries certifies 2.22, against 2.84 from 10,000 runs per hypothesis.

## Black-box audit of the trained network {#sec-audit-blackbox}

To see what the released network gives away, plant **mislabeled canaries** and attack the final model with a loss score or the likelihood-ratio attack of [Membership inference](../part02-attacks/02-membership-inference.md). Run a non-private control next to it.

The non-private model memorizes its canaries and certifies 2.96, but the DP-SGD model certifies only 0.13. A model with clipping but no noise, which has no finite $\varepsilon$ at all, certifies only 0.24. So a black-box bound of 0.5 against a claimed 8 does not show that the model is safe, or that its real $\varepsilon$ is 0.5. It measures this attacker only.

## Auditing in continuous integration {#sec-audit-ci}

The per-step audit is cheap enough to run as a unit test on every commit. Feed the code's own clip-and-noise step a Dirac canary's gradient, written by hand. In Opacus, for example, write it into a parameter's `p.grad_sample` before the `DPOptimizer` runs `clip_and_accumulate` and `add_noise`, and read the noisy gradient afterwards. Test the code path that ships, not a copy of it. Run the steps with the canary and without, record each step's noisy value on the canary's coordinate, and bound $\mu$ with @eq-audit-mu. The test fails when the bound exceeds $1/\sigma$.

Fix the seeds so that a failure can be reproduced, and use a $\beta$ below 5%, or correct code fails for one seed in twenty.

## Code

This snippet computes the certified bound of @eq-audit-eps-lo from counts, and prints the number quoted in the text.

```python
from collections.abc import Callable
from math import exp, lgamma, log


def binom_cdf(k: int, n: int, p: float) -> float:
    """Pr[Bin(n, p) <= k], summed in logs so that large n is safe."""
    log_terms = (lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * log(p) + (n - i) * log(1 - p) for i in range(k + 1))
    return min(1.0, sum(exp(t) for t in log_terms))


def bisect(f: Callable[[float], float], target: float, lo: float, hi: float) -> float:
    """The x in [lo, hi] with f(x) = target, for f decreasing in x."""
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
    return lo


def eps_lower_bound(fp: int, tp: int, n: int, delta: float, beta: float = 0.05) -> float:
    """The eps certified with confidence 1 - beta from fp false and tp true positives in n trials each."""
    fpr_hi = bisect(lambda p: binom_cdf(fp, n, p), beta / 2, 1e-12, 1 - 1e-12)
    tpr_lo = 1 - bisect(lambda p: binom_cdf(n - tp, n, p), beta / 2, 1e-12, 1 - 1e-12)
    return max(log((tpr_lo - delta) / fpr_hi), log((1 - fpr_hi - delta) / (1 - tpr_lo)))


print(f"{eps_lower_bound(1, 500, 1_000, 1e-5):.2f}")  # 4.43
```

The full edition's script runs every audit:

```bash
uv run python code/part05-dp-ml/auditing.py
```

## Cheat sheet

| Result | Formula | Conditions |
| --- | --- | --- |
| Certified bound | @eq-audit-eps-lo, Clopper–Pearson at $\beta/2$ | Independent trials; threshold chosen on other trials |
| Ceiling | @eq-audit-ceiling, 5.60 at 1,000 trials | A perfect attack |
| One run | @eq-audit-onerun, plus at most $2m\delta$ | Canaries included by independent fair coins |

: The results of this chapter. {#tbl-audit-cheat}

**Rules of thumb**

- An audit can refute a claim, never prove one, so report its threat model with its number.
- Budget trials before you start, at least $e^\varepsilon\ln(2/\beta)$ per hypothesis, and never call an audit tight with too few.
- Choose thresholds and list lengths on separate data, or pay with a Bonferroni correction.
- Read the sampling and accounting code yourself, because the cheap audits cannot see them.
- Never read a small black-box bound as the model's real $\varepsilon$.

## Further reading

- Nasr, Hayes, Steinke, Balle, Tramèr, Jagielski, Carlini & Terzis, "Tight Auditing of Differentially Private Machine Learning" (USENIX Security 2023): Dirac canaries, per-step audits and real bugs.
- Steinke, Nasr & Jagielski, "Privacy Auditing with One (1) Training Run" (NeurIPS 2023): the one-run audit.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/10-auditing.md).
