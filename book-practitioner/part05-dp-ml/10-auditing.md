# Auditing private training

## Why this matters

When you train with differential privacy (DP), the accountant turns your configuration into a privacy level $\varepsilon$ at a slack $\delta$. That number is a proof about an algorithm, but what ships is code. A bug in the noise, the clipping or the sampling makes the model less private but barely changes its accuracy, which is flat across a wide range of noise levels ([the training workflow](13-train-dp-sgd.md)). So the usual tests never notice the bug.

An **audit** checks the code by attacking it. It plants records called **canaries**, trains with and without them, and counts how often a membership test guesses right. DP caps how often any test can be right, so a test that is right too often proves that the code is less private than claimed. A test that is rarely right proves nothing, since a weak attack also fails against a leaky model.

## Known bugs, and which audit catches them {#sec-audit-bugs}

These bugs break the private form of stochastic gradient descent (SGD), [DP-SGD](02-dp-sgd.md).

- **The variance passed where the standard deviation is expected,** or **noise scaled for the mean but added to the sum,** which adds the batch size times too little noise. A unit test in continuous integration (CI) catches both (@sec-audit-ci).
- **Clipping applied after averaging, or noise from a weak seed.** Nasr et al. (2023) found both in real code with per-step audits like the CI test.
- **The accountant told the wrong numbers,** such as a record count that includes augmented copies, which makes $\varepsilon$ look smaller than it is. Only an end-to-end audit can catch it, and only if the error is large.
- **Batches shuffled but accounted as Poisson sampling,** which can make the true $\varepsilon$ several times the reported one ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md)). No audit here exercises the sampling, so check the loader itself.

Audits catch errors only in the unsafe direction, so code that adds too much noise passes them all.

## A CI test for the noise and the clipping {#sec-audit-ci}

The cheapest strong audit tests one step of the optimizer, many times. A **Dirac canary** is a per-example gradient that is zero except on one coordinate, where every other record's gradient is zero. Put it into every step of one run and leave it out of another. The reading $x_t$ is step $t$'s noisy gradient sum on that coordinate, divided by the clipping norm $C$. The canary is longer than $C$, so a correct optimizer clips it to length $C$ and it adds exactly 1 to the reading. The optimizer also adds noise of standard deviation $\sigma C$, where $\sigma$ is the noise multiplier, so each reading follows
$$x_t \sim \mathcal N(0, \sigma^2) \text{ without the canary}, \qquad x_t \sim \mathcal N(1, \sigma^2) \text{ with it}.$$ {#eq-audit-perstep}

A threshold test says "canary in" whenever a reading exceeds a threshold. Its **true positive rate (TPR)** is how often it says so when the canary was there, and its **false positive rate (FPR)** how often when it was not.

For two normal distributions a distance $\mu$ apart, every threshold gives $\Phi^{-1}(\mathrm{TPR}) - \Phi^{-1}(\mathrm{FPR}) = \mu$, where $\Phi$ is the standard normal distribution function. So measured rates bound the true separation $\mu$, which a correct optimizer keeps at $1/\sigma$:
$$\mu \ge \Phi^{-1}(\mathrm{TPR}) - \Phi^{-1}(\mathrm{FPR}).$$ {#eq-audit-mu}
The test plugs in the low end of the TPR's Clopper–Pearson interval and the high end of the FPR's. Each end is taken at level $\beta/2$, so the bound holds with probability $1 - \beta$ ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). If this bound exceeds $1/\sigma$, the code adds less noise than it claims.

The script [practitioner_audit.py](../../code/part05-dp-ml/practitioner_audit.py) runs the test on Opacus's own `DPOptimizer`. It writes per-example gradients into `grad_sample` by hand, as Opacus's hooks would, and reads the noisy sum that `optimizer.step()` leaves in `grad`.

```{.python excerpt="audit-ci-observe"}
def canary_coordinate(sigma: float, planted_bug: str, with_canary: bool, steps: int) -> Array:
    """Run the optimizer for `steps` steps; return the canary coordinate of each noisy sum, in units of C."""
    layer = torch.nn.Linear(DIM_CI, 1, bias=False)
    optimizer = make_optimizer(layer, sigma, planted_bug)
    observations = np.empty(steps)
    for t in range(steps):
        setattr(layer.weight, "grad_sample", batch_grad_samples(with_canary))  # as Opacus's hooks would
        optimizer.step()  # clip each row to C, sum, add N(0, σ²C²)
        assert layer.weight.grad is not None
        observations[t] = layer.weight.grad[0, CANARY_COORD].item() / CLIP_CI
        optimizer.zero_grad()
    return observations
```

The canary's gradient is ten times $C$, so the test also exercises clipping. The script chooses the threshold on the first half of the steps and counts the rates on the second half. It also checks that the canary-free readings have a standard deviation within 10% of $\sigma$, since with no noise every reading is 0. The command `uv run python code/part05-dp-ml/practitioner_audit.py` tests three optimizers at $\sigma = 0.7$ and a batch of $B = 64$ records, the correct one, one with the variance bug and one with no noise. The variance bug removes noise only because $\sigma^2 < \sigma$ when $\sigma$ is below 1, and above 1 it would pass.

```text
sigma=0.7  C=1.0  B=64  steps per run=2000  claim mu=1/sigma=1.429
correct: noise std 0.687 (claim 0.7)  FP 206/1000  TP 710/1000
correct: mu_lo=1.201  pass
variance for std: noise std 0.481 (claim 0.7)  FP 206/1000  TP 862/1000
variance for std: mu_lo=1.722  FAIL
no noise: noise std 0.000 (claim 0.7)  FP 0/1000  TP 1000/1000
no noise: mu_lo=5.360  FAIL
```

The correct optimizer's bound on $\mu$ is 1.201, below its claimed 1.429, so it passes. The variance bug's bound is 1.722 and the missing noise's is 5.360, so both fail. The test takes seconds, so run it on every commit.

Three things go wrong in practice.

First, a correct optimizer still fails with probability $\beta$, here 5%, so an unseeded test fails one build in twenty. Fix the seed, or use a $\beta$ of $10^{-3}$ with more steps. Second, the script builds its own optimizer, so it tests Opacus, not your call to it. To test your code, have your setup function return the optimizer it makes, and drive that one. Third, the test never draws a batch or calls the accountant. So the same suite should assert that the loader is Opacus's `DPDataLoader` and that a second library's $\varepsilon$ matches yours ([Calibrate and account a DP-SGD run](../part04-privacy-accounting/10-calibrate-dp-sgd.md#sec-calib-crosscheck)).

## From an attack to a lower bound {#sec-audit-bound}

For a whole training run, the attack is a membership test on a planted canary. A **trial** is one training run, with or without the canary. Under $(\varepsilon, \delta)$-DP, every such test satisfies
$$\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta, \qquad 1 - \mathrm{FPR} \le e^\varepsilon\,(1 - \mathrm{TPR}) + \delta.$$ {#eq-audit-region}

Solving these for $\varepsilon$ turns measured rates into a lower bound on it. Write $\overline{\mathrm{FPR}}$ for the one-sided Clopper–Pearson upper bound on the FPR at level $\beta/2$, and $\underline{\mathrm{TPR}}$ for the TPR's lower bound. With probability at least $1 - \beta$, the code is then not $(\varepsilon, \delta)$-DP, at the claimed $\delta$, for any $\varepsilon$ below
$$\varepsilon_{\mathrm{lo}} = \max\Big\{\ln\frac{\underline{\mathrm{TPR}} - \delta}{\overline{\mathrm{FPR}}},\ \ln\frac{1 - \overline{\mathrm{FPR}} - \delta}{1 - \underline{\mathrm{TPR}}}\Big\}.$$ {#eq-audit-eps-lo}
The bound needs independent trials, and a threshold chosen on other trials than those it counts, because a threshold tuned on them inflates it.

No library installed with this book computes the bound, so the script imports `eps_lower_bound` from the book's helper module, [practitioner_privacy_tests.py](../../code/practitioner_privacy_tests.py). Suppose an attack catches the canary in 500 of 1,000 runs that include it, and raises one false alarm in 1,000 runs without it.

```{.python excerpt="audit-eps-bound"}
for false_pos, true_pos in ((1, 500), (0, 1_000)):  # out of 1,000 runs without and with the canary
    eps_lo = eps_lower_bound(false_pos, true_pos, 1_000, DELTA)
    print(f"FP {false_pos}/1000  TP {true_pos}/1000: certified eps lower bound {eps_lo:.2f}")
```

```text
FP 1/1000  TP 500/1000: certified eps lower bound 4.43
FP 0/1000  TP 1000/1000: certified eps lower bound 5.60
```

The raw rates alone would suggest 6.21, but the certified bound is only 4.43, and even a perfect attack certifies only 5.60 from 1,000 trials per hypothesis. So do not expect an end-to-end audit to certify all of a large claim.

## A one-run audit of the trained model {#sec-audit-onerun}

Thousands of runs are out of reach for a real model, so Steinke, Nasr & Jagielski (2023) get many trials from one run. Plant a number $m$ of canaries, include each one by its own fair coin, train once and score every canary. Then fix a list length $r$. Guess "in" for the $r/2$ highest scores and "out" for the $r/2$ lowest, and make no guess about the rest. Write $W$ for the number of correct guesses.

Under $\varepsilon$-DP, no attack guesses one canary's coin right with a probability above the limit $p_\varepsilon$ given below. So the count $W$ is no more likely to reach any value $v$ than a binomial count with that success rate is:
$$\Pr[W \ge v] \le \Pr\big[\mathrm{Bin}(r, p_\varepsilon) \ge v\big], \qquad p_\varepsilon = \frac{e^\varepsilon}{1 + e^\varepsilon}.$$ {#eq-audit-onerun}
An observed count that this binomial reaches with probability below $\beta$ rules out that $\varepsilon$, and the certified bound is the largest $\varepsilon$ ruled out. Under $(\varepsilon, \delta)$-DP, the paper adds at most $2m\delta$ to that probability. The coins must be independent and fair, and several list lengths need a Bonferroni correction, which runs each test at $\beta$ divided by their number.

The script runs it on the released pipeline of [Train a model with DP-SGD](13-train-dp-sgd.md) on the Adult census data. It adds only the canaries, 1,000 held-out rows with flipped labels. The coins put 537 of them in the training set, and the release's noise multiplier gives $\varepsilon = 3.91$ for this larger set at $\delta = 10^{-5}$. A non-private model uses the release's baseline recipe, and a memorizing control runs non-private Adam for 60 epochs with small batches. Each canary is scored by minus its loss, and @tbl-audit-adult gives each model's best of six list lengths from 20 to 1,000.

| Training | Mean canary loss, in / out | Correct guesses | Certified $\varepsilon$ |
| --- | --- | --- | --- |
| DP-SGD, release recipe | 4.105 / 4.006 | 10 of 20 | 0.00 |
| Non-private, release recipe | 2.240 / 2.222 | 8 of 20 | 0.00 |
| Memorizing control | 1.994 / 2.992 | 75 of 100 | 0.53 |

: One-run audit of the released pipeline on Adult with 1,000 mislabeled canaries, at 95% overall confidence and $\delta = 10^{-5}$. {#tbl-audit-adult}

Both release-recipe models certify nothing, and the reason is the recipe, not privacy. Ten epochs with large batches barely fit the flipped labels, so included canaries have the same loss as excluded ones, with or without noise. Only the memorizing control separates its canaries. Even it certifies just 0.53 for a model with no finite $\varepsilon$, because each canary is a single trial. A perfect list of all 1,000 guesses would certify at most 5.27.

So use the one-run audit to expose a bug, when a DP model certifies more than its claim, never as evidence of privacy. Report it with its threat model, a black-box attacker who sees only the final model, and with a control that shows the attack can detect leakage on your data.

Stronger black-box attacks, such as the likelihood ratio attack (LiRA) of [Pre-release privacy risk assessment](../part02-attacks/10-risk-assessment.md#sec-risk-lira), train **shadow models**. These are copies of the model, trained with its exact recipe on similar data. Run them before release and keep their results with the evidence ([Release and governance](12-release-governance.md#sec-relgov-evidence)). LiRA's TPR at a low FPR shows what a realistic attacker learns. But its (model, record) pairs share models, so they are not independent trials and give no certified bound.


## Cheat sheet

| Question | Audit and cost | Rule | Blind to |
| --- | --- | --- | --- |
| Does the code clip and add the noise it claims? | CI test, seconds | Fail if the bound on $\mu$, @eq-audit-mu, exceeds $1/\sigma$ | Sampling and accounting bugs |
| Is the claimed $\varepsilon$ right, accounting included? | End-to-end audit, thousands of runs | @eq-audit-eps-lo, Clopper–Pearson at $\beta/2$ | Bugs smaller than the bound's shortfall |
| Does a trained model leak planted records? | One-run audit, one extra run | @eq-audit-onerun plus $2m\delta$, Bonferroni | Most bugs, since its bounds are low |
| What can a realistic attacker learn? | Shadow-model attack ([LiRA](../part02-attacks/10-risk-assessment.md#sec-risk-lira)) | TPR at a low FPR; no certified bound | The most the code could leak |

: Which audit answers which question, and its rule. {#tbl-audit-cheat}

## References

- Nasr et al., "Tight Auditing of Differentially Private Machine Learning" (USENIX Security 2023): Dirac canaries, per-step audits and real bugs.
- Steinke, Nasr & Jagielski, "Privacy Auditing with One (1) Training Run" (NeurIPS 2023): the one-run audit.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/10-auditing.md).
