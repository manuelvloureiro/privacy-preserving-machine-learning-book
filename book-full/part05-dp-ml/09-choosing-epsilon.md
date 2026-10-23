# Choosing ε

## Why this matters

Every model trained with differential privacy (DP) ships with a number, its budget $\varepsilon$. DP is the guarantee that one record changes the probability of any output by at most a factor $e^\varepsilon$, up to a slack $\delta$. [Chapter 39](02-dp-sgd.md) trained the part's running example, a small neural network on 10,000 records, with stochastic gradient descent (SGD) made private by per-example clipping and Gaussian noise, DP-SGD, at five budgets, and measured what each costs in accuracy: test accuracy 0.9175 at $\varepsilon = 1$, 0.9441 at $\varepsilon = 2$, 0.9548 at $\varepsilon = 4$ and 0.9604 at $\varepsilon = 8$, against 0.9610 without privacy, all at $\delta = 10^{-5}$. That is half of a decision. The other half is what each budget means for the people in the data: what an attacker who sees the model can learn about one of them, and how sure it can be. Without that half, the choice of $\varepsilon$ is made by habit ("everyone uses 8") or by the accuracy curve alone, and neither survives a reviewer, a regulator or a data subject who asks what the number promises.

This chapter supplies the privacy half. It reads the guarantee in three operational ways: as a bound on any membership test, as a bound on how far an attacker's belief can move, and as a bound on reconstructing a record. For each it compares two numbers: what the single pair $(\varepsilon, \delta)$ promises, and what the actual training algorithm, DP-SGD, really allows. It then turns to $\delta$, to the budgets that real deployments have published and the units they are stated in, and to a procedure for choosing and defending a number.

### The one idea: ε is one line under a curve

An attacker who wants to know whether a record was in the training set runs a test, and every test has two error rates. The definition of DP is equivalent to a straight-line bound on those rates: at false positive rate $a$, the true positive rate is at most $e^\varepsilon a + \delta$. But a training algorithm has a whole **trade-off curve**, the best true positive rate at every false positive rate, and $(\varepsilon, \delta)$ is only one line that lies above it, touching it at one point. Far from that point, the line can be useless: at $\varepsilon = 8$ it allows a test that catches every member while almost never accusing a non-member, while the curve of DP-SGD at the same $\varepsilon$ caps the true positive rate at 0.0589 when the false positive rate is $10^{-3}$. Choosing $\varepsilon$ well means computing and reporting the curve, not just the line, and reading it at the error rates that matter for your threat.

@tbl-epsc-overview lists the results and where each is used.

| Result | Where it is used |
| --- | --- |
| What one $(\varepsilon, \delta)$ pair bounds: true positive rate at most $e^\varepsilon$ times the false positive rate plus $\delta$, and the advantage | Reading any published budget; the baseline every refinement improves on |
| The whole trade-off curve of DP-SGD: exactly from the privacy loss distribution, bounded by Rényi DP, approximated by Gaussian DP | Stating what a trained model allows at low false positive rates; [Auditing](10-auditing.md) |
| The Bayesian reading: posterior bounds, the realized privacy loss, the precision of a flagged member | Explaining a budget to someone who thinks in probabilities |
| Reconstruction: success is at most the membership true positive rate at a false positive rate equal to the baseline | Arguing that large budgets can still prevent reconstruction |
| What $\delta$ means and how small it must be | Choosing $\delta$; reading guarantees with large $\delta$ |
| Deployed budgets and their privacy units | Comparing your budget with practice, honestly |
| A procedure for choosing and justifying $\varepsilon$ | Writing the privacy statement of a model |

: The results of this chapter and where each is used. {#tbl-epsc-overview}

**Notation.** A **membership test** sees the trained model (or the whole sequence of iterates) and says "member" or "non-member" about a target record $z^*$. Its **false positive rate (FPR)** $a$ is the probability that it says "member" when $z^*$ was not in the training set, and its **true positive rate (TPR)** is the probability that it says "member" when $z^*$ was. $Q$ is the distribution of the output when $z^*$ is in the data, $P$ the distribution when it is not, and $L = \ln(Q(y)/P(y))$ is the **privacy loss** of an output $y$. $\delta(\varepsilon)$ is the smallest $\delta$ for which a mechanism is $(\varepsilon, \delta)$-DP at a given $\varepsilon$, its **privacy profile**. For DP-SGD, $q$ is the sampling rate, $\sigma$ the noise multiplier and $T$ the number of steps. The running example has $q = 0.01$, $T = 10{,}000$ and $n = 10{,}000$ records, and its noise multiplier is the one the Rényi differential privacy (RDP) accountant of [DP-SGD](02-dp-sgd.md) requires for each target $\varepsilon$ at $\delta = 10^{-5}$.

**How to read this chapter.** The toolkit and @sec-epsc-pair restate what [The definition of differential privacy](../part03-dp-foundations/01-definition.md) proves, in the form this chapter needs. @sec-epsc-curve is the core: the curve of DP-SGD against the line. @sec-epsc-bayes and @sec-epsc-recon read the same curve as beliefs and as reconstruction. @sec-epsc-delta, @sec-epsc-deployed and @sec-epsc-justify are about practice and can be read on their own.

## Toolkit: four facts every derivation uses

### Fact 1 — A membership attacker is a test, and DP bounds the probability of every event

Fix the training set without the target, $D^-$, and the target record $z^*$. The attacker sees an output $y$, drawn from $Q = M(D^- \cup \{z^*\})$ if $z^*$ is a member and from $P = M(D^-)$ if not, and says "member" on some set $S$ of outputs. Then $\mathrm{FPR} = P(S)$ and $\mathrm{TPR} = Q(S)$: both error rates are probabilities of one event. A mechanism $M$ is $(\varepsilon, \delta)$-DP when, for every pair of neighbouring datasets and every event $S$, in both orders,
$$Q(S) \le e^\varepsilon P(S) + \delta.$$
So DP is a statement about tests before it is a statement about anything else. This chapter uses add/remove neighbours, as DP-SGD with Poisson sampling does ([DP-SGD](02-dp-sgd.md)). An attacker that randomizes, saying "member" with probability $\phi(y)$, has $\mathrm{TPR} = \mathbb E_Q[\phi]$ and $\mathrm{FPR} = \mathbb E_P[\phi]$, and the same bound holds for it, because a randomized test is an average of events ([The definition of differential privacy](../part03-dp-foundations/01-definition.md#sec-dpdef-testing)).

### Fact 2 — The best test thresholds the privacy loss

Among all tests with a given FPR, the one with the largest TPR says "member" when the likelihood ratio $Q(y)/P(y)$ is large, that is, when the privacy loss $L(y)$ exceeds a threshold $\tau$, randomizing at the threshold if needed. This is the Neyman–Pearson lemma ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). The reason is an exchange argument: if a test said "member" on an output with a low ratio and "non-member" on one with a higher ratio, swapping the two (in equal amounts of $P$-probability) would keep its FPR and raise its TPR.

Two consequences follow. First, the best TPR at FPR $a$ depends only on the distribution of $L$. Write $\omega$ for the distribution of $L$ when $y \sim Q$, the **privacy loss distribution (PLD)**. Then the best test with threshold $\tau$ has
$$\mathrm{TPR}(\tau) = Q[L > \tau], \qquad \mathrm{FPR}(\tau) = P[L > \tau] = \mathbb E_Q\big[e^{-L};\ L > \tau\big],$$
where the second form is a **change of measure**: $P(y) = e^{-L(y)}Q(y)$ by the definition of $L$, so the $P$-probability of any event is the $Q$-average of $e^{-L}$ over it. Second, the best achievable pairs $(\mathrm{FPR}, \mathrm{TPR})$ form a region whose upper boundary, the **trade-off curve** written here as $\mathrm{TPR}^*(a)$, is increasing and concave: increasing because a test with a higher FPR can always say "member" more often, concave because randomizing between two tests reaches every point of the segment between them.

### Fact 3 — Beliefs multiply by the likelihood ratio

An attacker who believes beforehand that $z^*$ is a member with probability $p$, the **prior**, and then sees $y$, believes afterwards
$$\frac{\Pr[\text{member} \mid y]}{\Pr[\text{non-member} \mid y]} = \frac{p}{1 - p}\cdot\frac{Q(y)}{P(y)} = \frac{p}{1 - p}e^{L(y)}.$$
This is Bayes' rule in odds form: the posterior odds are the prior odds times the likelihood ratio, and the likelihood ratio is $e^L$. Converting odds $o$ back to a probability, $o/(1 + o)$, gives the posterior $p e^L/(p e^L + 1 - p)$.

### Fact 4 — DP-SGD's worst case is one pair of distributions, composed

Each step of DP-SGD adds Gaussian noise of standard deviation $\sigma C$ to a sum of per-example gradients clipped to length $C$, over a batch that contains each record independently with probability $q$. [DP-SGD](02-dp-sgd.md) showed that, for the target record, the worst case over models, data and directions is a one-dimensional pair in units of $C$: without the record the step's output is $\mathcal N(0, \sigma^2)$, with it the output is the mixture $(1 - q)\mathcal N(0, \sigma^2) + q\,\mathcal N(1, \sigma^2)$. The privacy loss of one step is $L_1(y) = \ln\big((1 - q) + q\,e^{(2y - 1)/(2\sigma^2)}\big)$, and $T$ steps release $T$ outputs whose losses add, so the PLD of the whole run is the distribution of a sum of $T$ such losses. Three accountants summarize it ([Choosing an accountant](../part04-privacy-accounting/09-choosing-an-accountant.md)):

- the **RDP accountant** bounds the Rényi divergence of order $\alpha$ by $\varepsilon(\alpha) = T\varepsilon_{q,\sigma}(\alpha)$ and converts it to $(\varepsilon, \delta)$; it is a valid upper bound, and it is what [DP-SGD](02-dp-sgd.md) used to set $\sigma$;
- the **numerical PLD accountant** discretizes the one-step loss on a grid of spacing $h$, rounding each loss up (pessimistic) or down (optimistic), and composes by fast Fourier transform (FFT) convolution; the truth lies between the two answers, which differ by at most a shift of $Th$ in the total loss ([Numerical accounting](../part04-privacy-accounting/06-numerical-accounting.md));
- the **Gaussian differential privacy (GDP)** central limit theorem (CLT) approximates the run by a single Gaussian pair $\mathcal N(0, 1)$ against $\mathcal N(\mu, 1)$ with $\mu = q\sqrt{T(e^{1/\sigma^2} - 1)}$; it is an approximation, not a bound ([f-DP and Gaussian DP](../part04-privacy-accounting/04-gaussian-dp.md)).

@tbl-epsc-accountants gives the noise multipliers of the running example and what the three accountants say about them.

| Target $\varepsilon$ (RDP) | $\sigma$ | RDP $\varepsilon$ | PLD $\varepsilon$, bracket | Grid $h$ | GDP $\mu$ | GDP $\varepsilon$ |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 7.72 | 0.500 | [0.445, 0.465] | $2\times10^{-6}$ | 0.130 | 0.454 |
| 1 | 4.13 | 0.999 | [0.903, 0.923] | $2\times10^{-6}$ | 0.246 | 0.909 |
| 2 | 2.28 | 1.998 | [1.825, 1.845] | $2\times10^{-6}$ | 0.461 | 1.819 |
| 4 | 1.36 | 3.993 | [3.629, 3.729] | $10^{-5}$ | 0.847 | 3.615 |
| 8 | 0.917 | 7.997 | [7.317, 7.417] | $10^{-5}$ | 1.511 | 7.116 |
| 16 | 0.695 | 15.955 | [14.582, 14.682] | $10^{-5}$ | 2.632 | 14.097 |

: The running example's noise multipliers at $q = 0.01$, $T = 10{,}000$, $\delta = 10^{-5}$, chosen by the RDP accountant, and the $\varepsilon$ each accountant reports for them. The upper end of the PLD bracket is the larger pessimistic value over the two neighbouring directions, the lower end the optimistic value of the direction a membership test uses. {#tbl-epsc-accountants}

Against the upper end of the PLD bracket, the RDP accountant overstates the true $\varepsilon$ by between 0.035 (at $\varepsilon = 0.5$) and 0.264 (at $\varepsilon = 4$), and by 0.58 at $\varepsilon = 8$, consistent with Part 4. Everything below uses these six values of $\sigma$, and calls each run by its target, "the $\varepsilon = 4$ run", even where the true $\varepsilon$ is a little smaller.

## What one (ε, δ) pair says about a membership test {#sec-epsc-pair}

### Context

A model card says "trained with $(\varepsilon, \delta) = (8, 10^{-5})$". Taken alone, what does that number forbid an attacker to do? This is the question to answer first, because it is the only one you can answer for a model whose algorithm you have not seen.

### Statement

If $M$ is $(\varepsilon, \delta)$-DP, every membership test, against every target, satisfies
$$\mathrm{TPR} \le \min\big\{1,\ e^\varepsilon\,\mathrm{FPR} + \delta,\ 1 - e^{-\varepsilon}(1 - \delta - \mathrm{FPR})\big\},$$ {#eq-epsc-pair}
and its **advantage**, $\mathrm{TPR} - \mathrm{FPR}$, the amount by which it beats guessing, satisfies
$$\mathrm{TPR} - \mathrm{FPR} \le \frac{e^\varepsilon - 1 + 2\delta}{e^\varepsilon + 1}.$$ {#eq-epsc-adv}

### Derivation

1. **The first line.** Apply the definition (Fact 1) with $D^- \cup \{z\}$ first and the event $S$ on which the test says "member": $Q(S) \le e^\varepsilon P(S) + \delta$, that is $\mathrm{TPR} \le e^\varepsilon\mathrm{FPR} + \delta$.
2. **The second line.** Apply it in the other order to the complement $S^c$, where the test says "non-member": $P(S^c) \le e^\varepsilon Q(S^c) + \delta$, that is $1 - \mathrm{FPR} \le e^\varepsilon(1 - \mathrm{TPR}) + \delta$. Solving for TPR gives $\mathrm{TPR} \le 1 - e^{-\varepsilon}(1 - \delta - \mathrm{FPR})$. A probability is at most 1, which is the third term of @eq-epsc-pair.
3. **The advantage.** $\mathrm{TPR} - \mathrm{FPR}$ is largest where the two lines of steps 1 and 2 cross, because below the crossing the first line rises faster than the diagonal (slope $e^\varepsilon \ge 1$) and above it the second line rises more slowly (slope $e^{-\varepsilon} \le 1$). Setting $e^\varepsilon a + \delta = 1 - e^{-\varepsilon}(1 - \delta - a)$ and multiplying by $e^\varepsilon$ gives $a(e^{2\varepsilon} - 1) = e^\varepsilon - 1 + \delta - \delta e^\varepsilon$, so $a = (1 - \delta)/(e^\varepsilon + 1)$. There the TPR is $e^\varepsilon(1 - \delta)/(e^\varepsilon + 1) + \delta$, and subtracting $a$ gives @eq-epsc-adv.

### Interpretation

**The bound has three regimes.** At small FPR the line $e^\varepsilon\mathrm{FPR} + \delta$ binds, and the ratio TPR/FPR is at most $e^\varepsilon$ plus $\delta/\mathrm{FPR}$. Below $\mathrm{FPR} \approx \delta e^{-\varepsilon}$ the $\delta$ term dominates, and the guarantee rests on $\delta$ alone. At large FPR the second line binds. And once $e^\varepsilon\mathrm{FPR} + \delta \ge 1$, which at $\varepsilon = 8$ happens from $\mathrm{FPR} = (1 - 10^{-5})e^{-8} = 3.35\times10^{-4}$ on, the bound allows a perfect true positive rate: the pair says nothing at all at the false positive rates where attacks are usually evaluated.

**Low FPR is where it matters.** [Membership inference](../part02-attacks/02-membership-inference.md) evaluates attacks by their TPR at an FPR of $10^{-3}$ or below, because an attack that flags members while almost never flagging non-members is one that confidently identifies people. At $\varepsilon = 1$ the pair caps that TPR at $e \times 10^{-3} + 10^{-5} = 0.002728$, a real constraint. At $\varepsilon = 4$ the cap is 0.05461, at $\varepsilon = 8$ it is 0.9997.

**The advantage is an average, and DP's guarantee is not about averages.** @eq-epsc-adv bounds the advantage at a single point of the curve, the one with equal error rates. It is 0.4621 at $\varepsilon = 1$ and 0.9640 at $\varepsilon = 4$, numbers that make $\varepsilon = 4$ sound hopeless. They are tight only for the worst mechanism with this $\varepsilon$, randomized response, whose curve is exactly the two lines. DP-SGD is not that mechanism, as the next section shows.

### Example: the pair at six budgets

@tbl-epsc-pair evaluates @eq-epsc-pair and @eq-epsc-adv for the six budgets of the running example. For $\varepsilon = 1$ and $\mathrm{FPR} = 0.01$, the first line gives $e \times 0.01 + 10^{-5} = 0.02719$.

| $\varepsilon$ | TPR at FPR $10^{-3}$ | TPR at FPR $10^{-2}$ | TPR at FPR $10^{-1}$ | Advantage |
| --- | --- | --- | --- | --- |
| 0.5 | 0.001659 | 0.0165 | 0.1649 | 0.2449 |
| 1 | 0.002728 | 0.02719 | 0.2718 | 0.4621 |
| 2 | 0.007399 | 0.0739 | 0.7389 | 0.7616 |
| 4 | 0.05461 | 0.546 | 0.9835 | 0.9640 |
| 8 | 0.9997 | 0.9997 | 0.9997 | 0.9993 |
| 16 | 1 | 1 | 1 | 1.0000 |

: The largest TPR and advantage of any membership test that the pair $(\varepsilon, 10^{-5})$ allows, from @eq-epsc-pair and @eq-epsc-adv. {#tbl-epsc-pair}

Read this way, the pair says that $\varepsilon \le 2$ is a strong guarantee everywhere, that $\varepsilon = 4$ is meaningful only at very low FPR, and that $\varepsilon \ge 8$ is no guarantee at all against membership inference. The rest of the chapter shows that, for DP-SGD, the second and third conclusions are far too pessimistic.

### Check yourself

- An audit reports TPR 0.02 at FPR 0.001 against a model claimed to be $(4, 10^{-5})$-DP. Does it refute the claim? *(No. The pair allows up to 0.05461 there. An audit refutes a claim only when the measured point lies above the line, and for $\varepsilon = 4$ that needs a TPR above about 5.5% at this FPR.)*
- Why can the advantage bound be close to 1 while the TPR at FPR $10^{-3}$ is tiny? *(They are different points of the curve. The advantage is read where the two error rates are equal, around FPR 0.27 for $\varepsilon = 1$; the low-FPR point is governed by the line $e^\varepsilon\mathrm{FPR} + \delta$, which is small there.)*

## The whole curve of DP-SGD {#sec-epsc-curve}

### Context

The pair $(\varepsilon, \delta)$ is one number out of a whole profile: the accountant could equally have reported $\varepsilon$ at $\delta = 10^{-6}$ or $10^{-3}$, and each of those pairs is another line that every test must satisfy. The best TPR an attacker can reach against DP-SGD is the curve that all those lines enclose, and for DP-SGD it can be computed. How far below the single line does it lie, and how well do cheaper accountants approximate it?

### Statement

For a mechanism with PLD $\omega$ (the law of $L$ under $Q$), the best TPR at FPR $a$ is reached by thresholding the total loss (Fact 2), and equals the lower envelope of all the lines of its profile:
$$\mathrm{TPR}^*(a) = \inf_{\varepsilon \in \mathbb R}\big(e^\varepsilon a + \delta(\varepsilon)\big), \qquad \delta(\varepsilon) = Q[L > \varepsilon] - e^\varepsilon P[L > \varepsilon].$$ {#eq-epsc-duality}
For DP-SGD the curve is traced exactly by
$$\big(\mathrm{FPR}(\tau), \mathrm{TPR}(\tau)\big) = \big(\mathbb E_\omega[e^{-L};\ L > \tau],\ \omega(L > \tau)\big), \qquad \tau \in \mathbb R,$$ {#eq-epsc-roc}
where $\omega$ is the $T$-fold convolution of the one-step PLD of Fact 4. Two cheaper curves bracket or approximate it. From the RDP curve $\varepsilon(\alpha)$ of the whole run, every test satisfies
$$\mathrm{TPR} \le \min_{\alpha > 1}\big(\mathrm{FPR}\cdot e^{\varepsilon(\alpha)}\big)^{(\alpha - 1)/\alpha},$$ {#eq-epsc-rdp}
a valid bound. The GDP central limit theorem approximates the curve by that of $\mathcal N(0, 1)$ against $\mathcal N(\mu, 1)$,
$$\mathrm{TPR}^*(a) \approx 1 - \Phi\big(\Phi^{-1}(1 - a) - \mu\big), \qquad \mu = q\sqrt{T\big(e^{1/\sigma^2} - 1\big)},$$ {#eq-epsc-gdp}
with $\Phi$ the standard normal distribution function; this is not a bound.

### Derivation

1. **Every line is valid.** For each $\varepsilon$, the mechanism is $(\varepsilon, \delta(\varepsilon))$-DP in this direction by the definition of the profile, so by @eq-epsc-pair every test has $\mathrm{TPR} \le e^\varepsilon\mathrm{FPR} + \delta(\varepsilon)$. Taking the smallest right-hand side over $\varepsilon$ gives "$\le$" in @eq-epsc-duality.
2. **The profile in terms of the loss.** For the event $\{L > \varepsilon\}$, $Q(L > \varepsilon) - e^\varepsilon P(L > \varepsilon) = \mathbb E_Q[(1 - e^{\varepsilon - L});\ L > \varepsilon]$ by the change of measure of Fact 2, and no other event does better, because adding an output with $L \le \varepsilon$ adds $Q(y) - e^\varepsilon P(y) = Q(y)(1 - e^{\varepsilon - L}) \le 0$ and removing one with $L > \varepsilon$ removes a positive amount. This is the formula for $\delta(\varepsilon)$.
3. **Each line touches the curve.** Take the Neyman–Pearson test with threshold $\tau$ and the line with $\varepsilon = \tau$. By step 2, $\delta(\tau) = \mathrm{TPR}(\tau) - e^\tau\mathrm{FPR}(\tau)$, so $\mathrm{TPR}(\tau) = e^\tau\mathrm{FPR}(\tau) + \delta(\tau)$: the test lies on the line. Since the tests with thresholds $\tau$ (and randomization between them) are the best ones by Fact 2, the infimum is attained and @eq-epsc-duality holds with equality.
4. **Composition adds losses.** The $T$ steps of DP-SGD use independent noise, so the likelihood ratio of all $T$ outputs is the product of the per-step ratios, and the total loss is the sum of the per-step losses. Even though each step's input depends on the earlier outputs, the worst-case pair of Fact 4 is the same at every step, so the per-step losses are independent and identically distributed under $Q$, and $\omega$ is the $T$-fold convolution of one step's PLD ([Numerical accounting](../part04-privacy-accounting/06-numerical-accounting.md)). Fact 2 then gives @eq-epsc-roc.
5. **Computing it with a guarantee.** Round each one-step loss up to the grid $h\mathbb Z$ and compose: the total loss is rounded up by at most $Th$, and the resulting pair is less private than the true one, so its curve lies above the true curve. Rounding down gives a curve below it. The script uses $h = 2\times10^{-6}$ for the three smallest budgets and $h = 10^{-5}$ for the others, so that the total loss is pinned within 0.02 and 0.1.
6. **The RDP bound.** The Rényi divergence can only shrink when both distributions are passed through the same test (data processing, [Rényi DP](../part04-privacy-accounting/01-renyi-dp.md)). After the test, the two distributions are coins with "member" probabilities TPR and FPR, so $e^{(\alpha - 1)\varepsilon(\alpha)} \ge \mathrm{TPR}^\alpha\mathrm{FPR}^{1 - \alpha} + (1 - \mathrm{TPR})^\alpha(1 - \mathrm{FPR})^{1 - \alpha}$. Dropping the second, non-negative term and solving for TPR gives $\mathrm{TPR}^\alpha \le e^{(\alpha - 1)\varepsilon(\alpha)}\mathrm{FPR}^{\alpha - 1}$, which is @eq-epsc-rdp after taking the power $1/\alpha$; it holds at every order, so at the best one.
7. **The GDP approximation.** For $\mathcal N(0, 1)$ against $\mathcal N(\mu, 1)$ the best test thresholds the output at $\Phi^{-1}(1 - a)$, and the shifted distribution exceeds that threshold with probability $1 - \Phi(\Phi^{-1}(1 - a) - \mu)$. The value of $\mu$ is the central-limit approximation of [f-DP and Gaussian DP](../part04-privacy-accounting/04-gaussian-dp.md#sec-gdp-sgd), which matches the mean and variance of the summed loss to leading order in $q$.

### Interpretation

**The pair overstates the attacker, and more so the larger $\varepsilon$.** @tbl-epsc-curve puts the four curves side by side at FPR $10^{-3}$. At $\varepsilon = 1$ the pair allows 0.002728 and the true curve is between 0.002209 and 0.00225, a factor of about 1.2. At $\varepsilon = 4$ the pair allows 0.05461 and the curve at most 0.01312, a factor of about 4. At $\varepsilon = 8$ the pair allows 0.9997 and the curve at most 0.0589, a factor of about 17. The reason is visible in @fig-epsc-tradeoff: the curve of DP-SGD is smooth and close to a Gaussian curve, and the line that the accountant reports at $\delta = 10^{-5}$ touches it only at a false positive rate of at most $e^{-\varepsilon}Q[L > \varepsilon]$, below $10^{-4}$ for every run here (@tbl-epsc-loss). Everywhere else the line is a loose summary.

**The RDP bound is the cheap middle ground.** @eq-epsc-rdp uses the same accountant that produced $\varepsilon$, costs nothing extra, and is a guarantee. At $\varepsilon = 8$ it caps the TPR at FPR $10^{-3}$ at 0.09016 instead of 0.9997. It is still looser than the exact curve, because one Rényi order summarizes the loss distribution by one moment, and because the conversion drops a term; the PLD keeps everything.

**GDP is close, but not on the safe side.** For budgets up to 8 the GDP curve lies inside the exact bracket at FPR $10^{-3}$, and within 0.001 of it at the other operating points: 0.05719 at $\varepsilon = 8$ and FPR $10^{-3}$, against the bracket [0.05555, 0.0589]. At $\varepsilon = 16$ it overshoots, 0.3234 against at most 0.2808, because $\mu = 2.632$ is far from the small-$q\sqrt T$ regime where the central limit theorem holds. Use it to explain, not to certify.

**The curve is the worst case of a strong attacker.** It is the best test of an attacker who knows every other record, the architecture, the initialization and every hyperparameter, sees every intermediate iterate, and targets a record whose clipped gradient is as large as clipping allows at every step on which it is sampled. Nasr et al. (2021) showed that an adversary who can choose the target's gradient and see every iterate does reach the accountant's guarantee, so the curve is attainable in that threat model. A typical record attacked through the final model alone is much safer. @tbl-epsc-attack measures it for one attack: the loss-threshold attack of [Membership inference](../part02-attacks/02-membership-inference.md), which calls a training record a member when its loss is below a threshold, run on one model trained at each budget, with the 10,000 training records as members and the 20,000 test records as non-members.

| $\varepsilon$ | Test accuracy | TPR at FPR $10^{-3}$ | TPR at FPR $10^{-2}$ | TPR at FPR $10^{-1}$ | Best advantage |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 0.8186 | 0.0010 | 0.0102 | 0.0989 | 0.0155 |
| 1 | 0.9266 | 0.0013 | 0.0108 | 0.1019 | 0.0085 |
| 2 | 0.9502 | 0.0011 | 0.0096 | 0.0983 | 0.0059 |
| 4 | 0.9538 | 0.0008 | 0.0072 | 0.0963 | 0.0116 |
| 8 | 0.9584 | 0.0007 | 0.0098 | 0.1007 | 0.0162 |
| 16 | 0.9632 | 0.0013 | 0.0102 | 0.0992 | 0.0158 |
| Non-private | 0.9597 | 0.0004 | 0.0099 | 0.0956 | 0.0221 |

: A loss-threshold membership attack on the running example, one trained model per budget. Members are the 10,000 training records, non-members the 20,000 test records. The best advantage is the largest TPR − FPR over 999 thresholds. {#tbl-epsc-attack}

The attack is at chance at every budget, and nearly at chance against the non-private model too: the task is noiseless and the network generalizes, so an average record's loss says little about membership. That is a statement about this attack, these records and this task, not a guarantee; [Auditing private training](10-auditing.md) builds the attacks that come close to the curve, with canaries chosen to be as exposed as possible.

![The largest TPR of a membership test against DP-SGD on the running example, on log axes, at $\varepsilon = 1$ (left) and $\varepsilon = 8$ (right), $\delta = 10^{-5}$: the bound from the single pair (dotted), the bound from the whole RDP curve (dash-dotted), the exact curve from the pessimistic PLD (solid red), and the GDP approximation (dashed). The grey diagonal is an attacker with no information.](figures/epsc_tradeoff.svg){#fig-epsc-tradeoff}

### Example: the curve at six budgets

@tbl-epsc-curve gives the four answers at FPR $10^{-3}$ and @tbl-epsc-curve-more the exact bracket at higher FPR and for the advantage.

| $\varepsilon$ | $\sigma$ | Pair, @eq-epsc-pair | RDP, @eq-epsc-rdp | Exact (PLD bracket) | GDP, @eq-epsc-gdp |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 7.72 | 0.001659 | 0.00161 | [0.001524, 0.001554] | 0.001537 |
| 1 | 4.13 | 0.002728 | 0.002427 | [0.002209, 0.00225] | 0.002224 |
| 2 | 2.28 | 0.007399 | 0.005033 | [0.004267, 0.004342] | 0.004273 |
| 4 | 1.36 | 0.05461 | 0.01668 | [0.01215, 0.01312] | 0.01244 |
| 8 | 0.917 | 0.9997 | 0.09016 | [0.05555, 0.0589] | 0.05719 |
| 16 | 0.695 | 1 | 0.4971 | [0.2714, 0.2808] | 0.3234 |

: The largest TPR of a membership test against the running example's DP-SGD at FPR $10^{-3}$, $\delta = 10^{-5}$: from the pair, from the RDP curve, exactly (optimistic and pessimistic PLD), and from the GDP approximation. {#tbl-epsc-curve}

| $\varepsilon$ | Exact TPR at FPR $10^{-2}$ | Exact TPR at FPR $10^{-1}$ | Exact advantage | Pair advantage |
| --- | --- | --- | --- | --- |
| 0.5 | [0.01391, 0.01418] | [0.1236, 0.126] | [0.0472, 0.0567] | 0.2449 |
| 1 | [0.01859, 0.01893] | [0.1489, 0.1515] | [0.0933, 0.1023] | 0.4621 |
| 2 | [0.03088, 0.03141] | [0.2043, 0.2076] | [0.1778, 0.1859] | 0.7616 |
| 4 | [0.06739, 0.0724] | [0.3211, 0.3416] | [0.3094, 0.3429] | 0.9640 |
| 8 | [0.1987, 0.2092] | [0.5682, 0.5901] | [0.5271, 0.5498] | 0.9993 |
| 16 | [0.5447, 0.5588] | [0.8637, 0.8752] | [0.7647, 0.7758] | 1.0000 |

: The exact trade-off curve of the running example's DP-SGD at two more false positive rates, and its largest advantage, against the advantage the pair allows. {#tbl-epsc-curve-more}

The advantage tells the same story as the low-FPR column. The pair allows 0.9640 at $\varepsilon = 4$; DP-SGD allows at most 0.3429. @fig-epsc-curves draws the six exact curves together. Each is nearly a straight line on log axes at low FPR, with a slope a little below 1, which means TPR/FPR, the factor by which the attacker's flag is more likely on a member, grows slowly as FPR falls: at $\varepsilon = 8$ it is about 21 at FPR $10^{-2}$ and about 59 at $10^{-3}$, nowhere near $e^8 \approx 2{,}981$.

![The exact trade-off curves (pessimistic PLD) of DP-SGD on the running example at the six budgets, on log axes. The dotted diagonal is an attacker with no information.](figures/epsc_curves.svg){#fig-epsc-curves}

### Check yourself

- Why does the line $e^\varepsilon\mathrm{FPR} + \delta$ at $\delta = 10^{-5}$ touch the curve only at a tiny FPR? *(By step 3 it touches the curve at the Neyman–Pearson test with threshold $\tau = \varepsilon$, whose FPR is $P[L > \varepsilon] \le e^{-\varepsilon}Q[L > \varepsilon]$, because $P = e^{-L}Q$ and $L > \varepsilon$ on that event. The accountant chose $\varepsilon$ so that the loss rarely exceeds it, so that FPR is tiny.)*
- A colleague proposes to report only the GDP $\mu$ of a run with $\mu = 2.6$. What do you tell them? *(That the GDP curve is an approximation that can be too optimistic, as at $\varepsilon = 16$ here, or too pessimistic. Report the PLD curve or the RDP bound as the guarantee, and $\mu$ as an explanation.)*

## The Bayesian reading {#sec-epsc-bayes}

### Context

People who are not statisticians rarely think in error rates. They ask "if someone sees the model, how sure can they be that I was in the data?" That is a question about posterior probabilities. What do $\varepsilon$, and the actual curve of DP-SGD, say about it, and why does the answer depend so much on the prior?

### Statement

If every output has privacy loss at most $\varepsilon$, then for every prior $p$ the posterior probability of membership is at most
$$\Pr[\text{member} \mid y] \le \frac{p\,e^\varepsilon}{p\,e^\varepsilon + 1 - p}.$$ {#eq-epsc-posterior}
Under $(\varepsilon, \delta)$-DP this holds at a level $\varepsilon' > \varepsilon$ except on outputs whose total probability is at most $\delta/(1 - e^{\varepsilon - \varepsilon'})$. For DP-SGD the posterior after the run is exactly $p\,e^L/(p\,e^L + 1 - p)$, with $L$ drawn from the PLD $\omega$. An attacker who acts only on a flag ("the test said member") is right with probability
$$\Pr[\text{member} \mid \text{flag}] = \frac{p\,\mathrm{TPR}}{p\,\mathrm{TPR} + (1 - p)\,\mathrm{FPR}} \le \frac{p\,(e^\varepsilon\mathrm{FPR} + \delta)}{p\,(e^\varepsilon\mathrm{FPR} + \delta) + (1 - p)\,\mathrm{FPR}}.$$ {#eq-epsc-precision}

### Derivation

1. **The posterior is a function of the loss.** By Fact 3, the posterior odds are $\frac{p}{1 - p}e^{L(y)}$, and the posterior is $pe^L/(pe^L + 1 - p)$. This is increasing in $L$.
2. **Pure bound.** If $L \le \varepsilon$, substitute $\varepsilon$ for $L$: @eq-epsc-posterior.
3. **Rare outputs.** Let $S$ be the outputs with $L > \varepsilon'$. On $S$, $P(y) < e^{-\varepsilon'}Q(y)$, so $P(S) < e^{-\varepsilon'}Q(S)$. The definition gives $Q(S) \le e^\varepsilon P(S) + \delta < e^{\varepsilon - \varepsilon'}Q(S) + \delta$, so $Q(S)(1 - e^{\varepsilon - \varepsilon'}) < \delta$ ([The definition of differential privacy](../part03-dp-foundations/01-definition.md#sec-dpdef-bayes)).
4. **The precision of a flag.** Bayes' rule with the event "flag" in place of $y$: the flag has probability TPR for a member and FPR for a non-member, so the posterior odds are $\frac{p}{1 - p}\cdot\frac{\mathrm{TPR}}{\mathrm{FPR}}$. Bounding TPR by @eq-epsc-pair, which can only increase the odds, gives @eq-epsc-precision. With $\delta = 0$ the bound equals @eq-epsc-posterior: a flag can never be more convincing than the most convincing single output.

### Interpretation

**DP bounds the change of belief, not the belief.** Small priors stay small at small $\varepsilon$. @tbl-epsc-posterior gives @eq-epsc-posterior at three priors. At $\varepsilon = 1$ an attacker who thinks there is a 1% chance that you are in the data can reach at most 2.67%. At $\varepsilon = 4$ it can reach 35.55%, at $\varepsilon = 8$ 96.79%: read through the pure bound, $\varepsilon = 8$ permits near certainty.

| $\varepsilon$ | Prior 0.01 | Prior 0.1 | Prior 0.5 |
| --- | --- | --- | --- |
| 0.5 | 0.0164 | 0.1548 | 0.6225 |
| 1 | 0.0267 | 0.2320 | 0.7311 |
| 2 | 0.0695 | 0.4509 | 0.8808 |
| 4 | 0.3555 | 0.8585 | 0.9820 |
| 8 | 0.9679 | 0.9970 | 0.9997 |
| 16 | 1.0000 | 1.0000 | 1.0000 |

: The largest posterior probability of membership when the privacy loss is at most $\varepsilon$, @eq-epsc-posterior. {#tbl-epsc-posterior}

**The realized loss is usually much smaller than $\varepsilon$.** The bound is attained only by outputs whose loss is $\varepsilon$. For DP-SGD, the loss is a random variable, and @tbl-epsc-loss reads its distribution off the pessimistic PLD for a member: the median, and the values exceeded with probability $10^{-3}$ and $10^{-5}$. In the $\varepsilon = 8$ run, half of all members have total loss at most 1.136, and the posterior of an attacker with prior 1% moves to at most 0.0305 for them; only one member in a thousand has loss above 5.971 (posterior 0.7984), and one in a hundred thousand above 7.910 (posterior 0.9649). @fig-epsc-posterior draws these curves against the prior.

| Run | Median loss | Loss exceeded w.p. $10^{-3}$ | Loss exceeded w.p. $10^{-5}$ | $Q[L > \varepsilon]$ | Posterior from prior 0.01: median, $10^{-3}$, $10^{-5}$ |
| --- | --- | --- | --- | --- | --- |
| $\varepsilon = 0.5$ | 0.018 | 0.421 | 0.575 | $1.10\times10^{-4}$ | 0.0102, 0.0152, 0.0176 |
| $\varepsilon = 1$ | 0.040 | 0.802 | 1.093 | $5.04\times10^{-5}$ | 0.0104, 0.0220, 0.0293 |
| $\varepsilon = 2$ | 0.115 | 1.549 | 2.099 | $2.52\times10^{-5}$ | 0.0112, 0.0454, 0.0762 |
| $\varepsilon = 4$ | 0.401 | 3.059 | 4.092 | $1.59\times10^{-5}$ | 0.0149, 0.1771, 0.3768 |
| $\varepsilon = 8$ | 1.136 | 5.971 | 7.910 | $7.86\times10^{-6}$ | 0.0305, 0.7984, 0.9649 |
| $\varepsilon = 16$ | 3.032 | 11.661 | 15.314 | $3.78\times10^{-6}$ | 0.1732, 0.9991, 1.0000 |

: The total privacy loss of a worst-case member in the running example's DP-SGD runs, from the pessimistic PLD (so slightly overstated), and the posterior it produces from a prior of 1%. {#tbl-epsc-loss}

This is the Bayesian face of the gap between the line and the curve. The pure bound treats every output as the worst one; the actual loss is concentrated far below $\varepsilon$, with a thin tail that reaches it. The column $Q[L > \varepsilon]$ shows how thin: the probability that the run's loss for the worst-case member exceeds the reported $\varepsilon$ is of the order of $\delta$, between $3.78\times10^{-6}$ and $1.10\times10^{-4}$.

**Base rates make flags unconvincing.** In @eq-epsc-precision the attacker's precision depends on the prior through $p/(1 - p)$, and at a small prior even a strong ratio TPR/FPR leaves most flags wrong. At $\varepsilon = 8$, prior 1% and FPR $10^{-2}$, the pair allows a precision of 0.5024, while the exact curve allows 0.1744: five flags out of six are false. At $\varepsilon = 4$ and FPR $10^{-3}$, the pair allows 0.3555 and the curve 0.1170. Who the attacker is matters: an insider who already knows that you are one of two candidates has prior 0.5; a stranger scanning a population has a prior near the fraction of the population in the dataset.

![Left: the largest posterior probability of membership against the prior, from @eq-epsc-posterior, for the six budgets (log axes). Right: for the $\varepsilon = 8$ run, the bound (dotted) and the posterior of a member whose total privacy loss is the median of the PLD, or is exceeded with probability $10^{-3}$ or $10^{-5}$.](figures/epsc_posterior.svg){#fig-epsc-posterior}

### Example: an insider and a stranger

An insider knows that the training set is one of two: with you, or with your colleague in your place. Its prior is 0.5. Against the $\varepsilon = 2$ run, @tbl-epsc-posterior caps its posterior at 0.8808 on the worst output. Against the same run, a stranger with prior 0.01 reaches at most 0.0695 on the worst output and, for a typical member, $0.01e^{0.115}/(0.01e^{0.115} + 0.99) = 0.0112$, the value in @tbl-epsc-loss. The guarantee is the same; what it protects against depends on how much the attacker knew before.

### Check yourself

- Can DP make an attacker less sure than its prior? *(Yes, for outputs with negative loss, which are typical for non-members. @eq-epsc-posterior bounds the increase; the lower bound $pe^{-\varepsilon}/(pe^{-\varepsilon} + 1 - p)$ bounds the decrease.)*
- Why is $Q[L > \varepsilon]$ for the $\varepsilon = 8$ run below $\delta = 10^{-5}$, while for the $\varepsilon = 4$ run it is above? *(Because $\delta(\varepsilon) = \mathbb E_Q[(1 - e^{\varepsilon - L})_+]$ is at most $Q[L > \varepsilon]$, but not equal to it; and because the RDP accountant overstates $\varepsilon$, by more at $\varepsilon = 8$, so the true loss rarely reaches the reported value.)*

## Reconstruction {#sec-epsc-recon}

### Context

Membership is a demanding standard: the attacker is given the target record and only has to decide whether it was used. Many harms need more: recovering a record the attacker does not know, such as a patient's diagnosis or a person's face. Balle, Cherubin & Hayes (2022) formalize this as **reconstruction robustness**: an informed adversary knows every training record except the target $Z$, knows a prior $\pi$ from which $Z$ was drawn, sees the trained model, and outputs a guess; it succeeds if the guess is within a distance $\xi$ of $Z$, its error radius. The baseline is the best success without seeing the model,
$$\kappa = \sup_{z_0}\Pr_{Z \sim \pi}\big[\mathrm{err}(Z, z_0) \le \xi\big],$$
for example $\kappa = 1/m$ when $Z$ is uniform over $m$ candidates and only an exact guess counts. How much can the model raise the success above $\kappa$?

### Statement

If every neighbouring pair of $M$ has trade-off curve at or below $\mathrm{TPR}^*$, then every reconstruction attack succeeds with probability at most
$$\Pr[\text{success}] \le \mathrm{TPR}^*(\kappa).$$ {#eq-epsc-rero}
With the pair $(\varepsilon, \delta)$ this is $e^\varepsilon\kappa + \delta$; with pure $\varepsilon$-DP, $e^\varepsilon\kappa$, and with one RDP order, $(\kappa e^{\varepsilon(\alpha)})^{(\alpha - 1)/\alpha}$, which are Balle, Cherubin & Hayes's bounds; with the exact curve of DP-SGD it is the bound of Hayes, Mahloujifar & Balle (2023), who write it as a "blow-up function", the largest probability under $Q$ of an event whose probability under $P$ is at most $\kappa$, which is the trade-off curve under another name.

### Derivation

1. **A reconstruction attack defines one test per candidate.** Fix $D^-$ and the attack $\mathcal A$. For a candidate $z$, let $S_z$ be the set of models $\theta$ for which $\mathcal A(\theta)$ is within $\xi$ of $z$. The attack succeeds on $z$ exactly when the model trained on $D^- \cup \{z\}$ falls in $S_z$.
2. **Each test is bounded by the curve.** "Say member when $\theta \in S_z$" is a membership test for $z$ against the pair $Q_z = M(D^- \cup \{z\})$, $P = M(D^-)$. Its FPR is $a_z = P(S_z)$, so its TPR satisfies $Q_z(S_z) \le \mathrm{TPR}^*(a_z)$.
3. **Average over the prior.** The success probability is $\mathbb E_{Z\sim\pi}[Q_Z(S_Z)] \le \mathbb E_Z[\mathrm{TPR}^*(a_Z)] \le \mathrm{TPR}^*(\mathbb E_Z[a_Z])$, by Jensen's inequality, since the curve is concave (Fact 2).
4. **The average FPR is at most the baseline.** $\mathbb E_Z[a_Z] = \Pr_{Z\sim\pi,\,\theta\sim M(D^-)}[\mathrm{err}(Z, \mathcal A(\theta)) \le \xi]$. Here $\theta$ is trained without $Z$, so it is independent of $Z$, and for each fixed $\theta$ the guess $\mathcal A(\theta)$ is a fixed $z_0$, which succeeds with probability at most $\kappa$. So $\mathbb E_Z[a_Z] \le \kappa$, and since the curve is increasing, @eq-epsc-rero follows.
5. **The special cases.** Substitute the line of @eq-epsc-pair, or the RDP bound @eq-epsc-rdp at one order, for $\mathrm{TPR}^*$.

The $(\varepsilon, \delta)$ form $e^\varepsilon\kappa + \delta$ is this chapter's corollary; Balle, Cherubin & Hayes state the pure-DP, RDP and zero-concentrated cases.

### Interpretation

**Reconstruction is membership at FPR κ.** The derivation turns any reconstruction attack into a membership test whose false positive rate is the baseline. So the trade-off curve, read at FPR $1/m$, answers "can the attacker pick the right one of $m$ candidates?". A large prior uncertainty is a small FPR, and that is where the curve of DP-SGD is far below the line.

**Large budgets can still prevent reconstruction.** @tbl-epsc-recon reads the curves of @fig-epsc-curves at $\kappa = 1/m$. At $\varepsilon = 8$ the pair gives no bound for any $m$ up to 1,000, but the curve caps the success of choosing among 1,000 equally likely candidates at 0.0589, against 0.001 by guessing. Balle, Cherubin & Hayes observed the same in experiments: in their words, DP with large values of $\varepsilon$ can protect against reconstruction attacks if the target is high-dimensional and the adversary's prior contains a large degree of uncertainty. Hayes, Mahloujifar & Balle found their bound close to their best attack in all their settings, and that different DP-SGD hyperparameters with the same $\varepsilon$ can give quite different reconstruction success: the curve, not $\varepsilon$, is what matters.

**Small priors are the honest assumption only if you can defend them.** If the attacker can narrow the target to two candidates, $\kappa = 0.5$ and reconstruction is as easy as membership at FPR 0.5. A defensible reconstruction claim states the prior: "an attacker who knows the target is one of $m$ records".

Guo, Karrer, Chaudhuri & van der Maaten (2022) bound reconstruction differently, for continuous targets: under RDP of order 2 at level $\varepsilon$, any unbiased reconstruction of a target that ranges over an interval of width $w$ has expected squared error at least $w^2/(4(e^\varepsilon - 1))$, and Fisher information gives a tighter, data-dependent bound. The bound needs unbiasedness: for small $\varepsilon$ a constant guess at the middle of the interval does better. @sec-apx-audit-recon in the appendices derives the bound and shows why it needs that condition.

### Example: one of m candidates

| $\varepsilon$ | $m = 10$: pair | $m = 10$: curve | $m = 100$: pair | $m = 100$: curve | $m = 1{,}000$: pair | $m = 1{,}000$: curve |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.1649 | 0.126 | 0.0165 | 0.01418 | 0.001659 | 0.001554 |
| 1 | 0.2718 | 0.1515 | 0.02719 | 0.01893 | 0.002728 | 0.00225 |
| 2 | 0.7389 | 0.2076 | 0.0739 | 0.03141 | 0.007399 | 0.004342 |
| 4 | 1 | 0.3416 | 0.546 | 0.0724 | 0.05461 | 0.01312 |
| 8 | 1 | 0.5901 | 1 | 0.2092 | 1 | 0.0589 |
| 16 | 1 | 0.8752 | 1 | 0.5588 | 1 | 0.2808 |

: The largest probability of reconstructing a record drawn uniformly from $m$ candidates (baseline $1/m$), from the pair $e^\varepsilon\kappa + \delta$ and from the exact (pessimistic) curve of the running example's DP-SGD. {#tbl-epsc-recon}

Ten candidates is a strong prior: at $\varepsilon = 4$ the curve allows the attacker 0.3416 against a baseline of 0.1. At a thousand candidates even $\varepsilon = 16$ keeps the success at 0.2808.

### Check yourself

- Why does the derivation need the curve to be concave? *(To move the average over candidates inside: the average of the per-candidate bounds is at most the bound at the average FPR. Without concavity, a few candidates with large FPR could do better than their average.)*
- An attacker knows a patient's record except one binary field. What is $\kappa$, and what does $\varepsilon = 4$ guarantee? *(If both values are equally likely a priori, $\kappa = 0.5$; the exact curve at FPR 0.5 is far above 0.5, so little. Attribute inference with a strong prior is membership inference in disguise.)*

## Interpreting δ {#sec-epsc-delta}

### Context

Every number so far was at $\delta = 10^{-5}$. What does $\delta$ mean, how small must it be, and what does a smaller one cost in $\varepsilon$ for DP-SGD?

### Statement

The mechanism that publishes each record in full, independently with probability $\delta$, is $(0, \delta)$-DP and publishes at least one of $n$ records with probability
$$1 - (1 - \delta)^n \approx 1 - e^{-n\delta}.$$ {#eq-epsc-release}
So $\delta = 1/n$ permits exposing a record with probability about 0.63, and a meaningful guarantee needs $n\delta \ll 1$. For a Gaussian-type mechanism such as DP-SGD, $\delta$ is not a probability of catastrophe: the privacy loss has a light tail, $\delta(\varepsilon) \le Q[L > \varepsilon]$, and shrinking $\delta$ by orders of magnitude raises $\varepsilon$ only moderately.

### Derivation

1. **The release mechanism is $(0, \delta)$-DP.** Run it on $D^- \cup \{z^*\}$ by running it on $D^-$ and tossing one more coin for $z^*$. The two outputs differ only when that coin publishes $z^*$, which has probability $\delta$, so for every event $S$ the two probabilities differ by at most $\delta$ in either direction ([The definition of differential privacy](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta)).
2. **Exposure.** Each record escapes with probability $1 - \delta$, independently, so all escape with probability $(1 - \delta)^n = e^{n\ln(1 - \delta)} \approx e^{-n\delta}$, which gives @eq-epsc-release.
3. **$\delta$ is at most the tail of the loss.** $\delta(\varepsilon) = \mathbb E_Q[(1 - e^{\varepsilon - L})_+]$ (step 2 of @sec-epsc-curve), and the integrand is at most 1 and is 0 unless $L > \varepsilon$.
4. **Smaller $\delta$ for DP-SGD.** Run the RDP conversion of [DP-SGD](02-dp-sgd.md) at several $\delta$ for the same $\sigma$; @tbl-epsc-delta and @fig-epsc-delta show the result.

### Interpretation

**Two kinds of $\delta$.** In the release mechanism, $\delta$ is spent at infinite loss: the published record is identified with certainty. In DP-SGD it is spent at moderate losses, just above $\varepsilon$, and the loss distribution has a Gaussian-like tail (@tbl-epsc-loss). That is why NIST's guidelines recommend, where possible, guarantees such as Rényi DP, zero-concentrated differential privacy (zCDP) or GDP, which do not allow catastrophic failure, and why a full privacy profile, or the curve, is better than any one pair.

**How small.** Near, Darais, Lefkovitz & Howarth (NIST SP 800-226, 2025) write that a common recommendation is $\delta \le 1/n^2$, that $\delta \le 1/(n\ln n)$ is another sensible choice, and that values of $\delta$ exceeding $10^{-5}$ are suspicious and should be justified. The running example, with $n = 10{,}000$ and $\delta = 10^{-5}$, has $n\delta = 0.1$: acceptable for DP-SGD, whose $\delta$ is not catastrophic, but not by the $1/n^2 = 10^{-8}$ rule. @tbl-epsc-delta shows what the stricter rule costs: the $\varepsilon = 4$ run is $(5.209, 10^{-8})$-DP.

**The unit decides n.** If one person can contribute many records, $n$ is the number of people and $\delta$ must be small relative to that; and the per-record guarantee is not a per-person guarantee at all ([User-level differential privacy](07-user-level.md)).

| $\sigma$ (run) | $\delta = 10^{-4}$ | $10^{-5}$ | $10^{-6}$ | $10^{-8}$ | $10^{-10}$ |
| --- | --- | --- | --- | --- | --- |
| 4.13 ($\varepsilon = 1$) | 0.852 | 0.999 | 1.129 | 1.354 | 1.548 |
| 1.36 ($\varepsilon = 4$) | 3.501 | 3.993 | 4.432 | 5.209 | 5.890 |
| 0.917 ($\varepsilon = 8$) | 7.088 | 7.997 | 8.821 | 10.301 | 11.627 |

: The $\varepsilon$ of three DP-SGD runs of the running example at five values of $\delta$, from the RDP accountant. {#tbl-epsc-delta}

![The privacy profile $\varepsilon(\delta)$ of three DP-SGD runs of the running example, from the RDP accountant. The dashed line marks $\delta = 1/n$.](figures/epsc_delta_profile.svg){#fig-epsc-delta}

### Example: ten thousand records

For $n = 10{,}000$, @eq-epsc-release gives an exposure probability of 0.6321 at $\delta = 10^{-4} = 1/n$, 0.09516 at $\delta = 10^{-5}$ and 0.00995 at $10^{-6}$ for the release mechanism. For DP-SGD the same tenfold reductions of $\delta$ cost little: from $10^{-5}$ to $10^{-6}$ the $\varepsilon = 4$ run's budget rises from 3.993 to 4.432, and down to $10^{-10}$ it reaches 5.890. Every factor of 100 in $\delta$ costs between about 0.2 and 0.9 in $\varepsilon$ for these two runs. When in doubt, use a smaller $\delta$; it is cheap.

### Check yourself

- A model is reported as $(1, 10^{-3})$-DP on 50,000 records. What is wrong? *(It allows $n\delta = 50$: the release mechanism with this $\delta$ would publish about 50 records. Unless the loss distribution is shown to be light-tailed, the guarantee permits catastrophic failure for some people.)*
- Why does the $\varepsilon$ in @tbl-epsc-delta grow roughly like $\sqrt{\ln(1/\delta)}$? *(For a nearly Gaussian loss with mean $\bar L$ and variance $v$, the tail beyond $\varepsilon$ falls like $e^{-(\varepsilon - \bar L)^2/(2v)}$, so reaching a tail of $\delta$ needs $\varepsilon - \bar L$ of order $\sqrt{2v\ln(1/\delta)}$.)*

## Deployed values and their units {#sec-epsc-deployed}

### Context

"Industry uses $\varepsilon$ between 1 and 10" is a common justification. What have real deployments published, and in which units? An $\varepsilon$ means nothing without the **privacy unit** (what one neighbouring change is: one record, one user's data for a day, one user's data forever) and the **period** over which the budget is spent.

### Statement

@tbl-epsc-deployed lists deployments whose parameters are documented in a primary source. The privacy unit and period differ from row to row, and so do the accounting notions, so the $\varepsilon$ values are not comparable without conversion. The zCDP guarantees convert to $(\varepsilon, \delta)$ by the simple bound
$$\varepsilon = \rho + 2\sqrt{\rho\ln(1/\delta)},$$ {#eq-epsc-zcdp}
of Bun & Steinke (2016), and tighter numerical conversions give somewhat smaller values; pure $\varepsilon$-DP guarantees for independent reports compose by adding. The Gboard models were trained by follow the regularized leader (FTRL) with noise added to sums of gradients, DP-FTRL ([DP-FTRL and matrix factorization](04-matrix-mechanisms.md)), in federated learning, where the data stay on users' devices.

| Deployment | Mechanism and accounting | Privacy unit and period | Published parameters |
| --- | --- | --- | --- |
| US Census Bureau, 2020 Census redistricting data | Discrete Gaussian, zCDP | One person's (or housing unit's) record, replace-one; one release | $\rho = 2.63$ in total, 2.56 for persons and 0.07 for housing units; reported as $\varepsilon = 17.14$ (persons) plus 2.47 (housing), 19.61 in total, at $\delta = 10^{-10}$ |
| US Census Bureau, 2020 Demographic and Housing Characteristics file | Discrete Gaussian, zCDP | As above | $\rho \approx 4.96$ for persons, $\approx 7.70$ for households |
| Google Gboard language models | DP-FTRL in federated learning, zCDP | One device's data, whole training run | $\rho$ from 0.25 to 1.86 across the models of the paper's privacy statement; $\rho = 0.25$ is $\varepsilon = 4.49$ and $\rho = 1.86$ is $\varepsilon = 13.69$ at $\delta = 10^{-10}$ (numerical conversion) |
| Google Gboard, Spanish model (2022) | As above | As above | $\rho = 0.81$, $(8.9, 10^{-10})$; later models down to $\varepsilon = 0.994$ |
| Apple, local DP on iOS and macOS | Local randomizers, pure $\varepsilon$ | One donation; limited donations per day; data kept up to three months | Emoji $\varepsilon = 4$, one donation per day; QuickType $\varepsilon = 8$, two per day; Health types $\varepsilon = 2$, one per day; Lookup Hints and Safari domains $\varepsilon = 4$, two per day |
| Google RAPPOR in Chrome (homepages) | Local randomized response, pure $\varepsilon$ | One report ($\varepsilon_1$); one unchanging value over all reports ($\varepsilon_\infty$) | $\varepsilon_1 = 0.5343$; $\varepsilon_\infty = 2.043$ from the paper's formula |
| Microsoft Windows app-usage telemetry | Local one-bit mean, pure $\varepsilon$, with memoization and $\alpha$-point rounding | One user, one collection round every 6 hours; across rounds, only changes of behaviour are covered | $\varepsilon = 1$ per round and app; 0.686 per round after output perturbation; 1.672 per round across all apps |
| LinkedIn Audience Engagements API | Gaussian and Laplace noise on counts | One member, per analyst, per month | $(0.15, 10^{-10})$ per returned result; $(34.9, 7\times10^{-9})$ per month |
| Facebook URLs dataset for researchers | Gaussian noise, zCDP | One action (one user, one URL, one kind of action); user level, for each kind of action, for users at or below its 99th percentile of URLs | Per action $\varepsilon$ from 0.00 (views) to 0.45 (kinds that users rarely repeat, such as comments) at $\delta = 10^{-5}$; per user and kind of action $\rho$ about 0.005, chosen so that $\varepsilon = 0.45$ at $\delta = 10^{-5}$ |

: Published differential-privacy parameters of real deployments, with their privacy units. Sources are in the references; values are as stated there, except the RAPPOR $\varepsilon_\infty$, computed here from the paper's formula. The Facebook values are those of the release's table of noise per kind of action. {#tbl-epsc-deployed}

### Derivation: making the rows comparable

1. **zCDP to $(\varepsilon, \delta)$.** @eq-epsc-zcdp with $\rho = 2.56$ and $\delta = 10^{-10}$ gives $2.56 + 2\sqrt{2.56 \times 23.026} = 17.915$, the persons' figure of Kazan & Reiter (2022); the Bureau's 17.14 comes from a tighter conversion. For the Spanish Gboard model, $\rho = 0.81$ gives 9.447 by @eq-epsc-zcdp, against the 8.9 Google reports from a numerical conversion. The Bureau's 19.61 adds the $\varepsilon$ of persons and housing units; converting the summed $\rho = 2.63$ once would give a smaller number (18.194 by @eq-epsc-zcdp).
2. **Replace-one to add/remove.** The Census guarantee is for replacing one person's record, which is a removal followed by an addition, so its $\varepsilon$ is not directly comparable with the add/remove $\varepsilon$ of DP-SGD; an add/remove guarantee of $\varepsilon$ implies a replace-one guarantee of $2\varepsilon$ ([The definition of differential privacy](../part03-dp-foundations/01-definition.md#sec-dpdef-neighbours)).
3. **Local reports over time.** A local randomizer with $\varepsilon$ per donation, run a number $k$ of times, has a total loss of at most $k\varepsilon$ by basic composition, and a person whose data are correlated across reports can lose all of it. Apple's documentation states $\varepsilon$ per donation and the number of donations per day, so QuickType's two donations of $\varepsilon = 8$ are up to 16 per day. Tang, Korolova et al. (2017) measured macOS 10.12 independently, by reverse-engineering it. They found a privacy loss of 1 or 2 per submitted datum. Because many data are submitted each day, the loss reached 16 per day for four applications, namely Emojis, New words, Deeplinks and Lookup Hints. They also found that the budget renews every day, so the total grows with the number of days since a user opted in. Tang et al.'s figures are for macOS 10.12 and count each submitted datum, while Apple's documentation states $\varepsilon$ per donation, so the two need not match. RAPPOR's memoization makes all the reports of a value that does not change cost at most its permanent level $\varepsilon_\infty$, which depends on the number $H$ of hash functions and the probability $f$ of replacing a bit by a coin,
$$\varepsilon_\infty = 2H\ln\frac{1 - f/2}{f/2}.$$
With $H = 2$ and $f = 0.75$ that is $4\ln(5/3) = 2.043$. But a value that changes is reported afresh ([Local, shuffle and distributed trust models](../part03-dp-foundations/09-trust-models.md)).
4. **Microsoft's rounds.** Ding, Kulkarni & Yekhanin (2017) collect one bit per app every six hours at $\varepsilon = 1$. To blur when a report changes, they flip each sent bit with a probability $\nu = 0.2$. They report 0.686 for one round after the flip, which they call output perturbation. Their own bound for one round after the flip is
$$\varepsilon' = \ln\frac{(1 - 2\nu)e^{\varepsilon}/(e^{\varepsilon} + 1) + \nu}{(1 - 2\nu)/(e^{\varepsilon} + 1) + \nu},$$
which gives 0.569 at these values. The paper does not explain the difference, and 0.686 is the more cautious figure. Their bound for all apps at once, whose usage adds up to at most the length of a round, is $\varepsilon + e^{\varepsilon} - 1$, which is 1.672 from 0.686. Across rounds, memoization means that a user whose behaviour does not change reveals nothing new, while each change of behaviour can show. A user whose usage changes every round could lose up to $4 \times 0.686 = 2.744$ per day.
5. **Action to user level.** The Facebook release adds Gaussian noise with standard deviation $\sigma$ to counts of distinct user–URL pairs, separately for each kind of action. One action moves one count by 1, so it is $\rho$-zCDP with $\rho = 1/(2\sigma^2)$. A user with $k$ distinct URLs moves $k$ counts by 1 each, so by composition of zCDP the user is $k/(2\sigma^2)$-zCDP. The release sets $k$ at the 99th percentile of URLs per user for each kind of action. It then picks $\sigma$ so that the user level is $\varepsilon = 0.45$ at $\delta = 10^{-5}$, using a tighter conversion from the same paper of Bun & Steinke (2016),
$$\varepsilon = \rho + \sqrt{4\rho\ln(\sqrt{\pi\rho}/\delta)}.$$
For clicks the release uses $k = 17$ and $\sigma = 40$. So one click has $\rho = 0.00031$, which is $\varepsilon = 0.10$, and a user's 17 clicks have $\rho = 0.0053$, which is $\varepsilon = 0.454$. For views, with $k = 51{,}914$ and $\sigma = 2{,}228$, one view costs almost nothing, while a user's views still add up to $\varepsilon = 0.450$. Users above the 99th percentile get proportionally more: twice the actions, twice the $\rho$.
6. **Per day versus total.** A budget per user per day of $\varepsilon = 1$ adds up to 365 over a year for someone whose behaviour is the same every day. NIST SP 800-226 uses exactly this example to warn that event-level and user-day units can be surprisingly weak.

### Interpretation

**The model rows are the comparable ones.** For training a model on user data, the Gboard models are the best documented: device-level guarantees for a whole training run, $\varepsilon$ between about 1 and 14 at $\delta = 10^{-10}$. The running example's budgets of 1 to 8 at $\delta = 10^{-5}$ are in that range, but at a much larger $\delta$ and with records, not users, as the unit.

**Large published values are not rare.** The Census Bureau published $\varepsilon = 17.14$ for persons. Many local deployments spend several units of $\varepsilon$ per day. NIST SP 800-226 notes that many deployments have used $1 < \varepsilon \le 20$. These are facts about practice, not evidence that a value is safe; Dwork, Kohli & Mulligan (2019) found no clear consensus on how deployments choose $\varepsilon$, and proposed an Epsilon Registry, a public record of deployed parameters, so that choices could be compared and criticized.

**Compare like with like.** Desfontaines' list of real-world uses of DP converts guarantees to $\delta = 10^{-5}$ and, for Apple, to a user-day unit, so its numbers differ from the primary sources on purpose. When you compare your budget with a deployment, convert both to the same unit, neighbouring relation, accounting and $\delta$, or say that you did not.

The deployments also teach what a published $\varepsilon$ leaves out. Five lessons recur, and each is a question to ask of any budget, including your own.

**A budget that renews has no lifetime bound.** The local deployments state $\varepsilon$ per report, per donation or per day, and refresh it. Rogers et al. (2021, Table 2) put several of them on a monthly scale for a user whose records change, so that memoization never applies. One RAPPOR report every 30 minutes at $\varepsilon_1 = 0.534$ then adds up to $0.534 \times 48 \times 30 = 769$ per month. Microsoft's four rounds a day at 0.686 add up to 82.3 per month. Apple's Safari domains, with two donations a day at $\varepsilon = 4$, add up to 240. These are upper bounds by basic composition, and memoization makes them much smaller for users whose data do not change, which is why the local deployments memoize. But no single number covers a user's lifetime.

**Some protections sit outside $\varepsilon$.** According to Apple's Differential Privacy Team (2017), Apple's devices send reports over an encrypted channel, Apple strips IP addresses before the reports are stored, and only a random sample of reports is sent. RAPPOR's instantaneous randomized response also makes a user's reports hard to link to each other, a benefit that $\varepsilon_1$ does not express. These measures matter, but the stated $\varepsilon$ does not cover them. A privacy statement should list them separately.

**Some guarantees rest on assumptions that are not mathematical.** LinkedIn's Audience Engagements API answers queries chosen by outside analysts, such as advertisers, and keeps a budget per analyst per month (Rogers et al. 2021). Two analysts who pooled their answers would together exceed it. The guarantee therefore assumes that analysts do not collude, which LinkedIn enforces by vetting and by contract, not by noise. It does blunt one form of collusion, because every analyst who asks the same query on the same day receives the same noisy answer, so pooling identical answers reveals nothing new.

**A value can be argued from a specific attack.** LinkedIn chose its per-result $\varepsilon = 0.15$ by modelling an analyst who asks the same count on each of the 30 days the data are kept and averages the noisy answers. With the API's Laplace noise of scale $2/\varepsilon$ on each answer, the source's normal approximation gives a probability of 0.115 that the average lands within $\frac12$ of the true count, which would reveal it after rounding. That is the "about 11%" chance the authors accepted. This is the procedure of @sec-epsc-justify in miniature: name the attack, compute its success, and choose the number from it.

**Partial user-level guarantees should say whom they cover.** The Facebook URLs release protects each action strongly, and protects whole users at $\varepsilon = 0.45$ for each kind of action only up to the 99th percentile of activity. Users above the 99th percentile get a weaker guarantee, whose $\varepsilon$ grows with how far they exceed it ([Exercise 7](../../book-exercises/part05-dp-ml/09-choosing-epsilon.md#sec-sol-epsc-7)). That is an honest way to state a guarantee that cannot cover everyone, and [User-level differential privacy](07-user-level.md) explains why bounding each user's contribution is the price of covering all of them.

### Example: the Census as a training run

Could the running example's $\varepsilon = 8$ run be described as "less than half the Census budget"? The Census $\varepsilon = 17.14$ is replace-one at $\delta = 10^{-10}$; the run's 7.997 is add/remove at $\delta = 10^{-5}$. At $\delta = 10^{-10}$ the run is at 11.627 (@tbl-epsc-delta), and replace-one doubles an add/remove guarantee in the worst case. After converting, the honest statement is that the two are of the same order, which is a weaker claim than the slogan.

### Check yourself

- A product reports "$\varepsilon = 2$ per user per day" for data kept a year. What total should you assume for a user whose behaviour never changes? *(Up to $2 \times 365 = 730$ by basic composition, unless the mechanism memoizes or the accounting shows otherwise.)*
- Why can the Census Bureau's 19.61 be larger than the $\varepsilon$ of the combined $\rho = 2.63$? *(Because adding two $\varepsilon$ values computed at the same $\delta$ is a valid but loose composition, while zCDP composes by adding $\rho$ and converts once.)*

## Justifying a choice {#sec-epsc-justify}

### Context

You must pick a target for the running example and write a statement that a sceptical reviewer will check. What should the procedure be, and what does it give here?

### Statement

A defensible choice of $\varepsilon$ fixes the claim before the number:

1. **The unit and the neighbouring relation.** Record, user, device; add/remove or replace-one; the period over which the budget is spent, including every training run and hyperparameter search that touched the data ([Hyperparameter tuning](03-hyperparameter-tuning.md)).
2. **The threat.** Membership, attribute inference or reconstruction; the attacker's knowledge and prior; what it observes (the final model, the iterates, an API).
3. **The whole guarantee.** The privacy profile $\varepsilon(\delta)$ or the trade-off curve, computed by a tight accountant, with $\delta \ll 1/n$, and read at the operating points of the threat: TPR at FPR $10^{-3}$ and below for membership, at FPR $\kappa$ for reconstruction.
4. **The evidence.** An empirical audit ([Auditing private training](10-auditing.md)) whose lower bound is consistent with the claim, and a check of the implementation against the assumptions of the proof ([DP-SGD](02-dp-sgd.md#sec-dpsgd-algorithm)).
5. **The utility.** The accuracy curve against $\varepsilon$, and where it bends: the smallest budget past the bend is the natural candidate.
6. **The comparison.** The tiers of Ponomareva et al. (2023): $\varepsilon \le 1$ gives strong formal guarantees; $\varepsilon \le 10$ is the widely used goal they call a reasonable level of anonymization for many applications; above 10, the guarantee alone cannot be taken as sufficient evidence of anonymization and additional measures such as auditing may be necessary. NIST SP 800-226 offers no specific rule for choosing $\varepsilon$, cites a study suggesting that $\varepsilon$ of 0.1 generally provides strong protection and values below 1 are reasonable, and warns that large values may not provide meaningful privacy, that releases with $\varepsilon$ of 10 or more can leave outliers vulnerable, and that organizations should evaluate the risk of the final data, especially for $\varepsilon$ above 1. It also recommends publishing the parameters, which it says creates no additional risk.

### Derivation: why each step is needed

1. **Unit.** Without it, the same number can mean protection of a single click or of a person's whole history (@sec-epsc-deployed).
2. **Threat.** The curve answers different questions at different points: a prior of 0.5 reads the curve near FPR 0.5, a thousand candidates read it at $10^{-3}$ (@sec-epsc-bayes, @sec-epsc-recon).
3. **Whole guarantee.** The single pair can overstate the attacker by a factor of 17 at FPR $10^{-3}$ and $\varepsilon = 8$ (@tbl-epsc-curve), so a statement based on the pair alone either undersells the model or tempts the reader to dismiss DP as vacuous.
4. **Evidence.** The curve is a theorem about an algorithm; whether the code runs that algorithm is an empirical fact, and audits are how bugs that break the proof are found.
5. **Utility.** The accuracy curve is steeply non-linear (@fig-epsc-accuracy), so the cost of halving $\varepsilon$ varies by an order of magnitude along it.
6. **Comparison.** Tiers and practice give a reader a reference point; they cannot replace steps 1 to 5.

### Interpretation

**The accuracy curve and the risk curve together.** @fig-epsc-accuracy puts the accuracies of [DP-SGD](02-dp-sgd.md) against the largest membership TPR at FPR 1%, from the pair and from the exact curve. Going from $\varepsilon = 1$ to 2 gains 2.66 points of accuracy (0.9175 to 0.9441) and moves the exact TPR at FPR 1% from at most 0.01893 to at most 0.03141. From 2 to 4 gains 1.07 points for a TPR of at most 0.0724. From 4 to 8 gains 0.56 points (0.9548 to 0.9604), the gap to the non-private model falling from 0.0062 to 0.0006, and nearly triples the TPR, to at most 0.2092. Past $\varepsilon = 4$, each unit of accuracy is bought with a much larger increase in worst-case risk.

**Rules of thumb are where the reasoning stops, not where it starts.** "$\varepsilon \le 1$ is strong" is true of every mechanism, because @eq-epsc-pair is already tight enough. "1 to 10 is common in machine learning" is true of practice, and this chapter shows why it can be reasonable for DP-SGD: at those budgets the actual curve is far below the line. "Above 10 needs justification" is true because there even the actual curve allows attacks with substantial power on worst-case records (TPR above 0.27 at FPR $10^{-3}$ at $\varepsilon = 16$).

![Left: test accuracy of the running example against $\varepsilon$ at $\delta = 10^{-5}$, from the DP-SGD chapter's table (three runs each), with the non-private model dashed. Right: the same accuracies against the largest TPR of a membership test at FPR 1%, from the exact curve (circles, labelled by $\varepsilon$) and from the pair alone (squares). The dotted vertical line is an attacker with no information.](figures/epsc_accuracy_risk.svg){#fig-epsc-accuracy}

### Example: a privacy statement for the running example

Following the procedure, a team might choose the $\varepsilon = 4$ run and write:

> The model was trained with DP-SGD (Poisson sampling rate 0.01, noise multiplier 1.36, clipping norm 0.5, 10,000 steps) on 10,000 records, one record per person. The privacy unit is one record under add/remove neighbours; no other run touched the training data. The RDP accountant gives $(3.993, 10^{-5})$-DP and $(5.209, 10^{-8})$-DP; a PLD accountant gives $\varepsilon \le 3.729$ at $\delta = 10^{-5}$. Against an attacker who knows every other record and sees every iterate, any membership test has TPR at most 0.01312 at FPR $10^{-3}$ and at most 0.0724 at FPR $10^{-2}$, and any attack that picks a record out of 100 equally likely candidates succeeds with probability at most 0.0724. Test accuracy is 0.9548, against 0.9610 without privacy.

Every number in it is checkable, and each comes with its unit, its accountant and the attacker it refers to. A reviewer who thinks $\varepsilon = 4$ is too large can now argue with the TPR at a specific FPR, which is the substance, rather than with the number.

### Check yourself

- Your accuracy curve is flat above $\varepsilon = 2$. Which budget should you choose, and why? *(About 2, or slightly above to leave room for run-to-run variation: past the bend, more budget buys almost no accuracy and raises the worst-case risk.)*
- A reviewer asks why you report the TPR at FPR $10^{-3}$ rather than the advantage. *(Because an attack is dangerous when its flags are reliable, which is a low-FPR property, and because the advantage is read at one point near the middle of the curve, where it says little about confident identification.)*

## Code: check every number by computation

The script [code/choosing_epsilon.py](../../code/part05-dp-ml/choosing_epsilon.py) calibrates the running example's noise multipliers with the RDP accountant of [DP-SGD](02-dp-sgd.md), discretizes the one-step privacy loss of the subsampled Gaussian pessimistically and optimistically, composes it 10,000 times by FFT, and reads off the trade-off curve, the privacy loss quantiles and the bounds of this chapter. It also trains one model per budget and runs the loss-threshold attack. From the repository root:

```bash
uv run python code/part05-dp-ml/choosing_epsilon.py
```

It takes two to three minutes, most of it in the PLD compositions. **Output** (seed 0), summarized in @tbl-epsc-code:

| Experiment | Result |
| --- | --- |
| Accountants | RDP $\varepsilon$ = target; PLD brackets 0.02 to 0.1 wide and below it; GDP $\varepsilon$ below the PLD lower end except at $\varepsilon = 0.5$ and 1 (@tbl-epsc-accountants) |
| Membership at FPR $10^{-3}$ | Pair 0.002728 / 0.05461 / 0.9997 at $\varepsilon = 1$ / 4 / 8; exact curve at most 0.00225 / 0.01312 / 0.0589 |
| Advantage | Pair 0.9640 at $\varepsilon = 4$; exact at most 0.3429 |
| Bayes | Median total loss 0.401 at $\varepsilon = 4$, 1.136 at $\varepsilon = 8$; $Q[L > \varepsilon]$ between $3.78\times10^{-6}$ and $1.10\times10^{-4}$ |
| Reconstruction, 1,000 candidates | At most 0.0589 at $\varepsilon = 8$, 0.2808 at $\varepsilon = 16$ |
| $\delta$ | $\varepsilon = 4$ run: 3.993 at $10^{-5}$, 5.209 at $10^{-8}$, 5.890 at $10^{-10}$ |
| Conversions | $\rho = 2.56$ gives 17.915 at $\delta = 10^{-10}$ by @eq-epsc-zcdp; RAPPOR $\varepsilon_\infty = 2.043$ |
| Deployment checks | Facebook clicks: $\rho = 0.0053$, $\varepsilon = 0.454$ per user; Microsoft one round by Theorem 6: 0.569 (reported 0.686); LinkedIn averaging attack at $\varepsilon = 0.15$: 0.115; monthly totals 769 (RAPPOR) and 82.3 (Microsoft) |
| Loss-threshold attack | TPR within 0.004 of the FPR at every budget and every FPR (@tbl-epsc-attack) |

: The script's output with the default seed. {#tbl-epsc-code}

[tests/test_choosing_epsilon.py](../../tests/test_choosing_epsilon.py) checks that the output still matches.

**What to take from it.** The exact trade-off curve of DP-SGD is cheap to compute, with the same machinery as a numerical accountant, and it changes the conversation: at the budgets machine learning uses, the curve is far below the line that the reported $(\varepsilon, \delta)$ draws, and it, not the line, is what an attacker faces.

## References

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (Foundations and Trends in Theoretical Computer Science, 2014). The definition and its testing and Bayesian readings.
- Kairouz, Oh & Viswanath, "The Composition Theorem for Differential Privacy" (ICML 2015), and Dong, Roth & Su, "Gaussian Differential Privacy" (JRSS B, 2022). The testing view of DP, trade-off curves, the duality of @eq-epsc-duality and the central limit theorem for composition.
- Koskela, Jälkö & Honkela, "Computing Tight Differential Privacy Guarantees Using FFT" (AISTATS 2020), and Gopi, Lee & Wutschitz, "Numerical Composition of Differential Privacy" (NeurIPS 2021). Numerical PLD accounting, which @eq-epsc-roc reuses for the curve.
- Nasr, Song, Thakurta, Papernot & Carlini, "Adversary Instantiation: Lower Bounds for Differentially Private Machine Learning" (IEEE S&P 2021). Attacks that reach the accountant's guarantee when the adversary controls the target's gradient and sees every iterate.
- Balle, Cherubin & Hayes, "Reconstructing Training Data with Informed Adversaries" (IEEE S&P 2022). Reconstruction robustness and its bounds from pure DP, RDP and zCDP.
- Hayes, Mahloujifar & Balle, "Bounding Training Data Reconstruction in DP-SGD" (NeurIPS 2023). The reconstruction bound from the trade-off curve of DP-SGD, @eq-epsc-rero, and experiments showing it is nearly tight.
- Guo, Karrer, Chaudhuri & van der Maaten, "Bounding Training Data Reconstruction in Private (Deep) Learning" (ICML 2022). Lower bounds on the mean squared error of reconstruction from RDP and Fisher information.
- Near, Darais, Lefkovitz & Howarth, *Guidelines for Evaluating Differential Privacy Guarantees* (NIST SP 800-226, March 2025). Privacy units, $\delta \le 1/n^2$, the hazards of large $\varepsilon$ and of event-level units, and the case for publishing parameters.
- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023). The three tiers of $\varepsilon$ for machine learning and what to report.
- Dwork, Kohli & Mulligan, "Differential Privacy in Practice: Expose your Epsilons!" (Journal of Privacy and Confidentiality, 2019). Interviews with practitioners and the proposal of an Epsilon Registry.
- Desfontaines, "A list of real-world uses of differential privacy" (blog, desfontain.es, updated 2025). A maintained list of deployments, converted to a common $\delta$ and unit.
- Abowd et al., "The 2020 Census Disclosure Avoidance System TopDown Algorithm" (Harvard Data Science Review, 2022), and US Census Bureau, "Census Bureau Sets Key Parameters to Protect Privacy in 2020 Census Results" (press release, June 2021). The redistricting data's $\rho$ and $\varepsilon$. Cumings-Menon et al., "Disclosure Avoidance for the 2020 Census Demographic and Housing Characteristics File" (HDSR, 2023). Kazan & Reiter, "Assessing Statistical Disclosure Risk for Differentially Private, Hierarchical Count Data" (arXiv 2022), for the 17.91 conversion.
- Xu, Zhang, Andrew, Choquette-Choo, Kairouz, McMahan, Rosenstock & Zhang, "Federated Learning of Gboard Language Models with Differential Privacy" (ACL 2023 Industry Track); McMahan & Thakurta, "Federated Learning with Formal Differential Privacy Guarantees" (Google AI blog, 2022); Xu & Zhang, "Advances in private training for production on-device language models" (Google Research blog, 2024). The Gboard rows.
- Apple Differential Privacy Team, "Learning with Privacy at Scale" (Apple Machine Learning Journal, 2017), and Apple, "Differential Privacy Overview" (2017). Tang, Korolova, Bai, Wang & Wang, "Privacy Loss in Apple's Implementation of Differential Privacy on MacOS 10.12" (arXiv 2017). The Apple rows.
- Erlingsson, Pihur & Korolova, "RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response" (CCS 2014). Ding, Kulkarni & Yekhanin, "Collecting Telemetry Data Privately" (NeurIPS 2017), §6 for the deployment and Theorems 6 and 7 for the per-round values. Rogers et al., "LinkedIn's Audience Engagements API: A Privacy Preserving Data Analytics System at Scale" (Journal of Privacy and Confidentiality, 2021), §8 for the non-collusion assumption, the attack behind $\varepsilon = 0.15$, and Table 2, which puts deployments on a monthly scale. Messing et al., "Facebook Privacy-Protected Full URLs Data Set" (Harvard Dataverse, 2020), the privacy section and its table of noise per kind of action. The other rows of @tbl-epsc-deployed.
- Bun & Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds" (TCC 2016). The conversion @eq-epsc-zcdp.

## Cheat sheet

$Q$, $P$: output laws with and without the target; $L = \ln(Q/P)$ its privacy loss; $\omega$ the law of $L$ under $Q$; $\delta(\varepsilon)$ the privacy profile; $p$ the attacker's prior; $\kappa$ the reconstruction baseline. @tbl-epsc-cheat collects the results.

| Result | Formula | Conditions |
| --- | --- | --- |
| One pair | $\mathrm{TPR} \le e^\varepsilon\mathrm{FPR} + \delta$; advantage $\le \frac{e^\varepsilon - 1 + 2\delta}{e^\varepsilon + 1}$ | Any $(\varepsilon, \delta)$-DP mechanism |
| Whole curve | $\mathrm{TPR}^*(a) = \inf_\varepsilon(e^\varepsilon a + \delta(\varepsilon))$ | Exact, from the full profile |
| Curve from the PLD | $(\mathbb E_\omega[e^{-L}; L > \tau], \omega(L > \tau))$ | Composition: convolve per-step PLDs |
| From RDP | $\mathrm{TPR} \le \min_\alpha(\mathrm{FPR}\,e^{\varepsilon(\alpha)})^{(\alpha - 1)/\alpha}$ | Valid bound, cheap |
| GDP | $\mathrm{TPR} \approx 1 - \Phi(\Phi^{-1}(1 - a) - \mu)$, $\mu = q\sqrt{T(e^{1/\sigma^2} - 1)}$ | Approximation, not a bound |
| Posterior | $\le pe^\varepsilon/(pe^\varepsilon + 1 - p)$; exactly $pe^L/(pe^L + 1 - p)$ | Fails with probability $\le \delta/(1 - e^{\varepsilon - \varepsilon'})$ at level $\varepsilon'$ |
| Precision of a flag | $p\,\mathrm{TPR}/(p\,\mathrm{TPR} + (1 - p)\mathrm{FPR})$ | Base rates dominate at small $p$ |
| Reconstruction | Success $\le \mathrm{TPR}^*(\kappa)$; pair: $e^\varepsilon\kappa + \delta$ | Informed adversary, prior baseline $\kappa$ |
| $\delta$ | Release mechanism exposes $1 - (1 - \delta)^n$; use $\delta \ll 1/n$, NIST: $\le 1/n^2$ | Per privacy unit |
| zCDP to $(\varepsilon, \delta)$ | $\varepsilon = \rho + 2\sqrt{\rho\ln(1/\delta)}$ | Simple, slightly loose |
| Action to user level | $k$ actions, each $\rho$-zCDP, give $k\rho$ | Covers only users with at most $k$ actions |
| Running example, $\varepsilon = 4$ | TPR $\le 0.01312$ at FPR $10^{-3}$ (pair: 0.05461); accuracy 0.9548 | $q = 0.01$, $\sigma = 1.36$, $T = 10{,}000$ |
| Running example, $\varepsilon = 8$ | TPR $\le 0.0589$ at FPR $10^{-3}$ (pair: 0.9997); accuracy 0.9604 | $\sigma = 0.917$ |

: Every result of the chapter. {#tbl-epsc-cheat}

**Rules of thumb**

- Report the curve, not just the pair: the TPR at FPR $10^{-3}$ and $10^{-2}$, computed with a PLD accountant or bounded with RDP.
- State the unit, the neighbouring relation, the period and $\delta$ with every $\varepsilon$; convert before comparing with anyone else's.
- $\varepsilon \le 1$ is strong for any mechanism. Between 1 and 10, DP-SGD's actual curve is far below the pair's line, and that curve is your justification. Above 10, show the curve and an audit, and expect worst-case records to be exposed.
- Choose $\delta$ well below $1/n$ for the right $n$; for Gaussian-type mechanisms a smaller $\delta$ costs little $\varepsilon$.
- Put the accuracy curve next to the risk curve, and pick the smallest budget past the bend in accuracy.
- Ask of every published budget how it grows over time, which protections sit outside it, which assumptions it rests on, and which users it covers.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part05-dp-ml/09-choosing-epsilon.md).
