# From Rényi DP to (ε, δ) and to tests

## Why this matters

Rényi differential privacy (RDP) is the quantity accountants track, because it composes by adding curves ([previous chapter](01-renyi-dp.md)). Recall what the curve is. For two neighbouring datasets, which differ in one person's record, a mechanism produces two output distributions, $P$ on the dataset without your record and $Q$ on the one with it. For each **order** $\alpha > 1$, the Rényi divergence $D_\alpha(P\|Q)$ measures how far apart they are: it is a logarithmic average of the likelihood ratio $P/Q$ raised to the power $\alpha - 1$, so larger orders weigh the outputs where $P$ is much more likely than $Q$ more heavily. The **RDP curve** $\varepsilon(\alpha)$ bounds that divergence, over all neighbouring pairs, at every order. When mechanisms run one after another, their curves add.

The curve is not the quantity anyone reports. A paper, a model card or a regulator asks for $(\varepsilon, \delta)$-differential privacy (DP), or for a statement about attacks: how much better than guessing can a test tell whether your record was used? So every RDP analysis ends with a conversion, and the conversion decides the number you publish. This chapter answers three questions: how much the standard conversion loses, and where; what the best $(\varepsilon, \delta)$ is that you can claim from an RDP curve, and which order to use for it; and what an RDP curve says about tests, and what it can never say.

One idea organizes all three answers: **an RDP curve is a summary of the privacy loss distribution, and converting it asks for the largest $\delta$ consistent with that summary.** The privacy loss of an output is the logarithm of how much more likely that output is under $P$ than under $Q$. Drawn at random from the mechanism, it is the privacy loss random variable (PRV), and $\delta$ is a particular average over it. The RDP curve records the moment generating function (MGF) of the PRV, the average of $e^{\lambda L}$ for a loss $L$, at the points $\lambda = \alpha - 1$, and nothing else. A conversion has to bound the average that defines $\delta$ using only those MGF values. Mironov's conversion does this with Markov's inequality and throws away most of what the summary says. The improved conversion, and the hypothesis-testing view behind it, recover almost all of it: for the Gaussian mechanism the improved conversion agrees to three decimals with the best any method could do with the curve alone. What remains, 5 to 10% of $\varepsilon$, is not a defect of any formula. A two-outcome mechanism has an RDP curve no larger than the Gaussian's and a larger $\delta$, so no conversion that sees only the curve can do better. @tbl-conv-uses lists the results.

| Result | Where it is used |
| --- | --- |
| Mironov's conversion $\varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1)$ | The original moments accountant; still in older code and papers |
| Improved conversion (Balle et al.; Canonne, Kamath & Steinke) | Every modern RDP accountant: Opacus, Google's `dp_accounting` |
| RDP as a constraint on tests | Reading an RDP guarantee as a bound on attack power; comparing with $f$-DP |
| Choosing the order $\alpha$ | Designing the order grid of an accountant |
| The slack no conversion removes | Knowing when to switch to a numerical accountant |
| $(\varepsilon, \delta)$ does not imply RDP; $\varepsilon$-DP does | Mixing pure-DP and RDP components in one analysis |

: The results in this chapter and where they are used. {#tbl-conv-uses}

**Conventions.** $P = M(D)$ and $Q = M(D')$ are the output distributions of the mechanism $M$ on neighbouring datasets $D$ and $D'$. The same letters name their densities at an output $y$, $P(y)$ and $Q(y)$, because $q$ is the sampling rate. As in Part 3, $D'$ is the dataset that holds your record and $D$ the one without it. A guarantee must hold for every ordered pair of neighbours, so every bound below holds just as well with the roles of $P$ and $Q$ swapped. The privacy loss at an output $y$ is $L(y) = \ln(P(y)/Q(y))$: positive when $y$ is more likely under $D$ than under $D'$, negative when it is less likely. With $Y \sim P$, the random number $L(Y)$ is the PRV ([composition](../part03-dp-foundations/04-composition.md)). Throughout, **$\alpha$ is the Rényi order and never an error rate.** A test of "the output came from $P$" against "it came from $Q$", that is, of "your record was not used" against "it was", looks at the output and either accepts or rejects. It has **type I error $a$**, the probability that it rejects although the output came from $P$ (for a membership test, its false-positive rate: saying $D'$ when the data were $D$), and **type II error $b$**, the probability that it accepts although the output came from $Q$, as in [hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md). Its **power** $1 - b$ is the probability that it correctly rejects when the output came from $Q$.

The running example is the Gaussian mechanism. One step adds noise $\mathcal N(0, \sigma^2)$ to a query of sensitivity 1, where $\sigma$ is the noise multiplier. After dividing the output by $\sigma$, the two worlds are $\mathcal N(0, 1)$ and $\mathcal N(1/\sigma, 1)$. With $k$ independent steps the outputs form a vector, and only its component along the direction of the shift differs between the worlds; that shift has length $\sqrt{k}/\sigma$. So every quantity in this chapter depends only on $\mu = \sqrt k/\sigma$, and the composed mechanism is the pair $\mathcal N(0, 1)$, $\mathcal N(\mu, 1)$. Its RDP curve is $\varepsilon(\alpha) = \alpha\mu^2/2$: one step has curve $\alpha/(2\sigma^2)$ ([previous chapter](01-renyi-dp.md)), and $k$ steps add $k$ copies, $k\alpha/(2\sigma^2) = \alpha\mu^2/2$. All numbers use $\delta = 10^{-5}$ and the part's grid of 154 orders: 1.1 to 10.9 in steps of 0.1, every integer from 12 to 63, and then 128, 256 and 512.

## Toolkit: four facts every derivation uses

### Fact 1 — δ is an expectation over the privacy loss

$(\varepsilon, \delta)$-DP for the ordered pair $(P, Q)$ says $P(S) \le e^\varepsilon Q(S) + \delta$ for every event $S$ ([definition](../part03-dp-foundations/01-definition.md)): any set of outputs is at most $e^\varepsilon$ times more likely under $D$ than under $D'$, up to an additive slack $\delta$. The smallest $\delta$ that works is therefore the largest value of $P(S) - e^\varepsilon Q(S)$ over all events. That largest value is reached by putting into $S$ exactly the outputs with $P > e^\varepsilon Q$, because each such output adds a positive amount $P - e^\varepsilon Q$ and every other output would subtract. The result, $\int (P - e^\varepsilon Q)_+$, where $(x)_+ = \max(x, 0)$, is the hockey-stick divergence $E_{e^\varepsilon}(P\|Q)$ of [divergences](../part01-prerequisites/03-divergences.md). Written in terms of the privacy loss, it is an average over the PRV:
$$\delta(\varepsilon) = E_{e^\varepsilon}(P\|Q) = \mathbb E_P\big[(1 - e^{\varepsilon - L})_+\big] = \Pr_P[L > \varepsilon] - e^\varepsilon \Pr_Q[L > \varepsilon].$$ {#eq-conv-delta}

The second form comes from factoring $P$ out of the integrand. Where $P > 0$, $(P - e^\varepsilon Q)_+ = P\,(1 - e^\varepsilon Q/P)_+$, and $Q/P = e^{-L}$, so this is $P\,(1 - e^{\varepsilon - L})_+$; integrating $P$ times a function of the output is taking its average under $P$. The integrand is positive exactly where $L > \varepsilon$, and integrating $P - e^\varepsilon Q$ over that set gives the last form.

The second form is the one to remember. Each output contributes to $\delta$ only if its privacy loss exceeds $\varepsilon$, and then only the fraction $1 - e^{\varepsilon - L}$ of its probability. An output with loss $L = \varepsilon + \ln 2$ contributes half its probability, since $1 - e^{-\ln 2} = 1 - \frac12$. An output with loss just above $\varepsilon$ contributes almost nothing, and only an output with infinite loss contributes all of its probability.

### Fact 2 — RDP is the MGF of the privacy loss

The Rényi divergence of order $\alpha$ is defined by $D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\ln\int P^\alpha Q^{1 - \alpha}$. Write $P^\alpha Q^{1 - \alpha} = P\,(P/Q)^{\alpha - 1}$ and $(P/Q)^{\alpha - 1} = e^{(\alpha - 1)L}$: the integral is the average under $P$ of $e^{(\alpha - 1)L}$, which is the MGF of the PRV at the point $\alpha - 1$. An RDP bound $D_\alpha(P\|Q) \le \varepsilon(\alpha)$ is therefore a bound on that MGF value:
$$\mathbb E_P\big[e^{(\alpha - 1)L}\big] = e^{(\alpha - 1)D_\alpha(P\|Q)} \le e^{(\alpha - 1)\varepsilon(\alpha)}.$$ {#eq-conv-mgf}
This is @eq-renyi-psi of [divergences](../part01-prerequisites/03-divergences.md), with the exponent multiplied out.

So knowing the RDP curve at the orders $\alpha_1, \alpha_2, \dots$ is knowing (an upper bound on) the MGF of the PRV at $\alpha_1 - 1, \alpha_2 - 1, \dots$. That is the whole summary. Every conversion is a way of bounding the average in @eq-conv-delta using only the averages in @eq-conv-mgf.

### Fact 3 — Data processing on a test

A test $\phi$ maps the output to "reject" or "accept", so its outcome is a coin. Under $P$ the coin shows "reject" with probability $a$; under $Q$ it shows "reject" with probability $1 - b$. Turning the output into this coin is post-processing, and post-processing cannot increase a Rényi divergence: this is the data processing inequality of [divergences](../part01-prerequisites/03-divergences.md). The short reason is that merging outputs can only shrink the sum $\int P^\alpha Q^{1 - \alpha}$. The function $x^\alpha y^{1 - \alpha}$ is convex for $\alpha > 1$ and scales linearly when $x$ and $y$ are both multiplied by the same factor, and such a function satisfies $g(x_1 + x_2, y_1 + y_2) \le g(x_1, y_1) + g(x_2, y_2)$; merging two outputs replaces the right side by the left (@sec-apx-tail-perspective in the appendices proves this). (A test that randomizes its decision is handled the same way; [divergences](../part01-prerequisites/03-divergences.md) gives the general proof.) So, writing $\mathrm{Bern}(x)$ for "reject with probability $x$",
$$D_\alpha\big(\mathrm{Bern}(a)\,\|\,\mathrm{Bern}(1 - b)\big) \le D_\alpha(P\|Q).$$
In words: a test cannot tell the two worlds apart better, in Rényi divergence, than the full output does.

### Fact 4 — The Gaussian pair, exactly

For the running example everything can be computed in closed form, which is what lets this chapter measure each conversion against the truth. Take $P = \mathcal N(0, 1)$ and $Q = \mathcal N(\mu, 1)$. The privacy loss is the difference of the two log densities, $L(y) = -\frac{y^2}{2} + \frac{(y - \mu)^2}{2} = \frac{\mu^2}{2} - \mu y$. It is a linear function of $y$, so under $P$, where $y$ is standard normal, the PRV is normal with mean $\mu^2/2$ and standard deviation $\mu$: $\mathcal N(\mu^2/2, \mu^2)$.

Now evaluate the last form of @eq-conv-delta. Write $\Phi(x) = \Pr[Z \le x]$ for the standard normal cumulative distribution function. Under $P$, $L > \varepsilon$ means $y < \mu/2 - \varepsilon/\mu$, which has probability $\Phi(\mu/2 - \varepsilon/\mu)$. Under $Q$, write $y = \mu + z$ with $z$ standard normal; then $L = -\mu^2/2 - \mu z$, and $L > \varepsilon$ means $z < -\mu/2 - \varepsilon/\mu$. So
$$\delta(\varepsilon) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^\varepsilon\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big),$$ {#eq-conv-gauss-delta}
the closed form already derived in [divergences](../part01-prerequisites/03-divergences.md). It falls as $\varepsilon$ grows, because the integrand $(1 - e^{\varepsilon - L})_+$ does, so inverting it by bisection gives the exact $\varepsilon$ at any $\delta$. The script evaluates the second term in logarithms so that it stays accurate when $e^\varepsilon$ overflows. As a check, at $\mu = 1$ a Riemann sum of $\mathbb E_P[(1 - e^{\varepsilon - L})_+]$ over the normal density gives $1.2694\times10^{-1}$ at $\varepsilon = 1$ and $1.5372\times10^{-3}$ at $\varepsilon = 3$, the same as the closed form to all printed digits.

## Where Mironov's conversion loses {#sec-conv-mironov}

### Context

[Rényi DP](01-renyi-dp.md#sec-rdp-conversion) derived Mironov's conversion: an $(\alpha, \varepsilon(\alpha))$-RDP mechanism, one whose Rényi divergence of order $\alpha$ is at most $\varepsilon(\alpha)$, is $(\varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1), \delta)$-DP. It is easy to remember and easy to compute, and @sec-apx-conv-rdp in the appendices derives it from scratch, with the Gaussian's best order in closed form. This section rewrites that proof as two inequalities on the privacy loss and measures what each one costs on the Gaussian, so that you can see where the excess comes from.

### Statement

The proof first replaces $\delta$ by the probability that the loss exceeds $\varepsilon$, and then bounds that probability with the MGF. For every order $\alpha > 1$ and every $\varepsilon$,
$$\delta(\varepsilon) \;\overset{(1)}{\le}\; \Pr_P[L > \varepsilon] \;\overset{(2)}{\le}\; e^{(\alpha - 1)(\varepsilon(\alpha) - \varepsilon)},$$ {#eq-conv-mironov-chain}
and setting the right side equal to $\delta$ gives **Mironov's conversion**
$$\varepsilon' = \varepsilon(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}.$$ {#eq-conv-mironov}

### Derivation

The proof in [Rényi DP](01-renyi-dp.md#sec-rdp-conversion) has exactly these two steps. Written with Fact 1 and Fact 2:

1. **Drop the fraction.** Fact 1 says each output with $L > \varepsilon$ contributes the fraction $1 - e^{\varepsilon - L}$ of its probability. Step 1 rounds that fraction up to 1: $(1 - e^{\varepsilon - L})_+ \le \mathbf 1[L > \varepsilon]$, because the left side is 0 when $L \le \varepsilon$ and below 1 otherwise. Averaging under $P$ gives inequality (1). This is the step that charges the whole tail $\Pr_P[L > \varepsilon]$ to $\delta$.
2. **Bound the tail by the MGF (Chernoff).** Markov's inequality says a nonnegative random variable exceeds a level $t$ with probability at most its average divided by $t$. Apply it to $e^{(\alpha - 1)L}$, which exceeds $e^{(\alpha - 1)\varepsilon}$ exactly when $L > \varepsilon$: $\Pr_P[L > \varepsilon] \le \mathbb E_P[e^{(\alpha - 1)L}]\,e^{-(\alpha - 1)\varepsilon}$. Fact 2 bounds the average by $e^{(\alpha - 1)\varepsilon(\alpha)}$, which gives inequality (2).
3. **Solve for $\varepsilon$.** Setting $e^{(\alpha - 1)(\varepsilon(\alpha) - \varepsilon)} = \delta$ and taking logarithms gives $(\alpha - 1)(\varepsilon(\alpha) - \varepsilon) = -\ln(1/\delta)$, which is @eq-conv-mironov. DP also needs the inequality with the roles of the two datasets swapped, the pair $(Q, P)$. RDP bounds $D_\alpha(Q\|P)$ by the same curve, because swapping two neighbours gives two neighbours, so the same argument applies.

### Interpretation

The two steps lose in different places. @tbl-conv-mironov-steps measures them on the Gaussian, converting each inequality of @eq-conv-mironov-chain into an $\varepsilon$ at $\delta = 10^{-5}$. The first row is the exact $\varepsilon$ from @eq-conv-gauss-delta. The second is the $\varepsilon$ at which the tail $\Pr_P[L > \varepsilon] = \Phi(\mu/2 - \varepsilon/\mu)$ equals $\delta$. Since $\Phi(x) = 10^{-5}$ at $x = -\Phi^{-1}(1 - 10^{-5}) = -4.2649$, where $\Phi^{-1}$ is the inverse of $\Phi$, this is $\varepsilon = \mu^2/2 + 4.2649\,\mu$: $0.5 + 4.2649 \approx 4.765$ at $\mu = 1$ and $50 + 42.649 = 92.649$ at $\mu = 10$. The third row is Mironov's $\varepsilon'$ minimized over the grid.

| | $\mu = 1$ (one step, $\sigma = 1$) | $\mu = 10$ (100 steps, $\sigma = 1$) |
| --- | --- | --- |
| Exact $\varepsilon$ | 4.377 | 91.817 |
| After step 1, $\Pr_P[L > \varepsilon] = \delta$ | 4.765 | 92.649 |
| After step 2, Mironov | 5.299 at $\alpha = 5.8$ | 98.026 at $\alpha = 1.5$ |
| Loss from step 1 | 0.388 | 0.832 |
| Loss from step 2 | 0.534 | 5.377 (0.041 of it from the grid) |

: Mironov's conversion for the Gaussian at $\delta = 10^{-5}$, split into its two steps. {#tbl-conv-mironov-steps}

Step 1 ignores how far past $\varepsilon$ the loss goes. An output with $L$ slightly above $\varepsilon$ counts fully toward the tail but almost nothing toward $\delta$. The density of the Gaussian PRV is falling steeply at $\varepsilon$, which sits several standard deviations above its mean, so most of the tail sits just above $\varepsilon$, and step 1 overcounts it.

Step 2 replaces a tail by an MGF, and that costs more. Markov's inequality is tight only for a variable that puts all its mass on 0 and on the level $t$; the Gaussian PRV is nothing like that, so the bound is far above the true tail at 4.765. At $\mu = 10$ this step costs 5.377 of the $98.026 - 91.817 = 6.209$ total excess (@tbl-conv-mironov-steps). Of that, 0.041 is the grid's: over all real orders Mironov's conversion gives 97.985 at $\alpha = 1.480$, against 98.026 at the grid's 1.5.

On the $\delta$ side the losses are multiplicative. At Mironov's $\varepsilon = 5.299$ for $\mu = 1$, the exact $\delta$ is $1.300\times10^{-7}$. The tail is $7.992\times10^{-7}$, 6.1 times larger, and Mironov's bound is $1.000\times10^{-5}$, 12.5 times larger again. You pay for $\delta = 10^{-5}$ and get $1.3\times10^{-7}$.

### Example: one release of a mean

You release the mean of a bounded statistic with the Gaussian mechanism at noise multiplier $\sigma = 1$, so $\mu = 1$ and $\varepsilon(\alpha) = \alpha/2$. Mironov's conversion gives $\alpha/2 + \ln(10^5)/(\alpha - 1)$ at each order. The first term grows with the order and the second shrinks, so there is a best order in between. Over the continuous range of orders the sum is smallest at $\alpha = 5.799$, with $\varepsilon' = 5.299$ (@sec-conv-order derives both); the grid's best order, 5.8, gives the same 5.299. The exact answer is 4.377. Reporting Mironov's number overstates $\varepsilon$ by 0.921, which means $e^{0.921}$ more "allowed" likelihood ratio for every output.

### Check yourself

- Why does step 2 lose about ten times more at $\mu = 10$ than at $\mu = 1$? *(The PRV has mean $\mu^2/2$ and standard deviation $\mu$. At the best order, Mironov's $\varepsilon^* = \mu^2/2 + 2\sqrt{(\mu^2/2)\ln(1/\delta)} = \mu^2/2 + \mu\sqrt{2\ln(1/\delta)}$ sits $\sqrt{2\ln(1/\delta)} \approx 4.799$ standard deviations above the mean, while the true tail point sits $\Phi^{-1}(1 - \delta) = 4.2649$ standard deviations above it. The gap is a fixed number of standard deviations, about 0.534, so in $\varepsilon$ it grows like $\mu$: 0.534 at $\mu = 1$, about ten times that at $\mu = 10$.)*
- Is step 1 ever tight? *(Only when every output with $L > \varepsilon$ has $L = \infty$, where $1 - e^{\varepsilon - L} = 1$, such as a mechanism that sometimes reveals the record outright, with the pair ordered so that $P$ is the output on the dataset that holds the record.)*

## The improved conversion {#sec-conv-improved}

### Context

Both of Mironov's steps together replace the integrand $(1 - e^{\varepsilon - z})_+$ of @eq-conv-delta, as a function of the loss value $z$, by an exponential $c\,e^{(\alpha - 1)z}$ that lies above it. An exponential is used because it is the only kind of function whose average Fact 2 tells you. Mironov's choice is $c = e^{-(\alpha - 1)\varepsilon}$, the exponential that equals 1 at $z = \varepsilon$ and so passes through the corner of the indicator. That is not the smallest exponential above the integrand, because the integrand is well below 1 near $z = \varepsilon$. Using the smallest one is the whole improvement.

### Statement

An $(\alpha, \varepsilon(\alpha))$-RDP mechanism is $(\varepsilon', \delta)$-DP with
$$\varepsilon' = \varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1},$$ {#eq-conv-improved}
or equivalently, at a given $\varepsilon$,
$$\delta = \frac{e^{(\alpha - 1)(\varepsilon(\alpha) - \varepsilon)}}{\alpha}\Big(1 - \frac1\alpha\Big)^{\alpha - 1}.$$ {#eq-conv-improved-delta}
This is Theorem 21 of Balle, Barthe, Gaboardi, Hsu & Sato (2020) and Proposition 12 of Canonne, Kamath & Steinke (2020), who also give the $\delta$ form; Asoodeh, Liao, Calmon, Kosut & Sankar proved it independently (2020, Lemma 1, Eq. 20). We call it the **improved conversion**. Compared with @eq-conv-mironov, its $\delta$ is Mironov's $e^{(\alpha - 1)(\varepsilon(\alpha) - \varepsilon)}$ multiplied by the factor $\frac1\alpha(1 - \frac1\alpha)^{\alpha - 1}$, which is below 1.

### Derivation

The plan: find the smallest constant $c$ for which $c\,e^{(\alpha - 1)z}$ stays above the integrand, then average.

1. **Bound the integrand by an exponential.** If $(1 - e^{\varepsilon - z})_+ \le c\,e^{(\alpha - 1)z}$ for every real $z$, then the same holds with $z$ replaced by the random loss $L$, and averaging under $P$ keeps the inequality. By Fact 1 the left side averages to $\delta(\varepsilon)$, and by Fact 2 the right side averages to at most $c\,e^{(\alpha - 1)\varepsilon(\alpha)}$. So $\delta(\varepsilon) \le c\,\mathbb E_P[e^{(\alpha - 1)L}] \le c\,e^{(\alpha - 1)\varepsilon(\alpha)}$.
2. **The smallest $c$.** The condition of step 1 says $c \ge (1 - e^{\varepsilon - z})_+\,e^{-(\alpha - 1)z}$ for every $z$, so the smallest valid $c$ is $c^* = \sup_z (1 - e^{\varepsilon - z})_+\,e^{-(\alpha - 1)z}$. For $z \le \varepsilon$ the expression is 0, so only $z > \varepsilon$ matters. There it is $g(z) = e^{-(\alpha - 1)z} - e^{\varepsilon - \alpha z}$.
3. **Find the maximum.** Differentiate: $g'(z) = -(\alpha - 1)e^{-(\alpha - 1)z} + \alpha e^{\varepsilon - \alpha z} = e^{-\alpha z}\big(\alpha e^\varepsilon - (\alpha - 1)e^z\big)$. The bracket equals $e^\varepsilon > 0$ at $z = \varepsilon$ and decreases as $z$ grows, so $g$ first rises and then falls. Its maximum is where the bracket vanishes, at $e^{z^*} = \frac{\alpha}{\alpha - 1}e^\varepsilon$, that is, $z^* = \varepsilon - \ln(1 - 1/\alpha)$.
4. **Evaluate it.** Factor $g(z^*) = e^{-\alpha z^*}(e^{z^*} - e^\varepsilon)$. At $z^*$, $e^{z^*} - e^\varepsilon = e^\varepsilon\big(\frac{\alpha}{\alpha - 1} - 1\big) = e^\varepsilon\cdot\frac{1}{\alpha - 1}$, and $e^{-\alpha z^*} = e^{-\alpha\varepsilon}(1 - 1/\alpha)^\alpha$. Multiplying,
$$c^* = \frac{e^{-(\alpha - 1)\varepsilon}}{\alpha - 1}\Big(1 - \frac1\alpha\Big)^\alpha.$$
5. **Collect.** Step 1 gives $\delta \le c^* e^{(\alpha - 1)\varepsilon(\alpha)}$. Since $(1 - 1/\alpha)/(\alpha - 1) = 1/\alpha$, the constant in front simplifies to $\frac1\alpha(1 - \frac1\alpha)^{\alpha - 1}$, and this is @eq-conv-improved-delta.
6. **Solve for $\varepsilon$.** Set the bound equal to $\delta$ and take logarithms: $\ln\delta = (\alpha - 1)(\varepsilon(\alpha) - \varepsilon) - \ln\alpha + (\alpha - 1)\ln(1 - 1/\alpha)$. Divide by $\alpha - 1$ and solve for $\varepsilon$, using $\ln(1 - 1/\alpha) = \ln\frac{\alpha - 1}{\alpha}$, to get @eq-conv-improved. As before, the reverse pair $(Q, P)$ follows because RDP bounds $D_\alpha(Q\|P)$ by the same curve.

The script checks step 4 by brute force: at $\alpha = 6$ and $\varepsilon = 4$ the formula gives $c^* = 1.380552\times10^{-10}$ at $z^* = 4 - \ln(5/6) = 4.1823$, and the maximum of $g$ over a grid of two million points is the same to all seven digits.

@fig-conv-majorants draws the two exponentials. Mironov's passes through the corner of the indicator at $z = \varepsilon$; the smallest one sits lower and touches the integrand at $z^*$, slightly to the right of $\varepsilon$.

![The hockey-stick integrand $(1 - e^{\varepsilon - z})_+$ (black) at $\varepsilon = 3$, the indicator $\mathbf 1[z > \varepsilon]$ that Mironov's step 1 replaces it with (dashed), and the two exponentials of order $\alpha = 6$ that bound it: Mironov's, through the corner of the indicator (red), and the smallest one, which touches the integrand at $z^* = \varepsilon - \ln(1 - 1/\alpha)$ (blue). The shaded curve is the density of the Gaussian privacy loss at $\mu = 1$: the conversions average the exponentials against it.](figures/rdp_conversions_majorants.svg){#fig-conv-majorants}

### Interpretation

The gain depends only on the order. Both conversions add something to the same $\varepsilon(\alpha)$, so when you subtract @eq-conv-improved from @eq-conv-mironov at the same $\alpha$, the $\varepsilon(\alpha)$ terms cancel, and so do the $\ln(1/\delta)$ terms. What is left is how much smaller the improved $\varepsilon'$ is:
$$\Delta(\alpha) = \frac{\ln\alpha}{\alpha - 1} + \ln\frac{\alpha}{\alpha - 1},$$ {#eq-conv-gain}
whatever the mechanism and whatever $\delta$. On the $\delta$ side, the improved $\delta$ is Mironov's times $\frac1\alpha(1 - \frac1\alpha)^{\alpha - 1}$. Because $(1 - 1/\alpha)^{\alpha - 1} \to 1/e$, that factor behaves like $1/(e\alpha)$ for large $\alpha$. @tbl-conv-gain gives values; at $\alpha = 2$, for instance, the factor is $\frac12\cdot\frac12 = 0.25$ and the saving is $\ln 2 + \ln 2 = 1.3863$.

| $\alpha$ | $\delta$ factor | $1/(e\alpha)$ | $\varepsilon$ saving $\Delta(\alpha)$ |
| --- | --- | --- | --- |
| 2 | 0.2500 | 0.1839 | 1.3863 |
| 10 | 0.0387 | 0.0368 | 0.3612 |
| 100 | 0.0037 | 0.0037 | 0.0566 |

: What the improved conversion gains over Mironov's at the same order. {#tbl-conv-gain}

Over a grid, the saving is at least $\Delta(\alpha_M)$, where $\alpha_M$ is the order Mironov's conversion picks, for any RDP curve. The reason is that the improved conversion takes its own best order, so its minimum is at most its value at $\alpha_M$, and that value is Mironov's minimum minus $\Delta(\alpha_M)$ by @eq-conv-gain. So you can read the saving of any accountant off @tbl-conv-gain from its best order: about 1.39 at order 2, 0.36 at order 10, 0.06 at order 100. For the Gaussian at $\mu = 1$, Mironov picks order 5.8, $\Delta(5.8) = \frac{\ln 5.8}{4.8} + \ln\frac{5.8}{4.8} = 0.555$, and the actual saving is 0.570, a little more because the improved conversion moves to a better order. If two accountants report $\varepsilon$ about 0.4 apart from the same curve at a best order near 10, one of them is using Mironov's conversion. Opacus and `dp_accounting` both use the improved one.

The improved conversion also beats both of Mironov's steps, not only the second. For the Gaussian at $\mu = 1$ it gives 4.729 (at $\alpha = 5.4$), below even the 4.765 of the tail $\Pr_P[L > \varepsilon] = \delta$ after step 1 in @tbl-conv-mironov-steps. That is possible because it never replaces the integrand by an indicator: its exponential is allowed to dip below 1 just above $\varepsilon$, where the integrand is small. At $\mu = 10$ it gives 96.116 against Mironov's 98.026 and the exact 91.817. At $\varepsilon = 4.729$ the exact Gaussian $\delta$ is $2.097\times10^{-6}$, so the guarantee still holds with room to spare.

For one order, it is close to optimal. The next section computes the best possible conversion from the RDP values on a whole grid, and a single order is the special case of a grid with one point. Take only the forward constraint $D_\alpha(P\|Q) \le \varepsilon(\alpha)$ at one order, as the improved conversion does, at $\mu = 1$. At order 2, where $\varepsilon(2) = 1$, Mironov gives $1 + \ln(10^5) = 12.513$, the improved conversion $1 + \ln\frac12 + \ln(10^5) - \ln 2 = 11.127$, and the optimum 10.668. At orders 8 and 32 the improved conversion and the optimum agree to three decimals (5.214 and 16.228). [Exercise 6](../../book-exercises/part04-privacy-accounting/02-rdp-conversions.md#sec-sol-conv-6) explains why the gap appears only at small orders.

Balle et al. derive the same formula geometrically, as a tangent line. In the plane of test errors $(a, b)$, the $(\varepsilon, \delta)$ constraint $1 - a \le e^\varepsilon b + \delta$ is a straight line, and the best line that keeps a relaxed version of the region RDP allows (@sec-conv-tests) on its feasible side touches the region's boundary. @fig-conv-majorants shows the same tangency in the privacy-loss variable: the blue exponential touches the integrand at $z^*$.

### Example: the same release, converted better

For the mean released with $\sigma = 1$, the improved conversion over the grid gives $(4.729, 10^{-5})$-DP at order 5.4. Mironov's gave 5.299 at order 5.8, and the exact value is 4.377. The improved conversion removes 0.570 of the 0.921 excess, at no cost: it uses the same RDP curve and one extra line of code.

### Check yourself

- Why must the bounding function be an exponential in $z$, rather than, say, a polynomial that hugs the integrand more closely? *(Fact 2 only tells you $\mathbb E_P[e^{(\alpha - 1)L}]$. To bound $\mathbb E_P[g(L)]$ you need $g \le c\,e^{(\alpha - 1)z}$; any other $g$ has an expectation the RDP curve does not determine.)*
- The improved conversion can return a negative $\varepsilon'$ when $\varepsilon(\alpha)$ is tiny. What should you report? *($\varepsilon' = 0$. $\delta(\varepsilon)$ can only fall as $\varepsilon$ grows, so a bound at some $\varepsilon' < 0$ also holds at 0. The script clips at 0, as Canonne et al. do.)*

## RDP as a constraint on tests {#sec-conv-tests}

### Context

An $(\varepsilon, \delta)$ statement is one straight line in the plane of test errors: every test has $1 - a \le e^\varepsilon b + \delta$. An RDP curve says more. By Fact 3, each order constrains the pair $(a, b)$ of every test, and the constraints for different orders bend differently. This section draws the region an RDP curve allows, following Balle et al. (2020), compares it with the Gaussian's true trade-off curve, and uses it to compute the best possible conversion. The **trade-off function** $T(P, Q)(a)$ of [hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) is the smallest type II error any test can have at type I error $a$; it describes the most powerful possible attack at each false-positive rate.

### Statement

The statement turns Fact 3 into an explicit inequality on $(a, b)$ for each order, in both directions. Let $M$ have RDP curve $\varepsilon(\cdot)$ at the orders of a grid. Every test between $P = M(D)$ and $Q = M(D')$, with type I error $a$ and type II error $b$, satisfies for every order $\alpha$ in the grid
$$a^\alpha(1 - b)^{1 - \alpha} + (1 - a)^\alpha b^{1 - \alpha} \le e^{(\alpha - 1)\varepsilon(\alpha)}, \qquad (1 - b)^\alpha a^{1 - \alpha} + b^\alpha(1 - a)^{1 - \alpha} \le e^{(\alpha - 1)\varepsilon(\alpha)}.$$ {#eq-conv-region}
Call the set of $(a, b)$ satisfying all of these the **RDP privacy region**, and let $f_{\mathrm{RDP}}(a)$ be the smallest $b$ in it at type I error $a$. Two claims follow.

1. **Trade-off bound.** $T(P, Q)(a) \ge f_{\mathrm{RDP}}(a)$ for every $a$: no test is more powerful than the region allows.
2. **Optimal conversion.** Every pair consistent with the curve has
$$\delta(\varepsilon) \le \delta_{\mathrm{opt}}(\varepsilon) = \max_a\big[1 - a - e^\varepsilon f_{\mathrm{RDP}}(a)\big],$$ {#eq-conv-opt}
and some two-outcome pair consistent with the curve attains it. So $\delta_{\mathrm{opt}}$ is the best $\delta$ that any conversion using only these RDP values can certify.

### Derivation

The first three steps compute the Rényi divergence of a test's outcome and apply the RDP curve to it. The last two rewrite $\delta$ as a statement about tests, so that the region bounds it.

1. **The outcome of a test is a coin (Fact 3).** @tbl-conv-test-outcome lists the outcome's distribution under each hypothesis.

| Outcome | Probability under $P$ | Probability under $Q$ |
| --- | --- | --- |
| reject | $a$ | $1 - b$ |
| accept | $1 - a$ | $b$ |

: The outcome of a test with errors $(a, b)$, as a two-point distribution under each hypothesis. {#tbl-conv-test-outcome}

2. **Compute its Rényi divergence.** The definition $D_\alpha = \frac{1}{\alpha - 1}\ln\sum P^\alpha Q^{1 - \alpha}$ for a two-point distribution is a sum of two terms, one per row of @tbl-conv-test-outcome: $D_\alpha(\text{outcome under } P\,\|\,\text{outcome under } Q) = \frac{1}{\alpha - 1}\ln\big[a^\alpha(1 - b)^{1 - \alpha} + (1 - a)^\alpha b^{1 - \alpha}\big]$.
3. **Apply the RDP curve.** Data processing (Fact 3) gives this divergence $\le D_\alpha(P\|Q) \le \varepsilon(\alpha)$. Multiply by $\alpha - 1 > 0$ and exponentiate to get the first inequality of @eq-conv-region. The second comes the same way from $D_\alpha(Q\|P) \le \varepsilon(\alpha)$, which RDP also guarantees because swapping two neighbours gives two neighbours; it is the same sum with the columns of the table swapped. Since every test's $(a, b)$ lies in the region, its $b$ is at least the smallest $b$ the region allows at its $a$. This proves claim 1.
4. **Write $\delta$ as a best test.** Let $S$ be the acceptance region of a test, the outputs on which it accepts $P$, so $P(S) = 1 - a$ and $Q(S) = b$. Fact 1 says $\delta(\varepsilon)$ is the largest $P(S) - e^\varepsilon Q(S)$ over all events, so $\delta(\varepsilon) = \max_S[P(S) - e^\varepsilon Q(S)] = \max_\phi[1 - a - e^\varepsilon b]$. Every test's $(a, b)$ is in the region, so $b \ge f_{\mathrm{RDP}}(a)$, and replacing $b$ by $f_{\mathrm{RDP}}(a)$ can only increase the expression. That gives the bound in @eq-conv-opt.
5. **Show that the bound is attained.** Let $(a^*, b^*)$ maximize @eq-conv-opt, with $b^* = f_{\mathrm{RDP}}(a^*)$. Build a mechanism whose output is just the coin of @tbl-conv-test-outcome with these errors: $M(D) = \mathrm{Bern}(1 - a^*)$ and $M(D') = \mathrm{Bern}(b^*)$, where the outcome 1 is "accept". By step 2 its Rényi divergences in both directions are exactly the left sides of @eq-conv-region at $(a^*, b^*)$, which lie in the region, so the mechanism satisfies the curve. The test "accept on outcome 1" has errors $(a^*, b^*)$, so this mechanism's $\delta(\varepsilon)$ is at least $1 - a^* - e^\varepsilon b^*$, which is $\delta_{\mathrm{opt}}(\varepsilon)$.

**Computing it.** For fixed $a$, both left sides of @eq-conv-region equal 1 at $b = 1 - a$: the first becomes $a^\alpha a^{1 - \alpha} + (1 - a)^\alpha(1 - a)^{1 - \alpha} = a + (1 - a)$, the second likewise. That is where the test ignores the output and the two coins are identical. Both are convex in $b$ (each term has the form $x^\alpha y^{1 - \alpha}$, which is jointly convex in $(x, y)$ for $\alpha > 1$, by @sec-apx-tail-perspective), and their smallest value is at $b = 1 - a$, so they decrease on $[0, 1 - a]$. The constraints therefore hold for $b$ above some cut-off and fail below it, and $f_{\mathrm{RDP}}(a)$ is that cut-off, found by bisection. The script works with logarithms of $a$, $1 - a$, $b$ and $1 - b$, because the interesting tests have errors like $10^{-6}$. The objective in @eq-conv-opt is concave in $a$, because $f_{\mathrm{RDP}}$ is convex: the region is an intersection of convex sets, one per constraint, and the lower boundary of a convex set is a convex function. So a grid search refined by a one-dimensional bracketing search finds the maximum. The $\varepsilon$ at a given $\delta$ is the maximum over $a$ of $\ln\big((1 - a - \delta)/f_{\mathrm{RDP}}(a)\big)$, the $\varepsilon$ at which the line through the point $(a, f_{\mathrm{RDP}}(a))$ just reaches $\delta$.

### Interpretation

To see how loose the region is, compare it with the truth for the Gaussian. The best test between $\mathcal N(0, 1)$ and $\mathcal N(1, 1)$ rejects when the output exceeds a threshold $\tau$ (the Neyman–Pearson lemma: the best test thresholds the likelihood ratio, which here increases with the output). Its type I error is $a = 1 - \Phi(\tau)$, so $\tau = \Phi^{-1}(1 - a)$, and its type II error is the probability that an output from $\mathcal N(1, 1)$ falls below $\tau$, which is $G_1(a) = \Phi(\Phi^{-1}(1 - a) - 1)$. At $a = 0.1$, $\Phi^{-1}(0.9) = 1.2816$, so $G_1 = \Phi(0.2816) = 0.6109$ and the true power is $0.3891$. @fig-conv-tradeoff plots, for the Gaussian at $\mu = 1$, the largest power $1 - f_{\mathrm{RDP}}(a)$ that its RDP curve allows, against this true power $1 - G_1(a)$.

![Largest power $1 - b$ at type I error $a$ allowed by the Gaussian's RDP curve at $\mu = 1$ (red, all orders of the grid together), by the single orders 2 and 8 (blue), and achieved by the Gaussian itself (black). Left: linear axes. Right: log axes, with the $(\varepsilon, \delta)$ line $1 - b \le e^\varepsilon a + \delta$ from the improved conversion (grey), which touches the red curve.](figures/rdp_conversions_tradeoff.svg){#fig-conv-tradeoff}

Each order guards a different range of $a$. For small $a$ and $b$ near 1, the term $(1 - b)^\alpha a^{1 - \alpha}$ of the second inequality of @eq-conv-region is by far the larger, because $a^{1 - \alpha}$ blows up as $a \to 0$. Keeping only that term and solving for $1 - b$ gives $1 - b \lesssim e^{(\alpha - 1)\varepsilon(\alpha)/\alpha}\,a^{(\alpha - 1)/\alpha}$. On log axes this is a line of slope $(\alpha - 1)/\alpha$, with intercept $(\alpha - 1)\varepsilon(\alpha)/\alpha = (\alpha - 1)/2$ when $\varepsilon(\alpha) = \alpha/2$. Order 2 has slope $\frac12$ and is weak at small $a$: at $a = 10^{-6}$ it allows power $1.3118\times10^{-3}$, while order 8 (slope $\frac78$) allows $1.8622\times10^{-4}$. In the bulk the roles swap: at $a = 0.1$, order 2 allows $0.4932$ and order 8 allows $0.9838$. The two lines meet where $\frac12 + \frac12\ln a = \frac72 + \frac78\ln a$, that is $-3 = \frac38\ln a$, at $a = e^{-8} \approx 3.35\times10^{-4}$. The lines are bounds on the boundaries, not the boundaries themselves, because they drop a term. Order 8's line is exact at small $a$, but order 2's overstates its power: at $a = 10^{-6}$ it gives $e^{1/2}\cdot 10^{-3} \approx 1.6487\times10^{-3}$ against the exact $1.3118\times10^{-3}$. So the exact boundaries of the two orders cross a little lower, at $a \approx 1.87\times10^{-4}$, the value the script finds by solving for equal power. At $a = 10^{-3}$ neither is binding: the binding order is 3.7, which alone allows $2.4910\times10^{-2}$, the grid's value. Large orders guard small $a$, small orders the bulk.

The RDP region is a good but loose summary of the trade-off curve. @tbl-conv-power compares powers. The curve allows 1.24 to 1.36 times the power the Gaussian has, at every $a$ shown; at $a = 0.1$, for instance, $0.48376/0.38914 = 1.24$.

| Type I error $a$ | True power $1 - G_1(a)$ | Largest power allowed by the RDP curve | Ratio |
| --- | --- | --- | --- |
| 0.1 | $3.8914\times10^{-1}$ | $4.8376\times10^{-1}$ | 1.24 |
| $10^{-3}$ | $1.8298\times10^{-2}$ | $2.4910\times10^{-2}$ | 1.36 |
| $10^{-6}$ | $8.7218\times10^{-5}$ | $1.1636\times10^{-4}$ | 1.33 |

: Power of the best test against the Gaussian mechanism at $\mu = 1$, true and allowed by its RDP curve. {#tbl-conv-power}

The RDP region is not an $(\varepsilon, \delta)$ region. The grey line in @fig-conv-tradeoff is the improved conversion's guarantee. It touches the red curve at one small $a$ and lies far above it elsewhere, because an $(\varepsilon, \delta)$ pair is a single straight line, and the best it can do is be tangent to the curved region at one point. A whole trade-off curve is the envelope of such lines, one for each $\varepsilon$, which is the duality that [Gaussian DP](04-gaussian-dp.md) develops.

Nor does the region characterize RDP. A test squeezes the output into a single bit, and the Rényi divergence of a pair is in general larger than the largest Rényi divergence of any single test's outcome: information spread over many outputs is lost when they are merged into "accept" and "reject". So a pair can have every test inside the region and still violate the RDP curve. The region is what RDP implies about tests, not an equivalent definition.

### Example: a membership test against a released statistic

An auditor runs a membership test against the mean released with $\sigma = 1$ and wants a false-positive rate of $a = 10^{-3}$. The RDP curve alone guarantees that no test detects membership with probability above $2.4910\times10^{-2}$. The Gaussian's exact trade-off curve says the best test reaches $1.8298\times10^{-2}$. The $(4.729, 10^{-5})$ guarantee from the improved conversion only says power $\le e^\varepsilon a + \delta$, which is $e^{4.729}\times10^{-3} + 10^{-5} = 0.1131$, more than four times the region's bound. Reading an RDP result through one $(\varepsilon, \delta)$ pair throws away most of what it says about low false-positive rates.

### Check yourself

- Why are there two inequalities per order in @eq-conv-region? *(One for $D_\alpha(P\|Q)$ and one for $D_\alpha(Q\|P)$. RDP bounds both, because the neighbouring relation is symmetric; the first matters for tests with $a$ near 1, the second for $a$ near 0.)*
- Why is the optimum in @eq-conv-opt attained by a two-outcome mechanism, rather than needing a richer one? *(Any mechanism's $\delta$ is the best value of $1 - a - e^\varepsilon b$ over its tests, and each test only sees a coin. The coin with the extreme errors is itself a mechanism with the same constraints.)*

## Choosing the order {#sec-conv-order}

### Context

Both conversions hold at every order, so you take the best one. Each converted $\varepsilon'$ is the RDP curve, which grows with the order, plus a term in $\ln(1/\delta)/(\alpha - 1)$, which shrinks with it, so the best order balances the two. For the Gaussian the curve is linear in $\alpha$, and the best order has a closed form or nearly so. The formulas show how the best order moves with the number of steps and with $\delta$, and that tells you what an accountant's grid of orders must contain.

### Statement

Let $\varepsilon(\alpha) = \rho\alpha$, with slope $\rho = \mu^2/2 = k/(2\sigma^2)$ for the composed Gaussian. Over continuous $\alpha > 1$, the best orders are as follows. For **Mironov's conversion**, the best order is $\alpha^* = 1 + \sqrt{\ln(1/\delta)/\rho}$, and $\varepsilon^* = \rho + 2\sqrt{\rho\ln(1/\delta)}$. For the **improved conversion**, the best order solves
$$\rho(\alpha - 1)^2 = \ln\frac{1}{\alpha\delta},$$ {#eq-conv-alpha-star}
which the script solves by iterating $\alpha \leftarrow 1 + \sqrt{\ln(1/(\alpha\delta))/\rho}$ from Mironov's $\alpha^*$. The only difference from Mironov's condition is the extra $\alpha$ inside the logarithm, which pulls the best order slightly lower.

### Derivation

Each step sets a derivative to zero, as for any smooth minimum.

1. **Mironov.** The derivative of $\rho\alpha + \ln(1/\delta)/(\alpha - 1)$ is $\rho - \ln(1/\delta)/(\alpha - 1)^2$. Setting it to zero gives $(\alpha - 1)^2 = \ln(1/\delta)/\rho$, which is $\alpha^*$. Substitute back: $\rho\alpha^* = \rho + \rho\sqrt{\ln(1/\delta)/\rho} = \rho + \sqrt{\rho\ln(1/\delta)}$, and the second term is $\ln(1/\delta)/\sqrt{\ln(1/\delta)/\rho} = \sqrt{\rho\ln(1/\delta)}$ too. At the optimum the two terms contribute equally beyond $\rho$.
2. **Improved, derivative.** Write @eq-conv-improved as $\rho\alpha + \ln(1 - \frac1\alpha) + \frac{\ln(1/\delta) - \ln\alpha}{\alpha - 1}$ and differentiate term by term: $\frac{\mathrm d}{\mathrm d\alpha}\ln(1 - \frac1\alpha) = \frac{1}{\alpha(\alpha - 1)}$, and $\frac{\mathrm d}{\mathrm d\alpha}\frac{\ln(1/\delta) - \ln\alpha}{\alpha - 1} = -\frac{1}{\alpha(\alpha - 1)} - \frac{\ln(1/\delta) - \ln\alpha}{(\alpha - 1)^2}$ by the quotient rule. The $\frac{1}{\alpha(\alpha - 1)}$ terms cancel, leaving $\rho - \frac{\ln(1/(\alpha\delta))}{(\alpha - 1)^2}$. Setting it to zero gives @eq-conv-alpha-star.
3. **It is a minimum.** The derivative of that expression is $\frac{1}{\alpha(\alpha - 1)^2} + \frac{2\ln(1/(\alpha\delta))}{(\alpha - 1)^3}$, which is positive whenever $\alpha < 1/\delta$, so the objective is convex there and the stationary point is its minimum.
4. **Scaling.** Ignoring the slowly varying $\ln\alpha$, $\alpha^* - 1 \approx \sqrt{\ln(1/\delta)/\rho} = \sigma\sqrt{2\ln(1/\delta)/k}$. More steps or less noise push the best order toward 1; smaller $\delta$ pushes it up, but only through $\sqrt{\ln(1/\delta)}$.

### Interpretation

@fig-conv-orders shows the two converted $\varepsilon'(\alpha)$ for $\mu = 1$. Both are U-shaped: at small $\alpha$ the $\ln(1/\delta)/(\alpha - 1)$ term explodes, at large $\alpha$ the curve $\varepsilon(\alpha)$ itself does. The improved curve sits below Mironov's everywhere by $\Delta(\alpha)$ of @eq-conv-gain, and its minimum is at a slightly smaller order.

![Converted $\varepsilon$ at $\delta = 10^{-5}$ for each order of the grid, for the Gaussian at $\mu = 1$, with Mironov's conversion (red) and the improved one (blue). Dots mark the best order of each. The dashed line is the exact $\varepsilon$, and the grey curve is the RDP curve $\varepsilon(\alpha) = \alpha/2$ that both conversions add to.](figures/rdp_conversions_orders.svg){#fig-conv-orders}

@tbl-conv-orders shows how the best order moves over four settings. In the first row, for instance, $\rho = 1/(2\cdot 100) = 0.005$, and Mironov's best order is $1 + \sqrt{\ln(10^5)/0.005} = 1 + \sqrt{2302.6} = 48.985$; the improved conversion's extra $\ln\alpha$ brings it down to 40.525.

| $\sigma$ | $k$ | $\rho$ | Best order, improved (continuous / grid) | $\varepsilon$ continuous / grid | Mironov's best order |
| --- | --- | --- | --- | --- | --- |
| 10 | 1 | 0.005 | 40.525 / 41 | 0.375 / 0.375 | 48.985 |
| 1 | 1 | 0.5 | 5.432 / 5.4 | 4.728 / 4.729 | 5.799 |
| 1 | 100 | 50 | 1.472 / 1.5 | 96.035 / 96.116 | 1.480 |
| 1 | 10,000 | 5000 | 1.048 / 1.1 | 5475.791 / 5611.778 | 1.048 |

: Best orders for the composed Gaussian at $\delta = 10^{-5}$. {#tbl-conv-orders}

The grid needs orders near 1 for long runs with little noise. At $\rho = 5000$ the best order is 1.048, below the grid's smallest order 1.1, and the grid's answer is $5611.778 - 5475.791 = 135.987$ larger than the continuous optimum. (That setting is a huge $\varepsilon$ anyway; [Exercise 5](../../book-exercises/part04-privacy-accounting/02-rdp-conversions.md#sec-sol-conv-5) measures the relative loss.)

The grid also needs large orders, for small total $\rho$. One step at $\sigma = 10$ already wants order 40.525, and at $\rho = 5\times10^{-5}$ the best order is 338.312 ([Exercise 5](../../book-exercises/part04-privacy-accounting/02-rdp-conversions.md#sec-sol-conv-5)). Subsampled mechanisms ([subsampling](05-subsampling.md)), which use each record only in a random fraction of the steps, have tiny curves at small orders, which is why the grid ends at 128, 256 and 512.

The value of $\delta$ matters little, because it enters only through $\sqrt{\ln(1/\delta)}$. At $\mu = 1$, moving from $\delta = 10^{-5}$ to $10^{-10}$ doubles $\ln(1/\delta)$ but moves the improved best order only from 5.432 to 7.483.

### Example: tuning a grid for a training run

You plan $k = 100$ full-batch noisy gradient steps at $\sigma = 1$, so $\rho = 100/2 = 50$. @eq-conv-alpha-star puts the best order at 1.472, and the grid's 1.5 costs $96.116 - 96.035 = 0.081$ in $\varepsilon$. If you instead plan a single release at $\sigma = 10$, the best order is 40.525, and the grid's integer orders match it to three decimals of $\varepsilon$. One grid serves both because it is dense near 1 and runs from 1.1 to 512.

### Check yourself

- Why does the best order go to 1 as $k \to \infty$? *(The curve's slope $\rho$ grows with $k$, so the $\rho\alpha$ term punishes large orders more, while the $\ln(1/\delta)/(\alpha - 1)$ term does not change. The balance point $\alpha - 1 \approx \sqrt{\ln(1/\delta)/\rho}$ shrinks like $1/\sqrt k$.)*
- If an accountant reports the best order at the edge of its grid, what should you do? *(Extend the grid in that direction: the true optimum lies outside, and the reported $\varepsilon$ is larger than necessary.)*

## The slack no conversion removes {#sec-conv-slack}

### Context

The improved conversion is better than Mironov's, and the optimal conversion of @eq-conv-opt is better still, in principle. Yet both stay above the exact $\varepsilon$ of the Gaussian. Is that a weakness of the formulas, or of the summary? This section shows it is the summary: a different mechanism satisfies the same RDP curve and has a larger $\delta$, and a conversion that sees only the curve must cover both. It also looks at the opposite direction, from $(\varepsilon, \delta)$ to RDP.

### Statement

The three claims are about what the RDP curve can and cannot carry.

1. **The curve does not determine $\delta$.** For the Gaussian, the optimal conversion from the RDP values on the grid agrees with the improved conversion to three decimals, and both exceed the exact $\varepsilon$ by 5 to 10% (@tbl-conv-slack). The excess is real: a two-outcome mechanism satisfies the Gaussian's RDP curve $\alpha/2$ at every order $\alpha > 1$ and has $\delta(\varepsilon) = 4.4698\times10^{-5}$ at the Gaussian's exact $\varepsilon = 4.377$.
2. **$(\varepsilon, \delta)$-DP with $\delta > 0$ implies no RDP bound at any order.**
3. **$\varepsilon$-DP implies $(\alpha, \min(\varepsilon, \alpha\varepsilon^2/2))$-RDP for every $\alpha > 1$** (Bun & Steinke 2016, Proposition 3.3). The constant $\tfrac12\varepsilon^2$ can be sharpened to $\varepsilon\tanh(\varepsilon/2)$ (@eq-cdp-pure-sharp in [concentrated DP](03-concentrated-dp.md)).

### Derivation

**Claim 1.** The idea is to build the worst mechanism the curve allows and compute its $\delta$. By @eq-conv-opt, the largest $\delta$ consistent with a set of orders is attained by the two-outcome pair at the maximizer. On the part's grid, at $\mu = 1$ and $\varepsilon = 4.377$, that pair has $\delta = 4.4700\times10^{-5}$. But it satisfies the curve only at the grid's orders: its Rényi divergence is a smooth function of the order, and between two grid orders it can poke above $\alpha/2$. To get a pair that satisfies the curve at every real order, the script maximizes over a dense set of orders (steps of 0.002 from 1.01 to 30, and every integer up to 512) and then raises $Q(\text{accept})$ by a relative $10^{-5}$, which pushes the pair strictly inside the curve. @tbl-conv-worst-pair shows the result. The privacy loss column is $\ln(P/Q)$ for each outcome: $\ln(2.2779\times10^{-4}/2.2998\times10^{-6}) = 4.5956$ for "accept", and for "reject" a tiny negative number, because $P$ puts slightly less mass there than $Q$.

| Outcome | $P = M(D)$ | $Q = M(D')$ | Privacy loss $L$ |
| --- | --- | --- | --- |
| accept | $2.2779\times10^{-4}$ | $2.2998\times10^{-6}$ | 4.5956 |
| reject | rest | rest | $-2.255\times10^{-4}$ |

: A two-outcome mechanism that satisfies the Gaussian's RDP curve at $\mu = 1$ at every order and has nearly the largest $\delta$ at $\varepsilon = 4.377$ that any such mechanism can have. {#tbl-conv-worst-pair}

1. **Check that it satisfies the curve.** On a scan of $\alpha$ from $1 + 10^{-8}$ to $10^6$ (steps of $10^{-4}$ between 2 and 30), refined around the maximum, the largest value of $D_\alpha(P\|Q) - \alpha/2$ is $-9.97\times10^{-6}$, at $\alpha = 5.0955$; it is negative, so the curve holds everywhere scanned. The reverse divergences $D_\alpha(Q\|P)$ stay about 0.5 below the curve, closest as $\alpha \to 1$. The limits are safe too: as $\alpha \to \infty$, $D_\alpha(P\|Q)$ tends to the largest privacy loss, $4.5956$, while $\alpha/2$ grows without bound. The rounded values in the table also satisfy the curve, with largest excess $-1.93\times10^{-5}$.
2. **Compute its $\delta$.** By @eq-conv-delta only outcomes with $L > \varepsilon$ contribute, and here that is only "accept". So $\delta(4.377) = P(\text{accept}) - e^{4.377}\,Q(\text{accept})$; with $e^{4.377} \approx 79.6$ this is $2.2779\times10^{-4}$ minus about $1.831\times10^{-4}$, which is $4.4698\times10^{-5}$ with unrounded values, against the Gaussian's $10^{-5}$.
3. **Conclude.** So no curve-based method can certify $(4.377, 10^{-5})$ for the Gaussian, because the same certificate would have to cover this mechanism, which has the same curve and about 4.5 times the $\delta$. The improved conversion's $\delta$ at that $\varepsilon$ is $4.4701\times10^{-5}$, within $10^{-8}$ of this pair's.

The pair puts almost all its mass at $L \approx 0$ and a sliver at $L = 4.5956$, near the touching point $z^*$ of @fig-conv-majorants. That is the distribution for which the improved conversion's exponential bound is nearly an equality, since the exponential and the integrand meet there. It is very unlike the Gaussian PRV, which spreads its tail smoothly over all large losses. The RDP curve cannot tell them apart: it only sees MGF values (Fact 2).

**Claim 2.** A single example suffices. Take the mechanism that outputs nothing, except that with probability $\delta$ it publishes your record. Order the pair with the dataset that holds you first, $P = M(D')$ and $Q = M(D)$, which a guarantee must also cover. Then $P$ puts mass $\delta$ on an output that $Q$ never produces, and otherwise matches $Q$. Moving mass $\delta$ can change the probability of any event by at most $\delta$, so $P(S) \le Q(S) + \delta$ for every event, and the mechanism is $(0, \delta)$-DP. But the sum $\sum P^\alpha Q^{1 - \alpha}$ in the Rényi divergence contains the term $\delta^\alpha\cdot 0^{1 - \alpha}$ for the published record, and $0^{1 - \alpha} = \infty$ for $\alpha > 1$, so $D_\alpha(P\|Q) = \infty$ at every order. RDP, like pure DP, rules out catastrophic events however rare; $(\varepsilon, \delta)$-DP allows them with probability $\delta$.

**Claim 3.** The first bound, $\varepsilon(\alpha) \le \varepsilon$, follows directly from Fact 2. $\varepsilon$-DP says the privacy loss never exceeds $\varepsilon$, so $e^{(\alpha - 1)L} \le e^{(\alpha - 1)\varepsilon}$ for every output, and its average under $P$ is at most $e^{(\alpha - 1)\varepsilon}$; taking logarithms and dividing by $\alpha - 1$ gives $D_\alpha \le \varepsilon$. (Equivalently, Rényi divergence is non-decreasing in $\alpha$ and tends to the max-divergence, the largest privacy loss, which $\varepsilon$-DP bounds by $\varepsilon$; see [divergences](../part01-prerequisites/03-divergences.md).) The second bound, $\alpha\varepsilon^2/2$, says an $\varepsilon$-DP mechanism is $\varepsilon^2/2$-zero-concentrated: the logarithm of the MGF of its privacy loss grows at most quadratically. Bun & Steinke (2016, Proposition 3.3) prove it through a two-point variable and an inequality between hyperbolic sines; [concentrated DP](03-concentrated-dp.md#sec-cdp-pure) gives a shorter proof with Hoeffding's lemma, and also proves the sharper line $\alpha\varepsilon\tanh(\varepsilon/2)$ of @eq-cdp-pure-sharp, which randomized response attains as $\alpha \to 1$. This section uses both without proof. For randomized response at $\varepsilon = 0.5$, the bound at order 2 is $\min(0.5,\ 2\cdot 0.25/2) = 0.2500$, at order 4 it is $\min(0.5,\ 4\cdot 0.25/2) = 0.5000$, and at order 10 it is $\min(0.5,\ 1.25) = 0.5000$; the script gives $D_2 = 0.2273$ against the bound 0.2500, and $D_4 = 0.3519$ and $D_{10} = 0.4473$ against 0.5000.

### Interpretation

@tbl-conv-slack and @fig-conv-slack compare all four answers for the Gaussian with $\sigma = 10$ per step. The per-step noise is chosen so that $\mu = \sqrt k/10$ runs from 0.1 to 10 over $k = 1$ to $10{,}000$, and the exact $\varepsilon$ from 0.341 to 91.817: the range where accounting choices matter. Because everything depends only on $\mu$, the row $k = 100$, where $\mu = 1$, repeats the single release at $\sigma = 1$ of the earlier sections.

| $k$ | $\mu$ | Exact | Optimal from the curve | Improved | Mironov | Ratios to exact (opt. / impr. / Mir.) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.100 | 0.341 | 0.375 | 0.375 | 0.485 | 1.10 / 1.10 / 1.42 |
| 10 | 0.316 | 1.199 | 1.308 | 1.308 | 1.568 | 1.09 / 1.09 / 1.31 |
| 100 | 1.000 | 4.377 | 4.729 | 4.729 | 5.299 | 1.08 / 1.08 / 1.21 |
| 1000 | 3.162 | 17.857 | 19.054 | 19.054 | 20.175 | 1.07 / 1.07 / 1.13 |
| 10,000 | 10.000 | 91.817 | 96.116 | 96.116 | 98.026 | 1.05 / 1.05 / 1.07 |

: $\varepsilon$ at $\delta = 10^{-5}$ for the Gaussian with $\sigma = 10$ composed $k$ times: exact, the best any conversion from the RDP grid can give, the improved conversion and Mironov's. {#tbl-conv-slack}

![Left: $\varepsilon$ at $\delta = 10^{-5}$ for the Gaussian with $\sigma = 10$ composed $k$ times, exact (black), optimal from the RDP curve (wide green), improved (blue, on top of the green) and Mironov (red). Right: each divided by the exact value. The improved conversion is optimal for the curve; the remaining 5 to 10% belongs to the RDP summary itself.](figures/rdp_conversions_slack.svg){#fig-conv-slack}

The optimal and improved conversions coincide. At $\mu = 1$ they are 4.728505 and 4.728507, a difference of $2\times10^{-6}$. The optimal conversion can use all orders at once, but that helps only if more than one order constrains the worst mechanism. Here it buys nothing measurable over the best single order: at $\varepsilon = 4.377$ the grid's worst pair is held back only by the constraint at order 5.1, and every other order of the grid leaves at least 0.0011 of room.

Mironov's excess shrinks fast with $k$, the unavoidable excess slowly. Mironov is 42% above the exact value at $k = 1$ and 7% at $k = 10{,}000$; the RDP summary costs 10% and 5%. Both shrink in relative terms because for large $\mu$ the $\rho\alpha$ part, which every method shares, dominates.

When the slack matters, change the summary, not the formula. To get the exact 4.377 you must track more than an MGF: the trade-off function ([Gaussian DP](04-gaussian-dp.md)) or the full privacy loss distribution ([numerical accounting](06-numerical-accounting.md)).

The 5 to 10% is for the plain Gaussian. The differentially private version of stochastic gradient descent (SGD), DP-SGD, composes subsampled Gaussian steps, in which each step uses a random fraction $q$ of the records, and their RDP curve is not a line. [Subsampling](05-subsampling.md) applies the same conversions to the part's running example ($q = 0.01$, $\sigma = 1$): 7.43 with Mironov's conversion and 6.71 with the improved one after $T = 10{,}000$ steps. There the gap to the exact value can be much larger, and [choosing an accountant](09-choosing-an-accountant.md) measures it.

### Example: what to tell a reviewer

Your paper reports $(4.729, 10^{-5})$-DP from an RDP accountant for a mechanism that is in fact a Gaussian with $\mu = 1$. A reviewer asks whether a better conversion would lower it. The answer is no. The two-outcome mechanism of @tbl-conv-worst-pair satisfies the same RDP curve and has $\delta = 4.4698\times10^{-5}$ at $\varepsilon = 4.377$, so any certificate computed from the curve alone must also hold for it. The same construction at $\delta = 10^{-5}$ gives a pair inside the curve at every order whose $\varepsilon$ is 4.728376, and the improved conversion at the best real order, 5.4318, gives 4.728387. The first is a mechanism that a certificate must cover, so the best certificate is at least 4.728376; the second is a valid certificate, so the best is at most 4.728387. So the best value any method can certify from the curve at every real order lies between 4.728376 and 4.728387, which is 4.7284 to four decimals; a method that sees only the grid's orders must report at least 4.728505. A lower number needs an accountant that sees the mechanism, and for the Gaussian that number is 4.377.

### Check yourself

- Does the two-outcome mechanism contradict the Gaussian's $\delta = 10^{-5}$ at $\varepsilon = 4.377$? *(No. It is a different mechanism that satisfies the same RDP curve. The conversion must cover both, so it cannot certify the smaller $\delta$ for either.)*
- A mechanism is $(1, 10^{-8})$-DP. What RDP curve does that give you? *(None: $\delta > 0$ allows a rare event with infinite privacy loss, and claim 2 shows that makes every $D_\alpha$ infinite. You need to know how the $\delta$ arises, for example from Gaussian noise, to get an RDP curve.)*

## Code: check every conversion by computation

The script [code/rdp_conversions.py](../../code/part04-privacy-accounting/rdp_conversions.py) computes every number in this chapter and draws its figures. The conversions are small typed functions: `rdp_to_dp_mironov` and `rdp_to_dp_improved` return the best $\varepsilon$ and order over a grid, `rdp_to_delta_improved` goes the other way, `rdp_tradeoff_lower_bound` computes $f_{\mathrm{RDP}}$ by bisection in logarithms, `rdp_to_dp_tradeoff` maximizes @eq-conv-opt, `continuous_worst_pair` builds the two-outcome pair of @tbl-conv-worst-pair and `max_gaussian_curve_excess` checks it against the curve at every real order, and `gaussian_delta_exact` and `gaussian_eps_exact` give the exact Gaussian answers. From the repository root:

```bash
uv run python code/part04-privacy-accounting/rdp_conversions.py
```

**Output** (seed 0; nothing is random):

| Experiment | Result |
| --- | --- |
| Fact 1, Gaussian $\mu = 1$, $\varepsilon = 1$ | closed form $1.2694\times10^{-1}$; Riemann sum $1.2694\times10^{-1}$ |
| Mironov at $\mu = 1$ | 5.299 at order 5.8; continuous optimum 5.299 at 5.799 |
| Optimal constant at $\alpha = 6$, $\varepsilon = 4$ | formula $1.380552\times10^{-10}$; grid $1.380552\times10^{-10}$ |
| Improved at $\mu = 1$ | 4.729 at order 5.4; exact 4.377 |
| Single order 2, $\varepsilon(2) = 1$ | Mironov 12.513, improved 11.127, optimal 10.668 |
| Power at $a = 10^{-3}$, $\mu = 1$ | true $1.8298\times10^{-2}$; RDP allows $2.4910\times10^{-2}$ |
| Best order, $\rho = 5000$ | continuous 1.048; grid 1.1 |
| Optimal from the curve vs improved, $\mu = 1$ | 4.728505 and 4.728507 |
| Worst two-outcome pair at $\varepsilon = 4.377$ | grid orders only: $\delta = 4.4700\times10^{-5}$; every real order: $\delta = 4.4698\times10^{-5}$, largest excess over $\alpha/2$ is $-9.97\times10^{-6}$ |
| Worst pair at $\delta = 10^{-5}$, every real order | $\varepsilon = 4.728376$; improved at order 5.4318: 4.728387 |
| Randomized response $\varepsilon = 0.5$, order 2 | $D_2 = 0.2273 \le 0.2500$ |

: Results printed by the script. {#tbl-conv-code}

[tests/test_rdp_conversions.py](../../tests/test_rdp_conversions.py) checks that the output still matches @tbl-conv-code and every other number in the text.

**What to take from it.** The improved conversion is free and nearly optimal: it agrees with the optimal conversion from the curve to three decimals at every $k$ of @tbl-conv-slack, and at $\mu = 1$ the two differ by $2\times10^{-6}$. The gap that remains to the exact value is real, because a concrete two-outcome mechanism with the same curve realizes it, so no formula can close it. What still matters is the grid, where it is sparse or ends: very long runs push the best order below 1.1, and very short ones push it into the sparse stretch between 63 and 512, or beyond 512.

## References

- Mironov, "Rényi Differential Privacy" (IEEE CSF 2017): the definition, composition, the Gaussian and Laplace curves, and the conversion of @eq-conv-mironov (Proposition 3).
- Balle, Barthe, Gaboardi, Hsu & Sato, "Hypothesis Testing Interpretations and Rényi Differential Privacy" (AISTATS 2020; arXiv:1905.09982): privacy regions of divergences, the test region of @eq-conv-region (§5.1.1), and the improved conversion as a tangent line (Theorem 21).
- Canonne, Kamath & Steinke, "The Discrete Gaussian for Differential Privacy" (NeurIPS 2020; arXiv:2004.00010): the short proof of the improved conversion through the optimal constant $c^*$ (Proposition 12), and its use for concentrated DP (Corollary 13).
- Asoodeh, Liao, Calmon, Kosut & Sankar, "A Better Bound Gives a Hundred Rounds: Enhanced Privacy Guarantees via f-Divergences" (IEEE International Symposium on Information Theory, ISIT 2020; arXiv:2001.05990): an independent proof of the improved conversion (Lemma 1, Eq. 20), applied to noisy SGD.
- Asoodeh, Liao, Calmon, Kosut & Sankar, "Three Variants of Differential Privacy: Lossless Conversion and Applications" (IEEE Journal on Selected Areas in Information Theory, 2021; arXiv:2008.06529): the optimal conversion from one Rényi order through the joint range of two $f$-divergences, and RDP as a trade-off between type I and type II errors.
- Bun & Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds" (TCC 2016): $\varepsilon$-DP implies $\varepsilon^2/2$-zero-concentrated DP (Proposition 3.3) and the standard zero-concentrated conversion.
- Steinke, "Composition of Differential Privacy & Privacy Amplification by Subsampling" (2022 survey, arXiv:2210.00597): a unified treatment of privacy loss, RDP, conversions and their tightness.
- Opacus (`opacus/accountants/analysis/rdp.py`) and Google's `dp_accounting` (`rdp_privacy_accountant.py`): production implementations of the improved conversion, citing Balle et al. Theorem 21 and Canonne et al. Proposition 12 respectively.

## Cheat sheet

@tbl-conv-cheat-sheet collects every result in the chapter.

| Result | Statement | Where it comes from |
| --- | --- | --- |
| $\delta$ as an expectation | $\delta(\varepsilon) = \mathbb E_P[(1 - e^{\varepsilon - L})_+]$ | Hockey-stick divergence |
| RDP as an MGF | $\mathbb E_P[e^{(\alpha - 1)L}] = e^{(\alpha - 1)D_\alpha(P\Vert Q)}$ | Definition of $D_\alpha$ |
| Mironov | $\varepsilon' = \varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1)$ | Indicator, then Markov |
| Improved | $\varepsilon' = \varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}$; $\delta = \frac{e^{(\alpha - 1)(\varepsilon(\alpha) - \varepsilon)}}{\alpha}(1 - \frac1\alpha)^{\alpha - 1}$ | Smallest exponential above the integrand |
| Gain at a fixed order | $\Delta(\alpha) = \frac{\ln\alpha}{\alpha - 1} + \ln\frac{\alpha}{\alpha - 1}$ in $\varepsilon$; factor $\approx 1/(e\alpha)$ in $\delta$ | Subtracting the two |
| Test region | $a^\alpha(1 - b)^{1 - \alpha} + (1 - a)^\alpha b^{1 - \alpha} \le e^{(\alpha - 1)\varepsilon(\alpha)}$, and with $a, b$ swapped | Data processing on the test's outcome |
| Optimal conversion | $\delta_{\mathrm{opt}} = \max_a[1 - a - e^\varepsilon f_{\mathrm{RDP}}(a)]$, attained by a two-outcome pair | Region plus attainment |
| Best order, linear curve | Mironov $\alpha^* = 1 + \sqrt{\ln(1/\delta)/\rho}$; improved $\rho(\alpha - 1)^2 = \ln(1/(\alpha\delta))$ | Calculus |
| Unavoidable slack (Gaussian) | 5 to 10% of $\varepsilon$ at $\delta = 10^{-5}$ | A two-outcome pair inside the same curve |
| $(\varepsilon, \delta)$ ⇒ RDP | False for any $\delta > 0$ | An atom of $P$ that $Q$ lacks |
| $\varepsilon$-DP ⇒ RDP | $\varepsilon(\alpha) \le \min(\varepsilon, \alpha\varepsilon^2/2)$; sharply $\alpha\varepsilon\tanh(\varepsilon/2)$ in place of $\alpha\varepsilon^2/2$ | Monotonicity; Bun & Steinke, proved in [concentrated DP](03-concentrated-dp.md#sec-cdp-pure) |

: Every result in this chapter. {#tbl-conv-cheat-sheet}

**Rules of thumb**

- Always use the improved conversion. It saves $\Delta(\alpha)$, about 0.36 at order 10, and on the Gaussian it matches the best any method can do with the curve to three decimals.
- Check the best order. If it sits at either end of the grid, extend the grid.
- Expect an RDP accountant to overstate the plain Gaussian's $\varepsilon$ by 5 to 10%, and subsampled DP-SGD's by more. If that matters, use a privacy loss distribution accountant.
- Read an RDP guarantee as a bound on tests with the whole region, not with one $(\varepsilon, \delta)$ line: at low false-positive rates the region is much tighter.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part04-privacy-accounting/02-rdp-conversions.md).
