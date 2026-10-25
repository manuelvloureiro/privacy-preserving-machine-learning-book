# Choosing an accountant

## Why this matters

You have trained a model with differential privacy (DP). The training loop was stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, called DP-SGD. At each step every record joins the batch independently with probability $q$ (Poisson sampling), each record's gradient is clipped to norm at most $C$, and Gaussian noise of standard deviation $\sigma C$ is added to the sum, so $\sigma$, the **noise multiplier**, says how large the noise is compared with one record's largest possible contribution. Your run used $q = 0.01$, $\sigma = 1$ and $T = 10{,}000$ steps. You must now write one sentence in a paper or a model card: "the model is $(\varepsilon, 10^{-5})$-DP". That sentence means: for any two datasets that differ in one record, and for any event about what training released, the probability of the event on one dataset is at most $e^\varepsilon$ times its probability on the other, plus $\delta = 10^{-5}$. A small $\varepsilon$ means one record barely changes what anyone can see; $\delta$ is a small allowance for that promise to fail.

The run is fixed, so its true $\varepsilon$ is one number. But nobody can compute it exactly, and a procedure that bounds or estimates it is called an **accountant**. The chapters of this part give you six of them. Composing per-step guarantees gives 11296.93. Rényi differential privacy (RDP) gives 7.4287 or 6.7127, depending on the conversion. The Gaussian differential privacy (GDP) central limit theorem (CLT) gives 6.0071. A privacy loss distribution (PLD) accountant gives an interval, $[6.1377, 6.2377]$. These numbers describe the same training run. Which one do you report, and what else must you say so that someone else can check it?

### The one idea: bounds and approximations

The numbers differ because the accountants make two different kinds of statement, and you can only report one kind.

An **upper bound** comes with a proof that holds for every pair of neighbouring datasets. It may be loose, so the true $\varepsilon$ may be smaller, but the run really is $(\varepsilon, \delta)$-DP with the $\varepsilon$ it reports. Per-step composition, RDP with any of its conversions, and the pessimistic PLD are upper bounds. An **approximation** has no such proof. It may land close to the truth, but nothing stops it from landing below, and then it reports a number for which the run is *not* $(\varepsilon, \delta)$-DP. The GDP central-limit approximation is of this kind, and on the running example it does land below: it falls under the PLD *lower* bound at every $T$.

The optimistic PLD makes a third kind of statement, a certified **lower bound**: the true $\varepsilon$ is at least this large. It is not a privacy guarantee, since it says the run is no better than this, but it tells you how far any valid upper bound can fall. The true $\varepsilon$ sits between the pessimistic and the optimistic PLD.

So the rule is short. **Report the tightest upper bound whose assumptions hold for the code you ran, and report enough that someone else can recompute it.** Each section of this chapter unpacks one clause of that rule. @tbl-choose-overview lists the sections.

| Section | Question | Where you use it |
| --- | --- | --- |
| @sec-choose-compare | How do the accountants compare on one DP-SGD run, and which are bounds? | Picking an accountant |
| @sec-choose-inverse | How much noise does each accountant ask for at a target $\varepsilon$? | Calibrating $\sigma$ before training |
| @sec-choose-decision | Which accountant is tight for which mechanism? | Accounting for anything other than DP-SGD |
| @sec-choose-match | Do the accountant's assumptions match the training code? | Code review of a private training pipeline |
| @sec-choose-report | What must a privacy statement contain? | Papers, model cards, audits |
| @sec-choose-check | How do you check an accountant? | Testing your own, or someone else's, accountant |

: The sections of this chapter and where each one is used. {#tbl-choose-overview}

This chapter adds little new theory: the ordering of the bounds and the monotonicity behind the inverse problem are the only new derivations. Every accountant is the one derived earlier in the part, and the script imports it from that chapter's code rather than re-implementing it. Each section below restates what an accountant does where it uses it, so you can read this chapter on its own. The running example and the order grid are those of the whole part.

## Every accountant on one run {#sec-choose-compare}

### Context

This is the question the part has built up to: for DP-SGD with $q = 0.01$, $\sigma = 1$ and up to $T = 10{,}000$ steps, compute $\varepsilon$ at $\delta = 10^{-5}$ with every accountant, and see what choosing a loose one costs. [Amplification by subsampling](05-subsampling.md) and [Numerical accounting](06-numerical-accounting.md) each computed some of these numbers. This section puts them in one table, adds the optimal conversion of an RDP curve, and says for each accountant what it assumes and whether it is a bound.

### Statement

All accountants start from the same object: what one step of DP-SGD reveals about one record. Look at the noisy gradient sum along the direction of that record's clipped gradient, and measure it in units of the clipping norm $C$. Without the record, the output is pure noise, $\mathcal N(0, \sigma^2)$. With the record, the record is in the batch with probability $q$, and then its gradient shifts the output by 1; otherwise it is absent and the output is the same noise. So the output is the mixture $(1 - q)\mathcal N(0, \sigma^2) + q\mathcal N(1, \sigma^2)$. The accountants differ in what they keep of this pair and how they combine $T$ copies of it. Here they are, each with where it was derived:

1. **Per step, then composition** ([Amplification by subsampling](05-subsampling.md), @sec-sub-dpsgd). First find an $(\varepsilon_0, \delta_0)$ guarantee for one Gaussian step on its own. Then use subsampling: because the record is missing from the batch with probability $1 - q$, the factor $e^{\varepsilon_0}$ by which it can change a probability is diluted to $1 + q(e^{\varepsilon_0} - 1)$, which is @eq-sub-amplify, $\varepsilon_1 = \ln(1 + q(e^{\varepsilon_0} - 1))$. Finally compose $T$ steps, either by basic composition, where the $\varepsilon$'s simply add to $T\varepsilon_1$, or by advanced composition, which grows like $\sqrt T$ at the price of an extra $\delta'$.
2. **RDP, Mironov's conversion** ([Rényi DP](01-renyi-dp.md)). RDP describes one step by a curve $\varepsilon(\alpha)$, the Rényi divergence of order $\alpha > 1$ between the two output distributions; for the subsampled Gaussian it is @eq-sub-rdp. Its appeal is that curves of composed steps simply add, so $T$ steps have curve $T\,\varepsilon(\alpha)$. To get back to $(\varepsilon, \delta)$, Mironov's conversion uses the fact that $\delta$ is at most the chance that the privacy loss $L$ (defined in item 6) exceeds $\varepsilon$. The curve fixes the moment $\mathbb E[e^{(\alpha - 1)L}] = e^{(\alpha - 1)T\varepsilon(\alpha)}$, so Markov's inequality gives $\Pr[L > \varepsilon] \le e^{(\alpha - 1)(T\varepsilon(\alpha) - \varepsilon)}$. Setting the right side equal to $\delta$ and solving gives, at each order, $\varepsilon = T\,\varepsilon(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}$, and the conversion takes the best order: $\varepsilon = \min_\alpha\big[T\,\varepsilon(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}\big]$.
3. **RDP, improved conversion** ([From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md), @eq-conv-improved). The same curve, converted with a sharper analysis of the same tail (Balle et al. 2020) that saves a fixed amount at each order: $\varepsilon = \min_\alpha\big[T\,\varepsilon(\alpha) + \ln\frac{\alpha - 1}\alpha - \frac{\ln\delta + \ln\alpha}{\alpha - 1}\big]$.
4. **RDP, optimal conversion** (@sec-conv-tests). The best any conversion of the curve can do: the smallest $\varepsilon$ that every pair of distributions with the curve $T\,\varepsilon(\alpha)$ satisfies at $\delta$.
5. **GDP central-limit approximation** ([f-DP and Gaussian DP](04-gaussian-dp.md), @eq-gdp-sgd-mu). A mechanism is **$\mu$-GDP** if telling its two output distributions apart is no easier than telling $\mathcal N(0, 1)$ from $\mathcal N(\mu, 1)$. The privacy loss of $T$ steps is a sum of $T$ small losses, so by the central limit theorem it is close to Gaussian, and the approximation treats the run as $\mu$-GDP with $\mu = q\sqrt{T(e^{1/\sigma^2} - 1)}$. It then reads $\varepsilon$ off the exact $(\varepsilon, \delta)$ curve of a $\mu$-GDP mechanism, @eq-gdp-delta.
6. **PLD, pessimistic and optimistic** ([Numerical accounting](06-numerical-accounting.md), @eq-num-sandwich). The **privacy loss** at an output $y$ is the log of how much more likely $y$ is on one of the two neighbouring datasets than on the other, and its distribution over outputs is the PLD. Each order of the pair gives its own PLD: the **remove** direction puts the dataset with the record first, so the loss measures how much more likely $y$ is with the record than without it, and the **add** direction puts it second. Under composition the losses add, so the PLD of $T$ steps is the $T$-fold convolution of one step's PLD, which a computer does quickly with the fast Fourier transform (FFT). The convolution needs the loss on a grid, here of width $h = 10^{-5}$. Rounding every loss up gives the pessimistic PLD, rounding it down the optimistic one. Solve each for $\varepsilon$, and take the larger of the add and remove directions, since the guarantee must hold whichever of the two datasets has the record (each direction's pair dominates its case, as @sec-apx-amp-poisson in the appendices proves).

@tbl-choose-main gives every accountant's $\varepsilon$ on the running example.

| $T$ | Basic | Advanced | RDP, Mironov | RDP, improved | RDP, optimal | GDP CLT | PLD upper | PLD lower |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.20 | 1.13 | 1.3175 | 0.9555 | 0.4899 | 0.0368 | 0.1995 | 0.1994 |
| 10 | 3.64 | 7.88 | 1.4145 | 1.0353 | 0.7951 | 0.1304 | 0.3800 | 0.3799 |
| 100 | 58.01 | 78.10 | 1.6118 | 1.2141 | 1.1538 | 0.4575 | 0.7185 | 0.7175 |
| 1,000 | 839.39 | 1258.64 | 2.5380 | 2.1014 | 2.1009 | 1.6177 | 1.8332 | 1.8232 |
| 10,000 | 11296.93 | 24379.55 | 7.4287 | 6.7127 | 6.7127 | 6.0071 | 6.2377 | 6.1377 |

: DP-SGD with $q = 0.01$, $\sigma = 1$: $\varepsilon$ at $\delta = 10^{-5}$ after $T$ steps under every accountant of this part. Basic and advanced are per-step amplification followed by composition; the PLD columns use the grid $h = 10^{-5}$. {#tbl-choose-main}

### Derivation: what each accountant assumes, and why it is a bound

Every row of @tbl-choose-main assumes the same model of the training run: Poisson sampling at rate $q$, add/remove neighbours (the two datasets differ by one record being present or absent), per-example clipping to norm $C$ and Gaussian noise of standard deviation $\sigma C$ on the summed gradients, and every intermediate noisy gradient released. Under that model, one step is the Poisson-subsampled Gaussian pair of the statement, $(1 - q)\mathcal N(0, \sigma^2) + q\mathcal N(1, \sigma^2)$ against $\mathcal N(0, \sigma^2)$, in units of $C$ ([Amplification by subsampling](05-subsampling.md)). The accountants differ only in what they keep of this pair, and each step below says whether what is kept still gives a proof.

1. **Per step, then composition: an upper bound.** One step is $(\varepsilon_1, q\delta_0)$-DP by the amplification theorem (@sec-sub-eps-delta), and basic and advanced composition hold even when each step is chosen after seeing the earlier outputs, which is how SGD works ([Composition](../part03-dp-foundations/04-composition.md)). So the result is a proof. It is very loose because each step's whole privacy loss distribution is replaced by just two numbers, $\varepsilon_1$ and $q\delta_0$, and composition has to assume the worst distribution consistent with them at every step.
2. **RDP with Mironov's or the improved conversion: an upper bound.** Three facts make this a proof. The subsampled curve @eq-sub-rdp is the exact Rényi divergence of the pair, not an estimate. RDP curves add under adaptive composition (@eq-rdp-compose), because the Rényi divergence is the log of a moment of the likelihood ratio, and the moment of a product of independent factors is the product of their moments. And both conversions are proved for every pair with the curve (@sec-conv-mironov, @sec-conv-improved). The accountant evaluates the curve only on a grid of orders, and minimizing over fewer orders can only give a larger $\varepsilon$, so the grid costs tightness, never validity.
3. **RDP with the optimal conversion: an upper bound.** It computes the largest $\delta(\varepsilon)$ over all pairs of distributions with the given RDP curve. The real pair has that curve, so it is in the set, and its $\delta(\varepsilon)$ is no larger than the maximum (@sec-conv-tests). The same argument shows it is the tightest bound any accountant that sees only the curve can give: some pair in the set attains the maximum, and such an accountant cannot tell that pair from the real one.
4. **The GDP central-limit approximation: not a bound.** @eq-gdp-sgd-mu comes from two approximations (@sec-gdp-sgd). It is a limit as $T \to \infty$ with $q\sqrt T$ held fixed, and it uses a second-order expansion of the per-step loss, keeping its mean and variance and dropping the rest. Neither step comes with an error bound in the tail of the loss distribution, and $\delta = 10^{-5}$ is read exactly there: it asks about the chance of a rare, large loss, which is where a Gaussian approximation is least reliable.
5. **The pessimistic PLD: an upper bound; the optimistic PLD: a lower bound.** Rounding every loss up describes a pair that leaks at least as much as the real one (a dominating pair), and composing two dominating pairs gives a dominating pair, so the rounding error keeps its sign through all $T$ convolutions. Rounding down does the same in the other direction. So $\delta^\downarrow(\varepsilon) \le \delta(\varepsilon) \le \delta^\uparrow(\varepsilon)$ for every $\varepsilon$ (@eq-num-sandwich). The upper bound holds under adaptive composition of the fixed per-step pair (@sec-apx-comp-dominating in the appendices proves it). How far apart are the two? Each step's loss moves by less than one grid width $h$ between rounding down and rounding up, so after $T$ steps the two composed losses differ by at most $Th$, and so do the $\varepsilon$'s they give, as long as no tail is truncated (@sec-num-discretize). Here $Th = 10{,}000 \times 10^{-5} = 0.1$.

Steps 2, 3 and 5 put the bounds in two orders, which together read
$$\varepsilon_{\text{RDP, Mironov}} \;\ge\; \varepsilon_{\text{RDP, improved}} \;\ge\; \varepsilon_{\text{RDP, optimal}} \;\ge\; \varepsilon_{\text{true}}, \qquad \varepsilon_{\text{PLD}}^{\uparrow} \;\ge\; \varepsilon_{\text{true}} \;\ge\; \varepsilon_{\text{PLD}}^{\downarrow}.$$ {#eq-choose-order}
The second chain is the sandwich of step 5. In the first chain, the last inequality is step 3. Mironov's conversion is above the improved one because at every order the improved one is smaller by $\Delta(\alpha) = \frac{\ln\alpha}{\alpha - 1} + \ln\frac{\alpha}{\alpha - 1} > 0$ (@eq-conv-gain); both terms are positive for $\alpha > 1$, and if one expression is above another at every order, its minimum over the grid is above too. The improved conversion is above the optimal one because the optimal one is, by definition, the smallest $\varepsilon$ valid for every pair with the curve, and the improved one is one such valid $\varepsilon$.

The theory does not order the two chains against each other. You might expect the RDP bounds to always sit above the PLD upper bound, because RDP throws away information. But the pessimistic PLD describes a rounded-up pair, not the real one, and its $\varepsilon$ may exceed the truth by up to $Th$. A bound from the curve alone that came very close to the truth could therefore fall below it. Without truncation, all that follows is $\varepsilon_{\text{RDP, optimal}} \ge \varepsilon_{\text{true}} \ge \varepsilon^\uparrow_{\text{PLD}} - Th$. On the running example the script checks both chains at all nine values of $T$ from 1 to 10,000. It also finds RDP optimal $\ge$ PLD upper at every one of them, because the curve slack (at least 0.2677, from @tbl-choose-slack below) is far larger than $Th$ (at most 0.1000). That is an observation about this example, not a theorem. The script also checks that the CLT value is below the PLD lower bound at every $T$.

### Interpretation

@fig-choose-eps-k plots the table on a log scale, with each accountant's ratio to the PLD upper bound beside it. @tbl-choose-ratio lists the ratios. Since the PLD upper bound is within $Th$ of the truth, a ratio well above 1 measures how much an accountant overstates $\varepsilon$, and a ratio below the PLD lower column means the number is not a valid guarantee.

| $T$ | Basic | Advanced | RDP, Mironov | RDP, improved | RDP, optimal | GDP CLT | PLD lower |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.00 | 5.66 | 6.61 | 4.79 | 2.46 | 0.18 | 1.00 |
| 10 | 9.58 | 20.74 | 3.72 | 2.72 | 2.09 | 0.34 | 1.00 |
| 100 | 80.73 | 108.69 | 2.24 | 1.69 | 1.61 | 0.64 | 1.00 |
| 1,000 | 457.88 | 686.57 | 1.38 | 1.15 | 1.15 | 0.88 | 0.99 |
| 10,000 | 1811.07 | 3908.41 | 1.19 | 1.08 | 1.08 | 0.96 | 0.98 |

: Each accountant's $\varepsilon$ divided by the PLD upper bound, from @tbl-choose-main. {#tbl-choose-ratio}

**Per-step composition is useless beyond a handful of steps.** It is the right answer only at $T = 1$, where there is nothing to compose and basic composition is just the amplified one-step guarantee. At $T = 10{,}000$ it is off by a factor of over a thousand, because adding $\varepsilon$'s assumes every step leaks its worst case, while in reality the losses of different steps mostly cancel. Advanced composition is worse than basic here, for the reason given in @sec-sub-dpsgd. Its bound is $\varepsilon_1\sqrt{2T\ln(1/\delta')} + T\varepsilon_1(e^{\varepsilon_1} - 1)$: a $\sqrt T$ term with a large constant (at $\delta' = 10^{-5}$, $\sqrt{2\ln(10^5)} \approx 4.80$), plus a term that still grows like $T$. The $\sqrt T$ only wins when $\varepsilon_1$ is so small that the second term is negligible, and the amplified per-step $\varepsilon_1$ is too large for that.

**RDP is a safe bound, and its slack has two sources.** RDP with the improved conversion exceeds the PLD upper bound for two separate reasons, and @tbl-choose-slack splits the excess accordingly. The first is the conversion: turning the curve into one $(\varepsilon, \delta)$ pair with the improved formula loses against the best possible conversion of the same curve, and that loss is improved minus optimal. The second is the curve itself: even the best conversion of the curve is above the PLD, because the curve keeps only a family of moments of the privacy loss, not its whole distribution, and that loss is optimal minus PLD upper.

**Conversion slack dominates short runs.** At $T = 1$ the conversion loses 0.4656 of the 0.7561 excess. The reason is the term $\ln(1/\delta)/(\alpha - 1)$ in the conversion (about $11.5/(\alpha - 1)$ at $\delta = 10^{-5}$), which you pay however small the curve is. When the curve is tiny, as after one step, this fixed toll is most of the answer.

**Curve slack dominates long runs.** After many steps the curve $T\,\varepsilon(\alpha)$ is large, the best order moves up, and the fixed toll shrinks in comparison. From $T = 1000$ on, the conversion slack is negligible (0.0005 at $T = 1000$, zero to four decimals at $T = 10{,}000$), and almost all the slack is in the curve. That last 7.6% at $T = 10{,}000$ is what only a numerical accountant removes, because only it keeps the whole distribution.

**The CLT under-reports here.** It is below the certified lower bound at every $T$: at $T = 10{,}000$, 6.0071 against 6.1377. Its ratio climbs toward 1 as $T$ grows, as a central limit theorem should, since a sum of more terms is closer to Gaussian. But at no $T$ in the table is it a valid guarantee.

**The PLD bracket is the truth, to within $Th$.** At $T = 10{,}000$ the width is $6.2377 - 6.1377 = 0.1000$ on this grid, exactly $Th$. If that is too wide, refine $h$. Its only other error, truncating the far tails of the loss, is pushed in the safe direction on each side (mass is moved so that the upper bound only rises and the lower bound only falls), so $h$ is the one knob to turn.

| $T$ | Conversion slack | Curve slack | Total | Excess over PLD upper |
| --- | --- | --- | --- | --- |
| 1 | 0.4656 | 0.2905 | 0.7561 | 379.1% |
| 10 | 0.2402 | 0.4151 | 0.6554 | 172.5% |
| 100 | 0.0603 | 0.4353 | 0.4956 | 69.0% |
| 1,000 | 0.0005 | 0.2677 | 0.2681 | 14.6% |
| 10,000 | 0.0000 | 0.4750 | 0.4750 | 7.6% |

: Where the RDP accountant (improved conversion) loses against the PLD upper bound. Conversion slack is improved minus optimal conversion; curve slack is optimal conversion minus PLD upper bound. {#tbl-choose-slack}

![Left: $\varepsilon$ at $\delta = 10^{-5}$ against the number of steps $T$ for DP-SGD with $q = 0.01$, $\sigma = 1$, under per-step basic and advanced composition, RDP with Mironov's and the improved conversion, the GDP central-limit approximation, and the pessimistic and optimistic PLD on the grid $h = 10^{-5}$ (the two PLD curves overlap). Right: RDP (both conversions), the CLT and the PLD lower bound divided by the PLD upper bound; per-step composition (ratios 1 to 3908) is off the scale. Ratios above 1 are valid but loose; the shaded region below 1 is where only the PLD lower bound may lie, and the CLT lies there at every $T$.](figures/choosing_an_accountant_eps_vs_k.svg){#fig-choose-eps-k}

### Example: the number you would have reported

Suppose you ran 1,000 steps. Depending on the accountant, your paper says $\varepsilon = 2.54$ (RDP with Mironov's conversion, as in older code), 2.10 (RDP with the improved conversion, as in Opacus's RDP accountant), 1.83 (a PLD accountant), or 1.62 (the CLT). The first three are true statements about the run, because each is an upper bound; the third is the most informative, because it is the smallest. The fourth is false: the optimistic PLD proves $\varepsilon \ge 1.8232$, so the true $\varepsilon$ is larger than 1.62 and the run is not $(1.62, 10^{-5})$-DP.

### Check yourself

- At $T = 1$, per-step basic composition gives 0.20 and RDP gives 0.9555. Does that make the RDP accountant wrong? *(No. Both are upper bounds, and the basic one happens to be tighter for a single step, because RDP's conversion pays $\ln(1/\delta)/(\alpha - 1)$ even when the curve is tiny. RDP wins as soon as there is something to compose.)*
- Why can the optimal conversion of an RDP curve never beat the pessimistic PLD by more than $Th$? *(It must cover every pair with the same RDP curve, and the real pair is one of them, so it is at least the true $\varepsilon$. The pessimistic PLD is at most $Th$ above the true $\varepsilon$. So the optimal conversion can fall below the PLD upper bound, but by at most $Th$ (without truncation). Here the curve slack is much larger than $Th$, and the script finds optimal $\ge$ PLD upper at every $T$.)*

## The inverse problem: noise for a target ε {#sec-choose-inverse}

### Context

Practitioners rarely compute $\varepsilon$ after the fact. They fix a budget, say $\varepsilon = 3$ at $\delta = 10^{-5}$, fix the number of steps and the batch size, and ask how much noise they must add. The noise multiplier is then the knob that sets utility. The noise added to the gradient sum has standard deviation $\sigma C$, and the sum is divided by the expected batch size $qn$ for $n$ records, so the noise in the averaged gradient has standard deviation $\sigma C/(qn)$. A loose accountant overstates $\varepsilon$ at every $\sigma$, so it asks for a larger $\sigma$ to meet the same budget: its slack turns directly into extra noise.

### Statement

For each accountant $A$, write $\varepsilon_A(\sigma)$ for the $\varepsilon$ it reports at $\delta$ for the run with noise multiplier $\sigma$. The noise it requires for a target $\varepsilon^*$ is the smallest $\sigma$ at which its reported $\varepsilon$ meets the target:
$$\sigma_A(\varepsilon^*) = \min\{\sigma > 0 : \varepsilon_A(\sigma) \le \varepsilon^*\}.$$ {#eq-choose-inverse}
@tbl-choose-sigma gives it for $T = 10{,}000$, $q = 0.01$, $\delta = 10^{-5}$ and three targets.

| Target $\varepsilon$ | Advanced | RDP, Mironov | RDP, improved | GDP CLT | PLD upper | PLD lower |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 24.682 | 4.974 | 4.126 | 3.796 | 3.988 | 3.654 |
| 3 | 10.505 | 1.854 | 1.662 | 1.549 | 1.584 | 1.547 |
| 8 | 5.711 | 0.962 | 0.917 | 0.868 | 0.885 | 0.880 |

: The noise multiplier $\sigma$ each accountant requires for a target $\varepsilon$ at $\delta = 10^{-5}$, with $T = 10{,}000$ steps at $q = 0.01$. PLD on the grid $h = 10^{-5}$; the PLD lower column is the $\sigma$ at which the optimistic bound reaches the target. {#tbl-choose-sigma}

### Derivation

The plan is to show that more noise never raises any accountant's $\varepsilon$, so that @eq-choose-inverse asks for the point where a decreasing function crosses a level, and then to find that point numerically.

1. **Every bound decreases in $\sigma$.** The idea is that a noisier run can be produced from a less noisy one without touching the data, and that cannot reveal more. Fix $\sigma < \sigma'$. (a) *Post-processing:* the variances of independent Gaussians add, so adding independent $\mathcal N(0, \sigma'^2 - \sigma^2)$ noise to the output of the step with noise $\sigma$ gives exactly the step with noise $\sigma'$, in both runs and in every step. That map does not look at the data. (b) *Data processing:* the data-processing inequality ([Divergences](../part01-prerequisites/03-divergences.md)) says that passing two distributions through the same data-independent map can only make them harder to tell apart, so every divergence between them can only fall. That applies to the per-step and the composed pair, and to both the Rényi divergences and the hockey-stick divergence that defines $\delta(\varepsilon)$. Every accountant is built from those divergences, so its $\varepsilon$ falls too. For the discretized PLD this holds only up to grid effects of order $Th$, since rounding can move a value by up to that much. The CLT's $\mu = q\sqrt{T(e^{1/\sigma^2} - 1)}$ falls in $\sigma$ directly, because $e^{1/\sigma^2}$ does.
2. **So @eq-choose-inverse is a root of a decreasing function,** $\varepsilon_A(\sigma) - \varepsilon^*$, and any bracketing method finds it: start from a low $\sigma$ where the function is positive and a high one where it is negative, and shrink the interval while keeping the sign change inside. The script uses false position with the Illinois modification. False position places each new guess where the straight line through the two ends crosses zero, rather than at the midpoint as bisection does. The Illinois modification halves the function value kept at an end that has not moved for two rounds, which stops one end from getting stuck. It needs only a handful of evaluations. That matters for the PLD, where each evaluation is two FFT compositions of 10,000 steps. The script stops when the bracket is narrower than $10^{-4}$ and returns the upper end, where $\varepsilon_A \le \varepsilon^*$. The method does not need exact monotonicity for this. Up to discretization, the returned $\sigma$ may not be the minimum in @eq-choose-inverse, but it was checked to satisfy $\varepsilon_A(\sigma) \le \varepsilon^*$, which is all safety needs. For an upper-bound accountant the returned $\sigma$ is therefore safe.
3. **The bracket for the PLD comes from the other accountants.** The RDP value serves as the upper end: RDP was above the PLD upper bound at every $T$ in @tbl-choose-main, so it should need more noise, though the theory does not guarantee it (@eq-choose-order). Ninety percent of the smaller of the RDP and CLT values serves as the lower end. Because neither end is guaranteed, the solver checks that the two ends really bracket the root, with the function positive at one end and negative at the other, and stops with an error if they do not.
4. **The CLT has a closed form.** First find the $\mu^*$ whose GDP curve @eq-gdp-delta gives $\varepsilon^*$ at $\delta$. Then solve $\mu = q\sqrt{T(e^{1/\sigma^2} - 1)}$ for $\sigma$: squaring gives $q^2T(e^{1/\sigma^2} - 1) = \mu^{*2}$, so $e^{1/\sigma^2} = 1 + \mu^{*2}/(q^2T)$, and taking logs and square roots, $\sigma = 1/\sqrt{\ln(1 + \mu^{*2}/(q^2T))}$.

As a check, the RDP column reproduces $\sigma = 4.126$ for $\varepsilon = 1$ from [Exercise 4](../../book-exercises/part04-privacy-accounting/05-subsampling.md#sec-sol-sub-4) of [Amplification by subsampling](05-subsampling.md), and the PLD upper bound at $\sigma = 3.9884$ is 1.0000, as the solution of $\varepsilon_A(\sigma) = 1$ should give.

### Interpretation

@fig-choose-noise shows each accountant's $\sigma$ relative to the PLD upper bound's, the tightest certified one.

**Advanced composition needs six times the noise.** It asks for 6.19 to 6.63 times the $\sigma$ of the PLD ($24.682/3.988 \approx 6.19$ at $\varepsilon = 1$, $10.505/1.584 \approx 6.63$ at $\varepsilon = 3$), a noise level at which DP-SGD rarely trains a useful model.

**The RDP conversions cost a few percent of noise.** Mironov's conversion costs 8.6 to 24.7% more noise, most at the strictest target, where its fixed $\ln(1/\delta)/(\alpha - 1)$ toll is the largest share of a small $\varepsilon$. RDP with the improved conversion costs 3.4 to 4.9% more noise. This is the $\sigma$-side image of the 7.6% $\varepsilon$-excess in @tbl-choose-slack, and it is smaller because $\varepsilon$ falls faster than $\sigma$ rises: in this range a small relative increase in $\sigma$ buys a larger relative decrease in $\varepsilon$, so closing a gap in $\varepsilon$ costs a smaller gap in $\sigma$. Seen from the other side, at $\sigma = 0.8851$, where the PLD certifies $\varepsilon = 8$, the RDP accountant reports 8.6330.

**The CLT asks for less noise than the certified bound needs,** 2.0 to 4.8% less. At $\varepsilon = 8$ this is provably too little: at the CLT's $\sigma = 0.8676$ the *optimistic* PLD, a lower bound on the true $\varepsilon$, already gives 8.2594, so the run is not $(8, 10^{-5})$-DP. At $\varepsilon = 1$ and 3 the grid is too coarse to tell. At the CLT's $\sigma$ the PLD bracket is $[0.9550, 1.0550]$ and $[2.9938, 3.0938]$, and both contain the target, so the truth could be on either side.

**The sandwich width matters most at small targets.** The width $Th = 0.1$ is a fixed amount of $\varepsilon$, and it is 10% of $\varepsilon = 1$, so the PLD brackets $\sigma$ only to $[3.654, 3.988]$. For a tight $\sigma$ at a small target, refine $h$.

![For each target $\varepsilon$ at $\delta = 10^{-5}$ ($T = 10{,}000$, $q = 0.01$), the noise multiplier each accountant requires, divided by the one the PLD upper bound requires. RDP with Mironov's conversion asks for up to 25% more noise and RDP with the improved conversion about 4% more. The CLT and the PLD lower bound fall in the shaded region: they ask for less noise than any certified accountant can justify. Advanced composition (6.19 to 6.63) is off the chart.](figures/choosing_an_accountant_noise.svg){#fig-choose-noise}

### Example: what 4% of noise is worth

With $n = 50{,}000$ records and $q = 0.01$, the expected batch holds $0.01 \times 50{,}000 = 500$ records. At $\varepsilon = 8$, the RDP accountant sets $\sigma = 0.917$, so the averaged gradient carries noise of standard deviation $0.917C/500$. The PLD accountant sets $\sigma = 0.885$: the RDP accountant asks for 3.6% more noise for the same certified budget (the ratio 1.036 in @fig-choose-noise). The same budget, with a tighter accountant, buys a less noisy model. In the other direction, the PLD lets you keep $\sigma$ and train longer; the exercises quantify that.

### Check yourself

- Why does the script return the upper end of the final bracket rather than its midpoint? *(The upper end is a $\sigma$ at which the accountant's $\varepsilon$ was checked to be at most the target. For an upper-bound accountant that $\sigma$ certifiably meets the budget; the midpoint might not.)*
- A colleague calibrates $\sigma$ with the CLT and then reports the $\varepsilon$ of the PLD upper bound at that $\sigma$. Is their report valid? *(Yes. The reported number comes from a bound, so it is valid. It is just above the target they meant to meet: 8.3594 instead of 8 at the $\varepsilon = 8$ target.)*

## Where each accountant is tight {#sec-choose-decision}

### Context

DP-SGD is one mechanism. You will also release statistics with the Gaussian mechanism, answer queries with pure-DP mechanisms (those with $\delta = 0$), shuffle locally randomized reports, or stop training adaptively. The ranking of accountants changes with the mechanism, because what makes an accountant tight is whether the summary it keeps describes that mechanism's privacy loss exactly. This section collects, from the earlier chapters of this part, which accountant is tight for which mechanism, and puts numbers on three of the rows.

### Statement

@tbl-choose-decision is the decision table.

| Mechanism | Use | Why, and how tight | Avoid |
| --- | --- | --- | --- |
| Gaussian, no subsampling (any number of steps, any sensitivity mix) | GDP or zero-concentrated differential privacy (zCDP), converted exactly with @eq-gdp-delta | $\mu^2$ adds and the Gaussian profile is exact ([f-DP and Gaussian DP](04-gaussian-dp.md)). RDP and zCDP keep the curve exactly but the conversion loses 5 to 10% (@sec-conv-slack) | Per-step $(\varepsilon, \delta)$ composition |
| Poisson-subsampled Gaussian (DP-SGD) | PLD, pessimistic, with the optimistic value as error bar | Certified bracket of width $Th$ ([Numerical accounting](06-numerical-accounting.md)). RDP with the improved conversion is 7.6 to 69% above it on the running example for $T \ge 100$ | zCDP, which cannot express amplification (@sec-cdp-tcdp); the CLT as a guarantee |
| Pure-DP steps (randomized response, Laplace, exponential mechanism) | PLD (exact for finite outputs), or RDP with the mechanism's own curve | The privacy loss takes few values, so the PLD is exact or nearly so. zCDP with $\rho = \varepsilon^2/2$ is convenient but loose for large $\varepsilon$ (@sec-cdp-pure) | Advanced composition when a tighter accountant is available |
| Adaptive choice of noise, stopping or per-record budgets | RDP filter at a fixed order ([Adaptive composition](08-adaptive-composition.md)) | Free at a fixed order (@sec-adapt-renyi-filter); an odometer costs a union bound over orders | Plugging realized $\varepsilon_t$ into a fixed-parameter theorem |
| Local randomizers behind a shuffler | Exact computation for binary randomized response; numeric clone bound otherwise ([Amplification by shuffling and by iteration](07-shuffling-iteration.md)) | The closed-form clone bound is 4.47 to 7.55 times the exact value there | The Poisson-subsampling accountant |
| Only the last iterate released | Amplification by iteration ([Amplification by shuffling and by iteration](07-shuffling-iteration.md)) | Needs a convex, smooth loss, contractive steps and no released checkpoints; rarely applies to deep learning | Any hidden-state claim for non-convex training |

: Which accountant to use for which mechanism, with the chapter or section that establishes it. {#tbl-choose-decision}

### Derivation: three rows with numbers

The script computes three rows at $\delta = 10^{-5}$, one where the composed loss is exactly Gaussian, one where it takes only a few values, and the DP-SGD case where it is neither.

1. **Gaussian without subsampling.** A Gaussian step with sensitivity 1 and noise multiplier $\sigma$ is exactly $\mu$-GDP with $\mu = 1/\sigma$, and under composition the squares of $\mu$ add, just as variances of independent Gaussians do ([f-DP and Gaussian DP](04-gaussian-dp.md)). So one hundred steps with $\sigma = 10$ compose to $\mu = \sqrt{100 \times (1/10)^2} = \sqrt{100}/10 = 1$. The exact $\varepsilon$ from @eq-gdp-delta is 4.3772. RDP's curve for this run, $\alpha\mu^2/2$, is also exact, but converting it costs: it gives 4.7285 with the improved conversion, 8.0% above, and 5.2985 with Mironov's. Here zCDP, which describes the curve by a line $\rho\alpha$, is the same as RDP, because the curve already is a line.
2. **Pure DP: one hundred steps of randomized response with $\varepsilon_0 = 0.1$.** In randomized response each output is $e^{\varepsilon_0}$ times more likely under one dataset than the other, in one direction or the other, so its privacy loss is $\pm\varepsilon_0$. The composed loss is $\varepsilon_0$ times (number of $+$ steps minus number of $-$ steps), so the composed PLD is a binomial and the exact $\delta(\varepsilon)$ is a finite sum ([Numerical accounting](06-numerical-accounting.md)). The exact $\varepsilon$ is 4.3068. RDP with the exact randomized-response curve gives 4.6154. zCDP needs a $\rho$ with $\varepsilon(\alpha) \le \rho\alpha$; the smallest that works for randomized response is $\varepsilon_0\tanh(\varepsilon_0/2)$ per step, which is also the sharp charge every $\varepsilon_0$-DP mechanism satisfies (@eq-cdp-pure-sharp in [Concentrated DP](03-concentrated-dp.md)), and it gives 4.7263. Advanced composition with $\delta' = \delta$ gives 5.8502, and basic composition $100 \times 0.1 = 10.0000$.
3. **The subsampled Gaussian at $T = 10{,}000$.** From @tbl-choose-main, the PLD brackets $\varepsilon$ in $[6.1377, 6.2377]$, RDP (improved) gives 6.7127, 7.6% above the upper bound, and the CLT gives 6.0071, below the bracket.

### Interpretation

**The more structure the accountant keeps, the tighter it is.** The ranking in each row follows the ladder of the part's introduction. An $(\varepsilon, \delta)$ pair keeps two numbers, zCDP one line, RDP a whole curve, GDP the whole trade-off function (the best type II error of any test at each type I error) when that function is Gaussian, and the PLD the entire distribution of the privacy loss. Each rung loses information the next one keeps, and lost information has to be filled in with a worst case.

**The right tool depends on what is Gaussian.** When the composed pair is exactly Gaussian, as in the first row, GDP is exact and costs nothing. When subsampling makes it a mixture, as in DP-SGD, no closed form is exact, and only the PLD is both certified and tight.

**Adaptivity is the one place where RDP is the natural tool.** If the noise or the stopping time is chosen while training runs, based on what has been released, the per-step pair is not fixed in advance. The Rényi filter of @sec-adapt-renyi-filter handles that and loses nothing at a fixed order, whereas the PLD accountant above assumes the per-step pair is fixed in advance. The references of [Adaptive composition](08-adaptive-composition.md) include filters built on GDP and PLDs (Koskela, Tobaben & Honkela 2023) for when the RDP slack matters.

### Example: an analytics pipeline

A team releases 100 noisy counts with the Gaussian mechanism at $\sigma = 10$ and reports $\varepsilon = 5.2985$, from an old RDP accountant with Mironov's conversion. The same release is exactly $(4.3772, 10^{-5})$-DP, as GDP shows. The reported number is 21.0% above the truth ($5.2985/4.3772 \approx 1.210$): the team could have answered more queries, or added less noise, at no cost in privacy.

### Check yourself

- For the randomized-response row, why is the PLD exact rather than a bracket? *(The privacy loss takes only the values $\pm\varepsilon_0$, so the composed loss lies on the lattice $\{(2j - k)\varepsilon_0\}$, where $j$ is the number of $+$ steps, and no rounding is needed; $\delta(\varepsilon)$ is a finite binomial sum.)*
- Why is zCDP the wrong accountant for DP-SGD even though it is exact for the Gaussian? *(The ratio $\varepsilon(\alpha)/\alpha$ of the subsampled curve tends to $1/(2\sigma^2)$ as $\alpha$ grows, so the smallest $\rho$ with $\varepsilon(\alpha) \le \rho\alpha$ for all $\alpha$ is the unsubsampled $1/(2\sigma^2)$, even though the curve is far below that line at small orders. zCDP cannot express the amplification. See @sec-cdp-tcdp.)*

## Matching the accountant to the code {#sec-choose-match}

### Context

Every number in @tbl-choose-main assumes a model of the training run. The accountant cannot check that model against your code: it takes $q$, $\sigma$ and $T$ and trusts that the code does what the theorem assumes. Ponomareva et al. (2023, Section 5.3.1) put the point in one heading: "privacy accounting assumptions should match training reality". They note that many papers have reported the Poisson-sampling number while training with shuffled fixed-size batches, going back to Abadi et al. (2016). A mismatch does not make the number a little loose. It makes it a statement about a different algorithm.

### Statement

Before reporting $\varepsilon$, check each item of @tbl-choose-checklist against the code.

| Item | What the accountant assumes | What to check in the code | If it does not match |
| --- | --- | --- | --- |
| Sampling scheme | Poisson sampling at rate $q$ | Is each record included independently with probability $q$, giving variable batch sizes? | Fixed-size random batches: account the Poisson pair with $\sigma/2$ (@tbl-sub-matching). Shuffle and partition: no amplification theorem applies; report the guarantee without amplification, for zero-out neighbours |
| Neighbouring relation | Add/remove one record | Is the guarantee stated for add/remove, replace-one or zero-out? | Replace-one under Poisson sampling costs more per step: the group-privacy bound is about $e^\varepsilon + 1$ times the add/remove value, and the exact pure-DP cost is about twice it for small $\varepsilon$ (@sec-sub-schemes, @eq-sub-replace-pure) |
| Privacy unit | One example | Can one person contribute several examples? | Account for users: clip and sample per user, or pay group privacy |
| Released outputs | Every noisy gradient | Are checkpoints, logs or gradients exposed? | Nothing to fix: the guarantee covers them all. Claiming less leakage needs @sec-shuf-hidden and its conditions |
| Hyperparameter tuning | Not counted | Were learning rate, clipping norm or $\sigma$ chosen by looking at results on the private data? | Account for the tuning runs too, or tune on public data |
| Adaptive parameters | $\sigma$, $q$ and $T$ fixed in advance | Early stopping, noise schedules, per-record budgets? | Use a filter (@sec-adapt-renyi-filter) |
| Implementation | Exact real-valued Gaussian noise, exact clipping | Floating-point noise sampling, gradient accumulation across devices, clipping applied before or after accumulation | Out of scope of the accountant: a question for the implementation and its verification, not for this part |

: A checklist for matching the accountant to the training code. {#tbl-choose-checklist}

### Derivation: the price of three mismatches

1. **Fixed-size batches accounted as Poisson.** Suppose each step draws a fresh random batch of exactly $qn$ records. With a fixed batch size, adding a record has to push another one out, so in effect one record is swapped for another, and the batch sum can move by up to $2C$ instead of $C$. Lebeda et al. (2024, Proposition 9; @tbl-sub-matching) show that such a run is covered by the Poisson pair with noise multiplier $\sigma/2$: the same noise, measured against a shift twice as large. At $T = 10{,}000$ the RDP accountant then gives 47.42 instead of 6.71, as in [Amplification by subsampling](05-subsampling.md), and the PLD upper bound gives 43.42 instead of 6.24.
2. **Shuffle and partition.** Here each epoch shuffles the data once and cuts it into consecutive batches, so the record is used exactly once per epoch, and there is no amplification theorem that holds for every dataset (@sec-sub-schemes). What remains certain is the composition of the record's own steps, one per epoch, for zero-out neighbours, the relation of @tbl-sub-matching for this sampler: the two datasets differ in that the record's gradient is replaced by zero, so the batch that holds it moves by at most $C$ in each epoch. With noise $\sigma C$, that is a Gaussian mechanism with $\mu = 1/\sigma$. Then 100 epochs at $\sigma = 1$ are 100 Gaussian mechanisms with $\mu = 1$, whose squares add to $\mu = \sqrt{100} = 10$ (@sec-gdp-composition), so @eq-gdp-delta gives $\varepsilon = 91.8173$. This needs the zero-out relation. Under add/remove there is no clean bound at all, because inserting a record shifts the fixed partition into batches and changes every batch after it. Under replace-one, the batch sum moves by up to $2C$, so $\mu = 2$ per epoch, $\mu = \sqrt{100} \times 2 = 20$ in total, and $\varepsilon = 284.39$. Ponomareva et al. recommend reporting such a guarantee without amplification whenever the sampling assumption does not hold.
3. **Tuning on the private data.** If you train 10 configurations on the private data and release the best, which one is best depends on all 10 runs, so the release is a post-processing of all 10 of them, and its privacy cost is that of the 10 runs composed, ten times the run. With RDP, ten times the curve converts to $\varepsilon = 27.23$; for 100 configurations, 145.17. Papernot & Steinke (2022) give much better bounds for tuning procedures that try a random number of configurations. NIST SP 800-226 (Section 4.3) recommends tuning on non-sensitive data, or with a differentially private algorithm.

### Interpretation

**The sampling scheme is the largest single risk.** A factor of 6.96 in $\varepsilon$ ($43.42/6.24 \approx 6.96$) from a one-line change in the data loader is more than the choice between the RDP and PLD accountants (at most 1.19 here, from @tbl-choose-ratio). Opacus, for one, uses Poisson sampling by default when you make a model private.

**The privacy unit decides what is protected.** The guarantee is about datasets that differ in one unit, so it protects one unit. NIST SP 800-226 (Section 2.4) calls a guarantee whose unit is one event, when a person contributes many, a privacy hazard. Example-level DP-SGD on a dataset with many examples per user protects each example, not each user.

**The accountant does not see the implementation.** The theorems assume exact real-valued Gaussian noise. Floating-point sampling of noise can break the guarantee of a mechanism that is correct on paper (NIST SP 800-226, Section 4.3). That is a question for formal methods and careful implementation, not for an accountant.

### Example: a code review

A training script uses a `DataLoader` with `shuffle=True` and `batch_size=500` on 50,000 records for 100 epochs, and calls an RDP accountant with `sample_rate=0.01`. The checklist flags the first row. The sampler is shuffle and partition, so the Poisson accountant's 6.71 does not apply. The certified guarantee is the unamplified 91.8173, for zero-out neighbours. Switching the loader to Poisson sampling, a change of a few lines, makes the accountant's assumptions true and the guarantee 6.24 with a PLD accountant.

### Check yourself

- Your code draws fixed-size random batches, and you want the Poisson accountant's 6.24 to be honest. What noise multiplier must you train with? *(Twice the one in the accountant: $\sigma = 2$, since fixed-size batches are dominated by the Poisson pair with $\sigma/2$, and $2/2 = 1$ is the $\sigma$ the accountant used.)*
- Does releasing every checkpoint of a DP-SGD run cost extra privacy? *(No. The accountant already assumes every noisy gradient is released, and checkpoints are a post-processing of them.)*

## What to report {#sec-choose-report}

### Context

An $\varepsilon$ without its assumptions cannot be checked, compared or even read: the same number means different things for one example or one user, with or without tuning, at $\delta = 10^{-5}$ or $10^{-10}$. Two guidance documents say what a statement should contain. NIST SP 800-226, "Guidelines for Evaluating Differential Privacy Guarantees" (Near, Darais, Lefkovitz & Howarth, March 2025), is written for anyone evaluating a DP claim. Ponomareva et al., "How to DP-fy ML" (JAIR 2023), Section 5.3.3, is written for machine learning.

### Statement

From Ponomareva et al. (2023, Section 5.3.3), a report should give:

1. **The DP setting:** central or local, and who is trusted.
2. **The instantiated definition:** which data accesses are covered (one training run, or tuning too), what the mechanism's output is, the unit of privacy, and the neighbouring relation.
3. **Accounting details:** the type of accounting (RDP, PLD, ...), its assumptions and whether they hold, and the formal statement, such as the $(\varepsilon, \delta)$ or $\rho$ values, for the final model and for the tuning.
4. **Transparency:** code, when possible, using standard libraries for the mechanism and the accountant.

On $\delta$, both documents want it small compared with $1/n$ for $n$ units of privacy. The reason is that $\delta$ is the probability with which the promise may fail. A mechanism that publishes one randomly chosen record in full changes its output distribution by only about $1/n$ when one record changes, so it meets a $\delta$ of about $1/n$ with $\varepsilon = 0$, and yet it exposes someone completely. A $\delta$ near $1/n$ therefore allows guarantees that protect nobody. Ponomareva et al. (Section 5.3.2) give the convention $\delta \ll 1/n$, for example $\delta = n^{-1.1}$, with $n$ counted in units of privacy. NIST SP 800-226 (Section 2.3) gives $\delta \le 1/n^2$ as a common recommendation, calls values above $10^{-5}$ suspicious, and asks you to report the original parameters of an RDP, zCDP or GDP guarantee alongside any $(\varepsilon, \delta)$ conversion, so that others can interpret it more precisely.

For DP-SGD, add what the accountant needs to recompute $\varepsilon$: the sampling scheme, $q$ (or batch size and $n$), $T$ (or epochs), $\sigma$, and the accountant's version and discretization. Report the clipping norm $C$ too, although $\varepsilon$ does not depend on it.

### Derivation: the numbers in the statement

Take the running example with $n = 50{,}000$ records. Then the expected batch is $qn = 500$, $T = 10{,}000$ steps are $Tq = 100$ epochs (each step touches a fraction $q$ of the data), and the conventions give $1/n = 2.0\times10^{-5}$, $n^{-1.1} = 6.8\times10^{-6}$ and $1/n^2 = 4.0\times10^{-10}$. The pessimistic PLD composed for $T = 10{,}000$ is computed once. Solving it at several $\delta$ is cheap, because it only means evaluating $\delta(\varepsilon)$ on the same composed distribution, and @tbl-choose-profile gives the result.

| $\delta$ | PLD upper | PLD lower | RDP, improved |
| --- | --- | --- | --- |
| $2.0\times10^{-5}$ ($1/n$) | 6.0068 | 5.9068 | 6.4884 |
| $1.0\times10^{-5}$ | 6.2377 | 6.1377 | 6.7127 |
| $6.8\times10^{-6}$ ($n^{-1.1}$) | 6.3641 | 6.2641 | 6.8353 |
| $1.0\times10^{-6}$ | 6.9574 | 6.8574 | 7.4140 |
| $4.0\times10^{-10}$ ($1/n^2$) | 9.0502 | 8.9378 | 9.4540 |

: The running example's privacy profile: $\varepsilon$ at several $\delta$ for $T = 10{,}000$, $q = 0.01$, $\sigma = 1$. {#tbl-choose-profile}

$\delta = 10^{-5}$ is half of $1/n = 2.0\times10^{-5}$ for this dataset, so it does not meet Ponomareva et al.'s "much smaller than $1/n$". At $1/n^2$, as NIST suggests, the certified $\varepsilon$ rises to 9.0502.

The PLD bracket has width 0.1000, the $Th = 10{,}000 \times 10^{-5}$ of [Numerical accounting](06-numerical-accounting.md), at every $\delta$ down to $10^{-6}$, and $9.0502 - 8.9378 = 0.1124$ at $1/n^2$, slightly more. Without truncation the width is at most $Th$ (@sec-num-discretize), so the excess comes from the tails the accountant truncates. Each step's PLD moves up to $10^{-15}$ of tail mass on each side, in the safe direction. Over $10^4$ steps that is of order $10^4 \times 10^{-15} = 10^{-11}$, a few percent of $\delta = 4\times10^{-10}$, so it starts to matter only at such a small $\delta$.

The RDP accountant's original parameter, the thing NIST asks you to report, is its curve. At the best order $\alpha = 4.1$ the composed curve is $\varepsilon(4.1) = 3.7336$, which the improved conversion turns into 6.7127.

### Interpretation

**Report the profile, not a point.** A single $(\varepsilon, \delta)$ pair cannot be converted to a smaller $\delta$ ([Exercise 3](../../book-exercises/part04-privacy-accounting/09-choosing-an-accountant.md#sec-sol-choose-3)), because many different mechanisms share that one pair and they behave differently at smaller $\delta$. So a reader who wants $\delta = 1/n^2$ needs either the PLD or the RDP curve, or your table. @tbl-choose-profile costs one line.

**Report the accountant's error bar.** For a PLD accountant, the optimistic value says how much the number could still fall. For RDP, nothing does: it has no matching lower bound. That is one more reason to prefer the PLD.

**State the fallback.** If any item of @tbl-choose-checklist is in doubt, give the guarantee that does not rely on it, as Ponomareva et al. recommend for amplification. A reader can then see what holds even in the worst case.

### Example: a filled-in privacy statement

> **Privacy guarantee.** The model was trained with DP-SGD in the central model: the training server is trusted to run the code below, and the guarantee covers every intermediate noisy gradient, hence every checkpoint and the final model. **Unit of privacy:** one training example; each person contributes at most one example. **Neighbouring relation:** add or remove one example. **Training:** Poisson sampling with rate $q = 0.01$ ($n = 50{,}000$ examples, expected batch size 500), $T = 10{,}000$ steps (100 epochs), per-example clipping norm $C = 1$, Gaussian noise multiplier $\sigma = 1$. **Accountant:** privacy loss distributions, pessimistic discretization on the grid $h = 10^{-5}$, both add and remove directions (this book's `numerical_accounting.py`). **Guarantee:** $(6.24, 10^{-5})$-DP and $(9.06, 4.0\times10^{-10})$-DP. The optimistic discretization gives 6.1377 and 8.9378, so the accountant's own error is at most 0.1000 and 0.1124. For comparison, the RDP accountant with the improved conversion gives 6.7127 at $\delta = 10^{-5}$. **Not covered:** hyperparameters were tuned on a public proxy dataset; no tuning on the private data was performed. **If the sampling assumption failed** (for example, shuffled fixed-size batches), the guarantee without amplification is $(91.82, 10^{-5})$-DP for zero-out neighbours (one example's gradient replaced by zero).

Each number in the statement is in @tbl-choose-profile or @sec-choose-match. Upper bounds are rounded up, never to the nearest value: 9.0502 becomes 9.06, because 9.05 would be a claim the accountant has not certified.

### Check yourself

- Why does the statement give the clipping norm even though $\varepsilon$ does not depend on it? *(The noise added is $\sigma C$, so $C$ with $\sigma$ tells a reader the actual noise level and lets them check the implementation. Ponomareva et al. (Section 4.2) note that the common convention is to report $\sigma$ and $C$ separately, so that the noise reads as a fraction of the clipping norm and transfers between models.)*
- A paper reports "$\varepsilon = 3$" and nothing else. Name two items without which you cannot interpret it. *(Any two of: $\delta$, the unit of privacy, the neighbouring relation, the sampling scheme, the accountant.)*

## Checking your accountant {#sec-choose-check}

### Context

An accountant is code, and code has bugs. Before you trust a number, whether from your own accountant or a library's, run checks whose answers you know. A good check is a case where the truth is known exactly, or a property every correct accountant must have.

### Statement

Four checks, each run by the script:

1. **Set $q = 1$ and compare with the closed form.** A bound must lie above the exact $\varepsilon$ of $T$ full-batch Gaussian steps.
2. **Run the PLD machinery on a pair with a closed form.** The PLD bracket for the plain Gaussian must contain the exact $\varepsilon$.
3. **Check both chains of @eq-choose-order at every $T$,** and that no upper bound falls below the optimistic PLD.
4. **Round-trip $\delta$.** The composed pessimistic PLD, evaluated at its own $\varepsilon$, must give back the $\delta$ it was solved at, and its $\varepsilon$ must increase with $T$.

### Derivation: why each answer is known

1. **$q = 1$.** Without subsampling every record is in every batch, so each step is the plain Gaussian mechanism. That mechanism is exactly $\mu$-GDP with $\mu = 1/\sigma$, and since squares of $\mu$ add under composition, $T$ of them are exactly $\sqrt{T \times (1/\sigma)^2} = \sqrt T/\sigma$-GDP (@sec-gdp-composition). @eq-gdp-delta then gives the exact $\varepsilon$, and every upper bound must be at least this.
2. **The plain Gaussian PLD.** The same closed form is the truth for the composed Gaussian pair. The pessimistic and optimistic discretizations bracket the truth, and without truncation they are within $Th$ of each other (@eq-num-sandwich), so the bracket must contain the closed form.
3. **The chains.** They were proved above for every correct implementation, so a violation is a bug, not slack.
4. **The round trip.** Solving $\delta(\varepsilon) = \delta$ for $\varepsilon$ and evaluating $\delta$ at the solution are inverse operations on the same decreasing profile, so they must give back $\delta$. For the growth in $T$: whatever the first $T$ steps release is part of what the first $T + 1$ steps release, so the $T$-step transcript is obtained from the $(T + 1)$-step one by dropping the last output. Dropping an output is a data-independent map, so by the data-processing inequality the $T$-step pair is no easier to tell apart, and no correct accountant can decrease in $T$.

### Example: the checks on this book's accountants

1. For $\sigma = 1$, $T = 10$, $q = 1$: the exact $\mu$ is $\sqrt{10}/1 = 3.1623$, and the exact $\varepsilon$ is 17.8566. The RDP accountant gives 19.0536, above it as a bound must be. The CLT formula, with $q = 1$, gives $\mu = \sqrt{10(e - 1)} = 4.1452$ instead of $3.1623$ and $\varepsilon = 25.5853$: it fails the check, because its derivation assumes small $q$ (@sec-gdp-sgd).
2. On the plain Gaussian with $\mu = 1$ and $T = 10$ at $h = 10^{-3}$, the PLD bracket $[17.8516, 17.8616]$ contains the closed form 17.8566. Its width is $10 \times 10^{-3} = 0.01$, which is $Th$.
3. Both chains hold on the running example at all nine values of $T$, and the CLT lies below the PLD lower bound at every one.
4. Evaluating the composed pessimistic PLD at its own $\varepsilon$ gives back $\delta = 1.000\times10^{-5}$, and the PLD upper bound increases with $T$.

### Interpretation: comparing with independent libraries

This book's accountants use no library, so that every line can be read. Production code should use a maintained one and compare against a second. Three are widely used: Google's `dp_accounting` (RDP and PLD accountants), Opacus (RDP, privacy loss random variable (PRV) and GDP accountants; its default is the PRV accountant), and Microsoft's `prv_accountant` (Gopi, Lee & Wutschitz 2021), which returns a lower bound, an estimate and an upper bound for $\varepsilon$. What agreement should you expect?

**RDP accountants agree when the grid and the conversion agree.** Two RDP accountants differ only in the orders they try and the formula they convert with. Opacus's default orders are this part's grid without 128, 256 and 512, and its conversion is the improved one (Balle et al. 2020, Theorem 21). On the running example the best orders lie between 4.1 and 9.9, inside both grids, so the results should agree to the precision of the fractional-order computation. If they differ by about $\Delta(\alpha)$ of @eq-conv-gain, the fixed amount by which the improved conversion beats Mironov's at order $\alpha$, one of them uses Mironov's conversion. At order 10 that gap is $\frac{\ln 10}{9} + \ln\frac{10}{9} = 0.3612$ (@tbl-conv-gain).

**Fractional orders are a known weak spot.** At an integer order the subsampled Gaussian's RDP is a finite sum, but at a fractional order it needs an infinite series with terms of both signs. `dp_accounting` 0.6.0 computes it with a series whose negative terms it counts as positive, so it returns a loose upper bound there, not the exact value. Opacus keeps the signs and is exact. Checking the fixed-size-batch value 47.42 of @sec-choose-match with `dp_accounting` 0.6.0 gives 49.43 (computed with that library, not by this book's scripts). Compare at integer orders, or against Opacus.

**PLD and PRV accountants agree within their error bars.** Different discretizations give different numbers, but every certified upper bound must lie above the truth and every certified lower bound below it, so every upper bound must lie above every lower bound. If two libraries' brackets do not overlap, one of them has a bug.

**No accountant may fall below a certified lower bound.** The optimistic PLD is the referee: any reported upper bound below it, like the CLT's, is wrong.

A cross-check against the libraries, run outside this book's scripts, gives a clean result. The RDP and PLD numbers of this chapter agree with Opacus 1.6.0, `dp_accounting` 0.6.0 and `prv_accountant` 0.2.0 to the printed precision, and the libraries' tight PLD estimate at $T = 10{,}000$ is about 6.19, inside the bracket $[6.1377, 6.2377]$. These library values (and the 49.43 above) are quoted from the libraries; the book's scripts do not import them and cannot print them.

### Check yourself

- Your RDP accountant reports 7.43 for the running example at $T = 10{,}000$ and a colleague's reports 6.71. Which conversion does each use? *(7.43 is Mironov's, 6.71 the improved one; see @tbl-choose-main.)*
- Why is $q = 1$ a good test case for a DP-SGD accountant? *(The composed pair is exactly Gaussian, so the true $\varepsilon$ has a closed form. A bound must lie above it, and an approximation that misses it is caught at once.)*

## Code

The script [code/choosing_an_accountant.py](../../code/part04-privacy-accounting/choosing_an_accountant.py) imports every accountant from the chapter that derived it and implements nothing new except the comparison, the inverse problem and the checks. Its public functions are `accountant_table(q, sigma, delta, ks)`, which returns one `AccountantRow` per $T$ with every accountant's $\varepsilon$, and `sigma_for_target(eps_target, q, steps, delta, accountant)`, which solves @eq-choose-inverse for any accountant named in `ACCOUNTANTS`. From the repository root:

```bash
uv run python code/part04-privacy-accounting/choosing_an_accountant.py
```

It runs in about two and a half minutes, almost all of it in PLD compositions. **Output** (seed 0; nothing is random), summarized in @tbl-choose-code:

| Experiment | Result |
| --- | --- |
| Every accountant, $T = 1$ to 10,000 | @tbl-choose-main, @tbl-choose-ratio, @tbl-choose-slack; both chains of @eq-choose-order hold at every $T$, and RDP optimal $\ge$ PLD upper is observed at every $T$; CLT below the PLD lower bound at every $T$ |
| Noise for a target $\varepsilon$ | @tbl-choose-sigma; PLD upper bound at $\sigma = 3.9884$ is 1.0000 |
| Other mechanisms | Gaussian $\mu = 1$: 4.3772 exact, 4.7285 RDP; randomized response: 4.3068 exact, 4.6154 RDP, 4.7263 zCDP, 5.8502 advanced |
| Mismatches | Fixed-size batches: 47.42 (RDP), 43.42 (PLD); no amplification: 91.8173 (zero-out), 284.39 (replace-one); 10 tuning runs: 27.23 |
| Privacy profile | @tbl-choose-profile |
| Sanity checks | $q = 1$: 17.8566 exact, 19.0536 RDP, 25.5853 CLT; Gaussian PLD $[17.8516, 17.8616]$ |

: The script's output with the default seed. {#tbl-choose-code}

[tests/test_choosing_an_accountant.py](../../tests/test_choosing_an_accountant.py) checks that the output still matches, and calls the two public functions directly.

**What to take from it.** First, the accountants of the part agree where they must. Every number shared with [Amplification by subsampling](05-subsampling.md) and [Numerical accounting](06-numerical-accounting.md) is reproduced to the printed digits, and the bounds satisfy both proved chains of @eq-choose-order. Second, the gaps between them are the cost of the choice: per-step composition is off by orders of magnitude, RDP by 7.6% at 10,000 steps and much more for short runs, and the CLT is invalid on this example. Third, mismatched assumptions cost more than the choice between RDP and PLD: a factor of 6.96 over the Poisson PLD bound for fixed-size batches, and 14.72 without amplification ($91.8173/6.2377$).

## References

- Ponomareva, Hazimeh, Kurakin, Xu, Denison, McMahan, Vassilvitskii, Chien & Thakurta, "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (Journal of Artificial Intelligence Research 77, 2023; arXiv:2303.00654). Section 5.3.1 on matching accounting to data processing (Table 3), Section 5.3.2 on setting $\delta$, Section 5.3.3 on what to report, Section 5.2.1 on target $\varepsilon$ tiers. The practical companion to this chapter.
- Near, Darais, Lefkovitz & Howarth, "Guidelines for Evaluating Differential Privacy Guarantees" (NIST Special Publication 800-226, March 2025). The unit of privacy (Section 2.4), the choice of $\delta$ and reporting original parameters of RDP, zCDP and GDP guarantees (Section 2.3), floating-point and hyperparameter-tuning hazards (Section 4.3). Read it for what an evaluator will ask.
- Gopi, Lee & Wutschitz, "Numerical Composition of Differential Privacy" (NeurIPS 2021). The PRV accountant with certified error bounds, and examples where the CLT under-reports; the basis of `prv_accountant` and Opacus's default accountant.
- Doroshenko, Ghazi, Kamath, Kumar & Manurangsi, "Connect the Dots: Tighter Discrete Approximations of Privacy Loss Distributions" (PoPETs 2022). The pessimistic PLD discretization used in Google's `dp_accounting`.
- Mironov, Talwar & Zhang, "Rényi Differential Privacy of the Sampled Gaussian Mechanism" (arXiv 2019). The RDP of the subsampled Gaussian, computed exactly; the core of every RDP accountant for DP-SGD.
- Balle, Barthe, Gaboardi, Hsu & Sato, "Hypothesis Testing Interpretations and Rényi Differential Privacy" (AISTATS 2020). Theorem 21 is the improved conversion that Opacus and `dp_accounting` use.
- Dong, Roth & Su, "Gaussian Differential Privacy" (Journal of the Royal Statistical Society Series B, 2022), and Bu, Dong, Long & Su, "Deep Learning with Gaussian Differential Privacy" (Harvard Data Science Review, 2020). The GDP CLT for DP-SGD; read with this chapter's warning that it is not a bound.
- Lebeda, Regehr, Kamath & Steinke, "Avoiding Pitfalls for Privacy Accounting of Subsampled Mechanisms under Composition" (arXiv:2405.20769, 2024; IEEE SaTML 2025). Fixed-size batches versus Poisson sampling, and the factor that the wrong accountant costs.
- Chua, Ghazi, Kamath, Kumar, Manurangsi, Sinha & Zhang, "How Private are DP-SGD Implementations?" (ICML 2024). Shuffled batches can be much less private than the Poisson accountant reports.
- Papernot & Steinke, "Hyperparameter Tuning with Renyi Differential Privacy" (ICLR 2022). Privacy cost of tuning with a random number of trials, far below naive composition.
- Google, `dp_accounting` library; Meta, Opacus; Microsoft, `prv_accountant`. Documentation of the production RDP, PLD and PRV accountants to check this book's numbers against.

## Cheat sheet

@tbl-choose-cheat collects the chapter's results for DP-SGD with $q = 0.01$, $\sigma = 1$, $\delta = 10^{-5}$.

| Accountant | Kind | $\varepsilon$ at $T = 10{,}000$ | $\sigma$ for $\varepsilon = 3$ | When to use it |
| --- | --- | --- | --- | --- |
| Per step + advanced composition | Upper bound | 24379.55 | 10.505 | Never for DP-SGD |
| RDP, Mironov conversion | Upper bound | 7.4287 | 1.854 | Never: the improved conversion is free |
| RDP, improved conversion | Upper bound | 6.7127 | 1.662 | Quick bounds, adaptive filters, many heterogeneous mechanisms |
| PLD, pessimistic | Upper bound, within $Th$ of the truth | 6.2377 | 1.584 | The number to report |
| PLD, optimistic | Lower bound | 6.1377 | 1.547 | The error bar, and the referee for other accountants |
| GDP CLT | Approximation | 6.0071 | 1.549 | Intuition only; under-reports here |

: The accountants on the running example, what kind of statement each makes, and when to use it. {#tbl-choose-cheat}

**Rules of thumb**

- Report the pessimistic PLD, with the optimistic value beside it. Use RDP with the improved conversion when you need adaptivity or a quick bound, and expect it to overstate $\varepsilon$ by 5 to 10% for long runs and by far more for short ones.
- Never report a CLT number as a guarantee. Check any reported number against the PLD lower bound.
- Calibrate $\sigma$ with the accountant you will report: a looser accountant costs noise, an approximation can cost the guarantee.
- Check the sampling scheme first. Fixed-size or shuffled batches accounted as Poisson is the most expensive error in practice.
- Report the privacy unit, the neighbouring relation, $q$, $T$, $\sigma$, $C$, the accountant and its discretization, $\varepsilon$ at several $\delta$ including one well below $1/n$, and what was not accounted for.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part04-privacy-accounting/09-choosing-an-accountant.md).
