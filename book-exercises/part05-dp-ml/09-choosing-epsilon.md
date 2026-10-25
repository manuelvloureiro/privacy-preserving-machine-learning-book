# Exercises: Choosing ε

Exercises for [Choosing ε](../../book-full/part05-dp-ml/09-choosing-epsilon.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part05-dp-ml/solutions_choosing_epsilon.py` computes every number quoted here; run it with `uv run python code/part05-dp-ml/solutions_choosing_epsilon.py`.

The solutions use the chapter's notation. A mechanism satisfies differential privacy (DP) with parameters $(\varepsilon, \delta)$, or is $(\varepsilon, \delta)$-DP, when one record changes the probability of any output by at most a factor $e^\varepsilon$ plus $\delta$. A membership test has a false positive rate (FPR) and a true positive rate (TPR). The running example is stochastic gradient descent (SGD) with clipping and Gaussian noise, DP-SGD, with a sampling rate $q$ of 0.01 and $T = 10{,}000$ steps on 10,000 records, at $\delta = 10^{-5}$; its noise multipliers come from the Rényi differential privacy (RDP) accountant.

## What an audit can prove {#sec-sol-epsc-1}

> **Exercise 1.** **What an audit can prove.** An audit measures a membership test's FPR and TPR on a trained model. Using @eq-epsc-pair, find the smallest $\varepsilon$ consistent with a measurement, as a function of TPR, FPR and $\delta = 10^{-5}$. Evaluate it for TPR 0.05 at FPR $10^{-3}$, and then for the best tests against the running example's $\varepsilon = 8$ run read off @tbl-epsc-curve and @tbl-epsc-curve-more (upper ends of the brackets): TPR 0.0589 at FPR $10^{-3}$, 0.2092 at $10^{-2}$ and 0.5901 at $10^{-1}$. What does this say about audits? *Check: 3.912, 4.076, 3.041 and 1.775.* *(Inverting the testing bound)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Invert the line.** Every test satisfies $\mathrm{TPR} \le e^\varepsilon\mathrm{FPR} + \delta$, so a measured pair is possible only if $e^\varepsilon \ge (\mathrm{TPR} - \delta)/\mathrm{FPR}$, that is $\varepsilon \ge \ln\frac{\mathrm{TPR} - \delta}{\mathrm{FPR}}$.
2. **The first measurement.** $\ln\frac{0.05 - 10^{-5}}{10^{-3}} = \ln 49.99 = 3.912$. A model claimed to be $(3, 10^{-5})$-DP would be refuted by it; one claimed $(4, 10^{-5})$-DP would not.
3. **The best possible tests against the $\varepsilon = 8$ run.** $\ln(58.89) = 4.076$ at FPR $10^{-3}$, $\ln\frac{0.2092 - 10^{-5}}{0.01} = 3.041$ at $10^{-2}$ and $\ln\frac{0.5901 - 10^{-5}}{0.1} = 1.775$ at $10^{-1}$.
4. **Reading it.** These are the largest lower bounds that any audit of this run can reach at these FPRs, even with the optimal attack and exact error rates, and they are far below the run's true $\varepsilon$ of about 7.3. The reason is the chapter's: the line $e^\varepsilon\mathrm{FPR} + \delta$ touches the curve of DP-SGD only at a tiny FPR, so an audit can certify the accountant's $\varepsilon$ only by measuring at FPRs of that order.

*Tight audits need very low false positive rates, hence many trials and very exposed canaries; [Auditing private training](../../book-full/part05-dp-ml/10-auditing.md) takes this up. An audit that compares its measurement with the whole curve, rather than with one line, can be much more informative.*
:::

## The advantage of Gaussian DP {#sec-sol-epsc-2}

> **Exercise 2.** **The advantage of Gaussian DP.** Show that the largest advantage $\mathrm{TPR} - \mathrm{FPR}$ against $\mathcal N(0, 1)$ versus $\mathcal N(\mu, 1)$ is $2\Phi(\mu/2) - 1$. Evaluate it for the Gaussian differential privacy (GDP) approximations of the $\varepsilon = 4$ and $\varepsilon = 8$ runs, $\mu = 0.847$ and 1.511 (@tbl-epsc-accountants), and find the $\varepsilon$ of randomized response, whose advantage is $(e^\varepsilon - 1)/(e^\varepsilon + 1)$, with the same advantage. *Check: 0.3281 and 0.5501; randomized response with $\varepsilon = 0.681$ and 1.237.* *(Total variation of two Gaussians)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The best test.** By the Neyman–Pearson lemma the best tests threshold the output at some $\tau$: $\mathrm{FPR} = 1 - \Phi(\tau)$, $\mathrm{TPR} = 1 - \Phi(\tau - \mu)$, so the advantage is $\Phi(\tau) - \Phi(\tau - \mu)$.
2. **Maximize over $\tau$.** The derivative is $\varphi(\tau) - \varphi(\tau - \mu)$, with $\varphi$ the standard normal density, which is zero where the two densities cross, at $\tau = \mu/2$ by symmetry. There the advantage is $\Phi(\mu/2) - \Phi(-\mu/2) = 2\Phi(\mu/2) - 1$.
3. **Numbers.** $2\Phi(0.4235) - 1 = 0.3281$ and $2\Phi(0.7555) - 1 = 0.5501$. The chapter's 0.3280 for the first comes from the unrounded $\mu$.
4. **Randomized response.** Solving $(e^\varepsilon - 1)/(e^\varepsilon + 1) = A$ for $\varepsilon$ gives $e^\varepsilon = (1 + A)/(1 - A)$: $\ln(1.3281/0.6719) = 0.681$ and $\ln(1.5501/0.4499) = 1.237$.

*Measured by advantage alone, the $\varepsilon = 8$ run of DP-SGD is about as private as randomized response with $\varepsilon = 1.24$. The comparison is fair only at that one point of the curve: at low FPR the Gaussian curve rises above the randomized-response line, which is why a single summary number never suffices.*
:::

## When can a flag be trusted? {#sec-sol-epsc-3}

> **Exercise 3.** **When can a flag be trusted?** An attacker with prior $p = 0.01$ wants a test whose flags are right at least half the time. Using @eq-epsc-precision, find the largest FPR at which the pair $(\varepsilon, 10^{-5})$ allows a precision of 1/2, for $\varepsilon = 1$, 4 and 8. *Check: $1.04\times10^{-7}$, $2.25\times10^{-7}$, and every FPR at $\varepsilon = 8$.* *(Base rates)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The condition.** Precision at least 1/2 means $p(e^\varepsilon a + \delta) \ge (1 - p)a$, with $a$ the FPR: the flagged members at least as many as the flagged non-members.
2. **Solve for $a$.** $a\big((1 - p) - pe^\varepsilon\big) \le p\delta$. If $(1 - p) > pe^\varepsilon$, this is $a \le p\delta/\big((1 - p) - pe^\varepsilon\big)$. If not, every $a$ works.
3. **Numbers.** At $\varepsilon = 1$: $0.01 \times 10^{-5}/(0.99 - 0.0272) = 1.04\times10^{-7}$. At $\varepsilon = 4$: $10^{-7}/(0.99 - 0.546) = 2.25\times10^{-7}$. At $\varepsilon = 8$: $0.01e^8 = 29.8 > 0.99$, so the pair allows even odds at every FPR.
4. **Reading it.** For $\varepsilon \le 4$, the only way the pair allows a trustworthy flag at this prior is through $\delta$, at FPRs around $10^{-7}$, where at most a fraction $\delta$ of members can be flagged.

*At small priors DP limits confident identification much more than the advantage suggests; but the pair at $\varepsilon = 8$ limits nothing, which is why the chapter reads the actual curve instead.*
:::

## A family in the data {#sec-sol-epsc-4}

> **Exercise 4.** **A family in the data.** Three members of one family are in the training set, one record each, and the privacy unit is a record. Group privacy says that a mechanism that is $(\varepsilon, \delta)$-DP for one record is $(k\varepsilon, ke^{(k - 1)\varepsilon}\delta)$-DP for $k$ records. Compute the family's guarantee for $(1, 10^{-5})$ and for $(4, 10^{-5})$, and the largest posterior that the family took part, from a prior of 0.01, when the loss is at most $3\varepsilon$ for $\varepsilon = 1$. *Check: $(3, 2.22\times10^{-4})$, posterior 0.1687; $(12, 0.0894)$.* *(Group privacy)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Why group privacy.** Datasets that differ in $k$ records are joined by a chain of $k$ neighbouring steps. Applying the definition along the chain multiplies the factor $e^\varepsilon$ $k$ times, and each step's $\delta$ is multiplied by the factors $e^\varepsilon$ of the steps after it; summing gives at most $ke^{(k - 1)\varepsilon}\delta$.
2. **At $\varepsilon = 1$.** $(3 \times 1, 3e^2 \times 10^{-5}) = (3, 2.22\times10^{-4})$.
3. **Posterior.** @eq-epsc-posterior with $3$ in place of $\varepsilon$: $0.01e^3/(0.01e^3 + 0.99) = 0.1687$, against 0.0267 for one person.
4. **At $\varepsilon = 4$.** $(12, 3e^8 \times 10^{-5}) = (12, 0.0894)$: a $\delta$ of almost 9% means no meaningful guarantee for the family through this route.

*Group privacy is a worst-case bound, and the actual curve of three records in DP-SGD is better, because their gradients share the noise; but a per-record guarantee should never be presented as protecting families or users. That needs a different unit ([User-level differential privacy](../../book-full/part05-dp-ml/07-user-level.md)).*
:::

## Two ways to add up the Census budget {#sec-sol-epsc-5}

> **Exercise 5.** **Two ways to add up the Census budget.** The 2020 Census redistricting data spent $\rho = 2.56$ of zero-concentrated differential privacy (zCDP) on persons and $\rho = 0.07$ on housing units. Using @eq-epsc-zcdp at $\delta = 10^{-10}$, convert each separately and add the two $\varepsilon$, then convert the total $\rho$ once. Which is the valid and tighter statement of the combined guarantee? *Check: $17.915 + 2.609 = 20.524$ against 18.194.* *(Composition in zCDP)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Separately.** $\ln(10^{10}) = 23.026$. Persons: $2.56 + 2\sqrt{2.56 \times 23.026} = 17.915$. Housing: $0.07 + 2\sqrt{0.07 \times 23.026} = 2.609$. Sum: 20.524, at $\delta = 2 \times 10^{-10}$ by basic composition of $(\varepsilon, \delta)$ guarantees.
2. **Together.** zCDP composes by adding $\rho$, so the combined release is $2.63$-zCDP, and one conversion gives $2.63 + 2\sqrt{2.63 \times 23.026} = 18.194$ at $\delta = 10^{-10}$.
3. **Which is better.** Both are valid. The second is smaller and has the smaller $\delta$, because the conversion's square root is concave: $\sqrt{\rho_1} + \sqrt{\rho_2} \ge \sqrt{\rho_1 + \rho_2}$, so converting once wastes less.

*This is the same lesson as the chapter's, in another form: keep the guarantee in its richest form (here $\rho$, there the curve) as long as possible, and convert to $(\varepsilon, \delta)$ once, at the end. The Census Bureau's own 17.14 and 2.47 come from a tighter conversion than @eq-epsc-zcdp.*
:::

## Retraining every month {#sec-sol-epsc-6}

> **Exercise 6.** **Retraining every month.** A team retrains the running example's $\varepsilon = 2$ configuration (noise multiplier 2.28) every month on the same data and publishes every model. What is the total $\varepsilon$ at $\delta = 10^{-5}$ after 1, 3 and 12 months under RDP composition, and under basic composition of the $(\varepsilon, \delta)$ guarantees? *Check: RDP 1.998, 3.675, 8.185; basic 2, 6, 24.* *(Composition over releases)*

::: {.callout-tip collapse="true" title="Solution"}
1. **RDP composition.** Each run's RDP curve is $T\varepsilon_{q,\sigma}(\alpha)$, and $k$ runs on the same data add up to $kT\varepsilon_{q,\sigma}(\alpha)$, exactly one run of $kT$ steps. Converting at $\delta = 10^{-5}$ gives 1.998, 3.675 and 8.185.
2. **Basic composition.** Adding $\varepsilon$ and $\delta$ gives $2k$ at $\delta = k \times 10^{-5}$: 2, 6 and 24, at a larger $\delta$.
3. **Reading it.** A year of monthly releases at "$\varepsilon = 2$" is an $\varepsilon \approx 8$ guarantee for the year, by the tightest of the two accountings. The budget in the model card must be the budget for the period the data are used, not for one run.

*State the period with the budget, and account for every release that touched the data, including discarded runs and hyperparameter searches ([Hyperparameter tuning](../../book-full/part05-dp-ml/03-hyperparameter-tuning.md)).*
:::

## From actions to users {#sec-sol-epsc-7}

> **Exercise 7.** **From actions to users.** The Facebook URLs release adds Gaussian noise with $\sigma = 22$ to the counts of likes, one count per URL and group of users, and sets $k = 5$, the 99th percentile of URLs a user likes. Each like moves one count by 1. Find the zCDP parameter $\rho$ and the $\varepsilon$ at $\delta = 10^{-5}$ for one like, for a user with 5 likes, and for a user with 10 likes. Use the release's conversion $\varepsilon = \rho + \sqrt{4\rho\ln(\sqrt{\pi\rho}/\delta)}$, and compare it with @eq-epsc-zcdp. *Check: $\varepsilon = 0.190$, 0.447 and 0.647; @eq-epsc-zcdp gives 0.219, 0.493 and 0.700.* *(Composition in zCDP, user-level units)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One like.** The Gaussian mechanism with noise $\sigma$ on a count that one action moves by 1 is $\rho$-zCDP with $\rho = 1/(2\sigma^2) = 1/968 = 0.00103$. The conversion gives $\varepsilon = 0.190$.
2. **A user with $k$ likes.** Each like is on a different URL, so it moves a different count by one. zCDP composes by adding $\rho$, so the user is $k/(2\sigma^2)$-zCDP: $\rho = 0.00517$ for $k = 5$, which converts to $\varepsilon = 0.447$, the release's user-level target of about 0.45.
3. **A heavier user.** With 10 likes, $\rho$ doubles to 0.01033 and $\varepsilon$ rises to 0.647. It does not double, because $\varepsilon$ grows roughly like $\sqrt\rho$ when $\rho$ is small.
4. **The two conversions.** @eq-epsc-zcdp gives 0.219, 0.493 and 0.700. Both conversions are valid bounds. The release's conversion is tighter because it keeps a factor $\sqrt{\pi\rho}$ inside the logarithm, which is below 1 for small $\rho$.

*A per-action guarantee becomes a per-user one only for users whose number of actions you have bounded. Above the bound, the guarantee weakens smoothly rather than failing, and a privacy statement should say how.*
:::
