# The exponential mechanism

The Laplace and Gaussian mechanisms release numeric values. Other tasks select an object, such as a price or classifier, and small numeric perturbations may sharply reduce its quality. The exponential mechanism samples objects with probability proportional to an exponential of their score. High-scoring objects are more likely, while the selection satisfies differential privacy (DP). This chapter proves its privacy and utility guarantees and applies it to three problems.

## Selecting an object {#sec-em-setting}

A seller offers a digital good, such as an e-book, that costs nothing to copy. Each of $n$ buyers values it at $v_i$ and buys if the price $p$ is at most $v_i$. Revenue at price $p$ is the price multiplied by the number of buyers,

$$
p \cdot \big|\{i : v_i \ge p\}\big| .
$$

The revenue-maximizing price can reveal an individual valuation. If one buyer values the good much more than others, that valuation may determine the optimal price.

Adding noise to the optimal price can sharply reduce revenue. Suppose three buyers value the good at \$1 and one at \$2.50. A price of \$1 earns \$4; a price of \$1.01 earns \$1.01 because only one buyer purchases. Thus one cent of noise reduces revenue by about three quarters. Instead of perturbing the price, select among candidate prices according to their revenue.

In general, a selection problem specifies:

- a dataset $D \in \mathcal X^n$,
- a set $\mathcal R$ of objects,
- a **score function** $u\colon \mathcal X^n \times \mathcal R \to \mathbb R$, where $u(D,r)$ says how good the object $r$ is for the dataset $D$.

For pricing, $D$ is the list of valuations, $\mathcal R$ is the set of prices, and $u(D,p)$ is revenue. The candidate objects and score function are public; only $D$ is private. Score sensitivity measures the maximum change in a fixed candidate's score when one record changes.

::: {#def-em-sensitivity name="Sensitivity of a score function"}
The **sensitivity** of a score function $u$ is

$$
\Delta u = \sup_{r \in \mathcal R}\ \sup_{D \sim D'} \big|u(D,r) - u(D',r)\big| ,
$$

where the inner supremum is over neighbouring datasets, which differ in one record (the replace-one relation).
:::

The score is scalar, so no norm is needed. The mechanism assigns weight $\exp(\varepsilon u(D,r)/(2\Delta))$ to each object $r$, where $\Delta$ upper-bounds the score sensitivity. Dividing by the sum of weights gives probabilities. For an infinite object set, such as an interval of prices, the weights define a density and the normalizer is an integral with respect to a fixed measure $\mu$.

::: {#def-em-mechanism name="Exponential mechanism"}
Let $\varepsilon > 0$, and let $\Delta > 0$ satisfy $\Delta \ge \Delta u$. Let $\mu$ be a measure on $\mathcal R$, taken to be the counting measure, which gives each object weight $1$, when $\mathcal R$ is finite. Assume that the normalizer

$$
Z(D) = \int_{\mathcal R} \exp\!\left(\frac{\varepsilon\, u(D,r')}{2\Delta}\right) \mu(\mathrm d r')
$$

is positive and finite for every $D$. The **exponential mechanism** $M_E$ outputs an object $r \in \mathcal R$ drawn with density, with respect to $\mu$,

$$
p_D(r) = \frac{1}{Z(D)}\exp\!\left(\frac{\varepsilon\, u(D,r)}{2\Delta}\right) .
$$ {#eq-em-density}
:::

For finite $\mathcal R$, @eq-em-density assigns probability proportional to $\exp(\varepsilon u(D,r)/(2\Delta))$. Higher scores are more likely. Smaller $\varepsilon$ or larger $\Delta$ makes the distribution more uniform, increasing privacy and reducing utility.

The definition does not specify an efficient sampling method. For finite $\mathcal R$, direct sampling takes time proportional to $|\mathcal R|$, which is infeasible when the set is exponentially large. For infinite sets, discretization is one option, as in @sec-em-pricing. The mechanism is well-defined but not always computationally efficient.

## Privacy {#sec-em-privacy}

Between neighbours, each score changes by at most $\Delta$, so its weight changes by at most $e^{\varepsilon/2}$. The normalizer changes by the same factor. Their ratio therefore changes by at most $e^\varepsilon$.

::: {#thm-em-privacy}
The exponential mechanism is $\varepsilon$-DP.
:::

::: {.proof}

1. Fix neighbours $D \sim D'$ and an object $r$. By @eq-em-density, the ratio of the densities is

   $$
   \frac{p_D(r)}{p_{D'}(r)} = \exp\!\left(\frac{\varepsilon\,(u(D,r) - u(D',r))}{2\Delta}\right)\cdot\frac{Z(D')}{Z(D)} .
   $$

2. By @def-em-sensitivity and $\Delta \ge \Delta u$, the difference $u(D,r) - u(D',r)$ is at most $\Delta$, so the first factor is at most $e^{\varepsilon/2}$.
3. For every object $r'$, the same bound gives $u(D',r') \le u(D,r') + \Delta$, so $\exp(\varepsilon u(D',r')/(2\Delta)) \le e^{\varepsilon/2}\exp(\varepsilon u(D,r')/(2\Delta))$. Integrating over $r'$ gives $Z(D') \le e^{\varepsilon/2} Z(D)$, so the second factor is at most $e^{\varepsilon/2}$.
4. So $p_D(r) \le e^\varepsilon p_{D'}(r)$ for every $r$. For an event $S \subseteq \mathcal R$, integrating over $S$ gives $\Pr[M_E(D)\in S] \le e^\varepsilon\Pr[M_E(D')\in S]$.
:::

The factor of $2$ in the denominator allocates half the privacy budget to the score and half to the normalizer. If the normalizer is data-independent, the mechanism does not need this allocation.

::: {#cor-em-fixed-normalizer}
Suppose that $Z(D)$, computed with $\Delta$ in place of $2\Delta$, is the same for every dataset $D$. Then the mechanism with density proportional to $\exp(\varepsilon u(D,r)/\Delta)$ is $\varepsilon$-DP.
:::

::: {.proof}
In step 1 of the proof of @thm-em-privacy, with $\Delta$ in place of $2\Delta$, the factor $Z(D')/Z(D)$ equals $1$, and the first factor is at most $\exp(\varepsilon\Delta/\Delta) = e^\varepsilon$. Step 4 then applies unchanged.
:::

## Utility {#sec-em-utility}

Utility requires an output whose score is near the maximum. Define the maximum score and its set of maximizers by

$$
\mathrm{OPT}(D) = \max_{r\in\mathcal R} u(D,r), \qquad \mathcal R^*(D) = \{r \in \mathcal R : u(D,r) = \mathrm{OPT}(D)\} .
$$

The theorem bounds the score loss by approximately $(2\Delta/\varepsilon)\ln|\mathcal R|$. The loss is linear in sensitivity and $1/\varepsilon$, but logarithmic in the number of candidates.

::: {#thm-em-utility}
Let $\mathcal R$ be finite. For every dataset $D$ and every $t \ge 0$,

$$
\Pr\left[u(D, M_E(D)) \le \mathrm{OPT}(D) - \frac{2\Delta}{\varepsilon}\left(\ln\frac{|\mathcal R|}{|\mathcal R^*(D)|} + t\right)\right] \le e^{-t} .
$$
:::

::: {.proof}

1. Fix a threshold $\tau$. The output has score at most $\tau$ exactly when it is one of the objects with $u(D,r) \le \tau$, so

   $$
   \Pr[u(D, M_E(D)) \le \tau] = \frac{\sum_{r : u(D,r)\le \tau}\exp(\varepsilon u(D,r)/(2\Delta))}{\sum_{r'\in\mathcal R}\exp(\varepsilon u(D,r')/(2\Delta))} .
   $$

2. *Numerator.* There are at most $|\mathcal R|$ terms, and each is at most $\exp(\varepsilon \tau/(2\Delta))$ because its score is at most $\tau$.
3. *Denominator.* Every term is positive, so keeping only the $|\mathcal R^*(D)|$ terms with score $\mathrm{OPT}(D)$ gives a lower bound of $|\mathcal R^*(D)|\exp(\varepsilon\,\mathrm{OPT}(D)/(2\Delta))$.
4. Together, steps 2 and 3 give

   $$
   \Pr[u(D, M_E(D)) \le \tau] \le \frac{|\mathcal R|}{|\mathcal R^*(D)|}\exp\!\left(\frac{\varepsilon\,(\tau - \mathrm{OPT}(D))}{2\Delta}\right) .
   $$

5. Set $\tau = \mathrm{OPT}(D) - (2\Delta/\varepsilon)\big(\ln(|\mathcal R|/|\mathcal R^*(D)|) + t\big)$. The exponent becomes $-\ln(|\mathcal R|/|\mathcal R^*(D)|) - t$, and the right-hand side becomes $e^{-t}$.
:::

Since $|\mathcal R^*(D)|\ge1$, the following corollary gives the commonly used, simpler bound.

::: {#cor-em-utility}
Let $\mathcal R$ be finite. For every dataset $D$ and every $t \ge 0$,

$$
\Pr\left[u(D, M_E(D)) \le \mathrm{OPT}(D) - \frac{2\Delta}{\varepsilon}\big(\ln|\mathcal R| + t\big)\right] \le e^{-t} .
$$ {#eq-em-utility}
:::

::: {.proof}
1. At least one object attains the best score, so $|\mathcal R^*(D)| \ge 1$ and $\ln(|\mathcal R|/|\mathcal R^*(D)|) \le \ln|\mathcal R|$. The threshold in @eq-em-utility is therefore at most the threshold of @thm-em-utility for the same $t$, so the event here is contained in the event there, and its probability is at most $e^{-t}$.
:::

## Applications {#sec-em-applications}

Each application specifies the candidate set and score, bounds sensitivity, and applies @thm-em-privacy and @cor-em-utility.

### The Laplace mechanism {#sec-em-laplace}

Let $f\colon\mathcal X^n \to \mathbb R$ have sensitivity $\Delta f$, which means that $|f(D) - f(D')| \le \Delta f$ for all neighbours. Take the real line as the set of objects, with Lebesgue measure, and score each candidate answer by its distance from the true one,

$$
u(D, r) = -|f(D) - r| .
$$

By the triangle inequality, $\big||f(D) - r| - |f(D') - r|\big| \le |f(D) - f(D')| \le \Delta f$, so $\Delta u \le \Delta f$. With $\Delta = \Delta f$, the density @eq-em-density is proportional to $\exp(-\varepsilon|f(D) - r|/(2\Delta f))$, which is the density of $f(D)$ plus Laplace noise of scale $2\Delta f/\varepsilon$. So the exponential mechanism contains a Laplace mechanism, but with twice the noise scale $\Delta f/\varepsilon$ of the [Laplace mechanism](02-laplace-mechanism.md). @cor-em-fixed-normalizer removes the extra factor.

::: {#prp-em-laplace}
Let $f\colon\mathcal X^n \to \mathbb R$ have sensitivity $\Delta f > 0$. The mechanism that outputs $f(D)$ plus noise from the Laplace distribution with scale $\Delta f/\varepsilon$ is $\varepsilon$-DP.
:::

::: {.proof}

1. The output has density proportional to $\exp(-\varepsilon|f(D) - r|/(\Delta f)) = \exp(\varepsilon u(D,r)/(\Delta f))$, with the score above, whose sensitivity is at most $\Delta f$.
2. The normalizer is $\int_{\mathbb R}\exp(-\varepsilon|f(D)-r|/(\Delta f))\,\mathrm d r = 2\Delta f/\varepsilon$, after the substitution $u = r - f(D)$. It does not depend on $D$.
3. @cor-em-fixed-normalizer, with $\Delta = \Delta f$, applies and gives $\varepsilon$-DP.
:::

### Pricing a digital good {#sec-em-pricing}

Return to the seller of @sec-em-setting, now with a dataset of valuations $D = (v_1,\dots,v_n) \in [0,1]^n$ and prices in $[0,1]$. Write $\gamma$ for the best revenue over all prices,

$$
\gamma = \max_{p\in[0,1]} p\cdot|\{i : v_i \ge p\}| .
$$

The maximum exists: revenue increases linearly between consecutive valuations and drops immediately above each valuation, so an optimum occurs at a valuation or at $0$.

Because the continuous price set is difficult to sample from, use a grid. For an integer $m\ge1$, set $\alpha=1/m$ and use the $m$ prices

$$
\mathcal R = \{\alpha, 2\alpha, \dots, 1\} .
$$

Write $\mathrm{OPT}(D)$ for the maximum grid revenue. The next lemma bounds the loss from rounding the optimal price down to the grid.

::: {#lem-em-grid}
The best revenue on the grid satisfies $\mathrm{OPT}(D) \ge \gamma - \alpha n$.
:::

::: {.proof}

1. Let $p^*$ attain $\gamma$. If $p^* < \alpha$, then $\gamma \le p^* n < \alpha n$, so $\gamma - \alpha n < 0$. Revenue is never negative, so $\mathrm{OPT}(D) \ge 0 > \gamma - \alpha n$.
2. Otherwise, round $p^*$ down to the grid, $p' = \alpha\lfloor p^*/\alpha\rfloor \in \mathcal R$, so that $p^* - \alpha < p' \le p^*$.
3. Every buyer with $v_i \ge p^*$ also has $v_i \ge p'$. So the revenue at $p'$ is at least $p'\cdot|\{i : v_i\ge p^*\}| \ge (p^* - \alpha)|\{i : v_i\ge p^*\}| \ge \gamma - \alpha n$.
:::

The private price incurs rounding loss and selection loss. The theorem bounds their sum using a grid of about $\varepsilon n$ prices.

::: {#thm-em-pricing}
Let $\varepsilon > 0$, $m = \lceil \varepsilon n\rceil$ and $\alpha = 1/m$. The exponential mechanism with the grid $\mathcal R$, the revenue $u(D,p) = p\cdot|\{i : v_i \ge p\}|$ as score and $\Delta = 1$ is $\varepsilon$-DP. For every $D\in[0,1]^n$ and $t \ge 0$, with probability at least $1 - e^{-t}$, the revenue at its output price is at least

$$
\gamma - \frac{1 + 2\ln\lceil\varepsilon n\rceil + 2t}{\varepsilon} .
$$ {#eq-em-pricing}
:::

::: {.proof}

1. *Sensitivity.* Changing one valuation changes the number of buyers at any price $p$ by at most $1$, and $p \le 1$. So $\Delta u \le 1 = \Delta$, and @thm-em-privacy gives $\varepsilon$-DP.
2. *Selection.* By @cor-em-utility with $|\mathcal R| = m$, with probability at least $1 - e^{-t}$ the output has revenue at least $\mathrm{OPT}(D) - (2/\varepsilon)(\ln m + t)$.
3. *Discretization.* By @lem-em-grid, $\mathrm{OPT}(D) \ge \gamma - n/m$, and $n/m \le n/(\varepsilon n) = 1/\varepsilon$ because $m \ge \varepsilon n$.
4. Adding the two losses gives @eq-em-pricing.
:::

The loss is $O(\log(\varepsilon n)/\varepsilon)$. If a constant fraction of buyers values the good above a constant price, then $\gamma$ grows linearly with $n$, and the loss is a vanishing fraction of optimal revenue. Grid size balances rounding loss, $\alpha n$, against selection loss, $(2/\varepsilon)\ln(1/\alpha)$. The next example compares the bound with observed utility.

::: {#exm-em-pricing}
Draw $n = 1{,}000$ valuations uniformly from $[0,1]$ and take $\varepsilon = 0.5$, so that the grid has $m = 500$ prices. The best revenue is $\gamma = 267.40$, and the best grid price earns $266.88$. With $t = \ln 20$, @eq-em-pricing guarantees a revenue of at least $267.40 - 38.84 = 228.56$ with probability $0.95$. The exact output distribution does much better. Its mean revenue is $263.86$, and its 5th percentile is $256.52$, a loss of $10.88$, so the guarantee is about 3.6 times too pessimistic. @fig-em-pricing shows the output distribution on a smaller example.
:::

![The exponential mechanism for pricing, with $200$ buyers whose valuations are uniform on $[0,1]$ and a grid of $200$ prices. Top, the revenue. Bottom, the probability of each output price for $\varepsilon = 0.2$ and $\varepsilon = 1$. A larger $\varepsilon$ puts more weight on prices near the best revenue.](figures/em-pricing.svg){#fig-em-pricing}

### Private PAC learning {#sec-em-pac}

In the probably approximately correct (PAC) model, a learner receives labelled records and outputs a classifier accurate on new samples from the same distribution. Here, records are points in $\{0,1\}^d$ with labels in $\{0,1\}$. Accuracy is specified by error $\alpha$ and failure probability $\beta$; here $\alpha$ is not the pricing grid step.

::: {#def-em-pac name="PAC learning"}
A **concept class** is a finite set $\mathcal C$ of functions $c\colon\{0,1\}^d\to\{0,1\}$. An unknown **target** $c^* \in \mathcal C$ and an unknown distribution $\mathcal D$ on $\{0,1\}^d$ generate the dataset. Each record is a point $x_i$, drawn independently from $\mathcal D$, together with its label $y_i = c^*(x_i)$. So the dataset is $D = (z_1, \dots, z_n)$ with labelled records $z_i = (x_i, y_i)$. The **error** of a function $h$ is

$$
\mathrm{err}(h) = \Pr_{x\sim \mathcal D}[h(x) \ne c^*(x)] .
$$

A learner is **$(\alpha,\beta)$-accurate** if, for every target and every $\mathcal D$, it outputs a function $\hat h$ with $\mathrm{err}(\hat h) \le \alpha$ with probability at least $1-\beta$.
:::

For example, $\mathcal C$ may be a finite set of half-spaces that label points on one side of a hyperplane as $1$.

Without privacy, a consistent learner outputs a function in $\mathcal C$ that labels every record correctly; the target guarantees such a function exists. Uniform convergence ensures that every function's sample error is close to its true error. Define the sample error by

$$
\widehat{\mathrm{err}}(h) = \frac1n\big|\{i : h(x_i)\ne y_i\}\big| .
$$

Hoeffding's inequality bounds the probability that an average of $n$ independent $[0,1]$ variables falls below its mean by at least $s$ by $\exp(-2ns^2)$. Apply it to each function's error indicators and take a union bound over $\mathcal C$.

::: {#lem-em-uniform}
Let $\alpha, \beta \in (0,1)$ and $n \ge 2\ln(|\mathcal C|/\beta)/\alpha^2$. With probability at least $1-\beta$ over the sample, every $h\in\mathcal C$ satisfies $\mathrm{err}(h) \le \widehat{\mathrm{err}}(h) + \alpha/2$. In particular, every function in $\mathcal C$ that labels the whole sample correctly has error at most $\alpha/2$.
:::

::: {.proof}

1. Fix $h \in \mathcal C$. The indicators $W_i = \mathbf 1[h(x_i)\ne c^*(x_i)]$ are independent, lie in $[0,1]$, and have mean $\mathrm{err}(h)$, because the $x_i$ are independent draws from $\mathcal D$. Their average is $\widehat{\mathrm{err}}(h)$.
2. Hoeffding's inequality with $s = \alpha/2$ gives $\Pr[\widehat{\mathrm{err}}(h) \le \mathrm{err}(h) - \alpha/2] \le \exp(-n\alpha^2/2) \le \beta/|\mathcal C|$, by the bound on $n$.
3. A union bound over the $|\mathcal C|$ functions shows that some $h$ fails with probability at most $\beta$.
4. A function that labels the sample correctly has $\widehat{\mathrm{err}}(h) = 0$, so its error is at most $\alpha/2$.
:::

Thus, $O(\ln|\mathcal C|/\alpha^2)$ records suffice.

For privacy, neighbours differ in one labelled record, and the replacement $(x_i',y_i')$ need not follow $\mathcal D$ or satisfy $y_i'=c^*(x_i')$. Accuracy assumes data from the learning model; privacy must hold for every dataset.

The private learner selects a function using the exponential mechanism with score equal to negative sample error. Functions with fewer mistakes receive higher probability.

::: {#thm-em-pac name="Private PAC learning"}
Let $\varepsilon > 0$ and $\alpha,\beta\in(0,1)$. Run the exponential mechanism with $\mathcal R = \mathcal C$, the score $u(D,h) = -\widehat{\mathrm{err}}(h)$ and $\Delta = 1/n$. It is $\varepsilon$-DP for every dataset. If

$$
n \ge \frac{2\ln(2|\mathcal C|/\beta)}{\alpha^2} \qquad\text{and}\qquad n \ge \frac{4\ln(2|\mathcal C|/\beta)}{\alpha\varepsilon},
$$

then it is an $(\alpha,\beta)$-accurate learner.
:::

::: {.proof}

1. *Privacy.* Replacing one labelled record changes the number of mistakes of each $h$ by at most $1$, so $\Delta u \le 1/n$. @thm-em-privacy gives $\varepsilon$-DP for all datasets, whatever their labels.
2. *Selection.* The target makes no mistakes, so $\mathrm{OPT} = 0$. By @cor-em-utility with $t = \ln(2/\beta)$, with probability at least $1 - \beta/2$ over the mechanism's coins, the output $\hat h$ has

   $$
   \widehat{\mathrm{err}}(\hat h) \le \frac{2}{\varepsilon n}\big(\ln|\mathcal C| + \ln(2/\beta)\big) = \frac{2\ln(2|\mathcal C|/\beta)}{\varepsilon n} \le \frac\alpha2 ,
   $$

   where the last step uses the second bound on $n$.

3. *Generalization.* The first bound on $n$ is the condition of @lem-em-uniform with $\beta/2$ in place of $\beta$. So with probability at least $1-\beta/2$ over the sample, every $h\in\mathcal C$, including $\hat h$, has $\mathrm{err}(h)\le\widehat{\mathrm{err}}(h)+\alpha/2$.
4. By a union bound, both events hold with probability at least $1-\beta$, and then $\mathrm{err}(\hat h) \le \alpha/2 + \alpha/2 = \alpha$.
:::

The first bound on $n$ is what learning needs without privacy, and the second is the cost of privacy. Both are logarithmic in $|\mathcal C|$, and for $\varepsilon$ of constant size and small $\alpha$, the cost of privacy is a lower-order term. For instance, with $|\mathcal C| = 2^{20}$, $\alpha = \beta = 0.05$ and $\varepsilon = 1$, we have $\ln(2|\mathcal C|/\beta) = 17.552$, and the two bounds are $14{,}041$ and $1{,}404$. The price is computation, because the mechanism scores every function in $\mathcal C$.

## Summary {#sec-em-summary}

| Result | Statement |
| --- | --- |
| @def-em-mechanism | output $r$ with probability proportional to $\exp(\varepsilon u(D,r)/(2\Delta))$ |
| @thm-em-privacy | the exponential mechanism is $\varepsilon$-DP |
| @cor-em-fixed-normalizer | with a normalizer independent of $D$, $\exp(\varepsilon u/\Delta)$ is $\varepsilon$-DP |
| @thm-em-utility | the score falls below $\mathrm{OPT} - \frac{2\Delta}{\varepsilon}(\ln\frac{|\mathcal R|}{|\mathcal R^*|} + t)$ with probability at most $e^{-t}$ |
| @cor-em-utility | the same with $\ln|\mathcal R|$ |
| @prp-em-laplace | the Laplace mechanism is the exponential mechanism with score $-|f(D) - r|$ |
| @thm-em-pricing | a private price earns $\gamma - O(\log(\varepsilon n)/\varepsilon)$ |
| @lem-em-uniform | $2\ln(|\mathcal C|/\beta)/\alpha^2$ records make every sample error close to the true error |
| @thm-em-pac | private PAC learning with $O\big(\ln(|\mathcal C|/\beta)(1/\alpha^2 + 1/(\alpha\varepsilon))\big)$ records |

: The chapter's results. {#tbl-em-summary}

## References {#sec-em-notes}

- Frank McSherry and Kunal Talwar, "Mechanism Design via Differential Privacy", 2007. Defines the exponential mechanism and applies it to the digital goods auction.
- Cynthia Dwork and Aaron Roth, *The Algorithmic Foundations of Differential Privacy*, 2014. Presents the mechanism and its privacy and utility theorems in Section 3.4.
- Salil Vadhan, "The Complexity of Differential Privacy", 2017. Treats private PAC learning in Section 8.
- Andrew V. Goldberg, Jason D. Hartline and Andrew Wright, "Competitive Auctions and Digital Goods", 2001. Poses the digital goods auction.
- Leslie G. Valiant, "A Theory of the Learnable", 1984. Defines the PAC model of learning.
- Shiva Prasad Kasiviswanathan, Homin K. Lee, Kobbi Nissim, Sofya Raskhodnikova and Adam Smith, "What Can We Learn Privately?", 2008. Learns every finite class privately with the exponential mechanism.
- Wassily Hoeffding, "Probability Inequalities for Sums of Bounded Random Variables", 1963. Gives the tail bound behind @lem-em-uniform.
