# The Exponential Mechanism

The Laplace and Gaussian mechanisms release numbers, and they work because a small change in the number does little harm. Many tasks instead ask for the best object from a set, such as a price, a classifier or a candidate, and there a small change in the output can ruin its quality. This chapter answers that need with the exponential mechanism. It samples each object with probability that grows exponentially with a score, so that good objects are likely while the choice reveals little about any one person, as differential privacy (DP) requires. The chapter proves that the mechanism is private and that its output is nearly optimal, and then applies it to three problems.

## Selecting an object {#sec-em-setting}

A seller has a digital good, such as an e-book, that costs nothing to copy, and $n$ buyers who each want one copy. Buyer $i$ values the good at $v_i$ and buys it if the price $p$ is at most $v_i$. The seller's revenue at price $p$ is the price times the number of buyers,

$$
p \cdot \big|\{i : v_i \ge p\}\big| .
$$

The seller could compute the price with the highest revenue from the valuations, but that price can reveal a single buyer. If one buyer values the good far above the others, the best price may be that buyer's valuation.

Adding noise to the best price does not help, because revenue is very sensitive to the price. Take four buyers, three who value the good at \$1 and one who values it at \$2.50. At a price of \$1, all four buy and the revenue is \$4. At \$1.01 only the last buyer buys, so the revenue falls to \$1.01. It climbs back to \$2.50 at a price of \$2.50, and falls to nothing at \$2.51. So a cent of noise can cost three quarters of the revenue. The prices are therefore better treated as objects with a quality, the revenue, rather than as numbers to perturb.

In general, the input to the selection problem has three parts:

- a dataset $X \in \mathcal X^n$,
- a set $\mathcal H$ of objects,
- a **score function** $s\colon \mathcal X^n \times \mathcal H \to \mathbb R$, where $s(X,h)$ says how good the object $h$ is for the dataset $X$.

In the example, $X$ is the list of valuations, $\mathcal H$ is the set of prices, and $s(X,p)$ is the revenue at price $p$. Only the dataset is private. The set of objects and the score function are public, so the sensitivity measures how much the score of a fixed object can move when one entry of the dataset changes.

::: {#def-em-sensitivity name="Sensitivity of a score function"}
The **sensitivity** of a score function $s$ is

$$
\Delta_s = \sup_{h \in \mathcal H}\ \sup_{X \sim X'} \big|s(X,h) - s(X',h)\big| ,
$$

where the inner supremum is over neighbouring datasets, which differ in one entry.
:::

The score is a single real number, so no choice of norm is needed, unlike the $\ell_1$ and $\ell_2$ sensitivities of the Laplace and Gaussian mechanisms. The mechanism samples an object with weight $\exp(\varepsilon s/(2\Delta))$. For an infinite set of objects, such as the real line, the weight is a density with respect to a base measure.

::: {#def-em-mechanism name="Exponential mechanism"}
Let $\varepsilon > 0$, and let $\Delta > 0$ satisfy $\Delta \ge \Delta_s$. Let $\mu$ be a measure on $\mathcal H$, the counting measure when $\mathcal H$ is finite, such that the normalizer

$$
Z(X) = \int_{\mathcal H} \exp\!\left(\frac{\varepsilon\, s(X,h')}{2\Delta}\right) \mu(dh')
$$

is positive and finite for every $X$. The **exponential mechanism** $M_E$ outputs an object $h \in \mathcal H$ drawn with density, with respect to $\mu$,

$$
p_X(h) = \frac{1}{Z(X)}\exp\!\left(\frac{\varepsilon\, s(X,h)}{2\Delta}\right) .
$$ {#eq-em-density}
:::

For finite $\mathcal H$, @eq-em-density says that $\Pr[M_E(X) = h]$ is proportional to $\exp(\varepsilon s(X,h)/(2\Delta))$. A higher score makes an object more likely. A smaller $\varepsilon$ or a larger $\Delta$ flattens the distribution towards uniform, which is more private and less accurate.

The definition says nothing about how to sample. For a finite set, you can compute every weight, add them up and sample, in time proportional to $|\mathcal H|$. That is slow when $\mathcal H$ is exponentially large, which is common. For an infinite set, there is often no direct way to sample, and the usual remedy is to discretize, as @sec-em-pricing does. So the exponential mechanism always gives an algorithm, though not always an efficient one, and it is often the first step towards one.

## Privacy {#sec-em-privacy}

The score of $h$ changes by at most $\Delta$ between neighbours, so the weight of $h$ changes by at most a factor $e^{\varepsilon/2}$. The normalizer is a sum of such weights, so it also changes by at most $e^{\varepsilon/2}$. The two factors together give $e^\varepsilon$.

::: {#thm-em-privacy}
The exponential mechanism is $\varepsilon$-DP.
:::

::: {.proof}

1. Fix neighbours $X \sim X'$ and an object $h$. By @eq-em-density, the ratio of the densities is

   $$
   \frac{p_X(h)}{p_{X'}(h)} = \exp\!\left(\frac{\varepsilon\,(s(X,h) - s(X',h))}{2\Delta}\right)\cdot\frac{Z(X')}{Z(X)} .
   $$

2. By @def-em-sensitivity and $\Delta \ge \Delta_s$, the difference $s(X,h) - s(X',h)$ is at most $\Delta$, so the first factor is at most $e^{\varepsilon/2}$.
3. For every object $h'$, the same bound gives $s(X',h') \le s(X,h') + \Delta$, so $\exp(\varepsilon s(X',h')/(2\Delta)) \le e^{\varepsilon/2}\exp(\varepsilon s(X,h')/(2\Delta))$. Integrating over $h'$ gives $Z(X') \le e^{\varepsilon/2} Z(X)$, so the second factor is at most $e^{\varepsilon/2}$.
4. So $p_X(h) \le e^\varepsilon p_{X'}(h)$ for every $h$. For an event $T \subseteq \mathcal H$, integrating over $T$ gives $\Pr[M_E(X)\in T] \le e^\varepsilon\Pr[M_E(X')\in T]$.
:::

Step 3 is where the factor $2$ in $2\Delta$ is spent. When the normalizer does not depend on the dataset, that half of the budget is not needed.

::: {#cor-em-fixed-normalizer}
Suppose that $Z(X)$, computed with $\Delta$ in place of $2\Delta$, is the same for every dataset $X$. Then the mechanism with density proportional to $\exp(\varepsilon s(X,h)/\Delta)$ is $\varepsilon$-DP.
:::

::: {.proof}
In step 1 of the proof of @thm-em-privacy, with $\Delta$ in place of $2\Delta$, the factor $Z(X')/Z(X)$ equals $1$, and the first factor is at most $\exp(\varepsilon\Delta/\Delta) = e^\varepsilon$. Step 4 then applies unchanged.
:::

## Utility {#sec-em-utility}

The mechanism should output an object whose score is close to the best one. Write the best score and the set of objects that attain it as

$$
\mathrm{OPT}(X) = \max_{h\in\mathcal H} s(X,h), \qquad \mathcal H^*(X) = \{h \in \mathcal H : s(X,h) = \mathrm{OPT}(X)\} .
$$

The theorem says that the output's score is rarely more than about $(2\Delta/\varepsilon)\ln|\mathcal H|$ below $\mathrm{OPT}(X)$. The loss grows linearly with the sensitivity and with $1/\varepsilon$, but only logarithmically with the number of objects.

::: {#thm-em-utility}
Let $\mathcal H$ be finite. For every dataset $X$ and every $t \ge 0$,

$$
\Pr\left[s(X, M_E(X)) \le \mathrm{OPT}(X) - \frac{2\Delta}{\varepsilon}\left(\ln\frac{|\mathcal H|}{|\mathcal H^*(X)|} + t\right)\right] \le e^{-t} .
$$
:::

::: {.proof}

1. Fix a threshold $c$. The output has score at most $c$ exactly when it is one of the objects with $s(X,h) \le c$, so

   $$
   \Pr[s(X, M_E(X)) \le c] = \frac{\sum_{h : s(X,h)\le c}\exp(\varepsilon s(X,h)/(2\Delta))}{\sum_{h'\in\mathcal H}\exp(\varepsilon s(X,h')/(2\Delta))} .
   $$

2. *Numerator.* There are at most $|\mathcal H|$ terms, and each is at most $\exp(\varepsilon c/(2\Delta))$ because its score is at most $c$.
3. *Denominator.* Every term is positive, so keeping only the $|\mathcal H^*(X)|$ terms with score $\mathrm{OPT}(X)$ gives a lower bound of $|\mathcal H^*(X)|\exp(\varepsilon\,\mathrm{OPT}(X)/(2\Delta))$.
4. Together, steps 2 and 3 give

   $$
   \Pr[s(X, M_E(X)) \le c] \le \frac{|\mathcal H|}{|\mathcal H^*(X)|}\exp\!\left(\frac{\varepsilon\,(c - \mathrm{OPT}(X))}{2\Delta}\right) .
   $$

5. Set $c = \mathrm{OPT}(X) - (2\Delta/\varepsilon)\big(\ln(|\mathcal H|/|\mathcal H^*(X)|) + t\big)$. The exponent becomes $-\ln(|\mathcal H|/|\mathcal H^*(X)|) - t$, and the right-hand side becomes $e^{-t}$.
:::

At least one object attains the best score, so $|\mathcal H^*(X)| \ge 1$. Dropping it gives the form that is used most often.

::: {#cor-em-utility}
Let $\mathcal H$ be finite. For every dataset $X$ and every $t \ge 0$,

$$
\Pr\left[s(X, M_E(X)) \le \mathrm{OPT}(X) - \frac{2\Delta}{\varepsilon}\big(\ln|\mathcal H| + t\big)\right] \le e^{-t} .
$$ {#eq-em-utility}
:::

## Applications {#sec-em-applications}

Each application below follows the same recipe. It chooses the set of objects and the score function, bounds the sensitivity, and reads off the privacy and accuracy from @thm-em-privacy and @cor-em-utility.

### The Laplace mechanism {#sec-em-laplace}

Let $f\colon\mathcal X^n \to \mathbb R$ have sensitivity $\Delta$, which means that $|f(X) - f(X')| \le \Delta$ for all neighbours. Take the real line as the set of objects, with Lebesgue measure, and score each candidate answer by its distance from the true one,

$$
s(X, h) = -|f(X) - h| .
$$

By the triangle inequality, $\big||f(X) - h| - |f(X') - h|\big| \le |f(X) - f(X')| \le \Delta$, so $\Delta_s \le \Delta$. The density @eq-em-density is proportional to $\exp(-\varepsilon|f(X) - h|/(2\Delta))$. That is the density of $f(X)$ plus Laplace noise of scale $2\Delta/\varepsilon$, so the exponential mechanism contains a Laplace mechanism.

Its noise is twice what the [Laplace mechanism](02-laplace-mechanism.md) needs, which adds noise of scale $\Delta/\varepsilon$. @cor-em-fixed-normalizer removes the extra factor.

::: {#prp-em-laplace}
Let $f\colon\mathcal X^n \to \mathbb R$ have sensitivity $\Delta > 0$. The mechanism that outputs $f(X) + Y$, where $Y$ has the Laplace distribution with scale $\Delta/\varepsilon$, is $\varepsilon$-DP.
:::

::: {.proof}

1. The output has density proportional to $\exp(-\varepsilon|f(X) - h|/\Delta) = \exp(\varepsilon s(X,h)/\Delta)$, with the score above, whose sensitivity is at most $\Delta$.
2. The normalizer is $\int_{\mathbb R}\exp(-\varepsilon|f(X)-h|/\Delta)\,dh = 2\Delta/\varepsilon$, after the substitution $u = h - f(X)$. It does not depend on $X$.
3. @cor-em-fixed-normalizer applies and gives $\varepsilon$-DP.
:::

### Pricing a digital good {#sec-em-pricing}

Return to the seller of @sec-em-setting, now with valuations $v = (v_1,\dots,v_n) \in [0,1]^n$ and prices in $[0,1]$. Write $\gamma$ for the best revenue over all prices,

$$
\gamma = \max_{p\in[0,1]} p\cdot|\{i : v_i \ge p\}| .
$$

The maximum exists, because the revenue increases linearly between consecutive valuations and drops just after each one, so it is attained at one of the $v_i$ or at $0$. A buyer also gains little from misreporting their valuation to a private mechanism, because their report barely changes the price. That makes such auctions approximately truthful, a link explored by McSherry and Talwar (2007).

The set of all prices is infinite, and the revenue has no structure that makes sampling easy, so the mechanism runs on a grid. Fix an integer $m \ge 1$ and the step $\alpha = 1/m$, and use the $m$ prices

$$
\mathcal H = \{\alpha, 2\alpha, \dots, 1\} .
$$

::: {#lem-em-grid}
The best revenue on the grid satisfies $\mathrm{OPT}(v) \ge \gamma - \alpha n$.
:::

::: {.proof}

1. Let $p^*$ attain $\gamma$. If $p^* < \alpha$, then $\gamma \le p^* n < \alpha n$, and $\mathrm{OPT}(v) \ge 0 > \gamma - \alpha n$, because revenue is never negative.
2. Otherwise, round $p^*$ down to the grid, $p' = \alpha\lfloor p^*/\alpha\rfloor \in \mathcal H$, so that $p^* - \alpha < p' \le p^*$.
3. Every buyer with $v_i \ge p^*$ also has $v_i \ge p'$. So the revenue at $p'$ is at least $p'\cdot|\{i : v_i\ge p^*\}| \ge (p^* - \alpha)|\{i : v_i\ge p^*\}| \ge \gamma - \alpha n$.
:::

::: {#thm-em-pricing}
Let $\varepsilon > 0$, $m = \lceil \varepsilon n\rceil$ and $\alpha = 1/m$. The exponential mechanism with the grid $\mathcal H$, the revenue $s(v,p) = p\cdot|\{i : v_i \ge p\}|$ as score and $\Delta = 1$ is $\varepsilon$-DP. For every $v\in[0,1]^n$ and $t \ge 0$, with probability at least $1 - e^{-t}$, the revenue at its output price is at least

$$
\gamma - \frac{1 + 2\ln\lceil\varepsilon n\rceil + 2t}{\varepsilon} .
$$ {#eq-em-pricing}
:::

::: {.proof}

1. *Sensitivity.* Changing one valuation changes the number of buyers at any price $p$ by at most $1$, and $p \le 1$. So $\Delta_s \le 1 = \Delta$, and @thm-em-privacy gives $\varepsilon$-DP.
2. *Selection.* By @cor-em-utility with $|\mathcal H| = m$, with probability at least $1 - e^{-t}$ the output has revenue at least $\mathrm{OPT}(v) - (2/\varepsilon)(\ln m + t)$.
3. *Discretization.* By @lem-em-grid, $\mathrm{OPT}(v) \ge \gamma - n/m$, and $n/m \le n/(\varepsilon n) = 1/\varepsilon$ because $m \ge \varepsilon n$.
4. Adding the two losses gives @eq-em-pricing.
:::

The loss is $O(\log(\varepsilon n)/\varepsilon)$. When a constant fraction of buyers values the good above a constant price, $\gamma$ grows linearly in $n$, so the loss is a vanishing fraction of the best revenue. The grid size balances the two losses. A finer grid loses less to rounding, $\alpha n$, but more to selection, $(2/\varepsilon)\ln(1/\alpha)$.

::: {#exm-em-pricing}
Draw $n = 1{,}000$ valuations uniformly from $[0,1]$ and take $\varepsilon = 0.5$, so that the grid has $m = 500$ prices. The best revenue is $\gamma = 267.40$, and the best grid price earns $266.88$. With $t = \ln 20$, @eq-em-pricing guarantees a revenue of at least $267.40 - 38.84 = 228.56$ with probability $0.95$. The exact output distribution does much better. Its mean revenue is $263.86$, and its 5th percentile is $256.52$, a loss of $10.88$, so the guarantee is about 3.6 times too pessimistic. @fig-em-pricing shows the output distribution on a smaller example.
:::

![The exponential mechanism for pricing. The top panel is the revenue of $200$ buyers with valuations drawn uniformly from $[0,1]$, on a grid of $200$ prices. The bottom panel is the probability with which the mechanism outputs each price, for $\varepsilon = 0.2$ and $\varepsilon = 1$. Prices near the best revenue are the most likely, and a larger $\varepsilon$ concentrates the distribution on them.](figures/em-pricing.svg){#fig-em-pricing}

### Private PAC learning {#sec-em-pac}

In the probably approximately correct (PAC) model of learning, a learner sees labelled examples and must output a classifier that is accurate on new examples from the same source. Here the examples are points of the Boolean cube $\{0,1\}^d$, labelled $0$ or $1$.

::: {#def-em-pac name="PAC learning"}
A **concept class** is a finite set $\mathcal C$ of functions $c\colon\{0,1\}^d\to\{0,1\}$. An unknown **target** $c^* \in \mathcal C$ and an unknown distribution $D$ on $\{0,1\}^d$ generate the dataset $\big((x_1, y_1),\dots,(x_n,y_n)\big)$, where the $x_i$ are drawn independently from $D$ and $y_i = c^*(x_i)$. The **error** of a function $h$ is

$$
\mathrm{err}(h) = \Pr_{x\sim D}[h(x) \ne c^*(x)] .
$$

A learner is **$(\alpha,\beta)$-accurate** if, for every target and every $D$, it outputs a function $\hat h$ with $\mathrm{err}(\hat h) \le \alpha$ with probability at least $1-\beta$.
:::

For example, $\mathcal C$ might be a finite set of half-spaces, each labelling the points on one side of a hyperplane $1$. The learner sees only the labelled points, and any half-space that separates them correctly is a reasonable guess.

Without privacy, the classic learner outputs any function in $\mathcal C$ that labels every example correctly. One exists, because the target does. It works because, for every function at once, the fraction of mistakes on the sample is close to the error. Write that fraction as

$$
\widehat{\mathrm{err}}(h) = \frac1n\big|\{i : h(x_i)\ne y_i\}\big| .
$$

The tool is again Hoeffding's inequality. Take $n$ independent random variables with values in $[0,1]$. Hoeffding's inequality says that their average falls below its mean by a margin $\tau \ge 0$ or more with probability at most $\exp(-2n\tau^2)$.

::: {#lem-em-uniform}
Let $\alpha, \beta \in (0,1)$ and $n \ge 2\ln(|\mathcal C|/\beta)/\alpha^2$. With probability at least $1-\beta$ over the sample, every $h\in\mathcal C$ satisfies $\mathrm{err}(h) \le \widehat{\mathrm{err}}(h) + \alpha/2$. In particular, every function in $\mathcal C$ that labels the whole sample correctly has error at most $\alpha/2$.
:::

::: {.proof}

1. Fix $h \in \mathcal C$. The indicators $W_i = \mathbf 1[h(x_i)\ne c^*(x_i)]$ are independent, lie in $[0,1]$, and have mean $\mathrm{err}(h)$, because the $x_i$ are independent draws from $D$. Their average is $\widehat{\mathrm{err}}(h)$.
2. Hoeffding's inequality with $\tau = \alpha/2$ gives $\Pr[\widehat{\mathrm{err}}(h) \le \mathrm{err}(h) - \alpha/2] \le \exp(-n\alpha^2/2) \le \beta/|\mathcal C|$, by the bound on $n$.
3. A union bound over the $|\mathcal C|$ functions shows that some $h$ fails with probability at most $\beta$.
4. A function that labels the sample correctly has $\widehat{\mathrm{err}}(h) = 0$, so its error is at most $\alpha/2$.
:::

So about $\ln|\mathcal C|/\alpha^2$ examples suffice. For a function that labels the sample correctly, a multiplicative Chernoff bound improves this to about $\ln|\mathcal C|/\alpha$. For infinite classes, the uniform convergence theorem replaces $\ln|\mathcal C|$ by the Vapnik–Chervonenkis dimension.

For privacy, neighbouring datasets differ in one labelled example $(x_i, y_i)$, which may be replaced by any pair $(x_i', y_i')$. The new point need not come from $D$, and its label need not equal $c^*(x_i')$. Accuracy is proved for data drawn from the model, but privacy must hold for every dataset, so that a wrong model never costs anyone their privacy.

::: {#thm-em-pac name="Private PAC learning"}
Let $\varepsilon > 0$ and $\alpha,\beta\in(0,1)$. Run the exponential mechanism with $\mathcal H = \mathcal C$, the score $s\big((x,y),h\big) = -\widehat{\mathrm{err}}(h)$ and $\Delta = 1/n$. It is $\varepsilon$-DP for every dataset. If

$$
n \ge \frac{2\ln(2|\mathcal C|/\beta)}{\alpha^2} \qquad\text{and}\qquad n \ge \frac{4\ln(2|\mathcal C|/\beta)}{\alpha\varepsilon},
$$

then it is an $(\alpha,\beta)$-accurate learner.
:::

::: {.proof}

1. *Privacy.* Replacing one labelled example changes the number of mistakes of each $h$ by at most $1$, so $\Delta_s \le 1/n$. @thm-em-privacy gives $\varepsilon$-DP for all datasets, whatever their labels.
2. *Selection.* The target makes no mistakes, so $\mathrm{OPT} = 0$. By @cor-em-utility with $t = \ln(2/\beta)$, with probability at least $1 - \beta/2$ over the mechanism's coins, the output $\hat h$ has

   $$
   \widehat{\mathrm{err}}(\hat h) \le \frac{2}{\varepsilon n}\big(\ln|\mathcal C| + \ln(2/\beta)\big) = \frac{2\ln(2|\mathcal C|/\beta)}{\varepsilon n} \le \frac\alpha2 ,
   $$

   where the last step uses the second bound on $n$.

3. *Generalization.* The first bound on $n$ is the condition of @lem-em-uniform with $\beta/2$ in place of $\beta$. So with probability at least $1-\beta/2$ over the sample, every $h\in\mathcal C$, including $\hat h$, has $\mathrm{err}(h)\le\widehat{\mathrm{err}}(h)+\alpha/2$.
4. By a union bound, both events hold with probability at least $1-\beta$, and then $\mathrm{err}(\hat h) \le \alpha/2 + \alpha/2 = \alpha$.
:::

The first bound on $n$ is what learning needs without privacy, and the second is the cost of privacy. Both are logarithmic in $|\mathcal C|$. For $\varepsilon$ of constant size and small $\alpha$, the cost of privacy is a lower-order term. For instance, with $|\mathcal C| = 2^{20}$, $\alpha = \beta = 0.05$ and $\varepsilon = 1$, we have $\ln(2|\mathcal C|/\beta) = 17.552$, and the two bounds are $14{,}041$ and $1{,}404$. The price is computation, since the mechanism scores every function in $\mathcal C$.

## Summary {#sec-em-summary}

| Result | Statement |
| --- | --- |
| @def-em-mechanism | output $h$ with probability proportional to $\exp(\varepsilon s(X,h)/(2\Delta))$ |
| @thm-em-privacy | the exponential mechanism is $\varepsilon$-DP |
| @cor-em-fixed-normalizer | with a normalizer independent of $X$, $\exp(\varepsilon s/\Delta)$ is $\varepsilon$-DP |
| @thm-em-utility | the score falls below $\mathrm{OPT} - \frac{2\Delta}{\varepsilon}(\ln\frac{|\mathcal H|}{|\mathcal H^*|} + t)$ with probability at most $e^{-t}$ |
| @cor-em-utility | the same with $\ln|\mathcal H|$ |
| @prp-em-laplace | the Laplace mechanism is the exponential mechanism with score $-|f(X) - h|$ |
| @thm-em-pricing | a private price earns $\gamma - O(\log(\varepsilon n)/\varepsilon)$ |
| @lem-em-uniform | $2\ln(|\mathcal C|/\beta)/\alpha^2$ examples make every sample error close to the true error |
| @thm-em-pac | private PAC learning with $O\big(\ln(|\mathcal C|/\beta)(1/\alpha^2 + 1/(\alpha\varepsilon))\big)$ examples |

: The chapter's results. {#tbl-em-summary}

## Notes {#sec-em-notes}

- Frank McSherry and Kunal Talwar, "Mechanism Design via Differential Privacy", 2007. The exponential mechanism, with the digital goods auction and its approximate truthfulness.
- Cynthia Dwork and Aaron Roth, *The Algorithmic Foundations of Differential Privacy*, 2014. Section 3.4 presents the exponential mechanism and its privacy and utility theorems, and Section 10.1 the link with truthful mechanism design.
- Salil Vadhan, "The Complexity of Differential Privacy", 2017. Section 8 treats private PAC learning, where the exponential mechanism learns every finite concept class.
- Andrew V. Goldberg, Jason D. Hartline and Andrew Wright, "Competitive Auctions and Digital Goods", 2001. The digital goods auction as a problem in mechanism design.
- Leslie G. Valiant, "A Theory of the Learnable", 1984. The PAC model of learning.
- Shiva Prasad Kasiviswanathan, Homin K. Lee, Kobbi Nissim, Sofya Raskhodnikova and Adam Smith, "What Can We Learn Privately?", 2008. Private PAC learning of every finite class with $O(\ln|\mathcal C|)$ examples, by the exponential mechanism.
- Wassily Hoeffding, "Probability Inequalities for Sums of Bounded Random Variables", 1963. The tail bound behind @lem-em-uniform.
