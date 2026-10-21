# Selection: exponential mechanism, report-noisy-max, sparse vector

## Why this matters

Many questions about a dataset ask you to **pick** something rather than to compute a number. Which of 50 hyperparameter settings scores best on the validation set? Which word is most frequent in a corpus? Which split point gives the best decision stump? Which of a thousand monitored statistics is the first to cross an alarm level? The answer is a name from a list, and adding noise to it makes no sense: "model 7 plus Laplace noise" is not a model.

The [Laplace and Gaussian mechanisms](03-laplace-gaussian.md), the basic tools of differential privacy (DP), do not help directly. They add noise to numbers. You could use them to release a noisy score for every candidate and then take the largest, but the noise has to be scaled to how far one record can move the whole released vector. That distance is the $\ell_1$ **sensitivity**: the largest total change, summed over all coordinates, that replacing, adding or removing one record can cause. If one record can move each of $k$ scores by up to 1, it can move the vector by up to $k$ in total, so the Laplace mechanism needs $k$ times more noise than for a single score. Put differently, releasing $k$ noisy scores is $k$ releases, and under basic composition ([Post-processing, group privacy and composition](02-properties.md)), which says that the privacy costs of several releases add up, the cost grows with the number of candidates. That number is often in the thousands.

This chapter shows that you never have to pay for the losers. Three mechanisms, all built from one move, make selection cost as much as answering a single query:

- The **exponential mechanism** samples a candidate with probability that grows exponentially in its score.
- **Report-noisy-max** adds noise to every score and releases only the name of the winner, never the noisy scores.
- The **sparse vector technique** goes through a long stream of queries and reports only which one first exceeds a threshold. Every query that stays below costs nothing.

### The one idea: release the position, not the values

In each mechanism the output is a position (which candidate, which query), and the noise is added to scores that are never released. The privacy proof fixes everything except the noise of the candidate in question, and notices that this candidate is chosen exactly when its own noise is large enough to clear a bar set by the others. It then asks one question: how far can one record move that *bar*? The bar is a maximum over the other candidates, and a maximum of numbers that each move by at most $\Delta$ moves by at most $\Delta$, however many numbers there are. That is why the number of candidates $k$ does not appear in the privacy cost. It appears only in the accuracy, and only as $\ln k$.

@tbl-sel-overview lists the results of the chapter and where each is used.

| Result | Where it is used |
| --- | --- |
| Exponential mechanism is $\varepsilon$-DP | [Private model selection](../part05-dp-ml/03-hyperparameter-tuning.md); choosing the worst query in [private multiplicative weights](06-query-release.md) |
| Its utility: gap $\le \frac{2\Delta}{\varepsilon}(\ln k + t)$ with probability $1 - e^{-t}$ | Choosing $\varepsilon$ for a selection; the $\ln k$ that [lower bounds](10-lower-bounds.md) show is necessary |
| Applications: the Laplace mechanism, a private learner, a private median | Why the factor 2 can sometimes go; learning from $\ln\lvert C\rvert$ records; private clipping bounds |
| Report-noisy-max with Laplace noise | Noisy argmax over vote counts, as in private aggregation of teacher ensembles in a later part |
| Gumbel noise turns report-noisy-max into the exponential mechanism | Sampling the exponential mechanism stably; top-$c$ selection |
| AboveThreshold is $\varepsilon$-DP for any number of queries below the threshold | Monitoring; stopping a [hyperparameter search](../part05-dp-ml/03-hyperparameter-tuning.md) at the first good candidate; [online private multiplicative weights](06-query-release.md); the reusable holdout of [adaptive data analysis](../part05-dp-ml/11-adaptive-data-analysis.md) |
| Sparse: $c$ answers above, with basic or advanced composition | Finding several large queries; paying $\sqrt c$ instead of $c$ |
| Buggy variants of the sparse vector technique | Reviewing a privacy proof; what a correct implementation must not change |

: The results of this chapter and where each is used. {#tbl-sel-overview}

**How to read this chapter.** Read the toolkit, then the exponential mechanism and its utility, which are the core. The applications section shows the mechanism at work on three problems and can be read later. Report-noisy-max and the Gumbel section are two views of the same mechanism and are short. The sparse vector section has the subtlest proof in this part; read its statement and the picture first, then the derivation slowly. The last section, on buggy variants, shows what goes wrong when one line of that proof is skipped.

**What this chapter assumes, restated.** A **mechanism** $M$ is a randomized algorithm that takes a dataset and returns a random output. Two datasets are **neighbours**, written $D \simeq D'$, when they differ in one person's record: under **add/remove** one of them has one extra record, and under **replace** one record has been swapped for another. $M$ is **$\varepsilon$-differentially private** ($\varepsilon$-DP) if, for every pair of neighbours and every set $S$ of outputs, $\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S]$: no output becomes more than $e^\varepsilon$ times more likely because one record changed. For a single output $y$, the **privacy loss** is $\ln\big(\Pr[M(D) = y]/\Pr[M(D') = y]\big)$, and $\varepsilon$-DP says its absolute value is at most $\varepsilon$. All of this is in [The definition of differential privacy](01-definition.md). Basic composition, from [Post-processing, group privacy and composition](02-properties.md), says that running an $\varepsilon_1$-DP and an $\varepsilon_2$-DP mechanism on the same data is $(\varepsilon_1 + \varepsilon_2)$-DP. The **Laplace distribution** $\mathrm{Lap}(b)$ with scale $b$ has density $\frac1{2b}e^{-|z|/b}$, a two-sided exponential; its upper tail is $\Pr[Z > x] = \frac12e^{-x/b}$ for $x \ge 0$, and so $\Pr[|Z| \ge x] = e^{-x/b}$ ([Probability distributions](../part01-prerequisites/01-distributions.md), [Concentration inequalities](../part01-prerequisites/02-concentration.md)). The **Laplace mechanism** of [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md) adds $\mathrm{Lap}(\Delta/\varepsilon)$ noise to a number of sensitivity $\Delta$.

**Notation.** A selection problem has a finite set $\mathcal R$ of $k$ candidates and a **score function** $u(D, r)$: how good candidate $r$ is on dataset $D$. How much noise a selection needs depends on how far one record can move a score, so the key quantity is the **sensitivity** of the score, the largest change of any one candidate's score between any two neighbours:
$$\Delta u = \max_{r \in \mathcal R}\ \max_{D \simeq D'}\big|u(D, r) - u(D', r)\big|.$$
We write $\Delta$ for $\Delta u$ when the score is clear, and $\mathrm{OPT}(D) = \max_r u(D, r)$ for the best score. Scores are **monotone** if for every pair of neighbours either $u(D', r) \ge u(D, r)$ for all $r$, or $u(D', r) \le u(D, r)$ for all $r$: one record pushes all scores the same way. Counts under add/remove are monotone, because adding a record can only increase each count and removing one can only decrease each count. Logarithms are natural.

## Toolkit: four facts every derivation uses

Every privacy proof in this chapter has the same shape: pick one output, fix all the randomness except one noise variable, see that the output happens exactly when that noise clears a bar, bound how far one record can move the bar, and pay for that move with a Laplace tail. The four facts below are the four pieces of that shape, in order. Each has a one-line proof. The second is the one readers stall on, so it gets a worked case.

### Fact 1 — Pure DP can be checked outcome by outcome

The definition of $\varepsilon$-DP asks about every *set* of outputs, and there are far more sets than outputs. When the outputs can be listed (a candidate's name, the index of a query), it is enough to check single outputs. The reason is that the probability of a set is the sum of the probabilities of its outputs, and if each term on the left is at most $e^\varepsilon$ times the matching term on the right, so is the sum.

If the output set is finite and $\Pr[M(D) = y] \le e^\varepsilon\Pr[M(D') = y]$ for every outcome $y$ and every ordered pair of neighbours, then $M$ is $\varepsilon$-DP.

In symbols, for any set $S$, $\Pr[M(D) \in S] = \sum_{y \in S}\Pr[M(D) = y] \le \sum_{y \in S}e^\varepsilon\Pr[M(D') = y] = e^\varepsilon\Pr[M(D') \in S]$. This is Fact 1 of [The definition of differential privacy](01-definition.md), which says the same thing from the other side: the worst set for a ratio is a single outcome. Every proof below therefore bounds the probability of one output, "candidate $i$ wins" or "the run stops at query $k$".

### Fact 2 — Fix the other randomness, then average

A mechanism often draws several random numbers, and only one of them matters for the output you are looking at. You would like to treat the others as fixed constants, prove the bound, and be done. That is allowed as long as the randomness you fix is drawn the same way whatever the data are. Then the probability under $D$ and the probability under $D'$ are the same weighted average of conditional probabilities, with the same weights, and if every conditional probability under $D$ is at most $e^\varepsilon$ times the one under $D'$, the averages keep that ratio.

Let $W$ be randomness inside the mechanism whose distribution does not depend on the data. If $\Pr[A \mid D, W = w] \le e^\varepsilon\Pr[A \mid D', W = w]$ for every value $w$, then $\Pr[A \mid D] \le e^\varepsilon\Pr[A \mid D']$.

The proof is one line of averaging, $\Pr[A \mid D] = \sum_w\Pr[W = w]\Pr[A \mid D, w] \le \sum_w\Pr[W = w]\,e^\varepsilon\Pr[A \mid D', w] = e^\varepsilon\Pr[A \mid D']$. The weights $\Pr[W = w]$ are the same on both sides only because $W$ does not depend on the data; if they differed, the two averages could weight the conditional probabilities differently and the ratio could escape.

To see it with numbers, take a mechanism that flips a fair coin $W$ and then does something that depends on the data. @tbl-sel-fix-coin shows one event $A$.

| Coin $w$ | $\Pr[W = w]$ | $\Pr[A \mid D, w]$ | $\Pr[A \mid D', w]$ | Ratio |
| --- | --- | --- | --- | --- |
| heads | 0.5 | 0.30 | 0.20 | 1.50 |
| tails | 0.5 | 0.10 | 0.08 | 1.25 |
| average | | 0.20 | 0.14 | 1.43 |

: Fixing a data-independent coin: the ratio after averaging, 1.43, lies between the conditional ratios 1.25 and 1.50, so a bound on every conditional ratio bounds the average. {#tbl-sel-fix-coin}

The averages are $0.5(0.30) + 0.5(0.10) = 0.20$ and $0.5(0.20) + 0.5(0.08) = 0.14$, and their ratio $0.20/0.14 = 1.43$ can never exceed the larger conditional ratio, 1.50, because a ratio of two averages taken with the same weights lies between the smallest and the largest of the individual ratios. In report-noisy-max, $W$ will be the noise of every candidate except one; in the sparse vector technique, it will be the noise of every query before the one that stops the run.

### Fact 3 — Moving a Laplace threshold by $s$ changes its tail by at most $e^{|s|/b}$

After Fact 2 has fixed the other noise, the output will happen exactly when one Laplace variable $Z$ lands above some bar $t$, and moving to a neighbouring dataset will move the bar to $t - s$. The question is how much that can change $\Pr[Z > t]$. The answer comes from the shape of the Laplace density: it falls by a factor $e^{-1/b}$ for each unit you move away from 0, so sliding it by $s$ changes its height at any point by at most a factor $e^{|s|/b}$, and therefore changes the probability of any tail by at most that factor. For $Z \sim \mathrm{Lap}(b)$ and all real $t$ and $s$,
$$\Pr[Z > t - s] \le e^{|s|/b}\,\Pr[Z > t].$$ {#eq-sel-shift}

To prove it, compare the density at $z$ with the density at $z - s$. They are $p(z) = \frac{1}{2b}e^{-|z|/b}$ and $p(z - s) = \frac1{2b}e^{-|z - s|/b}$, so $p(z - s)/p(z) = e^{(|z| - |z - s|)/b}$, and the triangle inequality says that two points $s$ apart differ in distance from 0 by at most $|s|$: $\big||z| - |z - s|\big| \le |s|$. Hence $p(z - s) \le e^{|s|/b}p(z)$ at every $z$. Then change variables: $\Pr[Z > t - s] = \int_{t - s}^\infty p(z)\,dz = \int_t^\infty p(z - s)\,dz \le e^{|s|/b}\int_t^\infty p(z)\,dz$.

With $b = 1$ and $s = 1$, the script gives $\Pr[Z > -1] = 1 - \frac12e^{-1} = 0.8161$ against $\Pr[Z > 0] = 0.5000$, a ratio of 1.6321, well below $e$. With $t = 2$ it gives $\Pr[Z > 1] = \frac12e^{-1} = 0.1839$ against $\Pr[Z > 2] = \frac12e^{-2} = 0.0677$, a ratio of $2.7183 = e$ exactly. The bound is attained there because both bars lie in the upper tail, where every probability has the form $\frac12e^{-x/b}$ and moving the bar by 1 multiplies it by exactly $e^{1/b}$. Every privacy proof in this chapter ends with this step.

### Fact 4 — A maximum moves no more than its arguments

The bar in Fact 3 will be a maximum over many other candidates. If each of them can move by up to $\Delta$, you might fear that the maximum moves by more, perhaps by $\Delta$ for each of them. It does not, because the maximum is just the value of whichever term is on top, and no term can rise or fall by more than $\Delta$.

If $|a'_j - a_j| \le \Delta$ for every $j$, then $|\max_j a'_j - \max_j a_j| \le \Delta$, for any number of terms.

For the proof, raise every term to its highest possible value: $\max_j a'_j \le \max_j(a_j + \Delta) = \max_j a_j + \Delta$. The same argument with the roles of $a$ and $a'$ swapped gives $\max_j a_j \le \max_j a'_j + \Delta$, and the two together are the claim. If all terms move the same way, say $a'_j \in [a_j, a_j + \Delta]$ for every $j$, then the same argument gives $\max_j a'_j - \max_j a_j \in [0, \Delta]$: the maximum moves the same way, by at most $\Delta$. This direction will matter for monotone scores. The bar a candidate has to clear is a maximum over the others, and this fact is why the number of candidates, and the number of queries below a threshold, never enters the privacy cost.

## The exponential mechanism {#sec-sel-exp}

### Context

You want to output a good candidate, and "good" is measured by a score with bounded sensitivity. The natural private answer is a random candidate, tilted toward high scores. How much tilting can you afford? McSherry and Talwar (2007) answered: give each candidate a weight $e^{\varepsilon u/(2\Delta)}$ and pick it with probability proportional to that weight. One record changes each score by at most $\Delta$, so it changes every weight by at most a factor $e^{\varepsilon/2}$, and that is enough for $\varepsilon$-DP whatever the number of candidates. @sec-apx-cal-exponential in the appendices derives this and the utility bound from scratch, with worked numbers.

### Statement

The **exponential mechanism** $M_E$ turns each score into a weight that grows exponentially with it, and picks a candidate with probability equal to its share of the total weight. The total weight $Z(D)$ is the **normalizer** that makes the probabilities add up to 1. So $M_E$ outputs $r \in \mathcal R$ with probability
$$\Pr[M_E(D) = r] = \frac{\exp\big(\frac{\varepsilon\,u(D, r)}{2\Delta}\big)}{Z(D)}, \qquad Z(D) = \sum_{s \in \mathcal R}\exp\Big(\frac{\varepsilon\,u(D, s)}{2\Delta}\Big).$$ {#eq-sel-em}

1. **Privacy.** $M_E$ is $\varepsilon$-DP.
2. **Monotone scores.** If the scores are monotone, the mechanism with weights $\exp\big(\varepsilon u(D, r)/\Delta\big)$, without the 2, is $\varepsilon$-DP.

### Derivation

By Fact 1 it is enough to bound the ratio of the probabilities of one output $r$ on two neighbours. That probability is a weight divided by a total, and one record can change both: it can move $r$'s own score, and it can move all the other scores that make up the total. So split the ratio into those two effects and bound each one. Fix neighbours $D \simeq D'$ and a candidate $r$:
$$\frac{\Pr[M_E(D) = r]}{\Pr[M_E(D') = r]} = \underbrace{\exp\Big(\frac{\varepsilon\big(u(D, r) - u(D', r)\big)}{2\Delta}\Big)}_{\text{numerator}}\cdot\underbrace{\frac{Z(D')}{Z(D)}}_{\text{normalizer}}.$$

1. **Bound the numerator, $r$'s own weight.** By the definition of sensitivity, $u(D, r) - u(D', r) \le \Delta$, so this factor is at most $e^{\varepsilon\Delta/(2\Delta)} = e^{\varepsilon/2}$.
2. **Bound the normalizer, the total weight.** For every candidate $s$, $u(D', s) \le u(D, s) + \Delta$, again by sensitivity, so every term of $Z(D')$ is at most $e^{\varepsilon/2}$ times the matching term of $Z(D)$. Adding up the terms, $Z(D') \le e^{\varepsilon/2}Z(D)$.
3. **Multiply the two bounds.** The ratio is at most $e^{\varepsilon/2}\cdot e^{\varepsilon/2} = e^\varepsilon$. The pair of neighbours and the candidate were arbitrary, so Fact 1 gives $\varepsilon$-DP.

**Monotone scores.** Now the weights are $e^{\varepsilon u/\Delta}$, so each factor alone could be as large as $e^\varepsilon$, and their product as large as $e^{2\varepsilon}$. Monotonicity stops them from both being large at once. For $r$'s own weight to grow from $D'$ to $D$, its score must go down from $D$ to $D'$; for the total to grow, the scores must go up. When all scores move the same way, only one of these can happen.

1. *If every score goes up from $D$ to $D'$,* the numerator $\exp(\varepsilon(u(D, r) - u(D', r))/\Delta) \le 1$, because the exponent is not positive. The normalizer $Z(D')/Z(D) \le e^\varepsilon$, by the argument of step 2 with $\Delta$ in place of $2\Delta$ in the weights.
2. *If every score goes down,* the numerator is at most $e^\varepsilon$, by step 1 with the new weights, and every term of $Z(D')$ is at most the matching term of $Z(D)$, so the normalizer is at most 1.
3. *Either way* the product is at most $e^\varepsilon$. The neighbours in the other order are covered by the same two cases with the directions swapped.

### Interpretation

The 2 in the exponent pays for two things that can both move. One record can raise the score of $r$ and lower the scores of its competitors at the same time: the first makes $r$'s weight larger, the second makes the normalizer smaller, and each effect alone can use up to $e^{\varepsilon/2}$. [Exercise 1](../../book-exercises/part03-dp-foundations/05-selection.md#sec-sol-selection-1) shows a pair of neighbours where the ratio approaches $e^{2\varepsilon}$ if you drop the 2 for scores that are not monotone. With monotone scores the two effects pull in opposite directions, as the derivation showed, so half the budget is enough. A second way out, a normalizer that does not depend on the data, is in @sec-sel-apps.

If you have met the **softmax** of machine learning, which turns scores $x_r$ into probabilities proportional to $e^{x_r/\tau}$ for a temperature $\tau$, you have met this mechanism: it is a softmax of the scores with temperature $2\Delta/\varepsilon$. As $\varepsilon \to 0$ the temperature grows, every weight tends to 1, and the output is uniform and reveals nothing. As $\varepsilon \to \infty$ the largest score's weight dominates and the mechanism becomes the argmax. @fig-sel-em-probs shows the whole range on the model-selection example below.

Three things must hold for the proof to apply. First, the candidate set must not depend on the data. $\mathcal R$ is fixed in advance, because the proof compares the same list of candidates on $D$ and $D'$; if the list were read off the data, a candidate's presence or absence would itself leak. Second, the candidate set may be infinite. On a continuous range, replace the sum by an integral against a fixed base measure $\mu$: the density is proportional to $\mu(r)\,e^{\varepsilon u(D, r)/(2\Delta)}$, and the same two-factor proof goes through, since it never used that the sum was finite. The private median over an interval is the standard example, worked out in @sec-sel-apps; [Exercise 2](../../book-exercises/part03-dp-foundations/05-selection.md#sec-sol-selection-2) does the discrete version. Third, and in the other direction, nothing in the proof uses the structure of $\mathcal R$. The candidates can be models, split points, strings or graphs. This is what makes the mechanism the general-purpose tool for non-numeric outputs.

![Left: the scores of 50 models, model $r$ getting $900 - r$ of 1,000 validation examples right. Right: the probability that the exponential mechanism selects each model, for four values of $\varepsilon$. Small $\varepsilon$ spreads the probability toward uniform; large $\varepsilon$ concentrates it on the best model.](figures/select_em_probs.svg){#fig-sel-em-probs}

### Example: a vote with four options

Six people vote among options A, B, C, D. On dataset $D$ the counts are $(3, 2, 1, 0)$; its neighbour $D'$ has one more vote, for B, so the counts are $(3, 3, 1, 0)$. The score is the count. One added or removed vote changes one count by 1, so $\Delta = 1$ under add/remove. Take $\varepsilon = 1$, so the weights are $e^{u/2}$. @tbl-sel-em-votes shows the script's output.

| Option | A | B | C | D |
| --- | --- | --- | --- | --- |
| $\Pr[M_E(D) = r]$ | 0.4551 | 0.2760 | 0.1674 | 0.1015 |
| $\Pr[M_E(D') = r]$ | 0.3859 | 0.3859 | 0.1420 | 0.0861 |
| Ratio $D/D'$ | 1.1790 | 0.7151 | 1.1790 | 1.1790 |

: The exponential mechanism with $\varepsilon = 1$ on a vote and its neighbour with one extra vote for B. {#tbl-sel-em-votes}

Check the derivation's two factors by hand. The weights on $D$ are $e^{1.5}$, $e^{1}$, $e^{0.5}$ and $e^0 = 1$, which sum to $Z(D) = 9.8487$; on $D'$ B's weight rises to $e^{1.5}$ and the sum to $Z(D') = 11.6121$. Only B's score changed, so for A, C and D the numerator is 1 and the ratio is the normalizer ratio $Z(D')/Z(D) = 11.6121/9.8487 = 1.1790$, below its bound $e^{1/2} = 1.6487$. For B the numerator is $e^{(2 - 3)/2} = e^{-1/2}$, giving $1.1790/1.6487 = 0.7151$. The largest privacy loss on this pair is $|\ln 0.7151| = 0.3353$, a third of $\varepsilon$. That is no contradiction: the bound must hold for every pair, and this pair moves only a single score, so only one of the two factors is at work for each option. Counts are monotone here, so the weights $e^{u}$ are also allowed; they give a sharper selection (A with probability 0.6439 on $D$) and a largest privacy loss of 0.6585, still below 1.

### Check yourself

- If you add the same constant to every score, does the exponential mechanism change? *(No. Every weight is multiplied by the same factor, which cancels against the same factor in the normalizer. Only differences between scores matter.)*
- The score is the number of correct validation predictions, and you switch to accuracy, the same count divided by $n$. What happens to $\Delta$ and to the output distribution? *($\Delta$ becomes $1/n$, and $u/\Delta$ is unchanged, so the distribution is the same. The mechanism is invariant to rescaling the score.)*

## Utility of the exponential mechanism {#sec-sel-utility}

### Context

Privacy came for free in $k$. Accuracy does not: with more candidates there are more bad ones to pick by accident. How far below the best score can the selected candidate be? The answer is a tail bound with only a $\ln k$ cost. The idea of the proof is to compare weights. A bad candidate, one at least $c$ below the best, has a weight at most $e^{-\varepsilon c/(2\Delta)}$ times the weight of the best candidate, which sits in the normalizer. So each bad candidate is picked with probability at most $e^{-\varepsilon c/(2\Delta)}$, and there are at most $k$ of them.

### Statement

Call the **gap** the difference between the best score and the score of the candidate the mechanism picks. The probability that the gap exceeds a few multiples of the temperature $2\Delta/\varepsilon$ falls exponentially. For every dataset $D$ and every $t > 0$,
$$\Pr\Big[u(D, M_E(D)) \le \mathrm{OPT}(D) - \frac{2\Delta}{\varepsilon}(\ln k + t)\Big] \le e^{-t},$$ {#eq-sel-em-utility}
and the expected gap is
$$\mathbb E\big[\mathrm{OPT}(D) - u(D, M_E(D))\big] \le \frac{2\Delta}{\varepsilon}(\ln k + 1).$$
If $k_{\mathrm{OPT}}$ candidates attain $\mathrm{OPT}$, $\ln k$ can be replaced by $\ln(k/k_{\mathrm{OPT}})$. With monotone scores and weights $e^{\varepsilon u/\Delta}$, replace $2\Delta$ by $\Delta$.

### Derivation

Write $G = \mathrm{OPT}(D) - u(D, M_E(D)) \ge 0$ for the gap, and let $B_c = \{r : u(D, r) \le \mathrm{OPT} - c\}$ be the bad candidates, those at least $c$ below the best. The gap is at least $c$ exactly when the mechanism picks one of them, so $\Pr[G \ge c]$ is their total weight divided by $Z(D)$. Bound the top from above and the bottom from below.

1. **Bad candidates are light.** The weight grows with the score, so each $r \in B_c$ has weight $e^{\varepsilon u(D, r)/(2\Delta)} \le e^{\varepsilon(\mathrm{OPT} - c)/(2\Delta)}$.
2. **The normalizer is heavy.** $Z(D)$ is a sum of positive weights that includes the weight of an optimal candidate, so $Z(D) \ge e^{\varepsilon\,\mathrm{OPT}/(2\Delta)}$, and $k_{\mathrm{OPT}}$ times that if there are $k_{\mathrm{OPT}}$ of them.
3. **Divide.** $\Pr[G \ge c] = \Pr[M_E(D) \in B_c] \le |B_c|\,e^{\varepsilon(\mathrm{OPT} - c)/(2\Delta)}/e^{\varepsilon\,\mathrm{OPT}/(2\Delta)} = |B_c|\,e^{-\varepsilon c/(2\Delta)} \le k\,e^{-\varepsilon c/(2\Delta)}$, since there are at most $k$ bad candidates. (With $k_{\mathrm{OPT}}$ optimal candidates, the bad ones number at most $k$ and the normalizer is $k_{\mathrm{OPT}}$ times larger, which gives the $\ln(k/k_{\mathrm{OPT}})$ version.)
4. **Solve for $c$.** Ask which gap $c$ has probability at most $e^{-t}$. Setting $k\,e^{-\varepsilon c/(2\Delta)} = e^{-t}$ and taking logarithms gives $\ln k - \varepsilon c/(2\Delta) = -t$, so $c = \frac{2\Delta}{\varepsilon}(\ln k + t)$, which is @eq-sel-em-utility.
5. **Expectation.** For a non-negative variable, the mean is the integral of its tail, $\mathbb E G = \int_0^\infty\Pr[G \ge c]\,dc$. The bound of step 3 is larger than 1, and so useless, until $c_0 = \frac{2\Delta}{\varepsilon}\ln k$, where $k\,e^{-\varepsilon c_0/(2\Delta)} = 1$. So bound the integrand by 1 up to $c_0$ and by step 3 beyond: $\mathbb E G \le c_0 + \int_{c_0}^\infty k\,e^{-\varepsilon c/(2\Delta)}\,dc = c_0 + \frac{2\Delta}{\varepsilon}$, since the integral equals $\frac{2\Delta}{\varepsilon}k\,e^{-\varepsilon c_0/(2\Delta)} = \frac{2\Delta}{\varepsilon}$.

### Interpretation

The cost of $k$ is logarithmic. Doubling the number of candidates adds $\frac{2\Delta}{\varepsilon}\ln 2$ to the gap, because $\ln(2k) = \ln k + \ln 2$. You can select among $2^{20}$ candidates at the price of $\ln 2^{20} = 20\ln 2 \approx 14$ in units of $2\Delta/\varepsilon$. A [lower bound](10-lower-bounds.md) shows that no $\varepsilon$-DP mechanism can avoid the $\ln k$.

How good is the bound? The derivation made two approximations. Step 3 replaced the number of bad candidates $|B_c|$ by $k$, and their weights by the largest possible bad weight; step 2 kept only one term of $Z$. Both are nearly exact when there is one best candidate and all $k - 1$ others sit exactly $c$ below it. @fig-sel-em-utility (right) shows that case: once the bound is small, the exact probability of missing the best and the bound are almost the same (0.0022 against 0.0023 at a gap of 20). With $k = 50$ and $\varepsilon = 1$, the bound is $50e^{-10}$ and the exact probability is $49e^{-10}/(1 + 49e^{-10})$, so the two differ only in counting 50 bad candidates instead of 49 and in dropping the runners-up from the normalizer.

When scores are spread out, the bound is loose. Then few candidates are near the top, and most bad candidates are much further than $c$ below the best and so much lighter than the bound assumes. @fig-sel-em-utility (left) shows the bound several orders of magnitude above the truth for the model-selection scores below.

![Probability that the exponential mechanism's choice is at least $c$ below the best score, exact against the bound $k e^{-\varepsilon c/2\Delta}$. Left: the 50 models with scores $900 - r$, for $\varepsilon = 0.5$ and 1; the bound is loose because the scores are spread out. Right: one best model and 49 runners-up all exactly $g$ below it, with $\varepsilon = 1$; the bound is tight.](figures/select_em_utility.svg){#fig-sel-em-utility}

### Example: choosing among 50 models

You have trained 50 models and want to release the best one according to a private validation set of 1,000 examples. Model $r$ gets $900 - r$ right (@fig-sel-em-probs, left), so the best is at 90.0% accuracy and neighbouring ranks differ by 0.1%. The score is the count of correct predictions; adding or removing one validation example changes each count by at most 1, so $\Delta = 1$. @tbl-sel-em-models shows the script's output.

| $\varepsilon$ | $\Pr[\text{best}]$ | $\mathbb E[\text{gap}]$ | Bound $2(\ln 50 + 1)/\varepsilon$ | $\mathbb E[\text{gap}]$, monotone weights |
| --- | --- | --- | --- | --- |
| 0.1 | 0.0531 | 15.03 | 98.24 | 9.17 |
| 1 | 0.3935 | 1.54 | 9.82 | 0.58 |
| 5 | 0.9179 | 0.09 | 1.96 | 0.01 |

: The exponential mechanism selecting among 50 models with scores $900 - r$. The gap is in correct predictions out of 1,000. {#tbl-sel-em-models}

At $\varepsilon = 1$ you lose 1.54 correct predictions on average, 0.15% accuracy, although you pick the best model only 39% of the time: the mechanism mostly picks one of the near-ties, which cost only one or two correct predictions. The tail bound with $t = 3$ says the gap exceeds $2(\ln 50 + 3) = 13.82$ with probability at most $e^{-3} = 0.0498$; the exact probability is $9.12\times10^{-4}$, about fifty times smaller, because the scores are spread out. At $\varepsilon = 0.1$ the expected-gap bound, 98.24, exceeds the largest possible gap of 49 (the worst model has score $900 - 49$) and says nothing. On the tied profile of @fig-sel-em-utility (right) with $\varepsilon = 1$, the exact probability of a gap of at least 20 is 0.0022 against a bound of 0.0023.

Counts of correct predictions are monotone under add/remove of a validation example, since one more example can only add correct predictions, so the weights $e^{\varepsilon u}$ are allowed too. They halve the temperature and cut the expected gap at $\varepsilon = 1$ from 1.54 to 0.58.

### Check yourself

- You have $k = 10^6$ candidates instead of 50. By how much does the tail bound at $\varepsilon = 1$, $\Delta = 1$, $t = 3$ grow? *(From $2(\ln 50 + 3) = 13.82$ to $2(\ln 10^6 + 3) = 33.63$: a factor of 20,000 in candidates costs a factor of 2.4 in the gap.)*
- Why does step 3 of the derivation not need the bad candidates to be few? *(It bounds each bad candidate's probability by $e^{-\varepsilon c/(2\Delta)}$ and adds them up; there are at most $k$ of them, which is where $\ln k$ comes from.)*

## Applications of the exponential mechanism {#sec-sel-apps}

### Context

To use the exponential mechanism on a new problem, you choose three things: a candidate set fixed in advance, a score, and the score's sensitivity. Everything else follows from @sec-sel-exp and @sec-sel-utility. This section makes those choices three times, each for a question that comes back later in the book. The first choice recovers the Laplace mechanism, and it shows a second case, besides monotone scores, in which the factor 2 can go. The second turns the mechanism into a learning algorithm that picks a classifier from a finite class. The third finds a median over a continuous interval. Private training in Part 5 clips each example's gradient to a maximum norm. The third application lets you set that norm from the data privately, instead of reading the exact median off the data and leaking it.

### Statement

**The Laplace mechanism is an exponential mechanism.** Take a query $f$ of sensitivity $\Delta$ and let every real number $h$ be a candidate. With a continuum of candidates, the weight $e^{\varepsilon u/(2\Delta)}$ becomes a density along the real line, and the normalizer $Z(D)$ becomes its integral. The fixed base measure of @sec-sel-exp is plain length here, so equal stretches of the line start with equal weight. Score each candidate by how close it is to the true answer, $u(D, h) = -|f(D) - h|$. Then the exponential mechanism's density is proportional to $e^{-\varepsilon|f(D) - h|/(2\Delta)}$, which is a Laplace density of scale $2\Delta/\varepsilon$ centred at $f(D)$.

This normalizer is the same for every dataset, because changing the data only slides the density along the line without changing its shape. A normalizer that cannot move cannot work against any candidate, so the half of the budget reserved for it can go. In general, if $Z(D) = Z(D')$ for every pair of neighbours, then the mechanism with weights $e^{\varepsilon u(D, h)/\Delta}$ is $\varepsilon$-DP. Here that mechanism adds $\mathrm{Lap}(\Delta/\varepsilon)$ noise to $f(D)$, which is the Laplace mechanism itself.

**Learning by choosing a classifier** (Kasiviswanathan, Lee, Nissim, Raskhodnikova & Smith 2008). A learning problem has a finite class $C$ of classifiers and $n$ labelled examples drawn independently from a fixed but unknown distribution. Write $\mathrm{err}_D(h)$ for the fraction of the training examples that a classifier $h$ gets wrong, its **training error**, and $\mathrm{err}(h)$ for the probability that it errs on a fresh example, its **population error**. The private learner is the exponential mechanism over $C$ whose score is minus the number of training mistakes, $u(D, h) = -n\,\mathrm{err}_D(h)$, so $\Delta = 1$, as step 1 of the derivation checks. Its **excess error** is how much worse its choice is than the best classifier in $C$. Fix a small failure probability $\beta$. With probability at least $1 - 2\beta$, the classifier $\hat h$ it returns satisfies
$$\mathrm{err}(\hat h) \le \min_{h \in C}\mathrm{err}(h) + 2\sqrt{\frac{\ln(2|C|/\beta)}{2n}} + \frac{2\ln(|C|/\beta)}{\varepsilon n}.$$ {#eq-sel-learner}
The first excess term is the price of sampling, which you pay without privacy too, and the second is the price of privacy. To reach an excess error of at most $\alpha$, ask each term to be at most $\alpha/2$. Step 5 shows that this is guaranteed once
$$n \ge \max\Big\{\frac{8\ln(2|C|/\beta)}{\alpha^2},\ \frac{4\ln(|C|/\beta)}{\alpha\varepsilon}\Big\}.$$

**A median over an interval.** Take records $x_1, \dots, x_n$ in an interval $[0, B]$ whose upper end $B$ is fixed before looking at the data. Score every point $r$ of the interval by how evenly it splits the data, $u(D, r) = -\big|\#\{i : x_i < r\} - \#\{i : x_i > r\}\big|$, which has $\Delta = 1$ under add/remove (step 1). Call the absolute value in the score the **imbalance** of $r$. The candidates are weighted by length, as for the Laplace mechanism. Here the 2 stays. A new record raises the score of some candidates and lowers it for others, depending on where they sit relative to the record and the median, so the scores change shape instead of shifting. The normalizer therefore changes with the data, and the scores are not monotone either.

Sort the records into $x_{(1)} \le \dots \le x_{(n)}$ and add the end points $x_{(0)} = 0$ and $x_{(n+1)} = B$. Between two consecutive points the counts below and above do not change, so the score is constant on each of the $n + 1$ **pieces** into which the points cut the interval. Write $L_j$ for the length of the piece from $x_{(j)}$ to $x_{(j+1)}$. Exactly $j$ records lie below it, so its score is $-|2j - n|$. So the mechanism can be sampled exactly. Pick piece $j$ with probability proportional to its length times its weight, $L_je^{-\varepsilon|2j - n|/2}$, and then return a uniform point inside it. Like every exponential mechanism, it is $\varepsilon$-DP. For any $t > 0$, with probability at least $1 - e^{-t}$ its output $r$ satisfies
$$\big|\#\{i : x_i < r\} - \#\{i : x_i > r\}\big| \le \frac2\varepsilon\Big(\ln\frac{B}{L^\star} + t\Big),$$ {#eq-sel-median}
where $L^\star$ is the total length of the pieces with the best score.

### Derivation

**The Laplace mechanism.** The proof of @sec-sel-exp split the ratio of the two densities at a candidate $h$ into the candidate's own factor and the normalizer ratio $Z(D')/Z(D)$. The plan is to show that the second factor is exactly 1 here.

1. **The score's sensitivity.** Between neighbours, $f$ moves by at most $\Delta$, and by the triangle inequality $\big||f(D) - h| - |f(D') - h|\big| \le |f(D) - f(D')| \le \Delta$. So the score has sensitivity $\Delta$, for every candidate $h$.
2. **The normalizer does not depend on the data.** With weights $e^{\varepsilon u/\Delta}$, substitute $v = h - f(D)$, which only shifts the variable of integration: $Z(D) = \int e^{-\varepsilon|f(D) - h|/\Delta}\,dh = \int e^{-\varepsilon|v|/\Delta}\,dv = 2\Delta/\varepsilon$. The answer is the same for every dataset, so $Z(D')/Z(D) = 1$.
3. **Only the candidate's own factor is left.** It is $e^{\varepsilon(u(D, h) - u(D', h))/\Delta} \le e^{\varepsilon}$ by step 1. The probability of a set of outputs is the integral of the density over the set. So a bound on the ratio of the densities at every point bounds the ratio of the probabilities of every set, in the same way that Fact 1 adds up single outcomes. Hence the mechanism is $\varepsilon$-DP.
4. **It is the Laplace mechanism.** A density proportional to $e^{-\varepsilon|h - f(D)|/\Delta}$ is the $\mathrm{Lap}(\Delta/\varepsilon)$ density shifted to $f(D)$. Steps 2 and 3 used nothing about the Laplace shape except that $Z(D) = Z(D')$. So the same argument proves the general claim, that a data-independent normalizer allows the weights $e^{\varepsilon u/\Delta}$.

**The learner.** Privacy comes from the exponential mechanism, and accuracy from two bounds: the selection is nearly as good as the best training error, and every training error is close to its population error.

1. **Sensitivity.** Adding, removing or replacing one example changes the number of mistakes of every classifier by at most 1, so $\Delta = 1$, and the learner is $\varepsilon$-DP. The privacy guarantee does not assume that the examples are drawn independently. The example in which two neighbours differ can be any example at all, including one with a wrong label.
2. **The selection is nearly the best on the training set.** @eq-sel-em-utility with $t = \ln(1/\beta)$ says that, with probability at least $1 - \beta$, the selected classifier makes at most $\frac2\varepsilon(\ln|C| + \ln(1/\beta))$ more mistakes than the best one. Divide by $n$: $\mathrm{err}_D(\hat h) \le \min_h\mathrm{err}_D(h) + c_E$ with $c_E = 2\ln(|C|/\beta)/(\varepsilon n)$.
3. **Training errors are close to population errors.** For one fixed classifier, the training error is an average of $n$ independent indicators with mean $\mathrm{err}(h)$, so Hoeffding's inequality ([Concentration inequalities](../part01-prerequisites/02-concentration.md)) gives $\Pr[|\mathrm{err}_D(h) - \mathrm{err}(h)| > s] \le 2e^{-2ns^2}$. A union bound over the $|C|$ classifiers makes this hold for all of them at once except with probability $2|C|e^{-2ns^2}$. Setting that to $\beta$ gives $s = \sqrt{\ln(2|C|/\beta)/(2n)}$.
4. **Chain the bounds.** Let $h^\star$ be the classifier with the smallest population error. On the event where both steps 2 and 3 hold, which has probability at least $1 - 2\beta$, move from the population to the training set and back:
$$\mathrm{err}(\hat h) \le \mathrm{err}_D(\hat h) + s \le \mathrm{err}_D(h^\star) + c_E + s \le \mathrm{err}(h^\star) + c_E + 2s.$$
The first and last inequalities are step 3, and the middle one is step 2, since $\mathrm{err}_D(h^\star)$ is at least the smallest training error. This is @eq-sel-learner.
5. **Records needed.** Ask that each of the two terms be at most $\alpha/2$. The sampling term needs $2s \le \alpha/2$, that is $\ln(2|C|/\beta)/(2n) \le \alpha^2/16$, or $n \ge 8\ln(2|C|/\beta)/\alpha^2$. The privacy term needs $2\ln(|C|/\beta)/(\varepsilon n) \le \alpha/2$, or $n \ge 4\ln(|C|/\beta)/(\alpha\varepsilon)$.

**The median.**

1. **Sensitivity.** Adding or removing a record changes either the count below $r$ or the count above $r$ by 1, or neither, so the difference of the two counts moves by at most 1, and so does its absolute value. Hence $\Delta = 1$ under add/remove, as in [Exercise 2](../../book-exercises/part03-dp-foundations/05-selection.md#sec-sol-selection-2). Replacing a record is a removal followed by an addition, so under replace neighbours $\Delta = 2$.
2. **Privacy.** This is the exponential mechanism on a continuous range with length as the base measure, which @sec-sel-exp showed is $\varepsilon$-DP, because the two-factor proof bounds the ratio of the densities at every point.
3. **Exact sampling.** Inside the piece from $x_{(j)}$ to $x_{(j+1)}$, exactly $j$ records lie below $r$ and $n - j$ above, so the score is $-|j - (n - j)| = -|2j - n|$. The probability of the piece is the integral of the density over it, which is $L_je^{-\varepsilon|2j - n|/2}/Z$, and within the piece the density is flat. The end points of the pieces have length zero and can be ignored.
4. **Utility.** Repeat the derivation of @sec-sel-utility with sums over candidates replaced by integrals over length. Write $\mathrm{OPT}$ for the best score, as in @sec-sel-utility. The candidates whose score is at least $c$ below it have total length at most $B$, and each has weight at most $e^{\varepsilon(\mathrm{OPT} - c)/2}$, so together they weigh at most $Be^{\varepsilon(\mathrm{OPT} - c)/2}$. The normalizer includes the best pieces, which weigh $L^\star e^{\varepsilon\,\mathrm{OPT}/2}$. Dividing, the probability that the output's score falls $c$ or more below the best is at most $(B/L^\star)e^{-\varepsilon c/2}$. Setting this to $e^{-t}$ and solving for $c$ gives @eq-sel-median.

### Interpretation

**Two ways to drop the 2.** The 2 in the exponent pays for the normalizer, which can move against the candidate. Monotone scores avoid that payment because the two factors can never both be large, and a data-independent normalizer avoids it because the second factor is always 1. The second case needs two conditions. The candidates must fill the whole real line, or another set that looks the same after a shift. And changing the data must only shift the scores along the candidates, without changing their shape. Only then does the shifted density integrate to the same total. That is why the Laplace mechanism needs noise of scale $\Delta/\varepsilon$, while the generic recipe would give it $2\Delta/\varepsilon$, twice as much.

**Privacy adds a term that shrinks faster.** In @eq-sel-learner the sampling term falls like $1/\sqrt n$ and the privacy term like $1/n$. So for fine accuracy privacy costs little. The ratio of the two requirements, privacy over sampling, is about $\alpha/(2\varepsilon)$, because the two logarithms are close. So when $\alpha$ is much smaller than $\varepsilon$, the sampling term sets the number of records. This is the same message as the sample-complexity section of [Lower bounds](10-lower-bounds.md), which also shows that the $\ln|C|$ cannot be avoided under pure DP. The learner has two limits. Its running time is proportional to $|C|$, because it scores every classifier. And it needs a finite class, which is more than a technicality, because some infinite classes cannot be learned privately at all. Thresholds at every real number are one example. Alon, Livni, Malliaris & Moran (2019) showed that no private algorithm learns them in the probably approximately correct (PAC) sense of learning theory, even with $\delta > 0$. Choosing among a few trained models on a private validation set, as in [Hyperparameter tuning](../part05-dp-ml/03-hyperparameter-tuning.md), is the same mechanism with $C$ the list of models.

**The range costs only its logarithm.** The upper end $B$ enters @eq-sel-median only through $\ln(B/L^\star)$, so a generous range is cheap. What the guarantee controls is the imbalance, that is, how far the output's rank is from the middle, not how far its value is from the median. If the data are dense near the median, a small error in rank is a small error in value. If they are sparse, the best piece is long, and the output can be anywhere in it. Two conditions are essential. The range must be fixed without looking at the data, and records outside it must be clipped into it. A range read off the data, such as the observed maximum, leaks exactly as a data-dependent clipping bound does in [Sensitivity](07-sensitivity.md). Other quantiles work the same way. For the quantile $q$, use the score
$$u(D, r) = -\big|(1 - q)\,\#\{i : x_i < r\} - q\,\#\{i : x_i > r\}\big|.$$
It reaches its best value, zero, when a fraction $q$ of the records lie below $r$. One added record moves it by at most $\max(q, 1 - q)$, which is at most 1, so it still has sensitivity 1.

![Left: excess population error of the learner that picks one of 1,001 threshold classifiers with the exponential mechanism, against the non-private learner that picks the fewest training mistakes, averaged over 100 datasets per point (the text's numbers use 200); the non-private learner finds the best threshold exactly at $n = 10{,}000$, which a log axis cannot show. Right: 1,000 lognormal gradient norms (histogram) and the density of the private median over $[0, 10]$ for $\varepsilon = 0.1$ and $\varepsilon = 1$, zoomed in around the true median.](figures/select_em_applications.svg){#fig-sel-em-apps}

### Example: a threshold classifier and a clipping bound

*Learning a threshold.* The data are $x$ uniform on $[0, 1]$ with label 1 when $x \ge 0.3$, and each label is flipped with probability 0.1, so the best classifier still errs on 10% of examples. The class $C$ holds the 1,001 thresholds $0, 0.001, \dots, 1$. Over 200 datasets of $n = 1{,}000$ records, the private learner's average excess error is 0.0024 at $\varepsilon = 1$ and 0.0206 at $\varepsilon = 0.1$. The non-private learner, which picks the fewest training mistakes, averages 0.0008 (@fig-sel-em-apps, left). The bound @eq-sel-learner at $\beta = 0.05$ has a sampling term of 0.1456 and a privacy term of 0.0198, both far above the observed errors. The sampling term is the looser one, because it guards against the worst of all 1,001 classifiers at once, through a union bound, while neighbouring thresholds have nearly the same errors. For an excess error of $\alpha = 0.05$ the sampling term asks for 33,913 records and the privacy term at $\varepsilon = 1$ for only 793. Privacy becomes the larger cost only at strong privacy, and at $\varepsilon = 0.01$ it asks for 79,236 records.

*A clipping norm for private training.* Private training clips each example's gradient to a maximum norm, and a common choice for that norm is the median of the gradient norms. Take 1,000 norms drawn from a lognormal distribution with median 1, capped at the range $B = 10$. Their median is 0.9804, and their 45th and 55th percentiles are 0.9003 and 1.0589. At $\varepsilon = 1$ the private median is on average 0.0013 away from the true one. That average is exact, computed piece by piece, and 20,000 simulated draws give the same 0.0013. To leave the band between the 45th and 55th percentiles, the output must have at most 450 records on one side and at least 550 on the other, an imbalance of 100 or more. At $\varepsilon = 1$ that happens with probability $1.31\times10^{-21}$. At $\varepsilon = 0.1$ the average distance is 0.0157, and the probability of leaving those percentiles is 0.0111 (@fig-sel-em-apps, right). With $t = \ln 20$, so that the guarantee holds with probability 0.95, the bound @eq-sel-median allows an imbalance of 26.4 at $\varepsilon = 1$ and 264.5 at $\varepsilon = 0.1$. A range of 1,000 instead of 10 raises the first bound to 35.7 and leaves the average distance at 0.0013.

### Check yourself

- Why can't the median mechanism drop its factor 2 the way the Laplace mechanism does? *(Its normalizer depends on the data: a new record raises the score of the pieces on one side of the median and lowers it on the other, so the scores change shape instead of shifting. They are not monotone either.)*
- You refine the learner's grid from 1,001 thresholds to $10^6$. By how much does each term of @eq-sel-learner grow at $\beta = 0.05$? *(The sampling term contains $\ln(2|C|/\beta)$, which grows from $\ln 40{,}040 = 10.60$ to $\ln(4\times10^7) = 17.50$, so the term grows by $\sqrt{17.50/10.60} = 1.28$. The privacy term contains $\ln(|C|/\beta)$, which grows from $\ln 20{,}020 = 9.90$ to $\ln(2\times10^7) = 16.81$, a factor of 1.70.)*

## Report-noisy-max {#sec-sel-rnm}

### Context

Here is the obvious way to select privately. Add independent noise to every score, and report the index of the largest noisy score. Release *only the index*. If you released the noisy scores as well, you would be releasing a vector, and its $\ell_1$ sensitivity, the total change over all coordinates, would set the noise. For $k$ counting queries that one record can all change (for example "how many records have feature $j$", for $k$ features), that sensitivity is $k$, and the Laplace mechanism would need noise of scale $k/\varepsilon$. Releasing only the argmax needs noise of scale $1/\varepsilon$, for any $k$, because of the bar argument sketched at the start of this chapter.

### Statement

**Report-noisy-max** adds an independent Laplace noise of scale $b$ to every score and outputs the candidate with the largest noisy score:
$$M_N(D) = \arg\max_{r \in \mathcal R}\big(u(D, r) + Z_r\big), \qquad Z_r \sim \mathrm{Lap}(b) \text{ independent}.$$
Ties have probability zero, because the noise is continuous.

1. **General scores.** With $b = 2\Delta/\varepsilon$, $M_N$ is $\varepsilon$-DP.
2. **Monotone scores** (Dwork & Roth, Claim 3.9, stated there for counting queries with $\Delta = 1$). With $b = \Delta/\varepsilon$, $M_N$ is $\varepsilon$-DP.
3. **Accuracy.** With probability at least $1 - \beta$, $\mathrm{OPT}(D) - u(D, M_N(D)) \le 2b\ln(k/\beta)$.

### Derivation

The proof below covers both privacy claims. It is the argument of Dwork & Roth's Claim 3.9, written so that the general and the monotone case differ in one step. It uses all four facts of the toolkit in order: look at one output (Fact 1), freeze the other candidates' noise (Fact 2), see that the output happens when one Laplace noise clears a bar, bound the bar's movement (Fact 4), and pay for it (Fact 3).

1. **Fix everything but one noise.** Fix neighbours $D \simeq D'$, a candidate $i$, and the noise $z_{-i}$ of all other candidates. The noises are drawn the same way whatever the data, so by Fact 2 it is enough to bound $\Pr[M_N(D) = i \mid z_{-i}]$ against the same probability on $D'$.
2. **Candidate $i$ wins when its noise clears a bar.** With the others' noise fixed, their noisy scores are fixed numbers, and $i$ wins on $D$ exactly when its own noisy score beats the best of them, $u(D, i) + Z_i > \max_{j \ne i}(u(D, j) + z_j)$. Moving $u(D, i)$ to the right-hand side, that is $Z_i > t(D)$ with the bar
$$t(D) = \max_{j \ne i}\big(u(D, j) + z_j\big) - u(D, i).$$
So $\Pr[M_N(D) = i \mid z_{-i}] = \Pr[Z_i > t(D)]$, and likewise for $D'$.
3. **The bar moves by at most $2\Delta$.** The first term of $t$ is a maximum of numbers $u(\cdot, j) + z_j$, and each moves by at most $\Delta$ between $D$ and $D'$ because the $z_j$ are fixed. By Fact 4 the maximum moves by at most $\Delta$, however many terms it has. The second term, $u(\cdot, i)$, moves by at most $\Delta$ too. So $|t(D) - t(D')| \le 2\Delta$.
4. **With monotone scores, by at most $\Delta$.** If every score goes up by an amount in $[0, \Delta]$, the maximum goes up by an amount in $[0, \Delta]$ (Fact 4, the same-direction case) and so does $u(\cdot, i)$. Their difference, the bar, changes by an amount in $[-\Delta, \Delta]$, because subtracting two numbers in $[0, \Delta]$ gives a number in $[-\Delta, \Delta]$. The same holds if every score goes down.
5. **Shift the threshold.** By Fact 3 with $s = t(D) - t(D')$, $\Pr[Z_i > t(D)] \le e^{|t(D) - t(D')|/b}\Pr[Z_i > t(D')]$. In the general case that factor is at most $e^{2\Delta/b} = e^\varepsilon$ with $b = 2\Delta/\varepsilon$, and in the monotone case at most $e^{\Delta/b} = e^\varepsilon$ with $b = \Delta/\varepsilon$.
6. **Average and collect.** Fact 2 removes the conditioning on $z_{-i}$, and Fact 1 turns the bound for each $i$ into $\varepsilon$-DP.

@fig-sel-rnm-threshold draws steps 2 to 5 for $b = 1$, $t(D) = 0.7$ and a bar lowered by $s = 1$ on $D'$: the winning probability rises from $\Pr[Z_i > 0.7] = \frac12e^{-0.7} = 0.2483$ to $\Pr[Z_i > -0.3] = 1 - \frac12e^{-0.3} = 0.6296$, a ratio of 2.5357, below $e = 2.7183$. The ratio falls short of $e$ because the lowered bar has crossed 0, out of the exponential tail where Fact 3 is exact.

![The proof of report-noisy-max in one picture. With the other candidates' noise fixed, candidate $i$ wins when its own noise $Z_i \sim \mathrm{Lap}(1)$ exceeds a bar $t$. Moving to a neighbouring dataset moves the bar by at most $s$, and the winning probability changes by at most a factor $e^{s/b}$.](figures/select_rnm_threshold.svg){#fig-sel-rnm-threshold}

**Accuracy.** The idea is that if no noise is large, the noise cannot reorder candidates whose scores are far apart. A single Laplace noise exceeds $x$ in absolute value with probability $e^{-x/b}$. By the union bound, the probability that at least one of $k$ events happens is at most the sum of their probabilities, so some $|Z_r|$ reaches $x$ with probability at most $k\,e^{-x/b}$. Setting this to $\beta$ gives $x = b\ln(k/\beta)$: every $|Z_r| < b\ln(k/\beta)$ with probability at least $1 - \beta$. This is the maximum-of-Laplace rule of [Concentration inequalities](../part01-prerequisites/02-concentration.md). On that event the winner $i$ beat a best candidate $r^*$ despite noise of at most $b\ln(k/\beta)$ on each side, so $u(D, i) + b\ln(k/\beta) > u(D, i) + Z_i \ge u(D, r^*) + Z_{r^*} > \mathrm{OPT} - b\ln(k/\beta)$, and the gap is less than $2b\ln(k/\beta)$. This is [Exercise 5](../../book-exercises/part01-prerequisites/02-concentration.md#sec-sol-concentration-5) of that chapter.

### Interpretation

The privacy cost is not $k\varepsilon$ because of step 3. That is where the losers would cost something, and they don't: they enter only through one maximum, which moves by at most $\Delta$ however many of them there are (Fact 4).

The noise must be $2\Delta/\varepsilon$ in general because the bar has two parts that can move in opposite directions: the competitors can rise while candidate $i$ falls, and then the bar rises by both amounts. Monotone scores cannot do that, which halves the noise. This is the same factor of 2 as in the exponential mechanism, for the same reason.

Should you use Laplace report-noisy-max or the exponential mechanism? They are different mechanisms with the same privacy, and neither dominates. On the 50 models at $\varepsilon = 1$, Laplace report-noisy-max with $b = 2$ picks the best model with probability 0.4241 against 0.3935 and has expected gap 1.40 against 1.54. With two candidates the exponential mechanism is slightly better ([Exercise 6](../../book-exercises/part03-dp-foundations/05-selection.md#sec-sol-selection-6)). Adding one-sided exponential noise instead of Laplace noise gives a third mechanism, which equals the "permute-and-flip" mechanism of McKenna & Sheldon (2020); its expected score is never below the exponential mechanism's.

Finally, release only the index. Releasing the winning noisy score as well breaks the proof: the bar argument conditions on everything but $Z_i$ and averages over $Z_i$, and $Z_i$ itself would then be published. To release the winner's score, spend extra budget on a fresh Laplace answer for that one query.

### Example: the vote and the 50 models

On the vote of @tbl-sel-em-votes, counts are monotone, so $b = 1/\varepsilon = 1$ is allowed. The script integrates the winning probabilities exactly: on $D$ they are $(0.6551, 0.2380, 0.0794, 0.0275)$, on $D'$ $(0.4639, 0.4639, 0.0534, 0.0189)$, and the largest privacy loss over the four outputs, the largest $|\ln|$ of a ratio of matching entries, is 0.6674, below $\varepsilon = 1$. With the general scale $b = 2$ it is 0.3626.

On the 50 models at $\varepsilon = 1$, the general mechanism ($b = 2$) has expected gap 1.40 (1.39 in 200,000 simulated runs), and the monotone one ($b = 1$) has 0.54 and picks the best model with probability 0.6463. The accuracy bound at $\beta = 0.05$ is $2b\ln(50/0.05) = 2b\ln 1000$, which is 27.63 for $b = 2$ and 13.82 for $b = 1$. Like the exponential mechanism's bound, it is loose for spread-out scores.

### Check yourself

- Where does the proof use that the noises are independent? *(In step 1: conditioning on $z_{-i}$ must leave $Z_i$ with its Laplace distribution, so that Fact 3 applies.)*
- If the scores are a histogram under replace neighbours (one count goes down by 1, another goes up by 1), can you use $b = 1/\varepsilon$? *(No. The scores move in opposite directions, so they are not monotone; use $b = 2/\varepsilon$.)*

## Gumbel noise: report-noisy-max is the exponential mechanism {#sec-sel-gumbel}

### Context

Report-noisy-max with Laplace noise is close to the exponential mechanism but not equal to it (@fig-sel-gumbel-em). Is there a noise distribution for which "add noise and take the argmax" gives exactly the softmax probabilities of @eq-sel-em? There is: the Gumbel distribution. This is the **Gumbel-max trick**, used throughout machine learning to sample from a softmax. For privacy it gives a second proof that the exponential mechanism is private, a numerically stable way to sample it, and a fast way to select the top $c$ candidates.

### Statement

The **Gumbel distribution** arises as the limit of the maximum of many independent variables with exponential tails, after shifting; what matters here is only its formula. $G \sim \mathrm{Gumbel}(\beta)$, with scale $\beta$, has cumulative distribution function (CDF) $\Pr[G \le g] = \exp(-e^{-g/\beta})$; NumPy's `rng.gumbel(0, beta)` samples it. The claim is that adding independent Gumbel noise to each score and taking the argmax picks candidate $i$ with exactly its softmax probability. For independent $G_r \sim \mathrm{Gumbel}(\beta)$,
$$\Pr\Big[\arg\max_r\big(u_r + G_r\big) = i\Big] = \frac{e^{u_i/\beta}}{\sum_r e^{u_r/\beta}}.$$ {#eq-sel-gumbel}
With $\beta = 2\Delta/\varepsilon$ this is the exponential mechanism.

### Derivation

Rescaling all scores and noises by $1/\beta$ does not change which one is largest, so take $\beta = 1$. The Gumbel CDF is then $F(g) = \exp(-e^{-g})$, and its derivative, the density, is $f(g) = e^{-g}\exp(-e^{-g})$.

**Two candidates first.** Candidate 1 wins when $G_2 - G_1 < u_1 - u_2$, so you need the distribution of the difference of two independent Gumbels. It is the **logistic** distribution, with CDF $\Pr[G_1 - G_2 \le x] = 1/(1 + e^{-x})$, and $G_2 - G_1$ has the same distribution, because swapping the names of two identically distributed independent noises changes nothing. (To get it, integrate $F(g + x)$ against $f(g)$; it is the general calculation below with $k = 2$.) So
$$\Pr[u_1 + G_1 > u_2 + G_2] = \Pr[G_2 - G_1 < u_1 - u_2] = \frac{1}{1 + e^{-(u_1 - u_2)}} = \frac{e^{u_1}}{e^{u_1} + e^{u_2}},$$
where the last step multiplies top and bottom by $e^{u_1}$. That is the softmax of two scores.

**In general.** The plan is to condition on the value of the winner's noise, use independence to multiply the probabilities that each other candidate falls short, and notice that the resulting integral can be done exactly.

1. **Condition on the winner's noise.** If candidate $i$'s noise is $G_i = g$, it wins when every other candidate's noisy score is below $u_i + g$, that is, when every other $G_j < g + u_i - u_j$. Each of these has probability $F(g + u_i - u_j)$, and the noises are independent, so the probabilities multiply. Averaging over $g$ with density $f$,
$$\Pr[i \text{ wins}] = \int_{-\infty}^\infty f(g)\prod_{j \ne i}F(g + u_i - u_j)\,dg.$$
2. **Collect the exponentials.** The density can be written $f(g) = e^{-g}F(g)$, and $F(g)$ is $F(g + u_i - u_j)$ for $j = i$. Each factor is $F(g + u_i - u_j) = \exp(-e^{-g}e^{u_j - u_i})$, and a product of exponentials is the exponential of the sum, so the product of all the $F$ factors, including the one inside $f$, is $\exp\big(-e^{-g}\sum_je^{u_j - u_i}\big) = \exp(-e^{-g}A)$ with $A = e^{-u_i}\sum_je^{u_j}$. The integrand is $e^{-g}\exp(-e^{-g}A)$.
3. **Substitute $w = e^{-g}$**, so $dw = -e^{-g}\,dg$, and $g$ from $-\infty$ to $\infty$ is $w$ from $\infty$ to 0; the minus sign flips the limits back:
$$\Pr[i \text{ wins}] = \int_0^\infty e^{-wA}\,dw = \frac1A = \frac{e^{u_i}}{\sum_je^{u_j}}.$$

### Interpretation

The exponential mechanism now has two proofs. It can be analysed by its weights (@sec-sel-exp) or as a noisy max. The bar argument of @sec-sel-rnm also works with Gumbel noise of scale $2\Delta/\varepsilon$, because the Gumbel tail also satisfies $\Pr[G > t - s] \le e^{s/\beta}\Pr[G > t]$ for $s \ge 0$ (for $s < 0$ the probability only falls). To see why, write the tail as $1 - e^{-y}$ with $y = e^{-t/\beta}$. Lowering $t$ by $s$ multiplies $y$ by $e^{s/\beta} \ge 1$, and because $1 - e^{-y}$ is zero at $y = 0$ and concave, multiplying $y$ by a factor of at least 1 multiplies $1 - e^{-y}$ by at most that factor.

The noisy-max form also gives stable sampling. Computing $e^{\varepsilon u/(2\Delta)}$ for scores in the thousands overflows. The noisy max works in score units directly. (Subtracting the largest score before exponentiating, as the script does, is the other fix; it is allowed because adding a constant to every score does not change the mechanism.)

It also gives top-$c$ selection in one shot. Taking the $c$ largest of $u_r + G_r$ gives the same distribution as running the exponential mechanism $c$ times, removing each winner (Durfee & Rogers 2019; @sec-apx-mech-gumbel in the appendices derives it through a race of exponential clocks). Its privacy is that of $c$ exponential mechanisms, $c\varepsilon$ by basic composition.

Laplace noise is close, not equal. Laplace and Gumbel noise of the same scale have similar bodies but different tails, so the selection probabilities differ by a few percent.

![Selection probabilities for five candidates with scores $(4, 3, 3, 1, 0)$ and noise scale $2\Delta/\varepsilon$ with $\varepsilon = 1$: the exponential mechanism's exact probabilities, the frequencies of the Gumbel noisy max in 200,000 runs, and the exact probabilities of the Laplace noisy max.](figures/select_gumbel_em.svg){#fig-sel-gumbel-em}

### Example: five candidates, a million draws

For scores $(4, 3, 3, 1, 0)$ and $\varepsilon = 1$, $\Delta = 1$, the noise scale is $2\Delta/\varepsilon = 2$. @tbl-sel-gumbel compares the three mechanisms.

| Candidate score | 4 | 3 | 3 | 1 | 0 |
| --- | --- | --- | --- | --- | --- |
| Exponential mechanism, exact | 0.3889 | 0.2359 | 0.2359 | 0.0868 | 0.0526 |
| Gumbel noisy max, $10^6$ runs | 0.3891 | 0.2351 | 0.2362 | 0.0871 | 0.0525 |
| Laplace noisy max, exact | 0.4105 | 0.2341 | 0.2341 | 0.0764 | 0.0449 |

: Selection probabilities with noise scale 2 for five candidates. {#tbl-sel-gumbel}

The Gumbel frequencies are within 0.0008 of the exponential mechanism. That is about the sampling error of a million draws: a frequency estimated from $n$ draws has standard deviation $\sqrt{p(1 - p)/n}$, here $\sqrt{0.39\times0.61/10^6} \approx 0.0005$. The Laplace noisy max is further away. Measured by the total variation distance, half the sum of the absolute differences between the two rows, which is the largest difference in probability the two can assign to any set of candidates, it is at distance 0.0216 from the exponential mechanism. Here that is exactly the extra $0.4105 - 0.3889 = 0.0216$ it puts on the top candidate: it favours the top candidate more and the bottom ones less.

### Check yourself

- How do you sample from the exponential mechanism over $10^7$ candidates without computing a normalizer? *(Draw one Gumbel per candidate, add it to $\varepsilon u/(2\Delta)$, and take the argmax. It is one pass over the scores.)*
- Does adding Gumbel noise and releasing the top 3 cost $\varepsilon$ or $3\varepsilon$? *($3\varepsilon$ under basic composition: it has the distribution of three exponential mechanisms run one after the other.)*

## The sparse vector technique {#sec-sel-svt}

### Context

Now the candidates arrive one at a time and you only care about the first one that is good enough. You monitor a thousand statistics and want to know which one first crosses an alarm level. You try hyperparameter settings until one reaches a target validation accuracy. Most queries in such a stream are uninteresting: their answer is "below". Answering each one with the Laplace mechanism costs $\varepsilon$ per query, so a thousand queries would cost $1000\varepsilon$ under basic composition. If all the queries were known in advance, you would not need a new tool: report-noisy-max or the Gumbel top-$c$ of @sec-sel-gumbel would pick the largest ones at a cost that grows only like $\ln k$ in accuracy. The difficulty is that the queries arrive one at a time, and you must answer each before seeing the next. The **sparse vector technique** (Dwork, Naor, Reingold, Rothblum & Vadhan 2009) compares each query with a single noisy threshold and pays only for the one that is above. Its basic form, **AboveThreshold**, is in Dwork & Roth (2014), Algorithm 1.

### Statement

AboveThreshold adds noise to the threshold once, adds fresh noise to each query, and stops at the first noisy query that reaches the noisy threshold. It writes $\bot$ ("below") for each query that does not and $\top$ ("above") for the one that does.

**AboveThreshold.** Input: a dataset $D$, a stream of queries $f_1, f_2, \dots$, each with sensitivity $\Delta$ (they may be chosen adaptively), a threshold $T$, and budgets $\varepsilon_1, \varepsilon_2 > 0$.

1. Draw the threshold noise $\rho \sim \mathrm{Lap}(\Delta/\varepsilon_1)$ once.
2. For each query $i = 1, 2, \dots$: draw fresh noise $\nu_i \sim \mathrm{Lap}(2\Delta/\varepsilon_2)$. If $f_i(D) + \nu_i \ge T + \rho$, output $\top$ and **halt**. Otherwise output $\bot$ and continue.

The output is the sequence $\bot^{k-1}\top$ ($k - 1$ answers "below", then one "above"), or $\bot^m$ if all $m$ queries are below; equivalently, the index $k$ of the first $\top$.

1. **Privacy** (Dwork & Roth Thm 3.23 for $\varepsilon_1 = \varepsilon_2 = \varepsilon/2$; Lyu, Su & Li 2017, Algorithm 1 with $c = 1$, for a general split). AboveThreshold is $(\varepsilon_1 + \varepsilon_2)$-DP, for any number of queries.
2. **Monotone queries** (Lyu, Su & Li 2017). If the queries are monotone, as counts under add/remove are, $\nu_i \sim \mathrm{Lap}(\Delta/\varepsilon_2)$ suffices.
3. **Accuracy** (Dwork & Roth Thm 3.24). With $\varepsilon_1 = \varepsilon_2 = \varepsilon/2$, that is $\rho \sim \mathrm{Lap}(2\Delta/\varepsilon)$ and $\nu_i \sim \mathrm{Lap}(4\Delta/\varepsilon)$, run AboveThreshold on a stream of at most $k$ queries, where $k$ is fixed before the run starts. The noisy comparisons can be wrong only for queries close to the threshold, and the margin $\alpha$ of "close" is the size the noises stay below with high probability. With probability at least $1 - \beta$, every $\bot$ query has $f_i(D) < T + \alpha$, and the $\top$ query has $f_k(D) \ge T - \alpha$, where
$$\alpha = \frac{8\Delta\big(\ln k + \ln(2/\beta)\big)}{\varepsilon}.$$ {#eq-sel-svt-alpha}
4. **$c$ answers above** (Dwork & Roth's algorithm Sparse, Thm 3.26). To find the first $c$ queries above the threshold, restart AboveThreshold with fresh threshold noise after each $\top$ and halt after the $c$-th. Give each run the budget $\varepsilon/c$; by basic composition the whole is $\varepsilon$-DP. Each run must also fail with probability at most $\beta/c$, so that all $c$ runs succeed together with probability $1 - \beta$. Both changes enter @eq-sel-svt-alpha, and the margin becomes
$$\alpha_c = \frac{8c\Delta\big(\ln k + \ln(2c/\beta)\big)}{\varepsilon}.$$ {#eq-sel-sparse-alpha}
5. **$c$ answers above, with $\delta > 0$.** Give each run instead the largest budget $\varepsilon_0$ whose $c$-fold advanced composition ([Advanced and optimal composition](04-composition.md)) is at most $\varepsilon$ at slack $\delta$. The whole is then $(\varepsilon, \delta)$-DP, and the margin is @eq-sel-svt-alpha with $\varepsilon_0$ in place of $\varepsilon$ and $\beta/c$ in place of $\beta$. Since $\varepsilon_0$ is about $\varepsilon/\sqrt{2c\ln(1/\delta)}$, the margin grows like $\sqrt c$ instead of $c$.

### Derivation

**Privacy.** The output "stop at query $k$" says two things: the noisy threshold sits above every earlier noisy answer, and it sits at or below the $k$-th noisy answer. Moving from $D$ to $D'$ moves those noisy answers. The proof undoes that move by sliding the two noises that matter. It slides the threshold noise $\rho$ by however much the largest earlier answer moved, so that the threshold keeps its place relative to all the earlier answers at once; by Fact 4 that is at most $\Delta$, however many earlier queries there are. Then it slides the $k$-th query's noise $\nu_k$ so that query $k$ keeps its place relative to the slid threshold, which takes at most $2\Delta$. Each slide is paid for by the Laplace density, as in Fact 3.

Fix neighbours $D \simeq D'$. Write $\nu = \nu_i$ for the noise of a single query and $g(D) = \max_{i < k}\big(f_i(D) + \nu_i\big)$ for the largest noisy answer before query $k$.

1. **Fix the noise of the earlier queries.** Consider the output "stop at query $k$". Fix $\nu_1, \dots, \nu_{k-1}$. By Fact 2 it is enough to bound the conditional probability, because these noises do not depend on the data. (With adaptive queries, the queries $f_1, \dots, f_k$ are fixed too: they can depend only on earlier answers, which on this output are all $\bot$.)
2. **Write the output as two conditions on $(\rho, \nu_k)$.** All earlier queries say $\bot$ exactly when even the largest of them is below the noisy threshold, $g(D) < T + \rho$. Query $k$ says $\top$ when $f_k(D) + \nu_k \ge T + \rho$. The probability of the output is the probability that both hold, which is the integral of the two noise densities $p_\rho$ and $p_\nu$ over the region where both indicators are 1:
$$\Pr[\text{stop at } k \mid D] = \iint p_\rho(r)\,p_\nu(v)\,\mathbf 1\big[g(D) < T + r\big]\,\mathbf 1\big[f_k(D) + v \ge T + r\big]\,dr\,dv.$$
3. **Shift both noises.** Substitute $r = r' + g(D) - g(D')$ and $v = v' + \big(g(D) - g(D')\big) - \big(f_k(D) - f_k(D')\big)$. Then $g(D) < T + r$ becomes $g(D') < T + r'$, and $f_k(D) + v \ge T + r$ becomes $f_k(D') + v' \ge T + r'$ (substitute and cancel the terms that appear on both sides). The two indicators are exactly those for $D'$ in the variables $(r', v')$. The substitution is a translation, so $dr\,dv = dr'\,dv'$. What has changed is only the densities, which are now evaluated at shifted points: $p_\rho(r' + g(D) - g(D'))$ and $p_\nu(v' + \text{shift of } v)$.
4. **Size of the shifts.** $g$ is a maximum of numbers $f_i + \nu_i$ that each move by at most $\Delta$ (the $\nu_i$ are fixed), so $|g(D) - g(D')| \le \Delta$ by Fact 4, whatever $k$ is. The shift of $v$ is a difference of two terms of size at most $\Delta$, so at most $2\Delta$.
5. **Pay for the shifts.** The Laplace density satisfies $p(x) \le e^{|x - x'|/b}p(x')$, as shown in the proof of Fact 3. The threshold noise has $b = \Delta/\varepsilon_1$ and shift at most $\Delta$: a factor $e^{\varepsilon_1}$. The query noise has $b = 2\Delta/\varepsilon_2$ and shift at most $2\Delta$: a factor $e^{\varepsilon_2}$. Pulling both factors out of the integral leaves exactly the integral for $D'$, so $\Pr[\text{stop at } k \mid D, \nu_{<k}] \le e^{\varepsilon_1 + \varepsilon_2}\Pr[\text{stop at } k \mid D', \nu_{<k}]$.
6. **The output "all below".** Only the first indicator is present, with the maximum over all $m$ queries. Shifting $\rho$ alone costs $e^{\varepsilon_1}$.
7. **Collect.** Fact 2 removes the conditioning, and Fact 1 gives $(\varepsilon_1 + \varepsilon_2)$-DP.

*Monotone queries.* If all $f_i$ move the same way by at most $\Delta$, then $g$ and $f_k$ move the same way, each by an amount in $[0, \Delta]$ or each in $[-\Delta, 0]$ (Fact 4). The shift of $v$ in step 3 is their difference, which is then at most $\Delta$ in size. A query noise with $b = \Delta/\varepsilon_2$ then costs $e^{\varepsilon_2}$.

**Sparse.** Each run is an AboveThreshold whose queries may depend on the earlier runs' outputs, so the $c$ runs form an adaptive composition. Basic composition of $c$ runs at $\varepsilon/c$ gives $\varepsilon$, and advanced composition at $\varepsilon_0$ gives $(\varepsilon, \delta)$. For accuracy, apply the AboveThreshold bound to each run with failure probability $\beta/c$ and take a union bound over the $c$ runs: $\ln(2/\beta)$ becomes $\ln(2c/\beta)$, and $\varepsilon$ becomes the budget of one run. That gives @eq-sel-sparse-alpha.

**Accuracy.** A comparison of noisy numbers gives the same answer as the comparison of true numbers whenever the noises are smaller than the distance between the true numbers. So the proof finds a margin that all the noises stay below with probability $1 - \beta$, splitting the failure probability $\beta$ equally between the threshold noise and the query noises. Let $\varepsilon_1 = \varepsilon_2 = \varepsilon/2$.

1. **Threshold noise.** The Laplace tail $\Pr[|\rho| \ge x] = e^{-x/b}$ with $b = 2\Delta/\varepsilon$ and $x = \alpha/2$ gives $\Pr[|\rho| \ge \alpha/2] = e^{-(\alpha/2)/(2\Delta/\varepsilon)} = e^{-\alpha\varepsilon/(4\Delta)}$.
2. **Query noise.** Each $\nu_i$ has scale $4\Delta/\varepsilon$, so $\Pr[|\nu_i| \ge \alpha/2] = e^{-\alpha\varepsilon/(8\Delta)}$. By the union bound over $k$ queries, the probability that any of them is that large is $\Pr[\max_i|\nu_i| \ge \alpha/2] \le k\,e^{-\alpha\varepsilon/(8\Delta)}$.
3. **Both small.** Setting each to at most $\beta/2$ needs $\alpha \ge 8\Delta\ln(2k/\beta)/\varepsilon$ for the second (the first then holds too, since its exponent is twice as large and it has no factor $k$), which is @eq-sel-svt-alpha because $\ln(2k/\beta) = \ln k + \ln(2/\beta)$. The union bound runs over a list of $k$ noises fixed in advance. That is why $k$ must be the length of the stream, chosen before the run. The index at which the run happens to stop depends on the noises themselves, and a union bound over a random number of terms proves nothing.
4. **Read off the answers.** If $|\rho| < \alpha/2$ and every $|\nu_i| < \alpha/2$, the noises together can move a comparison by less than $\alpha$. A $\bot$ means $f_i + \nu_i < T + \rho$, so $f_i < T + \rho - \nu_i < T + \alpha$; the $\top$ means $f_k + \nu_k \ge T + \rho$, so $f_k > T - \alpha$.

### Interpretation

The $\bot$ answers are free because the derivation pays for all of them at once, in step 4, through one maximum $g$ and one shift of the threshold noise. Fact 4 makes that shift at most $\Delta$ whether there are 3 or 3 million queries below the threshold. The single $\top$ is paid for separately, by its own noise.

The run must halt because that payment covers only one $\top$. After the first $\top$, the remaining answers would still compare against the same $\rho$. The proof's shift of $\rho$ was chosen to keep the $\bot$s consistent; each further $\top$ would need another shift of size up to $2\Delta$ in its own noise, and so another factor $e^{\varepsilon_2}$. With $c$ answers above, Lyu, Su & Li's Algorithm 1 keeps one threshold noise and uses $\nu_i \sim \mathrm{Lap}(2c\Delta/\varepsilon_2)$, so that the $c$ shifts together cost $e^{\varepsilon_2}$ (@sec-apx-mech-svt in the appendices proves it by extending the steps above). Without any limit on $c$ the cost is unbounded (@sec-sel-svt-bugs).

AboveThreshold answers a yes/no question. It outputs positions, not values. To also release the value of the query that fired, spend extra budget on a fresh Laplace answer to that query.

The split between $\varepsilon_1$ and $\varepsilon_2$ matters for accuracy, not privacy. Any $\varepsilon_1 + \varepsilon_2 = \varepsilon$ is $\varepsilon$-DP. The equal split is simple but not optimal: [Exercise 4](../../book-exercises/part03-dp-foundations/05-selection.md#sec-sol-selection-4) finds the split that minimizes $\alpha$, and Lyu, Su & Li recommend $\varepsilon_1 : \varepsilon_2 = 1 : (2c)^{2/3}$ to maximize the probability of correct answers.

The proof is tight in the limit. On the neighbouring pair of the next section, the largest privacy loss of AboveThreshold, at the output "stop at query $m + 1$", approaches $\varepsilon$ as $m$ grows: it is 0.994011 at $m = 10$, 0.999981 at $m = 20$ and 1.000000 to six places at $m = 40$ (@tbl-sel-svt-loss). Both shifts in step 4 approach their maximum, but never quite reach it.

@fig-sel-svt-path shows one run.

![One run of AboveThreshold with $\varepsilon = 1$ on 30 queries whose true answers rise from 8 to 22.5, with threshold $T = 20$. Each query's noisy answer is compared with the same noisy threshold. The first one above it is reported and the run halts; later queries are never looked at.](figures/select_svt_path.svg){#fig-sel-svt-path}

### Example: an alarm among 1,000 statistics

You monitor $k = 1000$ counting queries ($\Delta = 1$) with threshold $T = 500$ at $\varepsilon = 1$. With $\beta = 0.05$, @eq-sel-svt-alpha gives $\alpha = 8(\ln 1000 + \ln 40) = 84.77$, since $2/\beta = 40$. So with probability at least 0.95, no query below $500 - 84.77 = 415.23$ fires (a $\top$ needs $f_k \ge T - \alpha$) and no query above $500 + 84.77 = 584.77$ is skipped (a $\bot$ needs $f_i < T + \alpha$). In the script's stream, every query has count 400 except query 700, which has 600. The counts of 400 are below 415.23 and cannot fire, and the count of 600 is above 584.77 and cannot be skipped, so the guarantee says the output is query 700 with probability at least 0.95; in 10,000 simulated runs it was query 700 every time.

The union bound behind $\alpha$ is nearly tight. The margin that the noises actually needed is $2\max(|\rho|, \max_i|\nu_i|)$, the smallest $\alpha$ for which every noise is below $\alpha/2$. Its 95% quantile over 10,000 simulated runs is 78.59, against 84.77. The alternative, answering all 1,000 queries with the Laplace mechanism at $\varepsilon/1000$ each, adds noise of scale 1,000 to counts of a few hundred.

Now suppose you want the first $c$ alarms, not just one, from the same 1,000 statistics at $\varepsilon = 1$ and $\beta = 0.05$. With basic composition, @eq-sel-sparse-alpha gives a margin of 1,032 for $c = 10$ and 12,161 for $c = 100$. With advanced composition at $\delta = 10^{-6}$, each run of $c = 10$ gets $\varepsilon_0 = 0.0591$ and a margin of 1,746, which is *worse* than basic composition. With $c = 100$ each run gets $\varepsilon_0 = 0.0187$ and the margin falls to 6,506, about half the basic one. This is the crossover of [Advanced and optimal composition](04-composition.md) again: the $\sqrt c$ route pays only once $c$ is large compared with $2\ln(1/\delta)$, which is about 28 here.

### Check yourself

- AboveThreshold answers 999 queries with $\bot$ and halts at the 1000th. How much of its budget did the 999 $\bot$ answers use? *(All of them together used $\varepsilon_1$, through the threshold noise; the $\top$ used $\varepsilon_2$. With 9 or 999,999 $\bot$ answers the cost would be the same.)*
- Why does the accuracy bound depend on $k$, while the privacy does not? *(Accuracy needs every query noise to be small at once, which costs a union bound over $k$ queries. Privacy needs only the maximum of the noisy answers to move by at most $\Delta$, which holds for any $k$.)*

## When the sparse vector technique goes wrong {#sec-sel-svt-bugs}

### Context

The sparse vector technique is easy to state and easy to break. Lyu, Su & Li (2017) surveyed published variants and found that several were not differentially private at all, although each looked like a harmless optimization of AboveThreshold. Each bug removes one ingredient that a step of the proof above relies on. Seeing which step fails is the fastest way to review an implementation.

### Statement

@tbl-sel-svt-variants lists the variants analysed by Lyu, Su & Li (2017) and what each changes.

| Variant (as numbered by Lyu, Su & Li) | What it changes | Privacy shown by Lyu, Su & Li |
| --- | --- | --- |
| Algorithm 1 (theirs); Algorithm 2 (Dwork & Roth) | Correct, with $c$ answers above | $\varepsilon$ |
| Algorithm 3 (Roth's 2011 lecture notes) | Outputs the noisy value $f_i + \nu_i$ for each $\top$ | $\infty$ |
| Algorithm 4 (Lee & Clifton 2014) | Query noise does not grow with $c$ | $\frac{1 + 6c}{4}\varepsilon$ |
| Algorithm 5 (Stoddard et al. 2014) | No noise on the queries | $\infty$ |
| Algorithm 6 (Chen et al. 2015) | Never halts | $\infty$ |

: Variants of the sparse vector technique and their actual privacy, from Lyu, Su & Li (2017). {#tbl-sel-svt-variants}

This section proves the last row for the **no-halt variant** of AboveThreshold: the same noise ($\rho \sim \mathrm{Lap}(2\Delta/\varepsilon)$, $\nu_i \sim \mathrm{Lap}(4\Delta/\varepsilon)$), but every query is answered and the run never stops. To show a mechanism is not private, it is enough to find one pair of neighbours and one output whose probability ratio is too large. The pair below is built so that every answer of the output $\bot^m\top^m$ ($m$ answers "below", then $m$ answers "above") is a little more likely under $D$: on $D$ the first $m$ queries are low, which makes their $\bot$ easier, and the last $m$ are high, which makes their $\top$ easier. Each of the $2m$ answers contributes a small factor, and without a halt nothing stops the factors from multiplying. Take $\Delta = 1$, $2m$ queries, and neighbours with
$$f(D) = (\underbrace{0, \dots, 0}_m, \underbrace{1, \dots, 1}_m), \qquad f(D') = (\underbrace{1, \dots, 1}_m, \underbrace{0, \dots, 0}_m).$$
Each query changes by 1 between the two, so each has sensitivity 1. For the output $\bot^m\top^m$, the privacy loss grows at least linearly in $m$:
$$\ln\frac{\Pr[\bot^m\top^m \mid D]}{\Pr[\bot^m\top^m \mid D']} \ge \frac{m\varepsilon}{4}.$$ {#eq-sel-no-halt}
The privacy loss grows without bound in $m$, so the variant is not $\varepsilon'$-DP for any $\varepsilon'$.

### Derivation

The answers are not independent, because they all compare against the same noisy threshold. But once the threshold is fixed they are, since each then depends only on its own fresh noise. So the proof fixes the threshold, multiplies the per-query probabilities, shows that the ratio is at least $e^{1/b}$ per pair of queries for every threshold, and averages over the threshold at the end. Let $b = 4/\varepsilon$, the query-noise scale with $\Delta = 1$, and let $F$ and $S = 1 - F$ be the CDF and the survival function of $\mathrm{Lap}(b)$: $F(x) = \Pr[\nu \le x]$ and $S(x) = \Pr[\nu > x]$.

1. **Fix the threshold.** Condition on $T + \rho = t$. The query noises are independent, so the answers are independent. A query with value $f$ says $\bot$ when $f + \nu < t$, that is $\nu < t - f$, with probability $F(t - f)$, and $\top$ with probability $S(t - f)$.
2. **The two conditional probabilities.** Under $D$ the output needs $m$ queries of value 0 to say $\bot$ and $m$ of value 1 to say $\top$; under $D'$ it needs $m$ of value 1 to say $\bot$ and $m$ of value 0 to say $\top$. Multiplying,
$$a_D(t) = F(t)^m\,S(t - 1)^m, \qquad a_{D'}(t) = F(t - 1)^m\,S(t)^m.$$
3. **Their ratio is a power.** $a_D(t)/a_{D'}(t) = h(t)^m$ with $h(t) = \frac{F(t)}{F(t - 1)}\cdot\frac{S(t - 1)}{S(t)}$. Both factors are at least 1, because $F$ increases and $S$ decreases: a query of value 0 is more likely to say $\bot$ than one of value 1, and a query of value 1 is more likely to say $\top$.
4. **$h(t) \ge e^{1/b}$ for every $t$.** The Laplace CDF is $F(x) = \frac12e^{x/b}$ for $x \le 0$ and the survival function is $S(x) = \frac12e^{-x/b}$ for $x \ge 0$; on each of those half-lines, moving $x$ by 1 changes the value by exactly the factor $e^{1/b}$. There are three cases, according to where $t$ sits. If $t \le 0$, both $t$ and $t - 1$ are on the half-line where $F$ is exponential, so $F(t)/F(t - 1) = e^{1/b}$, and the second factor is at least 1. If $t \ge 1$, both $t - 1$ and $t$ are on the half-line where $S$ is exponential, so $S(t - 1)/S(t) = e^{1/b}$, and the first factor is at least 1. If $0 < t < 1$, the two points straddle 0, and each factor gets part of the shift: $F(t) \ge \frac12$ and $F(t - 1) = \frac12e^{(t - 1)/b}$, so the first factor is at least $e^{(1 - t)/b}$; likewise $S(t - 1) \ge \frac12$ and $S(t) = \frac12e^{-t/b}$, so the second is at least $e^{t/b}$. The product is at least $e^{(1 - t)/b + t/b} = e^{1/b}$.
5. **Average over the threshold.** The threshold noise has the same distribution under $D$ and $D'$, so $\Pr[\bot^m\top^m \mid D] = \mathbb E[a_D(T + \rho)] \ge e^{m/b}\,\mathbb E[a_{D'}(T + \rho)] = e^{m\varepsilon/4}\Pr[\bot^m\top^m \mid D']$, using $h^m \ge e^{m/b}$ at every threshold and $m/b = m\varepsilon/4$.

**Why AboveThreshold's proof does not apply.** Every $\top$ in the second half is a new comparison with the same threshold. Step 3 of the AboveThreshold proof can shift $\rho$ once to fix all the $\bot$ answers, but each $\top$ needs its own noise shifted by up to $2\Delta$, at a cost of $e^{\varepsilon/2}$ each with this noise scale. The halt is what limits the number of $\top$ answers to one.

**No query noise (Algorithm 5).** Without $\nu_i$ even the halting version fails, with two queries: $f(D) = (0, 1)$ and $f(D') = (1, 0)$. Without query noise, a query with value $f$ says $\top$ exactly when $f \ge T + \rho$, a deterministic comparison with the noisy threshold. The output $\bot\top$ needs $0 < T + \rho \le 1$ on $D$, which has positive probability, and $1 < T + \rho \le 0$ on $D'$, which is impossible. A probability ratio of $\infty$ cannot be bounded by any $e^{\varepsilon}$. [Exercise 5](../../book-exercises/part03-dp-foundations/05-selection.md#sec-sol-selection-5) computes the first probability.

### Interpretation

Each ingredient of AboveThreshold protects one step of its proof. The fresh noise $\nu_k$ pays for the $\top$ (step 5 of the AboveThreshold proof): remove it and the $\top$ reveals a deterministic comparison, as in Algorithm 5. The halt limits the $\top$s to one: remove it and the cost per $\top$ accumulates, as in the no-halt variant. Publishing $f_k + \nu_k$ releases the very noise that the proof integrates over, as in Algorithm 3.

The bug is in the accounting, not in the noise. The no-halt variant adds exactly as much noise per query as AboveThreshold. A reviewer who checks only noise scales will not find it.

Empirical audits catch it, and you do not need the exact probabilities to run one. If a mechanism is $\varepsilon$-DP, then every event, including "the test says $D$", is at most $e^\varepsilon$ times as likely under $D$ as under $D'$. So any test whose probability of firing under $D$ is more than $e^\varepsilon$ times its probability under $D'$ proves the mechanism is not $\varepsilon$-DP, and the log of that ratio is a lower bound on the true $\varepsilon$. This is the hypothesis-testing form of DP in [The definition of differential privacy](01-definition.md). A simple test, "are there more $\top$ answers in the second half than in the first?", already exceeds what $\varepsilon = 1$ allows once $m$ is a few dozen (@fig-sel-svt-buggy, right).

![Left: privacy loss on the neighbouring pair $f(D) = (0^m, 1^m)$, $f(D') = (1^m, 0^m)$ with $\varepsilon = 1$, computed exactly by integrating over the threshold noise. For the no-halt variant, the loss of the output $\bot^m\top^m$ grows linearly in $m$, above the lower bound $m\varepsilon/4$. For AboveThreshold, the largest loss over all outputs levels off at $\varepsilon$. Right: the no-halt variant under a simple test, the probability that the second half has more $\top$ answers than the first. Once the probability under $D$ exceeds $e^\varepsilon$ times the probability under $D'$ (dotted), no $\varepsilon$-DP mechanism could produce it.](figures/select_svt_buggy.svg){#fig-sel-svt-buggy}

### Example: the loss, computed and simulated

@tbl-sel-svt-loss gives the script's exact values for $\varepsilon = 1$ and $T = 0$. For AboveThreshold the table takes the largest privacy loss over all its possible outputs on this pair; for the no-halt variant it takes the loss of the one output $\bot^m\top^m$.

| $m$ | 1 | 5 | 10 | 20 | 40 |
| --- | --- | --- | --- | --- | --- |
| AboveThreshold, largest loss over outputs | 0.3958 | 0.8998 | 0.9940 | 1.0000 | 1.0000 |
| No halt, loss of $\bot^m\top^m$ | 0.3958 | 2.0985 | 4.3220 | 8.8741 | 18.1246 |
| Lower bound $m\varepsilon/4$ | 0.25 | 1.25 | 2.50 | 5.00 | 10.00 |

: Exact privacy loss of AboveThreshold and of its no-halt variant on the counterexample, with $\varepsilon = 1$. {#tbl-sel-svt-loss}

At $m = 1$ the two mechanisms coincide, because $\bot\top$ is a complete AboveThreshold output: AboveThreshold halts after the $\top$ anyway. From then on the no-halt loss grows by about $18.1246/40 = 0.45$ nats (units of natural logarithm) for each pair of queries added, while AboveThreshold never exceeds $\varepsilon$. The lower bound is loose by a factor of a little under 2 (18.1246 against 10.00 at $m = 40$), because step 4 of the derivation used a lower bound on $h(t)$ that holds at every threshold, and at every threshold $h$ is in fact larger.

The integrals are checked by simulation at $m = 2$: the exact probability of $\bot\bot\top\top$ under $D$ is 0.0669, and 400,000 runs give 0.0671. The sign test of @fig-sel-svt-buggy, run 200,000 times per dataset in the script (the figure uses 100,000 runs per point), gives
$\Pr[\text{more }\top\text{ in the second half}]$ of 0.5529 under $D$ and 0.2552 under $D'$ at $m = 10$, a log-ratio of 0.77; 0.7399 and 0.1832 at $m = 40$, a log-ratio of 1.40; and 0.9151 and 0.0659 at $m = 160$, a log-ratio of 2.63. From $m = 40$ on, the log-ratio is above 1, so this test alone proves that the variant is not 1-DP.

### Check yourself

- A colleague's implementation of AboveThreshold reuses a single noise draw $\nu$ for all queries to "save randomness". Which step of the proof fails? *(Step 1: fixing $\nu_1, \dots, \nu_{k-1}$ would also fix $\nu_k$, so the $\top$ could not be paid for by shifting its own noise. With one shared $\nu$, every answer compares $f_i$ with the same random level $T + \rho - \nu$, which is Algorithm 5's bug.)*
- Is releasing the index of the first $\top$ different from releasing the whole sequence $\bot^{k-1}\top$? *(No. Each determines the other. The bug in the no-halt variant is answering after the $\top$, not the output format.)*

## Code: check every mechanism by computation and simulation

The script [code/selection.py](../../code/part03-dp-foundations/selection.py) computes every number in this chapter and draws every figure. The exponential mechanism's probabilities are exact sums. Laplace report-noisy-max and the sparse vector technique are integrated exactly on a fine grid over one noise variable (the winner's noise, or the threshold noise), and simulations check the integrals. From the repository root:

```bash
uv run python code/part03-dp-foundations/selection.py
```

**Output** (seed 0), summarized in @tbl-sel-code:

| Experiment | Result |
| --- | --- |
| Laplace threshold shift, $b = s = 1$ | Ratio 1.6321 at $t = 0$, 2.5357 at $t = 0.7$, 2.7183 at $t = 1$ and $t = 2$ |
| Exponential mechanism on the vote, $\varepsilon = 1$ | @tbl-sel-em-votes; normalizer ratio 1.1790; largest loss 0.3353 (monotone weights: 0.6585) |
| Exponential mechanism on 50 models | @tbl-sel-em-models; $\Pr[\text{gap} \ge 13.82] = 9.12\times10^{-4}$ against $e^{-3} = 0.0498$; tied profile at gap 20: 0.0022 against 0.0023 |
| Laplace report-noisy-max, 50 models, $\varepsilon = 1$ | $b = 2$: $\Pr[\text{best}]$ 0.4241, expected gap 1.40 (simulated 1.39). $b = 1$: 0.6463 and 0.54 |
| Laplace report-noisy-max on the vote, $b = 1$ | Largest loss 0.6674 ($b = 2$: 0.3626) |
| Gumbel-max, five candidates | Within 0.0008 of the exponential mechanism over $10^6$ runs; Laplace at total variation 0.0216 |
| AboveThreshold, $k = 1000$, $\beta = 0.05$ | $\alpha = 84.77$; simulated 95% quantile 78.59; alarm stream correct in 10,000 of 10,000 runs |
| AboveThreshold against no halt | @tbl-sel-svt-loss; sign test log-ratios 0.77, 1.40, 2.63 at $m = 10, 40, 160$ |
| AboveThreshold, worst loss to six places | 0.994011, 0.999981, 1.000000 at $m = 10, 20, 40$ |
| Sparse, $k = 1000$, $\beta = 0.05$ | Margin 1,032 ($c = 10$) and 12,161 ($c = 100$); with advanced composition at $\delta = 10^{-6}$, 1,746 and 6,506 |
| Threshold learner, $n = 1000$ | Excess error 0.0024 ($\varepsilon = 1$), 0.0206 ($\varepsilon = 0.1$), 0.0008 non-private; records for $\alpha = 0.05$: 33,913 (sampling), 793 (privacy at $\varepsilon = 1$) |
| Private median of 1,000 gradient norms | Median 0.9804; mean distance 0.0013 ($\varepsilon = 1$) and 0.0157 ($\varepsilon = 0.1$); imbalance bound 26.4 and 264.5 |

: The script's output with the default seed. {#tbl-sel-code}

[tests/test_selection.py](../../tests/test_selection.py) checks that the output still matches.

Four things stand out. First, the privacy proofs are right, and some are tight: on every neighbouring pair the script tries, the largest privacy loss stays below $\varepsilon$, and AboveThreshold comes within $2\times10^{-5}$ of it. Second, the utility bounds are right but often loose. The exponential mechanism's tail bound is tight only when all runners-up are tied; for spread-out scores it overestimates the probability of a large gap by orders of magnitude, because it treats every bad candidate as if it were barely bad. Third, Gumbel noise is the exponential mechanism, and Laplace noise is a close relative; neither of the two private argmaxes dominates the other. Fourth, one missing line breaks the sparse vector technique. Without the halt, the privacy loss grows linearly in the number of queries, and a two-line test detects it.

## References

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), §3.3 (report-noisy-max, Claim 3.9), §3.4 (the exponential mechanism, Theorems 3.10 and 3.11) and §3.6 (the sparse vector technique: AboveThreshold, Theorems 3.23 and 3.24, and the algorithms Sparse and NumericSparse, with Theorem 3.26 for their accuracy). The proofs above follow theirs. Their version of Sparse with $\delta > 0$ uses a closed-form per-run budget where this chapter solves for the largest one.
- McSherry & Talwar, "Mechanism Design via Differential Privacy" (FOCS 2007). The exponential mechanism, with the continuous version and its use in auctions.
- Lyu, Su & Li, "Understanding the Sparse Vector Technique for Differential Privacy" (Proceedings of the VLDB Endowment, 2017). The general algorithm with $c$ answers above and a budget split, the monotone improvement, the survey of incorrect variants in @tbl-sel-svt-variants, and the optimal split. The best single source on the technique.
- Dwork, Naor, Reingold, Rothblum & Vadhan, "On the Complexity of Differentially Private Data Release" (STOC 2009). Where the sparse vector idea first appears.
- Kasiviswanathan, Lee, Nissim, Raskhodnikova & Smith, "What Can We Learn Privately?" (FOCS 2008; SIAM Journal on Computing 2011). The private learner of @eq-sel-learner: any finite class is learnable privately with $O(\ln|C|)$ records.
- Alon, Livni, Malliaris & Moran, "Private PAC Learning Implies Finite Littlestone Dimension" (STOC 2019). Why the class must be finite: thresholds over an infinite domain cannot be learned privately.
- Smith, "Privacy-Preserving Statistical Estimation with Optimal Convergence Rates" (STOC 2011), and Gillenwater, Joseph & Kulesza, "Differentially Private Quantiles" (ICML 2021). The exponential mechanism for medians and quantiles over an interval, sampled exactly piece by piece.
- Dwork, Feldman, Hardt, Pitassi, Reingold & Roth, "The Reusable Holdout: Preserving Validity in Adaptive Data Analysis" (Science 2015). Thresholdout, an application of the sparse vector technique to validation sets reused across many experiments, treated in [Adaptive data analysis](../part05-dp-ml/11-adaptive-data-analysis.md).
- Durfee & Rogers, "Practical Differentially Private Top-k Selection with Pay-what-you-get Composition" (NeurIPS 2019). The one-shot Gumbel top-$c$ and its equivalence to repeated exponential mechanisms.
- McKenna & Sheldon, "Permute-and-Flip: A New Mechanism for Differentially Private Selection" (NeurIPS 2020), and Ding et al., "The Permute-and-Flip Mechanism is Identical to Report-Noisy-Max with Exponential Noise" (2021).
- Maddison, Tarlow & Minka, "A* Sampling" (NeurIPS 2014): the Gumbel-max trick in machine learning, with its history going back to Gumbel and to Yellott (1977).
- Near & Abuah, *Programming Differential Privacy*, the chapters on the exponential mechanism and the sparse vector technique: short implementations alongside Dwork & Roth.

## Cheat sheet

$\mathcal R$ has $k$ candidates with scores $u(D, r)$ of sensitivity $\Delta$; $\mathrm{OPT} = \max_r u(D, r)$. Monotone scores all move the same way between neighbours. @tbl-sel-cheat collects the mechanisms.

| Mechanism | Output | Privacy | Accuracy |
| --- | --- | --- | --- |
| Exponential mechanism | $r$ with probability $\propto e^{\varepsilon u(D, r)/(2\Delta)}$ | $\varepsilon$; $e^{\varepsilon u/\Delta}$ for monotone scores or a data-independent normalizer | Gap $\le \frac{2\Delta}{\varepsilon}(\ln k + t)$ w.p. $1 - e^{-t}$; mean gap $\le \frac{2\Delta}{\varepsilon}(\ln k + 1)$ |
| Private learner | Exponential mechanism over a finite class, score $-$(training mistakes) | $\varepsilon$ | Excess error $\le 2\sqrt{\ln(2\lvert C\rvert/\beta)/(2n)} + 2\ln(\lvert C\rvert/\beta)/(\varepsilon n)$ w.p. $1 - 2\beta$ |
| Private median on $[0, B]$ | Piece $j$ w.p. $\propto L_je^{-\varepsilon\lvert 2j - n\rvert/2}$, then uniform | $\varepsilon$ | Imbalance $\le \frac2\varepsilon(\ln(B/L^\star) + t)$ w.p. $1 - e^{-t}$ |
| Report-noisy-max, Laplace | $\arg\max_r(u + \mathrm{Lap}(2\Delta/\varepsilon))$ | $\varepsilon$; $\mathrm{Lap}(\Delta/\varepsilon)$ for monotone scores | Gap $\le 2b\ln(k/\beta)$ w.p. $1 - \beta$ |
| Report-noisy-max, Gumbel | $\arg\max_r(u + \mathrm{Gumbel}(2\Delta/\varepsilon))$ | Identical to the exponential mechanism | Same |
| AboveThreshold | Index of the first $f_i + \mathrm{Lap}(2\Delta/\varepsilon_2) \ge T + \mathrm{Lap}(\Delta/\varepsilon_1)$, then halt | $\varepsilon_1 + \varepsilon_2$, any number of $\bot$ | $\alpha = 8\Delta(\ln k + \ln\frac2\beta)/\varepsilon$ for the equal split, $k$ the stream length fixed in advance |
| Sparse ($c$ answers above) | AboveThreshold restarted $c$ times at $\varepsilon/c$ | $\varepsilon$ by basic composition | $\alpha_c = 8c\Delta(\ln k + \ln\frac{2c}\beta)/\varepsilon$ |
| Sparse with $\delta > 0$ | $c$ runs at the $\varepsilon_0$ advanced composition allows | $(\varepsilon, \delta)$ | $\alpha$ with $\varepsilon_0 \approx \varepsilon/\sqrt{2c\ln(1/\delta)}$ for $\varepsilon$ and $\beta/c$ for $\beta$ |

: The selection mechanisms of this chapter. {#tbl-sel-cheat}

**Proof pattern.** Fix every noise except one (Fact 2). The output happens when that noise clears a bar. The bar is built from a maximum, which moves by at most $\Delta$ however many terms it has (Fact 4). Shifting a Laplace variable by $s$ costs $e^{s/b}$ (Fact 3).

**Rules of thumb**

- Release positions, not values. Every released noisy value needs its own budget.
- The number of candidates costs $\ln k$ in accuracy and nothing in privacy.
- Use $2\Delta$ unless you have checked that the scores are monotone under your neighbouring relation; counts are monotone under add/remove, a histogram under replace is not.
- To sample the exponential mechanism, add Gumbel noise and take the argmax.
- Over a continuous range, fix the range before looking at the data and sample piece by piece; the range costs only its logarithm.
- An implementation of the sparse vector technique must draw fresh noise for every query, keep one threshold noise per run, halt after the allowed number of $\top$ answers, and never publish the noisy values it compared.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part03-dp-foundations/05-selection.md).
