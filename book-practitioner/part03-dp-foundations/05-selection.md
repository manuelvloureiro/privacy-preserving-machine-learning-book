# Selection: exponential mechanism, report-noisy-max, sparse vector

## Why this matters

Many questions about a dataset ask you to pick something rather than to compute a number. You might want the hyperparameter setting that scores best on a validation set, the most frequent word in a corpus, or the best split point for a decision stump. Or you might watch a thousand statistics and want to know which one first crosses an alarm level. In each case the answer is a name from a list, and "model 7 plus noise" is not a model.

The [Laplace and Gaussian mechanisms](03-laplace-gaussian.md) of differential privacy (DP) add noise to numbers, so they do not help directly. You could use them to release a noisy score for every candidate and then take the largest. But each noisy score is one more release, and under basic composition ([Post-processing, group privacy and composition](02-properties.md)) the privacy costs of releases add up. So the noise would have to grow with the number of candidates, often thousands of them, when you only wanted to know the winner.

This chapter shows that you never have to pay for the losers. Three mechanisms make a selection cost as much as a single query, and they all rest on one idea, which is to **release the position, not the values**. Each one publishes only which candidate won, or which query came first, and keeps secret the noisy scores behind that choice. For the exponential mechanism those noisy scores stay implicit, as the section on Gumbel noise shows. One record can change who wins only a little, however many candidates there are, so the number of candidates never enters the privacy cost. It enters only the accuracy, and only through its logarithm.

## The exponential mechanism {#sec-sel-exp}

Start with what a selection problem looks like. There is a private dataset $D$ and a list of $k$ candidates, such as models, split points or words. Each candidate has a **score** that says how good it is on the data, for example the number of validation examples a model gets right. Write $u(D, r)$ for the score of candidate $r$ on the dataset $D$.

Recall from [The definition of differential privacy](01-definition.md) that two datasets are neighbours when they differ in one person's record. There are two common ways to make that precise. Under the add/remove relation one dataset has one extra record, while under the replace relation one record has been swapped for another. The **sensitivity** $\Delta$ of the score is the largest change in any one candidate's score between two neighbours.

Simply returning the candidate with the highest score is not private. When two candidates are nearly tied, one record can tip the balance, and the output then reveals that record. So a private selection must be random, tilted toward high scores but never certain. The question is how strongly you can tilt it.

McSherry and Talwar (2007) answered with the **exponential mechanism**. It gives each candidate a weight that grows exponentially with its score, and picks each candidate with probability equal to its share of the total weight. With a privacy budget $\varepsilon$, the mechanism outputs candidate $r$ with probability
$$\Pr[M_E(D) = r] = \frac{\exp\big(\varepsilon\,u(D, r)/(2\Delta)\big)}{\sum_s\exp\big(\varepsilon\,u(D, s)/(2\Delta)\big)}.$$ {#eq-sel-em}
The sum in the denominator runs over all $k$ candidates and makes the probabilities add up to 1. That sum is the total weight, also called the **normalizer**.

The exponential is what makes the mechanism private. One record moves each score by at most $\Delta$, and the exponential turns that additive change into a multiplicative one. So each weight changes by at most a factor $e^{\varepsilon/2}$, whatever the score. The normalizer is a sum of weights, and since each of its terms changes by at most that factor, the sum does too.

The probability of a candidate is its own weight divided by the normalizer, and one record can move both the numerator and the denominator against you at once. It can raise the candidate's own score, which makes its weight larger, and lower all the other scores, which makes the normalizer smaller. Each of these two effects costs at most $e^{\varepsilon/2}$, so together they cost at most $e^\varepsilon$, and the mechanism is $\varepsilon$-DP. The 2 in the exponent is there to split the budget between the two effects, and the number of candidates appears nowhere in the argument.

The argument checks one output at a time, and that is enough whenever there are finitely many outputs. The probability of a set of outputs is the sum of their probabilities. So if each output's probability changes by at most a factor $e^\varepsilon$, so does the probability of any set. Every privacy argument in this chapter uses this shortcut.

A small vote shows the two effects at work. Six people vote among options A, B, C and D. On dataset $D$ the counts are 3, 2, 1 and 0, and its neighbour $D'$ has one more vote, for B. The score is the count, and adding or removing one vote changes one count by 1, so $\Delta = 1$ under add/remove. @tbl-sel-em-votes shows the selection probabilities at $\varepsilon = 1$.

| Option | A | B | C | D |
| --- | --- | --- | --- | --- |
| $\Pr[M_E(D) = r]$ | 0.4551 | 0.2760 | 0.1674 | 0.1015 |
| $\Pr[M_E(D') = r]$ | 0.3859 | 0.3859 | 0.1420 | 0.0861 |
| Ratio $D/D'$ | 1.1790 | 0.7151 | 1.1790 | 1.1790 |

: The exponential mechanism with $\varepsilon = 1$ on a vote and its neighbour with one extra vote for B. {#tbl-sel-em-votes}

Only B's score changed, so for A, C and D only the normalizer moved. Their ratio, 1.1790, is the ratio of the two normalizers, and it stays below its bound of $e^{1/2}$. For B the extra vote also raised its own weight, which pulls its ratio the other way, down to 0.7151. The largest privacy loss on this pair, the largest absolute log of a ratio, is 0.3353. That is only a third of $\varepsilon$, because this pair moves a single score, so only one of the two effects acts on each option. The guarantee must still cover pairs where both act at once.

### Monotone scores

The 2 can be dropped when the scores are **monotone**, which means that for every pair of neighbours, one record pushes all the scores the same way. Counts under add/remove are monotone, because adding a record can only raise each count, and removing one can only lower each count. With monotone scores the two effects pull against each other. If one record lowers the candidate's own score, it lowers its competitors' scores too, so the normalizer shrinks and wins back part of the loss. If it raises the competitors' scores, it raises the candidate's own score as well. So on any pair of neighbours, at most one of the two factors can exceed 1, and that one factor may use the whole budget. Weights $e^{\varepsilon u/\Delta}$, without the 2, let each factor reach $e^\varepsilon$ but no more, so they are also $\varepsilon$-DP.

The condition matters. For scores that are not monotone, there are pairs of neighbours on which the ratio approaches $e^{2\varepsilon}$ if you drop the 2. A histogram under replace neighbours is the common trap. Moving one record from one bin to another lowers one count and raises another, so the scores move in opposite directions, and you must keep the 2.

### What the mechanism looks like

If you have met the softmax of machine learning, you have met this mechanism. A softmax turns scores into probabilities proportional to $e^{x/\tau}$, where $x$ is a score and $\tau$ is a temperature. The exponential mechanism is a softmax of the scores at temperature $2\Delta/\varepsilon$. As $\varepsilon$ shrinks the temperature grows, every weight approaches 1, and the output becomes uniform and reveals nothing. As $\varepsilon$ grows the best score's weight dominates, and the mechanism becomes the plain argmax. @fig-sel-em-probs shows the whole range on the model-selection example of the next section.

![Left: the scores of 50 models, model $r$ getting $900 - r$ of 1,000 validation examples right. Right: the probability that the exponential mechanism selects each model, for four values of $\varepsilon$. Small $\varepsilon$ spreads the probability toward uniform; large $\varepsilon$ concentrates it on the best model.](figures/select_em_probs.svg){#fig-sel-em-probs}

Because only ratios of weights matter, adding the same constant to every score changes nothing, since the common factor cancels in the normalizer. Rescaling the score together with $\Delta$ also changes nothing. So a count of correct predictions with $\Delta = 1$ gives the same mechanism as an accuracy with $\Delta = 1/n$, where $n$ is the number of validation examples.

### The conditions behind the proof

The proof compares the same list of candidates on two neighbouring datasets, so the list must be fixed before you look at the data. Suppose you build the list from the data, for example from the words that occur in a corpus. Then the mere presence of a candidate can reveal a record, and the guarantee no longer covers it. On the other hand, the proof never uses the structure of the candidates, which can be models, strings, split points or graphs. That is what makes the exponential mechanism the general tool for outputs that are not numbers.

The list may even be infinite. On a continuous range the sum becomes an integral, taken against a fixed weighting of the range that is chosen before seeing the data. The same two-effect argument still works. The standard example is the private median, where the candidates are all the points of an interval.

## Utility of the exponential mechanism {#sec-sel-utility}

The number of candidates was free for privacy, but it is not free for accuracy. With more candidates there are more bad ones to pick by accident. Write $\mathrm{OPT}(D)$ for the best score on dataset $D$. The **gap** is the best score minus the score of the candidate the mechanism picks, so it is never negative, and it measures what the randomness cost you.

The bound on the gap comes from comparing weights. Take a bad candidate, one whose score is at least some distance $c$ below the best. Its weight is at most $e^{-\varepsilon c/(2\Delta)}$ times the best candidate's weight. The best candidate's weight is part of the normalizer, so this bad candidate is picked with probability at most $e^{-\varepsilon c/(2\Delta)}$. There are at most $k$ bad candidates, and together they are picked with probability at most $k$ times that.

Now choose a small failure probability and write it as $e^{-t}$, for some $t > 0$. Solving for the gap that is that unlikely gives the tail bound, which holds for every dataset $D$:
$$\Pr\Big[\mathrm{OPT}(D) - u(D, M_E(D)) \ge \frac{2\Delta}{\varepsilon}(\ln k + t)\Big] \le e^{-t}.$$ {#eq-sel-em-utility}
The bound on the probability exceeds 1, and so says nothing, until the gap reaches $\frac{2\Delta}{\varepsilon}\ln k$. Adding up the tail beyond that point contributes one more unit of $2\Delta/\varepsilon$, which gives the bound on the average gap,
$$\mathbb E\big[\mathrm{OPT}(D) - u(D, M_E(D))\big] \le \frac{2\Delta}{\varepsilon}(\ln k + 1).$$
Two refinements follow from the same argument. First, if several candidates tie for the best score, say $k_{\mathrm{OPT}}$ of them, each adds its weight to the normalizer, and $\ln k$ becomes $\ln(k/k_{\mathrm{OPT}})$. Second, with monotone scores and the weights $e^{\varepsilon u/\Delta}$, the factor $2\Delta$ becomes $\Delta$.

The logarithm comes from the factor $k$ in front of an exponential. So doubling the number of candidates adds only $\ln 2$ to the bracket, and a million candidates, $2^{20}$ of them, cost about 14 units of $2\Delta/\varepsilon$. No $\varepsilon$-DP mechanism can avoid this logarithm, as [Lower bounds](09-lower-bounds.md) shows.

### How good the bound is

The bound is nearly exact when there is one best candidate and all the others sit the same distance below it. Its two approximations, treating every bad candidate as if it sat exactly at the cut-off and keeping only the best candidate's weight in the normalizer, then cost almost nothing. For example, take 50 candidates at $\varepsilon = 1$, with all 49 runners-up a distance of 20 below the best. The exact probability that the mechanism misses the best is 0.0022, and the bound gives 0.0023 (@fig-sel-em-utility, right). When the scores are spread out, most bad candidates sit far more than $c$ below the best, so the bound can be orders of magnitude too high (left). So use the bound to choose $\varepsilon$, and expect the actual gap to be smaller.

![Probability that the exponential mechanism's choice is at least $c$ below the best score, exact against the bound $k e^{-\varepsilon c/2\Delta}$. Left: the 50 models with scores $900 - r$, for $\varepsilon = 0.5$ and 1; the bound is loose because the scores are spread out. Right: one best model and 49 runners-up all exactly $g$ below it, with $\varepsilon = 1$; the bound is tight.](figures/select_em_utility.svg){#fig-sel-em-utility}

### Example: choosing among 50 models

Suppose you have trained 50 models and want to release the best one, judged on a private validation set of 1,000 examples. Model number $r$ gets $900 - r$ examples right, so the best model has 90.0% accuracy. Models next to each other in the ranking differ by 0.1%. The score is the count of correct predictions, and since adding or removing one validation example changes each count by at most 1, the sensitivity is $\Delta = 1$. @tbl-sel-em-models shows the result for three values of $\varepsilon$.

| $\varepsilon$ | $\Pr[\text{best}]$ | $\mathbb E[\text{gap}]$ | Bound $2(\ln 50 + 1)/\varepsilon$ | $\mathbb E[\text{gap}]$, monotone weights |
| --- | --- | --- | --- | --- |
| 0.1 | 0.0531 | 15.03 | 98.24 | 9.17 |
| 1 | 0.3935 | 1.54 | 9.82 | 0.58 |
| 5 | 0.9179 | 0.09 | 1.96 | 0.01 |

: The exponential mechanism selecting among 50 models with scores $900 - r$. The gap is in correct predictions out of 1,000. {#tbl-sel-em-models}

At $\varepsilon = 1$ the mechanism picks the best model only 39% of the time. Yet it loses just 1.54 correct predictions on average, which is 0.15% accuracy, because its mistakes are mostly near-ties that cost one or two correct predictions. Take $t = 3$. The tail bound then says that a gap of 13.82 or more has probability at most $e^{-3}$, about 0.05. The exact probability is $9.12\times10^{-4}$, about fifty times smaller, because these scores are spread out. At $\varepsilon = 0.1$ the bound on the average gap is 98.24. That says nothing here, since no gap can exceed 49, the distance from the best model to the worst.

Counts of correct predictions are monotone under add/remove, since one more validation example can only add correct predictions. So the weights $e^{\varepsilon u}$ are allowed too. They halve the temperature, and at $\varepsilon = 1$ they cut the average gap to 0.58.

## Report-noisy-max {#sec-sel-rnm}

The other natural recipe adds noise to every score and reports the candidate with the largest noisy score. **Report-noisy-max** draws an independent Laplace noise $Z_r$ for each candidate $r$, all with the same scale $b$. It adds each noise to its candidate's score, and outputs
$$M_N(D) = \arg\max_r\big(u(D, r) + Z_r\big).$$
The key rule is to release only the index of the winner. Releasing the noisy scores as well would release $k$ numbers. For $k$ counting queries, one record can change every count, so the Laplace mechanism would need noise of scale $k/\varepsilon$. Releasing only the winner needs noise of scale $2\Delta/\varepsilon$, whatever $k$ is.

### Why the losers cost nothing

The reason is the bar argument, and it takes four steps. The first step picks one candidate, written $i$, and fixes the noise of every other candidate. Two conditions make this allowed. The noise must be drawn the same way whatever the data are, and the noises must be independent, so that fixing the others leaves $Z_i$ with its Laplace distribution. Then the probability that $i$ wins is an average over the possible values of the other noises, with the same weights on both datasets. So a bound that holds for every fixed value of the other noises also holds for the average.

The second step looks at when candidate $i$ wins. With the others' noise fixed, their noisy scores are fixed numbers, and $i$ wins exactly when its own noisy score beats the best of them. Moving $i$'s score to the other side, $i$ wins when its noise $Z_i$ exceeds a bar. The bar is the best of the other noisy scores minus $i$'s own score.

The third step asks how far one record can move that bar. The best of the other noisy scores is a maximum, and each number in it moves by at most $\Delta$, since only its score part can move. Every number can rise by at most $\Delta$, so the new maximum is at most $\Delta$ above the old one. Every number can fall by at most $\Delta$, so the new maximum is at most $\Delta$ below the old one, even if a different number ends up on top. So the maximum moves by at most $\Delta$, however many candidates it ranges over. Candidate $i$'s own score moves by at most $\Delta$ too, so the bar moves by at most $2\Delta$ in all.

The last step prices that movement. The chance that a Laplace noise $Z$ clears a bar is the area under its density beyond the bar. The density falls by a factor $e^{-1/b}$ for each unit you move away from zero. So sliding it by a distance $s$ changes its height at every point by at most a factor $e^{|s|/b}$. The area beyond the bar then changes by at most the same factor. For every bar $t$,
$$\Pr[Z > t - s] \le e^{|s|/b}\,\Pr[Z > t].$$ {#eq-sel-shift}
For example, with $b = 1$, lowering the bar from 2 to 1 raises the chance of clearing it from 0.0677 to 0.1839. That is exactly $e$ times as much, because both bars lie in the upper tail, where every such chance has the form $\frac12e^{-t}$.

Putting the pieces together, a bar that moves by at most $2\Delta$ costs at most $e^{2\Delta/b}$. With $b = 2\Delta/\varepsilon$ that factor is $e^\varepsilon$, so report-noisy-max is $\varepsilon$-DP. @fig-sel-rnm-threshold illustrates one such shift, with $b = 1$ and a bar $t(D)$ at 0.7 that the neighbour lowers by $s = 1$. The winning probability rises from 0.2483 to 0.6296. Their ratio, 2.5357, is below $e^{s/b} = e$, because the lowered bar has crossed zero and left the tail where the bound is exact.

That is also why the winning noisy score must stay secret. The proof averages over the winner's own noise $Z_i$, and publishing that score would publish $Z_i$ itself. If you need the score, spend extra budget on a fresh Laplace answer for that one query.

![The proof of report-noisy-max in one picture. With the other candidates' noise fixed, candidate $i$ wins when its own noise $Z_i \sim \mathrm{Lap}(1)$ exceeds a bar $t$. Moving to a neighbouring dataset moves the bar by at most $s$, and the winning probability changes by at most a factor $e^{s/b}$.](figures/select_rnm_threshold.svg){#fig-sel-rnm-threshold}

### Monotone scores and the factor of 2

The bar moves by $2\Delta$ only when its two parts move in opposite directions, with the competitors rising while candidate $i$ falls. With monotone scores that cannot happen. The maximum and $i$'s own score then move the same way, each by at most $\Delta$, so their difference moves by at most $\Delta$. In that case $b = \Delta/\varepsilon$ is enough, which Dwork and Roth prove for counting queries (Claim 3.9). This is the same factor of 2 as in the exponential mechanism, for the same reason.

### Report-noisy-max or the exponential mechanism?

The two mechanisms have the same privacy but are different mechanisms, and neither is always better. On the 50 models at $\varepsilon = 1$ with $b = 2$, report-noisy-max picks the best model with probability 0.4241, against 0.3935 for the exponential mechanism. Its average gap is smaller too, 1.40 against 1.54. With only two candidates, though, the full edition's exercises show that the exponential mechanism is slightly better.

A third mechanism adds noise from the exponential distribution, which is never negative, instead of two-sided Laplace noise. It is the same as the "permute-and-flip" mechanism of McKenna and Sheldon (2020), and its expected score is never below the exponential mechanism's.

### Accuracy

Noise can only pick a bad candidate if some noise is large. A single Laplace noise exceeds a level $a$ in absolute value with probability $e^{-a/b}$. By the union bound, the chance that any of the $k$ noises does so is at most $k$ times that. Setting this to a failure probability $\beta$ shows that, with probability at least $1 - \beta$, every noise is smaller than $b\ln(k/\beta)$. This is the rule for the largest of several Laplace draws from [Concentration inequalities](../part01-prerequisites/02-concentration.md). On that event, the winner's noisy score beat the best candidate's noisy score. The winner's noise added less than $b\ln(k/\beta)$ and the best candidate's noise took away less than that, so with probability at least $1 - \beta$,
$$\mathrm{OPT}(D) - u(D, M_N(D)) \le 2b\ln(k/\beta).$$
Like the exponential mechanism's bound, this one is loose for spread-out scores. On the 50 models at $\beta = 0.05$ it is 27.63 for $b = 2$, while the average gap is 1.40.

## Gumbel noise: report-noisy-max is the exponential mechanism {#sec-sel-gumbel}

Report-noisy-max with Laplace noise is close to the exponential mechanism but not equal to it. With a different noise, drawn from the Gumbel distribution, it becomes exactly the exponential mechanism. This is the **Gumbel-max trick**, used throughout machine learning to sample from a softmax.

A Gumbel noise $G$ has a scale $\tau$, which will play the role of the softmax temperature. Its cumulative distribution function (CDF) is
$$\Pr[G \le g] = \exp\big(-e^{-g/\tau}\big),$$
and NumPy's `rng.gumbel(0, tau)` samples it. Now add independent Gumbel noise $G_r$ of scale $\tau$ to the score $u_r$ of each candidate $r$, and take the argmax. The claim is that this picks each candidate $i$ with exactly its softmax probability at temperature $\tau$:
$$\Pr\Big[\arg\max_r\big(u_r + G_r\big) = i\Big] = \frac{e^{u_i/\tau}}{\sum_r e^{u_r/\tau}}.$$ {#eq-sel-gumbel}
With the scale $\tau = 2\Delta/\varepsilon$, this is the exponential mechanism of @eq-sel-em.

The case of two candidates shows why it works. Dividing every score and noise by $\tau$ does not change which is largest, so you can take $\tau = 1$. Candidate 1 wins when the difference of the two noises is smaller than the difference of the two scores. The difference of two independent Gumbel variables follows the logistic distribution, whose CDF is the logistic function $1/(1 + e^{-x})$. Evaluated at the score difference, the logistic function is exactly the softmax of the two scores. The full edition extends the same calculation to any number of candidates.

The Gumbel tail also obeys the shift rule @eq-sel-shift for a bar that moves down, as the full edition shows from the shape of the Gumbel CDF. A bar that moves up only lowers the chance of clearing it. So the four-step argument of @sec-sel-rnm works with Gumbel noise of scale $2\Delta/\varepsilon$ too, which gives a second proof that the exponential mechanism is $\varepsilon$-DP.

### Why it matters in practice

The noisy max is the stable way to sample the exponential mechanism. Computing $e^{\varepsilon u/(2\Delta)}$ directly overflows when scores are in the thousands. The noisy max works with the scores themselves, and never computes a weight or a normalizer. The other fix is to subtract the largest score from every score before you take exponentials. That is allowed because a constant added to every score cancels.

Gumbel noise also selects the top $c$ candidates in one shot, for a number $c$ you choose. Taking the $c$ largest of the noisy scores $u_r + G_r$ gives the same distribution as running the exponential mechanism $c$ times and removing each winner (Durfee & Rogers 2019). Its privacy cost is therefore that of $c$ exponential mechanisms, which is $c\varepsilon$ under basic composition.

### Example: five candidates, a million draws

Take five candidates with the scores in @tbl-sel-gumbel, and set $\varepsilon = 1$ and $\Delta = 1$, so that the noise scale is 2. The table compares the three mechanisms, and @fig-sel-gumbel-em draws the same comparison from a separate simulation of 200,000 runs.

| Candidate score | 4 | 3 | 3 | 1 | 0 |
| --- | --- | --- | --- | --- | --- |
| Exponential mechanism, exact | 0.3889 | 0.2359 | 0.2359 | 0.0868 | 0.0526 |
| Gumbel noisy max, $10^6$ runs | 0.3891 | 0.2351 | 0.2362 | 0.0871 | 0.0525 |
| Laplace noisy max, exact | 0.4105 | 0.2341 | 0.2341 | 0.0764 | 0.0449 |

: Selection probabilities with noise scale 2 for five candidates. {#tbl-sel-gumbel}

The Gumbel frequencies are within 0.0008 of the exact probabilities, which is about what a million draws can resolve. The Laplace noisy max puts 0.0216 more probability on the top candidate. That is also its **total variation distance** from the exponential mechanism, the largest difference in probability that the two can give to any set of candidates. Laplace and Gumbel noise of the same scale have similar bodies but different tails, so their selection probabilities differ by a few percent. Here the Laplace version favours the top candidate a little more and the bottom ones a little less.

![Selection probabilities for five candidates with scores $(4, 3, 3, 1, 0)$ and noise scale $2\Delta/\varepsilon$ with $\varepsilon = 1$: the exponential mechanism's exact probabilities, the frequencies of the Gumbel noisy max in 200,000 runs, and the exact probabilities of the Laplace noisy max.](figures/select_gumbel_em.svg){#fig-sel-gumbel-em}

## The sparse vector technique {#sec-sel-svt}

Now the candidates arrive one at a time, and you only care about the first one that is good enough. You might monitor a thousand statistics and want to know which one first crosses an alarm level. Most queries in such a stream are uninteresting, because their answer is "below". Answering each one with the Laplace mechanism would cost $\varepsilon$ per query, so a thousand queries would cost $1000\varepsilon$ under basic composition.

The **sparse vector technique** (Dwork et al. 2009) compares each query with a single noisy threshold, and pays only for the one query that is above it. Its basic form is called **AboveThreshold** (Dwork & Roth 2014, Algorithm 1), and it takes a threshold $T$ and a stream of queries, each a number computed from the data. Write $f_i$ for the $i$-th query, and $\Delta$ for the sensitivity that every query shares. The queries may be chosen adaptively, each one after seeing the earlier answers. The algorithm splits the budget $\varepsilon$ into two parts, $\varepsilon_1$ for the threshold and $\varepsilon_2$ for the queries, and then works in two steps.

1. Draw the threshold noise $\rho$ from $\mathrm{Lap}(\Delta/\varepsilon_1)$, once for the whole run.
2. For each query $f_i$, draw a fresh query noise, written $\nu_i$, from $\mathrm{Lap}(2\Delta/\varepsilon_2)$. If the noisy answer $f_i(D) + \nu_i$ is at least the noisy threshold $T + \rho$, output $\top$ ("above") and halt. Otherwise output $\bot$ ("below") and go on to the next query.

The output is a run of $\bot$ answers followed by one $\top$, or only $\bot$ answers if no query fires. Equivalently, it is the index of the query that fired. AboveThreshold is $(\varepsilon_1 + \varepsilon_2)$-DP, for any number of $\bot$ answers. @fig-sel-svt-path shows one run.

![One run of AboveThreshold with $\varepsilon = 1$ on 30 queries whose true answers rise from 8 to 22.5, with threshold $T = 20$. Each query's noisy answer is compared with the same noisy threshold. The first one above it is reported and the run halts; later queries are never looked at.](figures/select_svt_path.svg){#fig-sel-svt-path}

### Why the "below" answers are free

Take a run that stops at query $k$, so that it compared $k$ queries in all. The proof first fixes the noise of the $k - 1$ earlier queries, which is allowed because that noise does not depend on the data. The earlier queries are then fixed too, even if they were chosen adaptively. Each could depend only on earlier answers, and on this output those were all $\bot$.

All the earlier answers are $\bot$ exactly when the largest earlier noisy answer is below the noisy threshold. That largest answer is a maximum, so one record moves it by at most $\Delta$, however many earlier queries there were. Sliding the threshold noise $\rho$ by that much keeps every $\bot$ in place at once. The same density argument that gives @eq-sel-shift prices that slide, and with threshold noise of scale $\Delta/\varepsilon_1$ it costs $e^{\varepsilon_1}$.

The single $\top$ is paid for separately, by its own fresh noise $\nu_k$. Query $k$ says $\top$ when its noisy answer reaches the noisy threshold. On the neighbour, the threshold has been slid by up to $\Delta$, and the query's own value has changed by up to $\Delta$. Sliding $\nu_k$ by the sum of the two, at most $2\Delta$, restores the comparison. With query noise of scale $2\Delta/\varepsilon_2$, the density argument prices that slide at $e^{\varepsilon_2}$. The two costs multiply to $e^{\varepsilon_1 + \varepsilon_2}$, while the output "all below" needs only the first slide.

With monotone queries, such as counts under add/remove, the largest earlier answer and the $k$-th query move the same way. Their changes then partly cancel, so the second slide is at most $\Delta$. Query noise of scale $\Delta/\varepsilon_2$ is then enough (Lyu, Su & Li 2017).

The proof is tight. On the neighbouring pair used in @sec-sel-svt-bugs, the largest privacy loss of AboveThreshold reaches exactly $\varepsilon$ once each half of that pair has 20 queries (@tbl-sel-svt-loss).

### Accuracy

The privacy cost does not depend on the number of queries, but the accuracy does. A noisy comparison gives the same answer as the true comparison whenever the noises are smaller than the distance between the true numbers. So the accuracy guarantee needs every noise in the run to be small at once, and with $k$ queries in the run that takes a union bound over $k$ events. For the equal split of the budget and a failure probability $\beta$, the guarantee uses a margin of error, written $\alpha$, which is
$$\alpha = \frac{8\Delta\big(\ln k + \ln(2/\beta)\big)}{\varepsilon}.$$ {#eq-sel-svt-alpha}

The 8 comes from the equal split, which gives each query noise the scale $4\Delta/\varepsilon$, and from asking every noise to stay below half the margin. With probability at least $1 - \beta$, every query answered $\bot$ has a true value below $T + \alpha$. On the same event, the query answered $\top$ has a true value of at least $T - \alpha$. So only queries within $\alpha$ of the threshold can go either way, while queries further below do not fire and queries further above are not skipped.

### Halting, and several answers above

The run must halt because the proof pays for only one $\top$. After the first $\top$, later queries would still compare against the same threshold noise. The slide of $\rho$ keeps the $\bot$ answers consistent. But each further $\top$ would need its own noise slid by up to $2\Delta$, which costs another factor $e^{\varepsilon_2}$ that nobody paid for. @sec-sel-svt-bugs shows what happens without the halt.

To find the first $c$ queries above the threshold, restart AboveThreshold with fresh threshold noise after each $\top$, with budget $\varepsilon/c$ for each run. By basic composition the whole procedure is $\varepsilon$-DP, and Dwork and Roth call it Sparse. Its margin is $\alpha$ with $\varepsilon/c$ in place of $\varepsilon$. Lyu, Su and Li give an alternative that keeps one threshold noise and widens each query's noise to scale $2c\Delta/\varepsilon_2$, so that the $c$ slides together cost $e^{\varepsilon_2}$.

AboveThreshold answers a yes-or-no question. To release the value of the query that fired, spend extra budget on a fresh Laplace answer to that query. The split between $\varepsilon_1$ and $\varepsilon_2$ affects only accuracy. The equal split is simple, but Lyu, Su and Li recommend $\varepsilon_1 : \varepsilon_2 = 1 : (2c)^{2/3}$ to maximize the chance of correct answers.

### Example: an alarm among 1,000 statistics

Suppose you monitor $k = 1000$ counting queries, each with sensitivity 1, against the threshold $T = 500$ with $\varepsilon = 1$. With $\beta = 0.05$, @eq-sel-svt-alpha gives the margin $\alpha = 84.77$. So with probability at least 0.95, no query with a count below 415.23 fires. On the same event, none with a count above 584.77 is skipped. The full edition's stream gives every query the count 400, except query 700, whose count is 600. The guarantee therefore says the output is query 700 with probability at least 0.95, and in 10,000 simulated runs it was query 700 every time.

The union bound behind $\alpha$ is nearly tight. In each run, the noises needed a margin of twice the largest of them in absolute value, counting the threshold noise. In 95% of the runs that margin was no more than 78.59, not far under 84.77. Answering every query with the Laplace mechanism instead, at $\varepsilon/1000$ each, would add noise of scale 1,000 to counts of a few hundred.

## When the sparse vector technique goes wrong {#sec-sel-svt-bugs}

Lyu, Su and Li (2017) surveyed published variants that each looked like a harmless optimization, and found that several were not differentially private at all (@tbl-sel-svt-variants).

| Variant (as numbered by Lyu, Su & Li) | What it changes | Privacy shown by Lyu, Su & Li |
| --- | --- | --- |
| Algorithm 1 (theirs); Algorithm 2 (Dwork & Roth) | Correct, with $c$ answers above | $\varepsilon$ |
| Algorithm 3 (Roth's 2011 lecture notes) | Outputs the noisy value $f_i + \nu_i$ for each $\top$ | $\infty$ |
| Algorithm 4 (Lee & Clifton 2014) | Query noise does not grow with $c$ | $\frac{1 + 6c}{4}\varepsilon$ |
| Algorithm 5 (Stoddard et al. 2014) | No noise on the queries | $\infty$ |
| Algorithm 6 (Chen et al. 2015) | Never halts | $\infty$ |

: Variants of the sparse vector technique and their actual privacy, from Lyu, Su & Li (2017). {#tbl-sel-svt-variants}

Each broken variant removes one ingredient of the proof. Algorithm 3 publishes the noisy value of each $\top$, which releases the very noise the proof averages over. Algorithm 5 drops the fresh query noise that pays for the $\top$, so once the threshold noise is drawn, every comparison is deterministic. Algorithm 6 drops the halt that limits a run to the one $\top$ that was paid for, so the cost of each $\top$ adds up with nothing to stop it.

### The no-halt variant

The no-halt variant is the instructive case. It uses exactly AboveThreshold's noise with the equal split, so the query noise has scale $4\Delta/\varepsilon$, and it only skips the halt. To show that a mechanism is not private, it is enough to find one pair of neighbours and one output whose probability ratio is too large. Take $\Delta = 1$ and $2m$ queries, where $m$ is any whole number. On $D$, the first $m$ queries have the value 0 and the last $m$ have the value 1, while on $D'$ it is the other way round. Each query changes by 1 between the two, so each has sensitivity 1.

Write $\bot^m\top^m$ for the output of $m$ answers "below" followed by $m$ answers "above". Every one of its $2m$ answers is a little more likely under $D$. On $D$ the first half of the queries are low, which makes their $\bot$ easier, and the second half are high, which makes their $\top$ easier.

Once the threshold is fixed, the answers are independent, because each depends only on its own fresh noise. So their small factors multiply. The full edition shows that each pair of one low and one high query contributes at least a factor $e^{\varepsilon/4}$, whatever the threshold. So the privacy loss of the output, which is the log of its probability ratio, grows at least linearly in $m$:
$$\ln\frac{\Pr[\bot^m\top^m \mid D]}{\Pr[\bot^m\top^m \mid D']} \ge \frac{m\varepsilon}{4}.$$ {#eq-sel-no-halt}
The loss grows without bound as $m$ grows, so the variant is not DP for any value of $\varepsilon$. AboveThreshold escapes this because it halts after the first $\top$, so a run never contains more than one $\top$ to pay for.

@tbl-sel-svt-loss gives the exact losses at $\varepsilon = 1$ and $T = 0$. For AboveThreshold it shows the largest loss over all possible outputs on this pair. For the no-halt variant it shows the loss of $\bot^m\top^m$.

| $m$ | 1 | 5 | 10 | 20 | 40 |
| --- | --- | --- | --- | --- | --- |
| AboveThreshold, largest loss over outputs | 0.3958 | 0.8998 | 0.9940 | 1.0000 | 1.0000 |
| No halt, loss of $\bot^m\top^m$ | 0.3958 | 2.0985 | 4.3220 | 8.8741 | 18.1246 |
| Lower bound $m\varepsilon/4$ | 0.25 | 1.25 | 2.50 | 5.00 | 10.00 |

: Exact privacy loss of AboveThreshold and of its no-halt variant on the counterexample, with $\varepsilon = 1$. {#tbl-sel-svt-loss}

At $m = 1$ the two mechanisms agree, because AboveThreshold halts after the $\top$ anyway. After that, the no-halt loss grows by about 0.45 for each pair of queries, while AboveThreshold levels off at $\varepsilon$. The lower bound is loose by a factor a little under 2. It uses the smallest factor per pair over all thresholds, and at any given threshold the factor is larger.

Dropping the query noise breaks even the halting version, with only two queries. Let the values be 0 and then 1 on $D$, and 1 and then 0 on $D'$. Without query noise, a query says $\top$ exactly when its value reaches the noisy threshold. The output "below, then above" needs the noisy threshold to lie above 0 and at or below 1 on $D$, which has a positive probability. On $D'$ it needs the threshold to lie above 1 and at or below 0, which is impossible. A positive probability against zero is a ratio that no factor $e^\varepsilon$ can bound.

### Catching the bug

The bug is in the accounting, not in the noise. The no-halt variant adds exactly as much noise per query as AboveThreshold, so a reviewer who checks only noise scales will miss it.

An empirical audit catches it, and it needs no exact probabilities. If a mechanism is $\varepsilon$-DP, then every event is at most $e^\varepsilon$ times as likely under $D$ as under $D'$. That includes the event that some test you run on the output says "$D$". So the log of the ratio of a test's firing rates under the two datasets is a lower bound on the true $\varepsilon$. This is the hypothesis-testing view of DP from [The definition of differential privacy](01-definition.md).

A simple test asks whether the second half of the output has more $\top$ answers than the first. Over 200,000 runs on each dataset, the log of the ratio of its firing rates grows with $m$, from 0.77 at $m = 10$ to 2.63 at $m = 160$. It passes 1 at $m = 40$, so from there on this test alone proves that the variant is not 1-DP (@fig-sel-svt-buggy, right).

![Left: privacy loss on the neighbouring pair $f(D) = (0^m, 1^m)$, $f(D') = (1^m, 0^m)$ with $\varepsilon = 1$, computed exactly by integrating over the threshold noise. For the no-halt variant, the loss of the output $\bot^m\top^m$ grows linearly in $m$, above the lower bound $m\varepsilon/4$. For AboveThreshold, the largest loss over all outputs levels off at $\varepsilon$. Right: the no-halt variant under a simple test, the probability that the second half has more $\top$ answers than the first. Once the probability under $D$ exceeds $e^\varepsilon$ times the probability under $D'$ (dotted), no $\varepsilon$-DP mechanism could produce it.](figures/select_svt_buggy.svg){#fig-sel-svt-buggy}

When you review an implementation, check the accounting, not only the noise scales. Reusing one noise draw for all queries, for example, has the effect of Algorithm 5. Every query is then compared with the same random level, the threshold noise minus the shared draw, so once that level is drawn every comparison is deterministic. Releasing the index of the first $\top$ is the same as releasing the answers up to it, so the output format is never the problem. Answering after the $\top$ is.

## Code {#sec-sel-code}

Each mechanism fits in a few lines. Note where AboveThreshold draws its two noises.

```python
import random
from math import exp, log
from typing import Sequence


def exponential_mechanism_probs(scores: Sequence[float], eps: float, sensitivity: float) -> list[float]:
    """Selection probabilities, proportional to exp(eps * u / (2 * sensitivity)), without overflow."""
    top = max(scores)
    weights = [exp(eps * (u - top) / (2 * sensitivity)) for u in scores]
    total = sum(weights)
    return [w / total for w in weights]


def gumbel_select(scores: Sequence[float], eps: float, sensitivity: float, rng: random.Random) -> int:
    """One draw of the exponential mechanism: add Gumbel noise of scale 2 * sensitivity / eps, take the argmax."""
    tau = 2 * sensitivity / eps
    noisy = [u - tau * log(-log(1 - rng.random())) for u in scores]
    return max(range(len(scores)), key=noisy.__getitem__)


def laplace(scale: float, rng: random.Random) -> float:
    return rng.expovariate(1 / scale) - rng.expovariate(1 / scale)


def above_threshold(
    answers: Sequence[float], threshold: float, eps: float, sensitivity: float, rng: random.Random
) -> int | None:
    """Index of the first answer whose noisy value reaches the noisy threshold; equal budget split."""
    # With eps1 = eps2 = eps / 2: threshold scale sensitivity / eps1, query scale 2 * sensitivity / eps2.
    noisy_threshold = threshold + laplace(2 * sensitivity / eps, rng)  # drawn once per run
    for i, answer in enumerate(answers):
        if answer + laplace(4 * sensitivity / eps, rng) >= noisy_threshold:  # fresh noise every query
            return i  # halt, and release only the index
    return None


def svt_margin(k: int, eps: float, sensitivity: float, beta: float) -> float:
    return 8 * sensitivity * (log(k) + log(2 / beta)) / eps


print([round(p, 4) for p in exponential_mechanism_probs([3, 2, 1, 0], 1.0, 1.0)])  # [0.4551, 0.276, 0.1674, 0.1015]
print(f"{svt_margin(1000, 1.0, 1.0, 0.05):.2f}")  # 84.77
counts = [600.0 if i == 699 else 400.0 for i in range(1000)]  # query 700 is the only high one
print(above_threshold(counts, 500.0, 1.0, 1.0, random.Random(0)))  # 699
```

The first printed line is the vote of @tbl-sel-em-votes at $\varepsilon = 1$. The last finds query 700 of the alarm stream, at index 699 because Python counts from zero. Floating-point samplers like these are fine for checking a calculation but not for production, as [Discrete and floating-point-safe mechanisms](07-discrete-mechanisms.md) explains, so use a vetted DP library there. The full edition's script, [selection.py](../../code/part03-dp-foundations/selection.py), computes every number in this chapter. It checks each mechanism both by exact integration and by simulation.

```bash
uv run python code/part03-dp-foundations/selection.py
```

## Cheat sheet

| Mechanism | Output | Privacy | Accuracy |
| --- | --- | --- | --- |
| Exponential mechanism | $r$ with probability $\propto e^{\varepsilon u(D, r)/(2\Delta)}$ | $\varepsilon$; $e^{\varepsilon u/\Delta}$ for monotone scores | Gap $< \frac{2\Delta}{\varepsilon}(\ln k + t)$ w.p. $\ge 1 - e^{-t}$; mean gap $\le \frac{2\Delta}{\varepsilon}(\ln k + 1)$ |
| Report-noisy-max, Laplace | $\arg\max_r(u + \mathrm{Lap}(2\Delta/\varepsilon))$, index only | $\varepsilon$; $\mathrm{Lap}(\Delta/\varepsilon)$ for monotone scores | Gap $\le 2b\ln(k/\beta)$ w.p. $\ge 1 - \beta$ |
| Report-noisy-max, Gumbel | $\arg\max_r(u + \mathrm{Gumbel}(2\Delta/\varepsilon))$ | Identical to the exponential mechanism | Same |
| AboveThreshold | Index of the first $f_i + \mathrm{Lap}(2\Delta/\varepsilon_2) \ge T + \mathrm{Lap}(\Delta/\varepsilon_1)$, then halt | $\varepsilon_1 + \varepsilon_2$, any number of $\bot$ | $\alpha = 8\Delta(\ln k + \ln\frac2\beta)/\varepsilon$, equal split |
| Sparse ($c$ answers above) | AboveThreshold restarted $c$ times at $\varepsilon/c$ | $\varepsilon$ by basic composition | $\alpha$ with $\varepsilon/c$ in place of $\varepsilon$ |

: The selection mechanisms of this chapter. Here $k$ is the number of candidates or queries, $\Delta$ the sensitivity of each score or query, $b$ the Laplace scale, $\beta$ the failure probability and $t > 0$ the tail parameter of @eq-sel-em-utility. {#tbl-sel-cheat}

**Rules of thumb**

- Release positions, not values. Every released noisy value needs its own budget.
- The number of candidates costs nothing in privacy and only $\ln k$ in accuracy.
- Keep the factor $2\Delta$ unless you have checked that the scores are monotone under your neighbouring relation. Counts under add/remove are monotone, and a histogram under replace is not.
- Fix the candidate list before you look at the data.
- To sample the exponential mechanism, add Gumbel noise and take the argmax.
- A sparse vector implementation must draw fresh noise for every query and keep one threshold noise per run. It must halt after the allowed number of $\top$ answers, and never publish the noisy values it compared.

## Further reading

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), §3.3, §3.4 and §3.6: report-noisy-max, the exponential mechanism and AboveThreshold, with proofs.
- Lyu, Su & Li, "Understanding the Sparse Vector Technique for Differential Privacy" (Proceedings of the VLDB Endowment, 2017): the general algorithm, the budget split and the survey of broken variants. The best single source on the technique.
- Durfee & Rogers, "Practical Differentially Private Top-k Selection with Pay-what-you-get Composition" (NeurIPS 2019): one-shot Gumbel top-$c$ selection.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/05-selection.md) or in the exercises that accompany it, except the following, which it cites: the top-$c$ Gumbel selection (Durfee & Rogers), the permute-and-flip results (McKenna & Sheldon; Ding et al.), the variant with $c$ answers above and one threshold noise, the recommended budget split and the privacy of Algorithms 3 and 4 in @tbl-sel-svt-variants (Lyu, Su & Li), and the lower bound on the $\ln k$, which is derived in [Lower bounds](../../book-full/part03-dp-foundations/09-lower-bounds.md).
