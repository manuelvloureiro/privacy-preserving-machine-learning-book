# Private prediction: PATE and in-context learning

## Why this matters

Training a model with differential privacy (DP) by stochastic gradient descent (SGD), as DP-SGD does ([DP-SGD](02-dp-sgd.md)), makes its weights private. Every later query to the model is then free, because it is post-processing of the weights. The price is noise on every gradient, paid during training, and you must own the training loop.

Sometimes that price cannot be paid, or is not worth paying. The model may be a large language model (LLM) that you can prompt but not train, or a model with no gradients, like a forest of decision trees. Or it may answer only a few dozen questions in its whole life, and then a private training run is overkill. In each of these cases there is another option. You train ordinary, non-private models, **never release them**, and make each *answer* private instead.

The idea that makes this work is "one record, one vote". Split the private records into $k$ separate groups, called shards, and train one model, a **teacher**, on each shard. To answer a query, let every teacher vote for a label, add noise to the vote counts, and release only the label whose noisy count is largest. A record lives in exactly one shard, so it can change only one teacher, and so only one vote. That holds however the teachers were trained and however badly one of them overfits its shard, which is why the privacy argument needs nothing about the teachers.

Accuracy is a different matter. It comes down to the **gap** between the two largest vote counts, compared with the size of the noise. So private prediction is cheap for a few queries on which the teachers agree, and expensive otherwise. This chapter measures both, compares the vote with a private model, and then shows how private aggregation of teacher ensembles (PATE) turns private answers into a model whose use is free. It ends with the examples placed in an LLM's prompt.

## The noisy vote: LNMax and GNMax {#sec-pate-noisy-max}

A query arrives, the $k$ teachers vote, and you must release one label. How much one record can change the votes depends on how the records are split into shards. Suppose the split is fixed before looking at the data, by a hash of a record identifier. Adding or removing a record then changes only the shard its hash points to, and every other teacher stays exactly as it was. The changed teacher may now vote differently on a query, so one count goes down by 1 and another goes up by 1, or nothing changes. The list of vote counts, called the **vote histogram**, therefore moves by at most $\sqrt{1^2 + 1^2} = \sqrt2$ in Euclidean length. This is the subsample-and-aggregate idea of [Beyond global sensitivity](../part03-dp-foundations/06-sensitivity.md#sec-sens-saa), with a vote as the way of combining the shards.

The fixed split is a condition of the guarantee, not a convenience. If shards were cut by position in the file, inserting one record at the front would shift every later record into the next shard. Every shard would change, and possibly every teacher, so one record could move every vote.

Two mechanisms release the label, and they differ in their noise. The original PATE paper (Papernot et al. 2017) used **LNMax**, which adds Laplace noise to each count, with a scale written $b$. Releasing the label with the largest noisy count is report-noisy-max ([Selection](../part03-dp-foundations/05-selection.md#sec-sel-rnm)) on counts that one record moves by at most 1 each. Each LNMax answer is $(2/b)$-DP, with a factor 2 because one record can move both sides of a comparison at once, lowering the winner's count and raising its rival's.

Its successor (Papernot et al. 2018) uses **GNMax**, which adds Gaussian noise instead, with a standard deviation written $\sigma$. That is the Gaussian mechanism on a vector that one record moves by at most $\sqrt2$. Taking the largest noisy count is then post-processing, which cannot add to the cost.

The cost of GNMax is best measured in Rényi differential privacy (RDP) ([Rényi DP](../part04-privacy-accounting/01-renyi-dp.md)). RDP describes a mechanism by a curve $\varepsilon(\alpha)$, which gives a privacy cost at each **order** $\alpha$. The order is a number above 1, and larger orders pay more attention to rare outputs that reveal a lot. For the Gaussian mechanism the curve is a straight line through zero, with slope equal to the squared sensitivity divided by twice the noise variance. With a sensitivity of $\sqrt2$, each GNMax answer therefore costs
$$\varepsilon(\alpha) = \frac{\alpha}{\sigma^2}.$$
RDP curves add up under composition, even when each query is chosen after seeing the earlier answers, so a number $Q$ of answers costs a line with slope $Q/\sigma^2$.

At the end, [From Rényi DP to (ε, δ)](../part04-privacy-accounting/02-rdp-conversions.md) converts the total curve into the usual guarantee, with a privacy level $\varepsilon$ and a small slack $\delta$. The conversion of a line depends only on its slope, because the slope is the only thing that distinguishes one line through zero from another. So write $\rho(\varepsilon, \delta)$ for the largest slope whose line still converts to your budget. Setting $Q/\sigma^2$ equal to that slope gives the noise that $Q$ answers need,
$$\sigma = \sqrt{\frac{Q}{\rho(\varepsilon, \delta)}}.$$ {#eq-pate-sigma}
This chapter uses $\delta = 10^{-5}$ throughout, where a budget of $\varepsilon = 1$ allows a slope of at most 0.0306, and $\varepsilon = 4$ a slope of up to 0.3731.

Two consequences follow from @eq-pate-sigma. First, the noise grows like the square root of the number of answers, because every answer is a release and the costs of all of them add up. A hundred times more answers need ten times more noise. Second, neither the number of teachers nor anything about them enters the formula. A teacher can be a neural network trained without any noise, a decision forest, or a human annotator who saw one shard.

The analysis holds only if you release the label alone. Publishing the noisy counts as well would release a whole vector, and every count would then have to be paid for as a release of its own.

@tbl-pate-noise gives the noise for a few numbers of answers. To compare the two mechanisms on one scale, it gives the Laplace noise by its standard deviation, $\sqrt2b$. It charges each LNMax answer the RDP curve of randomized response, which is valid for any pure-DP step ([Concentrated differential privacy](../part04-privacy-accounting/03-concentrated-dp.md#sec-cdp-pure)).

| Budget | Answers $Q$ | GNMax $\sigma$ | LNMax $b$ | LNMax standard deviation $\sqrt2b$ |
| --- | --- | --- | --- | --- |
| $\varepsilon = 1$ | 1 | 5.72 | 2.02 | 2.85 |
| | 100 | 57.2 | 79.7 | 113 |
| | 1,000 | 181 | 255 | 361 |
| | 10,000 | 572 | 809 | 1,140 |
| $\varepsilon = 4$ | 1 | 1.64 | 0.501 | 0.709 |
| | 100 | 16.4 | 22.7 | 32.1 |
| | 1,000 | 51.8 | 73.1 | 103 |
| | 10,000 | 164 | 231 | 327 |

: Noise per count for $Q$ answers at $(\varepsilon, 10^{-5})$, from the RDP accountant; Laplace answers are charged the randomized-response curve. {#tbl-pate-noise}

For a single answer, Laplace needs less noise, because a pure-DP analysis is cheap for one step. As answers accumulate the Gaussian wins, because each Laplace answer is charged for its worst case every time. From 100 answers on, the Gaussian noise is about half as large, so GNMax is the default for anything beyond a handful of answers.

The table also shows the limit of private prediction. At $\varepsilon = 1$, 10,000 answers need $\sigma = 572$. A gap can never exceed the number of teachers, so noise that large swamps the vote of any ensemble with fewer than a few thousand teachers. The RDP accountant itself is close to exact here. Where it says $\varepsilon = 1$, the exact analysis of a Gaussian histogram ([f-DP and Gaussian DP](../part04-privacy-accounting/04-gaussian-dp.md)) gives 0.915, at every number of answers.

## How accurate is a noisy vote {#sec-pate-accuracy}

The cost of an answer does not depend on the teachers, but its accuracy depends on little else. Start with two labels, and write $g$ for the gap, the plurality's count minus the other count. GNMax returns the plurality label unless the other count's noise beats the plurality's noise by more than the gap. So only the difference of the two noises matters. The difference of two independent Gaussians with standard deviation $\sigma$ is a Gaussian with standard deviation $\sqrt2\sigma$, because the variances of independent quantities add. So the noisy vote returns the plurality with probability
$$\Pr[\text{plurality}] = \Phi\Big(\frac{g}{\sqrt2\,\sigma}\Big),$$ {#eq-pate-two-class}
where $\Phi$ is the standard normal distribution function, the chance that a standard normal draw falls below its argument. The formula is exact for two labels.

With more labels, the plurality can lose to any of its competitors. The chance that it loses to at least one is at most the sum of the chances that it loses to each, by the union bound ([Concentration inequalities](../part01-prerequisites/02-concentration.md#sec-conc-union)). Write $n_i$ for the count of label $i$ and $i^*$ for the plurality label. Each term is the chance of losing to one competitor, the same quantity as in @eq-pate-two-class written with the complementary error function $\operatorname{erfc}$ that numerical libraries provide. So the chance that the noisy vote returns anything but the plurality is at most
$$\tilde q = \sum_{i \ne i^*}\frac12\operatorname{erfc}\Big(\frac{n_{i^*} - n_i}{2\sigma}\Big)$$ {#eq-pate-qtilde}
(Papernot et al. 2018, Proposition 7). The bound double-counts the outcomes in which several competitors win at once, so it is close when one competitor causes most errors and loose when many labels are nearly tied.

The plurality is not always the correct label, so accuracy needs one more step. On two labels, suppose a fraction $a$ of the $k$ teachers vote for the correct label on a query. Measured toward the correct label, the gap is the number of teachers who are right minus the number who are wrong. That is $k(2a - 1)$, which is negative when most teachers are wrong. Putting this signed gap into @eq-pate-two-class gives the chance of a correct answer, and averaging over queries gives the expected accuracy.

Without noise, the plurality vote is simply an ensemble of the teachers, and its accuracy is a ceiling, because noise can only pull each answer toward a coin flip. On hard queries, where $a$ is near one half, any noise makes the answer close to random, so these are the answers the noise spoils first.

This part's running example, introduced in [DP-SGD](02-dp-sgd.md), makes this concrete. Its inputs are 50-dimensional standard normal vectors, labelled $\pm1$ by a fixed random network. It has $n = 10{,}000$ private training records and 20,000 test records, which are split in two. The first half forms a **public pool**, whose inputs are treated as public and unlabelled, and the second half is the evaluation set for every accuracy in this chapter. The teachers are regularized logistic regressions, the model of [Private empirical risk minimization](01-dp-erm.md), each fitted on one shard.

@fig-pate-votes shows what noise does to real votes. It uses 100 teachers, with the noise that 1,000 answers need at $\varepsilon = 4$, $\sigma = 51.8$. When 98 teachers vote one way and 2 the other, the noisy vote returns the plurality with probability 0.9051. When the vote is 52 to 48, that probability is 0.5218, barely better than a coin. Across the public pool, the votes pile up near unanimity one way or the other, but a large share of queries sits in between.

![Left three panels: the votes of 100 teachers on three public queries, with bars of one noise standard deviation ($\sigma = 51.8$, the GNMax noise for 1,000 answers at $\varepsilon = 4$); the title gives the probability that the noisy vote returns the plurality label. Right: across all 10,000 public queries, the fraction of the 100 teachers voting $+1$.](figures/pate_votes.svg){#fig-pate-votes}

### How many teachers

For a fixed number of answers and budget, the noise is fixed, because $k$ does not appear in @eq-pate-sigma. More teachers multiply the gap $k(2a - 1)$ by $k$, but each teacher then sees only $n/k$ records, so it agrees less with the others and $2a - 1$ shrinks. Which effect wins depends on how fast a teacher's accuracy falls as its shard shrinks.

@tbl-pate-teachers splits the 10,000 private records among 25 to 500 teachers. Its last column, the median gap divided by $k$, measures how strongly the teachers agree.

| Teachers $k$ | Records each | Mean teacher accuracy | Plurality vote accuracy | Median $|g|/k$ |
| --- | --- | --- | --- | --- |
| 25 | 400 | 0.8954 | 0.9403 | 1.000 |
| 50 | 200 | 0.8518 | 0.9389 | 0.880 |
| 100 | 100 | 0.7896 | 0.9384 | 0.660 |
| 250 | 40 | 0.7031 | 0.9336 | 0.408 |
| 500 | 20 | 0.6459 | 0.9339 | 0.276 |

: Teachers on the running example, measured on the evaluation set: each teacher's accuracy, the accuracy of the noiseless plurality vote, and the median gap as a fraction of the number of teachers. {#tbl-pate-teachers}

The teachers' accuracy falls from 0.8954 with 400 records each to 0.6459 with 20. Yet the plurality vote hardly moves, staying between 0.9336 and 0.9403. Each small teacher's errors come mostly from the chance content of its own shard, so they are close to independent across shards. A majority of many weak, nearly independent voters is right far more often than any one of them. The absolute median gap therefore grows, from 25 votes with 25 teachers to 138 with 500, and more teachers win on this task.

Applying @eq-pate-two-class to each query's gap gives the expected accuracy of the private answers. With 100 answers at $\varepsilon = 4$, 500 teachers are right 0.9237 of the time, about one point below their noiseless vote. At $\varepsilon = 1$ and 1,000 answers, the noise is $\sigma = 181$, and the same teachers fall to 0.6987. Twenty-five teachers fall to 0.5308 there, because their gaps can never exceed 25, which is small against that noise.

## Private answers against a private model {#sec-pate-versus}

You can spend one budget on training a model with DP-SGD, whose answers are then free, or on answering queries one at a time with a teacher ensemble. Which gives more accurate answers depends on how many queries you expect.

A released model's accuracy does not depend on the number of queries. The vote's accuracy falls as answers accumulate, because its noise grows like $\sqrt Q$ and, on this task, the plurality is right on most queries. So the two curves cross at most once. Write $A_{\mathrm{model}}(\varepsilon)$ for the private model's accuracy at budget $\varepsilon$. Write $A_{\mathrm{vote}}(Q, \varepsilon)$ for the expected accuracy of $Q$ private answers at that budget. Private answers are at least as accurate for every $Q$ up to the **break-even** number
$$Q^*(\varepsilon) = \max\{Q : A_{\mathrm{vote}}(Q, \varepsilon) \ge A_{\mathrm{model}}(\varepsilon)\},$$ {#eq-pate-breakeven}
which is zero when even one answer is worse than the model.

On the running example, the DP-SGD baseline is the configuration of [DP-SGD](02-dp-sgd.md), retrained at each budget. @tbl-pate-breakeven gives its accuracy and the break-even number of queries. For each budget, it reports the number of teachers with the largest break-even number.

| $\varepsilon$ | DP-SGD accuracy (2 runs) | Break-even $Q^*$ | Best $k$ |
| --- | --- | --- | --- |
| 0.5 | $0.8164 \pm 0.0060$ | 58 | 500 |
| 1 | $0.9128 \pm 0.0207$ | 18 | 500 |
| 2 | $0.9438 \pm 0.0039$ | 0 | — |
| 4 | $0.9573 \pm 0.0024$ | 0 | — |
| 8 | $0.9605 \pm 0.0004$ | 0 | — |

: DP-SGD on the running example against private answers by GNMax: the largest number of queries for which the answers' expected accuracy is at least the private model's. {#tbl-pate-breakeven}

At $\varepsilon = 0.5$, DP-SGD reaches only 0.8164, and 500 teachers answer up to 58 queries at least as accurately as the private model. At $\varepsilon = 1$ the range shrinks to 18 queries. From $\varepsilon = 2$ on, DP-SGD beats even the noiseless vote, the ensemble's ceiling, because that ceiling is at most 0.9403 and DP-SGD reaches 0.9438. @fig-pate-answers shows the whole curve.

![Expected accuracy of $Q$ private answers by GNMax against $Q$, for 25, 100 and 500 teachers, at $\varepsilon = 1$ (left) and $\varepsilon = 4$ (right), $\delta = 10^{-5}$. The dashed line is a DP-SGD model trained at the same budget, whose accuracy does not depend on the number of queries.](figures/pate_answers.svg){#fig-pate-answers}

The pattern is general. At a small budget, DP-SGD gives a model much worse than a non-private one, while the vote's first answers are almost as good as the noiseless ensemble. So there is a window of queries where the vote wins. At a large budget DP-SGD loses little, and the ensemble's ceiling decides.

The comparison leaves out two things, and they pull in opposite directions. It assumes you *can* run DP-SGD. When the teachers are LLMs behind an interface, models without gradients, or models whose owners will not pool their data, private prediction is the only private option. It also ignores the running cost, since DP-SGD trains once while private prediction runs $k$ models on every query.

## PATE: from private answers to a private model {#sec-pate-student}

Private answers cost budget for as long as you give them, while a private model costs it once. PATE combines the two. It spends the budget on labelling $Q$ *public, unlabelled* inputs with GNMax answers. Then it trains a new model, the **student**, on those inputs and labels. The student is released, and the teachers never are.

The student's training reads only the public inputs and the $Q$ noisy labels, never the teachers or the private records. So the student's weights, and every prediction it ever makes, are post-processing of the labels. They cost exactly what the $Q$ GNMax answers cost, whatever the student's size, its training algorithm or the number of predictions it makes. The inputs to be labelled may even be chosen adaptively, because RDP composition allows it.

Three things limit the student. It learns from $Q$ examples rather than from all $n$ private records, some of its labels are wrong, and the plurality vote itself has the ensemble's ceiling. The first two pull against each other, since more labels mean more noise on each label by @eq-pate-sigma. So for each budget there is a best number of labels.

PATE needs public, unlabelled inputs from the same distribution as the private data. That is easy for images or text, where only the labels are sensitive, and impossible for medical records, which have no public pool. The pool must really be public, since nothing here protects an input that is someone's private record. Both PATE papers let the student also learn from the unlabelled inputs, the first with a generative adversarial network (GAN). This chapter's student uses only its labelled ones, the choice least favourable to PATE.

The guarantee covers the released student and its predictions, and nothing else. The teachers are non-private models of the data, and an attacker who steals one can run any attack of [Part 2](../part02-attacks/00-introduction.md) on its shard. So PATE moves the trust from the training algorithm to the server that holds the teachers.

On the running example, take $k = 250$ teachers, whose plurality vote labels the public pool with accuracy 0.9322. The student, the one-hidden-layer network of [DP-SGD](02-dp-sgd.md), is trained non-privately on the first $Q$ public inputs and their GNMax labels. In @tbl-pate-student, the first two rows are references without privacy.

| Labels | $Q = 250$ | $Q = 500$ | $Q = 1{,}000$ | $Q = 2{,}000$ |
| --- | --- | --- | --- | --- |
| True labels (not private) | 0.8726 | 0.8990 | 0.9188 | 0.9410 |
| Noiseless plurality (not private) | 0.8913 | 0.9106 | 0.9229 | 0.9297 |
| GNMax, $\varepsilon = 1$ | 0.8238 (0.7760) | 0.7595 (0.6880) | 0.6995 (0.6350) | 0.6749 (0.6260) |
| GNMax, $\varepsilon = 2$ | 0.7980 (0.8040) | 0.8393 (0.8140) | 0.8141 (0.7490) | 0.7318 (0.6885) |
| GNMax, $\varepsilon = 4$ | 0.8633 (0.8720) | 0.8754 (0.8740) | 0.8898 (0.8360) | 0.8259 (0.7840) |
| GNMax, $\varepsilon = 8$ | 0.8950 (0.9160) | 0.9037 (0.9040) | 0.9049 (0.8870) | 0.8884 (0.8525) |

: Test accuracy of a PATE student with 250 teachers, against the number of labelled public queries $Q$, one run each; in parentheses, the fraction of the student's GNMax labels that are correct. {#tbl-pate-student}

At $\varepsilon = 1$ the best student reaches 0.8238, while DP-SGD reaches 0.9128. At $\varepsilon = 4$ the best student reaches 0.8898, against 0.9573 for DP-SGD. In fact DP-SGD is better at each of the four budgets measured, 1, 2, 4 and 8 (@fig-pate-student), and the table shows why. With true labels, the student needs 2,000 examples to reach 0.9410, but at $\varepsilon = 4$ the noise for 2,000 labels makes only 0.7840 of them right, and with fewer labels the student has too few examples. The best number of labels was chosen here by looking at test accuracy, which is itself a leak, and [Hyperparameter tuning](03-hyperparameter-tuning.md) shows how to pay for it.

![Test accuracy of the PATE student with 250 teachers and 500 or 2,000 labelled public queries, and of DP-SGD, against $\varepsilon$ at $\delta = 10^{-5}$, one run each. The dotted line is the accuracy of the teachers' noiseless plurality vote.](figures/pate_student.svg){#fig-pate-student}

PATE loses here for reasons that point to where it wins. On a binary task a wrong label is the other label, so label noise does the most damage it can. The teachers are weak and often disagree, so many gaps are small against the noise. And the student needs thousands of labels. PATE's published successes are on tasks where the teachers agree on most inputs and a student learns from few labels plus many unlabelled inputs. On the handwritten-digit benchmark MNIST, Papernot et al. (2018) report a student with 98.5% accuracy, against 99.2% without privacy. It used 286 answered queries, at a data-dependent $\varepsilon$ of 1.97. The next section explains that analysis, and why such an $\varepsilon$ must itself be released privately before it counts as a guarantee.

## Data-dependent analysis and Confident-GNMax {#sec-pate-data-dependent}

The cost $\alpha/\sigma^2$ of a GNMax answer is a worst case over all vote histograms, including those where the vote is tied and one teacher decides it. But when 249 of 250 teachers agree, the noisy vote returns the plurality with overwhelming probability, with or without any one record. A **data-dependent analysis** charges such a query less, by looking at the actual votes.

The pure-DP version (Papernot et al. 2017, Theorem 3) shows where the saving comes from. Take an $\varepsilon_0$-DP mechanism, such as LNMax with $\varepsilon_0 = 2/b$, whose answer on the actual data differs from the plurality with probability $q$. Pure DP limits how much a neighbour can change the unlikely answers, which carry only probability $q$, and the likely answer takes the rest. So the cost is zero when $q$ is zero, and about $q$ times a constant when $q$ is small. For example, an LNMax answer with $\varepsilon_0 = 0.1$ costs 0.0367 at order 8 in the worst case, but only 0.00248 when the plurality fails with probability 0.01.

The theorem holds only when $q$ is below $e^{-\varepsilon_0}$. A neighbour can raise the chance of the unlikely answers to $e^{\varepsilon_0}q$, and without the condition that could reach one, making the likely answer impossible. The true $q$ is unknown, so the bound is taken at its worst over every failure probability up to the computable $\tilde q$ of @eq-pate-qtilde.

The Gaussian version for GNMax (Papernot et al. 2018, Theorem 6) follows the same plan with a more involved formula, whose proof is in their paper. Its conditions require the failure bound to be small enough, and the order to stay below a limit that depends on the noise and that bound. Each answer is charged the smaller of this bound and the data-independent $\alpha/\sigma^2$, and the charges add up over answers.

@fig-pate-dd-cost draws the cost of one GNMax answer with $\sigma = 40$ at order 8, against the gap on two labels. Up to a gap of 100, the data-dependent bound is no better than the data-independent cost of 0.005. Then it falls steeply, to $1.23 \times 10^{-3}$ at a gap of 150. The exact data-dependent cost on two labels, the dashed curve, lies below the bound everywhere. At a gap of 150 it is five times lower, so Theorem 6 is valid but not tight.

![The RDP cost at order 8 of one GNMax answer with $\sigma = 40$, against the gap between the two vote counts, on a log scale: the data-independent $\alpha/\sigma^2$, the data-dependent bound of Papernot et al. (2018), and the exact data-dependent cost on two labels.](figures/pate_dd_cost.svg){#fig-pate-dd-cost}

The total cost is a sum over queries, so the saving depends on how many queries have large gaps. With 250 teachers on the running example, only 0.2879 of the public queries have a gap of at least 150, so most answers pay close to full price. @tbl-pate-dd compares the two analyses for the first $Q$ public queries, answered with $\sigma = 40$. Its data-dependent column is not yet a guarantee, as the next subsection explains.

| Answers $Q$ | Data-independent $\varepsilon$ | Data-dependent $\varepsilon$ (not sanitized) |
| --- | --- | --- |
| 250 | 2.45 | 1.93 |
| 1,000 | 5.38 | 4.37 |
| 4,000 | 12.30 | 10.23 |

: $\varepsilon$ at $\delta = 10^{-5}$ for $Q$ GNMax answers with 250 teachers and $\sigma = 40$ on the running example's public pool. {#tbl-pate-dd}

The saving is about 20% at every $Q$ (@fig-pate-epsilon-queries). On a task where nearly all teachers agree on nearly all inputs, as on handwritten digits, almost every answer is nearly free, and $\varepsilon$ falls by a large factor.

![$\varepsilon$ at $\delta = 10^{-5}$ for $Q$ answers with 250 teachers on the running example's public queries, on log–log axes: GNMax with $\sigma = 40$ under the data-independent and data-dependent analyses, and LNMax with the same noise standard deviation.](figures/pate_epsilon_queries.svg){#fig-pate-epsilon-queries}

### The data-dependent ε is itself private

There is a trap here. The data-dependent $\varepsilon$ is computed from the votes, so publishing it reveals something about the teachers, and so about the data. Its sensitivity is also large. One record changes one teacher, and that teacher's votes can change on *every* query, moving every term of the sum. On a query with a large gap the cost is almost zero, but if the change brings the vote close to a tie, the cost jumps to the full $\alpha/\sigma^2$. Noise calibrated to the worst case would therefore be about as large as the data-independent total.

Papernot et al. therefore release it with Gaussian noise scaled to its **smooth sensitivity**, and they pay for that release too. Smooth sensitivity is the tool of [Beyond global sensitivity](../part03-dp-foundations/06-sensitivity.md#sec-sens-smooth) for functions whose worst-case sensitivity is far above their typical one. In their run on MNIST with 640 queries, the data-dependent $\varepsilon$ is 2.00. The sanitized one is 2.52, with a standard deviation of 0.385 from the release noise. Until it is released privately, a data-dependent $\varepsilon$ is a measurement, not a guarantee.

### Confident-GNMax

**Confident-GNMax** (Papernot et al. 2018, Algorithm 1) answers only the queries on which the teachers agree, so it first checks the consensus. It adds Gaussian noise of standard deviation $\sigma_1$ to the largest vote count and compares the result with a threshold $T$. If the noisy count falls short, it answers nothing. If it passes, it releases a GNMax answer with a separate noise $\sigma_2$.

The check costs something on every query, answered or not. One record changes the largest count by at most 1, so the check is a Gaussian mechanism with sensitivity 1 and costs $\alpha/(2\sigma_1^2)$. The answer costs $\alpha/\sigma_2^2$, as before, but only when it is given. Whether it is given depends on a released output, so the costs are chosen adaptively. The Rényi filter of [Adaptive composition](../part04-privacy-accounting/08-adaptive-composition.md#sec-adapt-renyi-filter) says such costs still add up as if fixed, on one condition: their total must stay below a budget fixed in advance. Write $P$ for the number of queries checked and $A$ for the number answered. With data-independent accounting, the total cost is
$$\varepsilon(\alpha) = \Big(\frac{P}{2\sigma_1^2} + \frac{A}{\sigma_2^2}\Big)\alpha.$$ {#eq-pate-confident}

The queries that pass the check are the ones with a strong consensus, which are cheap under the data-dependent analysis and more likely to be labelled correctly. With $\sigma_1$ much larger than $\sigma_2$ the check is cheap, and it skips the expensive, likely-wrong queries. The price is that the student never sees the hard inputs near the decision boundary.

Papernot et al. used $T = 200$, $\sigma_1 = 150$ and $\sigma_2 = 40$ on MNIST, and @tbl-pate-confident runs the 250 teachers of the running example with the same settings.

| Pool $P$ | Answered $A$ | Label accuracy | Data-independent $\varepsilon$ | Data-dependent $\varepsilon$ (not sanitized) | Student accuracy |
| --- | --- | --- | --- | --- | --- |
| 1,000 | 459 | 0.8693 | 3.59 | 2.82 | 0.8894 |
| 4,000 | 1,773 | 0.8917 | 7.87 | 6.28 | 0.9043 |

: Confident-GNMax with 250 teachers, $T = 200$, $\sigma_1 = 150$, $\sigma_2 = 40$, on the running example; the student is trained on the answered queries only. {#tbl-pate-confident}

At a data-independent $\varepsilon$ of 3.59, Confident-GNMax trains a student to 0.8894. That is about what plain GNMax reached at $\varepsilon = 4$, so it improves on GNMax a little, but not enough to catch DP-SGD at 0.9573. For the pool of 1,000, the check adds only 0.022 to the slope of @eq-pate-confident, while the answers add 0.287.

## Private in-context learning {#sec-pate-icl}

An LLM can learn a task from labelled examples placed in its prompt, called **exemplars**, without any training. This **in-context learning** has its own privacy problem. If the exemplars are private records, every answer is computed from them, and an attacker who can query the prompted model can learn about them. Duan, Dziedzic, Papernot and Boenisch (NeurIPS 2023) demonstrate this with a membership inference attack (MIA). When GPT-3 is prompted with one private example, its probability for the correct label is markedly higher on that example than on other records.

Duan et al. and Wu et al. (ICLR 2024) propose the same fix, which is private prediction with prompts as teachers. Split the private exemplars into disjoint groups by a fixed rule, such as a hash. Prompt the model once per group, with that group's exemplars and the query, and release the GNMax label of the $k$ answers. Adding or removing an exemplar changes one group's prompt, so it moves at most one vote, whatever the LLM computes. Each answer therefore costs $\alpha/\sigma^2$, exactly as before.

Wu et al. add one more step. For each query, they first draw a fresh Poisson sample of the exemplars, keeping each record independently with a probability called the sampling rate, written $r$. They then split the sample into groups by the fixed rule. A record takes part in only a fraction $r$ of the queries on average. So each answer costs what the subsampled Gaussian of [Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md) costs, the accountant DP-SGD uses.

That accountant is stated for a shift of 1 and takes a noise multiplier, the noise's standard deviation divided by the shift. One changed vote moves the histogram by $\sqrt2$, so the multiplier here is $\sigma/\sqrt2$. Write $\varepsilon_{r, s}(\alpha)$ for the RDP curve of the subsampled Gaussian with sampling rate $r$ and noise multiplier $s$. Since the $Q$ answers use independent samples, their costs add up, and the total is
$$\varepsilon(\alpha) = Q\,\varepsilon_{r,\,\sigma/\sqrt2}(\alpha).$$ {#eq-pate-icl}
Within one query, though, the groups must still be disjoint, or a record placed in two groups would move two votes.

**PromptPATE** (Duan et al.) turns the answers into a public prompt, as PATE turns them into a student. It labels a few public inputs with Confident-GNMax over an ensemble of prompted models, and uses them as the exemplars of one prompt. That prompt contains only public inputs and private labels, so it is post-processing of the labels and can be used without further cost, however often it is queried.

Free text needs another way of combining the answers than a vote. Wu et al. map each group's answer into a space where averaging makes sense, such as sentence embeddings with Gaussian noise or privately selected keywords. They then ask the model to write an answer from that private aggregate.

In-context learning is the setting where private prediction is strongest. A prompted LLM learns well from very few examples, because most of what it knows comes from pretraining. Its errors with different exemplars are correlated, so its votes agree far more than those of independently trained small models. High consensus is exactly what makes the answers accurate and the data-dependent costs small.

The papers report numbers that show it, on a movie-review sentiment task with GPT-3 models. With PromptPATE, Duan et al. report 92.7% accuracy at $\varepsilon = 0.147$ (with $\delta = 10^{-6}$), close to the 95.2% of a non-private prompt. Check in the paper whether that $\varepsilon$ is a data-dependent one, which would not yet be a guarantee. Wu et al. answer 10,000 queries, each by a vote of ten prompts with four exemplars each, and report 91.97% accuracy at $\varepsilon = 1$. That is about as good as a single non-private prompt with four exemplars, at 91.89%, and far better than the model with no exemplars at all, at 86.58%.

### A simulated in-context learner

The full edition's script has no language model, so it simulates the mechanics with a stand-in. The stand-in adds up its exemplars' inputs, each multiplied by its label of $\pm1$, and labels a query by which side of that sum it falls on. It is a linear classifier with no pretraining, far weaker than an LLM, so what carries over is the accounting and the dependence on agreement, not the accuracy.

Without noise, the stand-in leaks its exemplars to an attacker who scores a candidate record by the learner's confidence in its correct label. The attack's area under the curve (AUC), the probability that a random member scores above a random non-member, is 1.000 against a learner with one exemplar. It is 0.996 against a learner with 4 exemplars, and 0.889 with 16. So a prompt with few exemplars reveals them almost perfectly, as Duan et al. found for GPT-3.

For private answers, each query draws a Poisson sample of about 400 records, at rate $r = 0.04$, and splits it into 25 groups of about 16. A single learner with 16 exemplars is right on 0.6232 of the evaluation set. The noiseless vote of 25 such learners does much better, at 0.8090. @tbl-pate-icl adds the noise for a budget spread over $Q$ queries, from @eq-pate-icl.

| Budget | Answers $Q$ | $\sigma$ | Accuracy |
| --- | --- | --- | --- |
| $\varepsilon = 1$ | 100 | 2.75 | 0.7780 |
| | 1,000 | 7.39 | 0.6800 |
| | 10,000 | 22.93 | 0.5615 |
| $\varepsilon = 4$ | 100 | 1.30 | 0.8080 |
| | 1,000 | 2.36 | 0.7910 |
| | 10,000 | 6.65 | 0.7025 |

: Private in-context learning with the stand-in learner: 25 groups of about 16 exemplars per query, Poisson sampling rate 0.04, GNMax noise for $Q$ answers at $(\varepsilon, 10^{-5})$, accuracy on 2,000 evaluation queries. {#tbl-pate-icl}

Subsampling makes the noise small. For 1,000 answers at $\varepsilon = 1$, it needs $\sigma = 7.39$ here, against 181 without subsampling in @tbl-pate-noise. But the vote has only 25 voters, so even that noise is large against its gaps (@fig-pate-icl).

![Accuracy of private answers from the stand-in in-context learner against the number of queries answered at $\varepsilon = 1$ and $\varepsilon = 4$ ($\delta = 10^{-5}$), with 25 groups of about 16 exemplars drawn per query at rate 0.04. The dashed line is the vote without noise and the dotted line a single 16-shot learner.](figures/pate_icl.svg){#fig-pate-icl}

A student prompt works as PromptPATE does. The script splits all the records among 625 disjoint learners of 16 exemplars each, whose noiseless vote is right on 0.9366 of the evaluation set. It labels public inputs with their GNMax vote at $\varepsilon = 1$ and uses them as the exemplars of one public learner. The student prompt improves as it gets more labelled inputs, and with 256 of them it reaches 0.7954. That is about the accuracy of 100 private answers at the same budget, but now for any number of queries.

## Code {#sec-pate-code}

This snippet answers the two questions to ask before deploying a noisy vote. The first is how much noise $Q$ answers need, from @eq-pate-sigma, with the largest slope found by searching the RDP conversion over orders. The second is how often a two-label vote with a given gap survives that noise, from @eq-pate-two-class.

```python
from math import erfc, log, sqrt


def eps_of_line(rho: float, delta: float) -> float:
    """(eps, delta) of the RDP line rho * alpha, by the conversion of Balle et al. (2020) at the best order."""
    orders = [1 + j / 100 for j in range(1, 6_000)]
    return min(rho * a + log((a - 1) / a) - (log(delta) + log(a)) / (a - 1) for a in orders)


def largest_slope(eps: float, delta: float) -> float:
    """The largest rho whose RDP line rho * alpha converts to (eps, delta), by bisection."""
    lo, hi = 1e-6, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if eps_of_line(mid, delta) <= eps else (lo, mid)
    return lo


def gnmax_sigma(answers: int, eps: float, delta: float) -> float:
    """Noise per vote count so that `answers` GNMax answers cost (eps, delta) in total."""
    return sqrt(answers / largest_slope(eps, delta))


def p_plurality(gap: float, sigma: float) -> float:
    """Probability that a two-label GNMax vote returns the plurality label."""
    return 1 - 0.5 * erfc(gap / (2 * sigma))


print(f"{largest_slope(1.0, 1e-5):.4f}")  # 0.0306
print(f"{gnmax_sigma(100, 1.0, 1e-5):.1f}")  # 57.2
print(f"{gnmax_sigma(10_000, 1.0, 1e-5):.0f}")  # 572
sigma = gnmax_sigma(1000, 4.0, 1e-5)
print(f"{sigma:.1f}")  # 51.8
print(f"{p_plurality(96, sigma):.4f}")  # 0.9051
print(f"{p_plurality(4, sigma):.4f}")  # 0.5218
```

The full edition's script trains the teachers, the students and the DP-SGD baselines. It also implements the data-dependent accountants and Confident-GNMax, and simulates private in-context learning:

```bash
uv run python code/part05-dp-ml/private_prediction.py
```

## Cheat sheet

| Result | Formula | Conditions and notes |
| --- | --- | --- |
| One record, one vote | Histogram moves by $\sqrt2$ in Euclidean length | Shards fixed by a hash, not by position |
| LNMax | $(2/b)$-DP per answer | Release the label only; best for one or two answers |
| GNMax | $(\alpha, \alpha/\sigma^2)$-RDP per answer | Release the label only; adaptive queries allowed |
| Noise for $Q$ answers | $\sigma = \sqrt{Q/\rho(\varepsilon, \delta)}$; $\rho = 0.0306$ for $\varepsilon = 1$, 0.3731 for $\varepsilon = 4$, at $\delta = 10^{-5}$ | Does not depend on $k$ |
| Accuracy, two labels | $\Phi(g/(\sqrt2\sigma))$ | Exact; capped by the noiseless vote |
| Failure, many labels | $\tilde q = \sum_{i\ne i^*}\frac12\operatorname{erfc}((n_{i^*} - n_i)/(2\sigma))$ | Union bound; loose when many labels are near a tie |
| Break-even | Answers at least as accurate as a private model for $Q \le Q^*(\varepsilon)$ | Same budget; on the running example 58 at $\varepsilon = 0.5$, none from $\varepsilon = 2$ |
| PATE student | Costs its $Q$ labels; its predictions are free | Public unlabelled inputs; teachers stay secret |
| Data-dependent cost | About proportional to the chance of overturning the plurality | Conditions of Theorem 6 on $\tilde q$ and $\alpha$; not a guarantee until released privately |
| Confident-GNMax | $(P/(2\sigma_1^2) + A/\sigma_2^2)\alpha$ | Budget fixed in advance |
| Private in-context learning | $Q\,\varepsilon_{r,\sigma/\sqrt2}(\alpha)$ | Poisson rate $r$; disjoint groups within each query |

: The results of this chapter. {#tbl-pate-cheat}

**Rules of thumb**

- Count your queries before choosing between private answers and a private model.
- Add teachers as long as their vote gaps grow.
- Use Gaussian noise for more than a handful of answers, and Laplace noise only for one or two.
- A teacher ensemble is only as good as its noiseless vote, so measure that ceiling first.
- Report the data-independent $\varepsilon$, or pay for a private release of the data-dependent one.
- Never put private exemplars in a prompt that answers users directly. Vote over disjoint groups with noise, or distil the answers into a public prompt.

## Further reading

- Papernot, Song, Mironov, Raghunathan, Talwar & Erlingsson, "Scalable Private Learning with PATE" (ICLR 2018): GNMax, the data-dependent analysis, Confident-GNMax and how to release the data-dependent $\varepsilon$ privately.
- Papernot, Abadi, Erlingsson, Goodfellow & Talwar, "Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data" (ICLR 2017): the original PATE, with LNMax and semi-supervised students.
- Duan, Dziedzic, Papernot & Boenisch, "Flocks of Stochastic Parrots" (NeurIPS 2023), and Wu, Panda, Wang & Mittal, "Privacy-Preserving In-Context Learning for Large Language Models" (ICLR 2024): the attack on prompts, PromptPATE, and subsampled voting over exemplar groups, including text generation.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/05-private-prediction.md), except three things. It states the Gaussian data-dependent bound (Papernot et al. 2018, Theorem 6) with its conditions and cites the proof, it describes and cites the private release of the data-dependent $\varepsilon$ (their Appendix B), and it cites the results quoted from the papers.
