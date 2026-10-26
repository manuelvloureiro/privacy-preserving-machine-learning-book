---
title: "AIM, Explained From Scratch"
subtitle: "How a differentially private synthetic-data mechanism chooses, measures and fits marginals"
author: "Technical study notes"
date: "23 October 2026"
abstract: |
  AIM is one of the best algorithms for releasing a synthetic copy of a table of personal records under differential privacy. This note explains it step by step, for a reader who knows basic probability and nothing about privacy or graphical models. It defines the task, the privacy guarantee and the two noisy building blocks AIM is made of. It then explains Private-PGM, the engine that turns a pile of noisy, mutually inconsistent statistics into one dataset, without the machinery its name suggests. The second half walks through AIM round by round on a toy dataset of a thousand people, with every number computed by a script, and proves that the whole procedure keeps its privacy promise. The results of the paper's experiments close the note.
---

An organization holds a table of records about people, one row each, with attributes such as age group, occupation and a medical condition. Researchers want to study the table, but it cannot leave the building. One answer is **synthetic data**, a new table of made-up rows. It is built to give nearly the same answers as the real one to the statistical questions that matter, while revealing almost nothing about any one real person. AIM, by McKenna, Mullins, Sheldon and Miklau (2022), is a way of building such a table. In the paper's experiments it beat every competitor on most settings.

The name stands for *adaptive and iterative mechanism*, and the three words say what it does. It is a mechanism, which in this field means a randomized algorithm whose privacy is analysed. It is iterative, because it builds its model of the data a piece at a time, each time choosing one small statistic to measure with noise. And it is adaptive, because each choice depends on what the earlier measurements taught it, and because it spends more privacy budget on a measurement when it finds that the earlier, cheaper ones taught too little.

Three questions organize the note. What does AIM promise, and how is that promise counted? How does a pile of noisy statistics become one dataset? And how does AIM decide what to measure next, and with how much noise? You need only basic probability to follow it: expectations, the normal distribution and a little calculus. Everything else is defined where it is first used. @tbl-map lists the question each section answers, and @tbl-symbols lists the symbols for reference. A script carries the running example, and every number the note quotes comes from it:

```bash
uv run python code/notes/aim.py
```

| Section | The question it answers |
|:---|:---|
| @sec-task | What does synthetic data have to get right, and how is its error measured? |
| @sec-dp | What does the privacy promise say, and how are its costs counted? |
| @sec-smg | Why not measure every statistic directly, and what three problems that leaves? |
| @sec-pgm | How does one dataset come out of many noisy, inconsistent statistics? |
| @sec-aim | What does AIM do in each round, and why? |
| @sec-privacy | Why does AIM keep its privacy promise? |
| @sec-error | How does AIM know how wrong its answers are? |
| @sec-results | How well does it work, and which of its ideas matter most? |

: The question each section answers. {#tbl-map}

| Symbol | Meaning |
|:---|:---|
| $D$, $D'$; $D \sim D'$ | a dataset of $n$ records, and a neighbouring one, which has one record more or one fewer |
| $d$ | the number of attributes |
| $r$, $s$ | sets of attributes |
| $\mu_r(D)$, $m_r$ | the marginal of $D$ over the attributes in $r$, a vector of counts, and its number of cells |
| $\mathcal Q$, $c_s$ | the workload, a list of attribute sets whose marginals the synthetic data should get right, with their weights |
| $\mathcal Q_+$ | the downward closure of the workload, the marginals AIM may measure |
| $w_r$ | the weight AIM gives a candidate marginal $r$ |
| $\hat p$, $\hat p_t$ | the model's table of counts, and the model after round $t$ |
| $\hat D$ | the synthetic dataset |
| $\varepsilon$, $\delta$ | the parameters of a differential-privacy guarantee |
| $\rho$, $\rho_{\mathrm{used}}$ | the privacy budget in zero-concentrated form, and the part of it spent so far |
| $\sigma_t$, $\varepsilon_t$ | the noise level and the selection budget of round $t$; $\sigma_0$ is the starting noise level |
| $T$ | the number of rounds AIM plans for |
| $u(D, r)$, $\Delta u$ | the score AIM gives candidate $r$, and the most one record can change it |
| $D_\alpha(P \| Q)$ | the Rényi divergence of order $\alpha$ between two distributions |
| $\theta$ | the parameters of the model |

: The symbols the note uses. Each is also explained where it first appears. {#tbl-symbols}

# The task: synthetic data that gets the marginals right {#sec-task}

Before any privacy enters, we need to say what a good synthetic dataset is. This section fixes the shape of the data, defines the statistics that AIM works with, called marginals, and states the error that the paper measures. The running example of the note is introduced here too.

## Records, attributes and the full table

A dataset $D$ is a list of $n$ records, and each record gives a value for each of $d$ attributes. AIM requires every attribute to be **discrete**, which means that it takes one of a fixed, finite list of values. An attribute such as income has to be cut into bands first.

The running example is a toy dataset of $n = 1{,}000$ people with four attributes. *Smoker* has the values no and yes, and *exercise* the values rarely and often. *Age* has three bands, under 40, 40 to 64, and 65 and over. *Condition* records whether the person has a certain medical condition. The script draws the dataset from a fixed population in which smoking, exercise and the condition all depend on age, and the condition also depends on smoking, so there is structure for AIM to find.

A record is one combination of values, and the number of possible combinations is the product of the attributes' sizes. For the toy dataset that is $2 \times 3 \times 2 \times 2 = 24$ combinations. The **full table** of a dataset counts how many records fall on each combination. It holds everything there is to know about the dataset, apart from the order of the rows, and the whole of this note can be read as the question of how to release a noisy version of it. For real data it is far too big to release or even to store, because the product of fifteen attribute sizes runs into the billions, and most of its cells hold zero or one person. That is why AIM works with marginals instead.

## Marginals

A marginal looks at a few attributes at a time and ignores the rest.

::: {#def-marginal name="Marginal"}
Let $r$ be a set of attributes. The **marginal** of $D$ over $r$, written $\mu_r(D)$, is the vector that gives, for every combination of values of the attributes in $r$, the number of records of $D$ with that combination. Its number of entries, or **cells**, is written $m_r$. A marginal over one attribute is a **one-way** marginal, over two attributes a **two-way** marginal, and so on.
:::

So a marginal is a contingency table, flattened into a vector. @tbl-marginal shows the toy dataset's marginal over smoker and condition, which has $m_r = 4$ cells.

{{< include figures/_marginal.md >}}

Two facts about marginals drive everything that follows. First, a marginal over a set $r$ determines the marginals over every subset of $r$, by adding up cells: the row sums of @tbl-marginal are the one-way marginal of smoker. Second, every marginal is a sum over the full table. Summing the full table over the attributes outside $r$ gives $\mu_r(D)$, and this is a linear operation, which is what lets @sec-pgm fit a table to noisy marginals.

## The workload and the error

Whoever asks for the synthetic data says which marginals they care about. The **workload** $\mathcal Q$ is a list of attribute sets. Each set $s$ in it comes with a weight, written $c_s$, that says how much that marginal matters. In the toy example the workload is all six two-way marginals, with equal weights. In the paper's experiments it is typically every three-way marginal of a dataset, several hundred of them.

The synthetic dataset $\hat D$ is a dataset of the same shape as $D$, so its marginals are computed in the same way. Its quality is the weighted distance between its workload marginals and the true ones. The paper measures distance by the $L_1$ norm, the sum of the absolute differences of the cells, and divides by the number of marginals and by $n$:

$$\mathrm{Error}(D, \hat D) = \frac{1}{|\mathcal Q|\, n} \sum_{s \in \mathcal Q} c_s \,\bigl\|\mu_s(D) - \mu_s(\hat D)\bigr\|_1.$$ {#eq-error}

Dividing by $n$ turns counts into shares of the population, so an error of $0.05$ means that a typical workload marginal is off by five percent of the records in total, spread over its cells. The toy run in @sec-aim starts at about $0.14$ and ends at about $0.05$.

# Differential privacy, as much as AIM needs {#sec-dp}

AIM promises that the synthetic data reveal almost nothing about any one person, a promise called differential privacy (DP). This section says exactly what that means, introduces the way of counting privacy costs that AIM uses, and presents the only two noisy operations AIM ever performs on the data. Everything AIM does to the data is one of these two operations, so once their costs are known, the privacy of the whole algorithm is an exercise in adding them up.

## Neighbouring datasets and the promise

The promise is stated by comparing two datasets that differ in one person.

::: {#def-neighbours name="Neighbouring datasets"}
Two datasets $D$ and $D'$ are **neighbouring**, written $D \sim D'$, if one can be obtained from the other by adding or removing a single record.
:::

Some papers instead call datasets neighbouring when one record is replaced by another. AIM uses the add-or-remove version, and the choice matters for the noise levels below, because adding a record changes a marginal in one cell, while replacing one changes two cells.

A **mechanism** $M$ is a randomized algorithm that takes a dataset and returns an output, here a synthetic dataset. Because it is random, its output on $D$ has a probability distribution, and the promise is that this distribution barely changes when $D$ is replaced by a neighbour.

::: {#def-dp name="Differential privacy"}
Let $\varepsilon \ge 0$ and $\delta \in [0, 1]$. A mechanism $M$ is **$(\varepsilon, \delta)$-differentially private**, or $(\varepsilon, \delta)$-DP, if for every pair of neighbouring datasets $D \sim D'$ and every set $S$ of possible outputs,
$$\Pr[M(D) \in S] \le e^{\varepsilon}\,\Pr[M(D') \in S] + \delta.$$
:::

In words, no outcome becomes much more likely because one person's record is in the data. For small $\varepsilon$ the factor $e^\varepsilon$ is about $1 + \varepsilon$, so $\varepsilon = 1$ allows an outcome to become at most about 2.7 times more likely, and $\delta$ is a small probability of exceptions, typically far below one in a million. Since the definition covers both orders of the pair, it also bounds the other direction. An observer who sees the output therefore cannot tell with any confidence whether a given person's record was used, whatever else they know.

## Counting the cost in concentrated form

AIM performs many noisy operations, and the privacy of the whole must be computed from the privacy of the parts. Adding up $(\varepsilon, \delta)$ pairs directly gives weak totals, so AIM counts in a different currency, called zero-concentrated differential privacy, in which costs simply add and convert to an $(\varepsilon, \delta)$ pair at the end. Its definition uses a way of measuring the distance between two distributions.

::: {#def-renyi name="Rényi divergence"}
Let $P$ and $Q$ be two distributions on the same outputs, and write $p$ and $q$ for their densities, or their mass functions if the outputs are discrete. Let the order $\alpha$ be a number greater than $1$. Their **Rényi divergence of order $\alpha$** is
$$D_\alpha(P \| Q) = \frac{1}{\alpha - 1} \ln \mathbb E_{Y \sim Q}\!\left[\left(\frac{p(Y)}{q(Y)}\right)^{\alpha}\right].$$
:::

The ratio $p(y)/q(y)$ says how much more likely an output is under $P$ than under $Q$, and the divergence is an average of a power of that ratio, on a logarithmic scale. It is zero when the distributions are equal and grows as they separate, and a higher order $\alpha$ weights the outputs where the ratio is largest more heavily.

::: {#def-zcdp name="Zero-concentrated differential privacy"}
Let $\rho \ge 0$. A mechanism $M$ is **$\rho$-zero-concentrated differentially private**, or $\rho$-zCDP, if for every pair of neighbouring datasets $D \sim D'$ and every order $\alpha > 1$,
$$D_\alpha\bigl(M(D) \,\|\, M(D')\bigr) \le \rho\,\alpha.$$
:::

The number $\rho$ is the **privacy budget** in this currency. The three facts AIM needs about it are standard, and the note cites their proofs.

::: {#prp-zcdp name="Three properties of zCDP"}
(a) *Composition.* If $M_1$ is $\rho_1$-zCDP and $M_2$ is $\rho_2$-zCDP, where $M_2$ may be chosen after seeing the output of $M_1$, then releasing both outputs is $(\rho_1 + \rho_2)$-zCDP.
(b) *Post-processing.* Any computation on the output of a $\rho$-zCDP mechanism, which does not look at the data again, is still $\rho$-zCDP.
(c) *Conversion.* A $\rho$-zCDP mechanism is $(\varepsilon, \delta)$-DP for every $\varepsilon \ge 0$ with
$$\delta = \min_{\alpha > 1} \frac{\exp\bigl((\alpha - 1)(\alpha\rho - \varepsilon)\bigr)}{\alpha - 1}\left(1 - \frac1\alpha\right)^{\alpha}.$$
:::
<!-- proof: cited -->

Bun and Steinke (2016) prove parts (a) and (b), and Canonne, Kamath and Steinke (2020) prove part (c). All three proofs work with the Rényi divergence directly and need nothing beyond its definition. Part (a) is the whole reason for the currency: the costs of AIM's rounds add, even though each round's choices depend on the previous rounds' outputs. Part (b) is what makes most of AIM free. Fitting a model to released noisy marginals, deciding what to measure next from that model, and rounding the model to a dataset all look only at released values, so they cost nothing. Part (c) is how a budget is set. The user picks $\varepsilon$ and $\delta$, and the script finds the largest $\rho$ whose conversion gives them. For $\varepsilon = 1$ and $\delta = 10^{-6}$, the values the toy run uses, the budget is $\rho = 0.0244$.

## The two building blocks

The first building block releases a vector of counts with Gaussian noise. What it costs depends on how much one person can change the vector.

::: {#def-sensitivity name="Sensitivity"}
Let $f$ map datasets to vectors. Its **$L_2$ sensitivity** is the largest Euclidean distance between its values on two neighbouring datasets,
$$\Delta_2 f = \max_{D \sim D'} \bigl\| f(D) - f(D') \bigr\|_2.$$
:::

A marginal has the smallest sensitivity a count can have.

::: {#lem-marginal-sensitivity name="A marginal has sensitivity one"}
For every attribute set $r$, the marginal $\mu_r$ has $L_2$ sensitivity $1$.
:::

::: {.proof}
Adding one record to $D$ adds one to exactly one cell of $\mu_r(D)$, the cell of that record's values on the attributes in $r$, and leaves every other cell unchanged. Removing a record subtracts one from one cell. In both cases the difference vector has a single entry of $\pm 1$, whose Euclidean length is $1$.
:::

::: {#prp-gaussian name="The Gaussian mechanism"}
Let $f$ have $L_2$ sensitivity $\Delta_2 f$, and let $\sigma > 0$. The mechanism that releases $f(D) + Z$, where $Z$ is a vector of independent $\mathcal N(0, \sigma^2)$ noise, one entry per coordinate, is $\rho$-zCDP with
$$\rho = \frac{(\Delta_2 f)^2}{2\sigma^2}.$$
:::
<!-- proof: cited -->

Bun and Steinke (2016) prove this for their Gaussian mechanism proposition. The proof computes the Rényi divergence between two normal distributions with the same variance, which is $\alpha$ times the squared distance of their means over $2\sigma^2$, and bounds the distance by the sensitivity. For a marginal, by @lem-marginal-sensitivity, a measurement with noise level $\sigma$ costs $1/(2\sigma^2)$, whatever the marginal's size. That independence from the size is deceptive. A marginal with many cells spreads the same $n$ records thinly, so each cell's count is small compared with the noise, which is why @sec-aim will prefer smaller marginals when the budget is tight.

The second building block chooses one option from a list, favouring options with a high score, without revealing the scores.

::: {#prp-exponential name="The exponential mechanism"}
Let $\mathcal R$ be a finite set of candidates, and let $u(D, r)$ be a score for each candidate $r$, whose **sensitivity** $\Delta u$ is the most that adding or removing one record can change any candidate's score. The mechanism that chooses candidate $r$ with probability proportional to $\exp\bigl(\varepsilon\, u(D, r)/(2\Delta u)\bigr)$ is $\varepsilon$-DP, and it is $\rho$-zCDP with
$$\rho = \frac{\varepsilon^2}{8}.$$
:::
<!-- proof: cited -->

The $\varepsilon$-DP claim is the original theorem of McSherry and Talwar (2007). The zCDP cost is proved by Cesar and Rogers (2021), from the fact that the mechanism's privacy loss is not only bounded by $\varepsilon$ but confined to an interval of length $\varepsilon$. The exponent is what makes the mechanism useful. A candidate whose score is higher by $2\Delta u/\varepsilon$ is $e$ times more likely to be chosen. So with a large $\varepsilon$ the mechanism almost always picks the best candidate, and with a tiny $\varepsilon$ it picks nearly at random. @sec-aim shows both regimes on the toy dataset.

# Select, measure, generate {#sec-smg}

Why does AIM not simply measure every marginal in the workload with Gaussian noise and release the results? This section works out what that costs, and finds the three problems that any better method has to solve. AIM belongs to a family of methods, which the paper calls *select-measure-generate*, that solve them in a common pattern.

Suppose the workload has $k$ marginals and each is measured once with the same noise level $\sigma$. By @prp-gaussian and @prp-zcdp (a), the total cost is $k/(2\sigma^2)$, so spending the whole budget gives $\sigma = \sqrt{k/(2\rho)}$. On the toy workload, with $k = 6$ and $\rho = 0.0244$, that is $\sigma = 11.1$, which is small next to counts in the hundreds. On a workload of all three-way marginals of fifteen attributes, with $k = 455$, it is $\sigma = 96.6$. A three-way marginal of a dataset with fifty thousand records might have a hundred cells averaging five hundred records each, so the noise would be a fifth of a typical count, and far more than most of the smaller cells. The budget, in short, does not stretch to every marginal at once. That is the first problem, **selection**: measure only the marginals that matter, and let the rest be inferred from them.

The second problem appears even on the toy workload. The six noisy marginals are not a dataset. Two of them that share an attribute will disagree about its one-way marginal, because each carries its own noise. Some cells will be negative, which no count can be. And a researcher who wants a marginal that was never measured, such as a three-way one, gets no answer at all. This is the problem of **generation**: turn the noisy measurements into one dataset whose marginals are all consistent, non-negative, and close to the measurements. @sec-pgm is about this problem alone.

The third problem is the **budget split**. A measurement with little noise costs a great deal, and a measurement with much noise may be worthless. Spreading the budget evenly over the measurements, as the calculation above did, ignores the fact that some marginals are far from what the model already knows and others are not.

The select-measure-generate family answers the three problems in a loop. Select a marginal, measure it with noise, update a model of the data to agree with every measurement so far, and repeat, then generate a dataset from the model at the end. The ancestor of AIM in this family is a method called the multiplicative weights exponential mechanism (MWEM), whose selection step picks the marginal on which the current model is most wrong, by the exponential mechanism, and whose model is updated by a simple multiplicative rule. AIM keeps the loop and changes every step in it, as @sec-aim explains. But first the generation step, which both share.

# Private-PGM, without graphical models {#sec-pgm}

The name Private-PGM promises probabilistic graphical models, and the paper that introduced it (McKenna, Sheldon and Miklau, 2019) uses their full machinery. This section explains what the method computes without that machinery, because what it computes is simple. It then says, in words, why the machinery is needed to compute it at scale, and what the one number that AIM takes from it, the size of the model, means.

## One table that explains every measurement

Suppose that a sequence of marginals has been measured. The $i$-th measurement is a noisy vector $\tilde y_i$, equal to $\mu_{r_i}(D)$ plus Gaussian noise of level $\sigma_i$ in each cell. We want one dataset that agrees with all of them. Private-PGM's answer is to look for a full table $p$, a vector of non-negative numbers over all combinations of values whose entries sum to the number of records, and to ask that its marginals be as close to the measurements as possible. A table's marginal $\mu_r(p)$ is computed exactly as a dataset's, by summing over the attributes outside $r$. The closeness is measured by squared error, weighted so that noisier measurements count for less:

$$\hat p = \arg\min_{p} \sum_{i} \frac{1}{\sigma_i} \bigl\| \mu_{r_i}(p) - \tilde y_i \bigr\|_2^2.$$ {#eq-pgm}

Squared error is the natural choice because the noise is Gaussian. If the weights were $1/\sigma_i^2$, the minimizer would be exactly the maximum-likelihood table, the one under which the observed measurements were most probable. The paper uses $1/\sigma_i$, which still weights the precise measurements more and worked better in practice.

Two problems from @sec-smg vanish at once. The marginals of one table cannot disagree with each other, because they are all sums of the same numbers, and they cannot be negative, because the table is not. The third benefit is quieter. A table answers every marginal, including the ones never measured, by combining what the measured ones imply. @tbl-init shows how. After the toy run has measured only the four one-way marginals, the model's marginal over smoker and condition keeps the measured row and column totals and spreads them as if the two attributes were independent, because nothing measured says otherwise.

{{< include figures/_init.md >}}

The number of records $n$ needs one remark. Under add-or-remove neighbours, $n$ is itself private, since it changes between neighbouring datasets, so the table's total cannot be set to $n$. Each noisy one-way marginal sums to a noisy copy of $n$, and the script uses the average of these sums, a post-processing step that costs nothing. The toy run estimates $1{,}019$ records.

## Why the table never has to be built

The difficulty is size. On the toy domain, $p$ has 24 entries, and the script simply searches over them. On a real domain with billions of combinations, no computer can hold $p$, let alone optimize it. The observation that rescues @eq-pgm is that the objective depends on $p$ only through its marginals over the measured sets $r_1, r_2, \dots$, and that forces the shape of the solution.

::: {#thm-pgm name="The solution is a product of small tables"}
Write $x_{r_i}$ for the combination of values that a record $x$ takes on the attributes in $r_i$. Among the minimizers of @eq-pgm there is a table of the form
$$p_\theta(x) \propto \exp\Bigl( \sum_i \theta_i[x_{r_i}] \Bigr),$$
where each $\theta_i$ is a table of numbers with one entry per cell of the $i$-th measured marginal.
:::
<!-- proof: cited -->

McKenna, Sheldon and Miklau (2019) prove this by a convex duality argument. The objective of @eq-pgm depends on the table only through linear functions of it, the marginals, and the maximum-entropy table with given marginals has this exponential form. The practical content is that the model has as many parameters as the measurements have cells, however large the domain. Fitting it means adjusting the parameter tables $\theta_i$ until the model's marginals match the measurements, which is what the script's optimizer does, and what Private-PGM does with a more careful algorithm.

The one-way case shows the form at work. With only one-way marginals measured, the model is proportional to $\exp(\theta_1[x_1]) \exp(\theta_2[x_2]) \cdots$, a product of one factor per attribute, which is the definition of independent attributes. That is why @tbl-init shows an independence table. After the marginal over smoker and condition is measured, the model gains a factor that depends on both attributes, and only that pair becomes dependent. In general, the model carries exactly the dependencies that were measured and no others, which is the most cautious reading of the measurements.

What the machinery of graphical models adds is a way to compute the marginals of $p_\theta$ without ever writing out $p_\theta$. The sum over all combinations of a product of factors can be reorganized attribute by attribute, summing out one attribute at a time, and the cost is set by the largest intermediate table that this process has to form. That largest table is governed by how the measured attribute sets overlap. Sets that form chains and small clusters keep it small, while many sets that overlap in a tangled way make it enormous. The paper calls the memory the computation needs the **junction tree size**, written $\mathrm{JT\text{-}SIZE}(r_1, \dots, r_t)$ for the measured sets, in megabytes, and it is the one thing from the graphical-model world that AIM reads. @sec-aim shows how AIM keeps it below a limit.

## From the table to records

The fitted table $\hat p$ has fractional entries, and a dataset needs whole records. Private-PGM produces the synthetic dataset $\hat D$ by rounding. Each cell's count is rounded down, and the records lost to rounding are given back to the cells with the largest fractional parts, with a random tie-break in the original. Sampling records from $\hat p$ would also work, but the rounding adds less error. By @prp-zcdp (b) this step is free. In the toy run, the model's workload error is $0.0529$ and the rounded dataset's is $0.0537$.

# AIM, round by round {#sec-aim}

This is the heart of the note. AIM takes a dataset $D$, a workload $\mathcal Q$ with weights $c_s$, and a budget $\rho$, and returns a synthetic dataset. It has three settings, which the paper fixes once and for all. It plans for $T = 16d$ rounds, where $d$ is the number of attributes. It gives nine tenths of each round's budget to the measurement and one tenth to the selection. And it caps the model's junction tree size at 80 megabytes. The sections below follow the algorithm in order, and @tbl-rounds at the end shows every round on the toy dataset.

## The schedule

Each round $t$ selects a marginal with the exponential mechanism at a budget $\varepsilon_t$ and measures it with Gaussian noise at a level $\sigma_t$. By @prp-exponential, @prp-gaussian and @lem-marginal-sensitivity, the round costs

$$\rho_t = \frac{\varepsilon_t^2}{8} + \frac{1}{2\sigma_t^2}.$$ {#eq-round-cost}

AIM chooses the starting values so that $T$ rounds at those values would spend the budget in the planned split. Nine tenths of $\rho$ over $T$ measurements means $T/(2\sigma_0^2) = 0.9\rho$, and one tenth over $T$ selections means $T\varepsilon_1^2/8 = 0.1\rho$. Solving each for its unknown gives

$$\sigma_0 = \sqrt{\frac{T}{2 \cdot 0.9\,\rho}}, \qquad \varepsilon_1 = \sqrt{\frac{8 \cdot 0.1\,\rho}{T}}.$$ {#eq-start}

These are deliberately cautious. With $T = 16d$ the starting noise is large, and the starting selection budget is tiny. The schedule is only a starting point, because the annealing rule below raises the per-round budget whenever a round turns out to teach too little, and AIM stops when the budget is gone rather than after $T$ rounds. @tbl-schedule gives the schedule for the toy dataset, where $d = 4$.

{{< include figures/_schedule.md >}}

## Initialization

Before the loop, AIM measures every one-way marginal that appears in some workload marginal, each with noise level $\sigma_0$, and fits the model to them. The cost is at most $d/(2\sigma_0^2)$, which by @eq-start equals $0.9\rho\, d/T$, a sixteenth of the measurement budget when $T = 16d$. The reason is economy. The paper observed that MWEM's early rounds go to one-way marginals anyway, each paid for with a selection as well as a measurement, so measuring them outright saves the selections. The result is the independence model of @tbl-init, which already has every attribute's distribution about right.

## The candidates

Which marginals may AIM measure? Not only the workload's, but every marginal over a subset of a workload set. The paper calls this collection the **downward closure** of the workload,

$$\mathcal Q_+ = \{\, r : r \subseteq s \text{ for some } s \in \mathcal Q \,\}.$$

A large marginal spreads the records over many cells, so each cell's count is small and the noise swamps it. Its sub-marginals have fewer cells and bigger counts, and by @thm-pgm the model combines them into an estimate of the large one. When the budget is small, AIM therefore often does better to measure pieces of a workload marginal than the marginal itself. On the toy workload of six two-way marginals, the closure adds the four one-way marginals, for ten candidates in all.

Each candidate gets a weight that says how much of the workload it serves. The weight adds up, over the workload marginals, the number of attributes the candidate shares with each, times that marginal's weight:

$$w_r = \sum_{s \in \mathcal Q} c_s\, |r \cap s|.$$ {#eq-weight}

A two-way candidate on the toy workload shares both attributes with itself and one attribute with four other workload marginals, so its weight is $6$, while a one-way candidate shares its attribute with three workload marginals, for a weight of $3$.

Finally, in each round AIM drops the candidates that would make the model too big. A candidate $r$ stays in the round's candidate set only if the junction tree size of the measured sets plus $r$ is at most $(\rho_{\mathrm{used}}/\rho)$ times the 80 megabyte cap. The allowance grows with the fraction of the budget spent, so early rounds, which have measured little, are kept to small models, and the full cap is available only near the end. On the toy domain every model is tiny, so the script does not need this filter.

## Selection: the score

Here is AIM's first real idea. MWEM scores a candidate by how wrong the current model is on it, the $L_1$ distance $\|\mu_r(D) - \mu_r(\hat p_{t-1})\|_1$ between the true marginal and the model's. That rewards large marginals, which have large errors simply because they have many cells, even when measuring them would replace the model's error by a larger error from the noise. AIM subtracts what the noise would cost. The noise cost is the expected $L_1$ norm of the noise vector, which needs one lemma.

::: {#lem-l1-noise name="Expected size of Gaussian noise"}
If $Z \sim \mathcal N(0, \sigma^2)$, then $\mathbb E|Z| = \sigma\sqrt{2/\pi}$. Hence a vector of $m$ independent such entries has expected $L_1$ norm $\sigma m \sqrt{2/\pi}$.
:::

::: {.proof}
By symmetry of the density about zero, the expectation of $|Z|$ is twice the integral over the positive half-line:
$$\mathbb E|Z| = 2 \int_0^\infty \frac{z}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{z^2}{2\sigma^2}\right) \mathrm d z.$$
The integrand is, up to the constant in front, the derivative of $-\sigma^2 \exp(-z^2/(2\sigma^2))$, which runs from $-\sigma^2$ at zero to $0$ at infinity. So the integral equals $\sigma^2/(\sigma\sqrt{2\pi})$, and doubling it gives $2\sigma/\sqrt{2\pi} = \sigma\sqrt{2/\pi}$. The $L_1$ norm of a vector is the sum of the absolute values of its entries, and expectation is linear, so $m$ entries give $m$ times the single-entry value.
:::

The constant $\sqrt{2/\pi}$ is about $0.798$. AIM's score for candidate $r$ in round $t$ is then

$$u(D, r) = w_r \Bigl( \bigl\|\mu_r(D) - \mu_r(\hat p_{t-1})\bigr\|_1 - \sqrt{2/\pi}\;\sigma_t\, m_r \Bigr).$$ {#eq-score}

Read it as a balance in units of records. The first term is how far the model is from the truth on $r$, and the second is how far the noisy measurement would be from the truth, in expectation. A candidate earns a positive score only when measuring it is expected to leave the model closer to the truth than it is now, and the weight scales the balance by how much of the workload the candidate serves. The script's version is short:

```{.python excerpt="score"}
def score(
    data: Array,
    model: Array,
    attrs: Attrs,
    weight: float,
    sigma: float,
) -> float:
    """AIM's quality score of a candidate marginal: how far the model is from the data
    on it, minus the L1 error that measuring it with noise of standard deviation sigma
    would add in expectation, times the weight."""
    model_error = float(np.abs(marginal(data, attrs) - marginal(model, attrs)).sum())
    penalty = SQRT_2_OVER_PI * sigma * cells(attrs)
    return weight * (model_error - penalty)
```

The exponential mechanism needs the score's sensitivity, and the weight sets it.

::: {#prp-score-sensitivity name="Sensitivity of the score"}
Adding or removing one record changes $u(D, r)$ by at most $w_r$. So, over a candidate set $\mathcal R$, the sensitivity is $\Delta u = \max_{r \in \mathcal R} w_r$.
:::

::: {.proof}
The model $\hat p_{t-1}$ and the noise level are fixed when the score is computed, so only $\mu_r(D)$ changes between neighbouring datasets, and by the argument of @lem-marginal-sensitivity it changes in a single cell by $1$. The $L_1$ distance to the fixed vector $\mu_r(\hat p_{t-1})$ therefore changes by at most $1$, by the triangle inequality applied to that one cell. Multiplying by $w_r$ gives the claim, and the sensitivity over the set is the largest of these.
:::

AIM then chooses $r_t$ with probability proportional to $\exp\bigl(\varepsilon_t u(D, r)/(2\Delta u)\bigr)$, as @prp-exponential prescribes. @tbl-scores and @fig-scores show the first round on the toy dataset. The model is the independence model, so the candidates with real dependence to find, such as smoker with condition and age with exercise, have the largest errors, and two of them clear their noise penalty by a wide margin.

{{< include figures/_scores.md >}}

![The first round's balance for each candidate: the model's error on the marginal against the $L_1$ error that measuring it would add in expectation. Only candidates whose blue bar is taller than their grey bar get a positive score.](figures/scores.svg){#fig-scores}

The last column of @tbl-scores is the sobering part. At the starting budget $\varepsilon_1 = 0.0174$, the best candidate's score of $396$ translates into an exponent of $0.0174 \times 396 / 12 \approx 0.57$, so it is only about $e^{0.57} \approx 1.8$ times as likely as an average candidate. The selection is nearly random, and in the toy run round 1 chose the one-way marginal of smoker, which the model already knew. This is not a flaw in the score. It is what the cautious schedule of @eq-start buys, and the annealing rule exists to notice it and raise the stakes.

## Measurement and estimation

The chosen marginal is measured with Gaussian noise at level $\sigma_t$, by @prp-gaussian, and appended to the list of measurements. Then the model is refitted to the whole list by @eq-pgm, which by @prp-zcdp (b) is free. Refitting to everything, rather than adjusting the model for the new measurement alone, is what keeps the model consistent with every earlier measurement, weighted by its precision.

## Annealing: when a round teaches too little

Here is AIM's second idea, which makes the number of rounds and the per-round budget automatic. After refitting, AIM compares the model's marginal over $r_t$ before and after the measurement. If the measurement moved it by no more than the noise it carried, that is, if

$$\bigl\| \mu_{r_t}(\hat p_t) - \mu_{r_t}(\hat p_{t-1}) \bigr\|_1 \le \sqrt{2/\pi}\;\sigma_t\, m_{r_t},$$ {#eq-anneal}

then the round taught too little, and AIM doubles $\varepsilon_{t+1}$ and halves $\sigma_{t+1}$. By @eq-round-cost both changes multiply the cost of a round by four. The condition can hold for two reasons, and the response fixes both. Either no candidate was worth measuring at this noise level, in which case halving $\sigma$ makes more candidates' scores positive, or a good candidate existed and the near-random selection missed it, in which case doubling $\varepsilon$ makes the selection sharper. The comparison looks only at two fitted models, so it is free.

A second rule ends the loop cleanly. After the annealing decision, if the remaining budget $\rho - \rho_{\mathrm{used}}$ is at most twice the next round's cost, AIM makes the next round the last, and sets its parameters to spend exactly what is left, in the same nine-to-one split:

$$\varepsilon_{t+1} = \sqrt{8 \cdot 0.1\,(\rho - \rho_{\mathrm{used}})}, \qquad \sigma_{t+1} = \sqrt{\frac{1}{2 \cdot 0.9\,(\rho - \rho_{\mathrm{used}})}}.$$ {#eq-final}

Substituting these into @eq-round-cost gives $0.1$ plus $0.9$ of the remainder, which is all of it, so the loop's condition $\rho_{\mathrm{used}} < \rho$ fails after that round and AIM stops. The script's version of both rules is short:

```{.python excerpt="anneal"}
def anneal(
    eps: float, sigma: float, change: float, penalty: float, remaining: float
) -> tuple[float, float, bool, bool]:
    """The next round's (eps, sigma). If the last measurement moved the model by less
    than the noise it carried, double eps and halve sigma. Then, if what is left cannot
    pay for two such rounds, make the next round the last one and let it spend exactly
    the remaining budget. Also returns whether it annealed and whether it is the last."""
    annealed = change <= penalty
    if annealed:
        eps, sigma = 2 * eps, sigma / 2
    last = remaining <= 2 * round_cost(eps, sigma)
    if last:
        eps = math.sqrt(8 * (1 - MEASURE_SHARE) * remaining)
        sigma = math.sqrt(1 / (2 * MEASURE_SHARE * remaining))
    return eps, sigma, annealed, last
```

## The run on the toy dataset

@tbl-rounds lists every round, and @fig-rounds plots the workload error of @eq-error after each. The run is short, because the budget of $\rho = 0.0244$ is small and each annealing quadruples the cost of a round.

{{< include figures/_rounds.md >}}

![The model's workload error after each round of the toy run. Round 0 is the independence model from the one-way marginals. Circled rounds annealed, so the round after each paid four times as much.](figures/rounds.svg){#fig-rounds}

Round 1 is the near-random selection discussed above. Measuring smoker again barely moved the model, the change of one record is far below the penalty, so the round annealed. Rounds 2 and 3, at twice the selection budget and half the noise, picked the two marginals with the most dependence to find, and the workload error fell by almost half. Round 4 picked exercise with condition and fixed it; round 5 picked it again, learned nothing new, and annealed. Round 6, at four times the original stakes, found the dependence between age and condition, the last large error. Round 7 was the final round, set by @eq-final to spend the rest, and it refined a marginal that was already close. The final error is $0.053$, from $0.141$ at the start, and @tbl-final shows the marginal the note has followed throughout.

| smoker | true: no condition | true: condition | final model: no condition | final model: condition |
|:---|---:|---:|---:|---:|
| no | 627 | 98 | 626.3 | 103.0 |
| yes | 181 | 94 | 185.4 | 104.0 |

: The marginal over smoker and condition in the toy dataset and in AIM's final model. Compare @tbl-init, the model before any two-way marginal was measured. {#tbl-final}

One comparison is owed to the reader. On this toy workload, measuring all six marginals directly with the whole budget, at noise level $\sigma = 11.1$, and fitting the model to them gives a workload error of $0.035$, better than AIM's $0.053$. Six marginals are few enough for the budget, and none of them is redundant, so there is nothing for selection to gain and the selection budget is spent for nothing. AIM's advantage is on real workloads, where hundreds of marginals compete for the budget and most are nearly implied by a few others. The ablations in @sec-results measure that advantage.

# Why AIM keeps its promise {#sec-privacy}

Everything AIM does to the data is a Gaussian measurement or an exponential-mechanism selection, and everything else is post-processing. The privacy proof therefore has two parts, which this section sets out: show that each round's cost is @eq-round-cost, and show that the costs never add up to more than $\rho$. The second part needs care, because the rounds' costs are chosen adaptively, after seeing earlier outputs, and plain composition assumes they are fixed in advance. The tool that handles this is a budget filter.

::: {#lem-filter name="Budget filter"}
Consider a sequence of mechanisms in which the $t$-th is $\rho_t$-zCDP, where $\rho_t$ may be chosen after seeing the outputs of the earlier mechanisms. If, for every choice the earlier outputs can lead to, the sum of the $\rho_t$ over the rounds that run is at most $\rho$, then releasing all the outputs is $\rho$-zCDP.
:::
<!-- proof: cited -->

This is Theorem 3.1 of Feldman and Zrnic (2021), proved for Rényi differential privacy, which contains zCDP as the case of all orders at once. The proof bounds the Rényi divergence of the whole output by a supermartingale argument over the rounds, which is the machinery the plain composition proof does not need.

::: {#thm-aim-privacy name="AIM is ρ-zCDP"}
With the paper's settings, $T = 16d$ and a nine-to-one split, AIM is $\rho$-zCDP. By @prp-zcdp (c) it is therefore $(\varepsilon, \delta)$-DP for every pair that the conversion gives.
:::

::: {.proof}
The plan is to list every access to the data with its cost, confirm that nothing else touches the data, and then show that the loop's bookkeeping keeps the total at $\rho$, so that @lem-filter applies.

**Step 1: the accesses and their costs.** The initialization measures at most $d$ one-way marginals, each at noise level $\sigma_0$. By @lem-marginal-sensitivity and @prp-gaussian each costs $1/(2\sigma_0^2)$, so the initialization costs at most $d/(2\sigma_0^2) = 0.9\rho\, d/T$ by @eq-start, which is $0.9\rho/16$ for $T = 16d$. Round $t$ makes two accesses. The selection is the exponential mechanism with the score of @eq-score at budget $\varepsilon_t$, whose sensitivity @prp-score-sensitivity gives, so by @prp-exponential it costs $\varepsilon_t^2/8$. The measurement is a Gaussian release at level $\sigma_t$, which costs $1/(2\sigma_t^2)$. Together they are @eq-round-cost, which is exactly what the algorithm adds to $\rho_{\mathrm{used}}$.

**Step 2: nothing else touches the data.** The model fits of @eq-pgm use only the released measurements. The candidate weights depend on the workload alone, and the candidate filter on the sets already selected and on $\rho_{\mathrm{used}}$, both of which are released. The annealing test of @eq-anneal compares two fitted models, and the final-round rule of @eq-final reads $\rho_{\mathrm{used}}$. The rounding to records uses the final model. By @prp-zcdp (b), none of this adds to the cost. In particular, the next round's $\varepsilon_{t+1}$ and $\sigma_{t+1}$ are functions of earlier outputs, which is exactly the situation @lem-filter allows.

**Step 3: the total never exceeds $\rho$.** After the initialization, by Step 1, the remaining budget is at least $\rho(1 - 0.9/16)$, and the first round costs $\rho/T = \rho/(16d)$ by @eq-start, which is smaller. So the first round fits. Now consider any later round $t + 1$, and look at the decision taken at the end of round $t$. If the final-round rule fired, round $t + 1$ costs exactly the remainder, by the arithmetic below @eq-final, so after it $\rho_{\mathrm{used}} = \rho$ and the loop stops. If it did not fire, the remainder was more than twice the cost of round $t + 1$, so that round fits with budget to spare, and the argument repeats at the end of it. Either way $\rho_{\mathrm{used}}$ never exceeds $\rho$, on every path the random choices can take.

By Steps 1 to 3, AIM is a sequence of mechanisms with adaptively chosen costs whose sum is at most $\rho$ on every path, and @lem-filter makes it $\rho$-zCDP.
:::

The proof also shows why the final-round rule is not a detail. Without it, a round could begin with less budget left than it costs, and the only remedies would be to overspend or to waste what is left. With it, the last round always spends exactly the remainder, in the same split as every other round. The script checks this at the end of every run, where $\rho_{\mathrm{used}}$ equals $\rho$ to floating-point precision.

# What AIM says about its own error {#sec-error}

A synthetic dataset is more useful when it comes with a statement of how far its marginals may be from the truth. AIM provides one, computed from quantities it has already released and so at no extra privacy cost. This section states the simpler of the paper's two bounds and explains where it comes from; the proof is cited.

Consider a marginal $r$ that is contained in one or more of the measured sets. Each such measurement $\tilde y_i$ gives, by adding up its cells, a noisy copy of $\mu_r(D)$, and the copies can be averaged, with weights by their precision, into one estimate $\bar y_r$. That estimate is the truth plus Gaussian noise of a known level $\bar\sigma_r$, computable from the $\sigma_i$ and the cell counts. The question is how large the $L_1$ norm of that noise can be.

::: {#thm-error name="A confidence bound for a measured marginal"}
Let $\bar y_r = \mu_r(D) + Z$, where $Z$ has $m_r$ independent $\mathcal N(0, \bar\sigma_r^2)$ entries. For every $\lambda \ge 0$, with probability at least $1 - e^{-\lambda^2}$,
$$\bigl\| \mu_r(D) - \bar y_r \bigr\|_1 \le \sqrt{2 \ln 2}\; \bar\sigma_r\, m_r + \lambda\, \bar\sigma_r \sqrt{2 m_r}.$$
:::
<!-- proof: cited -->

This is Theorem 3 of the AIM paper. The proof has two steps. The first bounds the expected $L_1$ norm by Jensen's inequality, which gives the first term and is slightly looser than @lem-l1-noise. The second applies a concentration inequality to the $L_1$ norm, which is a Lipschitz function of the Gaussian vector, and that gives the second term. The bound on the synthetic dataset follows by the triangle inequality: the distance from $\mu_r(\hat D)$ to the truth is at most its distance to $\bar y_r$, which is known, plus the bound above. For a marginal that no measured set contains, the paper gives a weaker bound built from the selection scores. Those are bounded too, because a candidate with a very high score would almost surely have been selected. In the paper's experiments, with $\lambda$ set for 95% confidence, the bounds held and were within a factor of about four of the true error for measured marginals and eight for the others.

# How well it works {#sec-results}

The paper evaluates AIM on six public datasets with between nine and sixteen attributes, and between about a thousand and a million records. Each is run on three workloads of three-way marginals, at nine privacy budgets from $\varepsilon = 0.01$ to $\varepsilon = 100$, with $\delta = 10^{-9}$. Three of the competitors ignore the workload, among them MST, the winner of a NIST synthetic-data competition. The other three use it, among them MWEM with Private-PGM as its model, and two methods that fit neural networks or relaxed tables instead of a graphical model.

Over all settings, AIM had the lowest workload error in 76% of them, and in every setting with $\varepsilon \ge 1$. Against the next best method in each setting it was better by a factor of 1.6 on average, and by up to 5.7 on the workloads that concentrate on a few attributes, where selecting what to measure matters most.

The more instructive experiment removes AIM's ideas one at a time and measures how much the error grows, as @tbl-ablation reports.

| Idea removed | Error grows by a factor of |
|:---|---:|
| Private-PGM, replaced by a relaxed table that need not be a distribution | 2.36 |
| adaptive rounds and the annealed budget split, replaced by fixed rounds | 1.48 |
| the noise-aware score of @eq-score, replaced by the model error alone | 1.37 |
| Gaussian noise, replaced by Laplace noise | 1.18 |
| add-or-remove neighbours, replaced by replace-one | 1.13 |
| the nine-to-one split, replaced by an even split | 1.08 |
| the initialization on one-way marginals | 1.03 |

: The paper's ablation, as geometric means over the datasets and budgets of the ratio of the error without the idea to the error with it. {#tbl-ablation}

The two largest factors are the two things this note spent most of its pages on. Generating from a real distribution, which Private-PGM does, matters more than anything else, and the adaptive budget comes second. The two ideas the note presented as AIM's own, the annealing and the noise-aware score, together account for a factor of about two.

Three limits of the method are worth stating. Every attribute must be discrete, so numeric attributes are binned, and the bins are a choice the user makes before any privacy is spent. The model's size is capped, so a workload whose marginals overlap in a tangled way cannot all be measured, however large the budget. And the running time grows with the model, since every round refits it. The paper ran AIM on one processor core with a memory limit of four gigabytes, and the 80 megabyte cap on the model is what keeps that feasible.

# What the note proves and what it cites

The note proves the facts that are specific to AIM. A marginal has sensitivity one (@lem-marginal-sensitivity), Gaussian noise has the expected size that sets the score's penalty (@lem-l1-noise), the score has the sensitivity the weights give it (@prp-score-sensitivity), and the whole algorithm is private (@thm-aim-privacy). It cites the general tools. These are the properties of zCDP (@prp-zcdp), the costs of the two building blocks (@prp-gaussian and @prp-exponential), the budget filter (@lem-filter), the form of Private-PGM's solution (@thm-pgm) and the error bound (@thm-error). Each citation says what its source proves.

# Sources

1. R. McKenna, B. Mullins, D. Sheldon and G. Miklau (2022), *AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data*, PVLDB 15(11). The algorithm, its privacy theorem, the error bounds and the experiments. <https://arxiv.org/abs/2201.12677>
2. R. McKenna, D. Sheldon and G. Miklau (2019), *Graphical-model based estimation and inference for differential privacy*, ICML. Private-PGM: the objective of @eq-pgm, the form of its solution (@thm-pgm) and the algorithms that fit it without building the full table. <https://arxiv.org/abs/1901.09436>
3. M. Bun and T. Steinke (2016), *Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds*. The definition of zCDP, its composition and post-processing, and the Gaussian mechanism's cost. <https://arxiv.org/abs/1605.02065>
4. C. Canonne, G. Kamath and T. Steinke (2020), *The Discrete Gaussian for Differential Privacy*. The conversion from zCDP to $(\varepsilon, \delta)$-DP in @prp-zcdp (c). <https://arxiv.org/abs/2004.00010>
5. F. McSherry and K. Talwar (2007), *Mechanism Design via Differential Privacy*, FOCS. The exponential mechanism and its $\varepsilon$-DP guarantee.
6. M. Cesar and R. Rogers (2021), *Bounding, Concentrating, and Truncating: Unifying Privacy Loss Composition for Data Analytics*, ALT. The $\varepsilon^2/8$ zCDP cost of the exponential mechanism. <https://arxiv.org/abs/2004.07223>
7. V. Feldman and T. Zrnic (2021), *Individual Privacy Accounting via a Rényi Filter*, NeurIPS. The budget filter of @lem-filter. <https://arxiv.org/abs/2008.11193>
8. M. Hardt, K. Ligett and F. McSherry (2012), *A Simple and Practical Algorithm for Differentially Private Data Release*, NeurIPS. MWEM, the ancestor of AIM's select-measure-generate loop. <https://arxiv.org/abs/1012.4763>
