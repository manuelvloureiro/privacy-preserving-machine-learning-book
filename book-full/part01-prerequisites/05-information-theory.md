# Information theory

## Why this matters

You observe something, and you want to know how much it tells you about something else you cannot see. A word in an email says something about whether the email is spam. A noisy sensor reading says something about the true state. A model's prediction says something about the label. Information theory turns "how much" into a number, and it does so in two different ways, because there are two different questions you might be asking.

The first question is: **how many bits, on average, does the observation save you when you describe the hidden value?** Shannon entropy and mutual information answer it. They are the language of compression, of the cross-entropy loss you train classifiers with, of decision-tree splits and of sending data over noisy links. The second question is: **how much more likely is a single best guess to be right once you have seen the observation?** Min-entropy and vulnerability answer that one. They are the language of password strength, and of anyone who gets only one try.

The idea that organizes the chapter is that **every quantity here is built from the logarithm of a probability, either averaged or maximized.** Shannon entropy is the average of $-\log p(x)$ over the outcomes $x$; min-entropy uses only the most likely outcome instead. Because of this, almost every derivation below is one of two moves. Either it applies Jensen's inequality, which compares the average of a logarithm with the logarithm of an average, or it takes the product rule $p(x, y) = p(y)\,p(x \mid y)$ (the probability of both equals the probability of $y$ times the probability of $x$ given $y$) and turns it, by taking logs, into a sum.

| Result | Where it is used |
| --- | --- |
| Entropy, $0 \le H(X) \le \log\lvert\mathcal X\rvert$ | Cross-entropy loss, compression, the scale of everything below |
| Chain rule, conditional entropy | Splitting the uncertainty of several variables into pieces |
| Mutual information $I(X; Y)$ as a Kullback–Leibler (KL) divergence | Feature selection, information gain; the link to [Divergences](03-divergences.md) |
| Data-processing inequality | Processing an observation cannot add information about its source |
| Channel capacity, binary symmetric channel $1 - h(p)$ | The most information a noisy channel can carry |
| Vulnerability and min-entropy | One-guess success; password strength |
| Conditional min-entropy, min-entropy leakage, min-capacity | How much an observation helps a guesser, and its worst case over priors |
| Fano's inequality | Lower bounds: no estimator can beat a given error rate |

: The results of this chapter and where they are used. {#tbl-info-uses}

@tbl-info-uses lists the results. Later parts of the book measure what an observation reveals about a hidden value. They need both views, the average one and the one-guess one, and they need to know where the two disagree.

Logarithms in this chapter are base 2, so quantities are in **bits**, unless stated otherwise. [Divergences](03-divergences.md) uses natural logarithms, whose unit is the **nat**. To convert, note that $\log_2 x = \ln x / \ln 2$, so one nat is $1/\ln 2 = \log_2 e \approx 1.443$ bits.

## Toolkit: three facts every derivation uses

### Fact 1 — Jensen's inequality for the logarithm

The logarithm grows more and more slowly: going from 1 to 2 adds one bit, going from 2 to 3 adds much less. Because of this bend, if you average some positive numbers and then take the log, you get more than if you take the logs first and then average. Concretely, the numbers 1 and 4 have logs 0 and 2, whose average is 1, while their average 2.5 has log $\log_2 2.5 \approx 1.32$. In general, for a positive random variable $Z$,
$$\mathbb E[\log Z] \le \log\mathbb E[Z],$$
with equality if and only if $Z$ is constant (with probability 1), since only then is there nothing for the bend to act on.

The proof uses the fact that a curve that bends downward, a **concave** function, lies below each of its tangent lines. Take the tangent to $\log$ at the point $m = \mathbb E Z$. Its slope is the derivative of $\log_2 z$ at $m$, which is $1/(m\ln 2)$, so for every $z > 0$
$$\log z \le \log m + \frac{z - m}{m\ln 2}.$$
Now put $Z$ in place of $z$ and average both sides. The left side becomes $\mathbb E[\log Z]$. On the right, $\log m$ stays as it is and the second term averages to zero, because $\mathbb E[Z - m] = 0$ by the choice of $m$. That leaves $\mathbb E[\log Z] \le \log m = \log\mathbb E Z$. The tangent touches $\log$ only at $m$, so equality needs $Z = m$ wherever $Z$ has probability: $Z$ must be constant.

This fact turns "an average of logs" into "a log of an average", and that is how every bound in this chapter is proved. Multiplying both sides by $-1$ gives the mirror version for $-\log$, which bends the other way (it is **convex**): $-\log\mathbb E Z \le \mathbb E[-\log Z]$.

### Fact 2 — Gibbs' inequality: KL divergence is non-negative

To say how different two distributions $p$ and $q$ on the same finite set are, information theory uses the **Kullback–Leibler (KL) divergence** $D(p\,\|\,q)$. It is the average, over outcomes drawn from $p$, of the log of how much more likely each outcome is under $p$ than under $q$. If $p$ and $q$ are the same, every ratio is 1, every log is 0, and the divergence is 0. Gibbs' inequality says that it can never be negative:
$$D(p\,\|\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)} \ge 0,$$
with equality if and only if $p = q$. This is not obvious, because individual log ratios can be negative (wherever $q(x) > p(x)$). The point is that on average, weighted by $p$, they cannot be.

The proof is Jensen's inequality (Fact 1) applied to the ratio $Z = q(X)/p(X)$, with $X$ drawn from $p$. The average of this ratio is easy: the $p(x)$ in the weight cancels the $p(x)$ in the denominator, leaving $\sum q(x)$ over the outcomes that $p$ can produce, which is at most 1. So, summing only over $x$ with $p(x) > 0$,
$$-D(p\,\|\,q) = \mathbb E\left[\log\frac{q(X)}{p(X)}\right] \le \log\sum_{x:\,p(x) > 0} p(x)\frac{q(x)}{p(x)} = \log\sum_{x:\,p(x) > 0} q(x) \le \log 1 = 0.$$
Both inequalities in this line must be equalities for $D = 0$. The first needs the ratio $q/p$ to be constant where $p$ has mass (the equality case of Fact 1), and the second needs $q$ to put no mass outside that set. Together these force $q = p$.

Every "$\ge 0$" and every "$\le \log\lvert\mathcal X\rvert$" below is this fact in disguise. [Divergences](03-divergences.md) studies $D$ itself in more detail.

### Fact 3 — Conventions

Three conventions keep the formulas short. First, $0\log 0 = 0$. The expression $0\log 0$ is not defined on its own, but $t\log t \to 0$ as $t \to 0$, and this choice says that outcomes of probability zero contribute nothing, which is what you want from a measure of uncertainty.

Second, a joint distribution of two variables with finitely many values is a table $p(x, y)$, with one row for each value of $X$ and one column for each value of $Y$. Adding up a row gives the **marginal** probability $p(x)$ that $X = x$, whatever $Y$ is; adding up a column gives $p(y)$.

Third, a **channel** from $X$ to $Y$ is a rule that, for each input $x$, says how likely each output $y$ is. It is also a table, $C[x, y] = p(y \mid x)$, and each row sums to 1 because, given the input, some output must occur. A channel alone does not say how likely each input is. That comes from a **prior** $\pi(x)$, and the two together give the joint table by the product rule, $p(x, y) = \pi(x)\,C[x, y]$: the probability that the input is $x$, times the probability that the channel then produces $y$.

## Shannon entropy

### Context

How uncertain is a random variable? A fair coin should count as one unit of uncertainty, two fair coins as two, and a coin that always lands heads as zero. Shannon entropy is the measure with these properties, and it is also the average number of bits that the best possible code needs per symbol. You meet it every time you train a classifier. The cross-entropy loss, averaged over the data, equals $H(p) + D(p\,\|\,q)$: the entropy of the true label distribution $p$, which no model can remove, plus the KL divergence from $p$ to the model's predicted distribution $q$, which training drives down.

### Statement

The rarer an outcome, the more it tells you when it happens. So measure the information in an outcome $x$ by its **surprisal** $\log(1/p(x))$, which is 0 for a certain outcome and grows as $p(x)$ shrinks: an outcome of probability $\frac12$ carries 1 bit, one of probability $\frac14$ carries 2 bits. The **entropy** of a discrete random variable $X$ with distribution $p$ is its average surprisal:
$$H(X) = -\sum_x p(x)\log p(x) = \mathbb E\left[\log\frac{1}{p(X)}\right].$$
For a variable that is 1 with probability $p$ and 0 otherwise, a $\mathrm{Bernoulli}(p)$ variable, the sum has two terms. It is called the **binary entropy function**:
$$h(p) = -p\log p - (1 - p)\log(1 - p).$$

**Bounds.** Entropy cannot be negative, because surprisals are not, and it is largest when no outcome is favoured. For $X$ taking values in a finite set $\mathcal X$, with $\lvert\mathcal X\rvert$ elements,
$$0 \le H(X) \le \log\lvert\mathcal X\rvert.$$
The lower bound holds with equality exactly when $X$ is constant, the upper bound exactly when $X$ is uniform.

**Joint and conditional entropy.** For two variables, $H(X, Y)$ is the entropy of the pair, computed from the joint table. Once you observe a particular value $Y = y$, the uncertainty about $X$ is the entropy of the distribution of $X$ given that value, $H(X \mid Y = y) = -\sum_x p(x \mid y)\log p(x \mid y)$. Before you observe $Y$, you do not know which $y$ you will see, so the natural summary is the average over $y$. That average is the **conditional entropy**:
$$H(X \mid Y) = \sum_y p(y)\,H(X \mid Y = y).$$

**Chain rule.** The uncertainty in the pair is the uncertainty in $Y$ plus whatever uncertainty about $X$ remains once $Y$ is known, and the same holds with the roles swapped:
$$H(X, Y) = H(Y) + H(X \mid Y) = H(X) + H(Y \mid X).$$ {#eq-chain-rule}

### Derivation

*Lower bound.* The idea is that entropy is an average of surprisals, and no surprisal is negative.

1. Each probability satisfies $0 \le p(x) \le 1$, so $1/p(x) \ge 1$ and $\log(1/p(x)) \ge 0$. Each term $p(x)\log(1/p(x))$ is therefore non-negative (Fact 3 handles $p(x) = 0$).
2. A sum of non-negative terms is zero only if every term is zero. A term $p(x)\log(1/p(x))$ is zero only when $p(x) = 0$ or $p(x) = 1$. So $H(X) = 0$ forces one value to have probability 1: $X$ is constant.

*Upper bound.* The idea is that the gap between $\log\lvert\mathcal X\rvert$ and $H(X)$ is exactly the KL divergence from $p$ to the uniform distribution, which Fact 2 says is non-negative.

1. Let $u(x) = 1/\lvert\mathcal X\rvert$ be the uniform distribution. Since $\sum_x p(x) = 1$, you can write the constant $\log\lvert\mathcal X\rvert$ as $\sum_x p(x)\log\lvert\mathcal X\rvert$. Then $\log\lvert\mathcal X\rvert - H(X) = \sum_x p(x)\log\lvert\mathcal X\rvert + \sum_x p(x)\log p(x)$.
2. Combine the two logs in each term, using $\log\lvert\mathcal X\rvert = \log(1/u(x))$: the sum becomes $\sum_x p(x)\log\frac{p(x)}{u(x)} = D(p\,\|\,u)$.
3. By Gibbs' inequality (Fact 2), $D(p\,\|\,u) \ge 0$, with equality iff $p = u$. So $H(X) \le \log\lvert\mathcal X\rvert$, with equality iff $X$ is uniform.

*Chain rule.* The idea is that the product rule for probabilities becomes a sum rule for surprisals, and averaging preserves sums.

1. For every pair with $p(x, y) > 0$, the product rule gives $p(x, y) = p(y)\,p(x \mid y)$. Taking $\log(1/\cdot)$ of both sides turns the product into a sum: $\log\frac{1}{p(x, y)} = \log\frac{1}{p(y)} + \log\frac{1}{p(x \mid y)}$.
2. Average both sides over the pair $(X, Y)$. The left side is the average surprisal of the pair, $H(X, Y)$. The first term on the right is $H(Y)$. The second term, grouped by $y$, is $\sum_y p(y)\sum_x p(x \mid y)\log\frac{1}{p(x \mid y)}$, which is the average over $y$ of $H(X \mid Y = y)$, that is $H(X \mid Y)$.
3. Starting from $p(x, y) = p(x)\,p(y \mid x)$ instead, the same steps give the second form.

### Interpretation

@fig-binary-entropy shows the binary entropy $h(p)$. It is 1 bit at $p = \frac12$, where the coin is fair and you have no idea which way it will land, and zero at $p \in \{0, 1\}$, where you know the outcome. It is very steep near the ends, so even a quite lopsided coin keeps a lot of uncertainty: $h(0.11) = 0.4999$, so a coin with an 11% chance of heads still carries half a bit. The figure also shows the min-entropy of the same coin, a one-guess measure that @sec-min-entropy defines and that is always below $h$.

The chain rule splits uncertainty into pieces: first reveal $Y$, then what remains of $X$. Applying @eq-chain-rule repeatedly, peeling off one variable at a time, extends it to any number of variables: $H(X_1, \dots, X_n) = \sum_i H(X_i \mid X_1, \dots, X_{i-1})$, the uncertainty of the first variable, plus that of the second given the first, and so on.

Entropy depends only on the probabilities, not on the values themselves. Relabelling the outcomes, for instance calling them "red" and "blue" instead of 0 and 1, does not change it.

![The binary entropy $h(p)$ against the min-entropy $-\log_2\max(p, 1-p)$ of the same Bernoulli variable. Both are 1 bit at $p = \frac12$; the min-entropy is lower everywhere else.](figures/information_theory_binary_entropy.svg){#fig-binary-entropy}

### Conditioning reduces entropy on average, not pointwise

Intuitively, learning something should not make you more uncertain. That is true on average: averaged over what you might observe, observing $Y$ cannot increase the uncertainty about $X$,
$$H(X \mid Y) \le H(X),$$
with equality if and only if $X$ and $Y$ are independent. The proof is short once mutual information is defined in the next section. There the gap $H(X) - H(X \mid Y)$ turns out to be a KL divergence (@eq-mi-kl), and KL divergences are non-negative by Gibbs' inequality (Fact 2).

But a *particular* observation can increase the uncertainty. $H(X \mid Y = y) > H(X)$ is perfectly possible, as the next example shows: if an observation is rare, it can push you from a confident belief into a toss-up, and the average stays low only because the common observations make up for it.

### Example: a spam filter's keyword

Let $X$ be the label of an email (ham or spam) and $Y$ whether it contains the word "offer". One email in eight is spam. Every spam email contains the word, and so does one ham email in seven. Since ham emails are $\frac78$ of the total, the ham emails with the word are $\frac17\cdot\frac78 = \frac18$ of all emails, and the ham emails without it are $\frac78 - \frac18 = \frac34$. @tbl-spam gives the full joint distribution.

| | $Y = 0$ (absent) | $Y = 1$ (present) | $p(x)$ |
| --- | --- | --- | --- |
| ham | 3/4 | 1/8 | 7/8 |
| spam | 0 | 1/8 | 1/8 |
| $p(y)$ | 3/4 | 1/4 | |

: Joint distribution of an email's label and whether it contains the keyword. {#tbl-spam}

Work out each quantity from @tbl-spam. Before you look at the email, the label is spam with probability $\frac18$, so $H(X) = h(1/8) = \frac18\log_2 8 + \frac78\log_2\frac87 = 0.375 + 0.1686 = 0.5436$ bits: the label is fairly predictable, because most email is ham.

If the word is absent, you are in the first column. Its only non-zero entry is ham, so the email is certainly ham and $H(X \mid Y = 0) = 0$.

If the word is present, you are in the second column, where ham and spam each have probability $\frac18$ out of $\frac14$. Given the word, they are equally likely, so $H(X \mid Y = 1) = h(1/2) = 1$ bit. **Seeing the word made you more uncertain** than you were before: from 0.5436 bits up to a full bit.

On average, though, the word helps. You see it absent with probability $\frac34$ and present with probability $\frac14$, so $H(X \mid Y) = \frac34\cdot0 + \frac14\cdot1 = 0.25$ bits, less than $H(X)$.

The chain rule can be checked on the same table. The word is present with probability $\frac14$, so $H(Y) = h(1/4) = 0.8113$, and $H(Y) + H(X \mid Y) = 0.8113 + 0.25 = 1.0613$. That equals the entropy of the pair computed directly from the three non-zero cells $(\frac34, \frac18, \frac18)$.

@fig-conditioning draws the four numbers. The common observation (word absent) removes all uncertainty; the rare one (word present) increases it; the average goes down. That is the general pattern: conditioning reduces entropy only in expectation.

![Entropy of the spam label before observing the keyword, after each of the two possible observations, and on average. The rare observation raises the entropy above $H(X)$ (dashed line); the average falls below it.](figures/information_theory_conditioning.svg){#fig-conditioning}

### Check yourself

- What is the entropy of a uniformly random 4-digit PIN, and why? *(There are 10,000 equally likely PINs, so $H = \log_2 10{,}000 = 13.2877$ bits: the upper bound $\log\lvert\mathcal X\rvert$, attained because the distribution is uniform.)*
- Using the chain rule, show that $H(X, Y) \le H(X) + H(Y)$, with equality iff $X$ and $Y$ are independent. *($H(X, Y) = H(Y) + H(X \mid Y)$, and $H(X \mid Y) \le H(X)$ with equality iff independent.)*

## Mutual information

### Context

How much does observing $Y$ tell you about $X$? A natural answer is the drop in uncertainty it causes on average, $H(X) - H(X \mid Y)$. This is the **information gain** a decision tree uses to choose a split, and the criterion many feature-selection methods use to rank features. It turns out to be a KL divergence, which makes it symmetric and non-negative, and it obeys a rule that drives a great deal of later reasoning: processing cannot create information.

### Statement

The **mutual information** between $X$ and $Y$ is the average drop in uncertainty about $X$ from seeing $Y$. It can also be read as a measure of dependence: the KL divergence from the true joint distribution $P_{XY}$ to the product $P_XP_Y$ of its marginals, which is what the joint distribution would be if $X$ and $Y$ were independent. The two readings give the same number:
$$I(X; Y) = H(X) - H(X \mid Y) = D\big(P_{XY}\,\big\|\,P_XP_Y\big) = \sum_{x, y}p(x, y)\log\frac{p(x, y)}{p(x)\,p(y)}.$$ {#eq-mi-kl}
It follows that mutual information is symmetric, $I(X; Y) = I(Y; X) = H(Y) - H(Y \mid X)$: $Y$ tells you as much about $X$ as $X$ tells you about $Y$. It is non-negative, and it is zero if and only if $X$ and $Y$ are independent.

Sometimes a third variable $Z$ is already known. The **conditional mutual information** is the same quantity computed with every distribution conditioned on $Z = z$, then averaged over $z$. It measures what $Y$ adds about $X$ beyond what $Z$ already told you, and it is non-negative because it is an average of KL divergences:
$$I(X; Y \mid Z) = H(X \mid Z) - H(X \mid Y, Z) = \sum_z p(z)\,D(P_{XY\mid z}\,\|\,P_{X\mid z}P_{Y\mid z}) \ge 0.$$

**Chain rule for mutual information.** What a pair $(Y, Z)$ tells you about $X$ is what $Y$ tells you, plus what $Z$ adds once $Y$ is known:
$$I(X; Y, Z) = I(X; Y) + I(X; Z \mid Y).$$

**Data-processing inequality.** Suppose $Z$ is computed from $Y$ alone, possibly with fresh randomness, so that once you know $Y$, also knowing $X$ tells you nothing more about $Z$. In other words, $X$ and $Z$ are independent given $Y$. This is called a **Markov chain** and written $X \to Y \to Z$. Then $Z$ cannot know more about $X$ than $Y$ does:
$$I(X; Z) \le I(X; Y).$$

### Derivation

*Mutual information is a KL divergence.* The idea is to write both entropies as averages over the same pairs $(x, y)$, so that they can be combined term by term.

1. Start from $H(X) - H(X \mid Y)$. The entropy $H(X)$ is an average over $x$, but since $\sum_y p(x, y) = p(x)$ it can be written as an average over pairs: $H(X) = \sum_{x, y}p(x, y)\log\frac{1}{p(x)}$. The conditional entropy is already an average over pairs, $H(X \mid Y) = \sum_{x, y}p(x, y)\log\frac{1}{p(x \mid y)}$.
2. Subtract, combining the logs in each term: $\sum_{x, y}p(x, y)\log\frac{p(x \mid y)}{p(x)}$.
3. Multiply the fraction inside the log by $p(y)/p(y)$ and use $p(x \mid y)\,p(y) = p(x, y)$: $\frac{p(x \mid y)}{p(x)} = \frac{p(x, y)}{p(x)\,p(y)}$. The sum is now $\sum_{x, y}p(x, y)\log\frac{p(x, y)}{p(x)\,p(y)}$, which is the definition of $D(P_{XY}\,\|\,P_XP_Y)$.

*Symmetry, non-negativity, independence.* The expression at the end of step 3 does not change when you swap the roles of $X$ and $Y$, so $I(X; Y) = I(Y; X)$, and running the same steps with the roles swapped gives $H(Y) - H(Y \mid X)$. Gibbs' inequality (Fact 2) gives $I \ge 0$, with equality iff $p(x, y) = p(x)\,p(y)$ for all pairs, which is the definition of independence. Since $I(X; Y) = H(X) - H(X \mid Y) \ge 0$, this also proves $H(X \mid Y) \le H(X)$.

*Chain rule for mutual information.* The idea is to pass from knowing nothing to knowing $(Y, Z)$ through the intermediate stage of knowing only $Y$.

1. By definition, $I(X; Y, Z) = H(X) - H(X \mid Y, Z)$: the total drop in uncertainty.
2. Add and subtract the uncertainty at the intermediate stage, $H(X \mid Y)$: $\big[H(X) - H(X \mid Y)\big] + \big[H(X \mid Y) - H(X \mid Y, Z)\big]$.
3. The first bracket is the drop from learning $Y$, which is $I(X; Y)$. The second is the further drop from learning $Z$ when $Y$ is already known, which is $I(X; Z \mid Y)$.

*Data-processing inequality.* The idea is to expand what the pair $(Y, Z)$ tells you about $X$ in two orders. In one order, the part $Z$ adds after $Y$ is zero, because $Z$ was computed from $Y$.

1. Apply the chain rule with $Z$ first and with $Y$ first:
$$I(X; Z) + I(X; Y \mid Z) = I(X; Y, Z) = I(X; Y) + I(X; Z \mid Y).$$
2. The Markov chain says $X$ and $Z$ are independent given $Y$, so the conditional mutual information $I(X; Z \mid Y)$ is zero (it is an average of KL divergences between identical distributions).
3. Hence $I(X; Z) = I(X; Y) - I(X; Y \mid Z) \le I(X; Y)$, since conditional mutual information is non-negative.

Equality holds iff $I(X; Y \mid Z) = 0$: once you know $Z$, $Y$ tells you nothing more about $X$. Such a $Z$ has kept everything in $Y$ that matters for $X$, and is called a **sufficient statistic** for $X$.

### Interpretation

Mutual information measures dependence of any kind, not only linear correlation. Two variables can have zero correlation and still be strongly dependent, as $Y$ and $Y^2$ are when $Y$ is symmetric around zero. Their mutual information is positive in that case, and it is zero only under full independence.

The data-processing inequality says that no function of the data, however clever, can contain more information about the source than the data itself. Denoising, feature extraction and training a model are all processing steps. They can make information easier to use, but never larger.

Because $I(X; Y) = D(P_{XY}\,\|\,P_XP_Y)$, every property of KL divergence in [Divergences](03-divergences.md) applies to mutual information. In particular, KL divergence has its own data-processing inequality there: passing two distributions through the same channel cannot increase the divergence between them. That inequality implies the one here.

Mutual information is an average over both $x$ and $y$. A small value can therefore hide a few outcomes in which $Y$ reveals $X$ completely, because those outcomes carry little weight in the average. @sec-shannon-vs-min shows how far apart the average and the worst case can be.

### Example: the keyword, then a noisy detector

In the spam table, the word reduced the uncertainty from $H(X) = 0.5436$ bits to $H(X \mid Y) = 0.25$ bits on average, so $I(X; Y) = 0.5436 - 0.25 = 0.2936$ bits. The script computes the same value from the KL formula of @eq-mi-kl and from $H(Y) - H(Y \mid X)$.

Now suppose the keyword is detected by a sloppy tokenizer that gets the answer wrong with probability 0.1, whatever the email. Call its output $Z$. A channel on $\{0, 1\}$ that flips its input with a fixed probability is a **binary symmetric channel**, studied in the next section, so $Z$ is $Y$ passed through a binary symmetric channel with flip probability 0.1. The tokenizer looks only at the word, not at the label, so $X \to Y \to Z$ is a Markov chain, and the data-processing inequality predicts $I(X; Z) \le I(X; Y)$. The script finds $I(X; Z) = 0.1668$ bits. Since $0.1668/0.2936 \approx 0.57$, the noisy feature carries 57% of the information of the clean one: a 10% error rate in the detector costs over 40% of the information.

### Check yourself

- If $Z = g(Y)$ for an invertible function $g$, what is $I(X; Z)$? *(Equal to $I(X; Y)$. Apply the data-processing inequality to $X \to Y \to Z$, which gives $\le$, and to $X \to Z \to Y$, which is also a Markov chain because $Y = g^{-1}(Z)$ is computed from $Z$, and gives $\ge$.)*
- Can $I(X; Y)$ exceed $H(X)$? *(No. $I(X; Y) = H(X) - H(X \mid Y)$ and $H(X \mid Y) \ge 0$: an observation cannot remove more uncertainty than there was.)*

## Channels and capacity

### Context

A channel turns an input $X$ into a random output $Y$: a noisy storage device, a communication link, a lossy sensor. You control the input distribution but not the noise. How much information can one use of the channel carry at best? That maximum is the **capacity**. Shannon's channel coding theorem says it is exactly the rate, in bits per channel use, at which you can send data with an error probability as small as you like, using long enough codes. This chapter does not prove that theorem; it computes the capacity, which is the number the theorem is about.

### Statement

A channel is fixed, but how much information passes through it depends on how often you send each input, the prior $\pi$. So the capacity is the mutual information between input and output, maximized over the input prior. For a channel $C[x, y] = p(y \mid x)$,
$$\mathcal C = \max_{\pi}I(X; Y), \qquad X \sim \pi,\ Y \mid X = x \sim C[x, \cdot].$$
The **binary symmetric channel** $\mathrm{BSC}(p)$ has inputs and outputs in $\{0, 1\}$ and flips the input with probability $p$. Its capacity is one bit, the most a binary output can carry, minus the entropy of the noise:
$$\mathcal C_{\mathrm{BSC}}(p) = 1 - h(p),$$ {#eq-bsc-capacity}
attained by the uniform input.

### Derivation

The idea is to use the form $I(X; Y) = H(Y) - H(Y \mid X)$: the uncertainty in the output, minus the part of it that comes from the noise. The noise part turns out not to depend on the input prior, so you only have to make the output as uncertain as possible. Let the input prior be $q = \Pr[X = 1]$.

1. Use the symmetric form of mutual information: $I(X; Y) = H(Y) - H(Y \mid X)$.
2. Given $X = x$, the output equals $x$ with probability $1 - p$ and is flipped with probability $p$, so $H(Y \mid X = x) = h(p)$ for both inputs. Averaging over $x$ gives $H(Y \mid X) = h(p)$, whatever $q$ is.
3. $Y$ takes two values, so by the entropy upper bound $H(Y) \le \log 2 = 1$, with equality iff $Y$ is uniform.
4. The output is 1 if the input is 1 and not flipped, or the input is 0 and flipped, so $\Pr[Y = 1] = q(1 - p) + (1 - q)p$. At $q = \frac12$ this is $\frac12(1 - p) + \frac12 p = \frac12$, so the uniform input makes $Y$ uniform and the bound in step 3 is attained.
5. Hence $\max_q I(X; Y) = 1 - h(p)$.

### Interpretation

@fig-bsc (left) shows $I(X; Y)$ as a function of the input prior $q$: a concave hump peaking at $q = \frac12$ for every $p$. Away from the peak you lose a lot. At $p = 0.11$ and $q = 0.2$, the output is 1 with probability $0.2(0.89) + 0.8(0.11) = 0.266$, so $H(Y) = h(0.266) = 0.8357$ and the channel carries only $0.8357 - 0.4999 = 0.3358$ bits, against 0.5001 at the uniform input.

The right panel shows the capacity against $p$. It is zero at $p = \frac12$, where the output is a fair coin flip whatever the input, so the output is independent of the input. It is symmetric about $\frac12$, because $h(p) = h(1 - p)$: a channel that flips almost always is as good as one that never flips, since the receiver can flip every output back.

For channels without this symmetry, the maximum over $\pi$ has no closed form. The **Blahut–Arimoto** algorithm computes it numerically, by alternately improving the prior and the receiver's best reading of the output until neither changes. The script uses it to check @eq-bsc-capacity, and to compute two capacities in the [exercises](../../book-exercises/part01-prerequisites/05-information-theory.md).

![Left: mutual information of a binary symmetric channel as a function of the input prior, for several flip probabilities; every curve peaks at $q = \frac12$. Right: the capacity $1 - h(p)$ against $p$, with the min-capacity of @sec-min-capacity for comparison.](figures/information_theory_bsc.svg){#fig-bsc}

### Example: a storage medium that flips bits

A cheap storage medium flips each stored bit independently with probability $p = 0.11$. Its capacity is $1 - h(0.11) = 1 - 0.4999 = 0.5001$ bits per stored bit. So with good error-correcting codes you can store about one bit of data per two raw bits, with an error rate as small as you like, and by the channel coding theorem no code can do better. At $p = 0.25$ the capacity drops to $1 - h(0.25) = 1 - 0.8113 = 0.1887$. Since $1/0.1887 \approx 5.3$, that is about five raw bits per data bit.

### Check yourself

- Why does the derivation not need to compute $H(Y)$ for a general $q$? *(Because $H(Y \mid X)$ does not depend on $q$, so maximizing $I$ means maximizing $H(Y)$, and its largest possible value 1 is attainable.)*
- What is the capacity of a channel whose output is independent of its input? *(Zero: $I(X; Y) = 0$ for every prior, because mutual information is zero exactly under independence.)*

## Min-entropy and guessing {#sec-min-entropy}

### Context

Shannon entropy counts the bits needed to describe a value on average. A different question is how likely you are to guess it in **one try**. A password chosen by a person might have high entropy, because most people choose something unusual, yet be guessed one time in ten, because one popular choice dominates. For the one-try question, all that matters is the probability of the most likely value: that is the best you can do with one guess.

### Statement

The **vulnerability** of $X$ is the success probability of the best single guess, the largest probability of any value. To put it on the same bit scale as entropy, take its negative log; that is the **min-entropy**:
$$V(X) = \max_x p(x), \qquad H_\infty(X) = -\log V(X).$$
A fair coin has $V = \frac12$ and $H_\infty = 1$ bit, the same as its Shannon entropy. In general, min-entropy uses the surprisal of the most likely outcome, which is the smallest surprisal, while Shannon entropy averages all surprisals. So min-entropy is never larger than Shannon entropy:
$$H_\infty(X) \le H(X) \le \log\lvert\mathcal X\rvert,$$
with equality in the first inequality iff $X$ is uniform on its **support**, the set of values it takes with positive probability.

### Derivation

*The best single guess.* If you guess $g$, you are right with probability $p(g)$. The best you can do is pick $g$ with the largest $p(g)$, so the optimal success probability is $\max_x p(x)$.

*$H_\infty \le H$.* The idea is that an average of numbers is at least their smallest value.

1. Every outcome has $p(x) \le \max_{x'}p(x') = V(X)$, so its surprisal satisfies $\log\frac{1}{p(x)} \ge \log\frac{1}{V(X)}$.
2. Average over $X$: $H(X) = \mathbb E[\log\frac{1}{p(X)}] \ge \log\frac{1}{V(X)} = H_\infty(X)$.
3. Equality needs every surprisal to equal the smallest one, $p(x) = V(X)$ for every $x$ with $p(x) > 0$: all outcomes in the support are equally likely.

### Interpretation

Min-entropy has a direct reading: $H_\infty$ is the number of bits of a uniform distribution that is exactly as easy to guess as $X$. A uniform $k$-bit string is guessed in one try with probability $2^{-k}$, so a value with $H_\infty = k$ bits is as hard to guess in one try as a uniform $k$-bit string, no matter how large $H(X)$ is.

Both entropies belong to one family. The **Rényi entropy** of order $\alpha$ is $H_\alpha(X) = \frac{1}{1 - \alpha}\log\sum_x p(x)^\alpha$. As $\alpha$ grows, the sum $\sum_x p(x)^\alpha$ is dominated more and more by the largest probability, and as $\alpha \to \infty$ the formula tends to $-\log\max_x p(x) = H_\infty$. As $\alpha \to 1$ the formula becomes $0/0$, and its limit is Shannon's $H$. This parallels the family of Rényi divergences in [Divergences](03-divergences.md).

For a coin, @fig-binary-entropy shows the gap. The min-entropy $H_\infty = -\log\max(p, 1 - p)$ is smooth on each side of $\frac12$ but has a kink there, where the most likely outcome switches from tails to heads, and it lies below $h(p)$.

### Example: a PIN with one popular choice

A uniform 4-digit PIN has $H = H_\infty = \log_2 10{,}000 = 13.2877$ bits, and a single guess succeeds with probability $10^{-4}$. Now suppose 10% of people choose 1234 and the rest choose uniformly among the other 9,999 PINs. The script gives two very different numbers.

The Shannon entropy is $H = 12.4278$ bits, less than a bit below the uniform case. You can see where it comes from with the chain rule: first reveal whether the PIN is 1234, which costs $h(0.1) = 0.4690$ bits, and then, in the 90% of cases where it is not, reveal which of the 9,999 others it is, which costs $\log_2 9{,}999$ bits. That gives $0.4690 + 0.9\log_2 9{,}999 = 0.4690 + 11.9588 = 12.4278$.

The min-entropy is $H_\infty = \log_2 10 = 3.3219$ bits, because the best guess, "1234", succeeds with probability $0.1$, and $-\log_2 0.1 = \log_2 10$.

The Shannon entropy says this PIN distribution is almost as good as uniform. The min-entropy says that one guess in ten succeeds. If the guesser gets one try, only the second number matters.

### Check yourself

- Which is larger for any $X$: $V(X)$ or $2^{-H(X)}$? *($V(X)$. $H_\infty \le H$ means $-\log V \le H$, so $\log V \ge -H$ and $V \ge 2^{-H}$.)*
- Give a distribution on a million values whose Shannon entropy is above 10 bits but whose min-entropy is 1 bit. *(Probability $\frac12$ on one value and $\frac12$ spread uniformly over the other 999,999. The best guess wins with probability $\frac12$, so $H_\infty = 1$, while by the chain rule $H = 1 + \frac12\log_2 999{,}999 \approx 10.97$: one bit to say which half, then, half the time, the index among 999,999.)*

## Conditional min-entropy and min-entropy leakage {#sec-min-capacity}

### Context

Now the guesser sees an observation $Y$ before guessing the hidden value $X$, which this section calls the secret. How much does that help? The natural answer, from Smith (2009), keeps the meaning of vulnerability as a success probability: the posterior vulnerability is the success probability of the best guessing strategy that may depend on what was observed. Comparing it with the vulnerability before the observation gives the **min-entropy leakage**, and maximizing that over priors gives the **min-capacity**, the one-guess analogue of channel capacity.

### Statement

The guesser who sees $y$ should guess the value of $X$ that is most likely given $y$. Its success probability is then the average over $y$ of $\max_x p(x \mid y)$, and since $p(y)\max_x p(x \mid y) = \max_x p(x, y)$, this is the sum of the largest entry in each column of the joint table. For a joint distribution $p(x, y)$, the **posterior vulnerability** and the **conditional min-entropy** are
$$V(X \mid Y) = \sum_y\max_x p(x, y), \qquad H_\infty(X \mid Y) = -\log V(X \mid Y).$$ {#eq-posterior-vulnerability}
How much the observation helps is the factor by which it multiplies the chance of a correct guess, measured in bits by its log. For a prior $\pi$ and channel $C$, the **min-entropy leakage** is
$$\mathcal L(\pi, C) = H_\infty(X) - H_\infty(X \mid Y) = \log\frac{V(X \mid Y)}{V(X)}.$$
The leakage depends on the prior. Its largest value over all priors, the **min-capacity**, depends only on the channel. It has a closed form, the log of the sum of the column maxima of the channel table, and it is attained at the uniform prior:
$$\mathcal{ML}(C) = \max_\pi\mathcal L(\pi, C) = \log\sum_y\max_x C[x, y].$$ {#eq-min-capacity}
It satisfies $0 \le \mathcal{ML}(C) \le \log\min(\lvert\mathcal X\rvert, \lvert\mathcal Y\rvert)$: an observation never hurts the guesser, and it cannot multiply the chance of success by more than the number of possible secrets or the number of possible observations.

### Derivation

*Posterior vulnerability is the success of the best strategy.* The idea is that the guesser's choice for one observation does not affect its success on any other observation, so it can make the best choice column by column.

1. A strategy assigns a guess $g(y)$ to each observation $y$. It succeeds when $X = g(Y)$, which happens with probability $\sum_y p(g(y), y)$: the strategy picks one cell from each column of the joint table, and its success is the sum of the picked cells.
2. Each column can be optimized separately, since $g(y)$ affects only the term for $y$. The best choice in column $y$ is the row with the largest entry. Dividing a column by $p(y)$ does not change which row is largest, so this is also the value of $x$ with the largest posterior probability $p(x \mid y)$, the **maximum a posteriori (MAP) guess** $\arg\max_x p(x, y) = \arg\max_x p(x \mid y)$.
3. So the best success probability is $\sum_y\max_x p(x, y)$, which is @eq-posterior-vulnerability. Writing $p(x, y) = p(y)\,p(x \mid y)$ gives the equivalent form $V(X \mid Y) = \sum_y p(y)\max_x p(x \mid y)$: the average over $y$ of the vulnerability $V(X \mid Y = y)$ of the posterior distribution.

*Min-capacity.* The idea is to bound the posterior vulnerability by pulling the prior out of each column maximum, and then to notice that the uniform prior loses nothing in that step.

1. Write $p(x, y) = \pi(x)C[x, y]$. In each column, the largest product is at most the largest prior times the largest channel entry: $\max_x\pi(x)C[x, y] \le \big(\max_x\pi(x)\big)\big(\max_x C[x, y]\big) = V(X)\max_x C[x, y]$.
2. Sum over $y$ and divide by $V(X)$: $\frac{V(X \mid Y)}{V(X)} \le \sum_y\max_x C[x, y]$, for every prior. Taking logs, $\mathcal L(\pi, C) \le \log\sum_y\max_x C[x, y]$.
3. At the uniform prior $\pi(x) = 1/\lvert\mathcal X\rvert$, the inequality in step 1 is an equality, because the constant factor $\pi(x) = V(X)$ comes out of the max. So the bound is attained, and @eq-min-capacity follows.
4. *Range.* For the upper bound, there are two ways to bound the sum of column maxima. Each column maximum is at most 1, so the sum is at most $\lvert\mathcal Y\rvert$. Each column maximum is also at most its column sum, and all column sums together add up to $\sum_x\sum_y C[x, y] = \lvert\mathcal X\rvert$, because each of the $\lvert\mathcal X\rvert$ rows sums to 1. For the lower bound, fix any row $x_0$: each column maximum is at least $C[x_0, y]$, and these add up to 1 across the row, so the sum of column maxima is at least 1 and $\mathcal{ML} \ge \log 1 = 0$.

### Interpretation

Leakage is a ratio of success probabilities on a log scale: $\mathcal L = 1$ bit means the observation doubles the chance of a correct first guess, and 2 bits means it quadruples it. That is a statement an engineer can act on, and it is why min-entropy is the usual measure in the field of quantitative information flow, which studies how much a program's outputs reveal about its secret inputs.

**Smith's definition is not the average of $H_\infty(X \mid Y = y)$.** It averages the posterior vulnerabilities and then takes the log, $-\log\sum_y p(y)V(X \mid Y = y)$, rather than averaging the logs, $\sum_y p(y)\big(-\log V(X \mid Y = y)\big)$. These differ, and Jensen's inequality (Fact 1, in the mirror version for the convex $-\log$: the $-\log$ of an average is at most the average of $-\log$) says Smith's value is never larger. The reason to prefer Smith's version is its meaning: it is the log of an actual success probability, that of the MAP guesser. The average of logs is not the log of any success probability.

The min-capacity is the worst case over the prior. That matters whenever you do not know the prior: a single number, computed from the channel alone, bounds the leakage for every prior. It is also cheap to compute, one maximum per column, where the Shannon capacity needs an iterative algorithm.

### Example: walking through a small table

A secret $X$ takes four values with prior $\pi = (0.4, 0.3, 0.2, 0.1)$. An observation $Y$ takes three values, through the channel in @tbl-guess-channel.

| $C[x, y]$ | $y_1$ | $y_2$ | $y_3$ |
| --- | --- | --- | --- |
| $x_1$ | 0.8 | 0.2 | 0 |
| $x_2$ | 0.1 | 0.8 | 0.1 |
| $x_3$ | 0 | 0.5 | 0.5 |
| $x_4$ | 0 | 0.2 | 0.8 |

: The channel of the guessing walk-through: rows are secret values, columns are observations. {#tbl-guess-channel}

Multiply each row by its prior to get the joint table, @tbl-guess-joint, also drawn in @fig-guessing. For example, the first row is $0.4 \times (0.8, 0.2, 0) = (0.32, 0.08, 0)$.

| $p(x, y)$ | $y_1$ | $y_2$ | $y_3$ |
| --- | --- | --- | --- |
| $x_1$ | **0.32** | 0.08 | 0 |
| $x_2$ | 0.03 | **0.24** | 0.03 |
| $x_3$ | 0 | 0.10 | **0.10** |
| $x_4$ | 0 | 0.02 | 0.08 |
| $p(y)$ | 0.35 | 0.44 | 0.21 |

: The joint distribution $p(x, y) = \pi(x)C[x, y]$ of the walk-through, with the column maxima in bold. {#tbl-guess-joint}

1. **Without the observation**, the best guess is the most likely value $x_1$, right with probability $V(X) = 0.4$. So $H_\infty(X) = -\log_2 0.4 = 1.3219$ bits.
2. **With the observation**, guess the row with the largest entry in each column (in bold): $x_1$ on $y_1$, $x_2$ on $y_2$, $x_3$ on $y_3$. The success probability is $0.32 + 0.24 + 0.10 = 0.66 = V(X \mid Y)$, and $H_\infty(X \mid Y) = -\log_2 0.66 = 0.5995$ bits. Notice that on $y_3$ the guesser picks $x_3$ although the channel entry $C[x_4, y_3] = 0.8$ is larger than $C[x_3, y_3] = 0.5$. The guesser compares joint probabilities, not channel entries, and $x_3$ is twice as likely as $x_4$ a priori, which more than makes up the difference: $0.2 \times 0.5 = 0.10$ beats $0.1 \times 0.8 = 0.08$.
3. **Simulation.** The script draws $(X, Y)$ a million times and lets the MAP guesser play; it wins 0.6594 of the time, close to the predicted 0.66.
4. **Leakage.** $\mathcal L = \log_2(0.66/0.4) = 0.7225$ bits: the observation multiplies the guessing success by $0.66/0.4 = 1.65$. For comparison, the Shannon leakage of the same table, the mutual information, is $I(X; Y) = 0.6865$ bits out of $H(X) = 1.8464$.
5. **Min-capacity.** The column maxima of the channel $C$ (not of the joint table) are 0.8, 0.8, 0.8, so $\mathcal{ML}(C) = \log_2(0.8 + 0.8 + 0.8) = \log_2 2.4 = 1.2630$ bits. Under the uniform prior, the leakage is exactly this value, and the script confirms that no prior on a grid of step 0.05 does better. The Shannon capacity of the same channel, from Blahut–Arimoto, is 0.8269 bits.
6. **Smith versus the average of logs.** The posterior vulnerabilities are the column maxima divided by the column totals, $0.32/0.35$, $0.24/0.44$ and $0.10/0.21$. Their $-\log_2$ are 0.1293, 0.8745 and 1.0704 bits, and averaging these with weights $p(y) = 0.35, 0.44, 0.21$ gives 0.6548 bits, larger than Smith's 0.5995, as Jensen predicts.

![The joint distribution $p(x, y) = \pi(x)C[x, y]$ of the walk-through. The MAP guesser picks the boxed cell in each column, and its success probability is their sum $V(X \mid Y) = 0.66$.](figures/information_theory_guessing.svg){#fig-guessing}

### Check yourself

- What is the min-capacity of the binary symmetric channel? *($\log_2(2\max(p, 1 - p))$: each of the two columns has maximum $\max(p, 1 - p)$. At $p = 0.11$ it is $\log_2 1.78 = 0.8319$ bits, against a Shannon capacity of 0.5001. It is the dashed curve in @fig-bsc.)*
- Why can the min-entropy leakage for a given prior be smaller than the min-capacity? *(A skewed prior already makes the best blind guess good, so the observation adds less. Here 0.7225 bits at the given prior against 1.2630 at the uniform one.)*

## Shannon leakage versus min-entropy leakage {#sec-shannon-vs-min}

### Context

Shannon leakage, the mutual information $I(X; Y)$, and min-entropy leakage $\mathcal L$ both measure how much an observation reveals about a secret. They often agree. When they don't, the disagreement is large, and it is always the same kind: a channel that reveals the secret completely on a few outcomes and nothing on the rest. Shannon leakage averages logs, so it gives those rare outcomes little weight; min-entropy leakage asks how much the guesser's success grows, and a rare full reveal can make it grow a lot. This section works through such channels.

### Statement

Let $X$ be uniform on $n$-bit strings, and consider two families of **deterministic** channels, whose output is a fixed function of the input. The first reveals part of the secret every time; the second reveals all of it, but only rarely.

- **reveal-low($m$)** always outputs the lowest $m$ bits of $X$.
- **reveal-if-zero($j$)** outputs $X$ itself if its lowest $j$ bits are all zero, which happens with probability $2^{-j}$, and a fixed symbol $\perp$ otherwise.

For the first family both leakages equal $m$. For the second, the Shannon leakage is about $n$ bits times the probability $2^{-j}$ of a reveal, plus a small term, while the min-entropy leakage is about $n - j$ bits:
$$\text{reveal-low}(m):\quad I(X; Y) = \mathcal L = m,$$
$$\text{reveal-if-zero}(j),\ j \ge 1:\quad I(X; Y) = 2^{-j}n + (1 - 2^{-j})\log\frac{1}{1 - 2^{-j}}, \qquad \mathcal L = \log\big(2^{n-j} + 1\big).$$

### Derivation

Two observations about deterministic channels make both leakages easy to compute. For Shannon leakage: the output is fixed once the input is known, so $H(Y \mid X) = 0$ and $I(X; Y) = H(Y) - H(Y \mid X) = H(Y)$. For min-entropy leakage: under the uniform prior, $V(X) = 2^{-n}$, and each column of the joint table contains only entries $2^{-n}$ (for inputs that produce that output) or 0. So each column maximum is $2^{-n}$, and $V(X \mid Y) = 2^{-n}\times(\text{number of distinct outputs})$. The ratio is $V(X \mid Y)/V(X) = \text{number of distinct outputs}$, so $\mathcal L = \log(\text{number of distinct outputs})$.

*reveal-low($m$).* The lowest $m$ bits of a uniform string are uniform on $2^m$ values, so $I(X; Y) = H(Y) = m$. There are $2^m$ distinct outputs, so $\mathcal L = m$.

*reveal-if-zero($j$).*

1. Count the outputs and their probabilities. There are $2^{n-j}$ strings whose lowest $j$ bits are zero (the other $n - j$ bits are free). Each is output as itself, with probability $2^{-n}$. Every other string gives the output $\perp$, which therefore has probability $1 - 2^{n-j}\cdot2^{-n} = 1 - 2^{-j}$.
2. Compute $H(Y)$ as the average surprisal. The $2^{n-j}$ revealed outputs each have surprisal $\log 2^n = n$, and $\perp$ has surprisal $\log\frac{1}{1 - 2^{-j}}$. So $H(Y) = 2^{n-j}\cdot2^{-n}\log2^n + (1 - 2^{-j})\log\frac{1}{1 - 2^{-j}} = 2^{-j}n + (1 - 2^{-j})\log\frac{1}{1 - 2^{-j}}$.
3. The number of distinct outputs is the $2^{n-j}$ revealed strings plus $\perp$, which is $2^{n-j} + 1$, and its log gives $\mathcal L$.

The script checks both closed forms against the general formulas applied to the explicit channel matrices for $n = 6$; they agree to $7.8\times10^{-16}$, which is rounding error.

### Interpretation

@fig-leakage plots the two leakages against each other for $n = 64$. The reveal-low family lies on the diagonal: Shannon and min-entropy agree. The reveal-if-zero family hugs the top-left corner: almost no Shannon leakage, almost total min-entropy leakage.

The reason is where the logarithm sits. Shannon leakage averages the log over outcomes, so an outcome of probability $2^{-j}$ that reveals $n$ bits contributes only $2^{-j}n$. Min-entropy leakage takes the log of an averaged probability. The same outcome raises the guessing success from $2^{-n}$ to more than $2^{-j}$, because the guesser wins whenever the reveal happens, and $\log(2^{-j}/2^{-n}) = n - j$: almost $n - j$ bits.

![Shannon leakage against min-entropy leakage for two families of deterministic channels on a uniform 64-bit secret. Revealing the low $m$ bits gives equal leakages (diagonal). Revealing the whole secret with probability $2^{-j}$ gives a small Shannon leakage but a min-entropy leakage close to $64 - j$.](figures/information_theory_leakage.svg){#fig-leakage}

### Example: two channels that Shannon ranks the wrong way

Take $n = 64$ and compare reveal-if-zero(3), which outputs the whole secret one time in eight, with reveal-low(9).

For reveal-if-zero(3), the Shannon leakage is $2^{-3}\cdot64 + \frac78\log_2\frac87 = 8 + 0.1686 = 8.1686$ bits. The guesser wins whenever the secret is revealed, and otherwise has one chance in $2^{64} - 2^{61}$, so its success is just above $\frac18 = 0.1250$; in bits, $\log_2(2^{61} + 1)$ is 61 to many decimal places. For reveal-low(9), both leakages are 9 bits, and the guesser, knowing 9 of the 64 bits, must still guess the other 55, so it succeeds with probability $2^{-55} \approx 2.78\times10^{-17}$. @tbl-two-channels puts these side by side.

| Channel | Shannon leakage | Min-entropy leakage | One-guess success |
| --- | --- | --- | --- |
| reveal-if-zero(3) | 8.1686 bits | 61.0000 bits | 0.1250 |
| reveal-low(9) | 9.0000 bits | 9.0000 bits | $2.78\times10^{-17}$ |

: Shannon and min-entropy leakage of two deterministic channels on a uniform 64-bit secret. {#tbl-two-channels}

By Shannon leakage, reveal-low(9) is the leakier channel. By one-guess success, reveal-if-zero(3) is worse by a factor of more than $10^{15}$: $0.125/(2.78\times10^{-17}) \approx 4.5\times10^{15}$. Smith (2009) used a pair of programs like these to argue that Shannon leakage is the wrong measure for one-try guessing.

**The extreme case: reveal the secret only if it is all zeros.** Set $j = n$: the channel outputs $X$ if $X = 0\cdots0$ and $\perp$ otherwise. For a 4-bit secret, the output reveals the secret with probability $\frac1{16}$, so the Shannon leakage is $H(Y) = h(1/16) = 0.3373$ bits. The min-entropy leakage is exactly 1 bit: the guesser wins if it sees $0000$, and otherwise guesses one of the 15 remaining values, winning with probability $\frac1{16} + \frac{15}{16}\cdot\frac1{15} = \frac2{16}$. Its success doubles from $\frac1{16}$ to $\frac2{16}$, and $\log_2 2 = 1$. For a 32-bit secret the Shannon leakage falls to $h(2^{-32}) = 7.786\times10^{-9}$ bits, and the min-entropy leakage stays at 1 bit, by the same doubling argument.

Both numbers are small, and both miss something: for the one secret value $0000$, the channel reveals everything, $H_\infty(X \mid Y = 0000) = 0$. Any measure that averages over the secret, Shannon or min-entropy, can hide a value that is fully exposed, because that value carries only its small share of the average. A worst-case question ("how much can the observation reveal about any single secret value?") needs a worst-case measure, and later parts build one.

### Check yourself

- For reveal-if-zero($j$) at $n = 64$, roughly where does the Shannon leakage drop below 1 bit? *(Between $j = 6$ and $j = 7$: the main term $2^{-j}\cdot64$ equals 1 at $j = 6$, where the total is just above 1 bit because of the small $\perp$ term. That is the point labelled $j = 6$ in @fig-leakage.)*
- Can a channel have min-entropy leakage 0 but positive Shannon leakage? *(Yes. If the MAP guess is the same for every $y$, as when one secret value is very likely a priori, then $V(X \mid Y) = V(X)$, even though $Y$ changes the posterior.)*

## Fano's inequality

### Context

Mutual information also gives **lower bounds**: statements that *no* estimator can do better than some error rate. You need these whenever you want to show that a problem is hard, rather than that a method is good. Fano's inequality is the basic tool. It says that if $X$ is still uncertain after seeing $Y$, every guess of $X$ from $Y$ must often be wrong. The reason is that an estimator with a small error rate would leave little uncertainty: if you knew you were usually right, you would be nearly certain of $X$.

### Statement

Let $X$ take values in a finite set of size $M$, and let $\hat X$ be any estimate of $X$ computed from $Y$ (deterministically or with extra randomness), with error probability $P_e = \Pr[\hat X \ne X]$. Fano's inequality bounds how much uncertainty a guess with this error rate can leave. Describing $X$ given the guess takes at most two pieces: one binary piece saying whether the guess was right, worth at most $h(P_e)$ bits, and, in the fraction $P_e$ of cases where it was wrong, which of the other $M - 1$ values $X$ is, worth at most $\log(M - 1)$ bits. So
$$H(X \mid Y) \le h(P_e) + P_e\log(M - 1).$$ {#eq-fano}
Using $h(P_e) \le 1$ and $\log(M - 1) \le \log M$ gives $H(X \mid Y) \le 1 + P_e\log M$, and solving for $P_e$ gives the weaker but explicit form
$$P_e \ge \frac{H(X \mid Y) - 1}{\log M}.$$ {#eq-fano-weak}

### Derivation

Introduce the error indicator $E = \mathbf 1[\hat X \ne X]$, which is 1 when the guess is wrong and 0 when it is right, so $\Pr[E = 1] = P_e$. The idea is to compute the uncertainty of the pair $(E, X)$ given $\hat X$, written $H(E, X \mid \hat X)$, in two ways with the chain rule. Revealing $X$ first makes $E$ free; revealing $E$ first splits $X$'s uncertainty into the "right" and "wrong" cases, which are easy to bound. Every entropy below is conditioned on $\hat X$.

1. **First order: $X$, then $E$.** $H(E, X \mid \hat X) = H(X \mid \hat X) + H(E \mid X, \hat X)$. Given both $X$ and $\hat X$, you know whether they differ, so $E$ is determined and the last term is 0.
2. **Second order: $E$, then $X$.** $H(E, X \mid \hat X) = H(E \mid \hat X) + H(X \mid E, \hat X)$.
3. **Bound $H(E \mid \hat X)$.** Conditioning reduces entropy on average, so $H(E \mid \hat X) \le H(E)$, and $E$ is a Bernoulli variable with parameter $P_e$, so $H(E) = h(P_e)$.
4. **Bound $H(X \mid E, \hat X)$** by splitting on $E$. If $E = 0$, then $X = \hat X$ and there is no uncertainty left. If $E = 1$, then $X$ is one of the $M - 1$ values other than $\hat X$, so its entropy is at most $\log(M - 1)$ by the entropy upper bound. Averaging over $E$: $H(X \mid E, \hat X) \le (1 - P_e)\cdot0 + P_e\log(M - 1)$.
5. **Combine** steps 1–4. The two orders give the same quantity, so $H(X \mid \hat X) = H(E \mid \hat X) + H(X \mid E, \hat X) \le h(P_e) + P_e\log(M - 1)$.
6. **Replace $\hat X$ by $Y$.** The estimate is computed from $Y$, so $X \to Y \to \hat X$ is a Markov chain, and the data-processing inequality gives $I(X; \hat X) \le I(X; Y)$. Writing both sides as $H(X)$ minus a conditional entropy, this is the same as $H(X \mid Y) \le H(X \mid \hat X)$, and the bound of step 5 carries over.

### Interpretation

The right side of @eq-fano increases with $P_e$ on $[0, (M - 1)/M]$, the range from a perfect estimator to one that guesses blindly. So you can invert it numerically: the **Fano bound** is the smallest $P_e$ that satisfies the inequality for the given $H(X \mid Y)$. The script finds it by bisection, halving an interval that contains it until it is small enough.

**When it is tight.** Equality needs step 4 to be tight: when the guess is wrong, the true value must be uniform over the other $M - 1$ values. The $M$-ary **symmetric channel**, which keeps the input with probability $1 - \varepsilon$ and otherwise outputs one of the other $M - 1$ symbols uniformly, has exactly this structure. For it, given the output, the input is the output itself with probability $1 - \varepsilon$ and otherwise uniform over the rest, so $H(X \mid Y) = h(\varepsilon) + \varepsilon\log(M - 1)$, and the MAP guess (the output itself) errs with probability $\varepsilon$. Fano's bound is exact (@fig-fano, left).

**When it is loose.** If errors are concentrated on a few wrong values, $H(X \mid Y)$ is small even when $P_e$ is not, because the leftover uncertainty after a wrong guess is small. The **shift channel**, which outputs $x + 1 \bmod M$ instead of $x$ with probability $\varepsilon$, has the same MAP error $\varepsilon$ (for $\varepsilon \le \frac12$). But given the output, the input is one of only two values, so $H(X \mid Y) = h(\varepsilon)$ only. At $M = 8$ and $\varepsilon = 0.2$, Fano's bound is 0.2000 for the symmetric channel, equal to the MAP error, but only 0.0954 for the shift channel, against a MAP error of 0.2000 (@fig-fano, right). The weak form @eq-fano-weak gives 0.0945 and 0 respectively. For the symmetric channel, $H(X \mid Y) = h(0.2) + 0.2\log_2 7 = 0.7219 + 0.5615 = 1.2834$, and $(1.2834 - 1)/3 = 0.0945$. For the shift channel, $H(X \mid Y) = h(0.2) = 0.7219$ is below 1, so the weak form gives a negative number, which says nothing beyond $P_e \ge 0$. The weak form pays for replacing $h(P_e)$ by 1, and is useless whenever $H(X \mid Y) \le 1$.

![MAP error against Fano's lower bound, for $M = 8$ and a uniform $X$, as the noise level $\varepsilon$ grows. Left: on the symmetric channel the bound is exact. Right: on the shift channel, where errors go to a single wrong value, the bound is loose and the weak form is zero.](figures/information_theory_fano.svg){#fig-fano}

### Example: how hard is a classification problem?

**Eight classes and one bit.** A label $X$ is uniform over $M = 8$ classes, so $H(X) = \log_2 8 = 3$ bits, and you know the features $Y$ carry at most $I(X; Y) \le 1$ bit about it. Then $H(X \mid Y) = 3 - I(X; Y) \ge 2$ bits. Every classifier, however it is trained, therefore has error at least the Fano bound at $H(X \mid Y) = 2$, the $P_e$ that solves $h(P_e) + P_e\log_2 7 = 2$: 0.3730. The weak form gives $(2 - 1)/3 = 0.3333$.

**Fano's method for estimation.** The same idea shows that estimating a parameter needs many samples. Turn estimation into identification: pick $M$ candidate parameter values $\theta_1, \dots, \theta_M$ that are far apart, so that estimating the parameter well would identify which candidate it is. Let $X$ be a uniformly random index, and draw $n$ independent samples $Y_1, \dots, Y_n$ (together written $Y^n$) from $P_{\theta_X}$. If the samples carry little information about the index, Fano says no estimator can identify it, so none can estimate the parameter well. The argument has three steps.

1. **Information adds over independent samples.** Given $X$ the samples are independent, so the chain rule gives $H(Y^n \mid X) = \sum_i H(Y_i \mid X)$ exactly. Without conditioning on $X$, the chain rule plus "conditioning reduces entropy" gives only $H(Y^n) \le \sum_i H(Y_i)$. Subtracting the first from the second, $I(X; Y^n) \le \sum_i I(X; Y_i) = n\,I(X; Y_1)$, since all samples have the same distribution.
2. **Each sample carries at most the largest pairwise KL divergence.** Let $\bar P = \frac1M\sum_x P_{\theta_x}$ be the average of the candidate distributions, which is the distribution of $Y_1$ when the index is random. By the KL form of @eq-mi-kl, $I(X; Y_1)$ is the average KL divergence from each candidate to this average, $\frac1M\sum_x D(P_{\theta_x}\,\|\,\bar P)$. Expanding the logs shows that for any distribution $Q$, $\frac1M\sum_x D(P_{\theta_x}\,\|\,Q) = \frac1M\sum_x D(P_{\theta_x}\,\|\,\bar P) + D(\bar P\,\|\,Q)$, so the average divergence to $\bar P$ is the smallest average divergence to any $Q$. Taking $Q = P_{\theta_{x'}}$ gives $I(X; Y_1) \le \frac1M\sum_x D(P_{\theta_x}\,\|\,P_{\theta_{x'}})$ for each $x'$, and hence $I(X; Y_1) \le \kappa = \max_{x, x'}D(P_{\theta_x}\,\|\,P_{\theta_{x'}})$.
3. **Apply Fano.** The index is uniform, so $H(X) = \log M$ and $H(X \mid Y^n) = \log M - I(X; Y^n) \ge \log M - n\kappa$. Putting this into @eq-fano-weak gives
$$P_e \ge 1 - \frac{n\kappa + 1}{\log M}.$$

With $M = 1024$ candidates ($\log M = 10$ bits) whose sample distributions are within $\kappa = 0.01$ bits of each other, $n = 100$ samples leave $P_e \ge 1 - (100 \times 0.01 + 1)/10 = 0.8000$, and even $n = 400$ samples leave $P_e \ge 1 - (4 + 1)/10 = 0.5000$. Choosing the candidates well, so that they are many and far apart yet hard to tell apart, is the art of the method; Wainwright (2019) works through many cases.

### Check yourself

- If $X$ is a deterministic function of $Y$, what does Fano's inequality say? *($H(X \mid Y) = 0$, so it allows $P_e = 0$, as it should.)*
- Why does step 6 of the derivation need the data-processing inequality? *(Steps 1–5 only bound $H(X \mid \hat X)$. The bound must be stated in terms of what the estimator sees, $Y$, and the data-processing inequality says $\hat X$ knows no more about $X$ than $Y$ does.)*

## Code: check every formula numerically

The script [code/information_theory.py](../../code/part01-prerequisites/information_theory.py) computes every number quoted in this chapter and draws its figures. Entropy, mutual information, vulnerability, leakage, capacity and the Fano bound are small typed functions of a joint table or a channel matrix; the script calls them once for the numbers and loops them over parameters for the figures. From the repository root:

```bash
uv run python code/part01-prerequisites/information_theory.py
```

@tbl-info-output summarizes the output (seed 0; only the guessing simulation uses the seed).

| Experiment | Result |
| --- | --- |
| Spam keyword | $H(X) = 0.5436$, $H(X \mid Y = 1) = 1.0000$, $H(X \mid Y) = 0.2500$, $I(X; Y) = 0.2936$ (from $H(X) - H(X \mid Y)$ and from the KL form); $I(X; Z) = 0.1668$ through a $\mathrm{BSC}(0.1)$ |
| PIN, 10% choose 1234 | $H = 12.4278$, $H_\infty = 3.3219$ (uniform: 13.2877) |
| $\mathrm{BSC}(0.11)$ | $1 - h(p) = 0.5001$; Blahut–Arimoto 0.5001 at $\Pr[X = 1] = 0.5000$; min-capacity 0.8319 |
| Guessing walk-through | $V(X) = 0.4000$, $V(X \mid Y) = 0.6600$, simulated 0.6594; $\mathcal L = 0.7225$; $\mathcal{ML} = 1.2630$; Shannon capacity 0.8269 |
| Reveal-if-zero(3) vs reveal-low(9), $n = 64$ | Shannon 8.1686 vs 9.0000; min-entropy 61.0000 vs 9.0000 |
| Fano, $M = 8$, $\varepsilon = 0.2$ | Symmetric: bound 0.2000, MAP 0.2000. Shift: bound 0.0954, MAP 0.2000 |

: Output of the chapter's script with seed 0. All quantities are in bits except probabilities. {#tbl-info-output}

[tests/test_information_theory.py](../../tests/test_information_theory.py) checks that the output still matches.

Five lessons come out of these runs. The first is that two formulas give one number: mutual information computed as an entropy difference and as a KL divergence agree, which is a useful check on any implementation. The second is that the closed forms are right. Blahut–Arimoto, which knows nothing about the binary symmetric channel, finds $1 - h(p)$ and the uniform input, and the leakage closed forms match the explicit matrices. The third is that vulnerability is a real success rate: the simulated MAP guesser wins at the rate $V(X \mid Y)$ predicts. The fourth is that Shannon and min-entropy measure different things. They can rank channels in opposite orders, and neither sees a single exposed secret value. The last is that Fano's bound is tight only when errors are spread out over the wrong values; concentrated errors make it loose.

## References

- Cover & Thomas, *Elements of Information Theory* (2nd ed., 2006), ch. 2 (entropy, mutual information, the chain rules in §2.5, Jensen and its consequences in §2.6, the data-processing inequality in §2.8, Fano's inequality in §2.10) and ch. 7 (channel capacity, the binary symmetric channel in §7.1, the channel coding theorem in §7.7): the standard textbook for every Shannon result here
- MacKay, *Information Theory, Inference, and Learning Algorithms* (2003), ch. 2 and 8–10: the same material with a machine-learning slant, free online
- Smith, "On the Foundations of Quantitative Information Flow" (FoSSaCS 2009): vulnerability, conditional min-entropy $V(X \mid Y) = \sum_y\max_x p(x, y)$, and the argument against Shannon leakage for one-try guessing
- Braun, Chatzikokolakis & Palamidessi, "Quantitative Notions of Leakage for One-try Attacks" (MFPS 2009): min-capacity and the fact that the uniform prior attains it
- Alvim, Chatzikokolakis, McIver, Morgan, Palamidessi & Smith, *The Science of Quantitative Information Flow* (2020): the full theory of vulnerability, leakage and capacity, including gain functions that generalize the one-guess adversary
- Wainwright, *High-Dimensional Statistics* (2019), ch. 15: minimax lower bounds, including Fano's method, worked through for many estimation problems
- Yu, "Assouad, Fano, and Le Cam" (1997): the three classical lower-bound methods side by side

## Cheat sheet

@tbl-info-cheat collects the results.

| Quantity | Formula | Key facts |
| --- | --- | --- |
| Entropy | $H(X) = -\sum_x p(x)\log p(x)$ | $0 \le H \le \log\lvert\mathcal X\rvert$; uniform attains the maximum |
| Binary entropy | $h(p) = -p\log p - (1 - p)\log(1 - p)$ | $h(\frac12) = 1$; $h(0.11) \approx \frac12$ |
| Conditional entropy | $H(X \mid Y) = \sum_y p(y)H(X \mid Y = y)$ | $\le H(X)$ on average; one $y$ can raise it |
| Chain rule | $H(X, Y) = H(Y) + H(X \mid Y)$ | Extends to $n$ variables |
| Mutual information | $I(X; Y) = H(X) - H(X \mid Y) = D(P_{XY}\,\|\,P_XP_Y)$ | Symmetric, $\ge 0$, zero iff independent |
| Data processing | $X \to Y \to Z \Rightarrow I(X; Z) \le I(X; Y)$ | From the chain rule for $I$ |
| Capacity | $\mathcal C = \max_\pi I(X; Y)$ | $\mathrm{BSC}(p)$: $1 - h(p)$ at the uniform input |
| Vulnerability, min-entropy | $V(X) = \max_x p(x)$, $H_\infty = -\log V$ | $H_\infty \le H$; one-guess success |
| Posterior vulnerability | $V(X \mid Y) = \sum_y\max_x p(x, y)$ | Success of the MAP guesser |
| Min-entropy leakage | $\mathcal L = \log V(X \mid Y)/V(X)$ | Factor by which guessing success grows |
| Min-capacity | $\mathcal{ML}(C) = \log\sum_y\max_x C[x, y]$ | Attained at the uniform prior; $\le \log\min(\lvert\mathcal X\rvert, \lvert\mathcal Y\rvert)$ |
| Fano | $H(X \mid Y) \le h(P_e) + P_e\log(M - 1)$ | Weak form $P_e \ge (H(X \mid Y) - 1)/\log M$; tight when errors are spread evenly |

: Every result of the chapter. {#tbl-info-cheat}

**Rules of thumb**

- Shannon quantities average $\log$ over outcomes; min-entropy quantities take $\log$ of a best-case probability. When a rare outcome reveals a lot, the two disagree, by orders of magnitude.
- Both are averages over the secret. Neither tells you how exposed a single secret value is.
- To prove an estimator must fail, bound the mutual information between the unknown and the data, then apply Fano.
- Processing never adds information: compute the bound on the raw observation, and it holds for everything computed from it.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part01-prerequisites/05-information-theory.md).
