# Exercises: Attribute inference and model inversion

Exercises for [Attribute inference and model inversion](../../book-full/part02-attacks/03-attribute-inference.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part02-attacks/solutions_attribute_inference.py` computes every number quoted here; run it with `uv run python code/part02-attacks/solutions_attribute_inference.py`.

The solutions use the chapter's notation. A record is $(s, x, y)$, $f_y(x, s)$ is the model's confidence in label $y$, and $V(S \mid O) = \sum_o\max_s p(s, o)$ is the success of the best guess of $s$ from $O$, the maximum a posteriori (MAP) guess. A membership test has a true positive rate (TPR) on members and a false positive rate (FPR) on non-members. In the smoking table of the chapter, $s = 1$ is "smoker", $x$ the age band and $y = 1$ "has the disease".

## A balanced prior {#sec-sol-attr-1}

> **Exercise 1.** **A balanced prior.** Redo the smoking table with $\Pr[\text{smoker}] = 0.5$ and everything else unchanged. Compute $V(S)$, $V(S \mid X)$, $V(S \mid X, Y)$ and the two leakages. *Check: $V(S \mid X) = 0.6$, $V(S \mid X, Y) = 0.7$.* Why does age now change the guess when it did not before? *(MAP guess, vulnerability)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The joint table.** Multiply $\Pr[s] = 0.5$ by $\Pr[x \mid s]$ and $\Pr[y \mid x, s]$. For example, $p(\text{smoker}, \text{old}, \text{disease}) = 0.5 \times 0.6 \times 0.7 = 0.21$.

| $p(s, x, y)$ | young, no disease | young, disease | old, no disease | old, disease |
| --- | --- | --- | --- | --- |
| non-smoker | **0.27** | 0.03 | **0.14** | 0.06 |
| smoker | 0.12 | **0.08** | 0.09 | **0.21** |
| $\Pr[\text{smoker} \mid x, y]$ | 0.3077 | 0.7273 | 0.3913 | 0.7778 |

: The smoking table with a balanced prior, column maxima in bold. {#tbl-sol-attribute-inference-balanced}

2. **No observation.** Both values have probability 0.5, so $V(S) = 0.5$.
3. **Age alone.** Young: non-smoker $0.27 + 0.03 = 0.3$ against smoker $0.12 + 0.08 = 0.2$. Old: non-smoker $0.2$ against smoker $0.3$. The best guess is "non-smoker" for the young and "smoker" for the old, and $V(S \mid X) = 0.3 + 0.3 = 0.6$.
4. **Age and disease.** The bold column maxima add up to $V(S \mid X, Y) = 0.27 + 0.08 + 0.14 + 0.21 = 0.7$. Both check values are right.
5. **Leakages.** From age: $\log_2(0.6/0.5) = 0.2630$ bits. From age and disease: $\log_2(0.7/0.5) = 0.4854$ bits.

**Why age now matters.** Being old multiplies the odds of smoking by $\Pr[\text{old} \mid \text{smoker}]/\Pr[\text{old} \mid \text{non-smoker}] = 0.6/0.4 = 1.5$. With prior odds $0.3/0.7$, the posterior odds for an old person are $\frac37 \times 1.5 = 0.643 < 1$, so the guess stays "non-smoker". With prior odds 1 they are $1.5 > 1$, and the guess flips.

*Whether a piece of evidence changes the best guess depends on the prior as much as on the evidence: a skewed prior absorbs weak evidence.*
:::

## The marginal prior {#sec-sol-attr-2}

> **Exercise 2.** **The marginal prior.** Fredrikson et al.'s attack uses $p(s)$ in place of $p(s \mid x)$. On the smoking table, with a calibrated model, compute the success of $\arg\max_s p(s)\,f_y(x, s)$, breaking ties towards "non-smoker". *Check: 0.706, against 0.748 for imputation.* Which cell causes the loss? *(Bayes' rule, Fact 1)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Scores.** With a calibrated model, $f_y(x, s) = \Pr[y \mid x, s]$. The scores $p(s)\Pr[y \mid x, s]$ for (non-smoker, smoker) in each cell are:
   - young, no disease: $0.7 \times 0.9 = 0.63$ against $0.3 \times 0.6 = 0.18$, guess non-smoker;
   - young, disease: $0.7 \times 0.1 = 0.07$ against $0.3 \times 0.4 = 0.12$, guess smoker;
   - old, no disease: $0.7 \times 0.7 = 0.49$ against $0.3 \times 0.3 = 0.09$, guess non-smoker;
   - old, disease: $0.7 \times 0.3 = 0.21$ against $0.3 \times 0.7 = 0.21$, a tie, broken towards non-smoker.
2. **Success.** Add the mass of the guessed value in each cell of the smoking table: $0.378 + 0.048 + 0.196 + 0.084 = 0.706$. The check value is right.
3. **Where the loss is.** Only the last cell differs from imputation, which guesses "smoker" there and gets $0.126$ instead of $0.084$. The loss is $0.748 - 0.706 = 0.042$.
4. **Why.** Bayes' rule needs $p(s \mid x)$: an old person is a smoker with probability $0.3913$, not $0.3$. With it the scores are $0.3913 \times 0.7 = 0.2739$ against $0.6087 \times 0.3 = 0.1826$, and the tie is broken the right way. Dropping the dependence of $s$ on $x$ throws that evidence away. By Fact 1, any rule other than the MAP guess can only do worse.

*An inversion attack is only as good as the adversary's model of how the sensitive attribute relates to what it already knows.*
:::

## The reduction at low false positive rate {#sec-sol-attr-3}

> **Exercise 3.** **The reduction at low false positive rate.** The membership test $\mathcal B$ built from the toy attack has TPR 0.7832 at FPR 0.7262. Using the mixing of tests in [Hypothesis testing](../../book-full/part01-prerequisites/04-hypothesis-testing.md), find the best TPR it can reach at FPR $10^{-3}$. *Check: $0.00108$.* Why can it not do better? *Hint: its only output is one bit.* *(Neyman–Pearson, mixing tests)*

::: {.callout-tip collapse="true" title="Solution"}
1. **What the test sees.** $\mathcal B$ observes one bit: whether the attribute guess was right. "Right" happens with probability 0.7832 for members and 0.7262 for non-members; "wrong" with 0.2168 and 0.2738.
2. **Every test on that bit.** A test fires with probability $\phi_1$ on "right" and $\phi_0$ on "wrong". Its rates are $\mathrm{FPR} = 0.7262\,\phi_1 + 0.2738\,\phi_0$ and $\mathrm{TPR} = 0.7832\,\phi_1 + 0.2168\,\phi_0$.
3. **Neyman–Pearson.** The likelihood ratio of "right" is $0.7832/0.7262 = 1.0785$, and of "wrong" $0.2168/0.2738 = 0.7918$. The most powerful test spends its false positive budget on the outcome with the larger ratio first: $\phi_0 = 0$ and $\phi_1 = 10^{-3}/0.7262$.
4. **The result.** $\mathrm{TPR} = 0.7832 \times 10^{-3}/0.7262 = 0.00108$. At FPR 0.1 the same calculation gives 0.10785.

**Why not better.** Every test on one bit is a mix of the four deterministic tests "never", "on right", "on wrong" and "always", so its trade-off curve consists of two straight segments. Near the origin the slope is the largest likelihood ratio, 1.0785, so at low FPR the TPR is at most 1.0785 times the FPR. A dedicated membership attack looks at the whole confidence vector and can have a far steeper curve near zero.

*An attribute attack is a membership test that has thrown away everything but one bit, so its power at low false positive rate is barely above chance.*
:::

## Label-only inversion {#sec-sol-attr-4}

> **Exercise 4.** **Label-only inversion.** The hospital releases only a label: "disease" when $\Pr[y = 1 \mid x, s] > \frac12$. On the smoking table, the adversary queries the label at both values of $s$, scores $p(s \mid x)\,\mathbf 1[\mathrm{label}(x, s) = y]$, and breaks ties with $p(s \mid x)$. Compute its success. *Check: 0.742, a loss of 0.006 against imputation.* *(Fact 3, MAP guess)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The labels.** Only the old smoker has disease probability above $\frac12$ (0.7). So the label is "disease" at (old, smoker) and "no disease" at the other three inputs.
2. **Cell by cell.** $p(\text{smoker} \mid \text{young}) = 0.2222$ and $p(\text{smoker} \mid \text{old}) = 0.3913$, so ties go to "non-smoker".
   - young, no disease: both labels match $y$, a tie, guess non-smoker, right on $0.378$;
   - young, disease: neither label matches, a tie, guess non-smoker, right on $0.042$;
   - old, no disease: only the non-smoker label matches, guess non-smoker, right on $0.196$;
   - old, disease: only the smoker label matches, guess smoker, right on $0.126$.
3. **Success.** $0.378 + 0.042 + 0.196 + 0.126 = 0.742$. The check value is right.
4. **The loss.** Imputation guesses "smoker" for young people with the disease and gets $0.048$ there instead of $0.042$: a loss of $0.006$. The label is a function of the confidences, so by Fact 3 a loss, not a gain, was the only possibility.

*A coarse output costs the attacker only in the cells where the coarse answer cannot tell the candidates apart; everywhere else it is as good as the full confidence.*
:::

## Two independent clues {#sec-sol-attr-5}

> **Exercise 5.** **Two independent clues.** Let $s \in \{-1, +1\}$ be uniform, and let $u \sim \mathcal N(as, 1)$ and $v \sim \mathcal N(bs, 1)$ be independent given $s$. Show that the imputation success from $(u, v)$ is $\Phi\big(\sqrt{a^2 + b^2}\big)$. *Check: with $a = 0.5$ and $b = 1$, $\Phi(1.1180) = 0.8682$, against $\Phi(0.5) = 0.6915$ from $u$ alone.* *Hint: the log-likelihood ratios add.* *(Likelihood ratios, Gaussian tails)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Log-likelihood ratio of one clue.** $\ln\frac{\mathcal N(u; a, 1)}{\mathcal N(u; -a, 1)} = \frac{-(u - a)^2 + (u + a)^2}{2} = 2au$. Likewise $2bv$ for $v$.
2. **They add.** Given $s$, the clues are independent, so the joint density is the product and the log-likelihood ratio is $L = 2au + 2bv$.
3. **The MAP guess.** With a uniform prior the posterior odds equal the likelihood ratio, so the MAP guess is $+1$ when $L > 0$ and $-1$ when $L < 0$.
4. **Distribution of $L$ given $s = +1$.** $u$ has mean $a$ and $v$ mean $b$, both with variance 1. So $L$ is normal with mean $2a^2 + 2b^2$ and variance $4a^2 + 4b^2$.
5. **Success.** $\Pr[L > 0 \mid s = +1] = \Phi\Big(\frac{2(a^2 + b^2)}{2\sqrt{a^2 + b^2}}\Big) = \Phi\big(\sqrt{a^2 + b^2}\big)$. By symmetry the same holds for $s = -1$, so this is the overall success.
6. **Numbers.** $\sqrt{0.25 + 1} = 1.1180$ and $\Phi(1.1180) = 0.8682$. A million simulated draws give 0.8686. From $u$ alone ($b = 0$) the success is $\Phi(0.5) = 0.6915$, and from $v$ alone $\Phi(1) = 0.8413$.

*Independent clues add in squared separation, not in success probability: a weak public feature and a strong label combine to more than either, but much less than their sum.*
:::

## A model that memorizes {#sec-sol-attr-6}

> **Exercise 6.** **A model that memorizes.** A lookup model stores each training record under a unique record identifier, such as a patient number. Queried with a person's identifier and a candidate $(x, s, y)$, it returns $f_y(x, s) = 1$ when that person is a training member and the candidate is their stored record, and the population's $p(y \mid x, s)$ for every other query. The lookup is keyed on the identifier, not on the values, because the smoking table has only eight kinds of record and many people share each one. So a non-member's query never hits a stored record, and a member's query hits only at the member's own true values. On the smoking table, compute the success of MAP inversion on members and on non-members. *Check: 0.874 on members, 0.748 on non-members, an advantage of 0.126.* Which members does the attack still get wrong? *(The game, the non-member ceiling)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Non-members.** None of their queries is stored, so the model answers with $p(y \mid x, s)$ at both candidates. It is calibrated, so MAP inversion is imputation and succeeds with $V(S \mid X, Y) = 0.748$.
2. **Members.** At the true value $s$ the model returns 1. At the other value $s'$ it returns the population's $p(y \mid x, s')$. The attack is right exactly when $p(s \mid x) \times 1 > p(s' \mid x)\,p(y \mid x, s')$.
3. **Check each of the eight records.** With $p(\text{smoker} \mid \text{young}) = 0.2222$ and $p(\text{smoker} \mid \text{old}) = 0.3913$:

| True $s$ | $x$, $y$ | Mass | True score | Other score | Right? |
| --- | --- | --- | --- | --- | --- |
| non-smoker | young, no disease | 0.378 | 0.7778 | $0.2222 \times 0.6 = 0.1333$ | yes |
| non-smoker | young, disease | 0.042 | 0.7778 | $0.2222 \times 0.4 = 0.0889$ | yes |
| non-smoker | old, no disease | 0.196 | 0.6087 | $0.3913 \times 0.3 = 0.1174$ | yes |
| non-smoker | old, disease | 0.084 | 0.6087 | $0.3913 \times 0.7 = 0.2739$ | yes |
| smoker | young, no disease | 0.072 | 0.2222 | $0.7778 \times 0.9 = 0.7000$ | no |
| smoker | young, disease | 0.048 | 0.2222 | $0.7778 \times 0.1 = 0.0778$ | yes |
| smoker | old, no disease | 0.054 | 0.3913 | $0.6087 \times 0.7 = 0.4261$ | no |
| smoker | old, disease | 0.126 | 0.3913 | $0.6087 \times 0.3 = 0.1826$ | yes |

: MAP inversion against a model that memorizes its training records, for a member of each kind. {#tbl-sol-attribute-inference-lookup}

4. **Success on members.** Everything except the two "no" rows: $1 - 0.072 - 0.054 = 0.874$.
5. **Advantage.** $0.874 - 0.748 = 0.126$. Members are guessed 0.126 above the non-member ceiling, which no population-level knowledge could achieve.

**Which members it misses.** Smokers without the disease. For them the population evidence for "non-smoker", $p(\text{non-smoker} \mid x)$ times the high probability that a non-smoker has no disease, outweighs the memorized confidence of 1 times the low prior of smoking. Even perfect memorization does not overcome a strong enough prior.

*Memorization turns the model's confidence at the true value into a near-certain signal; what limits the attack is then only how improbable the true value was to begin with.*
:::
