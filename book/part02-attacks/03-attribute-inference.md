# Attribute inference and model inversion

## Why this matters

A hospital trains a model that predicts whether a patient will develop a lung disease from their age band and whether they smoke, and puts it behind an interface that anyone can query. An insurer knows your age and, from a claim, that you have the disease. Can it use the model to find out whether you smoke?

It can make a better guess than it could without the model. But that is not yet a privacy failure. If smokers get the disease more often, the insurer learns something about every person with the disease, whether or not their record was used for training. A published study saying "smoking causes this disease" would let it make the same guess. The question that matters for *you* is different: **does the model reveal more about your attribute because your record was in its training data?**

That question needs a baseline, and this chapter builds it and measures attacks against it. The attacks have a name. **Attribute inference** asks an adversary to recover one sensitive field of a record, such as smoking status, a genotype or an income band, from the other fields of the record and access to a model. **Model inversion** is the classic way to do it: you query the model once for each candidate value of the sensitive field, and keep the value whose prediction best explains the label you already know. In the insurer's case, you ask the model "how likely is the disease for a smoker of this age?" and "for a non-smoker of this age?", and lean towards whichever answer makes the known diagnosis less surprising.

### The one idea: compare with the guesser who never sees the model

Imagine a second insurer who has no access to the model but knows the population perfectly: how common smoking is, how it relates to age and how it relates to the disease. For each combination of the fields it can see, it picks the value of the sensitive field that is most probable in the population. That guess is called the maximum a posteriori (MAP) guess, "a posteriori" because it is made after seeing the other fields, and its success rate is the **imputation baseline**. No model is involved in computing it. ([Information theory](../part01-prerequisites/05-information-theory.md) calls this success rate the posterior vulnerability.)

The chapter rests on one claim about this baseline. **For a record that was not in the training set, no attack can beat the imputation baseline. For a record that was, an attack can.** The reason is that a model trained without your record was built from other people's data only, so it cannot carry any information about you specifically; at most it teaches the adversary about the population, and the baseline already assumes the population is known perfectly. So everything an attack achieves up to the baseline is population inference. What it achieves on training records beyond what it achieves on fresh ones is the leak about individuals. Measuring that leak means comparing people who were in the training set with people who were not, so attribute inference turns out to be a membership question in disguise, which ties this chapter to [Membership inference](02-membership-inference.md).

@tbl-attr-uses lists the results and where each is used.

| Result | Where it is used |
| --- | --- |
| The attribute inference game and its advantage | Measuring an attack as a difference between training and fresh records ([Threat models](01-threat-models.md)) |
| Attribute advantage is a membership advantage | Turning any attribute attack into a membership test; bounding it by total variation |
| The non-member ceiling: imputation is the best possible on fresh records | Separating population inference from individual leakage |
| MAP model inversion; it equals imputation for a calibrated model | Reading an inversion attack as Bayes' rule with the model as likelihood |
| Overfitting lets members beat the baseline | Why the leak shrinks with the training-set size |
| Masking the confidences cannot help the best attacker | What rounding or label-only outputs do, and what they don't |

: The results of this chapter and where each is used. {#tbl-attr-uses}

**What you need.** Everything this chapter borrows is restated where it is used, so the links below are pointers for more detail. The MAP guess, vulnerability and min-entropy leakage are in [Information theory](../part01-prerequisites/05-information-theory.md). Bayes' rule, true and false positive rates, and the fact that the best advantage of a test is the total variation distance are in [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md). Games and the vocabulary of adversaries are in [Threat models](01-threat-models.md), and the loss-threshold attack and the generalization gap are in [Membership inference](02-membership-inference.md).

**Notation.** A record is $z = (s, x, y)$: a sensitive attribute $s$ that takes a few values (smoker or not), public features $x$ (the age band), and a label $y$ (the disease). Records come from a population distribution $\mathcal D$, and $p(s, x, y)$ is the probability of a record (or its density, when $x$ is continuous). A training set $S \sim \mathcal D^n$ is $n$ records drawn independently from $\mathcal D$, and a training algorithm turns it into a model $M$. The model's **confidence** in label $y$ at input $(x, s)$ is $f_y(x, s)$, the probability the model assigns to $y$.

The chapter keeps returning to one number: how often the best possible guess of $s$ is right when the guesser sees $x$ and $y$ and knows the population. For each pair $(x, y)$, the best guess is the value of $s$ with the largest probability $p(s, x, y)$, and it is right on exactly that much probability. Adding over all pairs gives the success rate, written

$$V(S \mid X, Y) = \sum_{x, y}\max_s p(s, x, y).$$

For continuous $x$, read it as $\mathbb E\big[\max_s p(s \mid X, Y)\big]$: the average, over records, of the probability of the most likely value of $s$ given that record's $x$ and $y$. The capital letters $S, X, Y$ stand for the random fields; the letter $S$ does double duty as the training set, and the context always says which is meant.

## Toolkit: three facts every derivation uses

### Fact 1 — The MAP guess is the best guess

Suppose you must guess $s$ after seeing some observation $o$. For each value of $o$, whatever you answer is right exactly when the true $s$ is your answer, so the best you can do is answer the value of $s$ that is most probable together with that $o$. That answer is the **MAP guess**, $\arg\max_s p(s \mid o)$, and its success probability is the sum of the largest probabilities, one per observation:

$$V(S \mid O) = \sum_o\max_s p(s, o) = \mathbb E_O\Big[\max_s p(s \mid O)\Big].$$

The second form is the same sum written as an average: $p(s, o) = p(o)\,p(s \mid o)$, and $p(o)$ does not depend on $s$, so it comes out of the maximum. $V$ stands for **vulnerability**, the name [Information theory](../part01-prerequisites/05-information-theory.md) gives this success rate.

The proof is the counting argument made precise. A deterministic guessing rule $g$ maps each observation $o$ to a guess $g(o)$. It is right when $o$ occurs and $s = g(o)$, which has probability $p(g(o), o)$, so its success is $\sum_o p(g(o), o)$, one term per value of $o$. Each term is at most $\max_s p(s, o)$, and equals it when $g(o)$ is the MAP guess. A randomized rule picks among deterministic rules at random, so its success is an average of their successes, and an average is never above the best one.

For a concrete case, take no observation at all and a prior of 0.3 that a person smokes. The rule "always say non-smoker" is right 70% of the time and "always say smoker" 30%, so the best guess is "non-smoker", with $V(S) = 0.7$.

### Fact 2 — An independent observation is useless

If an extra observation $M$ has nothing to do with $s$ and $o$, seeing it cannot change which value of $s$ is most likely, so it cannot raise the best success. In symbols: if $M$ is independent of $(S, O)$, then $V(S \mid O, M) = V(S \mid O)$.

Independence means the probabilities multiply, $p(s, o, m) = p(s, o)\,p(m)$. Put that into Fact 1 with the observation $(o, m)$:
$$V(S \mid O, M) = \sum_{o, m}\max_s p(s, o)\,p(m) = \sum_m p(m)\sum_o\max_s p(s, o) = V(S \mid O).$$
The middle step pulls $p(m)$ out of the maximum, which is allowed because $p(m) \ge 0$ does not depend on $s$; the last step uses that the $p(m)$ add up to 1. For instance, a coin flip you see before guessing does not change which smoking status is more likely, so it cannot change your success. This fact is what gives the non-member ceiling: a model trained without you is, as far as your record goes, such an independent observation.

### Fact 3 — Processing an observation cannot help the best guesser

If you only see a transformed version $g(O)$ of the observation, you have at most the information in $O$ itself, so you cannot do better: for any function $g$, $V(S \mid g(O)) \le V(S \mid O)$.

The reason is that whoever sees $o$ can compute $g(o)$ and then guess. A rule $h$ that guesses from $g(o)$ is therefore also a rule on $o$, namely "apply $g$, then $h$", written $h \circ g$. So every rule on $g(O)$ is among the rules on $O$, and by Fact 1 none of those beats $V(S \mid O)$. For example, rounding a confidence of 0.62 to 0.6 can only merge observations that a guesser could have told apart; it never separates two that were the same. This is the guessing version of the data-processing inequality of [Divergences](../part01-prerequisites/03-divergences.md), and the chapter uses it for confidence masking.

## The attribute inference game {#sec-attr-game}

### Context

"The attack recovered 78% of smoking statuses" sounds alarming, but it means nothing until you know two other numbers: how often the attribute could be guessed without the model, and how often the attack succeeds on people who were not in the training set. Yeom et al. (2018) made this precise with a game in the style of [Threat models](01-threat-models.md): a challenger sets up a hidden situation, the adversary gets a fixed view of it, and its guess is scored. Here the challenger flips a coin to decide whether the target record comes from the training set or fresh from the population. The coin is not a model of anything an attacker does; it is just a way to run both situations, member and non-member, under one set of rules, so that the attack's success in one can be compared with its success in the other.

### Statement

**The game** (Yeom et al. 2018) is played for a training algorithm, a population $\mathcal D$, a training-set size $n$ and an adversary $\mathcal A$. It has three steps.

1. The challenger draws a training set $S \sim \mathcal D^n$ and trains a model $M$ on it.
2. It draws a hidden bit $b \in \{0, 1\}$, each value with probability $\frac12$. If $b = 1$, it picks the target $z = (s, x, y)$ uniformly from $S$, so the target is a **member** of the training set. If $b = 0$, it draws $z \sim \mathcal D$ afresh, so the target is a **non-member**.
3. The adversary receives the target's public fields and label $(x, y)$, but not $s$, and access to $M$. It may also know some or all of $\mathcal D$: that is its auxiliary knowledge. It outputs a guess $\hat s$ of the sensitive attribute.

The score should measure how much better the attack does on people who were in the training data than on people who were not, because that difference is what the model knows about its training records beyond the population. So the **attribute advantage** is the success rate on members minus the success rate on non-members,
$$\mathrm{Adv}^{\mathrm{attr}}(\mathcal A) = \Pr[\hat s = s \mid b = 1] - \Pr[\hat s = s \mid b = 0].$$ {#eq-attr-advantage}
Here $\Pr[\hat s = s \mid b = 1]$ is the probability that the guess equals the target's true attribute, over everything random in the game, given that the target is a member.

**Attribute advantage is membership advantage.** Any attribute attack can be turned into a membership attack, by a simple observation: if the attribute attack is right more often on members, then "the attribute attack was right" is itself evidence of membership. Build the membership adversary $\mathcal B$ that receives the whole record $z = (s, x, y)$ and $M$, runs $\mathcal A$ on $(x, y)$ and $M$, and says "member" exactly when $\hat s = s$. A membership attack is described by two rates: its true positive rate (TPR), the probability that it says "member" for a member, and its false positive rate (FPR), the probability that it says "member" for a non-member. Its advantage is their difference. For $\mathcal B$ these are
$$\mathrm{TPR}_{\mathcal B} = \Pr[\hat s = s \mid b = 1], \qquad \mathrm{FPR}_{\mathcal B} = \Pr[\hat s = s \mid b = 0], \qquad \mathrm{TPR}_{\mathcal B} - \mathrm{FPR}_{\mathcal B} = \mathrm{Adv}^{\mathrm{attr}}(\mathcal A).$$

Membership attacks have a ceiling: no attack can tell two distributions apart better than their **total variation distance**, which is half the total absolute difference between their probabilities, $\mathrm{TV}(P, Q) = \frac12\sum_v|p(v) - q(v)|$. Because $\mathcal B$ is a membership attack, its ceiling is also a ceiling on every attribute attack:
$$\mathrm{Adv}^{\mathrm{attr}}(\mathcal A) \le \mathrm{TV}\big(P_{\mathrm{in}}, P_{\mathrm{out}}\big),$$ {#eq-attr-tv}
where $P_{\mathrm{in}}$ and $P_{\mathrm{out}}$ are the distributions of the pair $(z, M)$, the target record together with the model, when $z$ is a member and when it is not.

### Derivation

1. **The membership test fires when the attribute guess is right.** $\mathcal B$ is given $s$, so it can check whether $\hat s = s$, and it says "member" on exactly that event.
2. **Read off its rates.** Its TPR is the probability that it says "member" when $b = 1$, and it says "member" exactly when the guess is right, so the TPR is $\Pr[\hat s = s \mid b = 1]$. In the same way its FPR, the probability of "member" when $b = 0$, is $\Pr[\hat s = s \mid b = 0]$.
3. **Subtract.** $\mathrm{TPR} - \mathrm{FPR}$ is the right-hand side of @eq-attr-advantage term by term.
4. **Bound by total variation.** Why can no test beat the total variation distance? A test that says "member" on a set $A$ of views $v = (z, M)$ has $\mathrm{TPR} = P_{\mathrm{in}}(A)$ and $\mathrm{FPR} = P_{\mathrm{out}}(A)$, so its advantage is $\sum_{v \in A}\big(p_{\mathrm{in}}(v) - p_{\mathrm{out}}(v)\big)$. That sum is largest when $A$ contains exactly the views with a positive term, those more likely for members, and its largest value is the total excess of $p_{\mathrm{in}}$ over $p_{\mathrm{out}}$. Since both distributions add up to 1, the total excess one way equals the total excess the other way, so it is half the total absolute difference, $\mathrm{TV}(P_{\mathrm{in}}, P_{\mathrm{out}})$. A randomized test averages such sets and does no better. ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) proves this in general.) Applied to $\mathcal B$, this gives @eq-attr-tv.

### Interpretation

The subtraction in @eq-attr-advantage is what makes the advantage a measure of privacy. Any success that comes from knowing the population, for example "people with the disease tend to smoke", is the same for members and non-members, so it appears in both terms and cancels. What is left is what the model knows about its own training records.

The reduction also says which leak implies which. If a model reveals nothing about membership, in the sense that $P_{\mathrm{in}}$ and $P_{\mathrm{out}}$ are equal and their total variation is 0, then @eq-attr-tv forces the attribute advantage of every adversary to 0. The converse fails. A membership leak does not have to come with an attribute leak: if $s$ is irrelevant to the label, a model can memorize who was in its training set without its predictions depending on $s$ at all, and then no attribute attack gains anything on members.

Because the advantage is a difference, it is not enough to report it alone. Report the success on members and on non-members, as a membership attack reports its TPR and FPR. The advantage by itself hides how much of the success is population inference, which is the part common to both.

Finally, the membership test $\mathcal B$ that the reduction produces is a weak one. It outputs a single bit, so when it says "member" the only evidence it carries is how much more often that happens for members than for non-members, the ratio $\mathrm{TPR}/\mathrm{FPR}$. An attack that is right on most people of the population, members or not, has an FPR close to its TPR and a ratio close to 1, so at a low FPR it is much weaker than the dedicated attacks of [Membership inference](02-membership-inference.md) ([Exercise 3](solutions/03-attribute-inference.md#sec-sol-attribute-inference-3)).

### Example: one model, 200 times

The toy experiment of this chapter (@sec-attr-code) has a binary sensitive attribute with prior $\Pr[s = 1] = 0.3$, ten public features and a binary label. A logistic regression with one weight vector per value of $s$ is trained on $n = 100$ records, and this is repeated 200 times. Each time, the MAP inversion attack of @sec-attr-inversion guesses $s$ for the 100 members and for 100 fresh records. It is right on 0.7832 of members and 0.7262 of non-members, and the attribute advantage is the difference (computed before rounding the two rates),
$$\mathrm{Adv}^{\mathrm{attr}} = 0.7832 - 0.7262 = 0.0569,$$
with a standard error of 0.0045 over the 200 models. Read as a membership test, the same attack has TPR 0.7832 at FPR 0.7262. Its likelihood ratio when it fires is $0.7832/0.7262 = 1.08$: being right is barely more likely for a member than for a non-member. Its balanced accuracy, the average of its accuracy on members (the TPR) and on non-members ($1 - \mathrm{FPR}$), is $\frac12\big(\mathrm{TPR} + 1 - \mathrm{FPR}\big) = \frac12 + \frac12 \times 0.0569 = 0.5285$, hardly better than the $\frac12$ of a coin flip. Most of its 78% success would be there even if the person had never been in the training data.

### Check yourself

- An attack guesses the right attribute for 90% of members and 90% of non-members. What is its attribute advantage, and does it leak anything about individuals? *(Zero, since $0.9 - 0.9 = 0$. It is doing population inference: whatever it learned applies equally to people outside the training set.)*
- Why does the membership adversary $\mathcal B$ need the sensitive attribute $s$ of the target, while $\mathcal A$ does not get it? *(The membership game hands the adversary the whole record and asks only whether it was used. $\mathcal B$ uses $s$ to check whether $\mathcal A$'s guess was right.)*

## The imputation baseline {#sec-attr-imputation}

### Context

The game removes population inference by subtracting it. It is just as useful to measure it directly, because it tells you how much of an attack's success any model trained on the same population would give, or any published statistics about it. The measure is the success of the best guess of $s$ from $(x, y)$ when the population law is known. Statisticians call filling in a missing field from the others **imputation**, so this is the imputation baseline. Jayaraman & Evans (2022) found that many published attribute inference attacks do no better than imputation from similar data, and argued that this baseline is the one an attack must beat.

### Statement

On a non-member, the model was trained without the target, so it is an independent observation (Fact 2) and the best any adversary can do is the best guess from $(x, y)$ alone (Fact 1). That gives the **non-member ceiling**: for every adversary in the game,
$$\Pr[\hat s = s \mid b = 0] \le V(S \mid X, Y) = \sum_{x, y}\max_s p(s, x, y).$$ {#eq-attr-ceiling}

The second result is about attacks that never look at the model. A training record, picked at random from a training set drawn from the population, looks exactly like a fresh draw if you ignore the model, so such attacks cannot tell the two worlds apart. **Model-free attacks have zero advantage:** if $\hat s$ depends on $(x, y)$ only, not on $M$, then $\Pr[\hat s = s \mid b = 1] = \Pr[\hat s = s \mid b = 0]$, and the Bayes imputation guess $\arg\max_s p(s \mid x, y)$, the MAP guess under the true population, attains the bound @eq-attr-ceiling in both worlds.

### Derivation

**The ceiling.**

1. **A non-member is independent of the model.** When $b = 0$, $z$ is drawn afresh, separately from $S$. The model depends only on $S$ and the randomness of training, so $M$ is independent of $z = (s, x, y)$.
2. **So the model is a useless observation for the guess.** Fact 2 says that adding an observation independent of $(s, x, y)$ does not change the best success: $V(S \mid X, Y, M) = V(S \mid X, Y)$.
3. **No guessing rule beats the MAP guess.** The adversary's rule is a function of $(x, y, M)$, possibly randomized. By Fact 1 its success is at most the best success from that observation, $V(S \mid X, Y, M)$, which step 2 showed is $V(S \mid X, Y)$.

**Model-free attacks.**

1. **A member is a population record.** $S$ consists of $n$ independent draws from $\mathcal D$, and the target is one of them chosen uniformly at random. Whichever one is chosen, it was drawn from $\mathcal D$, so on its own (ignoring the model) the target has distribution $\mathcal D$, the same as a fresh draw.
2. **A rule that ignores $M$ sees the same distribution in both worlds.** Its success is $\Pr[g(X, Y) = S]$ for a record drawn from $\mathcal D$ either way, so the two terms of @eq-attr-advantage are equal and the advantage is 0.
3. **Imputation attains the ceiling.** With $g$ the MAP guess under $\mathcal D$, Fact 1 says this common success is $V(S \mid X, Y)$.

### Interpretation

Below the ceiling, the model only teaches you the population. An adversary who does not know $\mathcal D$ can use the model to learn it, and so raise its success on non-members towards $V(S \mid X, Y)$, but it cannot go further. For a person outside the training data, the model is no more dangerous than a perfect description of the population.

This is the point of the "smoking causes cancer" example in Dwork & Roth (2014, §1). A study shows that smoking causes cancer. An insurer who reads it can now infer more about any smoker's health, whether or not that smoker took part in the study. The study disclosed a fact about the world, not about its participants, and a privacy definition that forbade this would forbid learning anything from data. The privacy question is the gap between members and non-members, which the ceiling isolates.

That does not make population inference harmless. Being guessed to be a smoker hurts whether or not the guess used your record. But no mechanism that limits what one record contributes can prevent it, since the harm would be there with your record removed. It is a question for policy and fairness, not for the attacks of this part.

Note also that the ceiling is a property of the population, not of the model, and it can be high. If $x$ and $y$ nearly determine $s$, every model and every statistic about the population "reveals" $s$. That is why attacks should be reported relative to it.

### Example: the smoking table

Let $s = 1$ mean "smoker", $x$ the age band (young or old) and $y = 1$ "has the disease". The population has $\Pr[s = 1] = 0.3$. Among non-smokers a fraction $\Pr[\text{old} \mid s] = 0.4$ is old, and among smokers $0.6$. The disease probabilities $\Pr[y = 1 \mid x, s]$ are 0.1 (young non-smoker), 0.3 (old non-smoker), 0.4 (young smoker) and 0.7 (old smoker). The probability of a full record is the chance of the smoking status, times the chance of the age band given it, times the chance of the disease outcome given both; multiplying gives the joint table, @tbl-attr-smoking. For example, $p(\text{smoker}, \text{old}, \text{disease}) = 0.3 \times 0.6 \times 0.7 = 0.126$, and $p(\text{non-smoker}, \text{young}, \text{no disease}) = 0.7 \times 0.6 \times 0.9 = 0.378$.

| $p(s, x, y)$ | young, no disease | young, disease | old, no disease | old, disease |
| --- | --- | --- | --- | --- |
| non-smoker ($s = 0$) | **0.378** | 0.042 | **0.196** | 0.084 |
| smoker ($s = 1$) | 0.072 | **0.048** | 0.054 | **0.126** |
| $\Pr[\text{smoker} \mid x, y]$ | 0.1600 | 0.5333 | 0.2160 | 0.6000 |

: The smoking table: the joint distribution of smoking status $s$, age band $x$ and disease $y$, with the column maxima in bold and the posterior probability of smoking in each column. {#tbl-attr-smoking}

The last row is each column's smoker entry divided by the column total; for young people with the disease, $0.048/(0.042 + 0.048) = 0.5333$. Now watch the best guess improve as the guesser sees more.

With no observation, the best guess is "non-smoker", right with probability $V(S) = 0.7$, as in Fact 1.

With age alone, the guesser adds the two columns of each age band. Young smokers have probability $0.3 \times 0.4 = 0.12$ and young non-smokers $0.7 \times 0.6 = 0.42$, so a young person is a smoker with probability $0.12/0.54 = 0.2222$. Old smokers have $0.3 \times 0.6 = 0.18$ and old non-smokers $0.7 \times 0.4 = 0.28$, so an old person is a smoker with probability $0.18/0.46 = 0.3913$. Both are below $\frac12$, so the guess stays "non-smoker" in both age bands and $V(S \mid X) = 0.7$. Age changes the odds but never the decision, so it adds nothing to the success.

With age and disease, the posterior crosses $\frac12$ in the two disease columns (@fig-attr-smoking), so the MAP guess becomes "smoker" there. By Fact 1 the success is the sum of the largest entry in each column, the bold ones: $V(S \mid X, Y) = 0.378 + 0.048 + 0.196 + 0.126 = 0.748$.

The **min-entropy leakage** of [Information theory](../part01-prerequisites/05-information-theory.md) turns this gain into bits: it is the base-2 logarithm of how many times more likely the best guess is to succeed with the observation than without. Here the disease label, given age, raises the success from 0.7 to 0.748, a leakage of $\log_2(0.748/0.7) = 0.0957$ bits.

Every one of these numbers comes from the population. A model trained on 10,000 patients from it, if it predicts well, lets an insurer reach 0.748 on *anyone* drawn from the population, and the ceiling @eq-attr-ceiling says nobody outside the training set can be attacked more successfully than that.

![The posterior probability of smoking in each cell of the smoking table, with each cell's probability in brackets. Red bars cross $\frac12$, where imputation guesses "smoker"; together they add 0.048 to the 0.7 of the blind guess.](figures/attr_smoking_posterior.svg){#fig-attr-smoking}

### Check yourself

- In the smoking table, which cell would the imputation guess get wrong most often, measured by probability mass? *(Young, no disease: the guess there is "non-smoker", which is wrong on the 0.072 of smokers in that cell, the largest non-bold entry.)*
- Does the ceiling @eq-attr-ceiling hold for an adversary that sees the whole training set $S$, not only the model? *(Yes. The argument only needs the observation to be independent of the fresh target, and $S$ is.)*

## Model inversion: the MAP attack {#sec-attr-inversion}

### Context

The imputation guess needs $p(y \mid x, s)$, the link between the label and the sensitive attribute, and the adversary usually does not know it. But a model that predicts $y$ from $x$ and $s$ is an estimate of exactly that link. Fredrikson et al. (2014) used this in a study of warfarin dosing: a linear model predicted the stable dose of the drug from demographics and two genetic markers, and an adversary who knew a patient's demographics and dose could infer their genotype. They called the attack **model inversion**: run the model "backwards" from its output to an input. Fredrikson, Jha & Ristenpart (2015) extended it to models that release **confidences**, the predicted probability of each label, and showed that confidences make the attack much stronger.

### Statement

The idea is Bayes' rule. The value of $s$ that best explains what you see is the one with the largest product of "how plausible is $s$ for someone with these features" and "how likely is the observed label if the attribute were $s$". The adversary supplies the first factor from what it knows about the population, and gets the second by querying the model with each candidate value of $s$. The **MAP inversion attack** guesses
$$\hat s_{\mathrm{inv}}(x, y) = \arg\max_s\; p(s \mid x)\, f_y(x, s).$$ {#eq-attr-map-inversion}
Here $p(s \mid x)$ is what the adversary knows about how $s$ relates to the public features, and $f_y(x, s)$ is the model's confidence in the observed label $y$ when it is fed the candidate $s$. Fredrikson et al. used the marginal $p(s)$ in place of $p(s \mid x)$, which treats $s$ as independent of $x$.

If the model's confidences were exactly the population's probabilities, the second factor would be the true likelihood, and the attack would be the imputation guess. A model with that property is called **calibrated**. So, **for a calibrated model, inversion is imputation**: if $f_y(x, s) = p(y \mid x, s)$ for all $s, x, y$, then $\hat s_{\mathrm{inv}}$ is the Bayes imputation guess, and its success is $V(S \mid X, Y)$ on members and non-members alike.

### Derivation

1. **Bayes' rule.** The joint probability of $s$ and $y$ given $x$ can be built in two stages, first $s$ from $x$ and then $y$ from both: $p(s, y \mid x) = p(s \mid x)\,p(y \mid x, s)$. Dividing by $p(y \mid x)$ turns it into the posterior of $s$:
$$p(s \mid x, y) = \frac{p(s \mid x)\,p(y \mid x, s)}{p(y \mid x)}.$$
2. **The denominator does not depend on $s$.** It is the same number for every candidate, so it cannot change which candidate is largest: $\arg\max_s p(s \mid x, y) = \arg\max_s p(s \mid x)\,p(y \mid x, s)$.
3. **Plug in the model.** Replace the unknown $p(y \mid x, s)$ by the model's estimate $f_y(x, s)$. This is @eq-attr-map-inversion.
4. **Calibrated case.** If $f_y = p(y \mid x, s)$ exactly, step 3 changes nothing and the guess is the MAP guess of Fact 1, with success $V(S \mid X, Y)$. The guess then uses nothing about $M$ beyond the population's probabilities, so it is a model-free attack, and the argument of @sec-attr-imputation gives equal success on members and non-members.

### Interpretation

The attack is Bayes' rule with the model as the likelihood, so everything it does well or badly comes from how the model's $f_y(x, s)$ differs from the population's $p(y \mid x, s)$. Two differences matter, one on members and one on non-members.

On members, the model is too sure of the truth. Training lowers the loss $-\ln f_y(x, s)$ on each training record at its *true* $s$, which pushes $f_y(x, s)$ up towards 1. The other candidate value $s'$ was never paired with this record during training, so $f_y(x, s')$ stays closer to an honest prediction. (In the toy model below this holds exactly: each value of $s$ has its own weight vector, so the record never touched the weights used at $s'$.) The ratio $f_y(x, s)/f_y(x, s')$ that decides the attack is therefore tilted towards the truth for members. This is the generalization gap of [Membership inference](02-membership-inference.md), the difference between a model's loss on its training records and on fresh ones, seen through the sensitive attribute. In the toy experiment the mean log-loss is 0.2965 on members and 0.6690 on non-members.

On non-members, an overfit model does worse than imputation. Its confidences carry the noise of a small training set rather than the population's exact probabilities, so the attack's guesses fall below the ceiling.

More data closes both gaps. As $n$ grows, the model approaches $p(y \mid x, s)$ and both success rates approach $V(S \mid X, Y)$ (@fig-attr-success-vs-n). The population inference gets better and the individual leak gets smaller.

The same idea can invert the whole input rather than one field. Fredrikson, Jha & Ristenpart (2015) also attacked a face recognition model in which each class is one person. They searched for the input image that maximizes the confidence of a chosen class, by gradient descent on the image, and obtained a recognizable, blurred image of that person. It is the same idea with $s$ replaced by the whole input, and the prior $p(s \mid x)$ replaced by an image regularizer that keeps the search among plausible images. What it recovers is a typical input of the class, which is population inference about the class; it is a privacy problem because the class is a single person.

### Example: walking through the inversion

**On the smoking table, with a calibrated model.** Take an old patient with the disease. From the age-alone step of the smoking-table example, the adversary knows $\Pr[\text{smoker} \mid \text{old}] = 0.3913$, so $\Pr[\text{non-smoker} \mid \text{old}] = 1 - 0.3913 = 0.6087$. It queries the model at both values of $s$, and a calibrated model returns the disease probabilities of the population: $f_1(\text{old}, \text{smoker}) = 0.7$ and $f_1(\text{old}, \text{non-smoker}) = 0.3$. The scores of @eq-attr-map-inversion are $0.3913 \times 0.7 = 0.2739$ for "smoker" and $0.6087 \times 0.3 = 0.1826$ for "non-smoker", so it guesses "smoker". Dividing the smoker score by the sum of the two, which is the normalization that step 1 of the derivation leaves out, gives $0.2739/(0.2739 + 0.1826) = 0.6000$, the posterior of @tbl-attr-smoking. The attack is imputation.

**On the toy model.** The population draws $s$ with $\Pr[s = 1] = 0.3$ and ten standard normal features, the first shifted by 0.5 when $s = 1$. The label follows a logistic model whose weights depend on $s$. Guessing from $x$ alone is right 0.7045 of the time, and Bayes imputation, which knows the true logistic weights, is right 0.7517 of the time (both on a million fresh records). Over 200 models trained on $n = 100$ records, imputation is right on 0.7508 of members and 0.7555 of non-members. The difference, $-0.0046$, is within one standard error of zero, as the model-free argument predicts. MAP inversion, which uses the trained model's confidences, is right on 0.7832 of members, $0.7832 - 0.7517 = 0.0315$ *above* the imputation ceiling, and on 0.7262 of non-members, 0.0254 below it (computed before rounding).

The member success above the ceiling is something no population-level knowledge could produce, since the ceiling is the best that perfect population knowledge achieves. @fig-attr-evidence shows where it comes from. It plots the attack's evidence for the true value $s$ over the other value $s'$, the log of the ratio of their two scores, $\ln\frac{p(s \mid x)f_y(x, s)}{p(s' \mid x)f_y(x, s')}$. The attack is right when this is positive. For members the distribution has a thinner left tail: training has removed most of the cases where the model strongly prefers the wrong value. @fig-attr-success-vs-n repeats the game for training sets from 25 to 1600 records. At $n = 25$ the attack is right on 0.8420 of members and 0.6810 of non-members, an advantage of $0.8420 - 0.6810 = 0.1610$. At $n = 400$ the advantage is 0.0111, and at $n = 1600$ it is 0.0052.

![The evidence of MAP inversion for the true attribute, the log of the score of the true value over the other, for members and non-members of 50 models trained on 100 records each. The attack is right when the evidence is positive; members have far fewer large negative values.](figures/attr_inversion_evidence.svg){#fig-attr-evidence}

![Success of MAP inversion on members and non-members against the training-set size $n$, 40 models per point. Members start above the Bayes imputation ceiling and non-members below it; both converge to the ceiling as the model approaches the population.](figures/attr_success_vs_n.svg){#fig-attr-success-vs-n}

### Check yourself

- If the model ignored $s$ entirely, $f_y(x, s) = f_y(x, s')$, what would MAP inversion guess? *(The MAP guess from $x$ alone, $\arg\max_s p(s \mid x)$: the confidence is the same factor for every candidate in @eq-attr-map-inversion, so it cannot change which candidate wins.)*
- Why does using the marginal $p(s)$ instead of $p(s \mid x)$ in @eq-attr-map-inversion lose success even with a calibrated model? *(It drops the evidence that $x$ carries about $s$, so the score is no longer proportional to $p(s \mid x, y)$. [Exercise 2](solutions/03-attribute-inference.md#sec-sol-attribute-inference-2) computes the loss on the smoking table.)*

## Confidence masking {#sec-attr-masking}

### Context

If confidences make inversion stronger, why release them? Fredrikson, Jha & Ristenpart (2015) suggested rounding the confidences an interface returns, and many deployed interfaces return only the top label. Such **confidence masking** is cheap and leaves the predicted label unchanged. Does it close the leak?

### Statement

Masking only coarsens what the adversary sees, so by Fact 3 it can never help the best attacker. To write this down, let $F = \big(f(x, s)\big)_s$ be the list of answers the adversary gets by querying the model at every candidate value of $s$, where each answer $f(x, s)$ is the model's vector of confidences over the labels. Let $g$ be any masking function applied to these answers: rounding, keeping only the top label, adding noise. Then the best attack on the masked answers does no better than the best attack on the full ones, on members and on non-members alike:
$$V\big(S \mid X, Y, g(F)\big) \le V\big(S \mid X, Y, F\big).$$ {#eq-attr-masking}
The inequality is about the *best* attacker in each world separately. It says nothing about the *difference* between members and non-members, and masking need not shrink it.

### Derivation

1. **A masked attack is an unmasked one.** Anyone holding the full answers can apply the mask themselves. So a rule $h$ that guesses from $(x, y, g(F))$ is also the rule $(x, y, F) \mapsto h(x, y, g(F))$ on the unmasked answers.
2. **The best unmasked rule is at least as good.** This is Fact 3, with the observation $O = (X, Y, F)$ and the processing function $(x, y, F) \mapsto (x, y, g(F))$. It gives @eq-attr-masking, separately for $b = 0$ and for $b = 1$, since the argument never used which world the target came from.
3. **Why the difference can survive.** Write $V_{\mathrm{in}}$ and $V_{\mathrm{out}}$ for the best success on members and on non-members. Masking lowers both, by amounts that depend on $g$ and on how the confidences of members and non-members differ. Nothing forces the two drops to be equal, so $V_{\mathrm{in}} - V_{\mathrm{out}}$ can rise or fall.

### Interpretation

Labels alone still leak, for the same reason confidences do. With only the top label, the adversary still sees, for each candidate $s$, whether the predicted label $\mathrm{label}(x, s)$ equals the known label $y$. On a member, the model usually predicts the training label at the true $s$, because it fit that record. At the other value it gives an honest prediction, which matches $y$ less often. So "the label matches only at one candidate" points to that candidate, and on members it usually points to the truth. Non-members rarely get this help: the two predicted labels usually agree, and the attack falls back on the guess from $x$ alone. Label-only membership inference (Choquette-Choo et al. 2021) exploits the same asymmetry.

This is why masking hurts the population inference more than the individual leak. The toy experiment below shows it: non-member success drops almost to the $x$-only guess, while member success stays above the imputation ceiling.

Do not read too much into small changes either way. The inequality is about the optimal attacker, and MAP inversion with an overfit model is not optimal, so a particular masking can even raise its success slightly. That does not contradict @eq-attr-masking.

Above all, masking is obfuscation, not a guarantee. It removes one attack's input without bounding what any attack can learn about a member. Part 3 introduces differential privacy, which bounds the member/non-member difference itself, for every attacker and every output format.

### Example: rounding and labels on the toy model

The same 200 models of @sec-attr-inversion, attacked through three interfaces, give @tbl-attr-masking and @fig-attr-masking.

| Interface | Members | Non-members | Advantage |
| --- | --- | --- | --- |
| Full confidences | 0.7832 | 0.7262 | 0.0569 |
| Confidences rounded to one decimal | 0.7836 | 0.7253 | 0.0583 |
| Top label only | 0.7741 | 0.7092 | 0.0650 |

: MAP inversion success through three interfaces, averaged over 200 models trained on 100 records each. Guessing from $x$ alone gives 0.7045 and Bayes imputation 0.7517. {#tbl-attr-masking}

Rounding to one decimal changes almost nothing. The attack only compares the scores of the two candidates, and most confidences are far enough from the rounding boundaries that the comparison keeps its sign. With labels only, non-member success drops from 0.7262 to 0.7092, close to the 0.7045 of guessing from $x$ alone. Member success drops less, to 0.7741, still $0.7741 - 0.7517 = 0.0224$ above the imputation ceiling. So the advantage *grows*, from 0.0569 to 0.0650. The interface now leaks less about the population and as much as before about the people in the training set.

![MAP inversion success on members and non-members through three interfaces, for models trained on 100 records. The dashed line is Bayes imputation, the dotted line the guess from the public features alone.](figures/attr_masking.svg){#fig-attr-masking}

### Check yourself

- Can adding Gaussian noise to the returned confidences increase the best attacker's success? *(No. Noise is a randomized function of $F$, so a rule on the noisy answers is a randomized rule on $F$: it draws the noise itself and then guesses. Fact 1 shows a randomized rule does no better than the best deterministic one.)*
- Why does the label-only attack on a non-member usually fall back on the guess from $x$ alone? *(For a fresh record, the predicted labels at the two candidate values usually agree. Both then match $y$ or both miss, the masked confidences tie, and the tie is broken by $p(s \mid x)$.)*

## Code: the game on a toy model {#sec-attr-code}

The script [code/attribute_inference.py](code/attribute_inference.py) computes every number in this chapter and draws every figure. It works out the smoking table exactly. For the toy model it samples the population, trains a ridge-regularized logistic regression with one weight vector per value of $s$ by Newton's method, and plays the game: each run trains a model on $n$ records and attacks those $n$ members and $n$ fresh non-members with MAP inversion, Bayes imputation and the guess from $x$ alone. The real attacks of Fredrikson et al. used a warfarin dosing model and face recognition networks; the toy keeps the structure (a sensitive input, a known label, a model queried at each candidate value) and runs in a few seconds. From the repository root:

```bash
uv run python book/part02-attacks/code/attribute_inference.py
```

**Output** (seed 0), summarized in @tbl-attr-code:

| Experiment | Result |
| --- | --- |
| Smoking table | $V(S) = 0.7000$; $V(S \mid X) = 0.7000$; $V(S \mid X, Y) = 0.7480$; leakage 0.0957 bits; posteriors 0.1600, 0.5333, 0.2160, 0.6000 |
| Toy population, $10^6$ fresh records | Guess from $x$ alone 0.7045; Bayes imputation 0.7517 |
| Game, $n = 100$, 200 models: mean log-loss | Members 0.2965; non-members 0.6690; gap 0.3725 |
| Guess from $x$ alone | Members 0.7017; non-members 0.7089 |
| Bayes imputation | Members 0.7508; non-members 0.7555; advantage $-0.0046$ |
| MAP inversion | Members 0.7832; non-members 0.7262; advantage 0.0569 (standard error 0.0045) |
| MAP inversion against the ceiling | Members $+0.0315$; non-members $-0.0254$ |
| As a membership test | TPR 0.7832; FPR 0.7262; balanced accuracy 0.5285 |
| Masking | Rounded: 0.7836 / 0.7253, advantage 0.0583. Label only: 0.7741 / 0.7092, advantage 0.0650 |
| Training-set size, 40 models each | $n = 25$: 0.8420 / 0.6810. $n = 400$: 0.7611 / 0.7500. $n = 1600$: 0.7550 / 0.7497 |

: The script's output with the default seed. Pairs are success on members / non-members. {#tbl-attr-code}

[tests/test_attribute_inference.py](../../tests/test_attribute_inference.py) checks that the output still matches.

**What to take from it.** First, the ceiling holds: non-member success never exceeds Bayes imputation, at any training-set size, as the independence argument of @sec-attr-imputation requires. Second, members beat it. At $n = 100$ the attack is 0.0315 above the best population-level guess on members, and since no knowledge of the population can do better than that guess, only individual leakage can explain the excess. Third, the model-free attacks show no advantage, as they should. Imputation and the $x$-only guess differ between members and non-members by less than 0.01, which is sampling noise: each rate is an average over $200 \times 100 = 20{,}000$ records, and a success rate near 0.75 averaged over that many records has a standard error of about $\sqrt{0.75 \times 0.25/20{,}000} = 0.0031$. Fourth, overfitting drives the leak. The advantage falls from 0.1610 at $n = 25$ to 0.0052 at $n = 1600$, as the gap between member and non-member loss closes. Finally, masking is not a fix: label-only outputs lower success on both members and non-members, and the advantage stays.

## References

- Yeom, Giacomelli, Fredrikson & Jha, "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting" (IEEE CSF 2018). The attribute inference game with members and non-members, its advantage, the reduction between attribute and membership advantage, and the link to the generalization gap.
- Fredrikson, Lantz, Jha, Lin, Page & Ristenpart, "Privacy in Pharmacogenetics: An End-to-End Case Study of Personalized Warfarin Dosing" (USENIX Security 2014). The first model inversion attack: inferring genetic markers from a dosing model and a patient's demographics and dose.
- Fredrikson, Jha & Ristenpart, "Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures" (CCS 2015). Inversion from confidences for decision trees and for face recognition, and rounding as a countermeasure.
- Wu, Fredrikson, Jha & Naughton, "A Methodology for Formalizing Model-Inversion Attacks" (IEEE CSF 2016). Model inversion as a game, with the distinction between what a model reveals about its training data and about the population.
- Jayaraman & Evans, "Are Attribute Inference Attacks Just Imputation?" (CCS 2022). The imputation baseline made central: many attacks do no better, and what it takes to beat it.
- Choquette-Choo, Tramèr, Carlini & Papernot, "Label-Only Membership Inference Attacks" (ICML 2021). Why returning only labels does not stop inference about members.
- Salem, Cherubin, Evans, Köpf, Paverd, Suri, Tople & Zanella-Béguelin, "SoK: Let the Privacy Games Begin! A Unified Treatment of Data Inference Privacy in Machine Learning" (IEEE S&P 2023). Attribute inference among the other privacy games, with their relationships.
- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), §1. The smoking example and why learning about the population is not a privacy breach of the individual.

## Cheat sheet

A record is $(s, x, y)$; $f_y(x, s)$ is the model's confidence in label $y$; $V(S \mid O) = \sum_o\max_s p(s, o)$ is the success of the best guess of $s$ from $O$. @tbl-attr-cheat collects the results.

| Result | Statement | Where it comes from |
| --- | --- | --- |
| Attribute advantage | $\Pr[\hat s = s \mid \text{member}] - \Pr[\hat s = s \mid \text{non-member}]$ | Yeom et al.'s game |
| Reduction to membership | Fire when $\hat s = s$: TPR $-$ FPR $=$ attribute advantage $\le \mathrm{TV}(P_{\mathrm{in}}, P_{\mathrm{out}})$ | Best advantage is total variation |
| Non-member ceiling | $\Pr[\hat s = s \mid \text{non-member}] \le V(S \mid X, Y)$ | Model independent of a fresh record; MAP is optimal |
| Model-free attacks | Equal success on members and non-members; imputation attains $V(S \mid X, Y)$ | A member is marginally a population draw |
| MAP inversion | $\hat s = \arg\max_s p(s \mid x)\,f_y(x, s)$ | Bayes' rule with the model as likelihood |
| Calibrated model | Inversion = imputation, zero advantage | $f_y = p(y \mid x, s)$ |
| Masking | $V(S \mid X, Y, g(F)) \le V(S \mid X, Y, F)$, but the member/non-member gap can grow | Processing cannot help the best guesser |

: Every result in this chapter. {#tbl-attr-cheat}

**Rules of thumb**

- Report success on members and non-members, and the imputation baseline. A single success rate says nothing about privacy.
- Success up to the imputation baseline is population inference; any model or statistic of the population gives it.
- Success on members beyond what non-members get is the individual leak. It comes from overfitting and shrinks as the training set grows.
- An attribute attack is a membership test that outputs one bit. It is weak at low false positive rates.
- Rounding confidences or returning labels only changes the attacks, not the fact that the model has memorized its training records.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](solutions/03-attribute-inference.md).
