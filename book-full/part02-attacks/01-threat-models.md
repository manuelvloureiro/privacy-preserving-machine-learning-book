# Threat models and privacy games

## Why this matters

A hospital trains a model on patient records and offers it to doctors through a web service. Is the model "private"? The question has no answer until you say **against whom**. Can the attacker only send inputs and read predicted labels, or does it hold the weights? Does it know the other patients in the training set, or only what the population looks like? Does it follow the rules of the service, or does it also slip crafted records into the training data? And what does it want: to learn that one person was a patient, to learn their diagnosis, or to rebuild the whole table? A **threat model** answers these questions. Without one, "the model leaks" and "the model is safe" are both untestable claims.

Once the threat model is fixed, an attack can be written down as a **game**, the way cryptographers write down attacks on encryption. A **challenger** runs the training pipeline with some hidden **secret**, such as whether one patient's record was used. It hands the **adversary** what the threat model allows it to see, and the adversary guesses the secret. An attack is good if it guesses better than someone who never saw the model.

That is the one idea of this chapter: **an attack is a guess about a hidden secret, made from a view that the threat model fixes.** When the secret is a single bit, such as "was this patient in the training set?", the game is a hypothesis test, so everything in [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) carries over. The attack's **advantage** is its true positive rate (TPR) minus its false positive rate (FPR). No attack can push it above the total variation (TV) distance between what the adversary sees in the two worlds, and one curve bounds the attack's whole error trade-off.

The chapter builds this up in four steps. It first lists what a threat model has to specify, and proves the one ordering that always holds: more access never makes the adversary weaker. It then writes the attacks of this part as games, each with its own baseline. Next it turns the hypothesis-testing results into ceilings on any attack, and explains why the low false-positive end of the curve is where privacy is really lost. Finally it shows how far apart the average-case and worst-case versions of the same game can be. Every later chapter of this part is an instance of these games: [membership inference](02-membership-inference.md), [attribute inference](03-attribute-inference.md), [dataset properties](04-property-inference.md) and [reconstruction](09-reconstruction.md). @tbl-threat-uses lists the results and where they are used.

| Result | Where it is used |
| --- | --- |
| Threat-model taxonomy: access, behaviour, trust, side information | Stating what an attack assumes, before reading any number it reports |
| More access never lowers the best advantage | Comparing label-only, black-box and white-box attacks |
| Privacy games: membership, attribute, reconstruction, distinguishing | A precise definition of every attack in this part |
| Baselines and the Bayes-optimal guess | Telling an attack apart from a good guess about the population |
| Advantage $\le$ TV; attack's error curve below the trade-off curve | The ceiling every attack runs into; the target defences aim at |
| TPR at a low FPR, precision | How attacks are evaluated in practice |
| Average case $\le$ worst case | Why a defence must hold for every record and every dataset |

: The results of this chapter and where each is used. {#tbl-threat-uses}

**What this chapter assumes.** The hypothesis-testing vocabulary (type I and II errors, the Neyman–Pearson lemma, the trade-off function $T(P, Q)$, receiver operating characteristic (ROC) curves) is in [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md). The total variation distance and the hockey-stick divergence $E_\gamma$ are in [Divergences](../part01-prerequisites/03-divergences.md). Security games and advantage, in their cryptographic form, are in [Cryptography basics](../part01-prerequisites/08-crypto-basics.md). No privacy definition is introduced here: this part only attacks. The definition that defends against these games comes in [Part 3](../part03-dp-foundations/01-definition.md).

**Conventions.** $D$ is a dataset of records drawn from a population, the data distribution $\mathcal D$. A training algorithm $\mathcal T$ maps $D$ to a model $\theta = \mathcal T(D)$, possibly at random. In a two-world game a hidden bit $b$ selects the world: $b = 1$ is the world in which the target's record is used ("member", the alternative hypothesis), and $b = 0$ the world in which it is not (the null). Everything the adversary gets to look at, including what it knew beforehand, is its **view** $V$. $P$ and $Q$ are the distributions of $V$ when $b = 0$ and $b = 1$. An attack outputs a guess $b'$, 1 for "member" and 0 for "not a member". Two numbers describe how it behaves. Its **true positive rate** is the fraction of members it correctly flags, and its **false positive rate** is the fraction of non-members it wrongly flags:

$$\mathrm{TPR} = \Pr[b' = 1 \mid b = 1], \qquad \mathrm{FPR} = \Pr[b' = 1 \mid b = 0].$$

## Toolkit: three facts every derivation uses

### Fact 1 — A game with a hidden bit is a hypothesis test

The challenger flips the hidden bit before anything else happens: $b = 1$ with probability $\pi_1$ and $b = 0$ with probability $\pi_0 = 1 - \pi_1$. The adversary then sees its view and announces $b'$. It wins when its guess matches the hidden bit, whichever value the bit has: $b' = b$ means "$b' = 1$ and $b = 1$, or $b' = 0$ and $b = 0$". So $\Pr[b' = b]$ is the probability that the guess is correct, taken over the challenger's coin flip and everything else that is random in the game. It is not the probability that $b$ is 1, which is just $\pi_1$. The probability of a correct guess is

$$\Pr[b' = b] = \pi_1\,\mathrm{TPR} + \pi_0\,(1 - \mathrm{FPR}).$$ {#eq-threat-win}

*Derivation.* The idea is to split the games by which world the challenger chose, count the wins in each world, and add.

1. **Split the win into the two worlds.** There are exactly two ways to win: the bit is 1 and the attack says 1, or the bit is 0 and the attack says 0. In a single game the bit has one value, either 1 or 0, so a game can win in one of these ways or the other, never both. Events that cannot happen together are called **disjoint**, and the probability that one of two disjoint events happens is the sum of their probabilities. Counting games shows why: if you count the games won with $b = 1$ and the games won with $b = 0$, every won game is counted exactly once, because no game belongs to both groups. (For events that can overlap you would have to subtract the overlap, which here is empty.) So
$$\Pr[b' = b] = \Pr[b = 1 \text{ and } b' = 1] + \Pr[b = 0 \text{ and } b' = 0].$$
2. **Write each joint probability as "prior times conditional".** For any two events, $\Pr[A \text{ and } B] = \Pr[A]\Pr[B \mid A]$. So $\Pr[b = 1 \text{ and } b' = 1] = \Pr[b = 1]\Pr[b' = 1 \mid b = 1] = \pi_1\,\mathrm{TPR}$, by the definition of the TPR.
3. **Do the same in the other world.** $\Pr[b = 0 \text{ and } b' = 0] = \pi_0\Pr[b' = 0 \mid b = 0]$. When the bit is 0 the attack says either 0 or 1, so $\Pr[b' = 0 \mid b = 0] = 1 - \Pr[b' = 1 \mid b = 0] = 1 - \mathrm{FPR}$.
4. **Add the two worlds.** This gives @eq-threat-win.

*A concrete count.* Play 1,000 games with an attack whose TPR is 0.6 and FPR is 0.2. @tbl-threat-win-count counts the wins for a fair bit, $\pi_1 = \frac12$, and for a bit that is 1 only one time in ten, $\pi_1 = 0.1$.

| Prior $\pi_1$ | Games with $b = 1$ | Wins there ($b' = 1$) | Games with $b = 0$ | Wins there ($b' = 0$) | Wins in total |
| --- | --- | --- | --- | --- | --- |
| $\frac12$ | 500 | $0.6 \times 500 = 300$ | 500 | $0.8 \times 500 = 400$ | 700, so $\Pr[b' = b] = 0.7$ |
| 0.1 | 100 | $0.6 \times 100 = 60$ | 900 | $0.8 \times 900 = 720$ | 780, so $\Pr[b' = b] = 0.78$ |

: Expected wins in 1,000 games of an attack with TPR 0.6 and FPR 0.2. The win rate in each world is fixed by the attack; the prior decides how many games are played in each world. {#tbl-threat-win-count}

The formula gives the same numbers: $\frac12(0.6) + \frac12(0.8) = 0.7$ and $0.1(0.6) + 0.9(0.8) = 0.78$. The second row is a warning. The attack wins more often there, but only because most games are played in the world where saying 0 is easy. Always saying 0, with no attack at all, would win 900 of those games. Accuracy alone cannot tell a good attack from a good guess about the prior, which is why every game below is scored against a baseline.

Most evaluations use a **fair bit**, $\pi_1 = \frac12$: half the targets are members and half are not, so that neither answer is right more often by default. In that game an adversary who never looks at its view, and so has nothing to go on, wins exactly half the games, whatever it answers; a coin flip, "always say member" and "always say non-member" all score $\frac12$. That is the sense of "coin flip" below: not a model of any attack, but the score of knowing nothing in a fair game. With a fair bit, @eq-threat-win becomes $\frac12\mathrm{TPR} + \frac12(1 - \mathrm{FPR}) = \frac12 + \frac12(\mathrm{TPR} - \mathrm{FPR})$: the know-nothing score, plus half of the difference between the two rates. That difference is the **advantage**,

$$\mathrm{Adv} = \mathrm{TPR} - \mathrm{FPR} = 2\Pr[b' = b] - 1,$$ {#eq-threat-adv}

and it measures how much better the attack does than an adversary who ignores its view. The second equality is the fair-bit formula solved for $\mathrm{TPR} - \mathrm{FPR}$: from $\Pr[b' = b] = \frac12 + \frac12(\mathrm{TPR} - \mathrm{FPR})$, subtract $\frac12$ and multiply by 2.

Why does this number measure the gain over knowing nothing? Look at what it is: $\mathrm{TPR} - \mathrm{FPR}$ is how much more often the attack says "member" when the target is a member than when it is not. Three cases show what that means.

1. **An attack that ignores its view scores 0.** A coin flip, or "always say member", does not look at anything that differs between the two worlds, so it says 1 with the same probability in both. Its TPR equals its FPR, its advantage is 0, and by the fair-bit formula it wins with probability exactly $\frac12$.
2. **A perfect attack scores 1.** It says 1 every time the target is a member and never otherwise: $\mathrm{TPR} = 1$, $\mathrm{FPR} = 0$, advantage 1, and it wins every game.
3. **Every attack in between is scored by its extra wins.** Rearranged, @eq-threat-adv says $\mathrm{Adv} = 2\big(\Pr[b' = b] - \frac12\big)$: the amount by which the attack's win probability exceeds the know-nothing score $\frac12$, doubled so that a perfect attack scores 1. An attack can only win more than half the time if its answer depends on which world it is in, and the advantage measures exactly how much it does.

A negative advantage means the attack says "member" more often for non-members; flipping its answers turns it into an attack with the same advantage, positive. In the first row of @tbl-threat-win-count the advantage is $0.6 - 0.2 = 0.4 = 2(0.7) - 1$: the attack wins 0.2 more often than a coin flip, and 0.2 doubled is 0.4. Unlike the win rate, the advantage does not depend on the prior, because TPR and FPR are each measured inside one world. That matters because real targets are rarely half members. With an unequal prior the know-nothing score is no longer $\frac12$ but the larger of $\pi_0$ and $\pi_1$, which is 0.9 in the second row of @tbl-threat-win-count, where always saying 0 wins 900 games of 1,000. The advantage needs no adjustment: an attack that ignores its view still has $\mathrm{TPR} = \mathrm{FPR}$ and advantage 0, at any prior. Every two-world game in this part is scored by its advantage, and the best possible guess with an unequal prior (@eq-threat-bayes) starts from @eq-threat-win. The same fact, in the language of security games, is Fact 1 of [Cryptography basics](../part01-prerequisites/08-crypto-basics.md).

### Fact 2 — Processing the view cannot help the adversary

If the adversary throws information away, or adds noise of its own, it cannot do better. Formally, let $V' = g(V, R)$ for any function $g$ and any randomness $R$ independent of $b$. Then every attack that sees $V'$ has the same TPR and FPR as some attack that sees $V$, so the best advantage from $V'$, and the best TPR at every FPR, are at most those from $V$.

The reason is that whoever holds $V$ can always produce $V'$ themselves. Given an attack $\mathcal A'$ on $V'$, build an attack $\mathcal A$ on $V$ that draws $R$, computes $g(V, R)$ and runs $\mathcal A'$ on the result. In each world $\mathcal A$'s output has the same distribution as $\mathcal A'$'s, so it has the same TPR and FPR. For instance, if a service returns a confidence score $s$ and you only look at whether $s > 0.5$, you can do no better than someone who looks at $s$ itself, because they could apply the same cut-off.

This is the data processing inequality of [Divergences](../part01-prerequisites/03-divergences.md), seen from the attacker's side. It is what orders the access levels of a threat model.

### Fact 3 — Mixing hides, and knowing the mixture index undoes it

This fact compares adversaries who know different amounts, so it needs a number for "the best any attack can do against views $P$ and $Q$". That number is the **total variation distance**. For views with probabilities $p(x)$ and $q(x)$, it is half the total absolute difference between the two probability tables,
$$\mathrm{TV}(P, Q) = \tfrac12\sum_x|p(x) - q(x)|.$$
It is the best possible advantage for a simple reason. An attack that says "member" on a set $A$ of views has $\mathrm{TPR} = Q(A)$ and $\mathrm{FPR} = P(A)$, so its advantage is $Q(A) - P(A) = \sum_{x \in A}(q(x) - p(x))$. That sum is largest when $A$ contains exactly the views where $q(x) > p(x)$, the views that are more likely in the member world, and its largest value is the total excess of $q$ over $p$, which is half the total absolute difference. @sec-threat-advantage returns to this with the full error curve.

Now the fact itself. Often the adversary's view depends on something it does not know, such as the other records in the dataset. Not knowing it works like extra noise: the adversary sees an average of several possible situations, and averaging two distributions can only blur the difference between them. Knowing it removes that noise, and being allowed to choose it lets the adversary pick the situation where the difference is largest.

In symbols, call the unknown $\Omega$, with weights $w_\omega$, and let $P_\omega$ and $Q_\omega$ be the views in the two worlds when $\Omega = \omega$. An adversary who does not see $\Omega$ faces the averages $\bar P = \sum_\omega w_\omega P_\omega$ and $\bar Q = \sum_\omega w_\omega Q_\omega$. An adversary who sees it faces the pair $(\Omega, V)$. Their best advantages are ordered, from not knowing, to knowing, to choosing:

$$\mathrm{TV}(\bar P, \bar Q) \;\le\; \sum_\omega w_\omega\,\mathrm{TV}(P_\omega, Q_\omega) \;=\; \mathrm{TV}\big((\Omega, V)_{b=0}, (\Omega, V)_{b=1}\big) \;\le\; \max_\omega \mathrm{TV}(P_\omega, Q_\omega).$$ {#eq-threat-mixing}

Each step has a short proof. The first inequality is the triangle inequality, $|u + v| \le |u| + |v|$, which says that differences can cancel when they are added up before taking absolute values, but not after: $\mathrm{TV}(\bar P, \bar Q) = \frac12\sum_x\big|\sum_\omega w_\omega(p_\omega(x) - q_\omega(x))\big| \le \frac12\sum_x\sum_\omega w_\omega|p_\omega(x) - q_\omega(x)|$, and swapping the sums gives the middle term. The equality holds because the pair $(\Omega, V)$ has probability $w_\omega p_\omega(x)$ in world 0 and $w_\omega q_\omega(x)$ in world 1, so half the $L^1$ distance between the two worlds is exactly $\sum_\omega w_\omega\mathrm{TV}(P_\omega, Q_\omega)$. The last inequality says that an average is at most its largest term.

A tiny case shows how much the unknown can hide. Let there be one other record, with a bit $\omega \in \{0, 1\}$ that is 0 or 1 with probability $\frac12$ each, and let the mechanism release the exact count $\omega + b$. An adversary who knows $\omega$ subtracts it and reads off $b$, so for each $\omega$ the TV is 1. An adversary who does not know it sees counts 0 or 1 in world 0 and 1 or 2 in world 1, each with probability $\frac12$. A count of 1 could come from either world, and the TV drops to $\frac12$ ([Exercise 7](../../book-exercises/part02-attacks/01-threat-models.md#sec-sol-threat-7)). This fact is what separates the average-case and worst-case games, and it is how the chapter prices side information.

## Threat models {#sec-threat-models}

### Context

Papers on privacy attacks report numbers such as "membership advantage 0.2" or "90% of records reconstructed". Those numbers mean nothing until you know what the attacker was assumed to see, know and do. A threat model fixes these assumptions. This section lists what a threat model has to specify, and proves the one ordering that always holds among the choices: an adversary with more access is never weaker.

### Statement

A threat model answers five questions.

1. **Goal: what is the secret?** Whether a given record was in the training data (membership), the value of a sensitive field of a known record (attribute), the records themselves (reconstruction), or a statistic of the dataset as a whole (property).
2. **Access: what does the adversary observe?** From least to most: only the model's decisions on inputs it chooses, such as a predicted class or an "anomaly" flag (**label-only**); the model's scores on those inputs, such as confidences, losses or log-probabilities (**black-box**); the parameters $\theta$ themselves (**white-box**); or everything the training process exposed along the way, such as intermediate models and per-step gradients (the **transcript**).
3. **Behaviour: does it follow the rules?** An **honest-but-curious** (semi-honest) adversary follows the protocol and only analyses what it legitimately sees. A **malicious** (active) adversary may deviate: choose its queries adaptively, insert crafted records into the training data (**poisoning**), or, if it runs part of the system, send tampered models to the other participants.
4. **Trust: who sees the raw data?** In the **central** model a trusted curator holds the data, and the adversary sees only what is released. In the **local** model nobody is trusted: each person's data leave their device already protected, and the adversary may be the collector itself. Distributed settings, where many parties each hold a slice, sit in between.
5. **Side information: what does it know already?** The population distribution, a public dataset drawn from it, some or all of the other records, the target's non-sensitive fields.

Of these, access is the one with a natural order, and the reason is Fact 2. Each level of access lets the adversary compute the level below it: from the weights it can compute any score, and from a score it can compute the label. So whatever an attack does with less access, an attack with more access can do too, and the best attack can only get better as access grows. Write $\mathrm{Adv}^*(V) = \max_{\mathcal A} \mathrm{Adv}(\mathcal A)$ for the best advantage of any attack whose input is the view $V$. For the same model and the same side information $S$,

$$\mathrm{Adv}^*(\text{labels}, S) \le \mathrm{Adv}^*(\text{scores}, S) \le \mathrm{Adv}^*(\theta, S) \le \mathrm{Adv}^*(\text{transcript}, S),$$ {#eq-threat-access-chain}

and the same chain holds for the best TPR at every fixed FPR. In the same way, more side information never lowers the best advantage, $\mathrm{Adv}^*(V) \le \mathrm{Adv}^*(V, S)$, and neither does allowing malicious behaviour.

### Derivation

1. **Each view can be computed from the next one.** A label is a function of the scores: the class with the largest score, or "score above a cut-off". A score on a query $x$ is a function of $\theta$ and $x$, and the adversary chose $x$ itself. The final $\theta$ is the last entry of the transcript.
2. **So Fact 2 applies at each link.** Each view on the left of @eq-threat-access-chain is $g(V, R)$ for the view $V$ to its right. Its achievable $(\mathrm{FPR}, \mathrm{TPR})$ pairs are therefore a subset of those to its right, and a maximum over a smaller set is smaller.
3. **Side information works the same way.** $V$ is what you get from $(V, S)$ by forgetting $S$, so Fact 2 applies again.
4. **Behaviour is a larger set of strategies.** An honest-but-curious adversary is a malicious one that chooses not to deviate, and a maximum over more strategies is at least as large.

### Interpretation

The chain compares the *best* attacks at each level, not the attacks people actually run. A particular white-box attack can do worse than a well-designed black-box one, and the example below shows the gap inside a single access level can be larger than the gap between levels.

The access levels map directly onto deployments. Label-only and black-box access describe a model behind an application programming interface (API), such as a cloud prediction service or a chatbot. White-box access describes released weights, and also every participant who receives the global model in federated learning. The transcript describes a federated-learning server that sees each client's updates, the setting of [Gradient inversion](06-gradient-inversion.md). @tbl-threat-deployments places four common deployments on all the axes. The same trained model can sit in several rows at once, and its risk is set by the weakest one.

| Deployment | Access | Behaviour | Trust | Typical side information |
| --- | --- | --- | --- | --- |
| Prediction service that returns the top class | label-only | honest-but-curious user, adaptive queries | central | public data from the same population |
| Service that returns confidences or log-probabilities | black-box | honest-but-curious user, adaptive queries | central | public data; the target's record |
| Model weights published for download | white-box | anyone; can run any computation | central | public data; possibly some training records |
| Federated training with a coordinating server | transcript (per-client updates) | server may be malicious | distributed | the global model at every round |

: Four deployments read as threat models. {#tbl-threat-deployments}

Trust works differently from the other axes: it decides *who* the adversary is, not how strong it is. In the central model the threat is whoever sees the released model, and the curator is outside the threat model. In the local model the curator is inside it, so protection has to be applied before data leave the user. Part 3 compares the two.

Side information is the hardest axis to bound, because the defender usually does not know what the adversary knows. That is why the strongest assumption, "the adversary knows every record except the target's", is the one a guarantee should survive (@sec-threat-average).

### Example: a centroid anomaly detector at four access levels

A security team fits the simplest anomaly detector there is: the mean $\theta = \frac1n\sum_i x_i$ of $n = 100$ records $x_i \sim \mathcal N(0, I_d)$ with $d = 50$ features. A point $x$ gets the anomaly score $\|x - \theta\|^2$ and is flagged when the score exceeds $\tau = d$. The adversary holds a target $z$ and plays the membership game: if $b = 1$, $z$ is one of the $n$ records; if $b = 0$, it is a fresh draw from the population. Here is what the adversary can do at each level of access.

With the weights (**white-box**), the adversary knows $\theta$. The best thing it can do with $\theta$ is to ask how much more likely this exact $\theta$ is in the member world than in the non-member world. That ratio of the two probability densities is the **likelihood ratio**, and the Neyman–Pearson lemma of [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) says that, for any target false positive rate, no attack catches more members than one that says "member" when the likelihood ratio is above a threshold. Intuitively, the views that are most typical of members relative to non-members are the ones to flag first.

Here the two densities are Gaussians. If $z$ was used, $\theta = (z + W)/n$, where $W \sim \mathcal N(0, (n - 1)I_d)$ is the sum of the other records, so $\theta$ is pulled a little towards $z$. If it was not, $\theta \sim \mathcal N(0, I_d/n)$. Expanding the squares in the two Gaussian densities (@sec-apx-opt-gibbs in the appendices does it step by step) gives the log of the likelihood ratio,

$$\ln\frac{q(\theta \mid z)}{p(\theta \mid z)} = \frac{n}{2(n - 1)}\big(2\langle z, \theta\rangle - \|\theta\|^2\big) - \frac{\|z\|^2}{2(n - 1)} + \text{const},$$

and the best white-box attack thresholds it. The bracket $2\langle z, \theta\rangle - \|\theta\|^2$ is the part that depends on $\theta$: it grows when $\theta$ points towards $z$, which is exactly what training on $z$ does.

With only the score (**black-box**), the adversary gets $\|z - \theta\|^2 = \|z\|^2 - \big(2\langle z, \theta\rangle - \|\theta\|^2\big)$. It knows $z$, so the **calibrated score** $\|z\|^2 - \|z - \theta\|^2$ gives back exactly the bracket that the white-box likelihood ratio needs, and the remaining term $-\|z\|^2/(2(n - 1))$ it can compute itself. The calibrated score drops that term, which only shifts the threshold from one target to the next, so it is almost, not exactly, the likelihood-ratio test: in @tbl-threat-access the two agree to four decimals. In this model the score carries everything about $\theta$ that matters for the decision (it is a **sufficient statistic**: the likelihood ratio can be computed from it alone), so white-box access adds nothing. A simpler black-box attack ignores $\|z\|^2$ and says "member" when the raw score is low. That attack is confused by targets that are simply far from the mean: they have a high score whether or not they were used, so it mixes a target's typicality with its membership. With only the flag (**label-only**), the adversary says "member" when $z$ is not flagged.

The script plays 100,000 games in each world and reports the best advantage over thresholds of each score (@tbl-threat-access).

| View | Attack | Best advantage $\mathrm{TPR} - \mathrm{FPR}$ |
| --- | --- | --- |
| White-box | likelihood ratio of $\theta$ | 0.2789 |
| Black-box | calibrated score $\|z\|^2 - \|z - \theta\|^2$ | 0.2789 |
| Black-box | raw loss $-\|z - \theta\|^2$ | 0.0407 |
| Label-only | not flagged, $\tau = d$ | 0.0400 |

: Membership advantage against the centroid detector with $n = 100$ and $d = 50$, from 100,000 games per world. {#tbl-threat-access}

The chain @eq-threat-access-chain holds: $0.0400 \le 0.2789 \le 0.2789$. But the large gap is *inside* the black-box level, between two attacks on the same view. Calibrating each target against its own typicality multiplies the advantage by about seven. That is the idea behind the per-example attacks of [Membership inference](02-membership-inference.md). Sablayrolles et al. (2019) give a general version: if training draws the parameters with a probability that decreases exponentially in the training loss, the Bayes-optimal membership attack depends on the parameters, to a good approximation, only through the target's loss, calibrated for each target. White-box access then adds little, as here, where the dependence is exact (@sec-apx-opt-gibbs in the appendices derives both).

@fig-threat-access shows the calibrated scores in the two worlds, and the advantage at each level as the dimension grows. With more parameters per record, a single record moves the model in more directions at once, and the best advantage climbs from 0.0326 at $d = 1$ to 0.7407 at $d = 500$ (100,000 games per world).

![Left: the calibrated black-box score $\|z\|^2 - \|z - \theta\|^2$ for members and non-members of the centroid detector, $n = 100$, $d = 50$. Right: the best advantage at each access level against the dimension $d$, from 20,000 games per world and point. White-box and calibrated black-box coincide; raw loss and label-only are far below.](figures/threat_access.svg){#fig-threat-access}

### Check yourself

- A service adds a little random noise to every confidence score it returns. Can that make some attack stronger? *(No. The noisy score is $g(\text{score}, R)$ with $R$ independent of $b$, so by Fact 2 every attack on it is matched by an attack on the clean score. Noise can only lower the best advantage, though it may not lower it by much.)*
- Why can a black-box attack beat a white-box attack in practice, despite @eq-threat-access-chain? *(The chain compares the best attack at each level. A particular white-box attack can be badly designed, just as the raw-loss attack is far worse than the calibrated one in @tbl-threat-access.)*

## Privacy games {#sec-threat-games}

### Context

The same name, such as "membership inference", is used for attacks with different assumptions and different ways of scoring success. Salem et al. ("SoK: Let the Privacy Games Begin!", 2023) collect the attacks in the literature and write each one as a game between a challenger and an adversary. A game pins down three things: how the secret and the data are drawn, what the adversary sees, and how its guess is scored against a baseline. This section defines the games used in the rest of the part, and works out the best possible guess when the secret is a bit.

### Statement

Every game in this part has the same four steps. The challenger draws a secret $s$ from a known distribution, together with a dataset $D$ whose distribution may depend on $s$. It trains $\theta = \mathcal T(D)$ and gives the adversary the view $V$ that the threat model allows, along with the side information $S$. The adversary outputs a guess $s'$ and earns a **gain** $g(s, s')$, usually $\mathbf 1[s' = s]$. Finally, the attack is compared with a **baseline**: the best expected gain of a guess made from $S$ alone, without looking at $V$. @fig-threat-game draws the membership game, and @tbl-threat-games lists the others.

| Game | Secret $s$ | Data in the two worlds | Gain | Baseline |
| --- | --- | --- | --- | --- |
| Membership | bit $b$ | $b = 1$: target $z$ drawn from $D$; $b = 0$: $z$ fresh from $\mathcal D$ | $\mathbf 1[b' = b]$ | $\frac12$ |
| Attribute | sensitive field $a$ of a known member $z = (u, a)$ | $z \in D$; adversary knows $u$ | $\mathbf 1[a' = a]$ | best guess of $a$ from $u$ and $\mathcal D$; or the same attack on non-members |
| Reconstruction | the records of $D$ (or one column) | $D$ drawn from $\mathcal D$ | fraction of entries recovered | guessing from $\mathcal D$ |
| Distinguishing | bit $b$ | adversary chooses $D_0, D_1$ differing in one record; challenger trains on $D_b$ | $\mathbf 1[b' = b]$ | $\frac12$ |
| Property | bit $b$ | $D$ drawn from $\mathcal D_{t_b}$, populations differing in a statistic $t$ | $\mathbf 1[b' = b]$ | $\frac12$ |

: The privacy games of this part. The first three are average-case: data come from the population. The distinguishing game is worst-case: the adversary picks the data. {#tbl-threat-games}

![The membership game. The challenger trains on a dataset, flips a bit that decides whether the target was in it, and shows the adversary the target and a view of the model. The adversary's guess is scored by its advantage.](figures/threat_game.svg){#fig-threat-game}

When the secret is a bit that is not equally likely to be 0 or 1, the best possible guess has a closed form, and the idea behind it is simple. Say the bit is 1 only rarely, with prior $\pi_1 < \frac12$. Then the safe default is to guess 0, which already wins with probability $\pi_0$. The view should overturn that default only where it is strong enough evidence for 1 to outweigh the prior: where a view is more than $\gamma = \pi_0/\pi_1$ times as likely when the bit is 1 as when it is 0. With $\pi_1 = 0.1$ that is 9 times as likely. The best attack's gain over the default is the probability mass on those strongly convincing views.

The quantity that measures this mass is the **hockey-stick divergence**. With $p$ and $q$ the probabilities of a view when the bit is 0 and 1,
$$E_\gamma(Q\|P) = \sum_x\max\big(q(x) - \gamma p(x),\, 0\big),$$
the total by which $q$ exceeds $\gamma$ times $p$, counted only on the views where it does. At $\gamma = 1$ it is the total variation distance of Fact 3. The best attack wins with probability

$$\max_{\mathcal A}\Pr[b' = b] = \pi_0 + \pi_1\,E_\gamma(Q\|P), \qquad \gamma = \frac{\pi_0}{\pi_1}.$$ {#eq-threat-bayes}

The first term is the baseline of always guessing 0; the second is everything the view adds.

### Derivation

**The Bayes-optimal guess** (@eq-threat-bayes).

1. **Write an attack as a set.** A deterministic attack says 1 on some set $A$ of views. By @eq-threat-win it wins with probability $\pi_1Q(A) + \pi_0(1 - P(A))$.
2. **Pull out the baseline.** This equals $\pi_0 + \pi_1\big(Q(A) - \gamma P(A)\big)$ with $\gamma = \pi_0/\pi_1$.
3. **Choose the best set.** $Q(A) - \gamma P(A) = \sum_{x \in A}\big(q(x) - \gamma p(x)\big)$ is a sum over the views in $A$. To make it as large as possible, put a view in $A$ exactly when its term is positive, that is when $q(x) > \gamma p(x)$. The maximum is then the sum of the positive terms, which is $E_\gamma(Q\|P)$. In words: say 1 when the likelihood ratio $q/p$ exceeds the prior odds $\gamma$ against the bit being 1. A randomized attack says 1 on view $x$ with some probability $\phi(x)$ and gets $\sum_x\phi(x)(q(x) - \gamma p(x))$, which is also largest with $\phi = 1$ on the positive terms and 0 elsewhere, so randomizing does not help.
4. **Check the fair case.** With $\pi_1 = \frac12$, $\gamma = 1$ and $E_1 = \mathrm{TV}$, so the best attack wins with probability $\frac12 + \frac12\mathrm{TV}$, matching Fact 1 with the best advantage TV of Fact 3.

**Reconstruction from one exact count.** The secret is $m$ uniform bits $x_1, \dots, x_m$, and the release is their sum $K$.

1. **What the count says about each bit.** Given $K = k$, every arrangement of $k$ ones among $m$ places is equally likely, so each bit is 1 with probability $k/m$.
2. **The best guess for each bit.** Guess 1 when $k/m > \frac12$ and 0 otherwise. Every bit gets the same guess, the majority value, and the expected number right is $\max(k, m - k)$.
3. **Average over the count.** With $K \sim \mathrm{Bin}(m, \frac12)$, the expected fraction recovered is $\mathbb E[\max(K, m - K)]/m = \frac12 + \mathbb E|K - \frac m2|/m$.
4. **How large the gain is.** The count is a sum of $m$ fair bits, each with variance $\frac14$, so by the central limit theorem $K - \frac m2$ is approximately normal with mean 0 and variance $m/4$. The average distance of a normal variable from its mean is its standard deviation times $\sqrt{2/\pi}$ (integrate $|x|$ against the normal density), here $\frac{\sqrt m}{2}\sqrt{2/\pi}$. Dividing by $m$, the fraction recovered is approximately $\frac12 + \frac1{\sqrt{2\pi m}}$. @sec-apx-prob-stirling in the appendices computes it exactly, as $\frac12 + \frac12\binom{2j}{j}4^{-j}$ with $j = \lfloor m/2\rfloor$, and proves the approximation without the central limit theorem.

### Interpretation

The baseline is part of the game, not an afterthought. A model that lets an attacker guess a rare diagnosis "with 90% accuracy" reveals nothing if 90% of people do not have that diagnosis. Accuracy only means something next to the baseline of @eq-threat-bayes.

Attribute inference needs particular care, because it has two possible baselines that answer different questions. Comparing with a guess from the population asks "does the model help?". Comparing members with non-members asks "does the model help *because this person was in the training data*?". Only the second is a breach of that person's privacy; the first may be ordinary statistics that any study of the population would reveal. [Attribute inference and model inversion](03-attribute-inference.md) develops this.

The games also differ in where the data come from. In the membership, attribute and reconstruction games the data are drawn from the population, so the score is an average over datasets and targets. In the distinguishing game the adversary chooses both datasets, so the score is a worst case. @sec-threat-average shows that the first is never larger than the second, and how far apart they can be. Finally, games and threat models combine freely: every row of @tbl-threat-games can be played at every access level, so "label-only membership" and "white-box reconstruction" are both well defined.

### Example: one noisy count, three games

A data team publishes how many people in a dataset have a sensitive condition, with Gaussian noise added: $Y = \sum_{x \in D}x + \sigma Z$, with bits $x \in \{0, 1\}$, $Z \sim \mathcal N(0, 1)$ and $\sigma = 1$. Suppose the adversary knows every record except the target's, so the rest of the count is a known number $c$. The same release can be attacked in three games.

In the **membership** game, the target has the condition. Without the target, $Y \sim \mathcal N(c, 1)$; with it, $Y \sim \mathcal N(c + 1, 1)$: the same bell curve, shifted right by one. The best attack says "member" on the views more likely in the member world, which are the counts to the right of the point where the two curves cross, $c + \frac12$. Write $\Phi$ for the standard normal cumulative distribution function (CDF), so $\Phi(t)$ is the probability that a standard normal variable is below $t$. A member's count lands above $c + \frac12$ with probability $\Phi(\frac12)$, its TPR, and a non-member's with probability $1 - \Phi(\frac12)$, its FPR. The advantage is $2\Phi(\frac12) - 1$, and by Fact 1 the attack wins with probability $\frac12 + \frac12\big(2\Phi(\frac12) - 1\big) = \Phi(\frac12) = 0.6915$.

In the **attribute** game with a fair prior, the target is known to be in $D$ and its bit is the secret. The view has the same two distributions, so the best guess is again right with probability 0.6915 (0.6925 in 200,000 simulated games). Now make the condition rare, with prior $\pi_1 = 0.1$. Always guessing "no" already wins with probability 0.9, and by @eq-threat-bayes the release should overturn that only where the count is at least $\gamma = 9$ times as likely with the condition as without. For two unit-variance normals one apart, the likelihood ratio at $y$ is $e^{(y - c) - 1/2}$, which exceeds 9 when $y > c + \frac12 + \ln 9$. On those counts, $Q$ puts mass $\Phi(\frac12 - \ln 9)$ and $P$ puts mass $\Phi(-\frac12 - \ln 9)$, so $E_9 = \Phi(\frac12 - \ln 9) - 9\Phi(-\frac12 - \ln 9) = 0.0134$. The best accuracy is $0.9 + 0.1 \times 0.0134 = 0.9013$ (simulated: 0.9028). The same release that lifts a fair guess from 0.5 to 0.69 lifts this one by about a tenth of a percentage point, because the likelihood ratio is rarely large enough to overturn 9-to-1 odds.

In the **reconstruction** game, release the exact count of $m$ uniform bits instead. The best reconstruction recovers a fraction 0.6230 of the bits for $m = 10$ (0.6236 in simulation), 0.5398 for $m = 100$ and 0.5126 for $m = 1000$. The approximation $\frac12 + 1/\sqrt{2\pi m}$ gives 0.6262, 0.5399 and 0.5126. One aggregate reveals little about each record, and less as $m$ grows. Many aggregates are a different matter: [Reconstruction attacks](09-reconstruction.md) shows how a few times as many noisy subset counts as there are bits give away almost every bit.

### Check yourself

- In the distinguishing game the adversary picks $D_0$ and $D_1$. Why does that make its best advantage at least the best advantage in the membership game, for the same mechanism? *(It can pick the pair that is hardest to hide, for example the dataset and target on which the mechanism leaks the most. The membership game averages over pairs drawn from the population, and a maximum is at least an average, as @sec-threat-average makes precise.)*
- An attribute attack is right 75% of the time on members and 75% of the time on non-members. Is it a privacy attack on the members? *(No. It does equally well on people who were never in the data, so it is inferring the attribute from the population, not from the person's participation.)*

## Advantage, total variation and the trade-off curve {#sec-threat-advantage}

### Context

A two-world game is a test of $H_0: V \sim P$ ("not a member") against $H_1: V \sim Q$ ("member"). So everything [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) proved about tests becomes a statement about attacks. This section draws out three: a ceiling on the advantage, a ceiling on the whole ROC curve, and a warning about which point of that curve to report.

### Statement

Fact 3 already showed that no attack's advantage exceeds the total variation distance between the two worlds' views. The first bound restates that. The second goes further: it bounds the attack's TPR at *every* false positive rate, not just at the one point where TPR − FPR is largest. The bound comes from the **trade-off function** $T(P, Q)$: for each false positive rate $\alpha$, $T(P, Q)(\alpha)$ is the smallest fraction of members that any attack with false positive rate $\alpha$ must miss. An attack can catch at most the rest. For every attack in a two-world game,

$$\mathrm{Adv} = \mathrm{TPR} - \mathrm{FPR} \le \mathrm{TV}(P, Q), \qquad \Pr[b' = b] \le \tfrac12 + \tfrac12\mathrm{TV}(P, Q),$$ {#eq-threat-tv}

$$\mathrm{TPR} \le 1 - T(P, Q)(\mathrm{FPR}).$$ {#eq-threat-tradeoff}

Both bounds are attained by likelihood-ratio attacks, which flag the views that are most typical of members relative to non-members first.

The third quantity answers the question a person who is flagged cares about: given that the attack says "member", how likely is that to be true? Of all the targets the attack flags, some are members (a fraction $\pi_1$ of targets are members, and it flags a fraction TPR of them) and some are non-members (a fraction $\pi_0$, of which it flags FPR). The **precision** is the members' share of the flagged:

$$\Pr[b = 1 \mid b' = 1] = \frac{\pi_1\,\mathrm{TPR}}{\pi_1\,\mathrm{TPR} + \pi_0\,\mathrm{FPR}}.$$ {#eq-threat-precision}

### Derivation

1. **The attack is a test.** It says "member", rejecting $H_0$, with probability $\phi(V)$. Its type I error is the FPR and its type II error is $1 - \mathrm{TPR}$.
2. **The advantage ceiling** (@eq-threat-tv). The advantage of an attack that says "member" with probability $\phi(x)$ on view $x$ is $\sum_x\phi(x)(q(x) - p(x))$. It is largest when $\phi = 1$ exactly where $q > p$, that is, where the likelihood ratio $q/p$ is above 1, and that largest value is $\mathrm{TV}(P, Q)$, as in Fact 3. The winning probability follows from Fact 1.
3. **The curve ceiling** (@eq-threat-tradeoff). The fraction of members an attack misses is $1 - \mathrm{TPR}$. By the definition of the trade-off function, no attack with false positive rate FPR misses fewer than $T(P, Q)(\mathrm{FPR})$, so $1 - \mathrm{TPR} \ge T(P, Q)(\mathrm{FPR})$. Which attack attains it? The Neyman–Pearson lemma answers: with a budget of false positives to spend, spend it on the views where members are most over-represented, that is, flag views in decreasing order of the likelihood ratio $q/p$ until the false positive rate reaches the budget. Each view flagged that way buys the most members per false positive.
4. **Precision** (@eq-threat-precision). The precision is a conditional probability, so it is the probability that a target is a member *and* flagged, divided by the probability that it is flagged. The first is $\pi_1\mathrm{TPR}$, as in step 2 of Fact 1. A target is flagged either as a member, $\pi_1\mathrm{TPR}$, or as a non-member, $\pi_0\mathrm{FPR}$, and these are disjoint, so the second is their sum.

### Interpretation

The advantage is only one point of the ROC curve, the plot of an attack's TPR against its FPR as its threshold varies. Because the advantage is TPR − FPR, it is the height of that curve above the diagonal, and TV is the largest such height, reached at the likelihood-ratio threshold 1, and it says nothing about what happens at the low false-positive end. Yet that end is where privacy breaks. An attack that flags 1% of members with no false alarms has an advantage of only about 0.01, yet it identifies those members with certainty. An attack with advantage 0.1, right on 55% of members and wrong on 45% of non-members, identifies nobody with confidence. Carlini et al. (2022) argue from this that membership attacks should be reported by their TPR at a low FPR, such as $10^{-3}$, on ROC curves with log axes. The rest of this part does so.

Precision tells the same story from the attacker's side. If only one target in a hundred is a member, even a good attack produces mostly false alarms (Jayaraman et al. 2021). A low FPR is what keeps precision high when members are rare.

The two ceilings depend only on $P$ and $Q$, so they bound every attack, including ones nobody has invented yet. That is what makes them the right target for a defence: a mechanism whose trade-off curve $T(P, Q)$ stays close to the diagonal for every target defeats every attack at once.

### Example: the noisy count, worst case

Take the noisy count of the previous example, $\sigma = 1$, with the other $m = 100$ bits known (the script fixes them at 50 ones). Then $P = \mathcal N(50, 1)$, $Q = \mathcal N(51, 1)$, and as computed there the best attack says "member" when $Y > 50.5$, the point where the two curves cross, with advantage $\mathrm{TV} = 2\Phi(\frac12) - 1 = 0.3829$. Over 200,000 games in each world it scores $\mathrm{TPR} = 0.6901$ and $\mathrm{FPR} = 0.3092$: advantage 0.3809 and winning probability 0.6905, against the ceilings 0.3829 and 0.6915.

Now hold the false positive rate at $10^{-3}$. The attack still flags large counts, but it has to move its threshold $\tau$ up until only one non-member in a thousand exceeds it: $\Pr[\mathcal N(50, 1) > \tau] = 10^{-3}$, so $\tau = 50 + \Phi^{-1}(0.999) = 50 + 3.0902 = 53.09$, where $\Phi^{-1}$ is the inverse of $\Phi$. A member's count, centred one higher, exceeds $\tau$ with probability $1 - \Phi(\tau - 51) = 1 - \Phi(3.0902 - 1) = 0.0183$. That is the TPR, and by the Neyman–Pearson argument above it is the best possible at this FPR (simulated: 0.0184, at an FPR of 0.00109). The attack now catches fewer than 2% of members, but when it fires it is right with precision $0.0183/(0.0183 + 0.001) = 0.948$ at a fair prior. If only one target in a hundred is a member, the precision drops to 0.156.

The black curve in @fig-threat-tradeoff is this ROC curve on log axes. The green one is the same game when the adversary does not know the other records, which is the subject of the next section.

![Left: the released count in the two worlds when the other 100 records are known (solid) and unknown (dashed). Right: the membership ROC curves of the best attacks on log axes, exact (lines) and from 100,000 simulated games per world (dots). The dotted diagonal is random guessing.](figures/threat_tradeoff.svg){#fig-threat-tradeoff}

### Check yourself

- Attack A has TPR 0.5 at FPR 0.3; attack B has TPR 0.02 at FPR $10^{-4}$. Which has the larger advantage, and which is the bigger privacy threat? *(A has advantage 0.2, B about 0.02. B is the bigger threat to the people it flags: its precision at a fair prior is $0.02/(0.02 + 0.0001) = 0.995$, against $0.5/0.8 = 0.625$ for A.)*
- Why can no attack, however clever, beat @eq-threat-tradeoff? *(The bound is a property of the two view distributions. Any attack is a test of $P$ against $Q$, and the Neyman–Pearson lemma says no test with that FPR has more power than the likelihood-ratio test.)*

## Average case and worst case {#sec-threat-average}

### Context

A membership game can be scored in two ways. In the **average case**, the data and the target come from the population, the adversary knows only the population, and the advantage is an average over all of that randomness. In the **worst case**, the adversary knows, or even picks, the rest of the data and the target, and the advantage is the maximum over those choices. Evaluations of real models usually measure the first; privacy guarantees have to control the second. This section shows how far apart the two can be.

### Statement

The comparison is Fact 3 again, with the unknown being the rest of the game. Let $\omega$ collect everything that the average-case game draws at random and that a worst-case adversary is allowed to fix: the other records, and the target. An average-case adversary does not know $\omega$, so it sees an average over its possible values, which blurs the two worlds together. An adversary who knows $\omega$ sees each situation sharply, and one who chooses $\omega$ picks the sharpest. Let $P_\omega, Q_\omega$ be the views given $\omega$, and $\bar P, \bar Q$ their averages. Then, by Fact 3,

$$\underbrace{\mathrm{TV}(\bar P, \bar Q)}_{\text{adversary does not know }\omega} \;\le\; \underbrace{\mathbb E_\omega\,\mathrm{TV}(P_\omega, Q_\omega)}_{\text{adversary knows }\omega} \;\le\; \underbrace{\max_\omega\,\mathrm{TV}(P_\omega, Q_\omega)}_{\text{adversary chooses }\omega}.$$ {#eq-threat-avg-worst}

Both gaps can be as large as possible: there are mechanisms for which the left side is close to 0 and the right side is 1.

### Derivation

1. **Three adversaries, three views.** An adversary that does not know $\omega$ sees $V$, distributed as the average $\bar P$ or $\bar Q$, so its best advantage is $\mathrm{TV}(\bar P, \bar Q)$ by @eq-threat-tv. One that knows $\omega$ sees the pair $(\omega, V)$, and by Fact 3 its best advantage is $\mathbb E_\omega\mathrm{TV}(P_\omega, Q_\omega)$. One that chooses $\omega$ picks the most revealing value, which gives the maximum.
2. **The inequalities** are the two inequalities of @eq-threat-mixing.
3. **The gaps can be extreme.** For the left gap, release the exact count, with the target's bit 1 and $m$ other bits the adversary does not know. An adversary who knew the others would read $b$ off the count, so the middle term is 1. One who does not know them sees the count $K$ in world 0 and $K + 1$ in world 1, where $K \sim \mathrm{Bin}(m, \frac12)$ is the unknown sum of the others. The two distributions are the same list of binomial weights $w_k = \Pr[K = k]$, the second shifted one place to the right. Their total variation is the total excess of the shifted list over the original, counted where it is positive. The binomial weights rise up to the middle value $\lfloor m/2\rfloor$ and fall after it (@sec-apx-prob-stirling in the appendices proves it), so the shifted list is the larger one exactly to the right of the middle, and there its excess at $k$ is $w_{k-1} - w_k$. Adding these differences from the middle to the end, every weight cancels except the first, so the total is the largest weight, $\binom{m}{\lfloor m/2\rfloor}2^{-m}$. For large $m$ the binomial is close to a normal curve with standard deviation $\sqrt m/2$, whose peak height is $1/\big(\sqrt{2\pi}\cdot\sqrt m/2\big) = \sqrt{2/(\pi m)}$ ([Exercise 2](../../book-exercises/part02-attacks/01-threat-models.md#sec-sol-threat-2) works this out). It tends to 0 as $m$ grows. For the right gap, release one record of $D$ chosen at random, worked out in the example below.

### Interpretation

Averaging over datasets hides the target in a crowd, but only from an adversary who does not know the crowd. In the noisy count, the unknown other records act as extra noise with variance $m/4$. A defence that relies on that noise fails against an adversary who knows the other records, or who planted them, and inserting records the adversary controls is exactly what a malicious adversary of @sec-threat-models can do.

Averaging over targets hides the worst-treated person. If a mechanism exposes one person in a thousand completely and everyone else not at all, its average advantage is about 0.001. Privacy is a promise to each person, so the number that matters is the advantage against that one person. Outliers and duplicated records are the usual victims, as [Membership inference](02-membership-inference.md) and [Memorization and training-data extraction](05-memorization-extraction.md) show.

Average-case numbers are still useful. They measure the risk that a typical record faces from a realistic attacker, and they are what an audit can estimate from data. But they are lower bounds on the worst case, never upper bounds. An audit narrows the gap by playing the adversary that chooses the target: it plants **canaries**, records built to be as exposed as possible, and measures the attack on those ([Memorization and training-data extraction](05-memorization-extraction.md)).

This is where the part is heading. A guarantee that holds for every pair of neighbouring datasets bounds the right-hand side of @eq-threat-avg-worst, and therefore every membership-style game in this part at once, every game whose two worlds differ in one record. A game whose two worlds differ in many records, such as the property games of [Property inference](04-property-inference.md), is bounded only through a weaker guarantee for groups of records, which degrades with the size of the group. That is how the definition in [Part 3](../part03-dp-foundations/01-definition.md) is designed.

### Example: the noisy count and the random-record release

**The noisy count.** Keep $\sigma = 1$ and the target's bit 1, but now let the $m = 100$ other bits be fresh $\mathrm{Bern}(\frac12)$ draws that the adversary does not see. The count is then $K + b + Z$ with $K \sim \mathrm{Bin}(100, \frac12)$. In the two worlds the count has the same distribution, shifted by 1, and its likelihood ratio increases with the observed count. The reason is that the binomial weights are log-concave and that normal noise makes a larger observation favour a larger centre. The density itself need not be log-concave, and for smaller noise it is not; @sec-apx-tail-noisy-count in the appendices derives the increasing ratio step by step. So the best attack is still a threshold, and by symmetry the two densities cross at $m/2 + \frac12 = 50.5$. @tbl-threat-avg-worst compares the adversaries of @eq-threat-avg-worst, and adds the case where the target's own bit is random.

| Adversary | Best advantage (exact) | Simulated (200,000 games per world) |
| --- | --- | --- |
| Knows the other records ($\sigma = 1$) | 0.3829 | 0.3809 |
| Does not know them, target's bit 1 | 0.0779 | 0.0751 |
| Does not know them, target's bit random | 0.0390 | 0.0398 |
| Knows the other records, exact count ($\sigma = 0$) | 1 | — |
| Does not know them, exact count ($\sigma = 0$) | 0.0796 | — |

: Membership advantage on a noisy count with 100 other records, for adversaries with different knowledge. {#tbl-threat-avg-worst}

Not knowing the crowd cuts the advantage by a factor of five. A normal approximation explains the number: treating $K$ as $\mathcal N(50, 25)$ makes the two worlds Gaussians with shift 1 and standard deviation $\sqrt{25 + 1}$, which gives $2\Phi\big(1/(2\sqrt{26})\big) - 1 = 0.0781$. When the target's own bit is random, the count does not depend on $b$ for the half of targets whose bit is 0, and the advantage halves to $0.0779/2 = 0.0390$. At an FPR of $10^{-3}$ the average-case attack has TPR 0.0019 (simulated 0.0016), against 0.0183 for the adversary who knows the rest; these are the green and black curves of @fig-threat-tradeoff. @fig-threat-average-worst shows the average-case advantage falling as the crowd grows, while the worst case does not move.

![Best membership advantage on the noisy count against the number $m$ of other records the adversary does not know, for three noise levels (solid), and the same mechanism against an adversary who knows them (dashed). Dots: the exact count, $\sigma = 0$, whose worst case is 1.](figures/threat_average_worst.svg){#fig-threat-average-worst}

**The random-record release.** A mechanism publishes one record of $D$, chosen uniformly among $n = 1000$ records that are all different. The attack says "member" if the published record is $z$. If $z$ is in $D$, it is published with probability $1/n$, so $\mathrm{TPR} = 0.001$; if not, it is never published, so $\mathrm{FPR} = 0$. The script measures $\mathrm{TPR} = 0.00098$ and $\mathrm{FPR} = 0$ over 200,000 games per world. By the average-case measure, an advantage of $1/n = 0.001$, the mechanism looks almost perfectly safe. Yet every run exposes one person completely, and when the attack fires its precision is 1. Against the target that is published, the advantage is 1. Only the right side of @eq-threat-avg-worst, which fixes the dataset and asks about the worst target, reveals the problem.

### Check yourself

- A team audits its model by running a membership attack on 10,000 random training and held-out examples and reports advantage 0.01. What can you conclude about the worst-case risk? *(Only that it is at least 0.01. The audit averages over targets and datasets. A small group of outliers could be exposed far more, and the average would barely show it.)*
- In the noisy count with unknown others, how could a malicious adversary recover the worst-case advantage? *(By controlling the other records, for example by flooding the dataset with records whose bits it knows. Then the rest of the count is known, and the advantage goes back to the 0.3829 of the first row of @tbl-threat-avg-worst.)*

## Code: check every number by computation and simulation

The script [code/threat_models.py](../../code/part02-attacks/threat_models.py) computes every number in this chapter and draws every figure. Exact values come from closed forms (normal tails via `math.erfc`, binomial weights) and are checked against games played by simulation. From the repository root:

```bash
uv run python code/part02-attacks/threat_models.py
```

**Output** (seed 0), summarized in @tbl-threat-code:

| Experiment | Result |
| --- | --- |
| Centroid detector, $n = 100$, $d = 50$ | Best advantage: white-box 0.2789; calibrated black-box 0.2789; raw loss 0.0407; label-only 0.0400 |
| Noisy count, rest known, $\sigma = 1$ | TV 0.3829; simulated TPR 0.6901, FPR 0.3092, advantage 0.3809, winning probability 0.6905 |
| Same, at FPR $10^{-3}$ | TPR 0.0183 (simulated 0.0184); precision 0.948 at prior $\frac12$, 0.156 at prior 0.01 |
| Noisy count, 100 unknown others | TV 0.0779 (normal approximation 0.0781; simulated 0.0751); TPR at FPR $10^{-3}$ 0.0019 (simulated 0.0016) |
| Same, random target bit | Advantage 0.0390 (simulated 0.0398) |
| Exact count, 100 unknown others | TV 0.0796; $\sqrt{2/(\pi m)} = 0.0798$ |
| Attribute from the noisy count | Prior $\frac12$: 0.6915 (simulated 0.6925). Prior 0.1: baseline 0.9, best 0.9013, $E_9 = 0.0134$ (simulated 0.9028) |
| Reconstruction from one exact count | $m = 10$: 0.6230 (simulated 0.6236); $m = 100$: 0.5398; $m = 1000$: 0.5126 |
| Random-record release, $n = 1000$ | Simulated TPR 0.00098, FPR 0; $1/n = 0.001$ |

: The script's output with the default seed. {#tbl-threat-code}

[tests/test_threat_models.py](../../tests/test_threat_models.py) checks that the output still matches.

Four things stand out. First, the ceilings are reached: every simulated likelihood-ratio membership attack lands within 0.003 of its TV ceiling, within two standard errors for 200,000 games per world, and the small differences in either direction are sampling error. Second, access is not the only lever: in the centroid example the biggest gap is between two attacks on the same black-box view, not between access levels. Third, knowledge changes everything: the same noisy count has advantage 0.38 against an adversary who knows the other records and 0.08 against one who does not, and the exact count goes from 1 to 0.08. Fourth, one number is not enough: advantage, TPR at a low FPR, precision and accuracy against a baseline each answer a different question, and they can disagree by orders of magnitude.

## References

- Salem, Cherubin, Evans, Köpf, Paverd, Suri, Tople & Zanella-Béguelin, "SoK: Let the Privacy Games Begin! A Unified Treatment of Data Inference Privacy in Machine Learning" (IEEE S&P 2023). The source of the game-based view of this chapter: membership, attribute, reconstruction and property inference written as games, with the relations between them. Read §§3–5.
- Yeom, Giacomelli, Fredrikson & Jha, "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting" (IEEE CSF 2018). The membership and attribute games with advantage $\mathrm{TPR} - \mathrm{FPR}$, and the loss-threshold attack.
- Carlini, Chien, Nasr, Song, Terzis & Tramèr, "Membership Inference Attacks From First Principles" (IEEE S&P 2022). Why attacks should be reported by TPR at low FPR on log-scaled ROC curves; the attack of [Membership inference](02-membership-inference.md).
- Sablayrolles, Douze, Schmid, Ollivier & Jégou, "White-box vs Black-box: Bayes Optimal Strategies for Membership Inference" (ICML 2019). Conditions under which the optimal attack depends only on the loss, so white-box access does not help.
- Jayaraman, Wang, Knipmeyer, Gu & Evans, "Revisiting Membership Inference Under Realistic Assumptions" (PETS 2021). Membership inference when members are rare: priors, precision and threshold choice.
- Choquette-Choo, Tramèr, Carlini & Papernot, "Label-Only Membership Inference Attacks" (ICML 2021). Attacks that see only predicted labels, and how close they come to score-based ones.
- Nasr, Shokri & Houmansadr, "Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning" (IEEE S&P 2019). A taxonomy of white-box and federated adversaries, honest-but-curious and active.
- Boneh & Shoup, *A Graduate Course in Applied Cryptography*, ch. 2–3: security games, advantage and the challenger–adversary vocabulary this chapter borrows.

## Cheat sheet

$P$ and $Q$ are the adversary's view without and with the target ($b = 0$ and $b = 1$); $\pi_1$ is the prior of $b = 1$. @tbl-threat-cheat collects the results.

| Result | Statement | Where |
| --- | --- | --- |
| Winning and advantage | $\Pr[b' = b] = \pi_1\mathrm{TPR} + \pi_0(1 - \mathrm{FPR})$; fair prior: $\frac12 + \frac12\mathrm{Adv}$, $\mathrm{Adv} = \mathrm{TPR} - \mathrm{FPR}$ | @eq-threat-win, @eq-threat-adv |
| Access chain | labels $\le$ scores $\le$ parameters $\le$ transcript, for the best attack | @eq-threat-access-chain |
| Ceiling on advantage | $\mathrm{Adv} \le \mathrm{TV}(P, Q)$ | @eq-threat-tv |
| Ceiling on the ROC curve | $\mathrm{TPR} \le 1 - T(P, Q)(\mathrm{FPR})$ | @eq-threat-tradeoff |
| Bayes-optimal guess | $\max\Pr[b' = b] = \pi_0 + \pi_1E_{\pi_0/\pi_1}(Q\|P)$ for $\pi_1 \le \frac12$ | @eq-threat-bayes |
| Precision | $\pi_1\mathrm{TPR}/(\pi_1\mathrm{TPR} + \pi_0\mathrm{FPR})$ | @eq-threat-precision |
| Average against worst case | $\mathrm{TV}(\bar P, \bar Q) \le \mathbb E_\omega\mathrm{TV}(P_\omega, Q_\omega) \le \max_\omega\mathrm{TV}(P_\omega, Q_\omega)$ | @eq-threat-avg-worst |
| Noisy count, shift 1, noise $\sigma$ | Rest known: $2\Phi(\frac1{2\sigma}) - 1$. $m$ unknown others: $\approx 2\Phi\big(1/(2\sqrt{m/4 + \sigma^2})\big) - 1$ | @tbl-threat-avg-worst |

: The results of this chapter. {#tbl-threat-cheat}

**Rules of thumb**

- Before reading an attack's number, read its threat model: access, behaviour, trust, side information, and the game.
- Compare every attack with its baseline. Accuracy near the prior is no attack.
- Report TPR at a low FPR, with the ROC curve on log axes; advantage and accuracy hide the cases that matter.
- An average-case number is a lower bound on the worst case. A defence has to hold for the worst record, the worst dataset and the best-informed adversary.
- A better attack on the same view can matter more than more access: calibrate scores per target.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part02-attacks/01-threat-models.md).
