# Adaptive composition: filters, odometers and individual accounting

## Why this matters

A composition theorem tells you how much privacy a sequence of steps loses in total, given what each step loses. Every composition theorem in the earlier chapters assumes that the privacy parameters of the steps are fixed before the run starts ([Advanced and optimal composition](../part03-dp-foundations/04-composition.md)). The mechanisms may be chosen after seeing earlier outputs, but the budget of each step and the number of steps must be known in advance. Real systems often break this assumption. If you train until the validation loss stops improving, the number of steps depends on the outputs. If you lower the noise when training stalls, so does each step's budget.

The third case comes from training with differential privacy (DP). This usually means stochastic gradient descent (SGD) with clipped per-record gradients and Gaussian noise, called DP-SGD. A record whose clipped gradient is small moves the noisy sum less than the clipping allows. The noise hides a smaller move better, so the record's privacy cost is below the worst case, by an amount that depends on the earlier outputs. It is tempting to wait until the run ends, plug the values that occurred into a fixed-parameter theorem and report the result. That does not work. In the next section, an analyst who just chooses when to stop makes such a report, at failure probability 0.01, wrong over three times as often as it claims.

Once the parameters adapt, the question "what is the privacy parameter of this run?" splits in two. A **privacy filter** answers "may I run one more step without exceeding a budget fixed in advance?" A **privacy odometer** answers "how much have I spent so far?", and since you may stop at any moment, its answer must hold at every step at once. Both rest on one fact about games that are fair or tilted against you: no rule for when to quit can raise your expected final wealth. The privacy loss of a run can be tracked as such a game. Rényi differential privacy (RDP) measures each step by a curve indexed by an order, and at an order fixed before the run, this fact makes adaptive spending almost free. A filter as good as advanced composition, a cheap odometer and per-record accounting in DP-SGD all follow from it.

## Why plugging in the realized parameters fails {#sec-adapt-why-breaks}

Start with what is being bounded. Two datasets are **neighbours** if they differ in one person's record. The **privacy loss** of an output is the log of how much likelier that output is on one neighbour than on the other. It is positive when the output favours the first dataset and negative when it favours the second. A run releases one output per step, and each step's loss is computed given the outputs before it. The loss of the whole run is the sum of the step losses, and $L_{1:t}$ denotes that total after $t$ steps. A guarantee with the failure probability $\delta$ says that this total stays below some bound except with probability $\delta$.

Advanced composition is such a guarantee. Take a run of $k$ steps, each satisfying DP with the same privacy parameter $\varepsilon$, and with both numbers fixed in advance. Advanced composition says that the total loss exceeds
$$k\varepsilon\tanh(\varepsilon/2) + \varepsilon\sqrt{2k\ln(1/\delta)}$$
with probability at most $\delta$. The first term is the mean of the total loss. The second is the typical spread of a sum of $k$ bounded losses, widened by a factor that pays for the failure probability. A tempting shortcut for an adaptive run is to apply this bound **ex post**: stop whenever you like, then report advanced composition of the parameters you actually used. Rogers, Roth, Ullman & Vadhan (2016) showed that this shortcut is not valid.

Rogers et al. also gave the two objects of this chapter precise definitions. Both face an **analyst**, who chooses each step and its parameter from the outputs so far. A filter has a global budget, written $(\varepsilon_g, \delta)$ and fixed before the run. Before each step, including the parameter of that step, it says *continue* or *halt*. It is valid if the stopped run is $(\varepsilon_g, \delta)$-DP for every analyst. An odometer is instead a running bound $U_t$, computed from the parameters of the first $t$ steps. It is valid if, for every analyst and every pair of neighbours, the probability that the loss ever exceeds the running bound is at most $\delta$:
$$\Pr\big[\exists\, t: L_{1:t} > U_t\big] \le \delta.$$ {#eq-adapt-odometer-def}
The "there is a step $t$" is the point. You may stop at any time, so the bound you report when you stop is safe only if it holds at all times at once.

Here is an analyst who breaks the plug-in bound. Randomized response reports a person's bit truthfully with probability $e^\varepsilon/(1 + e^\varepsilon)$ and flips it otherwise. In this example every step runs it with $\varepsilon = 0.1$ on the same person's bit. Take that person's bit to be 1 in the first dataset and 0 in the second. A report of 1 is then $e^\varepsilon$ times likelier on the first dataset, so its loss is $+\varepsilon$, and a report of 0 has loss $-\varepsilon$. The total loss is therefore a random walk that moves up or down by 0.1 at each step. On the first dataset a report of 1 is slightly likelier, so the walk drifts slowly upward, by $\varepsilon\tanh(\varepsilon/2)$ per step on average.

The candidate odometer is the plug-in bound: $U_t$ is advanced composition with $k = t$. At each fixed step it is a correct bound. Azuma's inequality ([Concentration inequalities](../part01-prerequisites/02-concentration.md)) bounds how far a sum can drift when each of its steps averages zero given the past and lies in a short interval. Applied to the walk with its drift removed, it gives exactly the spread term. The exact probabilities are smaller still, because Azuma's inequality assumes the largest variance a step in that interval could have. The last column of @tbl-adapt-plugin shows how much smaller. At $\delta = 0.05$, the exact probability of exceeding the bound at any one fixed step in its range is at most 0.0110.

Now let the analyst stop the first time the loss crosses the bound. That is a legitimate stopping rule, because the loss counts the reports of 1 minus the reports of 0, in units of $\varepsilon$, and the analyst sees those reports. At the stopping step $T$, the report "the loss is at most $U_T$" fails exactly when the walk has crossed the bound at some step. So its failure probability is a probability over the whole path, which @tbl-adapt-plugin gives exactly.

| Horizon $n$ | 10 | 100 | 1000 | 10,000 | Worst single fixed $t \le 1000$ |
| --- | --- | --- | --- | --- | --- |
| $\delta = 0.05$ | 0.0110 | 0.0575 | 0.1004 | 0.1424 | 0.0110 ($t = 17$) |
| $\delta = 0.01$ | 0.0000 | 0.0090 | 0.0202 | 0.0318 | 0.0016 ($t = 21$) |
| $\delta = 10^{-5}$ | 0 | $3.12\times10^{-6}$ | $1.80\times10^{-5}$ | $3.69\times10^{-5}$ | $8.18\times10^{-7}$ ($t = 758$) |

: Exact probability that randomized response with $\varepsilon = 0.1$ crosses the plug-in advanced-composition bound by step $n$, computed by following the distribution of the walk step by step. The last column is the largest probability of exceeding the bound at a single fixed step, over every $t \le 1000$, and the step where it occurs. {#tbl-adapt-plugin}

Take $\delta = 0.01$, where the bound has been crossed with probability 0.0318 by the longest horizon. So a report of "$\varepsilon = U_T$" at this $\delta$ is wrong 3.18 times as often as it claims. With the larger $\delta$ in the first row, it reaches 0.1424, which is 2.85 times $\delta$. The failures come early. The bound is narrow while $\sqrt t$ is small, so a few walks cross it then and the analyst freezes them there, while most walks never come near it (@fig-adapt-plugin-failure).

![Left: forty runs of randomized response with $\varepsilon = 0.1$, the loss in units of $\varepsilon$, and the plug-in advanced-composition bound at $\delta = 0.01$. An analyst who stops at the first crossing freezes the red paths there. Right: the exact probability that the bound is crossed by step $n$, for $\delta = 0.05$ and $0.01$ (dashed lines); it passes $\delta$ and keeps growing.](figures/adaptive_composition_plugin_failure.svg){#fig-adapt-plugin-failure}

The crossing probability keeps growing with the horizon, and in fact it tends to 1. The plug-in bound sits a fixed multiple of $\sqrt t$ above the mean of the walk. But the law of the iterated logarithm, which the full edition cites, says that the walk infinitely often reaches beyond a curve that grows like $\sqrt{t\ln\ln t}$. Because $\ln\ln t$ keeps growing, that curve eventually overtakes the bound, so given enough steps the walk crosses it with probability 1.

Every valid odometer must pay for this. Rogers et al. (2016, Theorem 6.1) turned the same stop-when-ahead construction into a lower bound for any odometer over $n$ steps. Where a fixed-time bound has $\ln(1/\delta)$ inside its square root, a valid odometer needs, up to smaller terms, the larger $\ln(\ln n/\delta)$. That is the same as adding $\ln\ln n$ to $\ln(1/\delta)$. When $\delta$ is far below $1/n$, the addition is small, because the original term is then larger than $\ln n$, which dwarfs the added one. Even so, a small $\delta$ does not rescue the plug-in bound. In the last row of @tbl-adapt-plugin it is crossed with a probability 3.69 times $\delta$.

The theorem was never wrong; its use was. Advanced composition bounds the loss at a time fixed in advance, and a time chosen from the data is a different random variable. Nor is the analyst far-fetched, since any analyst can stop on a statistic correlated with the loss, such as "the answers look unusual". The definitions cover every analyst, so a bound must survive the worst one.

Two kinds of guarantee escape the problem. Basic composition, the plain sum of the realized per-step parameters, bounds the loss of every output, so there is no bad event for a stopping rule to select. It is a valid filter and a valid odometer for pure DP. A filter also escapes, because it fixes its budget in advance and makes no running claim that the analyst could catch out at a bad moment. Whether the next step fits turns out to be answerable with no $\ln\ln$ loss (@sec-adapt-dp-filters).

## Stopping a fair game {#sec-adapt-stopping}

The rest of the chapter reuses three facts about stopping a random process. The first says how to compute the loss of a run that stops at a random step. Write $T$ again for the step at which the run stops, and suppose the decision to stop is made from the outputs alone. Then the loss of the stopped run is the sum of the losses of the $T$ steps it ran, exactly as if $T$ had been fixed. The probability of a transcript is the product of the step-by-step probabilities, and given the outputs, the decision to stop has the same probability on both neighbours. So it cancels from their ratio and adds nothing to the loss. A rule that looked at the raw data would not cancel, and the stopping time itself would leak.

The second fact is about betting games. Think of a quantity $N_t$ as your wealth after $t$ bets, in a game that is fair or tilted against you. Such a process is a **supermartingale**: given everything seen so far, its expected next value is at most its current value. It is a **martingale** if the expected next value is exactly the current one, as in a fair game. You may decide when to quit by looking at everything that has happened, but not at the result of the bet you are about to make. A rule of that kind is a **stopping time**.

**Optional stopping** says that stopping a supermartingale at a stopping time cannot raise its mean above its starting value. With $T$ now the quitting time,
$$\mathbb E[N_T] \le \mathbb E[N_0].$$ {#eq-adapt-optional-stopping}
The reason is that each step the stopped process takes adds an increment whose mean, given the past, is at most zero. Whether to take that step was decided before its result was known, so the decision cannot favour the steps that turn out well.

The fact needs two conditions. The stopping decision must not look at the step it is about to take, because otherwise you could skip exactly the losing bets. And either the run has a fixed largest number of steps or the process is non-negative. Without either, even a fair coin game can be won. With unlimited time and credit, "quit when you are one ahead" wins with probability 1, at the price of arbitrarily deep debt on the way. The processes in this chapter are non-negative, so they cannot run up that debt, and the horizon can be dropped.

The third fact turns optional stopping into a bound on the whole path. Markov's inequality says that a non-negative quantity with mean 1 is at least a threshold $c$ with probability at most $1/c$. Applied at one fixed time, it bounds the chance that the process is large at that time. **Ville's inequality** says that for a non-negative supermartingale starting at 1, the same bound holds for the chance that it is ever large:
$$\Pr\big[\exists\, t: N_t \ge c\big] \le \frac1c.$$ {#eq-adapt-ville}

The reason is optional stopping again. "Stop the first time the process reaches $c$" is a legitimate stopping rule, so the stopped value still has mean at most 1. That stopped value is at least $c$ on every path that ever got there, and by Markov's argument those paths have total probability at most $1/c$. For example, in a fair game that starts with wealth 1, the chance that your wealth ever reaches 10 is at most a tenth, however long you play.

## The Rényi filter {#sec-adapt-renyi-filter}

[Rényi DP](01-renyi-dp.md) describes a mechanism by a curve. The curve is indexed by an order $\alpha$, a number above 1 that you choose. At each order it gives a bound $\varepsilon(\alpha)$ on the Rényi divergence of that order between the outputs on two neighbours. That divergence is $1/(\alpha - 1)$ times the logarithm of the average of $e^{(\alpha - 1)L}$, where $L$ is the privacy loss of an output drawn on the first neighbour. So the curve controls an exponential average of the loss. With parameters fixed in advance, composition adds the curves of the steps. But what if a step's curve depends on earlier outputs? Feldman & Zrnic (2021) and Lécuyer (2021) showed that nothing is lost at a fixed order.

Fix an order $\alpha$ and an RDP budget $B$ before the run. Let step $t$ be $(\alpha, \varepsilon_t(\alpha))$-RDP whatever outputs came before it. Its **spend** $\varepsilon_t(\alpha)$ may depend on those earlier outputs, but it must be known before the step runs, and a parameter of this kind is called **predictable**. The **Rényi filter** runs step $t$ only if the spends so far fit in the budget, with the spend of step $t$ itself included:
$$\sum_{j \le t}\varepsilon_j(\alpha) \le B.$$ {#eq-adapt-renyi-filter}
Then the stopped run is $(\alpha, B)$-RDP, exactly as if the spends had been fixed in advance. Put the other way round, if the spends add up to at most $B$ on every path, the adaptive composition is $(\alpha, B)$-RDP.

The reason is a betting game. Fix two neighbours, and define a wealth $N_t$ that starts at $N_0 = 1$ and compares the loss so far with the spend so far:
$$N_t = \exp\Big((\alpha - 1)\Big(L_{1:t} - \sum_{j\le t}\varepsilon_j(\alpha)\Big)\Big).$$ {#eq-adapt-supermartingale}
Each step multiplies the wealth by $e^{(\alpha - 1)L_t}$, where $L_t$ is that step's loss, and divides it by $e^{(\alpha - 1)\varepsilon_t(\alpha)}$. Given the past, the step's RDP bounds the average of the first factor by the second, so on average a step does not raise the wealth. The wealth is therefore a supermartingale, and the filter's rule is a legitimate stopping rule. By optional stopping, the wealth at the end has mean at most 1. Undoing the definition, the exponential of the final loss is the final wealth times $e^{(\alpha - 1)\times\text{spend}}$. The spend is at most $B$ on every path, so the exponential average of the stopped run's loss is at most $e^{(\alpha - 1)B}$. That is exactly what $(\alpha, B)$-RDP says.

Each condition of the result is a step of that argument, and each fails in a specific way without it. The spend must be predictable. A spend that could depend on its own step's output could be small exactly on the outputs that reveal the most. The wealth would then no longer be a supermartingale. The budget must hold on every path, not merely on average. Paths with a large spend may also have a large wealth. Then the average of the wealth times $e^{(\alpha - 1)\times\text{spend}}$ is not bounded by the product of their averages. And the order must be fixed before the run, because the argument works at one order.

You may track several orders and stop when any one of them runs out. Then on every path the budget held at each tracked order, so each of those orders is certified. What you cannot do for free is keep running as long as *some* order still fits, with the order chosen after seeing the spends. @sec-adapt-odometers prices that choice.

A small example from the full edition shows what the filter gains over fixed curves. Take order 2. There the randomized-response curve of [Rényi DP](01-renyi-dp.md) charges 0.8473 to a strong step, at privacy parameter $\ln 3$. A weak step, at $\ln 1.5$, is charged 0.1542. The first step is strong. If it outputs 1, the second step is strong and the third weak, and if it outputs 0 the second is weak and the third strong. Every path therefore spends 1.8487. The Rényi divergence of order 2 between the transcript's distributions on the two datasets is exactly that, so the filter's bound is attained. Composition with fixed curves must charge both later steps as strong and gives 2.5419. In general the filter's bound is the largest spend on any path, which is never more than fixed-parameter composition gives, and less when branches trade costs.

The bound is attained whenever every path spends exactly $B$ and each step's RDP parameter is its exact divergence, because the wealth is then a martingale. Gaussian steps and randomized response have exact RDP parameters. For example, @fig-adapt-supermartingale shows adaptive Gaussian steps stopped by a Rényi filter at order 2 with budget 1. Each step shifts the output by 0.3 standard deviations of the noise after a positive output, and by 0.6 otherwise. The wealth of each path wanders, but its average stays at 1.

![Adaptive Gaussian steps under a Rényi filter with $\alpha = 2$ and $B = 1$. Left: $\ln N_t$ along thirty paths; each path freezes when the filter stops it. Right: the Monte Carlo mean of $N_t$ over $2\times10^5$ paths stays at its starting value 1, as optional stopping says.](figures/adaptive_composition_supermartingale.svg){#fig-adapt-supermartingale}

At the end of a run you convert to $(\varepsilon, \delta)$-DP. Mironov's conversion ([RDP conversions](02-rdp-conversions.md)) turns an $(\alpha, B)$-RDP guarantee into an $(\varepsilon, \delta)$-DP one for every $\delta$. The resulting $\varepsilon$ is the budget plus a conversion term,
$$\varepsilon = B + \frac{\ln(1/\delta)}{\alpha - 1}.$$
It holds because the exponential average limits the chance that the loss exceeds this threshold to $\delta$, as in Markov's inequality. Outputs below the threshold are at most $e^\varepsilon$ times likelier on one neighbour than on the other, which is the ratio DP requires. Run backwards, a DP target $(\varepsilon_g, \delta)$ fixes the budget at the chosen order:
$$B = \varepsilon_g - \frac{\ln(1/\delta)}{\alpha - 1}.$$

Suppose you train with full-batch noisy gradient descent towards $(5.299, 10^{-5})$ at order 5.8. That target is the guarantee of 100 steps at a fixed noise level, which the examples below reuse, and the formula sets the budget to $B = 2.900$. Write $\sigma_t$ for the noise multiplier of step $t$, the noise standard deviation in units of the sensitivity. A full-batch Gaussian step then spends $\alpha/(2\sigma_t^2)$ at order $\alpha$, since the Gaussian divergence goes with the square of the shift in units of the noise. With $\sigma_t = 10$ throughout, each step spends 0.0290, so the budget lasts exactly the planned number of steps. Now suppose training plateaus at step 60 and you switch to $\sigma_t = 5$. Each later step costs 0.1160, so the filter allows 10 more steps. Either way the run is $(5.8, 2.900)$-RDP, whatever rule decided the switch, as long as that rule looks only at outputs already released.

## Filters for pure differential privacy {#sec-adapt-dp-filters}

Filters were first built for interactive query systems with pure-DP steps. Such a system has a global budget $(\varepsilon_g, \delta)$. Each query brings its own parameter $\varepsilon_t$, which may be chosen from the earlier answers but must be known before the query runs. A rule must then say, before each query, whether it may run. The simplest rule is the **basic filter** of Rogers et al. (2016, Theorem 3.4). It continues while the sum of the $\varepsilon_t$, the next one included, is at most $\varepsilon_g$, and the stopped run is then $(\varepsilon_g, 0)$-DP. The reason is that pure DP bounds the loss of every single output by $\varepsilon_t$, not just its typical size. The loss of the stopped run is the sum of its step losses, so it is at most $\varepsilon_g$ on every path. That bound on every output is exactly what pure DP requires.

The basic filter is exact, but the budget it needs grows like the number of steps, while advanced composition grows only like its square root. The question is how to keep that slower growth when the $\varepsilon_t$ adapt, and the Rényi filter answers it once the steps are measured in zero-concentrated differential privacy (zCDP). A mechanism is $\rho$-zCDP, for a slope $\rho$, if its RDP curve lies below the line $\alpha\rho$ at every order ([Concentrated differential privacy](03-concentrated-dp.md)). An $\varepsilon_t$-DP step is $(\varepsilon_t^2/2)$-zCDP, because its loss is confined to an interval of length $2\varepsilon_t$ and a bounded loss has light tails. So it spends at most $\alpha\varepsilon_t^2/2$ at every order.

Now consider the rule "continue while half the sum of the $\varepsilon_t^2$ is at most a level $\rho^*$", fixed in advance. Then at any fixed order, multiplying both sides by $\alpha$ turns this rule into a Rényi filter with budget $\alpha\rho^*$ on the spends $\alpha\varepsilon_t^2/2$. Because the rule does not mention the order, it is a Rényi filter at every order at once, so the stopped run is $\rho^*$-zCDP. And because one run is certified at every order, the conversion may pick the best order afterwards. For zCDP, that conversion gives
$$\varepsilon = \rho^* + 2\sqrt{\rho^*\ln(1/\delta)},$$
and the level that makes it equal to $\varepsilon_g$ is
$$\rho^* = \Big(\sqrt{\ln(1/\delta) + \varepsilon_g} - \sqrt{\ln(1/\delta)}\Big)^2.$$
Rewriting the rule in terms of $\varepsilon_g$ gives the **Whitehouse filter** of Whitehouse, Ramdas, Rogers & Wu (2023). It continues while
$$\sqrt{2\ln(1/\delta)\textstyle\sum_t\varepsilon_t^2} + \tfrac12\textstyle\sum_t\varepsilon_t^2 \le \varepsilon_g,$$ {#eq-adapt-whitehouse}
with the sums over every step so far, the one about to run included, and its stopped run is $(\varepsilon_g, \delta)$-DP.

The rule reads like advanced composition, with the spread of the total loss under the square root and its mean as the second term. It is the $\tanh$ form of @sec-adapt-why-breaks with the mean term per step raised slightly, to $\varepsilon_t^2/2$. It is also exactly what composing in zCDP gives for parameters fixed in advance, so adaptivity costs essentially nothing.

The original form of advanced composition, due to Dwork, Rothblum & Vadhan, has a larger mean term, $\varepsilon_t(e^{\varepsilon_t} - 1)$ per step. That term is at least $\varepsilon_t^2$, so it lies above the Whitehouse filter's $\varepsilon_t^2/2$. A filter that halts when this form reaches $\varepsilon_g$ therefore halts no later than the Whitehouse filter, and so it is valid too. Halting earlier only discards the rest of the transcript, which is post-processing of the longer run. The argument does not cover the $\tanh$ form, whose statistic is slightly below the Whitehouse filter's.

The first filter whose budget grows like the square root of the number of steps is that of Rogers et al. (2016), valid for $\delta < 1/e$. Its spread term uses the sum of the $\varepsilon_t^2$ as a variance, but that sum is now random, while Azuma's inequality needs it fixed in advance. So they use an inequality that holds whatever the variance turns out to be, and pay for it with a padded variance and an extra logarithmic factor.

@tbl-adapt-budgets measures the difference. It asks what global budget each rule needs to let $k$ steps through, each with the same parameter $\varepsilon_0$, at $\delta = 10^{-6}$. Its advanced-composition column has $k$ fixed in advance and uses the Dwork–Rothblum–Vadhan form, not the $\tanh$ form. Take 1,000 steps of 0.01 each. The Whitehouse filter needs a budget of 1.712 for them, and advanced composition in that form needs 1.763. The Whitehouse filter wins, despite allowing adaptivity, because its mean term $\varepsilon_t^2/2$ is smaller than its mean term $\varepsilon_t(e^{\varepsilon_t} - 1)$. The Rogers filter needs 1.40 to 1.51 times the budget of the Whitehouse filter.

| $\varepsilon_0$ | $k$ | Basic | Advanced (fixed $k$, Dwork–Rothblum–Vadhan) | Rogers filter | Whitehouse filter | Rogers / Whitehouse |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 100 | 1.000 | 0.536 | 0.803 | 0.531 | 1.51 |
| 0.01 | 1000 | 10.000 | 1.763 | 2.570 | 1.712 | 1.50 |
| 0.01 | 10,000 | 100.000 | 6.262 | 8.450 | 5.757 | 1.47 |
| 0.1 | 100 | 10.000 | 6.308 | 8.472 | 5.757 | 1.47 |
| 0.1 | 1000 | 100.000 | 27.140 | 30.244 | 21.623 | 1.40 |

: Global budget $\varepsilon_g$ needed for $k$ steps of $\varepsilon_0$ at $\delta = 10^{-6}$. Advanced composition is in the Dwork–Rothblum–Vadhan form, with mean term $\varepsilon_0(e^{\varepsilon_0} - 1)$ per step. For the Rogers filter, the smallest $\varepsilon_g$ at which its rule holds. {#tbl-adapt-budgets}

![Global budget needed to run $k$ steps of $\varepsilon_0 = 0.01$ at $\delta = 10^{-6}$, on log–log axes: basic composition, advanced composition with $k$ fixed in advance, and the Rogers et al. and Whitehouse et al. filters. The Whitehouse filter lies on or below fixed-$k$ advanced composition; the Rogers filter sits a constant factor above.](figures/adaptive_composition_filter_budgets.svg){#fig-adapt-filter-budgets}

You can also turn the question around and fix the budget, as @tbl-adapt-steps does for steps of 0.01 and 0.001. With a global budget of 1, the Whitehouse filter allows 349 steps of this size. Fixed-$k$ advanced composition in the Dwork–Rothblum–Vadhan form allows 337, and the Rogers filter allows 154, fewer than half as many as the Whitehouse filter. So use the Whitehouse form, and check it *before* each step, with that step's parameter included, since the guarantee is about the decision to run it.

| $\varepsilon_0$ | Basic | Advanced (fixed $k$, Dwork–Rothblum–Vadhan) | Rogers filter | Whitehouse filter |
| --- | --- | --- | --- | --- |
| 0.01 | 100 | 337 | 154 | 349 |
| 0.001 | 1000 | 33,785 | 15,481 | 34,937 |

: Number of steps of $\varepsilon_0$ allowed by a global budget $\varepsilon_g = 1$ at $\delta = 10^{-6}$. {#tbl-adapt-steps}

All of this is for pure DP. For steps with their own failure probability $\delta_t > 0$, Whitehouse et al. add a second rule, which halts once the sum of the $\delta_t$ would pass a second budget $\delta''$. The guarantee then becomes $(\varepsilon_g, \delta + \delta'')$-DP.

## Odometers {#sec-adapt-odometers}

Early stopping, training that runs until it reaches a target accuracy, and open-ended analyses have no budget to stop at. They need a running answer to "how much have I spent?" that stays valid however the stopping time is chosen, which is the odometer defined in @sec-adapt-why-breaks. The wealth of the Rényi filter gives one. It is a non-negative supermartingale that starts at 1, whether or not a filter stops it, because the argument used only that the spends are predictable. Ville's inequality then says that the wealth rarely ever gets large, and "the wealth is large" unpacks to "the loss is far above the spend".

Concretely, fix an order $\alpha$ and write $S_t(\alpha)$ for the spend so far at that order. Ville's inequality with the threshold $1/\delta$ says that, except with probability $\delta$, at every step at once,
$$L_{1:t} \le S_t(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}.$$
This is Mironov's fixed-time conversion, now valid at every step at once, so at one order fixed in advance a running bound costs nothing extra.

The cost comes from the order. The best order for 10 steps is not the best for 1,000, so you want to choose it late, after seeing the spends. To do that, fix a grid $\Lambda$ of orders before the run, and write $|\Lambda|$ for their number. Give each order a share $\delta/|\Lambda|$ of the failure probability. By the union bound, the chance that the bound fails at some order is at most the sum of these shares, which is $\delta$. Outside that failure event the bound holds at every order and every step, so it holds at the best order, chosen afterwards. The result is the **Ville odometer**: with probability at least $1 - \delta$, simultaneously for all $t$,
$$L_{1:t} \le U_t = \min_{\alpha\in\Lambda}\Big[S_t(\alpha) + \frac{\ln(|\Lambda|/\delta)}{\alpha - 1}\Big].$$ {#eq-adapt-ville-odometer}
It is Mironov's conversion at the best order of the grid, with $\delta$ replaced by $\delta/|\Lambda|$, and that replacement is the whole price.

A small grid is cheaper, although the price grows only with the logarithm of its size. At $\delta = 10^{-5}$ the numerator of the conversion term is 11.513. This part's accountants use a grid of 154 orders, which raises it to 16.550. As a heuristic, the full edition suggests a few orders per factor of two in spend. The price is then a $\ln\ln$ of the range of spends, the same shape as the $\ln\ln n$ in the lower bound of Rogers et al.

At a single order, the Ville odometer is tight for Gaussian steps. Their total loss is a random walk with a steady drift. In continuous time, the full edition shows, such a walk crosses the Ville bound with probability exactly $\delta$. For example, at $\delta = 0.05$ its simulation gives 0.0461, slightly less because checking at whole steps misses crossings between them. At any single fixed time, the same bound is crossed with probability at most 0.0072. So the slack that Mironov's conversion has at a fixed time is what pays for holding at all times.

Lécuyer (2021) builds a different odometer from a sequence of Rényi filters whose budgets double. It reports the budget of the first filter that has not yet run out, plus a conversion term at a confidence that shrinks with the filter's index. Rounding the spend up to a doubled budget costs at most a factor of 2, and in the examples below it is slightly looser than the Ville odometer.

For pure-DP steps, Rogers et al. (2016) give an odometer shaped like advanced composition that pays the $\ln\ln$ price in its square-root term. Whitehouse et al. (2023) give tighter ones.

@tbl-adapt-odometers compares the running bounds for full-batch Gaussian steps with noise multiplier $\sigma = 10$, at $\delta = 10^{-5}$. Its fixed-time column is valid only if the number of steps was chosen in advance. Over the part's grid, the Ville odometer costs 15 to 19% more than the fixed-time value. A Ville odometer at the single order 5.8, chosen in advance for 100 steps, is free at that design point but poor elsewhere. After 1,000 steps it gives 31.399, because its order is far from the best one there.

| Steps $t$ | Fixed-time conversion | Ville odometer (154 orders) | Lécuyer odometer | Ville at $\alpha = 5.8$ only |
| --- | --- | --- | --- | --- |
| 10 | 1.568 | 1.869 | 1.916 | 2.689 |
| 100 | 5.299 | 6.253 | 6.507 | 5.299 |
| 300 | 9.812 | 11.465 | 11.892 | 11.099 |
| 1000 | 20.175 | 23.194 | 24.633 | 31.399 |

: Running $\varepsilon$ at $\delta = 10^{-5}$ for $t$ full-batch Gaussian steps with $\sigma = 10$. Only the last three columns are valid when $t$ is chosen from the outputs. {#tbl-adapt-odometers}

![Running privacy bounds at $\delta = 10^{-5}$ for full-batch Gaussian steps with $\sigma = 10$: Mironov's conversion at a time fixed in advance (dashed, not a valid odometer), the Ville odometer over the part's 154 orders, Lécuyer's odometer with doubling budgets, and a Ville odometer at the single order 5.8.](figures/adaptive_composition_odometers.svg){#fig-adapt-odometers}

The same pattern holds for this part's DP-SGD example, which runs $10^4$ subsampled steps ([Amplification by subsampling](05-subsampling.md)). There Mironov's fixed-time conversion gives 7.43. An odometer valid whenever the run stops costs 8.82 with the Ville construction, or 9.08 with Lécuyer's.

An odometer is a tail bound, not a DP guarantee. Reporting "$\varepsilon = U_T$" means that the loss of this run exceeds a random value $U_T$ with probability at most $\delta$. To get an ordinary $(\varepsilon_g, \delta)$-DP guarantee, stop by a rule that keeps $U_T \le \varepsilon_g$. The loss then exceeds $\varepsilon_g$ with probability at most $\delta$. Outputs below that level satisfy the ratio DP requires, so the run is $(\varepsilon_g, \delta)$-DP. Because $U_T$ is a minimum over the grid, this rule continues while *some* order still fits, and its $\ln|\Lambda|$ is the price of choosing the order late. If you have a budget anyway, a Rényi filter at one order fixed in advance needs no union bound and is cheaper.

## Individual privacy accounting in DP-SGD {#sec-adapt-individual}

A step of DP-SGD computes each record's gradient and **clips** it, scaling it down if needed so that its norm is at most the clipping norm $C$. It then adds up the clipped gradients, adds Gaussian noise whose standard deviation is $\sigma C$ for a noise multiplier $\sigma$, and releases the sum. The usual analysis charges every record the worst case at every step, as if its clipped gradient always had norm $C$. Most records do not. Once the model fits a record, its gradient shrinks, and it moves the released sum less. Feldman & Zrnic (2021) observed that a record's actual privacy cost is then a random but predictable quantity, so the Rényi filter can be applied record by record. This section treats full-batch steps, which use every record.

With full-batch steps, removing a record $i$ shifts the mean of the released sum by that record's clipped gradient, written $\bar g_{i,t}$ at step $t$. The noise is the same in every direction, so only the length of that shift matters, measured in units of the noise. The Gaussian RDP for that shift gives record $i$'s spend at order $\alpha$:
$$\varepsilon_{i,t}(\alpha) = \frac{\alpha\|\bar g_{i,t}\|^2}{2\sigma^2C^2} \le \frac{\alpha}{2\sigma^2}.$$ {#eq-adapt-individual-rdp}
The bound on the right is the worst case, reached when the clipped gradient has norm $C$.

This spend is predictable. The clipped gradient depends only on record $i$ and the current model, and the model depends only on the earlier releases. That condition matters for the filter below, which drops a record once its own spend runs out. If a record's spend depended on the whole dataset, removing record $i$ would change which other records get dropped. The outputs with and without $i$ would then differ even at steps where $i$ itself is dropped, and the argument would fail. For a sum of clipped gradients this cannot happen.

The **individual filter** (Feldman & Zrnic 2021) runs one Rényi filter per record. Fix an order $\alpha$ and a budget $B$. At each step, record $i$ joins the sum only if its own spends so far, counting only the steps it took part in, stay at most $B$. Once dropped, the record no longer changes the output, so its spend for that step is zero. Two neighbouring datasets differ in one record, whose spends were predictable and never passed $B$. So the Rényi filter makes the run $(\alpha, B)$-RDP for that pair, and since this holds for every record, the whole run is $(\alpha, B)$-RDP.

The filter thus gives the same guarantee with more training. No step costs a record more than the worst case $\alpha/(2\sigma^2)$, so every record trains at least as long as in the worst-case run. Records whose gradients shrink spend less per step and keep taking part long after.

The full edition simulates 10,000 records with full-batch steps, a clipping norm of 1 and a noise multiplier of 10. Nine records in ten are typical records, whose gradients shrink as the model fits them. The other tenth are hard records, whose gradients never shrink because the model never fits them. The worst-case run of 100 steps has the guarantee $(5.299, 10^{-5})$ from the noise-schedule example above. Its RDP spend is best converted at order 5.8, where the spend is $B = 2.900$.

After those 100 steps, converting each record's own spend at order 5.8 gives an individual $\varepsilon$, which is an estimate, not a guarantee, for reasons given below. Because the conversion term is the same for every record, no individual $\varepsilon$ can fall below it, 2.399. @tbl-adapt-individual and @fig-adapt-individual-hist show the distribution, in which the median typical record has an individual $\varepsilon$ of 3.17, against 5.299 for the worst case. In all, 16.3% of records are below half the worst case. Hard records sit just under the worst case, at a median of 5.02, because their gradients are usually near or above the clipping norm. Choosing the order separately for each record lowers the typical median to 2.61. That is because a small spend is best converted at a larger order, where the conversion term is smaller.

| Records | 10th percentile | Median | 90th percentile | Median, order chosen per record |
| --- | --- | --- | --- | --- |
| All | 2.56 | 3.25 | 3.81 | 2.75 |
| Typical | 2.55 | 3.17 | 3.67 | 2.61 |
| Hard | 4.23 | 5.02 | 5.26 | 5.01 |

: Individual $\varepsilon$ at $\delta = 10^{-5}$ after 100 full-batch steps with $\sigma = 10$, at the fixed order 5.8 and, in the last column, with the order optimized per record over the part's grid. The worst case is 5.299. {#tbl-adapt-individual}

![Individual $\varepsilon$ at $\delta = 10^{-5}$ and order $\alpha^* = 5.8$ after 100 full-batch steps with $\sigma = 10$. Typical records (blue) spend far less than the worst case (dashed); hard records (red) pile up just below it. The dotted line is the floor $\ln(1/\delta)/(\alpha^* - 1)$ that a fixed order cannot go below.](figures/adaptive_composition_individual_hist.svg){#fig-adapt-individual-hist}

Now run the individual filter for 300 steps, with the same budget and order. Every record takes part in the first 100 steps, because no step costs more than the worst-case spend of 0.0290 and the budget covers that many such steps. After that, the hard records run out. The first is dropped at step 101 and the median one at step 111. By step 200 none takes part. A few return on steps where their gradient fits what is left of their budget, 0.1% of them at step 300. No typical record is ever dropped, and at step 300 their median spend is 0.778, about a quarter of the budget. Yet every record, typical or hard, has the same $(5.8, 2.900)$-RDP guarantee as the 100-step worst-case run (@fig-adapt-individual-spend).

![Left: cumulative spend at order 5.8 under the individual filter, for four typical records (blue) and three hard records (red), against the worst case (dotted) and the budget $B = 2.90$ (dashed). Hard records reach the budget and stop; typical records level off far below it. Right: the fraction of each group that takes part in the gradient sum at each step.](figures/adaptive_composition_individual_spend.svg){#fig-adapt-individual-spend}

Read the per-record numbers with care, for four reasons. First, the filter certifies one order and one budget for every record, and converted at that order it gives the run's $(\varepsilon, \delta)$. A per-record $\varepsilon$ computed from the realized spend is an estimate, not a guarantee. It is a fixed-time bound evaluated at a random quantity, the mistake of @sec-adapt-why-breaks. The same holds in an ordinary run of fixed length. Without a filter, a record's realized spend is random, and the Rényi filter needs a bound on every path. Yu et al. (2023) study these ex post, output-specific guarantees for DP-SGD.

Second, an individual $\varepsilon$ is itself sensitive. A record's spend is a function of its gradient norms, so a small spend says "the model fits this record well". Feldman & Zrnic note that it can be shown to the record's owner without hurting anyone else. It depends on the other records only through the released outputs, which are already private. Publishing all of them, however, leaks.

Third, there is a fairness cost. Records that the model fits poorly, often from under-represented groups, keep large gradients. They spend their budget first and are dropped first, so the model sees even less of them. Yu et al. (2023) find that the classes a model serves worst also get the weakest individual guarantees. This is one mechanism behind the uneven effect of DP training on accuracy across groups, which a later part discusses.

Fourth, the argument extends to Poisson-subsampled DP-SGD, where each record joins each step's batch with a sampling probability $q$. A record's spend is then the RDP of the subsampled Gaussian ([Amplification by subsampling](05-subsampling.md)) with noise multiplier $\sigma C/\|\bar g_{i,t}\|$, which is still predictable. But Feldman & Zrnic note that this needs every record's gradient norm at every step, sampled or not, since the spend formula uses it. Koskela, Tobaben & Honkela (2023) carry out this accounting, and tighten it with Gaussian differential privacy.

## Code {#sec-adapt-code}

A filter or an odometer is just a sum of spends and a test. This snippet codes the Whitehouse filter, the Rényi filter for a noise schedule and the Ville odometer for full-batch Gaussian steps. It prints the numbers quoted above.

```python
from math import log, sqrt


def whitehouse_statistic(eps_steps: list[float], delta: float) -> float:
    """Advanced composition of pure-DP budgets with mean term eps^2 / 2: valid under adaptivity."""
    v = sum(e * e for e in eps_steps)
    return sqrt(2 * log(1 / delta) * v) + v / 2


def steps_allowed(eps_step: float, eps_global: float, delta: float) -> int:
    """Run steps of eps_step while the Whitehouse filter, checked before each step, says continue."""
    spent: list[float] = []
    while whitehouse_statistic([*spent, eps_step], delta) <= eps_global:
        spent.append(eps_step)
    return len(spent)


def gaussian_rdp(alpha: float, sigma: float) -> float:
    """RDP at order alpha of one full-batch Gaussian step with noise multiplier sigma."""
    return alpha / (2 * sigma**2)


def renyi_filter(sigmas: list[float], alpha: float, budget: float) -> int:
    """Number of steps of a noise schedule that a Renyi filter at order alpha lets through."""
    spent, steps = 0.0, 0
    for sigma in sigmas:
        cost = gaussian_rdp(alpha, sigma)
        if spent + cost > budget + 1e-12:  # tolerance for floating-point sums
            break
        spent, steps = spent + cost, steps + 1
    return steps


ORDERS = [1 + x / 10 for x in range(1, 100)] + [float(x) for x in range(12, 64)] + [128.0, 256.0, 512.0]


def ville_odometer(t: int, sigma: float, delta: float, orders: list[float]) -> float:
    """Running epsilon after t full-batch Gaussian steps, valid whenever the run stops."""
    return min(t * gaussian_rdp(a, sigma) + log(len(orders) / delta) / (a - 1) for a in orders)


print(f"{whitehouse_statistic([0.01] * 1000, 1e-6):.3f}")  # 1.712
print(steps_allowed(0.01, 1.0, 1e-6))  # 349
alpha, budget = 5.8, 2.9
print(renyi_filter([10.0] * 1000, alpha, budget))  # 100
print(renyi_filter([10.0] * 60 + [5.0] * 1000, alpha, budget) - 60)  # 10
print(f"{budget + log(1 / 1e-5) / (alpha - 1):.3f}")  # 5.299
print(len(ORDERS))  # 154
print(f"{ville_odometer(100, 10.0, 1e-5, ORDERS):.3f}")  # 6.253
print(f"{ville_odometer(100, 10.0, 1e-5, [alpha]):.3f}")  # 5.299
print(f"{ville_odometer(1000, 10.0, 1e-5, [alpha]):.3f}")  # 31.399
```

`ORDERS` is the part's grid of orders. The full edition's script, [adaptive_composition.py](../../code/part04-privacy-accounting/adaptive_composition.py), does more. It finds the exact plug-in failure, checks the Rényi filter by simulation and runs the per-record experiment:

```bash
uv run python code/part04-privacy-accounting/adaptive_composition.py
```

## Cheat sheet

| Object | Rule or bound | Guarantee |
| --- | --- | --- |
| Basic filter and odometer (pure DP) | Continue while $\sum_t\varepsilon_t \le \varepsilon_g$; report $\sum_{j\le t}\varepsilon_j$ | $(\varepsilon_g, 0)$-DP; the bound holds on every path |
| Whitehouse filter (pure DP) | $\sqrt{2\ln(1/\delta)\sum_t\varepsilon_t^2} + \frac12\sum_t\varepsilon_t^2 \le \varepsilon_g$ | $(\varepsilon_g, \delta)$-DP |
| Rogers filter (pure DP, $\delta < 1/e$) | Same shape, mean term $\varepsilon_t(e^{\varepsilon_t} - 1)/2$, padded spread term | $(\varepsilon_g, \delta)$-DP, at 1.40–1.51 times the Whitehouse budget |
| Rényi filter | $\sum_{j\le t}\varepsilon_j(\alpha) \le B$ on every path, $\alpha$ fixed in advance | $(\alpha, B)$-RDP; for a DP target, $B = \varepsilon_g - \ln(1/\delta)/(\alpha - 1)$ |
| Ville odometer | $U_t = \min_{\alpha\in\Lambda}[S_t(\alpha) + \ln(\lvert\Lambda\rvert/\delta)/(\alpha - 1)]$ | Loss above $U_t$ at some $t$ with probability at most $\delta$ |
| Lécuyer odometer | Doubling Rényi filters at each order, with shrinking shares of $\delta$ | Same as Ville; slightly looser in practice |
| Individual spend, Gaussian sum | $\varepsilon_{i,t}(\alpha) = \alpha\lVert\bar g_{i,t}\rVert^2/(2\sigma^2C^2)$, at most $\alpha/(2\sigma^2)$ | Per-record filter gives $(\alpha, B)$-RDP for every record |

: Filters, odometers and individual accounting. Parameters $\varepsilon_t$ and $\varepsilon_t(\alpha)$ may depend on earlier outputs but must be known before their step runs. {#tbl-adapt-cheat}

**Rules of thumb**

- Never report a fixed-time bound at a time chosen from the outputs. Use a filter if you have a budget, and an odometer if you do not.
- In RDP, a filter at an order fixed before the run costs nothing. Track a few orders if you want the best one, and stop when any of them runs out.
- An odometer replaces $\ln(1/\delta)$ by $\ln(|\Lambda|/\delta)$ to pay for choosing the order late, so keep the grid small.
- Per-record $\varepsilon$ values are sensitive estimates, not guarantees, and the records an individual filter drops first are the ones the model already serves worst.

## Further reading

- Rogers, Roth, Ullman & Vadhan, "Privacy Odometers and Filters: Pay-as-you-Go Composition" (NeurIPS 2016): the definitions, the basic and first advanced filters, and the lower bound that makes odometers cost more.
- Whitehouse, Ramdas, Rogers & Wu, "Fully-Adaptive Composition in Differential Privacy" (ICML 2023): the filter that matches advanced composition, and the place to look for steps with $\delta_t > 0$.
- Feldman & Zrnic, "Individual Privacy Accounting via a Rényi Filter" (NeurIPS 2021): the Rényi filter and per-record accounting.
- Lécuyer, "Practical Privacy Filters and Odometers with Rényi Differential Privacy and Applications to Differentially Private Deep Learning" (2021): an independent proof of the Rényi filter, and the odometer with doubling budgets.

**Derivations.** [The full edition's chapter](../../book-full/part04-privacy-accounting/08-adaptive-composition.md) derives every result here step by step, except those it cites: the law of the iterated logarithm, the lower bound for odometers and the Rogers et al. filter (Rogers et al. 2016), the pure-DP odometers, Lécuyer's odometer (whose construction it sketches), the filters for steps with $\delta_t > 0$, individual accounting under subsampling and in Gaussian differential privacy, and the findings of Yu et al. (2023). The zCDP cost of a pure-DP step, the zCDP conversion, Mironov's conversion and the Gaussian RDP curve come from the earlier chapters on [Rényi DP](../../book-full/part04-privacy-accounting/01-renyi-dp.md), [RDP conversions](../../book-full/part04-privacy-accounting/02-rdp-conversions.md) and [concentrated DP](../../book-full/part04-privacy-accounting/03-concentrated-dp.md).
