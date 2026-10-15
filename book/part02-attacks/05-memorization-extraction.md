# Memorization and training-data extraction

## Why this matters

A language model trained on scraped text sees phone numbers, addresses, keys and private messages. Ask it to continue "Anna Smith's phone number is" and it may finish the sentence with the real number. Nobody queried a database and nothing was leaked by a breach: the number is stored in the weights, and it comes out through the model's normal interface. That is what **training-data extraction** means. Membership inference ([Membership inference](02-membership-inference.md)) asks "was this record in the training set?" and needs the record in hand. Extraction asks for more: the attacker gets the record itself, often without knowing it existed.

This chapter answers three questions, in order. The first is what it means for a model to memorize a string. There are several definitions, from "greedy decoding reproduces it when you prompt with the text that preceded it in training" to "some prompt makes the model emit it", and they differ in what the attacker must already know. Once you know what to measure, the second question is which strings get memorized. Two things decide it: duplication and model capacity. A string repeated often, or seen by a model large enough to give it a context of its own, comes back verbatim. The third question is how you measure memorization and how an attacker finds it. Planted random secrets called canaries, and a score called exposure, measure how much a model has memorized a secret you put there yourself. Generate-rank-verify pipelines find real training data among a model's samples.

### The one idea: the rank of the secret under the model

A language model is a distribution over strings: it gives every possible string a probability. Memorization is the model putting a training string *far above* the other strings that could have stood in its place. To make that precise, fix the context in which the secret appears, say "my pin is", and list all possible fillers $r$ (all five-digit pins) in order of the model's probability $p(r \mid \text{context})$, most likely first. The **rank** of the true secret is its position in that list. It is also the number of guesses an attacker needs if they try fillers from most to least likely, so a low rank means the model has done most of the attacker's work.

Every measure in this chapter reads that rank, in a different way. Greedy extraction, which writes out the model's single most likely next symbol at each step, reproduces the secret when the secret's next symbol is the top choice, rank 1, at every step. Exposure compares the secret's rank with the size of the space, on a logarithmic scale: it is $\log_2$ of the number of candidates minus $\log_2$ of the rank, which counts how many halvings of the search the model has saved the attacker, in bits. Sampling-based extraction, which draws random outputs from the model, hits the secret quickly when the secret has high probability, and high probability is what puts it near the top of the list.

@tbl-memo-uses lists the results of the chapter and where each is used.

| Result | Where it is used |
| --- | --- |
| Definitions: verbatim, approximate, discoverable, extractable, $k$-eidetic | Reading and comparing extraction papers; choosing what to measure |
| Samples needed to extract a string, $m \approx \ln(1/\beta)/p$ | Turning a probability into an attack budget |
| Copies needed for greedy extraction, $k^*$ | Why duplication and model size drive memorization; why deduplication helps |
| Exposure, its null distribution $\Pr[\text{exposure} \ge e] \le 2^{-e}$ | Auditing a trained model with planted canaries |
| Exposure as bits of guessing saved | Linking memorization to min-entropy and guessing |
| Ranking generations by a likelihood ratio to a reference model | Extraction attacks without the prefix; the same calibration as in membership inference |

: The results of this chapter and where each is used. {#tbl-memo-uses}

The chapter builds on four earlier ones, and restates what it uses from them where it uses it. [Algorithmic stability and generalization](../part01-prerequisites/07-stability-generalization.md) defined memorization as self-influence and showed that atypical examples are memorized most. [Information theory](../part01-prerequisites/05-information-theory.md) introduced min-entropy and guessing, which give exposure its meaning. [Threat models and privacy games](01-threat-models.md) set up attacks as games, and [membership inference](02-membership-inference.md) introduced the likelihood-ratio tests and reference models that extraction attacks reuse. The defence that bounds every quantity here, differential privacy, starts in Part 3.

**The running example.** Every experiment uses one toy that runs in seconds. The training set is a contact list: each line is a user name made of three random syllables, a space, and a five-digit pin, such as `toneka 14524`. There are 5,000 background lines, each seen once, and 600 secret lines, 100 each duplicated 1, 2, 4, 8, 16 and 32 times: 11,300 lines and 146,900 symbols in all, counting the end of each line. The model is a character $n$-gram language model with add-$\alpha$ smoothing, which predicts each character from the few characters before it by counting (Fact 2 below says exactly how). The order $n$ (the model sees the last $n - 1$ characters) plays the role of model size: an order-3 model stores 2,734 distinct $n$-grams and an order-8 model 54,614. It is the smallest model that has every property this chapter needs: a probability for every string, greedy decoding, sampling, and a capacity you can turn up. When a context is written out, `_` stands for a space: `gule_` is the five characters before the first digit of `migule 16430`.

## Toolkit: three facts every derivation uses

### Fact 1 — A language model is the chain rule

A language model never scores a whole string at once. It answers one smaller question many times: given everything written so far, how likely is each possible next symbol? Write $p(x_t \mid x_{<t})$ for the probability it gives to the next symbol $x_t$ after the symbols $x_{<t} = x_1 \dots x_{t-1}$ that come before it. To get the probability of a whole string $x = x_1\dots x_L$ of length $L$ after a prompt $c$, you multiply these one-step probabilities. That is the chain rule of probability: the chance that the first symbol is $x_1$, *and then* the second is $x_2$ given the first, and so on, is the product of the conditional chances. Taking logarithms turns the product into a sum:
$$p(x \mid c) = \prod_{t=1}^{L}p(x_t \mid c, x_{<t}), \qquad \ln p(x \mid c) = \sum_{t=1}^{L}\ln p(x_t \mid c, x_{<t}).$$ {#eq-memo-chain}

For the pin `16430` after the prompt `migule `, the model multiplies five numbers: the probability of `1` after `migule_`, of `6` after `migule_1`, of `4` after `migule_16`, of `3` after `migule_164`, and of `0` after `migule_1643`.

The negative of the log in base 2, $-\log_2p(x \mid c)$, is called the **log-perplexity** of $x$ in bits. It measures how surprised the model is by $x$: a string the model finds likely has low log-perplexity, and each extra bit means the model found it half as likely. Every score in this chapter (exposure, the probability that a sample reproduces a string, the ranking of generations) is built from this sum of per-symbol log-probabilities.

### Fact 2 — An $n$-gram model is a table of counts

An order-$n$ model looks only at the last $n - 1$ symbols, which it calls the context $c$. Training is nothing more than counting: for every context $c$ and every possible next symbol $w$, the model stores the count $N(c, w)$ of times $w$ followed $c$ in the training text. The natural prediction would be the fraction of times $w$ followed $c$, but that gives probability 0 to anything never seen, and one zero in the chain rule makes a whole string impossible. **Add-$\alpha$ smoothing** fixes this by pretending every symbol was seen $\alpha$ extra times in every context. With an alphabet of $V$ symbols, the $\alpha$ is added to each of the $V$ counts, so the total grows by $\alpha V$:
$$p(w \mid c) = \frac{N(c, w) + \alpha}{N(c) + \alpha V}, \qquad N(c) = \sum_wN(c, w).$$ {#eq-memo-ngram}

Three consequences of this formula do most of the work in the chapter.

The first is that **training is counting**. A line of text passes through a sequence of (context, next symbol) pairs, and training on it adds 1 to each of those counts. Training on $k$ copies of a line adds $k$ to each, and changes no other count.

The second is that **smoothing does not change the most likely symbol**. In a fixed context $c$, every count gets the same $+\alpha$, and the denominator $N(c) + \alpha V$ is the same for every $w$. So the symbol with the largest count still has the largest probability: $\arg\max_wp(w \mid c) = \arg\max_wN(c, w)$, where $\arg\max_w$ means "the $w$ that makes this largest". Greedy decoding, which emits the most likely symbol, therefore just follows the largest counts.

The third is what happens at a **context that occurs only in the copies of one line**. Say the line has been inserted $k$ times and its context $c$ appears nowhere else. Then the line's own next symbol has count $k$, every other symbol has count 0, and the total is $N(c) = k$. Putting those counts into @eq-memo-ngram gives
$$p(\text{true} \mid c) = \frac{k + \alpha}{k + \alpha V}, \qquad p(\text{other} \mid c) = \frac{\alpha}{k + \alpha V}.$$ {#eq-memo-unseen}
The two share a denominator, so the true symbol is $(k + \alpha)/\alpha$ times as likely as any other, whatever $V$ is. A context never seen at all has every count 0, and gives every symbol $\alpha/(\alpha V) = 1/V$.

The toy uses $\alpha = 0.1$ and $V = 39$ symbols: 26 letters, 10 digits, the space, a start symbol and an end-of-line symbol, $26 + 10 + 1 + 1 + 1 = 39$. @tbl-memo-unseen evaluates @eq-memo-unseen. For one copy, $\alpha V = 3.9$, so $p(\text{true} \mid c) = 1.1/4.9 = 0.2245$ and $p(\text{other} \mid c) = 0.1/4.9 = 0.0204$, a ratio of $1.1/0.1 = 11$. The last column turns the ratio into bits, $\log_2 11 = 3.46$: one copy makes the true digit as much more likely than any rival as 3.46 halvings would.

| Copies $k$ | $p(\text{true}\mid c)$ | $p(\text{other}\mid c)$ | Ratio | Bits per digit, $\log_2$ ratio |
| --- | --- | --- | --- | --- |
| 1 | 0.2245 | 0.0204 | 11.0 | 3.46 |
| 4 | 0.5190 | 0.0127 | 41.0 | 5.36 |
| 16 | 0.8090 | 0.0050 | 161.0 | 7.33 |

: Add-$\alpha$ probabilities at a context that occurs only in the $k$ copies of one line ($\alpha = 0.1$, $V = 39$). {#tbl-memo-unseen}

So a single copy already makes the true digit 11 times more likely than any other. That is the mechanism of memorization in this toy: a context that belongs to one training line alone, followed by that line's next symbol.

### Fact 3 — Guessing in order of probability takes "rank" guesses

Let a secret $r$ come from a finite set $\mathcal R$ of possible values, with $|\mathcal R|$ elements. An attacker who has a score for every candidate, here the model's probability, should guess candidates from the highest score down, because each guess then has the best remaining chance of being right. The number of guesses this takes is the rank of the true secret: the number of candidates at least as likely as it,
$$\operatorname{rank}(r) = \big|\{r' \in \mathcal R : p(r') \ge p(r)\}\big|,$$
counting $r$ itself and every candidate tied with it (ties are counted against the attacker, who may have to try them all first).

Without a model, the attacker has no reason to prefer one order over another. A uniform secret is then equally likely to sit at each position $1, \dots, |\mathcal R|$ of whatever order is used, so the number of guesses is on average the middle position, $(|\mathcal R| + 1)/2$. Information theory measures the same difficulty with one guess. The best single guess is the most likely value, and it succeeds with probability $\max_rp(r)$, which [Information theory](../part01-prerequisites/05-information-theory.md) calls the **vulnerability**. Its negative logarithm, the **min-entropy** $H_\infty = -\log_2\max_rp(r)$, expresses that difficulty in bits: a secret with $H_\infty$ bits is as hard to guess in one try as a uniform choice among $2^{H_\infty}$ values. A uniform 5-digit pin has $10^5$ equally likely values, so $H_\infty = \log_2 10^5 = 5\log_2 10 = 16.61$ bits.

The rank is what the rest of the chapter measures. Exposure is defined from the rank of a secret among all secrets of its format, and greedy decoding asks for rank 1 among the possible next symbols at every step.

## Verbatim, approximate, discoverable, extractable {#sec-memo-definitions}

### Context

"The model memorized its training data" can mean several things. A model that completes a famous quotation has memorized it, but so has every reader, and nothing private leaked. A model that emits a phone number seen once in training has leaked it. Some definitions assume the attacker knows the beginning of the training string. Others ask whether any prompt at all brings it out. Before measuring anything you need to say which notion you measure, because the numbers differ by orders of magnitude.

### Statement

Let $f$ be a language model trained on a dataset $X$ of documents, and let $\mathrm{greedy}_f(c)$ be the string produced by greedy decoding from the prompt $c$: at each step, append the most likely next symbol. The definitions below are parallel options that differ in what the attacker gets to use and in how close the output must be.

1. **Extractable** (Carlini et al. 2021): a string $s$ is extractable from $f$ if there is some prompt $c$, any prompt at all, such that $\mathrm{greedy}_f(c)$ begins with $s$.
2. **$k$-eidetic** (Carlini et al. 2021): $s$ is $k$-eidetically memorized if it is extractable and it appears in at most $k$ documents of $X$. Small $k$ is the dangerous case: a secret seen once and given back.
3. **Discoverable** (Carlini et al. 2023, named by Nasr et al. 2023): if the training data contain $[p \,\|\, x]$, a prefix $p$ followed by a suffix $x$, then $x$ is discoverably memorized if prompting with that same prefix makes greedy decoding produce the suffix: $\mathrm{greedy}_f(p)$ begins with $x$.
4. **Approximate** (Ippolito et al. 2023): $x$ is approximately memorized if $\mathrm{greedy}_f(p)$ is within distance $\tau$ of $x$ in some string distance $d$. In this chapter's toy $d$ is the number of wrong digits and $\tau = 1$, so a pin with at most one wrong digit counts. Definitions 1 to 3 are **verbatim**: the output must match exactly.

Two relations follow directly from the definitions. Discoverable is a special case of extractable, with the training prefix as the prompt, and an exact match is a special case of a close one:
$$\text{discoverable} \;\Rightarrow\; \text{extractable}, \qquad \text{verbatim} \;\Rightarrow\; \text{approximate}.$$

Real systems sample rather than decode greedily, so it is useful to know how many samples bring a string out. Each sample is a fresh, independent try that produces the string $x$ with its probability $p = p(x \mid c)$ after the prompt $c$, like a biased coin that lands heads with probability $p$. The chance that at least one of $m$ samples is $x$ is one minus the chance that all of them miss, $1 - (1 - p)^m$, and this is at least $1 - e^{-mp}$. Asking that to be at least $1 - \beta$, for a small failure probability $\beta$, gives a sample budget:
$$m \ge \frac{\ln(1/\beta)}{p} \quad\text{samples suffice to extract } x \text{ with probability at least } 1 - \beta.$$ {#eq-memo-samples}

### Derivation

**Discoverable implies extractable.** If $\mathrm{greedy}_f(p)$ begins with $x$, then $c = p$ is a prompt that makes the model emit $x$, which is the definition of extractable. The converse fails: an extractable string may come out after a prompt that is not its training prefix, or after a prompt the attacker cannot know, while its own prefix leads elsewhere.

**Verbatim implies approximate.** A verbatim match has distance $0$, and $0 \le \tau$.

**Samples needed.** The idea is to compute the probability that every sample misses, and bound it by an exponential that is easy to solve for $m$.

1. **One sample hits with probability $p$.** A temperature-1 sample from $c$ draws each next symbol with exactly the model's probabilities $p(x_t \mid c, x_{<t})$. By the chain rule (@eq-memo-chain), the chance that it writes out exactly $x$ is the product of those probabilities, which is $p = p(x \mid c)$.
2. **All $m$ samples miss with probability $(1 - p)^m$.** Each sample misses with probability $1 - p$, and the samples are independent, so the misses multiply.
3. **Replace $1 - p$ by $e^{-p}$.** The curve $e^{-p}$ is convex, and the straight line $1 - p$ is its tangent at $p = 0$ (both equal 1 there, with slope $-1$). A convex curve lies above its tangent lines, so $1 - p \le e^{-p}$, and raising both sides to the power $m$ gives $(1 - p)^m \le e^{-mp}$.
4. **Solve for $m$.** The miss probability is then at most $\beta$ as soon as $e^{-mp} \le \beta$. Taking logarithms, $-mp \le \ln\beta$, that is $m \ge \ln(1/\beta)/p$.

### Interpretation

The definitions answer different threat models. Discoverable memorization assumes an attacker who holds the training prefix. That is a strong assumption, but it makes the quantity easy to measure: prompt with every training prefix and compare the output with the true suffix. Extractable memorization fits the real attacker, who has only the model, but it is hard to measure, because you would have to search over all prompts. So the discoverable rates are what papers report on open models, and attacks then show how much of that an outsider can actually reach.

Verbatim matching is a weak notion of leakage, because it misses near copies. A model that emits a phone number with one digit changed, or a paragraph with its punctuation changed, has leaked it to any reader. Ippolito et al. (2023) show that a filter blocking verbatim training text is easy to bypass with prompts that change the style, while the approximate copies still come out. A verbatim rate is therefore a lower bound on the leak, not a measure of it.

Greedy decoding is a convenience for measurement; real deployments sample. @eq-memo-samples converts a probability into an attack budget. For a string with probability $10^{-3}$ after its prefix and 99% confidence ($\beta = 0.01$), the budget is $\ln(100)/10^{-3} = 4.605/0.001$, so the string comes out within about 4,600 samples.

### Example: one secret, six model sizes

Take the secret line `migule 16430`, inserted 4 times, and prompt each model with its prefix `migule `. Greedy decoding gives `07` followed by an end of line at order 3, `81554` at order 4, `14032` at order 5, and the true `16430` at orders 6, 7 and 8. The order-5 output gets the first and fourth digits right and the other three wrong, so it is not approximately memorized either. The reason the switch happens at order 6 is Fact 2. An order-6 model reads five characters of context, and from the first digit on its contexts (`gule_`, then `ule_1`, and so on) occur in this line and nowhere else. In such a context the line's own next digit has count 4 and every rival digit has count 0, so greedy decoding, which follows the largest count, emits the line's digit every time. Shorter contexts, such as the four characters `ule_` an order-5 model reads, are shared with other lines, whose digits can outvote the four copies.

Sampling gives a graded version of the same picture. At order 5, the probability that one sample from the true prefix reproduces a secret's pin, averaged over the 100 secrets of each group, is $1.4 \times 10^{-3}$ at 4 copies and $0.142$ at 32 copies. With $\beta = 0.01$ that means 3,252 and 31 samples for a 99% chance of extracting the pin. These are the exact budgets, the smallest $m$ with $(1 - p)^m \le \beta$; @eq-memo-samples gives slightly larger ones, since it replaces $1 - p$ by the larger $e^{-p}$. For $p = 0.142$, for instance, $\ln(0.01)/\ln(1 - 0.142) = 4.605/0.153 = 30.1$, which rounds up to 31. Repetition turns a needle in a haystack into the model's favourite answer.

### Check yourself

- Can a string be extractable but not discoverable? *(Yes. Another prompt, say a different name that shares the last syllables, may lead greedy decoding into it while its true prefix leads elsewhere.)*
- A model emits a training string with probability $10^{-6}$ after its prefix. Is it memorized? *(Not in the greedy sense, and by @eq-memo-samples an attacker needs about $\ln(100)/10^{-6} \approx 4.6 \times 10^6$ samples for 99% success. Whether that counts as a leak depends on the attacker's budget, which is why exposure, which uses the rank rather than the probability, is useful.)*

## Duplication and model size: the copies a string needs {#sec-memo-copies}

### Context

Carlini et al. (2023) measured discoverable memorization on a family of open models and found that it grows log-linearly with three things: the number of times a string is duplicated in the training set, the size of the model, and the length of the prompt. Kandpal, Wallace & Raffel (2022) found that a sequence repeated 10 times is generated about 1,000 times more often than one seen once. The $n$-gram toy lets you derive why, for every string, from counts.

### Statement

The idea is a vote. Greedy decoding follows the largest count at each step (Fact 2). Each copy of a line adds one vote for the line's own next symbol in every context the line passes through. So the line comes out when, at every step, its own votes plus whatever the rest of the corpus already gave its symbol beat the strongest rival symbol. The number of copies needed is set by the step where the rival is furthest ahead.

To write this down, let $s$ be a training line with prompt $p$ and secret suffix $x = x_1\dots x_L$, inserted $k$ times. Let $c_t$ be the context at step $t$ (the last $n - 1$ symbols of the prompt followed by the first $t - 1$ symbols of the suffix, $p\,x_{<t}$), and let $N_-(c, w)$ be the counts in the rest of the training set, *without* the copies of $s$. At step $t$, the **gap** is how far the strongest rival $w \ne x_t$ is ahead of the true symbol in those counts, $\max_{w \ne x_t}N_-(c_t, w) - N_-(c_t, x_t)$. Assume each context $c_t$ occurs only once within $s$, so a copy of $s$ adds to the true pair $(c_t, x_t)$ and to no rival pair $(c_t, w)$; this holds for almost every line. Then greedy decoding from $p$ emits $x$ if and only if $k \ge k^*(s)$, where the copies needed are the largest gap plus one:
$$k^*(s) = \max_{t \le L}\Big[\max_{w \ne x_t}N_-(c_t, w) - N_-(c_t, x_t)\Big] + 1.$$ {#eq-memo-kstar}
There is one refinement for ties. Greedy decoding breaks a tie toward the symbol with the lowest code, so at a step where the true symbol has the lower code, a tie is already a win, and the bracket is reduced by one there.

Over a population of lines, the fraction memorized at $k$ copies is then the fraction of lines whose $k^*$ is at most $k$:
$$\Pr[\text{discoverably memorized at } k \text{ copies}] = \Pr[k^*(s) \le k].$$
In particular, a line whose contexts are unique to it ($N_- = 0$ everywhere, so every gap is 0) has $k^* = 1$: it is memorized from a single copy.

### Derivation

1. **Greedy picks the largest count.** By Fact 2, adding $\alpha$ to every count does not change which count is largest, so at context $c_t$ greedy decoding emits $\arg\max_wN(c_t, w)$, the symbol with the largest count.
2. **What the copies add.** Training is counting (Fact 2), so $k$ copies of $s$ add $k$ to $N(c_t, x_t)$. By the assumption that $c_t$ occurs once in $s$, they add nothing to $N(c_t, w)$ for $w \ne x_t$. With the copies, the true symbol's count is $N_-(c_t, x_t) + k$ and each rival's is still $N_-(c_t, w)$.
3. **Winning one step.** The true symbol wins strictly when its count beats the largest rival count, $N_-(c_t, x_t) + k > \max_{w \ne x_t}N_-(c_t, w)$. Moving $N_-(c_t, x_t)$ to the right, this is $k > \text{gap}_t$, with $\text{gap}_t$ the bracket in @eq-memo-kstar. Since $k$ and the gap are whole numbers, that is $k \ge \text{gap}_t + 1$.
4. **Every step must win.** Greedy decoding stays on the line's path $c_1, c_2, \dots$ only while it keeps emitting the true symbols. One wrong symbol puts it in a different context, from which in general it does not return to $x$. So $x$ comes out exactly when every step wins, that is when $k \ge \text{gap}_t + 1$ for every $t$, which is $k \ge \max_t(\text{gap}_t + 1)$, @eq-memo-kstar.
5. **Unique contexts.** If $N_-(c_t, \cdot) = 0$ for every $t$, every gap is $0 - 0 = 0$, so $k^* = 1$.
6. **The population rate.** Each line is memorized at $k$ copies exactly when $k^*(s) \le k$. The fraction of lines memorized is therefore the fraction for which this holds, which is $\Pr[k^*(s) \le k]$ for a line drawn at random from the population.

@fig-memo-copies-needed draws steps 3 and 4 for one line.

![For one secret line inserted once, an order-5 model: at each step of its pin, the count of the true next digit (blue) and of its strongest rival (orange) in the rest of the training set. The label is the gap plus one, the copies that step needs. The line needs as many copies as its hardest step.](figures/memo_copies_needed.svg){#fig-memo-copies-needed}

### Interpretation

Duplication works as a vote. Each copy is one more vote for the line's continuation in every context it passes through, and a line comes out when it outvotes the rest of the corpus at its most contested step.

Model size works by splitting the vote. A longer context is a finer partition of the training positions: fewer positions share any one context of length 7 than of length 2. So the rival counts in a longer context are smaller, while the line's own $k$ votes stay with it in every context it passes through. In the limit every line has contexts of its own and $k^* = 1$. This is the toy version of "larger models memorize more": a larger model can afford to give rare strings their own parameters.

For the same reason, rare data are exposed first. A line whose name shares few syllables with other names reaches a context of its own sooner. This is the pattern [Algorithmic stability and generalization](../part01-prerequisites/07-stability-generalization.md) found for **self-influence**, how much a model's prediction on a training example changes when that one example is removed: atypical examples can only be fitted by memorizing them, so they have the largest self-influence.

Deduplication is the cheapest defence, because it lowers $k$ for every repeated string. Lee et al. (2022) found that models trained on deduplicated data emit memorized text about ten times less often, without losing quality, and Kandpal et al. (2022) found that deduplication sharply lowers the success of extraction attacks. It does nothing for singletons in a high-capacity model, where $k = 1$ already and $k^* = 1$.

### Example: discoverable memorization against copies and order

The script prompts each of the 600 secrets with its true prefix (the name and the space), decodes five symbols greedily, and compares them with the pin. @tbl-memo-discoverable gives the verbatim rate and the approximate rate (at most one digit wrong) for each order and number of copies. It also computes $k^*$ for every secret from @eq-memo-kstar, and the prediction $\Pr[k^* \le k]$ matches the verbatim rate exactly in all 36 cells, as the derivation says it must: the formula is not an approximation but a restatement of what greedy decoding does with counts.

| Order $n$ | $k = 1$ | $k = 2$ | $k = 4$ | $k = 8$ | $k = 16$ | $k = 32$ |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.00 / 0.00 | 0.00 / 0.03 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.02 | 0.00 / 0.05 |
| 4 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.01 | 0.00 / 0.00 | 0.00 / 0.00 | 0.13 / 0.17 |
| 5 | 0.00 / 0.00 | 0.00 / 0.00 | 0.05 / 0.06 | 0.11 / 0.15 | 0.42 / 0.42 | 0.70 / 0.70 |
| 6 | 0.08 / 0.13 | 0.65 / 0.69 | 0.76 / 0.78 | 0.80 / 0.85 | 0.90 / 0.91 | 0.96 / 0.97 |
| 7 | 0.70 / 0.78 | 0.95 / 0.96 | 0.98 / 0.98 | 0.98 / 0.99 | 0.99 / 0.99 | 0.97 / 0.98 |
| 8 | 0.98 / 0.98 | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 |

: Fraction of 100 secret lines per cell whose pin greedy decoding reproduces from the true prefix, verbatim / with at most one digit wrong. Rows are model orders, columns copies in the training set. {#tbl-memo-discoverable}

Read it along both axes. Along a row, more copies means more memorization: at order 5 the rate climbs from 0 at one copy to 0.70 at 32. Down a column, a larger model memorizes more: a line seen once is safe at orders up to 5, memorized 8% of the time at order 6, 70% at order 7 and 98% at order 8. The order-8 model has memorized its training set almost completely, singletons included. In the language of the definitions, those singletons are 1-eidetic: seen in one document and given back. @fig-memo-discoverable plots the same table for a second training set.

![Discoverable memorization against copies of the secret in the training set, for $n$-gram models of order 3 to 8, on an independent draw of the corpus. Left: greedy decoding reproduces the pin exactly. Right: at most one digit is wrong. Both rise with duplication and with order, and the approximate rate is only slightly higher.](figures/memo_discoverable.svg){#fig-memo-discoverable}

The approximate rates are barely above the verbatim ones. In this toy, a pin either sits in contexts of its own and comes out whole, or it loses a contest early, after which greedy decoding is in the wrong context and wanders off. Natural text has many near-ties (synonyms, punctuation), and that is where approximate memorization adds the most (Ippolito et al. 2023).

The real experiment of Carlini et al. (2023) has the same shape at scale. Take GPT-Neo models from 125 million to 2.7 billion parameters and GPT-J (6 billion), all trained on the Pile, a large public collection of web and book text. Sample training sequences, grouped by how many times they are duplicated in the Pile. Prompt each model with the first 50 tokens and decode 50 tokens greedily. Count exact matches. They found that the largest model reproduces at least 1% of its training data this way, and that the rate rises log-linearly with model size, duplication and prompt length. The toy replaces tokens by characters, parameters by the context length and the Pile by a contact list, and keeps the measurement.

### Check yourself

- In @fig-memo-copies-needed, why does the line need as many copies as its worst step, and not the sum over steps? *(The copies count at every step at once: each copy adds one to every pair on the line's path. So $k$ copies close every gap up to $k$, and the largest gap decides.)*
- Does deduplication help an order-8 model on this corpus? *(Barely. Its singletons are already memorized 98% of the time, because their contexts are unique. Deduplication works by lowering $k$, and here $k^* = 1$.)*

## Canaries and the exposure metric {#sec-memo-exposure}

### Context

Discoverable memorization is all or nothing: the secret comes out or it does not. An auditor wants a graded answer: how much closer to the secret does the model put an attacker? Carlini et al. (2019, "The Secret Sharer") proposed planting **canaries**, random secrets of a fixed format such as "my pin is" followed by five random digits, in the training data, then measuring where the trained model ranks the planted secret among all secrets of the same format. Because the canary is random and planted, you know the right answer and the size of the space, so you can quantify memorization exactly.

### Statement

A canary has a format $s[\cdot]$ with a hole, and a randomness space $\mathcal R$ of values that can fill the hole. Here $s[r]$ is "my pin is $r$", with $r$ one of the $|\mathcal R| = 10^5$ pins. A value $\hat r$ is drawn uniformly at random from $\mathcal R$, and $s[\hat r]$ is inserted $k$ times into the training data. After training, order all of $\mathcal R$ by the model's probability $p(r \mid \text{"my pin is"})$; the **rank** of $\hat r$ is its position in that order (Fact 3), the number of guesses an attacker who follows the model needs.

The exposure turns the rank into a number of bits saved. Without the model, the attacker faces all $|\mathcal R|$ candidates, which is $\log_2|\mathcal R|$ bits of search. With the model, it faces $\operatorname{rank}(\hat r)$ candidates, $\log_2\operatorname{rank}(\hat r)$ bits. The **exposure** is the difference:
$$\mathrm{exposure}(\hat r) = \log_2|\mathcal R| - \log_2\operatorname{rank}(\hat r).$$ {#eq-memo-exposure}

Two properties follow from this definition directly. Because the rank lies between 1 and $|\mathcal R|$, the exposure lies between 0 and $\log_2|\mathcal R|$, with the maximum when the canary is the model's top candidate. And exposure counts guesses saved: an attacker who guesses in the model's order needs $\operatorname{rank}(\hat r) = 2^{\log_2|\mathcal R| - \mathrm{exposure}}$ guesses instead of $|\mathcal R|$, so each bit of exposure halves the work.

The third property is what makes exposure an audit tool: you know how it behaves when nothing was memorized. If the model does not depend on $\hat r$, for instance because the canary was never inserted, then $\hat r$ is just a random point in a fixed ordering, and a high exposure is unlikely. Precisely, for every $e \ge 0$,
$$\Pr\big[\mathrm{exposure}(\hat r) \ge e\big] \le 2^{-e},$$ {#eq-memo-null}
and, when no two candidates tie, its average is a fixed small number of bits:
$$\mathbb E\big[\mathrm{exposure}(\hat r)\big] = \log_2|\mathcal R| - \frac{\log_2(|\mathcal R|!)}{|\mathcal R|} \;\longrightarrow\; \log_2e = 1.4427 \text{ bits}.$$ {#eq-memo-null-mean}

So exposure is $-\log_2$ of a p-value. A **p-value** is the probability, under the assumption that nothing happened, of seeing a result at least as extreme as the one observed. Here it is the probability that a canary that was never inserted would look at least this memorized, and @eq-memo-null says it is at most $2^{-\text{exposure}}$.

### Derivation

**Range and guesses.** The rank is between 1 and $|\mathcal R|$, so $\log_2\operatorname{rank}$ is between 0 and $\log_2|\mathcal R|$. Subtracting from $\log_2|\mathcal R|$ gives an exposure between $\log_2|\mathcal R|$ and 0. Solving @eq-memo-exposure for the rank gives $\operatorname{rank} = 2^{\log_2|\mathcal R| - \mathrm{exposure}}$.

**Null distribution.** The idea is that, without the canary, the model's scores are fixed before $\hat r$ is drawn, so $\hat r$ lands at a uniformly random position in their order.

1. **The order is fixed.** If the model was trained without the canary, its scores $p(r \mid \cdot)$ are fixed numbers that do not depend on $\hat r$. With no ties they order $\mathcal R$ into positions $1, \dots, |\mathcal R|$.
2. **The canary's position is uniform.** $\hat r$ is uniform on $\mathcal R$ and independent of the scores, so its position is equally likely to be any of $1, \dots, |\mathcal R|$.
3. **Count the positions with high exposure.** By @eq-memo-exposure, exposure $\ge e$ means $\log_2\operatorname{rank} \le \log_2|\mathcal R| - e$, that is $\operatorname{rank} \le |\mathcal R|2^{-e}$. There are $\lfloor|\mathcal R|2^{-e}\rfloor$ such positions (the whole numbers up to that bound), each with probability $1/|\mathcal R|$, so the probability is $\lfloor|\mathcal R|2^{-e}\rfloor/|\mathcal R| \le 2^{-e}$.
4. **Ties only help the bound.** The rank counts every candidate at least as likely as $\hat r$, so a tie can only raise the rank of $\hat r$. A higher rank means a lower exposure, so the probability of a high exposure can only fall, and the bound still holds.
5. **The mean.** With the rank uniform on $1, \dots, |\mathcal R|$, the average of $\log_2\operatorname{rank}$ is $\frac{1}{|\mathcal R|}\sum_{i=1}^{|\mathcal R|}\log_2 i$. A sum of logarithms is the logarithm of the product, and $1 \cdot 2 \cdots |\mathcal R| = |\mathcal R|!$, so this is $\log_2(|\mathcal R|!)/|\mathcal R|$; subtracting it from $\log_2|\mathcal R|$ gives @eq-memo-null-mean. For the limit, Stirling's approximation of the factorial says $\ln N! = N\ln N - N + O(\ln N)$, where $O(\ln N)$ is a correction that grows no faster than a multiple of $\ln N$. Dividing by $N\ln 2$ to get base-2 logarithms per candidate, $\log_2(N!)/N = \log_2N - 1/\ln 2 + O(\ln N/N) = \log_2N - \log_2e + O(\ln N/N)$, since $1/\ln 2 = \log_2 e$. The correction vanishes as $N$ grows, so the mean exposure tends to $\log_2 e$.

A space of four candidates makes all of this concrete. With $|\mathcal R| = 4$ the rank is 1, 2, 3 or 4 with probability $\frac14$ each, and the exposure is $2 - \log_2(\text{rank})$:

| Rank | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| Exposure $2 - \log_2(\text{rank})$ | 2 | 1 | 0.415 | 0 |

: The four equally likely exposures of a canary from a space of four candidates that was never inserted. {#tbl-memo-four}

The third entry is $2 - \log_2 3 = 2 - 1.585 = 0.415$. The mean is $\frac14(2 + 1 + 0.415 + 0) = 0.854$, which matches $2 - \log_2(24)/4 = 2 - 4.585/4$, since $4! = 24$. The probability of exposure at least 1 is $\frac12 = 2^{-1}$ (ranks 1 and 2), and of exposure at least 2 is $\frac14 = 2^{-2}$ (rank 1 only): the bound is tight at integer exposures.

### Interpretation

Exposure is a membership test with a known null. "Was this canary inserted?" is a membership question ([Membership inference](02-membership-inference.md)), and the exposure is the score of a test that says "inserted" when the exposure is at least a threshold $e$. By @eq-memo-null that test has false positive rate (the chance of saying "inserted" about a canary that was not) at most $2^{-e}$. In membership inference, calibrating a test means training **shadow models**, extra models trained with and without a record, to learn how the score behaves for non-members. Here none are needed: the randomness of the canary does the calibration, playing the role that the population of non-members plays in membership inference.

Exposure is also guessing work saved, in the sense of Fact 3. A uniform canary has min-entropy $\log_2|\mathcal R|$: before seeing the model, the attacker's best single guess succeeds with probability $1/|\mathcal R|$. After the model, the attacker who guesses in the model's order needs $2^{\log_2|\mathcal R| - \mathrm{exposure}}$ tries. Carlini et al. (2019) note that exposure close to $\log_2|\mathcal R|$ means the secret can actually be extracted, for example by a shortest-path search: treat each digit position as a step, weight each choice of digit by its negative log-probability, and find the cheapest path through the model's predictions, which is the most likely pin.

For large spaces exposure has to be estimated, because enumerating $\mathcal R$ is impossible for, say, a 9-digit number. Two estimators replace enumeration. The first is sampling: score $m$ random candidates and use the fraction of them at least as likely as the canary as the estimate of $\operatorname{rank}/|\mathcal R|$. It cannot resolve exposures above $\log_2m$, because a canary above all $m$ sampled candidates only tells you its rank fraction is below about $1/m$. The second is extrapolation: fit a distribution to the sampled scores and read off its tail beyond the canary's score; Carlini et al. (2019) fit a skew-normal, a normal distribution with an extra parameter that lets one tail be heavier than the other. The fit is only as good as its tail, and the tail is exactly the part you cannot sample.

Ties hide memorization. A model that has never seen the canary's contexts gives every candidate the same score. Every candidate is then tied with all the others, so every rank is $|\mathcal R|$ and every exposure 0. On this corpus the order-8 model without a canary gives every pin exactly the same probability after "my pin is", and the exposure of every candidate is 0.

Finally, canaries test defences, not only models. Because you control what was planted, you can compare exposure across two training runs that differ in one thing. Train with and without deduplication, and the change in exposure measures what deduplication bought. Train with a differentially private algorithm (Part 3), and exposure near the null mean says the planted secrets were not memorized; Carlini et al. (2019) found that private training kept their canaries at the exposure of canaries never inserted. Plant canaries, unlearn them, and measure exposure again, and you have a test of an unlearning method. Auditing differentially private training uses the same tool, with canaries designed to be as exposed as possible.

### Example: one canary, three model sizes

The script plants "my pin is 58745" and computes the exact exposure by scoring all $10^5$ pins. @tbl-memo-exposure gives it for orders 2, 3 and 4.

| Order $n$ | $k = 0$ | $k = 1$ | $k = 2$ | $k = 4$ | $k = 8$ | $k = 16$ | $k = 32$ | $k = 64$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.40 | 0.41 | 0.43 | 0.47 | 0.55 | 0.73 | 1.21 | 2.54 |
| 3 | 0.89 | 4.05 | 4.44 | 4.71 | 5.24 | 6.59 | 9.69 | 15.61 |
| 4 | 3.76 | 9.98 | 12.15 | 15.02 | 15.61 | 16.61 | 16.61 | 16.61 |

: Exposure in bits of the canary "my pin is 58745" against the number of copies $k$ in the training set. The maximum is $\log_2 10^5 = 16.61$. {#tbl-memo-exposure}

The jump from $k = 0$ to $k = 1$ follows from Fact 2: what matters is whether a digit of the canary sits in a context that only the canary's copies fill. No background line contains `s_`, an `s` followed by a space, because names end in a vowel. So the three orders behave differently.

At order 2 the model reads one character of context, and the first digit's context is a single space, shared by all 11,300 lines. No digit of the canary has a context of its own, so one copy is a single vote among thousands, and it moves the exposure by $0.41 - 0.40 = 0.01$ bits.

At order 3 the first digit's context is `s_`, which occurs only in the canary. By Fact 2 and @tbl-memo-unseen, one copy makes the first digit 11 times as likely as each other first digit, a factor worth $\log_2 11 = 3.46$ bits. The exposure goes from 0.89 to 4.05, a rise of $4.05 - 0.89 = 3.16$ bits: close to 3.46 but not equal, because the rank also depends on how the other pins' scores are spread.

At order 4 the first two digits have unique contexts, `is_` and `s_5`, worth about $2 \times 3.46 = 6.92$ bits. The exposure goes from 3.76 to 9.98, a rise of $9.98 - 3.76 = 6.22$ bits.

The remaining digits share their contexts with the 5,000 background pins and move only as the copies outvote them. @fig-memo-exposure-histogram shows the same movement on the full distribution of scores, and @fig-memo-exposure-vs-copies averages several canaries.

![Histogram of $\log_2p(r \mid \text{my pin is})$ over all $10^5$ pins for an order-4 model without the canary (grey), and the score of the canary after it is inserted 1, 4, 16 and 64 times (dashed). Each insertion pushes the canary further into the right tail, past more and more of the other pins.](figures/memo_exposure_histogram.svg){#fig-memo-exposure-histogram}

![Exposure against copies of the canary for $n$-gram models of order 2, 3 and 4. Thin lines are individual canaries and markers their mean; the dotted line is the maximum, $\log_2 10^5$. Larger models reach the maximum with fewer copies.](figures/memo_exposure_vs_copies.svg){#fig-memo-exposure-vs-copies}

**The null in practice.** For a pin that was never inserted, the derivation predicts a uniform rank. For orders 2, 3 and 4 the scores have no ties, so the exposures of all $10^5$ uninserted pins are exactly the uniform-rank values: mean 1.443 bits, against 1.4426 from @eq-memo-null-mean, and a fraction $0.03125 = 1/32 = 2^{-5}$ with exposure at least 5, exactly on the bound @eq-memo-null.

**Estimating instead of enumerating.** At order 3, the script estimates the canary's exposure from 10,000 random pins, by the two estimators of the interpretation: counting how many sampled pins score at least as high as the canary, and fitting a normal distribution to the sampled scores and reading off its tail (@tbl-memo-estimate).

| Copies $k$ | Exact exposure | Rank | Sampling estimate | Normal fit |
| --- | --- | --- | --- | --- |
| 1 | 4.05 | 6,018 | 4.13 | 4.47 |
| 2 | 4.44 | 4,606 | 4.43 | 5.57 |
| 4 | 4.71 | 3,823 | 4.68 | 6.81 |
| 8 | 5.24 | 2,648 | 5.35 | 8.28 |

: Exposure of the canary at order 3, computed exactly over all $10^5$ pins and estimated from 10,000 random pins, by counting and by fitting a normal to their scores. {#tbl-memo-estimate}

Counting is accurate here, because every exposure is well below $\log_2 10^4 = 13.3$ bits, the largest value 10,000 samples can resolve. The normal fit overestimates by up to 3 bits (at 8 copies, $8.28 - 5.24 = 3.04$). The score distribution has a heavier right tail than a normal, so the normal puts too few pins above the canary, which makes the canary's rank look smaller and its exposure larger. This is why the choice of tail model matters, and why Carlini et al. (2019) fit a skewed distribution.

### Check yourself

- An auditor plants one canary, trains, and measures exposure 10 bits on a space of $10^9$. What is the chance a non-inserted canary scores that high? *(At most $2^{-10} = 1/1024 \approx 0.001$, by @eq-memo-null. The size of the space does not enter.)*
- Why does the order-2 model barely react to the canary even at 64 copies? *(Every context it sees is a single character, shared with thousands of training positions. The 64 votes are a small fraction of the counts in each context. The model is too small to give the canary anything of its own.)*

## Extraction without the prefix: generate, rank, verify {#sec-memo-attack}

### Context

The attacker of @sec-memo-definitions had the training prefix. A real attacker has only the model. Carlini et al. (2021) showed how such an attacker extracts training data from GPT-2: generate a large number of samples, rank them by how likely they are to be memorized, and check the top of the ranking. The checking step is easy for whoever has the training data (a search) and, for an outside attacker, can be done against public sources. The hard step is the ranking, and it is a hypothesis test: for each sample, decide between two explanations of why the model produced it.

### Statement

A model can find a string likely for two different reasons. The string may be ordinary, the kind of text any model of the language would find likely, or the model may have memorized it from its training set. Only the second is a leak. So for a generated string $x$, test $H_0$: "$x$ is likely because text like it is common" against $H_1$: "$x$ is likely because the target model memorized it". To score the two explanations you need a model of each. A **reference model** $p_{\mathrm{ref}}$, trained on data from the same distribution but not on the target's training set, stands in for "common text", the distribution under $H_0$. The target model $p_\theta$ itself stands in for $H_1$. The best evidence for memorization is then how much more likely the target finds $x$ than the reference does, which on a log scale is a difference. The most powerful test at every false positive rate ranks generations by this log-likelihood ratio,
$$\Lambda(x) = \ln p_\theta(x) - \ln p_{\mathrm{ref}}(x),$$ {#eq-memo-ratio}
not by $\ln p_\theta(x)$ alone.

### Derivation

1. **The two hypotheses give two probabilities.** Under $H_0$ the string has probability $p_{\mathrm{ref}}(x)$, and under $H_1$ it has probability $p_\theta(x)$.
2. **The best test thresholds their ratio.** The Neyman–Pearson lemma of [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) says that, among all tests with a given false positive rate (the fraction of $H_0$ strings wrongly flagged), the one with the highest true positive rate (the fraction of $H_1$ strings caught) flags a string when the likelihood ratio $p_\theta(x)/p_{\mathrm{ref}}(x)$ is above a threshold. The reason is a budget argument: each flagged string costs false positives in proportion to $p_{\mathrm{ref}}(x)$ and gains true positives in proportion to $p_\theta(x)$, so for a fixed budget of false positives you catch the most by flagging the strings with the largest ratio first.
3. **Take logarithms.** The logarithm is increasing, so it keeps the order of the ratios, and thresholding the ratio is the same as thresholding its logarithm, @eq-memo-ratio.
4. **Why raw likelihood falls short.** Ranking by $\ln p_\theta(x)$ alone is the same as using @eq-memo-ratio with $p_{\mathrm{ref}}$ constant, as if every string were equally common a priori. That assumption is wrong exactly for strings that are likely under any model, such as boilerplate, licences and famous quotations, and those are the strings raw likelihood puts first.

This is the same move as the per-example calibration of [Membership inference](02-membership-inference.md): there, a record's raw loss under the target model is compared with the loss that models which did not see the record would give it.

### Interpretation

Raw likelihood finds the common, not the private. The most likely generations of a large model are text that appears everywhere: software licences, news boilerplate, number sequences. These are in the training data, but extracting them reveals nothing. A reference model knows them too, so their ratio is close to 1 and the difference in @eq-memo-ratio cancels them.

The reference can be crude. Carlini et al. (2021) used three: a smaller model of the same family, the length of the text after compression with zlib (a general-purpose compressor, so the compressed length is a model-free measure of how predictable the text is), and the target's own likelihood on the lowercased text. Each gives a different idea of "generic", and each catches strings the others miss.

What the attacker cares about is precision, not recall. An attacker who checks the top 100 candidates by hand cares about how many of those are real training data, the precision of the top of the ranking, not about how many of all memorized strings it eventually finds. A high precision among a few flagged strings is a high true positive rate at a tiny false positive rate, the same operating point that matters in membership inference.

### Example: extracting secrets from the contact-list model

The attacker samples 20,000 lines from the order-6 model, starting each at the beginning of a line. To make the ranking problem realistic, 50 **public** lines (think of a published directory) are added 100 times each to the target's training set; the attacker's own reference data, an independent contact list of 5,000 lines from the same generator, also contain them. The reference model is trained on those reference data. A generation counts as a hit if it is a private training line: in the target's training set and not public.

Of the 20,000 samples, 14,852 are distinct. Among them are 197 private training lines and all 50 public lines. By duplication group, the samples contain none of the secrets inserted 1, 2 or 4 times, 11% of those inserted 8 times, 86% of those inserted 16 times and all of those inserted 32 times. Compare the discoverable rates at order 6 (@tbl-memo-discoverable): 8% of the singletons and 65% of the lines seen twice come out when you hold their prefix. Without the prefix, a sample must first produce the right name before it can produce the pin, and only heavily duplicated lines make their names likely enough. What this attack extracts is much less than what is discoverable.

Ranking matters (@fig-memo-extraction). Among the top 100 distinct generations by target likelihood, 45 are private training lines; the rest are mostly the 50 public lines, which, repeated 100 times each, are the most likely strings the model knows. Ranking by @eq-memo-ratio, 85 of the top 100 are private training lines, because the reference model also finds the public lines likely and cancels them.

![Fraction of the top-ranked distinct generations that are private training lines (solid) or public lines (dashed), from 20,000 samples of an order-6 model trained with 50 public lines repeated 100 times. Ranking by likelihood puts the public lines first. Ranking by the likelihood ratio to a reference model trained on other data puts the private lines first.](figures/memo_extraction.svg){#fig-memo-extraction}

### The production attack in prose

Nasr et al. (2023) carried this to deployed models. Their measurements separate the two notions of @sec-memo-definitions, discoverable and extractable. On open models whose training data are public (Pythia, GPT-Neo and others), they measured how much of what an attacker generates is verbatim training data, by checking every 50-token window of the output against a suffix array of the training set, an index that finds any substring of a huge text quickly. For closed models, whose training data are unknown, they built a 9-terabyte auxiliary corpus of public web text and counted 50-token matches against it as a lower bound.

The aligned chat model they attacked, ChatGPT (gpt-3.5-turbo), rarely emitted training data when used normally: alignment training, a later stage that fine-tunes the model to follow instructions, had taught it to answer as an assistant, not to continue documents. Their **divergence attack** asked it to repeat a single word forever. After many repetitions the model diverged from the chat behaviour and began to emit text that looked like pretraining documents, and a sizeable part of that text matched the web corpus verbatim. They report that the attack made the model emit training data at a rate about 150 times higher than normal use, and that about 200 US dollars of queries extracted over 10,000 unique memorized training examples, including personal contact details. The structure is the one of this chapter: generate (with a prompt that escapes the aligned behaviour), verify (against a corpus), and estimate the total from what was found.

### Check yourself

- Why are none of the secrets inserted 4 times generated, although at order 6 76% of them are discoverably memorized? *(An unprompted sample must first produce the secret's name. With 343,000 possible names, a sample hits a particular one only if that name is likely, which takes many copies.)*
- When would ranking by raw likelihood be as good as the ratio? *(When nothing is generically likely: if every string had the same prior probability, $p_{\mathrm{ref}}$ would be constant and the two rankings would agree.)*

## Code: check every number by computation and simulation

The script [code/memorization.py](code/memorization.py) builds the contact list, trains the $n$-gram models by counting, and runs every experiment in this chapter: greedy and sampled extraction from true prefixes, the copies needed $k^*$ for every secret, exact and estimated exposure of canaries over all $10^5$ pins, and the generate-rank-verify attack. Every model is a small typed dataclass of sorted count arrays, so scoring all $10^5$ pins is one vectorized lookup. From the repository root:

```bash
uv run python book/part02-attacks/code/memorization.py
```

**Output** (seed 0), summarized in @tbl-memo-code:

| Experiment | Result |
| --- | --- |
| Corpus | 5,000 background lines, 600 secrets (100 each at 1, 2, 4, 8, 16, 32 copies), 11,300 lines, 146,900 symbols |
| Model size, distinct $n$-grams | Order 3: 2,734; 4: 19,129; 5: 42,407; 6: 52,610; 7: 54,334; 8: 54,614 |
| One secret, `migule 16430` × 4 | $k^* = 151, 43, 9, 1, 1, 1$ for orders 3 to 8; greedy reproduces it from order 6 |
| Discoverable memorization | @tbl-memo-discoverable; $\Pr[k^* \le k]$ matches greedy in 36 of 36 cells |
| Sampling from the prefix, order 5 | Mean $p(\text{pin} \mid \text{prefix})$ $1.4 \times 10^{-3}$ at 4 copies, 0.142 at 32; 3,252 and 31 samples for 99% |
| Exposure null, orders 2–4 | Mean 1.443 bits (theory 1.4426); $\Pr[\text{exposure} \ge 5] = 0.03125$ (bound $2^{-5}$); order 8: all tied, exposure 0 |
| Canary exposure | @tbl-memo-exposure |
| Exposure estimates, order 3 | @tbl-memo-estimate |
| Extraction, order 6 | 14,852 distinct of 20,000 samples; 197 private lines; top 100 precision 0.45 by likelihood, 0.85 by ratio |

: The script's output with the default seed. {#tbl-memo-code}

[tests/test_memorization.py](../../tests/test_memorization.py) checks that the output still matches.

**What to take from it.** Four things stand out. First, the count formula is exact: $k^*$ from @eq-memo-kstar predicts which secrets greedy decoding reproduces in every cell, ties included, because it is a restatement of how greedy decoding reads counts. The `migule 16430` row shows the mechanism in one line: the secret, inserted 4 times, would need 151 copies at order 3, 43 at order 4 and 9 at order 5, but only 1 from order 6 on, which is why greedy decoding reproduces it from order 6. Second, duplication and model size both matter, and they trade off against each other: a small model needs dozens of copies, a large one needs one. Third, exposure behaves like a p-value. Without the canary, exposures follow the uniform-rank law exactly; with it, each digit of the canary that has a context of its own adds about 3.5 bits at one copy, the $\log_2 11$ of Fact 2. Fourth, the attacker without the prefix gets less, and needs a reference. Only heavily duplicated secrets come out of unprompted samples, and ranking by the likelihood ratio nearly doubles the precision of the top 100, from 0.45 to 0.85.

## References

- Carlini, Liu, Erlingsson, Kos & Song, "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks" (USENIX Security 2019): canaries, the exposure metric, its estimation by sampling and skew-normal extrapolation, and extraction of planted secrets by search.
- Carlini et al., "Extracting Training Data from Large Language Models" (USENIX Security 2021): extractable and $k$-eidetic memorization; the generate-rank-verify attack on GPT-2 with perplexity, reference-model, zlib and lowercase ranking.
- Carlini, Ippolito, Jagielski, Lee, Tramèr & Zhang, "Quantifying Memorization Across Neural Language Models" (ICLR 2023): memorization measured by prompting with training prefixes; log-linear growth with model size, duplication and prompt length.
- Nasr et al., "Scalable Extraction of Training Data from (Production) Language Models" (2023): discoverable against extractable memorization, measurement on open models against their training sets, and the divergence attack on an aligned chat model.
- Kandpal, Wallace & Raffel, "Deduplicating Training Data Mitigates Privacy Risks in Language Models" (ICML 2022): how the rate of regenerated sequences grows with duplication.
- Lee et al., "Deduplicating Training Data Makes Language Models Better" (ACL 2022): near-duplicate detection at scale and its effect on emitted training text and model quality.
- Ippolito et al., "Preventing Generation of Verbatim Memorization in Language Models Gives a False Sense of Privacy" (INLG 2023): approximate memorization and why verbatim filters fail.
- Feldman, "Does Learning Require Memorization? A Short Tale about a Long Tail" (STOC 2020): why rare examples must be memorized for accuracy, which explains why they are exposed first.
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft), ch. 3: $n$-gram language models, smoothing and perplexity.

## Cheat sheet

$p$ is a training prefix and $x$ its suffix, $k$ the number of copies of a string in the training set, $n$ the order of an $n$-gram model, $\mathcal R$ a canary's randomness space. @tbl-memo-cheat collects the results.

| Result | Statement | Key fact |
| --- | --- | --- |
| Extractable | Some prompt $c$ makes greedy decoding emit $s$ | The real attacker's notion; hard to measure |
| Discoverable | Greedy from the true prefix $p$ emits $x$ | Easy to measure; implies extractable |
| $k$-eidetic | Extractable and in at most $k$ documents | $k = 1$: a secret seen once comes back |
| Approximate | Greedy output within distance $\tau$ of $x$ | Verbatim rates understate the leak |
| Samples needed | $m \ge \ln(1/\beta)/p(x \mid c)$ | From $(1 - p)^m \le e^{-mp}$ |
| Copies needed ($n$-gram) | $k^* = \max_t[\max_{w \ne x_t}N_-(c_t, w) - N_-(c_t, x_t)] + 1$ | Largest gap on the path decides; unique contexts give $k^* = 1$ |
| Exposure | $\log_2|\mathcal R| - \log_2\operatorname{rank}$ | Bits of guessing saved |
| Exposure null | $\Pr[\mathrm{exposure} \ge e] \le 2^{-e}$; mean $\to\log_2e = 1.443$ | Exposure is $-\log_2$ of a p-value |
| Ranking generations | $\ln p_\theta(x) - \ln p_{\mathrm{ref}}(x)$ | Neyman–Pearson; cancels generically likely text |

: Every result in this chapter. {#tbl-memo-cheat}

**Rules of thumb**

- Memorization is a rank: the secret above the strings it could have been.
- Duplication is a vote in every context the string passes through; the most contested context decides.
- Larger models split contexts and give rare strings their own parameters, so they memorize more and from fewer copies.
- Deduplicate first: it is cheap and helps, but not for singletons in a high-capacity model.
- Report exposure, not just extraction: it is graded and has a known null.
- Rank generations against a reference, never by raw likelihood.
- An attacker without the prefix gets less than the discoverable rate suggests, but production attacks show "less" is still thousands of records.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](solutions/05-memorization-extraction.md).
