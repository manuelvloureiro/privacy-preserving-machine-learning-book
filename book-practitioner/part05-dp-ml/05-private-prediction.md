# Private prediction: PATE and in-context learning

## Why this matters

Training a model with differential privacy (DP) by stochastic gradient descent (SGD), as [DP-SGD](02-dp-sgd.md) does, makes every later query free, but you must own the training loop. The model may be a large language model (LLM) that you can prompt but not train, or one that answers only a few dozen questions in its life. In those cases you can train ordinary models, keep them secret, and make each *answer* private instead.

The idea is "one record, one vote". You split the private records into $k$ disjoint shards, train one model, a **teacher**, on each, and release a noisy winner of their vote. Because a record lives in one shard, it can change only one vote. This chapter says when such answers beat a private model, how private aggregation of teacher ensembles (PATE) turns them into a free-to-use model, and how the vote protects an LLM's prompt.

## The noisy vote: GNMax {#sec-pate-noisy-max}

Assign records to shards by a fixed rule chosen before looking at the data, such as a hash of a record identifier, or of a user identifier if the privacy unit is a user. Adding or removing a record then changes one teacher, which can move one vote from one label to another. The list of vote counts, the **vote histogram**, therefore moves by at most $\sqrt2$ in Euclidean length. The fixed rule is a condition, because if you cut shards by position in a file, inserting one record shifts every later record and can change every teacher. Likewise, if one person's records land in several shards, that person moves several votes.

**GNMax** (Papernot et al. 2018) adds Gaussian noise to each count and releases only the label with the largest noisy count. Write $\sigma$ for the noise's standard deviation. Rényi differential privacy (RDP) describes a mechanism by its privacy loss $\varepsilon(\alpha)$ at each order $\alpha$ ([Rényi DP](../part04-privacy-accounting/01-renyi-dp.md)). In those terms each answer costs a straight line:
$$\varepsilon(\alpha) = \frac{\alpha}{\sigma^2}.$$
The lines of $Q$ answers add up, even when each query is chosen after seeing the earlier answers. Now write $\rho(\varepsilon, \delta)$ for the largest slope whose line still converts to your target, a privacy loss $\varepsilon$ with slack $\delta$ ([From Rényi DP to (ε, δ)](../part04-privacy-accounting/02-rdp-conversions.md)). The noise that $Q$ answers need is then
$$\sigma = \sqrt{\frac{Q}{\rho(\varepsilon, \delta)}}.$$ {#eq-pate-sigma}
This chapter uses $\delta = 10^{-5}$ throughout. A budget of $\varepsilon = 1$ then allows a slope of 0.0306, and $\varepsilon = 4$ allows 0.3731.

| Budget | Answers $Q$ | GNMax $\sigma$ |
| --- | --- | --- |
| $\varepsilon = 1$ | 1 | 5.72 |
| | 100 | 57.2 |
| | 1,000 | 181 |
| | 10,000 | 572 |
| $\varepsilon = 4$ | 1 | 1.64 |
| | 1,000 | 51.8 |

: Noise per count for $Q$ answers at $(\varepsilon, 10^{-5})$, from the RDP accountant. {#tbl-pate-noise}

Two facts follow from @eq-pate-sigma. The noise grows like the square root of the number of answers, and it does not depend on the teachers at all. So a teacher can be any model, or even a human annotator who saw one shard. The original PATE used **LNMax**, with Laplace noise. It needs less noise for a single answer, but about twice as much from 100 answers on, so use GNMax for anything beyond one or two answers.

Publishing the noisy counts as well as the label releases a whole vector, which the analysis does not cover. And since a gap can never exceed the number of teachers, the noise for 10,000 answers at $\varepsilon = 1$ swamps any ensemble with fewer than a few thousand teachers.

## How accurate a noisy vote is {#sec-pate-accuracy}

On two labels, the noisy vote returns the plurality label unless the difference of the two noises exceeds the **gap** $g$ between the two counts. That difference is Gaussian with standard deviation $\sqrt2\sigma$, so the plurality wins with probability
$$\Pr[\text{plurality}] = \Phi\Big(\frac{g}{\sqrt2\,\sigma}\Big),$$ {#eq-pate-two-class}
where $\Phi$ is the standard normal distribution function. The noiseless plurality is a ceiling, because noise can only pull an answer toward a coin flip.

Take this part's running example from [DP-SGD](02-dp-sgd.md), whose 10,000 private records have two labels, and split them among 100 teachers. With the noise for 1,000 answers at $\varepsilon = 4$, a lopsided vote of 98 to 2 returns the plurality with probability 0.9051. But a close vote of 52 to 48 does so with probability 0.5218, barely better than a coin.

More teachers mean smaller shards and weaker teachers, so the right number is found by measuring. On the running example, shrinking the shards to 20 records cuts a teacher's accuracy to 0.6459. Yet the noiseless plurality stays between 0.9336 and 0.9403, because weak teachers err nearly independently. The gaps therefore grow with the number of teachers, and 500 teachers did best. So add teachers as long as the median gap keeps growing, and measure the noiseless vote first, since no budget gets you above it.

## Private answers or a private model {#sec-pate-versus}

The same budget can train a private model, whose answers are free, or pay for answers one at a time. A private model's accuracy does not depend on the number of queries, while the vote's falls as answers accumulate. So private answers are at least as accurate up to a **break-even** number of queries $Q^*(\varepsilon)$, and the private model wins beyond it.

| $\varepsilon$ | DP-SGD accuracy | Break-even $Q^*$ | Best number of teachers |
| --- | --- | --- | --- |
| 0.5 | 0.8164 | 58 | 500 |
| 1 | 0.9128 | 18 | 500 |
| 2 | 0.9438 | 0 | — |
| 4 | 0.9573 | 0 | — |

: DP-SGD on the running example against GNMax answers: the largest number of queries for which the answers' expected accuracy is at least the private model's. {#tbl-pate-breakeven}

At small budgets DP-SGD loses a lot of accuracy, so the vote wins a window of queries, but from $\varepsilon = 2$ on DP-SGD beats even the ensemble's ceiling. To decide for your task, estimate the private model's accuracy at your budget, and compute the vote's expected accuracy from @eq-pate-two-class and the teachers' gaps on held-out queries. Then compare them at your expected number of queries.

The table assumes you *can* run DP-SGD, which is false for LLMs behind an interface.

## PATE: from private answers to a private model {#sec-pate-student}

PATE spends the budget once. It labels $Q$ *public, unlabelled* inputs with GNMax answers and trains a new model, the **student**, on them. The student is released and the teachers never are. Its weights and predictions cost exactly what the $Q$ answers cost.

The recipe has four conditions. You need public inputs from the same distribution as the private data, which is easy for images or text and impossible for most medical records. The pool must really be public, because nothing protects an input that is someone's private record. The number of labels needs tuning, since more labels mean more noise on each. And the teachers stay on a server you trust, because they are non-private models open to every attack of [Part 2](../part02-attacks/00-introduction.md).

On the running example, PATE loses to DP-SGD at every budget measured. At $\varepsilon = 4$, the best student of 250 teachers reaches 0.8898, against 0.9573 for DP-SGD. The weak teachers often disagree, and PATE wins only where teachers agree on most inputs and the student also learns from unlabelled inputs. On handwritten digits, Papernot et al. (2018) report a student within one point of the non-private accuracy.

## Data-dependent accounting and Confident-GNMax {#sec-pate-data-dependent}

The cost $\alpha/\sigma^2$ is a worst case over all vote histograms. But when nearly all teachers agree, one record almost never changes the answer, and a **data-dependent analysis** (Papernot et al. 2018) charges such a query far less. On tasks where teachers are nearly unanimous, as on digits, $\varepsilon$ falls by a large factor. On the running example the saving is only about 20%, because few queries have large gaps.

The trap is that a data-dependent $\varepsilon$ is computed from the votes, so publishing it leaks information about the data. Papernot et al. release it with noise scaled to its smooth sensitivity ([Beyond global sensitivity](../part03-dp-foundations/07-sensitivity.md#sec-sens-smooth)) and pay for that release. Until it is released privately, a data-dependent $\varepsilon$ is a measurement, not a guarantee.

Two refinements exist. **Confident-GNMax** answers only the queries on which a noisy check finds the teachers in agreement, which gave a student of 0.8894 on the running example, a small gain over plain GNMax. **Stable aggregation** (Bassily, Thakkar & Thakurta 2018) releases the exact plurality of lopsided votes, but it is mainly a tool for proofs.

## Private in-context learning {#sec-pate-icl}

An LLM can learn a task from labelled examples placed in its prompt, called **exemplars**. If the exemplars are private, the answers leak them. Duan et al. (2023) show that GPT-3, prompted with one private example, favours that example's correct label enough for a membership inference attack.

The fix is private prediction with prompts as teachers (Duan et al. 2023; Wu et al. 2024). You split the private exemplars into disjoint groups by a fixed rule such as a hash, prompt the model once per group with the query, and release the GNMax label of the answers. So each answer costs $\alpha/\sigma^2$ as before, whatever the LLM computes.

Wu et al. also draw a fresh Poisson sample of the exemplars for each query, at a sampling rate $r$, and split only the sample into groups. So each answer costs the subsampled Gaussian of [Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md), whose noise multiplier here is $\sigma/\sqrt2$. If $\varepsilon_{r,s}(\alpha)$ is that curve at rate $r$ and multiplier $s$, then $Q$ answers cost
$$\varepsilon(\alpha) = Q\,\varepsilon_{r,\,\sigma/\sqrt2}(\alpha).$$ {#eq-pate-icl}
Within one query the groups must still be disjoint, or one record would move two votes. Subsampling cuts the noise sharply, from 181 to 7.39 in the full edition's simulation of 1,000 answers at $\varepsilon = 1$.

In-context learning is where private prediction is strongest, because prompted LLMs agree with each other far more than small models do. Wu et al. report 91.97% sentiment accuracy at $\varepsilon = 1$, about as good as one non-private prompt. **PromptPATE** (Duan et al.) goes one step further, as PATE does. It labels a few public inputs privately and uses them as the exemplars of one public prompt, which then answers any number of queries at no further cost. Check whether a reported $\varepsilon$ for it is data-dependent before treating it as a guarantee.

## Code {#sec-pate-code}

The vote itself is a few lines of NumPy: count each label, add Gaussian noise to the counts and take the largest. Account for the answers with a maintained library rather than by hand. In Google's `dp_accounting`, one GNMax answer is a `GaussianDpEvent` whose noise multiplier is $\sigma/\sqrt2$. Compose $Q$ of them with `SelfComposedDpEvent`, or wrap each in a `PoissonSampledDpEvent` for subsampled in-context learning. Then read $\varepsilon$ off an `RdpAccountant` or a `PLDAccountant`. The full edition's script [private_prediction.py](../../code/part05-dp-ml/private_prediction.py) reproduces this chapter's numbers.

## Cheat sheet

| Situation | Use | Watch out for |
| --- | --- | --- |
| Few queries, small budget, or a model you cannot train privately | GNMax over teachers on hashed shards | Release the label only; noise grows like $\sqrt Q$ |
| Many queries, public unlabelled inputs available | PATE student | Teachers stay secret; tune the number of labels |
| Private exemplars in a prompt | Vote over disjoint groups, or PromptPATE | Never let one prompt with private exemplars answer users |
| Many queries, DP-SGD possible, $\varepsilon \ge 2$ | A private model | On the running example the vote never wins there |

: Which private-prediction tool for which situation; GNMax noise is $\sigma = \sqrt{Q/\rho(\varepsilon, \delta)}$. {#tbl-pate-cheat}

**Rules of thumb**

- Count your queries before choosing between private answers and a private model.
- Measure the noiseless vote first, because it caps every private answer.
- Add teachers as long as the vote gaps grow.
- Assign shards and exemplar groups by a hash, never by position.
- Report the data-independent $\varepsilon$, or pay for a private release of the data-dependent one.
- Fix the number of answers $Q$ in advance and stop answering when it is reached, since an open-ended query stream has no guarantee.

## Further reading

- Papernot, Song, Mironov, Raghunathan, Talwar & Erlingsson, "Scalable Private Learning with PATE" (ICLR 2018): GNMax, the data-dependent analysis and Confident-GNMax.
- Duan, Dziedzic, Papernot & Boenisch, "Flocks of Stochastic Parrots" (NeurIPS 2023), and Wu, Panda, Wang & Mittal, "Privacy-Preserving In-Context Learning for Large Language Models" (ICLR 2024): attacks on prompts, PromptPATE and subsampled voting.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/05-private-prediction.md).
