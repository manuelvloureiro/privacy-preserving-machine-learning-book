# PROSE.md

How to make, and prove, that a page of any of the four books, or a study note, reads well. [AGENTS.md](AGENTS.md) says what to write; this file says how to check that it is written clearly, and how to fix it when it is not. It applies to every book and note alike; only the thresholds differ (`tools/prose.toml`).

## Why this process exists

The first practitioner edition explained the right things in prose that was hard to read: long sentences that chained several claims with colons and semicolons, symbols used before the text said what they were, lists of facts packed into one sentence, and arguments squeezed into a clause. The writers had been asked to add reasoning without adding length, and the only way to do both was to compress. The measurements below show it: on the same topic, the practitioner chapter was harder to read than the full edition it was meant to simplify.

| Concentration inequalities | Mean sentence | Sentences flagged |
| --- | --- | --- |
| Full edition | 15.4 words | 14% |
| Practitioner edition, first version | 22.1 words | 49% |

: The same chapter in the two editions, measured by `tools/prose.py`. {#tbl-prose-why}

The first rewrite then overshot the other way. Its sentences were short and clear one at a time, but they came in runs, each starting afresh with no word to say how it related to the last, and the author found that the page read "in hiccups". Its mean sentence was 15 words, well under the limit. Splitting a long sentence is only half the fix. The other half is saying how the pieces connect.

Three lessons follow. Length must follow content: if a page is too long, cut repetition and secondary detail, never a topic and never the words that make a sentence readable. Short is not the same as readable, because a reader who must supply every *because* and *so* is doing the writer's work. And readability has to be checked, not assumed, because a page can be correct, complete and well organized and still be hard to read.

## The rules

These are the rules the checks and the reviews apply. Each comes from a standard source on clear technical writing.

1. **One idea per sentence.** A sentence carries one claim and, if needed, its reason, its consequence or its example. Splitting that claim from its reason is the opposite mistake (rule 9). Plain-language style guides aim for an average of about 15 words and a ceiling near 25 (Australian Government *Style Manual*); readers judge length against a benchmark of about 17 words (a 2023 *PLOS One* study). The books allow a little more because technical sentences name things: a mean of 20 words and a ceiling of 30 in the practitioner edition, 22 and 35 in the full edition and the exercise book.
2. **Old before new.** Begin a sentence with what the reader already has, the *topic position*, and end it with what is new, the *stress position*, so that each sentence picks up where the last one left off (Gopen & Swan, "The Science of Scientific Writing", 1990).
3. **Characters as subjects, actions as verbs.** Write "the mechanism adds noise", not "the addition of noise is performed". Avoid stacks of nominalizations (words ending in *-tion*, *-ment*, *-ness*), and get to the main verb quickly instead of opening with a long clause (Williams & Bizup, *Style: Lessons in Clarity and Grace*).
4. **Words before symbols.** Explain every symbol in words where it first appears. Never start a sentence with a formula. Use only the formulas and numbers the sentence needs (Halmos, "How to Write Mathematics": "resist symbols"; Bertsekas, "Ten Simple Rules for Mathematical Writing").
5. **Few new things at once.** Introduce at most three new concepts (terms, symbols, acronyms) per paragraph, and a concept before the result that uses it. Give an example before a generalization when you can.
6. **Explain, don't list.** Every result comes with what connects it to the page, in words: in the full edition and the exercise book, the reason it holds; in the practitioner edition, why it matters and how to use it, with at most a sentence on why it holds. Every condition a result needs (the quantity cannot be negative, the samples are independent, the values are bounded) is named as a condition, never stated as a fact about the quantity, and the text says what goes wrong without it.
7. **No compression.** Don't drop articles, verbs or connectives to save words. Don't chain clauses with colons and semicolons. Don't open with a list that a colon then gathers up.
8. **Plain words.** Prefer the common word, define jargon once, and keep terms consistent across the book.
9. **Connect the sentences.** Each sentence says how it relates to the one before. When one sentence is the reason, consequence, example or contrast of the last, say so: join the two with *because*, *so*, *while* or *but*, or open the second with *That*, *So*, *For example* or *Then*. Several short sentences in a row, each with a new subject and no link, read in hiccups. Vary the rhythm: a mix of short and medium sentences reads better than any one length (Williams & Bizup on cohesion; Gopen & Swan's topic position).
10. **Display the results.** A named result, a definition or a formula the reader must take in as a whole goes in display math on its own line, introduced by a sentence that names its symbols in words. Values and short arithmetic stay inline.

An example of rules 1, 4, 7 and 9 at work, from the concentration chapter, in three drafts:

> **Rejected, compressed.** A noisy answer, a measured error rate, the size of a sampled minibatch: each is random, and the practical question is how wide an error bar must be to fail with probability at most $\beta$.
>
> **Rejected, choppy.** Almost every number in a private system is random. The noise added to a count is random by design. An error rate measured on a test set depends on which examples happened to be in it. The number of examples in a randomly sampled minibatch changes from one step to the next. Before you can use any of these numbers, you need to know how far each one can stray from the value you care about.
>
> **Approved.** Almost every number in a private system is random. Some are random by design, like the noise a mechanism adds to a count. Others are random because of how they were measured. An error rate depends on which examples happened to land in the test set, and the size of a randomly sampled minibatch changes from one step to the next. Before you rely on any of these numbers, you need to know how far it can stray from the value you actually care about.

The first draft opens with a list that a colon gathers up, puts three claims in one sentence, and uses $\beta$ before saying what it is. The second fixes all three but lists the examples as four sentences that never say how they relate. The approved version sorts them into two kinds, random by design and random by measurement, and lets the sentences carry that sorting.

An example of rule 10:

> **Rejected.** For a quantity $X$ that cannot be negative, with average $\mathbb E[X]$, and any positive threshold $a$, Markov's inequality says that $\Pr[X \ge a] \le \mathbb E[X]/a$.
>
> **Approved.** In general, take a quantity $X$ that cannot be negative, and write $\mathbb E[X]$ for its average. For any positive threshold $a$, Markov's inequality says that
> $$\Pr[X \ge a] \le \frac{\mathbb E[X]}{a}.$$

Some failures are invisible to the analyzer, and only a reader catches them. An example of rule 6, from the same chapter:

> **Rejected.** Start with the weakest thing you could know about a quantity: it is never negative, and you know its average.
>
> **Approved.** Every bound in this chapter starts from Markov's inequality. It applies only to quantities that cannot be negative, such as a waiting time, a count or a squared error. [...] A quantity that can be negative can balance a large value with negative ones, and then its average says nothing about how often it is large.

The rejected sentence passes every mechanical check. But it states a condition of the result as if it were a fact about all quantities, and never says why the result needs it. Passages like this are kept in `REVIEW_ONLY` in [tests/test_prose.py](tests/test_prose.py), and the test checks that the review prompts still name the failure they illustrate.

## Rules for mathematics

The rules above make sentences readable. A page of mathematics can pass all of them and still be impossible to learn from, because its difficulty lies in the order of ideas: a theorem stated two sections before its proof, a section whose heading promises a result the reader has no reason to want, a display that packs three moves into one line and explains them afterwards. The author rejected a study note for exactly these faults. The rules below fix the order. They apply to every page with derivations, in every book and in `notes/`, and a reader catches their failures, not the analyzer.

11. **Open every section with its question.** The first paragraph of a section says what problem it solves, in words a reader without context understands, and why that reader needs the answer at this point: what earlier step left open, or what later step will use it. The heading names the topic or the question in plain words. It is never a compressed claim that only a reader who already knows the result can parse (Tao, "Motivate the paper": a brief paragraph at the start of each section describing its purpose, so that the reader always knows the near-term and long-term objectives; Knuth et al., rule 12: "What does the reader know so far? What does the reader expect next and why?").
12. **Keep a result with its proof.** A result appears in one place, in this order: the question it answers, the idea in plain words, the statement, the proof, then what it means and how good it is. The proof comes right after the statement, in the same section. A tool the proof needs (a lemma, an inequality) is stated just before the proof that uses it, with a sentence saying which step will need it. Never state a theorem in one section and prove it in a later one (Tao, "Organise the paper": the statement and proof of a lemma should be placed close to where it is used; Lee: state precisely what you are proving, then prove it).
13. **Give the plan before the proof.** A proof of more than three steps opens with a sentence giving its plan ("We bound the average loss, then show that the total rarely strays far above its average, then turn that into a guarantee"). Each step then opens with its goal, so that the reader knows what the algebra is for before reading it (Lee: if the structure of a proof is anything other than a simple direct proof, say at the beginning what type of proof it is; include motivation at the beginning of a proof and at the transitions between its parts).
14. **One move per display.** A display shows one move: one substitution, one simplification, or one inequality. The sentence before it says what the move is and what makes it valid, such as the fact, the assumption or the definition it uses. Never chain moves of different kinds in one display (simplify, then bound, then substitute), and never explain a step after the display that made it ("the first inequality is the triangle inequality"). A run of routine algebraic equalities may share a display (Knuth et al., rule 10: don't merely list a sequence of formulas, but tie them together with a running commentary; Lee: every statement in a proof must be justified).
15. **The page reads without its formulas.** Replace every formula except the simplest by "blah": the sentences must still say what is happening and why (Knuth et al., rule 13). A paragraph that is a formula with connecting words is a homework solution, not an explanation.
16. **Concrete before abstract, and one running example.** Before a general definition or theorem, show the smallest case with numbers. Carry one example through the page, such as the Laplace or Gaussian mechanism on a counting query, so that each new notion is first seen on something familiar (Tao, "Motivate the paper": first discuss a less technical special case or "toy" result).
17. **Introduce a lemma by its use.** Before a lemma or a side result, say which later step needs it and why. A result whose purpose the reader cannot see (a converse that fails, an exact identity) is either motivated in a sentence or cut. A detail obvious to the writer gets its sentence anyway (Tao, "Give appropriate amounts of detail": even a brief sentence of explanation is much better than none at all). Never write "clearly", "it is easy to see" or "by standard arguments".
18. **Notation earns its place and keeps its meaning.** Introduce a symbol only for something used three or more times, name it in words at its first use, and never give one symbol two meanings on a page. Which symbol a concept gets is not the page's choice: [NOTATION.md](NOTATION.md) fixes it for every book. A page with more than about eight symbols opens with a short table of them (Tao, "Use good notation"; Knuth et al., rule 14: don't use the same notation for two different things).

An example of rule 14, from the study note on concentrated privacy:

> **Rejected.** The normalizing constants cancel in the ratio, leaving
> $$\log\frac{p_x(y)}{p_{x'}(y)}=\frac{|y-f(x')|-|y-f(x)|}{b}\le\frac{|f(x)-f(x')|}{b}\le\frac{\Delta}{b}.$$
> The first inequality is the triangle inequality, $|u|-|v|\le|u-v|$ with $u=y-f(x')$ and $v=y-f(x)$.
>
> **Approved.** Both densities have the same normalizing constant, so it cancels, and the logarithm of a ratio of exponentials is the difference of the exponents:
> $$\log\frac{p_x(y)}{p_{x'}(y)}=\frac{|y-f(x')|-|y-f(x)|}{b}.$$
> The numerator compares how far the output lies from each of the two true answers. That difference can never exceed the distance between the answers themselves, by the triangle inequality: [...]

The rejected display computes the log-ratio and bounds it twice in one line, then names the reason for one of the bounds after the reader has passed it. An example of rules 11 and 12, from the same note: a section headed "From a tail bound on the loss to $(\varepsilon,\delta)$-DP, and why the converse fails" gave the reader no reason to want either result, and a later section stated the advanced composition theorem while its proof sat in the two sections after it. The approved version opens the section by saying that every later guarantee is proved by showing that large privacy losses are rare, and keeps the composition theorem, its plan and its proof in one section.

## The process

Every page goes through the same six stages, in order. A page is done when it passes every gate.

### 1. Write

Write to the rules and to [AGENTS.md](AGENTS.md). Read the reference chapter of the book first. When several writers work at once, each keeps its scratch files in its own folder, named after its pages; shared names such as `07-head.md` let writers overwrite each other's drafts, which happened once.

### 2. Measure: `tools/prose.py`

```bash
make prose BOOK=book-practitioner PAGE=part01-prerequisites/02-concentration.md   # report, worst paragraphs first
make prose-check BOOK=book-practitioner                                             # pass or fail, per page
uv run python tools/prose.py text "A sentence to try out, with $x$ in it."          # test a rewrite
```

The analyzer strips tables, figures, code, captions and reference lists, counts each formula as one word, and checks every sentence and paragraph against the book's limits in [tools/prose.toml](tools/prose.toml):

| Check | What it catches | Fix |
| --- | --- | --- |
| long sentence | more words than the book's limit | split it, one idea per sentence |
| overloaded | many commas, colons, semicolons, parentheses or dashes | split, or turn the aside into its own sentence |
| chained clauses | two or more colons and semicolons | one sentence per clause |
| front-loaded list before a colon | "A, B, C: each is ..." | give each item its own sentence, or name the group first |
| formula pile-up | more formulas than the limit in one sentence | keep the one the sentence needs; say the rest in words |
| number pile-up | more numbers than the limit in one sentence | keep the number that makes the point |
| starts with a formula | a sentence that opens with a symbol | open with words |
| new symbol not explained | a symbol's first use with no words naming it ("the noise multiplier $\sigma$", "where $n$ is the number of records") | name it where it first appears |
| new symbols in one sentence | several symbols introduced at once | introduce them one at a time |
| new concepts in one paragraph | more bold terms, new symbols and new acronyms than the limit | spread them over paragraphs, each with its explanation |
| long opening | many words before the main clause | put the main clause first |
| nominalizations | several *-tion*, *-ment*, *-ness* words in one sentence | turn them back into verbs |
| paragraph length | a paragraph over the limit | split at a change of idea |
| choppy | a run of short sentences, each starting afresh with no word linking it to the last | join the ones that are reason, consequence, example or contrast of each other, or open with a linking word |
| result in inline math | an inline formula that states a relation between probabilities, averages, sums or logarithms | display it on its own line |
| section grade | Flesch–Kincaid grade of a section above the limit | usually fixed by the sentence-level fixes |

The choppy and inline-result checks work on paragraphs and do not count toward the page's share of flagged sentences. A short sentence is often the right one, and a relation inside a derivation often belongs inline, so only a reader can tell. Treat it as a pointer for the reviews. A page fails the check if more than the book's share of its sentences are flagged (15% for the practitioner edition, 25% for the others) or if its mean sentence length is over the limit. The checks are deliberately simple and will sometimes flag a fine sentence; when a flag is wrong, leave the sentence, and if the same false flag recurs, fix the check and add the sentence to the anchors (stage 6). They cannot tell whether an explanation is right or whether a paragraph follows from the one before. Stages 3 and 4 do that.

### 3. Cold-reader test

Give the page, and nothing else, to a reviewer who has not seen the full edition or the writer's notes: a person, or a model in a fresh context with the prompt in [tools/prose-review/cold-reader.md](tools/prose-review/cold-reader.md). The reviewer restates each paragraph in one sentence and lists every stall: an undefined term, an unexplained jump, an overload, an ambiguous reference, an unmotivated result, a garbled sentence. Every stall is either fixed or, if it is the reviewer's misreading, noted as such. This is the only stage that measures understanding directly.

### 4. Editorial review

A reviewer with the page, the analyzer report, the cold-reader report and, for the practitioner edition, the full-edition chapter applies the rules paragraph by paragraph with the prompt in [tools/prose-review/editor.md](tools/prose-review/editor.md), and proposes rewrites that keep every claim and number. For the practitioner edition the reviewer also checks each rewrite against the full edition: a clearer sentence that claims more than the full edition supports is a new error. Apply the rewrites, then repeat stage 2.

### 5. Optional: register check

The `humanize-text` skill's fast pass (`bash ~/.claude/skills/humanize-text/run.sh analyze FILE`) reports patterns typical of generated text: stock phrases, uniform rhythm, empty transitions. Its readability numbers count formulas as words, so compare a page only with itself across drafts.

### 6. Author sign-off

The author reads at least one page per part, chosen from those the analyzer ranks worst, before the part is committed. When the author rejects a passage, add it to `REJECTED` in [tests/test_prose.py](tests/test_prose.py) with its approved rewrite in `APPROVED`. The test fails if the checks stop flagging the rejected passage or start flagging the approved one, so the checks stay calibrated to the author's judgement. Thresholds change only if the anchors still pass.

## Gates

A page is ready to commit when:

- `make prose-check BOOK=...` and `make structure-check BOOK=...` pass for it (the second checks the page's shape against its kind's skeleton in `tools/structure.toml`);
- every cold-reader stall is fixed or explained;
- every editorial rewrite is applied or rejected with a reason;
- for the practitioner edition, the correctness review against the full edition has no open items;
- `uv run pytest tests/test_prose.py` passes.

## Sources

- Gopen & Swan, "The Science of Scientific Writing", *American Scientist* 78(6), 1990: topic and stress positions, old before new.
- Williams & Bizup, *Style: Lessons in Clarity and Grace*: characters and actions, nominalizations, getting to the main verb.
- Halmos, "How to Write Mathematics" (1970): resist symbols, never start a sentence with one.
- Bertsekas, "Ten Simple Rules for Mathematical Writing": the mix of natural and mathematical language.
- Australian Government *Style Manual*, "Sentence length": an average of about 15 words, a ceiling of 25.
- A 2023 *PLOS One* study of how readers perceive sentence length (PubMed Central PMC9955962): the 17-word benchmark.
