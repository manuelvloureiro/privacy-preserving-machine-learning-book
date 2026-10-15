# PROSE.md

How to make, and prove, that a page of any of the three books reads well. [AGENTS.md](AGENTS.md) says what to write; this file says how to check that it is written clearly, and how to fix it when it is not. It applies to the full edition, the exercise book and the practitioner edition alike; only the thresholds differ.

## Why this process exists

The first practitioner edition explained the right things in prose that was hard to read: long sentences that chained several claims with colons and semicolons, symbols used before the text said what they were, lists of facts packed into one sentence, and arguments squeezed into a clause. The writers had been asked to add reasoning without adding length, and the only way to do both was to compress. The measurements below show it: on the same topic, the practitioner chapter was harder to read than the full edition it was meant to simplify.

| Concentration inequalities | Mean sentence | Sentences flagged |
| --- | --- | --- |
| Full edition | 15.4 words | 14% |
| Practitioner edition, first version | 22.1 words | 49% |

: The same chapter in the two editions, measured by `tools/prose.py`. {#tbl-prose-why}

Two lessons follow. Length must follow content: if a page is too long, cut a topic, never the words that make a sentence readable. And readability has to be checked, not assumed, because a page can be correct, complete and well organized and still be hard to read.

## The rules

These are the rules the checks and the reviews apply. Each comes from a standard source on clear technical writing.

1. **One idea per sentence.** A sentence carries one claim and, if needed, its reason. Plain-language style guides aim for an average of about 15 words and a ceiling near 25 (Australian Government *Style Manual*); readers judge length against a benchmark of about 17 words (a 2023 *PLOS One* study). The books allow a little more because technical sentences name things: a mean of 20 words and a ceiling of 30 in the practitioner edition, 22 and 35 in the full edition and the exercise book.
2. **Old before new.** Begin a sentence with what the reader already has, the *topic position*, and end it with what is new, the *stress position*, so that each sentence picks up where the last one left off (Gopen & Swan, "The Science of Scientific Writing", 1990).
3. **Characters as subjects, actions as verbs.** Write "the mechanism adds noise", not "the addition of noise is performed". Avoid stacks of nominalizations (words ending in *-tion*, *-ment*, *-ness*), and get to the main verb quickly instead of opening with a long clause (Williams & Bizup, *Style: Lessons in Clarity and Grace*).
4. **Words before symbols.** Explain every symbol in words where it first appears. Never start a sentence with a formula. Use only the formulas and numbers the sentence needs (Halmos, "How to Write Mathematics": "resist symbols"; Bertsekas, "Ten Simple Rules for Mathematical Writing").
5. **Few new things at once.** Introduce at most three new concepts (terms, symbols, acronyms) per paragraph, and a concept before the result that uses it. Give an example before a generalization when you can.
6. **Explain, don't list.** Every result comes with the reason it holds, in words. Every condition a result needs (the quantity cannot be negative, the samples are independent, the values are bounded) is named as a condition, never stated as a fact about the quantity, and the text says what goes wrong without it.
7. **No compression.** Don't drop articles, verbs or connectives to save words. Don't chain clauses with colons and semicolons. Don't open with a list that a colon then gathers up.
8. **Plain words.** Prefer the common word, define jargon once, and keep terms consistent across the book.

An example of rules 1, 4 and 7 at work, from the concentration chapter:

> **Rejected.** A noisy answer, a measured error rate, the size of a sampled minibatch: each is random, and the practical question is how wide an error bar must be to fail with probability at most $\beta$.
>
> **Approved.** Almost every number in a private system is random. The noise added to a count is random by design. An error rate measured on a test set depends on which examples happened to be in it. [...] A concentration inequality tells you how wide to make an interval around your estimate so that the true value falls outside it only rarely, say once in a thousand runs.

The rejected sentence opens with a list that a colon gathers up, puts three claims in one sentence, and uses $\beta$ before saying what it is. The approved version gives each example its own sentence and states the question in words.

Some failures are invisible to the analyzer, and only a reader catches them. An example of rule 6, from the same chapter:

> **Rejected.** Start with the weakest thing you could know about a quantity: it is never negative, and you know its average.
>
> **Approved.** Every bound in this chapter starts from Markov's inequality. It applies only to quantities that cannot be negative, such as a waiting time, a count or a squared error. [...] A quantity that can be negative can balance a large value with negative ones, and then its average says nothing about how often it is large.

The rejected sentence passes every mechanical check. But it states a condition of the result as if it were a fact about all quantities, and never says why the result needs it. Passages like this are kept in `REVIEW_ONLY` in [tests/test_prose.py](tests/test_prose.py), and the test checks that the review prompts still name the failure they illustrate.

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
| section grade | Flesch–Kincaid grade of a section above the limit | usually fixed by the sentence-level fixes |

A page fails the check if more than the book's share of its sentences are flagged (15% for the practitioner edition, 25% for the others) or if its mean sentence length is over the limit. The checks are deliberately simple and will sometimes flag a fine sentence; when a flag is wrong, leave the sentence, and if the same false flag recurs, fix the check and add the sentence to the anchors (stage 6). They cannot tell whether an explanation is right or whether a paragraph follows from the one before. Stages 3 and 4 do that.

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

- `make prose-check BOOK=... ` passes for it;
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
