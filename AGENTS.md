# AGENTS.md

How to write content for the full edition (`book-full/`) and its exercise book (`book-exercises/`). Every page of every book must also pass the prose process in [PROSE.md](PROSE.md): its rules, its checks (`make prose`) and its reviews. For how the book is built (Quarto, `.md` vs `.qmd`, cross-reference syntax, checks), see [CLAUDE.md](CLAUDE.md). The reference chapters are [01-distributions.md](book-full/part01-prerequisites/01-distributions.md) and [02-concentration.md](book-full/part01-prerequisites/02-concentration.md): when in doubt, do what they do.

## What the book is

A book on privacy-preserving machine learning that follows [book-full/study_plan.md](book-full/study_plan.md), one phase per part. Chapters build on each other, so each chapter may only use what earlier chapters introduced.

**Units.** The book uses Quarto's units. A **part** is a folder `book-full/partNN-<topic>/`. A **chapter** is one file in it; Quarto numbers chapters 1, 2, … across the whole book, so Part 4's fifth file is chapter 31. A **section** is a `##` heading in a chapter, numbered 31.2. Write "Part 3" or "this part" for a folder, "chapter" for a file and "section" for a `##` heading.

**The prerequisites part teaches mathematics, not privacy.** Do not introduce differential privacy, privacy mechanisms (Laplace, Gaussian, randomized response, ...), privacy definitions, privacy accounting, DP-SGD, privacy attacks (membership inference, reconstruction, ...), secure aggregation, MPC protocols or federated learning there. Motivate each topic with machine learning, statistics or computer-science examples instead. A short forward pointer is fine ("later chapters compare distributions this way"), as long as it names no mechanism or definition. Distributions such as Laplace and normal are fine as distributions, and objects such as per-record influence can be named by what they are.

## Files

- A part is a folder `book-full/partNN-<topic>/` with a `README.md` and one file per chapter, `NN-<topic>.md`, numbered in reading order. The `README.md` is the GitHub overview (readings, exit check) and is not part of the rendered book, so anything readers of the book need must live in a chapter file.
- Each part opens with `00-introduction.md`, listed in `book-full/_quarto.yml` as the part's `part:` file so it becomes the part page. It is one or two pages: what the part is for and what it assumes, the thread that ties its chapters together, a captioned table mapping each chapter to the question it answers, a reading order, shared notation, and what the reader will be able to do at the end. It does not follow the chapter structure below.
- A chapter's code lives in `code/<part>/<topic>.py` at the repository's root, shared by the three books. Its figures are written to `_build/figures/<part>/` and staged into the part's `figures/` folder, so a chapter references them as `figures/<name>.svg`.
- A chapter's exercises and their worked solutions live in `book-exercises/<part>/NN-<topic>.md`, with the same file name as the chapter. These pages form the **exercise book**, a separate book (website only) listed in `book-exercises/_quarto.yml`. The chapter pages themselves have no exercises. An exercise page may reference the full edition's equations, sections, figures and tables as `@eq-name` and so on: the build turns them into links to the full edition (`tools/xrefs.py`, `filters/xrefs.lua`). Numbers the exercise book quotes come from `code/<part>/solutions_<script>.py`, named after the chapter's script, and are checked by `tests/test_solutions_<script>.py`, like a chapter's.
- When you add, remove or renumber a chapter, update every link to it, the part's `README.md`, `book-full/README.md`, the chapter lists in `book-full/_quarto.yml` and `book-exercises/_quarto.yml`, the matching practitioner chapter, and the global chapter numbers quoted in the READMEs and `book-full/study_plan.md`.
- Each chapter's script has a test in `tests/` that runs it and checks every number the text quotes.

## Chapter structure

Every chapter follows the same shape:

1. **`# Title`**, without a number. Quarto numbers chapters and sections itself.
2. **`## Why this matters`**: what question the topic answers and why the reader needs it, then the one idea that organizes the whole chapter, then a table mapping each result to where it is used.
3. **`## Toolkit`** (when the derivations share facts): the few facts every derivation reuses, each stated and proved once.
4. **One `##` section per topic**, each with the same arc, under `###` headings:
   - **Context**: the question this result answers, in one paragraph.
   - **Statement**: the result, in display math.
   - **Derivation**: numbered steps. Each step does one thing and says which fact or assumption it uses.
   - **Interpretation**: what the result means, when it is tight, where it is loose.
   - **Example**: a concrete calculation with real numbers, from machine learning, statistics or computer science.
   - **Check yourself**: two questions, with the answer in italics in parentheses right after each.
5. **`## Code`**: what the script checks, the command to run it, the table of results, and what to take from them.
6. **`## References`**: standard textbooks and papers by author, title and year, with what each is good for.
7. **`## Cheat sheet`**: one table of every result in the chapter, plus a few rules of thumb.
8. **The pointer to the exercises**, the last line of the page: `**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/<part>/NN-<topic>.md).`

**Exercises** live only in the exercise book, never on the chapter page. They are problems that extend the derivations. Each names the technique it practises in italics and, where possible, gives a check value or a hint. On its exercise-book page, each exercise has a `##` heading with an id (`{#sec-sol-<prefix>-N}`), the statement as a blockquote starting `> **Exercise N.**`, then its solution inside `::: {.callout-tip collapse="true" title="Solution"}`: a derivation in numbered steps that reaches the check value. A chapter that mentions an exercise links to it, as `[Exercise 3](../../book-exercises/<part>/NN-<topic>.md#sec-sol-<prefix>-3)`. In the PDF the link becomes "Exercise 3 in the exercise book".

## Writing

The reader is a person reading one page at a time, not someone who holds the whole book in mind. Three rules follow from that, and they come before every other writing rule.

- **Logic before or with the math, always.** Before an equation, or in the same sentence, say in plain words what it says and why it should be true: the idea, the intuition, the reason this is the quantity to look at. Then give the formula and its derivation. Never let an equation appear first and be explained later, and never leave a step's reason implicit ("by symmetry", "it follows that", "this is the X pair") without saying what the symmetry or the step is. Name every symbol in words where it is used.
- **Every page stands on its own.** When a page uses a concept, a fact or a result from another page, restate it where it is used: what it means in a sentence, and, if a step depends on it, a short argument for why it holds. A link is a pointer for more detail, never a substitute for the explanation. It does not matter that the explanation also exists elsewhere.
- **Prose, not inventories.** Explain in connected paragraphs that say why each piece comes next. Use a list only when the items really are a list (the five questions a threat model answers), and never nest bullets of *Proof*, *Concretely*, *Use* under a fact.
- **Derive, don't assert.** Every result is either derived step by step or cited to a named source. If a derivation is too long, say what it needs and cite where it is done.
- **Make abstract moves concrete.** When a step is abstract (an indicator, an expectation over a conditional, a change of measure), walk through a small concrete case, such as a die or a coin, with a table showing each outcome. Readers stall on exactly these steps.
- **Every number is checked.** Numbers quoted in the text come from running the chapter's script, or from a calculation shown in the text. Never estimate or round from memory, and never copy a number without checking it.
- **Say how good each bound is.** Compare it with the exact value or with simulation, and say how loose it is and why.
- **Voice:** address the reader as "you". Write short, plain sentences. Bold the key term where it is defined. Avoid filler and hype.
- **Math:** LaTeX, with `$...$` inline and `$$...$$` for display. Display equations are numbered automatically. Label only the equations you refer back to, with `$$ ... $$ {#eq-name}`, and reference them as `@eq-name`.
- **Every table has a caption.** Put it on the line right after the table, as `: Caption text. {#tbl-name}`, and reference the table from the text as `@tbl-name`. The build warns about any table without one.
- **Links:** link to other chapters by relative path (`[01-distributions.md](01-distributions.md)`). Use `@fig-name`, `@tbl-name`, `@eq-name` and `@sec-name` for cross-references, never "the figure below", "the table above" or "equation (3)".

## Code

- **Fully type hinted.** Every function has annotated parameters and a return type. `uv run pyright` runs in strict mode and must report 0 errors.
- **Functions compute, loops plot.** Each bound, probability and simulation is a small pure function with typed inputs and outputs. A figure function calls those functions in a loop over a parameter range. It never re-implements the computation.
- **One script per chapter,** `code/<part>/<topic>.py`, with this layout:
  - typed pure functions, grouped under comment headers that match the text's sections;
  - `print_table(rng)`, which prints every number the text quotes, using `np.random.default_rng(seed)` with seed 0;
  - `plot_*(...) -> Figure` functions, which use a separate generator seeded with `seed + 1` so the table stays reproducible;
  - `main()`, which takes `--seed` and `--out`, runs `print_table`, and saves every figure.
- **Importable without side effects.** Nothing runs at import time except constants and matplotlib configuration. `.qmd` cells import these functions.
- **Standard library and NumPy for the math.** Use `math.erfc` for the normal CDF rather than SciPy, which has no type stubs here. Don't add dependencies without asking.
- **Use pure Python where it is clearer,** and NumPy where it is faster. Convert NumPy scalars to `float` at function boundaries.

## Figures

- **SVG only,** because it stays sharp at any size. Never PNG or JPG.
- **Generated, never committed.** Every figure is written by its chapter's script, into `_build/figures/`, never into a book's folder: the build regenerates them (`make figures`). A figure that no script produces cannot be in the book.
- **Deterministic output,** so the same code always draws the same figure. Anything random draws from a generator seeded from `--seed` (seed + 1 for figures), never from global or unseeded state. Set `matplotlib.rcParams["svg.hashsalt"]` to the module name and `svg.fonttype` to `"none"`, use the `Agg` backend, and save with `metadata={"Date": None}`.
- **Figures illustrate the text.** Draw the concept a section explains, not only the simulation checks: the die behind Markov's inequality, the Chernoff bound as a function of λ, two tails on a log scale. Place each figure in the section it illustrates, and reference it from the text.
- **Captions** are full sentences that say what is plotted. Use matplotlib mathtext for math in labels (`r"$e^{-t^2/2}$"`), not raw `^{...}`.
- **Look at every figure** before handing over a change that draws one. Check that labels are rendered, titles are not cut off, and the axis ranges show the point.

## Checks before handing over

```bash
uv run python code/<part>/<topic>.py   # numbers match the text; figures regenerate
uv run pytest
uv run pyright
make website   # fails on any warning: uncaptioned tables, unresolved @references, broken links
```

## The practitioner edition

`book-practitioner/` covers the same topics as the full edition, in the same parts and chapters with the same file names, for people who build, review and audit private machine-learning systems. It has no proofs. A reader should come away knowing what each result says, when it applies, how to use it, and what goes wrong in practice. The rules above apply except where this section changes them. The reference chapter is [01-definition.md](book-practitioner/part03-dp-foundations/01-definition.md).

**Explain, don't list.** This is the rule that matters most. A practitioner page is not a summary of the full chapter's results; it is an explanation of them. Every result comes with the reason it holds, in words: what problem it answers, why the answer has to look the way it does, what would go wrong otherwise. A paragraph that states five facts in five sentences, none of them justified, is the failure to avoid. Fewer results, each understood, beat more results listed. Keep a number only when the argument uses it. Warnings and pitfalls follow from the reasoning ("because the noise does not shrink with $n$, small subgroups are expensive"), not from a separate inventory.

**How much math.** It depends on the chapter. Keep exactly the math a practitioner needs to do the work: the formulas they calibrate, compute or check with (noise scales, composition, conversions, attack scores, error rates), each explained in words. Drop the math that only serves a proof. A chapter about a computation keeps its formulas; a chapter about a concept may need almost none.

**Length.** Length follows the explanation, never the other way round. A chapter keeps every topic of its full-edition chapter and drops only the proofs, which usually leaves a quarter to a third of the full chapter's words (about 2,500–3,500 for a typical chapter, less in Part 1, which keeps only the facts later parts use). A word limit that forces compression is the wrong limit: if a chapter runs long, merge or shorten topics, never the sentences that explain them ([PROSE.md](PROSE.md)).

**Chapter structure.**

1. **`# Title`**, the same title as the full edition's chapter.
2. **`## Why this matters`**: one to three paragraphs of prose: the practical question, and the one idea that answers it. No table, no list of the sections, no toolkit section.
3. **One `##` section per topic**, usually fewer than the full chapter has: merge sections that a practitioner uses together (the classic and exact Gaussian calibrations become one section), and reduce a theory-heavy one to a paragraph and a pointer to the full edition when it changes nothing in practice. Each section is connected prose that builds an argument, with no fixed subsections or bold lead-in labels: start from the problem, reason to the result (each formula introduced by saying in words what it says and why), carry the argument with a worked example with real numbers, and end with the consequences for practice, including what goes wrong and why.
4. **`## Code`**, only where code is the point: a short, self-contained snippet a reader can copy (standard library only, typed, each `print` line ending with its expected output as a comment, so `tests/test_practitioner_snippets.py` can run it), or the command that runs the full edition's script. Decide chapter by chapter.
5. **`## Cheat sheet`**: one table of the chapter's results and a few rules of thumb.
6. **`## Further reading`**: two to four references, with what each is good for.
7. **The pointer to the derivations**, the last line of the page: `**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/<part>/NN-<topic>.md).`

**What changes from the full edition's rules.**

- **No derivations.** State results and explain in words why they hold. "Derive, don't assert" becomes "explain, then cite": every result either is explained in words or cites its source, and the full edition has the derivation.
- **Recap, then link.** When a page uses a concept from another chapter, restate it in one sentence and link to it. Don't restate arguments.
- **No exercises and no check-yourself questions,** and no links to the exercise book.
- **Numbers come from the full edition's scripts.** Quote only numbers that the full edition's chapter quotes (so its test already checks them) or that a snippet on the page prints. Never compute a new number by hand.
- **Figures come from the full edition's scripts.** Reuse its figures by name (`figures/<name>.svg`); a figure the practitioner edition needs and the full edition lacks is added to the chapter's script.
- **Labels and links.** Use the full edition's label prefix for the same chapter (`sec-lg-`, `eq-lg-`, `fig-lg-`, `tbl-lg-`), which keeps labels unique across the book. Use `@`-references only within a chapter; refer to another chapter by a link to its file. Link to practitioner chapters for recaps, by their final path even before they are written (`filters/fallback-links.lua` sends the reader to the full edition until then), and to the full edition only for derivations.
- **Part pages.** Each part's `00-introduction.md` is one page of about 500–800 words of prose: what the part is for and who needs it, the thread that runs through its chapters and why they come in this order, and what the reader will be able to do at the end. No table of chapters.
