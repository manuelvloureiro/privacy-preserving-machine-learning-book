# AGENTS.md

How to write content for this book. For how the book is built (Quarto, `.md` vs `.qmd`, cross-reference syntax, checks), see [CLAUDE.md](CLAUDE.md). The reference chapters are [01-distributions.md](book/part01-prerequisites/01-distributions.md) and [02-concentration.md](book/part01-prerequisites/02-concentration.md): when in doubt, do what they do.

## What the book is

A book on privacy-preserving machine learning that follows [book/study_plan.md](book/study_plan.md), one phase per part. Chapters build on each other, so each chapter may only use what earlier chapters introduced.

**Units.** The book uses Quarto's units. A **part** is a folder `book/partNN-<topic>/`. A **chapter** is one file in it; Quarto numbers chapters 1, 2, … across the whole book, so Part 4's fifth file is chapter 31. A **section** is a `##` heading in a chapter, numbered 31.2. Write "Part 3" or "this part" for a folder, "chapter" for a file and "section" for a `##` heading.

**The prerequisites part teaches mathematics, not privacy.** Do not introduce differential privacy, privacy mechanisms (Laplace, Gaussian, randomized response, ...), privacy definitions, privacy accounting, DP-SGD, privacy attacks (membership inference, reconstruction, ...), secure aggregation, MPC protocols or federated learning there. Motivate each topic with machine learning, statistics or computer-science examples instead. A short forward pointer is fine ("later chapters compare distributions this way"), as long as it names no mechanism or definition. Distributions such as Laplace and normal are fine as distributions, and objects such as per-record influence can be named by what they are.

## Files

- A part is a folder `book/partNN-<topic>/` with a `README.md` and one file per chapter, `NN-<topic>.md`, numbered in reading order. The `README.md` is the GitHub overview (readings, exit check) and is not part of the rendered book, so anything readers of the book need must live in a chapter file.
- Each part opens with `00-introduction.md`, listed in `book/_quarto.yml` as the part's `part:` file so it becomes the part page. It is one or two pages: what the part is for and what it assumes, the thread that ties its chapters together, a captioned table mapping each chapter to the question it answers, a reading order, shared notation, and what the reader will be able to do at the end. It does not follow the chapter structure below.
- A chapter's code lives in `code/<topic>.py` and its figures in `figures/<name>.svg`, both inside the part's folder.
- A chapter's exercises and their worked solutions live in `solutions/NN-<topic>.md` inside the part's folder, with the same file name as the chapter. These pages form the **exercise book**, an appendix of the website only (not the PDF), listed in `book/_quarto-website.yml` under the "Exercises" part and in `index_skip_pages` in `book/terms.toml`. The chapter pages themselves have no exercises. Numbers the exercise book quotes come from `code/solutions_<script>.py`, named after the chapter's script, and are checked by `tests/test_solutions_<script>.py`, like a chapter's.
- When you add, remove or renumber a chapter, update every link to it, the part's `README.md`, `book/README.md`, the chapter list in `book/_quarto.yml` and the global chapter numbers quoted in the READMEs and `book/study_plan.md`.
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
8. **The pointer to the exercises**, the last line of the page: `**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](solutions/NN-<topic>.md).`

**Exercises** live only in the exercise book, never on the chapter page. They are problems that extend the derivations. Each names the technique it practises in italics and, where possible, gives a check value or a hint. On its exercise-book page, each exercise has a `##` heading with an id (`{#sec-sol-<prefix>-N}`), the statement as a blockquote starting `> **Exercise N.**`, then its solution inside `::: {.callout-tip collapse="true" title="Solution"}`: a derivation in numbered steps that reaches the check value. A chapter that mentions an exercise links to it, as `[Exercise 3](solutions/NN-<topic>.md#sec-sol-<prefix>-3)`. In the PDF the link becomes "Exercise 3 in the web edition".

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
- **One script per chapter,** `code/<topic>.py`, with this layout:
  - typed pure functions, grouped under comment headers that match the text's sections;
  - `print_table(rng)`, which prints every number the text quotes, using `np.random.default_rng(seed)` with seed 0;
  - `plot_*(...) -> Figure` functions, which use a separate generator seeded with `seed + 1` so the table stays reproducible;
  - `main()`, which takes `--seed` and `--out`, runs `print_table`, and saves every figure.
- **Importable without side effects.** Nothing runs at import time except constants and matplotlib configuration. `.qmd` cells import these functions.
- **Standard library and NumPy for the math.** Use `math.erfc` for the normal CDF rather than SciPy, which has no type stubs here. Don't add dependencies without asking.
- **Use pure Python where it is clearer,** and NumPy where it is faster. Convert NumPy scalars to `float` at function boundaries.

## Figures

- **SVG only,** because it stays sharp at any size. Never PNG or JPG.
- **Generated, never committed.** Every figure is written by its chapter's script, and `book/*/figures/*.svg` is git-ignored: the build regenerates them (`make figures`). A figure that no script produces cannot be in the book.
- **Deterministic output,** so the same code always draws the same figure. Anything random draws from a generator seeded from `--seed` (seed + 1 for figures), never from global or unseeded state. Set `matplotlib.rcParams["svg.hashsalt"]` to the module name and `svg.fonttype` to `"none"`, use the `Agg` backend, and save with `metadata={"Date": None}`.
- **Figures illustrate the text.** Draw the concept a section explains, not only the simulation checks: the die behind Markov's inequality, the Chernoff bound as a function of λ, two tails on a log scale. Place each figure in the section it illustrates, and reference it from the text.
- **Captions** are full sentences that say what is plotted. Use matplotlib mathtext for math in labels (`r"$e^{-t^2/2}$"`), not raw `^{...}`.
- **Look at every figure** before handing over a change that draws one. Check that labels are rendered, titles are not cut off, and the axis ranges show the point.

## Checks before handing over

```bash
uv run python book/<part>/code/<topic>.py   # numbers match the text; figures regenerate
uv run pytest
uv run pyright
make website   # fails on any warning: uncaptioned tables, unresolved @references, broken links
```
