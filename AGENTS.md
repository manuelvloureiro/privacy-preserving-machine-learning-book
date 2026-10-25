# AGENTS.md

How to write content for the four books and the study notes. The rules below are the full edition's (`book-full/`) and its exercise book's (`book-exercises/`); the practitioner edition, the course and the notes each have a section at the end that says what changes for them. Every page must also pass the prose process in [PROSE.md](PROSE.md) (its rules, `make prose-check` and its reviews), write its symbols, names and spelling as [NOTATION.md](NOTATION.md) fixes them (`make notation-check`), and have the shape of its kind (`make structure-check`, which reads `tools/structure.toml`). How the books are built and checked is in [CLAUDE.md](CLAUDE.md). 

## What the book is

A book on privacy-preserving machine learning. Chapters build on each other, so each chapter may only use what earlier chapters introduced.

**Units.** The book uses Quarto's units. A **part** is a folder `book-full/partNN-<topic>/`. A **chapter** is one file in it; Quarto numbers chapters 1, 2, … across the whole book, so Part 4's fifth file is chapter 31. A **section** is a `##` heading in a chapter, numbered 31.2. Write "Part 3" or "this part" for a folder, "chapter" for a file and "section" for a `##` heading.

**The prerequisites part teaches mathematics, not privacy.** Do not introduce differential privacy, privacy mechanisms (Laplace, Gaussian, randomized response, ...), privacy definitions, privacy accounting, DP-SGD, privacy attacks (membership inference, reconstruction, ...), secure aggregation, MPC protocols or federated learning there. Motivate each topic with machine learning, statistics or computer-science examples instead. A short forward pointer is fine ("later chapters compare distributions this way"), as long as it names no mechanism or definition. Distributions such as Laplace and normal are fine as distributions, and objects such as per-record influence can be named by what they are.

## Files

- A part is a folder `book-full/partNN-<topic>/` with a `README.md` and one file per chapter, `NN-<topic>.md`, numbered in reading order. The `README.md` is the GitHub overview (readings, exit check) and is not part of the rendered book, so anything readers of the book need must live in a chapter file.
- Each part opens with `00-introduction.md`, listed in `book-full/_quarto.yml` as the part's `part:` file so it becomes the part page. It is one or two pages: what the part is for and what it assumes, the thread that ties its chapters together, a captioned table mapping each chapter to the question it answers, a reading order, shared notation, and what the reader will be able to do at the end. It does not follow the chapter structure below.
- A chapter's code lives in `code/<part>/<topic>.py` at the repository's root, shared by the full, exercise and practitioner books. Its figures are written to `_build/figures/<part>/` and staged into the part's `figures/` folder, so a chapter references them as `figures/<name>.svg`.
- A chapter's exercises and their worked solutions live in `book-exercises/<part>/NN-<topic>.md`, with the same file name as the chapter. These pages form the **exercise book**, a separate book (website only) listed in `book-exercises/_quarto.yml`. The chapter pages themselves have no exercises. An exercise page may reference the full edition's equations, sections, figures and tables as `@eq-name` and so on: the build turns them into links to the full edition (`tools/xrefs.py`, `filters/xrefs.lua`). Numbers the exercise book quotes come from `code/<part>/solutions_<script>.py`, named after the chapter's script, and are checked by `tests/test_solutions_<script>.py`, like a chapter's.
- When you add, remove or renumber a chapter, update every link to it, the part's `README.md`, `book-full/README.md`, the chapter lists in `book-full/_quarto.yml` and `book-exercises/_quarto.yml`, the matching practitioner chapter, and the global chapter numbers quoted in the READMEs.
- Each chapter's script has a test in `tests/` that runs it and checks every number the text quotes.

## Chapter structure

Every chapter has the same shape, which `make structure-check` keeps against `tools/structure.toml`. The reference chapter is [03-laplace-gaussian.md](book-full/part03-dp-foundations/03-laplace-gaussian.md): when in doubt, do what it does. Chapters listed under `legacy` in `tools/structure.toml` still have the earlier shape (a toolkit, six fixed sub-sections per topic, check-yourself questions, a cheat sheet) and are not models for new writing; converting one removes its line.

1. **`# Title`**, without a number. Quarto numbers chapters and sections itself.
2. **An opening of one to three paragraphs, with no heading**: the question the chapter answers, the one idea that organizes the answer, and the running example the chapter carries through, with its numbers. No table of contents, no list of prerequisites, no notation block; a captioned table of symbols only when the page uses more than about eight (PROSE.md rule 18).
3. **One `##` section per topic, in teaching order**, each with an id `{#sec-<prefix>-<name>}`. A section opens with the question it answers and why the reader needs it now, then runs through its results as numbered environments joined by prose: a sentence or two before each environment saying what it will say and why it should be true, the statement with every hypothesis named, the proof right after it in numbered steps (a plan sentence first when there are more than three), then at most two paragraphs on what the result means, when it is tight and where it is loose, and one worked example on the running example. A tool the proof needs is a lemma placed just before that proof, introduced by which step needs it. `###` headings are allowed for a sub-topic; they are never *Context*, *Statement*, *Derivation*, *Interpretation*, *Example* or *Check yourself*.
4. **`## Code`**: what the script checks, the command to run it, the captioned table of its results, and the test that checks them. No summary of the results in prose.
5. **`## Summary`**: one captioned table, `{#tbl-<prefix>-summary}`, each row a result of the chapter by `@`-reference, its statement in a line, and where it is used or the rule of thumb it gives.
6. **`## References`**: standard textbooks and papers by author, title and year, with what each is good for.
7. **The pointer to the exercises**, the last line of the page: `**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/<part>/NN-<topic>.md).`

A chapter has at most 8,000 words and no section over 2,500; the reference chapter is about 6,500. There is no toolkit section, no cheat sheet, no rules-of-thumb list and no check-yourself questions: a fact an earlier chapter proved is restated in one sentence where it is used, with a link; a fact this chapter is the first to prove is a lemma, stated just before the proof that uses it; what a cheat sheet or a rule of thumb would say goes in the Summary table's last column.

**Environments.** Definitions, results, proofs and examples use Quarto's numbered environments, written as fenced divs, exactly as the course does (see its section below): `#def-` for definitions, `#thm-` for theorems, `#lem-` for lemmas, `#prp-` for propositions, `#cor-` for corollaries and `#exm-` for worked examples, with `.proof` and `.remark` for unnumbered blocks and `name="..."` for a named result. Ids are `<type>-<prefix>-<name>` with the chapter's label prefix, and are referenced as `@thm-lg-laplace` from any page of the full edition and from the exercise book. Every theorem, lemma, proposition and corollary is followed by its `.proof`. When the proof is cited instead, or given elsewhere on the page, the line `<!-- proof: cited -->` (or `<!-- proof: in @sec-name -->`) follows the statement and the next sentence says what the proof needs and where it is done. Bold lead-ins such as `**Theorem.**` are not used.

**Exercises** live only in the exercise book, never on the chapter page. They are problems that extend the derivations, and an exercise never repeats a derivation the chapter gives: when a chapter result is the starting point, the solution cites it (`@thm-lg-laplace`) and does only the new step, and when a side result is derived in the exercise book, the chapter states it in a sentence and points to the exercise. A derivation lives in exactly one of the two places. Each exercise names the technique it practises in italics and, where possible, gives a check value or a hint. On its exercise-book page, each exercise has a `##` heading with an id (`{#sec-sol-<prefix>-N}`), the statement as a blockquote starting `> **Exercise N.**`, then its solution inside `::: {.callout-tip collapse="true" title="Solution"}`: a derivation in numbered steps that reaches the check value. A chapter that mentions an exercise links to it, as `[Exercise 3](../../book-exercises/<part>/NN-<topic>.md#sec-sol-<prefix>-3)`. In the PDF the link becomes "Exercise 3 in the exercise book".

## Appendices of derivations

The full edition ends with appendices of derivations, in `book-full/appendix-derivations/`, of two kinds. The first four derive the **core formulas** from scratch, for a reader who knows only probability: how much Laplace or Gaussian noise to add, what composition costs, how DP-SGD's $\varepsilon$ is computed, and what $\varepsilon$ means. The rest derive the results a chapter states but does not prove in full: a theorem it cites to a paper, a proof it sketches, a fact it takes as given. A chapter that would otherwise say "the proof is in X" keeps the citation and points to the appendix section ("Appendix @sec-apx-comp-fdp derives it step by step"). Each appendix groups results by mathematical theme, not by chapter, and is listed in both `_quarto-website.yml` and `_quarto-pdf.yml`, before the glossary and the index.

- **Shape of a core appendix.** An opening without a heading and a captioned table mapping each formula to the chapter sections that use it, then one `##` section per formula with the `###` headings *The question*, *The answer* (the formula in display math), *A worked example* (every number shown), *Why it works* (numbered steps, each a line of algebra and its reason), *Checking it* and *Recipe and pitfalls*. It restates every definition it uses and points to the chapter for the general version. The chapter does not derive the same special case a second time: it states and proves the general result once, and the sentence that introduces the theorem points to the appendix for the reader who wants the one-dimensional case from scratch.
- **Shape of the other appendices.** An opening without a heading says what the appendix derives and gives a captioned table mapping each derivation to the chapter sections that use it. Then one `##` section per result, labelled `sec-apx-<prefix>-<name>`, with these `###` headings: *What the chapter uses* (the result restated, where the chapter states it, and what it says about the proof), *The idea* (the whole argument in plain words, with a small concrete case for any abstract step), *Derivation* (numbered steps), *Checking it* (a numerical check and a figure, and how tight the result is) and, when needed, *What it does not cover* (the conditions, and any stronger version that stays cited). The page ends with `## Code` and `## References`. There are no exercises.
- **Plain English.** The appendices are for a reader who stalled on a cited step, so they use the simplest correct argument, restate every fact they use, and say each step's reason in words.
- **Prove what is stated.** When a short, correct proof covers only a special case (integer orders, pure DP, a strongly convex loss), prove that case and say what stays cited. Never present a sketch as a proof.
- **Code.** Each appendix has one script, `code/appendix-derivations/apx_<topic>.py`, with the same layout and rules as a chapter's, and a test, `tests/test_appendix_<topic>.py`. The script is self-contained: it imports nothing from the chapter scripts. Its figures go to `_build/figures/appendix-derivations/`, which staging copies into the appendix folder.

## Writing

The reader is a person reading one page at a time, not someone who holds the whole book in mind. Three rules follow from that, and they come before every other writing rule.

- **Logic before or with the math, always.** Before an equation, or in the same sentence, say in plain words what it says and why it should be true: the idea, the intuition, the reason this is the quantity to look at. Then give the formula and its derivation. Never let an equation appear first and be explained later, and never leave a step's reason implicit ("by symmetry", "it follows that", "this is the X pair") without saying what the symmetry or the step is. Name every symbol in words where it is used.
- **Every page stands on its own, in one sentence.** When a page uses a result from another page, say what it says in one sentence where it is used, with the symbols the step needs, and link to the page or `@`-reference the environment that proves it. Re-prove it only when the proof is the point of this page: the result is what this page derives, or the earlier proof has to change to apply here. Never re-prove an earlier chapter's result in a preamble, and never collect such facts in a toolkit.
- **Prose, not inventories.** Explain in connected paragraphs that say why each piece comes next. Use a list only when the items really are a list (the five questions a threat model answers), and never nest bullets of *Proof*, *Concretely*, *Use* under a fact.
- **Order the ideas for a learner.** Follow PROSE.md's rules for mathematics: every section opens with the question it answers and why the reader needs it now, a result stays in one place with its proof (question, idea, statement, proof, meaning), a long proof gives its plan first, and each display makes one move whose reason comes before it.
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
- **Standard library, NumPy and SciPy for the math,** with SciPy's stubs in `typings/`. Don't add dependencies without asking.
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
make terms-check notation-check prose-check structure-check BOOK=<book>   # acronyms, notation, prose, shape
make website   # fails on any warning: uncaptioned tables, unresolved @references, broken links
```

## The practitioner edition

`book-practitioner/` is for the people who build, review and audit private machine-learning systems: an **ML engineer** who trains and ships models, an **analytics engineer** who releases statistics and tables, and a **privacy reviewer** who checks someone else's claim. It covers every topic of the full edition, but it is organised around their work, and its code runs on real libraries and real public datasets. It is a practical book with minimal theory: no proofs, and no theory in words either. A reader should come away knowing what each result says, why it matters for their work, how to use it, and what goes wrong in practice. Why a result holds is the full edition's job. The rules above apply except where this section changes them. The reference chapter is [01-definition.md](book-practitioner/part03-dp-foundations/01-definition.md).

**Practice, not theory.** This is the rule that matters most. Each topic answers four questions, in prose: what the result says (the formula in display math, its symbols named in words), why it matters (the practical question it answers), how to use it (a recipe: what to plug in, what you get, how to solve it for the number you need, such as a width, a sample size, a noise scale or an $\varepsilon$, carried by a worked example with real numbers), and what goes wrong (the conditions it needs and what happens when they fail). Why the result holds gets at most a sentence of intuition, and only when it helps the reader use it correctly; proof ideas, mechanisms of arguments ("apply Markov to $e^{\lambda X}$"), and refinements that change no decision go to the full edition. Still explain, don't list: a paragraph of five unconnected facts is a failure here too, but the connections are practical ones (when to use which, what it costs, what breaks), not proof steps.

**How much math.** It depends on the page. Keep exactly the math a practitioner needs to do the work: the formulas they calibrate, compute or check with (noise scales, composition, conversions, attack scores, error rates), each explained in words. Drop the math that only serves a proof.

### Structure: three kinds of page

The book keeps the full edition's parts and file names, so labels and the derivation pointers keep working. Within that frame each page is one of three kinds, and the page shows which: a reference page opens with `## What it is`, a workflow chapter is numbered after the full edition's chapters of its part and ends with a checklist, and a full chapter has a cheat sheet.

- **Full chapter** (about 2,000–2,150 words): a topic a practitioner uses directly, with the chapter structure below.
- **Reference page** (about 600–900 words): a theory-heavy topic that a practitioner meets but rarely computes with. It has `# Title`, then, in prose and without fixed headings beyond these, `## What it is` (the result in a sentence or two and its formula, symbols named in words), `## When you meet it` (the situation, the library or paper where it shows up, and the decision it changes, if any), `## How to use it` (the formula or the library call, with one worked example or a short snippet), `## What goes wrong` (the pitfall, and how to spot it), then the derivation pointer. No cheat sheet; a reference page may end with one or two references under `## References`.
- **Workflow chapter** (about 2,500–3,500 words): a job done end to end with real libraries on a real public dataset, such as training a model with DP-SGD, releasing a table, or assessing a model's privacy risk before release. It is a new file in the part, numbered after the part's existing files (`12-release-governance.md`), and listed in `book-practitioner/_quarto.yml` where it belongs in reading order. It has `# Title`, `## Why this matters`, one `##` section per step of the job (each step: the decision, its default and why, the call, what it prints, and how to tell it went wrong), `## Checklist` (a table of the steps, their defaults and the check for each), `## References`, and a last line pointing to the full-edition chapters it draws on (or none, if it has no counterpart). It uses its own label prefix, one no full-edition chapter uses.

A workflow chapter quotes the results it relies on in a sentence each and links to the page that explains them; it does not re-teach them. A full chapter or reference page that a workflow supersedes keeps its topics and points to the workflow for the job.

Part 1 is maths only, under the prerequisites rule above, and consists of reference pages except [04-hypothesis-testing.md](book-practitioner/part01-prerequisites/04-hypothesis-testing.md), the full chapter the rest of the book links to most. The DP-SGD operating rules (per-example clipping, step size after noise, batch size against noise) live in Part 5, not Part 1.

### Length

The whole practitioner book is 250–300 PDF pages, at most about 105,000 words (measured: about 360 words per PDF page, including code and tables), with about 380 words per part introduction. Check the page count with `pdfinfo` after `make practitioner-pdf`. New content is paid for by cuts elsewhere. Every topic of the full chapter still appears, but a minor one gets a paragraph or a sentence and a pointer to the full edition. The content is distinct from the full edition, not a condensed copy: it is organised around what a practitioner does (decision tables, recipes, library calls, defaults, checklists, pitfalls). Stay short by choosing what to say, never by compressing sentences ([PROSE.md](PROSE.md)): cut theory, repetition, secondary examples and recaps of other chapters.

### Chapter structure (full chapters)

1. **`# Title`**, the same title as the full edition's chapter.
2. **`## Why this matters`**: one to three paragraphs of prose: the practical question, and the one idea that answers it. No table, no list of the sections, no toolkit section.
3. **One `##` section per topic**, usually fewer than the full chapter has: merge sections that a practitioner uses together, and reduce a theory-heavy one to a paragraph and a pointer to the full edition when it changes nothing in practice. Each section is connected prose with no fixed subsections or bold lead-in labels: start from the practical problem, state the result (each formula introduced by saying in words what it says), show how to use it with a worked example with real numbers, and end with what goes wrong and how to avoid it.
4. **`## Code`**, only where code is the point (see the code rules below).
5. **`## Cheat sheet`**: one table of the chapter's results and a few rules of thumb.
6. **`## References`**: two to four references, with what each is good for.
7. **The pointer to the derivations**, the last line of the page: `**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/<part>/NN-<topic>.md).`

### Code and data

The practitioner edition's code runs on the libraries a practitioner uses: Opacus and PyTorch (CPU build), `dp-accounting`, OpenDP (with Polars), diffprivlib, `peft`, `transformers`, scikit-learn and SciPy, pinned exactly in the `practitioner` dependency group of `pyproject.toml`, which `uv` installs by default. Every piece of code on a page has been run, and a test checks that it still runs and prints what the page says. Load Hugging Face models with `dtype=torch.float32`: transformers loads some checkpoints in float16, which is about sixty times slower on a CPU.

- **Real, public data only.** No synthetic toy data stands in for a real dataset in a workflow. Prefer the datasets bundled with scikit-learn (breast cancer, diabetes, digits), which need no download. Next, use the public sets that [code/practitioner_datasets.py](code/practitioner_datasets.py) loads: Adult (UCI 2, the same 48,842 rows as OpenML 1590), Online Retail (UCI 352, many invoice lines per customer, for contribution bounding), Tiny Shakespeare and the Pythia-14m model (for fine-tuning a language model). OpenML is not reachable from the build machine, so use the UCI copies. Add a dataset there, with its source, version and licence in the docstring, rather than downloading inside a script. Choose the smallest dataset that makes the point, so that it runs on a CPU. Name the dataset in the page's prose, and record its source, its version or OpenML ID and its licence in the script's docstring. A short snippet that illustrates one call (a count with noise, a conversion) uses a bundled scikit-learn dataset or a handful of literal numbers, so that it runs on its own; anything that needs a downloaded dataset is a workflow script.
- **Downloads are cached outside the sources.** Scripts fetch into `data/` (git-ignored) and work offline once it is filled; `make datasets` fills it, and `make test` runs it first. Never commit a dataset.
- **Snippets.** A ```` ```python ```` block on a page is a snippet: it runs on its own (imports included), and every `print` line ends with its expected output as a comment, `print(f"{x:.4f}")  # 3.7306`. `tests/test_practitioner_snippets.py` runs it and compares. A snippet that imports a library from the `practitioner` group runs under the `libs` pytest marker. Keep snippets to seconds of CPU time.
- **Workflow scripts.** A job too long for a snippet lives in `code/<part>/practitioner_<topic>.py`. It imports its data from `practitioner_datasets` (after `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`), it is typed and passes strict pyright, it is seeded (seed 0 for the numbers a page quotes; `torch.manual_seed` as well as NumPy), it has `main()` with `--seed`, and a test in `tests/test_practitioner_<topic>.py` (marked `libs`) runs it and checks every number the page quotes. If a run is too slow for the test suite (aim for under a minute per script), shrink it (fewer epochs, a subset), say so on the page, and test the reduced run. Libraries that ship without type information (Opacus, dp-accounting, diffprivlib, scikit-learn, SciPy) are typed by the hand-written stubs in `typings/`; when a script calls something new, add it there with the library's real signature rather than casting in the script. The workflow scripts draw no figures and are not run by `make figures`. OpenDP's noise cannot be seeded, so a page quotes only OpenDP's deterministic outputs (scales, accuracies, thresholds) in prose, shows released values in a `text` block marked as one run, and the test checks those against the true values within their stated accuracy.
- **Excerpts.** A page never shows a whole script. It shows an excerpt: a ```` ```{.python excerpt="<name>"} ```` block of at most about 15–20 lines with only the calls that matter, such as the library setup, the one call that does the work, and what it prints. Mark the region in the script with a `# excerpt: <name>` line before it and a `# end excerpt` line after it. `tests/test_practitioner_excerpts.py` checks that every excerpt block on a page is, line for line, the region of that name in some practitioner script (ignoring common indentation), and that no other code block on a page is anything but a snippet, a `text` block or a `bash` command. So an excerpt can never drift from code that has been run. Link the script by its relative path (`../../code/<part>/practitioner_<topic>.py`, which `filters/links.lua` turns into a GitHub link) and give the one command that runs it, `uv run python code/<part>/practitioner_<topic>.py`.
- **No long code blocks.** No code block on a page runs much past 20 lines. Output a reader should compare against goes in a ```` ```text ```` block, copied from a run that the test checks.
- **Libraries that cannot run here.** A library that cannot be installed in the test environment, or is too slow on a CPU (Tumult Analytics, which needs Spark; JAX Privacy), is shown as a ```` ```text ```` block that names the pinned version, with a sentence saying it was not run.

### Numbers

Quote a number only if a test checks it: a number the full edition's chapter quotes (its test already checks it), one a snippet on the page prints, or one a practitioner workflow script prints with seed 0 and its test checks. Never compute a new number by hand, and never quote a runtime, an accuracy or an $\varepsilon$ you have not seen code print. Use one default for $\delta$ throughout: $\delta \le 1/n^{1.1}$ for $n$ records (or users, at user level), rounded down to a power of ten, and say so where a page chooses $\delta$.

### Other rules that change

- **No derivations, and no theory in their place.** "Derive, don't assert" becomes "state, use, cite": state the result, show how to use it, and point to the full edition for why it holds.
- **Recap, then link.** When a page uses a concept from another chapter, restate it in one sentence and link to it. Don't restate arguments.
- **No exercises and no check-yourself questions,** and no links to the exercise book.
- **Figures come from scripts.** Reuse the full edition's figures by name (`figures/<name>.svg`); a figure the practitioner edition needs and the full edition lacks is added to the chapter's script, or drawn by a practitioner script into `_build/figures/<part>/`.
- **Labels and links.** Use the full edition's label prefix for the same chapter (`sec-lg-`, `eq-lg-`, `fig-lg-`, `tbl-lg-`), which keeps labels unique across the book; a workflow chapter uses its own. Use `@`-references only within a chapter; refer to another chapter by a link to its file. Link to practitioner chapters for recaps, and to the full edition only for derivations.
- **Part pages.** Each part's `00-introduction.md` is one page of about 400 words of prose: what the part is for and who needs it, the thread that runs through its chapters and why they come in this order, which pages are workflows and which are reference pages, and what the reader will be able to do at the end. No table of chapters.

## The course

`book-course/` is a course in differential privacy, one topic per chapter, for a reader who knows probability and wants the subject from first principles. It is light in text and complete in its mathematics: every result is stated precisely and proved, and the prose says only what a reader needs to follow the proofs and to see why each result comes next. It is a separate book with its own parts, chapters and labels, and it does not depend on the other three. The rules above apply except where this section changes them. The reference chapter is [01-randomized-response.md](book-course/part02-dp-basics/01-randomized-response.md).

**Self-contained.** The course links to no other book. Each chapter may use what earlier chapters of the course proved, and refers to it by link or `@`-reference with a one-sentence restatement of the result.

### Chapter structure

1. **`# Title`**, without a number.
2. **An opening of one or two short paragraphs, with no heading**: the question the chapter answers and how it answers it.
3. **One `##` section per topic, in teaching order**, each with an id (`{#sec-<prefix>-<name>}`). A section is a short run of numbered environments joined by prose: a sentence or two before each to say why it comes next, and a sentence after a theorem when its meaning or tightness is not obvious. `###` headings are allowed for a sub-topic.
4. **`## Summary`**: one captioned table, each row a result of the chapter (by `@`-reference) and its statement in a line.
5. **`## References`**: the papers and books the results come from, by author, title and year, each with one sentence on what it contains.

There are no exercises, no check-yourself questions, no cheat sheet and no toolkit section.

### Environments

Definitions, results, proofs and examples use Quarto's numbered environments, written as fenced divs:

```markdown
::: {#thm-lap-privacy name="Laplace mechanism"}
Statement, with display math.
:::

::: {.proof}
1. One step, with the fact it uses.
2. ...
:::
```

Use `#def-` for definitions, `#thm-` for theorems, `#lem-` for lemmas, `#cor-` for corollaries, `#prp-` for propositions and `#exm-` for worked examples, and `.proof` and `.remark` for unnumbered blocks. Give `name="..."` to a named result. The chapter's label prefix follows the type (`thm-rr-privacy`, `sec-rr-definition`, `eq-rr-estimator`), and every prefix is unique in the course. Reference an environment as `@thm-rr-privacy`.

### Rigour

- **Prove what is stated.** Every theorem, lemma and proposition has a proof: complete, in numbered steps, each step naming the fact or assumption it uses. A proof that is long and standard may be cited instead ("the proof is in Dwork and Roth, 2014, Theorem 3.20"), but then the text says what the proof needs and how it goes in a sentence or two. Never present a sketch as a proof.
- **Exact constants.** Prefer exact statements to $O(\cdot)$ statements: give the constant a proof produces, then the asymptotic form. When a result is stated only up to constants, say so.
- **Every hypothesis named.** A condition a result needs (bounded sensitivity, independence, $\varepsilon \le 1$, a convex loss) appears in the statement, not in the prose around it.
- **Every number checked.** A number in an example comes from the chapter's script, which a test checks, or from arithmetic shown in the text.

### Prose

The prose rules of [PROSE.md](PROSE.md) apply, with the course's thresholds in `tools/prose.toml`. Light in text means few sentences, not compressed ones: cut motivation that repeats, history and asides, and keep the words that say what a symbol is and why a step holds.

### Code and figures

A chapter has at most one script, `code/<part>/<topic>.py`, with the layout and rules of a full-edition script, and a test `tests/test_course_<topic>.py` that runs it and checks every number the chapter quotes. A chapter with no computed numbers and no figure has no script. Figures are optional: draw one only where a picture makes a definition or a bound easier to grasp, such as two neighbouring output densities or an error curve against simulation.

## Study notes

`notes/<name>/` holds standalone study notes, one page each, built by `make notes` (see [CLAUDE.md](CLAUDE.md)). A note is for a reader who knows probability but has no other context: it links to no book, defines everything it uses, and follows the Writing rules above and every rule in [PROSE.md](PROSE.md), with the full edition's thresholds. It opens with the question it answers, a table mapping each section to the question it answers, and a table of its symbols, which follows [NOTATION.md](NOTATION.md) even where the note's sources write otherwise. Results use Quarto's numbered environments, as in the course (`#def-`, `#thm-`, `#lem-`, `.proof`), and each section follows PROSE.md's rules for mathematics. Every number is computed in the text or by a script in `code/notes/`.
