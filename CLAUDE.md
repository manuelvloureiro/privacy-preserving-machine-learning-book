# CLAUDE.md

How the books are built. For how to write book content (chapter structure, voice, derivations, code and figure rules), see [AGENTS.md](AGENTS.md).

## Build: Quarto

The repository holds three [Quarto](https://quarto.org) book projects side by side, and they share `code/`, `filters/` and `terms.toml`:

- `book-full/` (`book-full/_quarto.yml`): the full edition, with every derivation. Website and PDF.
- `book-exercises/` (`book-exercises/_quarto.yml`): the full edition's exercises with worked solutions, one page per chapter. Website only.
- `book-practitioner/` (`book-practitioner/_quarto.yml`): the practitioner edition, the same parts and chapters without proofs. Website and PDF.

Do not add another build system (MkDocs, Jupyter Book, a hand-rolled pandoc script).

```bash
make website   # every website, into _book/book-full/, _book/book-exercises/, _book/book-practitioner/
make pdf       # the full and practitioner PDFs, into _book/
make book      # both
make full      # one book: full, full-pdf, exercises, practitioner, practitioner-pdf
make preview   # serve the full edition from its staged copy (rerun after editing book-full/)
```

Never render a book's folder in place. The make targets copy it to `_build/<name>-<profile>/` (for example `_build/full-website/`), add `filters/` and the figures, and render there, so Quarto's cache, LaTeX's intermediate files and the generated glossary and index never land among the sources. The rendered website is then copied to `_book/<book>/`, the PDF to `_book/`. A render fails on any Quarto warning or error, so a clean build has none.

**Links between the books.** A page links to another book, to `code/` or to `tests/` by its relative path in the repository (`../../book-exercises/part03-dp-foundations/03-laplace-gaussian.md#sec-sol-lapgauss-2`), so the links also work when browsing the source. `filters/links.lua` rewrites them for the output: on a website a link to another book points to its page there, since `_book/` mirrors the repository's layout; in a PDF it becomes plain text followed by "in the exercise book", "in the full edition" or "in the practitioner edition". The exercise book also cross-references the full edition's equations, sections, figures and tables as `@eq-name` and so on, which Quarto resolves only within one book: `make exercises` therefore builds the full edition first, `tools/xrefs.py` reads its labels and numbers into the staged `_xrefs.json`, and `filters/xrefs.lua` turns each reference into a link that reads "Equation 20.9".

The full edition's two formats differ only in their appendices, set by Quarto profiles: `book-full/_quarto-website.yml` (the default) adds the study plan, `book-full/_quarto-pdf.yml` leaves it out.

Quarto comes from the `quarto-cli` dev dependency, so there's nothing to install globally. The PDF is built with LaTeX (lualatex, TinyTeX in `~/.TinyTeX`) and needs `rsvg-convert` (`librsvg2-bin`) to embed SVG figures. If a render stops on `File 'x.sty' not found`, install the package with `~/.TinyTeX/bin/x86_64-linux/tlmgr --repository https://ctan.math.illinois.edu/systems/texlive/tlnet install <pkg>`. The default mirror on this machine is stale, so Quarto can't install packages by itself. Typst was tried and dropped: pandoc's LaTeX-to-Typst math conversion mistranslates some commands, for example `\approx` after a superscript.

The SVG figures are build output too. Each script in `code/<part>/` writes its figures into `_build/figures/<part>/`, deterministically, and staging copies them into each book's `<part>/figures/`, so every book references a figure as `figures/<name>.svg` and a figure is never committed. Every build runs `make figures` first, which reruns a script only when a file in its `code/<part>/` folder has changed (its printed tables go to `_build/logs/`, which marks it as up to date). Regenerating everything takes about ten minutes, so pass `-j`. `make figures-force` reruns them all. A figure must therefore come from a script: no hand-drawn or downloaded SVGs.

`_book/` and `_build/` are build output and are git-ignored. Python's bytecode caches go to `_build/pycache/` when you run through make. When `.qmd` files with code exist, their `_freeze/` cache is written in the staged copy; copy it back to the book's `_freeze/` to commit it.

## Adding a chapter

The book's units are Quarto's: a folder `partNN-<topic>/` is a **part**, each file in it is a **chapter** (numbered 1, 2, … across the whole book), and each `##` heading in a file is a **section** (31.2).


- List every new file in `book-full/_quarto.yml` under its part in the `chapters:` list, its exercise-book page (`book-exercises/<part>/NN-<topic>.md`) in `book-exercises/_quarto.yml`, and its practitioner chapter in `book-practitioner/_quarto.yml`. Quarto does not pick up new files on its own.
- Write internal links as relative links to the source file (`[divergences](03-divergences.md)`). Quarto rewrites them for each output, and `filters/links.lua` rewrites links to the other books.

## `.md` or `.qmd`

- **`.md`:** text with no computed content. Figures that a chapter script generates go in with `![caption](figures/name.svg){#fig-name}`.
- **`.qmd`:** a chapter that draws figures or quotes numbers computed by code. Quarto runs its Python cells at build time.

In a `.qmd`:

- **Keep cells thin.** The logic stays in typed pure functions in `code/<part>/<topic>.py`, where strict pyright and pytest cover it. A cell imports and calls those functions. It never defines math or plotting logic itself.
- **Put computed numbers in with inline expressions,** such as `` `{python} f"{delta:.1e}"` ``. Never copy them into the text by hand. A test is only needed for numbers that still appear as literal text.
- **Hide code by default** (`echo: false`). Show it only when the code itself is the point.
- **Use the same seeds as the scripts:** seed 0 for quoted numbers, seed + 1 for figures, so the book and `uv run python code/<part>/<topic>.py` agree.

## Cross-references

Use Quarto's syntax everywhere, in `.md` and `.qmd`, within a book:

- Figures: `![caption](path){#fig-name}` or a cell with `#| label: fig-name`, referenced as `@fig-name`.
- Equations: `$$ ... $$ {#eq-name}`, referenced as `@eq-name`.
- Sections: `## Title {#sec-name}`, referenced as `@sec-name`.

Don't write "the figure below" or "equation (3)" by hand.

## Acronyms, glossary and index

`terms.toml`, shared by the three books, is the curated list of acronyms (with their expansions), names that aren't acronyms (venues, tools, organizations) and index terms. `tools/terms.py` reads it:

- `make terms-check` checks each book and fails on an acronym missing from `terms.toml` or not expanded at its first use on a page. Each page is a web page a reader can land on directly, so every page expands its acronyms once, as "moment generating function (MGF)" or "PPT (probabilistic polynomial time)", using the wording in `terms.toml`. An acronym used only once in the book is spelled out instead.
- `make acronyms` reports each acronym with its uses per page. `make index-candidates` suggests bold phrases not yet indexed.
- The build targets first generate `glossary.md`, `term-index.md` and `_terms.json` in each staged copy (`tools/terms.py build`), never in a book's folder, using that book's and profile's pages. The exercise book has no glossary or index.

The web index links each term to the section that names it on each page. The PDF index has real page numbers: `filters/term-index.lua` marks every paragraph that mentions a term (it reads `_terms.json`), and LaTeX's `imakeidx` prints the index.

Every display equation is numbered by `filters/book-checks.lua`. Label one yourself (`{#eq-name}`) only to reference it.

## Checks

`make check` runs everything below plus the acronym check. `make help` lists all targets.

```bash
uv run pytest
uv run pyright
make website                          # a chapter that fails to render, or warns, is a broken chapter
```
