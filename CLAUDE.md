# CLAUDE.md

How the book is built. For how to write book content (chapter structure, voice, derivations, code and figure rules), see [AGENTS.md](AGENTS.md).

## Build: Quarto

The book is a [Quarto](https://quarto.org) book project rooted at `book/` (`book/_quarto.yml`). It renders to an HTML site and a PDF from the same sources. Do not add another build system (MkDocs, Jupyter Book, a hand-rolled pandoc script).

```bash
make website   # HTML site, with the solutions, into _book/
make pdf       # PDF, without the solutions, into _book/
make book      # both
make preview   # serve the website from the staged copy (rerun after editing book/)
```

Never render `book/` in place. The make targets copy `book/` to `_build/website/` or `_build/pdf/` and render there, so Quarto's cache, LaTeX's intermediate files and the generated glossary and index never land among the sources. The rendered output is then copied to `_book/`. A render fails on any Quarto warning or error, so a clean build has none.

The two formats differ only in their appendices, set by Quarto profiles. `book/_quarto-website.yml` (the default) includes the exercise book: the "Exercises" part, made of the `solutions/` pages, with every chapter's exercises and their worked solutions. `book/_quarto-pdf.yml` leaves it out. The PDF's staged copy has no `solutions/` folders, and `filters/pdf-links.lua` turns each link into the exercise book into plain text followed by "in the web edition".

Quarto comes from the `quarto-cli` dev dependency, so there's nothing to install globally. The PDF is built with LaTeX (lualatex, TinyTeX in `~/.TinyTeX`) and needs `rsvg-convert` (`librsvg2-bin`) to embed SVG figures. If a render stops on `File 'x.sty' not found`, install the package with `~/.TinyTeX/bin/x86_64-linux/tlmgr --repository https://ctan.math.illinois.edu/systems/texlive/tlnet install <pkg>`. The default mirror on this machine is stale, so Quarto can't install packages by itself. Typst was tried and dropped: pandoc's LaTeX-to-Typst math conversion mistranslates some commands, for example `\approx` after a superscript.

The SVG figures are build output too. Each chapter script writes its figures into its part's `figures/` folder, deterministically, and `book/*/figures/*.svg` is git-ignored, so a figure is never committed. Every build runs `make figures` first, which reruns a script only when a file in its `code/` folder has changed (its printed tables go to `_build/figures/`, which marks it as up to date). Regenerating everything takes about ten minutes, so pass `-j`. `make figures-force` reruns them all, for example after deleting a figure by hand. A figure must therefore come from a script: no hand-drawn or downloaded SVGs.

`_book/` and `_build/` are build output and are git-ignored. Python's bytecode caches go to `_build/pycache/` when you run through make. When `.qmd` files with code exist, their `_freeze/` cache is written in the staged copy; copy it back to `book/_freeze/` to commit it.

## Adding a chapter

The book's units are Quarto's: a folder `partNN-<topic>/` is a **part**, each file in it is a **chapter** (numbered 1, 2, … across the whole book), and each `##` heading in a file is a **section** (31.2).


- List every new file in `book/_quarto.yml` under its part in the `chapters:` list, and its exercise-book page (`solutions/NN-<topic>.md`) in `book/_quarto-website.yml`. Quarto does not pick up new files on its own.
- Write internal links as relative links to the source file (`[divergences](03-divergences.md)`). Quarto rewrites them for each output.

## `.md` or `.qmd`

- **`.md`:** text with no computed content. Figures that a chapter script generates go in with `![caption](figures/name.svg){#fig-name}`.
- **`.qmd`:** a chapter that draws figures or quotes numbers computed by code. Quarto runs its Python cells at build time.

In a `.qmd`:

- **Keep cells thin.** The logic stays in typed pure functions in `code/<topic>.py`, where strict pyright and pytest cover it. A cell imports and calls those functions. It never defines math or plotting logic itself.
- **Put computed numbers in with inline expressions,** such as `` `{python} f"{delta:.1e}"` ``. Never copy them into the text by hand. A test is only needed for numbers that still appear as literal text.
- **Hide code by default** (`echo: false`). Show it only when the code itself is the point.
- **Use the same seeds as the scripts:** seed 0 for quoted numbers, seed + 1 for figures, so the book and `uv run python book/.../code/<topic>.py` agree.

## Cross-references

Use Quarto's syntax everywhere, in `.md` and `.qmd`:

- Figures: `![caption](path){#fig-name}` or a cell with `#| label: fig-name`, referenced as `@fig-name`.
- Equations: `$$ ... $$ {#eq-name}`, referenced as `@eq-name`.
- Sections: `## Title {#sec-name}`, referenced as `@sec-name`.

Don't write "the figure below" or "equation (3)" by hand.

## Acronyms, glossary and index

`book/terms.toml` is the curated list of acronyms (with their expansions), names that aren't acronyms (venues, tools, organizations) and index terms. `tools/terms.py` reads it:

- `make terms-check` fails on an acronym missing from `terms.toml` or not expanded at its first use on a page. Each page is a web page a reader can land on directly, so every page expands its acronyms once, as "moment generating function (MGF)" or "PPT (probabilistic polynomial time)", using the wording in `terms.toml`. An acronym used only once in the book is spelled out instead.
- `make acronyms` reports each acronym with its uses per page. `make index-candidates` suggests bold phrases not yet indexed.
- `make terms` generates `glossary.md`, `term-index.md` and `_terms.json` in each staged copy, never in `book/`, using that profile's pages. So the PDF's glossary does not link to the solutions. The build targets do this first.

The web index links each term to the section that names it on each page. The PDF index has real page numbers: `filters/term-index.lua` marks every paragraph that mentions a term (it reads `_terms.json`), and LaTeX's `imakeidx` prints the index.

Every display equation is numbered by `filters/book-checks.lua`. Label one yourself (`{#eq-name}`) only to reference it.

## Checks

`make check` runs everything below plus the acronym check. `make help` lists all targets.

```bash
uv run pytest
uv run pyright
make website                          # a chapter that fails to render, or warns, is a broken chapter
```
