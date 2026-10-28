# CLAUDE.md

How the books are built and checked. The other three documents cover the rest: [AGENTS.md](AGENTS.md) says what to write (the books, the shape of every kind of page, code and figures), [PROSE.md](PROSE.md) how to make and prove that a page reads well, and [NOTATION.md](NOTATION.md) which symbol, spelling and word to use for each concept. Each rule lives in one of the four; the others link to it.

## The projects

Four [Quarto](https://quarto.org) book projects sit side by side and share `code/`, `filters/`, `terms.toml`, `notation.toml` and the Quarto settings in `_common.yml` and `_book-common.yml`:

- `book-full/`: the full edition, with every derivation. Website and PDF.
- `book-practitioner/`: the practitioner edition, the same parts and topics without proofs, plus workflow chapters that run real libraries. Website and PDF.
- `book-course/`: a course in differential privacy, one topic per chapter, light in text and complete in its proofs. It links to no other book. Website and PDF.
- `book-opendp/`: the OpenDP book, which teaches the OpenDP library from its first release to synthetic data in the order of its documentation, with its own parts; its code runs the library on the France Labour Force Survey sample. Website and PDF.

Next to them, `notes/<name>/` holds standalone study notes: each is a one-page Quarto project (`_quarto.yml` and `index.md`, with a script in `code/notes/` when it computes anything) that renders to a web page with its PDF. The landing page `site/index.html` links to every book, PDF and note.

Each `_quarto.yml` holds only what is its own: the title, the PDF's `output-file`, the chapter list and the filters it runs. Everything shared comes from `_common.yml` (filters' companions: fonts, theme, PDF engine and header) and, for the books, `_book-common.yml` (author, the dated version on the cover, chapter-wise numbering, the KOMA book class), pulled in through `metadata-files`. The Makefile copies both into each staged copy, which is why a project can name a file that does not sit in its folder. There are no Quarto profiles.

Do not add another build system (MkDocs, Jupyter Book, a hand-rolled pandoc script).

## Building

```bash
make website      # every website, into _book/<book>/
make pdf          # the three PDFs, into _book/
make notes        # the study notes, into _book/notes/<name>/
make book         # all of the above
make full         # one target: full, full-pdf, practitioner, practitioner-pdf, course, course-pdf, opendp, opendp-pdf
make preview      # serve the full edition from its staged copy (rerun after editing book-full/)
make help         # every target
```

`make publish` puts `_book/` and `site/` on the `gh-pages` branch and pushes it; GitHub Pages serves it at `https://manuelvloureiro.github.io/privacy-preserving-machine-learning-book/`, one book per folder. It publishes whatever is in `_book/`, so run `make book` first.

**Never render a project's folder in place.** The targets copy it to `_build/<name>-<format>/`, add `filters/`, the shared settings and the figures, and render there, so Quarto's cache, LaTeX's intermediate files and the generated glossary and index never land among the sources. The website is then copied to `_book/<book>/`, the PDF to `_book/`. A render fails on any Quarto warning or error, so a clean build has none.

**Figures are build output.** Each script in `code/<folder>/` writes its SVG figures into `_build/figures/<folder>/`, deterministically, and staging copies them into the book folder of the same name (`<part>/figures/`, `appendix-derivations/figures/`), so a page references a figure as `figures/<name>.svg` and no figure is committed. Every build runs `make figures` first, which reruns a script only when a file in its folder changed; its printed tables go to `_build/logs/`, which doubles as the up-to-date marker. Regenerating everything takes about ten minutes, so pass `-j`. `make figures-force` reruns them all. Five scripts run their experiments in a process pool: the trade-off note's two and the Python libraries note's in `code/notes/`, `practitioner_calibrate.py`, and `practitioner_adult_studies.py`, which runs the Adult studies of the practitioner DP-SGD pages in about 25 minutes and defaults to four workers. `PRIVACY_WORKERS` sets the number of processes. Each worker of the synthetic-data sweep loads JAX and PyTorch, so that script defaults to four. Running it with a worker per core, next to other jobs, ran this machine out of memory.

**Three naming schemes share `code/`.** The full and practitioner editions use the parts `part01-prerequisites` … `part05-dp-ml`; the course has its own, `part01-privacy-failures` … `part05-deployments`; the OpenDP book has `part01-getting-started` … `part05-synthetic-data`. All three sets of folders sit in `code/`, and only the folder name ties a script's figures to a book's part, so a new part must not reuse a name from another scheme. The practitioner workflows (`code/<part>/practitioner_*.py`), the OpenDP book's scripts (`code/<part>/opendp_*.py`) and the notes' scripts (`code/notes/`) draw no figures and are not run by `make figures`.

**Links between the books.** A page links to another book, to `code/` or to `tests/` by its relative path in the repository (`../../book-full/part03-dp-foundations/03-laplace-gaussian.md#sec-lg-laplace`), so the links also work when browsing the source. `filters/links.lua` rewrites them for the output: a link to `code/` or `tests/` points to the file on GitHub; on a website a link to another book points to its page there, since `_book/` mirrors the repository's layout; in a PDF it becomes plain text followed by "in the full edition", "in the practitioner edition", "in the course" or "in the OpenDP book". Quarto resolves `@`-references only within one book, so a page refers to another book's result by a link to its section.

**LaTeX.** The PDF is built with lualatex from TinyTeX in `~/.TinyTeX` and needs `rsvg-convert` (`librsvg2-bin`) to embed SVG figures; `make tex-deps` installs the packages. If a render stops on `File 'x.sty' not found`, install the package with `~/.TinyTeX/bin/x86_64-linux/tlmgr --repository https://ctan.math.illinois.edu/systems/texlive/tlnet install <pkg>`: the default mirror on this machine is stale, so Quarto cannot install packages by itself. Typst was tried and dropped, because pandoc's LaTeX-to-Typst math conversion mistranslates some commands.

`_book/` and `_build/` are build output and git-ignored. Python's bytecode caches go to `_build/pycache/` when you run through make. Every page is a `.md` file; the books have no `.qmd` files and no code cells. If one is ever needed, keep its cells thin (import from `code/`, compute nothing in the cell) and commit its `_freeze/` cache.

## Adding a chapter or a book

- List a new file in its book's `_quarto.yml` (Quarto does not pick up files on its own): a full chapter also gets its practitioner page in `book-practitioner/_quarto.yml`; a course chapter goes in `book-course/_quarto.yml` only. Then add the page to `legacy` in `tools/structure.toml` only if it is not yet in its skeleton's shape (see Checks).
- A new book needs: its folder with `_quarto.yml` and `index.md`, a `PDF_<name>` line and a `$(eval $(call BOOK_TARGETS,<name>,<folder>))` line plus two documented stubs in the Makefile, its name in `BOOKS`, an entry in `tools/structure.toml`, `tools/prose.toml` and `notation.py`'s `BOOKS`, and a card in `site/index.html`. A new note needs only its folder (and its script in `code/notes/` if it computes anything) and the card.

## Acronyms, glossary and index

`terms.toml`, shared by every project, is the curated list of acronyms with their expansions, names that are not acronyms (venues, tools, organizations) and index terms. `tools/terms.py` reads it: `make terms-check` fails on an acronym missing from `terms.toml` or not expanded at its first use on a page (each page is a web page a reader can land on directly, so every page expands its acronyms once, using the wording in `terms.toml`; an acronym used only once in a book is spelled out instead); `make acronyms` reports each acronym's uses per page; `make index-candidates` suggests bold phrases not yet indexed. The build targets generate `glossary.md`, `term-index.md` and `_terms.json` in each staged copy, never in a source folder. The web index links each term to the section that names it; the PDF index has page numbers, from `filters/term-index.lua` and LaTeX's `imakeidx`. The notes have no glossary or index.

## Checks

`make check` runs all of these, over every book and every note, and a chapter that fails any of them is not done:

```bash
make test               # uv run pytest, including the practitioner libraries (make test-fast skips them)
make typecheck          # strict pyright over code/, tests/ and tools/
make terms-check        # acronyms expanded and listed (BOOK= for one book or notes/<name>)
make notation-check     # the notation NOTATION.md fixes (notation.toml)
make prose-check        # readability within the book's limits (tools/prose.toml); see PROSE.md
make structure-check    # every page in its kind's shape (tools/structure.toml); see AGENTS.md
make duplicates-check   # no paragraph repeats another, in or across books, unless tools/duplicates.toml allows it
make website            # a chapter that fails to render, or warns, is a broken chapter
```

`make structure-check` reads `tools/structure.toml`: for each book, which skeleton each page follows (full chapter, part page, appendix, practitioner page, course chapter, OpenDP chapter, note) and what the skeleton requires. Pages listed under a book's `legacy` have not been converted to their skeleton yet and are skipped until their line is removed; `make structure BOOK=book-full` reports words per page and section.

**The practitioner edition's code** runs real libraries (Opacus, CPU PyTorch, `dp-accounting`, OpenDP, diffprivlib, `peft`, `transformers`, scikit-learn, SciPy, ART), pinned in the `practitioner` dependency group, which `uv` installs by default. Its tests are marked `libs`. The workflow scripts read public datasets through `code/practitioner_datasets.py`, which caches them in `data/` (git-ignored); `make datasets` fills the cache and `make test` runs it first. Untyped libraries are typed by hand-written stubs in `typings/`. `tests/test_practitioner_excerpts.py` checks that every code block on a practitioner or OpenDP page is a snippet, a `text` or `bash` block, or an excerpt that matches a marked region of a script; `tests/test_practitioner_snippets.py` runs the snippets of both books. The OpenDP book's scripts (`code/<part>/opendp_*.py`, tests marked `libs`) run OpenDP on the France Labour Force Survey sample, which `code/practitioner_datasets.py` caches like the other datasets. [AGENTS.md](AGENTS.md) has the rules.
