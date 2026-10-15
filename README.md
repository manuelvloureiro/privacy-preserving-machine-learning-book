# Privacy-Preserving Machine Learning

Notes and code for a book on privacy-preserving machine learning, written alongside a [study plan](book-full/study_plan.md).

## Layout

The repository holds three books, built with Quarto from the same code:

- **The full edition** (`book-full/`) derives every result, for readers who want the mathematics.
- **The exercise book** (`book-exercises/`) has the full edition's exercises, each with a worked solution, one page per chapter.
- **The practitioner edition** (`book-practitioner/`, in progress) covers the same topics in the same parts and chapters, without proofs, for people who build and audit private machine-learning systems.

```
book-full/                       the full edition
  _quarto.yml                    the Quarto book: parts, chapters, PDF settings, filters
  _quarto-website.yml            website profile: adds the study plan
  _quarto-pdf.yml                PDF profile
  study_plan.md                  the plan the book follows, one phase per part
  README.md                      table of contents
  partNN-<topic>/                one part of the book
    README.md                    part overview, readings, exit check
    00-introduction.md           the part page
    NN-<topic>.md                one file per chapter
book-exercises/                  the exercise book (website only)
  _quarto.yml
  partNN-<topic>/NN-<topic>.md   a chapter's exercises and worked solutions, same file name as the chapter
book-practitioner/               the practitioner edition, same parts and chapters as book-full/
code/partNN-<topic>/             runnable scripts, one per chapter (and one per exercise page),
                                 shared by the three books
filters/                         Lua filters used when rendering any of the books
terms.toml                       acronyms and index terms, shared by the three books
tests/                           checks that the scripts run and match the text
tools/terms.py                   acronym checks, glossary and index generation
tools/xrefs.py                   the full edition's labels, for the exercise book's cross-references
data/                            local datasets, not committed
```

How to write content (chapter structure, voice, code and figure rules) is in [AGENTS.md](AGENTS.md); how the books are built is in [CLAUDE.md](CLAUDE.md).

Code is fully type hinted and checked with strict pyright. Figures are SVG files that the scripts generate deterministically (every random draw is seeded) into `_build/figures/`; they are not committed, and every build regenerates the ones whose code changed. When a page quotes numbers from a script, a test in `tests/` checks that the script still produces them.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync
```

## Running

Run any chapter's script from the repository root:

```bash
uv run python code/part01-prerequisites/concentration.py
```

Build the book with [Quarto](https://quarto.org), through make. The targets render staged copies in `_build/`, so no build output lands next to the sources, and they fail on any warning:

```bash
make website   # every website, into _book/book-full/, _book/book-exercises/, _book/book-practitioner/
make pdf       # the full and practitioner PDFs, into _book/
make book      # both
make full      # one book only: also full-pdf, exercises, practitioner, practitioner-pdf
```

The version on the cover is the build date. The PDF needs LaTeX (`make tex-deps` installs the packages) and `rsvg-convert` (`librsvg2-bin` on Debian/Ubuntu) to embed the SVG figures. `make help` lists every target. See [CLAUDE.md](CLAUDE.md) for the build conventions.

Run the tests and the type checker:

```bash
uv run pytest
uv run pyright
```
