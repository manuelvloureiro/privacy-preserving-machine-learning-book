# Privacy-Preserving Machine Learning

Notes and code for a book on privacy-preserving machine learning, written alongside a [study plan](book/study_plan.md).

## Layout

```
book/
  _quarto.yml                    the Quarto book: parts, chapters, PDF settings, filters
  _quarto-website.yml            website profile: adds the exercise book
  _quarto-pdf.yml                PDF profile: no exercise book
  study_plan.md                  the plan the book follows, one phase per part
  terms.toml                     acronyms and index terms
  README.md                      table of contents
  filters/                       Lua filters used when rendering
  partNN-<topic>/             one part of the book
    README.md                    part overview, readings, exit check
    00-introduction.md           the part page
    NN-<topic>.md                one file per chapter
    solutions/NN-<topic>.md      the chapter's exercises and worked solutions (the exercise book)
    code/                        runnable scripts, one per chapter
    figures/                     SVG figures the scripts generate (not committed)
tests/                           checks that the scripts run and match the text
tools/terms.py                   acronym checks, glossary and index generation
data/                            local datasets, not committed
```

How to write content (chapter structure, voice, code and figure rules) is in [AGENTS.md](AGENTS.md); how the book is built is in [CLAUDE.md](CLAUDE.md).

Each part keeps its code next to its text. Code is fully type hinted and checked with strict pyright. Figures are SVG files that the scripts generate deterministically (every random draw is seeded); they are not committed, and every build regenerates the ones whose code changed. When a chapter quotes numbers from a script, a test in `tests/` checks that the script still produces them.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync
```

## Running

Run any chapter's script from the repository root:

```bash
uv run python book/part01-prerequisites/code/concentration.py
```

Build the book with [Quarto](https://quarto.org), through make. The targets render staged copies in `_build/`, so no build output lands next to the sources, and they fail on any warning:

```bash
make website   # HTML site with the exercise book, into _book/
make pdf       # PDF without the exercise book, into _book/
make book      # both
```

The version on the cover is the build date. The PDF needs LaTeX (`make tex-deps` installs the packages) and `rsvg-convert` (`librsvg2-bin` on Debian/Ubuntu) to embed the SVG figures. `make help` lists every target. See [CLAUDE.md](CLAUDE.md) for the build conventions.

Run the tests and the type checker:

```bash
uv run pytest
uv run pyright
```
