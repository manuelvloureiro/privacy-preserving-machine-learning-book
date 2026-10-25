# Privacy-Preserving Machine Learning

Five books and a set of study notes on privacy-preserving machine learning, built with [Quarto](https://quarto.org) from one repository, and published at <https://manuelvloureiro.github.io/privacy-preserving-machine-learning-book/>.

- **The full edition** (`book-full/`) derives every result, for readers who want the mathematics.
- **The exercise book** (`book-exercises/`) has the full edition's exercises, each with a worked solution, one page per chapter.
- **The practitioner edition** (`book-practitioner/`) covers the same topics without proofs, organized around the work of people who build, release and audit private machine-learning systems, with workflows that run the real libraries on public data.
- **The course** (`book-course/`) teaches differential privacy one topic per chapter, light in text and complete in its proofs; it stands alone.
- **Study notes** (`notes/<name>/`) are one-page treatments of a single question, each with its PDF.

## Layout

```
book-full/                       the full edition: _quarto.yml, index.md, one folder per part
  partNN-<topic>/                a part: 00-introduction.md (the part page), NN-<topic>.md per chapter, README.md for browsing
  appendix-derivations/          derivations of what the chapters state or cite
book-exercises/partNN-<topic>/   a chapter's exercises and worked solutions, same file name as the chapter
book-practitioner/               the practitioner edition: the same parts and chapters, plus workflow chapters
book-course/partNN-<topic>/      the course, with its own parts and chapters
notes/<name>/                    a study note: _quarto.yml and index.md
code/<folder>/                   one script per chapter (the figures and every number a page quotes),
                                 one per exercise page (solutions_*.py), one per practitioner workflow
                                 (practitioner_*.py), one per note (code/notes/); folders are named after the parts
                                 (code/notes/); folders are named after the parts
tests/                           one test per script, checking that it still prints what the pages quote
tools/                           terms.py (acronyms, glossary, index), notation.py, prose.py, structure.py, xrefs.py
filters/                         Lua filters applied when rendering
_common.yml, _book-common.yml    Quarto settings every project shares
terms.toml, notation.toml        the acronyms and index terms; the notation rules a tool can check
typings/                         hand-written type stubs for the untyped libraries
data/                            downloaded public datasets, not committed
```

Four documents say how the repository works, each owning its rules: [CLAUDE.md](CLAUDE.md) (how everything is built and checked), [AGENTS.md](AGENTS.md) (what to write and the shape of each kind of page), [PROSE.md](PROSE.md) (how to make a page readable and prove it) and [NOTATION.md](NOTATION.md) (one symbol, spelling and word for each concept).

Code is fully type hinted and checked with strict pyright. Figures are SVG files that the scripts generate deterministically into `_build/figures/`; they are not committed, and every build regenerates the ones whose code changed. When a page quotes a number, a test in `tests/` checks that the script still produces it.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+; the PDFs also need LaTeX (TinyTeX, `make tex-deps`) and `rsvg-convert`.

```bash
make setup
```

## Running

```bash
uv run python code/part01-prerequisites/concentration.py   # one chapter's script, from the repository root
make website        # every website, into _book/<book>/
make pdf            # every PDF, into _book/
make notes          # the study notes, into _book/notes/<name>/
make book           # all of the above
make check          # tests, types, acronyms, notation, prose, structure and a clean render of every website
make help           # every target
```

The targets render staged copies in `_build/`, so no build output lands next to the sources, and they fail on any warning. The version on a cover is the build date. See [CLAUDE.md](CLAUDE.md) for the conventions.
