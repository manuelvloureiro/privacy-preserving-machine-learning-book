# Preface {.unnumbered}

This book teaches the OpenDP library, from installation and a first noisy release to synthetic data generation. OpenDP implements differential privacy (DP), a guarantee that limits how much a release reveals about any one person. The documentation is organized by library components; this book is organized by learning sequence. Each chapter identifies the documentation pages it covers so readers can consult both.

All code has been run. Each chapter uses a script in the repository's `code/` directory; tests check every number quoted in the text. Short code blocks are executable snippets or excerpts from the chapter script. OpenDP uses an unseeded cryptographic generator for noise, so releases vary between runs. Chapters quote reproducible values, such as noise scales, error bars and thresholds, and label sample releases as single runs.

The book uses the French Labour Force Survey (LFS) sample from the OpenDP tutorials, with about 3.8 million quarterly responses from 1.7 million people. Chapters count, sum, group, bound and model the data, then generate a synthetic table.

Readers need Python and basic familiarity with NumPy or Polars, but no prior knowledge of differential privacy. Each chapter defines the results it uses and links to [the full edition](../book-full/index.md) of *Privacy-Preserving Machine Learning* for derivations.

## How this book is organized {.unnumbered}

1. **Getting started** installs the library, releases a first number and creates a context for later queries.
2. **Tabular data** releases statistics and grouped tables with Polars, bounds contributions, prepares data within the guarantee and interprets error bars.
3. **The framework** defines domains, metrics, measures, transformations and measurements; explains chaining and composition; and covers parameter-search utilities.
4. **Extending the library** implements transformations and measurements as plugins, fits regression and principal component models, and describes OpenDP's limitations.
5. **Synthetic data** releases contingency tables, generates synthetic microdata with AIM and MST, and evaluates the results.

The repository pins OpenDP 0.16.0 and Polars 1.36.1. Rerun a chapter from the repository root with `uv run python code/<part>/opendp_<topic>.py`.
