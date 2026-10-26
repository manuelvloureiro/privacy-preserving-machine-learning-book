# Preface {.unnumbered}

This book teaches the OpenDP library, from installing it and releasing one noisy number to generating a synthetic copy of a dataset. OpenDP is a library for differential privacy (DP), the guarantee that a release reveals almost nothing about any one person in the data. The library's documentation is a reference, organized by what the library contains. This book is a course, organized by what you learn next, and it follows the documentation closely enough that every chapter names the pages it covers, so you can read the two side by side.

Every piece of code here has been run. A chapter is built around one script in the repository's `code/` directory, which a test runs and checks against every number the chapter quotes. A short block on a page is either a snippet that runs on its own or an excerpt of that script, so the code you read is the code that produced the output next to it. OpenDP draws its noise from a cryptographic generator that cannot be seeded, so a released value is different on every run. A chapter therefore quotes the numbers that are the same on every run, such as noise scales, error bars and thresholds, and shows released values in a block marked as one run.

One dataset runs through the book: the sample of the French Labour Force Survey (LFS) that the OpenDP documentation uses in its own tutorials, about 3.8 million quarterly survey responses from 1.7 million people. Each chapter releases something new about it, so by the end you will have counted, summed, grouped, bounded and modelled it, and finally replaced it with a synthetic table.

You need Python and enough NumPy or Polars to read a short query. You do not need to know differential privacy already. Each chapter says in a sentence what the results it uses mean, and links to [the full edition](../book-full/index.md) of *Privacy-Preserving Machine Learning* for their derivations, which this book never repeats.

## How this book is organized {.unnumbered}

1. **Getting started** installs the library, releases a first number, and sets up the context that mediates every later query to the data.
2. **Tabular data** releases counts, sums, means, quantiles and grouped tables with Polars, bounds what one person can contribute, prepares the data inside the privacy guarantee, and reads error bars.
3. **The framework** opens the box: domains, metrics and measures, transformations and measurements, how they chain and compose, and the utilities that search for parameters.
4. **Extending the library** writes new transformations and measurements as plugins, fits regression and principal component models, and knows what the library does not promise.
5. **Synthetic data** releases contingency tables and generates synthetic microdata with the AIM and MST algorithms, then measures how good the copy is.

The code runs on OpenDP 0.16.0 with Polars 1.36.1, pinned in the repository so that every number stays reproducible. To rerun a chapter, run `uv run python code/<part>/opendp_<topic>.py` from the repository's root.
