# Getting started

This part introduces OpenDP contexts, which hold a dataset and privacy budget. It assumes no prior knowledge of differential privacy. A differentially private release adds noise calibrated to the maximum effect of one person's data. OpenDP calibrates the noise and tracks the budget.

The first chapter installs the library and releases a noisy number, first with a manually built mechanism and then with the Context API. The context uses the privacy unit and budget to calibrate noise. The chapter also presents the five steps of an OpenDP analysis.

The second chapter loads the French Labour Force Survey (LFS) sample used in the OpenDP tutorials and builds a context for it. Because a person may have rows from multiple quarters, the privacy unit must cover all of that person's rows. The chapter also explains how contexts divide a budget among queries and how to define a margin.

| Chapter | Question it answers |
| --- | --- |
| [A first release](01-first-release.md) | How do I install OpenDP, and what happens when I release one noisy number? |
| [The data and the context](02-the-data-and-the-context.md) | How do I put a real dataset behind a context, and what do its unit, budget and margins mean? |

: The chapters of this part and the questions they answer. {#tbl-p1-chapters}

By the end of this part, you can specify a release's privacy unit and budget and build a context for data with multiple rows per person.
