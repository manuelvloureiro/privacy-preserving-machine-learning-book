# Getting started

This part takes you from an empty Python environment to a context that holds a real dataset and a privacy budget. It assumes nothing about differential privacy. The one idea it teaches is simple to state. A differentially private release is a randomized computation whose noise is calibrated to how much one person can change the answer. OpenDP does that calibration for you, and it keeps count of what has been spent.

The first chapter installs the library and releases a single noisy number. It does so first with the low-level framework, where you build the mechanism by hand, and then with the Context API, where you state the privacy unit and the budget and let the library find the noise. On the way it introduces the five steps of every analysis in OpenDP. You identify the privacy unit, set the budget, collect public information, mediate access to the data, and submit queries.

The second chapter brings in the dataset that runs through the book. It is the sample of the French Labour Force Survey (LFS) that the OpenDP documentation uses in its tutorials. The chapter shows how to load the data lazily with Polars and why a person who answers the survey in many quarters needs a privacy unit of many rows. It then explains how a context splits one budget over several queries, and what a margin is.

| Chapter | Question it answers |
| --- | --- |
| [A first release](01-first-release.md) | How do I install OpenDP, and what happens when I release one noisy number? |
| [The data and the context](02-the-data-and-the-context.md) | How do I put a real dataset behind a context, and what do its unit, budget and margins mean? |

: The chapters of this part and the questions they answer. {#tbl-p1-chapters}

By the end of this part you will be able to install OpenDP and explain the privacy unit and the budget of a release in a sentence each. You will also be able to build a context for a dataset in which one person owns many rows.
