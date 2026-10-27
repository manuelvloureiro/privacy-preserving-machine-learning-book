# Getting started

This part takes you from an empty Python environment to a context that holds a real dataset and a privacy budget. It assumes nothing about differential privacy, and its one idea is this. A differentially private release is a randomized computation whose noise is calibrated to how much one person can change the answer. OpenDP does that calibration for you, and it keeps count of what you spend.

The first chapter installs the library and releases one noisy number, first with a mechanism built by hand and then with the Context API. With the Context API, you state the privacy unit and the budget, and the library finds the noise. On the way, the chapter gives the five steps of every analysis in OpenDP.

The second chapter loads the dataset of the whole book, the French Labour Force Survey (LFS) sample of the OpenDP tutorials. Then it builds a context on it. A person who answers the survey in many quarters owns many rows, and the chapter explains why the privacy unit must cover all of them. It also shows how a context splits one budget over several queries, and what a margin is.

| Chapter | Question it answers |
| --- | --- |
| [A first release](01-first-release.md) | How do I install OpenDP, and what happens when I release one noisy number? |
| [The data and the context](02-the-data-and-the-context.md) | How do I put a real dataset behind a context, and what do its unit, budget and margins mean? |

: The chapters of this part and the questions they answer. {#tbl-p1-chapters}

At the end of this part, you can explain the privacy unit and the budget of a release in one sentence each. You can also build a context for a dataset in which one person owns many rows.
