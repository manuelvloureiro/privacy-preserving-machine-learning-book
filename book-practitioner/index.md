# Preface {.unnumbered}

This is the practitioner edition of *Privacy-Preserving Machine Learning*. It is for engineers, data scientists and auditors who build or review machine-learning systems that handle personal data, and it tells them what each result says, when it applies, how to use it and what goes wrong, without proofs.

It covers the same topics as the [full edition](../book-full/index.md), in the same parts and chapters, but it is organised around the work rather than the theory. Each topic states the result with its formula, says why it matters, and gives a recipe with a worked example. It ends with what goes wrong. Each chapter closes with a cheat sheet and a link to the matching chapter of the full edition, which derives every result step by step. A separate exercise book, keyed to the full edition, has worked solutions.

You need some Python and NumPy, basic probability, and enough machine learning to know what a loss and stochastic gradient descent are.

## How this book is organized {.unnumbered}

1. **Prerequisites** is a mathematical reference to dip into when a later chapter cites it.
2. **Threat models and attacks** shows how models and data leak.
3. **Differential privacy foundations** defines the guarantee and builds its mechanisms.
4. **Privacy accounting** tracks the guarantee through private training.
5. **Differential privacy in machine learning** trains, tunes and audits private models.

You do not need to read Part 1 first. With a background in statistics or machine learning, start at Part 2 or Part 3 and follow the links back when you need a fact.

The numbers in the text come from the scripts in the repository's `code/` directory, which the full edition also uses, and tests check that the scripts still produce them. To rerun one, run `uv run python code/<part>/<topic>.py`.
