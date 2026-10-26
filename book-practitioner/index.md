# Preface {.unnumbered}

This is the practitioner edition of *Privacy-Preserving Machine Learning*. It is for engineers, data scientists and auditors who build or review machine-learning systems that handle personal data, and it tells them what each result says, when it applies, how to use it and what goes wrong, without proofs.

It keeps the parts and the chapter files of the [full edition](../book-full/index.md), so that every page here has a twin there that derives its results step by step, but it is organized around the work rather than the theory. Three kinds of page do that. A **workflow chapter** does one job end to end with real libraries on a real public dataset: assessing a model's privacy risk before release, releasing statistics with OpenDP and diffprivlib, calibrating and reporting a run of differentially private stochastic gradient descent, training and fine-tuning with Opacus, and governing a release. Each part reads in the order of the work, so a workflow comes early in its part and the pages it relies on follow. A **full chapter** covers a topic a practitioner computes with: the result and its formula, why it matters, a recipe with a worked example, what goes wrong, and a cheat sheet. A **reference page** covers a theory-heavy topic a practitioner meets but rarely computes with, in a few hundred words: what it is, when you meet it, how to use it and what goes wrong. Every page ends with a link to the full edition's chapter.

You need some Python and NumPy, basic probability, and enough machine learning to know what a loss and stochastic gradient descent are.

## How this book is organized {.unnumbered}

1. **Prerequisites** is a mathematical reference to dip into when a later chapter cites it.
2. **Threat models and attacks** shows how models and data leak, and assesses a model's risk before release.
3. **Differential privacy foundations** defines the guarantee, builds its mechanisms, and releases statistics with them.
4. **Privacy accounting** tracks the guarantee through private training, and calibrates and reports a run.
5. **Differential privacy in machine learning** trains, fine-tunes, audits and releases private models.

You do not need to read Part 1 first. With a background in statistics or machine learning, start at Part 2 or Part 3 and follow the links back when you need a fact.

Every piece of code on a page has been run: a snippet runs on its own, and an excerpt is a marked region of a workflow script in the repository's `code/` directory, which a test runs and checks against the numbers the page quotes. To rerun a workflow, run `uv run python code/<part>/practitioner_<topic>.py`.
