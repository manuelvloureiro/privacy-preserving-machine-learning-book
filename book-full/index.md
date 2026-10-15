# Preface {.unnumbered}

Notes and code for a book on privacy-preserving machine learning, written alongside the [study plan](study_plan.md).

The figures and the numbers quoted in the text are produced by each chapter's script, in the repository's `code/` directory, one folder per part.

## How this book is organized {.unnumbered}

The book has four parts, each building on the ones before it. Each part covers one phase of the study plan, and each of its chapters one topic of that phase.

1. **Prerequisites** collects the mathematics: distributions, concentration, divergences and hypothesis testing, then information theory, optimization and stability, and the basics of cryptography.
2. **Threat models and attacks** shows how models and data leak before any defence is introduced. It first attacks trained models, from one bit of membership up to whole training inputs, then attacks released data, from records with names removed to records hidden in crowds to aggregate statistics alone.
3. **Differential privacy foundations** defines the guarantee those attacks call for, builds its mechanisms, composes them, and ends with what privacy costs.
4. **Privacy accounting** tracks the guarantee through many steps of training, as accurately as possible.

The appendices hold a glossary of acronyms and an index; the web edition also has the study plan. The exercises, with a worked solution for each, are in the exercise book, a companion website, and each chapter ends with a link to its page there.
