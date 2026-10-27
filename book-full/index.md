# Preface {.unnumbered}

This is the full edition of *Privacy-Preserving Machine Learning*: every result is derived step by step. If differential privacy is new to you, read the [course in differential privacy](../book-course/index.md) first: it teaches the definition and its core algorithms one topic per chapter, with complete proofs and little else, and this book then goes deeper and wider. The [practitioner edition](../book-practitioner/index.md) shares this book's parts and chapter files, states the same results without proofs and shows how to use them with real libraries; read it next if you build or review private systems.

The figures and the numbers quoted in the text are produced by each chapter's script, in the repository's `code/` directory, one folder per part.

## How this book is organized {.unnumbered}

The book has five parts, each building on the ones before it.

1. **Prerequisites** collects the mathematics: distributions, concentration, divergences and hypothesis testing, then information theory, optimization and stability, and the basics of cryptography.
2. **Threat models and attacks** shows how models and data leak before any defence is introduced. It first attacks trained models, from one bit of membership up to whole training inputs, then attacks released data, from records with names removed to records hidden in crowds to aggregate statistics alone.
3. **Differential privacy foundations** defines the guarantee those attacks call for, builds its mechanisms, composes them, and ends with what privacy costs.
4. **Privacy accounting** tracks the guarantee through many steps of training, as accurately as possible.
5. **Differential privacy in machine learning** trains models with the guarantee: where the noise goes, how the knobs of noisy, clipped gradient descent trade privacy for accuracy, what else spends the budget, who pays for the noise, which $\varepsilon$ to choose, and how to audit the result.

The appendices derive, in plain steps, the results a chapter cites or sketches, and the book ends with a glossary of acronyms and an index.
