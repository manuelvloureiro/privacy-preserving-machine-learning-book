# Preface {.unnumbered}

This is the full edition of *Privacy-Preserving Machine Learning*. It derives results step by step. If differential privacy is new to you, start with the [course in differential privacy](../book-course/index.md), which introduces the definition and core algorithms with proofs. The [practitioner edition](../book-practitioner/index.md) uses the same parts and chapter topics without proofs, and shows how to apply the results with real libraries.

The figures and the numbers quoted in the text are produced by each chapter's script, in the repository's `code/` directory, one folder per part.

## How this book is organized {.unnumbered}

The book has five parts. Each builds on earlier results.

1. **Prerequisites** develops probability distributions, concentration, divergences, hypothesis testing, information theory, optimization, stability and cryptography.
2. **Threat models and attacks** defines adversaries and evaluates attacks on trained models and released data. The attacks cover membership inference, reconstruction, linkage, syntactic anonymity and aggregate-data inference.
3. **Differential privacy foundations** defines the guarantee, develops mechanisms, proves composition results and studies privacy costs.
4. **Privacy accounting** tracks guarantees across iterative training.
5. **Differential privacy in machine learning** applies differential privacy to model training. It covers noise placement, clipped gradient descent, budget allocation, disparate impact, budget selection and auditing.

The appendices derive results that chapters cite or sketch. The book ends with a glossary of acronyms and an index.
