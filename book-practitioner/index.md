# Preface {.unnumbered}

This is the practitioner edition of *Privacy-Preserving Machine Learning*. It is for people who build, review and audit machine-learning systems that handle personal data, and who need to know what each result says, when it applies, how to use it and what goes wrong in practice, without working through its proof.

It covers the same topics as the [full edition](../book-full/index.md), in the same parts and chapters. Each topic is explained in four steps: the idea and why it works, the result or rule, a worked example with real numbers, and the pitfalls that catch people in practice. Each chapter ends with a cheat sheet and a link to the full edition's chapter, where every result is derived step by step.

## How this book is organized {.unnumbered}

1. **Prerequisites** collects the mathematics the later parts use: distributions, tail bounds, ways to compare distributions, hypothesis testing, and the basics of information theory, optimization, stability and cryptography.
2. **Threat models and attacks** shows how models and data leak: membership inference, attribute and property inference, memorization and extraction, gradient inversion, re-identification, and reconstruction from aggregate statistics.
3. **Differential privacy foundations** defines the guarantee those attacks call for, builds its mechanisms, composes them, and says what privacy costs.
4. **Privacy accounting** tracks the guarantee through the many steps of private training, and ends with how to choose an accountant and what to report.

The numbers in the text are computed by the scripts in the repository's `code/` directory, the same ones the full edition uses, and tests check that the scripts still produce them.
