# Preface {.unnumbered}

This book is a course in differential privacy. It starts from the failures that motivate the subject, builds the definition and its basic algorithms, and goes on to answering many queries, private statistics and machine learning, lower bounds, and deployments.

Each chapter takes one topic. The text is short, and the mathematics is complete. Definitions, theorems and proofs are set apart and numbered, and every proof is given in full or, where it is long and standard, cited with what it needs. The prose between them says why each result comes next and what it means.

The reader should know discrete probability, including expectation, variance, independence and the inequalities of Markov, Chebyshev and Hoeffding, and basic linear algebra and calculus. Results from probability that the course needs beyond these are stated and proved where they are first used.

**Notation.** A dataset is $D = (x_1, \dots, x_n) \in \mathcal X^n$, a list of $n$ records from a universe $\mathcal X$. Two datasets are neighbours, $D \sim D'$, when they differ in one record, which is the replace-one relation. A mechanism $M$ maps a dataset to a random output in a set $\mathcal Y$. The privacy parameters are $\varepsilon \ge 0$ and $\delta \in [0,1]$. The natural logarithm is written $\ln$, and the logarithm to base 2 is written $\log_2$.
