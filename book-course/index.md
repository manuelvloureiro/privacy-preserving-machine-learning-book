# Preface {.unnumbered}

This course develops differential privacy from motivating failures to formal guarantees and algorithms. Later parts cover query answering, private statistics and machine learning, lower bounds, and deployments.

Each chapter covers one topic. Numbered definitions, theorems, and proofs make the mathematics explicit. Proofs are complete or cite a standard source and state the needed prerequisites. The prose explains each result and its implications.

The course assumes discrete probability, including expectation, variance, independence, and Markov's, Chebyshev's, and Hoeffding's inequalities. It also assumes basic linear algebra and calculus. The course states and proves any additional probability results where it first uses them.

**Notation.** A dataset is $D = (x_1, \dots, x_n) \in \mathcal X^n$, a list of $n$ records from a universe $\mathcal X$. Two datasets are neighbours, $D \sim D'$, when they differ in one record, which is the replace-one relation. A mechanism $M$ maps a dataset to a random output in a set $\mathcal Y$. The privacy parameters are $\varepsilon \ge 0$ and $\delta \in [0,1]$. The natural logarithm is written $\ln$, and the logarithm to base 2 is written $\log_2$.
