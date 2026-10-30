# The basics of differential privacy

The first part showed that statistics about people can expose them. This part gives the answer that the rest of the course builds on, the definition of differential privacy (DP). Under DP, a change to one person's data changes the probability of any outcome by at most a factor $e^\varepsilon$, for a small privacy parameter $\varepsilon$. The part then builds the first private algorithms and the rules that combine them. Every later part uses these results, so each chapter gives exact constants and full proofs.

| Chapter | Question it answers |
| --- | --- |
| [Randomized response and the definition](01-randomized-response.md) | How can one collect a sensitive bit from each person, and what does it mean for an algorithm to be private? |
| [The Laplace mechanism](02-laplace-mechanism.md) | How much noise makes a numeric statistic private, and how do private algorithms combine? |
| [Approximate differential privacy](03-approximate-dp.md) | What does a small additive slack $\delta$ allow, how small must it be, and what does the Gaussian mechanism gain? |
| [Advanced composition](04-advanced-composition.md) | Why does the privacy cost of $k$ analyses grow only like $\sqrt k$? |
| [The exponential mechanism](05-exponential-mechanism.md) | How can one privately choose the best object from a set? |

: The chapters of this part and the questions they answer. {#tbl-p2-chapters}
