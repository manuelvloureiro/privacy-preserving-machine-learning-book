# The basics of differential privacy

The first part showed that statistics can expose individuals. This part defines differential privacy (DP), the course's central guarantee. Under DP, replacing one record changes the probability of any event by at most a factor of $e^\varepsilon$, where $\varepsilon$ is the privacy parameter. The chapters then prove basic mechanisms and composition rules with exact constants.

| Chapter | Question it answers |
| --- | --- |
| [Randomized response and the definition](01-randomized-response.md) | How can one collect a sensitive bit from each person, and what does it mean for an algorithm to be private? |
| [The Laplace mechanism](02-laplace-mechanism.md) | How much noise makes a numeric statistic private, and how do private algorithms combine? |
| [Approximate differential privacy](03-approximate-dp.md) | What does a small additive slack $\delta$ allow, how small must it be, and what does the Gaussian mechanism gain? |
| [Advanced composition](04-advanced-composition.md) | Why does the privacy cost of $k$ analyses grow only like $\sqrt k$? |
| [The exponential mechanism](05-exponential-mechanism.md) | How can one privately choose the best object from a set? |

: The chapters of this part and the questions they answer. {#tbl-p2-chapters}
