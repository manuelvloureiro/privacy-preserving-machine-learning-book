# The Basics of Differential Privacy

The first part showed that releasing statistics about people can expose them, even when the statistics look harmless. This part gives the answer that the rest of the book builds on. It states a definition of privacy, differential privacy (DP), and develops the first algorithms that satisfy it and the rules for combining them.

The definition comes first, from the oldest private algorithm. In randomized response, each person flips their sensitive bit with some probability before reporting it, so every report is deniable. Differential privacy turns that deniability into a bound. Changing one person's data may change the probability of any outcome by at most a factor $e^\varepsilon$, where the privacy parameter $\varepsilon$ is small.

The next two chapters give the basic tools for releasing numbers. The Laplace mechanism adds noise scaled to how much one person can change a statistic, and on a dataset of $n$ people it estimates a fraction with error of order $1/(\varepsilon n)$ instead of the $1/(\varepsilon\sqrt n)$ of randomized response. The same chapter proves that privacy survives post-processing, degrades gracefully for groups and adds up under composition. The third chapter relaxes the definition to approximate DP, which allows a small probability $\delta$ of failure. Its examples show that $\delta$ must be much smaller than $1/n$, and in exchange the relaxation admits the Gaussian mechanism, which is more accurate in high dimension.

The last two chapters go further. Advanced composition shows that the privacy cost of $k$ analyses grows like $\sqrt k$ rather than $k$, at the price of a small extra $\delta$. The exponential mechanism handles tasks whose answer is an object rather than a number, such as a price or a classifier, by sampling objects with probability that grows exponentially with their quality.

Every chapter of the later parts uses these results. Each chapter here states them with exact constants and proves them in full.

| Chapter | Question it answers |
| --- | --- |
| [Randomized response and the definition](01-randomized-response.md) | How can one collect a sensitive bit from each person, and what does it mean for an algorithm to be private? |
| [The Laplace mechanism](02-laplace-mechanism.md) | How much noise makes a numeric statistic private, and how do private algorithms combine? |
| [Approximate differential privacy](03-approximate-dp.md) | What does a small failure probability $\delta$ allow, how small must it be, and what does the Gaussian mechanism gain? |
| [Advanced composition](04-advanced-composition.md) | Why does the privacy cost of $k$ analyses grow only like $\sqrt k$? |
| [The exponential mechanism](05-exponential-mechanism.md) | How can one privately choose the best object from a set? |

: The chapters of this part and the questions they answer. {#tbl-p2-chapters}
