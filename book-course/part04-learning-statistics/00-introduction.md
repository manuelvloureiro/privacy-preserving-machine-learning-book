# Private Learning and Statistics

The earlier parts released statistics of a fixed dataset under differential privacy (DP). This part turns to the goals of machine learning and statistics: fitting models, and learning about the population from which the data was drawn.

It opens by asking what privacy should protect. A trained model can predict a person's sensitive label, and the prediction can feel like a leak even when nothing about that person's record was used. [The first chapter](01-privacy-violations.md) separates the inferences the definition is designed to allow from those it rules out. It proves that a private learner can still find a rule that holds in the population, and that it cannot memorize a secret held in one person's records.

The next two chapters train models privately. [Private empirical risk minimization](02-private-erm.md) handles convex losses, and adds noise to the output of a solver, to its objective or to each gradient step. Noisy stochastic gradient descent comes with proofs of both privacy and accuracy. [Private deep learning](03-private-deep-learning.md) carries that algorithm over to neural networks by clipping gradients. It replaces composition with a tighter accounting based on the moments of the privacy loss, and it presents a second route, a private vote among models trained on disjoint parts of the data.

The last two chapters study the population. [Private mean estimation](04-mean-estimation.md) asks how many samples a private algorithm needs to estimate the mean of a distribution, from bounded data to unbounded Gaussians and heavy-tailed data. [Adaptive data analysis](05-adaptive-data-analysis.md) shows that reusing one dataset for questions chosen in sequence makes the answers overfit the sample. Answering through a private mechanism prevents this, because a private algorithm cannot find a question on which the sample and the population disagree.

| Chapter | Question it answers |
| --- | --- |
| [What differential privacy protects](01-privacy-violations.md) | Which inferences from a trained model are privacy violations, and which are learning at work? |
| [Private empirical risk minimization](02-private-erm.md) | How can a convex model be fitted privately, and at what cost in accuracy? |
| [Private deep learning](03-private-deep-learning.md) | How are neural networks trained privately, and how tightly can their privacy be accounted? |
| [Private mean estimation](04-mean-estimation.md) | How many samples does a private estimate of a population mean need? |
| [Adaptive data analysis](05-adaptive-data-analysis.md) | How can one dataset answer adaptively chosen questions without overfitting? |

: The chapters of this part and the questions they answer. {#tbl-p4-chapters}
