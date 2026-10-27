# Private Learning and Statistics

The earlier parts released statistics of a fixed dataset under differential privacy (DP). This part turns to the goals of machine learning and statistics, which are to fit models and to learn about the population that the data comes from. It starts with the question of what privacy must protect, because a model can predict a person's sensitive label without any use of that person's record. Then it trains convex models and neural networks privately, and finds how many samples a private estimate of a population mean needs. Last, it shows that a private mechanism lets one dataset answer adaptively chosen questions without overfitting. @tbl-p4-chapters gives the question of each chapter.

| Chapter | Question it answers |
| --- | --- |
| [What differential privacy protects](01-privacy-violations.md) | Which inferences from a trained model are privacy violations, and which are learning at work? |
| [Private empirical risk minimization](02-private-erm.md) | How can a convex model be fitted privately, and at what cost in accuracy? |
| [Private deep learning](03-private-deep-learning.md) | How are neural networks trained privately, and how tightly can their privacy be accounted? |
| [Private mean estimation](04-mean-estimation.md) | How many samples does a private estimate of a population mean need? |
| [Adaptive data analysis](05-adaptive-data-analysis.md) | How can one dataset answer adaptively chosen questions without overfitting? |

: The chapters of this part and the questions they answer. {#tbl-p4-chapters}
