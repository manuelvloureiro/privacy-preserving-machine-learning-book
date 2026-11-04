# Private learning and statistics

The earlier parts released statistics of fixed datasets under differential privacy (DP). This part studies model fitting and population inference. It distinguishes prediction from memorization: a model can predict a person's sensitive label without using that person's record. The next chapters cover private training of convex models and neural networks, then bound the sample size for private mean estimation. The final chapter shows how DP lets one dataset answer adaptively chosen questions without overfitting.

| Chapter | Question it answers |
| --- | --- |
| [What differential privacy protects](01-privacy-violations.md) | Which inferences from a trained model are privacy violations, and which are learning at work? |
| [Private empirical risk minimization](02-private-erm.md) | How can a convex model be fitted privately, and at what cost in accuracy? |
| [Private deep learning](03-private-deep-learning.md) | How are neural networks trained privately, and how tightly can their privacy be accounted? |
| [Private mean estimation](04-mean-estimation.md) | How many samples does a private estimate of a population mean need? |
| [Adaptive data analysis](05-adaptive-data-analysis.md) | How can one dataset answer adaptively chosen questions without overfitting? |

: The chapters of this part and the questions they answer. {#tbl-p4-chapters}
