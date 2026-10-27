# Property inference

## What it is

A **property** is a fact about a training set as a whole, such as the share of a hospital's patients who are over 65. **Property inference** recovers it from a trained model. In the property game ([Threat models and privacy games](01-threat-models.md)), the data owner flips a fair coin to pick one of two shares for a subgroup. It then trains a model on data with that share. The attacker knows both shares, can train models the way the owner does, and guesses which share was used, so a random guess is right half the time.

A property can leak when no single record does. Write $n$ for the number of training records and $g$ for the gap between the two shares. Switching shares changes about $n g$ records at once, so it moves the model far more than one record can. The leak grows with $n$, while the risk to one record shrinks. Even a release of only the exact count of subgroup records shows this. At 10,000 records, the best attacker asking whether one patient is in the data is right with probability 0.502. The best one asking whether the share was 0.45 or 0.55 is right with probability 1.0000.

## When you meet it

You meet it when you share or publish a model trained on your organization's data. First decide whether the property is a secret at all. It is harmful to reveal only when it describes the *data owner* rather than the world. One hospital's patient mix is a secret, but that smokers get more lung cancer is the point of a medical study. If it is not a secret, record that in the threat model of [Pre-release privacy risk assessment](10-risk-assessment.md) and move on.

If it is a secret, differential privacy (DP) does not protect it. DP protects one record at a time, and its guarantee for a group multiplies the privacy level $\varepsilon$ by the group's size ([Post-processing, group privacy and composition](../part03-dp-foundations/02-properties.md)). For a hospital of 1,000 patients and shares a tenth apart, the group is the $n g = 100$ patients that differ between the two datasets. So the property gets only $100\varepsilon$, which promises nothing at any useful $\varepsilon$.

## How to use it

To measure the risk, run the shadow-model attack of Ateniese et al. (2015), which treats whole models as data points. Draw many datasets of your training set's size at each of the two shares, from auxiliary data of the same population or from your own data. Train a **shadow model** on each with your architecture and training code, and record its outputs on a fixed set of probe inputs. A linear **meta-classifier**, such as scikit-learn's `LogisticRegression`, learns to tell the two shares apart from those outputs, and its accuracy on held-out shadow models is the attack's accuracy. If the property is a secret, block the release when that accuracy's confidence interval lies above one half ([Pre-release privacy risk assessment](10-risk-assessment.md#sec-risk-threat)).

Use probe outputs rather than raw weights, because reordering hidden units, or flipping the signs of tanh units, changes the weights but not the network's function. On the full edition's small tanh networks, every raw weight therefore averages to zero under both shares. A linear meta-classifier on raw weights scores 0.497, which is chance, and on the probe outputs 1.000. Probes need only query access, so a model behind an application programming interface is exposed too.

## What goes wrong

The commonest error is to assume that DP, or a passed membership audit, covers the property. Nor is a weak measured attack evidence of safety, because with few records and a small gap, a meta-classifier fitted to noisy shadow models finds the shift badly.

There is no cheap defence. Removing the group column leaves its correlates, and noise large enough to hide $n g$ records destroys the model's use. If the property is a secret, the strongest option is not to release a model trained on that mix. If you must, report the attack's accuracy and state in the release notes that DP protects individuals only.

## References

- Ganju, Wang, Yang, Gunter & Borisov, "Property Inference Attacks on Fully Connected Neural Networks using Permutation Invariant Representations" (CCS 2018): the practical attack on neural networks, and why raw weights fail.
- Zhang, Tople & Ohrimenko, "Leakage of Dataset Properties in Multi-Party Machine Learning" (USENIX Security 2021): property inference through queries alone.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/04-property-inference.md).
