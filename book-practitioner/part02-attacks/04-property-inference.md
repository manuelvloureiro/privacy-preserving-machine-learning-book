# Property inference

## What it is

A **property** describes a training set as a whole, such as the share of a hospital's patients over age 65. **Property inference** attempts to recover that property from a trained model. In the game described in [Threat models and privacy games](01-threat-models.md), a fair coin selects one of two subgroup shares. The owner trains on data with the selected share, and an attacker who knows both shares and the training method guesses which was used. Random guessing succeeds half the time.

A property may leak even when no individual record does. Let $n$ be the number of training records and $g$ the gap between the two shares. Changing the share changes about $ng$ records, so its effect on the model can exceed that of changing one record. The property leak can therefore grow with $n$ even as the risk to an individual decreases.

Even a release of only the exact count of subgroup records shows this gap. With 10,000 records, the best guess of whether one patient is in the data is right with probability 0.502. The best guess of whether the share was 0.45 or 0.55 is right with probability 1.0000.

## When you meet it

Assess property inference before sharing or publishing a model trained on organizational data. First determine whether the property is secret—whether it describes the *data owner* rather than the general population. A hospital's patient mix may be secret; the association between smoking and lung cancer may be the intended finding of a medical study. If the property is not secret, document that in the threat model of [Pre-release privacy risk assessment](10-risk-assessment.md).

Differential privacy (DP) does not protect a secret property of the whole dataset. It protects one record at a time; for a group, the bound scales with the group's size ([Post-processing, group privacy and composition](../part03-dp-foundations/02-properties.md)). For a hospital with 1,000 patients and subgroup shares that differ by 0.1, the datasets differ in $ng=100$ patients. The group guarantee is therefore $100\varepsilon$, which is not useful for typical values of $\varepsilon$.

## How to use it

Measure risk with the shadow-model attack of Ateniese et al. (2015), which treats models as data points. Draw many datasets of the training set's size from the same population, with each of the two shares. Train a **shadow model** on each dataset using the target architecture and training procedure. Record each model's outputs on a fixed set of probe inputs.

Fit a linear **meta-classifier**, such as scikit-learn's `LogisticRegression`, to distinguish the two shares from the recorded outputs. Its accuracy on held-out shadow models measures the attack. If the property is secret, block release when the confidence interval for accuracy lies above one half ([Pre-release privacy risk assessment](10-risk-assessment.md#sec-risk-threat)).

Use probe outputs rather than raw weights. Permuting hidden units or flipping the signs of tanh units changes the weights without changing the network function. In the full edition's small tanh-network experiment, each raw weight averages to zero under both shares. A linear meta-classifier scores 0.497 from raw weights, near chance, and 1.000 from probe outputs. Because probes require only query access, this attack also applies to models behind an application programming interface.

## What goes wrong

Do not assume that DP or a passed membership audit protects a dataset property. A weak measured attack also does not establish safety: a meta-classifier trained on noisy shadow models may fail to detect a small shift.

Removing the group column does not remove its correlates. Noise large enough to hide $ng$ records may destroy model utility. If the property is secret, do not release a model trained on that data mix. If release is necessary, report the attack accuracy and state that DP protects individuals, not dataset properties.

## References

- Ganju, Wang, Yang, Gunter & Borisov, "Property Inference Attacks on Fully Connected Neural Networks using Permutation Invariant Representations" (CCS 2018): the practical attack on neural networks, and why raw weights fail.
- Zhang, Tople & Ohrimenko, "Leakage of Dataset Properties in Multi-Party Machine Learning" (USENIX Security 2021): property inference through queries alone.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/04-property-inference.md).
