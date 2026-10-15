# Threat models and attacks

This part teaches you to break models before the rest of the book teaches you to protect them. Every defence is measured against an attack. A guarantee is only as meaningful as the strongest adversary it stops, and an evaluation is only as honest as the attack it runs. So the book starts with the adversaries: what they see, what they want, and how well the best of them can do.

You need the [Prerequisites](../part01-prerequisites/00-introduction.md), or their content. This part leans hardest on three of them. It uses hypothesis testing, especially reading a test at a low false-positive rate, the total variation distance, and the link between stability and generalization. The defence that most of the book is about, differential privacy, arrives only in the [next part](../part03-dp-foundations/00-introduction.md). It is designed against exactly the games defined here. The only privacy definitions in this part are the older ones of its second-to-last chapter, such as k-anonymity, and they appear as targets.

## The common thread

Every attack in this part has the same shape. A challenger trains a model or releases data, and what it releases depends on a hidden **secret**. An adversary then guesses the secret from whatever the **threat model** lets it see. Writing an attack this way, as a **game**, makes it precise enough to score and to compare with other attacks. What changes from chapter to chapter is the secret. It can be one bit saying whether a record was used, a person's sensitive attribute, a string the model memorized, a fact about the whole training population, a client's input to federated training, a person's name, or the entire dataset.

Two ideas run through all the chapters. The first is that **an attack is a hypothesis test**. When the secret is a single bit, the adversary must decide which of two worlds produced what it sees. The best possible attack is then the likelihood-ratio test from [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md), and no attack can beat the ceiling that test sets. That ceiling is what lets you say how well the best attack could do, not just how well the attack you tried did.

The second idea is that **the worst case is not the average case**. Attacks succeed first on outliers, on strings repeated many times, on people with a rare combination of attributes, and on the one example in a batch that a neuron singles out. An average over all records can look harmless while a few records are fully exposed. That is why attacks are reported by how many members they catch at a very low false-positive rate. It is also why the defence of the next part has to protect every record, not just the typical one.

A third idea closes the part. Answering too many statistics too accurately gives the data away, however cleverly the answers are computed. This is the **Fundamental Law of Information Recovery**. It means some noise is unavoidable, and the question becomes how little noise is enough.

## Why the chapters come in this order

[Threat models and privacy games](01-threat-models.md) comes first, because it fixes the vocabulary that the other chapters use without redefining it. That vocabulary covers the challenger and the adversary, what the adversary sees, and how an attack is scored. Then the part splits in two.

The first half attacks trained models, from the smallest secret to the largest. [Membership inference](02-membership-inference.md) asks whether one record was in the training set, and it is the reference attack the next chapters build on. Building on it, [Attribute inference](03-attribute-inference.md) asks whether the model reveals a person's sensitive attribute beyond what the population already does. [Property inference](04-property-inference.md) asks whether a model reveals a fact about its training population that no single record reveals. [Memorization and extraction](05-memorization-extraction.md) asks when a model can be made to emit its training text. [Gradient inversion](06-gradient-inversion.md) shows that a gradient computed on private data can give the data back exactly. After membership inference, these chapters can be read in any order.

The second half attacks released data, from the weakest protection to the strongest. [Linkage and re-identification](07-linkage-reidentification.md) shows why removing names does not hide whose record is whose. [Syntactic anonymity](08-syntactic-anonymity.md) starts from that linkage game and asks what hiding records in crowds guarantees, and which attack gets past each version. [Reconstruction attacks](09-reconstruction.md) shows that even exact aggregate statistics, with no records released at all, can rebuild a whole dataset. Read these three in order. The attacks on syntactic anonymity are the requirements that the next part's definition is built to meet. The reconstruction results return there as limits on what any private release can achieve.

## How to read a chapter

Each chapter states its attack as a game, reasons its way to the best possible attack and its ceiling, and measures both on an experiment. The experiments are small enough to run in seconds: logistic regression on synthetic data, a small neural network, a language model built from letter counts with planted secrets. Each is chosen so that the best attack can be computed exactly and compared with the practical one. Where the published experiment needs a large training pipeline, the chapter describes that pipeline in words and reports what it found.

The chapters contain no proofs. Each one ends with a link to the matching chapter of the full edition, which derives every result and has a script that checks every number quoted.

## What you will be able to do

After this part, you will be able to state any privacy attack as a game, name its threat model and its baseline, and bound how well any attack could do. You will be able to run a membership attack and report it properly, by the members it catches at a false-positive rate of one in a thousand or lower.

You will be able to tell what a model reveals about a person from what the population already reveals about them. You will also know how a property of the training population can leak when no single record does. You will know when a language model memorizes a string well enough to emit it, and why averaging gradients over a batch does not hide the inputs.

On the data side, you will be able to estimate how many people a set of attributes makes unique. You will also be able to run a linkage attack like the one that de-anonymized the Netflix Prize data. You will be able to say what k-anonymity, l-diversity and t-closeness each guarantee and which attack breaks each one. Finally, you will be able to explain why answering many statistics accurately lets an attacker rebuild the data behind them.
