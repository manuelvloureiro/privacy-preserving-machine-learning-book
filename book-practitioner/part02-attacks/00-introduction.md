# Threat models and attacks

This part teaches you to break models before the rest of the book teaches you to protect them. Every defence is measured against an attack: a guarantee is only as meaningful as the strongest adversary it stops, and an evaluation is only as honest as the attack it runs. This part describes the adversaries, what they see and want, and how well the best of them can do, and gives you the attacks to run yourself when you audit or red-team a system.

## Who this part is for

Read it if you have to decide whether a trained model, a released dataset or a set of published statistics leaks information about the people behind it, whether you build the system, audit it or review a report about it. You will need a little probability from the [Prerequisites](../part01-prerequisites/00-introduction.md): the true positive rate (TPR) and false positive rate (FPR) of a test, the receiver operating characteristic (ROC) curve, and the total variation distance between two distributions. Each chapter restates what it uses.

Differential privacy (DP), the defence most of the book is about, comes in the [next part](../part03-dp-foundations/00-introduction.md). It is designed against exactly the games defined here. The only privacy definitions in this part are the syntactic ones of its eighth chapter, k-anonymity and its successors, and they appear as targets.

## The common thread

Every attack in this part is the same object: a **game** in which a challenger trains or releases something that depends on a hidden **secret**, and an adversary guesses the secret from what the **threat model** lets it see. Only the secret changes from chapter to chapter, from one bit of membership to the whole dataset. @tbl-attacks-map lists the chapters and the practical question each one answers.

| Chapter | The practical question it answers |
| --- | --- |
| [Threat models and privacy games](01-threat-models.md) | What does an attack assume, and how do you score it honestly? |
| [Membership inference](02-membership-inference.md) | Can someone tell that a record was in the training set, and how do you measure that on your own model? |
| [Attribute inference and model inversion](03-attribute-inference.md) | Does the model reveal a person's sensitive field beyond what the population already reveals? |
| [Property inference](04-property-inference.md) | Can a model reveal a fact about its training population that no single record reveals? |
| [Memorization and training-data extraction](05-memorization-extraction.md) | Can the model be made to emit its training text, and how do you test for it? |
| [Gradient inversion](06-gradient-inversion.md) | What does a shared gradient reveal about the data it was computed on? |
| [Linkage and re-identification](07-linkage-reidentification.md) | When does removing names fail to hide who a record belongs to? |
| [Syntactic anonymity and how it breaks](08-syntactic-anonymity.md) | What do k-anonymity, l-diversity and t-closeness promise, and which attacks get past each? |
| [Reconstruction attacks](09-reconstruction.md) | How many accurate statistics does it take to rebuild a whole private dataset? |

: The chapters of this part and the practical question each one answers. {#tbl-attacks-map}

Two ideas run through all nine chapters. The first is that **an attack is a hypothesis test.** When the secret is one bit, the best attack is a likelihood-ratio test, and its **advantage**, TPR minus FPR, can never exceed the total variation distance between the adversary's views in the two worlds. The strongest practical attacks, such as the likelihood ratio attack (LiRA) on membership, approximate that test, and they are what you should run when you audit.

The second is that **the worst case is not the average case.** Attacks succeed first on outliers, on duplicated strings, on unique combinations of attributes and on the one example in a batch that a neuron isolates. An average over records can look harmless while a few records are fully exposed. That is why attacks are reported by their TPR at a low FPR, not by accuracy or area under the curve (AUC), and why the defence of the next part has to hold for every record.

A third idea closes the part: answering too many statistics too accurately destroys privacy however the answers are computed. Some noise is not optional, and the question becomes how little is enough.

## How to read this part

Read [Threat models and privacy games](01-threat-models.md) first: it fixes the vocabulary (view, game, advantage, baseline) the other chapters use. Then read [Membership inference](02-membership-inference.md), the reference attack and the one you will run most often in an audit. The next four chapters can be read in any order. The last three attack released data rather than trained models, from the weakest defence to the strongest: names removed, records hidden in crowds, only aggregates published. Read them in order if you publish data or statistics.

Every chapter opens with the practical question and a table of the decisions it covers, gives each topic as an idea, a result, a worked example and the pitfalls, and closes with a cheat sheet. The examples are small toys whose numbers come from the full edition's scripts; where an audit needs the full-size pipeline, the chapter says how to run it. The derivations are in the full edition.

## What you will be able to do

By the end of this part you will be able to state any privacy attack as a game with its threat model and baseline, and refuse to read a number until you know both. You will be able to run a membership audit with shadow models and LiRA, report it as a TPR at a low FPR with the number of non-members behind it, and say which records are exposed and why. You will tell an attribute leak about individuals from population inference, and know why rounding confidences or returning labels only does not close it. You will also know when memorized text can be extracted, what a shared gradient gives away, how few attributes make a person unique, what k-anonymity does not protect against, and why publishing many accurate statistics amounts to publishing the data.
