# Threat models and attacks

This part teaches you to break models before the rest of the book teaches you to protect them, and the order is deliberate. A privacy defence is a promise that some adversary cannot learn some secret, and you cannot judge the promise until you know who the adversary is, what it sees and how well the best one could do. An evaluation that calls a model private is likewise only as honest as the attack it ran.

Read this part if you have to decide whether a trained model, a released dataset or a set of published statistics leaks information about the people behind it, whether you build the system, audit it or review a report about it. You need the true positive rate (TPR) and false positive rate (FPR) of a test from the [Prerequisites](../part01-prerequisites/00-introduction.md); each chapter restates what it uses. Differential privacy (DP), the defence most of the book is about, comes in the [next part](../part03-dp-foundations/00-introduction.md), and it is built to defeat exactly the games set up here.

## The common thread

Every attack in this part is a **game**. A challenger trains a model or releases statistics that depend on a hidden **secret**, the adversary gets the view its **threat model** allows, and it guesses the secret. Only the secret changes from chapter to chapter: whether a record was used, a sensitive field, a memorized string, a name, the whole dataset. Writing attacks this way forces two questions that reports often leave open: what was the adversary assumed to see and know, and how well would a guess have done without the view?

When the secret is one bit, the game is a hypothesis test. The adversary sees one output and must decide which world produced it, the one with the record or the one without. The best it can do is ask how much more likely that output is in one world than in the other, and accuse where that ratio is largest. This best test depends only on the two distributions of what the adversary sees, so its success caps every attack, including those not yet invented. The strongest practical membership attack, the likelihood ratio attack (LiRA), approximates it, which is why it is the one to run in an audit.

The second idea is that the worst case is not the average case. A model leaks most about the records it cannot predict without having seen them: outliers, duplicated strings, unique combinations of attributes. An average over all records dilutes those few into a harmless-looking number. Yet a verdict like "this person was in the data" only harms someone when it is confident, which requires that the attack almost never accuses the wrong person. So attacks are scored by their TPR at a low FPR, and a defence has to hold for every record.

A third idea closes the part: answer enough statistics accurately enough and the answers pin the data down, however they were computed. Some noise is not optional, and the next part asks how little is enough.

## Why the chapters come in this order

[Threat models and privacy games](01-threat-models.md) comes first because it fixes the vocabulary the others use: view, game, advantage, baseline, and the ceiling on any attack. [Membership inference](02-membership-inference.md) follows because its secret is the smallest, one bit, and the attacks on models after it build on it; it is also the attack you will run most often. The next four chapters attack models for larger secrets, in any order: a sensitive field in [Attribute inference and model inversion](03-attribute-inference.md), a fact about the training population in [Property inference](04-property-inference.md), training text in [Memorization and training-data extraction](05-memorization-extraction.md), and the input behind a gradient in [Gradient inversion](06-gradient-inversion.md).

The last three chapters attack released data, from the weakest defence to the strongest. [Linkage and re-identification](07-linkage-reidentification.md) shows that removing names fails when a few attributes single a person out. [Syntactic anonymity and how it breaks](08-syntactic-anonymity.md) hides each record in a crowd of look-alikes and shows the attacks that get past it. [Reconstruction attacks](09-reconstruction.md) releases only aggregates and shows that enough accurate ones give the data back. Read these in order: each releases less than the one before, and each still leaks.

Each chapter opens with the practical question and the idea that answers it, explains each result with its reason, and ends with a cheat sheet. The examples are toys whose numbers come from the full edition's scripts, which also hold the derivations.

## What you will be able to do

You will be able to state any attack as a game with a threat model and a baseline, and refuse to read its number until you know both. You will run a membership audit with shadow models and LiRA, report it as a TPR at a low FPR with the number of non-members behind it, and explain why the exposed records are the outliers. You will tell a leak about individuals from inference about the population, and know why rounding confidences does not close it. And you will know why removing names, hiding records in crowds and publishing only aggregates each fail, which is the case for the guarantee of the next part.
