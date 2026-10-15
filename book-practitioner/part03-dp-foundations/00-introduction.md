# Differential privacy foundations

The previous part showed that models and statistics leak. They can reveal who was in the training data, a person's sensitive attributes, memorized text, and even whole datasets. This part introduces the defence that the rest of the book is built on, **differential privacy (DP)**. It goes far enough that you can check whether a mechanism is private, calibrate its noise and combine it with other mechanisms. You will also be able to say what any private mechanism must cost in accuracy. The finer tools for tracking privacy over many training steps come in the [next part](../part04-privacy-accounting/00-introduction.md), which builds directly on this one.

You need the [Prerequisites](../part01-prerequisites/00-introduction.md) and [Threat models and attacks](../part02-attacks/00-introduction.md). This part uses the divergences and the hypothesis-testing view of Part 1, and the tail bounds for Laplace and normal noise. It also uses the privacy games of Part 2, because the definition of differential privacy is a bound on exactly those games.

## The common thread

A randomized algorithm is differentially private when changing one person's record changes the distribution of its output only a little. "A little" means by at most a factor $e^\varepsilon$, where the number $\varepsilon$ is the privacy parameter, up to a small slack $\delta$. Everything in this part follows from reading that one sentence in three ways.

The first reading is as a **bound on a divergence** between the output distributions on two datasets that differ in one record. Because processing data can never increase a divergence, anything computed from a private output is just as private. The log of the likelihood ratio between the two outputs, called the **privacy loss**, is the quantity whose sum tracks privacy over many releases.

The second reading is as a **bound on every attack**. An attacker who tries to tell whether a record was used is running a hypothesis test, and differential privacy limits how good that test can be at every false-positive rate. So every game of Part 2, from membership inference to reconstruction, is capped by a number you can compute before the data is touched.

The third reading is as a **budget to spend**. Each release uses up some privacy. Noise is set by how much one record can move the answer, which is called its **sensitivity**. The losses of several releases add up under composition, and lower bounds say how much accuracy no mechanism can avoid giving up.

## Why the chapters come in this order

The first four chapters are the core, and each needs the one before. [The definition of differential privacy](01-definition.md) states it, says what counts as changing one record, and translates a guarantee into a bound on an attacker. [Post-processing, group privacy and composition](02-properties.md) shows what makes the definition usable: processing an output cannot weaken it, it extends to groups of records, and guarantees add up when mechanisms are combined. [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md) gives the two ways of adding noise that almost everything else is built from, and shows how to calibrate each. [Advanced and optimal composition](04-composition.md) finishes the story of combining mechanisms. It gives the tighter accounting that the next part relies on, and the privacy loss that it is built from.

The next two chapters add mechanisms for harder cases. [Selection](05-selection.md) covers choosing the best of many options and answering only the queries that cross a threshold. [Beyond global sensitivity](06-sensitivity.md) handles data whose worst case is far from its typical case, where noise calibrated to the worst case would be wasteful.

[Discrete and floating-point-safe mechanisms](07-discrete-mechanisms.md) is short and practical, and you should read it before you implement any mechanism yourself. It shows why naive floating-point noise is not private and how to sample noise exactly. [Local, shuffle and distributed trust models](08-trust-models.md) changes who is trusted with the raw data, from a central curator to each person adding noise locally, and says what shuffling and secure aggregation buy in between. [Lower bounds](09-lower-bounds.md) is the most theoretical chapter. It ties back to the reconstruction and membership attacks of Part 2 and closes the part with what privacy must cost.

## How to read a chapter

Each chapter starts from the practical question it answers, reasons its way to each result, and shows the result at work with real numbers. Every formula you need to calibrate or check a mechanism is kept and explained in words, together with the conditions it needs and what goes wrong without them. A cheat sheet at the end collects the results.

The chapters contain no proofs. Each one ends with a link to the matching chapter of the full edition, which proves every result. Its scripts also check most claims numerically, for example by computing a mechanism's exact privacy and comparing it with the bound.

## What you will be able to do

After this part, you will be able to state a differential-privacy guarantee precisely, including what counts as one person's data. You will also be able to translate it into a bound on how well any attacker can do. You will know which properties make the definition safe to build on, and which combinations do not compose the way you might expect.

You will be able to calibrate the Laplace and Gaussian mechanisms, and account for privacy across many releases with both the simple and the tight composition rules. You will be able to choose among options privately and handle data whose sensitivity depends on the data.

You will know how to sample noise without the floating-point bugs that break real implementations. You will be able to design a release in which each person adds their own noise, and say what that costs. Finally, you will be able to say why some accuracy loss is unavoidable at a given privacy level, and roughly how much.
