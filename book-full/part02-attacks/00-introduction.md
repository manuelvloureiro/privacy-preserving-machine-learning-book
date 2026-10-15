# Threat models and attacks

This part teaches you to break models before it teaches you to protect them. Every defence in the rest of the book is measured against an attack: a guarantee is only as meaningful as the strongest adversary it stops, and an evaluation is only as honest as the attack it runs. So you start here, with the adversaries, what they see, what they want, and how well the best of them can do.

## Who this part is for

You are expected to have read the [Prerequisites](../part01-prerequisites/00-introduction.md), or to know their content. This part leans hardest on hypothesis testing (the Neyman–Pearson lemma, trade-off curves, receiver operating characteristic (ROC) curves read at a low false positive rate), on the total variation distance, and on the link between stability and generalization. Information theory appears in the chapters on attribute inference, memorization and linkage. You should also be comfortable with logistic regression, gradients of a small neural network, and an n-gram language model at the level of writing them in NumPy.

Differential privacy, the defence most of the book is about, is introduced in the [next part](../part03-dp-foundations/00-introduction.md), and it is designed against exactly the games defined here. The only privacy definitions in this part are the syntactic ones of the last chapter, k-anonymity and its successors, and they appear as targets: the chapter shows what each one promises and the attack that breaks it.

## The common thread

Every attack in this part is the same object: a **game** in which a challenger trains or releases something that depends on a hidden **secret**, and an adversary guesses the secret from what the **threat model** lets it see. What changes from chapter to chapter is the secret: one bit of membership, a sensitive attribute, a memorized string, the whole dataset, a client's input, a property of the population behind the data, or a person's name. @tbl-attacks-map lists them.

| Chapter | The question it answers | Builds on |
| --- | --- | --- |
| [Threat models and privacy games](01-threat-models.md) | What can an adversary see and do, and how is an attack stated and scored? | 3, 4, 8 |
| [Membership inference](02-membership-inference.md) | Was this record in the training set? | 4, 7, 9 |
| [Attribute inference and model inversion](03-attribute-inference.md) | Does the model reveal a person's sensitive attribute beyond what the population already does? | 4, 5, 9, 10 |
| [Property inference](04-property-inference.md) | Can a model reveal a fact about its training population that no single record reveals? | 3, 4, 9, 10 |
| [Memorization and training-data extraction](05-memorization-extraction.md) | Can the model be made to emit its training text, and when? | 5, 7, 9, 10 |
| [Gradient inversion](06-gradient-inversion.md) | What does a gradient computed on private data reveal about that data? | 6, 9 |
| [Linkage and re-identification](07-linkage-reidentification.md) | When does removing names fail to hide who a record belongs to? | 1, 2, 5, 9 |
| [Syntactic anonymity and how it breaks](08-syntactic-anonymity.md) | What do k-anonymity, l-diversity and t-closeness guarantee, and which attacks get past each? | 1, 3, 5, 11, 15 |
| [Reconstruction attacks](09-reconstruction.md) | How many accurate statistics does it take to rebuild a whole private dataset? | 2, 5, 9 |

: The chapters of this part, the question each one answers, and the earlier chapters it uses. {#tbl-attacks-map}

Two ideas run through all nine chapters.

- **An attack is a hypothesis test.** When the secret is one bit, the best attack is the likelihood-ratio test of the Neyman–Pearson lemma, its **advantage** (true positive rate (TPR) minus false positive rate (FPR)) is at most the total variation distance between the adversary's two possible views, and its whole error trade-off is bounded by one curve. Membership inference, attribute inference and property inference are each this statement with a different pair of views. The strongest practical attacks, such as the likelihood ratio attack (LiRA) on membership, are explicit approximations of the optimal test.
- **The worst case is not the average case.** Attacks succeed first on outliers, on duplicated strings, on unique combinations of attributes and on the one example in a batch that a neuron isolates. An average over records can look harmless while a few records are fully exposed. This is why attacks are reported as the TPR at a low FPR rather than as accuracy or area under the curve (AUC), and why the defence of the next part has to hold for every record and every dataset.

A third idea closes the part and frames what comes after it. Reconstruction attacks show that answering too many statistics too accurately destroys privacy no matter how the answers are computed: this is the **Fundamental Law of Information Recovery**. Some noise is not optional, and the question becomes how little noise is enough.

## How to read this part

**Order.** The part has two halves. Chapters 9 to 14 attack trained models, from the smallest secret to the largest: one bit of membership, a sensitive attribute, a property of the training population, memorized text, and whole inputs recovered from gradients. Chapters 15 to 17 attack released data, from the weakest defence to the strongest: names removed (linkage), records hidden in crowds (syntactic anonymity), and only aggregate statistics published (reconstruction).

Read chapter 9 first. It fixes the vocabulary (challenger, adversary, view, game, advantage, baseline) that the others use without redefining. Then read chapter 10, since membership inference is the reference attack that chapters 11, 12 and 13 extend. Chapters 11 to 14 can then be read in any order. Read chapters 15 to 17 in order: chapter 16 starts from the linkage game of chapter 15 and uses the imputation baseline of chapter 11, and chapter 17 shows that even exact aggregate statistics, with no records released at all, give the data away. Chapters 16 and 17 matter most for what comes next: the attacks of chapter 16 are the requirements the next part's definition is built to meet, and the results of chapter 17 return as lower bounds there.

**Structure of a chapter.** Every chapter has the same shape as in the prerequisites: why the topic matters, a toolkit, one section per result with a context, statement, derivation, interpretation, worked example and two *check yourself* questions, then code, references and a cheat sheet. Exercises with worked solutions are collected in the exercise book, a companion website, with a link at the end of each chapter.

**Toy experiments, real pipelines.** The attacks in the literature run on image classifiers and large language models trained on GPUs. Here every experiment is a NumPy toy that runs in seconds: logistic regression on synthetic data, Gaussian mean estimation, a small multilayer perceptron written by hand, an n-gram language model with planted canaries. Each toy is chosen so that the optimal attack can be computed exactly and compared with the practical one. Where a chapter's study-plan exercise needs the full pipeline (such as LiRA with 16 shadow models on CIFAR-10), the chapter describes that pipeline in prose and runs a faithful scaled-down version.

**Run the code.** From the repository root, for example,

```bash
uv run python code/part02-attacks/membership_inference.py
```

The scripts use fixed seeds, so their output matches the tables in the text.

## Notation

The chapters share the conventions in @tbl-attacks-notation, set in [Threat models and privacy games](01-threat-models.md). Where a chapter departs from them, it says so at its start.

| Symbol | Meaning |
| --- | --- |
| $D$, $\pi$ | A dataset of records; the population they are drawn from |
| $T$, $\theta = T(D)$ | A (possibly randomized) training algorithm, and the model it outputs |
| $b$ | The hidden bit of a two-world game; $b = 1$ is the world where the target record is used (the member world) |
| $V$; $P$, $Q$ | The adversary's view; its distributions when $b = 0$ and $b = 1$ |
| $\mathrm{TPR}$, $\mathrm{FPR}$ | $\Pr[b' = 1 \mid b = 1]$ and $\Pr[b' = 1 \mid b = 0]$ for the guess $b'$ |
| $\mathrm{Adv}$ | The advantage, $\mathrm{TPR} - \mathrm{FPR}$, at most $\mathrm{TV}(P, Q)$ |
| $\ell(\theta, z)$ | The loss of model $\theta$ on record $z$ |

: Notation shared by the chapters of this part. {#tbl-attacks-notation}

## What you will be able to do

By the end of this part you should be able to:

- state any privacy attack as a game, name its threat model and its baseline, and bound its advantage by a total variation distance;
- derive the loss-threshold membership attack from the generalization gap and LiRA from the Neyman–Pearson lemma, and evaluate both by the TPR at a low FPR;
- separate attribute inference from what the population already reveals, and say why masking confidences does not stop it;
- explain how a property of the training population can leak when no individual record does;
- define verbatim, discoverable and extractable memorization, and compute a canary's exposure;
- recover an input exactly from the gradient of a fully connected layer, and say why averaging a batch does not remove the leak;
- compute the fraction of a population made unique by a set of quasi-identifiers, and run a Netflix-style de-anonymization;
- say what k-anonymity, l-diversity and t-closeness each guarantee, and break each one with homogeneity, skewness and composition attacks;
- prove that answers with error $o(\sqrt n)$ to random subset sums allow almost all of a dataset to be reconstructed.

Each of these is a result derived in one of the chapters that follow.
