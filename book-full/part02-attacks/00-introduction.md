# Threat models and attacks

This part teaches you to break models before it teaches you to protect them. Every defence in the rest of the book is measured against an attack. A guarantee means only as much as the strongest adversary it stops, and an evaluation is only as honest as the attack it runs. So you start with the adversaries: what they see, what they want, and how well the best of them can do.

## Who this part is for

You are expected to know the content of the [Prerequisites](../part01-prerequisites/00-introduction.md). This part uses [hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) most: the Neyman–Pearson lemma, trade-off curves, and receiver operating characteristic (ROC) curves read at a low false positive rate. It also uses the total variation distance of [Divergences](../part01-prerequisites/03-divergences.md) and the link between [stability and generalization](../part01-prerequisites/07-stability-generalization.md). [Information theory](../part01-prerequisites/05-information-theory.md) appears in the chapters on attribute inference, memorization and linkage. You should be able to write logistic regression, the gradients of a small neural network and an n-gram language model in NumPy.

Differential privacy, the defence most of the book is about, starts in the [next part](../part03-dp-foundations/00-introduction.md), and it is designed against the games defined here. The only privacy definitions in this part are the syntactic ones of the last chapter, k-anonymity and its successors. They appear as targets: the chapter shows what each one promises and the attack that breaks it.

## The common thread

Every attack in this part is a **game**. A challenger trains or releases something that depends on a hidden **secret**, and an adversary guesses the secret from what the **threat model** lets it see. Only the secret changes from chapter to chapter: one bit of membership, a sensitive attribute, a memorized string, the whole dataset, a client's input, a property of the population, or a person's name. @tbl-attacks-map lists them.

| Chapter | The question it answers | Builds on |
| --- | --- | --- |
| [Threat models and privacy games](01-threat-models.md) | What can an adversary see and do, and how is an attack stated and scored? | 3, 4, 8 |
| [Membership inference](02-membership-inference.md) | Was this record in the training set? | 4, 7, 9 |
| [Attribute inference and model inversion](03-attribute-inference.md) | Does the model reveal a person's sensitive attribute beyond what the population already does? | 3–5, 9, 10 |
| [Property inference](04-property-inference.md) | Can a model reveal a fact about its training population that no single record reveals? | 1, 3, 4, 9, 10 |
| [Memorization and training-data extraction](05-memorization-extraction.md) | Can the model be made to emit its training text, and when? | 4, 5, 7, 9, 10 |
| [Gradient inversion](06-gradient-inversion.md) | What does a gradient computed on private data reveal about that data? | 6, 9 |
| [Linkage and re-identification](07-linkage-reidentification.md) | When does removing or hashing names fail to hide who a record belongs to? | 1, 2, 4, 5, 8, 9 |
| [Syntactic anonymity and how it breaks](08-syntactic-anonymity.md) | What do k-anonymity, l-diversity and t-closeness guarantee, and which attacks get past each? | 1, 3, 5, 11, 15 |
| [Reconstruction attacks](09-reconstruction.md) | How many accurate statistics does it take to rebuild a whole private dataset? | 2, 5, 9 |

: The chapters of this part, the question each one answers, and the earlier chapters it uses. {#tbl-attacks-map}

Two ideas run through all nine chapters. The first is that **an attack is a hypothesis test**. When the secret is one bit, the best attack is the likelihood-ratio test of the Neyman–Pearson lemma. Its **advantage**, the true positive rate (TPR) minus the false positive rate (FPR), is at most the total variation distance between the adversary's two possible views. One curve also bounds its whole error trade-off. Membership, attribute and property inference are each this statement with a different pair of views. The strongest practical attacks, such as the likelihood ratio attack (LiRA) on membership, approximate the optimal test.

The second idea is that **the worst case is not the average case**. Attacks succeed first on outliers, on duplicated strings, on unique combinations of attributes and on the one example in a batch that a neuron isolates. So an average over records can look harmless while a few records are fully exposed. That is why attacks are reported as the TPR at a low FPR, not as accuracy or area under the curve (AUC). It is also why the defence of the next part must hold for every record and every dataset.

A third idea closes the part. Reconstruction attacks show that answering too many statistics too accurately destroys privacy, however the answers are computed. This is the **Fundamental Law of Information Recovery**. So some noise is necessary, and the question becomes how little noise is enough.

The second half of the part tells one story about releasing less, in which each release fails. Records with the names removed or replaced by codes fail to linkage. Coarsened records fail to attacks on the sensitive attribute. Statistics alone fail to reconstruction, and a fixed amount of noise on each statistic fails once enough statistics are asked, as an attack on a deployed query system shows. A few hundred published averages can even reveal whether one person is in the data, the tracing attack of [Lower bounds](../part03-dp-foundations/10-lower-bounds.md#sec-lower-tracing). The definition of the next part answers this run of failures.

## How to read this part

**Order.** Chapter 9 sets up the vocabulary, and two halves follow. Chapters 10 to 14 attack trained models, from the smallest secret to the largest: membership, a sensitive attribute, a property of the training population, memorized text, and whole inputs recovered from gradients. Chapters 15 to 17 attack released data, from the weakest defence to the strongest: names removed (linkage), records hidden in crowds (syntactic anonymity), and only aggregate statistics published (reconstruction).

Read chapter 9 first, because it fixes the vocabulary (challenger, adversary, view, game, advantage, baseline) that the others use. Then read chapter 10, because membership inference is the reference attack that chapters 11, 12 and 13 extend. You can read chapters 11 to 14 in any order. Read chapters 15 to 17 in order. Chapter 16 starts from the linkage game of chapter 15 and uses the imputation baseline of chapter 11, and chapter 17 shows that even exact aggregate statistics give the data away. The attacks of chapter 16 are the requirements that the next part's definition is built to meet, and the results of chapter 17 return there as lower bounds.

**Toy experiments, real pipelines.** The attacks in the literature run on image classifiers and large language models trained on GPUs. Here every experiment is instead a NumPy toy that runs in seconds, such as logistic regression on synthetic data or an n-gram language model with planted canaries. Each toy is small enough that the optimal attack can be computed exactly and compared with the practical one. Where the published experiment needs a full pipeline, such as LiRA with 16 shadow models on CIFAR-10, the chapter describes it and runs a faithful scaled-down version.

**Run the code.** From the repository root, for example,

```bash
uv run python code/part02-attacks/membership_inference.py
```

The scripts use fixed seeds, so their output matches the tables in the text.

## Notation

The chapters share the conventions of @tbl-attacks-notation, which [Threat models and privacy games](01-threat-models.md) sets. A chapter that departs from them says so at its start.

| Symbol | Meaning |
| --- | --- |
| $D$, $\mathcal D$ | A dataset of records; the data distribution (the population) they are drawn from |
| $\mathcal T$, $\theta = \mathcal T(D)$ | A (possibly randomized) training algorithm, and the model it outputs |
| $\mathcal A$ | The adversary |
| $b$, $b'$ | The hidden bit of a two-world game, and the adversary's guess; $b = 1$ is the world where the target record is used (the member world) |
| $\pi$ | A prior over a secret; $\pi_1$ is the prior probability of $b = 1$ |
| $V$; $P$, $Q$ | The adversary's view; its distributions when $b = 0$ and $b = 1$ |
| $\mathrm{TPR}$, $\mathrm{FPR}$ | $\Pr[b' = 1 \mid b = 1]$ and $\Pr[b' = 1 \mid b = 0]$ |
| $\mathrm{Adv}$ | The advantage, $\mathrm{TPR} - \mathrm{FPR}$, at most $\mathrm{TV}(P, Q)$ |
| $T(P, Q)$ | The trade-off function: the smallest false negative rate $1 - \mathrm{TPR}$ of any test at each FPR |
| $\ell(\theta; z)$ | The loss of model $\theta$ on record $z$ |

: Notation shared by the chapters of this part. {#tbl-attacks-notation}

## What you will be able to do

By the end of this part you should be able to:

- state any privacy attack as a game, name its threat model and its baseline, and bound its advantage by a total variation distance;
- derive the loss-threshold membership attack from the generalization gap and LiRA from the Neyman–Pearson lemma, and evaluate both by the TPR at a low FPR;
- separate attribute inference from what the population already reveals, and say why masking confidences does not stop it;
- explain how a property of the training population can leak when no individual record does;
- define verbatim, discoverable and extractable memorization, and compute a canary's exposure;
- recover an input exactly from the gradient of a fully connected layer, and say why averaging a batch does not remove the leak;
- invert a hashed identifier by enumeration, compute the fraction of a population made unique by a set of quasi-identifiers, and run a Netflix-style de-anonymization;
- say what k-anonymity, l-diversity and t-closeness each guarantee, and break each one with homogeneity, skewness and composition attacks;
- prove that answers with error $o(\sqrt n)$ to random subset sums allow almost all of a dataset to be reconstructed, and predict how many queries an attack on a deployed query system needs.

Each of these is a result derived in one of the chapters that follow.
