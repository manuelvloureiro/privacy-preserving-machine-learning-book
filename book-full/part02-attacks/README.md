# Part 2 — Threat models and attacks

How models and statistics leak, and how to measure it: threat models and privacy games, membership inference, attribute inference and model inversion, memorization and extraction, reconstruction, gradient inversion, property inference, linkage, and the syntactic anonymity definitions (k-anonymity, l-diversity, t-closeness) with the attacks that break them. No defence with a lasting guarantee is introduced here. Differential privacy starts in [Part 3](../part03-dp-foundations/README.md) and is designed against exactly these games.

Start with the [introduction](00-introduction.md), which is part of the book. Read chapter 9 first (it fixes the vocabulary), then chapter 10. Chapters 11 to 14 (attacks on models) can then be read in any order; read chapters 15 to 17 (attacks on released data) in order.

This file is the GitHub overview. It is not part of the rendered book.

## Chapters

| Ch. | Chapter | Covers | Script |
|-----|---------|--------|--------|
| 9 | [Threat models and privacy games](01-threat-models.md) | Access, behaviour, trust and side information; attacks as games; advantage, total variation and the trade-off curve; average against worst case | [threat_models.py](../../code/part02-attacks/threat_models.py) |
| 10 | [Membership inference](02-membership-inference.md) | The loss attack and the generalization gap; shadow models; LiRA as a Neyman–Pearson test; TPR at low FPR; why outliers are exposed | [membership_inference.py](../../code/part02-attacks/membership_inference.py) |
| 11 | [Attribute inference and model inversion](03-attribute-inference.md) | The attribute game and its link to membership; the imputation baseline; the MAP inversion attack; confidence masking | [attribute_inference.py](../../code/part02-attacks/attribute_inference.py) |
| 12 | [Property inference](04-property-inference.md) | The property game; why a property leaks when no record does; meta-classifiers; permutation invariance | [property_inference.py](../../code/part02-attacks/property_inference.py) |
| 13 | [Memorization and training-data extraction](05-memorization-extraction.md) | Verbatim, discoverable and extractable memorization; duplication; canaries and exposure; extraction without a prefix | [memorization.py](../../code/part02-attacks/memorization.py) |
| 14 | [Gradient inversion](06-gradient-inversion.md) | Exact recovery from a fully connected layer; label recovery; Deep Leakage from Gradients; batches and averaging | [gradient_inversion.py](../../code/part02-attacks/gradient_inversion.py) |
| 15 | [Linkage and re-identification](07-linkage-reidentification.md) | Quasi-identifiers; hashed identifiers and the New York taxi release; how many people are unique; the Netflix attack and eccentricity; why sampling does not protect | [linkage.py](../../code/part02-attacks/linkage.py) |
| 16 | [Syntactic anonymity and how it breaks](08-syntactic-anonymity.md) | k-anonymity and Mondrian; homogeneity and background knowledge; l-diversity; skewness and t-closeness; the composition attack | [syntactic_anonymity.py](../../code/part02-attacks/syntactic_anonymity.py) |
| 17 | [Reconstruction attacks](09-reconstruction.md) | The exponential attack; Dinur–Nissim; least-squares decoding; the Fundamental Law of Information Recovery; the Cohen–Nissim attack on Diffix; a census block rebuilt from its tables | [reconstruction.py](../../code/part02-attacks/reconstruction.py) |

Each script prints the numbers its chapter quotes and regenerates the chapter's figures in `_build/figures/part02-attacks/` (the build copies them into each book):

```bash
uv run python code/part02-attacks/<script>.py
```

Every chapter's exercises, each with a worked solution, are in the exercise book, [book-exercises/part02-attacks/](../../book-exercises/part02-attacks/), one file per chapter. The exercise book is its own website, and `code/part02-attacks/solutions_<script>.py` prints the numbers they quote.

## Core readings

- Shokri et al., "Membership Inference Attacks against Machine Learning Models" (S&P 2017) → chapter 10
- Carlini et al., "Membership Inference Attacks From First Principles" (S&P 2022) → chapter 10
- Carlini et al., "Extracting Training Data from Large Language Models" (USENIX Security 2021) → chapter 13
- Carlini et al., "Quantifying Memorization Across Neural Language Models" (ICLR 2023) → chapter 13
- Dinur & Nissim, "Revealing Information while Preserving Privacy" (PODS 2003) → chapter 17
- Narayanan & Shmatikov, "Robust De-anonymization of Large Sparse Datasets" (S&P 2008) → chapter 15
- Sweeney, "k-Anonymity: A Model for Protecting Privacy" (2002) → chapter 16
- Machanavajjhala et al., "l-Diversity: Privacy Beyond k-Anonymity" (TKDD 2007) → chapter 16
- Li, Li & Venkatasubramanian, "t-Closeness: Privacy Beyond k-Anonymity and l-Diversity" (ICDE 2007) → chapter 16
- Ganta, Kasiviswanathan & Smith, "Composition Attacks and Auxiliary Information in Data Privacy" (KDD 2008) → chapter 16

## Further readings

- Fredrikson et al., "Model Inversion Attacks that Exploit Confidence Information" (CCS 2015) → chapter 11
- Zhu et al., "Deep Leakage from Gradients" (NeurIPS 2019) → chapter 14
- Ganju et al., "Property Inference Attacks on Fully Connected Neural Networks" (CCS 2018) → chapter 12
- Nasr et al., "Scalable Extraction of Training Data from (Production) Language Models" (2023) → chapter 13
- Cohen & Nissim, "Linear Program Reconstruction in Practice" (JPC 2020) → chapter 17
- Garfinkel, Abowd & Martindale, "Understanding Database Reconstruction Attacks on Public Data" (CACM 2019) → chapter 17
- Rocher, Hendrickx & de Montjoye, "Estimating the success of re-identifications in incomplete datasets" (Nature Communications 2019) → chapter 15
- Salem et al., "SoK: Let the Privacy Games Begin!" (S&P 2023) → chapter 9
- LeFevre, DeWitt & Ramakrishnan, "Mondrian Multidimensional k-Anonymity" (ICDE 2006) → chapter 16

Each chapter ends with its own references, with what each is good for.

## Exercises

Train a CIFAR-10 ResNet, run LiRA with 16 shadow models, and report TPR at 0.1% FPR. [Membership inference](02-membership-inference.md) describes this pipeline step by step and runs a scaled-down version (logistic regression, 64 shadow models) in seconds. Keep the pipeline: a planned later part on auditing private models reuses it.

k-anonymize a census extract with ARX, then link it back to a public auxiliary table and measure the re-identification rate. [Syntactic anonymity and how it breaks](08-syntactic-anonymity.md) does this with a Mondrian anonymizer on a synthetic hospital table, then measures what the release reveals about the sensitive attribute and what two releases reveal together.

## Exit check

You are done with this part when you can do each of these on paper, without notes:

- [ ] State the membership, attribute, reconstruction and property games, and prove that every attack's advantage is at most the total variation distance between the two views.
- [ ] Explain why attacks are reported as TPR at a low FPR, and give two attacks with the same AUC that differ greatly there.
- [ ] Derive the advantage of the loss-threshold attack from the generalization gap.
- [ ] Derive LiRA's per-example Gaussian test from the Neyman–Pearson lemma, and say what the online and offline variants trade.
- [ ] Prove that on a non-member no attribute attack beats the imputation baseline.
- [ ] Define discoverable and extractable memorization and the exposure of a canary.
- [ ] Sketch the Dinur–Nissim proof that error $o(\sqrt n)$ on random subset sums allows reconstruction of almost every bit.
- [ ] Recover an input from the gradient of a fully connected layer with bias, and the label from the last-layer gradient.
- [ ] Explain why a dataset property can leak with advantage near 1 while membership advantage is near 0.
- [ ] Explain why hashing an identifier is not anonymization, and why a random pseudonym still links a person's records.
- [ ] Compute the expected fraction of unique people for a given number of bits of quasi-identifier, and explain the eccentricity test.
- [ ] Prove that k-anonymity caps linkage at $1/k$, show a homogeneity and a skewness attack on a k-anonymous table, and run the composition attack on two independent releases.
