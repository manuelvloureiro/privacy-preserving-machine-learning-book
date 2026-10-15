# Contents

The book follows the [study plan](study_plan.md), one phase per part. Each folder is a part, and each file in it is a chapter, numbered across the whole book as in the rendered edition.

## Part 1 · [Prerequisites](part01-prerequisites/README.md) · [introduction](part01-prerequisites/00-introduction.md)

1. [Probability distributions](part01-prerequisites/01-distributions.md) · [code](../code/part01-prerequisites/distributions.py)
2. [Concentration inequalities](part01-prerequisites/02-concentration.md) · [code](../code/part01-prerequisites/concentration.py)
3. [Divergences](part01-prerequisites/03-divergences.md) · [code](../code/part01-prerequisites/divergences.py)
4. [Hypothesis testing](part01-prerequisites/04-hypothesis-testing.md) · [code](../code/part01-prerequisites/hypothesis_testing.py)
5. [Information theory](part01-prerequisites/05-information-theory.md) · [code](../code/part01-prerequisites/information_theory.py)
6. [Optimization](part01-prerequisites/06-optimization.md) · [code](../code/part01-prerequisites/optimization.py)
7. [Algorithmic stability and generalization](part01-prerequisites/07-stability-generalization.md) · [code](../code/part01-prerequisites/stability.py)
8. [Cryptography basics](part01-prerequisites/08-crypto-basics.md) · [code](../code/part01-prerequisites/crypto_basics.py)

## Part 2 · [Threat models and attacks](part02-attacks/README.md) · [introduction](part02-attacks/00-introduction.md)

9. [Threat models and privacy games](part02-attacks/01-threat-models.md) · [code](../code/part02-attacks/threat_models.py)
10. [Membership inference](part02-attacks/02-membership-inference.md) · [code](../code/part02-attacks/membership_inference.py)
11. [Attribute inference and model inversion](part02-attacks/03-attribute-inference.md) · [code](../code/part02-attacks/attribute_inference.py)
12. [Property inference](part02-attacks/04-property-inference.md) · [code](../code/part02-attacks/property_inference.py)
13. [Memorization and training-data extraction](part02-attacks/05-memorization-extraction.md) · [code](../code/part02-attacks/memorization.py)
14. [Gradient inversion](part02-attacks/06-gradient-inversion.md) · [code](../code/part02-attacks/gradient_inversion.py)
15. [Linkage and re-identification](part02-attacks/07-linkage-reidentification.md) · [code](../code/part02-attacks/linkage.py)
16. [Syntactic anonymity and how it breaks](part02-attacks/08-syntactic-anonymity.md) · [code](../code/part02-attacks/syntactic_anonymity.py)
17. [Reconstruction attacks](part02-attacks/09-reconstruction.md) · [code](../code/part02-attacks/reconstruction.py)

## Part 3 · [Differential privacy foundations](part03-dp-foundations/README.md) · [introduction](part03-dp-foundations/00-introduction.md)

18. [The definition of differential privacy](part03-dp-foundations/01-definition.md) · [code](../code/part03-dp-foundations/definition.py)
19. [Post-processing, group privacy and composition](part03-dp-foundations/02-properties.md) · [code](../code/part03-dp-foundations/properties.py)
20. [The Laplace and Gaussian mechanisms](part03-dp-foundations/03-laplace-gaussian.md) · [code](../code/part03-dp-foundations/laplace_gaussian.py)
21. [Advanced and optimal composition](part03-dp-foundations/04-composition.md) · [code](../code/part03-dp-foundations/composition.py)
22. [Selection: exponential mechanism, report-noisy-max, sparse vector](part03-dp-foundations/05-selection.md) · [code](../code/part03-dp-foundations/selection.py)
23. [Beyond global sensitivity](part03-dp-foundations/06-sensitivity.md) · [code](../code/part03-dp-foundations/sensitivity.py)
24. [Discrete and floating-point-safe mechanisms](part03-dp-foundations/07-discrete-mechanisms.md) · [code](../code/part03-dp-foundations/discrete_mechanisms.py)
25. [Local, shuffle and distributed trust models](part03-dp-foundations/08-trust-models.md) · [code](../code/part03-dp-foundations/trust_models.py)
26. [Lower bounds: what privacy costs](part03-dp-foundations/09-lower-bounds.md) · [code](../code/part03-dp-foundations/lower_bounds.py)

## Part 4 · [Privacy accounting](part04-privacy-accounting/README.md) · [introduction](part04-privacy-accounting/00-introduction.md)

27. [Rényi differential privacy](part04-privacy-accounting/01-renyi-dp.md) · [code](../code/part04-privacy-accounting/renyi_dp.py)
28. [From Rényi DP to (ε, δ) and to tests](part04-privacy-accounting/02-rdp-conversions.md) · [code](../code/part04-privacy-accounting/rdp_conversions.py)
29. [Concentrated differential privacy](part04-privacy-accounting/03-concentrated-dp.md) · [code](../code/part04-privacy-accounting/concentrated_dp.py)
30. [f-DP and Gaussian differential privacy](part04-privacy-accounting/04-gaussian-dp.md) · [code](../code/part04-privacy-accounting/gaussian_dp.py)
31. [Amplification by subsampling](part04-privacy-accounting/05-subsampling.md) · [code](../code/part04-privacy-accounting/subsampling.py)
32. [Numerical accounting with privacy loss distributions](part04-privacy-accounting/06-numerical-accounting.md) · [code](../code/part04-privacy-accounting/numerical_accounting.py)
33. [Amplification by shuffling and by iteration](part04-privacy-accounting/07-shuffling-iteration.md) · [code](../code/part04-privacy-accounting/shuffling_iteration.py)
34. [Adaptive composition: filters, odometers and individual accounting](part04-privacy-accounting/08-adaptive-composition.md) · [code](../code/part04-privacy-accounting/adaptive_composition.py)
35. [Choosing an accountant](part04-privacy-accounting/09-choosing-an-accountant.md) · [code](../code/part04-privacy-accounting/choosing_an_accountant.py)

The exercise book (a separate website, [book-exercises/](../book-exercises/)) has every chapter's exercises and worked solutions, one page per chapter: [Part 1](../book-exercises/part01-prerequisites/), [Part 2](../book-exercises/part02-attacks/), [Part 3](../book-exercises/part03-dp-foundations/), [Part 4](../book-exercises/part04-privacy-accounting/).
