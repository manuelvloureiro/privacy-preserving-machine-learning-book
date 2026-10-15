# Contents

The book follows the [study plan](study_plan.md), one phase per part. Each folder is a part, and each file in it is a chapter, numbered across the whole book as in the rendered edition.

## Part 1 · [Prerequisites](part01-prerequisites/README.md) · [introduction](part01-prerequisites/00-introduction.md)

1. [Probability distributions](part01-prerequisites/01-distributions.md) · [code](part01-prerequisites/code/distributions.py)
2. [Concentration inequalities](part01-prerequisites/02-concentration.md) · [code](part01-prerequisites/code/concentration.py)
3. [Divergences](part01-prerequisites/03-divergences.md) · [code](part01-prerequisites/code/divergences.py)
4. [Hypothesis testing](part01-prerequisites/04-hypothesis-testing.md) · [code](part01-prerequisites/code/hypothesis_testing.py)
5. [Information theory](part01-prerequisites/05-information-theory.md) · [code](part01-prerequisites/code/information_theory.py)
6. [Optimization](part01-prerequisites/06-optimization.md) · [code](part01-prerequisites/code/optimization.py)
7. [Algorithmic stability and generalization](part01-prerequisites/07-stability-generalization.md) · [code](part01-prerequisites/code/stability.py)
8. [Cryptography basics](part01-prerequisites/08-crypto-basics.md) · [code](part01-prerequisites/code/crypto_basics.py)

## Part 2 · [Threat models and attacks](part02-attacks/README.md) · [introduction](part02-attacks/00-introduction.md)

9. [Threat models and privacy games](part02-attacks/01-threat-models.md) · [code](part02-attacks/code/threat_models.py)
10. [Membership inference](part02-attacks/02-membership-inference.md) · [code](part02-attacks/code/membership_inference.py)
11. [Attribute inference and model inversion](part02-attacks/03-attribute-inference.md) · [code](part02-attacks/code/attribute_inference.py)
12. [Property inference](part02-attacks/04-property-inference.md) · [code](part02-attacks/code/property_inference.py)
13. [Memorization and training-data extraction](part02-attacks/05-memorization-extraction.md) · [code](part02-attacks/code/memorization.py)
14. [Gradient inversion](part02-attacks/06-gradient-inversion.md) · [code](part02-attacks/code/gradient_inversion.py)
15. [Linkage and re-identification](part02-attacks/07-linkage-reidentification.md) · [code](part02-attacks/code/linkage.py)
16. [Syntactic anonymity and how it breaks](part02-attacks/08-syntactic-anonymity.md) · [code](part02-attacks/code/syntactic_anonymity.py)
17. [Reconstruction attacks](part02-attacks/09-reconstruction.md) · [code](part02-attacks/code/reconstruction.py)

## Part 3 · [Differential privacy foundations](part03-dp-foundations/README.md) · [introduction](part03-dp-foundations/00-introduction.md)

18. [The definition of differential privacy](part03-dp-foundations/01-definition.md) · [code](part03-dp-foundations/code/definition.py)
19. [Post-processing, group privacy and composition](part03-dp-foundations/02-properties.md) · [code](part03-dp-foundations/code/properties.py)
20. [The Laplace and Gaussian mechanisms](part03-dp-foundations/03-laplace-gaussian.md) · [code](part03-dp-foundations/code/laplace_gaussian.py)
21. [Advanced and optimal composition](part03-dp-foundations/04-composition.md) · [code](part03-dp-foundations/code/composition.py)
22. [Selection: exponential mechanism, report-noisy-max, sparse vector](part03-dp-foundations/05-selection.md) · [code](part03-dp-foundations/code/selection.py)
23. [Beyond global sensitivity](part03-dp-foundations/06-sensitivity.md) · [code](part03-dp-foundations/code/sensitivity.py)
24. [Discrete and floating-point-safe mechanisms](part03-dp-foundations/07-discrete-mechanisms.md) · [code](part03-dp-foundations/code/discrete_mechanisms.py)
25. [Local, shuffle and distributed trust models](part03-dp-foundations/08-trust-models.md) · [code](part03-dp-foundations/code/trust_models.py)
26. [Lower bounds: what privacy costs](part03-dp-foundations/09-lower-bounds.md) · [code](part03-dp-foundations/code/lower_bounds.py)

## Part 4 · [Privacy accounting](part04-privacy-accounting/README.md) · [introduction](part04-privacy-accounting/00-introduction.md)

27. [Rényi differential privacy](part04-privacy-accounting/01-renyi-dp.md) · [code](part04-privacy-accounting/code/renyi_dp.py)
28. [From Rényi DP to (ε, δ) and to tests](part04-privacy-accounting/02-rdp-conversions.md) · [code](part04-privacy-accounting/code/rdp_conversions.py)
29. [Concentrated differential privacy](part04-privacy-accounting/03-concentrated-dp.md) · [code](part04-privacy-accounting/code/concentrated_dp.py)
30. [f-DP and Gaussian differential privacy](part04-privacy-accounting/04-gaussian-dp.md) · [code](part04-privacy-accounting/code/gaussian_dp.py)
31. [Amplification by subsampling](part04-privacy-accounting/05-subsampling.md) · [code](part04-privacy-accounting/code/subsampling.py)
32. [Numerical accounting with privacy loss distributions](part04-privacy-accounting/06-numerical-accounting.md) · [code](part04-privacy-accounting/code/numerical_accounting.py)
33. [Amplification by shuffling and by iteration](part04-privacy-accounting/07-shuffling-iteration.md) · [code](part04-privacy-accounting/code/shuffling_iteration.py)
34. [Adaptive composition: filters, odometers and individual accounting](part04-privacy-accounting/08-adaptive-composition.md) · [code](part04-privacy-accounting/code/adaptive_composition.py)
35. [Choosing an accountant](part04-privacy-accounting/09-choosing-an-accountant.md) · [code](part04-privacy-accounting/code/choosing_an_accountant.py)

Appendix (website only): the exercise book, with every chapter's exercises and worked solutions, one page per chapter: [Part 1](part01-prerequisites/solutions/), [Part 2](part02-attacks/solutions/), [Part 3](part03-dp-foundations/solutions/), [Part 4](part04-privacy-accounting/solutions/).
