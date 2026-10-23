# Contents

Each folder is a part, and each file in it is a chapter, numbered across the whole book as in the rendered edition.

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
23. [Query release and private multiplicative weights](part03-dp-foundations/06-query-release.md) · [code](../code/part03-dp-foundations/query_release.py)
24. [Beyond global sensitivity](part03-dp-foundations/07-sensitivity.md) · [code](../code/part03-dp-foundations/sensitivity.py)
25. [Discrete and floating-point-safe mechanisms](part03-dp-foundations/08-discrete-mechanisms.md) · [code](../code/part03-dp-foundations/discrete_mechanisms.py)
26. [Local, shuffle and distributed trust models](part03-dp-foundations/09-trust-models.md) · [code](../code/part03-dp-foundations/trust_models.py)
27. [Lower bounds: what privacy costs](part03-dp-foundations/10-lower-bounds.md) · [code](../code/part03-dp-foundations/lower_bounds.py)
28. [Private mean estimation](part03-dp-foundations/11-mean-estimation.md) · [code](../code/part03-dp-foundations/mean_estimation.py)

## Part 4 · [Privacy accounting](part04-privacy-accounting/README.md) · [introduction](part04-privacy-accounting/00-introduction.md)

29. [Rényi differential privacy](part04-privacy-accounting/01-renyi-dp.md) · [code](../code/part04-privacy-accounting/renyi_dp.py)
30. [From Rényi DP to (ε, δ) and to tests](part04-privacy-accounting/02-rdp-conversions.md) · [code](../code/part04-privacy-accounting/rdp_conversions.py)
31. [Concentrated differential privacy](part04-privacy-accounting/03-concentrated-dp.md) · [code](../code/part04-privacy-accounting/concentrated_dp.py)
32. [f-DP and Gaussian differential privacy](part04-privacy-accounting/04-gaussian-dp.md) · [code](../code/part04-privacy-accounting/gaussian_dp.py)
33. [Amplification by subsampling](part04-privacy-accounting/05-subsampling.md) · [code](../code/part04-privacy-accounting/subsampling.py)
34. [Numerical accounting with privacy loss distributions](part04-privacy-accounting/06-numerical-accounting.md) · [code](../code/part04-privacy-accounting/numerical_accounting.py)
35. [Amplification by shuffling and by iteration](part04-privacy-accounting/07-shuffling-iteration.md) · [code](../code/part04-privacy-accounting/shuffling_iteration.py)
36. [Adaptive composition: filters, odometers and individual accounting](part04-privacy-accounting/08-adaptive-composition.md) · [code](../code/part04-privacy-accounting/adaptive_composition.py)
37. [Choosing an accountant](part04-privacy-accounting/09-choosing-an-accountant.md) · [code](../code/part04-privacy-accounting/choosing_an_accountant.py)

## Part 5 · [Differential privacy in machine learning](part05-dp-ml/README.md) · [introduction](part05-dp-ml/00-introduction.md)

38. [Private empirical risk minimization](part05-dp-ml/01-dp-erm.md) · [code](../code/part05-dp-ml/dp_erm.py)
39. [DP-SGD](part05-dp-ml/02-dp-sgd.md) · [code](../code/part05-dp-ml/dp_sgd.py)
40. [Hyperparameter tuning and private selection](part05-dp-ml/03-hyperparameter-tuning.md) · [code](../code/part05-dp-ml/hyperparameter_tuning.py)
41. [DP-FTRL and matrix factorization](part05-dp-ml/04-matrix-mechanisms.md) · [code](../code/part05-dp-ml/matrix_mechanisms.py)
42. [Private prediction: PATE and in-context learning](part05-dp-ml/05-private-prediction.md) · [code](../code/part05-dp-ml/private_prediction.py)
43. [Public data and private fine-tuning](part05-dp-ml/06-public-data.md) · [code](../code/part05-dp-ml/public_data.py)
44. [User-level differential privacy](part05-dp-ml/07-user-level.md) · [code](../code/part05-dp-ml/user_level.py)
45. [Disparate impact](part05-dp-ml/08-disparate-impact.md) · [code](../code/part05-dp-ml/disparate_impact.py)
46. [Choosing ε](part05-dp-ml/09-choosing-epsilon.md) · [code](../code/part05-dp-ml/choosing_epsilon.py)
47. [Auditing private training](part05-dp-ml/10-auditing.md) · [code](../code/part05-dp-ml/auditing.py)
48. [Adaptive data analysis](part05-dp-ml/11-adaptive-data-analysis.md) · [code](../code/part05-dp-ml/adaptive_data_analysis.py)

## Appendices · derivations

The appendices derive the book's mathematics in plain words, with worked numbers and figures, and the chapters point to them where they use a result. Appendix letters follow the rendered book.

The first four derive the core formulas from scratch: how much Laplace or Gaussian noise to add, what repeated releases cost, how DP-SGD's $\varepsilon$ is computed, and what $\varepsilon$ means.

- A. [How much noise to add](appendix-derivations/01-calibration.md) · [code](../code/appendix-derivations/apx_core_calibration.py)
- B. [Composing private releases](appendix-derivations/02-composition.md) · [code](../code/appendix-derivations/apx_core_composition.py)
- C. [Accounting for DP-SGD](appendix-derivations/03-dp-sgd.md) · [code](../code/appendix-derivations/apx_core_dp_sgd.py)
- D. [What ε means, and converting between definitions](appendix-derivations/04-conversions.md) · [code](../code/appendix-derivations/apx_core_conversions.py)

The other ten derive the results that the chapters cite to a paper, sketch, or take as given, grouped by mathematical theme.

- E. [Sums, limits and counting](appendix-derivations/05-probability.md) · [code](../code/appendix-derivations/apx_probability.py)
- F. [Tail bounds, tests and divergences](appendix-derivations/06-tails-tests-divergences.md) · [code](../code/appendix-derivations/apx_tails.py)
- G. [Optimization and estimation](appendix-derivations/07-optimization-estimation.md) · [code](../code/appendix-derivations/apx_optimization.py)
- H. [Random matrices and factorizations](appendix-derivations/08-matrices.md) · [code](../code/appendix-derivations/apx_matrices.py)
- I. [Pseudorandom functions and learning with errors](appendix-derivations/09-cryptography.md) · [code](../code/appendix-derivations/apx_cryptography.py)
- J. [Mechanisms and their limits](appendix-derivations/10-mechanisms.md) · [code](../code/appendix-derivations/apx_mechanisms.py)
- K. [Composition of trade-off functions](appendix-derivations/11-composition.md) · [code](../code/appendix-derivations/apx_composition.py)
- L. [Subsampling, shuffling and iteration](appendix-derivations/12-amplification.md) · [code](../code/appendix-derivations/apx_amplification.py)
- M. [Private learning and tuning](appendix-derivations/13-private-learning.md) · [code](../code/appendix-derivations/apx_learning.py)
- N. [Auditing, reconstruction and adaptive analysis](appendix-derivations/14-auditing-adaptivity.md) · [code](../code/appendix-derivations/apx_auditing.py)

The exercise book (a separate website, [book-exercises/](../book-exercises/)) has every chapter's exercises and worked solutions, one page per chapter: [Part 1](../book-exercises/part01-prerequisites/), [Part 2](../book-exercises/part02-attacks/), [Part 3](../book-exercises/part03-dp-foundations/), [Part 4](../book-exercises/part04-privacy-accounting/), [Part 5](../book-exercises/part05-dp-ml/).
