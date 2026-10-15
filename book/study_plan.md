# Privacy-Preserving ML Study Plan

Oct 14, 2026 · @Manuel Loureiro

## How to use this plan

The plan runs about 32 weeks at 8–10 hours a week, attacks first, then differential privacy (DP) in depth, then the specialist tracks. Each phase lists the concepts you must be able to explain and derive, core readings (do these), further readings (pick by interest), and one exercise. Each phase that the book already covers also says which chapters cover it.

1. Phase 1 — Prerequisites (weeks 1–2, run alongside Phase 2)
2. Phase 2 — Threat models and attacks (weeks 1–3)
3. Phase 3 — DP foundations (weeks 4–7)
4. Phase 4 — Privacy accounting: Rényi DP and its relatives (weeks 8–10)
5. Phase 5 — DP in ML and auditing (weeks 11–14)
6. Phase 6 — Synthetic data (weeks 15–17)
7. Phase 7 — Formal methods (weeks 18–21)
8. Phase 8 — Cryptographic approaches (weeks 22–25)
9. Phase 9 — Federated learning (weeks 26–27, can run in parallel with Phase 8)
10. Phase 10 — Statistical disclosure control and de-identification in practice (week 28)
11. Phase 11 — Machine unlearning (week 29)
12. Phase 12 — Large language model (LLM) and agent privacy (weeks 30–31)
13. Phase 13 — Governance, law, standards (week 32, plus ongoing)

Rationale: attacks motivate everything, and the syntactic defences (k-anonymity, l-diversity, t-closeness) are studied with them, because the attacks that break them, composition above all, are the requirements DP is built to meet; DP is the backbone the later tracks reference, and accounting gets its own phase because every DP-ML claim is only as tight as the accountant behind it; formal methods land better after you have seen implementation bugs; crypto and federated learning (FL) are largely independent; LLM/agent privacy and governance draw on all of it and are where your profile differentiates.

Readings are standard references cited by author, title and year; search arXiv or the venue for the current version. Spine textbooks used throughout:

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014) — [free PDF](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- Near & Abuah, *Programming Differential Privacy* — [programming-dp.com](https://programming-dp.com/)
- Vadhan, *The Complexity of Differential Privacy* (2017)
- Evans, Kolesnikov & Rosulek, *A Pragmatic Introduction to Secure Multi-Party Computation* (2018)

## How the plan maps to the book

The book follows the plan one phase per part, so Phase N is Part N. Each chapter of a part covers one topic of its phase, and the website's exercise book has every chapter's exercises with worked solutions. Phases 1–4 are written. Phases 5–13 are not yet.

| Phase | Part | Chapters |
| --- | --- | --- |
| 1 — Prerequisites | [Prerequisites](part01-prerequisites/00-introduction.md) | 1–8 |
| 2 — Threat models and attacks | [Threat models and attacks](part02-attacks/00-introduction.md) | 9–17 |
| 3 — DP foundations | [Differential privacy foundations](part03-dp-foundations/00-introduction.md) | 18–26 |
| 4 — Privacy accounting | [Privacy accounting](part04-privacy-accounting/00-introduction.md) | 27–35 |
| 5–13 | Not yet written | — |

: The phases of the plan and the parts and chapters of the book that cover them. {#tbl-plan-map}

The book scales the plan's exercises down where they need hours of GPU time or a tool that is not a dependency of the repository. Each written phase says how.

## Phase 1 — Prerequisites (weeks 1–2)

Refresh only what you can't derive on demand; the divergence and hypothesis-testing material is the part most ML backgrounds lack.

**Key concepts**

- Distributions: Bernoulli, binomial, Poisson, exponential, Laplace and normal; moments and moment generating functions; comparing a density with its shifted copy
- Concentration: Markov, Chebyshev, Chernoff, Hoeffding, McDiarmid and Azuma; Gaussian and Laplace tails; sub-Gaussian variables
- Divergences: Kullback–Leibler (KL), Rényi divergence of order α, total variation, hockey-stick divergence E\_γ, f-divergences; the data-processing inequality; Pinsker's inequality
- Hypothesis testing: type I/II errors, Neyman–Pearson lemma, receiver operating characteristic (ROC) and trade-off curves; best advantage equals total variation; why the area under the curve (AUC) hides the regime of low false positive rate (FPR); the Bayesian view; estimating error rates from finite trials
- Information theory: Shannon entropy, mutual information, channel capacity, min-entropy and min-entropy leakage, Fano's inequality
- Optimization: stochastic gradient descent (SGD) convergence for convex, strongly convex and non-convex objectives, Lipschitz and smoothness, gradient noise and minibatches, sensitivity of regularized empirical risk minimization (ERM)
- Learning theory: algorithmic stability and its link to generalization; stability of regularized ERM and early-stopped SGD; overfitting as memorization of individual examples
- Crypto basics: negligible functions, computational indistinguishability and the hybrid argument, pseudorandom generators (PRGs) and pseudorandom functions (PRFs), perfect secrecy and security against chosen-plaintext attack (CPA), written IND-CPA, the simulation paradigm, learning with errors (LWE)

**Core readings**

- Vershynin, *High-Dimensional Probability* (2018), ch. 2 (concentration)
- van Erven & Harremoës, "Rényi Divergence and Kullback–Leibler Divergence" (2014), sections 1–3
- Lindell, "How to Simulate It — A Tutorial on the Simulation Proof Technique" (2017)

**Further**

- Boneh & Shoup, *A Graduate Course in Applied Cryptography*, ch. 2–3
- Bousquet & Elisseeff, "Stability and Generalization" (JMLR 2002)
- Cover & Thomas, *Elements of Information Theory*, ch. 2

**Exercise:** derive the Chernoff bound for a sum of Bernoullis; compute the KL and Rényi divergence between two Gaussians with equal variance by hand.

**In the book:** [Prerequisites](part01-prerequisites/00-introduction.md), chapters 1–8, one per bullet above. The Chernoff bound is derived in [Concentration inequalities](part01-prerequisites/02-concentration.md) and the two Gaussians are worked in [Divergences](part01-prerequisites/03-divergences.md). The part teaches the mathematics only: privacy starts in Part 2.

## Phase 2 — Threat models and attacks (weeks 1–3)

Learn to break models before protecting them: every defence in later phases is calibrated against one of these attacks.

**Key concepts**

- Adversary taxonomy: black-box vs white-box, honest-but-curious vs malicious, central vs local trust, auxiliary knowledge
- Privacy games: attacks as games, advantage, total variation and the trade-off curve; average-case vs worst-case privacy
- Membership inference: the loss attack and the generalization gap, shadow models, likelihood-ratio attacks (LiRA), evaluation by true positive rate (TPR) at low FPR rather than accuracy/AUC, why outliers are most exposed
- Attribute inference and model inversion: the imputation baseline, the maximum a posteriori (MAP) attack, confidence masking
- Property inference (dataset-level properties): why a property leaks when no record does, meta-classifiers
- Training data extraction: verbatim vs approximate memorization, extractable vs discoverable memorization, effect of duplication and model size
- Canaries: inserting random secrets into training data and measuring their exposure (rank of the true secret among all candidates); the measurement tool reused for auditing in Phase 5
- Gradient inversion in federated settings
- Linkage and re-identification: quasi-identifiers, uniqueness in high-dimensional data, whether releasing a sample protects
- Syntactic anonymity and its failures: k-anonymity (generalization, suppression, Mondrian), l-diversity, t-closeness, and the attacks that break each (homogeneity, background knowledge, skewness, similarity, composition)
- Reconstruction: Dinur–Nissim (the exponential attack, the linear-programming decoder, least squares), the Fundamental Law of Information Recovery, US Census 2010 reconstruction

**Core readings**

- Shokri et al., "Membership Inference Attacks against Machine Learning Models" (S&P 2017)
- Carlini et al., "Membership Inference Attacks From First Principles" (S&P 2022) — LiRA
- Carlini et al., "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks" (USENIX Security 2019) — canaries and exposure
- Carlini et al., "Extracting Training Data from Large Language Models" (USENIX Security 2021)
- Carlini et al., "Quantifying Memorization Across Neural Language Models" (ICLR 2023)
- Dinur & Nissim, "Revealing Information while Preserving Privacy" (PODS 2003)
- Narayanan & Shmatikov, "Robust De-anonymization of Large Sparse Datasets" (S&P 2008) — Netflix Prize
- Sweeney, "k-Anonymity: A Model for Protecting Privacy" (2002)
- Machanavajjhala et al., "l-Diversity: Privacy Beyond k-Anonymity" (TKDD 2007)
- Ganta, Kasiviswanathan & Smith, "Composition Attacks and Auxiliary Information in Data Privacy" (KDD 2008)

**Further**

- Fredrikson et al., "Model Inversion Attacks that Exploit Confidence Information" (CCS 2015)
- Zhu et al., "Deep Leakage from Gradients" (NeurIPS 2019)
- Ganju et al., "Property Inference Attacks on Fully Connected Neural Networks" (CCS 2018)
- Nasr et al., "Scalable Extraction of Training Data from (Production) Language Models" (2023)
- Garfinkel, Abowd & Martindale, "Understanding Database Reconstruction Attacks on Public Data" (CACM 2019)
- Rocher, Hendrickx & de Montjoye, "Estimating the success of re-identifications in incomplete datasets" (Nature Communications 2019)
- Salem et al., "SoK: Let the Privacy Games Begin!" (S&P 2023) — unifies attack definitions as games
- Li, Li & Venkatasubramanian, "t-Closeness: Privacy Beyond k-Anonymity and l-Diversity" (ICDE 2007)
- LeFevre, DeWitt & Ramakrishnan, "Mondrian Multidimensional k-Anonymity" (ICDE 2006)

**Exercise:** train a CIFAR-10 ResNet, run LiRA with 16 shadow models, and report TPR at 0.1% FPR. Keep this pipeline: you will reuse it to audit DP models in Phase 5. Then k-anonymize a census extract with ARX, link it back to a public auxiliary table, measure the re-identification rate, and intersect two independently anonymized releases.

**In the book:** [Threat models and attacks](part02-attacks/00-introduction.md), chapters 9–17, in the order of the bullets above, with canaries in [Memorization and training-data extraction](part02-attacks/05-memorization-extraction.md). [Membership inference](part02-attacks/02-membership-inference.md) runs LiRA at small scale (logistic regression, 64 shadow models) and lists what changes for a CIFAR-10 ResNet. [Syntactic anonymity and how it breaks](part02-attacks/08-syntactic-anonymity.md) uses its own Mondrian anonymizer on a synthetic hospital table in place of ARX on a census extract.

## Phase 3 — Differential privacy foundations (weeks 4–7)

This is the backbone: by week 7 you should be able to prove privacy of a new mechanism, calibrate its noise and compose it with others in (ε,δ) terms.

**Key concepts**

- Definitions: pure ε-DP, approximate (ε,δ)-DP; neighbouring relations (add/remove vs replace; example-, user-, and group-level); randomized response; divergence forms and the privacy loss; why δ must be much smaller than 1/n
- Interpretations: hypothesis testing and the bound on membership advantage, the Bayesian reading; what DP does not protect
- Properties: post-processing, group privacy, basic sequential and parallel composition; what composes and what doesn't
- Mechanisms: Laplace, Gaussian (classic calibration vs the exact privacy curve and analytic calibration); Laplace against Gaussian in many dimensions
- Composition: advanced (Dwork–Rothblum–Vadhan), optimal (Kairouz–Oh–Viswanath); adaptive composition; the privacy loss random variable and why (ε,δ) composes loosely
- Selection: exponential mechanism and its utility, report-noisy-max (with Gumbel noise it is the exponential mechanism), sparse vector technique and the ways it goes wrong
- Sensitivity: global, local, smooth sensitivity; propose-test-release; subsample-and-aggregate; clipping
- Discrete and floating-point-safe mechanisms: why floating-point Laplace breaks DP, the snapping mechanism, exact samplers, discrete Laplace and discrete Gaussian
- Trust models: central vs local DP and the √n price of local privacy, frequency estimation (k-ary randomized response, unary encoding), the shuffle model, distributed DP
- Lower bounds: packing arguments, fingerprinting codes and tracing attacks, Dinur–Nissim as a lower bound; sample-complexity cost of privacy

**Core readings**

- Dwork & Roth (2014), ch. 1–3 (definitions, basic mechanisms, composition) and ch. 8 (lower bounds)
- Near & Abuah, *Programming Differential Privacy*, all chapters (code alongside Dwork & Roth)
- Balle & Wang, "Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising" (ICML 2018)
- Kasiviswanathan et al., "What Can We Learn Privately?" (FOCS 2008) — local DP (amplification by subsampling is Phase 4)

**Further**

- Vadhan, *The Complexity of Differential Privacy* (2017) — lower bounds and theory depth
- Kairouz, Oh & Viswanath, "The Composition Theorem for Differential Privacy" (ICML 2015)
- Nissim, Raskhodnikova & Smith, "Smooth Sensitivity and Sampling in Private Data Analysis" (STOC 2007)
- Canonne, Kamath & Steinke, "The Discrete Gaussian for Differential Privacy" (NeurIPS 2020)
- Kifer & Machanavajjhala, "No Free Lunch in Data Privacy" (SIGMOD 2011) — what DP does and doesn't promise

**Exercise:** implement Laplace, Gaussian (classic and analytic calibration), exponential and sparse vector from scratch; then plot ε after k-fold composition of the Gaussian mechanism under basic, advanced and optimal composition.

**In the book:** [Differential privacy foundations](part03-dp-foundations/00-introduction.md), chapters 18–26, in the order of the bullets above, with the definition and its interpretations in one chapter. The mechanisms are implemented in the scripts of [The Laplace and Gaussian mechanisms](part03-dp-foundations/03-laplace-gaussian.md) and [Selection](part03-dp-foundations/05-selection.md), and the composition plot is in [Advanced and optimal composition](part03-dp-foundations/04-composition.md).

## Phase 4 — Privacy accounting: Rényi DP and its relatives (weeks 8–10)

(ε,δ)-DP is the guarantee you report, but it is a poor quantity to compose. Every accountant in practice tracks something else: a Rényi differential privacy (RDP) curve, a trade-off function, or the full privacy loss distribution. By week 10 you should be able to account for DP-SGD by hand in RDP, convert the result to (ε,δ), and explain how much the conversion loses against a numerical accountant.

**Key concepts**

- Rényi DP: ε(α) as the Rényi divergence of order α between neighbouring outputs; the RDP curve of randomized response, Laplace and Gaussian; composition by adding curves; post-processing; weak group privacy
- The moments accountant as RDP in disguise: the log moment generating function of the privacy loss equals (α − 1) times the RDP ε(α)
- Converting RDP to (ε,δ): Mironov's ε + log(1/δ)/(α − 1), the improved conversions, RDP as a constraint on tests, optimizing over α, and the slack that remains
- Concentrated differential privacy (CDP): ρ-zCDP as RDP linear in α; Gaussian is ρ = Δ²/2σ²; ε-DP implies (ε²/2)-zCDP; group privacy; approximate zCDP; truncated CDP and why plain zCDP does not amplify under subsampling
- f-DP and Gaussian differential privacy (GDP): trade-off functions, μ-GDP, duality with the (ε,δ) profile, composition and the central limit theorem (CLT), the CLT for DP-SGD; RDP as a lossy summary of the trade-off curve
- Amplification by subsampling: (ε,δ)-DP under Poisson sampling, Poisson vs fixed-size sampling and why the scheme must match the neighbouring relation (mixing them is a bug), the RDP of the Poisson-subsampled Gaussian, why there is no tight amplification theorem that works for every mechanism; DP-SGD accounted by RDP
- Numerical accounting: privacy loss distributions (PLD), composition as convolution, pessimistic vs optimistic discretization, fast Fourier transform (FFT) and privacy loss random variable (PRV) accountants, characteristic-function accounting
- Amplification by shuffling and by iteration: shuffled randomized response exactly, the clone bound; last-iterate (hidden-state) analysis of noisy SGD via shifted Rényi divergences
- Adaptive budgets: privacy filters and odometers, fully adaptive composition, Rényi filters for individual (per-example) privacy accounting
- Choosing and reporting an accountant: which one is tight for which mechanism, the noise needed for a target ε, matching the accountant to the training code, checking an accountant, and what NIST SP 800-226 asks you to disclose

**Core readings**

- Mironov, "Rényi Differential Privacy" (CSF 2017)
- Mironov, Talwar & Zhang, "Rényi Differential Privacy of the Sampled Gaussian Mechanism" (2019, arXiv)
- Bun & Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds" (TCC 2016)
- Dong, Roth & Su, "Gaussian Differential Privacy" (JRSS-B 2022)
- Balle, Barthe, Gaboardi, Hsu & Sato, "Hypothesis Testing Interpretations and Rényi Differential Privacy" (AISTATS 2020) — tighter RDP-to-(ε,δ) conversion
- Gopi, Lee & Wutschitz, "Numerical Composition of Differential Privacy" (NeurIPS 2021) — PRV accountant
- Steinke, "Composition of Differential Privacy & Privacy Amplification by Subsampling" (2022 survey, arXiv)

**Further**

- Dwork & Rothblum, "Concentrated Differential Privacy" (2016, arXiv)
- Bun, Dwork, Rothblum & Steinke, "Composable and Versatile Privacy via Truncated CDP" (STOC 2018)
- Wang, Balle & Kasiviswanathan, "Subsampled Rényi Differential Privacy and Analytical Moments Accountant" (AISTATS 2019)
- Zhu & Wang, "Poisson Subsampled Rényi Differential Privacy" (ICML 2019)
- Asoodeh, Liao, Calmon, Kosut & Sankar, "Three Variants of Differential Privacy: Lossless Conversion and Applications" (IEEE Journal on Selected Areas in Information Theory 2021)
- Balle, Barthe & Gaboardi, "Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences" (NeurIPS 2018)
- Koskela, Jälkö & Honkela, "Computing Tight Differential Privacy Guarantees Using FFT" (AISTATS 2020)
- Doroshenko et al., "Connect the Dots: Tighter Discrete Approximations of Privacy Loss Distributions" (PoPETs 2022)
- Zhu, Dong & Wang, "Optimal Accounting of Differential Privacy via Characteristic Function" (AISTATS 2022)
- Erlingsson et al., "Amplification by Shuffling" (SODA 2019); Feldman, McMillan & Talwar, "Hiding Among the Clones" (FOCS 2021)
- Feldman, Mironov, Talwar & Thakurta, "Privacy Amplification by Iteration" (FOCS 2018)
- Altschuler & Talwar, "Privacy of Noisy Stochastic Gradient Descent: More Iterations without More Privacy Loss" (NeurIPS 2022)
- Rogers et al., "Privacy Odometers and Filters: Pay-as-you-Go Composition" (NeurIPS 2016)
- Feldman & Zrnic, "Individual Privacy Accounting via a Rényi Filter" (NeurIPS 2021)
- Whitehouse, Ramdas, Rogers & Wu, "Fully-Adaptive Composition in Differential Privacy" (ICML 2023)
- Lebeda, Regehr, Kamath & Steinke, "Avoiding Pitfalls for Privacy Accounting of Subsampled Mechanisms under Composition" (IEEE SaTML 2025)
- Chua et al., "How Private are DP-SGD Implementations?" (ICML 2024)

**Exercise:** write an RDP accountant for the Poisson-subsampled Gaussian and check it against Google's `dp_accounting` within 1%. Then, for DP-SGD with q = 0.01, σ = 1 and up to 10,000 steps, plot ε at δ = 10⁻⁵ from four accountants: RDP with Mironov's conversion, RDP with the improved conversion, the GDP central-limit approximation, and PLD, alongside advanced composition applied per step. The gap between the curves is the cost of choosing a loose accountant.

**In the book:** [Privacy accounting](part04-privacy-accounting/00-introduction.md), chapters 27–35, in the order of the bullets above, with the moments accountant in the Rényi DP chapter. The RDP accountant is built in [Amplification by subsampling](part04-privacy-accounting/05-subsampling.md) and the four-way comparison is in [Choosing an accountant](part04-privacy-accounting/09-choosing-an-accountant.md). The check against `dp_accounting` is left to you, since the library is not a dependency of the repository.

## Phase 5 — DP in ML and auditing (weeks 11–14)

The practical core: train with DP, reason about which ε is defensible, and audit the claim empirically.

**Key concepts**

- DP-ERM: output, objective, and gradient perturbation
- DP-SGD: per-example clipping, noise multiplier, sampling rate; accounting under Poisson sampling; large-batch training
- Efficient per-sample gradients: vectorized/ghost clipping
- Hyperparameter tuning leaks privacy; private selection
- Alternatives: private aggregation of teacher ensembles (PATE), follow the regularized leader (FTRL) with tree aggregation (DP-FTRL), matrix-factorization mechanisms (no amplification assumption)
- Public-data leverage: public pretraining + private fine-tuning, DP-LoRA, parameter-efficient DP
- User-level vs example-level DP in practice
- Private prediction and DP in-context learning
- Disparate impact of DP on under-represented groups
- Choosing ε: deployed values and how to justify them
- Auditing: canary design (input-space canaries vs crafted gradient canaries, mislabeled examples, how many to insert), empirical ε lower bounds from membership inference, one-run auditing; finding implementation bugs

**Core readings**

- Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016)
- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023) — read in full
- De et al., "Unlocking High-Accuracy Differentially Private Image Classification through Scale" (2022)
- Li et al., "Large Language Models Can Be Strong Differentially Private Learners" (ICLR 2022)
- Papernot et al., "Scalable Private Learning with PATE" (ICLR 2018)
- Jagielski, Ullman & Oprea, "Auditing Differentially Private Machine Learning: How Private is Private SGD?" (NeurIPS 2020)
- Steinke, Nasr & Jagielski, "Privacy Auditing with One (1) Training Run" (NeurIPS 2023)

**Further**

- Chaudhuri, Monteleoni & Sarwate, "Differentially Private Empirical Risk Minimization" (JMLR 2011)
- Bassily, Smith & Thakurta, "Private Empirical Risk Minimization: Efficient Algorithms and Tight Error Bounds" (FOCS 2014)
- Kairouz et al., "Practical and Private (Deep) Learning without Sampling or Shuffling" (ICML 2021) — DP-FTRL
- Choquette-Choo et al., "Multi-Epoch Matrix Factorization Mechanisms for Private Machine Learning" (ICML 2023)
- Papernot & Steinke, "Hyperparameter Tuning with Rényi Differential Privacy" (ICLR 2022)
- Tramèr & Boneh, "Differentially Private Learning Needs Better Features (or Much More Data)" (ICLR 2021)
- Yu et al., "Differentially Private Fine-tuning of Language Models" (ICLR 2022)
- Bagdasaryan, Poursaeed & Shmatikov, "Differential Privacy Has Disparate Impact on Model Accuracy" (NeurIPS 2019)
- Nasr et al., "Adversary Instantiation: Lower Bounds for Differentially Private Machine Learning" (S&P 2021)
- Nasr et al., "Tight Auditing of Differentially Private Machine Learning" (USENIX Security 2023) — gradient canaries that make audits tight
- Duan et al., "Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models" (NeurIPS 2023); Wu et al., "Privacy-Preserving In-Context Learning for Large Language Models" (ICLR 2024)
- Desfontaines, "A list of real-world uses of differential privacy" (blog, damien.desfontaines.net) — deployed ε values
- Chua et al., "Mind the Privacy Unit! User-Level Differential Privacy for Language Model Fine-Tuning" (2024)

**Exercise:** implement DP-SGD from scratch in PyTorch (per-sample clipping via `torch.func.vmap`), match Opacus on CIFAR-10 at ε = 8, then audit both models with your Phase 2 LiRA pipeline using inserted canaries.

## Phase 6 — Synthetic data (weeks 15–17)

Synthetic data is only private if the generator is DP; the winning recipe for tabular data is select–measure–generate over marginals, not deep generative models.

**Key concepts**

- Why non-DP synthetic data leaks: generator memorization, membership inference on synthetic releases
- Marginal-based methods: multiplicative weights exponential mechanism (MWEM), PrivBayes, MST, AIM; graphical-model inference (Private-PGM)
- Select–measure–generate paradigm and query workloads
- Deep generative methods: generative adversarial networks (GANs) trained with DP (DP-GAN, PATE-GAN), DP diffusion; why they underperform on tabular data
- API-only DP synthesis with foundation models (Private Evolution)
- DP text synthesis via DP fine-tuned LLMs
- Evaluation: fidelity (marginals, correlations), utility (train-synthetic-test-real), privacy (attack-based, including canaries planted in the source data and searched for in the release); why distance-to-closest-record is not a privacy guarantee

**Core readings**

- McKenna, Miklau & Sheldon, "Winning the NIST Contest: A Scalable and General Approach to Differentially Private Synthetic Data" (JPC 2021) — MST
- McKenna et al., "AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data" (VLDB 2022)
- McKenna, Sheldon & Miklau, "Graphical-model based estimation and inference for differential privacy" (ICML 2019) — Private-PGM
- Stadler, Oprisanu & Troncoso, "Synthetic Data – Anonymisation Groundhog Day" (USENIX Security 2022)
- Tao et al., "Benchmarking Differentially Private Synthetic Data Generation Algorithms" (2021)

**Further**

- Hardt, Ligett & McSherry, "A Simple and Practical Algorithm for Differentially Private Data Release" (NeurIPS 2012) — MWEM
- Zhang et al., "PrivBayes: Private Data Release via Bayesian Networks" (TODS 2017)
- Jordon, Yoon & van der Schaar, "PATE-GAN" (ICLR 2019)
- Lin et al., "Differentially Private Synthetic Data via Foundation Model APIs 1: Images" (ICLR 2024); Xie et al., "…2: Text" (ICML 2024)
- Yue et al., "Synthetic Text Generation with Differential Privacy: A Simple and Practical Recipe" (ACL 2023)
- Dockhorn et al., "Differentially Private Diffusion Models" (TMLR 2023)
- Meeus et al., "The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text" (ICML 2025) — canaries designed to survive generation
- Jordon et al., "Synthetic Data — what, why and how?" (Royal Society report, 2022)
- Annamalai, Gadotti & Rocher, "A Linear Reconstruction Approach for Attribute Inference Attacks against Synthetic Data" (USENIX Security 2024)

**Exercise:** on the UCI Adult dataset, generate data with AIM (Private-PGM repo) and with CTGAN (non-DP), compare train-on-synthetic, test-on-real accuracy and run a membership inference attack against both.

## Phase 7 — Formal methods for privacy (weeks 18–21)

Formal methods close the gap between a paper proof and the code that ships: verify mechanisms, type-check sensitivity, hunt counterexamples, and quantify leakage beyond DP.

**Key concepts**

- Relational program logics: probabilistic relational Hoare logic, apRHL; approximate couplings / liftings as the proof technique for DP
- Randomness alignment: LightDP, ShadowDP
- Linear and sensitivity type systems: Fuzz, DFuzz, Duet
- Counterexample finding and testing: statistical testing (StatDP), optimization-based search (DP-Finder, DP-Sniper), CheckDP
- Implementation pitfalls: timing side channels; floating-point attacks on Laplace, the snapping mechanism and exact discrete samplers (introduced in Phase 3), now checked by machine
- Verified libraries: OpenDP's measurement/transformation framework with proofs; verified samplers in Lean (SampCert)
- Information flow: noninterference, declassification, taint tracking
- Quantitative information flow: channels, min-entropy leakage, g-leakage, capacity; relationship to DP
- Formal privacy policies: contextual integrity in temporal logic, purpose-based access control

**Core readings**

- Barthe et al., "Probabilistic Relational Reasoning for Differential Privacy" (POPL 2012) — CertiPriv/apRHL
- Barthe et al., "Proving Differential Privacy via Probabilistic Couplings" (LICS 2016)
- Reed & Pierce, "Distance Makes the Types Grow Stronger: A Calculus for Differential Privacy" (ICFP 2010) — Fuzz
- Mironov, "On Significance of the Least Significant Bits for Differential Privacy" (CCS 2012)
- Ding et al., "Detecting Violations of Differential Privacy" (CCS 2018) — StatDP
- Alvim, Chatzikokolakis, McIver, Morgan, Palamidessi & Smith, *The Science of Quantitative Information Flow* (Springer 2020), ch. 1–5 and the chapter relating quantitative information flow to DP
- Barth, Datta, Mitchell & Nissenbaum, "Privacy and Contextual Integrity: Framework and Applications" (S&P 2006)

**Further**

- Zhang & Kifer, "LightDP: Towards Automating Differential Privacy Proofs" (POPL 2017)
- Wang et al., "Proving Differential Privacy with Shadow Execution" (PLDI 2019); "CheckDP" (CCS 2020)
- Near et al., "Duet: An Expressive Higher-Order Language and Linear Type System for Statically Enforcing Differential Privacy" (OOPSLA 2019)
- Bichsel et al., "DP-Finder" (CCS 2018); "DP-Sniper" (S&P 2021)
- Barthe et al., *Foundations of Probabilistic Programming* (CUP 2020), the chapters on relational verification and couplings
- Sabelfeld & Myers, "Language-Based Information-Flow Security" (JSAC 2003)
- OpenDP documentation and Gaboardi et al., "A Programming Framework for OpenDP" (2020)
- Jin et al., "Are We There Yet? Timing and Floating-Point Attacks on Differential Privacy Systems" (S&P 2022)
- SampCert repository (verified DP samplers in Lean) and its accompanying paper
- Garg, Jia & Datta, "Policy Auditing over Incomplete Logs: Theory, Implementation and Applications" (CCS 2011) — enforcing HIPAA-style policies formalized in logic

**Exercise:** write an apRHL-style coupling proof for report-noisy-max by hand; then run StatDP or DP-Sniper against a deliberately buggy sparse vector implementation and against your Phase 3 one.

## Phase 8 — Cryptographic approaches (weeks 22–25)

Crypto protects inputs during computation; DP protects what the output reveals. Production systems combine them, so learn where each one's guarantee stops.

**Key concepts**

- Secure multi-party computation: additive and Shamir secret sharing, Beaver triples, garbled circuits, oblivious transfer; semi-honest vs malicious security; 2PC vs 3PC
- Private ML inference and training with secure multiparty computation (MPC): cost of non-linearities, fixed-point arithmetic
- Homomorphic encryption: Paillier; BFV, BGV, CKKS (approximate arithmetic for ML), TFHE; noise budgets, bootstrapping, polynomial approximation of activations; packing into single instruction, multiple data slots
- Hybrid homomorphic encryption (HE) + MPC private inference
- Secure aggregation for federated learning
- Zero-knowledge proofs: succinct non-interactive arguments of knowledge vs scalable transparent arguments of knowledge; zkML for proving inference or training integrity
- Private set intersection and private information retrieval for data joins
- Trusted execution environments: SGX, SEV-SNP, TDX, confidential GPUs; remote attestation; side-channel attacks
- Composition with DP: computational DP, distributed noise generation

**Core readings**

- Evans, Kolesnikov & Rosulek, *A Pragmatic Introduction to Secure Multi-Party Computation* (2018), ch. 1–6
- Mohassel & Zhang, "SecureML: A System for Scalable Privacy-Preserving Machine Learning" (S&P 2017)
- Cheon, Kim, Kim & Song, "Homomorphic Encryption for Arithmetic of Approximate Numbers" (ASIACRYPT 2017) — CKKS
- Gilad-Bachrach et al., "CryptoNets" (ICML 2016)
- Juvekar, Vaikuntanathan & Chandrakasan, "GAZELLE: A Low Latency Framework for Secure Neural Network Inference" (USENIX Security 2018)
- Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving Machine Learning" (CCS 2017)
- Costan & Devadas, "Intel SGX Explained" (2016)

**Further**

- Gentry, "Fully Homomorphic Encryption Using Ideal Lattices" (STOC 2009) — historical; skim
- Halevi & Shoup, HElib design notes; OpenFHE documentation for BFV/CKKS parameter selection
- Mohassel & Rindal, "ABY3" (CCS 2018)
- Knott et al., "CrypTen: Secure Multi-Party Computation Meets Machine Learning" (NeurIPS 2021)
- Huang et al., "Cheetah: Lean and Fast Secure Two-Party Deep Neural Network Inference" (USENIX Security 2022)
- Thaler, *Proofs, Arguments, and Zero-Knowledge* (2022, free online)
- Kang et al., "Scaling up Trustless Deep Neural Network Inference with Zero-Knowledge Proofs" (2022)
- Van Bulck et al., "Foreshadow" (USENIX Security 2018) — side channels in trusted execution environments
- NVIDIA, "Confidential Computing on H100 GPUs" whitepaper
- Mironov, Pandey, Reingold & Vadhan, "Computational Differential Privacy" (CRYPTO 2009)
- Kairouz, Liu & Steinke, "The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation" (ICML 2021)

**Exercise:** run encrypted logistic-regression inference with CKKS in TenSEAL or OpenFHE, tuning parameters for 128-bit security; then run the same model under 2PC in CrypTen and compare latency and communication.

## Phase 9 — Federated learning (weeks 26–27)

Federated learning keeps raw data local but leaks through updates; it is private only when combined with secure aggregation and DP.

**Key concepts**

- FedAvg; cross-device vs cross-silo; client heterogeneity (data that is not independent and identically distributed (IID), stragglers)
- Why FL alone is not private: gradient inversion, malicious-server attacks that defeat secure aggregation
- The production stack: FL + secure aggregation + client-level DP (DP-FedAvg, DP-FTRL)
- Robustness vs privacy: Byzantine-robust aggregation, poisoning and backdoors, why they conflict with secure aggregation
- Personalization under client-level DP

**Core readings**

- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS 2017) — FedAvg
- McMahan et al., "Learning Differentially Private Recurrent Language Models" (ICLR 2018) — DP-FedAvg
- Kairouz et al., "Advances and Open Problems in Federated Learning" (FnTML 2021), sections on privacy and robustness
- Xu et al., "Federated Learning of Gboard Language Models with Differential Privacy" (ACL Industry 2023)

**Further**

- Geiping et al., "Inverting Gradients — How easy is it to break privacy in federated learning?" (NeurIPS 2020)
- Boenisch et al., "When the Curious Abandon Honesty: Federated Learning Is Not Private" (EuroS&P 2023)
- Bagdasaryan et al., "How To Backdoor Federated Learning" (AISTATS 2020)
- Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (NeurIPS 2017) — Krum
- Flower framework documentation and tutorials

**Exercise:** in Flower, train a model across 100 simulated non-IID clients with client-level DP, then run a gradient inversion attack on a single client update with and without DP.

## Phase 10 — Statistical disclosure control and de-identification in practice (week 28)

Learn the toolkit that statistics offices, regulators and most enterprises still use, now that you can compare it with DP. k-anonymity and its successors, and the attacks on them, were covered with the attacks in Phase 2; this phase is about practice and law.

**Key concepts**

- Pseudonymization vs anonymization as a legal distinction; singling out, linkability and inference as the regulators' tests
- Statistical disclosure control: suppression, generalization, top-coding, data swapping, noise addition, cell-key method; how statistics offices choose and tune them
- Text de-identification: scrubbing personally identifiable information (PII) with named entity recognition, its recall limits, re-identification of redacted text
- Where statistical disclosure control (SDC) and DP meet: the US Census Bureau's move from swapping to DP for 2020

**Core readings**

- Ohm, "Broken Promises of Privacy: Responding to the Surprising Failure of Anonymization" (UCLA Law Review 2010)
- Article 29 Working Party, Opinion 05/2014 on Anonymisation Techniques
- Hundepool et al., *Statistical Disclosure Control* (Wiley 2012), the chapters on microdata and tabular data

**Further**

- Lison et al., "Anonymisation Models for Text Data: State of the Art, Challenges and Future Directions" (ACL 2021)
- Abowd, "The U.S. Census Bureau Adopts Differential Privacy" (KDD 2018)

**Exercise:** take a public microdata extract, apply the SDC methods a statistics office would (top-coding, swapping, cell-key perturbation of tables), and compare the disclosure risk and the accuracy of a few published tables against a DP release with the same accuracy budget.

## Phase 11 — Machine unlearning (week 29)

Unlearning is the operational answer to erasure requests; exact methods are expensive, approximate methods mostly fail rigorous audits.

**Key concepts**

- Exact unlearning: retraining, SISA (sharded, isolated, sliced and aggregated training)
- Certified approximate unlearning and its relation to DP (indistinguishability from retraining)
- LLM unlearning: gradient ascent, preference-based methods, task arithmetic; benchmarks (TOFU, WMDP, MUSE)
- Evaluation: membership inference as the unlearning test; canaries inserted then unlearned, with exposure measured before and after; relearning and extraction attacks on "unlearned" models

**Core readings**

- Bourtoule et al., "Machine Unlearning" (S&P 2021) — SISA
- Guo et al., "Certified Data Removal from Machine Learning Models" (ICML 2020)
- Maini et al., "TOFU: A Task of Fictitious Unlearning for LLMs" (2024)
- Cooper et al., "Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy, Research, and Practice" (2024)

**Further**

- Ginart et al., "Making AI Forget You: Data Deletion in Machine Learning" (NeurIPS 2019)
- Sekhari et al., "Remember What You Want to Forget: Algorithms for Machine Unlearning" (NeurIPS 2021)
- Shi et al., "MUSE: Machine Unlearning Six-Way Evaluation for Language Models" (2024)
- Hayes et al., "Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy" (2024)

**Exercise:** fine-tune a small LLM on TOFU, apply gradient-ascent unlearning, and test with both the TOFU metrics and a LiRA-style membership attack.

## Phase 12 — LLM and agent privacy (weeks 30–31)

This is the frontier and the least settled area: agents move data across tools, memories and principals, so contextual integrity becomes the right frame and DP covers only part of it.

**Key concepts**

- Memorization scaling and mitigations: deduplication, DP pretraining/fine-tuning, output filtering; canary exposure as the check that a mitigation worked
- PII in pretraining corpora; data provenance and documentation
- Inference-time privacy: prompts and context as sensitive data, logging and retention, confidential inference
- Retrieval-augmented generation leakage: extraction of retrieval corpora, DP retrieval
- Contextual integrity for LLMs and agents: appropriate information flows across tools, memory and users
- Agent memory as a privacy surface: persistence, cross-session and cross-user leakage
- Prompt injection as a data-exfiltration channel; information-flow control for agents
- Privacy-preserving telemetry and usage analytics (DP aggregation, clustering with privacy thresholds)

**Core readings**

- Brown et al., "What Does it Mean for a Language Model to Preserve Privacy?" (FAccT 2022)
- Lukas et al., "Analyzing Leakage of Personally Identifiable Information in Language Models" (S&P 2023)
- Mireshghallah et al., "Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory" (ICLR 2024)
- Greshake et al., "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection" (AISec 2023)
- Debenedetti et al., "Defeating Prompt Injections by Design" (2025) — CaMeL, information-flow control for agents
- Nissenbaum, *Privacy in Context* (2009), part III

**Further**

- Kandpal, Wallace & Raffel, "Deduplicating Training Data Mitigates Privacy Risks in Language Models" (ICML 2022)
- Zeng et al., "The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation" (ACL Findings 2024)
- Bagdasarian et al., "AirGapAgent: Protecting Privacy-Conscious Conversational Agents" (CCS 2024)
- Shao et al., "PrivacyLens: Evaluating Privacy Norm Awareness of Language Models in Action" (NeurIPS 2024)
- Wu et al., "IsolateGPT" (NDSS 2025) — execution isolation for LLM apps
- Tamkin et al., "Clio: Privacy-Preserving Insights into Real-World AI Use" (Anthropic, 2024)
- Apple Differential Privacy Team, "Learning with Privacy at Scale" (2017)

**Exercise:** build a two-tool agent with persistent memory, design 20 contextual-integrity test cases (appropriate vs inappropriate flows), measure violation rate, then add an information-flow policy layer and remeasure.

## Phase 13 — Governance, law and standards (week 32, then ongoing)

The technical guarantee only matters if it maps onto a legal standard; learn where GDPR's anonymization threshold and the AI Act's data duties meet DP, synthetic data and unlearning.

**Key concepts**

- GDPR: personal data, Recital 26 anonymization threshold (means reasonably likely to be used), pseudonymization, lawful basis, data protection impact assessments (DPIAs), Article 17 erasure, Article 22 automated decisions
- Whether trained models contain personal data: the EDPB position on AI models
- EU AI Act: Article 10 data governance, GPAI obligations, interplay with GDPR
- US: HIPAA Safe Harbor vs Expert Determination, CCPA/CPRA, sectoral regimes
- Standards: NIST SP 800-226 on evaluating DP guarantees, NIST Privacy Framework, ISO/IEC 27559 (de-identification), ISO/IEC 20889 (terminology)
- Communicating ε and privacy budgets to non-technical stakeholders; budgets as organizational policy
- Case studies: US Census 2020 Disclosure Avoidance System and the debate around it; Apple and Google DP telemetry

**Core readings**

- GDPR text: Articles 4, 5, 6, 17, 22, 25, 35 and Recital 26
- EDPB, Opinion 28/2024 on certain data protection aspects related to the processing of personal data in the context of AI models
- NIST SP 800-226, *Guidelines for Evaluating Differential Privacy Guarantees* (final, 2025)
- Nissim et al., "Bridging the Gap between Computer Science and Legal Approaches to Privacy" (Harvard JOLT 2018)
- Abowd et al., "The 2020 Census Disclosure Avoidance System TopDown Algorithm" (HDSR 2022)
- EU AI Act, Article 10 and the GPAI provisions (Articles 53–55)

**Further**

- Cohen & Nissim, "Towards formalizing the GDPR's notion of singling out" (PNAS 2020)
- boyd & Sarathy, "Differential Perspectives: Epistemic Disconnects Surrounding the U.S. Census Bureau's Use of Differential Privacy" (HDSR 2022)
- Dwork, Kohli & Mulligan, "Differential Privacy in Practice: Expose your Epsilons!" (JPC 2019)
- Veale, Binns & Edwards, "Algorithms that remember: model inversion attacks and data protection law" (Phil. Trans. R. Soc. A 2018)
- ICO guidance on anonymisation, pseudonymisation and privacy-enhancing technologies
- Irish DPC guidance on anonymisation and pseudonymisation
- HHS, Guidance Regarding Methods for De-identification of Protected Health Information under HIPAA

**Exercise:** write a two-page DPIA for the DP model from Phase 5 that states the privacy unit, ε and δ, accountant, threat model, residual risks, and how the release maps onto Recital 26.

## Capstones and tooling

Finish with one capstone that combines at least three phases; the first option sits closest to agentic AI governance work.

**Capstone options**

1. Governed agent memory: an agent memory layer with contextual-integrity policies, DP aggregate analytics over stored memories, and an audit harness (Phases 3, 7, 12, 13).
2. DP fine-tuning with end-to-end audit: user-level DP-LoRA on a domain corpus, PRV accounting, canary audit, and a published privacy card (Phases 4, 5, 13).
3. DP synthetic data service: AIM-based tabular synthesis with workload-driven evaluation and attack-based privacy tests (Phases 2, 6, 10).
4. Verified mechanism library: a small DP library in a typed language with sensitivity tracking, discrete samplers and counterexample tests (Phases 3, 7).

| Area | Tools |
| --- | --- |
| DP libraries | OpenDP, Google DP library, Tumult Analytics, `dp_accounting` |
| DP training | Opacus, TensorFlow Privacy, JAX Privacy |
| Synthetic data | Private-PGM, SmartNoise Synth, synthcity, SDV (non-DP baseline) |
| Attacks and auditing | ML Privacy Meter, LiRA reference code |
| MPC and HE | CrypTen, MP-SPDZ, OpenFHE, TenSEAL, Concrete ML |
| zkML | EZKL |
| Federated learning | Flower, TensorFlow Federated |
| Anonymization | ARX, Presidio (PII detection) |

: Tooling by area. {#tbl-tooling}

**Keeping current:** TPDP workshop and PETS proceedings, IEEE S&P / USENIX Security / CCS privacy tracks, Damien Desfontaines' blog, DifferentialPrivacy.org.
