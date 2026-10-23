# Part 4 — Privacy accounting

How to compute the privacy of a long run of noisy steps: Rényi differential privacy and its conversions, concentrated DP, f-DP and Gaussian DP, amplification by subsampling, shuffling and iteration, numerical accounting with privacy loss distributions, adaptive composition, and how to choose and report an accountant.

Start with the [introduction](00-introduction.md), which is part of the book. It assumes the [prerequisites](../part01-prerequisites/README.md), especially divergences and hypothesis testing, and the [DP foundations](../part03-dp-foundations/README.md) part, especially composition and the Gaussian mechanism.

This file is the GitHub overview. It is not part of the rendered book.

## Chapters

| Ch. | Chapter | Covers | Script |
|---|---------|--------|--------|
| 29 | [Rényi differential privacy](01-renyi-dp.md) | Definition and RDP curve; Gaussian, Laplace and randomized-response curves; composition; post-processing and group privacy; the moments accountant; Mironov's conversion | [renyi_dp.py](../../code/part04-privacy-accounting/renyi_dp.py) |
| 30 | [From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md) | Where the classic conversion loses; the improved conversion; the hypothesis-testing region of an RDP curve; optimizing the order; unavoidable slack | [rdp_conversions.py](../../code/part04-privacy-accounting/rdp_conversions.py) |
| 31 | [Concentrated differential privacy](03-concentrated-dp.md) | zCDP; pure DP ⇒ zCDP; zCDP ⇒ $(\varepsilon, \delta)$; group privacy; approximate zCDP; truncated CDP and subsampling | [concentrated_dp.py](../../code/part04-privacy-accounting/concentrated_dp.py) |
| 32 | [f-DP and Gaussian differential privacy](04-gaussian-dp.md) | Trade-off functions as privacy definitions; $\mu$-GDP; duality with $(\varepsilon, \delta)$; composition; the central limit theorem; the CLT for DP-SGD | [gaussian_dp.py](../../code/part04-privacy-accounting/gaussian_dp.py) |
| 33 | [Amplification by subsampling](05-subsampling.md) | Poisson and fixed-size sampling; matching the neighbouring relation; RDP of the subsampled Gaussian; the moments accountant; DP-SGD by RDP | [subsampling.py](../../code/part04-privacy-accounting/subsampling.py) |
| 34 | [Numerical accounting with privacy loss distributions](06-numerical-accounting.md) | PLDs; composition by convolution; pessimistic and optimistic discretization; FFT and PRV accountants; DP-SGD sandwiched | [numerical_accounting.py](../../code/part04-privacy-accounting/numerical_accounting.py) |
| 35 | [Amplification by shuffling and by iteration](07-shuffling-iteration.md) | Shuffled randomized response exactly; the clones bound; amplification by iteration; hidden-state analysis of noisy SGD | [shuffling_iteration.py](../../code/part04-privacy-accounting/shuffling_iteration.py) |
| 36 | [Adaptive composition: filters, odometers and individual accounting](08-adaptive-composition.md) | Filters and odometers; fully adaptive composition; the Rényi filter; individual privacy accounting in DP-SGD | [adaptive_composition.py](../../code/part04-privacy-accounting/adaptive_composition.py) |
| 37 | [Choosing an accountant](09-choosing-an-accountant.md) | Every accountant on the same DP-SGD run; which is tight where; what to report | [choosing_an_accountant.py](../../code/part04-privacy-accounting/choosing_an_accountant.py) |

Each script prints the numbers its chapter quotes, except values the text attributes to an external library or paper, and regenerates the chapter's figures in `_build/figures/part04-privacy-accounting/` (the build copies them into each book):

```bash
uv run python code/part04-privacy-accounting/<script>.py
```

Every chapter's exercises, each with a worked solution, are in the exercise book, [book-exercises/part04-privacy-accounting/](../../book-exercises/part04-privacy-accounting/), one file per chapter. The exercise book is its own website, and `code/part04-privacy-accounting/solutions_<script>.py` prints the numbers they quote.

## Suggested order

1. Chapters 29 → 28. Everything else uses the RDP curve and its conversion.
2. Chapter 32. The trade-off view explains what chapters 30 and 34 are approximating.
3. Chapters 33 → 32, the practical core. Chapter 31 fits anywhere after chapter 30, except its section on subsampling, which uses one formula from chapter 33.
4. Chapters 35 and 36 in either order, then chapter 37.

## Core readings

- Mironov, "Rényi Differential Privacy" (CSF 2017) → chapters 29, 30
- Mironov, Talwar & Zhang, "Rényi Differential Privacy of the Sampled Gaussian Mechanism" (2019) → chapter 33
- Bun & Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds" (TCC 2016) → chapter 31
- Dong, Roth & Su, "Gaussian Differential Privacy" (JRSS-B 2022) → chapter 32
- Balle, Barthe, Gaboardi, Hsu & Sato, "Hypothesis Testing Interpretations and Rényi Differential Privacy" (AISTATS 2020) → chapter 30
- Gopi, Lee & Wutschitz, "Numerical Composition of Differential Privacy" (NeurIPS 2021) → chapter 34
- Steinke, "Composition of Differential Privacy & Privacy Amplification by Subsampling" (2022) → chapters 33, 34

## Further readings

- Dwork & Rothblum, "Concentrated Differential Privacy" (2016); Bun, Dwork, Rothblum & Steinke, "Composable and Versatile Privacy via Truncated CDP" (STOC 2018) → chapter 31
- Wang, Balle & Kasiviswanathan (AISTATS 2019); Zhu & Wang (ICML 2019); Balle, Barthe & Gaboardi (NeurIPS 2018) → chapter 33
- Asoodeh et al., "Three Variants of Differential Privacy: Lossless Conversion and Applications" (2021) → chapter 30
- Koskela, Jälkö & Honkela (AISTATS 2020); Doroshenko et al. (PoPETs 2022); Zhu, Dong & Wang (AISTATS 2022) → chapter 34
- Erlingsson et al. (SODA 2019); Feldman, McMillan & Talwar (FOCS 2021); Feldman, Mironov, Talwar & Thakurta (FOCS 2018); Altschuler & Talwar (NeurIPS 2022) → chapter 35
- Rogers et al. (NeurIPS 2016); Feldman & Zrnic (NeurIPS 2021); Whitehouse, Ramdas, Rogers & Wu (ICML 2023) → chapter 36
- Lebeda, Regehr, Kamath & Steinke, "Avoiding Pitfalls for Privacy Accounting of Subsampled Mechanisms under Composition" (IEEE SaTML 2025); Chua et al., "How Private are DP-SGD Implementations?" (ICML 2024) → chapters 33, 37
- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023); NIST SP 800-226, *Guidelines for Evaluating Differential Privacy Guarantees* (2025) → chapter 37

Each chapter ends with its own references, with what each is good for.

## Exercise

Write an RDP accountant for the Poisson-subsampled Gaussian and check it against Google's `dp_accounting` within 1%. Then, for DP-SGD with $q = 0.01$, $\sigma = 1$ and up to 10,000 steps, plot $\varepsilon$ at $\delta = 10^{-5}$ from four accountants: RDP with Mironov's conversion, RDP with the improved conversion, the GDP central-limit approximation, and PLD, alongside advanced composition applied per step.

The accountant is built in [Amplification by subsampling](05-subsampling.md). The four-way comparison is worked in [Choosing an accountant](09-choosing-an-accountant.md). The check against `dp_accounting` is left to you, since the library is not a dependency of this repository.

## Exit check

You are done with this part when you can do each of these on paper, without notes:

- [ ] Derive the RDP curve of the Gaussian mechanism, and show that RDP curves add under adaptive composition.
- [ ] Show that the moments accountant's log-moment at $\lambda$ equals $\lambda\,\varepsilon(\lambda + 1)$.
- [ ] Derive Mironov's conversion from RDP to $(\varepsilon, \delta)$, and say where it loses.
- [ ] Prove that $\varepsilon$-DP implies $(\varepsilon^2/2)$-zCDP, and convert $\rho$-zCDP to $(\varepsilon, \delta)$.
- [ ] Explain why zCDP does not amplify under subsampling.
- [ ] Write $(\varepsilon, \delta)$-DP as a trade-off function, show that the Gaussian mechanism is $(1/\sigma)$-GDP, and compose two GDP guarantees.
- [ ] Derive amplification by Poisson subsampling for $(\varepsilon, \delta)$-DP, and say which neighbouring relation it needs.
- [ ] Write the RDP of the Poisson-subsampled Gaussian at integer order as a binomial sum.
- [ ] Explain why rounding privacy losses up gives an upper bound on $\delta$ for every composition.
- [ ] Say what amplification by shuffling and by iteration each assume, and when neither applies to DP-SGD.
- [ ] Explain why adaptively chosen privacy parameters need a filter, and why the Rényi filter is free at a fixed order.
- [ ] Given a DP-SGD configuration, name the accountant you would use and everything you would report alongside $\varepsilon$.
