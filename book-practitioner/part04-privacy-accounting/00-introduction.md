# Privacy accounting

A model trained with stochastic gradient descent (SGD) under differential privacy (DP), the method called DP-SGD, takes thousands of noisy steps. Yet you report one guarantee for the run, a privacy level $\varepsilon$ with a small slack $\delta$. The composition theorems of Part 3 are far too loose for that. An **accountant** does better, because it tracks a summary of each step's cost that composes tightly, and converts the total to $(\varepsilon, \delta)$ once, at the end.

The running example is a DP-SGD run of 10,000 steps. Each step samples every record with probability 0.01 and adds noise with multiplier 1. At $\delta = 10^{-5}$, advanced composition certifies $\varepsilon = 24{,}379.55$, while the privacy loss distribution accountant of Google's `dp_accounting` certifies 6.19. So an $\varepsilon$ means nothing until you name the accountant behind it, and the sampler and neighbouring relation it assumed.

The part starts with two jobs. [Report and review an ε claim](09-choosing-an-accountant.md) says what a privacy statement must contain and how to check one. [Calibrate and account a DP-SGD run](10-calibrate-dp-sgd.md) turns a target $\varepsilon$ into a noise multiplier for the Adult run of Part 5.

Two full chapters explain what those jobs rely on. [Amplification by subsampling](05-subsampling.md) shows how much a random batch helps, provided the data loader samples the way the accountant assumes. [Rényi differential privacy](01-renyi-dp.md) defines the curve most libraries track.

Short reference pages follow. Three finish the DP-SGD story: the [conversion from Rényi DP](02-rdp-conversions.md), [numerical accounting](06-numerical-accounting.md), and [Gaussian DP](04-gaussian-dp.md), which reads a guarantee as a limit on attacks. Then come [adaptive composition](08-adaptive-composition.md) for budgets decided during training, [concentrated DP](03-concentrated-dp.md) for analytics ledgers, and [shuffling and iteration](07-shuffling-iteration.md).

Every number comes from code you can run with Opacus or `dp_accounting`. By the end, you will be able to calibrate a DP-SGD run, account for it with two libraries, and report $\varepsilon$ so that others can reproduce it.
