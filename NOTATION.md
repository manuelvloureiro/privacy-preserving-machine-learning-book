# NOTATION.md

One nomenclature for all four books and the study notes: the symbol for each concept, how it is typeset, and the words and spelling used for it. A reader who moves from the course to the full edition, or from a chapter to its exercises, should never have to relearn what $D$, $q$ or $\theta$ means. For how to write content, see [AGENTS.md](AGENTS.md); for readable prose, [PROSE.md](PROSE.md), whose rule 18 (notation earns its place) applies on top of this file.

`make notation-check` enforces the part of this file that a pattern can check: `tools/notation.py` reads the forbidden spellings and the pairs of meanings that must not meet on a page from `notation.toml`. The rest, such as which letter names a concept, is checked by the writer and the reviewer against the tables below.

## Rules

1. **One concept, one symbol.** Every concept in the tables below is written with its symbol on every page of every book, whatever the source paper or lecture uses. When a source's notation differs, the page may say so once ("Mironov writes the order as $\lambda$"), and then uses ours.
2. **Reserved symbols keep their meaning everywhere.** $\varepsilon$, $\delta$, $M$, $D$, $D'$, $\simeq$, $\Delta$, $\mathrm{Lap}$, $\mathcal N$, $\Pr$, $\mathbb E$, $q$ (sampling rate; see the densities row for the one exception), $C$ (clipping norm), $\eta$ (learning rate) and $\theta$ (parameters) never take another meaning on any page.
3. **Other symbols in the tables** may be reused for something else only on a page where their concept does not appear, and the page names the other meaning in words where it first uses it. The overloads below are the only exceptions, because both meanings are fixed in the literature. The first five must not meet on a page, and `make notation-check` fails on a page that uses both meanings of one of them; the rest may, and the page says which meaning it means.
4. **Local letters are free.** Indices ($i$, $j$, $t$), constants ($c$, $c'$), dummy variables and the quantities of one derivation are chosen by the page and named in words where they first appear (PROSE.md rule 4).
5. **The neighbouring relation is stated, not implied.** The symbol is always $D \simeq D'$. Every page that uses it says which relation it means, in words: add/remove one record (the default of the full edition, the practitioner edition and the exercise book) or replace one record (the default of the course). The relation is a definition, so changing the symbol never changes it.
6. **Each book's notation tables follow this file.** The notation table of a part introduction, a course chapter or a note lists the symbols of this file that it uses, with the same meanings.

### Overloads that must not meet on a page

| Symbol | Primary meaning | Other meaning | If a page needs both |
| --- | --- | --- | --- |
| $\alpha$ | Rényi order | Error bound of an $(\alpha, \beta)$-accuracy guarantee | write the error bound $a$ |
| $\beta$ | Failure probability | Smoothness constant ($\beta$-smooth) | write the smoothness constant $\beta_s$ |
| $L$ | Privacy loss $L_{D,D'}(y)$ | Lipschitz constant ($L$-Lipschitz) | write the Lipschitz constant $G$ |
| $b$ | Laplace scale | Hidden bit of a two-world game; type II error of a test | write the Laplace scale out as $\Delta f/\varepsilon$, or name it $s$ |
| $\sigma$ | Noise standard deviation | Noise multiplier (standard deviation per unit of sensitivity) | allowed: they agree when the sensitivity is 1; otherwise the page says which, and writes the standard deviation $\sigma\Delta_2 f$ or $\sigma C$ |
| $q$ | Sampling rate | A counting query $q \in \mathcal Q$ in query release | write the query $f$ |
| $T$ | Number of iterations | Trade-off function $T(P, Q)$, always with its arguments | allowed: the arguments tell them apart |
| $L$ | Privacy loss $L_{D,D'}(y)$ | Decoder of a matrix factorization $A = LR$ | allowed: the privacy loss keeps its subscripts or argument there |
| $\lambda$ | Strong convexity constant | Weight of a ridge (squared-norm) regularizer | allowed when the regularizer is what makes the objective $\lambda$-strongly convex |

## Mathematics and probability

| Concept | Symbol | Not |
| --- | --- | --- |
| Probability of an event, under a distribution | $\Pr[A]$, $\Pr_{Y \sim P}[A]$, $\Pr_P[A]$ | $P[A]$, $\mathbb P$, $\Pr(A)$ |
| Mass a distribution gives to a set $S$ of outcomes (a measure, as in hockey-stick gaps) | $P(S)$, $Q(S)$ | |
| Expectation | $\mathbb E[X]$, $\mathbb E_{Y \sim P}[g(Y)]$, $\mathbb E_P[g]$ | $E[X]$, $\mathrm E$ |
| Variance, covariance | $\operatorname{Var}(X)$, $\operatorname{Cov}(X, Y)$ | $\mathrm{Var}$, $\operatorname{Var}[X]$, $\mathbb V$ |
| Indicator | $\mathbf 1[A]$ | $\mathbf 1_A$, $\mathbf 1\{A\}$, $\mathbb 1$, $\mathbb I$ |
| Natural logarithm | $\ln$ | $\log$ for the natural logarithm |
| Base-2 logarithm (bits) | $\log_2$ | bare $\log$ |
| Bare $\log$ | only inside $O(\cdot)$, $\Theta(\cdot)$, $\Omega(\cdot)$, where the base does not matter | |
| Normal distribution; its CDF | $\mathcal N(\mu, \sigma^2)$; $\Phi$ | $N(\mu, \sigma^2)$, $\mathcal N(\mu, \sigma)$ |
| Laplace distribution with scale $b$; centred at $\mu$ | $\mathrm{Lap}(b)$; $\mathrm{Lap}(\mu, b)$ | $\mathrm{Laplace}(b)$, $\operatorname{Lap}$, $\text{Lap}$ |
| Binomial, Bernoulli, Poisson, exponential, geometric | $\mathrm{Bin}(n, p)$, $\mathrm{Bern}(p)$, $\mathrm{Poisson}(\lambda)$, $\mathrm{Exp}(\lambda)$, $\mathrm{Geom}(p)$ | $\mathrm{Binomial}$, $\mathrm{Bernoulli}$ |
| Two distributions; their densities or mass functions | $P, Q$; $p, q$, or $p_0, p_1$ (or $P(y), Q(y)$) on a page that also has a sampling rate | |
| Norms | $\|x\|_1$, $\|x\|_2$; $\|x\|$ is $\|x\|_2$; inside a Markdown pipe table, where `\|` would end the cell, `\Vert x\Vert` | $\lVert x \rVert$ (renders the same; keep one spelling) |
| Moment generating function; cumulant generating function | $M_X(\lambda) = \mathbb E[e^{\lambda X}]$, always with its random variable as subscript (the one use of $M$ that is not a mechanism); $K_X(\lambda) = \ln M_X(\lambda)$ | bare $M(\lambda)$ |
| Reals, integers | $\mathbb R$, $\mathbb Z$ | |
| Transpose; differential | $x^\top$; $\mathrm d x$ | $x^T$, $x'$ |
| A function's type | $f\colon \mathcal X^n \to \mathbb R^d$ | $f: \mathcal X^n \to \mathbb R^d$ |
| Dimension of a vector, a query's output or a parameter | $d$ | $k$ for a dimension |

## Data

| Concept | Symbol | Not |
| --- | --- | --- |
| Dataset; its records | $D = (x_1, \dots, x_n)$ | $X$, $x$, $S$ for a dataset |
| Number of records | $n$ | $N$ |
| Data universe (the set of possible records) | $\mathcal X$, so $D \in \mathcal X^n$ | $\mathcal U$, $\mathcal D$ |
| Neighbouring datasets | $D \simeq D'$ | $D \sim D'$, $X \sim X'$, "adjacent" |
| Under add/remove, the dataset with the record $x$ | $D' = D \cup \{x\}$ | |
| Records drawn at random from a distribution | $X_1, \dots, X_n \sim P$ i.i.d., with $D = (X_1, \dots, X_n)$ | |
| Data distribution (a population) | $\mathcal D$ | $\pi$ for a population |
| A labelled record in learning | $z = (x, y)$: features $x$, label $y$, so $D = (z_1, \dots, z_n)$ | |
| Group size in group privacy | $k$; $m$ on a page where $k$ counts composed mechanisms | |

## Differential privacy

| Concept | Symbol | Not |
| --- | --- | --- |
| Mechanism (a randomized algorithm whose privacy is analysed) | $M$; several are $M_1, \dots, M_k$ | $\mathcal M$ (the message space of the cryptography chapters), $A$, $\mathcal A$ |
| Output; output space; an event | $y \in \mathcal Y$; $\mathcal Y$; $S \subseteq \mathcal Y$ | $\mathcal R$, $\mathcal O$ for the output space |
| The random output | $Y = M(D)$ | |
| Privacy parameters | $\varepsilon$, $\delta$ | $\epsilon$; ε or δ inside running text |
| Pure and approximate DP | $\varepsilon$-DP; $(\varepsilon, \delta)$-DP | $(\varepsilon, 0)$-DP for pure DP, $(\varepsilon, \eta)$-DP |
| Sensitivities of a query $f$ | $\Delta_1 f$, $\Delta_2 f$; $\Delta f$ when the norm is clear or $f$ is scalar; $\Delta_1$, $\Delta_2$ alone when the page has one query | $\Delta_1(f)$, $\Delta_f$, $\mathrm{GS}_f$, $\operatorname{sens}(f)$ |
| Sensitivity of a linear map $x \mapsto Rx$ (largest column norm) | $\Delta_2 R$ | $\operatorname{sens}(R)$ |
| Local sensitivity at $D$; smooth sensitivity | $\mathrm{LS}_f(D)$; $S_{f,s}(D)$, with smoothing parameter $s$ | |
| Privacy loss of output $y$; the privacy loss random variable | $L_{D,D'}(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)}$, shortened to $L(y)$ once the pair is fixed; $L = L_{D,D'}(Y)$ with $Y \sim M(D)$ | $\mathcal L$, $\ell$ |
| Number of mechanisms composed | $k$ | $T$, unless the mechanisms are the $T$ steps of one algorithm |
| Slack of advanced composition | $\delta'$ | $\eta$, $\tilde\delta$ |
| Total budget of a composition | $\varepsilon_{\mathrm{total}}$, $\delta_{\mathrm{total}}$ | |
| Parameter of one component whose guarantee is combined into $\varepsilon$: one step of a composition, one query of a mechanism, a local randomizer in shuffling | $\varepsilon_0$ | $\varepsilon'$ for a per-step budget |

## Divergences and accounting

| Concept | Symbol | Not |
| --- | --- | --- |
| Kullback–Leibler divergence; between Bernoullis | $D_{\mathrm{KL}}(P\|Q)$; $\mathrm{kl}(a\|b)$ | $\mathrm{KL}(P\|Q)$, $D(P\|Q)$ |
| Total variation distance | $\mathrm{TV}(P, Q)$ | $d_{\mathrm{TV}}$, $\|P - Q\|_{\mathrm{TV}}$ |
| Hockey-stick divergence at level $\gamma \ge 1$ | $E_\gamma(P\|Q)$ | $H_\gamma$, $H_\varepsilon$, $D_{e^\varepsilon}$ |
| Privacy profile: the smallest $\delta$ for a given $\varepsilon$ | $\delta(\varepsilon) = E_{e^\varepsilon}(P\|Q)$ | |
| $f$-divergence | $D_f(P\|Q)$ | |
| Rényi divergence of order $\alpha > 1$ | $D_\alpha(P\|Q)$ | $D_\lambda$, $\mathrm D_\alpha$ |
| Rényi DP; the RDP curve | $(\alpha, \varepsilon)$-RDP; $\varepsilon(\alpha)$ | $(\lambda, \varepsilon)$-RDP, $\varepsilon_\alpha$ |
| Variable of a moment generating function or cumulant | $\lambda$ (for the privacy loss, $\lambda = \alpha - 1$) | |
| zCDP parameter | $\rho$-zCDP | |
| Trade-off function of $P$ and $Q$; a generic one | $T(P, Q)$; $f$ | |
| Gaussian DP; the Gaussian trade-off function | $\mu$-GDP; $G_\mu = T(\mathcal N(0, 1), \mathcal N(\mu, 1))$ | |
| A test (rejection rule); its type I and type II errors | $\phi$; $a$ and $b$ | $\psi$ for the test; $\alpha$ and $\beta$ for the errors |
| Grid width of a discretized privacy loss distribution | $h$ | |
| Sampling rate | $q$ | $\gamma$, $p$, $r$ |

## Mechanisms

| Concept | Symbol | Not |
| --- | --- | --- |
| Laplace mechanism | $M(D) = f(D) + Z$, $Z \sim \mathrm{Lap}(b)^d$, $b = \Delta_1 f/\varepsilon$ | $\lambda$ or $\beta$ for the scale |
| Gaussian mechanism | $M(D) = f(D) + Z$, $Z \sim \mathcal N(0, \sigma^2 I_d)$ | |
| Noise multiplier | $\sigma$, the noise standard deviation in units of the sensitivity | $z$ |
| Exponential mechanism: candidates, score, its sensitivity | $r \in \mathcal R$; $u(D, r)$; $\Delta u$ | $q$ or $s$ for the score, $k$ for $\lvert\mathcal R\rvert$ |
| A query; a sequence of them; a workload of counting queries | $f$; $f_1, \dots, f_k$; $\mathcal Q$ with $q \in \mathcal Q$ | |
| Sparse vector: threshold; the answers "below" and "above" | $\tau$; $\bot$ and $\top$ | $T$ or $t$ for the threshold, $\perp$ |
| A threshold of any test or rule | $\tau$ | |
| Accuracy guarantee | $(\alpha, \beta)$-accurate: error at most $\alpha$ with probability at least $1 - \beta$ | |
| Failure probability | $\beta$ | $\eta$, or $\delta$ for anything but the DP parameter |
| Randomized response: probability of answering truthfully | stated by the page, never $\gamma$ or $q$ | |
| Local randomizer | $R\colon \mathcal X \to \mathcal Y$ | $\mathcal R$ (that is the candidate set) |

## Learning

| Concept | Symbol | Not |
| --- | --- | --- |
| Model parameters; their dimension | $\theta \in \mathbb R^d$ | $w$ |
| Iterate at step $t$; the minimizer; a private release | $\theta_t$; $\theta^*$; $\tilde\theta$ | |
| Loss of $\theta$ on one record | $\ell(\theta; z)$ | $\ell(\theta, z)$, $\ell(w, z)$ |
| Empirical risk; population risk | $\mathcal L(\theta; D)$, sum or average as the page states; $\mathcal L_{\mathcal D}(\theta)$ | $J$, $F$, $R$ for the empirical risk |
| Training algorithm, private or not | $M$ when its privacy is analysed; $\mathcal T$ in the attack chapters | $A$ |
| Per-example gradient; clipped | $g_i = \nabla_\theta \ell(\theta; z_i)$; $\bar g_i$ | |
| Clipping norm | $C$ | $Z$, $R$, $c$ |
| Noise multiplier; sampling rate; expected batch size | $\sigma$; $q$; $B = qn$ | |
| Number of steps (iterations, rounds) | $T$ | $k$ for steps |
| Learning rate | $\eta$; $\eta_t$ at step $t$ | $\gamma$, $\alpha$ |
| Lipschitz, smoothness and strong convexity constants | $L$-Lipschitz, $\beta$-smooth, $\lambda$-strongly convex | $G$-Lipschitz (except as in the overloads above), $\mu$-strongly convex |
| Regularization strength | $\lambda$, on pages without a strongly convex constant | |
| Uniform stability of a learning algorithm | $\gamma$-uniformly stable (as in Feldman and Vondrák; the literature's $\beta$ is the failure probability here) | $\beta$ |
| Design matrix of a regression (rows are feature vectors) | $X$ | |

## Attacks

| Concept | Symbol | Not |
| --- | --- | --- |
| Adversary | $\mathcal A$ | $A$, $\mathcal B$ (except a reduction's second adversary) |
| Hidden bit of a two-world game; the guess | $b$, $b'$ (see the overloads) | |
| True and false positive rates; advantage | $\mathrm{TPR}$, $\mathrm{FPR}$; $\mathrm{Adv} = \mathrm{TPR} - \mathrm{FPR}$ | |
| Target record | $x^*$, or $z^*$ for a labelled one | |
| Prior over a secret | $\pi$ | |
| Information channel (a conditional distribution of outputs given secrets) | $K$ | $C$ (the clipping norm) |
| Cryptography: adversary, a reduction, number of queries, advantage | $\mathcal A$, $\mathcal B$, $Q$, $\nu$ (the page says once that cryptography texts write $q$ and $\varepsilon$) | $q$, $\varepsilon$ |
| Model released by training on $D$ | $\theta = \mathcal T(D)$ | |
| Reconstruction: the secret bit vector; the query matrix | $d \in \{0,1\}^n$; $A$, so the answers are $Ad + e$ (as in Dinur and Nissim) | $D$, which would read as a matrix |

## Words

The wording side of the nomenclature: one name per concept, and one spelling.

- **Spelling is Oxford English:** *-ize* endings (randomized, minimize, generalize, normalize) and British spellings otherwise: neighbour, behaviour, centred, modelling, analyse, labelled, favour. Not *randomised*, *neighbor*, *behavior*, *centered*, *modeling*, *analyze*.
- **dataset**, never *data set*; *database* only in an established name (database reconstruction) or for a database system.
- **record** is one entry of a dataset; **person** or **individual** is the human it describes. In learning, *example* appears only in the fixed terms *per-example gradient* and *example-level privacy*.
- **mechanism** for a randomized algorithm whose privacy is analysed; *algorithm* otherwise.
- **neighbouring datasets**, never *adjacent*. The relations are **add/remove** and **replace-one**; *bounded* and *unbounded DP* are mentioned once, as other names, where the relations are defined.
- **pure DP** ($\varepsilon$-DP) and **approximate DP** ($(\varepsilon, \delta)$-DP); **Rényi DP** (RDP); **zCDP**; **Gaussian DP** (GDP); **$f$-DP**.
- **privacy loss**, **privacy loss random variable**, **privacy loss distribution** (PLD), **privacy profile** $\delta(\varepsilon)$; **privacy budget** is the total $\varepsilon$ (and $\delta$) a release may spend.
- **hockey-stick divergence**, **trade-off function**, **trade-off** (hyphenated, never *tradeoff*).
- **noise multiplier**, **clipping norm** (not *clipping threshold* or *clipping bound*), **sampling rate** (not *sampling probability*, *sampling fraction* or *keep probability*), **learning rate** (an optimization page may say once that it is also called the step size), **batch size**, **per-example gradient**, **DP-SGD**.
- **user-level** and **example-level** privacy, hyphenated.
- **ε** and **δ** as Unicode only where Markdown makes plain text: headings, link text, figure and table captions and environment names. Everywhere else, including tables and lists, they are math: `$\varepsilon$`, `$\delta$`.
