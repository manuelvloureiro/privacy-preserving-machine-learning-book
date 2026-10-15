# Numerical accounting with privacy loss distributions

## Why this matters

You trained a model with differential privacy (DP): 10,000 steps of stochastic gradient descent (SGD), each on a Poisson sample of the data with rate $q = 0.01$, with clipped gradients and Gaussian noise of multiplier $\sigma = 1$. This is DP-SGD, the running example of this part. You need the smallest $\varepsilon$ for which the whole run is $(\varepsilon, 10^{-5})$-DP. Rényi differential privacy (RDP), from [Rényi DP](01-renyi-dp.md), gives an upper bound that loses an unknown amount in its conversion. The Gaussian central limit theorem (CLT) of [f-DP and Gaussian DP](04-gaussian-dp.md) gives an approximation, not a bound. Neither says how far it is from the truth.

A **privacy loss distribution (PLD) accountant** does. The one idea: $\delta$ depends only on the distribution of the privacy loss, and composing steps adds their losses, so composition is a convolution of distributions. Put the loss on a grid, convolve numerically, and read off $\delta(\varepsilon)$. Rounding every loss up gives a certified upper bound, rounding down a lower bound, and you know in advance how far apart they are. @tbl-num-overview maps the decisions to the sections.

| Decision or result | Rule | Section |
| --- | --- | --- |
| What summary of a mechanism determines $\delta(\varepsilon)$ | The distribution of its privacy loss | @sec-num-pld |
| How to compose $k$ steps, adaptive or not | Convolve the per-step PLDs of a dominating pair | @sec-num-convolution |
| How to get a bound rather than an estimate | Round losses up (pessimistic) and down (optimistic) | @sec-num-discretize |
| How to choose the grid and avoid numerical traps | Width $kh$; pad every convolution | @sec-num-fft |
| Which pair to account for in DP-SGD | The subsampled Gaussian, in both directions | @sec-num-subsampled |
| What to report for the running example | The pessimistic $\varepsilon$, with the optimistic one as error bar | @sec-num-dpsgd |

: The decisions behind a numerical privacy accountant, and where this chapter makes each one. {#tbl-num-overview}

## The privacy loss distribution {#sec-num-pld}

**Idea.** Let $P$ be the distribution of the output on your dataset $D$ and $Q$ on a neighbour $D'$. The **privacy loss** of an output $y$ is $L(y) = \ln\frac{P(y)}{Q(y)}$, how much more likely $y$ is under $D$, on a log scale ([Composition](../part03-dp-foundations/04-composition.md)). With $Y$ drawn from $P$, $L(Y)$ is the privacy loss random variable (PRV), and its distribution is the PLD. Each output adds to $\delta(\varepsilon)$ an amount that depends only on its loss: nothing if the loss is at most $\varepsilon$, a growing fraction of its probability above. So $\delta(\varepsilon)$ is an average over the PLD, and the huge output space of a training run never enters.

**Result.** For every $\varepsilon$,
$$\delta(\varepsilon) = \mathbb E\big[(1 - e^{\varepsilon - L})_+\big] = P[L > \varepsilon] - e^{\varepsilon}\,Q[L > \varepsilon],$$ {#eq-num-delta}
where $(x)_+ = \max\{x, 0\}$: the chance under $P$ that the loss exceeds $\varepsilon$, minus $e^\varepsilon$ times the same chance under $Q$. An output impossible under $D'$ has infinite loss and counts fully. For the Gaussian mechanism with $\mu = \Delta_2f/\sigma$ (the sensitivity in noise standard deviations), the loss is Gaussian, $\mathcal N(\mu^2/2, \mu^2)$, which gives the closed form of [The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md):
$$\delta(\varepsilon) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^{\varepsilon}\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big),$$ {#eq-num-gauss}
with $\Phi$ the standard normal cumulative distribution function (CDF).

**Example.** A Gaussian mechanism with sensitivity 1 and $\sigma = 1$ has $\mu = 1$ and PLD $\mathcal N(0.5, 1)$. At $\varepsilon = 1$, @eq-num-gauss gives $\Phi(-0.5) - e\,\Phi(-1.5) = 0.3085 - 0.1816 = 0.1269$, the green area of @fig-num-pld-weight. Solving $\delta(\varepsilon) = 10^{-5}$ for $\varepsilon$ gives 4.3772.

![The Gaussian PLD for $\mu = 1$ (blue), the weight $(1 - e^{\varepsilon - \ell})_+$ at $\varepsilon = 1$ (orange, scaled by 0.4 to fit), and their product (green). The green area is $\delta(1) = 0.1269$: only losses above $\varepsilon$ count, and those just above it count little.](figures/numerical_accounting_pld_weight.svg){#fig-num-pld-weight}

**Pitfalls.** The PLD belongs to an ordered pair $(P, Q)$, and DP needs both orders, so an accountant must cover both (@sec-num-subsampled). Don't bound $\delta$ by the tail $P[L > \varepsilon]$ alone: it ignores the down-weighting near $\varepsilon$ and is loose.

## Composition is convolution {#sec-num-convolution}

**Idea.** For independent steps the probability of the sequence of outputs is a product, so the total privacy loss is the sum of the step losses, and the distribution of a sum of independent variables is a convolution.

**Result.** The PLD of $k$ steps is the $k$-fold convolution of the per-step PLDs. This is exact; every error in a PLD accountant comes from computing it. For Gaussian steps, means and variances add, so $k$ steps with shift $\mu$ behave like one Gaussian mechanism with $\mu_k = \sqrt k\,\mu$. DP-SGD steps are adaptive, since each depends on the model. The fix is a **dominating pair**, a fixed pair at least as easy to tell apart as the step whatever happened before; convolving the dominating pairs' PLDs bounds the adaptive run (Zhu, Dong & Wang 2022).

**Example.** Ten Gaussian steps with $\sigma = 1$ have PLD $\mathcal N(5, 10)$, a single Gaussian with $\mu = \sqrt{10}$, and are $(17.8566, 10^{-5})$-DP. A hundred steps have $\mu = 10$ and $\varepsilon = 91.8173$. Any accountant run with sampling rate $q = 1$ must reproduce them.

**Pitfalls.** The bound is only as tight as the dominating pair. For the Gaussian mechanism, $\mathcal N(0, \sigma^2)$ against $\mathcal N(1, \sigma^2)$ is attained by real datasets, so nothing is lost. For a subsampled mechanism the pair depends on how batches are sampled, and must match what your training code does.

## Pessimistic and optimistic discretization {#sec-num-discretize}

**Idea.** To convolve numerically, the loss goes on a grid of width $h$. The weight $(1 - e^{\varepsilon - \ell})_+$ never decreases as the loss $\ell$ grows, so moving every loss up can only raise $\delta$, and moving every loss down can only lower it, after any number of compositions.

**Result.** The **pessimistic** PLD rounds every loss up to the next grid point and moves any truncated upper tail to $+\infty$. The **optimistic** PLD rounds every loss down and drops the truncated lower tail. Composing each $k$ times gives
$$\delta_{\text{optimistic}}(\varepsilon) \;\le\; \delta(\varepsilon) \;\le\; \delta_{\text{pessimistic}}(\varepsilon)$$ {#eq-num-sandwich}
at every $\varepsilon$. The pessimistic PLD belongs to a real pair that dominates the original, so the upper bound holds for adaptive composition too. The optimistic side bounds the mechanism from below when real datasets attain the pair, as for the Gaussian and subsampled Gaussian. In $\varepsilon$ the two sides are at most $kh$ apart, since each step's two roundings differ by at most $h$; for a continuous PLD the width is almost exactly $kh$.

**Example.** Randomized response with $\varepsilon_0 = \ln 3$, composed three times and rounded on the grid $h = 0.5$, has exact $\delta(1) = 0.4190$, pessimistic 0.6758 and optimistic 0.3648 (@fig-num-rounding). For Gaussian steps with $\mu = 1$ on the grid $h = 10^{-3}$, @tbl-num-gauss-grid shows the sandwich around the exact $\varepsilon$, with width $kh$: 0.01 at $k = 10$ and 0.1 at $k = 100$.

| $k$ | Exact $\varepsilon$ | Pessimistic | Optimistic |
| --- | --- | --- | --- |
| 10 | 17.8566 | 17.8616 | 17.8516 |
| 100 | 91.8173 | 91.8673 | 91.7673 |

: Composed Gaussian steps with $\mu = 1$ on the grid $h = 10^{-3}$: $\varepsilon$ at $\delta = 10^{-5}$ from the pessimistic and optimistic PLDs against the exact value. {#tbl-num-gauss-grid}

![Left: the PLD of three-fold randomized response with $\varepsilon_0 = \ln 3$ (black), rounded up (red) and rounded down (blue) on the grid $h = 0.5$. Right: the three $\delta(\varepsilon)$ curves; the pessimistic one lies above the exact one and the optimistic one below it at every $\varepsilon$.](figures/numerical_accounting_rounding.svg){#fig-num-rounding}

**Pitfalls.** Only the two sides are guarantees; their midpoint, or rounding to the nearest grid point, is an estimate. The width grows linearly in $k$, so a grid fine for 100 steps is a hundred times too coarse for 10,000.

## Composition by FFT {#sec-num-fft}

**Idea.** On a grid, composing two PLDs is a discrete convolution of their mass vectors, which a fast Fourier transform (FFT) computes in $O(N\log N)$ operations for length $N$. Repeated squaring needs at most $2\log_2k$ convolutions for $k$ steps.

**Result.** Three settings decide whether the numbers are right. The **grid width** $h$ sets the sandwich width $kh$; pick it for the largest $k$ you will run. The **window** must not wrap: an FFT convolution is circular, so mass falling off one end silently reappears at the other unless each convolution is padded to the sum of its inputs' lengths. **Truncating** the tails at a tiny mass such as $10^{-15}$ keeps vectors short, if the mass moves in the safe direction (to $+\infty$ for the pessimistic side). Floating-point error is far below any $\delta$ you would report, though outside the certified bound.

Production variants refine the grid. The **PRV accountant** (Gopi, Lee & Wutschitz 2021) rounds to the nearest point and shifts the grid to preserve the mean loss, so errors partly cancel and $h \approx 1/\sqrt k$ suffices instead of $1/k$, with explicit error bounds on $\varepsilon$. **Connect the dots** (Doroshenko et al. 2022) builds the best pessimistic PLD on a given grid. Google's `dp_accounting` library implements PLD accounting with pessimistic and optimistic rounding and connect-the-dots.

**Example.** Ten Gaussian steps with $\mu = 1$ on the grid $h = 10^{-3}$ give a pessimistic $\delta(1) = 0.81891018$, against the exact 0.81851782. On an unpadded circular window from loss 0 to 10 the same computation gives 0.83478562, wrong in the second decimal; width 40 gives the padded value.

**Pitfalls.** A wraparound error has unknown sign, so a wrapped result is no bound at all. Each tenfold refinement of $h$ makes the vectors ten times longer; at large $k$ that cost is the case for the PRV accountant or connect-the-dots.

## The subsampled Gaussian {#sec-num-subsampled}

**Idea.** A DP-SGD step with Poisson sampling includes each record with probability $q$. For the record in question, the noisy sum (scaled by the clipping norm) is a Gaussian centred at 0 without it, and with it a mixture centred at 1 with probability $q$ and at 0 otherwise ([Amplification by subsampling](05-subsampling.md)).

**Result.** Under add/remove neighbours the two orders of the pair are different pairs. With $D'$ the dataset that holds the record, the **remove** direction puts it first, $P = M(D')$ and $Q = M(D)$:
$$P = (1 - q)\,\mathcal N(0, \sigma^2) + q\,\mathcal N(1, \sigma^2), \qquad Q = \mathcal N(0, \sigma^2),$$
and the **add** direction swaps $P$ and $Q$ (Zhu, Dong & Wang 2022). The remove loss, $L_{\mathrm{rem}}(y) = \ln\big((1 - q) + q\,e^{(2y - 1)/(2\sigma^2)}\big)$, is bounded below by $\ln(1 - q)$ with a long upper tail; the add loss is its negative. Discretize each direction, compose each $k$ times, and report the larger $\varepsilon$.

**Example.** For $q = 0.01$ and $\sigma = 1$, one step is $(0.1995, 10^{-5})$-DP in the remove direction and $(0.0091, 10^{-5})$-DP in the add direction.

**Pitfalls.** Don't assume one direction dominates. Here the remove direction is larger at every $k$, but that is an observation, not a theorem; an accountant using only the remove pair covers half of the add/remove relation. The pair assumes Poisson sampling: if your loader shuffles and partitions instead, these numbers do not apply ([Amplification by subsampling](05-subsampling.md)).

## The running example: DP-SGD {#sec-num-dpsgd}

**Idea.** Put the pieces together for $q = 0.01$, $\sigma = 1$, $\delta = 10^{-5}$, and compare with the other accountants.

**Result.** @tbl-num-dpsgd and @fig-num-eps-k give $\varepsilon$ from the PLD bounds on the grid $h = 10^{-5}$ (the larger direction), the RDP accountant with the improved conversion, and the Gaussian differential privacy (GDP) central-limit approximation.

| $k$ | PLD upper | PLD lower | Width | RDP | GDP CLT |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1995 | 0.1994 | 1.0e-05 | 0.9555 | 0.0368 |
| 10 | 0.3800 | 0.3799 | 1.0e-04 | 1.0353 | 0.1304 |
| 100 | 0.7185 | 0.7175 | 1.0e-03 | 1.2141 | 0.4575 |
| 1000 | 1.8332 | 1.8232 | 1.0e-02 | 2.1014 | 1.6177 |
| 10000 | 6.2377 | 6.1377 | 1.0e-01 | 6.7127 | 6.0071 |

: DP-SGD with $q = 0.01$, $\sigma = 1$: $\varepsilon$ at $\delta = 10^{-5}$ after $k$ steps, from the pessimistic and optimistic PLDs on the grid $h = 10^{-5}$ (width $kh$), the RDP accountant, and the GDP central-limit approximation. {#tbl-num-dpsgd}

![$\varepsilon$ at $\delta = 10^{-5}$ against the number of steps $k$ for DP-SGD with $q = 0.01$, $\sigma = 1$: the RDP accountant (green), the GDP central-limit approximation (orange), and the pessimistic (red) and optimistic (blue) PLD bounds, which overlap at this scale. RDP lies above them and the CLT below them.](figures/numerical_accounting_epsilon_vs_k.svg){#fig-num-eps-k}

**Example.** After 10,000 steps the true $\varepsilon$ lies in $[6.1377, 6.2377]$; report 6.2377. RDP's 6.7127 is valid but over-states by 0.4750; after one step it gives 0.9555 against 0.1995, because its conversion pays about $\ln(1/\delta)/(\alpha - 1)$ whatever the curve. The CLT's 6.0071 is below the PLD lower bound: the subsampled PLD has a heavy right tail, exactly where $\delta = 10^{-5}$ looks.

**Pitfalls.** The CLT under-reports at every $k$ here: quoting 1.62 for 1000 steps claims less than the lower bound 1.8232. And a PLD accountant is only as good as its grid. At $k = 1000$ on $h = 10^{-3}$ the bounds are 2.3279 and 1.3279, the upper one worse than RDP's 2.1014. Look at the width before you trust the upper bound.

## Code

The first part solves @eq-num-gauss for $\varepsilon$: the reference values any accountant should reproduce with $q = 1$. The second is a toy PLD accountant that reproduces the sandwich of @fig-num-rounding; the grid $h = \ln 3$ contains every loss, so it gives the exact values.

```python
from math import ceil, erfc, exp, floor, log, sqrt


def normal_cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def gaussian_delta(eps: float, mu: float) -> float:
    """Exact delta(eps) of the Gaussian PLD N(mu^2/2, mu^2); k steps of multiplier sigma have mu = sqrt(k)/sigma."""
    return normal_cdf(mu / 2 - eps / mu) - exp(eps) * normal_cdf(-mu / 2 - eps / mu)


def gaussian_epsilon(delta: float, mu: float) -> float:
    lo, hi = 0.0, 1000.0  # delta(eps) falls as eps grows: bisect for delta(eps) = delta
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gaussian_delta(mid, mu) > delta else (lo, mid)
    return hi


PLD = dict[int, float]  # grid index i -> probability of the loss i * h


def discretize(losses: dict[float, float], h: float, up: bool) -> PLD:
    pld: PLD = {}
    for loss, p in losses.items():
        i = ceil(loss / h) if up else floor(loss / h)
        pld[i] = pld.get(i, 0.0) + p
    return pld


def compose(a: PLD, b: PLD) -> PLD:
    """The PLD of the sum of two independent losses: a convolution."""
    out: PLD = {}
    for i, p in a.items():
        for j, r in b.items():
            out[i + j] = out.get(i + j, 0.0) + p * r
    return out


def pld_delta(pld: PLD, h: float, eps: float) -> float:
    return sum(p * (1 - exp(eps - i * h)) for i, p in pld.items() if i * h > eps)


print(f"{gaussian_epsilon(1e-5, 1.0):.4f}")  # 4.3772
print(f"{gaussian_epsilon(1e-5, sqrt(10)):.4f}")  # 17.8566
print(f"{gaussian_epsilon(1e-5, 10.0):.4f}")  # 91.8173

def three_fold_rr(h: float, up: bool) -> str:
    """delta(0), delta(1), delta(2) of randomized response with eps0 = ln 3, composed three times."""
    step = discretize({log(3): 0.75, -log(3): 0.25}, h, up)
    three = compose(compose(step, step), step)
    return " ".join(f"{pld_delta(three, h, eps):.4f}" for eps in (0, 1, 2))


print(three_fold_rr(log(3), up=True))  # 0.6875 0.4190 0.3064
print(three_fold_rr(0.5, up=True))  # 0.7820 0.6758 0.3872
print(three_fold_rr(0.5, up=False))  # 0.5669 0.3648 0.2667
```

The last three lines are the exact, pessimistic and optimistic $\delta(0)$, $\delta(1)$, $\delta(2)$. For training runs use a production accountant. The full edition's script, [numerical_accounting.py](../../code/part04-privacy-accounting/numerical_accounting.py), implements the FFT accountant for the subsampled Gaussian in NumPy and prints @tbl-num-dpsgd:

```bash
uv run python code/part04-privacy-accounting/numerical_accounting.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| $\delta$ from the PLD | $\delta(\varepsilon) = \mathbb E[(1 - e^{\varepsilon - L})_+] = P[L > \varepsilon] - e^\varepsilon Q[L > \varepsilon]$ | Infinite losses count fully |
| Gaussian PLD | $\mathcal N(\mu^2/2, \mu^2)$, $\delta$ from @eq-num-gauss | $k$ steps: $\mu \to \sqrt k\,\mu$ |
| Composition | Convolve the per-step PLDs | Adaptive steps: convolve dominating pairs |
| Sandwich | Round up (pessimistic) and down (optimistic) | Width in $\varepsilon$ at most $kh$ |
| FFT | Pad each convolution to the sum of the input lengths | A short window wraps mass around silently |
| Subsampled Gaussian | Remove: mixture vs $\mathcal N(0, \sigma^2)$; add: swapped | Take the larger $\varepsilon$ of the two |
| Running example, $k = 10{,}000$ | PLD $[6.1377, 6.2377]$; RDP 6.7127; CLT 6.0071 | $q = 0.01$, $\sigma = 1$, $\delta = 10^{-5}$ |

: The results of this chapter. {#tbl-num-cheat}

**Rules of thumb**

- Report the pessimistic $\varepsilon$, with the optimistic one next to it as the error bar.
- Choose $h$ so that $kh$ is an error you can accept at your largest $k$, and check the width rather than trusting the grid.
- Compute both directions of add/remove for a subsampled mechanism.
- Check your accountant against the Gaussian reference values (4.3772, 17.8566, 91.8173 at $\delta = 10^{-5}$) before you trust it on DP-SGD.
- Use RDP and the CLT as quick estimates only: RDP over-reports, and the CLT can under-report.

## Further reading

- Gopi, Lee & Wutschitz, "Numerical Composition of Differential Privacy" (NeurIPS 2021): the PRV accountant, with explicit error bounds and a running time of order $\sqrt k$; read it when you need a fast accountant with certified error bars.
- Doroshenko, Ghazi, Kamath, Kumar & Manurangsi, "Connect the Dots: Tighter Discrete Approximations of Privacy Loss Distributions" (PoPETs 2022): the cleanest account of why the pessimistic bound is valid, and the best discretization on a given grid.
- Zhu, Dong & Wang, "Optimal Accounting of Differential Privacy via Characteristic Function" (AISTATS 2022): dominating pairs, adaptive composition and the two directions of Poisson subsampling.
- Google, `dp_accounting` library documentation (the `pld` module): a production PLD accountant with pessimistic and optimistic estimates; use it to check the numbers here.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/06-numerical-accounting.md).
