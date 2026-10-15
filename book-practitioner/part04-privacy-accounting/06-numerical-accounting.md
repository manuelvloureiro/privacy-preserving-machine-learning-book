# Numerical accounting with privacy loss distributions

## Why this matters

You trained a model with differential privacy (DP), using the private form of stochastic gradient descent (SGD) that runs through this part, called DP-SGD. The run took 10,000 steps. Each step drew a Poisson sample of the data, which includes every record independently with the same probability, the sampling rate $q$, here 0.01. It then clipped each gradient to a fixed length, the clipping norm, and added Gaussian noise. The noise multiplier $\sigma$, the noise standard deviation in units of the clipping norm, was 1. Now you need one number, the smallest privacy parameter $\varepsilon$ for which the whole run is $(\varepsilon, \delta)$-DP, with the slack $\delta$ set to $10^{-5}$.

The earlier chapters of this part give two answers, and neither says how far it is from the truth. Rényi differential privacy (RDP), from [Rényi DP](01-renyi-dp.md), gives an upper bound, but converting it to $\varepsilon$ and $\delta$ loses an amount you cannot see. The Gaussian central limit theorem (CLT) of [f-DP and Gaussian DP](04-gaussian-dp.md) treats many composed steps as a single Gaussian mechanism. That gives an approximation rather than a bound, so it may land on either side of the truth.

This chapter computes the answer itself, to whatever precision you are willing to pay for. It rests on two facts. The first is that $\delta$ depends on a mechanism only through the distribution of one number, the privacy loss of its output. That loss measures how strongly the output favours one dataset over its neighbour. The second is that composing steps adds their losses. So the distribution of the loss for a whole run is the distribution of a sum, which a computer can evaluate on a grid. Rounding the losses onto that grid in a chosen direction gives the rounding error a known sign. So the computer returns a certified upper bound and a lower bound, and you know in advance how far apart they can be.

## The privacy loss distribution {#sec-num-pld}

[Composition](../part03-dp-foundations/04-composition.md) measured how much one output gives away by its privacy loss, and this chapter builds everything on that number. Call your dataset $D$ and a neighbouring dataset $D'$. The mechanism's output has some distribution $P$ when it runs on $D$, and a distribution $Q$ when it runs on $D'$. For an output $y$, the values $P(y)$ and $Q(y)$ are its probability, or its density, under each. The privacy loss of the output is the logarithm of how much more likely it is under $D$,
$$L(y) = \ln\frac{P(y)}{Q(y)}.$$
The loss is positive when $y$ favours $D$, negative when it favours $D'$, and zero when the output says nothing about which dataset was used. Since the output is random, so is its loss. When the output is drawn from $P$, the distribution of its loss is the **privacy loss distribution (PLD)**.

The PLD carries everything you need to compute $\delta$, and the reason lies in how $\delta$ is defined. As [the definition](../part03-dp-foundations/01-definition.md) explains, each output may be up to $e^\varepsilon$ times more likely under $D$ for free, and $\delta$ pays only for the excess. The excess of an output is its probability under $D$ minus $e^\varepsilon$ times its probability under $D'$. Take an output with loss $\ell$. By the definition of the loss, its probability under $Q$ is $e^{-\ell}$ times its probability under $P$. Its excess is therefore its probability under $P$ times the factor $1 - e^{\varepsilon - \ell}$. When $\ell$ is at most $\varepsilon$, that factor is not positive, so the output is within the free allowance and adds nothing. When $\ell$ is larger, the output adds the fraction $1 - e^{\varepsilon - \ell}$ of its probability under $P$, which depends only on its loss. Adding up these contributions means averaging that fraction over the PLD.

Write $(x)_+$ for the positive part of a number, which is $x$ when $x$ is positive and 0 otherwise. Then for every $\varepsilon$,
$$\delta(\varepsilon) = \mathbb E\big[(1 - e^{\varepsilon - L})_+\big] = P[L > \varepsilon] - e^{\varepsilon}\,Q[L > \varepsilon].$$ {#eq-num-delta}
The average in the middle is over the loss $L$ of an output drawn from $P$. The form on the right reads the same quantity as a gap. It is the chance under $P$ that the loss exceeds $\varepsilon$, minus $e^\varepsilon$ times the same chance under $Q$. That form also needs the distribution of the loss under $Q$, but this is no extra information. Every probability under $Q$ is a probability under $P$ reweighted by $e^{-\ell}$, so the PLD determines both.

An output that is impossible under $D'$ has an infinite loss, and it counts fully in $\delta$. Seeing such an output proves that the dataset was $D$, and no factor $e^\varepsilon$ can cover a ratio with zero below it. So a PLD may put some of its probability at $+\infty$, and that probability goes into $\delta$ at every $\varepsilon$.

Because @eq-num-delta reads $\delta$ off the PLD and nothing else, the output space never enters. Two pairs of distributions with the same PLD have the same $\delta(\varepsilon)$ at every $\varepsilon$. A DP-SGD run that releases millions of parameters and a one-dimensional Gaussian pair can have the same PLD, and then they have the same privacy guarantee.

Not every distribution on the real line is a PLD, though. Each output's probability under $Q$ is its probability under $P$ times $e^{-\ell}$. So averaging $e^{-L}$ over the PLD adds up the probabilities under $Q$ of all the outputs that $P$ can produce. A total probability is at most 1, so a PLD must have
$$\mathbb E\big[e^{-L}\big] \le 1.$$
The converse holds as well. Any distribution of losses that meets this condition is the PLD of some pair of distributions, and the full edition builds that pair directly from it. @sec-num-discretize uses this to show that a rounded PLD still belongs to a real pair, which is what lets a rounded bound cover adaptive runs.

The weight $(1 - e^{\varepsilon - \ell})_+$ explains the shape of the answer. It is 0 for losses up to $\varepsilon$, and above $\varepsilon$ it rises gradually towards 1. So $\delta(\varepsilon)$ is the upper tail of the PLD beyond $\varepsilon$, with the losses just above $\varepsilon$ counting only a little (@fig-num-pld-weight). The plain tail probability $P[L > \varepsilon]$ is therefore also an upper bound on $\delta$, since the weight never exceeds 1. But it is a loose one, because it counts the outputs that are barely over $\varepsilon$ in full.

For the Gaussian mechanism the PLD has a closed form, so it serves as a check on everything that follows. Let $P$ be a normal distribution centred at 0, with standard deviation $\sigma$, and let $Q$ be the same distribution shifted by the query's sensitivity $s$. Write $\mu = s/\sigma$ for the shift measured in noise standard deviations. The two densities have the same shape, so the squared terms cancel in their log-ratio, and the loss is a straight-line function of the output. A linear function of a normal variable is normal, so the PLD is the normal distribution with mean $\mu^2/2$ and variance $\mu^2$, written $\mathcal N(\mu^2/2, \mu^2)$.

Write $\Phi$ for the standard normal distribution function, the chance that a standard normal variable is at most a given value. Under $Q$ the loss is also normal, with the same variance and the opposite mean, $-\mu^2/2$. Putting the two tails of the loss into @eq-num-delta gives
$$\delta(\varepsilon) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^{\varepsilon}\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big).$$ {#eq-num-gauss}
This is the exact $\delta$ of the Gaussian mechanism from [The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md). It is also the privacy curve of $\mu$-Gaussian differential privacy (GDP) in [f-DP and Gaussian DP](04-gaussian-dp.md).

As an example, take sensitivity 1 and $\sigma = 1$, so the shift is one noise standard deviation and the PLD is $\mathcal N(0.5, 1)$. At $\varepsilon = 1$ the formula gives $\delta = 0.1269$, which is the green area in @fig-num-pld-weight. Because $\delta$ falls as $\varepsilon$ grows, you can also go the other way and solve for the privacy parameter that reaches a target $\delta$. At $\delta = 10^{-5}$ this mechanism is $(4.3772, 10^{-5})$-DP.

![The Gaussian PLD for $\mu = 1$ (blue), the weight $(1 - e^{\varepsilon - \ell})_+$ at $\varepsilon = 1$ (orange, scaled by 0.4 to fit), and their product (green). The green area is $\delta(1) = 0.1269$. Only losses above $\varepsilon$ count, and those just above it count little.](figures/numerical_accounting_pld_weight.svg){#fig-num-pld-weight}

## Composition is convolution {#sec-num-convolution}

A DP-SGD run of $k$ steps releases one noisy gradient per step, and the guarantee has to cover the whole sequence of outputs. Suppose first that the steps run independently. Then the probability of the whole sequence is the product of the per-step probabilities, on either dataset. The ratio of two products is the product of the ratios, and the logarithm turns that product into a sum. So the privacy loss of the run is the sum of the per-step losses, as [Composition](../part03-dp-foundations/04-composition.md) showed.

The per-step losses are independent, because each one is a function of its own step's output. The distribution of a sum of independent variables is the **convolution** of their distributions. Take two independent losses. For their sum to equal some value $\ell$, the first must take some value $a$ and the second the value $\ell - a$. So the chance of the sum $\ell$ adds up the chances of all these splits, and convolution is exactly that computation. The PLD of $k$ independent steps is therefore the $k$-fold convolution of the per-step PLDs. If any one step's loss is infinite, so is the total, because one step that gives the dataset away gives away the whole run.

This step is exact, and that sets it apart from the composition theorems of [Composition](../part03-dp-foundations/04-composition.md). Basic, advanced and even optimal composition see each step only through its own values of $\varepsilon$ and $\delta$. They must therefore cover the worst step that has those values. Convolution uses each step's actual PLD and loses nothing. All the error in a PLD accountant comes from computing the convolution on a machine, through the grid of @sec-num-discretize and the floating point of @sec-num-fft.

The Gaussian mechanism gives a check that any accountant must pass. When independent normal losses are added, their means add and so do their variances. So $k$ Gaussian steps with shift $\mu$ have a normal PLD with mean $k\mu^2/2$ and variance $k\mu^2$. That is the PLD of a single Gaussian step whose shift is $\sqrt k\,\mu$. Ten steps with $\sigma = 1$ therefore act like one step with shift $\sqrt{10}$, and by @eq-num-gauss they are $(17.8566, 10^{-5})$-DP. In the same way, a hundred steps act like one step with shift 10 and are $(91.8173, 10^{-5})$-DP. An accountant that cannot reproduce these values for Gaussian steps should not be trusted on DP-SGD.

The same picture explains why the CLT is plausible. A $k$-fold convolution is the distribution of a sum of $k$ independent losses. When each loss has a finite variance, the ordinary central limit theorem says that such a sum looks Gaussian for large $k$. That is the GDP central limit theorem of [f-DP and Gaussian DP](04-gaussian-dp.md). The PLD accountant does not need this approximation, because it computes the sum's distribution directly.

DP-SGD steps are not independent, though, and the argument above needs independence. Each gradient depends on the current model, and the model depends on the earlier outputs. So the pair of distributions for a step changes with what came before, and its loss is no longer independent of the earlier losses.

The repair is to replace each step by a fixed pair of distributions that is at least as easy to tell apart as the step, whatever happened before. Such a pair is called a **dominating pair**. Roughly, its $\delta$ curve must lie above the step's for every pair of neighbouring datasets. Precisely, the condition uses the hockey-stick divergence of [Divergences](../part01-prerequisites/03-divergences.md#sec-div-hockey), which is the $\delta$ of @eq-num-delta with a general weight in place of $e^\varepsilon$. The pair's divergence must be at least the step's at every weight that is not negative, including weights below 1. Zhu, Dong and Wang (2022) prove that if a fixed pair dominates each step, then the fixed pairs run independently in sequence dominate the adaptive run. So convolving the PLDs of the dominating pairs gives a valid $\delta(\varepsilon)$ for the adaptive run.

The bound is exactly as tight as the dominating pair. For the Gaussian mechanism with sensitivity 1, the normal distributions centred at 0 and at 1, both with standard deviation $\sigma$, dominate every step. Some pair of neighbouring datasets produces exactly these two distributions, so nothing is lost. A subsampled mechanism is less simple, because its pair depends on how the batches are drawn. It even needs two pairs, one for adding a record and one for removing it (@sec-num-subsampled).

## Pessimistic and optimistic discretization {#sec-num-discretize}

A computer cannot convolve continuous distributions, so it puts the losses on a grid, the whole-number multiples of a spacing $h$. Any rounding changes $\delta$. The question is whether you can round so that the error has a known sign for any number of steps. Then the answer would still be a guarantee.

You can, because of the weight in @eq-num-delta. The weight never decreases as the loss grows. So if you move every loss up, each output's contribution to $\delta$ can only grow, and so can their average. If you move every loss down, $\delta$ can only shrink. That gives two ways to round.

The **pessimistic** discretization moves every loss up. It rounds each loss up to the next grid point. To keep the grid finite, it also moves the far upper tail to $+\infty$, where that tail holds at most a tiny mass $\tau$. It lifts the far lower tail to the lowest grid point it keeps. The **optimistic** discretization does the opposite. It rounds each loss down to the grid point below and lowers the far upper tail to the highest grid point it keeps. It drops the far lower tail, so each of its moves is downward or removes mass.

Rounding each step before convolving keeps the direction of every move in the total. Take $k$ independent steps. In every outcome, each step's rounded-up loss is at least its true loss, which is at least its rounded-down loss. Adding over the steps keeps that order for the totals. The order carries over to $\delta$, because $\delta$ is the average of a weight that grows with the total loss. So for any number $k$ of independent steps and every $\varepsilon$,
$$\delta_{\text{optimistic}}(\varepsilon) \;\le\; \delta(\varepsilon) \;\le\; \delta_{\text{pessimistic}}(\varepsilon).$$ {#eq-num-sandwich}
The same holds for steps with different PLDs, as long as each is rounded the same way. The tails can also be truncated again after every convolution. Each truncation moves mass in the same direction as the rounding, or on the optimistic side removes it.

The width of this sandwich is known in advance. Each step's two roundings are at most one grid step apart, so after $k$ steps the two totals are at most $kh$ apart. Adding a constant to every loss shifts the whole $\delta$ curve to the right by that constant. The pessimistic total is at most the optimistic total plus $kh$, and $\delta$ grows with the loss. So the pessimistic curve lies no higher than the optimistic curve shifted right by $kh$. At your target $\delta$, therefore, the pessimistic and optimistic values of $\varepsilon$ are at most $kh$ apart when no tail was truncated.

For a continuous PLD the width is almost exactly $kh$, because a loss almost never sits on a grid point and so its two roundings are a full step apart. The error therefore grows linearly in the number of steps. To keep the same width after 10,000 steps as after one, you need a grid 10,000 times finer.

The two sides are not equally strong. Rounding up raises every loss, which lowers $e^{-L}$, so the rounded PLD still meets the condition of @sec-num-pld that the average of $e^{-L}$ is at most 1. By @sec-num-pld it is then the PLD of a real pair of distributions. That pair dominates the original one, since its $\delta$ is at least as large at every $\varepsilon$, and at weight 0 both divergences equal 1 (Doroshenko et al. 2022). So the upper bound also holds for adaptive composition, because the rounded pairs dominate the steps and @sec-num-convolution showed that convolving dominating pairs covers adaptive runs.

Rounding down lowers every loss, which can push the average of $e^{-L}$ above 1, and then no pair of distributions has that PLD. The optimistic value is still a lower bound on the $\delta$ of $k$ independent runs of the original pair, because the sandwich holds for them. It is a lower bound on the mechanism itself only if some neighbouring datasets actually produce that pair. This holds for the Gaussian mechanism and the subsampled Gaussian, the two cases of this chapter, because for each of them some neighbouring datasets produce exactly the pair used.

The midpoint of the two sides is a good estimate, but it is not a bound. Suppose the losses are spread roughly evenly inside each grid cell, as for a smooth PLD on a fine grid. Then rounding up moves a loss up by about $h/2$ on average, and rounding down moves it down by as much. So the midpoint is nearly unbiased. Rounding each loss to the nearest grid point is a good estimate for the same reason, but some losses then move down, so it gives no guarantee in either direction. Only the two sides are guarantees.

The truncated tails cost little. Each truncation moves a mass of at most $\tau$ to $+\infty$, where its weight is 1. So it adds at most $\tau$ to $\delta$. With $\tau = 10^{-15}$ and the few dozen truncations that @sec-num-fft makes, this is far below any $\delta$ you would report.

Randomized response shows the sandwich on a small example. Take the version whose privacy parameter is $\ln 3$ ([Randomized response](../part03-dp-foundations/01-definition.md#sec-dpdef-rr)). It reports the true bit with probability $3/4$, and otherwise the flipped bit. The true report is three times likelier under $D$ than under $D'$, so its loss is $\ln 3 = 1.0986$, while the flipped report has loss $-\ln 3$. On the coarse grid $h = 0.5$, rounding up moves the positive loss to 1.5 and the negative one to $-1$. Rounding down instead moves them to 1 and $-1.5$. The rounded-up PLD has an average of $e^{-L}$ of 0.8469, so it belongs to a real pair. The rounded-down PLD has 1.3963, above 1, so no pair of distributions has it.

Now compose three steps. The total loss depends only on how many of the three steps gave the true report, so it takes four possible values. In each case, the rounded-up total sits above the exact total and the rounded-down total below it. The gap between the two roundings is the same in every case, three steps times $h$, which is 1.5. @tbl-num-rr-delta gives the resulting values of $\delta$, and @fig-num-rounding draws them. At $\varepsilon = 1$, for example, the exact $\delta$ is 0.4190. The pessimistic value 0.6758 lies above it and the optimistic 0.3648 below it.

| $\varepsilon$ | Exact $\delta$ | Pessimistic | Optimistic |
| --- | --- | --- | --- |
| 0 | 0.6875 | 0.7820 | 0.5669 |
| 1 | 0.4190 | 0.6758 | 0.3648 |
| 2 | 0.3064 | 0.3872 | 0.2667 |

: $\delta(\varepsilon)$ of three-fold randomized response with privacy parameter $\ln 3$, exact and from the PLDs rounded on the grid $h = 0.5$. {#tbl-num-rr-delta}

![Left: the PLD of three-fold randomized response with privacy parameter $\ln 3$ (black), rounded up (red) and rounded down (blue) on the grid $h = 0.5$. Right: the three $\delta(\varepsilon)$ curves. The pessimistic one lies above the exact one and the optimistic one below it at every $\varepsilon$.](figures/numerical_accounting_rounding.svg){#fig-num-rounding}

## Composition by FFT {#sec-num-fft}

A discretized PLD is a vector of masses, one for each grid point, plus a mass at $+\infty$. On a common grid, adding two losses adds their grid indices. So the masses of the sum are the discrete convolution of the two mass vectors, and the mass at $+\infty$ collects every pair in which either loss is infinite. Computed directly, the convolution costs a number of operations proportional to the product of the two lengths, and the vectors are long. On the grid $h = 10^{-5}$, a single DP-SGD step of the running example has 329590 grid points. After 10,000 steps the composed PLD has 2103251.

The fast Fourier transform (FFT) makes this cheap, and Koskela, Jälkö and Honkela (2020) introduced it to PLD accounting. The Fourier transform attaches to each index $i$ the $i$-th power of a fixed complex number. Powers multiply by adding their exponents, so indices add exactly the way losses do. That turns the convolution of two vectors into the pointwise product of their transforms. The FFT computes a transform of length $N$ in a number of operations of order $N\log N$, rather than $N^2$.

The FFT has one trap, and it follows from the same construction. The transform works on a window of $N$ grid points, and its fixed complex number is one whose $N$-th power is 1. So its powers repeat every $N$ steps, and indices add modulo $N$. Mass whose index sum should fall off one end of the window reappears at the other end. This is called wraparound. The fix is to pad every convolution with zeros, to a length at least the sum of the two input lengths. Then no sum of indices reaches the end of the window, and nothing wraps around.

Ten Gaussian steps with $\mu = 1$ show the size of the error. On the grid $h = 10^{-3}$, the padded computation gives a pessimistic $\delta(1)$ of 0.81891018. Now make the window too short, covering only the losses from 0 to 10 around the mean total loss of 5. It gives 0.83478562 instead, which is wrong in the second decimal. Negative total losses, which should count nothing at $\varepsilon = 1$, wrap around to the top of the window, where they count almost fully. Only a window four times as wide recovers the padded value. In general the error from wraparound can go either way, so a wrapped result is no bound at all.

You could compose $k$ identical steps in one go, by raising the transform to the $k$-th power. But the padded window would then have to be $k$ times as wide as a single step, because the total loss can range $k$ times as far. The full edition's script avoids that window by composing in stages and trimming the tails in between. It builds the $k$-fold convolution by repeated squaring. It convolves the PLD with itself to get two steps, convolves that with itself to get four, then eight, and so on. It then combines the powers in the binary expansion of $k$, so that ten steps are eight steps convolved with two. This takes at most $2\log_2 k$ convolutions.

After each convolution, the script truncates both tails at the mass $\tau = 10^{-15}$, in the safe direction of @sec-num-discretize. The vector then grows like the spread of the composed PLD, not like the worst-case range of the total loss. That spread is the standard deviation of a sum of $k$ independent losses, which grows like $\sqrt k$. For the ten Gaussian steps above, the vector grows from 15884 to 50257 grid points, about $\sqrt{10}$ times as many. The result matches that of raising the transform to the tenth power on the full window.

The FFT is not exact in floating-point arithmetic. It computes each mass with an error of about $10^{-16}$ times the largest mass, so tiny masses far in the tails come out as noise, some of it negative. The script sets negative masses to zero. Summed over the two million grid points of the 10,000-step PLD, this noise is many orders of magnitude below a $\delta$ of $10^{-5}$. But it is not part of the certified bound. The privacy loss random variable (PRV) accountant, a variant described at the end of this section, also leaves it out of its error analysis, as negligible for $\delta$ above $10^{-10}$.

Setting up the grid takes two choices. The spacing $h$ follows from the sandwich width you can accept, which is $kh$, while the range of losses follows from the tails. The composed PLD has $k$ times the mean and $k$ times the variance of one step. So a window of many standard deviations around its mean holds all but a tiny mass. Chebyshev's inequality ([Concentration inequalities](../part01-prerequisites/02-concentration.md)) bounds the mass outside such a window from the variance alone, loosely but without needing anything else.

@tbl-num-gauss-grid shows the sandwich on Gaussian steps with $\mu = 1$, on the grid $h = 10^{-3}$, against the exact values of @sec-num-convolution. The width is exactly $kh$, which is 0.01 after ten steps and 0.1 after a hundred. The exact value sits in the middle, because rounding up moves each step's loss up by $h/2$ on average and rounding down moves it down by as much.

| $k$ | Exact $\varepsilon$ | Pessimistic | Optimistic |
| --- | --- | --- | --- |
| 10 | 17.8566 | 17.8616 | 17.8516 |
| 100 | 91.8173 | 91.8673 | 91.7673 |

: Composed Gaussian steps with $\mu = 1$ on the grid $h = 10^{-3}$. The table gives $\varepsilon$ at $\delta = 10^{-5}$ from the pessimistic and optimistic PLDs, against the exact value. {#tbl-num-gauss-grid}

Three variants of the same idea are in use. The **PRV accountant** of Gopi, Lee and Wutschitz (2021) also discretizes the loss and convolves with the FFT. It rounds each loss to the nearest grid point, and then shifts the whole grid by one constant so that the rounded loss keeps the mean of the true one. Once the means match, the per-step rounding errors average to zero, so over $k$ independent steps they partly cancel instead of piling up. The sum of the errors grows like $\sqrt k\,h$ instead of $kh$. To keep the error at a given size, the PRV accountant's grid spacing can therefore shrink like $1/\sqrt k$ as the steps grow, while rounding up needs $1/k$. The PRV accountant proves explicit error bounds on $\varepsilon$, and for $k$ identical steps its running time is of order $\sqrt k$, up to logarithmic factors.

**Connect the dots**, by Doroshenko and coauthors (2022), builds the pessimistic PLD on a given grid in a different way. Instead of rounding each loss, it starts from the values of $\delta(\varepsilon)$ at the grid points and interpolates the curve between them. It then takes the PLD on the grid whose $\delta$ curve follows that interpolation. The authors prove that this is the best pessimistic PLD on that grid, and it allows much coarser grids than rounding up. **Characteristic-function accounting** (Zhu, Dong and Wang, 2022) skips the grid of masses and works with the Fourier transform of the PLD directly. The transform of a sum of independent losses is the product of their transforms, for the same reason as above, and $\delta$ comes back by numerical integration. Google's `dp_accounting` library implements PLD accounting with pessimistic and optimistic rounding and with connect the dots.

## The subsampled Gaussian {#sec-num-subsampled}

A DP-SGD step with Poisson sampling includes each record independently with probability $q$, sums the clipped gradients of the records it drew, and adds Gaussian noise. [Amplification by subsampling](05-subsampling.md) reduces this step to a pair of one-dimensional distributions, seen from the record $x$ that one neighbour holds and the other lacks. This uses add/remove neighbours, where one dataset is the other with one record added. So $D$ lacks the record and $D' = D \cup \{x\}$ holds it. On $D$, the noisy sum measured along the worst-case line in gradient space, in units of the clipping norm, is a normal distribution centred at 0, with standard deviation $\sigma$. On $D'$, the record is drawn with probability $q$, and then it shifts the centre by 1, the sensitivity. Otherwise it changes nothing. So the output on $D'$ is a mixture, centred at 1 with probability $q$ and at 0 with probability $1 - q$.

A guarantee under add/remove must cover both orders of the pair, and the two orders give different PLDs. The **remove** direction, named for going from $D'$ to its neighbour by removing $x$, puts the dataset with the record in the role of $P$. So $P$ is the mixture on $D'$, and $Q$ is the plain normal distribution on $D$,
$$P = (1 - q)\,\mathcal N(0, \sigma^2) + q\,\mathcal N(1, \sigma^2), \qquad Q = \mathcal N(0, \sigma^2).$$
The **add** direction swaps the two. Zhu, Dong and Wang (2022) prove that each pair dominates its direction. So composing a pair's PLD over $k$ steps gives a valid $\delta$ for that direction, even for an adaptive run.

The loss of the remove pair follows from dividing the mixture by the plain normal density. The result is $1 - q$, plus $q$ times the ratio of the shifted normal density to the unshifted one. That ratio is an exponential in the output, as for the Gaussian PLD of @sec-num-pld, so the remove loss is
$$L_{\mathrm{rem}}(y) = \ln\Big((1 - q) + q\,e^{(2y - 1)/(2\sigma^2)}\Big),$$
with $y$ drawn from the mixture. Swapping $P$ and $Q$ flips the sign of a log-ratio, so the add loss is the negative of this function, with $y$ drawn from the plain normal distribution.

With this formula the PLD can be discretized with no approximation beyond the rounding itself. The remove loss increases with the output, so each bin of losses on the grid corresponds to an interval of outputs. The mass of the bin is the probability of that interval under the mixture, which is a weighted difference of values of $\Phi$. The script therefore never samples.

The formula also explains why the two directions behave so differently (@fig-num-subsampled-loss). For an ordinary output the exponential is small, and the remove loss stays just above its lower limit $\ln(1 - q)$. A large output, though, is far likelier to come from the shifted part of the mixture. There the remove loss grows without limit, because a rare large output betrays that the record was used. The add loss is the mirror image, bounded above by $-\ln(1 - q)$. For a sampling rate of 0.01 that bound is 0.0101. Since $\delta$ counts only losses above $\varepsilon$, a single add step has no $\delta$ at all once $\varepsilon$ passes that bound.

![Left: the densities of $\mathcal N(0, \sigma^2)$ and of the mixture $(1 - q)\mathcal N(0, \sigma^2) + q\mathcal N(1, \sigma^2)$ for $q = 0.01$, $\sigma = 1$, on a log scale. They differ only in the far right tail. Right: the privacy loss as a function of the output for the remove pair (red) and the add pair (blue).](figures/numerical_accounting_subsampled_loss.svg){#fig-num-subsampled-loss}

Take one step of the running example, with $q = 0.01$ and $\sigma = 1$. The remove loss then lies between $-0.0101$ and 3.2858, apart from a tiny mass in the far tail. On the grid $h = 10^{-5}$ that range takes the 329590 grid points mentioned in @sec-num-fft. The rounded-up remove PLD has an average of $e^{-L}$ of 0.999995, below 1, so it is a valid PLD, as @sec-num-discretize requires. A single step's $\delta$ also has a closed form, as two normal tail probabilities, and the pessimistic $\delta$ lies slightly above it wherever the script checks. At $\delta = 10^{-5}$, one step has $\varepsilon = 0.1995$ in the remove direction but only $0.0091$ in the add direction.

At $\varepsilon = 0$ the exact $\delta$ of the two directions is the same, $3.829\times10^{-3}$, and the reason is a symmetry. When $\varepsilon$ is 0, $\delta$ is the total amount by which $P$ exceeds $Q$, which is the total variation distance. Both distributions have total probability 1, so what one has in excess of the other must be balanced by an equal shortfall elsewhere. So swapping the two, which turns one direction into the other, leaves $\delta(0)$ unchanged.

In relative terms, composition brings the directions closer, because the add direction's upper bound grows to 5.6615 after 10,000 steps. Still, the remove direction gives the larger $\varepsilon$ at every number of steps in @tbl-num-dpsgd. That is an observation about these numbers, not a theorem, so an accountant should compose both directions and report the larger $\varepsilon$. Zhu, Dong and Wang (2022) make the same point. An accountant that composes only the remove pair covers only half of the add/remove relation.

This mixture, and so this pair, comes from Poisson sampling. A data loader that forms batches another way is not covered by it, and [Amplification by subsampling](05-subsampling.md) explains what applies instead.

## The running example: DP-SGD {#sec-num-dpsgd}

The pieces now fit together. For each direction, discretize the subsampled-Gaussian PLD up and down, and compose it $k$ times by FFT. Then read off the value of $\varepsilon$ at which $\delta$ reaches $10^{-5}$. That last step can be exact. Between two grid points, the set of grid losses above $\varepsilon$ does not change, so @eq-num-delta becomes a constant minus $e^\varepsilon$ times another constant, and you can solve it directly. Finally, take the larger $\varepsilon$ of the two directions.

@tbl-num-dpsgd compares these PLD bounds, on the grid $h = 10^{-5}$, with two accountants from earlier chapters. The RDP column is the RDP accountant of [Amplification by subsampling](05-subsampling.md), with the improved conversion of [From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md). The last column is the central-limit approximation of GDP. It replaces the $k$ steps by a single Gaussian pair with shift $\mu = q\sqrt{k(e^{1/\sigma^2} - 1)}$, as in [f-DP and Gaussian DP](04-gaussian-dp.md), and reads $\varepsilon$ off @eq-num-gauss. After 10,000 steps this shift is 1.3108.

| $k$ | PLD upper | PLD lower | Width | Add direction, upper | RDP | GDP CLT |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.1995 | 0.1994 | 1.0e-05 | 0.0091 | 0.9555 | 0.0368 |
| 10 | 0.3800 | 0.3799 | 1.0e-04 | 0.0655 | 1.0353 | 0.1304 |
| 100 | 0.7185 | 0.7175 | 1.0e-03 | 0.3387 | 1.2141 | 0.4575 |
| 1000 | 1.8332 | 1.8232 | 1.0e-02 | 1.4217 | 2.1014 | 1.6177 |
| 10000 | 6.2377 | 6.1377 | 1.0e-01 | 5.6615 | 6.7127 | 6.0071 |

: DP-SGD with $q = 0.01$ and $\sigma = 1$: $\varepsilon$ at $\delta = 10^{-5}$ after $k$ steps. The PLD columns come from the pessimistic and optimistic PLDs on the grid $h = 10^{-5}$, each the larger of the two directions, and the width is their difference, $kh$. The other columns give the pessimistic PLD of the add direction alone, the RDP accountant, and the GDP central-limit approximation. {#tbl-num-dpsgd}

![$\varepsilon$ at $\delta = 10^{-5}$ against the number of steps $k$ for DP-SGD with $q = 0.01$, $\sigma = 1$: the RDP accountant (green), the GDP central-limit approximation (orange), and the pessimistic (red) and optimistic (blue) PLD bounds on the grid $h = 10^{-5}$. The two PLD curves overlap at this scale. RDP lies above them and the CLT below them.](figures/numerical_accounting_epsilon_vs_k.svg){#fig-num-eps-k}

The PLD pins down the truth (@fig-num-eps-k). After 10,000 steps the true $\varepsilon$ lies between 6.1377 and 6.2377, and you report the upper value because only it is a guarantee. This bracket is wider than the one after 1000 steps, since the width grows like $kh$.

RDP gives a valid bound, but a loose one. After 10,000 steps it gives 6.7127, which is 0.4750 above the PLD upper bound. The single step shows the reason most clearly, with 0.9555 from RDP against 0.1995 from the PLD. For each of its orders $\alpha$, RDP summarizes the loss by a single average, a moment ([Rényi DP](01-renyi-dp.md)). Turning one moment into a tail bound costs about $\ln(1/\delta)/(\alpha - 1)$, whatever the true curve looks like.

The CLT errs in the other direction, which is worse, because it falls below the PLD lower bound at every $k$ in the table. After 1000 steps, for example, the lower bound is 1.8232 and the CLT gives only 1.6177. So quoting the CLT value would claim a guarantee that the run does not have. The reason lies in the shape of the subsampled PLD (@fig-num-histograms). Convolution makes the bulk of the distribution bell-shaped, but its right tail stays heavier than a Gaussian's. That matters because a $\delta$ of $10^{-5}$ reads exactly this far tail, where the Gaussian approximation is worst.

![The pessimistic remove-direction PLD of DP-SGD with $q = 0.01$, $\sigma = 1$ after 1, 10 and 100 steps (log scale). The dotted line is the mean. One step piles its mass just above $\ln(1 - q)$ and has a long right tail. Convolution moves the bulk towards a bell shape, but the right tail stays heavier than the left.](figures/numerical_accounting_histograms.svg){#fig-num-histograms}

A PLD accountant is only as good as its grid. @tbl-num-width repeats the run of 1000 steps on three grids, and the width is exactly $kh$ on each. On the coarsest grid the width is 1, and the upper bound of 2.3279 is then worse than RDP's 2.1014.

Finer grids cost memory and time, though, because each tenfold refinement of $h$ cuts the same range of losses into ten times as many bins. After 1000 steps, for example, the pessimistic remove PLD grows from 7687 bins on the coarsest grid to 767721 on the finest. So choose $h$ for the largest $k$ you will run, and check the width before you trust the upper bound.

| $h$ | Upper | Lower | Width |
| --- | --- | --- | --- |
| $10^{-3}$ | 2.3279 | 1.3279 | 1.0000 |
| $10^{-4}$ | 1.8782 | 1.7782 | 0.1000 |
| $10^{-5}$ | 1.8332 | 1.8232 | 0.0100 |

: DP-SGD after $k = 1000$ steps: the PLD bounds on $\varepsilon$ at $\delta = 10^{-5}$ on three grids. {#tbl-num-width}

## Code

The snippet below has two parts. The first solves @eq-num-gauss for $\varepsilon$, which gives the Gaussian reference values that any accountant should reproduce. The second is a toy PLD accountant that reproduces the sandwich of @tbl-num-rr-delta. On the grid $h = \ln 3$ every loss of randomized response is a grid point, so rounding changes nothing and the accountant returns the exact values.

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

The last three lines print the exact, pessimistic and optimistic values of $\delta$ at $\varepsilon$ equal to 0, 1 and 2. For real training runs, use a production accountant. The full edition's script, [numerical_accounting.py](../../code/part04-privacy-accounting/numerical_accounting.py), implements the FFT accountant for the subsampled Gaussian in NumPy and prints @tbl-num-dpsgd. It takes from half a minute to a few minutes, almost all of it on the runs of 10,000 steps.

```bash
uv run python code/part04-privacy-accounting/numerical_accounting.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| $\delta$ from the PLD | $\delta(\varepsilon) = \mathbb E[(1 - e^{\varepsilon - L})_+] = P[L > \varepsilon] - e^\varepsilon Q[L > \varepsilon]$ | Infinite losses count fully |
| Valid PLD | $\mathbb E[e^{-L}] \le 1$ | Any such distribution is the PLD of some pair |
| Gaussian PLD | $\mathcal N(\mu^2/2, \mu^2)$, $\delta$ from @eq-num-gauss | $k$ steps: $\mu \to \sqrt k\,\mu$ |
| Composition | Convolve the per-step PLDs | Adaptive steps: convolve dominating pairs |
| Sandwich | Round up (pessimistic) and down (optimistic) | Width in $\varepsilon$ at most $kh$; only the upper bound holds for adaptive runs |
| FFT | Pad each convolution to at least the sum of the input lengths | A short window wraps mass around silently |
| Subsampled Gaussian | Remove: mixture against $\mathcal N(0, \sigma^2)$; add: the two swapped | Compose both, report the larger $\varepsilon$ |
| Running example, $k = 10{,}000$ | PLD $[6.1377, 6.2377]$; RDP 6.7127; CLT 6.0071 | $q = 0.01$, $\sigma = 1$, $\delta = 10^{-5}$ |

: The results of this chapter. {#tbl-num-cheat}

**Rules of thumb**

- Report the pessimistic $\varepsilon$, with the optimistic one next to it as the error bar.
- Choose $h$ so that $kh$ is an error you can accept at your largest $k$, and check the width rather than trusting the grid.
- Pad every FFT convolution so that no mass wraps around, and truncate the tails in the safe direction to keep the vectors short.
- Before you trust an accountant on DP-SGD, check it against the Gaussian reference values of @sec-num-convolution, as the snippet in the Code section does.
- Use RDP and the CLT as quick estimates only. RDP over-reports, and the CLT can under-report in the tail that $\delta = 10^{-5}$ looks at.

## Further reading

- Gopi, Lee & Wutschitz, "Numerical Composition of Differential Privacy" (NeurIPS 2021): the PRV accountant, with explicit error bounds and a running time of order $\sqrt k$. Read it when you need a fast accountant with certified error bars.
- Doroshenko, Ghazi, Kamath, Kumar & Manurangsi, "Connect the Dots: Tighter Discrete Approximations of Privacy Loss Distributions" (PoPETs 2022): the cleanest account of why the pessimistic bound is valid, and the best discretization on a given grid.
- Zhu, Dong & Wang, "Optimal Accounting of Differential Privacy via Characteristic Function" (AISTATS 2022): dominating pairs, adaptive composition and the two directions of Poisson subsampling.
- Google, `dp_accounting` library documentation (the `pld` module): a production PLD accountant with pessimistic and optimistic estimates. Use it to check the numbers here.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/06-numerical-accounting.md), except the dominating-pair theorems of Zhu, Dong & Wang (2022), the error bounds and running time of the PRV accountant, the optimality of connect the dots and the GDP central-limit shift, which it cites or takes from [f-DP and Gaussian DP](04-gaussian-dp.md).
