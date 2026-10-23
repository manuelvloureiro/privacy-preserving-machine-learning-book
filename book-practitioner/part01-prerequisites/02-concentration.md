# Concentration inequalities

## What it is

A concentration inequality bounds the chance that a random quantity lands more than some distance from its mean. Call that event a failure, its chance the **failure probability** $\beta$, and the distance the **width**. Fix $\beta$ and solve for the width to get an error bar, or fix the width and solve for the number of samples.

**Markov's inequality** needs only the mean of a quantity that cannot be negative, and caps the chance of reaching a threshold at the mean divided by the threshold. **Chebyshev's inequality** adds the variance $\sigma^2$. With $\mu$ for the mean, the chance of landing at least a distance $t$ from it is at most
$$\Pr[|X - \mu| \ge t] \le \frac{\sigma^2}{t^2}.$$
Setting this to $\beta$ gives the width $\sigma/\sqrt\beta$ for any distribution, at a steep price. At 95% it asks for 4.5 standard deviations, where the one-sided bound $\sigma\sqrt{2\ln(1/\beta)}$ for a normal-shaped tail asks for 2.4. At one in a million the gap grows to 1,000 against 5.3.

## When you meet it

You meet these bounds when you put an error bar on a mean or a rate, plan a sample size, or report many numbers at once.

## How to use it

Find what you know in @tbl-conc-choose.

| What you know | Use | Width at failure probability $\beta$ |
| --- | --- | --- |
| The variance $\sigma^2$ | Chebyshev | $\sigma/\sqrt\beta$ |
| A mean, with standard deviation $s$ estimated from the data | Normal approximation | $z_{1-\beta/2}\,s/\sqrt n$, smaller but not a guarantee; solve for $n$ to plan |
| The exact noise distribution ([Probability distributions](01-distributions.md)) | Its quantile | Laplace $b\ln(1/\beta)$; normal $\sigma z_{1-\beta/2}$ |
| A count with mean $\mu$ | Chernoff | a miss by a fraction $r$ has chance $\le e^{-r^2\mu/(2+r)}$ above, $e^{-r^2\mu/2}$ below |
| The mean of $n$ values in a range of length $R$ | Hoeffding | $R\sqrt{\ln(2/\beta)/(2n)}$ |
| A rate: $k$ errors out of $n$ | Clopper–Pearson | exact binomial bound |

: Which bound to use, and the two-sided width it gives. {#tbl-conc-choose}

**Hoeffding's inequality** is the planning tool, because it needs only the range of the values. Solving its width for $n$ gives the sample size for a width $t$:
$$n \ge \frac{R^2\ln(2/\beta)}{2t^2}.$$
For accuracy, an average of zeros and ones, the range $R$ is 1. So measuring it to within one percentage point at 95% takes 18,445 test records.

For a small rate, Hoeffding is too wide: it assumes the largest variance the range allows. Take one false positive in 1,000 negatives, with one-sided 95% bounds. Hoeffding adds 0.039 to the measured rate, while **Bernstein's inequality** uses the variance $\sigma^2$ and adds only 0.0045. For values in $[0, 1]$ its width is $\sqrt{2\sigma^2\ln(1/\beta)/n} + 2\ln(1/\beta)/(3n)$, and a variance estimated from the same data makes it a heuristic. The exact **Clopper–Pearson** bound, from the binomial distribution, gives 0.0047. With no errors it gives 0.0030, close to the "rule of three" bound $3/n$.

Noise of a known distribution needs only its quantile: Laplace noise of scale 2 lies within 5.99 of zero with 95% probability. For many draws at once, the **union bound** caps the chance of any failure at the sum of the chances, so give each of $k$ draws $\beta/k$. Because the Laplace width grows only with the logarithm of $k$, covering a thousand draws widens 5.99 only to 19.81.

The snippet prints these numbers, with the one-sided bound for $k$ errors in $n$ as `stats.beta.ppf(0.95, k + 1, n - k)` and a two-sided interval for 10 errors in 10,000.

```python
import math

from scipy import stats

print(math.ceil(math.log(2 / 0.05) / (2 * 0.01**2)))  # 18445
print(f"{stats.beta.ppf(0.95, 1, 1000):.4f} {stats.beta.ppf(0.95, 2, 999):.4f}")  # 0.0030 0.0047
ci = stats.binomtest(10, 10_000).proportion_ci(confidence_level=0.95, method="exact")
print(f"{ci.low:.5f} {ci.high:.5f}")  # 0.00048 0.00184
print(f"{stats.laplace(scale=2).isf(0.05 / 2):.2f} {stats.laplace(scale=2).isf(0.05 / 1000 / 2):.2f}")  # 5.99 19.81
```

By the table's two Chernoff tails, a batch whose random size averages 600 strays more than a fifth from it with probability at most $2.4\times10^{-5}$.

A sum of $k$ Laplace draws has the far tail of a single draw. With scale $b$, its one-sided width is the larger of $\sqrt{8kb^2\ln(1/\beta)}$ and $4b\ln(1/\beta)$.

**McDiarmid's inequality** extends Hoeffding's to any function of independent inputs that no single input moves much. If $c_i$ is the most that changing input $i$ can move it, the chance of landing $t$ above its mean, or $t$ below, is at most
$$\exp\Big(-\frac{2t^2}{\sum_i c_i^2}\Big).$$
**Azuma's inequality** gives the same tail for a running total of steps that average zero given the past and lie in a range of length $c_i$.

## What goes wrong

Hoeffding needs bounded values, so clip unbounded data and use the clipping range as $R$. The bounds for sums, means and rates also need independent values, so with several rows per person, count people. A two-sided error bar must put $\beta/2$ on each side, not $\beta$.

The normal approximation is not a tail bound. For a sum of ten Laplace draws of scale 1, it understates the chance of a total above 20 about ninefold. When you pick the best of $m$ models on one validation set, the union bound asks for $\beta/m$, because the winner is the one most likely to look too good.

**Derivations.** Every result here except Bernstein's inequality is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/02-concentration.md).
