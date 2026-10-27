# Concentration inequalities

## What it is

A concentration inequality bounds the chance that a random quantity lands more than some distance from its mean. Call that event a failure, its chance the **failure probability** $\beta$, and the distance the **width**. With a fixed $\beta$, the bound gives an error bar, and with a fixed width, a sample size.

**Markov's inequality** needs only the mean of a quantity that cannot be negative. It caps the chance to reach a threshold at the mean divided by the threshold. **Chebyshev's inequality** also uses the variance $\sigma^2$. Write $\mu$ for the mean. Then the chance that $X$ lands a distance $t$ or more from it is at most
$$\Pr[|X - \mu| \ge t] \le \frac{\sigma^2}{t^2}.$$
Make this equal to $\beta$ to get the width $\sigma/\sqrt\beta$, which holds for every distribution at a high price. At one in a million, it asks for 1,000 standard deviations, where a normal tail needs 5.3.

## When you meet it

You meet these bounds when you put an error bar on a mean or a rate, plan a sample size, or report many numbers at once.

## How to use it

Pick the bound by what you know about the quantity (@tbl-conc-choose).

| What you know | Use | Width at failure probability $\beta$ |
| --- | --- | --- |
| The variance $\sigma^2$ | Chebyshev | $\sigma/\sqrt\beta$ |
| A mean, with standard deviation $s$ estimated from the data | Normal approximation | $z_{1-\beta/2}\,s/\sqrt n$, smaller but not a guarantee; solve for $n$ to plan |
| The exact noise distribution ([Probability distributions](01-distributions.md)) | Its quantile | Laplace $b\ln(1/\beta)$; normal $\sigma z_{1-\beta/2}$ |
| A count with mean $\mu$ | Chernoff | a miss by a fraction $r$ has chance $\le e^{-r^2\mu/(2+r)}$ above, $e^{-r^2\mu/2}$ below |
| The mean of $n$ values in a range of length $R$ | Hoeffding | $R\sqrt{\ln(2/\beta)/(2n)}$ |
| A rate: $k$ errors out of $n$ | Clopper–Pearson | exact binomial bound |

: Which bound to use, and the two-sided width it gives. {#tbl-conc-choose}

**Hoeffding's inequality** is the planning tool, because it needs only the range of the values. Solve its width for $n$ to get the sample size for a width $t$:
$$n \ge \frac{R^2\ln(2/\beta)}{2t^2}.$$
Accuracy averages zeros and ones, so $R = 1$, and a width of one percentage point at 95% needs 18,445 test records.

For a small rate, Hoeffding is too wide, because it takes the largest variance that the range allows. For one false positive in 1,000 negatives, its one-sided 95% bound adds 0.039 to the measured rate. **Bernstein's inequality** uses the variance and adds only 0.0045, but it is a heuristic with a variance from the same data. The exact **Clopper–Pearson** bound uses the binomial distribution itself and gives 0.0047, or 0.0030 with no errors.

For noise of a known distribution, use its quantile, so Laplace noise of scale 2 lies within 5.99 of zero 95% of the time. For many draws at once, the **union bound** says that the chance of one or more failures is at most the sum of the chances. So give each of $k$ draws $\beta/k$. The Laplace width grows only with $\ln(k/\beta)$, so for a thousand draws, 5.99 grows only to 19.81. The snippet prints these numbers, with the one-sided bound for $k$ errors in $n$ as `stats.beta.ppf(0.95, k + 1, n - k)`.

```python
import math

from scipy import stats

print(math.ceil(math.log(2 / 0.05) / (2 * 0.01**2)))  # 18445
print(f"{stats.beta.ppf(0.95, 1, 1000):.4f} {stats.beta.ppf(0.95, 2, 999):.4f}")  # 0.0030 0.0047
ci = stats.binomtest(10, 10_000).proportion_ci(confidence_level=0.95, method="exact")
print(f"{ci.low:.5f} {ci.high:.5f}")  # 0.00048 0.00184
print(f"{stats.laplace(scale=2).isf(0.05 / 2):.2f} {stats.laplace(scale=2).isf(0.05 / 1000 / 2):.2f}")  # 5.99 19.81
```

By the Chernoff tails, a sampled batch of average size 600 strays more than a fifth from it with probability at most $2.4\times10^{-5}$.

**McDiarmid's inequality** extends Hoeffding's from means to other functions of independent inputs. Suppose a change to the $i$-th input moves the function by at most an amount $c_i$. Then the chance that it lands $t$ or more above its mean, or below, is at most
$$\exp\Big(-\frac{2t^2}{\sum_i c_i^2}\Big).$$
**Azuma's inequality** gives the same tail for a running total of steps, each with average zero given the past.

## What goes wrong

Hoeffding needs bounded values, so clip unbounded data. The range of the clipped values is then $R$. The bounds also need independent values, so count people, not rows. And a two-sided error bar must put $\beta/2$ on each side, not $\beta$.

The normal approximation is not a tail bound. For a sum of ten Laplace draws of scale 1, it puts the chance of a total above 20 about nine times too low. And when you pick the best of $m$ models on one validation set, give each model $\beta/m$, because the winner is the one most likely to look too good.

**Derivations.** Every result here except Bernstein's inequality is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/02-concentration.md).
