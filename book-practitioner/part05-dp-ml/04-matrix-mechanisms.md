# DP-FTRL and matrix factorization

## Why this matters

Stochastic gradient descent (SGD) with differential privacy (DP), the [DP-SGD](02-dp-sgd.md) of this part, owes much of its accuracy to **Poisson sampling**. Many pipelines cannot honestly claim that sampling, because they cut fixed batches, read data in arrival order or take whichever phones check in. Using the Poisson accountant anyway can understate the privacy loss $\varepsilon$ considerably ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md#sec-sub-schemes)).

Giving up amplification honestly is expensive. On this part's running example, five passes at $\varepsilon = 2$ on a fixed order need a noise multiplier of 4.458 instead of 0.928, and test accuracy falls from 0.9460 to 0.8941. This chapter wins that loss back by correlating the noise across steps. The first such method used the follow the regularized leader (FTRL) optimizer and is called DP-FTRL. Its modern form, the **matrix factorization mechanism**, closes nearly all of the gap.

## The matrix mechanism {#sec-mfac-mechanism}

Plain gradient descent subtracts each step's noisy sum of clipped gradients, scaled by the step size $\eta$ over the batch size $B$, from the starting weights $w_0$:
$$w_{t+1} = w_0 - \frac\eta B\hat S_t, \qquad \hat S_t = \hat x_1 + \dots + \hat x_t,$$ {#eq-mfac-weights}
where $\hat x_t$ is the noisy gradient sum of step $t$. A training run is therefore a release of **prefix sums**, and momentum or a learning-rate schedule only changes the fixed weights with which they are combined. Every model is computed from them, so it is post-processing and inherits their privacy ([Post-processing, group privacy and composition](../part03-dp-foundations/02-properties.md)).

Stack the true gradient sums as the rows of a matrix $x$. The prefix sums are then $Ax$, where $A$ has ones on and below the diagonal. The **matrix mechanism** splits $A$ into an encoder $R$ and a decoder $L$ with $A = LR$. It adds Gaussian noise $z$ to the encoded gradients and decodes the result:
$$\hat S = L(Rx + z) = Ax + Lz.$$ {#eq-mfac-mechanism}
Each entry of the noise $z$ has standard deviation $sC$, where $C$ is the clipping norm and the **noise scale** $s$ plays the role of DP-SGD's noise multiplier.

Privacy depends only on the columns of the encoder. A record used at one step shifts $Rx$ by that step's column of $R$ times its clipped gradient. So the encoder's sensitivity, in units of $C$, is the length of its longest column $r_j$. The mechanism is then a Gaussian mechanism whose ratio of sensitivity to noise is
$$\mu = \frac{\operatorname{sens}(R)}{s}, \qquad \operatorname{sens}(R) = \max_j\|r_j\|.$$ {#eq-mfac-privacy}
To calibrate, look up the largest ratio that meets your target $(\varepsilon, \delta)$, written $\mu(\varepsilon, \delta)$, with the exact Gaussian calibration of [f-DP and Gaussian DP](../part04-privacy-accounting/04-gaussian-dp.md). Then set $s = \operatorname{sens}(R)/\mu(\varepsilon, \delta)$. This chapter uses $\delta = 10^{-5}$ throughout, and at $\varepsilon = 2$ the ratio is 0.502.

Accuracy depends on the rows of the decoder. Write $\ell_t$ for the row that produces the $t$-th prefix sum. Each coordinate of its error has standard deviation $sC\|\ell_t\|$, which is $C/\mu$ times the **error per unit of privacy**
$$e_t = \operatorname{sens}(R)\,\|\ell_t\|.$$ {#eq-mfac-error}

Two conditions make all this valid during training. First, both factors must be lower-triangular, so that each step's release uses only gradients that already exist. Second, a record's slot in the data order must be fixed in advance, independently of the data. The sensitivity above also assumes neighbours that add or remove a record. If they replace a record instead, its gradient can change by $2C$, and every sensitivity in this chapter doubles.

In code, the mechanism is noisy SGD that adds the correlated noise $R^{-1}z$ to the gradient sums. You keep a buffer of past noise draws and never form a large matrix.

## Choosing a factorization {#sec-mfac-sqrt}

For a single pass, each factorization trades the encoder's column length against the decoder's row length. The numbers below are for one epoch of $T = 100$ steps.

**Independent noise**, the identity encoder, is DP-SGD without amplification. Its prefix-sum noise is a random walk, with a final error per unit of privacy of 10.

The **binary tree**, the encoder of the original DP-FTRL (Kairouz et al. 2021), adds noise to the sum over every node of a binary tree over the steps. Its error grows like $\log T$, with a largest error of 6.93 over one epoch.

The **square-root factorization** uses the same lower-triangular matrix for both factors, a square root of $A$. You can compute it with `scipy.linalg.sqrtm` on the prefix-sum matrix, or from the closed-form coefficients in the snippet below, which are constant along each diagonal. Over one epoch its sensitivity is 1.591, and its final error is 2.53, a quarter of independent noise's.

Optimized factorizations exist in research code (Denisov et al. 2022), but they cost time cubic in $T$ and gain little, so the free square root is the default.

## Several passes: banding and the data order {#sec-mfac-epochs}

Over $k$ passes a record enters the encoder through $k$ columns. When no two columns have a negative inner product, as for the identity, the tree and the square root, the worst case is a record that contributes the same full-length gradient each time. The sensitivity is the length of the sum of its columns, maximized over the participation patterns that the data order allows:
$$\operatorname{sens}(R) = \max_{\text{patterns}}\Big\|\sum_{i=1}^kr_{\tau_i}\Big\|,$$ {#eq-mfac-multi}
where $\tau_1, \dots, \tau_k$ are the steps of one pattern.

Repeated participation costs correlated noise more, because the square root's columns for nearby steps overlap, so its sensitivity grows faster than $\sqrt k$. **Banding** fixes this by keeping only the first $b$ diagonals of the square root. Suppose every two participations of a record are at least $b$ steps apart, a condition called **$b$-min-separation**. Then the columns no longer overlap, and the sensitivity is at most $\sqrt k$ times the longest column, as for independent noise. Its decoder rows are longer, so banding pays only when participations repeat.

The running example reuses one public random permutation every epoch, so each record takes part exactly 100 steps apart, which is the band used below.

| Epochs $E$ | Independent: sens / final $e_T$ | Square root | Banded square root |
| --- | --- | --- | --- |
| 1 | 1.000 / 10.00 | 1.591 / 2.53 | 1.591 / 2.53 |
| 10 | 3.162 / 100.00 | 9.154 / 16.54 | 5.031 / 15.69 |
| 20 | 4.472 / 200.00 | 17.191 / 32.09 | 7.115 / 29.85 |

: Sensitivity (in units of $C$) and error of the final prefix sum per unit of privacy, for $E$ participations 100 steps apart, $T = 100E$. {#tbl-mfac-epochs}

At 20 epochs, the banded sensitivity is exactly $\sqrt{20}$ times its one-epoch value, while the full square root's is more than twice as large. So the data order is part of the mechanism, and a permutation kept for every epoch guarantees a separation of exactly one epoch. Reshuffling every epoch can put a record at the end of one epoch and the start of the next, one step apart, and that breaks the banded bound.

## Which method when {#sec-mfac-run}

@tbl-mfac-run trains the network of [DP-SGD](02-dp-sgd.md) with each method at equal $\varepsilon$, with DP-SGD on Poisson sampling and the Rényi differential privacy accountant.

| $\varepsilon$, epochs | DP-SGD (Poisson) | Independent | Tree | Square root | Banded square root |
| --- | --- | --- | --- | --- | --- |
| 2, 1 | 0.9343 | 0.9066 | 0.9134 | 0.9361 | 0.9358 |
| 2, 10 | 0.9493 | 0.8641 | 0.9096 | 0.9398 | 0.9491 |
| 8, 1 | 0.9371 | 0.9395 | 0.9412 | 0.9461 | 0.9434 |
| 8, 10 | 0.9520 | 0.9410 | 0.9484 | 0.9530 | 0.9549 |

: Test accuracy (mean of three runs) at equal $\varepsilon$, $\delta = 10^{-5}$. {#tbl-mfac-run}

Three patterns guide the choice. Independent noise without amplification is a poor option, trailing DP-SGD by 8.5 points at $\varepsilon = 2$ and ten epochs, while the banded square root comes within 0.02 points. At large $\varepsilon$ and few epochs, amplification gains little, so correlated noise can even beat DP-SGD. At small $\varepsilon$ over many epochs, DP-SGD keeps its lead, and only banding keeps up.

You can choose before training anything, by computing each candidate's noise in the final prefix sum, $sC$ times the length of the decoder's last row. On the running example, the method with the least of it was the most accurate, or nearly so, in every configuration.

Federated learning is the main use, because the server cannot sample devices at a known rate. If a device contributes at most once every $b$ rounds, a banded mechanism with band at most $b$ gives a provable guarantee, and Google trained its Gboard keyboard language models this way with DP-FTRL (Xu et al. 2023).

You rarely write the mechanism yourself. Google's JAX Privacy library implements matrix-factorization noise for JAX training, including banded factorizations. TensorFlow Federated provides tree-aggregation noise for federated training through `tff.aggregators.DifferentiallyPrivateFactory.tree_aggregation`. Opacus assumes Poisson sampling, so it is the wrong tool here. Whatever the library, check that its sensitivity uses the participation pattern and the neighbouring relation you actually have.

## Code {#sec-mfac-code}

To check a library's configuration, the snippet computes a banded square root's sensitivity for your separation and number of epochs.

```python
from math import sqrt


def sqrt_coeffs(t: int) -> list[float]:
    r = [1.0]
    for k in range(1, t):
        r.append(r[-1] * (2 * k - 1) / (2 * k))
    return r


def toeplitz_sensitivity(c: list[float], t: int, sep: int, epochs: int) -> float:
    best = 0.0
    for offset in range(sep):
        steps = range(offset, t, sep)[:epochs]
        col = [sum(c[i - p] for p in steps if 0 <= i - p < len(c)) for i in range(t)]
        best = max(best, sqrt(sum(v * v for v in col)))
    return best


T = 100  # steps per epoch
band = sqrt_coeffs(T)  # the banded encoder keeps the first T coefficients
print(f"{toeplitz_sensitivity(band, T * 20, T, 20):.3f}")  # 7.115
print(f"{toeplitz_sensitivity(band, T * 10, T, 10):.3f}")  # 5.031
```

Divide the sensitivity by $\mu(\varepsilon, \delta)$ to get the noise scale.

## Cheat sheet

| Situation | Use | What to expect |
| --- | --- | --- |
| You can implement Poisson sampling | DP-SGD with its accountant | Strongest at small $\varepsilon$ and many epochs |
| Fixed order, one pass | Square-root factorization | Close to the optimized factorization, with nothing to optimize |
| Fixed order, several passes | Banded square root, band at most the separation | Sensitivity at most $\sqrt k$ times one column |

: Which mechanism for which pipeline. {#tbl-mfac-cheat}

**Rules of thumb**

- If you cannot implement Poisson sampling, do not account as if you did.
- Fix the data order before training, independently of the data, and keep the same permutation every epoch.
- Compare methods by the noise in the final prefix sum at equal $\varepsilon$ and the same number of epochs.
- Never claim amplification on top of a matrix mechanism unless its analysis allows it, as the banded analysis of Choquette-Choo et al. (2023) does.
- Never reuse the single-participation sensitivity after reshuffling every epoch.

## Further reading

- Denisov, McMahan, Rush, Smith & Thakurta, "Improved Differential Privacy for SGD via Optimal Private Linear Operators on Adaptive Streams" (NeurIPS 2022): the mechanism for training.
- Choquette-Choo, Ganesh, McKenna, McMahan, Rush, Thakurta & Xu, "(Amplified) Banded Matrix Factorizations: A Unified Approach to Private Training" (NeurIPS 2023): banding and min-separation.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/04-matrix-mechanisms.md).
