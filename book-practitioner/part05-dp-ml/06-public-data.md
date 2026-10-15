# Public data and private fine-tuning

## Why this matters

The earlier chapters of this part trained models from random weights with differential privacy (DP), using stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, called DP-SGD ([DP-SGD](02-dp-sgd.md)). On the synthetic task that this part uses as its running example, that costs little accuracy at a privacy budget of $\varepsilon = 8$ but a great deal at $\varepsilon = 1$ and below. The most accurate private models are built differently. They start from a model **pretrained** without privacy on a large public dataset, such as text or images from the web, and only **fine-tune** it privately on the sensitive data. At the same $\varepsilon$, such models are far more accurate than the same models trained privately from scratch. Tramèr & Boneh (2021), Yu et al. (2022), Li et al. (2022) and De et al. (2022) all report this.

A second practice helps on top of the first. Fine-tuning only a small part of the model often beats fine-tuning all of it under DP. That small part can be the last layer, or a low-rank update to each weight matrix, a correction built from a few vectors (@sec-pub-lora). Without privacy it would usually be the other way round, so the explanation has to lie in the noise.

DP-SGD adds noise of the same size to every weight it trains, so the noise fills every direction in which the weights can move. The useful part of the gradient, by contrast, has a bounded length, because clipping caps each record's contribution, and it usually points in a few directions. So a weight that the task does not need only collects noise. Public data therefore help twice. They move the starting point close to a good model, and they can show, at no privacy cost, which few directions the remaining learning lies in. What they cannot do is protect the people whose records are in them, and the last section explains why.

## Why dimension costs, and when it does not {#sec-pub-dimension}

Recall that a step of DP-SGD clips each sampled record's gradient to length at most $C$, the clipping norm ([DP-SGD](02-dp-sgd.md#sec-dpsgd-algorithm)). It then adds Gaussian noise with standard deviation $\sigma C$ to every coordinate of the sum, where $\sigma$ is the noise multiplier. The privacy argument needs only that one record changes the sum by at most $C$. It never asks which weights the gradient belongs to, so the choice of trainable weights leaves $\varepsilon$ unchanged and changes only how much noise lands on the model.

Using public data is free for a related reason. DP limits how much the output can change when one *private* record changes. So anything computed from public data alone is a fixed input as far as the private data are concerned. The DP-SGD proof asks only that the starting weights be chosen independently of the private data. Weights chosen by looking at the private data would leak information that no noise covers. The same condition applies to everything else this chapter takes from public data, such as a subspace or a random starting value.

The noise grows with the number of weights you train. Write $k$ for that number, and $B$ for the expected batch size, the public constant that DP-SGD divides the noisy sum by. Each trainable coordinate of the averaged gradient then carries noise with standard deviation $s$, which equals $\sigma C/B$. The squared length of the noise is a sum of $k$ squared coordinates, each with average $s^2$. So its length is about $s\sqrt k$ ([Concentration inequalities](../part01-prerequisites/02-concentration.md)):
$$\text{noise length} \approx s\sqrt k = \frac{\sigma C\sqrt k}{B}.$$ {#eq-pub-noise-length}

The signal does not grow in the same way. It is the average clipped gradient over the private records, which is at most $C$ long whatever $k$ is. It can even be longer when $k$ is smaller. A record's gradient restricted to fewer coordinates is shorter, so it is clipped less often and less of it is cut away.

The running example shows both effects. Its private task has 50-dimensional inputs labelled by a fixed random network, the teacher, and the number of private training records is $n = 10{,}000$. The network being trained, the student, has 64 hidden units on 51 inputs. With its 65-weight last layer, the number of weights is $p = 3{,}329$ in all. It is first pretrained on a related public task (@sec-pub-run).

Fine-tuning then runs DP-SGD with clipping norm $C = 0.5$ and a sampling rate of 0.01. The expected batch size is therefore $B = 100$. Training runs for $T = 10{,}000$ steps, with $\delta = 10^{-5}$. @tbl-pub-noise compares three choices of trainable weights at four budgets, each with the noise multiplier that the accountant gives. The three are all 3,329 weights, a rank-4 adapter on the first layer with the last layer (@sec-pub-lora), and the last layer alone.

| $\varepsilon$ | $\sigma$ | All weights, $k = 3{,}329$ | Rank-4 adapter and last layer, $k = 525$ | Last layer, $k = 65$ |
| --- | --- | --- | --- | --- |
| 0.5 | 7.72 | 2.227 | 0.884 | 0.311 |
| 1 | 4.13 | 1.191 | 0.473 | 0.166 |
| 2 | 2.28 | 0.658 | 0.261 | 0.092 |
| 8 | 0.917 | 0.265 | 0.105 | 0.037 |

: Length of the noise on the averaged gradient, $\sigma C\sqrt k/B$ with $B = 100$ and $C = 0.5$, for three numbers of trainable weights. {#tbl-pub-noise}

Along each row the noise shrinks in proportion to $\sqrt k$ as fewer weights are trained, while the signal moves the other way. From the pretrained weights, the average clipped gradient is 0.0130 long when all weights are trained. It is 0.0207 for the adapter, and 0.0250 for the last layer alone (@fig-pub-noise-signal). At $\varepsilon = 1$ the noise is therefore 92.0 times the signal for all weights. The ratio falls to 22.9 for the adapter and to 6.7 for the last layer.

![Left: at $\varepsilon = 1$ ($\sigma = 4.13$, $B = 100$, $C = 0.5$), the noise length $\sigma C\sqrt k/B$ (line) against the number of trainable weights $k$, and the length of the average clipped gradient at the pretrained model (points) for the last layer, low-rank adapters of ranks 1 to 16, and all weights. Right: their ratio. The noise grows like $\sqrt k$ while the signal shrinks, because longer gradient vectors are clipped more.](figures/pub_noise_signal.svg){#fig-pub-noise-signal}

These ratios say how long training must run before the signal shows through the noise. While the average gradient keeps roughly the same direction, it adds up step after step, but independent noise adds up only like a random walk. So the signal outgrows the noise after about $(\text{noise}/\text{signal})^2$ steps ([DP-SGD](02-dp-sgd.md#sec-dpsgd-algorithm)). That is 8,455 steps for all weights and 523 for the adapter. For the last layer it is only 44 steps.

By this count alone, a model with many weights should be very expensive to fine-tune privately. Yet language models with hundreds of millions of weights are fine-tuned privately with good results. The rates of private empirical risk minimization (ERM) grow with the dimension ([Private empirical risk minimization](01-dp-erm.md#sec-dperm-compare)). But they are worst cases over every loss with a given Lipschitz constant, a bound on how long any one gradient can be. Li, Liu et al. (2022) replace the dimension by a measure of how the gradients spread over directions, because noise hurts only in the directions the loss cares about.

To see how much each direction costs, model the loss near the starting point as a quadratic bowl. Its shape is described by the curvature matrix $H$, whose eigenvectors are the bowl's principal directions. Each principal direction has a **curvature**, the matching eigenvalue, which says how fast the loss rises as the weights move along it.

Noise pushes the weights around in every trainable direction, and the curvature pulls them back. In a curved direction, the pull soon balances the pushes, and the weights settle into a steady spread around the bottom. That spread costs the same amount of loss whatever the curvature, because a stiffer direction wanders less but charges more for each unit of wandering. In a nearly flat direction, almost nothing pulls the weights back, so the noise keeps adding up as a random walk. Yet the walk costs little, because moving along a flat direction barely changes the loss.

Adding these costs over the directions gives the loss increase that the noise causes, compared with the same $T$ steps without noise. Write $\eta$ for the step size. Number the principal directions of the trainable weights, and write $h_j$ for the curvature of the $j$-th one. Then the expected loss rises by about
$$\Delta \approx \frac{\eta s^2}{4}\,d_{\text{eff}}, \qquad d_{\text{eff}} = \sum_{j=1}^k\Big(1 - (1 - \eta h_j)^{2T}\Big).$$ {#eq-pub-noise-cost}
The approximation assumes that the step size times each curvature is much smaller than 1. A direction with curvature $h$ needs about $1/(2\eta h)$ steps to reach its steady state. So its term is close to 1 when $h$ is well above $1/(2\eta T)$, and close to 0 when $h$ is well below that threshold. The **effective dimension** $d_{\text{eff}}$ therefore roughly counts the directions that the noise has time to fill.

Each term is at most 1, so $d_{\text{eff}}$ is never more than $k$. Each term is also at most $2\eta T h_j$, which gives a second bound,
$$d_{\text{eff}} \le \min\big(k,\ 2\eta T\operatorname{tr}H\big),$$ {#eq-pub-deff-bound}
where the trace $\operatorname{tr}H$ is the total curvature, the sum of the eigenvalues. On the running example a few large curvatures dominate the trace, and that second bound lies far above $k$.

On the running example, the curvature at the pretrained weights was estimated from 2,000 private records. Full fine-tuning uses the step size $\eta = 0.1$ for its 10,000 steps. That puts the threshold $1/(2\eta T)$ at $5 \times 10^{-4}$. The left panel of @fig-pub-subspace shows the curvatures falling steeply from 0.723. Only 406 of the 3,329 directions lie above the threshold, the dashed line. Directions just below it still add part of a term each, so the effective dimension for all weights is a little larger, 451. For the last layer alone it is 60.2 out of 65.

Those numbers compare the two choices. Full fine-tuning's noise is 7.2 times longer than the last layer's, so it has 51 times the squared length, which is also the ratio of the weight counts. But by @eq-pub-noise-cost it costs only about 7.5 times more, the ratio of the effective dimensions.

Fewer trainable weights still help, for two reasons. Freezing weights removes their directions from the sum, so the effective dimension can only fall. And shorter per-example gradients are clipped less, so the signal can be longer, as @fig-pub-noise-signal showed. Against both stands the **bias** of freezing. If the task needs a direction that is frozen, no amount of private data can move the model along it.

The step size and the number of steps also set the effective dimension, because a longer or more aggressive run fills more flat directions with noise. That is one reason private fine-tuning works best with smaller step sizes than training from scratch (@sec-pub-run). Treat @eq-pub-noise-cost as a guide for comparing strategies, not as a prediction of accuracy. The quadratic holds only near the starting point, and a rise in loss changes accuracy only when the noise flips predictions.

## Good features: the last layer as a convex problem {#sec-pub-features}

The simplest way to use a pretrained model is to freeze everything except the last layer, and train a linear classifier on the frozen features. This is called **linear probing**. On the running example it is logistic regression on the 64 hidden activations plus a constant, which makes 65 weights. Tramèr & Boneh (2021) showed that it beats end-to-end private training on standard image benchmarks at moderate budgets, even with handcrafted features.

Linear probing turns private deep learning back into the convex problem of [Private empirical risk minimization](01-dp-erm.md), where the methods and their rates are known. This needs the features to be computed from public data only, the condition of @sec-pub-dimension. Then they are a fixed transformation of each record, and the probe is ordinary private convex ERM on the transformed records.

The logistic loss also makes clipping optional for a probe. A record's gradient is its feature vector times a factor smaller than 1 in size. The student's hidden units use the tanh activation, so each feature lies strictly between $-1$ and 1, and the constant feature is exactly one. The gradient is therefore shorter than $\sqrt k$, which makes the loss $\sqrt k$-Lipschitz.

Clipping below that bound can still pay. The bound is reached only by a badly misclassified record whose activations are all saturated, while most records are classified with a wide margin and have tiny gradients. Clipping at a smaller $C$ biases those few long gradients, but it divides the noise by $\sqrt k/C$.

The error of a private probe splits into two parts. Write $R$ for the risk, the expected loss on new data, and $R_{\text{best}}$ for the best risk of any model of any architecture. Write $w^\phi$ for the best linear model on the features, and $\tilde w$ for the private probe. Then
$$R(\tilde w) - R_{\text{best}} = \underbrace{R(w^\phi) - R_{\text{best}}}_{\text{features}} + \underbrace{R(\tilde w) - R(w^\phi)}_{\text{privacy}}.$$ {#eq-pub-probe-split}
The first part is the **feature gap**. It compares two non-private models, so it does not depend on the number of records or on $\varepsilon$, and it is paid even with unlimited data and no privacy. The second part is the **privacy cost**, the cost of private convex ERM in $k$ dimensions.

Like the rates of [Private empirical risk minimization](01-dp-erm.md), the privacy cost assumes a Lipschitz constant $L$, which is at most $\sqrt k$ for the probe. It also assumes weights restricted to a ball of radius $R_w$. The best achievable privacy cost then has the order
$$\tilde\Theta\Big(\frac{LR_w\sqrt{k\ln(1/\delta)}}{n\varepsilon}\Big),$$
where the tilde hides logarithmic factors (Bassily, Smith & Thakurta 2014). Noisy gradient descent attains this rate.

So the features cap the accuracy, and privacy only decides how close the probe gets to the cap. With few poor features, the probe is private and cheap but bad, which is the point of Tramèr & Boneh's title: private learning needs better features or much more data. End-to-end training can eventually beat a probe because it learns better features. But it pays for every one of its weights, and at small $\varepsilon$ and small $n$ it does not catch up.

This makes linear probing the safe first baseline. The problem is convex, so the optimizer cannot wander into a bad region, and it has few hyperparameters. If a probe on public features already meets your accuracy target, there is little reason to fine-tune more.

On the running example, the pretrained network labels 0.9035 of the private test inputs correctly before seeing any private data. Fitting the best last layer exactly on the private records, without noise, raises this to the probe's ceiling of 0.9536. A network trained non-privately from scratch, standing in for the best model, reaches 0.9606, so the feature gap is 0.007. The private probe stays within 0.006 of its ceiling at every budget (@tbl-pub-run), and at no budget can it pass it.

## Low-rank adapters {#sec-pub-lora}

The last layer is often not enough, because the features of a model pretrained on one task are rarely exactly right for another. The probe then stops at its ceiling, while training all weights fixes the features but pays noise for every weight. **Low-rank adaptation (LoRA)** sits in between (Hu et al. 2022).

LoRA freezes each pretrained weight matrix $W_0$, which has $d_{\text{in}}$ inputs and $d_{\text{out}}$ outputs. On top of it, LoRA learns a correction whose rank, written $r$, is small. The adapted matrix is $W = W_0 + UV^\top$. The factor $U$ has $r$ columns as long as the layer's output, while the other factor, $V$, has as many columns as long as its input. Training starts with $V$ random and $U = 0$, so that the fine-tuned model begins exactly at the pretrained one. Counting the entries of the two factors gives
$$k = r(d_{\text{in}} + d_{\text{out}})$$ {#eq-pub-lora-count}
trainable weights, against $d_{\text{in}}d_{\text{out}}$ for the whole matrix.

The guarantee is again that of DP-SGD, provided the random starting value of $V$ is drawn independently of the private data. Yu et al. (2022) showed that private fine-tuning of language models with LoRA and similar parameter-efficient methods matches or beats private full fine-tuning, at a fraction of the cost.

Adapters also keep clipping cheap. For a dense layer with input $a$, one record's gradient with respect to the weight matrix is the outer product of $a$ with a vector $g$. That vector is the gradient of the loss with respect to the layer's output, which [DP-SGD](02-dp-sgd.md#sec-dpsgd-ghost) calls the record's error. Ghost clipping uses the fact that an outer product's length is the product of the two vectors' lengths. The chain rule through $UV^\top$ gives the adapter's gradients the same form:
$$\nabla_U\ell = g\,(V^\top a)^\top, \qquad \nabla_V\ell = a\,(U^\top g)^\top.$$ {#eq-pub-lora-grad}
Each is again an outer product, so ghost clipping applies, and no per-example matrix is ever formed.

The restriction does cost something. At the start $U = 0$, so by @eq-pub-lora-grad the gradient with respect to $V$ is zero, and only $U$ learns in the first step. The change to $W$ is then the change in $U$ times, to first order, the transpose of the still-random $V$. So the true gradient is roughly projected onto a random $r$-dimensional subspace of the layer's inputs, which keeps only a fraction of about $r/d_{\text{in}}$ of its squared length. Once $U$ is no longer zero, $V$ learns too, and the subspace moves towards the useful directions.

Low-rank adaptation therefore works when the change that the new task needs is itself close to low rank, as Hu et al. (2022) observed for language models. The rank $r$ trades bias against noise, just as the number of trainable weights does. And an adapter saves noise only at small rank, because at full rank it has more coordinates than $W$ itself.

Full fine-tuning is not ruled out. Li, Tramèr et al. (2022) fine-tuned all the weights of pretrained language models with DP-SGD and matched or beat the parameter-efficient methods. They needed large batches, a step size tuned for private training, and a fine-tuning objective that resembles the pretraining one. That fits the effective dimension, since full fine-tuning pays for $d_{\text{eff}}$ directions, not for all $p$ weights. The published evidence goes both ways, but the adapter keeps two advantages either way. It stores only $k$ numbers per task, and its gradients are much smaller to compute and clip.

On the running example, the adapter sits on the first layer, which maps 51 inputs to 64 hidden units. A rank-4 adapter trains $4 \times (51 + 64) = 460$ of that layer's 3,264 weights. Then adding the last layer brings the total to the $k = 525$ of @tbl-pub-noise. Larger models save more. For example, on a transformer's $768 \times 768$ projection matrix, a rank-8 adapter trains only 12,288 weights. That is 48 times fewer than the whole matrix, so its noise is 6.9 times shorter.

## Gradients near a public subspace {#sec-pub-subspace}

The effective dimension says that noise costs little in directions the loss does not use. So if public data can estimate the directions the private gradients do use, noise can be added only there. Compute per-example gradients on public records, find their top principal directions, and project the private gradients onto the first $m$ of them. Zhou, Wu & Banerjee (2021) do this as **projected DP-SGD**. Li, Liu et al. (2022) measured that the gradients of private fine-tuning are indeed dominated by a few principal directions.

Write $P_m$ for the projection onto the $m$-dimensional subspace. Each step releases the projected sum of clipped gradients, with Gaussian noise of standard deviation $\sigma C$ along each of the $m$ directions of the subspace:
$$P_m\Big(\sum_{i \in S_t}\bar g_i\Big) + \mathcal N(0, \sigma^2C^2P_m).$$ {#eq-pub-proj}
Here $S_t$ is the batch sampled at step $t$. Each record $i$ in it contributes its clipped gradient $\bar g_i$. Because a projection never lengthens a vector, one record still moves the sum by at most $C$, and the guarantee is that of DP-SGD. The subspace must come from public data, the condition of @sec-pub-dimension, since one computed from the private gradients would itself leak information about them.

The projection trades bias for noise. It acts on all $p$ weights, but the noise now has only $m$ coordinates. So its length falls by a factor of $\sqrt{p/m}$. The signal loses whatever part of it lies outside the subspace. Write $\bar G$ for the average clipped gradient, and $f_m$ for the fraction of its squared length, its energy, that lies in the subspace. The kept signal then has length $\sqrt{f_m}$ times that of $\bar G$, and by the random-walk argument of @sec-pub-dimension it outgrows the noise after about
$$T_{\text{even}}(m) = \Big(\frac{s\sqrt m}{\sqrt{f_m}\,\|\bar G\|}\Big)^2$$ {#eq-pub-break-even}
steps. The lost fraction $1 - f_m$ of the signal is a bias that no number of steps recovers.

How good the trade is depends on the public data, which set $f_m$. If the private gradients lie near the public subspace, $f_m$ is close to 1 even for small $m$, and the trade is excellent. Public data unrelated to the private task give a subspace no better than a random one, which captures a fraction of about $m/p$ of the energy. The projection then removes as much signal as noise and gains nothing.

A fixed subspace caps what training can learn, since it removes the same part of the signal at every step. Projected DP-SGD therefore recomputes the subspace from public gradients as training proceeds. **Gradient embedding perturbation** (Yu, Zhang, Chen & Liu 2021) instead releases the residual outside the subspace with its own noise, which removes the bias at a small cost in noise. LoRA, by contrast, picks its subspace by structure and lets training move it.

The same concentration explains why full fine-tuning can work. On a quadratic, the gradient is the curvature matrix times the distance to the minimum, so it lies mostly along the directions of large curvature. Gradients near a low-dimensional subspace thus go with a small effective dimension. Projection makes the saving explicit, and full fine-tuning gets part of it implicitly.

@tbl-pub-subspace measures this on the running example, at the pretrained weights. The public directions are the top principal directions of 500 clipped public per-example gradients. The table shows how much of the private gradient's energy they hold, and how much random subspaces of the same dimension hold.

| Dimension $m$ | 1 | 4 | 16 | 64 | 256 | 3,329 (all) |
| --- | --- | --- | --- | --- | --- | --- |
| Public top-$m$ directions, $f_m$ | 0.596 | 0.623 | 0.655 | 0.732 | 0.887 | 1 |
| $m$ random directions | 0.001 | 0.001 | 0.004 | 0.022 | 0.080 | 1 |
| Break-even steps at $\varepsilon = 1$, @eq-pub-break-even | 4 | 16 | 62 | 222 | 733 | 8,455 |

: How much of the private average clipped gradient lies in public subspaces, at the pretrained weights, and the steps after which the kept signal outgrows the noise at $\varepsilon = 1$ ($s = 0.0206$). {#tbl-pub-subspace}

A single public direction already holds 60% of the private gradient's energy. Sixty-four directions, only 2% of the dimensions, hold 73%. As a result, the kept signal in that subspace outgrows the noise after 222 steps at $\varepsilon = 1$. With all weights it takes 8,455 steps. The price is the 27% of the energy outside the subspace, which a fixed projection never learns.

![Left: eigenvalues of the Gauss–Newton curvature of the private loss at the pretrained weights, estimated from 2,000 private records, on log scales, with the thresholds $1/(2\eta T)$ for $\eta = 0.1$ and $0.5$ and $T = 10{,}000$; directions below a threshold contribute little noise cost. Right: the fraction of the energy of the private average clipped gradient that lies in the top $m$ principal directions of 500 public per-example gradients (solid), and in $m$ random directions (dashed).](figures/pub_subspace.svg){#fig-pub-subspace}

## Pretraining and fine-tuning the running example {#sec-pub-run}

The public task was built so that pretraining helps without solving the private task. Its teacher is a slightly altered copy of the private teacher. It labels 50,000 inputs drawn from the private input distribution, and it agrees with the private teacher on 0.9004 of the private test inputs. The student is pretrained on these public data by non-private SGD for 10,000 steps, after which it labels 0.9035 of the private test inputs correctly without seeing any private data.

From the pretrained weights, DP-SGD fine-tunes in three ways. *Full* fine-tuning trains all 3,329 weights, while *last layer* trains only the 65 weights of the last layer. In between, *adapter* trains a rank-4 adapter on the first layer together with the last layer, 525 weights in all. The baseline, *scratch*, trains all weights from a random initialization, as in [DP-SGD](02-dp-sgd.md). Because all four use the same sampling rate and number of steps, each budget needs the same noise multiplier for every method.

The private test set of 20,000 records is split in half. Each method's step size is chosen at $\varepsilon = 1$ on the first half (@tbl-pub-lr), and accuracy is reported on the second. That choice looks at private data without paying for it, which [Hyperparameter tuning and private selection](03-hyperparameter-tuning.md) shows how to fix, so the comparison is of methods at their best.

| Method | Step sizes tried, accuracy on the validation half | Chosen |
| --- | --- | --- |
| Scratch | 0.25: 0.9318; 0.5: 0.9303; 1: 0.8582 | 0.25 |
| Full | 0.03: 0.9559; 0.1: 0.9567; 0.3: 0.9433 | 0.1 |
| Last layer | 0.1: 0.9479; 0.3: 0.9499; 1: 0.9460 | 0.3 |
| Adapter, rank 4 | 0.03: 0.9556; 0.1: 0.9619; 0.3: 0.9472 | 0.1 |

: Step-size selection at $\varepsilon = 1$, one run each, on the validation half of the private test set. {#tbl-pub-lr}

| | Scratch | Full | Last layer | Adapter, rank 4 |
| --- | --- | --- | --- | --- |
| Trainable weights $k$ | 3,329 | 3,329 | 65 | 525 |
| $\varepsilon = 0.5$ | $0.8589 \pm 0.0075$ | $0.9395 \pm 0.0006$ | $0.9478 \pm 0.0010$ | $0.9464 \pm 0.0005$ |
| $\varepsilon = 1$ | $0.9246 \pm 0.0050$ | $0.9599 \pm 0.0013$ | $0.9497 \pm 0.0000$ | $0.9616 \pm 0.0001$ |
| $\varepsilon = 2$ | $0.9518 \pm 0.0010$ | $0.9663 \pm 0.0001$ | $0.9519 \pm 0.0005$ | $0.9653 \pm 0.0002$ |
| $\varepsilon = 8$ | $0.9611 \pm 0.0012$ | $0.9677 \pm 0.0001$ | $0.9514 \pm 0.0005$ | $0.9655 \pm 0.0001$ |
| Non-private | 0.9606 | 0.9704 | 0.9536 | — |

: Test accuracy on the report half of the private test set, mean and standard deviation over two runs, $\delta = 10^{-5}$. The pretrained model without fine-tuning has 0.9035. Non-private: SGD for 1,000 steps from scratch and from the pretrained weights, and the exact logistic regression on the frozen features. {#tbl-pub-run}

On this task, pretraining is worth about a factor of eight in $\varepsilon$. The adapter at $\varepsilon = 1$ matches training from scratch at $\varepsilon = 8$. It also matches the non-private model trained from scratch, and full fine-tuning at $\varepsilon = 2$ beats both. Even at $\varepsilon = 0.5$, every fine-tuning method beats training from scratch at $\varepsilon = 1$ (@tbl-pub-run). The public data carry information about the private task that 10,000 private records do not, and that information is free.

Which weights to train depends on the budget. At $\varepsilon = 0.5$ the last layer is best, because its noise is 7.2 times shorter than full fine-tuning's and its ceiling is not yet binding. From $\varepsilon = 1$ on, it is stuck near that ceiling, while the adapter and full fine-tuning keep improving. The noise is now small enough to let them correct the features. Full fine-tuning trails by 0.0083 at $\varepsilon = 0.5$. But with the right step size it catches up, and at the two largest budgets it is the best method, as Li, Tramèr et al. (2022) found for language models. The adapter is never more than 0.0022 below the best method, so the middle choice is the robust one.

The step size matters as much as the method. In @tbl-pub-lr, a step size three times too large costs full fine-tuning 1.3 percentage points of accuracy, and the adapter 1.5 points. Their best step size is 2.5 times smaller than the one for training from scratch, as the effective-dimension argument predicts.

Do not read small differences between the pretrained methods as rankings, because two runs estimate the spread only roughly. The rank sweep below repeats the adapter at $\varepsilon = 1$ and finds a standard deviation of 0.0023, larger than the table suggests. And @fig-pub-accuracy, a separate set of single runs, even orders the three pretrained methods differently at the two smallest budgets.

![Test accuracy of the four strategies against $\varepsilon$ at $\delta = 10^{-5}$, one run each with the step sizes of @tbl-pub-lr and a different seed from @tbl-pub-run, with the pretrained model's accuracy before fine-tuning (dotted). The pretrained methods are far ahead of training from scratch at small budgets; the last layer stays near its ceiling from $\varepsilon = 1$ on, and training from scratch passes it at larger budgets.](figures/pub_accuracy.svg){#fig-pub-accuracy}

The adapter's rank is the last choice to make. @tbl-pub-rank repeats the adapter at $\varepsilon = 1$ with three ranks, keeping its step size of 0.1.

| Rank $r$ | Trainable weights $k$ | Test accuracy |
| --- | --- | --- |
| 1 | 180 | $0.9590 \pm 0.0038$ |
| 4 | 525 | $0.9624 \pm 0.0023$ |
| 16 | 1,905 | $0.9586 \pm 0.0031$ |

: The adapter's rank at $\varepsilon = 1$, mean and standard deviation over two runs. {#tbl-pub-rank}

Rank 4 is best, but the differences are about the size of the run-to-run spread. A higher rank lets the update follow more of the directions the task needs, but it also adds noise. On this problem neither force dominates over a sixteenfold range of ranks, so the rank is a cheap hyperparameter to get roughly right.

## What public data does not buy {#sec-pub-caveats}

Everything so far treated the public data as free. Formally that is right, but "public" is a label someone chose, and the guarantee says nothing about the records under it. Tramèr, Kamath & Carlini (2024) argue that the label "differentially private" should therefore be used with care for models pretrained on web-scraped data.

Here is exactly what is guaranteed. For every fixed public dataset, the map from the private dataset to the fine-tuned model is $(\varepsilon, \delta)$-DP, by the condition of @sec-pub-dimension. With respect to the public dataset, there is no guarantee of any kind. Non-private pretraining can memorize individual records ([Memorization and training-data extraction](../part02-attacks/05-memorization-extraction.md)), and the fine-tuned model inherits whatever the pretrained weights encode.

This matters most for a person whose record is in both datasets. DP compares datasets that differ in the private copy only, with the public data the same in both, so only the private copy is protected. Consider any test of whether the record was in the private set, at $\varepsilon = 1$ and $\delta = 10^{-5}$. If its false positive rate is 0.01, its true positive rate is at most $e \times 0.01 + 10^{-5} = 0.0272$ ([The definition of differential privacy](../part03-dp-foundations/01-definition.md)). A membership test against the pretrained model, such as the likelihood-ratio attack of [Membership inference](../part02-attacks/02-membership-inference.md), has no such bound. It may still fail in practice, but no theorem says it will. Even a record that appears twice in the private set is better off. For the pair, group privacy still gives $(2\varepsilon, 2e^\varepsilon\delta)$ ([Post-processing, group privacy and composition](../part03-dp-foundations/02-properties.md#sec-dpprop-group)). Here that is $(2, 5.44 \times 10^{-5})$, weaker but real.

Such overlap is a practical risk, because web-scale "public" datasets contain personal data, such as posts, profiles and leaked documents. A private set of medical notes or emails may overlap with them, and the people in both are not protected at all. Deduplicating the private set against the public data is the minimum check, but it protects copies, not people. Exact matching misses paraphrases, and records about the same person need not be duplicates at all.

Public data also help only as much as they resemble the private task. Some benchmarks use a "private" set whose distribution is close to well-curated public data, such as two collections of natural images. They can overstate what the method will do on genuinely sensitive data, which is often unlike anything public: medical images, rare languages, internal logs. The running example used a public task close to the private one by design. When the public task is made less related, the last layer's gain shrinks, until it does worse than training from scratch.

Public is not the same as non-sensitive either. Even when every pretraining record is public, a model trained on them can reproduce them, link them, or make inferences about people that the original publication did not allow. And large pretrained models are often too large to fine-tune locally, so the private data are sent to whoever has the compute. That is a question of trust, which DP does not address ([Local, shuffle and distributed trust models](../part03-dp-foundations/08-trust-models.md)).

When you report $\varepsilon$, say what the public data are, how they were obtained, and whether their overlap with the private data was checked. Also state that the guarantee covers the private dataset given the pretrained model. Any choice made by looking at private data must be paid for, including which public model to start from and the step size. For example, suppose you fine-tune three public models privately and keep the one with the best private validation accuracy. That spends the budget of all three runs, composed, unless you use a private selection procedure ([Hyperparameter tuning and private selection](03-hyperparameter-tuning.md)). Choosing on public validation data would have been free.

## Code {#sec-pub-code}

Before choosing what to fine-tune, count the trainable weights of each candidate. The noise length for each set, at the noise multiplier your accountant gives, is the first number to compare with the signal.

```python
from math import sqrt


def noise_length(sigma: float, clip: float, batch: float, trainable: int) -> float:
    """Length of DP-SGD's noise on the averaged gradient over `trainable` coordinates."""
    return sigma * clip * sqrt(trainable) / batch


def lora_weights(rank: int, d_in: int, d_out: int) -> int:
    """Trainable weights of a rank-r adapter W0 + U V^T on a d_out x d_in matrix."""
    return rank * (d_in + d_out)


adapter = lora_weights(4, 51, 64) + 65  # first-layer adapter plus the 65-weight last layer
print(adapter)  # 525
print(f"{noise_length(4.13, 0.5, 100, 3329):.3f}")  # 1.191
print(f"{noise_length(4.13, 0.5, 100, adapter):.3f}")  # 0.473
print(f"{noise_length(4.13, 0.5, 100, 65):.3f}")  # 0.166
big = lora_weights(8, 768, 768)
print(big, 768 * 768 // big)  # 12288 48
print(f"{sqrt(768 * 768 / big):.1f}")  # 6.9
```

The full edition's script builds the public task and pretrains the student. It then runs every fine-tuning strategy, checks the adapter's ghost norms against explicit gradients, and estimates the curvature and the public subspace. It takes about four minutes on a laptop:

```bash
uv run python code/part05-dp-ml/public_data.py
```

## Cheat sheet

| Result | Formula | Conditions and notes |
| --- | --- | --- |
| Noise length | $\sigma C\sqrt k/B$ over $k$ trainable weights | Same $\varepsilon$ whatever is trained |
| Noise cost of $T$ steps | $\approx\frac{\eta s^2}{4}d_{\text{eff}}$, $d_{\text{eff}} = \sum_j(1 - (1 - \eta h_j)^{2T})$ | Quadratic loss near the start, $\eta h_j \ll 1$; compares strategies, does not predict accuracy |
| Effective dimension | $d_{\text{eff}} \le \min(k, 2\eta T\operatorname{tr}H)$ | Counts directions with curvature above about $1/(2\eta T)$ |
| Linear probe | Private convex ERM in $k$ dimensions, Lipschitz constant at most $\sqrt k$ | Features computed from public data only; each feature at most 1 in size |
| Probe error | Feature gap + privacy cost $\tilde\Theta(LR_w\sqrt{k\ln(1/\delta)}/(n\varepsilon))$ | The feature gap does not shrink with $n$ or $\varepsilon$ |
| LoRA weights | $r(d_{\text{in}} + d_{\text{out}})$ per matrix | Saves noise only at small rank |
| LoRA gradients | $\nabla_U\ell = g(V^\top a)^\top$, $\nabla_V\ell = a(U^\top g)^\top$ | Outer products, so ghost clipping applies |
| Public subspace | Same $(\varepsilon, \delta)$; break-even $(s\sqrt m/(\sqrt{f_m}\|\bar G\|))^2$; bias $1 - f_m$ | Subspace from public data only |
| Guarantee | $(\varepsilon, \delta)$-DP in the private data, for every fixed public dataset | Nothing for records in the public data |

: The results of this chapter. {#tbl-pub-cheat}

**Rules of thumb**

- Start from a model pretrained on public data that resembles the private task. On the running example this was worth about a factor of eight in $\varepsilon$.
- At very small budgets, train the last layer, which gives few weights, a convex problem and a known ceiling.
- At moderate budgets, train a low-rank adapter or all the weights. The adapter is the robust middle choice. Tune the step size on public data, and expect it to be smaller than from scratch.
- Count the trainable weights, but think in effective dimension. Noise in directions the loss does not use is cheap, and noise in the directions it does use is what you pay for.
- The guarantee covers the private data given the public model. Check the public data for overlap with the private data, and say what they are when you report $\varepsilon$.

## Further reading

- Tramèr & Boneh, "Differentially Private Learning Needs Better Features (or Much More Data)" (ICLR 2021): why linear probing on good features beats end-to-end private training at moderate budgets.
- Li, Tramèr, Liang & Hashimoto, "Large Language Models Can Be Strong Differentially Private Learners" (ICLR 2022), and Yu et al., "Differentially Private Fine-tuning of Language Models" (ICLR 2022): full and parameter-efficient private fine-tuning, and the hyperparameters that make each work.
- Li, Liu, Hashimoto, Inan, Kulkarni, Lee & Thakurta, "When Does Differentially Private Learning Not Suffer in High Dimensions?" (NeurIPS 2022): the effective-dimension view behind @sec-pub-dimension.
- Tramèr, Kamath & Carlini, "Position: Considerations for Differentially Private Learning with Large-Scale Public Pretraining" (ICML 2024): what public pretraining does and does not protect, and how benchmarks can mislead.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/06-public-data.md), except the rate of private convex ERM, which it cites from Bassily, Smith & Thakurta (2014), the published findings on language and image models, which it reports, and the effect of a less related public task, which an exercise of the full edition measures.
