# Measurements

A **measurement** ends a chain. It takes data and returns a random release, and it bounds how much the distribution of that release can change when one person's data change. [Transformations](09-transformations.md) built the deterministic half of a query, and this chapter builds the random half, which gives the privacy guarantee.

You read every measurement in OpenDP from four parts. The input domain says what it accepts, and the input metric compares two inputs. The output measure is the currency of the privacy cost, and the privacy map turns a distance between inputs into a cost in that currency. With these four parts, you can calibrate any mechanism in the library and compare two mechanisms fairly.

The script [opendp_measurements.py](../../code/part03-framework/opendp_measurements.py) builds every mechanism on this page on OpenDP 0.16.0 and prints its metric, measure and map. A thresholded count of `ILOSTAT` and a private median of `HWUSUAL` run on the Labour Force Survey (LFS), and the others use the documentation's toy inputs. The maps, checks, thresholds, scores and error messages are the same on every run. The releases change, because OpenDP's noise cannot be seeded, so they are in one block marked as one run.

```bash
uv run python code/part03-framework/opendp_measurements.py
```

## Additive noise {#sec-meas-additive}

The simplest measurement adds a random draw to a number. The constructor `make_laplace` adds Laplace noise and counts privacy in the budget $\varepsilon$ of pure differential privacy (DP), under the output measure `MaxDivergence`. The constructor `make_gaussian` adds Gaussian noise and counts privacy in the budget $\rho$ of zero-concentrated differential privacy (zCDP), under `ZeroConcentratedDivergence`. Both use the same input space.

```{.python excerpt="meas-scalar"}
def scalar_mechanisms(scale: float) -> tuple[dp.Measurement, dp.Measurement]:
    """Laplace and Gaussian noise on one float; the input space is the same, the output measures differ."""
    space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
    laplace = space >> dp.m.then_laplace(scale=scale)  # output measure: MaxDivergence, read as epsilon
    gaussian = space >> dp.m.then_gaussian(scale=scale)  # output measure: ZeroConcentratedDivergence, read as rho
    return laplace, gaussian
```

The Laplace map is the sensitivity divided by the scale, so a unit sensitivity at unit scale costs $\varepsilon = 1$. The Gaussian map is the squared sensitivity over twice the squared scale, so the same inputs give $\rho = 0.5$. The `check` method compares the map with a budget, so a sensitivity of 2 fails the budget that a sensitivity of 1 fits.

```text
laplace float: map(1.0)=1.0 measure=MaxDivergence
gaussian float: map(1.0)=0.5 measure=ZeroConcentratedDivergence
laplace float: check(1.0, 1.0)=True check(2.0, 1.0)=False
```

The two currencies promise different things. Pure DP with parameter $\varepsilon$ says that one person changes the probability of any release by a factor of at most $e^\varepsilon$. Zero-concentrated DP with parameter $\rho$ bounds the privacy loss on average and in its tails at once, which suits the thin tails of the Gaussian. [Combinators](11-combinators.md) converts $\rho$ into $\varepsilon$ with a slack $\delta$, and [Assessing utility and choosing the noise](../part02-tabular-data/07-utility-and-noise.md) compares the two. [The full edition's chapter on the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) derives both maps.

### Integers, floats and vectors

The same constructors accept integers. On an integer domain, `make_laplace` draws from the discrete Laplace distribution and `make_gaussian` from the discrete Gaussian, so the release is an integer. The maps do not change.

```text
laplace int: map(1)=1.0 output type=int
gaussian int: map(1)=0.5
```

A vector input changes the metric. Laplace noise needs a bound on the L1 distance, the sum of the absolute differences, because its map adds the cost of each coordinate. Gaussian noise needs a bound on the L2 distance, the square root of the summed squares, because its cost adds in squares. The constructor refuses a space with the wrong metric.

```{.python excerpt="meas-vector"}
def vector_mechanisms(scale: float) -> tuple[dp.Measurement, dp.Measurement]:
    """The same two mechanisms on a vector of floats: Laplace wants an L1 bound, Gaussian an L2 bound."""
    domain = dp.vector_domain(dp.atom_domain(T=float, nan=False))
    laplace = (domain, dp.l1_distance(T=float)) >> dp.m.then_laplace(scale=scale)
    gaussian = (domain, dp.l2_distance(T=float)) >> dp.m.then_gaussian(scale=scale)
    return laplace, gaussian
```

So doubling the L2 sensitivity multiplies $\rho$ by four, while the Laplace map stays linear. A plain float given to a vector mechanism is a type error, and the message says what the mechanism expected.

```text
laplace vector: metric=L1Distance(f64) map(1.0)=1.0
gaussian vector: metric=L2Distance(f64) map(1.0)=0.5
gaussian vector: map(2.0)=2.0 map(4.0)=8.0
laplace vector on a scalar: Expected type is Vec<f64> but input data is not a list.
gaussian on an L1 space: No match for concrete type L1Distance<f64>. See https://github.com/opendp/opendp/discussions/451.
```

### Why OpenDP rounds, and the parameter `k`

A computer cannot sample a continuous distribution, and with floating-point noise the gaps between floats can leak the input. So OpenDP turns a float input into an exact rational, adds discrete noise to its numerator, and rounds the result back to a float. This equals true continuous noise followed by rounding, which is post-processing, so the analysis stays exact.

The grid of that rational is $2^k$, where `k` is a parameter of the constructor. The default `k` is the smallest exponent of a 64-bit float, so the grid is exact and costs nothing. A coarser grid samples faster, but rounding to it can move two neighbouring inputs one grid step further apart. So the map adds that step to the sensitivity: a grid of a half adds a half to every map, even at zero sensitivity.

```text
laplace k=-1074: map(0.0)=0.0 map(1.0)=1.0
laplace k=-1: map(0.0)=0.5 map(1.0)=1.5
laplace k=0: map(0.0)=1.0 map(1.0)=2.0
laplace k=1: map(0.0)=2.0 map(1.0)=3.0
laplace k on integers: k is only valid for domains over floats
```

Integers need no grid, so the constructor rejects `k` on an integer domain. Change `k` only if a vector release is too slow, and then read the map again. [The full edition's chapter on discrete mechanisms](../../book-full/part03-dp-foundations/08-discrete-mechanisms.md) explains why the discrete distributions are safe.

### The geometric mechanism and `make_noise`

The constructor `make_geometric` is the Laplace mechanism on integers with one more option. With `bounds`, the sampler runs in constant time, which closes a timing side channel, and it clamps the release. It fails on floats, because the geometric distribution has no float version.

```{.python excerpt="meas-geometric"}
geometric = int_space >> dp.m.then_geometric(scale=1.0)
bounded = int_space >> dp.m.then_geometric(scale=1.0, bounds=(0, 10))  # runs in constant time; clamps the output
```

```text
geometric: map(1)=1.0 measure=MaxDivergence
geometric bounds (0, 10): map(1)=1.0
geometric on floats: No match for concrete type f64. See https://github.com/opendp/opendp/discussions/451.
```

The general constructor behind the two is `make_noise`. You name the output measure, and the library picks the distribution for it: Laplace for the max divergence and Gaussian for the zero-concentrated divergence. Use it in code that lets the caller choose the currency.

```{.python excerpt="meas-noise"}
noise_pure = space >> dp.m.then_noise(dp.max_divergence(), scale=1.0)  # the Laplace mechanism
noise_zcdp = space >> dp.m.then_noise(dp.zero_concentrated_divergence(), scale=1.0)  # the Gaussian mechanism
```

```text
make_noise MaxDivergence: map(1.0)=1.0 measure=MaxDivergence
make_noise ZeroConcentratedDivergence: map(1.0)=0.5 measure=ZeroConcentratedDivergence
```

## Canonical noise {#sec-meas-canonical}

The mechanisms above take a scale and give its cost. The canonical noise mechanism takes the sensitivity and the budget instead, and draws from the distribution that spends exactly that budget. The budget is approximate DP, which relaxes the pure guarantee with a slack $\delta$, the probability that the factor $e^\varepsilon$ fails. Its canonical distribution is the Tulap distribution, a discrete Laplace draw plus a uniform one, which leaves the most powerful hypothesis test possible at the budget. So use it when a release feeds a statistical test or a confidence interval.

```{.python excerpt="meas-canonical"}
def canonical_mechanism(d_in: float, epsilon: float, delta: float) -> dp.Measurement:
    """Canonical noise on one float, calibrated so that inputs at most d_in apart cost exactly (epsilon, delta)."""
    space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
    return space >> dp.m.then_canonical_noise(d_in=d_in, d_out=(epsilon, delta))
```

The map is a step and not a line, because the calibration is fixed at construction. Up to the declared sensitivity, the map returns the declared pair, and above it the map refuses, and so does `check`.

```text
canonical noise: measure=Approximate(MaxDivergence)
canonical noise: map(1.0)=(1.0, 1e-07) map(0.5)=(1.0, 1e-07)
canonical noise: check(1.0, (1.0, 1e-07))=True check(1.0, (0.9, 1e-07))=False
canonical noise map(2.0): d_in from the map (2) must be in [0, 1]
```

The installed version supports only a scalar float and an $(\varepsilon, \delta)$ budget. In the Context API, `query.canonical_noise()` with `binomial_size` gives confidence intervals and p-values for a binomial count, as the documentation's notebook shows.

## Thresholded noise {#sec-meas-threshold}

When the keys of a histogram come from the data, noise alone does not protect them: if a category appears in the release, someone in the data belongs to it. So the thresholded noise mechanisms release only the keys whose noisy count passes a threshold. A key with many contributors passes on both of two neighbouring datasets, so its presence says nothing about one person. A key with one contributor passes only if its noise is very large, and the slack $\delta$ pays for that chance. [Grouping](../part02-tabular-data/04-grouping.md) used this mechanism through Polars.

The input domain is a map from keys to counts, and the input metric is a triple of bounds. They bound how many keys one person can touch, the total change across all keys, and the change to any one key. The total is in L1 for Laplace and in L2 for Gaussian.

```{.python excerpt="meas-threshold"}
def threshold_mechanisms(scale: float, threshold: float) -> tuple[dp.Measurement, dp.Measurement]:
    """Laplace and Gaussian noise on a dictionary of float counts, releasing only the keys whose noisy count passes."""
    domain = dp.map_domain(dp.atom_domain(T=str), dp.atom_domain(T=float, nan=False))
    laplace = dp.m.make_laplace_threshold(
        domain, dp.l01inf_distance(dp.absolute_distance(T=float)), scale=scale, threshold=threshold
    )
    gaussian = dp.m.make_gaussian_threshold(
        domain, dp.l02inf_distance(dp.absolute_distance(T=float)), scale=scale, threshold=threshold
    )
    return laplace, gaussian
```

In the documentation's example, a person touches one key by at most one. At scale 1, the Laplace version prices that at $\varepsilon = 1$ plus a $\delta$, the chance that a count one person can make passes the threshold by noise alone. The Gaussian version prices it in $\rho$ with a far smaller $\delta$, because its tails fall faster.

```text
laplace threshold: metric=L01InfDistance(AbsoluteDistance(f64)) measure=Approximate(MaxDivergence)
laplace threshold scale 1 threshold 20: map((1, 1.0, 1.0))=(1.0, 2.8014e-09)
gaussian threshold scale 1 threshold 20: map((1, 1.0, 1.0))=(0.5, 1.1102e-16)
gaussian threshold: measure=Approximate(ZeroConcentratedDivergence)
```

But a smaller $\delta$ is not a smaller cost, because the currencies differ. At the Laplace $\delta$, the Gaussian guarantee converts to an $\varepsilon$ about six times larger ([Combinators](11-combinators.md)). The Gaussian wins only when a person touches many keys by a little, as in the second line. Then the L2 bound is much smaller than the L1 bound.

```text
gaussian threshold as approximate DP at delta=2.8014e-09: epsilon=6.3036
spread (100, 10.0, 0.001): laplace (0.1000, 1.0316e-07) gaussian epsilon=0.0500
```

In that line, the Laplace map charged a tenth, not ten. A hundred keys changed by at most a thousandth change the total by at most a tenth, so the library tightens the L1 bound to that product. Integer counts use the discrete distributions. The general `make_noise_threshold` takes the measure as an argument, and the installed version needs it wrapped as approximate, which its docstring does not say.

```text
discrete laplace threshold scale 1 threshold 10: map((1, 1, 1))=(1.0, 3.3190e-05)
make_noise_threshold Approximate(MaxDivergence): map((1, 1, 1))=(1.0, 3.3190e-05)
make_noise_threshold MaxDivergence: output_measure must be MaxDivergence or ZeroConcentratedDivergence
```

### Counts of `ILOSTAT`

The LFS codes the labour status of each response in `ILOSTAT`, with four codes in this sample. Suppose that you do not know which codes occur, so the keys must come from the data. The script counts the responses per code with Polars and gives the dictionary to a thresholded Laplace mechanism.

```{.python excerpt="meas-ilostat"}
def ilostat_counts(frame: pl.LazyFrame) -> dict[int, int]:
    """The number of responses in each ILOSTAT category, keyed by the category: the input of the thresholded count."""
    counts = frame.group_by("ILOSTAT").len().sort("ILOSTAT").collect()
    return {int(key): int(count) for key, count in zip(counts["ILOSTAT"], counts["len"])}


def ilostat_mechanism(scale: float, threshold: int) -> dp.Measurement:
    """Discrete Laplace noise on integer counts keyed by an integer category, with a threshold on the noisy count."""
    domain = dp.map_domain(dp.atom_domain(T=int), dp.atom_domain(T=int))
    metric = dp.l01inf_distance(dp.absolute_distance(T=int))
    return dp.m.make_laplace_threshold(domain, metric, scale=scale, threshold=threshold)
```

One person has at most 36 rows ([the privacy unit](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context)). Without more public knowledge, each entry of the triple is 36, because a person could touch 36 keys once each, or one key 36 times. At a budget of 1, the scale is 36, with an error bar of about 108 responses. So the threshold is the only free choice, and it sets $\delta$. The script prices three thresholds and finds the smallest that keeps $\delta$ under one in ten million.

```text
ILOSTAT counts: {1: 1517011, 2: 155661, 3: 1450283, 9: 688977}
ILOSTAT d_in: (36, 36, 36) scale: 36 95% accuracy: 108.3
ILOSTAT threshold 500: map=1.0, delta=4.4835e-05
ILOSTAT threshold 1000: map=1.0, delta=4.1667e-11
ILOSTAT threshold 1500: map=1.0, delta=3.9968e-15
ILOSTAT smallest threshold with delta <= 1e-07: 720
ILOSTAT threshold 10: threshold (10) must not be smaller than l-infinity sensitivity 36
```

Each $\delta$ is the chance that the noise lifts a count that one person can make alone, at most 36, over the threshold. It falls fast as the threshold rises. The mechanism refuses a threshold below the per-key sensitivity, because one person could then make a key with no noise. Here the smallest count is over 150,000, so a threshold of 720 drops nothing. The threshold removes only rare categories, which is where it must act. [The full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) shows, under partition selection, why it costs a $\delta$.

## The approximate Laplace projection {#sec-meas-alp}

The approximate Laplace projection is another answer to unknown keys. It releases a hashed sketch of the whole key space, which you then query for any key, including keys that never occurred. It lists no keys, so no key leaks, and queries after the release are post-processing. It has two costs: answers are rounded to multiples of `alpha`, four by default, and two frequent keys can collide in the hash unless the sketch is large enough. The size of the sketch grows with two public bounds. These are `total_limit` on the sum of all counts and `value_limit` on any one count.

```{.python excerpt="meas-alp"}
def projection_mechanism(scale: float) -> dp.Measurement:
    """A private projection of counts keyed by strings; releases a queryable that answers for any key, known or not."""
    domain = dp.map_domain(dp.atom_domain(T=str), dp.atom_domain(T=int))
    metric = dp.l01inf_distance(dp.absolute_distance(T=int))
    return dp.m.make_alp_queryable(domain, metric, scale=scale, total_limit=50_000, value_limit=1_000)
```

Be careful: here `scale` is not a noise scale but the privacy loss per unit of sensitivity, so a larger value spends more, not less. The map at unit sensitivity equals the scale, and it grows with the L1 entry of the triple.

```text
alp scale 1: map((1, 1, 1))=1.0 measure=MaxDivergence
alp scale 2: map((1, 1, 1))=2.0
alp scale 1: map((2, 2, 1))=2.0
```

The documentation finds the scale for a budget with `binary_search_chain` ([Utilities and the Context API inside](12-utilities-and-the-context-api.md)). On the toy table, the one-run answers are within a few units of the true counts. The name that is not in the table comes back near zero.

## Noisy max and selection {#sec-meas-noisy-max}

Some releases are a choice: which category is largest, which candidate is the median. The noisy max mechanisms add noise to each score and release the index of the largest. Their input metric is the L-infinity distance, a bound on the change of any one score. That is the right bound, because the release depends on how the scores compare. The output measure chooses the noise: exponential under the max divergence, and Gumbel under the zero-concentrated divergence.

```{.python excerpt="meas-noisy-max"}
def noisy_max_mechanisms(scale: float) -> tuple[dp.Measurement, dp.Measurement]:
    """Select the index of the largest score: exponential noise under pure DP, Gumbel noise under zCDP."""
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.linf_distance(T=float)
    exponential = space >> dp.m.then_noisy_max(dp.max_divergence(), scale=scale)
    gumbel = space >> dp.m.then_noisy_max(dp.zero_concentrated_divergence(), scale=scale)
    return exponential, gumbel
```

The exponential version's map is twice the sensitivity over the scale, because one person can raise one score and lower another at the same time. So at a scale of 2, a unit sensitivity costs $\varepsilon = 1$. If the scores can move only in one direction, declare the metric monotonic, which removes the factor of two. The Gumbel version's map is the square of the same ratio, divided by eight. Negated scores select the smallest at no extra cost, and `make_noisy_top_k` releases the best $k$ indices for $k$ times the cost.

```text
noisy max: metric=LInfDistance(T=f64) measure=MaxDivergence
noisy max exponential scale 2: map(1.0)=1.0
noisy max gumbel scale 2: map(1.0)=0.125 measure=ZeroConcentratedDivergence
noisy max monotonic scale 2: exponential map(1.0)=0.5 gumbel map(1.0)=0.03125
noisy min scale 2: map(1.0)=1.0
noisy top 2 scale 2: exponential map(1.0)=2.0 gumbel map(1.0)=0.25
```

The older `make_report_noisy_max_gumbel` still exists, but the installed version marks it deprecated. [The full edition's chapter on selection](../../book-full/part03-dp-foundations/05-selection.md) shows why selection costs twice what a count costs, and why exponential noise beats Gumbel under pure DP.

```text
report_noisy_max_gumbel scale 2: map(1.0)=1.0 measure=MaxDivergence
report_noisy_max_gumbel: Call to deprecated function (or staticmethod) make_report_noisy_max_gumbel
```

### A private median of `HWUSUAL`

OpenDP releases a quantile by selection. The transformation `make_quantile_score_candidates` of the previous chapter scores each candidate, here the hours from 1 to 98, by how far it is from splitting the data in half. The best candidate has the smallest score, so a noisy max over negated scores selects it. A lookup in the candidate list then turns the index into an answer, as post-processing. The constructor `make_private_quantile` does the three steps in one call.

```{.python excerpt="meas-median"}
def hours_of_the_working(frame: pl.LazyFrame) -> list[float]:
    """HWUSUAL for the people at work (ILOSTAT 1), keeping 1 to 98 hours: 0 means varying, 99 not applicable."""
    at_work = (pl.col("ILOSTAT") == 1) & pl.col("HWUSUAL").is_between(1, 98)
    return frame.filter(at_work).select("HWUSUAL").collect()["HWUSUAL"].to_list()


def median_by_hand(scale: float) -> tuple[dp.Transformation, dp.Measurement]:
    """The private median from its pieces: score each candidate, then pick an index by noisy max."""
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()
    scores = space >> dp.t.then_quantile_score_candidates(candidates=HOURS, alpha=0.5)
    select = scores >> dp.m.then_noisy_max(dp.max_divergence(), scale=scale, negate=True)  # the smallest score wins
    return scores, select  # the released index is looked up in HOURS afterwards, which is post-processing


def median_mechanism(scale: float) -> dp.Measurement:
    """The same median in one constructor."""
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()
    return space >> dp.m.then_private_quantile(dp.max_divergence(), candidates=HOURS, alpha=0.5, scale=scale)
```

The data are the usual hours of the people at work, without the codes for "varies" and "not applicable". A row filter adds no rows, so the input distance is still 36. Each integer score moves by at most the input distance, so the scores' L-infinity sensitivity is 36.

```text
HWUSUAL at work, 1 to 98 hours: n=1,490,113 true median=37
quantile scores: metric=LInfDistance(T=u64) map(36)=36
best candidate: 37 hours, score 9,111
second candidate: 36 hours, score 95,652
gap in noise scales: 1,202
scores >> noisy max scale 72: map(36)=1.0
make_private_quantile scale 36: map(36)=1.0
make_private_quantile zCDP scale 36: map(36)=0.125
```

Two scales appear because `make_private_quantile` doubles the scale you give it. The scores of a median are integers multiplied by two, the denominator of one half. So the constructor multiplies the scale by the same two, and the two maps agree at $\varepsilon = 1$. Read the scores before you trust the release. The best candidate beats the next by over a thousand noise scales, so the release is 37 hours on almost every run. On a smaller dataset the gaps are smaller, and the release moves among nearby candidates, which is the error of a private quantile.

## Randomized response {#sec-meas-rr}

All the mechanisms so far run on a trusted server that holds the data. Randomized response runs on the respondent's device before the answer is sent, so no server sees the truth. This is the local model of DP, where the privacy unit is one answer. Its metric is the discrete distance, which is 1 between two different inputs and 0 otherwise. OpenDP gives only the randomizers, not the transport or the aggregation.

```{.python excerpt="meas-rr"}
def randomized_response_mechanisms(prob: float) -> tuple[dp.Measurement, dp.Measurement]:
    """Randomized response on one yes/no answer and on one ILOSTAT category, each kept truthful with probability prob."""
    boolean = dp.m.make_randomized_response_bool(prob=prob)
    categorical = dp.m.make_randomized_response(categories=[1, 2, 3, 9], prob=prob)
    return boolean, categorical
```

The boolean version keeps the true answer with the probability you give and flips it otherwise. Its map is the logarithm of the odds of the truth, so a probability of three quarters costs about 1.1. It refuses a probability below one half. The categorical version keeps the truth with the same probability and otherwise picks one of the other categories at random. Its map grows with the logarithm of the number of alternatives, so four `ILOSTAT` codes cost twice what a yes/no answer costs.

```text
randomized response bool: domain=AtomDomain(T=bool) metric=DiscreteDistance() measure=MaxDivergence
randomized response bool prob 0.75: map(1)=1.0986 ln(0.75/0.25)=1.0986
randomized response bool prob 0.9: map(1)=2.1972
randomized response bool prob 0.4: probability must be within [0.5, 1]
randomized response 4 categories prob 0.75: map(1)=2.1972 ln(3*0.75/0.25)=2.1972
```

One randomized answer is almost useless by design. The answers become useful in aggregate, after a correction for the known flip rate. The third randomizer, for a bit vector, is the primitive behind RAPPOR, the first large deployment of local DP. Its domain bounds how many bits can be set, and it rerandomizes each bit with probability `f`. Its map is twice that weight times the log odds that a bit survives.

```{.python excerpt="meas-bitvec"}
def bitvector_mechanism(f: float, max_weight: int) -> dp.Measurement:
    """Randomized response on a bit vector with at most max_weight ones, each bit rerandomized with probability f."""
    return dp.m.make_randomized_response_bitvec(dp.bitvector_domain(max_weight=max_weight), dp.discrete_distance(), f=f)
```

```text
bit vector f=0.8 max_weight=3: domain=BitVectorDomain(weight=3) map(1)=2.4328 formula=2.4328
```

The script sends the documentation's 24-bit vector through the randomizer ten thousand times and runs `debias_randomized_response_bitvec` on the answers. The three set bits come back near ten thousand and the others near zero, with a spread of a few hundred. So local DP costs much accuracy ([the full edition's chapter on trust models](../../book-full/part03-dp-foundations/09-trust-models.md)).

Here are the releases of one run. The additive releases are within a few units of 100, the thresholded count kept all four codes, the projection answered the unknown name with zero, and both medians are 37.

```text
released (one run):
  laplace on 100.0: 104.05
  gaussian on 100.0: 100.26
  discrete laplace on 100: 100
  laplace vector on [0.0, 2.0, 2.0]: [-1.4, 1.17, 4.28]
  laplace k=-1 on 100.0, five draws: [99.5, 99.5, 102.5, 101.5, 100.0]
  bounded geometric on 10, five draws: [10, 10, 10, 9, 7]
  canonical noise on 100.0: 99.74
  laplace threshold on {a: 0, b: 20, c: 40}: {'c': 37.74439496773442}
  ILOSTAT counts: {1: 1517058, 2: 155697, 3: 1450247, 9: 689013}
  alp answers for Michael, James, Sharon, Lancelot: [748.0, 500.0, 48.0, 0.0]
  noisy max on [0, 2, 4, 6]: 3
  noisy min on [0, 2, 4, 6]: 0
  noisy top 2 on [0, 2, 4, 6]: [3, 1]
  median by hand: 37 hours
  median by make_private_quantile: 37 hours
  randomized response on True, ten draws: [True, True, False, True, True, False, False, False, True, True]
  randomized response on category 1, ten draws: [1, 1, 1, 1, 1, 1, 1, 2, 1, 3]
  bit vector release: 100110111000000010001110
  bit vector frequencies over 10,000 copies, bits 12, 20, 21: [10050, 10235, 10115]
  bit vector frequencies, largest other bit: 565
```

## Summary

| Constructor | Input metric | Output measure | Map at scale 1 for inputs at distance 1 | What it is for |
| --- | --- | --- | --- | --- |
| `make_laplace` | absolute or L1 distance | `MaxDivergence` | $\varepsilon = 1$ | One number or a vector under pure DP; integers get the discrete Laplace |
| `make_gaussian` | absolute or L2 distance | `ZeroConcentratedDivergence` | $\rho = 0.5$ | The same under zCDP; vectors cost by the L2 bound |
| `make_geometric` | absolute distance on integers | `MaxDivergence` | $\varepsilon = 1$ | Integer Laplace with `bounds` for constant time |
| `make_noise` | as above | the one you pass | as above | Lets the caller choose the currency |
| `make_canonical_noise` | absolute distance | `Approximate(MaxDivergence)` | the `d_out` you declared, up to your `d_in` | Exact $(\varepsilon, \delta)$ calibration; inference on counts |
| `make_laplace_threshold` | L0, L1, L-infinity triple | `Approximate(MaxDivergence)` | $(1, 2.8014 \times 10^{-9})$ at threshold 20 | Counts over keys from the data; $\delta$ pays for the threshold |
| `make_gaussian_threshold` | L0, L2, L-infinity triple | `Approximate(ZeroConcentratedDivergence)` | $(0.5, 1.1102 \times 10^{-16})$ at threshold 20 | The same when a person touches many keys by a little |
| `make_alp_queryable` | L0, L1, L-infinity triple | `MaxDivergence` | $\varepsilon = 1$ (the scale is a loss, not a noise) | Queries for any key after the release |
| `make_noisy_max` | L-infinity distance | `MaxDivergence` or `ZeroConcentratedDivergence` | $\varepsilon = 2$ or $\rho = 0.5$ | Select an index by score; monotonic halves the cost |
| `make_noisy_top_k` | L-infinity distance | either | $\varepsilon = 4$ for $k = 2$ | The best $k$ indices, at $k$ times the cost |
| `make_private_quantile` | symmetric distance | either | $\varepsilon = 1$ | Scores plus noisy max plus lookup in one call |
| `make_randomized_response_bool` | discrete distance | `MaxDivergence` | $1.099$ at probability 0.75 | One yes/no answer on the respondent's device |
| `make_randomized_response` | discrete distance | `MaxDivergence` | $2.197$ for four categories at 0.75 | One category; the cost grows with the alternatives |
| `make_randomized_response_bitvec` | discrete distance | `MaxDivergence` | $2.433$ for weight 3 at `f` = 0.8 | The RAPPOR primitive; debias the aggregate |

: The measurements of the chapter: their input metric, their output measure, the map the script printed at scale 1 for inputs at distance 1, and what each is for. {#tbl-meas-summary}

## References

- OpenDP documentation: "Measurements", "Additive Noise Mechanisms", "Canonical Noise Mechanism", "Thresholded Noise Mechanisms", "Approximate Laplace Projection", "Report Noisy Max Mechanisms" and "Randomized Response" in the API user guide.
- Dwork, McSherry, Nissim and Smith, "Calibrating Noise to Sensitivity in Private Data Analysis" (2006): the Laplace mechanism.
- Canonne, Kamath and Steinke, "The Discrete Gaussian for Differential Privacy" (2020): the discrete Gaussian and the discrete Laplace samplers OpenDP uses.
- Awan and Vadhan, "Canonical Noise Distributions and Private Hypothesis Tests" (2023): the Tulap distribution.
- Rogers, "A Unifying Privacy Analysis Framework for Unknown Domain Algorithms in Differential Privacy" (2023): the thresholded mechanisms' maps.
- Aumüller, Lebeda and Pagh, "Differentially Private Sparse Vectors with Low Error, Optimal Space, and Fast Access" (2021): the approximate Laplace projection.
- McKenna and Sheldon, "Permute-and-Flip: A New Mechanism for Differentially Private Selection" (2020): the exponential-noise noisy max.
- Smith, "Privacy-Preserving Statistical Estimation with Optimal Convergence Rates" (2011): the quantile by selection.
- Warner, "Randomized Response: A Survey Technique for Eliminating Evasive Answer Bias" (1965), and Erlingsson, Pihur and Korolova, "RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response" (2014).
- The full edition's chapters on [the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md), [selection](../../book-full/part03-dp-foundations/05-selection.md), [sensitivity and partition selection](../../book-full/part03-dp-foundations/07-sensitivity.md), [discrete mechanisms](../../book-full/part03-dp-foundations/08-discrete-mechanisms.md) and [trust models](../../book-full/part03-dp-foundations/09-trust-models.md) derive the results this chapter states.
