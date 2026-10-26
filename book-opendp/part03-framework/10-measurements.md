# Measurements

A transformation moves data to data and promises how far its output can move. A **measurement** is the step that ends the chain. It takes data and returns a random release, and its promise is a bound on how much the distribution of that release can change when one person's data changes. The previous chapter, [Transformations](09-transformations.md), built the deterministic half of a query. This chapter builds the random half, which is where the privacy guarantee actually comes from.

OpenDP ships a small catalogue of measurements, and every one of them is read the same way. Its input domain says what it accepts, and its input metric says how two inputs are compared. Its output measure says in which currency the privacy is counted, and its privacy map turns a distance between inputs into a bound in that currency. Once you can read those four things, you can calibrate any mechanism in the library and compare two mechanisms honestly. The chapter goes through the catalogue in the documentation's order. It starts with the additive noise mechanisms and the canonical noise mechanism. Then come the thresholded mechanisms, which release counts for keys the data chose, and the approximate Laplace projection. It ends with the noisy max family, the private quantile built on it, and randomized response.

The script [opendp_measurements.py](../../code/part03-framework/opendp_measurements.py) builds every mechanism on this page on OpenDP 0.16.0 and prints its metric, its measure and its map. Two of them run on the Labour Force Survey (LFS): a thresholded count of the `ILOSTAT` categories and a private median of `HWUSUAL`. The rest take the documentation's toy inputs. Every map, check, threshold, score and error message is the same on every run. The released values are not, because OpenDP's noise cannot be seeded, so they appear in one block marked as one run at the end.

```bash
uv run python code/part03-framework/opendp_measurements.py
```

## Additive noise {#sec-meas-additive}

The simplest measurement adds a random draw to a number. OpenDP offers two draws, and the choice between them is a choice of currency. The constructor `make_laplace` adds Laplace noise and counts privacy in the budget $\varepsilon$ of pure differential privacy (DP), under the output measure `MaxDivergence`. The constructor `make_gaussian` adds Gaussian noise and counts privacy in the budget $\rho$ of zero-concentrated differential privacy (zCDP), under `ZeroConcentratedDivergence`. Both are built on the same space, so the input domain and the input metric do not decide which you get.

```{.python excerpt="meas-scalar"}
def scalar_mechanisms(scale: float) -> tuple[dp.Measurement, dp.Measurement]:
    """Laplace and Gaussian noise on one float; the input space is the same, the output measures differ."""
    space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
    laplace = space >> dp.m.then_laplace(scale=scale)  # output measure: MaxDivergence, read as epsilon
    gaussian = space >> dp.m.then_gaussian(scale=scale)  # output measure: ZeroConcentratedDivergence, read as rho
    return laplace, gaussian
```

The map of the Laplace mechanism is the sensitivity divided by the scale, so a unit sensitivity at unit scale costs $\varepsilon = 1$. The map of the Gaussian mechanism is the squared sensitivity over twice the squared scale, which at the same inputs gives $\rho = 0.5$. The `check` method compares the map with a budget. A sensitivity of 2 does not fit within the budget that a sensitivity of 1 fits.

```text
laplace float: map(1.0)=1.0 measure=MaxDivergence
gaussian float: map(1.0)=0.5 measure=ZeroConcentratedDivergence
laplace float: check(1.0, 1.0)=True check(2.0, 1.0)=False
```

What the two currencies promise is different. Pure DP with parameter $\varepsilon$ says that one person changes the probability of any release by a factor of at most $e^\varepsilon$. Zero-concentrated DP with parameter $\rho$ bounds the privacy loss on average and in its tails at once, which suits the Gaussian's thin tails. [Combinators](11-combinators.md) converts a $\rho$ into an $\varepsilon$ with a slack $\delta$. [Assessing utility and choosing the noise](../part02-tabular-data/07-utility-and-noise.md) weighs the two for a release. [The full edition's chapter on the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) derives both maps.

### Integers, floats and vectors

The same constructors accept integers. On an integer domain, `make_laplace` draws from the discrete Laplace distribution and `make_gaussian` from the discrete Gaussian, so the release is an integer too. The maps are unchanged, and the sensitivity is an integer as well.

```text
laplace int: map(1)=1.0 output type=int
gaussian int: map(1)=0.5
```

A vector input changes the metric. Laplace noise on a vector wants a bound on the L1 distance between two inputs, which is the sum of the absolute differences. That is because its map adds up the cost of every coordinate. Gaussian noise on a vector wants a bound on the L2 distance, the square root of the summed squares, because its cost adds up in squares. The constructor refuses a space whose metric does not match.

```{.python excerpt="meas-vector"}
def vector_mechanisms(scale: float) -> tuple[dp.Measurement, dp.Measurement]:
    """The same two mechanisms on a vector of floats: Laplace wants an L1 bound, Gaussian an L2 bound."""
    domain = dp.vector_domain(dp.atom_domain(T=float, nan=False))
    laplace = (domain, dp.l1_distance(T=float)) >> dp.m.then_laplace(scale=scale)
    gaussian = (domain, dp.l2_distance(T=float)) >> dp.m.then_gaussian(scale=scale)
    return laplace, gaussian
```

The maps read as before. Doubling the L2 sensitivity quadruples $\rho$, which is the squared sensitivity showing through, while the Laplace map stays linear. Passing a plain float to a vector mechanism is a type error, and the message says what was expected.

```text
laplace vector: metric=L1Distance(f64) map(1.0)=1.0
gaussian vector: metric=L2Distance(f64) map(1.0)=0.5
gaussian vector: map(2.0)=2.0 map(4.0)=8.0
laplace vector on a scalar: Expected type is Vec<f64> but input data is not a list.
gaussian on an L1 space: No match for concrete type L1Distance<f64>. See https://github.com/opendp/opendp/discussions/451.
```

### Why OpenDP rounds, and the parameter `k`

Floating-point noise is a known source of broken implementations. A computer cannot sample a continuous distribution, and the gaps between floats leak which input was used. OpenDP avoids the problem by never sampling a continuous draw. A float input is turned into an exact rational, integer noise is added to its numerator by the discrete distribution, and the result is rounded back to a float. That is the same as sampling true continuous noise and then rounding, which is post-processing, so the analysis stays exact.

The grid of that rational is $2^k$, where the integer `k` is a parameter of the constructor. The default `k` is the smallest exponent a 64-bit float can express, so the grid is exact and costs nothing. A coarser grid samples faster. But rounding the input to the grid can move two neighbouring inputs apart by up to one extra grid step, so the map adds that step to the sensitivity. The script shows the penalty. At the default there is none, while a grid of a half adds a half to every map, even at zero sensitivity.

```text
laplace k=-1074: map(0.0)=0.0 map(1.0)=1.0
laplace k=-1: map(0.0)=0.5 map(1.0)=1.5
laplace k=0: map(0.0)=1.0 map(1.0)=2.0
laplace k=1: map(0.0)=2.0 map(1.0)=3.0
laplace k on integers: k is only valid for domains over floats
```

Integers need no grid, so the constructor rejects `k` on an integer domain. Leave `k` alone unless a vector release is too slow, and if you change it, read the map again. The full edition explains the discrete distributions and why they are safe in [its chapter on discrete mechanisms](../../book-full/part03-dp-foundations/08-discrete-mechanisms.md).

### The geometric mechanism and `make_noise`

The constructor `make_geometric` is the Laplace mechanism restricted to integers, with one extra option. Passing `bounds` makes the sampler run in constant time, which closes a timing side channel, and it clamps the release into the bounds. On floats the constructor fails, because the geometric distribution has no float version.

```{.python excerpt="meas-geometric"}
geometric = int_space >> dp.m.then_geometric(scale=1.0)
bounded = int_space >> dp.m.then_geometric(scale=1.0, bounds=(0, 10))  # runs in constant time; clamps the output
```

```text
geometric: map(1)=1.0 measure=MaxDivergence
geometric bounds (0, 10): map(1)=1.0
geometric on floats: No match for concrete type f64. See https://github.com/opendp/opendp/discussions/451.
```

Finally, `make_noise` is the general constructor behind the two. You name the output measure, and the library picks the distribution that goes with it: Laplace for the max divergence and Gaussian for zero-concentrated divergence. It is the constructor to reach for in code that lets the caller choose the currency.

```{.python excerpt="meas-noise"}
noise_pure = space >> dp.m.then_noise(dp.max_divergence(), scale=1.0)  # the Laplace mechanism
noise_zcdp = space >> dp.m.then_noise(dp.zero_concentrated_divergence(), scale=1.0)  # the Gaussian mechanism
```

```text
make_noise MaxDivergence: map(1.0)=1.0 measure=MaxDivergence
make_noise ZeroConcentratedDivergence: map(1.0)=0.5 measure=ZeroConcentratedDivergence
```

## Canonical noise {#sec-meas-canonical}

The mechanisms above take a scale and tell you what it costs. The canonical noise mechanism works the other way round. You state the sensitivity and the budget, and it draws from the distribution that spends exactly that budget. Approximate DP relaxes the pure guarantee with a slack $\delta$, the probability with which the factor $e^\varepsilon$ may fail. For that guarantee the canonical distribution is the Tulap distribution, a discrete Laplace draw plus a uniform one. It is the noise that leaves the most powerful hypothesis test possible at a given budget. So it is the mechanism to use when a release feeds a statistical test or a confidence interval rather than a table.

```{.python excerpt="meas-canonical"}
def canonical_mechanism(d_in: float, epsilon: float, delta: float) -> dp.Measurement:
    """Canonical noise on one float, calibrated so that inputs at most d_in apart cost exactly (epsilon, delta)."""
    space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
    return space >> dp.m.then_canonical_noise(d_in=d_in, d_out=(epsilon, delta))
```

The map is a step rather than a line, because the calibration is fixed at construction. For any sensitivity up to the one you declared it returns the pair you declared. Above that sensitivity it refuses, since the mechanism was never calibrated for it. The `check` method behaves accordingly.

```text
canonical noise: measure=Approximate(MaxDivergence)
canonical noise: map(1.0)=(1.0, 1e-07) map(0.5)=(1.0, 1e-07)
canonical noise: check(1.0, (1.0, 1e-07))=True check(1.0, (0.9, 1e-07))=False
canonical noise map(2.0): d_in from the map (2) must be in [0, 1]
```

The installed version supports only a scalar float input and only an $(\varepsilon, \delta)$ budget. In the Context API the same mechanism is `query.canonical_noise()`. Passing `binomial_size` to it returns an object with confidence intervals and p-values for a binomially distributed count, which the documentation's notebook walks through on a count of married people.

## Thresholded noise {#sec-meas-threshold}

A histogram over keys that come from the data has a problem that noise alone does not fix. If a category appears in the release at all, then someone in the data belongs to it. When that someone is one person, the key itself is the leak. The thresholded noise mechanisms release a dictionary of noisy counts and keep only the keys whose noisy count clears a threshold. A key with many contributors survives on either of two neighbouring datasets, so its presence says nothing about any one person. A key with one contributor survives only if its noise is huge, and the probability of that is what the slack $\delta$ pays for. [Grouping](../part02-tabular-data/04-grouping.md) used this mechanism through Polars without naming it. This is what runs underneath.

The input domain is a map from keys to counts, and the input metric is a triple. The first entry bounds how many keys one person can touch. The second bounds the total change across all keys, in L1 for Laplace and in L2 for Gaussian. The third bounds the change to any one key.

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

The documentation's example has a person who touches one key by at most one. At scale 1 the Laplace version prices that at $\varepsilon = 1$ plus a $\delta$. The $\delta$ is the chance that a count one person can create clears the threshold by noise alone. The Gaussian version prices it in $\rho$ with a far smaller $\delta$, because its tails fall faster.

```text
laplace threshold: metric=L01InfDistance(AbsoluteDistance(f64)) measure=Approximate(MaxDivergence)
laplace threshold scale 1 threshold 20: map((1, 1.0, 1.0))=(1.0, 2.8014e-09)
gaussian threshold scale 1 threshold 20: map((1, 1.0, 1.0))=(0.5, 1.1102e-16)
gaussian threshold: measure=Approximate(ZeroConcentratedDivergence)
```

A smaller $\delta$ is not a smaller cost, because the two maps are in different currencies. The script converts the Gaussian guarantee to approximate DP at the Laplace mechanism's $\delta$, with the combinators of [Combinators](11-combinators.md). Then the honest comparison comes out the other way, and the Gaussian's $\varepsilon$ is about six times larger. The picture flips when a person touches many keys by a little, which is the case the second line prices. There the L2 bound is much smaller than the L1 bound, and the Gaussian wins.

```text
gaussian threshold as approximate DP at delta=2.8014e-09: epsilon=6.3036
spread (100, 10.0, 0.001): laplace (0.1000, 1.0316e-07) gaussian epsilon=0.0500
```

Notice that the Laplace map charged a tenth rather than ten over one. A person who touches a hundred keys by at most a thousandth can change the total by at most a tenth. The library knows this, so it tightens the stated L1 bound to that product. Integer counts take the discrete distributions, as in the previous section. The general form is `make_noise_threshold`, which takes the measure as an argument. The installed version wants that measure wrapped as approximate, which its docstring does not say.

```text
discrete laplace threshold scale 1 threshold 10: map((1, 1, 1))=(1.0, 3.3190e-05)
make_noise_threshold Approximate(MaxDivergence): map((1, 1, 1))=(1.0, 3.3190e-05)
make_noise_threshold MaxDivergence: output_measure must be MaxDivergence or ZeroConcentratedDivergence
```

### Counts of `ILOSTAT`

The LFS codes each response's labour status in `ILOSTAT`, and in this sample only four codes occur. Suppose you did not know which, so the keys must come from the data. The script counts the responses per code with Polars. Then it hands the dictionary to a thresholded Laplace mechanism over integer keys and integer counts.

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

A person answers at most 36 times. Without further public knowledge the triple is therefore 36 in every entry, since a person could touch that many keys by one each, or one key that many times. The scale is the sensitivity over $\varepsilon$, which at a budget of 1 is 36. The error bar at that scale is about 108 responses. The threshold is then the only free choice, and it buys $\delta$. The script prices three thresholds. Then it searches for the smallest one that keeps $\delta$ under the chapter's default of one in ten million.

```text
ILOSTAT counts: {1: 1517011, 2: 155661, 3: 1450283, 9: 688977}
ILOSTAT d_in: (36, 36, 36) scale: 36 95% accuracy: 108.3
ILOSTAT threshold 500: map=1.0, delta=4.4835e-05
ILOSTAT threshold 1000: map=1.0, delta=4.1667e-11
ILOSTAT threshold 1500: map=1.0, delta=3.9968e-15
ILOSTAT smallest threshold with delta <= 1e-07: 720
ILOSTAT threshold 10: threshold (10) must not be smaller than l-infinity sensitivity 36
```

Every $\delta$ comes from the same tail. It is the chance that the noise lifts a count one person can make alone, which is 36 at most, over the threshold. That chance is then taken over every key the person could create. Raising the threshold shrinks it fast. A threshold below the per-key sensitivity is refused outright, because one person could then create a key with no noise at all. Here the smallest count is in the hundreds of thousands, so a threshold of 720 drops nothing, and all four counts come back within their error bars. The threshold only bites on rare categories, which is exactly where it should. Why it costs a $\delta$ and not an $\varepsilon$ is derived in [the full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md), under partition selection.

## The approximate Laplace projection {#sec-meas-alp}

Thresholding is one answer to unknown keys. The approximate Laplace projection is another. Instead of releasing a dictionary, it releases a compressed, hashed sketch of the whole key space. You then query the sketch for any key you like, including keys that never occurred. Nothing is dropped and nothing is listed, so no key leaks, and queries after the release are post-processing. The price is twofold. Answers are rounded to multiples of a parameter `alpha`, four by default, and two frequent keys can collide in the hash unless the sketch is large enough. Its size grows with two public bounds, `total_limit` on the sum of all counts and `value_limit` on any one count. The documentation's example sets them for a table of first names.

```{.python excerpt="meas-alp"}
def projection_mechanism(scale: float) -> dp.Measurement:
    """A private projection of counts keyed by strings; releases a queryable that answers for any key, known or not."""
    domain = dp.map_domain(dp.atom_domain(T=str), dp.atom_domain(T=int))
    metric = dp.l01inf_distance(dp.absolute_distance(T=int))
    return dp.m.make_alp_queryable(domain, metric, scale=scale, total_limit=50_000, value_limit=1_000)
```

One trap: here `scale` is not a noise scale but the privacy loss per unit of sensitivity, so a larger value spends more, not less. The map at unit sensitivity equals the scale, and it grows with the L1 entry of the triple.

```text
alp scale 1: map((1, 1, 1))=1.0 measure=MaxDivergence
alp scale 2: map((1, 1, 1))=2.0
alp scale 1: map((2, 2, 1))=2.0
```

The documentation finds the scale for a budget with `binary_search_chain`, which [Utilities and the Context API inside](12-utilities-and-the-context-api.md) covers. On the toy table, the one-run answers at the end of the page land within a few units of the true counts. The name that is not in the table comes back near zero.

## Noisy max and selection {#sec-meas-noisy-max}

Not every release is a number. Often the question is which one: which category is largest, which candidate value is the median, which model is best. The noisy max mechanisms take a vector of scores, add noise to each, and release the index of the largest. Their input metric is the L-infinity distance, a bound on how much any single score can change. That is the right bound because the release depends on how the scores compare, not on their sum. As with additive noise, the output measure chooses the distribution: exponential noise under the max divergence, and Gumbel noise under zero-concentrated divergence.

```{.python excerpt="meas-noisy-max"}
def noisy_max_mechanisms(scale: float) -> tuple[dp.Measurement, dp.Measurement]:
    """Select the index of the largest score: exponential noise under pure DP, Gumbel noise under zCDP."""
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.linf_distance(T=float)
    exponential = space >> dp.m.then_noisy_max(dp.max_divergence(), scale=scale)
    gumbel = space >> dp.m.then_noisy_max(dp.zero_concentrated_divergence(), scale=scale)
    return exponential, gumbel
```

The map of the exponential version is twice the sensitivity over the scale, so at a scale of 2 a unit sensitivity costs $\varepsilon = 1$. The factor of two is there because one person can raise one score and lower another at the same time. If the scores can only move in one direction, you declare the metric monotonic, and the factor goes away. The Gumbel version is the exponential mechanism of the theory, which picks a candidate with probability growing exponentially in its score. Its map is the square of that same ratio, divided by eight. Negating the scores selects the smallest, at no extra cost. The constructor `make_noisy_top_k` releases the best $k$ indices in order, for $k$ times the cost.

```text
noisy max: metric=LInfDistance(T=f64) measure=MaxDivergence
noisy max exponential scale 2: map(1.0)=1.0
noisy max gumbel scale 2: map(1.0)=0.125 measure=ZeroConcentratedDivergence
noisy max monotonic scale 2: exponential map(1.0)=0.5 gumbel map(1.0)=0.03125
noisy min scale 2: map(1.0)=1.0
noisy top 2 scale 2: exponential map(1.0)=2.0 gumbel map(1.0)=0.25
```

The older `make_report_noisy_max_gumbel` still exists and carries the pure-DP map. But the installed version marks it deprecated and points to `make_noisy_max`. The exponential mechanism, why selection costs twice what a count costs, and why exponential noise beats Gumbel under pure DP are in [the full edition's chapter on selection](../../book-full/part03-dp-foundations/05-selection.md).

```text
report_noisy_max_gumbel scale 2: map(1.0)=1.0 measure=MaxDivergence
report_noisy_max_gumbel: Call to deprecated function (or staticmethod) make_report_noisy_max_gumbel
```

### A private median of `HWUSUAL`

Selection is how OpenDP releases a quantile. The candidates are a list of possible answers, here every whole number of hours from 1 to 98. The transformation `make_quantile_score_candidates` from the previous chapter scores each one by how far it is from splitting the data in half. The best candidate has the smallest score, so the chain ends with a noisy max over negated scores. The released index is looked up in the candidate list afterwards, which is post-processing. The constructor `make_private_quantile` does the three steps in one call.

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

The data are the usual hours of the people at work, with the codes for varying and not applicable removed. A row filter keeps a person's contribution at most 36 rows, so the input distance is still 36 under the symmetric distance. The scores are integers, and each one moves by at most the input distance, so their L-infinity sensitivity is 36.

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

Two scales appear because the one-step constructor doubles the scale you pass. The scores of a median are integers scaled by two, the denominator of one half. So `make_private_quantile` multiplies the scale by that same two before handing it to noisy max, and the two maps agree at $\varepsilon = 1$. Read the scores before trusting the release. The best candidate beats the runner-up by over a thousand noise scales, so the mechanism returns 37 hours on practically every run, and the test checks for exactly that. On a smaller dataset the gaps shrink and the release starts to wander among nearby candidates. That wandering is the error you should expect from a private quantile.

## Randomized response {#sec-meas-rr}

Every mechanism so far runs on a trusted server that holds the data. Randomized response runs on the respondent's own device before the answer is sent, so that no server ever sees the truth. That is the local model of DP, in which the privacy unit is one answer. The metric is the discrete distance: two inputs are at distance 1 if they differ and 0 if they do not. OpenDP provides the randomizers only, and leaves the transport and the aggregation to you.

```{.python excerpt="meas-rr"}
def randomized_response_mechanisms(prob: float) -> tuple[dp.Measurement, dp.Measurement]:
    """Randomized response on one yes/no answer and on one ILOSTAT category, each kept truthful with probability prob."""
    boolean = dp.m.make_randomized_response_bool(prob=prob)
    categorical = dp.m.make_randomized_response(categories=[1, 2, 3, 9], prob=prob)
    return boolean, categorical
```

The boolean version keeps the true answer with the probability you pass and flips it otherwise. Its map at distance 1 is the logarithm of the odds of telling the truth, so a probability of three quarters costs about 1.1. A probability below one half is refused, because the answer would then be more often false than true. The categorical version keeps the truth with the same probability and otherwise picks one of the other categories uniformly. Its map grows with the logarithm of the number of alternatives, so four `ILOSTAT` codes cost twice what a yes/no answer costs.

```text
randomized response bool: domain=AtomDomain(T=bool) metric=DiscreteDistance() measure=MaxDivergence
randomized response bool prob 0.75: map(1)=1.0986 ln(0.75/0.25)=1.0986
randomized response bool prob 0.9: map(1)=2.1972
randomized response bool prob 0.4: probability must be within [0.5, 1]
randomized response 4 categories prob 0.75: map(1)=2.1972 ln(3*0.75/0.25)=2.1972
```

A single randomized answer is close to useless, and that is the point. The answers become useful in aggregate, after a correction for the known flipping rate, which the documentation derives for both versions. The third randomizer, for a bit vector, is the primitive behind RAPPOR, the first large deployment of local DP. The domain bounds how many bits can be set, and each bit is rerandomized with probability `f`. The map is then twice the weight times the logarithm of the odds of a bit surviving.

```{.python excerpt="meas-bitvec"}
def bitvector_mechanism(f: float, max_weight: int) -> dp.Measurement:
    """Randomized response on a bit vector with at most max_weight ones, each bit rerandomized with probability f."""
    return dp.m.make_randomized_response_bitvec(dp.bitvector_domain(max_weight=max_weight), dp.discrete_distance(), f=f)
```

```text
bit vector f=0.8 max_weight=3: domain=BitVectorDomain(weight=3) map(1)=2.4328 formula=2.4328
```

The script sends the documentation's 24-bit vector through the randomizer ten thousand times. Then it runs `debias_randomized_response_bitvec` over the answers. The three set bits come back near ten thousand and the others near zero, with a spread of a few hundred. That is what the correction can do at that many reports. Local DP buys its trust at a steep price in accuracy, and [the full edition's chapter on trust models](../../book-full/part03-dp-foundations/09-trust-models.md) quantifies it.

The releases of the run behind this page close the chapter. The additive releases sit within a few units of 100, the thresholded count kept all four codes, the projection answered the unknown name with zero, and both medians are 37.

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
