"""Measurements with OpenDP: the chapter "Measurements" of the OpenDP book (Part 3).

    uv run python code/part03-framework/opendp_measurements.py

Follows the documentation's Measurements pages: the additive noise mechanisms (Laplace, Gaussian, geometric and the
general make_noise, on scalars and vectors, integers and floats, with the granularity parameter k), the canonical
noise mechanism, the thresholded noise mechanisms, the approximate Laplace projection, the noisy max mechanisms with
the private quantile built on them, and randomized response. For each mechanism it builds the measurement, prints
its input metric and output measure, evaluates its privacy map and checks it. It prints every number the page
quotes, on labelled lines, in the order of the page's sections.

Data: the France Labour Force Survey sample (3,811,932 quarterly responses from 1,682,306 people), loaded through
code/practitioner_datasets.py, which caches it in data/ as Parquet; the OpenDP documentation reads the same file
through dp.examples.get_france_lfs_path(). The thresholded count is over the ILOSTAT categories, and the private
median is of HWUSUAL among the people at work (ILOSTAT 1, between 1 and 98 hours). Every other input is a toy
literal from the documentation. The privacy unit is a person's 36 possible rows (9 years by 4 quarters).

What is random: OpenDP draws its noise from a cryptographic generator that cannot be seeded, so every released value
changes from run to run; they are printed under "released (one run)" and the test checks them against the true
values within three times their 95% accuracy or a bound the page states. The privacy maps, checks, error messages,
thresholds, scores and true values are deterministic. The seed only names the run; nothing here samples with it.

Library versions: opendp[polars] 0.16.0 with polars 1.36.1.
"""

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import opendp.prelude as dp
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

dp.enable_features("contrib")

CONTRIBUTIONS = 36  # public: a person answers at most once per quarter over 9 years
EPSILON = 1.0
DELTA = 1e-7
HOURS = [float(h) for h in range(1, 99)]  # candidate medians of HWUSUAL: 1 to 98 hours
NAMES = {"Michael": 750, "James": 500, "Sharon": 50}  # the documentation's toy for the projection
SCORES = [0.0, 2.0, 4.0, 6.0]  # the documentation's toy for noisy max
BITS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  # the documentation's bit vector


def error_of(exc: Exception) -> str:
    """The first line of an OpenDP error, without the Rust variant around it."""
    text = str(exc).strip()
    if "(" in text and text.endswith(")"):
        text = text[text.index("(") + 1 : -1].strip('"')
    return text.splitlines()[0]


# ---------------------------------------------------------------------------------------------------------------
# Additive noise
# ---------------------------------------------------------------------------------------------------------------


# excerpt: meas-scalar
def scalar_mechanisms(scale: float) -> tuple[dp.Measurement, dp.Measurement]:
    """Laplace and Gaussian noise on one float; the input space is the same, the output measures differ."""
    space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
    laplace = space >> dp.m.then_laplace(scale=scale)  # output measure: MaxDivergence, read as epsilon
    gaussian = space >> dp.m.then_gaussian(scale=scale)  # output measure: ZeroConcentratedDivergence, read as rho
    return laplace, gaussian


# end excerpt


# excerpt: meas-vector
def vector_mechanisms(scale: float) -> tuple[dp.Measurement, dp.Measurement]:
    """The same two mechanisms on a vector of floats: Laplace wants an L1 bound, Gaussian an L2 bound."""
    domain = dp.vector_domain(dp.atom_domain(T=float, nan=False))
    laplace = (domain, dp.l1_distance(T=float)) >> dp.m.then_laplace(scale=scale)
    gaussian = (domain, dp.l2_distance(T=float)) >> dp.m.then_gaussian(scale=scale)
    return laplace, gaussian


# end excerpt


def print_additive() -> dict[str, Any]:
    """Section 1: the additive noise mechanisms; returns the releases of one run."""
    released: dict[str, Any] = {}
    laplace, gaussian = scalar_mechanisms(scale=1.0)
    print(f"laplace float: map(1.0)={laplace.map(1.0)} measure={laplace.output_measure}")
    print(f"gaussian float: map(1.0)={gaussian.map(1.0)} measure={gaussian.output_measure}")
    print(f"laplace float: check(1.0, 1.0)={laplace.check(1.0, 1.0)} check(2.0, 1.0)={laplace.check(2.0, 1.0)}")
    released["laplace on 100.0"] = laplace(100.0)
    released["gaussian on 100.0"] = gaussian(100.0)

    int_space = dp.atom_domain(T=int), dp.absolute_distance(T=int)
    laplace_int = int_space >> dp.m.then_laplace(scale=1.0)
    gaussian_int = int_space >> dp.m.then_gaussian(scale=1.0)
    print(f"laplace int: map(1)={laplace_int.map(1)} output type={type(laplace_int(100)).__name__}")
    print(f"gaussian int: map(1)={gaussian_int.map(1)}")
    released["discrete laplace on 100"] = laplace_int(100)

    laplace_vec, gaussian_vec = vector_mechanisms(scale=1.0)
    print(f"laplace vector: metric={laplace_vec.input_metric} map(1.0)={laplace_vec.map(1.0)}")
    print(f"gaussian vector: metric={gaussian_vec.input_metric} map(1.0)={gaussian_vec.map(1.0)}")
    print(f"gaussian vector: map(2.0)={gaussian_vec.map(2.0)} map(4.0)={gaussian_vec.map(4.0)}")
    try:
        laplace_vec(1.0)
    except TypeError as err:
        print(f"laplace vector on a scalar: {err}")
    try:
        _ = (dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.l1_distance(T=float)) >> dp.m.then_gaussian(1.0)
    except dp.OpenDPException as err:
        print(f"gaussian on an L1 space: {error_of(err)}")
    released["laplace vector on [0.0, 2.0, 2.0]"] = [round(float(v), 2) for v in laplace_vec([0.0, 2.0, 2.0])]

    space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
    for k in (-1074, -1, 0, 1):
        mech = dp.m.make_laplace(*space, scale=1.0, k=k)
        print(f"laplace k={k}: map(0.0)={mech.map(0.0)} map(1.0)={mech.map(1.0)}")
    try:
        dp.m.make_laplace(*int_space, scale=1.0, k=-1)
    except dp.OpenDPException as err:
        print(f"laplace k on integers: {error_of(err)}")
    coarse = dp.m.make_laplace(*space, scale=1.0, k=-1)
    released["laplace k=-1 on 100.0, five draws"] = [float(coarse(100.0)) for _ in range(5)]

    # excerpt: meas-geometric
    geometric = int_space >> dp.m.then_geometric(scale=1.0)
    bounded = int_space >> dp.m.then_geometric(scale=1.0, bounds=(0, 10))  # runs in constant time; clamps the output
    # end excerpt
    print(f"geometric: map(1)={geometric.map(1)} measure={geometric.output_measure}")
    print(f"geometric bounds (0, 10): map(1)={bounded.map(1)}")
    try:
        _ = space >> dp.m.then_geometric(scale=1.0)
    except dp.OpenDPException as err:
        print(f"geometric on floats: {error_of(err)}")
    released["bounded geometric on 10, five draws"] = [bounded(10) for _ in range(5)]

    # excerpt: meas-noise
    noise_pure = space >> dp.m.then_noise(dp.max_divergence(), scale=1.0)  # the Laplace mechanism
    noise_zcdp = space >> dp.m.then_noise(dp.zero_concentrated_divergence(), scale=1.0)  # the Gaussian mechanism
    # end excerpt
    print(f"make_noise MaxDivergence: map(1.0)={noise_pure.map(1.0)} measure={noise_pure.output_measure}")
    print(f"make_noise ZeroConcentratedDivergence: map(1.0)={noise_zcdp.map(1.0)} measure={noise_zcdp.output_measure}")
    return released


# ---------------------------------------------------------------------------------------------------------------
# Canonical noise
# ---------------------------------------------------------------------------------------------------------------


# excerpt: meas-canonical
def canonical_mechanism(d_in: float, epsilon: float, delta: float) -> dp.Measurement:
    """Canonical noise on one float, calibrated so that inputs at most d_in apart cost exactly (epsilon, delta)."""
    space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
    return space >> dp.m.then_canonical_noise(d_in=d_in, d_out=(epsilon, delta))


# end excerpt


def print_canonical() -> dict[str, Any]:
    """Section 2: the canonical noise mechanism; returns the releases of one run."""
    canonical = canonical_mechanism(d_in=1.0, epsilon=EPSILON, delta=DELTA)
    print(f"canonical noise: measure={canonical.output_measure}")
    print(f"canonical noise: map(1.0)={canonical.map(1.0)} map(0.5)={canonical.map(0.5)}")
    print(
        f"canonical noise: check(1.0, (1.0, 1e-07))={canonical.check(1.0, (1.0, 1e-7))}"
        f" check(1.0, (0.9, 1e-07))={canonical.check(1.0, (0.9, 1e-7))}"
    )
    try:
        canonical.map(2.0)
    except dp.OpenDPException as err:
        print(f"canonical noise map(2.0): {error_of(err)}")
    return {"canonical noise on 100.0": canonical(100.0)}


# ---------------------------------------------------------------------------------------------------------------
# Thresholded noise
# ---------------------------------------------------------------------------------------------------------------


# excerpt: meas-threshold
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


# end excerpt


# excerpt: meas-ilostat
def ilostat_counts(frame: pl.LazyFrame) -> dict[int, int]:
    """The number of responses in each ILOSTAT category, keyed by the category: the input of the thresholded count."""
    counts = frame.group_by("ILOSTAT").len().sort("ILOSTAT").collect()
    return {int(key): int(count) for key, count in zip(counts["ILOSTAT"], counts["len"])}


def ilostat_mechanism(scale: float, threshold: int) -> dp.Measurement:
    """Discrete Laplace noise on integer counts keyed by an integer category, with a threshold on the noisy count."""
    domain = dp.map_domain(dp.atom_domain(T=int), dp.atom_domain(T=int))
    metric = dp.l01inf_distance(dp.absolute_distance(T=int))
    return dp.m.make_laplace_threshold(domain, metric, scale=scale, threshold=threshold)


# end excerpt


def print_thresholded(frame: pl.LazyFrame) -> dict[str, Any]:
    """Section 3: the thresholded noise mechanisms; returns the releases of one run."""
    released: dict[str, Any] = {}
    laplace, gaussian = threshold_mechanisms(scale=1.0, threshold=20.0)
    print(f"laplace threshold: metric={laplace.input_metric} measure={laplace.output_measure}")
    sensitivity = (1, 1.0, 1.0)
    epsilon, delta = laplace.map(sensitivity)
    print(f"laplace threshold scale 1 threshold 20: map((1, 1.0, 1.0))=({epsilon}, {delta:.4e})")
    rho, delta_gauss = gaussian.map(sensitivity)
    print(f"gaussian threshold scale 1 threshold 20: map((1, 1.0, 1.0))=({rho}, {delta_gauss:.4e})")
    print(f"gaussian threshold: measure={gaussian.output_measure}")
    gaussian_approx = dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(gaussian), delta=delta)
    eps_gauss, _ = gaussian_approx.map(sensitivity)
    print(f"gaussian threshold as approximate DP at delta={delta:.4e}: epsilon={eps_gauss:.4f}")
    spread = (100, 10.0, 0.001)
    eps_spread, delta_spread = laplace.map(spread)
    eps_spread_gauss, _ = gaussian_approx.map(spread)
    print(
        f"spread (100, 10.0, 0.001): laplace ({eps_spread:.4f}, {delta_spread:.4e}) gaussian epsilon={eps_spread_gauss:.4f}"
    )
    released["laplace threshold on {a: 0, b: 20, c: 40}"] = laplace({"a": 0.0, "b": 20.0, "c": 40.0})

    int_domain = dp.map_domain(dp.atom_domain(T=str), dp.atom_domain(T=int))
    int_metric = dp.l01inf_distance(dp.absolute_distance(T=int))
    discrete = dp.m.make_laplace_threshold(int_domain, int_metric, scale=1.0, threshold=10)
    eps_int, delta_int = discrete.map((1, 1, 1))
    print(f"discrete laplace threshold scale 1 threshold 10: map((1, 1, 1))=({eps_int}, {delta_int:.4e})")
    general = dp.m.make_noise_threshold(int_domain, int_metric, dp.approximate(dp.max_divergence()), 1.0, 10)
    eps_gen, delta_gen = general.map((1, 1, 1))
    print(f"make_noise_threshold Approximate(MaxDivergence): map((1, 1, 1))=({eps_gen}, {delta_gen:.4e})")
    try:
        dp.m.make_noise_threshold(int_domain, int_metric, dp.max_divergence(), 1.0, 10)
    except dp.OpenDPException as err:
        print(f"make_noise_threshold MaxDivergence: {error_of(err)}")

    counts = ilostat_counts(frame)
    print(f"ILOSTAT counts: {counts}")
    d_in = (CONTRIBUTIONS, CONTRIBUTIONS, CONTRIBUTIONS)
    scale = CONTRIBUTIONS / EPSILON
    accuracy = dp.discrete_laplacian_scale_to_accuracy(scale=scale, alpha=0.05)
    print(f"ILOSTAT d_in: {d_in} scale: {scale:g} 95% accuracy: {accuracy:.1f}")
    for threshold in (500, 1000, 1500):
        eps_t, delta_t = ilostat_mechanism(scale, threshold).map(d_in)
        print(f"ILOSTAT threshold {threshold}: map={eps_t}, delta={delta_t:.4e}")
    smallest = int(
        dp.binary_search(lambda t: ilostat_mechanism(scale, int(t)).map(d_in)[1] <= DELTA, (36, 100_000), T=int)
    )
    print(f"ILOSTAT smallest threshold with delta <= {DELTA:g}: {smallest}")
    try:
        ilostat_mechanism(scale, 10).map(d_in)
    except dp.OpenDPException as err:
        print(f"ILOSTAT threshold 10: {error_of(err)}")
    released["ILOSTAT counts"] = dict(sorted(ilostat_mechanism(scale, smallest)(counts).items()))
    return released


# ---------------------------------------------------------------------------------------------------------------
# The approximate Laplace projection
# ---------------------------------------------------------------------------------------------------------------


# excerpt: meas-alp
def projection_mechanism(scale: float) -> dp.Measurement:
    """A private projection of counts keyed by strings; releases a queryable that answers for any key, known or not."""
    domain = dp.map_domain(dp.atom_domain(T=str), dp.atom_domain(T=int))
    metric = dp.l01inf_distance(dp.absolute_distance(T=int))
    return dp.m.make_alp_queryable(domain, metric, scale=scale, total_limit=50_000, value_limit=1_000)


# end excerpt


def print_projection() -> dict[str, Any]:
    """Section 4: the approximate Laplace projection; returns the releases of one run."""
    projection = projection_mechanism(scale=1.0)
    print(f"alp scale 1: map((1, 1, 1))={projection.map((1, 1, 1))} measure={projection.output_measure}")
    print(f"alp scale 2: map((1, 1, 1))={projection_mechanism(scale=2.0).map((1, 1, 1))}")
    print(f"alp scale 1: map((2, 2, 1))={projection.map((2, 2, 1))}")
    queryable = projection(NAMES)
    return {"alp answers for Michael, James, Sharon, Lancelot": [queryable(n) for n in [*NAMES, "Lancelot"]]}


# ---------------------------------------------------------------------------------------------------------------
# Noisy max and the private quantile
# ---------------------------------------------------------------------------------------------------------------


# excerpt: meas-noisy-max
def noisy_max_mechanisms(scale: float) -> tuple[dp.Measurement, dp.Measurement]:
    """Select the index of the largest score: exponential noise under pure DP, Gumbel noise under zCDP."""
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.linf_distance(T=float)
    exponential = space >> dp.m.then_noisy_max(dp.max_divergence(), scale=scale)
    gumbel = space >> dp.m.then_noisy_max(dp.zero_concentrated_divergence(), scale=scale)
    return exponential, gumbel


# end excerpt


# excerpt: meas-median
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


# end excerpt


def print_noisy_max(frame: pl.LazyFrame) -> dict[str, Any]:
    """Section 5: noisy max, top-k and the private median; returns the releases of one run."""
    released: dict[str, Any] = {}
    exponential, gumbel = noisy_max_mechanisms(scale=2.0)
    print(f"noisy max: metric={exponential.input_metric} measure={exponential.output_measure}")
    print(f"noisy max exponential scale 2: map(1.0)={exponential.map(1.0)}")
    print(f"noisy max gumbel scale 2: map(1.0)={gumbel.map(1.0)} measure={gumbel.output_measure}")
    domain = dp.vector_domain(dp.atom_domain(T=float, nan=False))
    monotonic = dp.linf_distance(T=float, monotonic=True)
    mono_exp = dp.m.make_noisy_max(domain, monotonic, dp.max_divergence(), scale=2.0)
    mono_gum = dp.m.make_noisy_max(domain, monotonic, dp.zero_concentrated_divergence(), scale=2.0)
    print(f"noisy max monotonic scale 2: exponential map(1.0)={mono_exp.map(1.0)} gumbel map(1.0)={mono_gum.map(1.0)}")
    negated = dp.m.make_noisy_max(domain, dp.linf_distance(T=float), dp.max_divergence(), scale=2.0, negate=True)
    print(f"noisy min scale 2: map(1.0)={negated.map(1.0)}")
    top2 = dp.m.make_noisy_top_k(domain, dp.linf_distance(T=float), dp.max_divergence(), k=2, scale=2.0)
    top2_gum = dp.m.make_noisy_top_k(
        domain, dp.linf_distance(T=float), dp.zero_concentrated_divergence(), k=2, scale=2.0
    )
    print(f"noisy top 2 scale 2: exponential map(1.0)={top2.map(1.0)} gumbel map(1.0)={top2_gum.map(1.0)}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = dp.m.make_report_noisy_max_gumbel(domain, dp.linf_distance(T=float), scale=2.0)
    print(f"report_noisy_max_gumbel scale 2: map(1.0)={old.map(1.0)} measure={old.output_measure}")
    print(f"report_noisy_max_gumbel: {str(caught[0].message).split('. ')[0]}")
    released["noisy max on [0, 2, 4, 6]"] = exponential(SCORES)
    released["noisy min on [0, 2, 4, 6]"] = negated(SCORES)
    released["noisy top 2 on [0, 2, 4, 6]"] = top2(SCORES)

    hours = hours_of_the_working(frame)
    true_median = float(np.median(hours))
    print(f"HWUSUAL at work, 1 to 98 hours: n={len(hours):,} true median={true_median:g}")
    scale = CONTRIBUTIONS / EPSILON
    scores, by_hand = median_by_hand(scale=2 * scale)
    print(f"quantile scores: metric={scores.output_metric} map({CONTRIBUTIONS})={scores.map(CONTRIBUTIONS)}")
    score_values = scores(hours)
    order = sorted(range(len(HOURS)), key=lambda i: score_values[i])
    best, second = order[0], order[1]
    print(f"best candidate: {HOURS[best]:g} hours, score {score_values[best]:,}")
    print(f"second candidate: {HOURS[second]:g} hours, score {score_values[second]:,}")
    print(f"gap in noise scales: {(score_values[second] - score_values[best]) / (2 * scale):,.0f}")
    print(f"scores >> noisy max scale {2 * scale:g}: map({CONTRIBUTIONS})={by_hand.map(CONTRIBUTIONS)}")
    median = median_mechanism(scale=scale)
    print(f"make_private_quantile scale {scale:g}: map({CONTRIBUTIONS})={median.map(CONTRIBUTIONS)}")
    zcdp = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()
    median_zcdp = zcdp >> dp.m.then_private_quantile(dp.zero_concentrated_divergence(), HOURS, 0.5, scale)
    print(f"make_private_quantile zCDP scale {scale:g}: map({CONTRIBUTIONS})={median_zcdp.map(CONTRIBUTIONS)}")
    released["median by hand"] = f"{HOURS[by_hand(hours)]:g} hours"
    released["median by make_private_quantile"] = f"{median(hours):g} hours"
    return released


# ---------------------------------------------------------------------------------------------------------------
# Randomized response
# ---------------------------------------------------------------------------------------------------------------


# excerpt: meas-rr
def randomized_response_mechanisms(prob: float) -> tuple[dp.Measurement, dp.Measurement]:
    """Randomized response on one yes/no answer and on one ILOSTAT category, each kept truthful with probability prob."""
    boolean = dp.m.make_randomized_response_bool(prob=prob)
    categorical = dp.m.make_randomized_response(categories=[1, 2, 3, 9], prob=prob)
    return boolean, categorical


# end excerpt


# excerpt: meas-bitvec
def bitvector_mechanism(f: float, max_weight: int) -> dp.Measurement:
    """Randomized response on a bit vector with at most max_weight ones, each bit rerandomized with probability f."""
    return dp.m.make_randomized_response_bitvec(dp.bitvector_domain(max_weight=max_weight), dp.discrete_distance(), f=f)


# end excerpt


def print_randomized_response() -> dict[str, Any]:
    """Section 6: randomized response; returns the releases of one run."""
    released: dict[str, Any] = {}
    boolean, categorical = randomized_response_mechanisms(prob=0.75)
    print(
        f"randomized response bool: domain={boolean.input_domain} metric={boolean.input_metric} measure={boolean.output_measure}"
    )
    print(f"randomized response bool prob 0.75: map(1)={boolean.map(1):.4f} ln(0.75/0.25)={math.log(0.75 / 0.25):.4f}")
    print(f"randomized response bool prob 0.9: map(1)={dp.m.make_randomized_response_bool(prob=0.9).map(1):.4f}")
    try:
        dp.m.make_randomized_response_bool(prob=0.4)
    except dp.OpenDPException as err:
        print(f"randomized response bool prob 0.4: {error_of(err)}")
    print(
        f"randomized response 4 categories prob 0.75: map(1)={categorical.map(1):.4f} ln(3*0.75/0.25)={math.log(9):.4f}"
    )
    released["randomized response on True, ten draws"] = [boolean(True) for _ in range(10)]
    released["randomized response on category 1, ten draws"] = [categorical(1) for _ in range(10)]

    bitvector = bitvector_mechanism(f=0.8, max_weight=3)
    formula = 2 * 3 * math.log((2 - 0.8) / 0.8)
    print(
        f"bit vector f=0.8 max_weight=3: domain={bitvector.input_domain} map(1)={bitvector.map(1):.4f} formula={formula:.4f}"
    )
    data = np.packbits(BITS, bitorder="little").tobytes()
    one = np.unpackbits(np.frombuffer(bitvector(data), dtype=np.uint8), bitorder="little")
    released["bit vector release"] = "".join(str(int(b)) for b in one)
    answers = [bitvector(data) for _ in range(10_000)]
    frequencies = dp.m.debias_randomized_response_bitvec(answers, f=0.8)
    released["bit vector frequencies over 10,000 copies, bits 12, 20, 21"] = [
        round(frequencies[i]) for i in (12, 20, 21)
    ]
    released["bit vector frequencies, largest other bit"] = round(
        max(abs(frequencies[i]) for i in range(24) if i not in (12, 20, 21))
    )
    return released


# ---------------------------------------------------------------------------------------------------------------
# Summary: every mechanism at scale 1 and d_in = 1
# ---------------------------------------------------------------------------------------------------------------


def print_summary() -> None:
    """The map of each mechanism at scale 1 for inputs at distance 1, for the chapter's closing table."""
    float_space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
    int_space = dp.atom_domain(T=int), dp.absolute_distance(T=int)
    vector = dp.vector_domain(dp.atom_domain(T=float, nan=False))
    laplace_thr, gaussian_thr = threshold_mechanisms(scale=1.0, threshold=20.0)
    exponential, gumbel = noisy_max_mechanisms(scale=1.0)
    rows: list[tuple[str, Any]] = [
        ("make_laplace", (float_space >> dp.m.then_laplace(1.0)).map(1.0)),
        ("make_gaussian", (float_space >> dp.m.then_gaussian(1.0)).map(1.0)),
        ("make_geometric", (int_space >> dp.m.then_geometric(1.0)).map(1)),
        ("make_laplace_threshold (threshold 20)", laplace_thr.map((1, 1.0, 1.0))),
        ("make_gaussian_threshold (threshold 20)", gaussian_thr.map((1, 1.0, 1.0))),
        ("make_alp_queryable", projection_mechanism(1.0).map((1, 1, 1))),
        ("make_noisy_max exponential", exponential.map(1.0)),
        ("make_noisy_max gumbel", gumbel.map(1.0)),
        (
            "make_noisy_top_k k=2 exponential",
            (vector, dp.linf_distance(T=float)) >> dp.m.then_noisy_top_k(dp.max_divergence(), 2, 1.0),
        ),
        ("make_private_quantile", median_mechanism(1.0).map(1)),
        ("make_randomized_response_bool prob 0.75", dp.m.make_randomized_response_bool(0.75).map(1)),
        ("make_randomized_response 4 categories prob 0.75", dp.m.make_randomized_response([1, 2, 3, 9], 0.75).map(1)),
        ("make_randomized_response_bitvec f=0.8 weight 3", bitvector_mechanism(0.8, 3).map(1)),
    ]
    for name, value in rows:
        if isinstance(value, dp.Measurement):
            value = value.map(1.0)
        if isinstance(value, tuple):
            print(f"summary {name}: ({value[0]:.4g}, {value[1]:.4e})")
        else:
            print(f"summary {name}: {value:.4g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="names the run; OpenDP's noise has no seed")
    parser.parse_args()
    frame = pl.scan_parquet(practitioner_datasets.france_lfs())

    print(f"opendp version: {dp.__version__}")
    released: dict[str, Any] = {}
    released.update(print_additive())
    released.update(print_canonical())
    released.update(print_thresholded(frame))
    released.update(print_projection())
    released.update(print_noisy_max(frame))
    released.update(print_randomized_response())
    print_summary()

    print("released (one run):")
    for name, value in released.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.2f}")
        else:
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
