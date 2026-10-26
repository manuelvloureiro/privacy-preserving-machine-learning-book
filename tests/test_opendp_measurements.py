"""code/part03-framework/opendp_measurements.py runs and prints what the OpenDP book's chapter "Measurements" quotes.

The deterministic lines (input metrics and output measures, privacy maps, checks, error messages, thresholds, the
quantile scores and the true values) are checked exactly. OpenDP's noise cannot be seeded, so the released values are
checked against the true values within three times their 95% accuracy where OpenDP gives one, and otherwise within a
bound stated here: the canonical noise within 20 (its tails are those of a Laplace with scale 1, so 20 is passed about
once in a billion runs), the projection's answers within 100 (runs give about 10), the bit-vector frequencies within
2,000 of their true values (about eight standard deviations over 10,000 copies), and the median exactly, because its
best score beats the runner-up by over a thousand noise scales.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part03-framework" / "opendp_measurements.py"

DETERMINISTIC = [
    "opendp version: 0.16.0",
    "laplace float: map(1.0)=1.0 measure=MaxDivergence",
    "gaussian float: map(1.0)=0.5 measure=ZeroConcentratedDivergence",
    "laplace float: check(1.0, 1.0)=True check(2.0, 1.0)=False",
    "laplace int: map(1)=1.0 output type=int",
    "gaussian int: map(1)=0.5",
    "laplace vector: metric=L1Distance(f64) map(1.0)=1.0",
    "gaussian vector: metric=L2Distance(f64) map(1.0)=0.5",
    "gaussian vector: map(2.0)=2.0 map(4.0)=8.0",
    "laplace vector on a scalar: Expected type is Vec<f64> but input data is not a list.",
    "gaussian on an L1 space: No match for concrete type L1Distance<f64>. See https://github.com/opendp/opendp/discussions/451.",
    "laplace k=-1074: map(0.0)=0.0 map(1.0)=1.0",
    "laplace k=-1: map(0.0)=0.5 map(1.0)=1.5",
    "laplace k=0: map(0.0)=1.0 map(1.0)=2.0",
    "laplace k=1: map(0.0)=2.0 map(1.0)=3.0",
    "laplace k on integers: k is only valid for domains over floats",
    "geometric: map(1)=1.0 measure=MaxDivergence",
    "geometric bounds (0, 10): map(1)=1.0",
    "geometric on floats: No match for concrete type f64. See https://github.com/opendp/opendp/discussions/451.",
    "make_noise MaxDivergence: map(1.0)=1.0 measure=MaxDivergence",
    "make_noise ZeroConcentratedDivergence: map(1.0)=0.5 measure=ZeroConcentratedDivergence",
    "canonical noise: measure=Approximate(MaxDivergence)",
    "canonical noise: map(1.0)=(1.0, 1e-07) map(0.5)=(1.0, 1e-07)",
    "canonical noise: check(1.0, (1.0, 1e-07))=True check(1.0, (0.9, 1e-07))=False",
    "canonical noise map(2.0): d_in from the map (2) must be in [0, 1]",
    "laplace threshold: metric=L01InfDistance(AbsoluteDistance(f64)) measure=Approximate(MaxDivergence)",
    "laplace threshold scale 1 threshold 20: map((1, 1.0, 1.0))=(1.0, 2.8014e-09)",
    "gaussian threshold scale 1 threshold 20: map((1, 1.0, 1.0))=(0.5, 1.1102e-16)",
    "gaussian threshold: measure=Approximate(ZeroConcentratedDivergence)",
    "gaussian threshold as approximate DP at delta=2.8014e-09: epsilon=6.3036",
    "spread (100, 10.0, 0.001): laplace (0.1000, 1.0316e-07) gaussian epsilon=0.0500",
    "discrete laplace threshold scale 1 threshold 10: map((1, 1, 1))=(1.0, 3.3190e-05)",
    "make_noise_threshold Approximate(MaxDivergence): map((1, 1, 1))=(1.0, 3.3190e-05)",
    "make_noise_threshold MaxDivergence: output_measure must be MaxDivergence or ZeroConcentratedDivergence",
    "ILOSTAT counts: {1: 1517011, 2: 155661, 3: 1450283, 9: 688977}",
    "ILOSTAT d_in: (36, 36, 36) scale: 36 95% accuracy: 108.3",
    "ILOSTAT threshold 500: map=1.0, delta=4.4835e-05",
    "ILOSTAT threshold 1000: map=1.0, delta=4.1667e-11",
    "ILOSTAT threshold 1500: map=1.0, delta=3.9968e-15",
    "ILOSTAT smallest threshold with delta <= 1e-07: 720",
    "ILOSTAT threshold 10: threshold (10) must not be smaller than l-infinity sensitivity 36",
    "alp scale 1: map((1, 1, 1))=1.0 measure=MaxDivergence",
    "alp scale 2: map((1, 1, 1))=2.0",
    "alp scale 1: map((2, 2, 1))=2.0",
    "noisy max: metric=LInfDistance(T=f64) measure=MaxDivergence",
    "noisy max exponential scale 2: map(1.0)=1.0",
    "noisy max gumbel scale 2: map(1.0)=0.125 measure=ZeroConcentratedDivergence",
    "noisy max monotonic scale 2: exponential map(1.0)=0.5 gumbel map(1.0)=0.03125",
    "noisy min scale 2: map(1.0)=1.0",
    "noisy top 2 scale 2: exponential map(1.0)=2.0 gumbel map(1.0)=0.25",
    "report_noisy_max_gumbel scale 2: map(1.0)=1.0 measure=MaxDivergence",
    "report_noisy_max_gumbel: Call to deprecated function (or staticmethod) make_report_noisy_max_gumbel",
    "HWUSUAL at work, 1 to 98 hours: n=1,490,113 true median=37",
    "quantile scores: metric=LInfDistance(T=u64) map(36)=36",
    "best candidate: 37 hours, score 9,111",
    "second candidate: 36 hours, score 95,652",
    "gap in noise scales: 1,202",
    "scores >> noisy max scale 72: map(36)=1.0",
    "make_private_quantile scale 36: map(36)=1.0",
    "make_private_quantile zCDP scale 36: map(36)=0.125",
    "randomized response bool: domain=AtomDomain(T=bool) metric=DiscreteDistance() measure=MaxDivergence",
    "randomized response bool prob 0.75: map(1)=1.0986 ln(0.75/0.25)=1.0986",
    "randomized response bool prob 0.9: map(1)=2.1972",
    "randomized response bool prob 0.4: probability must be within [0.5, 1]",
    "randomized response 4 categories prob 0.75: map(1)=2.1972 ln(3*0.75/0.25)=2.1972",
    "bit vector f=0.8 max_weight=3: domain=BitVectorDomain(weight=3) map(1)=2.4328 formula=2.4328",
    "summary make_laplace: 1",
    "summary make_gaussian: 0.5",
    "summary make_geometric: 1",
    "summary make_laplace_threshold (threshold 20): (1, 2.8014e-09)",
    "summary make_gaussian_threshold (threshold 20): (0.5, 1.1102e-16)",
    "summary make_alp_queryable: 1",
    "summary make_noisy_max exponential: 2",
    "summary make_noisy_max gumbel: 0.5",
    "summary make_noisy_top_k k=2 exponential: 4",
    "summary make_private_quantile: 1",
    "summary make_randomized_response_bool prob 0.75: 1.099",
    "summary make_randomized_response 4 categories prob 0.75: 2.197",
    "summary make_randomized_response_bitvec f=0.8 weight 3: 2.433",
    "  median by hand: 37 hours",
    "  median by make_private_quantile: 37 hours",
]

NUMBER = r"(-?[\d,]+(?:\.\d+)?)"
LAPLACE_1 = 2.9957  # dp.laplacian_scale_to_accuracy(1.0, 0.05)
DISCRETE_LAPLACE_1 = 3.0  # dp.discrete_laplacian_scale_to_accuracy(1.0, 0.05), rounded up
GAUSSIAN_1 = 1.9600  # dp.gaussian_scale_to_accuracy(1.0, 0.05)
ILOSTAT_ACCURACY = 108.3  # printed by the script for scale 36
ILOSTAT_TRUE = {1: 1_517_011, 2: 155_661, 3: 1_450_283, 9: 688_977}


def number(pattern: str, out: str) -> list[float]:
    m = re.search(pattern, out, re.MULTILINE)
    assert m, f"no line matches {pattern!r}"
    return [float(g.replace(",", "")) for g in m.groups()]


def released(name: str, out: str) -> str:
    m = re.search(rf"^  {re.escape(name)}: (.+)$", out, re.MULTILINE)
    assert m, f"no release named {name!r}"
    return m.group(1)


@pytest.fixture(scope="module")
def out() -> str:
    return subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout


@pytest.mark.libs
def test_deterministic_lines(out: str) -> None:
    lines = out.splitlines()
    for expected in DETERMINISTIC:
        assert expected in lines, f"missing line: {expected}"


@pytest.mark.libs
def test_additive_releases_near_truth(out: str) -> None:
    (laplace,) = number(rf"^  laplace on 100\.0: {NUMBER}$", out)
    assert abs(laplace - 100) <= 3 * LAPLACE_1
    (gaussian,) = number(rf"^  gaussian on 100\.0: {NUMBER}$", out)
    assert abs(gaussian - 100) <= 3 * GAUSSIAN_1
    (discrete,) = number(rf"^  discrete laplace on 100: {NUMBER}$", out)
    assert discrete == int(discrete) and abs(discrete - 100) <= 3 * DISCRETE_LAPLACE_1
    vector = [float(v) for v in released("laplace vector on [0.0, 2.0, 2.0]", out).strip("[]").split(", ")]
    assert len(vector) == 3
    for value, true in zip(vector, (0.0, 2.0, 2.0)):
        assert abs(value - true) <= 3 * LAPLACE_1
    coarse = [float(v) for v in released("laplace k=-1 on 100.0, five draws", out).strip("[]").split(", ")]
    assert len(coarse) == 5
    for value in coarse:
        assert (2 * value) == int(2 * value), "k=-1 draws lie on a grid of 0.5"
        assert abs(value - 100) <= 3 * LAPLACE_1 + 0.5
    bounded = [int(v) for v in released("bounded geometric on 10, five draws", out).strip("[]").split(", ")]
    assert len(bounded) == 5
    for value in bounded:
        assert 0 <= value <= 10 and abs(value - 10) <= 3 * DISCRETE_LAPLACE_1
    (canonical,) = number(rf"^  canonical noise on 100\.0: {NUMBER}$", out)
    assert abs(canonical - 100) <= 20


@pytest.mark.libs
def test_thresholded_releases_near_truth(out: str) -> None:
    toy = released("laplace threshold on {a: 0, b: 20, c: 40}", out)
    assert "'c'" in toy, "a count of 40 passes a threshold of 20 at scale 1 on all but one run in a billion"
    assert "'a'" not in toy, "a count of 0 fails a threshold of 20 at scale 1 on all but one run in a billion"
    counts = released("ILOSTAT counts", out)
    for key, true in ILOSTAT_TRUE.items():
        m = re.search(rf"\b{key}: (\d+)", counts)
        assert m, f"ILOSTAT {key} missing from the release: every true count is far above the threshold"
        assert abs(int(m.group(1)) - true) <= 3 * ILOSTAT_ACCURACY, key


@pytest.mark.libs
def test_projection_answers_near_truth(out: str) -> None:
    answers = [
        float(v) for v in released("alp answers for Michael, James, Sharon, Lancelot", out).strip("[]").split(", ")
    ]
    for value, true in zip(answers, (750, 500, 50, 0)):
        assert abs(value - true) <= 100


@pytest.mark.libs
def test_selection_releases_are_valid(out: str) -> None:
    assert released("noisy max on [0, 2, 4, 6]", out) in {"0", "1", "2", "3"}
    assert released("noisy min on [0, 2, 4, 6]", out) in {"0", "1", "2", "3"}
    top2 = released("noisy top 2 on [0, 2, 4, 6]", out).strip("[]").split(", ")
    assert len(top2) == 2 and len(set(top2)) == 2 and set(top2) <= {"0", "1", "2", "3"}


@pytest.mark.libs
def test_randomized_response_releases_are_valid(out: str) -> None:
    booleans = released("randomized response on True, ten draws", out).strip("[]").split(", ")
    assert len(booleans) == 10 and set(booleans) <= {"True", "False"}
    categories = released("randomized response on category 1, ten draws", out).strip("[]").split(", ")
    assert len(categories) == 10 and set(categories) <= {"1", "2", "3", "9"}
    bits = released("bit vector release", out)
    assert re.fullmatch(r"[01]{24}", bits)
    frequencies = [
        float(v)
        for v in released("bit vector frequencies over 10,000 copies, bits 12, 20, 21", out).strip("[]").split(", ")
    ]
    assert len(frequencies) == 3
    for value in frequencies:
        assert abs(value - 10_000) <= 2_000
    (largest_other,) = number(rf"^  bit vector frequencies, largest other bit: {NUMBER}$", out)
    assert largest_other <= 2_000
