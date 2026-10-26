"""code/part03-framework/opendp_utilities.py runs and prints what the OpenDP book's chapter on utilities and the
Context API quotes.

The deterministic lines (type descriptors, error messages, the scales the searches find, the privacy maps, the
budget splits and the accuracies) are checked exactly. OpenDP's noise cannot be seeded, so the released counts are
checked against the true values within three times their 95% accuracy.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part03-framework" / "opendp_utilities.py"

DETERMINISTIC = [
    "opendp version: 0.16.0",
    "int means i32, float means f64",
    "count output: VectorDomain(AtomDomain(nan=true, T=f64)) -> AtomDomain(T=i32)",
    "map(1.5) on an integer metric: TypeError: inferred type is f64, expected i32",
    "map(2) on an integer metric: 2.0",
    "float d_in on the count: TypeError: inferred type is f64, expected u32",
    "integer d_in on the count: 0.3333",
    "scalar epsilon on an approximate measure: TypeError: Cannot coerce a non-tuple type to a tuple",
    "pair on an approximate measure: True",
    "search: scale=3.0000000000000004 chain.map(1.0)=0.3333 chain.check(1.0, share)=True",
    "search: largest d_in at scale 1 within epsilon 1: 1.0",
    "search: gaussian scale for rho=1 at d_in=1: 0.7071",
    "search by hand: scale=3.000000 after 27 steps",
    "search with bounds (0, 100): 3.0000000000000004",
    "search with bounds (0, 1): ValueError: the decision boundary of the predicate is outside the bounds",
    "search on a constant predicate: ValueError: unable to infer bounds",
    'serialized: 2152 characters, starting {"__function__": "make_chain_mt",',
    "restored: map(1)=0.3333 same input space: True",
    "param before a round trip: 3.0000000000000004",
    "param after a round trip: None",
    "serialize a context: OpenDP JSON Encoder does not handle instances of <class 'opendp.context.Context'>",
    "plan: 1792 bytes",
    "plan on the server: scale=36.0 95% accuracy=108.3429",
    "context accountant: VectorDomain(AtomDomain(nan=true, T=f64)) SymmetricDistance() MaxDivergence",
    "context d_in: 1  d_mids: [0.3333, 0.3333, 0.3333]",
    "context accountant map(1): 1.0000",
    "by hand accountant map(1): 1.0000  queryable: Queryable(Q=AnyMeasurement)",
    "d_mids by weights [1, 2, 1]: [0.25, 0.5, 0.25]",
    "query chain: tuple -> Transformation -> PartialChain",
    "partial chain: True",
    "resolve a transformation: ValueError: Query is not yet a measurement or odometer.",
    "param of .laplace(): 3.0000000000000004",
    "param of .laplace(scale=3.0): None",
    "two missing parameters: ValueError: gaussian is missing 1 parameter(s).",
    "resolve: context param=3.0000000000000004 by hand=3.0000000000000004 equal=True",
    "d_mids after context(measurement): [0.3333, 0.3333]",
    "inner context: d_in=1 d_out=0.3333 d_mids=[0.1667, 0.1667]",
    "inner count scale: 6.0000",
    "outer d_mids after the inner compositor: [0.3333]",
    "approx share: (0.3333, 3.3e-08)",
    "laplace under (epsilon, delta): Approximate(MaxDivergence) map(1)=(0.3333, 0.0e+00)",
    "gaussian under (epsilon, delta): Approximate(MaxDivergence) map(1)=(0.3333, 3.3e-08)",
    "param after a cast: None",
    "laplace under rho: ZeroConcentratedDivergence map(1)=0.0333",
    "gaussian under rho: scale=3.8730",
    "gaussian under epsilon: ValueError: Unable to cast measure from ZeroConcentratedDivergence to MaxDivergence",
]

NUMBER = r"(-?[\d,]+(?:\.\d+)?)"


def number(pattern: str, out: str) -> list[float]:
    m = re.search(pattern, out, re.MULTILINE)
    assert m, f"no line matches {pattern!r}"
    return [float(g.replace(",", "")) for g in m.groups()]


@pytest.fixture(scope="module")
def out() -> str:
    return subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout


@pytest.mark.libs
def test_deterministic_lines(out: str) -> None:
    lines = out.splitlines()
    for expected in DETERMINISTIC:
        assert expected in lines, f"missing line: {expected}"


@pytest.mark.libs
def test_released_values_near_truth(out: str) -> None:
    (rows,) = number(rf"^  survey rows {NUMBER}  \(true 3811932, within 108\.3 with probability 0\.95\)$", out)
    assert abs(rows - 3_811_932) <= 3 * 108.3429
    (by_hand,) = number(rf"^  count by hand {NUMBER}  \(true 100, within 9\.4 with probability 0\.95\)$", out)
    assert abs(by_hand - 100) <= 3 * 9.4457
    (through,) = number(rf"^  count through the context {NUMBER}  \(true 100\)$", out)
    assert abs(through - 100) <= 3 * 9.4457
