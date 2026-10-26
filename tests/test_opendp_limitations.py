"""code/part04-extending/opendp_limitations.py runs and prints what the OpenDP book's limitations chapter quotes.

The deterministic lines (the sensitivities, every error message and warning, the inferred schemas, the scales,
the accuracy and the Rust stack trace) are checked exactly. OpenDP's noise cannot be seeded, so the one released
count is checked against the true count within three times its 95% accuracy.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part04-extending" / "opendp_limitations.py"

FEATURE_URL = "See https://docs.opendp.org/en/stable/api/user-guide/#feature-listing, then call enable_features"
TROUBLE_URL = (
    "See https://docs.opendp.org/en/stable/getting-started/trouble-shooting.html#inferred-type-is-xxx-expected-yyy"
)
MISMATCH = "DomainMismatch(\"Intermediate domains don't match. See https://github.com/opendp/opendp/discussions/297"

DETERMINISTIC = [
    "opendp version: 0.16.0",
    "float sum of 36 rows in [0, 80]: sensitivity 2880.0000007451",
    "int sum of 36 rows in [0, 80]: sensitivity 2880",
    f'laplace without contrib: OpenDPException: Attempted to use function that requires "contrib", but not enabled. {FEATURE_URL}("contrib")',
    f'plugin without honest-but-curious: OpenDPException: Attempted to use function that requires "contrib", "honest-but-curious", but not enabled. {FEATURE_URL}("contrib", "honest-but-curious")',
    "plugin with honest-but-curious: map(1.0)=1.0",
    "float laplace without idealized-numerics: no error",
    'enable_features("floating-point"): DeprecationWarning: "floating-point" is deprecated. Use "idealized-numerics" instead.',
    'assert_features("idealized-numerics") with the alias on: no error',
    f'assert_features("idealized-numerics") after disabling it: OpenDPException: Attempted to use function that requires "idealized-numerics", but not enabled. {FEATURE_URL}("idealized-numerics")',
    f'assert_features("contrib", "honest-but-curious"): OpenDPException: Attempted to use function that requires "contrib", "honest-but-curious", but not enabled. {FEATURE_URL}("contrib", "honest-but-curious")',
    "csv b'x\\n1\\n2\\n': column x inferred as Int64",
    "csv b'x\\n1\\n2.5\\n': column x inferred as Float64",
    "csv b'x\\n1\\nabc\\n': column x inferred as String",
    "csv with a text value on row 201, inferred from 100 rows: ComputeError: could not parse `abc` as dtype `i64` at column 'x' (column number 1)",
    "the same file with infer_schema=False: column x read as String",
    f"float bounds on integer data: TypeError: inferred type is f64, expected i32. {TROUBLE_URL}",
    "integer bounds on integer data: scale=10.0",
    "check(1.0, 1.0) on an approximate measurement: TypeError: Cannot coerce a non-tuple type to a tuple",
    "check(1.0, (1.0, 0.0)) on the same measurement: True",
    'clamp on floats that may be nan: OpenDPException: FailedFunction("Domain contains nan")',
    "impute, then clamp: scale=80.0",
    f"clamp to [0, 10] then a sum built for [0, 100]: OpenDPException: {MISMATCH}",
    "    output_domain: VectorDomain(AtomDomain(bounds=[0.0, 10.0], T=f64))",
    "    input_domain:  VectorDomain(AtomDomain(bounds=[0.0, 100.0], T=f64))",
    '")',
    "clamp to [0, 10] then then_sum(): sensitivity 10.0",
    "float sum without a margin: OpenDPException: MakeTransformation(\"must specify 'max_length' in a margin with by=[]\")",
    "float sum with max_length=5,400,000: scale=2,881.04",
    "count: scale=36.0 95% accuracy=108.34",
    "collect before release: ValueError: You must call `.release()` before executing a query.",
    'collect a OnceFrame twice: OpenDPException: FailedFunction("OnceFrame has been exhausted")',
    "query after the budget is spent: ValueError: Privacy allowance has been exhausted",
    "contingency table with keys for every column under (epsilon, delta): ValueError: delta (1e-07) must be zero because keys and cuts span all columns",
    "the same table under pure epsilon: ContingencyTableQuery",
    "variant: DomainMismatch",
    "message: Intermediate domains don't match. See https://github.com/opendp/opendp/discussions/297",
    "rust frames recorded: True",
    "Continued Rust stack trace:",
    "    opendp_combinators__make_chain_tt",
    "    opendp::combinators::chain::mismatch_error::<opendp::ffi::any::AnyDomain>",
    f"  {MISMATCH}",
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
def test_rust_frames_follow_the_trace_header(out: str) -> None:
    lines = out.splitlines()
    header = lines.index("Continued Rust stack trace:")
    assert lines[header + 2].startswith("    opendp::combinators::chain::make_chain_tt::<")


@pytest.mark.libs
def test_released_count_near_truth(out: str) -> None:
    (count,) = number(rf"^  count {NUMBER}  \(true 3,811,932, within 108 with probability 0\.95\)$", out)
    assert abs(count - 3_811_932) <= 3 * 108.34
