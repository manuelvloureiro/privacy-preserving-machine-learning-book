# Limitations and troubleshooting

OpenDP's guarantee covers only computations through measurements with valid proofs. It does not cover assumptions about arithmetic, code before the context or channels other than the output. Most library refusals enforce this boundary; error messages often indicate that the guarantee is working as intended.

This chapter follows the documentation's pages on limitations, features and troubleshooting. It describes unsupported guarantees, the role of each feature flag and behaviour outside the guarantee. It also explains recurring error messages and the checks reviewers should make before trusting a release.

The script [opendp_limitations.py](../../code/part04-extending/opendp_limitations.py) reproduces these messages with OpenDP 0.16.0 and Polars 1.36.1. Framework examples use small lists; Polars examples use the France Labour Force Survey (LFS) sample behind a context. Messages, sensitivities, scales and error bars are deterministic. OpenDP's noise cannot be seeded, so the released count varies and is labelled as one run.

```bash
uv run python code/part04-extending/opendp_limitations.py
```

## What OpenDP does not promise {#sec-limits-promise}

The documentation identifies five deployment limitations. First, privacy proofs use real arithmetic, while computers round floating-point sums. Casacuberta, Shoemate, Vadhan and Wagaman (2022) found that several libraries treated arithmetic as exact when computing sum sensitivity. Rounding increased the true sensitivity, making the added noise too small and the stated $\varepsilon$ incorrect.

OpenDP accounts for rounding in its proof. The script builds float and integer sums of values from 0 to 80 ([Transformations](../part03-framework/09-transformations.md)) and computes their sensitivity for 36 added or removed rows.

```{.python excerpt="limits-numerics"}
def sum_sensitivities(d_in: int) -> tuple[float, int]:
    """The sensitivity of a float sum and of an integer sum of values in [0, 80], at `d_in` added or removed rows."""
    float_sum = (dp.vector_domain(dp.atom_domain(bounds=HOURS_BOUNDS)), dp.symmetric_distance()) >> dp.t.then_sum()
    int_sum = (dp.vector_domain(dp.atom_domain(bounds=(0, 80))), dp.symmetric_distance()) >> dp.t.then_sum()
    return float_sum.map(d_in), int_sum.map(d_in)
```

```text
float sum of 36 rows in [0, 80]: sensitivity 2880.0000007451
int sum of 36 rows in [0, 80]: sensitivity 2880
```

With exact arithmetic, both sensitivities would be $36\times80$. The float sum's map adds a rounding term; integer addition needs none. This term grows with row count, so float sums on the LFS need a public row-count bound ([Essential statistics](../part02-tabular-data/03-essential-statistics.md)). Components that still assume exact arithmetic require the `idealized-numerics` flag. Prefer integer statistics when possible; otherwise use OpenDP's reported scale rather than a formula that ignores rounding.

Second, OpenDP is not hardened against side channels. Query time, memory use and cache-line access may depend on the data, and proofs do not cover information learned through these channels. Deployment must therefore limit who can interact with the system. An analyst submitting queries through a queue poses a different risk from one with shell access. Restrict the latter or account for the weaker guarantee.

Third, components require vetting. Each should have a written proof and review; until review is complete, it requires the `contrib` flag. Review is intended to identify errors in code or proofs. Releases using `contrib` components therefore rely on proofs not yet checked by a second reviewer. Record the components used so you can reassess a release after corrections.

Fourth, interfaces may change between minor releases before version 1.0.0. Pin the version, as this book does at 0.16.0, and test release code so changed defaults are detected before they alter results. Fifth, unknown software bugs remain possible. Keep a reproducible record of each run so you can recompute its guarantee, as described in the practitioner edition's [Release and governance](../../book-practitioner/part05-dp-ml/12-release-governance.md).

## The feature flags {#sec-limits-features}

Feature flags record acceptance of these limitations. OpenDP has four flags and one deprecated alias; all are off until `dp.enable_features` enables them. [A first release](../part01-getting-started/01-first-release.md) enabled `contrib` without showing the disabled behaviour, so the script disables it, tests the Laplace mechanism and enables it again.

```{.python excerpt="limits-contrib"}
dp.disable_features("contrib")
without_contrib = refusal(lambda: space >> dp.m.then_laplace(scale=1.0))
dp.enable_features("contrib")
```

```text
laplace without contrib: OpenDPException: Attempted to use function that requires "contrib", but not enabled. See https://docs.opendp.org/en/stable/api/user-guide/#feature-listing, then call enable_features("contrib")
```

The helper `refusal` returns an exception's type and message. Each flag's error names the flag and the call that enables it. To find which calls require a flag, run the script with that flag disabled.

The `honest-but-curious` flag gates constructors whose guarantees rely on user-supplied arguments, mainly the plugins in [Plugins](13-plugins.md). For example, `make_user_measurement` accepts a Python privacy map that OpenDP cannot verify. The name describes the assumed adversary: honest when constructing the measurement, curious when inspecting its outputs. The script builds a plugin that returns its input unchanged and claims privacy loss equal to input distance.

```{.python excerpt="limits-plugin"}
def plugin_measurement() -> dp.Measurement:
    """A plugin measurement whose privacy map is only a claim: the library cannot check `identity_map`."""
    return dp.m.make_user_measurement(
        dp.atom_domain(T=float, nan=False),
        dp.absolute_distance(T=float),
        dp.max_divergence(),
        function=identity,
        privacy_map=identity_map,
    )
```

```text
plugin without honest-but-curious: OpenDPException: Attempted to use function that requires "contrib", "honest-but-curious", but not enabled. See https://docs.opendp.org/en/stable/api/user-guide/#feature-listing, then call enable_features("contrib", "honest-but-curious")
plugin with honest-but-curious: map(1.0)=1.0
```

With the flag enabled, the plugin reports its claimed map: loss 1 at distance 1, with no verification. Enable it when developing and proving plugins, not where untrusted analysts submit queries; they could register a map claiming any cost. The flag also gates `OnceFrame.lazy()`, which exposes a released Polars plan that could run twice.

The third flag is `idealized-numerics`, which marks components whose proof assumes exact arithmetic. In 0.16.0 none of the constructors in this book needs it. The float Laplace mechanism and the float sums above build with the flag off, because their proofs account for rounding. The older name of the flag, `floating-point`, still works and gives a warning.

```text
float laplace without idealized-numerics: no error
enable_features("floating-point"): DeprecationWarning: "floating-point" is deprecated. Use "idealized-numerics" instead.
assert_features("idealized-numerics") with the alias on: no error
assert_features("idealized-numerics") after disabling it: OpenDPException: Attempted to use function that requires "idealized-numerics", but not enabled. See https://docs.opendp.org/en/stable/api/user-guide/#feature-listing, then call enable_features("idealized-numerics")
assert_features("contrib", "honest-but-curious"): OpenDPException: Attempted to use function that requires "contrib", "honest-but-curious", but not enabled. See https://docs.opendp.org/en/stable/api/user-guide/#feature-listing, then call enable_features("contrib", "honest-but-curious")
```

The last three lines use `dp.assert_features`, which raises the error of a gated constructor without building anything. Every gated constructor calls it first. At the top of a script, one such line also states every flag that the script relies on. The fourth flag, `rust-stack-trace`, changes no guarantee, because it only adds the Rust frames to an error message, as the section on reading an exception shows.

## What lies outside the guarantee {#sec-limits-outside}

The guarantee covers what happens to the data inside a measurement, and four things happen outside it in every analysis. The first is loading the data, which the documentation leaves to a trusted curator. One reason is that a CSV reader infers the type of each column from its values. The script reads three two-row files that differ in one value, then a longer file whose odd value comes after the rows that the inference reads.

```{.python excerpt="limits-csv"}
for raw in (b"x\n1\n2\n", b"x\n1\n2.5\n", b"x\n1\nabc\n"):
    print(f"csv {raw!r}: column x inferred as {pl.read_csv(io.BytesIO(raw)).schema['x']}")
late_text = b"x\n" + b"1\n" * 200 + b"abc\n"  # a text value after 200 numbers
failed = refusal(lambda: pl.read_csv(io.BytesIO(late_text), infer_schema_length=100))
print(f"csv with a text value on row 201, inferred from 100 rows: {first_line(failed)}")
as_strings = pl.read_csv(io.BytesIO(late_text), infer_schema=False).schema["x"]
print(f"the same file with infer_schema=False: column x read as {as_strings}")
```

```text
csv b'x\n1\n2\n': column x inferred as Int64
csv b'x\n1\n2.5\n': column x inferred as Float64
csv b'x\n1\nabc\n': column x inferred as String
csv with a text value on row 201, inferred from 100 rows: ComputeError: could not parse `abc` as dtype `i64` at column 'x' (column number 1)
the same file with infer_schema=False: column x read as String
```

So one person's value decides whether the column is integer, float or text, and a late value decides whether the load succeeds at all. If the result of the load is visible outside, it is a release that nothing accounted for. The documentation gives two fixes. Read a format that carries its schema, as this book does with Parquet. Or read every column as text with `infer_schema=False`, then cast it yourself. The LFS loader in `practitioner_datasets` converts the CSV to Parquet once, which is the curator's step, outside any guarantee.

The second thing outside the guarantee is public information. Every bound, imputed value, key list and margin in this book comes from the domain or from a published source. A bound read from the private data is a release of the data, and the library cannot tell the two apart.

The third is code that runs outside the context, which is why [the data and the context](../part01-getting-started/02-the-data-and-the-context.md) told you to delete your reference to the frame. Collecting the frame yourself is not a query, so nothing counts it. The fourth is the error channel. Whether a query succeeds or fails is itself an output, so a failure that depends on the data is a release. OpenDP guards against this inside a query, for example by replacing some Polars arguments so that an expression cannot fail on one value. The loading step and your own code have no such guard.

## The error messages you will meet {#sec-limits-errors}

The documentation's troubleshooting page has one entry, the type mismatch, and the earlier chapters met several more. This section gives each with its cause and its fix. The type mismatch comes first. OpenDP fixes the type of every argument from the domain of the data, and it refuses an argument of another type instead of converting it. The script puts three integers behind a context and clamps them with float bounds.

```text
float bounds on integer data: TypeError: inferred type is f64, expected i32. See https://docs.opendp.org/en/stable/getting-started/trouble-shooting.html#inferred-type-is-xxx-expected-yyy
integer bounds on integer data: scale=10.0
```

The message names the type it inferred from your argument and the type of the data. To fix it, pass bounds of the type of the data, `(0, 10)` for integers, or cast the data first. The documentation names one common cause, a scalar $\varepsilon$ where a measurement expects an $(\varepsilon, \delta)$ pair. To show it, the script asks an approximate measurement, built with `make_approximate` as in [Combinators](../part03-framework/11-combinators.md), whether it stays within a scalar budget.

```text
check(1.0, 1.0) on an approximate measurement: TypeError: Cannot coerce a non-tuple type to a tuple
check(1.0, (1.0, 0.0)) on the same measurement: True
```

The privacy loss of an approximate measurement is a pair, so its budget must be a pair too, and a pure budget is the pair $(\varepsilon, 0)$. The same message comes from `binary_search_param` with a scalar `d_out` on an approximate chain ([Utilities and the Context API](../part03-framework/12-utilities-and-the-context-api.md)).

The next message comes from a domain that allows missing values. The library infers a list of Python floats as a domain that may contain NaN. A clamp refuses such a domain, because NaN is neither inside nor outside a bound.

```text
clamp on floats that may be nan: OpenDPException: FailedFunction("Domain contains nan")
impute, then clamp: scale=80.0
```

The fix is the one of the first chapter, `impute_constant` before `clamp`, which maps the domain to one without NaN. In Polars the same need appears as `fill_null` before a `dp.sum`.

The next message comes from mismatched domains in a chain. The `>>` operator checks that the output domain of the left side equals the input domain of the right side, and refuses otherwise. The script clamps to one range and then chains a sum built for a wider one.

```{.python excerpt="limits-chain"}
vectors = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()
clamp = vectors >> dp.t.then_clamp((0.0, 10.0))
wide_sum = dp.t.make_sum(dp.vector_domain(dp.atom_domain(bounds=(0.0, 100.0))), dp.symmetric_distance())
mismatch = refusal(lambda: clamp >> wide_sum)  # the clamp's output has bounds [0, 10], the sum wants [0, 100]
chained = clamp >> dp.t.then_sum()  # a partial constructor takes the clamp's output space, so they agree
```

```text
clamp to [0, 10] then a sum built for [0, 100]: OpenDPException: DomainMismatch("Intermediate domains don't match. See https://github.com/opendp/opendp/discussions/297
    output_domain: VectorDomain(AtomDomain(bounds=[0.0, 10.0], T=f64))
    input_domain:  VectorDomain(AtomDomain(bounds=[0.0, 100.0], T=f64))
")
clamp to [0, 10] then then_sum(): sensitivity 10.0
```

The message prints both domains, so you can read that the bounds differ. The fix is the partial constructor `then_sum()`, which takes its input space from the output of the clamp ([Transformations](../part03-framework/09-transformations.md)).

The remaining messages come from the Polars side, with the LFS behind a context of one query and a budget of $\varepsilon = 1$. The unit is the 36 rows of [the data and the context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context).

```{.python excerpt="limits-context"}
def make_context(
    frame: pl.LazyFrame, delta: float | None = None, margins: Sequence[dp.polars.Margin] | None = None
) -> dp.Context:
    """The LFS behind a unit of 36 rows and a budget of epsilon = 1, with or without a delta, for one query."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(epsilon=EPSILON, delta=delta) if delta else dp.loss_of(epsilon=EPSILON),
        split_evenly_over=1,
        margins=margins,
    )
```

A float sum needs a margin with `max_length`, for the reason of the first section. Its sensitivity includes a rounding term that grows with the number of rows, so the proof needs a public bound on that number.

```text
float sum without a margin: OpenDPException: MakeTransformation("must specify 'max_length' in a margin with by=[]")
float sum with max_length=5,400,000: scale=2,881.04
```

The fix is the documentation's margin, `dp.polars.Margin(max_length=150_000 * 36)`. The scale is then a little above 2880, the sensitivity of the sum over $\varepsilon = 1$ in exact arithmetic. The extra is the rounding term at 5.4 million rows. A grouped sum needs the same bound in the margin of its grouping, which is what `by=[]` in the message refers to.

The next three messages come from one query. The script builds a count and tries to collect it before it releases it. Then it releases the count, collects the result twice, and submits another query to a context whose one share is spent.

```{.python excerpt="limits-once"}
count_query = context.query().select(dp.len(signed=True))
before_release = refusal(lambda: count_query.collect())  # the plan has no data; only a released OnceFrame does
once = count_query.release()
dp_count = int(once.collect()["len"][0])
twice = refusal(lambda: once.collect())  # a second collection would be a second release
exhausted = refusal(lambda: context.query().select(dp.len(signed=True)).release())
```

```text
count: scale=36.0 95% accuracy=108.34
collect before release: ValueError: You must call `.release()` before executing a query.
collect a OnceFrame twice: OpenDPException: FailedFunction("OnceFrame has been exhausted")
query after the budget is spent: ValueError: Privacy allowance has been exhausted
```

A `LazyFrameQuery` is a plan without data, so `collect` on it would compute nothing private. The library refuses anyway, so that the habit of collecting a plan never meets a frame that holds data. You can collect the `OnceFrame` from `release` only once, because the noise is drawn when the plan runs, so a second run would be a second release. The third message is the one that the first chapter ended on. None of the three is a bug to work around, so keep the DataFrame from the first `collect` if you need a value twice.

The last message is from [Contingency tables](../part05-synthetic-data/16-contingency-tables.md). A contingency table whose keys or cuts cover every column has no private key set to protect, so its release spends no $\delta$. The library refuses a budget with a $\delta$ instead of leaving it unspent. The script uses $\delta = 10^{-7}$, the value of the grouping chapters.

```text
contingency table with keys for every column under (epsilon, delta): ValueError: delta (1e-07) must be zero because keys and cuts span all columns
the same table under pure epsilon: ContingencyTableQuery
```

So state the budget as pure $\varepsilon$ when the keys are all public, or as the zero-concentrated budget $\rho$ of [Utility and noise](../part02-tabular-data/07-utility-and-noise.md). Keep $\delta$ for the tables whose keys the library must discover. Two more messages from earlier chapters belong in this list. A plain `mean` is refused, because OpenDP has no proof for it. A grouping without keys or a $\delta$ is refused, because its key set is private ([Grouping](../part02-tabular-data/04-grouping.md)).

## Reading an OpenDPException {#sec-limits-exception}

Most of the messages above came as an `OpenDPException`, which is how an error in the Rust core reaches Python. Its `variant` is the kind of error, from a fixed list such as `DomainMismatch` or `FailedFunction`, which code can match on. Its `message` is the text, and its `raw_traceback` is the Rust stack at the point of the error, which is recorded even if you do not display it. The script raises the domain mismatch again and reads the three, then switches on `rust-stack-trace` and prints the exception again.

```{.python excerpt="limits-exception"}
try:
    _ = clamp >> wide_sum
except dp.OpenDPException as err:
    print(f"variant: {err.variant}")
    print(f"message: {first_line(err.message or '')}")
    print(f"rust frames recorded: {err.raw_traceback is not None}")
    dp.enable_features("rust-stack-trace")
    print(str(err).strip())
    dp.disable_features("rust-stack-trace")
```

```text
variant: DomainMismatch
message: Intermediate domains don't match. See https://github.com/opendp/opendp/discussions/297
rust frames recorded: True
Continued Rust stack trace:
    opendp_combinators__make_chain_tt
    opendp::combinators::chain::make_chain_tt::<opendp::ffi::any::AnyDomain, opendp::ffi::any::AnyDomain, opendp::ffi::any::AnyDomain, opendp::ffi::any::AnyMetric, opendp::ffi::any::AnyMetric, opendp::ffi::any::AnyMetric>
    opendp::combinators::chain::mismatch_error::<opendp::ffi::any::AnyDomain>
  DomainMismatch("Intermediate domains don't match. See https://github.com/opendp/opendp/discussions/297
    output_domain: VectorDomain(AtomDomain(bounds=[0.0, 10.0], T=f64))
    input_domain:  VectorDomain(AtomDomain(bounds=[0.0, 100.0], T=f64))
")
```

The frames read from the outside in: the entry point that Python called, the chaining function, and the helper that formats the mismatch. They help when the message does not say where in a long chain the library stopped, and they locate a bug in the Rust source. The list of variants may change between versions, so match on a variant only in tests you control. The errors that are not an `OpenDPException` come from the Python layer, a `TypeError` from argument conversion and a `ValueError` from the context and the query classes.

## What a reviewer verifies {#sec-limits-review}

A release made with OpenDP claims that this table satisfies $(\varepsilon, \delta)$-differential privacy for this privacy unit, and @tbl-limits-review collects the checks of that claim. The first is the unit, because an $\varepsilon$ means nothing until you know what it protects. For example, `dp.unit_of(contributions=36)` protects a person only if nobody has more than 36 rows ([Identifier truncation](../part02-tabular-data/05-identifier-truncation.md)). The second is the budget, read back from the context.

The third is the flags, because with `honest-but-curious` the reviewer must read each privacy map that the library took on trust. The fourth is the source of each bound and key, and the fifth is what ran outside the context. The full edition's [chapter on the definition](../../book-full/part03-dp-foundations/01-definition.md) says what the finished claim must state.

| Check | Question | Where to look |
| --- | --- | --- |
| Unit | What does one privacy unit contain, and does anyone exceed it? | `privacy_unit` in the context; the identifier and any truncation |
| Budget | What was spent, under which measure, and was any release collected twice? | `context.current_privacy_loss()`; one `collect` per `OnceFrame` |
| Features | Which flags were on, and if `honest-but-curious`, which maps were claimed? | the `enable_features` line; every `make_user_*` call and `lazy()` |
| Public information | Where did each bound, imputed value, key list and margin come from? | the constants of the script, traced to their sources |
| Outside the context | What touched the data before and after the context? | the load and its schema; any code holding the frame |

: What a reviewer verifies before trusting a release made with OpenDP. {#tbl-limits-review}

## Summary

| Call or concept | What it does | Rule of thumb |
| --- | --- | --- |
| `dp.enable_features("contrib")` | Opts in to components whose proofs are not yet vetted | Every script in this book needs it; record it with the release |
| `honest-but-curious` | Opts in to constructors whose guarantee rests on your arguments, such as plugins and `OnceFrame.lazy()` | Only where you wrote and proved the map, never where an untrusted analyst queries |
| `idealized-numerics` | Opts in to components whose proof assumes exact arithmetic | No constructor in this book needs it in 0.16.0; its alias `floating-point` warns |
| `dp.assert_features(...)` | Raises the flag error without building anything | One line at the top of a script states what it relies on |
| `rust-stack-trace` | Adds the Rust frames to an exception's text | Switch on to locate a bug; it changes no guarantee |
| Loading the data | Outside the guarantee; CSV inference reads private values | Read Parquet, or `infer_schema=False` and cast |
| `inferred type is f64, expected i32` | An argument's type differs from the data's | Match the data's type or cast first; a scalar $\varepsilon$ for a pair is the common case |
| `Domain contains nan` | A float domain may hold NaN, and clamp refuses it | `impute_constant` or `fill_null` before `clamp` |
| `DomainMismatch` | Two chained components disagree on the domain between them | Use `then_*` constructors; read the two domains in the message |
| `must specify 'max_length' in a margin` | A float sum needs a public bound on the rows | `dp.polars.Margin(max_length=...)` from a public source |
| `You must call .release() before executing a query` | A plan was collected before it was released | `release()` first, then `collect()` once |
| `OnceFrame has been exhausted` | A released frame was collected twice | Keep the DataFrame from the first `collect` |
| `Privacy allowance has been exhausted` | Every share of the budget is spent | The refusal is the guarantee; a new context is a new budget |
| `delta (...) must be zero because keys and cuts span all columns` | A table with public keys spends no $\delta$ | Budget the query with pure $\varepsilon$ or with $\rho$ |
| `OpenDPException` | Carries `variant`, `message` and `raw_traceback` | Match on the variant only in tests; expect the list to change before 1.0 |

: The flags, the limits and the error messages of this chapter, with the rule of thumb for each. {#tbl-limits-summary}

## References

- OpenDP documentation: "Limitations" and the feature listing in the API user guide; "Troubleshooting" and "Next Steps" in Getting Started; the note on data loading in "Working with Tabular Data" and the margin and float-sum notes in "Essential Statistics".
- Casacuberta, Shoemate, Vadhan and Wagaman, "Widespread Underestimation of Sensitivity in Differentially Private Libraries and How to Fix It" (2022): the finite-precision sensitivity errors in earlier libraries, and the corrected proofs that OpenDP's sums follow.
- Mironov, "On Significance of the Least Significant Bits for Differential Privacy" (2012): the floating-point attack on the Laplace sampler.
- Jin, McMurtry, Rubinstein and Ohrimenko, "Are We There Yet? Timing and Floating-Point Attacks on Differential Privacy Systems" (2022): timing and floating-point side channels in differential privacy libraries.
- The full edition's chapters on [the definition](../../book-full/part03-dp-foundations/01-definition.md) and [discrete and floating-point-safe mechanisms](../../book-full/part03-dp-foundations/08-discrete-mechanisms.md) derive the results this chapter states; the practitioner edition's [Release and governance](../../book-practitioner/part05-dp-ml/12-release-governance.md) gives the record a reviewer asks for.
