# The Context API of OpenDP 0.16.0 (opendp/context.py). See typings/README.md.
from collections.abc import Callable, Sequence
from typing import Any, Type

from opendp.extras.polars import Bound, LazyFrameQuery, Margin
from opendp.mod import Domain, Measure, Measurement, Metric, Odometer, OdometerQueryable, Queryable, Transformation

class PartialChain:
    """A chain with one parameter still to find (`.laplace()` without a scale), fixed by the context's budget."""

    def __init__(self, f: Callable[..., Any], *args: Any, **kwargs: Any) -> None: ...
    def __call__(self, v: float) -> Transformation | Measurement: ...
    def fix(
        self,
        d_in: Any,
        d_out: Any,
        output_measure: Measure | None = None,
        bounds: tuple[float, float] | None = None,
        T: Type[float] | Type[int] | None = None,
    ) -> Transformation | Measurement: ...
    def __rshift__(self, other: Any) -> PartialChain: ...
    def __rrshift__(self, other: Any) -> PartialChain: ...

class Query:
    """A query on plain (non-Polars) data. OpenDP builds `.count()`, `.clamp()`, `.laplace()` and the other
    transformation and measurement constructors through `__getattr__`, each returning a new Query. It is typed
    Any here, rather than OpenDP's own `Callable[..., Query]`, so that `Context.query()`'s union stays usable on
    a LazyFrame, where the Polars methods are the typed ones."""

    def __getattr__(self, name: str) -> Any: ...
    def resolve(
        self,
        allow_transformations: bool = False,
        bounds: tuple[float, float] | None = None,
        T: Type[float] | Type[int] | None = None,
    ) -> Transformation | Measurement | Odometer: ...
    def release(
        self, data: Any = None, bounds: tuple[float, float] | None = None, T: Type[float] | Type[int] | None = None
    ) -> Any: ...
    def param(
        self,
        allow_transformations: bool = False,
        bounds: tuple[float, float] | None = None,
        T: Type[float] | Type[int] | None = None,
    ) -> float: ...
    def compositor(
        self,
        split_evenly_over: int | None = None,
        split_by_weights: Sequence[float] | None = None,
        d_out: float | tuple[float, float] | None = None,
        output_measure: Measure | None = None,
        alpha: float | None = None,
    ) -> Query: ...
    def new_with(self, *, chain: Any, wrap_release: Callable[[Any], Any] | None = None) -> Query: ...

class Context:
    """Mediates every query to the data: holds the accountant (an adaptive composition or an odometer), the
    privacy unit's distance `d_in`, the per-query losses `d_mids` and the total `d_out`."""

    accountant: Measurement  # an Odometer only for a context built around one; the compositor's is a Measurement
    queryable: Queryable | OdometerQueryable
    d_in: Any
    d_mids: list[Any] | None
    d_out: Any
    d_mids_consumed: list[Any]
    query_space: tuple[Domain, Metric] | None
    def __init__(
        self,
        accountant: Measurement,
        queryable: Queryable,
        d_in: float | Sequence[Bound],
        d_mids: list[float] | list[tuple[float, float]] | None = None,
        d_out: float | tuple[float, float] | None = None,
        query_space: tuple[Domain, Metric] | None = None,
    ) -> None: ...
    @staticmethod
    def compositor(
        data: Any,
        privacy_unit: tuple[Metric, Any],
        privacy_loss: tuple[Measure, Any],
        split_evenly_over: int | None = None,
        split_by_weights: Sequence[float] | None = None,
        domain: Domain | None = None,
        margins: Sequence[Margin] | None = None,
    ) -> Context: ...
    def __call__(self, query: Measurement) -> Any: ...
    # A LazyFrameQuery when the data is a Polars LazyFrame, a Query otherwise; narrow with isinstance.
    def query(self, **kwargs: float) -> Query | LazyFrameQuery: ...
    def current_privacy_loss(self) -> Any: ...
    def remaining_privacy_loss(self) -> Any: ...
    def deserialize_polars_plan(self, serialized_plan: bytes) -> LazyFrameQuery: ...

def loss_of(
    epsilon: float | None = None, delta: float | None = None, rho: float | None = None
) -> tuple[Measure, float | tuple[float, float]]: ...
def unit_of(
    *,
    contributions: int | Sequence[Bound] | None = None,
    changes: int | None = None,
    absolute: float | None = None,
    l1: float | None = None,
    l2: float | None = None,
    local: bool | None = None,
    identifier: str | Any | None = None,
    ordered: bool = False,
    U: Any = None,
) -> tuple[Metric, Any]: ...
def space_of(T: Any, M: Any = None, infer: bool = False) -> tuple[Domain, Metric]: ...
def domain_of(T: Any, infer: bool = False) -> Domain: ...
def metric_of(M: Any) -> Metric: ...
def register(
    constructor: Callable[..., Transformation | Measurement], name: str | None = None
) -> Callable[..., Transformation | Measurement]: ...
