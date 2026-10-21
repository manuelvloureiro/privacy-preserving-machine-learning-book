from collections.abc import Callable, Sequence
from typing import Any

import polars as pl

from opendp.extras import polars as polars
from opendp.extras.polars import Bound, LazyFrameQuery, Margin

__version__: str

def enable_features(*features: str) -> None: ...
def unit_of(
    *,
    contributions: int | Sequence[Bound] | None = None,
    changes: int | None = None,
    absolute: float | None = None,
    l1: float | None = None,
    l2: float | None = None,
    local: bool | None = None,
    identifier: str | None = None,
    ordered: bool = False,
    U: Any = None,
) -> tuple[Any, Any]: ...
def loss_of(epsilon: float | None = None, delta: float | None = None, rho: float | None = None) -> tuple[Any, Any]: ...
def len(scale: float | None = None, signed: bool | None = None) -> pl.Expr: ...

class Context:
    d_in: Any
    d_mids: list[Any] | None
    d_out: Any
    @staticmethod
    def compositor(
        data: Any,
        privacy_unit: tuple[Any, Any],
        privacy_loss: tuple[Any, Any],
        split_evenly_over: int | None = None,
        split_by_weights: Sequence[float] | None = None,
        domain: Any = None,
        margins: Sequence[Margin] | None = None,
    ) -> Context: ...
    def query(self, **kwargs: float) -> LazyFrameQuery: ...
    accountant: Measurement
    def current_privacy_loss(self) -> Any: ...
    def remaining_privacy_loss(self) -> Any: ...

def binary_search(
    predicate: Callable[[float], bool],
    bounds: tuple[float | None, float | None] | None = (None, None),
    T: type[float] | None = None,
    return_sign: bool = False,
) -> float: ...
def atom_domain(bounds: Any = None, nan: bool | None = None, T: Any = None) -> Any: ...
def absolute_distance(T: Any) -> Any: ...

class Measurement:
    def map(self, d_in: Any) -> Any: ...
    def __call__(self, arg: Any) -> Any: ...

class _Measurements:
    def make_gaussian(
        self,
        input_domain: Any,
        input_metric: Any,
        scale: float,
        k: int | None = None,
        MO: str = "ZeroConcentratedDivergence",
    ) -> Measurement: ...
    def make_laplace(
        self, input_domain: Any, input_metric: Any, scale: float, k: int | None = None, MO: str = "MaxDivergence"
    ) -> Measurement: ...

class _Combinators:
    def make_zCDP_to_approxDP(self, measurement: Measurement) -> Measurement: ...
    def make_fix_delta(self, measurement: Measurement, delta: float) -> Measurement: ...

m: _Measurements
c: _Combinators
