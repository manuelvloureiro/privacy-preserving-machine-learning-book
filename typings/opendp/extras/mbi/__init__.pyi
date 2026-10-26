# OpenDP's marginal-based inference extension (opendp/extras/mbi), 0.16.0: contingency tables fitted with Private-PGM
# (the `mbi` package) and the algorithms that choose which marginals to measure (AIM, MST, a fixed workload).
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from opendp.extras.polars import Bound
from opendp.mod import FrameDistance, LazyFrameDomain, Measure, Measurement

def mirror_descent(domain: Any, loss_fn: Any, *, potentials: Any = None) -> Any: ...
@dataclass
class Count:
    """A count query over the columns `by`; `weight` is its importance (AIM's priority, Fixed's budget share)."""

    by: tuple[str, ...]
    weight: float = 1.0

class Marginals:
    def __init__(self, by_clique: dict[tuple[str, ...], list[Any]] | None = None) -> None: ...
    def cliques(self) -> list[tuple[str, ...]]: ...
    def flatten(self) -> list[Any]: ...
    def copy(self) -> Marginals: ...
    def add(self, *new_measurements: Any) -> Marginals: ...

@dataclass(kw_only=True)
class Algorithm:
    estimator: Callable[..., Any] = ...
    oneway: Literal["all", "unkeyed"] = "all"
    oneway_split: float | None = None
    @property
    def needs_model(self) -> bool: ...
    def make_marginals(
        self,
        input_domain: LazyFrameDomain,
        input_metric: FrameDistance,
        output_measure: Measure,
        d_in: list[Bound],
        d_out: float,
        *,
        marginals: Marginals,
        model: Any,
    ) -> Measurement: ...

@dataclass(kw_only=True)
class AIM(Algorithm):
    queries: list[Count] | int = 3
    measure_split: float = 0.9
    max_size: float = 80.0
    rounds: int | None = None

@dataclass(kw_only=True)
class MST(Algorithm):
    measure_split: float = 0.9
    num_selections: int | None = None

@dataclass(kw_only=True)
class Fixed(Algorithm):
    oneway: Literal["all", "unkeyed"] = "unkeyed"
    queries: list[Count]

@dataclass(kw_only=True)
class Sequential(Algorithm):
    algorithms: list[Algorithm]
    weights: list[float] | None = None

@dataclass
class ContingencyTable:
    """A released contingency table: the keys and cuts of each column, the noisy marginals and the fitted model."""

    keys: dict[str, Any]
    cuts: dict[str, Any]
    marginals: Marginals
    model: Any
    thresholds: dict[str, int] = ...
    def synthesize(self, rows: int | None = None, method: Literal["round", "sample"] = "round") -> pl.DataFrame: ...
    def project(self, attrs: str | Sequence[str]) -> NDArray[np.float64]: ...
    def project_melted(self, attrs: str | Sequence[str]) -> pl.DataFrame: ...
    def std(self, attrs: Sequence[str]) -> float: ...
    @property
    def schema(self) -> dict[str, Any]: ...

def make_contingency_table(
    input_domain: LazyFrameDomain,
    input_metric: FrameDistance,
    output_measure: Measure,
    d_in: list[Bound],
    d_out: float | tuple[float, float],
    keys: Mapping[str, Sequence[Any]] | None = None,
    cuts: Mapping[str, Sequence[float]] | None = None,
    table: ContingencyTable | None = None,
    algorithm: Algorithm = ...,
) -> tuple[Measurement, float | None, int | None]: ...
