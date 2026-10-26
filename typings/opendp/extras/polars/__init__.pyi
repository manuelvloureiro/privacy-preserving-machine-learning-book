# OpenDP's Polars extension (opendp/extras/polars/__init__.py), 0.16.0. The `.dp` namespace that OpenDP adds to
# Polars expressions at run time (`pl.col("x").dp.sum(bounds)`) cannot be declared on pl.Expr, so scripts type such
# expressions Any; the class DPExpr below documents its methods.
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

from opendp.extras.mbi import Algorithm, ContingencyTable
from opendp.extras.polars.contingency_table import ContingencyTableQuery
from opendp.mod import Domain, LazyFrameDomain, Measure, Measurement, Metric, SeriesDomain, Transformation
from opendp.typing import RuntimeTypeDescriptor

@dataclass
class Bound:
    by: Sequence[Any] = ...
    per_group: int | None = None
    num_groups: int | None = None

@dataclass
class Margin:
    by: Sequence[Any] = ...
    max_length: int | None = None
    max_groups: int | None = None
    invariant: Literal["keys", "lengths"] | None = None

@dataclass
class SortBy:
    by: Any
    descending: bool | Sequence[bool] = False
    nulls_last: bool | Sequence[bool] = False
    multithreaded: bool = True
    maintain_order: bool = False

class DPExpr:
    """The methods of the `.dp` namespace on a Polars expression."""

    def noise(self, scale: float | None = None, distribution: Literal["Laplace", "Gaussian"] | None = None) -> pl.Expr: ...
    def laplace(self, scale: float | None = None) -> pl.Expr: ...
    def gaussian(self, scale: float | None = None) -> pl.Expr: ...
    def len(self, scale: float | None = None, signed: bool | None = None) -> pl.Expr: ...
    def count(self, scale: float | None = None, signed: bool | None = None) -> pl.Expr: ...
    def null_count(self, scale: float | None = None, signed: bool | None = None) -> pl.Expr: ...
    def n_unique(self, scale: float | None = None, signed: bool | None = None) -> pl.Expr: ...
    def sum(self, bounds: tuple[float, float], scale: float | None = None) -> pl.Expr: ...
    def mean(self, bounds: tuple[float, float], scale: float | None = None) -> pl.Expr: ...
    def quantile(self, alpha: float, candidates: list[float], scale: float | None = None) -> pl.Expr: ...
    def median(self, candidates: list[float], scale: float | None = None) -> pl.Expr: ...

class OnceFrame:
    def collect(self) -> pl.DataFrame: ...
    def lazy(self) -> pl.LazyFrame: ...

class LazyGroupByQuery:
    def agg(self, *aggs: pl.Expr | Any, **named_aggs: pl.Expr | Any) -> LazyFrameQuery: ...

class LazyFrameQuery:
    # The real class forwards any other name to the Polars plan; this also lets `Context.query()`'s union
    # (Query | LazyFrameQuery) accept the plain-data constructors (`.count()`, `.clamp()`, ...) that Query builds.
    def __getattr__(self, name: str) -> Any: ...
    def sort(self, by: Any, *more_by: Any, descending: bool | Sequence[bool] = False, nulls_last: bool | Sequence[bool] = False, maintain_order: bool = False, multithreaded: bool = True) -> LazyFrameQuery: ...
    def filter(self, *predicates: pl.Expr | Any, **constraints: Any) -> LazyFrameQuery: ...
    def select(self, *exprs: pl.Expr | Any, **named_exprs: pl.Expr | Any) -> LazyFrameQuery: ...
    def select_seq(self, *exprs: pl.Expr | Any, **named_exprs: pl.Expr | Any) -> LazyFrameQuery: ...
    def with_columns(self, *exprs: pl.Expr | Any, **named_exprs: pl.Expr | Any) -> LazyFrameQuery: ...
    def with_columns_seq(self, *exprs: pl.Expr | Any, **named_exprs: pl.Expr | Any) -> LazyFrameQuery: ...
    def group_by(
        self, *by: str | pl.Expr | Sequence[str | pl.Expr], maintain_order: bool = False, **named_by: Any
    ) -> LazyGroupByQuery: ...  # Polars' own group_by also takes a list of keys, as the documentation's group_by(quarterly)
    def join(
        self,
        other: pl.LazyFrame,
        on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
        how: Literal["inner", "left", "right", "full", "semi", "anti", "cross"] = "inner",
        *,
        left_on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
        right_on: str | pl.Expr | Sequence[str | pl.Expr] | None = None,
        suffix: str = "_right",
        validate: str = "m:m",
        nulls_equal: bool = False,
        coalesce: bool | None = None,
        maintain_order: str | None = None,
    ) -> LazyFrameQuery: ...
    def with_keys(self, keys: pl.LazyFrame | pl.DataFrame, on: list[str] | None = None) -> LazyFrameQuery: ...
    def truncate_per_group(
        self, k: int, by: list[Any] | None = None, keep: Literal["sample", "first", "last"] | SortBy = "sample"
    ) -> LazyFrameQuery: ...
    def truncate_num_groups(
        self, k: int, by: list[Any], keep: Literal["sample", "first", "last"] = "sample"
    ) -> LazyFrameQuery: ...
    def resolve(self) -> Measurement: ...
    def release(self) -> OnceFrame: ...
    def summarize(self, alpha: float | None = None) -> pl.DataFrame: ...
    def contingency_table(
        self,
        *,
        keys: Mapping[str, Sequence[Any]] | None = None,
        cuts: Mapping[str, Sequence[float]] | None = None,
        table: ContingencyTable | None = None,
        algorithm: Algorithm = ...,
    ) -> ContingencyTableQuery: ...

def dp_len(scale: float | None = None, signed: bool | None = None) -> pl.Expr: ...
def lazyframe_domain(series_domains: Sequence[SeriesDomain]) -> LazyFrameDomain: ...
def series_domain(name: str, element_domain: Domain) -> SeriesDomain: ...
def make_private_lazyframe(
    input_domain: Domain,
    input_metric: Metric,
    output_measure: Measure,
    lazyframe: Any,
    global_scale: float | None = None,
    threshold: int | None = None,
) -> Measurement: ...
def make_stable_lazyframe(
    input_domain: Domain, input_metric: Metric, lazyframe: Any, MO: RuntimeTypeDescriptor = "MI"
) -> Transformation: ...
def make_contingency_table(
    input_domain: LazyFrameDomain,
    input_metric: Metric,
    output_measure: Measure,
    d_in: list[Bound],
    d_out: float | tuple[float, float],
    keys: Mapping[str, Sequence[Any]] | None = None,
    cuts: Mapping[str, Sequence[float]] | None = None,
    table: ContingencyTable | None = None,
    algorithm: Algorithm = ...,
) -> tuple[Measurement, float | None, int | None]: ...
