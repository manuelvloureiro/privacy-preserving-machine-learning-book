from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

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

class OnceFrame:
    def collect(self) -> pl.DataFrame: ...
    def lazy(self) -> pl.LazyFrame: ...

class LazyGroupByQuery:
    def agg(self, *aggs: pl.Expr | Any, **named_aggs: pl.Expr | Any) -> LazyFrameQuery: ...

class LazyFrameQuery:
    def filter(self, *predicates: pl.Expr | Any, **constraints: Any) -> LazyFrameQuery: ...
    def select(self, *exprs: pl.Expr | Any, **named_exprs: pl.Expr | Any) -> LazyFrameQuery: ...
    def with_columns(self, *exprs: pl.Expr | Any, **named_exprs: pl.Expr | Any) -> LazyFrameQuery: ...
    def group_by(self, *by: str | pl.Expr, maintain_order: bool = False, **named_by: Any) -> LazyGroupByQuery: ...
    def with_keys(self, keys: pl.LazyFrame | pl.DataFrame, on: list[str] | None = None) -> LazyFrameQuery: ...
    def truncate_per_group(
        self, k: int, by: list[Any] | None = None, keep: Literal["sample", "first", "last"] | SortBy = "sample"
    ) -> LazyFrameQuery: ...
    def truncate_num_groups(
        self, k: int, by: list[Any], keep: Literal["sample", "first", "last"] = "sample"
    ) -> LazyFrameQuery: ...
    def summarize(self, alpha: float | None = None) -> pl.DataFrame: ...
    def release(self) -> OnceFrame: ...

def dp_len(scale: float | None = None, signed: bool | None = None) -> pl.Expr: ...
