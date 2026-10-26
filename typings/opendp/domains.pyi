# Generated from the installed OpenDP 0.16.0 by the stub generator (see typings/README.md), then kept by hand:
# the real signatures, with annotations the package leaves out filled in as Any.
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Optional, Type, TypeVar, Union

import numpy as np
import polars as pl
from numpy.typing import NDArray

from opendp.mod import (
    ApproximateDivergence,
    AtomDomain,
    ChangeOneIdDistance,
    Domain,
    ExtrinsicDistance,
    ExtrinsicDivergence,
    ExtrinsicDomain,
    FrameDistance,
    Function,
    LazyFrameDomain,
    Measure,
    Measurement,
    Metric,
    Odometer,
    OdometerQueryable,
    OptionDomain,
    PartialConstructor,
    PrivacyProfile,
    Queryable,
    SeriesDomain,
    SymmetricIdDistance,
    Transformation,
    VectorDomain,
)
from opendp.typing import RuntimeType, RuntimeTypeDescriptor

M = TypeVar("M", Transformation, Measurement)

def array_domain(element_domain: Domain, width: int) -> Domain: ...
def atom_domain(bounds: Any = None, nan: Any = None, T: RuntimeTypeDescriptor | None = None) -> AtomDomain: ...
def bitvector_domain(max_weight: Any = None) -> Domain: ...
def categorical_domain(categories: Any = None) -> Domain: ...
def datetime_domain(time_unit: str = 'us', time_zone: str | None = None) -> Domain: ...
def enum_domain(categories: Any) -> Domain: ...
def lazyframe_domain(series_domains: Any) -> LazyFrameDomain: ...
def map_domain(key_domain: Domain, value_domain: Domain) -> Domain: ...
def option_domain(element_domain: Domain, D: RuntimeTypeDescriptor | None = None) -> OptionDomain: ...
def series_domain(name: str, element_domain: Domain) -> SeriesDomain: ...
def user_domain(identifier: str, member: Any, descriptor: Any = None) -> Domain: ...
def vector_domain(atom_domain: Domain, size: Any = None) -> VectorDomain: ...
def wild_expr_domain(columns: Any, margin: Any = None) -> Domain: ...
def with_margin(frame_domain: Domain, margin: Any) -> LazyFrameDomain: ...
