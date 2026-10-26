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

def absolute_distance(T: RuntimeTypeDescriptor) -> Metric: ...
def change_one_distance() -> Metric: ...
def change_one_id_distance(identifier: Any) -> ChangeOneIdDistance: ...
def discrete_distance() -> Metric: ...
def frame_distance(inner_metric: Metric) -> Metric: ...
def hamming_distance() -> Metric: ...
def insert_delete_distance() -> Metric: ...
def l01inf_distance(metric: Metric) -> Metric: ...
def l02inf_distance(metric: Metric) -> Metric: ...
def l1_distance(T: RuntimeTypeDescriptor) -> Metric: ...
def l2_distance(T: RuntimeTypeDescriptor) -> Metric: ...
def linf_distance(T: RuntimeTypeDescriptor, monotonic: bool = False) -> Metric: ...
def symmetric_distance() -> Metric: ...
def symmetric_id_distance(identifier: Any) -> SymmetricIdDistance: ...
def user_distance(identifier: str, descriptor: Any = None) -> Metric: ...
