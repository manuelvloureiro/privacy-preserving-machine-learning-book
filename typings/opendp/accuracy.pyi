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

def accuracy_to_discrete_gaussian_scale(accuracy: Any, alpha: Any, T: RuntimeTypeDescriptor | None = None) -> Any: ...
def accuracy_to_discrete_laplacian_scale(accuracy: Any, alpha: Any, T: RuntimeTypeDescriptor | None = None) -> Any: ...
def accuracy_to_gaussian_scale(accuracy: Any, alpha: Any, T: RuntimeTypeDescriptor | None = None) -> Any: ...
def accuracy_to_laplacian_scale(accuracy: Any, alpha: Any, T: RuntimeTypeDescriptor | None = None) -> Any: ...
def discrete_gaussian_scale_to_accuracy(scale: Any, alpha: Any, T: RuntimeTypeDescriptor | None = None) -> Any: ...
def discrete_laplacian_scale_to_accuracy(scale: Any, alpha: Any, T: RuntimeTypeDescriptor | None = None) -> Any: ...
def gaussian_scale_to_accuracy(scale: Any, alpha: Any, T: RuntimeTypeDescriptor | None = None) -> Any: ...
def laplacian_scale_to_accuracy(scale: Any, alpha: Any, T: RuntimeTypeDescriptor | None = None) -> Any: ...
def summarize_polars_measurement(measurement: Measurement, alpha: Any = None) -> Any: ...
