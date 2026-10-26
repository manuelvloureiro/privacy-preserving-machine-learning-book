# OpenDP's NumPy extension (opendp/extras/numpy), 0.16.0.
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from opendp.mod import Domain, Metric, PartialConstructor, Transformation
from opendp.typing import RuntimeType, RuntimeTypeDescriptor

@dataclass(kw_only=True)
class NPArray2Domain:
    origin: NDArray[Any] | None
    norm: float | None
    p: Literal[1, 2, None]
    size: int | None
    num_columns: int | None
    nan: bool
    cardinalities: NDArray[Any] | None
    T: str | RuntimeType

@dataclass(kw_only=True)
class NPArrayDDomain:
    shape: tuple[int, ...]
    T: str | RuntimeType

def array2_domain(
    *,
    norm: float | None = None,
    p: Literal[1, 2, None] = None,
    origin: Any = None,
    size: int | None = None,
    num_columns: int | None = None,
    nan: bool | None = None,
    cardinalities: list[int] | NDArray[Any] | None = None,
    T: RuntimeTypeDescriptor | None = None,
) -> Domain: ...
def arrayd_domain(*, shape: tuple[int, ...], T: RuntimeTypeDescriptor) -> Domain: ...
def make_np_clamp(
    input_domain: Domain, input_metric: Metric, norm: float, p: Literal[1, 2], origin: Any = None
) -> Transformation: ...
def then_np_clamp(norm: float, p: Literal[1, 2], origin: Any = None) -> PartialConstructor[Transformation]: ...
