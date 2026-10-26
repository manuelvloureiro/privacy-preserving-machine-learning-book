# Generated from the installed OpenDP 0.16.0 by the stub generator (see typings/README.md), then kept by hand:
# the real signatures, with annotations the package leaves out filled in as Any.
from typing import Any, TypeVar


from opendp.mod import (
    Function,
    Measurement,
    Transformation,
)
from opendp.typing import RuntimeTypeDescriptor

M = TypeVar("M", Transformation, Measurement)

def as_array() -> Function: ...
def new_function(function: Any, TO: RuntimeTypeDescriptor) -> Function: ...
def new_queryable(transition: Any, Q: RuntimeTypeDescriptor = 'ExtrinsicObject', A: RuntimeTypeDescriptor = 'ExtrinsicObject') -> Any: ...
