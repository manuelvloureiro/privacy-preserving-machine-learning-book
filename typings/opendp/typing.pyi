# OpenDP's type descriptors (opendp/typing.py), 0.16.0: the strings and RuntimeType objects passed as `T=`, `MO=`
# and the like. A RuntimeTypeDescriptor is any of a RuntimeType, a string such as "f64", a Python type such as float,
# or a tuple of these, so it is Any here.
from collections.abc import Sequence
from typing import Any

RuntimeTypeDescriptor = Any

class RuntimeType:
    origin: str
    args: Sequence[RuntimeType | str]
    def __init__(self, origin: str, args: Sequence[RuntimeType | str] | None = None) -> None: ...
    @classmethod
    def parse(cls, type_name: Any, generics: Sequence[str] | None = None) -> RuntimeType | str: ...
    @classmethod
    def infer(cls, public_example: Any, py_object: bool = False) -> RuntimeType | str: ...
    def substitute(self, **kwargs: Any) -> RuntimeType | str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class SensitivityMetric(RuntimeType):
    def __getitem__(self, associated_type: Any) -> SensitivityMetric: ...

class Carrier(RuntimeType):
    def __getitem__(self, subdomains: Any) -> Carrier: ...

SymmetricDistance: str
InsertDeleteDistance: str
ChangeOneDistance: str
HammingDistance: str
DiscreteDistance: str
AbsoluteDistance: SensitivityMetric
L1Distance: SensitivityMetric
L2Distance: SensitivityMetric
MaxDivergence: str
SmoothedMaxDivergence: str
FixedSmoothedMaxDivergence: str
ZeroConcentratedDivergence: str
Vec: Carrier
NDArray: Carrier
HashMap: Carrier
i8: str
i16: str
i32: str
i64: str
i128: str
isize: str
u8: str
u16: str
u32: str
u64: str
u128: str
usize: str
f32: str
f64: str
String: str
BitVector: str
LazyFrame: str
DataFrame: str
Series: str
Expr: str

def set_default_int_type(T: RuntimeTypeDescriptor) -> None: ...
def set_default_float_type(T: RuntimeTypeDescriptor) -> None: ...
