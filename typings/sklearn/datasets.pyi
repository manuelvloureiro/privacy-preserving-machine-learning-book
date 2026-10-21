from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

class Bunch:
    data: NDArray[np.float64]
    target: NDArray[np.int64]
    feature_names: list[str]
    DESCR: str

@overload
def load_breast_cancer(*, return_X_y: Literal[True]) -> tuple[NDArray[np.float64], NDArray[np.int64]]: ...
@overload
def load_breast_cancer(*, return_X_y: Literal[False] = False) -> Bunch: ...
@overload
def load_digits(*, n_class: int = 10, return_X_y: Literal[True]) -> tuple[NDArray[np.float64], NDArray[np.int64]]: ...
@overload
def load_digits(*, n_class: int = 10, return_X_y: Literal[False] = False) -> Bunch: ...
@overload
def load_diabetes(
    *, return_X_y: Literal[True], scaled: bool = True
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
@overload
def load_diabetes(*, return_X_y: Literal[False] = False, scaled: bool = True) -> Bunch: ...
