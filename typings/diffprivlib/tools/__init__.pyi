from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from diffprivlib.accountant import BudgetAccountant

# With axis=None (the default) these return one value; with an axis they return an array.
def mean(
    array: ArrayLike,
    epsilon: float = 1.0,
    bounds: tuple[float, float] | None = None,
    axis: int | tuple[int, ...] | None = None,
    dtype: Any = None,
    keepdims: bool = False,
    random_state: int | np.random.RandomState | None = None,
    accountant: BudgetAccountant | None = None,
    **unused_args: Any,
) -> float: ...
def median(
    array: ArrayLike,
    epsilon: float = 1.0,
    bounds: tuple[float, float] | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    random_state: int | np.random.RandomState | None = None,
    accountant: BudgetAccountant | None = None,
    **unused_args: Any,
) -> float: ...
def count_nonzero(
    array: ArrayLike,
    epsilon: float = 1.0,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    random_state: int | np.random.RandomState | None = None,
    accountant: BudgetAccountant | None = None,
) -> int: ...
