from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sklearn.base import BaseEstimator, TransformerMixin

class StandardScaler(TransformerMixin, BaseEstimator):
    mean_: NDArray[np.float64]
    scale_: NDArray[np.float64]
    def __init__(self, *, with_mean: bool = True, with_std: bool = True) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self: ...
    def transform(self, X: ArrayLike) -> NDArray[np.float64]: ...

class OneHotEncoder(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        categories: str | list[list[Any]] = "auto",
        handle_unknown: str = "error",
        sparse_output: bool = True,
        dtype: type = ...,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self: ...
    def transform(self, X: ArrayLike) -> NDArray[np.float64]: ...
