from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

class BaseEstimator:
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> Self: ...

class ClassifierMixin:
    classes_: NDArray[Any]
    def score(self, X: ArrayLike, y: ArrayLike) -> float: ...

class TransformerMixin:
    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None) -> NDArray[np.float64]: ...
