from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sklearn.base import BaseEstimator, ClassifierMixin

class LogisticRegression(ClassifierMixin, BaseEstimator):
    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]
    def __init__(
        self,
        *,
        C: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
        solver: str = "lbfgs",
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[np.int64]: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]: ...
    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]: ...
