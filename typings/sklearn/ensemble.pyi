from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sklearn.base import BaseEstimator, ClassifierMixin

class RandomForestClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators: int = 100,
        *,
        max_depth: int | None = None,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[np.int64]: ...
