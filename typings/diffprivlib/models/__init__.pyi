from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from diffprivlib.accountant import BudgetAccountant
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB as _GaussianNB

class GaussianNB(_GaussianNB):
    def __init__(
        self,
        *,
        epsilon: float = 1.0,
        bounds: tuple[ArrayLike, ArrayLike] | None = None,
        priors: ArrayLike | None = None,
        var_smoothing: float = 1e-09,
        random_state: int | None = None,
        accountant: BudgetAccountant | None = None,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> Self: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]: ...

class DecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        max_depth: int = 5,
        *,
        epsilon: float = 1.0,
        bounds: tuple[ArrayLike, ArrayLike] | None = None,
        classes: ArrayLike | None = None,
        random_state: int | None = None,
        accountant: BudgetAccountant | None = None,
        criterion: str | None = None,
        **unused_args: Any,
    ) -> None: ...
    def fit(
        self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None, check_input: bool = True
    ) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[np.int64]: ...

class RandomForestClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators: int = 10,
        *,
        epsilon: float = 1.0,
        bounds: tuple[ArrayLike, ArrayLike] | None = None,
        classes: ArrayLike | None = None,
        n_jobs: int = 1,
        verbose: int = 0,
        accountant: BudgetAccountant | None = None,
        random_state: int | None = None,
        max_depth: int = 5,
        warm_start: bool = False,
        shuffle: bool = False,
        **unused_args: Any,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[np.int64]: ...

class LogisticRegression(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        *,
        epsilon: float = 1.0,
        data_norm: float | None = None,
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 100,
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
        accountant: BudgetAccountant | None = None,
        **unused_args: Any,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[np.int64]: ...
