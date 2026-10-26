from typing import Any

import numpy as np
from numpy.typing import NDArray

from art.estimators.classification.scikitlearn import ScikitlearnClassifier

_Array = NDArray[np.float64]

class AttributeInferenceBlackBox:
    def __init__(
        self,
        estimator: ScikitlearnClassifier,
        attack_model_type: str = "nn",
        attack_model: Any | None = None,
        attack_feature: int | slice = 0,
        is_continuous: bool | None = False,
        scale_range: tuple[float, float] | None = None,
        prediction_normal_factor: float | None = 1,
        scaler_type: str | None = "standard",
        non_numerical_features: list[int] | None = None,
        encoder: Any | None = None,
        nn_model_epochs: int = 100,
        nn_model_batch_size: int = 100,
        nn_model_learning_rate: float = 0.0001,
    ) -> None: ...
    def fit(self, x: _Array, y: NDArray[Any] | None = None) -> None: ...
    def infer(self, x: _Array, y: NDArray[Any] | None = None, **kwargs: Any) -> _Array: ...

class AttributeInferenceBaselineTrueLabel:
    def __init__(
        self,
        attack_model_type: str = "nn",
        attack_model: Any | None = None,
        attack_feature: int | slice = 0,
        is_continuous: bool | None = False,
        is_regression: bool | None = False,
        scale_range: tuple[float, float] | None = None,
        prediction_normal_factor: float = 1,
        non_numerical_features: list[int] | None = None,
        encoder: Any | None = None,
        nn_model_epochs: int = 100,
        nn_model_batch_size: int = 100,
        nn_model_learning_rate: float = 0.0001,
    ) -> None: ...
    def fit(self, x: _Array, y: NDArray[Any]) -> None: ...
    def infer(self, x: _Array, y: NDArray[Any] | None = None, **kwargs: Any) -> _Array: ...
