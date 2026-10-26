from typing import Any

import numpy as np
from numpy.typing import NDArray

from art.estimators.classification.scikitlearn import ScikitlearnClassifier

_Array = NDArray[np.float64]

class MembershipInferenceBlackBox:
    def __init__(
        self,
        estimator: ScikitlearnClassifier,
        input_type: str = "prediction",
        attack_model_type: str = "nn",
        attack_model: Any | None = None,
        scaler_type: str | None = "standard",
        nn_model_epochs: int = 100,
        nn_model_batch_size: int = 100,
        nn_model_learning_rate: float = 0.0001,
    ) -> None: ...
    def fit(
        self,
        x: _Array | None = None,
        y: _Array | None = None,
        test_x: _Array | None = None,
        test_y: _Array | None = None,
        pred: _Array | None = None,
        test_pred: _Array | None = None,
        **kwargs: Any,
    ) -> None: ...
    def infer(self, x: _Array, y: _Array | None = None, **kwargs: Any) -> _Array: ...

class MembershipInferenceBlackBoxRuleBased:
    def __init__(self, classifier: ScikitlearnClassifier) -> None: ...
    def infer(self, x: _Array, y: _Array | None = None, **kwargs: Any) -> _Array: ...

class ShadowModels:
    def __init__(
        self,
        shadow_model_template: ScikitlearnClassifier,
        num_shadow_models: int = 3,
        disjoint_datasets: bool = False,
        random_state: int | None = None,
    ) -> None: ...
    def generate_shadow_dataset(
        self, x: _Array, y: _Array, member_ratio: float = 0.5
    ) -> tuple[tuple[_Array, _Array, _Array], tuple[_Array, _Array, _Array]]: ...
