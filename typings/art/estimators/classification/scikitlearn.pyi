from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

class ScikitlearnClassifier:
    nb_classes: int
    @property
    def input_shape(self) -> tuple[int, ...]: ...
    def __init__(
        self,
        model: BaseEstimator,
        clip_values: tuple[float, float] | None = None,
        preprocessing_defences: Any | None = None,
        postprocessing_defences: Any | None = None,
        preprocessing: tuple[float, float] = (0.0, 1.0),
        use_logits: bool = False,
    ) -> None: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> None: ...
    def predict(self, x: NDArray[np.float64], **kwargs: Any) -> NDArray[np.float64]: ...
    def clone_for_refitting(self) -> ScikitlearnClassifier: ...

def SklearnClassifier(
    model: BaseEstimator,
    clip_values: tuple[float, float] | None = None,
    preprocessing_defences: Any | None = None,
    postprocessing_defences: Any | None = None,
    preprocessing: tuple[float, float] = (0.0, 1.0),
    use_logits: bool = False,
) -> ScikitlearnClassifier: ...
