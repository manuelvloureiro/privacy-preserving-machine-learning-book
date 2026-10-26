# OpenDP's differentially private linear regression (Theil-Sen), opendp/extras/sklearn/linear_model, 0.16.0.
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from opendp.mod import Measure, Measurement

class LinearRegression:
    measurement: Measurement  # the Theil-Sen measurement that `fit` invokes; its `map` gives the privacy loss
    coef_: NDArray[np.float64]
    intercept_: float
    def __init__(
        self,
        output_measure: Measure,
        x_bounds: Iterable[tuple[float, float]],
        y_bounds: tuple[float, float],
        scale: float,
        runs: int = 1,
        candidates_count: int = 100,
        fraction_bounds: tuple[float, float] = (0.25, 0.75),
    ) -> None: ...
    def fit(self, X: Any, y: Any) -> LinearRegression: ...
    def predict(self, X: Any) -> NDArray[np.float64]: ...
