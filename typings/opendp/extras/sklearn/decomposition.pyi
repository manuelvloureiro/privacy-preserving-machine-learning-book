# OpenDP's differentially private PCA (opendp/extras/sklearn/decomposition), 0.16.0.
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from opendp.mod import Domain, Measurement, Metric, PartialConstructor

@dataclass(kw_only=True)
class PCAEpsilons:
    eigvals: float
    eigvecs: Sequence[float]
    mean: float | None

class PCA:
    """A scikit-learn-style estimator: `fit` releases the mean, eigenvalues and eigenvectors under `epsilon`."""

    n_components_: int
    components_: NDArray[np.float64]
    explained_variance_: NDArray[np.float64]
    explained_variance_ratio_: NDArray[np.float64]
    singular_values_: NDArray[np.float64]
    mean_: NDArray[np.float64]
    noise_variance_: float
    n_samples_: int
    def __init__(
        self,
        *,
        epsilon: float,
        row_norm: float,
        n_samples: int,
        n_features: int,
        n_components: int | float | str | None = None,
        n_changes: int = 1,
        whiten: bool = False,
    ) -> None: ...
    @property
    def n_features(self) -> int: ...
    def fit(self, X: Any, y: Any = None) -> PCA: ...
    def measurement(self) -> Measurement: ...
    def transform(self, X: Any) -> NDArray[np.float64]: ...
    def fit_transform(self, X: Any, y: Any = None) -> NDArray[np.float64]: ...

def make_private_pca(
    input_domain: Domain,
    input_metric: Metric,
    unit_epsilon: float | PCAEpsilons,
    norm: float | None = None,
    num_components: int | None = None,
) -> Measurement: ...
def then_private_pca(
    unit_epsilon: float | PCAEpsilons, norm: float | None = None, num_components: int | None = None
) -> PartialConstructor[Measurement]: ...
