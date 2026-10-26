from collections.abc import Callable, Sequence

import jax
from jax.typing import ArrayLike

from mbi.factor import Factor

class LinearMeasurement:
    noisy_measurement: ArrayLike
    clique: tuple[str, ...]
    stddev: float
    query: Callable[[Factor], jax.Array]
    def __init__(
        self,
        noisy_measurement: ArrayLike,
        clique: Sequence[str],
        stddev: float = 1.0,
        query: Callable[[Factor], jax.Array] | None = None,
    ) -> None: ...

class MarginalLossFn:
    def __call__(self, marginals: object) -> jax.Array: ...
