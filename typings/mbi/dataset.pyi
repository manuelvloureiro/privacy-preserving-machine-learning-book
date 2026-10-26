from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mbi.domain import Domain
from mbi.factor import Factor

class Dataset:
    domain: Domain
    weights: NDArray[np.float64]
    def __init__(
        self,
        data: Mapping[str, NDArray[np.int64]] | NDArray[np.int64],
        domain: Domain,
        weights: NDArray[np.float64] | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, NDArray[np.int64]]: ...
    @property
    def df(self) -> Any: ...  # a pandas DataFrame, built on first use
    @staticmethod
    def synthetic(domain: Domain, N: int) -> Dataset: ...
    def project(self, cols: str | Sequence[str]) -> Factor: ...
    def supports(self, cols: str | Sequence[str]) -> bool: ...
    @property
    def records(self) -> int: ...
    def datavector(self, flatten: bool = True) -> NDArray[np.float64]: ...
