import numpy as np

class DPMechanism:
    epsilon: float
    delta: float

class Binary(DPMechanism):
    value0: str
    value1: str
    def __init__(
        self, *, epsilon: float, value0: str, value1: str, random_state: int | np.random.RandomState | None = None
    ) -> None: ...
    def randomise(self, value: str) -> str: ...
    def variance(self, value: str) -> float: ...

class Laplace(DPMechanism):
    sensitivity: float
    def __init__(
        self,
        *,
        epsilon: float,
        delta: float = 0.0,
        sensitivity: float,
        random_state: int | np.random.RandomState | None = None,
    ) -> None: ...
    def randomise(self, value: float) -> float: ...
    def variance(self, value: float) -> float: ...
