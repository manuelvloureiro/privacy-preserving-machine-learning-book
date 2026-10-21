from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

class ConfidenceInterval:
    low: float
    high: float

class BinomTestResult:
    k: int
    n: int
    pvalue: float
    statistic: float
    def proportion_ci(
        self, confidence_level: float = 0.95, method: Literal["exact", "wilson", "wilsoncc"] = "exact"
    ) -> ConfidenceInterval: ...

def binomtest(
    k: int, n: int, p: float = 0.5, alternative: Literal["two-sided", "greater", "less"] = "two-sided"
) -> BinomTestResult: ...

class _Continuous:
    @overload
    def cdf(self, x: float, *args: float, loc: float = 0, scale: float = 1) -> float: ...
    @overload
    def cdf(self, x: NDArray[np.float64], *args: float, loc: float = 0, scale: float = 1) -> NDArray[np.float64]: ...
    @overload
    def sf(self, x: float, *args: float, loc: float = 0, scale: float = 1) -> float: ...
    @overload
    def sf(self, x: NDArray[np.float64], *args: float, loc: float = 0, scale: float = 1) -> NDArray[np.float64]: ...
    @overload
    def ppf(self, q: float, *args: float, loc: float = 0, scale: float = 1) -> float: ...
    @overload
    def ppf(self, q: NDArray[np.float64], *args: float, loc: float = 0, scale: float = 1) -> NDArray[np.float64]: ...
    @overload
    def isf(self, q: float, *args: float, loc: float = 0, scale: float = 1) -> float: ...
    @overload
    def isf(self, q: NDArray[np.float64], *args: float, loc: float = 0, scale: float = 1) -> NDArray[np.float64]: ...
    @overload
    def logpdf(self, x: float, *args: float, loc: float = 0, scale: float = 1) -> float: ...
    @overload
    def logpdf(
        self, x: NDArray[np.float64], *args: float, loc: float | ArrayLike = 0, scale: float | ArrayLike = 1
    ) -> NDArray[np.float64]: ...
    def fit(self, data: ArrayLike) -> tuple[float, ...]: ...

class _Discrete:
    @overload
    def cdf(self, k: float, n: int, p: float) -> float: ...
    @overload
    def cdf(self, k: NDArray[np.float64], n: int, p: float) -> NDArray[np.float64]: ...
    @overload
    def sf(self, k: float, n: int, p: float) -> float: ...
    @overload
    def sf(self, k: NDArray[np.float64], n: int, p: float) -> NDArray[np.float64]: ...
    def ppf(self, q: float, n: int, p: float) -> float: ...

norm: _Continuous
beta: _Continuous
laplace: _Continuous
skewnorm: _Continuous
binom: _Discrete
