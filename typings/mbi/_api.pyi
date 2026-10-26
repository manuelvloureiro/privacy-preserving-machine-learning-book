from collections.abc import Sequence
from typing import Protocol

from mbi.dataset import Dataset
from mbi.domain import Domain
from mbi.factor import Factor

class Projectable(Protocol):
    @property
    def domain(self) -> Domain: ...
    def project(self, attrs: str | Sequence[str]) -> Factor: ...
    def supports(self, attrs: str | Sequence[str]) -> bool: ...

class Model(Projectable, Protocol):
    @property
    def total(self) -> float: ...
    def synthetic_data(self, rows: int | None = None) -> Dataset: ...
