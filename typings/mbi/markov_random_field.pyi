from collections.abc import Sequence

from mbi.clique_vector import CliqueVector
from mbi.dataset import Dataset
from mbi.domain import Domain
from mbi.factor import Factor

class MarkovRandomField:
    potentials: CliqueVector
    marginals: CliqueVector
    total: float
    def project(self, attrs: str | Sequence[str]) -> Factor: ...
    def supports(self, attrs: str | Sequence[str]) -> bool: ...
    def synthetic_data(self, rows: int | None = None, method: str = "round") -> Dataset: ...
    @property
    def domain(self) -> Domain: ...
    @property
    def cliques(self) -> Sequence[tuple[str, ...]]: ...
