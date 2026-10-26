from collections.abc import Sequence

from mbi.factor import Factor

class CliqueVector:
    cliques: Sequence[tuple[str, ...]]
    def expand(self, cliques: Sequence[tuple[str, ...]]) -> CliqueVector: ...
    def __getitem__(self, clique: tuple[str, ...]) -> Factor: ...
