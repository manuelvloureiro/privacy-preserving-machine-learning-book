from typing import Any, Type

from opendp.context import Query
from opendp.extras.mbi import ContingencyTable

class ContingencyTableQuery(Query):
    """A contingency-table query: `.release()` gives the fitted table, whose `.synthesize()` draws synthetic rows."""

    oneway_scale: float | None
    oneway_threshold: int | None
    def release(
        self, data: Any = None, bounds: tuple[float, float] | None = None, T: Type[float] | Type[int] | None = None
    ) -> ContingencyTable: ...
