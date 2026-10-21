from types import TracebackType
from typing import Generic, TypeVar

from torch.utils.data import DataLoader

from opacus.optimizers import DPOptimizer

_T = TypeVar("_T")

class BatchMemoryManager(Generic[_T]):
    def __init__(
        self, *, data_loader: DataLoader[_T], max_physical_batch_size: int, optimizer: DPOptimizer
    ) -> None: ...
    def __enter__(self) -> DataLoader[_T]: ...
    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None: ...
