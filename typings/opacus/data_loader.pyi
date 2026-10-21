from typing import TypeVar

import torch
from torch.utils.data import DataLoader

_T = TypeVar("_T")

class DPDataLoader(DataLoader[_T]):
    sample_rate: float
    @classmethod
    def from_data_loader(
        cls, data_loader: DataLoader[_T], *, generator: torch.Generator | None = None, rand_on_empty: bool = False
    ) -> "DPDataLoader[_T]": ...
