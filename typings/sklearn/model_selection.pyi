from typing import Any

from numpy.typing import NDArray

def train_test_split(
    *arrays: NDArray[Any],
    test_size: float | int | None = None,
    train_size: float | int | None = None,
    random_state: int | None = None,
    shuffle: bool = True,
    stratify: NDArray[Any] | None = None,
) -> list[NDArray[Any]]: ...
