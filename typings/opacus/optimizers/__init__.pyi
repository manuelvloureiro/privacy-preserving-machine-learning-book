from typing import Any

import torch

class DPOptimizer(torch.optim.Optimizer):
    original_optimizer: torch.optim.Optimizer
    noise_multiplier: float
    max_grad_norm: float
    expected_batch_size: int | None
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: int | None,
        loss_reduction: str = "mean",
        generator: torch.Generator | None = None,
        secure_mode: bool = False,
    ) -> None: ...
    @property
    def params(self) -> list[torch.nn.Parameter]: ...
    def zero_grad(self, set_to_none: bool = False) -> None: ...
    def step(self, closure: Any = None) -> torch.Tensor | None: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    def pre_step(self, closure: Any = None) -> bool | None: ...
    def clip_and_accumulate(self) -> None: ...
    def add_noise(self) -> None: ...
