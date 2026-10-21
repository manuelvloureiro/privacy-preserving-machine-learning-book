from typing import Any

import torch

class GradSampleModule(torch.nn.Module):
    _module: torch.nn.Module
    def __init__(
        self,
        m: torch.nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        strict: bool = True,
        force_functorch: bool = False,
    ) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def remove_hooks(self) -> None: ...
    def disable_hooks(self) -> None: ...
    def enable_hooks(self) -> None: ...
