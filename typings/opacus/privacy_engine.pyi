from os import PathLike
from typing import Any, TypeVar

import torch
from torch.utils.data import DataLoader

from opacus.accountants.accountant import IAccountant
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer

_T = TypeVar("_T")

class PrivacyEngine:
    accountant: IAccountant
    secure_mode: bool
    def __init__(self, *, accountant: str = "prv", secure_mode: bool = False) -> None: ...
    # With the default grad_sample_mode="hooks", Opacus returns (module, optimizer, data_loader); the
    # four-element return of grad_sample_mode="ghost" is not used by the book's scripts.
    def make_private(
        self,
        *,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module = ...,
        data_loader: DataLoader[_T],
        noise_multiplier: float,
        max_grad_norm: float | list[float],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator: torch.Generator | None = None,
        grad_sample_mode: str = "hooks",
        wrap_model: bool = True,
        rand_on_empty: bool = False,
    ) -> tuple[GradSampleModule, DPOptimizer, DataLoader[_T]]: ...
    def make_private_with_epsilon(
        self,
        *,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module = ...,
        data_loader: DataLoader[_T],
        target_epsilon: float,
        target_delta: float,
        epochs: int,
        max_grad_norm: float | list[float],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator: torch.Generator | None = None,
        grad_sample_mode: str = "hooks",
        wrap_model: bool = True,
    ) -> tuple[GradSampleModule, DPOptimizer, DataLoader[_T]]: ...
    def get_epsilon(self, delta: float) -> float: ...
    def save_checkpoint(
        self,
        *,
        path: str | PathLike[str],
        module: torch.nn.Module,
        optimizer: DPOptimizer | None = None,
        checkpoint_dict: dict[str, Any] | None = None,
    ) -> None: ...
    def load_checkpoint(
        self, *, path: str | PathLike[str], module: torch.nn.Module, optimizer: DPOptimizer | None = None
    ) -> dict[str, Any]: ...
