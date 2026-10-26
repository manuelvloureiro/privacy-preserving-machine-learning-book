from collections.abc import Callable
from typing import Protocol

from mbi.clique_vector import CliqueVector
from mbi.domain import Domain
from mbi.marginal_loss import LinearMeasurement, MarginalLossFn
from mbi.markov_random_field import MarkovRandomField

class Estimator(Protocol):
    def __call__(
        self, domain: Domain, loss_fn: MarginalLossFn | list[LinearMeasurement], *, known_total: float | None = None
    ) -> MarkovRandomField: ...

def mirror_descent(
    domain: Domain,
    loss_fn: MarginalLossFn | list[LinearMeasurement],
    *,
    known_total: float | None = None,
    potentials: CliqueVector | None = None,
    iters: int = 1000,
    stateful: bool = False,
    stepsize: float | None = None,
    callback_fn: Callable[[CliqueVector], None] = ...,
) -> MarkovRandomField: ...
