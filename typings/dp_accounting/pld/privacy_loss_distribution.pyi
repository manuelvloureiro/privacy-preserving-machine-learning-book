from collections.abc import Sequence
from typing import overload

import numpy as np

from dp_accounting.privacy_accountant import NeighboringRelation

class PrivacyLossDistribution:
    def get_epsilon_for_delta(self, delta: float) -> float: ...
    @overload
    def get_delta_for_epsilon(self, epsilon: float) -> float: ...
    @overload
    def get_delta_for_epsilon(self, epsilon: Sequence[float]) -> np.ndarray: ...
    def self_compose(self, num_times: int, tail_mass_truncation: float = 1e-15) -> "PrivacyLossDistribution": ...
    def compose(
        self, privacy_loss_distribution: "PrivacyLossDistribution", tail_mass_truncation: float = 1e-15
    ) -> "PrivacyLossDistribution": ...

def from_gaussian_mechanism(
    standard_deviation: float,
    sensitivity: float = 1,
    pessimistic_estimate: bool = True,
    value_discretization_interval: float = 1e-4,
    log_mass_truncation_bound: float = -50,
    sampling_prob: float = 1.0,
    use_connect_dots: bool = True,
    neighboring_relation: NeighboringRelation = ...,
) -> PrivacyLossDistribution: ...
def from_laplace_mechanism(
    parameter: float,
    sensitivity: float = 1,
    pessimistic_estimate: bool = True,
    value_discretization_interval: float = 1e-4,
    sampling_prob: float = 1.0,
    neighboring_relation: NeighboringRelation = ...,
) -> PrivacyLossDistribution: ...
