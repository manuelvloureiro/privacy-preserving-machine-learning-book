from dp_accounting.privacy_accountant import NeighboringRelation, PrivacyAccountant

class PLDAccountant(PrivacyAccountant):
    def __init__(
        self,
        neighboring_relation: NeighboringRelation = ...,
        value_discretization_interval: float = 1e-4,
    ) -> None: ...
