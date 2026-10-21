from diffprivlib.utils import Budget

class BudgetAccountant:
    epsilon: float
    delta: float
    slack: float
    spent_budget: list[tuple[float, float]]
    def __init__(
        self,
        epsilon: float = ...,
        delta: float = 1.0,
        slack: float = 0.0,
        spent_budget: list[tuple[float, float]] | None = None,
    ) -> None: ...
    def total(self, spent_budget: list[tuple[float, float]] | None = None, slack: float | None = None) -> Budget: ...
    def remaining(self, k: int = 1) -> Budget: ...
    def check(self, epsilon: float, delta: float) -> bool: ...
    def spend(self, epsilon: float, delta: float) -> BudgetAccountant: ...
