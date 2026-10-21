from opacus.accountants.accountant import IAccountant

class PRVAccountant(IAccountant):
    def get_epsilon(self, delta: float, *, eps_error: float = 0.01, delta_error: float | None = None) -> float: ...
