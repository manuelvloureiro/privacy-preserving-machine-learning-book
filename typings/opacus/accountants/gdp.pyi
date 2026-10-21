from opacus.accountants.accountant import IAccountant

class GaussianAccountant(IAccountant):
    def get_epsilon(self, delta: float, poisson: bool = True) -> float: ...
