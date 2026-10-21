from collections.abc import Callable

from dp_accounting.dp_event import DpEvent
from dp_accounting.privacy_accountant import PrivacyAccountant

class BracketInterval: ...

class LowerEndpointAndGuess(BracketInterval):
    def __init__(self, lower_endpoint: float, initial_guess: float) -> None: ...

class ExplicitBracketInterval(BracketInterval):
    def __init__(self, endpoint_1: float, endpoint_2: float) -> None: ...

def calibrate_dp_mechanism(
    make_fresh_accountant: Callable[[], PrivacyAccountant],
    make_event_from_param: Callable[[float], DpEvent],
    target_epsilon: float,
    target_delta: float,
    bracket_interval: BracketInterval | None = None,
    discrete: bool = False,
    tol: float | None = None,
) -> float: ...
