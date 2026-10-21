def get_noise_multiplier(
    *,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: int | None = None,
    steps: int | None = None,
    accountant: str = "rdp",
    epsilon_tolerance: float = 0.01,
) -> float: ...
