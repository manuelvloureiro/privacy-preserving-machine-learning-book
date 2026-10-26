import jax

from mbi.domain import Domain

class Factor:
    domain: Domain
    values: jax.Array
    def __init__(self, domain: Domain, values: jax.Array) -> None: ...
    def datavector(self, flatten: bool = True) -> jax.Array: ...
