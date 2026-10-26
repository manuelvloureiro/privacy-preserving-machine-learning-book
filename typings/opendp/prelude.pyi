# OpenDP 0.16.0 ships inline types, but its prelude re-exports everything through star imports with
# __all__ = ["t", "m", "c"], and many return types are unions or missing, which strict pyright rejects.
# These stubs cover the API the book's scripts call, with the real signatures (typings/README.md).
from opendp import combinators as combinators
from opendp import measurements as measurements
from opendp import transformations as transformations
from opendp.accuracy import *
from opendp.context import *
from opendp.core import as_array as as_array
from opendp.core import new_function as new_function
from opendp.core import new_queryable as new_queryable
from opendp.domains import *
from opendp.extras import examples as examples
from opendp.extras import mbi as mbi
from opendp.extras import numpy as numpy
from opendp.extras import polars as polars
from opendp.extras import sklearn as sklearn
from opendp.extras.polars import dp_len as dp_len
from opendp.measures import *
from opendp.metrics import *
from opendp.mod import *
from opendp.typing import *

__version__: str

# The prelude's three shortcut modules. A plain `import ... as m` is not a re-export in a stub, an assignment is.
t = transformations
m = measurements
c = combinators
len = dp_len  # `dp.len(...)`: the frame length as a DP expression
