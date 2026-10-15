"""Keep Python's bytecode caches out of the source tree when the tests run.

The tests run the book's scripts as subprocesses, which inherit this environment.
"""

import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
