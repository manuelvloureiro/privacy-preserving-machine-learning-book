# Type stubs for untyped libraries

Hand-written `.pyi` stubs for the parts of the practitioner libraries that ship without type information
(Opacus, dp-accounting, diffprivlib, scikit-learn, SciPy, the Adversarial Robustness Toolbox). Pyright reads them
from here (its default `stubPath`), so the practitioner scripts pass strict mode without casts in the code the book
shows.

Each stub covers only what the scripts call. When a script needs another function or argument, add it here
with the library's real signature (check it with `inspect.signature` at the pinned version), not a looser one.
