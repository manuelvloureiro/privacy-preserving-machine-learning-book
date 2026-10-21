# OpenDP 0.16.0 ships inline types, but its prelude re-exports everything through star imports with
# __all__ = ["t", "m", "c"], and many return types are unions or missing, which strict pyright rejects.
# These stubs cover the part of the API the practitioner scripts call, with the real signatures.

__version__: str
