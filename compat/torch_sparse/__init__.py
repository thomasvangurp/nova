"""Inference import shim for PSICHIC, which only uses SparseTensor as a type."""


class SparseTensor:
    pass


__version__ = "compat"
