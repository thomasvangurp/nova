"""Minimal stand-in for bittensor, for running BoltzgenWrapper OFF-CHAIN.

boltzgen_wrapper.py imports bittensor solely to call bt.logging.*. Installing
the real chain stack into the bgen env would drag in a second torch/numpy ABI
next to torch 2.7.1 and is exactly how the bz2 env got broken before. This
supplies only the logging surface actually used.

Deliberately NOT in site-packages: it lives in /root/bgen_shim and is reached
via PYTHONPATH, so it is visible, scoped to the scoring harness, and cannot
shadow a real bittensor for anything else.
"""
import sys


class _Logging:
    @staticmethod
    def _emit(level, *args, **kwargs):
        msg = " ".join(str(a) for a in args)
        print(f"[bt.{level}] {msg}", file=sys.stderr, flush=True)

    def info(self, *a, **k):     self._emit("info", *a, **k)
    def warning(self, *a, **k):  self._emit("warning", *a, **k)
    def error(self, *a, **k):    self._emit("error", *a, **k)
    def debug(self, *a, **k):    self._emit("debug", *a, **k)
    def success(self, *a, **k):  self._emit("success", *a, **k)
    def trace(self, *a, **k):    pass
    def exception(self, *a, **k): self._emit("exception", *a, **k)
    def set_trace(self, *a, **k): pass


logging = _Logging()
__version__ = "0.0.0-shim"
