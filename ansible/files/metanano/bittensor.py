"""Minimal stand-in for bittensor inside the Layer-2 filter env.

NOVA-nanobody-filter imports bittensor from metanano/utils/cdr_utils.py, but the
whole package uses exactly one symbol -- bt.logging.warning (verified by grep
across metanano/). Installing real bittensor here drags in
async-substrate-interface, whose scalecodec/cyscale namespace clash and ScaleObj
version skew break the import outright. The filters do no chain I/O, so the
dependency is transitive noise rather than a capability we are dropping.
"""
import logging as _logging

logging = _logging.getLogger("metanano")
if not logging.handlers:
    logging.addHandler(_logging.NullHandler())
