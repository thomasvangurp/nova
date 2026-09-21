"""Subnet-68 nanobody tracking dashboard.

Two stages, deliberately separate:

``collect``  pulls the public leaderboard and the slot-manager decision log
             into a local SQLite file.  It is incremental and idempotent, so
             it is safe to run on a timer.
``build``    reads that SQLite file and writes one self-contained ``index.html``.

Keeping them apart means a failed fetch never destroys a working dashboard,
and the page can be rebuilt offline from whatever has already been collected.
"""

DEFAULT_COLDKEY = "5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ"
EPOCH_LENGTH = 361
BLOCK_SECONDS = 12
