"""Root directory for pair caches, keep indices, and training runs.

Set ``TRANSMISSIONS_MNT`` to the directory that holds ``pairs/`` and run
outputs. Defaults to ``~/transmissions_data``.
"""

from __future__ import annotations

import os
from pathlib import Path

MNT = Path(os.environ.get("TRANSMISSIONS_MNT", Path.home() / "transmissions_data"))
