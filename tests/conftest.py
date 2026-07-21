from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that shared, memory, models,
# coordinator, subagent, and tools can be imported directly.
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# asyncio_mode = "auto" is configured in pytest.ini at the project root.
# All async test functions in this package automatically run under asyncio.
