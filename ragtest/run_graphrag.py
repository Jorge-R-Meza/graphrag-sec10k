"""GraphRAG CLI wrapper that pre-registers ragtest extensions.

Run as:
    uv run python ragtest/run_graphrag.py index --root ragtest
    uv run python ragtest/run_graphrag.py query --root ragtest --method global \
        --query "..."
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root (parent of `ragtest/`) is importable so `ragtest.extensions`
# resolves regardless of where the script is launched from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Inject the OS trust store into Python's SSL globally. Required on corporate
# networks that MITM TLS: certifi's bundle doesn't include the corp CA.
try:
    import truststore  # type: ignore

    truststore.inject_into_ssl()
except ImportError:
    pass

# Load ragtest/.env BEFORE the config loader sees any ${VAR} references.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
except ImportError:
    pass

# Importing this package registers the local embedding provider with the factory.
import ragtest.extensions  # noqa: E402, F401

from graphrag.cli.main import app  # noqa: E402


if __name__ == "__main__":
    app(prog_name="graphrag")
