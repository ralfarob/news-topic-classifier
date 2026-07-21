"""Pytest bootstrap for consistent import resolution across environments."""

from pathlib import Path
import sys


# Add repository root to sys.path so `import src...` works in CI and locally.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))