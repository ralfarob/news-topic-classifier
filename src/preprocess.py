"""Text normalization helpers used by training and prediction."""

import re
from typing import Iterable, List


def clean_text(text: str) -> str:
    # Handle missing text safely so downstream vectorization does not fail.
    if text is None:
        return ""

    # Keep a simple, consistent normalization pipeline for all entry points.
    value = text.lower().strip()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def clean_batch(texts: Iterable[str]) -> List[str]:
    # Reuse single-item cleaning logic to keep behavior identical.
    return [clean_text(item) for item in texts]
