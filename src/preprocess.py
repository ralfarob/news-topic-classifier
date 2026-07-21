import re
from typing import Iterable, List


def clean_text(text: str) -> str:
    if text is None:
        return ""

    value = text.lower().strip()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def clean_batch(texts: Iterable[str]) -> List[str]:
    return [clean_text(item) for item in texts]
