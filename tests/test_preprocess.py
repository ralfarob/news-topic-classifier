"""Unit tests for preprocessing helpers."""

from src.preprocess import clean_batch, clean_text


def test_clean_text_basic() -> None:
    # Verifies lowercase conversion, punctuation removal, and space cleanup.
    assert clean_text(" Breaking: NEW Policy!!! ") == "breaking new policy"


def test_clean_text_none() -> None:
    assert clean_text(None) == ""


def test_clean_batch() -> None:
    items = ["Sports UPDATE!!", "  Tech news... "]
    assert clean_batch(items) == ["sports update", "tech news"]
