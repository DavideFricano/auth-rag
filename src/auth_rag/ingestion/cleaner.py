from __future__ import annotations

import re


class Cleaner:
    """Normalizes raw text before chunking or embedding."""

    @staticmethod
    def make_plain(text: str) -> str:
        """Collapses all whitespace sequences to a single space."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def normalize_newlines(text: str) -> str:
        """Collapses 3+ consecutive newlines to 2, preserving paragraph breaks."""
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def remove_control_chars(text: str) -> str:
        """Removes non-printable control characters, keeping newlines and tabs."""
        return re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
