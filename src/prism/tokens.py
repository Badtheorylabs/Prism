"""Token estimation.

v1 uses a cheap char/4 heuristic so the core has zero dependencies and runs
on-device. If `tiktoken` is installed we use it for accuracy. The rest of the
system only depends on `estimate()`, so swapping in a real tokenizer for a
specific model is a one-line change.
"""

from __future__ import annotations

_ENCODER = None
try:  # optional accuracy upgrade, never required
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken is optional
    _ENCODER = None


def estimate(text: str) -> int:
    """Approximate token count for a piece of text."""
    if not text:
        return 0
    if _ENCODER is not None:  # pragma: no cover - depends on optional dep
        return len(_ENCODER.encode(text))
    # Heuristic: ~4 chars per token, min 1 token for any non-empty text.
    return max(1, len(text) // 4)
