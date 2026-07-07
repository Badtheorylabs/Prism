"""Path sandboxing for model-proposed file paths.

A small model's proposed FILE: path is untrusted input. Without this guard, an
absolute path (`/etc/x`) or a traversal (`../../x`) escapes the repo/overlay and
writes to the real filesystem — confirmed as a real bug. Every place that writes
a model-proposed path (execution overlay, agent --apply, harden --apply) must
route through `safe_join`.
"""

from __future__ import annotations

import os


def safe_join(root: str, relpath: str) -> str | None:
    """Join relpath under root, or return None if it escapes root.

    Rejects absolute paths, drive letters, and any `..` traversal that would
    resolve outside `root`.
    """
    if not relpath or os.path.isabs(relpath) or (os.path.splitdrive(relpath)[0]):
        return None
    root_abs = os.path.abspath(root)
    candidate = os.path.abspath(os.path.join(root_abs, relpath))
    if candidate == root_abs or candidate.startswith(root_abs + os.sep):
        return candidate
    return None
