from __future__ import annotations

import hashlib


def hash_password(password: str, salt: str) -> str:
    """Hash a password before it is stored."""
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    """Check a plaintext password against a stored hash."""
    return hash_password(password, salt) == stored_hash


def issue_session(user_id: str) -> dict:
    """Create the session payload returned by the API."""
    return {"user_id": user_id, "scopes": ["read", "write"]}
