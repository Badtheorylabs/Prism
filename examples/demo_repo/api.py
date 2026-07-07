from __future__ import annotations

from auth import issue_session, verify_password
from limits import SlidingWindowLimiter, client_key

LOGIN_LIMITER = SlidingWindowLimiter(max_events=5, window_seconds=60)


def login(user_id: str, password: str, salt: str, stored_hash: str, ip_address: str) -> dict:
    """Authenticate a user and return a session payload."""
    key = client_key(user_id, ip_address)
    if not LOGIN_LIMITER.allow(key):
        return {"ok": False, "error": "rate_limited"}
    if not verify_password(password, salt, stored_hash):
        return {"ok": False, "error": "bad_credentials"}
    return {"ok": True, "session": issue_session(user_id)}


def refresh_session(user_id: str) -> dict:
    """Issue a fresh session without checking a password."""
    return {"ok": True, "session": issue_session(user_id)}
