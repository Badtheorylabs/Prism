from auth import hash_password
from api import login, refresh_session


def test_login_ok():
    stored = hash_password("pw", "salt")
    response = login("u1", "pw", "salt", stored, "127.0.0.1")
    assert response["ok"]
    assert response["session"]["user_id"] == "u1"


def test_login_rejects_bad_password():
    stored = hash_password("pw", "salt")
    response = login("u1", "wrong", "salt", stored, "127.0.0.2")
    assert not response["ok"]
    assert response["error"] == "bad_credentials"


def test_refresh_session_skips_password_check():
    response = refresh_session("u1")
    assert response["ok"]
