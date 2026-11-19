import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import api, auth

UID = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
OTHER = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"


class FakeAuthClient:
    def __init__(self):
        self.calls = 0
        self.auth = self

    def get_user(self, token):
        self.calls += 1
        if token != "good":
            raise RuntimeError("bad jwt")
        return type("R", (), {"user": type("U", (), {"id": UID})()})()


@pytest.fixture
def sb(monkeypatch):
    fake = FakeAuthClient()
    monkeypatch.setattr(auth.db, "get_client", lambda: fake)
    auth._cache.clear()
    return fake


def set_auth_required(monkeypatch, value):
    monkeypatch.setattr(auth.get_settings(), "auth_required", value)
    monkeypatch.setattr(api.settings, "auth_required", value)


def test_valid_token_gives_user_id_and_is_cached(sb):
    assert auth.resolve_owner(None, "Bearer good") == UID
    assert auth.resolve_owner(None, "bearer good") == UID
    assert sb.calls == 1  # second call served from cache


def test_bad_token_or_header_is_401(sb):
    for h in ("Bearer forged", "Basic abc", "Bearer "):
        with pytest.raises(HTTPException) as e:
            auth.resolve_owner(None, h)
        assert e.value.status_code == 401


def test_dev_mode_trusts_owner_header(sb, monkeypatch):
    set_auth_required(monkeypatch, False)
    assert auth.resolve_owner(OTHER, None) == OTHER
    assert auth.resolve_owner(None, None) is None


def test_auth_required_ignores_self_minted_owner_header(sb, monkeypatch):
    set_auth_required(monkeypatch, True)
    assert auth.resolve_owner(OTHER, None) is None  # the old stopgap no longer grants identity
    assert auth.resolve_owner(OTHER, "Bearer good") == UID  # the token wins over the header


def test_upload_needs_sign_in_when_auth_required(sb, monkeypatch):
    set_auth_required(monkeypatch, True)
    c = TestClient(api.app)
    pdf = {"file": ("a.pdf", b"%PDF-1.4", "application/pdf")}
    assert c.post("/upload", files=pdf, headers={"X-Owner-Id": OTHER}).status_code == 401
    monkeypatch.setattr(api.db, "list_documents", lambda owner: {"core": [], "mine": [owner]})
    assert c.get("/documents", headers={"Authorization": "Bearer good"}).json()["mine"] == [UID]
    assert c.get("/documents").json()["mine"] == [None]  # anonymous may still browse the shared library
