import pytest
from fastapi.testclient import TestClient

from app import api
from app.ingest_service import Conflict, QuotaExceeded
from app.pdf_check import PdfRejected

OWNER_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
OWNER_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
PDF = {"file": ("My Plan.pdf", b"%PDF-1.4 fake", "application/pdf")}


@pytest.fixture
def client():
    return TestClient(api.app)


def test_upload_requires_valid_owner_header(client):
    assert client.post("/upload", files=PDF).status_code == 400
    assert client.post("/upload", files=PDF, headers={"X-Owner-Id": "not-a-uuid"}).status_code == 400


def test_upload_rejects_non_pdf_extension(client):
    r = client.post("/upload", files={"file": ("notes.txt", b"hi", "text/plain")}, headers={"X-Owner-Id": OWNER_A})
    assert r.status_code == 400


def test_upload_passes_server_chosen_private_doc_id(client, monkeypatch):
    seen = {}

    def fake_ingest(data, filename, **kw):
        seen.update(kw, filename=filename)
        return {"status": "success", "doc_id": kw["doc_id"], "title": filename, "chunks_inserted": 3, "pages": 1,
                "pages_with_text": 1, "warnings": []}

    monkeypatch.setattr(api, "ingest_pdf", fake_ingest)
    r = client.post("/upload", files=PDF, headers={"X-Owner-Id": OWNER_A})
    assert r.status_code == 200
    assert seen["scope"] == "user" and seen["owner_id"] == OWNER_A
    assert seen["doc_id"].startswith("u-")  # can never collide with a core id like 'nutrition-v1'


def test_user_doc_id_is_per_owner_and_deterministic():
    a1 = api.user_doc_id(OWNER_A, "Plan.pdf")
    assert a1 == api.user_doc_id(OWNER_A, "plan.PDF")  # same document -> same id (re-upload replaces)
    assert a1 != api.user_doc_id(OWNER_B, "Plan.pdf")  # two users, same filename -> different docs


@pytest.mark.parametrize(
    ("exc", "status"),
    [(PdfRejected("cut off"), 422), (QuotaExceeded("limit"), 409), (Conflict("taken"), 409), (RuntimeError("boom"), 502)],
)
def test_ingestion_errors_map_to_http_codes(client, monkeypatch, exc, status):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(api, "ingest_pdf", boom)
    r = client.post("/upload", files=PDF, headers={"X-Owner-Id": OWNER_A})
    assert r.status_code == status
    if status != 502:
        assert r.json()["detail"] == str(exc)  # user-facing message passes through
    else:
        assert "boom" not in r.text  # internal errors are not leaked


def test_delete_only_own_user_documents(client, monkeypatch):
    docs = {
        "u-mine": {"doc_id": "u-mine", "scope": "user", "owner_id": OWNER_A},
        "u-theirs": {"doc_id": "u-theirs", "scope": "user", "owner_id": OWNER_B},
        "core-1": {"doc_id": "core-1", "scope": "core", "owner_id": None},
    }
    deleted = []
    monkeypatch.setattr(api.db, "get_document", docs.get)
    monkeypatch.setattr(api.db, "delete_document", deleted.append)
    h = {"X-Owner-Id": OWNER_A}

    assert client.delete("/documents/u-theirs", headers=h).status_code == 404
    assert client.delete("/documents/core-1", headers=h).status_code == 404
    assert client.delete("/documents/missing", headers=h).status_code == 404
    assert client.delete("/documents/u-mine").status_code == 404  # no owner header
    assert deleted == []
    assert client.delete("/documents/u-mine", headers=h).status_code == 200
    assert deleted == ["u-mine"]


def test_documents_listing_passes_owner(client, monkeypatch):
    monkeypatch.setattr(api.db, "list_documents", lambda owner: {"core": [], "mine": [owner]})
    assert client.get("/documents", headers={"X-Owner-Id": OWNER_A}).json()["mine"] == [OWNER_A]
    assert client.get("/documents").json()["mine"] == [None]


def test_search_forwards_owner_and_scope(client, monkeypatch):
    calls = []
    monkeypatch.setattr(api.rag, "retrieve", lambda *a: calls.append(a) or [])
    client.post("/search", json={"query": "iron", "scope": "mine"}, headers={"X-Owner-Id": OWNER_A})
    q, _, _, min_sim, owner, scope = calls[0]
    assert (q, owner, scope, min_sim) == ("iron", OWNER_A, "mine", 0.0)


def test_search_rejects_unknown_scope_and_bad_owner(client):
    assert client.post("/search", json={"query": "iron", "scope": "everyone"}).status_code == 422
    assert client.post("/search", json={"query": "iron"}, headers={"X-Owner-Id": "nope"}).status_code == 400


