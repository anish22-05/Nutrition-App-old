import pytest

from app import ingest_service as svc
from app.pdf_check import PdfRejected
from tests.pdfs import make_pdf, text_page

OWNER = "11111111-1111-4111-8111-111111111111"


class FakeDb:
    """Records calls; behaviour is set per test."""

    def __init__(self):
        self.existing = None
        self.dup = None
        self.user_docs = 0
        self.stored = []

    def get_document(self, doc_id):
        return self.existing

    def find_document_by_hash(self, h, scope, owner):
        return self.dup

    def count_user_documents(self, owner, excluding_doc_id=None):
        return self.user_docs

    def replace_document(self, doc_id, filename, rows, **kw):
        self.stored.append((doc_id, filename, len(rows), kw))
        return len(rows)


@pytest.fixture
def env(monkeypatch):
    fake = FakeDb()
    embedded = []
    monkeypatch.setattr(svc, "db", fake)
    monkeypatch.setattr(svc.llm, "embed_texts", lambda texts: embedded.append(len(texts)) or [[0.1, 0.2]] * len(texts))
    return fake, embedded


def good_pdf():
    return make_pdf([text_page(tag=f"p{i}") for i in range(3)])


def test_user_upload_success_stores_with_scope_owner_and_hash(env):
    fake, embedded = env
    res = svc.ingest_pdf(good_pdf(), "plan.pdf", doc_id="u-abc", scope="user", owner_id=OWNER)
    assert res["status"] == "success" and res["chunks_inserted"] > 0 and res["pages"] == 3
    doc_id, title, n, kw = fake.stored[0]
    assert (doc_id, title) == ("u-abc", "plan.pdf")
    assert kw["scope"] == "user" and kw["owner_id"] == OWNER and len(kw["file_hash"]) == 64
    assert embedded == [n]


def test_blank_scanned_pdf_rejected_before_any_embedding(env):
    fake, embedded = env
    with pytest.raises(PdfRejected):
        svc.ingest_pdf(make_pdf(["", "", ""]), "scan.pdf", doc_id="u-abc", scope="user", owner_id=OWNER)
    assert embedded == [] and fake.stored == []  # no money spent, nothing stored


def test_duplicate_file_is_a_noop(env):
    fake, embedded = env
    fake.dup = {"doc_id": "u-old", "filename": "plan.pdf"}
    res = svc.ingest_pdf(good_pdf(), "copy.pdf", doc_id="u-new", scope="user", owner_id=OWNER)
    assert res["status"] == "duplicate" and res["doc_id"] == "u-old"
    assert embedded == [] and fake.stored == []


def test_quota_enforced_for_users_only(env):
    fake, _ = env
    fake.user_docs = 5
    with pytest.raises(svc.QuotaExceeded):
        svc.ingest_pdf(good_pdf(), "a.pdf", doc_id="u-abc", scope="user", owner_id=OWNER)
    # core ingestion (admin) is not subject to the per-user quota
    assert svc.ingest_pdf(good_pdf(), "a.pdf", doc_id="core-a", scope="core")["status"] == "success"


def test_cannot_take_over_someone_elses_or_a_core_doc_id(env):
    fake, _ = env
    fake.existing = {"doc_id": "nutrition-v1", "scope": "core", "owner_id": None}
    with pytest.raises(svc.Conflict):
        svc.ingest_pdf(good_pdf(), "x.pdf", doc_id="nutrition-v1", scope="user", owner_id=OWNER)
    fake.existing = {"doc_id": "u-abc", "scope": "user", "owner_id": "someone-else"}
    with pytest.raises(svc.Conflict):
        svc.ingest_pdf(good_pdf(), "x.pdf", doc_id="u-abc", scope="user", owner_id=OWNER)


def test_replacing_your_own_document_is_allowed(env):
    fake, _ = env
    fake.existing = {"doc_id": "u-abc", "scope": "user", "owner_id": OWNER}
    assert svc.ingest_pdf(good_pdf(), "x.pdf", doc_id="u-abc", scope="user", owner_id=OWNER)["status"] == "success"


def test_oversized_user_document_rejected(env, monkeypatch):
    monkeypatch.setattr(svc.get_settings(), "user_max_chunks", 1)
    with pytest.raises(PdfRejected, match="too large"):
        svc.ingest_pdf(good_pdf(), "big.pdf", doc_id="u-abc", scope="user", owner_id=OWNER)


def test_user_scope_requires_owner(env):
    with pytest.raises(ValueError):
        svc.ingest_pdf(good_pdf(), "x.pdf", doc_id="u-abc", scope="user")
