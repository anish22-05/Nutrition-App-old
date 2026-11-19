import argparse
import pickle

import numpy as np

from app import admin


def _store(tmp_path):
    state = {
        "documents": {
            "book": {"doc_id": "book", "filename": "Book", "scope": "core", "owner_id": None, "file_hash": "h1"},
            "u-x": {"doc_id": "u-x", "filename": "mine.pdf", "scope": "user", "owner_id": "o", "file_hash": "h2"},
        },
        "chunks": {
            "book": {"rows": [{"chunk_index": 0, "content": "iron", "metadata": {"page": 1}},
                              {"chunk_index": 1, "content": "zinc", "metadata": {"page": 2}}],
                     "vectors": np.eye(2, dtype=np.float32)},
            "u-x": {"rows": [{"chunk_index": 0, "content": "private", "metadata": {}}], "vectors": np.ones((1, 2))},
        },
    }
    p = tmp_path / "store.pkl"
    p.write_bytes(pickle.dumps(state))
    return p


def test_import_local_copies_core_docs_with_their_embeddings(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(admin.db, "replace_document",
                        lambda doc_id, filename, rows, **kw: calls.append((doc_id, filename, rows, kw)) or len(rows))
    rc = admin.cmd_import_local(argparse.Namespace(path=str(_store(tmp_path)), doc_id=None))
    assert rc == 0
    assert len(calls) == 1  # the private upload is not copied
    doc_id, filename, rows, kw = calls[0]
    assert (doc_id, filename, kw) == ("book", "Book", {"scope": "core", "file_hash": "h1"})
    assert [r["content"] for r in rows] == ["iron", "zinc"]
    assert rows[1]["embedding"] == [0.0, 1.0] and isinstance(rows[1]["embedding"][0], float)


def test_import_local_refuses_when_target_is_the_local_store(tmp_path, monkeypatch):
    monkeypatch.setattr(admin.get_settings(), "vector_store", "local")
    assert admin.cmd_import_local(argparse.Namespace(path=str(_store(tmp_path)), doc_id=None)) == 2
