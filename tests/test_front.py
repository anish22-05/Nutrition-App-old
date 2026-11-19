"""The Streamlit page renders and behaves with a stubbed API (no network, no Supabase)."""
import json

import requests
from streamlit.testing.v1 import AppTest

DOCS = {"core": [{"doc_id": "c", "filename": "Human Nutrition (2020)", "num_chunks": 3}], "mine": []}


class FakeResp:
    ok = True
    status_code = 200

    def __init__(self, payload=None):
        self.payload = DOCS if payload is None else payload

    def json(self):
        return self.payload


class FakeStream:
    """Context-manager response for requests.post(..., stream=True)."""

    ok = True
    status_code = 200
    def __init__(self, events):
        self.events = events
        self.headers = {"content-type": "application/x-ndjson"}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self, decode_unicode=True):
        return (json.dumps(e) for e in self.events)


def app(monkeypatch, **state) -> AppTest:
    monkeypatch.setattr(requests, "request", lambda *a, **k: FakeResp())
    at = AppTest.from_file("../front.py", default_timeout=20)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def test_empty_chat_shows_welcome_examples_and_library(monkeypatch):
    at = app(monkeypatch)
    assert not at.exception
    assert at.title[0].value == "🥗 Nutrition Assistant"
    assert len([b for b in at.button if b.key and b.key.startswith("ex_")]) == 4
    assert any("Human Nutrition (2020)" in c.value for c in at.sidebar.caption)
    assert at.query_params["uid"]  # a private id was minted for this browser
    assert not at.toggle(key="tech").value  # technical details hidden by default


def test_login_mode_hides_upload_until_signed_in(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "http://localhost")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    sent = []
    monkeypatch.setattr(requests, "request", lambda *a, **k: sent.append(k["headers"]) or FakeResp())
    at = AppTest.from_file("../front.py", default_timeout=20).run()
    assert not at.exception
    assert "uid" not in at.query_params  # no self-minted id in login mode
    assert sent and "X-Owner-Id" not in sent[0] and "Authorization" not in sent[0]  # anonymous browse only
    assert any("Sign in above" in c.value for c in at.sidebar.caption)


def test_answer_details_are_one_compact_panel(monkeypatch):
    turn = {"role": "assistant", "content": "Adults need 15 mcg [T1]; it is made in the skin [S1].", "request_id": "r" * 32, "cached": True,
            "safety": {"category": "medical", "notice": "General information from the sources, not medical advice."},
            "tools": [{"id": "T1", "name": "recommended_intake", "arguments": '{"nutrient": "vitamin d"}',
                       "result": {"nutrient": "vitamin d", "value": 15, "unit": "mcg/day", "age": 40, "sex": "female",
                                  "source": "US DRIs"}}],
            "chunks": [{"doc_id": "c", "chunk_index": 1, "content": "Vitamin D is made in the skin.", "similarity": 0.8,
                        "metadata": {"page": 600, "source": "Human Nutrition (2020)", "section": "Chapter 9. Vitamins > Vitamin D"}}],
            "trace": {"embed": 120, "generate": 900}}
    at = app(monkeypatch, chat=[{"role": "user", "content": "How much vitamin D?"}, turn])
    assert not at.exception
    assert [e.label for e in at.expander if e.label.startswith("Sources")] == ["Sources (2)"]
    assert any("not medical advice" in i.value for i in at.info)
    text = " ".join(m.value for m in at.markdown)
    assert "Recommended daily intake" in text and "15 mcg/day" in text and "p. 600" in text
    assert not any("similarity" in c.value for c in at.caption)  # technical details stay hidden


def test_technical_details_toggle_shows_scores_and_timings(monkeypatch):
    turn = {"role": "assistant", "content": "Iron [S1].", "request_id": "q" * 32, "cached": False,
            "chunks": [{"doc_id": "c", "chunk_index": 4, "content": "Iron...", "similarity": 0.71, "metadata": {"page": 2}}],
            "trace": {"embed": 120, "generate": 900}}
    at = app(monkeypatch, chat=[{"role": "user", "content": "iron?"}, turn], tech=True)
    captions = " ".join(c.value for c in at.caption)
    assert "similarity 0.710" in captions and "generate 900 ms" in captions


def test_example_question_streams_an_answer_into_the_chat(monkeypatch):
    events = [
        {"type": "sources", "chunks": [{"doc_id": "c", "chunk_index": 0, "content": "Lentils and spinach contain iron.",
                                        "similarity": 0.8, "metadata": {"page": 612, "source": "Human Nutrition (2020)"}}],
         "standalone_question": "Which foods are good sources of iron?", "request_id": "x" * 32, "safety": None, "cached": False},
        {"type": "token", "text": "Lentils and spinach "},
        {"type": "token", "text": "[S1]."},
        {"type": "done", "trace": {"generate": 800}},
    ]
    posted = []
    monkeypatch.setattr(requests, "post", lambda url, **k: posted.append(k["json"]) or FakeStream(events))
    at = app(monkeypatch)
    at.button(key="ex_0").click().run()
    assert not at.exception
    assert posted[0]["query"] == "Which foods are good sources of iron?" and posted[0]["scope"] == "all"
    chat = at.session_state["chat"]
    assert chat[-1]["content"] == "Lentils and spinach [S1]." and chat[-1]["request_id"] == "x" * 32
    assert [e.label for e in at.expander if e.label.startswith("Sources")] == ["Sources (1)"]


def test_emergency_answer_is_an_alert_without_details(monkeypatch):
    turn = {"role": "assistant", "content": "This sounds like it could be an emergency...", "emergency": True,
            "request_id": "e" * 32, "safety": {"category": "emergency"}}
    at = app(monkeypatch, chat=[{"role": "user", "content": "I took too many pills"}, turn])
    assert any("emergency" in e.value for e in at.error)
    assert not [e for e in at.expander if e.label.startswith("Sources")]


def test_offline_api_shows_one_clear_banner(monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "request", down)
    at = AppTest.from_file("../front.py", default_timeout=20).run()
    assert not at.exception
    assert len(at.warning) == 1 and "offline" in at.warning[0].value
    assert not at.error  # no stack of raw error messages


def test_only_cited_passages_are_listed_unless_technical_details_are_on(monkeypatch):
    chunks = [{"doc_id": "c", "chunk_index": i, "content": f"passage {i}", "similarity": 0.7, "metadata": {"page": i}}
              for i in (1, 2, 3)]
    turn = {"role": "assistant", "content": "About 2046 kcal [T1], see also [S2].", "request_id": "t" * 32, "chunks": chunks,
            "tools": [{"name": "energy_needs", "arguments": "{}", "result": {"estimated_energy_needs_kcal_per_day": 2046}}]}
    chat = [{"role": "user", "content": "calories?"}, turn]
    at = app(monkeypatch, chat=chat)
    assert [e.label for e in at.expander if e.label.startswith("Sources")] == ["Sources (2)"]  # T1 + S2 only
    text = " ".join(m.value for m in at.markdown)
    assert "**[S2]**" in text and "**[S1]**" not in text and "**[S3]**" not in text
    at = app(monkeypatch, chat=chat, tech=True)
    assert [e.label for e in at.expander if e.label.startswith("Sources")] == ["Sources (4)"]
