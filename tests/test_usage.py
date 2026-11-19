from types import SimpleNamespace as NS

from app import llm, usage


def test_cost_arithmetic_and_unknown_model():
    u = usage.Usage()
    u.add("gpt-4o-mini", 1_000_000, 1_000_000)
    assert round(u.cost_usd, 2) == 0.75
    u.add("some-new-model", 500)
    assert u.calls == 2 and round(u.cost_usd, 2) == 0.75  # unknown = free, no crash


def test_record_outside_request_is_a_noop():
    usage._current.set(None)
    usage.record("gpt-4o-mini", 10, 10)  # must not raise


def test_embedding_and_chat_calls_are_attributed(monkeypatch):
    emb = NS(data=[NS(embedding=[0.1])], usage=NS(total_tokens=1000))
    stream = [NS(choices=[NS(delta=NS(content="hi", tool_calls=None))], usage=None),
              NS(choices=[], usage=NS(prompt_tokens=200, completion_tokens=50))]
    client = NS(embeddings=NS(create=lambda **k: emb), chat=NS(completions=NS(create=lambda **k: iter(stream))))
    monkeypatch.setattr(llm, "get_client", lambda: client)
    u = usage.start()
    llm.embed_texts(["x"])
    llm.generate_answer("q", [{"content": "c", "metadata": {}, "doc_id": "d"}])
    assert (u.calls, u.input_tokens, u.output_tokens) == (2, 1200, 50)


def test_stream_records_usage_from_final_event(monkeypatch):
    events = [
        NS(choices=[NS(delta=NS(content="a", tool_calls=None))], usage=None),
        NS(choices=[], usage=NS(prompt_tokens=10, completion_tokens=2)),
    ]
    client = NS(chat=NS(completions=NS(create=lambda **k: iter(events))))
    monkeypatch.setattr(llm, "get_client", lambda: client)
    u = usage.start()
    assert list(llm.answer_events("q", [{"content": "c", "metadata": {}, "doc_id": "d"}])) == [{"type": "token", "text": "a"}]
    assert u.input_tokens == 10 and u.output_tokens == 2
