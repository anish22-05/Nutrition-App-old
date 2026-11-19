import pytest

from app.ingestion import chunk_sentences, clean_text, split_sentences


def sents(n, words=12):
    return [f"Sentence {i} " + "word " * words + "ends here." for i in range(n)]


def test_clean_text_rejoins_hyphenated_words():
    assert clean_text("nutri-\ntion is key") == "nutrition is key"


def test_split_sentences():
    assert split_sentences("Eat well. Sleep well! Why not?") == ["Eat well.", "Sleep well!", "Why not?"]


def test_no_duplicate_tail_chunk():
    # exactly one window's worth of sentences -> exactly one chunk (used to emit a duplicate tail)
    out = list(chunk_sentences(sents(10), 10, 2, min_tokens=1))
    assert len(out) == 1


def test_overlap_between_consecutive_chunks():
    out = list(chunk_sentences(sents(18), 10, 2, min_tokens=1))
    assert len(out) == 2
    # last 2 sentences of chunk 0 are the first 2 of chunk 1
    assert "Sentence 8 " in out[0] and "Sentence 8 " in out[1]
    assert "Sentence 9 " in out[0] and "Sentence 9 " in out[1]


def test_every_sentence_is_covered_when_windows_are_trimmed_by_token_limit():
    # tiny max_tokens forces trimming; previously the trimmed sentences were skipped (data loss)
    data = sents(30)
    out = " ".join(chunk_sentences(data, 10, 2, max_tokens=60, min_tokens=1))
    for i in range(30):
        assert f"Sentence {i} " in out, f"sentence {i} was lost"


def test_short_chunks_are_dropped():
    assert list(chunk_sentences(["Too short."], 10, 2, min_tokens=50)) == []


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        list(chunk_sentences(sents(5), 3, 3))


def test_find_boilerplate_strips_repeated_lines_but_not_unique_ones():
    from app.ingestion import find_boilerplate

    pages = [["ACME Training Deck", f"Page {i}", f"Fact {chr(97 + i)} says iron helps blood."] for i in range(10)]
    junk = find_boilerplate(pages)
    assert "acme training deck" in junk and "page #" in junk
    assert not any("says iron" in j for j in junk)
    assert find_boilerplate(pages[:3]) == set()  # too few pages to judge
