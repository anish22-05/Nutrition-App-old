"""PDF -> cleaned text -> overlapping sentence chunks. Pure functions, no network."""
import re
from collections.abc import Iterator

import pymupdf  # PyMuPDF
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")  # tokenizer used by OpenAI embeddings


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def clean_text(t: str) -> str:
    t = t.replace("\r", " ")
    t = re.sub(r"-\s*\n\s*", "", t)  # re-join words hyphenated across lines
    t = re.sub(r"\s+\n", "\n", t)
    return t.replace("\n", " ").strip()


def split_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]


def chunk_sentences(
    sents: list[str],
    sents_per_chunk: int = 10,
    overlap: int = 2,
    max_tokens: int = 1300,
    min_tokens: int = 50,
) -> Iterator[str]:
    """Sliding window over sentences.

    Fixes vs. the first version:
      * if a window is trimmed to respect `max_tokens`, the next window starts from
        the trimmed end (minus overlap) - before, the dropped sentences were skipped
        entirely and silently lost;
      * stop once the window reaches the last sentence - before, the tail was emitted
        again as a fully duplicated chunk.
    """
    if overlap >= sents_per_chunk:
        raise ValueError("overlap must be smaller than sents_per_chunk")

    i, n = 0, len(sents)
    while i < n:
        piece = sents[i : i + sents_per_chunk]
        chunk = " ".join(piece)
        while max_tokens and count_tokens(chunk) > max_tokens and len(piece) > 1:
            piece = piece[:-1]
            chunk = " ".join(piece)

        if count_tokens(chunk) >= min_tokens:
            yield chunk

        end = i + len(piece)
        if end >= n:
            break
        i += max(1, len(piece) - overlap)


def _line_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())  # "Page 12" and "Page 13" are the same line


def find_boilerplate(pages: list[list[str]], min_share: float = 0.3, min_pages: int = 5) -> set[str]:
    """Lines repeated on many pages (running headers, footers, page numbers, slide banners).

    They carry no content, but every chunk would start with the same sentence, which pulls all
    chunk embeddings toward each other and dilutes the real signal. Needs a minimum page count
    so a short document's genuinely repeated line is never stripped."""
    if len(pages) < min_pages:
        return set()
    seen: dict[str, int] = {}
    for lines in pages:
        for key in {_line_key(x) for x in lines if x.strip()}:
            seen[key] = seen.get(key, 0) + 1
    return {k for k, n in seen.items() if n / len(pages) >= min_share and len(k) > 1}


def pdf_pages_from_bytes(file_bytes: bytes, strip_boilerplate: bool = True) -> Iterator[tuple[int, str]]:
    """Yield (1-based page number, cleaned text)."""
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    try:
        raw = [(page.get_text("text") or "").splitlines() for page in doc]
    finally:
        doc.close()
    junk = find_boilerplate(raw) if strip_boilerplate else set()
    for i, lines in enumerate(raw):
        kept = [x for x in lines if _line_key(x) not in junk]
        yield i + 1, clean_text("\n".join(kept))


def page_sections(file_bytes: bytes, max_repeats: int = 2) -> dict[int, str]:
    """Page -> "Chapter > Section" from the PDF outline (bookmarks); empty if it has none.

    A chunk like "Its absorption is reduced by tannins..." doesn't say *what* is absorbed; the
    section title ("Iron") does. Titles repeated many times (publisher boilerplate in the outline)
    are ignored."""
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    try:
        toc, n_pages = doc.get_toc(simple=True), doc.page_count
    finally:
        doc.close()
    counts: dict[str, int] = {}
    for _lvl, title, _page in toc:
        counts[title.strip()] = counts.get(title.strip(), 0) + 1
    entries = sorted(
        ((page, lvl, title.strip()) for lvl, title, page in toc if title.strip() and counts[title.strip()] <= max_repeats and page >= 1),
        key=lambda e: e[0],
    )
    if not entries:
        return {}
    out: dict[int, str] = {}
    path: dict[int, str] = {}
    j = 0
    for page in range(1, n_pages + 1):
        while j < len(entries) and entries[j][0] <= page:
            _, lvl, title = entries[j]
            path = {k: v for k, v in path.items() if k < lvl}
            path[lvl] = title
            j += 1
        if path:
            out[page] = " > ".join(path[k] for k in sorted(path))[:200]
    return out


def embedding_text(chunk: str, meta: dict, contextual: bool = True) -> str:
    """What gets embedded: the chunk, prefixed by its document/section when that is known."""
    if not contextual:
        return chunk
    section = meta.get("section")
    return f"Section: {section}\n{chunk}" if section else chunk


def build_chunks_from_pdf(
    file_bytes: bytes,
    filename: str,
    *,
    sents_per_chunk: int = 10,
    overlap: int = 2,
    max_tokens: int = 1300,
    min_tokens: int = 50,
) -> tuple[list[str], list[dict]]:
    """Return (chunk_texts, metadatas). `chunk_index` in metadata is GLOBAL per
    document (it used to restart on every page and clash with the DB column)."""
    texts: list[str] = []
    metas: list[dict] = []
    sections = page_sections(file_bytes)

    for page_num, text in pdf_pages_from_bytes(file_bytes):
        if not text:
            continue
        for chunk in chunk_sentences(
            split_sentences(text), sents_per_chunk, overlap, max_tokens, min_tokens
        ):
            metas.append(
                {
                    "page": page_num,
                    "source": filename,
                    "chunk_index": len(texts),
                    "n_tokens": count_tokens(chunk),
                    **({"section": sections[page_num]} if page_num in sections else {}),
                }
            )
            texts.append(chunk)

    return texts, metas
