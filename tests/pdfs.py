"""Helpers that build small real PDFs in memory for tests."""
import pymupdf

SENTENCE = "Vitamin C helps the body absorb iron from plant foods and supports healthy skin and bones. "


def text_page(n_sentences: int = 14, tag: str = "") -> str:
    return " ".join(f"{tag} item {i}. {SENTENCE}" for i in range(n_sentences))


def make_pdf(pages: list[str], **save_kwargs) -> bytes:
    """One page per entry; an empty string makes a blank (image-only / scanned-like) page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(30, 30, 565, 810), text, fontsize=6)
    data = doc.tobytes(**save_kwargs)
    doc.close()
    return data
