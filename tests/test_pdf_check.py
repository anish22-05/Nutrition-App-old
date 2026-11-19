import pymupdf
import pytest

from app.pdf_check import PdfInfo, PdfRejected, assess_text, garble_ratio, inspect_pdf
from tests.pdfs import make_pdf, text_page


def test_valid_pdf_passes():
    info = inspect_pdf(make_pdf([text_page(), text_page()]), max_pages=10)
    assert info.pages == 2


def test_not_a_pdf_rejected():
    with pytest.raises(PdfRejected, match="not a PDF"):
        inspect_pdf(b"<html>hello</html>", max_pages=10)


def test_truncated_file_rejected_with_actionable_message():
    # The real-world case: a browser "Print to PDF" that was cut off mid-file.
    with pytest.raises(PdfRejected):
        inspect_pdf(make_pdf([text_page()])[:300], max_pages=10)


def test_password_protected_rejected():
    locked = make_pdf([text_page()], encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    with pytest.raises(PdfRejected, match="password"):
        inspect_pdf(locked, max_pages=10)


def test_too_many_pages_rejected():
    with pytest.raises(PdfRejected, match="limit"):
        inspect_pdf(make_pdf([text_page()] * 3), max_pages=2)


def test_garble_ratio():
    assert garble_ratio(["clean text here"]) == 0.0
    assert garble_ratio(["�� ab"]) == pytest.approx(2 / 5)
    assert garble_ratio([]) == 0.0


def _meta(*pages):
    return [{"page": p} for p in pages]


def test_assess_rejects_scanned_document():
    # 1 page of text out of 10 -> it is a scan with a stray text page
    with pytest.raises(PdfRejected, match="scanned"):
        assess_text(PdfInfo(10, False), ["some text"], _meta(1), min_text_page_ratio=0.3, max_garble_ratio=0.05)


def test_assess_rejects_when_no_text_at_all():
    with pytest.raises(PdfRejected, match="OCR"):
        assess_text(PdfInfo(3, False), [], [], min_text_page_ratio=0.3, max_garble_ratio=0.05)


def test_assess_rejects_garbled_text():
    with pytest.raises(PdfRejected, match="garbled"):
        assess_text(PdfInfo(1, False), ["�" * 50 + "x"], _meta(1), min_text_page_ratio=0.3, max_garble_ratio=0.05)


def test_assess_warns_about_skipped_pages_and_repairs():
    pages, warnings = assess_text(
        PdfInfo(10, True), ["a"] * 4, _meta(1, 2, 3, 4), min_text_page_ratio=0.3, max_garble_ratio=0.05
    )
    assert pages == 4
    assert any("6 of 10 pages" in w for w in warnings)
    assert any("repaired" in w for w in warnings)


def test_assess_clean_document_has_no_warnings():
    _, warnings = assess_text(PdfInfo(2, False), ["a", "b"], _meta(1, 2), min_text_page_ratio=0.3, max_garble_ratio=0.05)
    assert warnings == []
