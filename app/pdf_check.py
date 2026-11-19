"""Is this PDF actually worth ingesting?  Rejects bad files with a message a user can act on.

Real failures this was written for: a browser "Print to PDF" of a huge page that was cut off
mid-file (opens as 0 pages), a PDF whose pages are all scanned images (no text layer), and
files whose text extracts as mojibake. Without checks each of those is "successfully" ingested
as nothing, or as garbage that then pollutes every answer.
"""
from dataclasses import dataclass

import pymupdf

pymupdf.TOOLS.mupdf_display_errors(False)  # corrupt embedded images print noisy, harmless warnings


class PdfRejected(ValueError):
    """The message is shown to the end user as-is, so keep it plain and actionable."""


@dataclass
class PdfInfo:
    pages: int
    repaired: bool


def inspect_pdf(file_bytes: bytes, max_pages: int) -> PdfInfo:
    """Structural checks that need no text extraction."""
    if not file_bytes.startswith(b"%PDF"):
        raise PdfRejected("This file is not a PDF.")
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise PdfRejected("This PDF could not be opened; the file looks corrupt.") from e
    try:
        if doc.needs_pass:
            raise PdfRejected("This PDF is password-protected. Remove the password and upload it again.")
        pages = doc.page_count
        if pages == 0:
            raise PdfRejected(
                "This PDF has no readable pages. That usually means the download or 'Print to PDF' was "
                "cut off before the file finished. Download or export the original file instead."
            )
        if max_pages and pages > max_pages:
            raise PdfRejected(f"This PDF has {pages} pages; the limit for uploads is {max_pages}.")
        return PdfInfo(pages=pages, repaired=bool(doc.is_repaired))
    finally:
        doc.close()


def garble_ratio(texts: list[str]) -> float:
    """Share of characters that are replacement glyphs or control codes (extraction mojibake)."""
    total = sum(len(t) for t in texts)
    if not total:
        return 0.0
    bad = sum(ch == "�" or (ord(ch) < 32 and ch not in "\n\t\r") for t in texts for ch in t)
    return bad / total


def assess_text(
    info: PdfInfo,
    chunks: list[str],
    metas: list[dict],
    *,
    min_text_page_ratio: float,
    max_garble_ratio: float,
) -> tuple[int, list[str]]:
    """Judge the extracted text. Returns (pages_with_text, warnings) or raises PdfRejected."""
    if not chunks:
        raise PdfRejected(
            "No extractable text found. If this is a scan or a photo of a document it needs OCR first; "
            "upload a version where you can select the text."
        )
    pages_with_text = len({m["page"] for m in metas})
    ratio = pages_with_text / info.pages
    if ratio < min_text_page_ratio:
        raise PdfRejected(
            f"Only {pages_with_text} of {info.pages} pages contain readable text; this looks like a scanned "
            "document. Upload a text-based PDF."
        )
    if garble_ratio(chunks) > max_garble_ratio:
        raise PdfRejected("The text in this PDF extracts as garbled characters (broken font encoding).")

    warnings: list[str] = []
    if ratio < 0.7:
        warnings.append(f"{info.pages - pages_with_text} of {info.pages} pages had no usable text and were skipped.")
    if info.repaired:
        warnings.append("The file had structural damage and was repaired while reading; check the content.")
    return pages_with_text, warnings
