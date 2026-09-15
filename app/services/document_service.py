"""
Document-level concerns: validating the uploaded file and extracting raw
text from it. Deliberately does NOT chunk (that's ingestion_service.py) or
touch the vector store (that's Step 6) — one job per module.
"""

import io

from fastapi import UploadFile
from pypdf import PdfReader

from app.core.logging import get_logger
from app.utils.exceptions import EmptyDocumentError, UnsupportedFileTypeError

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def _get_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def validate_file(file: UploadFile) -> str:
    """
    Checks the filename has a supported extension. Returns the extension
    (e.g. ".pdf") so callers don't have to re-derive it.

    Raises UnsupportedFileTypeError for anything else — caught by the
    global exception handler and turned into a clean 415 response.
    """
    filename = file.filename or ""
    ext = _get_extension(filename)

    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext or 'unknown'}'. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return ext


def extract_pages(raw_bytes: bytes, ext: str) -> list[str]:
    """
    Extracts text as a list of page strings — one entry per PDF page, or a
    single entry for .txt (which has no page concept).

    This replaces the old extract_text()->single-string approach. Preserving
    page boundaries here, instead of joining everything into one blob before
    chunking, is what lets ingestion_service.chunk_pages() tag each chunk
    with a real page_number — which is what powers "filename, page N"
    source citations in rag_service.py, instead of only ever being able to
    say "filename, chunk N".

    Raises EmptyDocumentError if no page has usable text — e.g. a
    scanned/image-only PDF with no text layer, or a blank .txt file.
    """
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        pages = [p for p in pages if p]  # drop pages with no extractable text
    elif ext == ".txt":
        text = raw_bytes.decode("utf-8", errors="ignore").strip()
        pages = [text] if text else []
    else:
        # Defensive — validate_file() should have already rejected this.
        raise UnsupportedFileTypeError(f"Unsupported file type '{ext}'.")

    if not pages:
        raise EmptyDocumentError(
            "No extractable text found in the uploaded document. "
            "If this is a PDF, it may be a scanned image with no text layer."
        )

    logger.info("Extracted %d page(s) of text (ext=%s)", len(pages), ext)
    return pages
