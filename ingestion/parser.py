
try:
    import fitz
except Exception:
    fitz = None

import pytesseract
from pytesseract import TesseractNotFoundError
from PIL import Image
import io
import shutil
import os
from pypdf import PdfReader
from langchain_core.documents import Document

def is_valid_text(text: str) -> bool:
    text = text.strip()

    if len(text) < 100:
        return False

    if "http" in text and len(text.split()) < 20:
        return False

    return True


def _resolve_tesseract_cmd() -> str | None:
    candidates = [
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for cmd in candidates:
        if cmd and (shutil.which(cmd) or os.path.exists(cmd)):
            return cmd
    return None


def _ocr_available() -> bool:
    cmd = _resolve_tesseract_cmd()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
        return True
    return False


def _safe_ocr(img: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(img)
    except TesseractNotFoundError:
        return ""

def fast_doc_loader(file_path):

    ocr_enabled = _ocr_available()

    if file_path.lower().endswith((".png", ".jpg", ".jpeg")):

        img = Image.open(file_path)
        text = _safe_ocr(img) if ocr_enabled else ""

        yield Document(
            page_content=text,
            metadata={
                "source": file_path,
                "page": 1,
                "type": "ocr" if text else "ocr_unavailable"
            }
        )
        return

    if file_path.lower().endswith(".pdf"):

        if fitz is not None:
            pdf = fitz.open(file_path)

            for page_num in range(len(pdf)):

                page = pdf.load_page(page_num)

                text = page.get_text("text")

                use_ocr = False

                if not text or len(text.strip()) < 200:
                    use_ocr = True
                elif not is_valid_text(text):
                    use_ocr = True

                if use_ocr and ocr_enabled:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes()))
                    ocr_text = _safe_ocr(img)

                    if ocr_text.strip():
                        text = ocr_text
                        doc_type = "ocr"
                    else:
                        doc_type = "text_fallback"

                elif use_ocr and not ocr_enabled:
                    doc_type = "ocr_unavailable"
                else:
                    doc_type = "text"

                yield Document(
                    page_content=text,
                    metadata={
                        "page": page_num + 1,
                        "source": file_path,
                        "type": doc_type
                    }
                )

            return

        # Fallback when PyMuPDF cannot be loaded (e.g. blocked DLL policy).
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            yield Document(
                page_content=text,
                metadata={
                    "page": page_num,
                    "source": file_path,
                    "type": "text_no_fitz"
                }
            )
        return

    raise ValueError("Unsupported file type")