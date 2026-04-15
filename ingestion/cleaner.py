import re
from langchain_core.documents import Document


def _basic_cleanup(text: str) -> str:
    """
    General cleanup for both PDF and OCR text
    """

    if not text:
        return ""

    # Normalize line breaks
    text = text.replace("\n", " ")

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove weird unicode artifacts
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Fix spacing before punctuation
    text = re.sub(r"\s+([.,!?])", r"\1", text)

    # Remove repeated punctuation
    text = re.sub(r"\.{2,}", ".", text)

    return text.strip()


def _ocr_cleanup(text: str) -> str:
    """
    Extra cleaning for OCR text (more aggressive)
    """

    # Fix split numbers: "5 0 0" → "500"
    text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)

    # Fix units spacing
    text = re.sub(r"(\d)(mg|ml|g)", r"\1 \2", text)

    # Remove junk characters but keep medical symbols
    text = re.sub(r"[^a-zA-Z0-9.,/%\-+ ]", " ", text)

    return text


def _medical_normalization(text: str) -> str:
    """
    Normalize common medical abbreviations
    """

    text = re.sub(r"\bbp\b", "blood pressure", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhr\b", "heart rate", text, flags=re.IGNORECASE)
    text = re.sub(r"\btemp\b", "temperature", text, flags=re.IGNORECASE)

    return text


def clean_documents(docs: list[Document]) -> list[Document]:
    """
    Main function to clean a list of LangChain Documents
    """

    cleaned_docs = []

    for doc in docs:
        text = doc.page_content

        # Step 1: Basic cleanup
        text = _basic_cleanup(text)

        # Step 2: OCR-specific cleanup (detect via metadata if possible)
        if doc.metadata.get("type") == "ocr":
            text = _ocr_cleanup(text)

        # Step 3: Medical normalization
        text = _medical_normalization(text)

        cleaned_docs.append(
            Document(
                page_content=text,
                metadata=doc.metadata  # preserve metadata
            )
        )

    return cleaned_docs
import re
from langchain_core.documents import Document


def _basic_cleanup(text: str) -> str:
    """
    General cleanup for both PDF and OCR text.
    """
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)

    return text.strip()


def _ocr_cleanup(text: str) -> str:
    """
    Extra cleanup for OCR text.
    """
    text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)
    text = re.sub(r"(\d)(mg|ml|g)", r"\1 \2", text)
    text = re.sub(r"[^a-zA-Z0-9.,/%\-+ ]", " ", text)

    return text


def _medical_normalization(text: str) -> str:
    """
    Normalize common medical abbreviations.
    """
    text = re.sub(r"\bbp\b", "blood pressure", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhr\b", "heart rate", text, flags=re.IGNORECASE)
    text = re.sub(r"\btemp\b", "temperature", text, flags=re.IGNORECASE)

    return text


def clean_documents(docs: list[Document]) -> list[Document]:
    """
    Clean a list of LangChain Documents.
    """
    cleaned_docs = []

    for doc in docs:
        text = doc.page_content
        text = _basic_cleanup(text)

        if doc.metadata.get("type") == "ocr":
            text = _ocr_cleanup(text)

        text = _medical_normalization(text)

        cleaned_docs.append(
            Document(
                page_content=text,
                metadata=doc.metadata,
            )
        )

    return cleaned_docs
