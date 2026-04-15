import fitz
import pytesseract
from PIL import Image
import io
from langchain_core.documents import Document


def fast_doc_loader(file_path):

    docs = []

    # IMAGE
    if file_path.lower().endswith((".png",".jpg",".jpeg")):

        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)

        docs.append(Document(page_content=text, metadata={"source":file_path}))
        return docs


    # PDF
    if file_path.lower().endswith(".pdf"):

        pdf = fitz.open(file_path)

        for page_num in range(len(pdf)):

            page = pdf.load_page(page_num)

            text = page.get_text("text")

            # DIGITAL PAGE
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"page":page_num,"source":file_path}
                    )
                )

            # SCANNED PAGE
            else:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes()))

                ocr_text = pytesseract.image_to_string(img)

                docs.append(
                    Document(
                        page_content=ocr_text,
                        metadata={"page":page_num,"source":file_path}
                    )
                )

        return docs