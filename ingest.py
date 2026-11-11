from __future__ import annotations

import argparse

from langchain_core.documents import Document
from pathlib import Path
from pypdf import PdfReader


def _sanitize_text(text: str) -> str:
    """Make PDF-extracted text safe for PostgreSQL text columns.

    Some PDFs contain malformed font/encoding mappings that pypdf exposes as
    NUL characters. PostgreSQL rejects NUL in TEXT, so replace it with a space
    rather than silently concatenating the surrounding tokens.
    """
    return text.replace("\x00", " ")


def parse_pdf(pdf_path: str | Path):
    path = Path(pdf_path)
    reader = PdfReader(path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        text = _sanitize_text(text) if text else text
        if text and text.strip():
            document = Document(
                page_content=text.strip(),
                metadata={
                    "source": path.name,
                    "page": page_number
                }
            )
            documents.append(document)

    return documents


def ingest(pdf_path: str | Path) -> int:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from rag import index_documents

    documents = parse_pdf(pdf_path)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120
    ).split_documents(documents)
    if chunks:
        index_documents(chunks)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest one PDF into the local knowledge base.")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF to ingest")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the configured vector collection before ingesting",
    )
    args = parser.parse_args()
    if args.reset:
        from rag import reset_collection

        reset_collection()
    print(f"Ingested {ingest(args.pdf_path)} chunks.")


if __name__ == "__main__":
    main()
