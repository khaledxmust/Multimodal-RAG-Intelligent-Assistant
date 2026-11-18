import argparse
import hashlib
import re
import uuid

from pathlib import Path

from docx import Document
from pypdf import PdfReader
from PIL import Image, ImageOps
import pypdfium2 as pdfium
import pytesseract

from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
)

from retrieval.embeddings import EmbeddingModel
from retrieval.vector_store import (
    COLLECTION_NAME,
    get_point_count,
    create_collection,
    get_qdrant_client,
)


# ============================================================
# CONFIG
# ============================================================

MAX_CHUNK_CHARS = 1200
OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 40
MIN_OCR_CHARS = 5
BATCH_SIZE = 32

from config.settings import OCR_LANGUAGES, TESSERACT_CMD

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
OCR_CONFIG = "--oem 3 --psm 6"
PDF_OCR_DPI = 300

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
}


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_line(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()


def normalize_text(
    text: str,
) -> str:

    if not text:
        return ""

    lines = []
    previous_line = None

    for line in text.splitlines():

        line = normalize_line(
            line
        )

        if not line:
            continue

        key = line.lower()

        if key == previous_line:
            continue

        lines.append(
            line
        )

        previous_line = key

    return "\n".join(
        lines
    )


def detect_text_language(
    text: str,
) -> str:

    arabic_chars = len(
        re.findall(
            r"[\u0600-\u06FF]",
            text,
        )
    )

    english_chars = len(
        re.findall(
            r"[A-Za-z]",
            text,
        )
    )

    total = (
        arabic_chars
        + english_chars
    )

    if total == 0:
        return "unknown"

    arabic_ratio = (
        arabic_chars
        / total
    )

    english_ratio = (
        english_chars
        / total
    )

    if arabic_ratio >= 0.75:
        return "ar"

    if english_ratio >= 0.75:
        return "en"

    return "mixed"


def stable_uuid(
    value: str,
) -> str:

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            value,
        )
    )


def file_sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


# ============================================================
# OCR HELPERS
# ============================================================

def prepare_image_for_ocr(
    image: Image.Image,
) -> Image.Image:
    """
    Lightweight CPU-friendly preprocessing for Arabic/English OCR.
    """

    image = image.convert(
        "RGB"
    )

    grayscale = ImageOps.grayscale(
        image
    )

    grayscale = ImageOps.autocontrast(
        grayscale
    )

    # Upscale small screenshots/photos to help Tesseract.
    if grayscale.width < 1600:

        scale = min(
            2.0,
            1600 / max(
                grayscale.width,
                1,
            ),
        )

        if scale > 1.0:

            grayscale = grayscale.resize(
                (
                    int(
                        grayscale.width
                        * scale
                    ),
                    int(
                        grayscale.height
                        * scale
                    ),
                )
            )

    return grayscale


def ocr_image(
    image: Image.Image,
) -> str:

    prepared = prepare_image_for_ocr(
        image
    )

    text = pytesseract.image_to_string(
        prepared,
        lang=OCR_LANGUAGES,
        config=OCR_CONFIG,
    )

    return normalize_text(
        text
    )


def render_pdf_page(
    pdf_document,
    page_index: int,
) -> Image.Image:

    page = pdf_document[
        page_index
    ]

    scale = (
        PDF_OCR_DPI
        / 72.0
    )

    bitmap = page.render(
        scale=scale
    )

    image = bitmap.to_pil()

    try:
        bitmap.close()
    except Exception:
        pass

    try:
        page.close()
    except Exception:
        pass

    return image


# ============================================================
# EXTRACTION
# ============================================================

def extract_pdf(
    path: Path,
) -> list[dict]:
    """
    Prefer the PDF text layer.

    If a page has too little extractable text, render that page
    locally and OCR it with Tesseract (Arabic + English).
    """

    reader = PdfReader(
        str(
            path
        )
    )

    pdfium_document = None
    units = []

    try:

        for page_index, page in enumerate(
            reader.pages,
            start=1,
        ):

            text = (
                page.extract_text()
                or ""
            )

            text = normalize_text(
                text
            )

            extraction_method = (
                "text_layer"
            )

            if len(
                text
            ) < MIN_CHUNK_CHARS:

                if pdfium_document is None:

                    pdfium_document = (
                        pdfium.PdfDocument(
                            str(
                                path
                            )
                        )
                    )

                image = render_pdf_page(
                    pdf_document=pdfium_document,
                    page_index=(
                        page_index - 1
                    ),
                )

                try:

                    text = ocr_image(
                        image
                    )

                finally:

                    image.close()

                extraction_method = "ocr"

            minimum_chars = (
                MIN_OCR_CHARS
                if extraction_method == "ocr"
                else MIN_CHUNK_CHARS
            )

            if len(
                text
            ) < minimum_chars:

                continue

            units.append(
                {
                    "text":
                        text,

                    "page_number":
                        page_index,

                    "section":
                        "",

                    "extraction_method":
                        extraction_method,
                }
            )

    finally:

        if pdfium_document is not None:

            try:
                pdfium_document.close()
            except Exception:
                pass

    return units


def extract_docx(
    path: Path,
) -> list[dict]:
    """
    Extract normal DOCX paragraphs and tables.

    Physical page numbers are not reliably available in DOCX.
    """

    document = Document(
        str(
            path
        )
    )

    blocks = []

    for paragraph in document.paragraphs:

        text = normalize_line(
            paragraph.text
        )

        if text:

            blocks.append(
                text
            )

    for table in document.tables:

        for row in table.rows:

            cells = [
                normalize_line(
                    cell.text
                )
                for cell in row.cells
            ]

            cells = [
                cell
                for cell in cells
                if cell
            ]

            if cells:

                blocks.append(
                    " | ".join(
                        cells
                    )
                )

    text = normalize_text(
        "\n".join(
            blocks
        )
    )

    if len(
        text
    ) < MIN_CHUNK_CHARS:

        return []

    return [
        {
            "text":
                text,

            "page_number":
                None,

            "section":
                "",

            "extraction_method":
                "docx_text",
        }
    ]


def extract_txt(
    path: Path,
) -> list[dict]:

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    text = normalize_text(
        text
    )

    if len(
        text
    ) < MIN_CHUNK_CHARS:

        return []

    return [
        {
            "text":
                text,

            "page_number":
                None,

            "section":
                "",

            "extraction_method":
                "txt",
        }
    ]


def extract_image(
    path: Path,
) -> list[dict]:

    with Image.open(
        path
    ) as image:

        text = ocr_image(
            image
        )

    if len(
        text
    ) < MIN_OCR_CHARS:

        return []

    return [
        {
            "text":
                text,

            "page_number":
                None,

            "section":
                "",

            "extraction_method":
                "ocr",
        }
    ]


def extract_document(
    path: Path,
) -> list[dict]:

    extension = (
        path.suffix.lower()
    )

    if extension == ".pdf":

        return extract_pdf(
            path
        )

    if extension == ".docx":

        return extract_docx(
            path
        )

    if extension == ".txt":

        return extract_txt(
            path
        )

    if extension in {
        ".png",
        ".jpg",
        ".jpeg",
    }:

        return extract_image(
            path
        )

    raise ValueError(
        f"Unsupported file type: "
        f"{extension}"
    )


# ============================================================
# CHUNKING
# ============================================================

def split_long_unit(
    text: str,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
) -> list[str]:

    if len(
        text
    ) <= MAX_CHUNK_CHARS:

        return [
            text
        ]

    chunks = []

    start = 0

    text_length = len(
        text
    )

    while start < text_length:

        end = min(
            start
            + MAX_CHUNK_CHARS,
            text_length,
        )

        if end < text_length:

            window = text[
                start:end
            ]

            candidates = [
                window.rfind(
                    "\n"
                ),
                window.rfind(
                    ". "
                ),
                window.rfind(
                    "؟ "
                ),
                window.rfind(
                    "! "
                ),
                window.rfind(
                    "؛ "
                ),
                window.rfind(
                    "; "
                ),
                window.rfind(
                    " "
                ),
            ]

            split_position = max(
                candidates
            )

            if split_position > (
                MAX_CHUNK_CHARS
                * 0.55
            ):

                end = (
                    start
                    + split_position
                    + 1
                )

        chunk = normalize_text(
            text[
                start:end
            ]
        )

        if len(
            chunk
        ) >= min_chunk_chars:

            chunks.append(
                chunk
            )

        if end >= text_length:
            break

        start = max(
            end
            - OVERLAP_CHARS,
            start + 1,
        )

    return chunks


def build_chunks(
    source_units: list[dict],
) -> list[dict]:

    chunks = []

    for unit_index, unit in enumerate(
        source_units
    ):

        min_chunk_chars = (
            MIN_OCR_CHARS
            if unit.get(
                "extraction_method"
            ) == "ocr"
            else MIN_CHUNK_CHARS
        )

        text_chunks = split_long_unit(
            unit[
                "text"
            ],
            min_chunk_chars=min_chunk_chars,
        )

        for chunk_index, text in enumerate(
            text_chunks
        ):

            chunks.append(
                {
                    "text":
                        text,

                    "page_number":
                        unit.get(
                            "page_number"
                        ),

                    "section":
                        unit.get(
                            "section",
                            "",
                        ),

                    "extraction_method":
                        unit.get(
                            "extraction_method",
                            "",
                        ),

                    "source_unit_index":
                        unit_index,

                    "chunk_index":
                        chunk_index,

                    "chunk_total_in_unit":
                        len(
                            text_chunks
                        ),
                }
            )

    return chunks


# ============================================================
# QDRANT CLEANUP
# ============================================================

def delete_existing_document(
    client,
    source_name: str,
):
    """
    Replace an older upload with the same display filename.

    Telecom Egypt website chunks do not use source_name, so this
    filter does not affect the existing website corpus.
    """

    selector = FilterSelector(
        filter=Filter(
            must=[
                FieldCondition(
                    key="source_name",
                    match=MatchValue(
                        value=source_name
                    ),
                )
            ]
        )
    )

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=selector,
        wait=True,
    )


# ============================================================
# EMBEDDING TEXT
# ============================================================

def build_embedding_text(
    source_name: str,
    text: str,
) -> str:

    return (
        f"{source_name}\n"
        f"{text}"
    )


# ============================================================
# INGESTION
# ============================================================

def ingest_document(
    file_path: str | Path,
    display_name: str | None = None,
    client=None,
    embedding_model=None,
) -> dict:

    path = Path(
        file_path
    ).resolve()

    if not path.exists():

        raise FileNotFoundError(
            f"File not found: "
            f"{path}"
        )

    extension = (
        path.suffix.lower()
    )

    if extension not in SUPPORTED_EXTENSIONS:

        raise ValueError(
            "Supported file types are: "
            "PDF, DOCX, TXT, PNG, JPG and JPEG."
        )

    source_name = (
        display_name
        or path.name
    )

    print(
        "=" * 80
    )

    print(
        "USER DOCUMENT INGESTION"
    )

    print(
        "=" * 80
    )

    print(
        "File:",
        source_name,
    )

    print(
        "Type:",
        extension,
    )

    print()

    print(
        "Extracting text..."
    )

    source_units = (
        extract_document(
            path
        )
    )

    if not source_units:

        raise RuntimeError(
            "No useful text could be extracted "
            "from this document or image."
        )

    print(
        "Extracted units:",
        len(
            source_units
        ),
    )

    extraction_methods = sorted(
        {
            unit.get(
                "extraction_method",
                "unknown",
            )
            for unit in source_units
        }
    )

    print(
        "Extraction:",
        ", ".join(
            extraction_methods
        ),
    )

    for unit_number, unit in enumerate(
        source_units,
        start=1,
    ):

        if unit.get(
            "extraction_method"
        ) == "ocr":

            print(
                f"OCR unit {unit_number}: "
                f"{len(unit.get('text', ''))} chars"
            )

            # Extracted document content stays in the indexed payload, not stdout.

    chunks = build_chunks(
        source_units
    )

    if not chunks:

        raise RuntimeError(
            "No useful chunks were created."
        )

    print(
        "Chunks:",
        len(
            chunks
        ),
    )

    file_hash = file_sha256(
        path
    )

    document_id = (
        "user_"
        + file_hash[:24]
    )

    for global_index, chunk in enumerate(
        chunks
    ):

        chunk[
            "global_chunk_index"
        ] = global_index

        chunk[
            "content_language"
        ] = detect_text_language(
            chunk[
                "text"
            ]
        )

    if client is None:

        client = (
            get_qdrant_client()
        )

    if embedding_model is None:

        print()

        print(
            "Loading embedding model..."
        )

        embedding_model = (
            EmbeddingModel()
        )

        print(
            "Embedding model ready."
        )

    create_collection(client)

    points_before = (
        get_point_count(
            client
        )
    )

    print()

    print(
        "Qdrant points before:",
        points_before,
    )

    print(
        "Removing older version of "
        "the same filename, if present..."
    )

    delete_existing_document(
        client=client,
        source_name=source_name,
    )

    indexed_count = 0

    for start in range(
        0,
        len(
            chunks
        ),
        BATCH_SIZE,
    ):

        end = min(
            start
            + BATCH_SIZE,
            len(
                chunks
            ),
        )

        batch = chunks[
            start:end
        ]

        embedding_texts = [
            build_embedding_text(
                source_name=source_name,
                text=chunk[
                    "text"
                ],
            )
            for chunk in batch
        ]

        vectors = (
            embedding_model
            .encode_documents(
                embedding_texts
            )
        )

        points = []

        for chunk, vector in zip(
            batch,
            vectors,
        ):

            chunk_identity = (
                f"{document_id}|"
                f"{chunk['page_number']}|"
                f"{chunk['source_unit_index']}|"
                f"{chunk['chunk_index']}|"
                f"{chunk['text']}"
            )

            point_id = stable_uuid(
                chunk_identity
            )

            payload = {
                "source":
                    "user_document",

                "source_type":
                    "user_document",

                "source_kind":
                    "uploaded_document",

                "document_id":
                    document_id,

                "source_name":
                    source_name,

                "file_type":
                    extension.lstrip(
                        "."
                    ),

                "file_hash":
                    file_hash,

                "title":
                    source_name,

                "section_label":
                    chunk.get(
                        "section",
                        "",
                    ),

                "page_number":
                    chunk.get(
                        "page_number"
                    ),

                "chunk_index":
                    chunk[
                        "global_chunk_index"
                    ],

                "content_language":
                    chunk[
                        "content_language"
                    ],

                "extraction_method":
                    chunk.get(
                        "extraction_method",
                        "",
                    ),

                "text":
                    chunk[
                        "text"
                    ],

                "char_count":
                    len(
                        chunk[
                            "text"
                        ]
                    ),

                "url":
                    "",

                "final_url":
                    "",

                "citation_url":
                    "",
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=(
                        vector.tolist()
                    ),
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        indexed_count += len(
            points
        )

        print(
            f"Indexed "
            f"{indexed_count}/"
            f"{len(chunks)}"
        )

    points_after = (
        get_point_count(
            client
        )
    )

    print()

    print(
        "=" * 80
    )

    print(
        "DOCUMENT INGESTION COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "Document ID:",
        document_id,
    )

    print(
        "File:",
        source_name,
    )

    print(
        "Chunks indexed:",
        indexed_count,
    )

    print(
        "Qdrant points after:",
        points_after,
    )

    return {
        "document_id":
            document_id,

        "source_name":
            source_name,

        "file_type":
            extension.lstrip(
                "."
            ),

        "source_units":
            len(
                source_units
            ),

        "chunks_indexed":
            indexed_count,

        "qdrant_points":
            points_after,

        "extraction_methods":
            extraction_methods,
    }


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Index a PDF, DOCX, TXT, PNG, JPG or JPEG file "
            "into the TE RAG knowledge base."
        )
    )

    parser.add_argument(
        "file",
        help=(
            "Path to the document or image."
        ),
    )

    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Optional display filename used "
            "for citations."
        ),
    )

    args = parser.parse_args()

    ingest_document(
        file_path=args.file,
        display_name=args.name,
    )


if __name__ == "__main__":
    main()
