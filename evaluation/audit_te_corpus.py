import json
from collections import Counter
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_DIR
    / "ingestion"
    / "data"
)

DOCUMENTS_PATH = (
    DATA_DIR
    / "processed"
    / "te_documents.json"
)

CHUNKS_PATH = (
    DATA_DIR
    / "processed"
    / "te_chunks.json"
)


# ============================================================
# CONFIG
# ============================================================

MIN_RECOMMENDED_CHUNK_CHARS = 80
MAX_EXPECTED_CHUNK_CHARS = 1400


SUSPICIOUS_PHRASES = [
    "back to top",
    "الرجوع الي الأعلي",
    "الرجوع إلى الأعلى",
]


# ============================================================
# LOAD
# ============================================================

def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# MAIN
# ============================================================

def main():

    documents = load_json(
        DOCUMENTS_PATH
    )

    chunks = load_json(
        CHUNKS_PATH
    )

    print(
        "=" * 80
    )

    print(
        "TE CORPUS AUDIT"
    )

    print(
        "=" * 80
    )

    print()

    # --------------------------------------------------------
    # BASIC COUNTS
    # --------------------------------------------------------

    print(
        "Documents:",
        len(documents),
    )

    print(
        "Chunks:",
        len(chunks),
    )

    print()

    # --------------------------------------------------------
    # DOCUMENT DISTRIBUTION
    # --------------------------------------------------------

    doc_languages = Counter(
        document.get(
            "content_language",
            "unknown",
        )
        for document in documents
    )

    doc_kinds = Counter(
        document.get(
            "source_kind",
            "unknown",
        )
        for document in documents
    )

    print(
        "Document languages:"
    )

    for key, value in sorted(
        doc_languages.items()
    ):

        print(
            f"   {key}: {value}"
        )

    print()

    print(
        "Document source kinds:"
    )

    for key, value in sorted(
        doc_kinds.items()
    ):

        print(
            f"   {key}: {value}"
        )

    # --------------------------------------------------------
    # CHUNK DISTRIBUTION
    # --------------------------------------------------------

    chunk_languages = Counter(
        chunk.get(
            "content_language",
            "unknown",
        )
        for chunk in chunks
    )

    chunk_kinds = Counter(
        chunk.get(
            "source_kind",
            "unknown",
        )
        for chunk in chunks
    )

    print()

    print(
        "Chunk languages:"
    )

    for key, value in sorted(
        chunk_languages.items()
    ):

        print(
            f"   {key}: {value}"
        )

    print()

    print(
        "Chunk source kinds:"
    )

    for key, value in sorted(
        chunk_kinds.items()
    ):

        print(
            f"   {key}: {value}"
        )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    document_hashes = Counter(
        document.get(
            "text_hash",
            "",
        )
        for document in documents
    )

    duplicate_documents = [
        hash_value
        for hash_value, count
        in document_hashes.items()
        if hash_value and count > 1
    ]

    chunk_hashes = Counter(
        chunk.get(
            "text_hash",
            "",
        )
        for chunk in chunks
    )

    duplicate_chunks = [
        hash_value
        for hash_value, count
        in chunk_hashes.items()
        if hash_value and count > 1
    ]

    print()

    print(
        "Exact duplicate document hashes:",
        len(duplicate_documents),
    )

    print(
        "Exact duplicate chunk hashes:",
        len(duplicate_chunks),
    )

    # --------------------------------------------------------
    # SHORT / LONG CHUNKS
    # --------------------------------------------------------

    short_chunks = [
        chunk
        for chunk in chunks
        if chunk.get(
            "char_count",
            0,
        )
        < MIN_RECOMMENDED_CHUNK_CHARS
    ]

    long_chunks = [
        chunk
        for chunk in chunks
        if chunk.get(
            "char_count",
            0,
        )
        > MAX_EXPECTED_CHUNK_CHARS
    ]

    print()

    print(
        f"Chunks < "
        f"{MIN_RECOMMENDED_CHUNK_CHARS} chars:",
        len(short_chunks),
    )

    print(
        f"Chunks > "
        f"{MAX_EXPECTED_CHUNK_CHARS} chars:",
        len(long_chunks),
    )

    # --------------------------------------------------------
    # MISSING CITATION METADATA
    # --------------------------------------------------------

    missing_title = [
        chunk
        for chunk in chunks
        if not (
            chunk.get(
                "title",
                "",
            )
            or ""
        ).strip()
    ]

    missing_url = [
        chunk
        for chunk in chunks
        if not (
            chunk.get(
                "final_url",
                "",
            )
            or chunk.get(
                "url",
                "",
            )
            or ""
        ).strip()
    ]

    missing_text = [
        chunk
        for chunk in chunks
        if not (
            chunk.get(
                "text",
                "",
            )
            or ""
        ).strip()
    ]

    print()

    print(
        "Chunks missing title:",
        len(missing_title),
    )

    print(
        "Chunks missing URL:",
        len(missing_url),
    )

    print(
        "Chunks missing text:",
        len(missing_text),
    )

    # --------------------------------------------------------
    # PARTIAL SOURCES
    # --------------------------------------------------------

    partial_chunks = [
        chunk
        for chunk in chunks
        if (
            chunk.get(
                "page_status"
            )
            == "partial"
            or chunk.get(
                "page_has_control_errors",
                False,
            )
        )
    ]

    print()

    print(
        "Chunks from partial/error-control pages:",
        len(partial_chunks),
    )

    # --------------------------------------------------------
    # SUSPICIOUS BOILERPLATE
    # --------------------------------------------------------

    suspicious_chunks = []

    for chunk in chunks:

        text = (
            chunk.get(
                "text",
                "",
            )
            or ""
        ).lower()

        if any(
            phrase.lower() in text
            for phrase in SUSPICIOUS_PHRASES
        ):

            suspicious_chunks.append(
                chunk
            )

    print()

    print(
        "Chunks containing known UI boilerplate:",
        len(suspicious_chunks),
    )

    # --------------------------------------------------------
    # CHUNK SIZE STATS
    # --------------------------------------------------------

    char_counts = [
        chunk.get(
            "char_count",
            0,
        )
        for chunk in chunks
    ]

    if char_counts:

        print()

        print(
            "Chunk size:"
        )

        print(
            "   Minimum:",
            min(char_counts),
        )

        print(
            "   Maximum:",
            max(char_counts),
        )

        print(
            "   Average:",
            round(
                sum(char_counts)
                / len(char_counts),
                2,
            ),
        )

    # --------------------------------------------------------
    # VERDICT
    # --------------------------------------------------------

    blocking_issues = (
        len(duplicate_documents)
        + len(duplicate_chunks)
        + len(long_chunks)
        + len(missing_url)
        + len(missing_text)
    )

    print()

    print(
        "=" * 80
    )

    if blocking_issues == 0:

        print(
            "AUDIT RESULT: PASS"
        )

    else:

        print(
            "AUDIT RESULT: REVIEW REQUIRED"
        )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()