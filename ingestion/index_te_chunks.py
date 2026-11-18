import argparse
import json
import uuid

from pathlib import Path

from qdrant_client.models import PointStruct

from retrieval.embeddings import EmbeddingModel

from retrieval.vector_store import (
    COLLECTION_NAME,
    VECTOR_SIZE,
    get_qdrant_client,
    reset_collection,
    create_collection,
    get_point_count,
)


# ============================================================
# PATHS
# ============================================================

INGESTION_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    INGESTION_DIR
    / "data"
)

INPUT_PATH = INGESTION_DIR.parent / "samples" / "synthetic_chunks.json"

REPORT_PATH = (
    DATA_DIR
    / "processed"
    / "te_index_report.json"
)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 32

EMBEDDING_MODEL_NAME = (
    "intfloat/multilingual-e5-small"
)


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(
    path: Path,
):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def save_json(
    path: Path,
    data,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temp_path.replace(
        path
    )


# ============================================================
# EMBEDDING TEXT
# ============================================================

def build_embedding_text(
    chunk: dict,
) -> str:
    """
    Add the page title to the chunk before embedding.

    Example:

        WE Space - Telecom Egypt
        <actual chunk text>

    This helps retrieval understand which product/service
    the chunk belongs to.

    The original source text is still stored separately
    in the Qdrant payload.
    """

    parts = []

    title = (
        chunk.get(
            "title",
            "",
        )
        or ""
    ).strip()

    text = (
        chunk.get(
            "text",
            "",
        )
        or ""
    ).strip()

    if title:

        parts.append(
            title
        )

    if text:

        parts.append(
            text
        )

    return "\n".join(
        parts
    )


# ============================================================
# POINT ID
# ============================================================

def make_point_id(
    chunk_id: str,
) -> str:
    """
    Qdrant supports UUID point IDs.

    Convert our stable chunk ID to a deterministic UUID.

    This means the same chunk always receives the same
    Qdrant point ID.
    """

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            chunk_id,
        )
    )


# ============================================================
# QDRANT PAYLOAD
# ============================================================

def build_payload(
    chunk: dict,
) -> dict:

    return {
        "chunk_id":
            chunk.get(
                "chunk_id",
                "",
            ),

        "document_id":
            chunk.get(
                "document_id",
                "",
            ),

        "chunk_index":
            chunk.get(
                "chunk_index",
                0,
            ),

        "chunk_total":
            chunk.get(
                "chunk_total",
                1,
            ),

        "source":
            chunk.get(
                "source",
                "",
            ),

        "source_kind":
            chunk.get(
                "source_kind",
                "",
            ),

        "title":
            chunk.get(
                "title",
                "",
            ),

        "section_label":
            chunk.get(
                "section_label",
                "",
            ),

        "url":
            chunk.get(
                "url",
                "",
            ),

        "final_url":
            chunk.get(
                "final_url",
                "",
            ),

        "citation_url":
            (
                chunk.get(
                    "final_url"
                )
                or chunk.get(
                    "url",
                    "",
                )
            ),

        "content_language":
            chunk.get(
                "content_language",
                "unknown",
            ),

        "route_language":
            chunk.get(
                "route_language",
                "default",
            ),

        "final_route_language":
            chunk.get(
                "final_route_language",
                "unknown",
            ),

        "page_type":
            chunk.get(
                "page_type",
                "",
            ),

        "target_id":
            chunk.get(
                "target_id",
                "",
            ),

        "extraction_method":
            chunk.get(
                "extraction_method",
                "",
            ),

        "click_method":
            chunk.get(
                "click_method",
                "",
            ),

        "page_status":
            chunk.get(
                "page_status",
                "",
            ),

        "page_has_control_errors":
            chunk.get(
                "page_has_control_errors",
                False,
            ),

        "alternate_sources":
            chunk.get(
                "alternate_sources",
                [],
            ),

        "text":
            chunk.get(
                "text",
                "",
            ),

        "char_count":
            chunk.get(
                "char_count",
                0,
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Index prepared JSON chunks into the configured collection.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild only the configured collection.")
    args = parser.parse_args()

    print(
        "=" * 80
    )

    print(
        "TE QDRANT INDEXING"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    chunks = load_json(
        args.input
    )

    print(
        "Input chunks:",
        len(
            chunks
        ),
    )

    if not chunks:

        print(
            "No chunks found."
        )

        return

    # --------------------------------------------------------
    # Connect to Qdrant
    # --------------------------------------------------------

    print()

    print(
        "Connecting to Qdrant..."
    )

    client = (
        get_qdrant_client()
    )

    print(
        "Connected."
    )

    # --------------------------------------------------------
    # Reset collection
    # --------------------------------------------------------

    print()

    print(
        "Preparing collection..."
    )

    if args.reset:
        reset_collection(client)
    else:
        create_collection(client)

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Batch indexing
    # --------------------------------------------------------

    indexed_count = 0

    print()

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
                chunk
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

            points.append(
                PointStruct(
                    id=make_point_id(
                        chunk[
                            "chunk_id"
                        ]
                    ),

                    vector=(
                        vector.tolist()
                    ),

                    payload=(
                        build_payload(
                            chunk
                        )
                    ),
                )
            )

        client.upsert(
            collection_name=(
                COLLECTION_NAME
            ),
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

    # --------------------------------------------------------
    # Verify Qdrant
    # --------------------------------------------------------

    qdrant_points = (
        get_point_count(
            client
        )
    )

    report = {
        "collection":
            COLLECTION_NAME,

        "input_chunks":
            len(
                chunks
            ),

        "indexed_chunks":
            indexed_count,

        "qdrant_points":
            qdrant_points,

        "vector_size":
            VECTOR_SIZE,

        "distance":
            "cosine",

        "embedding_model":
            EMBEDDING_MODEL_NAME,

        "batch_size":
            BATCH_SIZE,
    }

    save_json(
        REPORT_PATH,
        report,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        "INDEXING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "Indexed chunks:",
        indexed_count,
    )

    print(
        "Qdrant points:",
        qdrant_points,
    )

    print(
        "Collection:",
        COLLECTION_NAME,
    )

    print(
        "Vector dimensions:",
        VECTOR_SIZE,
    )

    print(
        "Embedding model:",
        EMBEDDING_MODEL_NAME,
    )

    print(
        "Saved report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()